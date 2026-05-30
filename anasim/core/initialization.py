from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .state import AirwayType
from .utils import clamp
from . import projection as projection_core
from . import runtime as runtime_core

if TYPE_CHECKING:
    from .engine import SimulationEngine


STARTUP_BIS_BAND_SCALE = 8.0
STARTUP_TOL_WEIGHT = 4.0
STARTUP_REMI_EXCESS_WEIGHT = 0.5


@dataclass(frozen=True, slots=True)
class StartupProfile:
    name: str
    bis_target: float
    tol_target: float
    primary_hypnotic: str
    history_minutes: float = 30.0
    settle_seconds: float = 60.0
    settle_dt_seconds: float = 1.0
    primary_bounds: tuple[float, float] = (0.0, 1.0)
    remi_bounds: tuple[float, float] = (0.0, 1.0)
    remi_soft_cap: float | None = None
    maintenance_dial_multiplier: float = 1.0
    fgf_o2_l_min: float = 2.0


@dataclass(frozen=True, slots=True)
class StartupTargets:
    prop_ce: float = 0.0
    remi_ce: float = 0.0
    mac: float = 0.0
    volatile_target_pct: float = 0.0


TIVA_PROFILE = StartupProfile(
    name="steady_state_tiva",
    bis_target=55.0,
    tol_target=0.6,
    primary_hypnotic="propofol",
    primary_bounds=(2.8, 4.2),
    remi_bounds=(1.0, 2.0),
    remi_soft_cap=2.0,
)

BALANCED_PROFILE = StartupProfile(
    name="steady_state_balanced",
    bis_target=45.0,
    tol_target=0.9,
    primary_hypnotic="volatile",
    primary_bounds=(0.8, 1.2),
    remi_bounds=(0.0, 3.0),
    maintenance_dial_multiplier=1.55,
    fgf_o2_l_min=6.0,
)


def initialize_engine_state(engine: "SimulationEngine") -> None:
    """Initialize the engine's startup state without placeholder defaults."""
    if engine.config.mode == "steady_state":
        _initialize_steady_state(engine)
    else:
        _initialize_awake(engine)


def _initialize_awake(engine: "SimulationEngine") -> None:
    engine.state.airway_mode = AirwayType.NONE
    engine.resp.state.apnea = False
    engine.set_vent_settings(rr=0.0, vt=0.0, peep=0.0, ie="1:2", mode="VCV")
    engine.set_vaporizer(engine.active_agent, 0.0)


def _initialize_steady_state(engine: "SimulationEngine") -> None:
    profile = _select_profile(engine)
    targets = _solve_startup_targets(engine, profile)

    engine.state.airway_mode = AirwayType.ETT
    engine.resp.state.apnea = True
    _configure_controlled_ventilation(engine, targets)
    _seed_steady_state_subsystems(engine, profile, targets)
    projection_core.sync_state_from_models(engine)
    _run_hidden_settle(engine, profile)
    _attach_startup_controllers(engine, targets)
    engine.state.time = 0.0
    engine._next_nibp_time = 0.0


def _select_profile(engine: "SimulationEngine") -> StartupProfile:
    if "balanced" in str(engine.config.maint_type).lower():
        return BALANCED_PROFILE
    return TIVA_PROFILE


def _solve_startup_targets(engine: "SimulationEngine", profile: StartupProfile) -> StartupTargets:
    bis_model = engine.bis
    tol_model = engine.tol_pd

    def objective(x):
        primary_load = x[0]
        remi_ce = x[1]
        if profile.primary_hypnotic == "volatile":
            prop_ce = 0.0
            mac = primary_load
        else:
            prop_ce = primary_load
            mac = 0.0

        bis_val = bis_model.compute_bis(prop_ce, remi_ce, u_volatile=mac)
        tol_val = tol_model.compute_probability(prop_ce, remi_ce, mac=mac)
        tol_deficit = max(0.0, profile.tol_target - tol_val)
        remi_excess = 0.0
        if profile.remi_soft_cap is not None:
            remi_excess = max(0.0, remi_ce - profile.remi_soft_cap)
        return (
            ((bis_val - profile.bis_target) ** 2) / (STARTUP_BIS_BAND_SCALE ** 2)
            + STARTUP_TOL_WEIGHT * (tol_deficit ** 2)
            + STARTUP_REMI_EXCESS_WEIGHT * (remi_excess ** 2)
        )

    x0 = (sum(profile.primary_bounds) / 2.0, sum(profile.remi_bounds) / 2.0)
    result = minimize(
        objective,
        x0,
        bounds=(profile.primary_bounds, profile.remi_bounds),
        method="L-BFGS-B",
    )
    primary_load, remi_ce = result.x
    if profile.primary_hypnotic == "volatile":
        prop_ce = 0.0
        mac = float(primary_load)
    else:
        prop_ce = float(primary_load)
        mac = 0.0

    volatile_target_pct = 0.0
    if profile.primary_hypnotic == "volatile" and engine.pk_sevo:
        volatile_target_pct = float(engine.pk_sevo.mac_age) * mac * profile.maintenance_dial_multiplier
    return StartupTargets(
        prop_ce=prop_ce,
        remi_ce=float(remi_ce),
        mac=mac,
        volatile_target_pct=volatile_target_pct,
    )


def _configure_controlled_ventilation(engine: "SimulationEngine", targets: StartupTargets) -> None:
    baseline_rr = max(1.0, getattr(engine.patient, "baseline_rr", 12.0))
    baseline_vt_l = max(0.1, getattr(engine.patient, "baseline_vt", 500.0) / 1000.0)
    baseline_mv = baseline_rr * baseline_vt_l
    _depth_index, metabolic_factor = runtime_core.compute_depth_metabolic_context(
        engine,
        getattr(engine.patient, "baseline_temp", 37.0),
        targets.prop_ce,
        targets.mac,
        shiver_level=0.0,
    )
    target_mv = baseline_mv * metabolic_factor
    vent_vt = target_mv / baseline_rr if baseline_rr > 0 else baseline_vt_l
    vent_vt = clamp(vent_vt, 0.25, 0.8)
    engine.set_vent_settings(rr=baseline_rr, vt=vent_vt, peep=5.0, ie="1:2", mode="VCV")


def _seed_steady_state_subsystems(engine: "SimulationEngine", profile: StartupProfile, targets: StartupTargets) -> None:
    engine.set_vaporizer(engine.active_agent, 0.0)
    engine.set_fgf(profile.fgf_o2_l_min, 0.0, 0.0)
    engine.propofol_rate_mg_sec = 0.0
    engine.remi_rate_ug_sec = 0.0
    engine.nore_rate_ug_sec = 0.0

    if targets.prop_ce > 0.0:
        prop_rate_min = _seed_linear_history(engine.pk_prop, targets.prop_ce, profile.history_minutes, target_compartment="effect_site")
        engine.propofol_rate_mg_sec = prop_rate_min / 60.0
    if targets.remi_ce > 0.0:
        remi_rate_min = _seed_linear_history(engine.pk_remi, targets.remi_ce, profile.history_minutes, target_compartment="effect_site")
        engine.remi_rate_ug_sec = remi_rate_min / 60.0
    if profile.primary_hypnotic == "volatile" and engine.pk_sevo:
        _seed_volatile_history(engine, targets.mac, profile.history_minutes)
        engine.set_vaporizer("Sevoflurane", targets.volatile_target_pct)

    # Seed the hemodynamic stars near the solved maintenance point so the
    # short hidden settle only handles monitor/circuit transients.
    engine.hemo.state = engine.hemo.calculate_steady_state(
        getattr(engine.pk_prop.state, "c1", 0.0),
        getattr(engine.pk_remi.state, "c1", 0.0),
        getattr(engine.pk_nore.state, "ce", 0.0),
        mac_sevo=targets.mac,
    )


def _seed_linear_history(pk_model, target: float, duration_min: float, target_compartment: str) -> float:
    """Seed a linear PK model from a finite constant-input history using its state-space matrices."""
    if target <= 0.0:
        if hasattr(pk_model, "reset"):
            pk_model.reset()
        return 0.0

    A, B = pk_model.get_ss_matrices()
    A_aug, B_aug = _augment_effect_site_state(pk_model, A, B)

    target_id = 0 if target_compartment == "plasma" else A_aug.shape[0] - 1
    exp_term = expm(A_aug * duration_min)
    finite_horizon_gain = (np.eye(A_aug.shape[0]) - exp_term) @ (-np.linalg.solve(A_aug, B_aug))
    component_gain = float(finite_horizon_gain[target_id, 0])
    if component_gain <= 0.0:
        raise ValueError(f"Invalid steady-state gain for {pk_model.__class__.__name__}")

    input_rate_min = target / component_gain
    x_t = finite_horizon_gain * input_rate_min
    _set_linear_model_state(pk_model, x_t[:, 0])
    return float(input_rate_min)


def _augment_effect_site_state(pk_model, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Augment PK state-space matrices with an effect-site state when get_ss_matrices omits it."""
    if not hasattr(pk_model.state, "ce"):
        return A, B
    if A.shape[0] >= 4:
        return A, B
    if getattr(pk_model.state, "c3", None) not in (None, 0.0) and A.shape[0] == 4:
        return A, B
    if A.shape[0] == 3 and hasattr(pk_model.state, "c3"):
        return A, B

    ke0 = getattr(pk_model, "ke0", 0.0)
    if ke0 <= 0.0:
        return A, B

    n = A.shape[0]
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[n, 0] = ke0
    A_aug[n, n] = -ke0
    B_aug = np.zeros((n + 1, 1))
    B_aug[:n, :] = B
    return A_aug, B_aug


def _set_linear_model_state(pk_model, values: np.ndarray) -> None:
    state = pk_model.state
    if len(values) >= 1:
        state.c1 = max(0.0, float(values[0]))
    if hasattr(state, "c2") and len(values) >= 2:
        state.c2 = max(0.0, float(values[1]))
    if hasattr(state, "c3"):
        if len(values) >= 4:
            state.c3 = max(0.0, float(values[2]))
            state.ce = max(0.0, float(values[3]))
        elif len(values) >= 3:
            state.c3 = max(0.0, float(values[2]))
    if hasattr(state, "ce") and len(values) >= 2:
        state.ce = max(0.0, float(values[-1]))


def _seed_volatile_history(engine: "SimulationEngine", target_mac: float, duration_min: float) -> None:
    """Seed volatile tissue partial pressures from a 30-minute managed maintenance history."""
    target_frac = (engine.pk_sevo.mac_age * target_mac) / 100.0
    pk = engine.pk_sevo
    state = pk.state
    p_art = max(0.0, float(target_frac))
    q_co = max(getattr(engine.hemo, "base_co_l_min", 5.0), 0.1)
    q_vrg = q_co * pk.f_vrg_frac
    q_mus = q_co * pk.f_mus_frac
    q_fat = q_co * pk.f_fat_frac
    k_vrg = (q_vrg / pk.v_vrg) / pk.lambda_t_b_vrg
    k_mus = (q_mus / pk.v_mus) / pk.lambda_t_b_mus
    k_fat = (q_fat / pk.v_fat) / pk.lambda_t_b_fat

    state.p_alv = p_art
    state.p_art = p_art
    state.p_vrg = p_art * (1.0 - np.exp(-k_vrg * duration_min))
    state.p_mus = p_art * (1.0 - np.exp(-k_mus * duration_min))
    state.p_fat = p_art * (1.0 - np.exp(-k_fat * duration_min))
    state.p_ven = (
        q_vrg * state.p_vrg + q_mus * state.p_mus + q_fat * state.p_fat
    ) / max(q_co, 1e-6)
    corrected_mac_age = max(pk.mac_age, 1e-6)
    state.mac = (state.p_vrg * 100.0) / corrected_mac_age


def _run_hidden_settle(engine: "SimulationEngine", profile: StartupProfile) -> None:
    """Run a short settle for circuit and physiology transients without visible side effects."""
    saved_vol_clearance = getattr(engine.hemo, "vol_clearance", None)
    saved_maintenance_rate = engine.maintenance_fluid_rate_ml_min
    saved_time = engine.state.time
    engine.hemo.vol_clearance = 0.0
    engine.maintenance_fluid_rate_ml_min = 0.0

    steps = max(1, int(profile.settle_seconds / profile.settle_dt_seconds))
    for _ in range(steps):
        depth_index, metabolic_factor = runtime_core.compute_depth_metabolic_context(
            engine,
            getattr(engine.patient, "baseline_temp", 37.0),
            engine.state.propofol_ce,
            engine.state.mac,
            shiver_level=0.0,
        )
        engine._depth_index = depth_index
        engine._metabolic_factor = metabolic_factor
        fi_sevo, fi_n2o = runtime_core.step_machine(engine, profile.settle_dt_seconds)
        runtime_core.step_pk(engine, profile.settle_dt_seconds, fi_sevo, fi_n2o, engine.state.co)
        physiology = runtime_core.step_physiology(engine, profile.settle_dt_seconds, runtime_core.zero_disturbance())
        projection_core.project_runtime_physiology(engine, physiology)
        engine.state.time += profile.settle_dt_seconds

    engine.hemo.vol_clearance = saved_vol_clearance
    engine.maintenance_fluid_rate_ml_min = saved_maintenance_rate
    engine.state.temp_c = getattr(engine.patient, "baseline_temp", 37.0)
    engine._cached_temp_metabolic = engine.state.temp_c
    engine._cached_temp_metabolic_factor = 1.0
    engine._metabolic_factor = 1.0
    engine._shiver_level = 0.0
    engine._do2_ratio = max(0.0, getattr(engine.state, "oxygen_delivery_ratio", 1.0))
    engine.time_brady = 0.0
    engine.time_hypotension = 0.0
    engine.time_tachy = 0.0
    engine.state.time = saved_time


def _attach_startup_controllers(engine: "SimulationEngine", targets: StartupTargets) -> None:
    """Attach TCI controllers after seeding so they inherit and hold the post-settle state."""
    projection_core.sync_pk_state(engine)
    if targets.prop_ce > 0.0:
        engine.enable_tci("propofol", engine.pk_prop.state.ce, mode="effect_site")
        engine.propofol_rate_mg_sec = 0.0
    if targets.remi_ce > 0.0:
        engine.enable_tci("remi", engine.pk_remi.state.ce, mode="effect_site")
        engine.remi_rate_ug_sec = 0.0
