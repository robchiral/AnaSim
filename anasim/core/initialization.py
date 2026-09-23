from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq, minimize

from . import projection as projection_core
from . import runtime as runtime_core
from .state import AirwayType
from .utils import clamp

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
    fgf_o2_l_min: float = 2.0
    minimum_map: float = 65.0


@dataclass(frozen=True, slots=True)
class StartupTargets:
    prop_ce: float = 0.0
    remi_ce: float = 0.0
    mac: float = 0.0
    nore_ce: float = 0.0


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
    targets = _seed_steady_state_subsystems(engine, profile, targets)
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

        bis_val = bis_model.compute_bis(prop_ce, remi_ce, mac_sevo=mac)
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
    primary_load, remi_ce = map(float, result.x)
    if profile.primary_hypnotic == "volatile":
        return StartupTargets(remi_ce=remi_ce, mac=primary_load)
    return StartupTargets(prop_ce=primary_load, remi_ce=remi_ce)


def _configure_controlled_ventilation(engine: "SimulationEngine", targets: StartupTargets) -> None:
    baseline_rr = max(1.0, engine.patient.baseline_rr)
    baseline_vt_l = max(0.1, engine.patient.baseline_vt / 1000.0)
    baseline_mv = baseline_rr * baseline_vt_l
    _depth_index, metabolic_factor = runtime_core.compute_depth_metabolic_context(
        engine,
        engine.patient.baseline_temp,
        targets.prop_ce,
        targets.mac,
        shiver_level=0.0,
    )
    target_mv = baseline_mv * metabolic_factor
    vent_vt = target_mv / baseline_rr if baseline_rr > 0 else baseline_vt_l
    vent_vt = clamp(vent_vt, 0.25, 0.8)
    engine.set_vent_settings(rr=baseline_rr, vt=vent_vt, peep=5.0, ie="1:2", mode="VCV")


def _seed_steady_state_subsystems(
    engine: "SimulationEngine",
    profile: StartupProfile,
    targets: StartupTargets,
) -> StartupTargets:
    engine.set_vaporizer(engine.active_agent, 0.0)
    engine.set_fgf(profile.fgf_o2_l_min, 0.0, 0.0)
    engine.propofol_rate_mg_sec = 0.0
    engine.remi_rate_ug_sec = 0.0
    engine.nore_rate_ug_sec = 0.0

    if targets.prop_ce > 0.0:
        engine.propofol_rate_mg_sec = _seed_linear_history(engine.pk_prop, targets.prop_ce, profile.history_minutes) / 60.0
    if targets.remi_ce > 0.0:
        engine.remi_rate_ug_sec = _seed_linear_history(engine.pk_remi, targets.remi_ce, profile.history_minutes) / 60.0
    fi_agent = 0.0
    if profile.primary_hypnotic == "volatile":
        fi_agent = _seed_volatile_history(engine, targets.mac, profile.history_minutes)
    engine.circuit.equilibrate(_oxygen_uptake_l_min(engine, targets), fi_agent)
    engine.resp.equilibrate_oxygen(engine.circuit.composition.fio2)

    prop_cp = engine.pk_prop.state.c1
    remi_cp = engine.pk_remi.state.c1
    nore_target = _solve_visible_pressor_support(
        engine,
        prop_cp=prop_cp,
        remi_cp=remi_cp,
        mac=targets.mac,
        minimum_map=profile.minimum_map,
    )
    if nore_target > 0.0:
        engine.nore_rate_ug_sec = _seed_linear_history(engine.pk_nore, nore_target, profile.history_minutes) / 60.0
        targets = replace(targets, nore_ce=nore_target)

    # Seed hemodynamics near the managed point so the short hidden settle only
    # handles monitor and circuit transients. Pressor support remains visible
    # in the public drug state and attached controller.
    engine.hemo.state = engine.hemo.calculate_steady_state(
        prop_cp,
        remi_cp,
        engine.pk_nore.state.ce,
        mac_sevo=targets.mac,
    )
    return targets


def _solve_visible_pressor_support(
    engine: "SimulationEngine",
    *,
    prop_cp: float,
    remi_cp: float,
    mac: float,
    minimum_map: float,
) -> float:
    """Find the least norepinephrine concentration needed for a managed MAP floor."""
    def map_at(nore_ce: float) -> float:
        return engine.hemo.calculate_steady_state(
            prop_cp,
            remi_cp,
            nore_ce,
            mac_sevo=mac,
        ).map

    if map_at(0.0) >= minimum_map:
        return 0.0

    upper = 30.0
    if map_at(upper) < minimum_map:
        raise ValueError("Unable to initialize maintenance above the MAP safety floor")
    return float(brentq(lambda value: map_at(value) - minimum_map, 0.0, upper))


def _seed_linear_history(pk_model, target_ce: float, duration_min: float) -> float:
    """Seed the state after a constant infusion that reaches target Ce; return the rate per minute."""
    A, B = pk_model.get_ss_matrices()
    history_gain = (np.eye(A.shape[0]) - expm(A * duration_min)) @ np.linalg.solve(-A, B)
    rate_per_min = target_ce / float(history_gain[-1, 0])
    pk_model.set_state_vector(history_gain[:, 0] * rate_per_min)
    return rate_per_min


def _oxygen_uptake_l_min(engine: "SimulationEngine", targets: StartupTargets) -> float:
    _, metabolic_factor = runtime_core.compute_depth_metabolic_context(
        engine, engine.patient.baseline_temp, targets.prop_ce, targets.mac
    )
    return engine.resp.vco2 * metabolic_factor / engine.resp.rq / 1000.0


def _seed_volatile_history(engine: "SimulationEngine", target_mac: float, duration_min: float) -> float:
    """Seed tissue partial pressures after a managed maintenance history.

    Sets the vaporizer to the dial that holds alveolar partial pressure constant
    at the seeded uptake, and returns the inspired fraction.
    """
    pk = engine.pk_sevo
    state = pk.state
    p_art = pk.mac_age * target_mac / 100.0
    q_co = max(engine.hemo.base_co_l_min, 0.1)
    flows = {
        "p_vrg": (q_co * pk.f_vrg_frac, pk.v_vrg, pk.lambda_t_b_vrg),
        "p_mus": (q_co * pk.f_mus_frac, pk.v_mus, pk.lambda_t_b_mus),
        "p_fat": (q_co * pk.f_fat_frac, pk.v_fat, pk.lambda_t_b_fat),
    }
    state.p_alv = state.p_art = p_art
    for name, (flow, volume, partition) in flows.items():
        setattr(state, name, p_art * (1.0 - np.exp(-flow / volume / partition * duration_min)))
    state.p_ven = sum(flow * getattr(state, name) for name, (flow, _, _) in flows.items()) / q_co
    state.mac = state.p_vrg * 100.0 / pk.mac_age

    # Lung balance VA (Fi - FA) = Q λ (FA - Fv), then circuit balance FGF (dial - Fi) = uptake.
    resp_mech = engine.resp_mech
    va = resp_mech.set_rr * max(0.0, resp_mech.set_vt - engine.resp.vd_deadspace)
    uptake = q_co * pk.lambda_b_g * (state.p_alv - state.p_ven)
    fi_agent = state.p_alv + uptake / max(va, 0.1)
    dial_pct = 100.0 * (fi_agent + uptake / max(engine.circuit.fgf_total(), 0.1))
    engine.set_vaporizer("Sevoflurane", dial_pct)
    return fi_agent


def _run_hidden_settle(engine: "SimulationEngine", profile: StartupProfile) -> None:
    """Run a short settle for circuit and physiology transients without visible side effects."""
    saved_vol_clearance = engine.hemo.vol_clearance
    saved_maintenance_rate = engine.maintenance_fluid_rate_ml_min
    saved_time = engine.state.time
    engine.hemo.vol_clearance = 0.0
    engine.maintenance_fluid_rate_ml_min = 0.0

    steps = max(1, int(profile.settle_seconds / profile.settle_dt_seconds))
    for _ in range(steps):
        depth_index, metabolic_factor = runtime_core.compute_depth_metabolic_context(
            engine,
            engine.patient.baseline_temp,
            engine.state.propofol_ce,
            engine.state.mac,
            shiver_level=0.0,
        )
        engine._depth_index = depth_index
        engine._metabolic_factor = metabolic_factor
        runtime_core.update_pk_hemodynamics(engine, engine.state.co)
        fi_sevo, fi_n2o = runtime_core.step_machine(engine, profile.settle_dt_seconds)
        runtime_core.step_pk(engine, profile.settle_dt_seconds, fi_sevo, fi_n2o, engine.state.co)
        physiology = runtime_core.step_physiology(engine, profile.settle_dt_seconds, runtime_core.zero_disturbance())
        projection_core.project_runtime_physiology(engine, physiology)
        engine.state.time += profile.settle_dt_seconds

    engine.hemo.vol_clearance = saved_vol_clearance
    engine.maintenance_fluid_rate_ml_min = saved_maintenance_rate
    engine.state.temp_c = engine.patient.baseline_temp
    engine._metabolic_factor = 1.0
    engine._shiver_level = 0.0
    # Redistribution belongs to the maintenance history, not the visible start.
    engine._redistributed_heat_j = runtime_core.redistribution_target_j(engine, engine._depth_index)
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
    if targets.nore_ce > 0.0:
        maintenance_rate = engine.nore_rate_ug_sec
        engine.enable_tci("nore", targets.nore_ce, mode="plasma")
        engine.nore_rate_ug_sec = maintenance_rate
