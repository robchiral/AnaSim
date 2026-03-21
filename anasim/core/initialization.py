from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import TYPE_CHECKING

from scipy.optimize import minimize, root_scalar

from .constants import TEMP_METABOLIC_COEFFICIENT
from .state import AirwayType
from .step_helpers import ZERO_DIST
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
    map_target: float
    primary_hypnotic: str
    bootstrap_duration_sec: float = 30.0 * 60.0
    bootstrap_dt_sec: float = 2.0
    primary_bounds: tuple[float, float] = (0.0, 1.0)
    remi_bounds: tuple[float, float] = (0.0, 1.0)
    remi_soft_cap: float | None = None
    maintenance_dial_multiplier: float = 1.0
    bootstrap_fgf_o2_l_min: float = 2.0


@dataclass(frozen=True, slots=True)
class StartupTargets:
    prop_ce: float = 0.0
    remi_ce: float = 0.0
    nore_ce: float = 0.0
    mac: float = 0.0


TIVA_PROFILE = StartupProfile(
    name="steady_state_tiva",
    bis_target=55.0,
    tol_target=0.9,
    map_target=75.0,
    primary_hypnotic="propofol",
    primary_bounds=(2.8, 4.2),
    remi_bounds=(1.0, 2.8),
    remi_soft_cap=2.5,
)

BALANCED_PROFILE = StartupProfile(
    name="steady_state_balanced",
    bis_target=45.0,
    tol_target=0.9,
    map_target=75.0,
    primary_hypnotic="volatile",
    primary_bounds=(0.8, 1.2),
    remi_bounds=(0.0, 3.0),
    maintenance_dial_multiplier=1.55,
    bootstrap_fgf_o2_l_min=6.0,
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
    _apply_startup_targets(engine, profile, targets)
    _run_hidden_bootstrap(engine, profile)


def _select_profile(engine: "SimulationEngine") -> StartupProfile:
    if "balanced" in str(engine.config.maint_type).lower():
        return BALANCED_PROFILE
    return TIVA_PROFILE


def _solve_startup_targets(engine: "SimulationEngine", profile: StartupProfile) -> StartupTargets:
    bis_model = engine.bis
    tol_model = engine.tol_pd
    hemo_model = engine.hemo

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

    x0 = (
        sum(profile.primary_bounds) / 2.0,
        sum(profile.remi_bounds) / 2.0,
    )
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

    nore_ce = _solve_nore_target(hemo_model, prop_ce, float(remi_ce), mac, profile.map_target)
    return StartupTargets(prop_ce=prop_ce, remi_ce=float(remi_ce), nore_ce=nore_ce, mac=mac)


def _solve_nore_target(hemo_model, prop_ce: float, remi_ce: float, mac: float, map_target: float) -> float:
    def error_func(ce_nore: float) -> float:
        if ce_nore < 0.0:
            return -100.0
        state = hemo_model.calculate_steady_state(prop_ce, remi_ce, ce_nore, mac_sevo=mac)
        return state.map - map_target

    if error_func(0.0) >= 0.0:
        return 0.0

    try:
        result = root_scalar(error_func, bracket=[0.0, 50.0], method="brentq")
    except ValueError:
        return 50.0
    return float(result.root)


def _configure_controlled_ventilation(engine: "SimulationEngine", targets: StartupTargets) -> None:
    baseline_rr = max(1.0, getattr(engine.patient, "baseline_rr", 12.0))
    baseline_vt_l = max(0.1, getattr(engine.patient, "baseline_vt", 500.0) / 1000.0)
    baseline_mv = baseline_rr * baseline_vt_l
    temp_c = getattr(engine.patient, "baseline_temp", 37.0)
    depth_index = targets.mac + (targets.prop_ce / engine.thermal_tuning.depth_propofol_scale)
    depth_factor = min(1.0, depth_index)
    metabolic_factor = 1.0
    if abs(temp_c - 37.0) >= engine.thermal_tuning.metabolic_temp_threshold_c:
        metabolic_factor *= TEMP_METABOLIC_COEFFICIENT ** (37.0 - temp_c)
    metabolic_factor *= (1.0 - engine.thermal_tuning.metabolic_reduction_max * depth_factor)
    metabolic_factor = max(0.5, metabolic_factor)
    target_mv = baseline_mv * metabolic_factor
    vent_vt = target_mv / baseline_rr if baseline_rr > 0 else baseline_vt_l
    vent_vt = clamp(vent_vt, 0.25, 0.8)
    engine.set_vent_settings(rr=baseline_rr, vt=vent_vt, peep=5.0, ie="1:2", mode="VCV")


def _apply_startup_targets(
    engine: "SimulationEngine",
    profile: StartupProfile,
    targets: StartupTargets,
) -> None:
    engine.set_vaporizer(engine.active_agent, 0.0)

    if targets.prop_ce > 0.0:
        engine.enable_tci("propofol", targets.prop_ce, mode="effect_site")
    if targets.remi_ce > 0.0:
        engine.enable_tci("remi", targets.remi_ce, mode="effect_site")
    if targets.nore_ce > 0.0:
        engine.enable_tci("nore", targets.nore_ce, mode="plasma")

    if profile.primary_hypnotic == "volatile" and engine.pk_sevo:
        target_pct = float(engine.pk_sevo.mac_age) * targets.mac * profile.maintenance_dial_multiplier
        engine.set_vaporizer("Sevoflurane", target_pct)


def _run_hidden_bootstrap(engine: "SimulationEngine", profile: StartupProfile) -> None:
    rng_states = {
        "rng": copy.deepcopy(engine.rng.bit_generator.state),
        "capno": copy.deepcopy(engine._capno_rng.bit_generator.state),
        "ecg": copy.deepcopy(engine._ecg_rng.bit_generator.state),
        "nibp": copy.deepcopy(engine._nibp_rng.bit_generator.state),
    }

    saved_vol_clearance = getattr(engine.hemo, "vol_clearance", None)
    saved_fgf = (engine.circuit.fgf_o2, engine.circuit.fgf_air, engine.circuit.fgf_n2o)
    engine.hemo.vol_clearance = 0.0
    engine.set_fgf(profile.bootstrap_fgf_o2_l_min, 0.0, 0.0)

    steps = max(1, int(profile.bootstrap_duration_sec / profile.bootstrap_dt_sec))
    for _ in range(steps):
        _hidden_bootstrap_step(engine, profile.bootstrap_dt_sec)

    engine.hemo.vol_clearance = saved_vol_clearance
    engine.set_fgf(*saved_fgf)

    engine.rng.bit_generator.state = rng_states["rng"]
    engine._capno_rng.bit_generator.state = rng_states["capno"]
    engine._ecg_rng.bit_generator.state = rng_states["ecg"]
    engine._nibp_rng.bit_generator.state = rng_states["nibp"]

    baseline_temp = getattr(engine.patient, "baseline_temp", 37.0)
    engine.state.temp_c = baseline_temp
    engine._cached_temp_metabolic = baseline_temp
    engine._cached_temp_metabolic_factor = 1.0
    engine._metabolic_factor = 1.0
    engine._shiver_level = 0.0
    engine._do2_ratio = 1.0
    engine.time_brady = 0.0
    engine.time_hypotension = 0.0
    engine.time_tachy = 0.0
    engine.hemo.total_crystalloid_in_ml = 0.0
    engine.hemo.total_colloid_in_ml = 0.0
    engine.hemo.total_blood_in_ml = 0.0
    engine.hemo.total_urine_out_ml = 0.0
    engine.hemo.total_blood_out_ml = 0.0
    engine.hemo.total_leak_out_ml = 0.0
    engine.hemo.total_third_space_ml = 0.0
    engine.hemo.cumulative_fluid_given = 0.0
    engine._tci_accumulators = {}
    engine._next_nibp_time = 0.0
    engine.state.time = 0.0


def _hidden_bootstrap_step(engine: "SimulationEngine", dt: float) -> None:
    depth_scale = engine.thermal_tuning.depth_propofol_scale
    engine._depth_index = engine.state.mac + (engine.state.propofol_ce / depth_scale)
    temp_c = getattr(engine.patient, "baseline_temp", 37.0)
    metabolic_factor = 1.0
    if abs(temp_c - 37.0) >= engine.thermal_tuning.metabolic_temp_threshold_c:
        metabolic_factor *= TEMP_METABOLIC_COEFFICIENT ** (37.0 - temp_c)
    depth_factor = clamp(engine._depth_index, 0.0, 1.0)
    metabolic_factor *= (1.0 - engine.thermal_tuning.metabolic_reduction_max * depth_factor)
    engine._metabolic_factor = max(0.5, metabolic_factor)

    updated_pk_models = engine._update_pk_hemodynamics(engine.state.co)
    if updated_pk_models:
        engine.sync_active_tci_from_pk(*updated_pk_models)

    engine._step_tci(dt)
    fi_sevo, fi_n2o = engine._step_machine(dt)
    engine._step_pk(dt, fi_sevo, fi_n2o, engine.state.co)
    engine._step_physiology(dt, ZERO_DIST)
    engine.state.time += dt
