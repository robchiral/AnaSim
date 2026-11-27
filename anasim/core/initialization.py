from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from anasim.patient.patient import Patient
from anasim.patient.pd import BISModel, ClinicalResponseModel
from anasim.patient.pd.anesthesia import hypothermia_narcosis_fraction
from anasim.physiology.disturbances import DisturbanceEffects

from . import monitors as monitor_core
from . import projection as projection_core
from . import runtime as runtime_core
from .constants import CMH2O_TO_MMHG
from .drug_api import _tci_max_rate
from .drug_registry import get_drug_spec
from .state import AirwayType
from .tci import TCIController, _discretize
from .utils import clamp, clamp01

if TYPE_CHECKING:
    from .engine import SimulationEngine


STARTUP_BIS_BAND_SCALE = 8.0
STARTUP_TOL_WEIGHT = 4.0


@dataclass(frozen=True, slots=True)
class StartupProfile:
    bis_target: float
    tol_target: float
    primary_hypnotic: str
    history_minutes: float = 30.0
    settle_seconds: float = 60.0
    settle_dt_seconds: float = 1.0
    primary_bounds: tuple[float, float] = (0.0, 1.0)
    remi_bounds: tuple[float, float] = (0.0, 1.0)
    fgf_o2_l_min: float = 2.0


@dataclass(frozen=True, slots=True)
class StartupTargets:
    prop_ce: float = 0.0
    remi_ce: float = 0.0
    mac: float = 0.0


TIVA_PROFILE = StartupProfile(
    bis_target=50.0,
    tol_target=0.9,
    primary_hypnotic="propofol",
    primary_bounds=(0.5, 1.5),  # Multiples of the age-adjusted BIS Ce50
    remi_bounds=(1.0, 4.0),
)

BALANCED_PROFILE = StartupProfile(
    bis_target=45.0,
    tol_target=0.9,
    primary_hypnotic="volatile",
    primary_bounds=(0.8, 1.2),
    remi_bounds=(0.0, 3.0),
    fgf_o2_l_min=6.0,
)


def initialize_engine_state(engine: SimulationEngine) -> None:
    """Initialize the configured awake or maintenance state."""
    engine.pending_infusions.clear()
    if engine.config.mode == "steady_state":
        _initialize_steady_state(engine)
    else:
        _initialize_awake(engine)


def _initialize_awake(engine: SimulationEngine) -> None:
    engine.state.airway_mode = AirwayType.NONE
    engine.resp.state.apnea = False
    engine.set_vent_power(False)
    engine.set_vaporizer(0.0)
    base_temp = engine.patient.baseline_temp
    engine.thermal.update_depth(base_temp, 0.0, 0.0)
    engine.thermal.settle_at_depth()
    awake_bis = engine.bis.compute_bis(0.0, temp_c=base_temp)
    awake_unconscious = hypothermia_narcosis_fraction(base_temp)
    engine.thermal.shiver_level = engine.thermal.target_shivering(
        base_temp, awake_bis, 0.0, 1.0, unconscious=awake_unconscious, perfusing=True,
    )
    engine.thermal.update_depth(base_temp, 0.0, 0.0)
    engine.state.temp_c = base_temp
    runtime_core.update_pk_covariates(engine, engine.hemo.base_co_l_min)
    engine.pk_nore.equilibrate_endogenous()
    engine.hemo.initialize_steady_state(
        0.0, 0.0, engine.pk_nore.state.ce,
        pit=engine.hemo.config.pit_0,
        shivering=engine.thermal.shiver_level,
    )
    if base_temp != 37.0 or engine.thermal.shiver_level > 0.0:
        engine.resp.equilibrate_blood_gases(
            engine.hemo.state.co,
            engine.resp.vco2 * max(0.25, engine.thermal.metabolic_factor) / 1000.0,
        )
    if engine.thermal.shiver_level > 0.0:
        # Equilibrate chemoreflex ventilation to elevated shivering CO2 production.
        met_factor = max(0.25, engine.thermal.metabolic_factor)
        co = engine.hemo.state.co
        hb = engine.hemo.hb_conc
        hct = engine.hemo.get_hematocrit()
        bv = engine.hemo.blood_volume
        for _ in range(180):
            engine.resp.step(
                2.0, 0.0, 0.0, fio2=0.21, cardiac_output=co,
                metabolic_factor=met_factor, temp_c=base_temp,
                hb_g_dl=hb, hematocrit=hct, blood_volume_ml=bv,
                unconscious=awake_unconscious,
            )
    runtime_core.update_pk_covariates(engine, engine.hemo.state.co)


def _initialize_steady_state(engine: SimulationEngine) -> None:
    p = engine.patient
    nominal_patient = Patient(age=p.age, weight=p.weight, height=p.height, sex=p.sex)
    profile = _select_profile(engine)
    targets = _solve_startup_targets(engine, profile, nominal_patient=nominal_patient)

    engine.state.airway_mode = AirwayType.ETT
    engine.resp.state.apnea = True
    _configure_controlled_ventilation(engine, targets)
    nominal_pk = _seed_steady_state_subsystems(engine, profile, targets, nominal_patient=nominal_patient)
    projection_core.sync_state_from_models(engine)
    spec = engine.disturbances.spec
    if engine.disturbance_active and spec is not None and not spec.finite:
        # Sustained stimulation configured for a maintenance start, such as
        # ongoing surgery, is already established.
        engine.disturbance_start_time = engine.state.time - spec.established_s
    if engine.config.tci_enabled:
        _attach_startup_controllers(engine, targets)
    _run_hidden_settle(engine, profile, targets, nominal_pk=nominal_pk, nominal_patient=nominal_patient)
    runtime_core.update_pk_covariates(engine, engine.state.co)
    if engine.infusion_rates["nore"] <= 0.0:
        engine.pk_nore.equilibrate_endogenous()
    projection_core.sync_pk_state(engine)
    engine.state.time = 0.0
    engine._next_nibp_time = 0.0
    if engine.config.tci_enabled:
        for key in engine.tci:
            engine._recompute_tci(key, 0.0)


def _select_profile(engine: SimulationEngine) -> StartupProfile:
    if engine.config.maint_type == "balanced":
        return BALANCED_PROFILE
    return TIVA_PROFILE


def _solve_startup_targets(
    engine: SimulationEngine, profile: StartupProfile, *, nominal_patient: Patient | None = None,
) -> StartupTargets:
    if nominal_patient is None:
        p = engine.patient
        nominal_patient = Patient(age=p.age, weight=p.weight, height=p.height, sex=p.sex)
    bis_model = BISModel(nominal_patient)
    response_model = ClinicalResponseModel(nominal_patient.age)

    def objective(x):
        primary_load = x[0]
        remi_ce = x[1]
        if profile.primary_hypnotic == "volatile":
            prop_ce = 0.0
            mac = primary_load
        else:
            prop_ce = primary_load * bis_model.c50
            mac = 0.0

        bis_val = bis_model.compute_bis(prop_ce, mac_sevo=mac)
        tol_val = response_model.tolerance(prop_ce, remi_ce, mac_sevo=mac)
        tol_deficit = max(0.0, profile.tol_target - tol_val)
        return (
            ((bis_val - profile.bis_target) ** 2) / (STARTUP_BIS_BAND_SCALE ** 2)
            + STARTUP_TOL_WEIGHT * (tol_deficit ** 2)
        )

    x0 = (sum(profile.primary_bounds) / 2.0, sum(profile.remi_bounds) / 2.0)
    result = minimize(
        objective,
        x0,
        bounds=(profile.primary_bounds, profile.remi_bounds),
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-12},
    )
    primary_load, remi_ce = map(float, result.x)
    if profile.primary_hypnotic == "volatile":
        return StartupTargets(remi_ce=remi_ce, mac=primary_load)
    return StartupTargets(prop_ce=primary_load * bis_model.c50, remi_ce=remi_ce)


def _configure_controlled_ventilation(engine: SimulationEngine, targets: StartupTargets) -> None:
    resp = engine.resp
    # COPD keeps a lower rate to leave time for exhalation.
    rate_limit = 18.0 if engine.patient.copd_severity > 0.0 else 30.0
    baseline_rr = clamp(resp.rr_0, 1.0, rate_limit)
    _depth_index, metabolic_factor = engine.thermal.depth_and_metabolism(
        engine.patient.baseline_temp, targets.prop_ce, targets.mac,
    )
    target_paco2 = resp.paco2_setpoint * resp._pco2_temp_scale_now
    target_va = (
        resp.co2_constant * resp.vco2 * metabolic_factor
        / max(10.0, target_paco2)
        / (1.0 - resp.copd_alveolar_deadspace)
    )
    # Body mass raises CO2 production without enlarging the lungs, so VT stops
    # at 8 mL/kg predicted body weight (tapering to 6 mL/kg PBW in severe obesity
    # to avoid excessive peak pressures) and the rate rises instead.
    vt_per_kg_pbw = 0.008 - 0.002 * clamp01((engine.patient.bmi - 35.0) / 15.0)
    max_vt = min(0.8, vt_per_kg_pbw * engine.patient.predicted_body_weight())
    vent_vt = clamp(resp.vd_deadspace + target_va / baseline_rr, 0.25, max_vt)
    target_rr = target_va / max(0.05, vent_vt - resp.vd_deadspace)
    vent_rr = min(rate_limit, max(baseline_rr, target_rr))
    engine.set_vent_settings(rr=vent_rr, vt=vent_vt, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)


def _seed_steady_state_subsystems(
    engine: SimulationEngine,
    profile: StartupProfile,
    targets: StartupTargets,
    *,
    nominal_patient: Patient | None = None,
) -> dict[str, Any]:
    engine.set_vaporizer(0.0)
    engine.set_fgf(profile.fgf_o2_l_min, 0.0, 0.0)
    engine.infusion_rates["propofol"] = 0.0
    engine.infusion_rates["remi"] = 0.0
    engine.infusion_rates["nore"] = 0.0

    p = engine.patient
    engine.state.temp_c = p.baseline_temp
    runtime_core.update_pk_covariates(engine, engine.hemo.base_co_l_min)
    if nominal_patient is None:
        nominal_patient = Patient(age=p.age, weight=p.weight, height=p.height, sex=p.sex)
    nominal_pk = {
        key: get_drug_spec(key).pk_model(nominal_patient, engine.config)
        for key in ("propofol", "remi")
    }
    for key, target_ce in (("propofol", targets.prop_ce), ("remi", targets.remi_ce)):
        if target_ce <= 0.0:
            continue
        if engine.config.tci_enabled:
            engine.infusion_rates[key] = _seed_tci_history(
                engine, key, target_ce, profile.history_minutes,
            )
        else:
            nom_model = nominal_pk[key]
            rate_per_min = _seed_linear_history(
                engine.pk[key], target_ce, profile.history_minutes, engine.pump_pk.get(key),
                reference_model=nom_model,
            )
            nom_model.set_state_vector(_infusion_history(nom_model, profile.history_minutes) * rate_per_min)
            engine.infusion_rates[key] = _maintenance_rate(nom_model, target_ce)
    engine.resp.initialize_drug_effects(engine.pk["propofol"].state.ce_resp, engine.pk["remi"].state.ce_resp)
    prop_cp = engine.pk["propofol"].state.c1
    remi_cp = engine.pk["remi"].state.c1
    engine.pk_nore.update_propofol(prop_cp)
    engine.pk_nore.equilibrate_endogenous()
    initial_mac_sevo = (
        targets.mac * engine.pk_sevo.mac_at_temp(engine.patient.baseline_temp) / engine.pk_sevo.mac_age
        if profile.primary_hypnotic == "volatile"
        else 0.0
    )
    peep_init = engine.vent.settings.peep
    pit_init = engine.hemo.config.pit_0 + CMH2O_TO_MMHG * engine.hemo.config.paw_transmission * (peep_init + 3.0)
    engine.hemo.initialize_steady_state(
        prop_cp,
        remi_cp,
        engine.pk_nore.state.ce,
        mac_sevo=initial_mac_sevo,
        remifentanil_cp=remi_cp,
        pit=pit_init,
        peep_cmH2O=peep_init,
    )
    fi_agent = 0.0
    if profile.primary_hypnotic == "volatile":
        _seed_volatile_history(engine, targets.mac, profile.history_minutes)
        # Start at the untreated anesthetic steady state so the hidden settle
        # handles only circuit and monitor transients.
        mac_sevo_art = engine.pk_sevo.state.p_art * 100.0 / engine.pk_sevo.mac_age
        engine.hemo.initialize_steady_state(
            prop_cp,
            remi_cp,
            engine.pk_nore.state.ce,
            mac_sevo=mac_sevo_art,
            remifentanil_cp=remi_cp,
            pit=pit_init,
            peep_cmH2O=peep_init,
        )
        fi_agent = _seed_volatile_history(engine, targets.mac, profile.history_minutes)
    o2_uptake = _oxygen_uptake_l_min(engine, targets)
    engine.circuit.equilibrate(o2_uptake, fi_agent)
    engine.resp.equilibrate_oxygen(
        engine.circuit.composition.fio2,
        1.0,
        engine.vent.settings.peep,
        fin2=engine.circuit.composition.fin2,
    )
    # Tissue CO2 belongs to the same maintenance history as the seeded drugs.
    engine.resp.equilibrate_blood_gases(
        engine.hemo.state.co,
        o2_uptake * engine.resp.rq,
    )
    return nominal_pk


def _infusion_history(pk_model, duration_min: float) -> np.ndarray:
    """Return the compartment concentrations after duration_min of infusion at one unit per minute."""
    A, B = pk_model.get_ss_matrices()
    return ((np.eye(A.shape[0]) - expm(A * duration_min)) @ np.linalg.solve(-A, B))[:, 0]


def _maintenance_rate(pk_model, target_ce: float) -> float:
    """Return the infusion rate (model units/s) that holds C1 == target_ce."""
    vec = pk_model.state_vector()
    rate_per_min = pk_model.elimination_clearance * target_ce
    for idx, (v, cl) in enumerate(((pk_model.v2, pk_model.cl2), (pk_model.v3, pk_model.cl3)), start=1):
        if v > 0.0 and cl > 0.0:
            rate_per_min += cl * max(0.0, target_ce - float(vec[idx]))
    return rate_per_min / 60.0


def _seed_tci_history(engine: SimulationEngine, key: str, target_ce: float, duration_min: float) -> float:
    """Simulate duration_min of effect-site TCI at target_ce; return rate in model units/s."""
    spec = get_drug_spec(key)
    max_rate = _tci_max_rate(spec).internal_rate(engine.patient.weight)
    pk_model = engine.pk[key]
    pump = engine.pump_pk.get(key, pk_model)
    ctrl = TCIController(
        pump, target_compartment="effect_site", max_rate=max_rate, sampling_time=10.0, control_time=10.0,
    )
    ctrl.set_target(target_ce)
    A_pump, B_pump = pump.get_ss_matrices()
    Ad_pump, Bd_pump = _discretize(A_pump / 60.0, B_pump, 10.0)
    x_pump = pump.state_vector()
    if pump is not pk_model:
        A_pk, B_pk = pk_model.get_ss_matrices()
        Ad_pk, Bd_pk = _discretize(A_pk / 60.0, B_pk, 10.0)
        x_pk = pk_model.state_vector()
    steps = max(1, round(duration_min * 60.0 / 10.0))
    rate = 0.0
    for step in range(steps):
        pump.set_state_vector(x_pump)
        rate = ctrl.step(sim_time=step * 10.0)
        x_pump = Ad_pump @ x_pump + Bd_pump[:, 0] * rate
        if pump is not pk_model:
            x_pk = Ad_pk @ x_pk + Bd_pk[:, 0] * rate
    pump.set_state_vector(x_pump)
    if pump is not pk_model:
        pk_model.set_state_vector(x_pk)
    return rate


def _seed_linear_history(pk_model, target_ce: float, duration_min: float, pump_model=None, *, reference_model=None) -> float:
    """Seed the state after a constant infusion that reaches target Ce; return the rate per minute.

    If reference_model is given, nominal delivery follows it while pk_model and pump_model receive the same rate.
    """
    reference = reference_model if reference_model is not None else pk_model
    ref_history = _infusion_history(reference, duration_min)
    rate_per_min = target_ce / float(ref_history[-1])
    if reference is pk_model:
        pk_model.set_state_vector(ref_history * rate_per_min)
    else:
        pk_model.set_state_vector(_infusion_history(pk_model, duration_min) * rate_per_min)
    if pump_model is not None:
        pump_model.set_state_vector(_infusion_history(pump_model, duration_min) * rate_per_min)
    return rate_per_min


def _oxygen_uptake_l_min(engine: SimulationEngine, targets: StartupTargets) -> float:
    _, metabolic_factor = engine.thermal.depth_and_metabolism(
        engine.patient.baseline_temp, targets.prop_ce, targets.mac,
    )
    return engine.resp.vco2 * metabolic_factor / engine.resp.rq / 1000.0


def _seed_volatile_history(engine: SimulationEngine, target_mac: float, duration_min: float) -> float:
    """Seed tissue partial pressures after an anesthetic maintenance history.

    Sets the vaporizer to the dial that holds alveolar partial pressure constant
    at the seeded uptake, and returns the inspired fraction.
    """
    pk = engine.pk_sevo
    state = pk.state
    corrected_mac = pk.mac_at_temp(engine.patient.baseline_temp)
    p_art = corrected_mac * target_mac / 100.0
    q_co = max(engine.hemo.state.co, 0.1)
    shunt = clamp(engine.aeration.shunt_fraction, 0.0, 0.9)
    f_vrg, f_mus, f_fat = pk.flow_fractions(q_co)
    flows = {
        "p_vrg": (q_co * f_vrg, pk.v_vrg, pk.lambda_t_b_vrg),
        "p_mus": (q_co * f_mus, pk.v_mus, pk.lambda_t_b_mus),
        "p_fat": (q_co * f_fat, pk.v_fat, pk.lambda_t_b_fat),
    }
    for name, (flow, volume, partition) in flows.items():
        setattr(state, name, p_art * -math.expm1(-flow / volume / partition * duration_min))
    state.p_brain = state.p_vrg
    state.p_ven = sum(flow * getattr(state, name) for name, (flow, _, _) in flows.items()) / q_co
    state.p_alv = clamp01((p_art - shunt * state.p_ven) / max(0.1, 1.0 - shunt))
    state.p_art = (1.0 - shunt) * state.p_alv + shunt * state.p_ven
    state.mac = state.p_brain * 100.0 / corrected_mac

    # Lung balance VA (Fi - FA) = (1 - s) Q λ (FA - Fv), then STPD circle balance
    # vapor_flow - uptake = (FGF + vapor_flow - total_uptake) * Fi.
    settings = engine.vent.settings
    va = (
        settings.rr
        * max(0.0, settings.tv / 1000.0 - engine.resp.vd_deadspace)
        * (1.0 - engine.resp.copd_alveolar_deadspace)
        * engine.airway_status.ventilation_efficiency
    )
    pk._temp_c = float(engine.patient.baseline_temp)
    uptake_btps = q_co * (1.0 - shunt) * pk.lambda_b_g_at_temp(engine.patient.baseline_temp) * (state.p_alv - state.p_ven)
    fi_agent = clamp01(state.p_alv + uptake_btps / max(va, 0.1))
    uptake_stpd = uptake_btps * engine.resp._btps_to_stpd
    fgf = max(engine.circuit.fgf_total(), 0.1)
    o2_up = _oxygen_uptake_l_min(engine, StartupTargets(mac=target_mac))
    vapor_flow = max(0.0, (fgf - o2_up) * fi_agent / max(1e-6, 1.0 - fi_agent) + uptake_stpd)
    dial_pct = 100.0 * vapor_flow / (fgf + vapor_flow)
    engine.set_vaporizer(dial_pct)
    return fi_agent


def _run_hidden_settle(
    engine: SimulationEngine,
    profile: StartupProfile,
    targets: StartupTargets,
    *,
    nominal_pk: dict[str, Any] | None = None,
    nominal_patient: Patient | None = None,
) -> None:
    """Run a short settle for circuit and physiology transients without visible side effects."""
    saved_hemo_config = engine.hemo.config
    saved_maintenance_rate = engine.maintenance_fluid_rate_ml_min
    saved_time = engine.state.time
    saved_vaporizer_level = engine.vaporizer.state.level
    engine.hemo.config = replace(saved_hemo_config, vol_clearance=0.0)
    engine.maintenance_fluid_rate_ml_min = 0.0

    nom_pk = nominal_pk or {}
    if nominal_patient is None:
        p = engine.patient
        nominal_patient = Patient(age=p.age, weight=p.weight, height=p.height, sex=p.sex)
    nominal_bv_l = nominal_patient.estimate_blood_volume() / 1000.0
    steps = max(1, int(profile.settle_seconds / profile.settle_dt_seconds))
    # Capnograph and EtCO2 readout latch within ~2-3 breaths (~15 s), so only
    # collect 100 Hz airway samples during the final settle window.
    capno_start = max(0, steps - math.ceil(15.0 / profile.settle_dt_seconds))
    for step_idx in range(steps):
        collect_capno = step_idx >= capno_start
        thermal = engine.thermal
        thermal.depth_index, thermal.metabolic_factor = thermal.depth_and_metabolism(
            engine.patient.baseline_temp,
            projection_core.thermal_hypnotic_ce(engine),
            projection_core.thermal_mac(engine),
        )
        runtime_core.update_pk_covariates(engine, engine.state.co)
        if nom_pk:
            # Keep nominal manual-infusion reference free of patient disease covariates
            # while still matching normal anesthetized cardiac-output suppression.
            co_ratio = clamp(max(0.0, engine.state.co) / max(0.1, engine.hemo.base_co_l_min), 0.75, 1.0)
            for nom_model in nom_pk.values():
                nom_model.update_hemodynamics(
                    1.0, co_ratio, 1.0,
                    blood_volume_l=nominal_bv_l,
                )
        if engine.config.tci_enabled:
            runtime_core.step_tci(engine, profile.settle_dt_seconds)
        else:
            for key, target_ce in (("propofol", targets.prop_ce), ("remi", targets.remi_ce)):
                if target_ce > 0.0:
                    ref_model = nom_pk.get(key, engine.pk[key])
                    rate = _maintenance_rate(ref_model, target_ce)
                    engine.infusion_rates[key] = rate
                    if ref_model is not engine.pk[key]:
                        ref_model.step(profile.settle_dt_seconds, rate)
        fi_sevo, fi_n2o = runtime_core.step_machine(engine, profile.settle_dt_seconds)
        runtime_core.step_pk(engine, profile.settle_dt_seconds, fi_sevo, fi_n2o, engine.state.co)
        engine._tol_current = projection_core.stimulus_tolerance(engine)
        spec = engine.disturbances.spec
        stimulus = (
            DisturbanceEffects()
            if spec is None or spec.finite
            else runtime_core.stimulus_effects(engine, profile.settle_dt_seconds)
        )
        physiology = runtime_core.step_physiology(
            engine, profile.settle_dt_seconds, stimulus, collect_samples=collect_capno,
        )
        projection_core.project_runtime_physiology(engine, physiology)
        hemo_state = physiology.hemo_state
        engine.cerebral.step(
            profile.settle_dt_seconds,
            hemo_state.perfusion_map,
            engine.state.hb_g_dl,
            engine.state.sao2,
            physiology.resp_state.pa_co2_37,
            physiology.resp_state.ph_37,
            venous_pressure=hemo_state.venous_pressure,
            temp_c=engine.state.temp_c,
            anesthetic_depth=clamp01(engine.thermal.depth_index),
            volatile_luxury=clamp01(engine.state.mac_sevo + 0.5 * engine.state.mac_n2o),
            glucose_mg_dl=engine.glucose.arterial_glucose_mg_dl,
        )
        projection_core.seed_arterial_monitors(
            engine, hemo_state, hr=engine.state.hr, map_mmhg=engine.state.map, sv_ml=engine.state.sv,
        )
        if collect_capno:
            engine.state.capno_co2 = float(
                monitor_core.compute_capno_value(engine, physiology.resp_state, profile.settle_dt_seconds)[-1]
            )
        engine.state.time += profile.settle_dt_seconds

    engine.hemo.config = saved_hemo_config
    engine.maintenance_fluid_rate_ml_min = saved_maintenance_rate
    engine.vaporizer.state.level = saved_vaporizer_level
    transport = engine.resp.gas_transport
    transport.consumed_o2_l = 0.0
    transport.produced_l = 0.0
    transport.exhaled_l = 0.0
    transport.oxygen_from_air_l = 0.0
    engine.state.temp_c = engine.patient.baseline_temp
    # Redistribution belongs to the maintenance history, not the visible start.
    engine.thermal.settle_at_depth()
    engine.glucose.reset_to_baseline(engine.hemo.fluids)
    engine.cerebral.hypoglycemic_dysfunction = 0.0
    engine._pulseless_s = 0.0
    engine.vent.breath_count = 0
    engine.state.time = saved_time


def _attach_startup_controllers(engine: SimulationEngine, targets: StartupTargets) -> None:
    """Attach TCI controllers before the hidden settle so they hold the seeded targets through equilibration."""
    projection_core.sync_pk_state(engine)
    for key, target_ce in (("propofol", targets.prop_ce), ("remi", targets.remi_ce)):
        if target_ce > 0.0:
            engine.enable_tci(key, target_ce, mode="effect_site")
