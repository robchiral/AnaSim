from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from anasim.core.constants import (
    TEMP_METABOLIC_COEFFICIENT,
    RR_APNEA_THRESHOLD,
    SHIVER_BASE_THRESHOLD,
    SHIVER_DEPTH_DROP_MAX,
    SHIVER_REMI_DROP_MAX,
    SHIVER_DELTA_FULL,
    SHIVER_BIS_ON,
    SHIVER_BIS_FULL,
    SHIVER_MAX_MULTIPLIER,
    SHIVER_TAU_ON,
    SHIVER_TAU_OFF,
)
from anasim.core.drug_registry import PK_HEMODYNAMIC_TARGETS, TCI_TARGET_CONFIG
from anasim.core.enums import RhythmType
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.resp_mech import VentMode

from .monitors import phase_from_rr, step_monitors
from .projection import PhysiologyStepState, project_runtime_physiology, set_state_float_fields, sync_pk_state
from .state import AirwayType

if TYPE_CHECKING:
    from .engine import SimulationEngine

logger = logging.getLogger(__name__)


def zero_disturbance() -> DisturbanceEffects:
    return DisturbanceEffects()


def compute_depth_metabolic_context(
    engine: "SimulationEngine",
    temp_c: float,
    prop_ce: float,
    mac: float,
    shiver_level: float = 0.0,
) -> tuple[float, float]:
    """Return depth index and metabolic factor shared by runtime and initialization."""
    depth_scale = engine.thermal_tuning.depth_propofol_scale
    depth_index = mac + (prop_ce / depth_scale)
    depth_factor = clamp(depth_index, 0.0, 1.0)

    metabolic_factor = 1.0
    if abs(temp_c - 37.0) >= engine.thermal_tuning.metabolic_temp_threshold_c:
        metabolic_factor *= TEMP_METABOLIC_COEFFICIENT ** (37.0 - temp_c)
    metabolic_factor *= (1.0 - engine.thermal_tuning.metabolic_reduction_max * depth_factor)
    metabolic_factor = max(0.5, metabolic_factor)
    if shiver_level > 0.0:
        metabolic_factor *= (1.0 + SHIVER_MAX_MULTIPLIER * shiver_level)
    return depth_index, metabolic_factor


def step_simulation(engine: "SimulationEngine", dt: float) -> None:
    """Advance the simulation by one step."""
    if dt <= 0 or not engine.running:
        return

    depth_index, metabolic_factor = compute_depth_metabolic_context(
        engine,
        engine.state.temp_c,
        engine.state.propofol_ce,
        engine.state.mac,
        shiver_level=engine._shiver_level,
    )
    engine._depth_index = depth_index
    engine._metabolic_factor = metabolic_factor

    disturbances = step_disturbances(engine, dt)
    updated_pk_models = update_pk_hemodynamics(engine, engine.state.co)
    if updated_pk_models:
        engine.sync_active_tci_from_pk(*updated_pk_models)

    step_tci(engine, dt)
    fi_sevo, fi_n2o = step_machine(engine, dt)
    step_pk(engine, dt, fi_sevo, fi_n2o, engine.state.co)
    physiology = step_physiology(engine, dt, disturbances)
    project_runtime_physiology(engine, physiology)
    step_monitors(engine, dt, physiology.phase, physiology.hemo_state, physiology.resp_state, disturbances)
    update_shivering(engine, dt)
    step_temperature(engine, dt)
    check_patient_viability(engine, dt)


def step_mechanics(engine: "SimulationEngine", dt: float, vent_active: bool, bag_mask_active: bool):
    """Advance respiratory mechanics and return (mech_state, total_peep_effect, mech_rr_for_resp)."""
    resp_mech = engine.resp_mech

    def estimate_effort(vt_l: float) -> float:
        if resp_mech.compliance <= 0:
            return 0.0
        return clamp(vt_l / resp_mech.compliance, 0.0, 20.0)

    if vent_active:
        mech_rr_for_resp = resp_mech.set_rr
        if resp_mech.mode in (VentMode.PSV, VentMode.CPAP):
            saved_settings = resp_mech.snapshot_settings()
            spont_rr = max(0.0, engine.resp.state.rr)
            spont_vt_l = max(0.0, engine.resp.state.vt / 1000.0)
            is_apneic = spont_rr < RR_APNEA_THRESHOLD or engine.resp.state.apnea
            if resp_mech.mode == VentMode.PSV:
                if is_apneic:
                    engine._psv_apnea_timer += dt
                else:
                    engine._psv_apnea_timer = 0.0
                backup_rr = resp_mech.set_rr if resp_mech.set_rr > 0.0 else 0.0
                use_backup = (
                    is_apneic and backup_rr > 0.0 and engine._psv_apnea_timer >= engine.psv_apnea_backup_delay
                )
                if use_backup:
                    spont_rr = backup_rr
            else:
                engine._psv_apnea_timer = 0.0
                use_backup = False
            if is_apneic:
                spont_vt_l = 0.0
            mech_rr_for_resp = spont_rr

            effort_cm_h2o = estimate_effort(spont_vt_l)
            if use_backup:
                effort_cm_h2o = 0.0
            engine._last_patient_effort_cmH2O = effort_cm_h2o

            support_cm_h2o = resp_mech.set_p_insp if resp_mech.mode == VentMode.PSV else 0.0
            resp_mech.set_rr = mech_rr_for_resp
            resp_mech.set_p_insp = clamp(support_cm_h2o, 0.0, 40.0)
            resp_mech.patient_effort_cmH2O = effort_cm_h2o

            mech_state = resp_mech.step(dt)
            total_peep_effect = resp_mech.get_total_peep()
            resp_mech.restore_settings(saved_settings)
            resp_mech.patient_effort_cmH2O = 0.0
        else:
            engine._psv_apnea_timer = 0.0
            engine._last_patient_effort_cmH2O = 0.0
            resp_mech.patient_effort_cmH2O = 0.0
            mech_state = resp_mech.step(dt)
            total_peep_effect = resp_mech.get_total_peep()
    elif bag_mask_active:
        engine._psv_apnea_timer = 0.0
        engine._last_patient_effort_cmH2O = 0.0
        saved_settings = resp_mech.snapshot_settings()
        resp_mech.set_settings(engine.bag_mask_rr, engine.bag_mask_vt, 0.0, ie="1:2", mode="VCV")
        resp_mech.patient_effort_cmH2O = 0.0
        mech_state = resp_mech.step(dt)
        total_peep_effect = resp_mech.get_total_peep()
        resp_mech.restore_settings(saved_settings)
        mech_rr_for_resp = engine.bag_mask_rr
    else:
        engine._psv_apnea_timer = 0.0
        spont_rr = max(0.0, engine.resp.state.rr)
        spont_vt_l = max(0.0, engine.resp.state.vt / 1000.0)
        is_apneic = spont_rr < RR_APNEA_THRESHOLD or engine.resp.state.apnea
        if is_apneic:
            spont_vt_l = 0.0
        engine._last_patient_effort_cmH2O = estimate_effort(spont_vt_l)
        saved_settings = resp_mech.snapshot_settings()
        resp_mech.set_rr = 0.0
        resp_mech.set_peep = 0.0
        resp_mech.patient_effort_cmH2O = 0.0
        mech_state = resp_mech.step(dt)
        total_peep_effect = 0.0
        mech_state.paw_mean = 0.0
        mech_state.auto_peep = 0.0
        resp_mech.restore_settings(saved_settings)
        mech_rr_for_resp = 0.0
    return mech_state, total_peep_effect, mech_rr_for_resp


def update_shivering(engine: "SimulationEngine", dt: float) -> float:
    """Update shivering intensity based on temperature and anesthetic state."""
    state = engine.state
    thermal_tuning = engine.thermal_tuning
    depth_scale = thermal_tuning.depth_propofol_scale
    depth_metric = state.mac + (state.propofol_ce / depth_scale)
    depth_factor = clamp01(depth_metric)

    remi_effect = 0.0
    remi_effect = hill_function(state.remi_ce, engine.resp.c50_remi, engine.resp.gamma_remi)

    threshold = SHIVER_BASE_THRESHOLD - SHIVER_DEPTH_DROP_MAX * depth_factor - SHIVER_REMI_DROP_MAX * remi_effect
    temp_deficit = max(0.0, threshold - state.temp_c)
    cold_drive = clamp01(temp_deficit / SHIVER_DELTA_FULL)

    if SHIVER_BIS_FULL <= SHIVER_BIS_ON:
        emergence = 1.0 if state.bis >= SHIVER_BIS_ON else 0.0
    else:
        emergence = clamp01((state.bis - SHIVER_BIS_ON) / (SHIVER_BIS_FULL - SHIVER_BIS_ON))

    nmba_effect = hill_function(state.roc_ce, engine.resp.c50_nmba, engine.resp.gamma_nmba)
    muscle_factor = clamp01(1.0 - nmba_effect)

    target = cold_drive * emergence * muscle_factor
    tau = SHIVER_TAU_ON if target > engine._shiver_level else SHIVER_TAU_OFF
    if tau > 0:
        engine._shiver_level += (target - engine._shiver_level) * (dt / tau)
    else:
        engine._shiver_level = target
    engine._shiver_level = clamp01(engine._shiver_level)
    set_state_float_fields(state, shivering=engine._shiver_level)
    return engine._shiver_level


def step_temperature(engine: "SimulationEngine", dt: float) -> None:
    """Update patient temperature based on metabolic heat production and heat loss."""
    state = engine.state
    temp_c = state.temp_c
    thermal_tuning = engine.thermal_tuning
    depth_index = state.mac + (state.propofol_ce / thermal_tuning.depth_propofol_scale)
    depth_factor = min(1.0, depth_index)

    metabolic_factor = max(0.5, engine._metabolic_factor)
    current_production = engine.heat_production_basal * metabolic_factor
    t_ambient = thermal_tuning.ambient_temp_c
    base_conductance = thermal_tuning.base_conductance_w_per_c
    anest_conductance_boost = thermal_tuning.anesthetic_conductance_gain * depth_factor
    total_conductance = base_conductance * (1.0 + anest_conductance_boost) * (engine.surface_area / 1.9)

    heat_loss = total_conductance * (temp_c - t_ambient)
    d_depth = (depth_index - engine._last_depth_index) / dt
    engine._last_depth_index = depth_index
    if d_depth > 0:
        heat_loss += thermal_tuning.redistribution_gain_w_per_depth * d_depth

    warming_input = 0.0
    if state.bair_hugger_target > 0:
        dt_warming = max(0.0, state.bair_hugger_target - temp_c)
        warming_input = thermal_tuning.bair_hugger_gain_w_per_c * dt_warming

    net_heat_flux = current_production + warming_input - heat_loss
    heat_capacity = engine.patient.weight * engine.specific_heat
    d_temp = (net_heat_flux * dt) / heat_capacity
    set_state_float_fields(
        state,
        temp_c=clamp(temp_c + d_temp, thermal_tuning.temp_min_c, thermal_tuning.temp_max_c),
    )


def step_disturbances(engine: "SimulationEngine", dt: float) -> DisturbanceEffects:
    """Calculate disturbances and update event-driven volume state."""
    state = engine.state
    if not engine.disturbance_active or not engine.disturbances:
        effects = zero_disturbance()
    else:
        t_rel = max(0.0, state.time - engine.disturbance_start_time)
        effects = engine.disturbances.compute_dist(t_rel)

    depth_factor = min(1.0, engine._depth_index)
    stim_gain = max(0.3, 1.0 - 0.6 * depth_factor)
    effects = DisturbanceEffects(
        bis=effects.bis * stim_gain,
        svr=effects.svr * stim_gain,
        sv=effects.sv * stim_gain,
        hr=effects.hr * stim_gain,
    )

    hemo = engine.hemo
    if engine.active_hemorrhage and hemo:
        rate_sec = engine.hemorrhage_rate_ml_min / 60.0
        hemo.add_volume(-rate_sec * dt)

    if engine.pending_infusions:
        remaining = []
        for infusion in engine.pending_infusions:
            rate_sec = infusion.rate_ml_min / 60.0
            amount_this_step = min(infusion.remaining_ml, rate_sec * dt)
            infusion.remaining_ml -= amount_this_step
            if amount_this_step > 0 and hemo:
                hemo.add_volume(
                    amount_this_step,
                    hematocrit=infusion.hematocrit,
                    retention_fraction=infusion.retention_fraction,
                    label=infusion.label,
                    count_as_bolus=infusion.count_as_bolus,
                )
            if infusion.remaining_ml > 1e-3:
                remaining.append(infusion)
        engine.pending_infusions[:] = remaining

    if hemo and engine.maintenance_fluid_rate_ml_min > 0:
        rate_sec = engine.maintenance_fluid_rate_ml_min / 60.0
        hemo.add_volume(rate_sec * dt, hematocrit=0.0, label="crystalloid", count_as_bolus=False)

    if engine.active_anaphylaxis:
        engine.anaphylaxis_severity = min(1.0, engine.anaphylaxis_severity + engine.anaphylaxis_onset_rate * dt)
    else:
        engine.anaphylaxis_severity = max(0.0, engine.anaphylaxis_severity - engine.anaphylaxis_decay_rate * dt)

    if engine.active_sepsis:
        engine.sepsis_severity = min(1.0, engine.sepsis_severity + engine.sepsis_onset_rate * dt)
    else:
        engine.sepsis_severity = max(0.0, engine.sepsis_severity - engine.sepsis_decay_rate * dt)

    if hemo:
        engine.hemo.anaphylaxis_severity = engine.anaphylaxis_severity
        engine.hemo.sepsis_severity = engine.sepsis_severity

    return effects


def step_tci(engine: "SimulationEngine", dt: float) -> None:
    """Update TCI controllers."""
    if dt <= 0:
        return
    sim_time = engine.state.time
    for tci_attr, rate_attr in TCI_TARGET_CONFIG:
        controller = getattr(engine, tci_attr)
        if not controller:
            continue

        sampling_time = max(controller.sampling_time, 1e-6)
        acc = engine._tci_accumulators.get(tci_attr, 0.0) + dt
        steps = int(acc / sampling_time)
        last_rate = getattr(engine, rate_attr)

        for i in range(steps):
            step_time = sim_time + (i + 1) * sampling_time
            last_rate = controller.step(controller.target, sim_time=step_time)

        if steps > 0:
            setattr(engine, rate_attr, last_rate)

        engine._tci_accumulators[tci_attr] = acc - steps * sampling_time


def step_machine(engine: "SimulationEngine", dt: float) -> tuple[float, float]:
    """Update machine state and return inspired volatile fractions."""
    state = engine.state
    circuit = engine.circuit
    composition = circuit.composition
    vaporizer = engine.vaporizer
    volatile_enabled = engine._volatile_enabled
    connected = state.airway_mode != AirwayType.NONE
    total_va = state.va if (connected and state.va > 0) else 0.0

    if volatile_enabled:
        vaporizer.step(dt, circuit.fgf_total())
    else:
        vaporizer.set_concentration(0.0)
    circuit.vaporizer_agent = vaporizer.state.agent
    circuit.vaporizer_setting = vaporizer.state.setting if volatile_enabled else 0.0
    circuit.vaporizer_on = vaporizer.state.is_on if volatile_enabled else False

    if not connected:
        fi_sevo = 0.0
        fi_n2o = 0.0
        fio2 = 0.21
    else:
        fi_vapor_circuit = composition.fi_agent
        fi_sevo = fi_vapor_circuit if volatile_enabled and engine.active_agent == "Sevoflurane" else 0.0
        fi_n2o = composition.fin2o
        fio2 = composition.fio2

    set_state_float_fields(
        state,
        fio2=fio2,
        fi_sevo=fi_sevo * 100.0,
        fi_n2o=fi_n2o * 100.0,
    )

    p_alv_prev = engine.pk_sevo.state.p_alv
    uptake_sevo = (fi_sevo - p_alv_prev) * total_va if connected else 0.0
    if connected:
        p_alv_prev_n2o = engine.pk_n2o.state.p_alv
        uptake_n2o = (fi_n2o - p_alv_prev_n2o) * total_va
    else:
        uptake_n2o = 0.0

    if connected:
        metabolic_factor = max(0.5, engine._metabolic_factor)
        vco2_ml_min = engine.resp.vco2 * metabolic_factor
        uptake_o2 = (vco2_ml_min / 1000.0) / max(engine.resp.rq, 0.1)
    else:
        uptake_o2 = 0.0

    circuit.step(dt, uptake_o2, uptake_sevo, uptake_n2o)
    return fi_sevo, fi_n2o


def step_pk(engine: "SimulationEngine", dt: float, fi_sevo: float, fi_n2o: float, co_curr: float) -> None:
    """Update pharmacokinetic models and synchronize their public state."""
    state = engine.state
    update_pk_hemodynamics(engine, co_curr)

    engine.pk_sevo.step(dt, fi_sevo, state.va, co_curr, temp_c=state.temp_c)
    et_sevo = engine.pk_sevo.state.p_alv * 100.0
    mac_sevo = engine.pk_sevo.state.mac

    engine.pk_n2o.step(dt, fi_n2o, state.va, co_curr, temp_c=state.temp_c)
    et_n2o = engine.pk_n2o.state.p_alv * 100.0
    mac_n2o = engine.pk_n2o.state.mac
    set_state_float_fields(
        state,
        et_sevo=et_sevo,
        mac_sevo=mac_sevo,
        et_n2o=et_n2o,
        mac_n2o=mac_n2o,
        mac=mac_sevo + mac_n2o,
    )

    engine.pk_prop.step(dt, engine.propofol_rate_mg_sec)
    engine.pk_remi.step(dt, engine.remi_rate_ug_sec)
    engine.pk_nore.step(dt, engine.nore_rate_ug_sec, propofol_conc_ug_ml=engine.pk_prop.state.c1)
    engine.pk_roc.step(dt, engine.roc_rate_mg_sec)
    engine.pk_epi.step(dt, engine.epi_rate_ug_sec)
    engine.pk_phenyl.step(dt, engine.phenyl_rate_ug_sec)
    engine.pk_vaso.step(dt, engine.vaso_rate_mu_sec)
    engine.pk_dobu.step(dt, engine.dobu_rate_ug_sec)
    engine.pk_mil.step(dt, engine.mil_rate_ug_sec)
    sync_pk_state(engine)


def update_pk_hemodynamics(engine: "SimulationEngine", co_curr: float) -> tuple[str, ...]:
    """Scale PK parameters based on current blood volume and cardiac output."""
    base_bv = engine.hemo.blood_volume_0
    base_co = engine.hemo.base_co_l_min
    if base_bv <= 0.0 or base_co <= 0.0:
        return ()
    v_ratio = clamp(engine.hemo.blood_volume / base_bv, 0.1, 2.0)
    co_ratio = clamp(co_curr / base_co, 0.1, 2.0)

    last_scale = engine._pk_hemo_scale_cache
    if last_scale is not None:
        last_v_ratio, last_co_ratio = last_scale
        if math.isclose(v_ratio, last_v_ratio, rel_tol=1e-4, abs_tol=1e-4) and math.isclose(
            co_ratio, last_co_ratio, rel_tol=1e-4, abs_tol=1e-4
        ):
            return ()

    updated = []
    for drug_key, attr in PK_HEMODYNAMIC_TARGETS:
        model = getattr(engine, attr)
        if hasattr(model, "update_hemodynamics"):
            model.update_hemodynamics(v_ratio, co_ratio)
            updated.append(drug_key)
    engine._pk_hemo_scale_cache = (v_ratio, co_ratio)
    return tuple(updated)


def update_airway_complications(engine: "SimulationEngine", dt: float) -> None:
    """Update airway obstruction/bronchospasm/laryngospasm state."""
    state = engine.state
    tol = clamp01(
        engine.tol_pd.compute_probability(
            state.propofol_ce,
            state.remi_ce,
            mac=state.mac,
        )
    )

    stim_profile = engine.disturbance_profile or ""
    stim_active = bool(
        engine.auto_laryngospasm_enabled and engine.disturbance_active and ("intubation" in stim_profile)
    )
    stim_scale = 1.0

    nmba_effect = hill_function(state.roc_ce, engine.resp.c50_nmba, engine.resp.gamma_nmba)
    muscle_factor = clamp01(1.0 - nmba_effect)

    airway_tuning = engine.airway_tuning
    laryng_target = 0.0
    if state.airway_mode != AirwayType.ETT and stim_active:
        light_factor = clamp01(1.0 - tol)
        laryng_target = clamp01(light_factor * muscle_factor * stim_scale)

    tau = airway_tuning.laryngospasm_tau_on if laryng_target > engine.laryngospasm_severity else airway_tuning.laryngospasm_tau_off
    if tau > 0:
        engine.laryngospasm_severity += (laryng_target - engine.laryngospasm_severity) * (dt / tau)
    engine.laryngospasm_severity = clamp01(engine.laryngospasm_severity)

    upper_obstruction = engine.airway_obstruction_manual
    if state.airway_mode != AirwayType.ETT:
        upper_obstruction = max(upper_obstruction, engine.laryngospasm_severity)
    upper_obstruction = clamp01(upper_obstruction)

    bronch = 1.0 - (1.0 - engine.bronchospasm_manual) * (1.0 - engine.anaphylaxis_severity)
    bronch = clamp01(bronch)

    base_r = engine._base_airway_resistance
    r_upper = airway_tuning.upper_resistance_gain * upper_obstruction
    r_bronch = airway_tuning.bronch_resistance_gain * bronch
    engine.resp_mech.resistance = base_r + r_upper + r_bronch

    engine._airway_patency = clamp(1.0 - upper_obstruction, 0.0, 1.0)
    engine._ventilation_efficiency = clamp(
        1.0
        - airway_tuning.vent_efficiency_bronch_weight * bronch
        - airway_tuning.vent_efficiency_upper_weight * upper_obstruction,
        airway_tuning.vent_efficiency_min,
        1.0,
    )
    engine._capno_obstruction = clamp01(
        airway_tuning.capno_obstruction_upper_weight * upper_obstruction
        + airway_tuning.capno_obstruction_bronch_weight * bronch
    )
    engine._vq_mismatch = clamp01(
        airway_tuning.vq_mismatch_bronch_weight * bronch
        + airway_tuning.vq_mismatch_upper_weight * upper_obstruction
    )

    engine._tol_current = tol
    state.airway_obstruction = upper_obstruction
    state.bronchospasm = bronch
    state.laryngospasm = engine.laryngospasm_severity


def step_physiology(engine: "SimulationEngine", dt: float, disturbances: DisturbanceEffects) -> PhysiologyStepState:
    """Advance physiology models and return a projected runtime snapshot."""
    state = engine.state
    update_airway_complications(engine, dt)

    connected = state.airway_mode in (AirwayType.ETT, AirwayType.MASK)
    vent_active = connected and engine.vent.is_on
    bag_mask_active = engine.bag_mask_active and connected and not vent_active
    assisted_active = vent_active or bag_mask_active

    mech_state, total_peep_effect, mech_rr_for_resp = step_mechanics(engine, dt, vent_active, bag_mask_active)

    alpha_paw = 1.0 - math.exp(-dt / max(engine._mean_paw_tau_s, 1e-6))
    if mech_state.paw_mean > 0:
        engine.current_mean_paw = (1 - alpha_paw) * engine.current_mean_paw + alpha_paw * mech_state.paw_mean
    else:
        engine.current_mean_paw = (1 - alpha_paw) * engine.current_mean_paw + alpha_paw * mech_state.paw

    pit_base = engine.hemo.pit_0
    paw_to_mmhg = 0.74
    paw_transmission = 0.54
    effort_transmission = 0.30
    pit_estimate = pit_base + paw_to_mmhg * paw_transmission * (engine.current_mean_paw - 5.0)
    effort_mmhg = engine._last_patient_effort_cmH2O * paw_to_mmhg * effort_transmission
    pit_estimate -= effort_mmhg

    mech_rr = mech_rr_for_resp if vent_active else 0.0
    delivered_vt_raw_l = engine.resp_mech.set_vt if vent_active else 0.0
    delivered_vt_display_l = 0.0
    if vent_active:
        delivered_vt_display_l = mech_state.delivered_vt / 1000.0 if mech_state.delivered_vt > 0 else delivered_vt_raw_l
        if engine.resp_mech.mode != VentMode.VCV and mech_state.delivered_vt > 0:
            delivered_vt_raw_l = delivered_vt_display_l
    mech_vent_mv = mech_rr * delivered_vt_raw_l if vent_active else 0.0

    bag_mask_mv = 0.0
    assisted_rr_for_resp = mech_rr
    assisted_vt_for_resp = delivered_vt_raw_l
    assisted_vt_effective = delivered_vt_display_l * engine._airway_patency
    if bag_mask_active:
        bag_mask_mv = engine.bag_mask_rr * engine.bag_mask_vt
        assisted_rr_for_resp = engine.bag_mask_rr
        assisted_vt_for_resp = engine.bag_mask_vt
        assisted_vt_effective = engine.bag_mask_vt * engine._airway_patency

    total_assisted_mv = mech_vent_mv + bag_mask_mv
    mac_sevo = state.mac_sevo
    kwargs = engine.get_resp_step_kwargs(
        total_assisted_mv=total_assisted_mv,
        peep=total_peep_effect,
        mean_paw=engine.current_mean_paw,
        mech_rr=assisted_rr_for_resp,
        mech_vt_l=assisted_vt_for_resp,
        cardiac_output=state.co,
        mac_sevo=mac_sevo,
    )
    resp_state = engine.resp.step(dt, **kwargs)

    spont_rr = resp_state.rr
    spont_vt_l = resp_state.vt / 1000.0
    if assisted_active:
        eff_rr = max(assisted_rr_for_resp, spont_rr)
        eff_vt = max(assisted_vt_effective, spont_vt_l)
        total_patient_mv = eff_rr * eff_vt
    else:
        total_patient_mv = spont_rr * spont_vt_l

    assisted_rr = engine.bag_mask_rr if bag_mask_active else mech_rr
    phase = mech_state.phase
    if not assisted_active:
        phase = phase_from_rr(engine, spont_rr)
    elif bag_mask_active and not vent_active:
        phase = phase_from_rr(engine, engine.bag_mask_rr)

    rr_display = max(assisted_rr, spont_rr) if assisted_active else spont_rr

    hemo_state = engine.hemo.step(
        dt,
        state.propofol_cp,
        state.remi_cp,
        state.nore_ce,
        pit=pit_estimate,
        paco2=resp_state.pa_co2,
        pao2=resp_state.p_arterial_o2,
        dist_hr=disturbances.hr,
        dist_sv=disturbances.sv,
        dist_svr=disturbances.svr,
        mac_sevo=mac_sevo,
        ce_epi=state.epi_ce,
        ce_phenyl=state.phenyl_ce,
        ce_vaso=state.vaso_ce,
        ce_dobu=state.dobu_ce,
        ce_mil=state.mil_ce,
        temp_c=state.temp_c,
        peep_cmH2O=total_peep_effect,
    )

    engine.vent.step(dt, mech_state, rr_total=mech_rr)

    vt_display_ml = resp_state.vt
    if assisted_active and mech_state.delivered_vt > 0:
        vt_display_ml = mech_state.delivered_vt
    elif vent_active:
        vt_display_ml = engine.resp_mech.set_vt * 1000.0
    elif bag_mask_active:
        vt_display_ml = engine.bag_mask_vt * 1000.0

    paw_display = mech_state.paw
    flow_display = mech_state.flow
    volume_display = mech_state.volume
    if not assisted_active and not resp_state.apnea and spont_rr > 0 and resp_state.vt > 0:
        vt_l = resp_state.vt / 1000.0
        cycle_time = 60.0 / max(spont_rr, 0.1)
        insp_fraction = 1.0 / 3.0
        insp_duration = cycle_time * insp_fraction
        exp_duration = max(1e-3, cycle_time - insp_duration)
        t_cycle = state.time % cycle_time
        comp = max(engine.resp_mech.compliance, 1e-3)

        if t_cycle < insp_duration:
            phase_frac = t_cycle / max(insp_duration, 1e-6)
            flow_l_s = (vt_l * math.pi / max(insp_duration, 1e-6)) * math.sin(math.pi * phase_frac)
            volume_l = 0.5 * vt_l * (1.0 - math.cos(math.pi * phase_frac))
        else:
            phase_frac = (t_cycle - insp_duration) / exp_duration
            flow_l_s = -(vt_l * math.pi / exp_duration) * math.sin(math.pi * phase_frac)
            volume_l = 0.5 * vt_l * (1.0 + math.cos(math.pi * phase_frac))

        paw_display = clamp((volume_l / comp) - 2.0, -10.0, 40.0)
        flow_display = flow_l_s * 60.0
        volume_display = volume_l

    return PhysiologyStepState(
        hemo_state=hemo_state,
        resp_state=resp_state,
        phase=phase,
        pit_estimate=pit_estimate,
        rr_display=rr_display,
        vt_display_ml=vt_display_ml,
        mv_display_l_min=total_patient_mv,
        paw_display=paw_display,
        flow_display=flow_display,
        volume_display=volume_display,
        vent_active=vent_active,
    )


def check_patient_viability(engine: "SimulationEngine", dt: float) -> None:
    """Check if patient vitals are compatible with life."""
    if not engine.config.enable_death_detector or engine.state.is_dead:
        return

    map_critical_low = 20.0
    hr_critical_low = 10.0
    hr_critical_high = 220.0

    raw_map = engine._raw_map
    raw_hr = engine._raw_hr

    if raw_map < map_critical_low:
        engine.time_hypotension += dt
    else:
        engine.time_hypotension = max(0, engine.time_hypotension - dt)

    if raw_hr < hr_critical_low:
        engine.time_brady += dt
    else:
        engine.time_brady = max(0, engine.time_brady - dt)

    if raw_hr >= hr_critical_high:
        engine.time_tachy += dt
    else:
        engine.time_tachy = max(0, engine.time_tachy - dt)

    if engine.time_hypotension > engine.DEATH_GRACE_PERIOD:
        engine.state.is_dead = True
        engine.state.death_reason = "Extreme Hypotension / Cardiac Arrest (MAP < 20 mmHg)"
        logger.warning("DEATH TRIGGERED: Hypotension (MAP=%.1f mmHg)", engine.state.map)
    elif engine.time_brady > engine.DEATH_GRACE_PERIOD:
        engine.state.is_dead = True
        engine.state.death_reason = "Asystole / Extreme Bradycardia (HR < 10 bpm)"
        logger.warning("DEATH TRIGGERED: Bradycardia (HR=%.1f bpm)", engine.state.hr)
    elif engine.time_tachy > engine.DEATH_GRACE_PERIOD:
        engine.state.is_dead = True
        engine.state.death_reason = "Extreme Tachycardia / VFib (HR ≥ 220 bpm)"
        logger.warning("DEATH TRIGGERED: Tachycardia (HR=%.1f bpm)", engine.state.hr)
