from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from anasim.core.constants import (
    RR_APNEA_THRESHOLD,
    SHIVER_BASE_THRESHOLD,
    SHIVER_BIS_FULL,
    SHIVER_BIS_ON,
    SHIVER_DELTA_FULL,
    SHIVER_DEPTH_DROP_MAX,
    SHIVER_MAX_MULTIPLIER,
    SHIVER_REMI_DROP_MAX,
    SHIVER_TAU_OFF,
    SHIVER_TAU_ON,
    TEMP_METABOLIC_COEFFICIENT,
)
from anasim.core.drug_registry import PK_HEMODYNAMIC_TARGETS, TCI_TARGET_CONFIG
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.resp_mech import VentMode

from .monitors import phase_from_rr, step_monitors
from .projection import (
    PhysiologyStepState,
    project_runtime_physiology,
    set_state_float_fields,
    sync_inhaled_agents,
    sync_inspired_gas,
    sync_pk_state,
)
from .state import AirwayType

if TYPE_CHECKING:
    from .engine import SimulationEngine

logger = logging.getLogger(__name__)

# Circle-system resistance seen at the Y-piece during spontaneous breathing (cmH2O/(L/s)).
SPONTANEOUS_CIRCUIT_RESISTANCE = 2.0


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
    tuning = engine.thermal_tuning
    depth_index = mac + prop_ce / tuning.depth_propofol_scale
    metabolic_factor = TEMP_METABOLIC_COEFFICIENT ** (37.0 - temp_c)
    metabolic_factor *= 1.0 - tuning.metabolic_reduction_max * clamp01(depth_index)
    metabolic_factor = max(0.5, metabolic_factor) * (1.0 + SHIVER_MAX_MULTIPLIER * shiver_level)
    return depth_index, metabolic_factor


def redistribution_target_j(engine: "SimulationEngine", depth_index: float) -> float:
    """Core heat moved to the periphery at steady anesthetic depth."""
    tuning = engine.thermal_tuning
    heat_capacity = engine.patient.weight * engine.specific_heat
    return tuning.redistribution_core_drop_c * clamp01(depth_index) * heat_capacity


def step_simulation(engine: "SimulationEngine", dt: float) -> None:
    """Advance the simulation by one step."""
    state = engine.state
    engine._depth_index, engine._metabolic_factor = compute_depth_metabolic_context(
        engine,
        state.temp_c,
        state.propofol_ce,
        state.mac,
        shiver_level=engine._shiver_level,
    )
    engine._tol_current = clamp01(
        engine.tol_pd.compute_probability(state.propofol_ce, state.remi_ce, mac=state.mac)
    )

    disturbance_complete = _disturbance_completes_during_step(engine, dt)
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
    if disturbance_complete and engine.disturbance_active:
        engine.stop_disturbance()


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
    depth_factor = clamp01(engine._depth_index)
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
    """Update core temperature from metabolic heat, environmental loss, and redistribution."""
    state = engine.state
    temp_c = state.temp_c
    tuning = engine.thermal_tuning
    depth_factor = clamp01(engine._depth_index)

    production = engine.heat_production_basal * max(0.5, engine._metabolic_factor)
    conductance = (
        tuning.base_conductance_w_per_c
        * (1.0 + tuning.anesthetic_conductance_gain * depth_factor)
        * (engine.surface_area / 1.9)
    )
    heat_loss = conductance * (temp_c - tuning.ambient_temp_c)

    # Anesthetic vasodilation moves core heat to the periphery over the first
    # hour (Matsukawa 1995). Vasoconstriction on lightening traps it peripherally,
    # so the core deficit does not reverse.
    deficit_gap = redistribution_target_j(engine, engine._depth_index) - engine._redistributed_heat_j
    redistribution_w = max(0.0, deficit_gap) / tuning.redistribution_tau_s
    engine._redistributed_heat_j += redistribution_w * dt

    warming = 0.0
    if state.bair_hugger_target > 0:
        warming = tuning.bair_hugger_gain_w_per_c * max(0.0, state.bair_hugger_target - temp_c)

    net_heat_w = production + warming - heat_loss - redistribution_w
    heat_capacity = engine.patient.weight * engine.specific_heat
    set_state_float_fields(
        state,
        temp_c=clamp(temp_c + net_heat_w * dt / heat_capacity, tuning.temp_min_c, tuning.temp_max_c),
    )


def step_disturbances(engine: "SimulationEngine", dt: float) -> DisturbanceEffects:
    """Calculate disturbances and update event-driven volume state."""
    state = engine.state
    if not engine.disturbance_active or not engine.disturbances:
        effects = zero_disturbance()
    else:
        t_rel = max(0.0, state.time - engine.disturbance_start_time)
        effects = engine.disturbances.compute_average(t_rel, t_rel + dt)

    # Hypnotics and especially opioids blunt the response to noxious stimulation;
    # scale it by the probability of responding to laryngoscopy (Bouillon 2004).
    stim_gain = 1.0 - engine._tol_current
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
                )
            if infusion.remaining_ml > 1e-3:
                remaining.append(infusion)
        engine.pending_infusions[:] = remaining

    if hemo and engine.maintenance_fluid_rate_ml_min > 0:
        rate_sec = engine.maintenance_fluid_rate_ml_min / 60.0
        hemo.add_volume(rate_sec * dt, hematocrit=0.0, label="crystalloid")

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


def _disturbance_completes_during_step(engine: "SimulationEngine", dt: float) -> bool:
    if not engine.disturbance_active or not engine.disturbances:
        return False
    t_rel = max(0.0, engine.state.time - engine.disturbance_start_time)
    return engine.disturbances.is_complete(t_rel + dt)


def step_tci(engine: "SimulationEngine", dt: float) -> None:
    """Advance TCI controllers on their own sampling clock."""
    sim_time = engine.state.time
    for tci_attr, rate_attr in TCI_TARGET_CONFIG:
        controller = getattr(engine, tci_attr)
        if not controller:
            continue

        sampling_time = controller.sampling_time
        acc = engine._tci_accumulators.get(tci_attr, 0.0) + dt
        steps = int(acc / sampling_time)
        for i in range(steps):
            setattr(engine, rate_attr, controller.step(sim_time=sim_time + i * sampling_time))
        engine._tci_accumulators[tci_attr] = acc - steps * sampling_time


def step_machine(engine: "SimulationEngine", dt: float) -> tuple[float, float]:
    """Update machine state and return inspired volatile fractions."""
    circuit = engine.circuit
    vaporizer = engine.vaporizer
    if engine._volatile_enabled:
        vaporizer.step(dt, circuit.fgf_total())
    else:
        vaporizer.set_concentration(0.0)
    circuit.vaporizer_agent = vaporizer.state.agent
    circuit.vaporizer_setting = vaporizer.state.setting
    circuit.vaporizer_on = vaporizer.state.is_on

    fi_sevo, fi_n2o = sync_inspired_gas(engine)
    if engine.state.airway_mode == AirwayType.NONE:
        uptake_o2 = uptake_sevo = uptake_n2o = 0.0
    else:
        va = max(0.0, engine.state.va)
        uptake_sevo = (fi_sevo - engine.pk_sevo.state.p_alv) * va
        uptake_n2o = (fi_n2o - engine.pk_n2o.state.p_alv) * va
        uptake_o2 = engine.resp.vco2 * max(0.5, engine._metabolic_factor) / engine.resp.rq / 1000.0
    circuit.step(dt, uptake_o2, uptake_sevo, uptake_n2o)
    return fi_sevo, fi_n2o


def step_pk(engine: "SimulationEngine", dt: float, fi_sevo: float, fi_n2o: float, co_curr: float) -> None:
    """Update pharmacokinetic models and synchronize their public state."""
    state = engine.state
    engine.pk_sevo.step(dt, fi_sevo, state.va, co_curr, temp_c=state.temp_c)
    engine.pk_n2o.step(dt, fi_n2o, state.va, co_curr, temp_c=state.temp_c)
    sync_inhaled_agents(engine)

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

    for _, attr in PK_HEMODYNAMIC_TARGETS:
        getattr(engine, attr).update_hemodynamics(v_ratio, co_ratio)
    engine._pk_hemo_scale_cache = (v_ratio, co_ratio)
    return tuple(drug_key for drug_key, _ in PK_HEMODYNAMIC_TARGETS)


def update_airway_complications(engine: "SimulationEngine", dt: float) -> None:
    """Update airway obstruction/bronchospasm/laryngospasm state."""
    state = engine.state
    tol = engine._tol_current
    stim_profile = engine.disturbance_profile or ""
    stim_active = bool(
        engine.auto_laryngospasm_enabled and engine.disturbance_active and ("intubation" in stim_profile)
    )

    nmba_effect = hill_function(state.roc_ce, engine.resp.c50_nmba, engine.resp.gamma_nmba)
    muscle_factor = clamp01(1.0 - nmba_effect)

    airway_tuning = engine.airway_tuning
    laryng_target = 0.0
    if state.airway_mode != AirwayType.ETT and stim_active:
        light_factor = clamp01(1.0 - tol)
        laryng_target = clamp01(light_factor * muscle_factor)

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
    # Sevoflurane cardiovascular effects follow end-tidal MAC through the model's own ke0.
    mac_sevo = engine.pk_sevo.state.p_alv * 100.0 / engine.pk_sevo.mac_age
    kwargs = engine.get_resp_step_kwargs(
        total_assisted_mv=total_assisted_mv,
        peep=total_peep_effect,
        mean_paw=engine.current_mean_paw,
        mech_rr=assisted_rr_for_resp,
        mech_vt_l=assisted_vt_for_resp,
        cardiac_output=state.co,
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
    if not assisted_active:
        # Spontaneous breathing: the machine sensors see flow only through a
        # connected circuit, and inspiration draws Y-piece pressure slightly negative.
        paw_display = flow_display = volume_display = 0.0
        if connected and not resp_state.apnea and spont_rr > 0 and resp_state.vt > 0:
            flow_l_s, volume_display = _spontaneous_breath(state.time, spont_rr, resp_state.vt / 1000.0)
            paw_display = -SPONTANEOUS_CIRCUIT_RESISTANCE * flow_l_s
            flow_display = flow_l_s * 60.0

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


def _spontaneous_breath(time_s: float, rr: float, vt_l: float) -> tuple[float, float]:
    """Return sinusoidal (flow L/s, volume L) for a spontaneous breath with I:E 1:2."""
    cycle_time = 60.0 / max(rr, 0.1)
    insp_duration = cycle_time / 3.0
    exp_duration = cycle_time - insp_duration
    t_cycle = time_s % cycle_time
    if t_cycle < insp_duration:
        phase = math.pi * t_cycle / insp_duration
        return 0.5 * vt_l * math.pi / insp_duration * math.sin(phase), 0.5 * vt_l * (1.0 - math.cos(phase))
    phase = math.pi * (t_cycle - insp_duration) / exp_duration
    return -0.5 * vt_l * math.pi / exp_duration * math.sin(phase), 0.5 * vt_l * (1.0 + math.cos(phase))


def check_patient_viability(engine: "SimulationEngine", dt: float) -> None:
    """Check if patient vitals are compatible with life."""
    if not engine.config.enable_death_detector or engine.state.is_dead:
        return

    map_critical_low = 20.0
    hr_critical_low = 10.0
    hr_critical_high = 220.0

    raw_map = engine.state.map
    raw_hr = engine.state.hr

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
