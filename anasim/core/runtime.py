from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from anasim.core.constants import (
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
from anasim.core.enums import RhythmType
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.physiology.disturbances import DisturbanceEffects

from .monitors import step_monitors
from .projection import (
    PhysiologyStepState,
    circuit_ventilation,
    measured_ventilation,
    project_runtime_physiology,
    sync_inhaled_agents,
    sync_inspired_gas,
    sync_pk_state,
)
from .state import AirwayType

if TYPE_CHECKING:
    from .engine import SimulationEngine

logger = logging.getLogger(__name__)

# Cardiac arrest endpoint: no effective circulation for ARREST_CONFIRM_S seconds.
ARREST_MAP_MMHG = 20.0
ARREST_HR_BPM = 10.0
ARREST_CONFIRM_S = 15.0


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
    state = engine.state
    engine._depth_index, engine._metabolic_factor = compute_depth_metabolic_context(
        engine,
        state.temp_c,
        state.hypnotic_ce,
        state.mac,
        shiver_level=engine._shiver_level,
    )
    engine._tol_current = clamp01(
        engine.tol_pd.compute_probability(
            state.hypnotic_ce,
            state.opioid_ce,
            mac=state.mac,
            ce_ketamine=state.ketamine_ce,
            ce_lidocaine=state.lidocaine_ce,
        )
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
    step_monitors(engine, dt, physiology.hemo_state, physiology.resp_state, disturbances)
    update_shivering(engine, dt)
    step_temperature(engine, dt)
    check_cardiac_arrest(engine, dt)
    if disturbance_complete and engine.disturbance_active:
        engine.stop_disturbance()


def step_mechanics(engine: "SimulationEngine", dt: float, connected: bool, vent_active: bool,
                   bag_mask_active: bool) -> None:
    """Advance breathing through the workstation: ventilator, bag, or the patient's own effort."""
    resp = engine.resp.state
    lung = engine.resp_mech
    lung.aeration.unconscious = engine.state.loc
    lung.aeration.lung_water = engine.hemo.lung_water_ml_kg
    lung.effort.unconscious = engine.state.loc
    lung.aeration.spontaneous_breathing = not resp.apnea and resp.vt > 100.0 and engine._airway_patency > 0.5
    # The patient's unassisted breathing sets inspiratory effort for the breaths that follow.
    lung.effort.set_drive(0.0 if resp.apnea else resp.rr, resp.vt / 1000.0)
    if vent_active:
        source = "vent"
    elif bag_mask_active:
        source = "bag"
    else:
        source = "spontaneous" if connected else None
    engine.vent.step(dt, lung, source, bag=(engine.bag_mask_rr, engine.bag_mask_vt), collect_samples=True)
    engine._last_patient_effort_cmH2O = min(lung.effort.amplitude, 20.0)


def update_shivering(engine: "SimulationEngine", dt: float) -> float:
    """Update shivering intensity based on temperature and anesthetic state."""
    state = engine.state
    depth_factor = clamp01(engine._depth_index)
    remi_effect = hill_function(state.opioid_ce, engine.resp.c50_remi, engine.resp.gamma_remi)

    threshold = SHIVER_BASE_THRESHOLD - SHIVER_DEPTH_DROP_MAX * depth_factor - SHIVER_REMI_DROP_MAX * remi_effect
    temp_deficit = max(0.0, threshold - state.temp_c)
    cold_drive = clamp01(temp_deficit / SHIVER_DELTA_FULL)

    emergence = clamp01((state.bis - SHIVER_BIS_ON) / (SHIVER_BIS_FULL - SHIVER_BIS_ON))

    # Shivering uses peripheral muscle, as sensitive as the adductor pollicis.
    muscle_factor = 1.0 - engine.tof_pd.twitch_block(state.roc_ce, state.mac_sevo, state.mac_n2o)

    target = cold_drive * emergence * muscle_factor
    tau = SHIVER_TAU_ON if target > engine._shiver_level else SHIVER_TAU_OFF
    engine._shiver_level += (target - engine._shiver_level) * (dt / tau)
    engine._shiver_level = clamp01(engine._shiver_level)
    state.shivering = float(engine._shiver_level)
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
    state.temp_c = float(clamp(temp_c + net_heat_w * dt / heat_capacity, tuning.temp_min_c, tuning.temp_max_c))


def step_disturbances(engine: "SimulationEngine", dt: float) -> DisturbanceEffects:
    """Calculate disturbances and update event-driven volume state."""
    state = engine.state
    if not engine.disturbance_active:
        effects = DisturbanceEffects()
    else:
        t_rel = max(0.0, state.time - engine.disturbance_start_time)
        effects = engine.disturbances.compute_average(t_rel, t_rel + dt)

    # Hypnotics, lidocaine, and especially opioids blunt the response to noxious
    # stimulation; scale it by the probability of responding to laryngoscopy
    # (Bouillon 2004).
    stim_gain = 1.0 - engine._tol_current
    effects = DisturbanceEffects(
        bis=effects.bis * stim_gain,
        svr=effects.svr * stim_gain,
        sv=effects.sv * stim_gain,
        hr=effects.hr * stim_gain,
    )

    hemo = engine.hemo
    if engine.active_hemorrhage:
        rate_sec = engine.hemorrhage_rate_ml_min / 60.0
        hemo.add_volume(-rate_sec * dt)

    if engine.pending_infusions:
        remaining = []
        for infusion in engine.pending_infusions:
            rate_sec = infusion.rate_ml_min / 60.0
            amount_this_step = min(infusion.remaining_ml, rate_sec * dt)
            infusion.remaining_ml -= amount_this_step
            if amount_this_step > 0:
                hemo.add_volume(
                    amount_this_step,
                    hematocrit=infusion.hematocrit,
                    retention_fraction=infusion.retention_fraction,
                    label=infusion.label,
                )
            if infusion.remaining_ml > 1e-3:
                remaining.append(infusion)
        engine.pending_infusions[:] = remaining

    if engine.maintenance_fluid_rate_ml_min > 0:
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

    hemo.anaphylaxis_severity = engine.anaphylaxis_severity
    hemo.sepsis_severity = engine.sepsis_severity

    return effects


def _disturbance_completes_during_step(engine: "SimulationEngine", dt: float) -> bool:
    if not engine.disturbance_active:
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
    aeration = engine.resp_mech.aeration
    engine.pk_sevo.step(dt, fi_sevo, state.va, co_curr, temp_c=state.temp_c,
                        lung_volume_l=aeration.frc, shunt_fraction=aeration.shunt_fraction)
    engine.pk_n2o.step(dt, fi_n2o, state.va, co_curr, temp_c=state.temp_c,
                       lung_volume_l=aeration.frc, shunt_fraction=aeration.shunt_fraction)
    sync_inhaled_agents(engine)

    engine.pk_prop.step(dt, engine.propofol_rate_mg_sec)
    engine.pk_remi.step(dt, engine.remi_rate_ug_sec)
    engine.pk_fentanyl.step(dt, engine.fentanyl_rate_ug_sec)
    engine.pk_midazolam.step(dt, engine.midazolam_rate_ug_sec)
    engine.pk_etomidate.step(dt, 0.0)
    engine.pk_ketamine.step(dt, engine.ketamine_rate_mg_sec)
    engine.pk_lidocaine.step(dt, engine.lidocaine_rate_mg_sec)
    engine.pk_nore.step(dt, engine.nore_rate_ug_sec, propofol_conc_ug_ml=engine.pk_prop.state.c1)
    engine.pk_roc.step(dt, engine.roc_rate_mg_sec)
    # Free (sugammadex-unbound) rocuronium at the neuromuscular junction drives
    # TOF and every muscle effect.
    tof = engine.tof_pd.step_recovery(
        dt, engine.pk_roc.state.c1, mac_sevo=engine.pk_sevo.state.mac, mac_n2o=engine.pk_n2o.state.mac
    )
    state.tof = float(tof)
    engine.pk_epi.step(dt, engine.epi_rate_ug_sec)
    engine.pk_phenyl.step(dt, engine.phenyl_rate_ug_sec)
    engine.pk_vaso.step(dt, engine.vaso_rate_mu_sec)
    engine.pk_dobu.step(dt, engine.dobu_rate_ug_sec)
    engine.pk_mil.step(dt, engine.mil_rate_ug_sec)
    engine.pk_esmolol.step(dt, engine.esmolol_rate_mg_sec)
    engine.pk_labetalol.step(dt, 0.0)
    engine.pk_glyco.step(dt, 0.0)
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

    nmba_effect = hill_function(engine.tof_pd.ce_central, engine.resp.c50_nmba, engine.resp.gamma_nmba)
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
        # Mask CPAP or positive-pressure breaths hold the pharynx open.
        splint_pressure = 0.0
        if state.airway_mode == AirwayType.MASK:
            vent = engine.vent
            if vent.is_on:
                splint_pressure = vent.settings.peep
                if vent.settings.mode != "CPAP":
                    splint_pressure = max(splint_pressure, vent.monitors.paw_mean)
            elif engine.bag_mask_active:
                splint_pressure = vent.monitors.paw_mean
        relief = clamp01(splint_pressure / airway_tuning.collapse_relief_pressure)
        # Pharyngeal collapse excludes ketamine, which preserves airway tone.
        unconscious = engine.loc_pd.compute_probability(
            state.hypnotic_ce, state.opioid_ce, mac_sevo=state.mac_sevo, mac_n2o=state.mac_n2o
        )
        collapse = airway_tuning.unsupported_collapse_max * unconscious * (1.0 - relief)
        upper_obstruction = max(upper_obstruction, engine.laryngospasm_severity, collapse)
    upper_obstruction = clamp01(upper_obstruction)

    bronch = 1.0 - (1.0 - engine.bronchospasm_manual) * (1.0 - engine.anaphylaxis_severity)
    # Epinephrine relieves bronchospasm but not upper-airway obstruction.
    bronchodilation = airway_tuning.epi_bronchodilation_max * hill_function(
        state.epi_ce, airway_tuning.epi_bronchodilation_c50, 1.0
    )
    bronch *= 1.0 - bronchodilation
    bronch = clamp01(bronch)

    base_r = engine._base_airway_resistance
    r_upper = airway_tuning.upper_resistance_gain * upper_obstruction
    r_bronch = airway_tuning.bronch_resistance_gain * bronch
    engine.resp_mech.resistance = base_r + r_upper + r_bronch
    engine.resp_mech.bronchospasm = bronch
    engine.resp_mech.bronch_resistance = r_bronch

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

    step_mechanics(engine, dt, connected, vent_active, bag_mask_active)
    vent = engine.vent
    spirometry = vent.monitors

    alpha_paw = 1.0 - math.exp(-dt / max(engine._mean_paw_tau_s, 1e-6))
    mean_paw = spirometry.paw_mean if assisted_active else 0.0
    engine.current_mean_paw = (1 - alpha_paw) * engine.current_mean_paw + alpha_paw * mean_paw

    pit_base = engine.hemo.pit_0
    paw_to_mmhg = 0.74
    paw_transmission = 0.54
    effort_transmission = 0.30
    pit_estimate = pit_base + paw_to_mmhg * paw_transmission * (engine.current_mean_paw - 5.0)
    effort_mmhg = engine._last_patient_effort_cmH2O * paw_to_mmhg * effort_transmission
    pit_estimate -= effort_mmhg

    # Gas exchange uses recent exhaled breaths, so it lags at least one breath.
    assisted_rr, assisted_vt_l = circuit_ventilation(engine) if connected else (0.0, 0.0)
    if vent_active:
        total_peep_effect = vent.settings.peep + spirometry.auto_peep
    else:
        total_peep_effect = spirometry.auto_peep if bag_mask_active else 0.0
    # Sevoflurane cardiovascular effects follow end-tidal MAC through the model's own ke0.
    mac_sevo = engine.pk_sevo.state.p_alv * 100.0 / engine.pk_sevo.mac_age
    kwargs = engine.get_resp_step_kwargs(
        total_assisted_mv=assisted_rr * assisted_vt_l,
        mech_rr=assisted_rr,
        mech_vt_l=assisted_vt_l,
        cardiac_output=state.co,
    )
    resp_state = engine.resp.step(dt, **kwargs)

    rr_display, vt_display_ml, total_patient_mv = measured_ventilation(
        engine, connected, resp_state.rr, resp_state.vt / 1000.0
    )

    hemo_state = engine.hemo.step(
        dt,
        state.propofol_cp,
        state.opioid_cp,
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
        ce_esmolol=state.esmolol_ce,
        ce_labetalol=state.labetalol_ce,
        ce_glyco=state.glyco_ce,
        ce_ketamine=state.ketamine_ce,
        temp_c=state.temp_c,
        peep_cmH2O=total_peep_effect,
        sao2=resp_state.sao2,
    )

    # The machine's sensors see only gas that crosses the Y-piece.
    paw_display, flow_display, volume_display = (vent.paw, vent.flow, vent.volume) if connected else (0.0, 0.0, 0.0)
    return PhysiologyStepState(
        hemo_state=hemo_state,
        resp_state=resp_state,
        pit_estimate=pit_estimate,
        rr_display=rr_display,
        vt_display_ml=vt_display_ml,
        mv_display_l_min=total_patient_mv,
        paw_display=paw_display,
        flow_display=flow_display,
        volume_display=volume_display,
        paw_peak=spirometry.paw_peak if connected else 0.0,
        paw_plat=spirometry.paw_plat if assisted_active else math.nan,
        paw_mean=spirometry.paw_mean if connected else 0.0,
        peep=spirometry.peep if assisted_active else 0.0,
        compliance_dyn=spirometry.compliance_dyn if assisted_active else math.nan,
    )


def check_cardiac_arrest(engine: "SimulationEngine", dt: float) -> None:
    """End the session after 15 s without effective circulation (MAP < 20 or HR < 10).

    Resuscitation is not modeled, so a confirmed arrest is a session endpoint.
    """
    state = engine.state
    if not engine.config.end_on_cardiac_arrest or state.cardiac_arrest:
        return
    pulseless = state.map < ARREST_MAP_MMHG or state.hr < ARREST_HR_BPM
    engine._pulseless_s = engine._pulseless_s + dt if pulseless else 0.0
    if engine._pulseless_s < ARREST_CONFIRM_S:
        return

    rhythm = engine.hemo.rhythm_type
    if rhythm in (RhythmType.VFIB, RhythmType.ASYSTOLE):
        reason = rhythm.value
    elif state.hr < ARREST_HR_BPM:
        reason = f"Extreme bradycardia (HR < {ARREST_HR_BPM:g} bpm)"
    else:
        reason = f"Pulseless electrical activity (MAP < {ARREST_MAP_MMHG:g} mmHg)"
    state.cardiac_arrest = True
    state.arrest_reason = reason
    logger.warning("Cardiac arrest: %s (MAP=%.1f mmHg, HR=%.1f bpm)", reason, state.map, state.hr)
