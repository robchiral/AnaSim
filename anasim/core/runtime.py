from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, cast

from anasim.core.constants import CIRCULATION_LIMIT_FRACTION, CMH2O_TO_MMHG
from anasim.core.enums import RhythmType
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.patient.pd.anesthesia import opioid_equivalent
from anasim.patient.pd.nmba import succinylcholine_twitch_fraction
from anasim.patient.pk_models import MidazolamPK, PropofolPKEleveld
from anasim.physiology.disturbances import DisturbanceEffects

from .monitors import step_monitors
from .projection import (
    PhysiologyStepState,
    build_snapshot_from_models,
    cardiovascular_response_fractions,
    central_muscle_strength,
    loss_of_response,
    project_runtime_physiology,
    step_respiration,
    stimulus_tolerance,
    sync_inhaled_agents,
    sync_inspired_gas,
    sync_pk_state,
)
from .state import AirwayStatus, AirwayType

if TYPE_CHECKING:
    from .engine import SimulationEngine

logger = logging.getLogger(__name__)

ARREST_HR_BPM = 10.0
# PEA: negligible forward flow and pulse pressure with an organized rhythm.
ARREST_CO_L_MIN = 0.3
ARREST_PULSE_PRESSURE_MMHG = 5.0
ARREST_CONFIRM_S = 15.0
# Mean airway pressure is smoothed before it sets intrathoracic pressure.
MEAN_PAW_TAU_S = 0.25


def step_simulation(engine: SimulationEngine, dt: float) -> None:
    state = engine.state
    engine.thermal.update_depth(state.temp_c, state.hypnotic_ce, state.mac)
    engine._tol_current = clamp01(stimulus_tolerance(engine))

    disturbance_complete = _disturbance_completes_during_step(engine, dt)
    disturbances = step_disturbances(engine, dt)
    # hemo.state includes a rhythm change made since the last step.
    cardiac_output = engine.hemo.state.co
    update_pk_covariates(engine, cardiac_output)

    step_tci(engine, dt)
    update_airway_complications(engine, dt)
    fi_sevo, fi_n2o = step_machine(engine, dt)
    toxicity = engine.toxicity
    lidocaine = engine.pk["lidocaine"]
    toxicity.step_lipid(dt, lidocaine.circulation)
    lidocaine.effect_input_fraction = toxicity.effect_input_fraction
    step_pk(engine, dt, fi_sevo, fi_n2o, cardiac_output)
    toxicity.step(
        dt, neural_ce=lidocaine.state.ce_neuro, cardiac_ce=lidocaine.state.ce_cv, ph=state.ph,
        propofol_ce=state.propofol_ce, midazolam_ce=state.midazolam_ce,
        mac_sevo=state.mac_sevo, motor_strength=state.twitch / 100.0,
        cortical_function=engine.cerebral.cortical_function,
    )
    engine.thermal.metabolic_factor *= toxicity.metabolic_factor
    mh = engine.malignant_hyperthermia
    mh.step(dt, sevo_mac=state.mac_sevo, succinylcholine_cp=engine.pk["succinylcholine"].state.c1,
            dantrolene_ce=state.dantrolene_ce)
    engine.thermal.metabolic_factor += mh.excess_metabolism
    state.muscle_rigidity = mh.rigidity
    state.seizure_activity = toxicity.seizure_activity
    state.convulsions = toxicity.convulsion_intensity > 0.2
    physiology = step_physiology(engine, dt, disturbances)
    project_runtime_physiology(engine, physiology)
    engine.cerebral.step(dt, state.map, state.hb_g_dl, state.sao2, state.pa_co2, state.ph)
    step_monitors(engine, dt, physiology.hemo_state, physiology.resp_state, disturbances)
    update_shivering(engine, dt)
    step_temperature(engine, dt)
    check_cardiac_arrest(engine, dt)
    if disturbance_complete and engine.disturbance_active:
        engine.stop_disturbance()
    if engine.intubation_start_time is not None and engine.intubation_stimulus.is_complete(
        state.time + dt - engine.intubation_start_time,
    ):
        engine.intubation_start_time = None


def step_mechanics(engine: SimulationEngine, dt: float, connected: bool, vent_active: bool,
                   bag_mask_active: bool) -> None:
    """Advance breathing through the workstation: ventilator, bag, or the patient's own effort."""
    resp = engine.resp.state
    lung = engine.resp_mech
    aeration = engine.aeration
    aeration.unconscious = engine.state.loc
    aeration.lung_water = engine.hemo.lung_water_ml_kg
    # Alveolar nitrogen and argon slow absorption collapse.
    aeration.inert_fraction = clamp01(
        1.0 - (resp.p_alveolar_o2 + resp.p_alveolar_co2) / engine.resp._atm_dry
        - engine.pk_n2o.state.p_alv - engine.pk_sevo.state.p_alv
    )
    aeration.absorbed_volume = engine.resp.frc - engine.resp.gas_volume_l
    lung.effort.unconscious = engine.state.loc
    lung.effort.muscle.muscle_strength = central_muscle_strength(engine)
    lung.effort.muscle.oxygen_fraction = engine.resp.oxygen_supply_fraction
    aeration.spontaneous_breathing = not resp.apnea and resp.vt > 100.0 and engine.airway_status.patency > 0.5
    # The patient's unassisted breathing sets inspiratory effort for the breaths that follow.
    lung.effort.set_drive(resp.effort_rr, resp.effort_vt / 1000.0)
    source: str | None
    if vent_active:
        source = "vent"
    elif bag_mask_active:
        source = "bag"
    else:
        source = "spontaneous" if connected else None
    engine.vent.step(dt, lung, source, bag=(engine.bag_mask_rr, engine.bag_mask_vt), collect_samples=True)


def update_shivering(engine: SimulationEngine, dt: float) -> None:
    """Update shivering intensity based on temperature and anesthetic state."""
    state = engine.state
    remi_effect = hill_function(state.opioid_ce, engine.resp.c50_remi, engine.resp.gamma_remi)
    # Shivering muscle is as sensitive to block as the adductor pollicis.
    muscle_factor = state.twitch / 100.0
    state.shivering = float(engine.thermal.step_shivering(dt, state.temp_c, state.bis, remi_effect, muscle_factor))


def step_temperature(engine: SimulationEngine, dt: float) -> None:
    """Update core temperature from metabolic heat, environmental loss, and redistribution."""
    state = engine.state
    state.temp_c = float(engine.thermal.step_temperature(
        dt, state.temp_c, state.bair_hugger_target, state.surface_cooling,
    ))


def stimulus_effects(engine: SimulationEngine, dt: float) -> DisturbanceEffects:
    """Drug-attenuated effects of surgical and intubation stimulation over one step."""
    if not engine.disturbance_active and engine.intubation_start_time is None:
        return DisturbanceEffects()
    active = []
    if engine.disturbance_active:
        active.append((engine.disturbances, engine.disturbance_start_time))
    if engine.intubation_start_time is not None:
        active.append((engine.intubation_stimulus, engine.intubation_start_time))
    bis = svr = sv = heart_rate = 0.0
    for disturbance, started in active:
        spec = disturbance.spec
        if spec is None:
            continue
        t_rel = max(0.0, engine.state.time - started)
        stimulus = disturbance.compute_average(t_rel, t_rel + dt)
        # Movement/EEG arousal and cardiovascular attenuation are separate endpoints.
        pressure, hr = cardiovascular_response_fractions(engine, spec.sympathetic_intensity)
        bis += stimulus.bis * (1.0 - engine._tol_current)
        svr += stimulus.svr * pressure + spec.tonic_svr(t_rel + 0.5 * dt, 1000.0 * engine.hemo.tpr)
        sv += stimulus.sv * pressure
        heart_rate += stimulus.hr * hr
    return DisturbanceEffects(bis=bis, svr=svr, sv=sv, hr=heart_rate)


def step_disturbances(engine: SimulationEngine, dt: float) -> DisturbanceEffects:
    """Calculate disturbances and update event-driven volume state."""
    effects = stimulus_effects(engine, dt)
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
                if infusion.label == "lipid":
                    engine.toxicity.add_lipid(amount_this_step)
                hemo.add_volume(
                    amount_this_step,
                    hematocrit=infusion.hematocrit,
                    retention_fraction=infusion.retention_fraction,
                    label=infusion.label,
                )
            if infusion.remaining_ml > 1e-3:
                remaining.append(infusion)
        engine.pending_infusions[:] = remaining

    if engine.lipid_infusion_ml_min > 0.0:
        amount = engine.lipid_infusion_ml_min * dt / 60.0
        hemo.add_volume(amount, retention_fraction=1.0, label="lipid")
        engine.toxicity.add_lipid(amount)

    if engine.maintenance_fluid_rate_ml_min > 0:
        rate_sec = engine.maintenance_fluid_rate_ml_min / 60.0
        hemo.add_volume(rate_sec * dt, hematocrit=0.0, label="crystalloid")

    hemo.anaphylaxis_severity = engine.anaphylaxis.step(dt)
    hemo.sepsis_severity = engine.sepsis.step(dt)

    return effects


def _disturbance_completes_during_step(engine: SimulationEngine, dt: float) -> bool:
    if not engine.disturbance_active:
        return False
    t_rel = max(0.0, engine.state.time - engine.disturbance_start_time)
    return engine.disturbances.is_complete(t_rel + dt)


def step_tci(engine: SimulationEngine, dt: float) -> None:
    """Advance TCI controllers on their own sampling clock."""
    sim_time = engine.state.time
    for key, controller in engine.tci.items():
        sampling_time = controller.sampling_time
        acc = engine._tci_accumulators.get(key, 0.0) + dt
        steps = int(acc / sampling_time)
        for i in range(steps):
            engine.infusion_rates[key] = controller.step(sim_time=sim_time + i * sampling_time)
        engine._tci_accumulators[key] = acc - steps * sampling_time


def step_machine(engine: SimulationEngine, dt: float) -> tuple[float, float]:
    """Update machine state and return inspired volatile fractions."""
    circuit = engine.circuit
    delivered_setting = engine.vaporizer.step(dt, circuit.fgf_total())
    circuit.vaporizer_setting = delivered_setting
    circuit.vaporizer_on = delivered_setting > 0.0

    fi_sevo, fi_n2o = sync_inspired_gas(engine)
    if engine.state.airway_mode == AirwayType.NONE:
        uptake_o2 = uptake_sevo = uptake_n2o = 0.0
    else:
        airway = engine.airway_status
        va = max(0.0, engine.state.va) if airway.patency > 0.0 else 0.0
        uptake_sevo = (fi_sevo - engine.pk_sevo.state.p_alv) * va
        uptake_n2o = (fi_n2o - engine.pk_n2o.state.p_alv) * va
        uptake_o2 = engine.resp.oxygen_exchange_l_min(
            engine.state.fio2, va,
            airway.patency, airway.vq_mismatch, dt,
        )
    circuit.step(dt, uptake_o2, uptake_sevo, uptake_n2o)
    return fi_sevo, fi_n2o


def step_pk(engine: SimulationEngine, dt: float, fi_sevo: float, fi_n2o: float, co_curr: float) -> None:
    """Update pharmacokinetic models and synchronize their public state."""
    state = engine.state
    aeration = engine.aeration
    va = state.va if engine.airway_status.patency > 0.0 else 0.0
    engine.pk_sevo.step(dt, fi_sevo, va, co_curr, temp_c=state.temp_c,
                        lung_volume_l=aeration.frc, shunt_fraction=aeration.shunt_fraction)
    engine.pk_n2o.step(dt, fi_n2o, va, co_curr, temp_c=state.temp_c,
                       lung_volume_l=aeration.frc, shunt_fraction=aeration.shunt_fraction)
    sync_inhaled_agents(engine)

    rates = engine.infusion_rates
    for models in (engine.pk, engine.pump_pk):
        for key, model in models.items():
            model.step(dt, rates.get(key, 0.0))
    # Plasma binding and sugammadex-dependent effect-site efflux reduce free
    # rocuronium, which drives TOF and every muscle effect.
    engine.tof_pd.step_recovery(
        dt, engine.pk["roc"], mac_sevo=engine.pk_sevo.state.mac, mac_n2o=engine.pk_n2o.state.mac
    )
    sync_pk_state(engine)


def update_pk_covariates(engine: SimulationEngine, co_curr: float) -> None:
    """Update drug-interaction and hemodynamic PK covariates.

    Propofol slows norepinephrine and midazolam disposition; midazolam slows
    propofol clearance. Hemodynamic covariates are blood volume, cardiac
    output, and renal perfusion.
    """
    propofol = cast(PropofolPKEleveld, engine.pk["propofol"])
    midazolam = cast(MidazolamPK, engine.pk["midazolam"])
    engine.pk_nore.update_propofol(propofol.state.c1)
    # Both covariates use the concentrations before either parameter update.
    propofol_cp = propofol.state.c1
    midazolam_cp = midazolam.state.c1
    propofol.update_midazolam(midazolam_cp)
    midazolam.update_propofol(propofol_cp)
    base_bv = engine.hemo.blood_volume_0
    blood_lost = engine.hemo.total_blood_out_ml - engine._pk_blood_out_ml
    if blood_lost > 0.0:
        engine.toxicity.lipid_ml *= math.exp(-blood_lost / base_bv)
        for model in (*engine.pk.values(), engine.tof_pd.sugammadex, engine.tof_pd.complex):
            model.remove_blood(blood_lost / 1000.0, base_bv / 1000.0)
        engine._pk_blood_out_ml = engine.hemo.total_blood_out_ml
    # Heart failure's low output slows flow-dependent clearance. Pregnancy's
    # added output mostly perfuses the uterus and kidneys, so it is the reference.
    base_co = engine.hemo.base_co_l_min / engine.hemo.failure.output_factor
    if base_bv <= 0.0 or base_co <= 0.0:
        return
    # Below 30% of resting output, as in arrest and CPR, flow limits drug transport.
    circulation = clamp01(co_curr / (CIRCULATION_LIMIT_FRACTION * base_co))
    if circulation != propofol.circulation:
        for model in (*engine.pk.values(), engine.tof_pd.sugammadex, engine.tof_pd.complex):
            model.circulation = circulation
    v_ratio = clamp(engine.hemo.blood_volume / base_bv, 0.1, 2.0)
    co_ratio = clamp(co_curr / base_co, 0.1, 2.0)
    scale = (v_ratio, co_ratio, engine.hemo.renal_perfusion)

    last_scale = engine._pk_hemo_scale_cache
    # Changes below 0.1% leave concentrations unchanged at display precision.
    if last_scale is not None and all(
        math.isclose(value, last, rel_tol=1e-3, abs_tol=1e-3) for value, last in zip(scale, last_scale, strict=True)
    ):
        return

    for model in engine.pk.values():
        model.update_hemodynamics(*scale, blood_volume_l=base_bv / 1000.0)
    engine.tof_pd.update_hemodynamics(*scale, blood_volume_l=base_bv / 1000.0)
    engine._pk_hemo_scale_cache = scale


def update_airway_complications(engine: SimulationEngine, dt: float) -> None:
    """Update airway obstruction/bronchospasm/laryngospasm state."""
    state = engine.state
    tol = engine._tol_current
    # Laryngoscopy can provoke laryngospasm; the sympathetic response outlasts it.
    stim_active = (
        engine.auto_laryngospasm_enabled
        and engine.disturbance_active
        and engine.disturbances.stimulates_airway(state.time - engine.disturbance_start_time)
    )

    depolarizing_strength = succinylcholine_twitch_fraction(state.succinylcholine_ce, central=True)
    muscle_factor = central_muscle_strength(engine)

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
        # Mask CPAP or positive-pressure breaths hold the pharynx open. Mean
        # pressure over recent breaths includes any unassisted breaths between them.
        splint_pressure = 0.0
        if state.airway_mode == AirwayType.MASK:
            vent = engine.vent
            if vent.is_on:
                splint_pressure = vent.settings.peep
                if vent.settings.mode != "CPAP":
                    splint_pressure = max(splint_pressure, vent.monitors.paw_mean_recent)
            elif engine.bag_mask_active:
                splint_pressure = vent.monitors.paw_mean_recent
        relief = clamp01(splint_pressure / airway_tuning.collapse_relief_pressure)
        unconscious = loss_of_response(engine, ketamine=False)
        tone = max(0.1, engine.tof_pd.pharyngeal_tone_fraction(state.mac_sevo, state.mac_n2o) * depolarizing_strength)
        collapse = clamp01(airway_tuning.unsupported_collapse_max * unconscious / tone) * (1.0 - relief)
        upper_obstruction = max(upper_obstruction, engine.laryngospasm_severity, collapse)
    upper_obstruction = clamp01(upper_obstruction)

    bronch = 1.0 - (1.0 - engine.bronchospasm_manual) * (1.0 - engine.anaphylaxis.severity)
    # Epinephrine relieves bronchospasm but not upper-airway obstruction.
    bronchodilation = airway_tuning.epi_bronchodilation_max * hill_function(
        state.epi_ce, airway_tuning.epi_bronchodilation_c50, 1.0
    )
    bronch *= 1.0 - bronchodilation
    bronch = clamp01(bronch)

    base_r = engine._base_airway_resistance
    # Resistance diverges as the airway closes. The finite denominator keeps
    # the exact mechanics solver conditioned at complete occlusion.
    r_upper = airway_tuning.upper_resistance_gain * upper_obstruction / max(1e-6, 1.0 - upper_obstruction)
    r_bronch = airway_tuning.bronch_resistance_gain * bronch
    engine.resp_mech.reference_resistance = base_r
    engine.resp_mech.resistance = base_r + r_upper + r_bronch
    engine.resp_mech.bronchospasm = bronch
    engine.resp_mech.bronch_resistance = r_bronch

    engine.airway_status = AirwayStatus(
        patency=clamp(1.0 - upper_obstruction, 0.0, 1.0),
        ventilation_efficiency=clamp(
            1.0
            - airway_tuning.vent_efficiency_bronch_weight * bronch
            - airway_tuning.vent_efficiency_upper_weight * upper_obstruction,
            airway_tuning.vent_efficiency_min,
            1.0,
        ),
        capno_obstruction=clamp01(
            airway_tuning.capno_obstruction_upper_weight * upper_obstruction
            + airway_tuning.capno_obstruction_bronch_weight * engine.resp_mech.expiratory_obstruction
        ),
        vq_mismatch=clamp01(
            airway_tuning.vq_mismatch_bronch_weight * bronch
            + airway_tuning.vq_mismatch_upper_weight * upper_obstruction
        ),
    )

    state.airway_obstruction = upper_obstruction
    state.bronchospasm = bronch
    state.laryngospasm = engine.laryngospasm_severity


def step_physiology(engine: SimulationEngine, dt: float, disturbances: DisturbanceEffects) -> PhysiologyStepState:
    """Advance physiology models and return a projected runtime snapshot."""
    state = engine.state
    connected = state.airway_mode != AirwayType.NONE
    vent_active = connected and engine.vent.is_on
    bag_mask_active = engine.bag_mask_active and connected and not vent_active
    step_mechanics(engine, dt, connected, vent_active, bag_mask_active)
    vent = engine.vent
    spirometry = vent.monitors

    alpha_paw = 1.0 - math.exp(-dt / MEAN_PAW_TAU_S)
    mean_paw = vent.airway_pressure_area / dt
    engine.current_mean_paw = (1 - alpha_paw) * engine.current_mean_paw + alpha_paw * mean_paw

    hemo_cfg = engine.hemo.config
    effort_pressure = min(vent.muscle_pressure_area / dt, hemo_cfg.effort_pressure_max)
    pit_estimate = hemo_cfg.pit_0 + CMH2O_TO_MMHG * (
        hemo_cfg.paw_transmission * engine.current_mean_paw - hemo_cfg.effort_transmission * effort_pressure
    )

    if vent_active:
        total_peep_effect = engine.resp_mech.peep + spirometry.auto_peep
    else:
        total_peep_effect = spirometry.auto_peep if bag_mask_active else 0.0
    # Sevoflurane cardiovascular effects follow end-tidal MAC through the model's own ke0.
    mac_sevo = engine.pk_sevo.state.p_alv * 100.0 / engine.pk_sevo.mac_age
    resp_state = step_respiration(engine, dt, state.co)
    step_lactate(engine, dt)

    hemo = engine.hemo
    # Without flow, arterial pressure equals venous pressure, so coronary flow stops.
    hemo.coronary.step(
        dt, state.hr, state.sbp, state.dbp, max(state.lap, hemo.state.venous_pressure), hemo.hb_conc,
        resp_state.sao2, hemo.myocardial_hypoxia,
        fibrillating=hemo.rhythm_type == RhythmType.VFIB, compressions=hemo.compressions,
    )

    # Plasma remifentanil and propofol reach hemodynamic effect sites inside the model;
    # slower opioids enter at their own effect sites.
    hemo_state = engine.hemo.step(
        dt,
        state.propofol_cp,
        opioid_equivalent(state.remi_cp, state.fentanyl_ce, state.morphine_ce, state.hydromorphone_ce),
        state.nore_ce,
        pit=pit_estimate,
        paco2=resp_state.pa_co2,
        pao2=resp_state.p_arterial_o2,
        dist_hr=disturbances.hr + engine.malignant_hyperthermia.heart_rate_increase,
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
        ce_atropine=state.atropine_ce,
        ce_ephedrine=state.ephedrine_ce,
        ce_amiodarone=state.amiodarone_ce,
        sodium_block=engine.toxicity.conduction_block,
        toxic_contractility_loss=engine.toxicity.contractility_loss,
        ce_ketamine=engine.pk["ketamine"].state.ce_cv,
        ce_ketamine_hr=engine.pk["ketamine"].state.ce_hr,
        temp_c=state.temp_c,
        peep_cmH2O=total_peep_effect,
        sao2=resp_state.sao2,
        remifentanil_cp=state.remi_cp,
        ph=resp_state.ph,
    )
    hypoxia, energy_failure, silent = myocardial_substrate(engine)
    rhythm = engine.arrhythmias.step(
        dt, hemo.rhythm_type, ischemia=hemo.coronary.ischemia, hypoxia=hypoxia,
        energy_failure=energy_failure, electrically_silent=silent,
        perfusing=hemo_state.intrinsic_co > ARREST_CO_L_MIN,
        lidocaine=engine.pk["lidocaine"].state.ce_cv, amiodarone=state.amiodarone_ce,
    )
    if rhythm != hemo.rhythm_type:
        hemo.rhythm_type = rhythm
        hemo_state = hemo.state

    return build_snapshot_from_models(
        engine, hemo_state, resp_state, pit_estimate=pit_estimate,
    )


def myocardial_substrate(engine: SimulationEngine) -> tuple[float, float, bool]:
    """Myocardial hypoxia, energy failure, and electrical silence, including lidocaine toxicity."""
    coronary, toxicity = engine.hemo.coronary, engine.toxicity
    return (max(engine.hemo.myocardial_hypoxia, toxicity.cardiac_stress),
            max(coronary.energy_failure, toxicity.conduction_block),
            coronary.electrically_silent or toxicity.electrically_silent)


def step_lactate(engine: SimulationEngine, dt: float) -> None:
    """Advance lactate from this step's O2 balance and update extracellular buffer base."""
    if dt <= 0.0:
        return
    resp = engine.resp
    base_removed = engine.lactate.step(
        dt, resp.oxygen_deficit_l, resp.oxygen_supply_fraction,
        epinephrine_ng_ml=engine.state.epi_ce,
        sepsis_severity=engine.hemo.sepsis_severity,
        crystalloid_in_ml=engine.hemo.total_crystalloid_in_ml,
    )
    resp.oxygen_deficit_l = 0.0
    resp.remove_buffer_base(base_removed)


def check_cardiac_arrest(engine: SimulationEngine, dt: float) -> None:
    """Flag arrest after 15 s without effective pulses from the heart itself."""
    state = engine.state
    rhythm = engine.hemo.state.rhythm_type
    # Compressions mask the heart's own pulse pressure.
    no_pulse = state.chest_compressions or state.sbp - state.dbp < ARREST_PULSE_PRESSURE_MMHG
    pulseless = (rhythm in (RhythmType.VFIB, RhythmType.ASYSTOLE)
                 or state.hr < ARREST_HR_BPM
                 or (state.intrinsic_co < ARREST_CO_L_MIN and no_pulse))
    engine._pulseless_s = engine._pulseless_s + dt if pulseless else 0.0
    if not pulseless:
        state.cardiac_arrest = False
        state.arrest_reason = ""
        return
    if engine._pulseless_s < ARREST_CONFIRM_S:
        return

    if rhythm in (RhythmType.VFIB, RhythmType.ASYSTOLE):
        reason = rhythm.value
    elif state.hr < ARREST_HR_BPM:
        reason = f"Extreme bradycardia (HR < {ARREST_HR_BPM:g} bpm)"
    else:
        reason = "Pulseless electrical activity (negligible forward flow)"
    changed = not state.cardiac_arrest or state.arrest_reason != reason
    state.cardiac_arrest = True
    state.arrest_reason = reason
    if changed:
        logger.warning("Cardiac arrest: %s (MAP=%.1f mmHg, HR=%.1f bpm)", reason, state.map, state.hr)
