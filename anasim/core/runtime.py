from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, cast

from anasim.core.constants import (
    ARREST_CO_L_MIN,
    ARREST_CONFIRM_S,
    ARREST_HR_BPM,
    ARREST_MAP_MMHG,
    ARREST_PULSE_PRESSURE_MMHG,
    BLOOD_VOLUME_MIN,
    O2_FLUSH_FLOW_L_MIN,
    PULSELESS_VT_MAP_MMHG,
    VT_MIN,
)
from anasim.core.enums import PULSELESS_RHYTHMS, RhythmType
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.extracorporeal.oxygenator import recirculation_fraction
from anasim.extracorporeal.types import BloodGasManagement
from anasim.machine.ventilator import BAG_TI_MAX_S, VOLUME_TARGETED
from anasim.monitors.arterial import MIN_PULSE_PRESSURE_MMHG
from anasim.patient.pd.anesthesia import opioid_equivalent
from anasim.patient.pd.nmba import (
    succinylcholine_laryngeal_spasm_factor,
    succinylcholine_phase2_tof_fraction,
    succinylcholine_twitch_fraction,
)
from anasim.patient.pk_models import (
    HeparinPK,
    MidazolamPK,
    PropofolPKEleveld,
    ProtaminePK,
    _EpiduralAbsorptionPK,
)
from anasim.physiology.airway_crisis import ESOPHAGEAL_RETURN_FRACTION
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.toxicity import SILENT_BLOCK

from .monitors import step_monitors, waveform_substeps
from .projection import (
    PhysiologyStepState,
    StepGasCache,
    _succinylcholine_phase2_exposure,
    advance_tracheal_gas,
    assisted_positive_pressure,
    bis_hypnotic_components,
    build_snapshot_from_models,
    cardiovascular_response_fractions,
    central_muscle_strength,
    circuit_connected,
    circuit_inspired_mixture,
    compute_pit_estimates,
    effective_mask_seal,
    effective_midazolam_ce,
    extubation_tolerance,
    is_esophageal_intubation,
    is_infraglottic_airway,
    loss_of_response,
    mask_circuit_share,
    mask_leak,
    mask_seal_factor,
    naloxone_dose_ratio,
    project_hemodynamics,
    project_runtime_physiology,
    pulse_ox_target_spo2,
    reference_arterial_pressures,
    step_coronary_from_state,
    step_respiration,
    stimulus_tolerance,
    succinylcholine_central_ce,
    sync_inhaled_agents,
    sync_inspired_gas,
    sync_pk_state,
    thermal_hypnotic_ce,
    thermal_mac,
)
from .state import AirwayStatus, AirwayType

if TYPE_CHECKING:
    from .engine import SimulationEngine

logger = logging.getLogger(__name__)

# Mean airway pressure is smoothed before it sets intrathoracic pressure, while
# spontaneous inspiratory effort is breath-averaged across step sizes.
MEAN_PAW_TAU_S = 0.25
MEAN_EFFORT_TAU_S = 1.5
# Typical one-handed bag squeeze pressure that can vent surplus FGF through the
# APL valve; APL settings above 20 cmH2O (or FGF exceeding squeeze rate) trap
# gas and raise baseline PEEP.
BAG_VENT_PRESSURE_CMH2O = 20.0


def _step_monitors_light(
    engine: SimulationEngine,
    dt: float,
    hemo_state,
    resp_state,
    disturbances: DisturbanceEffects,
) -> None:
    """Fast scalar monitor update for hidden scenario prerolls without 100 Hz waveforms."""
    state = engine.state
    state.respiratory_fatigue = float(engine.resp_mech.effort.muscle.fatigue)
    prop_bis, adjunct_bis = bis_hypnotic_components(engine)
    bis_val = engine.bis.step(
        dt,
        prop_bis,
        mac_sevo=engine.pk_sevo.state.mac,
        cortical_function=engine.cerebral.cortical_function * (1.0 - 0.55 * engine.neuraxial.unconsciousness),
        ce_ketamine=state.ketamine_ce,
        ce_opioid=state.opioid_ce,
        temp_c=state.temp_c,
        disturbance_bis=disturbances.bis,
        ce_adjunct=adjunct_bis,
    )
    state.bis = float(clamp(bis_val, 0.0, 100.0))
    engine.smooth_bis = state.bis
    engine._bis_noise = 0.0
    state.display_bis = state.bis
    state.loc = float(loss_of_response(engine))
    engine._tol_current = stimulus_tolerance(engine)
    state.tol = float(engine._tol_current)
    defib = engine.defibrillator
    if defib.waiting_for_r and dt > 0.0:
        substeps = waveform_substeps(dt)
        sub_dt = dt / substeps
        start_time = state.time
        conduction_failure = engine.hemo.coronary.energy_failure
        for index in range(substeps):
            sample = engine.cardiac_cycle.step(
                sub_dt, hemo_state.hr, hemo_state.rhythm_type, conduction_failure,
            )
            if engine.step_defibrillator(start_time + (index + 1) * sub_dt, sample):
                hemo_state = engine.hemo.state
                project_hemodynamics(engine, hemo_state)
                engine.cardiac_cycle.seed(hemo_state.hr, hemo_state.rhythm_type)
                break
            if not defib.waiting_for_r:
                break
    state.display_hr = 0.0 if hemo_state.rhythm_type in PULSELESS_RHYTHMS else float(max(0.0, hemo_state.hr))
    sbp, dbp = reference_arterial_pressures(engine, hemo_state)
    state.sbp = float(sbp)
    state.dbp = float(dbp)
    state.art_map = float(hemo_state.map)
    state.art_sbp = float(sbp)
    state.art_dbp = float(dbp)
    connected = circuit_connected(engine)
    if is_esophageal_intubation(engine):
        state.display_etco2 = float(engine.airway_crisis.esophageal_co2_mmhg)
        state.etco2_signal_valid = engine.airway_crisis.esophageal_co2_mmhg > 0.5
    else:
        exhaling_breaths = (
            connected
            and not (
                resp_state.apnea
                and engine.vent.monitors.rr_total <= 0.0
                and not engine.vent.inspiring
                and not engine.vent.manual_breath_active
            )
        )
        state.display_etco2 = float(resp_state.etco2) if exhaling_breaths else 0.0
        state.etco2_signal_valid = exhaling_breaths
    ecmo = engine.extracorporeal
    pump_override = hemo_state.pump_flow > 0.0 or (
        ecmo.is_arterial_support and (ecmo.state.aortic_cross_clamp or ecmo.state.cardioplegia_arrest)
    )
    pp_light = sbp - dbp
    state.spo2_signal_valid = (
        hemo_state.rhythm_type not in PULSELESS_RHYTHMS
        and hemo_state.intrinsic_co > ARREST_CO_L_MIN
        and math.isfinite(pp_light)
        and pp_light >= MIN_PULSE_PRESSURE_MMHG
        and not (hemo_state.compressions and not pump_override)
    )
    spo2_src = pulse_ox_target_spo2(engine)
    state.spo2 = state.display_spo2 = float(spo2_src) if state.spo2_signal_valid else 0.0


def step_simulation(engine: SimulationEngine, dt: float, *, update_monitors: bool = True) -> None:
    if dt <= 0.0:
        return
    if dt > 0.25 and (
        engine.pk["succinylcholine"].state.c1 > 0.05
        or engine._manual_breath_requested
        or engine.vent.manual_breath_active
        or (engine.resp.state.apnea and engine.state.sao2 < 75.0)
    ):
        n_sub = max(2, math.ceil(dt / 0.25))
        sub_dt = dt / n_sub
        base_time = engine.state.time
        for idx in range(n_sub):
            engine.state.time = round(base_time + idx * sub_dt, 9)
            _step_simulation_once(engine, sub_dt, update_monitors=update_monitors)
        engine.state.time = base_time
        return
    _step_simulation_once(engine, dt, update_monitors=update_monitors)


def _step_simulation_once(engine: SimulationEngine, dt: float, *, update_monitors: bool = True) -> None:
    if dt <= 0.0:
        return
    state = engine.state
    update_shivering(engine, dt)
    hemo_pre = engine.hemo.state
    cardiac_output = hemo_pre.co
    pre_circ = engine.hemo.circulation_fraction(cardiac_output)
    engine.neuraxial.step(dt, circulation=pre_circ)
    nx = engine.neuraxial
    if nx.epidural_catheter and nx.epidural_infusion_ml_hr > 0.0 and pre_circ > 0.0:
        epi_pk = engine.pk.get(nx.epidural_infusion_drug)
        if isinstance(epi_pk, _EpiduralAbsorptionPK):
            epi_dose_mg = nx.epidural_infusion_ml_hr * nx.epidural_infusion_conc_pct * 10.0 * (dt / 3600.0)
            epi_pk.add_epidural_dose(epi_dose_mg, with_epinephrine=False)
    if engine.magnesium_infusion_g_hr > 0.0:
        mg_mmol = engine.magnesium_infusion_g_hr * 4.05729 * (dt / 3600.0)
        engine.hemo.fluids.give_magnesium_sulfate(mg_mmol, circulation=pre_circ, bolus_push_s=0.0)
    engine.thermal.update_depth(state.temp_c, thermal_hypnotic_ce(engine), thermal_mac(engine))
    if engine.disturbance_active or engine.intubation_start_time is not None:
        engine._tol_current = stimulus_tolerance(engine)

    disturbance_complete = _disturbance_completes_during_step(engine, dt)
    # hemo.state includes a rhythm change made since the last step.
    engine.hemo.fluids.sync_circulation(pre_circ, ph=engine.resp.state.ph_37)
    engine.glucose.sync_circulation(pre_circ, engine.hemo.fluids)
    disturbances = step_disturbances(engine, dt)
    update_pk_covariates(engine, cardiac_output, dt=dt)

    step_tci(engine, dt)
    engine.airway_crisis.step(dt)
    update_airway_complications(engine, dt, rhythm_type=hemo_pre.rhythm_type)
    fi_sevo, fi_n2o = step_machine(engine, dt, co_now=cardiac_output)
    toxicity = engine.toxicity
    lidocaine = engine.pk["lidocaine"]
    bupivacaine = engine.pk["bupivacaine"]
    ropivacaine = engine.pk["ropivacaine"]
    toxicity.step_lipid(dt, lidocaine.circulation)
    art_ph_pre = engine.resp.state.ph_37
    lidocaine.effect_input_fraction = toxicity.effect_input_fraction
    bupivacaine.effect_input_fraction = toxicity.effect_input_fraction_for("bupivacaine", ph=art_ph_pre)
    ropivacaine.effect_input_fraction = toxicity.effect_input_fraction_for("ropivacaine", ph=art_ph_pre)
    step_pk(engine, dt, fi_sevo, fi_n2o, cardiac_output)
    circ = engine.hemo.circulation_fraction(cardiac_output)
    art_ph_37 = engine.resp.state.ph_37
    ven_share = clamp01(circ / 0.10) * (1.0 - circ)
    tox_ph = art_ph_37 * (1.0 - ven_share) + min(art_ph_37, engine.resp.state.venous_ph_37) * ven_share
    midaz_eff = effective_midazolam_ce(engine)
    toxicity.step(
        dt, neural_ce=lidocaine.state.ce_neuro, cardiac_ce=lidocaine.state.ce_cv,
        ph=tox_ph,
        propofol_ce=state.propofol_ce, midazolam_ce=midaz_eff,
        mac_sevo=state.mac_sevo, motor_strength=state.twitch / 100.0,
        cortical_function=engine.cerebral.cortical_function,
        mac_n2o=state.mac_n2o,
        etomidate_ce=state.etomidate_ce,
        sodium=engine.hemo.fluids.arterial_sodium,
        bupivacaine_neuro_ce=bupivacaine.state.ce_neuro,
        bupivacaine_cv_ce=bupivacaine.state.ce_cv,
        ropivacaine_neuro_ce=ropivacaine.state.ce_neuro,
        ropivacaine_cv_ce=ropivacaine.state.ce_cv,
        hr=hemo_pre.hr,
        epi_ce=state.epi_ce,
        vaso_ce=state.vaso_ce,
        flumazenil_ce=state.flumazenil_ce,
    )
    opioid_raw = opioid_equivalent(
        engine.pk["remi"].state.ce, state.fentanyl_ce, state.morphine_ce, state.hydromorphone_ce,
    )
    engine.opioid_toxicity.step(
        dt,
        fentanyl_ce=state.fentanyl_ce,
        remi_ce=engine.pk["remi"].state.ce,
        opioid_ce_raw=opioid_raw,
        naloxone_ce=state.naloxone_ce,
        muscle_strength=central_muscle_strength(engine),
        propofol_ce=state.propofol_ce,
        midazolam_ce=midaz_eff,
        mac_sevo=state.mac_sevo,
        surgical_stress=1.0 if engine.disturbance_active else 0.0,
        paco2=state.pa_co2,
    )
    engine.methemoglobin.step(
        dt,
        methylene_blue_ce=state.methylene_blue_ce,
        circulation=circ,
    )
    engine.thermal.metabolic_factor *= toxicity.metabolic_factor
    mh = engine.malignant_hyperthermia
    sux_muscle_cp = max(state.succinylcholine_ce, engine.pk["succinylcholine"].state.c1 * circ)
    mh.step(dt, sevo_mac=state.mac_sevo, succinylcholine_cp=sux_muscle_cp,
            dantrolene_ce=state.dantrolene_ce, oxygen_supply=engine.resp.oxygen_supply_fraction)
    sepsis_sev = engine.sepsis.severity
    aerobic_heat_gate = clamp01(engine.resp.oxygen_supply_fraction)
    engine.thermal.metabolic_factor += (
        mh.excess_metabolism + 0.25 * sepsis_sev + engine.opioid_toxicity.metabolic_factor_add
    )
    engine.thermal.extra_heat_w = engine.hemo.fluids.lean_mass_kg * (
        mh.extra_heat_w_per_kg + 0.35 * sepsis_sev * aerobic_heat_gate
    )
    ptx_connected = circuit_connected(engine) and not is_esophageal_intubation(engine)
    vent_ppv = engine.vent.is_on and (
        engine.vent.settings.mode != "CPAP" or engine.vent.settings.peep > 1.0
    )
    apl_cpap = not engine.vent.is_on and (engine._apl_pressure > 1.0 or engine.resp_mech.peep > 1.0)
    ptx_ppv = ptx_connected and (vent_ppv or assisted_positive_pressure(engine) or apl_cpap)
    spont_breathing = (
        not engine.resp.state.apnea
        and engine.resp.state.vt > VT_MIN
        and not is_esophageal_intubation(engine)
    )
    engine.pneumothorax.step(
        dt,
        ppv_active=ptx_ppv,
        spontaneous_vent_active=spont_breathing,
        mean_paw_cmh2o=engine.current_mean_paw if ptx_connected else 0.0,
        peep_cmh2o=engine.resp_mech.peep if ptx_connected else 0.0,
        p_alv_n2o=engine.pk_n2o.state.p_alv,
        circulation=circ,
    )
    alv_o2_frac = clamp01(engine.resp.state.p_alveolar_o2 / 713.0)
    pulm_circ = engine.hemo.circulation_fraction(hemo_pre.pulmonary_flow)
    engine.embolism.step(
        dt,
        p_alv_n2o=engine.pk_n2o.state.p_alv,
        alveolar_o2_fraction=alv_o2_frac,
        position=engine.hemo.position,
        heparin_u_ml=state.heparin_u_ml,
        act_s=state.act_s,
        circulation=circ,
        pulmonary_circulation=pulm_circ,
        pulmonary_ventilation=engine.state.va > 0.1,
    )
    engine.pneumoperitoneum.step(dt, circulation=circ)
    engine.aeration.chest_wall_stiffness = (
        mh.chest_wall_stiffness
        + engine.pneumothorax.chest_wall_stiffness_add
        + engine.pneumoperitoneum.chest_wall_stiffness_add
        + engine.opioid_toxicity.chest_wall_stiffness_add
    )
    engine.aeration.pleural_collapse = engine.pneumothorax.lung_collapse_fraction
    engine.aeration.crisis_shunt = 1.0 - (
        (1.0 - engine.pneumothorax.shunt_fraction) * (1.0 - engine.embolism.extra_shunt)
    )
    state.opioid_rigidity = float(engine.opioid_toxicity.rigidity)
    state.muscle_rigidity = mh.rigidity or engine.opioid_toxicity.rigidity >= 0.30
    state.seizure_activity = toxicity.seizure_activity
    state.convulsions = toxicity.convulsion_intensity > 0.2
    physiology = step_physiology(engine, dt, disturbances, collect_samples=update_monitors)
    project_runtime_physiology(engine, physiology)
    hemo_state = physiology.hemo_state
    ecmo = engine.extracorporeal
    pump_override = hemo_state.pump_flow > 0.0 or (
        ecmo.is_arterial_support and (ecmo.state.aortic_cross_clamp or ecmo.state.cardioplegia_arrest)
    )
    if hemo_state.compressions and not pump_override and hemo_state.co > 0.0:
        cpr_share = clamp01((hemo_state.co - hemo_state.intrinsic_co) / hemo_state.co)
        intrinsic_art_map = (
            engine.hemo.lumen_map(hemo_state.intrinsic_map, hemo_state.venous_pressure)
            if hemo_state.intrinsic_co > 0.0
            else hemo_state.venous_pressure
        )
        cpr_driving_map = hemo_state.venous_pressure + hemo_state.cpr_perfusion_map
        cerebral_map = (1.0 - cpr_share) * intrinsic_art_map + cpr_share * cpr_driving_map
        base_co_ref = engine.hemo.base_co_l_min
        # Closed-chest CPR delivers ~20-34% of normal CBF: enough to protect against
        # rapid anoxic injury (Astrup 1981 >20% threshold) without restoring cortical
        # wakefulness (>35% electrical threshold).
        cpr_cbf_cap = clamp(1.35 * (hemo_state.co / max(0.1, base_co_ref)), 0.0, 0.34)
        flow_ceiling: float | None = 1.0 - cpr_share * (1.0 - cpr_cbf_cap)
    else:
        cerebral_map = hemo_state.perfusion_map
        flow_ceiling = None
    ecmo = engine.extracorporeal
    cerebral_sao2 = ecmo.state.upper_body_sao2 if ecmo.is_arterial_support else state.sao2
    cerebral_paco2 = ecmo.state.upper_body_paco2_37 if ecmo.is_arterial_support else physiology.resp_state.pa_co2_37
    cerebral_ph = ecmo.state.upper_body_ph_37 if ecmo.is_arterial_support else physiology.resp_state.ph_37
    ph_stat_active = ecmo.is_cpb and ecmo.state.blood_gas_strategy is BloodGasManagement.PH_STAT
    cerebral_hb = float(
        engine.hemo.hb_conc if engine.hemo.fluids._circulation >= 0.99 else engine.resp._arterial_hb
    )
    engine.cerebral.step(
        dt,
        cerebral_map,
        cerebral_hb,
        cerebral_sao2,
        cerebral_paco2,
        cerebral_ph,
        venous_pressure=hemo_state.venous_pressure,
        flow_ceiling=flow_ceiling,
        temp_c=state.temp_c,
        anesthetic_depth=clamp01(engine.thermal.depth_index),
        metabolic_stress=clamp01(state.seizure_activity),
        volatile_luxury=clamp01(state.mac_sevo + 0.5 * state.mac_n2o),
        ph_stat=ph_stat_active,
        glucose_mg_dl=engine.glucose.arterial_glucose_mg_dl,
    )
    if update_monitors:
        step_monitors(engine, dt, physiology.hemo_state, physiology.resp_state, disturbances)
    else:
        _step_monitors_light(engine, dt, physiology.hemo_state, physiology.resp_state, disturbances)
    step_temperature(engine, dt)
    check_cardiac_arrest(engine, dt)
    if disturbance_complete and engine.disturbance_active:
        engine.stop_disturbance()
    if engine.intubation_start_time is not None and engine.intubation_stimulus.is_complete(
        state.time + dt - engine.intubation_start_time,
    ):
        engine.intubation_start_time = None


def _manual_breath_delta_p(lung, vt_l: float, ti_s: float) -> float:
    """Peak airway pressure swing (cmH2O above PEEP) for a half-sine manual squeeze."""
    if vt_l <= 0.0 or ti_s <= 1e-6:
        return 0.0
    c = max(0.005, lung.compliance)
    r = max(0.0, lung.resistance)
    w = math.pi / ti_s
    elastance = 1.0 / c
    tau = max(1e-3, lung.viscoelastic_tau)
    rate = 1.0 / tau
    gain = lung.viscoelastic_ratio * elastance * 0.5 * w / (rate * rate + w * w)
    theta = math.pi - math.atan(r * c * w)
    peak = 0.0
    for t in (0.5 * ti_s, theta / w, 0.85 * ti_s, ti_s):
        wt = w * t
        flow = 0.5 * w * math.sin(wt)
        vol = 0.5 * (1.0 - math.cos(wt))
        p2 = gain * (rate * math.sin(wt) - w * math.cos(wt) + w * math.exp(-rate * t))
        paw = r * flow + elastance * vol + p2
        if paw > peak:
            peak = paw
    return vt_l * peak


def _advance_apl_pressure(
    pressure: float,
    dt: float,
    inflow_l_min: float,
    c_sys: float,
    target_ceiling: float,
    *,
    peep_vent: float | None = None,
    vent_max_l_min: float = 0.0,
) -> float:
    """Advance bag/APL baseline pressure over dt with optional squeeze venting above peep_vent."""
    r_in = (inflow_l_min / 60.0) / c_sys
    if peep_vent is None or vent_max_l_min <= 0.0:
        return clamp(pressure + r_in * dt, 0.0, target_ceiling)
    r_vent = (vent_max_l_min / 60.0) / c_sys
    rem = dt
    p = pressure
    if p < peep_vent - 1e-12:
        if r_in <= 0.0:
            return clamp(p + r_in * rem, 0.0, target_ceiling)
        t_hit = (peep_vent - p) / r_in
        if rem <= t_hit:
            return clamp(p + r_in * rem, 0.0, target_ceiling)
        p = peep_vent
        rem -= t_hit
    if p > peep_vent + 1e-12:
        r_net = r_in - r_vent
        if r_net >= 0.0:
            return clamp(p + r_net * rem, 0.0, target_ceiling)
        t_hit = (peep_vent - p) / r_net
        if rem <= t_hit:
            return clamp(p + r_net * rem, 0.0, target_ceiling)
        p = peep_vent
        rem -= t_hit
    if r_in > r_vent:
        p += (r_in - r_vent) * rem
    elif r_in < 0.0:
        p += r_in * rem
    return clamp(p, 0.0, target_ceiling)


def step_mechanics(engine: SimulationEngine, dt: float, connected: bool, vent_active: bool,
                   bag_mask_active: bool, *, collect_samples: bool = True) -> None:
    """Advance breathing through the workstation: ventilator, bag, or the patient's own effort."""
    resp = engine.resp.state
    lung = engine.resp_mech
    aeration = engine.aeration
    seal_scale = mask_seal_factor(engine)
    aeration.pressure_transmission = seal_scale if connected else 0.0
    aeration.unconscious = engine.state.loc
    aeration.bronchospasm = lung.bronchospasm
    aeration.lung_water = engine.hemo.lung_water_ml_kg
    # Closed-airway O2 absorption collapse shrinks alveolar gas volume and drives absorption shunt.
    aeration.collapse_deficit = clamp01(
        engine.resp._apneic_gas_deficit_l / max(0.5, aeration.reference_volume)
    )
    # Alveolar nitrogen and argon slow absorption collapse.
    aeration.inert_fraction = clamp01(engine.resp.alveolar_fn2)
    absorbed_btps = engine.resp.consume_mechanical_absorption_l()
    bellows_empty = engine.circuit.bellows_fraction <= 0.05
    if absorbed_btps > 0.0 and (not assisted_positive_pressure(engine) or bellows_empty):
        engine.vent.absorb_lung_gas(absorbed_btps, lung)
    lung.effort.unconscious = engine.state.loc
    lung.effort.muscle.muscle_strength = (
        central_muscle_strength(engine)
        * engine.malignant_hyperthermia.respiratory_excursion_fraction
        * engine.neuraxial.respiratory_muscle_strength
    )
    lung.effort.muscle.oxygen_fraction = engine.resp.oxygen_supply_fraction
    aeration.spontaneous_breathing = not resp.apnea and resp.vt > VT_MIN and engine.airway_status.patency > 0.5
    aeration.glottic_intact = not is_infraglottic_airway(engine)
    # The patient's unassisted neural demand sets inspiratory effort; muscle strength
    # and fatigue cap mechanical pressure inside PatientEffort._begin().
    lung.effort.set_drive(resp.neural_rr, resp.neural_vt / 1000.0)
    if is_esophageal_intubation(engine):
        # Ventilator/bag cycles into the esophagus/stomach; patient lungs receive no circuit pressure.
        aeration.pressure_transmission = 0.0
        aeration.glottic_intact = True
        lung.update_aeration(dt, 0.0, 0.0, cpr_active=engine.hemo.compressions)
        lung.effort.start(lung, 0.0)
        rem_effort = dt
        while rem_effort > 1e-12:
            step_e = min(rem_effort, max(1e-3, lung.effort.time_to_break()))
            lung.effort.advance(step_e, lung, 0.0)
            rem_effort -= step_e
        if lung.volume != 0.0:
            lung.volume *= math.exp(-dt / 1.5)
            if abs(lung.volume) < 1e-4:
                lung.volume = 0.0
        source_e: str | None = (
            "vent" if vent_active else ("bag" if bag_mask_active else ("spontaneous" if connected else None))
        )
        single_e = (
            engine._manual_breath_requested
            and source_e == "spontaneous"
            and engine.apl_cmh2o > 0.0
        )
        engine._manual_breath_requested = False
        if source_e in ("spontaneous", "bag") and engine.apl_cmh2o > 0.0 and engine.circuit.reservoir_volume > 0.05:
            target_apl = min(engine.apl_cmh2o, 10.0) if engine.circuit.total_inflow() > 0.0 else 0.0
            engine._apl_pressure += (target_apl - engine._apl_pressure) * -math.expm1(-dt / 2.0)
        else:
            engine._apl_pressure *= math.exp(-dt / 1.5)
            if engine._apl_pressure < 0.05:
                engine._apl_pressure = 0.0
        cache_e = engine._step_gas
        bellows_e = (
            cache_e.bellows_fraction
            if cache_e is not None and cache_e.bellows_fraction is not None
            else engine.circuit.bellows_fraction
        )
        res_e = (
            cache_e.reservoir_volume_l
            if cache_e is not None and cache_e.reservoir_volume_l is not None
            else engine.circuit.reservoir_volume
        )
        fresh_e = engine.circuit.total_inflow() / engine.resp._btps_to_stpd / 60.0
        esoph_mech = engine.airway_crisis.esophageal_mech
        esoph_mech.effort.set_drive(0.0, 0.0)
        engine.vent.step(
            dt, esoph_mech, source_e, bag=(engine.bag_mask_rr, engine.bag_mask_vt),
            collect_samples=collect_samples, bag_p_max=engine.apl_cmh2o,
            spontaneous_peep=engine._apl_pressure,
            bellows_fraction=bellows_e,
            reservoir_volume=res_e / engine.resp._btps_to_stpd,
            volume_delivery=1.0,
            effort_cap=0.0,
            fresh_flow_l_s=fresh_e,
            mask_leak_gap=0.0,
            trigger_manual_breath=single_e,
            cpr_active=False,
        )
        engine.airway_crisis.sync_esophageal_breaths(engine.vent.breath_count)
        return
    source: str | None
    if vent_active:
        source = "vent"
    elif bag_mask_active:
        source = "bag"
    else:
        source = "spontaneous" if connected else None
    single_breath = (
        engine._manual_breath_requested
        and source == "spontaneous"
        and engine.apl_cmh2o > 0.0
    )
    engine._manual_breath_requested = False
    if (
        engine.state.airway_mode == AirwayType.MASK
        and (source == "bag" or source == "vent" or single_breath)
    ):
        engine._had_mask_ventilation = True
    bag_full = engine.circuit.reservoir_volume >= engine.circuit.reservoir_capacity - 1e-6
    if source in ("spontaneous", "bag") and engine.apl_cmh2o > 0.0:
        if engine.state.airway_mode == AirwayType.MASK:
            seal_p = (
                engine.airway_tuning.two_hand_seal_pressure
                if engine.airway_crisis.two_hand_mask
                else engine.airway_tuning.mask_seal_pressure
            )
            apl_ceiling = min(engine.apl_cmh2o, seal_p)
        elif engine.state.airway_mode == AirwayType.SGA:
            apl_ceiling = min(engine.apl_cmh2o, engine.airway_tuning.sga_seal_pressure)
        else:
            apl_ceiling = engine.apl_cmh2o
        half_cap = 0.5 * max(1e-6, engine.circuit.reservoir_capacity)
        bag_tautness = (
            1.0
            if bag_full
            else clamp01((engine.circuit.reservoir_volume - half_cap) / half_cap)
        )
        if bag_tautness > 0.0:
            target_ceiling = apl_ceiling * bag_tautness
            frc_inflow_l_min = (
                max(0.0, aeration.frc - engine.resp.frc)
                * engine.resp._exchangeable_frc_fraction
                * clamp01(engine.airway_status.patency)
                * mask_circuit_share(engine)
                * engine.resp._btps_to_stpd
                * 60.0
                / dt
            )
            inflow_l_min = (
                max(0.0, engine.circuit.last_spill_flow) + frc_inflow_l_min
                if bag_full
                else (
                    engine.circuit.last_net_flow + frc_inflow_l_min
                    if engine.circuit.last_net_flow + frc_inflow_l_min > 0.0
                    else min(engine.circuit.last_net_flow, -0.5 * engine._apl_pressure)
                )
            )
            peep_vent: float | None = None
            vent_max_l_min = 0.0
            if source == "bag" and engine.bag_mask_rr > 0.0:
                period = 60.0 / engine.bag_mask_rr
                ti = min(BAG_TI_MAX_S, period / 3.0)
                delta_p = _manual_breath_delta_p(lung, engine.bag_mask_vt, ti)
                peep_vent = max(0.0, engine.apl_cmh2o - max(delta_p, BAG_VENT_PRESSURE_CMH2O))
                vent_max_l_min = max(engine.bag_mask_vt, 1.0) * engine.bag_mask_rr
            elif single_breath or engine.vent.manual_breath_active:
                delta_p = _manual_breath_delta_p(lung, engine.bag_mask_vt, BAG_TI_MAX_S)
                peep_vent = max(0.0, engine.apl_cmh2o - max(delta_p, BAG_VENT_PRESSURE_CMH2O))
                vent_max_l_min = max(engine.bag_mask_vt, 1.0) * 60.0 / BAG_TI_MAX_S
            engine._apl_pressure = _advance_apl_pressure(
                engine._apl_pressure,
                dt,
                inflow_l_min,
                0.05 + max(0.01, lung.compliance),
                target_ceiling,
                peep_vent=peep_vent,
                vent_max_l_min=vent_max_l_min,
            )
        else:
            engine._apl_pressure = 0.0
    else:
        engine._apl_pressure *= math.exp(-dt / 0.4)
        if engine._apl_pressure < 1e-3:
            engine._apl_pressure = 0.0
    cache = engine._step_gas
    bellows_frac = (
        cache.bellows_fraction
        if cache is not None and cache.bellows_fraction is not None
        else engine.circuit.bellows_fraction
    )
    reservoir_vol = (
        cache.reservoir_volume_l
        if cache is not None and cache.reservoir_volume_l is not None
        else engine.circuit.reservoir_volume
    )
    volume_delivery = (
        1.0 - mask_leak(engine)
        if (assisted_positive_pressure(engine) or single_breath)
        else mask_seal_factor(engine)
    )
    fresh_flow_btps_l_s = (
        engine.circuit.total_inflow()
        / engine.resp._btps_to_stpd
        / 60.0
    )
    mask_leak_gap = 1.0 - seal_scale
    engine.vent.step(
        dt, lung, source, bag=(engine.bag_mask_rr, engine.bag_mask_vt),
        collect_samples=collect_samples, bag_p_max=engine.apl_cmh2o,
        spontaneous_peep=engine._apl_pressure,
        bellows_fraction=bellows_frac,
        reservoir_volume=reservoir_vol / engine.resp._btps_to_stpd,
        volume_delivery=volume_delivery,
        effort_cap=engine.hemo.config.effort_pressure_max,
        fresh_flow_l_s=fresh_flow_btps_l_s,
        mask_leak_gap=mask_leak_gap,
        trigger_manual_breath=single_breath,
        cpr_active=engine.hemo.compressions,
        cpr_rate=engine.hemo.cpr_rate,
        cpr_depth_cm=engine.hemo.cpr_depth_cm,
        cpr_recoil=engine.hemo.cpr_recoil,
    )


def update_shivering(engine: SimulationEngine, dt: float) -> None:
    """Update shivering intensity based on temperature, perfusion, and anesthetic state."""
    state = engine.state
    remi_effect = hill_function(state.opioid_ce, engine.resp.c50_remi, engine.resp.gamma_remi)
    # Shivering muscle is as sensitive to block as the adductor pollicis.
    muscle_factor = state.twitch / 100.0
    perfusing = is_systemic_perfusing(engine) and not state.cardiac_arrest
    state.shivering = float(
        engine.thermal.step_shivering(
            dt, state.temp_c, state.bis, remi_effect, muscle_factor,
            unconscious=state.loc, perfusing=perfusing,
            neuraxial_shiver_shift_c=engine.neuraxial.shiver_threshold_shift_c,
            neuraxial_muscle_keep=engine.neuraxial.shiver_muscle_keep,
        )
    )


def step_temperature(engine: SimulationEngine, dt: float) -> None:
    """Update core temperature from metabolic heat, environmental loss, fluids, and redistribution."""
    state = engine.state
    remi_effect = hill_function(state.opioid_ce, engine.resp.c50_remi, engine.resp.gamma_remi)
    ecmo = engine.extracorporeal
    hx_watts = ecmo.compute_heat_exchanger_watts(state.temp_c) if ecmo.is_active else 0.0
    state.temp_c = float(engine.thermal.step_temperature(
        dt,
        state.temp_c,
        state.bair_hugger_target,
        state.surface_cooling,
        oxygen_supply_fraction=engine.resp.oxygen_supply_fraction,
        sepsis_severity=engine.sepsis.severity,
        perfusing=is_systemic_perfusing(engine),
        opioid_effect=remi_effect,
        extracorporeal_heat_w=hx_watts,
        extracorporeal_active=ecmo.is_active,
        neuraxial_redistribution_depth=engine.neuraxial.thermal_redistribution_depth,
        neuraxial_vc_shift_c=engine.neuraxial.vasoconst_threshold_shift_c,
        neuraxial_sweat_shift_c=engine.neuraxial.sweat_threshold_shift_c,
        circulation=engine.hemo.circulation_fraction(),
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
        nx_keep = (
            1.0 - engine.neuraxial.surgical_analgesia(engine.patient.surgical_site)
            if disturbance is engine.disturbances and spec.airway_stimulus_s <= 0.0
            else 1.0
        )
        bis += stimulus.bis * (1.0 - engine._tol_current) * nx_keep
        svr += (stimulus.svr * pressure + spec.tonic_svr(t_rel + 0.5 * dt, 1000.0 * engine.hemo.tpr)) * nx_keep
        sv += stimulus.sv * pressure * nx_keep
        heart_rate += stimulus.hr * hr * nx_keep
    return DisturbanceEffects(bis=bis, svr=svr, sv=sv, hr=heart_rate)


def step_disturbances(engine: SimulationEngine, dt: float) -> DisturbanceEffects:
    """Calculate disturbances and update event-driven volume state."""
    effects = stimulus_effects(engine, dt)
    hemo = engine.hemo
    if engine.active_hemorrhage:
        rate_sec = (
            engine.hemorrhage_rate_ml_min
            * engine.coagulation.state.bleeding_multiplier
            * engine.embolism.thrombolysis_bleeding_multiplier
            / 60.0
        )
        bled_ml = rate_sec * dt
        hemo.add_volume(-bled_ml)
        engine.coagulation.record_blood_loss(bled_ml, hemo.blood_volume_0)

    warmed = engine.state.fluid_warmer
    core_temp = engine.state.temp_c
    fluid_temp = 37.0 if warmed else 21.0

    if engine.pending_infusions:
        # Drain infusions sequentially per IV channel (blood, lipid, bolus dextrose,
        # bicarbonate, fluid) so repeated clicks on the same line queue while rescue
        # pushes do not wait behind large crystalloid bags.
        channel_dt: dict[str, float] = {}
        remaining = []
        for infusion in engine.pending_infusions:
            is_blood = infusion.hematocrit > 0.0 or infusion.label == "blood"
            is_plasma_product = infusion.label in ("ffp", "platelets", "cryoprecipitate")
            if is_blood or is_plasma_product:
                channel = "blood"
            elif infusion.label == "lipid":
                channel = "lipid"
            elif infusion.label in ("d10w", "d50w"):
                channel = "dextrose"
            elif infusion.label == "bicarbonate":
                channel = "bicarbonate"
            else:
                channel = "fluid"
            rem_dt = channel_dt.get(channel, dt)
            if rem_dt > 0.0 and infusion.rate_ml_min > 0.0:
                rate_sec = infusion.rate_ml_min / 60.0
                amount_this_step = min(infusion.remaining_ml, rate_sec * rem_dt)
                channel_dt[channel] = max(0.0, rem_dt - amount_this_step / rate_sec)
                infusion.remaining_ml -= amount_this_step
                if amount_this_step > 0:
                    if infusion.label == "lipid":
                        engine.toxicity.add_lipid(amount_this_step)
                    hemo.add_volume(
                        amount_this_step,
                        hematocrit=infusion.hematocrit,
                        retention_fraction=infusion.retention_fraction,
                        label=infusion.label,
                        storage_days=infusion.storage_days,
                    )
                    if is_plasma_product:
                        engine.coagulation.record_transfusion(
                            infusion.label,
                            amount_this_step,
                            hemo.fluids.plasma_l * 1000.0,
                        )
                    infusate_temp = 37.0 if warmed else (6.0 if is_blood else 21.0)
                    engine.thermal.add_fluid_heat_j(
                        amount_this_step, infusate_temp, core_temp, is_blood=is_blood,
                    )
            if infusion.remaining_ml > 1e-3:
                remaining.append(infusion)
        engine.pending_infusions[:] = remaining

    if engine.lipid_infusion_ml_min > 0.0:
        amount = engine.lipid_infusion_ml_min * dt / 60.0
        hemo.add_volume(amount, retention_fraction=0.25, label="lipid")
        engine.toxicity.add_lipid(amount)
        engine.thermal.add_fluid_heat_j(amount, fluid_temp, core_temp)

    if engine.maintenance_fluid_rate_ml_min > 0:
        rate_sec = engine.maintenance_fluid_rate_ml_min / 60.0
        maint_ml = rate_sec * dt
        hemo.add_volume(maint_ml, hematocrit=0.0, label="crystalloid")
        engine.thermal.add_fluid_heat_j(maint_ml, fluid_temp, core_temp)

    hemo.anaphylaxis_severity = engine.anaphylaxis.step(dt)
    hemo.sepsis_severity = engine.sepsis.step(dt)

    return effects


def _disturbance_completes_during_step(engine: SimulationEngine, dt: float) -> bool:
    if not engine.disturbance_active:
        return False
    t_rel = max(0.0, engine.state.time - engine.disturbance_start_time)
    return engine.disturbances.is_complete(t_rel + dt)


def _tci_time_eps(sim_time: float) -> float:
    return 1e-9 * max(1.0, abs(sim_time))


def _flush_tci_to_time(engine: SimulationEngine, key: str, controller, sim_time: float) -> float:
    sampling_time = controller.sampling_time
    eps = _tci_time_eps(sim_time)
    acc = engine._tci_accumulators.get(key, 0.0)
    steps = int((acc + eps) / sampling_time)
    if steps > 0:
        for i in range(steps):
            remaining = acc - (i + 1) * sampling_time
            sample_time = sim_time if abs(remaining) <= eps else sim_time - remaining
            engine.infusion_rates[key] = controller.step(sim_time=sample_time)
        acc = max(0.0, acc - steps * sampling_time)
    if sim_time + eps >= controller.next_update_time:
        engine.infusion_rates[key] = controller.step(sim_time=sim_time)
        acc = 0.0
    return acc


def step_tci(engine: SimulationEngine, dt: float) -> None:
    """Advance TCI controllers on their own sampling clock."""
    sim_time = engine.state.time
    eps = _tci_time_eps(sim_time + dt)
    for key, controller in engine.tci.items():
        acc = _flush_tci_to_time(engine, key, controller, sim_time)
        if controller.next_update_time >= sim_time + dt - eps:
            engine._tci_accumulators[key] = acc + dt
        else:
            # Defer mid-step pulse transitions until step_pk advances PK to next_update_time.
            engine._tci_accumulators[key] = acc


def _extracorporeal_volatile_kwargs(engine: SimulationEngine, *, is_sevo: bool) -> dict:
    """Optional oxygenator volatile-transfer kwargs; empty when extracorporeal support is off."""
    ecmo = engine.extracorporeal
    if not ecmo.is_active:
        return {}
    st = ecmo.state
    q_eff = st.effective_blood_flow_l_min if ecmo.is_vv else st.blood_flow_l_min
    if ecmo.is_vv and q_eff <= 0.0 and st.blood_flow_l_min > 0.0:
        q_eff = st.blood_flow_l_min * (1.0 - st.recirculation_fraction)
    return {
        "pulmonary_flow_l": engine.hemo.state.pulmonary_flow,
        "ecmo_blood_flow_l": q_eff,
        "ecmo_sweep_l": st.sweep_gas_l_min,
        "ecmo_fi_agent": (st.sweep_sevo_percent / 100.0) if (is_sevo and engine.config.sevoflurane_enabled) else 0.0,
        "ecmo_permeability": ecmo.volatile_permeability,
        "ecmo_is_arterial": ecmo.is_arterial_support,
    }


def _supported_minute_ventilation(engine: SimulationEngine) -> float:
    """Return active positive-pressure minute ventilation (L/min BTPS) for circuit leak accounting."""
    if engine.vent.has_measured_breath:
        return engine.vent.monitors.mv_supported
    if engine.vent.is_on and engine.vent.settings.mode in VOLUME_TARGETED:
        return engine.vent.settings.rr * engine.vent.settings.tv / 1000.0
    if engine.bag_mask_active and not engine.vent.is_on and engine.apl_cmh2o > 0.0:
        return engine.bag_mask_rr * engine.bag_mask_vt
    return 0.0


def step_machine(
    engine: SimulationEngine, dt: float, co_now: float | None = None,
) -> tuple[float, float]:
    """Update machine state and return inspired volatile fractions."""
    circuit = engine.circuit
    resp = engine.resp
    btps_to_stpd = resp.btps_to_stpd
    if engine._o2_flush_remaining_s > 0.0 and dt > 0.0:
        flush_dt = min(dt, engine._o2_flush_remaining_s)
        circuit.o2_flush_flow = O2_FLUSH_FLOW_L_MIN * (flush_dt / dt)
        engine._o2_flush_remaining_s = max(0.0, engine._o2_flush_remaining_s - dt)
    else:
        circuit.o2_flush_flow = 0.0
    delivered_setting = (
        engine.vaporizer.step(dt, circuit.fgf_total()) if engine.config.sevoflurane_enabled else 0.0
    )
    circuit.vaporizer_setting = (
        delivered_setting
        if circuit.fgf_total() > 0.0 or not engine.vaporizer.state.is_on
        else engine.vaporizer.state.setting
    )
    circuit.vaporizer_on = bool(
        engine.config.sevoflurane_enabled
        and (engine.vaporizer.state.is_on or delivered_setting > 0.0)
    )

    if co_now is None:
        co_now = engine.hemo.state.co
    shunt_now = engine.aeration.shunt_fraction
    temp_now = engine.state.temp_c
    pulm_q = engine.hemo.state.pulmonary_flow if engine.extracorporeal.is_active else None
    n2o_uptake = engine.pk_n2o.alveolar_uptake_l_min(co_now, shunt_now, temp_c=temp_now, pulmonary_flow_l=pulm_q)
    sevo_uptake = engine.pk_sevo.alveolar_uptake_l_min(co_now, shunt_now, temp_c=temp_now, pulmonary_flow_l=pulm_q)
    agent_uptake = n2o_uptake + sevo_uptake
    target_lung_vol = engine.aeration.frc
    circuit_share = mask_circuit_share(engine)
    apneic_limit: float | None = None
    leak_l_min = 0.0
    if engine.vent.is_on and (
        engine.state.airway_mode == AirwayType.NONE
        or (engine.state.airway_mode == AirwayType.MASK and engine.mask_seal <= 0.0)
    ):
        # An open Y-piece with the ventilator running vents driving breaths to room air,
        # draining the ascending bellows unless FGF matches or exceeds minute leak.
        s_vent = engine.vent.settings
        if s_vent.mode in VOLUME_TARGETED:
            open_mv = max(0.0, s_vent.rr * s_vent.tv / 1000.0)
        elif s_vent.mode == "CPAP":
            open_mv = 0.45 * max(0.0, s_vent.peep)
        else:
            open_mv = max(0.0, s_vent.rr * 0.5 * max(0.25, s_vent.p_insp / 15.0)) + 0.45 * max(0.0, s_vent.peep)
        leak_l_min = open_mv * btps_to_stpd
    elif is_esophageal_intubation(engine) and assisted_positive_pressure(engine):
        esoph_mv = _supported_minute_ventilation(engine)
        leak_l_min = max(0.0, esoph_mv) * (1.0 - ESOPHAGEAL_RETURN_FRACTION) * btps_to_stpd
    elif engine.state.airway_mode == AirwayType.SGA and assisted_positive_pressure(engine):
        sga_mv = _supported_minute_ventilation(engine)
        leak_l_min = max(0.0, sga_mv) * mask_leak(engine) * btps_to_stpd
    elif engine.state.airway_mode == AirwayType.MASK and circuit_connected(engine):
        eff_seal = effective_mask_seal(engine)
        if assisted_positive_pressure(engine):
            supported_mv = _supported_minute_ventilation(engine)
            leak_l_min = max(0.0, supported_mv) * mask_leak(engine) * btps_to_stpd
        if eff_seal < 0.999:
            cpap_cmh2o = (
                engine.vent.settings.peep
                if engine.vent.is_on
                else engine._apl_pressure
            )
            if cpap_cmh2o > 0.0:
                exp_share = 1.0
                if engine.vent.is_on:
                    period = engine.vent.mandatory_period_s
                    if period is not None and math.isfinite(period) and period > 0.0:
                        s_v = engine.vent.settings
                        if s_v.mode.startswith("SIMV"):
                            t_insp = min(s_v.t_insp, 0.8 * period)
                        else:
                            t_insp = period * s_v.ie_ratio / (1.0 + s_v.ie_ratio)
                        exp_share = max(0.0, 1.0 - t_insp / period)
                elif engine.bag_mask_active and engine.bag_mask_rr > 0.0 and engine.apl_cmh2o > 0.0:
                    period = 60.0 / engine.bag_mask_rr
                    exp_share = max(0.0, 1.0 - min(BAG_TI_MAX_S, period / 3.0) / period)
                leak_l_min += exp_share * 0.45 * cpap_cmh2o * (1.0 - eff_seal) * btps_to_stpd
    rebreathed_share = 1.0
    exchange = None
    va = 0.0
    sevo_result = None
    n2o_result = None
    if circuit_share <= 0.0:
        tracheal_fio2, fi_sevo, fi_n2o, tracheal_fin2 = advance_tracheal_gas(
            engine, dt, effective_va=0.0, circuit_share=0.0,
        )
        uptake_o2 = uptake_sevo = uptake_n2o = uptake_n2 = 0.0
    else:
        airway = engine.airway_status
        patency = clamp01(airway.patency)
        raw_va = max(0.0, engine.state.va) if patency > 0.0 else 0.0
        va = resp.exchange_va(raw_va, airway.vq_mismatch)
        va_circuit = va * circuit_share
        fresh_btps = circuit.total_inflow() / btps_to_stpd
        rebreathed_share = (
            clamp01(1.0 - fresh_btps / max(0.1, va_circuit))
            if va_circuit > 0.0
            else 0.0
        )
        composition = circuit.composition
        apneic_limit = circuit.deliverable_flow(dt) if circuit_share >= 0.999 else None
        terms = resp.passive_exchange_terms(
            target_lung_vol, patency, agent_uptake, dt, apneic_limit,
        )
        exchange = terms
        frc_inflow_stpd = max(0.0, terms.exhaled_vc_btps_l) * btps_to_stpd * 60.0 / dt if dt > 0.0 else 0.0
        inflow_stpd = terms.passive_stpd_l_min + frc_inflow_stpd
        tracheal_fio2, tracheal_fi_sevo, tracheal_fin2o, tracheal_fin2 = advance_tracheal_gas(
            engine, dt, inflow_stpd_l_min=inflow_stpd, effective_va=va, circuit_share=circuit_share,
        )
        if va <= 1e-4:
            circuit_fio2, circuit_fi_sevo, circuit_fin2o, circuit_fin2 = (
                tracheal_fio2, tracheal_fi_sevo, tracheal_fin2o, tracheal_fin2
            )
            fi_sevo = tracheal_fi_sevo
            fi_n2o = tracheal_fin2o
        else:
            circuit_fio2, circuit_fi_sevo, circuit_fin2o, circuit_fin2 = circuit_inspired_mixture(
                composition, tracheal_fio2, circuit_share
            )
            fi_sevo = circuit_share * circuit_fi_sevo
            fi_n2o = circuit_share * circuit_fin2o
        sevo_ecmo_kw = _extracorporeal_volatile_kwargs(engine, is_sevo=True)
        n2o_ecmo_kw = _extracorporeal_volatile_kwargs(engine, is_sevo=False)
        sevo_btps, sevo_result = engine.pk_sevo.preview_exchange(
            dt, fi_sevo, va, co_now, temp_c=temp_now, lung_volume_l=terms.gas_vol_l,
            exhaled_volume_change_l=terms.exhaled_vc_btps_l,
            shunt_fraction=shunt_now, companion_uptake_l_min=terms.companion_btps_l_min,
            passive_inflow_l_min=terms.passive_o2_btps_l_min,
            circuit_share=circuit_share, circuit_fi_agent=circuit_fi_sevo,
            **sevo_ecmo_kw,
        )
        uptake_sevo = sevo_btps * btps_to_stpd
        n2o_btps, n2o_result = engine.pk_n2o.preview_exchange(
            dt, fi_n2o, va, co_now, temp_c=temp_now, lung_volume_l=terms.gas_vol_l,
            exhaled_volume_change_l=terms.exhaled_vc_btps_l,
            shunt_fraction=shunt_now, companion_uptake_l_min=terms.companion_btps_l_min,
            passive_inflow_l_min=terms.passive_o2_btps_l_min,
            circuit_share=circuit_share, circuit_fi_agent=circuit_fin2o,
            **n2o_ecmo_kw,
        )
        uptake_n2o = n2o_btps * btps_to_stpd
        if circuit_share >= 0.999:
            uptake_n2 = resp.nitrogen_exchange_l_min(
                circuit_fin2, raw_va, airway.patency, airway.vq_mismatch, dt,
                apneic_flow_limit_l_min=apneic_limit, n2o_uptake_l_min=agent_uptake,
                lung_volume_l=target_lung_vol, cardiac_output=co_now, shunt_fraction=shunt_now,
                exchange=terms, effective_va_l_min=va,
            )
        else:
            mixing = va * dt / (120.0 * terms.gas_vol_l) if terms.gas_vol_l > 0.0 else 0.0
            alveolar_n2 = resp.alveolar_fn2
            frc_stpd_l_min = terms.exhaled_vc_btps_l * btps_to_stpd * 60.0 / dt if dt > 0.0 else 0.0
            frc_n2 = circuit_fin2 if frc_stpd_l_min >= 0.0 else alveolar_n2
            washout_stpd = terms.washout_stpd_l * 60.0 / dt if dt > 0.0 else 0.0
            tidal_scale = va_circuit * btps_to_stpd / (1.0 + mixing)
            uptake_n2 = tidal_scale * (circuit_fin2 - alveolar_n2) + circuit_share * (
                frc_stpd_l_min * frc_n2 + terms.passive_stpd_l_min * circuit_fin2 - washout_stpd * alveolar_n2
            )
        uptake_o2 = circuit_share * resp.oxygen_exchange_l_min(
            circuit_fio2, raw_va,
            airway.patency, airway.vq_mismatch, dt,
            n2o_uptake_l_min=agent_uptake,
            apneic_flow_limit_l_min=apneic_limit,
            lung_volume_l=target_lung_vol,
            exchange=terms, effective_va_l_min=va,
        )
        # During apneic bulk inflow, gas enters the conserved dead-space buffer at
        # circuit composition and leaves into the alveoli at tracheal composition.
        if va <= 1e-4 and inflow_stpd > 0.0:
            uptake_o2 += circuit_share * inflow_stpd * (composition.fio2 - tracheal_fio2)
            uptake_sevo += circuit_share * inflow_stpd * (composition.fi_agent - tracheal_fi_sevo)
            uptake_n2o += circuit_share * inflow_stpd * (composition.fin2o - tracheal_fin2o)
            uptake_n2 += circuit_share * inflow_stpd * (composition.fin2 - tracheal_fin2)
    pre_reservoir = circuit.reservoir_volume
    pre_bellows = circuit.bellows_fraction
    s_vent = engine.vent.settings
    pip_now = (
        engine.vent.monitors.paw_peak
        if engine.vent.has_measured_breath and engine.vent.monitors.paw_peak > 0.0
        else (s_vent.peep + s_vent.p_insp)
    )
    drive_demand = circuit.gas_supply.ventilator_drive_gas_demand_l_min(
        vent_is_on=engine.vent.is_on and circuit.gas_supply.vent_drive_available,
        mode=s_vent.mode,
        rr=s_vent.rr,
        vt_l=s_vent.tv / 1000.0,
        ie_ratio=s_vent.ie_ratio,
        t_insp=s_vent.t_insp,
        pip_cmh2o=pip_now,
        fgf_l_min=circuit.fgf_total(),
        mv_supported_l_min=engine.vent.monitors.mv_supported if engine.vent.has_measured_breath else None,
    )
    circuit.step(
        dt,
        uptake_o2,
        uptake_sevo,
        uptake_n2o,
        uptake_n2,
        leak_l_min=leak_l_min,
        rebreathed_share=rebreathed_share,
        drive_gas_l_min=drive_demand,
    )
    circuit.vaporizer_on = bool(engine.config.sevoflurane_enabled and engine.vaporizer.state.is_on)
    if not circuit.vaporizer_on:
        circuit.vaporizer_setting = 0.0
    sync_inspired_gas(engine)
    engine._step_gas = StepGasCache(
        tracheal_fio2=tracheal_fio2,
        n2o_uptake_l_min=n2o_uptake,
        lung_volume_l=target_lung_vol,
        tracheal_fin2=tracheal_fin2,
        sevo_uptake_l_min=sevo_uptake,
        apneic_limit_l_min=apneic_limit,
        reservoir_volume_l=pre_reservoir,
        bellows_fraction=pre_bellows,
        exchange=exchange,
        va=va,
        co=co_now,
        sevo_result=sevo_result,
        n2o_result=n2o_result,
        dt=dt,
    )
    return fi_sevo, fi_n2o


def step_pk(engine: SimulationEngine, dt: float, fi_sevo: float, fi_n2o: float, co_curr: float) -> None:
    """Update pharmacokinetic models and synchronize their public state."""
    state = engine.state
    aeration = engine.aeration
    airway = engine.airway_status
    resp = engine.resp
    patency = clamp01(airway.patency)
    va = resp.exchange_va(state.va, airway.vq_mismatch) if patency > 0.0 else 0.0
    cache = engine._step_gas
    sealed_circuit = circuit_connected(engine) and mask_circuit_share(engine) >= 0.999
    if cache is not None and cache.exchange is not None:
        terms = cache.exchange
    else:
        if cache is not None:
            agent_uptake = cache.n2o_uptake_l_min + cache.sevo_uptake_l_min
        else:
            pulm_q = engine.hemo.state.pulmonary_flow if engine.extracorporeal.is_active else None
            agent_uptake = (
                engine.pk_n2o.alveolar_uptake_l_min(
                    co_curr, aeration.shunt_fraction, temp_c=state.temp_c, pulmonary_flow_l=pulm_q,
                )
                + engine.pk_sevo.alveolar_uptake_l_min(
                    co_curr, aeration.shunt_fraction, temp_c=state.temp_c, pulmonary_flow_l=pulm_q,
                )
            )
        target_lung_vol = cache.lung_volume_l if cache is not None else aeration.frc
        fallback_flow_cap = max(0.0, engine.circuit.deliverable_flow_l_min) if sealed_circuit else None
        terms = resp.passive_exchange_terms(
            target_lung_vol,
            patency,
            agent_uptake,
            dt,
            cache.apneic_limit_l_min if cache is not None else None,
            fallback_flow_cap=fallback_flow_cap,
        )
    uptake_scale = engine.circuit.last_uptake_scale if sealed_circuit else 1.0
    va_sevo = va * uptake_scale if fi_sevo > engine.pk_sevo.state.p_alv else va
    va_n2o = va * uptake_scale if fi_n2o > engine.pk_n2o.state.p_alv else va
    if (
        cache is not None
        and cache.sevo_result is not None
        and va_sevo == cache.va
        and co_curr == cache.co
    ):
        engine.pk_sevo.commit_integration(cache.sevo_result, dt, temp_c=state.temp_c)
    else:
        engine.pk_sevo.step(
            dt, fi_sevo, va_sevo, co_curr, temp_c=state.temp_c,
            lung_volume_l=terms.gas_vol_l, shunt_fraction=aeration.shunt_fraction,
            companion_uptake_l_min=terms.companion_btps_l_min,
            passive_inflow_l_min=terms.passive_o2_btps_l_min,
            exhaled_volume_change_l=terms.exhaled_vc_btps_l,
            **_extracorporeal_volatile_kwargs(engine, is_sevo=True),
        )
    if (
        cache is not None
        and cache.n2o_result is not None
        and va_n2o == cache.va
        and co_curr == cache.co
    ):
        engine.pk_n2o.commit_integration(cache.n2o_result, dt, temp_c=state.temp_c)
    else:
        engine.pk_n2o.step(
            dt, fi_n2o, va_n2o, co_curr, temp_c=state.temp_c,
            lung_volume_l=terms.gas_vol_l, shunt_fraction=aeration.shunt_fraction,
            companion_uptake_l_min=terms.companion_btps_l_min,
            passive_inflow_l_min=terms.passive_o2_btps_l_min,
            exhaled_volume_change_l=terms.exhaled_vc_btps_l,
            **_extracorporeal_volatile_kwargs(engine, is_sevo=False),
        )
    if engine.extracorporeal.is_active:
        engine.extracorporeal.state.exhaust_sevo_percent = float(engine.pk_sevo.last_exhaust_fraction * 100.0)
    sync_inhaled_agents(engine)

    rates = engine.infusion_rates
    sim_time = state.time
    time_eps = _tci_time_eps(sim_time + dt)
    mac_sevo = engine.pk_sevo.state.mac
    mac_n2o = engine.pk_n2o.state.mac
    segmented_keys = (
        {
            key
            for key, ctrl in engine.tci.items()
            if dt > 0.0 and ctrl.next_update_time < sim_time + dt - time_eps
        }
        if engine.tci
        else ()
    )

    for key, model in engine.pk.items():
        if segmented_keys and key in segmented_keys:
            continue
        if key in ("heparin", "protamine"):
            continue
        rate = rates.get(key, 0.0)
        if key == "roc":
            _step_roc_drug(engine, dt, rate, mac_sevo, mac_n2o)
        elif rate or model.basal_input_rate or model.has_mass():
            model.step(dt, rate)
    heparin_pk = cast(HeparinPK, engine.pk["heparin"])
    protamine_pk = cast(ProtaminePK, engine.pk["protamine"])
    was_prot_rxn = engine.event_active("protamine_reaction")
    engine.coagulation.step(
        dt,
        heparin_pk,
        protamine_pk,
        heparin_rate_u_s=rates.get("heparin", 0.0),
        protamine_rate_mg_s=rates.get("protamine", 0.0),
        temp_c=engine.state.temp_c,
        hematocrit=engine.hemo.get_hematocrit(),
        baseline_hct=engine.hemo.baseline_hct,
        circulation=engine.hemo.circulation_fraction(co_curr),
        ionized_calcium_mmol_l=engine.hemo.fluids.ionized_calcium(engine.resp.state.ph_37),
    )
    if not was_prot_rxn and engine.event_active("protamine_reaction"):
        from anasim.core.action_log import ACTION_EVENT_START

        engine.actions.record(engine.state.time, ACTION_EVENT_START, label="protamine_reaction")
    if engine.config.tci_enabled:
        for key, model in engine.pump_pk.items():
            if segmented_keys and key in segmented_keys:
                continue
            if key in engine._pump_needs_reset:
                continue
            rate = rates.get(key, 0.0) if key in engine.tci else 0.0
            if rate or model.basal_input_rate or model.has_mass():
                model.step(dt, rate)
    for key in segmented_keys:
        ctrl = engine.tci[key]
        t_cur = sim_time
        t_end = sim_time + dt
        while t_cur < t_end - time_eps:
            t_next = min(t_end, max(t_cur, ctrl.next_update_time))
            if t_next > t_cur + 1e-12:
                seg_dt = t_next - t_cur
                rate = rates.get(key, 0.0)
                _step_patient_drug(engine, key, seg_dt, rate, mac_sevo, mac_n2o)
                if key in engine.pump_pk:
                    engine.pump_pk[key].step(seg_dt, rate)
                engine._tci_accumulators[key] = engine._tci_accumulators.get(key, 0.0) + seg_dt
                t_cur = t_next
            if t_cur < t_end - time_eps:
                engine._tci_accumulators[key] = _flush_tci_to_time(engine, key, ctrl, t_cur)
                if ctrl.next_update_time <= t_cur + time_eps:
                    rem_dt = t_end - t_cur
                    rate = rates.get(key, 0.0)
                    _step_patient_drug(engine, key, rem_dt, rate, mac_sevo, mac_n2o)
                    if key in engine.pump_pk:
                        engine.pump_pk[key].step(rem_dt, rate)
                    engine._tci_accumulators[key] = engine._tci_accumulators.get(key, 0.0) + rem_dt
                    break
    if engine.extracorporeal.is_active:
        ecmo = engine.extracorporeal
        if ecmo.is_vv:
            recirc = recirculation_fraction(
                ecmo.state.blood_flow_l_min, engine.hemo.state.pulmonary_flow, ecmo.state.cannulation,
            )
            seq_flow = ecmo.state.blood_flow_l_min * (1.0 - recirc)
        else:
            seq_flow = ecmo.state.blood_flow_l_min
        ecmo.sequestration.step(
            dt,
            seq_flow,
            engine.pk,
            membrane_area_m2=ecmo.state.membrane_area_m2,
            active=seq_flow > 1e-6,
        )
    sync_pk_state(engine)


def _step_roc_drug(
    engine: SimulationEngine, seg_dt: float, rate: float, mac_sevo: float, mac_n2o: float,
) -> None:
    roc = engine.pk["roc"]
    tof = engine.tof_pd
    has_sug = tof.sugammadex.state.c1 > 0.0 or tof.complex.state.c1 > 0.0
    if not (
        rate
        or roc.has_mass()
        or tof.ce
        or tof.ce_central
        or has_sug
        or tof.sugammadex.state.c2
        or tof.complex.state.c2
    ):
        return
    neo_ce = engine.pk["neostigmine"].state.ce_nmj
    if seg_dt > 0.05 and has_sug:
        substeps = max(1, math.ceil(seg_dt / 0.05))
        h = seg_dt / substeps
        for _ in range(substeps):
            roc.step(h, rate)
            tof.step_recovery(h, roc, mac_sevo=mac_sevo, mac_n2o=mac_n2o, neostigmine_ce=neo_ce)
        return
    roc.step(seg_dt, rate)
    tof.step_recovery(seg_dt, roc, mac_sevo=mac_sevo, mac_n2o=mac_n2o, neostigmine_ce=neo_ce)


def _step_patient_drug(
    engine: SimulationEngine, drug_key: str, seg_dt: float, rate: float, mac_sevo: float, mac_n2o: float,
) -> None:
    if drug_key == "roc":
        _step_roc_drug(engine, seg_dt, rate, mac_sevo, mac_n2o)
        return
    engine.pk[drug_key].step(seg_dt, rate)


def update_pk_covariates(engine: SimulationEngine, co_curr: float, *, dt: float = 0.0) -> None:
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
        pre_loss_bv = max(500.0, engine.hemo.blood_volume + blood_lost)
        engine.toxicity.lipid_ml *= max(0.0, 1.0 - blood_lost / pre_loss_bv)
        for model in (*engine.pk.values(), engine.tof_pd.sugammadex, engine.tof_pd.complex):
            model.remove_blood(blood_lost / 1000.0, base_bv / 1000.0)
        engine._pk_blood_out_ml = engine.hemo.total_blood_out_ml
    # Heart failure's low output slows flow-dependent clearance. Pregnancy's
    # added output mostly perfuses the uterus and kidneys, so it is the reference.
    base_co = engine.hemo.normal_base_co_l_min
    if base_bv <= 0.0 or base_co <= 0.0:
        return
    # Below 30% of resting output, as in arrest and CPR, flow limits drug transport.
    circulation = engine.hemo.circulation_fraction(co_curr)
    if circulation != propofol.circulation:
        for model in (*engine.pk.values(), engine.tof_pd.sugammadex, engine.tof_pd.complex):
            model.circulation = circulation
    v_max = 2.5 if engine.extracorporeal.is_active else 2.0
    v_ratio = clamp(engine.hemo.blood_volume / base_bv, 0.1, v_max)
    co_ratio = clamp(co_curr / base_co, 0.1, 2.0)
    renal_perf = engine.hemo.renal_perfusion
    temp_c = engine.state.temp_c

    # Hiraoka 2004 (CPB PK): unbound propofol fraction doubles during CPB alongside
    # albumin dilution and elevated free fatty acids. Extended across active
    # extracorporeal support as a smooth albumin-driven effect-site scaling.
    if engine.extracorporeal.is_active:
        alb_g_l = max(12.0, engine.hemo.fluids.plasma_albumin_g_l)
        propofol.effect_input_fraction = clamp(40.0 / alb_g_l, 1.0, 2.0)
    elif propofol.effect_input_fraction != 1.0:
        if dt > 0.0:
            propofol.effect_input_fraction += (1.0 - propofol.effect_input_fraction) * -math.expm1(-dt / 180.0)
            if abs(propofol.effect_input_fraction - 1.0) < 1e-4:
                propofol.effect_input_fraction = 1.0
        else:
            propofol.effect_input_fraction = 1.0

    is_cpb = engine.extracorporeal.is_cpb
    last_scale = engine._pk_hemo_scale_cache
    # Changes below 0.1% leave concentrations unchanged at display precision.
    if last_scale is not None:
        lv, lc, lr, lt, l_cpb = last_scale
        if (
            l_cpb == is_cpb
            and abs(v_ratio - lv) <= max(1e-3, 1e-3 * max(abs(v_ratio), abs(lv)))
            and abs(co_ratio - lc) <= max(1e-3, 1e-3 * max(abs(co_ratio), abs(lc)))
            and abs(renal_perf - lr) <= max(1e-3, 1e-3 * max(abs(renal_perf), abs(lr)))
            and abs(temp_c - lt) <= max(1e-3, 1e-3 * max(abs(temp_c), abs(lt)))
        ):
            return

    bv_l = base_bv / 1000.0
    temp_floor = 18.0 if is_cpb else 25.0
    clamped_t = clamp(temp_c, temp_floor, 38.5)
    temp_met = 2.0 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0
    temp_dist = 1.6 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0
    tof = engine.tof_pd
    tof._temp_c = temp_c
    for model in (*engine.pk.values(), tof.sugammadex, tof.complex):
        model.update_hemodynamics(
            v_ratio, co_ratio, renal_perf,
            blood_volume_l=bv_l, temp_c=temp_c, temp_met=temp_met, temp_dist=temp_dist,
        )
    engine._pk_hemo_scale_cache = (v_ratio, co_ratio, renal_perf, temp_c, is_cpb)


def _beta2_dose_ratio(engine: SimulationEngine) -> float:
    labetalol_ce = engine.state.labetalol_ce
    return 1.0 + labetalol_ce / engine.hemo.config.labetalol_kb_beta if labetalol_ce > 0.0 else 1.0


def update_airway_complications(
    engine: SimulationEngine, dt: float, *, rhythm_type: RhythmType | None = None,
) -> None:
    """Update airway obstruction/bronchospasm/laryngospasm state."""
    state = engine.state
    airway_tuning = engine.airway_tuning
    if is_infraglottic_airway(engine):
        engine.laryngospasm_severity = 0.0
        upper_obstruction = clamp01(engine.airway_obstruction_manual)
    else:
        # Laryngoscopy or light extubation can provoke laryngospasm; an aborted
        # fully awake intubation stops at oropharyngeal gagging before the glottis.
        dist_stim = (
            engine.disturbance_active
            and engine.disturbances.stimulates_airway(state.time - engine.disturbance_start_time)
        )
        intub_stim = (
            engine.intubation_start_time is not None
            and state.loc >= 0.08
            and engine.intubation_stimulus.stimulates_airway(state.time - engine.intubation_start_time)
        )
        extub_stim = state.time < engine._extubation_stimulus_until
        stim_active = engine.auto_laryngospasm_enabled and (dist_stim or intub_stim or extub_stim)

        sux_central = succinylcholine_central_ce(engine)
        exposure = _succinylcholine_phase2_exposure(engine)
        phase2_tof = succinylcholine_phase2_tof_fraction(exposure) if exposure > 32.0 else 1.0
        depolarizing_strength = (
            (succinylcholine_twitch_fraction(sux_central, central=True) if sux_central > 0.0 else 1.0)
            * (phase2_tof ** 0.35 if phase2_tof < 1.0 else 1.0)
        )
        muscle_factor = central_muscle_strength(engine)
        sux_laryng = succinylcholine_laryngeal_spasm_factor(sux_central) if sux_central > 0.0 else 1.0
        laryng_cap = clamp01(min(muscle_factor, sux_laryng))

        splint_pressure = 0.0
        if state.airway_mode in (AirwayType.MASK, AirwayType.SGA):
            vent = engine.vent
            if vent.is_on:
                splint_pressure = vent.settings.peep
                if vent.settings.mode != "CPAP":
                    splint_pressure = max(splint_pressure, vent.monitors.paw_mean_recent)
            elif engine.bag_mask_active:
                splint_pressure = max(engine._apl_pressure, vent.monitors.paw_mean_recent)
            elif vent.manual_breath_active:
                splint_pressure = max(engine._apl_pressure, vent.monitors.paw_mean_recent, vent.paw)
            else:
                splint_pressure = engine._apl_pressure
            seal_eff = effective_mask_seal(engine) if state.airway_mode == AirwayType.MASK else 1.0
            splint_pressure *= seal_eff * clamp01(engine.circuit.bellows_fraction / 0.05)

        rhythm_now = engine.hemo.state.rhythm_type if rhythm_type is None else rhythm_type
        brainstem_viable = (
            engine.cerebral.respiratory_drive > 0.1
            and not state.cardiac_arrest
            and rhythm_now not in PULSELESS_RHYTHMS
        )
        extub_tol: float | None = None
        laryng_target = 0.0
        if stim_active and brainstem_viable:
            if dist_stim or intub_stim:
                active_tol = engine._tol_current
            else:
                extub_tol = extubation_tolerance(engine)
                active_tol = extub_tol
            light_factor = clamp01(1.0 - active_tol)
            laryng_target = clamp01(light_factor * laryng_cap * engine.cerebral.respiratory_drive)

        if brainstem_viable and state.loc >= 0.20 and splint_pressure < 10.0:
            if extub_tol is None:
                extub_tol = extubation_tolerance(engine)
            stage_ii_active = extub_tol < 0.65
        else:
            stage_ii_active = False

        # Smoothly shorten spasm persistence as laryngeal neuromuscular tone weakens,
        # and cap active spasm rapidly when succinylcholine or rocuronium relaxes the vocal cords.
        if laryng_cap < engine.laryngospasm_severity and laryng_cap < 0.98:
            target = min(laryng_target, laryng_cap) if laryng_target > 0.0 else laryng_cap
            tau = 4.0 if laryng_cap < 0.35 else 8.0
            if laryng_target <= 0.0 and not stage_ii_active:
                target = 0.0
            engine.laryngospasm_severity += (target - engine.laryngospasm_severity) * -math.expm1(-dt / tau)
        elif laryng_target != engine.laryngospasm_severity:
            sustain_weight = clamp01((laryng_cap - 0.15) / 0.55) if stage_ii_active else 0.0
            tau_off = airway_tuning.laryngospasm_tau_off + (40.0 - airway_tuning.laryngospasm_tau_off) * sustain_weight
            tau = airway_tuning.laryngospasm_tau_on if laryng_target > engine.laryngospasm_severity else tau_off
            if tau > 0:
                engine.laryngospasm_severity += (laryng_target - engine.laryngospasm_severity) * -math.expm1(-dt / tau)
        engine.laryngospasm_severity = clamp01(engine.laryngospasm_severity)

        # SGA bypasses pharyngeal soft-tissue collapse; mask CPAP, OPA, and two-handed
        # jaw thrust relieve it to varying degrees (Joffe 2010).
        if state.airway_mode == AirwayType.SGA:
            collapse = 0.0
            pathology = engine.airway_crisis.sga_pathology_obstruction()
        else:
            relief = clamp01(splint_pressure / airway_tuning.collapse_relief_pressure)
            if relief < 1.0:
                unconscious = loss_of_response(engine, ketamine=False)
                nondepol_tone = engine.tof_pd.pharyngeal_tone_fraction(state.mac_sevo, state.mac_n2o)
                depol_tone = 0.46 + 0.54 * depolarizing_strength
                tone = min(nondepol_tone, depol_tone)
                # Passive pharyngeal collapse stays below complete airtight glottic occlusion
                # (matching non-depolarizing reserve 0.40 / 0.46 ~= 0.87) so apneic O2 insufflation persists.
                sedated_collapse = min(0.87, airway_tuning.unsupported_collapse_max * unconscious / tone)
                # Calibrated awake collapse during severe neuromuscular weakness (TOF < 35%
                # or depolarizing weakness); milder residual block (TOF 0.5-0.8) raises
                # pharyngeal collapsibility without spontaneous collapse at rest (Herbstreit 2009).
                roc_tof = (
                    engine.tof_pd.compute_tof_from_ce(engine.tof_pd.ce, state.mac_sevo, state.mac_n2o) / 100.0
                    if engine.tof_pd.ce > 1e-9
                    else 1.0
                )
                tof_frac = roc_tof * phase2_tof
                severe_weakness = clamp01((0.35 - min(tof_frac, depolarizing_strength)) / 0.35)
                awake_nmba_collapse = 0.65 * severe_weakness
                adjunct_mult = (
                    engine.airway_crisis.mask_collapse_multiplier()
                    if state.airway_mode == AirwayType.MASK
                    else 1.0
                )
                collapse = max(sedated_collapse * adjunct_mult, awake_nmba_collapse * adjunct_mult) * (1.0 - relief)
            else:
                collapse = 0.0
            pathology = (
                engine.airway_crisis.mask_pathology_obstruction()
                if state.airway_mode == AirwayType.MASK
                else clamp01(max(engine.airway_crisis.base_pathology_obstruction, engine.airway_crisis.trauma_obstruction))
            )
        upper_obstruction = clamp01(
            max(
                engine.airway_obstruction_manual,
                engine.laryngospasm_severity,
                collapse,
                pathology,
                engine.opioid_toxicity.glottic_obstruction,
            )
        )

    bronch = 1.0 - (1.0 - engine.bronchospasm_manual) * (1.0 - engine.anaphylaxis.severity)
    if bronch > 0.0:
        # Epinephrine relieves bronchospasm via beta-2 receptors (antagonized by
        # non-selective labetalol); sevoflurane and ketamine add direct bronchodilation.
        beta2_dr = _beta2_dose_ratio(engine)
        epi_total = state.epi_ce + engine.neuraxial.absorbed_epi_ng_ml
        epi_dil = (
            airway_tuning.epi_bronchodilation_max
            * hill_function(epi_total, airway_tuning.epi_bronchodilation_c50 * beta2_dr, 1.0)
            if epi_total > 0.0
            else 0.0
        )
        sevo_dil = 0.28 * hill_function(state.mac_sevo, 0.8, 1.5) if state.mac_sevo > 0.0 else 0.0
        ket_dil = 0.32 * hill_function(state.ketamine_ce, 1.0, 1.5) if state.ketamine_ce > 0.0 else 0.0
        bronchodilation = 1.0 - (1.0 - epi_dil) * (1.0 - sevo_dil) * (1.0 - ket_dil)
        bronch = clamp01(bronch * (1.0 - bronchodilation))

    base_r = engine._base_airway_resistance + (
        airway_tuning.fona_resistance_add if state.airway_mode == AirwayType.FONA else 0.0
    )
    # Resistance diverges as the airway closes. The finite denominator keeps
    # the exact mechanics solver conditioned at complete occlusion.
    r_upper = (
        airway_tuning.upper_resistance_gain * upper_obstruction / max(1e-6, 1.0 - upper_obstruction)
        if upper_obstruction > 0.0
        else 0.0
    )
    r_bronch = (
        airway_tuning.bronch_resistance_gain * bronch * (
            1.0 + max(0.15 * (bronch ** 4), 0.85 * (engine.anaphylaxis.severity ** 1.5) * bronch * bronch)
        )
        if bronch > 0.0
        else 0.0
    )
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


def step_physiology(
    engine: SimulationEngine, dt: float, disturbances: DisturbanceEffects, *, collect_samples: bool = True,
) -> PhysiologyStepState:
    """Advance physiology models and return a projected runtime snapshot."""
    state = engine.state
    connected = circuit_connected(engine)
    vent_active = connected and engine.vent.is_on and engine.circuit.gas_supply.vent_drive_available
    bag_mask_active = engine.bag_mask_active and connected and not vent_active
    vent = engine.vent
    spirometry = vent.monitors
    hemo = engine.hemo
    ecmo = engine.extracorporeal
    if ecmo.is_active and not ecmo.is_cpb:
        ecmo.apply_closed_circuit_drainage_limit(
            hemo._calc_stressed_volume(), hemo.patient_blood_volume, BLOOD_VOLUME_MIN,
        )
    co_now = hemo.state.co
    dual_coupled = (
        dt > 0.1
        and (vent_active or bag_mask_active)
        and vent.mandatory_period_s is not None
        and (engine.resp.state.neural_rr > 0.0 or engine.resp_mech.effort.rr > 0.0)
    )
    if dual_coupled:
        n_sub = max(2, math.ceil(dt / 0.1))
        sub_dt = dt / n_sub
        step_gas_cache = engine._step_gas
        all_samples = []
        paw_area = effort_area = muscle_area = 0.0
        for _ in range(n_sub):
            engine._step_gas = step_gas_cache
            step_mechanics(engine, sub_dt, connected, vent_active, bag_mask_active, collect_samples=collect_samples)
            if collect_samples:
                all_samples.extend(vent.samples)
            paw_area += vent.alveolar_pressure_area
            effort_area += vent.capped_muscle_pressure_area
            muscle_area += vent.muscle_pressure_area
            resp_state = step_respiration(engine, sub_dt, co_now)
        engine._step_gas = None
        if collect_samples:
            vent.samples = all_samples
        vent.alveolar_pressure_area = paw_area
        vent.capped_muscle_pressure_area = effort_area
        vent.muscle_pressure_area = muscle_area
    else:
        step_mechanics(engine, dt, connected, vent_active, bag_mask_active, collect_samples=collect_samples)

    alpha_paw = 1.0 - math.exp(-dt / MEAN_PAW_TAU_S)
    esophageal = is_esophageal_intubation(engine)
    mean_paw = 0.0 if esophageal else (vent.alveolar_pressure_area / dt)
    engine.current_mean_paw = (1 - alpha_paw) * engine.current_mean_paw + alpha_paw * mean_paw
    alpha_effort = -math.expm1(-dt / MEAN_EFFORT_TAU_S)
    step_effort = min(vent.capped_muscle_pressure_area / dt, engine.hemo.config.effort_pressure_max)
    engine.current_mean_effort += alpha_effort * (step_effort - engine.current_mean_effort)

    pit_estimate, hemo_pit = compute_pit_estimates(engine, dt)

    applied_peep = engine.resp_mech.peep if (connected and not esophageal) else 0.0
    intrinsic_peep = (
        0.0
        if esophageal
        else (
            spirometry.auto_peep
            if connected
            else (
                max(0.0, engine.resp_mech.elastic_pressure)
                if not vent.inspiring and engine.resp_mech.effort.pmus() == (0.0, 0.0)
                else 0.0
            )
        )
    )
    total_peep_effect = applied_peep + intrinsic_peep
    # Systemic volatile cardiovascular and metabolic effects follow arterial MAC
    # (critical on CPB/VA ECMO with oxygenator delivery and in high pulmonary shunt).
    mac_sevo = engine.pk_sevo.state.p_art * 100.0 / engine.pk_sevo.mac_age
    mac_n2o = engine.pk_n2o.state.p_art * 100.0 / engine.pk_n2o.mac_age
    if not dual_coupled:
        resp_state = step_respiration(engine, dt, co_now)
    hemo_pre = hemo.state
    beta2_dr = _beta2_dose_ratio(engine)
    dist_spec = engine.disturbances.spec if engine.disturbance_active else None
    surg_stim = (
        1.0
        if dist_spec is not None and not dist_spec.finite
        else (0.5 if dist_spec is not None else 0.0)
    )
    hem_shock = max(
        clamp01((hemo._volume_deficit_fraction - 0.12) / 0.28),
        clamp01((60.0 - hemo_pre.perfusion_map) / 28.0) * (1.0 - 0.35 * clamp01(engine.thermal.depth_index)),
    )
    hypox_shock = clamp01((82.0 - resp_state.sao2) / 32.0)
    shock_stress = clamp01(max(hem_shock, hypox_shock))
    glucose_state = engine.glucose.step(
        dt,
        fluids=hemo.fluids,
        exo_insulin_cp=engine.pk["insulin"].state.c1,
        exo_epinephrine_ng_ml=state.epi_ce + engine.neuraxial.absorbed_epi_ng_ml,
        beta2_dose_ratio=beta2_dr,
        mac_sevo=mac_sevo,
        surgical_stimulation=surg_stim,
        sepsis_severity=hemo.sepsis_severity,
        cpb_active=ecmo.is_cpb,
        temp_c=state.temp_c,
        co_ratio=clamp(co_now / max(0.1, hemo.base_co_l_min), 0.1, 2.0),
        renal_perfusion=hemo.renal_perfusion,
        oxygen_supply=engine.resp.oxygen_supply_fraction,
        circulation=hemo.circulation_fraction(co_now),
        shock_stress=shock_stress,
    )
    hypo_epi = glucose_state.hypoglycemic_epi_ng_ml
    hypo_epi_cv = hypo_epi * (1.0 - 0.65 * clamp01(engine.thermal.depth_index))
    step_lactate(engine, dt)

    mean_cpb_pump = 0.0
    hct_now = hemo.get_hematocrit()
    blood_visc_ratio = math.exp(2.2 * (hct_now - 0.41)) * (1.0 + 0.025 * max(0.0, 37.0 - state.temp_c))
    if ecmo.is_cpb:
        cpg_out = ecmo.step_cardioplegia(
            dt,
            core_temp_c=state.temp_c,
            patient_k_meq_l=hemo.fluids.concentration("potassium"),
            patient_mg_mmol_l=hemo.fluids.serum_magnesium(),
            coronary_flow_reserve=hemo.coronary._reserve,
            viscosity_ratio=blood_visc_ratio,
        )
        if cpg_out.net_crystalloid_ml > 0.0:
            before_bv = hemo.blood_volume
            hemo.add_volume(
                cpg_out.net_crystalloid_ml,
                hematocrit=0.0,
                retention_fraction=1.0,
                label="crystalloid",
                suppress_tde=True,
            )
            added_cpg = max(0.0, hemo.blood_volume - before_bv)
            ecmo.state.reservoir_volume_ml = min(
                ecmo.config.cpb_reservoir_max_ml,
                ecmo.state.reservoir_volume_ml + added_cpg,
            )
        if cpg_out.net_k_meq > 0.0:
            hemo.fluids.add_kcl(cpg_out.net_k_meq)
        if cpg_out.net_mg_mmol > 0.0:
            hemo.fluids.give_magnesium_sulfate(
                cpg_out.net_mg_mmol, circulation=hemo.circulation_fraction(co_now), bolus_push_s=0.0,
            )
        if cpg_out.net_lidocaine_mg > 0.0:
            lido_pk = engine.pk["lidocaine"]
            lido_pk.state.c1 += cpg_out.net_lidocaine_mg / max(1.0, lido_pk.v1)
        if cpg_out.delivered_total_ml > 0.0:
            engine.thermal.add_fluid_heat_j(
                cpg_out.delivered_total_ml, cpg_out.infusate_temp_c, state.temp_c, is_blood=False,
            )
        if cpg_out.hot_shot_energy_boost > 0.0:
            hemo.coronary.energy_reserve = clamp01(hemo.coronary.energy_reserve + cpg_out.hot_shot_energy_boost)

    # Without flow, arterial pressure equals venous pressure, so coronary flow stops.
    step_coronary_from_state(engine, dt, hemo_pre, resp_state.sao2)

    if ecmo.is_cpb:
        stressed_ml = hemo._calc_stressed_volume()
        mcfp_now = stressed_ml / hemo.cv if hemo.cv > 0.0 else hemo.mcfp_0
        min_pat_bv = max(BLOOD_VOLUME_MIN, hemo.unstressed_volume * 0.80)
        _, mean_cpb_pump = ecmo.compute_cpb_reservoir_step(
            dt,
            stressed_ml,
            mcfp_now,
            hemo.patient_blood_volume,
            min_pat_bv,
            venous_compliance_ml_mmhg=hemo.cv,
        )
        hemo.extracorporeal_volume_ml = ecmo.total_circuit_mixing_volume_ml
    if ecmo.is_active:
        ecmo.update_line_pressures(hemo_pre.map, blood_visc_ratio)
    ecmo_art_flow = (
        mean_cpb_pump
        if ecmo.is_cpb
        else (ecmo.state.blood_flow_l_min if ecmo.is_arterial_support else 0.0)
    )
    ecmo_ven_drain = (
        ecmo.state.venous_drainage_l_min
        if ecmo.is_cpb
        else (ecmo_art_flow if ecmo.is_va else None)
    )
    hemo_paco2 = ecmo.state.upper_body_paco2_37 if ecmo.is_arterial_support else resp_state.pa_co2_37
    hemo_pao2 = ecmo.state.upper_body_pao2 if ecmo.is_arterial_support else resp_state.p_arterial_o2
    hemo_sao2 = ecmo.state.coronary_sao2 if ecmo.is_arterial_support else resp_state.sao2
    hemo.post_shock_asystole = engine.arrhythmias.post_shock_asystole
    # Plasma remifentanil and propofol (scaled by unbound fraction during CPB hemodilution)
    # reach hemodynamic effect sites inside the model; slower opioids enter at their own effect sites.
    remi_thermo_eff = hill_function(state.opioid_ce, engine.resp.c50_remi, engine.resp.gamma_remi)
    vc_threshold_c = engine.thermal.vasoconstriction_threshold(
        remi_thermo_eff, engine.neuraxial.vasoconst_threshold_shift_c,
    )
    nal_dr = naloxone_dose_ratio(engine)
    hemo_opioid = opioid_equivalent(state.remi_cp, state.fentanyl_ce, state.morphine_ce, state.hydromorphone_ce)
    hemo_remi_cp = state.remi_cp
    if nal_dr > 1.0:
        hemo_opioid /= nal_dr
        hemo_remi_cp /= nal_dr
    cardioplegia_now = ecmo.is_arterial_support and ecmo.state.cardioplegia_arrest
    pace_status = engine.pacemaker.evaluate(
        intrinsic_rhythm=hemo.rhythm_type,
        intrinsic_hr=hemo.intrinsic_rhythm_hr(),
        lap=hemo_pre.lap,
        ischemia=hemo.coronary.ischemia,
        hypoxia=hemo.myocardial_hypoxia,
        energy_failure=hemo.coronary.energy_failure,
        hyperkalemia_block=hemo.hyperkalemia_block,
        sodium_block=engine.toxicity.conduction_block,
        magnesium_block=hemo.magnesium_block,
        temp_c=state.temp_c,
        cardioplegia_arrest=cardioplegia_now,
        stimulus_tolerance=engine._tol_current,
    )
    hemo.set_pacing_capture(
        captured=pace_status.captured,
        mode=pace_status.mode,
        rate_bpm=pace_status.paced_hr,
        sv_multiplier=pace_status.sv_multiplier,
    )
    pace_discomfort = pace_status.discomfort_stress
    hemo_state = engine.hemo.step(
        dt,
        state.propofol_cp * engine.pk["propofol"].effect_input_fraction,
        hemo_opioid,
        state.nore_ce,
        pit=hemo_pit,
        paco2=hemo_paco2,
        pao2=hemo_pao2,
        dist_hr=disturbances.hr,
        dist_sv=disturbances.sv,
        dist_svr=disturbances.svr,
        crisis_hr=(
            engine.malignant_hyperthermia.heart_rate_increase
            + engine.toxicity.heart_rate_increase
            + engine.opioid_toxicity.heart_rate_increase
            + 18.0 * pace_discomfort
        ),
        crisis_svr=(
            engine.toxicity.svr_increase
            + engine.opioid_toxicity.svr_increase
            + 2.5 * pace_discomfort
        ),
        mac_sevo=mac_sevo,
        ce_epi=state.epi_ce + hypo_epi_cv + engine.neuraxial.absorbed_epi_ng_ml,
        ce_phenyl=state.phenyl_ce,
        ce_vaso=state.vaso_ce,
        ce_dobu=state.dobu_ce,
        ce_mil=state.mil_ce,
        ce_esmolol=state.esmolol_ce,
        ce_labetalol=state.labetalol_ce,
        ce_glyco=state.glyco_ce,
        ce_atropine=state.atropine_ce,
        ce_ephedrine=state.ephedrine_ce,
        ce_neostigmine=state.neostigmine_ce,
        ce_amiodarone=state.amiodarone_ce,
        ce_amiodarone_vasc=engine.pk["amiodarone"].state.ce_cv,
        ce_amiodarone_hr=engine.pk["amiodarone"].state.ce_hr,
        sodium_block=engine.toxicity.conduction_block,
        toxic_contractility_loss=engine.toxicity.contractility_loss,
        ce_ketamine=engine.pk["ketamine"].state.ce_cv,
        ce_ketamine_hr=engine.pk["ketamine"].state.ce_hr,
        temp_c=state.temp_c,
        peep_cmH2O=total_peep_effect,
        sao2=hemo_sao2,
        remifentanil_cp=hemo_remi_cp,
        ph=resp_state.ph,
        ph_37=resp_state.ph_37,
        ce_midazolam=effective_midazolam_ce(engine, state.midazolam_ce),
        ce_etomidate=state.etomidate_ce,
        mac_n2o=mac_n2o,
        shivering=state.shivering,
        ecmo_arterial_flow_l_min=ecmo_art_flow,
        ecmo_venous_drain_l_min=ecmo_ven_drain,
        lv_vent_l_min=ecmo.state.lv_vent_l_min if ecmo.is_arterial_support else 0.0,
        aortic_cross_clamp=ecmo.state.aortic_cross_clamp if ecmo.is_arterial_support else False,
        cardioplegia_arrest=ecmo.state.cardioplegia_arrest if ecmo.is_arterial_support else False,
        cpb_open_drainage=ecmo.is_cpb,
        cpb_reservoir_ml=ecmo.state.reservoir_volume_ml if ecmo.is_cpb else 0.0,
        protamine_svr_factor=engine.coagulation.state.protamine_svr_factor,
        protamine_venous_pool_frac=engine.coagulation.state.protamine_venous_pool_frac,
        protamine_pvr_factor=engine.coagulation.state.protamine_pvr_factor,
        embolic_pvr_factor=engine.embolism.pvr_factor,
        pneumo_svr_factor=engine.pneumoperitoneum.svr_factor,
        pneumo_pvr_factor=engine.pneumoperitoneum.pvr_factor,
        pneumo_rvr_factor=engine.pneumoperitoneum.rvr_factor,
        pneumo_mcfp_add=engine.pneumoperitoneum.mcfp_add_mmhg,
        pneumo_icp_add=engine.pneumoperitoneum.intracavitary_pressure_add_mmhg,
        neuraxial_svr_factor=engine.neuraxial.svr_factor,
        neuraxial_venous_pool_frac=engine.neuraxial.venous_pool_fraction,
        neuraxial_symp_vasc_block=engine.neuraxial.symp_vasc_block,
        neuraxial_cardioaccel_block=engine.neuraxial.cardioaccelerator_block,
        neuraxial_vasomotor_frac=engine.neuraxial.vasomotor_symp_fraction * engine.neuraxial.density,
        osmotic_diuresis_ml_min=glucose_state.osmotic_diuresis_ml_min,
        vasoconstriction_threshold_c=vc_threshold_c,
        toxic_lung_water_ml_min=engine.toxicity.vaso_lung_water_ml_min,
    )
    fluid_circulation = hemo.circulation_fraction(hemo_state.co)
    osmotic_plasma_ml = hemo.fluids.step(
        dt, ph=resp_state.ph_37, paco2=resp_state.pa_co2_37, resting_paco2=engine.resp.paco2_setpoint,
        renal_function=engine.patient.renal_function, renal_perfusion=hemo.renal_perfusion,
        hepatic_function=engine.patient.hepatic_function, oxygen_supply=engine.resp.oxygen_supply_fraction,
        epinephrine=(state.epi_ce + hypo_epi + engine.neuraxial.absorbed_epi_ng_ml) / beta2_dr,
        succinylcholine_ce=state.succinylcholine_ce,
        mh_severity=engine.malignant_hyperthermia.severity,
        circulation=fluid_circulation,
        lactate_acid_meq_l=engine.lactate.excess_endogenous_mmol_l,
        temp_c=state.temp_c,
        insulin_k_uptake=glucose_state.insulin_k_uptake,
    )
    if osmotic_plasma_ml != 0.0:
        hemo.apply_osmotic_plasma_shift(osmotic_plasma_ml)
    hypoxia, energy_failure, silent = myocardial_substrate(engine)
    cardioplegia_silent = ecmo.is_arterial_support and ecmo.state.cardioplegia_arrest
    perfusing = is_systemic_perfusing(engine, hemo_state)
    rhythm = engine.arrhythmias.step(
        dt, hemo.rhythm_type, ischemia=hemo.coronary.ischemia, hypoxia=hypoxia,
        energy_failure=energy_failure, electrically_silent=silent or cardioplegia_silent,
        perfusing=perfusing,
        lidocaine=engine.pk["lidocaine"].state.ce_cv, amiodarone=state.amiodarone_ce,
        hyperkalemia=hemo.hyperkalemia_arrhythmia,
        hypothermia=clamp01((29.5 - state.temp_c) / 3.5),
        energy_reserve=hemo.coronary.energy_reserve,
        compressions=hemo.compressions,
        hr=hemo_state.hr,
        beta_block=hemo._beta_occupancy,
        toxic_stress=engine.toxicity.cardiac_stress,
    )
    if rhythm != hemo.rhythm_type:
        hemo.rhythm_type = rhythm
        hemo_state = hemo.state
    if (
        pace_status.async_competition
        and hemo.rhythm_type not in (RhythmType.VFIB, RhythmType.ASYSTOLE)
        and engine._shock_rng.random() < engine.pacemaker.competition_vf_probability(dt, True)
    ):
        hemo.rhythm_type = RhythmType.VFIB
        hemo.set_pacing_capture(captured=False)
        hemo_state = hemo.state

    if ecmo.is_arterial_support and ecmo._last_pass is not None:
        engine.resp._update_extracorporeal_regional_gases(
            ecmo, ecmo._last_pass, hemo_state.pulmonary_flow, hemo.hb_conc, state.temp_c,
        )

    return build_snapshot_from_models(
        engine, hemo_state, resp_state, pit_estimate=pit_estimate,
    )


def myocardial_substrate(engine: SimulationEngine) -> tuple[float, float, bool]:
    """Myocardial hypoxia, energy failure, and electrical silence, including lidocaine, K+, and cardioplegia."""
    coronary, toxicity, hemo = engine.hemo.coronary, engine.toxicity, engine.hemo
    cardioplegia_silent = (
        engine.extracorporeal.is_arterial_support and engine.extracorporeal.state.cardioplegia_arrest
    )
    return (
        max(hemo.myocardial_hypoxia, toxicity.cardiac_stress),
        coronary.energy_failure,
        coronary.electrically_silent
        or toxicity.electrically_silent
        or hemo.hyperkalemia_block >= SILENT_BLOCK
        or hemo.magnesium_block >= SILENT_BLOCK
        or cardioplegia_silent,
    )


def _intrinsic_pulse_pressure(engine: SimulationEngine, hemo_state) -> float:
    return engine.arterial_waveform.reference_pulse_pressure(
        hemo_state.intrinsic_map,
        hemo_state.sv,
        hemo_state.hr,
        pulse_pressure_factor=hemo_state.pulse_pressure_factor,
        static_pressure=hemo_state.venous_pressure,
    )


def _is_pulseless(
    engine: SimulationEngine,
    hemo_state,
    *,
    hr: float | None = None,
    intrinsic_co: float | None = None,
    pulse_pressure: float | None = None,
) -> bool:
    """Whether the heart's own beats fail to generate effective systemic perfusion."""
    rhythm = hemo_state.rhythm_type
    eff_hr = hemo_state.hr if hr is None else hr
    eff_co = hemo_state.intrinsic_co if intrinsic_co is None else intrinsic_co
    if rhythm in PULSELESS_RHYTHMS or eff_hr < ARREST_HR_BPM:
        return True
    imap = hemo_state.intrinsic_map
    if imap < ARREST_MAP_MMHG and hemo_state.svr < 10.0:
        return True
    if eff_co > ARREST_CO_L_MIN and rhythm != RhythmType.VTACH:
        return False
    pp = _intrinsic_pulse_pressure(engine, hemo_state) if pulse_pressure is None else pulse_pressure
    no_pulse = pp < ARREST_PULSE_PRESSURE_MMHG
    if no_pulse and eff_co <= ARREST_CO_L_MIN:
        return True
    return rhythm == RhythmType.VTACH and (
        pp <= MIN_PULSE_PRESSURE_MMHG or (no_pulse and imap < PULSELESS_VT_MAP_MMHG)
    )


def is_perfusing(engine: SimulationEngine, hemo_state=None) -> bool:
    """Whether the heart's own beats produce effective systemic perfusion."""
    state = engine.hemo.state if hemo_state is None else hemo_state
    return not _is_pulseless(engine, state)


def is_systemic_perfusing(engine: SimulationEngine, hemo_state=None) -> bool:
    """Whether native beats OR arterial mechanical support (VA ECMO / CPB) perfuse the body."""
    state = engine.hemo.state if hemo_state is None else hemo_state
    if engine.extracorporeal.is_arterial_support and state.pump_flow > ARREST_CO_L_MIN:
        return True
    return not _is_pulseless(engine, state)


def step_lactate(engine: SimulationEngine, dt: float) -> None:
    """Advance lactate from this step's O2 balance and update extracellular buffer base."""
    if dt <= 0.0:
        return
    resp = engine.resp
    hemo = engine.hemo
    hemo_state = hemo.state
    fluids = hemo.fluids
    engine.lactate.update_fluid_volume(fluids.lactate_volume_l, fluids.consume_lactate_loss_l())
    beta2_dr = _beta2_dose_ratio(engine)
    # Microcirculatory maldistribution during severe vasoplegic hypotension
    # (perfusion MAP < 40 mmHg) generates anaerobic lactate even when global DO2 > VO2.
    perf_map = max(0.0, hemo_state.perfusion_map)
    microcirc_deficit_frac = clamp01((40.0 - perf_map) / 25.0) ** 1.5
    extra_deficit_frac = max(0.0, 0.65 * microcirc_deficit_frac - (1.0 - resp.oxygen_supply_fraction))
    vo2_l_min = resp.vco2 * engine.thermal.metabolic_factor / resp.rq / 1000.0
    effective_deficit_l = resp.oxygen_deficit_l + extra_deficit_frac * vo2_l_min * (dt / 60.0)
    effective_supply = min(resp.oxygen_supply_fraction, 1.0 - 0.65 * microcirc_deficit_frac)
    circulation = hemo.circulation_fraction(hemo_state.co)
    base_removed = engine.lactate.step(
        dt, effective_deficit_l, effective_supply,
        epinephrine_ng_ml=(
            engine.state.epi_ce + engine.glucose.state.hypoglycemic_epi_ng_ml + engine.neuraxial.absorbed_epi_ng_ml
        ) / beta2_dr,
        sepsis_severity=hemo.sepsis_severity,
        infused_lactate_mmol=fluids.infused_l_lactate_mmol,
        convulsion_intensity=engine.toxicity.convulsion_intensity,
        mh_severity=engine.malignant_hyperthermia.severity,
        circulation=circulation,
        temp_c=engine.state.temp_c,
        extra_rate_mmol_kg_min=engine.toxicity.vaso_lactate_mmol_kg_min,
    )
    resp.oxygen_deficit_l = 0.0
    no_pulm_flow = engine.extracorporeal.is_active and hemo_state.pulmonary_flow <= 0.0
    venous_only_circ = circulation <= 0.02
    if base_removed != 0.0:
        resp.remove_buffer_base(
            base_removed,
            venous_only=(venous_only_circ or (base_removed > 0.0 and resp.oxygen_supply_fraction < 0.99)),
            exclude_pulmonary=no_pulm_flow,
        )
    if engine.lactate.last_cleared_infused_mmol > 0.0:
        resp.add_organic_carbon(
            0.85 * engine.lactate.last_cleared_infused_mmol,
            venous_only=venous_only_circ,
            exclude_pulmonary=no_pulm_flow,
        )


def check_cardiac_arrest(engine: SimulationEngine, dt: float) -> None:
    """Flag arrest after 15 s without effective pulses (suppressed when VA ECMO/CPB maintains perfusion)."""
    state = engine.state
    hemo_state = engine.hemo.state
    if engine.extracorporeal.is_arterial_support and hemo_state.pump_flow > ARREST_CO_L_MIN:
        engine._pulseless_s = 0.0
        state.cardiac_arrest = False
        state.arrest_reason = ""
        state.alarms.pop("Arrest", None)
        return
    rhythm = hemo_state.rhythm_type
    # Use deterministic intrinsic reference pulse pressure rather than monitor
    # waveform landmarks so arrest detection is identical across update_monitors modes.
    # Respect direct state overrides in unit tests that mutate engine.state without stepping hemo.
    state_overridden = (
        abs(state.intrinsic_co - hemo_state.intrinsic_co) > 1e-6
        or abs(state.hr - hemo_state.hr) > 1e-6
    )
    if state_overridden and not state.chest_compressions:
        eff_hr = state.hr
        eff_co = state.intrinsic_co
        pulse_pressure = max(0.0, state.sbp - state.dbp)
    else:
        eff_hr = state.hr if state_overridden else hemo_state.hr
        eff_co = state.intrinsic_co if state_overridden else hemo_state.intrinsic_co
        pulse_pressure = _intrinsic_pulse_pressure(engine, hemo_state)
    pulseless = _is_pulseless(
        engine,
        hemo_state,
        hr=eff_hr,
        intrinsic_co=eff_co,
        pulse_pressure=pulse_pressure,
    )
    if state.cardiac_arrest and not pulseless:
        # Small recovery hysteresis prevents arrest flag flapping right at boundary CO/PP.
        if (
            eff_hr < ARREST_HR_BPM + 5.0
            or (pulse_pressure < ARREST_PULSE_PRESSURE_MMHG + 2.0 and eff_co <= ARREST_CO_L_MIN + 0.15)
            or (hemo_state.intrinsic_map < ARREST_MAP_MMHG + 5.0 and hemo_state.svr < 10.0)
        ):
            pulseless = True
    engine._pulseless_s = engine._pulseless_s + dt if pulseless else 0.0
    if not pulseless:
        state.cardiac_arrest = False
        state.arrest_reason = ""
        state.alarms.pop("Arrest", None)
        return
    if engine._pulseless_s < ARREST_CONFIRM_S:
        return

    if rhythm in PULSELESS_RHYTHMS:
        reason = rhythm.value
    elif rhythm == RhythmType.VTACH:
        reason = "Pulseless ventricular tachycardia"
    elif eff_hr < ARREST_HR_BPM:
        reason = f"Extreme bradycardia (HR < {ARREST_HR_BPM:g} bpm)"
    else:
        reason = "Pulseless electrical activity (negligible forward flow)"
    changed = not state.cardiac_arrest or state.arrest_reason != reason
    state.cardiac_arrest = True
    state.arrest_reason = reason
    if rhythm not in (RhythmType.ASYSTOLE, RhythmType.VFIB, RhythmType.VTACH):
        state.alarms["Arrest"] = {"low": False, "high": True}
    if changed:
        logger.warning("Cardiac arrest: %s (MAP=%.1f mmHg, HR=%.1f bpm)", reason, state.map, state.hr)
