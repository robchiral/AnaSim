from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anasim.patient.pd import hypnotic_equivalent, opioid_equivalent
from anasim.patient.pd.anesthesia import (
    LARYNGOSCOPY_INTENSITY,
    PROPOFOL_LOC_TRANSFER,
    opioid_controller_fraction,
    opioid_rate_fraction,
)
from anasim.patient.pd.nmba import succinylcholine_twitch_fraction
from anasim.physiology.malignant_hyperthermia import dantrolene_twitch_fraction

from .drug_registry import DRUG_REGISTRY
from .state import AirwayType, SimulationState
from .utils import clamp, clamp01, hill_function

if TYPE_CHECKING:
    from anasim.monitors.arterial import ArterialPressureSample
    from anasim.physiology.hemo_types import HemoState
    from anasim.physiology.respiration import RespState

    from .engine import SimulationEngine


@dataclass(slots=True)
class PhysiologyStepState:
    hemo_state: HemoState
    resp_state: RespState
    pit_estimate: float
    rr_display: float
    vt_display_ml: float
    mv_display_l_min: float
    paw_display: float
    flow_display: float
    volume_display: float
    paw_peak: float
    paw_plat: float
    paw_mean: float
    peep: float
    compliance_dyn: float


def circuit_connected(engine: SimulationEngine) -> bool:
    """Whether gas can cross the Y-piece between the patient and circuit."""
    mode = engine.state.airway_mode
    return mode != AirwayType.NONE and (mode != AirwayType.MASK or engine.mask_seal > 0.0)


def assisted_positive_pressure(engine: SimulationEngine) -> bool:
    """Whether positive-pressure tidal or support breaths are active."""
    return (engine.vent.is_on and engine.vent.settings.mode != "CPAP") or engine.bag_mask_active


def mask_circuit_share(engine: SimulationEngine) -> float:
    """Fraction of inspired or exchanged gas carried by the breathing circuit."""
    mode = engine.state.airway_mode
    if mode == AirwayType.NONE:
        return 0.0
    if mode != AirwayType.MASK:
        return 1.0
    seal = engine.mask_seal
    if seal <= 0.0:
        return 0.0
    # Room air enters around a loose mask only during unsupported inspiration.
    # A pressurized mask leaks outward.
    spirometry = engine.vent.monitors
    supported = spirometry.mv_supported * (1.0 - mask_leak(engine))
    unsupported = max(0.0, spirometry.mv_exp - spirometry.mv_supported)
    if supported + unsupported <= 0.0:
        pressurized = assisted_positive_pressure(engine) or (engine.vent.is_on and engine.vent.settings.peep > 0.0)
        return 1.0 if pressurized else seal
    return (supported + seal * unsupported) / (supported + unsupported)


def mask_leak(engine: SimulationEngine) -> float:
    """Fraction of an assisted breath lost around a mask or into the stomach.

    Leakage applies only to positive-pressure breaths on a facemask, from an
    imperfect mask seal, upper airway obstruction, or peak pressure above the
    seal threshold.
    """
    if not assisted_positive_pressure(engine) or engine.state.airway_mode != AirwayType.MASK:
        return 0.0
    seal = engine.mask_seal
    if seal <= 0.0:
        return 1.0
    paw_peak = engine.vent.monitors.paw_peak
    threshold = engine.airway_tuning.mask_seal_pressure * seal
    overpressure = engine.airway_tuning.mask_overpressure_leak * max(0.0, paw_peak - threshold) / max(paw_peak, 1.0)
    kept = seal * engine.airway_status.patency * (1.0 - overpressure)
    return 1.0 - clamp01(kept)


def delivered_fraction(engine: SimulationEngine) -> float:
    """Fraction of measured circuit volume and pressure that reaches the patient."""
    if engine.state.airway_mode != AirwayType.MASK:
        return 1.0
    seal = engine.mask_seal
    spirometry = engine.vent.monitors
    if spirometry.mv_exp <= 0.0:
        return 1.0 - mask_leak(engine) if assisted_positive_pressure(engine) else seal
    # Supported breaths leak under pressure; unsupported breaths split at the seal.
    supported = min(1.0, spirometry.mv_supported / spirometry.mv_exp)
    return supported * (1.0 - mask_leak(engine)) + (1.0 - supported) * seal


def circuit_ventilation(engine: SimulationEngine) -> tuple[float, float]:
    """Return (RR, mean VT L) of all recent measured breaths that reach the lungs."""
    spirometry = engine.vent.monitors
    rr = spirometry.rr_total
    if rr <= 0.0:
        return 0.0, 0.0
    leak = mask_leak(engine)
    mv_lung = spirometry.mv_exp * (1.0 - leak)
    resp = engine.resp.state
    if leak > 0.0 and engine.mask_seal < 1.0 and resp.rr > 0.0:
        # Unassisted breaths draw room air around a loose mask, so the
        # leak of assisted breaths does not limit them.
        unassisted_mv = resp.mv / engine.airway_status.ventilation_efficiency
        if unassisted_mv > mv_lung:
            return resp.rr, unassisted_mv / resp.rr
    return rr, mv_lung / rr


def measured_ventilation(
    engine: SimulationEngine,
    connected: bool,
    spontaneous_rr: float,
    spontaneous_vt_l: float,
) -> tuple[float, float, float]:
    """Return (RR, exhaled VT mL, MV L/min) as the workstation measures them.

    Assisted VT comes from the last completed breath; MV uses recent exhaled
    breaths. Both exclude mask leak. The respiratory model applies the
    bronchospasm efficiency loss to alveolar ventilation only.
    """
    if connected:
        if not circuit_connected(engine):
            return 0.0, 0.0, 0.0
        spirometry = engine.vent.monitors
        kept = delivered_fraction(engine)
        # Before the first completed breath, the respiratory model supplies
        # the baseline rate; no tidal volume has yet been measured.
        rr = spirometry.rr_total if engine.vent.has_measured_breath else spontaneous_rr
        return rr, spirometry.tv_exp * kept, spirometry.mv_exp * kept
    # Impedance counts muscle effort even when obstruction prevents airflow.
    return engine.resp.state.effort_rr, spontaneous_vt_l * 1000.0, spontaneous_rr * spontaneous_vt_l


def loss_of_response(engine: SimulationEngine, *, ketamine: bool = True) -> float:
    """Probability of no response to shaking or shouting at the current effect sites.

    Airway tone excludes ketamine, which preserves it.
    """
    state = engine.state
    hypnotic = hypnotic_equivalent(
        PROPOFOL_LOC_TRANSFER * engine.pk["propofol"].state.ce_response, engine.pk["etomidate"].state.ce,
        engine.pk["midazolam"].state.ce, engine.midazolam_c50, engine.response.age_factor,
    )
    drug_response = engine.response.loss_of_response(
        hypnotic,
        state.opioid_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
        ce_ketamine=state.ketamine_ce if ketamine else 0.0,
    )
    return (1.0 - (1.0 - drug_response) * (1.0 - engine.cerebral.unresponsiveness)
            * (1.0 - engine.toxicity.unresponsiveness))


def stimulus_tolerance(engine: SimulationEngine, intensity: float = LARYNGOSCOPY_INTENSITY) -> float:
    """Probability of no response to a noxious stimulus of the given Bouillon intensity; laryngoscopy by default."""
    state = engine.state
    return engine.response.tolerance(
        state.hypnotic_response_ce,
        state.opioid_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
        ce_ketamine=state.ketamine_ce,
        ce_lidocaine=state.lidocaine_ce,
        intensity=intensity,
    )


def central_muscle_strength(engine: SimulationEngine) -> float:
    """Remaining central respiratory/airway strength after either muscle blocker."""
    block = hill_function(engine.tof_pd.ce_central, engine.resp.c50_nmba, engine.resp.gamma_nmba)
    return clamp01(1.0 - block) * succinylcholine_twitch_fraction(
        engine.state.succinylcholine_ce, central=True,
    )


def cardiovascular_response_fractions(engine: SimulationEngine, intensity: float) -> tuple[float, float]:
    """Pressure and HR stimulus fractions, separate from movement and EEG arousal.

    The fractions are calibrated at steady propofol concentrations, so propofol
    enters at its BIS effect site; the faster response site reads about twice as
    high early after induction and would block the intubation response.
    """
    state = engine.state
    return engine.response.cardiovascular_response_fractions(
        state.hypnotic_ce,
        opioid_equivalent(engine.pk["remi"].state.ce, 0.0, state.morphine_ce, state.hydromorphone_ce),
        mac_sevo=state.mac_sevo, mac_n2o=state.mac_n2o,
        ce_ketamine=state.ketamine_ce, ce_lidocaine=state.lidocaine_ce,
        intensity=intensity, ce_fentanyl=state.fentanyl_ce,
    )


# Built-in floats keep public state JSON-serializable.
def sync_pk_state(engine: SimulationEngine) -> None:
    """Synchronize PK concentrations from subsystem states to public state."""
    state = engine.state
    for spec in DRUG_REGISTRY:
        pk_state = engine.pk[spec.key].state
        if spec.ce_field is not None:
            setattr(state, spec.ce_field, float(pk_state.ce))
        if spec.cp_field is not None:
            setattr(state, spec.cp_field, float(pk_state.c1))
    state.roc_ce = float(engine.tof_pd.ce)
    state.roc_cp = float(engine.pk["roc"].state.c1 + engine.tof_pd.bound_rocuronium())
    state.opioid_ce = float(opioid_equivalent(
        engine.pk["remi"].state.ce, state.fentanyl_ce, state.morphine_ce, state.hydromorphone_ce,
    ))
    state.opioid_cp = float(opioid_equivalent(
        engine.pk["remi"].state.c1, state.fentanyl_cp, state.morphine_cp, state.hydromorphone_cp,
    ))
    sync_neuromuscular_state(engine)
    state.hypnotic_ce = float(hypnotic_equivalent(
        engine.pk["propofol"].state.ce,
        engine.pk["etomidate"].state.ce,
        engine.pk["midazolam"].state.ce,
        engine.midazolam_c50,
        engine.bis.age_factor,
    ))
    state.hypnotic_response_ce = float(hypnotic_equivalent(
        engine.pk["propofol"].state.ce_response, engine.pk["etomidate"].state.ce,
        engine.pk["midazolam"].state.ce, engine.midazolam_c50,
    ))


def sync_neuromuscular_state(engine: SimulationEngine) -> None:
    """T1 height includes phase I block, which depresses all four twitches without fade."""
    s = engine.state
    s.tof = float(engine.tof_pd.compute_tof_from_ce(s.roc_ce, s.mac_sevo, s.mac_n2o))
    roc_twitch = 1.0 - engine.tof_pd.twitch_block(s.roc_ce, s.mac_sevo, s.mac_n2o)
    s.twitch = (100.0 * roc_twitch * succinylcholine_twitch_fraction(s.succinylcholine_ce)
                * dantrolene_twitch_fraction(s.dantrolene_ce))


def sync_inspired_gas(engine: SimulationEngine) -> tuple[float, float]:
    """Project inspired gas at the airway and return (Fi sevo, Fi N2O) as fractions."""
    state = engine.state
    composition = engine.circuit.composition
    circuit_share = mask_circuit_share(engine)
    fio2 = 0.21 + circuit_share * (composition.fio2 - 0.21)
    fi_sevo = circuit_share * composition.fi_agent
    fi_n2o = circuit_share * composition.fin2o
    state.fio2 = float(fio2)
    state.fi_sevo = float(fi_sevo * 100.0)
    state.fi_n2o = float(fi_n2o * 100.0)
    return fi_sevo, fi_n2o


def sync_inhaled_agents(engine: SimulationEngine) -> None:
    """Project end-tidal and brain MAC values for sevoflurane and nitrous oxide."""
    state = engine.state
    sevo, n2o = engine.pk_sevo.state, engine.pk_n2o.state
    state.et_sevo = float(sevo.p_alv * 100.0)
    state.et_n2o = float(n2o.p_alv * 100.0)
    state.mac_sevo = float(sevo.mac)
    state.mac_n2o = float(n2o.mac)
    state.mac = float(sevo.mac + n2o.mac)
    state.et_mac = float(sevo.p_alv * 100.0 / engine.pk_sevo.mac_age + n2o.p_alv * 100.0 / engine.pk_n2o.mac_age)


def project_arterial(
    state: SimulationState, arterial_sample: ArterialPressureSample, art_reading: ArterialPressureSample,
) -> None:
    """Copy the ideal pulse landmarks and the catheter reading into the public snapshot."""
    state.sbp = float(arterial_sample.systolic)
    state.dbp = float(arterial_sample.diastolic)
    state.art_pressure = float(art_reading.pressure)
    state.art_sbp = float(art_reading.systolic)
    state.art_dbp = float(art_reading.diastolic)
    state.art_map = float(art_reading.mean)


def project_hemodynamics(engine: SimulationEngine, hemo_state: HemoState) -> None:
    """Copy hemodynamic model state into the public snapshot."""
    state = engine.state
    total_colloid = engine.hemo.total_colloid_in_ml
    blood_in_ml = engine.hemo.total_blood_in_ml
    urine_out_ml = engine.hemo.total_urine_out_ml
    blood_out_ml = engine.hemo.total_blood_out_ml
    fluid_in_ml = engine.hemo.total_crystalloid_in_ml + total_colloid
    state.map = float(hemo_state.map)
    state.hr = float(hemo_state.hr)
    state.sv = float(hemo_state.sv)
    state.svr = float(hemo_state.svr)
    state.co = float(hemo_state.co)
    state.intrinsic_co = float(hemo_state.intrinsic_co)
    state.chest_compressions = hemo_state.compressions
    state.blood_volume = float(engine.hemo.blood_volume)
    state.hb_g_dl = float(engine.hemo.hb_conc)
    state.hct = float(engine.hemo.get_hematocrit())
    state.colloid_in_ml = float(total_colloid)
    state.fluid_in_ml = float(fluid_in_ml)
    state.blood_in_ml = float(blood_in_ml)
    state.urine_out_ml = float(urine_out_ml)
    state.blood_out_ml = float(blood_out_ml)
    state.net_fluid_ml = float(fluid_in_ml + blood_in_ml - urine_out_ml - blood_out_ml)
    state.lap = float(hemo_state.lap)
    state.myocardial_ischemia = float(engine.hemo.coronary.ischemia)
    state.coronary_supply_demand_ratio = float(engine.hemo.coronary.supply_demand_ratio)
    state.st_depression_mv = float(engine.hemo.coronary.st_depression_mv)
    state.lung_water = float(engine.hemo.lung_water_ml_kg)


def _project_respiratory_observables(engine: SimulationEngine, snapshot: PhysiologyStepState) -> None:
    """Copy respiratory fields shared by startup sync and runtime projection."""
    state = engine.state
    resp_state = snapshot.resp_state
    sealed = circuit_connected(engine)
    state.rr = float(snapshot.rr_display)
    state.vt = float(snapshot.vt_display_ml)
    state.mv = float(snapshot.mv_display_l_min)
    state.va = float(resp_state.va)
    state.pa_co2 = float(resp_state.pa_co2)
    state.ph = float(resp_state.ph)
    state.hco3 = float(resp_state.hco3)
    state.base_excess = float(resp_state.base_excess)
    state.lactate = float(engine.lactate.concentration)
    fluids = engine.hemo.fluids
    state.sodium = fluids.concentration("sodium")
    state.potassium = fluids.concentration("potassium")
    state.chloride = fluids.concentration("chloride")
    state.ionized_calcium = fluids.ionized_calcium(resp_state.ph)
    state.albumin_g_l = fluids.plasma_buffers.albumin_g_l
    state.phosphate = fluids.concentration("phosphate")
    state.alveolar_co2 = float(resp_state.p_alveolar_co2)
    state.pao2 = float(resp_state.p_arterial_o2)
    state.sao2 = float(resp_state.sao2)
    state.spo2 = float(resp_state.sao2)
    state.etco2 = float(resp_state.etco2 if sealed else 0.0)
    state.et_o2 = float(resp_state.eto2 if sealed else 0.0)
    state.pit = float(snapshot.pit_estimate)
    state.paw = float(snapshot.paw_display)
    state.flow = float(snapshot.flow_display)
    state.volume = float(snapshot.volume_display)
    state.paw_peak = float(snapshot.paw_peak)
    state.paw_plat = float(snapshot.paw_plat)
    state.paw_mean = float(snapshot.paw_mean)
    state.peep = float(snapshot.peep)
    state.compliance_dyn = float(snapshot.compliance_dyn)
    state.apnea = bool(resp_state.apnea)


def step_respiration(engine: SimulationEngine, dt: float, cardiac_output: float) -> RespState:
    """Advance the respiratory model; dt 0 evaluates the current state without advancing it."""
    state = engine.state
    engine.resp.update_fluid_balance(engine.hemo.hb_conc, engine.hemo.get_hematocrit(), engine.hemo.blood_volume)
    sealed = circuit_connected(engine)
    # Gas exchange uses recent exhaled breaths, so it lags at least one breath.
    assisted_rr, assisted_vt_l = circuit_ventilation(engine) if sealed else (0.0, 0.0)
    airway = engine.airway_status
    peep_scale = engine.mask_seal if state.airway_mode == AirwayType.MASK else 1.0
    return engine.resp.step(
        dt,
        ce_prop=hypnotic_equivalent(
            engine.pk["propofol"].state.ce_resp,
            state.etomidate_ce,
            state.midazolam_ce,
            engine.midazolam_c50,
            ventilation=True,
        ),
        ce_remi=opioid_equivalent(engine.pk["remi"].state.ce_resp, state.fentanyl_ce),
        opioid_controller_fraction=opioid_controller_fraction(
            engine.pk["morphine"].state.ce_resp, engine.pk["hydromorphone"].state.ce_resp,
        ),
        opioid_rate_fraction=opioid_rate_fraction(
            engine.pk["morphine"].state.ce_rr, engine.pk["hydromorphone"].state.ce_rr,
            variability=engine.patient.variability.opioid_rate,
        ),
        mech_vent_mv=assisted_rr * assisted_vt_l,
        fio2=state.fio2,
        ce_roc=engine.tof_pd.ce_central,
        neuromuscular_strength=(succinylcholine_twitch_fraction(state.succinylcholine_ce, central=True)
                               * (1.0 - engine.resp_mech.effort.muscle.fatigue)),
        mac_sevo=state.mac_sevo,
        mech_rr=assisted_rr,
        mech_vt_l=assisted_vt_l,
        measured_breaths=sealed and engine.vent.has_measured_breath,
        airway_patency=airway.patency,
        ventilation_efficiency=airway.ventilation_efficiency,
        vq_mismatch=airway.vq_mismatch,
        hb_g_dl=engine.hemo.hb_conc,
        hematocrit=engine.hemo.get_hematocrit(),
        blood_volume_ml=engine.hemo.blood_volume,
        cardiac_output=cardiac_output,
        metabolic_factor=max(0.5, engine.thermal.metabolic_factor),
        unconscious=state.loc,
        lung_volume_l=engine.aeration.frc,
        shunt_fraction=engine.aeration.shunt_fraction,
        temp_c=state.temp_c,
        airway_pressure=engine.resp_mech.peep * peep_scale,
        # Brainstem drive persists until cerebral failure is nearly complete.
        central_drive_fraction=(1.0 - engine.cerebral.dysfunction ** 8) * engine.toxicity.respiratory_drive,
    )


def build_snapshot_from_models(
    engine: SimulationEngine, hemo_state: HemoState, resp_state: RespState, *, pit_estimate: float,
) -> PhysiologyStepState:
    """Build the respiratory projection shared by initialization and runtime."""
    connected = engine.state.airway_mode != AirwayType.NONE
    sealed = circuit_connected(engine)
    assisted = sealed and (engine.vent.is_on or engine.bag_mask_active)
    rr_display, vt_display_ml, mv_display_l_min = measured_ventilation(
        engine, connected, resp_state.rr, resp_state.vt / 1000.0
    )
    vent = engine.vent
    spirometry = vent.monitors
    return PhysiologyStepState(
        hemo_state=hemo_state,
        resp_state=resp_state,
        pit_estimate=pit_estimate,
        rr_display=rr_display,
        vt_display_ml=vt_display_ml,
        mv_display_l_min=mv_display_l_min,
        paw_display=vent.paw if sealed else 0.0,
        flow_display=vent.flow if sealed else 0.0,
        volume_display=vent.volume if sealed else 0.0,
        paw_peak=spirometry.paw_peak if sealed else 0.0,
        paw_plat=spirometry.paw_plat if assisted else math.nan,
        paw_mean=spirometry.paw_mean if sealed else 0.0,
        peep=spirometry.peep if assisted else 0.0,
        compliance_dyn=spirometry.compliance_dyn if assisted else math.nan,
    )


def project_runtime_physiology(engine: SimulationEngine, snapshot: PhysiologyStepState) -> None:
    """Project a runtime physiology step back into the public SimulationState."""
    state = engine.state
    project_hemodynamics(engine, snapshot.hemo_state)
    _project_respiratory_observables(engine, snapshot)
    state.oxygen_delivery_ratio = float(engine.hemo.compute_do2_ratio(
        max(0.0, state.sao2) / 100.0, max(0.0, state.pao2), state.co
    ))


def sync_monitor_baselines(engine: SimulationEngine) -> None:
    """Derive monitor baselines from the current physiologic snapshot."""
    state = engine.state
    bis_val = clamp(engine.bis.compute_bis(state.hypnotic_ce, state.mac_sevo), 0.0, 100.0)
    loc_val = loss_of_response(engine)
    tol_val = stimulus_tolerance(engine)
    engine._tol_current = tol_val
    cardiac_sample = engine.cardiac_cycle.seed(state.hr, engine.hemo.state.rhythm_type)
    arterial_sample = engine.arterial_waveform.step(
        cardiac_sample,
        state.map,
        state.sv,
        pulse_pressure_factor=engine.hemo.state.pulse_pressure_factor,
    )
    art_reading = engine.art_line.seed(arterial_sample)
    engine.airway_sensor.seed(state.paw, state.flow, state.volume)
    state.bis = float(bis_val)
    state.loc = float(loc_val)
    state.tol = float(tol_val)
    state.capno_co2 = state.ecg_voltage = state.pleth_voltage = 0.0
    project_arterial(state, arterial_sample, art_reading)
    state.display_hr = float(cardiac_sample.display_hr)
    state.display_bis = float(bis_val)
    engine.etco2_readout.seed(state.etco2)
    state.display_etco2 = float(state.etco2)
    state.display_spo2 = float(state.spo2)
    engine.bis.initialize(state.bis)
    engine.smooth_bis = state.bis


def sync_state_from_models(engine: SimulationEngine) -> None:
    """Derive the public SimulationState from current subsystem state."""
    state = engine.state
    state.temp_c = float(engine.patient.baseline_temp)
    sync_inspired_gas(engine)
    sync_inhaled_agents(engine)
    sync_pk_state(engine)
    hemo_state = engine.hemo.state
    resp_state = step_respiration(engine, 0.0, hemo_state.co)
    project_runtime_physiology(engine, build_snapshot_from_models(
        engine, hemo_state, resp_state, pit_estimate=engine.hemo.config.pit_0,
    ))
    sync_monitor_baselines(engine)
