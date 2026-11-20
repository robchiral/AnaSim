from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anasim.patient.pd import hypnotic_equivalent, opioid_equivalent
from anasim.patient.pd.anesthesia import (
    ETOMIDATE_LOSS_OF_RESPONSE,
    KETAMINE_MAC_EQUIVALENT,
    LARYNGOSCOPY_INTENSITY,
    PROPOFOL_LOC_TRANSFER,
    PROPOFOL_LOSS_OF_RESPONSE,
    opioid_controller_fraction,
    opioid_rate_fraction,
)
from anasim.patient.pd.nmba import succinylcholine_twitch_fraction
from anasim.physiology.malignant_hyperthermia import dantrolene_twitch_fraction
from anasim.physiology.resuscitation import compression_pressures

from .constants import CMH2O_TO_MMHG
from .drug_registry import DRUG_REGISTRY
from .enums import RhythmType
from .state import AirwayType, SimulationState
from .utils import clamp, clamp01

if TYPE_CHECKING:
    from anasim.machine.circuit import GasComposition
    from anasim.monitors.arterial import ArterialPressureSample
    from anasim.physiology.hemo_types import HemoState
    from anasim.physiology.respiration import RespState

    from .engine import SimulationEngine


@dataclass(slots=True)
class StepGasCache:
    tracheal_fio2: float
    n2o_uptake_l_min: float
    lung_volume_l: float
    sevo_uptake_l_min: float = 0.0
    circuit_fin2: float | None = None
    apneic_limit_l_min: float | None = None
    reservoir_volume_l: float | None = None
    bellows_fraction: float | None = None


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


def mask_seal_factor(engine: SimulationEngine) -> float:
    """Mask pressure and volume transmission factor; 1.0 when not using a facemask."""
    return engine.mask_seal if engine.state.airway_mode == AirwayType.MASK else 1.0


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
    # Room air enters around a loose mask during unsupported inspiration or
    # whenever positive-pressure delivery falls short of spontaneous demand.
    spirometry = engine.vent.monitors
    supported = spirometry.mv_supported * (1.0 - mask_leak(engine))
    unsupported = max(0.0, spirometry.mv_exp - spirometry.mv_supported)
    spont_mv = engine.resp.state.mv if not engine.resp.state.apnea else 0.0
    if not engine.vent.has_measured_breath and supported + unsupported <= 0.0:
        pressurized = (
            (
                (assisted_positive_pressure(engine) or (engine.vent.is_on and engine.vent.settings.peep > 0.0))
                and engine.circuit.bellows_fraction > 0.0
            )
            or engine._apl_pressure > 0.0
        )
        if pressurized and seal < 1.0 and spont_mv > 0.0 and assisted_positive_pressure(engine):
            if engine.bag_mask_active and not engine.vent.is_on:
                est_sup = engine.bag_mask_rr * engine.bag_mask_vt
            elif engine.vent.is_on and engine.vent.settings.rr > 0.0:
                est_sup = engine.vent.settings.rr * engine.vent.settings.tv / 1000.0
            else:
                est_sup = 0.0
            est_delivered = est_sup * (1.0 - mask_leak(engine))
            unassisted_est = max(0.0, spont_mv - est_delivered)
            total_est = est_delivered + unassisted_est
            share = (est_delivered + seal * unassisted_est) / total_est if total_est > 0.0 else seal
        else:
            share = 1.0 if pressurized else seal
    else:
        unassisted = max(unsupported, spont_mv - supported) if seal < 1.0 else unsupported
        total_mv = supported + unassisted
        share = (supported + seal * unassisted) / total_mv if total_mv > 0.0 else seal
    if seal < 0.999 and engine.circuit.bellows_fraction <= 0.05:
        avail_stpd = max(0.0, engine.circuit.fgf_total() + engine.circuit.vapor_flow())
        if engine.state.va > 0.0:
            avail_btps = avail_stpd / engine.resp._btps_to_stpd
            demand_btps = max(engine.state.va, spont_mv)
            share = min(share, avail_btps / max(0.1, demand_btps))
        else:
            resp = engine.resp
            apneic_stpd = max(0.0, resp._pulmonary_o2_uptake - resp._pulmonary_co2_delivery)
            share = min(share, avail_stpd / max(0.05, apneic_stpd)) if apneic_stpd > 1e-6 else 0.0
    return share


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


def _circuit_delivery_limits(engine: SimulationEngine, rr: float) -> tuple[float, float, bool]:
    """Return (leak_fraction, max_deliverable_mv_l_min, sealed_airway) for circuit breaths."""
    cache = engine._step_gas
    reservoir_vol = (
        cache.reservoir_volume_l
        if cache is not None and cache.reservoir_volume_l is not None
        else engine.circuit.reservoir_volume
    )
    max_deliverable_mv = (
        (
            reservoir_vol * rr
            + max(0.0, engine.circuit.fgf_total() + engine.circuit.vapor_flow())
        )
        / engine.resp._btps_to_stpd
    )
    sealed_airway = engine.state.airway_mode != AirwayType.MASK or engine.mask_seal >= 0.999
    return mask_leak(engine), max_deliverable_mv, sealed_airway


def circuit_ventilation(engine: SimulationEngine) -> tuple[float, float]:
    """Return (RR, mean VT L) of all recent measured breaths that reach the lungs."""
    spirometry = engine.vent.monitors
    rr = spirometry.rr_total
    resp = engine.resp.state
    assisted = assisted_positive_pressure(engine)
    if (
        not assisted
        and resp.apnea
        and not engine.vent.inspiring
        and not engine.vent.breath_inspiring
    ):
        return 0.0, 0.0
    leak_gap = 1.0 - engine.mask_seal if engine.state.airway_mode == AirwayType.MASK else 0.0
    if rr <= 0.0:
        return (resp.rr, leak_gap * resp.mv / resp.rr) if leak_gap > 0.0 and resp.rr > 0.0 else (0.0, 0.0)
    leak, max_deliverable_mv, sealed_airway = _circuit_delivery_limits(engine, rr)
    unsupported_mv = max(0.0, spirometry.mv_exp - spirometry.mv_supported)
    supported_mv = spirometry.mv_supported
    if assisted:
        supported_mv = min(supported_mv, max_deliverable_mv)
    if sealed_airway:
        unsupported_mv = min(unsupported_mv, max(0.0, max_deliverable_mv - supported_mv))
    mv_lung = supported_mv * (1.0 - leak) + unsupported_mv
    # During assisted PPV with a loose mask, patient effort can entrain room air
    # around the gap smoothly in proportion to (1 - mask_seal).
    if assisted and leak_gap > 0.0 and resp.rr > 0.0 and resp.mv > mv_lung:
        mv_lung += leak_gap * (resp.mv - mv_lung)
    return rr, mv_lung / rr


def circuit_alveolar_ventilation(
    engine: SimulationEngine, dead_space_l: float, upper_eff: float
) -> float:
    """Return per-breath alveolar ventilation (L/min) from recent measured circuit breaths."""
    spirometry = engine.vent.monitors
    rr = spirometry.rr_total
    resp = engine.resp.state
    assisted = assisted_positive_pressure(engine)
    if (
        not assisted
        and resp.apnea
        and not engine.vent.inspiring
        and not engine.vent.breath_inspiring
    ):
        return 0.0
    leak_gap = 1.0 - engine.mask_seal if engine.state.airway_mode == AirwayType.MASK else 0.0
    if rr <= 0.0:
        if leak_gap > 0.0 and resp.rr > 0.0:
            spont_va = resp.rr * max(0.0, resp.vt / 1000.0 - dead_space_l)
            return leak_gap * spont_va
        return 0.0
    leak, max_deliverable_mv, sealed_airway = _circuit_delivery_limits(engine, rr)
    va = engine.vent.alveolar_ventilation(
        dead_space_l,
        upper_eff=upper_eff,
        leak_fraction=leak,
        deliverable_mv_l_min=max_deliverable_mv,
        assisted=assisted,
        sealed_airway=sealed_airway,
    )
    if assisted and leak_gap > 0.0 and resp.rr > 0.0:
        spont_va = resp.rr * max(0.0, resp.vt / 1000.0 - dead_space_l)
        if spont_va > va:
            va += leak_gap * (spont_va - va)
    return va


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
    return clamp01(
        engine.response.tolerance(
            state.hypnotic_response_ce,
            state.opioid_ce,
            mac_sevo=state.mac_sevo,
            mac_n2o=state.mac_n2o,
            ce_ketamine=state.ketamine_ce,
            ce_lidocaine=state.lidocaine_ce,
            intensity=intensity,
        )
    )


def thermal_hypnotic_ce(engine: SimulationEngine) -> float:
    """Propofol-equivalent Ce (mcg/mL) for thermoregulation and redistribution.

    Kurz 1995 (Anesth Analg 81:393): 300 ng/mL plasma midazolam lowers the
    shivering threshold by 0.6 °C (35.9 to 35.3 °C), far less than general anesthetics.
    """
    state = engine.state
    etomidate = (
        max(0.0, state.etomidate_ce)
        * PROPOFOL_LOSS_OF_RESPONSE
        / ETOMIDATE_LOSS_OF_RESPONSE
    )
    midaz = max(0.0, state.midazolam_ce)
    ketamine = engine.thermal.tuning.depth_propofol_scale * max(0.0, state.ketamine_ce) / KETAMINE_MAC_EQUIVALENT
    return max(0.0, state.propofol_ce) + etomidate + ketamine + 1.80 * midaz / (midaz + 150.0)


def nmba_twitch_fraction(engine: SimulationEngine) -> float:
    """Remaining peripheral T1 twitch fraction from neuromuscular blockers alone."""
    s = engine.state
    return (
        (1.0 - engine.tof_pd.twitch_block(s.roc_ce, s.mac_sevo, s.mac_n2o))
        * succinylcholine_twitch_fraction(s.succinylcholine_ce)
    )


def central_muscle_strength(engine: SimulationEngine) -> float:
    """Remaining central respiratory/airway strength after either muscle blocker."""
    state = engine.state
    potentiation = engine.tof_pd.potentiation_factor(state.mac_sevo, state.mac_n2o)
    block = engine.resp.central_nmba_block(
        engine.tof_pd.ce_central / potentiation, state.mac_sevo, state.mac_n2o,
    )
    return clamp01(1.0 - block) * succinylcholine_twitch_fraction(
        state.succinylcholine_ce, central=True,
    )


def cardiovascular_hypnotic_ce(engine: SimulationEngine) -> float:
    """Propofol-equivalent Ce (mcg/mL) at the steady BIS site without EEG ceilings or BIS age scaling."""
    bis_equiv = float(hypnotic_equivalent(
        engine.pk["propofol"].state.ce,
        engine.pk["etomidate"].state.ce,
        engine.pk["midazolam"].state.ce,
        engine.midazolam_c50,
        engine.bis.age_factor,
        bis=True,
    ))
    if abs(engine.state.hypnotic_ce - bis_equiv) > 1e-9:
        return float(engine.state.hypnotic_ce)
    return float(hypnotic_equivalent(
        engine.pk["propofol"].state.ce,
        engine.pk["etomidate"].state.ce,
        engine.pk["midazolam"].state.ce,
        engine.midazolam_c50,
    ))


def cardiovascular_response_fractions(engine: SimulationEngine, intensity: float) -> tuple[float, float]:
    """Pressure and HR stimulus fractions, separate from movement and EEG arousal.

    The fractions are calibrated at steady propofol concentrations, so propofol
    enters at its BIS effect site. Early after induction, the concentration at
    the faster clinical-response site is about twice as high and would block the
    intubation response.
    """
    state = engine.state
    return engine.response.cardiovascular_response_fractions(
        cardiovascular_hypnotic_ce(engine),
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
    state.roc_cp = float(engine.pk["roc"].state.c1 + engine.tof_pd.bound_rocuronium(engine.pk["roc"].v1))
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
        bis=True,
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
    state.mac_sevo = float(sevo.mac)
    state.mac_n2o = float(n2o.mac)
    state.mac = float(sevo.mac + n2o.mac)
    if circuit_connected(engine):
        state.et_sevo = float(sevo.p_alv * 100.0)
        state.et_n2o = float(n2o.p_alv * 100.0)
        state.et_mac = float(
            sevo.p_alv * 100.0 / engine.pk_sevo.mac_age + n2o.p_alv * 100.0 / engine.pk_n2o.mac_age
        )
    else:
        state.et_sevo = state.et_n2o = state.et_mac = 0.0


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
    state.ionized_calcium = fluids.ionized_calcium(resp_state.ph_37)
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


DEADSPACE_APNEIC_STPD_L = 0.030


def advance_tracheal_fio2(
    engine: SimulationEngine, dt: float, *, inflow_stpd_l_min: float | None = None
) -> float:
    """Advance dead-space tracheal FiO2 and return the step-average FiO2 delivered to alveoli."""
    state = engine.state
    if dt <= 0.0:
        return engine._tracheal_fio2
    sealed = circuit_connected(engine)
    assisted_rr, assisted_vt_l = circuit_ventilation(engine) if sealed else (0.0, 0.0)
    assisted_breaths = sealed and (engine.vent.is_on or engine.bag_mask_active) and assisted_rr * assisted_vt_l > 0.05
    spont_breaths = not engine.resp.state.apnea and state.va > 0.1
    if assisted_breaths or spont_breaths:
        engine._tracheal_fio2 = state.fio2
        return state.fio2
    # During apnea, convective bulk STPD inflow displaces the conserved
    # anatomic dead-space buffer symmetrically for any circuit gas composition.
    patency = clamp01(engine.airway_status.patency)
    share = mask_circuit_share(engine) if sealed else 0.0
    sealed_circuit = sealed and share >= 0.999
    avail_flow = engine.circuit.deliverable_flow(dt) if sealed_circuit else math.inf
    old_fio2 = engine._tracheal_fio2
    target_fio2 = state.fio2
    if patency <= 0.0 or avail_flow <= 1e-9:
        return old_fio2
    if inflow_stpd_l_min is None:
        resp = engine.resp
        frc_exp_stpd = max(0.0, engine.aeration.frc - resp.frc) * patency * resp._btps_to_stpd
        apneic_inflow, refill_l_min, _ = resp._apneic_inflow_components(
            patency, 0.0, 0.0, dt, avail_flow if sealed_circuit else None,
        )
        inflow_l_s = min(avail_flow / 60.0, (apneic_inflow + refill_l_min) / 60.0 + frc_exp_stpd / dt)
    else:
        inflow_l_s = min(avail_flow / 60.0, max(0.0, inflow_stpd_l_min) / 60.0)
    # Nitrogen in the airway/circuit slows passive apneic wash-in; downward room-air
    # entrainment flushes dead space at the convective bulk inflow rate, while upward
    # wash-in blends smoothly from diffusive exchange to convective inflow as N2 clears.
    fast_rate = inflow_l_s / DEADSPACE_APNEIC_STPD_L
    if target_fio2 <= old_fio2:
        rate = fast_rate
    else:
        comp = engine.circuit.composition
        max_trach_fn2 = share * comp.fin2 + (1.0 - share) * 0.79 if sealed and share > 0.0 else 0.79
        preox_clear = clamp01((0.60 - engine.resp.alveolar_fn2) / 0.30)
        pure_o2 = clamp01(1.0 - max_trach_fn2 / 0.01)
        n2_clear = max(preox_clear, pure_o2)
        slow_rate = max(0.1, patency) / 240.0
        rate = (1.0 - n2_clear) * slow_rate + n2_clear * fast_rate
    u = rate * dt
    if u > 1e-9:
        avg_fio2 = target_fio2 + (old_fio2 - target_fio2) * (-math.expm1(-u) / u)
        engine._tracheal_fio2 = target_fio2 + (old_fio2 - target_fio2) * math.exp(-u)
        return avg_fio2
    return old_fio2


def circuit_inspired_mixture(
    composition: GasComposition, tracheal_fio2: float, circuit_share: float
) -> tuple[float, float, float, float]:
    """Return complete (fio2, fi_agent, fin2o, fin2) summing to 1 on the circuit branch."""
    fio2 = clamp01((tracheal_fio2 - (1.0 - circuit_share) * 0.21) / circuit_share)
    available = max(0.0, 1.0 - fio2)
    fi_agent = clamp01(composition.fi_agent)
    fin2o = clamp01(composition.fin2o)
    other = fi_agent + fin2o
    if other > available and other > 0.0:
        scale = available / other
        fi_agent *= scale
        fin2o *= scale
    fin2 = max(0.0, available - fi_agent - fin2o)
    return fio2, fi_agent, fin2o, fin2


def step_respiration(engine: SimulationEngine, dt: float, cardiac_output: float) -> RespState:
    """Advance the respiratory model; dt 0 evaluates the current state without advancing it."""
    state = engine.state
    engine.resp.update_fluid_balance(engine.hemo.hb_conc, engine.hemo.get_hematocrit(), engine.hemo.blood_volume)
    sealed = circuit_connected(engine)
    # Gas exchange uses recent exhaled breaths, so it lags at least one breath.
    assisted_rr, assisted_vt_l = circuit_ventilation(engine) if sealed else (0.0, 0.0)
    airway = engine.airway_status
    upper_eff = max(0.1, 1.0 - 0.2 * (1.0 - clamp01(airway.patency)))
    mech_va_l_min = (
        circuit_alveolar_ventilation(engine, engine.resp.vd_deadspace, upper_eff)
        if sealed
        else None
    )
    peep_scale = mask_seal_factor(engine)
    cache = engine._step_gas if dt > 0.0 else None
    if dt > 0.0:
        engine._step_gas = None
    tracheal_fio2 = cache.tracheal_fio2 if cache is not None else advance_tracheal_fio2(engine, dt)
    share = mask_circuit_share(engine)
    if share > 0.0:
        if cache is not None and cache.circuit_fin2 is not None:
            circuit_fin2 = cache.circuit_fin2
        else:
            _, _, _, circuit_fin2 = circuit_inspired_mixture(engine.circuit.composition, tracheal_fio2, share)
        inspired_fn2 = share * circuit_fin2 + (1.0 - share) * 0.79
    else:
        inspired_fn2 = 0.79
    if cache is not None and cache.apneic_limit_l_min is not None:
        apneic_limit: float | None = cache.apneic_limit_l_min
    else:
        apneic_limit = engine.circuit.deliverable_flow_l_min if sealed and share >= 0.999 else None
    if sealed and share >= 0.999 and dt > 0.0 and engine.circuit.last_uptake_scale < 1.0 - 1e-6:
        assisted_rr *= engine.circuit.last_uptake_scale
        if mech_va_l_min is not None:
            mech_va_l_min *= engine.circuit.last_uptake_scale
    if cache is not None and share > 0.0:
        agent_uptake = cache.n2o_uptake_l_min + cache.sevo_uptake_l_min
    elif dt > 0.0 and share <= 0.0:
        agent_uptake = (
            engine.pk_n2o.last_blood_uptake_l_min
            + engine.pk_sevo.last_blood_uptake_l_min
        )
    else:
        shunt_now = engine.aeration.shunt_fraction
        agent_uptake = (
            engine.pk_n2o.alveolar_uptake_l_min(cardiac_output, shunt_now)
            + engine.pk_sevo.alveolar_uptake_l_min(cardiac_output, shunt_now)
        )
    lung_volume_l = cache.lung_volume_l if cache is not None else engine.aeration.frc
    spirometry = engine.vent.monitors
    supported_share = (
        min(1.0, spirometry.mv_supported / spirometry.mv_exp)
        if spirometry.mv_exp > 0.0
        else (1.0 if assisted_positive_pressure(engine) else 0.0)
    )
    resp_state = engine.resp.step(
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
        fio2=tracheal_fio2,
        ce_roc=engine.tof_pd.ce_central / engine.tof_pd.potentiation_factor(state.mac_sevo, state.mac_n2o),
        neuromuscular_strength=(
            succinylcholine_twitch_fraction(state.succinylcholine_ce, central=True)
            * (1.0 - engine.resp_mech.effort.muscle.fatigue)
            * engine.malignant_hyperthermia.respiratory_excursion_fraction
        ),
        mac_sevo=state.mac_sevo,
        mech_rr=assisted_rr,
        mech_vt_l=assisted_vt_l,
        measured_breaths=sealed and (
            engine.vent.has_measured_breath
            or (peep_scale >= 0.999 and engine.circuit.bellows_fraction <= 0.05)
        ),
        airway_patency=airway.patency,
        ventilation_efficiency=airway.ventilation_efficiency,
        vq_mismatch=airway.vq_mismatch,
        hb_g_dl=engine.hemo.hb_conc,
        hematocrit=engine.hemo.get_hematocrit(),
        blood_volume_ml=engine.hemo.blood_volume,
        cardiac_output=cardiac_output,
        metabolic_factor=max(0.5, engine.thermal.metabolic_factor),
        unconscious=state.loc,
        lung_volume_l=lung_volume_l,
        shunt_fraction=engine.aeration.shunt_fraction,
        temp_c=state.temp_c,
        airway_pressure=engine.resp_mech.peep,
        # Brainstem drive persists until cerebral failure is nearly complete or CO2 narcosis is extreme.
        central_drive_fraction=engine.cerebral.respiratory_drive * engine.toxicity.respiratory_drive,
        ketamine_cp=engine.pk["ketamine"].state.c1,
        mac_n2o=state.mac_n2o,
        n2o_uptake_l_min=agent_uptake,
        fin2=inspired_fn2,
        apneic_flow_limit_l_min=apneic_limit,
        mech_supported_fraction=supported_share,
        mech_va_l_min=mech_va_l_min,
    )
    if dt > 0.0:
        gas_vol = engine.resp.gas_volume_l
        if gas_vol > 0.0:
            changed = False
            for model in (engine.pk_sevo, engine.pk_n2o):
                if model.frc != gas_vol:
                    model.state.p_alv = clamp01(model.state.p_alv * model.frc / gas_vol)
                    model.frc = gas_vol
                    changed = True
            if changed:
                sync_inhaled_agents(engine)
    return resp_state


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
    seal_scale = mask_seal_factor(engine)
    kept = delivered_fraction(engine) if sealed else 0.0
    paw_now = vent.paw * seal_scale if sealed else 0.0
    return PhysiologyStepState(
        hemo_state=hemo_state,
        resp_state=resp_state,
        pit_estimate=pit_estimate,
        rr_display=rr_display,
        vt_display_ml=vt_display_ml,
        mv_display_l_min=mv_display_l_min,
        paw_display=paw_now,
        flow_display=vent.flow * kept,
        volume_display=vent.volume * kept,
        paw_peak=max(spirometry.paw_peak * seal_scale, paw_now) if sealed else 0.0,
        paw_plat=spirometry.paw_plat * seal_scale if assisted else math.nan,
        paw_mean=max(spirometry.paw_mean, engine._apl_pressure) * seal_scale if sealed else 0.0,
        peep=(spirometry.peep if assisted else engine._apl_pressure) * seal_scale if sealed else 0.0,
        compliance_dyn=(
            spirometry.compliance_dyn * (kept / seal_scale)
            if assisted and kept > 1e-6 and seal_scale > 1e-6
            else math.nan
        ),
    )


def project_runtime_physiology(engine: SimulationEngine, snapshot: PhysiologyStepState) -> None:
    """Project a runtime physiology step back into the public SimulationState."""
    state = engine.state
    project_hemodynamics(engine, snapshot.hemo_state)
    _project_respiratory_observables(engine, snapshot)
    state.oxygen_delivery_ratio = float(engine.hemo.compute_do2_ratio(
        max(0.0, state.sao2) / 100.0, max(0.0, state.pao2), state.co
    ))


def compute_pit_estimate(engine: SimulationEngine, dt: float, *, smoothed: bool = False) -> float:
    """Estimate intrathoracic pressure (mmHg) from airway pressure and inspiratory effort."""
    hemo_cfg = engine.hemo.config
    raw_effort = (
        engine.current_mean_effort
        if smoothed or dt <= 0.0
        else engine.vent.capped_muscle_pressure_area / dt
    )
    effort_pressure = min(raw_effort, hemo_cfg.effort_pressure_max)
    return hemo_cfg.pit_0 + CMH2O_TO_MMHG * (
        hemo_cfg.paw_transmission * engine.current_mean_paw - hemo_cfg.effort_transmission * effort_pressure
    )


def step_coronary_from_state(
    engine: SimulationEngine, dt: float, hemo_state: HemoState, sao2: float,
) -> None:
    """Advance or sync coronary balance from arterial, CPR, and atrial pressures."""
    state = engine.state
    hemo = engine.hemo
    coronary_sbp = state.sbp
    coronary_dbp = state.dbp
    if hemo.compressions:
        if hemo_state.cpr_perfusion_map <= 0.0:
            hemo_state = hemo.evaluate_state(include_stim=False)
        cpr_sbp, cpr_dbp = compression_pressures(hemo_state.cpr_perfusion_map, hemo_state.venous_pressure)
        coronary_sbp = max(coronary_sbp, cpr_sbp) if hemo_state.intrinsic_co > 0.0 else cpr_sbp
        coronary_dbp = max(coronary_dbp, cpr_dbp) if hemo_state.intrinsic_co > 0.0 else cpr_dbp
    intrinsic_sbp = (
        0.0
        if hemo_state.intrinsic_co <= 0.05
        else state.sbp * (clamp01(hemo_state.intrinsic_map / max(1.0, state.map)) if hemo.compressions else 1.0)
    )
    coronary = hemo.coronary
    before = (coronary.ischemia, coronary.shock_stunning, coronary.injury, coronary.energy_failure)
    coronary.step(
        dt, state.hr, coronary_sbp, coronary_dbp, max(state.lap, hemo_state.venous_pressure), hemo.hb_conc,
        sao2, hemo.myocardial_hypoxia,
        fibrillating=hemo_state.rhythm_type == RhythmType.VFIB, compressions=hemo.compressions,
        venous_pressure=hemo_state.venous_pressure,
        intrinsic_sbp=intrinsic_sbp,
    )
    after = (coronary.ischemia, coronary.shock_stunning, coronary.injury, coronary.energy_failure)
    if after != before:
        hemo.invalidate_state_cache()


def sync_monitor_baselines(engine: SimulationEngine) -> None:
    """Derive monitor baselines from the current physiologic snapshot."""
    from . import runtime as runtime_core

    state = engine.state
    loc_val = loss_of_response(engine)
    tol_val = stimulus_tolerance(engine)
    engine._tol_current = tol_val
    cortical = engine.cerebral.cortical_function
    drug_bis = engine.bis.compute_bis(state.hypnotic_ce, state.mac_sevo, state.ketamine_ce)
    raw_bis = drug_bis * cortical
    bis_val = clamp(
        raw_bis + runtime_core.stimulus_effects(engine, 0.0).bis * cortical,
        0.0,
        100.0,
    )
    hemo_state = engine.hemo.state
    cardiac_sample = engine.cardiac_cycle.seed(state.hr, hemo_state.rhythm_type)
    arterial_sample = engine.arterial_waveform.seed(
        cardiac_sample,
        state.map,
        state.sv,
        static_pressure=hemo_state.venous_pressure,
        pulse_pressure_factor=hemo_state.pulse_pressure_factor,
    )
    art_reading = engine.art_line.seed(arterial_sample)
    engine.airway_sensor.seed(state.paw, state.flow, state.volume)
    state.bis = float(bis_val)
    state.loc = float(loc_val)
    state.tol = float(tol_val)
    state.capno_co2 = float(engine.capno.co2) if circuit_connected(engine) else 0.0
    state.ecg_voltage = state.pleth_voltage = 0.0
    project_arterial(state, arterial_sample, art_reading)
    state.display_hr = float(cardiac_sample.display_hr)
    state.display_bis = float(bis_val)
    engine.etco2_readout.seed(
        state.etco2,
        et_gases=(state.et_o2, state.et_sevo, state.et_n2o, state.et_mac),
    )
    state.etco2_signal_valid = engine.etco2_readout.valid
    if not state.etco2_signal_valid:
        state.et_o2 = state.et_sevo = state.et_n2o = state.et_mac = 0.0
    state.display_etco2 = float(state.etco2)
    if state.spo2_signal_valid:
        state.display_spo2 = float(state.spo2)
        engine.spo2_mon.display_saturation = float(state.spo2)
    else:
        engine.spo2_mon.display_saturation = None
        state.spo2 = state.display_spo2 = 0.0
    engine.bis.initialize(drug_bis, cortical_function=cortical)
    engine.smooth_bis = state.bis
    engine._bis_noise = 0.0


def sync_state_from_models(engine: SimulationEngine) -> None:
    """Derive the public SimulationState from current subsystem state."""
    state = engine.state
    state.temp_c = float(state.temp_c)
    sync_inspired_gas(engine)
    engine._tracheal_fio2 = state.fio2
    sync_inhaled_agents(engine)
    sync_pk_state(engine)
    state.shivering = float(engine.thermal.shiver_level)
    engine.thermal.update_depth(state.temp_c, thermal_hypnotic_ce(engine), state.mac)
    engine.thermal.metabolic_factor *= engine.toxicity.metabolic_factor
    engine.thermal.metabolic_factor += (
        engine.malignant_hyperthermia.excess_metabolism + 0.25 * engine.sepsis.severity
    )
    hemo_state = engine.hemo.state
    resp_state = step_respiration(engine, 0.0, hemo_state.co)
    project_runtime_physiology(engine, build_snapshot_from_models(
        engine, hemo_state, resp_state, pit_estimate=compute_pit_estimate(engine, 0.0),
    ))
    sync_monitor_baselines(engine)
    hemo = engine.hemo
    step_coronary_from_state(engine, 0.0, hemo_state, resp_state.sao2)
    state.myocardial_ischemia = float(hemo.coronary.ischemia)
    state.coronary_supply_demand_ratio = float(hemo.coronary.supply_demand_ratio)
    state.st_depression_mv = float(hemo.coronary.st_depression_mv)
