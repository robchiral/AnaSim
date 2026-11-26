from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anasim.monitors.ecg import st_analyzable_scale
from anasim.patient.pd import hypnotic_equivalent, opioid_equivalent
from anasim.patient.pd.anesthesia import (
    ETOMIDATE_LOSS_OF_RESPONSE,
    EXTUBATION_INTENSITY,
    KETAMINE_MAC_EQUIVALENT,
    LARYNGOSCOPY_INTENSITY,
    PROPOFOL_LOC_TRANSFER,
    PROPOFOL_LOSS_OF_RESPONSE,
    hypothermia_narcosis_fraction,
    opioid_controller_fraction,
    opioid_rate_fraction,
)
from anasim.patient.pd.nmba import (
    TOF_FADE_EXPONENT,
    succinylcholine_phase2_tof_fraction,
    succinylcholine_twitch_fraction,
)
from anasim.patient.pk_models import MorphinePK, SuccinylcholinePK
from anasim.physiology.malignant_hyperthermia import dantrolene_twitch_fraction
from anasim.physiology.resuscitation import compression_pressures

from .constants import CMH2O_TO_MMHG
from .enums import PULSELESS_RHYTHMS, RhythmType
from .state import AirwayType, SimulationState
from .utils import clamp, clamp01

if TYPE_CHECKING:
    from anasim.machine.circuit import GasComposition
    from anasim.monitors.arterial import ArterialPressureSample
    from anasim.patient.volatile_pk import _IntegrationResult
    from anasim.physiology.hemo_types import HemoState
    from anasim.physiology.respiration import PassiveExchangeTerms, RespState

    from .engine import SimulationEngine


@dataclass(slots=True)
class StepGasCache:
    tracheal_fio2: float
    n2o_uptake_l_min: float
    lung_volume_l: float
    tracheal_fin2: float
    sevo_uptake_l_min: float = 0.0
    apneic_limit_l_min: float | None = None
    reservoir_volume_l: float | None = None
    bellows_fraction: float | None = None
    exchange: PassiveExchangeTerms | None = None
    va: float = 0.0
    co: float = 0.0
    sevo_result: _IntegrationResult | None = None
    n2o_result: _IntegrationResult | None = None
    dt: float = 0.0


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
    return (
        (engine.vent.is_on and engine.vent.settings.mode != "CPAP")
        or (engine.apl_cmh2o > 0.0 and (engine.bag_mask_active or engine.vent.manual_breath_active))
    )


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
    assisted = assisted_positive_pressure(engine)
    keep = 1.0 - mask_leak(engine) if assisted else 1.0
    supported = spirometry.mv_supported * keep if spirometry.mv_supported > 0.0 else 0.0
    unsupported = max(0.0, spirometry.mv_exp - spirometry.mv_supported)
    spont_mv = engine.resp.state.mv if not engine.resp.state.apnea else 0.0
    if not engine.vent.has_measured_breath and supported + unsupported <= 0.0:
        pressurized = (
            (
                (assisted or (engine.vent.is_on and engine.vent.settings.peep > 0.0))
                and engine.circuit.bellows_fraction > 0.0
            )
            or engine._apl_pressure > 0.0
        )
        if pressurized and seal < 1.0 and spont_mv > 0.0 and assisted:
            if engine.bag_mask_active and not engine.vent.is_on:
                est_sup = engine.bag_mask_rr * engine.bag_mask_vt
            elif engine.vent.is_on and engine.vent.settings.rr > 0.0:
                est_sup = engine.vent.settings.rr * engine.vent.settings.tv / 1000.0
            else:
                est_sup = 0.0
            est_delivered = est_sup * keep
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
        avail_stpd = engine.circuit.total_inflow()
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
    imperfect mask seal or peak pressure above the seal threshold.
    """
    if not assisted_positive_pressure(engine) or engine.state.airway_mode != AirwayType.MASK:
        return 0.0
    seal = engine.mask_seal
    if seal <= 0.0:
        return 1.0
    delivered_peak = engine.vent.monitors.paw_peak * seal
    threshold = engine.airway_tuning.mask_seal_pressure * seal
    overpressure = (
        engine.airway_tuning.mask_overpressure_leak
        * max(0.0, delivered_peak - threshold)
        / max(delivered_peak, 1.0)
    )
    kept = seal * (1.0 - overpressure)
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
            + engine.circuit.total_inflow()
        )
        / engine.resp._btps_to_stpd
    )
    sealed_airway = engine.state.airway_mode != AirwayType.MASK or engine.mask_seal >= 0.999
    return mask_leak(engine), max_deliverable_mv, sealed_airway


def _circuit_ventilation_state(
    engine: SimulationEngine,
) -> tuple[bool, float, float, float, float, bool] | None:
    """Shared circuit-breath delivery preamble; None when the patient is apneic with no mandatory source."""
    vent = engine.vent
    resp = engine.resp.state
    if (
        vent.mandatory_period_s is None
        and resp.apnea
        and not vent.inspiring
        and not vent.breath_inspiring
        and not (vent._measured is not None and vent._measured.kind == "MANUAL")
    ):
        return None
    rr = vent.monitors.rr_total
    assisted = assisted_positive_pressure(engine)
    leak_gap = 1.0 - engine.mask_seal if engine.state.airway_mode == AirwayType.MASK else 0.0
    if rr <= 0.0:
        return assisted, rr, leak_gap, 0.0, 0.0, False
    leak, max_deliverable_mv, sealed_airway = _circuit_delivery_limits(engine, rr)
    return assisted, rr, leak_gap, leak, max_deliverable_mv, sealed_airway


def _circuit_ventilation_and_va(
    engine: SimulationEngine, dead_space_l: float, upper_eff: float,
) -> tuple[float, float, float]:
    """Return (RR, mean VT L, alveolar ventilation L/min) from recent measured circuit breaths."""
    info = _circuit_ventilation_state(engine)
    if info is None:
        return 0.0, 0.0, 0.0
    assisted, rr, leak_gap, leak, max_deliverable_mv, sealed_airway = info
    resp = engine.resp.state
    if rr <= 0.0:
        if leak_gap > 0.0 and resp.rr > 0.0:
            spont_va = resp.rr * max(0.0, resp.vt / 1000.0 - dead_space_l)
            return resp.rr, leak_gap * resp.mv / resp.rr, leak_gap * spont_va
        return 0.0, 0.0, 0.0
    spirometry = engine.vent.monitors
    unsupported_mv = max(0.0, spirometry.mv_exp - spirometry.mv_supported)
    supported_mv = spirometry.mv_supported
    if assisted:
        supported_mv = min(supported_mv, max_deliverable_mv)
    if sealed_airway:
        unsupported_mv = min(unsupported_mv, max(0.0, max_deliverable_mv - supported_mv))
    mv_lung = supported_mv * (1.0 - leak) + unsupported_mv
    va = engine.vent.alveolar_ventilation(
        dead_space_l,
        upper_eff=upper_eff,
        leak_fraction=leak,
        deliverable_mv_l_min=max_deliverable_mv,
        assisted=assisted,
        sealed_airway=sealed_airway,
    )
    if assisted and leak_gap > 0.0 and resp.rr > 0.0:
        if resp.mv > mv_lung:
            mv_lung += leak_gap * (resp.mv - mv_lung)
        spont_va = resp.rr * max(0.0, resp.vt / 1000.0 - dead_space_l)
        if spont_va > va:
            va += leak_gap * (spont_va - va)
    return rr, mv_lung / rr, va


def circuit_ventilation(engine: SimulationEngine) -> tuple[float, float]:
    """Return (RR, mean VT L) of all recent measured breaths that reach the lungs."""
    info = _circuit_ventilation_state(engine)
    if info is None:
        return 0.0, 0.0
    assisted, rr, leak_gap, leak, max_deliverable_mv, sealed_airway = info
    resp = engine.resp.state
    if rr <= 0.0:
        return (resp.rr, leak_gap * resp.mv / resp.rr) if leak_gap > 0.0 and resp.rr > 0.0 else (0.0, 0.0)
    spirometry = engine.vent.monitors
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
    return _circuit_ventilation_and_va(engine, dead_space_l, upper_eff)[2]


def measured_ventilation(
    engine: SimulationEngine,
    connected: bool,
    spontaneous_rr: float,
    spontaneous_vt_l: float,
    *,
    kept: float | None = None,
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
        frac = delivered_fraction(engine) if kept is None else kept
        # Before the first completed breath, the respiratory model supplies
        # the baseline rate; no tidal volume has yet been measured.
        rr = spirometry.rr_total if engine.vent.has_measured_breath else spontaneous_rr
        return rr, spirometry.tv_exp * frac, spirometry.mv_exp * frac
    # Impedance counts muscle effort even when obstruction prevents airflow.
    return engine.resp.state.effort_rr, spontaneous_vt_l * 1000.0, spontaneous_rr * spontaneous_vt_l


def loss_of_response(engine: SimulationEngine, *, ketamine: bool = True) -> float:
    """Probability of no response to shaking or shouting at the current effect sites.

    Airway tone excludes ketamine, which preserves it.
    """
    state = engine.state
    prop_ce = PROPOFOL_LOC_TRANSFER * engine.pk["propofol"].state.ce_response
    eto_ce = engine.pk["etomidate"].state.ce
    midaz_ce = engine.pk["midazolam"].state.ce
    hypnotic = (
        max(0.0, prop_ce)
        if eto_ce <= 0.0 and midaz_ce <= 0.0
        else hypnotic_equivalent(prop_ce, eto_ce, midaz_ce, engine.midazolam_c50, engine.response.age_factor)
    )
    drug_response = engine.response.loss_of_response(
        hypnotic,
        state.opioid_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
        ce_ketamine=state.ketamine_ce if ketamine else 0.0,
    )
    return _combine_coma_tolerance(engine, drug_response)


def _combine_coma_tolerance(engine: SimulationEngine, drug_tolerance: float) -> float:
    """Combine drug stimulus tolerance with non-drug coma (hypothermia, cerebral, toxicity)."""
    cold_narcosis = hypothermia_narcosis_fraction(engine.state.temp_c)
    return clamp01(
        1.0
        - (1.0 - clamp01(drug_tolerance))
        * (1.0 - engine.cerebral.unresponsiveness)
        * (1.0 - engine.toxicity.unresponsiveness)
        * (1.0 - cold_narcosis)
    )


def stimulus_tolerance(engine: SimulationEngine, intensity: float = LARYNGOSCOPY_INTENSITY) -> float:
    """Probability of no response to a noxious stimulus of the given Bouillon 2004a intensity; laryngoscopy by default."""
    state = engine.state
    drug_tol = engine.response.tolerance(
        state.hypnotic_response_ce,
        state.opioid_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
        ce_ketamine=state.ketamine_ce,
        ce_lidocaine=state.lidocaine_ce,
        intensity=intensity,
    )
    return _combine_coma_tolerance(engine, drug_tol)


def extubation_tolerance(engine: SimulationEngine) -> float:
    """Airway-reflex tolerance for smooth extubation (Inomata 1999; Lee 2009).

    Uses the shallower of the fast clinical-response site and the steady BIS
    site (unscaled by BIS age/ceiling) so neither an early propofol bolus spike
    nor sluggish emergence washout masks reactive Stage II airway reflexes.
    """
    state = engine.state
    steady_hyp = float(hypnotic_equivalent(
        engine.pk["propofol"].state.ce,
        engine.pk["etomidate"].state.ce,
        engine.pk["midazolam"].state.ce,
        engine.midazolam_c50,
        engine.response.noxious_age_factor,
    ))
    eff_hyp = min(state.hypnotic_response_ce, steady_hyp)
    drug_tol = engine.response.tolerance(
        eff_hyp,
        state.opioid_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
        ce_ketamine=state.ketamine_ce,
        ce_lidocaine=state.lidocaine_ce,
        intensity=EXTUBATION_INTENSITY,
    )
    return _combine_coma_tolerance(engine, drug_tol)


def _bis_hypnotic_equivalent(engine: SimulationEngine) -> float:
    prop_ce = engine.pk["propofol"].state.ce
    eto_ce = engine.pk["etomidate"].state.ce
    midaz_ce = engine.pk["midazolam"].state.ce
    if eto_ce <= 0.0 and midaz_ce <= 0.0:
        return max(0.0, float(prop_ce))
    return float(hypnotic_equivalent(
        prop_ce, eto_ce, midaz_ce, engine.midazolam_c50, engine.bis.age_factor, bis=True,
    ))


def bis_hypnotic_components(engine: SimulationEngine) -> tuple[float, float]:
    """Split BIS hypnotic equivalent into (propofol_ce, non_propofol_adjunct_ce).

    Propofol uses Eleveld's age-dependent BIS delay while etomidate/midazolam
    use the age-independent monitor processing delay (Pilge 2006). Preserves
    direct test overrides of state.hypnotic_ce.
    """
    prop_ce = max(0.0, float(engine.pk["propofol"].state.ce))
    expected = _bis_hypnotic_equivalent(engine)
    if abs(engine.state.hypnotic_ce - expected) > 1e-9:
        return max(0.0, float(engine.state.hypnotic_ce)), 0.0
    return prop_ce, max(0.0, expected - prop_ce)


def thermal_hypnotic_ce(engine: SimulationEngine) -> float:
    """Propofol-equivalent Ce (mcg/mL) for thermoregulation and redistribution.

    Kurz 1995b: 300 ng/mL plasma midazolam lowers the shivering threshold by
    0.6 °C (35.9 to 35.3 °C), far less than general anesthetics.
    Ikeda 2001: ketamine preserves peripheral vasoconstriction and limits 1st-h
    core redistribution drop (~0.5 °C vs 1.4 °C with propofol).
    """
    state = engine.state
    etomidate = (
        max(0.0, state.etomidate_ce)
        * PROPOFOL_LOSS_OF_RESPONSE
        / ETOMIDATE_LOSS_OF_RESPONSE
    )
    midaz = max(0.0, state.midazolam_ce)
    ketamine = 0.25 * engine.thermal.tuning.depth_propofol_scale * max(0.0, state.ketamine_ce) / KETAMINE_MAC_EQUIVALENT
    return max(0.0, state.propofol_ce) + etomidate + ketamine + 1.80 * midaz / (midaz + 150.0)


def thermal_mac(engine: SimulationEngine) -> float:
    """Thermoregulatory MAC equivalent (N2O impairs thermoregulation half as much as sevoflurane)."""
    state = engine.state
    return max(0.0, state.mac_sevo) + 0.5 * max(0.0, state.mac_n2o)


def _succinylcholine_phase2_exposure(engine: SimulationEngine) -> float:
    sux_pk = engine.pk["succinylcholine"]
    return sux_pk.phase2_exposure if isinstance(sux_pk, SuccinylcholinePK) else 0.0


def nmba_twitch_fraction(engine: SimulationEngine) -> float:
    """Remaining peripheral T1 twitch fraction from neuromuscular blockers alone."""
    s = engine.state
    roc_t1 = 1.0 - engine.tof_pd.twitch_block(s.roc_ce, s.mac_sevo, s.mac_n2o) if s.roc_ce > 1e-9 else 1.0
    sux_t1 = succinylcholine_twitch_fraction(s.succinylcholine_ce) if s.succinylcholine_ce > 0.0 else 1.0
    exposure = _succinylcholine_phase2_exposure(engine)
    phase2_t1 = (
        succinylcholine_phase2_tof_fraction(exposure) ** (1.0 / TOF_FADE_EXPONENT)
        if exposure > 32.0
        else 1.0
    )
    return roc_t1 * sux_t1 * phase2_t1


def succinylcholine_central_ce(engine: SimulationEngine) -> float:
    """Central/laryngeal succinylcholine effect-site concentration, falling back to peripheral Ce."""
    sux_state = engine.pk["succinylcholine"].state
    return (
        sux_state.ce_central
        if (sux_state.ce_central > 0.0 or sux_state.c1 > 0.0)
        else sux_state.ce
    )


def central_muscle_strength(engine: SimulationEngine) -> float:
    """Remaining central respiratory/airway strength after either muscle blocker."""
    state = engine.state
    ce_central = engine.tof_pd.ce_central
    if ce_central > 0.0:
        potentiation = engine.tof_pd.potentiation_factor(state.mac_sevo, state.mac_n2o)
        roc_strength = clamp01(1.0 - engine.resp.central_nmba_block(
            ce_central / potentiation, state.mac_sevo, state.mac_n2o,
        ))
    else:
        roc_strength = 1.0
    sux_ce = succinylcholine_central_ce(engine)
    sux_strength = succinylcholine_twitch_fraction(sux_ce, central=True) if sux_ce > 0.0 else 1.0
    exposure = _succinylcholine_phase2_exposure(engine)
    phase2_central = succinylcholine_phase2_tof_fraction(exposure) ** 0.35 if exposure > 32.0 else 1.0
    return roc_strength * sux_strength * phase2_central


def cardiovascular_hypnotic_ce(engine: SimulationEngine) -> float:
    """Propofol-equivalent Ce (mcg/mL) at the steady BIS site without EEG ceilings or BIS age scaling."""
    bis_equiv = _bis_hypnotic_equivalent(engine)
    if abs(engine.state.hypnotic_ce - bis_equiv) > 1e-9:
        return float(engine.state.hypnotic_ce)
    eto_ce = engine.pk["etomidate"].state.ce
    midaz_ce = engine.pk["midazolam"].state.ce
    prop_ce = engine.pk["propofol"].state.ce
    if eto_ce <= 0.0 and midaz_ce <= 0.0:
        return max(0.0, float(prop_ce))
    return float(hypnotic_equivalent(prop_ce, eto_ce, midaz_ce, engine.midazolam_c50))


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
    for model, ce_field, cp_field in engine._pk_sync_entries:
        pk_state = model.state
        if ce_field is not None:
            val = pk_state.ce
            if val or getattr(state, ce_field):
                setattr(state, ce_field, float(val))
        if cp_field is not None:
            val = pk_state.c1
            if val or getattr(state, cp_field):
                setattr(state, cp_field, float(val))
    state.roc_ce = float(engine.tof_pd.ce)
    state.roc_cp = float(engine.pk["roc"].state.c1 + engine.tof_pd.bound_rocuronium(engine.pk["roc"].v1))
    morphine_pk = engine.pk["morphine"]
    m6g_excess = morphine_pk.m6g_excess_ce if isinstance(morphine_pk, MorphinePK) else 0.0
    if m6g_excess > 0.0:
        state.morphine_ce = float(morphine_pk.state.ce + m6g_excess)
    state.opioid_ce = float(opioid_equivalent(
        engine.pk["remi"].state.ce, state.fentanyl_ce, state.morphine_ce, state.hydromorphone_ce,
    ))
    state.opioid_cp = float(opioid_equivalent(
        engine.pk["remi"].state.c1, state.fentanyl_cp, state.morphine_cp, state.hydromorphone_cp,
    ))
    sync_neuromuscular_state(engine)
    eto_ce = engine.pk["etomidate"].state.ce
    midaz_ce = engine.pk["midazolam"].state.ce
    prop_state = engine.pk["propofol"].state
    if eto_ce <= 0.0 and midaz_ce <= 0.0:
        state.hypnotic_ce = max(0.0, float(prop_state.ce))
        state.hypnotic_response_ce = max(0.0, float(prop_state.ce_response))
    else:
        state.hypnotic_ce = float(hypnotic_equivalent(
            prop_state.ce, eto_ce, midaz_ce, engine.midazolam_c50, engine.bis.age_factor, bis=True,
        ))
        state.hypnotic_response_ce = float(hypnotic_equivalent(
            prop_state.ce_response, eto_ce, midaz_ce, engine.midazolam_c50, engine.response.noxious_age_factor,
        ))


def sync_neuromuscular_state(engine: SimulationEngine) -> None:
    """Project TOF % and T1 height %; T1 also includes non-fading phase I succinylcholine and dantrolene depression."""
    s = engine.state
    exposure = _succinylcholine_phase2_exposure(engine)
    phase2_fade = succinylcholine_phase2_tof_fraction(exposure) if exposure > 32.0 else 1.0
    roc_tof = engine.tof_pd.compute_tof_from_ce(s.roc_ce, s.mac_sevo, s.mac_n2o) if s.roc_ce > 1e-9 else 100.0
    s.tof = float(roc_tof * phase2_fade)
    roc_t1 = 1.0 if roc_tof >= 100.0 else (roc_tof / 100.0) ** (1.0 / TOF_FADE_EXPONENT)
    sux_t1 = succinylcholine_twitch_fraction(s.succinylcholine_ce) if s.succinylcholine_ce > 0.0 else 1.0
    phase2_t1 = phase2_fade ** (1.0 / TOF_FADE_EXPONENT) if phase2_fade < 1.0 else 1.0
    dant_t1 = dantrolene_twitch_fraction(s.dantrolene_ce) if s.dantrolene_ce > 0.0 else 1.0
    s.twitch = 100.0 * roc_t1 * sux_t1 * phase2_t1 * dant_t1


def sync_inspired_gas(engine: SimulationEngine) -> tuple[float, float]:
    """Project inspired gas at the airway and return (Fi sevo, Fi N2O) as fractions."""
    state = engine.state
    fio2, fi_sevo, fi_n2o, _ = inspired_gas_mixture(engine)
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
            sevo.p_alv * 100.0 / engine.pk_sevo.mac_age_monitor
            + n2o.p_alv * 100.0 / engine.pk_n2o.mac_age_monitor
        )
    else:
        state.et_sevo = state.et_n2o = state.et_mac = 0.0


def project_arterial(
    state: SimulationState, arterial_sample: ArterialPressureSample, art_reading: ArterialPressureSample,
) -> None:
    """Copy the ideal pulse landmarks and the catheter reading into the public snapshot."""
    state.sbp = float(arterial_sample.systolic)
    state.dbp = float(arterial_sample.diastolic)
    if state.co > 0.0 and state.rhythm_type not in PULSELESS_RHYTHMS:
        if state.map < state.dbp or state.map > state.sbp:
            state.map = float(arterial_sample.mean)
    elif not state.chest_compressions and state.co <= 0.0 and state.sbp == state.dbp:
        # Non-pulsatile Windkessel runoff: keep SBP == DBP == MAP without phantom pulse pressure.
        state.map = float(arterial_sample.mean)
        state.sbp = state.dbp = state.map
    else:
        state.sbp = max(state.sbp, state.map)
        state.dbp = min(state.dbp, state.map)
    state.art_pressure = float(art_reading.pressure)
    state.art_sbp = float(art_reading.systolic)
    state.art_dbp = float(art_reading.diastolic)
    state.art_map = float(art_reading.mean)


def _projected_st_depression(hemo, hemo_state: HemoState) -> float:
    st_scale = st_analyzable_scale(
        hemo_state.rhythm_type,
        hemo.coronary.energy_failure,
        hemo.sodium_block,
        hemo.hyperkalemia_block,
    )
    return float(hemo.coronary.st_depression_mv * st_scale)


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
    state.rhythm_type = hemo_state.rhythm_type
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
    state.st_depression_mv = _projected_st_depression(engine.hemo, hemo_state)
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
    state.albumin_g_l = fluids.plasma_albumin_g_l
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


def inspired_gas_mixture(engine: SimulationEngine) -> tuple[float, float, float, float]:
    """Return airway-opening (fio2, fi_agent, fin2o, fin2) summing to 1."""
    if circuit_connected(engine):
        share = mask_circuit_share(engine)
        if share > 0.0:
            comp = engine.circuit.composition
            o2 = share * comp.fio2 + (1.0 - share) * 0.21
            sevo = share * comp.fi_agent
            n2o = share * comp.fin2o
            n2 = max(0.0, 1.0 - o2 - sevo - n2o)
            return (o2, sevo, n2o, n2)
    return (0.21, 0.0, 0.0, 0.79)


def advance_tracheal_gas(
    engine: SimulationEngine,
    dt: float,
    *,
    inflow_stpd_l_min: float | None = None,
    effective_va: float | None = None,
    circuit_share: float | None = None,
) -> tuple[float, float, float, float]:
    """Advance conserved dead-space gas and return step-average (O2, sevo, N2O, N2)."""
    old = engine._tracheal_gas
    if dt <= 0.0:
        return old
    patency = clamp01(engine.airway_status.patency)
    va = (
        (engine.resp.exchange_va(engine.state.va, engine.airway_status.vq_mismatch) if patency > 0.0 else 0.0)
        if effective_va is None
        else effective_va
    )
    target = inspired_gas_mixture(engine)
    if va > 1e-4:
        engine._tracheal_gas = target
        return target
    sealed = circuit_connected(engine)
    if sealed and (engine.vent.is_on or assisted_positive_pressure(engine)):
        assisted_rr, assisted_vt_l = circuit_ventilation(engine)
        if assisted_rr * assisted_vt_l > 0.05:
            engine._tracheal_gas = target
            return target
    # During apnea, convective bulk STPD inflow flushes the conserved dead-space
    # buffer symmetrically across all gas species so circuit, dead space, and lung
    # share one exact mass balance.
    share = (mask_circuit_share(engine) if sealed else 0.0) if circuit_share is None else circuit_share
    sealed_circuit = sealed and share >= 0.999
    avail_flow = engine.circuit.deliverable_flow(dt) if sealed_circuit else math.inf
    if patency <= 0.0 or avail_flow <= 1e-9:
        return old
    if inflow_stpd_l_min is None:
        resp = engine.resp
        terms = resp.passive_exchange_terms(
            engine.aeration.frc, patency, 0.0, dt, avail_flow if sealed_circuit else None,
        )
        frc_inflow_l_s = max(0.0, terms.exhaled_vc_btps_l) * resp.btps_to_stpd / dt
        inflow_l_s = min(avail_flow / 60.0, terms.passive_stpd_l_min / 60.0 + frc_inflow_l_s)
    else:
        inflow_l_s = min(avail_flow / 60.0, max(0.0, inflow_stpd_l_min) / 60.0)
    u = (inflow_l_s / DEADSPACE_APNEIC_STPD_L) * dt
    if u > 1e-9:
        decay = math.exp(-u)
        avg_w = -math.expm1(-u) / u
        avg = (
            target[0] + (old[0] - target[0]) * avg_w,
            target[1] + (old[1] - target[1]) * avg_w,
            target[2] + (old[2] - target[2]) * avg_w,
            target[3] + (old[3] - target[3]) * avg_w,
        )
        engine._tracheal_gas = (
            target[0] + (old[0] - target[0]) * decay,
            target[1] + (old[1] - target[1]) * decay,
            target[2] + (old[2] - target[2]) * decay,
            target[3] + (old[3] - target[3]) * decay,
        )
        return avg
    return old


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
    hct = engine.hemo.get_hematocrit()
    engine.resp.update_fluid_balance(
        engine.hemo.hb_conc,
        hct,
        engine.hemo.blood_volume,
        circulation=engine.hemo.circulation_fraction(cardiac_output),
    )
    sealed = circuit_connected(engine)
    airway = engine.airway_status
    airway_tuning = engine.airway_tuning
    upper_eff = max(
        airway_tuning.vent_efficiency_min,
        1.0 - airway_tuning.vent_efficiency_upper_weight * (1.0 - clamp01(airway.patency)),
    )
    vd_peep = engine.resp.vd_deadspace + 0.0025 * max(0.0, engine.resp_mech.peep - 12.0)
    # Gas exchange uses recent exhaled breaths, so it lags at least one breath.
    if sealed:
        assisted_rr, assisted_vt_l, mech_va_l_min = _circuit_ventilation_and_va(engine, vd_peep, upper_eff)
    else:
        assisted_rr, assisted_vt_l, mech_va_l_min = 0.0, 0.0, None
    peep_scale = mask_seal_factor(engine)
    cache = engine._step_gas if dt > 0.0 else None
    if dt > 0.0:
        engine._step_gas = None
    share = mask_circuit_share(engine)
    if cache is not None:
        tracheal_fio2 = cache.tracheal_fio2
        inspired_fn2 = cache.tracheal_fin2
    else:
        avg_gas = advance_tracheal_gas(engine, dt, circuit_share=share)
        tracheal_fio2, inspired_fn2 = avg_gas[0], avg_gas[3]
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
            engine.pk_n2o.alveolar_uptake_l_min(cardiac_output, shunt_now, temp_c=state.temp_c)
            + engine.pk_sevo.alveolar_uptake_l_min(cardiac_output, shunt_now, temp_c=state.temp_c)
        )
    lung_volume_l = cache.lung_volume_l if cache is not None else engine.aeration.frc
    spirometry = engine.vent.monitors
    supported_share = (
        min(1.0, spirometry.mv_supported / spirometry.mv_exp)
        if spirometry.mv_exp > 0.0
        else (1.0 if assisted_positive_pressure(engine) else 0.0)
    )
    morphine_pk = engine.pk["morphine"]
    m6g_excess = morphine_pk.m6g_excess_ce if isinstance(morphine_pk, MorphinePK) else 0.0
    exposure = _succinylcholine_phase2_exposure(engine)
    phase2_central = succinylcholine_phase2_tof_fraction(exposure) ** 0.35 if exposure > 32.0 else 1.0
    sux_central = succinylcholine_central_ce(engine)
    sux_strength = succinylcholine_twitch_fraction(sux_central, central=True) if sux_central > 0.0 else 1.0
    ce_central = engine.tof_pd.ce_central
    ce_roc_eff = (
        ce_central / engine.tof_pd.potentiation_factor(state.mac_sevo, state.mac_n2o)
        if ce_central > 0.0
        else 0.0
    )
    prop_resp = engine.pk["propofol"].state.ce_resp
    ce_prop_resp = (
        max(0.0, prop_resp)
        if state.etomidate_ce <= 0.0 and state.midazolam_ce <= 0.0
        else hypnotic_equivalent(
            prop_resp, state.etomidate_ce, state.midazolam_ce, engine.midazolam_c50, ventilation=True,
        )
    )
    cached_exchange = (
        cache.exchange
        if (
            cache is not None
            and cache.exchange is not None
            and share > 0.0
            and dt == cache.dt
            and apneic_limit == cache.apneic_limit_l_min
        )
        else None
    )
    resp_state = engine.resp.step(
        dt,
        ce_prop=ce_prop_resp,
        ce_remi=opioid_equivalent(engine.pk["remi"].state.ce_resp, state.fentanyl_ce),
        opioid_controller_fraction=opioid_controller_fraction(
            morphine_pk.state.ce_resp + m6g_excess, engine.pk["hydromorphone"].state.ce_resp,
        ),
        opioid_rate_fraction=opioid_rate_fraction(
            morphine_pk.state.ce_rr + m6g_excess, engine.pk["hydromorphone"].state.ce_rr,
            variability=engine.patient.variability.opioid_rate,
        ),
        mech_vent_mv=assisted_rr * assisted_vt_l,
        fio2=tracheal_fio2,
        ce_roc=ce_roc_eff,
        neuromuscular_strength=sux_strength * phase2_central * engine.malignant_hyperthermia.respiratory_excursion_fraction,
        muscle_fatigue=engine.resp_mech.effort.muscle.fatigue,
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
        hematocrit=hct,
        blood_volume_ml=engine.hemo.blood_volume,
        cardiac_output=cardiac_output,
        metabolic_factor=max(0.25, engine.thermal.metabolic_factor),
        unconscious=state.loc,
        lung_volume_l=lung_volume_l,
        shunt_fraction=engine.aeration.shunt_fraction,
        temp_c=state.temp_c,
        airway_pressure=engine.resp_mech.peep,
        # Brainstem drive persists until cerebral failure is nearly complete or CO2 narcosis is extreme.
        central_drive_fraction=engine.cerebral.respiratory_drive * engine.toxicity.respiratory_drive,
        ketamine_cp=engine.pk["ketamine"].state.c1,
        ketamine_ce=state.ketamine_ce,
        mac_n2o=state.mac_n2o,
        n2o_uptake_l_min=agent_uptake,
        fin2=inspired_fn2,
        apneic_flow_limit_l_min=apneic_limit,
        mech_supported_fraction=supported_share,
        mech_va_l_min=mech_va_l_min,
        exchange=cached_exchange,
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
    assisted = sealed and (engine.vent.is_on or assisted_positive_pressure(engine))
    kept = delivered_fraction(engine) if sealed else 0.0
    rr_display, vt_display_ml, mv_display_l_min = measured_ventilation(
        engine, connected, resp_state.rr, resp_state.vt / 1000.0, kept=kept if sealed else None,
    )
    vent = engine.vent
    spirometry = vent.monitors
    seal_scale = mask_seal_factor(engine)
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


def _pit_paw_baseline(engine: SimulationEngine) -> float:
    hemo_cfg = engine.hemo.config
    # Alveolar-to-pleural pressure transmission follows Ecw / (EL + Ecw) (Jardin 1985):
    # stiff lungs transmit less alveolar pressure to the thorax, stiff chest walls transmit more.
    aeration = engine.aeration
    cw_stiff = max(0.5, aeration.chest_wall_stiffness)
    lung_comp_ratio = clamp(
        engine.resp_mech.compliance * cw_stiff / max(1e-6, aeration.reference_compliance),
        0.15,
        1.35,
    )
    el_ecw_ratio = clamp(cw_stiff * lung_comp_ratio, 0.15, 2.5)
    trans_scale = clamp(el_ecw_ratio / (0.63 + 0.37 * el_ecw_ratio), 0.30, 1.35)
    return hemo_cfg.pit_0 + CMH2O_TO_MMHG * hemo_cfg.paw_transmission * trans_scale * engine.current_mean_paw


def compute_pit_estimate(engine: SimulationEngine, dt: float, *, smoothed: bool = False) -> float:
    """Estimate intrathoracic pressure (mmHg) from alveolar pressure and inspiratory effort."""
    hemo_cfg = engine.hemo.config
    raw_effort = (
        engine.current_mean_effort
        if smoothed or dt <= 0.0
        else engine.vent.capped_muscle_pressure_area / dt
    )
    effort_pressure = min(raw_effort, hemo_cfg.effort_pressure_max)
    return _pit_paw_baseline(engine) - CMH2O_TO_MMHG * hemo_cfg.effort_transmission * effort_pressure


def compute_pit_estimates(engine: SimulationEngine, dt: float) -> tuple[float, float]:
    """Return (instantaneous, smoothed) intrathoracic pressure estimates (mmHg)."""
    hemo_cfg = engine.hemo.config
    paw_base = _pit_paw_baseline(engine)
    eff_gain = CMH2O_TO_MMHG * hemo_cfg.effort_transmission
    effort_smooth = min(engine.current_mean_effort, hemo_cfg.effort_pressure_max)
    effort_inst = effort_smooth if dt <= 0.0 else min(engine.vent.capped_muscle_pressure_area / dt, hemo_cfg.effort_pressure_max)
    return paw_base - eff_gain * effort_inst, paw_base - eff_gain * effort_smooth


def reference_arterial_pressures(engine: SimulationEngine, hemo_state: HemoState) -> tuple[float, float]:
    """Return reference (SBP, DBP) mmHg including CPR envelope or pulseless stasis."""
    if hemo_state.compressions:
        cpr_sbp, cpr_dbp = compression_pressures(
            hemo_state.cpr_perfusion_map,
            hemo_state.venous_pressure,
            filling_factor=engine.hemo.cpr_filling_factor,
        )
        if hemo_state.intrinsic_co > 0.0 and hemo_state.rhythm_type not in PULSELESS_RHYTHMS:
            i_sbp, i_dbp = engine.arterial_waveform.reference_pressures(
                hemo_state.intrinsic_map,
                hemo_state.sv,
                hemo_state.hr,
                pulse_pressure_factor=hemo_state.pulse_pressure_factor,
                static_pressure=hemo_state.venous_pressure,
            )
            return max(i_sbp, cpr_sbp), max(i_dbp, cpr_dbp)
        return cpr_sbp, cpr_dbp
    if hemo_state.rhythm_type in PULSELESS_RHYTHMS or hemo_state.co <= 0.0 or hemo_state.hr <= 0.0:
        p_stat = float(hemo_state.venous_pressure)
        return p_stat, p_stat
    return engine.arterial_waveform.reference_pressures(
        hemo_state.map,
        hemo_state.sv,
        hemo_state.hr,
        pulse_pressure_factor=hemo_state.pulse_pressure_factor,
        static_pressure=hemo_state.venous_pressure,
    )


def seed_arterial_monitors(
    engine: SimulationEngine,
    hemo_state: HemoState,
    *,
    hr: float | None = None,
    map_mmhg: float | None = None,
    sv_ml: float | None = None,
):
    """Seed cardiac cycle, arterial waveform, and arterial line monitor, projecting into state."""
    cardiac_sample = engine.cardiac_cycle.seed(
        hemo_state.hr if hr is None else hr, hemo_state.rhythm_type,
    )
    arterial_sample = engine.arterial_waveform.seed(
        cardiac_sample,
        hemo_state.map if map_mmhg is None else map_mmhg,
        hemo_state.sv if sv_ml is None else sv_ml,
        static_pressure=hemo_state.venous_pressure,
        pulse_pressure_factor=hemo_state.pulse_pressure_factor,
    )
    project_arterial(engine.state, arterial_sample, engine.art_line.seed(arterial_sample))
    return cardiac_sample


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
        cpr_sbp, cpr_dbp = compression_pressures(
            hemo_state.cpr_perfusion_map, hemo_state.venous_pressure,
            filling_factor=hemo.cpr_filling_factor,
        )
        coronary_sbp = max(coronary_sbp, cpr_sbp) if hemo_state.intrinsic_co > 0.0 else cpr_sbp
        coronary_dbp = max(coronary_dbp, cpr_dbp) if hemo_state.intrinsic_co > 0.0 else cpr_dbp
    elif hemo_state.rhythm_type in PULSELESS_RHYTHMS or hemo_state.co <= 0.0:
        coronary_sbp = coronary_dbp = hemo_state.venous_pressure
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
    drug_bis = engine.bis.compute_bis(
        state.hypnotic_ce, state.mac_sevo, state.ketamine_ce, state.opioid_ce, temp_c=state.temp_c,
    )
    raw_bis = drug_bis * cortical
    bis_val = clamp(
        raw_bis + runtime_core.stimulus_effects(engine, 0.0).bis * cortical,
        0.0,
        100.0,
    )
    hemo_state = engine.hemo.state
    cardiac_sample = seed_arterial_monitors(
        engine, hemo_state, hr=state.hr, map_mmhg=state.map, sv_ml=state.sv,
    )
    engine.airway_sensor.seed(state.paw, state.flow, state.volume)
    state.bis = float(bis_val)
    state.loc = float(loc_val)
    state.tol = float(tol_val)
    state.capno_co2 = float(engine.capno.co2) if circuit_connected(engine) else 0.0
    state.ecg_voltage = state.pleth_voltage = 0.0
    state.display_hr = float(cardiac_sample.display_hr)
    state.display_bis = float(bis_val)
    engine.etco2_readout.seed(
        state.etco2,
        et_gases=(state.et_o2, state.et_sevo, state.et_n2o, state.et_mac),
    )
    state.etco2_signal_valid = engine.etco2_readout.valid
    if not engine.etco2_readout.gas_valid:
        state.et_o2 = state.et_sevo = state.et_n2o = state.et_mac = 0.0
    state.display_etco2 = float(state.etco2)
    if state.spo2_signal_valid:
        state.display_spo2 = float(state.spo2)
        engine.spo2_mon.display_saturation = float(state.spo2)
    else:
        engine.spo2_mon.display_saturation = None
        state.spo2 = state.display_spo2 = 0.0
    prop_bis, adjunct_bis = bis_hypnotic_components(engine)
    engine.bis.initialize(
        drug_bis,
        cortical_function=cortical,
        ce_prop=prop_bis,
        ce_adjunct=adjunct_bis,
        mac_sevo=state.mac_sevo,
        ce_ketamine=state.ketamine_ce,
        ce_opioid=state.opioid_ce,
        temp_c=state.temp_c,
    )
    engine.smooth_bis = state.bis
    engine._bis_noise = 0.0


def sync_state_from_models(engine: SimulationEngine) -> None:
    """Derive the public SimulationState from current subsystem state."""
    state = engine.state
    state.temp_c = float(state.temp_c)
    mix = inspired_gas_mixture(engine)
    state.fio2 = float(mix[0])
    state.fi_sevo = float(mix[1] * 100.0)
    state.fi_n2o = float(mix[2] * 100.0)
    engine._tracheal_gas = mix
    sync_inhaled_agents(engine)
    sync_pk_state(engine)
    state.shivering = float(engine.thermal.shiver_level)
    engine.thermal.update_depth(state.temp_c, thermal_hypnotic_ce(engine), thermal_mac(engine))
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
    state.st_depression_mv = _projected_st_depression(hemo, hemo_state)
