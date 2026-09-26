import numpy as np
import pytest

from anasim.core.drug_registry import DRUG_REGISTRY
from anasim.patient.patient import Patient
from anasim.patient.pd.nmba import TOFModel
from anasim.patient.pk_models import (
    MilrinonePK,
    NorepinephrinePK,
    PropofolPKEleveld,
    PropofolPKMarsh,
    RemifentanilPKMinto,
    RocuroniumPK,
)


def _infuse(model, seconds: int, rate: float, **kwargs) -> None:
    for _ in range(seconds):
        model.step(1.0, rate, **kwargs)


def _tof_trace(patient, roc_mg_kg, seconds, sugammadex=None):
    """TOF each second after a rocuronium bolus, computed as the engine does.

    ``sugammadex`` is an optional ``(time_s, mg_kg)`` pair.
    """
    pk = RocuroniumPK(patient)
    pd = TOFModel(patient)
    pk.state.c1 = roc_mg_kg * patient.weight / pk.v1
    trace = []
    for t in range(seconds):
        if sugammadex and t == sugammadex[0]:
            pd.give_sugammadex(sugammadex[1] * patient.weight)
        pk.step(1.0, 0.0)
        trace.append(pd.step_recovery(1.0, pk.state.c1))
    return np.asarray(trace)


def _first_second(trace, condition, start=0):
    hits = np.flatnonzero(condition(trace[start:]))
    return start + int(hits[0]) if hits.size else None


def test_eleveld_reference_adult_matches_publication():
    model = PropofolPKEleveld(Patient(age=35, weight=70, height=170, sex="male"))

    assert model.v1 == pytest.approx(6.283, abs=0.02)
    assert model.v2 == pytest.approx(25.501, abs=0.05)
    assert model.v3 == pytest.approx(272.817, abs=0.6)
    assert model.k10 * model.v1 == pytest.approx(1.790, abs=0.03)
    assert model.k12 * model.v1 == pytest.approx(1.831, abs=0.03)
    assert model.k13 * model.v1 == pytest.approx(1.109, abs=0.03)
    assert model.ke0 == pytest.approx(0.146, abs=0.002)


def test_propofol_half_time_is_context_sensitive(patient):
    """Hughes 1992: propofol CSHT is 10-40 min after 2 h and rises with duration."""

    def csht(hours):
        model = PropofolPKMarsh(patient)
        _infuse(model, int(hours * 3600), 10.0 * patient.weight / 3600.0)
        return model.simulate_decay(target_fraction=0.5, max_seconds=3600)

    two_hours = csht(2)
    assert 6 < two_hours < 45
    assert csht(4) > two_hours


def test_remifentanil_half_time_is_context_insensitive(patient):
    """Egan 1993; Kapila 1995: remifentanil CSHT stays near 3-5 min after 4 h."""
    model = RemifentanilPKMinto(patient)
    _infuse(model, 4 * 3600, 0.2 * patient.weight / 60.0)
    assert model.simulate_decay(target_fraction=0.5, max_seconds=1200) < 9


def test_propofol_slows_li_norepinephrine_clearance(patient):
    """Li 2024: propofol plasma concentration is a covariate on clearance."""
    concentrations = []
    for propofol_cp in (0.0, 4.0):
        model = NorepinephrinePK(patient, model="Li")
        _infuse(model, 600, 0.1 * patient.weight / 60.0, propofol_conc_ug_ml=propofol_cp)
        concentrations.append(model.state.c1)
    assert concentrations[1] > concentrations[0]


def test_organ_impairment_scales_pk():
    """Hepatic impairment expands Vd; renal impairment slows rocuronium and
    milrinone clearance but not propofol's."""
    normal = Patient(age=40, weight=70, height=170, sex="male")
    renal = Patient(age=40, weight=70, height=170, sex="male", renal_function=0.4)
    both = Patient(
        age=40, weight=70, height=170, sex="male", renal_function=0.4, hepatic_function=0.5
    )

    assert PropofolPKMarsh(both).v1 > PropofolPKMarsh(normal).v1 * 1.5
    assert PropofolPKMarsh(renal).k10 == pytest.approx(PropofolPKMarsh(normal).k10)
    assert RocuroniumPK(both).v1 > RocuroniumPK(normal).v1 * 1.3
    assert RocuroniumPK(both).k10 < RocuroniumPK(normal).k10 * 0.7
    assert MilrinonePK(renal).k10 < MilrinonePK(normal).k10 * 0.6


@pytest.mark.parametrize("volume_ratio", [0.5, 1.5])
def test_hemodynamic_rescaling_conserves_drug_mass(awake_engine, volume_ratio):
    """Changing effective V1 must not act as an unlogged drug dose or loss."""
    for spec in DRUG_REGISTRY:
        pk = getattr(awake_engine, spec.pk_attr)
        pk.state.c1, pk.state.c2, pk.state.c3, pk.state.ce = 4.0, 2.0, 1.0, 3.0
        original_v1 = pk.v1
        original_mass = pk.v1 * 4.0 + pk.v2 * 2.0 + pk.v3
        pk.update_hemodynamics(volume_ratio, 0.7)
        assert pk.v1 * pk.state.c1 + pk.v2 * pk.state.c2 + pk.v3 * pk.state.c3 == pytest.approx(original_mass)
        assert pk.state.c1 == pytest.approx(4.0 * original_v1 / pk.v1)
        assert (pk.state.c2, pk.state.c3, pk.state.ce) == (2.0, 1.0, 3.0)

        # A clearance-only change and repeated volume update cannot rescale twice.
        cp = pk.state.c1
        pk.update_hemodynamics(volume_ratio, 1.2)
        assert pk.state.c1 == cp
        pk.update_hemodynamics(1.0, 1.0)
        assert pk.state.c1 == pytest.approx(4.0)


def test_rocuronium_duration_and_spontaneous_recovery(patient):
    """0.6 mg/kg: TOF 25% at 25-70 min (Wierda 1991), then TOF ratio 90%."""
    tof = _tof_trace(patient, 0.6, 7200)
    block = _first_second(tof, lambda x: x < 5.0)
    assert block is not None

    duration = _first_second(tof, lambda x: x > 25.0, start=block)
    assert duration is not None and 25 * 60 < duration < 70 * 60
    recovery = _first_second(tof, lambda x: x >= 90.0, start=duration)
    assert recovery is not None and 20 * 60 < recovery < 125 * 60


@pytest.mark.parametrize(
    ("roc_mg_kg", "given_at_s", "sugammadex_mg_kg", "max_reversal_s"),
    [
        (0.6, 900, 2.0, 210),  # Pühringer 2010
        (0.6, 180, 4.0, 240),  # Pühringer 2010, deep block
        (1.2, 180, 16.0, 300),  # Kleijn 2011, immediate reversal
    ],
)
def test_sugammadex_reversal_time(patient, roc_mg_kg, given_at_s, sugammadex_mg_kg, max_reversal_s):
    tof = _tof_trace(
        patient, roc_mg_kg, given_at_s + 360, sugammadex=(given_at_s, sugammadex_mg_kg)
    )
    assert tof[given_at_s - 1] < 10.0

    recovered = _first_second(tof, lambda x: x >= 90.0, start=given_at_s)
    assert recovered is not None
    assert recovered - given_at_s <= max_reversal_s


class TestNeuromuscularEffectSite:
    """Free rocuronium drives the adductor pollicis and the central muscles."""

    def test_central_effect_site_leads_the_adductor_pollicis(self, anesthetized_engine):
        """The central site uses laryngeal kinetics (Plaud 1995: t1/2 ke0 2.7 vs 4.4 min)."""
        engine = anesthetized_engine
        tof_pd = engine.tof_pd
        engine.give_drug_bolus("roc", 0.6 * engine.patient.weight)

        peak_central_s = peak_ap_s = 0.0
        peak_central = peak_ap = 0.0
        for step in range(2400):
            engine.step(0.5)
            if tof_pd.ce_central > peak_central:
                peak_central, peak_central_s = tof_pd.ce_central, (step + 1) * 0.5
            if tof_pd.ce > peak_ap:
                peak_ap, peak_ap_s = tof_pd.ce, (step + 1) * 0.5

        # Onset leads at the central muscles, and they clear the drug first.
        assert peak_central_s < peak_ap_s
        assert tof_pd.ce_central < tof_pd.ce

    def test_sugammadex_restores_spontaneous_breathing(self, anesthetized_engine):
        engine = anesthetized_engine
        engine.give_drug_bolus("roc", 0.6 * engine.patient.weight)
        for _ in range(1200):
            engine.step(0.5)
        engine.disable_tci("propofol")
        engine.disable_tci("remi")
        for _ in range(1200):
            engine.step(0.5)
        assert engine.state.tof < 5.0
        assert engine.resp.state.muscle_factor < 0.5

        engine.give_drug_bolus("sugammadex", 4.0 * engine.patient.weight)
        for _ in range(360):
            engine.step(0.5)
        assert engine.state.tof > 90.0
        assert engine.resp.state.muscle_factor > 0.95

    def test_diaphragm_recovers_before_adductor_pollicis(self, anesthetized_engine):
        """The diaphragm needs about 1.8x the adductor concentration (Cantineau 1994)."""
        engine = anesthetized_engine
        engine.give_drug_bolus("roc", 0.6 * engine.patient.weight)
        for _ in range(240):
            engine.step(0.5)
        tof_when_breathing_recovers = None
        for _ in range(12000):
            engine.step(0.5)
            if engine.resp.state.muscle_factor > 0.5:
                tof_when_breathing_recovers = engine.state.tof
                break
        assert tof_when_breathing_recovers is not None
        assert tof_when_breathing_recovers < 25.0
