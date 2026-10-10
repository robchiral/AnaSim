import numpy as np
import pytest
from scipy.linalg import expm

from anasim.core.drug_registry import DRUG_REGISTRY
from anasim.patient.patient import Patient
from anasim.patient.pd.nmba import TOFModel
from anasim.patient.pk_models import (
    EpinephrinePK,
    MilrinonePK,
    NorepinephrinePK,
    PropofolPKEleveld,
    RemifentanilPKEleveld,
    RocuroniumPK,
)


def _infuse(model, seconds: int, rate: float, **kwargs) -> None:
    for _ in range(seconds):
        model.step(1.0, rate, **kwargs)


def _tof_trace(patient, roc_mg_kg, seconds):
    """TOF each second after a rocuronium bolus."""
    pk = RocuroniumPK(patient)
    pd = TOFModel(patient)
    pk.state.c1 = roc_mg_kg * patient.weight / pk.v1
    trace = []
    for _ in range(seconds):
        pk.step(1.0, 0.0)
        trace.append(pd.step_recovery(1.0, pk.state.c1))
    return np.asarray(trace)


def _first_second(trace, condition, start=0):
    hits = np.flatnonzero(condition(trace[start:]))
    return start + int(hits[0]) if hits.size else None


@pytest.mark.parametrize("model_type", [EpinephrinePK, RemifentanilPKEleveld])
def test_bolus_pk_with_reduced_volume_and_large_steps(patient, model_type):
    model = model_type(patient)
    model.update_hemodynamics(0.2, 0.5)
    model.state.c1 = 10.0
    matrix, _ = model.get_ss_matrices()
    expected = expm(matrix * 0.5) @ model.state_vector()
    model.step(30.0, 0.0)
    assert model.state_vector() == pytest.approx(expected, rel=0.05, abs=0.01)


def test_eleveld_reference_adult_matches_publication():
    model = PropofolPKEleveld(Patient(age=35, weight=70, height=170, sex="male"), concomitant_opioids=False)

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
        model = PropofolPKEleveld(patient)
        _infuse(model, int(hours * 3600), 10.0 * patient.weight / 3600.0)
        return model.simulate_decay(target_fraction=0.5, max_seconds=3600)

    two_hours = csht(2)
    assert 6 < two_hours < 45
    assert csht(4) > two_hours


def test_remifentanil_half_time_is_context_insensitive(patient):
    """Egan 1993; Kapila 1995: remifentanil CSHT stays near 3-5 min after 4 h."""
    model = RemifentanilPKEleveld(patient)
    _infuse(model, 4 * 3600, 0.2 * patient.weight / 60.0)
    assert model.simulate_decay(target_fraction=0.5, max_seconds=1200) < 9


def test_remifentanil_eleveld_reference_adult_matches_publication():
    """Eleveld 2017 Table 3, reference 35-year-old 70-kg 170-cm male."""
    model = RemifentanilPKEleveld(Patient(age=35, weight=70, height=170, sex="male"))
    assert (model.v1, model.v2, model.v3) == pytest.approx((5.81, 8.82, 5.03))
    assert (model.cl1, model.cl2, model.cl3) == pytest.approx((2.58, 1.72, 0.124))
    assert model.ke0 == pytest.approx(1.09)
    _infuse(model, 4 * 3600, 0.1 * 70 / 60.0)
    assert model.state.c1 == pytest.approx(7.0 / 2.58, rel=0.01)


def test_propofol_opioid_covariate_matches_nonmem():
    """Eleveld 2018 supplied control stream, reference male with opioids."""
    patient = Patient(age=35, weight=70, height=170, sex="male")
    model = PropofolPKEleveld(patient)
    assert model.v3 == pytest.approx(168.19, abs=0.05)
    assert model.cl1 == pytest.approx(1.6193, abs=0.001)
    assert model.cl3 == pytest.approx(0.7713, abs=0.002)
    no_opioid = PropofolPKEleveld(patient, concomitant_opioids=False)
    assert model.v1 == no_opioid.v1 and model.v2 == no_opioid.v2
    assert model.ke0 == no_opioid.ke0


def test_propofol_slows_li_norepinephrine_clearance(patient):
    """Li 2024: propofol plasma concentration is a covariate on clearance."""
    concentrations = []
    for propofol_cp in (0.0, 4.0):
        model = NorepinephrinePK(patient)
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

    assert PropofolPKEleveld(both).v1 > PropofolPKEleveld(normal).v1 * 1.5
    assert PropofolPKEleveld(renal).k10 == pytest.approx(PropofolPKEleveld(normal).k10)
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
def test_sugammadex_reversal_time(
    anesthetized_engine, advance_time, roc_mg_kg, given_at_s, sugammadex_mg_kg, max_reversal_s
):
    engine = anesthetized_engine
    engine.give_drug_bolus("roc", roc_mg_kg * engine.patient.weight)
    advance_time(engine, given_at_s)
    assert engine.state.tof < 10.0

    engine.give_drug_bolus("sugammadex", sugammadex_mg_kg * engine.patient.weight)
    recovered = None
    for elapsed in range(max_reversal_s + 1):
        if engine.state.tof >= 90.0:
            recovered = elapsed
            break
        engine.step(1.0)
    assert recovered is not None
    assert recovered <= max_reversal_s


def test_sugammadex_reversal_continues_once_free_rocuronium_reaches_zero(patient):
    model = TOFModel(patient)
    model.ce = model.ce_central = 4.0
    model.give_sugammadex(4.0 * patient.weight)
    for _ in range(4):
        model.step_recovery(30.0, 0.0)
    assert model.compute_tof_from_ce(model.ce) > 60.0


class TestNeuromuscularEffectSite:
    """Free rocuronium drives the adductor pollicis and the central muscles."""

    def test_sugammadex_restores_spontaneous_breathing(self, anesthetized_engine):
        engine = anesthetized_engine
        engine.give_drug_bolus("roc", 0.6 * engine.patient.weight)
        for _ in range(1200):
            engine.step(0.5)
        engine.set_drug_rate("propofol", 0.0)
        engine.set_drug_rate("remi", 0.0)
        for _ in range(1200):
            engine.step(0.5)
        assert engine.state.tof < 5.0
        assert engine.resp.state.muscle_factor < 0.5

        engine.give_drug_bolus("sugammadex", 4.0 * engine.patient.weight)
        for _ in range(360):
            engine.step(0.5)
        assert engine.state.tof > 90.0
        assert engine.resp.state.muscle_factor > 0.95
        engine.set_vent_power(False)
        for _ in range(180):
            engine.step(1.0)
        assert engine.resp.state.mv > 3.0
        assert engine.state.mv > 3.0
        assert engine.state.etco2_signal_valid

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
