import numpy as np
import pytest
from scipy.linalg import expm

from anasim.core.drug_registry import DRUG_REGISTRY
from anasim.core.projection import central_muscle_strength
from anasim.core.runtime import update_pk_covariates
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.patient.pd.nmba import TOFModel
from anasim.patient.pk_models import (
    AtropinePK,
    EpinephrinePK,
    EsmololPK,
    EtomidatePK,
    FentanylPK,
    GlycopyrrolatePK,
    HydromorphonePK,
    KetaminePK,
    LabetalolPK,
    LidocainePK,
    MammillaryPK,
    MidazolamPK,
    MilrinonePK,
    MorphinePK,
    PropofolPKEleveld,
    RemifentanilPKEleveld,
    RocuroniumPK,
    SuccinylcholinePK,
)


def _infuse(model, seconds: int, rate: float) -> None:
    for _ in range(seconds):
        model.step(1.0, rate)


def _tof_trace(patient, roc_mg_kg, seconds):
    """TOF each second after a rocuronium bolus."""
    pk = RocuroniumPK(patient)
    pd = TOFModel(patient)
    pk.state.c1 = roc_mg_kg * patient.weight / pk.v1
    trace = []
    for _ in range(seconds):
        pk.step(1.0, 0.0)
        trace.append(pd.step_recovery(1.0, pk))
    return np.asarray(trace)


def _first_second(trace, condition, start=0):
    hits = np.flatnonzero(condition(trace[start:]))
    return start + int(hits[0]) if hits.size else None


def test_residual_effect_site_decays_after_plasma_is_empty(patient):
    model = PropofolPKEleveld(patient)
    model.step(1.0, 0.0)
    model.state.ce_resp = 1.0
    model.step(1.0, 0.0)
    assert model.state.c1 == 0.0
    assert 0.0 < model.state.ce_resp < 1.0


@pytest.mark.parametrize("model_type", [RemifentanilPKEleveld, SuccinylcholinePK, EsmololPK])
def test_local_esterase_metabolism_continues_without_delivering_drug_to_tissue(patient, model_type):
    model = model_type(patient)
    model.circulation = 0.0
    model.state.c1 = 10.0
    initial_mass = model.v1 * model.state.c1
    _infuse(model, 60, 0.0)
    assert 0.0 < model.v1 * model.state.c1 < initial_mass
    assert model.state.c2 == model.state.c3 == model.state.ce == 0.0


@pytest.mark.parametrize("circulation", [0.0, 0.2, 1.0])
def test_pk_predictions_follow_delivery_during_arrest_and_cpr(patient, circulation):
    model = EpinephrinePK(patient)
    model.circulation = circulation
    model.state.c1 = 10.0
    matrix, _ = model.get_ss_matrices()
    expected = expm(matrix) @ model.state_vector()
    _infuse(model, 60, 0.0)
    assert model.state_vector() == pytest.approx(expected, rel=0.02, abs=0.001)


@pytest.mark.parametrize("model_type", [EpinephrinePK, RemifentanilPKEleveld, SuccinylcholinePK, KetaminePK])
def test_bolus_pk_with_reduced_volume_and_large_steps(patient, model_type):
    model = model_type(patient)
    model.update_hemodynamics(0.2, 0.5)
    model.state.c1 = 10.0
    matrix, _ = model.get_ss_matrices()
    expected = expm(matrix * 0.5) @ model.state_vector()
    model.step(30.0, 0.0)
    assert model.state_vector() == pytest.approx(expected, rel=0.05, abs=0.01)


def test_eleveld_reference_adult_and_opioid_covariate_match_publication():
    """Eleveld 2018 reference male with and without concomitant opioids."""
    patient = Patient(age=35, weight=70, height=170, sex="male")
    no_opioid = PropofolPKEleveld(patient, concomitant_opioids=False)
    assert no_opioid.v1 == pytest.approx(6.283, abs=0.02)
    assert no_opioid.v2 == pytest.approx(25.501, abs=0.05)
    assert no_opioid.v3 == pytest.approx(272.817, abs=0.6)
    assert no_opioid.k10 * no_opioid.v1 == pytest.approx(1.790, abs=0.03)
    assert no_opioid.k12 * no_opioid.v1 == pytest.approx(1.831, abs=0.03)
    assert no_opioid.k13 * no_opioid.v1 == pytest.approx(1.109, abs=0.03)
    assert no_opioid.ke0 == pytest.approx(0.146, abs=0.002)

    with_opioid = PropofolPKEleveld(patient)
    assert with_opioid.v3 == pytest.approx(168.19, abs=0.05)
    assert with_opioid.cl1 == pytest.approx(1.6193, abs=0.001)
    assert with_opioid.cl3 == pytest.approx(0.7713, abs=0.002)
    assert with_opioid.v1 == no_opioid.v1 and with_opioid.v2 == no_opioid.v2
    assert with_opioid.ke0 == no_opioid.ke0


def test_propofol_half_time_is_context_sensitive(patient):
    """Hughes 1992: propofol CSHT is 10-40 min after 2 h and rises with duration."""
    model = PropofolPKEleveld(patient)
    rate = 10.0 * patient.weight / 3600.0
    _infuse(model, 2 * 3600, rate)
    two_hours = model.simulate_decay(target_fraction=0.5, max_seconds=3600)
    _infuse(model, 2 * 3600, rate)
    four_hours = model.simulate_decay(target_fraction=0.5, max_seconds=3600)
    assert 6 < two_hours < 45
    assert four_hours > two_hours


def test_decay_prediction_preserves_basal_input_and_reports_first_crossing():
    model = MammillaryPK(v1=1.0, cl1=1.0, ke0=1.0, basal_input_rate=1.0 / 60.0)
    model.state.c1 = model.state.ce = 3.0
    initial = model.state_vector()
    # Ce(t) = 1 + 2(1 + t) exp(-t), with t in minutes. Ce reaches 1.5
    # after 161.56 seconds; basal input keeps it above 1 indefinitely.
    assert model.simulate_decay(max_seconds=600) == 162.0 / 60.0
    assert model.simulate_decay(target_fraction=0.25, max_seconds=600) == 10.0
    np.testing.assert_array_equal(model.state_vector(), initial)

    redist = MammillaryPK(v1=1.0, cl1=0.05, ke0=4.0, v2=10.0, cl2=0.2)
    redist.set_state_vector([0.0, 4.0, 1.0])
    # Ce drops below 0.5 at 12 s, then rises above 1 as peripheral drug returns.
    assert redist.simulate_decay(max_seconds=120) == 12 / 60


def test_remifentanil_reference_pk_and_context_insensitive_offset():
    """Eleveld 2017 Table 3; Egan 1993 and Kapila 1995 report a 3-5 min offset."""
    model = RemifentanilPKEleveld(Patient(age=35, weight=70, height=170, sex="male"))
    assert (model.v1, model.v2, model.v3) == pytest.approx((5.81, 8.82, 5.03))
    assert (model.cl1, model.cl2, model.cl3) == pytest.approx((2.58, 1.72, 0.124))
    assert model.ke0 == pytest.approx(1.09)
    rate = 0.1 * 70 / 60.0
    _infuse(model, 15 * 60, rate)
    short = model.simulate_decay(max_seconds=1200)
    _infuse(model, (240 - 15) * 60, rate)
    long = model.simulate_decay(max_seconds=1200)
    assert model.state.c1 == pytest.approx(7.0 / 2.58, rel=0.01)
    assert 2.0 < short < 6.0 and 2.0 < long < 6.0
    assert long == pytest.approx(short, abs=2.0)


def test_reciprocal_propofol_midazolam_pk_preserves_mass_and_reverses(patient):
    """Vuyk 2009 and Lichtenbelt 2010, drug-only columns in Table 2.

    Relative covariates transfer to AnaSim's baseline models. Changing V1
    must preserve drug amount, including when blood volume also changes.
    """
    prop, mid = PropofolPKEleveld(patient), MidazolamPK(patient)
    prop_cl = (prop.cl1, prop.cl2, prop.cl3)
    mid_cl = (mid.cl1, mid.cl2)
    mid.state.c1, mid.state.c2, mid.state.c3 = 100.0, 40.0, 10.0
    mid_mass = mid.v1 * mid.state.c1 + mid.v2 * mid.state.c2 + mid.v3 * mid.state.c3
    prop.update_midazolam(225.0)
    mid.update_propofol(1.2)
    assert (prop.elimination_clearance, prop.cl2, prop.cl3) == pytest.approx(
        np.asarray(prop_cl) * np.exp(-225.0 * np.asarray((0.000820, 0.00274, 0.00142)))
    )
    assert (mid.elimination_clearance, mid.cl2) == pytest.approx(
        np.asarray(mid_cl) * np.exp(-1200.0 * np.asarray((0.000205, 0.000232)))
    )
    assert mid.v1 / mid.v1_base == pytest.approx(0.581, abs=0.001)
    mid.update_hemodynamics(0.6, 0.7, blood_volume_l=patient.estimate_blood_volume() / 1000.0)
    mid.update_propofol(8.0)  # Higher concentrations retain the bounded covariate.
    assert mid.v1 * mid.state.c1 + mid.v2 * mid.state.c2 + mid.v3 * mid.state.c3 == pytest.approx(mid_mass)
    mid.update_propofol(0.0)
    mid.update_hemodynamics(1.0, 1.0)
    prop.update_midazolam(0.0)
    assert mid.v1 == mid.v1_base and mid.state.c1 == pytest.approx(100.0)
    assert (prop.elimination_clearance, prop.cl2, prop.cl3) == pytest.approx(prop_cl)


def test_hemorrhage_increases_anesthetic_exposure_and_resuscitation_restores_pk(patient):
    """Johnson 2001, 2003, 2004: swine-informed hemorrhagic shock PK covariates; crystalloid resuscitation restores clearance (propofol data in Johnson 2004, extended to remifentanil)."""
    for model_type in (PropofolPKEleveld, RemifentanilPKEleveld):
        normal, shock = model_type(patient), model_type(patient)
        shock.update_hemodynamics(0.5, 0.4, blood_volume_l=patient.estimate_blood_volume() / 1000.0)
        normal_cl = (normal.cl1, normal.cl2, normal.cl3)
        if model_type is RemifentanilPKEleveld:
            assert (shock.cl2, shock.cl3, shock.ke0) == (normal.cl2, normal.cl3, normal.ke0)
        _infuse(normal, 600, 1.0)
        _infuse(shock, 600, 1.0)
        assert shock.state.c1 > 1.4 * normal.state.c1
        shock.update_hemodynamics(1.0, 1.0, blood_volume_l=patient.estimate_blood_volume() / 1000.0)
        assert (shock.cl1, shock.cl2, shock.cl3) == pytest.approx(normal_cl)
        _infuse(normal, 3600, 1.0)
        _infuse(shock, 3600, 1.0)
        assert shock.state.c1 == pytest.approx(normal.state.c1, rel=0.08)


def test_organ_impairment_scales_pk():
    """Organ impairment affects each drug's distribution and elimination routes."""
    normal = Patient(age=40, weight=70, height=170, sex="male")
    renal = Patient(age=40, weight=70, height=170, sex="male", renal_function=0.4)
    both = Patient(
        age=40, weight=70, height=170, sex="male", renal_function=0.4, hepatic_function=0.5
    )

    assert PropofolPKEleveld(both).v1 > PropofolPKEleveld(normal).v1 * 1.5
    assert PropofolPKEleveld(renal).k10 == pytest.approx(PropofolPKEleveld(normal).k10)
    assert RocuroniumPK(both).v1 > RocuroniumPK(normal).v1 * 1.3
    assert RocuroniumPK(both).k10 < RocuroniumPK(normal).k10 * 0.7
    # Renal perfusion acts only on the renal clearance left after impairment.
    for model_type, fraction in (
        (AtropinePK, 0.568), (GlycopyrrolatePK, 0.7), (MilrinonePK, 0.9), (RocuroniumPK, 0.2),
    ):
        normal_model, renal_model = model_type(normal), model_type(renal)
        assert renal_model.cl1 == pytest.approx(normal_model.cl1 * (1.0 - fraction + fraction * 0.4))
        renal_model.update_hemodynamics(1.0, 1.0, renal_perfusion=0.5)
        assert renal_model.cl1 == pytest.approx(normal_model.cl1 * (1.0 - fraction + fraction * 0.4 * 0.5))

    impaired = Patient(age=80, weight=50, height=165, sex="male", renal_function=0.4, hepatic_function=0.5)
    obese = Patient(age=40, weight=150, height=170, sex="male")
    m_norm, m_obese = MorphinePK(normal), MorphinePK(obese)
    assert 1.25 < m_obese.v1 / m_norm.v1 < 1.55
    assert m_obese.v2 / m_norm.v2 == pytest.approx(m_obese.v1 / m_norm.v1)
    assert m_obese.cl2 / m_norm.cl2 == pytest.approx((m_obese.v1 / m_norm.v1) ** 0.75)
    assert MorphinePK(impaired).v1 < m_norm.v1 and MorphinePK(impaired).cl1 < 0.55 * m_norm.cl1
    assert HydromorphonePK(impaired).v1 < HydromorphonePK(normal).v1
    assert HydromorphonePK(impaired).cl1 < 0.55 * HydromorphonePK(normal).cl1
    assert MorphinePK(renal).cl1 / m_norm.cl1 < HydromorphonePK(renal).cl1 / HydromorphonePK(normal).cl1
    assert FentanylPK(impaired).cl1 < FentanylPK(normal).cl1
    assert FentanylPK(obese).v1 < 1.9 * FentanylPK(normal).v1
    assert RocuroniumPK(impaired).cl1 < 0.65 * RocuroniumPK(normal).cl1

    f_lean = FentanylPK(Patient(age=40, weight=68.5, height=170.0, sex="female"))
    f_obese = FentanylPK(Patient(age=40, weight=117.3, height=165.0, sex="female"))
    assert 1.32 < f_obese.cl1 / f_lean.cl1 < 1.48


def test_lidocaine_disposition_in_cirrhosis_renal_failure_and_heart_failure(engine_factory):
    """Stenson 1971 and Thomson 1973: cirrhosis expands lidocaine Vdss ~1.75x
    while reducing plasma clearance ~40% and prolonging elimination; severe renal
    impairment leaves Vdss and clearance unchanged; low cardiac output raises
    arterial concentrations during constant infusion in proportion to reduced flow.
    """
    normal = Patient(age=52, weight=70.0, height=172.0, sex="male")
    cirrhosis = Patient(age=52, weight=70.0, height=172.0, sex="male", hepatic_function=0.5)
    renal = Patient(
        age=52, weight=70.0, height=172.0, sex="male",
        renal_function=0.4, creatinine_clearance_ml_min=10.0,
    )
    pk_norm, pk_cirr, pk_renal = LidocainePK(normal), LidocainePK(cirrhosis), LidocainePK(renal)
    vdss_norm = pk_norm.v1 + pk_norm.v2 + pk_norm.v3
    vdss_cirr = pk_cirr.v1 + pk_cirr.v2 + pk_cirr.v3
    vdss_renal = pk_renal.v1 + pk_renal.v2 + pk_renal.v3

    assert 1.12 <= pk_cirr.v1 / pk_norm.v1 <= 1.20
    assert 1.70 <= vdss_cirr / vdss_norm <= 1.80
    assert 0.50 <= pk_cirr.cl1 / pk_norm.cl1 <= 0.65
    assert vdss_renal == pytest.approx(vdss_norm) and pk_renal.cl1 == pytest.approx(pk_norm.cl1)

    for pk in (pk_norm, pk_cirr):
        pk.state.c1 = 50.0 / pk.v1
    c30_norm = c30_cirr = 0.0
    for minute in range(1, 241):
        pk_norm.step(60.0, 0.0)
        pk_cirr.step(60.0, 0.0)
        if minute == 30:
            c30_norm, c30_cirr = pk_norm.state.c1, pk_cirr.state.c1
    assert c30_cirr < c30_norm  # Larger distribution space lowers early post-bolus levels.
    assert pk_cirr.state.c1 > pk_norm.state.c1  # Slower elimination crosses above normal later.
    assert (pk_cirr.state.c1 / c30_cirr) > 2.0 * (pk_norm.state.c1 / c30_norm)

    # Stenson 1971: 50 mg bolus + 40 mcg/kg/min (2.4 mg/kg/h) for 100 min;
    # whole-blood levels (~0.8x plasma base) were 1.5 vs 2.4 mcg/mL at CI 3.3 vs 1.9 L/min/m^2.
    pk_ci_norm = LidocainePK(normal)
    pk_ci_low = LidocainePK(normal)
    pk_ci_norm.update_hemodynamics(1.0, 3.3 / 3.0)
    pk_ci_low.update_hemodynamics(1.0, 1.9 / 3.0)
    for pk in (pk_ci_norm, pk_ci_low):
        pk.step(60.0, 50.0 / 60.0)
        for _ in range(99):
            pk.step(60.0, 0.04 * 70.0 / 60.0)
    blood_norm = 0.8 * pk_ci_norm.state.c1
    blood_low = 0.8 * pk_ci_low.state.c1
    assert 1.3 <= blood_norm <= 1.9 and 2.0 <= blood_low <= 2.8
    assert blood_low / blood_norm > 1.35

    hf = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True, age=58, weight=70.0, height=172.0, sex="male", heart_failure="hfref",
    )
    ctrl = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True, age=58, weight=70.0, height=172.0, sex="male",
    )
    for eng in (ctrl, hf):
        eng.give_drug_bolus("lidocaine", 50.0)
        eng.set_drug_rate("lidocaine", 2.4)
        eng.step(1.0)
        for _ in range(100):
            eng.pk["lidocaine"].step(60.0, eng.infusion_rates["lidocaine"])
    assert hf.pk["lidocaine"].state.c1 > 1.22 * ctrl.pk["lidocaine"].state.c1


def test_measured_creatinine_clearance_separates_reversal_from_complex_elimination():
    """Kleijn 2011: renal impairment prolongs complex retention, with rapid reversal.

    Compare equal-age adults at measured CrCl 10 and 119 mL/min, keeping the
    other organ-function factors equal to isolate the published CrCl covariates.
    """
    outcomes = {}
    for clearance in (10.0, 119.0):
        patient = Patient(age=43, weight=70, creatinine_clearance_ml_min=clearance)
        roc = RocuroniumPK(patient)
        pd = TOFModel(patient)
        roc.state.c1 = 42.0 / roc.v1
        reversal_s = None
        for second in range(1, 7201):
            roc.step(1.0, 0.0)
            tof = pd.step_recovery(1.0, roc)
            if second == 1800:
                assert tof < 10.0
                total_before = roc.state.c1 + pd.bound_rocuronium(roc.v1)
                pd.give_sugammadex(280.0)
                roc.step(0.01, 0.0)
                pd.step_recovery(0.01, roc)
                assert roc.state.c1 + pd.bound_rocuronium(roc.v1) == pytest.approx(total_before, rel=0.02)
            elif second > 1800 and reversal_s is None and tof >= 90.0:
                reversal_s = second - 1800
        assert reversal_s is not None and 20 < reversal_s < 180
        complex_ = pd.complex
        bound_amount = complex_.state.c1 * complex_.v1 + complex_.state.c2 * complex_.v2
        outcomes[clearance] = roc.v1, complex_.elimination_clearance, bound_amount, reversal_s
    assert outcomes[10.0][0] > outcomes[119.0][0]  # Rocuronium central distribution volume.
    assert outcomes[10.0][1] < 0.1 * outcomes[119.0][1]
    assert outcomes[10.0][2] > 1.5 * outcomes[119.0][2]
    assert abs(outcomes[10.0][3] - outcomes[119.0][3]) < 30

    # Kleijn 2011: complex clears ~1/3 as fast as free rocuronium, raising total plasma rocuronium.
    totals, free = [], []
    for sug_mg_kg in (0.0, 4.0):
        pt = Patient()
        roc = RocuroniumPK(pt)
        pd = TOFModel(pt)
        roc.state.c1 = 0.6 * pt.weight / roc.v1
        for _ in range(180):
            roc.step(1.0, 0.0)
            pd.step_recovery(1.0, roc)
        pd.give_sugammadex(sug_mg_kg * pt.weight)
        for _ in range(1800):
            roc.step(1.0, 0.0)
            pd.step_recovery(1.0, roc)
        totals.append(roc.state.c1 + pd.bound_rocuronium(roc.v1))
        free.append(roc.state.c1)
    assert totals[1] > 1.4 * totals[0]
    assert free[1] < 0.5 * free[0]


@pytest.mark.parametrize("volume_ratio", [0.5, 1.5])
def test_hemodynamic_rescaling_conserves_drug_mass(awake_engine, volume_ratio):
    """Changing effective V1 must not act as an unlogged drug dose or loss."""
    for spec in DRUG_REGISTRY:
        pk = awake_engine.pk[spec.key]
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


def test_shed_blood_removes_drug_without_instantaneous_concentration_rise(awake_engine):
    engine = awake_engine
    models = (engine.pk["propofol"], engine.pk["roc"], engine.tof_pd.sugammadex, engine.tof_pd.complex)
    for model in models:
        model.state.c1, model.state.c2, model.state.ce = 4.0, 2.0, 3.0
    masses = [model.v1 * model.state.c1 for model in models]
    engine.hemo.add_volume(-2000)
    update_pk_covariates(engine, engine.state.co)
    for model, mass in zip(models, masses, strict=True):
        assert model.state.c1 == pytest.approx(4.0)
        assert model.v1 * model.state.c1 < mass
        assert model.state.c2 == 2.0 and model.state.ce == 3.0
    update_pk_covariates(engine, engine.state.co)
    for model in models:
        assert model.state.c1 == pytest.approx(4.0)
    engine.hemo.add_volume(2000)
    update_pk_covariates(engine, engine.state.co)
    for model, mass in zip(models, masses, strict=True):
        assert model.state.c1 < 4.0
        assert model.v1 * model.state.c1 < mass


def test_rocuronium_spontaneous_recovery_slows_with_age(patient_factory):
    """Kleijn 2011: after 0.6 mg/kg under propofol, TOF ratio 0.9 at about 40-80 min.
    Clearance is 25% lower over 65 years (Varrique 2016), so recovery is later.
    """
    recovery = {}
    for age in (30, 70):
        tof = _tof_trace(patient_factory(age=age), 0.6, 9000)
        block = _first_second(tof, lambda x: x < 5.0)
        assert block is not None and block < 180
        recovery[age] = _first_second(tof, lambda x: x >= 90.0, start=block)
    assert 40 * 60 < recovery[30] < 80 * 60
    assert recovery[70] > recovery[30] + 10 * 60

    # Meyhoff 2009: in morbid obesity (IBW cohort median 132 kg, 173 cm, age 37),
    # 0.6 mg/kg IBW (42 mg, median TOF 0.9 at 63 min, range 43-107) recovers
    # meaningfully faster than 0.6 mg/kg IBW + 40% excess (56 mg, median 76 min).
    obese_pt = patient_factory(age=37, weight=132.0, height=173.0, sex="female")
    ibw_trace = _tof_trace(obese_pt, 42.0 / 132.0, 6000)
    cbw40_trace = _tof_trace(obese_pt, 56.0 / 132.0, 6000)
    ibw_rec = _first_second(ibw_trace, lambda x: x >= 90.0, start=180)
    cbw40_rec = _first_second(cbw40_trace, lambda x: x >= 90.0, start=180)
    assert ibw_rec is not None and cbw40_rec is not None
    assert 45 * 60 < ibw_rec < 68 * 60
    assert cbw40_rec > ibw_rec + 10 * 60


@pytest.mark.parametrize(
    ("roc_mg_kg", "given_at_s", "sugammadex_mg_kg", "max_reversal_s"),
    [
        (0.6, 1800, 2.0, 204),  # Moderate block: observed median 1.7 min, range 0.9-3.4
        (0.6, 900, 4.0, 180),  # Deep block: predicted median 1.5 min
        (1.2, 180, 16.0, 120),  # Immediate reversal: predicted 0.8 min, observed 1.3
    ],
)
def test_sugammadex_reversal_time(patient, roc_mg_kg, given_at_s, sugammadex_mg_kg, max_reversal_s):
    """Time to TOF ratio 0.9 after sugammadex (Kleijn 2011, Tables 5 and 6)."""
    roc = RocuroniumPK(patient)
    pd = TOFModel(patient)
    roc.state.c1 += roc_mg_kg * patient.weight / roc.v1
    for _ in range(given_at_s):
        roc.step(1.0, 0.0)
        tof = pd.step_recovery(1.0, roc)
    assert tof < 10.0

    pd.give_sugammadex(sugammadex_mg_kg * patient.weight)
    recovered = None
    for elapsed in range(max_reversal_s + 1):
        if tof >= 90.0:
            recovered = elapsed
            break
        roc.step(1.0, 0.0)
        tof = pd.step_recovery(1.0, roc)
    assert recovered is not None
    assert 20 <= recovered <= max_reversal_s


class TestNeuromuscularEffectSite:
    """Free rocuronium drives the adductor pollicis and the central muscles."""

    def test_sugammadex_restores_spontaneous_breathing(self, anesthetized_engine):
        engine = anesthetized_engine
        engine.give_drug_bolus("roc", 0.6 * engine.patient.weight)
        for _ in range(300):
            engine.step(2.0)
        engine.set_drug_rate("propofol", 0.0)
        engine.set_drug_rate("remi", 0.0)
        for _ in range(300):
            engine.step(2.0)
        assert engine.state.tof < 5.0
        assert engine.resp.state.muscle_factor < 0.5

        engine.give_drug_bolus("sugammadex", 4.0 * engine.patient.weight)
        for _ in range(90):
            engine.step(2.0)
        assert engine.state.tof > 90.0
        assert engine.resp.state.muscle_factor > 0.95
        assert engine.state.roc_cp > engine.pk["roc"].state.c1
        engine.set_vent_power(False)
        for _ in range(90):
            engine.step(2.0)
        assert engine.resp.state.mv > 3.0
        assert engine.state.mv > 3.0
        assert engine.state.etco2_signal_valid

    def test_diaphragm_recovers_before_adductor_pollicis(self, anesthetized_engine):
        """The diaphragm needs about 1.8x the adductor concentration (Cantineau 1994a)."""
        engine = anesthetized_engine
        engine.give_drug_bolus("roc", 0.6 * engine.patient.weight)
        for _ in range(60):
            engine.step(2.0)
        tof_when_breathing_recovers = None
        for _ in range(3000):
            engine.step(2.0)
            if engine.resp.state.muscle_factor > 0.5:
                tof_when_breathing_recovers = engine.state.tof
                break
        assert tof_when_breathing_recovers is not None
        assert tof_when_breathing_recovers < 25.0

        for ce_central in (0.6, 1.8):
            engine.tof_pd.ce_central = ce_central
            strengths = []
            for mac in (0.0, 0.5, 1.0):
                engine.state.mac_sevo = mac
                strengths.append(central_muscle_strength(engine))
            assert strengths[0] > strengths[1] > strengths[2]

        # Sugammadex V1 clamps at extreme body weight, and zero circulation halts TOF decay projection.
        assert (
            TOFModel(Patient(weight=200.0, height=190.0)).sugammadex.v1
            >= TOFModel(Patient(weight=150.0, height=190.0)).sugammadex.v1 * 0.95
        )
        roc_zero = RocuroniumPK(Patient())
        tof_zero = TOFModel(Patient())
        roc_zero.state.c1 = 5.0
        roc_zero.circulation = 0.0
        tof_zero.ce = 3.0
        assert tof_zero.simulate_decay(roc_zero, max_seconds=1800) == pytest.approx(30.0)


def test_obesity_lean_pk_scaling_prevents_200kg_underdosing_and_extreme_clearances(engine_factory):
    """Ingrande 2010: Lean-weight scaling prevents 200 kg supraphysiologic clearances and induction failure."""
    p70 = Patient(age=40, weight=70, height=175, sex="male")
    p200 = Patient(age=40, weight=200, height=185, sex="male")
    assert EtomidatePK(p200).cl1 < 2.8
    assert LabetalolPK(p200).cl1 < 2.3
    assert KetaminePK(p200).cl1 < 2.4
    assert AtropinePK(p200).v1 < 1.9 * AtropinePK(p70).v1

    eto = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True, age=40, weight=200, height=185, sex="male",
    )
    eto.give_drug_bolus("etomidate", 25.0)
    max_loc = 0.0
    for _ in range(180):
        eto.step(1.0)
        max_loc = max(max_loc, eto.state.loc)
    assert max_loc > 0.80

    atr = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True, age=40, weight=200, height=185, sex="male", baseline_hr=50.0,
    )
    hr0 = atr.state.hr
    atr.give_drug_bolus("atropine", 1.0)
    peak_hr = hr0
    for _ in range(240):
        atr.step(1.0)
        peak_hr = max(peak_hr, atr.state.hr)
    assert peak_hr > hr0 + 10.0


def test_low_dose_sugammadex_produces_recurarization_rebound(patient):
    """Eleveld 2007: subtherapeutic 0.5 mg/kg sugammadex under deep rocuronium rebounds."""
    roc = RocuroniumPK(patient)
    pd = TOFModel(patient)
    roc.state.c1 += 0.9 * patient.weight / roc.v1
    for _ in range(252):
        roc.step(10.0, 0.0)
        tof = pd.step_recovery(10.0, roc)
    assert tof < 5.0

    pd.give_sugammadex(0.5 * patient.weight)
    peak_tof = 0.0
    min_after_tof = 100.0
    for step_idx in range(1, 361):
        roc.step(5.0, 0.0)
        tof = pd.step_recovery(5.0, roc)
        if step_idx <= 60:
            peak_tof = max(peak_tof, tof)
        else:
            min_after_tof = min(min_after_tof, tof)
    assert peak_tof > 50.0
    assert min_after_tof < peak_tof - 8.0


def test_sevoflurane_slows_rocuronium_washout_more_than_tiva(engine_factory, advance_time):
    """Sevoflurane slows rocuronium effect-site washout relative to propofol TIVA (Kleijn 2011; Lowry 1998)."""
    tiva = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1), start=True)
    sevo = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1), start=True)
    for eng in (tiva, sevo):
        eng.give_drug_bolus("roc", 42.0)
        advance_time(eng, 180.0, dt=1.0)
    advance_time(tiva, 600.0, dt=2.0)
    advance_time(sevo, 600.0, dt=2.0)
    assert sevo.tof_pd.ce > tiva.tof_pd.ce


def test_morphine_overdose_causes_apnea_and_esrd_accumulates_m6g(engine_factory, advance_time):
    """High-dose morphine causes apnea; ESRD accumulates active M6G with longer CSHT and deeper late depression."""
    overdose = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    overdose.give_drug_bolus("morphine", 40.0)
    apnea_seen = False
    for _ in range(1650):
        overdose.step(2.0)
        if overdose.state.rr == 0.0 or overdose.state.apnea:
            apnea_seen = True
    assert apnea_seen
    assert overdose.state.pa_co2 > 75.0

    normal_renal = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    esrd = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, renal_function=0.05)
    normal_renal.give_drug_bolus("morphine", 20.0)
    esrd.give_drug_bolus("morphine", 20.0)
    # Advance morphine PK through 6 h so M6G accumulates in ESRD, then verify CSHT and full-engine respiration.
    norm_pk = normal_renal.pk["morphine"]
    esrd_pk = esrd.pk["morphine"]
    assert isinstance(norm_pk, MorphinePK) and isinstance(esrd_pk, MorphinePK)
    for _ in range(360):
        norm_pk.step(60.0, 0.0)
        esrd_pk.step(60.0, 0.0)
    assert esrd_pk.m6g_excess_ce > norm_pk.m6g_excess_ce + 8.0
    assert esrd_pk.simulate_decay(max_seconds=28800) > norm_pk.simulate_decay(max_seconds=28800) + 30.0
    advance_time(normal_renal, 600.0, dt=2.0)
    advance_time(esrd, 600.0, dt=2.0)
    assert esrd.state.pa_co2 > normal_renal.state.pa_co2 + 8.0
    assert esrd.state.sao2 < normal_renal.state.sao2 - 20.0


def test_rocuronium_hypothermia_cirrhosis_remifentanil_obesity_phase2_and_etomidate_seizures(
    engine_factory, advance_time
):
    """Hypothermia slows rocuronium onset (extrapolated from vecuronium in Caldwell 2000), cirrhosis expands V1/V2 (Khuenl-Brady and Sparr 1996; Magorian 1995),
    Eleveld remifentanil V3 caps in extreme obesity (Egan 1998) with Table 5 ke0 omega,
    succinylcholine Phase II fades central strength, and etomidate suppresses LAST seizures."""
    from anasim.patient.variability import ELEVELD_REMIFENTANIL_OMEGA
    from anasim.physiology.toxicity import LocalAnestheticToxicity

    onsets = []
    for temp in (37.0, 33.0):
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1),
            start=True,
            baseline_temp=temp,
        )
        eng.give_drug_bolus("roc", 0.6 * eng.patient.weight)
        t95 = 300.0
        for step_idx in range(1, 301):
            eng.state.temp_c = temp
            eng.step(1.0)
            if eng.state.twitch < 5.0:
                t95 = float(step_idx)
                break
        onsets.append(t95)
    assert onsets[1] > onsets[0] + 15.0

    roc_norm = RocuroniumPK(Patient())
    roc_cirr = RocuroniumPK(Patient(hepatic_function=0.5))
    assert roc_cirr.v1 > roc_norm.v1 * 1.10
    assert roc_cirr.v2 > roc_norm.v2 * 1.25

    remi_obese = RemifentanilPKEleveld(Patient(weight=200.0, height=185.0))
    assert remi_obese.v3 < 4.5
    assert ELEVELD_REMIFENTANIL_OMEGA["ke0"] == pytest.approx(0.947)

    sux_eng = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    for _ in range(8):
        sux_eng.give_drug_bolus("succinylcholine", 100.0)
        advance_time(sux_eng, 45.0, dt=1.0)
    advance_time(sux_eng, 600.0, dt=2.0)
    assert sux_eng.state.tof < 70.0
    assert sux_eng.resp.state.muscle_factor < 0.90

    tox_awake = LocalAnestheticToxicity(70.0)
    tox_etom = LocalAnestheticToxicity(70.0)
    for _ in range(30):
        tox_awake.step(
            1.0, neural_ce=9.0, cardiac_ce=9.0, ph=7.4,
            propofol_ce=0.0, midazolam_ce=0.0, mac_sevo=0.0,
            motor_strength=1.0, cortical_function=1.0,
        )
        tox_etom.step(
            1.0, neural_ce=9.0, cardiac_ce=9.0, ph=7.4,
            propofol_ce=0.0, midazolam_ce=0.0, mac_sevo=0.0,
            motor_strength=1.0, cortical_function=1.0, etomidate_ce=0.6,
        )
    assert tox_awake.seizure_activity > 0.5
    assert tox_etom.seizure_activity < tox_awake.seizure_activity * 0.35
