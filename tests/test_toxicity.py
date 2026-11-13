import math

import numpy as np
import pytest
from scipy.linalg import expm

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.monitors.ecg import ECGMonitor, displayed_rhythm
from anasim.patient.pk_models import LidocainePK
from anasim.physiology.arrhythmia import antiarrhythmic_effects
from anasim.physiology.toxicity import LocalAnestheticToxicity


def _step(tox, seconds=60, **changes):
    inputs = dict(neural_ce=0.0, cardiac_ce=0.0, ph=7.4, propofol_ce=0.0,
                  midazolam_ce=0.0, mac_sevo=0.0, motor_strength=1.0, cortical_function=1.0)
    inputs.update(changes)
    tox.step(seconds, **inputs)


def test_therapeutic_lidocaine_does_not_cause_toxicity_but_high_exposure_does():
    tox = LocalAnestheticToxicity(70)
    _step(tox, neural_ce=3, cardiac_ce=3)
    assert tox.seizure_activity == tox.conduction_block == tox.contractility_loss == tox.heart_rate_increase == 0
    _step(tox, neural_ce=8, cardiac_ce=8)
    assert tox.heart_rate_increase > 10 and tox.svr_increase > 1.2
    _step(tox, neural_ce=10, cardiac_ce=10)
    assert tox.seizure_activity > 0.95 and tox.unresponsiveness > 0.95
    assert tox.conduction_block == tox.contractility_loss == 0
    _step(tox, neural_ce=50, cardiac_ce=50)
    assert tox.neural_depression == tox.conduction_block == tox.contractility_loss == 1
    assert tox.respiratory_drive == 0 and tox.heart_rate_increase < 0.001


@pytest.mark.parametrize(
    "suppression",
    [{"midazolam_ce": 100}, {"propofol_ce": 4}, {"mac_sevo": 1.5}, {"cortical_function": 0}],
)
def test_anesthetics_and_cortical_failure_suppress_seizures_and_sympathetic_rise(suppression):
    awake = LocalAnestheticToxicity(70)
    _step(awake, neural_ce=8, cardiac_ce=8)
    blunted = LocalAnestheticToxicity(70)
    _step(blunted, neural_ce=8, cardiac_ce=8, **suppression)
    assert blunted.seizure_activity < awake.seizure_activity * 0.25
    assert blunted.heart_rate_increase < awake.heart_rate_increase * 0.25
    assert blunted.svr_increase < awake.svr_increase * 0.25


def test_lipid_needs_circulation_and_does_not_directly_switch_off_seizures():
    tox = LocalAnestheticToxicity(70)
    tox.add_lipid(100)
    for _ in range(150):
        tox.step_lipid(1, 0)
    assert tox.lipid_effect == 0 and tox.lipid_ml == 100
    for _ in range(150):
        tox.step_lipid(1, 1)
    assert 0.75 < tox.effect_input_fraction < 0.95
    _step(tox, neural_ce=10, cardiac_ce=20)
    assert tox.seizure_activity > 0.95
    assert tox.total_lipid_ml == 100


def test_lipid_effect_preserves_drug_mass_and_matches_the_pk_matrix(patient):
    model = LidocainePK(patient)
    model.state.c1 = 20
    model.effect_input_fraction = 0.8
    initial = model.state_vector()
    a, _ = model.get_ss_matrices()
    expected = expm(a * 0.1 / 60) @ initial
    model.step(0.1, 0)
    assert model.state_vector() == pytest.approx(expected, abs=0.002)
    untreated = LidocainePK(patient)
    untreated.state.c1 = 20
    untreated.step(0.1, 0)
    assert model.state.c1 == untreated.state.c1
    assert model.state.ce_cv < untreated.state.ce_cv


@pytest.mark.parametrize("sex", ["male", "female"])
def test_lidocaine_dose_exposure_scales_with_published_fat_free_mass(patient_factory, sex):
    """Foong 2025 Table IV: V1 follows FFM; excess fat does not dilute a bolus proportionally."""
    concentrations = []
    for weight in (70, 120):
        patient = patient_factory(weight=weight, sex=sex)
        model = LidocainePK(patient)
        denominator = 6680 + 216 * patient.bmi if sex == "male" else 8780 + 244 * patient.bmi
        ffm = 9270 * weight / denominator
        assert model.v1 == pytest.approx(25.2 * ffm / 56.1)
        # Return clearances to their measured anesthetized-output reference.
        model.update_hemodynamics(1, 0.82, 1)
        flow = (ffm / 56.1) ** 0.75
        assert model.cl1 == pytest.approx(45.9 / 60 * flow)
        assert model.cl2 == pytest.approx(142 / 60 * flow)
        assert model.cl3 == pytest.approx(5.81 / 60 * flow)
        model.state.c1 = 100 / model.v1
        concentrations.append(model.state.c1)
    assert concentrations[1] > concentrations[0] * 70 / 120


def test_toxic_lidocaine_does_not_keep_increasing_its_antiarrhythmic_benefit():
    therapeutic, _ = antiarrhythmic_effects(3, 0)
    toxic, _ = antiarrhythmic_effects(20, 0)
    assert therapeutic > 0.5 and toxic == 0


def test_sodium_block_widens_qrs_without_implying_ischemic_p_wave_loss():
    waveforms = []
    for block in (0, 0.8):
        clock, ecg = CardiacCycle(), ECGMonitor(np.random.default_rng(1))
        sample = clock.seed(70, RhythmType.SINUS)
        ecg.step(0.01, sample, sodium_block=block)
        qrs = []
        for _ in range(10):
            sample = clock.step(0.005, 70, RhythmType.SINUS)
            qrs.append(ecg.step(0.005, sample, sodium_block=block))
        waveforms.append(qrs)
    assert waveforms[1][-1] > waveforms[0][-1] + 0.1
    assert displayed_rhythm(RhythmType.SINUS, 0, 0.8) == "Wide-complex rhythm"


@pytest.mark.parametrize(("failure", "label"), [(0.8, "Idioventricular rhythm"), (0.98, "Agonal rhythm")])
def test_toxic_qrs_widening_does_not_hide_terminal_escape_rhythms(failure, label):
    assert displayed_rhythm(RhythmType.SINUS, failure, 0.8) == label


def _ventilated(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=7), start=True)
    engine.set_airway_mode("ETT")
    engine.set_fgf(10, 0)
    engine.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5, ie="1:2")
    engine.set_vent_power(True)
    return engine


def test_overdose_causes_seizures_despite_oxygenation_and_midazolam_treats_them(engine_factory):
    engine = _ventilated(engine_factory)
    engine.give_drug_bolus("lidocaine", 500)
    for _ in range(60):
        engine.step(1)
    assert engine.state.sao2 > 98 and engine.state.seizure_activity > 0.5
    assert engine.state.convulsions and engine.state.loc > 0.9
    engine.give_drug_bolus("midazolam", 5)
    for _ in range(90):
        engine.step(1)
    assert engine.state.seizure_activity < 0.2 and not engine.state.convulsions


def test_motor_block_hides_overdose_convulsions_without_treating_the_seizure(engine_factory):
    engine = _ventilated(engine_factory)
    engine.give_drug_bolus("lidocaine", 500)
    for _ in range(60):
        engine.step(1)
    assert engine.state.convulsions
    engine.give_drug_bolus("roc", 30)
    for _ in range(150):
        engine.step(1)
    assert engine.state.twitch < 10 and engine.state.seizure_activity > 0.5
    assert not engine.state.convulsions
    assert engine.toxicity.metabolic_factor > 1  # Cerebral activity persists.


def test_large_overdose_depresses_circulation_before_myocardial_energy_is_exhausted(engine_factory):
    engine = _ventilated(engine_factory)
    engine.give_drug_bolus("lidocaine", 2000)
    for _ in range(60):
        engine.step(1)
    assert engine.state.map < 10 and engine.state.intrinsic_co < 0.3
    assert engine.hemo.coronary.energy_reserve > 0.5
    assert engine.toxicity.conduction_block > 0.5


def test_cerebral_oxygen_failure_suppresses_toxic_convulsions_despite_retained_drug(engine_factory):
    engine = _ventilated(engine_factory)
    engine.give_drug_bolus("lidocaine", 500)
    for _ in range(60):
        engine.step(1)
    assert engine.state.convulsions
    engine.set_rhythm("VFIB")
    for _ in range(60):
        engine.step(1)
    assert engine.pk["lidocaine"].state.ce_neuro > 6
    assert engine.cerebral.cortical_function < 0.01
    assert engine.state.seizure_activity < 0.02 and not engine.state.convulsions


def test_lipid_rescue_gradually_reduces_cardiac_toxicity_during_supported_ventilation(engine_factory):
    outcomes = []
    for rescue in (False, True):
        engine = _ventilated(engine_factory)
        engine.give_drug_bolus("lidocaine", 1100)
        for i in range(240):
            if i == 30:
                engine.give_drug_bolus("midazolam", 5)
                if rescue:
                    engine.give_lipid_emulsion(100)
                    engine.set_lipid_infusion(17.5)
            engine.step(1)
            if rescue and i == 31:
                assert engine.toxicity.lipid_effect < 0.01
        outcomes.append((engine.state.map, engine.toxicity.conduction_block, engine.toxicity.contractility_loss))
    assert outcomes[1][0] > outcomes[0][0] + 10
    assert 0.3 < outcomes[1][1] < outcomes[0][1]  # Rescue leaves substantial toxicity.
    assert outcomes[1][2] < outcomes[0][2] - 0.15


def test_lidocaine_post_distribution_kinetics_and_subconvulsive_infusions(engine_factory, patient_factory):
    """Tucker and Boas 1971 (3 mg/kg over 3 or 19 min) and Rowland 1971 (110 mg bolus + 0.75 mg/min)."""
    patient = patient_factory(age=30, weight=69, height=175, sex="male")
    pk = LidocainePK(patient)
    # Foong's model relates HCl dose to base plasma concentration; Tucker's mean was 0.93 L/min.
    assert 0.85 < pk.cl1 < 1.05
    pk.step(60.0, 110.0 / 60.0)
    for minute in range(2, 241):
        pk.step(60.0, (0.75 / 60.0) if 31 <= minute <= 150 else 0.0)
        if minute == 150:
            plateau_hcl = pk.state.c1 / 0.8654  # Rowland reported concentrations as the HCl salt.
    assert 0.80 < plateau_hcl < 1.20 and pk.state.c1 / 0.8654 < 0.40

    for minutes, weight in ((3, 69), (19, 75)):
        engine = engine_factory(
            age=30, weight=weight, height=175, sex="male",
            config=SimulationConfig(mode="awake", rng_seed=1), start=True,
        )
        duration = minutes * 60
        engine.set_drug_rate("lidocaine", 3.0 * 60.0 / minutes)
        c1_10m = c1_30m = peak_c1 = 0.0
        convulsed = False
        for second in range(1, max(duration + 60, 1800) + 1):
            if second == duration + 1:
                engine.set_drug_rate("lidocaine", 0.0)
            engine.step(1)
            c1 = engine.pk["lidocaine"].state.c1
            peak_c1 = max(peak_c1, c1)
            convulsed = convulsed or engine.state.convulsions
            if second == 600:
                c1_10m = c1
            elif second == 1800:
                c1_30m = c1
        assert not convulsed
        if minutes == 3:
            assert 2.4 < c1_10m < 3.6 and 1.1 < c1_30m < 1.9
        else:
            assert 2.8 < peak_c1 < 4.0


def test_awake_convulsive_infusion_raises_rate_and_pressure_then_recovers(engine_factory):
    """Foldes 1960 (0.5 mg/kg/min for 12.8 min) and Jorfeldt 1968: HR and pressure rise before convulsions."""
    engine = engine_factory(
        age=28, weight=70, height=175, sex="male",
        config=SimulationConfig(mode="awake", rng_seed=1), start=True,
    )
    base_hr, base_sbp = engine.state.hr, engine.state.sbp
    engine.set_drug_rate("lidocaine", 30.0)
    pre_hr = pre_sbp = peak_hr = 0.0
    convulsed = False
    for second in range(1, 768 + 1140 + 1):
        if second == 769:
            engine.set_drug_rate("lidocaine", 0.0)
        engine.step(1)
        if not convulsed and not engine.state.convulsions:
            pre_hr = max(pre_hr, engine.state.hr)
            pre_sbp = max(pre_sbp, engine.state.sbp)
        peak_hr = max(peak_hr, engine.state.hr)
        convulsed = convulsed or engine.state.convulsions
    assert convulsed
    assert pre_hr > base_hr * 1.10 and pre_sbp > base_sbp * 1.10
    assert peak_hr > base_hr * 1.15
    assert not engine.state.convulsions and engine.state.loc < 0.05


def test_one_hour_overdose_in_hfref_stays_in_sinus_rhythm_and_recovers(engine_factory):
    """Armstrong 1988: 2 g over 1 h causes hypotension and coma in sinus rhythm.

    With linear kinetics and moderate HFrEF, the model clears lidocaine faster
    than in that case report (3.8 h total half-life at EF 15%).
    """
    engine = engine_factory(
        age=68, weight=74, height=172, sex="male", heart_failure="hfref",
        config=SimulationConfig(mode="awake", rng_seed=1), start=True,
    )
    engine.set_airway_mode("ETT")
    engine.set_fgf(2, 4)
    engine.set_vent_settings(mode="VCV", rr=12, vt=0.50, peep=5, ie="1:2")
    engine.set_vent_power(True)
    engine.set_drug_rate("lidocaine", 2000.0 / 74.0)
    map_60m = c1_75m = loc_75m = 0.0
    for minute in range(1, 241):
        if minute == 61:
            engine.set_drug_rate("lidocaine", 0.0)
        for _ in range(60):
            engine.step(1)
        rhythm = displayed_rhythm(
            engine.hemo.rhythm_type, engine.hemo.coronary.energy_failure, engine.toxicity.conduction_block
        )
        assert rhythm == "Sinus rhythm"
        assert not (minute > 75 and engine.state.convulsions)
        if minute == 60:
            map_60m = engine.state.map
        elif minute == 75:
            c1_75m = engine.pk["lidocaine"].state.c1
            loc_75m = engine.state.loc
            engine.give_drug_bolus("midazolam", 5)
        elif minute == 120:
            engine.give_drug_bolus("midazolam", 5)
    assert 45 < map_60m < 75
    assert 14.5 < c1_75m < 20.5 and loc_75m > 0.7
    # Recovery follows the model's faster washout, not Armstrong's 3.8 h total half-life.
    assert engine.pk["lidocaine"].state.c1 < 5.0
    assert engine.state.loc < 0.05 and engine.toxicity.respiratory_drive > 0.95


def test_volunteer_lipid_protocol_reduces_effective_lidocaine_auc(engine_factory):
    """Heinonen 2015: lipid reduces 30-min unentrapped lidocaine AUC by 23%.

    Both arms receive 1.5 mL/kg over 1 min before 1 mg/kg lidocaine,
    then 0.25 mL/kg/min of the same study fluid for 30 min.
    """
    bolus_ml = 1.5 * 75
    infusion_ml_min = 0.25 * 75
    aucs = []
    for use_lipid in (False, True):
        engine = engine_factory(
            age=25, weight=75, height=180, sex="male",
            config=SimulationConfig(mode="awake", rng_seed=1, maintenance_fluid_ml_hr=0), start=True,
        )
        # The study bolus takes 1 min; give_lipid_emulsion() takes 2.5 min.
        if use_lipid:
            engine.set_lipid_infusion(bolus_ml)
        else:
            engine.set_continuous_fluid_rate(bolus_ml * 60)
        for _ in range(60):
            engine.step(1)
        if use_lipid:
            assert engine.toxicity.total_lipid_ml == pytest.approx(bolus_ml)
            engine.set_lipid_infusion(infusion_ml_min)
        else:
            assert engine.hemo.total_crystalloid_in_ml == pytest.approx(bolus_ml)
            engine.set_continuous_fluid_rate(infusion_ml_min * 60)
        engine.give_drug_bolus("lidocaine", 75)
        auc = 0.0
        for _ in range(1800):
            engine.step(1)
            auc += engine.pk["lidocaine"].state.c1 * engine.toxicity.effect_input_fraction / 60.0
        total_ml = engine.toxicity.total_lipid_ml if use_lipid else engine.hemo.total_crystalloid_in_ml
        assert total_ml == pytest.approx(bolus_ml + infusion_ml_min * 30)
        aucs.append(auc)
    reduction = 1.0 - aucs[1] / aucs[0]
    assert 0.19 < reduction < 0.26


def test_lipid_is_delivered_over_time_accounted_as_fluid_and_can_be_stopped(engine_factory):
    engine = _ventilated(engine_factory)
    fluids = engine.hemo.fluids
    sodium, body_water = engine.state.sodium, fluids.ecf_l + fluids.icf_l
    engine.give_lipid_emulsion(100)
    engine.set_lipid_infusion(17.5)
    engine.step(1)
    assert 0 < engine.toxicity.total_lipid_ml < 2
    for _ in range(149):
        engine.step(1)
    assert engine.toxicity.total_lipid_ml == pytest.approx(143.75)
    assert engine.hemo.total_colloid_in_ml == pytest.approx(143.75)
    # Sodium-free lipid water spreads through total body water, not only extracellular fluid.
    assert sodium - engine.state.sodium == pytest.approx(sodium * 0.14375 / body_water, rel=0.1)
    assert engine.toxicity.lipid_effect > 0
    engine.set_lipid_infusion(0)
    given = engine.toxicity.total_lipid_ml
    for _ in range(10):
        engine.step(1)
    assert engine.toxicity.total_lipid_ml == given
    for invalid in (math.nan, math.inf, -1):
        with pytest.raises(ValueError):
            engine.set_lipid_infusion(invalid)
