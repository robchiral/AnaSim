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
    assert tox.seizure_activity == tox.conduction_block == tox.contractility_loss == 0
    _step(tox, neural_ce=10, cardiac_ce=10)
    assert tox.seizure_activity > 0.95 and tox.unresponsiveness > 0.95
    assert tox.conduction_block == 0
    _step(tox, neural_ce=30, cardiac_ce=30)
    assert tox.neural_depression == tox.conduction_block == tox.contractility_loss == 1
    assert tox.respiratory_drive == 0


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
        engine.give_drug_bolus("lidocaine", 700)
        for i in range(240):
            if i == 30:
                engine.give_drug_bolus("midazolam", 5)
                if rescue:
                    engine.give_lipid_emulsion(100)
                    engine.set_lipid_infusion(17.5)
            engine.step(1)
            if rescue and i == 31:
                assert engine.toxicity.lipid_effect < 0.01
        outcomes.append((engine.state.map, engine.toxicity.conduction_block))
    assert outcomes[1][0] > outcomes[0][0] + 10
    assert 0.3 < outcomes[1][1] < outcomes[0][1]  # Rescue leaves substantial toxicity.


def test_lipid_is_delivered_over_time_accounted_as_fluid_and_can_be_stopped(engine_factory):
    engine = _ventilated(engine_factory)
    engine.give_lipid_emulsion(100)
    engine.set_lipid_infusion(17.5)
    engine.step(1)
    assert 0 < engine.toxicity.total_lipid_ml < 2
    for _ in range(149):
        engine.step(1)
    assert engine.toxicity.total_lipid_ml == pytest.approx(143.75)
    assert engine.hemo.total_colloid_in_ml == pytest.approx(143.75)
    assert engine.toxicity.lipid_effect > 0
    engine.set_lipid_infusion(0)
    given = engine.toxicity.total_lipid_ml
    for _ in range(10):
        engine.step(1)
    assert engine.toxicity.total_lipid_ml == given
    for invalid in (math.nan, math.inf, -1):
        with pytest.raises(ValueError):
            engine.set_lipid_infusion(invalid)
