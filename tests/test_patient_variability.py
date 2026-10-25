"""Sampled patients: a repeatable seed and plausible responses to induction."""

from typing import Any

import numpy as np

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.patient.variability import TYPICAL
from anasim.web import WebSession

ADULT: dict[str, Any] = dict(age=45, weight=75, height=172, sex="male", baseline_map=95, baseline_hr=75)


def _bolus_induction(seed):
    """Seconds from propofol 2 mg/kg (after fentanyl 1.5 mcg/kg) to BIS < 60, and the MAP nadir (% fall) in 5 min."""
    engine = SimulationEngine(Patient(**ADULT), SimulationConfig(patient_variability=seed is not None, rng_seed=seed))
    engine.start()
    engine.set_airway_mode("Mask")
    engine.set_fgf(6.0, 0.0)
    for _ in range(15):
        engine.step(2.0)
    awake = engine.state.map
    engine.give_drug_bolus("fentanyl", 1.5 * ADULT["weight"])
    for _ in range(60):
        engine.step(2.0)
    engine.give_drug_bolus("propofol", 2.0 * ADULT["weight"])
    engine.give_drug_bolus("roc", 0.6 * ADULT["weight"])
    onset, lowest = None, awake
    for second in range(2, 301, 2):
        if second == 60:
            engine.set_airway_mode("ETT")
            engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
            engine.set_vent_power(True)
        engine.step(2.0)
        if onset is None and engine.state.bis < 60.0:
            onset = second
        lowest = min(lowest, engine.state.map)
    return onset, 100.0 * (1.0 - lowest / awake)


def test_a_seed_fixes_the_patient_and_the_typical_patient_is_the_default():
    patient = Patient(**ADULT)
    typical = SimulationEngine(patient, SimulationConfig(rng_seed=3))
    first, second = (SimulationEngine(patient, SimulationConfig(patient_variability=True, rng_seed=3)) for _ in range(2))
    assert typical.patient is patient and patient.variability == TYPICAL
    assert first.patient.variability == second.patient.variability != TYPICAL
    assert first.pk["propofol"].v1 == second.pk["propofol"].v1 != typical.pk["propofol"].v1
    assert first.hemo.config.ec50_prop_tpr != typical.hemo.config.ec50_prop_tpr
    assert WebSession({"patient_variability": True}).engine.patient.variability != TYPICAL


def test_sampled_bolus_inductions_vary_within_clinical_limits():
    """Propofol 2 mg/kg after fentanyl in 20 sampled adults. A few slow responders
    need more propofol. Pressure falls stay within routine practice (VitalDB TIVA
    lowest reading in 10 min, 1st-99th percentile 6-53%), and the typical patient
    sits mid-population."""
    typical_onset, typical_fall = _bolus_induction(None)
    results = [_bolus_induction(seed) for seed in range(1, 21)]
    onsets = np.array([onset for onset, _ in results if onset is not None and onset <= 180])
    falls = np.array([fall for _, fall in results])
    assert len(onsets) >= 16
    assert np.percentile(onsets, 75) - np.percentile(onsets, 25) >= 15.0
    assert np.percentile(onsets, 25) <= typical_onset <= np.percentile(onsets, 75)
    assert 15.0 <= falls.min() and falls.max() <= 50.0
    assert np.percentile(falls, 25) <= typical_fall <= np.percentile(falls, 75)
