"""Sampled patients: a repeatable seed and plausible responses to induction."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.patient.pd import ClinicalResponseModel
from anasim.patient.pd.anesthesia import BISModel
from anasim.patient.pk_models import NorepinephrinePK
from anasim.patient.variability import TYPICAL, sample_variability
from anasim.patient.volatile_pk import VolatilePK
from anasim.physiology.disturbances import Disturbances
from anasim.physiology.hemodynamics import HemodynamicModel
from anasim.web import WebSession

ADULT: dict[str, Any] = dict(age=45, weight=75, height=172, sex="male", baseline_map=95, baseline_hr=75)


def _bolus_induction(seed):
    """Seconds from propofol 2 mg/kg (after fentanyl 1.5 mcg/kg) to BIS < 60, and the MAP nadir (% fall) in 5 min."""
    engine = SimulationEngine(Patient(**ADULT), SimulationConfig(patient_variability=seed is not None, rng_seed=seed))
    engine.start()
    engine.set_airway_mode("Mask")
    engine.set_fgf(6.0, 0.0)
    for _ in range(5):
        engine.step(2.0)
    awake = engine.state.map
    engine.give_drug_bolus("fentanyl", 1.5 * ADULT["weight"])
    for _ in range(60):
        engine.step(2.0)
    engine.give_drug_bolus("propofol", 2.0 * ADULT["weight"])
    engine.give_drug_bolus("roc", 0.6 * ADULT["weight"])
    onset, lowest = None, awake
    for second in range(2, 211, 2):
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
    for drug in ("propofol", "remi", "fentanyl"):
        assert first.pk[drug].v1 == second.pk[drug].v1 != typical.pk[drug].v1
    assert first.hemo.config.ec50_prop_tpr != typical.hemo.config.ec50_prop_tpr
    assert first.hemo.config.sevo_ec50_tpr != typical.hemo.config.sevo_ec50_tpr
    assert first.patient.variability.opioid_rate != TYPICAL.opioid_rate
    assert WebSession({"patient_variability": True}).engine.patient.variability != TYPICAL


def test_tci_pumps_predict_with_the_typical_patient():
    """Pumps use population PK. After 10 min at an effect-site target of
    4 mcg/mL, sampled patients' propofol differs from the target and the typical
    patient's matches it."""
    concentrations = []
    for seed in (None, *range(1, 7)):
        engine = SimulationEngine(Patient(**ADULT), SimulationConfig(
            mode="awake", tci_enabled=True, patient_variability=seed is not None, rng_seed=seed))
        engine.start()
        engine.set_airway_mode("ETT")
        engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        engine.set_vent_power(True)
        engine.enable_tci("propofol", 4.0)
        for _ in range(300):
            engine.step(2.0)
        assert engine.prediction_model("propofol").state.ce == pytest.approx(4.0, rel=0.01)
        concentrations.append(engine.pk["propofol"].state.ce)
    typical, *sampled = concentrations
    assert typical == pytest.approx(4.0, rel=0.01)
    assert max(sampled) - min(sampled) > 0.5
    assert all(2.8 < ce < 5.2 for ce in sampled)


@pytest.mark.parametrize(("drug", "c25"), [("morphine", 7.7 / 0.518), ("hydromorphone", 7.7 * 0.518)])
def test_sampled_opioid_c25_reduces_engine_rate_by_one_quarter(drug, c25):
    engine = SimulationEngine(Patient(**ADULT), SimulationConfig(patient_variability=True, rng_seed=3))
    engine.start()
    engine.pk[drug].state.ce_rr = c25 * engine.patient.variability.opioid_rate.c25
    engine.step(0.1)
    assert engine.state.rr == pytest.approx(0.75 * engine.patient.baseline_rr, abs=0.02)


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


def _incision_rise(seed, sevo):
    """Largest HR or MAP rise within 5 min of incision at steady end-tidal sevoflurane alone."""
    patient = Patient(age=42, weight=57, height=160, sex="female")
    patient.variability = sample_variability(seed)
    pk_sevo = VolatilePK(patient, "Sevoflurane", lambda_b_g=0.65, mac_40=1.80)
    mac = sevo / pk_sevo.mac_age
    nore = NorepinephrinePK(patient)
    nore.equilibrate_endogenous()
    hemo = HemodynamicModel(patient)
    hemo.initialize_steady_state(0.0, 0.0, nore.state.ce, mac_sevo=mac, pit=-1.4, peep_cmH2O=5.0)
    response = ClinicalResponseModel(patient.age, patient.variability.sevoflurane_mac)
    disturbance = Disturbances("stim_sustained_surgery")
    assert disturbance.spec is not None
    p_frac, hr_frac = response.cardiovascular_response_fractions(
        0.0, 0.0, mac_sevo=mac, intensity=disturbance.spec.sympathetic_intensity,
    )
    hr, pressure = hemo.state.hr, hemo.state.map
    rise = 0.0
    for step_idx in range(150):
        t_rel = step_idx * 2.0
        stim = disturbance.compute_average(t_rel, t_rel + 2.0)
        svr_dist = stim.svr * p_frac + disturbance.spec.tonic_svr(t_rel + 1.0, 1000.0 * hemo.tpr)
        state = hemo.step(
            2.0, 0.0, 0.0, nore.state.ce, -1.4, 40.0, 450.0, mac_sevo=mac, peep_cmH2O=5.0,
            dist_svr=svr_dist, dist_sv=stim.sv * p_frac, dist_hr=stim.hr * hr_frac,
        )
        rise = max(rise, state.hr / hr - 1.0, state.map / pressure - 1.0)
    return rise


@pytest.mark.parametrize(("sevo", "fewest", "most"), [(4.15, 6, 14), (6.26, 0, 3)])
def test_sampled_patients_split_at_sevoflurane_mac_bar(sevo, fewest, most):
    """Katoh 1999: in patients aged 20-50, end-tidal sevoflurane blocked a >15%
    HR or MAP rise within 5 minutes of incision in half at 4.15% (MAC-BAR) and
    in 95% at 6.26%. Bounds are the 95% binomial range for 20 patients."""
    responders = sum(_incision_rise(seed, sevo) > 0.15 for seed in range(1, 21))
    assert fewest <= responders <= most


def test_sevoflurane_bis_preserves_interindividual_sensitivity_spread():
    """Individual hypnotic sensitivity shifts BIS during volatile-only exposure."""
    base = sample_variability(1)
    sensitive, resistant = Patient(**ADULT), Patient(**ADULT)
    sensitive.variability = replace(base, propofol=replace(base.propofol, ce50=0.7))
    resistant.variability = replace(base, propofol=replace(base.propofol, ce50=1.4))
    assert BISModel(resistant).compute_bis(0.0, 1.0, 0.0) > BISModel(sensitive).compute_bis(0.0, 1.0, 0.0) + 8.0

    var_eng = SimulationEngine(
        Patient(**ADULT), SimulationConfig(mode="awake", patient_variability=True, rng_seed=7)
    )
    assert var_eng.response.sevoflurane_mac == var_eng.patient.variability.sevoflurane_mac
