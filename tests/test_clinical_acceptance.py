"""Standard cases an attending would recognize in healthy adults.

Ranges sit between published inductions, which often use higher doses,
VitalDB recordings of routine practice, and clinical experience. TIVA ranges
are the interquartile ranges of recorded cuff MAP change in about 400 VitalDB
propofol-remifentanil inductions.
"""

from typing import Any

import numpy as np
import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient

ADULT: dict[str, Any] = dict(age=45, weight=75, height=172, sex="male", baseline_map=95, baseline_hr=75)
OLDER: dict[str, Any] = dict(age=70, weight=72, height=168, sex="male", baseline_map=105, baseline_hr=70)
YOUNG: dict[str, Any] = dict(age=28, weight=62, height=165, sex="female", baseline_map=85, baseline_hr=72)


def _ventilate(engine):
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.45, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(1.0, 1.0)


def _run(patient, events, seconds):
    """Awake MAP and HR, then MAP, HR, and SBP after each second."""
    engine = SimulationEngine(Patient(**patient), SimulationConfig(mode="awake", tci_enabled=True, rng_seed=1))
    engine.start()
    engine.set_airway_mode("Mask")
    engine.set_fgf(6.0, 0.0)
    for _ in range(60):
        engine.step(1.0)
    awake = engine.state.map, engine.state.hr
    pending, trace = sorted(events, key=lambda event: event[0]), []
    for second in range(seconds):
        while pending and pending[0][0] <= second:
            pending.pop(0)[1](engine)
        engine.step(1.0)
        trace.append((engine.state.map, engine.state.hr, engine.state.sbp))
    return awake, np.array(trace)


def _balanced(patient, propofol, fentanyl, vaporizer):
    """Fentanyl, then propofol 2 min later; sevoflurane from 1.5 min, intubation
    at 3 min, and incision at 15 min after propofol."""
    weight = patient["weight"]
    return _run(patient, [
        (0, lambda e: e.give_drug_bolus("fentanyl", fentanyl * weight)),
        (120, lambda e: e.give_drug_bolus("propofol", propofol * weight)),
        (130, lambda e: e.give_drug_bolus("roc", 0.6 * weight)),
        (210, lambda e: (_ventilate(e), e.set_vaporizer(vaporizer))),
        (300, lambda e: e.start_disturbance("stim_intubation_pulse")),
        (1020, lambda e: e.start_disturbance("stim_sustained_surgery")),
    ], 1320)


def _fall(awake, value):
    return 100.0 * (1.0 - value / awake)


def test_propofol_fentanyl_sevoflurane_induction_intubation_and_incision():
    """The typical VitalDB bolus induction with sevoflurane (83 recorded cases):
    age 55, 63 kg, fentanyl 1.4 mcg/kg, then propofol 1.8 mg/kg, end-tidal
    sevoflurane about 1.6% before incision and 1.9% during surgery, incision at
    24 min. Ranges are the recorded interquartile ranges: MAP -19% at 1-2 min,
    -12% at 2-5 min, lowest reading -28% in the first 10 min, -27% before
    incision, -19% during surgery; HR 65 before incision and 70 during surgery.
    """
    patient: dict[str, Any] = dict(age=55, weight=63, height=163, sex="female", baseline_map=98, baseline_hr=74)
    weight = patient["weight"]
    awake, trace = _run(patient, [
        (0, lambda e: e.give_drug_bolus("fentanyl", 1.4 * weight)),
        (120, lambda e: e.give_drug_bolus("propofol", 1.8 * weight)),
        (150, lambda e: e.give_drug_bolus("roc", 0.6 * weight)),
        (220, lambda e: (_ventilate(e), e.set_vaporizer(2.5))),
        (330, lambda e: e.start_disturbance("stim_intubation_pulse")),
        (1560, lambda e: (e.start_disturbance("stim_sustained_surgery"), e.set_vaporizer(3.0))),
    ], 3120)
    falls = 100.0 * (1.0 - trace[:, 0] / awake[0])
    assert 9.0 <= falls[180:240].mean() <= 25.0
    assert -3.0 <= falls[240:420].mean() <= 26.0
    assert 20.0 <= max(falls[k - 40:k].mean() for k in range(180, 720, 150)) <= 35.0
    assert 19.0 <= falls[960:1560].mean() <= 32.0
    assert 58.0 <= trace[960:1560, 1].mean() <= 73.0
    pre_intubation = trace[329]
    assert 15.0 <= trace[330:450, 0].max() - pre_intubation[0] <= 30.0
    assert 10.0 <= trace[330:450, 1].max() - pre_intubation[1] <= 20.0
    assert 10.0 <= trace[1560:1860, 0].max() - trace[1559, 0] <= 20.0
    assert 9.0 <= falls[2460:3120].mean() <= 26.0
    assert 63.0 <= trace[2460:3120, 1].mean() <= 81.0


def test_older_patient_falls_further():
    adult, adult_trace = _balanced(ADULT, 2.0, 1.5, 2.0)
    older, older_trace = _balanced(OLDER, 1.5, 1.0, 1.5)
    adult_fall = _fall(adult[0], adult_trace[180:300, 0].min())
    older_fall = _fall(older[0], older_trace[180:300, 0].min())
    assert 25.0 <= older_fall <= 45.0
    assert older_fall >= adult_fall + 3.0


def test_tiva_induction_intubation_and_maintenance():
    """Remifentanil effect-site 3 ng/mL, then propofol effect-site 4 mcg/mL;
    intubation 3 min and incision 15 min after propofol starts. VitalDB medians:
    -16% at 1-2 min, -21% at 2-5 min, lowest reading -31% in the first 10 min,
    -27% over the 10 min before incision. Remifentanil 3 ng/mL leaves most of
    the intubation response (Albertin 2005: 5 ng/mL blocked a 15% rise in half
    of patients at propofol 3.4 mcg/mL)."""
    awake, trace = _run(ADULT, [
        (0, lambda e: e.enable_tci("remi", 3.0)),
        (60, lambda e: e.enable_tci("propofol", 4.0)),
        (130, lambda e: e.give_drug_bolus("roc", 0.6 * ADULT["weight"])),
        (150, _ventilate),
        (240, lambda e: e.start_disturbance("stim_intubation_pulse")),
        (960, lambda e: e.start_disturbance("stim_sustained_surgery")),
    ], 1260)
    assert 8.0 <= _fall(awake[0], trace[120:180, 0].mean()) <= 24.0
    assert 13.0 <= _fall(awake[0], trace[180:360, 0].mean()) <= 29.0
    assert 24.0 <= _fall(awake[0], trace[60:660, 0].min()) <= 38.0
    before_incision = trace[360:960].mean(axis=0)
    assert 19.0 <= _fall(awake[0], before_incision[0]) <= 34.0
    assert 55.0 <= before_incision[1] <= 70.0
    assert 10.0 <= trace[240:360, 0].max() - trace[239, 0] <= 25.0
    assert 5.0 <= trace[240:360, 1].max() - trace[239, 1] <= 25.0


def test_propofol_alone_in_a_young_adult():
    awake, trace = _run(YOUNG, [
        (0, lambda e: e.give_drug_bolus("propofol", 2.5 * YOUNG["weight"])),
        (60, _ventilate),
    ], 300)
    assert 15.0 <= _fall(awake[0], trace[120:300, 0].min()) <= 30.0
    assert -3.0 <= trace[120:300, 1].mean() - awake[1] <= 10.0


@pytest.mark.parametrize("drug,amount,map_rise,hr_change", [
    ("phenyl", 100.0, (15.0, 25.0), (-15.0, -5.0)),
    # Meng 2011: 0.1 mg/kg under TIVA raised MAP 24-28 mmHg and HR 2-8 bpm.
    ("ephedrine", 0.1, (15.0, 30.0), (0.0, 10.0)),
])
def test_vasopressor_boluses_under_tiva(engine_factory, drug, amount, map_rise, hr_change):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=3), start=True)
    base_map, base_hr = engine.state.map, engine.state.hr
    engine.give_drug_bolus(drug, amount * engine.patient.weight if drug == "ephedrine" else amount)
    rows = []
    for _ in range(300):
        engine.step(1.0)
        rows.append((engine.state.map, engine.state.hr))
    trace = np.array(rows)
    peak = int(trace[:, 0].argmax())
    assert map_rise[0] <= trace[peak, 0] - base_map <= map_rise[1]
    assert hr_change[0] <= trace[peak, 1] - base_hr <= hr_change[1]


def test_epinephrine_bolus_under_tiva(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=3), start=True)
    base_sbp, base_hr = engine.state.sbp, engine.state.hr
    engine.give_drug_bolus("epi", 10.0)
    sbp, hr = [], []
    for _ in range(300):
        engine.step(1.0)
        sbp.append(engine.state.sbp)
        hr.append(engine.state.hr)
    assert 20.0 <= max(hr) - base_hr <= 40.0
    assert 30.0 <= max(sbp) - base_sbp <= 50.0
