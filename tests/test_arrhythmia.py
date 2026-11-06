import math

import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.patient.pk_models import AmiodaronePK, LidocainePK
from anasim.physiology.arrhythmia import ArrhythmiaModel
from anasim.physiology.resuscitation import ShockResult, shock_outcome


def _step(model, dt, rhythm, **changes):
    inputs = dict(ischemia=0.0, hypoxia=0.0, energy_failure=0.0,
                  electrically_silent=False, perfusing=True, lidocaine=0.0, amiodarone=0.0)
    inputs.update(changes)
    return model.step(dt, rhythm, **inputs)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_severe_ischemia_can_initiate_a_ventricular_rhythm_independent_of_step_size(dt):
    model = ArrhythmiaModel(np.random.default_rng(1))
    time, rhythm = 0.0, RhythmType.SINUS
    while rhythm == RhythmType.SINUS and time < 1000:
        rhythm = _step(model, dt, rhythm, ischemia=1)
        time += dt
    assert rhythm in (RhythmType.VTACH, RhythmType.VFIB)
    assert 643 <= time <= 644


def test_ischemic_vt_can_degenerate_to_vf_but_silent_myocardium_does_not():
    for silent in (False, True):
        model = ArrhythmiaModel(np.random.default_rng(1))
        rhythm = RhythmType.VTACH
        for _ in range(100):
            rhythm = _step(model, 1, rhythm, ischemia=1, electrically_silent=silent,
                           energy_failure=1 if silent else 0)
        assert rhythm == (RhythmType.VTACH if silent else RhythmType.VFIB)


@pytest.mark.parametrize("drug", ["lidocaine", "amiodarone"])
def test_delivered_antiarrhythmics_can_convert_perfusing_vt(drug):
    model = ArrhythmiaModel(np.random.default_rng(1))
    rhythm = RhythmType.VTACH
    for _ in range(1200):
        rhythm = _step(model, 1, rhythm, **{drug: 3.0})
    assert rhythm == RhythmType.SINUS


@pytest.mark.parametrize("rhythm", [RhythmType.VFIB, RhythmType.ASYSTOLE, RhythmType.VTACH])
def test_drugs_do_not_replace_a_shock_for_pulseless_ventricular_rhythms(rhythm):
    model = ArrhythmiaModel(np.random.default_rng(1))
    for _ in range(600):
        assert _step(model, 1, rhythm, perfusing=False, lidocaine=30, amiodarone=30) == rhythm


def test_successful_shock_can_leave_transient_bradycardia_and_recurrent_vf():
    model = ArrhythmiaModel(np.random.default_rng(1))
    rhythm = model.after_shock(RhythmType.VFIB, RhythmType.SINUS,
                              ischemia=1, hypoxia=0, energy_failure=0.3)
    assert rhythm == RhythmType.SINUS_BRADY
    for _ in range(120):
        rhythm = _step(model, 1, rhythm, ischemia=1, energy_failure=0.3)
        if rhythm == RhythmType.VFIB:
            break
    assert rhythm == RhythmType.VFIB


def test_post_shock_bradycardia_recovers_but_manually_selected_bradycardia_persists():
    model = ArrhythmiaModel(np.random.default_rng(1))
    rhythm = model.after_shock(RhythmType.VFIB, RhythmType.SINUS,
                              ischemia=1, hypoxia=0, energy_failure=0)
    for _ in range(40):
        rhythm = _step(model, 1, rhythm)
    assert rhythm == RhythmType.SINUS
    model.reset()
    for _ in range(60):
        assert _step(model, 1, RhythmType.SINUS_BRADY) == RhythmType.SINUS_BRADY


def test_treating_intervening_vt_does_not_erase_recent_vf_substrate():
    model = ArrhythmiaModel(np.random.default_rng(2))
    rhythm = model.after_shock(RhythmType.VFIB, RhythmType.SINUS,
                              ischemia=1, hypoxia=0, energy_failure=0)
    for _ in range(10):
        rhythm = _step(model, 1, rhythm, ischemia=0.5)
    rhythm = model.after_shock(RhythmType.VTACH, RhythmType.SINUS,
                              ischemia=0.5, hypoxia=0, energy_failure=0)
    for _ in range(120):
        rhythm = _step(model, 1, rhythm, ischemia=0.5)
        if rhythm == RhythmType.VFIB:
            break
    assert rhythm == RhythmType.VFIB


def test_antiarrhythmic_does_not_guarantee_freedom_from_rearrest():
    times = []
    for concentration in (0, 10):
        model = ArrhythmiaModel(np.random.default_rng(2))
        rhythm = model.after_shock(RhythmType.VFIB, RhythmType.SINUS,
                                  ischemia=1, hypoxia=0, energy_failure=0)
        for second in range(120):
            rhythm = _step(model, 1, rhythm, ischemia=0.5, amiodarone=concentration)
            if rhythm == RhythmType.VFIB:
                times.append(second)
                break
    # ALPS did not establish a reduction in rearrest with either drug.
    assert len(times) == 2 and times[0] == times[1]


def test_adding_vt_conversion_cannot_advance_the_degeneration_clock():
    times = []
    for concentration in (0, 10):
        model = ArrhythmiaModel(np.random.default_rng(2))
        rhythm = RhythmType.VTACH
        for second in range(100):
            rhythm = _step(model, 1, rhythm, ischemia=1, amiodarone=concentration)
            if rhythm != RhythmType.VTACH:
                times.append((second, rhythm))
                break
    assert len(times) == 2 and times[0] == times[1]


def test_antiarrhythmic_facilitates_defibrillation_but_does_not_make_it_certain():
    outcomes = []
    for effect in (0, 1):
        outcomes.append(shock_outcome(
            RhythmType.VFIB, synchronized=False, energy_failure=0,
            electrically_silent=False, vulnerable=False, antiarrhythmic=effect,
            rng=np.random.default_rng(4),
        ))
    assert outcomes == [RhythmType.VFIB, RhythmType.SINUS]
    assert shock_outcome(
        RhythmType.VFIB, synchronized=False, energy_failure=1,
        electrically_silent=False, vulnerable=False, antiarrhythmic=1,
        rng=np.random.default_rng(4),
    ) == RhythmType.VFIB


@pytest.mark.parametrize(("model_type", "site"), [(AmiodaronePK, "ce"), (LidocainePK, "ce_cv")])
def test_cardiac_drug_exposure_waits_for_flow_then_increases_with_cpr(patient, model_type, site):
    model = model_type(patient)
    model.state.c1 = 100 / model.v1
    initial = model.state.c1
    model.circulation = 0
    for _ in range(120):
        model.step(1, 0)
    assert model.state.c1 == initial and getattr(model.state, site) == 0
    model.circulation = 2 / 3
    for _ in range(120):
        model.step(1, 0)
    assert getattr(model.state, site) > 0.5 and model.state.c1 < initial


def test_acute_amiodarone_has_cardiovascular_effects_after_delivery(engine_factory):
    endpoints = []
    for dose in (0, 150):
        engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
        if dose:
            engine.give_drug_bolus("amiodarone", dose)
        for _ in range(120):
            engine.step(1)
        endpoints.append((engine.state.hr, engine.state.map))
    assert endpoints[1][0] < endpoints[0][0] - 2
    assert endpoints[1][1] < endpoints[0][1] - 5


def test_engine_projects_automatic_rhythm_changes_to_ecg_and_circulation(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    engine.hemo.coronary.ischemia = 1
    engine.arrhythmias._rhythm = RhythmType.SINUS
    engine.arrhythmias._remaining["onset"] = 0
    engine.step(0.01)
    assert engine.hemo.rhythm_type in (RhythmType.VFIB, RhythmType.VTACH)
    assert engine.cardiac_cycle.current.rhythm_type == engine.hemo.rhythm_type


def test_antiarrhythmic_bolus_in_vf_needs_circulation_before_it_can_help_a_shock(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    engine.set_rhythm("VFIB")
    engine.give_drug_bolus("amiodarone", 300)
    engine.charge_defibrillator()
    for _ in range(10):
        engine.step(1)
    assert engine.state.amiodarone_ce == 0
    assert engine.hemo.rhythm_type == RhythmType.VFIB
    engine.set_chest_compressions(True)
    for _ in range(60):
        engine.step(1)
    assert engine.state.amiodarone_ce > 1
    engine.charge_defibrillator()
    for _ in range(5):
        engine.step(1)
    assert engine.defibrillate() == ShockResult.DELIVERED
    assert math.isfinite(engine.state.amiodarone_ce)
