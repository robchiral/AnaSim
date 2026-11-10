import pytest

from anasim.core.clinical_actions import check_intubation
from anasim.physiology.cerebral import CerebralModel


def _run(engine, seconds):
    for _ in range(round(seconds * 10)):
        engine.step(0.1)


@pytest.mark.parametrize(("map_mmhg", "saturation"), [(0, 98), (90, 10)])
def test_severe_supply_failure_suppresses_responsiveness_bis_and_breathing_until_it_returns(
        awake_engine, monkeypatch, map_mmhg, saturation):
    brain = awake_engine.cerebral
    # Moderate hypotension and desaturation leave cerebral function intact.
    brain.step(600, 60, 13.5, 90, 40, 7.4)
    assert brain.dysfunction == 0
    step = CerebralModel.step
    monkeypatch.setattr(brain, "step", lambda dt, *_: step(brain, dt, map_mmhg, brain.baseline_hb, saturation, 40, 7.4))
    # Function starts to fail after a few seconds; BIS shows it after the monitor's processing delay.
    _run(awake_engine, 4)
    assert brain.dysfunction < 0.1
    _run(awake_engine, 8)
    assert brain.dysfunction > 0.85 and awake_engine.state.bis > 80
    _run(awake_engine, 28)
    assert awake_engine.state.bis < 5 and awake_engine.state.loc > 0.95
    assert awake_engine.resp.state.effort_rr < 5
    assert check_intubation(awake_engine) is None
    monkeypatch.undo()
    _run(awake_engine, 180)
    assert awake_engine.state.loc < 0.01 and awake_engine.state.bis > 80
    feedback = check_intubation(awake_engine)
    assert feedback is not None and feedback.blocked


def test_persistent_coma_does_not_force_flat_bis_after_oxygen_and_breathing_return(awake_engine):
    brain = awake_engine.cerebral
    brain.step(CerebralModel.COMA_EXPOSURE_S, 0, 13.5, 98, 40, 7.4)
    brain.step(600, 90, 13.5, 98, 40, 7.4)
    _run(awake_engine, 30)
    assert awake_engine.state.bis > 80 and awake_engine.state.loc > 0.95
    assert awake_engine.resp.state.effort_rr > 5


@pytest.mark.parametrize(("pa_co2", "ph", "low", "high"), [
    (40, 7.4, 0, 0.01), (40, 7.0, 0, 0.01),
    (60, 7.3, 0, 0.01), (80, 7.25, 0.1, 0.2),
    (100, 7.2, 0.45, 0.55), (100, 7.1, 0.99, 1.0),
    (100, 7.38, 0, 0.01), (140, 7.38, 0.99, 1.0),
])
def test_co2_response_depends_on_acidemia_and_retention_without_anoxic_injury(pa_co2, ph, low, high):
    brain = CerebralModel(13.5)
    brain.step(300, 90, 13.5, 98, pa_co2, ph)
    assert low <= brain.unresponsiveness <= high
    assert brain.cortical_function == 1 and brain.anoxic_s == 0


def test_co2_narcosis_is_delayed_reversible_and_separate_from_bis(awake_engine, monkeypatch):
    brain = awake_engine.cerebral
    step = CerebralModel.step
    monkeypatch.setattr(brain, "step", lambda dt, *_: step(brain, dt, 90, 13.5, 98, 110, 7.1))
    _run(awake_engine, 1)
    assert 0 < brain.co2_narcosis < 0.1
    _run(awake_engine, 179)
    assert awake_engine.state.loc > 0.99
    assert awake_engine.state.bis > 80
    assert brain.anoxic_s == 0
    monkeypatch.undo()
    _run(awake_engine, 180)
    assert brain.co2_narcosis < 0.01
    assert awake_engine.state.loc < 0.01


def test_oxygenated_hypoventilation_impairs_response_and_ventilation_reverses_it(awake_engine):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.set_fgf(10, 0)
    engine.give_drug_bolus("roc", 70)
    engine.set_drug_rate("roc", 8)
    engine.set_vent_settings(mode="VCV", rr=6, vt=0.2, peep=5, ie="1:2")
    engine.set_vent_power(True)
    for _ in range(1200):
        engine.step(1)
    state = engine.state
    assert state.pa_co2 > 90 and state.ph < 7.15
    assert state.sao2 > 98 and state.map > 65 and state.twitch < 1
    assert state.loc > 0.6 and state.bis > 80
    assert engine.cerebral.anoxic_s == 0
    engine.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5, ie="1:2")
    for _ in range(1200):
        engine.step(1)
    assert state.pa_co2 < 50 and state.loc < 0.01
