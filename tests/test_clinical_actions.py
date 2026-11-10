"""Learner checks must give feedback without hiding physiological consequences."""

import json

import pytest

from anasim.core.action_log import ACTION_AIRWAY, ACTION_INTUBATION_ATTEMPT
from anasim.core.projection import loss_of_response
from anasim.core.runtime import stimulus_effects
from anasim.web import WebSession


def command(session, name, **args):
    return json.loads(session.command(name, json.dumps(args)))


def advance(engine, seconds, dt=0.5):
    for _ in range(round(seconds / dt)):
        engine.step(dt)


@pytest.mark.parametrize("dt", [0.01, 0.1, 0.5])
def test_awake_intubation_fails_with_feedback_and_a_pressor_response(dt):
    session = WebSession({"mode": "awake"})
    engine = session.engine
    engine.start()
    command(session, "airway", mode="Mask")
    command(session, "bag_mask", active=True)
    before = engine.state.hr, engine.state.map
    result = command(session, "airway", mode="ETT")
    feedback = result["feedback"]
    assert feedback["level"] == "blocked" and feedback["code"] == "awake_intubation"
    assert "tube not placed" in feedback["message"]
    assert engine.state.airway_mode.value == "Mask" and engine.bag_mask_active
    assert engine.actions.records[-1].action == ACTION_INTUBATION_ATTEMPT
    assert engine.actions.records[-1].label == "failed"
    assert not any(r.action == ACTION_AIRWAY and r.label == "ETT" for r in engine.actions.records)
    advance(engine, 30, dt)
    assert engine.state.hr > before[0] + 20 and engine.state.map > before[1] + 20
    # An aborted attempt stops at gagging/resistance, before laryngeal passage.
    assert engine.state.laryngospasm == 0
    assert session.snapshot()["controls"]["clinical_feedback"] == feedback
    command(session, "dismiss_feedback")
    assert session.snapshot()["controls"]["clinical_feedback"] is None


def test_failed_attempt_does_not_complete_guided_intubation():
    session = WebSession({"scenario_id": "induction_balanced"})
    assert session.scenario is not None
    session.step_index = next(i for i, step in enumerate(session.scenario) if step.id == "INTUBATE")
    session._begin_step()
    command(session, "airway", mode="ETT")
    assert not session.snapshot()["scenario"]["met"]
    command(session, "scenario_next")
    assert session.snapshot()["scenario"]["id"] == "INTUBATE"


def test_a_corrective_action_clears_stale_failed_placement_feedback():
    session = WebSession({})
    command(session, "airway", mode="ETT")
    assert session.snapshot()["controls"]["clinical_feedback"]["level"] == "blocked"
    result = command(session, "airway", mode="Mask")
    assert result == {"feedback": None}
    assert session.snapshot()["controls"]["clinical_feedback"] is None
    assert session.engine.intubation_start_time is not None


def test_anesthetized_intubation_needs_no_muscle_block_and_starts_only_once():
    session = WebSession({"mode": "awake", "tci_enabled": True})
    engine = session.engine
    engine.start()
    command(session, "airway", mode="Mask")
    command(session, "bag_mask", active=True)
    command(session, "drug_target", key="propofol", target=4)
    command(session, "drug_target", key="remi", target=4)
    advance(engine, 300)
    assert engine.state.tof == engine.state.twitch == 100
    assert engine.state.loc > 0.9 and engine.state.tol > 0.9
    assert command(session, "airway", mode="ETT") is None
    assert engine.state.airway_mode.value == "ETT" and not engine.bag_mask_active
    started = engine.intubation_start_time
    advance(engine, 10)
    command(session, "airway", mode="ETT")
    assert engine.intubation_start_time == started
    assert len([r for r in engine.actions.records if r.action == ACTION_INTUBATION_ATTEMPT]) == 1


def test_awake_paralysis_does_not_supply_hypnosis_or_block_the_pressor_response():
    session = WebSession({"mode": "awake"})
    engine = session.engine
    engine.start()
    command(session, "airway", mode="Mask")
    command(session, "bag_mask", active=True)
    result = command(session, "drug_bolus", key="ROC", amount=56)
    assert result["feedback"]["code"] == "awake_neuromuscular_block"
    assert result["medication"]["key"] == "roc"
    advance(engine, 120)
    assert engine.state.tof < 1 and engine.state.loc == 0
    result = command(session, "airway", mode="ETT")
    assert result["feedback"]["code"] == "intubation_awareness"
    assert engine.state.airway_mode.value == "ETT"
    before = engine.state.hr
    command(session, "vent_power", on=True)
    advance(engine, 30)
    assert engine.state.hr > before + 20
    assert engine.state.loc == 0


@pytest.mark.parametrize("name,args", [
    ("drug_rate", {"key": "roc", "rate": 10}),
    ("drug_target", {"key": "roc", "target": 3}),
])
def test_awake_block_warning_covers_infusions_and_tci(name, args):
    session = WebSession({"tci_enabled": True})
    result = command(session, name, **args)
    assert result["feedback"]["code"] == "awake_neuromuscular_block"
    assert result["medication"]["key"] == "roc"
    assert command(session, name, **{**args, "rate" if name == "drug_rate" else "target": 0}).get("feedback") is None


@pytest.mark.parametrize("hypnotic,amount", [
    ("propofol", 150), ("etomidate", 20), ("ketamine", 70), ("midazolam", 15),
])
def test_induction_bolus_before_block_does_not_trigger_a_premature_awareness_warning(hypnotic, amount):
    session = WebSession({})
    command(session, "drug_bolus", key=hypnotic, amount=amount)
    result = command(session, "drug_bolus", key="roc", amount=42)
    assert "feedback" not in result
    assert result["medication"]["key"] == "roc"
    # Pending induction is not permission to instrument an awake, reactive airway.
    attempted = command(session, "airway", mode="ETT")
    assert attempted["feedback"]["code"] == "awake_intubation"


def test_small_hypnotic_dose_does_not_silence_an_awake_paralysis_warning():
    session = WebSession({})
    command(session, "drug_bolus", key="propofol", amount=5)
    result = command(session, "drug_bolus", key="roc", amount=42)
    assert result["feedback"]["code"] == "awake_neuromuscular_block"


@pytest.mark.parametrize("ce", [1.8, 2.2, 3.0])
def test_borderline_hypnosis_and_partial_airway_reflex_suppression_do_not_generate_advisories(ce):
    session = WebSession({})
    engine = session.engine
    engine.pk["propofol"].state.ce = engine.pk["propofol"].state.ce_response = ce
    assert command(session, "airway", mode="ETT") is None
    assert 0.1 < loss_of_response(engine) < 0.9
    assert command(session, "drug_bolus", key="roc", amount=42).get("feedback") is None


def test_intubation_transient_preserves_ongoing_surgery_and_expires():
    session = WebSession({"mode": "steady_state"})
    engine = session.engine
    engine.start()
    engine.set_airway_mode("Mask")
    profile, started = engine.disturbance_profile, engine.disturbance_start_time
    engine.attempt_intubation()
    assert engine.intubation_start_time == engine.state.time
    advance(engine, 20)
    combined = stimulus_effects(engine, 0.1)
    pulse_start = engine.intubation_start_time
    engine.intubation_start_time = None
    surgery = stimulus_effects(engine, 0.1)
    engine.intubation_start_time = pulse_start
    assert combined.hr > surgery.hr and combined.svr > surgery.svr
    advance(engine, 281, 1)
    assert engine.intubation_start_time is None
    assert engine.disturbance_active and (engine.disturbance_profile, engine.disturbance_start_time) == (profile, started)


def test_ventilation_changes_and_extubation_without_a_hazard_give_no_feedback():
    session = WebSession({})
    engine = session.engine
    # Setting up assistance while disconnected, and switching between bag and ventilator.
    assert command(session, "bag_mask", active=True) is None
    assert command(session, "vent_power", on=True) is None
    assert command(session, "airway", mode="Mask") is None
    # Slow spontaneous breathing is not apnea.
    engine.resp.state.rr, engine.resp.state.vt = 5.0, 150.0
    assert not engine.resp.state.apnea
    assert command(session, "vent_power", on=False) is None
    command(session, "bag_mask", active=True)
    assert command(session, "bag_mask", active=False) is None
    # Tube removal after recovery.
    engine.set_airway_mode("ETT")  # Fixture connection; no learner intubation.
    assert command(session, "airway", mode="Mask") is None


def test_airway_removal_and_stopping_ventilation_use_unassisted_breathing():
    session = WebSession({"mode": "steady_state"})
    engine = session.engine
    # The displayed RR/VT come from the ventilator despite absent spontaneous drive.
    assert engine.state.rr > 6 and engine.state.vt > 300
    stopped = command(session, "vent_power", on=False)
    assert stopped["feedback"]["code"] == "ventilation_stopped"
    removed = command(session, "airway", mode="Mask")
    assert removed["feedback"]["code"] == "unsafe_airway_removal"
    assert engine.state.airway_mode.value == "Mask" and not engine.vent.is_on
