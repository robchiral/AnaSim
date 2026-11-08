"""Case events and debriefs must work independently of instructional progression."""

import json

import pytest

from anasim.core.action_log import ACTION_DRUG_BOLUS
from anasim.monitors.neuromuscular import TwitchReading
from anasim.scenarios import SCENARIO_REGISTRY
from anasim.web import WebSession


def command(session, name, **args):
    return json.loads(session.command(name, json.dumps(args)))


def advance(session, seconds):
    command(session, "speed", value=50)
    command(session, "run", running=True)
    remaining = seconds
    while remaining > 1e-8:
        interval = min(remaining, 10)
        session.step(interval / 50)
        remaining -= interval


@pytest.mark.parametrize("spec", SCENARIO_REGISTRY, ids=lambda s: s.id)
def test_independent_cases_send_no_guide_text_and_end_with_a_debrief(spec):
    session = WebSession({"scenario_id": spec.id, "practice_mode": "independent"})
    state = session.snapshot()["scenario"]
    assert state["mode"] == "independent" and state["briefing"]
    assert not {"instruction", "title", "index", "status", "met", "target_tab", "outcomes"} & state.keys()
    assert state["hint"] is state["debrief"] is None
    with pytest.raises(ValueError, match="guided"):
        command(session, "scenario_next")
    result = command(session, "case_finish")["debrief"]
    assert len(result["outcomes"]) >= 3


@pytest.mark.parametrize("mode", ["guided", "independent"])
@pytest.mark.parametrize("case,event,dt", [
    ("hemorrhage_response", "hemorrhage", 0.1),
    ("sepsis_response", "sepsis", 0.01), ("sepsis_response", "sepsis", 0.1),
    ("anaphylaxis_response", "anaphylaxis", 0.01), ("anaphylaxis_response", "anaphylaxis", 0.1),
    ("malignant_hyperthermia", "malignant_hyperthermia", 0.1),
    ("oxygen_supply_failure", "oxygen_supply", 0.1),
])
def test_crises_start_once_on_the_case_clock_without_clicking_steps(mode, case, event, dt):
    session = WebSession({"scenario_id": case, "practice_mode": mode})
    session.engine.config.dt = dt
    assert session.case_run is not None and not session.case_run.fired
    assert session.snapshot()["scenario"]["actions"] == []
    advance(session, 31)
    assert session.case_run.fired == {event}
    observations = session.snapshot()["scenario"]["observations"]
    assert len(observations) == 1 and observations[0]["time"] == pytest.approx(30, abs=0.11)
    advance(session, 14)
    assert session.snapshot()["scenario"]["observations"] == observations
    if event in {"anaphylaxis", "sepsis"}:
        assert session.engine.state.map < 65
        assert session.engine.state.monitored_blood_pressure(True)[2] < 65


@pytest.mark.parametrize("dt", [0.1, 1.0])
def test_untreated_severe_anaphylaxis_does_not_recover_normal_pressure(dt):
    session = WebSession({"scenario_id": "anaphylaxis_response"})
    engine = session.engine
    engine.config.dt = dt
    engine.set_continuous_fluid_rate(0)
    model = engine.hemo
    initial_volume = engine.state.blood_volume
    cells, hemoglobin = model.red_cell_volume_ml, model.hb_mass
    advance(session, 900)
    assert engine.anaphylaxis.active and engine.anaphylaxis.severity == 1
    assert engine.state.map < 65
    assert engine.state.blood_volume < 0.7 * initial_volume
    assert engine.state.bronchospasm > 0.9
    assert model.red_cell_volume_ml == cells
    assert model.hb_mass == hemoglobin
    assert model.hb_conc > model.baseline_hb
    assert (model.blood_volume + model.total_third_space_ml + model.lung_water_ml
            + model.crystalloid.interstitial_ml + model.total_urine_out_ml) == pytest.approx(initial_volume)


@pytest.mark.parametrize("arterial_line", [False, True])
def test_anaphylaxis_recovery_requires_current_pressure_and_cuff_reassessment(arterial_line):
    session = WebSession({"scenario_id": "anaphylaxis_response", "arterial_line_enabled": arterial_line})
    session.engine.config.dt = 0.1
    advance(session, 31)
    command(session, "scenario_next")
    command(session, "drug_bolus", key="epi", amount=100)
    command(session, "fgf", o2=10, air=0, n2o=0)
    command(session, "scenario_next")
    command(session, "fluid", kind="crystalloid", volume_ml=1000)
    command(session, "scenario_next")
    command(session, "case_action", action="stop_trigger")
    assert session.engine.state.monitored_blood_pressure(arterial_line)[2] >= 65
    assert session.engine.state.map < 65
    state = session.snapshot()["scenario"]
    assert state["id"] == "STABILIZE" and not state["met"]
    assert "Reassess blood pressure and perfusion" in state["status"]
    command(session, "scenario_next")
    assert not session.snapshot()["scenario"]["complete"]

    command(session, "drug_rate", key="epi", rate=6)
    advance(session, 200)
    if not arterial_line:
        assert session.engine.state.map >= 65
        assert session.snapshot()["scenario"]["status"] == "Measure blood pressure again"
        command(session, "scenario_next")
        assert not session.snapshot()["scenario"]["complete"]
        command(session, "nibp")
        advance(session, 12)
    assert session.snapshot()["scenario"]["met"]
    command(session, "scenario_next")
    state = session.snapshot()["scenario"]
    assert state["complete"]
    assert all(outcome["achieved"] for outcome in state["debrief"]["outcomes"])


def test_early_crisis_treatment_counts_without_repeating_doses_or_following_the_guide():
    session = WebSession({"scenario_id": "anaphylaxis_response", "practice_mode": "independent"})
    session.engine.config.dt = 0.1
    advance(session, 31)
    command(session, "drug_bolus", key="epi", amount=100)
    command(session, "fluid", kind="crystalloid", volume_ml=1000)
    command(session, "blood_gas")
    command(session, "case_action", action="stop_trigger")
    report = command(session, "case_finish")["debrief"]
    outcomes = {o["id"]: o["achieved"] for o in report["outcomes"]}
    assert outcomes["epinephrine"] and outcomes["volume"] and outcomes["source"]
    timeline = [item["text"] for item in report["timeline"]]
    assert {"1000 mL lactated Ringer's started", "Arterial blood gas drawn"} <= set(timeline)
    doses = [r for r in session.engine.actions.since_case(ACTION_DRUG_BOLUS) if r.label == "epi"]
    assert len(doses) == 1 and doses[0].amount == 100


def test_independent_feedback_defers_explanations_but_keeps_observable_failed_actions():
    session = WebSession({"scenario_id": "induction_tiva", "practice_mode": "independent"})
    failed = command(session, "airway", mode="ETT")
    assert failed["feedback"]["level"] == "blocked"
    command(session, "dismiss_feedback")
    dose = command(session, "drug_bolus", key="roc", amount=42)
    assert "medication" in dose and "feedback" not in dose
    assert session.snapshot()["controls"]["clinical_feedback"] is None
    report = command(session, "case_finish")["debrief"]
    assert [event["code"] for event in report["safety_events"]] == ["awake_intubation", "awake_neuromuscular_block"]
    assert any("Rocuronium" in item["text"] for item in report["timeline"])
    assert len(report["learning_points"]) <= 3


def test_guided_feedback_remains_immediate_and_is_also_in_the_debrief():
    session = WebSession({"scenario_id": "induction_tiva"})
    assert command(session, "drug_bolus", key="roc", amount=42)["feedback"]["level"] == "warning"
    report = command(session, "case_finish")["debrief"]
    assert report["safety_events"][0]["code"] == "awake_neuromuscular_block"


def test_hints_are_requested_explicitly_and_reported_as_assistance():
    session = WebSession({"scenario_id": "induction_tiva", "practice_mode": "independent"})
    hint = command(session, "case_hint")["hint"]
    assert hint and session.snapshot()["scenario"]["hint"] == hint
    report = command(session, "case_finish")["debrief"]
    assert report["hints_used"] == 1
    assert any(event["kind"] == "hint" for event in report["timeline"])


def test_case_end_freezes_the_report_and_rejects_further_treatment(tmp_path):
    session = WebSession({"scenario_id": "induction_tiva", "practice_mode": "independent"}, recordings_dir=str(tmp_path))
    session.engine.config.dt = 0.1
    command(session, "record", active=True)
    advance(session, 2)
    report = command(session, "case_finish")["debrief"]
    snap = session.snapshot()
    assert snap["ended"] and not snap["running"] and not snap["recording"]
    assert snap["download"] and snap["scenario"]["debrief"] == report
    commands: list[tuple[str, dict[str, object]]] = [
        ("drug_bolus", {"key": "propofol", "amount": 50}), ("run", {"running": True}),
        ("record", {"active": True}), ("case_hint", {}),
    ]
    for name, args in commands:
        with pytest.raises(ValueError, match="ended"):
            command(session, name, **args)
    ended_at = session.engine.state.time
    session.step(0.2)
    assert session.engine.state.time == ended_at == report["duration"]
    assert command(session, "case_finish")["debrief"] == report


def test_instructor_commands_cannot_change_a_case_and_case_actions_require_the_event():
    session = WebSession({"scenario_id": "hemorrhage_response", "practice_mode": "independent"})
    for name in ("hemorrhage", "sepsis", "anaphylaxis", "stop_events", "oxygen_supply"):
        with pytest.raises(ValueError, match="controlled by the case"):
            command(session, name)
    with pytest.raises(ValueError, match="not currently available"):
        command(session, "case_action", action="hemostasis")


def test_debrief_metrics_account_for_elapsed_physiology_and_end_state_outcomes():
    session = WebSession({"scenario_id": "induction_tiva", "practice_mode": "independent"})
    run = session.case_run
    assert run is not None
    session.engine.state.map, session.engine.state.sao2 = 55, 85
    session.engine.state.time = 2
    run.observe(2, force=True)
    session.engine.state.map, session.engine.state.sao2 = 75, 98
    session.engine.state.time = 5
    run.observe(3, force=True)
    report = command(session, "case_finish")["debrief"]
    assert report["duration"] == 5
    assert report["physiology"] == {"lowest_map": 55, "lowest_spo2": 85, "hypotension_s": 2, "hypoxemia_s": 2}


def test_independent_induction_completes_without_a_walkthrough_or_mandatory_paralysis():
    session = WebSession({"scenario_id": "induction_tiva", "practice_mode": "independent", "tci_enabled": True})
    session.engine.config.dt = 0.1
    command(session, "airway", mode="Mask")
    command(session, "fgf", o2=10, air=0, n2o=0)
    advance(session, 240)
    command(session, "drug_target", key="propofol", target=4)
    command(session, "drug_target", key="remi", target=4)
    command(session, "bag_mask", active=True)
    advance(session, 300)
    command(session, "airway", mode="ETT")
    command(session, "vent_power", on=True)
    command(session, "drug_rate", key="nore", rate=6)
    advance(session, 60)
    assert session.engine.state.tof == session.engine.state.twitch == 100
    report = command(session, "case_finish")["debrief"]
    assert all(outcome["achieved"] for outcome in report["outcomes"])
    assert report["safety_events"] == [] and report["hints_used"] == 0


def test_cardiac_arrest_ends_the_case_with_a_debrief():
    session = WebSession({"scenario_id": "induction_tiva", "practice_mode": "independent"})
    session.engine.start()
    session.engine.state.cardiac_arrest = True
    session.engine.state.arrest_reason = "No effective circulation"
    session.engine.hemo.coronary.injury = 1.0
    session.step(0.1)
    snap = session.snapshot()
    assert snap["ended"] and not snap["running"]
    assert snap["scenario"]["debrief"]["reason"] == "cardiac_arrest"


def test_stale_monitor_values_do_not_mask_instability_in_the_debrief():
    session = WebSession({"scenario_id": "induction_tiva", "practice_mode": "independent"})
    session.engine.state.map = 55
    session.engine.state.display_spo2 = 98
    session.engine.state.nibp_map = 90
    report = command(session, "case_finish")["debrief"]
    stability = next(o for o in report["outcomes"] if o["id"] == "stability")
    assert not stability["achieved"] and report["physiology"]["lowest_map"] == 55


def test_earlier_readiness_does_not_mask_residual_block_at_actual_tube_removal():
    session = WebSession({"scenario_id": "emergence_tiva", "practice_mode": "independent"})
    engine, run = session.engine, session.case_run
    assert run is not None
    engine.state.display_bis, engine.state.display_spo2 = 90, 98
    engine.state.rr, engine.state.vt = 12, 500
    engine.resp.state.rr, engine.resp.state.apnea = 12, False
    engine.state.tof = engine.state.twitch = 100
    engine.nmb.reading = TwitchReading(0, "tof", (1, 1, 1, 1), 4, 100, 100, False)
    engine.set_vent_power(False)
    run.observe(force=True)
    assert "readiness" not in run.achieved_at
    # Actual residual block is still present in the underlying PK/PD models.
    command(session, "airway", mode="Mask")
    engine.state.tof = engine.state.twitch = 100
    run.observe(force=True)
    report = command(session, "case_finish")["debrief"]
    outcomes = {o["id"]: o["achieved"] for o in report["outcomes"]}
    assert not outcomes["block"] and not outcomes["readiness"]
    assert report["safety_events"][0]["code"] == "unsafe_airway_removal"
