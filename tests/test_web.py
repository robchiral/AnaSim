"""Browser sessions driven through the same JSON commands the web page sends."""

import json

import pytest

from anasim.web import WebSession, catalog

PATIENT = dict(age=40, weight=70, height=170, sex="male")


def _reject_constant(name):
    raise AssertionError(f"Snapshot is not valid browser JSON: {name}")


def parse(text):
    """Parse like JSON.parse, which rejects NaN and Infinity."""
    return json.loads(text, parse_constant=_reject_constant)


def cmd(session, name, /, **args):
    return parse(session.command(name, json.dumps(args)))


def run_seconds(session, seconds, tick_s=1.0):
    """Advance simulated seconds at 5x in 0.2 s real ticks; return the snapshots."""
    cmd(session, "speed", value=tick_s / 0.2)
    return [parse(session.advance(0.2)) for _ in range(round(seconds / tick_s))]


# Learner commands for each objective of the guided TIVA induction.
TIVA_INDUCTION = {
    "APPLY_MASK": [("airway", {"mode": "Mask"})],
    "SET_FGF_PREOX": [("fgf", {"o2": 10, "air": 0, "n2o": 0})],
    "START_ANALGESIA": [("drug_target", {"key": "remi", "target": 4})],
    "INDUCE": [
        ("drug_bolus", {"key": "propofol", "amount": 175}),
        ("drug_target", {"key": "propofol", "target": 4}),
    ],
    "MASK_VENTILATE": [("bag_mask", {"active": True})],
    "GIVE_NMB": [("drug_bolus", {"key": "roc", "amount": 50})],
    "INTUBATE": [("airway", {"mode": "ETT"})],
    "CONFIRM_ETT": [("vent_power", {"on": True})],
    "MAINTENANCE": [("fgf", {"o2": 2, "air": 0, "n2o": 0})],
}


def test_guided_tiva_induction_completes_through_browser_commands():
    session = WebSession({**PATIENT, "scenario_id": "induction_tiva"})
    info = parse(session.info())
    assert info["scenario"]["total"] == 12
    cmd(session, "run", running=True)
    # Continue does nothing until the objective is met.
    cmd(session, "scenario_next")
    assert parse(session.advance(0.0))["scenario"]["id"] == "APPLY_MASK"

    pending = dict(TIVA_INDUCTION)
    samples = 0
    snap = parse(session.advance(0.0))
    while not snap["scenario"]["complete"]:
        step = snap["scenario"]
        for name, args in pending.pop(step["id"], []):
            cmd(session, name, **args)
        if parse(session.advance(0.0))["scenario"]["met"]:
            cmd(session, "scenario_next")
        else:
            assert snap["time"] < 1200, f"{step['id']} unreachable: {step['status']}"
            (snap,) = run_seconds(session, 1.0)
            samples += len(snap["waves"]["ecg"])
        snap = parse(session.advance(0.0))

    controls = snap["controls"]
    # After intubation the ventilator takes over from the bag.
    assert controls["airway"] == "ETT" and not controls["bag_mask"] and controls["vent"]["on"]
    assert snap["vitals"]["etco2"] > 25
    assert controls["drugs"]["propofol"]["is_tci"] and controls["drugs"]["remi"]["is_tci"]
    assert snap["vitals"]["bis"] < 60
    assert snap["vitals"]["tof"] < 25
    # Every simulated step reaches the monitor sweep exactly once.
    assert samples == round(snap["time"] / info["sample_interval"])


def test_ventilator_settings_survive_bag_mask_handover():
    session = WebSession({**PATIENT, "mode": "steady_state"})
    snap = parse(session.advance(0.0))
    assert snap["controls"]["vent"]["on"]

    cmd(session, "vent", mode="PCV", p_insp=18, peep=8, p_max=35, trigger=2.5)
    cmd(session, "bag_mask", active=True)
    vent = parse(session.advance(0.0))["controls"]["vent"]
    assert not vent["on"]
    assert (vent["mode"], vent["p_insp"], vent["peep"], vent["p_max"], vent["trigger"]) == ("PCV", 18, 8, 35, 2.5)
    for bad in ({"mode": "SIMV"}, {"p_max": 8}, {"t_insp": 0}):
        with pytest.raises(ValueError):
            cmd(session, "vent", **bad)

    cmd(session, "vent_power", on=True)
    cmd(session, "run", running=True)
    snap = run_seconds(session, 60)[-1]
    settings = session.engine.vent.settings
    assert not snap["controls"]["bag_mask"]
    assert (settings.mode, settings.p_insp, settings.peep) == ("PCV", 18, 8)
    assert 25 <= snap["vitals"]["etco2"] <= 50


def test_ventilator_display_and_disconnection_alarm():
    session = WebSession({**PATIENT, "mode": "steady_state"})
    engine = session.engine
    cmd(session, "run", running=True)
    snaps = run_seconds(session, 30)
    v = snaps[-1]["vitals"]
    assert all(len(samples) == len(snap["waves"]["ecg"]) for snap in snaps for samples in snap["waves"].values())

    # VCV: constant inspiratory flow until the pause and a Paw trace from PEEP to
    # Ppeak. Tissue stress has not fully relaxed by the end of a short pause, so
    # Pplat sits above static recoil.
    paw = [p for snap in snaps[-10:] for p in snap["waves"]["paw"]]
    flow = [f for snap in snaps[-10:] for f in snap["waves"]["flow"]]
    assert v["peep"] == 5 and v["ppeak"] > v["pplat"] > v["pmean"] > v["peep"]
    assert v["pplat"] - v["peep"] > v["vte"] / (engine.resp_mech.compliance * 1000)
    assert v["cdyn"] == pytest.approx(v["vte"] / (v["ppeak"] - v["peep"]), abs=1)
    assert v["mv"] == pytest.approx(v["vte"] * v["rr"] / 1000, rel=0.01)
    assert (min(paw), max(paw)) == pytest.approx((v["peep"], v["ppeak"]), abs=0.5)
    flow_time = 60 / v["rr"] / 3 * (1 - engine.vent.settings.pause / 100)
    assert max(flow) == pytest.approx(v["vte"] / 1000 / flow_time * 60, abs=0.2)
    # Loops trace each breath as it happens; a completed one spans VTe and Ppeak.
    runs = {}
    for snap in snaps:
        for run in snap["loop"]:
            runs.setdefault(run["breath"], []).append(run)
    completed = runs[sorted(runs)[-2]]
    assert len(completed) > 1
    loop = {key: [x for run in completed for x in run[key]] for key in ("paw", "flow", "volume")}
    assert loop["volume"][0] == 0 and max(loop["volume"]) == pytest.approx(v["vte"], rel=0.05)
    assert max(loop["paw"]) == pytest.approx(v["ppeak"], abs=0.5)
    # A stable breath includes the zero-flow boundary at both ends, rather than
    # leaving a gap between the last expiration sample and the next inspiration.
    assert loop["flow"][0] == loop["flow"][-1] == 0
    assert loop["volume"][-1] == pytest.approx(0, abs=0.1)
    assert loop["paw"][-1] == pytest.approx(loop["paw"][0], abs=0.01)
    # A reloaded page gets the previous and current breaths back.
    session.replay_waves()
    replayed = parse(session.advance(0.0))["loop"]
    assert [run["breath"] for run in replayed] == sorted(runs)[-2:]
    assert replayed[0]["volume"] == loop["volume"]
    assert v["rr_source"] == "co2"
    # Oxygen uptake keeps end-tidal below inspired O2.
    assert 70 < v["eto2"] < v["fio2"] - 3
    assert snaps[-1]["alarms"] == {}

    # The circuit then measures no exhaled gas, and the MV alarm sounds after its delay.
    cmd(session, "airway", mode="None")
    snaps = run_seconds(session, 20)
    alarm_onset = next(i for i, snap in enumerate(snaps, 1) if snap["alarms"].get("MV") == "low")
    assert 15 <= alarm_onset <= 16
    v = snaps[-1]["vitals"]
    assert v["vte"] is v["ppeak"] is v["eto2"] is None
    # RR now counts the apneic patient's chest movement, not ventilator breaths.
    assert v["rr_source"] == "impedance" and v["rr"] == 0

    cmd(session, "airway", mode="ETT")
    assert "MV" not in run_seconds(session, 20)[-1]["alarms"]


@pytest.mark.parametrize("mode", [None, "PSV", "PCV", "SIMV-PC"])
def test_loops_retain_the_flow_boundary_across_modes(mode):
    session = WebSession({**PATIENT, "mode": "awake"})
    session.engine.resp.hcvr_slope_baseline = 0.0
    cmd(session, "airway", mode="ETT")
    cmd(session, "vent", mode=mode or "VCV", rr=6, peep=5, p_insp=12, p_support=5)
    cmd(session, "vent_power", on=mode is not None)
    cmd(session, "run", running=True)
    runs = {}
    for snap in run_seconds(session, 35, tick_s=0.1):
        for run in snap["loop"]:
            runs.setdefault(run["breath"], []).extend(zip(run["paw"], run["flow"], run["volume"]))
    for breath in sorted(runs)[-3:-1]:
        points = runs[breath]
        assert points[0][1] == points[-1][1] == 0.0
        assert points[0][2] == 0.0
        assert min(p[1] for p in points) < 0 < max(p[1] for p in points)
        assert max(p[2] for p in points) > 350.0
        if mode in (None, "PSV"):
            assert points[-1][2] == pytest.approx(0.0, abs=0.1)
        else:
            # Mixed breaths preserve changes in end-expiratory volume.
            assert points[-1][2] != pytest.approx(0.0, abs=0.5)


def test_medication_acknowledgments_reflect_accepted_actions_and_survive_reload():
    session = WebSession(PATIENT)
    assert parse(session.info())["medication_history"] == []
    first = cmd(session, "drug_bolus", key="Propofol", amount=50)["medication"]
    assert first["key"] == "propofol"
    second = cmd(session, "drug_bolus", key="propofol", amount=50)["medication"]
    assert first["time"] == second["time"] == 0
    assert second["id"] == first["id"] + 1
    assert first["text"] == "Propofol 50 mg given"
    cmd(session, "drug_target", key="propofol", target=3)
    stopped = cmd(session, "drug_rate", key="propofol", rate=0)["medication"]
    assert stopped["text"] == "Propofol infusion stopped"
    assert not session.engine.get_drug_state("propofol")["is_tci"]
    reversal = cmd(session, "sugammadex", mg_per_kg=2)["medication"]
    assert reversal["text"] == "Sugammadex 140 mg given (2 mg/kg)"
    before = parse(session.info())["medication_history"]
    concentration = session.engine.pk_prop.state.c1
    for key, amount in [("missing", 50), ("Propofol 10 mg/mL", 50), ("propofol", 0), ("propofol", float("nan"))]:
        with pytest.raises(ValueError):
            cmd(session, "drug_bolus", key=key, amount=amount)
    assert session.engine.pk_prop.state.c1 == concentration
    session.replay_waves()
    assert parse(session.info())["medication_history"] == before
    for rate in range(55):
        cmd(session, "drug_rate", key="propofol", rate=rate)
    history = parse(session.info())["medication_history"]
    assert len(history) == 50
    assert history[-1]["text"] == "Propofol infusion 54 mg/hr"


def test_recording_returns_the_session_as_csv_even_after_a_failure(tmp_path):
    session = WebSession(PATIENT, recordings_dir=str(tmp_path))
    cmd(session, "run", running=True)
    cmd(session, "record", active=True)
    run_seconds(session, 5)
    assert parse(session.advance(0.0))["recording"]

    result = cmd(session, "record", active=False)

    header, *rows = result["csv"].strip().splitlines()
    times = [float(row.split(",")[0]) for row in rows]
    assert header.startswith("time,")
    assert len(rows) == 5 and times == sorted(times)
    assert not any(tmp_path.iterdir())
    assert not parse(session.advance(0.0))["recording"]

    # A write failure pauses the session and still hands over the rows recorded so far.
    cmd(session, "record", active=True)
    run_seconds(session, 3)
    session.engine.recorder.file.close()
    (snap,) = run_seconds(session, 1)
    assert not snap["running"] and not snap["recording"]
    assert "incomplete" in snap["notice"]
    header, *rows = snap["download"]["csv"].strip().splitlines()
    assert header.startswith("time,") and len(rows) == 3
    assert not any(tmp_path.iterdir())


def test_cardiac_arrest_ends_the_session():
    session = WebSession({**PATIENT, "end_on_cardiac_arrest": True})
    engine = session.engine
    steps_after_arrest = []
    step = engine.step

    def recording_step(dt):
        if engine.state.cardiac_arrest:
            steps_after_arrest.append(dt)
        step(dt)

    engine.step = recording_step
    cmd(session, "run", running=True)
    cmd(session, "rhythm", name="Asystole")

    snaps = run_seconds(session, 40, tick_s=10.0)

    snap = snaps[-1]
    assert snap["ended"] and not snap["running"]
    assert snap["arrest_reason"]
    # The final state is the one at the arrest, even at 50x.
    assert not steps_after_arrest
    cmd(session, "run", running=True)
    assert not parse(session.advance(0.2))["running"]


def test_setup_choices_match_supported_sessions():
    choices = parse(catalog())
    for scenario in choices["scenarios"]:
        WebSession({**PATIENT, "scenario_id": scenario["id"]})

    with pytest.raises(ValueError, match="bmi"):
        WebSession({**PATIENT, "weight": 100, "height": 150})
    with pytest.raises(ValueError, match="Unknown session setting"):
        WebSession({**PATIENT, "tutorial_mode": True})
