"""Browser sessions driven through the same JSON commands the web page sends."""

import json

import pytest

from anasim.core.drug_registry import DRUG_REGISTRY
from anasim.machine.ventilator import MODES
from anasim.web import WebSession

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


def test_ventilator_settings_survive_bag_mask_handover():
    session = WebSession({**PATIENT, "mode": "steady_state"})
    snap = parse(session.advance(0.0))
    assert snap["controls"]["vent"]["on"]

    cmd(session, "vent", mode="PCV", p_insp=18, peep=8, p_max=35, trigger=2.5)
    cmd(session, "bag_mask", active=True)
    vent = parse(session.advance(0.0))["controls"]["vent"]
    assert not vent["on"]
    assert (vent["mode"], vent["p_insp"], vent["peep"], vent["p_max"], vent["trigger"]) == ("PCV", 18, 8, 35, 2.5)
    invalid_updates = (
        {"mode": "SIMV"},
        {"p_max": 8},
        {"t_insp": 0},
        {"rr": 18, "ie": "1:0"},
        {"rr": 18, "tv": float("nan")},
        {"rr": 18, "tv": None},
        {"rr": 18, "p_support": float("inf")},
        {"rr": 18, "unknown": 1},
    )
    for bad in invalid_updates:
        with pytest.raises(ValueError):
            cmd(session, "vent", **bad)
        assert parse(session.advance(0.0))["controls"]["vent"] == vent

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
    assert loop["volume"][-1] == pytest.approx(0, abs=1.0)  # Slow aeration changes can shift end-expiratory volume.
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

    # A brief disconnection must not accumulate into the next episode's delay.
    cmd(session, "airway", mode="None")
    assert all("MV" not in snap["alarms"] for snap in run_seconds(session, 10))
    cmd(session, "airway", mode="ETT")
    assert "MV" not in run_seconds(session, 20)[-1]["alarms"]

    # The circuit then measures no exhaled gas, and the MV alarm sounds after its delay.
    cmd(session, "airway", mode="None")
    snaps = run_seconds(session, 20)
    alarm_onset = next(i for i, snap in enumerate(snaps, 1) if snap["alarms"].get("MV") == "low")
    assert 15 <= alarm_onset <= 16
    v = snaps[-1]["vitals"]
    assert v["vte"] is v["ppeak"] is v["eto2"] is None
    # After disconnection RR comes from the patient's chest movement.
    assert v["rr_source"] == "impedance" and v["rr"] < 6.0
    assert v["rr"] == pytest.approx(engine.resp.state.rr, abs=0.1)

    cmd(session, "airway", mode="ETT")
    assert "MV" not in run_seconds(session, 20)[-1]["alarms"]


@pytest.mark.parametrize("mode", [None, *MODES])
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
    completed = [runs[breath] for breath in sorted(runs)[-3:-1]]
    assert max(p[2] for points in completed for p in points) > 350.0
    for points in completed:
        assert points[0][1] == points[-1][1] == 0.0
        assert points[0][2] == 0.0
        assert min(p[1] for p in points) < 0 < max(p[1] for p in points)
        assert max(p[2] for p in points) > 50.0  # Unsupported breaths between mandatory breaths can be smaller.
        if mode in (None, "PSV", "CPAP"):
            assert points[-1][2] == pytest.approx(0.0, abs=0.1)
        elif mode in ("PCV", "SIMV-PC"):
            # Mixed breaths preserve changes in end-expiratory volume.
            assert points[-1][2] != pytest.approx(0.0, abs=0.5)


def test_medication_acknowledgments_reflect_accepted_actions_and_survive_reload():
    session = WebSession({**PATIENT, "tci_enabled": True})
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
    assert history[-1]["text"] == "Propofol infusion 54 mcg/kg/min"


def test_manual_session_hides_and_rejects_tci_for_every_supported_drug():
    session = WebSession({**PATIENT, "mode": "steady_state"})
    drugs = parse(session.info())["drugs"]
    assert all(drug["tci_unit"] is drug["tci_range"] is drug["target_label"] is None for drug in drugs)
    before = session.snapshot()["controls"]["drugs"]
    for spec in DRUG_REGISTRY:
        if spec.has_tci:
            assert not before[spec.key]["is_tci"]
            with pytest.raises(ValueError, match="TCI is disabled"):
                cmd(session, "drug_target", key=spec.key, target=3)
            with pytest.raises(ValueError, match="TCI is disabled"):
                session.engine.enable_tci(spec.key, 3)
    assert session.snapshot()["controls"]["drugs"] == before


@pytest.mark.parametrize("weight", [60.0, 90.0])
def test_weight_based_rates_reach_the_engine_and_round_trip_through_browser_commands(weight):
    session = WebSession({**PATIENT, "weight": weight})
    specs = {spec["key"]: spec for spec in parse(session.info())["drugs"]}
    for key, rate in (("propofol", 100.0), ("remi", 0.15)):
        assert specs[key]["rate_unit"] == "mcg/kg/min"
        receipt = cmd(session, "drug_rate", key=key, rate=rate)["medication"]
        assert f"{rate:g} mcg/kg/min" in receipt["text"]
    assert session.engine.propofol_rate_mg_sec == pytest.approx(100.0 * weight / 60000)
    assert session.engine.remi_rate_ug_sec == pytest.approx(0.15 * weight / 60)
    cmd(session, "run", running=True)
    snap = run_seconds(session, 1)[-1]
    for key, rate in (("propofol", 100.0), ("remi", 0.15)):
        assert snap["controls"]["drugs"][key]["rate"] == pytest.approx(rate)
        assert not snap["controls"]["drugs"][key]["is_tci"]
    assert session.engine.pk_prop.state.c1 > 0 and session.engine.pk_remi.state.c1 > 0


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


@pytest.mark.parametrize("mode", ["SIMV-VC", "SIMV-PC", "SIMV-VG"])
def test_simv_shows_untriggered_breaths_in_numerics_and_loops(mode):
    session = WebSession({**PATIENT, "mode": "awake"})
    engine = session.engine
    engine.resp.hcvr_slope_baseline = 0.0
    engine.resp.rr_0, engine.resp.vt_0 = 18.0, 250.0
    cmd(session, "airway", mode="ETT")
    cmd(session, "vent", mode=mode, rr=3, p_insp=8, p_support=5, trigger=20, t_insp=0.5)
    cmd(session, "vent_power", on=True)
    cmd(session, "run", running=True)
    runs = {}
    for snap in run_seconds(session, 58):
        for loop in snap["loop"]:
            runs.setdefault(loop["breath"], []).extend(zip(loop["paw"], loop["flow"], loop["volume"]))
    assert snap["vitals"]["rr"] == pytest.approx(18, abs=0.2)
    assert 200 < snap["vitals"]["vte"] < 260
    assert snap["vitals"]["mv"] > 3.5
    assert snap["vitals"]["etco2"] > 20
    # The latest unsupported breaths retain both limbs and their zero-flow boundaries.
    for breath in sorted(runs)[-3:-1]:
        points = runs[breath]
        assert points[0][1] == points[-1][1] == 0
        assert min(p[1] for p in points) < 0 < max(p[1] for p in points) < 20
        assert 200 < max(p[2] for p in points) < 260
        assert max(p[0] for p in points) < 8  # No pressure support was triggered.
