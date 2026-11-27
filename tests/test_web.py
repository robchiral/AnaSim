"""Browser sessions driven through the same JSON commands the web page sends."""

import json
import re
from pathlib import Path

import pytest

from anasim import web
from anasim.core.drug_registry import DRUG_REGISTRY
from anasim.core.enums import RhythmType
from anasim.machine.ventilator import MODES
from anasim.scenarios import SCENARIO_REGISTRY
from anasim.web import WebSession

PATIENT = dict(age=40, weight=70, height=170, sex="male")


def test_condition_inputs_and_maternal_position_round_trip_through_browser_protocol():
    session = WebSession({**PATIENT, "sex": "female", "weight": 140, "mode": "awake",
                          "heart_failure": "hfpef", "copd_severity": 0.65, "surgical_site": "upper_abdominal",
                          "coronary_flow_reserve": 2, "gestational_age_weeks": 36,
                          "creatinine_clearance_ml_min": 30.0})
    info = parse(session.info())["patient"]
    assert info["creatinine_clearance_ml_min"] == session.engine.patient.creatinine_clearance() == 30.0
    assert info["heart_failure"] == "hfpef" and info["surgical_site"] == "upper_abdominal"
    assert info["copd_severity"] == 0.65 and info["coronary_flow_reserve"] == 2
    assert info["gestational_age_weeks"] == 36 and info["weight"] == 140
    cmd(session, "position", position="left_lateral")
    snap = session.snapshot()
    assert snap["controls"]["position"] == "left_lateral"
    assert snap["vitals"]["st_depression_mv"] == 0
    with pytest.raises(ValueError, match="position"):
        cmd(session, "position", position="upright")


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


@pytest.mark.parametrize("arterial", [True, False])
def test_trends_record_displayed_pressures_and_recover_after_reload(arterial):
    session = WebSession({**PATIENT, "mode": "steady_state", "arterial_line_enabled": arterial})
    initial = session.snapshot()["trends"]
    assert [sample["time"] for sample in initial] == [0]
    cmd(session, "run", running=True)
    snapshots = []
    # Poll at sampling boundaries to compare recorded and displayed measurements.
    for step in range(1200):
        session.step(0.01)
        if (step + 1) % 100 == 0:
            snapshots.append(session.snapshot())
    samples = initial + [sample for snap in snapshots for sample in snap["trends"]]
    assert [sample["time"] for sample in samples] == pytest.approx(range(13))
    for snap in snapshots:
        sample = snap["trends"][-1]
        vitals = snap["vitals"]
        displayed = vitals["art"] if arterial else vitals["nibp"]
        assert sample["map"] == displayed[2]
        assert (sample["hr"], sample["etco2"], sample["bis"]) == (
            vitals["hr"], vitals["etco2"], vitals["bis"],
        )
        if arterial:
            assert sample["map_time"] == sample["time"]
        else:
            assert sample["map_time"] <= sample["time"]
    cmd(session, "run", running=False)
    assert parse(session.advance(0.2))["trends"] == []
    session.replay_waves()
    assert parse(session.info())["trends"] == samples
    assert session.snapshot()["trends"] == []


def test_ventilator_and_bag_settings_survive_handover():
    session = WebSession({**PATIENT, "mode": "steady_state"})
    snap = parse(session.advance(0.0))
    assert snap["controls"]["vent"]["on"]

    cmd(session, "vent", mode="PCV", p_insp=18, peep=8, p_max=35, trigger=2.5)
    cmd(session, "bag_settings", rr=20, tv=400, p_max=25)
    cmd(session, "bag_mask", active=True)
    cmd(session, "mask_seal", percent=80)
    controls = parse(session.advance(0.0))["controls"]
    vent, bag = controls["vent"], controls["bag"]
    assert not vent["on"] and bag == {"rr": 20, "tv": 400}
    assert controls["apl"] == 25
    assert controls["mask_seal"] == 80
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
    for bad in (
        {"rr": 0, "tv": 400},
        {"rr": 41, "tv": 400},
        {"rr": 20, "tv": 1300},
        {"rr": 20, "tv": float("nan")},
        {"rr": 20, "tv": 400, "p_max": -1},
        {"rr": 20, "tv": 400, "p_max": 71},
        {"rr": 20, "tv": 400, "p_max": float("nan")},
    ):
        with pytest.raises(ValueError):
            cmd(session, "bag_settings", **bad)
        unchanged = parse(session.advance(0.0))["controls"]
        assert unchanged["bag"] == bag and unchanged["apl"] == 25
    for bad_seal in (-1, 101, float("nan")):
        with pytest.raises(ValueError):
            cmd(session, "mask_seal", percent=bad_seal)
        assert parse(session.advance(0.0))["controls"]["mask_seal"] == 80
    cmd(session, "mask_seal", percent=100)
    controls = parse(session.advance(0.0))["controls"]
    for name, args in (
        ("fgf", {"o2": float("inf"), "air": 0, "n2o": 0}),
        ("vaporizer", {"percent": float("nan")}),
        ("obstruction", {"percent": float("nan")}),
        ("bronchospasm", {"percent": 101}),
        ("bair_hugger", {"target_c": 400}),
        ("maintenance_fluid", {"ml_hr": None}),
        ("speed", {"value": float("nan")}),
        ("fluid", {"kind": "plasma", "volume_ml": 250}),
    ):
        with pytest.raises(ValueError):
            cmd(session, name, **args)
    assert parse(session.advance(0.0))["controls"] == controls

    # The monitor counts every bag breath, and expired volume is near the set volume.
    cmd(session, "run", running=True)
    vitals = run_seconds(session, 30)[-1]["vitals"]
    assert vitals["rr"] == 20 and 380 <= vitals["vte"] <= 440

    cmd(session, "vent_power", on=True)
    snap = run_seconds(session, 60)[-1]
    settings = session.engine.vent.settings
    assert not snap["controls"]["bag_mask"] and snap["controls"]["bag"] == bag
    assert snap["controls"]["apl"] == 25
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
    runs: dict[int, list] = {}
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
    cmd(session, "airway", mode="Mask")
    cmd(session, "vent", mode=mode or "VCV", rr=6, peep=5, p_insp=12, p_support=5)
    cmd(session, "vent_power", on=mode is not None)
    cmd(session, "run", running=True)
    runs: dict[int, list] = {}
    for snap in run_seconds(session, 35, tick_s=0.1):
        for run in snap["loop"]:
            runs.setdefault(run["breath"], []).extend(zip(run["paw"], run["flow"], run["volume"], strict=True))
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
    concentration = session.engine.pk["propofol"].state.c1
    for key, amount in [("missing", 50), ("Propofol 10 mg/mL", 50), ("propofol", 0), ("propofol", float("nan"))]:
        with pytest.raises(ValueError):
            cmd(session, "drug_bolus", key=key, amount=amount)
    assert session.engine.pk["propofol"].state.c1 == concentration
    session.replay_waves()
    assert parse(session.info())["medication_history"] == before
    for rate in range(55):
        cmd(session, "drug_rate", key="propofol", rate=rate)
    history = parse(session.info())["medication_history"]
    assert len(history) == 50
    assert history[-1]["text"] == "Propofol infusion 54 mcg/kg/min"


def test_blood_gas_reports_values_at_the_draw_after_analysis_and_survives_reload():
    session = WebSession({**PATIENT, "mode": "steady_state"})
    session.engine.config.dt = 0.25
    cmd(session, "run", running=True)
    run_seconds(session, 30, tick_s=10.0)
    state = session.engine.state
    drawn = cmd(session, "blood_gas")["blood_gas"]
    assert drawn["ready_at"] - drawn["drawn_at"] == pytest.approx(120)
    assert (drawn["ph"], drawn["lactate"], drawn["base_excess"]) == pytest.approx(
        (state.ph, state.lactate, state.base_excess), abs=0.01,
    )
    # Hyperventilation after the draw changes pH but not the reported sample.
    cmd(session, "vent", rr=30)
    run_seconds(session, 130, tick_s=10.0)
    assert session.engine.state.ph > drawn["ph"] + 0.02
    assert parse(session.info())["blood_gases"] == [drawn]


@pytest.mark.parametrize("kind,label", [
    ("crystalloid", "lactated Ringer's"), ("saline", "0.9% saline"), ("albumin", "5% albumin"), ("blood", "PRBC"),
])
def test_fluid_confirmation_follows_acceptance_and_delivery_continues_over_time(kind, label):
    session = WebSession(PATIENT)
    before = session.snapshot()["vitals"]
    receipt = cmd(session, "fluid", kind=kind, volume_ml=250)["fluid"]
    assert receipt == {"time": 0, "text": f"250 mL {label} started"}
    assert session.engine.pending_infusions[0].remaining_ml == 250
    after = session.snapshot()["vitals"]
    assert (after["fluid_in"], after["blood_in"]) == (before["fluid_in"], before["blood_in"])
    for volume in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            cmd(session, "fluid", kind=kind, volume_ml=volume)
    assert len(session.engine.pending_infusions) == 1
    cmd(session, "run", running=True)
    delivered = run_seconds(session, 2)[-1]["vitals"]
    assert 0 < delivered["blood_in" if kind == "blood" else "fluid_in"] < 250
    assert 0 < session.engine.pending_infusions[0].remaining_ml < 250


def test_blood_gas_electrolytes_freeze_at_draw_and_calcium_receipt_is_retained():
    session = WebSession(PATIENT)
    before = cmd(session, "blood_gas")["blood_gas"]
    receipt = cmd(session, "calcium_chloride", mg=1000)["medication"]
    assert receipt["text"] == "Calcium chloride 1000 mg given"
    after = cmd(session, "blood_gas")["blood_gas"]
    assert before["potassium"] == pytest.approx(4.0, abs=0.01)
    assert after["ionized_calcium"] > before["ionized_calcium"] + 0.2
    assert parse(session.info())["blood_gases"] == [before, after]
    assert parse(session.info())["medication_history"][-1] == receipt


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
    assert session.engine.infusion_rates["propofol"] == pytest.approx(100.0 * weight / 60000)
    assert session.engine.infusion_rates["remi"] == pytest.approx(0.15 * weight / 60)
    cmd(session, "run", running=True)
    snap = run_seconds(session, 1)[-1]
    for key, rate in (("propofol", 100.0), ("remi", 0.15)):
        assert snap["controls"]["drugs"][key]["rate"] == pytest.approx(rate)
        assert not snap["controls"]["drugs"][key]["is_tci"]
    assert session.engine.pk["propofol"].state.c1 > 0 and session.engine.pk["remi"].state.c1 > 0


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
    recorder = session.engine.recorder
    assert recorder is not None and recorder.file is not None
    recorder.file.close()
    (snap,) = run_seconds(session, 1)
    assert not snap["running"] and not snap["recording"]
    assert "incomplete" in snap["notice"]
    header, *rows = snap["download"]["csv"].strip().splitlines()
    assert header.startswith("time,") and len(rows) == 3
    assert not any(tmp_path.iterdir())


def test_session_continues_through_arrest_until_circulation_cannot_return(monkeypatch):
    session = WebSession(PATIENT)
    engine = session.engine
    engine.config.dt = 0.5
    steps_after_end = []
    step = engine.step

    def recording_step(dt):
        if engine.irreversible_arrest:
            steps_after_end.append(dt)
        step(dt)

    monkeypatch.setattr(engine, "step", recording_step)
    cmd(session, "run", running=True)
    cmd(session, "rhythm", name="Asystole")
    snaps = run_seconds(session, 60, tick_s=1.0)
    assert engine.state.cardiac_arrest
    assert snaps[-1]["running"] and not snaps[-1]["ended"]

    # Skip the minutes until ischemic injury is complete.
    engine.hemo.coronary.injury = 1.0
    snap = run_seconds(session, 20, tick_s=10.0)[-1]
    assert snap["ended"] and not snap["running"]
    assert snap["arrest_reason"] == "Asystole"
    # The final state is the one at the endpoint, even at 50x.
    assert not steps_after_end
    cmd(session, "run", running=True)
    assert not parse(session.advance(0.2))["running"]
    with pytest.raises(ValueError, match="ended"):
        cmd(session, "drug_bolus", key="epi", amount=100)


def test_open_session_arrest_finalizes_recording_and_commands_enforce_upper_bounds(tmp_path):
    session = WebSession({**PATIENT, "tci_enabled": True}, recordings_dir=str(tmp_path))
    cmd(session, "run", running=True)
    cmd(session, "record", active=True)
    session.step(0.1)
    session.engine.state.cardiac_arrest = True
    session.engine.state.arrest_reason = "Asystole"
    session.engine.hemo.coronary.injury = 1.0
    session.step(0.1)
    snap = session.snapshot()
    assert snap["ended"] and not snap["recording"] and snap["download"] is not None

    active = WebSession({**PATIENT, "tci_enabled": True})
    assert cmd(active, "drug_rate", key="fentanyl", rate=100.0)["medication"]["key"] == "fentanyl"
    for name, bad_args in (
        ("fgf", {"o2": 20, "air": 0, "n2o": 0}),
        ("drug_rate", {"key": "propofol", "rate": 2500}),
        ("drug_rate", {"key": "vaso", "rate": 5.0}),
        ("drug_target", {"key": "propofol", "target": 25}),
        ("drug_bolus", {"key": "propofol", "amount": 5000}),
        ("sugammadex", {"mg_per_kg": 40}),
        ("fluid", {"kind": "crystalloid", "volume_ml": 6000}),
        ("maintenance_fluid", {"ml_hr": 6000}),
        ("hemorrhage", {"active": True, "rate_ml_min": 6000}),
    ):
        with pytest.raises(ValueError):
            cmd(active, name, **bad_args)


class _ShockSucceeds:
    def random(self):
        return 0.0


def test_resuscitation_commands_reach_the_engine_and_warn_on_awake_cardioversion(monkeypatch):
    session = WebSession({**PATIENT, "mode": "awake"})
    engine = session.engine
    monkeypatch.setattr(engine, "_shock_rng", _ShockSucceeds())
    cmd(session, "run", running=True)
    cmd(session, "rhythm", name="Supraventricular tachycardia")
    assert "not charged" in cmd(session, "shock", synchronized=True)["shock"]["text"]
    cmd(session, "charge_defibrillator")
    assert session.snapshot()["controls"]["defibrillator"]["status"] == "charging"
    run_seconds(session, 6)
    assert session.snapshot()["controls"]["defibrillator"]["status"] == "ready"
    result = cmd(session, "shock", synchronized=True)
    assert result["shock"]["text"] == "Waiting for an R wave"
    snap = run_seconds(session, 1)[-1]
    assert snap["controls"]["defibrillator"]["last_shock"]["text"] == "Synchronized shock 200 J delivered"
    assert snap["controls"]["clinical_feedback"]["code"] == "shock_while_responsive"
    assert engine.hemo.rhythm_type.name == "SINUS"
    assert session.snapshot()["controls"]["ecg_rhythm"] == "Sinus rhythm"
    cmd(session, "rhythm", name="Ventricular fibrillation")
    cmd(session, "charge_defibrillator")
    run_seconds(session, 6)
    assert cmd(session, "shock", synchronized=True)["shock"]["text"] == "No R wave detected; no shock delivered"
    cmd(session, "compressions", active=True)
    started = engine.state.time
    snap = run_seconds(session, 30)[-1]
    assert snap["controls"]["compressions"] == {"active": True, "since": started}
    assert snap["vitals"]["art"][1] > 20
    # Unconscious in VF, the patient no longer prompts the warning.
    assert cmd(session, "shock", synchronized=False).get("feedback") is None
    assert session.snapshot()["controls"]["ecg_rhythm"] == "Sinus rhythm"


def test_lipid_bolus_and_infusion_reach_the_engine_and_round_trip_through_controls():
    session = WebSession({**PATIENT, "mode": "awake"})
    session.engine.config.dt = 0.5
    receipt = cmd(session, "fluid", kind="lipid", volume_ml=100)
    assert "20% lipid emulsion over 2.5 min" in receipt["fluid"]["text"]
    cmd(session, "lipid_rate", ml_min=17.5)
    controls = session.snapshot()["controls"]
    assert controls["lipid"] == {"rate": 17.5, "total_ml": 0}
    assert not controls["convulsions"]
    cmd(session, "run", running=True)
    controls = run_seconds(session, 151)[-1]["controls"]
    assert 143 < controls["lipid"]["total_ml"] < 145
    cmd(session, "lipid_rate", ml_min=0)
    before = session.engine.toxicity.total_lipid_ml
    run_seconds(session, 2)
    assert session.engine.toxicity.total_lipid_ml == before
    for rate in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cmd(session, "lipid_rate", ml_min=rate)


@pytest.mark.parametrize("mode", ["SIMV-VC", "SIMV-PC", "SIMV-VG"])
def test_simv_shows_untriggered_breaths_in_numerics_and_loops(mode):
    session = WebSession({**PATIENT, "mode": "awake"})
    engine = session.engine
    engine.resp.hcvr_slope_baseline = 0.0
    engine.resp.rr_0, engine.resp.vt_0 = 18.0, 250.0
    cmd(session, "airway", mode="Mask")
    cmd(session, "vent", mode=mode, rr=3, p_insp=8, p_support=5, trigger=20, t_insp=0.5)
    cmd(session, "vent_power", on=True)
    cmd(session, "run", running=True)
    runs: dict[int, list] = {}
    for snap in run_seconds(session, 58):
        for loop in snap["loop"]:
            runs.setdefault(loop["breath"], []).extend(zip(loop["paw"], loop["flow"], loop["volume"], strict=True))
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


def test_page_reads_only_snapshot_fields_and_commands_the_session_provides():
    """A renamed snapshot field or command fails here instead of silently in the browser."""
    assets = Path(web.__file__).parent / "web_assets"
    script = {path.name: path.read_text() for path in assets.glob("*.js")}
    session = WebSession({**PATIENT, "scenario_id": SCENARIO_REGISTRY[0].id})
    snap = parse(session.advance(0.0))

    def read(name, receiver):
        return set(re.findall(rf"\b{receiver}\.([a-z_][a-z0-9_]*)", script[name]))

    assert set().union(*(read(name, "snap") for name in ("app.js", "audio.js", "monitor.js", "trends.js"))) <= set(snap)
    assert read("monitor.js", "v") <= set(snap["vitals"])
    assert read("controls.js", "c") <= set(snap["controls"])
    commands = set(re.findall(r'\b(?:command|send)\(\s*"([a-z_]+)"', "".join(script.values())))
    commands |= set(re.findall(r'\bperformAction\([^,]+,\s*"([a-z_]+)"', "".join(script.values())))
    assert len(commands) > 20
    assert {name for name in commands if not hasattr(session, f"_cmd_{name}")} == set()
    page = (assets / "index.html").read_text()
    tabs = set(re.findall(r'data-tab="([^"]+)"', page))
    for spec in SCENARIO_REGISTRY:
        for step in spec.builder():
            assert step.target_tab is None or step.target_tab in tabs


def test_step_accumulator_executes_exact_sim_steps_across_float_boundaries():
    """Binary-float accumulator residuals (e.g. 0.016 * 50 = 16 * 0.05) do not drop a step."""
    session = WebSession(PATIENT)
    session.engine.config.dt = 0.05
    cmd(session, "run", running=True)
    cmd(session, "speed", value=50)
    session.step(0.016)
    assert session.engine.state.time == pytest.approx(0.80, abs=1e-12)


def test_first_spontaneous_breath_streams_loop_points_during_inspiration():
    """Breath 1 on a facemask streams inspiratory loop points before exhalation begins."""
    session = WebSession({**PATIENT, "mode": "awake"})
    cmd(session, "airway", mode="Mask")
    cmd(session, "run", running=True)
    insp_streamed = False
    for _ in range(30):
        snap = parse(session.advance(0.1))
        for run in snap["loop"]:
            if any(f > 2.0 for f in run["flow"]) and all(f >= -0.5 for f in run["flow"]):
                insp_streamed = True
                break
        if insp_streamed:
            break
    assert insp_streamed


def test_web_apl_and_fluid_warmer_command_round_trips():
    session = WebSession({**PATIENT, "mode": "awake"})
    assert session.snapshot()["controls"]["apl"] == 0
    assert session.snapshot()["controls"]["fluid_warmer"] is False
    assert session.snapshot()["controls"]["reservoir"]["state"] == "normal"
    assert session.snapshot()["controls"]["o2_flush"] is False
    cmd(session, "apl", cmh2o=0.5)
    cmd(session, "fluid_warmer", enabled=True)
    snap = session.snapshot()["controls"]
    assert snap["apl"] == 0.5
    assert snap["fluid_warmer"] is True

    cmd(session, "airway", mode="Mask")
    cmd(session, "apl", cmh2o=20.0)
    cmd(session, "run", running=True)
    session.engine.circuit.reservoir_volume = 0.0
    assert session.snapshot()["controls"]["reservoir"]["state"] == "empty"
    cmd(session, "o2_flush")
    assert session.snapshot()["controls"]["o2_flush"] is True
    for _ in range(40):
        session.engine.step(0.05)
    assert session.snapshot()["controls"]["reservoir"]["fill"] > 0.3
    assert session.snapshot()["controls"]["o2_flush"] is False

    breaths_before = session.engine.vent.breath_count
    cmd(session, "bag_breath")
    for _ in range(120):
        session.engine.step(0.05)
    assert session.engine.vent.breath_count >= breaths_before + 1

    with pytest.raises(ValueError, match="calcium chloride dose"):
        cmd(session, "calcium_chloride", mg=True)


def test_vfib_blanks_st_and_hr_in_web_snapshot_while_asystole_shows_zero_hr():
    """Web JSON blanks ST and HR on VFIB while keeping state floats and showing 0 HR in asystole."""
    session = WebSession({"mode": "steady_state"})
    session.engine.set_rhythm("ASYSTOLE")
    assert session.snapshot()["vitals"]["hr"] == 0.0
    session.engine.start()
    session.engine.set_rhythm(RhythmType.VFIB)
    session.engine.step(0.5)
    assert session.engine.state.st_depression_mv == 0.0
    snap = session.snapshot()["vitals"]
    assert snap["st_depression_mv"] is None
    assert snap["hr"] is None
    session.engine.set_rhythm("ASYSTOLE")
    session.engine.step(0.5)
    assert session.snapshot()["vitals"]["hr"] == 0.0


def test_disconnect_emits_loop_reset_and_vaporizer_refill_round_trips():
    """Disconnecting the airway resets retained spirometry loops and refill_vaporizer restores level."""
    session = WebSession({**PATIENT, "mode": "steady_state"})
    cmd(session, "run", running=True)
    run_seconds(session, 6)
    _ = session.snapshot()
    cmd(session, "airway", mode="None")
    disc = session.snapshot()
    assert disc["loop"] == [{"reset": True}]

    session.engine.vaporizer.state.level = 12.5
    assert session.snapshot()["controls"]["vaporizer_level"] == pytest.approx(12.5)
    cmd(session, "refill_vaporizer")
    assert session.snapshot()["controls"]["vaporizer_level"] == pytest.approx(250.0)


def test_extracorporeal_commands_and_multisite_blood_gas_round_trip():
    """VV ECMO, VA ECMO, CPB controls, ARDS shunt, and multi-site blood gases work via JSON commands."""
    session = WebSession({**PATIENT, "mode": "steady_state", "arterial_line_enabled": True})
    cmd(session, "run", running=True)
    snap0 = parse(session.advance(0.0))
    assert snap0["controls"]["extracorporeal"]["active"] is False
    assert snap0["controls"]["extracorporeal"]["mode"] == "none"

    cmd(session, "ards_shunt", percent=55)
    assert parse(session.advance(0.0))["controls"]["ards_shunt"] == pytest.approx(55.0)

    # Start peripheral femoral VA ECMO and verify site-specific ABGs
    cmd(
        session,
        "extracorporeal_start",
        mode="va_ecmo",
        blood_flow=3.5,
        sweep_gas=4.0,
        cannulation="peripheral_fem",
    )
    run_seconds(session, 15)
    ec_va = parse(session.advance(0.0))["controls"]["extracorporeal"]
    assert ec_va["active"] is True
    assert ec_va["mode"] == "va_ecmo"
    assert ec_va["cannulation"] == "peripheral_fem"
    assert ec_va["flow"] == pytest.approx(3.5, abs=0.05)
    assert 10.0 <= ec_va["pre_so2"] <= 100.0

    abg_rad = cmd(session, "blood_gas", site="right_radial")["blood_gas"]
    abg_fem = cmd(session, "blood_gas", site="femoral")["blood_gas"]
    abg_pre = cmd(session, "blood_gas", site="pre_oxy")["blood_gas"]
    assert abg_rad["site"] == "right_radial"
    assert abg_fem["site"] == "femoral"
    assert abg_pre["site"] == "pre_oxy"
    assert abg_fem["pao2"] > abg_rad["pao2"]
    assert 15.0 <= abg_pre["hco3"] <= 32.0

    # Switch to CPB, test cross-clamp, reservoir fluid, and clean stop
    cmd(session, "extracorporeal_start", mode="cpb", blood_flow=4.5, sweep_gas=3.0)
    cmd(session, "extracorporeal_settings", cross_clamp=True, cardioplegia=True, sweep_sevo=1.5, blood_gas_strategy="ph_stat")
    run_seconds(session, 10)
    snap_cpb = parse(session.advance(0.0))
    ec_cpb = snap_cpb["controls"]["extracorporeal"]
    assert ec_cpb["mode"] == "cpb"
    assert ec_cpb["cross_clamp"] is True
    assert ec_cpb["cardioplegia"] is True
    assert ec_cpb["sevo"] == pytest.approx(1.5)
    assert ec_cpb["blood_gas_strategy"] == "ph_stat"
    assert snap_cpb["vitals"]["art"][0] is None and snap_cpb["vitals"]["art"][2] > 40.0

    res_before = ec_cpb["reservoir_ml"]
    assert "upper_sao2" not in ec_cpb and "lower_sao2" not in ec_cpb
    receipt = cmd(session, "cpb_reservoir_fluid", volume_ml=500, kind="crystalloid")
    assert "CPB reservoir" in receipt["fluid"]["text"]
    assert parse(session.advance(0.0))["controls"]["extracorporeal"]["reservoir_ml"] > res_before + 350.0

    # Reservoir cap reports actual added mL and full message when reservoir is already at max
    session.engine.extracorporeal.state.reservoir_volume_ml = session.engine.extracorporeal.config.cpb_reservoir_max_ml - 100.0
    partial = cmd(session, "cpb_reservoir_fluid", volume_ml=500, kind="crystalloid")
    assert "100 mL" in partial["fluid"]["text"]
    full = cmd(session, "cpb_reservoir_fluid", volume_ml=500, kind="crystalloid")
    assert "full" in full["fluid"]["text"].lower()

    with pytest.raises(ValueError, match="Unknown extracorporeal"):
        cmd(session, "extracorporeal_settings", unknown_field=1)
    with pytest.raises(ValueError, match="Unknown extracorporeal mode"):
        cmd(session, "extracorporeal_start", mode="impella")

    cmd(session, "extracorporeal_stop", reinfuse=True)
    assert parse(session.advance(0.0))["controls"]["extracorporeal"]["active"] is False


def test_heparin_protamine_act_and_bypass_warnings_round_trip():
    """Heparin, protamine, ACT measurement, protamine reaction event, and CPB safety warnings work via JSON."""
    # Starting CPB without heparin emits non-blocking low-ACT warning, cleared by starting VV ECMO
    unheparinized = WebSession({**PATIENT, "mode": "steady_state", "arterial_line_enabled": True})
    cmd(unheparinized, "run", running=True)
    act0 = cmd(unheparinized, "act")["act"]
    assert act0["act_s"] == pytest.approx(120.0, abs=1.0)
    assert act0["ready_at"] == pytest.approx(act0["drawn_at"] + 60.0)
    cpb_res = cmd(unheparinized, "extracorporeal_start", mode="cpb", blood_flow=4.5, sweep_gas=3.0)
    assert cpb_res["feedback"]["code"] == "cpb_low_act"
    vv_res = cmd(unheparinized, "extracorporeal_start", mode="vv_ecmo", blood_flow=4.0, sweep_gas=4.0)
    assert vv_res == {"feedback": None}
    assert parse(unheparinized.advance(0.0)).get("feedback") is None

    # Full heparinization -> ACT >= 480 s -> CPB start clears warning
    session = WebSession({**PATIENT, "mode": "steady_state", "arterial_line_enabled": True})
    cmd(session, "run", running=True)
    cmd(session, "drug_bolus", key="heparin", amount=21000)
    run_seconds(session, 20)
    assert cmd(session, "act")["act"]["act_s"] >= 480.0
    cpb_ok = cmd(session, "extracorporeal_start", mode="cpb", blood_flow=4.5, sweep_gas=3.0)
    assert cpb_ok is None

    # Giving protamine during active CPB emits protamine_on_bypass warning
    prot_warn = cmd(session, "drug_bolus", key="protamine", amount=50)
    assert prot_warn["feedback"]["code"] == "protamine_on_bypass"
    cmd(session, "extracorporeal_stop", reinfuse=True)

    # Slow protamine reversal normalizes ACT <= 150 s
    cmd(session, "drug_rate", key="protamine", rate=25)
    run_seconds(session, 420, tick_s=5.0)
    cmd(session, "drug_rate", key="protamine", rate=0)
    run_seconds(session, 60, tick_s=5.0)
    assert cmd(session, "act")["act"]["act_s"] <= 150.0

    # Protamine reaction event toggle
    cmd(session, "protamine_reaction", active=True)
    assert parse(session.advance(0.0))["controls"]["protamine_reaction"] is True
    cmd(session, "protamine_reaction", active=False)
    assert parse(session.advance(0.0))["controls"]["protamine_reaction"] is False


def test_glucose_insulin_dextrose_and_poc_meter_round_trip():
    """Type 2 diabetes, baseline glucose, dextrose fluids, regular insulin, POC glucose, and warnings round-trip via JSON."""
    cat = parse(web.catalog())
    assert cat["ranges"]["baseline_glucose"] == [40.0, 400.0]

    session = WebSession({**PATIENT, "mode": "awake", "type2_diabetes": True, "baseline_glucose": 210.0})
    info = parse(session.info())
    assert info["patient"]["type2_diabetes"] is True
    assert info["patient"]["baseline_glucose"] == pytest.approx(210.0)
    assert info["glucoses"] == []
    assert any(
        drug["key"] == "insulin"
        and drug["rate_unit"] == "U/hr"
        and drug["name"].startswith(drug["generic_name"])
        for drug in info["drugs"]
    )

    cmd(session, "run", running=True)
    poc0 = cmd(session, "glucose")["glucose"]
    assert poc0["glucose"] == pytest.approx(210.0, abs=1.0)
    assert poc0["ready_at"] == pytest.approx(poc0["drawn_at"] + 10.0)
    abg0 = cmd(session, "blood_gas")["blood_gas"]
    assert abg0["glucose"] == pytest.approx(210.0, abs=1.0)

    # Regular insulin bolus and infusion lower glucose without warning when hyperglycemic.
    bolus_res = cmd(session, "drug_bolus", key="insulin", amount=5.0)
    assert bolus_res.get("feedback") is None
    assert "Insulin 5 U given" in bolus_res["medication"]["text"]
    rate_res = cmd(session, "drug_rate", key="insulin", rate=4.0)
    assert rate_res.get("feedback") is None
    assert "4 U/hr" in rate_res["medication"]["text"]

    # D50W and D10W route through Medications (returning both fluid and medication receipts with gram doses),
    # while D5W is a Circulation fluid; bolus caps are enforced.
    for kind, vol, label, grams in (("d50w", 50, "D50W", "25 g"), ("d10w", 100, "D10W", "10 g"), ("d5w", 250, "D5W", None)):
        res = cmd(session, "fluid", kind=kind, volume_ml=vol)
        assert label in res["fluid"]["text"]
        if kind in ("d50w", "d10w"):
            assert res["medication"]["key"] == kind
            assert label in res["medication"]["text"] and grams in res["medication"]["text"]
        else:
            assert "medication" not in res
    for kind, bad_vol in (("d50w", 600), ("d50w", 0), ("d10w", 2500), ("d10w", -10)):
        with pytest.raises(ValueError, match=r"fluid volume|positive"):
            cmd(session, "fluid", kind=kind, volume_ml=bad_vol)
    run_seconds(session, 15)
    assert cmd(session, "glucose")["glucose"]["glucose"] > 220.0

    # Sodium bicarbonate command logs a medication receipt, enforces bounds, and warns in alkalemia/hypernatremia.
    bicarb_res = cmd(session, "sodium_bicarbonate", meq=50)
    assert bicarb_res.get("feedback") is None
    assert bicarb_res["medication"]["key"] == "sodium_bicarbonate"
    assert "Sodium bicarbonate 50 mEq given" in bicarb_res["medication"]["text"]
    for bad_meq in (0, -50, 501, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cmd(session, "sodium_bicarbonate", meq=bad_meq)
    session.engine.state.ph = 7.56
    alk_warn = cmd(session, "sodium_bicarbonate", meq=50)
    assert alk_warn["feedback"]["code"] == "bicarbonate_in_alkalemia"
    assert alk_warn["feedback"]["level"] == "warning"
    assert alk_warn["medication"]["key"] == "sodium_bicarbonate"
    # Subsequent rescue calcium chloride clears the prior warning banner cleanly.
    ca_res = cmd(session, "calcium_chloride", mg=1000)
    assert ca_res["feedback"] is None
    assert ca_res["medication"]["key"] == "calcium_chloride"
    with pytest.raises(ValueError, match="positive"):
        cmd(session, "calcium_chloride", mg=0)
    session.engine.state.ph = 7.38
    session.engine.state.sodium = 158.0
    na_warn = cmd(session, "sodium_bicarbonate", meq=50)
    assert na_warn["feedback"]["code"] == "bicarbonate_in_hypernatremia"
    assert na_warn["feedback"]["level"] == "warning"
    # Reload info retains D50W, D10W, and sodium bicarbonate in medication history and excludes D5W.
    hist_keys = [entry["key"] for entry in parse(session.info())["medication_history"]]
    assert {"d50w", "d10w", "sodium_bicarbonate", "calcium_chloride"} <= set(hist_keys)
    assert "d5w" not in hist_keys

    # Giving insulin during hypoglycemia triggers a non-blocking safety warning cleared by D50W rescue.
    hypo = WebSession({**PATIENT, "mode": "awake", "baseline_glucose": 55.0})
    warn = cmd(hypo, "drug_bolus", key="insulin", amount=2.0)
    assert warn["feedback"]["code"] == "insulin_in_hypoglycemia"
    assert warn["feedback"]["level"] == "warning"
    rescue = cmd(hypo, "fluid", kind="d50w", volume_ml=50)
    assert rescue["feedback"] is None
    assert rescue["medication"]["key"] == "d50w"


def test_airway_crisis_thoracic_embolism_and_neuraxial_commands_round_trip():
    """Airway, thoracic, embolism, and neuraxial crisis catalog entries, open-sim controls, and case locks."""
    cat = parse(web.catalog())
    scenarios_by_id = {entry["id"]: entry for entry in cat["scenarios"]}
    for sid in (
        "failed_laryngoscopy",
        "esophageal_intubation",
        "tension_pneumothorax",
        "venous_air_embolism",
        "pulmonary_embolism",
        "high_spinal",
    ):
        assert scenarios_by_id[sid]["category"] == "crisis"

    # Open simulation: difficult airway tools, SGA, eFONA, and esophageal intubation triggers.
    session = WebSession({**PATIENT, "mode": "steady_state"})
    cmd(session, "run", running=True)
    cmd(session, "difficult_airway", profile="anterior_dl")
    cmd(session, "airway", mode="Mask")
    cmd(session, "airway_technique", technique="dl")
    fail_res = cmd(session, "airway", mode="ETT")
    assert fail_res["feedback"]["code"] == "failed_laryngoscopy"
    assert fail_res["feedback"]["level"] == "blocked"
    cmd(session, "call_airway_help")
    cmd(session, "oral_airway", placed=True)
    cmd(session, "two_hand_mask", active=True)
    ac = parse(session.advance(0.0))["controls"]["airway_crisis"]
    assert ac["difficult_profile"] == "anterior_dl"
    assert ac["help_called"] is True and ac["oral_airway"] is True and ac["two_hand_mask"] is True
    assert ac["attempts"] == 1 and ac["last_view"] == "Grade III"

    cmd(session, "airway", mode="SGA")
    assert parse(session.advance(0.0))["controls"]["airway"] == "SGA"
    cmd(session, "perform_fona")
    assert parse(session.advance(0.0))["controls"]["airway"] == "FONA"

    cmd(session, "difficult_airway", profile="none")
    cmd(session, "airway", mode="Mask")
    cmd(session, "esophageal_intubation", armed=True)
    assert parse(session.advance(0.0))["controls"]["airway_crisis"]["esophageal_armed"] is True
    cmd(session, "airway", mode="ETT", technique="dl")
    assert parse(session.advance(0.0))["controls"]["airway_crisis"]["esophageal_active"] is True
    cmd(session, "esophageal_intubation", active=False)
    assert parse(session.advance(0.0))["controls"]["airway_crisis"]["esophageal_active"] is False

    # Open simulation: pneumothorax, PE, and VAE events + crisis procedures.
    cmd(session, "pneumothorax", active=True)
    assert parse(session.advance(0.0))["controls"]["pneumothorax"]["active"] is True
    ptx_res = cmd(session, "chest_decompress")
    assert "Chest decompression" in ptx_res["procedure"]["text"]
    assert parse(session.advance(0.0))["controls"]["pneumothorax"]["decompressed"] is True

    cmd(session, "pulmonary_embolism", active=True)
    assert parse(session.advance(0.0))["controls"]["embolism"]["pe_active"] is True
    pe_res = cmd(session, "pe_thrombolysis")
    assert "thrombolysis" in pe_res["procedure"]["text"].lower()
    assert parse(session.advance(0.0))["controls"]["embolism"]["thrombolysis"] is True
    cmd(session, "pulmonary_embolism", active=False)

    cmd(session, "venous_air_embolism", active=True)
    assert parse(session.advance(0.0))["controls"]["embolism"]["vae_active"] is True
    vae_res = cmd(session, "vae_aspirate")
    assert "CVC" in vae_res["procedure"]["text"]
    cmd(session, "venous_air_embolism", active=False)

    # Open simulation: general neuraxial anesthesia controls (spinal, epidural bolus/infusion, level check).
    awake = WebSession({**PATIENT, "mode": "awake"})
    cmd(awake, "run", running=True)
    sp = cmd(awake, "neuraxial_spinal", drug="bupivacaine", dose_mg=12.5, baricity="hyperbaric")
    assert "Spinal hyperbaric bupivacaine 12.5 mg" in sp["medication"]["text"]
    run_seconds(awake, 180, tick_s=5.0)
    chk = cmd(awake, "neuraxial_check_level")["neuraxial_check"]
    assert chk["sensory_label"] != "None"
    assert parse(awake.advance(0.0))["controls"]["neuraxial"]["last_check"]["sensory_label"] == chk["sensory_label"]
    cmd(awake, "neuraxial_clear")
    assert parse(awake.advance(0.0))["controls"]["neuraxial"]["active"] is False

    cmd(awake, "neuraxial_epidural_catheter", placed=True)
    eb = cmd(awake, "neuraxial_epidural_bolus", drug="bupivacaine", volume_ml=12, concentration_pct=0.5, with_epinephrine=True)
    assert "1:200k epi" in eb["medication"]["text"]
    er = cmd(awake, "neuraxial_epidural_rate", ml_hr=8, drug="bupivacaine", concentration_pct=0.125)
    assert "8 mL/hr" in er["medication"]["text"]
    assert parse(awake.advance(0.0))["controls"]["neuraxial"]["epidural_rate"] == pytest.approx(8.0)
    cmd(awake, "high_spinal", active=True)
    assert parse(awake.advance(0.0))["controls"]["neuraxial"]["mode"] == "high_spinal"

    with pytest.raises(ValueError, match="technique"):
        cmd(session, "airway_technique", technique="blind_wand")
    # Invalid inline airway technique does not mutate stored technique.
    cmd(session, "airway_technique", technique="vl")
    with pytest.raises(ValueError, match="technique"):
        cmd(session, "airway", mode="ETT", technique="blind_wand")
    assert parse(session.advance(0.0))["controls"]["airway_crisis"]["technique"] == "vl"

    cmd(awake, "neuraxial_clear")
    with pytest.raises(ValueError, match="neuraxial drug"):
        cmd(awake, "neuraxial_spinal", drug="tetracaine", dose_mg=10)
    # Numeric string arguments validate before engine mutation.
    sp_str = cmd(awake, "neuraxial_spinal", drug="bupivacaine", dose_mg="10", baricity="isobaric")
    assert "10 mg" in sp_str["medication"]["text"]
    with pytest.raises(ValueError):
        cmd(awake, "neuraxial_epidural_rate", ml_hr="100", drug="bupivacaine")
    assert parse(awake.advance(0.0))["controls"]["neuraxial"]["epidural_rate"] == 0.0

    # Instructor event commands (including neuraxial_clear) are blocked during active cases.
    case = WebSession({"scenario_id": "failed_laryngoscopy"})
    for ev_name, ev_args in (
        ("difficult_airway", {"profile": "none"}),
        ("esophageal_intubation", {"active": True}),
        ("pneumothorax", {"active": True}),
        ("pulmonary_embolism", {"active": True}),
        ("venous_air_embolism", {"active": True}),
        ("high_spinal", {"active": True}),
        ("neuraxial_clear", {}),
    ):
        with pytest.raises(ValueError, match="Case events are controlled by the case"):
            cmd(case, ev_name, **ev_args)

    # NIBP-only stabilization steps require a fresh blood pressure reading instead of leaking hidden true MAP.
    for sid, step_id in (
        ("tension_pneumothorax", "STABILIZE_PTX"),
        ("venous_air_embolism", "STABILIZE_VAE"),
        ("pulmonary_embolism", "STABILIZE_PE"),
        ("high_spinal", "STABILIZE_SPINAL"),
    ):
        nibp_case = WebSession({"scenario_id": sid, "arterial_line_enabled": False})
        eng = nibp_case.engine
        assert nibp_case.scenario is not None
        step = next(s for s in nibp_case.scenario.steps if s.id == step_id)
        eng.state.time = 200.0
        eng.state.nibp_timestamp = 100.0  # > 60 s stale
        eng.state.map = eng.state.nibp_map = 80.0
        eng.state.sao2 = eng.state.display_spo2 = 99.0
        eng.state.spo2_signal_valid = True
        eng.state.etco2 = eng.state.display_etco2 = 35.0
        eng.state.etco2_signal_valid = True
        eng.state.hr = eng.state.display_hr = 75.0
        eng.set_airway_mode("ETT")
        eng.set_vent_power(True)
        if sid == "tension_pneumothorax":
            eng.start_pneumothorax()
            eng.decompress_pneumothorax()
        elif sid == "venous_air_embolism":
            eng.start_venous_air_embolism()
            eng.stop_venous_air_embolism()
        elif sid == "pulmonary_embolism":
            eng.start_pulmonary_embolism()
            eng.give_thrombolysis()
        met, msg = step.check_requirements(eng)
        assert not met and "Measure blood pressure again" in msg


def test_audit_web_esophageal_blanking_mask_zero_cpap_ecmo_and_no_duplicate_t0():
    """Verify esophageal gas blanking, Mask seal=0% disconnect projection, CPAP airway removal,
    extracorporeal_start fio2_percent/VA bypass_alarm_mode, and no duplicate t=0 trend after info()."""
    # 1. info() followed by snapshot() does not emit a duplicate t = 0 trend point.
    s0 = WebSession({**PATIENT, "mode": "steady_state"})
    info0 = parse(s0.info())
    assert [pt["time"] for pt in info0["trends"]] == [0]
    assert s0.snapshot()["trends"] == []

    # 2. Esophageal intubation blanks EtO2, EtSevo, EtN2O, and EtMAC in vitals.
    esoph = WebSession({**PATIENT, "mode": "steady_state"})
    esoph.engine.set_esophageal_intubation(True, gastric_co2_mmhg=18.0)
    run_seconds(esoph, 10)
    v_e = esoph.snapshot()["vitals"]
    assert v_e["eto2"] is None and v_e["et_sevo"] is None and v_e["et_n2o"] is None and v_e["et_mac"] is None

    # 3. Mask seal 0% projects disconnected spirometry (None) and impedance RR.
    m0 = WebSession({**PATIENT, "mode": "awake"})
    cmd(m0, "airway", mode="Mask")
    cmd(m0, "mask_seal", percent=0)
    v_m0 = m0.snapshot()["vitals"]
    assert v_m0["ppeak"] is None and v_m0["vte"] is None and v_m0["mv"] is None
    assert v_m0["rr_source"] == "impedance"

    # 4. Removing an airway on spontaneous CPAP does not emit a false ventilation_stopped warning.
    cpap = WebSession({**PATIENT, "mode": "steady_state"})
    cmd(cpap, "vent", mode="CPAP", peep=5)
    res_rm = cmd(cpap, "airway", mode="None")
    assert res_rm.get("feedback") is None or res_rm["feedback"]["code"] != "ventilation_stopped"

    # 5. extracorporeal_start accepts fio2_percent and VA ECMO bypass_alarm_mode.
    ec = WebSession({**PATIENT, "mode": "steady_state"})
    cmd(ec, "extracorporeal_start", mode="va_ecmo", blood_flow=4.0, sweep_gas=3.5, fio2_percent=80, bypass_alarm_mode=True)
    assert ec.engine.extracorporeal.state.sweep_fio2 == pytest.approx(0.80)
    assert ec.engine.extracorporeal.state.bypass_alarm_mode is True
