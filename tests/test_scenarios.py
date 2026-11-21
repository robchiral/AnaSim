import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from anasim.core.clinical_actions import check_airway_removal
from anasim.core.state import SimulationConfig
from anasim.monitors.neuromuscular import TwitchReading
from anasim.scenarios import (
    SCENARIO_REGISTRY,
    create_anaphylaxis_scenario,
    create_cardiac_arrest,
    create_emergence,
    create_hemorrhage_response,
    create_induction_balanced,
    create_malignant_hyperthermia,
    create_oxygen_supply_failure,
    create_sepsis_response,
)
from anasim.scenarios.base import (
    CrisisStartMAP,
    format_unmet_int,
    neuromuscular_response,
    patient_stable,
    require_crisis_resolved_with_map,
    require_hypotension_or_treatment,
    require_map_at_least,
    require_vasopressor_if_hypotensive,
)
from anasim.scenarios.emergence import _require_normocapnia
from anasim.scenarios.malignant_hyperthermia import _require_etco2_normal
from anasim.web import WebSession

# Browser commands for each objective; TCI targets become manual rates when disabled.
SCENARIO_WALKTHROUGHS: dict[str, dict[str, list[tuple[str, dict[str, Any]]]]] = {
    "malignant_hyperthermia": {
        "STOP_TRIGGERS": [
            ("vaporizer", {"percent": 0}),
            ("fgf", {"o2": 10, "air": 0, "n2o": 0}),
            ("vent", {"rr": 28}),
            ("drug_rate", {"key": "propofol", "rate": 120}),
        ],
        "DANTROLENE": [("drug_bolus", {"key": "dantrolene", "amount": 175})],
        "STABILIZE": [
            ("bair_hugger", {"target_c": 0}),
            ("vent", {"rr": 12}),
        ],
    },
    "hemorrhage_response": {
        "GIVE_FLUIDS": [("fluid", {"kind": "crystalloid", "volume_ml": 500})],
        "START_VASOPRESSOR": [("drug_rate", {"key": "nore", "rate": 12})],
        "STOP_BLEEDING": [("case_action", {"action": "hemostasis"})],
    },
    "sepsis_response": {
        "GIVE_FLUIDS": [("fluid", {"kind": "crystalloid", "volume_ml": 1000})],
        "START_VASOPRESSOR": [("drug_rate", {"key": "nore", "rate": 15})],
        "SOURCE_CONTROL": [("case_action", {"action": "source_control"})],
    },
    "anaphylaxis_response": {
        "EPINEPHRINE": [
            ("drug_bolus", {"key": "epi", "amount": 100}),
            ("fgf", {"o2": 10, "air": 0, "n2o": 0}),
        ],
        "FLUIDS": [("fluid", {"kind": "crystalloid", "volume_ml": 1000})],
        "STABILIZE": [
            ("case_action", {"action": "stop_trigger"}),
            ("drug_rate", {"key": "epi", "rate": 6}),
        ],
    },
    "induction_tiva": {
        "APPLY_MASK": [("airway", {"mode": "Mask"})],
        "SET_FGF_PREOX": [("fgf", {"o2": 10, "air": 0, "n2o": 0})],
        "START_ANALGESIA": [("drug_target", {"key": "remi", "target": 4})],
        "INDUCE": [
            ("drug_bolus", {"key": "propofol", "amount": 175}),
            ("drug_target", {"key": "propofol", "target": 4}),
        ],
        "MASK_VENTILATE": [("bag_mask", {"active": True})],
        "GIVE_NMB": [("drug_bolus", {"key": "roc", "amount": 42})],
        "INTUBATE": [("airway", {"mode": "ETT"})],
        "CONFIRM_ETT": [("vent_power", {"on": True})],
        "MAINTENANCE": [
            ("fgf", {"o2": 2, "air": 0, "n2o": 0}),
            ("drug_target", {"key": "propofol", "target": 3}),
            ("drug_bolus", {"key": "phenyl", "amount": 100}),
            ("drug_rate", {"key": "nore", "rate": 6}),
        ],
    },
    "induction_balanced": {
        "APPLY_MASK": [("airway", {"mode": "Mask"})],
        "SET_FGF_PREOX": [("fgf", {"o2": 10, "air": 0, "n2o": 0})],
        "GIVE_OPIOID": [("drug_bolus", {"key": "fentanyl", "amount": 100})],
        "INDUCE": [("drug_bolus", {"key": "propofol", "amount": 150})],
        "MASK_VENTILATE": [("bag_mask", {"active": True})],
        "GIVE_NMB": [("drug_bolus", {"key": "roc", "amount": 42})],
        "START_SEVO": [("vaporizer", {"percent": 3})],
        "INTUBATE": [("airway", {"mode": "ETT"})],
        "CONFIRM_ETT": [("vent_power", {"on": True})],
        "MAINTENANCE": [
            ("vaporizer", {"percent": 1.6}),
            ("fgf", {"o2": 1, "air": 1, "n2o": 0}),
            # Unstimulated sevoflurane before incision needs pressor support.
            ("drug_rate", {"key": "phenyl", "rate": 40}),
        ],
    },
    "emergence_tiva": {
        "ASSESS": [
            ("drug_bolus", {"key": "phenyl", "amount": 100}),
            ("vent", {"rr": 9}),
        ],
        "ASSESS_BLOCK": [("sugammadex", {"mg_per_kg": 2})],
        "STOP_AGENTS": [
            ("drug_rate", {"key": "propofol", "rate": 0}),
            ("drug_rate", {"key": "remi", "rate": 0}),
        ],
        "EXTUBATION_READINESS": [("vent_power", {"on": False})],
        "EXTUBATE": [("airway", {"mode": "Mask"})],
        "RECOVERY": [("fgf", {"o2": 5, "air": 0, "n2o": 0})],
    },
    "emergence_balanced": {
        "ASSESS": [
            ("drug_bolus", {"key": "phenyl", "amount": 100}),
            ("vent", {"rr": 9}),
        ],
        "ASSESS_BLOCK": [("sugammadex", {"mg_per_kg": 2})],
        "STOP_AGENTS": [
            ("vaporizer", {"percent": 0}),
            ("drug_rate", {"key": "remi", "rate": 0}),
            ("fgf", {"o2": 10, "air": 0, "n2o": 0}),
        ],
        "EXTUBATION_READINESS": [("vent_power", {"on": False})],
        "EXTUBATE": [("airway", {"mode": "Mask"})],
        "RECOVERY": [("fgf", {"o2": 5, "air": 0, "n2o": 0})],
    },
    "cardiac_arrest": {
        "START_COMPRESSIONS": [
            ("compressions", {"active": True}),
            ("fgf", {"o2": 10, "air": 0, "n2o": 0}),
        ],
        "DEFIBRILLATE": [("shock", {"synchronized": False})],
        "POST_ARREST": [("compressions", {"active": False})],
    },
    "oxygen_supply_failure": {
        "CONNECT_BACKUP_OXYGEN": [
            ("case_action", {"action": "backup_oxygen"}),
            ("fgf", {"o2": 10, "air": 0, "n2o": 0}),
        ],
    },
}


def _step(scenario, step_id):
    return next(step for step in scenario.steps if step.id == step_id)


def _early_extubation(engine):
    # Meet the awake criteria so only the airway action is missing.
    engine.state.display_bis = 90.0
    engine.resp.state.rr = 12.0
    engine.resp.state.vt = 500.0
    engine.state.rr = 12.0
    engine.state.vt = 500.0
    engine.resp.state.apnea = False
    engine.state.tof = engine.state.twitch = 100.0
    engine.state.display_spo2 = 98.0
    engine.state.spo2_signal_valid = True
    engine.set_vent_power(False)
    engine.set_airway_mode("Mask")
    engine.nmb.reading = TwitchReading(
        engine.state.time, "tof", (1.0, 1.0, 1.0, 1.0), 4, 100.0, 100.0, False,
    )


@pytest.mark.parametrize("builder,event", [
    (create_hemorrhage_response, "hemorrhage"),
    (create_sepsis_response, "sepsis"),
])
def test_vasopressor_objective_tracks_pressure_and_current_support(engine_factory, builder, event):
    # These crises start during surgery, as in the browser.
    engine = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="balanced",
                                                    disturbance_profile="stim_sustained_surgery"), start=True)
    step = _step(builder(), "START_VASOPRESSOR")
    # Recovered pressure does not justify another vasopressor.
    assert step.check_requirements(engine)[0]

    getattr(engine, f"start_{event}")()
    for _ in range(300):
        engine.step(2.0)
        if engine.state.monitored_blood_pressure(True)[2] < 65:
            break
    assert engine.state.monitored_blood_pressure(True)[2] < 65
    engine.set_drug_rate("nore", 5.0)

    assert step.check_requirements(engine)[0]
    if event == "hemorrhage":
        engine.set_drug_rate("nore", 0.0)
        engine.set_drug_rate("phenyl", 20.0)
        assert step.check_requirements(engine)[0]
        engine.set_drug_rate("phenyl", 0.0)
        # Neither infusion remains running.
        assert not step.check_requirements(engine)[0]
    engine.set_drug_rate("nore", 5.0)
    assert step.check_requirements(engine)[0]

    engine.set_drug_rate("nore", 0.0)
    assert not step.check_requirements(engine)[0]


def test_sepsis_fluid_objective_requires_crystalloid(engine_factory):
    engine = engine_factory(start=True)
    step = _step(create_sepsis_response(), "GIVE_FLUIDS")
    engine.start_sepsis()

    engine.give_blood(600)
    engine.give_albumin(500)
    assert not step.check_requirements(engine)[0]

    engine.give_fluid(500)
    assert step.check_requirements(engine)[0]


def test_extubation_requires_recovered_block_and_spontaneous_ventilation(engine_factory):
    """Residual block impairs the upper airway after breathing returns (Eikermann 2003)."""
    engine = engine_factory(start=True, weight=100, height=177)
    step = _step(create_emergence("tiva"), "EXTUBATE")
    _early_extubation(engine)
    engine.state.tof = 89.0
    assert not step.check_requirements(engine)[0]
    engine.state.tof = 90.0
    engine.state.twitch = 20.0
    met, message = step.check_requirements(engine)
    assert not met and "T1: 20%" in message
    engine.state.twitch = 100.0
    engine.set_vent_power(True)
    assert not step.check_requirements(engine)[0]
    engine.set_vent_power(False)
    assert step.check_requirements(engine)[0]
    engine.set_bag_mask_ventilation(True)
    assert not step.check_requirements(engine)[0]
    engine.set_bag_mask_ventilation(False)
    reading = engine.nmb.reading
    assert reading is not None
    for invalid in (None, replace(reading, time=engine.state.time - 61),
                    replace(reading, ratio=89.0), replace(reading, mode="single", ratio=None)):
        engine.nmb.reading = invalid
        assert not step.check_requirements(engine)[0]
    engine.nmb.reading = reading
    engine.state.vt = 200.0
    assert not step.check_requirements(engine)[0]


@pytest.mark.parametrize("scenario_id", ["emergence_tiva", "emergence_balanced"])
def test_emergence_starts_with_residual_block_reversed_by_sugammadex(scenario_id):
    """Sugammadex 2 mg/kg at reappearance of T2 after rocuronium 0.6 mg/kg gives
    a TOF ratio of 0.9 in a median 1.4 min (IQR 1.2-1.7) in the Bridion label's
    comparative trial, and 1.7-2.2 min in its other trials."""
    session = WebSession({"scenario_id": scenario_id})
    engine = session.engine
    assert session.scenario is not None
    step = _step(session.scenario, "ASSESS_BLOCK")
    engine.start()
    for _ in range(13):
        engine.step(2.0)
    reading = engine.nmb.reading
    assert reading is not None and reading.mode == "tof"
    assert 2 <= reading.count <= 3
    assert not step.check_requirements(engine)[0]

    session.command("sugammadex", '{"mg_per_kg": 2}')
    seconds = 0
    while engine.state.tof < 90.0 and seconds < 600:
        engine.step(2.0)
        seconds += 2
    assert 60 <= seconds <= 180
    for _ in range(13):
        engine.step(2.0)
    assert step.check_requirements(engine)[0]


@pytest.mark.parametrize("scenario_id,event,fluid_ml", [
    ("hemorrhage_response", "hemorrhage", 500),
    ("sepsis_response", "sepsis", 500),
    ("anaphylaxis_response", "anaphylaxis", 1000),
])
def test_crisis_objectives_accept_prompt_treatment_without_repeating_doses(scenario_id, event, fluid_ml):
    session = WebSession({"scenario_id": scenario_id})
    engine, scenario = session.engine, session.scenario
    assert scenario is not None
    fluid_step = _step(scenario, "FLUIDS" if event == "anaphylaxis" else "GIVE_FLUIDS")
    engine.give_fluid(fluid_ml)
    getattr(engine, f"start_{event}")()
    assert not fluid_step.check_requirements(engine)[0]  # Pre-event fluid does not count.
    if event == "hemorrhage":
        engine.give_blood(300)  # A single red-cell unit is an acceptable initial replacement.
    else:
        engine.give_fluid(fluid_ml)
    if event == "anaphylaxis":
        engine.give_drug_bolus("epi", 100)
        engine.set_fgf(10, 0, 0)
        step = _step(scenario, "EPINEPHRINE")
        assert step.check_requirements(engine)[0]
    else:
        step = _step(scenario, "RECOGNIZE_SHOCK" if event == "hemorrhage" else "RECOGNIZE_WARM_SHOCK")
        assert engine.state.map > 65.0
        assert step.check_requirements(engine)[0]  # Treatment need not wait for hypotension.
    assert fluid_step.check_requirements(engine)[0]
    getattr(engine, f"stop_{event}")()
    if event != "anaphylaxis":
        step = _step(scenario, "STOP_BLEEDING" if event == "hemorrhage" else "SOURCE_CONTROL")
        assert step.check_requirements(engine)[0]
    getattr(engine, f"start_{event}")()
    assert not fluid_step.check_requirements(engine)[0]  # A later event needs its own response.
    engine.give_saline(fluid_ml)
    assert fluid_step.check_requirements(engine)[0]

    if event == "anaphylaxis":
        arrest_session = WebSession({"scenario_id": "cardiac_arrest"})
        arrest_engine, arrest_scenario = arrest_session.engine, arrest_session.scenario
        assert arrest_scenario is not None
        arrest_scenario.events[0].apply(arrest_engine)
        arrest_engine.start()
        for _ in range(10):
            arrest_engine.step(2.0)
        epi_step = _step(arrest_scenario, "CONTINUE_CYCLES")
        arrest_engine.give_drug_bolus("epi", 500)
        assert not epi_step.check_requirements(arrest_engine)[0]
        arrest_engine.give_drug_bolus("epi", 500)
        assert epi_step.check_requirements(arrest_engine)[0]


@pytest.mark.parametrize("assisted", [False, True], ids=["tidal-breathing", "assisted-shallow-breathing"])
def test_preoxygenation_waits_for_lung_washin(engine_factory, assisted):
    spec = next(spec for spec in SCENARIO_REGISTRY if spec.id == "induction_tiva")
    step = _step(spec.builder(), "PREOXYGENATE")
    patient = dict(age=70, baseline_rr=24, baseline_vt=250) if assisted else {}
    engine = engine_factory(start=True, **patient)
    engine.set_airway_mode("Mask")
    engine.set_fgf(10.0, 0.0)
    for _ in range(30):
        engine.step(2.0)
    assert not step.check_requirements(engine)[0]
    for _ in range(60):
        engine.step(2.0)
    if assisted:
        # Rapid shallow breaths mostly move dead space, so the lungs wash in
        # slowly; a high inspired fraction alone is not preoxygenation.
        assert engine.state.fio2 > 0.95
        assert not step.check_requirements(engine)[0]
        engine.set_vent_settings(rr=16, vt=0.5, peep=5, ie="1:2", mode="VCV")
        engine.set_vent_power(True)
        for _ in range(180):
            engine.step(1.0)
    else:
        for _ in range(30):
            engine.step(2.0)
    assert engine.state.et_o2 >= 90
    assert step.check_requirements(engine)[0]

    # The flowmeter stays set, but no O2 is delivered.
    engine.set_oxygen_supply_connected(False)
    assert not step.check_requirements(engine)[0]


@pytest.mark.parametrize("spec", SCENARIO_REGISTRY, ids=lambda spec: spec.id)
@pytest.mark.parametrize("tci_enabled", [False, True], ids=["manual", "tci"])
def test_guided_scenario_completes_through_browser_commands(spec, tci_enabled):
    """Every guided objective must remain reachable through the browser protocol."""
    session = WebSession({"scenario_id": spec.id, "tci_enabled": tci_enabled})
    session.engine.config.dt = 0.25 if spec.id == "malignant_hyperthermia" else 0.5
    # About 1 in 10 first shocks fails; this seed's succeeds.
    session.engine._shock_rng = np.random.default_rng(0)
    info = json.loads(session.info())
    session.command("run", '{"running": true}')
    session.command("speed", '{"value": 10}')
    pending = dict(SCENARIO_WALKTHROUGHS[spec.id])
    samples = 0
    snap = json.loads(session.advance(0.0))
    # Intraoperative crises start during surgery, normotensive without vasopressors.
    assert session.engine.disturbance_active is spec.surgery
    if spec.surgery:
        assert session.engine.state.map > 65.0

    # Continue does nothing until the current objective is met.
    if not snap["scenario"]["met"]:
        session.command("scenario_next")
        assert json.loads(session.advance(0.0))["scenario"]["id"] == snap["scenario"]["id"]

    seen = set()
    while not snap["scenario"]["complete"]:
        step = snap["scenario"]
        if step["id"] not in seen:
            seen.add(step["id"])
            assert not (step["id"] == "RECOGNIZE_SHOCK" and step["met"])
        if spec.id == "induction_tiva":
            uses_tci_guidance = tci_enabled and step["id"] in {"START_ANALGESIA", "INDUCE"}
            assert ("TCI" in step["instruction"]) is uses_tci_guidance
            if not tci_enabled:
                assert "µg/mL" not in step["instruction"] and "ng/mL" not in step["instruction"]
        ran_cmd = False
        for name, args in pending.pop(step["id"], []):
            ran_cmd = True
            if name == "shock":
                session.command("charge_defibrillator")
                for _ in range(3):
                    charged = json.loads(session.advance(0.2))
                    samples += len(charged["waves"]["ecg"])
            if spec.id == "induction_tiva" and tci_enabled and name == "drug_bolus" and args["key"] == "propofol":
                continue  # The TCI controller provides induction loading.
            if name == "drug_target" and not tci_enabled:
                name = "drug_rate"
                key = args["key"]
                args = {"key": key, "rate": {"propofol": (100 if step["id"] == "MAINTENANCE" else 150), "remi": 0.2}[key]}
            session.command(name, json.dumps(args))
        met = json.loads(session.advance(0.0))["scenario"]["met"] if ran_cmd else step["met"]
        if met:
            session.command("scenario_next")
            snap = json.loads(session.advance(0.0))
        else:
            assert snap["time"] < 1800, f"{spec.id}/{step['id']} unreachable: {step['status']}"
            snap = json.loads(session.advance(0.2))
            samples += len(snap["waves"]["ecg"])

    assert not pending
    # Following the guide achieves every debrief outcome.
    assert [o["id"] for o in snap["scenario"]["debrief"]["outcomes"] if not o["achieved"]] == []
    # Every simulated step reaches the monitor sweep exactly once.
    assert samples == round(snap["time"] / info["sample_interval"])
    if spec.id == "induction_tiva":
        controls = snap["controls"]
        assert controls["airway"] == "ETT" and not controls["bag_mask"] and controls["vent"]["on"]
        assert snap["vitals"]["etco2"] > 25
        for key in ("propofol", "remi"):
            assert controls["drugs"][key]["is_tci"] is tci_enabled
            assert controls["drugs"][key]["rate"] > 0 or controls["drugs"][key]["target"] > 0
        assert snap["vitals"]["bis"] < 60
        assert session.engine.state.tof < 25


def test_emergence_gating_normocapnia_covariates_and_cpr_or_nmba_checks(engine_factory):
    """Stage II blocks extubation readiness and warns on tube removal; post-NMB
    TOF recovery persists across later emergence steps unless NMB is redosed;
    unmet integer status strings never round into passing bounds; emergence
    normocapnia adapts to pregnancy and COPD; stale or failed NIBP readings do
    not pass vasopressor/shock checks; chest compressions do not satisfy
    spontaneous MAP/stability checks; dantrolene weakness does not report false
    NMBA T1 block; CPAP is accepted at emergence; O2 recovery milestone persists
    when conserving flow; position changes appear in debriefs.
    """
    scenario = create_emergence("balanced")
    engine = engine_factory(start=True)
    _early_extubation(engine)

    # BIS > 80 with residual Stage II loss of consciousness is not ready to extubate.
    engine.set_airway_mode("ETT")
    engine.pk_sevo.state.mac = 0.42
    engine.state.display_bis = 82.0
    engine.state.loc = 0.27
    met, msg = _step(scenario, "EXTUBATION_READINESS").check_requirements(engine)
    assert not met and "Stage II" in msg
    feedback = check_airway_removal(engine)
    assert feedback is not None and "Stage II" in feedback.message

    engine.pk_sevo.state.mac = 0.0
    engine.state.loc = 0.0
    assert check_airway_removal(engine) is None
    engine.set_airway_mode("Mask")
    assert _step(scenario, "EXTUBATE").check_requirements(engine)[0]

    # Older post-NMB TOF >= 90% fails ASSESS_BLOCK (>60 s) but satisfies EXTUBATE
    # until a new neuromuscular blocker dose invalidates it.
    engine.state.time = 200.0
    assert engine.nmb.reading is not None
    engine.nmb.reading = replace(engine.nmb.reading, time=50.0)
    assert not _step(scenario, "ASSESS_BLOCK").check_requirements(engine)[0]
    assert _step(scenario, "EXTUBATE").check_requirements(engine)[0]
    engine.give_drug_bolus("roc", 5.0)
    engine.state.tof = engine.state.twitch = 100.0
    assert not _step(scenario, "EXTUBATE").check_requirements(engine)[0]
    engine.nmb.reading = replace(engine.nmb.reading, time=200.0)
    assert _step(scenario, "EXTUBATE").check_requirements(engine)[0]

    assert format_unmet_int(80.0, low=80) == 79
    assert format_unmet_int(45.3, low=35, high=45) == 46
    engine.state.art_map = 64.8
    met_map, map_msg = require_map_at_least(65)(engine)
    assert not met_map and "64/65+" in map_msg

    # CPAP is accepted as unassisted breathing at emergence.
    _early_extubation(engine)
    engine.set_airway_mode("ETT")
    engine.vent.update_settings(mode="CPAP", rr=0, peep=5.0)
    engine.set_vent_power(True)
    assert _step(create_emergence("tiva"), "EXTUBATION_READINESS").check_requirements(engine)[0]

    for overrides, etco2 in (
        ({}, 34.7),
        ({"sex": "female", "gestational_age_weeks": 39.0}, 31.0),
        ({"copd_severity": 1.0}, 31.5),
    ):
        case = engine_factory(config=SimulationConfig(rng_seed=1), **overrides)
        case.state.etco2_signal_valid = True
        case.state.display_etco2 = etco2
        assert _require_normocapnia(case)[0] is True

    no_art = engine_factory(config=SimulationConfig(arterial_line_enabled=False, rng_seed=1))
    no_art.state.nibp_map = 75.0
    no_art.state.nibp_timestamp = 0.0
    no_art.state.time = 120.0
    no_art.state.map = 52.0
    check = require_vasopressor_if_hypotensive("nore", fail_message="Start norepinephrine")
    met, msg = check(no_art)
    assert not met and "measure blood pressure again" in msg.lower()

    no_art.state.nibp_map = 0.0
    no_art.state.nibp_timestamp = None
    start_map = CrisisStartMAP()
    start_map.record(no_art)
    assert start_map.value is None
    assert not require_hypotension_or_treatment(start_map, "hemorrhage", "Assess BP")(no_art)[0]

    cpr_session = WebSession({"scenario_id": "hemorrhage_response", "practice_mode": "independent",
                              "sex": "female", "gestational_age_weeks": 39.0})
    cpr_session.command("position", '{"position": "left_lateral"}')
    cpr_eng = cpr_session.engine
    cpr_eng.set_rhythm("VFIB")
    cpr_eng.set_chest_compressions(True)
    cpr_eng.state.art_map = 72.0
    cpr_eng.state.map = 72.0
    cpr_eng.state.sao2 = 98.0
    assert not require_map_at_least(65)(cpr_eng)[0]
    assert not patient_stable(cpr_eng)
    assert cpr_session.case_run is not None
    cpr_session.case_run.observe(5.0, force=True)
    assert cpr_session.case_run.hypotension_s >= 5.0
    debrief = json.loads(cpr_session.command("case_finish", "{}"))["debrief"]
    assert any("Position: left lateral" in item["text"] for item in debrief["timeline"])

    mh_eng = engine_factory(start=True)
    mh_eng.give_drug_bolus("dantrolene", 175.0)
    for _ in range(15):
        mh_eng.step(2.0)
    assert mh_eng.state.twitch < 85.0
    _, val = neuromuscular_response(mh_eng)
    assert val >= 99.0

    ox_session = WebSession({"scenario_id": "oxygen_supply_failure", "practice_mode": "independent"})
    assert ox_session.case_run is not None
    ox_session.engine.start()
    # Raising O2 flow before the 30 s supply failure must not award premature recovery credit.
    ox_session.engine.set_fgf(10.0, 0.0, 0.0)
    for _ in range(5):
        ox_session.case_run.before_step()
        ox_session.engine.step(1.0)
        ox_session.case_run.observe(1.0, force=True)
    assert ox_session.engine.circuit.composition.fio2 > 0.40
    assert "oxygen" not in ox_session.case_run.achieved_at
    # After the supply fails, recovery stays uncredited until backup oxygen is connected.
    for _ in range(30):
        ox_session.case_run.before_step()
        ox_session.engine.step(1.0)
        ox_session.case_run.observe(1.0, force=True)
    assert not ox_session.engine.circuit.oxygen_supply_connected
    assert "oxygen" not in ox_session.case_run.achieved_at
    ox_session.engine.set_oxygen_supply_connected(True)
    ox_session.engine.circuit.composition.fio2 = 0.55
    ox_session.engine.state.sao2 = 98.0
    ox_session.case_run.observe(force=True)
    assert "oxygen" in ox_session.case_run.achieved_at
    # Milestone stays credited even if FiO2 subsequently drops below the 40% recovery threshold.
    ox_session.engine.circuit.composition.fio2 = 0.35
    ox_session.case_run.observe(force=True)
    assert "oxygen" in ox_session.case_run.achieved_at


def test_cardiac_arrest_step2_rosc_progression_stability_hint_and_finite_settle(engine_factory):
    """Immediate ROSC satisfies cardiac_arrest Step 2 without forcing compressions on a beating heart,
    require_crisis_resolved_with_map emits a non-empty hint on instability, and hidden settle skips
    finite disturbances."""
    scen = create_cardiac_arrest()
    eng = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    scen.events[0].apply(eng)
    eng.set_rhythm("SINUS")
    eng.step(0.5)
    step2 = _step(scen, "START_COMPRESSIONS")
    assert step2.check_requirements(eng)[0]

    req = require_crisis_resolved_with_map("hemorrhage")
    eng.state.display_spo2 = 93.0
    eng.state.sao2 = 93.0
    eng.state.etco2 = 35.0
    ok, msg = req(eng)
    assert not ok and bool(msg)

    finite = engine_factory(
        config=SimulationConfig(mode="steady_state", disturbance_profile="stim_intubation_pulse", rng_seed=1),
        start=True,
    )
    assert finite.disturbances.spec is not None and finite.disturbances.spec.finite
    assert finite.state.map < 110.0


def test_scenario_patient_scaling_and_mh_recovery_checks(engine_factory, advance_time):
    """O2 supply failure and anaphylaxis equilibrate baseline FiO2 across weights;
    MH starts with forced-air warming, caps COPD hyperventilation, and checks 30-45 mmHg EtCO2."""
    obese_ox = engine_factory(
        config=SimulationConfig(mode="steady_state", rng_seed=1),
        start=True, age=40, weight=200, height=185, sex="male",
    )
    ox_scen = create_oxygen_supply_failure()
    ox_scen.prepare(obese_ox)
    advance_time(obese_ox, 25.0, dt=1.0)
    assert 0.34 <= obese_ox.state.fio2 <= 0.43
    assert obese_ox.state.sao2 >= 95.0
    assert "FiO2" not in obese_ox.state.alarms

    ana_scen = create_anaphylaxis_scenario()
    ana_eng = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1), start=True)
    ana_scen.prepare(ana_eng)
    assert 0.35 <= ana_eng.state.fio2 <= 0.45
    assert "volatile" not in _step(ana_scen, "EPINEPHRINE").instruction_for(ana_eng).lower()

    mh_scen = create_malignant_hyperthermia()
    copd_obese = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1),
        start=True, age=60, weight=200, height=185, sex="male", copd_severity=1.0,
    )
    mh_scen.prepare(copd_obese)
    mh_scen.events[0].apply(copd_obese)
    assert copd_obese.state.bair_hugger_target == pytest.approx(38.0)
    copd_obese.state.etco2_signal_valid = True
    copd_obese.state.display_etco2 = 19.0
    assert _require_etco2_normal(copd_obese)[0] is False
    copd_obese.state.display_etco2 = 38.0
    assert _require_etco2_normal(copd_obese)[0] is True
    stop_step = _step(mh_scen, "STOP_TRIGGERS")
    copd_obese.set_vaporizer(0.0)
    copd_obese.set_fgf(10.0, 0.0, 0.0)
    copd_obese.set_drug_rate("propofol", 120.0)
    copd_obese.set_vent_settings(rr=24, vt=0.6, peep=5.0, ie="1:2", mode="VCV")
    copd_obese.state.mv = 14.0
    assert stop_step.check_requirements(copd_obese)[0] is True


def test_balanced_handover_bis_mode_text_and_shock_cuff_checks(engine_factory):
    """Balanced induction accepts transitional BIS 30-60 only while propofol washes out;
    cardiac arrest hides arterial-line prompts when disabled; failed shock NIBP satisfies crisis MAP check."""
    bal_scen = create_induction_balanced()
    maint_step = _step(bal_scen, "MAINTENANCE")
    bal = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1), start=True)
    bal.set_fgf(1.0, 1.0, 0.0)
    bal.state.et_mac = 0.8
    bal.state.propofol_ce = 1.2
    bal.state.display_bis = 35.0
    bal.state.art_map = 75.0
    bal.state.map = 75.0
    bal.state.spo2_signal_valid = True
    bal.state.display_spo2 = 98.0
    assert maint_step.check_requirements(bal)[0] is True
    bal.state.propofol_ce = 0.2
    assert maint_step.check_requirements(bal)[0] is False

    ca_scen = create_cardiac_arrest()
    no_art = engine_factory(config=SimulationConfig(mode="steady_state", arterial_line_enabled=False, rng_seed=1), start=True)
    start_cpr_text = _step(ca_scen, "START_COMPRESSIONS").instruction_for(no_art).lower()
    assert "arterial line" not in start_cpr_text
    assert "anesthetic" in start_cpr_text

    no_art.state.nibp_measurement_failed = True
    no_art.state.nibp_map = 0.0
    no_art.state.map = 22.0
    start_map = CrisisStartMAP()
    assert require_hypotension_or_treatment(start_map, "hemorrhage", "Assess BP")(no_art)[0] is True
    no_art.state.nibp_measurement_failed = False
    ok, msg = require_hypotension_or_treatment(start_map, "hemorrhage", "Assess BP")(no_art)
    assert not ok and "MAP: 0/" not in msg
