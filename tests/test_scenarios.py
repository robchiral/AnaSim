import ast
import json
from dataclasses import replace
from pathlib import Path
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
from anasim.scenarios.malignant_hyperthermia import (
    _require_controlled,
    _require_etco2_normal,
)
from anasim.scenarios.runtime import SAFETY_REVIEW
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
        "GIVE_FLUIDS": [("fluid", {"kind": "crystalloid", "volume_ml": 1000})],
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
        "MASK_VENTILATE": [("apl", {"cmh2o": 20}), ("bag_mask", {"active": True})],
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
        "MASK_VENTILATE": [("apl", {"cmh2o": 20}), ("bag_mask", {"active": True})],
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
            ("drug_rate", {"key": "propofol", "rate": 0}),
            ("drug_rate", {"key": "remi", "rate": 0}),
            ("fgf", {"o2": 10, "air": 0, "n2o": 0}),
        ],
        "DEFIBRILLATE": [("shock", {"synchronized": False})],
        "CONTINUE_CYCLES": [
            ("shock", {"synchronized": False}),
            ("drug_bolus", {"key": "epi", "amount": 1000}),
            ("shock", {"synchronized": False}),
        ],
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
    """Set awake, breathing, reversed state so individual extubation gates can be toggled."""
    engine.state.display_bis = 90.0
    engine.resp.state.rr = engine.state.rr = 12.0
    engine.resp.state.vt = engine.state.vt = 500.0
    engine.resp.state.apnea = False
    engine.state.tof = engine.state.twitch = 100.0
    engine.state.display_spo2 = 98.0
    engine.state.spo2_signal_valid = True
    engine.set_vent_power(False)
    engine.set_airway_mode("Mask")
    engine.nmb.reading = TwitchReading(
        engine.state.time, "tof", (1.0, 1.0, 1.0, 1.0), 4, 100.0, 100.0, False,
    )


@pytest.mark.parametrize("spec", SCENARIO_REGISTRY, ids=lambda spec: spec.id)
@pytest.mark.parametrize("tci_enabled", [False, True], ids=["manual", "tci"])
def test_guided_scenario_completes_through_browser_commands(spec, tci_enabled):
    """Every guided objective remains reachable through the browser protocol."""
    session = WebSession({"scenario_id": spec.id, "tci_enabled": tci_enabled})
    session.engine.config.dt = 0.25 if spec.id == "malignant_hyperthermia" else 0.5
    # Deterministic RNG so the post-refractory third shock succeeds reliably in cardiac arrest.
    session.engine._shock_rng = np.random.default_rng(0)
    info = json.loads(session.info())
    session.command("run", '{"running": true}')
    session.command("speed", '{"value": 10}')
    pending = dict(SCENARIO_WALKTHROUGHS[spec.id])
    samples = 0
    snap = json.loads(session.advance(0.0))
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
                continue  # TCI provides induction loading automatically.
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
    assert [o["id"] for o in snap["scenario"]["debrief"]["outcomes"] if not o["achieved"]] == []
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
        # Rapid shallow breaths mostly move dead space; high FiO2 alone is not preoxygenation.
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

    engine.set_oxygen_supply_connected(False)
    assert not step.check_requirements(engine)[0]


def test_induction_handover_accepts_transitional_bis_while_propofol_clears(engine_factory):
    """Both balanced and manual TIVA maintenance accept BIS 30-60 while an induction bolus clears."""
    bal = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1), start=True)
    bal_step = _step(create_induction_balanced(), "MAINTENANCE")
    bal.set_fgf(1.0, 1.0, 0.0)
    bal.state.et_mac = 0.8
    bal.state.propofol_ce = 1.2
    bal.state.display_bis = 35.0
    bal.state.art_map = bal.state.map = 75.0
    bal.state.spo2_signal_valid = True
    bal.state.display_spo2 = 98.0
    assert bal_step.check_requirements(bal)[0] is True
    bal.state.propofol_ce = 0.2
    assert bal_step.check_requirements(bal)[0] is False

    tiva = WebSession({"scenario_id": "induction_tiva", "tci_enabled": False})
    te = tiva.engine
    assert tiva.scenario is not None
    tiva_step = _step(tiva.scenario, "MAINTENANCE")
    te.set_airway_mode("ETT")
    te.set_vent_power(True)
    te.set_fgf(1.0, 1.0, 0.0)
    te.give_drug_bolus("propofol", 150.0)
    te.set_drug_rate("propofol", 100.0)
    te.set_drug_rate("remi", 0.2)
    te.state.display_bis = 34.0
    te.state.art_map = te.state.map = 75.0
    te.state.spo2_signal_valid = True
    te.state.display_spo2 = 98.0
    assert tiva_step.check_requirements(te)[0] is True
    te.state.time = 700.0
    assert tiva_step.check_requirements(te)[0] is False


@pytest.mark.parametrize("scenario_id", ["emergence_tiva", "emergence_balanced"])
def test_emergence_starts_with_residual_block_reversed_by_sugammadex(scenario_id):
    """Emergence starts stably at upper maintenance depth with TOF count 2-3 reversed by sugammadex 2 mg/kg."""
    session = WebSession({"scenario_id": scenario_id})
    engine = session.engine
    assert session.scenario is not None
    if scenario_id == "emergence_balanced":
        assert engine.circuit.fgf_total() == pytest.approx(2.0, abs=0.1)
    assess_step = _step(session.scenario, "ASSESS")
    block_step = _step(session.scenario, "ASSESS_BLOCK")
    engine.start()
    for _ in range(15):
        engine.step(2.0)
    assert assess_step.check_requirements(engine)[0]
    reading = engine.nmb.reading
    assert reading is not None and reading.mode == "tof"
    assert 2 <= reading.count <= 3
    assert not block_step.check_requirements(engine)[0]

    session.command("sugammadex", '{"mg_per_kg": 2}')
    seconds = 0
    while engine.state.tof < 90.0 and seconds < 600:
        engine.step(2.0)
        seconds += 2
    assert 60 <= seconds <= 180
    for _ in range(13):
        engine.step(2.0)
    assert block_step.check_requirements(engine)[0]


def test_extubation_readiness_and_airway_gating(engine_factory):
    """Extubation requires quantitative block recovery, unassisted breathing (or CPAP), wakefulness
    past Stage II, and patient-adjusted normocapnia; dantrolene weakness is not mistaken for NMBA fade."""
    engine = engine_factory(start=True, weight=100, height=177)
    balanced = create_emergence("balanced")
    tiva = create_emergence("tiva")
    extubate = _step(tiva, "EXTUBATE")
    readiness = _step(balanced, "EXTUBATION_READINESS")

    _early_extubation(engine)
    engine.state.tof = 89.0
    assert not extubate.check_requirements(engine)[0]
    engine.state.tof = 90.0
    engine.state.twitch = 20.0
    met, msg = extubate.check_requirements(engine)
    assert not met and "T1: 20%" in msg
    engine.state.twitch = 100.0

    for toggle in (engine.set_vent_power, engine.set_bag_mask_ventilation):
        toggle(True)
        assert not extubate.check_requirements(engine)[0]
        toggle(False)
        assert extubate.check_requirements(engine)[0]

    reading = engine.nmb.reading
    assert reading is not None
    for invalid in (None, replace(reading, time=engine.state.time - 61),
                    replace(reading, ratio=89.0), replace(reading, mode="single", ratio=None)):
        engine.nmb.reading = invalid
        assert not extubate.check_requirements(engine)[0]
    engine.nmb.reading = reading
    engine.state.vt = 200.0
    assert not extubate.check_requirements(engine)[0]

    # Stage II blocks extubation readiness and warns on tube removal.
    _early_extubation(engine)
    engine.set_airway_mode("ETT")
    engine.pk_sevo.state.mac = 0.42
    engine.state.display_bis = 82.0
    engine.state.loc = 0.27
    met, msg = readiness.check_requirements(engine)
    assert not met and "Stage II" in msg
    feedback = check_airway_removal(engine)
    assert feedback is not None and "Stage II" in feedback.message

    # Post-NMB recovery persists across later emergence steps unless NMB is redosed.
    engine.pk_sevo.state.mac = engine.state.loc = 0.0
    assert check_airway_removal(engine) is None
    engine.set_airway_mode("Mask")
    engine.state.time = 200.0
    engine.nmb.reading = replace(reading, time=50.0)
    assert not _step(balanced, "ASSESS_BLOCK").check_requirements(engine)[0]
    assert _step(balanced, "EXTUBATE").check_requirements(engine)[0]
    engine.give_drug_bolus("roc", 5.0)
    engine.state.tof = engine.state.twitch = 100.0
    assert not _step(balanced, "EXTUBATE").check_requirements(engine)[0]
    engine.nmb.reading = replace(reading, time=200.0)
    assert _step(balanced, "EXTUBATE").check_requirements(engine)[0]

    # CPAP is accepted as unassisted breathing at emergence.
    _early_extubation(engine)
    engine.set_airway_mode("ETT")
    engine.vent.update_settings(mode="CPAP", rr=0, peep=5.0)
    engine.set_vent_power(True)
    assert _step(tiva, "EXTUBATION_READINESS").check_requirements(engine)[0]

    # Normocapnia adapts to pregnancy and COPD.
    for overrides, etco2 in (
        ({}, 34.7),
        ({"sex": "female", "gestational_age_weeks": 39.0}, 31.0),
        ({"copd_severity": 1.0}, 31.5),
    ):
        case = engine_factory(config=SimulationConfig(rng_seed=1), **overrides)
        case.state.etco2_signal_valid = True
        case.state.display_etco2 = etco2
        assert _require_normocapnia(case)[0] is True

    # Dantrolene depresses twitch height without producing NMBA fade.
    mh_eng = engine_factory(start=True)
    mh_eng.give_drug_bolus("dantrolene", 175.0)
    for _ in range(15):
        mh_eng.step(2.0)
    assert mh_eng.state.twitch < 85.0
    assert neuromuscular_response(mh_eng)[1] >= 99.0


@pytest.mark.parametrize("scenario_id,event,fluid_ml", [
    ("hemorrhage_response", "hemorrhage", 1000),
    ("sepsis_response", "sepsis", 500),
    ("anaphylaxis_response", "anaphylaxis", 1000),
])
def test_crisis_objectives_accept_prompt_treatment_without_repeating_doses(scenario_id, event, fluid_ml):
    session = WebSession({"scenario_id": scenario_id})
    engine, scenario = session.engine, session.scenario
    assert scenario is not None
    fluid_step = _step(scenario, "FLUIDS" if event == "anaphylaxis" else "GIVE_FLUIDS")
    engine.give_fluid(fluid_ml)
    scenario.events[0].apply(engine)
    assert not fluid_step.check_requirements(engine)[0]  # Pre-event fluid does not count.
    if event == "hemorrhage":
        engine.give_fluid(500)
        engine.give_blood(300)  # 500 mL crystalloid + 1 U PRBC meets initial target.
    else:
        engine.give_fluid(fluid_ml)
    if event == "anaphylaxis":
        engine.give_drug_bolus("epi", 100)
        engine.set_fgf(10, 0, 0)
        assert _step(scenario, "EPINEPHRINE").check_requirements(engine)[0]
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


@pytest.mark.parametrize("builder,event", [
    (create_hemorrhage_response, "hemorrhage"),
    (create_sepsis_response, "sepsis"),
])
def test_vasopressor_objective_tracks_pressure_and_current_support(engine_factory, builder, event):
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced",
                                disturbance_profile="stim_sustained_surgery"),
        start=True,
    )
    step = _step(builder(), "START_VASOPRESSOR")
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
        assert not step.check_requirements(engine)[0]
        engine.set_drug_rate("nore", 5.0)
    engine.set_drug_rate("nore", 0.0)
    assert not step.check_requirements(engine)[0]


def test_sepsis_requires_crystalloid_vasopressor_and_retains_vasoplegia(engine_factory):
    """Sepsis fluid step requires crystalloid, a 500 mL bolus alone leaves MAP < 65, and source control
    does not terminate established vasoplegia."""
    eng = engine_factory(start=True)
    fluid_step = _step(create_sepsis_response(), "GIVE_FLUIDS")
    eng.start_sepsis()
    eng.give_blood(600)
    eng.give_albumin(500)
    assert not fluid_step.check_requirements(eng)[0]
    eng.give_fluid(500)
    assert fluid_step.check_requirements(eng)[0]

    session = WebSession({"scenario_id": "sepsis_response"})
    se = session.engine
    assert session.case_run is not None
    se.start()
    for _ in range(75):
        session.case_run.before_step()
        se.step(1.0)
        session.case_run.observe(1.0)
    assert se.state.map < 60.0
    se.give_fluid(500.0)
    for _ in range(180):
        session.case_run.before_step()
        se.step(1.0)
        session.case_run.observe(1.0)
    assert se.state.map < 65.0
    session.command("case_action", '{"action": "source_control"}')
    assert se.sepsis.active


def test_malignant_hyperthermia_scaling_and_resolution_status(engine_factory):
    """MH starts on low-flow maintenance with warmer on, adapts hyperventilation and EtCO2 bounds
    for COPD/pregnancy, and distinguishes active dantrolene resolution from undertreatment."""
    mh_scen = create_malignant_hyperthermia()
    copd_obese = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1),
        start=True, age=60, weight=200, height=185, sex="male", copd_severity=1.0,
    )
    mh_scen.prepare(copd_obese)
    mh_scen.events[0].apply(copd_obese)
    assert copd_obese.circuit.fgf_total() == pytest.approx(2.0, abs=0.1)
    assert copd_obese.state.bair_hugger_target == pytest.approx(38.0)
    copd_obese.state.etco2_signal_valid = True
    copd_obese.state.display_etco2 = 19.0
    assert _require_etco2_normal(copd_obese)[0] is False
    copd_obese.state.display_etco2 = 38.0
    assert _require_etco2_normal(copd_obese)[0] is True
    copd_obese.set_vaporizer(0.0)
    copd_obese.set_fgf(10.0, 0.0, 0.0)
    copd_obese.set_drug_rate("propofol", 120.0)
    copd_obese.set_vent_settings(rr=24, vt=0.6, peep=5.0, ie="1:2", mode="VCV")
    copd_obese.state.mv = 14.0
    assert _step(mh_scen, "STOP_TRIGGERS").check_requirements(copd_obese)[0] is True

    preg = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1),
        start=True, sex="female", gestational_age_weeks=38.0,
    )
    preg.state.etco2_signal_valid = True
    preg.state.display_etco2 = 27.0
    assert _require_etco2_normal(preg)[0] is True
    assert "22-37" in _step(mh_scen, "STABILIZE").instruction_for(preg)

    session = WebSession({"scenario_id": "malignant_hyperthermia"})
    me = session.engine
    me.start()
    session.command("vaporizer", '{"percent": 0}')
    session.command("fgf", '{"o2": 10, "air": 0, "n2o": 0}')
    session.command("vent", '{"rr": 24, "tv": 550}')
    session.command("drug_bolus", '{"key": "dantrolene", "amount": 175}')
    for _ in range(120):
        me.step(1.0)
    assert "resolving" in _require_controlled(me)[1].lower()


def test_cardiac_arrest_acls_objectives_and_debrief(engine_factory):
    """Immediate ROSC satisfies Step 2, non-arterial-line prompts omit arterial-line cues,
    epinephrine requires 1000 mcg total, and omitted epinephrine fails debrief after refractory shocks."""
    scen = create_cardiac_arrest()
    no_art = engine_factory(
        config=SimulationConfig(mode="steady_state", arterial_line_enabled=False, rng_seed=1), start=True,
    )
    start_cpr_text = _step(scen, "START_COMPRESSIONS").instruction_for(no_art).lower()
    assert "arterial line" not in start_cpr_text and "anesthetic" in start_cpr_text

    scen.events[0].apply(no_art)
    no_art.set_rhythm("SINUS")
    no_art.step(0.5)
    assert _step(scen, "START_COMPRESSIONS").check_requirements(no_art)[0]

    # Split epinephrine doses sum to 1000 mcg for persistent VF.
    ca = WebSession({"scenario_id": "cardiac_arrest", "practice_mode": "independent"})
    ce = ca.engine
    ce._shock_rng = np.random.default_rng(0)
    assert ca.case_run is not None and ca.scenario is not None
    ce.start()
    for _ in range(32):
        ca.case_run.before_step()
        ce.step(1.0)
        ca.case_run.observe(1.0)
    ce.set_chest_compressions(True)
    # Three shocks without epinephrine achieve ROSC on shock 3 but fail the epinephrine outcome.
    for _ in range(3):
        ca.command("charge_defibrillator")
        for _ in range(5):
            ce.step(1.0)
        ca.command("shock", '{"synchronized": false}')
    ce.set_chest_compressions(False)
    for _ in range(5):
        ce.step(1.0)
        ca.case_run.observe(1.0)
    debrief = json.loads(ca.command("case_finish", "{}"))["debrief"]
    outcomes = {o["id"]: o["achieved"] for o in debrief["outcomes"]}
    assert outcomes["rosc"] is True and outcomes["epinephrine"] is False

    # Split 500 + 500 mcg satisfies CONTINUE_CYCLES during persistent VF.
    split = WebSession({"scenario_id": "cardiac_arrest"})
    se, s_scen = split.engine, split.scenario
    assert s_scen is not None
    s_scen.events[0].apply(se)
    se.start()
    for _ in range(10):
        se.step(2.0)
    epi_step = _step(s_scen, "CONTINUE_CYCLES")
    se.give_drug_bolus("epi", 500)
    assert not epi_step.check_requirements(se)[0]
    se.give_drug_bolus("epi", 500)
    assert epi_step.check_requirements(se)[0]


def test_monitoring_edge_cases_and_oxygen_recovery_milestone(engine_factory, advance_time):
    """Covers weight-scaled baseline FiO2, unmet integer formatting, stale/failed NIBP, CPR MAP
    exclusion, position timeline logging, and post-failure O2 recovery milestone persistence."""
    obese_ox = engine_factory(
        config=SimulationConfig(mode="steady_state", rng_seed=1),
        start=True, age=40, weight=200, height=185, sex="male",
    )
    create_oxygen_supply_failure().prepare(obese_ox)
    advance_time(obese_ox, 25.0, dt=1.0)
    assert 0.34 <= obese_ox.state.fio2 <= 0.43
    assert obese_ox.state.sao2 >= 95.0
    assert "FiO2" not in obese_ox.state.alarms

    ana_scen = create_anaphylaxis_scenario()
    ana_eng = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1), start=True)
    ana_scen.prepare(ana_eng)
    assert 0.35 <= ana_eng.state.fio2 <= 0.45
    assert "volatile" not in _step(ana_scen, "EPINEPHRINE").instruction_for(ana_eng).lower()

    assert format_unmet_int(80.0, low=80) == 79
    assert format_unmet_int(45.3, low=35, high=45) == 46
    ana_eng.state.art_map = 64.8
    met_map, map_msg = require_map_at_least(65)(ana_eng)
    assert not met_map and "64/65+" in map_msg

    # Stale NIBP forces reassessment; failed shock NIBP satisfies onset hypotension check.
    no_art = engine_factory(config=SimulationConfig(arterial_line_enabled=False, rng_seed=1))
    no_art.state.nibp_map, no_art.state.nibp_timestamp, no_art.state.time, no_art.state.map = 75.0, 0.0, 120.0, 52.0
    met, msg = require_vasopressor_if_hypotensive("nore", fail_message="Start norepinephrine")(no_art)
    assert not met and "measure blood pressure again" in msg.lower()

    no_art.state.nibp_map, no_art.state.nibp_timestamp = 0.0, None
    start_map = CrisisStartMAP()
    start_map.record(no_art)
    assert start_map.value is None
    assert not require_hypotension_or_treatment(start_map, "hemorrhage", "Assess BP")(no_art)[0]
    no_art.state.nibp_measurement_failed, no_art.state.map = True, 22.0
    assert require_hypotension_or_treatment(start_map, "hemorrhage", "Assess BP")(no_art)[0] is True
    no_art.state.nibp_measurement_failed = False
    ok, msg = require_hypotension_or_treatment(start_map, "hemorrhage", "Assess BP")(no_art)
    assert not ok and "MAP: 0/" not in msg

    # Chest compressions never satisfy spontaneous MAP/stability checks; position logs in debrief.
    cpr_session = WebSession({"scenario_id": "hemorrhage_response", "practice_mode": "independent",
                              "sex": "female", "gestational_age_weeks": 39.0})
    cpr_session.command("position", '{"position": "left_lateral"}')
    cpr_eng = cpr_session.engine
    cpr_eng.set_rhythm("VFIB")
    cpr_eng.set_chest_compressions(True)
    cpr_eng.state.art_map = cpr_eng.state.map = 72.0
    cpr_eng.state.sao2 = 98.0
    assert not require_map_at_least(65)(cpr_eng)[0]
    assert not patient_stable(cpr_eng)
    assert cpr_session.case_run is not None
    cpr_session.case_run.observe(5.0, force=True)
    assert cpr_session.case_run.hypotension_s >= 5.0
    debrief = json.loads(cpr_session.command("case_finish", "{}"))["debrief"]
    assert any("Position: left lateral" in item["text"] for item in debrief["timeline"])

    # Instability produces a non-empty guidance hint.
    ana_eng.state.display_spo2 = ana_eng.state.sao2 = 93.0
    ana_eng.state.etco2 = 35.0
    ok, msg = require_crisis_resolved_with_map("hemorrhage")(ana_eng)
    assert not ok and bool(msg)

    # Pre-failure high O2 earns no recovery credit; post-backup recovery milestone persists.
    ox = WebSession({"scenario_id": "oxygen_supply_failure", "practice_mode": "independent"})
    assert ox.case_run is not None
    ox.engine.start()
    ox.engine.set_fgf(10.0, 0.0, 0.0)
    for _ in range(5):
        ox.case_run.before_step()
        ox.engine.step(1.0)
        ox.case_run.observe(1.0, force=True)
    assert ox.engine.circuit.composition.fio2 > 0.40 and "oxygen" not in ox.case_run.achieved_at
    for _ in range(30):
        ox.case_run.before_step()
        ox.engine.step(1.0)
        ox.case_run.observe(1.0, force=True)
    assert not ox.engine.circuit.oxygen_supply_connected and "oxygen" not in ox.case_run.achieved_at
    ox.engine.set_oxygen_supply_connected(True)
    ox.engine.circuit.composition.fio2, ox.engine.state.sao2 = 0.55, 98.0
    ox.case_run.observe(force=True)
    assert "oxygen" in ox.case_run.achieved_at
    ox.engine.circuit.composition.fio2 = 0.35
    ox.case_run.observe(force=True)
    assert "oxygen" in ox.case_run.achieved_at


def test_safety_review_covers_all_feedback_codes():
    """Every ActionFeedback code appears in SAFETY_REVIEW and surfaces in independent debriefs."""
    tree = ast.parse(Path("anasim/core/clinical_actions.py").read_text())
    feedback_codes = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ActionFeedback"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert feedback_codes <= set(SAFETY_REVIEW)

    indep = WebSession({"scenario_id": "induction_tiva", "practice_mode": "independent"})
    indep.engine.laryngospasm_severity = 0.9
    fb_resp = json.loads(indep.command("airway", json.dumps({"mode": "ETT"})))
    assert fb_resp["feedback"]["code"] == "laryngospasm_blocked"
    debrief = json.loads(indep.command("case_finish", "{}"))["debrief"]
    assert any("laryngospasm" in note.lower() for note in debrief["learning_points"])
