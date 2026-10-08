import json

import pytest

from anasim.core.state import SimulationConfig
from anasim.scenarios import (
    SCENARIO_REGISTRY,
    create_emergence,
    create_hemorrhage_response,
    create_sepsis_response,
)
from anasim.web import WebSession

# Browser commands for each objective; TCI targets become manual rates when disabled.
SCENARIO_WALKTHROUGHS = {
    "hemorrhage_response": {
        "START_HEMORRHAGE": [("hemorrhage", {"active": True, "rate_ml_min": 800})],
        "GIVE_FLUIDS": [("fluid", {"kind": "crystalloid", "volume_ml": 500})],
        "START_VASOPRESSOR": [("drug_rate", {"key": "nore", "rate": 12})],
        "STOP_BLEEDING": [("hemorrhage", {"active": False})],
    },
    "sepsis_response": {
        "START_SEPSIS": [("sepsis", {"active": True})],
        "GIVE_FLUIDS": [("fluid", {"kind": "crystalloid", "volume_ml": 1000})],
        "START_VASOPRESSOR": [("drug_rate", {"key": "nore", "rate": 15})],
        "SOURCE_CONTROL": [("sepsis", {"active": False})],
    },
    "anaphylaxis_response": {
        "RECOGNIZE": [("anaphylaxis", {"active": True})],
        "EPINEPHRINE": [
            ("drug_bolus", {"key": "epi", "amount": 100}),
            ("fgf", {"o2": 10, "air": 0, "n2o": 0}),
        ],
        "FLUIDS": [("fluid", {"kind": "crystalloid", "volume_ml": 1000})],
        "STABILIZE": [
            ("anaphylaxis", {"active": False}),
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
        "MAINTENANCE": [("fgf", {"o2": 2, "air": 0, "n2o": 0})],
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
            ("vaporizer", {"percent": 3}),
            ("fgf", {"o2": 1, "air": 1, "n2o": 0}),
        ],
    },
    "emergence_tiva": {
        "ASSESS": [("drug_bolus", {"key": "phenyl", "amount": 100})],
        "STOP_AGENTS": [
            ("drug_rate", {"key": "propofol", "rate": 0}),
            ("drug_rate", {"key": "remi", "rate": 0}),
        ],
        "EXTUBATION_READINESS": [("vent_power", {"on": False})],
        "EXTUBATE": [("airway", {"mode": "Mask"})],
        "RECOVERY": [("fgf", {"o2": 5, "air": 0, "n2o": 0})],
    },
    "emergence_balanced": {
        "STOP_AGENTS": [
            ("vaporizer", {"percent": 0}),
            ("drug_rate", {"key": "remi", "rate": 0}),
            ("fgf", {"o2": 10, "air": 0, "n2o": 0}),
        ],
        "EXTUBATION_READINESS": [("vent_power", {"on": False})],
        "EXTUBATE": [("airway", {"mode": "Mask"})],
        "RECOVERY": [("fgf", {"o2": 5, "air": 0, "n2o": 0})],
    },
    "oxygen_supply_failure": {
        "DISCONNECT_OXYGEN": [("oxygen_supply", {"connected": False})],
        "CONNECT_BACKUP_OXYGEN": [
            ("oxygen_supply", {"connected": True}),
            ("fgf", {"o2": 10, "air": 0, "n2o": 0}),
        ],
    },
}


def _step(scenario, step_id):
    return next(step for step in scenario.steps if step.id == step_id)


def _activate(engine, step):
    engine.actions.begin_step(step.id, engine.state.time)


def _early_induction(engine):
    engine.give_drug_bolus("propofol", 150)
    for _ in range(20):
        engine.step(1.0)


def _mask_ventilate(engine):
    engine.set_airway_mode("Mask")
    engine.set_bag_mask_ventilation(True)
    for _ in range(10):
        engine.step(1.0)


def _early_extubation(engine):
    # Meet the awake criteria so only the airway action is missing.
    engine.state.display_bis = 90.0
    engine.resp.state.rr = 12.0
    engine.resp.state.vt = 500.0
    engine.state.rr = 12.0
    engine.state.vt = 500.0
    engine.resp.state.apnea = False
    engine.state.tof = 100.0
    engine.state.display_spo2 = 98.0
    engine.state.spo2_signal_valid = True
    engine.set_vent_power(False)
    engine.set_airway_mode("Mask")


def _stop_tiva(engine):
    engine.set_drug_rate("propofol", 0.0)
    engine.set_drug_rate("remi", 0.0)


# (scenario, objective, action taken before the objective, the same action repeated during it)
EARLY_ACTIONS = [
    ("hemorrhage_response", "START_HEMORRHAGE",
     lambda e: e.start_hemorrhage(),
     lambda e: (e.stop_hemorrhage(), e.start_hemorrhage())),
    ("hemorrhage_response", "GIVE_FLUIDS",
     lambda e: e.give_fluid(1000),
     lambda e: e.give_fluid(500)),
    ("hemorrhage_response", "STOP_BLEEDING",
     lambda e: (e.start_hemorrhage(), e.stop_hemorrhage()),
     lambda e: (e.start_hemorrhage(), e.stop_hemorrhage())),
    ("induction_balanced", "SET_FGF_PREOX",
     lambda e: e.set_fgf(10.0, 0.0, 0.0),
     lambda e: e.set_fgf(9.0, 0.0, 0.0)),
    ("induction_balanced", "INDUCE",
     _early_induction,
     lambda e: e.give_drug_bolus("propofol", 20)),
    ("induction_balanced", "MASK_VENTILATE",
     _mask_ventilate,
     lambda e: (e.set_bag_mask_ventilation(False), _mask_ventilate(e))),
    ("induction_balanced", "INTUBATE",
     lambda e: e.set_airway_mode("ETT"),
     lambda e: (e.set_airway_mode("Mask"), e.set_airway_mode("ETT"))),
    ("emergence_balanced", "STOP_AGENTS",
     lambda e: (e.set_vaporizer(0.0), e.set_drug_rate("remi", 0.0), e.set_fgf(8.0, 0.0, 0.0)),
     lambda e: (e.set_vaporizer(0.0), e.set_drug_rate("remi", 0.1),
                e.set_drug_rate("remi", 0.0), e.set_fgf(9.0, 0.0, 0.0))),
    ("emergence_balanced", "EXTUBATE",
     _early_extubation,
     lambda e: (e.set_airway_mode("ETT"), e.set_airway_mode("Mask"))),
    ("emergence_tiva", "STOP_AGENTS",
     _stop_tiva,
     lambda e: (e.set_drug_rate("propofol", 100.0), e.set_drug_rate("remi", 0.1), _stop_tiva(e))),
    ("oxygen_supply_failure", "DISCONNECT_OXYGEN",
     lambda e: e.set_oxygen_supply_connected(False),
     lambda e: (e.set_oxygen_supply_connected(True), e.set_oxygen_supply_connected(False))),
]


@pytest.mark.parametrize(
    ("scenario_id", "step_id", "early", "repeat"),
    EARLY_ACTIONS,
    ids=[f"{scenario_id}-{step_id}" for scenario_id, step_id, *_ in EARLY_ACTIONS],
)
def test_objectives_ignore_actions_taken_before_activation(
    engine_factory, scenario_id, step_id, early, repeat
):
    spec = next(spec for spec in SCENARIO_REGISTRY if spec.id == scenario_id)
    engine = engine_factory(
        config=SimulationConfig(mode=spec.start_mode, maint_type=spec.maint_type)
    )
    scenario = spec.builder()
    scenario.prepare(engine)
    engine.start()
    step = _step(scenario, step_id)

    early(engine)
    _activate(engine, step)
    assert not step.check_requirements(engine)[0]

    repeat(engine)
    assert step.check_requirements(engine)[0]


@pytest.mark.parametrize("builder,event", [
    (create_hemorrhage_response, "hemorrhage"),
    (create_sepsis_response, "sepsis"),
])
def test_vasopressor_objective_tracks_pressure_and_current_support(engine_factory, builder, event):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="balanced"), start=True)
    step = _step(builder(), "START_VASOPRESSOR")
    _activate(engine, step)
    # Recovered pressure does not justify another vasopressor.
    assert step.check_requirements(engine)[0]

    getattr(engine, f"start_{event}")()
    for _ in range(600):
        engine.step(1.0)
        if engine.state.monitored_blood_pressure(True)[2] < 65:
            break
    assert engine.state.monitored_blood_pressure(True)[2] < 65
    engine.set_drug_rate("nore", 5.0)

    _activate(engine, step)
    assert not step.check_requirements(engine)[0]
    if event == "hemorrhage":
        engine.set_drug_rate("phenyl", 20.0)
        assert step.check_requirements(engine)[0]
        engine.set_drug_rate("phenyl", 0.0)
        # The earlier norepinephrine infusion cannot stand in for stopped phenylephrine.
        assert not step.check_requirements(engine)[0]
    engine.set_drug_rate("nore", 5.0)
    assert step.check_requirements(engine)[0]

    engine.set_drug_rate("nore", 0.0)
    assert not step.check_requirements(engine)[0]


def test_sepsis_fluid_objective_requires_crystalloid(engine_factory):
    engine = engine_factory(start=True)
    step = _step(create_sepsis_response(), "GIVE_FLUIDS")
    _activate(engine, step)

    engine.give_blood(600)
    engine.give_albumin(500)
    assert not step.check_requirements(engine)[0]

    engine.give_fluid(500)
    assert step.check_requirements(engine)[0]


@pytest.mark.parametrize("weight", [70, 100])
def test_extubation_requires_recovered_block_and_spontaneous_ventilation(engine_factory, weight):
    """Residual block impairs the upper airway after breathing returns (Eikermann 2003)."""
    engine = engine_factory(start=True, weight=weight, height=177)
    step = _step(create_emergence("tiva"), "EXTUBATE")
    _activate(engine, step)
    _early_extubation(engine)
    engine.state.tof = 89.0
    assert not step.check_requirements(engine)[0]
    engine.state.tof = 90.0
    engine.set_vent_power(True)
    assert not step.check_requirements(engine)[0]
    engine.set_vent_power(False)
    assert step.check_requirements(engine)[0]
    engine.set_bag_mask_ventilation(True)
    assert not step.check_requirements(engine)[0]
    engine.set_bag_mask_ventilation(False)
    engine.state.vt = 200.0
    assert not step.check_requirements(engine)[0]


@pytest.mark.parametrize("assisted", [False, True], ids=["tidal-breathing", "assisted-higher-bmi"])
def test_preoxygenation_waits_for_lung_washin(engine_factory, assisted):
    spec = next(spec for spec in SCENARIO_REGISTRY if spec.id == "induction_tiva")
    step = _step(spec.builder(), "PREOXYGENATE")
    patient = dict(age=70, weight=100, height=177) if assisted else {}
    engine = engine_factory(start=True, **patient)
    engine.set_airway_mode("Mask")
    engine.set_fgf(10.0, 0.0)
    _activate(engine, step)
    for _ in range(60):
        engine.step(1.0)
    assert not step.check_requirements(engine)[0]
    for _ in range(180):
        engine.step(1.0)
    if assisted:
        # A high inspired fraction alone does not establish adequate preoxygenation.
        assert engine.state.fio2 > 0.95
        assert not step.check_requirements(engine)[0]
        engine.set_vent_settings(rr=16, vt=0.5, peep=5, ie="1:2", mode="VCV")
        engine.set_vent_power(True)
        for _ in range(300):
            engine.step(1.0)
    assert engine.state.et_o2 >= 90
    assert step.check_requirements(engine)[0]

    # The flowmeter stays set, but no O2 is delivered.
    engine.set_oxygen_supply_connected(False)
    assert not step.check_requirements(engine)[0]


def test_intubation_objective_starts_one_laryngoscopy_stimulus():
    session = WebSession({"scenario_id": "induction_balanced"})
    engine = session.engine
    engine.start()
    session.step_index = next(i for i, step in enumerate(session.scenario) if step.id == "INTUBATE")
    session._begin_step()

    engine.set_airway_mode("ETT")
    session.snapshot()
    assert engine.disturbance_active and engine.disturbance_profile == "stim_intubation_pulse"
    started = engine.disturbance_start_time
    for _ in range(10):
        engine.step(1.0)
        session.snapshot()
    assert engine.disturbance_start_time == started


@pytest.mark.parametrize("spec", SCENARIO_REGISTRY, ids=lambda spec: spec.id)
@pytest.mark.parametrize("tci_enabled", [False, True], ids=["manual", "tci"])
def test_guided_scenario_completes_through_browser_commands(spec, tci_enabled):
    """Every guided objective must remain reachable through the browser protocol."""
    session = WebSession({"scenario_id": spec.id, "tci_enabled": tci_enabled})
    info = json.loads(session.info())
    session.command("run", '{"running": true}')
    session.command("speed", '{"value": 5}')
    pending = dict(SCENARIO_WALKTHROUGHS[spec.id])
    samples = 0
    snap = json.loads(session.advance(0.0))

    # Continue does nothing until the current objective is met.
    if not snap["scenario"]["met"]:
        session.command("scenario_next")
        assert json.loads(session.advance(0.0))["scenario"]["id"] == snap["scenario"]["id"]

    while not snap["scenario"]["complete"]:
        step = snap["scenario"]
        if spec.id == "induction_tiva":
            uses_tci_guidance = tci_enabled and step["id"] in {"START_ANALGESIA", "INDUCE"}
            assert ("TCI" in step["instruction"]) is uses_tci_guidance
            if not tci_enabled:
                assert "µg/mL" not in step["instruction"] and "ng/mL" not in step["instruction"]
        for name, args in pending.pop(step["id"], []):
            if name == "drug_target" and not tci_enabled:
                name = "drug_rate"
                key = args["key"]
                args = {"key": key, "rate": {"propofol": 150, "remi": 0.2}[key]}
            session.command(name, json.dumps(args))
        if json.loads(session.advance(0.0))["scenario"]["met"]:
            session.command("scenario_next")
            snap = json.loads(session.advance(0.0))
        else:
            assert snap["time"] < 1200, f"{spec.id}/{step['id']} unreachable: {step['status']}"
            snap = json.loads(session.advance(0.2))
            samples += len(snap["waves"]["ecg"])

    assert not pending
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
        assert snap["vitals"]["tof"] < 25
