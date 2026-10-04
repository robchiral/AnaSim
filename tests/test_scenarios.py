import pytest

from anasim.core.state import SimulationConfig
from anasim.scenarios import (
    SCENARIO_REGISTRY,
    create_emergence,
    create_hemorrhage_response,
    create_sepsis_response,
)
from anasim.web import WebSession

# Learner actions for each objective of every registered scenario.
SCENARIO_WALKTHROUGHS = {
    "hemorrhage_response": {
        "START_HEMORRHAGE": lambda e: e.start_hemorrhage(800.0),
        "GIVE_FLUIDS": lambda e: e.give_fluid(500),
        "START_VASOPRESSOR": lambda e: e.set_drug_rate("nore", 12.0),
        "STOP_BLEEDING": lambda e: e.stop_hemorrhage(),
        "REASSESS": lambda e: (
            e.give_blood(600),
            e.give_fluid(1000),
            e.set_drug_rate("nore", 25.0),
        ),
    },
    "sepsis_response": {
        "START_SEPSIS": lambda e: e.start_sepsis(),
        "GIVE_FLUIDS": lambda e: e.give_fluid(1000),
        "START_VASOPRESSOR": lambda e: e.set_drug_rate("nore", 15.0),
        "SOURCE_CONTROL": lambda e: e.stop_sepsis(),
    },
    "anaphylaxis_response": {
        "RECOGNIZE": lambda e: e.start_anaphylaxis(),
        "EPINEPHRINE": lambda e: e.give_drug_bolus("epi", 100),
        "FLUIDS": lambda e: e.give_fluid(500),
        "STABILIZE": lambda e: (e.stop_anaphylaxis(), e.set_drug_rate("epi", 6.0)),
    },
    "induction_tiva": {
        "APPLY_MASK": lambda e: e.set_airway_mode("Mask"),
        "SET_FGF_PREOX": lambda e: e.set_fgf(10.0, 0.0, 0.0),
        "START_ANALGESIA": lambda e: e.set_drug_target("remi", 4.0),
        "INDUCE": lambda e: (
            e.give_drug_bolus("propofol", 175),
            e.set_drug_target("propofol", 4.0),
        ),
        "MASK_VENTILATE": lambda e: e.set_bag_mask_ventilation(True),
        "GIVE_NMB": lambda e: e.give_drug_bolus("roc", 50),
        "INTUBATE": lambda e: (e.set_bag_mask_ventilation(False), e.set_airway_mode("ETT")),
        "CONFIRM_ETT": lambda e: e.set_vent_power(True),
        "MAINTENANCE": lambda e: e.set_fgf(2.0, 0.0, 0.0),
    },
    "induction_balanced": {
        "APPLY_MASK": lambda e: e.set_airway_mode("Mask"),
        "SET_FGF_PREOX": lambda e: e.set_fgf(10.0, 0.0, 0.0),
        "INDUCE": lambda e: e.give_drug_bolus("propofol", 175),
        "MASK_VENTILATE": lambda e: e.set_bag_mask_ventilation(True),
        "GIVE_NMB": lambda e: e.give_drug_bolus("roc", 50),
        "INTUBATE": lambda e: (e.set_bag_mask_ventilation(False), e.set_airway_mode("ETT")),
        "CONFIRM_ETT": lambda e: e.set_vent_power(True),
        "MAINTENANCE": lambda e: (
            e.set_vaporizer("Sevoflurane", 2.0),
            e.set_fgf(2.0, 0.0, 0.0),
        ),
    },
    "emergence_tiva": {
        "STOP_AGENTS": lambda e: (e.disable_tci("propofol"), e.disable_tci("remi")),
        "EXTUBATE": lambda e: (e.set_vent_power(False), e.set_airway_mode("Mask")),
    },
    "emergence_balanced": {
        "STOP_AGENTS": lambda e: (
            e.set_vaporizer("Sevoflurane", 0.0),
            e.disable_tci("remi"),
            e.set_fgf(10.0, 0.0, 0.0),
        ),
        "EXTUBATE": lambda e: (e.set_vent_power(False), e.set_airway_mode("Mask")),
    },
    "oxygen_supply_failure": {
        "DISCONNECT_OXYGEN": lambda e: e.set_oxygen_supply_connected(False),
        "CONNECT_BACKUP_OXYGEN": lambda e: (
            e.set_oxygen_supply_connected(True),
            e.set_fgf(10.0, 0.0, 0.0),
        ),
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
    assert engine.state.propofol_cp > 2.0


def _early_extubation(engine):
    # Meet the awake criteria so only the airway action is missing.
    engine.state.display_bis = 90.0
    engine.resp.state.rr = 12.0
    engine.resp.state.vt = 500.0
    engine.resp.state.apnea = False
    engine.state.tof = 100.0
    engine.state.display_spo2 = 98.0
    engine.state.spo2_signal_valid = True
    engine.set_airway_mode("Mask")


def _stop_tiva(engine):
    engine.disable_tci("propofol")
    engine.disable_tci("remi")


# (scenario, objective, action taken before the objective, the same action repeated during it)
EARLY_ACTIONS = [
    ("hemorrhage_response", "START_HEMORRHAGE",
     lambda e: e.start_hemorrhage(),
     lambda e: (e.stop_hemorrhage(), e.start_hemorrhage())),
    ("hemorrhage_response", "GIVE_FLUIDS",
     lambda e: e.give_fluid(1000),
     lambda e: e.give_fluid(500)),
    ("hemorrhage_response", "START_VASOPRESSOR",
     lambda e: e.set_drug_rate("nore", 5.0),
     lambda e: e.set_drug_rate("nore", 8.0)),
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
     lambda e: e.set_bag_mask_ventilation(True),
     lambda e: (e.set_bag_mask_ventilation(False), e.set_bag_mask_ventilation(True))),
    ("induction_balanced", "INTUBATE",
     lambda e: e.set_airway_mode("ETT"),
     lambda e: (e.set_airway_mode("Mask"), e.set_airway_mode("ETT"))),
    ("emergence_balanced", "STOP_AGENTS",
     lambda e: (e.set_vaporizer("Sevoflurane", 0.0), e.disable_tci("remi"), e.set_fgf(8.0, 0.0, 0.0)),
     lambda e: (e.set_vaporizer("Sevoflurane", 0.0), e.set_drug_target("remi", 2.0),
                e.disable_tci("remi"), e.set_fgf(9.0, 0.0, 0.0))),
    ("emergence_balanced", "EXTUBATE",
     _early_extubation,
     lambda e: (e.set_airway_mode("ETT"), e.set_airway_mode("Mask"))),
    ("emergence_tiva", "STOP_AGENTS",
     _stop_tiva,
     lambda e: (e.set_drug_target("propofol", 3.0), e.set_drug_target("remi", 2.0), _stop_tiva(e))),
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


def test_vasopressor_objective_needs_the_infusion_to_stay_running(engine_factory):
    engine = engine_factory(start=True)
    step = _step(create_hemorrhage_response(), "START_VASOPRESSOR")

    _activate(engine, step)
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


def test_extubation_waits_for_tof_ratio_90(engine_factory):
    """Residual block impairs the upper airway after breathing returns (Eikermann 2003)."""
    engine = engine_factory(start=True)
    step = _step(create_emergence("tiva"), "EXTUBATE")
    _activate(engine, step)
    _early_extubation(engine)
    engine.state.tof = 89.0
    assert not step.check_requirements(engine)[0]
    engine.state.tof = 90.0
    assert step.check_requirements(engine)[0]


def test_preoxygenation_waits_for_lung_washin(engine_factory):
    spec = next(spec for spec in SCENARIO_REGISTRY if spec.id == "induction_tiva")
    step = _step(spec.builder(), "PREOXYGENATE")
    engine = engine_factory(start=True)
    engine.set_airway_mode("Mask")
    engine.set_fgf(10.0, 0.0)
    _activate(engine, step)
    for _ in range(60):
        engine.step(1.0)
    assert not step.check_requirements(engine)[0]
    for _ in range(180):
        engine.step(1.0)
    assert step.check_requirements(engine)[0]

    # The flowmeter stays set, but no O2 is delivered.
    engine.set_oxygen_supply_connected(False)
    assert not step.check_requirements(engine)[0]


@pytest.mark.parametrize("spec", SCENARIO_REGISTRY, ids=lambda spec: spec.id)
def test_scenario_objectives_stay_reachable(spec):
    """Every objective must still be completable by acting while it is active."""
    scenario_id = spec.id
    session = WebSession({"scenario_id": scenario_id})
    engine = session.engine
    scenario = session.scenario
    engine.start()

    actions = dict(SCENARIO_WALKTHROUGHS[scenario_id])
    sim_seconds = 0.0
    while session.step_index < len(scenario):
        step = scenario[session.step_index]
        action = actions.pop(step.id, None)
        if action is not None:
            action(engine)
        session.snapshot()
        if session.step_met:
            session.command("scenario_next")
            continue
        assert sim_seconds < 1200, (
            f"{scenario_id}/{step.id} unreachable: {session.step_status}"
        )
        engine.step(1.0)
        sim_seconds += 1.0

    assert session.snapshot()["scenario"]["complete"]
