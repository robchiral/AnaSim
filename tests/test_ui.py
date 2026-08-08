import sys

import pytest
from PySide6.QtWidgets import QApplication

from anasim.core.drug_registry import DRUG_REGISTRY
from anasim.core.state import SimulationConfig
from anasim.ui.controls_widget import ControlPanelWidget
from anasim.ui.scenarios import (
    SCENARIO_BUILDERS,
    SCENARIO_REGISTRY,
    create_emergence,
    create_hemorrhage_response,
    create_induction_balanced,
    create_oxygen_supply_failure,
    create_sepsis_response,
)
from anasim.ui.tutorial_overlay import ScenarioOverlay

# Clinically reasonable responses for each objective of a scenario, used to
# confirm that every objective is still reachable through learner action.
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
        "INTUBATE": lambda e: e.set_airway_mode("ETT"),
    },
    "induction_balanced": {
        "APPLY_MASK": lambda e: e.set_airway_mode("Mask"),
        "SET_FGF_PREOX": lambda e: e.set_fgf(10.0, 0.0, 0.0),
        "INDUCE": lambda e: e.give_drug_bolus("propofol", 175),
        "MASK_VENTILATE": lambda e: e.set_bag_mask_ventilation(True),
        "GIVE_NMB": lambda e: e.give_drug_bolus("roc", 50),
        "INTUBATE": lambda e: e.set_airway_mode("ETT"),
        "MAINTENANCE": lambda e: (
            e.set_vaporizer("Sevoflurane", 2.0),
            e.set_fgf(2.0, 0.0, 0.0),
        ),
    },
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def _step(scenario, step_id):
    """Return a scenario step by id."""
    return next(step for step in scenario.steps if step.id == step_id)


def _activate(engine, step):
    """Activate a step the way the overlay does, without building the widget."""
    engine.actions.begin_step(step.id, engine.state.time)


def test_next_button_disabled_until_requirements_met(qapp, engine_factory):
    engine = engine_factory()
    overlay = ScenarioOverlay(create_induction_balanced(), engine)
    navigation = []
    overlay.navigate_requested.connect(navigation.append)

    engine.set_airway_mode("None")
    overlay.update_state()
    assert not overlay.btn_next.isEnabled()
    overlay.btn_target.click()
    assert navigation == ["Machine"]

    engine.set_airway_mode("Mask")
    overlay.update_state()
    assert overlay.btn_next.isEnabled()
    assert overlay.progress.value() > 0


def test_overlay_activates_each_objective_in_the_action_log(qapp, engine_factory):
    engine = engine_factory()
    overlay = ScenarioOverlay(create_induction_balanced(), engine)

    assert engine.actions.current_step.label == "APPLY_MASK"

    engine.set_airway_mode("Mask")
    overlay.update_state()
    overlay.btn_next.click()

    assert overlay.current_step == 1
    assert engine.actions.current_step.label == "SET_FGF_PREOX"
    assert engine.actions.current_step.time == engine.state.time


def test_fluid_objective_requires_a_bolus_given_during_the_objective(engine_factory):
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="tiva"),
        start=True,
    )
    step = _step(create_hemorrhage_response(), "GIVE_FLUIDS")

    engine.give_fluid(1000)  # Given before the objective appeared.
    _activate(engine, step)
    met, message = step.check_requirements(engine)
    assert not met
    assert "Give fluid bolus" in message

    engine.give_fluid(250)
    assert not step.check_requirements(engine)[0]
    assert "250/500 mL" in step.check_requirements(engine)[1]

    engine.give_fluid(250)
    assert step.check_requirements(engine)[0]


def test_vasopressor_objective_requires_dosing_during_the_objective(engine_factory):
    """Steady state starts with baseline norepinephrine support already running."""
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="tiva"),
        start=True,
    )
    step = _step(create_hemorrhage_response(), "START_VASOPRESSOR")

    engine.set_drug_rate("nore", 5.0)  # Started before the objective appeared.
    _activate(engine, step)
    met, message = step.check_requirements(engine)
    assert not met
    assert "Set the infusion rate or TCI target" in message

    engine.set_drug_rate("nore", 8.0)
    assert step.check_requirements(engine)[0]


def test_vasopressor_objective_needs_the_infusion_to_stay_running(engine_factory):
    engine = engine_factory(start=True)
    step = _step(create_hemorrhage_response(), "START_VASOPRESSOR")

    _activate(engine, step)
    engine.set_drug_rate("nore", 5.0)
    assert step.check_requirements(engine)[0]

    engine.set_drug_rate("nore", 0.0)
    assert not step.check_requirements(engine)[0]


def test_induction_objective_requires_a_bolus_during_the_objective(
    engine_factory, advance_time
):
    engine = engine_factory(start=True)
    step = _step(create_induction_balanced(), "INDUCE")

    engine.give_drug_bolus("propofol", 150)  # Given before the objective appeared.
    advance_time(engine, 20)
    assert engine.state.propofol_cp > 2.0

    _activate(engine, step)
    met, message = step.check_requirements(engine)
    assert not met
    assert "propofol" in message.lower()

    engine.give_drug_bolus("propofol", 20)
    assert step.check_requirements(engine)[0]


def test_machine_objectives_reject_actions_taken_before_activation(engine_factory):
    engine = engine_factory(start=True)
    scenario = create_induction_balanced()

    engine.set_fgf(10.0, 0.0, 0.0)
    fgf_step = _step(scenario, "SET_FGF_PREOX")
    _activate(engine, fgf_step)
    assert not fgf_step.check_requirements(engine)[0]
    engine.set_fgf(9.0, 0.0, 0.0)
    assert fgf_step.check_requirements(engine)[0]

    engine.set_bag_mask_ventilation(True)
    bag_step = _step(scenario, "MASK_VENTILATE")
    _activate(engine, bag_step)
    assert not bag_step.check_requirements(engine)[0]
    engine.set_bag_mask_ventilation(False)
    engine.set_bag_mask_ventilation(True)
    assert bag_step.check_requirements(engine)[0]

    engine.set_airway_mode("ETT")
    airway_step = _step(scenario, "INTUBATE")
    _activate(engine, airway_step)
    assert not airway_step.check_requirements(engine)[0]
    engine.set_airway_mode("Mask")
    engine.set_airway_mode("ETT")
    assert airway_step.check_requirements(engine)[0]


def test_crisis_objectives_reject_transitions_before_activation(engine_factory):
    engine = engine_factory(start=True)
    scenario = create_hemorrhage_response()

    engine.start_hemorrhage()
    start_step = _step(scenario, "START_HEMORRHAGE")
    _activate(engine, start_step)
    assert not start_step.check_requirements(engine)[0]
    engine.stop_hemorrhage()
    engine.start_hemorrhage()
    assert start_step.check_requirements(engine)[0]

    engine.stop_hemorrhage()
    stop_step = _step(scenario, "STOP_BLEEDING")
    _activate(engine, stop_step)
    assert not stop_step.check_requirements(engine)[0]
    engine.start_hemorrhage()
    engine.stop_hemorrhage()
    assert stop_step.check_requirements(engine)[0]


def test_emergence_objectives_reject_actions_taken_before_activation(engine_factory):
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced"),
        start=True,
    )
    scenario = create_emergence("balanced")

    engine.set_vaporizer("Sevoflurane", 0.0)
    engine.set_fgf(8.0, 0.0, 0.0)
    stop_step = _step(scenario, "STOP_AGENTS")
    _activate(engine, stop_step)
    assert not stop_step.check_requirements(engine)[0]
    engine.set_vaporizer("Sevoflurane", 0.0)
    engine.set_fgf(9.0, 0.0, 0.0)
    assert stop_step.check_requirements(engine)[0]

    engine.state.display_bis = 90.0
    engine.state.rr = 12.0
    engine.state.apnea = False
    engine.set_airway_mode("Mask")
    extubate_step = _step(scenario, "EXTUBATE")
    _activate(engine, extubate_step)
    assert not extubate_step.check_requirements(engine)[0]
    engine.set_airway_mode("ETT")
    engine.set_airway_mode("Mask")
    assert extubate_step.check_requirements(engine)[0]


def test_tiva_stop_objective_requires_each_stop_after_activation(engine_factory):
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="tiva"),
        start=True,
    )
    step = _step(create_emergence("tiva"), "STOP_AGENTS")

    engine.disable_tci("propofol")
    engine.disable_tci("remi")
    _activate(engine, step)
    assert not step.check_requirements(engine)[0]

    engine.set_drug_target("propofol", 3.0)
    engine.set_drug_target("remi", 2.0)
    engine.disable_tci("propofol")
    assert not step.check_requirements(engine)[0]
    engine.disable_tci("remi")
    assert step.check_requirements(engine)[0]


def test_oxygen_supply_objective_rejects_early_disconnect(engine_factory):
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="tiva"),
        start=True,
    )
    scenario = create_oxygen_supply_failure()
    scenario.prepare(engine)
    step = _step(scenario, "DISCONNECT_OXYGEN")

    engine.set_oxygen_supply_connected(False)
    _activate(engine, step)
    assert not step.check_requirements(engine)[0]

    engine.set_oxygen_supply_connected(True)
    engine.set_oxygen_supply_connected(False)
    assert step.check_requirements(engine)[0]


def test_sepsis_fluid_objective_requires_crystalloid(engine_factory):
    engine = engine_factory(start=True)
    step = _step(create_sepsis_response(), "GIVE_FLUIDS")
    _activate(engine, step)

    engine.give_blood(600)
    engine.give_albumin(500)
    assert not step.check_requirements(engine)[0]

    engine.give_fluid(500)
    assert step.check_requirements(engine)[0]


@pytest.mark.parametrize("scenario_id", sorted(SCENARIO_WALKTHROUGHS))
def test_scenario_objectives_stay_reachable(qapp, engine_factory, scenario_id):
    """Every objective must still be completable by acting while it is active."""
    spec = next(item for item in SCENARIO_REGISTRY if item.id == scenario_id)
    engine = engine_factory(
        config=SimulationConfig(mode=spec.start_mode, maint_type=spec.maint_type)
    )
    scenario = spec.builder()
    scenario.prepare(engine)
    overlay = ScenarioOverlay(scenario, engine)
    engine.start()

    actions = dict(SCENARIO_WALKTHROUGHS[scenario_id])
    sim_seconds = 0.0
    while overlay.current_step < len(scenario):
        step = scenario[overlay.current_step]
        action = actions.pop(step.id, None)
        if action is not None:
            action(engine)
        overlay.update_state()
        if overlay.requirements_met:
            overlay.btn_next.click()
            continue
        assert sim_seconds < 1200, (
            f"{scenario_id}/{step.id} unreachable: {overlay.check_requirements()[1]}"
        )
        engine.step(1.0)
        sim_seconds += 1.0

    assert overlay.btn_next.text() == "Complete"


def test_oxygen_supply_failure_scenario_progression(qapp, engine_factory):
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="tiva"),
        start=True,
    )
    scenario = create_oxygen_supply_failure()
    scenario.prepare(engine)

    assert scenario.id in SCENARIO_BUILDERS
    assert [step.id for step in scenario.steps] == [
        "CHECK_BASELINE",
        "DISCONNECT_OXYGEN",
        "RECOGNIZE_LOW_FIO2",
        "CONNECT_BACKUP_OXYGEN",
        "CONFIRM_RECOVERY",
    ]
    _activate(engine, scenario.steps[0])
    assert scenario.steps[0].check_requirements(engine)[0]

    _activate(engine, scenario.steps[1])
    engine.set_oxygen_supply_connected(False)
    assert scenario.steps[1].check_requirements(engine)[0]
    _activate(engine, scenario.steps[2])
    for _ in range(60):
        engine.step(1.0)
    assert scenario.steps[2].check_requirements(engine)[0]

    _activate(engine, scenario.steps[3])
    engine.set_oxygen_supply_connected(True)
    engine.set_fgf(10.0, 0.0, 0.0)
    assert scenario.steps[3].check_requirements(engine)[0]
    _activate(engine, scenario.steps[4])
    for _ in range(120):
        engine.step(1.0)
    assert scenario.steps[4].check_requirements(engine)[0]


def test_controls_are_generated_from_typed_registry(qapp, engine_factory):
    panel = ControlPanelWidget(engine_factory())

    assert tuple(panel.drug_widgets) == tuple(spec.key for spec in DRUG_REGISTRY)
    for spec in DRUG_REGISTRY:
        widgets = panel.drug_widgets[spec.key]
        assert widgets["rate"].suffix().strip() == spec.rate_unit
        assert widgets["target"].minimum() == spec.tci_range[0]
        assert widgets["target"].maximum() == spec.tci_range[1]
        assert widgets["bolus"].suffix().strip() == spec.bolus_unit
        assert widgets["bolus"].value() == spec.default_bolus


def test_controls_sync_external_engine_changes(qapp, engine_factory):
    engine = engine_factory()
    panel = ControlPanelWidget(engine)

    engine.set_airway_mode("ETT")
    engine.set_vent_settings(
        rr=10,
        vt=0.45,
        peep=8,
        ie="1:2",
        mode="PCV",
        p_insp=18,
    )
    engine.set_bronchospasm(0.4)
    engine.set_drug_rate("nore", 3.0)
    panel.sync_with_engine()

    assert panel.rb_ett.isChecked()
    assert panel.cb_vent_mode.currentData() == "PCV"
    assert panel.sb_rr.value() == 10
    assert panel.sb_peep.value() == 8
    assert panel.sb_pinsp.value() == 18
    assert panel.sb_bronchospasm.value() == 40
    assert panel.drug_widgets["nore"]["rate"].value() == 3.0


def test_hemorrhage_scenario_creation():
    scenario = create_hemorrhage_response()

    assert scenario.id == "hemorrhage_response"
    assert "Hemorrhage" in scenario.name
    assert [step.id for step in scenario.steps] == [
        "OBSERVE_BASELINE",
        "START_HEMORRHAGE",
        "RECOGNIZE_SHOCK",
        "GIVE_FLUIDS",
        "START_VASOPRESSOR",
        "STOP_BLEEDING",
        "REASSESS",
    ]
