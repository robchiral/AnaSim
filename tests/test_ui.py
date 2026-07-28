import sys

import pytest
from PySide6.QtWidgets import QApplication

from anasim.core.drug_registry import DRUG_REGISTRY
from anasim.core.state import SimulationConfig
from anasim.ui.controls_widget import ControlPanelWidget
from anasim.ui.scenarios import (
    SCENARIO_BUILDERS,
    create_hemorrhage_response,
    create_induction_balanced,
    create_oxygen_supply_failure,
)
from anasim.ui.tutorial_overlay import ScenarioOverlay


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def test_next_button_disabled_until_requirements_met(qapp, engine_factory):
    engine = engine_factory()
    overlay = ScenarioOverlay(create_induction_balanced())
    navigation = []
    overlay.navigate_requested.connect(navigation.append)

    engine.set_airway_mode("None")
    overlay.update_state(engine)
    assert not overlay.btn_next.isEnabled()
    overlay.btn_target.click()
    assert navigation == ["Machine"]

    engine.set_airway_mode("Mask")
    overlay.update_state(engine)
    assert overlay.btn_next.isEnabled()
    assert overlay.progress.value() > 0


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
    assert scenario.steps[0].check_requirements(engine)[0]

    engine.set_oxygen_supply_connected(False)
    assert scenario.steps[1].check_requirements(engine)[0]
    for _ in range(60):
        engine.step(1.0)
    assert scenario.steps[2].check_requirements(engine)[0]

    engine.set_oxygen_supply_connected(True)
    engine.set_fgf(10.0, 0.0, 0.0)
    assert scenario.steps[3].check_requirements(engine)[0]
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
