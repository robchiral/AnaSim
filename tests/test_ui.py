import sys

import pytest
from PySide6.QtWidgets import QApplication

from anasim.core.drug_registry import DRUG_REGISTRY
from anasim.ui.controls_widget import ControlPanelWidget
from anasim.ui.scenarios import create_hemorrhage_response, create_induction_balanced
from anasim.ui.tutorial_overlay import ScenarioOverlay


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def test_next_button_disabled_until_requirements_met(qapp, engine_factory):
    engine = engine_factory()
    overlay = ScenarioOverlay(create_induction_balanced())

    engine.set_airway_mode("None")
    overlay.update_state(engine)
    assert not overlay.btn_next.isEnabled()

    engine.set_airway_mode("Mask")
    overlay.update_state(engine)
    assert overlay.btn_next.isEnabled()


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
