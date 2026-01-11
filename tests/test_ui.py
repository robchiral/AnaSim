
import sys
import os
import unittest
from PySide6.QtWidgets import QApplication

# Ensure src path is in sys.path if not running from root with module (pytest typically handles this).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anasim.ui.tutorial_overlay import ScenarioOverlay
from anasim.ui.scenarios import create_induction_balanced
from anasim.core.engine import SimulationEngine, SimulationConfig, Patient

# Helper to get QApp
def get_qapp():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    return app

class TestTutorialOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_qapp()

    def setUp(self):
        self.patient = Patient(40, 70, 170, 'male')
        self.config = SimulationConfig()
        self.engine = SimulationEngine(self.patient, self.config)
        
    def test_next_button_disabled_until_requirements_met(self):
        """Test that Next button is disabled until step requirements are met."""
        overlay = ScenarioOverlay(create_induction_balanced())
        
        # Start with no airway so first step is not already met
        self.engine.set_airway_mode("None")
        
        # Initially button should be disabled
        overlay.update_state(self.engine)
        self.assertFalse(overlay.btn_next.isEnabled())
        
        # After meeting requirement, button should enable
        self.engine.set_airway_mode("Mask")
        overlay.update_state(self.engine)
        self.assertTrue(overlay.btn_next.isEnabled())


class TestHemorrhageScenario(unittest.TestCase):
    """Test the hemorrhage response scenario."""
    
    @classmethod
    def setUpClass(cls):
        cls.app = get_qapp()

    def setUp(self):
        self.patient = Patient(40, 70, 170, 'male')
        self.config = SimulationConfig(mode='steady_state')
        self.engine = SimulationEngine(self.patient, self.config)
        
    def test_hemorrhage_scenario_creation(self):
        """Test that hemorrhage scenario can be created and has correct steps."""
        from anasim.ui.scenarios import create_hemorrhage_response
        from anasim.ui.tutorial_overlay import ScenarioOverlay
        
        scenario = create_hemorrhage_response()
        
        # Verify scenario metadata
        self.assertEqual(scenario.id, "hemorrhage_response")
        self.assertIn("Hemorrhage", scenario.name)
        
        # Verify 7 steps
        self.assertEqual(len(scenario), 7)
        
        # Verify step IDs
        expected_steps = [
            "OBSERVE_BASELINE", "START_HEMORRHAGE", "RECOGNIZE_SHOCK",
            "GIVE_FLUIDS", "START_VASOPRESSOR", "STOP_BLEEDING", "REASSESS"
        ]
        actual_steps = [step.id for step in scenario.steps]
        self.assertEqual(actual_steps, expected_steps)
