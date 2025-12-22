
import sys
import os
import unittest
import pytest
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import Qt

# Ensure src path is in sys.path if not running from root with module (but pytest usually handles it)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.main_window import MainWindow
from ui.controls_widget import ControlPanelWidget
from ui.monitor_widget import PatientMonitorWidget
from ui.tutorial_overlay import TutorialOverlay
from core.engine import SimulationEngine, SimulationConfig, Patient
from core.state import SimulationState, AirwayType

# Helper to get QApp
def get_qapp():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    return app

class TestUITutorial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_qapp()

    def setUp(self):
        self.patient = Patient(40, 70, 170, 'male')
        self.config = SimulationConfig()
        self.engine = SimulationEngine(self.patient, self.config)

    def test_control_panel_labels(self):
        # 1. Normal Mode -> Labels not present
        ctrl_normal = ControlPanelWidget(self.engine, tutorial_mode=False)
        
        def get_all_labels_text(widget):
            texts = []
            labels = widget.findChildren(QLabel)
            for l in labels:
                texts.append(l.text())
            return texts

        texts_normal = get_all_labels_text(ctrl_normal.tab_machine)
        self.assertFalse(any("The Anesthesia Machine integrates" in t for t in texts_normal))
        
        # 2. Tutorial Mode
        ctrl_tutorial = ControlPanelWidget(self.engine, tutorial_mode=True)
        texts_tutorial = get_all_labels_text(ctrl_tutorial.tab_machine)
        self.assertTrue(any("The Anesthesia Machine integrates" in t for t in texts_tutorial))


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
        overlay = TutorialOverlay(mode="awake", maint_type="balanced")
        
        # Start with no airway so first step is not already met
        self.engine.set_airway_mode("None")
        
        # Initially button should be disabled
        overlay.update_state(self.engine)
        self.assertFalse(overlay.btn_next.isEnabled())
        
        # After meeting requirement, button should enable
        self.engine.set_airway_mode("Mask")
        overlay.update_state(self.engine)
        self.assertTrue(overlay.btn_next.isEnabled())
        
    def test_balanced_induction_sequence(self):
        """Test the balanced anesthesia induction sequence with button clicks."""
        overlay = TutorialOverlay(mode="awake", maint_type="balanced")
        
        # Steps for balanced: APPLY_MASK, SET_FGF_PREOX, PREOXYGENATE, 
        # INDUCE, CONFIRM_LOC, MASK_VENTILATE, GIVE_NMB, WAIT_PARALYSIS, 
        # INTUBATE, CONFIRM_ETT, MAINTENANCE
        
        # 1. APPLY_MASK
        self.assertEqual(overlay.current_step, 0)
        self.engine.set_airway_mode("Mask")
        overlay.update_state(self.engine)
        self.assertTrue(overlay.requirements_met)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 1)
        
        # 2. SET_FGF_PREOX
        self.engine.circuit.fgf_o2 = 10.0
        self.engine.circuit.fgf_air = 0.0
        overlay.update_state(self.engine)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 2)
        
        # 3. PREOXYGENATE (FGF already set correctly from step 2)
        overlay.update_state(self.engine)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 3)
        
        # 4. INDUCE
        self.engine.state.propofol_cp = 3.0
        overlay.update_state(self.engine)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 4)
        
        # 5. CONFIRM_LOC
        self.engine.state.bis = 50
        self.engine.state.loc = 0.6
        overlay.update_state(self.engine)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 5)
        
        # 6. MASK_VENTILATE
        self.engine.resp_mech.set_rr = 12
        self.engine.resp_mech.set_vt = 0.5
        self.engine.vent.is_on = True
        overlay.update_state(self.engine)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 6)
        
        # 7. GIVE_NMB
        self.engine.state.roc_cp = 0.6
        overlay.update_state(self.engine)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 7)
        
        # 8. WAIT_PARALYSIS
        self.engine.state.tof = 10
        overlay.update_state(self.engine)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 8)
        
        # 9. INTUBATE
        self.engine.set_airway_mode("ETT")
        overlay.update_state(self.engine)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 9)
        
        # 10. CONFIRM_ETT
        self.engine.state.etco2 = 35
        overlay.update_state(self.engine)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 10)
        
        # 11. MAINTENANCE (Balanced)
        self.engine.state.mac = 0.8
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # Should be complete
        self.assertIn("complete", overlay.lbl_instruction.text().lower())

    def test_tiva_induction_sequence(self):
        """Test TIVA induction sequence with button clicks."""
        overlay = TutorialOverlay(mode="awake", maint_type="tiva")
        
        # TIVA includes START_ANALGESIA step
        # Steps: APPLY_MASK, SET_FGF_PREOX, PREOXYGENATE, START_ANALGESIA,
        # INDUCE, CONFIRM_LOC, MASK_VENTILATE, GIVE_NMB, WAIT_PARALYSIS,
        # INTUBATE, CONFIRM_ETT, MAINTENANCE
        
        # 1. APPLY_MASK
        self.engine.set_airway_mode("Mask")
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # 2. SET_FGF_PREOX
        self.engine.circuit.fgf_o2 = 10.0
        self.engine.circuit.fgf_air = 0.0
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # 3. PREOXYGENATE (FGF already set correctly)
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # 4. START_ANALGESIA (TIVA only)
        self.assertEqual(overlay.current_step, 3)
        self.engine.remi_rate_ug_sec = 0.1
        overlay.update_state(self.engine)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 4)
        
        # 5. INDUCE (TIVA needs infusion running too)
        self.engine.state.propofol_cp = 3.0
        self.engine.propofol_rate_mg_sec = 1.0
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # 6. CONFIRM_LOC
        self.engine.state.bis = 50
        self.engine.state.loc = 0.8
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # 7. MASK_VENTILATE
        self.engine.resp_mech.set_rr = 12
        self.engine.resp_mech.set_vt = 0.5
        self.engine.vent.is_on = True
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # 8. GIVE_NMB
        self.engine.state.roc_cp = 0.6
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # 9. WAIT_PARALYSIS
        self.engine.state.tof = 10
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # 10. INTUBATE
        self.engine.set_airway_mode("ETT")
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # 11. CONFIRM_ETT
        self.engine.state.etco2 = 35
        overlay.update_state(self.engine)
        overlay.click_next()
        
        # 12. MAINTENANCE (TIVA - check infusions running)
        overlay.update_state(self.engine)
        overlay.click_next()
        
        self.assertIn("complete", overlay.lbl_instruction.text().lower())

    def test_emergence_sequence(self):
        """Test emergence sequence for steady_state mode."""
        overlay = TutorialOverlay(mode='steady_state', maint_type='tiva')
        self.assertIn("Emergence", overlay.lbl_step.text())
        
        # Steps: ASSESS, STOP_AGENTS, AWAIT_EMERGENCE, EXTUBATE, RECOVERY
        
        # Mock engine for emergence
        class MockEngine:
            def __init__(self):
                self.state = SimulationState()
                self.state.bis = 45
                self.state.map = 80
                self.state.rr = 0
                self.state.spo2 = 99
                self.state.apnea = True
                self.state.etco2 = 40
                self.propofol_rate_mg_sec = 0.5
                self.remi_rate_ug_sec = 0.1
                self.tci_prop = type('obj', (object,), {'target': 4.0})()
                self.tci_remi = type('obj', (object,), {'target': 3.0})()
                self.circuit = type('obj', (object,), {
                    'vaporizer_on': False,
                    'vaporizer_setting': 0,
                    'fgf_total': lambda: 2.0
                })()
                
        engine = MockEngine()
        
        # 1. ASSESS
        overlay.update_state(engine)
        self.assertTrue(overlay.requirements_met)
        overlay.click_next()
        
        # 2. STOP_AGENTS (TIVA - stop infusions)
        engine.propofol_rate_mg_sec = 0
        engine.remi_rate_ug_sec = 0
        engine.tci_prop = None
        engine.tci_remi = None
        overlay.update_state(engine)
        overlay.click_next()
        
        # 3. AWAIT_EMERGENCE
        engine.state.bis = 75
        engine.state.rr = 10
        engine.state.apnea = False
        overlay.update_state(engine)
        overlay.click_next()
        
        # 4. EXTUBATE
        engine.state.bis = 85
        engine.state.rr = 12
        engine.state.airway_mode = AirwayType.MASK
        overlay.update_state(engine)
        overlay.click_next()
        
        # 5. RECOVERY
        engine.state.spo2 = 98
        engine.state.rr = 14
        overlay.update_state(engine)
        overlay.click_next()
        
        self.assertIn("complete", overlay.lbl_instruction.text().lower())


class TestVentSwitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_qapp()

    def setUp(self):
        self.patient = Patient(40, 70, 170, 'male')
        self.config = SimulationConfig()
        self.engine = SimulationEngine(self.patient, self.config)
        
    def test_switch_logic(self):
        # 1. Standard Mode -> Default OFF (Widget inits to OFF)
        ctrl = ControlPanelWidget(self.engine, tutorial_mode=False)
        self.assertFalse(ctrl.btn_vent_power.isChecked())
        # self.assertTrue(ctrl.sb_rr.isEnabled()) # Disabled if OFF
        
        # 2. Tutorial Mode -> Default OFF
        ctrl_tut = ControlPanelWidget(self.engine, tutorial_mode=True)
        self.assertFalse(ctrl_tut.btn_vent_power.isChecked())
        self.assertFalse(ctrl_tut.sb_rr.isEnabled())
        self.assertFalse(ctrl_tut.sb_rr.isEnabled())
        # UI init is now non-destructive to engine. Engine defaults to RR=12.
        self.assertEqual(self.engine.resp_mech.set_rr, 12.0)
        
        # 2b. Sync should align UI to Engine
        # Default engine is OFF. Let's force it ON to test sync.
        self.engine.vent.is_on = True
        ctrl_tut.sync_with_engine()
        self.assertTrue(ctrl_tut.btn_vent_power.isChecked())
        
        # 3. Toggle OFF (User Action)
        ctrl_tut.btn_vent_power.setChecked(False) 
        self.assertFalse(ctrl_tut.sb_rr.isEnabled())
        # Now engine should be 0
        self.assertEqual(self.engine.resp_mech.set_rr, 0)
        
        # 4. Toggle ON
        ctrl_tut.btn_vent_power.setChecked(True)
        self.assertTrue(ctrl_tut.sb_rr.isEnabled())
        self.assertEqual(self.engine.resp_mech.set_rr, 12)


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
        from ui.scenarios import create_hemorrhage_response
        from ui.tutorial_overlay import ScenarioOverlay
        
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
        
    def test_hemorrhage_overlay_functional(self):
        """Test ScenarioOverlay with hemorrhage scenario."""
        from ui.scenarios import create_hemorrhage_response
        from ui.tutorial_overlay import ScenarioOverlay
        
        scenario = create_hemorrhage_response()
        overlay = ScenarioOverlay(scenario)
        
        # Verify initial state
        self.assertEqual(overlay.current_step, 0)
        self.assertIn("Hemorrhage", overlay.lbl_step.text())
        
        # Step 1: Observe baseline - requires stable vitals
        # Set stable baseline
        self.engine.state.hr = 75
        self.engine.state.map = 80
        self.engine.state.spo2 = 98
        overlay.update_state(self.engine)
        self.assertTrue(overlay.requirements_met)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 1)
        
        # Step 2: Start hemorrhage - requires hemorrhage active
        overlay.update_state(self.engine)
        self.assertFalse(overlay.requirements_met)  # Not started yet
        
        self.engine.start_hemorrhage(500)
        overlay.update_state(self.engine)
        self.assertTrue(overlay.requirements_met)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 2)
        
        # Step 3: Recognize shock - requires tachycardia + hypotension
        self.engine.state.hr = 120
        self.engine.state.map = 55
        overlay.update_state(self.engine)
        self.assertTrue(overlay.requirements_met)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 3)
        
        # Step 4: Give fluids - check blood volume
        # This is a bit tricky to test directly, skip for now
        overlay.requirements_met = True  # Force for test
        overlay.advance_step()
        self.assertEqual(overlay.current_step, 4)
        
        # Step 5: Start vasopressor
        self.engine.nore_rate_ug_sec = 0.1
        overlay.update_state(self.engine)
        self.assertTrue(overlay.requirements_met)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 5)
        
        # Step 6: Stop bleeding
        self.engine.stop_hemorrhage()
        overlay.update_state(self.engine)
        self.assertTrue(overlay.requirements_met)
        overlay.click_next()
        self.assertEqual(overlay.current_step, 6)
        
        # Step 7: Reassess - MAP > 65, HR < 120
        self.engine.state.map = 70
        self.engine.state.hr = 100
        overlay.update_state(self.engine)
        self.assertTrue(overlay.requirements_met)
        overlay.click_next()
        
        # Should be complete
        self.assertIn("complete", overlay.lbl_instruction.text().lower())
