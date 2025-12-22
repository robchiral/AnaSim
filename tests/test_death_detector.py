
import unittest
from core.engine import SimulationEngine, SimulationConfig
from patient.patient import Patient

class TestDeathDetector(unittest.TestCase):
    def setUp(self):
        self.patient = Patient(age=40, weight=70, height=170, sex='male')
        
    def test_default_disabled(self):
        """Test that death detector is disabled by default."""
        config = SimulationConfig()
        self.assertFalse(config.enable_death_detector)
        engine = SimulationEngine(self.patient, config)
        
        # Force lethal state
        engine.state.hr = 0
        engine.state.map = 0
        
        # Step for 20s (longer than 15s grace)
        for _ in range(200): # 20s
            engine.step(0.1)
            
        self.assertFalse(engine.state.is_dead)
        
    def test_enabled_trigger_bradycardia(self):
        """Test death trigger for bradycardia."""
        config = SimulationConfig(enable_death_detector=True)
        engine = SimulationEngine(self.patient, config)
        
        # Force lethal state
        engine.state.hr = 0
        engine.smooth_hr = 0 # To prevent smoothing from restoring it immediately
        engine.state.map = 80 # Normal MAP
        
        # Step for 10s (should be alive)
        for _ in range(100):
            engine.state.hr = 0 # Sustain the insult (engine reset might fight it)
            engine.smooth_hr = 0
            engine._check_patient_viability(0.1)
            
        self.assertFalse(engine.state.is_dead)
        
        # Step for another 10s (total 20s > 15s)
        for _ in range(100):
            engine.state.hr = 0
            engine.smooth_hr = 0
            engine._check_patient_viability(0.1)
            
        self.assertTrue(engine.state.is_dead)
        self.assertIn("Bradycardia", engine.state.death_reason)
        
    def test_enabled_trigger_hypotension(self):
        """Test death trigger for hypotension."""
        config = SimulationConfig(enable_death_detector=True)
        engine = SimulationEngine(self.patient, config)
        
        # Force lethal state
        engine.state.map = 5.0
        engine.smooth_map = 5.0
        
        # Step for 20s
        for _ in range(200):
            engine.state.map = 5.0
            engine.smooth_map = 5.0
            engine._check_patient_viability(0.1)
            
        self.assertTrue(engine.state.is_dead)
        self.assertIn("Hypotension", engine.state.death_reason)

if __name__ == '__main__':
    unittest.main()
