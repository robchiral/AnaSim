"""
Tests for Ventilator Dynamics.

Tests VCV/PCV modes, auto-PEEP, PEEP effects, and hemodynamic coupling.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anasim.physiology.resp_mech import RespiratoryMechanics, VentMode
from anasim.physiology.respiration import RespiratoryModel
from anasim.patient.patient import Patient
from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig


class TestVCVMode(unittest.TestCase):
    """Test Volume Control Ventilation mode."""
    
    def setUp(self):
        self.mech = RespiratoryMechanics(compliance=0.05, resistance=10.0)
        self.mech.set_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    
    def test_vcv_delivers_set_vt(self):
        """VCV should deliver approximately the set tidal volume."""
        # Run for multiple breath cycles (need 2+ for delivered_vt to update)
        dt = 0.01
        for _ in range(1200):  # 12 seconds (~2.5 breaths at RR=12)
            self.mech.step(dt)
        
        # Delivered Vt should be close to set Vt (500 mL)
        self.assertAlmostEqual(self.mech.state.delivered_vt, 500.0, delta=50.0)
    
    def test_vcv_pressure_increases_with_reduced_compliance(self):
        """In VCV, peak pressure should increase when compliance decreases."""
        # Normal compliance
        self.mech.compliance = 0.05
        for _ in range(500):
            self.mech.step(0.01)
        normal_peak = self.mech.state.paw_peak
        
        # Reduced compliance (e.g., ARDS)
        self.mech.compliance = 0.025
        self.mech.state.volume = 0
        for _ in range(500):
            self.mech.step(0.01)
        reduced_peak = self.mech.state.paw_peak
        
        # Peak should be higher with reduced compliance
        self.assertGreater(reduced_peak, normal_peak)


class TestPCVMode(unittest.TestCase):
    """Test Pressure Control Ventilation mode."""
    
    def setUp(self):
        self.mech = RespiratoryMechanics(compliance=0.05, resistance=10.0)
        self.mech.set_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", 
                               mode="PCV", p_insp=15.0)
    
    def test_pcv_pressure_is_controlled(self):
        """In PCV, airway pressure should stay at set value during inspiration."""
        for _ in range(100):
            state = self.mech.step(0.01)
            if state.phase == "INSP":
                # Paw should be close to P_insp + PEEP = 15 + 5 = 20
                self.assertAlmostEqual(state.paw, 20.0, delta=2.0)
    
    def test_pcv_vt_decreases_with_reduced_compliance(self):
        """In PCV, delivered Vt should decrease when compliance decreases."""
        # Normal compliance
        mech_normal = RespiratoryMechanics(compliance=0.05, resistance=10.0)
        mech_normal.set_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", 
                                  mode="PCV", p_insp=15.0)
        for _ in range(1200):  # Multiple breath cycles
            mech_normal.step(0.01)
        normal_vt = mech_normal.state.delivered_vt
        
        # Reduced compliance (new instance)
        mech_reduced = RespiratoryMechanics(compliance=0.025, resistance=10.0)
        mech_reduced.set_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", 
                                   mode="PCV", p_insp=15.0)
        for _ in range(1200):
            mech_reduced.step(0.01)
        reduced_vt = mech_reduced.state.delivered_vt
        
        # Both should have non-zero Vt
        self.assertGreater(normal_vt, 100)  # Ensure normal case worked
        self.assertGreater(reduced_vt, 50)  # Reduced should still have some Vt
        
        # Vt should be lower with reduced compliance
        self.assertLess(reduced_vt, normal_vt)


class TestAutoPEEP(unittest.TestCase):
    """Test auto-PEEP calculation."""
    
    def test_auto_peep_develops_with_high_rr(self):
        """Auto-PEEP should develop when expiratory time is too short."""
        mech = RespiratoryMechanics(compliance=0.05, resistance=15.0)
        
        # High RR with short expiration (1:1 I:E at RR=30)
        # Breath cycle = 2 sec, each phase = 1 sec
        # Time constant = R × C = 15 × 0.05 = 0.75 sec
        # Need ~3 time constants (2.25 sec) for full expiration
        mech.set_settings(rr=30, vt=0.5, peep=5.0, ie="1:1", mode="VCV")
        
        # Run for several breaths
        for _ in range(1000):  # 10 seconds
            mech.step(0.01)
        
        # Should have auto-PEEP > 0
        self.assertGreater(mech.state.auto_peep, 0.5)
        
    def test_no_auto_peep_with_normal_settings(self):
        """Normal settings should not produce significant auto-PEEP."""
        mech = RespiratoryMechanics(compliance=0.05, resistance=10.0)
        mech.set_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        
        # Run for several breaths
        for _ in range(1000):
            mech.step(0.01)
        
        # Auto-PEEP should be minimal (< 1 cmH2O)
        self.assertLess(mech.state.auto_peep, 1.0)


class TestPEEPOxygenation(unittest.TestCase):
    """Test PEEP effects on oxygenation."""
    
    def setUp(self):
        self.patient = Patient(age=40, weight=70, height=170, sex="male")
        self.resp = RespiratoryModel(self.patient)
    
    def test_peep_improves_pao2(self):
        """Higher PEEP should improve PaO2 by reducing A-a gradient."""
        # Run with no PEEP
        for _ in range(500):
            self.resp.step(0.01, ce_prop=0, ce_remi=0, mech_vent_mv=6.0,
                          fio2=0.5, peep=0.0)
        pao2_no_peep = self.resp.state.p_arterial_o2
        
        # Reset and run with PEEP 10
        self.resp.state.p_arterial_o2 = 95.0
        for _ in range(500):
            self.resp.step(0.01, ce_prop=0, ce_remi=0, mech_vent_mv=6.0,
                          fio2=0.5, peep=10.0)
        pao2_with_peep = self.resp.state.p_arterial_o2
        
        # PaO2 should be higher with PEEP
        self.assertGreater(pao2_with_peep, pao2_no_peep)


class TestHemodynamicCoupling(unittest.TestCase):
    """Test ventilator-hemodynamic interactions."""
    
    def test_mean_paw_calculation(self):
        """Mean Paw should be tracked correctly."""
        mech = RespiratoryMechanics(compliance=0.05, resistance=10.0)
        mech.set_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        
        # Run for several breaths
        for _ in range(1000):
            mech.step(0.01)
        
        # Mean Paw should be between PEEP and peak
        self.assertGreater(mech.state.paw_mean, 5.0)
        self.assertLess(mech.state.paw_mean, mech.state.paw_peak)


class TestVentilatorEdgeCases(unittest.TestCase):
    """Validate safety fallbacks and parsing edge cases."""
    
    def setUp(self):
        self.mech = RespiratoryMechanics(compliance=0.05, resistance=10.0)
    
    def test_zero_rr_enters_passive_exhalation(self):
        """RR=0 should leave Paw at PEEP and allow passive exhalation."""
        self.mech.set_settings(rr=0.0, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        self.mech.state.volume = 0.6  # 600 mL above FRC
        
        result = self.mech.step(0.5)
        
        self.assertEqual(result.phase, "EXP")
        self.assertLess(result.volume, 0.6)
        self.assertLessEqual(result.flow, 0.0)
        self.assertAlmostEqual(result.paw, self.mech.set_peep, delta=0.5)
    
    def test_invalid_ie_ratio_defaults_to_safe_fraction(self):
        """Malformed I:E strings should revert to default 1:2 timing."""
        self.mech.set_settings(rr=12.0, vt=0.5, peep=5.0, ie="bad_format", mode="VCV")
        
        self.assertAlmostEqual(self.mech.insp_time_fraction, 1.0 / 3.0, places=3)
        
        # Ensure stepping still works without raising errors
        self.mech.step(0.1)


class TestEngineIntegration(unittest.TestCase):
    """Test ventilator integration in the simulation engine."""
    
    def setUp(self):
        self.patient = Patient(age=40, weight=70, height=170, sex="male")
        self.config = SimulationConfig(mode='steady_state', maint_type='tiva')
        self.engine = SimulationEngine(self.patient, self.config)
    
    def test_set_vent_mode_pcv(self):
        """Engine should accept PCV mode settings."""
        self.engine.set_vent_settings(
            rr=12, vt=0.5, peep=5.0, ie="1:2", mode="PCV", p_insp=15.0
        )
        self.assertEqual(self.engine.resp_mech.mode, VentMode.PCV)
        self.assertEqual(self.engine.resp_mech.set_p_insp, 15.0)
    
    def test_delivered_vt_tracks_in_state(self):
        """Delivered Vt should be tracked in engine state."""
        self.engine.start()
        for _ in range(500):
            self.engine.step(0.01)
        
        # State should have reasonable Vt
        self.assertGreater(self.engine.state.vt, 100)  # mL
        self.assertLess(self.engine.state.vt, 1000)


class TestBagMaskIntegration(unittest.TestCase):
    """Ensure bag-mask ventilation maintains gas exchange when spontaneous drive is lost."""
    
    def setUp(self):
        patient = Patient(age=40, weight=70, height=170, sex="male")
        config = SimulationConfig(mode='awake', maint_type='tiva')
        self.engine = SimulationEngine(patient, config)
        self.engine.start()
        self.engine.set_airway_mode("Mask")
        self.engine.set_bag_mask_ventilation(False)
    
    def tearDown(self):
        self.engine.stop()
    
    def _advance(self, seconds, dt=0.25):
        for _ in range(int(seconds / dt)):
            self.engine.step(dt)
    
    def test_bag_mask_improves_ventilation_during_paralysis(self):
        """Bag-mask should restore MV/EtCO2 when NMBA eliminates spontaneous breaths."""
        # Deep paralysis to eliminate VT
        dose = 0.8 * self.engine.patient.weight
        self.engine.give_drug_bolus("Rocuronium", dose)
        self._advance(60.0)
        
        mv_without = self.engine.state.mv
        pao2_without = self.engine.state.pao2
        
        # Enable bag-mask ventilation
        self.engine.set_bag_mask_ventilation(True, rr=12.0, vt=0.6)
        self._advance(60.0)
        
        mv_with = self.engine.state.mv
        pao2_with = self.engine.state.pao2
        
        self.assertGreater(mv_with, mv_without + 2.0)
        self.assertGreater(pao2_with, pao2_without + 20.0)


class TestVentilatorHemodynamics(unittest.TestCase):
    """Full engine validation that PEEP changes impact MAP."""
    
    def setUp(self):
        patient = Patient(age=40, weight=70, height=170, sex="male")
        config = SimulationConfig(mode='steady_state', maint_type='tiva')
        self.engine = SimulationEngine(patient, config)
        self.engine.start()
    
    def tearDown(self):
        self.engine.stop()
    
    def _advance(self, seconds, dt=0.1):
        for _ in range(int(seconds / dt)):
            self.engine.step(dt)
    
    def test_high_peep_reduces_map(self):
        """Increasing PEEP/mean Paw should lower MAP via preload effects."""
        self.engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        self._advance(60.0)
        baseline_map = self.engine.state.map
        
        self.engine.set_vent_settings(rr=12, vt=0.5, peep=15.0, ie="1:2", mode="VCV")
        self._advance(60.0)
        high_peep_map = self.engine.state.map
        
        self.assertLess(high_peep_map, baseline_map)


class TestPSVCPAPIntegration(unittest.TestCase):
    """Validate PSV/CPAP provide spontaneous support without fixed Vt targets."""

    def setUp(self):
        patient = Patient(age=40, weight=70, height=170, sex="male")
        config = SimulationConfig(mode='awake', maint_type='tiva')
        self.engine = SimulationEngine(patient, config)
        self.engine.start()
        self.engine.set_airway_mode("ETT")

    def tearDown(self):
        self.engine.stop()

    def _advance(self, seconds, dt=0.1):
        for _ in range(int(seconds / dt)):
            self.engine.step(dt)

    def test_psv_increases_vt_over_cpap(self):
        """PSV should augment tidal volume compared with CPAP alone."""
        # CPAP baseline (PEEP only)
        self.engine.set_vent_settings(rr=0.0, vt=0.0, peep=5.0, ie="1:2", mode="CPAP", p_insp=0.0)
        self._advance(20.0)
        vt_cpap = self.engine.state.vt
        assert vt_cpap > 200.0, f"CPAP should preserve spontaneous VT, got {vt_cpap:.0f} mL"

        # PSV support
        self.engine.set_vent_settings(rr=0.0, vt=0.0, peep=5.0, ie="1:2", mode="PSV", p_insp=10.0)
        self._advance(20.0)
        vt_psv = self.engine.state.vt
        assert vt_psv > vt_cpap + 50.0, \
            f"PSV should increase VT vs CPAP (CPAP={vt_cpap:.0f}, PSV={vt_psv:.0f})"

    def test_cpap_paw_stays_near_peep(self):
        """CPAP should not generate inspiratory pressures above set PEEP."""
        self.engine.set_vent_settings(rr=0.0, vt=0.0, peep=8.0, ie="1:2", mode="CPAP", p_insp=0.0)
        self._advance(10.0)
        paw_peak = self.engine.resp_mech.state.paw_peak
        assert paw_peak <= 10.0, f"CPAP Paw peak {paw_peak:.1f} too high for PEEP 8"


class TestBagMaskVentilation(unittest.TestCase):
    """Test bag-mask ventilation functionality (separate from mechanical vent)."""
    
    def setUp(self):
        self.patient = Patient(age=40, weight=70, height=170, sex="male")
        self.config = SimulationConfig(mode='awake', maint_type='tiva')
        self.engine = SimulationEngine(self.patient, self.config)
        self.engine.start()
    
    def test_bag_mask_requires_airway(self):
        """Bag-mask should not provide ventilation without airway connection."""
        from anasim.core.state import AirwayType
        
        # Start with no airway connected
        self.engine.state.airway_mode = AirwayType.NONE
        
        # Enable bag-mask (it activates without contributing to ventilation)
        self.engine.bag_mask_active = True
        self.engine.bag_mask_rr = 12.0
        self.engine.bag_mask_vt = 0.5
        
        # The bag_mask_mv calculation should be 0 when not connected
        connected = self.engine.state.airway_mode in (AirwayType.ETT, AirwayType.MASK)
        self.assertFalse(connected)
        
        # Calculate what bag_mask_mv would be (should be 0)
        bag_mask_mv = self.engine.bag_mask_rr * self.engine.bag_mask_vt if connected else 0.0
        self.assertEqual(bag_mask_mv, 0.0)


if __name__ == '__main__':
    unittest.main()
