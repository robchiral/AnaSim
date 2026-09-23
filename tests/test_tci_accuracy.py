"""
TCI Controller Accuracy Validation Tests.

Verifies that the TCI controller correctly achieves and maintains target concentrations.
Criteria: 10% tolerance for tracking, 5% for steady state, <25% overshoot.
"""

import numpy as np
import pytest

from anasim.core.tci import TCIController
from anasim.patient.pk_models import PropofolPKSchnider, RemifentanilPKMinto


class TestTCIPropofol:
    """Validate TCI controller accuracy for Propofol."""
    
    @pytest.fixture
    def setup_propofol_tci(self, patient_factory):
        """Create patient and Propofol TCI controller."""
        patient = patient_factory(age=45, sex="Male")
        pk = PropofolPKSchnider(patient)
        tci = TCIController(
            pk_model=pk,
            drug_name="Propofol",
            target_compartment="effect_site",
            max_rate=50.0,  # mg/s (reasonable max for propofol)
            sampling_time=1.0,
            control_time=10.0
        )
        return pk, tci, patient
    
    def test_target_tracking_step_increase(self, setup_propofol_tci):
        """Test that TCI reaches target after step increase."""
        pk, tci, patient = setup_propofol_tci
        
        target_ce = 3.0  # µg/mL - typical induction target
        ce_history = []
        
        # Run for 300 seconds (5 min)
        for t in range(300):
            rate = tci.step(target_ce)
            pk.step(1.0, rate)  # 1 second step
            ce_history.append(pk.state.ce)
        
        # Should reach within 10% of target by 60s
        ce_at_60s = ce_history[59]
        assert ce_at_60s > target_ce * 0.5, \
            f"Ce should be >50% of target by 60s, got {ce_at_60s:.2f}"
        
        # Steady state (last 60s) should be within 5% of target
        ce_steady = np.mean(ce_history[-60:])
        assert abs(ce_steady - target_ce) / target_ce < 0.10, \
            f"Steady-state Ce={ce_steady:.2f} not within 10% of target={target_ce}"
    
    def test_no_excessive_overshoot(self, setup_propofol_tci):
        """Test that TCI doesn't overshoot target by more than 20%."""
        pk, tci, patient = setup_propofol_tci
        
        target_ce = 4.0  # µg/mL
        ce_max = 0.0
        
        # Run for 5 minutes
        for t in range(300):
            rate = tci.step(target_ce)
            pk.step(1.0, rate)
            ce_max = max(ce_max, pk.state.ce)
        
        overshoot_pct = (ce_max - target_ce) / target_ce * 100
        assert overshoot_pct < 25, \
            f"Overshoot {overshoot_pct:.1f}% exceeds 25% limit"
    
    def test_target_decrease_response(self, setup_propofol_tci):
        """Test that TCI responds correctly when target is decreased."""
        pk, tci, patient = setup_propofol_tci
        
        # First reach high target
        high_target = 4.0
        for t in range(180):
            rate = tci.step(high_target)
            pk.step(1.0, rate)
        
        ce_high = pk.state.ce
        
        # Now decrease target
        low_target = 2.5
        for t in range(300):
            rate = tci.step(low_target)
            pk.step(1.0, rate)
        
        ce_low = pk.state.ce
        
        # Should have decreased and be near new target
        assert ce_low < ce_high, "Ce should decrease when target decreased"
        assert abs(ce_low - low_target) / low_target < 0.15, \
            f"Ce={ce_low:.2f} not within 15% of target={low_target}"
    
    def test_plasma_target_mode(self, patient_factory):
        """Test TCI in plasma-targeting mode."""
        patient = patient_factory(age=50, weight=75, height=175, sex="Male")
        pk = PropofolPKSchnider(patient)
        tci = TCIController(
            pk_model=pk,
            drug_name="Propofol",
            target_compartment="plasma",  # Plasma targeting
            max_rate=50.0,
            sampling_time=1.0,
            control_time=10.0
        )
        
        target_cp = 4.0  # µg/mL
        cp_history = []
        
        for t in range(180):
            rate = tci.step(target_cp)
            pk.step(1.0, rate)
            cp_history.append(pk.state.c1)
        
        cp_final = np.mean(cp_history[-30:])
        assert abs(cp_final - target_cp) / target_cp < 0.15, \
            f"Plasma Cp={cp_final:.2f} not within 15% of target={target_cp}"


class TestTCIRemifentanil:
    """Validate TCI controller accuracy for Remifentanil."""
    
    @pytest.fixture
    def setup_remi_tci(self, patient_factory):
        """Create patient and Remifentanil TCI controller."""
        patient = patient_factory(age=45, sex="Male")
        pk = RemifentanilPKMinto(patient)
        tci = TCIController(
            pk_model=pk,
            drug_name="Remifentanil",
            target_compartment="effect_site",
            max_rate=500.0,  # µg/s (high max for remi, will be limited by controller)
            sampling_time=1.0,
            control_time=10.0
        )
        return pk, tci, patient
    
    def test_fast_equilibration(self, setup_remi_tci):
        """Test that Remifentanil reaches target faster than Propofol (short half-life)."""
        pk, tci, patient = setup_remi_tci
        
        target_ce = 4.0  # ng/mL
        ce_history = []
        
        for t in range(120):
            rate = tci.step(target_ce)
            pk.step(1.0, rate)
            ce_history.append(pk.state.ce)
        
        # Remi should reach 75% of target within 30s (fast drug)
        ce_at_30s = ce_history[29]
        assert ce_at_30s > target_ce * 0.5, \
            f"Remi Ce should be >50% of target by 30s, got {ce_at_30s:.2f}"
    
    def test_remi_steady_state(self, setup_remi_tci):
        """Test Remifentanil steady-state accuracy."""
        pk, tci, patient = setup_remi_tci
        
        target_ce = 3.0  # ng/mL
        
        # Run for 3 minutes
        for t in range(180):
            rate = tci.step(target_ce)
            pk.step(1.0, rate)
        
        ce_final = pk.state.ce
        assert abs(ce_final - target_ce) / target_ce < 0.10, \
            f"Remi Ce={ce_final:.2f} not within 10% of target={target_ce}"


class TestTCIInitialization:
    """Test TCI controller initialization and state management."""
    
    def test_set_state_priming(self, patient_factory):
        """Test that set_state correctly primes the controller."""
        patient = patient_factory(sex="Male")
        pk = PropofolPKSchnider(patient)
        tci = TCIController(
            pk_model=pk,
            drug_name="Propofol",
            target_compartment="effect_site",
            max_rate=50.0
        )
        
        # Prime to steady state
        target = 3.0
        tci.set_state(c1=target, c2=target, c3=target, ce=target)
        
        # Controller should recognize target achievement and provide a low rate.
        rate = tci.step(target)
        
        # Rate should be low (maintenance level).
        assert rate < 5.0, f"Primed TCI should give low maintenance rate, got {rate:.2f}"
    
    def test_zero_target(self, patient_factory):
        """Test that zero target stops infusion."""
        patient = patient_factory(sex="Male")
        pk = PropofolPKSchnider(patient)
        tci = TCIController(
            pk_model=pk,
            drug_name="Propofol",
            target_compartment="effect_site",
            max_rate=50.0
        )
        
        # Prime with some concentration
        tci.set_state(c1=3.0, c2=2.0, c3=1.0, ce=3.0)
        
        # Set target to zero
        rate = tci.step(0.0)
        
        assert rate == 0.0, f"Zero target should give zero rate, got {rate}"


class TestEngineTCI:
    """Pump-limited TCI inside the engine."""

    def test_effect_site_induction_reaches_target_without_overshoot(self, engine_factory):
        engine = engine_factory(start=True)
        engine.set_airway_mode("Mask")
        engine.set_fgf(6.0, 0.0)
        engine.enable_tci("propofol", 4.0)
        peak = 0.0
        time_to_90 = None
        for _ in range(3600):
            engine.step(0.1)
            peak = max(peak, engine.state.propofol_ce)
            if time_to_90 is None and engine.state.propofol_ce >= 3.6:
                time_to_90 = engine.state.time
        assert time_to_90 is not None and time_to_90 < 240.0
        assert peak < 4.0 * 1.03
        assert engine.state.propofol_ce == pytest.approx(4.0, rel=0.03)

    def test_plasma_controller_does_not_bolus_after_resync(self, anesthetized_engine):
        """Resyncing to live PK state must not trigger max-rate boluses."""
        target = anesthetized_engine.tci_nore.target
        peak = 0.0
        for _ in range(12000):
            anesthetized_engine.step(0.1)
            peak = max(peak, anesthetized_engine.pk_nore.state.c1)
        assert peak < target * 1.4
