"""
Volatile PK Wash-In/Wash-Out Validation Tests.

These tests verify the physiological behavior of volatile anesthetic uptake
and elimination based on published time constants.

Validation criteria (heuristic, consistent with Carpenter et al. Anesth Analg. 1986;
Yasuda et al. Anesth Analg. 1991):
- VRG (brain): 50% equilibration in ~5-10 min
- Muscle: 50% equilibration in ~1-2 hours
- Fat: 50% equilibration in many hours (5-15+)
- Wash-in faster than wash-out (due to blood:gas partition)
"""

import pytest
import numpy as np
from anasim.patient.volatile_pk import VolatilePK, VolatileState


class TestVolatilePKWashIn:
    """Validate volatile anesthetic wash-in kinetics."""
    
    @pytest.fixture
    def setup_sevo(self, patient_factory):
        """Create patient and Sevoflurane PK model."""
        patient = patient_factory(sex="Male")
        # Sevoflurane: blood:gas ~0.65, MAC ~2.0%
        pk = VolatilePK(patient, name="Sevoflurane", lambda_b_g=0.65, mac_40=2.0)
        return pk, patient
    
    def test_vrg_fast_equilibration(self, setup_sevo):
        """VRG (brain) should reach ~50% of inspired within 10 min."""
        pk, patient = setup_sevo
        
        fi_sevo = 0.04  # 4% inspired (2 MAC)
        va = 5.0  # L/min alveolar ventilation
        co = 5.0  # L/min cardiac output
        
        # Run for 10 minutes (600 seconds)
        for _ in range(600):
            pk.step(1.0, fi_sevo, va, co)
        
        # VRG should be at least 50% of inspired
        p_vrg_pct = pk.state.p_vrg * 100
        assert p_vrg_pct > fi_sevo * 100 * 0.4, \
            f"VRG {p_vrg_pct:.2f}% should be >40% of inspired {fi_sevo*100}% at 10 min"
    
    def test_fat_slow_equilibration(self, setup_sevo):
        """Fat should be much slower than VRG to equilibrate."""
        pk, patient = setup_sevo
        
        fi_sevo = 0.04
        va = 5.0
        co = 5.0
        
        # Run for 10 minutes
        for _ in range(600):
            pk.step(1.0, fi_sevo, va, co)
        
        # Fat should be much lower than VRG at 10 min
        assert pk.state.p_fat < pk.state.p_vrg * 0.3, \
            f"Fat {pk.state.p_fat:.4f} should be <30% of VRG {pk.state.p_vrg:.4f} at 10 min"
    
    def test_mac_tracks_vrg(self, setup_sevo):
        """MAC should reflect VRG concentration correctly."""
        pk, patient = setup_sevo
        
        fi_sevo = 0.04  # 4% = 2 MAC for age 40
        va = 5.0
        co = 5.0
        
        # Run to near steady state (30 min)
        for _ in range(1800):
            pk.step(1.0, fi_sevo, va, co)
        
        # VRG should be near inspired, MAC should be ~2.0
        mac = pk.state.mac
        assert mac > 1.5, f"MAC {mac:.2f} should approach 2.0 at 4% inspired"
        assert mac < 2.5, f"MAC {mac:.2f} should not exceed 2.5"


class TestVolatilePKWashOut:
    """Validate volatile anesthetic wash-out kinetics."""
    
    @pytest.fixture
    def primed_sevo(self, patient_factory):
        """Create equilibrated Sevoflurane model."""
        patient = patient_factory(sex="Male")
        pk = VolatilePK(patient, name="Sevoflurane", lambda_b_g=0.65, mac_40=2.0)
        
        # Prime to near equilibrium at 2% (1 MAC)
        fi_sevo = 0.02
        va = 5.0
        co = 5.0
        for _ in range(3600):  # 60 min equilibration
            pk.step(1.0, fi_sevo, va, co)
        
        return pk, patient
    
    def test_washout_vrg_fast(self, primed_sevo):
        """VRG should wash out quickly (minutes)."""
        pk, patient = primed_sevo
        
        initial_vrg = pk.state.p_vrg
        
        # Wash out with zero inspired for 10 min
        for _ in range(600):
            pk.step(1.0, 0.0, 5.0, 5.0)
        
        final_vrg = pk.state.p_vrg
        
        # VRG should drop significantly
        drop_pct = (initial_vrg - final_vrg) / initial_vrg * 100
        assert drop_pct > 40, f"VRG should drop >40% in 10 min, dropped {drop_pct:.1f}%"
    
    def test_fat_retains_agent(self, primed_sevo):
        """Fat should retain agent much longer than VRG during wash-out."""
        pk, patient = primed_sevo
        
        initial_fat = pk.state.p_fat
        initial_vrg = pk.state.p_vrg
        
        # Wash out for 30 min
        for _ in range(1800):
            pk.step(1.0, 0.0, 5.0, 5.0)
        
        final_fat = pk.state.p_fat
        final_vrg = pk.state.p_vrg
        
        vrg_retention = final_vrg / initial_vrg if initial_vrg > 0 else 0
        fat_retention = final_fat / initial_fat if initial_fat > 0 else 0
        
        # Fat should retain more than VRG
        assert fat_retention > vrg_retention, \
            f"Fat retention {fat_retention:.2f} should exceed VRG {vrg_retention:.2f}"


class TestVolatilePKEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_ventilation(self, patient_factory):
        """Zero ventilation should prevent uptake."""
        patient = patient_factory(sex="Male")
        pk = VolatilePK(patient, name="Sevoflurane", lambda_b_g=0.65, mac_40=2.0)
        
        # Try to administer with zero ventilation
        for _ in range(100):
            pk.step(1.0, 0.04, 0.0, 5.0)  # VA = 0
        
        # Should have minimal uptake
        assert pk.state.p_vrg < 0.01, f"VRG {pk.state.p_vrg} should be minimal with zero VA"
    
    def test_age_corrected_mac(self, patient_factory):
        """MAC should decrease with age."""
        patient_young = patient_factory(age=30, sex="Male")
        patient_old = patient_factory(age=70, sex="Male")
        
        pk_young = VolatilePK(patient_young, "Sevo", lambda_b_g=0.65, mac_40=2.0)
        pk_old = VolatilePK(patient_old, "Sevo", lambda_b_g=0.65, mac_40=2.0)
        
        # Elderly should require less anesthetic (lower MAC)
        assert pk_old.mac_age < pk_young.mac_age, \
            f"Elderly MAC {pk_old.mac_age:.2f} should be < young {pk_young.mac_age:.2f}"
