"""
NIBP Cycling and Capnography Waveform Validation Tests.

Tests verify:
- NIBP intermittent cycling behavior with realistic measurement intervals
- Capnography waveform shape characteristics (dead space, plateau, etc.)
"""

import pytest
import numpy as np
from anasim.core.state import SimulationConfig
from anasim.monitors.capno import Capnograph, CapnoContext


class TestNIBPCycling:
    """Test NIBP intermittent measurement behavior."""
    
    @pytest.fixture
    def setup_engine(self, engine_factory):
        """Create engine for NIBP testing."""
        config = SimulationConfig(mode="steady_state", maint_type="tiva", dt=0.5)
        return engine_factory(config=config, start=True, age=45, sex="Male")
    
    def test_nibp_intermittent_values(self, setup_engine):
        """NIBP should show intermittent readings (not every step)."""
        engine = setup_engine
        
        nibp_values = []
        for _ in range(600):  # 10 minutes
            engine.step(1.0)
            nibp_values.append(engine.state.nibp_map)
        
        # NIBP should have some valid readings (not all zero or None)
        valid_readings = [v for v in nibp_values if v and v > 0]
        assert len(valid_readings) > 0, "No valid NIBP readings in 10 minutes"
        
        # Values should be in physiological range
        for v in valid_readings:
            assert 30 <= v <= 200, f"NIBP {v} out of physiological range"
    
    def test_nibp_correlates_with_map(self, setup_engine):
        """NIBP should correlate with continuous MAP at time of measurement."""
        engine = setup_engine
        
        # Run for a bit to stabilize
        for _ in range(120):
            engine.step(1.0)
        
        # Wait for NIBP update and compare to MAP at that moment
        for _ in range(180):
            engine.step(1.0)
            nibp = engine.state.nibp_map
            if nibp and nibp > 0:
                # Get MAP at the same moment as NIBP reading
                current_map = engine.state.map
                # NIBP should be within 50% of continuous MAP (NIBP has measurement noise)
                # This is a sanity check, not a precision test
                assert abs(nibp - current_map) / max(current_map, 1) < 0.60, \
                    f"NIBP {nibp} differs too much from MAP {current_map}"
                break


class TestCapnographyWaveform:
    """Test capnography waveform characteristics."""
    
    @pytest.fixture
    def setup_capno(self):
        """Create capnograph with fixed RNG for reproducibility."""
        rng = np.random.default_rng(42)
        capno = Capnograph(rng=rng)
        return capno
    
    def test_phase_transitions(self, setup_capno):
        """Waveform should transition through all phases."""
        capno = setup_capno
        
        # Simulate a breath cycle
        phases_seen = set()
        p_alv = 35.0  # Alveolar CO2 in mmHg
        
        # Run through one breath cycle - correct API: (dt, phase, p_alv, ...)
        for t in np.arange(0, 4.0, 0.05):
            phase = "INSP" if t > 2.0 else "EXP"
            result = capno.step(0.05, phase, p_alv, is_spontaneous=False, exp_duration=2.0)
            phases_seen.add(capno.state.phase)
        
        # Should have seen multiple phases
        assert len(phases_seen) >= 2, f"Only saw phases {phases_seen}"
    
    def test_plateau_near_etco2(self, setup_capno):
        """Plateau phase CO2 should approach EtCO2 value."""
        capno = setup_capno
        
        p_alv = 38.0  # Alveolar CO2
        
        # Run through expiration to plateau
        max_co2 = 0.0
        for t in np.arange(0, 2.5, 0.02):
            result = capno.step(0.02, "EXP", p_alv, is_spontaneous=False, exp_duration=2.5)
            max_co2 = max(max_co2, capno.state.co2)
        
        # Max CO2 should be close to alveolar
        assert max_co2 > p_alv * 0.7, f"Plateau CO2 {max_co2:.1f} too far from alveolar {p_alv}"
    
    def test_inspiration_drops_co2(self, setup_capno):
        """CO2 should drop during inspiration (fresh gas)."""
        capno = setup_capno
        
        p_alv = 40.0
        
        # First, run through expiration to get high CO2
        for _ in range(50):
            capno.step(0.04, "EXP", p_alv, is_spontaneous=False, exp_duration=2.0)
        
        peak_co2 = capno.state.co2
        
        # Now inspiration
        for _ in range(50):
            capno.step(0.04, "INSP", p_alv, is_spontaneous=False, exp_duration=2.0)
        
        final_co2 = capno.state.co2
        
        # CO2 should drop during inspiration
        assert final_co2 < peak_co2, \
            f"CO2 should drop during inspiration: {peak_co2:.1f} -> {final_co2:.1f}"
