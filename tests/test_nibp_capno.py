"""
NIBP Cycling and Capnography Waveform Validation Tests.

Tests verify:
- NIBP intermittent cycling behavior with realistic measurement intervals
- Capnography waveform shape characteristics (dead space, plateau, etc.)
"""

import pytest
import numpy as np
from types import SimpleNamespace
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
        
        p_alv = 38.0  # End-tidal CO2 target
        
        # Run through expiration to plateau
        max_co2 = 0.0
        for t in np.arange(0, 2.5, 0.02):
            result = capno.step(0.02, "EXP", p_alv, is_spontaneous=False, exp_duration=2.5)
            max_co2 = max(max_co2, capno.state.co2)
        
        # Max CO2 should be close to alveolar
        assert max_co2 > p_alv * 0.7, f"Plateau CO2 {max_co2:.1f} too far from target {p_alv}"
    
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

    def test_context_prefers_spontaneous_when_rate_dominant(self):
        """Capno timing should follow patient-triggered breaths when spont RR dominates."""
        resp_state = SimpleNamespace(rr=12.0, drive_central=0.6, muscle_factor=1.0)
        ctx = Capnograph.build_context(resp_state, vent_rr=2.0, insp_fraction=0.33, vent_active=True)
        assert ctx.spontaneous_weight > 0.6, "Expected spontaneous timing when patient RR dominates"
        expected_exp = (60.0 / ctx.effective_rr) * 0.65
        assert abs(ctx.exp_duration - expected_exp) < 0.3

    def test_curare_cleft_dip_present(self):
        """Curare cleft should create a notch in the plateau during partial NMBA."""
        resp_state = SimpleNamespace(rr=2.0, drive_central=0.6, muscle_factor=0.5)
        ctx = Capnograph.build_context(resp_state, vent_rr=12.0, insp_fraction=0.33, vent_active=True)
        assert ctx.curare_active, "Expected curare cleft to be active in partial NMBA"

        capno_with = Capnograph(rng=np.random.default_rng(0))
        capno_without = Capnograph(rng=np.random.default_rng(0))
        p_alv = 40.0

        dips = []
        for _ in np.arange(0, ctx.exp_duration, 0.02):
            co2_with = capno_with.step(
                0.02, "EXP", p_alv,
                is_spontaneous=ctx.is_spontaneous,
                curare_cleft=ctx.curare_active,
                exp_duration=ctx.exp_duration,
                effort_scale=ctx.effort_scale,
                airway_obstruction=0.0,
            )
            co2_without = capno_without.step(
                0.02, "EXP", p_alv,
                is_spontaneous=ctx.is_spontaneous,
                curare_cleft=False,
                exp_duration=ctx.exp_duration,
                effort_scale=0.0,
                airway_obstruction=0.0,
            )
            dips.append(co2_without - co2_with)

        assert max(dips) > 1.0, "Curare cleft notch not evident in plateau"


class TestCapnographyVentSwitch:
    """Ensure capnography timing follows the active ventilation source."""

    def _count_insp_transitions(self, engine, seconds=20.0, dt=0.1):
        prev_phase = engine.capno.last_phase
        transitions = 0
        steps = int(seconds / dt)
        for _ in range(steps):
            engine.step(dt)
            phase = engine.capno.last_phase
            if phase == "INSP" and prev_phase != "INSP":
                transitions += 1
            prev_phase = phase
        return transitions / (seconds / 60.0)

    def test_capno_follows_vent_on_off_with_spontaneous(self, engine_factory):
        config = SimulationConfig(mode="awake")
        engine = engine_factory(config=config, start=True)
        engine.set_airway_mode("ETT")

        # Let spontaneous breathing stabilize.
        for _ in range(50):
            engine.step(0.1)

        # Vent on at low rate while spontaneous drive remains higher.
        engine.set_vent_settings(rr=6.0, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        for _ in range(20):
            engine.step(0.1)
        assert engine.state.rr > 8.0, "Spontaneous breathing should be present during vent support"

        vent_rr = self._count_insp_transitions(engine, seconds=20.0, dt=0.1)
        spont_rr_on = engine.state.rr
        assert abs(vent_rr - spont_rr_on) < 2.0, \
            f"Capno RR {vent_rr:.1f} should follow spontaneous rate {spont_rr_on:.1f}"
        assert abs(vent_rr - 6.0) > 1.5, "Capno should not lock to low set rate when spont dominates"

        # Vent off -> capnogram should return toward spontaneous rate.
        engine.set_vent_settings(rr=0.0, vt=0.0, peep=5.0, ie="1:2", mode="VCV")
        off_rr = self._count_insp_transitions(engine, seconds=20.0, dt=0.1)
        spont_rr_off = engine.state.rr
        assert abs(off_rr - spont_rr_off) < 2.0, \
            f"Capno RR {off_rr:.1f} should match spontaneous rate {spont_rr_off:.1f}"
