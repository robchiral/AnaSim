"""
Long-Duration Simulation Stability Tests.

These tests verify numerical stability over extended simulation runs
(equivalent to 4+ hour anesthesia cases).

Validation criteria:
- No NaN/Inf values in state
- Physiological values remain in realistic ranges
- Hemodynamic state doesn't drift unrealistically
"""

import numpy as np
import pytest

from anasim.core.state import SimulationConfig

PERTURBATION_EVENTS = {
    120: lambda engine: engine.give_drug_bolus("Propofol", 200),
    300: lambda engine: engine.start_hemorrhage(500),
    420: lambda engine: (engine.stop_hemorrhage(), engine.give_fluid(1000)),
    540: lambda engine: engine.give_blood(300),
}


class TestLongDurationStability:
    """Verify stability over extended simulation runs."""
    
    @pytest.fixture
    def long_running_engine(self, engine_factory):
        """Create engine for extended simulation."""
        config = SimulationConfig(mode="steady_state", maint_type="tiva", dt=0.5)
        return engine_factory(config=config, start=True, age=45, sex="Male")
    
    def test_four_hour_stability(self, long_running_engine):
        """Run equivalent of 4-hour case and verify stability under routine perturbations."""
        engine = long_running_engine
        
        target_duration = 4 * 3600  # 4 hours in seconds
        step_size = 1.0  # 1 second steps
        steps = int(target_duration / step_size)
        
        # For test efficiency, sample every 60 steps (once per simulated minute)
        sample_interval = 60
        
        hr_samples = []
        map_samples = []
        spo2_samples = []
        bis_samples = []
        temp_samples = []
        etco2_samples = []
        co_samples = []
        
        for i in range(steps):
            engine.step(step_size)
            
            if i % sample_interval == 0:
                state = engine.state
                hr_samples.append(state.hr)
                map_samples.append(state.map)
                spo2_samples.append(state.spo2)
                bis_samples.append(state.bis)
                temp_samples.append(state.temp_c)
                etco2_samples.append(state.etco2)
                co_samples.append(state.co)
                
                # Check for NaN/Inf
                assert np.isfinite(state.hr), f"HR is NaN/Inf at step {i}"
                assert np.isfinite(state.map), f"MAP is NaN/Inf at step {i}"
                assert np.isfinite(state.spo2), f"SpO2 is NaN/Inf at step {i}"
                assert np.isfinite(state.bis), f"BIS is NaN/Inf at step {i}"
                assert np.isfinite(state.temp_c), f"Temp is NaN/Inf at step {i}"
                assert np.isfinite(state.etco2), f"EtCO2 is NaN/Inf at step {i}"
                assert np.isfinite(state.co), f"CO is NaN/Inf at step {i}"
        
        # Verify values stayed in physiological range throughout
        assert all(30 <= hr <= 180 for hr in hr_samples), "HR went out of physiological range"
        assert all(20 <= map <= 200 for map in map_samples), "MAP went out of physiological range"
        assert all(90 <= spo2 <= 100 for spo2 in spo2_samples), "SpO2 went out of physiological range"
        assert all(5 <= bis <= 95 for bis in bis_samples), "BIS went out of physiological range"
        assert all(32.0 <= temp <= 40.0 for temp in temp_samples), "Temperature went out of physiological range"
        # Perfusion-coupled EtCO2 can transiently fall below 20 with low CO.
        assert all(10 <= etco2 <= 60 for etco2 in etco2_samples), "EtCO2 went out of physiological range"
        assert all(co > 0.5 for co in co_samples), "CO dropped below viable range"
        
        # Verify no significant drift (final should be close to median)
        median_map = np.median(map_samples)
        final_map = map_samples[-1]
        assert abs(final_map - median_map) < 20, \
            f"MAP drifted significantly: median {median_map:.1f}, final {final_map:.1f}"

        for i in range(1800):
            event = PERTURBATION_EVENTS.get(i)
            if event:
                event(engine)
            engine.step(1.0)

        for name in ("hr", "map", "spo2", "etco2", "bis", "temp_c", "co"):
            value = getattr(engine.state, name)
            assert np.isfinite(value), f"{name} is not finite after perturbation sequence"
    
    def test_temperature_drift_reasonable(self, long_running_engine):
        """Temperature should drift slowly and reasonably over time."""
        engine = long_running_engine
        
        initial_temp = engine.state.temp_c
        
        # Run for 2 simulated hours
        for _ in range(7200):
            engine.step(1.0)
        
        final_temp = engine.state.temp_c
        
        # Temperature change should be gradual (< 2°C over 2 hours)
        temp_change = abs(final_temp - initial_temp)
        assert temp_change < 2.0, \
            f"Temperature changed too rapidly: {initial_temp:.1f} -> {final_temp:.1f}°C"
        
        # Should still be in viable range
        assert 33.0 <= final_temp <= 40.0, f"Temperature {final_temp:.1f}°C out of range"
