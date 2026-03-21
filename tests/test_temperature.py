
from anasim.core.state import SimulationConfig


def _advance(engine, seconds: float, dt: float = 1.0) -> None:
    for _ in range(int(seconds / dt)):
        engine.step(dt)


class TestTemperature:
    def test_thermal_dynamics(self, engine_factory):
        """Verify patient cools down over time and Bair Hugger warms."""
        config = SimulationConfig(mode='steady_state')
        engine = engine_factory(config=config, start=True)
        
        assert engine.state.temp_c == 37.0, "Initial temp should be 37.0"
        
        # Cooling phase: advance 20 min under anesthesia.
        _advance(engine, 20 * 60, dt=2.0)
            
        temp_cold = engine.state.temp_c
        assert temp_cold < 37.0, "Patient should cool down under anesthesia"
        
        # Warming phase: Bair Hugger high setting (43C).
        engine.set_bair_hugger(43.0)
        assert engine.state.bair_hugger_target == 43.0
        
        _advance(engine, 20 * 60, dt=2.0)
            
        temp_warmed = engine.state.temp_c
        
        # Warming should reverse or slow cooling.
        assert temp_warmed > temp_cold, "Warming should reverse cooling or at least be warmer than cold state"

    def test_co2_production(self, engine_factory):
        """Verify VCO2 reduction with hypothermia affecting PaCO2."""
        engine = engine_factory(config=SimulationConfig(mode='steady_state'))
        
        # Force consistent ventilation.
        engine.vent.is_on = True
        engine.resp_mech.set_rr = 10
        engine.resp_mech.set_vt = 0.5
        engine.start()
    
        engine.state.temp_c = 37.0
        _advance(engine, 60, dt=1.0)
        paco2_37 = engine.state.pa_co2
    
        engine.state.temp_c = 30.0
        
        # Step enough to equilibrate CO2 (tau ~3 min) without oversampling.
        _advance(engine, 600, dt=1.0)
        paco2_30 = engine.state.pa_co2
        assert paco2_30 < paco2_37, "PaCO2 should drop with hypothermia (reduced production)"
