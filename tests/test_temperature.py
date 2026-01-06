
from anasim.core.state import SimulationConfig

class TestTemperature:
    def test_thermal_dynamics(self, engine_factory):
        """Verify patient cools down over time and Bair Hugger warms."""
        config = SimulationConfig(mode='steady_state')
        engine = engine_factory(config=config, start=True)
        
        assert engine.state.temp_c == 37.0, "Initial temp should be 37.0"
        
        # Cooling phase: advance 30 min under anesthesia.
        for _ in range(1800): 
            engine.step(1.0) 
            
        temp_cold = engine.state.temp_c
        assert temp_cold < 37.0, "Patient should cool down under anesthesia"
        
        # Warming phase: Bair Hugger high setting (43C).
        engine.set_bair_hugger(43.0)
        assert engine.state.bair_hugger_target == 43.0
        
        for _ in range(1800):
            engine.step(1.0)
            
        temp_warmed = engine.state.temp_c
        
        # Warming should reverse or slow cooling.
        assert temp_warmed > temp_cold, "Warming should reverse cooling or at least be warmer than cold state"
        
    def test_redistribution_flux(self, engine_factory):
        """Verify rapid temperature drop on rapid depth increase (Bolus)."""
        engine = engine_factory(config=SimulationConfig(mode='awake'), start=True)
        
        engine.state.temp_c = 37.0
        engine.step(1.0) # Step 1: depth 0->0. d_depth=0.
        temp_start = engine.state.temp_c
        
        # Simulate bolus by forcing PK state (step() will overwrite defaults).
        engine.pk_prop.state.ce = 4.0
        engine.pk_prop.state.c1 = 4.0 # plasma

        
        engine.step(1.0) # Step 2: depth 0->1.0. d_depth=1.0.
        temp_drop = engine.state.temp_c
        
        delta = temp_start - temp_drop
        
        # Expect a measurable drop across likely k values.
        assert delta > 0.05, "Should see immediate drop (>0.05C) due to redistribution flux"
        
    def test_mac_adjustment(self, engine_factory):
        """Verify MAC requirement decreases (fraction increases) with hypothermia."""
        engine = engine_factory(config=SimulationConfig(mode='steady_state'))
        pk = engine.pk_sevo
        
        # Force a concentration (e.g. 2.1% brain).
        pk.state.c_vrg = 0.021
        
        pk.step(1.0, 0.021, 5.0, 5.0, temp_c=37.0)
        mac_37 = pk.state.mac
        
        # Hypothermia should reduce requirement, increasing MAC fraction.
        pk.step(1.0, 0.021, 5.0, 5.0, temp_c=30.0)
        mac_30 = pk.state.mac
        assert mac_30 > mac_37, "MAC fraction should increase (requirement decrease) with hypothermia"

    def test_co2_production(self, engine_factory):
        """Verify VCO2 reduction with hypothermia affecting PaCO2."""
        engine = engine_factory(config=SimulationConfig(mode='steady_state'))
        
        # Force consistent ventilation.
        engine.vent.is_on = True
        engine.resp_mech.set_rr = 10
        engine.resp_mech.set_vt = 0.5
        engine.start()
    
        engine.state.temp_c = 37.0
        for _ in range(300): engine.step(0.1)
        paco2_37 = engine.state.paco2
    
        engine.state.temp_c = 30.0
        
        # Step enough to equilibrate CO2 (tau ~3 min).
        for _ in range(6000): engine.step(0.1) # 10 mins
        paco2_30 = engine.state.paco2
        assert paco2_30 < paco2_37, "PaCO2 should drop with hypothermia (reduced production)"
