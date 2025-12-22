
from core.engine import SimulationEngine
from core.state import SimulationConfig
from patient.patient import Patient

class TestTemperature:
    def test_thermal_dynamics(self):
        """Verify patient cools down over time and Bair Hugger warms."""
        patient = Patient(weight=70, age=40)
        config = SimulationConfig(mode='steady_state')
        engine = SimulationEngine(patient, config)
        engine.start()
        
        # 1. Initial State
        assert engine.state.temp_c == 37.0, "Initial temp should be 37.0"
        
        # 2. Cooling Phase (Anesthetized)
        # 2. Cooling Phase (Anesthetized)
        # Advance 30 mins
        for _ in range(1800): 
            engine.step(1.0) 
            
        temp_cold = engine.state.temp_c
        assert temp_cold < 37.0, "Patient should cool down under anesthesia"
        
        # 3. Warming Phase (Bair Hugger ON)
        engine.set_bair_hugger(43.0) # High setting (43C)
        assert engine.state.bair_hugger_target == 43.0
        
        # Advance 30 mins
        for _ in range(1800):
            engine.step(1.0)
            
        temp_warmed = engine.state.temp_c
        
        # Cooling should be slowed or reversed
        # Bair Hugger adds ~50W.
        # Net flux = Prod - Loss + 50.
        # It should be significantly higher than previous phase.
        
        assert temp_warmed > temp_cold, "Warming should reverse cooling or at least be warmer than cold state"
        
    def test_redistribution_flux(self):
        """Verify rapid temperature drop on rapid depth increase (Bolus)."""
        patient = Patient(weight=70, age=40)
        # Use awake mode so we start with 0 drug
        engine = SimulationEngine(patient, SimulationConfig(mode='awake'))
        engine.start() # Start!
        
        # Stabilize
        engine.state.temp_c = 37.0
        engine.step(1.0) # Step 1: depth 0->0. d_depth=0.
        temp_start = engine.state.temp_c
        
        # Simulate Bolus: Force PK state to high concentration
        # engine.step PK will overwrite state, so we must set PK model state
        engine.pk_prop.state.ce = 4.0
        engine.pk_prop.state.c1 = 4.0 # plasma

        
        engine.step(1.0) # Step 2: depth 0->1.0. d_depth=1.0.
        temp_drop = engine.state.temp_c
        
        # Check if drop is significant
        delta = temp_start - temp_drop
        
        # With k=500,000, drop should be ~2 deg.
        # With k=50,000, drop ~0.2 deg.
        assert delta > 0.05, "Should see immediate drop (>0.05C) due to redistribution flux"
        
    def test_mac_adjustment(self):
        """Verify MAC requirement decreases (fraction increases) with hypothermia."""
        patient = Patient(weight=70, age=40)
        # Default MAC_40 ~ 2.1%
        pk = engine = SimulationEngine(patient, SimulationConfig(mode='steady_state')).pk_sevo
        
        # Force a concentration (e.g. 2.1% brain)
        pk.state.c_vrg = 0.021
        
        # Test at 37C
        pk.step(1.0, 0.021, 5.0, 5.0, temp_c=37.0)
        mac_37 = pk.state.mac
        
        # Test at 30C (Hypothermia)
        # Requirement drops by 5% per degree -> 35% drop
        # Factor = 1 - 0.35 = 0.65.
        # MAC Fraction should INCREASE by 1/0.65 ~ 1.5x
        pk.step(1.0, 0.021, 5.0, 5.0, temp_c=30.0)
        mac_30 = pk.state.mac
        assert mac_30 > mac_37, "MAC fraction should increase (requirement decrease) with hypothermia"

    def test_co2_production(self):
        """Verify VCO2 reduction with hypothermia affecting PaCO2."""
        patient = Patient(weight=70, age=40)
        engine = SimulationEngine(patient, SimulationConfig(mode='steady_state'))
        
        # Force consistent ventilation
        engine.vent.is_on = True
        engine.resp_mech.set_rr = 10
        engine.resp_mech.set_vt = 0.5
        engine.start() # Start simulation!
    
        # Stabilize at 37C
        engine.state.temp_c = 37.0
        for _ in range(300): engine.step(0.1)
        paco2_37 = engine.state.paco2
    
        # Drop temp to 30C
        engine.state.temp_c = 30.0
        
        # Step enough to equilibrate CO2 (tau ~ 3 mins)
        for _ in range(6000): engine.step(0.1) # 10 mins
        paco2_30 = engine.state.paco2
        assert paco2_30 < paco2_37, "PaCO2 should drop with hypothermia (reduced production)"
