
import pytest
import unittest
import numpy as np
from anasim.patient.patient import Patient
from anasim.core.state import SimulationConfig, AirwayType
from anasim.core.engine import SimulationEngine
from anasim.physiology.hemodynamics import HemoState
from anasim.physiology.respiration import RespState


# --- Fixtures ---

@pytest.fixture
def engine(patient):
    config = SimulationConfig(mode="awake", dt=0.5)
    sim = SimulationEngine(patient, config)
    return sim

# --- Tests from test_backend_sanity.py ---

class TestPhysiologicalSanity:
    """Verifies that the simulation produces physiologically reasonable values."""
    
    def test_awake_baselines(self, engine):
        """Check vital signs for a resting, awake patient."""
        engine.start()
        # Connect mask for monitoring (default is now NONE)
        engine.set_airway_mode("Mask")
        
        # Settle for a few seconds
        for _ in range(50): 
            engine.step(0.1)
            
        state = engine.state
        
        # 1. Heart Rate: 50-100 bpm
        assert 50 <= state.hr <= 100, f"HR {state.hr} out of awake range"
        
        # 2. MAP: 70-110 mmHg
        assert 70 <= state.map <= 110, f"MAP {state.map} out of awake range"
        
        # 3. SpO2: > 95% on Room Air (healthy)
        assert state.spo2 > 95, f"SpO2 {state.spo2} too low for healthy awake patient"
        
        # 4. EtCO2: allow broader range with perfusion coupling.
        if state.rr > 0:
            assert 25 <= state.etco2 <= 50, f"EtCO2 {state.etco2} abnormal"
            
    def test_svr_units(self, engine):
        """SVR should be in Wood Units (mmHg*min/L), ~10-30."""
        assert 10.0 <= engine.state.svr <= 30.0, \
            f"SVR {engine.state.svr} is not in Wood Units! (Expected 10-30)"

    def test_default_continuous_fluids(self, patient):
        """Default continuous IV fluids should be 1 mL/kg/hr."""
        config = SimulationConfig(mode="steady_state", maint_type="tiva", dt=1.0)
        engine = SimulationEngine(patient, config)
        engine.start()

        expected_ml_hr = max(0.0, float(getattr(patient, "weight", 0.0)))
        assert engine.get_continuous_fluid_rate() == pytest.approx(expected_ml_hr, rel=0, abs=1e-6)

    def test_albumin_infusion_increases_volume(self, patient):
        """Albumin bolus should increase colloid totals and blood volume."""
        config = SimulationConfig(mode="steady_state", maint_type="tiva", dt=0.5)
        engine = SimulationEngine(patient, config)
        engine.start()

        assert engine.hemo is not None
        base_bv = engine.hemo.blood_volume

        engine.give_albumin(250)
        # Infuse for ~2 minutes of simulated time (rate = 150 mL/min).
        for _ in range(240):  # 120s at 0.5s step
            engine.step(0.5)

        assert engine.hemo.total_colloid_in_ml == pytest.approx(250.0, rel=0, abs=1e-2)
        assert engine.hemo.blood_volume > base_bv + 100.0

class TestPathologyResponse:
    """Verifies physiological responses to adverse events."""
    
    def test_hemorrhage_shock(self, engine):
        """
        REGRESSION TEST: Verify 'Unstressed Volume' fix.
        Massive hemorrhage should cause severe hypotension (Shock).
        """
        engine.start()
        # Run baseline
        for _ in range(10): engine.step(0.1)
        base_map = engine.state.map
        
        # Start Massive Hemorrhage (2L/min)
        engine.start_hemorrhage(2000.0)
        
        # Run for 60s
        for _ in range(600):
            engine.step(0.1)
            
        final_map = engine.state.map
        final_sv = engine.state.sv
        
        # Expect MAP to drop by at least 40%
        assert final_map < base_map * 0.6, "MAP did not drop significantly (<40%) during massive hemorrhage!"
        # SV should be critically low; allow slight tolerance for model variability.
        assert final_sv < 22.0, "Stroke Volume is unrealistically preserved during massive volume loss!"

class TestMonitorStability:
    """Verifies monitors do not crash the engine."""
    
    def test_bis_integration(self, engine):
        """
        REGRESSION TEST: BIS Monitor `step` method existence.
        """
        engine.start()
        try:
            # Run simulation which calls bis.step() internally
            for _ in range(20):
                engine.step(0.1)
            assert engine.state.bis > 0
        except AttributeError as e:
            pytest.fail(f"BIS Monitor caused crash: {e}")

class TestPharmacologyBasics:
    """Verifies PK/PD basics."""
    
    def test_propofol_bolus(self, engine):
        """Verify bolus increases concentration."""
        engine.start()
        engine.step(0.1)
        base_cp = engine.state.propofol_cp
        
        # Give 200mg Propofol
        engine.give_drug_bolus("Propofol", 200.0)
        engine.step(0.1)
        
        peak_cp = engine.state.propofol_cp
        assert peak_cp > base_cp + 1.0, f"Propofol Cp did not rise significantly after bolus (Got {peak_cp})"


@pytest.mark.parametrize(
    "drug,mode,pk_attr,controller_attr,state_values,max_rate",
    [
        ("propofol", "effect_site", "pk_prop", "tci_prop", {"c1": 1.0, "c2": 2.0, "c3": 3.0, "ce": 4.0}, "prop"),
        ("remi", "effect_site", "pk_remi", "tci_remi", {"c1": 1.5, "c2": 2.5, "c3": 3.5, "ce": 4.5}, "remi"),
        ("nore", "plasma", "pk_nore", "tci_nore", {"c1": 5.0, "c2": 2.0, "ce": 4.0}, "nore"),
        ("epi", "plasma", "pk_epi", "tci_epi", {"c1": 3.0, "ce": 2.0}, "epi"),
        ("phenyl", "plasma", "pk_phenyl", "tci_phenyl", {"c1": 6.0, "c2": 1.0, "ce": 2.5}, "phenyl"),
        ("roc", "effect_site", "pk_roc", "tci_roc", {"c1": 2.0, "c2": 2.0, "c3": 2.0, "ce": 2.0}, "roc"),
    ],
)
def test_tci_seeds_state_and_caps_rate(engine, drug, mode, pk_attr, controller_attr, state_values, max_rate):
    """TCI should seed from PK state and use a realistic max rate for each drug."""
    weight = engine.patient.weight
    pk_model = getattr(engine, pk_attr)
    for key, val in state_values.items():
        setattr(pk_model.state, key, val)

    engine.enable_tci(drug, 2.0, mode)
    controller = getattr(engine, controller_attr)
    assert controller is not None

    c1 = getattr(pk_model.state, "c1", 0.0)
    c2 = getattr(pk_model.state, "c2", 0.0)
    c3 = getattr(pk_model.state, "c3", 0.0)
    ce = getattr(pk_model.state, "ce", 0.0)

    assert controller.x[0, 0] == pytest.approx(c1)
    if controller.n_state >= 2:
        assert controller.x[1, 0] == pytest.approx(c2)
    if controller.n_state == 3:
        assert controller.x[2, 0] == pytest.approx(ce)
    if controller.n_state >= 4:
        assert controller.x[2, 0] == pytest.approx(c3)
        assert controller.x[3, 0] == pytest.approx(ce)

    if max_rate == "prop":
        expected = weight * 0.3 / 60.0
    elif max_rate == "remi":
        expected = weight * 0.5 / 60.0
    elif max_rate == "nore":
        expected = weight * 1.0 / 60.0
    elif max_rate == "epi":
        expected = weight * 0.5 / 60.0
    elif max_rate == "phenyl":
        expected = weight * 2.0 / 60.0
    else:
        expected = weight * 1.0 / 3600.0
    assert controller.max_rate == pytest.approx(expected)

def test_hr_disturbance_not_double_applied():
    """HR disturbance should be applied only once (physiology, not display)."""
    patient = Patient(age=40, weight=70, height=170, sex="male")
    config = SimulationConfig(mode="awake", dt=0.5)

    def build_engine():
        eng = SimulationEngine(patient, config)
        eng.state.time = 0.0
        eng._next_nibp_time = 1e9
        eng.state.airway_mode = AirwayType.MASK
        eng.rng = np.random.default_rng(123)
        return eng

    engine1 = build_engine()
    engine2 = build_engine()

    hemo_state = HemoState(map=80.0, hr=85.0, sv=70.0, svr=16.0, co=5.0)
    resp_state = RespState(
        rr=12.0,
        vt=500.0,
        mv=6.0,
        va=4.0,
        apnea=False,
        p_alveolar_co2=40.0,
        etco2=40.0,
        p_arterial_o2=95.0,
        drive_central=1.0,
        muscle_factor=1.0,
    )
    engine1.smooth_hr = hemo_state.hr
    engine2.smooth_hr = hemo_state.hr

    engine1._step_monitors(1.0, "EXP", hemo_state, resp_state, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    engine2._step_monitors(1.0, "EXP", hemo_state, resp_state, (0.0, 0.0, 0.0, 0.0, 0.0, 10.0))

    assert engine1.state.hr == pytest.approx(engine2.state.hr, rel=1e-6)

def test_lbm_fallback_for_extreme_bmi():
    """LBM should remain positive for extreme BMI values."""
    patient = Patient(age=40, weight=300.0, height=160.0, sex="male")
    assert patient.lbm > 0.0

def test_pk_hemodynamic_scaling_applies_to_propofol():
    """PK central volume should scale with blood volume ratio."""
    patient = Patient(age=40, weight=70, height=170, sex="male")
    config = SimulationConfig(mode="awake", dt=0.5)
    engine = SimulationEngine(patient, config)
    base_v1 = engine.pk_prop.v1
    engine.hemo.blood_volume = engine.hemo.blood_volume_0 * 0.5
    engine.state.co = engine.hemo.base_co_l_min * 0.5

    engine._step_pk(0.5, 0.0, engine.state.co)

    assert engine.pk_prop.v1 == pytest.approx(base_v1 * 0.5, rel=0.05)

# --- Coupling / Integration Tests ---

def _run_for(engine, seconds: float, dt: float = 0.1) -> None:
    steps = max(1, int(seconds / dt))
    for _ in range(steps):
        engine.step(dt)

def test_peep_increases_pit_and_reduces_preload():
    """Higher PEEP should raise Pit and reduce preload via coupling."""
    patient = Patient(age=40, weight=70, height=170, sex="male")
    config = SimulationConfig(mode="awake", dt=0.5)
    engine = SimulationEngine(patient, config)
    engine.start()
    engine.set_airway_mode("Mask")

    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    _run_for(engine, 20.0)
    pit_low = engine.state.pit
    preload_low = engine.hemo.state.preload_factor

    engine.set_vent_settings(rr=12, vt=0.5, peep=15.0, ie="1:2", mode="VCV")
    _run_for(engine, 20.0)
    pit_high = engine.state.pit
    preload_high = engine.hemo.state.preload_factor

    assert pit_high > pit_low + 0.5, "Higher PEEP should increase Pit"
    assert preload_high < preload_low, "Higher PEEP should reduce preload factor"

def test_positive_pressure_reduces_preload_vs_spontaneous():
    """Positive pressure ventilation should reduce preload vs spontaneous breathing."""
    patient = Patient(age=40, weight=70, height=170, sex="male")
    config = SimulationConfig(mode="awake", dt=0.5)
    engine = SimulationEngine(patient, config)
    engine.start()
    engine.set_airway_mode("Mask")

    engine.vent.is_on = False
    engine.bag_mask_active = False
    _run_for(engine, 20.0)
    preload_spont = engine.hemo.state.preload_factor

    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    _run_for(engine, 20.0)
    preload_vent = engine.hemo.state.preload_factor

    assert preload_vent < preload_spont, "Positive pressure should reduce preload vs spontaneous"

# --- Tests from test_steady_state.py ---

class TestSteadyStateMode(unittest.TestCase):
    def setUp(self):
        self.patient = Patient(age=40, weight=70, height=170, sex="male")

    def test_steady_state_tiva_init(self):
        config = SimulationConfig(mode='steady_state', maint_type='tiva')
        engine = SimulationEngine(self.patient, config)
        
        # Check Propofol Pre-fill (engine uses 3.0 for steady state)
        self.assertAlmostEqual(engine.pk_prop.state.c1, 3.0, delta=0.2)
        
        # Check Remi (engine uses 2.0 for steady state)
        self.assertAlmostEqual(engine.pk_remi.state.c1, 2.0, delta=0.2)
        
        # Check TCI Enabled
        self.assertTrue(engine.tci_prop is not None)
        self.assertEqual(engine.tci_prop.target, 3.0)
        
        engine.start() # Start simulation loop
        
        # Check Hemodynamic Init
        # MAP should be lower than baseline (usually ~80-90) due to Propofol 3.5
        self.assertLess(engine.state.map, 95.0)
        self.assertGreater(engine.state.map, 50.0) # Not crashed
        
        # Run for 2 seconds (200 steps at 0.01) and ensure MAP stays stable
        initial_map = engine.state.map
        
        for _ in range(200):
            engine.step(0.01)
            
        self.assertAlmostEqual(engine.state.map, initial_map, delta=10.0)
        # BIS settles to steady-state value for Prop 3.0 + Remi 2.0 (~45-60)
        self.assertAlmostEqual(engine.state.bis, 55.0, delta=8.0)
