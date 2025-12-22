
import pytest
import unittest
from physiology.hemodynamics import HemodynamicModel, HemoStateExtended, HemoState
from patient.patient import Patient

# --- Unit Tests ---

class TestHemodynamicsUnit(unittest.TestCase):
    """Basic initialization and mechanics tests."""
    
    def setUp(self):
        self.patient = Patient(age=40, weight=70, height=170, sex="male")
        self.model = HemodynamicModel(self.patient)
        self.model.tde_hr = 0.0 # Remove drift for deterministic testing
        self.model.tde_sv = 0.0
        
    def test_initialization(self):
        # Initial step
        state = self.model.step(0.01, 0,0,0, -2, 40, 95)
        # MAP ~90
        self.assertTrue(80 < state.map < 120)
        self.assertTrue(self.model.blood_volume > 0)
        
    def test_steady_state_calculation_type(self):
        ss_state = self.model.calculate_steady_state(4.0, 0, 0)
        self.assertIsInstance(ss_state, HemoStateExtended)

# --- Sanity Checks ---

class TestHemodynamicSanity:
    """
    Validates model against physiological literature values.
    """
    
    @pytest.fixture
    def model(self):
        p = Patient(age=40, weight=70, sex="male")
        m = HemodynamicModel(p)
        m.tde_hr = 0.0
        m.tde_sv = 0.0
        return m
        
    def test_baseline_values(self, model):
        """Values should be within normal adult physiological ranges."""
        state = model.step(1.0, 0,0,0, -2, 40, 95)
        
        # CO: 4.0 - 7.0 L/min
        assert 3.5 < state.co < 8.0, f"CO {state.co} out of range"
        # SVR: 800-1200 dyn or 10-30 Wood units
        assert 8.0 < state.svr < 30.0, f"SVR {state.svr} out of range"
        
    def test_propofol_hypotension(self, model):
        """Propofol 4 ug/mL -> 20-30% MAP drop."""
        base_state = model.step(1.0, 0,0,0, -2, 40, 95)
        
        # 10 min infusion effect (steady state approx)
        for _ in range(600): state = model.step(1.0, 4.0, 0,0, -2, 40, 95)
            
        drop = (base_state.map - state.map) / base_state.map
        assert 0.15 < drop < 0.45, f"Propofol drop {drop:.1%} expected 15-45%"
        
    def test_epinephrine_biphasic(self, model):
        """
        Low dose (1 ng/mL) -> HR++ (Beta1), SVR--/neutral (Beta2).
        High dose (10+ ng/mL) -> MAP+++, SVR++ (Alpha).
        """
        base = model.step(1.0, 0,0,0, -2, 40, 95)
        
        # --- Low Dose ---
        low = model.step(1.0, 0,0,0, -2, 40, 95, ce_epi=1.0)
        assert low.hr > base.hr + 1.0, "Low dose epi should raise HR"
        # SVR should not rise significantly (vasodilation offsets alpha)
        assert low.svr < base.svr * 1.05
        
        # --- High Dose ---
        high = model.step(1.0, 0,0,0, -2, 40, 95, ce_epi=20.0)
        assert high.map > base.map + 20.0, "High dose epi should raise MAP significantly"
        assert high.svr > base.svr * 1.10, "High dose epi should raise SVR (Alpha)"
        
    def test_phenylephrine_alpha_only(self, model):
        """
        Phenyl -> SVR increase, MAP increase. 
        NO direct HR/Contractility increase (Beta effects absent).
        """
        base = model.step(1.0, 0,0,0, -2, 40, 95)
        
        # High dose phenyl
        phenyl = model.step(1.0, 0,0,0, -2, 40, 95, ce_phenyl=100.0) # High dose for clarity
        
        assert phenyl.map > base.map + 10.0, "Phenyl should raise MAP"
        assert phenyl.svr > base.svr + 4.0, "Phenyl should raise SVR"
        
        # Check internal 'star' parameters which drive direct cardiac effects
        # (Assuming model exposes them or we infer from lack of massive HR/SV jump despite afterload)
        # SVR increase usually lowers SV due to afterload, unless inotropy increases.
        # So SV should likely drop or stay same, definitely not rise huge.
        assert phenyl.sv <= base.sv * 1.1, "Phenyl should not cause large SV increase (no beta)"
        
    def test_hemorrhage_shock_class_iii(self, model):
        """
        35% blood loss -> Tachycardia, Hypotension.
        """
        base = model.step(1.0, 0,0,0, -2, 40, 95)
        init_vol = model.blood_volume
        
        # Remove 35%
        model.blood_volume = init_vol * 0.65
        
        # Compensation time
        for _ in range(300):
            state = model.step(1.0, 0,0,0, -2, 40, 95)
            
        assert state.hr > base.hr + 20, "Hemorrhage should cause tachycardia > 20 bpm"
        assert state.map < base.map - 10, "Hemorrhage should cause hypotension"


class TestAnemiaHandling:
    @pytest.fixture
    def model(self):
        p = Patient(age=40, weight=70, sex="male", baseline_hb=13.5, baseline_hct=0.42)
        m = HemodynamicModel(p)
        m.tde_hr = 0.0
        m.tde_sv = 0.0
        return m

    def test_hemodilution_decreases_hemoglobin(self, model):
        hb_before = model.hb_conc
        model.add_volume(1000.0, hematocrit=0.0)
        assert model.hb_conc < hb_before

    def test_blood_transfusion_restores_hemoglobin(self, model):
        model.add_volume(1000.0, hematocrit=0.0)
        hb_mass_after_dilution = model.hb_mass
        model.add_volume(500.0, hematocrit=0.45)
        assert model.hb_mass > hb_mass_after_dilution

    def test_do2_ratio_tracks_changes(self, model):
        state = model.step(1.0, 0, 0, 0, -2, 40, 95)
        ratio = model.compute_do2_ratio(sao2_frac=0.98, pao2=95.0, co_l_min=state.co)
        assert 0.8 < ratio < 1.2
        # Severe anemia and low CO -> ratio well below 1
        model.add_volume(1500.0, hematocrit=0.0)
        low_ratio = model.compute_do2_ratio(sao2_frac=0.9, pao2=60.0, co_l_min=state.co * 0.6)
        assert low_ratio < 0.8

# --- Interaction Tests ---

class TestDrugInteractions:
    
    @pytest.fixture
    def model(self):
        p = Patient(age=40, weight=70, sex="male")
        m = HemodynamicModel(p)
        m.tde_hr = 0.0; m.tde_sv = 0.0
        return m
        
    def test_propofol_phenylephrine_interaction(self, model):
        """
        Scenario: Patient hypotensive from Propofol.
        Intervention: Phenylephrine bolus/infusion.
        Result: MAP should improve.
        """
        # 1. Establish Propofol Hypotension
        for _ in range(300):
            state_prop = model.step(1.0, 4.0, 0,0, -2, 40, 95)
        
        hypotensive_map = state_prop.map
        
        # 2. Add Phenylephrine (e.g., 20 ng/mL)
        for _ in range(60):
            state_combined = model.step(1.0, 4.0, 0,0, -2, 40, 95, ce_phenyl=20.0)
            
        restored_map = state_combined.map
        
        assert restored_map > hypotensive_map + 5.0, \
            "Phenylephrine should restore MAP during Propofol anesthesia"
            
    def test_hemorrhage_norepinephrine_interaction(self, model):
        """
        Scenario: Hypovolemic shock (MAP low).
        Intervention: Norepinephrine.
        Result: MAP increases (vasoconstriction), though volume is still low.
        """
        # 1. Hypovolemia
        vol = model.blood_volume
        model.blood_volume = vol * 0.85 # 15% loss (20% depleted all stressed volume)
        
        for _ in range(300):
            shock_state = model.step(1.0, 0,0,0, -2, 40, 95)
            
        shock_map = shock_state.map
        
        # Verify we aren't on the floor (usually 30) so we can see improvement
        assert shock_map > 31.0, f"Shock MAP {shock_map} too low to test improvement (hit floor)"
        
        # 2. Norepinephrine (10 ng/mL)
        for _ in range(60):
            treat_state = model.step(1.0, 0, 0, 10.0, -2, 40, 95, ce_epi=0.0, ce_phenyl=0.0)
        
        assert treat_state.map > shock_map + 8.0, \
            "Norepinephrine should raise MAP even in hypovolemia (vasoconstriction)"
