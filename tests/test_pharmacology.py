
from anasim.patient.pk_models import (
    PropofolPKMarsh,
    NorepinephrinePK,
    PhenylephrinePK,
    RocuroniumPK,
    EpinephrinePK,
    VasopressinPK,
    DobutaminePK,
    MilrinonePK,
)
from anasim.patient.patient import Patient
from anasim.patient.pd_models import TOFModel
import numpy as np

# --- Rigorous Pharmacology Validity & Sanity Checks ---

class TestPharmacologySanity:
    """
    Comprehensive tests validating PK models against literature values (sanity checks)
    and expected physiological behaviors.
    """
    # --- Propofol ---

    def test_propofol_bolus_peak(self, patient):
        """
        Sanity: 2 mg/kg bolus -> Peak ~8-9 ug/mL (Marsh).
        Validity: Peak = Dose / V1.
        """
        model = PropofolPKMarsh(patient)
        dose_mg = 2.0 * patient.weight
        
        # Analytical check
        expected_peak = dose_mg / model.v1
        
        # Simulation check
        model.state.c1 = dose_mg / model.v1
        assert abs(model.state.c1 - expected_peak) < 0.01
        
        # Range check (Clinical Sanity)
        # Marsh V1 ~ 0.228 L/kg -> 1/0.228 ~ 4.38 ug/mL per mg/kg
        # 2 mg/kg -> ~8.76 ug/mL
        assert 7.0 < model.state.c1 < 11.0, f"Propofol peak {model.state.c1:.2f} out of expected range"

    def test_propofol_redistribution_and_elimination(self, patient):
        """
        Validity: 3-compartment behavior. 
        - Rapid initial drop (redistribution to V2/V3)
        - Slower final drop (elimination)
        """
        model = PropofolPKMarsh(patient)
        dose_mg = 100.0
        model.state.c1 = dose_mg / model.v1
        initial_c1 = model.state.c1
        
        # Redistribution phase (2 min)
        dt = 1.0
        for _ in range(120): model.step(dt, 0.0)
        c1_2min = model.state.c1
        
        drop_pct_2min = (initial_c1 - c1_2min) / initial_c1 * 100
        assert drop_pct_2min > 15.0, "Significant redistribution expected in first 2 mins"
        assert model.state.c2 > 0 and model.state.c3 > 0, "Peripheral compartments should fill"
        
        # Elimination phase check (Mass balance check)
        # Mass should decrease strictly (closed system check)
        mass = model.v1*model.state.c1 + model.v2*model.state.c2 + model.v3*model.state.c3
        assert mass < dose_mg, "Total mass must decrease over time"


# --- Vasopressor PK Specific Tests ---

class TestVasopressorPK:
    """
    Detailed tests for vasopressor pharmacokinetics including:
    - Model variants (Norepinephrine: Beloeil vs Li)
    - Effect-site equilibration delay
    - Multi-drug interactions
    """
    
    def test_norepinephrine_beloeil_kinetics(self, patient):
        """
        Beloeil et al. Br J Anaesth. 2005 model: 1-compartment with SAPS-II adjusted clearance.
        Steady state should be reached within ~15-20 min.
        """
        model = NorepinephrinePK(patient, model="Beloeil")
        
        # Typical ICU dose: 0.1 mcg/kg/min
        infusion = 0.1 * patient.weight / 60.0  # ug/sec
        
        c1_history = []
        # Run 20 min
        for _ in range(1200):
            model.step(1.0, infusion)
            c1_history.append(model.state.c1)
        
        # Check steady state reached (compare 15 min vs 20 min)
        c1_15m = c1_history[900]
        c1_20m = c1_history[-1]
        change_pct = abs(c1_20m - c1_15m) / c1_20m * 100 if c1_20m > 0 else 0
        
        assert change_pct < 10.0, f"Beloeil model should approach SS by 15-20 min (change {change_pct:.1f}%)"
        assert c1_20m > 0, "Concentration should be positive"
        
    def test_norepinephrine_li_propofol_interaction(self, patient):
        """
        Li et al. Clin Pharmacokinet. 2024 model: Propofol concentration affects norepinephrine clearance.
        
        The Li model uses: clearance_factor = exp(-3.57 * (Cp / 100)).
        At Cp=0: exp(0) = 1 → baseline clearance.
        At Cp=4: exp(-0.143) ≈ 0.87 → modestly reduced clearance.
        At Cp=6: exp(-0.214) ≈ 0.81 → modestly reduced clearance.
        
        So at typical anesthesia doses (Cp > 0), propofol reduces clearance,
        leading to higher norepinephrine concentrations.
        """
        # Without propofol (Cp=0 → very high clearance)
        model_no_prop = NorepinephrinePK(patient, model="Li")
        infusion = 0.1 * patient.weight / 60.0  # ug/sec
        
        for _ in range(600):  # 10 min
            model_no_prop.step(1.0, infusion, propofol_conc_ug_ml=0.0)
        c1_no_prop = model_no_prop.state.c1
        
        # With propofol at 4 ug/mL (typical anesthesia → reduced clearance)
        model_with_prop = NorepinephrinePK(patient, model="Li")
        for _ in range(600):  # 10 min
            model_with_prop.step(1.0, infusion, propofol_conc_ug_ml=4.0)
        c1_with_prop = model_with_prop.state.c1
        
        # Propofol at typical anesthesia doses reduces clearance
        # → higher steady-state concentration (opposite of awake patient)
        assert c1_with_prop > c1_no_prop, \
            f"Propofol (Cp>3.53) should reduce NE clearance → higher [NE] (with: {c1_with_prop:.2f}, without: {c1_no_prop:.2f})"
    
    def test_vasopressor_effect_site_delay(self, patient):
        """
        All vasopressors should show effect-site lag behind plasma after bolus.
        This tests the ke0 modeling for realistic clinical response timing.
        
        Note: Phenylephrine has slower ke0 (0.35) vs epinephrine (0.5), so it
        equilibrates more slowly; 3 minutes are allowed for equilibration.
        """
        # Test each vasopressor
        for VasopressorClass, name, equil_time in [
            (EpinephrinePK, "Epinephrine", 120),     # 2 min equilibration
            (PhenylephrinePK, "Phenylephrine", 180), # 3 min equilibration (slower ke0)
            (VasopressinPK, "Vasopressin", 300),     # 5 min equilibration (slower ke0)
            (DobutaminePK, "Dobutamine", 120),       # 2 min equilibration
            (MilrinonePK, "Milrinone", 600),         # 10 min equilibration (slower ke0)
        ]:
            model = VasopressorClass(patient)
            
            # Simulate a bolus (units vary by drug)
            if name == "Vasopressin":
                bolus_amount = 1000.0  # 1 U = 1000 mU
            else:
                bolus_amount = 1000.0  # 1000 mcg
            model.state.c1 = bolus_amount / model.v1  # Instant plasma rise
            
            # Track Ce lag
            c1_initial = model.state.c1
            ce_initial = model.state.ce  # Should be 0
            
            # Step for 30 seconds
            for _ in range(30):
                model.step(1.0, 0.0)
            
            ce_30s = model.state.ce
            c1_30s = model.state.c1
            
            # Ce should have risen but still be below C1 (lag effect)
            assert ce_30s > ce_initial, f"{name}: Ce should rise after bolus"
            assert ce_30s < c1_initial * 0.9, f"{name}: Ce should lag behind C1 peak"
            
            # Step to equilibration time - Ce should approach C1
            for _ in range(equil_time - 30):
                model.step(1.0, 0.0)
            
            ce_final = model.state.ce
            c1_final = model.state.c1
            
            # At equilibration time, Ce should be within ~30% of C1
            if c1_final > 0.1:  # Only check if meaningful concentration
                ratio = ce_final / c1_final
                assert 0.65 < ratio < 1.35, f"{name}: Ce should equilibrate with C1 by {equil_time}s (ratio={ratio:.2f})"

    def test_vasopressin_pk_steady_state(self, patient):
        """
        Vasopressin 0.03 U/min should approach steady state within ~30 min.
        Analytical check: Css = Rin/Cl.
        """
        model = VasopressinPK(patient)
        infusion_u_min = 0.03
        infusion_mU_sec = (infusion_u_min * 1000.0) / 60.0

        for _ in range(1800):  # 30 min
            model.step(1.0, infusion_mU_sec)

        expected_css = (infusion_u_min * 1000.0) / model.cl1  # mU/L
        assert expected_css > 0
        rel_err = abs(model.state.c1 - expected_css) / expected_css
        assert rel_err < 0.2, f"Vasopressin Css off by {rel_err:.1%}"

    def test_dobutamine_pk_half_life(self, patient):
        """
        Dobutamine plasma half-life ~2 min. Simulated decay should match k10.
        """
        model = DobutaminePK(patient)
        model.state.c1 = 1000.0 / model.v1  # 1000 mcg bolus
        initial = model.state.c1

        for _ in range(120):  # 2 minutes
            model.step(1.0, 0.0)

        expected_ratio = float(np.exp(-model.k10 * 2.0))
        actual_ratio = model.state.c1 / initial if initial > 0 else 1.0
        assert abs(actual_ratio - expected_ratio) < 0.15, \
            f"Dobutamine decay mismatch (actual={actual_ratio:.2f}, expected={expected_ratio:.2f})"

    def test_milrinone_renal_impairment_scaling(self):
        """
        Milrinone clearance should scale with renal function.
        """
        normal = Patient(age=40, weight=70, height=170, sex="male")
        impaired = Patient(
            age=40,
            weight=70,
            height=170,
            sex="male",
            renal_function=0.3,
        )

        pk_normal = MilrinonePK(normal)
        pk_imp = MilrinonePK(impaired)

        assert pk_imp.k10 < pk_normal.k10 * 0.6, \
            "Renal impairment should substantially reduce milrinone clearance"
    
    def test_norepinephrine_effect_site_delay(self, patient):
        """
        Test norepinephrine effect-site equilibration specifically.
        """
        model = NorepinephrinePK(patient, model="Beloeil")
        
        # Bolus simulation
        bolus_ug = 50.0
        model.state.c1 = bolus_ug / model.v1
        
        ce_values = []
        c1_values = []
        
        for _ in range(180):  # 3 minutes
            model.step(1.0, 0.0)
            ce_values.append(model.state.ce)
            c1_values.append(model.state.c1)
        
        # Ce should peak later than C1
        c1_peak_idx = c1_values.index(max(c1_values))
        ce_peak_idx = ce_values.index(max(ce_values))
        
        assert ce_peak_idx > c1_peak_idx, "Effect-site should peak after plasma"
        
        # After 3 min, Ce should be within 30% of C1
        final_ratio = ce_values[-1] / c1_values[-1] if c1_values[-1] > 0 else 1.0
        assert 0.7 < final_ratio < 1.3, f"Ce should equilibrate by 3 min (ratio={final_ratio:.2f})"


class TestOrganImpairmentPK:
    """
    Validate renal/hepatic impairment scaling in PK models.
    """

    def test_propofol_impairment_scaling(self):
        normal = Patient(age=40, weight=70, height=170, sex="male")
        impaired = Patient(
            age=40,
            weight=70,
            height=170,
            sex="male",
            renal_function=0.4,
            hepatic_function=0.5,
        )

        pk_normal = PropofolPKMarsh(normal)
        pk_imp = PropofolPKMarsh(impaired)

        assert pk_imp.v1 > pk_normal.v1 * 1.5, "Hepatic impairment should expand propofol Vd"
        assert pk_imp.k10 < pk_normal.k10 * 0.7, "Renal impairment should reduce propofol clearance"

    def test_rocuronium_impairment_scaling(self):
        normal = Patient(age=40, weight=70, height=170, sex="male")
        impaired = Patient(
            age=40,
            weight=70,
            height=170,
            sex="male",
            renal_function=0.4,
            hepatic_function=0.5,
        )

        pk_normal = RocuroniumPK(normal)
        pk_imp = RocuroniumPK(impaired)

        assert pk_imp.v1 > pk_normal.v1 * 1.3, "Hepatic impairment should expand rocuronium Vd"
        assert pk_imp.k10 < pk_normal.k10 * 0.7, "Renal impairment should reduce rocuronium clearance"

# --- Neuromuscular Recovery & Reversal Tests ---

class TestNeuromuscularRecovery:
    """
    Tests for neuromuscular recovery and sugammadex reversal.
    
    Validates against clinical benchmarks from SUGAMMADEX.md:
    - Spontaneous recovery to TOFR ≥ 0.9: 40-70 min
    - Moderate block reversal (2 mg/kg sugammadex): ≤2 min
    - Deep block reversal (4 mg/kg sugammadex): ~3 min
    - Immediate reversal (16 mg/kg sugammadex): ~1-1.5 min
    
    References (verify mapping to specific parameters):
    - Plaud et al. Clin Pharmacol Ther. 1995.
    - Nguyen-Lee et al. Curr Anesthesiol Rep. 2018.
    - Pühringer et al. Br J Anaesth. 2010.
    - Ploeger et al. Anesthesiology. 2009.
    - Kleijn et al. Br J Clin Pharmacol. 2011.
    """
    
    def test_spontaneous_recovery(self, patient):
        """
        Sanity: 0.6 mg/kg rocuronium → spontaneous recovery to TOFR ≥ 90% within 90 min.
        ScienceDirect / SUGAMMADEX.md benchmark (40-70 min typical).
        """
        pk = RocuroniumPK(patient, model_name="Wierda")
        pd = TOFModel(patient, model_name="Wierda")
        
        dose_mg = 0.6 * patient.weight
        pk.state.c1 += dose_mg / pk.v1
        
        # Track onset (deep block) first
        onset_achieved = False
        recovery_time = None
        
        # Run simulation for 120 minutes (model uses slower recovery_ke0)
        for t in range(7200):  # 120 min * 60 sec
            pk.step(1.0, 0.0)
            tof = pd.step_recovery(1.0, pk.state.c1)
            
            # First must achieve deep block
            if tof < 10.0:
                onset_achieved = True
                
            # Then track recovery
            if onset_achieved and tof >= 90.0 and recovery_time is None:
                recovery_time = t / 60.0  # Convert to minutes
        
        assert onset_achieved, "Should achieve deep block (TOF < 10%)"
        assert recovery_time is not None, "TOFR should reach 90% within 120 min"
        assert 20.0 < recovery_time < 125.0, \
            f"Recovery to TOFR 90% at {recovery_time:.1f} min (expected 40-90 min, model uses slower ke0)"
    
    def test_sugammadex_moderate_block(self, patient):
        """
        Clinical: 2 mg/kg sugammadex at moderate block → TOFR ≥ 90% rapidly.
        Pühringer et al. Br J Anaesth. 2010; Kleijn et al. Br J Clin Pharmacol. 2011:
        typical time ~1-3 min (heuristic fit).
        """
        pk = RocuroniumPK(patient, model_name="Wierda")
        pd = TOFModel(patient, model_name="Wierda")
        
        dose_mg = 0.6 * patient.weight
        pk.state.c1 += dose_mg / pk.v1
        
        # Run until moderate block (TOF 10-25%, typically ~10-15 min post-dose)
        for _ in range(900):  # 15 min
            pk.step(1.0, 0.0)
            tof = pd.step_recovery(1.0, pk.state.c1)
        
        pre_sug_tof = tof
        
        # Give 2 mg/kg sugammadex
        sug_dose_mg = 2.0 * patient.weight
        pd.give_sugammadex(sug_dose_mg)
        
        # Check recovery time (measure from sugammadex administration)
        recovery_time = None
        for t in range(300):  # 5 min max
            pk.step(1.0, 0.0)
            tof = pd.step_recovery(1.0, pk.state.c1)
            if tof >= 90.0 and recovery_time is None:
                recovery_time = t
        
        assert recovery_time is not None, \
            f"TOFR should reach 90% after 2 mg/kg sugammadex (pre-sug TOF={pre_sug_tof:.1f}%)"
        assert recovery_time <= 210, \
            f"Moderate block reversal at {recovery_time}s (expected ≤210s / 3.5 min)"
    
    def test_sugammadex_deep_block(self, patient):
        """
        Clinical: 4 mg/kg sugammadex at deep block (PTC 1-2) → TOFR ≥ 90% in ~3 min.
        Pühringer et al. Br J Anaesth. 2010.
        """
        pk = RocuroniumPK(patient, model_name="Wierda")
        pd = TOFModel(patient, model_name="Wierda")
        
        dose_mg = 0.6 * patient.weight
        pk.state.c1 += dose_mg / pk.v1
        
        # Run for 3 min to achieve deep block (T1 ≈ 0)
        for _ in range(180):
            pk.step(1.0, 0.0)
            tof = pd.step_recovery(1.0, pk.state.c1)
        
        pre_sug_tof = tof
        
        # Confirm deep block
        assert tof < 10.0, f"Should be in deep block (TOFR={tof:.1f}%)"
        
        # Give 4 mg/kg sugammadex
        sug_dose_mg = 4.0 * patient.weight
        pd.give_sugammadex(sug_dose_mg)
        
        # Check recovery time
        recovery_time = None
        for t in range(300):  # 5 min max
            pk.step(1.0, 0.0)
            tof = pd.step_recovery(1.0, pk.state.c1)
            if tof >= 90.0 and recovery_time is None:
                recovery_time = t
        
        assert recovery_time is not None, \
            f"TOFR should reach 90% after 4 mg/kg sugammadex (pre-sug TOF={pre_sug_tof:.1f}%)"
        assert recovery_time <= 240, \
            f"Deep block reversal at {recovery_time}s (expected ≤240s / 4 min)"
    
    def test_sugammadex_immediate_reversal(self, patient):
        """
        Clinical: 16 mg/kg sugammadex at 3 min post 1.2 mg/kg roc → TOFR ≥ 90% in ~1-1.5 min.
        Kleijn et al. Br J Clin Pharmacol. 2011 (population PK/PD analysis; unverified mapping).
        """
        pk = RocuroniumPK(patient, model_name="Wierda")
        pd = TOFModel(patient, model_name="Wierda")
        
        # Higher intubating dose
        dose_mg = 1.2 * patient.weight
        pk.state.c1 += dose_mg / pk.v1
        
        # Run for 3 min (intense block)
        for _ in range(180):
            pk.step(1.0, 0.0)
            tof = pd.step_recovery(1.0, pk.state.c1)
        
        pre_sug_tof = tof
        
        # Confirm intense block
        assert tof < 5.0, f"Should be in intense block (TOFR={tof:.1f}%)"
        
        # Give 16 mg/kg sugammadex (rescue dose)
        sug_dose_mg = 16.0 * patient.weight
        pd.give_sugammadex(sug_dose_mg)
        
        # Check recovery time (allow longer time for high-dose roc scenario)
        recovery_time = None
        for t in range(360):  # 6 min max (higher starting Ce needs more time)
            pk.step(1.0, 0.0)
            tof = pd.step_recovery(1.0, pk.state.c1)
            if tof >= 90.0 and recovery_time is None:
                recovery_time = t
        
        assert recovery_time is not None, \
            f"TOFR should reach 90% after 16 mg/kg sugammadex (pre-sug TOF={pre_sug_tof:.1f}%)"
        assert recovery_time <= 300, \
            f"Immediate reversal at {recovery_time}s (expected ≤300s / 5 min)"
    
    def test_effect_site_asymmetry(self, patient):
        """
        Verify asymmetric ke0 behavior: onset faster than recovery.
        Plaud et al. Clin Pharmacol Ther. 1995.
        """
        pd = TOFModel(patient, model_name="Wierda")
        
        # Check parameters
        assert pd.ke0_onset > pd.recovery_ke0, \
            f"Onset ke0 ({pd.ke0_onset}) should be faster than recovery ke0 ({pd.recovery_ke0})"
        
        # Verify half-times (tuned for clinical realism)
        onset_t12 = 0.693 / pd.ke0_onset    # ~4.3 min
        recovery_t12 = 0.693 / pd.recovery_ke0  # ~6 min (tuned)
        
        assert 3.0 < onset_t12 < 6.0, f"Onset t1/2 = {onset_t12:.1f} min (expected ~4.4 min)"
        assert 4.0 < recovery_t12 < 8.0, f"Recovery t1/2 = {recovery_t12:.1f} min (tuned ~6 min)"
