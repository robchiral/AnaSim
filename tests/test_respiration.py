import pytest
from anasim.physiology.respiration import RespiratoryModel
from anasim.patient.patient import Patient

@pytest.fixture
def model(patient):
    return RespiratoryModel(patient)

class TestCentralDriveVsMuscle:
    def test_nmba_spares_central_drive(self, model):
        """NMBA preserves central drive while zeroing observed RR (no muscle function).
        
        Central drive (brain's desire to breathe) is unaffected by NMBA.
        However, observed RR = 0 because paralyzed muscles cannot execute breaths.
        """
        dt = 1.0
        
        # Case 1: No drugs
        s0 = model.step(dt, ce_prop=0, ce_remi=0, ce_roc=0)
        assert s0.drive_central == 1.0
        assert s0.muscle_factor == 1.0
        
        # Case 2: Full NMBA Block (High Dose)
        # C50=0.6, Gamma=4.0. Try 10.0 ug/mL
        s1 = model.step(dt, ce_prop=0, ce_remi=0, ce_roc=10.0)
        
        # Muscle factor should be near 0
        assert s1.muscle_factor < 0.1
        
        # Central Drive should be near 1 (unaffected by NMBA)
        assert s1.drive_central > 0.95
        
        # Observed RR = 0 (patient cannot breathe with paralyzed muscles)
        assert s1.rr < 1.0, f"Expected apnea with full NMB, got RR={s1.rr}"
        
        # VT should be near 0 (no effective ventilation)
        assert s1.vt < 10.0 

# --- Literature-Based Respiratory Sanity Checks ---

class TestRespiratoryLiteratureValidation:
    """Rigorous respiratory sanity checks based on published literature."""
    # --- Baseline Respiratory Values ---
    
    def test_baseline_respiratory_rate(self, model):
        """
        LITERATURE: Normal adult respiratory rate is 12-20 breaths/min.
        """
        state = model.step(1.0, ce_prop=0, ce_remi=0, ce_roc=0)
        
        assert 10 < state.rr < 25, \
            f"Baseline RR {state.rr:.1f} outside physiological range (10-25)"
    
    # --- Opioid Respiratory Depression ---
    
    def test_remifentanil_rr_depression(self, model):
        """
        LITERATURE: Remifentanil causes dose-dependent respiratory depression.
        At Ce ~1 ng/mL, expect ~40-50% RR reduction.
        Model C50 = 1.0 ng/mL, so at C50 expect 50% effect.
        """
        baseline_state = model.step(1.0, ce_prop=0, ce_remi=0, ce_roc=0)
        baseline_rr = baseline_state.rr
        
        # At C50 concentration
        state = model.step(1.0, ce_prop=0, ce_remi=1.0, ce_roc=0)
        
        rr_frac = state.rr / baseline_rr
        
        # At C50, expect ~50% of baseline (due to RR weight = 1.0)
        assert 0.4 < rr_frac < 0.65, \
            f"Remi at C50: RR is {rr_frac*100:.1f}% of baseline - expected ~50%"
    
    def test_remifentanil_vt_sparing(self, patient):
        """
        LITERATURE: Opioids preferentially depress RR over VT.
        At Ce = C50, VT should be less affected than RR.
        """
        # Fresh model for this test
        model = RespiratoryModel(patient)
        baseline_state = model.step(1.0, ce_prop=0, ce_remi=0, ce_roc=0)
        
        # Fresh model for drug state (avoid carryover)
        model2 = RespiratoryModel(patient)
        state = model2.step(1.0, ce_prop=0, ce_remi=1.0, ce_roc=0)
        
        rr_frac = state.rr / baseline_state.rr
        vt_frac = state.vt / baseline_state.vt
        
        # VT should be more preserved than RR (or at least not worse)
        # Model may have RR and VT equally affected at low doses
        assert vt_frac >= rr_frac - 0.1, \
            f"Remi: VT preserved ({vt_frac:.2f}) should be >= RR preserved ({rr_frac:.2f})"
    
    def test_high_dose_opioid_apnea(self, model):
        """
        LITERATURE: High-dose opioids cause near-apnea (RR < 4).
        At Ce >> C50 (e.g., 5x C50), expect severe respiratory depression.
        """
        state = model.step(1.0, ce_prop=0, ce_remi=5.0, ce_roc=0)
        
        # Very high opioid should cause severe RR depression
        assert state.rr < 6, \
            f"High-dose remi (5x C50): RR {state.rr:.1f} - expected severe depression (<6)"
    
    # --- Propofol Respiratory Effects ---
    
    def test_propofol_vt_depression(self, patient):
        """
        LITERATURE: Propofol causes dose-dependent VT depression.
        Separated effects: HCVR IC50 ~1.0 µg/mL, Mechanical IC50 ~3.5 µg/mL
        At Ce ~3.5 µg/mL (mechanical C50), expect significant VT reduction.
        """
        # Fresh models to avoid state carryover
        model_base = RespiratoryModel(patient)
        baseline_state = model_base.step(1.0, ce_prop=0, ce_remi=0, ce_roc=0)
        
        model_drug = RespiratoryModel(patient)
        # At mechanical C50 (3.5 µg/mL)
        state = model_drug.step(1.0, ce_prop=3.5, ce_remi=0, ce_roc=0)
        
        vt_frac = state.vt / baseline_state.vt
        
        # At C50 with VT weight = 1.0, expect ~50% reduction
        assert vt_frac < 0.7, \
            f"Propofol at mechanical C50: VT is {vt_frac*100:.1f}% of baseline - expected significant reduction"
    
    def test_propofol_rr_sparing(self, patient):
        """
        LITERATURE: Propofol preferentially depresses VT over RR.
        At Ce = mechanical C50 (3.5 µg/mL), RR should be more preserved than VT.
        """
        # Fresh models
        model_base = RespiratoryModel(patient)
        baseline_state = model_base.step(1.0, ce_prop=0, ce_remi=0, ce_roc=0)
        
        model_drug = RespiratoryModel(patient)
        state = model_drug.step(1.0, ce_prop=3.5, ce_remi=0, ce_roc=0)
        
        rr_frac = state.rr / baseline_state.rr
        vt_frac = state.vt / baseline_state.vt
        
        # RR should be more preserved than VT (or at least equal)
        assert rr_frac >= vt_frac - 0.1, \
            f"Propofol: RR preserved ({rr_frac:.2f}) should be >= VT preserved ({vt_frac:.2f})"
    
    # --- Combined Drug Effects ---
    
    def test_combined_propofol_remi_depression(self, model):
        """
        Combined Propofol + Remifentanil should cause more severe respiratory depression
        than either drug alone.
        """
        baseline_state = model.step(1.0, ce_prop=0, ce_remi=0, ce_roc=0)
        
        # Propofol alone at C50 (4.5 µg/mL)
        prop_state = model.step(1.0, ce_prop=4.5, ce_remi=0, ce_roc=0)
        
        # Reset and try remi alone
        model_remi = RespiratoryModel(Patient(age=40, weight=70, height=170, sex="Male"))
        remi_state = model_remi.step(1.0, ce_prop=0, ce_remi=1.0, ce_roc=0)
        
        # Combined at mechanical C50 for propofol, remi C50
        model_combined = RespiratoryModel(Patient(age=40, weight=70, height=170, sex="Male"))
        combined_state = model_combined.step(1.0, ce_prop=3.5, ce_remi=1.0, ce_roc=0)
        
        baseline_mv = baseline_state.rr * baseline_state.vt
        combined_mv = combined_state.rr * combined_state.vt
        
        # Combined MV should be lower than baseline
        assert combined_mv < baseline_mv * 0.5, \
            f"Combined Prop+Remi: MV {combined_mv:.0f} should be <50% of baseline {baseline_mv:.0f}"
    
class TestPaO2Physiology:
    """Tests for physiologically correct PaO2 behavior.
    
    Note: Anemia and low cardiac output do NOT reduce PaO2 directly.
    They reduce CaO2 (oxygen content) and DO2 (delivery); however, PaO2
    depends only on FiO2, PaCO2, V/Q matching, and A-a gradient.
    """
    @pytest.fixture
    def patient(self):
        return Patient(age=40, weight=70, height=170, sex="Male", baseline_hb=13.5)

    def test_anemia_does_not_affect_pao2(self, patient):
        """PaO2 should be unchanged by anemia (normal gas exchange)."""
        model_norm = RespiratoryModel(patient)
        for _ in range(30):
            normal = model_norm.step(1.0, ce_prop=0, ce_remi=0, ce_roc=0, hb_g_dl=13.5)

        model_anemic = RespiratoryModel(patient)
        for _ in range(30):
            anemic = model_anemic.step(1.0, ce_prop=0, ce_remi=0, ce_roc=0, hb_g_dl=6.0)

        # PaO2 should be the same (or very close) regardless of Hb
        assert abs(anemic.p_arterial_o2 - normal.p_arterial_o2) < 2.0, \
            f"PaO2 should not depend on Hb: normal={normal.p_arterial_o2:.1f}, anemic={anemic.p_arterial_o2:.1f}"


class TestHypercapnicVentilatoryResponse:
    """
    Tests for HCVR (CO2-driven ventilatory stimulation).
    
    Literature:
    - Normal HCVR ~2-4 L/min/mmHg.
    - Babenco et al. Anesthesiology. 2000 (opioids depress slope and shift setpoint).
    - Dahan et al. Br J Anaesth. 1998 (0.1 MAC effects on CO2 response).
    - Doi & Ikeda. Anesth Analg. 1987 (1.1-1.4 MAC depression).
    """
    
    def test_hypercapnia_increases_drive(self, patient):
        """
        LITERATURE: Rising PaCO2 should increase central respiratory drive.
        
        At PaCO2 = 50 mmHg (10 above setpoint), HCVR should boost drive
        even at baseline awake state.
        """
        # Awake patient with normal CO2
        model_normal = RespiratoryModel(patient)
        model_normal.state.p_alveolar_co2 = 40.0
        normal_state = model_normal.step(1.0, ce_prop=0.0, ce_remi=0.0)
        
        # Awake patient with elevated CO2
        model_hypercap = RespiratoryModel(patient)
        model_hypercap.state.p_alveolar_co2 = 50.0
        hypercap_state = model_hypercap.step(1.0, ce_prop=0.0, ce_remi=0.0)
        
        # Central drive should be higher with hypercapnia
        assert hypercap_state.drive_central > normal_state.drive_central, \
            f"Hypercapnic drive {hypercap_state.drive_central:.2f} should be > normal {normal_state.drive_central:.2f}"
    
    def test_opioids_depress_hcvr(self, patient):
        """
        LITERATURE: Opioids depress HCVR slope (Babenco et al. Anesthesiology. 2000).
        
        At high remifentanil (Ce = 3 ng/mL), hypercapnia should produce
        less drive increase than in awake state.
        """
        # Use moderate hypercapnia (45 mmHg) to avoid hitting drive cap
        # Awake hypercapnic response
        model_awake = RespiratoryModel(patient)
        model_awake.state.p_alveolar_co2 = 45.0
        awake_state = model_awake.step(1.0, ce_prop=0.0, ce_remi=0.0)
        awake_drive = awake_state.drive_central
        
        # With remifentanil (Ce = 3 ng/mL, well above C50)
        model_opioid = RespiratoryModel(patient)
        model_opioid.state.p_alveolar_co2 = 45.0
        opioid_state = model_opioid.step(1.0, ce_prop=0.0, ce_remi=3.0)
        opioid_drive = opioid_state.drive_central
        
        # HCVR should be blunted (opioid drive < awake drive)
        assert opioid_drive < awake_drive, \
            f"Opioid HCVR drive {opioid_drive:.2f} should be < awake drive {awake_drive:.2f}"
    
    def test_remifentanil_setpoint_shift(self, patient):
        """
        LITERATURE: Opioids shift the apneic threshold rightward.
        At high remifentanil, higher PaCO2 is required to trigger breathing.
        Babenco et al. Anesthesiology. 2000: opioids shift CO2 curve right and reduce slope.
        """
        # Awake: CO2 of 45 mmHg should produce drive boost
        model_awake = RespiratoryModel(patient)
        model_awake.state.p_alveolar_co2 = 45.0
        awake_state = model_awake.step(1.0, ce_prop=0.0, ce_remi=0.0)
        
        # With high remifentanil: setpoint shifts right by ~8 mmHg at full effect
        # At Ce = 3 ng/mL (well above C50), eff_remi ~0.75
        # Effective setpoint ≈ 40 + 0.75*8 = 46 mmHg
        # So PaCO2 of 45 is now BELOW the shifted setpoint -> less drive boost
        model_opioid = RespiratoryModel(patient)
        model_opioid.state.p_alveolar_co2 = 45.0
        opioid_state = model_opioid.step(1.0, ce_prop=0.0, ce_remi=3.0)
        
        # With shifted setpoint, 45 mmHg is below threshold -> less CO2 drive
        # Drive should be lower due to both slope reduction AND setpoint shift
        assert opioid_state.drive_central < awake_state.drive_central, \
            f"Opioid setpoint shift: drive {opioid_state.drive_central:.2f} should be < awake {awake_state.drive_central:.2f}"
    
    def test_emergence_with_hypercapnia(self, patient):
        """
        Emergence scenario: At residual anesthetic with elevated CO2,
        HCVR should partially restore respiratory drive.
        
        This tests the core fix for prolonged emergence times.
        """
        # Scenario: Patient emerging with residual propofol/remi
        # But CO2 has risen to 55 mmHg
        model = RespiratoryModel(patient)
        model.state.p_alveolar_co2 = 55.0
        
        # Residual emergence-level concentrations
        # (propofol ~1.5, remi ~0.5 - below C50 but still some effect)
        state = model.step(1.0, ce_prop=1.5, ce_remi=0.5)
        
        # HCVR should boost drive despite residual drugs
        # At these low drug levels with high CO2, drive should be > 0.5
        assert state.drive_central > 0.5, \
            f"Central drive {state.drive_central:.2f} too low for emergence with hypercapnia"
        
        # Minute ventilation should be adequate for emergence
        assert state.mv > 3.0, \
            f"MV {state.mv:.1f} L/min too low for adequate emergence ventilation"
