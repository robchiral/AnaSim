import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import root_scalar
from patient.patient import Patient
from core.constants import (
    HR_MIN, HR_MAX, MAP_MAX, SBP_MAX, DBP_MAX, TPR_MIN, BLOOD_VOLUME_MIN, TEMP_TPR_COEFFICIENT
)
from core.utils import hill_function, clamp, clamp01


@dataclass
class HemoState:
    """
    Hemodynamic state snapshot.
    
    Defaults represent typical adult clinical baselines. These are overridden
    at runtime by model calculations based on patient demographics.
    """
    map: float = 80.0   # Mean Arterial Pressure (mmHg)
    hr: float = 75.0    # Heart Rate (bpm)
    sv: float = 70.0    # Stroke Volume (mL)
    svr: float = 16.0   # SVR (mmHg*min/L) ~ Wood Units
    co: float = 5.25    # Cardiac Output (L/min)
    sbp: float = 115.0  # Systolic Blood Pressure (mmHg)
    dbp: float = 70.0   # Diastolic Blood Pressure (mmHg)


@dataclass
class HemoStateExtended(HemoState):
    """
    Extended hemodynamic state with internal model variables.
    
    Includes internal state variables from the ODE-based model.
    """
    # Internal state variables
    tpr: float = 0.016      # Total Peripheral Resistance (mmHg*min/mL)
    sv_star: float = 82.2   # Stroke Volume state variable
    hr_star: float = 56.0   # Heart Rate state variable  
    tde_sv: float = 7.4     # Time-dependent error for SV
    tde_hr: float = 6.7     # Time-dependent error for HR
    
    # Volatile effect site concentration (MAC units)
    ce_sevo: float = 0.0


class HemodynamicModel:
    """
    Integrated Hemodynamic Model for Anesthesia Simulation.
    
    Core model from Su et al. (2023) BJA "Mechanistic-based interaction model..."
    Extended with ventilator-circulation coupling, blood volume tracking,
    chemoreflexes, and additional vasoactive drug support.
    
    Literature References:
    ----------------------
    Core Model:
        Su H et al. BJA 2023; Mechanistic-based hemodynamic model during
        propofol-remifentanil anesthesia. Parameters: EC50, Emax, FB, γ values.
    
    Arterial Compliance:
        Nichols WW, O'Rourke MF. McDonald's Blood Flow in Arteries (6th ed).
        Age-dependent compliance: C(age) = 1.5 × (1 - 0.008×(age-40)) mL/mmHg
    
    Vasopressors:
        - Norepinephrine: Beloeil H et al. Br J Anaesth 2005; PK/PD in septic shock/trauma
        - Epinephrine: Clutter WE et al. J Clin Invest 1980; FDA NDA 205029 Review
        - Phenylephrine: Anderson BJ et al. Paediatr Anaesth 2017; PK/PD concentration-response
    
    Volatile Agents:
        Mapleson WW (1973/1996); Physiological pharmacokinetics
        Ebert TJ et al. Anesthesiology 1995; Cardiovascular responses to sevoflurane
    
    Blood Volume:
        Guyton AC. Textbook of Medical Physiology (MCFP, stressed volume)
        
    Units:
        - MAP, SBP, DBP: mmHg
        - HR: bpm
        - SV: mL
        - CO: L/min
        - SVR: Wood Units (mmHg×min/L)
        - TPR: mmHg×min/mL (internal)
    """
    def __init__(self, patient: Patient):
        self.patient = patient
        
        # =====================================================================
        # Core Model Parameters (Su et al. 2023 BJA)
        # =====================================================================
        
        # Speed of hemodynamic autoregulation (tuned for simulator responsiveness)
        # Note: Su et al. (2023) use 0.072 min^-1.
        self.kout = 0.072  # min^-1
        
        # Hemoglobin / hematocrit baselines
        self.baseline_hb = getattr(patient, 'baseline_hb', 13.5)
        self.baseline_hct = getattr(patient, 'baseline_hct', 0.42)
        
        # Baseline HR from patient or default
        self.base_hr = patient.baseline_hr if patient.baseline_hr > 0 else 56.0
        
        # Baseline SV derived from cardiac index and BSA
        ci_0 = 2.5 if patient.age > 70 else 3.0
        co_0 = ci_0 * patient.bsa if patient.bsa > 0 else ci_0 * 1.9
        self.base_sv = (co_0 * 1000.0) / self.base_hr if self.base_hr > 0 else 82.2
        
        # Baseline TPR from MAP and flow
        flow_ml_min = self.base_hr * self.base_sv
        self.base_tpr = patient.baseline_map / flow_ml_min if flow_ml_min > 0 else 0.016
        self.base_co_l_min = flow_ml_min / 1000.0
        self.baseline_caO2 = self.calc_oxygen_content(self.baseline_hb, 0.98, 95.0)
        self.baseline_do2 = self.baseline_caO2 * self.base_co_l_min * 10.0
        
        # Arterial Compliance (for realistic pulse pressure calculation)
        # 
        # Literature basis:
        # - Compliance decreases ~50% from age 20 to 80 (Nichols WW et al. McDonald's Blood Flow in Arteries 2011)
        # - Young adult: ~1.5-2.0 mL/mmHg, Elderly: ~0.8-1.0 mL/mmHg
        # - Formula: C(age) = C_ref × (1 - 0.01 × (age - 40))
        #   where C_ref = 1.5 mL/mmHg at age 40
        # Linear formula with a clamp to keep values in a physiologic range.
        # 
        # Effect on pulse pressure:
        # - PP = SV / C (approximately)
        # - Lower C → wider PP (elderly have higher SBP with preserved DBP)
        #
        # Reference: Nichols WW, O'Rourke MF. McDonald's Blood Flow in Arteries.
        age = patient.age
        self.arterial_compliance_ref = 1.5  # mL/mmHg at age 40
        age_factor = 1.0 - 0.008 * (age - 40)  # 0.8% decrease per year from 40
        age_factor = clamp(age_factor, 0.5, 1.2)  # Clamp to realistic range
        self.arterial_compliance = self.arterial_compliance_ref * age_factor
        
        # Feedback parameters
        # Su et al. define FB = 0.66 in Table 2.
        # However, we implement self.fb = -0.66 to ensure stable negative feedback 
        # given the structure of our ODE implementation (Production ~ RMAP^FB).
        # This sign flip is a deviation for numerical stability/correctness in this solver.
        self.fb = -0.66
        self.hr_sv_coupling = 0.312
        self.k_drift = 0.067      # Drift rate (min^-1)        
        # External Modifiers (Persistence for property access)
        self.delta_tpr_vasopressors = 0.0
        self.dist_svr = 0.0
        self.dist_hr = 0.0
        self.dist_sv = 0.0
        self.chemoreflex_active = True
        
        # Instantaneous effects with rate limiting (exponential smoothing)
        # Target values (calculated each step)
        self.current_epi_hr = 0.0
        self.current_chemo_hr = 0.0
        # Smoothed values (used in _calc_hr) with time constants
        self.smoothed_epi_hr = 0.0
        self.smoothed_chemo_hr = 0.0
        # Time constants: Physiologically fast but not instant
        # tau_hr = 5s: Peak effect within ~15s (3*tau), matches clinical phenylephrine bolus
        self.tau_hr_fast = 5.0  # seconds (faster for vasopressor bolus realism)
        
        # Vasopressor SV factor (combined inotropy)
        self.vasopressor_sv_factor = 1.0
        
        # Initial Conditions
        self.sv_star = self.base_sv
        self.hr_star = self.base_hr
        self.tpr = self.base_tpr
        
        # Initial drifts (start at zero for stable baseline)
        self.tde_sv = 0.0
        self.tde_hr = 0.0
        
        # State cache (for efficiency - avoid recalculating on every property access)
        # Invalidated at start of step() and when state is set
        self._cached_state: Optional[HemoStateExtended] = None
        
        # Derived Kin for equilibrium at RMAP=1
        self.kin_tpr = self.kout * self.base_tpr
        self.kin_sv = self.kout * self.base_sv
        self.kin_hr = self.kout * self.base_hr
        
        # =====================================================================
        # Drug Effect Parameters (Su et al. 2023 + literature extensions)
        # =====================================================================
        
        # Propofol hemodynamic effects
        self.ec50_prop_tpr = 3.21      # ug/mL
        self.emax_prop_tpr = -0.78
        self.gamma_prop = 1.83
        self.ec50_prop_sv = 0.44
        self.emax_prop_sv_typ = -0.15
        self.age_emax_sv = 0.033
        
        # Remifentanil hemodynamic effects
        self.ec50_remi_tpr = 4.59      # ng/mL
        self.emax_remi_tpr = -1.0
        self.gamma_remi_tpr = 1.0
        self.sl_remi_hr = 0.033
        self.sl_remi_sv = 0.058
        
        # Propofol-Remifentanil interaction
        self.int_tpr = 1.00
        self.int_hr = -0.12
        self.ec50_int_hr = 0.20        # ug/mL (Propofol)
        self.int_sv = -0.21
        
        # Norepinephrine PD (Beloeil H et al. Br J Anaesth 2005)
        self.nore_c50 = 7.04       # ng/mL (EC50 for MAP effect)
        self.nore_gamma = 1.8
        self.nore_emax_map = 98.7  # mmHg max MAP increase (alpha-1)
        self.nore_emax_hr = 10.0   # bpm max (beta-1, usually masked by reflex, heuristic)
        self.nore_emax_sv = 0.15   # +15% SV max (beta-1 inotropy, heuristic)
        
        # --- 4.5 Volatile Parameters (Mechanism-based extension) ---
        # Effect site kinetics (ke0)
        self.ke0_sevo = 0.25 # min^-1

        
        # PD Parameters (Emax, EC50, Gamma)
        # These are reasonable approximations chosen to reproduce approximate MAP/SVR changes
        # seen at 1-1.5 MAC in Ebert 1995 and related studies. Not a direct model fit.
        # Sevo
        self.sevo_emax_tpr = -0.45 # Moderate vasodilation
        self.sevo_ec50_tpr = 1.0   # MAC
        self.sevo_gamma_tpr = 1.5
        
        self.sevo_emax_sv = -0.15 # Minimal depression
        self.sevo_ec50_sv = 1.0
        

        
        # Remi interaction strength (EC50 shift)
        self.vol_remi_shift_max = 0.6
        self.vol_remi_ec50 = 2.0 # ng/mL interaction potency
        
        # State init
        self.ce_sevo = 0.0
        
        # =========================================================================
        # Blood Volume and Preload Tracking
        # =========================================================================
        
        # --- Blood Volume Tracking (MCFP / Stressed Volume) ---
        if hasattr(patient, 'estimate_blood_volume'):
            self.blood_volume = patient.estimate_blood_volume()
        else:
            self.blood_volume = 5000.0  # Default 5L
        self.blood_volume_0 = self.blood_volume  # Save baseline for hemorrhage calc
        self.hb_mass = self.baseline_hb * (self.blood_volume / 100.0)  # grams
        self.hb_conc = self.baseline_hb
        
        # Unstressed volume: ~70% of total (standard physiology)
        # This leaves ~30% as stressed volume (~1400mL for 5L blood volume)
        # Hemorrhage depletes stressed volume first, reducing preload/MCFP
        self.unstressed_volume = self.blood_volume * 0.70
        
        # Venous compliance (mL/mmHg)
        self.cv = 100.0
        
        # Baseline MCFP (Mean Circulatory Filling Pressure)
        # Uses Guytonian concept: MCFP = Stressed_Vol / Compliance
        # Values are heuristics to match standard physiology (~10-15 mmHg)
        stressed_vol_0 = max(0.0, self.blood_volume - self.unstressed_volume)
        self.mcfp_0 = max(1.0, stressed_vol_0 / self.cv)  # ~10-15 mmHg normally
        
        # Volume clearance (mL/min) - represents urine + third-spacing
        self.vol_clearance = 1.0
        
        # --- Intrathoracic Pressure (Pit) → Preload Coupling ---
        self.pit_0 = -2.0  # Baseline intrathoracic pressure (mmHg)
        self.alpha_peep = 0.04  # Sensitivity: 1/mmHg (how much PEEP reduces preload)
        self.f_preload_pit = 1.0 # Current PEEP preload factor (stored for state calc)
        
        # --- CO2/O2 Chemoreflex ---
        self.paco2_set = 40.0  # Normal PaCO2 setpoint (mmHg)
        self.pao2_set = 85.0   # Normal PaO2 setpoint (mmHg)
        
        # Chemoreflex gains (Tuned Heuristics - behave physiologically but no direct citation)
        self.g_hr_co2 = 15.0   # HR increase per unit normalized CO2 error
        self.g_hr_o2 = 15.0    # HR increase per unit normalized O2 error
        self.k_tpr_co2 = 0.3   # TPR production boost per unit CO2 error
        
        # --- Hemorrhage Response (Staged Sympathetic Compensation) ---
        # Class-based heuristic multipliers designed to mimic ATLS class I-IV responses.
        # No quantitative human data support the exact coefficients.
        # Initialized here, computed dynamically in step()
        self.hemorrhage_hr_mult = 1.0
        self.hemorrhage_tpr_mult = 1.0
        
        # --- Cumulative Fluid Tracking (for scenario requirements) ---
        self.cumulative_fluid_given = 0.0
        
        # =========================================================================
        # Vasopressor PD Models
        # =========================================================================
        
        # --- Epinephrine PD (Primary Anchors: Clutter WE et al. J Clin Invest 1980, FDA NDA 205029) ---
        # Thresholds from Clutter: HR 50-100 pg/mL, SBP 75-125 pg/mL, DBP 150-200 pg/mL
        # FDA Review: 0.06 µg/kg/min (C~1.0 ng/mL) -> HR +7 bpm, SBP +11 mmHg, DBP -14 mmHg.
        #   Result: Net vasodilation (Beta-2) dominates at 1.0 ng/mL.
        # FDA Review: 0.2 µg/kg/min (C~3-3.5 ng/mL) -> HR +30 bpm, SBP +40 mmHg.
        #   Result: Stronger pressor effect, approaching Alpha dominance.
        #
        # Parameters below are calibrated heuristics to approximate the above data points.
        # They are not a direct fit to a single published PD model.
        
        # HR (β1 chronotropy)
        # Calibrated: EC50 ~1.5, Gamma ~3.0 fits (1.0ng->~7bpm, 3.0ng->~27bpm)
        self.epi_c50 = 1.5          # ng/mL (EC50 for HR)
        self.epi_gamma = 2.5        # Steep dose-response
        self.epi_emax_hr = 40.0     # bpm max (FDA max >30 at 0.2ug/kg/min)
        
        # SV (β1 inotropy)
        self.epi_emax_sv = 0.25     # 25% max SV increase
        
        # SVR (β2 → α transition)
        # At ~1 ng/mL, DBP drops (-14 mmHg), so SVR must be reduced.
        # Threshold for Alpha dominance must be > 1.0 ng/mL.
        self.epi_emax_svr_low = -0.20   # -20% SVR at low C (Beta-2 dominant)
        self.epi_emax_svr_high = 0.40   # +40% SVR at high C (Alpha dominant)
        self.epi_threshold = 2.0        # ng/mL (2000 pg/mL): Alpha effects begin to dominate
        
        # --- Phenylephrine PD (Reference: Anderson BJ et al. Paediatr Anaesth 2017) ---
        # Anderson 2017: Pediatric EC50 ~10.3 ng/mL; Adult predicted Emax ~50 mmHg MAP.
        #
        # Tuned Approximations:
        # - EC50 raised to 15 ng/mL to avoid early saturation in adults.
        # - Emax mapped to 60% SVR increase (assuming MAP ~ CO * SVR).
        # These are tuned heuristics, not direct literature fits.
        self.phenyl_c50 = 15.0       # ng/mL (shifted right to avoid early saturation)
        self.phenyl_gamma = 1.5
        self.phenyl_emax_svr = 0.60  # 60% SVR increase max (allows overdose effects)

    def set_nore_pd(self, c50: float, emax: float = 98.7, gamma: float = 1.8):
        """Configure Norepinephrine PD parameters based on selected PK model."""
        self.nore_c50 = c50
        self.nore_emax_map = emax
        self.nore_gamma = gamma
    
    @staticmethod
    def _calc_hill(ce: float, c50: float, gamma: float) -> float:
        """Wrapper to centralized hill_function for backward compatibility."""
        return hill_function(ce, c50, gamma)

    @staticmethod
    def _vol_hill(ce: float, emax: float, ec50_base: float, gamma: float, 
                  vol_ec50_mult: float = 1.0, use_shift: bool = True) -> float:
        """Hill function for volatile anesthetic effects with EC50 shift."""
        eff_ec50 = ec50_base * vol_ec50_mult if use_shift else ec50_base
        return emax * hill_function(ce, eff_ec50, gamma)
        
    def add_volume(self, amount_ml: float, hematocrit: float = 0.0):
        """
        Simulate fluid bolus or hemorrhage.
        
        Positive values: fluid administration (crystalloid, colloid, blood)
        Negative values: blood loss
        
        Updates blood_volume directly for MCFP-based preload calculation.
        Also applies transient SV boost via tde_sv (Frank-Starling).
        """
        self.blood_volume += amount_ml
        
        # Prevent nonsensical volumes
        self.blood_volume = max(500.0, self.blood_volume)  # Minimum ~500mL

        if amount_ml < 0:
            hb_loss = self.hb_conc * (-amount_ml) / 100.0
            self.hb_mass = max(0.0, self.hb_mass - hb_loss)
        elif amount_ml > 0 and hematocrit > 0:
            hb_gain = self.baseline_hb * hematocrit * amount_ml / 100.0
            self.hb_mass += hb_gain

        self._update_hb_conc()
        
        # Transient SV effect (Frank-Starling response to acute volume change)
        # Gain scaled to baseline SV for patient-specific response
        if amount_ml > 0:
            gain = 0.02 * (self.base_sv / 80.0)
            self.tde_sv += amount_ml * gain
            # Track cumulative fluid given (for scenario requirements)
            self.cumulative_fluid_given += amount_ml

    def _update_hb_conc(self):
        if self.blood_volume <= 0:
            self.hb_conc = 0.0
        else:
            self.hb_conc = self.hb_mass / (self.blood_volume / 100.0)

    def get_hematocrit(self) -> float:
        if self.baseline_hb <= 0:
            return self.baseline_hct
        ratio = self.hb_conc / self.baseline_hb
        return clamp(self.baseline_hct * ratio, 0.0, 0.7)

    @staticmethod
    def calc_oxygen_content(hb_g_dl: float, sao2_frac: float, pao2: float) -> float:
        """
        Oxygen content (mL O2/dL blood).
        """
        sao2_frac = clamp01(sao2_frac)
        return hb_g_dl * 1.34 * sao2_frac + 0.003 * pao2

    def compute_do2_ratio(self, sao2_frac: float, pao2: float, co_l_min: float) -> float:
        caO2 = self.calc_oxygen_content(self.hb_conc, sao2_frac, pao2)
        do2 = caO2 * max(0.0, co_l_min) * 10.0
        if self.baseline_do2 <= 0:
            return 1.0
        ratio = do2 / self.baseline_do2
        return clamp(ratio, 0.0, 2.0)
    
    # =========================================================================
    # New Helper Methods
    # =========================================================================
    
    def _frank_starling(self, preload_factor: float, inotropy: float = 1.0) -> float:
        """
        Frank-Starling curve: SV increases with preload but plateaus.
        
        Args:
            preload_factor: Normalized preload (1.0 = baseline)
            inotropy: Contractility modifier (1.0 = normal, >1 = positive inotrope)
        
        Returns:
            SV multiplier factor (relative to baseline)
            
        Based on: SV = SVmax × (1 - exp(-k × LVEDV))
        Approximated as: SV_factor = inotropy × (1 - exp(-2.0 × preload_factor))
        
        At baseline (pf=1.0), factor ≈ 0.86. Normalized so baseline = 1.0.
        
        Clinical floor: Even in severe hypovolemia, stressed volume depletion,
        the heart maintains ~25-30% of baseline SV due to:
        - Residual venous return from unstressed volume mobilization
        - Catecholamine-driven contractility boost (sympathetic activation)
        - Incomplete emptying at low preload still ejects some volume
        """
        # Clamp preload_factor to avoid extreme values
        pf = clamp(preload_factor, 0.01, 2.5)
        
        # Raw Frank-Starling curve
        raw_factor = 1.0 - np.exp(-2.0 * pf)
        
        # Normalize so that pf=1.0 gives factor=1.0
        baseline_raw = 1.0 - np.exp(-2.0)  # ≈ 0.865
        normalized = raw_factor / baseline_raw
        
        # Apply inotropy
        result = inotropy * normalized
        
        # Clinical floor: minimum SV = 5% of baseline even in severe hypovolemia
        # This accounts for compensatory mechanisms not fully modeled
        return max(0.05, result)
    
    def _calc_hemorrhage_response(self) -> tuple:
        """
        Calculate hemorrhage class-based sympathetic compensation.
        
        Returns:
            (hr_multiplier, tpr_multiplier): Factors to apply to baseline HR/TPR
            
        Based on ATLS hemorrhage classification:
        - Class I (0-15%): Minimal response (HR <100, normal BP)
        - Class II (15-30%): Moderate tachycardia (HR 100-120), mild BP decrease
        - Class III (30-40%): Peak compensation (HR 120-140), decreased BP
        - Class IV (>40%): Decompensation (HR >140 then fails, severe hypotension)
        
        Clinical targets (awake patient, baseline HR ~70):
        - Class III: HR 120-140 bpm (1.7-2.0× baseline)
        - MAP ~60-70 mmHg (with compensation)
        
        Note: These multipliers work IN ADDITION to the baroreflex (rmap^fb).
        Values are tuned to avoid double-counting. The baroreflex provides ~50%
        compensation when MAP drops to 50%, so hemorrhage multipliers are modest:
        - Total HR boost = hemorrhage_mult × baroreflex × baseline_HR
        
        Preload/MCFP effect is handled separately via f_frank_starling.
        """
        if self.blood_volume_0 <= 0:
            return 1.0, 1.0
            
        vol_deficit = (self.blood_volume_0 - self.blood_volume) / self.blood_volume_0
        vol_deficit = max(0.0, vol_deficit)  # Only consider loss, not gain
        
        if vol_deficit < 0.15:
            # Class I: Minimal compensation
            # Target: HR <100 (up to 1.4× at 15% loss, combined with baroreflex)
            hr_mult = 1.0 + 0.10 * (vol_deficit / 0.15)
            tpr_mult = 1.0 + 0.10 * (vol_deficit / 0.15)
        elif vol_deficit < 0.30:
            # Class II: Moderate compensation
            # Target: HR 100-120 (combined effect)
            progress = (vol_deficit - 0.15) / 0.15
            hr_mult = 1.10 + 0.10 * progress  # Up to 20% direct HR boost
            tpr_mult = 1.10 + 0.20 * progress  # Up to 30% TPR increase
        elif vol_deficit < 0.40:
            # Class III: Peak compensation, SV dropping due to preload loss
            # Target: HR 120-140 (combined with baroreflex)
            # Direct multiplier to achieve peak sympathetic response
            progress = (vol_deficit - 0.30) / 0.10
            hr_mult = 1.25 + 0.15 * progress  # Up to 40% direct boost
            tpr_mult = 1.30 + 0.10 * progress  # TPR peaks at +40%
        else:
            # Class IV: Decompensation - sympathetic exhaustion
            # HR initially high then fails; TPR collapses
            progress = min(1.0, (vol_deficit - 0.40) / 0.20)
            hr_mult = max(0.7, 1.30 - 0.60 * progress)  # HR fails to 70% of boost
            tpr_mult = max(0.4, 1.40 - 1.00 * progress)  # TPR collapses
            
        return hr_mult, tpr_mult
    
    def _calc_epi_effects(self, ce_epi: float) -> tuple:
        """
        Calculate epinephrine effects on HR, SV, and SVR.
        
        Epinephrine has complex dose-dependent effects:
        - Low dose: Beta-2 vasodilation, Beta-1 chronotropy/inotropy
        - High dose: Alpha vasoconstriction dominates
        
        Returns:
            (delta_hr, sv_factor, svr_factor): Additive HR, multiplicative SV/SVR
        """
        if ce_epi <= 0:
            return 0.0, 1.0, 1.0
        
        # Basic Hill function for beta effects
        epi_hill = self._calc_hill(ce_epi, self.epi_c50, self.epi_gamma)
        
        # Chronotropy (HR increase)
        delta_hr = self.epi_emax_hr * epi_hill
        
        # Inotropy (SV increase)
        sv_factor = 1.0 + self.epi_emax_sv * epi_hill
        
        # SVR: Biphasic (low dose vasodilation, high dose vasoconstriction)
        if ce_epi < self.epi_threshold:
            # Beta-2 dominates: vasodilation
            svr_factor = 1.0 + self.epi_emax_svr_low * epi_hill
        else:
            # Transition to alpha-dominant
            # Blend the effects based on concentration
            low_fraction = self.epi_threshold / ce_epi
            excess = ce_epi - self.epi_threshold
            
            low_contrib = self.epi_emax_svr_low * self._calc_hill(self.epi_threshold, self.epi_c50, self.epi_gamma)
            
            high_hill = self._calc_hill(excess, self.epi_c50, self.epi_gamma)
            high_contrib = self.epi_emax_svr_high * high_hill
            
            svr_factor = 1.0 + low_contrib * low_fraction + high_contrib
        
        return delta_hr, sv_factor, max(0.5, svr_factor)
    
    def _calc_phenyl_effects(self, ce_phenyl: float) -> float:
        """
        Calculate phenylephrine SVR effect.
        
        Phenylephrine is a pure alpha-1 agonist with minimal direct cardiac effects.
        Secondary reflex bradycardia may occur but is handled by baroreflex.
        
        Returns:
            svr_factor: Multiplicative SVR factor
        """
        if ce_phenyl <= 0:
            return 1.0
        
        phenyl_hill = self._calc_hill(ce_phenyl, self.phenyl_c50, self.phenyl_gamma)
        
        return 1.0 + self.phenyl_emax_svr * phenyl_hill
    
    def _calc_nore_effects(self, ce_nore: float) -> tuple:
        """
        Calculate norepinephrine effects on HR, SV, and SVR.
        
        Norepinephrine is a potent alpha-1 agonist with moderate beta-1 effects.
        Unlike epinephrine, alpha-mediated vasoconstriction dominates at all doses.
        
        Clinical pharmacology:
        - Primary effect: Vasoconstriction (alpha-1) → increased SVR/MAP
        - Secondary: Modest chronotropy/inotropy (beta-1) → small HR/SV increase
        - Reflex bradycardia may offset direct HR increase (handled by baroreflex)
        
        # Literature basis:
        # - Beloeil H et al. BJA 2005: EC50 ~7 ng/mL for MAP effect
        # - Typical ICU range: 0.05-0.5 mcg/kg/min → Ce ~2-20 ng/mL
        
        Returns:
            (delta_hr, sv_factor, svr_factor): Additive HR, multiplicative SV/SVR
        """
        if ce_nore <= 0:
            return 0.0, 1.0, 1.0
        
        nore_hill = self._calc_hill(ce_nore, self.nore_c50, self.nore_gamma)
        
        # Beta-1 chronotropy (modest compared to epinephrine)
        delta_hr = self.nore_emax_hr * nore_hill
        
        # Beta-1 inotropy
        sv_factor = 1.0 + self.nore_emax_sv * nore_hill
        
        # Alpha-1 vasoconstriction (dominant effect)
        # Convert Emax_MAP to SVR factor: delta_MAP / base_MAP
        # At Emax (100 mmHg increase), SVR increases ~125% (from ~80 to ~180 mmHg)
        svr_factor = 1.0 + (self.nore_emax_map * nore_hill) / 80.0
        
        return delta_hr, sv_factor, svr_factor
    
    @property
    def state(self) -> HemoStateExtended:
        # Return cached state if available (avoids redundant calculation)
        if self._cached_state is not None:
            return self._cached_state
            
        # Construct state from internals
        current_hr = self._calc_hr() + self.dist_hr
        current_hr = max(HR_MIN, current_hr)
        # Upper bound: physiological maximum (safe limit)
        current_hr = min(HR_MAX, current_hr)
        
        # SV - with preload modulation from current blood volume/MCFP
        term = 1.0 - self.hr_sv_coupling * np.log(max(1.0, current_hr / self.base_hr))
        raw_sv = (self.sv_star + self.tde_sv) * term + self.dist_sv
        
        # Apply preload factor (MCFP-dependent via Frank-Starling)
        # This provides immediate SV reduction when blood volume drops
        stressed_vol = max(0.0, self.blood_volume - self.unstressed_volume)
        mcfp = stressed_vol / self.cv if self.cv > 0 else self.mcfp_0
        preload_ratio = mcfp / self.mcfp_0 if self.mcfp_0 > 0 else 1.0
        
        # Combine volume status with current PEEP/Pit effect
        total_preload = preload_ratio * self.f_preload_pit
        
        preload_sv_factor = self._frank_starling(total_preload)
        
        # Apply preload and vasopressor inotropy for immediate visibility
        current_sv = raw_sv * preload_sv_factor * self.vasopressor_sv_factor
        current_sv = max(1.0, current_sv)
        
        # CO
        co = current_hr * current_sv / 1000.0
        
        # MAP
        eff_tpr = self.tpr + self.delta_tpr_vasopressors + (self.dist_svr / 1000.0)
        # TPR floor: Minimum physiologically viable resistance
        # 0.001 is too low - use ~0.006 (equivalent to ~6 Wood Units at normal flow)
        eff_tpr = max(0.006, eff_tpr)
        
        map_val = current_hr * current_sv * eff_tpr
        
        # MAP floor: 30 mmHg is minimum compatible with life
        # MAP Cap: Avoid numerical explosion
        map_val = clamp(map_val, 5.0, 300.0)
        
        # SVR output
        svr_val = map_val / co if co > 0 else 0.0
        
        # SBP/DBP calculation with age-dependent arterial compliance
        #
        # Physiology:
        # - Pulse Pressure (PP) = SV / Arterial_Compliance
        # - Elderly (stiff arteries) → lower compliance → wider PP
        # - Hemorrhage (low SV) → narrower PP (weak pulse)
        #
        # Formula: PP = SV / C, distributed around MAP as:
        #   SBP = MAP + (2/3) × PP × distribution_factor
        #   DBP = MAP - (1/3) × PP × distribution_factor
        # where distribution_factor accounts for the typical 1/3:2/3 diastolic duration
        #
        pulse_pressure = current_sv / self.arterial_compliance
        
        # Clamp pulse pressure to physiological range
        # Normal: 30-60 mmHg, Wide: up to 100 in elderly/atherosclerotic
        pulse_pressure = clamp(pulse_pressure, 10.0, 120.0)
        
        # Distribute around MAP (diastole = 2/3 of cardiac cycle in duration,
        # but pressure time-weighted toward diastole, so MAP is closer to DBP)
        # Standard approximation: MAP ≈ DBP + (1/3)×PP
        # => DBP = MAP - (1/3)×PP, SBP = MAP + (2/3)×PP
        dbp = map_val - (1.0 / 3.0) * pulse_pressure
        sbp = map_val + (2.0 / 3.0) * pulse_pressure
        
        # Physiological bounds
        dbp = clamp(dbp, 5.0, 200.0)
        sbp = clamp(sbp, 10.0, 300.0)
        if sbp <= dbp:
            sbp = dbp + 15.0
        
        computed_state = HemoStateExtended(
            map=map_val,
            hr=current_hr,
            sv=current_sv,
            svr=svr_val,
            co=co,
            sbp=sbp,
            dbp=dbp,
            tpr=self.tpr,
            sv_star=self.sv_star,
            hr_star=self.hr_star,
            tde_sv=self.tde_sv,
            tde_hr=self.tde_hr,
            ce_sevo=self.ce_sevo,
        )
        
        # Cache the computed state
        self._cached_state = computed_state
        return computed_state
        
    @state.setter
    def state(self, new_state: HemoState):
        # Invalidate cache when state is set externally
        self._cached_state = None
        
        if isinstance(new_state, HemoStateExtended):
            self.tpr = new_state.tpr
            self.sv_star = new_state.sv_star
            self.hr_star = new_state.hr_star
            self.tde_sv = new_state.tde_sv
            self.tde_hr = new_state.tde_hr
            self.ce_sevo = new_state.ce_sevo

        else:
            # Fallback for vanilla HemoState (approximation)
            # Assuming TDEs are 0 (steady state forced)
            self.tde_sv = 0.0
            self.tde_hr = 0.0
            
            self.hr_star = new_state.hr
            
            # SV = SV_star * (1 - k * ln(HR/Base))
            term = 1.0 - self.hr_sv_coupling * np.log(max(1.0, self.hr_star / self.base_hr))
            if term < 0.1: term = 0.1
            self.sv_star = new_state.sv / term
            
            # TPR = MAP / (HR * SV)
            denom = new_state.hr * new_state.sv
            if denom > 0:
                self.tpr = new_state.map / denom
            else:
                self.tpr = self.base_tpr
                
    def _calc_hr(self):
        # Includes base HR star, slow drifts, and rate-limited fast effects
        return self.hr_star + self.tde_hr + self.smoothed_chemo_hr + self.smoothed_epi_hr
        
    def _calc_sv(self):
        hr = self._calc_hr()
        # SV = (SV_* + TDE_{SV}) * (1 - HR_{SV} * ln(HR/BaseHR))
        term = 1.0 - self.hr_sv_coupling * np.log(max(1.0, hr / self.base_hr))
        term = max(0.1, term)  # Prevent negative SV at extreme HR
        return (self.sv_star + self.tde_sv) * term
        
    def _calc_map(self):
        # RMAP = (HR*SV*TPR)/(BaseHR*BaseSV*BaseTPR)
        return self._calc_hr() * self._calc_sv() * self.tpr
        
    def _calc_co(self):
        return self._calc_hr() * self._calc_sv() / 1000.0 # L/min
        
    def step(self, dt: float, ce_prop: float, ce_remi: float, ce_nore: float, pit: float, paco2: float, pao2: float,
             dist_hr: float = 0.0, dist_sv: float = 0.0, dist_svr: float = 0.0, mac: float = 0.0,
             mac_sevo: float = 0.0, ce_epi: float = 0.0, ce_phenyl: float = 0.0, temp_c: float = 37.0) -> HemoState:
        """
        Advance hemodynamic model by dt seconds.
        
        Args:
            dt: Time step in seconds
            ce_prop: Propofol effect-site concentration (µg/mL)
            ce_remi: Remifentanil effect-site concentration (ng/mL)
            ce_nore: Norepinephrine plasma concentration (ng/mL)
            pit: Intrathoracic pressure (mmHg) - from ventilator coupling
            paco2: Arterial CO2 partial pressure (mmHg) - for chemoreflex
            pao2: Arterial O2 partial pressure (mmHg) - for chemoreflex
            dist_hr: External HR disturbance (bpm) - e.g., surgical stimulation
            dist_sv: External SV disturbance (mL)
            dist_svr: External SVR disturbance (Wood Units)
            mac: Generic MAC (legacy support)
            mac_sevo: Sevoflurane end-tidal MAC fraction (0.0-2.0+)
            ce_epi: Epinephrine plasma concentration (ng/mL)
            ce_phenyl: Phenylephrine plasma concentration (ng/mL)
            temp_c: Patient temperature (deg C)
            
        Returns:
            HemoState: Current hemodynamic state (MAP, HR, SV, SVR, CO)
        """
        # Invalidate state cache at start of step (state will change)
        self._cached_state = None
        
        dt_min = dt / 60.0
        
        # --- 0. Update Volatile Effect Site Concentrations ---
        # dCe/dt = ke0 * (ET - Ce)
        # Inputs are ET MAC fractions.
        self.ce_sevo += self.ke0_sevo * (mac_sevo - self.ce_sevo) * dt_min
        
        # --- 0.1 Volume Clearance ---
        # Reduce blood volume by clearance (urine, etc.)
        self.blood_volume -= self.vol_clearance * dt_min
        self.blood_volume = max(500.0, self.blood_volume) # Safety floor
        self._update_hb_conc()

        
        # --- 1. Calculate Effects ---
        
        # Age effect on Propofol SV Emax
        # Emax_PROP_SV(age) = Emax_PROP_SV_typ * exp(AGE_Emax_SV * (age - 35.0))
        emax_prop_sv = self.emax_prop_sv_typ * np.exp(self.age_emax_sv * (self.patient.age - 35.0))
        
        # --- 3.1 Propofol Effects ---
        
        # 1. On TPR (with Remi interaction)
        # Int term = INT_TPR * (Cremi / (EC50_REMI_TPR + Cremi))
        # Eff = (Emax + IntTerm) * (Cprop^g / (EC50^g + Cprop^g))
        
        # Safe vars
        cp = max(0.0, ce_prop)
        cr = max(0.0, ce_remi)
        cn = max(0.0, ce_nore)
        
        remi_int_term = self.int_tpr * (cr / (self.ec50_remi_tpr + cr + 1e-9))
        
        prop_hill = self._calc_hill(cp, self.ec50_prop_tpr, self.gamma_prop)
        
        eff_prop_tpr = (self.emax_prop_tpr + remi_int_term) * prop_hill
        
        # 2. On SV (no interaction)
        prop_hill_sv = self._calc_hill(cp, self.ec50_prop_sv, 1.0)
        eff_prop_sv = emax_prop_sv * prop_hill_sv
        
        # --- Volatile Effects (Mechanism-Based) ---
        
        # Remi Interaction Shift (Shift factor for EC50)
        # Shift S varies from 0 to max upon Remi
        # S = S_max * (Cremi / (EC50_int + Cremi))
        # EC50_eff = EC50 * (1 - S)
        remi_shift_factor = self.vol_remi_shift_max * (cr / (self.vol_remi_ec50 + cr + 1e-9))
        vol_ec50_mult = 1.0 - remi_shift_factor
        
        # Sevo (using class-level helper)
        eff_sevo_tpr = self._vol_hill(self.ce_sevo, self.sevo_emax_tpr, self.sevo_ec50_tpr, 
                                       self.sevo_gamma_tpr, vol_ec50_mult, use_shift=True)
        eff_sevo_sv = self._vol_hill(self.ce_sevo, self.sevo_emax_sv, self.sevo_ec50_sv, 
                                      1.0, vol_ec50_mult, use_shift=False)
        eff_sevo_hr = 0.0 # No HR effect
        
        # Combine Effects (Additive)
        total_eff_tpr = max(-0.95, eff_prop_tpr + eff_sevo_tpr)
        total_eff_sv = max(-0.95, eff_prop_sv + eff_sevo_sv)
        
        # HR Effects: Volatile HR is additive to HR production (EffVolHR is positive)
        total_eff_hr_prod = eff_sevo_hr
        
        # --- 3.2 Remi Effects ---
        
        # 3. On TPR dissipation
        remi_hill_tpr = self._calc_hill(cr, self.ec50_remi_tpr, self.gamma_remi_tpr)
        
        eff_remi_tpr = self.emax_remi_tpr * remi_hill_tpr
        
        # 4. On SV dissipation (slope modulated by Prop)
        slope_sv = self.sl_remi_sv + self.int_sv * (cp / (self.ec50_prop_sv + cp + 1e-9))
        eff_remi_sv = slope_sv * cr
        
        # 5. On HR dissipation (slope modulated by Prop)
        slope_hr = self.sl_remi_hr + self.int_hr * (cp / (self.ec50_int_hr + cp + 1e-9))
        eff_remi_hr = np.clip(slope_hr * cr, -0.9, 0.9)  # Prevent instability in (1 - eff)
        
        # =====================================================================
        # Physiological State Calculations
        # =====================================================================
        
        # --- 4.1 Blood Volume / MCFP / Preload ---
        # Calculate current MCFP from blood volume
        stressed_vol = max(0.0, self.blood_volume - self.unstressed_volume)
        mcfp = stressed_vol / self.cv
        
        # Preload factor from volume status
        f_preload_vol = mcfp / self.mcfp_0 if self.mcfp_0 > 0 else 1.0
        
        # --- 4.2 Intrathoracic Pressure (Pit) Effect on Preload ---
        # PEEP/PPV increases Pit, reducing venous return and preload
        delta_pit = pit - self.pit_0
        self.f_preload_pit = 1.0 / (1.0 + self.alpha_peep * max(0.0, delta_pit))
        
        # Combined preload factor (Note: This local calc was for ODE, now ODE doesn't use it, 
        # but we keep f_preload_pit updated for the state property above)
        f_preload = f_preload_vol * self.f_preload_pit
        
        # Apply Frank-Starling curve
        f_frank_starling = self._frank_starling(f_preload)
        
        # --- 4.3 CO2/O2 Chemoreflex ---
        if self.chemoreflex_active:
            # Hypercapnia increases HR and TPR production
            e_co2 = max(0.0, (paco2 - self.paco2_set) / self.paco2_set) if self.paco2_set > 0 else 0.0
            # Hypoxia increases HR
            e_o2 = max(0.0, (self.pao2_set - pao2) / self.pao2_set) if self.pao2_set > 0 else 0.0
            
            # Chemoreflex contributions
            chemo_hr_boost = self.g_hr_co2 * e_co2 + self.g_hr_o2 * e_o2
            chemo_tpr_factor = 1.0 + self.k_tpr_co2 * e_co2
        else:
            chemo_hr_boost = 0.0
            chemo_tpr_factor = 1.0
        
        # --- 4.4 Hemorrhage Response ---
        self.hemorrhage_hr_mult, self.hemorrhage_tpr_mult = self._calc_hemorrhage_response()
        
        # --- 4.5 Unified Vasopressor Effects ---
        # All vasopressors now use consistent helper architecture
        epi_delta_hr, epi_sv_factor, epi_svr_factor = self._calc_epi_effects(ce_epi)
        nore_delta_hr, nore_sv_factor, nore_svr_factor = self._calc_nore_effects(ce_nore)
        phenyl_svr_factor = self._calc_phenyl_effects(ce_phenyl)
        
        # Combine vasopressor effects
        # SVR: Multiplicative (each drug independently increases vascular resistance)
        combined_svr_factor = epi_svr_factor * nore_svr_factor * phenyl_svr_factor
        # SV: Multiplicative (inotropy effects stack)
        combined_sv_factor = epi_sv_factor * nore_sv_factor
        # HR: Additive (chronotropy effects sum)
        combined_delta_hr = epi_delta_hr + nore_delta_hr
        
        # Store for inotropy application in ODE
        self.vasopressor_sv_factor = combined_sv_factor
        
        # Convert combined SVR factor to delta TPR for consistent integration
        delta_tpr_vasopressors = self.base_tpr * (combined_svr_factor - 1.0)
        self.delta_tpr_vasopressors = delta_tpr_vasopressors
        
        # --- ODE Integration ---
        
        # RMAP = (HR * SV * TPR) / (BaseHR * BaseSV * BaseTPR)
        current_hr = self._calc_hr()
        
        # Apply preload to SV for baroreflex sensing (mirrors state property)
        raw_sv = self._calc_sv()
        current_sv = raw_sv * f_frank_starling
        
        # TPR is self.tpr but baroreflex senses MAP which includes drug effects
        # Use dist_svr (argument) instead of self.dist_svr (stored) for lag-free response
        effective_tpr = self.tpr + self.delta_tpr_vasopressors + (dist_svr / 1000.0)
        effective_tpr = max(0.006, effective_tpr)
        
        rmap = (current_hr * current_sv * effective_tpr) / (self.base_hr * self.base_sv * self.base_tpr)
        rmap = clamp(rmap, 0.1, 5.0)  # Clamp to prevent NaN from negative exponent
        rmap_fb = rmap ** self.fb
        
        # Derivatives with new physiological factors
        
        # Thermoregulation (Vasoconstriction)
        # Threshold 36.5 C. 
        # Gain: Increase TPR.
        # Inhibition: Propofol/Volatiles reduce threshold or gain.
        # Sessler 2016: Anesthesia decreases threshold from ~37 to ~34.5.
        
        # Calculate anesthetic depth effect on threshold
        # Approx 1 MAC or 4ug/mL Prop -> Threshold drops by ~2.5 C
        depth_metric = mac_sevo + (ce_prop / 4.0)
        threshold_drop = 2.5 * min(1.0, depth_metric)
        vasoconstriction_threshold = 36.5 - threshold_drop
        
        thermo_tpr_mult = 1.0
        if temp_c < vasoconstriction_threshold:
            # Cold and below anesthetized threshold -> Vasoconstrict
            delta_t = vasoconstriction_threshold - temp_c
            thermo_tpr_mult = 1.0 + TEMP_TPR_COEFFICIENT * delta_t  # ~10% per degree
            
        # Limit max vasoconstriction
        thermo_tpr_mult = min(2.0, thermo_tpr_mult)

        # dTPR/dt
        # Apply chemoreflex TPR boost, hemorrhage TPR compensation, and thermoregulation
        tpr_production = self.kin_tpr * rmap_fb * (1.0 + total_eff_tpr) * chemo_tpr_factor * self.hemorrhage_tpr_mult * thermo_tpr_mult
        tpr_dissipation = self.kout * self.tpr * (1.0 - eff_remi_tpr)
        d_tpr = tpr_production - tpr_dissipation
        
        # dSV_star/dt
        # Apply combined vasopressor inotropy
        # Preload effect is applied directly to output SV in the state property.
        sv_production = self.kin_sv * rmap_fb * (1.0 + total_eff_sv) * combined_sv_factor
        sv_dissipation = self.kout * self.sv_star * (1.0 - eff_remi_sv)
        d_sv_star = sv_production - sv_dissipation
        
        # dHR_star/dt
        # Note: eff_remi_hr can be negative due to propofol interaction
        # Negative eff should INCREASE dissipation (reduce HR) - use (1 - eff)
        hr_production = self.kin_hr * rmap_fb * (1.0 + total_eff_hr_prod) * self.hemorrhage_hr_mult
        hr_dissipation = self.kout * self.hr_star * (1.0 - eff_remi_hr)
        d_hr_star = hr_production - hr_dissipation
        
        # Store chemoreflex and combined vasopressor HR effects as targets
        self.current_chemo_hr = chemo_hr_boost
        self.current_epi_hr = combined_delta_hr  # Now includes all vasopressor HR effects
        
        # Apply rate limiting via exponential smoothing
        # dx/dt = (target - x) / tau  =>  x_new = x + (target - x) * dt / tau
        alpha_fast = dt / self.tau_hr_fast if self.tau_hr_fast > 0 else 1.0
        alpha_fast = min(1.0, alpha_fast)  # Clamp to avoid overshoot
        self.smoothed_chemo_hr += (self.current_chemo_hr - self.smoothed_chemo_hr) * alpha_fast
        self.smoothed_epi_hr += (self.current_epi_hr - self.smoothed_epi_hr) * alpha_fast
        
        # Drifts (Standard decay)
        d_tde_hr = -self.k_drift * self.tde_hr
        d_tde_sv = -self.k_drift * self.tde_sv
        
        # Update state (Euler integration)
        self.tpr += d_tpr * dt_min
        self.sv_star += d_sv_star * dt_min
        self.hr_star += d_hr_star * dt_min
        self.tde_hr += d_tde_hr * dt_min
        self.tde_sv += d_tde_sv * dt_min
        
        # Store disturbances for state property
        self.dist_hr = dist_hr
        self.dist_sv = dist_sv
        self.dist_svr = dist_svr
        
        return self.state

    def calculate_steady_state(self, ce_prop: float, ce_remi: float, ce_nore: float, mac: float = 0.0) -> HemoState:
        """
        Run simulation for a long time to find steady state.
        Analytical solution is hard due to feedback loops.
        """
        # Save state
        saved_tpr = self.tpr
        saved_sv = self.sv_star
        saved_hr = self.hr_star
        saved_dist = (self.tde_sv, self.tde_hr)
        
        # Fast forward
        # Drifts decay to 0 over time
        self.tde_sv = 0
        self.tde_hr = 0
        
        # At SS: d/dt = 0
        # This reduces to a system of 1 variable: RMAP (Relative MAP)
        # We solve f(Z) = Z_calc(Z) - Z = 0 where Z = RMAP
        
        # Calculate drug effects
        cp = max(0.0, ce_prop)
        cr = max(0.0, ce_remi)
        cn = max(0.0, ce_nore)
        
        # Effects
        emax_prop_sv = self.emax_prop_sv_typ * np.exp(self.age_emax_sv * (self.patient.age - 35.0))
        remi_int_term = self.int_tpr * (cr / (self.ec50_remi_tpr + cr + 1e-9))
        prop_hill = self._calc_hill(cp, self.ec50_prop_tpr, self.gamma_prop)
        eff_prop_tpr = (self.emax_prop_tpr + remi_int_term) * prop_hill
        
        prop_hill_sv = self._calc_hill(cp, self.ec50_prop_sv, 1.0)
        eff_prop_sv = emax_prop_sv * prop_hill_sv
        
        # Volatile effects (Ce = MAC at steady state)
        ce_sevo_ss = mac
        
        # Remi interaction shift (EC50 modifier)
        remi_shift_factor = self.vol_remi_shift_max * (cr / (self.vol_remi_ec50 + cr + 1e-9))
        vol_ec50_mult = 1.0 - remi_shift_factor
        
        # Sevo (using class-level helper)
        eff_sevo_tpr = self._vol_hill(ce_sevo_ss, self.sevo_emax_tpr, self.sevo_ec50_tpr, 
                                       self.sevo_gamma_tpr, vol_ec50_mult, use_shift=True)
        eff_sevo_sv = self._vol_hill(ce_sevo_ss, self.sevo_emax_sv, self.sevo_ec50_sv, 
                                      1.0, vol_ec50_mult, use_shift=False)
        eff_sevo_hr = 0.0 
        
        # Combined Effects (Additive)
        total_eff_tpr = max(-0.95, eff_prop_tpr + eff_sevo_tpr)
        total_eff_sv = max(-0.95, eff_prop_sv + eff_sevo_sv)
        total_eff_hr_prod = eff_sevo_hr # 0
        
        remi_hill_tpr = self._calc_hill(cr, self.ec50_remi_tpr, self.gamma_remi_tpr)
        eff_remi_tpr = self.emax_remi_tpr * remi_hill_tpr
        
        slope_sv = self.sl_remi_sv + self.int_sv * (cp / (self.ec50_prop_sv + cp + 1e-9))
        eff_remi_sv = slope_sv * cr
        
        slope_hr = self.sl_remi_hr + self.int_hr * (cp / (self.ec50_int_hr + cp + 1e-9))
        eff_remi_hr = slope_hr * cr
        
        # Norepinephrine effects
        nore_hill = self._calc_hill(cn, self.nore_c50, self.nore_gamma)
        delta_map_nore = self.nore_emax_map * nore_hill
        
        # Delta TPR from Norepi
        base_co_ml = self.base_hr * self.base_sv
        delta_tpr_nore = delta_map_nore / base_co_ml
        
        def residual(z):
            if z <= 0.01: z = 0.01
            z_fb = z ** self.fb
            
            # HR = Base * RMAP^FB * (1+EffVolHR) / (1 - EffRemiHR)
            # Using (1 - eff) for dissipation to match step() function
            hr_z = self.base_hr * z_fb * (1.0 + total_eff_hr_prod) / (1.0 - eff_remi_hr)
            sv_star_z = self.base_sv * z_fb * (1.0 + total_eff_sv) / (1.0 - eff_remi_sv)
            tpr_z = self.base_tpr * z_fb * (1.0 + total_eff_tpr) / (1.0 - eff_remi_tpr)
            
            # Coupling
            term = 1.0 - self.hr_sv_coupling * np.log(max(1.0, hr_z / self.base_hr))
            sv_z = sv_star_z * term
            
            # Effective TPR (Norepi)
            eff_tpr_z = tpr_z + delta_tpr_nore
            
            # Recalc Z
            z_new = (hr_z * sv_z * eff_tpr_z) / (self.base_hr * self.base_sv * self.base_tpr)
            return z_new - z
            
        try:
            sol = root_scalar(residual, bracket=[0.01, 5.0], method='brentq')
            z_ss = sol.root
        except (ValueError, RuntimeError):
            z_ss = 1.0  # fallback to baseline
            
        # Set State
        z_fb = z_ss ** self.fb
        # Using (1 - eff) for dissipation to match step() function
        self.hr_star = self.base_hr * z_fb * (1.0 + total_eff_hr_prod) / (1.0 - eff_remi_hr)
        self.sv_star = self.base_sv * z_fb * (1.0 + total_eff_sv) / (1.0 - eff_remi_sv)
        self.tpr = self.base_tpr * z_fb * (1.0 + total_eff_tpr) / (1.0 - eff_remi_tpr)
        self.tde_hr = 0
        self.tde_sv = 0
        
        # Set Internal Volatile State
        self.ce_sevo = ce_sevo_ss

        
        # Final Output
        # Re-run calc from self values
        # Add Norepi
        # Pass mac_sevo explicitly based on mac input
        ret = self.step(0.0, ce_prop, ce_remi, ce_nore, -2.0, 40.0, 95.0, 0,0,0, 0.0, mac_sevo=mac)
        
        # Revert internal state
        self.tpr = saved_tpr
        self.sv_star = saved_sv
        self.hr_star = saved_hr
        self.tde_sv = saved_dist[0]
        self.tde_hr = saved_dist[1]
        
        return ret
