import math
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import root_scalar
from anasim.patient.patient import Patient
from anasim.core.constants import (
    HR_MIN, HR_MAX, MAP_MAX, SBP_MAX, DBP_MAX, TPR_MIN, BLOOD_VOLUME_MIN, TEMP_TPR_COEFFICIENT
)
from anasim.core.utils import hill_function, clamp, clamp01
from anasim.core.enums import RhythmType
from .hemo_config import HemodynamicConfig


@dataclass
class HemoState:
    """
    Hemodynamic state snapshot representing typical adult baselines.

    These are overridden at runtime by model calculations.
    """
    map: float = 80.0   # Mean Arterial Pressure (mmHg)
    hr: float = 75.0    # Heart Rate (bpm)
    sv: float = 70.0    # Stroke Volume (mL)
    svr: float = 16.0   # SVR (mmHg*min/L) ~ Wood Units
    co: float = 5.25    # Cardiac Output (L/min)
    sbp: float = 115.0  # Systolic Blood Pressure (mmHg)
    dbp: float = 70.0   # Diastolic Blood Pressure (mmHg)
    rhythm_type: RhythmType = RhythmType.SINUS


@dataclass
class HemoStateExtended(HemoState):
    """
    Extended hemodynamic state with internal model variables.
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
    Core model from Su et al. Br J Anaesth. 2023 (mechanistic interaction model).
    Extended with ventilator-circulation coupling, blood volume tracking,
    chemoreflexes, and additional vasoactive drug support.
    
    Literature References:
    ----------------------
    Core Model:
        Su et al. Br J Anaesth. 2023.
    
    Arterial Compliance:
        Age-dependent compliance: C(age) = 1.5 × (1 - 0.008×(age-40)) mL/mmHg
    
    Vasopressors:
        - Norepinephrine: Beloeil et al. Br J Anaesth. 2005.
        - Epinephrine: Clutter et al. J Clin Invest. 1980.
        - Phenylephrine: Anderson et al. Paediatr Anaesth. 2017 (population PK in children; PD here is heuristic).
    
    Volatile Agents:
        Davis & Mapleson. Br J Anaesth. 1981 (physiological model of inhaled agents).
        Ebert et al. Anesthesiology. 1995.
        
    Units:
        - MAP, SBP, DBP: mmHg
        - HR: bpm
        - SV: mL
        - CO: L/min
        - SVR: Wood Units (mmHg×min/L)
        - TPR: mmHg×min/mL (internal)
    """
    def __init__(self, patient: Patient, fidelity_mode: str = "clinical",
                 config: Optional[HemodynamicConfig] = None):
        self.patient = patient
        use_literature = (fidelity_mode == "literature")
        self.config = config or HemodynamicConfig()
        # Caches should exist before any config-driven invalidation.
        self._cached_state: Optional[HemoStateExtended] = None
        self._hill_cache = {}
        self._apply_config(self.config)

        # =====================================================================
        # Core Model Parameters (Su et al. Br J Anaesth. 2023)
        # =====================================================================

        # Hemoglobin / hematocrit baselines
        self.baseline_hb = getattr(patient, 'baseline_hb', self.baseline_hb)
        self.baseline_hct = getattr(patient, 'baseline_hct', self.baseline_hct)

        # Baseline HR from anasim.patient or default
        self.base_hr = patient.baseline_hr if patient.baseline_hr > 0 else self.base_hr

        # Baseline SV derived from cardiac index and BSA
        ci_0 = self.ci_elderly if patient.age > self.ci_elderly_age else self.ci_adult
        co_0 = ci_0 * patient.bsa if patient.bsa > 0 else ci_0 * self.bsa_fallback
        self.base_sv = (co_0 * 1000.0) / self.base_hr if self.base_hr > 0 else self.base_sv

        # Baseline TPR from MAP and flow
        flow_ml_min = self.base_hr * self.base_sv
        self.base_tpr = patient.baseline_map / flow_ml_min if flow_ml_min > 0 else self.base_tpr
        self.base_co_l_min = flow_ml_min / 1000.0
        self.baseline_caO2 = self.calc_oxygen_content(self.baseline_hb, 0.98, 95.0)
        self.baseline_do2 = self.baseline_caO2 * self.base_co_l_min * 10.0

        # Arterial Compliance (age-dependent model)
        # Decreases with age -> wider pulse pressure in elderly
        # C(age) ≈ 1.5 * (1 - 0.008 * (age - 40))
        age = patient.age
        age_factor = 1.0 - self.arterial_age_slope * (age - self.arterial_age_ref)
        age_factor = clamp(age_factor, self.arterial_age_min, self.arterial_age_max)
        self.arterial_compliance = self.arterial_compliance_ref * age_factor

        # Feedback parameters
        # Su et al. Br J Anaesth. 2023 define FB = 0.66 in Table 2.
        # Implementing self.fb = -0.66 ensures stable negative feedback
        # given the structure of our ODE implementation (Production ~ RMAP^FB).
        # This sign flip is a deviation for numerical stability/correctness in this solver.
        # External Modifiers (Persistence for property access)
        self.delta_tpr_vasopressors = 0.0
        self.dist_svr = 0.0
        self.dist_hr = 0.0
        self.dist_sv = 0.0
        self.chemoreflex_active = True
        self.sepsis_severity = 0.0
        self.anaphylaxis_severity = 0.0
        
        # Instantaneous effects with rate limiting (exponential smoothing)
        # Target values (calculated each step)
        self.current_epi_hr = 0.0
        self.current_chemo_hr = 0.0
        # Smoothed values (used in _calc_hr) with time constants
        self.smoothed_epi_hr = 0.0
        self.smoothed_chemo_hr = 0.0
        # Time constants: Physiologically fast, although not instantaneous.
        # tau_hr = 5s: Peak effect within ~15s (3*tau), matches clinical phenylephrine bolus
        
        # Initial Conditions
        self.sv_star = self.base_sv
        self.hr_star = self.base_hr
        self.tpr = self.base_tpr
        
        # Initial drifts (start at zero for stable baseline)
        self.tde_sv = 0.0
        self.tde_hr = 0.0

        # Frank-Starling normalization constant (used every step)
        self._frank_starling_baseline_raw = 1.0 - math.exp(-2.0)  # ≈ 0.865
        
        # Derived Kin for equilibrium at RMAP=1
        self.kin_tpr = self.kout * self.base_tpr
        self.kin_sv = self.kout * self.base_sv
        self.kin_hr = self.kout * self.base_hr
        
        # =====================================================================
        # Drug Effect Parameters (Su et al. Br J Anaesth. 2023 + literature extensions)
        # =====================================================================
        # Su et al. Br J Anaesth. 2023 found Emax -0.78 (78% drop). This often yields
        # clinically unrealistic hypotension (MAP <60) in steady state simulation.
        # Clinical realism uses -0.50 (50% max drop) to maintain MAP ~70-75 mmHg during maintenance.
        self.emax_prop_tpr = self.emax_prop_tpr_literature if use_literature else self.emax_prop_tpr_clinical

        # State init
        self.ce_sevo = 0.0

        # =========================================================================
        # Blood Volume and Preload Tracking
        # =========================================================================
        
        # --- Blood Volume Tracking (MCFP / Stressed Volume) ---
        if hasattr(patient, 'estimate_blood_volume'):
            self.blood_volume = patient.estimate_blood_volume()
        else:
            self.blood_volume = self.default_blood_volume  # Default 5L
        self.blood_volume_0 = self.blood_volume  # Save baseline for hemorrhage calc
        self.hb_mass = self.baseline_hb * (self.blood_volume / 100.0)  # grams
        self.hb_conc = self.baseline_hb
        
        # Unstressed volume: ~70% of total (standard physiology)
        # This leaves ~30% as stressed volume (~1400mL for 5L blood volume)
        # Hemorrhage depletes stressed volume first, reducing preload/MCFP
        self.unstressed_volume = self.blood_volume * self.unstressed_volume_fraction
        
        # Venous compliance (mL/mmHg)
        self.cv = self.venous_compliance
        
        # Baseline MCFP (Mean Circulatory Filling Pressure)
        # Uses Guytonian concept: MCFP = Stressed_Vol / Compliance
        # Values are heuristics to match standard physiology (~10-15 mmHg)
        stressed_vol_0 = max(0.0, self.blood_volume - self.unstressed_volume)
        self.mcfp_0 = max(self.mcfp_floor, stressed_vol_0 / self.cv)  # ~10-15 mmHg normally
        
        self.cumulative_fluid_given = 0.0
        
        self.rhythm_type = RhythmType.SINUS
        self.f_preload_pit = 1.0
        self.vasopressor_sv_factor = 1.0

        # Hill function parameters cache
        self._hill_params = {
            "prop_tpr": (self.ec50_prop_tpr ** self.gamma_prop, self.gamma_prop),
            "prop_sv": (self.ec50_prop_sv, 1.0),
            "remi_tpr": (self.ec50_remi_tpr, 1.0),
        }

    def _apply_config(self, config: HemodynamicConfig) -> None:
        for name, value in vars(config).items():
            setattr(self, name, value)
        self._hill_cache.clear()

    def invalidate_state_cache(self) -> None:
        """Public cache invalidation for external state mutations."""
        self._cached_state = None

    def set_nore_pd(self, c50: float, emax: float = 98.7, gamma: float = 1.8):
        self.nore_c50 = c50
        self.nore_emax_map = emax
        self.nore_gamma = gamma
        self._hill_cache.clear()
    
    def _hill_cached(self, key: str, ce: float) -> float:
        if ce <= 0:
            return 0.0
        cached_ce, cached_result = self._hill_cache.get(key, (0.0, 0.0))
        if cached_ce > 0 and abs(ce - cached_ce) / cached_ce < self.cache_tolerance:
            return cached_result
        c50_pow, gamma = self._hill_params[key]
        ce_pow = ce ** gamma
        result = ce_pow / (c50_pow + ce_pow)
        self._hill_cache[key] = (ce, result)
        return result

    @staticmethod
    def _vol_hill(ce: float, emax: float, ec50_base: float, gamma: float,
                  vol_ec50_mult: float = 1.0, use_shift: bool = True) -> float:
        """Hill function for volatile anesthetic effects with EC50 shift."""
        eff_ec50 = ec50_base * vol_ec50_mult if use_shift else ec50_base
        return emax * hill_function(ce, eff_ec50, gamma)
        
    def add_volume(self, amount_ml: float, hematocrit: float = 0.0):
        """
        Simulate fluid bolus or hemorrhage.
        Updates blood_volume directly for MCFP-based preload calculation.
        Also applies transient SV boost via tde_sv (Frank-Starling).
        """
        self._cached_state = None
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

    def _calc_stressed_volume(self, sepsis_sev: Optional[float] = None) -> float:
        """
        Compute stressed volume with sepsis-related venous pooling.

        Sepsis increases venous capacitance (relative hypovolemia),
        modeled as a fraction of baseline volume shifted to unstressed.
        """
        sev = clamp01(self.sepsis_severity) if sepsis_sev is None else sepsis_sev
        pooling = self.sepsis_pooling_fraction * self.blood_volume_0 * sev
        return max(0.0, self.blood_volume - self.unstressed_volume - pooling)

    def get_hematocrit(self) -> float:
        if self.baseline_hb <= 0:
            return self.baseline_hct
        ratio = self.hb_conc / self.baseline_hb
        return clamp(self.baseline_hct * ratio, 0.0, 0.7)

    @staticmethod
    def calc_oxygen_content(hb_g_dl: float, sao2_frac: float, pao2: float) -> float:
        sao2_frac = clamp01(sao2_frac)
        return hb_g_dl * 1.34 * sao2_frac + 0.003 * pao2

    def compute_do2_ratio(self, sao2_frac: float, pao2: float, co_l_min: float) -> float:
        caO2 = self.calc_oxygen_content(self.hb_conc, sao2_frac, pao2)
        do2 = caO2 * max(0.0, co_l_min) * 10.0
        if self.baseline_do2 <= 0:
            return 1.0
        ratio = do2 / self.baseline_do2
        return clamp(ratio, 0.0, 2.0)
    
    def _frank_starling(self, preload_factor: float, inotropy: float = 1.0) -> float:
        """
        Frank-Starling curve: SV increases with preload until reaching a plateau.
        Based on: SV = SVmax × (1 - exp(-k × LVEDV))
        Approximated as: SV_factor = inotropy × (1 - exp(-2.0 × preload_factor))
        Includes clinical floor for severe hypovolemia (~25-30% baseline SV).
        """
        # Clamp preload_factor to avoid extreme values
        pf = clamp(preload_factor, 0.01, 2.5)
        
        # Raw Frank-Starling curve
        raw_factor = 1.0 - math.exp(-2.0 * pf)
        
        # Normalize so that pf=1.0 gives factor=1.0
        normalized = raw_factor / self._frank_starling_baseline_raw
        
        # Apply inotropy
        result = inotropy * normalized
        
        # Clinical floor: minimum SV = 5% of baseline even in severe hypovolemia
        # This accounts for compensatory mechanisms not fully modeled
        return max(0.05, result)
    
    def _calc_hemorrhage_response(self) -> tuple:
        """
        Calculate hemorrhage class-based sympathetic compensation (HR/TPR multipliers).
        Based on ATLS hemorrhage classification (I-IV).
        These multipliers work IN ADDITION to the baroreflex.
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
        epi_hill = hill_function(ce_epi, self.epi_c50, self.epi_gamma)
        
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
            
            low_contrib = self.epi_emax_svr_low * hill_function(self.epi_threshold, self.epi_c50, self.epi_gamma)

            high_hill = hill_function(excess, self.epi_c50, self.epi_gamma)
            high_contrib = self.epi_emax_svr_high * high_hill
            
            svr_factor = 1.0 + low_contrib * low_fraction + high_contrib
        
        return delta_hr, sv_factor, max(0.5, svr_factor)
    
    def _calc_phenyl_effects(self, ce_phenyl: float) -> float:
        """Phenylephrine SVR effect (pure alpha-1)."""
        if ce_phenyl <= 0: return 1.0
        return 1.0 + self.phenyl_emax_svr * hill_function(ce_phenyl, self.phenyl_c50, self.phenyl_gamma)
    
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
        # - Beloeil et al. Br J Anaesth. 2005 (PK/PD context); EC50 here is heuristic.
        # - Typical ICU range: 0.05-0.5 mcg/kg/min → Ce ~2-20 ng/mL
        
        Returns:
            (delta_hr, sv_factor, svr_factor): Additive HR, multiplicative SV/SVR
        """
        if ce_nore <= 0:
            return 0.0, 1.0, 1.0
        
        nore_hill = hill_function(ce_nore, self.nore_c50, self.nore_gamma)
        
        # Beta-1 chronotropy (modest compared to epinephrine)
        delta_hr = self.nore_emax_hr * nore_hill
        
        # Beta-1 inotropy
        sv_factor = 1.0 + self.nore_emax_sv * nore_hill
        
        # Alpha-1 vasoconstriction (dominant effect)
        # Convert Emax_MAP to SVR factor: delta_MAP / base_MAP
        # At Emax (100 mmHg increase), SVR increases ~125% (from ~80 to ~180 mmHg)
        svr_factor = 1.0 + (self.nore_emax_map * nore_hill) / 80.0
        
        return delta_hr, sv_factor, svr_factor
    
    def _calc_anesthetic_effects(self, ce_prop: float, ce_remi: float, ce_sevo: float) -> tuple:
        """
        Calculate combined anesthetic effects on TPR, SV, and HR.
        Returns: (total_eff_tpr, total_eff_sv, total_eff_hr, eff_remi_tpr, eff_remi_sv, eff_remi_hr)
        """
        cp = max(0.0, ce_prop)
        cr = max(0.0, ce_remi)
        
        # --- Propofol Effects ---
        # Age effect on Propofol SV Emax
        emax_prop_sv = self.emax_prop_sv_typ * math.exp(self.age_emax_sv * (self.patient.age - 35.0))
        
        # 1. On TPR (with Remi interaction)
        remi_int_term = self.int_tpr * (cr / (self.ec50_remi_tpr + cr + 1e-9))
        prop_hill_tpr = self._hill_cached("prop_tpr", cp)
        eff_prop_tpr = (self.emax_prop_tpr + remi_int_term) * prop_hill_tpr
        
        # 2. On SV (no interaction)
        prop_hill_sv = self._hill_cached("prop_sv", cp)
        eff_prop_sv = emax_prop_sv * prop_hill_sv
        
        # --- Volatile Effects (Mechanism-Based) ---
        # Remi Interaction Shift (Shift factor for EC50)
        remi_shift_factor = self.vol_remi_shift_max * (cr / (self.vol_remi_ec50 + cr + 1e-9))
        vol_ec50_mult = 1.0 - remi_shift_factor
        
        # Sevo
        eff_sevo_tpr = self._vol_hill(ce_sevo, self.sevo_emax_tpr, self.sevo_ec50_tpr, 
                                       self.sevo_gamma_tpr, vol_ec50_mult, use_shift=True)
        eff_sevo_sv = self._vol_hill(ce_sevo, self.sevo_emax_sv, self.sevo_ec50_sv, 
                                      1.0, vol_ec50_mult, use_shift=False)
        eff_sevo_hr = 0.0 # No HR effect
        
        # --- Combined Main Effects (Additive) ---
        total_eff_tpr = max(-0.95, eff_prop_tpr + eff_sevo_tpr)
        total_eff_sv = max(-0.95, eff_prop_sv + eff_sevo_sv)
        total_eff_hr = eff_sevo_hr
        
        # --- Remi Effects (Dissipation Modulation) ---
        # 1. On TPR dissipation
        remi_hill_tpr = self._hill_cached("remi_tpr", cr)
        eff_remi_tpr = self.emax_remi_tpr * remi_hill_tpr
        
        # 2. On SV dissipation (slope modulated by Prop)
        slope_sv = self.sl_remi_sv + self.int_sv * (cp / (self.ec50_prop_sv + cp + 1e-9))
        eff_remi_sv = slope_sv * cr
        
        # 3. On HR dissipation (slope modulated by Prop)
        slope_hr = self.sl_remi_hr + self.int_hr * (cp / (self.ec50_int_hr + cp + 1e-9))
        eff_remi_hr = max(-0.9, min(0.9, slope_hr * cr))
        
        return total_eff_tpr, total_eff_sv, total_eff_hr, eff_remi_tpr, eff_remi_sv, eff_remi_hr
    
    def _compute_state(
        self,
        preload_sv_factor: Optional[float] = None,
        sepsis_sev: Optional[float] = None,
        anaph_sev: Optional[float] = None,
        distributive_tpr_offset: Optional[float] = None,
        hr_base: Optional[float] = None,
    ) -> HemoStateExtended:
        if sepsis_sev is None:
            sepsis_sev = clamp01(self.sepsis_severity)
        if anaph_sev is None:
            anaph_sev = clamp01(self.anaphylaxis_severity)

        if hr_base is None:
            hr_base = self._calc_hr()
        current_hr = hr_base + self.dist_hr
        current_hr = max(HR_MIN, current_hr)
        current_hr = min(HR_MAX, current_hr)

        # --- Arrhythmia Overrides (HR) ---
        # References:
        # - AFib RVR: AHA guidelines, typical rate 110-150 bpm untreated
        # - SVT: Rate typically 150-220 bpm (Wikipedia, Mayo Clinic)
        # - VTach: Rate typically 150-250 bpm (Cleveland Clinic)
        if self.rhythm_type == RhythmType.SVT:
            current_hr = 160.0  # Typical SVT rate
        elif self.rhythm_type == RhythmType.VTACH:
            current_hr = 180.0  # Monomorphic VT rate
        elif self.rhythm_type == RhythmType.VFIB:
            current_hr = 0.0  # No organized contraction
        elif self.rhythm_type == RhythmType.ASYSTOLE:
            current_hr = 0.0
        elif self.rhythm_type == RhythmType.AFIB:
            # Untreated AFib with RVR (Rapid Ventricular Response)
            # Rate typically 110-150 bpm. Use floor of 110 if otherwise lower.
            current_hr = max(current_hr, 110.0)
        elif self.rhythm_type == RhythmType.SINUS_BRADY:
            # Sinus bradycardia: rate < 60 bpm
            current_hr = min(current_hr, 50.0)

        # SV - with preload modulation from current blood volume/MCFP
        term = 1.0 - self.hr_sv_coupling * math.log(max(1.0, current_hr / self.base_hr))
        raw_sv = (self.sv_star + self.tde_sv) * term + self.dist_sv

        # Apply preload factor (MCFP-dependent via Frank-Starling)
        # This provides immediate SV reduction when blood volume drops
        if preload_sv_factor is None:
            stressed_vol = self._calc_stressed_volume(sepsis_sev)
            mcfp = stressed_vol / self.cv if self.cv > 0 else self.mcfp_0
            preload_ratio = mcfp / self.mcfp_0 if self.mcfp_0 > 0 else 1.0
            total_preload = preload_ratio * self.f_preload_pit
            preload_sv_factor = self._frank_starling(total_preload)

        # Apply preload and vasopressor inotropy for immediate visibility
        current_sv = raw_sv * preload_sv_factor * self.vasopressor_sv_factor
        current_sv = max(1.0, current_sv)

        # --- Arrhythmia Overrides (SV) ---
        if self.rhythm_type == RhythmType.AFIB:
            # Loss of atrial kick
            current_sv *= 0.80
        elif self.rhythm_type == RhythmType.VTACH:
            # Loss of AV synchrony + very short filling time + dyssynchronous contraction
            # Target: CO ~40% of baseline (HR 180/75 = 2.4x, need SV ~0.17x)
            # HR-SV coupling already reduces SV to ~0.75x at HR 180. Additional penalty: 0.25x
            current_sv *= 0.25
        elif self.rhythm_type == RhythmType.VFIB or self.rhythm_type == RhythmType.ASYSTOLE:
            current_sv = 0.0
        elif self.rhythm_type == RhythmType.SVT:
            # Moderate compromise due to reduced filling time
            # Target: CO ~80-90% baseline (HR ~160/75 = 2.1x, need SV ~0.4x)
            # HR-SV coupling gives ~0.80x. Additional penalty: 0.50x
            current_sv *= 0.50

        co = current_hr * current_sv / 1000.0

        if distributive_tpr_offset is None:
            distributive_svr_drop = (self.sepsis_svr_drop_wood * sepsis_sev +
                                     self.anaphylaxis_svr_drop_wood * anaph_sev)
            distributive_tpr_offset = -distributive_svr_drop / 1000.0
        eff_tpr = self.tpr + self.delta_tpr_vasopressors + (self.dist_svr / 1000.0) + distributive_tpr_offset
        # TPR floor: Minimum physiologically viable resistance
        # 0.001 is too low - use ~0.006 (equivalent to ~6 Wood Units at normal flow)
        eff_tpr = max(0.006, eff_tpr)

        map_val = current_hr * current_sv * eff_tpr

        # Clamp MAP for numerical stability and physiological sanity
        map_val = clamp(map_val, 5.0, 300.0)

        # SVR output
        svr_val = map_val / co if co > 0 else 0.0

        # SBP/DBP calculation with age-dependent arterial compliance
        # Pulse Pressure (PP) = SV / Arterial_Compliance
        # Distributed around MAP: DBP = MAP - 1/3*PP, SBP = MAP + 2/3*PP
        pulse_pressure = current_sv / self.arterial_compliance

        # Clamp pulse pressure to physiological range
        # Normal: 30-60 mmHg, Wide: up to 100 in elderly/atherosclerotic
        pulse_pressure = clamp(pulse_pressure, 10.0, 120.0)

        # Distribute around MAP (diastole = 2/3 of cardiac cycle in duration;
        # however, pressure is time-weighted toward diastole, making MAP closer to DBP)
        dbp = map_val - (1.0 / 3.0) * pulse_pressure
        sbp = map_val + (2.0 / 3.0) * pulse_pressure

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
            rhythm_type=self.rhythm_type
        )

        self._cached_state = computed_state
        return computed_state

    @property
    def state(self) -> HemoStateExtended:
        if self._cached_state is not None:
            return self._cached_state
        return self._compute_state()
        
    @state.setter
    def state(self, new_state: HemoState):
        self._cached_state = None
        
        if isinstance(new_state, HemoStateExtended):
            self.tpr = new_state.tpr
            self.sv_star = new_state.sv_star
            self.hr_star = new_state.hr_star
            self.tde_sv = new_state.tde_sv
            self.tde_hr = new_state.tde_hr
            self.ce_sevo = new_state.ce_sevo
            self.rhythm_type = new_state.rhythm_type

        else:
            # Fallback for vanilla HemoState (approximation)
            # Assuming TDEs are 0 (steady state forced)
            self.tde_sv = 0.0
            self.tde_hr = 0.0
            
            self.hr_star = new_state.hr
            
            # SV = SV_star * (1 - k * ln(HR/Base))
            term = 1.0 - self.hr_sv_coupling * math.log(max(1.0, self.hr_star / self.base_hr))
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
        term = 1.0 - self.hr_sv_coupling * math.log(max(1.0, hr / self.base_hr))
        term = max(0.1, term)  # Prevent negative SV at extreme HR
        return (self.sv_star + self.tde_sv) * term
        
    def _calc_map(self):
        # MAP proxy (mmHg) from current HR/SV/TPR state
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
        sepsis_sev = clamp01(self.sepsis_severity)
        anaph_sev = clamp01(self.anaphylaxis_severity)
        
        # --- 0. Update Volatile Effect Site Concentrations ---
        # dCe/dt = ke0 * (ET - Ce)
        # Inputs are ET MAC fractions.
        self.ce_sevo += self.ke0_sevo * (mac_sevo - self.ce_sevo) * dt_min
        
        # --- 0.1 Volume Clearance ---
        # Reduce blood volume by clearance (urine, etc.)
        blood_volume = self.blood_volume - (self.vol_clearance * dt_min)
        if sepsis_sev > 0.0:
            # Capillary leak: plasma loss without direct RBC loss
            dt_hr = dt / 3600.0
            leak_fraction = self.sepsis_leak_fraction_per_hr * sepsis_sev
            leak_ml = blood_volume * leak_fraction * dt_hr
            blood_volume -= leak_ml
        self.blood_volume = max(500.0, blood_volume) # Safety floor
        self._update_hb_conc()

        # --- 1. Calculate Effects ---
        (total_eff_tpr, total_eff_sv, total_eff_hr_prod, 
         eff_remi_tpr, eff_remi_sv, eff_remi_hr) = self._calc_anesthetic_effects(ce_prop, ce_remi, self.ce_sevo)

        
        # =====================================================================
        # Physiological State Calculations
        # =====================================================================
        
        # --- 4.1 Blood Volume / MCFP / Preload ---
        # Calculate current MCFP from blood volume
        stressed_vol = self._calc_stressed_volume(sepsis_sev)
        mcfp = stressed_vol / self.cv
        
        # Preload factor from volume status
        f_preload_vol = mcfp / self.mcfp_0 if self.mcfp_0 > 0 else 1.0
        
        # --- 4.2 Intrathoracic Pressure (Pit) Effect on Preload ---
        # PEEP/PPV increases Pit, reducing venous return and preload
        delta_pit = pit - self.pit_0
        self.f_preload_pit = 1.0 / (1.0 + self.alpha_peep * max(0.0, delta_pit))
        
        # Combined preload factor; f_preload_pit is also reused in the state property.
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

        # --- 4.5 Sepsis / Distributive Shock ---
        sepsis_hr_mult = 1.0 + (self.sepsis_hr_increase / max(self.base_hr, 1.0)) * sepsis_sev
        sepsis_tpr_mult = 1.0 - (1.0 - self.sepsis_tpr_floor) * sepsis_sev
        
        # --- 4.6 Unified Vasopressor Effects ---
        # All vasopressors now use consistent helper architecture
        epi_delta_hr, epi_sv_factor, epi_svr_factor = self._calc_epi_effects(ce_epi)
        nore_delta_hr, nore_sv_factor, nore_svr_factor = self._calc_nore_effects(ce_nore)
        phenyl_svr_factor = self._calc_phenyl_effects(ce_phenyl)
        
        # Combine vasopressor effects
        # SVR: Multiplicative (each drug independently increases vascular resistance)
        combined_svr_factor = epi_svr_factor * nore_svr_factor * phenyl_svr_factor
        pressor_resistance = clamp01(self.sepsis_pressor_resistance * sepsis_sev)
        if pressor_resistance > 0:
            combined_svr_factor = 1.0 + (combined_svr_factor - 1.0) * (1.0 - pressor_resistance)
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
        term = 1.0 - self.hr_sv_coupling * math.log(max(1.0, current_hr / self.base_hr))
        term = max(0.1, term)  # Prevent negative SV at extreme HR
        raw_sv = (self.sv_star + self.tde_sv) * term
        current_sv = raw_sv * f_frank_starling
        
        # TPR is self.tpr; however, the baroreflex senses MAP, which includes drug effects.
        # Use dist_svr (argument) instead of self.dist_svr (stored) for lag-free response
        distributive_svr_drop = (self.sepsis_svr_drop_wood * sepsis_sev +
                                 self.anaphylaxis_svr_drop_wood * anaph_sev)
        distributive_tpr_offset = -distributive_svr_drop / 1000.0
        effective_tpr = self.tpr + self.delta_tpr_vasopressors + (dist_svr / 1000.0) + distributive_tpr_offset
        effective_tpr = max(0.006, effective_tpr)
        
        rmap = (current_hr * current_sv * effective_tpr) / (self.base_hr * self.base_sv * self.base_tpr)
        rmap = clamp(rmap, 0.1, 5.0)  # Clamp to prevent NaN from negative exponent
        rmap_fb = rmap ** self.fb
        
        # Derivatives with new physiological factors
        
        # Thermoregulation (Vasoconstriction)
        # Threshold 36.5 C. 
        # Gain: Increase TPR.
        # Inhibition: Propofol/Volatiles reduce threshold or gain.
        # Sessler. Anesthesiology. 2000 (review): anesthesia lowers vasoconstriction threshold.
        
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
        tpr_production = self.kin_tpr * rmap_fb * (1.0 + total_eff_tpr) * chemo_tpr_factor * self.hemorrhage_tpr_mult * thermo_tpr_mult * sepsis_tpr_mult
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
        hr_production = self.kin_hr * rmap_fb * (1.0 + total_eff_hr_prod) * self.hemorrhage_hr_mult * sepsis_hr_mult
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

        hr_base_for_state = self._calc_hr()
        return self._compute_state(
            preload_sv_factor=f_frank_starling,
            sepsis_sev=sepsis_sev,
            anaph_sev=anaph_sev,
            distributive_tpr_offset=distributive_tpr_offset,
            hr_base=hr_base_for_state,
        )

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
        # Solve f(Z) = Z_calc(Z) - Z = 0 where Z = RMAP
        
        # Calculate drug effects
        cn = max(0.0, ce_nore)
        
        # Use helper for combined anesthetic effects
        # For steady state, we assume ce_sevo = mac (equilibrium)
        (total_eff_tpr, total_eff_sv, total_eff_hr_prod, 
         eff_remi_tpr, eff_remi_sv, eff_remi_hr) = self._calc_anesthetic_effects(ce_prop, ce_remi, mac)

        
        # Norepinephrine effects
        nore_hill = hill_function(cn, self.nore_c50, self.nore_gamma)
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
            term = 1.0 - self.hr_sv_coupling * math.log(max(1.0, hr_z / self.base_hr))
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
        self.ce_sevo = mac

        # Final Output
        ret = self.step(0.0, ce_prop, ce_remi, ce_nore, -2.0, 40.0, 95.0, 0,0,0, 0.0, mac_sevo=mac)
        
        # Revert internal state
        self.tpr = saved_tpr
        self.sv_star = saved_sv
        self.hr_star = saved_hr
        self.tde_sv = saved_dist[0]
        self.tde_hr = saved_dist[1]
        self._cached_state = None
        
        return ret
