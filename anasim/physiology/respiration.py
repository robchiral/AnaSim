import math
from dataclasses import dataclass
from anasim.core.constants import (
    RR_APNEA_THRESHOLD, RR_BRADYPNEA_THRESHOLD, VT_MIN, TEMP_METABOLIC_COEFFICIENT,
    SHIVER_MAX_MULTIPLIER
)
from anasim.patient.patient import Patient
from anasim.core.utils import hill_function, clamp01

@dataclass
class RespState:
    rr: float = 12.0 # Respiratory Rate
    vt: float = 500.0 # Tidal Volume
    mv: float = 6.0 # Minute Ventilation (L/min)
    va: float = 4.0 # Alveolar ventilation (L/min)
    apnea: bool = False
    p_alveolar_co2: float = 40.0 # mmHg
    etco2: float = 40.0 # mmHg
    p_arterial_o2: float = 95.0 # mmHg (PaO2)
    sao2: float = 98.0 # Arterial oxygen saturation (%), perfusion-adjusted
    drive_central: float = 1.0
    muscle_factor: float = 1.0

class RespiratoryModel:
    """
    Respiratory depression model based on Propofol and Remifentanil.
    Models suppression of Respiratory Drive.
    """
    def __init__(self, patient: Patient, fidelity_mode: str = "clinical"):
        self.patient = patient
        self.rr_0 = patient.baseline_rr
        self.vt_0 = patient.baseline_vt
        self.baseline_hb = getattr(patient, 'baseline_hb', 13.5)
        # Drug Effect Parameters (literature-anchored where available, otherwise heuristic)
        # 
        # 1. Propofol - Separated effects for HCVR vs mechanical depression
        # Propofol depresses ventilatory drive (Blouin et al. Anesthesiology. 1993);
        # HCVR EC50 is not well established; use a proxy aligned to
        # Nieuwenhuijs et al. Anesthesiology. 2001.
        self.c50_prop_hcvr = 2.0
        self.gamma_prop_hcvr = 2.0
        
        # Mechanical depression (VT/RR): Higher concentrations needed.
        # Effect-site EC50 ~4.0 mcg/mL for respiratory depression (Lee et al. 2011).
        self.c50_prop_mech = 4.0
        self.gamma_prop_mech = 2.0

        # 2. Remifentanil (Ventilatory Depression)
        # Remifentanil ventilatory depression: EC50 ~1.1-1.2 ng/mL with steep slope
        # in CO2 response studies (Glass et al. 1999; Babenco et al. 2000).
        self.c50_remi = 1.2
        self.gamma_remi = 1.7
        
        # Remifentanil CO2 setpoint shift (Babenco et al. Anesthesiology. 2000)
        self.remi_setpoint_shift_max = 8.0  # mmHg
        
        # 3. Sevoflurane (Central Drive)
        # Low-dose (0.1 MAC) sevo shows minimal CO2-response change (Pandit et al. 1999, BJA).
        # Doi & Ikeda. Anesth Analg. 1987: CO2 response depressed at 1.1-1.4 MAC.
        # Magnitude at 1 MAC is heuristic (interpolated between low-MAC and 1.1-1.4 MAC).
        self.c50_sevo_mac = 1.1
        self.gamma_sevo = 2.0
        
        # 4. NMBA (Muscle Strength Only)
        # Rocuronium: steep Hill curve (γ ~3-5), normalized scale
        self.c50_nmba = 0.6
        self.gamma_nmba = 4.0

        # =====================================================
        # Hypercapnic Ventilatory Response (HCVR) Parameters
        # =====================================================
        # Normal HCVR: Ventilation increases ~2-3 L/min per mmHg rise in PaCO2
        # (dynamic end-tidal forcing studies: Nieuwenhuijs 2001; Pandit 1999).
        
        # Baseline HCVR slope (L/min per mmHg above setpoint)
        self.hcvr_slope_baseline = 2.2
        
        # CO2 setpoint for ventilatory response (mmHg)
        self.paco2_setpoint = 40.0
        
        # Drug-specific HCVR depression weights (fraction of slope reduction at full effect)
        # Remifentanil: Most potent HCVR depressant
        # Babenco et al. Anesthesiology. 2000: opioids shift CO2 curve right and reduce slope.
        self.hcvr_depression_remi = 0.70
        
        # Propofol: Moderate HCVR depression
        # Nieuwenhuijs et al. Anesthesiology. 2001: propofol shifts apneic threshold with modest slope change.
        self.hcvr_depression_prop = 0.40
        
        # Sevoflurane: Moderate-strong HCVR depression
        # Low-MAC effect is small (Pandit 1999); 1 MAC slope reduction remains heuristic
        # using Doi & Ikeda 1987 (1.1-1.4 MAC) as an upper anchor.
        self.hcvr_depression_sevo = 0.50

        # Differential Effects Weights (0.0 - 1.0)
        # Propofol: VT falls more than RR
        # Increased w_prop_rr for more realistic RR depression at surgical depth (heuristic).
        self.w_prop_rr = 0.6
        self.w_prop_vt = 0.8
        
        # Remifentanil: RR falls steeply, VT less so
        self.w_remi_rr = 1.0
        self.w_remi_vt = 0.35
        
        # Sevoflurane: Moderate on RR, Strong on VT
        self.w_sevo_rr = 0.4
        self.w_sevo_vt = 0.8
        
        self.state = RespState(self.rr_0, self.vt_0, (self.rr_0 * self.vt_0)/1000.0)
        
        # Respiratory quotient for gas exchange coupling.
        self.rq = 0.8

        # CO2 Params
        # Resting VO2 ~3.6 mL/kg/min in large adult CPET cohorts; VCO2 ≈ VO2 * RQ.
        # (Thorax 2021; CPET reference values).
        self.vo2_ml_kg_min = 3.6
        self.vco2 = self.vo2_ml_kg_min * patient.weight * self.rq  # mL/min
        self.frc = 2.5 # L - Functional Residual Capacity (Alveolar volume buffer)
        
        # Initialize PACO2 at 40 mmHg
        self.state.p_alveolar_co2 = 40.0
        self.state.etco2 = 40.0
        self.state.p_arterial_o2 = 95.0
        
        # Gas Exchange Parameters
        # Deadspace ~ 2.2 mL/kg
        self.vd_deadspace = 2.2 * patient.weight / 1000.0 # L
        self.va_baseline = max(0.1, (self.vt_0/1000.0 - self.vd_deadspace) * self.rr_0) # Baseline Alveolar Vent (L/min)

        # Time constants for gas equilibration
        # CO2: Large body stores (~120L) equilibrate slowly
        # During apnea: PaCO2 rises 3-5 mmHg/min clinically
        # Tau = 3 min gives realistic apnea CO2 rise rate
        self.tau_co2 = 180.0 # Time constant for CO2 (s) - was 45s, too fast
        self.tau_o2 = 15.0 # Time constant for O2 (s) - small O2 stores equilibrate quickly
        self.atm_p = 760.0
        self.vapor_p = 47.0
        # Age-adjusted A-a gradient (mmHg): ~age/4 + 4 (PIOPED/Chest 1995).
        self.aa_grad_base = max(5.0, (self.patient.age / 4.0) + 4.0)

        # Age-adjusted MAC for Sevo (MapTanner formula)
        # MAC_40 ~ 2.1%
        self._mac_sevo_age = 2.1 * (10 ** (-0.00269 * (self.patient.age - 40)))

        # Cache for temperature-dependent metabolic factor
        # Temperature changes slowly (thermal time constants ~minutes), so caching
        # avoids exponential calculation on every step
        self._cached_temp = 37.0
        self._cached_metabolic_factor = 1.0
        self._arrest_desat_time = 0.0
        # Cached SaO2 curve constants (Hill equation)
        self._p50 = 26.6
        self._n_hill = 2.7

    def step(self, dt: float, ce_prop: float, ce_remi: float, mech_vent_mv: float = 0.0, 
             fio2: float = 0.21, ce_roc: float = 0.0, et_sevo: float = 0.0, mac_sevo: float = None,
             peep: float = 0.0, mean_paw: float = 5.0, temp_c: float = 37.0,
             mech_rr: float = 0.0, mech_vt_l: float = 0.0,
             airway_patency: float = 1.0, ventilation_efficiency: float = 1.0,
             vq_mismatch: float = 0.0,
             hb_g_dl: float = None, oxygen_delivery_ratio: float = 1.0,
             shiver_level: float = 0.0,
             cardiac_output: float = 5.0) -> RespState:
        """
        Step respiration.
        
        Args:
            dt: Time step (seconds)
            ce_prop: Propofol effect-site concentration (ug/mL)
            ce_remi: Remifentanil effect-site concentration (ng/mL)
            mech_vent_mv: Minute ventilation provided by mechanical ventilator (L/min)
            fio2: Fraction of inspired oxygen (0.0-1.0)
            ce_roc: Rocuronium effect-site concentration (ug/mL)
            et_sevo: End-tidal sevoflurane (percentage)
            peep: Set PEEP level (cmH2O) - affects oxygenation
            mean_paw: Mean airway pressure (cmH2O) - affects hemodynamics
            temp_c: Patient body temperature (deg C)
            airway_patency: 0-1 upper airway patency (upper obstruction)
            ventilation_efficiency: 0-1 gas exchange efficiency (bronchospasm)
            vq_mismatch: 0-1 V/Q mismatch severity (affects A-a gradient)
            shiver_level: 0-1 shivering intensity (metabolic multiplier)
        """
        state = self.state
        hill = hill_function
        clamp01_local = clamp01

        # 1. Calculate MAC fraction for Sevoflurane
        # If mac_sevo is supplied (effect-site), use it directly; otherwise derive from ET.
        if mac_sevo is None:
            mac_sevo = et_sevo / self._mac_sevo_age

        # 2. Calculate fractional inhibition (0 to 1) for each drug
        # Propofol: separate effects for HCVR (central) vs mechanical (RR/VT)
        eff_prop_hcvr = hill(ce_prop, self.c50_prop_hcvr, self.gamma_prop_hcvr)
        eff_prop_mech = hill(ce_prop, self.c50_prop_mech, self.gamma_prop_mech)
        
        eff_remi = hill(ce_remi, self.c50_remi, self.gamma_remi)
        eff_sevo = hill(mac_sevo, self.c50_sevo_mac, self.gamma_sevo)
        eff_nmba = hill(ce_roc, self.c50_nmba, self.gamma_nmba)

        # 3. Calculate Central Drive (Brain's desire to breathe)
        # Multiplicative interaction -> Synergy
        # Use eff_prop_hcvr for central drive (early potent effect)
        drive_central = (1.0 - eff_prop_hcvr) * (1.0 - eff_remi) * (1.0 - eff_sevo)
        
        # =====================================================
        # 3a. Hypercapnic Ventilatory Response (HCVR)
        # =====================================================
        # Rising PaCO2 stimulates increased ventilation via central chemoreceptors.
        # Anesthetics depress this response (shift curve right, reduce slope).
        #
        # Model: VA_boost = HCVR_slope × (PaCO2 - setpoint)
        #        HCVR_slope = baseline × (1 - drug_depression)
        #
        # Reference: Babenco et al. Anesthesiology. 2000 (opioids),
        # Nieuwenhuijs et al. Anesthesiology. 2001 (propofol CO2 response),
        # Pandit et al. Br J Anaesth. 1999 (0.1 MAC sevo),
        # Doi & Ikeda. Anesth Analg. 1987 (1.1-1.4 MAC).
        
        # Calculate drug-induced HCVR depression
        # Use multiplicative interaction to avoid saturation artifacts
        # Slope factor = Product of (1 - drug_effect * potency)
        
        factor_remi = max(0.0, 1.0 - self.hcvr_depression_remi * eff_remi)
        factor_prop = max(0.0, 1.0 - self.hcvr_depression_prop * eff_prop_hcvr)
        factor_sevo = max(0.0, 1.0 - self.hcvr_depression_sevo * eff_sevo)

        slope_factor = factor_remi * factor_prop * factor_sevo

        # Current HCVR slope
        hcvr_slope = self.hcvr_slope_baseline * slope_factor
        
        # CO2-driven ventilatory boost (only when PaCO2 > setpoint)
        # Opioids shift setpoint rightward (require higher CO2 to trigger breathing)
        effective_setpoint = self.paco2_setpoint + (self.remi_setpoint_shift_max * eff_remi)
        co2_above_setpoint = max(0.0, state.p_alveolar_co2 - effective_setpoint)
        
        # Calculate boost in alveolar ventilation (L/min)
        va_boost_from_co2 = hcvr_slope * co2_above_setpoint
        
        # Convert VA boost to drive boost (normalized to baseline VA)
        # At baseline (VA ~4.2 L/min), a 4.2 L/min boost = 100% drive increase
        if self.va_baseline > 0:
            co2_drive_boost = va_boost_from_co2 / self.va_baseline
        else:
            co2_drive_boost = 0.0
        
        # Augment central drive with HCVR (additive, capped at 200% of baseline)
        drive_central = min(2.0, drive_central + co2_drive_boost)
        
        # 4. Calculate Muscle Factor (Ability to execute breath)
        # NMBA only affects mechanics, not central drive
        muscle_factor = 1.0 - eff_nmba

        # 5. Calculate Component Inhibitions (RR vs VT)
        # Weighted sum of effects for specific dimensions
        # Use eff_prop_mech for mechanical (RR/VT) depression
        rr_inhib_base = (self.w_prop_rr * eff_prop_mech + 
                         self.w_remi_rr * eff_remi + 
                         self.w_sevo_rr * eff_sevo)
                    
        vt_inhib_base = (self.w_prop_vt * eff_prop_mech + 
                         self.w_remi_vt * eff_remi + 
                         self.w_sevo_vt * eff_sevo)

        # HCVR counteracts drug inhibition
        # At high CO2, patient "fights through" depression
        # Reduce effective inhibition proportionally to CO2 drive boost
        hcvr_counteraction = min(0.5, co2_drive_boost * 0.3)  # Max 50% counteraction
        
        rr_inhib = rr_inhib_base * (1.0 - hcvr_counteraction)
        vt_inhib = vt_inhib_base * (1.0 - hcvr_counteraction)

        # Clamp inhibitions
        rr_inhib = clamp01_local(rr_inhib)
        vt_inhib = clamp01_local(vt_inhib)

        rr_fraction = 1.0 - rr_inhib
        vt_fraction = 1.0 - vt_inhib
        
        # Apply Muscle Factor to both VT and RR
        # With complete NMB, patient cannot generate any effective breaths
        vt_fraction *= muscle_factor
        rr_fraction *= muscle_factor

        # 6. Apply to Baseline
        # Allow Hyperventilation: If drive_central > 1.0, boost RR above baseline
        if drive_central > 1.0:
            # Distribute excess drive between RR and VT
            # e.g. 60% of excess drive goes to RR boost
            excess_drive = drive_central - 1.0
            rr_boost = 1.0 + (excess_drive * 0.6)
            rr_fraction = rr_fraction * rr_boost

        current_rr = self.rr_0 * rr_fraction
        
        # Gradual Apnea Transition (instead of hard cutoff)
        # - RR < APNEA: Complete apnea
        # - RR APNEA-BRADYPNEA: Severe bradypnea with irregular, ineffective breathing
        # - RR >= BRADYPNEA: Normal spontaneous breathing
        if current_rr < RR_APNEA_THRESHOLD:
            current_rr = 0.0
            state.apnea = True
        elif current_rr < RR_BRADYPNEA_THRESHOLD:
            # Bradypnea range: reduce effective RR and mark as partial apnea
            # Respiratory drive produces occasional gasping breaths with inadequate ventilation.
            current_rr = current_rr * 0.5  # Effective RR even lower due to irregularity
            state.apnea = False  # Represents inadequate ventilation rather than complete apnea.
        else:
            state.apnea = False
            
        current_vt = self.vt_0 * vt_fraction
        if current_vt < VT_MIN: current_vt = 0.0

        # Apply airway obstruction / ventilation efficiency to spontaneous VT
        airway_patency = clamp01_local(airway_patency)
        ventilation_efficiency = clamp01_local(ventilation_efficiency)
        vent_factor = airway_patency * ventilation_efficiency
        current_vt *= vent_factor
        
        # Couple observed RR to effective ventilation
        # If VT too low for capnography detection, observed RR = 0
        # Threshold: 100 mL (below which waveform is unreliable)
        if current_vt < 100.0:
            current_rr = 0.0
            state.apnea = True
        
        # 7. Curare Cleft (Patient-Ventilator Dyssynchrony)
        # The curare_cleft flag logic is handled in the monitor/disturbance layer
        
        # 8. Calculate Alveolar Ventilation
        # Dead space (anatomical, ~2.2 mL/kg)
        vd = self.vd_deadspace
        
        # Spontaneous Alveolar Ventilation: VA = RR * (VT - VD)
        vt_eff_spont = max(0.0, current_vt / 1000.0 - vd)
        va_spont = current_rr * vt_eff_spont  # L/min
        
        # Mechanical Alveolar Ventilation
        # Use delivered Vt from ventilator mechanics if available.
        mech_vt_l *= vent_factor
        mech_vent_mv *= vent_factor
        alveolar_vt_mech = max(0.0, mech_vt_l - vd)
        if mech_vt_l <= 0 and mech_vent_mv > 0 and mech_rr > 0:
            inferred_vt = mech_vent_mv / mech_rr
            alveolar_vt_mech = max(0.0, inferred_vt - vd)
        va_mech_eff = max(0.0, mech_rr) * alveolar_vt_mech
        
        # Total VA = synchronized mechanical + spontaneous
        # Fix: Prevent double counting in AC/SIMV modes.
        # If mechanical ventilation is active, it typically augments or replaces spontaneous breaths.
        # Model this as the patient receiving the more effective of the two ventilations per cycle.
        
        # Effective RR for gas exchange
        ref_rr_mech = max(0.0, mech_rr)
        ref_rr_spont = max(0.0, current_rr)
        
        # In synchronized mode, the rate is the max of set rate and patient rate
        # If vent is off (mech_rr=0), effective is spont
        effective_rate = max(ref_rr_mech, ref_rr_spont)
        
        # Effective Alveolar Vt per breath
        # If patient triggers the vent, use supported Vt.
        # Effective Vt is approximated as the larger of the two values to represent support.
        vt_mech_avail = max(0.0, alveolar_vt_mech)
        vt_spont_avail = max(0.0, vt_eff_spont)
        
        if mech_rr > 0:
             # Vent active: Patient gets supported breaths
             effective_vt_alv = max(vt_mech_avail, vt_spont_avail)
        else:
             effective_vt_alv = vt_spont_avail
             
        total_va_l_min = effective_rate * effective_vt_alv
        state.va = total_va_l_min
        
        # 9. CO2 Dynamics
        # PaCO2 approaches equilibrium based on VA and metabolic rate
        va_baseline = self.va_baseline
        if va_baseline <= 0:
            va_baseline = 4.2  # Fallback baseline VA (L/min)
            self.va_baseline = va_baseline
        
        # Prevent division by zero; V/Q mismatch reduces effective CO2 elimination.
        vq_mismatch = clamp01_local(vq_mismatch)
        effective_va = max(0.1, total_va_l_min * (1.0 - 0.6 * vq_mismatch))
        
        # PaCO2 Equilibrium: PaCO2_eq = PaCO2_base * (VA_base / VA) * metabolic_factor
        # Temperature effect: VCO2 decreases ~7% per °C below 37°C (Q10 ≈ 2.0)
        # Use cached value - temperature changes slowly (thermal time constants ~minutes)
        if abs(temp_c - self._cached_temp) > 0.01:
            self._cached_temp = temp_c
            self._cached_metabolic_factor = TEMP_METABOLIC_COEFFICIENT ** (37.0 - temp_c)
        metabolic_factor = self._cached_metabolic_factor
        shiver_mult = 1.0 + SHIVER_MAX_MULTIPLIER * clamp01_local(shiver_level)
        metabolic_factor *= shiver_mult
        
        paco2_base = 40.0
        paco2_eq = paco2_base * metabolic_factor * (va_baseline / effective_va)
        
        # Physiological ceiling (severe respiratory failure)
        paco2_eq = min(150.0, paco2_eq)
        
        # Exponential approach to equilibrium: dPaCO2 = (Target - Current) / Tau
        d_paco2 = (paco2_eq - state.p_alveolar_co2) / self.tau_co2 * dt
        
        # Apneic rise rate clamp: clinical rate is ~3-4.5 mmHg/min during anesthesia
        # (Respiratory Acidosis tests; Anesthesiology 2011 apnea studies).
        # Only clamp rising values - CO2 washout during hyperventilation can be rapid
        if d_paco2 > 0:
            max_rise_rate = 0.06 * dt  # 3.6 mmHg/min
            d_paco2 = min(d_paco2, max_rise_rate)
            
        state.p_alveolar_co2 += d_paco2
        
        # EtCO2 is slightly lower than PaCO2; gradient increases with deadspace and
        # obstructive physiology (Russell 1990; Lujan 2008).
        if mech_rr > 0:
            vt_for_gradient_l = max(mech_vt_l, current_vt / 1000.0)
        else:
            vt_for_gradient_l = current_vt / 1000.0
        vt_l = max(0.05, vt_for_gradient_l)
        vd_vt = min(0.95, self.vd_deadspace / vt_l)
        vd_vt_excess = max(0.0, vd_vt - 0.30)
        etco2_gradient = 4.0 + 15.0 * vd_vt_excess + 8.0 * vq_mismatch + 6.0 * (1.0 - ventilation_efficiency)
        etco2_gradient = min(20.0, etco2_gradient)
        state.etco2 = max(0.0, state.p_alveolar_co2 - etco2_gradient)
        
        # 10. O2 Dynamics (Alveolar Gas Equation)
        # PAO2 = FiO2 * (Patm - PH2O) - PaCO2 / RQ
        p_ideal_alveolar_o2 = fio2 * (self.atm_p - self.vapor_p) - (state.p_alveolar_co2 / self.rq)
        
        # A-a gradient with PEEP recruitment effect
        # PEEP recruits alveoli, improving V/Q matching and reducing A-a gradient
        # Effect: A-a_eff = A-a_base / (1 + k * PEEP), floor at 3 mmHg
        k_peep_recruit = 0.08
        aa_grad_effective = self.aa_grad_base / (1.0 + k_peep_recruit * peep)
        aa_grad_effective = max(3.0, aa_grad_effective)

        # V/Q mismatch (bronchospasm/obstruction) increases A-a gradient
        vq_mismatch = clamp01_local(vq_mismatch)
        aa_grad_effective *= (1.0 + 2.5 * vq_mismatch)
        aa_grad_effective = min(80.0, aa_grad_effective)
        
        pao2_target = max(20.0, p_ideal_alveolar_o2 - aa_grad_effective)
        
        # Note: Anemia and low cardiac output do NOT reduce PaO2 directly.
        # They reduce oxygen CONTENT (CaO2 = 1.34 × Hb × SaO2 + 0.003 × PaO2)
        # and delivery (DO2 = CaO2 × CO); however, PaO2 depends only on:
        # FiO2, PaCO2, V/Q matching, and A-a gradient.
        
        # Exponential approach to equilibrium
        d_pao2 = (pao2_target - state.p_arterial_o2) / self.tau_o2 * dt
        state.p_arterial_o2 += d_pao2
        
        state.drive_central = drive_central
        state.muscle_factor = muscle_factor
        
        state.rr = current_rr
        state.vt = current_vt
        state.mv = (current_rr * current_vt) / 1000.0
        
        # 11. Perfusion-Adjusted SaO2
        # PaO2-based SaO2 (Hill equation for oxyhemoglobin dissociation)
        # SaO2 = PaO2^n / (PaO2^n + P50^n), P50 ~ 26.6 mmHg, n ~ 2.7
        pao2_safe = max(0.1, state.p_arterial_o2)
        p50 = self._p50
        n_hill = self._n_hill
        base_sao2 = 100.0 * (pao2_safe ** n_hill) / (pao2_safe ** n_hill + p50 ** n_hill)
        
        # Cardiac arrest desaturation: When CO < 0.5 L/min, tissue O2 stores deplete
        # Clinical: SpO2 drops from ~100% to 60% in 60-90 seconds during arrest
        if cardiac_output < 0.5:
            # Track cumulative arrest time
            self._arrest_desat_time += dt
            
            # Desaturation: exponential decay toward severe hypoxia
            # Time constant ~30s means 63% drop in 30s, ~95% drop in 90s
            tau_desat = 30.0
            desat_fraction = 1.0 - (1.0 - 0.4) * (1.0 - math.exp(-self._arrest_desat_time / tau_desat))
            state.sao2 = base_sao2 * desat_fraction
        else:
            # Reset arrest timer and use normal SaO2
            self._arrest_desat_time = 0.0
            state.sao2 = base_sao2
        
        # Floor at 40% (pulse ox artifact threshold)
        state.sao2 = max(40.0, state.sao2)
        
        return state
