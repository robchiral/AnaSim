import numpy as np
from dataclasses import dataclass
from core.constants import (
    RR_APNEA_THRESHOLD, RR_BRADYPNEA_THRESHOLD, VT_MIN, TEMP_METABOLIC_COEFFICIENT
)
from patient.patient import Patient
from core.utils import hill_function, clamp01

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
    drive_central: float = 1.0
    muscle_factor: float = 1.0

class RespiratoryModel:
    """
    Respiratory depression model based on Propofol and Remifentanil.
    Models suppression of Respiratory Drive.
    """
    def __init__(self, patient: Patient):
        self.patient = patient
        self.rr_0 = patient.baseline_rr
        self.vt_0 = patient.baseline_vt
        self.baseline_hb = getattr(patient, 'baseline_hb', 13.5)
        
        # Drug Effect Parameters (literature-based values)
        # 
        # 1. Propofol - Separated effects for HCVR vs mechanical depression
        # Sarton et al. Anesthesiology 2001: Central CO2 sensitivity reduced ~40% at 1.3 µg/mL
        # JPET 2024: Propofol IC50 ≈ 1 µg/mL for HCVR slope suppression
        # Clinical sedation/apnea requires higher concentrations (~3-4 µg/mL)
        # 
        # HCVR/Central Drive: Early potent effect on CO2 sensitivity
        self.c50_prop_hcvr = 1.0  # µg/mL - IC50 for HCVR slope depression
        self.gamma_prop_hcvr = 2.0
        
        # Mechanical depression (VT/RR): Higher concentrations needed for frank apnea
        self.c50_prop_mech = 3.5  # µg/mL - EC50 for VT/RR mechanical suppression
        self.gamma_prop_mech = 2.0

        # 2. Remifentanil (Ventilatory Depression)
        # Bouillon TW et al. Anesthesiology 2003: C50 ~ 0.92 ng/mL, Hill ~1.25
        # Hannam JA et al. PAGE 2015: IC50 ~ 1.13 ng/mL
        self.c50_remi = 1.0  # ng/mL
        self.gamma_remi = 1.25
        
        # Remifentanil CO2 setpoint shift (Bouillon 2003)
        # Opioids shift apneic threshold rightward (increase PaCO2 required to trigger breathing)
        self.remi_setpoint_shift_max = 8.0  # mmHg - max rightward shift at full effect
        
        # 3. Sevoflurane (Central Drive)
        # Van den Elsen BJA 1998, Pandit BJA 1999: Even 0.1 MAC reduces HCVR
        # Dahan: ~50% depression at ~1 MAC for halogenated volatiles
        # Adjusted to 0.75 MAC for better alignment with human data
        self.c50_sevo_mac = 0.75
        self.gamma_sevo = 2.0
        
        # 4. NMBA (Muscle Strength Only)
        # Rocuronium: steep Hill curve (γ ~3-5), normalized scale
        self.c50_nmba = 0.6
        self.gamma_nmba = 4.0

        # =====================================================
        # Hypercapnic Ventilatory Response (HCVR) Parameters
        # =====================================================
        # Normal HCVR: Ventilation increases ~2-4 L/min per mmHg rise in PaCO2
        # Reference: Nunn's Respiratory Physiology, 8th ed; Duffin J. J Appl Physiol 2011
        
        # Baseline HCVR slope (L/min per mmHg above setpoint)
        self.hcvr_slope_baseline = 2.5
        
        # CO2 setpoint for ventilatory response (mmHg)
        self.paco2_setpoint = 40.0
        
        # Drug-specific HCVR depression weights (fraction of slope reduction at full effect)
        # Remifentanil: Most potent HCVR depressant
        # Bouillon TW et al. Anesthesiology 2003: Opioids shift CO2 curve right AND reduce slope by 50-70%
        self.hcvr_depression_remi = 0.70
        
        # Propofol: Moderate HCVR depression
        # Nieuwenhuijs DJ et al. Anesthesiology 2001: Propofol shifts apneic threshold but less effect on slope
        self.hcvr_depression_prop = 0.40
        
        # Sevoflurane: Moderate-strong HCVR depression
        # Dahan A et al. 1996 (likely Anesthesiology): ~50% slope reduction at 1 MAC
        self.hcvr_depression_sevo = 0.50

        # Differential Effects Weights (0.0 - 1.0)
        # Propofol: VT falls more than RR
        # Increased w_prop_rr from 0.3 to 0.6 for more realistic RR depression at surgical depth
        self.w_prop_rr = 0.6
        self.w_prop_vt = 1.0
        
        # Remifentanil: RR falls steeply, VT less so
        self.w_remi_rr = 1.0
        self.w_remi_vt = 0.5
        
        # Sevoflurane: Moderate on RR, Strong on VT
        self.w_sevo_rr = 0.4
        self.w_sevo_vt = 0.8
        
        self.state = RespState(self.rr_0, self.vt_0, (self.rr_0 * self.vt_0)/1000.0)
        
        # CO2 Params
        self.vco2 = 200.0 # mL/min - Metabolic production
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
        # CO2: Large body stores (~120L) equilibrate slowly (Nunn's Respiratory Physiology)
        # During apnea: PaCO2 rises 3-5 mmHg/min clinically
        # Tau = 3 min gives realistic apnea CO2 rise rate
        self.tau_co2 = 180.0 # Time constant for CO2 (s) - was 45s, too fast
        self.tau_o2 = 15.0 # Time constant for O2 (s) - small O2 stores equilibrate quickly
        self.rq = 0.8
        self.atm_p = 760.0
        self.vapor_p = 47.0
        self.aa_grad = 10.0 # A-a gradient (mmHg)

    def step(self, dt: float, ce_prop: float, ce_remi: float, mech_vent_mv: float = 0.0, 
             fio2: float = 0.21, ce_roc: float = 0.0, et_sevo: float = 0.0, mac_sevo: float = None,
             peep: float = 0.0, mean_paw: float = 5.0, temp_c: float = 37.0,
             mech_rr: float = 0.0, mech_vt_l: float = 0.0,
             hb_g_dl: float = None, oxygen_delivery_ratio: float = 1.0) -> RespState:
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
        """
        
        # 1. Calculate MAC fraction for Sevoflurane
        # If mac_sevo is supplied (effect-site), use it directly; otherwise derive from ET.
        if mac_sevo is None:
            # Age-adjusted MAC (MapTanner formula)
            # MapTanner formula: MAC_age = MAC_40 * 10^(-0.00269 * (age - 40))
            # Sevo MAC_40 ~ 2.1%
            mac_40 = 2.1
            mac_age = mac_40 * (10 ** (-0.00269 * (self.patient.age - 40)))
            mac_sevo = et_sevo / mac_age

        # 2. Calculate fractional inhibition (0 to 1) for each drug
        # Propofol: separate effects for HCVR (central) vs mechanical (RR/VT)
        eff_prop_hcvr = hill_function(ce_prop, self.c50_prop_hcvr, self.gamma_prop_hcvr)
        eff_prop_mech = hill_function(ce_prop, self.c50_prop_mech, self.gamma_prop_mech)
        
        eff_remi = hill_function(ce_remi, self.c50_remi, self.gamma_remi)
        eff_sevo = hill_function(mac_sevo, self.c50_sevo_mac, self.gamma_sevo)
        eff_nmba = hill_function(ce_roc, self.c50_nmba, self.gamma_nmba)

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
        # References:
        # - Bouillon TW et al. Anesthesiology 2003: Opioid depression of CO2 response
        # - Dahan A et al. 1996: Volatile depression of CO2 response
        # - Nunn's Respiratory Physiology: Normal HCVR ~2-4 L/min/mmHg
        
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
        co2_above_setpoint = max(0.0, self.state.p_alveolar_co2 - effective_setpoint)
        
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
        rr_inhib = clamp01(rr_inhib)
        vt_inhib = clamp01(vt_inhib)

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
            self.state.apnea = True
        elif current_rr < RR_BRADYPNEA_THRESHOLD:
            # Bradypnea range: reduce effective RR and mark as partial apnea
            # Patient takes occasional gasping breaths but inadequate ventilation
            current_rr = current_rr * 0.5  # Effective RR even lower due to irregularity
            self.state.apnea = False  # Not complete apnea but inadequate
        else:
            self.state.apnea = False
            
        current_vt = self.vt_0 * vt_fraction
        if current_vt < VT_MIN: current_vt = 0.0
        
        # Couple observed RR to effective ventilation
        # If VT too low for capnography detection, observed RR = 0
        # Threshold: 100 mL (below which waveform is unreliable)
        if current_vt < 100.0:
            current_rr = 0.0
            self.state.apnea = True
        
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
        alveolar_vt_mech = max(0.0, mech_vt_l - vd)
        if mech_vt_l <= 0 and mech_vent_mv > 0 and mech_rr > 0:
            inferred_vt = mech_vent_mv / mech_rr
            alveolar_vt_mech = max(0.0, inferred_vt - vd)
        va_mech_eff = max(0.0, mech_rr) * alveolar_vt_mech
        
        # Total VA = synchronized mechanical + spontaneous
        # Fix: Prevent double counting in AC/SIMV modes.
        # If mechanical ventilation is active, it typically augments or replaces spontaneous breaths.
        # We model this as the patient getting the "better" of the two ventilations per cycle.
        
        # Effective RR for gas exchange
        ref_rr_mech = max(0.0, mech_rr)
        ref_rr_spont = max(0.0, current_rr)
        
        # In synchronized mode, the rate is the max of set rate and patient rate
        # If vent is off (mech_rr=0), effective is spont
        effective_rate = max(ref_rr_mech, ref_rr_spont)
        
        # Effective Alveolar Vt per breath
        # If patient triggers the vent, use supported Vt.
        # We approximate effective Vt as the larger of the two to represent support.
        vt_mech_avail = max(0.0, alveolar_vt_mech)
        vt_spont_avail = max(0.0, vt_eff_spont)
        
        if mech_rr > 0:
             # Vent active: Patient gets supported breaths
             effective_vt_alv = max(vt_mech_avail, vt_spont_avail)
        else:
             effective_vt_alv = vt_spont_avail
             
        total_va_l_min = effective_rate * effective_vt_alv
        self.state.va = total_va_l_min
        
        # 9. CO2 Dynamics
        # PaCO2 approaches equilibrium based on VA and metabolic rate
        if self.va_baseline <= 0:
            self.va_baseline = 4.2  # Fallback baseline VA (L/min)
        
        # Prevent division by zero
        effective_va = max(0.1, total_va_l_min)
        
        # PaCO2 Equilibrium: PaCO2_eq = PaCO2_base * (VA_base / VA) * metabolic_factor
        # Temperature effect: VCO2 decreases ~7% per °C below 37°C (Q10 ≈ 2.0)
        metabolic_factor = TEMP_METABOLIC_COEFFICIENT ** (37.0 - temp_c)
        
        paco2_base = 40.0
        paco2_eq = paco2_base * metabolic_factor * (self.va_baseline / effective_va)
        
        # Physiological ceiling (severe respiratory failure)
        paco2_eq = min(150.0, paco2_eq)
        
        # Exponential approach to equilibrium: dPaCO2 = (Target - Current) / Tau
        d_paco2 = (paco2_eq - self.state.p_alveolar_co2) / self.tau_co2 * dt
        
        # Apneic rise rate clamp: clinical rate is 3-5 mmHg/min during anesthesia
        # (Brain-death testing / anesthesia literature supports 3 mmHg/min)
        # Only clamp rising values - CO2 washout during hyperventilation can be rapid
        if d_paco2 > 0:
            max_rise_rate = 0.05 * dt  # 3 mmHg/min
            d_paco2 = min(d_paco2, max_rise_rate)
            
        self.state.p_alveolar_co2 += d_paco2
        
        # EtCO2 approximates PaCO2 (small gradient in healthy lungs)
        self.state.etco2 = self.state.p_alveolar_co2
        
        # 10. O2 Dynamics (Alveolar Gas Equation)
        # PAO2 = FiO2 * (Patm - PH2O) - PaCO2 / RQ
        p_ideal_alveolar_o2 = fio2 * (self.atm_p - self.vapor_p) - (self.state.p_alveolar_co2 / self.rq)
        
        # A-a gradient with PEEP recruitment effect
        # PEEP recruits alveoli, improving V/Q matching and reducing A-a gradient
        # Effect: A-a_eff = A-a_base / (1 + k * PEEP), floor at 3 mmHg
        k_peep_recruit = 0.08
        aa_grad_effective = self.aa_grad / (1.0 + k_peep_recruit * peep)
        aa_grad_effective = max(3.0, aa_grad_effective)
        
        pao2_target = max(20.0, p_ideal_alveolar_o2 - aa_grad_effective)
        
        # Note: Anemia and low cardiac output do NOT reduce PaO2 directly.
        # They reduce oxygen CONTENT (CaO2 = 1.34 × Hb × SaO2 + 0.003 × PaO2)
        # and delivery (DO2 = CaO2 × CO), but PaO2 depends only on:
        # FiO2, PaCO2, V/Q matching, and A-a gradient.
        
        # Exponential approach to equilibrium
        d_pao2 = (pao2_target - self.state.p_arterial_o2) / self.tau_o2 * dt
        self.state.p_arterial_o2 += d_pao2
        
        self.state.drive_central = drive_central
        self.state.muscle_factor = muscle_factor
        
        self.state.rr = current_rr
        self.state.vt = current_vt
        self.state.mv = (current_rr * current_vt) / 1000.0
        
        return self.state
