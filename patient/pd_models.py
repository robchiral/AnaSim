import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from .patient import Patient
from core.utils import hill_function, clamp, clamp01

# =============================================================================
# REMIFENTANIL UNIT CONVENTIONS
# =============================================================================
#
# This codebase uses two representations for remifentanil:
#
# 1. ce_remi: Effect-site concentration in ng/mL
#    - Output from the Minto PK model
#    - Used in Bouillon/Eleveld PD interaction models
#    - Typical clinical range: 1-8 ng/mL
#
# 2. remi_rate_ug_kg_min: Infusion rate in µg/kg/min  
#    - Input to TCI controller and engine
#    - Used in MAC-BIS models for volatile interactions
#    - Typical clinical range: 0.05-0.3 µg/kg/min
#
# Conversion (Minto steady-state approximation):
#   At steady state, Ce_remi ≈ 25 × Rate (for typical 70kg patient)
#   Example: 0.1 µg/kg/min → Ce ≈ 2.5 ng/mL
#            0.2 µg/kg/min → Ce ≈ 5.0 ng/mL
#
#   Therefore: Rate ≈ Ce_remi × 0.04 (inverse of 25)
#
# Reference: Minto CF et al. Anesthesiology 1997;86(1):10-23
# =============================================================================

# -----------------------------------------------------------------------------
# Volatile BIS Model (Evidence-Based from Published Literature)
# -----------------------------------------------------------------------------
#
# Core equation: BIS as function of effective MAC (M_eff)
# BIS_base(M_eff) = 35 + 60 / (1 + (M_eff / 0.31)^1.45)
#
# Key validation points:
# - M_eff = 0.0  → BIS ≈ 95 (awake)
# - M_eff = 0.5  → BIS ≈ 55
# - M_eff = 0.66 → BIS ≈ 50 (hypnotic threshold)
# - M_eff = 1.0  → BIS ≈ 44
# - M_eff = 1.5  → BIS ≈ 40
#
# Agent-specific offsets based on Kim TK et al. Anesthesiology 2005 (Sevoflurane/Remifentanil interaction):
# - Sevoflurane: BIS ≈ 41.5 (reference, offset = 0)
#
# Drug interactions (additive in MAC-equivalent space):
# - Remifentanil: ΔM_eff = 0.44 × (R / 0.2) where R in µg/kg/min
#   At R = 0.2 µg/kg/min: adds ~0.44 MAC equivalent
# - Propofol: ΔM_eff = 0.18 × C_p where C_p in µg/mL
#   At C_p = 3.7 µg/mL: adds ~0.66 MAC equivalent (BIS ~50)
#
# -----------------------------------------------------------------------------

def compute_bis_from_mac(m_eff: float, agent_offset: float = 0.0) -> float:
    """
    Core BIS-MAC relationship based on published human data.

    Calibrated to Schumacher PM et al. Anesthesiology 2003 studies.

    Args:
        m_eff: Effective MAC (age-adjusted, including drug interactions)
        agent_offset: Agent-specific BIS offset (Sevo: 0)

    Returns:
        BIS value (30-98)

    Validation points (Sevoflurane reference):
        M_eff = 0.0  → BIS ≈ 95 (awake)
        M_eff = 0.5  → BIS ≈ 55 (light anesthesia)
        M_eff = 0.66 → BIS ≈ 50 (hypnotic threshold)
        M_eff = 1.0  → BIS ≈ 44 (surgical depth)
        M_eff = 1.5  → BIS ≈ 40 (deep anesthesia)
    """
    if m_eff < 0:
        m_eff = 0.0

    # Base BIS-MAC curve from Schumacher/Kim/Wrobel data
    # BIS = 35 + 60 / (1 + (M_eff / 0.31)^1.45)
    bis_base = 35.0 + 60.0 / (1.0 + (m_eff / 0.31) ** 1.45)

    # Apply agent-specific offset
    bis_final = bis_base + agent_offset

    # Clamp to realistic BIS range
    return clamp(bis_final, 30.0, 98.0)


def compute_mac_equivalent_from_drugs(ce_prop: float, remi_rate_ug_kg_min: float) -> float:
    """
    Convert IV drug concentrations to MAC-equivalent hypnotic effect.

    Based on empirical calibration to published BIS response surfaces.

    Args:
        ce_prop: Propofol effect-site concentration (µg/mL)
        remi_rate_ug_kg_min: Remifentanil infusion rate (µg/kg/min)

    Returns:
        MAC-equivalent contribution (dimensionless)

    Calibration:
        - Propofol 3.7 µg/mL alone → ΔM = 0.66 (BIS ~50)
        - Remifentanil 0.2 µg/kg/min → ΔM = 0.44 (with volatile)
    """
    # Propofol MAC-equivalent: 0.18 per µg/mL
    # At 3.7 µg/mL: 0.18 × 3.7 = 0.66 MAC (produces BIS ~50)
    delta_m_prop = 0.18 * ce_prop

    # Remifentanil MAC-equivalent: 0.44 × (R / 0.2)
    # Linear in clinical range 0.05-0.3 µg/kg/min
    delta_m_remi = 0.44 * (remi_rate_ug_kg_min / 0.2)

    # Additive interaction (first-order approximation)
    # Cap at 2.0 MAC-equivalent to prevent unrealistic values during massive overdose
    return min(delta_m_prop + delta_m_remi, 2.0)


# -----------------------------------------------------------------------------
# BIS Models
# -----------------------------------------------------------------------------

@dataclass
class BISModelParams:
    c50p: float
    c50r: float
    gamma: float
    beta: float # Interaction (Greco or Minto)
    e0: float
    emax: float
    delay: float # seconds

class BISModel:
    """
    BIS PD Model for IV agents (Propofol/Remifentanil) and Volatile anesthetics.

    Uses two approaches:
    1. IV-only: Traditional Bouillon/Eleveld interaction models.
       - Bouillon: beta=0 (additive) by default.
       - Eleveld: Strictly Propofol-only (c50r=0), ignores Remifentanil interaction.
    2. Volatile-based: Evidence-based MAC-BIS relationship (Schumacher PM et al. 2003 / Kim TK et al. 2005)

    When volatiles are present, uses MAC-based approach with IV drug contributions
    converted to MAC-equivalents.
    """
    def __init__(self, patient: Patient, model_name: str = "Bouillon"):
        self.model_name = model_name
        self.patient = patient

        # Smoothing
        self.bis_smoothed = 98.0
        self.alpha_smooth = 0.1 # Exponential moving average factor

        # Default Params (Bouillon for IV agents)
        self.params = BISModelParams(
            c50p=4.47, c50r=19.3, gamma=1.43, beta=0.0, e0=97.4, emax=97.4, delay=0.0
        )

        if model_name == "Bouillon":
            # Minto-type interaction with default parameters
            pass

        elif model_name == "Eleveld":
            # Eleveld DJ et al. Br J Anaesth 2018 with age-dependent parameters
            age = patient.age
            # Functions
            def faging(x): return np.exp(x * (age - 35))
            def fdelay(x): return 15 + np.exp(x * age)

            c50p = 3.08 * faging(-0.00635)
            # Gamma is 1.89 usually, but 1.47 if Ce > C50 (handled in compute)

            self.params.c50p = c50p
            self.params.c50r = 0.0 # No interaction in base Eleveld Propofol model
            self.params.gamma = 1.89
            self.params.gamma2 = 1.47 # Special for Eleveld
            self.params.e0 = 93.0
            self.params.emax = 93.0
            self.params.delay = fdelay(0.0517)

        elif model_name == "Fuentes":
            # Greco-type
            self.params.c50p = 2.99
            self.params.c50r = 21.0
            self.params.gamma = 2.69
            self.params.beta = 0.0
            self.params.e0 = 94.0
            self.params.emax = 94.0 * 0.81

        elif model_name == "Yumuk":
            # Greco-type with synergy
            self.params.c50p = 7.66
            self.params.c50r = 149.62
            self.params.gamma = 4.07
            self.params.beta = 15.03 # Synergy!
            self.params.e0 = 93.97
            self.params.emax = 93.97

        # Store initial c50p for blood loss mechanics
        self.c50p_init = self.params.c50p

        # Delay buffer
        # dt_buffer removed (unused)
        self.output_buffer = [self.params.e0]

    def initialize(self, bis_target: float, dt: float = 0.01):
        """Force internal state to target BIS."""
        self.bis_smoothed = bis_target

        # Fill buffer
        steps_delay = int(np.ceil(self.params.delay / dt)) if self.params.delay > 0 else 10
        self.output_buffer = [bis_target] * steps_delay

    def update_param_blood_loss(self, v_ratio: float):
        """
        Update c50p based on blood loss (Johnson KB et al. Anesthesiology 2003).
        v_ratio = current_vol / init_vol
        """
        # Decrease C50 (increase sensitivity) with blood loss
        # Formula: c50 = c50_init - 6 * (1 - v_ratio)
        new_c50 = self.c50p_init - 6.0 * (1.0 - v_ratio)
        self.params.c50p = max(0.1, new_c50) # Safety floor

    def step(self, dt: float, ce_prop: float, ce_remi: float = 0.0,
             mac_sevo: float = 0.0,
             remi_rate_ug_kg_min: float = 0.0) -> float:
        """
        Compute BIS from IV and volatile anesthetic effects.

        Args:
            dt: Time step (seconds)
            ce_prop: Propofol effect-site concentration (µg/mL)
            ce_remi: Remifentanil effect-site concentration (ng/mL)
            mac_sevo: Sevoflurane age-adjusted MAC fraction
            remi_rate_ug_kg_min: Remifentanil infusion rate (µg/kg/min) for MAC conversion

        Returns:
            BIS value (30-98)
        """
        total_mac = mac_sevo

        if total_mac > 0.01:
            # Use MAC-based model when volatiles present
            bis_raw = self._compute_bis_volatile(
                ce_prop, mac_sevo, remi_rate_ug_kg_min
            )
        else:
            # Use traditional IV-only model
            bis_raw = self._compute_bis_iv_only(ce_prop, ce_remi)

        # Smoothing
        self.bis_smoothed = (1.0 - self.alpha_smooth) * self.bis_smoothed + self.alpha_smooth * bis_raw

        bis_out = self.bis_smoothed

        # Apply delay if configured
        steps_delay = int(np.ceil(self.params.delay / dt)) if self.params.delay > 0 else 0

        if steps_delay > 0:
            if len(self.output_buffer) != steps_delay:
                 # Resize buffer if needed
                 self.output_buffer = [self.bis_smoothed] * steps_delay

            self.output_buffer.append(bis_out)
            bis_delayed = self.output_buffer.pop(0)
            return bis_delayed
        else:
            return bis_out

    def compute_bis(self, ce_prop: float, ce_remi: float = 0.0,
                   u_volatile: float = 0.0) -> float:
        """
        Compute steady-state BIS for equilibrium solver (no state update, no delay).

        Args:
            ce_prop: Propofol effect-site concentration (µg/mL)
            ce_remi: Remifentanil effect-site concentration (ng/mL)
            u_volatile: Normalized volatile load (MAC fraction, 0.0-3.0)

        Returns:
            BIS value (30-98)
        """
        if u_volatile > 0.01:
            # Use MAC-based model
            # Assume sevoflurane as default volatile for equilibrium
            # Convert ce_remi (ng/mL) to approximate infusion rate (µg/kg/min)
            # Based on Minto steady-state: Ce ≈ 25 × Rate
            # Therefore: Rate ≈ Ce × 0.04
            # See unit documentation header for derivation
            remi_rate_ug_kg_min = ce_remi * 0.04
            bis_raw = self._compute_bis_volatile(
                ce_prop, mac_sevo=u_volatile,
                remi_rate_ug_kg_min=remi_rate_ug_kg_min
            )
        else:
            # Use traditional IV-only model
            bis_raw = self._compute_bis_iv_only(ce_prop, ce_remi)

        return bis_raw

    def _compute_bis_volatile(self, ce_prop: float, mac_sevo: float,
                               remi_rate_ug_kg_min: float) -> float:
        """
        Compute BIS using evidence-based MAC-BIS relationship.

        Uses equations from Schumacher PM et al. 2003, Kim TK et al. 2005 validated against clinical data.
        """
        # Convert IV drugs to MAC-equivalent
        delta_m_iv = compute_mac_equivalent_from_drugs(ce_prop, remi_rate_ug_kg_min)

        # Compute BIS for each volatile agent (they don't combine linearly in BIS space)
        # Use the dominant agent
        bis_values = []

        if mac_sevo > 0.01:
            m_eff_sevo = mac_sevo + delta_m_iv
            bis_sevo = compute_bis_from_mac(m_eff_sevo, agent_offset=0.0)
            bis_values.append((mac_sevo, bis_sevo))

        # Note: Previous fallback for !bis_values is unreachable because
        # this method is only called if total_mac > 0.01, and we assume sevo is dominant/present.

        # Weight by MAC fraction
        # This is valid for standard MAC-weighted averaging of effects
        total_mac = sum(mac for mac, _ in bis_values)
        if total_mac == 0:
            return 98.0

        weighted_bis = sum(mac * bis for mac, bis in bis_values) / total_mac

        return weighted_bis

    def _compute_bis_iv_only(self, ce_prop: float, ce_remi: float) -> float:
        """
        Compute BIS using traditional Bouillon/Eleveld IV interaction models.
        """
        p = self.params

        # Propofol U
        u_prop = ce_prop / p.c50p if p.c50p > 1e-6 else 0.0

        # Remi U
        u_remi = ce_remi / p.c50r if (p.c50r > 1e-6) else 0.0

        interaction = 0.0
        gamma = p.gamma

        if self.model_name == "Bouillon":
             # Minto-type interaction
             # Phi = Up / (Up + Ur)
             if (u_prop + u_remi) > 0:
                 phi = u_prop / (u_prop + u_remi)
             else:
                 phi = 0

             # U50 = 1 - beta * (Phi - Phi^2)
             u50 = 1 - p.beta * (phi - phi**2)

             interaction = (u_prop + u_remi) / u50

        else:
            # Greco-type interaction (Fuentes, Yumuk, etc)
            # I = Up + Ur + beta * Up * Ur
            interaction = u_prop + u_remi + p.beta * u_prop * u_remi

            # Eleveld case
            if self.model_name == "Eleveld":
                 if hasattr(p, 'gamma2') and u_prop > 1.0:
                     gamma = p.gamma2

        # Sigmoid
        # E = Emax * (I^gamma / (1 + I^gamma))
        term = interaction ** gamma
        effect = p.emax * (term / (1 + term))

        return max(0.0, p.e0 - effect)

    def inverse_hill(self, bis: float, ce_remi: float = 0.0) -> float:
        """
        Compute Propofol Ce from BIS and Remi Ce (IV-only mode).
        Inverse of _compute_bis_iv_only.
        """
        p = self.params

        # 1. Check bounds
        if bis >= p.e0: return 0.0
        if bis <= (p.e0 - p.emax): return 999.0 # Saturation

        # Effect E = E0 - BIS
        effect = p.e0 - bis

        # Term T = I^gamma = E / (Emax - E)
        term = effect / (p.emax - effect)
        if term < 0: term = 0

        # Interaction I = term ^ (1/gamma)
        gamma = p.gamma
        if self.model_name == "Eleveld":
             if bis < (p.e0 - (p.emax / 2)):
                 if hasattr(p, 'gamma2'): gamma = p.gamma2

        interaction = term ** (1.0 / gamma)

        # Now solve for Up (Propofol Normalized)
        ur = ce_remi / p.c50r if (p.c50r > 0 and ce_remi > 0) else 0.0

        # Propofol only
        if p.c50r == 0 or ur == 0:
            up = interaction
        elif self.model_name == "Bouillon":
            # Polynomial solver for Minto interaction
            yr = ur
            i_val = interaction
            beta = p.beta

            b = 3 * yr - i_val
            c = 3 * yr**2 - (2 - beta) * yr * i_val
            d = yr**3 - yr**2 * i_val

            roots = np.roots([1, b, c, d])
            # Find positive real root
            up = 0.0
            for r in roots:
                if np.isreal(r) and np.real(r) > 0:
                    up = np.real(r)
                    break

        else:
            # Greco: I = Up + Ur + beta*Up*Ur
            # Up = (I - Ur) / (1 + beta*Ur)
            if (1 + p.beta * ur) != 0:
                up = (interaction - ur) / (1 + p.beta * ur)
            else:
                up = 0.0

        if up < 0: up = 0.0

        return up * p.c50p


# -----------------------------------------------------------------------------
# LOC / Tolerance Models
# -----------------------------------------------------------------------------

class LOCModel:
    """
    Loss of Consciousness Probability (0.0 - 1.0).
    Uses Greco-type interaction.
    """
    def __init__(self, model_name: str = "Kern"):
        self.model_name = model_name
        # Defaults (Kern S et al. Acta Anaesthesiol Scand 2005 - Note: Study was on Cisatracurium, verify applicability to Atracurium)
        self.c50p = 1.80
        self.c50r = 12.5
        self.gamma = 3.76
        self.beta = 5.1

        if model_name == "Mertens":
            self.c50p = 2.92
            self.c50r = 5.15
            self.gamma = 3.88
            self.beta = 0.0
        elif model_name == "Johnson":
            self.c50p = 2.20
            self.c50r = 33.1
            self.gamma = 5.00
            self.beta = 3.60

    def compute_probability(self, ce_prop: float, ce_remi: float) -> float:
        up = ce_prop / self.c50p
        ur = ce_remi / self.c50r

        interaction = up + ur + self.beta * up * ur
        term = interaction ** self.gamma

        # Sigmoid 0-1
        return term / (1 + term)

class TOLModel:
    """
    Tolerance of Laryngoscopy Probability (0.0 - 1.0).
    Hierarchical model (Bouillon).
    """
    def __init__(self, model_name: str = "Bouillon"):
        # Parameters from PAS / Bouillon TW et al. Anesthesiology 2004
        self.c50p = 8.04
        self.c50r = 1.07
        self.gamma_p = 5.1
        self.gamma_r = 0.97
        self.pre_intensity = 1.05

    def compute_probability(self, ce_prop: float, ce_remi: float, mac: float = 0.0) -> float:
        """
        Compute P(TOL).
        """
        # 1. Post-Opioid Intensity
        c50r_scaled = self.c50r * self.pre_intensity

        if c50r_scaled == 0:
            fsig_r = 0.0
        else:
            fsig_r = (ce_remi**self.gamma_r) / (c50r_scaled**self.gamma_r + ce_remi**self.gamma_r)

        post_opioid = self.pre_intensity * (1.0 - fsig_r)

        # 2. Tolerance Probability
        c50p_scaled = self.c50p * post_opioid

        if c50p_scaled <= 1e-6:
             return 1.0

        # Effective Propofol = Ce_prop + MAC equivalent
        ce_effective = ce_prop + (mac * self.c50p)

        tol = (ce_effective**self.gamma_p) / (c50p_scaled**self.gamma_p + ce_effective**self.gamma_p)

        return tol


class TOFModel:
    """
    Train of Four (TOF) Model for Rocuronium with Recovery and Reversal.
    
    Features:
    - Effect-site compartment with asymmetric ke0 (onset vs recovery)
    - Sigmoid Emax block model at neuromuscular junction
    - Spontaneous recovery when plasma concentration drops
    - Sugammadex PK and rocuronium-sugammadex binding
    - Volatile potentiation (balanced anesthesia)
    
    References:
    - Plaud B et al. Clin Pharmacol Ther 1995 (ke0, Ce50)
    - Zwiers A et al. Clin Drug Investig 2011 (Ka binding constant)
    - Nguyen-Lee J et al. Curr Anesthesiol Rep 2018 (Sugammadex PK)
    - Ploeger BA et al. Anesthesiology 2009 (mechanism-based PK-PD)
    """
    
    # Molecular weights for molar conversions
    MW_ROCURONIUM = 609.7   # g/mol
    MW_SUGAMMADEX = 2178.0  # g/mol
    
    def __init__(self, patient: Patient, model_name: str = "Wierda", anesthesia_type: str = "TIVA",
                 fidelity_mode: str = "clinical"):
        self.patient = patient
        self.model_name = model_name
        self.anesthesia_type = anesthesia_type
        use_literature = (fidelity_mode == "literature")
        
        age = patient.age
        # sex: 0 male, 1 female
        sex = 1 if patient.sex.lower() == "female" else 0
        age_term = age - 50.0
        
        # 1. Base PD Parameters (original model for Ce50/gamma derivation)
        # -------------------------------------------------------------------------
        # Model         | Ce50 base | Gamma base | Ce50-age | Gamma-age | Gamma-sex
        # -------------------------------------------------------------------------
        # Wierda        | 1.08      | 6.41       | -0.00605 | -0.0494   | -1.24
        # Szenohradszky | 1.44      | 8.30       | -0.00862 | -0.0981   | None
        # Cooper        | 0.980     | 6.18       | -0.00557 | -0.0341   | -1.32
        # Alvarez-Gomez | 0.900     | 5.99       | -0.00539 | -0.0443   | -1.14
        # McCoy         | 1.08      | 4.20       | -0.00770 | -0.0283   | None
        # -------------------------------------------------------------------------
        
        theta7 = None  # Gamma-sex
        
        if model_name == "Szenohradszky":
            theta2 = 1.44; theta3 = 8.30; theta5 = -0.00862; theta6 = -0.0981
        elif model_name == "Cooper":
            theta2 = 0.980; theta3 = 6.18; theta5 = -0.00557; theta6 = -0.0341; theta7 = -1.32
        elif model_name == "Alvarez-Gomez":
            theta2 = 0.900; theta3 = 5.99; theta5 = -0.00539; theta6 = -0.0443; theta7 = -1.14
        elif model_name == "McCoy":
            theta2 = 1.08; theta3 = 4.20; theta5 = -0.00770; theta6 = -0.0283
        else:  # Wierda
            theta2 = 1.08; theta3 = 6.41; theta5 = -0.00605; theta6 = -0.0494; theta7 = -1.24
            
        # Calculate Ce50 and Gamma from covariate model
        self.ce50_base = theta2 + theta5 * age_term
        self.gamma = theta3 + theta6 * age_term
        
        if theta7 is not None:
            self.gamma += theta7 * sex
            
        # Global floors
        self.ce50_base = max(self.ce50_base, 0.01)
        self.gamma = max(self.gamma, 0.5)
        
        # 2. Effect-site PD parameters (from SUGAMMADEX.md)
        # Plaud B et al. Clin Pharmacol Ther 1995
        self.ke0_onset = 0.16       # min^-1, t1/2 = 4.4 min at AP
        
        # Recovery ke0: determines spontaneous recovery time
        # Literature: ke0 = 0.07 min^-1 (t1/2 ~10 min) for pure modeling
        # Tuned: 0.12 min^-1 (t1/2 ~6 min) to achieve clinical 40-70 min recovery
        self.recovery_ke0 = 0.07 if use_literature else 0.12    # min^-1 (literature 0.07, tuned 0.12)
        
        # Sigmoid Emax model parameters
        # Plaud B et al. Clin Pharmacol Ther 1995: Ce50 = 823 µg/L = 0.823 mg/L
        self.Ce50_T1 = 0.8          # mg/L for 50% block at AP
        self.gamma_T1 = 3.0         # Hill coefficient
        self.beta_TOF = 1.5         # Power law for TOF fade (T4/T1 ratio)
        
        # Reversal ke0: rapid equilibration during sugammadex reversal
        # Clinical: sugammadex reversal in 1-3 min (de Kam 2011, Ploeger 2009)
        # Mechanism: large gradient (Ce >> Cp_free) drives fast redistribution
        # Tuned: 1.0 min^-1 (t1/2 ~0.7 min) to achieve clinical reversal times
        self.reversal_ke0 = 1.0     # min^-1, tuned for 1-2 min sugammadex reversal
        
        # 3. Volatile Potentiation Factors (f_anesth)
        # Iso/Sevo/Des ~ 0.7-0.75 (Ce50 reduced -> more potent)
        self.f_sevo = 0.75
        
        # 4. Sugammadex PK parameters
        # Nguyen-Lee J et al. Curr Anesthesiol Rep 2018
        # APSF 2025 Safety Review
        self.Vs_L_kg = 0.18        # Volume of distribution (L/kg)
        self.Cl_s_mL_min = 88.0    # Clearance (mL/min)
        self.Vs = self.Vs_L_kg * patient.weight  # Total volume (L)
        self.kel_s = (self.Cl_s_mL_min / 1000.0) / self.Vs  # min^-1
        
        # 5. Binding constant
        # Zwiers A et al. Clin Drug Investig 2011
        # Naguib M. Anesth Analg 2007
        self.Ka = 1.79e7           # M^-1 (association constant)
        
        # State variables
        self.ce = 0.0              # Effect-site rocuronium concentration (mg/L)
        self.prev_cp = 0.0         # Previous plasma concentration for onset/recovery detection
        self.sugammadex_amount_umol = 0.0  # Sugammadex amount in central compartment (µmol)
        
    def step_recovery(self, dt_sec: float, cp_roc_mg_l: float, mac_sevo: float = 0.0) -> float:
        """
        Update effect-site concentration and compute TOF ratio.
        
        This is the main step function that should be called each simulation step.
        It handles:
        1. Sugammadex PK update (exponential decay)
        2. Rocuronium-sugammadex binding equilibrium
        3. Effect-site equilibration with asymmetric ke0
        4. TOF computation via sigmoid Emax model
        
        Args:
            dt_sec: Time step in seconds
            cp_roc_mg_l: Total rocuronium plasma concentration (mg/L)
            mac_sevo: Sevoflurane MAC fraction for potentiation
            
        Returns:
            TOF ratio as percentage (0-100)
        """
        dt_min = dt_sec / 60.0
        
        # 1. Update sugammadex PK (first-order elimination)
        if self.sugammadex_amount_umol > 0:
            self.sugammadex_amount_umol *= np.exp(-self.kel_s * dt_min)
        
        # 2. Compute free rocuronium after sugammadex binding
        cp_free = self._compute_free_rocuronium(cp_roc_mg_l)
        
        # 3. Determine ke0 based on concentration dynamics
        # Plaud B et al. Clin Pharmacol Ther 1995: asymmetric equilibration
        #
        # Special case: When sugammadex causes a large Cp_free drop (Ce >> Cp_free),
        # use faster ke0 because the large concentration gradient drives rapid
        # redistribution from effect-site back to plasma.
        # de Kam PJ et al. Br J Clin Pharmacol 2011: rapid reversal kinetics
        
        if cp_free >= self.prev_cp:
            # Rising or stable - use onset ke0
            ke0 = self.ke0_onset
        elif self.ce > cp_free * 3.0 and self.sugammadex_amount_umol > 0:
            # Active sugammadex reversal: Ce significantly exceeds Cp_free
            # Use fast reversal ke0 for rapid redistribution
            # de Kam PJ et al. Br J Clin Pharmacol 2011: ~1-3 min reversal times
            ke0 = self.reversal_ke0
        else:
            # Normal spontaneous recovery - use slower recovery ke0
            ke0 = self.recovery_ke0
        self.prev_cp = cp_free
        
        # 4. Effect-site compartment update
        # dCe/dt = ke0 * (Cp_free - Ce)
        # Discrete: Ce_next = Ce + ke0 * (Cp_free - Ce) * dt
        self.ce += ke0 * (cp_free - self.ce) * dt_min
        self.ce = max(0.0, self.ce)
        
        # 5. Compute TOF with volatile potentiation
        return self._compute_tof_from_ce(self.ce, mac_sevo)
    
    def _compute_free_rocuronium(self, cp_total_mg_l: float) -> float:
        """
        Compute free rocuronium concentration after sugammadex binding.
        
        Uses quadratic equilibrium solution for 1:1 binding.
        Zwiers A et al. Clin Drug Investig 2011: Ka = 1.79e7 M^-1
        
        Args:
            cp_total_mg_l: Total rocuronium plasma concentration (mg/L)
            
        Returns:
            Free rocuronium concentration (mg/L)
        """
        if self.sugammadex_amount_umol <= 0 or cp_total_mg_l <= 0:
            return cp_total_mg_l
        
        # Convert to molar concentrations
        # R_tot (mol/L) = (mg/L) / (g/mol) / 1000
        R_tot = (cp_total_mg_l / self.MW_ROCURONIUM) / 1000.0  # mol/L
        
        # Sugammadex concentration: amount (µmol) / volume (L) / 1e6 = mol/L
        S_tot = (self.sugammadex_amount_umol / self.Vs) / 1e6  # mol/L
        
        if R_tot <= 0 or S_tot <= 0:
            return cp_total_mg_l
        
        # Quadratic solution for complex concentration C
        # Ka*C^2 - [Ka(R_tot + S_tot) + 1]*C + Ka*R_tot*S_tot = 0
        # Ploeger BA et al. Anesthesiology 2009
        
        Ka = self.Ka
        a = Ka
        b = -(Ka * (R_tot + S_tot) + 1.0)
        c_coef = Ka * R_tot * S_tot
        
        discriminant = b * b - 4.0 * a * c_coef
        discriminant = max(0.0, discriminant)  # Numerical safety
        
        # Choose the smaller root (physically meaningful: C <= min(R_tot, S_tot))
        C_complex = (-b - np.sqrt(discriminant)) / (2.0 * a)
        
        # Free rocuronium
        R_free = R_tot - C_complex
        R_free = max(0.0, R_free)
        
        # Convert back to mg/L
        return R_free * self.MW_ROCURONIUM * 1000.0
    
    def _compute_tof_from_ce(self, ce: float, mac_sevo: float = 0.0) -> float:
        """
        Compute TOF ratio from effect-site concentration.
        
        Uses sigmoid Emax model with volatile potentiation.
        
        Args:
            ce: Effect-site rocuronium concentration (mg/L)
            mac_sevo: Sevoflurane MAC fraction
            
        Returns:
            TOF ratio as percentage (0-100)
        """
        # Volatile potentiation - reduce Ce50 (increase potency)
        total_mac_vol = mac_sevo
        f_effective = 1.0  # Default TIVA (no potentiation)
        
        if total_mac_vol > 0.1:
            # Interpolate: f = 1.0 at MAC=0, f = f_sevo at MAC≥1
            target_f = self.f_sevo
            f_effective = 1.0 - (1.0 - target_f) * min(total_mac_vol, 1.0)
            
        Ce50_eff = self.Ce50_T1 * f_effective
        
        if ce < 0:
            ce = 0
        
        # Sigmoid Emax: B = Ce^gamma / (Ce50^gamma + Ce^gamma)
        # T1 = 1 - B (twitch height)
        # TOFR = T1^beta (fade model)
        # de Kam PJ et al. Br J Clin Pharmacol 2011
        
        if ce <= 1e-9:
            return 100.0  # No block
        
        Ce50_g = Ce50_eff ** self.gamma_T1
        Ce_g = ce ** self.gamma_T1
        
        B = Ce_g / (Ce50_g + Ce_g)  # Fractional block (0 to 1)
        T1 = 1.0 - B                 # Twitch height (0 to 1)
        T1 = clamp01(T1)
        
        TOFr = T1 ** self.beta_TOF   # TOF ratio with fade
        
        return TOFr * 100.0  # Convert to percentage
        
    def give_sugammadex(self, dose_mg: float, weight_kg: float = None):
        """
        Administer sugammadex bolus.
        
        Args:
            dose_mg: Total dose in mg (or mg/kg if weight provided)
            weight_kg: Patient weight; if provided, dose_mg is treated as mg/kg
        """
        if weight_kg is not None:
            total_dose_mg = dose_mg * weight_kg
        else:
            total_dose_mg = dose_mg
        
        # Convert mg to µmol: (mg / (g/mol)) * 1000 = µmol
        dose_umol = (total_dose_mg / self.MW_SUGAMMADEX) * 1000.0
        
        self.sugammadex_amount_umol += dose_umol
        
    def compute_tof(self, ce_roc: float, mac_sevo: float = 0.0) -> float:
        """
        Compute TOF Ratio % (0-100) - simplified interface for backward compatibility.
        
        Note: For full recovery dynamics, use step_recovery() instead.
        This method uses the original Ce50/gamma formulation for compatibility.
        
        Args:
            ce_roc: Effect site concentration (µg/mL = mg/L)
            mac_sevo: MAC fraction of volatiles (used for potentiation)
            
        Returns:
            TOF ratio as percentage (0-100)
        """
        # Use the new effect-site based computation
        return self._compute_tof_from_ce(ce_roc, mac_sevo)
    
    def reset(self):
        """Reset model state to initial conditions."""
        self.ce = 0.0
        self.prev_cp = 0.0
        self.sugammadex_amount_umol = 0.0
