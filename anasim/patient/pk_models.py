import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
from .patient import Patient
from anasim.core.utils import clamp

# =============================================================================
# PHARMACOKINETIC MODELS - LITERATURE REFERENCES
# =============================================================================
#
# This module implements 3-compartment mammillary PK models for anesthetic drugs.
#
# Propofol Models:
#   - Marsh: Marsh et al. Br J Anaesth. 1991; Weight-based model
#   - Schnider: Schnider et al. Anesthesiology. 1998; Age, LBM, height adjusted
#   - Eleveld: Eleveld et al. Br J Anaesth. 2018; Population PK meta-analysis, allometric scaling
#
# Remifentanil:
#   - Minto: Minto et al. Anesthesiology. 1997; Age and LBM adjusted
#
# Rocuronium:
#   - Wierda JM: Wierda et al. Can J Anaesth. 1991; Standard PK parameters
#
# Vasopressors:
#   - Norepinephrine:
#       - Beloeil et al. Br J Anaesth. 2005;95:782-788 (septic shock/trauma, 1-comp PK-PD)
#       - Oualha et al. Br J Clin Pharmacol. 2014;78:886-897 (critically ill children, popPK-PD)
#       - Li et al. Clin Pharmacokinet. 2024;63:1597-1608 (healthy volunteers, 2-comp with propofol interaction)
#   - Epinephrine: Heuristic 1-compartment model calibrated to t1/2 ~2.5 min, Vd ~0.15 L/kg;
#       not a direct implementation of a specific population PK publication.
#   - Phenylephrine: Heuristic 1-compartment model calibrated to t1/2 ~7 min, Vd ~0.20 L/kg;
#       loosely based on clinical time course from historical hemodynamic studies, not a
#       published population PK model.
#
# Units:
#   - Compartment volumes: L
#   - Rate constants (k10, k12, etc.): min^-1
#   - Effect-site equilibration (ke0): min^-1
#   - Concentrations: µg/mL (propofol), ng/mL (remi, vasopressors)
# =============================================================================

@dataclass
class PKState:
    """State of the 3-compartment PK model."""
    c1: float = 0.0 # Central compartment concentration
    c2: float = 0.0 # Fast peripheral
    c3: float = 0.0 # Slow peripheral
    ce: float = 0.0 # Effect site

class ThreeCompartmentPK:
    """
    Generic 3-compartment Pharmacokinetic model.
    """
    def __init__(self, v1: float, v2: float, v3: float, 
                 k10: float, k12: float, k21: float, k13: float, k31: float, 
                 ke0: float):
        self.state = PKState()
        
        # Parameters
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        
        # Rate constants (min^-1)
        self.k10 = k10
        self.k12 = k12
        self.k21 = k21
        self.k13 = k13
        self.k31 = k31
        self.ke0 = ke0
        
        # Store baselines for hemodynamic scaling
        self.v1_base = v1
        self.v2_base = v2
        self.v3_base = v3
        self.k10_base = k10
        self.k12_base = k12
        self.k21_base = k21
        self.k13_base = k13
        self.k31_base = k31
        
        # Cached volume ratios (avoids repeated division in step())
        self._update_volume_ratios()
        
    def _update_volume_ratios(self):
        """Cache volume ratios for ODE calculations."""
        self._v2_v1 = self.v2 / self.v1
        self._v3_v1 = self.v3 / self.v1
        self._v1_v2 = self.v1 / self.v2
        self._v1_v3 = self.v1 / self.v3
        
    def update_hemodynamics(self, v_ratio: float, co_ratio: float):
        """
        Scale parameters based on blood volume ratio (v_ratio) and Cardiac Output ratio (co_ratio).
        v_ratio = current_blood_vol / initial_blood_vol
        co_ratio = current_CO / initial_CO
        """
        # Update V1
        self.v1 = self.v1_base * v_ratio
        
        # Infer Flows (Cl) from baselines
        # Cl = k * V_source
        cl1_base = self.k10_base * self.v1_base
        cl2_base = self.k12_base * self.v1_base 
        cl3_base = self.k13_base * self.v1_base
        
        # Scale Flows by CO
        cl1 = cl1_base * co_ratio
        cl2 = cl2_base * co_ratio
        cl3 = cl3_base * co_ratio
        
        # Recalculate k constants
        self.k10 = cl1 / self.v1
        self.k12 = cl2 / self.v1
        self.k21 = cl2 / self.v2
        self.k13 = cl3 / self.v1
        self.k31 = cl3 / self.v3
        
        # Update cached volume ratios
        self._update_volume_ratios()


    def step(self, dt_sec: float, infusion_rate_mg_sec: float) -> PKState:
        """
        Advance the PK state by dt_sec.
        
        Args:
            dt_sec: Time step in seconds.
            infusion_rate_mg_sec: Infusion rate in mg/sec (or units consistent with V1).
        """
        # Convert dt to minutes for rate constants (which are usually 1/min)
        dt_min = dt_sec / 60.0
        
        # Rin(t)/V1 term: (mg/min) / L = mg/L/min
        rin_mg_min = infusion_rate_mg_sec * 60.0
        
        c1, c2, c3, ce = self.state.c1, self.state.c2, self.state.c3, self.state.ce
        k10 = self.k10
        k12 = self.k12
        k13 = self.k13
        k21 = self.k21
        k31 = self.k31
        v1 = self.v1
        v2_v1 = self._v2_v1
        v3_v1 = self._v3_v1
        v1_v2 = self._v1_v2
        v1_v3 = self._v1_v3
        k_sum = k10 + k12 + k13
        rin_term = rin_mg_min / v1
        
        # 3-Compartment Mammillary Model ODE (Mass-Conserving Formulation)
        # ------------------------------------------------------------------
        # This formulation uses volume ratios to ensure mass balance when
        # k-constants are defined per the original Marsh/Eleveld publications.
        #
        # Rate constants defined as:
        #   k10 = Cl1/V1 (elimination from central)
        #   k12 = Cl2/V1 (transfer C1->C2, rate constant for C1 perspective)
        #   k21 = Cl2/V2 (transfer C2->C1, rate constant for C2 perspective)
        #   k13 = Cl3/V1, k31 = Cl3/V3 (similar for C3)
        #
        # For mass conservation in concentration ODEs:
        #   dC1/dt: Mass flow C2->C1 = k21*V2*C2. Conc change in C1 = k21*(V2/V1)*C2
        #   dC2/dt: Mass flow C1->C2 = k12*V1*C1. Conc change in C2 = k12*(V1/V2)*C1
        #
        # At equilibrium: C1 = C2 = C3 (uniform concentration), and all flows balance.
        
        gradient_c1 = -k_sum * c1 + \
                       k21 * v2_v1 * c2 + \
                       k31 * v3_v1 * c3 + \
                       rin_term
                       
        gradient_c2 = k12 * v1_v2 * c1 - k21 * c2
                       
        gradient_c3 = k13 * v1_v3 * c1 - k31 * c3
                       
        gradient_ce = self.ke0 * (c1 - ce)
        
        self.state.c1 += gradient_c1 * dt_min
        self.state.c2 += gradient_c2 * dt_min
        self.state.c3 += gradient_c3 * dt_min
        self.state.ce += gradient_ce * dt_min
        
        return self.state

    def reset(self):
        self.state = PKState()

    def simulate_decay(self, target_fraction: float = 0.5, max_seconds: int = 3600) -> float:
        """
        Simulate decay from current state and return time to reach target_fraction of Ce.
        
        Used for context-sensitive half-time (CSHT) prediction without duplicating
        ODE logic. Runs a lightweight simulation without modifying actual state.
        
        Args:
            target_fraction: Fraction of current Ce to decay to (0.5 for half-time)
            max_seconds: Maximum simulation time in seconds
            
        Returns:
            Time in minutes to reach target, or max_seconds/60 if not reached
        """
        ce_current = self.state.ce
        if ce_current < 0.5:  # Below clinically relevant threshold
            return 0.0
            
        target_ce = ce_current * target_fraction
        
        # Copy current compartment values for simulation
        c1, c2, c3, ce = self.state.c1, self.state.c2, self.state.c3, self.state.ce
        
        # Use 1-second steps
        dt_min = 1.0 / 60.0
        k10 = self.k10
        k12 = self.k12
        k13 = self.k13
        k21 = self.k21
        k31 = self.k31
        k_sum = k10 + k12 + k13
        v2_v1 = self._v2_v1
        v3_v1 = self._v3_v1
        v1_v2 = self._v1_v2
        v1_v3 = self._v1_v3
        ke0 = self.ke0
        
        for t in range(max_seconds):
            # Euler ODE step (no infusion)
            dc1 = -k_sum * c1 + k21 * v2_v1 * c2 + k31 * v3_v1 * c3
            dc2 = k12 * v1_v2 * c1 - k21 * c2
            dc3 = k13 * v1_v3 * c1 - k31 * c3
            dce = ke0 * (c1 - ce)
            
            c1 += dc1 * dt_min
            c2 += dc2 * dt_min
            c3 += dc3 * dt_min
            ce += dce * dt_min
            
            if ce <= target_ce:
                return t / 60.0
                
        return max_seconds / 60.0

    def get_ss_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return Continuous State Space Matrices A, B.
        State vector: [c1, c2, c3, ce]
        Input: Rin (units consistent with step, i.e., mass/time)
        B vector scales input to concentration/time.
        dC1/dt = ... + Rin/V1
        """
        # A matrix (Mass-Conserving Formulation with Volume Ratios)
        # dC1/dt
        a11 = -(self.k10 + self.k12 + self.k13)
        a12 = self.k21 * self._v2_v1
        a13 = self.k31 * self._v3_v1
        a14 = 0.0
        
        # dC2/dt
        a21 = self.k12 * self._v1_v2
        a22 = -self.k21
        a23 = 0.0
        a24 = 0.0
        
        # dC3/dt
        a31 = self.k13 * self._v1_v3
        a32 = 0.0
        a33 = -self.k31
        a34 = 0.0
        
        # dCe/dt
        a41 = self.ke0
        a42 = 0.0
        a43 = 0.0
        a44 = -self.ke0
        
        A = np.array([
            [a11, a12, a13, a14],
            [a21, a22, a23, a24],
            [a31, a32, a33, a34],
            [a41, a42, a43, a44]
        ])
        
        # B Vector
        # Input Rin (mg/min if dt=min). dC/dt = Rin/V1. 
        # Input Rin is typically supplied in mg/sec or ug/min.
        # If input u is in "Mass/Time" (e.g. mg/min), then B = [1/V1, 0, 0, 0].
        # The units of A and B must be consistent with time unit of simulation.
        # If simulation dt is passed to TCI in seconds, TCI usually discretizes A (1/min) -> Ad.
        # Return A in 1/min units.
        B = np.array([
            [1.0 / self.v1],
            [0.0],
            [0.0],
            [0.0]
        ])
        
        return A, B

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _organ_function(patient: Patient, attr: str) -> float:
    try:
        value = float(getattr(patient, attr, 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return clamp(value, 0.1, 1.0)

def organ_clearance_scaler(patient: Patient, hepatic_fraction: float = 0.0, renal_fraction: float = 0.0) -> float:
    """
    Weighted clearance scaler based on organ function fractions.
    Fractions should sum to <= 1.0; remaining clearance is assumed unaffected.
    """
    hepatic = _organ_function(patient, "hepatic_function")
    renal = _organ_function(patient, "renal_function")
    other = max(0.0, 1.0 - hepatic_fraction - renal_fraction)
    return hepatic_fraction * hepatic + renal_fraction * renal + other

def hepatic_vd_multiplier(patient: Patient, severe_multiplier: float) -> float:
    """
    Scale volume of distribution with hepatic impairment.
    Severe impairment is assumed at hepatic_function = 0.5.
    """
    hepatic = _organ_function(patient, "hepatic_function")
    severity = clamp((1.0 - hepatic) / 0.5, 0.0, 1.0)
    return 1.0 + (severe_multiplier - 1.0) * severity

def james_lbm(weight_kg: float, height_cm: float, sex: str) -> float:
    """
    James formula for Lean Body Mass.
    """
    if sex.lower() == "male":
        return 1.1 * weight_kg - 128 * ((weight_kg / height_cm) ** 2)
    else:
        return 1.07 * weight_kg - 148 * ((weight_kg / height_cm) ** 2)

# -----------------------------------------------------------------------------
# Propofol Models
# -----------------------------------------------------------------------------

class PropofolPKMarsh(ThreeCompartmentPK):
    """
    Marsh model for Propofol (Paediatric/Adult).
    """
    def __init__(self, patient: Patient):
        # Marsh et al. Br J Anaesth. 1991
        # V1, V2, V3 proportional to weight
        # k constants fixed (or derived from Cl)
        
        w = patient.weight
        
        v1 = 0.228 * w
        v2 = 0.463 * w
        v3 = 2.893 * w
        
        # Rate constants derived from volumes and clearance in original paper, 
        # or given directly. PAS uses:
        # cl1 = 0.119 * v1  (which implies k10 = 0.119 min^-1)
        k10 = 0.119
        k12 = 0.112
        k21 = 0.055
        k13 = 0.042
        k31 = 0.0033

        # Hepatic impairment: increase Vd with preserved clearances
        v_scale = hepatic_vd_multiplier(patient, severe_multiplier=1.6)
        if v_scale != 1.0:
            cl1 = k10 * v1
            cl2 = k12 * v1
            cl3 = k13 * v1
            v1 *= v_scale
            v2 *= v_scale
            v3 *= v_scale
            k10 = cl1 / v1
            k12 = cl2 / v1
            k21 = cl2 / v2
            k13 = cl3 / v1
            k31 = cl3 / v3

        # Renal impairment: propofol clearance includes significant renal extraction
        clearance_scale = organ_clearance_scaler(patient, hepatic_fraction=0.0, renal_fraction=0.3)
        k10 *= clearance_scale
        
        # Ke0:
        # Original Marsh et al. Br J Anaesth. 1991 PK paper does not specify ke0.
        # "Marsh modified" for effect-site TCI commonly uses ke0 = 1.2 min^-1.
        # This value is from clinical TCI implementations, not the original publication.
        ke0 = 1.2
        
        super().__init__(v1, v2, v3, k10, k12, k21, k13, k31, ke0)


class PropofolPKSchnider(ThreeCompartmentPK):
    """
    Schnider model for Propofol.
    """
    def __init__(self, patient: Patient):
        # Schnider et al. Anesthesiology. 1998
        age = patient.age
        w = patient.weight
        h = patient.height
        lbm = patient.lbm
        
        # Volumes [L]
        v1 = 4.27
        v2 = 18.9 - 0.391 * (age - 53)
        v3 = 238.0
        
        # Clearances [L/min]
        cl1 = 1.89 + 0.0456 * (w - 77) - 0.0681 * (lbm - 59) + 0.0264 * (h - 177)
        cl2 = 1.29 - 0.024 * (age - 53)
        cl3 = 0.836

        # Hepatic impairment: increase Vd with preserved clearances
        v_scale = hepatic_vd_multiplier(patient, severe_multiplier=1.6)
        if v_scale != 1.0:
            v1 *= v_scale
            v2 *= v_scale
            v3 *= v_scale

        # Renal impairment: propofol clearance includes significant renal extraction
        cl1 *= organ_clearance_scaler(patient, hepatic_fraction=0.0, renal_fraction=0.3)
        
        # Rate constants
        k10 = cl1 / v1
        k12 = cl2 / v1
        k21 = cl2 / v2
        k13 = cl3 / v1
        k31 = cl3 / v3
        
        # Ke0
        ke0 = 0.456
        
        super().__init__(v1, v2, v3, k10, k12, k21, k13, k31, ke0)


class PropofolPKEleveld(ThreeCompartmentPK):
    """
    Eleveld et al. Br J Anaesth. 2018 (General Purpose PK-PD Model for Propofol).
    """
    def __init__(self, patient: Patient):
        # Eleveld et al. Br J Anaesth. 2018
        age = patient.age
        w = patient.weight
        h = patient.height_m if hasattr(patient, "height_m") else patient.height / 100.0
        sex_is_male = (patient.sex.lower() == "male")
        bmi = w / (h**2)
        
        # Reference patient
        AGE_ref = 35
        WGT_ref = 70
        BMI_ref = WGT_ref / (1.7**2)  # ~24.2
        
        # Simplified Eleveld implementation based on published theta vector
        theta = [
            0, # dummy
            6.2830780766822, 25.5013145036879, 272.8166615043603, # V1ref, V2ref, V3ref
            1.7895836588902, 1.7500983738779, 1.1085424008536,    # Clref, Q2ref, Q3ref
            0.191307, 42.2760190602615, 9.0548452392807,          # ...
            -0.015633, -0.00285709, 33.5531248778544,             # V2 age, CL age, W50
            -0.0138166, 68.2767978846832,                         # V3 age, Q3 mat
            2.1002218877899, 1.3042680471360,                     # CL female, Q2 mat
            1.4189043652084, 0.6805003109141                      # Venous...
        ]
        
        # Helper functions
        def faging(x): return np.exp(x * (age - AGE_ref))
        def fsig(x, C50, gam): return x**gam / (C50**gam + x**gam)
        def fcentral(x): return fsig(x, theta[12], 1)
        
        def fal_sallami(is_male, w_val, age_val, bmi_val):
            if is_male:
                return (0.88 + (1 - 0.88) / (1 + (age_val / 13.4)**(-12.7))) * (9270 * w_val) / (6680 + 216 * bmi_val)
            else:
                return (1.11 + (1 - 1.11) / (1 + (age_val / 7.1)**(-1.1))) * (9270 * w_val) / (8780 + 244 * bmi_val)

        # Maturation functions
        pma_weeks = (age * 52) + 40
        pma_ref_weeks = (AGE_ref * 52) + 40
        
        fCLmat = fsig(pma_weeks, theta[8], theta[9])
        fCLmat_ref = fsig(pma_ref_weeks, theta[8], theta[9])
        
        fQ3mat = fsig(pma_weeks, theta[14], 1)
        fQ3mat_ref = fsig(pma_ref_weeks, theta[14], 1)
        
        # Fat Free Mass (Al-Sallami)
        ffm = fal_sallami(sex_is_male, w, age, bmi)
        ffm_ref = fal_sallami(True, WGT_ref, AGE_ref, BMI_ref) # Reference is male
        
        # Opiate co-administration factor (assume True/present for typical anaesthesia, or make configurable)
        # PAS assumes True by default for Eleveld
        fopiate_v3 = np.exp(theta[13] * age) # if opiate present
        fopiate_cl = np.exp(theta[11] * age) # PAS code line 325: fopiate(theta[11]) where fopiate(x) returns exp(x*age) if opiate
        
        # Volumes
        v1 = theta[1] * fcentral(w) / fcentral(WGT_ref)
        
        v2 = theta[2] * (w / WGT_ref) * faging(theta[10]) # theta[10] = V2 age slope
        v2_ref = theta[2]
        
        v3 = theta[3] * (ffm / ffm_ref) * fopiate_v3
        v3_ref = theta[3]
        
        # Clearances
        # theta[4] is male Cl, theta[15] is female Cl
        cl_base = theta[4] if sex_is_male else theta[15]
        cl1 = cl_base * (w / WGT_ref)**0.75 * (fCLmat / fCLmat_ref) * fopiate_cl
        
        cl2 = theta[5] * (v2 / v2_ref)**0.75 * (1 + theta[16] * (1 - fQ3mat))
        
        cl3 = theta[6] * (v3 / v3_ref)**0.75 * (fQ3mat / fQ3mat_ref)

        # Hepatic impairment: increase Vd with preserved clearances
        v_scale = hepatic_vd_multiplier(patient, severe_multiplier=1.6)
        if v_scale != 1.0:
            v1 *= v_scale
            v2 *= v_scale
            v3 *= v_scale

        # Renal impairment: propofol clearance includes significant renal extraction
        cl1 *= organ_clearance_scaler(patient, hepatic_fraction=0.0, renal_fraction=0.3)
        
        # Arterial ke0
        ke0 = 0.146 * (w / 70)**(-0.25)
        
        # Rate constants
        k10 = cl1 / v1
        k12 = cl2 / v1
        k21 = cl2 / v2
        k13 = cl3 / v1
        k31 = cl3 / v3
        
        super().__init__(v1, v2, v3, k10, k12, k21, k13, k31, ke0)


# -----------------------------------------------------------------------------
# Remifentanil Models
# -----------------------------------------------------------------------------

class RemifentanilPKMinto(ThreeCompartmentPK):
    """
    Minto model for Remifentanil.
    """
    def __init__(self, patient: Patient):
        # Minto et al. Anesthesiology. 1997
        age = patient.age
        lbm = patient.lbm
        
        # Clearances [L/min]
        cl1 = 2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55)
        cl2 = 2.05 - 0.0301 * (age - 40)
        cl3 = 0.076 - 0.00113 * (age - 40)
        
        # Volumes [L]
        v1 = 5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55)
        v2 = 9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55)
        v3 = 5.42
        
        # Ke0 [min^-1]
        ke0 = 0.595 - 0.007 * (age - 40)
        
        # Rate constants
        k10 = cl1 / v1
        k12 = cl2 / v1
        k21 = cl2 / v2
        k13 = cl3 / v1
        k31 = cl3 / v3
        
        super().__init__(v1, v2, v3, k10, k12, k21, k13, k31, ke0)


# -----------------------------------------------------------------------------
# Rocuronium Models
# -----------------------------------------------------------------------------

class RocuroniumPK(ThreeCompartmentPK):
    """
    Rocuronium PK Model (Wierda, Szenohradszky, Cooper, Alvarez-Gomez, McCoy).
    3-Compartment (except McCoy 2-comp) with Effect Site.
    """
    def __init__(self, patient: Patient, model_name: str = "Wierda"):
        self.patient = patient
        self.model_name = model_name
        
        age = patient.age
        weight = patient.weight
        # sex: 0 male, 1 female
        sex_code = 0 if patient.sex.lower() == "male" else 1
        age_term = age - 50.0
        
        # 1. Select Base PK Parameters (Adult)
        # ---------------------------------------------------------
        # Model          | v1_bw |  k10  |  k12  |  k21  |  k13  |  k31
        # ---------------------------------------------------------
        # Wierda         | 0.0440| 0.100 | 0.210 | 0.130 | 0.028 | 0.010
        # Szenohradszky  | 0.0769| 0.0376| 0.1143| 0.1748| 0.0196| 0.0189
        # Cooper         | 0.0385| 0.119 | 0.259 | 0.163 | 0.060 | 0.012
        # Alvarez-Gomez  | 0.0570| 0.0952| 0.2807| 0.2149| 0.0322| 0.0166
        # McCoy (2-comp) | 0.0622| 0.0530| 0.0334| 0.0141|     0 |     0
        # ---------------------------------------------------------
        
        if model_name == "Szenohradszky":
            v1_bw = 0.0769; k10 = 0.0376; k12 = 0.1143; k21 = 0.1748; k13 = 0.0196; k31 = 0.0189
            theta4 = 0.247; theta8 = -0.00343
        elif model_name == "Cooper":
            v1_bw = 0.0385; k10 = 0.119; k12 = 0.259; k21 = 0.163; k13 = 0.060; k31 = 0.012
            theta4 = 0.0820; theta8 = -0.00109
        elif model_name == "Alvarez-Gomez":
            v1_bw = 0.0570; k10 = 0.0952; k12 = 0.2807; k21 = 0.2149; k13 = 0.0322; k31 = 0.0166
            theta4 = 0.110; theta8 = -0.00158
        elif model_name == "McCoy":
            v1_bw = 0.0622; k10 = 0.0530; k12 = 0.0334; k21 = 0.0141; k13 = 0.0; k31 = 0.0
            theta4 = 0.113; theta8 = None
        else: # Wierda (Default)
            v1_bw = 0.0440; k10 = 0.100; k12 = 0.210; k21 = 0.130; k13 = 0.028; k31 = 0.010
            theta4 = 0.100; theta8 = -0.00138
            
        # 2. Calculate Real Parameters
        v1 = v1_bw * weight
        
        # Volumes 2 and 3 not explicitly given as V2, V3 but derived from k constants if needed
        # ThreeCompartmentPK requires explicit V2 and V3 values.
        # Calculate V2 and V3 from rate constants (k12, k21, etc.).
        # k21 = Cl2 / V2  => V2 = Cl2 / k21 = (k12 * V1) / k21
        # k31 = Cl3 / V3  => V3 = Cl3 / k31 = (k13 * V1) / k31
        
        # Avoid division by zero for McCoy
        v2 = (k12 * v1) / k21 if k21 > 0 else 0.0
        v3 = (k13 * v1) / k31 if k31 > 0 else 0.0

        # Hepatic impairment: increase Vd with preserved clearances
        v_scale = hepatic_vd_multiplier(patient, severe_multiplier=1.4)
        if v_scale != 1.0:
            cl1 = k10 * v1
            cl2 = k12 * v1
            cl3 = k13 * v1
            v1 *= v_scale
            v2 *= v_scale
            v3 *= v_scale
            k10 = cl1 / v1
            k12 = cl2 / v1
            k21 = cl2 / v2 if v2 > 0 else 0.0
            k13 = cl3 / v1
            k31 = cl3 / v3 if v3 > 0 else 0.0
        
        # 3. Effect Site (ke0)
        # Masui Covariate Model: ke0(age) = theta4 + theta8 * (age - 50)
        # If theta8 is None, ke0 = theta4
        
        if theta8 is not None:
            ke0 = theta4 + theta8 * age_term
        else:
            ke0 = theta4
            
        # Floor ke0 to avoid negative or zero values
        ke0 = max(ke0, 0.01)

        # Renal impairment: reduce clearance (k10)
        clearance_scale = organ_clearance_scaler(patient, hepatic_fraction=0.0, renal_fraction=0.6)
        k10 *= clearance_scale
        
        super().__init__(v1, v2, v3, k10, k12, k21, k13, k31, ke0)


# -----------------------------------------------------------------------------
# Vasopressor State (Unified for Norepinephrine, Epinephrine, Phenylephrine)
# -----------------------------------------------------------------------------

@dataclass
class VasopressorState:
    """
    Unified state for vasopressor PK models.
    
    All vasopressors use this common state to ensure consistent interface.
    """
    c1: float = 0.0   # Central/plasma concentration (ng/mL)
    c2: float = 0.0   # Peripheral compartment (if 2-compartment model)
    ce: float = 0.0   # Effect-site concentration (ng/mL)


# -----------------------------------------------------------------------------
# Norepinephrine Models
# -----------------------------------------------------------------------------

class NorepinephrinePK:
    """
    Norepinehprine PK Models (Beloeil, Oualha, Li).
    """
    def __init__(self, patient: Patient, model: str = "Beloeil"):
        self.model = model
        self.patient = patient
        w = patient.weight
        age = patient.age

        self._is_oualha = (model == "Oualha")
        self._is_li = (model == "Li")
        
        self.u_endo = 0.0 # Endogenous production (ug/min)
        self.u_endo_ug_min = 0.0
        self.c_prop_base = 3.53 # for Li model
        self.k12_base = 0.0
        self.v2 = 0.0
        self.cl2 = 0.0
        self.k12 = 0.0
        self.k21 = 0.0
        self.v2_base_ref = 0.0
        
        # Initialize parameters
        if model == "Beloeil":
            # Beloeil et al. Br J Anaesth. 2005;95:782-788.
            # CL is a function of SAPS II severity score. This implementation assumes
            # SAPS II = 30 (moderate severity), which could be exposed as a parameter.
            self.v1 = 8.840  # L
            self.cl1 = 59.6 / 30.0  # L/min (CL = 59.6/SAPS_II from Beloeil)
            self.v2 = 0.0
            self.cl2 = 0.0
            self.k10 = self.cl1 / self.v1
            
        elif model == "Oualha":
            # Oualha et al. Br J Clin Pharmacol. 2014 (children)
            self.v1 = 0.08 * w
            self.cl1 = 0.11 * (w**0.75)
            self.u_endo = 0.052 * (w**0.75) # ug/min
            
            self.v2 = 0.0
            self.cl2 = 0.0
            self.k10 = self.cl1 / self.v1
            
        elif model == "Li":
            # Li et al. Clin Pharmacokinet. 2024
            self.cl1 = 2.1 * np.exp(-0.377/100 * (age - 35)) * (w/70)**0.75
            self.v1 = 2.4 * (w/70)
            self.cl2 = 0.6 * (w/70)**0.75
            self.v2 = 3.6 * (w/70)
            self.v2_base_ref = self.v2  # cache for hemodynamic scaling
            
            # Endogenous: PAS converts to ug/s and adds to input.
            self.u_endo_ug_min = (497.7 * (w/70)**0.75) / 1000.0 
            
            self.k10 = self.cl1 / self.v1
            self.k12 = self.cl2 / self.v1
            self.k21 = self.cl2 / self.v2
            self.k12_base = self.k12
        
        # Effect-site equilibration rate constant
        # ~1-2 min to peak effect for norepinephrine
        self.ke0 = 0.4  # min^-1 (t_peak ~1.7 min)
            
        self.state = VasopressorState()
        
        # Store baselines
        self.v1_base = self.v1
        self.cl1_base = self.cl1
        self.v2_base = self.v2
        self.cl2_base = self.cl2
        self.k10_base = self.k10
        self.k12_base = self.k12
        self.k21_base = self.k21
        
        # Init state with endogenous equilibrium if applicable
        if self.u_endo > 0 or (self._is_li and self.u_endo_ug_min > 0):
             # Steady state: Rin_total = Cl * C_ss => C_ss = Rin_endo / Cl
             endo = self.u_endo if self._is_oualha else self.u_endo_ug_min
             self.state.c1 = endo / self.cl1 # ug/L = ng/mL
             self.state.ce = self.state.c1  # Effect-site at equilibrium
             if self.v2 > 0:
                 self.state.c2 = self.state.c1

    def update_hemodynamics(self, v_ratio: float, co_ratio: float):
        """
        v_ratio = current_vol / initial_vol
        co_ratio = current_co / initial_co
        """
        # Update V1
        self.v1 = self.v1_base * v_ratio
        
        # Update Cl
        current_cl1 = self.cl1_base * co_ratio
        self.cl1 = current_cl1
        
        self.k10 = self.cl1 / self.v1
        
        if self._is_li:
            current_cl2 = self.cl2_base * co_ratio
            self.cl2 = current_cl2
            
            # Scale V2 proportionally with blood volume
            self.v2 = self.v2_base_ref * v_ratio
                 
            self.k12 = self.cl2 / self.v1
            self.k21 = self.cl2 / self.v2


    def step(self, dt_sec: float, infusion_rate_ug_sec: float, propofol_conc_ug_ml: float = 0.0) -> VasopressorState:
        """
        infusion_rate_ug_sec: ug/s
        """
        dt_min = dt_sec / 60.0
        rin_ug_min = infusion_rate_ug_sec * 60.0
        state = self.state
        
        # Add endogenous
        if self._is_oualha:
            rin_ug_min += self.u_endo
        elif self._is_li:
            rin_ug_min += self.u_endo_ug_min
            
        # Update Li model parameters based on Propofol
        if self._is_li:
            # Propofol effect on NE clearance (Li et al. Clin Pharmacokinet. 2024)
            # The formula exp(-3.57 * (Cp - 3.53)) causes massive values at Cp=0 (Awake).
            # Clamp current_cl1 to a physiological ceiling (5.0 L/min) to prevent instant elimination.
            # 5.0 L/min is approx 2.5x baseline clearance, providing t1/2 ~0.5 min instead of 0.05s.
            prop_effect = np.exp(-3.57 * (propofol_conc_ug_ml - self.c_prop_base))
            current_cl1 = self.cl1 * prop_effect
            if current_cl1 > 5.0: current_cl1 = 5.0
            
            current_k10 = current_cl1 / self.v1
        else:
            current_k10 = self.k10
        c1 = state.c1 # ng/mL
        rin_term = rin_ug_min / self.v1
        
        if self._is_li:
            c2 = state.c2
            
            gradient_c1 = -(current_k10 + self.k12) * c1 + self.k21 * c2 + rin_term
            gradient_c2 = self.k12 * c1 - self.k21 * c2
            
            state.c1 += gradient_c1 * dt_min
            state.c2 += gradient_c2 * dt_min
        else:
            # dC1 = -k10 C1 + Rin/V1
            gradient_c1 = -current_k10 * c1 + rin_term
            state.c1 += gradient_c1 * dt_min
        
        # Effect-site equilibration (all models)
        gradient_ce = self.ke0 * (state.c1 - state.ce)
        state.ce += gradient_ce * dt_min
        state.ce = max(0.0, state.ce)
            
        return state

    def reset(self):
        self.state = VasopressorState()

    def get_ss_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return continuous A, B matrices (units 1/min).
        State: [c1, c2] (c2 is 0 if 1-comp)
        """
        # For TCI, use baseline k10 (Li model has dynamic k10 based on Propofol)
        k10 = self.k10
        
        if self._is_li:
            # 2-compartment state-space model
            a11 = -(k10 + self.k12)
            a12 = self.k21
            a21 = self.k12
            a22 = -self.k21
            
            A = np.array([
                [a11, a12],
                [a21, a22]
            ])
            
            B = np.array([
                [1.0 / self.v1],
                [0.0]
            ])
        else:
            # 1-compartment model
            A = np.array([[-k10]])
            B = np.array([[1.0 / self.v1]])
            
        return A, B


# -----------------------------------------------------------------------------
# Epinephrine PK Model
# -----------------------------------------------------------------------------

class EpinephrinePK:
    """
    Epinephrine (Adrenaline) Pharmacokinetic Model.
    
    Population PK is typically 1-compartment with an effect-site for dynamics.
    Default model: Clutter et al. (healthy adults).
    Optional adult septic shock model: Abboud et al.
    Optional pediatric model: Oualha et al.
    """
    def __init__(self, patient: Patient, model: str = "Clutter", sapsii: float = None):
        self.patient = patient
        self.model = model
        w = patient.weight

        if model == "Oualha":
            # Oualha et al. 2014 (peds): V = 0.08 * BW; CL = 2.00 * BW^0.75 (L/hr)
            self.v1 = 0.08 * w
            cl_l_hr = 2.00 * (w ** 0.75)
        elif model == "Abboud":
            # Abboud et al. 2009 (adult septic shock): CL = 127*(BW/70)^0.60*(SAPSII/50)^(-0.67); V ~ 7.9 L
            self.v1 = 7.9
            if sapsii is None:
                sapsii = self._estimate_sapsii(patient)
            cl_l_hr = 127.0 * (w / 70.0) ** 0.60 * (sapsii / 50.0) ** (-0.67)
        else:
            # Clutter et al. 1980 (healthy adults): clearance ~52-89 mL/min/kg
            # Use 70 mL/min/kg as a representative baseline.
            self.v1 = 0.15 * w
            cl_l_hr = 0.07 * w * 60.0
        
        self.cl1 = cl_l_hr / 60.0  # L/min
        self.k10 = self.cl1 / self.v1
        
        # Effect-site equilibration rate constant
        # Epinephrine effects peak within ~1 min of IV bolus
        self.ke0 = 0.5  # min^-1 (t_peak ~1.4 min)
        
        # Store baselines
        self.v1_base = self.v1
        self.cl1_base = self.cl1
        self.k10_base = self.k10
        
        self.state = VasopressorState()

    @staticmethod
    def _estimate_sapsii(patient: Patient) -> float:
        """
        Estimate SAPS II using available baseline data and reasonable defaults
        for unavailable labs/vitals (anesthesia/non-ICU context).
        """
        age = patient.age
        hr = patient.baseline_hr
        map_val = patient.baseline_map
        temp_c = patient.baseline_temp

        # Approximate SBP from MAP with a typical pulse pressure of ~40 mmHg.
        sbp = map_val + 27.0

        score = 0

        # Age
        if age < 40:
            score += 0
        elif age < 60:
            score += 7
        elif age < 70:
            score += 12
        elif age < 75:
            score += 15
        elif age < 80:
            score += 16
        else:
            score += 18

        # Heart rate
        if hr < 40:
            score += 11
        elif hr < 70:
            score += 2
        elif hr < 120:
            score += 0
        elif hr < 160:
            score += 4
        else:
            score += 7

        # Systolic BP
        if sbp < 70:
            score += 13
        elif sbp < 100:
            score += 5
        elif sbp < 200:
            score += 0
        else:
            score += 2

        # Temperature >= 39 C
        if temp_c >= 39.0:
            score += 3

        # GCS, PaO2/FiO2, BUN, urine output, sodium, potassium, bicarbonate,
        # bilirubin, WBC, chronic disease, admission type:
        # assume normal/low-risk values in anesthesia context.

        return float(score)
    
    def update_hemodynamics(self, v_ratio: float, co_ratio: float):
        """Scale PK parameters based on hemodynamic changes."""
        self.v1 = self.v1_base * v_ratio
        self.cl1 = self.cl1_base * co_ratio
        self.k10 = self.cl1 / self.v1
    
    def step(self, dt_sec: float, infusion_rate_ug_sec: float) -> VasopressorState:
        """
        Advance PK state by dt_sec.
        
        Args:
            dt_sec: Time step in seconds
            infusion_rate_ug_sec: Infusion rate in µg/sec
        """
        dt_min = dt_sec / 60.0
        rin_ug_min = infusion_rate_ug_sec * 60.0
        state = self.state
        c1 = state.c1
        ce = state.ce
        
        # dC1/dt = -k10 * C1 + Rin/V1
        # Units: ng/mL (µg/L)
        gradient_c1 = -self.k10 * c1 + (rin_ug_min / self.v1)
        state.c1 += gradient_c1 * dt_min
        state.c1 = max(0.0, state.c1)
        
        # Effect-site equilibration
        gradient_ce = self.ke0 * (state.c1 - ce)
        state.ce += gradient_ce * dt_min
        state.ce = max(0.0, state.ce)
        
        return state
    
    
    def get_ss_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return state-space matrices (A, B) for TCI control.
        1-Compartment Model:
        dX/dt = -k10*X + Rin
        dC/dt = -k10*C + Rin/V1
        
        State vector x = [C1]
        A = [-k10] (units: min^-1)
        B = [1/V1] (units: L^-1) -> if Rin is mass/min, B*Rin = conc/min
        """
        A = np.array([[-self.k10]])
        B = np.array([[1.0 / self.v1]])
        return A, B
    
    def reset(self):
        self.state = VasopressorState()


# -----------------------------------------------------------------------------
# Phenylephrine PK Model
# -----------------------------------------------------------------------------

class PhenylephrinePK:
    """
    Phenylephrine 2-Compartment Pharmacokinetic Model.
    
    Based on FDA NDA 203826 Clinical Pharmacology Review (2012),
    which re-analyzed data from Hengstmann & Goronzy (Eur J Clin Pharmacol. 1982).
    are for IV phenylephrine in healthy adult volunteers.
    
    Two-compartment structure with:
    - Fast redistribution phase: t½,α ≈ 2.3 min
    - Slow elimination phase: t½,β ≈ 53 min
    
    This produces realistic behavior:
    - Rapid onset (<5 min to peak effect)
    - ~80% plasma decay within ~10 min after stopping short infusion
    - Long low-level tail during prolonged infusions
    
    Effect-site assumes minimal hysteresis per NDA review (direct effect),
    Effective ke0 is added for numerical stability and realistic bolus response.
    
    References:
    - FDA NDA 203826 Clinical Pharmacology Review. 2012.
    - Hengstmann & Goronzy. Eur J Clin Pharmacol. 1982.
    
    Units:
    - Volumes: L
    - Rate constants: min^-1
    - Concentrations: ng/mL (µg/L)
    """
    def __init__(self, patient: Patient):
        self.patient = patient
        w = patient.weight
        
        # Weight scaling factor (reference: 70 kg adult)
        wt_scale = w / 70.0
        
        # Compartment volumes (FDA NDA 203826 parameters, weight-scaled)
        # Reference values: V1 = 20.4 L, V2 = 100 L (Vss ≈ 121 L)
        self.v1 = 20.4 * wt_scale  # Central compartment (L)
        self.v2 = 100.0 * wt_scale  # Peripheral compartment (L)
        
        # Micro-rate constants (min^-1) from FDA re-analysis
        self.k10 = 0.124   # Elimination from central
        self.k12 = 0.155   # Distribution central → peripheral
        self.k21 = 0.0314  # Return peripheral → central
        
        # Derived clearances for hemodynamic scaling
        # CL = k10 × V1 ≈ 2.53 L/min (at 70 kg)
        # Q = k12 × V1 ≈ 3.16 L/min (at 70 kg)
        self.cl1 = self.k10 * self.v1  # Systemic clearance
        self.cl2 = self.k12 * self.v1  # Intercompartmental clearance
        
        # Effect-site equilibration rate constant
        # NDA indicates minimal hysteresis; ke0 ≈ 0.7 min^-1 gives
        # effect-site t½ ≈ 1 min for realistic bolus response
        self.ke0 = 0.7  # min^-1
        
        # Store baselines for hemodynamic scaling
        self.v1_base = self.v1
        self.v2_base = self.v2
        self.cl1_base = self.cl1
        self.cl2_base = self.cl2
        self.k10_base = self.k10
        self.k12_base = self.k12
        self.k21_base = self.k21
        
        self.state = VasopressorState()
    
    def update_hemodynamics(self, v_ratio: float, co_ratio: float):
        """
        Scale PK parameters based on hemodynamic changes.
        
        Args:
            v_ratio: current_blood_vol / initial_blood_vol
            co_ratio: current_CO / initial_CO
        """
        # Scale volumes by blood volume ratio
        self.v1 = self.v1_base * v_ratio
        self.v2 = self.v2_base * v_ratio
        
        # Scale clearances by cardiac output ratio
        self.cl1 = self.cl1_base * co_ratio
        self.cl2 = self.cl2_base * co_ratio
        
        # Recalculate rate constants
        self.k10 = self.cl1 / self.v1
        self.k12 = self.cl2 / self.v1
        self.k21 = self.cl2 / self.v2
    
    def step(self, dt_sec: float, infusion_rate_ug_sec: float) -> VasopressorState:
        """
        Advance PK state by dt_sec using 2-compartment model.
        
        Args:
            dt_sec: Time step in seconds
            infusion_rate_ug_sec: Infusion rate in µg/sec
            
        Returns:
            Updated VasopressorState with c1 (central), c2 (peripheral), ce (effect-site)
        """
        dt_min = dt_sec / 60.0
        rin_ug_min = infusion_rate_ug_sec * 60.0
        state = self.state
        c1 = state.c1
        c2 = state.c2
        ce = state.ce
        
        # 2-Compartment ODEs (concentration formulation)
        # dC1/dt = -(k10 + k12)*C1 + k21*C2 + Rin/V1
        # dC2/dt = k12*C1 - k21*C2
        k_sum = self.k10 + self.k12
        gradient_c1 = -k_sum * c1 + self.k21 * c2 + (rin_ug_min / self.v1)
        gradient_c2 = self.k12 * c1 - self.k21 * c2
        
        # Effect-site equilibration
        # dCe/dt = ke0 * (C1 - Ce)
        gradient_ce = self.ke0 * (c1 - ce)
        
        # Euler integration
        state.c1 += gradient_c1 * dt_min
        state.c2 += gradient_c2 * dt_min
        state.ce += gradient_ce * dt_min
        
        # Ensure non-negative concentrations
        state.c1 = max(0.0, state.c1)
        state.c2 = max(0.0, state.c2)
        state.ce = max(0.0, state.ce)
        
        return state

    def get_ss_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return state-space matrices (A, B) for TCI control.
        
        2-Compartment model with effect-site:
        State vector: [C1, C2, Ce]
        Input: Rin (µg/min)
        
        Returns:
            A: 3x3 state matrix (min^-1)
            B: 3x1 input matrix
        """
        # A matrix
        a11 = -(self.k10 + self.k12)
        a12 = self.k21
        a13 = 0.0
        
        a21 = self.k12
        a22 = -self.k21
        a23 = 0.0
        
        a31 = self.ke0
        a32 = 0.0
        a33 = -self.ke0
        
        A = np.array([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]
        ])
        
        # B vector (input goes to central compartment only)
        B = np.array([
            [1.0 / self.v1],
            [0.0],
            [0.0]
        ])
        
        return A, B
    
    def reset(self):
        self.state = VasopressorState()
