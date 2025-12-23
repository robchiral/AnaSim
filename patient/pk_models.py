import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from .patient import Patient

# =============================================================================
# PHARMACOKINETIC MODELS - LITERATURE REFERENCES
# =============================================================================
#
# This module implements 3-compartment mammillary PK models for anesthetic drugs.
#
# Propofol Models:
#   - Marsh: Marsh B et al. Br J Anaesth 1991; Weight-based model
#   - Schnider: Schnider TW et al. Anesthesiology 1998; Age, LBM, height adjusted
#   - Eleveld: Eleveld DJ et al. Br J Anaesth 2018; Population PK meta-analysis, allometric scaling
#
# Remifentanil:
#   - Minto: Minto CF et al. Anesthesiology 1997; Age and LBM adjusted
#
# Rocuronium:
#   - Wierda JM: Wierda JM et al. Anesthesiology 1991; Standard PK parameters
#
# Vasopressors:
#   - Norepinephrine:
#       - Beloeil H et al. Br J Anaesth 2005;95:782-788 (septic shock/trauma, 1-comp PK-PD)
#       - Oualha M et al. Br J Clin Pharmacol 2014;78:886-897 (critically ill children, popPK-PD)
#       - Li Y et al. Clin Pharmacokinet 2024;63:1597-1608 (healthy volunteers, 2-comp with propofol interaction)
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
        # k10 = cl1 / v1
        self.k10 = cl1 / self.v1
        
        # k12 = cl2 / v1
        self.k12 = cl2 / self.v1
        
        # k21 = cl2 / v2 (V2 constant)
        self.k21 = cl2 / self.v2
        
        # k13 = cl3 / v1
        self.k13 = cl3 / self.v1
        
        # k31 = cl3 / v3 (V3 constant)
        self.k31 = cl3 / self.v3

    def step(self, dt_sec: float, infusion_rate_mg_sec: float) -> PKState:
        """
        Advance the PK state by dt_sec.
        
        Args:
            dt_sec: Time step in seconds.
            infusion_rate_mg_sec: Infusion rate in mg/sec (or units consistent with V1).
        """
        # Convert dt to minutes for rate constants (which are usually 1/min)
        dt_min = dt_sec / 60.0
        
        # Infusion rate usually mg/sec input, need to check units.
        # If parameters are in Liters and min^-1, and C in mg/L (ug/mL).
        # Rate Rin should be in mg/min for the ODE if dt is min.
        # Rin(t)/V1 term: (mg/min) / L = mg/L/min.
        
        # Input infusion_rate is likely in mg/sec from the pump.
        rin_mg_min = infusion_rate_mg_sec * 60.0
        
        c1, c2, c3, ce = self.state.c1, self.state.c2, self.state.c3, self.state.ce
        
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
        
        gradient_c1 = -(self.k10 + self.k12 + self.k13) * c1 + \
                       self.k21 * (self.v2 / self.v1) * c2 + \
                       self.k31 * (self.v3 / self.v1) * c3 + \
                       (rin_mg_min / self.v1)
                       
        gradient_c2 = self.k12 * (self.v1 / self.v2) * c1 - self.k21 * c2
                       
        gradient_c3 = self.k13 * (self.v1 / self.v3) * c1 - self.k31 * c3
                       
        gradient_ce = self.ke0 * (c1 - ce)
        
        self.state.c1 += gradient_c1 * dt_min
        self.state.c2 += gradient_c2 * dt_min
        self.state.c3 += gradient_c3 * dt_min
        self.state.ce += gradient_ce * dt_min
        
        return self.state

    def reset(self):
        self.state = PKState()

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
        a12 = self.k21 * (self.v2 / self.v1)
        a13 = self.k31 * (self.v3 / self.v1)
        a14 = 0.0
        
        # dC2/dt
        a21 = self.k12 * (self.v1 / self.v2)
        a22 = -self.k21
        a23 = 0.0
        a24 = 0.0
        
        # dC3/dt
        a31 = self.k13 * (self.v1 / self.v3)
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
        # But we usually supply Rin in mg/sec or ug/min.
        # If input u is in "Mass/Time" (e.g. mg/min), then B = [1/V1, 0, 0, 0].
        # The units of A and B must be consistent with time unit of simulation.
        # If simulation dt is passed to TCI in seconds, TCI usually discretizes A (1/min) -> Ad.
        # So we return A in 1/min units.
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
        # Marsh B et al. Br J Anaesth 1991
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
        
        # Ke0:
        # Original Marsh B et al. 1991 PK paper does not specify ke0.
        # "Marsh modified" for effect-site TCI commonly uses ke0 = 1.2 min^-1.
        # This value is from clinical TCI implementations, not the original publication.
        ke0 = 1.2
        
        super().__init__(v1, v2, v3, k10, k12, k21, k13, k31, ke0)


class PropofolPKSchnider(ThreeCompartmentPK):
    """
    Schnider model for Propofol.
    """
    def __init__(self, patient: Patient):
        # Schnider TW et al. Anesthesiology 1998
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
    Eleveld DJ et al. General Purpose PK-PD Model for Propofol (Br J Anaesth 2018).
    """
    def __init__(self, patient: Patient):
        # Eleveld DJ et al. 2018
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
        
        # Ke0
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
        # Minto CF et al. Anesthesiology 1997
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
        # But ThreeCompartmentPK expects V2, V3.
        # We can calculate V2, V3 from k12, k21 etc.
        # k21 = Cl2 / V2  => V2 = Cl2 / k21 = (k12 * V1) / k21
        # k31 = Cl3 / V3  => V3 = Cl3 / k31 = (k13 * V1) / k31
        
        # Avoid division by zero for McCoy
        v2 = (k12 * v1) / k21 if k21 > 0 else 0.0
        v3 = (k13 * v1) / k31 if k31 > 0 else 0.0
        
        # 3. Effect Site (ke0)
        # Masui Covariate Model: ke0(age) = theta4 + theta8 * (age - 50)
        # If theta8 is None, ke0 = theta4
        
        if theta8 is not None:
            ke0 = theta4 + theta8 * age_term
        else:
            ke0 = theta4
            
        # Floor ke0 to avoid negative or zero values
        ke0 = max(ke0, 0.01)
        
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
        
        self.u_endo = 0.0 # Endogenous production (ug/min)
        self.c_prop_base = 3.53 # for Li model
        self.k12_base = 0.0
        
        # Initialize parameters
        if model == "Beloeil":
            # Beloeil H et al. "Norepinephrine kinetics and dynamics in septic shock
            # and trauma patients", Br J Anaesth 2005;95:782-788
            # Note: CL is function of SAPS II severity score. Here we assume
            # SAPS II = 30 (moderate severity). Consider exposing this as a parameter.
            self.v1 = 8.840  # L
            self.cl1 = 59.6 / 30.0  # L/min (CL = 59.6/SAPS_II from Beloeil)
            self.v2 = 0.0
            self.cl2 = 0.0
            self.k10 = self.cl1 / self.v1
            
        elif model == "Oualha":
            # Oualha M et al. Br J Clin Pharmacol 2014 (Children)
            self.v1 = 0.08 * w
            self.cl1 = 0.11 * (w**0.75)
            self.u_endo = 0.052 * (w**0.75) # ug/min
            
            self.v2 = 0.0
            self.cl2 = 0.0
            self.k10 = self.cl1 / self.v1
            
        elif model == "Li":
            # Li Y et al. Clin Pharmacokinet 2024
            self.cl1 = 2.1 * np.exp(-0.377/100 * (age - 35)) * (w/70)**0.75
            self.v1 = 2.4 * (w/70)
            self.cl2 = 0.6 * (w/70)**0.75
            self.v2 = 3.6 * (w/70)
            
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
        if hasattr(self, 'cl2'): self.cl2_base = self.cl2
        if hasattr(self, 'k12'): self.k12_base = self.k12
        if hasattr(self, 'k10'): self.k10_base = self.k10
        if hasattr(self, 'k21'): self.k21_base = self.k21
        
        # Init state with endogenous equilibrium if applicable
        if self.u_endo > 0 or (model == "Li" and self.u_endo_ug_min > 0):
             # Steady state: Rin_total = Cl * C_ss => C_ss = Rin_endo / Cl
             endo = self.u_endo if model == "Oualha" else self.u_endo_ug_min
             self.state.c1 = endo / self.cl1 # ug/L = ng/mL
             self.state.ce = self.state.c1  # Effect-site at equilibrium
             if hasattr(self, 'v2') and self.v2 > 0:
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
        
        if self.model == "Li":
            current_cl2 = self.cl2_base * co_ratio
            self.cl2 = current_cl2
            
            # Scale V2 proportionally with blood volume
            v2_base = 3.6 * (self.patient.weight / 70)
            self.v2 = v2_base * v_ratio
                 
            self.k12 = self.cl2 / self.v1
            self.k21 = self.cl2 / self.v2


    def step(self, dt_sec: float, infusion_rate_ug_sec: float, propofol_conc_ug_ml: float = 0.0) -> VasopressorState:
        """
        infusion_rate_ug_sec: ug/s
        """
        dt_min = dt_sec / 60.0
        rin_ug_min = infusion_rate_ug_sec * 60.0
        
        # Add endogenous
        if self.model == "Oualha":
            rin_ug_min += self.u_endo
        elif self.model == "Li":
            rin_ug_min += self.u_endo_ug_min
            
        # Update Li model parameters based on Propofol
        if self.model == "Li":
            # Propofol effect on NN clearance (Li Y et al. 2024)
            # The formula exp(-3.57 * (Cp - 3.53)) causes massive values at Cp=0 (Awake).
            # We clamp current_cl1 to a physiological ceiling (5.0 L/min) to prevent instant elimination.
            # 5.0 L/min is approx 2.5x baseline clearance, providing t1/2 ~0.5 min instead of 0.05s.
            prop_effect = np.exp(-3.57 * (propofol_conc_ug_ml - self.c_prop_base))
            current_cl1 = self.cl1 * prop_effect
            if current_cl1 > 5.0: current_cl1 = 5.0
            
            current_k10 = current_cl1 / self.v1
        else:
            current_k10 = self.k10

        c1 = self.state.c1 # ng/mL
        
        if self.model == "Beloeil" or self.model == "Oualha":
            # dC1 = -k10 C1 + Rin/V1
            gradient_c1 = -current_k10 * c1 + (rin_ug_min / self.v1)
            self.state.c1 += gradient_c1 * dt_min
            
        elif self.model == "Li":
            c2 = self.state.c2
            
            gradient_c1 = -(current_k10 + self.k12) * c1 + \
                           self.k21 * c2 + \
                           (rin_ug_min / self.v1)
                           
            gradient_c2 = self.k12 * c1 - self.k21 * c2
            
            self.state.c1 += gradient_c1 * dt_min
            self.state.c2 += gradient_c2 * dt_min
        
        # Effect-site equilibration (all models)
        gradient_ce = self.ke0 * (self.state.c1 - self.state.ce)
        self.state.ce += gradient_ce * dt_min
        self.state.ce = max(0.0, self.state.ce)
            
        return self.state

    def reset(self):
        self.state = VasopressorState()

    def get_ss_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return continuous A, B matrices (units 1/min).
        State: [c1, c2] (c2 is 0 if 1-comp)
        """
        # For TCI, use baseline k10 (Li model has dynamic k10 based on Propofol)
        k10 = self.k10
        
        if self.model == "Li":
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
    
    HEURISTIC one-compartment model with effect-site. Parameters are calibrated
    to produce clinically-reasonable behavior (t1/2 ~2.5 min, Vd ~0.15 L/kg,
    peak effect ~1 min) based on general pharmacology references and clinical
    experience, NOT from a specific published population PK model.
    
    Rapid metabolism by COMT/MAO gives very short apparent half-life.
    Effect-site adds equilibration delay for realistic bolus response.
    
    Note: If higher fidelity is required, parameters should be replaced with
    values from a formal multi-compartment PK study.
    """
    def __init__(self, patient: Patient):
        self.patient = patient
        w = patient.weight
        
        # Volume of distribution: ~0.15-0.20 L/kg (hydrophilic)
        self.v1 = 0.15 * w  # L
        
        # Clearance: Based on t1/2 = 2.5 min
        # k10 = ln(2) / t1/2 = 0.693 / 2.5 = 0.277 min^-1
        # Cl = k10 * V1
        self.k10 = 0.277  # min^-1
        self.cl1 = self.k10 * self.v1  # L/min
        
        # Effect-site equilibration rate constant
        # Epinephrine effects peak within ~1 min of IV bolus
        self.ke0 = 0.5  # min^-1 (t_peak ~1.4 min)
        
        # Store baselines
        self.v1_base = self.v1
        self.cl1_base = self.cl1
        self.k10_base = self.k10
        
        self.state = VasopressorState()
    
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
        
        c1 = self.state.c1
        ce = self.state.ce
        
        # dC1/dt = -k10 * C1 + Rin/V1
        # Units: ng/mL (µg/L)
        gradient_c1 = -self.k10 * c1 + (rin_ug_min / self.v1)
        self.state.c1 += gradient_c1 * dt_min
        self.state.c1 = max(0.0, self.state.c1)
        
        # Effect-site equilibration
        gradient_ce = self.ke0 * (self.state.c1 - ce)
        self.state.ce += gradient_ce * dt_min
        self.state.ce = max(0.0, self.state.ce)
        
        return self.state
    
    
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
    
    Based on FDA NDA 203826 Clinical Pharmacology Review, which re-analyzed
    data from Hengstmann & Goronzy (Eur J Clin Pharmacol 1982). Parameters
    are for IV phenylephrine in healthy adult volunteers.
    
    Two-compartment structure with:
    - Fast redistribution phase: t½,α ≈ 2.3 min
    - Slow elimination phase: t½,β ≈ 53 min
    
    This produces realistic behavior:
    - Rapid onset (<5 min to peak effect)
    - ~80% plasma decay within ~10 min after stopping short infusion
    - Long low-level tail during prolonged infusions
    
    Effect-site assumes minimal hysteresis per NDA review (direct effect),
    but ke0 added for numerical stability and realistic bolus response.
    
    References:
    - FDA NDA 203826Orig1s000 Clinical Pharmacology Review (2012)
    - Hengstmann JH, Goronzy J. Eur J Clin Pharmacol 1982;21:335-341
    
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
        
        c1 = self.state.c1
        c2 = self.state.c2
        ce = self.state.ce
        
        # 2-Compartment ODEs (concentration formulation)
        # dC1/dt = -(k10 + k12)*C1 + k21*C2 + Rin/V1
        # dC2/dt = k12*C1 - k21*C2
        gradient_c1 = -(self.k10 + self.k12) * c1 + self.k21 * c2 + (rin_ug_min / self.v1)
        gradient_c2 = self.k12 * c1 - self.k21 * c2
        
        # Effect-site equilibration
        # dCe/dt = ke0 * (C1 - Ce)
        gradient_ce = self.ke0 * (c1 - ce)
        
        # Euler integration
        self.state.c1 += gradient_c1 * dt_min
        self.state.c2 += gradient_c2 * dt_min
        self.state.ce += gradient_ce * dt_min
        
        # Ensure non-negative concentrations
        self.state.c1 = max(0.0, self.state.c1)
        self.state.c2 = max(0.0, self.state.c2)
        self.state.ce = max(0.0, self.state.ce)
        
        return self.state

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
