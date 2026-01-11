from dataclasses import dataclass
from .patient import Patient

@dataclass
class VolatileState:
    """
    Volatile anesthetic state using partial pressures throughout.
    
    All values are expressed as fractional partial pressures (0.0-1.0 range,
    multiply by 100 for percentage). The advantage of using partial pressures
    is that at equilibrium, P_arterial = P_tissue for all compartments.
    """
    p_alv: float = 0.0   # Alveolar partial pressure (fraction, 0.02 = 2%)
    p_art: float = 0.0   # Arterial partial pressure (≈ p_alv at steady state)
    p_ven: float = 0.0   # Mixed venous partial pressure
    p_vrg: float = 0.0   # Vessel-Rich Group (brain, heart, kidneys) partial pressure
    p_mus: float = 0.0   # Muscle partial pressure  
    p_fat: float = 0.0   # Fat partial pressure
    mac: float = 0.0     # Current MAC fraction (p_vrg / MAC_age_corrected)

class VolatilePK:
    """
    Physiologically Based PK (PBPK) model for Volatile Agents.
    4 Compartments: Lungs, VRG, Muscle, Fat.
    """
    def __init__(
        self,
        patient: Patient,
        name: str,
        lambda_b_g: float,
        lambda_t_b_vrg: float = 1.6,
        mac_40: float = 2.0,
        lambda_t_b_mus: float = None,
        lambda_t_b_fat: float = None,
    ):
        self.patient = patient
        self.name = name
        
        # Physicochemical properties
        self.lambda_b_g = lambda_b_g  # Blood:Gas partition coef
        self.lambda_t_b_vrg = lambda_t_b_vrg # Tissue:Blood VRG
        # Generic defaults for others if not specified
        self.lambda_t_b_mus = 2.5 if lambda_t_b_mus is None else lambda_t_b_mus
        self.lambda_t_b_fat = 50.0 if lambda_t_b_fat is None else lambda_t_b_fat
        
        # MAC (Age corrected)
        # Mapleson age correction for MAC.
        # Reference: Mapleson. Br J Anaesth. 1996.
        self.mac_40 = mac_40 
        self.mac_age = self.mac_40 * (10 ** (-0.00269 * (self.patient.age - 40)))
        
        # Volumes (L)
        # VRG ~ 10% weight, Muscle ~ 50%, Fat ~ 20%
        w_kg = self.patient.weight
        self.v_vrg = 0.1 * w_kg
        self.v_mus = 0.5 * w_kg
        self.v_fat = 0.2 * w_kg
        
        # Flows (Fraction of CO)
        self.f_vrg_frac = 0.75
        self.f_mus_frac = 0.19
        self.f_fat_frac = 0.06
        
        self.state = VolatileState()
        
    def step(self, dt: float, fi_agent: float, alveolar_vent_l: float, cardiac_output_l: float, temp_c: float = 37.0):
        """
        dt: seconds
        fi_agent: Fraction inspired (0-1)
        alveolar_vent_l: L/min of alveolar ventilation
        cardiac_output_l: L/min
        temp_c: Patient temperature (deg C)
        """
        state = self.state
        dt_min = dt / 60.0

        # Flows L/min
        q_co = cardiac_output_l
        q_vrg = q_co * self.f_vrg_frac
        q_mus = q_co * self.f_mus_frac
        q_fat = q_co * self.f_fat_frac
        
        alveolar_vent = max(0.0, alveolar_vent_l)
        
        # --- Gas Exchange (Lung) ---
        # dPalv/dt = (VA*(Fi - Palv) + Q*Lambda_bg*(Pven - Palv)) / V_FRC
        # Pressures in fraction (0-1)
        v_frc = 2.5  # Functional Residual Capacity (L)
        lambda_q = q_co * self.lambda_b_g
        
        # Current Pressures
        p_alv = state.p_alv
        p_ven = state.p_ven
        
        # Derivative
        dP_dt = (alveolar_vent * (fi_agent - p_alv) + lambda_q * (p_ven - p_alv)) / v_frc
        
        # Update Alveolar
        p_alv_new = max(0.0, p_alv + dP_dt * dt_min) # dt in sec, flow in min
        state.p_alv = p_alv_new
        
        # Arterial Partial Pressure equilibrates with Alveolar
        state.p_art = p_alv_new
        
        # --- Tissue Uptake (PBPK Model) ---
        # Tissue uptake model:
        # - Tissues with high λ (fat: 50) store more anesthetic per unit pressure
        # - Higher λ means slower equilibration rate (1/λ factor)
        # - This correctly models: VRG equilibrates in minutes, fat in hours
        #
        # Reference: Yasuda et al. Anesth Analg. 1991 (sevo vs iso uptake);
        # Davis & Mapleson. Br J Anaesth. 1981 (physiological model).
        # Rate constant: k = (flow/volume) × (1/partition_coefficient)
        # Units: (L/min / L) × (1/unitless) = 1/min
        p_art = p_alv_new
        k_vrg = (q_vrg / self.v_vrg) / self.lambda_t_b_vrg
        k_mus = (q_mus / self.v_mus) / self.lambda_t_b_mus
        k_fat = (q_fat / self.v_fat) / self.lambda_t_b_fat
        
        # Update each tissue compartment
        p_vrg_new = state.p_vrg + k_vrg * (p_art - state.p_vrg) * dt_min
        p_mus_new = state.p_mus + k_mus * (p_art - state.p_mus) * dt_min
        p_fat_new = state.p_fat + k_fat * (p_art - state.p_fat) * dt_min
        
        state.p_vrg = p_vrg_new
        state.p_mus = p_mus_new
        state.p_fat = p_fat_new
        
        # --- Mixed Venous Pressure ---
        # Blood leaving each tissue equilibrates with tissue pressure.
        # Mixed venous = flow-weighted average of all tissue pressures.
        if q_co > 1e-6:
            p_ven_new = (q_vrg * p_vrg_new + q_mus * p_mus_new + q_fat * p_fat_new) / q_co
        else:
            p_ven_new = state.p_ven
        state.p_ven = p_ven_new
        
        # MAC Fraction (using VRG/brain pressure)
        
        # Temperature Correction for MAC Requirement
        # MAC decreases by ~5% per degree drop
        # Factor = 1.0 - 0.05 * (37 - temp)
        temp_diff = 37.0 - temp_c
        temp_factor = max(0.5, 1.0 - 0.05 * temp_diff) # clamp to avoid div by zero or negative
        
        corrected_mac_age = self.mac_age * temp_factor
        
        # MAC = VRG pressure (as %) / age-corrected MAC requirement
        # Example: VRG at 0.021 (2.1%) / MAC_age 2.0% = 1.05 MAC
        state.mac = (state.p_vrg * 100.0) / corrected_mac_age
