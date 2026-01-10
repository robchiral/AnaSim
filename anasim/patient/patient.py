from dataclasses import dataclass

@dataclass
class Patient:
    """
    Patient demographics and baseline physiology.
    """
    age: float = 40.0       # years
    weight: float = 70.0    # kg
    height: float = 170.0   # cm
    sex: str = "male"       # "male" or "female"
    asa: int = 1            # ASA physical status 1-5
    baseline_temp: float = 37.0 # Celsius
    baseline_hb: float = 13.5   # g/dL
    baseline_hct: float = 0.42  # Fraction
    renal_function: float = 1.0   # 0.0-1.0 (eGFR fraction, 1.0 = normal)
    hepatic_function: float = 1.0 # 0.0-1.0 (Child-Pugh fraction, 1.0 = normal)
    renal_status: str = "Normal"
    hepatic_status: str = "Normal"

    # Baselines
    baseline_hr: float = 70.0
    baseline_map: float = 90.0
    baseline_rr: float = 12.0
    baseline_vt: float = 500.0 # mL
    
    # Derived parameters (can be computed post-init)
    lbm: float = 0.0 # Lean Body Mass
    bmi: float = 0.0 # Body Mass Index
    bsa: float = 0.0 # Body Surface Area

    def __post_init__(self):
        self._calculate_metric()
        self._sanitize_organ_function()

    def _calculate_metric(self):
        """Calculate BMI, LBM, BSA based on demographics."""
        # BMI
        self.bmi = self.weight / ((self.height / 100.0) ** 2)
        
        # BSA (DuBois)
        self.bsa = 0.007184 * (self.weight ** 0.425) * (self.height ** 0.725)
        
        # LBM (James formula, with Janmahasatian fallback for extreme BMI)
        # Verify valid formulas for male/female
        sex = self.sex.lower()
        if sex == "male":
            lbm = 1.1 * self.weight - 128 * ((self.weight / self.height) ** 2)
        else:
            lbm = 1.07 * self.weight - 148 * ((self.weight / self.height) ** 2)
        if lbm <= 0:
            lbm = self._janmahasatian_lbm()
        self.lbm = max(0.1, lbm)

    def _sanitize_organ_function(self):
        """Clamp organ function inputs to [0.1, 1.0] and set default labels."""
        try:
            self.renal_function = float(self.renal_function)
        except (TypeError, ValueError):
            self.renal_function = 1.0
        try:
            self.hepatic_function = float(self.hepatic_function)
        except (TypeError, ValueError):
            self.hepatic_function = 1.0
        self.renal_function = max(0.1, min(1.0, self.renal_function))
        self.hepatic_function = max(0.1, min(1.0, self.hepatic_function))
        if not self.renal_status:
            self.renal_status = "Normal" if self.renal_function >= 0.95 else "Impaired"
        if not self.hepatic_status:
            self.hepatic_status = "Normal" if self.hepatic_function >= 0.95 else "Impaired"

    def _janmahasatian_lbm(self) -> float:
        """Compute Janmahasatian LBM as a fallback for high-BMI cases."""
        if self.bmi <= 0:
            return 0.0
        sex = self.sex.lower()
        if sex == "male":
            return (9270.0 * self.weight) / (6680.0 + 216.0 * self.bmi)
        return (9270.0 * self.weight) / (8780.0 + 244.0 * self.bmi)

    def estimate_blood_volume(self) -> float:
        """Estimate Total Blood Volume in mL."""
        # Nadler's formula or simple 70ml/kg
        # Men: 75 ml/kg, Women: 65 ml/kg approximate
        # Or Nadler:
        # Men: 0.3669 * H^3 + 0.03219 * W + 0.6041
        # Women: 0.3561 * H^3 + 0.03308 * W + 0.1833
        # (H in meters). Result in Liters.
        
        h_m = self.height / 100.0
        
        sex = self.sex.lower()
        if sex == "male":
             vol_l = 0.3669 * (h_m**3) + 0.03219 * self.weight + 0.6041
        else:
             vol_l = 0.3561 * (h_m**3) + 0.03308 * self.weight + 0.1833
             
        return vol_l * 1000.0
