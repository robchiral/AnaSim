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
        self._sanitize_demographics()
        self._calculate_metric()
        self._sanitize_organ_function()

    def _sanitize_demographics(self):
        """Normalize demographics and reject ages outside the adult model domain."""
        def _as_float(value, default):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        try:
            self.age = float(self.age)
        except (TypeError, ValueError) as exc:
            raise ValueError("age must be a number between 18 and 70 years") from exc
        if not 18.0 <= self.age <= 70.0:
            raise ValueError("age must be between 18 and 70 years for the validated adult model domain")
        self.weight = max(1.0, _as_float(self.weight, 70.0))
        self.height = max(30.0, _as_float(self.height, 170.0))
        self.sex = (self.sex or "male").strip().lower()
        if self.sex not in ("male", "female"):
            self.sex = "male"
        try:
            self.asa = int(self.asa)
        except (TypeError, ValueError):
            self.asa = 1
        self.asa = max(1, min(5, self.asa))

        self.baseline_hr = max(10.0, _as_float(self.baseline_hr, 70.0))
        self.baseline_map = max(20.0, _as_float(self.baseline_map, 90.0))
        self.baseline_rr = max(0.0, _as_float(self.baseline_rr, 12.0))
        self.baseline_vt = max(50.0, _as_float(self.baseline_vt, 500.0))
        self.baseline_temp = _as_float(self.baseline_temp, 37.0)
        self.baseline_temp = max(25.0, min(42.0, self.baseline_temp))
        self.baseline_hb = max(1.0, _as_float(self.baseline_hb, 13.5))
        self.baseline_hct = _as_float(self.baseline_hct, 0.42)
        self.baseline_hct = max(0.1, min(0.7, self.baseline_hct))

    def _calculate_metric(self):
        """Calculate BMI, LBM, BSA based on demographics."""
        # BMI
        self.bmi = self.weight / ((self.height / 100.0) ** 2)
        
        # BSA (DuBois)
        self.bsa = 0.007184 * (self.weight ** 0.425) * (self.height ** 0.725)
        
        # Janmahasatian et al. 2005 remains well behaved at high BMI, unlike
        # switching formulas only after the James equation becomes negative.
        self.lbm = self._janmahasatian_lbm()

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
        """Compute Janmahasatian lean body mass."""
        sex = self.sex.lower()
        if sex == "male":
            return (9270.0 * self.weight) / (6680.0 + 216.0 * self.bmi)
        return (9270.0 * self.weight) / (8780.0 + 244.0 * self.bmi)

    def estimate_blood_volume(self) -> float:
        """Estimate Total Blood Volume in mL using Nadler's formula."""        
        h_m = self.height / 100.0
        
        sex = self.sex.lower()
        if sex == "male":
             vol_l = 0.3669 * (h_m**3) + 0.03219 * self.weight + 0.6041
        else:
             vol_l = 0.3561 * (h_m**3) + 0.03308 * self.weight + 0.1833
             
        return vol_l * 1000.0
