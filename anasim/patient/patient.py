from dataclasses import dataclass, field

from .domain import (
    AGE_RANGE_YEARS,
    BMI_RANGE_KG_M2,
    HEIGHT_RANGE_CM,
    HEMATOCRIT_RANGE,
    HEMOGLOBIN_RANGE_G_DL,
    HEPATIC_FUNCTION_RANGE,
    RENAL_FUNCTION_RANGE,
    WEIGHT_RANGE_KG,
    bounded_number,
    finite_number,
)


@dataclass
class Patient:
    """Patient demographics and baseline physiology."""

    age: float = 40.0
    weight: float = 70.0
    height: float = 170.0
    sex: str = "male"
    asa: int = 1
    baseline_temp: float = 37.0
    baseline_hb: float = 13.5
    baseline_hct: float | None = None
    renal_function: float = 1.0
    hepatic_function: float = 1.0
    renal_status: str = field(init=False)
    hepatic_status: str = field(init=False)

    # Baselines
    baseline_hr: float = 70.0
    baseline_map: float = 90.0
    baseline_rr: float = 12.0
    baseline_vt: float = 500.0

    lbm: float = field(init=False)
    bmi: float = field(init=False)
    bsa: float = field(init=False)

    def __post_init__(self):
        self._validate_demographics()
        self._calculate_metrics()
        self._validate_body_composition()
        self._validate_organ_function()

    def _validate_demographics(self):
        """Normalize categorical values and enforce supported numeric inputs."""
        self.age = bounded_number("age", self.age, *AGE_RANGE_YEARS, unit="years")
        self.weight = bounded_number("weight", self.weight, *WEIGHT_RANGE_KG, unit="kg")
        self.height = bounded_number("height", self.height, *HEIGHT_RANGE_CM, unit="cm")

        if not isinstance(self.sex, str):
            raise ValueError("sex must be 'male' or 'female'")
        self.sex = self.sex.strip().lower()
        if self.sex not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")

        asa = finite_number("asa", self.asa)
        if not asa.is_integer() or not 1 <= asa <= 5:
            raise ValueError("asa must be an integer between 1 and 5")
        self.asa = int(asa)

        self.baseline_hr = bounded_number("baseline_hr", self.baseline_hr, 10.0)
        self.baseline_map = bounded_number("baseline_map", self.baseline_map, 20.0)
        self.baseline_rr = bounded_number("baseline_rr", self.baseline_rr, 0.0)
        self.baseline_vt = bounded_number("baseline_vt", self.baseline_vt, 50.0)
        self.baseline_temp = bounded_number(
            "baseline_temp", self.baseline_temp, 25.0, 42.0, unit="°C"
        )
        self.baseline_hb = bounded_number(
            "baseline_hb", self.baseline_hb, *HEMOGLOBIN_RANGE_G_DL, unit="g/dL"
        )
        expected_hct = 0.03 * self.baseline_hb
        if self.baseline_hct is None:
            self.baseline_hct = expected_hct
        else:
            self.baseline_hct = bounded_number(
                "baseline_hct", self.baseline_hct, *HEMATOCRIT_RANGE
            )
            if abs(self.baseline_hct - expected_hct) > 0.12:
                raise ValueError(
                    f"baseline_hb={self.baseline_hb:.1f} g/dL and "
                    f"baseline_hct={self.baseline_hct:.2f} are grossly inconsistent"
                )

    def _calculate_metrics(self):
        """Calculate BMI, LBM, BSA based on demographics."""
        self.bmi = self.weight / ((self.height / 100.0) ** 2)
        self.bsa = 0.007184 * (self.weight**0.425) * (self.height**0.725)

        # Janmahasatian et al. 2005 remains well behaved at high BMI, unlike
        # switching formulas only after the James equation becomes negative.
        self.lbm = self._janmahasatian_lbm()

    def _validate_body_composition(self):
        """Reject weight and height combinations outside the supported BMI range."""
        minimum, maximum = BMI_RANGE_KG_M2
        if not minimum <= self.bmi <= maximum:
            raise ValueError(
                f"bmi derived from weight and height must be between {minimum:g} "
                f"and {maximum:g} kg/m²; got {self.bmi:.1f}"
            )

    def _validate_organ_function(self):
        """Enforce the organ-function factors represented by the UI model."""
        self.renal_function = bounded_number(
            "renal_function", self.renal_function, *RENAL_FUNCTION_RANGE
        )
        self.hepatic_function = bounded_number(
            "hepatic_function", self.hepatic_function, *HEPATIC_FUNCTION_RANGE
        )
        if self.renal_function >= 0.9:
            self.renal_status = "Normal"
        elif self.renal_function >= 0.7:
            self.renal_status = "Mild"
        elif self.renal_function >= 0.5:
            self.renal_status = "Moderate"
        else:
            self.renal_status = "Severe"

        if self.hepatic_function >= 0.95:
            self.hepatic_status = "Normal"
        elif self.hepatic_function >= 0.8:
            self.hepatic_status = "Mild"
        elif self.hepatic_function >= 0.6:
            self.hepatic_status = "Moderate"
        else:
            self.hepatic_status = "Severe"

    def _janmahasatian_lbm(self) -> float:
        """Compute Janmahasatian lean body mass."""
        if self.sex == "male":
            return (9270.0 * self.weight) / (6680.0 + 216.0 * self.bmi)
        return (9270.0 * self.weight) / (8780.0 + 244.0 * self.bmi)

    def estimate_blood_volume(self) -> float:
        """Estimate total blood volume in mL using Nadler's formula."""
        h_m = self.height / 100.0

        if self.sex == "male":
            vol_l = 0.3669 * (h_m**3) + 0.03219 * self.weight + 0.6041
        else:
            vol_l = 0.3561 * (h_m**3) + 0.03308 * self.weight + 0.1833

        return vol_l * 1000.0
