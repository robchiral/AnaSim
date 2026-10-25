import math
from dataclasses import dataclass, field

from .conditions import (
    COPD_RESISTANCE,
    HEART_FAILURE_PROFILES,
    POSITIONS,
    PregnancyFactors,
    pregnancy_factors,
)
from .domain import (
    AGE_RANGE_YEARS,
    BMI_RANGE_KG_M2,
    COPD_SEVERITY_RANGE,
    CORONARY_FLOW_RESERVE_RANGE,
    CREATININE_CLEARANCE_RANGE_ML_MIN,
    GESTATIONAL_AGE_RANGE_WEEKS,
    HEIGHT_RANGE_CM,
    HEMATOCRIT_RANGE,
    HEMOGLOBIN_RANGE_G_DL,
    HEPATIC_FUNCTION_RANGE,
    RENAL_FUNCTION_RANGE,
    WEIGHT_RANGE_KG,
    bounded_number,
)
from .variability import TYPICAL, PatientVariability


@dataclass
class Patient:
    """Patient demographics and baseline physiology."""

    age: float = 40.0
    # kg; the pre-pregnancy weight during pregnancy, since the gestational
    # factors already include the pregnancy's own mass and volume changes.
    weight: float = 70.0
    height: float = 170.0
    sex: str = "male"
    baseline_temp: float = 37.0
    baseline_hb: float = 13.5
    baseline_hct: float | None = None
    renal_function: float = 1.0
    creatinine_clearance_ml_min: float | None = None
    hepatic_function: float = 1.0
    coronary_flow_reserve: float = 5.0
    heart_failure: str = "none"
    copd_severity: float = 0.0
    gestational_age_weeks: float = 0.0
    position: str = "supine"  # Starting maternal position
    pregnancy: PregnancyFactors = field(init=False)
    renal_status: str = field(init=False)
    hepatic_status: str = field(init=False)

    # Baselines
    baseline_hr: float = 70.0
    baseline_map: float = 90.0
    baseline_rr: float = 12.0
    baseline_vt: float | None = None  # mL; None follows CO2 production and dead space

    bmi: float = field(init=False)
    bsa: float = field(init=False)
    # Set by the engine when the session samples an individual patient.
    variability: PatientVariability = field(default=TYPICAL, init=False, repr=False)

    def __post_init__(self):
        self._validate_demographics()
        self._calculate_metrics()
        self._validate_body_composition()
        self._validate_organ_function()
        self._validate_conditions()

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

        self.baseline_hr = bounded_number("baseline_hr", self.baseline_hr, 10.0)
        self.baseline_map = bounded_number("baseline_map", self.baseline_map, 20.0)
        self.baseline_rr = bounded_number("baseline_rr", self.baseline_rr, 0.0)
        if self.baseline_vt is not None:
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
        """Derive BMI and BSA (Du Bois)."""
        self.bmi = self.weight / ((self.height / 100.0) ** 2)
        self.bsa = 0.007184 * (self.weight**0.425) * (self.height**0.725)

    def _validate_body_composition(self):
        """Reject weight and height combinations outside the supported BMI range."""
        minimum, maximum = BMI_RANGE_KG_M2
        # Accept boundary entries that round, such as 173.4 kg at 170 cm.
        if not minimum - 1e-10 <= self.bmi <= maximum + 1e-10:
            raise ValueError(
                f"bmi derived from weight and height must be between {minimum:g} "
                f"and {maximum:g} kg/m²; got {self.bmi:.1f}"
            )

    def _validate_organ_function(self):
        """Enforce the organ-function factors represented by the UI model."""
        self.renal_function = bounded_number(
            "renal_function", self.renal_function, *RENAL_FUNCTION_RANGE
        )
        if self.creatinine_clearance_ml_min is not None:
            self.creatinine_clearance_ml_min = bounded_number(
                "creatinine_clearance_ml_min", self.creatinine_clearance_ml_min,
                *CREATININE_CLEARANCE_RANGE_ML_MIN, unit="mL/min",
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

    def _validate_conditions(self):
        """Enforce supported disease profiles and pregnancy inputs."""
        self.coronary_flow_reserve = bounded_number(
            "coronary_flow_reserve", self.coronary_flow_reserve, *CORONARY_FLOW_RESERVE_RANGE
        )
        self.copd_severity = bounded_number("copd_severity", self.copd_severity, *COPD_SEVERITY_RANGE)
        if not isinstance(self.heart_failure, str) or self.heart_failure not in HEART_FAILURE_PROFILES:
            raise ValueError("heart_failure must be 'none', 'hfref', or 'hfpef'")
        if not isinstance(self.position, str) or self.position not in POSITIONS:
            raise ValueError("position must be 'supine' or 'left_lateral'")
        minimum, maximum = GESTATIONAL_AGE_RANGE_WEEKS
        weeks = bounded_number("gestational_age_weeks", self.gestational_age_weeks, 0.0, maximum, unit="weeks")
        if weeks and (self.sex != "female" or weeks < minimum):
            raise ValueError(
                f"gestational_age_weeks must be 0, or {minimum:g} to {maximum:g} weeks in a female patient"
            )
        self.gestational_age_weeks = weeks
        self.pregnancy = pregnancy_factors(weeks)

    def predicted_body_weight(self) -> float:
        """Devine predicted body weight (kg), which tracks lung size."""
        return (50.0 if self.sex == "male" else 45.5) + 0.91 * (self.height - 152.4)

    def respiratory_compliance(self) -> float:
        """Static respiratory-system compliance (L/cmH2O) anesthetized and supine at PEEP 5.

        It grows with lung size and falls exponentially with BMI (Pelosi 1998).
        Exponents come from VitalDB recordings; the constant gives the recorded
        plateau compliance once AnaSim's tissue viscoelasticity is included. See
        docs/REFERENCES.md#ventilator-waveforms. Above BMI 32, the VitalDB fit
        gives way to Pelosi's saturating curve. The COPD rise is a calibration.
        """
        bmi = min(self.bmi, 32.0)
        apparent = 0.0475 * (self.predicted_body_weight() / 60.0) ** 0.62 * math.exp(-0.0277 * (bmi - 23.0))
        if self.bmi > 32.0:
            apparent *= (233.3 * math.exp(-0.086 * self.bmi) + 40.0) / (233.3 * math.exp(-0.086 * 32.0) + 40.0)
        return 1.09 * apparent * (1.0 + 0.6 * self.copd_severity)

    def functional_residual_capacity(self) -> float:
        """Supine FRC (L), scaled by lung size, BMI, COPD, and pregnancy.

        The BMI dependence follows Pelosi 1998, using its saturating curve above
        BMI 32; the coefficient, reference volume, and COPD hyperinflation are
        teaching estimates.
        """
        frc = 2.5 * (self.predicted_body_weight() / 65.0) ** 0.8 * math.exp(-0.04 * (min(self.bmi, 32.0) - 24.0))
        if self.bmi > 32.0:
            frc *= (11.97 * math.exp(-0.096 * self.bmi) + 0.46) / (11.97 * math.exp(-0.096 * 32.0) + 0.46)
        return frc * (1.0 + self.copd_severity) * self.pregnancy.frc

    def airway_resistance(self) -> float:
        """Airway and tube resistance, cmH2O/(L/s); the BMI slope above 32 follows Pelosi 1998."""
        return 10.0 + 0.23 * max(0.0, self.bmi - 32.0) + COPD_RESISTANCE * self.copd_severity

    @property
    def cardiac_output_factor(self) -> float:
        """Resting cardiac output relative to a healthy nonpregnant adult."""
        return HEART_FAILURE_PROFILES[self.heart_failure].output_factor * self.pregnancy.cardiac_output

    @property
    def renal_clearance_factor(self) -> float:
        """Renal function scaled by the rise in filtration during pregnancy."""
        return self.renal_function * self.pregnancy.filtration

    def creatinine_clearance(self) -> float:
        """Absolute creatinine clearance (mL/min) for Kleijn's covariates.

        A supplied measurement already includes renal and pregnancy effects.
        Otherwise use Kleijn 2011's adult imputation, scaled by renal function.
        """
        if self.creatinine_clearance_ml_min is not None:
            return self.creatinine_clearance_ml_min
        return (183.0 - 1.34 * self.age) * self.renal_clearance_factor

    def estimate_blood_volume(self) -> float:
        """Estimate total blood volume in mL using Nadler's formula."""
        h_m = self.height / 100.0

        if self.sex == "male":
            vol_l = 0.3669 * (h_m**3) + 0.03219 * self.weight + 0.6041
        else:
            vol_l = 0.3561 * (h_m**3) + 0.03308 * self.weight + 0.1833

        return vol_l * 1000.0 * self.pregnancy.blood_volume
