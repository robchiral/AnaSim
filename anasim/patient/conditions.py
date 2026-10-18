"""Patient disease profiles and pregnancy factors; see docs/REFERENCES.md."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HeartFailureProfile:
    output_factor: float = 1.0
    filling_pressure: float = 9.0  # Resting LAP, mmHg
    stiffness_factor: float = 1.0
    coupling_ratio: float = 0.0  # Ea/Ees
    tachycardia_gain: float = 0.0


# Schwartzenberg 2012, Table 2: HFrEF PCWP and Ea/Ees for both profiles.
# Westermann 2008, Tables 3-4: HFpEF EDP, stiffness constant, and rate
# sensitivity. Output factors, HFrEF stiffness, and rate gains are calibrations.
HEART_FAILURE_PROFILES = {
    "none": HeartFailureProfile(),
    "hfref": HeartFailureProfile(output_factor=0.70, filling_pressure=25.0,
                                 stiffness_factor=2.0, coupling_ratio=3.3, tachycardia_gain=0.15),
    "hfpef": HeartFailureProfile(output_factor=0.90, filling_pressure=16.1,
                                 stiffness_factor=0.028 / 0.0098, coupling_ratio=0.58, tachycardia_gain=0.55),
}
POSITIONS = ("supine", "left_lateral")
COPD_RESISTANCE = 12.0  # cmH2O/(L/s) added at severity 1; calibrated


@dataclass(frozen=True)
class PregnancyFactors:
    cardiac_output: float = 1.0
    blood_volume: float = 1.0
    filtration: float = 1.0
    heart_rate: float = 1.0
    metabolism: float = 1.0
    frc: float = 1.0
    paco2: float = 40.0
    ph: float = 7.4
    supine_output_loss: float = 0.0


def pregnancy_factors(gestational_age_weeks: float) -> PregnancyFactors:
    """Maternal factors relative to the nonpregnant state; 0 weeks means no pregnancy.

    Gestational age counts from the last menstrual period, about two weeks
    before fertilization.
    """
    if gestational_age_weeks == 0.0:
        return PregnancyFactors()
    fa = gestational_age_weeks - 2.0
    # Dallmann 2017 equations 19, 21, and 29 use fertilization age; normalize
    # them to their values at conception.
    co = 6.09 * math.exp(-math.exp(-0.352 * math.log(fa) + 1.36)) + 5.14
    hct = (0.000401 * fa**3 - 0.0180 * fa**2 + 0.0299 * fa + 40.1) / 100.0
    plasma = -0.0000817 * fa**3 + 0.00395 * fa**2 + 0.0000841 * fa + 2.36
    blood = (plasma / (1.0 - hct)) / (2.36 / (1.0 - 0.401))
    # The printed GFR polynomial has inconsistent units. Interpolate the
    # reported 113 to 150 mL/min rise by FA 14, then keep the reported plateau.
    filtration = 1.0 + (150.0 / 113.0 - 1.0) * min(fa / 14.0, 1.0)
    early = min(fa / 12.0, 1.0)
    late = max(0.0, min((gestational_age_weeks - 13.0) / 24.0, 1.0))
    compression = max(0.0, min((gestational_age_weeks - 20.0) / 16.0, 1.0))
    # Contreras 1991: FRC 2184 mL at week 37 and 2524 mL postpartum. Heart
    # rate, metabolic rise, compensated PCO2, and the time courses are calibrations.
    return PregnancyFactors(
        cardiac_output=co / 5.14,
        blood_volume=blood,
        filtration=filtration,
        heart_rate=1.0 + 0.20 * early,
        metabolism=1.0 + 0.20 * fa / 38.0,
        frc=1.0 - (1.0 - 2184.0 / 2524.0) * late,
        paco2=40.0 - 8.0 * early,
        ph=7.4 + 0.04 * early,
        # Humphries 2019: supine CO is 16.4% below left lateral at 35 to 38 weeks.
        supine_output_loss=0.164 * compression,
    )
