"""
Physiological and Numerical Constants for AnaSim.

This module centralizes magic numbers used throughout the simulation.

NOTE: Only add constants here that are ACTIVELY IMPORTED elsewhere.
"""

from dataclasses import dataclass

# Physiological bounds (used in hemodynamics.py).

# Heart Rate Bounds (bpm)
HR_MIN = 10.0  # Below this, functional asystole
HR_MAX = 220.0  # Maximum physiological HR 

# Blood Pressure Bounds (mmHg) - imported but available for bounds checking
MAP_MAX = 200.0  # Severe hypertensive crisis
SBP_MAX = 260.0  # Extreme hypertension
DBP_MAX = 160.0  # Extreme diastolic hypertension

# Total Peripheral Resistance minimum (Wood Units)
TPR_MIN = 0.006  # Minimum physiologically reasonable (severe vasodilation)

# Blood Volume (mL)
BLOOD_VOLUME_MIN = 500.0  # Below this, effectively exsanguinated

# Respiratory constants (used in respiration.py).

# Respiratory Rate Thresholds (bpm)
RR_APNEA_THRESHOLD = 2.0  # Below this, complete apnea
RR_BRADYPNEA_THRESHOLD = 4.0  # 2-4 bpm = severe bradypnea with irregular breathing

# Tidal Volume Minimum (mL)
VT_MIN = 50.0  # Threshold for functional apnea

# Pharmacokinetic constants (used in utils.py hill_function).

# Epsilon for preventing division by zero in Hill functions
HILL_EPSILON = 1e-12

# Maximum Hill coefficient (gamma) to prevent numerical overflow
GAMMA_MAX = 20.0

# Concentration ratio above which Hill function returns near-saturation
CONCENTRATION_RATIO_SATURATION = 100.0

# TCI controller constants (used in tci.py).

# Minimum interval between target changes (seconds)
TCI_MIN_TARGET_CHANGE_INTERVAL = 5.0

# Maximum TCI peak time search (seconds)
TCI_PEAK_TIME_MAX = 600.0

# Thermoregulation constants (used in respiration.py, hemodynamics.py).

# Temperature coefficient for metabolic rate (Q10 effect)
# VCO2 decreases ~7% per °C below 37°C (Q10 ≈ 2.0)
# Reference: Sessler. Anesthesiology. 2000.
TEMP_METABOLIC_COEFFICIENT = 0.93

# Thermoregulation effect on TPR per degree deviation from 37°C
# TPR increases ~10% per °C below normal (vasoconstriction)
# Reference: Frank et al. JAMA. 1997.
TEMP_TPR_COEFFICIENT = 0.10

# Shivering model constants (used in step_helpers.py, respiration.py).
# Clinically, shivering appears near 36.5°C in awake patients and is
# suppressed by anesthetics/opioids; maximal shivering can raise
# metabolic rate ~3-5x baseline.
SHIVER_BASE_THRESHOLD = 36.5
SHIVER_DEPTH_DROP_MAX = 2.0
SHIVER_REMI_DROP_MAX = 0.8
SHIVER_DELTA_FULL = 1.5
SHIVER_BIS_ON = 60.0
SHIVER_BIS_FULL = 80.0
SHIVER_MAX_MULTIPLIER = 3.0
SHIVER_TAU_ON = 30.0
SHIVER_TAU_OFF = 90.0

# Apnea PaCO2 rise rates (mmHg/min) and fast-phase duration.
APNEA_PACO2_RISE_FAST_MMHG_MIN = 10.0
APNEA_PACO2_RISE_SLOW_MMHG_MIN = 3.6
APNEA_PACO2_RISE_FAST_DURATION_SEC = 60.0


@dataclass(frozen=True)
class AirwayTuning:
    """Centralized airway tuning parameters."""
    # Laryngospasm dynamics (seconds)
    laryngospasm_tau_on: float = 1.0
    laryngospasm_tau_off: float = 8.0

    # Resistance impact (cmH2O/(L/s))
    upper_resistance_gain: float = 40.0
    bronch_resistance_gain: float = 20.0

    # Ventilation efficiency weighting
    vent_efficiency_bronch_weight: float = 0.5
    vent_efficiency_upper_weight: float = 0.2
    vent_efficiency_min: float = 0.1

    # Capnography obstruction weighting
    capno_obstruction_upper_weight: float = 1.0
    capno_obstruction_bronch_weight: float = 0.7

    # V/Q mismatch weighting
    vq_mismatch_bronch_weight: float = 0.85
    vq_mismatch_upper_weight: float = 0.25


@dataclass(frozen=True)
class ThermalTuning:
    """Centralized thermal model tuning parameters."""
    ambient_temp_c: float = 20.0
    base_conductance_w_per_c: float = 3.0
    anesthetic_conductance_gain: float = 0.5
    redistribution_gain_w_per_depth: float = 50000.0
    bair_hugger_gain_w_per_c: float = 7.0
    metabolic_reduction_max: float = 0.2
    depth_propofol_scale: float = 4.0
    metabolic_temp_threshold_c: float = 0.5
    specific_heat_j_kg_k: float = 3470.0
    temp_min_c: float = 25.0
    temp_max_c: float = 42.0
