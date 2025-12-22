"""
Physiological and Numerical Constants for AnaSim.

This module centralizes magic numbers used throughout the simulation,
providing documentation and literature references for each constant.

NOTE: Only add constants here that are ACTIVELY IMPORTED elsewhere.
"""

# =============================================================================
# PHYSIOLOGICAL BOUNDS (used in hemodynamics.py)
# =============================================================================

# Heart Rate Bounds (bpm)
HR_MIN = 10.0  # Below this, essentially asystole
HR_MAX = 220.0  # Maximum physiological HR (220 - age for exercise, but this is absolute max)

# Blood Pressure Bounds (mmHg) - imported but available for bounds checking
MAP_MAX = 200.0  # Severe hypertensive crisis
SBP_MAX = 260.0  # Extreme hypertension
DBP_MAX = 160.0  # Extreme diastolic hypertension

# Total Peripheral Resistance minimum (Wood Units)
TPR_MIN = 0.006  # Minimum physiologically reasonable (severe vasodilation)

# Blood Volume (mL)
BLOOD_VOLUME_MIN = 500.0  # Below this, effectively exsanguinated

# =============================================================================
# RESPIRATORY CONSTANTS (used in respiration.py)
# =============================================================================

# Respiratory Rate Thresholds (bpm)
RR_APNEA_THRESHOLD = 2.0  # Below this, complete apnea
RR_BRADYPNEA_THRESHOLD = 4.0  # 2-4 bpm = severe bradypnea with irregular breathing

# Tidal Volume Minimum (mL)
VT_MIN = 50.0  # Below this, consider apneic

# =============================================================================
# PHARMACOKINETIC CONSTANTS (used in utils.py hill_function)
# =============================================================================

# Epsilon for preventing division by zero in Hill functions
HILL_EPSILON = 1e-12

# Maximum Hill coefficient (gamma) to prevent numerical overflow
GAMMA_MAX = 20.0

# Concentration ratio above which Hill function returns near-saturation
CONCENTRATION_RATIO_SATURATION = 100.0

# =============================================================================
# TCI CONTROLLER CONSTANTS (used in tci.py)
# =============================================================================

# Minimum interval between target changes (seconds)
TCI_MIN_TARGET_CHANGE_INTERVAL = 5.0

# Maximum TCI peak time search (seconds)
TCI_PEAK_TIME_MAX = 600.0  # 10 minutes

# =============================================================================
# THERMOREGULATION CONSTANTS (used in respiration.py, hemodynamics.py)
# =============================================================================

# Temperature coefficient for metabolic rate (Q10 effect)
# VCO2 decreases ~7% per °C below 37°C (Q10 ≈ 2.0)
# Reference: Sessler DI, "Perioperative Heat Balance", Anesthesiology 2000
TEMP_METABOLIC_COEFFICIENT = 0.93

# Thermoregulation effect on TPR per degree deviation from 37°C
# TPR increases ~10% per °C below normal (vasoconstriction)
# Reference: Frank SM et al., JAMA 1997
TEMP_TPR_COEFFICIENT = 0.10
