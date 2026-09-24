"""Shared physiologic and numerical constants."""

from dataclasses import dataclass

HR_MIN = 10.0  # bpm
HR_MAX = 220.0  # bpm
BLOOD_VOLUME_MIN = 500.0  # mL

# Spontaneous rates (breaths/min): apnea below 2, irregular bradypnea 2-4.
RR_APNEA_THRESHOLD = 2.0
RR_BRADYPNEA_THRESHOLD = 4.0
VT_MIN = 50.0  # mL; smaller breaths count as apnea

# Hill function guards.
HILL_EPSILON = 1e-12
GAMMA_MAX = 20.0
CONCENTRATION_RATIO_SATURATION = 100.0

# VCO2 falls about 7% per °C below 37 °C (Sessler 2000).
TEMP_METABOLIC_COEFFICIENT = 0.93
# TPR rises about 10% per °C below the vasoconstriction threshold (Frank 1997).
TEMP_TPR_COEFFICIENT = 0.10

# Shivering starts near 36.5 °C when awake, is suppressed by anesthetics and
# opioids, and can raise metabolic rate 3-5 times.
SHIVER_BASE_THRESHOLD = 36.5
SHIVER_DEPTH_DROP_MAX = 2.0
SHIVER_REMI_DROP_MAX = 0.8
SHIVER_DELTA_FULL = 1.5
SHIVER_BIS_ON = 60.0
SHIVER_BIS_FULL = 80.0
SHIVER_MAX_MULTIPLIER = 3.0
SHIVER_TAU_ON = 30.0
SHIVER_TAU_OFF = 90.0

# Apneic PaCO2 rise (mmHg/min): fast for the first minute, then slow.
APNEA_PACO2_RISE_FAST_MMHG_MIN = 10.0
APNEA_PACO2_RISE_SLOW_MMHG_MIN = 3.6
APNEA_PACO2_RISE_FAST_DURATION_SEC = 60.0


@dataclass(frozen=True)
class AirwayTuning:
    """Airway complication parameters."""

    # Laryngospasm onset and offset time constants (s).
    laryngospasm_tau_on: float = 1.0
    laryngospasm_tau_off: float = 8.0
    # Added resistance at full severity, cmH2O/(L/s).
    upper_resistance_gain: float = 40.0
    bronch_resistance_gain: float = 20.0
    vent_efficiency_bronch_weight: float = 0.5
    vent_efficiency_upper_weight: float = 0.2
    vent_efficiency_min: float = 0.1
    capno_obstruction_upper_weight: float = 1.0
    capno_obstruction_bronch_weight: float = 0.7
    vq_mismatch_bronch_weight: float = 0.85
    vq_mismatch_upper_weight: float = 0.25

    # Upper-airway obstruction at full loss of consciousness without an ETT or
    # positive pressure. Scales with LOC because collapsibility rises abruptly
    # there (Hillman 2009); partial because closing pressure reaches only about
    # +1.4 cmH2O at deep propofol (Eastwood 2005).
    unsupported_collapse_max: float = 0.4


@dataclass(frozen=True)
class ThermalTuning:
    """Heat balance parameters."""

    ambient_temp_c: float = 20.0
    base_conductance_w_per_c: float = 3.0
    anesthetic_conductance_gain: float = 0.5
    # Redistribution accounts for about 1.3 °C of the 1.6 °C first-hour core
    # drop after induction (Matsukawa 1995).
    redistribution_core_drop_c: float = 1.3
    redistribution_tau_s: float = 1200.0
    bair_hugger_gain_w_per_c: float = 7.0
    metabolic_reduction_max: float = 0.2
    depth_propofol_scale: float = 4.0
    specific_heat_j_kg_k: float = 3470.0
    temp_min_c: float = 25.0
    temp_max_c: float = 42.0
