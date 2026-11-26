"""Chest compressions and electrical therapy.

Compressions are calibrated to adult CPR targets: forward flow near a fifth of
resting output, and diastolic pressure near the 25 mmHg CPR target (Meaney 2013),
providing >=15 mmHg coronary perfusion pressure (Paradis 1990). First-shock biphasic VF termination and
post-shock asystole follow out-of-hospital arrest recordings (Stiell 2007;
Koster 2008; Pierce 2015). Pulseless VT responses, penalties for fine VF or
hypoxia, synchronized cardioversion, and R-on-T risk are calibrations.
"""

import math
from bisect import bisect_right
from enum import Enum

import numpy as np

from anasim.core.enums import RhythmType
from anasim.core.utils import clamp01

COMPRESSION_RATE = 110.0  # /min, within the guideline 100-120
# Share of resting cardiac output at normal filling and vascular resistance;
# less when underfilled or when pressor afterload lowers forward flow (modeling
# the post-epinephrine EtCO2 drop in Cantineau 1994b alongside diastolic pressure
# rises in Kim 2025 and CPP targets in Paradis 1990). Calibrated so anesthetized adult
# compressions produce ~20% resting output and ~70/25 mmHg arterial pressure.
COMPRESSION_OUTPUT_FRACTION = 0.25
COMPRESSION_AFTERLOAD_COUPLING = 0.45
# Each compression raises arterial and venous pressure together, so the pulse
# adds to arterial pressure without driving flow.
COMPRESSION_PULSE_MMHG = 45.0
COMPRESSION_DUTY = 0.5  # Share of each cycle spent compressing
COMPRESSION_PERIOD_S = 60.0 / COMPRESSION_RATE
# Effective thoracic pressure drive and sub-FRC / airway-closure respiratory
# coupling during 110/min sternal compressions, calibrated to adult CPR
# mechanics and capnography studies (Deakin 2007; Cordioli 2016; Duchatelet
# 2018; Grieco 2019; Neumamm 2021; Orlob 2021; Lesimple 2022).
COMPRESSION_PRESSURE_CMH2O = 31.0  # Effective pleural/thoracic pulse amplitude
COMPRESSION_SUB_FRC_COMPLIANCE_FRAC = 0.28  # Stiffened sub-equilibrium compliance fraction
COMPRESSION_SUB_FRC_KNEE_L = 0.16  # Smooth transition scale below PEEP equilibrium (L)
COMPRESSION_AIRWAY_CLOSURE_R = 96.0  # Peak small-airway closure resistance below AOP (cmH2O/(L/s))
COMPRESSION_CLOSURE_OFFSET_CMH2O = 1.9  # Logistic midpoint shift below LungAeration.closing_pressure
COMPRESSION_CLOSURE_WIDTH_CMH2O = 1.35  # Logistic transition width around airway opening pressure
COMPRESSION_VALVE_INSP_REV_R = 20.0  # Ventilator inspiratory relief backflow impedance (cmH2O/(L/s))
COMPRESSION_VALVE_EXP_OUT_R = 14.0  # Ventilator expiratory valve surge impedance (cmH2O/(L/s))
COMPRESSION_VALVE_EXP_IN_R = 3.5  # Ventilator demand-valve recoil impedance (cmH2O/(L/s))
COMPRESSION_BAG_OUT_R = 8.0  # Circle/APL outward compression impedance (cmH2O/(L/s))
COMPRESSION_BAG_IN_R = 3.0  # Circle/bag inward recoil impedance (cmH2O/(L/s))
_COMPRESSION_DRIVE_SEGMENTS = 8
_COMPRESSION_ACTIVE_S = COMPRESSION_PERIOD_S * COMPRESSION_DUTY
_COMPRESSION_KNOTS = (
    tuple(_COMPRESSION_ACTIVE_S * i / _COMPRESSION_DRIVE_SEGMENTS for i in range(1, _COMPRESSION_DRIVE_SEGMENTS + 1))
    + (COMPRESSION_PERIOD_S,)
)
_COMPRESSION_VALUES = (
    tuple(math.sin(math.pi * i / _COMPRESSION_DRIVE_SEGMENTS) for i in range(_COMPRESSION_DRIVE_SEGMENTS + 1))
    + (0.0,)
)

SHOCK_ENERGY_J = 200.0  # Biphasic
# The ZOLL R Series charges to 200 J in under 7 s and disarms 60 s after charging.
CHARGE_TIME_S = 5.0
CHARGE_HOLD_S = 60.0
# First 200 J biphasic shocks terminate ~90% of VF (Stiell 2007; Koster 2008).
# Penalties for fine VF or hypoxia are calibrations.
DEFIBRILLATION_SUCCESS = 0.9
FINE_VF_SUCCESS_LOSS = 0.6
# Synchronized shocks.
CARDIOVERSION_SUCCESS = {
    RhythmType.AFIB: 0.85,
    RhythmType.SVT: 0.95,
    RhythmType.VTACH: 0.95,
}
# VF risk from an unsynchronized shock near the T-wave peak, a teaching calibration.
R_ON_T_PROBABILITY = 0.15


def compression_pressures(
    perfusion_pressure: float, venous_pressure: float, *, filling_factor: float = 1.0,
) -> tuple[float, float]:
    """Arterial pressure at the compression peak and between compressions (mmHg)."""
    relaxation = venous_pressure + perfusion_pressure
    pulse = COMPRESSION_PULSE_MMHG * clamp01(filling_factor)
    return relaxation + pulse, relaxation


def compression_mean_pressure(systolic: float, diastolic: float) -> float:
    """Mean arterial pressure over a half-sine chest compression cycle (mmHg)."""
    return diastolic + (systolic - diastolic) * 2.0 * COMPRESSION_DUTY / math.pi


def compression_waveform_pressure(phase: float, systolic: float, diastolic: float) -> float:
    """Instantaneous arterial pressure at normalized compression cycle phase [0, 1)."""
    compression = phase / COMPRESSION_DUTY
    shape = math.sin(math.pi * compression) if compression < 1.0 else 0.0
    return diastolic + (systolic - diastolic) * shape


def compression_pressure_drive(elapsed_s: float) -> tuple[float, float, float]:
    """Return (pressure_cmh2o, slope_cmh2o_s, time_to_next_knot_s) for CPR thoracic compression.

    Piecewise-linearized half-sine pulse across COMPRESSION_DUTY of the cycle so
    exact respiratory segments stay dt-invariant and phase-locked to compressions.
    """
    t = elapsed_s % COMPRESSION_PERIOD_S
    if t >= COMPRESSION_PERIOD_S - 1e-12:
        t = 0.0
    i = bisect_right(_COMPRESSION_KNOTS, t + 1e-12)
    if i >= len(_COMPRESSION_KNOTS):
        t = 0.0
        i = 0
    start = _COMPRESSION_KNOTS[i - 1] if i > 0 else 0.0
    end = _COMPRESSION_KNOTS[i]
    v0, v1 = _COMPRESSION_VALUES[i], _COMPRESSION_VALUES[i + 1]
    slope = (v1 - v0) / (end - start)
    value = v0 + slope * (t - start)
    return COMPRESSION_PRESSURE_CMH2O * value, COMPRESSION_PRESSURE_CMH2O * slope, end - t


class ShockResult(Enum):
    NOT_READY = "not_ready"
    PENDING = "pending"  # Charged; waiting for the next R wave
    NOT_DELIVERED = "not_delivered"  # Synchronized mode found no R wave
    DELIVERED = "delivered"


class Defibrillator:
    """One 200 J charge per shock."""

    def __init__(self):
        self.ready_at: float | None = None
        self.waiting_for_r = False
        self.last_shock: tuple[float, bool] | None = None

    def status(self, time: float) -> str:
        if self.ready_at is None:
            return "idle"
        eps = 1e-9 * max(1.0, abs(time))
        if time >= self.ready_at + CHARGE_HOLD_S - eps:
            self.disarm()
            return "idle"
        if time < self.ready_at - eps:
            return "charging"
        return "waiting_for_r" if self.waiting_for_r else "ready"

    def charge(self, time: float) -> None:
        if self.status(time) == "idle":
            self.ready_at = time + CHARGE_TIME_S

    def disarm(self) -> None:
        self.ready_at = None
        self.waiting_for_r = False

    def consume(self, time: float, synchronized: bool) -> None:
        self.disarm()
        self.last_shock = time, synchronized


def shock_outcome(rhythm: RhythmType, *, synchronized: bool, energy_failure: float,
                  electrically_silent: bool, vulnerable: bool, antiarrhythmic: float,
                  rng: np.random.Generator, hypoxia: float = 0.0,
                  perfusing: bool = True, energy_reserve: float = 1.0,
                  temp_c: float = 37.0, hyperkalemia_block: float = 0.0) -> RhythmType:
    """Return the rhythm after a discharge.

    A terminated arrhythmia becomes sinus rhythm, or transient post-shock
    asystole after prolonged pulseless arrest (Pierce 2015); whether an
    organized rhythm perfuses depends on the myocardium's energy state.
    Deep hypothermia (<30.5 °C) and severe hyperkalemia sharply reduce electrical
    conversion probability until the underlying cause is treated.
    """
    if electrically_silent:
        return RhythmType.ASYSTOLE
    if rhythm == RhythmType.ASYSTOLE:
        return rhythm
    depletion = max(energy_failure, 0.8 * clamp01(hypoxia))
    hypothermia_penalty = clamp01((30.5 - temp_c) / 3.0)
    hyperk_penalty = clamp01((hyperkalemia_block - 0.40) / 0.45)
    reversible_cause_scale = (1.0 - 0.90 * hypothermia_penalty) * (1.0 - 0.85 * hyperk_penalty)
    if not synchronized and rhythm != RhythmType.VFIB and perfusing and vulnerable and rng.random() < R_ON_T_PROBABILITY:
        return RhythmType.VFIB
    if rhythm == RhythmType.VFIB or (rhythm == RhythmType.VTACH and (not perfusing or not synchronized)):
        success = DEFIBRILLATION_SUCCESS * (1.0 - FINE_VF_SUCCESS_LOSS * depletion)
        # Either drug reduced the shocks needed (Kudenchuk 2016). The effect size is a calibration.
        success += 0.5 * (1.0 - success) * clamp01(antiarrhythmic)
        success *= reversible_cause_scale
        u = rng.random()
        if u >= success:
            return rhythm
        if rhythm == RhythmType.VFIB or not perfusing:
            arrest_depth = clamp01((max(1.0 - energy_reserve, energy_failure, clamp01(hypoxia)) - 0.25) / 0.75)
            asystole_share = min(0.78, 0.60 * arrest_depth + 0.25 * energy_failure)
            if u >= success * (1.0 - asystole_share):
                return RhythmType.ASYSTOLE
        return RhythmType.SINUS
    success = CARDIOVERSION_SUCCESS.get(rhythm, 0.0) * (1.0 - 0.5 * depletion) * reversible_cause_scale
    return RhythmType.SINUS if success and rng.random() < success else rhythm
