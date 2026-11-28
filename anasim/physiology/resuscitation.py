"""Chest compressions and electrical therapy.

Compressions are calibrated to adult CPR targets: forward flow near a fifth of
resting output, and diastolic pressure near the 25 mmHg CPR target (Meaney 2013),
providing >=15 mmHg coronary perfusion pressure (Paradis 1990). Variable CPR
rate, depth, and chest-wall recoil/leaning scale output, right atrial diastolic
pressure, CPP, and EtCO2 using Feneley 1988, Yannopoulos 2005, Zuercher 2010,
and Sheak 2015. Selectable biphasic shock energy curves follow Mittal 1999,
Mittal 2000, Stiell 2007, Glover 2008, Koster 2008, Stec 2011, Pierce 2015,
and Schmidt 2020.
"""

import math
from bisect import bisect_right
from enum import Enum

import numpy as np

from anasim.core.enums import RhythmType
from anasim.core.utils import clamp, clamp01
from anasim.patient.domain import bounded_number

COMPRESSION_RATE = 110.0  # /min, within the guideline 100-120
COMPRESSION_DEPTH_CM = 5.5  # cm, guideline 5-6 cm (Meaney 2013; Sheak 2015)
COMPRESSION_RECOIL = 1.0  # Full chest decompression fraction (1.0 = no leaning)
COMPRESSION_RATE_RANGE = (60.0, 160.0)
COMPRESSION_DEPTH_RANGE_CM = (2.0, 7.0)
COMPRESSION_RECOIL_RANGE = (0.0, 1.0)
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
# coupling during sternal compressions, calibrated to adult CPR mechanics and
# capnography studies (Deakin 2007; Cordioli 2016; Duchatelet 2018; Grieco 2019;
# Neumamm 2021; Orlob 2021; Lesimple 2022).
COMPRESSION_PRESSURE_CMH2O = 31.0  # Effective pleural/thoracic pulse amplitude at 5.5 cm depth
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
_COMPRESSION_PHASE_KNOTS = (
    tuple(COMPRESSION_DUTY * i / _COMPRESSION_DRIVE_SEGMENTS for i in range(1, _COMPRESSION_DRIVE_SEGMENTS + 1))
    + (1.0,)
)
_COMPRESSION_VALUES = (
    tuple(math.sin(math.pi * i / _COMPRESSION_DRIVE_SEGMENTS) for i in range(_COMPRESSION_DRIVE_SEGMENTS + 1))
    + (0.0,)
)

SHOCK_ENERGY_J = 200.0  # Default biphasic energy
SHOCK_ENERGY_RANGE_J = (20.0, 360.0)
SHOCK_ENERGY_STEPS_J = (50, 70, 100, 120, 150, 200, 300, 360)
# The ZOLL R Series charges to 200 J in under 7 s and disarms 60 s after charging.
CHARGE_TIME_S = 5.0
CHARGE_HOLD_S = 60.0
# First 200 J biphasic shocks terminate ~90% of VF (Stiell 2007; Koster 2008).
# Penalties for fine VF or hypoxia are calibrations.
DEFIBRILLATION_SUCCESS = 0.9
FINE_VF_SUCCESS_LOSS = 0.6
# Synchronized shocks at default 200 J, normal BMI.
CARDIOVERSION_SUCCESS = {
    RhythmType.AFIB: 0.85,
    RhythmType.SVT: 0.95,
    RhythmType.VTACH: 0.95,
}
# VF risk from an unsynchronized shock near the T-wave peak, a teaching calibration.
R_ON_T_PROBABILITY = 0.15


def compression_quality_factors(
    rate: float = COMPRESSION_RATE,
    depth_cm: float = COMPRESSION_DEPTH_CM,
    recoil: float = COMPRESSION_RECOIL,
) -> tuple[float, float, float]:
    """Return (co_multiplier, pulse_multiplier, rap_leaning_add_mmhg) for CPR quality.

    Returns exact (1.0, 1.0, 0.0) at guideline defaults (110/min, 5.5 cm, 1.0 recoil).
    Rate follows Feneley 1988 and Meaney 2013; depth follows Sheak 2015;
    incomplete recoil / leaning follows Yannopoulos 2005 and Zuercher 2010.
    """
    r = clamp(float(rate), *COMPRESSION_RATE_RANGE)
    d = clamp(float(depth_cm), *COMPRESSION_DEPTH_RANGE_CM)
    rec = clamp(float(recoil), *COMPRESSION_RECOIL_RANGE)
    leaning = 1.0 - rec

    if r <= COMPRESSION_RATE:
        rate_co = (r / COMPRESSION_RATE) ** 0.72
    else:
        rate_co = (
            1.0
            + 0.04 * clamp01((r - COMPRESSION_RATE) / 10.0)
            - 0.38 * (max(0.0, (r - 115.0) / 45.0) ** 1.6)
        )

    if d <= COMPRESSION_DEPTH_CM:
        depth_co = (d / COMPRESSION_DEPTH_CM) ** 1.25
    else:
        depth_co = 1.0 + 0.06 * math.tanh((d - COMPRESSION_DEPTH_CM) / 1.0)

    recoil_co = max(0.25, 1.0 - 0.90 * leaning)
    co_mult = max(0.05, rate_co * depth_co * recoil_co)
    pulse_mult = max(0.25, (d / COMPRESSION_DEPTH_CM) * (1.0 - 0.30 * leaning))
    # Zuercher 2010: 20% leaning raises diastolic RAP ~4 mmHg (9 -> 13 mmHg), cutting CPP and MBF.
    rap_leaning_add = 16.0 * leaning
    return co_mult, pulse_mult, rap_leaning_add


def compression_pressures(
    perfusion_pressure: float,
    venous_pressure: float,
    *,
    filling_factor: float = 1.0,
    pulse_scale: float = 1.0,
) -> tuple[float, float]:
    """Arterial pressure at the compression peak and between compressions (mmHg)."""
    relaxation = venous_pressure + perfusion_pressure
    pulse = COMPRESSION_PULSE_MMHG * clamp01(filling_factor) * max(0.0, float(pulse_scale))
    return relaxation + pulse, relaxation


def compression_mean_pressure(systolic: float, diastolic: float) -> float:
    """Mean arterial pressure over a half-sine chest compression cycle (mmHg)."""
    return diastolic + (systolic - diastolic) * 2.0 * COMPRESSION_DUTY / math.pi


def compression_waveform_pressure(phase: float, systolic: float, diastolic: float) -> float:
    """Instantaneous arterial pressure at normalized compression cycle phase [0, 1)."""
    compression = phase / COMPRESSION_DUTY
    shape = math.sin(math.pi * compression) if compression < 1.0 else 0.0
    return diastolic + (systolic - diastolic) * shape


def compression_pressure_drive(
    elapsed_s: float,
    *,
    rate: float = COMPRESSION_RATE,
    depth_cm: float = COMPRESSION_DEPTH_CM,
    recoil: float = COMPRESSION_RECOIL,
) -> tuple[float, float, float]:
    """Return (pressure_cmh2o, slope_cmh2o_s, time_to_next_knot_s) for CPR thoracic compression.

    Piecewise-linearized half-sine pulse across COMPRESSION_DUTY of the cycle so
    exact respiratory segments stay dt-invariant and phase-locked to compressions.
    """
    period_s = 60.0 / clamp(float(rate), *COMPRESSION_RATE_RANGE)
    t = elapsed_s % period_s
    if t >= period_s - 1e-12:
        t = 0.0
    phase = t / period_s
    i = bisect_right(_COMPRESSION_PHASE_KNOTS, phase + 1e-12)
    if i >= len(_COMPRESSION_PHASE_KNOTS):
        t = 0.0
        i = 0
    start = (_COMPRESSION_PHASE_KNOTS[i - 1] * period_s) if i > 0 else 0.0
    end = _COMPRESSION_PHASE_KNOTS[i] * period_s
    v0, v1 = _COMPRESSION_VALUES[i], _COMPRESSION_VALUES[i + 1]
    slope = (v1 - v0) / (end - start)
    value = v0 + slope * (t - start)
    amp = COMPRESSION_PRESSURE_CMH2O * (clamp(float(depth_cm), *COMPRESSION_DEPTH_RANGE_CM) / COMPRESSION_DEPTH_CM)
    # Incomplete decompression leaves positive residual intrathoracic pressure (Yannopoulos 2005).
    leaning_offset = 8.0 * clamp01(1.0 - float(recoil))
    return leaning_offset + amp * value, amp * slope, end - t


class ShockResult(Enum):
    NOT_READY = "not_ready"
    PENDING = "pending"  # Charged; waiting for the next R wave
    NOT_DELIVERED = "not_delivered"  # Synchronized mode found no R wave
    DELIVERED = "delivered"


class Defibrillator:
    """Selectable-energy biphasic defibrillator (20-360 J, default 200 J)."""

    def __init__(self, energy_j: float = SHOCK_ENERGY_J):
        self.energy_j: float = SHOCK_ENERGY_J
        self.charged_energy_j: float = SHOCK_ENERGY_J
        self.set_energy(energy_j)
        self.ready_at: float | None = None
        self.waiting_for_r = False
        self.last_shock: tuple[float, bool] | None = None
        self.last_shock_energy_j: float = self.energy_j

    @staticmethod
    def charge_time_s(energy_j: float) -> float:
        """Charging duration (s); equals CHARGE_TIME_S (5.0 s) at 200 J."""
        return CHARGE_TIME_S * (0.60 + 0.40 * (clamp(float(energy_j), *SHOCK_ENERGY_RANGE_J) / SHOCK_ENERGY_J))

    def set_energy(self, energy_j: float) -> float:
        val = bounded_number("shock energy", energy_j, *SHOCK_ENERGY_RANGE_J, unit="J")
        if val != self.energy_j:
            self.energy_j = val
            if self.ready_at is not None:
                self.disarm()
        return self.energy_j

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

    def charge(self, time: float, energy_j: float | None = None) -> None:
        if energy_j is not None:
            self.set_energy(energy_j)
        if self.status(time) == "idle":
            self.charged_energy_j = self.energy_j
            self.ready_at = time + self.charge_time_s(self.charged_energy_j)

    def disarm(self) -> None:
        self.ready_at = None
        self.waiting_for_r = False

    def consume(self, time: float, synchronized: bool) -> None:
        delivered_energy = self.charged_energy_j
        self.disarm()
        self.last_shock = time, synchronized
        self.last_shock_energy_j = delivered_energy


def _biphasic_energy_scale(rhythm: RhythmType, energy_j: float, bmi: float) -> float:
    """Energy- and impedance-dependent conversion efficacy relative to 200 J at BMI <= 25."""
    e = clamp(float(energy_j), *SHOCK_ENERGY_RANGE_J)
    excess_bmi = max(0.0, float(bmi) - 25.0)
    if rhythm == RhythmType.VFIB:
        # Mittal 1999: rectilinear biphasic high efficacy from 120-200 J, minimal BMI penalty above 120 J;
        # sub-100 J energies drop sharply. Normalized to 1.0 at (200 J, BMI <= 25).
        imp = 1.0 + 0.015 * excess_bmi
        eff_e = e / imp
        e50 = 68.0
        ref = (SHOCK_ENERGY_J ** 2.6) / (SHOCK_ENERGY_J ** 2.6 + e50 ** 2.6)
        raw = (eff_e ** 2.6) / (eff_e ** 2.6 + e50 ** 2.6)
        return clamp(raw / ref, 0.05, 1.06)
    if rhythm == RhythmType.AFIB:
        # Mittal 2000; Glover 2008 (BEST AF); Schmidt 2020 (CHESS): strong energy-response
        # and BMI/impedance dependence across 70-360 J. Normalized to 1.0 at (200 J, BMI <= 25).
        imp = 1.0 + 0.030 * excess_bmi
        eff_e = e / imp
        e50 = 69.0
        ref = (SHOCK_ENERGY_J ** 2.0) / (SHOCK_ENERGY_J ** 2.0 + e50 ** 2.0)
        raw = (eff_e ** 2.0) / (eff_e ** 2.0 + e50 ** 2.0)
        return clamp(raw / ref, 0.05, 1.12)
    if rhythm == RhythmType.SVT:
        # Stec 2011: 50 J ~74%, 100 J ~93%, 200 J ~96%.
        imp = 1.0 + 0.020 * excess_bmi
        eff_e = e / imp
        e50 = 34.0
        ref = (SHOCK_ENERGY_J ** 2.4) / (SHOCK_ENERGY_J ** 2.4 + e50 ** 2.4)
        raw = (eff_e ** 2.4) / (eff_e ** 2.4 + e50 ** 2.4)
        return clamp(raw / ref, 0.05, 1.03)
    # VTACH synchronized cardioversion
    imp = 1.0 + 0.020 * excess_bmi
    eff_e = e / imp
    e50 = 45.0
    ref = (SHOCK_ENERGY_J ** 2.4) / (SHOCK_ENERGY_J ** 2.4 + e50 ** 2.4)
    raw = (eff_e ** 2.4) / (eff_e ** 2.4 + e50 ** 2.4)
    return clamp(raw / ref, 0.05, 1.04)


def shock_outcome(rhythm: RhythmType, *, synchronized: bool, energy_failure: float,
                  electrically_silent: bool, vulnerable: bool, antiarrhythmic: float,
                  rng: np.random.Generator, hypoxia: float = 0.0,
                  perfusing: bool = True, energy_reserve: float = 1.0,
                  temp_c: float = 37.0, hyperkalemia_block: float = 0.0,
                  energy_j: float = SHOCK_ENERGY_J, bmi: float = 24.0) -> RhythmType:
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
        energy_scale = _biphasic_energy_scale(RhythmType.VFIB, energy_j, bmi)
        base_success = min(0.98, DEFIBRILLATION_SUCCESS * energy_scale)
        success = base_success * (1.0 - FINE_VF_SUCCESS_LOSS * depletion)
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
    base_cv = CARDIOVERSION_SUCCESS.get(rhythm, 0.0)
    energy_scale = _biphasic_energy_scale(rhythm, energy_j, bmi) if base_cv > 0.0 else 0.0
    success = min(0.98, base_cv * energy_scale) * (1.0 - 0.5 * depletion) * reversible_cause_scale
    return RhythmType.SINUS if success and rng.random() < success else rhythm
