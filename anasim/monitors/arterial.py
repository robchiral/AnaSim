from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from anasim.core.utils import clamp

from .cardiac_cycle import CardiacCycleSample


@dataclass(frozen=True, slots=True)
class ArterialWaveformConfig:
    """Parameters for the Su-constrained arterial pulse renderer."""

    compliance_ref_ml_mmhg: float = 1.5
    compliance_age_ref: float = 40.0
    compliance_age_slope: float = 0.008
    compliance_age_min_factor: float = 0.5
    compliance_age_max_factor: float = 1.2
    electromechanical_delay_s: float = 0.04
    notch_height: float = 0.46
    dicrotic_peak_height: float = 0.52
    diastolic_fast_weight: float = 0.55
    diastolic_fast_rate: float = 5.0
    diastolic_slow_rate: float = 1.5


@dataclass(frozen=True, slots=True)
class ArterialPressureSample:
    pressure: float
    systolic: float
    diastolic: float
    mean: float


@dataclass(frozen=True, slots=True)
class _PulseShape:
    peak_phase: float
    notch_phase: float
    dicrotic_phase: float
    notch_height: float
    dicrotic_height: float
    mean: float


def _quintic_smoothstep(value: float) -> float:
    value = clamp(value, 0.0, 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


class ArterialWaveformRenderer:
    """Render a Mahdi-informed pulse constrained by Su MAP and stroke volume."""

    def __init__(self, age: float, config: ArterialWaveformConfig | None = None):
        self.config = ArterialWaveformConfig() if config is None else config
        if self.config.compliance_ref_ml_mmhg <= 0.0:
            raise ValueError("compliance_ref_ml_mmhg must be greater than zero")
        if not (
            0.0
            < self.config.compliance_age_min_factor
            <= self.config.compliance_age_max_factor
        ):
            raise ValueError("arterial compliance age factors are invalid")
        if self.config.electromechanical_delay_s < 0.0:
            raise ValueError("electromechanical_delay_s cannot be negative")
        if not 0.0 <= self.config.diastolic_fast_weight <= 1.0:
            raise ValueError("diastolic_fast_weight must be between zero and one")
        if (
            self.config.diastolic_fast_rate <= 0.0
            or self.config.diastolic_slow_rate <= 0.0
        ):
            raise ValueError("diastolic decay rates must be greater than zero")
        if not (
            0.0
            <= self.config.notch_height
            <= self.config.dicrotic_peak_height
            <= 1.0
        ):
            raise ValueError("dicrotic landmark heights are invalid")
        age_factor = 1.0 - self.config.compliance_age_slope * (
            float(age) - self.config.compliance_age_ref
        )
        age_factor = clamp(
            age_factor,
            self.config.compliance_age_min_factor,
            self.config.compliance_age_max_factor,
        )
        self.arterial_compliance = self.config.compliance_ref_ml_mmhg * age_factor
        self._initialized = False
        # Mean-beat pressures, latched at each R wave.
        self._mean_pressure = 0.0
        self._systolic = 0.0
        self._diastolic = 0.0
        self._reference_pp = 0.0
        self._runoff_tau_s = 1.0
        self._next_shape = self._build_shape(60.0 / 70.0, 70.0)
        # The drawn beat, latched at its pressure upstroke.
        self._shape = self._next_shape
        self._beat_rr_s = 60.0 / 70.0
        self._pulse_pressure = 0.0
        self._foot_pressure = 0.0
        self._end_pressure = 0.0
        self._beat_pending = False

    def _diastolic_normalized_integral(self) -> float:
        cfg = self.config
        w_fast = cfg.diastolic_fast_weight
        k_fast = cfg.diastolic_fast_rate
        k_slow = cfg.diastolic_slow_rate
        end_value = w_fast * math.exp(-k_fast) + (1.0 - w_fast) * math.exp(-k_slow)
        raw_integral = (
            w_fast * (1.0 - math.exp(-k_fast)) / k_fast
            + (1.0 - w_fast) * (1.0 - math.exp(-k_slow)) / k_slow
        )
        return (raw_integral - end_value) / (1.0 - end_value)

    def _build_shape(self, rr_interval_s: float, hr: float) -> _PulseShape:
        rr = float(rr_interval_s)
        ejection_s = clamp(0.413 - 0.0016 * float(hr), 0.18, 0.36)
        notch_phase = clamp(ejection_s / rr, 0.25, 0.68)
        peak_phase = clamp(0.38 * notch_phase, 0.09, notch_phase - 0.06)
        dicrotic_phase = clamp(notch_phase + 0.055 / rr, notch_phase + 0.025, 0.78)

        notch_height = self.config.notch_height
        dicrotic_height = self.config.dicrotic_peak_height

        area = 0.5 * peak_phase
        area += 0.5 * (1.0 + notch_height) * (notch_phase - peak_phase)
        area += 0.5 * (notch_height + dicrotic_height) * (dicrotic_phase - notch_phase)
        area += (
            dicrotic_height
            * self._diastolic_normalized_integral()
            * (1.0 - dicrotic_phase)
        )
        return _PulseShape(
            peak_phase=peak_phase,
            notch_phase=notch_phase,
            dicrotic_phase=dicrotic_phase,
            notch_height=notch_height,
            dicrotic_height=dicrotic_height,
            mean=clamp(area, 1e-6, 1.0 - 1e-6),
        )

    def _shape_value(self, phase: float) -> float:
        shape = self._shape
        phase = phase % 1.0
        if phase <= shape.peak_phase:
            u = phase / shape.peak_phase
            return _quintic_smoothstep(u)
        if phase <= shape.notch_phase:
            u = (phase - shape.peak_phase) / (shape.notch_phase - shape.peak_phase)
            return 1.0 + (shape.notch_height - 1.0) * _quintic_smoothstep(u)
        if phase <= shape.dicrotic_phase:
            u = (phase - shape.notch_phase) / (shape.dicrotic_phase - shape.notch_phase)
            return shape.notch_height + (
                shape.dicrotic_height - shape.notch_height
            ) * _quintic_smoothstep(u)

        return shape.dicrotic_height * self._diastolic_decay(phase)

    def _diastolic_decay(self, phase: float) -> float:
        """Diastolic pressure above the beat's end value, from 1 at the dicrotic peak to 0."""
        shape = self._shape
        phase = phase % 1.0
        if phase <= shape.dicrotic_phase:
            return 1.0
        u = (phase - shape.dicrotic_phase) / (1.0 - shape.dicrotic_phase)
        cfg = self.config
        raw = (
            cfg.diastolic_fast_weight * math.exp(-cfg.diastolic_fast_rate * u)
            + (1.0 - cfg.diastolic_fast_weight) * math.exp(-cfg.diastolic_slow_rate * u)
        )
        end_value = (
            cfg.diastolic_fast_weight * math.exp(-cfg.diastolic_fast_rate)
            + (1.0 - cfg.diastolic_fast_weight) * math.exp(-cfg.diastolic_slow_rate)
        )
        normalized = (raw - end_value) / (1.0 - end_value)
        return clamp(normalized, 0.0, 1.0)

    def _update_reference(self, map_value: float, stroke_volume_ml: float, sample: CardiacCycleSample) -> None:
        mean_pressure = max(0.0, float(map_value))
        if not sample.organized or mean_pressure <= 0.0 or stroke_volume_ml <= 0.0:
            self._mean_pressure = self._systolic = self._diastolic = self._reference_pp = 0.0
            return

        shape = self._build_shape(sample.rr_interval_s, sample.measured_hr)
        requested_pp = float(stroke_volume_ml) / self.arterial_compliance
        self._reference_pp = min(requested_pp, mean_pressure / shape.mean)
        self._mean_pressure = mean_pressure
        self._diastolic = mean_pressure - self._reference_pp * shape.mean
        self._systolic = self._diastolic + self._reference_pp
        self._next_shape = shape
        flow_ml_s = float(stroke_volume_ml) / sample.mean_rr_s
        self._runoff_tau_s = mean_pressure / flow_ml_s * self.arterial_compliance

    def _start_beat(self, sample: CardiacCycleSample) -> None:
        """Rise from the previous beat's end pressure by this beat's stroke volume,
        then run off for this beat's R-R with the Windkessel time constant."""
        self._shape = self._next_shape
        self._beat_rr_s = sample.rr_interval_s
        if self._mean_pressure <= 0.0:
            self._pulse_pressure = self._foot_pressure = self._end_pressure = 0.0
            return
        self._pulse_pressure = self._reference_pp * sample.stroke_fraction
        self._foot_pressure = self._end_pressure if self._end_pressure > 0.0 else self._diastolic
        self._end_pressure = self._diastolic * math.exp(
            -(sample.rr_interval_s - sample.mean_rr_s) / self._runoff_tau_s
        )

    def step(
        self,
        sample: CardiacCycleSample,
        map_value: float,
        stroke_volume_ml: float,
    ) -> ArterialPressureSample:
        if not sample.organized:
            self._update_reference(0.0, 0.0, sample)
            self._pulse_pressure = self._foot_pressure = self._end_pressure = 0.0
            self._beat_pending = False
            self._initialized = True
            return ArterialPressureSample(0.0, 0.0, 0.0, 0.0)

        delay_s = self.config.electromechanical_delay_s
        if not self._initialized or sample.beat_started:
            self._update_reference(map_value, stroke_volume_ml, sample)
            self._beat_pending = True
        # The drawn beat changes at its upstroke, after the electromechanical delay.
        if self._beat_pending and (sample.elapsed_s >= delay_s or not self._initialized):
            self._start_beat(sample)
            self._beat_pending = False
        self._initialized = True

        if self._beat_pending:
            phase = (self._beat_rr_s - delay_s + sample.elapsed_s) / self._beat_rr_s
        else:
            phase = ((sample.elapsed_s - delay_s) / self._beat_rr_s) % 1.0
        phase = min(phase, 1.0 - 1e-9)
        runoff = 1.0 - self._diastolic_decay(phase)
        pressure = (
            self._foot_pressure
            + self._pulse_pressure * self._shape_value(phase)
            + (self._end_pressure - self._foot_pressure) * runoff
        )
        return ArterialPressureSample(
            pressure=float(max(0.0, pressure)),
            systolic=float(self._systolic),
            diastolic=float(max(0.0, self._diastolic)),
            mean=float(self._mean_pressure),
        )


class ArterialLineMonitor:
    """Apply fluid-filled line dynamics and average completed-beat numerics."""

    def __init__(
        self,
        natural_frequency_hz: float = 20.0,
        damping_ratio: float = 0.65,
        numeric_window_s: float = 5.0,
    ):
        self.natural_frequency_hz = float(natural_frequency_hz)
        self.damping_ratio = float(damping_ratio)
        # Bedside monitors average pressure numerics over several seconds of beats.
        self.numeric_window_s = float(numeric_window_s)
        if self.natural_frequency_hz <= 0.0:
            raise ValueError("natural_frequency_hz must be greater than zero")
        if self.damping_ratio < 0.0:
            raise ValueError("damping_ratio cannot be negative")
        self._pressure = 0.0
        self._velocity = 0.0
        self._coefficient_cache: dict[tuple[float, float, float], tuple[float, ...]] = {}
        self._initialized = False
        self._previous_pressure = 0.0
        self._beat_min = math.inf
        self._beat_max = -math.inf
        self._beat_integral = 0.0
        self._beat_duration = 0.0
        self._recent_beats: deque[tuple[float, float, float, float]] = deque()
        self._latest_sbp = 0.0
        self._latest_dbp = 0.0
        self._latest_map = 0.0

    def _coefficients(self, dt: float) -> tuple[float, ...]:
        cache_key = (float(dt), self.natural_frequency_hz, self.damping_ratio)
        cached = self._coefficient_cache.get(cache_key)
        if cached is not None:
            return cached

        omega = 2.0 * math.pi * self.natural_frequency_hz
        augmented = np.array(
            [
                [0.0, 1.0, 0.0],
                [-omega * omega, -2.0 * self.damping_ratio * omega, omega * omega],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        transition = expm(augmented * dt)
        coefficients = tuple(float(value) for value in transition[:2].flat)
        # Variable headless step sizes must not grow the cache without bound.
        if len(self._coefficient_cache) >= 16:
            self._coefficient_cache.pop(next(iter(self._coefficient_cache)))
        self._coefficient_cache[cache_key] = coefficients
        return coefficients

    def seed(self, sample: ArterialPressureSample) -> ArterialPressureSample:
        self._pressure = float(sample.pressure)
        self._velocity = 0.0
        self._previous_pressure = sample.pressure
        self._beat_min = sample.pressure
        self._beat_max = sample.pressure
        self._beat_integral = 0.0
        self._beat_duration = 0.0
        self._recent_beats.clear()
        self._latest_sbp = sample.systolic
        self._latest_dbp = sample.diastolic
        self._latest_map = sample.mean
        self._initialized = True
        return ArterialPressureSample(
            pressure=float(sample.pressure),
            systolic=float(sample.systolic),
            diastolic=float(sample.diastolic),
            mean=float(sample.mean),
        )

    def _complete_beat(self) -> None:
        if self._beat_duration <= 0.0:
            return
        beats = self._recent_beats
        beats.append((self._beat_max, self._beat_min, self._beat_integral, self._beat_duration))
        while len(beats) > 1 and sum(beat[3] for beat in beats) - beats[0][3] >= self.numeric_window_s:
            beats.popleft()
        self._latest_sbp = sum(beat[0] for beat in beats) / len(beats)
        self._latest_dbp = sum(beat[1] for beat in beats) / len(beats)
        self._latest_map = sum(beat[2] for beat in beats) / sum(beat[3] for beat in beats)

    def _reset_beat(self, pressure: float) -> None:
        self._beat_min = pressure
        self._beat_max = pressure
        self._beat_integral = 0.0
        self._beat_duration = 0.0

    def step(
        self,
        dt: float,
        cycle: CardiacCycleSample,
        sample: ArterialPressureSample,
    ) -> ArterialPressureSample:
        if dt <= 0.0:
            raise ValueError("arterial line dt must be greater than zero")
        if not self._initialized:
            raise RuntimeError("arterial line must be seeded before stepping")

        a00, a01, b0, a10, a11, b1 = self._coefficients(dt)
        previous, velocity = self._pressure, self._velocity
        self._pressure = a00 * previous + a01 * velocity + b0 * sample.pressure
        self._velocity = a10 * previous + a11 * velocity + b1 * sample.pressure
        pressure = max(0.0, self._pressure)

        if not cycle.organized:
            self._recent_beats.clear()
            self._latest_sbp = 0.0
            self._latest_dbp = 0.0
            self._latest_map = 0.0
            self._reset_beat(pressure)
        else:
            if cycle.beat_started:
                self._complete_beat()
                self._reset_beat(pressure)
            self._beat_integral += 0.5 * (self._previous_pressure + pressure) * dt
            self._beat_duration += dt
            self._beat_min = min(self._beat_min, pressure)
            self._beat_max = max(self._beat_max, pressure)

        self._previous_pressure = pressure
        return ArterialPressureSample(
            pressure=pressure,
            systolic=float(self._latest_sbp),
            diastolic=float(self._latest_dbp),
            mean=float(self._latest_map),
        )
