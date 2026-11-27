from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.linalg import expm

from anasim.core.utils import clamp
from anasim.physiology.resuscitation import (
    compression_mean_pressure,
    compression_waveform_pressure,
)

from .cardiac_cycle import CardiacCycleSample

# Arterial runoff toward the filling pressure once flow stops.
NO_FLOW_TAU_S = 2.0
# Pulse detection and numeric hold times.
MIN_PULSE_PRESSURE_MMHG = 3.0
MIN_BEAT_DURATION_S = 0.20
NO_PULSE_TIMEOUT_S = 3.5
MAX_PULSE_TIMEOUT_S = 12.0
NONPULSATILE_MAP_WINDOW_S = 2.0
_PULSE_ALLOWANCE_S = 1.5
_MAP_SLICE_S = 0.1


@dataclass(frozen=True, slots=True)
class ArterialWaveformConfig:
    """Parameters for the arterial pulse renderer."""

    compliance_ref_ml_mmhg: float = 1.5
    # Compliance falls 0.8% per year after 40, matching normotensive pulse
    # pressure in Franklin 1997: flat until about 45, then 35-40% higher by 70-74.
    compliance_age_ref: float = 40.0
    compliance_age_slope: float = 0.008
    compliance_age_min_factor: float = 0.5
    compliance_age_max_factor: float = 1.0
    # Arterial compliance rises as distending pressure falls (Langewouters 1984),
    # so pulse pressure narrows with MAP at a fixed stroke volume; the reference
    # compliance applies at MAP 93 mmHg. Propofol lowers SBP roughly three to
    # four times as much as DBP in Ebert 1992 and Jeleazcov 2015 (exponent 0.5
    # gives roughly twice as much; steeper exponents break epinephrine bolus SBP fits).
    compliance_pressure_ref_mmhg: float = 93.0
    compliance_pressure_exponent: float = 0.5
    electromechanical_delay_s: float = 0.04
    notch_height: float = 0.46
    dicrotic_peak_height: float = 0.52
    diastolic_fast_weight: float = 0.55
    diastolic_fast_rate: float = 5.0
    diastolic_slow_rate: float = 1.5


class ArterialPressureSample(NamedTuple):
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
    """Render an arterial pulse after Mahdi 2017, constrained by Su's MAP and stroke volume."""

    def __init__(
        self, age: float, config: ArterialWaveformConfig | None = None, *, bsa: float = 1.8,
    ):
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
        size_factor = max(0.5, float(bsa)) / 1.8
        self.arterial_compliance = self.config.compliance_ref_ml_mmhg * age_factor * size_factor
        cfg = self.config
        # Unnormalized decay at the end of diastole.
        self._diastolic_end = (
            cfg.diastolic_fast_weight * math.exp(-cfg.diastolic_fast_rate)
            + (1.0 - cfg.diastolic_fast_weight) * math.exp(-cfg.diastolic_slow_rate)
        )
        self._diastolic_integral = self._diastolic_normalized_integral()
        self._initialized = False
        # Mean-beat pressures, latched at each R wave.
        self._mean_pressure = 0.0
        self._systolic = 0.0
        self._diastolic = 0.0
        self._reference_pp = 0.0
        self._reference_mean_rr_s = 60.0 / 70.0
        self._runoff_tau_s = 1.0
        self._next_shape = self._build_shape(60.0 / 70.0, 70.0)
        # The drawn beat, latched at its pressure upstroke.
        self._shape = self._next_shape
        self._beat_rr_s = 60.0 / 70.0
        self._pulse_pressure = 0.0
        self._foot_pressure = 0.0
        self._end_pressure = 0.0
        self._windkessel_offset = 0.0
        self._recovering_from_no_flow = False
        self._beat_pending = False
        self._pressure = 0.0

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
            * self._diastolic_integral
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

    def _shape_value(self, phase: float, diastolic_decay: float) -> float:
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

        return shape.dicrotic_height * diastolic_decay

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
        end_value = self._diastolic_end
        normalized = (raw - end_value) / (1.0 - end_value)
        return clamp(normalized, 0.0, 1.0)

    def compliance_at(self, mean_pressure: float) -> float:
        """Arterial compliance (mL/mmHg) at a mean pressure."""
        cfg = self.config
        ratio = cfg.compliance_pressure_ref_mmhg / max(20.0, float(mean_pressure))
        return self.arterial_compliance * ratio**cfg.compliance_pressure_exponent

    def steady_pressures(self, map_value: float, stroke_volume_ml: float, hr: float) -> tuple[float, float]:
        """Steady-state (systolic, diastolic) pressures (mmHg) at fixed MAP, SV, and HR."""
        return self.reference_pressures(map_value, stroke_volume_ml, hr)

    def _steady_pp_and_shape(
        self,
        mean_pressure: float,
        floor_p: float,
        stroke_volume_ml: float,
        hr: float,
        pulse_pressure_factor: float,
    ) -> tuple[float, float]:
        """Return (steady pulse pressure, shape mean) matching _update_reference()."""
        hr_f = float(hr)
        shape = self._build_shape(60.0 / hr_f, hr_f)
        compliance = self.compliance_at(max(mean_pressure, floor_p))
        requested_pp = float(stroke_volume_ml) * float(pulse_pressure_factor) / compliance
        pp_budget = max(0.0, mean_pressure - floor_p) + 0.05 * min(mean_pressure, floor_p)
        return min(requested_pp, pp_budget / shape.mean), shape.mean

    def reference_pulse_pressure(
        self,
        map_value: float,
        stroke_volume_ml: float,
        hr: float,
        *,
        pulse_pressure_factor: float = 1.0,
        static_pressure: float = 0.0,
    ) -> float:
        """Expected steady pulse pressure (mmHg) matching _update_reference()."""
        mean_pressure = max(0.0, float(map_value))
        if mean_pressure <= 0.0 or stroke_volume_ml <= 0.0 or hr <= 0.0:
            return 0.0
        pp, _ = self._steady_pp_and_shape(
            mean_pressure, max(0.0, float(static_pressure)), stroke_volume_ml, hr, pulse_pressure_factor,
        )
        return pp

    def reference_pressures(
        self,
        map_value: float,
        stroke_volume_ml: float,
        hr: float,
        *,
        pulse_pressure_factor: float = 1.0,
        static_pressure: float = 0.0,
    ) -> tuple[float, float]:
        """Expected steady (systolic, diastolic) pressures (mmHg) matching _update_reference()."""
        mean_pressure = max(0.0, float(map_value))
        floor_p = max(0.0, float(static_pressure))
        if mean_pressure <= 0.0 or stroke_volume_ml <= 0.0 or hr <= 0.0:
            return floor_p, floor_p
        pp, shape_mean = self._steady_pp_and_shape(
            mean_pressure, floor_p, stroke_volume_ml, hr, pulse_pressure_factor,
        )
        diastolic = max(floor_p, mean_pressure - pp * shape_mean)
        return diastolic + pp, diastolic

    def _update_reference(
        self,
        map_value: float,
        stroke_volume_ml: float,
        sample: CardiacCycleSample,
        pulse_pressure_factor: float,
        static_pressure: float = 0.0,
    ) -> None:
        mean_pressure = max(0.0, float(map_value))
        if not sample.organized or mean_pressure <= 0.0 or stroke_volume_ml <= 0.0:
            self._mean_pressure = self._systolic = self._diastolic = self._reference_pp = 0.0
            return

        floor_p = max(0.0, float(static_pressure))
        shape = self._build_shape(sample.rr_interval_s, sample.measured_hr)
        compliance = self.compliance_at(max(mean_pressure, floor_p))
        requested_pp = float(stroke_volume_ml) * pulse_pressure_factor / compliance
        pp_budget = max(0.0, mean_pressure - floor_p) + 0.05 * min(mean_pressure, floor_p)
        self._reference_pp = min(requested_pp, pp_budget / shape.mean)
        self._diastolic = max(floor_p, mean_pressure - self._reference_pp * shape.mean)
        self._mean_pressure = self._diastolic + self._reference_pp * shape.mean
        self._systolic = self._diastolic + self._reference_pp
        self._next_shape = shape
        self._reference_mean_rr_s = sample.mean_rr_s
        flow_ml_s = float(stroke_volume_ml) / sample.mean_rr_s
        driving_p = max(0.0, mean_pressure - floor_p)
        raw_tau = driving_p / flow_ml_s * compliance if flow_ml_s > 1e-6 and driving_p > 0.5 else NO_FLOW_TAU_S
        self._runoff_tau_s = clamp(raw_tau, 0.25, 6.0)

    def _start_beat(self, sample: CardiacCycleSample, static_pressure: float = 0.0) -> None:
        """Rise from the previous beat's end pressure by this beat's stroke volume,
        then run off for this beat's R-R with the Windkessel time constant."""
        self._shape = self._next_shape
        self._beat_rr_s = sample.rr_interval_s
        if self._mean_pressure <= 0.0:
            self._pulse_pressure = self._foot_pressure = self._end_pressure = self._windkessel_offset = 0.0
            return
        floor_p = max(0.0, float(static_pressure))
        self._pulse_pressure = self._reference_pp * sample.stroke_fraction
        if self._recovering_from_no_flow:
            self._recovering_from_no_flow = False
            self._foot_pressure = self._pressure
            self._windkessel_offset = self._pressure - self._diastolic
        elif self._end_pressure > 0.0:
            self._foot_pressure = max(floor_p, self._pressure if self._pressure > 0.0 else self._end_pressure)
        else:
            self._foot_pressure = self._diastolic
            self._windkessel_offset = 0.0
        self._windkessel_offset *= math.exp(-sample.rr_interval_s / self._runoff_tau_s)
        if abs(self._windkessel_offset) < 1e-3:
            self._windkessel_offset = 0.0
        above_floor = max(0.0, self._diastolic - floor_p)
        runoff_exponent = clamp(
            -(sample.rr_interval_s - self._reference_mean_rr_s) / self._runoff_tau_s,
            -20.0,
            1.5,
        )
        self._end_pressure = max(
            floor_p,
            floor_p
            + above_floor * math.exp(runoff_exponent)
            + self._windkessel_offset,
        )

    def _waveform_pressure(self, sample: CardiacCycleSample) -> float:
        delay_s = self.config.electromechanical_delay_s
        if self._beat_pending:
            phase = (self._beat_rr_s - delay_s + sample.elapsed_s) / self._beat_rr_s
        else:
            phase = ((sample.elapsed_s - delay_s) / self._beat_rr_s) % 1.0
        phase = min(phase, 1.0 - 1e-9)
        decay = self._diastolic_decay(phase)
        return max(
            0.0,
            self._foot_pressure
            + self._pulse_pressure * self._shape_value(phase, decay)
            + (self._end_pressure - self._foot_pressure) * (1.0 - decay),
        )

    def seed(
        self,
        sample: CardiacCycleSample,
        map_value: float,
        stroke_volume_ml: float,
        *,
        static_pressure: float = 0.0,
        pulse_pressure_factor: float = 1.0,
    ) -> ArterialPressureSample:
        """Seed steady-state arterial waveform parameters at the given cycle phase."""
        self._initialized = False
        self._recovering_from_no_flow = False
        self._end_pressure = 0.0
        self._windkessel_offset = 0.0
        self._beat_pending = False
        self._pressure = max(0.0, float(static_pressure))
        if not sample.organized or map_value <= 0.0 or stroke_volume_ml <= 0.0:
            return self.step(
                sample,
                map_value,
                stroke_volume_ml,
                dt=0.0,
                static_pressure=static_pressure,
                pulse_pressure_factor=pulse_pressure_factor,
            )
        self._update_reference(
            map_value, stroke_volume_ml, sample, pulse_pressure_factor, static_pressure=static_pressure
        )
        self._start_beat(sample, static_pressure=static_pressure)
        delay_s = self.config.electromechanical_delay_s
        self._beat_pending = sample.elapsed_s < delay_s
        self._initialized = True
        self._pressure = pressure = float(self._waveform_pressure(sample))
        return ArterialPressureSample(
            pressure=pressure,
            systolic=float(self._systolic),
            diastolic=float(self._diastolic),
            mean=float(self._mean_pressure),
        )

    def step(
        self,
        sample: CardiacCycleSample,
        map_value: float,
        stroke_volume_ml: float,
        *,
        dt: float = 0.0,
        static_pressure: float = 0.0,
        pulse_pressure_factor: float = 1.0,
    ) -> ArterialPressureSample:
        """Render the beat; without pulsatile flow, pressure runs off to static_pressure over dt."""
        if not sample.organized or map_value <= 0.0 or stroke_volume_ml <= 0.0:
            self._update_reference(0.0, 0.0, sample, 1.0, static_pressure=static_pressure)
            self._pulse_pressure = self._foot_pressure = self._end_pressure = self._windkessel_offset = 0.0
            self._beat_pending = False
            self._recovering_from_no_flow = True
            self._initialized = True
            decay = math.exp(-dt / NO_FLOW_TAU_S) if dt > 0.0 else 0.0
            self._pressure = pressure = max(0.0, float(static_pressure + (self._pressure - static_pressure) * decay))
            return ArterialPressureSample(pressure, pressure, pressure, pressure)

        delay_s = self.config.electromechanical_delay_s
        if not self._initialized or sample.beat_started:
            self._update_reference(
                map_value, stroke_volume_ml, sample, pulse_pressure_factor, static_pressure=static_pressure
            )
            self._beat_pending = True
        # The drawn beat changes at its upstroke, after the electromechanical delay.
        if self._beat_pending and (sample.elapsed_s >= delay_s or not self._initialized):
            self._start_beat(sample, static_pressure=static_pressure)
            self._beat_pending = False
        self._initialized = True

        if self._recovering_from_no_flow:
            decay = math.exp(-dt / NO_FLOW_TAU_S) if dt > 0.0 else 1.0
            pressure = max(0.0, static_pressure + (self._pressure - static_pressure) * decay)
            if self._mean_pressure <= 0.0:
                self._pressure = pressure = float(pressure)
                return ArterialPressureSample(pressure, pressure, pressure, pressure)
        else:
            pressure = self._waveform_pressure(sample)

        self._pressure = pressure
        return ArterialPressureSample(
            pressure=pressure,
            systolic=self._systolic,
            diastolic=self._diastolic,
            mean=self._mean_pressure,
        )

    def step_compressions(self, sample: CardiacCycleSample, systolic: float, diastolic: float) -> ArterialPressureSample:
        """Render a compression pulse over the relaxation pressure between compressions."""
        pressure = compression_waveform_pressure(sample.phase, systolic, diastolic)
        # Stopping compressions releases pressure to the relaxation level.
        self._pressure = float(diastolic)
        # Organized beats restart from their own diastolic pressure after compressions.
        self._initialized = self._recovering_from_no_flow = False
        self._end_pressure = self._windkessel_offset = 0.0
        mean = compression_mean_pressure(systolic, diastolic)
        return ArterialPressureSample(float(pressure), float(systolic), float(diastolic), float(mean))

    def note_compression_baseline(self, diastolic: float) -> None:
        """Keep the arterial runoff baseline at least at the compression relaxation pressure."""
        baseline = float(diastolic)
        if baseline > self._pressure:
            self._pressure = baseline
            self._recovering_from_no_flow = True
            self._end_pressure = max(self._end_pressure, baseline)


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
        if self.numeric_window_s <= 0.0:
            raise ValueError("numeric_window_s must be greater than zero")
        self._pressure = 0.0
        self._velocity = 0.0
        self._coefficient_cache: dict[tuple[float, float, float], tuple[float, ...]] = {}
        self._last_dt = -1.0
        self._last_freq = -1.0
        self._last_damp = -1.0
        self._last_coeffs: tuple[float, ...] = ()
        self._initialized = False
        self._previous_pressure = 0.0
        self._running_min = math.inf
        self._running_max = -math.inf
        self._best_upstroke = 0.0
        self._upstroke_sbp = 0.0
        self._beat_integral = 0.0
        self._beat_duration = 0.0
        self._pending_min = math.inf
        self._pending_max = -math.inf
        self._pending_integral = 0.0
        self._pending_duration = 0.0
        self._recent_beats: deque[tuple[float, float, float, float]] = deque()
        self._window_slices: deque[tuple[float, float]] = deque()
        self._window_duration = 0.0
        self._slice_integral = 0.0
        self._slice_duration = 0.0
        self._since_last_pulse_s = 0.0
        self._pulsatile_rr_s = 0.0
        self._early_pulse_active = False
        self._latest_sbp = math.nan
        self._latest_dbp = math.nan
        self._latest_map = 0.0

    def _coefficients(self, dt: float) -> tuple[float, ...]:
        if (
            dt == self._last_dt
            and self.natural_frequency_hz == self._last_freq
            and self.damping_ratio == self._last_damp
        ):
            return self._last_coeffs
        cache_key = (float(dt), self.natural_frequency_hz, self.damping_ratio)
        cached = self._coefficient_cache.get(cache_key)
        if cached is not None:
            self._last_dt, self._last_freq, self._last_damp = cache_key
            self._last_coeffs = cached
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
        # Bound the cache for variable headless step sizes.
        if len(self._coefficient_cache) >= 16:
            self._coefficient_cache.pop(next(iter(self._coefficient_cache)))
        self._coefficient_cache[cache_key] = coefficients
        self._last_dt, self._last_freq, self._last_damp = cache_key
        self._last_coeffs = coefficients
        return coefficients

    def seed(self, sample: ArterialPressureSample) -> ArterialPressureSample:
        raw_pressure = float(sample.pressure)
        pressure = max(0.0, raw_pressure)
        self._pressure = raw_pressure
        self._velocity = 0.0
        self._previous_pressure = pressure
        self._reset_beat(pressure)
        self._clear_pending()
        self._recent_beats.clear()
        self._window_slices.clear()
        self._window_duration = 0.0
        self._slice_integral = 0.0
        self._slice_duration = 0.0
        self._pulsatile_rr_s = 0.0
        pulsatile = (
            math.isfinite(sample.systolic)
            and math.isfinite(sample.diastolic)
            and sample.systolic - sample.diastolic >= MIN_PULSE_PRESSURE_MMHG
        )
        mean = float(sample.mean)
        if pulsatile:
            self._latest_sbp = max(0.0, float(sample.systolic))
            self._latest_dbp = max(0.0, float(sample.diastolic))
            self._latest_map = clamp(mean, self._latest_dbp, self._latest_sbp)
            self._since_last_pulse_s = 0.0
        else:
            self._latest_sbp = math.nan
            self._latest_dbp = math.nan
            self._latest_map = max(0.0, mean)
            self._since_last_pulse_s = MAX_PULSE_TIMEOUT_S
        self._initialized = True
        return self.reading

    def _pulse_timeout(self, cycle: CardiacCycleSample) -> float:
        if not cycle.organized or self._pulsatile_rr_s <= 0.0:
            return NO_PULSE_TIMEOUT_S
        return min(MAX_PULSE_TIMEOUT_S, max(NO_PULSE_TIMEOUT_S, self._pulsatile_rr_s + _PULSE_ALLOWANCE_S))

    def _record_slice(self, step_integral: float, dt: float) -> None:
        window_s = min(self.numeric_window_s, NONPULSATILE_MAP_WINDOW_S)
        self._slice_integral += step_integral
        self._slice_duration += dt
        if self._slice_duration >= _MAP_SLICE_S - 1e-12:
            slices = self._window_slices
            slices.append((self._slice_integral, self._slice_duration))
            total_dur = self._window_duration + self._slice_duration
            self._slice_integral = 0.0
            self._slice_duration = 0.0
            popped = False
            while len(slices) > 1 and total_dur - slices[0][1] >= window_s - 1e-9:
                total_dur -= slices.popleft()[1]
                popped = True
            self._window_duration = sum(s[1] for s in slices) if popped else total_dur

    def _continuous_map(self, fallback: float) -> float:
        window_s = min(self.numeric_window_s, NONPULSATILE_MAP_WINDOW_S)
        total_integral = sum(s[0] for s in self._window_slices) + self._slice_integral
        total_duration = sum(s[1] for s in self._window_slices) + self._slice_duration
        if total_duration <= 0.0:
            return fallback
        excess = total_duration - window_s
        for oldest_integral, oldest_duration in self._window_slices:
            if excess <= 0.0:
                break
            trim = min(1.0, excess / oldest_duration)
            total_integral -= oldest_integral * trim
            total_duration -= oldest_duration * trim
            excess -= oldest_duration * trim
        return total_integral / total_duration if total_duration > 0.0 else fallback

    def _clear_pending(self) -> None:
        self._pending_min = math.inf
        self._pending_max = -math.inf
        self._pending_integral = 0.0
        self._pending_duration = 0.0

    def _stash_pending_beat(self) -> None:
        if not math.isnan(self._latest_sbp) and self._beat_duration > 0.0:
            self._pending_min = min(self._pending_min, self._running_min)
            self._pending_max = max(self._pending_max, self._running_max)
            self._pending_integral += self._beat_integral
            self._pending_duration += self._beat_duration

    def _complete_beat(self, next_rr_s: float) -> None:
        if self._beat_duration < MIN_BEAT_DURATION_S:
            self._stash_pending_beat()
            return
        if self._best_upstroke < MIN_PULSE_PRESSURE_MMHG:
            self._stash_pending_beat()
            self._pulsatile_rr_s = min(self._pulsatile_rr_s, next_rr_s)
            return
        beat_sbp = max(self._running_max, self._pending_max)
        beat_dbp = min(self._running_min, self._pending_min)
        total_integral = self._beat_integral + self._pending_integral
        total_duration = self._beat_duration + self._pending_duration
        self._clear_pending()
        beats = self._recent_beats
        beats.append((beat_sbp, beat_dbp, total_integral, total_duration))
        while len(beats) > 1 and sum(beat[3] for beat in beats) - beats[0][3] >= self.numeric_window_s:
            beats.popleft()
        window_duration = sum(beat[3] for beat in beats)
        sbp_weighted = sum(beat[0] * beat[3] for beat in beats)
        dbp_weighted = sum(beat[1] * beat[3] for beat in beats)
        map_integral = sum(beat[2] for beat in beats)
        if len(beats) > 1 and window_duration > self.numeric_window_s:
            oldest_sbp, oldest_dbp, oldest_integral, oldest_duration = beats[0]
            excess = min(oldest_duration, window_duration - self.numeric_window_s)
            sbp_weighted -= oldest_sbp * excess
            dbp_weighted -= oldest_dbp * excess
            map_integral -= oldest_integral * (excess / oldest_duration)
            window_duration -= excess
        self._latest_sbp = sbp_weighted / window_duration
        self._latest_dbp = dbp_weighted / window_duration
        self._latest_map = map_integral / window_duration

    def _reset_beat(self, pressure: float) -> None:
        self._running_min = pressure
        self._running_max = pressure
        self._best_upstroke = 0.0
        self._upstroke_sbp = pressure
        self._beat_integral = 0.0
        self._beat_duration = 0.0
        self._early_pulse_active = False

    def step(
        self,
        dt: float,
        cycle: CardiacCycleSample,
        sample: ArterialPressureSample,
    ) -> ArterialPressureSample:
        self.advance(dt, cycle, sample)
        return self.reading

    def advance(
        self,
        dt: float,
        cycle: CardiacCycleSample,
        sample: ArterialPressureSample,
    ) -> None:
        """Advance the filter and beat numerics without building a reading."""
        if dt <= 0.0:
            raise ValueError("arterial line dt must be greater than zero")
        if not self._initialized:
            raise RuntimeError("arterial line must be seeded before stepping")

        a00, a01, b0, a10, a11, b1 = self._coefficients(dt)
        previous, velocity = self._pressure, self._velocity
        self._pressure = a00 * previous + a01 * velocity + b0 * sample.pressure
        self._velocity = a10 * previous + a11 * velocity + b1 * sample.pressure
        pressure = max(0.0, self._pressure)

        step_integral = 0.5 * (self._previous_pressure + pressure) * dt
        self._record_slice(step_integral, dt)
        self._since_last_pulse_s += dt

        if not cycle.organized:
            self._reset_beat(pressure)
            self._clear_pending()
            self._pulsatile_rr_s = 0.0
        else:
            if cycle.beat_started:
                self._complete_beat(cycle.rr_interval_s)
                self._reset_beat(self._previous_pressure)
            self._beat_integral += step_integral
            self._beat_duration += dt
            if pressure < self._running_min:
                self._running_min = pressure
            if pressure > self._running_max:
                self._running_max = pressure
            upstroke = pressure - self._running_min
            if upstroke > self._best_upstroke:
                self._best_upstroke = upstroke
                self._upstroke_sbp = pressure
                if upstroke >= MIN_PULSE_PRESSURE_MMHG:
                    self._since_last_pulse_s = 0.0
                    self._pulsatile_rr_s = cycle.rr_interval_s
            elif (
                math.isnan(self._latest_sbp)
                and self._best_upstroke >= MIN_PULSE_PRESSURE_MMHG
                and pressure <= self._upstroke_sbp - min(MIN_PULSE_PRESSURE_MMHG, 0.25 * self._best_upstroke)
            ):
                self._early_pulse_active = True

            if self._early_pulse_active and self._beat_duration > 0.0:
                self._latest_sbp = self._running_max
                self._latest_dbp = self._running_min
                self._latest_map = self._beat_integral / self._beat_duration

        if self._since_last_pulse_s >= self._pulse_timeout(cycle):
            self._recent_beats.clear()
            self._clear_pending()
            self._pulsatile_rr_s = 0.0
            self._early_pulse_active = False
            self._latest_sbp = math.nan
            self._latest_dbp = math.nan

        if math.isnan(self._latest_sbp):
            self._latest_map = self._continuous_map(pressure)

        self._previous_pressure = pressure

    @property
    def pressure(self) -> float:
        """Latest filtered pressure, mmHg."""
        return float(self._previous_pressure)

    @property
    def reading(self) -> ArterialPressureSample:
        """Latest filtered pressure and completed-beat numerics."""
        return ArterialPressureSample(
            pressure=self.pressure,
            systolic=float(self._latest_sbp),
            diastolic=float(self._latest_dbp),
            mean=float(self._latest_map),
        )
