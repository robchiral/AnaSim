from dataclasses import dataclass

import numpy as np

from anasim.core.constants import ARREST_PULSE_PRESSURE_MMHG
from anasim.core.enums import PULSELESS_RHYTHMS, RhythmType


@dataclass(frozen=True, slots=True)
class NIBPReading:
    systolic: float = 120.0
    diastolic: float = 80.0
    map: float = 93.0
    timestamp: float | None = None

class NIBPMonitor:
    """Simulate an oscillometric NIBP cuff."""

    def __init__(self, interval_min: float = 5.0, rng=None):
        if interval_min <= 0.0:
            raise ValueError("interval_min must be greater than zero")
        self.interval = interval_min * 60.0
        self.is_cycling = False
        self.is_inflating = False
        self.cuff_pressure = 0.0
        self._deflation_rate = 10.0
        self.latest_reading = NIBPReading()
        self.measurement_failed = False
        self.failed_timestamp: float | None = None
        self.rng = rng if rng is not None else np.random.default_rng()
        self._reset_sweep()

    def _reset_sweep(self) -> None:
        self._deflation_dt = 0.0
        self._avg_map = 0.0
        self._avg_sys = 0.0
        self._avg_dia = 0.0
        self._disrupted_dt = 0.0
        self._saw_afib = False

    def trigger(self) -> None:
        """Start a measurement manually."""
        if self.is_cycling:
            return
        self.is_cycling = True
        self.is_inflating = True
        self.cuff_pressure = 0.0
        self._deflation_rate = 10.0
        self._reset_sweep()

    def _shock_failure_probability(self, true_map: float) -> float:
        if true_map >= 60.0:
            return 0.0
        if true_map <= 30.0:
            return 1.0
        severity = (60.0 - true_map) / 30.0
        return min(0.9, 0.9 * (severity ** 1.2))

    def step(
        self,
        dt: float,
        current_time: float,
        true_map: float,
        true_sys: float,
        true_dia: float,
        rhythm_type: RhythmType,
        compressions: bool = False,
    ) -> float:
        """Advance a cuff cycle; current_time is the end of the step."""
        if dt <= 0.0:
            raise ValueError("NIBP monitor dt must be greater than zero")

        if not self.is_cycling:
            return self.cuff_pressure

        remaining = dt
        if self.is_inflating:
            target_cuff = max(160.0, min(300.0, true_sys + 30.0))
            self._deflation_rate = max(10.0, (target_cuff - 50.0) / 11.0)
            inflation_time = max(0.0, (target_cuff - self.cuff_pressure) / 400.0)
            elapsed = min(remaining, inflation_time)
            self.cuff_pressure = min(target_cuff, self.cuff_pressure + 400.0 * elapsed)
            remaining -= elapsed
            if elapsed < inflation_time:
                return self.cuff_pressure
            self.is_inflating = False

        # Accumulate hemodynamic validity and pressures across the active deflation sweep.
        deflation_time = (self.cuff_pressure - 50.0) / self._deflation_rate
        sweep_dt = min(remaining, max(0.0, deflation_time))
        arrest_rhythm = rhythm_type in PULSELESS_RHYTHMS
        pulseless_pea = (true_sys - true_dia) < ARREST_PULSE_PRESSURE_MMHG
        step_disrupted = arrest_rhythm or compressions or pulseless_pea or true_map <= 30.0
        if sweep_dt > 0.0:
            if self._deflation_dt <= 0.0:
                self._avg_map, self._avg_sys, self._avg_dia = true_map, true_sys, true_dia
                self._deflation_dt = sweep_dt
            else:
                self._deflation_dt += sweep_dt
                weight = sweep_dt / self._deflation_dt
                self._avg_map += (true_map - self._avg_map) * weight
                self._avg_sys += (true_sys - self._avg_sys) * weight
                self._avg_dia += (true_dia - self._avg_dia) * weight
            if step_disrupted:
                self._disrupted_dt += sweep_dt
            if rhythm_type == RhythmType.AFIB:
                self._saw_afib = True

        # Carry unused time across the phase boundary, without overshooting.
        if remaining < deflation_time - 1e-9:
            self.cuff_pressure -= self._deflation_rate * remaining
            return self.cuff_pressure

        completed_at = current_time - max(0.0, remaining - deflation_time)
        self.is_cycling = False
        self.cuff_pressure = 0.0
        if self._deflation_dt > 0.0:
            avg_map, avg_sys, avg_dia = self._avg_map, self._avg_sys, self._avg_dia
        else:
            avg_map, avg_sys, avg_dia = true_map, true_sys, true_dia
        disrupted = self._disrupted_dt > 0.5 or step_disrupted or avg_map <= 30.0
        saw_afib = self._saw_afib
        self._reset_sweep()

        # Compressions, arrest, pulseless intervals, hypotension, and irregular AFib
        # stroke volumes can corrupt oscillometric detection.
        if (
            disrupted
            or self.rng.random() < self._shock_failure_probability(avg_map)
            or (saw_afib and self.rng.random() < 0.12)
        ):
            self.measurement_failed = True
            self.failed_timestamp = completed_at
            return self.cuff_pressure

        self.measurement_failed = False
        self.failed_timestamp = None

        low_flow_bias = 0.0
        if avg_map < 60.0:
            severity = (60.0 - avg_map) / 30.0
            low_flow_bias = 14.0 * severity
        meas_map = avg_map + low_flow_bias

        meas_sys = max(40.0, avg_sys + low_flow_bias * 1.15)
        meas_dia = max(20.0, avg_dia + low_flow_bias * 0.85)
        if saw_afib:
            map_jitter = (self.rng.random() - 0.5) * 10.0
            pp_jitter = (self.rng.random() - 0.5) * 14.0
            meas_map = max(25.0, meas_map + map_jitter)
            meas_sys = max(40.0, meas_sys + map_jitter + 0.5 * pp_jitter)
            meas_dia = max(20.0, meas_dia + map_jitter - 0.5 * pp_jitter)
        if meas_dia >= meas_sys:
            meas_dia = max(10.0, meas_sys - 10.0)
        meas_map = min(max(meas_map, meas_dia + 1.0), meas_sys - 1.0)

        self.latest_reading = NIBPReading(meas_sys, meas_dia, meas_map, completed_at)
        return self.cuff_pressure
