from dataclasses import dataclass

import numpy as np

from anasim.core.enums import RhythmType


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
        self.latest_reading = NIBPReading()
        self.measurement_failed = False
        self.rng = rng if rng is not None else np.random.default_rng()

    def trigger(self) -> None:
        """Start a measurement manually."""
        if self.is_cycling:
            return
        self.measurement_failed = False
        self.is_cycling = True
        self.is_inflating = True
        self.cuff_pressure = 0.0

    def _shock_failure_probability(self, true_map: float) -> float:
        if true_map >= 60.0:
            return 0.0
        if true_map <= 30.0:
            return 1.0
        severity = (60.0 - true_map) / 30.0
        return min(0.9, 0.15 + 0.75 * (severity ** 1.5))

    def step(
        self,
        dt: float,
        current_time: float,
        true_map: float,
        true_sys: float,
        true_dia: float,
        rhythm_type: RhythmType,
    ) -> float:
        """Advance a cuff cycle; current_time is the end of the step."""
        if dt <= 0.0:
            raise ValueError("NIBP monitor dt must be greater than zero")

        if not self.is_cycling:
            return self.cuff_pressure

        remaining = dt
        if self.is_inflating:
            inflation_time = (160.0 - self.cuff_pressure) / 400.0
            elapsed = min(remaining, inflation_time)
            self.cuff_pressure = min(160.0, self.cuff_pressure + 400.0 * elapsed)
            remaining -= elapsed
            if elapsed < inflation_time:
                return self.cuff_pressure
            self.is_inflating = False

        # Carry unused time across the phase boundary, without overshooting.
        deflation_time = (self.cuff_pressure - 50.0) / 10.0
        if remaining < deflation_time - 1e-9:
            self.cuff_pressure -= 10.0 * remaining
            return self.cuff_pressure

        completed_at = current_time - max(0.0, remaining - deflation_time)
        self.is_cycling = False
        self.cuff_pressure = 0.0
        arrest_rhythm = rhythm_type in (RhythmType.VFIB, RhythmType.ASYSTOLE)
        if arrest_rhythm or true_map <= 30.0:
            self.measurement_failed = True
            return self.cuff_pressure

        if self.rng.random() < self._shock_failure_probability(true_map):
            self.measurement_failed = True
            return self.cuff_pressure

        low_flow_bias = 0.0
        if true_map < 60.0:
            severity = (60.0 - true_map) / 30.0
            low_flow_bias = 4.0 + 10.0 * severity
        meas_map = true_map + low_flow_bias

        meas_sys = max(40.0, true_sys + low_flow_bias * 1.15)
        meas_dia = max(20.0, true_dia + low_flow_bias * 0.85)
        if meas_dia >= meas_sys:
            meas_dia = meas_sys - 10.0

        self.latest_reading = NIBPReading(meas_sys, meas_dia, meas_map, completed_at)
        return self.cuff_pressure
