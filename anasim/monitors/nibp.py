from dataclasses import dataclass
from typing import Optional

import numpy as np

from anasim.core.enums import RhythmType


@dataclass(frozen=True, slots=True)
class NIBPReading:
    systolic: float = 120.0
    diastolic: float = 80.0
    map: float = 93.0
    timestamp: Optional[float] = None

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
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def trigger(self) -> None:
        """Start a measurement manually."""
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
        """Advance a cuff cycle and return its display pressure."""
        if dt <= 0.0:
            raise ValueError("NIBP monitor dt must be greater than zero")

        if self.is_cycling:
            if self.is_inflating:
                self.cuff_pressure += 400.0 * dt
                if self.cuff_pressure >= 160.0:
                    self.is_inflating = False
            else:
                self.cuff_pressure -= 10.0 * dt
                if self.cuff_pressure < 50.0:
                    self.is_cycling = False
                    self.cuff_pressure = 0.0

                    arrest_rhythm = rhythm_type in (
                        RhythmType.VFIB,
                        RhythmType.ASYSTOLE,
                    )
                    if arrest_rhythm or true_map < 30.0:
                        return self.cuff_pressure

                    if self.rng.random() < self._shock_failure_probability(true_map):
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

                    self.latest_reading = NIBPReading(
                        meas_sys,
                        meas_dia,
                        meas_map,
                        current_time,
                    )

        return self.cuff_pressure
