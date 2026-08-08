from dataclasses import dataclass

import numpy as np

from anasim.core.enums import RhythmType


@dataclass
class NIBPReading:
    systolic: float = 120.0
    diastolic: float = 80.0
    map: float = 93.0
    timestamp: float = 0.0

class NIBPMonitor:
    """
    Simulates NIBP Cuff.
    """
    def __init__(self, interval_min: float = 5.0, rng=None):
        self.interval = interval_min * 60.0
        self.is_cycling = False
        self.is_inflating = False
        self.cuff_pressure = 0.0
        self.latest_reading = NIBPReading()
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def trigger(self):
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

    @staticmethod
    def _estimate_systolic_from_map(map_value: float) -> float:
        base_pp = 40.0
        base_map = 93.0
        pp_scale = max(0.5, map_value / base_map)
        pulse_pressure = max(15.0, base_pp * pp_scale)
        return map_value + (2.0 / 3.0) * pulse_pressure

    def step(
        self,
        dt: float,
        current_time: float,
        true_map: float,
        true_sys: float = None,
        rhythm_type: RhythmType = None,
    ) -> float:
        """
        Returns Cuff Pressure (for display).
        """
        if self.is_cycling:
            # Simple Inflate/Deflate simulation
            if self.is_inflating:
                # Inflate fast to 160
                self.cuff_pressure += 400.0 * dt # fast inflation
                if self.cuff_pressure >= 160.0:
                    self.is_inflating = False
            else:
                 # Deflate slow
                 self.cuff_pressure -= 10.0 * dt
                 if self.cuff_pressure < 50.0:
                     # Finish
                     self.is_cycling = False
                     self.cuff_pressure = 0.0
                     
                     arrest_rhythm = rhythm_type in (RhythmType.VFIB, RhythmType.ASYSTOLE)
                     if arrest_rhythm or true_map < 30.0:
                         return self.cuff_pressure

                     if self.rng.random() < self._shock_failure_probability(true_map):
                         return self.cuff_pressure

                     low_flow_bias = 0.0
                     if true_map < 60.0:
                         severity = (60.0 - true_map) / 30.0
                         low_flow_bias = 4.0 + 10.0 * severity
                     meas_map = true_map + low_flow_bias

                     if true_sys is not None:
                         meas_sys = true_sys + low_flow_bias * 1.15
                     else:
                         meas_sys = self._estimate_systolic_from_map(meas_map)

                     meas_dia = (3.0 * meas_map - meas_sys) / 2.0
                     meas_sys = max(40.0, meas_sys)
                     meas_dia = max(20.0, meas_dia)
                     if meas_dia >= meas_sys:
                         meas_dia = meas_sys - 10.0

                     self.latest_reading = NIBPReading(meas_sys, meas_dia, meas_map, current_time)
                     
        return self.cuff_pressure
