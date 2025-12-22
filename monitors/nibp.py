
from dataclasses import dataclass

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
    def __init__(self, interval_min: float = 5.0):
        self.interval = interval_min * 60.0
        self.is_cycling = False
        self.is_inflating = False
        self.cuff_pressure = 0.0
        self.latest_reading = NIBPReading()
        
    def trigger(self):
        """Start a measurement manually."""
        self.is_cycling = True
        self.is_inflating = True
        self.cuff_pressure = 0.0
        
    def step(self, dt: float, current_time: float, true_map: float, true_sys: float = None) -> float:
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
                     
                     # "Measure"
                     # Generate reading based on true_map
                     meas_map = true_map
                     
                     # Calculate SBP/DBP from MAP using physiologically realistic formulas
                     # Normal relationship: MAP ≈ DBP + 1/3 * (SBP - DBP)
                     # Which gives: MAP ≈ (SBP + 2*DBP) / 3
                     # Assuming pulse pressure (PP) = SBP - DBP scales with MAP
                     # Normal PP is ~40 mmHg at MAP of 93
                     # In shock, PP narrows but doesn't go negative
                     
                     if true_sys is not None:
                         meas_sys = true_sys
                     else:
                         # Estimate pulse pressure based on MAP
                         # PP scales roughly linearly with MAP, minimum PP ~20
                         # At MAP 93, PP ~40; at MAP 50, PP ~25; at MAP 30, PP ~20
                         base_pp = 40.0
                         base_map = 93.0
                         pp_scale = max(0.5, meas_map / base_map)  # Floor at 50% of normal PP
                         pulse_pressure = max(15.0, base_pp * pp_scale)
                         
                         # From MAP = (SBP + 2*DBP) / 3 and PP = SBP - DBP:
                         # SBP = MAP + (2/3) * PP
                         # DBP = MAP - (1/3) * PP
                         meas_sys = meas_map + (2.0/3.0) * pulse_pressure
                     
                     # Calculate DBP from MAP and SBP
                     # MAP = (SBP + 2*DBP) / 3 => DBP = (3*MAP - SBP) / 2
                     meas_dia = (3.0 * meas_map - meas_sys) / 2.0
                     
                     # Apply physiological minimums
                     # SBP can't realistically be below ~40 mmHg and still have a measureable pulse
                     # DBP can't realistically be below ~20 mmHg
                     meas_sys = max(40.0, meas_sys)
                     meas_dia = max(20.0, meas_dia)
                     
                     # Ensure logical relationship: SBP > DBP
                     if meas_dia >= meas_sys:
                         meas_dia = meas_sys - 10.0
                     
                     self.latest_reading = NIBPReading(meas_sys, meas_dia, meas_map, current_time)
                     
        return self.cuff_pressure
