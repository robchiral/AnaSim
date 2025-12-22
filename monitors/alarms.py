
from collections import deque
import numpy as np

class AlarmSystem:
    """
    Real-time alarm system for patient monitors.
    """
    def __init__(self, thresholds: dict = None, delays: dict = None, dt: float = 1.0):
        # Default Thresholds
        self.thresholds = thresholds or {
            'BIS_min': 20, 'BIS_max': 70,
            'MAP_min': 60, 'MAP_max': 110,
            'HR_min': 45, 'HR_max': 120,
            'SpO2_min': 90, 'SpO2_max': 100,
            'EtCO2_min': 30, 'EtCO2_max': 45
        }
        
        # Default Delays (seconds)
        self.delays = delays or {
            'BIS': 0,
            'MAP': 0,
            'HR': 0,
            'SpO2': 5,
            'EtCO2': 0
        }
        
        self.dt = dt
        
        # Buffers for delay logic
        # Store recent values: deque of (time, value) or just values if fixed dt
        self.buffers = {k: deque() for k in self.delays.keys()}
        
        # Active Alarms state
        # {'HR': {'high': False, 'low': True}, ...}
        self.active_alarms = {}
        
    def update(self, state_dict: dict):
        """
        Update the alarm system with a dictionary of current values.
        e.g. {'HR': 60, 'MAP': 80, ...}
        """
        current_alarms = {}
        
        for name, delay_sec in self.delays.items():
            if name not in state_dict:
                continue
                
            val = state_dict[name]
            
            # Update buffer
            # We need window size ~ delay_sec / dt
            window_len = max(1, int(delay_sec / self.dt))
            
            buf = self.buffers[name]
            buf.append(val)
            while len(buf) > window_len:
                buf.popleft()
                
            # Check thresholds
            # Condition must be true for the ENTIRE window (meaning sustained for delay)
            # Only check if we have enough history
            
            thresh_min = self.thresholds.get(f'{name}_min')
            thresh_max = self.thresholds.get(f'{name}_max')

            is_low = False
            is_high = False
            
            if len(buf) >= window_len:
                vals = list(buf)
                
                if thresh_max is not None:
                    # High alarm: All values > max
                    if all(v > thresh_max for v in vals):
                        is_high = True
                        
                if thresh_min is not None:
                    # Low alarm: All values < min
                    if all(v < thresh_min for v in vals):
                        is_low = True
                    
            if is_low or is_high:
                current_alarms[name] = {'low': is_low, 'high': is_high}
        
        self.active_alarms = current_alarms
        return self.active_alarms

