import csv
import os
import time
from enum import Enum
from typing import List, Any
from .state import SimulationState

class DataRecorder:
    """
    Records simulation data to CSV.
    """
    def __init__(self, output_dir: str = ".", sample_interval_sec: float = 1.0):
        self.output_dir = output_dir
        self.filename = f"anasim_log_{int(time.time())}.csv"
        self.file_path = f"{output_dir}/{self.filename}"
        self.file = None
        self.writer = None
        self.is_recording = False
        self.sample_interval_sec = max(0.0, sample_interval_sec)
        self._last_sample_time = None
        
    def start(self):
        try:
             os.makedirs(self.output_dir, exist_ok=True)
             self.file = open(self.file_path, 'w', newline='')
             self.writer = csv.writer(self.file)
             self.is_recording = True
             # Header based on SimulationState dataclass fields.
             from dataclasses import fields
             header = [f.name for f in fields(SimulationState)]
             self.writer.writerow(header)
        except Exception as e:
             print(f"Failed to start recording: {e}")
             self.is_recording = False
             
    def log(self, state: SimulationState):
        if not self.is_recording or not self.writer:
            return

        if self.sample_interval_sec > 0.0:
            now = getattr(state, "time", None)
            if now is not None:
                if self._last_sample_time is not None and (now - self._last_sample_time) < self.sample_interval_sec:
                    return
                self._last_sample_time = now
            
        from dataclasses import asdict
        row = list(asdict(state).values())
        # Convert complex objects (dict/list/Enum) to strings for CSV.
        row = [
            (x.value if isinstance(x, Enum) else str(x))
            if isinstance(x, (dict, list, Enum))
            else x
            for x in row
        ]
        self.writer.writerow(row)
        
    def stop(self):
        if self.file:
            self.file.close()
        self.is_recording = False
