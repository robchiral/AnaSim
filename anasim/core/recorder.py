import csv
import os
import time
import logging
from pathlib import Path
from dataclasses import fields
from enum import Enum
from .state import SimulationState

logger = logging.getLogger(__name__)
STATE_FIELD_NAMES = tuple(field.name for field in fields(SimulationState))

class DataRecorder:
    """
    Records simulation data to CSV.
    """
    def __init__(self, output_dir: str = ".", sample_interval_sec: float = 1.0):
        self.output_dir = output_dir
        self.filename = f"anasim_log_{int(time.time())}.csv"
        self.file_path = str(Path(output_dir) / self.filename)
        self.file = None
        self.writer = None
        self.is_recording = False
        self.sample_interval_sec = max(0.0, sample_interval_sec)
        self._last_sample_time = None

    @staticmethod
    def _serialize_value(value):
        """Convert non-scalar state values into CSV-safe cells."""
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (dict, list)):
            return str(value)
        return value
        
    def start(self):
        try:
             os.makedirs(self.output_dir, exist_ok=True)
             self.file = open(self.file_path, 'w', newline='')
             self.writer = csv.writer(self.file)
             self.is_recording = True
             self.writer.writerow(STATE_FIELD_NAMES)
        except (OSError, csv.Error, ValueError):
             logger.exception("Failed to start recording")
             self.is_recording = False
             
    def log(self, state: SimulationState):
        if not self.is_recording or not self.writer:
            return

        if self.sample_interval_sec > 0.0:
            now = state.time
            if self._last_sample_time is not None and (now - self._last_sample_time) < self.sample_interval_sec:
                return
            self._last_sample_time = now
            
        row = [self._serialize_value(getattr(state, name)) for name in STATE_FIELD_NAMES]
        try:
            self.writer.writerow(row)
        except (OSError, csv.Error, ValueError):
            logger.exception("Failed to write record")
        
    def stop(self):
        if self.file:
            self.file.close()
        self.is_recording = False
