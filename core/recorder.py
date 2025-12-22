
import csv
import time
from typing import List, Any
from .state import SimulationState

class DataRecorder:
    """
    Records simulation data to CSV.
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        self.filename = f"anasim_log_{int(time.time())}.csv"
        self.file_path = f"{output_dir}/{self.filename}"
        self.file = None
        self.writer = None
        self.is_recording = False
        
    def start(self):
        try:
             self.file = open(self.file_path, 'w', newline='')
             self.writer = csv.writer(self.file)
             self.is_recording = True
             # Write header based on SimulationState fields
             # We inspect one dummy state or hardcode
             # Using dataclass fields is robust
             from dataclasses import fields
             header = [f.name for f in fields(SimulationState)]
             self.writer.writerow(header)
        except Exception as e:
             print(f"Failed to start recording: {e}")
             self.is_recording = False
             
    def log(self, state: SimulationState):
        if not self.is_recording or not self.writer:
            return
            
        from dataclasses import asdict
        row = list(asdict(state).values())
        # Convert complex objects (dict) to string if any
        # alarms is Dict
        row = [str(x) if isinstance(x, (dict, list)) else x for x in row]
        self.writer.writerow(row)
        
    def stop(self):
        if self.file:
            self.file.close()
        self.is_recording = False
