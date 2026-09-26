import csv
import os
import time
from dataclasses import fields
from enum import Enum
from pathlib import Path

from .state import SimulationState

STATE_FIELD_NAMES = tuple(field.name for field in fields(SimulationState))


class RecordingError(RuntimeError):
    """A recording could not be started, written, flushed, or closed."""


class DataRecorder:
    """Write SimulationState rows to CSV at a fixed sample interval."""

    def __init__(self, output_dir: str = ".", sample_interval_sec: float = 1.0):
        self.output_dir = output_dir
        self.filename = f"anasim_log_{time.time_ns()}.csv"
        self.file_path = str(Path(output_dir) / self.filename)
        self.file = None
        self.writer = None
        self.is_recording = False
        self.sample_interval_sec = max(0.0, sample_interval_sec)
        self._last_sample_time = None

    def start(self):
        if self.is_recording:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.file = open(self.file_path, 'x', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
            self.writer.writerow(STATE_FIELD_NAMES)
            self.file.flush()
            self._last_sample_time = None
            self.is_recording = True
        except (OSError, csv.Error, ValueError) as error:
            self._fail("start", error)

    def _fail(self, operation, error):
        message = f"Could not {operation} recording '{self.file_path}': {error}"
        try:
            self.stop()
        except RecordingError as cleanup_error:
            message += f". Cleanup also failed: {cleanup_error}"
        raise RecordingError(message) from error

    def log(self, state: SimulationState):
        if not self.is_recording or not self.writer:
            return

        if (self.sample_interval_sec > 0.0 and self._last_sample_time is not None
                and state.time - self._last_sample_time < self.sample_interval_sec):
            return

        values = (getattr(state, name) for name in STATE_FIELD_NAMES)
        try:
            self.writer.writerow(value.value if isinstance(value, Enum) else value for value in values)
        except (OSError, csv.Error, ValueError) as error:
            self._fail("write", error)
        try:
            self.file.flush()
        except (OSError, ValueError) as error:
            self._fail("flush", error)
        self._last_sample_time = state.time

    def stop(self):
        file = self.file
        self.file = None
        self.writer = None
        self.is_recording = False
        if file is None:
            return
        errors = []
        try:
            file.flush()
        except (OSError, ValueError) as error:
            errors.append(("flush", error))
        try:
            file.close()
        except (OSError, ValueError) as error:
            errors.append(("close", error))
        if errors:
            details = "; ".join(f"{operation}: {error}" for operation, error in errors)
            raise RecordingError(
                f"Could not finish recording '{self.file_path}' ({details})"
            ) from errors[0][1]
