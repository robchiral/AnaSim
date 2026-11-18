"""CSV lifecycle failures must reach API and CLI callers."""

import copy
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anasim.cli import run_headless
from anasim.core.recorder import RecordingError


class FailingFile(io.StringIO):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure
        self.writes = 0
        self.close_attempted = False

    def write(self, text):
        self.writes += 1
        if (self.failure == "header" and self.writes == 1) or (
            self.failure == "write" and self.writes > 1
        ):
            raise OSError("injected write failure")
        return super().write(text)

    def close(self):
        self.close_attempted = True
        super().close()
        if self.failure == "close":
            raise OSError("injected close failure")


@pytest.fixture
def inject_file(monkeypatch):
    def inject(failure):
        file = FailingFile(failure)
        monkeypatch.setattr("anasim.core.recorder.open", lambda *_args, **_kwargs: file, raising=False)
        return file
    return inject


def test_recording_keeps_its_schedule_and_restart_uses_new_file(awake_engine, tmp_path):
    engine = awake_engine
    engine.start_recording(str(tmp_path), sample_interval_sec=1.0)
    first = engine.recorder
    engine.start_recording(str(tmp_path))
    assert engine.recorder is first
    blood_gases = {}
    for _ in range(round(5.0 / 0.3)):
        engine.step(0.3)
        blood_gases[engine.state.time] = (
            engine.state.ph, engine.state.hco3, engine.state.lactate, engine.state.base_excess,
        )
    with open(first.file_path, newline="") as file:
        rows = list(csv.DictReader(file))
    # Steps that cross a deadline keep the 1 s schedule instead of drifting.
    assert [float(row["time"]) for row in rows] == pytest.approx([0.3, 1.5, 2.4, 3.3, 4.5])
    assert float(rows[-1]["hr"]) > 0
    assert tuple(float(rows[-1][field]) for field in ("ph", "hco3", "lactate", "base_excess")) == pytest.approx(
        blood_gases[float(rows[-1]["time"])],
    )
    engine.stop_recording()
    original = Path(first.file_path).read_bytes()
    engine.stop_recording()
    engine.start_recording(str(tmp_path))
    assert engine.recorder.file_path != first.file_path
    engine.step(0.25)
    engine.stop_recording()
    assert Path(first.file_path).read_bytes() == original


@pytest.mark.parametrize("failure", ["header", "write", "close"])
def test_engine_propagates_recording_failures_and_releases_file(
    awake_engine, tmp_path, inject_file, failure
):
    file = inject_file(failure)
    with pytest.raises(RecordingError, match="injected") as error:
        awake_engine.start_recording(str(tmp_path))
        awake_engine.step(0.1)
        awake_engine.stop_recording()
    assert str(tmp_path) in str(error.value)
    assert isinstance(error.value.__cause__, OSError)
    assert file.close_attempted
    assert file.closed
    assert not awake_engine.recorder.is_recording
    assert awake_engine.recorder.file is None
    awake_engine.stop_recording()
    awake_engine.step(0.1)  # A caller can continue without recording.


def test_headless_recording_failure_exits_unsuccessfully(tmp_path, inject_file, capsys):
    inject_file("write")
    args = SimpleNamespace(
        config=None, duration=0.1, record=True,
        record_dir=str(tmp_path), record_interval=1.0,
    )
    with pytest.raises(SystemExit) as error:
        run_headless(args)
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "Recording failed:" in output.err
    assert "injected" in output.err
    assert "Simulation completed" not in output.out


def test_deepcopy_during_active_recording_and_json_alarm_serialization(awake_engine, tmp_path):
    """Deepcopying an active recording engine detaches the copy's recorder (#34),
    and CSV alarm dictionaries serialize as valid JSON (#43).
    """
    engine = awake_engine
    engine.give_drug_bolus("roc", 50.0)
    for _ in range(30):
        engine.step(2.0)
    assert engine.state.alarms

    engine.start_recording(str(tmp_path), sample_interval_sec=1.0)
    cloned = copy.deepcopy(engine)
    assert engine.recorder.is_recording
    assert not cloned.recorder.is_recording and cloned.recorder.file is None

    engine.step(1.0)
    csv_path = Path(engine.recorder.file_path)
    engine.stop_recording()

    with open(csv_path, newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows
    parsed_alarms = json.loads(rows[-1]["alarms"])
    assert isinstance(parsed_alarms, dict) and "BIS" in parsed_alarms

    recorder = awake_engine.recorder
    recorder.start()
    first_path = recorder.file_path
    recorder.stop()
    recorder.start()
    second_path = recorder.file_path
    recorder.stop()
    assert first_path != second_path
