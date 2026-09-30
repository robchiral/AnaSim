"""CSV lifecycle failures must reach API and CLI callers."""

import csv
import io
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
        monkeypatch.setattr("anasim.core.recorder.open", lambda *a, **kw: file, raising=False)
        return file
    return inject


def test_recording_keeps_its_schedule_and_restart_uses_new_file(awake_engine, tmp_path):
    engine = awake_engine
    engine.start_recording(str(tmp_path), sample_interval_sec=1.0)
    first = engine.recorder
    engine.start_recording(str(tmp_path))
    assert engine.recorder is first
    for _ in range(round(5.0 / 0.3)):
        engine.step(0.3)
    with open(first.file_path, newline="") as file:
        rows = list(csv.DictReader(file))
    # Steps that cross a deadline keep the 1 s schedule instead of drifting.
    assert [float(row["time"]) for row in rows] == pytest.approx([0.3, 1.5, 2.4, 3.3, 4.5])
    assert float(rows[-1]["hr"]) > 0
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
