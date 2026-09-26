"""CSV lifecycle failures must reach API, CLI, and desktop callers."""

import csv
import io
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from anasim.cli import run_headless
from anasim.core.recorder import RecordingError
from anasim.ui.config_dialog import SimulationSetupDialog
from anasim.ui.main_window import MainWindow


class FailingFile(io.StringIO):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure
        self.writes = 0
        self.flushes = 0
        self.close_attempted = False

    def write(self, text):
        self.writes += 1
        if (self.failure == "header" and self.writes == 1) or (
            self.failure == "write" and self.writes > 1
        ):
            raise OSError("injected write failure")
        return super().write(text)

    def flush(self):
        self.flushes += 1
        if self.failure == "header_flush" or (
            self.failure == "flush" and self.flushes > 1
        ):
            raise OSError("injected flush failure")
        return super().flush()

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


def test_recording_samples_are_readable_before_stop_and_restart_uses_new_file(
    awake_engine, tmp_path
):
    engine = awake_engine
    engine.start_recording(str(tmp_path), sample_interval_sec=1.0)
    first = engine.recorder
    engine.start_recording(str(tmp_path))
    assert engine.recorder is first
    for _ in range(7):
        engine.step(0.25)
    with open(first.file_path, newline="") as file:
        rows = list(csv.DictReader(file))
    assert [float(row["time"]) for row in rows] == [0.25, 1.25]
    assert float(rows[-1]["hr"]) > 0
    engine.stop_recording()
    original = Path(first.file_path).read_bytes()
    engine.stop_recording()
    engine.start_recording(str(tmp_path))
    assert engine.recorder.file_path != first.file_path
    engine.step(0.25)
    engine.stop_recording()
    assert Path(first.file_path).read_bytes() == original


def test_start_failure_reaches_engine_caller(awake_engine, tmp_path):
    output = tmp_path / "file"
    output.write_text("existing data")
    with pytest.raises(RecordingError, match="start"):
        awake_engine.start_recording(str(output))
    assert not awake_engine.recorder.is_recording
    assert output.read_text() == "existing data"


@pytest.mark.parametrize("failure", ["header", "header_flush", "write", "flush", "close"])
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


def test_write_error_preserved_when_cleanup_also_fails(awake_engine, tmp_path, inject_file):
    file = inject_file("write")
    awake_engine.start_recording(str(tmp_path))

    def fail_close():
        raise OSError("cleanup failure")

    file.close = fail_close
    with pytest.raises(RecordingError, match="injected write failure.*cleanup failure") as error:
        awake_engine.step(0.1)
    assert str(error.value.__cause__) == "injected write failure"
    io.StringIO.close(file)


@pytest.mark.parametrize("failure", ["header", "header_flush", "write", "flush", "close"])
def test_headless_recording_failure_exits_unsuccessfully(
    tmp_path, inject_file, failure, capsys
):
    inject_file(failure)
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


@pytest.fixture
def window(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    dialog = SimulationSetupDialog()
    dialog.accept()
    params = {**dialog.result_data, "mode": "awake", "tutorial_mode": False}
    widget = MainWindow(params)
    yield widget
    widget.close()
    app.processEvents()


@pytest.mark.parametrize("failure", ["header", "write", "flush", "close"])
def test_desktop_recording_failure_pauses_and_resets_toggle(
    window, inject_file, monkeypatch, failure
):
    file = inject_file(failure)
    warnings = []
    monkeypatch.setattr(
        "anasim.ui.main_window.QMessageBox.warning",
        lambda *args: warnings.append(args[2]),
    )
    window.btn_start.click()
    window.btn_record.click()
    if failure == "close":
        window.btn_record.click()
    elif failure != "header":
        window.last_real_time = time.perf_counter() - 0.1
        window.game_loop()

    assert len(warnings) == 1
    assert "injected" in warnings[0]
    assert file.close_attempted
    assert not window.btn_record.isChecked()
    assert window.btn_record.text() == "Record CSV"
    assert not window.engine.running
    assert not window.timer.isActive()
    assert window.time_accumulator == 0
    window.btn_start.click()
    assert window.engine.running


@pytest.mark.parametrize("action", ["close", "reset"])
def test_desktop_finishes_recording_on_close_or_reset(window, action):
    window.btn_record.click()
    file = window.engine.recorder.file
    if action == "close":
        window.close()
    else:
        window.init_simulation()
    assert file.closed
    assert not window.btn_record.isChecked()
