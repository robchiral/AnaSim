import sys
from types import SimpleNamespace
from typing import Any

import pytest

from anasim.cli import build_models_from_config, main, run_headless
from anasim.core.engine import SimulationEngine


def test_default_command_launches_the_local_browser_server(monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setitem(sys.modules, "anasim.local", SimpleNamespace(run=run))
    monkeypatch.setattr("sys.argv", ["anasim", "--no-browser", "--port", "8765", "--record-dir", "output"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    assert calls == [{"port": 8765, "open_browser": False, "recordings_dir": "output"}]


def test_config_builder_rejects_invalid_documents(tmp_path, capsys):
    invalid_documents: tuple[tuple[Any, str], ...] = (
        ([], "JSON object"),
        ({"weight": float("nan")}, "weight"),
        ({"pk_model_propofol": []}, "pk_model_propofol"),
        ({"renal_status": "Normal"}, "Unknown configuration"),
    )
    for config_data, error in invalid_documents:
        with pytest.raises(ValueError, match=error):
            build_models_from_config(config_data)

    config_path = tmp_path / "invalid.json"
    config_path.write_text('{"disturbance_profile": "unknown"}')
    args = SimpleNamespace(
        config=str(config_path),
        duration=1.0,
        record=False,
        record_dir="recordings",
        record_interval=1.0,
    )
    with pytest.raises(SystemExit, match="1"):
        run_headless(args)
    assert "Error loading config: dist_profile" in capsys.readouterr().out


def test_headless_prints_arterial_map_during_arrest_and_blanks_failed_nibp(tmp_path, capsys, monkeypatch):
    class _ArrestEngine(SimulationEngine):
        def start(self):
            super().start()
            self.set_rhythm("VFIB")
            self.measure_nibp()

    monkeypatch.setattr("anasim.cli.SimulationEngine", _ArrestEngine)

    def run_case(arterial_line: bool) -> str:
        cfg_path = tmp_path / f"cfg_{int(arterial_line)}.json"
        cfg_path.write_text(f'{{"dt": 0.2, "arterial_line_enabled": {str(arterial_line).lower()}}}')
        run_headless(
            SimpleNamespace(
                config=str(cfg_path), duration=20.4, record=False, record_dir="recordings", record_interval=1.0,
            )
        )
        return capsys.readouterr().out

    art_lines = [line for line in run_case(True).splitlines() if line.startswith("Time: 20.")]
    nibp_lines = [line for line in run_case(False).splitlines() if line.startswith("Time: 20.")]
    assert art_lines and "MAP: --" not in art_lines[-1]
    assert nibp_lines and "MAP: -- |" in nibp_lines[-1]


def test_headless_zero_record_interval_records_every_step(tmp_path):
    """--record-interval 0 records every simulation step."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"dt": 0.2}')
    out_dir = tmp_path / "rec"
    run_headless(
        SimpleNamespace(
            config=str(cfg_path),
            duration=1.0,
            record=True,
            record_dir=str(out_dir),
            record_interval=0.0,
        )
    )
    files = list(out_dir.glob("*.csv")) + list(out_dir.glob("*.parquet")) + list(out_dir.glob("*.json*"))
    assert files
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) >= 6
