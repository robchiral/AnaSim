import sys
from types import SimpleNamespace

import pytest

from anasim.cli import build_models_from_config, main, run_headless


def test_default_command_launches_the_local_browser_server(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "anasim.local", SimpleNamespace(run=lambda **kwargs: calls.append(kwargs) or 0))
    monkeypatch.setattr("sys.argv", ["anasim", "--no-browser", "--port", "8765", "--record-dir", "output"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    assert calls == [{"port": 8765, "open_browser": False, "recordings_dir": "output"}]


def test_config_builder_rejects_invalid_documents(tmp_path, capsys):
    invalid_documents = (
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
