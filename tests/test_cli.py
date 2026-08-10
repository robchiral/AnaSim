from types import SimpleNamespace

import pytest

from anasim.cli import build_models_from_config, run_headless


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


def test_config_builder_preserves_null_hematocrit_derivation():
    patient, config = build_models_from_config({"baseline_hb": 8.0, "baseline_hct": None})

    assert patient.baseline_hb == pytest.approx(8.0)
    assert patient.baseline_hct == pytest.approx(0.24)
    assert not hasattr(config, "baseline_hct")
