import pytest

from anasim.cli import build_models_from_config
from anasim.core.engine import SimulationEngine


def test_awake_initial_snapshot_uses_patient_baselines():
    patient, config = build_models_from_config(
        {
            "baseline_hb": 8.0,
            "baseline_hct": None,
            "baseline_hr": 95.0,
            "baseline_map": 105.0,
            "baseline_rr": 16.0,
            "baseline_vt": 620.0,
            "rng_seed": 123,
        }
    )
    engine = SimulationEngine(patient, config)

    assert engine.state.hr == pytest.approx(95.0, abs=1e-3)
    assert engine.state.map == pytest.approx(105.0, abs=1e-3)
    assert engine.state.display_hr == pytest.approx(engine.state.hr, abs=1e-3)
    assert engine.state.art_map == pytest.approx(engine.state.map, abs=1e-3)
    assert engine.state.rr == pytest.approx(16.0, abs=1e-3)
    assert engine.state.vt == pytest.approx(620.0, abs=1e-3)
    assert engine.state.hb_g_dl == pytest.approx(8.0, abs=1e-6)
    # A null configured hematocrit is derived from hemoglobin.
    assert engine.state.hct == pytest.approx(0.24, abs=1e-6)
    assert engine.patient.baseline_hct == pytest.approx(0.24, abs=1e-6)
    assert engine.state.nibp_map == pytest.approx(engine.state.map, abs=1e-3)
