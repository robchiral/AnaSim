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

    engine.start()
    engine.step(0.1)
    assert engine.state.alarms == {}

    engine.give_drug_bolus("roc", 50.0)
    for _ in range(600):
        engine.step(0.1)
    snapshot = engine.get_latest_state()
    assert snapshot.alarms["BIS"]["high"]
    snapshot.alarms["BIS"]["high"] = False
    assert engine.state.alarms["BIS"]["high"]
    snapshot.alarms.clear()
    assert "BIS" in engine.state.alarms
    engine.step(0.1)
    assert snapshot.alarms == {}


def test_steady_state_startup_capno_sync_and_projection_preserves_temperature():
    """Steady-state startup resets breath_count and syncs capno_co2, and projection
    preserves live hyperthermic temperature (#32).
    """
    from anasim.core.projection import sync_state_from_models

    patient, config = build_models_from_config({"mode": "steady_state", "rng_seed": 1})
    engine = SimulationEngine(patient, config)
    engine.start()
    assert engine.vent.breath_count == 0
    assert engine.state.capno_co2 == pytest.approx(engine.capno.co2)

    engine.state.temp_c = 39.2
    sync_state_from_models(engine)
    assert engine.state.temp_c == pytest.approx(39.2)
