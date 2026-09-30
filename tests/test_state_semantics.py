import pytest

from anasim.cli import build_models_from_config
from anasim.core import monitors as monitor_core
from anasim.core import projection as projection_core
from anasim.core.engine import SimulationEngine
from anasim.core.state import AirwayType, SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.hemodynamics import HemoState
from anasim.physiology.respiration import RespState

FLOAT_CONTRACT_FIELDS = (
    "time",
    "propofol_ce",
    "remi_ce",
    "map",
    "hr",
    "co",
    "sbp",
    "dbp",
    "art_pressure",
    "art_sbp",
    "art_dbp",
    "art_map",
    "bis",
    "display_bis",
    "fio2",
    "fi_sevo",
    "et_sevo",
    "mac",
    "et_mac",
    "rr",
    "vt",
    "mv",
    "etco2",
    "spo2",
    "nibp_map",
    "hb_g_dl",
    "hct",
    "temp_c",
    "oxygen_delivery_ratio",
)


def _assert_builtin_float_contract(state) -> None:
    for field_name in FLOAT_CONTRACT_FIELDS:
        value = getattr(state, field_name)
        assert type(value) is float, f"{field_name} should be built-in float, got {type(value).__name__}"


def test_arterial_renderer_does_not_overwrite_su_mean_state():
    patient = Patient(age=40, weight=70, height=170, sex="male")
    engine = SimulationEngine(patient, SimulationConfig(mode="awake", dt=0.5, rng_seed=123))
    engine.state.airway_mode = AirwayType.MASK
    engine.state.map = 45.0
    engine.state.hr = 42.0
    engine.state.sao2 = 98.0

    hemo_state = HemoState(map=45.0, hr=42.0, sv=55.0, svr=14.0, co=2.3)
    resp_state = RespState(
        rr=12.0, vt=500.0, mv=6.0, va=4.0, apnea=False, p_alveolar_co2=40.0, pa_co2=40.0,
        etco2=38.0, p_arterial_o2=95.0, sao2=98.0, drive_central=1.0, muscle_factor=1.0,
    )

    monitor_core.step_monitors(engine, 0.5, "EXP", hemo_state, resp_state, DisturbanceEffects())

    assert engine.state.map == pytest.approx(45.0)
    assert engine.state.hr == pytest.approx(42.0)
    assert engine.state.sbp > engine.state.dbp
    assert engine.state.art_sbp >= engine.state.art_dbp


@pytest.mark.parametrize(
    ("mode", "maint_type"),
    [
        ("awake", None),
        ("steady_state", "tiva"),
        ("steady_state", "balanced"),
    ],
)
def test_engine_snapshots_preserve_public_contract(mode: str, maint_type: str | None):
    patient = Patient(age=40, weight=70, height=170, sex="male")
    config = SimulationConfig(mode=mode, maint_type=maint_type, rng_seed=123) if maint_type else SimulationConfig(mode=mode, rng_seed=123)
    engine = SimulationEngine(patient, config)

    assert engine.state.co == pytest.approx(engine.state.hr * engine.state.sv / 1000.0, rel=0.03)
    assert engine.state.map == pytest.approx((engine.state.sbp + 2.0 * engine.state.dbp) / 3.0, abs=8.0)
    assert 0.0 <= engine.state.hct <= 0.7
    assert engine.state.hb_g_dl >= 0.0
    _assert_builtin_float_contract(engine.state)

    engine.start()
    engine.step(0.5)
    _assert_builtin_float_contract(engine.state)


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


def test_startup_projection_matches_runtime_projection_path():
    patient_a = Patient(age=40, weight=70, height=170, sex="male")
    patient_b = Patient(age=40, weight=70, height=170, sex="male")
    config_a = SimulationConfig(mode="awake", rng_seed=123)
    config_b = SimulationConfig(mode="awake", rng_seed=123)
    engine_sync = SimulationEngine(patient_a, config_a)
    engine_runtime = SimulationEngine(patient_b, config_b)

    projection_core.sync_state_from_models(engine_sync)
    hemo_state = engine_runtime.hemo.state
    resp_state = projection_core.snapshot_respiratory_state(engine_runtime, hemo_state)
    snapshot = projection_core.build_snapshot_from_models(engine_runtime, hemo_state, resp_state)
    projection_core.project_runtime_physiology(engine_runtime, snapshot)
    projection_core.sync_monitor_baselines(engine_runtime)

    for field_name in ("map", "hr", "rr", "vt", "mv", "etco2", "pa_co2", "pao2", "sao2", "spo2"):
        assert getattr(engine_sync.state, field_name) == pytest.approx(getattr(engine_runtime.state, field_name))


@pytest.mark.parametrize("maint_type", ["tiva", "balanced"])
def test_steady_state_starts_from_live_model_state(maint_type):
    engine = SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="steady_state", maint_type=maint_type, rng_seed=123),
    )
    state = engine.state
    expected_bis = engine.bis.compute_bis(state.propofol_ce, state.remi_ce, mac_sevo=state.mac_sevo)

    assert state.bis == pytest.approx(expected_bis, abs=1e-3)
    assert 38.0 <= state.bis <= 55.0
    assert 65.0 <= state.map <= 85.0
    assert state.nibp_map == pytest.approx(state.map, abs=1e-3)
    # Hidden settling does not count toward visible totals.
    assert state.fluid_in_ml == pytest.approx(0.0, abs=1e-6)
    assert state.urine_out_ml == pytest.approx(0.0, abs=1e-6)
    assert state.temp_c == pytest.approx(37.0, abs=1e-6)
    if maint_type == "tiva":
        assert engine.tci_nore is not None
        assert state.fi_sevo == pytest.approx(0.0, abs=1e-6)
    else:
        assert state.fi_sevo > 0.0
        assert state.et_sevo > 0.0
        assert 0.8 <= state.mac_sevo <= 1.05
