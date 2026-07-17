import numpy as np
import pytest

from anasim.core.engine import SimulationEngine
from anasim.core import monitors as monitor_core
from anasim.core import projection as projection_core
from anasim.core import runtime as runtime_core
from anasim.core.state import AirwayType, SimulationConfig
from anasim.core.enums import RhythmType
from anasim.patient.patient import Patient
from anasim.physiology.hemodynamics import HemoState
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.respiration import RespState, RespiratoryModel


FLOAT_CONTRACT_FIELDS = (
    "time",
    "propofol_ce",
    "remi_ce",
    "map",
    "hr",
    "co",
    "sbp",
    "dbp",
    "bis",
    "display_bis",
    "fio2",
    "fi_sevo",
    "et_sevo",
    "mac",
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


def _build_engine(dt: float = 0.5) -> SimulationEngine:
    patient = Patient(age=40, weight=70, height=170, sex="male")
    engine = SimulationEngine(patient, SimulationConfig(mode="awake", dt=dt, rng_seed=123))
    engine.state.airway_mode = AirwayType.MASK
    engine._next_nibp_time = 1e9
    engine._monitor_noise_std = np.zeros(3)
    return engine


def _steady_resp_state() -> RespState:
    return RespState(
        rr=12.0,
        vt=500.0,
        mv=6.0,
        va=4.0,
        apnea=False,
        p_alveolar_co2=40.0,
        pa_co2=40.0,
        etco2=38.0,
        p_arterial_o2=95.0,
        sao2=98.0,
        drive_central=1.0,
        muscle_factor=1.0,
    )


def _assert_builtin_float_contract(state) -> None:
    for field_name in FLOAT_CONTRACT_FIELDS:
        value = getattr(state, field_name)
        assert type(value) is float, f"{field_name} should be built-in float, got {type(value).__name__}"


def test_raw_hemodynamics_are_not_overwritten_by_display_smoothing():
    engine = _build_engine(dt=0.5)
    engine.state.map = 45.0
    engine.state.hr = 42.0
    engine.state.sbp = 62.0
    engine.state.dbp = 34.0
    engine.state.sao2 = 98.0
    engine.smooth_map = 90.0
    engine.smooth_hr = 80.0

    hemo_state = HemoState(map=45.0, hr=42.0, sv=55.0, svr=14.0, co=2.3, sbp=62.0, dbp=34.0)
    resp_state = _steady_resp_state()

    monitor_core.step_monitors(engine, 0.5, "EXP", hemo_state, resp_state, DisturbanceEffects())

    assert engine.state.map == pytest.approx(45.0)
    assert engine.state.hr == pytest.approx(42.0)
    assert engine.state.sbp == pytest.approx(62.0)
    assert engine.state.dbp == pytest.approx(34.0)
    assert engine.state.display_map != engine.state.map
    assert engine.state.display_hr != engine.state.hr


def test_arrest_display_bypass_collapses_without_coarse_dt_lag():
    engine = _build_engine(dt=0.5)
    engine.state.map = 0.0
    engine.state.hr = 0.0
    engine.state.sbp = 0.0
    engine.state.dbp = 0.0
    engine.state.sao2 = 98.0
    engine.smooth_map = 88.0
    engine.smooth_hr = 76.0

    hemo_state = HemoState(
        map=0.0,
        hr=0.0,
        sv=0.0,
        svr=0.0,
        co=0.0,
        sbp=0.0,
        dbp=0.0,
        rhythm_type=RhythmType.ASYSTOLE,
    )
    resp_state = _steady_resp_state()

    monitor_core.step_monitors(engine, 0.5, "EXP", hemo_state, resp_state, DisturbanceEffects())

    assert engine.state.display_map == pytest.approx(0.0)
    assert engine.state.display_hr == pytest.approx(0.0)
    assert engine.state.display_sbp == pytest.approx(0.0)
    assert engine.state.display_dbp == pytest.approx(0.0)


@pytest.mark.parametrize("dt", [0.01, 0.1, 0.5])
def test_display_map_response_depends_on_time_not_step_count(dt: float):
    engine = _build_engine(dt=dt)
    engine.state.map = 50.0
    engine.state.hr = 80.0
    engine.state.sbp = 70.0
    engine.state.dbp = 40.0
    engine.state.sao2 = 98.0
    engine.smooth_map = 90.0
    engine.smooth_hr = 80.0

    hemo_state = HemoState(map=50.0, hr=80.0, sv=65.0, svr=16.0, co=5.2, sbp=70.0, dbp=40.0)
    resp_state = _steady_resp_state()

    elapsed = 0.0
    while elapsed < 2.0 - 1e-9:
        monitor_core.step_monitors(engine, dt, "EXP", hemo_state, resp_state, DisturbanceEffects())
        elapsed += dt

    assert engine.state.display_map == pytest.approx(60.5, abs=1.0)


def test_low_flow_high_fio2_does_not_force_arterial_desaturation():
    patient = Patient(age=40, weight=70, height=170, sex="male")
    resp = RespiratoryModel(patient)

    for _ in range(600):
        state = resp.step(
            0.1,
            ce_prop=0.0,
            ce_remi=0.0,
            mech_vent_mv=6.0,
            fio2=1.0,
            ce_roc=0.0,
            et_sevo=0.0,
            mac_sevo=0.0,
            peep=5.0,
            mean_paw=8.0,
            temp_c=37.0,
            mech_rr=12.0,
            mech_vt_l=0.5,
            airway_patency=1.0,
            ventilation_efficiency=1.0,
            vq_mismatch=0.0,
            hb_g_dl=13.5,
            oxygen_delivery_ratio=1.0,
            shiver_level=0.0,
            cardiac_output=0.1,
            metabolic_factor=1.0,
        )

    assert state.p_arterial_o2 > 300.0
    assert state.sao2 > 95.0


def test_low_flow_widens_pa_co2_etco2_gap():
    patient = Patient(age=40, weight=70, height=170, sex="male")

    def run_model(cardiac_output: float):
        resp = RespiratoryModel(patient)
        out = None
        for _ in range(300):
            out = resp.step(
                0.1,
                ce_prop=0.0,
                ce_remi=0.0,
                mech_vent_mv=6.0,
                fio2=0.5,
                ce_roc=0.0,
                et_sevo=0.0,
                mac_sevo=0.0,
                peep=5.0,
                mean_paw=8.0,
                temp_c=37.0,
                mech_rr=12.0,
                mech_vt_l=0.5,
                airway_patency=1.0,
                ventilation_efficiency=1.0,
                vq_mismatch=0.0,
                hb_g_dl=13.5,
                oxygen_delivery_ratio=1.0,
                shiver_level=0.0,
                cardiac_output=cardiac_output,
                metabolic_factor=1.0,
            )
        return out

    normal = run_model(5.0)
    low_flow = run_model(0.5)

    normal_gap = normal.pa_co2 - normal.etco2
    low_flow_gap = low_flow.pa_co2 - low_flow.etco2

    assert low_flow.etco2 < normal.etco2
    assert low_flow_gap > normal_gap + 3.0


@pytest.mark.parametrize(
    ("mode", "maint_type"),
    [
        ("awake", None),
        ("steady_state", "tiva"),
        ("steady_state", "balanced"),
    ],
)
def test_engine_snapshots_preserve_core_physiologic_invariants(mode: str, maint_type: str | None):
    patient = Patient(age=40, weight=70, height=170, sex="male")
    config = SimulationConfig(mode=mode, maint_type=maint_type, rng_seed=123) if maint_type else SimulationConfig(mode=mode, rng_seed=123)
    engine = SimulationEngine(patient, config)

    assert engine.state.co == pytest.approx(engine.state.hr * engine.state.sv / 1000.0, rel=0.03)
    assert engine.state.map == pytest.approx((engine.state.sbp + 2.0 * engine.state.dbp) / 3.0, abs=8.0)
    assert 0.0 <= engine.state.hct <= 0.7
    assert engine.state.hb_g_dl >= 0.0


@pytest.mark.parametrize(
    ("mode", "maint_type"),
    [
        ("awake", None),
        ("steady_state", "tiva"),
        ("steady_state", "balanced"),
    ],
)
def test_public_state_numeric_fields_use_builtin_floats(mode: str, maint_type: str | None):
    patient = Patient(age=40, weight=70, height=170, sex="male")
    config = SimulationConfig(mode=mode, maint_type=maint_type, rng_seed=123) if maint_type else SimulationConfig(mode=mode, rng_seed=123)
    engine = SimulationEngine(patient, config)

    _assert_builtin_float_contract(engine.state)

    engine.start()
    engine.step(0.5)
    _assert_builtin_float_contract(engine.state)


def test_tci_controller_resyncs_after_bolus_and_pk_scaling():
    engine = _build_engine(dt=0.5)
    engine.enable_tci("propofol", 2.0)
    controller = engine.tci_prop
    assert controller is not None

    baseline_signature = controller._model_signature
    engine.give_drug_bolus("Propofol", 100.0)

    assert controller.x[0, 0] == pytest.approx(engine.pk_prop.state.c1)

    engine.hemo.blood_volume = engine.hemo.blood_volume_0 * 0.5
    engine.state.co = engine.hemo.base_co_l_min * 0.5
    runtime_core.update_pk_hemodynamics(engine, engine.state.co)
    engine.sync_active_tci_from_pk("propofol")

    assert controller._model_signature != baseline_signature


def test_awake_initial_snapshot_uses_patient_baselines():
    patient = Patient(
        age=40,
        weight=70,
        height=170,
        sex="male",
        baseline_hr=95.0,
        baseline_map=105.0,
        baseline_rr=16.0,
        baseline_vt=620.0,
    )
    engine = SimulationEngine(
        patient,
        SimulationConfig(mode="awake", baseline_hb=8.0, rng_seed=123),
    )

    assert engine.state.hr == pytest.approx(95.0, abs=1e-3)
    assert engine.state.map == pytest.approx(105.0, abs=1e-3)
    assert engine.state.display_hr == pytest.approx(engine.state.hr, abs=1e-3)
    assert engine.state.display_map == pytest.approx(engine.state.map, abs=1e-3)
    assert engine.state.rr == pytest.approx(16.0, abs=1e-3)
    assert engine.state.vt == pytest.approx(620.0, abs=1e-3)
    assert engine.state.hb_g_dl == pytest.approx(8.0, abs=1e-6)
    assert engine.state.hct == pytest.approx(0.24, abs=1e-6)
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
    engine_runtime._sync_display_state_from_raw()

    for field_name in ("map", "hr", "rr", "vt", "mv", "etco2", "pa_co2", "pao2", "sao2", "spo2"):
        assert getattr(engine_sync.state, field_name) == pytest.approx(getattr(engine_runtime.state, field_name))


def test_steady_state_tiva_snapshot_uses_live_model_state():
    awake = SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="awake", rng_seed=123),
    )
    engine = SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=123),
    )

    expected_bis = engine.bis.compute_bis(
        engine.state.propofol_ce,
        engine.state.remi_ce,
        u_volatile=engine.state.mac_sevo,
    )

    assert engine.state.bis == pytest.approx(expected_bis, abs=1e-3)
    assert engine.state.bis < 55.0
    assert 65.0 <= engine.state.map <= 85.0
    assert engine.tci_nore is not None
    assert engine.state.nore_ce > awake.state.nore_ce
    assert engine.state.bis != pytest.approx(45.0, abs=1e-3)
    assert engine.state.fi_sevo == pytest.approx(0.0, abs=1e-6)
    assert engine.state.et_sevo == pytest.approx(0.0, abs=1e-6)
    assert engine.state.nibp_map == pytest.approx(engine.state.map, abs=1e-3)
    assert engine.state.fluid_in_ml == pytest.approx(0.0, abs=1e-6)
    assert engine.state.urine_out_ml == pytest.approx(0.0, abs=1e-6)
    assert engine.state.temp_c == pytest.approx(37.0, abs=1e-6)


def test_steady_state_balanced_snapshot_syncs_volatile_state():
    engine = SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=123),
    )

    expected_bis = engine.bis.compute_bis(
        engine.state.propofol_ce,
        engine.state.remi_ce,
        u_volatile=engine.state.mac_sevo,
    )

    assert engine.state.fi_sevo > 0.0
    assert engine.state.et_sevo > 0.0
    assert 0.8 <= engine.state.mac_sevo <= 1.05
    assert 38.0 <= engine.state.bis <= 50.0
    assert 70.0 <= engine.state.map <= 85.0
    assert engine.state.bis == pytest.approx(expected_bis, abs=1e-3)
    assert engine.state.nibp_map == pytest.approx(engine.state.map, abs=1e-3)
    assert engine.state.fluid_in_ml == pytest.approx(0.0, abs=1e-6)
    assert engine.state.urine_out_ml == pytest.approx(0.0, abs=1e-6)
    assert engine.state.temp_c == pytest.approx(37.0, abs=1e-6)


@pytest.mark.parametrize(
    ("maint_type", "bis_band", "map_band"),
    [
        ("tiva", (50.0, 58.0), (60.0, 86.0)),
        ("balanced", (38.0, 50.0), (70.0, 82.0)),
    ],
)
def test_steady_state_profiles_remain_in_band_for_first_15_minutes(maint_type, bis_band, map_band):
    engine = SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="steady_state", maint_type=maint_type, rng_seed=123),
    )
    engine.start()

    bis_values = [float(engine.state.bis)]
    map_values = [float(engine.state.map)]
    for _ in range(900):
        engine.step(1.0)
        bis_values.append(float(engine.state.bis))
        map_values.append(float(engine.state.map))

    assert min(bis_values) >= bis_band[0]
    assert max(bis_values) <= bis_band[1]
    assert min(map_values) >= map_band[0]
    assert max(map_values) <= map_band[1]


def test_baseline_hct_is_derived_from_hb_when_omitted():
    engine = SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="awake", baseline_hb=8.0),
    )

    assert engine.patient.baseline_hct == pytest.approx(0.24, abs=1e-6)
    assert engine.state.hct == pytest.approx(0.24, abs=1e-6)


def test_explicit_baseline_hct_is_preserved_when_consistent():
    engine = SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="awake", baseline_hb=8.0, baseline_hct=0.27),
    )

    assert engine.patient.baseline_hct == pytest.approx(0.27, abs=1e-6)
    assert engine.state.hct == pytest.approx(0.27, abs=1e-6)


def test_grossly_inconsistent_hb_hct_pair_is_rejected():
    with pytest.raises(ValueError, match="grossly inconsistent"):
        SimulationEngine(
            Patient(age=40, weight=70, height=170, sex="male"),
            SimulationConfig(mode="awake", baseline_hb=8.0, baseline_hct=0.42),
        )


@pytest.mark.parametrize("age", [17, 71])
def test_patient_age_outside_adult_model_domain_is_rejected(age: int):
    with pytest.raises(ValueError, match="18 and 70"):
        Patient(age=age)


def test_unknown_model_selection_is_rejected_instead_of_falling_back():
    with pytest.raises(ValueError, match="resp_model"):
        SimulationConfig(resp_model="legacy")


def test_capno_numeric_requires_recent_exhaled_gas():
    engine = _build_engine(dt=0.5)
    hemo_state = HemoState(map=90.0, hr=70.0, sv=70.0, svr=18.0, co=4.9)
    resp_state = _steady_resp_state()

    engine.state.airway_mode = AirwayType.NONE
    monitor_core.step_monitors(engine, 0.5, "EXP", hemo_state, resp_state, DisturbanceEffects())
    assert not engine.state.etco2_signal_valid
    assert engine.state.display_etco2 == 0.0

    engine.state.airway_mode = AirwayType.MASK
    engine.state.rr = 12.0
    for _ in range(5):
        engine.state.time = 2.0
        monitor_core.step_monitors(engine, 0.5, "EXP", hemo_state, resp_state, DisturbanceEffects())
    engine.state.time = 0.0
    monitor_core.step_monitors(engine, 0.5, "INSP", hemo_state, resp_state, DisturbanceEffects())
    assert engine.state.etco2_signal_valid
    assert engine.state.display_etco2 > 30.0

    engine.state.rr = 0.0
    resp_state.rr = 0.0
    resp_state.apnea = True
    for _ in range(31):
        monitor_core.step_monitors(engine, 0.5, "EXP", hemo_state, resp_state, DisturbanceEffects())
    assert not engine.state.etco2_signal_valid
    assert engine.state.display_etco2 == 0.0
