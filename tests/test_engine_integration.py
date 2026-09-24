import numpy as np
import pytest

from anasim.core import monitors as monitor_core
from anasim.core import runtime as runtime_core
from anasim.core.drug_registry import get_drug_spec
from anasim.core.engine import SimulationEngine
from anasim.core.state import AirwayType, SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.hemodynamics import HemoState
from anasim.physiology.respiration import RespState


@pytest.fixture
def engine(patient):
    return SimulationEngine(patient, SimulationConfig(mode="awake", dt=0.5))


def _run_for(engine, seconds: float, dt: float = 0.1) -> None:
    for _ in range(max(1, int(seconds / dt))):
        engine.step(dt)


def test_output_buffer_holds_ten_seconds_of_per_step_samples(patient):
    engine = SimulationEngine(patient, SimulationConfig(mode="awake", dt=0.01))
    engine.start()
    initial_len = len(engine.output_buffer)
    engine.step(0.01)
    engine.step(0.01)
    assert len(engine.output_buffer) == initial_len + 2
    assert engine.output_buffer[-1].time == pytest.approx(engine.state.time)

    _run_for(engine, 12.0)
    span = engine.output_buffer[-1].time - engine.output_buffer[0].time
    assert 9.8 <= span <= 10.0 + 1e-9


def test_tci_seeds_state_and_caps_rate(engine):
    """TCI seeds from the PK state and uses each drug's registry pump limit."""
    weight = engine.patient.weight
    cases = (
        ("propofol", "effect_site", "pk_prop", "tci_prop", {"c1": 1.0, "c2": 2.0, "c3": 3.0, "ce": 4.0}),
        ("remi", "effect_site", "pk_remi", "tci_remi", {"c1": 1.5, "c2": 2.5, "c3": 3.5, "ce": 4.5}),
        ("nore", "plasma", "pk_nore", "tci_nore", {"c1": 5.0, "c2": 2.0, "ce": 4.0}),
        ("epi", "plasma", "pk_epi", "tci_epi", {"c1": 3.0, "ce": 2.0}),
        ("phenyl", "plasma", "pk_phenyl", "tci_phenyl", {"c1": 6.0, "c2": 1.0, "ce": 2.5}),
        ("roc", "effect_site", "pk_roc", "tci_roc", {"c1": 2.0, "c2": 2.0, "c3": 2.0, "ce": 2.0}),
    )
    for drug, mode, pk_attr, controller_attr, state_values in cases:
        pk_model = getattr(engine, pk_attr)
        for key, value in state_values.items():
            setattr(pk_model.state, key, value)

        engine.enable_tci(drug, 2.0, mode)
        controller = getattr(engine, controller_attr)
        assert controller is not None

        expected = [state_values.get(name, 0.0) for name in pk_model.state_fields]
        assert controller.x[:, 0] == pytest.approx(expected)
        assert pk_model.state_fields[0] == "c1" and pk_model.state_fields[-1] == "ce"

        expected_max_rate = get_drug_spec(drug).max_rate.internal_rate(weight)
        assert controller.max_rate == pytest.approx(expected_max_rate)


def test_hr_disturbance_is_not_applied_again_by_the_monitor():
    patient = Patient(age=40, weight=70, height=170, sex="male")
    config = SimulationConfig(mode="awake", dt=0.5)

    def build_engine():
        eng = SimulationEngine(patient, config)
        eng.state.time = 0.0
        eng._next_nibp_time = 1e9
        eng.state.airway_mode = AirwayType.MASK
        eng.rng = np.random.default_rng(123)
        return eng

    engine1 = build_engine()
    engine2 = build_engine()

    hemo_state = HemoState(map=80.0, hr=85.0, sv=70.0, svr=16.0, co=5.0)
    resp_state = RespState(
        rr=12.0,
        vt=500.0,
        mv=6.0,
        va=4.0,
        apnea=False,
        p_alveolar_co2=40.0,
        etco2=40.0,
        p_arterial_o2=95.0,
        drive_central=1.0,
        muscle_factor=1.0,
    )
    monitor_core.step_monitors(engine1, 1.0, "EXP", hemo_state, resp_state, DisturbanceEffects())
    monitor_core.step_monitors(engine2, 1.0, "EXP", hemo_state, resp_state, DisturbanceEffects(hr=10.0))

    assert engine1.state.display_hr == pytest.approx(engine2.state.display_hr, rel=1e-6)


def test_propofol_central_volume_scales_with_blood_volume():
    engine = SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="awake", dt=0.5),
    )
    base_v1 = engine.pk_prop.v1
    engine.hemo.blood_volume = engine.hemo.blood_volume_0 * 0.5
    engine.state.co = engine.hemo.base_co_l_min * 0.5

    runtime_core.update_pk_hemodynamics(engine, engine.state.co)

    assert engine.pk_prop.v1 == pytest.approx(base_v1 * 0.5, rel=0.05)


def test_peep_increases_pit_and_reduces_preload():
    engine = SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="steady_state", maint_type="tiva", dt=0.5, rng_seed=123),
    )
    engine.start()

    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    _run_for(engine, 60.0, dt=0.5)
    pit_low = engine.state.pit
    preload_low = engine.hemo.state.preload_factor
    map_low = engine.state.map

    engine.set_vent_settings(rr=12, vt=0.5, peep=15.0, ie="1:2", mode="VCV")
    _run_for(engine, 60.0, dt=0.5)

    assert engine.state.pit > pit_low + 0.5
    assert engine.hemo.state.preload_factor < preload_low
    assert engine.state.map < map_low


def test_positive_pressure_reduces_preload_vs_spontaneous(engine):
    engine.start()
    engine.set_airway_mode("Mask")
    _run_for(engine, 20.0)
    preload_spont = engine.hemo.state.preload_factor

    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    _run_for(engine, 20.0)

    assert engine.hemo.state.preload_factor < preload_spont
