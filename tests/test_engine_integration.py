import pytest

from anasim.core import runtime as runtime_core
from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient


@pytest.fixture
def engine(patient):
    return SimulationEngine(patient, SimulationConfig(mode="awake", dt=0.5))


def _run_for(engine, seconds: float, dt: float = 0.1) -> None:
    for _ in range(max(1, int(seconds / dt))):
        engine.step(dt)


def test_output_buffer_holds_one_respiratory_sweep_of_per_step_samples(patient):
    engine = SimulationEngine(patient, SimulationConfig(mode="awake", dt=0.01))
    engine.start()
    initial_len = len(engine.output_buffer)
    engine.step(0.01)
    engine.step(0.01)
    assert len(engine.output_buffer) == initial_len + 2
    assert engine.output_buffer[-1].time == pytest.approx(engine.state.time)

    _run_for(engine, 22.0)
    span = engine.output_buffer[-1].time - engine.output_buffer[0].time
    assert 19.8 <= span <= 20.0 + 1e-9


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


def test_peep_recruits_lung_but_reduces_preload():
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
    pao2_low = engine.state.pao2

    engine.set_vent_settings(rr=12, vt=0.5, peep=15.0, ie="1:2", mode="VCV")
    _run_for(engine, 60.0, dt=0.5)

    assert engine.state.pit > pit_low + 0.5
    assert engine.hemo.state.preload_factor < preload_low
    assert engine.state.map < map_low
    assert engine.state.pao2 > pao2_low


def test_positive_pressure_reduces_preload_vs_spontaneous(engine):
    engine.start()
    engine.set_airway_mode("Mask")
    _run_for(engine, 20.0)
    preload_spont = engine.hemo.state.preload_factor

    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    _run_for(engine, 20.0)

    assert engine.hemo.state.preload_factor < preload_spont
