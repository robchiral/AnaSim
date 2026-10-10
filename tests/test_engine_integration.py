import pytest

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


def test_peep_reduces_preload_without_automatically_recruiting_lung():
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
    aeration_low = engine.aeration.recruited

    engine.set_vent_settings(rr=12, vt=0.5, peep=15.0, ie="1:2", mode="VCV")
    _run_for(engine, 60.0, dt=0.5)

    assert engine.state.pit > pit_low + 0.5
    assert engine.hemo.state.preload_factor < preload_low
    assert engine.state.map < map_low
    assert engine.aeration.recruited == pytest.approx(aeration_low, abs=1e-4)


def test_positive_pressure_reduces_preload_vs_spontaneous(engine):
    engine.start()
    engine.set_airway_mode("Mask")
    _run_for(engine, 20.0)
    preload_spont = engine.hemo.state.preload_factor

    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    _run_for(engine, 20.0)

    assert engine.hemo.state.preload_factor < preload_spont
