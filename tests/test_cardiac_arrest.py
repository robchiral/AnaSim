import pytest

from anasim.core import runtime as runtime_core
from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig


def _hold(engine, seconds, hr=None, map_value=None):
    for _ in range(int(seconds / 0.1)):
        if hr is not None:
            engine.state.hr = hr
        if map_value is not None:
            engine.state.map = map_value
        runtime_core.check_cardiac_arrest(engine, 0.1)


@pytest.fixture
def arrest_engine(engine_factory):
    return engine_factory(config=SimulationConfig(end_on_cardiac_arrest=True))


def test_arrest_endpoint_is_off_by_default(engine_factory):
    engine = engine_factory(config=SimulationConfig())
    _hold(engine, 20.0, hr=0.0, map_value=0.0)
    assert not engine.state.cardiac_arrest


def test_extreme_bradycardia_confirms_after_fifteen_seconds(arrest_engine):
    _hold(arrest_engine, 10.0, hr=0.0, map_value=80.0)
    assert not arrest_engine.state.cardiac_arrest
    _hold(arrest_engine, 10.0, hr=0.0, map_value=80.0)
    assert arrest_engine.state.cardiac_arrest
    assert "bradycardia" in arrest_engine.state.arrest_reason


def test_pulseless_electrical_activity(arrest_engine):
    _hold(arrest_engine, 20.0, map_value=5.0)
    assert arrest_engine.state.cardiac_arrest
    assert "Pulseless electrical activity" in arrest_engine.state.arrest_reason


def test_transient_pulselessness_resets(arrest_engine):
    _hold(arrest_engine, 10.0, map_value=5.0)
    _hold(arrest_engine, 1.0, map_value=70.0)
    _hold(arrest_engine, 10.0, map_value=5.0)
    assert not arrest_engine.state.cardiac_arrest


def test_ventricular_fibrillation_reports_rhythm(engine_factory):
    engine = engine_factory(
        config=SimulationConfig(end_on_cardiac_arrest=True, dt=0.1), start=True
    )
    engine.set_rhythm("VFIB")
    for _ in range(200):
        engine.step(0.1)
    assert engine.state.cardiac_arrest
    assert engine.state.arrest_reason == RhythmType.VFIB.value
