import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.arterial import (
    ArterialLineMonitor,
    ArterialPressureSample,
    ArterialWaveformRenderer,
)
from anasim.monitors.cardiac_cycle import CardiacCycle


def test_arterial_line_has_unity_steady_state_gain():
    cycle = CardiacCycle(np.random.default_rng(1))
    cardiac_sample = cycle.seed(60.0, RhythmType.SINUS)
    pressure_sample = ArterialPressureSample(100.0, 100.0, 100.0, 100.0)
    monitor = ArterialLineMonitor()
    reading = monitor.seed(pressure_sample)

    for _ in range(200):
        cardiac_sample = cycle.step(0.01, 60.0, RhythmType.SINUS)
        reading = monitor.step(0.01, cardiac_sample, pressure_sample)

    assert reading.pressure == pytest.approx(100.0, abs=1e-6)


def test_completed_art_reading_tracks_filtered_beat():
    cycle = CardiacCycle(np.random.default_rng(2))
    renderer = ArterialWaveformRenderer(age=40)
    monitor = ArterialLineMonitor()
    cardiac_sample = cycle.seed(75.0, RhythmType.SINUS)
    pressure_sample = renderer.step(cardiac_sample, 90.0, 70.0)
    reading = monitor.seed(pressure_sample)

    for _ in range(320):
        cardiac_sample = cycle.step(0.01, 75.0, RhythmType.SINUS)
        pressure_sample = renderer.step(cardiac_sample, 90.0, 70.0)
        reading = monitor.step(0.01, cardiac_sample, pressure_sample)

    assert reading.systolic > reading.diastolic
    assert reading.mean == pytest.approx(90.0, abs=1.0)


def test_arrest_clears_beat_numerics_while_pressure_line_settles():
    cycle = CardiacCycle(np.random.default_rng(3))
    renderer = ArterialWaveformRenderer(age=40)
    monitor = ArterialLineMonitor()
    cardiac_sample = cycle.seed(70.0, RhythmType.SINUS)
    pressure_sample = renderer.step(cardiac_sample, 90.0, 70.0)
    monitor.seed(pressure_sample)

    cardiac_sample = cycle.step(0.01, 0.0, RhythmType.ASYSTOLE)
    pressure_sample = renderer.step(cardiac_sample, 0.0, 0.0)
    reading = monitor.step(0.01, cardiac_sample, pressure_sample)

    assert reading.systolic == pytest.approx(0.0)
    assert reading.diastolic == pytest.approx(0.0)
    assert reading.mean == pytest.approx(0.0)


@pytest.mark.parametrize("outer_dt", [0.5, 1.0])
def test_engine_art_numerics_are_accurate_with_coarse_step(engine_factory, outer_dt):
    engine = engine_factory(
        config=SimulationConfig(mode="awake", dt=outer_dt, rng_seed=7),
        start=True,
    )

    for _ in range(round(20.0 / outer_dt)):
        engine.step(outer_dt)

    assert engine.state.art_sbp > engine.state.art_dbp + 20.0
    assert engine.state.art_map == pytest.approx(engine.state.map, abs=1.0)
