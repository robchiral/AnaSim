import numpy as np
import pytest
from scipy.integrate import solve_ivp

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


@pytest.mark.parametrize("damping", [0.1, 0.65, 1.0, 2.0])
def test_filter_matches_continuous_second_order_system(damping):
    """Verify the optimized recurrence against an independent ODE solver."""
    monitor = ArterialLineMonitor(damping_ratio=damping)
    cycle = CardiacCycle(np.random.default_rng(7))
    cycle.seed(75.0, RhythmType.SINUS)
    monitor.seed(ArterialPressureSample(90.0, 120.0, 70.0, 90.0))
    reference = np.array([90.0, 0.0])
    omega = 2.0 * np.pi * monitor.natural_frequency_hz
    for index in range(100):
        dt = (0.01, 0.007, 0.003)[index % 3]
        pressure = 90 + 30 * np.sin(index / 9)
        solution = solve_ivp(
            lambda t, x: [x[1], omega**2 * (pressure - x[0]) - 2 * damping * omega * x[1]],
            (0.0, dt), reference, rtol=1e-10, atol=1e-10,
        )
        reference = solution.y[:, -1]
        reading = monitor.step(
            dt, cycle.step(dt, 75.0, RhythmType.SINUS),
            ArterialPressureSample(pressure, 120.0, 60.0, 90.0),
        )
        assert reading.pressure == pytest.approx(max(0.0, reference[0]), abs=1e-7)


def test_filter_cache_is_bounded_and_tracks_line_settings():
    monitor = ArterialLineMonitor()
    for dt in np.linspace(0.001, 0.01, 100):
        monitor._coefficients(dt)
    assert len(monitor._coefficient_cache) <= 16
    original = monitor._coefficients(0.01)
    monitor.damping_ratio = 0.1
    assert monitor._coefficients(0.01) != original
    assert monitor._coefficients(0.01) == ArterialLineMonitor(damping_ratio=0.1)._coefficients(0.01)
