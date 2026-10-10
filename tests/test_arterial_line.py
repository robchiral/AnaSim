import numpy as np
import pytest
from scipy.integrate import solve_ivp

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.arterial import ArterialLineMonitor, ArterialPressureSample
from anasim.monitors.cardiac_cycle import CardiacCycle


@pytest.mark.parametrize("dt", [0.01, 1.0])
def test_arterial_monitor_tracks_circulation_across_step_sizes(engine_factory, dt):
    engine = engine_factory(config=SimulationConfig(mode="awake", dt=dt, rng_seed=7), start=True)

    for _ in range(round(20.0 / dt)):
        engine.step(dt)
        # Waveform rendering must preserve the circulation's mean pressure and rate.
        assert engine.state.map == pytest.approx(engine.hemo.state.map)
        assert engine.state.hr == pytest.approx(engine.hemo.state.hr)

    assert engine.state.art_sbp > engine.state.art_dbp + 20.0
    assert engine.state.art_map == pytest.approx(engine.state.map, abs=1.0)


@pytest.mark.parametrize("damping", [0.1, 2.0])
def test_filter_matches_continuous_second_order_system(damping):
    """Verify the optimized recurrence against an independent ODE solver."""
    monitor = ArterialLineMonitor()
    cycle = CardiacCycle(np.random.default_rng(7))
    cycle.seed(75.0, RhythmType.SINUS)
    start = ArterialPressureSample(90.0, 120.0, 70.0, 90.0)
    monitor.seed(start)
    monitor.step(0.01, cycle.step(0.01, 75.0, RhythmType.SINUS), start)
    # A changed line setting applies from the next step.
    monitor.damping_ratio = damping
    monitor.seed(start)
    reference = np.array([90.0, 0.0])
    omega = 2.0 * np.pi * monitor.natural_frequency_hz
    for index in range(100):
        dt = (0.01, 0.007, 0.003)[index % 3]
        pressure = 90 + 30 * np.sin(index / 9)
        solution = solve_ivp(
            lambda t, x, pressure=pressure: [x[1], omega**2 * (pressure - x[0]) - 2 * damping * omega * x[1]],
            (0.0, dt), reference, rtol=1e-10, atol=1e-10,
        )
        reference = solution.y[:, -1]
        reading = monitor.step(
            dt, cycle.step(dt, 75.0, RhythmType.SINUS),
            ArterialPressureSample(pressure, 120.0, 60.0, 90.0),
        )
        assert reading.pressure == pytest.approx(max(0.0, reference[0]), abs=1e-7)
