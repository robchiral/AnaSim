import math
from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.arterial import (
    NO_PULSE_TIMEOUT_S,
    ArterialLineMonitor,
    ArterialPressureSample,
    ArterialWaveformRenderer,
)
from anasim.monitors.cardiac_cycle import CardiacCycle


@pytest.mark.parametrize("dt", [0.01, 1.0])
def test_arterial_monitor_tracks_circulation_across_step_sizes(engine_factory, dt):
    engine = engine_factory(config=SimulationConfig(mode="awake", dt=dt, rng_seed=7), start=True)

    for _ in range(round(8.0 / dt)):
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


def test_stopping_compressions_holds_then_clears_pulsatile_numerics(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=7), start=True)
    engine.set_rhythm("VFIB")
    for _ in range(150):
        engine.step(0.1)
    assert math.isnan(engine.state.art_sbp) and math.isnan(engine.state.art_dbp)

    engine.set_chest_compressions(True)
    for _ in range(100):
        engine.step(0.1)
    cpr_sbp, cpr_dbp, cpr_map = engine.state.art_sbp, engine.state.art_dbp, engine.state.art_map
    assert 60.0 < cpr_sbp < 80.0 and 20.0 < cpr_dbp < 30.0

    # Stop mid-compression and verify SBP/DBP never equalize to static runoff pressure.
    engine.set_chest_compressions(False)
    for step in range(100):
        engine.step(0.1)
        elapsed = (step + 1) * 0.1
        state = engine.state
        assert state.art_map < cpr_map + 1.0
        if not math.isnan(state.art_sbp):
            assert state.art_dbp <= state.art_map <= state.art_sbp
        if elapsed < 2.5:
            assert state.art_sbp == pytest.approx(cpr_sbp)
            assert state.art_dbp == pytest.approx(cpr_dbp)
            assert state.art_map == pytest.approx(cpr_map)
        elif elapsed >= NO_PULSE_TIMEOUT_S + 0.1:
            assert math.isnan(state.art_sbp) and math.isnan(state.art_dbp)

    assert engine.state.art_map == pytest.approx(engine.hemo.state.mcfp, abs=0.5)
    assert engine.state.alarms["MAP"]["low"]

    engine.set_chest_compressions(True)
    for _ in range(6):
        engine.step(0.1)
        state = engine.state
        if not math.isnan(state.art_sbp):
            assert state.art_dbp <= state.art_map <= state.art_sbp
    assert engine.state.art_sbp > engine.state.art_dbp + 20.0


def test_pea_runoff_rejects_nonpulsatile_organized_cycles():
    cycle = CardiacCycle(np.random.default_rng(3))
    renderer = ArterialWaveformRenderer(age=50)
    monitor = ArterialLineMonitor()
    seeded = cycle.seed(75.0, RhythmType.SINUS)
    monitor.seed(renderer.step(seeded, 85.0, 70.0, dt=0.01, static_pressure=10.0))
    for _ in range(300):
        beat = cycle.step(0.01, 75.0, RhythmType.SINUS)
        monitor.advance(0.01, beat, renderer.step(beat, 85.0, 70.0, dt=0.01, static_pressure=10.0))
    last_sbp, last_dbp = monitor.reading.systolic, monitor.reading.diastolic
    assert last_sbp > last_dbp + 25.0

    # Advance to the next R wave so no pre-arrest upstroke remains in progress.
    while True:
        beat = cycle.step(0.01, 75.0, RhythmType.SINUS)
        if beat.beat_started:
            monitor.advance(0.01, beat, renderer.step(beat, 0.0, 0.0, dt=0.01, static_pressure=10.0))
            break
        monitor.advance(0.01, beat, renderer.step(beat, 85.0, 70.0, dt=0.01, static_pressure=10.0))
    held = monitor.reading

    pea_pressures = []
    for step in range(1500):
        beat = cycle.step(0.01, 75.0, RhythmType.SINUS)
        sample = renderer.step(beat, 0.0, 0.0, dt=0.01, static_pressure=10.0)
        pea_pressures.append(sample.pressure)
        reading = monitor.step(0.01, beat, sample)
        if not math.isnan(reading.systolic):
            assert (reading.systolic, reading.diastolic, reading.mean) == (held.systolic, held.diastolic, held.mean)
        if (step + 1) * 0.01 >= NO_PULSE_TIMEOUT_S + 0.8:
            assert math.isnan(reading.systolic) and math.isnan(reading.diastolic)

    assert float(np.max(np.diff(pea_pressures))) <= 1e-9
    assert pea_pressures[-1] == pytest.approx(10.0, abs=0.2)
    assert monitor.reading.mean == pytest.approx(10.0, abs=0.1)

    recovery_pressures = []
    for _ in range(100):
        beat = cycle.step(0.01, 75.0, RhythmType.SINUS)
        sample = renderer.step(beat, 85.0, 70.0, dt=0.01, static_pressure=10.0)
        recovery_pressures.append(sample.pressure)
        reading = monitor.step(0.01, beat, sample)
        if not math.isnan(reading.systolic):
            assert reading.diastolic <= reading.mean <= reading.systolic
    assert min(recovery_pressures) >= 10.0 - 0.1
    assert reading.systolic > reading.diastolic + 20.0


@pytest.mark.parametrize(("hr", "conduction_failure"), [(15.0, 0.0), (10.0, 0.0), (10.0, 0.95)])
def test_slow_and_agonal_pulsatile_rhythms_retain_numerics(hr, conduction_failure):
    cycle = CardiacCycle(np.random.default_rng(7))
    renderer = ArterialWaveformRenderer(age=40)
    monitor = ArterialLineMonitor()
    seeded = cycle.seed(hr, RhythmType.SINUS_BRADY)
    # 40 mL at MAP 90 mmHg gives ~27 mmHg pulse pressure in a 40-year-old.
    monitor.seed(renderer.step(seeded, 90.0, 40.0, static_pressure=10.0))

    steady_maps = []
    for step in range(4000):
        beat = cycle.step(0.01, hr, RhythmType.SINUS_BRADY, conduction_failure=conduction_failure)
        reading = monitor.step(0.01, beat, renderer.step(beat, 90.0, 40.0, dt=0.01, static_pressure=10.0))
        assert reading.systolic > reading.diastolic + 20.0
        assert reading.diastolic <= reading.mean <= reading.systolic
        if step >= 1500:
            steady_maps.append(reading.mean)

    if conduction_failure == 0.0:
        assert min(steady_maps) == pytest.approx(90.0, abs=0.2)
        assert max(steady_maps) == pytest.approx(90.0, abs=0.2)

    # Recovering from arrest into a 15 bpm rhythm publishes coherent numerics on the first pulse.
    asystole = cycle.seed(0.0, RhythmType.ASYSTOLE)
    monitor.seed(renderer.step(asystole, 0.0, 0.0, static_pressure=10.0))
    assert math.isnan(monitor.reading.systolic) and math.isnan(monitor.reading.diastolic)
    for _ in range(100):
        beat = cycle.step(0.01, 15.0, RhythmType.SINUS_BRADY)
        reading = monitor.step(0.01, beat, renderer.step(beat, 90.0, 40.0, dt=0.01, static_pressure=10.0))
        if not math.isnan(reading.systolic):
            assert reading.diastolic <= reading.mean <= reading.systolic
    assert reading.systolic > reading.diastolic + 20.0


def test_pulse_timeout_tracks_rate_changes_without_inheriting_pulseless_rr():
    # Sinus -> 10 bpm PEA must not inherit a 6 s hold window without a slow pulse.
    cycle = CardiacCycle(np.random.default_rng(1))
    renderer = ArterialWaveformRenderer(age=40)
    monitor = ArterialLineMonitor()
    monitor.seed(renderer.step(cycle.seed(75.0, RhythmType.SINUS), 90.0, 70.0, static_pressure=10.0))
    for _ in range(300):
        beat = cycle.step(0.01, 75.0, RhythmType.SINUS)
        monitor.advance(0.01, beat, renderer.step(beat, 90.0, 70.0, dt=0.01, static_pressure=10.0))

    blank_time = None
    for index in range(600):
        beat = cycle.step(0.01, 10.0, RhythmType.SINUS_BRADY)
        reading = monitor.step(0.01, beat, renderer.step(beat, 0.0, 0.0, dt=0.01, static_pressure=10.0))
        if math.isnan(reading.systolic) and blank_time is None:
            blank_time = (index + 1) * 0.01
    assert blank_time is not None and blank_time <= NO_PULSE_TIMEOUT_S + 0.8

    # 10 bpm pulsatile -> 180 bpm pulseless VT drops the hold window back to NO_PULSE_TIMEOUT_S.
    monitor.seed(renderer.step(cycle.seed(10.0, RhythmType.SINUS_BRADY), 90.0, 40.0, static_pressure=10.0))
    for _ in range(120):
        beat = cycle.step(0.01, 10.0, RhythmType.SINUS_BRADY)
        monitor.advance(0.01, beat, renderer.step(beat, 90.0, 40.0, dt=0.01, static_pressure=10.0))

    vt_blank_time = None
    for index in range(500):
        beat = cycle.step(0.01, 180.0, RhythmType.VTACH)
        reading = monitor.step(0.01, beat, renderer.step(beat, 0.0, 0.0, dt=0.01, static_pressure=10.0))
        if math.isnan(reading.systolic) and vt_blank_time is None:
            vt_blank_time = (index + 1) * 0.01
    assert vt_blank_time is not None and vt_blank_time <= NO_PULSE_TIMEOUT_S

    nonpulsatile = ArterialLineMonitor()
    vfib_sample = renderer.seed(cycle.seed(0.0, RhythmType.VFIB), 12.0, 0.0, static_pressure=12.0)
    seeded = nonpulsatile.seed(vfib_sample)
    assert math.isnan(seeded.systolic) and math.isnan(seeded.diastolic)
    assert seeded.mean == pytest.approx(12.0)


@pytest.mark.parametrize("static_pressure", [5.0, 12.0])
@pytest.mark.parametrize("method", ["seed", "step"])
def test_low_flow_pressure_falls_continuously_to_venous_pressure(static_pressure, method):
    cycle = CardiacCycle(np.random.default_rng(1))
    beat = cycle.seed(75.0, RhythmType.SINUS)
    pressures = []
    for mean in (90.0, static_pressure + 0.01, static_pressure, static_pressure - 0.01, 1.0, 0.0):
        renderer = ArterialWaveformRenderer(age=40)
        sample = getattr(renderer, method)(beat, mean, 0.8 * mean, static_pressure=static_pressure)
        assert sample.diastolic >= static_pressure
        if mean > 0.0:
            assert sample.systolic > sample.diastolic
        pressures.append(sample.mean)

    assert pressures[0] == pytest.approx(90.0)
    assert all(after <= before for before, after in pairwise(pressures))
    assert pressures[1] - pressures[3] <= 0.02 + 1e-9
    assert pressures[-1] == pytest.approx(static_pressure)


def test_surgical_steady_state_seeds_arterial_renderer_without_first_beat_overshoot(engine_factory):
    """The first live beats stay near the seeded surgical baseline."""
    engine = engine_factory(
        config=SimulationConfig(
            mode="steady_state", disturbance_profile="stim_sustained_surgery", dt=0.05, rng_seed=1
        ),
        start=True,
    )
    sbp_0 = engine.state.art_sbp
    peaks = []
    for _ in range(60):
        engine.step(0.05)
        peaks.append(engine.state.art_sbp)
    assert max(peaks) - sbp_0 < 8.0
