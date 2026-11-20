import math

import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.arterial import ArterialLineMonitor, ArterialWaveformRenderer
from anasim.monitors.cardiac_cycle import CardiacCycle


def _render_beat(*, age: float, map_value: float, hr: float, sv: float, dt: float = 0.0005):
    cycle = CardiacCycle(np.random.default_rng(1))
    renderer = ArterialWaveformRenderer(age=age)
    sample = cycle.seed(hr, RhythmType.SINUS)
    renderer.step(sample, map_value, sv)
    values = []
    steps = round((60.0 / hr) / dt)
    for _ in range(steps):
        sample = cycle.step(dt, hr, RhythmType.SINUS)
        values.append(renderer.step(sample, map_value, sv).pressure)
    return np.asarray(values), renderer


def test_waveform_mean_and_pulse_pressure_match_su_targets():
    values, renderer = _render_beat(age=40, map_value=90.0, hr=75.0, sv=70.0)

    assert float(np.mean(values)) == pytest.approx(90.0, abs=0.02)
    assert float(np.ptp(values)) == pytest.approx(70.0 / renderer.compliance_at(90.0), abs=0.02)


def test_pulse_pressure_rises_with_age_after_45():
    """Franklin 1997, normotensive groups: pulse pressure is flat from 30 to 45
    and 27-42% higher at 65-74 than at 40-44."""
    pulse = {age: np.ptp(_render_beat(age=age, map_value=90.0, hr=70.0, sv=70.0)[0]) for age in (25, 40, 70)}

    assert pulse[25] == pytest.approx(pulse[40])
    assert 1.27 < pulse[70] / pulse[40] < 1.42


def test_nonnegative_constraint_preserves_map_in_extreme_state():
    values, _ = _render_beat(age=70, map_value=5.0, hr=70.0, sv=120.0)

    assert float(np.min(values)) >= 0.0
    assert float(np.mean(values)) == pytest.approx(5.0, abs=0.02)

    # Across consecutive beats at 100 Hz monitor resolution, zero diastolic
    # pressure must not drift upward.
    cycle = CardiacCycle(np.random.default_rng(1))
    renderer = ArterialWaveformRenderer(age=70)
    renderer.step(cycle.seed(70.0, RhythmType.SINUS), 5.0, 120.0, dt=0.01)
    multi = [
        renderer.step(cycle.step(0.01, 70.0, RhythmType.SINUS), 5.0, 120.0, dt=0.01).pressure
        for _ in range(round(5 * (60.0 / 70.0) / 0.01))
    ]
    assert float(np.min(multi)) >= 0.0
    assert float(np.mean(multi)) == pytest.approx(5.0, abs=0.05)


def _af_beats(hr: float, sv: float, map_value: float = 85.0, dt: float = 0.005):
    """Per-beat preceding R-R and pulse pressure over 60 s of AF, plus mean pressure."""
    cycle = CardiacCycle(np.random.default_rng(1))
    renderer = ArterialWaveformRenderer(age=40)
    monitor = ArterialLineMonitor()
    seeded = cycle.seed(hr, RhythmType.AFIB)
    monitor.seed(renderer.step(seeded, map_value, sv, static_pressure=10.0))
    beat_steps, pressures = [0], []
    for index in range(1, round(60.0 / dt)):
        sample = cycle.step(dt, hr, RhythmType.AFIB)
        if sample.beat_started:
            beat_steps.append(index)
        art_sample = renderer.step(sample, map_value, sv, dt=dt, static_pressure=10.0)
        pressures.append(art_sample.pressure)
        reading = monitor.step(dt, sample, art_sample)
        assert reading.systolic > reading.diastolic + 10.0
        assert reading.diastolic <= reading.mean <= reading.systolic
    trace = np.asarray(pressures)
    upstroke = round(renderer.config.electromechanical_delay_s / dt)
    preceding_rr, pulse_pressure = [], []
    for previous, start, end in zip(beat_steps[1:], beat_steps[2:], beat_steps[3:], strict=False):
        preceding_rr.append((start - previous) * dt)
        pulse_pressure.append(trace[start:end].max() - trace[start + upstroke])
    return np.asarray(preceding_rr), np.asarray(pulse_pressure), float(np.mean(trace)), monitor.reading.mean


def test_af_pulse_pressure_follows_filling_time():
    """AF stroke volume rises with the preceding R-R (Hardman 1998), and its beat-to-beat
    variability grows with ventricular rate (Kerr 1998). Mean pressure stays at MAP."""
    _, slow_pp, slow_mean, _ = _af_beats(hr=80.0, sv=60.0)
    fast_rr, fast_pp, fast_mean, fast_mon_map = _af_beats(hr=140.0, sv=40.0)

    assert np.corrcoef(fast_rr, fast_pp)[0, 1] > 0.8
    assert np.std(fast_pp) / np.mean(fast_pp) > 2.0 * np.std(slow_pp) / np.mean(slow_pp)
    assert slow_mean == pytest.approx(85.0, abs=1.5)
    assert fast_mean == pytest.approx(85.0, abs=1.5)
    assert fast_mon_map == pytest.approx(85.0, abs=1.5)

    # Near-venous-pressure low flow clamps Windkessel runoff tau cleanly.
    renderer = ArterialWaveformRenderer(age=40)
    cycle = CardiacCycle(np.random.default_rng(2))
    beat = cycle.seed(60.0, RhythmType.SINUS)
    renderer.step(beat, 15.2, 5.0, dt=0.01, static_pressure=15.0)
    assert 0.25 <= renderer._runoff_tau_s <= 6.0


def test_sudden_hr_acceleration_reference_pp_and_pea_compression_runoff(engine_factory):
    """Sudden bradycardia-to-tachycardia transitions keep Windkessel pressure bounded,
    reference_pulse_pressure matches _update_reference, and stopping PEA compressions
    runs off smoothly from the compression diastolic baseline."""
    cycle = CardiacCycle(np.random.default_rng(1))
    renderer = ArterialWaveformRenderer(age=40)
    sample = cycle.seed(15.0, RhythmType.SINUS_BRADY)
    renderer.step(sample, 40.0, 60.0, dt=0.01, static_pressure=8.0)
    for _ in range(5):
        sample = cycle.step(0.01, 15.0, RhythmType.SINUS_BRADY)
        renderer.step(sample, 40.0, 60.0, dt=0.01, static_pressure=8.0)
    peak = 0.0
    for _ in range(300):
        sample = cycle.step(0.01, 150.0, RhythmType.SINUS)
        art = renderer.step(sample, 90.0, 70.0, dt=0.01, static_pressure=8.0)
        peak = max(peak, art.pressure)
    assert math.isfinite(peak) and peak < 250.0

    ref_sample = cycle.seed(75.0, RhythmType.SINUS)
    renderer.seed(ref_sample, 50.0, 45.0, static_pressure=12.0, pulse_pressure_factor=0.6)
    assert renderer.reference_pulse_pressure(
        50.0, 45.0, 75.0, pulse_pressure_factor=0.6, static_pressure=12.0
    ) == pytest.approx(renderer._reference_pp, abs=1e-6)

    pea = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    pea.hemo.add_volume(-2600.0)
    for _ in range(30):
        pea.step(1.0)
    pea.set_chest_compressions(True)
    for _ in range(10):
        pea.step(0.5)
    dbp_before = pea.state.art_dbp
    pea.set_chest_compressions(False)
    pea.step(0.01)
    assert pea.state.art_pressure >= dbp_before - 5.0
