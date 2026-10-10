import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.monitors.arterial import ArterialWaveformRenderer
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
    assert float(np.ptp(values)) == pytest.approx(70.0 / renderer.arterial_compliance, abs=0.02)


def test_age_related_compliance_changes_pulse_pressure():
    younger, _ = _render_beat(age=25, map_value=90.0, hr=70.0, sv=70.0)
    older, _ = _render_beat(age=70, map_value=90.0, hr=70.0, sv=70.0)

    assert np.ptp(older) > np.ptp(younger)
    assert np.mean(older) == pytest.approx(np.mean(younger), abs=0.02)


def test_nonnegative_constraint_preserves_map_in_extreme_state():
    values, _ = _render_beat(age=70, map_value=5.0, hr=70.0, sv=120.0)

    assert float(np.min(values)) >= 0.0
    assert float(np.mean(values)) == pytest.approx(5.0, abs=0.02)


def _af_beats(hr: float, sv: float, map_value: float = 85.0, dt: float = 0.002):
    """Per-beat preceding R-R and pulse pressure over 60 s of AF, plus mean pressure."""
    cycle = CardiacCycle(np.random.default_rng(1))
    renderer = ArterialWaveformRenderer(age=40)
    renderer.step(cycle.seed(hr, RhythmType.AFIB), map_value, sv)
    beat_steps, pressures = [0], []
    for index in range(1, round(60.0 / dt)):
        sample = cycle.step(dt, hr, RhythmType.AFIB)
        if sample.beat_started:
            beat_steps.append(index)
        pressures.append(renderer.step(sample, map_value, sv).pressure)
    trace = np.asarray(pressures)
    upstroke = round(renderer.config.electromechanical_delay_s / dt)
    preceding_rr, pulse_pressure = [], []
    for previous, start, end in zip(beat_steps[1:], beat_steps[2:], beat_steps[3:], strict=False):
        preceding_rr.append((start - previous) * dt)
        pulse_pressure.append(trace[start:end].max() - trace[start + upstroke])
    return np.asarray(preceding_rr), np.asarray(pulse_pressure), float(np.mean(trace))


def test_af_pulse_pressure_follows_filling_time():
    """AF stroke volume rises with the preceding R-R (Hardman 1998), and its beat-to-beat
    variability grows with ventricular rate (Kerr 1998). Mean pressure stays at MAP."""
    _, slow_pp, slow_mean = _af_beats(hr=80.0, sv=60.0)
    fast_rr, fast_pp, fast_mean = _af_beats(hr=140.0, sv=40.0)

    assert np.corrcoef(fast_rr, fast_pp)[0, 1] > 0.8
    assert np.std(fast_pp) / np.mean(fast_pp) > 2.0 * np.std(slow_pp) / np.mean(slow_pp)
    assert slow_mean == pytest.approx(85.0, abs=1.5)
    assert fast_mean == pytest.approx(85.0, abs=1.5)
