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
