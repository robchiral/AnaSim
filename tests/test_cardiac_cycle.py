import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.monitors.arterial import ArterialWaveformRenderer
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.monitors.ecg import ECGMonitor
from anasim.monitors.spo2 import SpO2Monitor


def test_regular_cycle_reports_exact_beat_boundaries():
    cycle = CardiacCycle(np.random.default_rng(1))
    seeded = cycle.seed(60.0, RhythmType.SINUS)

    assert seeded.beat_started
    assert seeded.phase == pytest.approx(0.0)

    partial = cycle.step(0.4, 60.0, RhythmType.SINUS)
    assert not partial.beat_started
    assert partial.phase == pytest.approx(0.4)

    completed = cycle.step(0.6, 60.0, RhythmType.SINUS)
    assert completed.beat_started
    assert completed.phase == pytest.approx(0.0)
    assert completed.measured_hr == pytest.approx(60.0)


def _af_intervals(seed: int) -> list[float]:
    cycle = CardiacCycle(np.random.default_rng(seed))
    cycle.seed(120.0, RhythmType.AFIB)
    intervals = []
    while len(intervals) < 6:
        sample = cycle.step(0.01, 120.0, RhythmType.AFIB)
        if sample.beat_started:
            intervals.append(60.0 / sample.measured_hr)
    return intervals


def test_af_variability_is_beatwise_and_reproducible():
    first = _af_intervals(11)
    second = _af_intervals(11)

    assert first == pytest.approx(second)
    assert np.std(first) > 0.02


@pytest.mark.parametrize("rhythm", [RhythmType.VFIB, RhythmType.ASYSTOLE])
def test_arrest_rhythms_have_no_organized_cycle(rhythm):
    cycle = CardiacCycle(np.random.default_rng(3))
    sample = cycle.seed(0.0, rhythm)

    assert not sample.organized
    assert not sample.beat_started
    assert sample.measured_hr == pytest.approx(0.0)


def test_shared_cycle_orders_qrs_art_and_pleth():
    cycle = CardiacCycle(np.random.default_rng(4))
    renderer = ArterialWaveformRenderer(age=40)
    ecg = ECGMonitor(rng=np.random.default_rng(5))
    spo2 = SpO2Monitor()
    sample = cycle.seed(60.0, RhythmType.SINUS)

    ecg_values = []
    art_values = []
    pleth_values = []
    for index in range(100):
        if index > 0:
            sample = cycle.step(0.01, 60.0, RhythmType.SINUS)
        ecg_values.append(ecg.step(0.01, sample))
        art_values.append(renderer.step(sample, 90.0, 70.0).pressure)
        pleth_values.append(spo2.step(0.01, sample, 98.0, 1.0)[0])

    assert int(np.argmax(ecg_values)) < int(np.argmax(art_values))
    assert int(np.argmax(art_values)) < int(np.argmax(pleth_values))
