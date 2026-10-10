import numpy as np

from anasim.core.enums import RhythmType
from anasim.monitors.arterial import ArterialWaveformRenderer
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.monitors.ecg import ECGMonitor
from anasim.monitors.spo2 import SpO2Monitor


def _r_wave_width_ms(rhythm: RhythmType, hr: float) -> float:
    """Mean R-wave width at half height."""
    dt = 0.001
    cycle = CardiacCycle(np.random.default_rng(6))
    ecg = ECGMonitor(rng=np.random.default_rng(7))
    sample = cycle.seed(hr, rhythm)
    voltages, beats = [], []
    for index in range(4000):
        if index:
            sample = cycle.step(dt, hr, rhythm)
        if sample.beat_started:
            beats.append(index)
        voltages.append(ecg.step(dt, sample))
    trace = np.array(voltages)
    widths = []
    for beat in beats[1:-1]:
        window = trace[beat - 100:beat + 100]
        widths.append(np.count_nonzero(window > 0.5 * window.max()) * dt * 1000.0)
    return float(np.mean(widths))


def test_qrs_width_is_rate_independent_and_wide_in_vt():
    """QRS duration does not scale with R-R; monomorphic VT is wide (QRS > 120 ms)."""
    narrow = [
        _r_wave_width_ms(rhythm, hr)
        for rhythm, hr in (
            (RhythmType.SINUS_BRADY, 45.0),
            (RhythmType.SINUS, 120.0),
            (RhythmType.AFIB, 110.0),
            (RhythmType.SVT, 160.0),
        )
    ]
    assert max(narrow) - min(narrow) < 3.0
    assert _r_wave_width_ms(RhythmType.VTACH, 180.0) > 2.0 * max(narrow)


def test_pulse_oximeter_requires_perfusion_and_an_organized_pulse():
    cycle = CardiacCycle(np.random.default_rng(2))
    sample = cycle.seed(60.0, RhythmType.SINUS)
    spo2 = SpO2Monitor()
    spo2.step(0.1, sample, 98.0, 1.0)

    # Signal loss cannot track unmeasurable saturation or generate a pulse.
    for rhythm, perfusion in ((RhythmType.SINUS, 0.0), (RhythmType.VFIB, 1.0)):
        sample = cycle.seed(60.0, rhythm)
        pleth, saturation = spo2.step(10.0, sample, 50.0, perfusion)
        assert not spo2.signal_valid
        assert pleth == 0.0
        assert saturation == 98.0

    sample = cycle.seed(60.0, RhythmType.SINUS)
    _, saturation = spo2.step(0.1, sample, 50.0, 1.0)
    assert spo2.signal_valid
    assert 50.0 < saturation < 98.0


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
