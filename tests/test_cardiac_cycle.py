from dataclasses import replace

import numpy as np

from anasim.core.enums import RhythmType
from anasim.core.monitors import step_cardiac_monitors
from anasim.core.state import SimulationConfig
from anasim.monitors.arterial import ArterialWaveformRenderer
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.monitors.ecg import ECGMonitor
from anasim.monitors.spo2 import SpO2Monitor


def _beats(rhythm: RhythmType, hr: float, failure: float = 0.0) -> list[np.ndarray]:
    """ECG from 300 ms before to 500 ms after each R peak, in 1 ms samples."""
    dt = 0.001
    cycle = CardiacCycle(np.random.default_rng(6))
    ecg = ECGMonitor(rng=np.random.default_rng(7))
    sample = cycle.seed(hr, rhythm)
    voltages, peaks = [], []
    for index in range(max(4000, round(8 * 60.0 / hr / dt))):
        if index:
            sample = cycle.step(dt, hr, rhythm, failure)
        if sample.beat_started:
            peaks.append(index)
        voltages.append(ecg.step(dt, sample, conduction_failure=failure))
    trace = np.array(voltages)
    return [trace[peak - 300:peak + 500] for peak in peaks[1:-1] if peak + 500 <= len(trace)]


def _r_wave_width_ms(rhythm: RhythmType, hr: float, failure: float = 0.0) -> float:
    """Mean R-wave width at half height."""
    windows = [beat[200:400] for beat in _beats(rhythm, hr, failure)]
    return float(np.mean([np.count_nonzero(window > 0.5 * window.max()) for window in windows]))


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


def test_ischemic_conduction_failure_removes_p_waves_then_widens_qrs():
    """Junctional escape loses the P wave; idioventricular escape widens the QRS and inverts the T wave."""
    sinus, junctional, idioventricular = (
        np.mean(_beats(RhythmType.SINUS, 40.0, failure), axis=0) for failure in (0.0, 0.5, 0.8)
    )
    p_peak = 300 - 165
    assert sinus[p_peak] > 0.06 and abs(junctional[p_peak]) < 0.01
    narrow = _r_wave_width_ms(RhythmType.SINUS, 40.0)
    assert abs(_r_wave_width_ms(RhythmType.SINUS, 40.0, 0.5) - narrow) < 1.0
    assert _r_wave_width_ms(RhythmType.SINUS, 40.0, 0.8) > 1.8 * narrow
    t_region = slice(300 + 150, 800)
    assert sinus[t_region].max() > 0.1
    assert idioventricular[t_region].min() < -0.1 and idioventricular[t_region].max() < 0.05


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


def test_ischemic_st_depression_is_horizontal_from_the_j_point_into_the_t_wave():
    """The QRS is unchanged, the segment is flat, and it returns to baseline through the T wave."""
    for rhythm, hr in ((RhythmType.SINUS, 60), (RhythmType.SVT, 160), (RhythmType.VTACH, 180)):
        cycle = CardiacCycle(np.random.default_rng(1))
        normal = ECGMonitor(np.random.default_rng(2))
        ischemic = ECGMonitor(np.random.default_rng(2))
        sample = cycle.seed(hr, rhythm)
        depression = {}
        for index in range(round(60000 / hr)):
            if index:
                sample = cycle.step(0.001, hr, rhythm)
            depression[index] = normal.step(0.001, sample) - ischemic.step(0.001, sample, 0.2)
        if rhythm == RhythmType.VTACH:
            assert not any(depression.values())
            continue
        assert not any(depression[ms] for ms in range(40))
        assert all(0.19 < depression[ms] < 0.201 for ms in range(60, 101))
        if rhythm == RhythmType.SINUS:
            # Still depressed at the T peak (248 ms at 60/min); baseline after the T wave.
            assert depression[248] > 0.08
            assert not any(depression[ms] for ms in range(380, 1000))


def test_batched_monitor_updates_preserve_waveforms_readings_and_noise(engine_factory):
    """Long physiology steps must publish the same monitor samples as 100 Hz updates."""
    reference, batched = [
        engine_factory(config=SimulationConfig(mode="awake", rng_seed=7)) for _ in range(2)
    ]
    fields = ("ecg_voltage", "pleth_voltage", "art_pressure", "art_sbp", "art_dbp", "art_map",
              "sbp", "dbp", "spo2", "spo2_signal_valid")
    for rhythm, hr, saturation, output in (
        (RhythmType.SINUS, 75.0, 98.0, 5.0),
        (RhythmType.AFIB, 110.0, 85.0, 3.0),
        (RhythmType.SVT, 160.0, 93.0, 2.0),
        (RhythmType.VTACH, 180.0, 90.0, 1.0),
        (RhythmType.VFIB, 0.0, 60.0, 0.0),
        (RhythmType.ASYSTOLE, 0.0, 50.0, 0.0),
        (RhythmType.SINUS_BRADY, 45.0, 99.0, 4.0),
    ):
        hemo = replace(reference.hemo.state, rhythm_type=rhythm, hr=hr, co=output)
        for engine in (reference, batched):
            engine.hemo.coronary.ischemia = 0.5
        for _ in range(5):
            for _ in range(100):
                step_cardiac_monitors(reference, 0.01, hemo, saturation)
            step_cardiac_monitors(batched, 1.0, hemo, saturation)
            assert tuple(getattr(batched.state, name) for name in fields) == tuple(
                getattr(reference.state, name) for name in fields
            )
        assert batched.ecg.rng.bit_generator.state == reference.ecg.rng.bit_generator.state
        assert batched.cardiac_cycle.rng.bit_generator.state == reference.cardiac_cycle.rng.bit_generator.state
