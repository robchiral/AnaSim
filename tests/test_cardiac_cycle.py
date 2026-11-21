import copy
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.core.monitors import step_cardiac_monitors
from anasim.core.state import SimulationConfig
from anasim.monitors.arterial import ArterialWaveformRenderer
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.monitors.ecg import ECGMonitor
from anasim.monitors.spo2 import SpO2Monitor
from anasim.web import WebSession


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

    # Without a usable signal, the pleth is flat and saturation holds its last reading.
    for rhythm, perfusion in ((RhythmType.SINUS, 0.0), (RhythmType.VFIB, 1.0)):
        sample = cycle.seed(60.0, rhythm)
        pleth, saturation = spo2.step(10.0, sample, 50.0, perfusion)
        assert not spo2.signal_valid
        assert pleth == 0.0
        assert saturation == 98.0

    # Multi-beat confirmation prevents single-sample reacquisition, then tracks finger transit.
    sample = cycle.seed(60.0, RhythmType.SINUS)
    _, saturation = spo2.step(0.1, sample, 50.0, 1.0)
    assert not spo2.signal_valid
    for _ in range(30):
        sample = cycle.step(0.1, 60.0, RhythmType.SINUS)
        _, saturation = spo2.step(0.1, sample, 50.0, 1.0)
    assert spo2.signal_valid
    for _ in range(150):
        sample = cycle.step(0.1, 60.0, RhythmType.SINUS)
        _, saturation = spo2.step(0.1, sample, 50.0, 1.0)
    assert 50.0 < saturation < 95.0


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
            waves = [step_cardiac_monitors(reference, 0.01, hemo, saturation)[1][0] for _ in range(100)]
            assert step_cardiac_monitors(batched, 1.0, hemo, saturation)[1] == waves
            assert tuple(getattr(batched.state, name) for name in fields) == tuple(
                getattr(reference.state, name) for name in fields
            )
        assert batched.ecg.rng.bit_generator.state == reference.ecg.rng.bit_generator.state
        assert batched.cardiac_cycle.rng.bit_generator.state == reference.cardiac_cycle.rng.bit_generator.state


def test_pulse_oximeter_blanks_on_dropout_and_reacquires_without_clearing_alarm():
    """SpO2 dropout blanks UI vitals without clearing the low-SpO2 alarm, requires
    multi-beat confirmation to reacquire, and tracks current SaO2 after finger transit."""
    for dt in (0.5, 0.05):
        session = WebSession({"mode": "awake"})
        engine = session.engine
        engine.config.dt = dt
        engine.start()
        engine.set_rhythm("VFIB")
        for _ in range(round(10.0 / dt)):
            engine.step(dt)
        assert not engine.state.spo2_signal_valid
        assert session.snapshot()["vitals"]["spo2"] is None
        assert engine.state.alarms["SpO2"]["low"]

        engine.resp.gas_transport.oxygen[1] *= 0.82
        engine.set_rhythm("SINUS")
        # Single brief sample does not latch a valid pulse oximeter reading.
        engine.step(dt)
        assert not engine.state.spo2_signal_valid
        assert engine.state.alarms["SpO2"]["low"]
        # Reacquires after multi-beat confirmation (~2-3 s) without clearing low-SpO2 alarm.
        for _ in range(round(3.0 / dt)):
            engine.step(dt)
            assert engine.state.alarms["SpO2"]["low"]
        assert engine.state.spo2_signal_valid
        assert engine.state.display_spo2 < 90.0
        assert engine.state.display_spo2 == pytest.approx(engine.state.sao2, abs=4.0)


def test_chest_compressions_add_110_per_minute_ecg_artifact_during_sinus_rhythm(engine_factory):
    """Chest compressions during perfusing sinus rhythm generate 110 /min ECG compression artifact."""
    cpr_only = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=1), start=True)
    no_cpr = copy.deepcopy(cpr_only)
    cpr_only.set_chest_compressions(True)
    for _ in range(6):
        cpr_only.step(0.5)
        no_cpr.step(0.5)
    diffs = [
        a.ecg_voltage - b.ecg_voltage
        for a, b in zip(list(cpr_only.output_buffer)[-300:], list(no_cpr.output_buffer)[-300:], strict=True)
    ]
    peaks = [
        i * 0.01
        for i in range(1, len(diffs) - 1)
        if diffs[i] > 0.15 and diffs[i] > diffs[i - 1] and diffs[i] >= diffs[i + 1]
    ]
    periods = [b - a for a, b in pairwise(peaks)]
    assert len(periods) >= 4
    for period in periods:
        assert period == pytest.approx(60.0 / 110.0, abs=0.02)


def test_display_hr_window_decay_and_rhythm_switch_flush(awake_engine, advance_time):
    """CardiacCycle decays display_hr during pauses, engine.set_rhythm flushes stale intervals,
    and SpO2Monitor latches peripheral perfusion at beat upstroke.
    """
    cycle = CardiacCycle(np.random.default_rng(1))
    for _ in range(300):
        sample = cycle.step(0.01, 80.0, RhythmType.SINUS)
    assert sample.display_hr == pytest.approx(80.0, abs=1.0)
    for _ in range(900):
        sample = cycle.step(0.01, 12.0, RhythmType.SINUS_BRADY)
    assert sample.display_hr < 20.0

    advance_time(awake_engine, 8.0, dt=0.5)
    awake_engine.set_rhythm("SVT")
    awake_engine.step(0.5)
    assert awake_engine.state.display_hr > 130.0

    spo2 = SpO2Monitor()
    b0 = cycle.seed(60.0, RhythmType.SINUS)
    spo2.step(0.01, b0, 98.0, 0.20)
    b1 = cycle.step(0.10, 60.0, RhythmType.SINUS)
    spo2.step(0.10, b1, 98.0, 1.00)
    assert spo2._beat_perf == pytest.approx(0.20)


def test_agonal_hr_recovery_rescales_pending_interval_and_zero_hr_ecg_flatlines():
    """Recovering from agonal bradycardia rescales the pending R-R interval without a long stall,
    and an unorganized non-VF cycle (hr == 0) renders baseline ECG noise without stale QRS waves."""
    cycle = CardiacCycle(np.random.default_rng(1))
    cycle.seed(1.0, RhythmType.SINUS_BRADY)
    for _ in range(1000):
        cycle.step(0.01, 1.0, RhythmType.SINUS_BRADY)
    for _ in range(300):
        sample = cycle.step(0.01, 80.0, RhythmType.SINUS)
    assert 70.0 <= sample.display_hr <= 90.0

    ecg = ECGMonitor(np.random.default_rng(1))
    s_on = cycle.seed(70.0, RhythmType.SINUS)
    ecg.step(0.01, s_on)
    s_zero = cycle.step(0.01, 0.0, RhythmType.SINUS)
    assert not s_zero.organized
    assert max(abs(ecg.step(0.01, s_zero)) for _ in range(50)) <= 0.02


def test_spo2_cpr_invalidation_vasoconstriction_dropout_first_beat_and_transit(engine_factory, advance_time):
    """CPR invalidates numeric SpO2; 28 C + phenylephrine drops pleth; first beat waits for peripheral_delay_s;
    vasoconstriction lengthens lung-to-finger transit (MacLeod 2005)."""
    cpr = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    cpr.set_rhythm("VFIB")
    cpr.set_chest_compressions(True)
    advance_time(cpr, 15.0, dt=0.5)
    assert not cpr.state.spo2_signal_valid
    assert cpr.state.spo2 == 0.0

    cold_vaso = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1), start=True, baseline_temp=28.0,
    )
    cold_vaso.give_drug_bolus("phenyl", 1000.0)
    advance_time(cold_vaso, 45.0, dt=0.5)
    assert not cold_vaso.state.spo2_signal_valid

    cycle = CardiacCycle(np.random.default_rng(1))
    spo2 = SpO2Monitor()
    s_vf = cycle.seed(0.0, RhythmType.VFIB)
    spo2.step(0.5, s_vf, 98.0, 0.0)
    s_first = cycle.seed(60.0, RhythmType.SINUS)
    pleth_0, _ = spo2.step(0.01, s_first, 98.0, 1.0)
    assert pleth_0 == 0.0 and not spo2.signal_valid

    fast_mon, slow_mon = SpO2Monitor(), SpO2Monitor()
    c_fast, c_slow = CardiacCycle(np.random.default_rng(2)), CardiacCycle(np.random.default_rng(2))
    fast_mon.step(0.1, c_fast.seed(70.0, RhythmType.SINUS), 98.0, 1.0, delay_scale=1.0)
    slow_mon.step(0.1, c_slow.seed(70.0, RhythmType.SINUS), 98.0, 0.4, delay_scale=4.0)
    for _ in range(150):
        sf = c_fast.step(0.1, 70.0, RhythmType.SINUS)
        ss = c_slow.step(0.1, 70.0, RhythmType.SINUS)
        _, v_fast = fast_mon.step(0.1, sf, 75.0, 1.0, delay_scale=1.0)
        _, v_slow = slow_mon.step(0.1, ss, 75.0, 0.4, delay_scale=4.0)
    assert v_fast < v_slow - 6.0


def test_hyperkalemia_ecg_keeps_upright_t_prolongs_pr_and_improves_with_calcium(engine_factory, advance_time):
    """Severe hyperkalemia widens QRS and prolongs PR without inverting the T wave;
    IV calcium lowers effective potassium rather than spiking T waves."""
    def sample_beat(potassium: float, hyperk_block: float) -> np.ndarray:
        cycle = CardiacCycle(np.random.default_rng(3))
        ecg = ECGMonitor(np.random.default_rng(4))
        s = cycle.seed(60.0, RhythmType.SINUS)
        trace = []
        for idx in range(1000):
            if idx:
                s = cycle.step(0.001, 60.0, RhythmType.SINUS)
            trace.append(ecg.step(0.001, s, potassium=potassium, hyperkalemia_block=hyperk_block))
        return np.array(trace)

    normal = sample_beat(4.0, 0.0)
    mild_k = sample_beat(6.8, 0.20)
    severe_k = sample_beat(8.5, 0.77)
    t_window = slice(180, 500)
    assert severe_k[t_window].max() > 0.30
    assert severe_k[t_window].min() > -0.15
    qrs_width_norm = int(np.sum(normal[:120] > 0.5 * normal[:120].max()))
    qrs_width_severe = int(np.sum(severe_k[:120] > 0.5 * severe_k[:120].max()))
    assert qrs_width_severe > 1.5 * qrs_width_norm
    p_norm_idx = int(np.argmax(normal[700:920]))
    p_mild_idx = int(np.argmax(mild_k[700:920]))
    assert p_mild_idx < p_norm_idx - 25

    eng = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    eng.hemo.fluids.amounts["potassium"] = 7.8 * eng.hemo.fluids.ecf_l
    advance_time(eng, 10.0, dt=1.0)
    eff_before = eng.hemo.effective_potassium
    eng.give_calcium_chloride(1000.0)
    advance_time(eng, 90.0, dt=1.0)
    assert eng.hemo.effective_potassium < eff_before - 0.5
