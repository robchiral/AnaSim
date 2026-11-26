
import copy
import math
from itertools import pairwise

import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.capno import Capnograph, EtCO2Readout
from anasim.monitors.nibp import NIBPMonitor
from anasim.scenarios.base import require_map_at_least


class _FixedRng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def _cuff_cycle(rng, true_map, true_sys, true_dia, rhythm=RhythmType.SINUS):
    monitor = NIBPMonitor(rng=rng)
    monitor.trigger()
    t = 0.0
    while monitor.is_cycling:
        t += 0.5
        monitor.step(
            0.5, t, true_map=true_map, true_sys=true_sys, true_dia=true_dia, rhythm_type=rhythm
        )
    return monitor.latest_reading


class TestNIBP:
    @pytest.mark.parametrize("dt", [0.01, 0.1, 0.5, 1.0, 15.0])
    def test_cuff_cycle_timing_does_not_depend_on_step_size(self, dt):
        monitor = NIBPMonitor(rng=_FixedRng(0.99))
        monitor.trigger()
        time = 0.0
        while monitor.is_cycling:
            time += dt
            pressure = monitor.step(dt, time, 90.0, 120.0, 75.0, RhythmType.SINUS)
            assert 0.0 <= pressure <= 160.0
        assert monitor.latest_reading.timestamp == pytest.approx(11.4)
        assert monitor.latest_reading.map == 90.0

    def test_cuff_reads_on_its_interval_and_tracks_map(self, engine_factory):
        engine = engine_factory(
            config=SimulationConfig(mode="awake", dt=0.5, rng_seed=123),
            start=True,
        )
        readings = []
        last_timestamp = engine.state.nibp_timestamp
        for _ in range(620):
            engine.step(1.0)
            if engine.state.nibp_timestamp != last_timestamp:
                last_timestamp = engine.state.nibp_timestamp
                readings.append((engine.state.time, engine.state.nibp_map, engine.state.map))

        assert len(readings) >= 3
        assert np.diff([time for time, _, _ in readings]) == pytest.approx(300.0, abs=2.0)
        for _, nibp_map, true_map in readings:
            assert nibp_map == pytest.approx(true_map, abs=5.0)

    def test_cuff_fails_or_overreads_in_shock_and_gives_nothing_in_arrest(self):
        failed = _cuff_cycle(_FixedRng(0.0), true_map=35.0, true_sys=55.0, true_dia=25.0)
        assert failed.timestamp is None

        reading = _cuff_cycle(_FixedRng(0.99), true_map=40.0, true_sys=60.0, true_dia=30.0)
        assert reading.timestamp is not None
        assert reading.map > 40.0
        assert reading.systolic > 60.0

        arrest = _cuff_cycle(np.random.default_rng(0), 0.0, 0.0, 0.0, rhythm=RhythmType.ASYSTOLE)
        assert arrest.timestamp is None

    def test_shock_boundary_is_continuous_and_failed_cuff_prompts_remeasurement(self, awake_engine):
        """Crossing MAP 60 mmHg does not step-jump NIBP bias or failure rate,
        and a failed cuff measurement prompts a re-check rather than awaiting first BP.
        """
        above = [
            _cuff_cycle(_FixedRng(q), true_map=60.0, true_sys=85.0, true_dia=48.0)
            for q in np.linspace(0.05, 0.95, 19)
        ]
        below = [
            _cuff_cycle(_FixedRng(q), true_map=59.9, true_sys=84.9, true_dia=47.9)
            for q in np.linspace(0.05, 0.95, 19)
        ]
        assert all(r.timestamp is not None for r in above)
        assert all(r.timestamp is not None for r in below)
        assert abs(np.mean([r.map for r in above]) - np.mean([r.map for r in below])) < 0.5

        awake_engine.config.arterial_line_enabled = False
        awake_engine.state.nibp_measurement_failed = True
        awake_engine.state.nibp_sys = awake_engine.state.nibp_dia = awake_engine.state.nibp_map = float("nan")
        met, msg = require_map_at_least(65)(awake_engine)
        assert not met and msg == "Measure blood pressure again"

    def test_deflation_sweep_invalidation_and_afib_scatter_and_ordering(self):
        """VF during cuff deflation fails even if sinus returns on the final step;
        AFib exhibits modest oscillometric scatter, occasional retry failure, and DBP <= MAP <= SBP."""
        mon = NIBPMonitor(rng=np.random.default_rng(1))
        mon.trigger()
        t = 0.0
        while mon.is_cycling:
            t += 0.5
            rhythm = RhythmType.VFIB if t < 10.8 else RhythmType.SINUS
            p_map, p_sys, p_dia = (8.0, 8.0, 8.0) if rhythm == RhythmType.VFIB else (80.0, 105.0, 65.0)
            mon.step(0.5, t, p_map, p_sys, p_dia, rhythm)
        assert mon.measurement_failed and mon.latest_reading.timestamp is None

        rng = np.random.default_rng(7)
        afib_maps = []
        afib_failures = 0
        for _ in range(40):
            m = NIBPMonitor(rng=rng)
            m.trigger()
            t = 0.0
            while m.is_cycling:
                t += 0.5
                m.step(0.5, t, 85.0, 115.0, 70.0, RhythmType.AFIB)
            if m.measurement_failed:
                afib_failures += 1
            else:
                r = m.latest_reading
                assert r.diastolic <= r.map <= r.systolic
                afib_maps.append(r.map)
        assert afib_failures >= 1
        assert np.std(afib_maps) > 1.5

        # Narrow pulse pressure under AFib still preserves DBP <= MAP <= SBP.
        narrow = NIBPMonitor(rng=np.random.default_rng(42))
        t = 0.0
        valid = 0
        for _ in range(40):
            narrow.trigger()
            for _ in range(20):
                t += 1.0
                narrow.step(1.0, t, 64.0, 72.0, 60.0, RhythmType.AFIB)
            if not narrow.measurement_failed:
                valid += 1
                r = narrow.latest_reading
                assert r.diastolic <= r.map <= r.systolic
            t += 5.0
        assert valid >= 20


class TestCapnography:
    def test_obstruction_keeps_impedance_effort_and_expires_end_tidal_reading(self, awake_engine):
        engine = awake_engine
        engine.set_airway_mode("ETT")
        for _ in range(60):
            engine.step(0.5)
        last_endpoint = engine.state.display_etco2
        assert engine.state.etco2_signal_valid and last_endpoint > 20.0

        engine.set_airway_obstruction(1.0)
        engine.step(0.1)
        assert engine.state.capno_co2 == 0.0
        assert engine.state.etco2_signal_valid
        assert engine.state.display_etco2 == last_endpoint
        for _ in range(160):
            engine.step(0.1)
        assert not engine.state.etco2_signal_valid and engine.state.display_etco2 == 0.0

        engine.set_airway_mode("None")
        engine.step(0.1)
        assert engine.state.rr > 8.0
        assert engine.state.vt == engine.state.mv == engine.state.va == 0.0
        effort_before_block = engine.resp_mech.effort.amplitude
        engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
        for _ in range(240):
            engine.step(0.5)
        # Residual diaphragm contractions are physically capped by remaining muscle capacity.
        assert engine.resp.state.muscle_factor < 0.2
        assert engine.resp_mech.effort.amplitude < effort_before_block
        assert engine.resp_mech.effort.amplitude <= engine.resp_mech.effort.muscle.available_pressure + 0.1
        assert engine.state.vt == engine.state.mv == engine.state.va == 0.0
        assert not engine.state.etco2_signal_valid

    def test_patient_effort_during_exhalation_notches_the_plateau(self):
        """A curare cleft draws fresh gas past the port without latching a false low EtCO2."""
        def exhale(capno, notch, readout=None):
            trace = []
            for k in range(150):
                volume = -0.004
                if notch and 50 <= k < 55:
                    volume = 0.006  # 30 mL drawn in over 0.1 s
                elif notch and 55 <= k < 62:
                    volume = -0.0043  # Breathed out again, so the drawn-in gas returns first
                co2 = capno.step(0.02, volume, 38.0)
                if readout is not None:
                    readout.update(
                        0.02, capno.exhaling, co2, inhaled_l=capno._inhaled,
                        dead_space_l=capno.dead_space, breath_index=capno.breath_index,
                    )
                trace.append(co2)
            return trace

        smooth = exhale(Capnograph(0.15), notch=False)
        notched_capno = Capnograph(0.15)
        readout = EtCO2Readout()
        notched_capno.step(0.5, 0.5, 38.0)
        readout.update(
            0.5, notched_capno.exhaling, notched_capno.co2, inhaled_l=notched_capno._inhaled,
            dead_space_l=notched_capno.dead_space, breath_index=notched_capno.breath_index,
        )
        notched = exhale(notched_capno, notch=True, readout=readout)
        notched_capno.step(0.5, 0.5, 38.0)
        readout.update(
            0.5, notched_capno.exhaling, notched_capno.co2, inhaled_l=notched_capno._inhaled,
            dead_space_l=notched_capno.dead_space, breath_index=notched_capno.breath_index,
        )
        # Phase I holds inspired gas until the dead space empties, then the plateau rises to end-tidal.
        assert smooth[5] < 1.0 and smooth[-1] == pytest.approx(38.0, abs=1.5)
        dip = max(a - b for a, b in zip(smooth, notched, strict=True))
        assert dip > 10.0
        assert notched[-1] == pytest.approx(smooth[-1], abs=1.0)
        assert readout.value == pytest.approx(38.0, abs=1.5)

    @staticmethod
    def _capnogram_rate(engine, seconds=20.0, dt=0.1):
        """Rate from intervals between CO2 rises, without partial-window count bias."""
        rises = []
        above = engine.state.capno_co2 > 15.0
        for _ in range(int(seconds / dt)):
            engine.step(dt)
            now = engine.state.capno_co2 > 15.0
            if now and not above:
                rises.append(engine.state.time)
            above = now
        return 60.0 / np.mean(np.diff(rises)) if len(rises) >= 2 else 0.0

    def test_changing_tidal_volumes_preserves_the_current_co2_endpoint(self):
        capno = Capnograph(0.15)
        peaks = []
        for volume in (0.5, 1.0, 0.5, 1.0):
            for _ in range(150):
                capno.step(0.01, volume / 150, 38.0)
            expired = []
            for i in range(350):
                change = -volume * (np.exp(-i * 0.01 / 0.6) - np.exp(-(i + 1) * 0.01 / 0.6))
                expired.append(capno.step(0.01, change, 38.0))
            peaks.append(max(expired))
        assert peaks == pytest.approx([38.0] * 4, abs=0.5)

        capno = Capnograph(0.15)
        capno.step(0.1, 0.5, 40.0)
        for _ in range(30):
            capno.step(0.1, -0.015, 40.0)
        for _ in range(200):
            capno.step(0.1, -0.0001, 40.0)
        assert capno.co2 < 2.0
        for _ in range(15):
            resumed = capno.step(0.1, -0.015, 40.0)
        assert resumed == pytest.approx(40.0, abs=0.5)

    @pytest.mark.parametrize(("mode", "p_insp"), [("VCV", 0.0), ("PSV", 10.0), ("CPAP", 0.0)])
    def test_capnogram_follows_spontaneous_breaths_over_backup_and_after_stop(
        self, engine_factory, mode, p_insp
    ):
        engine = engine_factory(config=SimulationConfig(mode="awake"), start=True)
        engine.set_airway_mode("ETT")
        for _ in range(50):
            engine.step(0.1)

        engine.set_vent_settings(rr=6.0, vt=0.5, peep=5.0, ie="1:2", mode=mode, p_insp=p_insp)
        engine.set_vent_power(True)
        for _ in range(20):
            engine.step(0.1)
        assert engine.state.rr > 8.0

        capno_rr = self._capnogram_rate(engine)
        assert capno_rr == pytest.approx(engine.state.rr, abs=2.0)
        assert abs(capno_rr - 6.0) > 1.5

        engine.set_vent_power(False)
        assert self._capnogram_rate(engine) == pytest.approx(engine.state.rr, abs=2.0)

    def test_psv_apnea_backup_restores_ventilation_and_capnography(self, awake_engine):
        engine = awake_engine
        engine.set_airway_mode("ETT")
        # This pressure and duration provide adequate ventilation for the backup scenario.
        engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="PSV", p_insp=10.0, t_insp=1.7)
        engine.set_vent_power(True)
        engine.give_drug_bolus("Rocuronium", 0.8 * engine.patient.weight)
        for _ in range(300):
            engine.step(0.5)
        assert engine.resp.state.apnea
        assert engine.state.mv > 5.0
        assert engine.state.etco2_signal_valid
        assert engine.state.display_etco2 == pytest.approx(engine.state.etco2, abs=2.0)
        assert self._capnogram_rate(engine) == pytest.approx(12.0, abs=1.0)

        engine.set_airway_mode("None")
        engine.step(0.1)
        assert engine.state.capno_co2 == 0.0
        assert not engine.state.etco2_signal_valid
        engine.set_airway_mode("ETT")
        engine.step(0.1)
        assert not engine.state.etco2_signal_valid  # A fresh exhalation is required.
        for _ in range(round((engine.vent.apnea_backup_s + 10.0) / 0.1)):
            engine.step(0.1)
        assert engine.state.etco2_signal_valid

        engine.set_vent_settings(rr=0, vt=0, peep=5.0, ie="1:2", mode="CPAP", p_insp=0)
        for _ in range(42):
            engine.step(0.5)
        assert engine.state.mv == 0.0
        assert engine.state.capno_co2 == 0.0
        assert not engine.state.etco2_signal_valid
        assert engine.state.alarms["EtCO2"]["low"]

    def test_first_breath_capnography_apnea_washout_and_unplanned_disconnect(
        self, engine_factory, advance_time
    ):
        """Breath 1 after apneic intubation shows EtCO2; sustained apnea zeros VT/RR/MV
        and washes capno_co2 to 0 within 15 s; unplanned disconnect alarms immediately.
        """
        engine = engine_factory(config=SimulationConfig(mode="awake", dt=0.1, rng_seed=1), start=True)
        engine.give_drug_bolus("propofol", 200.0)
        engine.give_drug_bolus("roc", 70.0)
        advance_time(engine, 90.0, dt=1.0)

        engine.set_airway_mode("ETT")
        engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        engine.set_vent_power(True)

        peak_capno = 0.0
        for _ in range(50):
            engine.step(0.1)
            peak_capno = max(peak_capno, engine.state.capno_co2)
        assert peak_capno > 20.0
        advance_time(engine, 15.0, dt=0.5)
        assert engine.state.etco2_signal_valid and engine.state.display_etco2 > 25.0

        engine.set_vent_power(False)
        advance_time(engine, 15.0, dt=0.5)
        assert engine.state.capno_co2 == 0.0
        advance_time(engine, 5.0, dt=0.5)
        assert engine.state.vt == engine.state.rr == engine.state.mv == 0.0

        # Re-ventilate to establish a valid breath, then disconnect while vent stays on.
        engine.set_vent_power(True)
        advance_time(engine, 10.0, dt=0.5)
        assert engine.state.etco2_signal_valid
        engine.set_airway_mode("None")
        engine.step(0.1)
        assert not engine.etco2_readout.awaiting_first_breath
        assert not engine.state.etco2_signal_valid
        assert engine.state.alarms["EtCO2"]["low"]

    def test_partial_obstruction_endpoint_and_apneic_end_tidal_gas_hold(self, engine_factory, advance_time):
        """Partial airway narrowing does not scale the alveolar EtCO2 ceiling by patency,
        and end-tidal gas fractions hold the last exhaled sample during apnea until timeout.
        """
        narrowed = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=1), start=True)
        narrowed.set_airway_obstruction(0.45)
        narrowed.set_vent_settings(rr=8, vt=0.65, peep=5.0, ie="1:3", mode="VCV")
        advance_time(narrowed, 45.0, dt=0.5)
        assert narrowed.airway_status.patency < 0.70
        assert narrowed.state.display_etco2 == pytest.approx(narrowed.resp.state.etco2, abs=2.5)

        apneic = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=0.5, rng_seed=1),
            start=True,
        )
        apneic.give_drug_bolus("roc", 80.0)
        apneic.set_vent_power(False)
        advance_time(apneic, 3.0, dt=0.5)
        sevo_0, o2_0 = apneic.state.et_sevo, apneic.state.et_o2
        assert sevo_0 > 1.0 and o2_0 > 25.0
        apneic.pk_sevo.state.p_alv = 0.001
        advance_time(apneic, 5.0, dt=0.5)
        assert apneic.state.etco2_signal_valid
        assert apneic.state.et_sevo == pytest.approx(sevo_0, rel=1e-4)
        assert apneic.state.et_o2 == pytest.approx(o2_0, rel=1e-4)
        advance_time(apneic, 12.0, dt=0.5)
        assert not apneic.state.etco2_signal_valid
        assert apneic.state.et_sevo == apneic.state.et_o2 == apneic.state.et_mac == 0.0

    def test_slow_vcv_rate_keeps_etco2_valid(self, engine_factory, advance_time):
        """Slow VCV (RR 3.5/min, 17.1 s period) does not blank EtCO2 or fire false LOW EtCO2."""
        slow = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
        slow.set_vent_settings(mode="VCV", rr=3.5, vt=0.65, peep=5.0, ie="1:2")
        advance_time(slow, 60.0, dt=0.5)
        for _ in range(60):
            slow.step(0.5)
            assert slow.state.etco2_signal_valid
            assert slow.state.display_etco2 > 30.0
            assert not slow.state.alarms.get("EtCO2", {}).get("low")

    def test_arrest_etco2_washout_is_monotonic_and_cpr_couples_to_airway_waveforms(
        self, engine_factory, advance_time
    ):
        """Rapid no-flow EtCO2 washout declines monotonically while keeping live end-tidal gases;
        110/min CPR mechanically couples into Paw, Flow, and CO2 oscillations when sealed,
        produces sub-dead-space passive ripples with apnea timeout when the ventilator is off,
        and clears cleanly when disconnected."""
        washout = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=0.5, rng_seed=1),
            start=True,
        )
        washout.set_rhythm("ASYSTOLE")
        assert washout.state.dbp <= washout.state.map <= washout.state.sbp
        assert washout.state.sbp == pytest.approx(washout.hemo.state.venous_pressure)
        readings = []
        for _ in range(360):
            washout.step(0.5)
            if washout.state.etco2_signal_valid:
                readings.append(washout.state.display_etco2)
        assert readings and readings[0] > 30.0 and readings[-1] < 5.0
        for prev, curr in pairwise(readings):
            assert curr <= prev + 1e-6
        assert washout.etco2_readout.gas_valid
        assert washout.state.et_o2 > 50.0
        assert washout.state.et_sevo > 0.5

        cpr = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.2, rng_seed=1), start=True)
        cpr.set_rhythm("ASYSTOLE")
        cpr.set_chest_compressions(True)
        advance_time(cpr, 40.0, dt=0.2)
        cpr.output_buffer.clear()
        advance_time(cpr, 5.0, dt=0.2)
        samples = list(cpr.output_buffer)
        paws = [s.paw for s in samples]
        flows = [s.flow for s in samples]
        co2s = [s.capno_co2 for s in samples]
        assert max(paws) > 35.0 and cpr.state.paw_peak > 35.0
        assert math.isnan(cpr.state.paw_plat)
        assert max(flows) > 15.0 and min(flows) < -35.0
        assert 15.0 <= cpr.state.display_etco2 <= 25.0
        assert cpr.state.display_etco2 == pytest.approx(cpr.state.etco2, abs=5.5)
        plateau = [c for c in co2s if c > 4.0]
        assert max(plateau) - min(plateau) > 5.0

        # Apneic CPR shows passive compression Paw/Flow/CO2 ripples without fake breaths or EtCO2 lock.
        cpr.set_vent_power(False)
        advance_time(cpr, 25.0, dt=0.2)
        cpr.output_buffer.clear()
        advance_time(cpr, 3.0, dt=0.2)
        apneic = list(cpr.output_buffer)
        assert max(s.paw for s in apneic) > 1.0 and min(s.paw for s in apneic) < 0.0
        assert max(s.flow for s in apneic) > 3.0 and min(s.flow for s in apneic) < -8.0
        assert max(s.capno_co2 for s in apneic) < 10.0
        assert cpr.state.rr == 0.0 and cpr.state.mv == 0.0
        assert not cpr.state.etco2_signal_valid

        # Disconnecting removes Y-piece compression signals completely.
        cpr.set_airway_mode("None")
        advance_time(cpr, 1.0, dt=0.2)
        cpr.output_buffer.clear()
        advance_time(cpr, 2.0, dt=0.2)
        disc = list(cpr.output_buffer)
        assert all(abs(s.paw) < 0.05 and abs(s.flow) < 0.2 and s.capno_co2 == 0.0 for s in disc)

    def test_cpr_respiratory_modes_airway_closure_auto_trigger_and_dt_invariance(
        self, engine_factory, advance_time
    ):
        """Coupled CPR mechanics reproduce mode-specific Ppeak/VT responses (Neumamm 2021;
        Orlob 2021), PEEP-dependent airway closure and thoracic distension (Grieco 2019;
        Lesimple 2022), PSV recoil auto-triggering (extended from Neumamm 2021), and dt-invariance."""
        # 1. VCV vs PCV under CPR (Neumamm 2021; Orlob 2021).
        base_pcv = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.2, rng_seed=7), start=True)
        base_pcv.set_rhythm("ASYSTOLE")
        base_pcv.set_vent_settings(mode="PCV", rr=10, vt=0.60, peep=0.0, ie="1:5", p_insp=16.0, p_max=100.0)
        advance_time(base_pcv, 30.0, dt=0.2)
        pcv_ppeak_0, pcv_vt_0 = base_pcv.state.paw_peak, base_pcv.state.vt
        cpr_pcv = copy.deepcopy(base_pcv)
        cpr_pcv.set_chest_compressions(True)
        advance_time(cpr_pcv, 30.0, dt=0.2)
        assert cpr_pcv.state.paw_peak > pcv_ppeak_0 + 10.0
        assert cpr_pcv.state.vt < 0.90 * pcv_vt_0

        vcv_low_pmax = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.2, rng_seed=7), start=True)
        vcv_low_pmax.set_rhythm("ASYSTOLE")
        vcv_low_pmax.set_vent_settings(mode="VCV", rr=12, vt=0.50, peep=5.0, ie="1:2", p_max=40.0)
        vcv_low_pmax.set_chest_compressions(True)
        advance_time(vcv_low_pmax, 30.0, dt=0.2)
        assert vcv_low_pmax.vent.pressure_limited
        vcv_low_pmax.set_vent_settings(mode="VCV", rr=12, vt=0.50, peep=5.0, ie="1:2", p_max=60.0)
        advance_time(vcv_low_pmax, 20.0, dt=0.2)
        assert not vcv_low_pmax.vent.pressure_limited
        assert 450.0 < vcv_low_pmax.state.vt < 560.0

        # 2. PEEP-dependent airway closure (Grieco 2019) & thoracic distension (Lesimple 2022).
        def cpr_exp_metrics(peep: float, vt: float = 0.50, bronch: float = 0.0, **patient_kw):
            eng = engine_factory(
                config=SimulationConfig(mode="steady_state", dt=0.2, rng_seed=7),
                start=True,
                **patient_kw,
            )
            if bronch > 0.0:
                eng.set_bronchospasm(bronch)
            eng.set_rhythm("ASYSTOLE")
            eng.set_vent_settings(mode="VCV", rr=10, vt=vt, peep=peep, ie="1:5", p_max=65.0)
            eng.set_chest_compressions(True)
            advance_time(eng, 40.0, dt=0.2)
            eng.output_buffer.clear()
            advance_time(eng, 6.0, dt=0.2)
            buf = list(eng.output_buffer)
            i_vmax = max(range(len(buf)), key=lambda i: buf[i].volume)
            late = [buf[(i_vmax + k) % len(buf)] for k in range(250, 450)]
            exp_dpaw = max(x.paw for x in late) - peep
            co2s = [buf[(i_vmax + k) % len(buf)].capno_co2 for k in range(30, 480)]
            pks = [
                co2s[i] for i in range(1, len(co2s) - 1)
                if co2s[i] >= co2s[i - 1] and co2s[i] > co2s[i + 1] and co2s[i] > 2.0
            ]
            trs = [
                co2s[i] for i in range(1, len(co2s) - 1)
                if co2s[i] <= co2s[i - 1] and co2s[i] < co2s[i + 1] and co2s[i] > 0.3
            ]
            aois = [(p - t) / p for p, t in zip(pks, trs, strict=False) if p > 2.0]
            aoi = sum(aois) / len(aois) if aois else 0.0
            # Time from peak lung inflation until expiratory flow first turns positive (recoil).
            t_pos = next(
                (k * 0.01 for k in range(10, 480) if buf[(i_vmax + k) % len(buf)].flow > 0.5),
                5.0,
            )
            return exp_dpaw, aoi, eng.state.display_etco2, eng.state.etco2, t_pos

        dpaw_0, aoi_0, disp_0, et_0, _ = cpr_exp_metrics(0.0)
        dpaw_5, aoi_5, disp_5, et_5, _ = cpr_exp_metrics(5.0)
        dpaw_10, aoi_10, disp_10, et_10, _ = cpr_exp_metrics(10.0)
        assert dpaw_0 < dpaw_5 < dpaw_10
        assert aoi_0 < aoi_5
        assert aoi_10 > aoi_0 + 0.15
        for disp, true_et in ((disp_0, et_0), (disp_5, et_5), (disp_10, et_10)):
            assert disp == pytest.approx(true_et, abs=3.5)

        # Obesity raises airway closing pressure, causing closure at PEEP 5 that opens at PEEP 10.
        dpaw_ob5, aoi_ob5, _, _, _ = cpr_exp_metrics(5.0, weight=110, height=170)
        dpaw_ob10, aoi_ob10, _, _, _ = cpr_exp_metrics(10.0, weight=110, height=170)
        assert dpaw_ob5 < dpaw_5 - 2.0
        assert dpaw_ob10 > dpaw_ob5 + 3.0 and aoi_ob10 > aoi_ob5 + 0.15

        # Thoracic distension (larger VT or longer RC time constant) delays recoil oscillations.
        _, _, _, _, t_pos_small = cpr_exp_metrics(5.0, vt=0.35)
        _, _, _, _, t_pos_large = cpr_exp_metrics(5.0, vt=0.85)
        _, _, _, _, t_pos_slow = cpr_exp_metrics(5.0, vt=0.65, bronch=0.45)
        assert t_pos_large > t_pos_small + 0.35
        assert t_pos_slow > t_pos_large + 0.8

        # 3. PSV chest-recoil auto-triggering stops when flow trigger is raised above recoil flow.
        psv_low = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.2, rng_seed=7), start=True)
        psv_low.set_rhythm("ASYSTOLE")
        psv_low.set_vent_settings(mode="PSV", rr=10, vt=0.50, ie="1:2", p_support=8.0, p_insp=15.0, peep=5.0, trigger=3.0)
        psv_low.set_chest_compressions(True)
        advance_time(psv_low, 30.0, dt=0.2)
        assert psv_low.state.rr > 40.0

        psv_high = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.2, rng_seed=7), start=True)
        psv_high.set_rhythm("ASYSTOLE")
        psv_high.set_vent_settings(mode="PSV", rr=10, vt=0.50, ie="1:2", p_support=8.0, p_insp=15.0, peep=5.0, trigger=20.0)
        psv_high.set_chest_compressions(True)
        advance_time(psv_high, 40.0, dt=0.2)
        assert psv_high.state.rr == pytest.approx(10.0, abs=1.5)
        assert psv_high.state.etco2_signal_valid

        # 4. Step-size (dt) invariance during active CPR.
        results = []
        for dt in (0.1, 1.0):
            eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=dt, rng_seed=7), start=True)
            eng.set_rhythm("ASYSTOLE")
            eng.set_vent_settings(mode="VCV", rr=12, vt=0.50, peep=5.0, ie="1:2", p_max=40.0)
            eng.set_chest_compressions(True)
            advance_time(eng, 40.0, dt=dt)
            results.append((eng.state.vt, eng.state.paw_peak, eng.state.display_etco2, eng.state.art_dbp))
        for a, b in zip(*results, strict=True):
            assert a == pytest.approx(b, abs=0.5)

    def test_shallow_breathing_and_zero_co2_gas_valid_behavior(self):
        """Shallow breathing (>3 mmHg) stays valid past 20 s; unsampled zero-CO2 cycles do not set gas_valid."""
        empty_ro = EtCO2Readout()
        empty_ro.update(0.5, False, 0.0, inhaled_l=0.4, dead_space_l=0.15, breath_index=1)
        assert not empty_ro.gas_valid

        ro = EtCO2Readout()
        for _ in range(10):
            for _ in range(15):
                ro.update(0.1, True, 18.0, inhaled_l=0.0, dead_space_l=0.15, breath_index=0)
            for _ in range(10):
                ro.update(0.1, False, 0.0, inhaled_l=0.08, dead_space_l=0.15, breath_index=0)
        assert ro.valid and ro.value == pytest.approx(18.0)
