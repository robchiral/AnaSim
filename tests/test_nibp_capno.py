
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
        for _ in range(900):
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

    def test_shock_boundary_is_continuous_and_failed_cuff_prompts_remeasurement(self, engine_factory):
        """Crossing MAP 60 mmHg does not step-jump NIBP bias or failure rate (#35),
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

        engine = engine_factory(
            config=SimulationConfig(mode="awake", arterial_line_enabled=False, rng_seed=1),
            start=True,
        )
        engine.state.nibp_measurement_failed = True
        engine.state.nibp_sys = engine.state.nibp_dia = engine.state.nibp_map = float("nan")
        met, msg = require_map_at_least(65)(engine)
        assert not met and msg == "Measure blood pressure again"


class TestCapnography:
    def test_obstruction_keeps_impedance_effort_and_expires_end_tidal_reading(self, awake_engine):
        engine = awake_engine
        engine.set_airway_mode("ETT")
        for _ in range(300):
            engine.step(0.1)
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
        for _ in range(1200):
            engine.step(0.1)
        # Residual diaphragm contractions can persist without airflow.
        assert engine.resp.state.muscle_factor < 0.2
        assert engine.resp_mech.effort.amplitude < 0.6 * effort_before_block
        assert engine.state.vt == engine.state.mv == engine.state.va == 0.0
        assert not engine.state.etco2_signal_valid

    def test_patient_effort_during_exhalation_notches_the_plateau(self):
        """A curare cleft: an inspiratory effort draws fresh gas past the sampling port."""
        def exhale(capno, notch):
            trace = []
            for k in range(150):
                volume = -0.004
                if notch and 50 <= k < 55:
                    volume = 0.006  # 30 mL drawn in over 0.1 s
                elif notch and 55 <= k < 62:
                    volume = -0.0043  # Breathed out again, so the drawn-in gas returns first
                trace.append(capno.step(0.02, volume, 38.0))
            return trace

        smooth = exhale(Capnograph(0.15), notch=False)
        notched = exhale(Capnograph(0.15), notch=True)
        # Phase I holds inspired gas until the dead space empties, then the plateau rises to end-tidal.
        assert smooth[5] < 1.0 and smooth[-1] == pytest.approx(38.0, abs=1.5)
        dip = max(a - b for a, b in zip(smooth, notched, strict=True))
        assert dip > 10.0
        assert notched[-1] == pytest.approx(smooth[-1], abs=1.0)

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
        assert capno._window_co2 == pytest.approx(sum(v * p for v, p in capno._window), abs=1e-12)
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
        for _ in range(1500):
            engine.step(0.1)
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
        for _ in range(200):
            engine.step(0.1)
        assert engine.state.mv == 0.0
        assert engine.state.capno_co2 == 0.0
        assert not engine.state.etco2_signal_valid
        assert engine.state.alarms["EtCO2"]["low"]

    def test_first_breath_capnography_and_apnea_spirometry_reset(self, engine_factory, advance_time):
        """Breath 1 after apneic intubation shows EtCO2; sustained apnea zeros VT/RR/MV."""
        engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
        engine.give_drug_bolus("propofol", 200.0)
        engine.give_drug_bolus("roc", 60.0)
        advance_time(engine, 90.0, dt=0.5)

        engine.set_airway_mode("ETT")
        engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        engine.set_vent_power(True)

        peak_capno = 0.0
        for _ in range(50):
            engine.step(0.1)
            peak_capno = max(peak_capno, engine.state.capno_co2)
        assert peak_capno > 20.0

        engine.set_vent_power(False)
        advance_time(engine, 20.0, dt=0.5)
        assert engine.state.vt == engine.state.rr == engine.state.mv == 0.0

    def test_apnea_washout_curare_cleft_endpoint_and_unplanned_disconnect(self, engine_factory, advance_time):
        """Post-expiratory apneic absorption creep washes capno_co2 to 0 within 15 s (#2);
        a small curare cleft does not latch a false low EtCO2, and an unplanned disconnect
        with active ventilation does not re-arm first-breath alarm grace (#17).
        """
        engine = engine_factory(config=SimulationConfig(mode="awake", dt=0.1, rng_seed=1), start=True)
        engine.give_drug_bolus("propofol", 200.0)
        engine.give_drug_bolus("roc", 70.0)
        advance_time(engine, 90.0, dt=0.5)
        engine.set_airway_mode("ETT")
        engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        engine.set_vent_power(True)
        advance_time(engine, 20.0, dt=0.1)
        assert engine.state.etco2_signal_valid and engine.state.display_etco2 > 25.0

        engine.set_vent_power(False)
        advance_time(engine, 15.0, dt=0.1)
        assert engine.state.capno_co2 == 0.0

        # Re-ventilate to establish a valid breath, then disconnect while vent stays on.
        engine.set_vent_power(True)
        advance_time(engine, 10.0, dt=0.1)
        assert engine.state.etco2_signal_valid
        engine.set_airway_mode("None")
        engine.step(0.1)
        assert not engine.etco2_readout.awaiting_first_breath
        assert not engine.state.etco2_signal_valid
        assert engine.state.alarms["EtCO2"]["low"]

        # Curare cleft (< dead space) during mid-plateau must not latch the pre-notch value.
        capno = Capnograph(0.15)
        readout = EtCO2Readout()
        capno.step(0.5, 0.5, 38.0)
        readout.update(0.5, capno.exhaling, capno.co2, inhaled_l=capno._inhaled,
                       dead_space_l=capno.dead_space, breath_index=capno.breath_index)
        for k in range(150):
            volume = 0.006 if 40 <= k < 45 else (-0.0043 if 45 <= k < 52 else -0.004)
            co2 = capno.step(0.02, volume, 38.0)
            readout.update(0.02, capno.exhaling, co2, inhaled_l=capno._inhaled,
                           dead_space_l=capno.dead_space, breath_index=capno.breath_index)
        capno.step(0.5, 0.5, 38.0)
        readout.update(0.5, capno.exhaling, capno.co2, inhaled_l=capno._inhaled,
                       dead_space_l=capno.dead_space, breath_index=capno.breath_index)
        assert readout.value == pytest.approx(38.0, abs=1.5)

    def test_partial_obstruction_endpoint_and_apneic_end_tidal_gas_hold(self, engine_factory, advance_time):
        """Partial airway narrowing does not scale the alveolar EtCO2 ceiling by patency,
        and end-tidal gas fractions hold the last exhaled sample during apnea until timeout.
        """
        narrowed = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.1, rng_seed=1), start=True)
        narrowed.set_airway_obstruction(0.45)
        narrowed.set_vent_settings(rr=8, vt=0.65, peep=5.0, ie="1:3", mode="VCV")
        advance_time(narrowed, 45.0, dt=0.1)
        assert narrowed.airway_status.patency < 0.70
        assert narrowed.state.display_etco2 == pytest.approx(narrowed.resp.state.etco2, abs=2.5)

        sevo = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=0.2, rng_seed=1), start=True
        )
        advance_time(sevo, 15.0, dt=0.2)
        sevo.give_drug_bolus("roc", 80.0)
        sevo.set_vent_power(False)
        advance_time(sevo, 3.0, dt=0.2)
        held_sevo = sevo.state.et_sevo
        held_o2 = sevo.state.et_o2
        assert held_sevo > 1.0 and held_o2 > 25.0
        sevo.pk_sevo.state.p_alv = 0.001
        advance_time(sevo, 5.0, dt=0.2)
        assert sevo.state.etco2_signal_valid
        assert sevo.state.et_sevo == pytest.approx(held_sevo, rel=1e-4)
        assert sevo.state.et_o2 == pytest.approx(held_o2, rel=1e-4)
        advance_time(sevo, 12.0, dt=0.2)
        assert not sevo.state.etco2_signal_valid
        assert sevo.state.et_sevo == sevo.state.et_o2 == sevo.state.et_mac == 0.0
