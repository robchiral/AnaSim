
import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.capno import Capnograph
from anasim.monitors.nibp import NIBPMonitor


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
        engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
        for _ in range(1200):
            engine.step(0.1)
        assert engine.state.rr == 0.0

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
        for _ in range(1000):
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
