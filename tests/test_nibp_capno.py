from types import SimpleNamespace

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
    def test_cuff_reads_on_its_interval_and_tracks_map(self, engine_factory):
        engine = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="tiva", dt=0.5),
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
    def test_curare_cleft_notches_the_plateau_during_partial_block(self):
        resp_state = SimpleNamespace(rr=2.0, drive_central=0.6, muscle_factor=0.5)
        ctx = Capnograph.build_context(resp_state, vent_rr=12.0, insp_fraction=0.33, vent_active=True)
        assert ctx.curare_active

        capno_with = Capnograph(rng=np.random.default_rng(0))
        capno_without = Capnograph(rng=np.random.default_rng(0))
        dips = []
        for _ in np.arange(0, ctx.exp_duration, 0.02):
            common = dict(
                is_spontaneous=ctx.is_spontaneous,
                exp_duration=ctx.exp_duration,
                airway_obstruction=0.0,
            )
            co2_with = capno_with.step(
                0.02, "EXP", 40.0, curare_cleft=True, effort_scale=ctx.effort_scale, **common
            )
            co2_without = capno_without.step(
                0.02, "EXP", 40.0, curare_cleft=False, effort_scale=0.0, **common
            )
            dips.append(co2_without - co2_with)

        assert max(dips) > 1.0

    @staticmethod
    def _capnogram_rate(engine, seconds=20.0, dt=0.1):
        prev_phase = engine.capno.last_phase
        breaths = 0
        for _ in range(int(seconds / dt)):
            engine.step(dt)
            phase = engine.capno.last_phase
            if phase == "INSP" and prev_phase != "INSP":
                breaths += 1
            prev_phase = phase
        return breaths / (seconds / 60.0)

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
        engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="PSV", p_insp=10.0)
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
        for _ in range(round((engine.psv_apnea_backup_delay + 10.0) / 0.1)):
            engine.step(0.1)
        assert engine.state.etco2_signal_valid

        engine.set_vent_settings(rr=0, vt=0, peep=5.0, ie="1:2", mode="CPAP", p_insp=0)
        for _ in range(200):
            engine.step(0.1)
        assert engine.state.mv == 0.0
        assert engine.state.capno_co2 == 0.0
        assert not engine.state.etco2_signal_valid
        assert engine.state.alarms["EtCO2"]["low"]
