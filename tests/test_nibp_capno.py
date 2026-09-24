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

    def test_cuff_can_fail_in_shock(self):
        reading = _cuff_cycle(_FixedRng(0.0), true_map=35.0, true_sys=55.0, true_dia=25.0)
        assert reading.timestamp is None

    def test_successful_shock_reading_overestimates_pressure(self):
        reading = _cuff_cycle(_FixedRng(0.99), true_map=40.0, true_sys=60.0, true_dia=30.0)
        assert reading.timestamp is not None
        assert reading.map > 40.0
        assert reading.systolic > 60.0

    def test_cuff_gives_no_reading_in_arrest(self):
        reading = _cuff_cycle(
            np.random.default_rng(0), 0.0, 0.0, 0.0, rhythm=RhythmType.ASYSTOLE
        )
        assert reading.timestamp is None


class TestCapnography:
    def test_breath_rises_to_alveolar_plateau_and_clears_on_inspiration(self):
        capno = Capnograph(rng=np.random.default_rng(42))
        p_alv = 40.0
        for _ in range(125):
            capno.step(0.02, "EXP", p_alv, is_spontaneous=False, exp_duration=2.5)
        plateau = capno.state.co2
        for _ in range(50):
            capno.step(0.02, "INSP", p_alv, is_spontaneous=False, exp_duration=2.5)

        assert plateau == pytest.approx(p_alv, abs=2.0)
        assert capno.state.co2 < 2.0

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
    def test_capnogram_follows_spontaneous_breaths_over_a_low_backup_rate(
        self, engine_factory, mode, p_insp
    ):
        engine = engine_factory(config=SimulationConfig(mode="awake"), start=True)
        engine.set_airway_mode("ETT")
        for _ in range(50):
            engine.step(0.1)

        engine.set_vent_settings(rr=6.0, vt=0.5, peep=5.0, ie="1:2", mode=mode, p_insp=p_insp)
        for _ in range(20):
            engine.step(0.1)
        assert engine.state.rr > 8.0

        capno_rr = self._capnogram_rate(engine)
        assert capno_rr == pytest.approx(engine.state.rr, abs=2.0)
        assert abs(capno_rr - 6.0) > 1.5

    def test_capnogram_follows_spontaneous_breaths_after_the_ventilator_stops(self, engine_factory):
        engine = engine_factory(config=SimulationConfig(mode="awake"), start=True)
        engine.set_airway_mode("ETT")
        engine.set_vent_settings(rr=6.0, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        for _ in range(70):
            engine.step(0.1)

        engine.set_vent_settings(rr=0.0, vt=0.0, peep=5.0, ie="1:2", mode="VCV")
        assert self._capnogram_rate(engine) == pytest.approx(engine.state.rr, abs=2.0)
