import numpy as np
import pytest

from anasim.core.action_log import ACTION_SHOCK
from anasim.core.enums import RhythmType
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.monitors.ecg import ECGMonitor
from anasim.physiology.resuscitation import (
    CHARGE_HOLD_S,
    CHARGE_TIME_S,
    Defibrillator,
    ShockResult,
    shock_outcome,
)


def test_charging_consumption_disarming_and_expiry_use_simulation_time():
    device = Defibrillator()
    device.charge(10)
    device.charge(11)  # Repeated requests do not restart or finish charging.
    assert device.status(10 + CHARGE_TIME_S - 0.01) == "charging"
    assert device.status(10 + CHARGE_TIME_S) == "ready"
    device.consume(16, False)
    assert device.status(16) == "idle"
    assert device.last_shock == (16, False)
    device.charge(20)
    device.waiting_for_r = True
    assert device.status(20 + CHARGE_TIME_S + CHARGE_HOLD_S) == "idle"
    assert not device.waiting_for_r
    device.charge(100)
    device.disarm()
    assert device.status(110) == "idle"


@pytest.mark.parametrize("rhythm", [RhythmType.SINUS, RhythmType.AFIB])
def test_r_on_t_risk_follows_rendered_t_wave_including_irregular_rr(rhythm):
    cycle, ecg = CardiacCycle(), ECGMonitor()
    cycle.seed(80, rhythm)
    seen_vulnerable = seen_safe = False
    for _ in range(400):
        sample = cycle.step(0.01, 80, rhythm)
        ecg.step(0.01, sample)
        vulnerable = ecg.vulnerable(sample)
        outcome = shock_outcome(
            rhythm, synchronized=False, energy_failure=0, electrically_silent=False,
            vulnerable=vulnerable, antiarrhythmic=0, rng=np.random.default_rng(3),
        )
        if vulnerable:
            seen_vulnerable = True
            assert outcome == RhythmType.VFIB
        else:
            seen_safe = True
            assert outcome != RhythmType.VFIB
        if sample.beat_started:
            assert not vulnerable
    assert seen_vulnerable and seen_safe


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_sync_fires_on_next_r_at_waveform_resolution(engine_factory, monkeypatch, dt):
    engine = engine_factory(start=True)
    engine.set_rhythm("SVT")
    engine.charge_defibrillator()
    for _ in range(5):
        engine.step(1)
    # Start the next beat away from its R peak, with a stable known interval.
    sample = engine.cardiac_cycle.seed(180, RhythmType.SVT)
    engine.ecg.step(0.01, sample)
    sample = engine.cardiac_cycle.step(0.1, 180, RhythmType.SVT)
    engine.ecg.step(0.1, sample)
    requested = engine.state.time
    deliver = engine._deliver_shock
    phases = []

    def capture(time, *, synchronized):
        phases.append(engine.cardiac_cycle.current.elapsed_s)
        deliver(time, synchronized=synchronized)

    monkeypatch.setattr(engine, "_deliver_shock", capture)
    assert engine.defibrillate(synchronized=True) == ShockResult.PENDING
    assert not engine.actions.since_case(ACTION_SHOCK)
    for _ in range(round(1 / dt)):
        engine.step(dt)
    records = engine.actions.since_case(ACTION_SHOCK)
    assert len(records) == 1
    assert records[0].time == pytest.approx(requested + 60 / 180 - 0.1, abs=1e-8)
    assert len(phases) == 1 and 0 <= phases[0] < 0.01
    assert engine.defibrillator.status(engine.state.time) == "idle"


def test_pending_sync_can_be_cancelled_and_does_not_fire_on_vf_or_compressions(engine_factory):
    engine = engine_factory(start=True)
    engine.charge_defibrillator()
    engine.step(CHARGE_TIME_S)
    assert engine.defibrillate(synchronized=True) == ShockResult.PENDING
    engine.disarm_defibrillator()
    engine.step(1)
    assert not engine.actions.since_case(ACTION_SHOCK)
    engine.charge_defibrillator()
    engine.step(CHARGE_TIME_S)
    assert engine.defibrillate(synchronized=True) == ShockResult.PENDING
    engine.set_rhythm("VFIB")
    engine.set_chest_compressions(True)
    engine.step(1)
    assert not engine.actions.since_case(ACTION_SHOCK)
    assert engine.defibrillator.status(engine.state.time) == "ready"
    assert engine.defibrillate(synchronized=True) == ShockResult.NOT_DELIVERED


def test_unready_shocks_do_not_change_rhythm_or_consume_random_draws(engine_factory, monkeypatch):
    engine = engine_factory(start=True)

    class _NoDraw:
        def random(self):
            raise AssertionError("An unready shock must not draw an outcome")

    monkeypatch.setattr(engine, "_shock_rng", _NoDraw())
    engine.set_rhythm("VFIB")
    engine.charge_defibrillator()
    engine.step(CHARGE_TIME_S - 0.1)
    for _ in range(10):
        assert engine.defibrillate() == ShockResult.NOT_READY
    assert engine.hemo.rhythm_type == RhythmType.VFIB
    assert not engine.actions.since_case(ACTION_SHOCK)
