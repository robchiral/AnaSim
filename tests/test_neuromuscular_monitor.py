import json

import pytest

from anasim.monitors.neuromuscular import NeuromuscularMonitor
from anasim.web import WebSession


@pytest.mark.parametrize("t1,ratio,count,low", [
    (0.04, 1, 0, False),
    (0.1, 0.03, 1, False),
    (0.2, 0.09, 2, False),
    (0.3, 0.16, 3, False),
    (0.8, 0.8, 4, False),
    (0.4, 0.25, 4, False),
    (0.1, 1, 4, True),
])
def test_tof_detects_individual_responses_and_qualifies_weak_trains(t1, ratio, count, low):
    monitor = NeuromuscularMonitor()
    monitor.establish_baseline(1)
    for time in (0, 0.5, 1, 1.5):
        monitor.step(time, t1, ratio)
    reading = monitor.reading
    assert reading is not None
    assert reading.count == count
    assert reading.low_signal is low
    assert reading.t1 == pytest.approx(t1 * 100)
    assert reading.ratio == (pytest.approx(ratio * 100) if count == 4 else None)
    assert reading.responses[0] == t1
    assert reading.responses[-1] == pytest.approx(t1 * ratio)


def test_reading_holds_until_four_separate_pulses_finish():
    monitor = NeuromuscularMonitor()
    for time in (0, 0.5, 1, 1.5):
        monitor.step(time, 1, 1)
    baseline = monitor.reading
    monitor.step(19, 0.2, 0.1)
    assert monitor.reading is baseline
    monitor.step(20, 0.2, 0.1)
    monitor.step(20.49, 0.2, 0.1)
    assert monitor.snapshot(20.49)["pulses"] == 1
    monitor.step(20.5, 0.2, 0.1)
    monitor.step(21, 0.2, 0.1)
    assert monitor.reading is baseline
    monitor.step(21.5, 0.2, 0.1)
    assert monitor.reading is not baseline
    assert monitor.reading is not None and monitor.reading.count == 2


def test_manual_stimulation_mode_change_and_missing_baseline():
    monitor = NeuromuscularMonitor()
    monitor.establish_baseline(0.4)
    monitor.set_repeat(0, 0)
    monitor.set_mode("single", 0)
    monitor.step(5, 0.4, 1)
    assert monitor.reading is None
    monitor.measure(5)
    monitor.step(5, 0.4, 1)
    assert monitor.reading is not None
    assert monitor.reading.t1 is None
    assert monitor.reading.responses == (0.4,)
    with pytest.raises(ValueError, match="10 seconds"):
        monitor.measure(14.9)
    monitor.measure(15)
    with pytest.raises(ValueError, match="finish"):
        monitor.set_mode("tof", 15)
    monitor.step(15, 1, 1)
    monitor.set_mode("tof", 15)
    monitor.set_repeat(20, 15)
    monitor.step(24.9, 1, 1)
    assert monitor.reading is None
    monitor.step(25, 1, 1)
    assert not monitor.snapshot(25)["measuring"]
    monitor.step(35, 1, 1)
    assert monitor.snapshot(35)["measuring"]


def test_phase_i_web_monitor_keeps_selected_mode_and_survives_reconnect(advance_time):
    session = WebSession({"age": 40, "weight": 70, "height": 170, "sex": "male", "mode": "awake"})
    engine = session.engine
    engine.start()
    advance_time(engine, 2, dt=0.5)
    initial = session.snapshot()["nmb"]
    assert initial["baseline"] == 1.0
    assert initial["reading"]["ratio"] == 100
    session.command("drug_bolus", json.dumps({"key": "succinylcholine", "amount": 70}))
    advance_time(engine, 88, dt=0.5)
    snap = session.snapshot()
    # Readings taken between polls all reach the browser trend.
    assert snap["nmb_trend"] == engine.nmb.trend_history(after=initial["reading"]["time"])
    assert len(snap["nmb_trend"]) == 4
    assert snap["nmb"]["mode"] == "tof"
    assert snap["nmb"]["reading"]["count"] == 0
    assert snap["nmb"]["reading"]["ratio"] is None
    assert engine.state.tof == 100
    session.command("nmb_mode", '{"mode":"single"}')
    advance_time(engine, 2, dt=0.5)
    snap = session.snapshot()["nmb"]
    assert snap["repeat"] == 10
    assert snap["reading"]["t1"] < 5
    assert len(snap["reading"]["responses"]) == 1
    session.command("run", '{"running":false}')
    session.step(1)
    assert session.snapshot()["nmb"] == snap
    session.replay_waves()
    assert session.snapshot()["nmb"] == snap
    assert json.loads(session.info())["nmb_history"] == engine.nmb.trend_history()
