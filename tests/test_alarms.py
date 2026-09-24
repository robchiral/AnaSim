from anasim.monitors.alarms import AlarmSystem


def test_alarms_flag_limit_violations():
    alarms = AlarmSystem()
    assert alarms.update({"HR": 60, "MAP": 80, "BIS": 50}) == {}

    active = alarms.update({"HR": 30, "MAP": 150, "BIS": 50})
    assert active == {"HR": {"low": True, "high": False}, "MAP": {"low": False, "high": True}}


def test_alarm_needs_a_continuous_violation_for_its_delay():
    alarms = AlarmSystem(delays={"HR": 2.0})
    sounded = ["HR" in alarms.update({"HR": hr}, dt=1.0) for hr in (30, 100, 30, 30)]
    assert sounded == [False, False, False, True]

    # The delay is in seconds, whatever the update interval.
    alarms = AlarmSystem(delays={"HR": 2.0})
    sounded = ["HR" in alarms.update({"HR": 30}, dt=0.5) for _ in range(4)]
    assert sounded == [False, False, False, True]
