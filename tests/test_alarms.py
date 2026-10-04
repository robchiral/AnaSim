from anasim.monitors.alarms import AlarmSystem


def test_alarm_needs_a_continuous_violation_for_its_delay():
    alarms = AlarmSystem(delays={"HR": 2.0})
    sounded = ["HR" in alarms.update({"HR": hr}, dt=1.0) for hr in (30, 100, 30, 30)]
    assert sounded == [False, False, False, True]

    # A missing or invalid signal interrupts the continuous violation.
    for missing in ({}, {"HR": None}):
        assert alarms.update(missing, dt=1.0) == {}
        assert alarms.update({"HR": 30}, dt=1.0) == {}
        assert "HR" in alarms.update({"HR": 30}, dt=1.0)

    # The delay is in seconds, whatever the update interval.
    alarms = AlarmSystem(delays={"HR": 2.0})
    sounded = ["HR" in alarms.update({"HR": 30}, dt=0.5) for _ in range(4)]
    assert sounded == [False, False, False, True]
