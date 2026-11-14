from anasim.core.state import SimulationConfig
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


def test_pea_arrest_alarms_without_arterial_line(engine_factory, advance_time):
    """Pulseless arrest without an arterial line triggers low SpO2 and MAP alarms."""
    engine = engine_factory(
        config=SimulationConfig(mode="awake", arterial_line_enabled=False, rng_seed=1),
        start=True,
    )
    engine.set_airway_mode("ETT")
    engine.set_vent_power(True)
    engine.set_fgf(6.0, 0.0)
    engine.hemo.sv_tone = 0.0
    engine.hemo.kin_sv = 0.0
    advance_time(engine, 20.0, dt=0.5)
    engine.nibp.trigger()
    advance_time(engine, 25.0, dt=0.5)

    assert engine.state.alarms["MAP"]["low"] is True
    assert engine.state.alarms["SpO2"]["low"] is True
