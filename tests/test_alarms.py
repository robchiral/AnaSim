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


def test_integer_channel_alarms_match_rounded_monitor_numerics(engine_factory):
    """Monitor alarms evaluate rounded integer display values so numerics at the limit do not alarm."""
    engine = engine_factory(
        config=SimulationConfig(mode="awake", arterial_line_enabled=False, rng_seed=1),
        start=True,
    )
    engine.step(0.1)

    # Integrated step_monitors applies display_int rounding to fractional pressures.
    engine.state.nibp_map = 59.6
    engine.step(0.1)
    assert "MAP" not in engine.state.alarms
    engine.state.nibp_map = 59.4
    engine.step(0.1)
    assert engine.state.alarms["MAP"]["low"] is True


def test_pregnancy_adapts_low_etco2_alarm_limit(engine_factory, advance_time):
    """Term pregnancy lowers EtCO2_min so gravid normocapnia (~31 mmHg) does not alarm."""
    gravid = engine_factory(
        config=SimulationConfig(mode="steady_state", rng_seed=1),
        start=True,
        sex="female",
        gestational_age_weeks=39.0,
    )
    assert gravid.alarms.thresholds["EtCO2_min"] <= 26.0
    advance_time(gravid, 15.0, dt=1.0)
    assert "EtCO2" not in gravid.state.alarms

    apneic = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    apneic.give_drug_bolus("propofol", 200.0)
    apneic.give_drug_bolus("remi", 150.0)
    advance_time(apneic, 10.0, dt=0.5)
    assert apneic.state.rr == 0.0 and "RR" not in apneic.state.alarms
    advance_time(apneic, 12.0, dt=0.5)
    assert apneic.state.alarms["RR"]["low"] is True
