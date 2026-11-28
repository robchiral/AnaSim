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


def test_ppeak_temperature_lethal_rhythm_alarms_and_st_blanking(engine_factory, advance_time):
    """Default VCV Ppeak alarms at 40 cmH2O limit and clears on mode switch; low bag APL does not false-alarm;
    O2Supply, Temp, PEA Arrest, and lethal rhythms alarm; ST blanks in VF/asystole."""
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    engine.set_oxygen_supply_connected(False)
    engine.step(0.5)
    assert engine.state.alarms.get("O2Supply", {}).get("low") is True
    engine.set_oxygen_supply_connected(True)
    engine.step(0.5)
    assert "O2Supply" not in engine.state.alarms

    engine.set_airway_obstruction(1.0)
    advance_time(engine, 10.0, dt=0.2)
    assert engine.state.alarms.get("Ppeak", {}).get("high") is True

    engine.set_airway_obstruction(0.0)
    engine.set_vent_settings(mode="PCV", rr=12, vt=0.5, p_insp=12.0, peep=5.0, ie="1:2")
    advance_time(engine, 6.0, dt=0.2)
    assert "Ppeak" not in engine.state.alarms

    engine.set_vent_power(False)
    engine.set_bag_mask_settings(rr=12.0, vt=0.8, p_max=20.0)
    engine.set_bag_mask_ventilation(True)
    engine.set_airway_obstruction(0.95)
    advance_time(engine, 10.0, dt=0.2)
    assert engine.vent.pressure_limited is True
    assert "Ppeak" not in engine.state.alarms
    engine.set_airway_obstruction(0.0)
    engine.set_bag_mask_ventilation(False)
    engine.set_vent_power(True)

    engine.state.temp_c = 39.2
    engine.step(0.5)
    assert engine.state.alarms.get("Temp", {}).get("high") is True

    engine.state.temp_c = 34.5
    engine.step(0.5)
    assert engine.state.alarms.get("Temp", {}).get("low") is True

    engine.set_rhythm("VFIB")
    advance_time(engine, 60.0, dt=1.0)
    assert engine.state.alarms.get("VFIB", {}).get("high") is True
    assert engine.state.st_depression_mv == 0.0
    assert engine.hemo.coronary.st_depression_mv > 0.1

    engine.set_rhythm("ASYSTOLE")
    engine.step(1.0)
    assert engine.state.alarms.get("Asystole", {}).get("high") is True
    assert engine.state.st_depression_mv == 0.0

    # PEA Arrest alarm toggles on the exact step cardiac_arrest changes (no 1-step lag).
    pea = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    pea.hemo.coronary.injury = 0.99
    pea.hemo.add_volume(-2500.0)
    for _ in range(30):
        pea.step(1.0)
        assert ("Arrest" in pea.state.alarms) == pea.state.cardiac_arrest
        if pea.state.cardiac_arrest:
            break
    assert pea.state.cardiac_arrest is True and pea.state.alarms.get("Arrest", {}).get("high") is True


def test_ppeak_and_vte_alarms_cover_lowered_pmax_loose_mask_disconnect_and_mode_switch(
    engine_factory, advance_time
):
    """Pressure-limited VCV alarms at lowered p_max and loose mask, empty-bellows PCV-VG stays silent,
    AirwayType.NONE disconnect fires VTe, and PCV->VCV mode switch does not false-alarm VTe."""
    low_pmax = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    low_pmax.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5.0, ie="1:2", p_max=30.0)
    low_pmax.set_airway_obstruction(1.0)
    advance_time(low_pmax, 10.0, dt=0.2)
    assert low_pmax.state.alarms.get("Ppeak", {}).get("high") is True

    loose = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    loose.set_airway_mode("Mask")
    loose.set_mask_seal(0.5)
    loose.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5.0, ie="1:2", p_max=40.0)
    loose.set_airway_obstruction(1.0)
    advance_time(loose, 10.0, dt=0.2)
    assert loose.state.alarms.get("Ppeak", {}).get("high") is True

    empty_vg = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    empty_vg.set_vent_settings(mode="PCV-VG", rr=12, vt=0.5, peep=5.0, ie="1:2", p_max=40.0)
    empty_vg.set_fgf(0.0, 0.0, 0.0)
    empty_vg.circuit.reservoir_volume = 0.0
    advance_time(empty_vg, 15.0, dt=0.5)
    assert "Ppeak" not in empty_vg.state.alarms

    disc = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    disc.set_airway_mode("None")
    advance_time(disc, 12.0, dt=0.5)
    assert disc.state.alarms.get("VTe", {}).get("low") is True

    switched = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    switched.set_vent_settings(mode="PCV", rr=12, vt=0.5, p_insp=14.0, peep=5.0, ie="1:2")
    advance_time(switched, 12.0, dt=0.5)
    switched.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5.0, ie="1:2")
    advance_time(switched, 12.0, dt=0.5)
    assert "VTe" not in switched.state.alarms


def test_separate_pipeline_and_cylinder_supply_alarms(engine_factory):
    """Pipeline, backup cylinder, and total O2 loss raise separate alarms."""
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    engine.step(0.5)
    assert "O2Pipeline" not in engine.state.alarms
    assert "O2Cylinder" not in engine.state.alarms
    assert "AirPipeline" not in engine.state.alarms

    # Wall O2 failure with backup E-cylinder open clears O2Supply while keeping O2Pipeline active.
    engine.set_pipeline_connected("o2", False)
    engine.step(0.5)
    assert engine.state.alarms.get("O2Supply", {}).get("low") is True
    assert engine.state.alarms.get("O2Pipeline", {}).get("low") is True
    assert "O2Cylinder" not in engine.state.alarms

    engine.set_cylinder_valve("o2", True)
    engine.step(0.5)
    assert "O2Supply" not in engine.state.alarms
    assert engine.state.alarms.get("O2Pipeline", {}).get("low") is True
    assert "O2Cylinder" not in engine.state.alarms

    # Low open backup cylinder (< 500 psig) raises O2Cylinder without yet triggering O2Supply.
    engine.replace_cylinder("o2", 0.20)
    engine.step(0.5)
    assert engine.state.o2_cylinder_psig < 500.0
    assert engine.state.alarms.get("O2Cylinder", {}).get("low") is True
    assert "O2Supply" not in engine.state.alarms

    # Disconnecting Air and N2O pipelines raises their respective pipeline alarms.
    engine.set_pipeline_connected("air", False)
    engine.set_pipeline_connected("n2o", False)
    engine.step(0.5)
    assert engine.state.alarms.get("AirPipeline", {}).get("low") is True
    assert engine.state.alarms.get("N2OPipeline", {}).get("low") is True
