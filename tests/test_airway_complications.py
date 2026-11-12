import pytest

from anasim.core.state import SimulationConfig


@pytest.mark.parametrize("enabled", [True, False])
def test_intubation_stimulus_triggers_laryngospasm_unless_disabled(engine_factory, enabled):
    engine = engine_factory(start=True)
    engine.set_airway_mode("Mask")
    engine.set_auto_laryngospasm(enabled)
    engine.start_disturbance("stim_intubation_pulse")

    for _ in range(10):
        engine.step(0.2)

    if not enabled:
        assert engine.state.laryngospasm < 0.01
        return
    assert engine.state.laryngospasm > 0.1
    assert engine.state.airway_obstruction >= engine.state.laryngospasm

    # Laryngoscopy ends at 40 s and the spasm resolves while the pressor
    # response continues.
    for _ in range(190):
        engine.step(0.2)
    assert engine.disturbance_active
    severity_at_end = engine.state.laryngospasm
    for _ in range(100):
        engine.step(0.2)
    assert engine.state.laryngospasm < severity_at_end * 0.15


def test_mask_obstruction_raises_airway_pressure_and_leaks_delivered_breaths(awake_engine):
    """The ventilator still pushes set VT, so Ppeak rises; gas that cannot enter
    the lungs leaks at the mask, so exhaled VT and MV fall together."""
    engine = awake_engine
    engine.set_airway_mode("Mask")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)

    # Suppress spontaneous drive to isolate assisted ventilation.
    dose = 0.8 * engine.patient.weight
    engine.give_drug_bolus("Rocuronium", dose)
    for _ in range(120):
        engine.step(0.5)

    baseline = engine.get_latest_state()

    engine.set_airway_obstruction(0.8)
    for _ in range(120):
        engine.step(0.5)

    obstructed = engine.state
    assert obstructed.paw_peak > baseline.paw_peak + 5.0
    assert obstructed.vt < baseline.vt * 0.5
    assert obstructed.mv < baseline.mv * 0.5


def test_loss_of_consciousness_collapses_unsupported_airway(engine_factory):
    engine = engine_factory(config=SimulationConfig(tci_enabled=True), start=True)
    # Oxygen without positive pressure keeps hypoxemia from adding cerebral depression.
    engine.set_airway_mode("Mask")
    engine.set_fgf(10.0, 0.0)
    # Light unconsciousness retains respiratory effort for the assisted-breath
    # comparison below; deeper exposure can produce apnea.
    engine.enable_tci("propofol", 2.5)
    for _ in range(3000):
        engine.step(0.1)
    assert engine.state.loc > 0.5
    assert engine.state.airway_obstruction > 0.2
    unsupported_obstruction = engine.state.airway_obstruction

    engine.set_airway_mode("Mask")
    engine.set_vent_settings(rr=0.0, vt=0.0, peep=0.0, ie="1:2", mode="CPAP")
    engine.set_vent_power(True)
    engine.step(0.1)
    assert engine.state.airway_obstruction > 0.2

    engine.set_vent_settings(rr=0.0, vt=0.0, peep=5.0, ie="1:2", mode="CPAP")
    engine.step(0.1)
    assert engine.state.airway_obstruction == 0.0

    # Bagging over spontaneous breaths splints by mean pressure across both,
    # so obstruction does not switch with the type of the last breath.
    engine.set_vent_power(False)
    # Slow assisted breaths allow spontaneous effort rather than washing CO2
    # below the anesthetized apneic threshold.
    engine.set_bag_mask_ventilation(True, rr=6.0, vt=0.5)
    for _ in range(900):
        engine.step(0.1)
    assert not engine.resp.state.apnea
    obstruction = []
    for _ in range(200):
        engine.step(0.1)
        obstruction.append(engine.state.airway_obstruction)
    # Switching would alternate between full relief and unsupported collapse (> 0.2).
    assert 0.05 < min(obstruction) <= max(obstruction) < 0.8 * unsupported_obstruction


def test_residual_block_increases_sedated_airway_collapse_but_preserves_awake_patency(engine_factory):
    """Herbstreit 2009: resting awake ventilation can persist at TOF 0.8,
    while the weakened pharynx is more susceptible to a closing challenge.
    """
    collapse = {}
    for sedated in (False, True):
        for blocked in (False, True):
            engine = engine_factory(config=SimulationConfig(mode="awake", tci_enabled=True), start=True)
            engine.set_continuous_fluid_rate(0.0)
            engine.set_airway_mode("Mask")
            if sedated:
                engine.enable_tci("propofol", 2.5)
            for _ in range(600):
                engine.step(1.0)
            if blocked:
                engine.give_drug_bolus("rocuronium", 0.2 * engine.patient.weight)
            for _ in range(240):
                engine.step(1.0)
            if blocked:
                assert 40.0 < engine.state.tof < 85.0
            collapse[sedated, blocked] = engine.state.airway_obstruction
            if sedated and blocked:
                engine.set_vent_settings(mode="CPAP", rr=0, vt=0, peep=5.0, ie="1:2")
                engine.set_vent_power(True)
                engine.step(1.0)
                assert engine.state.airway_obstruction == 0.0
                engine.set_airway_mode("ETT")
                engine.set_vent_power(False)
                engine.step(1.0)
                assert engine.state.airway_obstruction == 0.0
    assert collapse[False, False] == collapse[False, True] == 0.0
    assert collapse[True, True] > 1.4 * collapse[True, False] > 0.05


def test_near_recovered_tof_retains_pharyngeal_vulnerability(engine_factory, advance_time):
    """Herbstreit 2009: closing-pressure reserve remains 16% below baseline
    at normalized TOF 0.98. Transfer to sedated collapse is a calibration.
    """
    engines = [
        engine_factory(config=SimulationConfig(mode="awake", tci_enabled=True), start=True)
        for _ in range(2)
    ]
    for engine in engines:
        engine.set_continuous_fluid_rate(0.0)
        engine.set_airway_mode("Mask")
        # Maintain gas exchange while block recovers. Prolonged unsupported
        # sedation would add CO2 narcosis to the pharyngeal comparison.
        engine.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5, ie="1:2")
        engine.set_vent_power(True)
        engine.enable_tci("propofol", 2.5)
        advance_time(engine, 600)
    control, blocked = engines
    blocked.give_drug_bolus("rocuronium", 0.2 * blocked.patient.weight)
    for second in range(1, 3601):
        for engine in engines:
            engine.step(1.0)
        if second >= 240 and blocked.state.tof >= 98.0:
            break
    assert 98.0 <= blocked.state.tof < 98.1
    for engine in engines:
        assert engine.cerebral.co2_narcosis < 0.01
        engine.set_vent_power(False)
        engine.step(0.1)
    # A recovered thumb response still understates the airway's vulnerability.
    ratio = blocked.state.airway_obstruction / control.state.airway_obstruction
    assert 1.15 < ratio < 1.25


def test_obstructed_airway_preserves_effort_without_restoring_ventilation(awake_engine):
    engine = awake_engine
    engine.resp.hcvr_slope_baseline = 0.0
    engine.set_airway_mode("ETT")
    for _ in range(200):
        engine.step(0.1)
    baseline_effort = engine.resp_mech.effort.amplitude
    baseline_co2 = engine.resp.state.pa_co2

    engine.set_airway_obstruction(1.0)
    for _ in range(600):
        engine.step(0.1)
    resp, effort = engine.resp.state, engine.resp_mech.effort
    assert resp.drive_central > 0.0 and resp.muscle_factor == 1.0
    assert effort.rr > 8.0 and effort.target_vt > 0.4
    # Hypoxic drive raises effort while the airway remains obstructed.
    assert effort.amplitude > baseline_effort * 1.1
    assert engine.state.mv < 0.01 and resp.va < 0.01
    assert resp.pa_co2 > baseline_co2 + 5.0

    co2_at_one_minute, saturation = resp.pa_co2, resp.sao2
    stored_carbon = engine.resp.gas_transport.co2_l
    for _ in range(600):
        engine.step(0.1)
    assert resp.sao2 < saturation - 20.0
    # Desaturation changes CO2 binding, while total carbon continues to accumulate.
    assert resp.pa_co2 > co2_at_one_minute - 0.5
    assert engine.resp.gas_transport.co2_l > stored_carbon

    engine.set_airway_obstruction(0.0)
    for _ in range(300):
        engine.step(0.1)
    assert engine.state.vt > 400.0 and engine.resp.state.va > 3.0

    engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
    for _ in range(1200):
        engine.step(0.1)
    assert engine.resp_mech.effort.amplitude == 0.0


def test_mask_seal_scales_ventilation_splinting_preoxygenation_and_capnography(engine_factory):
    """Joffe 2010: at 15 cmH2O facemask PCV (PEEP 0, RR 15, I:E 1:1) in apneic
    nonparalyzed adults, a two-handed hold delivered 8.60 +/- 2.31 mL/kg PBW and
    a one-handed hold 6.80 +/- 3.10 mL/kg PBW (79%). Below 4 mL/kg PBW,
    ventilation is inadequate.
    """
    volumes = {}
    obstructions = {}
    gases = {}
    for seal in (1.0, 0.8, 0.4, 0.0):
        engine = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="tiva", dt=0.1), start=True)
        engine.give_drug_bolus("sugammadex", 4.0 * engine.patient.weight)
        engine.set_airway_mode("Mask")
        engine.set_mask_seal(seal)
        engine.set_vent_settings(rr=15, vt=0.5, peep=0.0, ie="1:1", mode="PCV", p_insp=15.0)
        engine.set_vent_power(True)
        for _ in range(300):
            engine.step(0.1)
        pbw = engine.patient.predicted_body_weight()
        volumes[seal] = engine.state.vt / pbw
        obstructions[seal] = engine.state.airway_obstruction
        gases[seal] = (engine.state.fio2, engine.state.display_etco2)
    assert 6.3 < volumes[1.0] < 10.9
    assert volumes[0.8] == pytest.approx(0.8 * volumes[1.0], rel=0.03)
    assert 1.0 < volumes[0.4] < 4.0
    assert volumes[0.0] == 0.0
    assert obstructions[1.0] == 0.0
    assert 0.1 < obstructions[0.4] < obstructions[0.0]
    # Pressurized breaths leak outward, so only circuit gas is inspired. Too
    # little exhaled gas reaches the sampling port to clear the dead space.
    assert gases[0.4][0] == pytest.approx(gases[1.0][0])
    assert gases[0.8][1] > gases[1.0][1] - 2.0
    assert gases[0.4][1] < 5.0

    # Awake breathing around a loose mask preserves room-air CO2 clearance
    # and the capnogram, and stalls oxygen wash-in until the seal is tightened.
    awake = engine_factory(config=SimulationConfig(mode="awake", dt=0.1), start=True)
    awake.set_airway_mode("Mask")
    awake.set_fgf(10.0, 0.0, 0.0)
    awake.set_mask_seal(0.4)
    for _ in range(1800):
        awake.step(0.1)
    assert 35.0 < awake.state.et_o2 < 55.0
    assert 37.0 < awake.state.pa_co2 < 43.0
    loose_etco2, loose_fio2 = awake.state.display_etco2, awake.state.fio2
    # Bag breaths pressurize the mask, so only the breaths between them draw room air.
    awake.set_bag_mask_ventilation(True, rr=6, vt=0.5)
    for _ in range(600):
        awake.step(0.1)
    assert loose_fio2 + 0.05 < awake.state.fio2 < 0.9
    awake.set_bag_mask_ventilation(False)
    awake.set_mask_seal(1.0)
    for _ in range(1800):
        awake.step(0.1)
    assert awake.state.et_o2 >= 90.0
    assert loose_etco2 == pytest.approx(awake.state.display_etco2, abs=1.0)

    # At 0% mask seal, no gas crosses the Y-piece and volume-targeted modes alarm.
    awake.set_mask_seal(0.0)
    awake.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    awake.set_vent_power(True)
    for _ in range(200):
        awake.step(0.1)
    assert (awake.state.paw, awake.state.flow, awake.state.volume) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert awake.state.alarms["VTe"]["low"] and awake.state.alarms["MV"]["low"]

    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            awake.set_mask_seal(bad)
    assert awake.mask_seal == 0.0


def test_high_airway_pressure_opens_facemask_leak_above_seal_threshold(engine_factory):
    """Bouvet 2014: facemask pressure-controlled ventilation at 20 and 25 cmH2O,
    but not at 10 or 15 cmH2O, distended the gastric antrum. Pressures above
    20 cmH2O leak even with a tight seal.
    """
    outcomes = {}
    for bronchospasm in (0.0, 1.0):
        for airway in ("ETT", "Mask"):
            engine = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.2), start=True)
            engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
            engine.set_airway_mode(airway)
            engine.set_bronchospasm(bronchospasm)
            engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV", p_max=50.0)
            engine.set_vent_power(True)
            for _ in range(900):
                engine.step(0.2)
            outcomes[bronchospasm, airway] = (engine.state.paw_peak, engine.state.vt, engine.state.pa_co2)

    assert outcomes[0.0, "Mask"][0] < 20.0
    assert outcomes[0.0, "Mask"][1] == pytest.approx(outcomes[0.0, "ETT"][1], rel=0.01)
    assert outcomes[1.0, "Mask"][0] > 25.0
    assert outcomes[1.0, "Mask"][1] < 0.9 * outcomes[1.0, "ETT"][1]
    assert outcomes[1.0, "Mask"][2] > outcomes[1.0, "ETT"][2] + 0.8
