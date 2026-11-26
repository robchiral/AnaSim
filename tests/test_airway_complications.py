import copy

import pytest

from anasim.core.state import AirwayType, SimulationConfig


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
    engine.set_vent_settings(rr=12, vt=0.6, peep=5.0, ie="1:2", mode="VCV", p_max=45.0)
    engine.set_vent_power(True)

    # Suppress spontaneous drive to isolate assisted ventilation.
    dose = 0.8 * engine.patient.weight
    engine.give_drug_bolus("Rocuronium", dose)
    for _ in range(60):
        engine.step(1.0)

    baseline = engine.get_latest_state()

    engine.set_airway_obstruction(0.85)
    for _ in range(50):
        engine.step(1.0)
    volumes = []
    for _ in range(100):
        engine.step(0.1)
        volumes.append(engine.state.volume)

    obstructed = engine.state
    assert obstructed.paw_peak > baseline.paw_peak + 5.0
    assert obstructed.vt < baseline.vt * 0.5
    assert obstructed.mv < baseline.mv * 0.5
    assert (max(volumes) - min(volumes)) * 1000.0 < 2.0 * obstructed.vt


def test_loss_of_consciousness_collapses_unsupported_airway(engine_factory):
    engine = engine_factory(config=SimulationConfig(tci_enabled=True), start=True)
    # Oxygen without positive pressure keeps hypoxemia from adding cerebral depression.
    engine.set_airway_mode("Mask")
    engine.set_fgf(10.0, 0.0)
    # Light unconsciousness retains respiratory effort for the assisted-breath
    # comparison below; deeper exposure can produce apnea.
    engine.enable_tci("propofol", 2.5)
    for _ in range(300):
        engine.step(1.0)
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

    # When low-APL bag breaths alternate with spontaneous ones, the mean pressure
    # across both partially holds the airway open without switching with the type
    # of the last breath. Raising APL to 20 cmH2O adds baseline CPAP/PEEP and
    # fully relieves pharyngeal collapse.
    engine.set_vent_power(False)
    engine.set_fgf(4.0, 0.0)
    # Slow assisted breaths at low APL allow spontaneous effort without continuous CPAP.
    engine.set_bag_mask_ventilation(True, rr=6.0, vt=0.5, p_max=10.0)
    for _ in range(180):
        engine.step(0.5)
    assert not engine.resp.state.apnea
    obstruction = []
    for _ in range(200):
        engine.step(0.1)
        obstruction.append(engine.state.airway_obstruction)
    # Switching would alternate between full relief and unsupported collapse (> 0.2).
    assert 0.05 < min(obstruction) <= max(obstruction) < 0.85 * unsupported_obstruction

    engine.set_fgf(10.0, 0.0)
    engine.set_apl_valve(20.0)
    for _ in range(60):
        engine.step(0.5)
    assert engine.state.peep >= 5.0
    assert engine.state.airway_obstruction == 0.0


def test_residual_and_near_recovered_block_increase_sedated_collapse_while_preserving_awake_patency(
    engine_factory,
):
    """Herbstreit 2009: resting awake ventilation persists at TOF 0.8 while sedated
    collapse increases; at normalized TOF 0.98 closing-pressure reserve remains 16%
    below baseline.
    """
    awake = engine_factory(config=SimulationConfig(mode="awake", tci_enabled=True), start=True)
    awake.set_continuous_fluid_rate(0.0)
    awake.set_airway_mode("Mask")
    assert awake.state.airway_obstruction == 0.0
    awake.give_drug_bolus("rocuronium", 0.2 * awake.patient.weight)
    for _ in range(120):
        awake.step(2.0)
    assert 40.0 < awake.state.tof < 85.0
    assert awake.state.airway_obstruction == 0.0

    control = engine_factory(config=SimulationConfig(mode="awake", tci_enabled=True), start=True)
    control.set_continuous_fluid_rate(0.0)
    control.set_airway_mode("Mask")
    # Maintain gas exchange while block recovers so CO2 narcosis does not confound collapse.
    control.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5, ie="1:2")
    control.set_vent_power(True)
    control.enable_tci("propofol", 2.5)
    for _ in range(300):
        control.step(2.0)
    blocked = copy.deepcopy(control)
    blocked.give_drug_bolus("rocuronium", 0.2 * blocked.patient.weight)
    for _ in range(120):
        control.step(2.0)
        blocked.step(2.0)

    c_mid, b_mid = copy.deepcopy(control), copy.deepcopy(blocked)
    for m in (c_mid, b_mid):
        m.set_vent_power(False)
        m.step(0.1)
    assert 40.0 < b_mid.state.tof < 85.0
    assert b_mid.state.airway_obstruction > 1.4 * c_mid.state.airway_obstruction > 0.05
    b_mid.set_vent_settings(mode="CPAP", rr=0, vt=0, peep=5.0, ie="1:2")
    b_mid.set_vent_power(True)
    b_mid.step(1.0)
    assert b_mid.state.airway_obstruction == 0.0
    b_mid.set_airway_mode("ETT")
    b_mid.set_vent_power(False)
    b_mid.step(1.0)
    assert b_mid.state.airway_obstruction == 0.0

    elapsed = 240.0
    while elapsed < 3600.0:
        dt = 1.0 if blocked.state.tof >= 97.5 else 2.0
        control.step(dt)
        blocked.step(dt)
        elapsed += dt
        if blocked.state.tof >= 98.0:
            break
    assert 98.0 <= blocked.state.tof < 98.1
    for engine in (control, blocked):
        assert engine.cerebral.co2_narcosis < 0.01
        engine.set_vent_power(False)
        engine.step(0.1)
    ratio = blocked.state.airway_obstruction / control.state.airway_obstruction
    assert 1.15 < ratio < 1.25


def test_obstructed_airway_preserves_effort_without_drawing_from_circuit(awake_engine):
    engine = awake_engine
    engine.resp.hcvr_slope_baseline = 0.0
    engine.set_airway_mode("ETT")
    engine.set_fgf(0.0, 6.0)
    for _ in range(40):
        engine.step(0.5)
    baseline_effort = engine.resp_mech.effort.amplitude
    baseline_co2 = engine.resp.state.pa_co2

    engine.set_fgf(0.0, 0.0)
    engine.circuit.composition.fio2 = 0.5
    engine.circuit.composition.fin2 = 0.5
    engine.set_airway_obstruction(1.0)
    for _ in range(120):
        engine.step(0.5)
    resp, effort = engine.resp.state, engine.resp_mech.effort
    assert resp.drive_central > 0.0 and resp.muscle_factor == 1.0
    assert effort.rr > 8.0 and effort.target_vt > 0.4
    # Hypoxic drive raises effort while the closed circuit remains untouched.
    assert effort.amplitude > baseline_effort * 1.1
    assert engine.state.mv < 0.01 and resp.va < 0.01
    assert engine.circuit.composition.fio2 == pytest.approx(0.5, abs=1e-8)
    assert resp.pa_co2 > baseline_co2 + 5.0

    co2_at_one_minute, saturation = resp.pa_co2, resp.sao2
    stored_carbon = engine.resp.gas_transport.co2_l
    for _ in range(120):
        engine.step(0.5)
    assert resp.sao2 < saturation - 20.0
    # Desaturation changes CO2 binding, while total carbon continues to accumulate.
    assert resp.pa_co2 > co2_at_one_minute - 0.5
    assert engine.resp.gas_transport.co2_l > stored_carbon

    engine.set_airway_obstruction(0.0)
    for _ in range(60):
        engine.step(0.5)
    assert engine.state.vt > 400.0 and engine.resp.state.va > 3.0
    assert engine.circuit.composition.fio2 < 0.48


def test_mask_seal_scales_ventilation_splinting_preoxygenation_and_capnography(engine_factory):
    """Joffe 2010: at 15 cmH2O facemask PCV (PEEP 0, RR 15, I:E 1:1) in apneic
    nonparalyzed adults, a two-handed hold delivered 8.60 +/- 2.31 mL/kg PBW and
    a one-handed hold 6.80 +/- 3.10 mL/kg PBW (79%). Below 4 mL/kg PBW,
    ventilation is inadequate.
    """
    base = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="tiva", dt=0.5), start=True)
    base.give_drug_bolus("sugammadex", 4.0 * base.patient.weight)
    base.set_airway_mode("Mask")
    base.set_vent_settings(rr=15, vt=0.5, peep=0.0, ie="1:1", mode="PCV", p_insp=15.0)
    base.set_vent_power(True)

    volumes = {}
    obstructions = {}
    gases = {}
    pressures = {}
    for seal in (1.0, 0.8, 0.4, 0.2, 0.0):
        engine = copy.deepcopy(base)
        engine.set_mask_seal(seal)
        for _ in range(60):
            engine.step(0.5)
        pbw = engine.patient.predicted_body_weight()
        volumes[seal] = engine.state.vt / pbw
        obstructions[seal] = engine.state.airway_obstruction
        gases[seal] = (engine.state.fio2, engine.state.display_etco2)
        pressures[seal] = engine.state.paw_peak
        assert engine.aeration.pressure_transmission == pytest.approx(seal)
        if seal == 0.4:
            engine.set_fgf(0.3, 0.0)
            for _ in range(60):
                engine.step(0.5)
            assert engine.circuit.reservoir_volume < 1.5
    assert 6.3 < volumes[1.0] < 10.9
    assert volumes[0.8] == pytest.approx(0.8 * volumes[1.0], rel=0.03)
    assert 1.0 < volumes[0.4] < 4.0
    assert volumes[0.0] == 0.0
    assert pressures[0.4] < 0.5 * pressures[1.0]
    assert obstructions[1.0] == 0.0
    assert 0.1 < obstructions[0.4] < obstructions[0.0]
    # Pressurized breaths leak outward, so only circuit gas is inspired. With a
    # very loose seal, too little exhaled gas reaches the port to clear dead space.
    assert gases[0.4][0] == pytest.approx(gases[1.0][0])
    assert gases[0.8][1] > gases[1.0][1] - 2.0
    assert gases[0.2][1] < 5.0

    # Awake breathing around a loose mask preserves room-air CO2 clearance
    # and the capnogram, and stalls oxygen wash-in until the seal is tightened.
    awake = engine_factory(config=SimulationConfig(mode="awake", dt=1.0), start=True)
    awake.set_airway_mode("Mask")
    awake.set_fgf(10.0, 0.0, 0.0)
    awake.set_mask_seal(0.4)
    for _ in range(180):
        awake.step(1.0)
    assert 35.0 < awake.state.et_o2 < 55.0
    assert 37.0 < awake.state.pa_co2 < 43.0
    loose_etco2, loose_fio2 = awake.state.display_etco2, awake.state.fio2
    # Bag breaths pressurize the mask, so only the breaths between them draw room air.
    awake.set_bag_mask_ventilation(True, rr=6, vt=0.5, p_max=20.0)
    for _ in range(60):
        awake.step(1.0)
    assert loose_fio2 + 0.05 < awake.state.fio2 < 0.9
    awake.set_bag_mask_ventilation(False)
    awake.set_apl_valve(0.0)
    awake.set_mask_seal(1.0)
    for _ in range(180):
        awake.step(1.0)
    assert awake.state.et_o2 >= 90.0
    assert loose_etco2 == pytest.approx(awake.state.display_etco2, abs=1.0)

    # At 0% mask seal, no gas crosses the Y-piece and volume-targeted modes alarm.
    awake.set_mask_seal(0.0)
    awake.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    awake.set_vent_power(True)
    for _ in range(40):
        awake.step(0.5)
    assert (awake.state.paw, awake.state.flow, awake.state.volume) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert awake.state.alarms["VTe"]["low"] and awake.state.alarms["MV"]["low"]

    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            awake.set_mask_seal(bad)
    assert awake.mask_seal == 0.0


def test_high_airway_pressure_opens_facemask_leak_above_seal_threshold(engine_factory):
    """Bouvet 2014: facemask pressure-controlled ventilation at 20 and 25 cmH2O,
    but not at 10 or 15 cmH2O, distended the gastric antrum. AnaSim opens a
    mask leak above 20 cmH2O even with a tight seal.
    """
    base = engine_factory(config=SimulationConfig(mode="steady_state", dt=1.0), start=True)
    base.give_drug_bolus("roc", 0.8 * base.patient.weight)
    base.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV", p_max=50.0)
    base.set_vent_power(True)
    outcomes = {}
    for bronchospasm in (0.0, 1.0):
        for airway in ("ETT", "Mask"):
            engine = copy.deepcopy(base)
            engine.set_airway_mode(airway)
            engine.set_bronchospasm(bronchospasm)
            for _ in range(180):
                engine.step(1.0)
            outcomes[bronchospasm, airway] = (engine.state.paw_peak, engine.state.vt, engine.state.pa_co2)

    assert outcomes[0.0, "Mask"][0] < 20.0
    assert outcomes[0.0, "Mask"][1] == pytest.approx(outcomes[0.0, "ETT"][1], rel=0.01)
    assert outcomes[1.0, "Mask"][0] > 25.0
    assert outcomes[1.0, "Mask"][1] < 0.9 * outcomes[1.0, "ETT"][1]
    assert outcomes[1.0, "Mask"][2] > outcomes[1.0, "ETT"][2] + 0.8


def test_stage_ii_extubation_laryngospasm_and_bronchodilation(engine_factory, advance_time):
    """Extubation at Stage II hypnosis provokes laryngospasm; epinephrine (blocked by
    non-selective beta blockade) and ketamine relieve bronchospasm without mask double-counting.
    """
    extub = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    extub.give_drug_bolus("propofol", 110.0)
    advance_time(extub, 60.0, dt=1.0)
    extub.state.airway_mode = AirwayType.ETT
    extub.set_airway_mode("Mask")
    advance_time(extub, 15.0, dt=0.5)
    assert extub.laryngospasm_severity > 0.20

    epi_free = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    mask_bronch = copy.deepcopy(epi_free)
    mask_bronch.set_airway_mode("Mask")
    mask_bronch.set_fgf(8.0, 0.0)
    epi_blocked = copy.deepcopy(epi_free)
    epi_blocked.give_drug_bolus("labetalol", 40.0)
    for model in (epi_free, epi_blocked, mask_bronch):
        model.set_bronchospasm(0.8)
        advance_time(model, 60.0, dt=1.0)
    assert mask_bronch.state.va > 0.65 * epi_free.state.va
    ket_bronch = copy.deepcopy(epi_free)
    before_ket = ket_bronch.state.bronchospasm
    for model in (epi_free, epi_blocked):
        model.give_drug_bolus("epi", 20.0)
        advance_time(model, 30.0, dt=0.5)
    assert epi_blocked.state.bronchospasm > epi_free.state.bronchospasm + 0.10

    ket_bronch.give_drug_bolus("ketamine", 100.0)
    advance_time(ket_bronch, 60.0, dt=1.0)
    assert ket_bronch.state.bronchospasm < before_ket - 0.10


def test_partial_airway_patency_and_flow_limited_reopen_conserve_dry_gas_pressure(
    engine_factory, advance_time
):
    """Partial airway obstruction and flow-limited reopening conserve dry alveolar pressure and volatile agent mass."""
    engine = engine_factory(config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1), start=True)
    resp = engine.resp

    def _dry_pressure() -> float:
        return (
            resp.state.p_alveolar_o2
            + resp.state.p_alveolar_co2
            + resp.alveolar_fn2 * resp._atm_dry
            + (engine.pk_sevo.state.p_alv + engine.pk_n2o.state.p_alv) * resp._atm_dry
        )

    initial_dry = _dry_pressure()
    engine.give_drug_bolus("roc", 70.0)
    engine.set_vent_power(False)
    engine.set_fgf(8.0, 0.0, 0.0)
    engine.set_airway_obstruction(0.95)
    advance_time(engine, 90.0, dt=1.0)
    assert _dry_pressure() == pytest.approx(initial_dry, abs=2.5)

    engine.set_airway_obstruction(1.0)
    advance_time(engine, 120.0, dt=1.0)
    deficit_closed = resp._apneic_gas_deficit_l
    assert deficit_closed > 0.05
    engine.set_airway_obstruction(0.6)
    advance_time(engine, 30.0, dt=1.0)
    assert resp._apneic_gas_deficit_l < 0.2 * deficit_closed
    assert _dry_pressure() == pytest.approx(initial_dry, abs=6.0)

    sevo = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=1.0, rng_seed=1), start=True
    )
    sevo.give_drug_bolus("roc", 70.0)
    sevo.set_vent_power(False)
    sevo.set_fgf(0.0, 0.0, 0.0)
    sevo.circuit.reservoir_volume = 1.5

    def _total_sevo_stpd() -> float:
        pk = sevo.pk_sevo
        stpd = sevo.resp._btps_to_stpd
        circuit_l = (3.0 + sevo.circuit.reservoir_volume) * sevo.circuit.composition.fi_agent
        alv_l = pk.frc * pk.state.p_alv * stpd
        tissue_l = (
            pk.lambda_b_g
            * (
                pk.v_vrg * pk.lambda_t_b_vrg * pk.state.p_vrg
                + pk.v_mus * pk.lambda_t_b_mus * pk.state.p_mus
                + pk.v_fat * pk.lambda_t_b_fat * pk.state.p_fat
            )
            * stpd
        )
        return circuit_l + alv_l + tissue_l

    sevo_0 = _total_sevo_stpd()
    sevo.set_airway_obstruction(1.0)
    advance_time(sevo, 90.0, dt=1.0)
    assert _total_sevo_stpd() == pytest.approx(sevo_0, abs=1e-3)
    sevo.set_airway_obstruction(0.6)
    advance_time(sevo, 30.0, dt=1.0)
    assert _total_sevo_stpd() == pytest.approx(sevo_0, abs=1e-3)


def test_psv_bronchospasm_reduces_va_and_mask_leak_entrainment_scales_smoothly(engine_factory, advance_time):
    """Bronchospasm lowers alveolar ventilation during PSV, and a 99% facemask seal does not
    bypass empty-reservoir starvation via a discontinuous room-air leak cliff."""
    psv_a = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    psv_a.set_airway_mode("ETT")
    psv_a.set_fgf(6.0, 0.0)
    psv_a.set_vent_settings(mode="PSV", rr=0, vt=0.0, peep=5.0, p_support=12.0, ie="1:2")
    psv_a.set_vent_power(True)
    advance_time(psv_a, 30.0, dt=1.0)
    psv_b = copy.deepcopy(psv_a)
    psv_b.set_bronchospasm(0.8)
    advance_time(psv_a, 30.0, dt=1.0)
    advance_time(psv_b, 30.0, dt=1.0)
    assert psv_b.state.va < 0.75 * psv_a.state.va

    starved = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    starved.set_airway_mode("Mask")
    starved.set_fgf(0.0, 0.0)
    starved.circuit.reservoir_volume = 0.0
    starved.set_mask_seal(0.99)
    advance_time(starved, 10.0, dt=0.5)
    assert starved.state.va < 0.5


def test_succinylcholine_preserves_apneic_oxygenation_and_severe_awake_block_obstructs(
    engine_factory, advance_time
):
    """Succinylcholine does not seal the airway airtight; severe awake TOF < 10% causes obstruction."""
    sux = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True, weight=115.0, height=170.0,
    )
    sux.set_airway_mode("Mask")
    sux.set_fgf(10.0, 0.0)
    advance_time(sux, 180.0, dt=1.0)
    sux.give_drug_bolus("propofol", 200.0)
    sux.give_drug_bolus("succinylcholine", 120.0)
    advance_time(sux, 180.0, dt=1.0)
    assert sux.airway_status.patency >= 0.10
    assert sux.state.sao2 > 95.0

    awake_roc = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    awake_roc.set_airway_mode("Mask")
    awake_roc.give_drug_bolus("rocuronium", 25.0)
    advance_time(awake_roc, 240.0, dt=1.0)
    assert awake_roc.state.tof < 10.0
    assert awake_roc.state.loc < 0.05
    assert awake_roc.state.airway_obstruction > 0.3


def test_succinylcholine_laryngospasm_relief_is_dose_responsive(engine_factory, advance_time):
    """Higher succinylcholine rescue doses should break laryngospasm faster via central NMJ."""
    times = {}
    for dose_mg in (10.0, 50.0):
        eng = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
        eng.give_drug_bolus("propofol", 100.0)
        advance_time(eng, 45.0, dt=1.0)
        eng.laryngospasm_severity = 0.95
        eng.give_drug_bolus("succinylcholine", dose_mg)
        relieved_t = None
        for t in range(1, 91):
            eng.step(1.0)
            if relieved_t is None and eng.laryngospasm_severity < 0.25:
                relieved_t = t
        times[dose_mg] = relieved_t
    assert times[50.0] is not None
    assert times[10.0] is None or times[50.0] < times[10.0]
