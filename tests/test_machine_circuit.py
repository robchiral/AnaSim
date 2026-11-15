import pytest

from anasim.core.state import SimulationConfig
from anasim.machine.circuit import CircleSystem
from anasim.machine.volatile import Vaporizer
from anasim.scenarios.oxygen_supply import create_oxygen_supply_failure


def test_low_flow_mixture_balances_uptake_spill_and_vaporizer_consumption():
    """Every supplied gas leaves through patient uptake or the spill valve."""
    circuit = CircleSystem()
    circuit.fgf_o2, circuit.fgf_air, circuit.fgf_n2o = 0.8, 0.6, 0.6
    vaporizer = Vaporizer()
    vaporizer.set_concentration(4.0)
    circuit.vaporizer_setting, circuit.vaporizer_on = 4.0, True
    carrier = circuit.fgf_total()
    vapor = carrier * 0.04 / 0.96

    circuit.equilibrate(uptake_o2=0.25, fi_agent=0.03)
    initial_spill = (carrier - 0.25) / 0.97
    gas = circuit.composition
    assert (gas.fio2, gas.fi_agent, gas.fin2o, gas.fin2) == pytest.approx((
        (0.8 + 0.21 * 0.6 - 0.25) / initial_spill,
        0.03,
        0.6 / initial_spill,
        0.79 * 0.6 / initial_spill,
    ))

    uptake = (0.25, 0.01, 0.05)
    spill = carrier + vapor - sum(uptake)
    expected = (
        (0.8 + 0.21 * 0.6 - uptake[0]) / spill,
        (vapor - uptake[1]) / spill,
        (0.6 - uptake[2]) / spill,
        0.79 * 0.6 / spill,
    )
    for _ in range(3600):
        circuit.step(1.0, *uptake)
    assert (gas.fio2, gas.fi_agent, gas.fin2o, gas.fin2) == pytest.approx(expected, abs=1e-7)

    initial_liquid = vaporizer.state.level
    vaporizer.step(60.0, carrier)
    assert initial_liquid - vaporizer.state.level == pytest.approx(vapor * 1000.0 / 180.0)

    # The last partial dose still enters the circuit when the reservoir empties.
    vaporizer.state.level = 0.01
    delivered = vaporizer.step(60.0, carrier) / 100.0
    assert carrier * delivered / (1.0 - delivered) == pytest.approx(0.01 * 0.18)


def test_closed_circuit_stays_stable_until_oxygen_supply_fails():
    circuit = CircleSystem()
    circuit.fgf_o2 = 0.25
    circuit.composition.fio2, circuit.composition.fin2 = 0.8, 0.2
    # Oxygen supply exactly replaces uptake; no gas spills.
    for _ in range(60):
        circuit.step(1.0, uptake_o2=0.25, uptake_agent=0.0)
    assert circuit.composition.fio2 == pytest.approx(0.8)

    circuit.oxygen_supply_connected = False
    for _ in range(60):
        circuit.step(1.0, uptake_o2=0.25, uptake_agent=0.0)
    assert 0.0 < circuit.composition.fio2 < 0.8


def test_obstructed_patient_uses_oxygen_stores_without_drawing_from_circuit(engine_factory):
    engine = engine_factory(start=True)
    engine.set_airway_mode("ETT")
    engine.set_fgf(0.0, 6.0)
    for _ in range(200):
        engine.step(0.1)
    assert engine.state.va > 3.0
    engine.set_fgf(0.0, 0.0)
    engine.circuit.composition.fio2 = 0.5
    engine.circuit.composition.fin2 = 0.5
    engine.set_airway_obstruction(1.0)
    initial_o2 = engine.resp.state.p_alveolar_o2
    initial_co2 = engine.state.pa_co2
    for _ in range(300):
        engine.step(0.1)
    assert engine.state.va == 0.0
    assert engine.circuit.composition.fio2 == pytest.approx(0.5, abs=1e-8)
    assert engine.resp.state.p_alveolar_o2 < initial_o2 - 10.0
    assert engine.state.pa_co2 > initial_co2 + 4.0

    engine.set_airway_obstruction(0.0)
    for _ in range(300):
        engine.step(0.1)
    assert engine.state.va > 3.0
    assert engine.circuit.composition.fio2 < 0.48


def test_oxygen_supply_failure_stops_o2_and_n2o_delivery():
    circuit = CircleSystem(volume_l=6.0)
    circuit.fgf_o2 = 2.0
    circuit.fgf_air = 8.0
    circuit.fgf_n2o = 4.0
    circuit.composition.fio2 = 0.37
    circuit.composition.fin2 = 0.63

    circuit.oxygen_supply_connected = False

    assert circuit.delivered_o2_flow() == 0.0
    assert circuit.delivered_n2o_flow() == 0.0
    assert circuit.fgf_total() == 8.0

    for _ in range(60):
        circuit.step(1.0, uptake_o2=0.25, uptake_agent=0.0)

    assert circuit.composition.fio2 < 0.30
    assert circuit.fgf_o2 == 2.0
    assert circuit.fgf_n2o == 4.0


def test_disabling_sevoflurane_preserves_nitrous_oxide_delivery(engine_factory):
    engine = engine_factory(config=SimulationConfig(sevoflurane_enabled=False), start=True)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(2.0, 0.0, 2.0)
    engine.set_vaporizer(4.0)
    liquid_level = engine.vaporizer.state.level
    for _ in range(120):
        engine.step(0.5)

    assert engine.state.fi_sevo == engine.state.mac_sevo == 0.0
    assert not engine.circuit.vaporizer_on
    assert engine.vaporizer.state.level == liquid_level
    assert engine.state.fi_n2o > 10.0
    assert engine.state.mac_n2o > 0.0


def test_engine_volatile_washin_washout(engine_factory):
    """Circuit + volatile PK integration: vaporizer raises MAC, washout lowers it."""
    config = SimulationConfig(mode="awake")
    engine = engine_factory(config=config, start=True)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(8.0, 0.0)

    engine.set_vaporizer(2.0)
    initial_level = engine.vaporizer.state.level
    for _ in range(300):  # 5 min wash-in
        engine.step(1.0)

    mac_on = engine.state.mac
    fi_sevo = engine.state.fi_sevo
    fio2 = engine.state.fio2

    assert fi_sevo > 1.0, f"FiSevo too low with vaporizer on: {fi_sevo:.2f}%"
    assert fio2 > 0.8, f"FiO2 did not track circuit wash-in: {fio2:.2f}"
    assert mac_on > 0.2, f"MAC did not rise with vaporizer on: {mac_on:.2f}"
    assert engine.vaporizer.state.level < initial_level

    # Running out of liquid stops delivery and lets the circuit and patient wash out.
    engine.vaporizer.state.level = 0.01
    engine.step(1.0)
    assert engine.vaporizer.state.level == 0.0
    assert not engine.vaporizer.state.is_on
    assert engine.vaporizer.state.setting == 0.0
    engine.set_fgf(10.0, 0.0)
    for _ in range(300):  # 5 min washout
        engine.step(1.0)

    assert engine.state.mac < mac_on * 0.7, "MAC should decrease with washout/high FGF"
    assert engine.state.fi_sevo < fi_sevo * 0.1
    assert engine.state.fio2 > 0.99


def test_engine_n2o_washin_washout(engine_factory):
    """N2O wash-in should raise FiN2O and MAC fraction; washout should reduce them."""
    config = SimulationConfig(mode="awake")
    engine = engine_factory(config=config, start=True)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(2.0, 0.0, n2o_l_min=4.0)

    for _ in range(300):  # 5 min wash-in
        engine.step(1.0)

    fi_n2o = engine.state.fi_n2o
    mac_n2o = engine.state.mac_n2o

    assert fi_n2o > 30.0, f"FiN2O too low with N2O on: {fi_n2o:.2f}%"
    assert mac_n2o > 0.2, f"N2O MAC fraction too low: {mac_n2o:.2f}"

    engine.set_fgf(6.0, 0.0, n2o_l_min=0.0)
    for _ in range(300):  # 5 min washout
        engine.step(1.0)

    assert engine.state.fi_n2o < fi_n2o * 0.4, "FiN2O should decrease with washout/high FGF"
    assert engine.state.mac_n2o < mac_n2o * 0.5, "N2O MAC should decrease with washout/high FGF"


def test_fio2_blender_with_n2o(engine_factory):
    """FiO2 blender should hit target even with fixed N2O flow."""
    config = SimulationConfig(mode="awake")
    engine = engine_factory(config=config, start=False)
    engine.set_fgf(1.0, 1.0, n2o_l_min=2.0)
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV", fio2=0.5)

    o2 = engine.circuit.fgf_o2
    air = engine.circuit.fgf_air
    n2o = engine.circuit.fgf_n2o
    total = o2 + air + n2o
    fio2 = (o2 + 0.21 * air) / total if total > 0 else 0.21
    assert abs(fio2 - 0.5) < 0.05

    engine.set_oxygen_supply_connected(False)
    engine.set_vent_settings(
        rr=12,
        vt=0.5,
        peep=5.0,
        ie="1:2",
        mode="VCV",
        fio2=0.3,
    )
    configured_total = (
        engine.circuit.fgf_o2
        + engine.circuit.fgf_air
        + engine.circuit.fgf_n2o
    )
    configured_fio2 = (
        engine.circuit.fgf_o2 + 0.21 * engine.circuit.fgf_air
    ) / configured_total
    assert configured_total == pytest.approx(total)
    assert configured_fio2 == pytest.approx(0.3)


def test_n2o_hypoxic_guard_step_invariance_and_low_flow_hypoxia(engine_factory, advance_time):
    """Circle enforces >=25% O2 with N2O, conserves O2 across dt, accumulates exhaled N2
    at metabolic flow, preserves vaporizer_on across zero FGF, and turns hypoxic on O2 loss.
    """
    c_fine, c_coarse = CircleSystem(volume_l=6.0), CircleSystem(volume_l=6.0)
    c_fine.fgf_o2, c_fine.fgf_n2o = 0.0, 8.0
    assert c_fine.delivered_n2o_flow() == 0.0
    c_fine.fgf_o2 = 1.0
    assert c_fine.delivered_n2o_flow() == pytest.approx(3.0)

    for c in (c_fine, c_coarse):
        c.fgf_o2 = c.fgf_n2o = 0.0
        c.composition.fio2, c.composition.fin2 = 0.5, 0.5
    for _ in range(600):
        c_fine.step(0.1, uptake_o2=0.25, uptake_agent=0.0)
    for _ in range(12):
        c_coarse.step(5.0, uptake_o2=0.25, uptake_agent=0.0)
    assert c_fine.composition.fio2 == pytest.approx(c_coarse.composition.fio2, abs=1e-9)
    assert c_fine.composition.fio2 * (3.0 + c_fine.reservoir_volume) == pytest.approx(2.75, abs=1e-9)

    low_flow = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    low_flow.circuit.composition.fio2, low_flow.circuit.composition.fin2 = 1.0, 0.0
    low_flow.set_airway_mode("ETT")
    low_flow.set_vent_power(True)
    low_flow.set_vaporizer(2.0)
    low_flow.set_fgf(0.0, 0.0)
    low_flow.step(1.0)
    assert low_flow.circuit.vaporizer_on is True
    low_flow.set_vaporizer(0.0)
    low_flow.set_fgf(0.3, 0.0)
    advance_time(low_flow, 180.0, dt=1.0)
    assert low_flow.circuit.composition.fin2 > 0.15

    ox = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    create_oxygen_supply_failure().prepare(ox)
    ox.set_oxygen_supply_connected(False)
    advance_time(ox, 800.0, dt=2.0)
    assert ox.circuit.composition.fio2 < 0.15
    assert ox.state.spo2 < 85.0

    empty = CircleSystem()
    empty.reservoir_volume = 0.0
    empty.composition.fio2, empty.composition.fin2 = 1.0, 0.0
    empty.step(10.0, uptake_o2=0.25, uptake_agent=0.0)
    assert empty.composition.fin2 == 0.0

    starved = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    starved.set_vent_settings(rr=12, vt=0.6, peep=5.0, ie="1:2", mode="PCV-VG")
    advance_time(starved, 15.0, dt=0.5)
    converged_vg = starved.vent._vg_pressure
    assert converged_vg is not None and converged_vg > 8.0
    starved.set_fgf(0.0, 0.0)
    starved.circuit.reservoir_volume = 0.02
    advance_time(starved, 15.0, dt=0.5)
    assert starved.state.vt < 100.0 and starved.vent._vg_pressure == pytest.approx(converged_vg, abs=1e-6)

    starved.set_fgf(2.0, 3.0, 3.0)
    starved.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV", fio2=0.21)
    assert starved.circuit.fgf_o2 >= starved.circuit.fgf_n2o / 3.0 - 1e-9

    apneic = engine_factory(config=SimulationConfig(mode="awake", dt=0.1, rng_seed=1), start=True)
    apneic.give_drug_bolus("roc", 100.0)
    advance_time(apneic, 60.0, dt=0.1)
    apneic.set_airway_mode("ETT")
    apneic.set_fgf(0.0, 0.0, 0.0)
    apneic.circuit.composition.fio2, apneic.circuit.composition.fin2 = 0.90, 0.10
    apneic.aeration.frc = apneic.resp.frc + 0.05
    assert apneic.state.apnea and apneic._tracheal_fio2 == pytest.approx(0.21, abs=0.01)
    reservoir_0 = apneic.circuit.reservoir_volume
    circuit_o2_0 = (3.0 + reservoir_0) * apneic.circuit.composition.fio2
    resp_o2_0 = apneic.resp.gas_transport.oxygen_from_air_l
    apneic.step(0.1)
    reservoir_lost = reservoir_0 - apneic.circuit.reservoir_volume
    circuit_o2_lost = circuit_o2_0 - (3.0 + apneic.circuit.reservoir_volume) * apneic.circuit.composition.fio2
    resp_o2_gained = apneic.resp.gas_transport.oxygen_from_air_l - resp_o2_0
    assert circuit_o2_lost == pytest.approx(resp_o2_gained, rel=1e-6, abs=1e-9)
    assert reservoir_lost == pytest.approx(resp_o2_gained / apneic._tracheal_fio2, rel=1e-6, abs=1e-9)

    apneic.circuit.composition.fio2, apneic.circuit.composition.fin2 = 0.90, 0.10
    apneic.circuit.reservoir_volume = 0.001
    frc_0 = apneic.resp.frc
    reservoir_0 = apneic.circuit.reservoir_volume
    circuit_o2_0 = (3.0 + reservoir_0) * apneic.circuit.composition.fio2
    resp_o2_0 = apneic.resp.gas_transport.oxygen_from_air_l
    for _ in range(20):
        apneic.aeration.frc = frc_0 + 0.05
        apneic.step(0.1)
    stpd_expansion = (apneic.resp.frc - frc_0) * apneic.resp._btps_to_stpd
    reservoir_lost = reservoir_0 - apneic.circuit.reservoir_volume
    circuit_o2_lost = circuit_o2_0 - (3.0 + apneic.circuit.reservoir_volume) * apneic.circuit.composition.fio2
    resp_o2_gained = apneic.resp.gas_transport.oxygen_from_air_l - resp_o2_0
    assert stpd_expansion == pytest.approx(0.001, abs=1e-9)
    assert reservoir_lost == pytest.approx(0.001, abs=1e-9)
    assert circuit_o2_lost == pytest.approx(resp_o2_gained, rel=1e-6, abs=1e-9)

    apneic.circuit.composition.fio2, apneic.circuit.composition.fin2 = 1.0, 0.0
    apneic._tracheal_fio2 = 1.0
    apneic.circuit.reservoir_volume = 0.001
    apneic.aeration.frc = apneic.resp.frc + 0.05
    circuit_o2_0 = (3.0 + apneic.circuit.reservoir_volume) * apneic.circuit.composition.fio2
    resp_o2_0 = apneic.resp.gas_transport.oxygen_from_air_l
    apneic.step(1.0)
    circuit_o2_lost = circuit_o2_0 - (3.0 + apneic.circuit.reservoir_volume) * apneic.circuit.composition.fio2
    resp_o2_gained = apneic.resp.gas_transport.oxygen_from_air_l - resp_o2_0
    assert circuit_o2_lost == pytest.approx(0.001, abs=1e-9)
    assert resp_o2_gained == pytest.approx(0.001, abs=1e-9)

    loose_mask = engine_factory(config=SimulationConfig(mode="awake", dt=0.1, rng_seed=1), start=True)
    loose_mask.set_airway_mode("Mask")
    loose_mask.set_mask_seal(0.4)
    loose_mask.set_fgf(0.0, 0.0, 0.0)
    loose_mask.circuit.reservoir_volume = 0.0
    advance_time(loose_mask, 60.0, dt=0.1)
    assert loose_mask.state.va > 3.5 and loose_mask.state.sao2 > 96.5


def test_shrinking_reservoir_o2_conservation_stpd_inert_exchange_and_n2o_fio2_blender(engine_factory):
    """Shrinking half-full reservoir conserves O2 mass without phantom spill (#4);
    circle volatile/N2O uptake converts BTPS to STPD consistently with N2 (#5);
    set_fio2 adjusts active N2O/air/O2 flows to hit target FiO2 (#24).
    """
    circuit = CircleSystem()
    circuit.reservoir_volume = 1.5
    circuit.fgf_o2 = 0.1
    circuit.composition.fio2, circuit.composition.fin2 = 0.6, 0.4
    fixed = circuit.volume - circuit.reservoir_capacity
    o2_initial = (fixed + circuit.reservoir_volume) * circuit.composition.fio2
    circuit.step(6.0, uptake_o2=0.3, uptake_agent=0.0)
    o2_final = (fixed + circuit.reservoir_volume) * circuit.composition.fio2
    assert o2_final == pytest.approx(o2_initial + (0.1 - 0.3) * 0.1, abs=1e-9)

    inert = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    inert.pk_n2o.lambda_b_g = 0.0
    inert.set_fgf(0.0, 0.0, 0.0)
    inert.circuit.reservoir_volume = 2.0
    inert.circuit.composition.fio2, inert.circuit.composition.fin2o, inert.circuit.composition.fin2 = 0.5, 0.5, 0.0
    inert.resp.equilibrate_oxygen(0.5, 1.0, inert.vent.settings.peep, fin2=0.5)
    inert._tracheal_fio2 = 0.5
    stpd = inert.resp._btps_to_stpd
    inert_0 = (
        (3.0 + inert.circuit.reservoir_volume) * (inert.circuit.composition.fin2 + inert.circuit.composition.fin2o)
        + inert.resp.frc * stpd * (inert.resp.alveolar_fn2 + inert.pk_n2o.state.p_alv)
    )
    for _ in range(30):
        inert.step(1.0)
    inert_1 = (
        (3.0 + inert.circuit.reservoir_volume) * (inert.circuit.composition.fin2 + inert.circuit.composition.fin2o)
        + inert.resp.frc * stpd * (inert.resp.alveolar_fn2 + inert.pk_n2o.state.p_alv)
    )
    assert inert_1 == pytest.approx(inert_0, abs=0.02)

    blender = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=False)
    blender.set_fgf(1.0, 1.0, 3.0)
    blender.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV", fio2=0.60)
    total_60 = blender.circuit.fgf_o2 + blender.circuit.fgf_air + blender.circuit.fgf_n2o
    assert total_60 == pytest.approx(5.0)
    assert (blender.circuit.fgf_o2 + 0.21 * blender.circuit.fgf_air) / total_60 == pytest.approx(0.60)
    blender.set_fgf(2.0, 0.0, 2.0)
    blender.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV", fio2=0.30)
    total_30 = blender.circuit.fgf_o2 + blender.circuit.fgf_air + blender.circuit.fgf_n2o
    assert total_30 == pytest.approx(4.0)
    assert (blender.circuit.fgf_o2 + 0.21 * blender.circuit.fgf_air) / total_30 == pytest.approx(0.30)
