import pytest

from anasim.core.state import SimulationConfig
from anasim.machine.circuit import CircleSystem
from anasim.machine.volatile import Vaporizer


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


def test_engine_volatile_washin_washout(engine_factory):
    """Circuit + volatile PK integration: vaporizer raises MAC, washout lowers it."""
    config = SimulationConfig(mode="awake")
    engine = engine_factory(config=config, start=True)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(8.0, 0.0)

    engine.set_vaporizer("Sevoflurane", 2.0)
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
