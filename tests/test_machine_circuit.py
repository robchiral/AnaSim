from anasim.machine.circuit import CircleSystem
from anasim.machine.volatile import Vaporizer
from anasim.core.state import SimulationConfig


def test_circuit_oxygen_washin_and_washout():
    """High FGF O2 should wash in; air should wash out to ~room FiO2."""
    circuit = CircleSystem(volume_l=6.0)
    circuit.fgf_o2 = 10.0
    circuit.fgf_air = 0.0
    circuit.fgf_n2o = 0.0

    for _ in range(300):  # 5 minutes
        circuit.step(1.0, uptake_o2=0.25, uptake_agent=0.0)

    assert circuit.composition.fio2 > 0.8, "High O2 flow should wash in quickly"

    circuit.fgf_o2 = 0.0
    circuit.fgf_air = 10.0
    for _ in range(300):  # 5 minutes washout
        circuit.step(1.0, uptake_o2=0.25, uptake_agent=0.0)

    assert circuit.composition.fio2 < 0.5, "Air washout should reduce FiO2 toward room air"


def test_circuit_agent_washin_and_washout():
    """Volatile agent should wash in with vaporizer and wash out when off."""
    circuit = CircleSystem(volume_l=6.0)
    circuit.fgf_o2 = 6.0
    circuit.fgf_air = 0.0
    circuit.vaporizer_setting = 2.0
    circuit.vaporizer_on = True

    for _ in range(300):  # ~5 time constants
        circuit.step(1.0, uptake_o2=0.0, uptake_agent=0.0)

    fi_agent_on = circuit.composition.fi_agent
    assert fi_agent_on > 0.015, f"Agent wash-in too small: {fi_agent_on:.3f}"

    circuit.vaporizer_on = False
    circuit.vaporizer_setting = 0.0
    for _ in range(300):
        circuit.step(1.0, uptake_o2=0.0, uptake_agent=0.0)

    assert circuit.composition.fi_agent < fi_agent_on * 0.6, "Agent should wash out when vaporizer off"


def test_vaporizer_consumption_and_empty_shutdown():
    """Vaporizer should consume agent and shut off when empty."""
    vap = Vaporizer()
    vap.set_concentration(2.0)
    start_level = vap.state.level

    for _ in range(60):  # 1 hour total in 1-minute steps
        vap.step(60.0, fgf_l_min=5.0)

    assert vap.state.level < start_level, "Vaporizer should consume liquid agent over time"

    vap.state.level = 0.01
    vap.set_concentration(2.0)
    vap.step(60.0, fgf_l_min=10.0)

    assert vap.state.level == 0.0
    assert not vap.state.is_on
    assert vap.state.setting == 0.0


def test_engine_volatile_washin_washout(engine_factory):
    """Circuit + volatile PK integration: vaporizer raises MAC, washout lowers it."""
    config = SimulationConfig(mode="awake")
    engine = engine_factory(config=config, start=True)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_fgf(8.0, 0.0)

    engine.set_vaporizer("Sevoflurane", 2.0)
    for _ in range(300):  # 5 min wash-in
        engine.step(1.0)

    mac_on = engine.state.mac
    fi_sevo = engine.state.fi_sevo
    fio2 = engine.state.fio2

    assert fi_sevo > 1.0, f"FiSevo too low with vaporizer on: {fi_sevo:.2f}%"
    assert fio2 > 0.8, f"FiO2 did not track circuit wash-in: {fio2:.2f}"
    assert mac_on > 0.2, f"MAC did not rise with vaporizer on: {mac_on:.2f}"

    engine.set_vaporizer("Sevoflurane", 0.0)
    engine.set_fgf(10.0, 0.0)
    for _ in range(300):  # 5 min washout
        engine.step(1.0)

    assert engine.state.mac < mac_on * 0.7, "MAC should decrease with washout/high FGF"


def test_engine_n2o_washin_washout(engine_factory):
    """N2O wash-in should raise FiN2O and MAC fraction; washout should reduce them."""
    config = SimulationConfig(mode="awake")
    engine = engine_factory(config=config, start=True)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
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
