import copy

import pytest

from anasim.core.projection import DEADSPACE_APNEIC_STPD_L
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
    balanced = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1), start=True)
    assert balanced.vaporizer.state.level == pytest.approx(250.0)
    assert balanced.resp.gas_transport.consumed_o2_l == 0.0
    assert balanced.resp.gas_transport.oxygen_from_air_l == 0.0

    config = SimulationConfig(mode="awake", dt=2.0)
    engine = engine_factory(config=config, start=True)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(8.0, 0.0)

    engine.set_vaporizer(2.0)
    initial_level = engine.vaporizer.state.level
    for _ in range(150):  # 5 min wash-in
        engine.step(2.0)

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
    assert engine.vaporizer.state.setting == 2.0
    assert not engine.circuit.vaporizer_on
    engine.set_fgf(10.0, 0.0)
    for _ in range(150):  # 5 min washout
        engine.step(2.0)

    assert engine.state.mac < mac_on * 0.7, "MAC should decrease with washout/high FGF"
    assert engine.state.fi_sevo < fi_sevo * 0.1
    assert engine.state.fio2 > 0.99


def test_engine_n2o_washin_washout(engine_factory):
    """N2O wash-in raises FiN2O and MAC fraction; disconnect zeros end-tidal gases; washout reduces MAC."""
    config = SimulationConfig(mode="awake", dt=2.0)
    engine = engine_factory(config=config, start=True)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(2.0, 0.0, n2o_l_min=4.0)
    engine.set_vaporizer(1.0)

    for _ in range(150):  # 5 min wash-in
        engine.step(2.0)

    fi_n2o = engine.state.fi_n2o
    mac_n2o = engine.state.mac_n2o

    assert fi_n2o > 30.0, f"FiN2O too low with N2O on: {fi_n2o:.2f}%"
    assert mac_n2o > 0.2, f"N2O MAC fraction too low: {mac_n2o:.2f}"

    disconnected = copy.deepcopy(engine)
    disconnected.set_airway_mode("None")
    disconnected.step(0.5)
    assert disconnected.state.et_sevo == disconnected.state.et_n2o == disconnected.state.et_mac == 0.0
    assert disconnected.state.mac > 0.2

    engine.set_vaporizer(0.0)
    engine.set_fgf(6.0, 0.0, n2o_l_min=0.0)
    for _ in range(150):  # 5 min washout
        engine.step(2.0)

    assert engine.state.fi_n2o < fi_n2o * 0.4, "FiN2O should decrease with washout/high FGF"
    assert engine.state.mac_n2o < mac_n2o * 0.5, "N2O MAC should decrease with washout/high FGF"


def test_fio2_blender_with_n2o(engine_factory):
    """FiO2 blender hits target with fixed N2O flow while respecting the >=25% O2 hypoxic guard."""
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
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV", fio2=0.3)
    configured_total = engine.circuit.fgf_o2 + engine.circuit.fgf_air + engine.circuit.fgf_n2o
    configured_fio2 = (engine.circuit.fgf_o2 + 0.21 * engine.circuit.fgf_air) / configured_total
    assert configured_total == pytest.approx(total)
    assert configured_fio2 == pytest.approx(0.3)

    engine.set_oxygen_supply_connected(True)
    engine.set_fgf(1.0, 1.0, 3.0)
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV", fio2=0.60)
    total_60 = engine.circuit.fgf_o2 + engine.circuit.fgf_air + engine.circuit.fgf_n2o
    assert total_60 == pytest.approx(5.0)
    assert (engine.circuit.fgf_o2 + 0.21 * engine.circuit.fgf_air) / total_60 == pytest.approx(0.60)

    engine.set_fgf(2.0, 3.0, 3.0)
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV", fio2=0.21)
    assert engine.circuit.fgf_o2 >= engine.circuit.fgf_n2o / 3.0 - 1e-9


def test_n2o_hypoxic_guard_step_invariance_and_low_flow_nitrogen(engine_factory, advance_time):
    """Circle enforces >=25% O2 with N2O, conserves O2 across dt, accumulates exhaled N2
    at metabolic flow, and preserves vaporizer_on across zero FGF.
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

    low_flow = engine_factory(config=SimulationConfig(mode="awake", dt=2.0, rng_seed=1), start=True)
    low_flow.circuit.composition.fio2, low_flow.circuit.composition.fin2 = 1.0, 0.0
    low_flow.set_airway_mode("ETT")
    low_flow.set_vent_power(True)
    low_flow.set_vaporizer(2.0)
    low_flow.set_fgf(0.0, 0.0)
    low_flow.step(1.0)
    assert low_flow.circuit.vaporizer_on is True
    low_flow.set_vaporizer(0.0)
    low_flow.set_fgf(0.3, 0.0)
    advance_time(low_flow, 180.0, dt=2.0)
    assert low_flow.circuit.composition.fin2 > 0.15


def test_shrinking_reservoir_o2_conservation_and_stpd_inert_exchange(engine_factory):
    """Shrinking half-full reservoir conserves O2 mass without phantom spill, and
    circle volatile/N2O uptake converts BTPS to STPD consistently with N2.
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
    for _ in range(15):
        inert.step(2.0)
    inert_1 = (
        (3.0 + inert.circuit.reservoir_volume) * (inert.circuit.composition.fin2 + inert.circuit.composition.fin2o)
        + inert.resp.frc * stpd * (inert.resp.alveolar_fn2 + inert.pk_n2o.state.p_alv)
    )
    assert inert_1 == pytest.approx(inert_0, abs=0.02)


def test_apneic_deadspace_washin_and_closed_circuit_multigas_dt_invariance(
    engine_factory, advance_time
):
    """Apneic oxygenation washes dead-space tracheal O2 upward across dt while conserving O2,
    and closed-circuit volatile/N2O/N2 transport stays consistent across dt.
    """
    washin = []
    for dt in (0.2, 1.0):
        apneic = engine_factory(config=SimulationConfig(mode="awake", dt=dt, rng_seed=1), start=True)
        apneic.give_drug_bolus("roc", 90.0)
        advance_time(apneic, 60.0, dt=dt)
        apneic.set_airway_mode("ETT")
        apneic.set_fgf(0.0, 0.0, 0.0)
        apneic.circuit.reservoir_volume = 2.0
        apneic.circuit.composition.fio2, apneic.circuit.composition.fin2 = 1.0, 0.0
        assert apneic._tracheal_fio2 < 0.30
        o2_circuit_0 = (3.0 + apneic.circuit.reservoir_volume) * apneic.circuit.composition.fio2
        o2_resp_0 = apneic.resp.gas_transport.oxygen_from_air_l
        advance_time(apneic, 90.0, dt=dt)
        assert apneic._tracheal_fio2 > 0.65
        o2_circuit_lost = o2_circuit_0 - (3.0 + apneic.circuit.reservoir_volume) * apneic.circuit.composition.fio2
        o2_resp_gained = apneic.resp.gas_transport.oxygen_from_air_l - o2_resp_0
        assert o2_circuit_lost == pytest.approx(o2_resp_gained, abs=0.04)
        washin.append(apneic._tracheal_fio2)
    assert abs(washin[0] - washin[1]) < 0.015

    endpoints = []
    for dt in (0.1, 1.0):
        e = engine_factory(config=SimulationConfig(mode="steady_state", dt=dt, rng_seed=1), start=True)
        e.give_drug_bolus("roc", 70.0)
        e.set_fgf(0.25, 0.0, 0.0)
        e.circuit.composition.fio2 = 0.50
        e.circuit.composition.fin2o = 0.30
        e.circuit.composition.fin2 = 0.18
        e.circuit.composition.fi_agent = 0.02
        advance_time(e, 60.0, dt=dt)
        endpoints.append((
            e.circuit.composition.fi_agent,
            e.circuit.composition.fin2o,
            e.pk_sevo.state.p_alv,
            e.pk_n2o.state.p_alv,
        ))
    assert endpoints[1] == pytest.approx(endpoints[0], abs=0.004)

    # Empty-reservoir CPAP draws no phantom O2, and sevoflurane_enabled=False blocks vaporizer delivery.
    starved = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=1), start=True)
    starved.give_drug_bolus("roc", 70.0)
    starved.vent.update_settings(mode="CPAP", rr=0, peep=8.0)
    starved.set_fgf(0.0, 0.0, 0.0)
    starved.circuit.reservoir_volume = 0.0
    o2_0 = starved.resp.gas_transport.oxygen_from_air_l
    advance_time(starved, 10.0, dt=0.5)
    assert starved.resp.gas_transport.oxygen_from_air_l <= o2_0 + 1e-6

    no_sevo = engine_factory(config=SimulationConfig(mode="awake", sevoflurane_enabled=False, dt=0.5, rng_seed=1), start=True)
    no_sevo.set_airway_mode("ETT")
    no_sevo.set_vent_power(True)
    no_sevo.vaporizer.set_concentration(4.0)
    advance_time(no_sevo, 15.0, dt=0.5)
    assert no_sevo.state.fi_sevo == 0.0 and no_sevo.pk_sevo.state.p_alv == 0.0


def test_mixed_gas_apneic_inflow_and_copd_exchangeable_frc_conserve_gas(engine_factory, advance_time):
    """Circuit, dead space, and lung share one gas balance during mixed-gas apneic FRC expansion,
    and severe COPD uses the same exchangeable FRC change for inhaled agents and respiratory gases."""
    mixed = engine_factory(config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1), start=True)
    mixed.set_airway_mode("ETT")
    mixed.give_drug_bolus("roc", 90.0)
    mixed.set_fgf(0.0, 0.0, 0.0)
    advance_time(mixed, 90.0, dt=1.0)
    assert mixed.resp.state.apnea and mixed.state.va == 0.0
    mixed.circuit.reservoir_volume = 2.0
    mixed.circuit.composition.fio2 = 0.5
    mixed.circuit.composition.fin2 = 0.5
    mixed._tracheal_fio2 = 0.21
    o2_circ_0 = (3.0 + mixed.circuit.reservoir_volume) * mixed.circuit.composition.fio2
    o2_lung_0 = mixed.resp.gas_transport.oxygen_from_air_l
    o2_ds_0 = DEADSPACE_APNEIC_STPD_L * mixed._tracheal_fio2
    mixed.aeration.frc = mixed.resp.frc + 0.5
    mixed.step(1.0)
    o2_circ_lost = o2_circ_0 - (3.0 + mixed.circuit.reservoir_volume) * mixed.circuit.composition.fio2
    o2_lung_gained = mixed.resp.gas_transport.oxygen_from_air_l - o2_lung_0
    o2_ds_gained = DEADSPACE_APNEIC_STPD_L * mixed._tracheal_fio2 - o2_ds_0
    assert o2_circ_lost > 0.15
    assert o2_circ_lost == pytest.approx(o2_lung_gained + o2_ds_gained, abs=1e-3)

    copd = engine_factory(
        config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1), start=True, copd_severity=1.0,
    )
    copd.set_airway_mode("ETT")
    copd.give_drug_bolus("roc", 90.0)
    copd.set_fgf(0.0, 0.0, 0.0)
    advance_time(copd, 90.0, dt=1.0)
    copd.pk_n2o.lambda_b_g = 0.0
    copd.pk_sevo.lambda_b_g = 0.0
    copd.circuit.reservoir_volume = 2.0
    copd.circuit.composition.fio2 = 0.75
    copd.circuit.composition.fin2o = 0.25
    copd.circuit.composition.fin2 = 0.0
    copd.circuit.composition.fi_agent = 0.0
    copd._tracheal_gas = (0.75, 0.0, 0.25, 0.0)
    stpd = copd.resp._btps_to_stpd
    n2o_circ_0 = (3.0 + copd.circuit.reservoir_volume) * copd.circuit.composition.fin2o
    n2o_lung_0 = copd.resp.gas_volume_l * stpd * copd.pk_n2o.state.p_alv
    n2o_ds_0 = DEADSPACE_APNEIC_STPD_L * copd._tracheal_gas[2]
    copd.aeration.frc = copd.resp.frc + 0.5
    copd.step(1.0)
    n2o_circ_lost = n2o_circ_0 - (3.0 + copd.circuit.reservoir_volume) * copd.circuit.composition.fin2o
    n2o_lung_gained = copd.resp.gas_volume_l * stpd * copd.pk_n2o.state.p_alv - n2o_lung_0
    n2o_ds_gained = DEADSPACE_APNEIC_STPD_L * copd._tracheal_gas[2] - n2o_ds_0
    assert n2o_circ_lost > 0.03
    assert n2o_circ_lost == pytest.approx(n2o_lung_gained + n2o_ds_gained, abs=1e-3)


def test_loose_mask_low_va_preserves_inspired_agents_and_equilibrate_oxygen_sums_to_one(engine_factory):
    """Loose mask with shallow spontaneous breathing (1e-4 < VA <= 0.1) keeps circuit agents instead
    of zeroing them, and equilibrate_oxygen alveolar gas fractions sum to unity."""
    from anasim.core.runtime import step_machine

    eng = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    eng.resp.equilibrate_oxygen(0.21)
    frac_sum = (
        (eng.resp.state.p_alveolar_o2 + eng.resp.state.p_alveolar_co2) / eng.resp._atm_dry
        + eng.resp.alveolar_fn2
    )
    assert frac_sum == pytest.approx(1.0, abs=1e-12)

    eng.set_airway_mode("Mask")
    eng.set_mask_seal(0.5)
    eng.circuit.composition.fio2 = 0.45
    eng.circuit.composition.fi_agent = 0.05
    eng.circuit.composition.fin2o = 0.50
    eng.circuit.composition.fin2 = 0.0
    eng._tracheal_gas = (0.95, 0.0, 0.0, 0.05)
    eng.state.va = 0.05
    eng.resp.state.apnea = False
    fi_sevo, fi_n2o = step_machine(eng, 0.05)
    assert fi_sevo == pytest.approx(0.025, abs=1e-6)
    assert fi_n2o == pytest.approx(0.25, abs=1e-6)
