import pytest

from anasim.core.state import SimulationConfig


def _advance(engine, seconds: float, dt: float = 1.0) -> None:
    for _ in range(int(seconds / dt)):
        engine.step(dt)


def test_anesthetized_patient_cools_and_forced_air_rewarms(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state"), start=True)
    # Redistribution belongs to the hidden settle, not visible time.
    engine.step(0.1)
    assert engine.state.temp_c == pytest.approx(37.0, abs=0.005)

    _advance(engine, 20 * 60, dt=2.0)
    cooled = engine.state.temp_c
    assert cooled < 37.0

    engine.set_bair_hugger(43.0)
    _advance(engine, 20 * 60, dt=2.0)
    assert engine.state.temp_c > cooled


def test_hypothermia_lowers_paco2_at_fixed_ventilation(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state"))
    engine.vent.is_on = True
    engine.start()

    engine.state.temp_c = 37.0
    _advance(engine, 60)
    normothermic = engine.state.pa_co2

    engine.state.temp_c = 30.0
    _advance(engine, 600)
    assert engine.state.pa_co2 < normothermic


def test_hypothermia_slows_apneic_co2_rise(engine_factory):
    rises = []
    for temperature in (30.0, 37.0):
        engine = engine_factory(start=True, baseline_temp=temperature)
        engine.set_airway_mode("ETT")
        engine.set_fgf(10.0, 0.0)
        engine.set_vent_power(True)
        engine.give_drug_bolus("roc", engine.patient.weight)
        _advance(engine, 180)
        # Apneic oxygenation keeps SaO2 up. Prior shivering and tissue stores
        # also affect the transient, so it need not scale directly with VCO2.
        engine.set_vent_power(False)
        start = engine.state.pa_co2
        _advance(engine, 120)
        rises.append(engine.state.pa_co2 - start)
    assert rises[0] < 0.9 * rises[1]


def test_core_drop_after_induction(engine_factory):
    """Matsukawa 1995: core temperature fell 1.6 ± 0.3 °C in the first hour and 2.8 ± 0.5 °C by 3 h."""
    engine = engine_factory(config=SimulationConfig(tci_enabled=True), start=True)
    # Awake thermoregulation holds core temperature before induction.
    _advance(engine, 1800, dt=2.0)
    assert engine.state.temp_c == pytest.approx(37.0, abs=0.02)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.enable_tci("propofol", 3.5)
    engine.enable_tci("remi", 3.0)
    _advance(engine, 3600, dt=2.0)
    assert 37.0 - engine.state.temp_c == pytest.approx(1.6, abs=0.3)
    _advance(engine, 7200, dt=2.0)
    assert 37.0 - engine.state.temp_c == pytest.approx(2.8, abs=0.5)


def test_obese_redistribution_cooling_sepsis_fever_and_hypothermic_bradycardia(engine_factory):
    """Kurz 1995: obesity blunts first-hour redistribution cooling (~0.4-0.6 °C vs 1.6 °C);
    sepsis elevates core temperature; severe hypothermia (<33 °C) depresses resting sinus HR.
    """
    obese = engine_factory(
        config=SimulationConfig(tci_enabled=True, rng_seed=1),
        start=True, age=40, weight=150.0, height=165.0, sex="female",
    )
    obese.set_airway_mode("ETT")
    obese.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    obese.set_vent_power(True)
    obese.enable_tci("propofol", 3.5)
    obese.enable_tci("remi", 3.0)
    _advance(obese, 3600, dt=4.0)
    assert 0.30 < 37.0 - obese.state.temp_c < 0.80

    sepsis = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    sepsis.start_sepsis()
    _advance(sepsis, 1200, dt=2.0)
    assert sepsis.state.temp_c > 37.12

    cold = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, baseline_temp=27.0)
    assert cold.state.hr < 55.0
    cold.step(1.0)
    assert cold.state.hr < 55.0


def test_balanced_mac_temp_parity_abg_warming_to_37c_and_awake_febrile_setpoint(engine_factory):
    """Balanced steady-state starts at uniform MAC across baseline temperatures (#15);
    draw_blood_gas() anaerobically warms hypothermic samples to 37 °C (#38);
    awake febrile baseline_temp holds its setpoint without cooling toward 37 °C (#42).
    """
    macs = [
        engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1),
            start=True,
            baseline_temp=temp,
        ).state.mac
        for temp in (32.0, 37.0, 39.0)
    ]
    assert macs[0] == pytest.approx(macs[1], abs=0.02)
    assert macs[2] == pytest.approx(macs[1], abs=0.02)

    hypo = engine_factory(
        config=SimulationConfig(mode="steady_state", rng_seed=1),
        start=True,
        baseline_temp=32.0,
    )
    abg = hypo.draw_blood_gas()
    assert abg.pa_co2 > hypo.state.pa_co2 + 3.0
    assert abg.pao2 > hypo.state.pao2
    assert abg.ph < hypo.state.ph - 0.04

    febrile = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True,
        baseline_temp=38.5,
    )
    _advance(febrile, 1800, dt=5.0)
    assert febrile.state.temp_c == pytest.approx(38.5, abs=0.03)
