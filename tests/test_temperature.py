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
