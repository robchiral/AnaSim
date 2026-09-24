import pytest

from anasim.core.state import SimulationConfig


def _advance(engine, seconds: float, dt: float = 1.0) -> None:
    for _ in range(int(seconds / dt)):
        engine.step(dt)


def test_anesthetized_patient_cools_and_forced_air_rewarms(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state"), start=True)
    assert engine.state.temp_c == 37.0

    _advance(engine, 20 * 60, dt=2.0)
    cooled = engine.state.temp_c
    assert cooled < 37.0

    engine.set_bair_hugger(43.0)
    _advance(engine, 20 * 60, dt=2.0)
    assert engine.state.temp_c > cooled


def test_hypothermia_lowers_paco2_at_fixed_ventilation(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state"))
    engine.vent.is_on = True
    engine.resp_mech.set_rr = 10
    engine.resp_mech.set_vt = 0.5
    engine.start()

    engine.state.temp_c = 37.0
    _advance(engine, 60)
    normothermic = engine.state.pa_co2

    engine.state.temp_c = 30.0
    _advance(engine, 600)
    assert engine.state.pa_co2 < normothermic


class TestRedistributionHypothermia:
    """Core temperature after induction (Matsukawa et al. Anesthesiology. 1995)."""

    def test_steady_state_start_has_no_temperature_step(self, engine_factory):
        engine = engine_factory(config=SimulationConfig(mode="steady_state"), start=True)
        engine.step(0.1)
        assert engine.state.temp_c == pytest.approx(37.0, abs=0.005)

    def test_first_hour_core_drop_after_induction(self, engine_factory):
        engine = engine_factory(start=True)
        engine.set_airway_mode("ETT")
        engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        engine.enable_tci("propofol", 3.5)
        engine.enable_tci("remi", 3.0)
        _advance(engine, 3600, dt=1.0)
        assert 0.8 < 37.0 - engine.state.temp_c < 1.7
