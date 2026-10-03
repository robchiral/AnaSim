import pytest

from anasim.core.state import SimulationConfig


def test_paralysis_stops_shivering_and_reduces_co2_production(engine_factory, advance_time):
    engine = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=123), start=True, baseline_temp=35.0
    )
    # Hold ventilation constant to distinguish metabolism from respiratory compensation.
    engine.resp.hcvr_slope_baseline = 0.0
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5, ie="1:2")
    engine.set_vent_power(True)
    advance_time(engine, 300)
    assert engine.state.shivering > 0.4
    shivering_co2, ventilation = engine.state.pa_co2, engine.state.mv

    engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
    advance_time(engine, 360)
    assert engine.state.tof < 5.0
    assert engine.state.shivering < 0.1
    assert engine.state.temp_c < 36.0
    assert engine.state.mv == pytest.approx(ventilation, rel=0.02)
    assert engine.state.pa_co2 < shivering_co2 - 10.0
