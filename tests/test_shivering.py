import pytest

from anasim.core.constants import SHIVER_MAX_MULTIPLIER
from anasim.core.state import SimulationConfig


def test_paralysis_stops_shivering_and_reduces_co2_production(engine_factory, advance_time):
    engine = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=123), start=True, baseline_temp=35.0, baseline_rr=0.0
    )
    # Hold ventilation constant to distinguish metabolism from respiratory compensation.
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5, ie="1:2")
    engine.set_vent_power(True)
    engine.step(1.0)
    base_depth_met = engine.thermal.depth_and_metabolism(engine.state.temp_c, 0.0, 0.0, 0.0)[1]
    assert engine.state.shivering > 0.02
    assert engine.thermal.metabolic_factor == pytest.approx(
        base_depth_met * (1.0 + SHIVER_MAX_MULTIPLIER * engine.state.shivering), abs=1e-4
    )
    advance_time(engine, 299, dt=2.0)
    assert engine.state.shivering > 0.4
    shivering_co2, ventilation = engine.state.pa_co2, engine.state.mv
    shivering_metabolism = engine.thermal.metabolic_factor

    engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
    advance_time(engine, 360, dt=2.0)
    assert engine.state.tof < 5.0
    assert engine.state.shivering < 0.1
    assert engine.state.temp_c < 36.0
    assert engine.state.mv == pytest.approx(ventilation, rel=0.02)
    assert engine.thermal.metabolic_factor < 0.7 * shivering_metabolism
    # Stored CO2 continues returning after metabolic production falls.
    assert engine.state.pa_co2 < shivering_co2


def test_shivering_recruits_hr_and_co_without_severe_lactic_acidosis(engine_factory, advance_time):
    """Ralley 1988: Healthy shivering patients recruit HR and CO aerobically instead of severe lactic acidosis."""
    healthy = engine_factory(
        config=SimulationConfig(mode="awake", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
        baseline_temp=34.8,
    )
    normo = engine_factory(
        config=SimulationConfig(mode="awake", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
        baseline_temp=37.0,
    )
    assert healthy.state.shivering > 0.4
    advance_time(healthy, 600.0, dt=2.0)
    advance_time(normo, 600.0, dt=2.0)
    assert healthy.state.hr > normo.state.hr + 15.0
    assert healthy.state.co > normo.state.co * 1.25
    assert healthy.state.lactate < 2.5


def test_shivering_threshold_arrest_cutoff_and_awake_init(engine_factory):
    """Normothermic 36.1 C stays quiet; 35.0 C seeds awake shivering; <=31 C and VFIB stop shivering."""
    normo = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, baseline_temp=36.1)
    normo.step(60.0)
    assert normo.state.shivering == 0.0

    cool = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, baseline_temp=35.0)
    assert cool.state.shivering > 0.35
    cool.set_rhythm("VFIB")
    cool.step(1.0)
    assert cool.state.shivering == 0.0

    severe = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, baseline_temp=28.0)
    assert severe.state.shivering == 0.0
    severe.step(60.0)
    assert severe.state.shivering == 0.0
