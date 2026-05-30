import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient


@pytest.fixture
def engine():
    return SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="awake", dt=0.5),
    )


@pytest.mark.parametrize(
    ("drug", "user_rate", "rate_attr", "expected_internal"),
    [
        ("propofol", 3600.0, "propofol_rate_mg_sec", 1.0),
        ("remi", 60.0, "remi_rate_ug_sec", 1.0),
        ("nore", 60.0, "nore_rate_ug_sec", 1.0),
        ("vaso", 0.06, "vaso_rate_mu_sec", 1.0),
        ("phenyl", 60.0, "phenyl_rate_ug_sec", 1.0),
        ("epi", 60.0, "epi_rate_ug_sec", 1.0),
        ("dobu", 60.0, "dobu_rate_ug_sec", 1.0),
        ("milri", 60.0, "mil_rate_ug_sec", 1.0),
        ("roc", 3600.0, "roc_rate_mg_sec", 1.0),
    ],
)
def test_controllable_drug_rates_convert_to_internal_units(engine, drug, user_rate, rate_attr, expected_internal):
    engine.set_drug_rate(drug, user_rate)

    assert getattr(engine, rate_attr) == pytest.approx(expected_internal)
    assert engine.get_drug_state(drug)["rate"] == pytest.approx(user_rate)
