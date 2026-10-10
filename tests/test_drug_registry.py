import pytest

from anasim.core.drug_registry import DRUG_REGISTRY
from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient


@pytest.fixture
def engine():
    return SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="awake", dt=0.5),
    )


def test_infusion_rates_convert_the_prescribed_units(engine):
    rate_cases = (
        ("propofol", 60000.0 / engine.patient.weight),
        ("remi", 60.0 / engine.patient.weight),
        ("fentanyl", 3600.0),
        ("midazolam", 3.6),
        ("ketamine", 3600.0),
        ("lidocaine", 3600.0),
        ("nore", 60.0),
        ("vaso", 0.06),
        ("phenyl", 60.0),
        ("epi", 60.0),
        ("dobu", 60.0),
        ("milri", 60.0),
        ("esmolol", 60.0),
        ("roc", 3600.0),
    )
    # Each user rate is one model unit per second.
    assert {drug for drug, _ in rate_cases} == set(engine.infusion_rates)
    for drug, user_rate in rate_cases:
        engine.set_drug_rate(drug, user_rate)
        assert engine.infusion_rates[drug] == pytest.approx(1.0)
        assert engine.get_drug_state(drug)["rate"] == pytest.approx(user_rate)


def test_boluses_deliver_the_prescribed_mass_to_each_pk_model(engine):
    # Prescribed mg or units become mcg or milliunits in these four models.
    model_scales = {"midazolam": 1000.0, "vaso": 1000.0, "labetalol": 1000.0, "glyco": 1000.0}
    for spec in DRUG_REGISTRY:
        model = getattr(engine, spec.pk_attr)
        initial_c1 = model.state.c1

        engine.give_drug_bolus(spec.generic_name, 2.0)

        assert model.state.c1 == pytest.approx(
            initial_c1 + 2.0 * model_scales.get(spec.key, 1.0) / model.v1
        )


def test_controls_a_drug_lacks_fail_explicitly(engine):
    assert engine.get_drug_state("glyco") == {"rate": 0.0, "target": 0.0, "is_tci": False}
    with pytest.raises(ValueError, match="Glycopyrrolate has no infusion"):
        engine.set_drug_rate("glyco", 1.0)
    with pytest.raises(ValueError, match="Esmolol has no target-controlled infusion"):
        engine.set_drug_target("esmolol", 1.0)
