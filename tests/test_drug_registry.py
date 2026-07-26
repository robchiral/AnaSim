import pytest

from anasim.core.drug_registry import (
    DRUG_REGISTRY,
    MaxRateBasis,
    MaxRatePolicy,
    get_drug_spec,
    resolve_bolus_drug,
)
from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient


@pytest.fixture
def engine():
    return SimulationEngine(
        Patient(age=40, weight=70, height=170, sex="male"),
        SimulationConfig(mode="awake", dt=0.5),
    )


def test_registry_generates_all_controller_metadata(engine):
    assert engine.get_controllable_drugs() is DRUG_REGISTRY

    for spec in DRUG_REGISTRY:
        assert getattr(engine, spec.pk_attr) is not None
        assert getattr(engine, spec.rate_attr) == 0.0
        assert getattr(engine, spec.tci_attr) is None


@pytest.mark.parametrize(
    ("policy", "weight_kg", "expected"),
    [
        (MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 0.5), 60.0, 0.5),
        (MaxRatePolicy(MaxRateBasis.PER_KG_HOUR, 1.0), 60.0, 1.0 / 60.0),
        (
            MaxRatePolicy(
                MaxRateBasis.ABSOLUTE_PER_MINUTE,
                0.1,
                model_unit_scale=1000.0,
            ),
            60.0,
            100.0 / 60.0,
        ),
    ],
)
def test_max_rate_policy_converts_to_model_units(policy, weight_kg, expected):
    assert policy.internal_rate(weight_kg) == pytest.approx(expected)


@pytest.mark.parametrize("spec", DRUG_REGISTRY, ids=lambda spec: spec.key)
def test_bolus_routes_and_units_come_from_registry(engine, spec):
    model = getattr(engine, spec.pk_attr)
    initial_c1 = model.state.c1

    engine.give_drug_bolus(spec.tci_name, 2.0)

    assert model.state.c1 == pytest.approx(
        initial_c1 + 2.0 * spec.bolus_model_scale / model.v1
    )
    assert resolve_bolus_drug(spec.key) is spec
    assert resolve_bolus_drug(spec.name.upper()) is spec


def test_unknown_drug_names_fail_explicitly(engine):
    with pytest.raises(ValueError, match="Unknown controllable drug"):
        get_drug_spec("not-a-drug")
    with pytest.raises(ValueError, match="Unknown bolus drug"):
        engine.give_drug_bolus("not-a-drug", 1.0)
