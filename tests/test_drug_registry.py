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


def test_registry_drives_controller_metadata_and_rate_units(engine):
    assert engine.get_controllable_drugs() is DRUG_REGISTRY

    for spec in DRUG_REGISTRY:
        assert getattr(engine, spec.pk_attr) is not None
        assert getattr(engine, spec.rate_attr) == 0.0
        assert getattr(engine, spec.tci_attr) is None

    rate_cases = (
        ("propofol", 3600.0, "propofol_rate_mg_sec", 1.0),
        ("remi", 60.0, "remi_rate_ug_sec", 1.0),
        ("nore", 60.0, "nore_rate_ug_sec", 1.0),
        ("vaso", 0.06, "vaso_rate_mu_sec", 1.0),
        ("phenyl", 60.0, "phenyl_rate_ug_sec", 1.0),
        ("epi", 60.0, "epi_rate_ug_sec", 1.0),
        ("dobu", 60.0, "dobu_rate_ug_sec", 1.0),
        ("milri", 60.0, "mil_rate_ug_sec", 1.0),
        ("roc", 3600.0, "roc_rate_mg_sec", 1.0),
    )
    for drug, user_rate, rate_attr, expected_internal in rate_cases:
        engine.set_drug_rate(drug, user_rate)
        assert getattr(engine, rate_attr) == pytest.approx(expected_internal)
        assert engine.get_drug_state(drug)["rate"] == pytest.approx(user_rate)


def test_max_rate_policy_converts_to_model_units():
    cases = (
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
    )
    for policy, weight_kg, expected in cases:
        assert policy.internal_rate(weight_kg) == pytest.approx(expected)


def test_bolus_routes_and_units_come_from_registry(engine):
    for spec in DRUG_REGISTRY:
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
