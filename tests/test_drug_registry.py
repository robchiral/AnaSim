import math

import pytest

from anasim.core.drug_registry import DRUG_REGISTRY, get_drug_spec
from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.core.units import convert_rate
from anasim.core.utils import clamp, clamp01
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
        ("morphine", 3.6),
        ("hydromorphone", 3.6),
        ("ketamine", 3600.0 / engine.patient.weight),
        ("lidocaine", 3600.0 / engine.patient.weight),
        ("amiodarone", 60.0),
        ("nore", 60.0),
        ("vaso", 0.06),
        ("phenyl", 60.0),
        ("epi", 60.0),
        ("dobu", 60.0 / engine.patient.weight),
        ("milri", 60.0 / engine.patient.weight),
        ("esmolol", 60000.0 / engine.patient.weight),
        ("roc", 60000.0 / engine.patient.weight),
    )
    # Each user rate is one model unit per second.
    assert {drug for drug, _ in rate_cases} == set(engine.infusion_rates)
    for drug, user_rate in rate_cases:
        engine.set_drug_rate(drug, user_rate)
        assert engine.infusion_rates[drug] == pytest.approx(1.0)
        assert engine.get_drug_state(drug)["rate"] == pytest.approx(user_rate)


def test_boluses_deliver_the_prescribed_mass_to_each_pk_model(engine):
    # Prescribed mg or units become mcg or milliunits in these models.
    model_scales = dict.fromkeys(
        ("midazolam", "vaso", "labetalol", "glyco", "morphine", "hydromorphone", "atropine", "ephedrine"),
        1000.0,
    )
    for spec in DRUG_REGISTRY:
        model = engine.pk[spec.key]
        initial_c1 = model.state.c1

        engine.give_drug_bolus(spec.generic_name, 2.0)

        assert model.state.c1 == pytest.approx(
            initial_c1 + 2.0 * model_scales.get(spec.key, 1.0) / model.v1
        )


def test_controls_a_drug_lacks_or_invalid_inputs_fail_explicitly(engine):
    assert engine.get_drug_state("glyco") == {"rate": 0.0, "target": 0.0, "is_tci": False}
    with pytest.raises(ValueError, match="Glycopyrrolate has no infusion"):
        engine.set_drug_rate("glyco", 1.0)
    with pytest.raises(ValueError, match="Esmolol has no target-controlled infusion"):
        engine.set_drug_target("esmolol", 1.0)

    tci_eng = SimulationEngine(
        Patient(age=45, weight=70, height=170, sex="male"),
        SimulationConfig(mode="awake", tci_enabled=True, dt=0.5),
    )
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            tci_eng.set_drug_rate("propofol", bad)
        with pytest.raises(ValueError):
            tci_eng.enable_tci("propofol", bad)
        with pytest.raises(ValueError):
            tci_eng.set_drug_target("propofol", bad)

    assert math.isnan(clamp(math.nan, 0.0, 1.0))
    assert clamp01(math.nan) == 0.0
    assert get_drug_spec("milri").max_bolus >= 5000.0
    assert get_drug_spec("vaso").max_bolus <= 50.0
    assert get_drug_spec("amiodarone").default_bolus == 150.0


def test_rate_unit_aliases_zero_tci_target_and_cirrhosis_startup_ce(engine_factory):
    """Rate converter normalizes /kg/ and hr aliases; enable_tci(..., 0) disables TCI;
    manual startup uses nominal PK at current CO.
    """
    assert convert_rate(60.0, "mg/kg/hr", "mg/sec", weight_kg=70.0) == pytest.approx(70.0 / 60.0)
    assert convert_rate(70.0 / 60.0, "mg/sec", "mg/kg/h", weight_kg=70.0) == pytest.approx(60.0)
    assert convert_rate(3.6, "mg/hr", "ug/sec") == pytest.approx(1.0)

    tci_eng = SimulationEngine(
        Patient(age=45, weight=70, height=170, sex="male"),
        SimulationConfig(mode="awake", tci_enabled=True, dt=0.5),
    )
    tci_eng.enable_tci("propofol", 3.0)
    assert tci_eng.get_drug_state("propofol")["is_tci"] is True
    tci_eng.enable_tci("propofol", 0.0)
    assert tci_eng.get_drug_state("propofol") == {"rate": 0.0, "target": 0.0, "is_tci": False}

    cirr = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="tiva", tci_enabled=False, rng_seed=1),
        start=True, age=55, weight=70, height=172, sex="male", hepatic_function=0.5, heart_failure="hfref",
    )
    assert 40.0 <= cirr.state.bis <= 58.0
