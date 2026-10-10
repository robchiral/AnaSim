import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig, SimulationState
from anasim.patient.domain import (
    AGE_RANGE_YEARS,
    HEIGHT_RANGE_CM,
    HEMATOCRIT_RANGE,
    HEMOGLOBIN_RANGE_G_DL,
    HEPATIC_FUNCTION_RANGE,
    RENAL_FUNCTION_RANGE,
    WEIGHT_RANGE_KG,
)
from anasim.patient.patient import Patient

BOUNDARY_PATIENTS = (
    {
        "age": AGE_RANGE_YEARS[0],
        "weight": WEIGHT_RANGE_KG[0],
        "height": HEIGHT_RANGE_CM[0],
        "sex": "female",
        "baseline_hb": HEMOGLOBIN_RANGE_G_DL[0],
        "baseline_hct": HEMATOCRIT_RANGE[0],
        "renal_function": RENAL_FUNCTION_RANGE[0],
        "hepatic_function": HEPATIC_FUNCTION_RANGE[0],
    },
    {
        "age": AGE_RANGE_YEARS[1],
        "weight": WEIGHT_RANGE_KG[1],
        "height": 177.0,
        "sex": "male",
        "baseline_hb": HEMOGLOBIN_RANGE_G_DL[1],
        "baseline_hct": HEMATOCRIT_RANGE[1],
        "renal_function": RENAL_FUNCTION_RANGE[1],
        "hepatic_function": HEPATIC_FUNCTION_RANGE[1],
    },
    {"age": 40, "weight": 72.0, "height": HEIGHT_RANGE_CM[1], "sex": "male"},
    {"age": 40, "weight": 72.0, "height": HEIGHT_RANGE_CM[0], "sex": "female"},
)


def _run(
    patient: Patient,
    config: SimulationConfig,
    boluses: tuple[tuple[str, float], ...],
    duration_s: float,
) -> tuple[SimulationEngine, SimulationState]:
    engine = SimulationEngine(patient, config)
    engine.start()
    baseline = engine.get_latest_state()
    for drug, dose in boluses:
        engine.give_drug_bolus(drug, dose)
    for _ in range(int(duration_s / config.dt)):
        engine.step(config.dt)
    return engine, baseline


def _assert_plausible_integrated_state(engine: SimulationEngine) -> None:
    assert 20.0 <= engine.state.hr <= 220.0
    assert 10.0 <= engine.state.map <= 220.0
    assert 0.0 < engine.state.co <= 20.0
    assert 0.0 <= engine.state.bis <= 100.0
    assert 0.0 <= engine.state.loc <= 1.0
    assert 0.0 <= engine.state.spo2 <= 100.0
    assert 0.0 < engine.state.hb_g_dl <= 25.0
    assert 0.0 < engine.state.hct <= 0.70
    assert engine.state.propofol_cp >= 0.0
    assert engine.state.propofol_ce >= 0.0


def test_supported_patient_boundaries_run_through_integrated_model():
    for patient_kwargs in BOUNDARY_PATIENTS:
        patient = Patient(**patient_kwargs)
        config = SimulationConfig(mode="awake", dt=0.5, rng_seed=1)
        engine, baseline = _run(
            patient,
            config,
            (("Propofol", 1.5 * patient.weight),),
            duration_s=60.0,
        )

        _assert_plausible_integrated_state(engine)
        assert engine.state.propofol_ce > 0.0
        assert engine.state.bis < baseline.bis


@pytest.mark.parametrize("concomitant_opioids", [False, True])
def test_eleveld_regimens_run(concomitant_opioids):
    patient = Patient()
    config = SimulationConfig(mode="awake", dt=0.5, concomitant_opioids=concomitant_opioids, rng_seed=1)
    engine, baseline = _run(patient, config, (("Propofol", 1.5 * patient.weight),), duration_s=60.0)
    _assert_plausible_integrated_state(engine)
    assert engine.state.propofol_ce > 0.0
    assert engine.state.bis < baseline.bis


def test_vasoactive_models_run():
    patient = Patient()
    config = SimulationConfig(mode="awake", dt=0.5, rng_seed=1)
    engine, baseline = _run(patient, config, (("nore", 10.0), ("epi", 10.0)), duration_s=30.0)
    assert engine.state.nore_ce > 0.0
    assert engine.state.epi_ce > 0.0
    _assert_plausible_integrated_state(engine)
    assert engine.state.hr > baseline.hr + 5.0
    assert engine.state.co > baseline.co


def test_invalid_patient_values_are_rejected():
    invalid_values = (
        ({"age": float("nan")}, "age"),
        ({"age": 17.0}, "age"),
        ({"age": 71.0}, "age"),
        ({"weight": 49.9}, "weight"),
        ({"height": "unknown"}, "height"),
        ({"height": 200.1}, "height"),
        ({"sex": "other"}, "sex"),
        ({"baseline_hb": 5.9}, "baseline_hb"),
        ({"baseline_hct": 0.61}, "baseline_hct"),
        ({"baseline_hb": 8.0, "baseline_hct": 0.42}, "grossly inconsistent"),
        ({"renal_function": 0.39}, "renal_function"),
        ({"hepatic_function": 1.01}, "hepatic_function"),
        ({"weight": 50.0, "height": 200.0}, "bmi"),
    )
    for patient_kwargs, field_name in invalid_values:
        with pytest.raises(ValueError, match=field_name):
            Patient(**patient_kwargs)


def test_invalid_simulation_config_is_rejected():
    invalid_values = (
        ({"dt": float("nan")}, "dt"),
        ({"concomitant_opioids": "false"}, "concomitant_opioids"),
        ({"sevoflurane_enabled": "false"}, "sevoflurane_enabled"),
        ({"maint_type": "balanced", "sevoflurane_enabled": False}, "sevoflurane_enabled"),
    )
    for config_kwargs, field_name in invalid_values:
        with pytest.raises(ValueError, match=field_name):
            SimulationConfig(**config_kwargs)
