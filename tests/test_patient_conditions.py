"""Integrated responses of the reduced patient-condition models."""

import math
from dataclasses import asdict

import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.enums import RhythmType
from anasim.core.state import AirwayType, SimulationConfig
from anasim.monitors.capno import Capnograph
from anasim.patient.patient import Patient


def _engine(**patient):
    engine = SimulationEngine(Patient(**patient), SimulationConfig(mode="awake", dt=0.5, rng_seed=1))
    engine.start()
    return engine


def _advance(engine, seconds):
    for _ in range(int(seconds / 0.5)):
        engine.step(0.5)


def _ventilate(engine, rr=12):
    engine.set_airway_mode(AirwayType.ETT)
    engine.set_fgf(5.0, 0.0)
    engine.set_vent_settings(rr=rr, vt=0.5, peep=5, ie="1:2", mode="VCV")
    engine.set_vent_power(True)


def test_limited_coronary_reserve_causes_reversible_ischemia_during_normoxic_tachycardia():
    engines = [_engine(coronary_flow_reserve=reserve, baseline_rr=0) for reserve in (5, 1.3)]
    for engine in engines:
        _ventilate(engine)
        _advance(engine, 30)
        assert engine.state.myocardial_ischemia < 0.01
        engine.set_rhythm(RhythmType.SVT.value)
        _advance(engine, 90)
        assert engine.state.sao2 > 98
    normal, coronary = engines
    peak = coronary.state.myocardial_ischemia
    assert normal.state.myocardial_ischemia < 0.01
    assert peak > 0.25
    assert coronary.state.coronary_supply_demand_ratio < 1
    assert coronary.state.st_depression_mv > 0.075
    assert coronary.state.co < 0.90 * normal.state.co
    coronary.set_rhythm(RhythmType.SINUS.value)
    _advance(coronary, 180)
    assert coronary.state.myocardial_ischemia < 0.15 * peak
    assert coronary.state.co > 0.9 * coronary.hemo.base_co_l_min

    anemic = _engine(coronary_flow_reserve=1.3, baseline_hb=8, baseline_rr=0)
    _ventilate(anemic)
    _advance(anemic, 90)
    assert anemic.state.sao2 > 98
    assert anemic.state.myocardial_ischemia > 0.15


def test_heart_failure_profiles_change_output_afterload_response_and_fluid_tolerance():
    output, pressure, water, pressor_output = {}, {}, {}, {}
    for failure in ("none", "hfref", "hfpef"):
        engine = _engine(heart_failure=failure)
        _advance(engine, 30)
        # Chronic filling pressure alone does not flood the lungs.
        assert engine.state.lung_water == 0
        output[failure] = engine.state.co
        engine.give_drug_bolus("Phenylephrine", 200)
        _advance(engine, 60)
        pressor_output[failure] = engine.state.co / output[failure]

        engine = _engine(heart_failure=failure)
        engine.give_fluid(1000)
        _advance(engine, 300)
        pressure[failure], water[failure] = engine.state.lap, engine.state.lung_water
    assert output["hfref"] < output["hfpef"] < output["none"]
    assert pressor_output["hfref"] < pressor_output["hfpef"]
    assert pressure["hfpef"] > 1.5 * pressure["none"]
    assert pressure["hfref"] > 25
    assert water["none"] == 0
    assert water["hfref"] > water["hfpef"] > 0.1


def test_absent_coronary_perfusion_causes_ischemia_despite_oxygenated_blood():
    engine = _engine(baseline_rr=0)
    _ventilate(engine)
    _advance(engine, 60)
    engine.set_rhythm(RhythmType.ASYSTOLE.value)
    _advance(engine, 60)
    assert engine.state.co == 0
    assert engine.state.sao2 > 98
    assert engine.state.myocardial_ischemia > 0.9
    engine.set_rhythm(RhythmType.SINUS.value)
    _advance(engine, 180)
    assert engine.state.myocardial_ischemia < 0.1
    assert engine.state.co > 0.9 * engine.hemo.base_co_l_min


def test_chronic_low_output_changes_flow_sensitive_drug_exposure():
    normal, failure = (_engine(heart_failure=condition, baseline_rr=0) for condition in ("none", "hfref"))
    for engine in (normal, failure):
        _ventilate(engine)
        engine.give_drug_bolus("Fentanyl", 100)
        _advance(engine, 120)
        assert engine.state.sao2 > 98
    assert failure.state.fentanyl_cp > 1.15 * normal.state.fentanyl_cp


def test_copd_has_persistent_obstruction_gas_trapping_and_wasted_ventilation():
    normal, copd = (_engine(copd_severity=severity, baseline_rr=0) for severity in (0, 1))
    for engine in (normal, copd):
        _ventilate(engine, rr=24)
        _advance(engine, 120)
        assert engine.state.vt == pytest.approx(500, abs=5)
    assert copd.resp.frc > 1.5 * normal.resp.frc
    assert copd.vent.monitors.auto_peep > normal.vent.monitors.auto_peep + 3
    assert copd.state.paw_peak > normal.state.paw_peak + 5
    assert copd.state.pa_co2 > normal.state.pa_co2 + 2
    # Acute bronchospasm resolves; the chronic obstruction remains.
    copd.set_bronchospasm(0.8)
    _advance(copd, 10)
    copd.set_bronchospasm(0)
    _advance(copd, 30)
    assert copd.state.bronchospasm == 0
    assert copd.resp_mech.resistance > 1.5 * normal.resp_mech.resistance

    # Replay the expired gas through an analyzer without obstruction. COPD
    # delays the CO2 rise for the same flow and end-tidal value.
    analyzer = Capnograph(copd.capno.dead_space)
    copd.capno.reset()
    differences = []
    for _ in range(120):
        copd.step(0.1)
        for duration, change, _, _, _ in copd.vent.samples:
            analyzer.step(duration, change, copd.resp.state.etco2)
        differences.append(analyzer.co2 - copd.state.capno_co2)
    assert copd.state.etco2_signal_valid
    assert max(differences) > 5


def test_copd_startup_calculates_tidal_volume_at_the_limited_rate():
    engine = SimulationEngine(Patient(copd_severity=1, baseline_rr=30),
                              SimulationConfig(mode="steady_state", dt=0.5, rng_seed=1))
    engine.start()
    _advance(engine, 120)
    assert engine.vent.settings.rr == 18
    assert 38 < engine.state.pa_co2 < 42
    assert 7.38 < engine.state.ph < 7.42


def test_pregnancy_changes_resting_gases_and_circulation_and_responds_to_position():
    control = _engine(sex="female")
    pregnant = _engine(sex="female", gestational_age_weeks=36, position="supine")
    lateral = _engine(sex="female", gestational_age_weeks=36, position="left_lateral")
    for engine in (control, pregnant, lateral):
        _advance(engine, 60)
    assert 30 < pregnant.state.pa_co2 < 34
    assert 7.42 < pregnant.state.ph < 7.46
    assert 19 < pregnant.state.hco3 < 24
    assert 0.8 < pregnant.resp.frc / control.resp.frc < 0.95
    assert 1.25 < pregnant.state.blood_volume / control.state.blood_volume < 1.6
    assert lateral.state.co > 1.3 * control.state.co
    assert 0.8 < pregnant.state.co / lateral.state.co < 0.95
    # Hemoglobin is the entered starting measurement, not diluted a second time.
    assert pregnant.state.hb_g_dl == pytest.approx(control.state.hb_g_dl, abs=0.05)
    before = pregnant.state.co
    pregnant.set_position("left_lateral")
    _advance(pregnant, 60)
    assert pregnant.state.co > 1.1 * before
    assert pregnant.state.co == pytest.approx(lateral.state.co, rel=0.04)


def test_severe_obesity_reduces_respiratory_reserve_at_the_same_height():
    normal, obese = (_engine(weight=weight, baseline_rr=0) for weight in (70, 140))
    assert obese.patient.predicted_body_weight() == normal.patient.predicted_body_weight()
    for engine in (normal, obese):
        _advance(engine, 45)
    assert obese.resp.frc < 0.5 * normal.resp.frc
    assert obese.resp_mech.reference_compliance < 0.75 * normal.resp_mech.reference_compliance
    assert obese.state.sao2 < normal.state.sao2 - 15


@pytest.mark.parametrize("technique", ["tiva", "balanced"])
def test_combined_conditions_start_and_run_with_finite_drug_and_physiology_outputs(technique):
    patient = Patient(sex="female", weight=140, gestational_age_weeks=36,
                      heart_failure="hfpef", copd_severity=0.65, coronary_flow_reserve=2)
    engine = SimulationEngine(patient, SimulationConfig(mode="steady_state", maint_type=technique, dt=0.5, rng_seed=1))
    engine.start()
    _advance(engine, 30)
    state = engine.state
    assert engine.vent.settings.tv <= 8.0 * patient.predicted_body_weight()
    assert engine.vent.settings.rr <= 18.0
    assert all(math.isfinite(value) for value in asdict(state).values() if isinstance(value, float))
    assert 20 < state.map < 160 and 0 < state.co < 20
    assert 25 < state.pa_co2 < 50 and 7.25 < state.ph < 7.55
    assert 30 < state.bis < 65 and state.sao2 > 94
