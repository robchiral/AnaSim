"""Integrated responses of the reduced patient-condition models."""

import copy
import math
from dataclasses import asdict

import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.enums import RhythmType
from anasim.core.state import AirwayType, SimulationConfig
from anasim.monitors.capno import Capnograph
from anasim.patient.patient import Patient
from anasim.patient.pk_models import (
    EtomidatePK,
    GlycopyrrolatePK,
    MilrinonePK,
    MorphinePK,
    RocuroniumPK,
)
from anasim.patient.volatile_pk import VolatilePK
from anasim.physiology.respiration import RespiratoryModel


def _engine(**patient):
    engine = SimulationEngine(Patient(**patient), SimulationConfig(mode="awake", dt=1.0, rng_seed=1))
    engine.start()
    return engine


def _advance(engine, seconds, dt=1.0):
    for _ in range(int(seconds / dt)):
        engine.step(dt)


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

    overload = SimulationEngine(
        Patient(heart_failure="hfref"), SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1)
    )
    overload.start()
    overload.set_fgf(0.2, 1.8)
    for _ in range(60):
        overload.step(1.0)
    co_before = overload.state.co
    overload.give_fluid(3000.0)
    for _ in range(600):
        overload.step(2.0)
    assert not overload.state.cardiac_arrest
    assert overload.state.co <= co_before
    assert overload.state.lung_water > 4.0 and overload.state.spo2 < 95.0


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


def test_copd_controlled_and_high_flow_oxygen_responses():
    """Wagner 1977, DeGaute 1981, and Robinson 2000 COPD oxygen responses.

    Advanced COPD breathing air has PaO2 38-71 mmHg from V/Q mismatch rather
    than shunt, so supplemental O2 corrects hypoxemia (Wagner 1977). Controlled
    oxygen (FiO2 0.28) raises PaO2 and PvO2 with a modest PaCO2 rise and pH fall
    (DeGaute's cardiac-index drop from 3.9 to 3.4 L/min/m^2 in moderate hypoxemia
    vs 4.0 unchanged at PaO2 40 mmHg is not modeled). On 100% O2, hypercapnic
    retention and wasted ventilation increase when severe exacerbation lowers
    room-air PaO2 and narrows tidal volume toward dead space.
    """
    exac = _engine(age=64, weight=68, height=170, sex="male", copd_severity=1.0)
    exac.set_bronchospasm(0.45)
    exac.set_airway_mode(AirwayType.MASK)
    exac.set_fgf(0.0, 15.0)
    _advance(exac, 600, dt=1.0)
    air_pao2 = exac.state.pao2
    air_pvo2 = exac.resp.state.pv_o2
    air_paco2 = exac.state.pa_co2
    air_ph = exac.state.ph

    o2_flow_28 = 15.0 * (0.28 - 0.21) / 0.79
    exac.set_fgf(o2_flow_28, 15.0 - o2_flow_28)
    _advance(exac, 2400, dt=1.0)
    assert 48.0 < air_pao2 < 60.0 and air_paco2 > 45.0
    assert 68.0 < exac.state.pao2 < 82.0
    assert 4.0 < exac.resp.state.pv_o2 - air_pvo2 < 9.0
    assert 2.0 < exac.state.pa_co2 - air_paco2 < 6.0
    assert -0.045 < exac.state.ph - air_ph < -0.015

    # 100% O2 (>= 20 min): mild hypoxemia vs severe hypoxemic exacerbation.
    deltas = {}
    wasted_shifts = {}
    for label, bronch in (("mild", 0.0), ("severe_exac", 0.55)):
        eng = _engine(age=64, weight=68, height=170, sex="male", copd_severity=1.0)
        eng.set_bronchospasm(bronch)
        eng.set_airway_mode(AirwayType.MASK)
        eng.set_fgf(0.0, 15.0)
        step_dt = 1.0 if bronch > 0.0 else 2.0
        _advance(eng, 600, dt=step_dt)
        if label == "mild":
            assert 38.0 < eng.state.pao2 < 71.0
        paco2_air = eng.state.pa_co2
        wasted_air = 1.0 - eng.resp.state.va / eng.resp.state.mv

        eng.set_fgf(15.0, 0.0)
        _advance(eng, 1200, dt=step_dt)
        assert eng.state.pao2 > 450.0
        deltas[label] = eng.state.pa_co2 - paco2_air
        wasted_shifts[label] = (1.0 - eng.resp.state.va / eng.resp.state.mv) - wasted_air

    assert deltas["mild"] <= 3.0
    assert 5.0 < deltas["severe_exac"] < 10.0
    assert wasted_shifts["severe_exac"] > 0.01 > wasted_shifts["mild"]


def _gradient_kpa(state):
    """Room-air A-a gradient as Lindberg 1992 calculated it, with R = 0.8."""
    return (0.21 * (760.0 - 47.0) - state.pa_co2 / 0.8 - state.pao2) / 7.50062


def _postoperative_gradient_kpa(engine_factory, site):
    """A-a gradient 30 min after extubation to room air, following emergence on 100% O2."""
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1),
        age=68, surgical_site=site, start=True,
    )
    engine.set_vaporizer(0.0)
    engine.set_drug_rate("remi", 0.0)
    engine.set_fgf(10.0, 0.0)
    engine.give_drug_bolus("sugammadex", 2.0 * engine.patient.weight)
    for _ in range(900):
        if engine.state.bis > 70.0 and engine.resp.state.mv > 1.0:
            engine.set_vent_power(False)
        if engine.state.bis > 80.0 and engine.resp.state.mv > 3.0:
            break
        engine.step(2.0)
    else:
        raise AssertionError("Patient did not meet extubation criteria")
    engine.set_airway_mode("None")
    for _ in range(900):
        engine.step(2.0)
    return _gradient_kpa(engine.state)


def test_abdominal_surgery_keeps_airway_closure_after_emergence(engine_factory):
    """Lindberg 1992, colon surgery at age 68 (n = 13): the A-a gradient was
    1.8 +/- 1.7 kPa awake and 3.7 +/- 1.9 kPa 2 h after surgery. The gradient
    excludes the PaCO2 rise from postoperative opioids. Bands are 2 SE. After
    upper abdominal surgery, FRC is at or below its anesthetized value (Wahba
    1975), so the gradient widens further. The site has no effect before
    anesthesia.
    """
    awake = []
    for site in ("other", "upper_abdominal"):
        engine = _engine(age=68, surgical_site=site)
        _advance(engine, 30, dt=2.0)
        awake.append(_gradient_kpa(engine.state))
    assert awake[0] == awake[1] == pytest.approx(1.8, abs=2 * 1.7 / math.sqrt(13))
    gradients = {site: _postoperative_gradient_kpa(engine_factory, site)
                 for site in ("other", "lower_abdominal", "upper_abdominal")}
    assert gradients["lower_abdominal"] == pytest.approx(3.7, abs=2 * 1.9 / math.sqrt(13))
    assert gradients["other"] < gradients["lower_abdominal"] < gradients["upper_abdominal"]


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
    assert pregnant.state.base_excess == pytest.approx(-2.2, abs=0.2)
    assert 0.8 < pregnant.resp.frc / control.resp.frc < 0.95
    assert 1.25 < pregnant.state.blood_volume / control.state.blood_volume < 1.6
    assert 1.15 < pregnant.hemo.fluids.ecf_l / control.hemo.fluids.ecf_l < 1.25
    assert 1.08 < pregnant.hemo.fluids.interstitial_l / control.hemo.fluids.interstitial_l < 1.20
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
    # Plateau and dynamic compliance are blank when breathing effort coincides with a mandatory breath.
    unmeasured = {"paw_plat", "compliance_dyn"}
    assert all(math.isfinite(value) for key, value in asdict(state).items()
               if isinstance(value, float) and key not in unmeasured)
    assert 20 < state.map < 160 and 0 < state.co < 20
    assert 25 < state.pa_co2 < 50 and 7.25 < state.ph < 7.55
    assert 30 < state.bis < 65 and state.sao2 > 94


def test_hypertensive_vs_normotensive_propofol_fentanyl_intubation_matches_sawano_2013():
    """Sawano 2013 (n = 90, ages 65-67): under propofol 4 mcg/mL TCI, controlled
    hypertensive patients (MAP ~120 mmHg) have larger absolute post-induction MAP
    drops and higher post-intubation peaks than normotensive patients (MAP ~98
    mmHg); fentanyl 2 mcg/kg keeps normotensive peak MAP near baseline whereas
    hypertensives require 4 mcg/kg.
    """
    results: dict[tuple[str, float], tuple[float, float]] = {}
    for label, base_map, base_hr in (("normo", 98.0, 74.0), ("htn", 120.0, 75.0)):
        prefix = SimulationEngine(
            Patient(age=65, weight=57, height=160, sex="male", baseline_map=base_map, baseline_hr=base_hr),
            SimulationConfig(mode="awake", tci_enabled=True, dt=1.0, rng_seed=1),
        )
        prefix.start()
        prefix.set_airway_mode("Mask")
        prefix.set_fgf(10.0, 0.0)
        prefix.enable_tci("propofol", 4.0, mode="plasma")
        for _ in range(60):
            prefix.step(1.0)
        prefix.give_drug_bolus("roc", 0.9 * 57.0)
        prefix.set_bag_mask_ventilation(True, rr=12, vt=0.45, p_max=20.0)
        for _ in range(60):
            prefix.step(1.0)
        for fent_ug_kg in (0.0, 2.0, 4.0):
            engine = copy.deepcopy(prefix)
            if fent_ug_kg > 0.0:
                engine.give_drug_bolus("fentanyl", fent_ug_kg * 57.0)
            for _ in range(120):
                engine.step(1.0)
            pre_intub_drop = base_map - engine.state.map

            engine.set_bag_mask_ventilation(False)
            engine.set_airway_mode("ETT")
            engine.set_vent_settings(mode="VCV", vt=0.45, rr=12, peep=5, ie="1:2")
            engine.set_vent_power(True)
            engine.start_disturbance("stim_intubation_pulse")
            peak_map = 0.0
            for _ in range(120):
                engine.step(1.0)
                peak_map = max(peak_map, engine.state.map)
            results[(label, fent_ug_kg)] = (pre_intub_drop, peak_map)

    for fent_ug_kg in (0.0, 2.0, 4.0):
        assert results[("htn", fent_ug_kg)][0] > results[("normo", fent_ug_kg)][0] + 5.0
        assert results[("htn", fent_ug_kg)][1] > results[("normo", fent_ug_kg)][1] + 10.0
    assert results[("normo", 0.0)][1] > results[("normo", 2.0)][1] > results[("normo", 4.0)][1]
    assert results[("htn", 0.0)][1] > results[("htn", 2.0)][1] > results[("htn", 4.0)][1]
    assert results[("normo", 2.0)][1] == pytest.approx(103.0, abs=18.0)
    assert results[("htn", 0.0)][1] == pytest.approx(144.0, abs=19.0)
    assert results[("htn", 4.0)][1] < 115.0


def test_ketamine_induction_in_atherosclerotic_disease_and_labetalol_blunting():
    """Bålfors 1983 (control arm, n = 8 atherosclerotic disease without
    droperidol): ketamine 2 mg/kg over 1 min on FiO2 0.25 raises MAP (103 ->
    146 +/- 25.5 SD) and HR (73 -> 84 +/- 11.3 SD) at 2-3 min without ischemia,
    and combined alpha/beta blockade with labetalol blunts both surges.
    """
    control = SimulationEngine(
        Patient(age=57, weight=75, height=175, sex="male", baseline_map=103.0,
                baseline_hr=73.0, coronary_flow_reserve=2.0),
        SimulationConfig(mode="awake", dt=1.0, rng_seed=1),
    )
    control.start()
    control.set_airway_mode("Mask")
    control.set_fgf(0.5, 9.5)  # FiO2 ~0.25
    blocked = copy.deepcopy(control)
    blocked.give_drug_bolus("labetalol", 40.0)
    for eng in (control, blocked):
        for _ in range(60):
            eng.step(2.0)
        eng.set_drug_rate("ketamine", 2.0 * 60.0)
        for _ in range(30):
            eng.step(2.0)
        eng.set_drug_rate("ketamine", 0.0)
        for _ in range(75):
            eng.step(2.0)

    assert control.state.map == pytest.approx(146.0, abs=2.0 * 25.5)
    assert control.state.map > 120.0
    assert control.state.hr == pytest.approx(84.0, abs=2.0 * 11.3)
    assert control.state.hr > 80.0
    assert control.state.coronary_supply_demand_ratio > 1.0
    assert control.state.myocardial_ischemia < 0.01
    assert blocked.state.map < control.state.map - 15.0
    assert blocked.state.hr < control.state.hr - 8.0


def test_low_vs_normal_cardiac_output_propofol_fentanyl_matches_sherry_1995():
    """Sherry 1995 (n = 39 cardiac surgery): propofol 8 mg/kg/h to LOC followed
    by 4 mg/kg/h (T2) and fentanyl 15 mcg/kg (T3) lowers MAP in both normal- and
    low-output patients, while elevated resting filling pressure in low-output
    patients preserves stroke volume and attenuates the fractional fall in CI.
    """
    outcomes = {}
    for condition, base_map, base_hr in (("none", 98.1, 70.2), ("hfref", 93.8, 63.8)):
        engine = SimulationEngine(
            Patient(age=60, weight=75, height=172, sex="male", heart_failure=condition,
                    baseline_map=base_map, baseline_hr=base_hr),
            SimulationConfig(mode="awake", dt=1.0, rng_seed=1),
        )
        engine.start()
        engine.set_airway_mode("Mask")
        engine.set_fgf(10.0, 0.0)
        for _ in range(30):
            engine.step(1.0)
        bsa = engine.patient.bsa
        base = (engine.state.map, engine.state.co / bsa, engine.state.sv)

        engine.set_drug_rate("propofol", 8000.0 / 60.0)
        t_loc = 0
        while engine.state.loc < 0.5 and t_loc < 600:
            engine.step(1.0)
            t_loc += 1
        assert 150 <= t_loc <= 420

        engine.set_drug_rate("propofol", 4000.0 / 60.0)
        engine.set_bag_mask_ventilation(True, rr=12, vt=0.50, p_max=20.0)
        for _ in range(180):
            engine.step(1.0)
        t2 = (engine.state.map, engine.state.co / bsa, engine.state.sv)

        engine.give_drug_bolus("fentanyl", 7.5 * 75.0)
        for _ in range(60):
            engine.step(1.0)
        engine.give_drug_bolus("fentanyl", 7.5 * 75.0)
        for _ in range(120):
            engine.step(1.0)
        t3 = (engine.state.map, engine.state.co / bsa, engine.state.sv)
        outcomes[condition] = (
            t2[0] / base[0] - 1.0,
            t2[1] / base[1] - 1.0,
            t2[2] / base[2] - 1.0,
            t3[0] / base[0] - 1.0,
            t3[1] / base[1] - 1.0,
            t3[2] / base[2] - 1.0,
        )

    norm, low = outcomes["none"], outcomes["hfref"]
    assert -0.30 < norm[0] < -0.05 and -0.30 < low[0] < -0.05
    assert -0.35 < norm[3] < -0.10 and -0.35 < low[3] < -0.10
    # Low-CO patients lose less fractional SV and CI than normal-CO patients.
    assert low[2] > norm[2]
    assert low[1] > norm[1]
    assert low[5] > norm[5] + 0.04
    assert low[4] > norm[4] + 0.03


def test_end_stage_renal_disease_propofol_infusion_matches_ickx_1998():
    """Ickx 1998 (n = 11 ESRD vs n = 9 controls): 3-stage propofol infusion
    (21 -> 12 -> 6 mg/kg/h) with fentanyl 100 mcg and 60% N2O yields unchanged
    steady-state propofol Cp (~3.1-3.5 mcg/mL), total body clearance, and
    maintenance MAP in ESRD, with prompt emergence after infusion stop.
    """
    results = {}
    for group, wt, ht, hb, rf, crcl in (
        ("control", 72.6, 171.6, 14.2, 1.0, 102.8),
        ("esrd", 62.3, 164.0, 9.9, 0.4, 7.3),
    ):
        engine = SimulationEngine(
            Patient(age=45.0, weight=wt, height=ht, sex="female", baseline_hb=hb,
                    renal_function=rf, creatinine_clearance_ml_min=crcl),
            SimulationConfig(mode="awake", dt=1.0, rng_seed=1),
        )
        engine.start()
        engine.set_airway_mode("Mask")
        engine.set_fgf(10.0, 0.0)
        for _ in range(30):
            engine.step(1.0)
        engine.give_drug_bolus("fentanyl", 100.0)
        engine.set_drug_rate("propofol", 350.0)
        for _ in range(300):
            engine.step(1.0)
        engine.give_drug_bolus("roc", 0.6 * wt)
        engine.set_airway_mode("ETT")
        engine.set_fgf(2.0, 0.0, 3.0)
        engine.set_vent_settings(mode="VCV", vt=0.48, rr=11, peep=5, ie="1:2")
        engine.set_vent_power(True)
        engine.set_drug_rate("propofol", 200.0)
        for _ in range(300):
            engine.step(2.0)
        engine.set_drug_rate("propofol", 100.0)
        for _ in range(1500):
            engine.step(2.0)
        cp_maint = engine.state.propofol_cp
        cl_ml_kg_min = engine.pk["propofol"].elimination_clearance * 1000.0 / wt
        map_maint = engine.state.map

        engine.set_drug_rate("propofol", 0.0)
        engine.set_fgf(8.0, 0.0, 0.0)
        wake_s = None
        for elapsed in range(2, 1801, 2):
            engine.step(2.0)
            if engine.state.loc < 0.5:
                wake_s = elapsed
                break
        assert wake_s is not None
        results[group] = (cp_maint, cl_ml_kg_min, map_maint, wake_s)

    ctrl, esrd = results["control"], results["esrd"]
    assert ctrl[0] == pytest.approx(3.11, abs=0.74)
    assert esrd[0] == pytest.approx(3.49, abs=0.97)
    assert esrd[0] == pytest.approx(ctrl[0], rel=0.10)
    assert esrd[1] == pytest.approx(ctrl[1], rel=0.10)
    assert esrd[2] == pytest.approx(ctrl[2], abs=4.0)
    assert 400 <= esrd[3] <= 1200 and 400 <= ctrl[3] <= 1200


def test_severe_copd_does_not_outlast_healthy_during_preoxygenated_apnea():
    """Trapped COPD hyperinflation does not double exchangeable alveolar O2 reserve."""
    times = []
    for copd in (0.0, 1.0):
        model = RespiratoryModel(Patient(age=40, weight=70, height=170, copd_severity=copd))
        model.equilibrate_oxygen(1.0)
        sec = 0
        while model.state.sao2 >= 90.0 and sec < 900:
            model.step(1.0, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0, airway_patency=0.0, unconscious=1.0)
            sec += 1
        times.append(sec)
    assert times[1] < times[0]


def test_hfref_anaphylaxis_unloading_cap_inodilator_lap_and_propofol_low_co(engine_factory, advance_time):
    """HFrEF caps distributive-shock unloading, lowers LAP with inodilators, and raises propofol Cp."""
    anaph = engine_factory(
        config=SimulationConfig(mode="steady_state", rng_seed=1),
        start=True,
        heart_failure="hfref",
    )
    base_co = anaph.state.co
    anaph.start_anaphylaxis()
    advance_time(anaph, 180.0, dt=1.0)
    assert anaph.state.co < 1.45 * base_co

    hf_ctrl = engine_factory(
        config=SimulationConfig(mode="steady_state", rng_seed=1),
        start=True,
        heart_failure="hfref",
    )
    hf_ctrl.give_fluid(1000.0)
    advance_time(hf_ctrl, 120.0, dt=1.0)
    hf = copy.deepcopy(hf_ctrl)
    hf.give_drug_bolus("milri", 3500.0)
    hf.set_drug_rate("milri", 0.5)
    hf.set_drug_rate("dobu", 5.0)
    advance_time(hf_ctrl, 240.0, dt=1.0)
    advance_time(hf, 240.0, dt=1.0)
    assert hf.state.lap < hf_ctrl.state.lap - 4.0
    assert hf.state.co > hf_ctrl.state.co + 0.5

    norm = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, heart_failure="none")
    hfref = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, heart_failure="hfref")
    for eng in (norm, hfref):
        eng.set_airway_mode("ETT")
        eng.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        eng.set_vent_power(True)
        eng.set_fgf(4.0, 0.0)
        eng.set_drug_rate("propofol", 100.0)
    advance_time(norm, 1200.0, dt=2.0)
    advance_time(hfref, 1200.0, dt=2.0)
    assert hfref.state.propofol_cp > norm.state.propofol_cp * 1.06
    assert hfref.state.bis < norm.state.bis - 1.5


def test_pregnancy_mac_v1_expansion_and_esrd_clearance_propagation():
    """Term pregnancy lowers MAC ~28% and expands central V1 (Gin & Chan 1994; Dallmann 2017);
    ESRD creatinine_clearance_ml_min propagates to renal_function and renally cleared drugs."""
    nonpreg = Patient(age=30, weight=65, height=165, sex="female", gestational_age_weeks=0.0)
    preg = Patient(age=30, weight=65, height=165, sex="female", gestational_age_weeks=38.0)
    mac_ratio = (
        VolatilePK(preg, "Sevo", 0.65, mac_40=2.0).mac_age
        / VolatilePK(nonpreg, "Sevo", 0.65, mac_40=2.0).mac_age
    )
    assert 0.68 <= mac_ratio <= 0.76
    assert RocuroniumPK(preg).v1 > RocuroniumPK(nonpreg).v1 * 1.15
    assert EtomidatePK(preg).v1 > EtomidatePK(nonpreg).v1 * 1.15

    normal = Patient(age=45, weight=70, height=170, sex="male")
    esrd = Patient(age=45, weight=70, height=170, sex="male", creatinine_clearance_ml_min=5.0)
    assert esrd.renal_function == pytest.approx(0.05)
    assert esrd.renal_status == "Severe"
    assert MilrinonePK(esrd).cl1 < 0.25 * MilrinonePK(normal).cl1
    assert GlycopyrrolatePK(esrd).cl1 < 0.45 * GlycopyrrolatePK(normal).cl1
    assert MorphinePK(esrd).cl1 < 0.75 * MorphinePK(normal).cl1
