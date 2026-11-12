"""Epinephrine calibration against published cohort responses.

Infusion: Freyschuss 1986 (arterial samples, basal 0.27 nmol/L, mean age 24),
Ensinger 1992 (arterial concentration-response through 23.9 nmol/L),
Esler 1995 (brachial arterial tracer clearance across age), Marker 1998
(sequential infusions across age), and Leenen 2005 (cardiac responses across
age with and without neuronal uptake and reflex buffering), plus Stratton 1985
(relative flow and cuff pressure during two infusion sequences). The PK model
holds exogenous drug only, so targets are increments above basal converted with
1 nmol/L = 0.1832 ng/mL. Bolus: Takahashi 2002, 5-15 mcg IV under propofol
(mean age 32), and Guinard 1995, 10-15 mcg IV while awake. Tolerances are 1.5
reported SD for peaks and approximate for timing.
"""

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.arterial import ArterialWaveformConfig, ArterialWaveformRenderer
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.patient.patient import Patient
from anasim.patient.pk_models import EpinephrinePK
from anasim.physiology.hemo_config import HemodynamicConfig
from anasim.physiology.hemodynamics import HemodynamicModel


def _step(model, ce=0.0, *, dt=1.0, propofol=0.0, mac=0.0):
    return model.step(dt, propofol, 0.0, 0.0, -2.0, 40.0, 95.0, ce_epi=ce, mac_sevo=mac)


def test_freyschuss_arterial_infusion_steps():
    """Three 15-min steps, sampled at 12 min; baseline 0.27 nmol/L is endogenous."""
    patient = Patient(age=24, weight=70, height=178, baseline_hr=60, baseline_map=91)
    model = HemodynamicModel(patient)
    base = _step(model, dt=0.0)
    measurements = []
    for total_nmol_l in (1.34, 2.30, 6.02):
        ce = (total_nmol_l - 0.27) * 0.1832
        for _ in range(720):
            state = _step(model, ce)
        measurements.append(state)
        for _ in range(180):
            _step(model, ce)

    for state, hr, sv, co, svr in zip(
        (measurements[0], measurements[2]), (5, 13), (10, 34), (19, 60), (-15, -38), strict=True
    ):
        assert state.hr - base.hr == pytest.approx(hr, abs=3.0)
        assert 100 * (state.sv / base.sv - 1) == pytest.approx(sv, abs=7.0)
        assert 100 * (state.co / base.co - 1) == pytest.approx(co, abs=10.0)
        assert 100 * (state.svr / base.svr - 1) == pytest.approx(svr, abs=6.0)
    # The paper found no significant MAP change.
    assert all(abs(s.map - base.map) < 6.0 for s in measurements)
    assert measurements[0].co < measurements[1].co < measurements[2].co
    assert measurements[0].svr > measurements[1].svr > measurements[2].svr


def test_ensinger_arterial_concentration_response():
    """Ensinger 1992, Eur J Clin Pharmacol Table 2: five 30-min steps.

    Use arterial increments to isolate PD from clearance. Body dimensions are
    range midpoints; resting pressure matches the cohort. Bounds use 1.5 of
    the reported endpoint SD, rather than an unavailable SD of paired changes.
    """
    hr, sbp, dbp = 57.0, 132.0, 66.0
    renderer = ArterialWaveformRenderer(age=24.5)
    area = renderer._build_shape(60 / hr, hr).mean
    patient = Patient(age=24.5, weight=77, height=180, sex="male", baseline_hr=hr,
                      baseline_map=dbp + area * (sbp - dbp))
    model = HemodynamicModel(patient)
    base = _step(model, dt=0)
    nominal = ArterialWaveformConfig()
    modeled_pp = base.sv / renderer.compliance_at(base.map)
    renderer = ArterialWaveformRenderer(patient.age, replace(
        nominal, compliance_ref_ml_mmhg=nominal.compliance_ref_ml_mmhg * modeled_pp / (sbp - dbp),
    ))
    cycle = CardiacCycle(np.random.default_rng(0))
    bp0 = renderer.step(cycle.seed(base.hr, RhythmType.SINUS), base.map, base.sv)
    responses = []
    for total_nmol_l in (1.75, 8.06, 12.2, 17.6, 23.9):
        pressure, heart_rate = [], []
        for second in range(1, 1801):
            state = _step(model, (total_nmol_l - 0.29) * 0.1832)
            bp = renderer.step(cycle.step(1, state.hr, state.rhythm_type), state.map, state.sv,
                               pulse_pressure_factor=state.pulse_pressure_factor)
            if second > 1500:
                pressure.append((bp.systolic - bp0.systolic, bp.diastolic - bp0.diastolic))
            if second > 1680:
                heart_rate.append(state.hr - base.hr)
        responses.append([np.mean(heart_rate), *np.mean(pressure, axis=0)])
    observed = np.array([[1, 8, -2], [11, 28, -9], [22, 30, -9], [31, 31, -8], [38, 42, -5]])
    sd = np.array([[7, 5, 7], [10, 8, 7], [14, 10, 6], [13, 15, 6], [14, 22, 7]])
    assert np.all(np.abs(responses - observed) < 1.5 * sd), responses


def test_stratton_graded_infusions():
    """Table 1: five subjects per sequence, with each dose lasting 14 min.

    Size and absolute flow were unavailable; assume 70 kg, 175 cm and CI 3.0.
    Match resting BP before dosing. MAE limits allow the infusion/bolus
    calibration tradeoff; they are not reported cohort uncertainty.
    """
    sequences = []
    for hr, sbp, rest_map, doses in ((59, 120, 89, (25, 50)), (62, 115, 86, (50, 100))):
        dbp = (3 * rest_map - sbp) / 2
        renderer = ArterialWaveformRenderer(age=27)
        area = renderer._build_shape(60 / hr, hr).mean
        patient = Patient(age=27, weight=70, height=175, baseline_hr=hr,
                          baseline_map=dbp + area * (sbp - dbp))
        model = HemodynamicModel(patient, replace(HemodynamicConfig(), cardiac_index=3.0))
        pk = EpinephrinePK(patient)
        base = _step(model, dt=0)
        modeled_pp = base.sv / renderer.compliance_at(base.map)
        nominal = ArterialWaveformConfig()
        renderer = ArterialWaveformRenderer(27, replace(
            nominal, compliance_ref_ml_mmhg=nominal.compliance_ref_ml_mmhg * modeled_pp / (sbp - dbp),
        ))
        cycle = CardiacCycle(np.random.default_rng(0))
        bp0 = renderer.step(cycle.seed(base.hr, RhythmType.SINUS), base.map, base.sv)
        map0 = (bp0.systolic + 2 * bp0.diastolic) / 3
        responses = []
        for dose in doses:
            late = []
            for second in range(1, 841):
                state = _step(model, pk.step(1, dose * 1e-3 * patient.weight / 60).ce)
                bp = renderer.step(cycle.step(1, state.hr, state.rhythm_type), state.map, state.sv,
                                   pulse_pressure_factor=state.pulse_pressure_factor)
                if second > 720:
                    late.append((state.hr, state.sv, state.co))
            mean_hr, mean_sv, mean_co = np.mean(late, axis=0)
            co_ratio = mean_co / base.co
            cuff_map = (bp.systolic + 2 * bp.diastolic) / 3
            responses.append(np.array([
                mean_hr - base.hr, 100 * (mean_sv / base.sv - 1), 100 * (co_ratio - 1),
                100 * ((cuff_map / map0) / co_ratio - 1),
                cuff_map - map0, bp.diastolic - bp0.diastolic,
            ]))
        sequences.append(responses)
    pooled = np.array([sequences[0][0], (sequences[0][1] + sequences[1][0]) / 2, sequences[1][1]])
    observed = np.array([[8, 26, 41, -31], [12, 31, 58, -42], [15, 40, 74, -48]])
    mae = np.abs(pooled[:, :4] - observed).mean(axis=0)
    assert np.all(mae < [8.0, 10.0, 10.0, 9.0]), mae
    # Guard the final pressure response, not only flow and derived resistance.
    assert pooled[-1, 4] < 3.0
    assert pooled[-1, 1] > pooled[1, 1]


def _bolus_response(dose, *, dt=0.2, mac=0.0, baroreflex=True):
    patient = Patient(age=32, weight=56, height=160, sex="female")
    config = HemodynamicConfig()
    if not baroreflex:
        config = replace(config, fb=0.0, baro_gain_brady=0.0, baro_gain_tachy=0.0)
    model = HemodynamicModel(patient, config)
    # Fixed propofol Cp 3 approximates the trial's propofol/N2O anesthetic.
    model.initialize_steady_state(3.0, 0.0, 0.0, mac_sevo=mac)
    base = _step(model, dt=0.0, propofol=3.0, mac=mac)
    cycle = CardiacCycle(np.random.default_rng(0))
    renderer = ArterialWaveformRenderer(age=patient.age)
    baseline_pressure = renderer.step(cycle.seed(base.hr, RhythmType.SINUS), base.map, base.sv, pulse_pressure_factor=base.pulse_pressure_factor)
    pk = EpinephrinePK(patient)
    pk.state.c1 = dose / pk.v1
    rows = []
    for _ in range(round(300 / dt)):
        state = _step(model, pk.step(dt, 0.0).ce, dt=dt, propofol=3.0, mac=mac)
        pressure = renderer.step(cycle.step(dt, state.hr, state.rhythm_type), state.map, state.sv, pulse_pressure_factor=state.pulse_pressure_factor)
        rows.append((state.hr - base.hr, pressure.systolic - baseline_pressure.systolic, state.map))
    response = np.asarray(rows)
    hr_peak, sbp_peak = response[:, :2].max(axis=0)
    hr_time, sbp_time = (response[:, :2].argmax(axis=0) + 1) * dt
    late_hr = response[round(160 / dt):, 0].min()
    return np.array([hr_peak, sbp_peak, hr_time, sbp_time, late_hr, response[-1, 0],
                     response[:, 2].min() - base.map, response[:, 2].max()])


@pytest.mark.parametrize(
    "dose,hr_mean,hr_sd,sbp_mean,sbp_sd,hr_time,sbp_time",
    [(5, 25, 8, 24, 11, 60, 99), (10, 36, 11, 46, 16, 53, 88), (15, 43, 16, 75, 27, 53, 84)],
)
def test_takahashi_bolus_response(dose, hr_mean, hr_sd, sbp_mean, sbp_sd, hr_time, sbp_time):
    response = _bolus_response(dose)
    assert response[0] == pytest.approx(hr_mean, abs=1.5 * hr_sd)
    assert response[1] == pytest.approx(sbp_mean, abs=1.5 * sbp_sd)
    assert response[2] == pytest.approx(hr_time, abs=15.0)
    assert response[3] == pytest.approx(sbp_time, abs=25.0)
    assert response[3] > response[2] + 15.0
    assert response[4] < 0.0  # Figure 1: late fall below baseline.
    assert abs(response[5]) < 5.0


def test_takahashi_mean_dose_response():
    peaks = np.array([_bolus_response(dose)[:2] for dose in (5, 10, 15)])
    observed = np.array([[25, 24], [36, 46], [43, 75]])
    # Check the cohort means and dose trend in addition to individual SD bounds.
    rmse = np.sqrt(np.mean((peaks - observed) ** 2, axis=0))
    assert np.all(rmse < [4.0, 8.0]), rmse
    assert np.all(np.diff(peaks, axis=0) > [3.0, 15.0]), peaks


def test_awake_large_boluses_keep_raising_hr_and_sbp(engine_factory):
    """Extrapolation guards for a healthy adult, not measured high-dose targets."""
    rows = []
    for dose in (15, 25, 50, 100):
        engine = engine_factory(start=True, age=45)
        baseline = np.array([engine.state.hr, engine.state.sbp])
        peak = np.zeros(2)
        engine.give_drug_bolus("epi", dose)
        for _ in range(300):
            engine.step(1.0)
            peak = np.maximum(peak, [engine.state.hr, engine.state.sbp] - baseline)
        rows.append(peak)
    peaks = np.array(rows)
    assert np.all(np.diff(peaks, axis=0) > [1.0, 8.0]), peaks
    assert 40.0 < peaks[-1, 0] < 65.0
    assert 80.0 < peaks[-1, 1] < 180.0


def test_higher_infusion_rates_restore_vascular_resistance(engine_factory):
    """Guard the alpha upturn; its exact dose and size are extrapolated."""
    resistance, dbp = [], []
    for rate in (0.05, 0.1, 0.2, 0.4):
        engine = engine_factory(start=True, age=45)
        base_svr, base_dbp = engine.hemo.state.svr, engine.state.dbp
        engine.set_drug_rate("epi", rate * engine.patient.weight)
        for _ in range(1200):
            engine.step(1.0)
        resistance.append(engine.hemo.state.svr / base_svr)
        dbp.append(engine.state.dbp - base_dbp)
    assert resistance[1] < resistance[0] - 0.01, resistance
    assert np.all(np.diff(resistance[1:]) > [0.03, 0.05]), resistance
    assert resistance[-1] > 0.85
    assert dbp[-1] > 5.0


@pytest.mark.parametrize("age", [25, 65, 85])
def test_high_dose_response_across_age(engine_factory, age):
    """Extrapolation guards for arterial alpha's age response."""
    engine = engine_factory(start=True, age=age)
    baseline_sbp = peak = engine.state.sbp
    engine.give_drug_bolus("epi", 100.0)
    for _ in range(300):
        engine.step(1.0)
        peak = max(peak, engine.state.sbp)
    assert 60.0 < peak - baseline_sbp < 200.0

    resistance, dbp = [], []
    for rate in (0.1, 0.2, 0.4):
        engine = engine_factory(start=True, age=age)
        base_svr, base_dbp = engine.hemo.state.svr, engine.state.dbp
        engine.set_drug_rate("epi", rate * engine.patient.weight)
        for _ in range(1200):
            engine.step(1.0)
        resistance.append(engine.hemo.state.svr / base_svr)
        dbp.append(engine.state.dbp - base_dbp)
    assert np.all(np.diff(resistance) > 0.03), resistance
    assert dbp[-1] > 5.0


@pytest.mark.parametrize("mode", ["awake", "steady_state"])
@pytest.mark.parametrize("seed", [0, 3])
def test_shock_infusion_rates_keep_increasing_resistance(engine_factory, mode, seed):
    """Extrapolation guard beyond the measured infusion protocols."""
    resistance = []
    for rate in (0.4, 0.5, 0.7, 1.0, 1.5):
        engine = engine_factory(config=SimulationConfig(mode=mode, rng_seed=seed), start=True, age=45)
        baseline = engine.hemo.state.svr
        engine.set_drug_rate("epi", rate * engine.patient.weight)
        for _ in range(1200):
            engine.step(1.0)
        resistance.append(engine.hemo.state.svr / baseline)
    assert np.all(np.diff(resistance) > 0.01), resistance
    assert resistance[2] > 0.97, resistance
    assert resistance[3] > 1.05, resistance


def test_bolus_timing_converges_and_late_dip_requires_feedback():
    fine = _bolus_response(10, dt=0.1)
    coarse = _bolus_response(10, dt=0.5)
    assert coarse[:2] == pytest.approx(fine[:2], abs=0.6)
    assert coarse[2:4] == pytest.approx(fine[2:4], abs=2.0)
    assert _bolus_response(10, baroreflex=False)[4] > 0.0 > fine[4]


def test_bolus_has_no_large_initial_pressure_dip_or_large_dose_clipping():
    assert _bolus_response(10)[6] > -5.0
    responses = np.array([_bolus_response(dose) for dose in (15, 50, 100, 200)])
    assert np.all(np.diff(responses[:3, :2], axis=0) > 0.0), responses[:, :2]
    peaks = responses[:, 7]
    assert all(a < b for a, b in itertools.pairwise(peaks))
    assert max(peaks) < 280.0  # Leave headroom below the 300 mmHg safety cap.


def test_volatile_blunts_bolus_chronotropy():
    assert _bolus_response(10, mac=1.0)[0] < 0.85 * _bolus_response(10)[0]


@pytest.mark.parametrize(
    "age,baseline_hr,hr_means,hr_sds,sbp_means,sbp_sds",
    [(30, 71, (40, 39), (10, 9), (22, 28), (7, 10)),
     (50, 82, (28, 29), (8, 8), (22, 28), (9, 10)),
     (65, 67, (29, 31), (10, 8), (28, 33), (12, 14))],
)
@pytest.mark.parametrize("dose_index,dose", [(0, 10.0), (1, 15.0)])
def test_guinard_awake_bolus_by_age(age, baseline_hr, hr_means, hr_sds, sbp_means, sbp_sds,
                                  dose_index, dose):
    """Guinard 1995, Tables 1-2: unpremedicated, supine, 3-s IV injections.

    Use the cohort's mean size and age-group resting HR, with representative
    ages in AnaSim's adult domain. Only SBP is reported, so baseline MAP is fixed.
    """
    patient = Patient(age=age, weight=74.5, height=173, baseline_hr=baseline_hr, baseline_map=85)
    model, pk = HemodynamicModel(patient), EpinephrinePK(patient)
    cycle = CardiacCycle(np.random.default_rng(0))
    renderer = ArterialWaveformRenderer(age=age)
    base = _step(model, dt=0.0)
    baseline_pressure = renderer.step(cycle.seed(base.hr, RhythmType.SINUS), base.map, base.sv, pulse_pressure_factor=base.pulse_pressure_factor)
    hr_peak = sbp_peak = 0.0
    for index in range(1500):
        ce = pk.step(0.2, dose / 3.0 if index < 15 else 0.0).ce
        state = _step(model, ce, dt=0.2)
        pressure = renderer.step(cycle.step(0.2, state.hr, state.rhythm_type), state.map, state.sv, pulse_pressure_factor=state.pulse_pressure_factor)
        if index < 600:  # Continuous peak measurement for 2 min; recovery at 5 min.
            hr_peak = max(hr_peak, state.hr - base.hr)
            sbp_peak = max(sbp_peak, pressure.systolic - baseline_pressure.systolic)
    assert hr_peak == pytest.approx(hr_means[dose_index], abs=1.5 * hr_sds[dose_index])
    assert sbp_peak == pytest.approx(sbp_means[dose_index], abs=1.5 * sbp_sds[dose_index])
    assert abs(state.hr - base.hr) < 5.0
    assert abs(pressure.systolic - baseline_pressure.systolic) < 5.0


def test_pk_infusion_mass_balance_and_washout():
    pk = EpinephrinePK(Patient(age=25))
    assert pk.cl1_base == pytest.approx(0.046 * 70)  # Calibrated young-adult arterial clearance, L/min.
    # Clearance must not scale with the resulting rise in cardiac output.
    pk.update_hemodynamics(1.0, 1.6)
    rate = 0.2 * 70 / 60  # mcg/kg/min -> mcg/s
    eliminated_ug = 0.0
    for _ in range(3600):
        c_before = pk.state.c1
        c_after = pk.step(0.5, rate).c1
        eliminated_ug += 0.5 * (c_before + c_after) * pk.cl1_base * (0.5 / 60.0)
    infused_ug = rate * 1800.0
    retained_ug = pk.state.c1 * pk.v1
    assert retained_ug + eliminated_ug == pytest.approx(infused_ug, rel=1e-3)
    assert pk.state.c1 == pytest.approx(rate * 60 / pk.cl1_base, rel=0.005)
    assert pk.state.ce == pytest.approx(pk.state.c1, rel=0.005)
    for _ in range(3600):
        pk.step(0.5, 0.0)
    assert pk.state.ce < 0.02


def test_age_clearance_matches_esler():
    """Esler 1995: brachial arterial clearance fell from 2.80 L/min at 20-30 yr
    to 2.24 L/min at 60-75 yr in men of similar BMI.
    """
    young = EpinephrinePK(Patient(age=25, weight=70))
    older = EpinephrinePK(Patient(age=67.5, weight=70))
    assert older.cl1_base / young.cl1_base == pytest.approx(2.24 / 2.80, rel=0.03)


def test_marker_and_leenen_infusion_cardiovascular_responses_by_age():
    """Marker 1998 Table 4: five 30-min steps (7.5-60 ng/kg/min) raised HR by 21
    and 25 bpm and lowered DBP by up to 7 and 11 mmHg at 23 and 63 yr.
    Leenen 2005 Tables 1 and 3, Figs. 1 and 3: HR, SVI, and CI rises during 8-min
    steps did not differ significantly by age (SVI +17 vs +14 mL/m2 at 120
    ng/kg/min). With neuronal uptake and ganglia blocked, HR and EF rises at 61 yr
    were 0.59 and 0.50 of those at 29 yr; the model has no neuronal uptake, whose
    decline with age offsets part of that loss.
    """
    marker_hr, marker_dbp_nadir, marker_dbp_final = {}, {}, {}
    for age, weight, height, base_hr, sbp0, dbp0 in (
        (23, 69.4, 174, 64, 115, 64),
        (63, 75.4, 169, 63, 128, 73),
    ):
        map0 = round((sbp0 + 2 * dbp0) / 3)
        patient = Patient(age=age, weight=weight, height=height, baseline_hr=base_hr, baseline_map=map0)
        model, pk = HemodynamicModel(patient), EpinephrinePK(patient)
        cycle = CardiacCycle(np.random.default_rng(0))
        renderer = ArterialWaveformRenderer(age=age)
        base = _step(model, dt=0.0)
        bp_base = renderer.step(cycle.seed(base.hr, RhythmType.SINUS), base.map, base.sv, pulse_pressure_factor=base.pulse_pressure_factor)
        dbp_changes = []
        for ng_kg_min in (7.5, 15.0, 30.0, 45.0, 60.0):
            rate_ug_s = (ng_kg_min * 1e-3 * weight) / 60.0
            for _ in range(1800):
                state = _step(model, pk.step(1.0, rate_ug_s).ce)
                bp = renderer.step(cycle.step(1.0, state.hr, state.rhythm_type), state.map, state.sv, pulse_pressure_factor=state.pulse_pressure_factor)
            dbp_changes.append(bp.diastolic - bp_base.diastolic)
        marker_hr[age] = state.hr - base.hr
        marker_dbp_nadir[age] = min(dbp_changes)
        marker_dbp_final[age] = dbp_changes[-1]

    assert 17.0 < marker_hr[23] < 27.0
    assert 17.0 < marker_hr[63] < 27.0
    assert abs(marker_hr[63] - marker_hr[23]) < 5.0
    assert -12.0 < marker_dbp_nadir[23] < -4.0
    assert -12.0 < marker_dbp_nadir[63] < -4.0
    assert marker_dbp_final[63] == pytest.approx(-11.0, abs=3.0)

    leenen_intact = {}
    for age, weight, height, base_hr, sbp0, dbp0 in ((29, 72, 170, 59, 111, 63), (61, 71, 175, 51, 127, 71)):
        map0 = round((sbp0 + 2 * dbp0) / 3)
        patient = Patient(age=age, weight=weight, height=height, baseline_hr=base_hr, baseline_map=map0)
        model, pk = HemodynamicModel(patient), EpinephrinePK(patient)
        base = _step(model, dt=0.0)
        for ng_kg_min in (20.0, 40.0, 80.0, 120.0):
            rate_ug_s = (ng_kg_min * 1e-3 * weight) / 60.0
            for _ in range(480):
                state = _step(model, pk.step(1.0, rate_ug_s).ce)
        leenen_intact[age] = (
            state.hr - base.hr,
            (state.sv - base.sv) / patient.bsa,
            (state.co - base.co) / patient.bsa,
        )
    assert leenen_intact[61][0] == pytest.approx(leenen_intact[29][0], rel=0.15)
    assert leenen_intact[61][1] < leenen_intact[29][1]
    assert leenen_intact[61][2] == pytest.approx(leenen_intact[29][2], rel=0.21)

    no_baro = replace(HemodynamicConfig(), fb=0.0, baro_gain_brady=0.0, baro_gain_tachy=0.0, vol_clearance=0.0)
    intrinsic = {}
    for age in (29, 61):
        patient = Patient(age=age, weight=78, height=176, baseline_hr=65, baseline_map=85)
        model = HemodynamicModel(patient, no_baro)
        base = _step(model, dt=0.0)
        for _ in range(480):
            state = _step(model, 0.42)
        intrinsic[age] = (state.hr - base.hr, state.sv / base.sv - 1.0)
    assert 0.65 < intrinsic[61][0] / intrinsic[29][0] < 0.85
    assert 0.65 < intrinsic[61][1] / intrinsic[29][1] < 0.85


def test_inotropy_is_applied_once():
    config = replace(HemodynamicConfig(), fb=0.0, baro_gain_brady=0.0, baro_gain_tachy=0.0, vol_clearance=0.0)
    model = HemodynamicModel(Patient(), config)
    base = _step(model, dt=0.0)
    for _ in range(3600):
        state = model.step(1.0, 0, 0, 0, -2, 40, 95, ce_dobu=10.0)
    # Inotropy scales output SV only, not SV production.
    _, factor, _ = model._calc_hr_sv_svr_effects(
        10.0, config.dobu_c50, config.dobu_gamma, config.dobu_emax_hr,
        config.dobu_emax_sv, config.dobu_emax_svr,
    )
    coupling = 1 - config.hr_sv_coupling * math.log(max(1.0, state.hr / model.base_hr))
    assert state.sv / base.sv / coupling == pytest.approx(factor, rel=0.001)
    assert state.sv_star == pytest.approx(base.sv_star, rel=0.001)


def test_age_blunts_bolus_but_not_infusion_tachycardia(engine_factory):
    """Leenen 2005: infusions raised HR equally at 29 and 61 years, as a smaller
    beta response offset weaker baroreflex buffering. Guinard 1995: 10 mcg IV
    raised HR 40 bpm before age 40 and 28-29 bpm after.
    """
    infusion, bolus = {}, {}
    for age in (29, 65):
        engine = engine_factory(start=True, age=age)
        base = engine.state.hr
        engine.set_drug_rate("epi", 0.12 * engine.patient.weight)  # 120 ng/kg/min
        for _ in range(600):
            engine.step(1.0)
        infusion[age] = engine.state.hr - base

        engine = engine_factory(start=True, age=age)
        base = peak = engine.state.hr
        engine.give_drug_bolus("epi", 10.0)
        for _ in range(1200):
            engine.step(0.1)
            peak = max(peak, engine.state.hr)
        bolus[age] = peak - base
    assert infusion[65] == pytest.approx(infusion[29], rel=0.15)
    assert 0.65 < bolus[65] / bolus[29] < 0.95
