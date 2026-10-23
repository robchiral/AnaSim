"""Epinephrine calibration against published cohort responses.

Infusion: Freyschuss 1986 (arterial samples, basal 0.27 nmol/L, mean age 24).
The PK model holds exogenous drug only, so targets are increments above basal
converted with 1 nmol/L = 0.1832 ng/mL. Bolus: Takahashi 2002, 5-15 mcg IV under
propofol (mean age 32), and Guinard 1995, 10-15 mcg IV while awake. Tolerances
are 1.5 reported SD for peaks and approximate for timing.
"""

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.monitors.arterial import ArterialWaveformRenderer
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
    baseline_pressure = renderer.step(cycle.seed(base.hr, RhythmType.SINUS), base.map, base.sv)
    pk = EpinephrinePK(patient)
    pk.state.c1 = dose / pk.v1
    rows = []
    for _ in range(round(300 / dt)):
        state = _step(model, pk.step(dt, 0.0).ce, dt=dt, propofol=3.0, mac=mac)
        pressure = renderer.step(cycle.step(dt, state.hr, state.rhythm_type), state.map, state.sv)
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


def test_bolus_timing_converges_and_late_dip_requires_feedback():
    fine = _bolus_response(10, dt=0.1)
    coarse = _bolus_response(10, dt=0.5)
    assert coarse[:2] == pytest.approx(fine[:2], abs=0.6)
    assert coarse[2:4] == pytest.approx(fine[2:4], abs=2.0)
    assert _bolus_response(10, baroreflex=False)[4] > 0.0 > fine[4]


def test_bolus_has_no_large_initial_pressure_dip_or_large_dose_clipping():
    assert _bolus_response(10)[6] > -5.0
    peaks = [_bolus_response(dose)[7] for dose in (15, 50, 100, 200)]
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
    baseline_pressure = renderer.step(cycle.seed(base.hr, RhythmType.SINUS), base.map, base.sv)
    hr_peak = sbp_peak = 0.0
    for index in range(1500):
        ce = pk.step(0.2, dose / 3.0 if index < 15 else 0.0).ce
        state = _step(model, ce, dt=0.2)
        pressure = renderer.step(cycle.step(0.2, state.hr, state.rhythm_type), state.map, state.sv)
        if index < 600:  # Continuous peak measurement for 2 min; recovery at 5 min.
            hr_peak = max(hr_peak, state.hr - base.hr)
            sbp_peak = max(sbp_peak, pressure.systolic - baseline_pressure.systolic)
    assert hr_peak == pytest.approx(hr_means[dose_index], abs=1.5 * hr_sds[dose_index])
    assert sbp_peak == pytest.approx(sbp_means[dose_index], abs=1.5 * sbp_sds[dose_index])
    assert abs(state.hr - base.hr) < 5.0
    assert abs(pressure.systolic - baseline_pressure.systolic) < 5.0


def test_pk_infusion_mass_balance_and_washout():
    pk = EpinephrinePK(Patient())
    assert pk.cl1_base == pytest.approx(0.046 * 70)  # Ensinger arterial clearance, L/min.
    # Clearance must not scale with the resulting rise in cardiac output.
    pk.update_hemodynamics(1.0, 1.6)
    rate = 0.2 * 70 / 60  # mcg/kg/min -> mcg/s
    for _ in range(3600):
        pk.step(0.5, rate)
    assert pk.state.c1 == pytest.approx(rate * 60 / pk.cl1_base, rel=0.005)
    assert pk.state.ce == pytest.approx(pk.state.c1, rel=0.005)
    for _ in range(3600):
        pk.step(0.5, 0.0)
    assert pk.state.ce < 0.02


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
