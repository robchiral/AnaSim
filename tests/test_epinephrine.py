"""Epinephrine calibration against published cohort responses.

Infusion: Freyschuss 1986 (arterial samples, basal 0.27 nmol/L). The PK model
holds exogenous drug only, so targets are increments above basal converted with
1 nmol/L = 0.1832 ng/mL. Bolus: Takahashi 2002, 5-15 mcg IV under propofol.
Tolerances are 1.5 reported SD for peaks and approximate for timing.
"""

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
    patient = Patient(age=25, weight=70, height=178, baseline_hr=60, baseline_map=91)
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
        (measurements[0], measurements[2]), (5, 13), (10, 34), (19, 60), (-15, -38)
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
    patient = Patient(age=40, weight=56, height=160, sex="female")
    config = HemodynamicConfig()
    if not baroreflex:
        config = replace(config, fb=0.0, baro_gain_brady=0.0, baro_gain_tachy=0.0)
    model = HemodynamicModel(patient, config)
    # Fixed propofol Cp 3 approximates the trial's propofol/N2O anesthetic.
    model.state = model.calculate_steady_state(3.0, 0.0, 0.0, mac_sevo=mac)
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
        rows.append((state.hr - base.hr, pressure.systolic - baseline_pressure.systolic))
    response = np.asarray(rows)
    hr_peak, sbp_peak = response.max(axis=0)
    hr_time, sbp_time = (response.argmax(axis=0) + 1) * dt
    late_hr = response[round(160 / dt):, 0].min()
    return np.array([hr_peak, sbp_peak, hr_time, sbp_time, late_hr, response[-1, 0]])


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
    assert response[4] < -0.3
    assert abs(response[5]) < 5.0


def test_bolus_timing_converges_and_late_dip_requires_feedback():
    fine = _bolus_response(10, dt=0.1)
    coarse = _bolus_response(10, dt=0.5)
    assert coarse[:2] == pytest.approx(fine[:2], abs=0.6)
    assert coarse[2:4] == pytest.approx(fine[2:4], abs=2.0)
    assert _bolus_response(10, baroreflex=False)[4] > fine[4] + 1.0


def test_volatile_blunts_bolus_chronotropy():
    assert _bolus_response(10, mac=1.0)[0] < 0.85 * _bolus_response(10)[0]


@pytest.mark.parametrize("model_name", ["HealthyAdult", "Abboud"])
def test_pk_infusion_mass_balance_and_washout(model_name):
    pk = EpinephrinePK(Patient(), model=model_name)
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


def test_effects_remain_continuous_and_bounded_at_high_concentrations():
    model = HemodynamicModel(Patient())
    assert model._calc_epi_effects(0.0, 0.0) == (0.0, 1.0, 1.0)
    for c in np.geomspace(1e-6, 1e4, 100):
        effects = model._calc_epi_effects(c, c)
        nearby = model._calc_epi_effects(c * 1.00001, c * 1.00001)
        assert all(math.isfinite(x) for x in effects)
        assert 0 <= effects[0] <= model.epi_emax_hr
        assert 1 <= effects[1] <= 1 + model.epi_emax_sv
        assert 0.2 <= effects[2] <= 1 + model.epi_emax_svr_alpha
        assert nearby == pytest.approx(effects, abs=0.001)
    assert model._calc_epi_effects(0.2, 0.2)[2] < 1.0
    assert model._calc_epi_effects(20.0, 20.0)[2] > 1.0


def test_inotropy_is_applied_once():
    config = replace(HemodynamicConfig(), fb=0.0, baro_gain_brady=0.0, baro_gain_tachy=0.0, vol_clearance=0.0)
    model = HemodynamicModel(Patient(), config)
    base = _step(model, dt=0.0)
    for _ in range(3600):
        state = model.step(1.0, 0, 0, 0, -2, 40, 95, ce_dobu=10.0)
    # Inotropy scales output SV only, not SV production.
    _, factor, _ = model._calc_hr_sv_svr_effects(
        10.0, model.dobu_c50, model.dobu_gamma, model.dobu_emax_hr,
        model.dobu_emax_sv, model.dobu_emax_svr,
    )
    coupling = 1 - model.hr_sv_coupling * math.log(max(1.0, state.hr / model.base_hr))
    assert state.sv / base.sv / coupling == pytest.approx(factor, rel=0.001)
    assert state.sv_star == pytest.approx(base.sv_star, rel=0.001)
