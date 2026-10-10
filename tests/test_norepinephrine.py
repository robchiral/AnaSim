"""Norepinephrine calibration on Li PK and adult cohort response summaries.

de Keijzer 2026 and Li 2024 report the same volunteer cohort. Joachim 2024
provides the separate 10 mcg peripheral-bolus benchmark. These checks hold
anesthetic concentrations, gas exchange, and fluid volume constant.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from anasim.patient.patient import Patient
from anasim.patient.pk_models import NorepinephrinePK
from anasim.physiology.hemo_config import HemodynamicConfig
from anasim.physiology.hemodynamics import HemodynamicModel


def _equilibrated(patient, propofol, remifentanil):
    pk = NorepinephrinePK(patient)
    model = HemodynamicModel(patient, replace(HemodynamicConfig(), vol_clearance=0.0))
    for _ in range(1800):
        pk.step(1.0, 0.0, propofol)
    model.initialize_steady_state(propofol, remifentanil, pk.state.ce)
    for _ in range(900):
        state = model.step(1.0, propofol, remifentanil, pk.state.ce, -2.0, 40.0, 95.0)
    return pk, model, state


def test_li_infusion_concentration_with_propofol_and_changed_cardiac_output():
    patient = Patient(age=52, weight=71, height=163, sex="female")
    pk = NorepinephrinePK(patient)
    # Li's final covariates are age, weight, and measured propofol Cp.
    # A change in CO must not add a second clearance/distribution covariate.
    pk.update_hemodynamics(1.0, 1.6)
    propofol = 3.53
    rate = 10.0  # mcg/min
    for _ in range(7200):
        pk.step(0.5, rate / 60.0, propofol)
    size = patient.weight / 70.0
    clearance = 2.1 * size**0.75 * math.exp(-0.00344 * (patient.age - 35))
    clearance *= math.exp(-0.0357 * propofol)
    expected = (rate + 0.4977 * size**0.75) / clearance
    assert pk.state.c1 == pytest.approx(expected, rel=0.005)
    assert pk.state.ce == pytest.approx(expected, rel=0.005)
    for _ in range(10800):
        pk.step(0.5, 0.0, propofol)
    assert pk.state.c1 == pytest.approx(0.4977 * size**0.75 / clearance, rel=0.005)


def test_de_keijzer_infusion_pressure_and_cardiac_output():
    patient = Patient(age=35, weight=70, height=170, baseline_map=88)
    phases = []
    for propofol, remifentanil in ((0.0, 0.0), (3.53, 3.6)):
        pk, model, baseline = _equilibrated(patient, propofol, remifentanil)
        states = [baseline]
        rates = (0.0, 0.04, 0.08, 0.12, 0.16, 0.20)  # mcg/kg/min
        for rate in rates[1:]:
            for _ in range(900):
                pk.step(1.0, rate * patient.weight / 60.0, propofol)
                state = model.step(1.0, propofol, remifentanil, pk.state.ce, -2.0, 40.0, 95.0)
            states.append(state)
        phases.append(states)
        slope = np.polyfit(rates, [s.map for s in states], 1)[0]
        if propofol == 0.0:
            assert 81.0 < slope < 124.0  # Reported awake 95% CI.
            assert states[-1].co == pytest.approx(baseline.co, rel=0.15)
        else:
            assert slope == pytest.approx(222.0, abs=35.0)
            assert states[-1].co > states[1].co
            assert states[-1].hr == pytest.approx(states[1].hr, abs=10.0)
    # CO recovers toward the awake baseline as MAP is restored.
    assert phases[1][-1].co == pytest.approx(phases[0][0].co, rel=0.15)
    assert phases[1][0].map == pytest.approx(54.0, abs=6.0)


def test_joachim_peripheral_bolus_pressure_and_timing():
    patient = Patient(age=52, weight=71, height=163, sex="female", baseline_map=88)
    propofol, remifentanil = 3.53, 3.6
    pk, model, baseline = _equilibrated(patient, propofol, remifentanil)
    pk.state.c1 += 10.0 / pk.v1
    dt = 0.2
    pressures = []
    for _ in range(round(300.0 / dt)):
        pk.step(dt, 0.0, propofol)
        state = model.step(dt, propofol, remifentanil, pk.state.ce, -2.0, 40.0, 95.0)
        pressures.append(state.map)
    rise = 100.0 * (max(pressures) / baseline.map - 1.0)
    peak_time = (np.argmax(pressures) + 1) * dt
    assert 15.0 < rise < 31.0  # Reported IQR; median 24%.
    assert 53.0 < peak_time < 94.0  # Reported IQR; median 74 s.
    assert pressures[-1] < baseline.map + 5.0
