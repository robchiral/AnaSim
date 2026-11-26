"""Norepinephrine calibration on Li PK and adult cohort response summaries.

de Keijzer 2026 and Li 2024 report the same volunteer cohort. Joachim 2024
provides the separate 10 mcg peripheral-bolus benchmark. These checks hold
anesthetic concentrations, gas exchange, and fluid volume constant.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.patient.pk_models import NorepinephrinePK
from anasim.physiology.hemo_config import HemodynamicConfig
from anasim.physiology.hemodynamics import HemodynamicModel


def _equilibrated(patient, propofol, remifentanil):
    pk = NorepinephrinePK(patient)
    pk.update_propofol(propofol)
    model = HemodynamicModel(patient, replace(HemodynamicConfig(), vol_clearance=0.0))
    for _ in range(1800):
        pk.step(1.0, 0.0)
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
    pk.update_propofol(propofol)
    rate = 10.0  # mcg/min
    for _ in range(7200):
        pk.step(0.5, rate / 60.0)
    size = patient.weight / 70.0
    clearance = 2.1 * size**0.75 * math.exp(-0.00344 * (patient.age - 35))
    clearance *= math.exp(-0.0357 * propofol)
    expected = (rate + 0.4977 * size**0.75) / clearance
    assert pk.state.c1 == pytest.approx(expected, rel=0.005)
    assert pk.state.ce == pytest.approx(expected, rel=0.005)
    for _ in range(10800):
        pk.step(0.5, 0.0)
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
                pk.step(1.0, rate * patient.weight / 60.0)
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
    # The cohort was enrolled for hypotension, so its MAP bounds the model within 2 SD.
    assert phases[1][0].map == pytest.approx(54.0, abs=18.0)


def test_joachim_peripheral_bolus_pressure_and_timing():
    patient = Patient(age=52, weight=71, height=163, sex="female", baseline_map=88)
    propofol, remifentanil = 3.53, 3.6
    pk, model, baseline = _equilibrated(patient, propofol, remifentanil)
    pk.state.c1 += 10.0 / pk.v1
    dt = 0.2
    pressures = []
    for _ in range(round(300.0 / dt)):
        pk.step(dt, 0.0)
        state = model.step(dt, propofol, remifentanil, pk.state.ce, -2.0, 40.0, 95.0)
        pressures.append(state.map)
    rise = 100.0 * (max(pressures) / baseline.map - 1.0)
    peak_time = (np.argmax(pressures) + 1) * dt
    assert 15.0 < rise < 31.0  # Reported IQR; median 24%.
    assert 53.0 < peak_time < 94.0  # Reported IQR; median 74 s.
    assert pressures[-1] < baseline.map + 5.0


def test_propofol_bolus_clearance_cap_and_hypovolemic_pressor_response(patient):
    """Bolus propofol Cp spikes cap at 8 mcg/mL, and clinical norepinephrine
    infusion monotonically raises MAP during TIVA hypovolemia.
    """
    pk = NorepinephrinePK(patient)
    base_cl = pk.elimination_clearance
    pk.update_propofol(60.0)
    assert pk.elimination_clearance >= 0.70 * base_cl
    pk_cap = NorepinephrinePK(patient)
    pk_cap.update_propofol(8.0)
    assert pk.elimination_clearance == pytest.approx(pk_cap.elimination_clearance)

    pk_h, model_h, _ = _equilibrated(patient, 3.5, 3.0)
    model_h.add_volume(-1400.0)
    for _ in range(180):
        pk_h.step(1.0, 0.0)
        pre_nore = model_h.step(1.0, 3.5, 3.0, pk_h.state.ce, -2.0, 40.0, 95.0)
    maps = []
    for _ in range(120):
        pk_h.step(1.0, 0.15 * patient.weight / 60.0)
        maps.append(model_h.step(1.0, 3.5, 3.0, pk_h.state.ce, -2.0, 40.0, 95.0).map)
    assert min(maps) >= pre_nore.map - 0.1
    assert maps[-1] > pre_nore.map + 12.0


def test_zero_flow_arrest_halts_endogenous_norepinephrine_spillover(patient):
    """Zero-flow cardiac arrest halts basal sympathetic spillover, and reset restores baseline."""
    pk = NorepinephrinePK(patient)
    c1_0, ce_0 = pk.state.c1, pk.state.ce
    pk.update_hemodynamics(1.0, 0.0, 0.0)
    for _ in range(120):
        pk.step(1.0, 0.0)
    assert pk.state.c1 == pytest.approx(c1_0, abs=1e-12)
    assert pk.state.ce == pytest.approx(ce_0, abs=1e-12)

    pk.state.c1 = 40.0
    pk.reset()
    assert pk.state.c1 == pytest.approx(c1_0)
    assert pk.circulation == 1.0 and pk.cl1_scale == 1.0

    # Equilibrating endogenous norepinephrine at zero circulation neither divides by zero
    # nor inflates basal concentration, and drops effective_basal_input_rate to zero.
    pk.circulation = 0.0
    pk.equilibrate_endogenous()
    assert math.isfinite(pk.state.c1) and pk.state.c1 == pytest.approx(c1_0, rel=1e-6)
    assert pk.effective_basal_input_rate == 0.0


def test_norepinephrine_tci_uses_circulation_scaled_basal_input(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="awake", tci_enabled=True, rng_seed=1), start=True)
    engine.pk["nore"].circulation = 0.5
    engine.enable_tci("nore", 2.0, mode="plasma")
    assert engine.tci["nore"].basal_input_rate == pytest.approx(
        engine.pk["nore"].effective_basal_input_rate, rel=1e-9
    )
