"""Fentanyl and stimulation cardiovascular calibration, with induction and intubation replays."""

from types import SimpleNamespace

import numpy as np
import pytest

from anasim.core.projection import cardiovascular_response_fractions, stimulus_tolerance
from anasim.core.runtime import step_disturbances
from anasim.core.state import SimulationConfig
from anasim.patient.pd.anesthesia import opioid_equivalent
from validation.intubation import (
    BILLARD_CASES,
    SAWANO_CASES,
    replay_billard,
    replay_sawano,
)


@pytest.mark.parametrize("fentanyl,propofol,baseline_sbp,increase,reference,hr,hr_percent,hr_reference", [
    (1.2, 16.4, 92.8, 19.5, 92.3 * 0.245, 70.3, 8.5, 83.7 * 0.203),
    (1.2, 32.6, 79.6, 20.2, 70.3 * 0.287, 60.4, 13.2, 65.2 * 0.199),
    (2.6, 10.6, 95.5, 15.0, 92.3 * 0.245, 64.9, 7.7, 83.7 * 0.203),
    (2.6, 21.2, 86.4, 11.9, 70.3 * 0.287, 60.8, 4.8, 65.2 * 0.199),
    (4.1, 9.8, 90.6, 12.0, 92.3 * 0.245, 62.1, 4.8, 83.7 * 0.203),
    (4.1, 19.6, 78.0, 7.8, 70.3 * 0.287, 61.4, 3.3, 65.2 * 0.199),
    (5.5, 7.9, 88.7, 7.2, 92.3 * 0.245, 60.9, 3.3, 83.7 * 0.203),
    (5.5, 15.7, 74.8, 2.0, 70.3 * 0.287, 56.5, 1.8, 65.2 * 0.199),
])
def test_kazama_fentanyl_cardiovascular_calibration(
    engine_factory, fentanyl, propofol, baseline_sbp, increase, reference, hr, hr_percent, hr_reference,
):
    """Kazama 1997 Tables 2-3: absolute SBP rises at the motor Cp50 and Cp95,
    relative to the no-fentanyl rise. Infusions were held to equilibrium, so
    plasma concentrations stand for effect sites. SBP and HR have separate fits.
    """
    engine = engine_factory(start=True)
    engine.start_disturbance("stim_intubation_pulse")
    engine.state.time += 40.0
    engine.state.hypnotic_response_ce = engine.state.hypnotic_ce = propofol
    engine._tol_current = stimulus_tolerance(engine)
    control = step_disturbances(engine, 0.1)
    engine.state.fentanyl_ce = fentanyl
    engine.state.opioid_ce = opioid_equivalent(0.0, fentanyl)
    engine._tol_current = stimulus_tolerance(engine)
    treated = step_disturbances(engine, 0.1)
    expected_fraction = baseline_sbp * increase / 100.0 / reference
    assert treated.svr / control.svr == pytest.approx(expected_fraction, abs=0.11)
    assert treated.sv / control.sv == pytest.approx(treated.svr / control.svr)
    expected_hr_fraction = hr * hr_percent / 100.0 / hr_reference
    assert treated.hr / control.hr == pytest.approx(expected_hr_fraction, abs=0.14)


def test_fentanyl_stimulation_transfer_with_other_anesthetics(engine_factory):
    """Fentanyl blunting adds to sevoflurane and remifentanil."""
    engine = engine_factory(start=True)
    engine.state.mac_sevo = 1.0
    control, control_hr = cardiovascular_response_fractions(engine, 1.64)
    engine.state.fentanyl_ce = 3.0
    engine.state.opioid_ce = opioid_equivalent(0.0, 3.0)
    fentanyl, fentanyl_hr = cardiovascular_response_fractions(engine, 1.64)
    engine.pk["remi"].state.ce = 1.0
    engine.state.opioid_ce = opioid_equivalent(1.0, 3.0)
    combined, combined_hr = cardiovascular_response_fractions(engine, 1.64)
    assert 0.0 < combined < fentanyl < control
    assert 0.0 < combined_hr < fentanyl_hr < control_hr


def test_billard_bolus_pressure_and_hr_calibration():
    """Billard 1994 Tables 1-3: bolus transfer of the Kazama fentanyl fit, before
    and 1 min after intubation averaged over 10 s, for all four propofol doses.

    These cohorts also calibrate the stimulus profile, so the RMSE bounds guard
    the calibration. SBP rises remain too small at every fentanyl dose, and HR
    rises with fentanyl 6-7 bpm too small. python -m validation.intubation
    prints the full errors.
    """
    pressure_rises, hr_rises = [], []
    errors = []
    for dose in (0.0, 2.0, 4.0):
        cohorts = [case for case in BILLARD_CASES if case.fentanyl == dose]
        readings = [replay_billard(case) for case in cohorts]
        pressure_rises.append(float(np.mean([after.sbp - before.sbp for before, after in readings])))
        for case, (before, after) in zip(cohorts, readings, strict=True):
            errors.append((before.sbp - case.pre_sbp,
                           (after.sbp - before.sbp) - (case.post_sbp - case.pre_sbp),
                           (after.hr - before.hr) - (case.post_hr - case.pre_hr)))
        hr_rises.append(float(np.mean([after.hr - before.hr for before, after in readings])))
    assert 0.0 < pressure_rises[2] < pressure_rises[1] < pressure_rises[0]
    assert 0.0 < hr_rises[2] < hr_rises[1] < hr_rises[0]
    pre_sbp_rmse, sbp_rise_rmse, hr_rise_rmse = np.sqrt(np.mean(np.square(errors), axis=0))
    assert pre_sbp_rmse < 15.0
    assert sbp_rise_rmse < 25.0
    assert hr_rise_rmse < 8.0


@pytest.mark.xfail(strict=True, reason=(
    "Billard's propofol-alone arms started near MAP 103 and fell less than Morley 2010's "
    "slope of fall on pre-induction MAP predicts, so model SBP before intubation is 13 mmHg low."
))
def test_billard_propofol_alone_pressure_before_intubation():
    """Billard 1994 Table 1: propofol alone lowered SBP 13-21% before intubation."""
    errors = [replay_billard(case)[0].sbp - case.pre_sbp for case in BILLARD_CASES if case.fentanyl == 0.0]
    assert abs(np.mean(errors)) < 10.0


@pytest.mark.parametrize("case,ci_sd,svi_sd", [
    (SAWANO_CASES[0], 1.4, 16.0),
    (SAWANO_CASES[1], 1.1, 7.0),
    (SAWANO_CASES[2], 0.9, 11.0),
], ids=["no_fentanyl", "fentanyl_2", "fentanyl_4"])
def test_sawano_normotensive_pressure_and_flow_calibration(case, ci_sd, svi_sd):
    """Sawano 2013 Table 2a: MAP before and 1-10 min after intubation within two
    of the cohort's median SDs across readings, as for other research
    inductions; HR at 1 and 3-10 min, and CI and SVI at 1 min, within one SD.
    CI and SVI rise at intubation, and HR does not rebound after 3 min.
    """
    predicted = replay_sawano(case)
    map_sd = float(np.median(case.map_sd))
    for model, observed in zip(predicted[1:], case.observed[1:], strict=True):
        assert model.map == pytest.approx(observed.map, abs=2 * map_sd)
    for index in (2, 4, 5, 6, 7):
        assert predicted[index].hr == pytest.approx(case.observed[index].hr, abs=case.hr_sd[index])
    assert predicted[2].ci == pytest.approx(case.observed[2].ci, abs=ci_sd)
    assert predicted[2].svi == pytest.approx(case.observed[2].svi, abs=svi_sd)
    assert predicted[2].ci > predicted[1].ci
    assert predicted[2].svi > predicted[1].svi
    # Observed HR falls 3-7 bpm from 3 to 10 min; allow 1 bpm for rounding.
    assert max(point.hr for point in predicted[5:]) <= predicted[4].hr + 1.0


def _replay_hayakawa(engine_factory, intubation_minutes):
    """Hayakawa-Fujii 2001 PF arm, Table 1 and Figures 1-2; not used for calibration.

    Fentanyl 3 mcg/kg precedes propofol 2 mg/kg by 60 s; intubation is 4-5 min
    after propofol. Returns states 3 min after induction, 1 min after
    intubation, and 3 min after intubation.

    Only the age range (40-65) is reported; assume its midpoint, 60 kg, 165 cm,
    and male sex, with controlled ventilation from baseline. Assume an immediate
    bolus, then 6 mg/kg/h and 50% N2O from intubation.
    """
    engine = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=4), start=True,
        age=52.5, weight=60, height=165, sex="male", baseline_hr=71,
        baseline_map=77.0 + (134.0 - 77.0) / 3.0,
    )
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.6, peep=0.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(6.0, 0.0)
    engine.set_continuous_fluid_rate(10.0 * engine.patient.weight)
    engine.give_drug_bolus("fentanyl", 3.0 * engine.patient.weight)
    intubation = 60.0 + intubation_minutes * 60.0
    readings = {}

    def reading():
        state = engine.state
        return SimpleNamespace(sbp=state.sbp, map=state.map, hr=state.hr, co=state.co)

    for index in range(round((intubation + 180.0) / 0.5)):
        time = index * 0.5
        if time == 60.0:
            engine.give_drug_bolus("propofol", 2.0 * engine.patient.weight)
        if time == intubation:
            engine.start_disturbance("stim_intubation_pulse")
            engine.set_drug_rate("propofol", 100.0)  # mcg/kg/min = 6 mg/kg/h.
            engine.set_fgf(3.0, 0.0, n2o_l_min=3.0)
        engine.step(0.5)
        if time + 0.5 == 240.0:  # Before instrumentation.
            readings["induction"] = reading()
        if time + 0.5 == intubation + 60.0:
            readings["intubation"] = reading()
    readings["after"] = reading()
    return readings


@pytest.mark.parametrize("intubation_minutes", [4.0, 5.0])
def test_hayakawa_propofol_fentanyl_pressure_hr_and_flow(engine_factory, intubation_minutes):
    """Hayakawa-Fujii 2001 Table 1: SBP/DBP 102/57, HR 67, and CO 4.4 L/min
    3 min after induction; MAP near 94 and HR near 72 in the first minute after
    intubation; SBP/DBP 110/61, HR 70, and CO 5.2 L/min 3 min after
    intubation. Bounds allow the reported SD.
    """
    readings = _replay_hayakawa(engine_factory, intubation_minutes)
    induction, intubation, after = readings["induction"], readings["intubation"], readings["after"]
    # MAP = (SBP + 2 DBP) / 3; bounds allow fully correlated BP SDs.
    assert induction.sbp == pytest.approx(102.0, abs=13.0)
    assert induction.map == pytest.approx((102.0 + 2.0 * 57.0) / 3.0, abs=11.0)
    assert induction.hr == pytest.approx(67.0, abs=11.0)
    assert induction.co == pytest.approx(4.4, abs=1.2)
    assert intubation.map == pytest.approx(94.0, abs=20.0)
    assert intubation.hr == pytest.approx(72.0, abs=11.0)
    assert after.sbp == pytest.approx(110.0, abs=20.0)
    assert after.map == pytest.approx((110.0 + 2.0 * 61.0) / 3.0, abs=16.0)
    assert after.hr == pytest.approx(70.0, abs=11.0)
    assert after.co == pytest.approx(5.2, abs=1.3)
