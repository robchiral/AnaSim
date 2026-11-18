import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.arterial import ArterialWaveformRenderer
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.patient.patient import Patient
from anasim.patient.pk_models import NorepinephrinePK, PropofolPKEleveld
from anasim.physiology.disturbances import Disturbances
from anasim.physiology.hemo_config import HemodynamicConfig
from anasim.physiology.hemodynamics import HemodynamicModel
from validation.intubation import (
    POTERMAN_ARMS,
    SJOEN_2023_ARMS,
    SJOEN_ARMS,
    replay_poterman,
    replay_sjoen,
)


def _model(sepsis_severity: float = 0.0) -> HemodynamicModel:
    model = HemodynamicModel(Patient(age=40, weight=70, sex="male"))
    model.sepsis_severity = sepsis_severity
    return model


def _steady_exposure(**ce: float):
    model = _model()
    base = model.step(1.0, 0, 0, 0, -2, 40, 95)
    for _ in range(60):
        state = model.step(1.0, 0, 0, 0, -2, 40, 95, **ce)
    return base, state


def _intubation_peaks(age: float, propofol: float = 3.4, sevo: float = 0.0) -> tuple[float, float]:
    """Peak HR and MAP rises over an unstimulated control at fixed exposure."""
    models = [HemodynamicModel(Patient(age=age)) for _ in range(2)]
    for model in models:
        model.initialize_steady_state(propofol, 0.0, 0.0, mac_sevo=sevo)
    stimulus = Disturbances("stim_intubation_pulse")
    hr_rise = map_rise = 0.0
    for index in range(600):
        effects = stimulus.compute_average(index * 0.2, (index + 1) * 0.2)
        stimulated = models[0].step(
            0.2, propofol, 0.0, 0.0, -2.0, 40.0, 95.0,
            dist_hr=effects.hr, dist_sv=effects.sv, dist_svr=effects.svr, mac_sevo=sevo,
        )
        control = models[1].step(0.2, propofol, 0.0, 0.0, -2.0, 40.0, 95.0, mac_sevo=sevo)
        hr_rise = max(hr_rise, stimulated.hr - control.hr)
        map_rise = max(map_rise, stimulated.map - control.map)
    return hr_rise, map_rise


def test_age_blunts_intubation_tachycardia_with_a_retained_pressor_response():
    """Bullington 1989: less tachycardia with age, with a retained BP response.

    The study used thiopental, so only direction is checked.
    """
    younger, older = _intubation_peaks(30), _intubation_peaks(65)
    assert 0.0 < older[0] < younger[0]
    assert younger[1] > 0.0 and older[1] > 0.0


@pytest.mark.parametrize("age,propofol,sevo,hr_peak,map_peak", [
    (39, 3.4, 0.0, 36.0, 35.0),
    (28, 0.0, 1.0, 38.0, 40.0),
])
def test_intubation_retains_ebert_peak_response(age, propofol, sevo, hr_peak, map_peak):
    """Ebert 1992 Fig. 7 and 1995 Fig. 3: no-opioid peak rises at fixed
    exposures. The 8 bpm/mmHg bounds allow for reading values from the figures.
    """
    hr_rise, map_rise = _intubation_peaks(age, propofol, sevo)
    assert hr_rise == pytest.approx(hr_peak, abs=8.0)
    assert map_rise == pytest.approx(map_peak, abs=8.0)


@pytest.mark.parametrize("renal_function", [1.0, 0.4])
def test_urine_output_tapers_with_pressure_and_preserves_fluid_balance(renal_function):
    patient = Patient(renal_function=renal_function)
    rates = []
    for pressure in (20.0, 40.0, 60.0, 65.0, 90.0):
        model = HemodynamicModel(patient)
        model._prev_map = pressure
        before = model.blood_volume
        model.step(1.0, 0, 0, 0, -2, 40, 95)
        urine = model.total_urine_out_ml
        assert before - model.blood_volume == pytest.approx(urine)
        rates.append(urine * 3600 / patient.weight)
    assert rates[0] == 0.0
    assert rates[0] < rates[1] < rates[2] < rates[3]
    assert 0.4 * renal_function < rates[2] < 0.5 * renal_function
    assert rates[3] == rates[4] == pytest.approx(0.5 * renal_function)


def test_crystalloid_distribution_pressure_age_and_component_conservation():
    """Hahn 2017: Ringer's distributes over time; low MAP and age slow urine.
    Hahn 2021: low MAP also slows distribution, so more stays intravascular.
    Blood loss removes mixed cells and Hb, while water exchange preserves them.
    """
    outcomes = {}
    for age, pressure, dt in ((30, 50, 1.0), (30, 100, 1.0), (70, 100, 1.0), (30, 100, 0.2)):
        model = HemodynamicModel(Patient(age=age), HemodynamicConfig(uop_ml_kg_hr=0.0, third_space_refill_tau_hr=0.0))
        baseline = model.blood_volume
        hemoglobin, cells = model.hb_mass, model.red_cell_volume_ml
        for _ in range(round(1800 / dt)):
            model.add_volume(1000 / 1800 * dt)
            model._prev_map = pressure  # Match the pressure-conditioned volume-kinetic experiment.
            model.step(dt, 0, 0, 0, -2, 40, 95)
        blood_gain = model.blood_volume - baseline
        assert 300.0 < blood_gain < (950.0 if pressure < 81.3 else 600.0)
        assert blood_gain + model.crystalloid.interstitial_ml + model.total_third_space_ml + model.lung_water_ml + model.total_urine_out_ml == pytest.approx(1000, abs=1e-7)
        assert model.hb_conc * model.blood_volume / 100 == pytest.approx(hemoglobin)
        assert model.get_hematocrit() * model.blood_volume == pytest.approx(cells)
        outcomes[age, pressure, dt] = blood_gain, model.total_urine_out_ml
        # Redistribution persists after the infusion stops.
        for _ in range(round(1800 / dt)):
            model._prev_map = pressure
            model.step(dt, 0, 0, 0, -2, 40, 95)
        assert model.blood_volume - baseline < blood_gain
    assert outcomes[30, 50, 1.0][1] < 0.1 * outcomes[30, 100, 1.0][1]
    assert outcomes[30, 50, 1.0][0] > outcomes[30, 100, 1.0][0] + 100.0
    assert outcomes[70, 100, 1.0][1] < 0.4 * outcomes[30, 100, 1.0][1]
    # Infusion enters at the start of each step, giving a half-step timing error.
    assert outcomes[30, 100, 0.2] == pytest.approx(outcomes[30, 100, 1.0], abs=0.3)


@pytest.mark.parametrize(("hematocrit", "dt"), [(0.24, 1.0), (0.32, 2.0)])
def test_transfusion_preserves_donor_cells_and_hemoglobin(
    engine_factory, advance_time, hematocrit, dt,
):
    """The same PRBC unit adds the same Hb to normochromic and hypochromic anemia.

    Hemorrhage removes both red cells and Hb; crystalloid and urine change
    their concentrations without changing their amounts.
    """
    engine = engine_factory(start=True, baseline_hb=8.0, baseline_hct=hematocrit)
    engine.set_continuous_fluid_rate(0.0)
    model = engine.hemo
    remaining_volume = model.blood_volume - 1000.0
    expected_hb_g = 8.0 * remaining_volume / 100.0
    expected_cells_ml = hematocrit * remaining_volume
    model.add_volume(-1000.0)

    engine.give_blood(300.0, hematocrit=0.55)
    engine.give_fluid(500.0)
    advance_time(engine, 300.0, dt=dt)
    expected_hb_g += 55.0  # 300 mL at Hct 0.55 and donor MCHC 33.33 g/dL.
    expected_cells_ml += 165.0
    assert model.total_blood_in_ml == pytest.approx(300.0)
    assert engine.state.hb_g_dl * engine.state.blood_volume / 100.0 == pytest.approx(expected_hb_g)
    assert engine.state.hct * engine.state.blood_volume == pytest.approx(expected_cells_ml)
    assert model.total_urine_out_ml > 0.0

    # Subsequent bleeding removes the mixed donor and recipient cells.
    remaining_fraction = 1.0 - 300.0 / model.blood_volume
    model.add_volume(-300.0)
    advance_time(engine, 10.0, dt=dt)
    assert engine.state.hb_g_dl * engine.state.blood_volume / 100.0 == pytest.approx(expected_hb_g * remaining_fraction)
    assert engine.state.hct * engine.state.blood_volume == pytest.approx(expected_cells_ml * remaining_fraction)


def test_sjoen_remifentanil_doses_give_the_same_induction_fall():
    """Sjøen 2024: remifentanil doses giving peak effect-site concentrations of 2,
    4, and 8 ng/mL with propofol 1.8 mg/kg lowered invasive SAP 30-32% (SD 6-7%),
    with no dose effect. These research inductions fall further than routine
    practice (test_clinical_acceptance.py), so they bound the model within 2 SD.
    """
    falls = [replay_sjoen(arm, dt=1.0)["sap"] for arm in SJOEN_ARMS]
    for fall, arm in zip(falls, SJOEN_ARMS, strict=True):
        assert fall == pytest.approx(arm.sap_change, abs=2 * 6.6)
    assert abs(falls[2] - falls[0]) < 6.0


def test_sjoen_doubled_propofol_deepens_the_induction_fall():
    """Sjøen 2023: propofol 1.4 or 2.7 mg/kg with remifentanil near 2 mcg/kg
    lowered invasive SAP 31 ± 6.6% and 36 ± 7.3%, within 2 SD as above.
    """
    low, high = (replay_sjoen(arm, dt=1.0)["sap"] for arm in SJOEN_2023_ARMS)
    assert low == pytest.approx(SJOEN_2023_ARMS[0].sap_change, abs=2 * 6.6)
    assert high == pytest.approx(SJOEN_2023_ARMS[1].sap_change, abs=2 * 7.3)
    assert high < low


def _pressures(model, renderer, cycle, state, seed=False):
    sample = cycle.seed(state.hr, RhythmType.SINUS) if seed else cycle.step(1.0, state.hr, state.rhythm_type)
    return renderer.step(sample, state.map, state.sv)


def test_ebert_propofol_induction_without_opioid():
    """Ebert 1992, unpremedicated patients aged about 37: propofol 2.5 mg/kg
    then 200 mcg/kg/min lowered SBP 27 and DBP 7-8 mmHg (MAP 16%) by 3-4 min, and
    HR rose 12 bpm. The model's falls run about 5 points deeper, with SBP still
    falling more than DBP; its HR rises less.
    """
    patient = Patient(age=37, weight=81, height=176, sex="male", baseline_map=88.4, baseline_hr=61.5)
    model = HemodynamicModel(patient)
    pk = PropofolPKEleveld(patient, concomitant_opioids=False)
    renderer, cycle = ArterialWaveformRenderer(age=37), CardiacCycle(np.random.default_rng(0))
    awake = model.step(0.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
    baseline = _pressures(model, renderer, cycle, awake, seed=True)
    for second in range(240):
        rate = 2.5 * 81 / 20.0 if second < 20 else 0.2 * 81 / 60.0
        state = model.step(1.0, pk.step(1.0, rate).c1, 0.0, 0.0, -2.0, 40.0, 95.0, remifentanil_cp=0.0)
        pressure = _pressures(model, renderer, cycle, state)
    sbp_fall, dbp_fall = baseline.systolic - pressure.systolic, baseline.diastolic - pressure.diastolic
    assert 10.0 < 100.0 * (1.0 - state.map / awake.map) < 30.0
    assert sbp_fall > 1.5 * dbp_fall


@pytest.mark.parametrize("age", [62.4, 75.0])
def test_claeys_propofol_induction_and_maintenance_without_opioid(age):
    """Claeys 1988 Table II (n=10, ages 50-75, mean 62.4 ± 7.6, lorazepam 1 mg
    and glycopyrrolate 0.4 mg premedication, spontaneous room air with PaCO2
    rising from 38.0 to 42.5 mmHg): propofol 2 mg/kg over 30 s then 6 mg/kg/h
    lowered SAP/DAP from 136 ± 17 / 66 ± 10 mmHg to 99 ± 22 / 53 ± 15 at 2 min,
    93 ± 18 / 50 ± 12 at 6 min, 93 ± 17 / 48 ± 11 at 30 min, and 101 ± 20 /
    51 ± 12 at 60 min. Pressures stay within 1 SD at both the cohort mean and
    upper age, while the model puts more of the sustained fall in SV/CO than SVR.
    """
    patient = Patient(
        age=age, weight=72.1, height=168.5, sex="male",
        baseline_map=(136.0 + 2.0 * 66.0) / 3.0, baseline_hr=74.0,
    )
    model = HemodynamicModel(patient)
    pk = PropofolPKEleveld(patient, concomitant_opioids=False)
    renderer, cycle = ArterialWaveformRenderer(age=age), CardiacCycle(np.random.default_rng(0))
    awake = model.step(0.0, 0.0, 0.0, 0.0, -2.0, 38.0, 95.0)
    _pressures(model, renderer, cycle, awake, seed=True)
    checks = {
        120: (99.0, 22.0, 53.0, 15.0),
        360: (93.0, 18.0, 50.0, 12.0),
        1800: (93.0, 17.0, 48.0, 11.0),
        3600: (101.0, 20.0, 51.0, 12.0),
    }
    for second in range(1, 3601):
        rate = (2.0 * 72.1 / 30.0 if second <= 30 else 0.0) + 6.0 * 72.1 / 3600.0
        ramp = min(1.0, second / 2700.0)
        state = model.step(
            1.0, pk.step(1.0, rate).c1, 0.0, 0.0, -2.0, 38.0 + 4.5 * ramp, 95.0, ph=7.38 - 0.08 * ramp,
        )
        pressure = _pressures(model, renderer, cycle, state)
        if second in checks:
            obs_sbp, sd_sbp, obs_dbp, sd_dbp = checks[second]
            assert pressure.systolic == pytest.approx(obs_sbp, abs=sd_sbp)
            assert pressure.diastolic == pytest.approx(obs_dbp, abs=sd_dbp)


def test_morley_propofol_fall_grows_with_preinduction_pressure():
    """Morley 2010, non-hypertensive adults aged about 38 without opioid: propofol
    40 mg/kg/h until BIS 50 (mean 2.0 mg/kg, so 3 min), then 8 mg/kg/h. The
    largest of 15 one-minute MAP readings fell 25.6% (SD 7.3), 0.44 points more
    per mmHg of baseline, about 0.33 after regression to the mean from a single
    baseline reading.
    """
    falls = {}
    for baseline_map in (80.0, 97.0, 115.0):
        patient = Patient(age=38, weight=72, height=165, sex="female", baseline_map=baseline_map, baseline_hr=75)
        model = HemodynamicModel(patient)
        pk = PropofolPKEleveld(patient, concomitant_opioids=False)
        awake = model.step(0.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0).map
        pressures = []
        for second in range(900):
            rate = (40.0 if second < 180 else 8.0) * 72 / 3600.0
            pressures.append(model.step(1.0, pk.step(1.0, rate).c1, 0.0, 0.0, -2.0, 40.0, 95.0).map)
        readings = [np.mean(pressures[minute * 60 - 40:minute * 60]) for minute in range(1, 16)]
        falls[baseline_map] = 100.0 * (1.0 - min(readings) / awake)
    assert falls[97.0] == pytest.approx(25.6, abs=7.3)
    assert 0.2 < (falls[115.0] - falls[80.0]) / 35.0 < 0.45


def test_van_den_berg_equipotent_propofol_remifentanil_pairs():
    """van den Berg 2021: after 11 min at propofol/remifentanil effect-site
    pairs of 8.6/1, 5.9/2, 3.6/4, and 2.0/8 (equal predicted tolerance of
    laryngoscopy), median SBP was 90-104 mmHg in every group and HR fell from 71
    to 60 with more remifentanil. Ephedrine rescues before the measurement and
    unreported awake values leave the absolute level unchecked.
    """
    readings = []
    for propofol, remifentanil in ((8.6, 1.0), (5.9, 2.0), (3.6, 4.0), (2.0, 8.0)):
        model = HemodynamicModel(Patient(age=53, weight=78, height=172, sex="male"))
        renderer, cycle = ArterialWaveformRenderer(age=53), CardiacCycle(np.random.default_rng(0))
        state = model.step(0.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
        _pressures(model, renderer, cycle, state, seed=True)
        for _ in range(1200):
            state = model.step(1.0, propofol, remifentanil, 0.0, -2.0, 40.0, 95.0)
            pressure = _pressures(model, renderer, cycle, state)
        readings.append((pressure.systolic, state.hr))
    systolic = [sbp for sbp, _ in readings]
    assert all(85.0 <= sbp <= 109.0 for sbp in systolic)  # Observed 90-104.
    assert readings[0][1] > readings[3][1] + 5.0


def test_poterman_high_dose_remifentanil_with_and_without_atropine():
    """Poterman 2017 Table 2: propofol Ce 2.5 mcg/mL and remifentanil Ce 8 ng/mL
    lowered MAP by a median 37 mmHg (IQR 31-41) and HR by 19 ± 11 bpm at 10 min.
    Methylatropine at induction kept HR unchanged (0 ± 13 bpm) and limited the
    MAP fall to 24 mmHg (IQR 21-40). The model's HR fall is about half the
    observed mean. Like other research inductions, these bound the model within
    2 SD, taking the SD as the IQR over 1.35.
    """
    results = {name: replay_poterman(arm) for name, arm in POTERMAN_ARMS.items()}
    for name, arm in POTERMAN_ARMS.items():
        map_change, hr_change = results[name]
        spread = (arm.map_change[2] - arm.map_change[1]) / 1.35
        assert map_change == pytest.approx(arm.map_change[0], abs=2 * spread)
        assert hr_change == pytest.approx(arm.hr_change[0], abs=arm.hr_change[1])
    assert results["atropine"][0] > results["saline"][0]
    assert results["atropine"][1] > results["saline"][1]


def _kazama_sbp(age, target, ic50, gamma):
    """Model SBP fall (%) after 25 min at constant plasma propofol, and the fall
    Kazama's curve predicts as a share of the fall from baseline to 80 mmHg."""
    patient = Patient(age=age, weight=70, height=170, sex="male", baseline_map=(130.0 + 2.0 * 80.0) / 3.0)
    hemo = HemodynamicModel(patient)
    nore = NorepinephrinePK(patient)
    nore.equilibrate_endogenous()
    hemo.initialize_steady_state(0.0, 0.0, nore.state.ce)
    renderer = ArterialWaveformRenderer(age=age)
    cycle = CardiacCycle(np.random.default_rng(0))
    s0 = hemo.state
    baseline = renderer.step(
        cycle.seed(s0.hr, RhythmType.SINUS), s0.map, s0.sv, pulse_pressure_factor=s0.pulse_pressure_factor,
    ).systolic
    nore.update_propofol(target)
    for _ in range(300):
        nore.step(5.0, 0.0)
        state = hemo.step(5.0, target, 0.0, nore.state.ce, -2.0, 40.0, 95.0, remifentanil_cp=0.0)
    sbp = renderer.step(
        cycle.seed(state.hr, state.rhythm_type), state.map, state.sv,
        pulse_pressure_factor=state.pulse_pressure_factor,
    ).systolic
    expected = 100.0 * (1.0 - 80.0 / baseline) * target**gamma / (ic50**gamma + target**gamma)
    return 100.0 * (1.0 - sbp / baseline), expected


@pytest.mark.parametrize("age,ic50,gamma", [(30, 4.61, 1.75), (50, 4.13, 2.12), (65, 3.96, 1.64)])
def test_kazama_equilibrated_propofol_sbp_by_age(age, ic50, gamma):
    """Kazama 1999 Table 2: SBP after 22-28 min at constant plasma propofol, with
    these IC50s and slopes."""
    for target in (4.0, 6.0):
        fall, expected = _kazama_sbp(age, target, ic50, gamma)
        assert fall == pytest.approx(expected, abs=5.0)


def test_kazama_oldest_group_falls_further():
    """Kazama 1999: at 70-85 years (mean 79) the SBP IC50 was 2.09 mcg/mL (SE
    0.19, n=10), below the 3.96 at 60-69. At 4 mcg/mL the fall exceeds the curve
    at IC50 plus 2 SD."""
    fall, bound = _kazama_sbp(79, 4.0, 2.09 + 2.0 * 0.19 * 10**0.5, 1.98)
    younger, _ = _kazama_sbp(65, 4.0, 3.96, 1.64)
    assert fall > bound
    assert fall > younger


def test_pressors_raise_svr_without_beta_effects():
    base, phenyl = _steady_exposure(ce_phenyl=100.0)
    assert phenyl.map > base.map + 10.0
    assert phenyl.svr > base.svr + 4.0
    # Reflex slowing can raise output SV through filling, without beta inotropy.
    assert phenyl.sv_star < base.sv_star
    assert phenyl.hr < base.hr
    assert phenyl.co < base.co

    base, vaso = _steady_exposure(ce_vaso=40.0)
    assert vaso.map > base.map + 5.0
    assert vaso.svr > base.svr * 1.1
    assert vaso.hr <= base.hr + 5.0


def test_propofol_changes_venous_pressure_and_resistance_without_losing_blood():
    """de Wit 2016, Table 1: relative MSFP and Rvr at Cp 3 and 6.5 mcg/mL."""
    observations = []
    for propofol in (3.0, 6.5):
        model = HemodynamicModel(Patient(age=62), replace(HemodynamicConfig(), vol_clearance=0.0))
        volume, hemoglobin = model.blood_volume, model.hb_mass
        model.initialize_steady_state(propofol, 0.0, 0.0)
        for _ in range(600):
            state = model.step(1.0, propofol, 0.0, 0.0, -2.0, 40.0, 95.0)
        assert model.blood_volume == pytest.approx(volume)
        assert model.hb_mass == pytest.approx(hemoglobin)
        observations.append((state.mcfp, model.venous_return_resistance))
    low, high = observations
    assert high[0] / low[0] == pytest.approx(21.4 / 27.9, rel=0.01)
    assert high[1] / low[1] == pytest.approx(2.6 / 3.7, rel=0.01)


def test_phenylephrine_output_response_depends_on_preload():
    """Rebet 2016: CO falls at normal preload and is preserved at low preload."""
    responses = []
    for deficit in (0.0, 0.15):
        model = HemodynamicModel(Patient(age=64), replace(HemodynamicConfig(), vol_clearance=0.0))
        model.initialize_steady_state(3.5, 3.6, 0.0)
        model.add_volume(-deficit * model.blood_volume)
        volume, hemoglobin = model.blood_volume, model.hb_mass
        for _ in range(600):
            before = model.step(1.0, 3.5, 3.6, 0.0, -2.0, 40.0, 95.0)
        for _ in range(60):
            after = model.step(1.0, 3.5, 3.6, 0.0, -2.0, 40.0, 95.0, ce_phenyl=4.0)
        assert after.map > before.map + 10.0
        assert after.mcfp > before.mcfp
        assert after.hr < before.hr
        assert model.blood_volume == pytest.approx(volume)
        assert model.hb_mass == pytest.approx(hemoglobin)
        responses.append((after.co / before.co, after.sv / before.sv))
    normal, low = responses
    assert 0.7 < normal[0] < 0.9 and normal[1] < 0.95
    assert 0.9 < low[0] < 1.1 and low[1] > 1.0


@pytest.mark.parametrize("propofol,sevo,nore", [
    (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (3.53, 0.0, 10.0),
])
def test_steady_state_initialization_keeps_fast_and_slow_drug_effects(patient, propofol, sevo, nore):
    model = HemodynamicModel(patient, replace(HemodynamicConfig(), vol_clearance=0.0))
    initialized = model.initialize_steady_state(propofol, 3.6, nore, mac_sevo=sevo)
    if nore:
        assert initialized.map > patient.baseline_map
    initial_values = (initialized.map, initialized.hr, initialized.co)
    assert (model.state.map, model.state.hr, model.state.co) == pytest.approx(initial_values)
    for second in range(1, 901):
        state = model.step(1.0, propofol, 3.6, nore, -2.0, 40.0, 95.0, mac_sevo=sevo)
        if second in (60, 300, 900):
            assert (state.map, state.hr, state.co) == pytest.approx(initial_values, rel=0.01)

def test_ebert_sevoflurane_induction_and_steady_state():
    """Ebert 1995, volunteers aged 19-32 without stimulation: propofol 2 mg/kg
    lowered MAP 14% before mask sevoflurane began, and the fall reached 24-25%
    by 4.5-9.5 minutes as end-tidal rose to 2.1% (Fig. 1). After intubation, MAP
    was 24, 29, and 40% below awake 10 minutes after end-tidal reached 1, 2, and
    3% (Fig. 4). These rested volunteers fall further than routine patients
    (test_clinical_acceptance.py), so they bound the model within 2 SD, taking
    the SD as 9 points from the figures' standard errors with 12 subjects.
    """
    patient = Patient(age=25, weight=75, height=178, sex="male", baseline_map=92, baseline_hr=60)
    model = HemodynamicModel(patient)
    pk = PropofolPKEleveld(patient, concomitant_opioids=False)
    mac = 1.80 * 10 ** (-0.00269 * (25 - 40))
    awake = model.step(0.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0).map
    ramp = (0.2, 0.5, 1.0, 1.4, 1.5, 1.8, 2.0, 2.1, 2.1, 2.1)  # End-tidal % by minute
    intubation = Disturbances("stim_intubation_pulse")
    falls = {}
    for second in range(4560):
        if second < 120:
            end_tidal = 0.0
        elif second < 720:
            end_tidal = ramp[(second - 120) // 60]
        else:
            # Vaporizer 1% for 20 min, then end-tidal steps to 2 and 3% every 18 min.
            step = min(2, max(0, (second - 1920) // 1080 + 1))
            start = 720 if step == 0 else 1920 + (step - 1) * 1080
            previous, target = ((2.1, 1.0), (1.0, 2.0), (2.0, 3.0))[step]
            end_tidal = target + (previous - target) * math.exp(-(second - start) / 180.0)
        effects = intubation.compute_average(second - 720.0, second - 719.0)
        state = model.step(1.0, pk.step(1.0, 7.5 if second < 20 else 0.0).c1, 0.0, 0.0, -2.0, 40.0, 95.0,
                           mac_sevo=end_tidal / mac, dist_hr=effects.hr, dist_sv=effects.sv, dist_svr=effects.svr)
        falls[second + 1] = 100.0 * (1.0 - state.map / awake)
    # Fig. 1 falls by second: before the vaporizer started, then 0.5-9.5 min after.
    figure1 = {110: 14, 150: 16, 210: 18, 270: 20, 330: 22, 390: 24, 570: 24, 630: 25, 690: 25}
    for second, fall in figure1.items():
        assert falls[second] == pytest.approx(fall, abs=2 * 9.0)
    for second, fall in ((1920, 24), (3000, 29), (4080, 40)):
        assert falls[second] == pytest.approx(fall, abs=2 * 9.0)
    assert falls[690] > falls[150] + 3.0  # Gradual fall on mask sevoflurane.
    assert falls[1920] < falls[3000] < falls[4080]


def test_sevo_effect_site_kinetics_do_not_depend_on_step_size(patient):
    expected_peak = -math.expm1(-HemodynamicConfig().ke0_sevo * 5.0)
    for dt in (1.0, 300.0):
        model = HemodynamicModel(patient, replace(HemodynamicConfig(), vol_clearance=0.0))
        for _ in range(round(300.0 / dt)):
            state = model.step(dt, 0, 0, 0, -2, 40, 95, mac_sevo=1.0)
        assert state.ce_sevo == pytest.approx(expected_peak)
        for _ in range(round(300.0 / dt)):
            state = model.step(dt, 0, 0, 0, -2, 40, 95)
        assert state.ce_sevo == pytest.approx(expected_peak * (1.0 - expected_peak))


def test_thoracic_pressure_tracks_breathing_effort_and_applied_cpap(awake_engine):
    engine = awake_engine
    pressure = []
    for _ in range(150):
        engine.step(0.1)
        pressure.append(engine.state.pit)
    assert max(pressure) == pytest.approx(engine.hemo.config.pit_0, abs=0.01)
    assert min(pressure) < engine.hemo.config.pit_0 - 2.0

    engine.set_airway_mode("ETT")
    engine.set_fgf(6.0, 0.0)
    engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
    for _ in range(120):
        engine.step(1.0)
    assert engine.resp_mech.effort.amplitude == 0.0
    assert engine.state.pit == pytest.approx(engine.hemo.config.pit_0, abs=0.1)

    engine.set_vent_settings(rr=0, vt=0, peep=5, ie="1:2", mode="CPAP")
    engine.set_vent_power(True)
    for _ in range(40):
        engine.step(0.5)
    assert engine.state.pit > engine.hemo.config.pit_0 + 1.8


@pytest.mark.parametrize(("drug", "min_co_ratio"), [("ce_dobu", 1.1), ("ce_mil", 1.05)])
def test_inodilators_raise_output_and_lower_svr(drug, min_co_ratio):
    """Magnani 1977 (dobutamine); Baim 1983 (milrinone)."""
    base, state = _steady_exposure(**{drug: 200.0})
    assert state.co > base.co * min_co_ratio
    assert state.svr < base.svr * 0.9


def test_hypoxia_and_peep_raise_pvr():
    """Carlsson 1985 (hypoxic vasoconstriction); Koganov 1997 (PEEP)."""
    for pao2, peep in ((50, 5.0), (95, 15.0)):
        model = _model()
        base = model.step(1.0, 0, 0, 0, -2, 40, 95, peep_cmH2O=5.0)
        for _ in range(30):
            state = model.step(1.0, 0, 0, 0, -2, 40, pao2, peep_cmH2O=peep)
        assert state.pvr > base.pvr * 1.2


def test_pulmonary_transit_delays_lv_inflow(patient):
    for dt in (0.1, 1.0, 6.8):
        model = HemodynamicModel(patient, replace(HemodynamicConfig(), vol_clearance=0.0))
        base = model.step(0.0, 0, 0, 0, -2, 40, 95, peep_cmH2O=5.0)
        # Acute hypoxia and high PEEP drop RV output; LV inflow follows after transit.
        elapsed = 0.0
        while elapsed < model.config.pulmonary_transit_time_s - 1e-10:
            interval = min(dt, model.config.pulmonary_transit_time_s - elapsed)
            state = model.step(interval, 0, 0, 0, -2, 40, 30, peep_cmH2O=20.0)
            elapsed += interval
        assert state.rv_co < state.lv_inflow < base.lv_inflow
        expected = state.rv_co + (base.lv_inflow - state.rv_co) * math.exp(-1.0)
        assert state.lv_inflow == pytest.approx(expected)


class TestSepsis:
    def test_capillary_leak_reduces_volume(self):
        model = _model(sepsis_severity=1.0)
        base_volume = model.blood_volume
        for _ in range(360):
            model.step(10.0, 0, 0, 0, -2, 40, 95)
        assert model.blood_volume < base_volume - 120.0

    def test_sepsis_blunts_norepinephrine_vasoconstriction(self):
        """Compare SVR rather than MAP, which also includes reflex HR (Bellissant 2000)."""

        def svr_response(sepsis_severity: float) -> float:
            model = _model(sepsis_severity)
            for _ in range(300):
                base = model.step(1.0, 0, 0, 0, -2, 40, 95)
            for _ in range(60):
                treated = model.step(1.0, 0, 0, 10.0, -2, 40, 95)
            return treated.svr - base.svr

        control = svr_response(0.0)
        septic = svr_response(1.0)
        assert control > 5.0
        assert septic < control * 0.8


class TestReflexesAndHypoxia:
    """Fast baroreflex and myocardial hypoxia in the integrated engine."""

    @staticmethod
    def _step_until_sao2_below(engine, sao2, max_seconds=600.0):
        for _ in range(int(max_seconds / 0.5)):
            if engine.state.sao2 <= sao2:
                return True
            engine.step(0.5)
        return False

    def test_severe_hypoxemia_slows_the_heart_and_oxygen_rescues(self, engine_factory):
        engine = engine_factory(config=SimulationConfig(dt=0.5, rng_seed=1), start=True)
        engine.give_drug_bolus("propofol", 140.0)
        engine.give_drug_bolus("roc", 42.0)
        assert self._step_until_sao2_below(engine, 60.0)
        hr_at_60 = engine.state.hr
        assert self._step_until_sao2_below(engine, 35.0)
        assert engine.state.hr < hr_at_60 - 10.0
        for _ in range(30):
            engine.step(0.5)
        assert engine.state.spo2_signal_valid
        assert engine.state.display_spo2 < 40.0
        assert engine.state.alarms["SpO2"]["low"]

        engine.set_airway_mode("Mask")
        engine.set_fgf(10.0, 0.0)
        engine.set_bag_mask_ventilation(True, 12.0, 0.5)
        depressed_map = engine.state.map
        for _ in range(360):
            engine.step(0.5)
        assert engine.state.sao2 > 97.0
        assert engine.state.display_spo2 > 94.0
        assert "SpO2" not in engine.state.alarms
        assert engine.state.map > depressed_map + 15.0

    def test_tachycardia_unloading_and_vascular_baroreflex_bounds(
        self, engine_factory, advance_time, monkeypatch,
    ):
        """Arrhythmias, bradycardia, arrest, PPV swings, heart failure, and pacing offset stay bounded."""
        for rhythm in ("AFIB", "SVT", "VTACH"):
            e = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
            e.set_airway_mode("Mask")
            e.set_fgf(8.0, 0.0)
            advance_time(e, 30)
            sinus_co = e.state.co
            e.set_rhythm(rhythm)
            advance_time(e, 5)
            assert e.state.co < sinus_co
            assert e.hemo._unstim_rate == 0.0 and e.hemo._unstim_rate_unloaded == 0.0

        brady = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
        brady.set_rhythm("SINUS_BRADY")
        advance_time(brady, 60)
        brady.give_drug_bolus("phenyl", 100.0)
        for _ in range(60):
            brady.step(1.0)
            assert brady.hemo.smoothed_baro_vasc == 0.0
        brady.give_drug_bolus("atropine", 0.5)
        advance_time(brady, 180)
        assert brady.state.hr < brady.hemo.base_hr
        assert brady.hemo._unstim_rate == 0.0 and brady.hemo._unstim_rate_unloaded == 0.0

        arrest = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=7), start=True)
        arrest.set_rhythm("VFIB")
        advance_time(arrest, 120)
        arrest.set_chest_compressions(True)
        arrest.give_drug_bolus("atropine", 1.0)
        arrest.give_drug_bolus("epi", 2000.0)
        for _ in range(120):
            arrest.step(1.0)
            assert arrest.hemo.smoothed_baro_vasc == 0.0
            assert arrest.hemo._unstim_rate == 0.0 and arrest.hemo._unstim_rate_unloaded == 0.0

        vent = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1), start=True,
        )
        base_sv = [vent.step(0.5) or vent.state.sv for _ in range(40)]
        base_osc = max(base_sv[-20:]) - min(base_sv[-20:])
        vent.give_drug_bolus("atropine", 1.0)
        atro_sv = [vent.step(0.5) or vent.state.sv for _ in range(240)]
        assert max(atro_sv[-20:]) - min(atro_sv[-20:]) < 1.25 * base_osc

        hfref = engine_factory(
            config=SimulationConfig(mode="awake", rng_seed=1), start=True, heart_failure="hfref",
        )
        hfref.set_airway_mode("Mask")
        hfref.set_fgf(8.0, 0.0)
        advance_time(hfref, 60)
        hf_sv0 = hfref.state.sv
        hfref.give_drug_bolus("atropine", 1.0)
        advance_time(hfref, 120)
        assert hfref.state.sv < hf_sv0

        paced = _model()
        base_paced = paced.step(1.0, 0, 0, 0, -2, 40, 95)
        with monkeypatch.context() as mp:
            mp.setattr(paced, "_rhythm_rate", lambda _: 120.0)
            for _ in range(240):
                paced.step(1.0, 0, 0, 0, -2, 40, 95)
        for _ in range(15):
            recovered = paced.step(1.0, 0, 0, 0, -2, 40, 95)
        assert recovered.sv == pytest.approx(base_paced.sv, rel=0.03)


def test_matched_bleeding_high_peep_rap_and_beta_blocked_contractility_gating(engine_factory, advance_time):
    """Matched hemorrhage and PRBC replacement keeps slow SV adaptation near zero;
    volume loading with high PEEP raises RAP above baseline while clamping preload
    coupling; beta blockade suppresses catecholamine SV compensation during shock.
    """
    matched = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    overload = copy.deepcopy(matched)
    unblocked = copy.deepcopy(matched)
    blocked = copy.deepcopy(matched)
    base_co = matched.state.co
    matched.give_calcium_chloride(1000.0)
    for _ in range(240):
        matched.hemo.add_volume(-10.0)
        matched.hemo.add_volume(10.0, hematocrit=0.38, label="blood")
        matched.step(1.0)
    assert abs(matched.hemo.tde_sv) < 1.0
    assert matched.state.co == pytest.approx(base_co, rel=0.10)

    overload.give_fluid(2500.0)
    overload.set_vent_settings(rr=12, vt=0.5, peep=25.0, ie="1:2", mode="VCV")
    advance_time(overload, 60.0, dt=1.0)
    assert overload.hemo.state.rap > 7.0
    overload.hemo.step(1.0, 0.0, 0.0, 0.0, pit=50.0, paco2=40.0, pao2=95.0, peep_cmH2O=45.0)
    assert overload.hemo.f_preload_pit < 0.35

    blocked.give_drug_bolus("esmolol", 150.0)
    blocked.set_drug_rate("esmolol", 300.0)
    blocked.hemo.myocardial_hypoxia = 0.9
    for model in (unblocked, blocked):
        model.hemo.add_volume(-1500.0)
        advance_time(model, 180.0, dt=1.0)
    assert blocked.hemo.sv_tone < unblocked.hemo.sv_tone


def test_stim_map_bounds_afib_baroreflex_elderly_cfr_and_hfref_startup(engine_factory, advance_time):
    """_stim_map stays bounded by realized pressure in sinus bradycardia;
    hemorrhagic shock in AFIB does not invert vascular baroreflex;
    awake elderly low-CFR baseline has no resting ischemia;
    HFrEF balanced steady-state startup holds MAP without jump.
    """
    brady = engine_factory(
        config=SimulationConfig(
            mode="steady_state", disturbance_profile="stim_sustained_surgery", rng_seed=1
        ),
        start=True,
    )
    brady.set_rhythm("SINUS_BRADY")
    advance_time(brady, 30.0, dt=1.0)
    assert 0.0 <= brady.hemo._stim_map <= brady.hemo.state.intrinsic_map

    afib = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    afib.set_rhythm("AFIB")
    afib.hemo.add_volume(-1500.0)
    advance_time(afib, 60.0, dt=1.0)
    assert afib.hemo.smoothed_baro_vasc >= 0.0

    elderly = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True,
        age=85,
        coronary_flow_reserve=1.2,
    )
    elderly.step(1.0)
    assert elderly.hemo.coronary.supply_demand_ratio >= 1.0
    assert elderly.hemo.coronary.st_depression_mv == 0.0

    hfref = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1),
        start=True,
        heart_failure="hfref",
    )
    map_0 = hfref.state.map
    advance_time(hfref, 10.0, dt=0.5)
    assert hfref.state.map == pytest.approx(map_0, abs=2.5)


def test_awake_startup_equilibration_across_baseline_states(engine_factory, advance_time):
    """Awake initialization holds step-1 MAP, gravid supine baseline, and cold baseline steady."""
    awake_hi = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True,
        baseline_map=105.0,
        baseline_hr=95.0,
    )
    awake_hi.step(0.1)
    assert awake_hi.state.map == pytest.approx(105.0, abs=0.2)
    assert awake_hi.state.hr == pytest.approx(95.0, abs=0.2)

    gravid = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True,
        sex="female",
        gestational_age_weeks=39.0,
    )
    gravid_map_0 = gravid.state.map
    advance_time(gravid, 60.0, dt=1.0)
    assert gravid.state.map == pytest.approx(gravid_map_0, abs=2.5)

    cold = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True,
        baseline_temp=28.0,
    )
    cold_map_0 = cold.state.map
    advance_time(cold, 180.0, dt=1.0)
    assert cold.state.map == pytest.approx(cold_map_0, abs=2.5)
    assert cold.state.lactate < 1.05


def test_respiratory_hemodynamic_coupling_and_alpha_blocked_cold_tone(
    engine_factory, advance_time
):
    """Spontaneous high-drive MAP is step-size invariant, positive-pressure inspiration
    augments LV stroke volume, and alpha blockade blunts hypothermic vasoconstrictor tone.
    """
    dt_maps = []
    for dt in (0.10, 1.00):
        e = engine_factory(config=SimulationConfig(mode="awake", dt=dt, rng_seed=1), start=True)
        e.give_drug_bolus("lidocaine", 500.0)
        e.give_lipid_emulsion(105.0)
        advance_time(e, 120.0, dt=dt)
        dt_maps.append(e.state.map)
    assert max(dt_maps) - min(dt_maps) < 1.5

    ppv = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.1, rng_seed=1), start=True)
    advance_time(ppv, 10.0, dt=0.2)
    insp_sv, exp_sv = [], []
    for _ in range(150):
        ppv.step(0.1)
        if ppv.vent.flow > 0.05:
            insp_sv.append(ppv.state.sv)
        elif ppv.state.paw <= ppv.vent.settings.peep + 0.5:
            exp_sv.append(ppv.state.sv)
    assert sum(insp_sv) / len(insp_sv) >= sum(exp_sv) / len(exp_sv)

    cold_unblocked, cold_blocked = _model(), _model()
    for model, labetalol in ((cold_unblocked, 0.0), (cold_blocked, 1000.0)):
        model.initialize_steady_state(2.5, 3.6, 0.0)
        for _ in range(60):
            model.step(5.0, 2.5, 3.6, 0.0, -1.4, 40.0, 95.0, temp_c=33.0, ce_labetalol=labetalol)
    assert cold_blocked.tpr_tone < cold_unblocked.tpr_tone - 0.0005


def test_hematocrit_bounds_lipid_retention_vasopressin_alpha_invariance_and_afib_baroreflex(
    engine_factory, advance_time
):
    """Blood/volume hematocrit inputs reject invalid values; 20% lipid uses 0.25 vascular
    retention; labetalol alpha-1 block does not attenuate vasopressin SVR rise; hypertensive
    AFib above the 110 bpm RVR floor slows rate without engaging ectopic baro_vasc.
    """
    hemo = _model()
    for bad_hct in (-0.1, 1.2, float("nan")):
        with pytest.raises(ValueError, match="hematocrit"):
            hemo.add_volume(100.0, hematocrit=bad_hct)
    eng = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    with pytest.raises(ValueError, match="hematocrit"):
        eng.give_blood(300.0, hematocrit=0.0)

    lipid_hemo = _model()
    bv0 = lipid_hemo.blood_volume
    lipid_hemo.add_volume(100.0, hematocrit=0.0, label="lipid")
    assert lipid_hemo.blood_volume - bv0 == pytest.approx(25.0, abs=0.5)

    unblocked, blocked = _model(), _model()
    unblocked.step(1.0, 0, 0, 0, -2, 40, 95, ce_labetalol=0.0, ce_vaso=0.0)
    u0_delta = unblocked.delta_tpr_vasopressors
    unblocked.step(1.0, 0, 0, 0, -2, 40, 95, ce_labetalol=0.0, ce_vaso=40.0)
    u1_delta = unblocked.delta_tpr_vasopressors
    blocked.step(1.0, 0, 0, 0, -2, 40, 95, ce_labetalol=500.0, ce_vaso=0.0)
    b0_delta = blocked.delta_tpr_vasopressors
    blocked.step(1.0, 0, 0, 0, -2, 40, 95, ce_labetalol=500.0, ce_vaso=40.0)
    b1_delta = blocked.delta_tpr_vasopressors
    assert (u1_delta - u0_delta) == pytest.approx(b1_delta - b0_delta, rel=1e-9)

    afib = _model()
    afib.rhythm_type = RhythmType.AFIB
    afib.smoothed_epi_hr = 35.0
    for _ in range(5):
        afib.step(1.0, 0, 0, 0, -2, 40, 95, ce_epi=3.0, ce_phenyl=60.0)
    assert afib.state.hr > 115.0
    assert afib.smoothed_baro_vasc == 0.0


def test_body_size_mcfp_starling_collapse_afib_ceiling_and_eval_purity(engine_factory, advance_time):
    """Venous compliance scales with baseline blood volume so resting MCFP matches across body size;
    intrathoracic pressure above MCFP collapses venous return; AFib + high-dose epinephrine stays
    below 205 bpm with forward CO; evaluate_state() does not mutate _stim_map; shock stunning stacks
    multiplicatively with hypoxic depression.
    """
    small_hemo = HemodynamicModel(Patient(weight=50, height=155, sex="female"))
    large_hemo = HemodynamicModel(Patient(weight=160, height=185, sex="male"))
    assert small_hemo.mcfp_0 == pytest.approx(large_hemo.mcfp_0, rel=1e-6)

    tamponade = _model()
    for _ in range(60):
        s_coll = tamponade.step(1.0, 0, 0, 0, pit=25.0, paco2=40.0, pao2=95.0)
    assert s_coll.rap >= 25.0 and s_coll.co < 0.05

    afib_epi = _model()
    afib_epi.rhythm_type = RhythmType.AFIB
    for _ in range(30):
        s_af = afib_epi.step(1.0, 0, 0, 0, -2, 40, 95, ce_epi=25.0)
    assert s_af.hr <= 205.0 and s_af.co > 1.5

    eval_hemo = _model()
    eval_hemo.step(1.0, 0, 0, 0, -2, 40, 95)
    stim_before = eval_hemo._stim_map
    for _ in range(10):
        eval_hemo.evaluate_state(include_stim=True)
    assert eval_hemo._stim_map == stim_before

    normoxic, hypoxic = _model(), _model()
    normoxic.coronary.shock_stunning = hypoxic.coronary.shock_stunning = 0.25
    hypoxic.myocardial_hypoxia = 0.85
    s_norm = normoxic.step(1.0, 0, 0, 0, -2, 40, 95)
    s_hyp = hypoxic.step(1.0, 0, 0, 0, -2, 40, 30)
    assert s_hyp.sv < 0.65 * s_norm.sv

    hfref_eng = engine_factory(
        heart_failure="hfref",
        config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1),
        start=True,
    )
    hfref_eng.set_drug_rate("phenyl", 100.0 / 60.0)
    hfref_eng.give_fluid(1000.0)
    advance_time(hfref_eng, 300.0, dt=1.0)
    hs_hfref = hfref_eng.hemo.state
    assert hs_hfref.rv_co == pytest.approx(hs_hfref.intrinsic_co, abs=0.25)
