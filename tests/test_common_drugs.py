"""Atropine, ephedrine, succinylcholine, morphine, and hydromorphone against published protocols."""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.patient.pd.anesthesia import opioid_rate_fraction
from anasim.patient.pk_models import HydromorphonePK, MorphinePK
from anasim.scenarios.base import require_tof_at_most


@pytest.mark.parametrize(
    "model_type,v1,rates,dose,half_life,rr_half_life,c25,gamma",
    [
        (MorphinePK, 20.0, (5.1, 4.3, 1.3, 1.2, 0.306), 14_000.0,
         3.3 / 0.694, 2.2 / 0.736, 7.7 / 0.518, 0.43 / 1.08),
        (HydromorphonePK, 13.6 / 0.908, (6.1, 6.6, 1.6, 1.6, 0.41), 3500.0,
         3.3 * 0.694, 2.2 * 0.736, 7.7 * 0.518, 0.43 * 1.08),
    ],
)
def test_opioid_two_hour_infusion_matches_source_odes(
    model_type, v1, rates, dose, half_life, rr_half_life, c25, gamma,
):
    """Meissner 2025 PK and Olofsen 2026 controller and rate effect sites.

    Independently solve the amount equations for 0.2 mg/kg morphine and
    0.05 mg/kg hydromorphone over two hours, followed by washout.
    """
    model = model_type(Patient(age=28, weight=70, height=170, sex="male"))
    k10, k12, k21, k13, k31 = rates
    matrix = np.array([
        [-k10 - k12 - k13, k21, k31, 0, 0],
        [k12, -k21, 0, 0, 0], [k13, 0, -k31, 0, 0],
        [math.log(2) / half_life / v1, 0, 0, -math.log(2) / half_life, 0],
        [math.log(2) / rr_half_life / v1, 0, 0, 0, -math.log(2) / rr_half_life],
    ])
    transition = expm(matrix / 60.0)
    input_vector = np.array([dose / 2.0, 0, 0, 0, 0])  # mcg/h for two hours
    equilibrium = np.linalg.solve(-matrix, input_vector)
    expected = np.zeros(5)
    for minute in range(360):
        target = equilibrium if minute < 120 else np.zeros(5)
        expected = transition @ (expected - target) + target
        model.step(60.0, dose / 7200.0 if minute < 120 else 0.0)
        if minute in (9, 119, 239, 359):
            assert model.state.c1 == pytest.approx(expected[0] / v1, rel=0.015)
            assert model.state.ce_resp == pytest.approx(expected[3], rel=0.015)
            assert model.state.ce_rr == pytest.approx(expected[4], rel=0.015)
            concentrations = (model.state.ce_rr, 0.0) if model_type is MorphinePK else (0.0, model.state.ce_rr)
            expected_rate = 1.0 / (1.0 + (expected[4] / c25) ** gamma / 3.0)
            assert opioid_rate_fraction(*concentrations) == pytest.approx(expected_rate, rel=0.005)
    assert model.state.ce_resp > model.state.c1


@pytest.mark.parametrize("drug,dose,peak_hours,rate_fraction,nadir_hours", [
    ("morphine", 0.2, (2.7, 8.3), 0.740, (2.3, 2.8)),
    ("hydromorphone", 0.05, (1.5, 4.5), 0.696, (2.0, 2.5)),
])
def test_long_opioid_infusions_follow_separate_respiratory_endpoints(
    engine_factory, advance_time, drug, dose, peak_hours, rate_fraction, nadir_hours,
):
    """Meissner 2023: 2-h infusions of morphine 0.2 or hydromorphone 0.05 mg/kg.

    CO2 checks use observed peak sizes and timing. Rate checks use Olofsen
    2026 typical PK/PD predictions, independently reproduced above; the
    observed mean rate nadirs (53-65% of baseline) remain a validation gap.
    """
    e = engine_factory(config=SimulationConfig(mode="awake", rng_seed=4), start=True)
    baseline_rr, baseline_co2 = e.state.rr, e.state.pa_co2
    e.set_drug_rate(drug, dose * e.patient.weight / 2.0)
    peak_co2, peak_hour, min_rr, nadir_hour = 0.0, 0.0, baseline_rr, 0.0
    for minute in range(1, 4 * 60 + 1):
        advance_time(e, 60, dt=5.0)
        if minute == 120:
            e.set_drug_rate(drug, 0.0)
            assert e.state.bis > 85.0
        if e.state.pa_co2 - baseline_co2 > peak_co2:
            peak_co2, peak_hour = e.state.pa_co2 - baseline_co2, minute / 60
        if e.state.rr < min_rr:
            min_rr, nadir_hour = e.state.rr, minute / 60.0
    assert 4.0 < peak_co2 < 12.0
    assert min_rr / baseline_rr == pytest.approx(rate_fraction, abs=0.02)
    assert nadir_hours[0] < nadir_hour < nadir_hours[1]
    assert peak_hours[0] < peak_hour < peak_hours[1]


@pytest.mark.parametrize("propofol", [False, True])
def test_atropine_dose_response_awake_and_during_propofol(engine_factory, advance_time, propofol):
    """Horiguchi 2002: atropine 5 mcg/kg every 2 min, HR read 2 min after each dose.

    At 10 mcg/kg HR rose 28 ± 13 bpm awake and 9 ± 6 bpm with propofol
    10 mg/kg/h and N2O. At 20 mcg/kg all awake patients but only 13% of that
    group exceeded +20 bpm. HR follows Hinderling's peripheral compartment.
    """
    e = engine_factory(config=SimulationConfig(mode="awake", rng_seed=4), start=True)
    weight = e.patient.weight
    if propofol:
        e.set_airway_mode("ETT")
        e.set_vent_power(True)
        e.set_fgf(2.0, 0.0, 4.0)
        e.give_drug_bolus("propofol", 2.5 * weight)
        e.set_drug_rate("propofol", 10.0 * 1000 / 60)
        e.give_drug_bolus("roc", 0.6 * weight)
        advance_time(e, 25 * 60, dt=0.5)
    baseline = e.state.hr
    rises = []
    for _ in range(4):
        e.give_drug_bolus("atropine", 0.005 * weight)
        advance_time(e, 120, dt=0.5)
        rises.append(e.state.hr - baseline)
    if propofol:
        assert 3 < rises[1] < 15
        assert rises[3] < 20
    else:
        assert 15 < rises[1] < 41
        assert rises[3] > 20
    assert e.pk["atropine"].state.ce == pytest.approx(e.pk["atropine"].state.c2)
    e.set_rhythm("Asystole")
    advance_time(e, 10)
    assert e.state.hr == 0.0


def test_ephedrine_raises_pressure_and_preserves_output(engine_factory, advance_time):
    """Meng 2011 under TIVA, at peak MAP: MAP +24 to +28 mmHg, HR +2 to +8 bpm, CO preserved.

    The 30-min washout checks have no matched single-bolus data.
    """
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=4), start=True)
    advance_time(engine, 60)
    baseline = engine.state.map, engine.state.hr, engine.state.co
    engine.give_drug_bolus("ephedrine", 0.1 * engine.patient.weight)
    peak = baseline
    for _ in range(360):
        engine.step(0.5)
        if engine.state.map > peak[0]:
            peak = engine.state.map, engine.state.hr, engine.state.co
    assert 15 < peak[0] - baseline[0] < 40
    assert 0 < peak[1] - baseline[1] < 12
    assert peak[2] > baseline[2]
    advance_time(engine, 30 * 60)
    assert engine.state.map < peak[0] - 5
    assert abs(engine.state.hr - baseline[1]) < 8


def test_ephedrine_preserves_pressure_during_induction(engine_factory):
    """Sjøen 2026: ephedrine 0.1 mg/kg at propofol-remifentanil induction reduced
    the SBP fall by 15 mmHg (95% CI 9-22) and retained benefit at five minutes.

    Eleveld plasma targets approximate the study's Marsh/Minto protocol;
    effect-site targets would leave the patient unresponsive during sedation.
    """
    traces = []
    baselines = []
    for dose in (0.0, 0.1):
        e = engine_factory(
            config=SimulationConfig(mode="awake", tci_enabled=True, rng_seed=4),
            start=True, age=35, weight=73, sex="female", height=165,
        )
        e.set_airway_mode("ETT")
        e.set_vent_power(True)
        e.enable_tci("propofol", 1.0, mode="plasma")
        e.enable_tci("remi", 1.0, mode="plasma")
        sedation = []
        for _ in range(300):
            e.step(0.5)
            sedation.append((e.state.sbp, e.state.hr, e.state.co))
        # The paper averaged the interval 55-5 s before the induction bolus.
        baseline = np.mean(sedation[190:290], axis=0)
        assert e.state.loc < 0.5
        e.enable_tci("propofol", 6.0, mode="plasma")
        e.enable_tci("remi", 8.0, mode="plasma")
        samples = []
        reduced = False
        # At 1 mg/mL and 0.5 mL/s, 0.1 mg/kg takes 0.2 s/kg to inject.
        remaining = dose * e.patient.weight
        for _ in range(570):
            injection = min(remaining, 0.5 * 0.5)
            if injection > 0.0:
                e.give_drug_bolus("ephedrine", injection)
                remaining -= injection
            e.step(0.5)
            if not reduced and e.state.loc > 0.95:
                e.enable_tci("propofol", 2.0, mode="plasma")
                e.enable_tci("remi", 5.0, mode="plasma")
                reduced = True
            samples.append((e.state.sbp, e.state.hr, e.state.co))
        assert reduced
        assert remaining == pytest.approx(0.0)
        # Ten-second sliding means, five-second overlap, through 285 s.
        traces.append(np.array([
            np.mean(samples[i:i + 20], axis=0) for i in range(0, len(samples) - 19, 10)
        ]))
        baselines.append(baseline)
    control, treated = traces
    nadirs = [trace[:, 0].min() - baseline[0]
              for trace, baseline in zip(traces, baselines, strict=True)]
    assert 5.0 < nadirs[1] - nadirs[0] < 25.0
    # Gradual onset, and benefit that persists through five minutes.
    pressure_benefit = treated[:, 0] - control[:, 0]
    assert pressure_benefit[2] < 3.0  # Mean of the 10-20 s interval.
    assert 10.0 < pressure_benefit[-1] < 30.0
    assert pressure_benefit[-1] > 0.5 * pressure_benefit.max()
    assert treated[-1, 1] > control[-1, 1]
    assert treated[-1, 2] > control[-1, 2]


def test_ephedrine_repeated_boluses_have_a_smaller_pressor_response(engine_factory, advance_time):
    """Calibrated IV tolerance, with Persky 2004 tolerance timing, limits repeated doses."""
    final_pressures = []
    for tolerance in (False, True):
        e = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=4), start=True)
        if not tolerance:
            e.hemo.config = replace(e.hemo.config, ephedrine_tolerance_max=0.0)
        advance_time(e, 60)
        rises = []
        for _ in range(4):
            baseline = e.state.map
            e.give_drug_bolus("ephedrine", 5.0)
            peak = baseline
            for _ in range(300):
                e.step(1.0)
                peak = max(peak, e.state.map)
            rises.append(peak - baseline)
        final_pressures.append(e.state.map)
        if tolerance:
            assert rises[0] > 10.0
            assert rises[-1] < 0.2 * rises[0]
    assert final_pressures[0] - final_pressures[1] > 15.0


def test_succinylcholine_paralysis_and_sugammadex_independence(engine_factory, advance_time):
    """Kopman 2003 / Dhonneur 1999: profound short block without hypnosis.

    Phase I block depresses T1 without TOF fade, and sugammadex leaves it
    unchanged. Breathing returns before peripheral T1 recovers.
    """
    traces = []
    for reverse in (False, True):
        e = engine_factory(config=SimulationConfig(mode="awake", rng_seed=4), start=True)
        e.set_airway_mode("ETT")
        e.set_vent_power(True)
        e.give_drug_bolus("succinylcholine", e.patient.weight)
        advance_time(e, 90, dt=0.5)
        assert e.state.twitch < 5
        assert e.state.tof == 100.0
        assert require_tof_at_most(5)(e)[0]
        assert e.state.loc < 0.01 and e.state.tol < 0.01
        assert e.resp.state.effort_rr == 0.0
        if reverse:
            e.give_drug_bolus("sugammadex", 16 * e.patient.weight)
        advance_time(e, 510, dt=0.5)
        traces.append(e.state.twitch)
        assert e.resp.state.effort_rr > 0.0
        advance_time(e, 300, dt=0.5)
        assert e.state.twitch > 95.0
    assert traces[0] == pytest.approx(traces[1], abs=0.01)


@pytest.mark.parametrize("dose,nadir,onset,recovery", [
    (0.4, 6.8, 105, (4.4, 5.0, 6.6)),
    (0.6, 0.0, 81, (5.5, 6.1, 7.6)),
    (1.0, 0.0, 71, (6.6, 7.4, 9.3)),
])
def test_succinylcholine_dose_response_matches_kopman_calibration(
    engine_factory, dose, nadir, onset, recovery,
):
    """Kopman 2003 Table 2 means for T1 nadir, onset, and 25/50/90% recovery (calibration targets)."""
    e = engine_factory(config=SimulationConfig(mode="awake", rng_seed=4), start=True)
    e.set_airway_mode("ETT")
    e.set_vent_power(True)
    e.give_drug_bolus("succinylcholine", dose * e.patient.weight)
    trace = []
    for _ in range(1440):
        e.step(0.5)
        trace.append((e.state.time, e.state.twitch, e.resp.state.muscle_factor))
    peak_time, minimum, _ = min(trace, key=lambda point: point[1])
    assert minimum == pytest.approx(nadir, abs=8.8 if dose == 0.4 else 2.0)
    # The recorded plateau starts within 0.1 percentage point of the nadir.
    plateau = next(t for t, twitch, _ in trace if twitch <= minimum + 0.1)
    assert plateau == pytest.approx(onset, abs=25)
    for fraction, expected_min in zip((25, 50, 90), recovery, strict=True):
        recovered = next(t for t, twitch, _ in trace if t > peak_time and twitch >= fraction)
        assert recovered / 60.0 == pytest.approx(expected_min, abs=0.6)
    if dose == 1.0:
        central_recovery = next(t for t, _, strength in trace if t > peak_time and strength >= 0.9)
        assert central_recovery / 60.0 == pytest.approx(7.2, abs=1.0)  # Dhonneur 1999
