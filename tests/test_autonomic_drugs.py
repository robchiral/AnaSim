"""Esmolol, labetalol, and glycopyrrolate through the engine and circulation; docstrings cite each bound."""

import copy

import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.arterial import ArterialWaveformRenderer
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.patient.patient import Patient
from anasim.patient.pk_models import EsmololPK, LabetalolPK
from anasim.physiology.hemodynamics import HemodynamicModel


def _advance(engine, seconds, dt=1.0):
    for _ in range(round(seconds / dt)):
        engine.step(dt)


def _extreme_hr(engine, seconds, pick, dt=0.5):
    values = []
    for _ in range(round(seconds / dt)):
        engine.step(dt)
        values.append(engine.state.hr)
    return pick(values)


def _tiva(engine_factory):
    return engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=3), start=True
    )


def test_labetalol_lowers_heart_rate_longer_than_blood_pressure():
    """Abernethy 1987, 50 mg over 10 min in young hypertensive adults (32 y, 95 kg):
    HR fell up to 17 bpm and SBP up to 25 mmHg; BP recovered within 30 min, and
    HR stayed low for 3 h. In AnaSim a smaller BP fall lasts as long as the beta
    block, because in sinus rhythm the baroreflex adjusts HR only.
    """
    patient = Patient(age=32, weight=95, height=180)
    hemo = HemodynamicModel(patient)
    pk = LabetalolPK(patient)
    cycle = CardiacCycle(np.random.default_rng(3))
    renderer = ArterialWaveformRenderer(age=32)
    base = hemo.step(0.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
    bp0 = renderer.step(
        cycle.seed(base.hr, RhythmType.SINUS), base.map, base.sv,
        static_pressure=base.venous_pressure, pulse_pressure_factor=base.pulse_pressure_factor,
    )
    base_hr, base_sbp = base.hr, bp0.systolic

    base_bv = hemo.blood_volume_0
    base_co = hemo.base_co_l_min / hemo.failure.output_factor
    state = base
    changes = []
    for minute in range(180):
        if minute < 10:
            pk.state.c1 += 5000.0 / pk.v1
        for _ in range(12):
            pk.update_hemodynamics(hemo.blood_volume / base_bv, state.co / base_co, hemo.renal_perfusion)
            ce = pk.step(5.0, 0.0).ce
            state = hemo.step(5.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_labetalol=ce)
        bp = renderer.step(
            cycle.step(1.0, state.hr, state.rhythm_type), state.map, state.sv,
            static_pressure=state.venous_pressure, pulse_pressure_factor=state.pulse_pressure_factor,
        )
        changes.append((state.hr - base_hr, bp.systolic - base_sbp))

    max_hr_fall = min(hr for hr, _ in changes)
    max_sbp_fall = min(sbp for _, sbp in changes)
    assert -20.0 < max_hr_fall < -8.0
    assert -35.0 < max_sbp_fall < -15.0
    assert changes[59][1] > 0.5 * max_sbp_fall
    # The paper reports a persistent decrease at 3 h, without a fixed peak ratio.
    assert changes[179][0] < -3.0


def test_esmolol_onset_offset_and_atrial_fibrillation_rate(engine_factory):
    """Wiest 2012: a 500 mcg/kg load gives 90% of beta block by 5 min, recovery
    takes 18-30 min after stopping, and esmolol controls the AF ventricular rate.
    Sum 1983: steady-state blood level 0.569 mcg/mL at 150 mcg/kg/min.
    """
    patient = Patient()
    hemo = HemodynamicModel(patient)
    pk = EsmololPK(patient)
    base_hr = hemo.step(0.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0).hr

    def run_esmolol(seconds: int, mcg_kg_min: float, dt: float = 1.0):
        rate_mg_s = mcg_kg_min * patient.weight / 60000.0
        for _ in range(round(seconds / dt)):
            ce = pk.step(dt, rate_mg_s).ce
            state = hemo.step(dt, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_esmolol=ce)
        return state

    run_esmolol(60, 500.0)
    state_5m = run_esmolol(240, 150.0)
    fall_at_5_min = state_5m.hr - base_hr
    steady = run_esmolol(1500, 150.0, dt=5.0)
    steady_fall = steady.hr - base_hr
    assert steady_fall < -6.0
    assert fall_at_5_min < 0.8 * steady_fall
    assert pk.state.c1 == pytest.approx(0.569, rel=0.1)

    state_18m = run_esmolol(18 * 60, 0.0, dt=5.0)
    assert state_18m.hr - base_hr > 0.35 * steady_fall
    state_30m = run_esmolol(12 * 60, 0.0, dt=5.0)
    assert state_30m.hr - base_hr > -1.5

    engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=3), start=True)
    engine.set_rhythm("AFIB")
    _advance(engine, 120, dt=2.0)
    assert engine.state.hr == pytest.approx(110.0, abs=2.0)
    engine.set_drug_rate("esmolol", 500.0)
    _advance(engine, 60, dt=2.0)
    engine.set_drug_rate("esmolol", 150.0)
    _advance(engine, 240, dt=2.0)
    assert engine.state.hr < 95.0


def test_esmolol_reduces_intubation_and_epinephrine_tachycardia(engine_factory):
    """Wiest 2012: esmolol 1-2 mg/kg attenuates the HR response to laryngoscopy.
    Competitive beta-1 block shifts the epinephrine chronotropy curve right.
    """
    base_engine = _tiva(engine_factory)
    # Let remifentanil wash out once, then branch for control vs esmolol.
    base_engine.set_drug_rate("remi", 0.0)
    _advance(base_engine, 900, dt=2.0)

    responses = {}
    for esmolol_mg in (0.0, 70.0):
        engine = copy.deepcopy(base_engine) if esmolol_mg == 0.0 else base_engine
        engine.give_drug_bolus("esmolol", esmolol_mg)
        _advance(engine, 120, dt=2.0)
        base_hr = engine.state.hr
        engine.start_disturbance("stim_intubation_pulse")
        stimulation = _extreme_hr(engine, 60, max, dt=0.5) - base_hr
        _advance(engine, 60, dt=2.0)
        base_hr = engine.state.hr
        engine.give_drug_bolus("epi", 10.0)
        responses[esmolol_mg] = stimulation, _extreme_hr(engine, 120, max, dt=0.5) - base_hr

    (stim_off, epi_off), (stim_on, epi_on) = responses[0.0], responses[70.0]
    assert stim_off > 3.0
    assert stim_on < 0.6 * stim_off
    assert epi_on < 0.6 * epi_off


def test_glycopyrrolate_raises_rate_and_blocks_vagal_bradycardia(engine_factory):
    """Ali-Melkkilä 1993: 4 mcg/kg after induction gave 11.8 ng/mL about 5 min
    later and prevented reflex bradycardia during eye-muscle traction.
    """
    engine = _tiva(engine_factory)
    base_hr = engine.state.hr
    engine.give_drug_bolus("phenyl", 100.0)
    reflex_before = _extreme_hr(engine, 180, min) - base_hr
    _advance(engine, 300)

    base_hr = engine.state.hr
    engine.give_drug_bolus("glyco", 0.004 * engine.patient.weight)
    _advance(engine, 60)
    assert engine.state.hr - base_hr > 8.0
    _advance(engine, 240)
    assert engine.pk["glyco"].state.c1 == pytest.approx(11.8, rel=0.25)
    # Poterman 2017: under remifentanil-propofol, methylatropine HR rose from
    # mean 80 +/- 14 bpm to a peak median 102 bpm (IQR 86-116).
    assert engine.state.hr - base_hr < 36.0

    base_hr = engine.state.hr
    engine.give_drug_bolus("phenyl", 100.0)
    reflex_after = _extreme_hr(engine, 180, min) - base_hr
    assert reflex_before < -6.0
    assert reflex_after > 0.5 * reflex_before

    engine.set_rhythm("SINUS_BRADY")
    _advance(engine, 120)
    assert engine.state.hr > 60.0

    engine.set_rhythm("SINUS")
    _advance(engine, 30)
    hr_after_glyco = engine.state.hr
    engine.give_drug_bolus("atropine", 0.5)
    _advance(engine, 60)
    assert engine.state.hr >= hr_after_glyco - 0.5


def test_ephedrine_competitive_blockade_and_acidosis_blunting(patient):
    low_eph, high_eph = HemodynamicModel(patient), HemodynamicModel(patient)
    s_low = low_eph.step(0.5, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_labetalol=400.0, ce_ephedrine=300.0)
    s_high = high_eph.step(0.5, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_labetalol=400.0, ce_ephedrine=10000.0)
    assert s_high.svr > s_low.svr and s_high.sv > s_low.sv
    for _ in range(20):
        s_low = low_eph.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_labetalol=400.0, ce_ephedrine=300.0)
        s_high = high_eph.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_labetalol=400.0, ce_ephedrine=10000.0)
    assert s_high.hr > s_low.hr + 5.0 and s_high.hr < 150.0

    norm_p, acid_p = HemodynamicModel(patient), HemodynamicModel(patient)
    b_norm = norm_p.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ph=7.40)
    b_acid = acid_p.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ph=6.90)
    for _ in range(15):
        r_norm = norm_p.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_ephedrine=400.0, ph=7.40)
        r_acid = acid_p.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_ephedrine=400.0, ph=6.90)
    assert (r_acid.map - b_acid.map) < 0.55 * (r_norm.map - b_norm.map)

    # Tolerance builds from raw exposure independently of extracellular pH.
    tol_norm, tol_acid = HemodynamicModel(patient), HemodynamicModel(patient)
    for _ in range(180):
        tol_norm.step(5.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_ephedrine=500.0, ph=7.40)
        tol_acid.step(5.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_ephedrine=500.0, ph=6.85)
    assert tol_acid._ephedrine_tolerance_ce == pytest.approx(tol_norm._ephedrine_tolerance_ce, rel=1e-9)


def test_atropine_exceeds_glycopyrrolate_under_propofol_and_ephedrine_repeats_at_one_hour(
    engine_factory, advance_time
):
    """Under propofol, atropine raises HR more than glycopyrrolate; ephedrine retains >55% effect at 1 h."""
    tiva_a = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    tiva_g = copy.deepcopy(tiva_a)
    tiva_a.give_drug_bolus("atropine", 1.0)
    tiva_g.give_drug_bolus("glyco", 0.4)
    advance_time(tiva_a, 300.0, dt=1.0)
    advance_time(tiva_g, 300.0, dt=1.0)
    assert tiva_a.state.hr > tiva_g.state.hr

    eph = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=4), start=True)
    advance_time(eph, 30.0, dt=1.0)
    m0 = eph.state.map
    eph.give_drug_bolus("ephedrine", 10.0)
    peak1 = m0
    for _ in range(300):
        eph.step(1.0)
        peak1 = max(peak1, eph.state.map)
    advance_time(eph, 3300.0, dt=2.0)
    m1 = eph.state.map
    eph.give_drug_bolus("ephedrine", 10.0)
    peak2 = m1
    for _ in range(300):
        eph.step(1.0)
        peak2 = max(peak2, eph.state.map)
    assert (peak2 - m1) > 0.55 * (peak1 - m0)


def test_awake_hfref_milrinone_unloads_svr_and_raises_co(engine_factory, advance_time):
    """Label milrinone dosing (50 mcg/kg over 10 min + 0.5 mcg/kg/min; DailyMed 2026b) in awake HFrEF raises CO and lowers SVR (Baim 1983)."""
    engine = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1), start=True,
        age=65, weight=75, height=172, sex="male", heart_failure="hfref",
    )
    base_co = engine.state.co
    base_svr = engine.state.svr
    base_map = engine.state.map

    # milri rate unit is mcg/kg/min: 5.0 mcg/kg/min for 10 min = 50 mcg/kg load.
    engine.set_drug_rate("milri", 5.0)
    advance_time(engine, 600.0, dt=2.0)
    engine.set_drug_rate("milri", 0.5)
    advance_time(engine, 600.0, dt=2.0)

    assert engine.state.co > base_co * 1.25
    assert engine.state.svr < base_svr * 0.80
    assert abs(engine.state.map - base_map) < 10.0

    # By contrast, a rapid 50 mcg/kg milrinone bolus in a normal anesthetized heart lowers MAP.
    normal = engine_factory(
        config=SimulationConfig(mode="steady_state", rng_seed=1), start=True,
        age=65, weight=75, height=172, sex="male",
    )
    norm_map0 = normal.state.map
    normal.give_drug_bolus("milri", 50.0 * 75.0)
    advance_time(normal, 300.0, dt=2.0)
    assert normal.state.map < norm_map0 - 8.0


def test_atropine_chronotropic_response_declines_with_age(engine_factory, advance_time):
    """Older adults have smaller peak HR rises after atropine (Jose 1970; Dauchot 1971)."""
    young = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, age=25)
    elderly = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, age=75)
    for eng in (young, elderly):
        eng.give_drug_bolus("atropine", 1.0)
        advance_time(eng, 120.0, dt=1.0)
    young_rise = young.state.hr - young.patient.baseline_hr
    assert young_rise > 25.0
    assert (elderly.state.hr - elderly.patient.baseline_hr) < 0.78 * young_rise


def test_beta_blockers_do_not_spike_map_under_induction_drugs(engine_factory, advance_time):
    """Esmolol and labetalol do not paradoxically raise MAP during propofol or etomidate induction."""
    for hypnotic, h_dose, blockers in (
        ("propofol", 140.0, (("labetalol", 20.0),)),
        ("etomidate", 20.0, (("esmolol", 50.0), ("labetalol", 20.0))),
    ):
        ctrl = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
        ctrl.set_airway_mode("ETT")
        ctrl.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        ctrl.set_vent_power(True)
        ctrl.give_drug_bolus(hypnotic, h_dose)
        advance_time(ctrl, 60.0, dt=2.0)

        treated = []
        for blocker, b_dose in blockers:
            arm = copy.deepcopy(ctrl)
            arm.give_drug_bolus(blocker, b_dose)
            treated.append(arm)

        max_excess = [-1e9] * len(treated)
        for _ in range(90):
            ctrl.step(2.0)
            for idx, arm in enumerate(treated):
                arm.step(2.0)
                max_excess[idx] = max(max_excess[idx], arm.state.map - ctrl.state.map)
        assert all(excess <= 0.5 for excess in max_excess)
