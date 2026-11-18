import math

import numpy as np
import pytest

from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.patient.pk_models import AmiodaronePK, LidocainePK, NorepinephrinePK
from anasim.physiology.arrhythmia import ArrhythmiaModel
from anasim.physiology.hemodynamics import HemodynamicModel
from anasim.physiology.resuscitation import ShockResult, shock_outcome


def _step(model, dt, rhythm, **changes):
    inputs = dict(ischemia=0.0, hypoxia=0.0, energy_failure=0.0,
                  electrically_silent=False, perfusing=True, lidocaine=0.0, amiodarone=0.0)
    inputs.update(changes)
    return model.step(dt, rhythm, **inputs)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_severe_ischemia_can_initiate_a_ventricular_rhythm_independent_of_step_size(dt):
    model = ArrhythmiaModel(np.random.default_rng(1))
    time, rhythm = 0.0, RhythmType.SINUS
    while rhythm == RhythmType.SINUS and time < 1000:
        rhythm = _step(model, dt, rhythm, ischemia=1)
        time += dt
    assert rhythm in (RhythmType.VTACH, RhythmType.VFIB)
    assert 643 <= time <= 644


def test_ischemic_vt_can_degenerate_to_vf_but_silent_myocardium_does_not():
    for silent in (False, True):
        model = ArrhythmiaModel(np.random.default_rng(1))
        rhythm = RhythmType.VTACH
        for _ in range(100):
            rhythm = _step(model, 1, rhythm, ischemia=1, electrically_silent=silent,
                           energy_failure=1 if silent else 0)
        assert rhythm == (RhythmType.VTACH if silent else RhythmType.VFIB)

    outcomes = []
    for perfusing in (True, False):
        rng = np.random.default_rng(7)
        converted = sum(
            shock_outcome(
                RhythmType.VTACH, synchronized=False, energy_failure=0.0,
                electrically_silent=False, vulnerable=False, antiarrhythmic=0.0,
                rng=rng, perfusing=perfusing,
            ) == RhythmType.SINUS
            for _ in range(200)
        )
        outcomes.append(converted)
    assert outcomes[0] == outcomes[1] and outcomes[0] > 160


def _vt_conversion_times(concs: list[float], drug: str, dt: float = 5.0, n: int = 400) -> list[float]:
    rng = np.random.default_rng(1)
    times: list[float] = []
    for _ in range(n):
        model = ArrhythmiaModel(rng)
        rhythm = RhythmType.VTACH
        for i, conc in enumerate(concs):
            rhythm = _step(model, dt, rhythm, **{drug: conc})
            if rhythm == RhythmType.SINUS:
                times.append((i + 1) * dt)
                break
    return times


def _effect_site_trace(pk, rate, site: str, duration_s: int, every_s: int = 5) -> list[float]:
    trace: list[float] = []
    for s in range(1, duration_s + 1):
        pk.step(1.0, rate(s))
        if s % every_s == 0:
            trace.append(getattr(pk.state, site))
    return trace


def test_vt_conversion_matches_somberg_and_scheinman_protocols(patient_factory):
    """Somberg 2002: immediate VT termination was 78% on amiodarone vs 27% on
    lidocaine (67% vs 9% alive and VT-free at 1 h). Scheinman 1995: median
    conversion was 0.85 h (51 min) at 1000 mg/24 h vs 0.94 h at 125 mg/24 h
    (1.42 vs 3.11 h pooled with Kowey, P = 0.15); AnaSim separates the two doses.
    """
    def two_boluses(mg: float, maintenance_mg_min: float):
        return lambda s: maintenance_mg_min / 60.0 + (mg / 120.0 if s <= 120 or 600 < s <= 720 else 0.0)

    def load_then_infuse(load_mg: float, maintenance_mg_min: float):
        return lambda s: load_mg / 600.0 if s <= 600 else maintenance_mg_min / 60.0

    somberg_amio = _effect_site_trace(
        AmiodaronePK(patient_factory(age=68, weight=74, height=172)), two_boluses(150.0, 0.417), "ce", 3600,
    )
    somberg_lido = _effect_site_trace(
        LidocainePK(patient_factory(age=64, weight=85, height=175)), two_boluses(100.0, 2.0), "ce_cv", 3600,
    )
    assert 0.55 <= len(_vt_conversion_times(somberg_amio, "amiodarone")) / 400 <= 0.78
    assert 0.15 <= len(_vt_conversion_times(somberg_lido, "lidocaine")) / 400 <= 0.32

    scheinman = AmiodaronePK(patient_factory(age=65, weight=78, height=174))
    high_trace = _effect_site_trace(scheinman, load_then_infuse(150.0, 1.0), "ce", 7200, every_s=30)
    high_times = _vt_conversion_times(high_trace, "amiodarone", dt=30.0)
    high_censored = sorted(high_times + [7200.0] * (400 - len(high_times)))
    assert 35.0 <= float(np.median(high_censored)) / 60.0 <= 60.0

    scheinman = AmiodaronePK(patient_factory(age=65, weight=78, height=174))
    low_trace = _effect_site_trace(scheinman, load_then_infuse(18.75, 0.125), "ce", 14400, every_s=30)
    low_times = _vt_conversion_times(low_trace, "amiodarone", dt=30.0)
    low_censored = sorted(low_times + [14400.0] * (400 - len(low_times)))
    assert 1.1 <= float(np.median(low_censored)) / 3600.0 <= 2.8


@pytest.mark.parametrize("rhythm", [RhythmType.VFIB, RhythmType.ASYSTOLE, RhythmType.VTACH])
def test_drugs_do_not_replace_a_shock_for_pulseless_ventricular_rhythms(rhythm):
    model = ArrhythmiaModel(np.random.default_rng(1))
    for _ in range(600):
        assert _step(model, 1, rhythm, perfusing=False, lidocaine=30, amiodarone=30) == rhythm


def test_successful_shock_can_leave_transient_bradycardia_and_recurrent_vf():
    model = ArrhythmiaModel(np.random.default_rng(1))
    rhythm = model.after_shock(RhythmType.VFIB, RhythmType.SINUS,
                              ischemia=1, hypoxia=0, energy_failure=0.3)
    assert rhythm == RhythmType.SINUS_BRADY
    for _ in range(120):
        rhythm = _step(model, 1, rhythm, ischemia=1, energy_failure=0.3)
        if rhythm == RhythmType.VFIB:
            break
    assert rhythm == RhythmType.VFIB


def test_post_shock_bradycardia_recovers_but_manually_selected_bradycardia_persists():
    model = ArrhythmiaModel(np.random.default_rng(1))
    rhythm = model.after_shock(RhythmType.VFIB, RhythmType.SINUS,
                              ischemia=1, hypoxia=0, energy_failure=0)
    for _ in range(40):
        rhythm = _step(model, 1, rhythm)
    assert rhythm == RhythmType.SINUS
    model.reset()
    for _ in range(60):
        assert _step(model, 1, RhythmType.SINUS_BRADY) == RhythmType.SINUS_BRADY


def test_treating_intervening_vt_does_not_erase_recent_vf_substrate():
    model = ArrhythmiaModel(np.random.default_rng(2))
    rhythm = model.after_shock(RhythmType.VFIB, RhythmType.SINUS,
                              ischemia=1, hypoxia=0, energy_failure=0)
    for _ in range(10):
        rhythm = _step(model, 1, rhythm, ischemia=0.5)
    rhythm = model.after_shock(RhythmType.VTACH, RhythmType.SINUS,
                              ischemia=0.5, hypoxia=0, energy_failure=0)
    for _ in range(120):
        rhythm = _step(model, 1, rhythm, ischemia=0.5)
        if rhythm == RhythmType.VFIB:
            break
    assert rhythm == RhythmType.VFIB


def test_antiarrhythmic_does_not_guarantee_freedom_from_rearrest():
    times = []
    for concentration in (0, 10):
        model = ArrhythmiaModel(np.random.default_rng(2))
        rhythm = model.after_shock(RhythmType.VFIB, RhythmType.SINUS,
                                  ischemia=1, hypoxia=0, energy_failure=0)
        for second in range(120):
            rhythm = _step(model, 1, rhythm, ischemia=0.5, amiodarone=concentration)
            if rhythm == RhythmType.VFIB:
                times.append(second)
                break
    # ALPS did not establish a reduction in rearrest with either drug.
    assert len(times) == 2 and times[0] == times[1]


def test_adding_vt_conversion_cannot_advance_the_degeneration_clock():
    times = []
    for concentration in (0, 10):
        model = ArrhythmiaModel(np.random.default_rng(2))
        rhythm = RhythmType.VTACH
        for second in range(100):
            rhythm = _step(model, 1, rhythm, ischemia=1, amiodarone=concentration)
            if rhythm != RhythmType.VTACH:
                times.append((second, rhythm))
                break
    assert len(times) == 2 and times[0] == times[1]


def test_antiarrhythmic_facilitates_defibrillation_but_does_not_make_it_certain():
    outcomes = []
    for effect in (0, 1):
        outcomes.append(shock_outcome(
            RhythmType.VFIB, synchronized=False, energy_failure=0,
            electrically_silent=False, vulnerable=False, antiarrhythmic=effect,
            rng=np.random.default_rng(4),
        ))
    assert outcomes == [RhythmType.VFIB, RhythmType.SINUS]
    assert shock_outcome(
        RhythmType.VFIB, synchronized=False, energy_failure=1,
        electrically_silent=False, vulnerable=False, antiarrhythmic=1,
        rng=np.random.default_rng(4),
    ) == RhythmType.VFIB


@pytest.mark.parametrize(("model_type", "site"), [(AmiodaronePK, "ce"), (LidocainePK, "ce_cv")])
def test_cardiac_drug_exposure_waits_for_flow_then_increases_with_cpr(patient, model_type, site):
    model = model_type(patient)
    model.state.c1 = 100 / model.v1
    initial = model.state.c1
    model.circulation = 0
    for _ in range(120):
        model.step(1, 0)
    assert model.state.c1 == initial and getattr(model.state, site) == 0
    model.circulation = 2 / 3
    for _ in range(120):
        model.step(1, 0)
    assert getattr(model.state, site) > 0.5 and model.state.c1 < initial


def test_post_shock_vf_recurrence_incidence_and_timing():
    """Koster 2008 and van Alem 2003: about half of shockable arrests refibrillate
    within 2 min of a shock, with biphasic recurrence median 36 s (IQR 17-77 s).
    """
    dt = 2.0
    rng = np.random.default_rng(1)
    recurrences: list[float] = []
    steps = round(180.0 / dt)
    for _ in range(400):
        model = ArrhythmiaModel(rng)
        rhythm = model.after_shock(
            RhythmType.VFIB, RhythmType.SINUS, ischemia=1.0, hypoxia=0.0, energy_failure=0.1,
        )
        for i in range(1, steps + 1):
            t = i * dt
            rhythm = _step(model, dt, rhythm, ischemia=math.exp(-t / 60.0), energy_failure=0.1)
            if rhythm == RhythmType.VFIB:
                recurrences.append(t)
                break

    assert 0.45 <= sum(t <= 120.0 for t in recurrences) / 400 <= 0.60
    q25, q50, q75 = np.percentile(recurrences, [25, 50, 75])
    assert 10.0 <= q25 <= 20.0
    assert 25.0 <= q50 <= 42.0
    assert 55.0 <= q75 <= 85.0


def test_post_shock_asystole_incidence_duration_and_competing_refibrillation():
    """Stiell 2007 and Koster 2008: first 200 J biphasic shocks terminate ~90%
    of VF. Pierce 2015: post-shock asystole occurs in about half of OHCA shocks
    (27% in hospital-discharge survivors), lasting a median 20 s (IQR 36 s),
    with time to organized rhythm median 7 s (IQR 26 s) and VF recurring from
    both post-shock asystole and organized rhythms. Energy-dependent incidence
    and the absence of post-shock asystole after brief VF are calibrations.
    """
    dt = 2.0
    brief_rng = np.random.default_rng(11)
    brief = [
        shock_outcome(
            RhythmType.VFIB, synchronized=False, energy_failure=0.0,
            electrically_silent=False, vulnerable=False, antiarrhythmic=0.0,
            rng=brief_rng, perfusing=False, energy_reserve=0.90,
        )
        for _ in range(400)
    ]
    assert RhythmType.ASYSTOLE not in brief
    assert 0.85 <= brief.count(RhythmType.SINUS) / 400 <= 0.94

    survivor_rng = np.random.default_rng(12)
    survivor = [
        shock_outcome(
            RhythmType.VFIB, synchronized=False, energy_failure=0.0,
            electrically_silent=False, vulnerable=False, antiarrhythmic=0.0,
            rng=survivor_rng, perfusing=False, energy_reserve=0.40,
        )
        for _ in range(400)
    ]
    assert 0.18 <= survivor.count(RhythmType.ASYSTOLE) / 400 <= 0.36

    rng = np.random.default_rng(19)
    shock_rng = np.random.default_rng(23)
    steps = round(240.0 / dt)
    asystole_shocks = 0
    asystole_durations: list[float] = []
    time_to_organized: list[float] = []
    refib_from_asystole = refib_from_organized = 0
    for _ in range(500):
        ef0 = float(shock_rng.uniform(0.0, 0.55))
        er0 = max(0.0, 1.0 - float(shock_rng.uniform(0.75, 1.4)))
        outcome = shock_outcome(
            RhythmType.VFIB, synchronized=False, energy_failure=ef0,
            electrically_silent=False, vulnerable=False, antiarrhythmic=0.0,
            rng=shock_rng, perfusing=False, energy_reserve=er0,
        )
        if outcome == RhythmType.VFIB:
            continue
        model = ArrhythmiaModel(rng)
        rhythm = model.after_shock(
            RhythmType.VFIB, outcome, ischemia=1.0, hypoxia=0.0,
            energy_failure=ef0, energy_reserve=er0,
        )
        started_asystole = rhythm == RhythmType.ASYSTOLE
        if started_asystole:
            asystole_shocks += 1
        else:
            time_to_organized.append(3.5)
        left_asystole = not started_asystole
        for i in range(1, steps + 1):
            t = i * dt
            ef = ef0 * math.exp(-t / 60.0)
            isch = max(0.2, math.exp(-t / 120.0))
            prev = rhythm
            rhythm = _step(
                model, dt, rhythm, ischemia=isch, energy_failure=ef,
                perfusing=False, energy_reserve=er0, compressions=True,
            )
            if not left_asystole and rhythm != RhythmType.ASYSTOLE:
                left_asystole = True
                asystole_durations.append(t)
                if rhythm != RhythmType.VFIB:
                    time_to_organized.append(3.5 + t)
            if rhythm == RhythmType.VFIB:
                if prev == RhythmType.ASYSTOLE:
                    refib_from_asystole += 1
                else:
                    refib_from_organized += 1
                break

    assert 0.45 <= asystole_shocks / 500 <= 0.62
    aq25, aq50, aq75 = np.percentile(asystole_durations, [25, 50, 75])
    oq25, oq50, oq75 = np.percentile(time_to_organized, [25, 50, 75])
    assert 15.0 <= aq50 <= 26.0 and 22.0 <= aq75 - aq25 <= 40.0
    assert 3.5 <= oq50 <= 11.0 and 16.0 <= oq75 - oq25 <= 32.0
    assert refib_from_asystole >= 40 and refib_from_organized >= 40


def test_acute_amiodarone_hemodynamics_follow_rate_and_lv_function(patient_factory):
    """Munoz 1988 and Kosinski 1984: rapid IV boluses briefly lower pressure
    with reflex HR support and maintained output, HR and output fall modestly by
    15-30 min in preserved LV function (Munoz 1988; acute CI rose 12.5% in
    Kosinski's 4 normal-EF patients), while HFrEF loses ~19% cardiac output (Kosinski 1984).
    """
    def run_infusion(hf: str, dose_mg: float, duration_s: int, total_s: int) -> dict[int, tuple[float, float, float]]:
        p = patient_factory(age=60, weight=75, height=175, heart_failure=hf)
        hemo = HemodynamicModel(p)
        nore = NorepinephrinePK(p)
        nore.equilibrate_endogenous()
        hemo.initialize_steady_state(0.0, 0.0, nore.state.ce)
        pk = AmiodaronePK(p)
        base_bv = hemo.blood_volume_0
        base_co = hemo.base_co_l_min / hemo.failure.output_factor
        hr0, map0, co0 = hemo.state.hr, hemo.state.map, hemo.state.co
        rate = dose_mg / duration_s
        snapshots: dict[int, tuple[float, float, float]] = {}
        for s in range(2, total_s + 1, 2):
            pk.update_hemodynamics(
                hemo.blood_volume / base_bv, hemo.state.co / base_co,
                hemo.renal_perfusion, blood_volume_l=base_bv / 1000.0,
            )
            pk.step(2.0, rate if s <= duration_s else 0.0)
            nore.step(2.0, 0.0)
            st = hemo.step(
                2.0, 0.0, 0.0, nore.state.ce, -2.0, 40.0, 95.0,
                ce_amiodarone=pk.state.ce, ce_amiodarone_vasc=pk.state.ce_cv,
                ce_amiodarone_hr=pk.state.ce_hr,
            )
            if s in (180, 300, 360, 600, 1800):
                snapshots[s] = (1.0 - st.hr / hr0, 1.0 - st.map / map0, 1.0 - st.co / co0)
        return snapshots

    slow_normal = run_infusion("none", 150.0, 600, 600)
    munoz_normal = run_infusion("none", 375.0, 180, 1800)
    kosinski_normal = run_infusion("none", 300.0, 300, 600)
    kosinski_hfref = run_infusion("hfref", 300.0, 300, 600)

    assert slow_normal[600][1] <= 0.06

    assert munoz_normal[180][0] < 0.0
    assert 0.12 <= munoz_normal[180][1] <= 0.22
    assert -0.06 <= munoz_normal[180][2] <= 0.03
    assert munoz_normal[360][1] <= 0.06
    assert 0.04 <= munoz_normal[1800][0] <= 0.09
    assert munoz_normal[1800][1] <= 0.06
    assert 0.04 <= munoz_normal[1800][2] <= 0.09

    assert 0.08 <= kosinski_normal[300][1] <= 0.14
    assert -0.04 <= kosinski_normal[300][2] <= 0.03
    assert kosinski_normal[600][1] <= 0.06
    assert 0.03 <= kosinski_normal[600][2] <= 0.08

    assert 0.13 <= kosinski_hfref[600][2] <= 0.22
    assert 0.05 <= kosinski_hfref[600][1] <= 0.15


def test_engine_projects_automatic_rhythm_changes_to_ecg_and_circulation(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    engine.hemo.coronary.ischemia = 1
    engine.arrhythmias._rhythm = RhythmType.SINUS
    engine.arrhythmias._remaining["onset"] = 0
    engine.step(0.01)
    assert engine.hemo.rhythm_type in (RhythmType.VFIB, RhythmType.VTACH)
    assert engine.cardiac_cycle.current.rhythm_type == engine.hemo.rhythm_type


def test_antiarrhythmic_bolus_in_vf_needs_circulation_before_it_can_help_a_shock(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    engine.set_rhythm("VFIB")
    engine.give_drug_bolus("amiodarone", 300)
    engine.charge_defibrillator()
    for _ in range(10):
        engine.step(1)
    assert engine.state.amiodarone_ce == 0
    assert engine.hemo.rhythm_type == RhythmType.VFIB
    engine.set_chest_compressions(True)
    for _ in range(60):
        engine.step(1)
    assert engine.state.amiodarone_ce > 1
    engine.charge_defibrillator()
    for _ in range(5):
        engine.step(1)
    assert engine.defibrillate() == ShockResult.DELIVERED
    assert math.isfinite(engine.state.amiodarone_ce)


def test_sinus_bradycardia_chronotropy_extreme_tachycardia_and_afib_rvr_control(engine_factory, advance_time):
    """SINUS_BRADY responds to epinephrine; extreme tachycardia reduces diastolic filling
    and CO; esmolol rate-controls AFib with RVR while raising SV and CO (Clark 1997, Hardman 1998).
    """
    brady = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    assert brady.hemo._compute_state(hr_base=210.0).co < brady.hemo._compute_state(hr_base=120.0).co
    brady.hemo.rhythm_type = RhythmType.SINUS_BRADY
    brady.step(1.0)
    brady_hr = brady.hemo.state.hr
    assert brady_hr < 52.0
    brady.give_drug_bolus("epi", 50.0)
    advance_time(brady, 30.0, dt=0.5)
    assert brady.hemo.state.hr > brady_hr + 12.0

    afib = engine_factory(
        config=SimulationConfig(mode="steady_state", disturbance_profile="stim_sustained_surgery", rng_seed=1),
        start=True,
    )
    afib.set_drug_rate("remi", 0.0)
    afib.give_drug_bolus("epi", 25.0)
    afib.set_drug_rate("epi", 8.0)
    advance_time(afib, 120.0, dt=1.0)
    afib.set_rhythm("AFIB")
    advance_time(afib, 60.0, dt=1.0)
    rvr_hr, rvr_sv, rvr_co = afib.state.hr, afib.state.sv, afib.state.co
    assert rvr_hr > 135.0
    afib.give_drug_bolus("esmolol", 70.0)
    afib.set_drug_rate("esmolol", 180.0)
    advance_time(afib, 90.0, dt=1.0)
    assert afib.state.hr < rvr_hr - 25.0
    assert afib.state.sv > rvr_sv * 1.25 and afib.state.co > rvr_co * 1.05
