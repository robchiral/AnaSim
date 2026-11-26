"""Fentanyl, midazolam, etomidate, ketamine, and lidocaine through the engine; docstrings cite each bound."""

import copy

import pytest

from anasim.core.projection import PROPOFOL_LOC_TRANSFER, hypnotic_equivalent
from anasim.core.state import SimulationConfig
from anasim.core.tci import TCIController
from anasim.patient.patient import Patient
from anasim.patient.pd.anesthesia import (
    PROPOFOL_LOSS_OF_RESPONSE,
    ClinicalResponseModel,
    midazolam_loss_of_response,
)
from anasim.patient.pk_models import (
    ANESTHETIZED_OUTPUT,
    EsmololPK,
    EtomidatePK,
    FentanylPK,
    KetaminePK,
    LabetalolPK,
    MidazolamPK,
    PropofolPKEleveld,
)
from anasim.physiology.hemodynamics import HemodynamicModel


def _advance(engine, seconds, dt=1.0):
    for _ in range(round(seconds / dt)):
        engine.step(dt)


def _awake(engine_factory, **patient):
    engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=3), start=True, **patient)
    engine.set_airway_mode("Mask")
    engine.set_fgf(8.0, 0.0)
    _advance(engine, 30)
    return engine


def _measure_intubation_peak(engine):
    base_hr, base_map, tolerance = engine.state.hr, engine.state.map, engine.state.tol
    engine.start_disturbance("stim_intubation_pulse")
    peak_hr, peak_map = base_hr, base_map
    for _ in range(120):
        engine.step(0.5)
        peak_hr = max(peak_hr, engine.state.hr)
        peak_map = max(peak_map, engine.state.map)
    return peak_hr - base_hr, peak_map - base_map, tolerance


def test_ketamine_induction_keeps_breathing_and_raises_pressure(engine_factory):
    """Tweed 1972: without atropine, 2 mg/kg raised HR 33%, cardiac index 29%,
    and aortic pressure 28%, with SVR unchanged. Idvall 1979: patients woke at
    0.64 mcg/mL. Bourke 1987: 3 mg/kg shifted the CO2 response 2 mmHg. In
    AnaSim, ketamine preserves pharyngeal tone and BIS.
    """
    engine = _awake(engine_factory)
    base_map, base_hr, base_co, base_svr = engine.state.map, engine.state.hr, engine.state.co, engine.state.svr
    base_mv = engine.state.mv

    engine.give_drug_bolus("ketamine", 2.0 * engine.patient.weight)
    _advance(engine, 60)
    assert engine.state.loc > 0.9
    _advance(engine, 120)
    assert 1.15 * base_map < engine.state.map < 1.4 * base_map
    assert 1.2 * base_hr < engine.state.hr < 1.45 * base_hr
    assert 1.15 * base_co < engine.state.co < 1.45 * base_co
    assert abs(engine.state.svr / base_svr - 1.0) < 0.1
    assert engine.state.mv > 0.85 * base_mv
    assert engine.state.airway_obstruction < 0.05
    assert engine.state.bis > 90.0
    assert engine.state.tol >= 0.45

    for _ in range(30 * 60):
        engine.step(1.0)
        if engine.state.loc < 0.5:
            break
    assert 0.5 < engine.state.ketamine_ce < 0.8


def test_ketamine_nitrous_maintenance_settles_to_baseline(engine_factory):
    """Idvall 1979, simplified: pressure rises with 65% N2O and settles as the
    infusion falls. The study's titrated infusions and premedication are not replayed.
    """
    engine = _awake(engine_factory, age=67, weight=66)
    base_map, base_hr, base_co = engine.state.map, engine.state.hr, engine.state.co
    engine.give_drug_bolus("ketamine", 2.0 * engine.patient.weight)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(1.0, 0.0, n2o_l_min=2.0)
    engine.set_drug_rate("ketamine", 0.058 * 60.0)  # 58 mcg/kg/min
    peak_map = 0.0
    for _ in range(600):
        engine.step(1.0)
        peak_map = max(peak_map, engine.state.map)
    assert peak_map > 1.12 * base_map

    engine.set_drug_rate("ketamine", 0.030 * 60.0)
    _advance(engine, 50 * 60, dt=2.0)
    assert abs(engine.state.map / base_map - 1.0) < 0.10
    assert abs(engine.state.hr / base_hr - 1.0) < 0.10
    assert engine.state.co < 1.03 * base_co


def test_ketamine_with_atropine_and_nitrous_uses_premedicated_hr_baseline(engine_factory):
    """Idvall 1979: after atropine 0.5 mg IM, ketamine 2 mg/kg over 1 min and
    65% N2O, HR peaked 19% above the postpremedication baseline within 10 min,
    with wide variability. IV atropine 30 min earlier stands in for IM
    premedication; the infusion is the reported first-30-min mean.
    """
    engine = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=3), start=True, age=67, weight=66.5,
    )
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(4.0, 0.0)
    unpremedicated_hr = engine.state.hr
    engine.give_drug_bolus("atropine", 0.5)
    for _ in range(28):
        engine.pk["atropine"].step(60.0, 0.0)
    _advance(engine, 120, dt=2.0)
    premedicated_hr = engine.state.hr
    assert premedicated_hr > unpremedicated_hr + 10.0

    engine.set_drug_rate("ketamine", 2.0 * 60.0)
    trace = []
    for second in range(600):
        if second == 60:
            engine.set_drug_rate("ketamine", 0.058 * 60.0)
            engine.set_fgf(2.1, 0.0, n2o_l_min=3.9)
        engine.step(1.0)
        trace.append(engine.state.hr)
    peak_hr = max(trace)
    peak_time = trace.index(peak_hr) + 1
    assert 1.05 < peak_hr / premedicated_hr < 1.40
    assert 60 < peak_time < 600
    assert engine.state.mac_n2o > 0.4  # N2O has reached the alveoli.


def test_bovill_ketamine_hr_and_map_rises_after_iv_atropine(engine_factory):
    """Bovill 1972 Table 1 and Fig. 1: ketamine 2 mg/kg over 30 s added
    17 ± 2.5 bpm and 27 ± 1.7 mmHg peak (21 mmHg on the minute-mean curve)
    without premedication, and 18 ± 4.7 bpm and 19 ± 2.1 mmHg above
    pre-ketamine (25 ± 2.0 mmHg above rest) after 2 mg IV atropine. Atropine
    raised HR from 90 to 149 bpm and MAP from 92 to 98 mmHg before ketamine.

    Doses and timing match; the paper gives no ages or weights for its fit young
    women, so assume 30 years, 60 kg, and 165 cm.
    """
    dt = 1.0
    hr_rises = []
    map_rises = []
    for atropine in (0.0, 2.0):
        engine = engine_factory(
            config=SimulationConfig(mode="awake", rng_seed=3), start=True,
            age=30, weight=60, height=165, sex="female",
            baseline_hr=90 if atropine else 95, baseline_map=92 if atropine else 95,
        )
        engine.set_airway_mode("Mask")
        engine.set_fgf(8.0, 0.0)
        _advance(engine, 60, dt)
        resting_hr, resting_map = engine.state.hr, engine.state.map
        # Supplement over 5 min, then 2 min before ketamine.
        for _ in range(round(300 / dt)):
            engine.give_drug_bolus("atropine", atropine * dt / 300.0)
            engine.step(dt)
        _advance(engine, 120, dt)
        before_hr, before_map = engine.state.hr, engine.state.map
        if atropine:
            assert 45.0 < before_hr - resting_hr < 70.0
            assert 1.0 < before_map - resting_map < 10.0
        engine.set_drug_rate("ketamine", 2.0 * 120.0)
        readings = []
        for index in range(round(420 / dt)):
            if index == round(30 / dt):
                engine.set_drug_rate("ketamine", 0.0)
            engine.step(dt)
            if (index + 1) % round(60 / dt) == 0:
                readings.append((engine.state.hr, engine.state.map))
        peak_hr = max(hr for hr, _ in readings)
        peak_map = max(map_value for _, map_value in readings)
        peak_minute = max(range(len(readings)), key=lambda index: readings[index][0]) + 1
        hr_rise = peak_hr - before_hr
        map_rise = peak_map - before_map
        assert hr_rise == pytest.approx(18.0 if atropine else 17.0, abs=6.0)
        assert peak_minute <= 3
        if atropine:
            assert 15.0 < map_rise < 27.0
            assert 20.0 < peak_map - resting_map < 32.0
        else:
            assert 19.0 < map_rise < 31.0
        hr_rises.append(hr_rise)
        map_rises.append(map_rise)
    assert abs(hr_rises[1] - hr_rises[0]) < 8.0
    assert map_rises[1] < map_rises[0]


def test_ketamine_pressor_response_persists_during_pacing_and_beta_block_until_alpha_blocked(
    engine_factory, monkeypatch,
):
    """Tweed 1972 Table 3: right atrial pacing at 1.5x resting HR (n=6 vs n=3
    unpaced) lowered SVI 29% and SVR 16% with a 5.5 mmHg pressure rise, and
    ketamine 2 mg/kg over 1 min then raised CI 26-34% and aortic pressure
    17-24.5 mmHg (SVR +19-32%, NS). Bovill 1972 and Bålfors 1983: beta blockade
    leaves a substantial pressor response, whereas droperidol (modeled here via
    alpha-adrenergic blockade) attenuates it.
    """
    paced = _awake(engine_factory, age=39, weight=72.3, height=172, sex="male", baseline_hr=73.5, baseline_map=89.0)
    unpaced_sv, unpaced_svr, unpaced_map = paced.state.sv, paced.state.svr, paced.state.map
    # Fixed-rate right atrial pacing at 1.5x resting HR (110 bpm).
    monkeypatch.setattr(paced.hemo, "_rhythm_rate", lambda _: 110.0)
    paced.hemo.invalidate_state_cache()
    _advance(paced, 240, dt=2.0)
    ctrl_co, ctrl_svr, ctrl_map = paced.state.co, paced.state.svr, paced.state.map
    assert 0.65 * unpaced_sv < paced.state.sv < 0.88 * unpaced_sv
    assert 0.75 * unpaced_svr < ctrl_svr < 0.90 * unpaced_svr
    assert 0.0 < ctrl_map - unpaced_map < 8.0

    for _ in range(60):
        paced.give_drug_bolus("ketamine", 2.0 * paced.patient.weight / 60.0)
        paced.step(1.0)
    traces = []
    for _ in range(300):
        paced.step(1.0)
        traces.append((paced.state.co, paced.state.svr, paced.state.map))
    assert max(co for co, _, _ in traces) > 1.10 * ctrl_co
    assert max(svr for _, svr, _ in traces) > 1.10 * ctrl_svr
    assert 15.0 < max(mp for _, _, mp in traces) - ctrl_map < 30.0

    early_map_rises = {}
    pt_bovill = Patient(age=30, weight=60, height=165, sex="female", baseline_hr=95, baseline_map=95)
    for arm in ("none", "esmolol", "labetalol"):
        hemo = HemodynamicModel(pt_bovill)
        ket_pk = KetaminePK(pt_bovill)
        esm_pk = EsmololPK(pt_bovill)
        lab_pk = LabetalolPK(pt_bovill)
        base_bv = hemo.blood_volume_0
        base_co = hemo.base_co_l_min / hemo.failure.output_factor
        state = hemo.state
        for _ in range(300):
            esm_rate = 80.0 * 60.0 / 60000.0 if arm == "esmolol" else 0.0
            if arm == "labetalol":
                lab_pk.state.c1 += (40.0 * 1000.0 / 300.0) / lab_pk.v1
            lab_pk.update_hemodynamics(1.0, state.co / base_co, hemo.renal_perfusion)
            ce_esm = esm_pk.step(1.0, esm_rate).ce
            ce_lab = lab_pk.step(1.0, 0.0).ce
            state = hemo.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_esmolol=ce_esm, ce_labetalol=ce_lab)
        for _ in range(120):
            lab_pk.update_hemodynamics(1.0, state.co / base_co, hemo.renal_perfusion)
            ce_esm = esm_pk.step(1.0, 80.0 * 60.0 / 60000.0 if arm == "esmolol" else 0.0).ce
            ce_lab = lab_pk.step(1.0, 0.0).ce
            state = hemo.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ce_esmolol=ce_esm, ce_labetalol=ce_lab)
        before_map = state.map
        peaks = []
        for idx in range(180):
            ket_rate = 2.0 * 60.0 / 30.0 if idx < 30 else 0.0
            ket_pk.update_hemodynamics(hemo.blood_volume / base_bv, state.co / base_co, hemo.renal_perfusion)
            lab_pk.update_hemodynamics(hemo.blood_volume / base_bv, state.co / base_co, hemo.renal_perfusion)
            ket_pk.step(1.0, ket_rate)
            ce_esm = esm_pk.step(1.0, 80.0 * 60.0 / 60000.0 if arm == "esmolol" else 0.0).ce
            ce_lab = lab_pk.step(1.0, 0.0).ce
            state = hemo.step(
                1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0,
                ce_esmolol=ce_esm, ce_labetalol=ce_lab,
                ce_ketamine=ket_pk.state.ce_cv, ce_ketamine_hr=ket_pk.state.ce_hr,
            )
            peaks.append(state.map)
        early_map_rises[arm] = max(peaks) - before_map
    assert early_map_rises["esmolol"] > 14.0
    assert early_map_rises["labetalol"] < 0.45 * early_map_rises["none"]

    direct = HemodynamicModel(paced.patient)
    base_direct = direct.step(1.0, 0, 0, 0, -2, 40, 95)
    for _ in range(60):
        ket_direct = direct.step(1.0, 0, 0, 0, -2, 40, 95, ce_ketamine=2.5)
    assert ket_direct.hr > base_direct.hr + 10.0
    assert ket_direct.map > base_direct.map + 15.0


def test_midazolam_sedates_more_with_age_and_potentiates_propofol(engine_factory):
    """Albrecht 1999: elderly volunteers needed half the young EC50. Short 1992:
    midazolam lowered the propofol ED50 for anesthesia by 52%.
    """
    engine = _awake(engine_factory)
    engine.give_drug_bolus("midazolam", 2.0)
    _advance(engine, 180, dt=2.0)
    assert engine.state.loc < 0.05
    assert engine.state.bis > 85.0

    peak_loc = {}
    for age in (26, 70):
        pt = Patient(age=age)
        mid_pk = MidazolamPK(pt)
        crm = ClinicalResponseModel(age=age)
        mid_c50 = midazolam_loss_of_response(age)
        mid_pk.state.c1 += 0.1 * pt.weight * 1000.0 / mid_pk.v1
        peaks = []
        for _ in range(900):
            ce = mid_pk.step(1.0, 0.0).ce
            hyp = hypnotic_equivalent(0.0, 0.0, ce, mid_c50, crm.age_factor)
            peaks.append(crm.loss_of_response(hyp, 0.0))
        peak_loc[age] = max(peaks)
    assert peak_loc[70] > 2.0 * peak_loc[26]

    # Probe near the LOC transition; 70 and 140 mg both saturate the clinical
    # response and cannot distinguish the interaction.
    def _induction_loc(midazolam_mg: float, propofol_mg: float) -> float:
        pt = Patient()
        prop_pk = PropofolPKEleveld(pt)
        mid_pk = MidazolamPK(pt)
        crm = ClinicalResponseModel(age=pt.age)
        mid_c50 = midazolam_loss_of_response(pt.age)
        if midazolam_mg > 0.0:
            mid_pk.state.c1 += midazolam_mg * 1000.0 / mid_pk.v1
            for _ in range(180):
                prop_cp, mid_cp = prop_pk.state.c1, mid_pk.state.c1
                prop_pk.update_midazolam(mid_cp)
                mid_pk.update_propofol(prop_cp)
                prop_pk.step(1.0, 0.0)
                mid_pk.step(1.0, 0.0)
        prop_pk.state.c1 += propofol_mg / prop_pk.v1
        for _ in range(90):
            prop_cp, mid_cp = prop_pk.state.c1, mid_pk.state.c1
            prop_pk.update_midazolam(mid_cp)
            mid_pk.update_propofol(prop_cp)
            prop_pk.step(1.0, 0.0)
            mid_pk.step(1.0, 0.0)
        hyp = hypnotic_equivalent(
            PROPOFOL_LOC_TRANSFER * prop_pk.state.ce_response, 0.0, mid_pk.state.ce, mid_c50, crm.age_factor
        )
        return crm.loss_of_response(hyp, 0.0)

    half_loc = _induction_loc(0.0, 25.0)
    with_midaz_loc = _induction_loc(2.0, 25.0)
    full_loc = _induction_loc(0.0, 50.0)
    assert half_loc < 0.4
    assert with_midaz_loc > full_loc


def test_fentanyl_depresses_breathing_and_accumulates(engine_factory):
    """Fentanyl acts as 0.82 times its concentration of remifentanil (isoflurane
    MAC reduction: McEwan 1993; Lang 1996). Hughes 1992: the half-time after an
    infusion depends on its duration.
    """
    engine = _awake(engine_factory)
    engine.set_fgf(0.0, 8.0)
    base_paco2 = engine.state.pa_co2
    base_rr = engine.resp.state.rr
    engine.give_drug_bolus("fentanyl", 100.0)
    _advance(engine, 300)
    # Tissue uptake buffers the CO2 rise; the RR bound tests drug depression.
    assert engine.state.pa_co2 > base_paco2
    assert engine.resp.state.rr < 0.85 * base_rr

    engine = engine_factory(config=SimulationConfig(mode="awake", tci_enabled=True, rng_seed=3), start=True)
    engine.enable_tci("fentanyl", 2.0)
    _advance(engine, 15 * 60, dt=2.0)
    assert engine.prediction_model("fentanyl").state.ce == pytest.approx(2.0, rel=0.03)
    short_half_time = engine.get_predicted_csht("fentanyl")

    # Keep the short pump history in the engine. The three-hour accumulation
    # comparison only needs PK; long full-engine stability has its own test.
    pk = FentanylPK(engine.patient)
    pk.update_hemodynamics(1.0, ANESTHETIZED_OUTPUT)
    tci = TCIController(pk, max_rate=1000.0 / 60.0, sampling_time=1.0)
    for _ in range(180 * 60):
        pk.step(1.0, tci.step(2.0))
    assert pk.state.ce == pytest.approx(2.0, rel=0.03)
    long_half_time = pk.simulate_decay(max_seconds=7200)
    assert long_half_time > 3.0 * short_half_time
    assert long_half_time > 60.0


def test_fentanyl_blunts_the_intubation_response_more_than_lidocaine(engine_factory):
    """Ebert 1992: without opioid, intubation after propofol raised HR 36 bpm and
    MAP 35 mmHg. Qin 2025 (18 trials): lidocaine 1-2 mg/kg lowered the MAP rise at
    intubation by 3.85 mmHg (95% CI 1.09-6.61), with no significant HR effect in
    Western trials; Wilson 1991 found HR rose 21-26% with or without it. Okuda
    1990: intubation 30 s after lidocaine raised MAP and HR, unlike at 1-3 min.
    """
    base = _awake(engine_factory)
    fent = copy.deepcopy(base)
    fent.give_drug_bolus("fentanyl", 140.0)
    _advance(base, 90, dt=2.0)
    _advance(fent, 90, dt=2.0)

    lido = copy.deepcopy(base)
    lido.give_drug_bolus("lidocaine", 105.0)
    for eng in (base, fent, lido):
        eng.give_drug_bolus("propofol", 140.0)
        _advance(eng, 90, dt=1.0)
        eng.give_drug_bolus("roc", 0.6 * eng.patient.weight)
        _advance(eng, 90, dt=1.0)
        eng.set_airway_mode("ETT")
        eng.set_vent_power(True)

    rushed = copy.deepcopy(base)
    rushed.give_drug_bolus("lidocaine", 105.0)
    for eng in (base, fent, lido, rushed):
        _advance(eng, 30, dt=1.0)

    hr_alone, map_alone, tol_alone = _measure_intubation_peak(base)
    assert 28.0 < hr_alone < 45.0
    assert 28.0 < map_alone < 45.0
    hr_fentanyl, map_fentanyl, tol_fentanyl = _measure_intubation_peak(fent)
    hr_lidocaine, map_lidocaine, _ = _measure_intubation_peak(lido)
    _, map_rushed, _ = _measure_intubation_peak(rushed)

    assert tol_fentanyl > tol_alone + 0.2
    assert hr_fentanyl < 0.7 * hr_alone
    assert 0.0 < map_fentanyl < map_alone
    assert 1.09 < map_alone - map_lidocaine < 6.61
    assert hr_lidocaine > 0.8 * hr_alone
    # Lidocaine reaches its effect site over minutes, so 30 s gives clearly less blunting.
    assert map_alone - map_rushed < 0.8 * (map_alone - map_lidocaine)


def test_etomidate_induction_is_brief_and_hemodynamically_stable(engine_factory):
    """Valk 2021: 0.3 mg/kg gives 5-10 min of hypnosis with minimal change in
    blood pressure and less ventilatory depression than other hypnotics.
    Arden 1986: a smaller initial volume deepens the effect in the elderly.
    """
    base_awake = _awake(engine_factory)
    prop = copy.deepcopy(base_awake)
    etom = base_awake

    etom_base_map = etom.state.map
    etom.give_drug_bolus("etomidate", 0.3 * etom.patient.weight)
    trace = []
    for _ in range(450):
        etom.step(2.0)
        trace.append((etom.state.loc, etom.state.map, etom.state.mv))
    assert trace[59][0] > 0.85
    awake_at = 2 * next(idx for idx, (loc, _, _) in enumerate(trace) if idx > 30 and loc < 0.5)
    assert 4 * 60 < awake_at < 10 * 60
    assert min(map_ for _, map_, _ in trace) > 0.95 * etom_base_map

    prop_base = prop.state.map
    prop.give_drug_bolus("propofol", 2.0 * prop.patient.weight)
    propofol_trace = []
    for _ in range(240):
        prop.step(1.0)
        propofol_trace.append((prop.state.loc, prop.state.map, prop.state.mv))
    assert min(map_ for _, map_, _ in propofol_trace) / prop_base < min(map_ for _, map_, _ in trace) / etom_base_map
    assert min(mv for _, _, mv in trace) > min(mv for _, _, mv in propofol_trace)

    peak_loc = {}
    for age in (25, 70):
        pt = Patient(age=age)
        pk = EtomidatePK(pt)
        crm = ClinicalResponseModel(age=age)
        pk.state.c1 += 0.2 * pt.weight / pk.v1
        peaks = []
        for _ in range(300):
            ce = pk.step(1.0, 0.0).ce
            hyp = hypnotic_equivalent(0.0, ce, 0.0, 605.0, crm.age_factor)
            peaks.append(crm.loss_of_response(hyp, 0.0))
        peak_loc[age] = max(peaks)
    assert peak_loc[70] > peak_loc[25] + 0.1


def test_ketamine_raises_ga_bis_and_depresses_catecholamine_depleted_shock(engine_factory):
    """Ketamine increases BIS during general anesthesia (calibrated from sevoflurane
    data in Hans 2005 and extended to TIVA) and unmasks direct myocardial depression
    when sympathetic reserve is exhausted in hemorrhagic shock (Waxman 1980).
    """
    ga = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    shock_ctrl = copy.deepcopy(ga)
    bis_before = ga.state.bis
    ga.give_drug_bolus("ketamine", 140.0)
    _advance(ga, 90)
    assert ga.state.bis > bis_before + 4.0

    shock_ctrl.hemo.add_volume(-1850.0)
    _advance(shock_ctrl, 60)
    shock_ket = copy.deepcopy(shock_ctrl)
    shock_ket.give_drug_bolus("ketamine", 140.0)
    for model in (shock_ctrl, shock_ket):
        _advance(model, 90)
    assert shock_ket.state.map < 0.90 * shock_ctrl.state.map


def test_midazolam_induction_preserves_breathing_bis_and_thermoregulation_until_opioid(engine_factory):
    """0.3 mg/kg midazolam preserves spontaneous breathing with moderate PaCO2 rise
    (Marty 1986, with nasal O2) and causes little core cooling (Kurz 1995b), while
    AnaSim caps midazolam-only BIS above 65 and lowers MAP further when combined
    with an opioid (consistent with midazolam-fentanyl hypotension in Heikkilä 1984).
    """
    midaz = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1),
        start=True, age=43, weight=72.0, height=175.0, sex="male",
    )
    map_0 = midaz.state.map
    for _ in range(15):
        midaz.give_drug_bolus("midazolam", 0.3 * 72.0 / 15.0)
        midaz.step(4.0)
    _advance(midaz, 540, dt=4.0)
    assert midaz.state.bis > 65.0
    assert midaz.state.rr > 8.0 and not midaz.state.apnea
    assert 42.0 < midaz.state.pa_co2 < 49.0 and midaz.state.spo2 > 90.0

    _advance(midaz, 1200, dt=4.0)
    assert 37.0 - midaz.state.temp_c < 0.45

    midaz.set_airway_mode("ETT")
    midaz.set_vent_power(True)
    midaz.set_fgf(6.0, 0.0)
    midaz.give_drug_bolus("fentanyl", 250.0)
    _advance(midaz, 420, dt=2.0)
    assert midaz.state.map < map_0 - 10.0


def test_lidocaine_mac_sparing_and_awake_ceiling():
    """Systemic lidocaine spares sevoflurane MAC tolerance while capping lidocaine-only
    tolerance below surgical unresponsiveness.
    """
    crm = ClinicalResponseModel(40.0)
    assert crm.tolerance(0.0, 0.0, ce_lidocaine=50.0) <= 0.251
    assert crm.tolerance(0.0, 0.0, mac_sevo=1.0, ce_lidocaine=4.0) > (
        crm.tolerance(0.0, 0.0, mac_sevo=1.0, ce_lidocaine=0.0) + 0.05
    )


def test_short_1992_equal_share_propofol_midazolam_isobole():
    """Short 1992: equal C50 shares of propofol and midazolam (0.315 each, 37% less than additive)
    produce 50% loss of response."""
    crm = ClinicalResponseModel(40.0)
    m_c50 = midazolam_loss_of_response(40.0)
    eq = hypnotic_equivalent(
        0.315 * PROPOFOL_LOSS_OF_RESPONSE * crm.age_factor,
        0.0,
        0.315 * m_c50,
        m_c50,
        crm.age_factor,
    )
    assert crm.loss_of_response(eq, 0.0) == pytest.approx(0.5, abs=0.02)
