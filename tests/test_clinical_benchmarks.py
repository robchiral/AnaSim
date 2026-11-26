"""Consolidated primary-literature clinical trial benchmarks.

Each benchmark replays a published human study protocol (primary references
cataloged in docs/REFERENCES.md) and checks quantitative direction and magnitude
bounds so parameter changes cannot silently break clinical trial calibration.
"""

from __future__ import annotations

import copy

import pytest

from anasim.core.clinical_actions import check_intubation
from anasim.core.state import SimulationConfig
from anasim.monitors.capno import Capnograph
from anasim.patient.patient import Patient
from anasim.patient.pd.nmba import TOFModel
from anasim.patient.pk_models import RemifentanilPKEleveld, RocuroniumPK
from anasim.physiology.hemodynamics import HemodynamicModel

pytestmark = pytest.mark.clinical


def test_drobin_1999_hemorrhage_suppresses_crystalloid_diuresis():
    """Drobin & Hahn 1999 (n=10 crossover men, 25 mL/kg Ringer's acetate over 30 min after 0/450/900 mL bleed;
    modeled with balanced crystalloid): renal elimination k_r fell -59% at 450 mL and -68% at 900 mL despite modest MAP reduction."""
    patient = Patient(age=28.0, weight=76.0, height=178.0, sex="male")
    uops = []
    for bleed_ml in (0.0, 450.0, 900.0):
        hemo = HemodynamicModel(patient)
        hemo.step(0.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
        if bleed_ml > 0.0:
            hemo.add_volume(-bleed_ml)
            for _ in range(12):
                hemo.step(5.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
        u0 = hemo.total_urine_out_ml
        bolus_ml = 25.0 * 76.0
        for _ in range(30):
            hemo.add_volume(bolus_ml / 30.0, label="crystalloid")
            for _ in range(12):
                hemo.step(5.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
        for _ in range(150 * 12):
            hemo.step(5.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
        uops.append(hemo.total_urine_out_ml - u0)
    assert uops[0] > uops[1] > uops[2]
    assert uops[2] < 0.68 * uops[0]


def test_fowler_1986_and_steen_1978_hfref_beta1_downregulation_vs_calcium(engine_factory, advance_time):
    """Fowler 1986: severe heart failure blunts dobutamine inotropy (modeled in HFrEF).
    Steen 1978 (n=34 post-CPB patients with preop LV dysfunction): epi 0.04 mcg/kg/min raises CO and widens PP."""
    def _dob_ci_gain(heart_failure: str) -> float:
        eng = engine_factory(
            config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1),
            start=True,
            heart_failure=heart_failure,
        )
        advance_time(eng, 30.0, dt=1.0)
        ci0 = eng.state.co / eng.patient.bsa
        eng.set_drug_rate("dobu", 8.2)
        advance_time(eng, 300.0, dt=1.0)
        return eng.state.co / eng.patient.bsa - ci0

    normal_gain = _dob_ci_gain("none")
    hfref_gain = _dob_ci_gain("hfref")
    assert normal_gain > 0.5
    assert 0.25 * normal_gain < hfref_gain < 0.85 * normal_gain

    # Steen 1978 dose (0.04 mcg/kg/min epinephrine) applied in HFrEF raises CO and widens PP.
    epi = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1),
        start=True,
        heart_failure="hfref",
    )
    pp0 = epi.state.sbp - epi.state.dbp
    co0 = epi.state.co
    epi.set_drug_rate("epi", 0.04 * epi.patient.weight)
    advance_time(epi, 300.0, dt=1.0)
    assert epi.state.co > co0 * 1.05
    assert (epi.state.sbp - epi.state.dbp) > pp0 * 1.15


def test_richards_1977_iv_labetalol_bolus_hypotension(engine_factory, advance_time):
    """Richards 1977 (n=6 supine adults, IV labetalol 1.5 mg/kg): acute alpha-mediated MAP fall
    of ~15-25% with blunted reflex tachycardia."""
    eng = engine_factory(
        config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1),
        start=True,
        age=33.0,
        weight=72.0,
        baseline_map=84.0,
        baseline_hr=68.0,
    )
    advance_time(eng, 30.0, dt=1.0)
    map0, hr0 = eng.state.map, eng.state.hr
    eng.give_drug_bolus("labetalol", 1.5 * 72.0)
    min_map = map0
    max_hr = hr0
    for _ in range(300):
        eng.step(1.0)
        min_map = min(min_map, eng.state.map)
        max_hr = max(max_hr, eng.state.hr)
    rel_drop = (map0 - min_map) / map0
    assert 0.10 <= rel_drop <= 0.36
    assert max_hr - hr0 < 15.0


def test_caldwell_2000_and_smeulers_1995_hypothermic_rocuronium():
    """Caldwell 2000 (vecuronium at 34.0-38.0 °C, CL -11.3%/°C, slower ke0; extrapolated to rocuronium)
    and Smeulers 1995 (rocuronium at 27.7 °C CPB): core cooling progressively prolongs 0.6 mg/kg rocuronium duration to T1 25%."""
    patient = Patient()
    t25s = []
    for temp_c in (37.0, 34.3, 27.7):
        pk = RocuroniumPK(patient)
        tof = TOFModel(patient)
        pk.update_hemodynamics(1.0, 1.0, 1.0, temp_c=temp_c)
        tof._temp_c = temp_c
        pk.state.c1 += 0.6 * patient.weight / pk.v1
        t25 = None
        for sec in range(1, 18000):
            pk.step(1.0, 0.0)
            tof.step_recovery(1.0, pk)
            tw = (1.0 - tof.twitch_block(tof.ce)) * 100.0
            if sec > 300 and tw >= 25.0:
                t25 = float(sec)
                break
        assert t25 is not None
        t25s.append(t25)
    assert t25s[0] < t25s[1] < t25s[2]
    assert t25s[1] > t25s[0] * 1.15
    assert t25s[2] > t25s[0] * 2.0


def test_van_miert_1997_and_lowry_1998_rocuronium_cirrhosis_and_sevoflurane(engine_factory):
    """van Miert 1997 (cirrhosis vs healthy): similar 90% onset and ~50% longer TOF recovery.
    Lowry 1998 (sevoflurane vs propofol): rapid ~1 min bolus onset with prolonged TOF 0.7 recovery."""
    def _onset_and_tof70(maint_type: str = "tiva", hepatic_function: float = 1.0) -> tuple[float, float]:
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type=maint_type, dt=1.0, rng_seed=1),
            start=True,
            hepatic_function=hepatic_function,
        )
        eng.give_drug_bolus("roc", 0.6 * eng.patient.weight)
        onset_s = None
        for _ in range(300):
            eng.step(1.0)
            if onset_s is None and eng.state.twitch <= 10.0:
                onset_s = eng.state.time
        # Advance rocuronium PK/PD directly at settled steady-state MAC/CO to keep recovery replay fast.
        roc = eng.pk["roc"]
        tof = eng.tof_pd
        mac_sevo, mac_n2o = eng.state.mac_sevo, eng.state.mac_n2o
        tof70_s = None
        for sec in range(305, 10805, 5):
            roc.step(5.0, 0.0)
            tof_pct = tof.step_recovery(5.0, roc, mac_sevo=mac_sevo, mac_n2o=mac_n2o)
            tw_pct = (1.0 - tof.twitch_block(tof.ce, mac_sevo, mac_n2o)) * 100.0
            if tof_pct >= 70.0 and tw_pct >= 70.0:
                tof70_s = float(sec)
                break
        assert onset_s is not None and tof70_s is not None
        return onset_s, tof70_s

    on_norm, rec_norm = _onset_and_tof70("tiva", 1.0)
    on_cirr, rec_cirr = _onset_and_tof70("tiva", 0.5)
    on_sevo, rec_sevo = _onset_and_tof70("balanced", 1.0)

    # Cirrhosis preserves bolus onset within 25 s; sevoflurane does not delay bolus onset (~1 min).
    assert abs(on_cirr - on_norm) < 25.0
    assert 40.0 <= on_sevo <= on_norm + 15.0
    # Markedly prolonged recovery in both cirrhosis (>25%) and sevoflurane (>35%).
    assert rec_cirr > rec_norm * 1.25
    assert rec_sevo > rec_norm * 1.35


def test_egan_1998_remifentanil_obese_deep_volume_does_not_collapse():
    """Egan 1998 (n=12 obese 113 kg vs n=12 lean 64 kg): remifentanil distributes with lean mass;
    Eleveld V3 must stay bounded rather than collapsing exponentially at extreme TBW."""
    lean = RemifentanilPKEleveld(Patient(age=38.0, weight=64.0, height=170.0, sex="female"))
    obese = RemifentanilPKEleveld(Patient(age=38.0, weight=113.0, height=171.0, sex="female"))
    super_obese = RemifentanilPKEleveld(Patient(age=38.0, weight=195.0, height=182.0, sex="male"))
    assert obese.v3 > 0.30 * lean.v3
    assert super_obese.v3 > 1.0


def test_kimura_1994_unparalyzed_sevoflurane_intubation_threshold(engine_factory, advance_time):
    """Kimura 1994 (n=86, MAC_EI = 4.52% = 2.86x MAC): light sevoflurane alone blocks unparalyzed
    intubation, whereas deep volatile anesthesia (>= 1.5 MAC) permits placement."""
    eng = engine_factory(config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1), start=True)
    eng.pk_sevo.state.p_brain = eng.pk_sevo.state.p_alv = 0.5 * eng.pk_sevo.mac_age / 100.0
    eng.pk_sevo.state.mac = 0.5
    light = check_intubation(eng)
    assert light is not None and light.blocked

    eng.pk_sevo.state.p_brain = eng.pk_sevo.state.p_alv = 1.6 * eng.pk_sevo.mac_age / 100.0
    eng.pk_sevo.state.mac = 1.6
    deep = check_intubation(eng)
    assert deep is None or not deep.blocked


def test_ludbrook_1958_hypothermic_citrate_clearance(engine_factory):
    """Ludbrook & Wynn 1958 (28-29 °C hypothermia slows citrate metabolism ~42% and raises peak levels ~65%;
    AnaSim's Q10 scaling yields ~64% clearance at 28.5 °C)."""
    warm = engine_factory(config=SimulationConfig(mode="awake", dt=2.0, rng_seed=1), start=True, baseline_temp=37.0)
    cold = engine_factory(config=SimulationConfig(mode="awake", dt=2.0, rng_seed=1), start=True, baseline_temp=28.5)
    warm.hemo.fluids.citrate_mmol = 10.0
    cold.hemo.fluids.citrate_mmol = 10.0
    for _ in range(300):
        warm.state.temp_c = 37.0
        cold.state.temp_c = 28.5
        warm.step(2.0)
        cold.step(2.0)
    warm_cleared = 10.0 - warm.hemo.fluids.citrate_mmol
    cold_cleared = 10.0 - cold.hemo.fluids.citrate_mmol
    ratio = cold_cleared / warm_cleared
    assert 0.50 <= ratio <= 0.78


def test_ozaki_1997_elderly_vasoconstriction_threshold_shift(engine_factory, advance_time):
    """Ozaki 1997 (n=25, 60-80 vs 30-50 yr under 50% N2O / 1% sevoflurane): elderly vasoconstriction threshold shifts ~0.8 °C lower."""
    young = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=1.0, rng_seed=1),
        start=True,
        age=45.0,
    )
    elderly = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=1.0, rng_seed=1),
        start=True,
        age=73.0,
    )
    advance_time(young, 10.0, dt=1.0)
    advance_time(elderly, 10.0, dt=1.0)
    # At 34.4 °C under balanced anesthesia, the young patient vasoconstricts more strongly than the elderly patient.
    y_loss = young.thermal.step_temperature(60.0, 34.4, 0.0, perfusing=True)
    e_loss = elderly.thermal.step_temperature(60.0, 34.4, 0.0, perfusing=True)
    assert e_loss < y_loss


def test_malan_1995_sevoflurane_lowers_co_before_steep_svr_drop(engine_factory, advance_time):
    """Malan 1995 (n=7 sevoflurane-only volunteers): at unstimulated ~1 MAC sevoflurane,
    CO falls while SVR stays near baseline."""
    eng = engine_factory(
        config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1),
        start=True,
        age=26.0,
        weight=75.0,
        height=178.0,
        sex="male",
    )
    advance_time(eng, 30.0, dt=1.0)
    co0, svr0 = eng.state.co, eng.state.svr
    eng.set_airway_mode("ETT")
    eng.set_vent_settings(rr=12, vt=0.5, peep=0.0, ie="1:2", mode="VCV")
    eng.set_vent_power(True)
    eng.set_fgf(6.0, 0.0)
    eng.set_vaporizer(2.5)
    advance_time(eng, 900.0, dt=1.0)
    co_ratio_1mac = eng.state.co / co0
    svr_ratio_1mac = eng.state.svr / svr0
    assert 0.65 <= co_ratio_1mac <= 0.90
    assert svr_ratio_1mac > 0.85


def test_paradis_1990_kim_2025_cantineau_1994_cpr_epinephrine(engine_factory, advance_time):
    """Paradis 1990 (CPP >= 15 mmHg for ROSC), Kim 2025 (epinephrine raises CPR arterial DBP;
    +13.7 mmHg larger delta DBP in sustained-ROSC vs non-ROSC patients), and Cantineau 1994b (epi lowers fixed-ventilation CPR EtCO2)."""
    eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=1), start=True)
    eng.set_rhythm("VFIB")
    advance_time(eng, 15.0, dt=0.5)
    eng.set_chest_compressions(True)
    advance_time(eng, 60.0, dt=0.5)
    dbp0 = eng.state.dbp
    etco2_0 = eng.state.etco2
    sdr0 = eng.hemo.coronary.supply_demand_ratio
    assert 12.0 <= etco2_0 <= 24.0

    ctrl = copy.deepcopy(eng)
    eng.give_drug_bolus("epi", 1000.0)
    advance_time(eng, 120.0, dt=0.5)
    advance_time(ctrl, 120.0, dt=0.5)
    assert eng.state.dbp - dbp0 >= 8.0
    assert eng.hemo.coronary.supply_demand_ratio > sdr0 * 1.3
    assert eng.state.etco2 < ctrl.state.etco2


def test_yaron_1996_and_blanch_1999_obstructive_capnogram_slope():
    """Time-based Phase III steepening calibrated to Yaron 1996 (acute asthma ~2.0x control)
    and bounded above using Blanch 1999 (volumetric capnography in severe ALI/ARDS, ~3-4x)."""
    def _slope(obstruction: float) -> float:
        cap = Capnograph(0.15)
        for _ in range(100):
            cap.step(0.01, 0.005, 40.0)
        vals = [cap.step(0.01, -0.003, 40.0, obstruction=obstruction) for _ in range(180)]
        return (vals[175] - vals[120]) / 0.55

    normal = _slope(0.0)
    moderate = _slope(0.5)
    severe = _slope(1.0)
    assert 2.0 < normal < 10.0
    assert 1.8 <= moderate / normal <= 3.5
    assert 3.0 <= severe / normal <= 5.5


def test_rizza_1979_insulin_hypoglycemia_counterregulation(engine_factory):
    """Rizza 1979 (n=13 healthy adults, 0.04 U/kg IV insulin bolus): reported saline control nadir
    47 +/- 2 mg/dL (SEM) at 30 min, peak Ra 3.70 +/- 0.24 mg/kg/min at 40 min, recovery to 92 +/- 3 mg/dL
    by 85-90 min; checks model nadir, EGP surge, and spontaneous recovery bands."""
    from anasim.core.runtime import step_simulation

    eng = engine_factory(
        config=SimulationConfig(mode="awake", dt=2.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        start=True,
    )
    eng.give_drug_bolus("insulin", 0.04 * 70.0)
    min_g = eng.state.glucose
    peak_egp = 0.0
    for _ in range(90 * 30):
        step_simulation(eng, 2.0, update_monitors=False)
        eng.state.time += 2.0
        min_g = min(min_g, eng.state.glucose)
        peak_egp = max(peak_egp, eng.glucose.state.egp_mg_kg_min)
    assert 42.0 <= min_g <= 62.0
    assert 2.8 <= peak_egp <= 4.6
    assert eng.state.glucose > min_g + 12.0


def test_allon_1990_esrd_insulin_d50w_hyperkalemia_and_rebound_hypoglycemia(engine_factory):
    """Allon and Copkney 1990 (n=12 hemodialysis patients, 10 U regular insulin + 25 g D50W):
    reported K+ fall 0.65 +/- 0.09 mmol/L (SE) alone (1.21 +/- 0.19 with 20 mg nebulized albuterol),
    and glucose 10.3 +/- 0.6 mmol/L (~186 mg/dL) at 15 min falling to 2.8 +/- 0.3 mmol/L (~50 mg/dL) at 60 min."""
    from anasim.core.runtime import step_simulation

    eng = engine_factory(
        config=SimulationConfig(mode="awake", dt=2.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        start=True,
        renal_function=0.05,
    )
    k0 = eng.state.potassium
    eng.give_drug_bolus("insulin", 10.0)
    eng.give_d50w(50.0)
    g_15 = None
    min_k = k0
    for minute in range(1, 61):
        for _ in range(30):
            step_simulation(eng, 2.0, update_monitors=False)
            eng.state.time += 2.0
        min_k = min(min_k, eng.state.potassium)
        if minute == 15:
            g_15 = eng.state.glucose
    assert g_15 is not None and 155.0 <= g_15 <= 230.0
    assert 0.55 <= k0 - min_k <= 1.25
    assert eng.state.glucose < 70.0


def test_hillier_1999_hyperglycemic_osmotic_hyponatremia(engine_factory):
    """Hillier 1999 (n=6 somatostatin-clamped healthy adults): acute hyperglycemia draws water out of cells,
    lowering Na+ by ~1.6 mEq/L per 100 mg/dL below 400 mg/dL (2.4 overall across >600 mg/dL)."""
    eng = engine_factory(config=SimulationConfig(mode="awake", dt=1.0, maintenance_fluid_ml_hr=0.0))
    na0 = eng.state.sodium
    eng.hemo.fluids.glucose_mmol += (200.0 / 18.016) * eng.hemo.fluids.ecf_l
    eng.hemo.fluids._equilibrate_cell_water()
    g1 = (eng.hemo.fluids.glucose_mmol / eng.hemo.fluids.ecf_l) * 18.016
    slope_per_100 = (eng.hemo.fluids.concentration("sodium") - na0) / ((g1 - 90.0) / 100.0)
    assert -2.1 <= slope_per_100 <= -1.4


def test_tanaka_2005_sevoflurane_blunts_glucose_stimulated_insulin(engine_factory):
    """Tanaka 2005 (n=15 sevoflurane + n=15 isoflurane across 0.5-1.5 MAC with 66% N2O vs n=5 awake controls,
    20 g D50W IVGTT): volatile anesthesia suppresses acute insulin secretion by ~77% and slows glucose clearance."""
    from anasim.core.runtime import step_simulation

    awake = engine_factory(config=SimulationConfig(mode="awake", dt=2.0, maintenance_fluid_ml_hr=0.0))
    sevo = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=2.0, maintenance_fluid_ml_hr=0.0),
    )
    awake.give_d50w(40.0)  # 20 g IVGTT
    sevo.give_d50w(40.0)
    peak_awake_delta = 0.0
    peak_sevo_delta = 0.0
    for _ in range(30 * 30):
        step_simulation(awake, 2.0, update_monitors=False)
        step_simulation(sevo, 2.0, update_monitors=False)
        awake.state.time += 2.0
        sevo.state.time += 2.0
        peak_awake_delta = max(peak_awake_delta, awake.state.insulin_mu_l - 10.0)
        peak_sevo_delta = max(peak_sevo_delta, sevo.state.insulin_mu_l - 10.0)
    suppression = 1.0 - peak_sevo_delta / max(1e-6, peak_awake_delta)
    assert 0.55 <= suppression <= 0.88
    assert sevo.state.glucose > awake.state.glucose + 10.0


def test_cooper_1990_acidosis_bicarbonate_acid_base_etco2_and_ica(engine_factory):
    """Cooper 1990 (n=14 ICU lactic acidosis, 0.9 M NaHCO3 2 mmol/kg over 15 min on fixed MV,
    replayed equimolarly with AnaSim's 8.4% NaHCO3 in fixed mineral acidosis): raised arterial
    pH (+0.14), HCO3- (+6 mmol/L), PaCO2 (+6 mmHg in Results/Fig 1), and EtCO2 (+6 mmHg, SD, Table 2)
    while lowering ionized calcium by ~8.5% (-0.08 mmol/L, n=10, SE, Fig 3)."""
    import copy

    from anasim.core.runtime import step_simulation

    base = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=2.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        age=60.0, weight=70.0, height=170.0, sex="male", start=True,
    )
    base.set_vent_settings(rr=14.0, vt=0.48, peep=5.0, ie="1:2", mode="VCV")
    base.resp.remove_buffer_base(220.0)
    for _ in range(1800):
        step_simulation(base, 2.0, update_monitors=False)
        base.state.time += 2.0
    ctrl = copy.deepcopy(base)
    for _ in range(450):
        base.hemo.add_volume(140.0 / 450.0, label="bicarbonate")
        step_simulation(base, 2.0, update_monitors=False)
        step_simulation(ctrl, 2.0, update_monitors=False)
        base.state.time += 2.0
        ctrl.state.time += 2.0
    dph = base.state.ph - ctrl.state.ph
    dhco3 = base.state.hco3 - ctrl.state.hco3
    dpaco2 = base.state.pa_co2 - ctrl.state.pa_co2
    detco2 = base.state.etco2 - ctrl.state.etco2
    dica = base.state.ionized_calcium - ctrl.state.ionized_calcium
    assert 0.08 <= dph <= 0.19
    assert 4.5 <= dhco3 <= 10.0
    assert 3.5 <= dpaco2 <= 9.5
    assert 3.5 <= detco2 <= 9.0
    assert -0.14 <= dica <= -0.05


def test_mark_1993_intraoperative_bicarbonate_bolus_kinetics(engine_factory):
    """Mark 1993 (n=40 OR RCT, n=20 NaHCO3 vs n=20 NaCl on fixed MV, 72 +/- 16 mmol SD of
    7.2% / 0.86 M NaHCO3 over 30 s, replayed equimolarly with AnaSim's 8.4% solution against
    an untreated control): raised arterial pH (+0.03), HCO3- (+4 at 5 min, +2 at 20 min), Na+
    (+2 mEq/L), and PaCO2 transiently with negligible net K+ change vs equimolar NaCl."""
    import copy

    from anasim.core.runtime import step_simulation

    base = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        age=60.0, weight=72.0, height=172.0, sex="male", start=True,
    )
    base.set_vent_settings(rr=12.0, vt=0.52, peep=5.0, ie="1:2", mode="VCV")
    base.resp.remove_buffer_base(75.0)
    for _ in range(1800):
        step_simulation(base, 2.0, update_monitors=False)
        base.state.time += 2.0
    ctrl = copy.deepcopy(base)
    peak_dpaco2 = 0.0
    for sec in range(1, 1201):
        if sec <= 30:
            base.hemo.add_volume(72.0 / 30.0, label="bicarbonate")
        step_simulation(base, 1.0, update_monitors=False)
        step_simulation(ctrl, 1.0, update_monitors=False)
        base.state.time += 1.0
        ctrl.state.time += 1.0
        peak_dpaco2 = max(peak_dpaco2, base.state.pa_co2 - ctrl.state.pa_co2)
        if sec == 300:
            dph_5 = base.state.ph - ctrl.state.ph
            dhco3_5 = base.state.hco3 - ctrl.state.hco3
            dna_5 = base.state.sodium - ctrl.state.sodium
            dk_5 = base.state.potassium - ctrl.state.potassium
    assert peak_dpaco2 >= 6.0
    assert 0.02 <= dph_5 <= 0.08
    assert 2.0 <= dhco3_5 <= 5.5
    assert 1.0 <= dna_5 <= 3.0
    assert abs(dk_5) <= 0.25
    assert 1.5 <= base.state.hco3 - ctrl.state.hco3 <= 4.5


def test_blumberg_1988_esrd_hyperkalemia_bicarbonate_minimal_vs_acidemia(engine_factory):
    """Blumberg 1988 (n=10 ESRD, 240 mmol 8.4% NaHCO3 over 60 min): hypertonic NaHCO3 in
    non-acidemic ESRD raises pH, HCO3-, and Na+ without lowering K+ substantially (peak 5.83
    +/- 0.23 vs baseline 5.66 +/- 0.19 mmol/L SEM, NS), unlike reversing mineral metabolic acidemia."""
    import copy

    from anasim.core.runtime import step_simulation

    esrd = engine_factory(
        config=SimulationConfig(mode="awake", dt=2.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        age=55.0, weight=70.0, height=170.0, sex="male", renal_function=0.05, start=True,
    )
    scale = 5.66 / 4.0
    esrd.hemo.fluids.amounts["potassium"] *= scale
    esrd.hemo.fluids.cell_k_mmol *= scale
    for _ in range(150):
        step_simulation(esrd, 2.0, update_monitors=False)
        esrd.state.time += 2.0
    ctrl = copy.deepcopy(esrd)
    for _ in range(1800):
        esrd.hemo.add_volume(240.0 / 1800.0, label="bicarbonate")
        step_simulation(esrd, 2.0, update_monitors=False)
        step_simulation(ctrl, 2.0, update_monitors=False)
        esrd.state.time += 2.0
        ctrl.state.time += 2.0
    assert esrd.state.ph > ctrl.state.ph + 0.07
    assert esrd.state.hco3 > ctrl.state.hco3 + 9.0
    assert esrd.state.sodium > ctrl.state.sodium + 3.5
    assert abs(esrd.state.potassium - ctrl.state.potassium) < 0.25

    # In mineral metabolic acidemia, restoring pH toward 7.40 lowers K+ substantially.
    acid = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=2.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        start=True,
    )
    acid.resp.remove_buffer_base(220.0)
    for _ in range(300):
        step_simulation(acid, 2.0, update_monitors=False)
        acid.state.time += 2.0
    acid_ctrl = copy.deepcopy(acid)
    acid.give_sodium_bicarbonate(150.0)
    for _ in range(600):
        step_simulation(acid, 2.0, update_monitors=False)
        step_simulation(acid_ctrl, 2.0, update_monitors=False)
        acid.state.time += 2.0
        acid_ctrl.state.time += 2.0
    assert acid_ctrl.state.potassium - acid.state.potassium > 0.45


def test_zaballos_2019_bicarbonate_rapidly_reduces_last_conduction_block(engine_factory):
    """Zaballos 2019 (porcine bupivacaine cardiotoxicity, 2 mEq/kg NaHCO3 over 1-2 min;
    extrapolated conservatively to lidocaine): NaHCO3 rapidly reduces Na+-channel conduction block."""
    import copy

    from anasim.core.runtime import step_simulation

    base = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        start=True,
    )
    base.set_vent_settings(rr=14.0, vt=0.55, peep=5.0, ie="1:2", mode="VCV")
    # Load lidocaine directly to a severe cardiotoxic steady concentration with substantial conduction block.
    lido = base.pk["lidocaine"]
    lido.state.c1 = lido.state.c2 = lido.state.c3 = lido.state.ce_cv = 32.0
    for _ in range(20):
        step_simulation(base, 1.0, update_monitors=False)
        base.state.time += 1.0
    ctrl = copy.deepcopy(base)
    base.give_sodium_bicarbonate(140.0)
    for _ in range(180):
        step_simulation(base, 1.0, update_monitors=False)
        step_simulation(ctrl, 1.0, update_monitors=False)
        base.state.time += 1.0
        ctrl.state.time += 1.0
    assert ctrl.toxicity.conduction_block > 0.25
    assert base.toxicity.conduction_block < ctrl.toxicity.conduction_block * 0.70
    assert base.state.sodium > ctrl.state.sodium + 2.0
    assert base.state.ph > ctrl.state.ph + 0.06
