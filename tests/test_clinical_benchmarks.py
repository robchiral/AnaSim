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
    """Allon and Copkney 1990 (n=12 hemodialysis patients, K0 = 5.48 +/- 0.21 SE mmol/L,
    10 U regular insulin + 25 g D50W): reported maximal K+ fall 0.65 +/- 0.09 mmol/L (SE)
    at 45 min (1.21 +/- 0.19 with 20 mg nebulized albuterol), and glucose 10.3 +/- 0.6 mmol/L
    (~186 mg/dL) at 15 min falling to 2.8 +/- 0.3 mmol/L (~50 mg/dL) at 60 min."""
    from anasim.core.runtime import step_simulation

    eng = engine_factory(
        config=SimulationConfig(mode="awake", dt=2.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        start=True,
        renal_function=0.05,
    )
    eng.hemo.fluids.amounts["potassium"] = 5.48 * eng.hemo.fluids.ecf_l
    eng.hemo.fluids.cell_k_mmol *= 5.48 / 4.0
    eng.state.potassium = eng.hemo.fluids.arterial_potassium
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
    assert 0.55 <= k0 - min_k <= 1.15
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


def test_chothia_2014_insulin_d50w_vs_d50w_alone_and_k0_independence(engine_factory):
    """Chothia 2014 (n=10 nondiabetic HD crossover, mean +/- SD): 10 U insulin + 50 g D50W
    lowered K+ by 0.83 +/- 0.53 mmol/L at 60 min (K0 = 6.01 +/- 0.87), while 50 g D50W alone
    lowered K+ by 0.50 +/- 0.31 mmol/L (K0 = 6.23 +/- 1.20); ΔK is independent of baseline K0
    (Alvestrand 1984)."""
    from anasim.core.runtime import step_simulation

    def _run_k(k0: float, give_insulin: bool) -> float:
        eng = engine_factory(
            config=SimulationConfig(mode="awake", dt=2.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
            start=True,
            renal_function=0.05,
        )
        eng.hemo.fluids.amounts["potassium"] = k0 * eng.hemo.fluids.ecf_l
        eng.hemo.fluids.cell_k_mmol *= k0 / 4.0
        eng.state.potassium = k0
        if give_insulin:
            eng.give_drug_bolus("insulin", 10.0)
        eng.give_d50w(100.0)  # 50 g dextrose
        for _ in range(60 * 30):
            step_simulation(eng, 2.0, update_monitors=False)
            eng.state.time += 2.0
        return eng.state.potassium - k0

    dk_ins_high = _run_k(6.0, give_insulin=True)
    dk_ins_norm = _run_k(4.0, give_insulin=True)
    dk_d50_only = _run_k(6.2, give_insulin=False)

    assert -1.20 <= dk_ins_high <= -0.60
    assert -0.85 <= dk_d50_only <= -0.35
    assert dk_ins_high < dk_d50_only - 0.12
    assert abs(dk_ins_high - dk_ins_norm) < 0.15


def test_puhringer_2008_sugammadex_rsi_dose_response():
    """Pühringer 2008 (n=167, Table 4, 1.2 mg/kg rocuronium + sugammadex at 3 min,
    median [range] min to TOF 0.9): 2 mg/kg 63.3 [36.3-117.2], 4 mg/kg 11.3 [5.3-28.5],
    8 mg/kg 3.6 [1.5-4.7], 12 mg/kg 1.9 [1.2-4.1], 16 mg/kg 1.3 [0.8-2.3]."""
    from anasim.patient.pd.nmba import TOFModel
    from anasim.patient.pk_models import RocuroniumPK

    p = Patient(age=40, weight=70, height=170, sex="male")
    times = {}
    for dose_mg_kg in (2.0, 4.0, 8.0, 12.0, 16.0):
        roc = RocuroniumPK(p)
        pd = TOFModel(p)
        roc.state.c1 += 1.2 * p.weight / roc.v1
        for _ in range(180):
            roc.step(1.0, 0.0)
            pd.step_recovery(1.0, roc)
        pd.give_sugammadex(dose_mg_kg * p.weight)
        rec = None
        for sec in range(1, 5401):
            roc.step(1.0, 0.0)
            if pd.step_recovery(1.0, roc) >= 90.0:
                rec = sec / 60.0
                break
        assert rec is not None
        times[dose_mg_kg] = rec

    assert 36.0 <= times[2.0] <= 115.0
    assert 5.3 <= times[4.0] <= 25.0
    assert 1.5 <= times[8.0] <= 4.7
    assert 1.2 <= times[12.0] <= 4.1
    assert 0.8 <= times[16.0] <= 2.3
    assert times[2.0] > times[4.0] > times[8.0] > times[12.0] > times[16.0]


def test_stecker_2001_vitez_1974_and_cohen_1977_deep_hypothermia_pd_pk(engine_factory):
    """Stecker 2001 (n=47 DHCA): EEG silence at 17.8 +/- 4 °C NP (20.6 +/- 5 °C central);
    Vitez 1974 (rat, n=5/agent): volatile MAC falls ~5%/°C monotonically on cooling;
    Cohen 1977 (n=11 CPB): 25 °C heparin decay 0.21 +/- 0.12 U/mL/hr (SD) vs 0.37-2.01 at 37 °C."""
    from anasim.patient.pd.anesthesia import BISModel
    from anasim.patient.pk_models import HeparinPK, NorepinephrinePK
    from anasim.patient.volatile_pk import VolatilePK

    p = Patient(age=55, weight=75, height=175, sex="male")
    bis = BISModel(p)
    assert 15.0 <= bis.compute_bis(2.5, temp_c=25.0) <= 35.0
    assert bis.compute_bis(2.5, temp_c=21.0) < bis.compute_bis(2.5, temp_c=25.0)
    assert bis.compute_bis(2.5, temp_c=18.0) < 1.0

    # Endogenous norepinephrine remains invariant during deep hypothermia (no artificial 3x NE surge).
    nore = NorepinephrinePK(p)
    base_ne = nore.state.c1
    nore.update_hemodynamics(1.0, 1.0, 1.0, temp_c=20.0)
    for _ in range(600):
        nore.step(1.0, 0.0)
    assert nore.state.c1 == pytest.approx(base_ne, rel=0.03)

    # Volatile MAC decreases monotonically down to 18 °C (outpacing blood solubility).
    sevo = VolatilePK(p, "Sevoflurane", 0.65, 1.7, 2.05)
    assert sevo.mac_at_temp(18.0) < sevo.mac_at_temp(22.0) < sevo.mac_at_temp(28.0) < sevo.mac_at_temp(37.0)

    # Cohen 1977: post-distribution heparin decay at 25 °C vs 37 °C, and nonlinear/protamine CSHT.
    decays = {}
    for temp_c in (37.0, 25.0):
        hep = HeparinPK(p)
        hep.state.c1 = hep.state.c2 = 3500.0  # Equilibrated 3.5 U/mL on CPB
        hep.update_hemodynamics(1.0, 1.0, 1.0, temp_c=temp_c)
        for _ in range(3600):
            hep.update_nonlinear_clearance()
            hep.step(1.0, 0.0)
        decays[temp_c] = (3500.0 - hep.state.c1) / 1000.0
    assert 0.37 <= decays[37.0] <= 1.20
    assert 0.09 <= decays[25.0] <= 0.36
    assert decays[25.0] < decays[37.0] * 0.72

    eng = engine_factory(config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1), start=True)
    eng.give_drug_bolus("heparin", 21000.0)
    for _ in range(180):
        eng.step(1.0)
    csht_alone = eng.get_predicted_csht("heparin")
    eng.give_drug_bolus("protamine", 210.0)
    csht_with_prot = eng.get_predicted_csht("heparin")
    assert csht_alone > 40.0
    assert csht_with_prot < 5.0


def test_nissinen_2009_al_sarraf_2011_hypothermic_cardioplegia_protection(engine_factory):
    """Nissinen 2009 (n=3,280) and Al-Sarraf 2011 (n=3,799): 60 min of hypothermic cardioplegic
    arrest preserves myocardial viability and weaning, whereas 60 min of warm unprotected cross-clamp
    produces severe ischemic necrosis."""
    from anasim.core.enums import RhythmType
    from anasim.core.runtime import step_simulation

    def _run_clamp(cold_cardioplegia: bool):
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=4, maintenance_fluid_ml_hr=0.0),
            start=True,
        )
        eng.give_drug_bolus("heparin", 25000.0)
        eng.start_cpb(
            blood_flow_l_min=4.5, sweep_gas_l_min=3.0, prime_volume_ml=1200.0,
            heater_cooler_on=True, water_target_c=28.0 if cold_cardioplegia else 37.0,
        )
        if cold_cardioplegia:
            eng.state.temp_c = 28.0
            eng.thermal.peripheral_temp_c = 28.0
            eng.set_cpb_cardioplegia(True)
        else:
            eng.set_cpb_cross_clamp(True)
        for _ in range(3600):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
        return eng

    protected = _run_clamp(cold_cardioplegia=True)
    unprotected = _run_clamp(cold_cardioplegia=False)
    assert protected.hemo.coronary.injury < 0.05
    assert unprotected.hemo.coronary.injury > 0.50
    protected.set_cpb_cardioplegia(False)
    protected.set_cpb_cross_clamp(False)
    protected.set_rhythm(RhythmType.SINUS)
    protected.state.temp_c = 36.5
    protected.thermal.peripheral_temp_c = 36.0
    protected.stop_extracorporeal(reinfuse=True)
    for _ in range(30):
        step_simulation(protected, 1.0, update_monitors=False)
        protected.state.time += 1.0
    assert protected.state.co > 2.5 and not protected.state.cardiac_arrest


def test_feneck_1992_milrinone_pvr_reduction_and_rv_septal_shift(engine_factory):
    """Feneck 1992 (n=99, mean +/- SEM): milrinone (50 mcg/kg + 0.5 mcg/kg/min) lowers
    elevated PVR by 15-40% (14.8% at 15 min, up to 35-40% at 12 h); acute severe PVR elevation
    with high RAP induces RV overdistension and septal-shift LV filling depression."""
    from anasim.core.runtime import step_simulation

    eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=2), start=True)
    eng.start_pulmonary_embolism(obstruction=0.55)
    for _ in range(120):
        step_simulation(eng, 1.0, update_monitors=False)
        eng.state.time += 1.0
    pvr_pre = eng.hemo._calc_pvr_factor(eng.state.pao2, eng.resp_mech.peep)
    eng.give_drug_bolus("milri", 50.0 * eng.patient.weight)
    eng.set_drug_rate("milri", 0.5)
    for _ in range(900):
        step_simulation(eng, 1.0, update_monitors=False)
        eng.state.time += 1.0
    pvr_post = eng.hemo._calc_pvr_factor(eng.state.pao2, eng.resp_mech.peep)
    drop_frac = 1.0 - pvr_post / pvr_pre
    assert 0.12 <= drop_frac <= 0.45

    # Volume loading under severe acute PE raises RAP into RV overdistension and lowers SV.
    eng_rv = engine_factory(config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=2), start=True)
    eng_rv.start_pulmonary_embolism(obstruction=0.72)
    for _ in range(120):
        step_simulation(eng_rv, 1.0, update_monitors=False)
        eng_rv.state.time += 1.0
    sv_before = eng_rv.state.sv
    eng_rv.hemo.add_volume(2000.0)
    for _ in range(30):
        step_simulation(eng_rv, 1.0, update_monitors=False)
        eng_rv.state.time += 1.0
    assert eng_rv.hemo.state.rap > 12.0
    assert eng_rv.state.sv < sv_before


def test_matsukawa_1995b_ozaki_1994_and_comet_2001_neuraxial_thermal_and_motor(engine_factory):
    """Matsukawa 1995b (n=6, T10 epidural): core temp drops 0.8 +/- 0.3 °C (SD) at 60 min;
    Ozaki 1994: T10 neuraxial block lowers shivering threshold ~0.5-0.7 °C and removes leg
    shivering capacity; COMET 2001 (n=1,054): 0.1% bupivacaine labor epidural spares motor
    function (89% no hip-flexion weakness at 30 min)."""
    from anasim.core.runtime import step_simulation

    epi = engine_factory(
        config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        start=True,
        age=35,
    )
    t0 = epi.state.temp_c
    epi.give_epidural_bolus("bupivacaine", 12.0, 0.5)
    for _ in range(3600):
        step_simulation(epi, 1.0, update_monitors=False)
        epi.state.time += 1.0
    assert 0.45 <= t0 - epi.state.temp_c <= 1.15

    # Ozaki 1994: neuraxial block lowers shivering threshold and attenuates shivering at 35.0 °C.
    assert epi.neuraxial.shiver_threshold_shift_c >= 0.30
    shiver_awake = epi.thermal.target_shivering(35.0, 97.0, 0.0, 1.0)
    shiver_blocked = epi.thermal.target_shivering(
        35.0, 97.0, 0.0, 1.0,
        neuraxial_shiver_shift_c=epi.neuraxial.shiver_threshold_shift_c,
        neuraxial_muscle_keep=epi.neuraxial.shiver_muscle_keep,
    )
    assert shiver_awake > 0.20
    assert shiver_blocked < shiver_awake * 0.65

    # COMET 2001: dilute 0.1% labor epidural maintains thoracic analgesia without dense motor block.
    labor = engine_factory(
        config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1),
        start=True,
        age=30, weight=75, height=165, sex="female", gestational_age_weeks=39.0,
    )
    labor.give_epidural_bolus("bupivacaine", 15.0, 0.1)
    labor.set_epidural_infusion(10.0, drug="bupivacaine", concentration_pct=0.1)
    for _ in range(1800):
        step_simulation(labor, 1.0, update_monitors=False)
        labor.state.time += 1.0
    assert labor.neuraxial.sensory_level >= 13.0  # >= T10
    assert labor.neuraxial.bromage_score == 0


def test_marti_2015_thrombolysis_bleeding_and_mirski_2007_vae_aspiration_cap(engine_factory):
    """Marti 2015 (15 RCTs, n=2,057, major bleeding OR 2.91): PE thrombolysis increases active
    surgical hemorrhage rate;
    Mirski 2007: CVC air aspiration removes ~45% of right-atrial air and cannot clear distal
    pulmonary microvascular air via rapid repeat clicks."""
    from anasim.core.runtime import step_simulation

    base = engine_factory(config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1), start=True)
    lysed = engine_factory(config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1), start=True)
    lysed.start_pulmonary_embolism(0.45)
    lysed.give_thrombolysis()
    for _ in range(60):
        step_simulation(lysed, 1.0, update_monitors=False)
        lysed.state.time += 1.0
    assert lysed.embolism.thrombolysis_bleeding_multiplier > 1.40

    base.start_hemorrhage(300.0)
    lysed.start_hemorrhage(300.0)
    for _ in range(60):
        step_simulation(base, 1.0, update_monitors=False)
        step_simulation(lysed, 1.0, update_monitors=False)
    assert lysed.state.blood_out_ml > base.state.blood_out_ml * 1.35

    # Rapid repeat VAE aspirations cannot evacuate distal pulmonary air.
    vae = engine_factory(config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1), start=True)
    vae.start_venous_air_embolism(rate_ml_min=0.0, initial_bolus_ml=100.0)
    vae.stop_venous_air_embolism()
    first = vae.aspirate_vae_air()
    second = vae.aspirate_vae_air()
    assert 40.0 <= first <= 50.0
    assert second == 0.0
    assert vae.state.vae_air_ml >= 50.0


def test_thwaites_1997_and_kirkbride_2001_propofol_vs_sevoflurane_induction(engine_factory):
    """Thwaites 1997 (n=102, age ~59, 66.7% N2O): 8% sevoflurane induction dropped MAP less than
    propofol 167 mg over 57 s (-8.5% vs -20.3% at 2 min; -7.7% vs -21.8% at 3 min).
    Kirkbride 2001 (n=45 elderly >75 yr, 50% N2O): 6-min MAP drop was 34 +/- 7% (SD) with propofol
    versus 21 +/- 6% with 8% sevoflurane."""
    from anasim.core.runtime import step_simulation

    def _run_induction(*, age: float, weight: float, n2o_l: float, o2_l: float, arm: str, duration_s: int) -> float:
        eng = engine_factory(
            config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
            start=True, age=age, weight=weight, height=170.0, sex="female",
            baseline_map=105.0, baseline_hr=76.0,
        )
        map0 = eng.state.map
        eng.set_airway_mode("ETT")
        eng.set_vent_settings(rr=12, vt=0.5, peep=0.0, ie="1:2", mode="VCV")
        eng.set_vent_power(True)
        eng.set_fgf(o2_l, 0.0, n2o_l)
        if arm == "propofol":
            bolus = 167.0 if age < 70 else 180.0
            for sec in range(duration_s):
                if sec < 60:
                    eng.give_drug_bolus("propofol", bolus / 60.0)
                elif sec == 60:
                    eng.set_vaporizer(2.0 if age < 70 else 1.5)
                step_simulation(eng, 1.0, update_monitors=False)
                eng.state.time += 1.0
        else:
            eng.set_vaporizer(8.0)
            for sec in range(duration_s):
                if sec == 120:
                    eng.set_vaporizer(2.0 if age < 70 else 1.5)
                step_simulation(eng, 1.0, update_monitors=False)
                eng.state.time += 1.0
        return (map0 - eng.state.map) / map0

    thw_prop = _run_induction(age=59.0, weight=73.0, n2o_l=4.0, o2_l=2.0, arm="propofol", duration_s=180)
    thw_sevo = _run_induction(age=59.0, weight=73.0, n2o_l=4.0, o2_l=2.0, arm="sevo", duration_s=180)
    assert 0.16 <= thw_prop <= 0.36
    assert 0.04 <= thw_sevo <= 0.22
    assert thw_prop - thw_sevo >= 0.08

    kirk_prop = _run_induction(age=77.0, weight=70.0, n2o_l=3.0, o2_l=3.0, arm="propofol", duration_s=360)
    kirk_sevo = _run_induction(age=77.0, weight=70.0, n2o_l=3.0, o2_l=3.0, arm="sevo", duration_s=360)
    assert 0.24 <= kirk_prop <= 0.45
    assert 0.10 <= kirk_sevo <= 0.30
    assert kirk_prop - kirk_sevo >= 0.08


def test_albertin_2004_and_manyam_2006_sevoflurane_remifentanil_synergy():
    """Manyam 2006 (n=16 volunteers, no N2O): remifentanil 0 -> 1.25 -> 5.0 -> 7.5 ng/mL lowers
    sevoflurane EC95 for OAA/S <= 1 (1.30 -> 0.78 -> 0.33 -> 0.23%) and tetany (6.48 -> 2.63 -> 0.90 -> 0.61%).
    Albertin 2004 (n=74 ASA I, 60% N2O): remifentanil 0 -> 1 -> 3 ng/mL lowers sevoflurane MAC-BAR
    from 2.8% [2.5-3.0] to 1.1% [0.9-1.3] and 0.2% [0.1-0.3]."""
    from anasim.patient.pd.anesthesia import ClinicalResponseModel
    from anasim.patient.volatile_pk import VolatilePK
    from anasim.physiology.disturbances import PROFILE_SPECS

    young = Patient(age=24.0, weight=70.0, height=175.0, sex="male")
    mac_young = VolatilePK(young, "Sevoflurane", 0.65, 1.7, 2.05).mac_age
    pd_young = ClinicalResponseModel(young.age)

    def _find_sevo_ec_pct(fn, target: float, remi: float) -> float:
        lo, hi = 0.0, 12.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if fn(0.0, remi, mid / mac_young, 0.0) >= target:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    loc95 = [_find_sevo_ec_pct(pd_young.loss_of_response, 0.95, r) for r in (0.0, 1.25, 5.0, 7.5)]
    tol95 = [_find_sevo_ec_pct(pd_young.tolerance, 0.95, r) for r in (0.0, 1.25, 5.0, 7.5)]
    assert loc95[0] > loc95[1] > loc95[2] > loc95[3]
    assert tol95[0] > tol95[1] > tol95[2] > tol95[3]
    assert 1.0 <= loc95[0] <= 2.2
    assert 0.10 <= loc95[3] <= 0.45
    assert tol95[0] > 2.5 and tol95[3] < 1.2

    # Albertin 2004: remifentanil progressively blunts incision cardiovascular surge at fixed sevo + 60% N2O.
    adult = Patient(age=37.0, weight=61.0, height=167.0, sex="female")
    mac_adult = VolatilePK(adult, "Sevoflurane", 0.65, 1.7, 2.05).mac_age
    pd_adult = ClinicalResponseModel(adult.age)
    inc_intensity = PROFILE_SPECS["stim_sustained_surgery"].sympathetic_intensity
    mac_n2o = 60.0 / 104.0
    fracs_fixed = [
        pd_adult.cardiovascular_response_fractions(
            0.0, remi_ce, 1.1 / mac_adult, mac_n2o, intensity=inc_intensity,
        )[0]
        for remi_ce in (0.0, 1.0, 3.0)
    ]
    assert fracs_fixed[0] > fracs_fixed[1] > fracs_fixed[2]
    assert fracs_fixed[2] < 0.20
    # At Albertin's no-remi and 1 ng/mL MAC-BAR points, acute cardiovascular surge is attenuated.
    for sevo_pct, remi_ce in ((2.8, 0.0), (1.1, 1.0)):
        map_frac, hr_frac = pd_adult.cardiovascular_response_fractions(
            0.0, remi_ce, sevo_pct / mac_adult, mac_n2o, intensity=inc_intensity,
        )
        assert map_frac < 0.70 and hr_frac < 0.70


def test_mertens_2003_propofol_remifentanil_synergy():
    """Mertens 2003 (n=29 ASA I-II women, age 37): remifentanil 0 -> 7 -> 10 ng/mL lowers awakening
    C50,prop from 3.49 +/- 0.64 (SE) to 0.6 and 0.4 mcg/mL; propofol 2 -> 7.3 mcg/mL lowers
    laryngoscopy C50,rem from 3.8 to 0 ng/mL."""
    from anasim.patient.pd.anesthesia import ClinicalResponseModel

    pd = ClinicalResponseModel(37.0)

    def _c50_prop(remi: float) -> float:
        lo, hi = 0.0, 12.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if pd.loss_of_response(mid, remi, 0.0, 0.0) >= 0.5:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def _c50_remi(prop: float) -> float:
        if pd.tolerance(prop, 0.0, 0.0, 0.0) >= 0.5:
            return 0.0
        lo, hi = 0.0, 20.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if pd.tolerance(prop, mid, 0.0, 0.0) >= 0.5:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    c50p = [_c50_prop(r) for r in (0.0, 7.0, 10.0)]
    assert 2.2 <= c50p[0] <= 4.2
    assert c50p[0] > c50p[1] > c50p[2]
    assert c50p[1] < 1.0 and c50p[2] < 0.6

    c50r_2 = _c50_remi(2.0)
    c50r_4 = _c50_remi(4.0)
    c50r_73 = _c50_remi(7.3)
    assert 1.2 <= c50r_2 <= 6.5
    assert c50r_2 > c50r_4 > c50r_73
    assert c50r_73 < 0.5


def test_kalmar_2018_and_poterman_2015_vasopressor_preload_and_afterload(engine_factory):
    """Kalmar 2018 (n=18, preload-dependent atropinized TIVA + epidural, 2 mcg/kg phenylephrine):
    raised MAP +34 mmHg, CO +26%, SV +28%, SVR +37% with flat HR.
    Poterman 2015 (n=57, propofol Ce 2.5 + remi Ce 8): phenylephrine 100 mcg + 0.49 mcg/kg/min and
    norepinephrine 10 mcg + 0.05 mcg/kg/min both restored MAP (+13 mmHg) via increased SVR."""
    from anasim.core.runtime import step_simulation

    # Kalmar 2018: pre-incision preload-dependent, vagolytically blocked patient given 2 mcg/kg phenylephrine.
    kal = engine_factory(
        config=SimulationConfig(
            mode="awake", dt=1.0, rng_seed=1, tci_enabled=True, maintenance_fluid_ml_hr=0.0,
        ),
        start=True, age=62.0, weight=74.0, height=172.0, sex="male",
    )
    kal.set_airway_mode("ETT")
    kal.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    kal.set_vent_power(True)
    kal.set_fgf(2.0, 2.0, 0.0)
    kal.set_drug_target("propofol", 3.0)
    kal.set_drug_target("remi", 4.0)
    kal.give_epidural_bolus("bupivacaine", 10.0, 0.5)
    kal.hemo.add_volume(-650.0)  # Fasted/bowel-prep preload dependence before surgical incision
    kal.give_drug_bolus("atropine", 0.5)  # Prophylactic vagolytic (methylatropine 0.5 mg)
    for _ in range(900):
        step_simulation(kal, 1.0, update_monitors=False)
        kal.state.time += 1.0
    m0, hr0, co0, sv0, svr0 = kal.state.map, kal.state.hr, kal.state.co, kal.state.sv, kal.state.svr
    kal.give_drug_bolus("phenyl", 2.0 * 74.0)
    peak_map, peak_co, peak_sv, peak_svr, min_hr = m0, co0, sv0, svr0, hr0
    for _ in range(300):
        step_simulation(kal, 1.0, update_monitors=False)
        kal.state.time += 1.0
        peak_map = max(peak_map, kal.state.map)
        peak_co = max(peak_co, kal.state.co)
        peak_sv = max(peak_sv, kal.state.sv)
        peak_svr = max(peak_svr, kal.state.svr)
        min_hr = min(min_hr, kal.state.hr)
    assert 20.0 <= peak_map - m0 <= 65.0
    assert (peak_co - co0) / co0 >= 0.12
    assert (peak_sv - sv0) / sv0 >= 0.12
    assert (peak_svr - svr0) / svr0 >= 0.20
    assert hr0 - min_hr <= 8.0

    # Poterman 2015: phenylephrine vs norepinephrine infusion under propofol/remifentanil.
    pot_results = {}
    for arm in ("phenyl", "nore"):
        eng = engine_factory(
            config=SimulationConfig(
                mode="steady_state", maint_type="tiva", dt=1.0, rng_seed=1,
                tci_enabled=True, maintenance_fluid_ml_hr=0.0,
            ),
            start=True, age=58.0, weight=81.0, height=175.0, sex="male",
        )
        eng.set_drug_target("propofol", 2.5)
        eng.set_drug_target("remi", 8.0)
        for _ in range(300):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
        pm0, phr0, psvr0 = eng.state.map, eng.state.hr, eng.state.svr
        if arm == "phenyl":
            eng.give_drug_bolus("phenyl", 100.0)
            eng.set_drug_rate("phenyl", 0.49 * 81.0)
        else:
            eng.give_drug_bolus("nore", 10.0)
            eng.set_drug_rate("nore", 0.05 * 81.0)
        for _ in range(240):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
        pot_results[arm] = (eng.state.map - pm0, eng.state.hr - phr0, eng.state.svr / psvr0)

    assert pot_results["phenyl"][0] >= 8.0 and pot_results["phenyl"][2] > 1.15
    assert pot_results["nore"][0] >= 8.0 and pot_results["nore"][2] > 1.04
    assert pot_results["phenyl"][1] < pot_results["nore"][1]


def test_su_2023_and_sjoen_2026_propofol_remifentanil_and_vasopressor_boluses(engine_factory):
    """Su 2023 (n=36 volunteers): stepped propofol 0 -> 1.5 -> 3.5 -> 6.0 mcg/mL monotonically
    lowers MAP, remifentanil deepens the drop at 1.5-3.5 mcg/mL, and age 60 drops more than 22 yr.
    Sjøen 2026 (n=122 women, propofol 6 / remi 8 TCI + simultaneous bolus): ephedrine 0.1 mg/kg
    attenuated SBP (-20% vs -32%) and CO (-22% vs -34%) falls, whereas phenylephrine 1 mcg/kg
    raised SVR (+46% vs +5.6%) with deeper HR slowing."""
    from anasim.core.runtime import step_simulation

    # Su 2023 steady-concentration MAP across propofol steps and remifentanil levels.
    def _steady_map(age: float, prop: float, remi: float) -> float:
        hemo = HemodynamicModel(
            Patient(age=age, weight=72.0, height=174.0, sex="male", baseline_map=82.0, baseline_hr=65.0),
        )
        for _ in range(360):
            hemo.step(5.0, prop, remi, 0.0, -2.0, 40.0, 98.0)
        return hemo.state.map

    m_alone = [_steady_map(35.0, c, 0.0) for c in (0.0, 1.5, 3.5, 6.0)]
    assert m_alone[0] > m_alone[1] > m_alone[2] > m_alone[3]
    assert _steady_map(35.0, 1.5, 4.0) < m_alone[1] - 3.0
    assert _steady_map(35.0, 3.5, 4.0) < m_alone[2] - 2.0
    assert _steady_map(60.0, 3.5, 2.0) < _steady_map(22.0, 3.5, 2.0)

    # Sjøen 2026: simultaneous prophylactic bolus during propofol 6 / remi 8 TCI induction.
    arms = {}
    for arm in ("placebo", "ephedrine", "phenyl"):
        eng = engine_factory(
            config=SimulationConfig(
                mode="awake", dt=1.0, rng_seed=1, tci_enabled=True, maintenance_fluid_ml_hr=0.0,
            ),
            start=True, age=40.0, weight=71.0, height=166.0, sex="female",
            baseline_map=88.0, baseline_hr=72.0,
        )
        eng.set_airway_mode("ETT")
        eng.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        eng.set_vent_power(True)
        eng.set_fgf(2.0, 2.0, 0.0)
        sbp0, co0, hr0, svr0 = eng.state.sbp, eng.state.co, eng.state.hr, eng.state.svr
        eng.set_drug_target("propofol", 6.0)
        eng.set_drug_target("remi", 8.0)
        if arm == "ephedrine":
            eng.give_drug_bolus("ephedrine", 0.1 * 71.0)
        elif arm == "phenyl":
            eng.give_drug_bolus("phenyl", 1.0 * 71.0)
        min_sbp, min_co, min_hr, max_svr = sbp0, co0, hr0, svr0
        for _ in range(300):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
            min_sbp = min(min_sbp, eng.state.sbp)
            min_co = min(min_co, eng.state.co)
            min_hr = min(min_hr, eng.state.hr)
            max_svr = max(max_svr, eng.state.svr)
        arms[arm] = (
            (min_sbp - sbp0) / sbp0,
            (min_co - co0) / co0,
            (min_hr - hr0) / hr0,
            (max_svr - svr0) / svr0,
        )

    assert arms["ephedrine"][0] > arms["placebo"][0] + 0.06
    assert arms["ephedrine"][1] > arms["placebo"][1] + 0.05
    assert arms["phenyl"][3] > arms["placebo"][3] + 0.20
    assert arms["phenyl"][2] < arms["ephedrine"][2]


def test_reich_2005_and_schonberger_2022_elderly_propofol_vs_etomidate(engine_factory):
    """Reich 2005 (n=4,096): severe post-induction hypotension rises with age >= 50 (OR 2.25)
    and propofol vs etomidate (OR 3.94).
    Schonberger 2022 (N=319,948 >= 65 yr, mean age 73): MAP fell 31.0 +/- 17.9 mmHg (NIBP) and
    37.0 +/- 20.8 mmHg (IBP), deepened per +1 mg/kg propofol, and was attenuated by etomidate."""
    from anasim.core.runtime import step_simulation

    def _run_case(*, age: float, agent: str, dose_mg_kg: float) -> tuple[float, float]:
        eng = engine_factory(
            config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
            start=True, age=age, weight=75.0, height=170.0, sex="male",
            baseline_map=95.0, baseline_hr=72.0,
        )
        eng.set_airway_mode("ETT")
        eng.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        eng.set_vent_power(True)
        eng.set_fgf(2.0, 2.0, 0.0)
        m0 = eng.state.map
        eng.give_drug_bolus("fentanyl", 1.5 * 75.0)
        eng.give_drug_bolus(agent, dose_mg_kg * 75.0)
        min_0_5 = m0
        min_5_10 = m0
        for sec in range(1, 601):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
            if sec <= 300:
                min_0_5 = min(min_0_5, eng.state.map)
            else:
                min_5_10 = min(min_5_10, eng.state.map)
        return m0 - min_0_5, m0 - min_5_10

    eld_17_early, eld_17_late = _run_case(age=73.0, agent="propofol", dose_mg_kg=1.7)
    _, eld_25_late = _run_case(age=73.0, agent="propofol", dose_mg_kg=2.5)
    _, yng_17_late = _run_case(age=35.0, agent="propofol", dose_mg_kg=1.7)
    _, eld_eto_late = _run_case(age=73.0, agent="etomidate", dose_mg_kg=0.25)

    assert 20.0 <= eld_17_late <= 45.0
    assert eld_17_late >= eld_17_early
    assert eld_25_late > eld_17_late + 2.0
    assert eld_17_late > yng_17_late + 4.0
    assert eld_eto_late < 15.0 and eld_eto_late < 0.45 * eld_17_late


def test_joris_1993_and_tan_1992_pneumoperitoneum_hemodynamics_and_gas_exchange(engine_factory):
    """Joris 1993 (n=15, 14 mmHg CO2 pneumoperitoneum): MAP rises, CI falls ~15-25%, SVR and PVR
    climb, RAP and PCWP/LAP rise with intrathoracic pressure, and airway pressures increase;
    Fahy 1995: respiratory compliance falls ~25-35%; Tan 1992 / Mullett 1993: peritoneal CO2
    absorption raises venous CO2 load ~30% without raising O2 consumption; VitalDB labs:
    PaCO2-EtCO2 widens ~2 mmHg; over-insufflation (22 mmHg) compresses the IVC and lowers CO."""
    from anasim.core.runtime import step_simulation

    eng = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=1.0, rng_seed=1),
        start=True,
        age=49.0, weight=67.0, height=165.0, sex="female",
    )
    eng.set_vent_settings(mode="VCV", vt=0.50, rr=12.0, peep=5.0, ie="1:2")
    for _ in range(300):
        step_simulation(eng, 1.0, update_monitors=False)
        eng.state.time += 1.0

    base_map = eng.state.map
    base_ci = eng.state.co / eng.patient.bsa
    base_svr = eng.state.svr
    base_pvr = eng.hemo.state.pvr
    base_rap = eng.hemo.state.rap
    base_lap = eng.hemo.state.lap
    base_comp = eng.aeration.static_compliance
    base_pplat = eng.state.paw_plat
    base_grad = eng.state.pa_co2 - eng.state.etco2
    base_vo2 = eng.resp._pulmonary_o2_uptake

    eng.start_pneumoperitoneum(14.0)
    for _ in range(300):
        step_simulation(eng, 1.0, update_monitors=False)
        eng.state.time += 1.0
    m5_map, m5_ci = eng.state.map, eng.state.co / eng.patient.bsa
    m5_svr, m5_pvr = eng.state.svr, eng.hemo.state.pvr
    m5_rap, m5_lap = eng.hemo.state.rap, eng.hemo.state.lap
    assert m5_map > base_map + 8.0
    assert 0.70 * base_ci <= m5_ci <= 0.90 * base_ci
    assert m5_svr > base_svr * 1.30
    assert m5_pvr > base_pvr * 1.45
    assert m5_rap > base_rap + 3.5
    assert m5_lap > base_lap + 1.0
    assert 0.62 * base_comp <= eng.aeration.static_compliance <= 0.80 * base_comp
    assert eng.state.paw_plat > base_pplat + 3.0
    assert (eng.state.pa_co2 - eng.state.etco2) > base_grad + 1.5

    # Sustained CO2 absorption over 25 more min raises PaCO2 without raising metabolic O2 demand.
    paco2_5 = eng.state.pa_co2
    for _ in range(1500):
        step_simulation(eng, 1.0, update_monitors=False)
        eng.state.time += 1.0
    assert eng.state.pa_co2 > paco2_5 + 3.0
    assert eng.pneumoperitoneum.co2_absorption_ml_min(eng.resp.vco2) > 40.0
    assert eng.resp._pulmonary_o2_uptake <= base_vo2 * 1.05

    # Over-insufflation (22 mmHg) causes IVC compression and marked cardiac output fall.
    eng.start_pneumoperitoneum(22.0)
    for _ in range(240):
        step_simulation(eng, 1.0, update_monitors=False)
        eng.state.time += 1.0
    assert eng.state.co / eng.patient.bsa < m5_ci * 0.82

    # Desufflation restores compliance, pressures, and CI while tissue CO2 washes out gradually.
    eng.stop_pneumoperitoneum()
    for _ in range(900):
        step_simulation(eng, 1.0, update_monitors=False)
        eng.state.time += 1.0
    assert eng.state.pneumoperitoneum_iap == pytest.approx(0.0, abs=0.1)
    assert eng.aeration.static_compliance == pytest.approx(base_comp, rel=0.05)
    assert eng.state.co / eng.patient.bsa > m5_ci * 1.15


def test_cronnelly_1979_kirkegaard_2002_kaufhold_2016_and_mirakhur_1977_neostigmine(engine_factory):
    """Cronnelly 1979 (n=18): ~53% renal clearance doubles neostigmine half-life in anephric patients.
    Jones 2008 / Kirkegaard 2002 / Kaufhold 2016 / Schaller 2010: competitive reversal shows a deep-block
    ceiling, saturates near 0.04-0.07 mg/kg, and reverses shallow/minimal block within minutes.
    Mirakhur 1977 (n=40): unopposed neostigmine causes severe muscarinic bradycardia; glycopyrrolate
    (0.2 mg/mg) stabilizes HR better than atropine's brief 1-min tachycardia and later dip.
    Herbstreit 2010 (n=10): unopposed neostigmine after full recovery impairs pharyngeal dilator tone."""
    from anasim.core.runtime import step_simulation
    from anasim.patient.pk_models import NeostigminePK

    # Cronnelly 1979: renal failure roughly halves clearance and prolongs elimination.
    normal_pk = NeostigminePK(Patient(age=40.0, weight=70.0, height=170.0, sex="male", renal_function=1.0))
    anephric_pk = NeostigminePK(Patient(age=40.0, weight=70.0, height=170.0, sex="male", renal_function=0.05))
    assert anephric_pk.cl1 < 0.55 * normal_pk.cl1
    for pk in (normal_pk, anephric_pk):
        pk.state.c1 += 4900.0 / pk.v1
        for _ in range(7200):
            pk.step(1.0, 0.0)
    assert anephric_pk.state.c1 > normal_pk.state.c1 * 1.5

    def _time_to_tof90(maint: str, wait_min: int, dose_mg_kg: float) -> float | None:
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type=maint, dt=1.0, rng_seed=1),
            start=True, age=40.0, weight=70.0, height=170.0, sex="male",
        )
        eng.give_drug_bolus("roc", 0.6 * 70.0)
        for _ in range(wait_min * 60):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
        if dose_mg_kg > 0.0:
            eng.give_drug_bolus("neostigmine", dose_mg_kg * 70.0)
            eng.give_drug_bolus("glyco", 0.2 * dose_mg_kg * 70.0)
        for sec in range(1, 35 * 60 + 1):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
            if eng.state.tof >= 90.0:
                return sec / 60.0
        return None

    # Deep block ceiling (Jones 2008) vs shallow/minimal block reversal (Kaufhold 2016; Schaller 2010).
    deep_t = _time_to_tof90("balanced", 20, 0.07)
    shallow_placebo = _time_to_tof90("tiva", 39, 0.0)
    shallow_40 = _time_to_tof90("tiva", 39, 0.04)
    shallow_70 = _time_to_tof90("tiva", 39, 0.07)
    minimal_40 = _time_to_tof90("tiva", 47, 0.04)
    assert deep_t is None or deep_t > 25.0
    assert shallow_placebo is not None and shallow_40 is not None and shallow_70 is not None
    assert minimal_40 is not None
    assert shallow_40 < shallow_placebo * 0.50
    assert shallow_70 <= shallow_40
    assert minimal_40 < 5.0

    # Mirakhur 1977: unopposed bradycardia vs glycopyrrolate stability vs atropine early spike.
    hr_traces: dict[str, list[float]] = {}
    for label, adj, adj_mg in (("alone", None, 0.0), ("glyco", "glyco", 0.5), ("atropine", "atropine", 1.2)):
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=1.0, rng_seed=1),
            start=True, age=53.0, weight=65.0, height=168.0, sex="female", baseline_hr=87.0,
        )
        for _ in range(120):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
        hr0 = eng.state.hr
        eng.give_drug_bolus("neostigmine", 2.5)
        if adj is not None:
            eng.give_drug_bolus(adj, adj_mg)
        deltas = []
        for _ in range(7):
            for _ in range(60):
                step_simulation(eng, 1.0, update_monitors=False)
                eng.state.time += 1.0
            deltas.append(eng.state.hr - hr0)
        hr_traces[label] = deltas

    assert min(hr_traces["alone"]) < -15.0
    assert abs(hr_traces["glyco"][0]) < abs(hr_traces["atropine"][0])
    assert hr_traces["atropine"][0] > 3.0 and hr_traces["atropine"][-1] < 0.0
    assert max(abs(x) for x in hr_traces["glyco"]) < max(abs(x) for x in hr_traces["atropine"])

    # Standard 5:1 neostigmine:glycopyrrolate balance holds across higher clinical reversal doses (3.5/0.7 and 5/1 mg).
    for neo_mg, glyco_mg in ((3.5, 0.7), (5.0, 1.0)):
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=1.0, rng_seed=1),
            start=True, age=53.0, weight=75.0, height=170.0, sex="male", baseline_hr=80.0,
        )
        eng.infusion_rates["remi"] = 0.0
        eng.pk["remi"].state.c1 = eng.pk["remi"].state.c2 = eng.pk["remi"].state.c3 = eng.pk["remi"].state.ce = 0.0
        for _ in range(120):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
        hr0 = eng.state.hr
        eng.give_drug_bolus("neostigmine", neo_mg)
        eng.give_drug_bolus("glyco", glyco_mg)
        late = []
        for minute in range(1, 8):
            for _ in range(60):
                step_simulation(eng, 1.0, update_monitors=False)
                eng.state.time += 1.0
            if minute >= 4:
                late.append(eng.state.hr - hr0)
        assert -6.0 < min(late) and max(late) < 6.0

    # Mirakhur 1981 (n=120): IV glycopyrrolate 5, 10, and 15 mcg/kg produce a graded
    # HR rise (+8, +17, +24 bpm at 6 min) rather than saturating at low doses.
    glyco_rises = []
    for dose_ug_kg, wt, hr0_base in ((5.0, 60.0, 86.0), (10.0, 63.0, 82.0), (15.0, 73.0, 78.0)):
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=1.0, rng_seed=1),
            start=True, age=35.0, weight=wt, height=168.0, sex="female", baseline_hr=hr0_base,
        )
        eng.infusion_rates["remi"] = 0.0
        eng.pk["remi"].state.c1 = eng.pk["remi"].state.c2 = eng.pk["remi"].state.c3 = eng.pk["remi"].state.ce = 0.0
        eng.set_vaporizer(1.2)
        for _ in range(120):
            step_simulation(eng, 2.0, update_monitors=False)
            eng.state.time += 2.0
        h0 = eng.state.hr
        eng.give_drug_bolus("glyco", dose_ug_kg * wt / 1000.0)
        for _ in range(360):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
        glyco_rises.append(eng.state.hr - h0)
    assert 5.0 < glyco_rises[0] < 16.0
    assert glyco_rises[0] + 4.0 < glyco_rises[1] < 24.0
    assert glyco_rises[1] + 2.0 < glyco_rises[2] < 30.0

    # Herbstreit 2010: unopposed neostigmine after full recovery reduces pharyngeal dilator tone.
    awake = engine_factory(
        config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1),
        start=True, age=35.0, weight=75.0, height=178.0, sex="male",
    )
    tone0 = awake.tof_pd.pharyngeal_tone_fraction()
    awake.give_drug_bolus("neostigmine", 0.04 * 75.0)
    awake.give_drug_bolus("glyco", 0.008 * 75.0)
    for _ in range(480):
        step_simulation(awake, 1.0, update_monitors=False)
        awake.state.time += 1.0
    assert awake.state.tof >= 90.0
    assert awake.tof_pd.pharyngeal_tone_fraction() < tone0 - 0.15


def test_poterman_2015_and_de_keijzer_2026_vasopressor_preload_reserve(engine_factory):
    """Preload-dependent norepinephrine response: unpreloaded hypotensive TIVA (de Keijzer 2026)
    shows stronger CO restoration than colloid-preloaded TIVA (Poterman 2015), where SVR rises
    more prominently and HR slows slightly."""
    from anasim.core.runtime import step_simulation

    def _run_ne(*, preload_ml: float) -> tuple[float, float, float, float]:
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="tiva", dt=1.0, rng_seed=1),
            start=True, age=44.0, weight=79.0, height=176.0, sex="male",
        )
        if preload_ml > 0.0:
            eng.give_albumin(preload_ml)
        for _ in range(900):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
        b_map, b_hr, b_co, b_svr = eng.state.map, eng.state.hr, eng.state.co, eng.state.svr
        eng.set_drug_rate("nore", 0.05 * 79.0)
        for _ in range(600):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
        return eng.state.map - b_map, eng.state.hr - b_hr, eng.state.co - b_co, eng.state.svr / b_svr

    u_dmap, _u_dhr, u_dco, _u_svr = _run_ne(preload_ml=0.0)
    p_dmap, p_dhr, p_dco, p_svr = _run_ne(preload_ml=500.0)
    assert u_dmap > 8.0 and p_dmap > 8.0
    assert u_dco > p_dco + 0.10
    assert p_dhr <= 1.0
    assert p_svr > 1.12


def test_kanaya_2002_propofol_enhances_ephedrine_over_sevoflurane(engine_factory):
    """Kanaya 2002 (n=30, 0.1 mg/kg IV ephedrine): propofol anesthesia enhances the
    pressor response compared with awake and 2% sevoflurane states."""
    from anasim.core.runtime import step_simulation

    rises = {}
    for arm in ("awake", "sevo", "propofol"):
        mode = "awake" if arm == "awake" else "steady_state"
        maint = "balanced" if arm == "sevo" else "tiva"
        eng = engine_factory(
            config=SimulationConfig(mode=mode, maint_type=maint, dt=1.0, rng_seed=1),
            start=True, age=42.0, weight=58.0, height=162.0, sex="female",
            baseline_map=85.0, baseline_hr=68.0,
        )
        if arm != "awake":
            eng.infusion_rates["remi"] = 0.0
            r = eng.pk["remi"]
            r.state.c1 = r.state.c2 = r.state.c3 = r.state.ce = 0.0
            if arm == "sevo":
                eng.set_vaporizer(2.3)
            else:
                eng.infusion_rates["propofol"] = 0.100 * 58.0 / 60.0
            for _ in range(180):
                step_simulation(eng, 2.0, update_monitors=False)
                eng.state.time += 2.0
        m0 = eng.state.map
        eng.give_drug_bolus("ephedrine", 0.1 * 58.0)
        peak = m0
        for _ in range(300):
            step_simulation(eng, 1.0, update_monitors=False)
            eng.state.time += 1.0
            peak = max(peak, eng.state.map)
        rises[arm] = peak - m0

    assert rises["propofol"] > rises["sevo"] + 3.0
    assert rises["propofol"] > rises["awake"] + 3.0
