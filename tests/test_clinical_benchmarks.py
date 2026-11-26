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
