"""Tests for heparin and protamine PK/PD, ACT, heparin rebound, and protamine reactions."""

import pytest

from anasim.core.clinical_actions import (
    check_drug_administration,
    check_extracorporeal_start,
)
from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient


def _step_for(engine: SimulationEngine, seconds: float, dt: float = 0.5) -> None:
    steps = round(seconds / dt)
    for _ in range(steps):
        engine.step(dt)


def _steady_engine(*, dt: float = 0.5, seed: int = 11) -> SimulationEngine:
    patient = Patient(age=60, weight=70, height=172, sex="male", baseline_hb=13.5)
    config = SimulationConfig(
        mode="steady_state", dt=dt, rng_seed=seed, arterial_line_enabled=True,
    )
    eng = SimulationEngine(patient, config)
    eng.start()
    return eng


def test_bull_1975_heparin_act_dose_response_and_cpb_hemodilution():
    """Bull 1975: baseline ACT ~120 s, 100 U/kg -> ~250-300 s, 300 U/kg -> >=480 s."""
    eng = _steady_engine()
    assert eng.state.act_s == pytest.approx(120.0, abs=1.0)
    assert eng.coagulation.state.bleeding_multiplier == 1.0
    assert eng.coagulation.state.protamine_svr_factor == 1.0
    assert eng.coagulation.state.protamine_pvr_factor == 1.0

    # 100 U/kg (7,000 U): vascular / ECMO range
    eng.give_drug_bolus("heparin", 7000.0)
    _step_for(eng, 120.0)
    act_100 = eng.state.act_s
    assert 220.0 <= act_100 <= 340.0

    # Additional 200 U/kg (total 300 U/kg): CPB therapeutic range (>=480 s)
    eng.give_drug_bolus("heparin", 14000.0)
    _step_for(eng, 120.0)
    act_300 = eng.state.act_s
    assert act_300 >= 480.0
    hep_pre_cpb = eng.state.heparin_u_ml

    # Point-of-care ACT measurement records drawn time and 60 s analysis latency
    sample = eng.measure_act()
    assert sample.act_s == pytest.approx(act_300, abs=1.0)
    assert sample.ready_at == pytest.approx(sample.drawn_at + 60.0)

    # CPB crystalloid prime expands plasma V1, diluting heparin concentration
    eng.start_cpb(blood_flow_l_min=4.5, prime_volume_ml=1500.0, prime_hematocrit=0.0)
    _step_for(eng, 30.0)
    assert eng.state.heparin_u_ml < hep_pre_cpb * 0.85


def test_bjornsson_1982_nonlinear_heparin_half_life():
    """Bjornsson 1982 (25-75 U/kg, extrapolated to 100/300 U/kg): heparin half-life lengthens markedly with dose."""
    def _time_to_half_c1(dose_u_kg: float) -> float:
        eng = _steady_engine(dt=2.0)
        eng.give_drug_bolus("heparin", dose_u_kg * eng.patient.weight)
        # Wait 10 min for initial central mixing, then measure apparent decay t1/2
        _step_for(eng, 600.0, dt=2.0)
        c0 = eng.state.heparin_u_ml
        elapsed_s = 0.0
        while elapsed_s < 18000.0 and eng.state.heparin_u_ml > 0.5 * c0:
            eng.step(2.0)
            elapsed_s += 2.0
        return elapsed_s / 60.0

    t_25 = _time_to_half_c1(25.0)
    t_100 = _time_to_half_c1(100.0)
    t_300 = _time_to_half_c1(300.0)
    assert 18.0 <= t_25 <= 45.0
    assert t_100 > t_25 + 15.0
    assert t_300 > t_100 + 30.0
    assert t_300 >= 95.0


def test_butterworth_2002_protamine_clearance_and_teoh_2004_heparin_rebound():
    """Butterworth 2002: rapid protamine disappearance (t1/2 ~4.5 min post-CPB);
    Teoh 2004: peripheral heparin redistribution produces heparin rebound after initial slow infusion reversal,
    abolished by low-dose continuous protamine infusion (25 mg/hr).
    """
    # 1. Rapid protamine clearance alone
    eng_p = _steady_engine(dt=1.0)
    eng_p.give_drug_bolus("protamine", 50.0)
    c0 = eng_p.state.protamine_ug_ml
    assert c0 > 7.0
    _step_for(eng_p, 270.0, dt=1.0)  # 4.5 min ~= 1 half-life
    assert 0.35 * c0 <= eng_p.state.protamine_ug_ml <= 0.65 * c0
    _step_for(eng_p, 930.0, dt=1.0)  # 20 min total -> nearly cleared
    assert eng_p.state.protamine_ug_ml < 0.08 * c0

    # 2. Heparin rebound after initial slow infusion reversal vs prevention with 25 mg/hr infusion
    def _run_reversal(post_infusion_mg_min: float) -> tuple[float, float]:
        eng = _steady_engine(dt=2.0)
        eng.give_drug_bolus("heparin", 300.0 * eng.patient.weight)
        # Allow 60 min for peripheral compartment loading (typical CPB duration)
        _step_for(eng, 3600.0, dt=2.0)
        # Neutralize remaining central heparin slowly (~150 mg over 10 min)
        eng.set_drug_rate("protamine", 15.0)
        _step_for(eng, 600.0, dt=2.0)
        eng.set_drug_rate("protamine", post_infusion_mg_min)
        _step_for(eng, 180.0, dt=2.0)
        act_post = eng.state.act_s
        # Observe 75 min post-reversal
        _step_for(eng, 4500.0, dt=2.0)
        return act_post, eng.state.act_s

    act_post_bolus, act_rebound = _run_reversal(0.0)
    act_post_inf, act_no_rebound = _run_reversal(25.0 / 60.0)  # 25 mg/hr (Teoh 2004)
    assert act_post_bolus <= 135.0
    assert act_rebound > act_post_bolus + 20.0
    assert act_post_inf <= 135.0
    assert act_no_rebound < act_rebound - 15.0
    assert act_no_rebound <= 135.0


def test_mochizuki_1998_excess_protamine_prolongs_act():
    """Mochizuki 1998 (in vitro): excess free protamine (>1.3:1) paradoxically prolongs ACT."""
    eng = _steady_engine(dt=0.5)
    eng.give_drug_bolus("heparin", 15000.0)  # 150 mg protamine equivalent
    _step_for(eng, 120.0)
    assert eng.state.act_s > 380.0

    # Appropriate slow neutralization (~150 mg over 10 min) normalizes ACT
    eng.set_drug_rate("protamine", 15.0)
    _step_for(eng, 600.0)
    eng.set_drug_rate("protamine", 0.0)
    _step_for(eng, 120.0)
    act_neutralized = eng.state.act_s
    assert act_neutralized <= 135.0

    # Excess protamine (additional 200 mg) prolongs ACT despite zero free heparin
    eng.give_drug_bolus("protamine", 200.0)
    _step_for(eng, 60.0)
    assert eng.state.heparin_u_ml < 0.05
    assert eng.state.protamine_ug_ml > 20.0
    assert eng.state.act_s > act_neutralized + 30.0


def test_horrow_1985_rate_dependent_type1_hypotension_vs_slow_infusion():
    """Horrow 1985: slow protamine infusion preserves MAP,
    whereas a rapid IV bolus causes acute Type I vasodilation and hypotension.
    """
    slow = _steady_engine(dt=0.5)
    fast = _steady_engine(dt=0.5)
    for eng in (slow, fast):
        eng.give_drug_bolus("heparin", 15000.0)
    _step_for(slow, 120.0)
    _step_for(fast, 120.0)
    map_base_slow = slow.state.map
    map_base_fast = fast.state.map

    # Slow infusion: 15 mg/min over 10 min (150 mg total)
    slow.set_drug_rate("protamine", 15.0)
    min_map_slow = map_base_slow
    for _ in range(1200):
        slow.step(0.5)
        min_map_slow = min(min_map_slow, slow.state.map)

    # Rapid IV bolus: 200 mg push
    fast.give_drug_bolus("protamine", 200.0)
    min_map_fast = map_base_fast
    min_svr_factor = 1.0
    for _ in range(240):
        fast.step(0.5)
        min_map_fast = min(min_map_fast, fast.state.map)
        min_svr_factor = min(min_svr_factor, fast.coagulation.state.protamine_svr_factor)

    assert (map_base_slow - min_map_slow) < 6.0
    assert min_svr_factor < 0.72
    assert (map_base_fast - min_map_fast) > 12.0

    # Recovers toward baseline over 6 minutes
    _step_for(fast, 360.0)
    assert fast.state.map > min_map_fast + 8.0


def test_lowenstein_1983_type3_catastrophic_pulmonary_vasoconstriction():
    """Lowenstein 1983: Type III protamine reaction triggers severe PVR surge,
    acute RV failure (high RAP, low CO, low EtCO2), and hypotension.
    """
    eng = _steady_engine(dt=0.5)
    eng.give_drug_bolus("heparin", 15000.0)
    _step_for(eng, 60.0)
    base_pvr = eng.hemo.state.pvr
    base_rap = eng.hemo.state.rap
    base_co = eng.state.co
    base_map = eng.state.map
    base_etco2 = eng.state.etco2

    # Arm reaction so it triggers automatically upon protamine delivery
    eng.arm_protamine_reaction(True)
    eng.set_drug_rate("protamine", 15.0)
    _step_for(eng, 120.0)

    assert eng.state.protamine_reaction is True
    assert eng.hemo.state.pvr > base_pvr * 3.0
    assert eng.hemo.state.rap > base_rap + 2.0
    assert eng.state.co < base_co * 0.75
    assert eng.state.map < base_map - 10.0
    assert eng.state.etco2 < base_etco2 - 3.0

    # Stopping the reaction restores PVR and cardiac output
    eng.set_drug_rate("protamine", 0.0)
    eng.stop_protamine_reaction()
    _step_for(eng, 480.0)
    assert eng.state.protamine_reaction is False
    assert eng.hemo.state.pvr < base_pvr * 1.4
    assert eng.state.co > base_co * 0.85


def test_anticoagulation_hemorrhage_and_clinical_safety_warnings():
    """High ACT amplifies surgical bleeding; safety checks warn on protamine during CPB/ECMO
    and on starting CPB with ACT < 400 s.
    """
    ctrl = _steady_engine(dt=0.5)
    hep = _steady_engine(dt=0.5)
    hep.give_drug_bolus("heparin", 300.0 * hep.patient.weight)
    _step_for(ctrl, 60.0)
    _step_for(hep, 60.0)

    # Low ACT CPB warning vs therapeutic ACT clearance
    warn_low = check_extracorporeal_start(ctrl, "cpb")
    assert warn_low is not None and warn_low.code == "cpb_low_act"
    assert check_extracorporeal_start(hep, "cpb") is None

    # Protamine during active CPB, VV ECMO, or VA ECMO triggers warning
    hep.start_cpb(blood_flow_l_min=4.5, prime_volume_ml=0.0)
    warn_prot = check_drug_administration(hep, "protamine")
    assert warn_prot is not None and warn_prot.code == "protamine_on_bypass"
    hep.start_vv_ecmo(blood_flow_l_min=4.0, prime_volume_ml=0.0)
    warn_vv = check_drug_administration(hep, "protamine")
    assert warn_vv is not None and warn_vv.code == "protamine_on_bypass"
    hep.start_va_ecmo(blood_flow_l_min=4.0, prime_volume_ml=0.0)
    warn_va = check_drug_administration(hep, "protamine")
    assert warn_va is not None and warn_va.code == "protamine_on_bypass"
    hep.stop_extracorporeal(reinfuse=False)
    assert check_drug_administration(hep, "protamine") is None

    # Anticoagulation amplifies active hemorrhage blood loss
    ctrl.start_hemorrhage(400.0)
    hep.start_hemorrhage(400.0)
    _step_for(ctrl, 60.0)
    _step_for(hep, 60.0)
    assert hep.state.blood_out_ml > ctrl.state.blood_out_ml * 1.25


def test_hiippala_1995_dilutional_coagulopathy_lethal_triad_and_targeted_rescue():
    """Hiippala 1995: during massive blood loss replaced with red cells and plasma-poor fluids,
    fibrinogen falls below 1.0 g/L first (~142% CBV), clotting factors / INR worsen next, and
    platelets reach 50 x 10^9/L last (~230% CBV). Dilutional coagulopathy, hypothermia, and
    severe hypocalcemia amplify active bleeding; FFP, platelets, and cryoprecipitate selectively
    restore factors/INR, platelet count, and fibrinogen; measure_coag_panel posts after 120 s.
    """
    eng = _steady_engine(dt=1.0)
    cbv = eng.hemo.blood_volume_0
    assert eng.state.fibrinogen_g_l == pytest.approx(3.2, abs=0.01)
    assert eng.state.platelets_10e9_l == pytest.approx(250.0, abs=1.0)
    assert eng.state.inr == pytest.approx(1.0, abs=0.01)
    assert eng.coagulation.state.bleeding_multiplier == 1.0

    # 1. Exact Hiippala 1995 washout milestones (142% CBV and 230% CBV)
    eng.coagulation.record_blood_loss(1.42 * cbv, cbv)
    assert eng.coagulation.state.fibrinogen_g_l == pytest.approx(1.00, abs=0.08)
    assert eng.coagulation.state.platelets_10e9_l > 80.0
    assert eng.coagulation.state.inr > 1.50

    eng.coagulation.record_blood_loss((2.30 - 1.42) * cbv, cbv)
    assert eng.coagulation.state.fibrinogen_g_l < 0.60
    assert eng.coagulation.state.platelets_10e9_l == pytest.approx(50.0, abs=5.0)
    assert eng.coagulation.state.inr > 2.2

    # 2. Lethal triad amplifies bleeding_multiplier cumulatively
    eng.coagulation.sync_hemostasis(temp_c=37.0, ionized_calcium_mmol_l=1.20)
    mult_dil = eng.coagulation.state.bleeding_multiplier
    assert mult_dil > 1.55
    eng.coagulation.sync_hemostasis(temp_c=33.0, ionized_calcium_mmol_l=0.70)
    mult_triad = eng.coagulation.state.bleeding_multiplier
    assert mult_triad > mult_dil + 0.25

    # 3. Live hemorrhage + targeted rescue + measure_coag_panel turnaround
    live = _steady_engine(dt=1.0)
    live.blood_infusion_rate_ml_min = 300.0
    # Simulate 1.5 CBV cumulative blood loss via active hemorrhage with PRBC/crystalloid maintenance
    live.coagulation.record_blood_loss(1.50 * live.hemo.blood_volume_0, live.hemo.blood_volume_0)
    live.step(1.0)
    panel_pre = live.measure_coag_panel()
    assert panel_pre.ready_at == pytest.approx(panel_pre.drawn_at + 120.0)
    assert panel_pre.fibrinogen_g_l < 1.0
    assert panel_pre.inr > 1.55
    assert panel_pre.platelets_10e9_l < 100.0

    # Cryoprecipitate (2 pools = 200 mL) selectively raises fibrinogen rapidly
    fib_before_cryo = live.state.fibrinogen_g_l
    plt_before_cryo = live.state.platelets_10e9_l
    live.give_cryoprecipitate(200.0)
    _step_for(live, 60.0, dt=1.0)
    assert live.state.fibrinogen_g_l > fib_before_cryo + 0.90
    assert abs(live.state.platelets_10e9_l - plt_before_cryo) < 5.0

    # FFP (1000 mL = 4 U) raises clotting factor activity and improves INR
    inr_before_ffp = live.state.inr
    live.give_ffp(1000.0)
    _step_for(live, 240.0, dt=1.0)
    assert live.state.inr < inr_before_ffp - 0.15

    # Platelets (500 mL = 2 apheresis units) selectively raises platelet count
    plt_before_plt = live.state.platelets_10e9_l
    live.give_platelets(500.0)
    _step_for(live, 120.0, dt=1.0)
    assert live.state.platelets_10e9_l > plt_before_plt + 40.0
