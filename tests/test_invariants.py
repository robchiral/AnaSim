"""Cross-module physical, numerical, monitor, and clinical interaction invariants.

These tests target failure classes that single-module unit tests miss:
domain-corner steady-state drift, zero-flow arterial isolation, conservation
laws, composite step-size divergence, Rhythm x Airway x VentMode display
semantics, and pairwise pathology x crisis x drug interactions.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from anasim.core.clinical_actions import check_intubation
from anasim.core.engine import SimulationEngine
from anasim.core.enums import PULSELESS_RHYTHMS, RhythmType
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.web import WebSession

pytestmark = pytest.mark.invariant


DOMAIN_CORNERS: list[tuple[str, dict[str, Any]]] = [
    ("young_lean_f", dict(age=18.0, weight=50.0, height=165.0, sex="female")),
    ("elderly_small_f", dict(age=85.0, weight=50.0, height=155.0, sex="female")),
    ("morbid_obese_m", dict(age=45.0, weight=195.0, height=182.0, sex="male")),
    ("hypothermic_30c", dict(age=40.0, weight=70.0, height=170.0, sex="male", baseline_temp=30.0)),
    ("febrile_39c", dict(age=40.0, weight=70.0, height=170.0, sex="male", baseline_temp=39.5)),
    ("cirrhosis", dict(age=55.0, weight=70.0, height=170.0, sex="male", hepatic_function=0.5)),
    ("esrd", dict(age=60.0, weight=70.0, height=170.0, sex="male", renal_function=0.05)),
    ("hfref_cad", dict(age=68.0, weight=78.0, height=172.0, sex="male", heart_failure="hfref", coronary_flow_reserve=1.6)),
    ("hfpef_copd", dict(age=72.0, weight=85.0, height=168.0, sex="female", heart_failure="hfpef", copd_severity=1.0)),
    ("t2d_hyperglycemic", dict(age=62.0, weight=92.0, height=172.0, sex="male", type2_diabetes=True, baseline_glucose=210.0)),
    ("term_pregnancy", dict(age=30.0, weight=65.0, height=165.0, sex="female", gestational_age_weeks=39.0, position="left_lateral")),
]

CORE_FINITE_FIELDS = (
    "map", "sbp", "dbp", "hr", "co", "svr", "spo2", "pao2", "pa_co2",
    "ph", "hco3", "base_excess", "lactate", "potassium", "ionized_calcium",
    "glucose", "insulin_mu_l",
    "temp_c", "bis", "tof", "mv", "vt", "rr",
)


def _assert_finite_state(engine: SimulationEngine) -> None:
    s = engine.state
    for name in CORE_FINITE_FIELDS:
        val = getattr(s, name)
        assert math.isfinite(val), f"{name} is not finite: {val}"
    assert s.sbp >= s.map - 1e-6 and s.map >= s.dbp - 1e-6 and s.dbp >= -1e-6
    assert 0.0 <= s.spo2 <= 100.0
    assert 0.0 <= s.bis <= 100.0
    assert 0.0 <= s.tof <= 100.0


@pytest.mark.parametrize("mode", ["awake", "steady_state"])
@pytest.mark.parametrize("label,overrides", DOMAIN_CORNERS, ids=[c[0] for c in DOMAIN_CORNERS])
def test_domain_corner_resting_stability(mode: str, label: str, overrides: dict[str, Any]):
    """Starting at any Patient domain corner remains finite, non-arresting, and drift-free over 5 min."""
    cfg = SimulationConfig(mode=mode, dt=1.0, rng_seed=1, maintenance_fluid_ml_hr=0.0)
    engine = SimulationEngine(Patient(**overrides), cfg)
    engine.start()
    lac0 = engine.state.lactate
    for _ in range(300):
        engine.step(1.0)
    _assert_finite_state(engine)
    s = engine.state
    assert not s.cardiac_arrest, f"{label} ({mode}) arrested at rest"
    assert s.co > 1.0, f"{label} ({mode}) collapsed CO={s.co}"
    assert 7.20 <= s.ph <= 7.56, f"{label} ({mode}) pH={s.ph}"
    assert 22.0 <= s.pa_co2 <= 60.0, f"{label} ({mode}) PaCO2={s.pa_co2}"
    # Normoxic rest must not invent lactic acidosis (guards hypothermic resting-pool drift too).
    assert abs(s.lactate - lac0) < 0.35, f"{label} ({mode}) lactate drifted {lac0} -> {s.lactate}"


def test_zero_flow_arterial_isolation_across_all_iv_inputs(engine_factory, advance_time):
    """During zero-flow VFIB without compressions, IV fluids, blood, CaCl2, and non-esterase drugs
    cannot alter arterial blood gas readings, electrolytes, Hb, or effect sites until circulation resumes."""
    eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=1), start=True)
    eng.set_rhythm("VFIB")
    advance_time(eng, 6.0, dt=0.5)
    abg0 = eng.draw_blood_gas()
    poc0 = eng.measure_glucose()
    ins0 = eng.state.insulin_mu_l
    prop0, roc0 = eng.state.propofol_ce, eng.state.roc_ce
    act0 = eng.state.act_s
    eng.give_saline(1500.0)
    eng.give_fluid(1000.0)
    eng.give_d50w(50.0)
    eng.give_sodium_bicarbonate(50.0)
    eng.give_blood(700.0)
    eng.give_calcium_chloride(1000.0)
    eng.give_drug_bolus("epi", 1000.0)
    eng.give_drug_bolus("roc", 70.0)
    eng.give_drug_bolus("propofol", 140.0)
    eng.give_drug_bolus("heparin", 21000.0)
    eng.give_drug_bolus("protamine", 50.0)
    eng.give_drug_bolus("insulin", 10.0)
    advance_time(eng, 15.0, dt=0.5)
    abg1 = eng.draw_blood_gas()
    poc1 = eng.measure_glucose()
    assert abs(abg1.ph - abg0.ph) < 0.005
    assert abs(abg1.pa_co2 - abg0.pa_co2) < 0.25
    assert abg1.hco3 == pytest.approx(abg0.hco3, abs=1e-5)
    assert abs(abg1.sao2 - abg0.sao2) < 0.1
    assert abg1.sodium == pytest.approx(abg0.sodium, abs=1e-6)
    assert abg1.potassium == pytest.approx(abg0.potassium, abs=1e-6)
    assert abg1.chloride == pytest.approx(abg0.chloride, abs=1e-6)
    assert abg1.ionized_calcium == pytest.approx(abg0.ionized_calcium, abs=1e-6)
    assert abg1.lactate == pytest.approx(abg0.lactate, abs=1e-6)
    assert abg1.glucose == pytest.approx(abg0.glucose, abs=1e-6)
    assert poc1.glucose == pytest.approx(poc0.glucose, abs=1e-6)
    assert eng.state.insulin_mu_l == pytest.approx(ins0, abs=1e-6)
    assert abg1.hb == pytest.approx(abg0.hb, abs=1e-9)
    assert abs(eng.state.propofol_ce - prop0) < 0.05
    assert abs(eng.state.roc_ce - roc0) < 0.05
    assert eng.state.act_s == pytest.approx(act0, abs=0.5)
    assert eng.state.heparin_u_ml == pytest.approx(0.0, abs=1e-6)
    assert eng.state.protamine_ug_ml == pytest.approx(0.0, abs=1e-6)
    assert eng.coagulation.state.protamine_svr_factor == pytest.approx(1.0, abs=1e-6)
    assert eng.hemo._hyperosmolar_svr_factor == pytest.approx(1.0, abs=1e-6)

    # Starting chest compressions restores circulation and delivers queued drugs/calcium/heparin/dextrose.
    eng.set_chest_compressions(True)
    advance_time(eng, 45.0, dt=0.5)
    abg2 = eng.draw_blood_gas()
    assert eng.state.epi_ce > 1.0
    assert eng.state.roc_ce > roc0 + 0.5
    assert abg2.ionized_calcium > abg1.ionized_calcium + 0.05
    assert abg2.lactate > abg1.lactate + 0.05
    assert abg2.glucose > abg1.glucose + 10.0
    assert eng.state.insulin_mu_l > ins0 + 10.0
    assert eng.state.act_s > act0 + 100.0
    assert eng.state.heparin_u_ml > 1.0


@pytest.mark.parametrize("mode", ["VCV", "PCV", "PCV-VG", "SIMV-VC", "SIMV-PC", "PSV", "CPAP"])
def test_empty_bellows_zero_fgf_collapses_peep_in_every_mode(engine_factory, advance_time, mode: str):
    """Draining the bellows with zero FGF collapses PEEP without oscillation and empties VT once recoil O2 is consumed."""
    eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1), start=True)
    eng.give_drug_bolus("roc", 70.0)
    eng.set_vent_settings(rr=12, vt=0.5, peep=10.0, ie="1:2", mode=mode, p_insp=18.0, p_support=12.0)
    eng.set_vent_power(True)
    advance_time(eng, 15.0, dt=1.0)
    eng.set_fgf(0.0, 0.0, 0.0)
    eng.circuit.reservoir_volume = 0.0
    advance_time(eng, 15.0, dt=1.0)
    assert eng.state.peep < 0.5, f"mode {mode} bounced PEEP={eng.state.peep} right after collapse"
    advance_time(eng, 285.0, dt=1.0)
    assert eng.state.peep < 0.5, f"mode {mode} retained phantom PEEP={eng.state.peep}"
    assert eng.state.vt < 50.0, f"mode {mode} delivered phantom VT={eng.state.vt}"


def test_volatile_temperature_cycle_and_vaporizer_conservation(engine_factory, advance_time):
    """Cooling then rewarming conserves dissolved volatile tissue mass; vaporizer depletes and refills cleanly."""
    eng = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=1.0, rng_seed=1),
        start=True,
    )
    pk = eng.pk_sevo

    def _dissolved_mass(temp_c: float) -> float:
        lbg = pk.lambda_b_g_at_temp(temp_c)
        fat_scale = (pk.lambda_b_g / lbg) ** 0.5
        s = pk.state
        return (
            pk.v_vrg * pk.lambda_t_b_vrg * lbg * s.p_vrg
            + pk.v_mus * pk.lambda_t_b_mus * lbg * s.p_mus
            + pk.v_fat * pk.lambda_t_b_fat * fat_scale * lbg * s.p_fat
        )

    m0 = _dissolved_mass(37.0)
    # Zero ventilation with open circulation isolates blood-tissue temperature exchange.
    for temp in (30.0, 37.0):
        for _ in range(30):
            pk.step(2.0, 0.0, 0.0, eng.state.co, temp_c=temp, lung_volume_l=pk.frc)
    # Small alveolar-tissue exchange stays bounded; no runaway mass creation or destruction.
    m1 = _dissolved_mass(37.0)
    assert abs(m1 - m0) / max(1e-6, m0) < 0.06

    # Vaporizer monotonically depletes, halts output when empty, and restores delivery on refill.
    eng.set_fgf(4.0, 0.0, 0.0)
    eng.set_vaporizer(4.0)
    lvl0 = eng.vaporizer.state.level
    advance_time(eng, 60.0, dt=1.0)
    assert eng.vaporizer.state.level < lvl0
    eng.vaporizer.state.level = 0.0
    assert eng.vaporizer.step(1.0, 4.0) == 0.0
    eng.refill_vaporizer()
    assert eng.vaporizer.state.level == pytest.approx(eng.vaporizer.CAPACITY_ML)
    eng.set_vaporizer(4.0)
    assert eng.vaporizer.step(1.0, 4.0) > 3.0


@pytest.mark.parametrize("scenario_name", [
    "mh_vcv",
    "hemorrhage_resuscitation",
    "anaphylaxis_psv",
    "vfib_cpr_epi",
    "hypothermic_shivering",
    "massive_transfusion_ca",
])
def test_composite_scenarios_are_step_size_invariant(engine_factory, advance_time, scenario_name: str):
    """High-stress composite scenarios agree closely between dt=0.1 s and dt=1.0 s."""
    outcomes = []
    for dt in (0.1, 1.0):
        if scenario_name == "mh_vcv":
            eng = engine_factory(
                config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=dt, rng_seed=1),
                start=True,
            )
            eng.start_malignant_hyperthermia()
            advance_time(eng, 150.0, dt=dt)
        elif scenario_name == "hemorrhage_resuscitation":
            eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=dt, rng_seed=1), start=True)
            eng.hemo.add_volume(-1200.0)
            eng.give_fluid(1000.0)
            eng.set_drug_rate("nore", 0.10 * eng.patient.weight)
            advance_time(eng, 150.0, dt=dt)
        elif scenario_name == "anaphylaxis_psv":
            eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=dt, rng_seed=1), start=True)
            eng.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="PSV", p_support=12.0)
            eng.start_anaphylaxis()
            advance_time(eng, 60.0, dt=dt)
            eng.set_drug_rate("epi", 0.08 * eng.patient.weight)
            advance_time(eng, 90.0, dt=dt)
        elif scenario_name == "vfib_cpr_epi":
            eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=dt, rng_seed=1), start=True)
            eng.set_rhythm("VFIB")
            advance_time(eng, 20.0, dt=dt)
            eng.set_chest_compressions(True)
            eng.give_drug_bolus("epi", 1000.0)
            advance_time(eng, 90.0, dt=dt)
        elif scenario_name == "hypothermic_shivering":
            eng = engine_factory(
                config=SimulationConfig(mode="awake", dt=dt, rng_seed=1),
                start=True,
                baseline_temp=34.5,
            )
            advance_time(eng, 120.0, dt=dt)
        else:
            eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=dt, rng_seed=1), start=True)
            eng.hemo.add_volume(-1000.0)
            eng.give_blood(1200.0)
            eng.give_calcium_chloride(1000.0)
            advance_time(eng, 120.0, dt=dt)
        _assert_finite_state(eng)
        outcomes.append(eng.state)

    fine, coarse = outcomes
    assert abs(coarse.map - fine.map) < 2.5
    assert abs(coarse.hr - fine.hr) < 3.0
    assert abs(coarse.pa_co2 - fine.pa_co2) < 1.5
    assert abs(coarse.pao2 - fine.pao2) < 6.0
    assert abs(coarse.ph - fine.ph) < 0.02
    assert abs(coarse.lactate - fine.lactate) < 0.25
    assert abs(coarse.temp_c - fine.temp_c) < 0.1


def test_rhythm_airway_and_session_projection_invariants(engine_factory, advance_time):
    """Every rhythm, airway disconnect, and vent/bag toggle obeys display and control invariants."""
    for rhythm in RhythmType:
        eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=1), start=True)
        eng.set_rhythm(rhythm.name)
        advance_time(eng, 10.0, dt=0.5)
        _assert_finite_state(eng)
        s = eng.state
        if rhythm in PULSELESS_RHYTHMS:
            assert s.co == 0.0
            assert s.sbp == pytest.approx(s.map, abs=1e-6)
            assert s.dbp == pytest.approx(s.map, abs=1e-6)
        else:
            assert s.sbp >= s.map >= s.dbp

    # Mutual exclusion and open-APL manual breath guard directly on SimulationEngine.
    eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=1), start=True)
    assert eng.vent.is_on and not eng.bag_mask_active
    eng.set_bag_mask_ventilation(True)
    assert eng.bag_mask_active and not eng.vent.is_on
    eng.set_vent_power(True)
    assert eng.vent.is_on and not eng.bag_mask_active
    eng.set_vent_power(False)
    eng.set_apl_valve(0.0)
    before_actions = len(eng.actions.records)
    eng.deliver_manual_breath()
    assert len(eng.actions.records) == before_actions

    # Airway disconnect resets WebSession spirometry loops and silences room-air low-FiO2 alarms.
    session = WebSession({"age": 40, "weight": 70, "height": 170, "sex": "male", "mode": "steady_state"})
    session.command("run", json.dumps({"running": True}))
    session.advance(4.0)
    session.command("airway", json.dumps({"mode": "None"}))
    snap = json.loads(session.advance(1.0))
    assert any(run.get("reset") for run in snap["loop"])
    assert not session.engine.state.alarms.get("FiO2", {}).get("low")


PAIRWISE_STRESS_CASES = [
    ("hfref_hemorrhage_pressors", dict(heart_failure="hfref", age=68), "hemorrhage"),
    ("hfpef_sepsis_nore", dict(heart_failure="hfpef", age=74), "sepsis"),
    ("copd_anaphylaxis_epi", dict(copd_severity=1.0, age=65), "anaphylaxis"),
    ("obese_mh_dantrolene", dict(weight=190.0, height=182.0, age=42), "mh"),
    ("elderly_hypothermic_cpr", dict(age=82.0, baseline_temp=29.0), "hypothermic_cpr"),
    ("cirrhosis_massive_prbc", dict(hepatic_function=0.5, age=58), "massive_prbc"),
    ("esrd_sux_hyperk", dict(renal_function=0.05, age=62), "sux_esrd"),
    ("pregnant_light_sevo_intubation", dict(sex="female", weight=65.0, height=165.0, gestational_age_weeks=38.0), "light_intub"),
]


@pytest.mark.parametrize("label,patient_kw,crisis", PAIRWISE_STRESS_CASES, ids=[c[0] for c in PAIRWISE_STRESS_CASES])
def test_pairwise_condition_crisis_clinical_invariants(
    engine_factory, advance_time, label: str, patient_kw: dict[str, Any], crisis: str,
):
    """Pairwise combinations of patient conditions, crises, and drugs obey hard clinical laws."""
    if crisis == "hypothermic_cpr":
        eng = engine_factory(
            config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1),
            start=True,
            **patient_kw,
        )
        eng.set_rhythm("VFIB")
        advance_time(eng, 15.0, dt=1.0)
        eng.set_chest_compressions(True)
        eng.give_drug_bolus("epi", 1000.0)
        advance_time(eng, 180.0, dt=1.0)
        _assert_finite_state(eng)
        # Hypothermic CPR must never restore cortical wakefulness or spontaneous tidal breathing.
        assert eng.state.loc > 0.85
        assert eng.resp.state.neural_rr == 0.0
        return

    if crisis == "light_intub":
        eng = engine_factory(
            config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1),
            start=True,
            **patient_kw,
        )
        eng.set_airway_mode("Mask")
        eng.set_fgf(6.0, 0.0)
        eng.set_vaporizer(1.2)
        advance_time(eng, 180.0, dt=1.0)
        _assert_finite_state(eng)
        eng.pk_sevo.state.p_brain = eng.pk_sevo.state.p_alv = 0.45 * eng.pk_sevo.mac_age / 100.0
        eng.pk_sevo.state.mac = 0.45
        fb = check_intubation(eng)
        assert fb is not None and fb.blocked, f"{label}: allowed unparalyzed intubation at 0.45 MAC"
        return

    maint = "balanced" if crisis == "mh" else "tiva"
    eng = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type=maint, dt=1.0, rng_seed=1),
        start=True,
        **patient_kw,
    )
    if crisis == "hemorrhage":
        base_co = eng.state.co
        eng.hemo.add_volume(-0.55 * eng.hemo.blood_volume_0)
        eng.set_drug_rate("nore", 0.30 * eng.patient.weight)
        eng.set_drug_rate("epi", 0.15 * eng.patient.weight)
        advance_time(eng, 120.0, dt=1.0)
        _assert_finite_state(eng)
        # Pressors alone cannot restore normal CO after 55% exsanguination without volume.
        assert eng.state.co < 0.60 * base_co
    elif crisis == "sepsis":
        eng.start_sepsis()
        advance_time(eng, 90.0, dt=1.0)
        untreated = copy.deepcopy(eng)
        eng.give_fluid(1000.0)
        eng.set_drug_rate("nore", 0.25 * eng.patient.weight)
        advance_time(eng, 120.0, dt=1.0)
        advance_time(untreated, 120.0, dt=1.0)
        _assert_finite_state(eng)
        assert eng.state.map > untreated.state.map + 10.0
    elif crisis == "anaphylaxis":
        eng.start_anaphylaxis()
        advance_time(eng, 90.0, dt=1.0)
        eng.set_drug_rate("epi", 0.12 * eng.patient.weight)
        eng.give_fluid(1000.0)
        advance_time(eng, 120.0, dt=1.0)
        _assert_finite_state(eng)
        assert eng.state.hr <= min(200.0, max(168.0, 208.0 - 0.7 * eng.patient.age)) + 1.0
    elif crisis == "mh":
        eng.start_malignant_hyperthermia()
        advance_time(eng, 180.0, dt=1.0)
        eng.set_vaporizer(0.0)
        eng.set_fgf(10.0, 0.0)
        eng.give_drug_bolus("dantrolene", 2.5 * eng.patient.weight)
        advance_time(eng, 600.0, dt=1.0)
        _assert_finite_state(eng)
        assert eng.malignant_hyperthermia.severity < 0.15
    elif crisis == "massive_prbc":
        eng.blood_infusion_rate_ml_min = 300.0
        eng.hemo.add_volume(-1200.0)
        eng.give_blood(1800.0)
        advance_time(eng, 360.0, dt=1.0)
        _assert_finite_state(eng)
        assert eng.state.ionized_calcium < 1.12
    elif crisis == "sux_esrd":
        k0 = eng.state.potassium
        eng.give_drug_bolus("succinylcholine", 1.5 * eng.patient.weight)
        advance_time(eng, 180.0, dt=1.0)
        _assert_finite_state(eng)
        assert eng.state.potassium > k0 + 0.25


def test_extracorporeal_mode_switching_and_colloid_prime_conservation(engine_factory, advance_time):
    """Cross-mode switching cleans up prior prime/flags without double-priming,
    and colloid/albumin priming plus reservoir fluid conserve 100% circuit volume."""
    eng = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=3, maintenance_fluid_ml_hr=0.0),
        start=True,
    )
    advance_time(eng, 5.0, dt=0.5)
    pat_bv0 = eng.hemo.patient_blood_volume

    # Colloid/albumin prime stays 100% intravascular (no instant third-space loss).
    eng.start_cpb(blood_flow_l_min=4.5, prime_volume_ml=1400.0, prime_label="albumin", initial_reservoir_ml=800.0)
    assert eng.hemo.patient_blood_volume == pytest.approx(pat_bv0, abs=1.0)
    assert eng.hemo.extracorporeal_volume_ml == pytest.approx(1400.0, abs=1e-6)
    added = eng.add_cpb_reservoir_fluid(500.0, kind="albumin")
    assert added == pytest.approx(500.0, abs=1e-6)
    assert eng.hemo.extracorporeal_volume_ml == pytest.approx(1900.0, abs=1e-6)

    # Apply CPB-specific controls, then switch directly to VV ECMO -> CPB flags clear and volume is not double-primed.
    eng.set_cpb_cardioplegia(True)
    eng.set_bypass_alarm_mode(True)
    advance_time(eng, 10.0, dt=0.5)
    pat_before_vv = eng.hemo.patient_blood_volume
    res_before_vv = eng.extracorporeal.state.reservoir_volume_ml
    eng.start_vv_ecmo(blood_flow_l_min=4.0, prime_volume_ml=600.0)
    assert eng.state.ecmo_mode == "vv_ecmo"
    assert eng.hemo.extracorporeal_volume_ml == pytest.approx(600.0, abs=1e-6)
    assert eng.hemo.patient_blood_volume == pytest.approx(pat_before_vv + res_before_vv, abs=1.0)
    assert eng.hemo._aortic_cross_clamp is False
    assert eng.hemo._cardioplegia_arrest is False
    assert eng.hemo._cpb_open_drainage is False
    assert eng.hemo._ecmo_arterial_flow_l_min == 0.0
    assert eng.extracorporeal.state.bypass_alarm_mode is False

    # Switch VV -> VA -> zero-prime CPB -> paused stop clears flags and telemetry immediately.
    eng.start_va_ecmo(blood_flow_l_min=3.5, prime_volume_ml=650.0)
    eng.set_ecmo_lv_vent(1.0)
    assert eng.hemo.extracorporeal_volume_ml == pytest.approx(650.0, abs=1e-6)
    bv_before_zero_cpb = eng.hemo.patient_blood_volume
    eng.start_cpb(blood_flow_l_min=4.5, prime_volume_ml=0.0, initial_reservoir_ml=800.0)
    assert eng.extracorporeal.state.lv_vent_l_min == 0.0
    assert eng.extracorporeal.state.reservoir_volume_ml == 0.0
    assert eng.hemo.extracorporeal_volume_ml == 0.0
    assert eng.hemo.blood_volume == pytest.approx(bv_before_zero_cpb, abs=1.0)
    eng.set_cpb_cardioplegia(True)
    eng.stop()
    eng.stop_extracorporeal(reinfuse=False)
    assert eng.state.ecmo_mode == "none"
    assert eng.state.pump_flow == 0.0
    assert eng.hemo.extracorporeal_volume_ml == 0.0
    assert eng.hemo._aortic_cross_clamp is False
    assert eng.hemo._cardioplegia_arrest is False
    assert eng.hemo._cpb_open_drainage is False
    assert eng.state.blood_volume == pytest.approx(eng.hemo.blood_volume, abs=1e-6)


def test_extracorporeal_mode_guards_and_vv_arrest_alarms(engine_factory, advance_time):
    """Mode-specific extracorporeal setters reject invalid modes and VV ECMO never silences VFIB alarms."""
    eng = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=5, arterial_line_enabled=True),
        start=True,
    )
    eng.start_vv_ecmo(blood_flow_l_min=4.0, sweep_gas_l_min=4.0, prime_volume_ml=0.0)
    with pytest.raises(ValueError, match="VA ECMO or CPB"):
        eng.set_bypass_alarm_mode(True)
    with pytest.raises(ValueError, match="VA ECMO or CPB"):
        eng.set_ecmo_lv_vent(1.5)
    with pytest.raises(ValueError, match="CPB"):
        eng.set_cpb_cross_clamp(True)
    with pytest.raises(ValueError, match="CPB"):
        eng.set_cpb_cardioplegia(True)
    with pytest.raises(ValueError, match="CPB"):
        eng.set_cpb_venous_clamp(0.0)
    with pytest.raises(ValueError, match="CPB"):
        eng.set_cpb_blood_gas_strategy("ph_stat")
    assert eng.extracorporeal.state.bypass_alarm_mode is False
    assert eng.extracorporeal.state.lv_vent_l_min == 0.0
    assert eng.extracorporeal.state.aortic_cross_clamp is False
    assert eng.extracorporeal.state.cardioplegia_arrest is False
    assert eng.extracorporeal.state.blood_gas_strategy.value == "alpha_stat"

    # Even if bypass_alarm_mode were forced directly, VV ECMO must still alarm on VFIB.
    eng.extracorporeal.state.bypass_alarm_mode = True
    eng.set_rhythm("VFIB")
    advance_time(eng, 20.0, dt=0.5)
    assert eng.state.cardiac_arrest is True
    assert "VFIB" in eng.state.alarms


def test_extracorporeal_abg_sites_and_temperature_parity(engine_factory, advance_time):
    """Pre/post-oxygenator ABGs require active support, fresh starts seed real patient gases,
    hypothermic warming shifts both PaO2 and SaO2 consistently, and apneic CPB reports sweep FiO2."""
    eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=7), start=True)
    with pytest.raises(ValueError, match="active"):
        eng.draw_blood_gas("pre_oxy")
    with pytest.raises(ValueError, match="active"):
        eng.draw_blood_gas("post_oxy")

    # Freshly started circuit before stepping should seed real patient gases (not default placeholders).
    eng.start_cpb(blood_flow_l_min=4.5, sweep_gas_l_min=3.0, sweep_fio2=0.80, prime_volume_ml=0.0)
    eng.set_vent_power(False)
    pre0 = eng.draw_blood_gas("pre_oxy")
    post0 = eng.draw_blood_gas("post_oxy")
    assert 20.0 <= pre0.pao2 <= 70.0
    assert post0.pao2 > 200.0
    assert eng.draw_blood_gas("arterial").fio2 == pytest.approx(0.80, abs=1e-6)

    # At 28 °C, warming blood anaerobically to 37 °C shifts P50 upward so warmed SaO2 aligns with warmed PaO2.
    eng.set_cpb_cardioplegia(True)
    eng.set_ecmo_heater_cooler(True, 28.0)
    eng.state.temp_c = 28.0
    eng.thermal.peripheral_temp_c = 28.0
    advance_time(eng, 20.0, dt=0.5)
    pre_cold = eng.draw_blood_gas("pre_oxy")
    assert pre_cold.pao2 > eng.extracorporeal.state.pre_oxy_po2 + 5.0
    assert pre_cold.sao2 <= eng.extracorporeal.state.pre_oxy_so2 * 100.0 + 1.5


def test_peripheral_va_ecmo_harlequin_step_size_invariance(engine_factory, advance_time):
    """Peripheral VA ECMO differential oxygenation agrees closely between dt=0.1 s and dt=2.0 s."""
    results = []
    for dt in (0.1, 2.0):
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", dt=dt, rng_seed=19, arterial_line_enabled=True),
            start=True,
        )
        eng.set_vent_settings(rr=10.0, vt=0.36, peep=5.0, ie="1:2", mode="VCV", fio2=0.25)
        eng.set_ards_shunt(0.65)
        eng.start_va_ecmo(
            blood_flow_l_min=3.2, sweep_gas_l_min=4.0, sweep_fio2=1.0,
            cannulation="peripheral_fem", prime_volume_ml=0.0,
        )
        advance_time(eng, 120.0, dt=dt)
        results.append(eng)

    fine, coarse = results
    assert abs(coarse.extracorporeal.state.upper_body_pao2 - fine.extracorporeal.state.upper_body_pao2) < 2.5
    assert abs(coarse.state.upper_sao2 - fine.state.upper_sao2) < 1.5
    assert abs(coarse.state.map - fine.state.map) < 2.0


def test_glucose_insulin_dextrose_step_size_invariance(engine_factory, advance_time):
    """Coupled glucose, insulin, K+, and osmotic Na+ dynamics agree closely across dt=0.2 s and dt=2.0 s."""
    results = []
    for dt in (0.2, 2.0):
        eng = engine_factory(
            config=SimulationConfig(mode="awake", dt=dt, rng_seed=11, maintenance_fluid_ml_hr=0.0),
            start=True,
            type2_diabetes=True,
            baseline_glucose=210.0,
        )
        eng.give_drug_bolus("insulin", 5.0)
        eng.set_drug_rate("insulin", 3.0)
        eng.give_d50w(25.0)
        advance_time(eng, 600.0, dt=dt)
        results.append(eng)

    fine, coarse = results
    assert abs(coarse.state.glucose - fine.state.glucose) < 1.5
    assert abs(coarse.state.insulin_mu_l - fine.state.insulin_mu_l) < 1.5
    assert abs(coarse.state.potassium - fine.state.potassium) < 0.05
    assert abs(coarse.state.sodium - fine.state.sodium) < 0.2


def test_glucose_low_flow_glycosuria_cpb_prime_and_hypo_settle_invariants(engine_factory):
    """Low-flow glycosuria conserves mass, CPB prime updates glucose immediately, and steady_state starts clean."""
    from anasim.physiology.glucose import MG_DL_PER_MMOL_L, MG_PER_MMOL, GlucoseModel

    # 1. Low-flow (circ=0.5) glycosuria mass removed matches reported glycosuria_mg_min (no double attenuation).
    g_perf = GlucoseModel(lean_mass_kg=70.0, ecf_l=14.0, baseline_glucose_mg_dl=350.0)
    g_anuric = GlucoseModel(lean_mass_kg=70.0, ecf_l=14.0, baseline_glucose_mg_dl=350.0)
    g_perf.step(60.0, renal_perfusion=1.0, circulation=0.5)
    g_anuric.step(60.0, renal_perfusion=0.0, circulation=0.5)
    removed_mg = (g_anuric._standalone_glucose_mmol - g_perf._standalone_glucose_mmol) * MG_PER_MMOL
    assert removed_mg == pytest.approx(g_perf.state.glycosuria_mg_min, rel=0.03)

    # 2. Immediate CPB prime and reservoir additions update arterial/POC/ABG glucose before the next step.
    cpb = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=3, maintenance_fluid_ml_hr=0.0),
        start=True,
        baseline_glucose=200.0,
    )
    cpb.start_cpb(blood_flow_l_min=4.5, prime_volume_ml=1400.0)
    expected_prime = (cpb.hemo.fluids.glucose_mmol / cpb.hemo.fluids.ecf_l) * MG_DL_PER_MMOL_L
    assert cpb.state.glucose == pytest.approx(expected_prime, abs=0.05)
    assert cpb.measure_glucose().glucose == pytest.approx(expected_prime, abs=0.05)
    assert cpb.draw_blood_gas().glucose == pytest.approx(expected_prime, abs=0.05)

    cpb.add_cpb_reservoir_fluid(500.0, kind="crystalloid")
    expected_res = (cpb.hemo.fluids.glucose_mmol / cpb.hemo.fluids.ecf_l) * MG_DL_PER_MMOL_L
    assert cpb.state.glucose == pytest.approx(expected_res, abs=0.05)
    assert cpb.measure_glucose().glucose == pytest.approx(expected_res, abs=0.05)

    # 3. Hidden steady_state settle at baseline_glucose=40 leaves t=0 cerebral hypoglycemic_dysfunction at 0.
    hypo_ss = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=3, maintenance_fluid_ml_hr=0.0),
        start=False,
        baseline_glucose=40.0,
    )
    assert hypo_ss.cerebral.hypoglycemic_dysfunction == 0.0


def test_sodium_bicarbonate_step_size_invariance(engine_factory, advance_time):
    """Coupled NaHCO3 acid-base, CO2, electrolyte, and SVR dynamics agree across dt=0.2 s and dt=2.0 s."""
    results = []
    for dt in (0.2, 2.0):
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", dt=dt, rng_seed=17, maintenance_fluid_ml_hr=0.0),
            start=True,
        )
        # Controlled VCV isolates NaHCO3 acid-base/osmotic dynamics from asynchronous spontaneous-over-VCV beat phase.
        eng.give_drug_bolus("roc", 50.0)
        eng.resp.remove_buffer_base(120.0)
        advance_time(eng, 60.0, dt=dt)
        eng.give_sodium_bicarbonate(100.0)
        advance_time(eng, 300.0, dt=dt)
        results.append(eng)

    fine, coarse = results
    assert abs(coarse.state.ph - fine.state.ph) < 0.008
    assert abs(coarse.state.pa_co2 - fine.state.pa_co2) < 0.6
    assert abs(coarse.state.hco3 - fine.state.hco3) < 0.4
    assert abs(coarse.state.sodium - fine.state.sodium) < 0.2
    assert abs(coarse.state.ionized_calcium - fine.state.ionized_calcium) < 0.02
    assert abs(coarse.state.map - fine.state.map) < 1.5


def test_airway_thoracic_embolism_and_neuraxial_arrest_isolation_and_stop_events(engine_factory, advance_time):
    """Zero-flow arrest halts blood-borne gas/drug exchange in PTX, PE, VAE, and epidural epinephrine,
    and stop_events resets all new crisis states cleanly."""
    from anasim.physiology.embolism import PulmonaryEmbolismModel
    from anasim.physiology.neuraxial import NeuraxialModel
    from anasim.physiology.pneumothorax import PneumothoraxModel

    # 1. Zero-flow isolation: PTX N2O diffusion, PE lysis, VAE washout/N2O, and epidural epi absorption halt at circ=0.
    ptx = PneumothoraxModel()
    ptx.set_closed_volume(0.30)
    ptx.step(120.0, ppv_active=False, mean_paw_cmh2o=0.0, peep_cmh2o=0.0, p_alv_n2o=0.70, circulation=0.0)
    assert ptx.n2o_volume_l == 0.0

    emb = PulmonaryEmbolismModel(reference_weight_kg=70.0)
    emb.start_pe(0.60)
    emb.pe_obstruction = 0.60
    emb.give_thrombolysis()
    emb.start_vae(rate_ml_min=0.0, initial_bolus_ml=60.0)
    emb.stop_vae()
    emb.step(120.0, p_alv_n2o=0.70, alveolar_o2_fraction=1.0, heparin_u_ml=0.5, circulation=0.0)
    assert emb.pe_obstruction == pytest.approx(0.60)
    assert emb.vae_air_ml == pytest.approx(60.0)
    assert emb.vae_n2o_factor == pytest.approx(1.0)

    nx = NeuraxialModel()
    nx.give_epidural_bolus("bupivacaine", 15.0, 0.5, with_epinephrine=True)
    nx.step(120.0, circulation=0.0)
    assert nx.absorbed_epi_ng_ml == 0.0

    # 2. stop_events clears pneumothorax, PE, VAE, and neuraxial catheter/block completely.
    eng = engine_factory(config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=1), start=True)
    eng.start_pneumothorax(severity=0.9, initial_volume_l=0.4)
    eng.start_pulmonary_embolism(0.5)
    eng.start_venous_air_embolism(rate_ml_min=50.0, initial_bolus_ml=40.0)
    eng.place_epidural(True)
    eng.trigger_high_spinal()
    advance_time(eng, 10.0, dt=0.5)
    eng.stop_events()
    assert not eng.pneumothorax.active and eng.pneumothorax.total_volume_l == 0.0
    assert not eng.embolism.pe_active and eng.embolism.vae_air_ml == 0.0
    assert not eng.neuraxial.active and not eng.neuraxial.epidural_catheter


def test_cardiovascular_response_fractions_monotonic_across_all_seven_drugs():
    """Increasing any single drug never increases residual cardiovascular stimulation."""
    import random

    from anasim.patient.pd.anesthesia import ClinicalResponseModel

    crm = ClinicalResponseModel(age=40.0)
    rng = random.Random(101)
    for _ in range(400):
        base = [
            rng.uniform(0.0, 8.0),   # ce_prop
            rng.uniform(0.0, 12.0),  # ce_remi
            rng.uniform(0.0, 2.5),   # mac_sevo
            rng.uniform(0.0, 0.8),   # mac_n2o
            rng.uniform(0.0, 4.0),   # ce_ketamine
            rng.uniform(0.0, 6.0),   # ce_lidocaine
            rng.uniform(0.0, 8.0),   # ce_fentanyl
        ]
        intensity = rng.uniform(0.8, 1.9)
        p0, h0 = crm.cardiovascular_response_fractions(
            base[0], base[1], base[2], base[3], base[4], base[5],
            intensity=intensity, ce_fentanyl=base[6],
        )
        for idx in range(7):
            bumped = list(base)
            bumped[idx] += rng.uniform(0.05, 2.0)
            p1, h1 = crm.cardiovascular_response_fractions(
                bumped[0], bumped[1], bumped[2], bumped[3], bumped[4], bumped[5],
                intensity=intensity, ce_fentanyl=bumped[6],
            )
            assert p1 <= p0 + 1e-12
            assert h1 <= h0 + 1e-12


def test_zero_flow_pea_isolation_and_dt_invariant_rosc_flush(engine_factory, advance_time):
    """Zero-flow PEA isolates arterial gases, volatile p_art, fluid thermal heat,
    PE/VAE kinetics, and protamine vasodilation exposure until circulation resumes."""
    from anasim.physiology.coagulation import CoagulationModel
    from anasim.physiology.embolism import PulmonaryEmbolismModel
    from anasim.physiology.thermal import ThermalModel

    # 1. Zero-flow PEA freezes arterial pH/PaCO2/p_art and isolates oxygenator / fluid heat.
    eng = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", dt=0.5, rng_seed=3),
        start=True,
    )
    advance_time(eng, 15.0, dt=0.5)
    eng.set_rhythm(RhythmType.ASYSTOLE)
    advance_time(eng, 2.0, dt=0.5)
    p_art_0 = eng.pk_sevo.state.p_art
    paco2_0 = eng.state.pa_co2
    eng.set_vaporizer(8.0)
    eng.set_fgf(10.0, 0.0, 0.0)
    eng.give_fluid(500.0)
    advance_time(eng, 20.0, dt=0.5)
    assert eng.pk_sevo.state.p_art == pytest.approx(p_art_0, abs=1e-4)
    assert eng.state.pa_co2 == pytest.approx(paco2_0, abs=0.5)

    # 2. Fluid thermal heat stays queued during zero perfusion and delivers once perfusion returns.
    th = ThermalModel(Patient())
    th.add_fluid_heat_j(1000.0, 6.0, 37.0, is_blood=True)
    assert th._pending_fluid_heat_j < -1e4
    th.step_temperature(1.0, 37.0, 0.0, perfusing=False)
    assert th._pending_fluid_heat_j < -1e4
    th.step_temperature(1.0, 37.0, 0.0, perfusing=True)
    assert th._pending_fluid_heat_j == 0.0

    # 3. PE onset and VAE entrainment halt at circ == 0; protamine ROSC flush is dt-invariant.
    emb = PulmonaryEmbolismModel(reference_weight_kg=70.0)
    emb.start_pe(0.65)
    emb.start_vae(rate_ml_min=80.0, initial_bolus_ml=0.0)
    emb.step(60.0, p_alv_n2o=0.5, alveolar_o2_fraction=0.5, circulation=0.0, pulmonary_circulation=0.0)
    assert emb.pe_obstruction == 0.0
    assert emb.vae_air_ml == 0.0

    exposures = []
    for dt in (0.1, 1.0):
        from anasim.patient.pk_models import HeparinPK, ProtaminePK

        p = Patient()
        coag = CoagulationModel()
        hep, prot = HeparinPK(p), ProtaminePK(p)
        coag.record_protamine_bolus(50.0, prot.v1, circulation=0.0)
        for _ in range(round(20.0 / dt)):
            coag.step(dt, hep, prot, circulation=0.5)
        exposures.append(coag._vaso_exposure_ug_ml)
    assert exposures[0] == pytest.approx(exposures[1], rel=0.03)


def test_extracorporeal_prime_drainage_bypass_silence_ecpr_and_actionlog_events(engine_factory, advance_time):
    """Verify prime validation, closed-circuit drainage throttling, bypass-alarm pump gate,
    eCPR monitor continuity, and pre-mutation crisis event logging."""
    from anasim.core.action_log import ACTION_EVENT_START, ACTION_EVENT_STOP

    eng = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=0.5, rng_seed=5, arterial_line_enabled=True),
        start=True,
    )
    with pytest.raises(ValueError, match="hematocrit"):
        eng.start_va_ecmo(prime_label="blood", prime_hematocrit=0.0)
    with pytest.raises(ValueError, match="hematocrit"):
        eng.start_va_ecmo(prime_label="crystalloid", prime_hematocrit=0.35)

    # Blood prime defaults to Hct 0.55 and does not drop Hb precipitously.
    hb0 = eng.hemo.hb_conc
    eng.start_va_ecmo(
        blood_flow_l_min=4.0, sweep_gas_l_min=4.0, prime_volume_ml=600.0,
        prime_label="blood", bypass_alarm_mode=True,
    )
    assert eng.hemo.hb_conc >= hb0 - 0.3

    # eCPR (VA ECMO + chest compressions) preserves VA ECMO arterial perfusion instead of CPR collapse.
    eng.set_rhythm(RhythmType.VFIB)
    eng.set_chest_compressions(True)
    advance_time(eng, 10.0, dt=0.5)
    assert eng.state.map >= 45.0
    assert eng.state.art_map >= 45.0
    assert not eng.state.cardiac_arrest

    # Bypass alarm silence disables when pump flow drops to zero.
    eng.set_ecmo_flow(0.0)
    advance_time(eng, 10.0, dt=0.5)
    assert "Asystole" in eng.state.alarms or "VFIB" in eng.state.alarms or "Arrest" in eng.state.alarms

    # Closed-circuit ECMO throttles delivered blood flow during severe hypovolemia.
    eng.set_chest_compressions(False)
    eng.set_rhythm(RhythmType.SINUS)
    eng.set_ecmo_flow(5.0)
    eng.hemo.add_volume(-2200.0)
    advance_time(eng, 5.0, dt=0.5)
    assert eng.extracorporeal.state.blood_flow_l_min < 4.5
    eng.stop_extracorporeal(reinfuse=False)

    # Pre-mutation crisis event logging across difficult airway, esophageal, high spinal, and armed protamine.
    ev = engine_factory(config=SimulationConfig(mode="awake", dt=0.5, rng_seed=7), start=True)
    ev.give_drug_bolus("propofol", 200.0)
    ev.give_drug_bolus("roc", 50.0)
    advance_time(ev, 45.0, dt=0.5)
    ev.set_difficult_airway("anterior_dl")
    assert any(r.label == "difficult_airway" for r in ev.actions.records if r.action == ACTION_EVENT_START)
    ev.set_difficult_airway("none")
    assert any(r.label == "difficult_airway" for r in ev.actions.records if r.action == ACTION_EVENT_STOP)

    ev.arm_esophageal_intubation(True)
    ev.attempt_intubation("dl")
    assert any(r.label == "esophageal_intubation" for r in ev.actions.records if r.action == ACTION_EVENT_START)
    ev.perform_fona()
    assert any(r.label == "esophageal_intubation" for r in ev.actions.records if r.action == ACTION_EVENT_STOP)

    ev.give_spinal("bupivacaine", 50.0)
    assert any(r.label == "high_spinal" for r in ev.actions.records if r.action == ACTION_EVENT_START)

    ev.arm_protamine_reaction(True)
    ev.give_drug_bolus("protamine", 50.0)
    assert any(r.label == "protamine_reaction" for r in ev.actions.records if r.action == ACTION_EVENT_START)

    # Low-flow acidemia does not inflate resting-pH arterial iCa used for hyperkalemia membrane threshold.
    fb = ev.hemo.fluids
    fb.sync_circulation(0.15, ph=7.05)
    assert fb.arterial_ionized_calcium(7.05) > fb.arterial_ionized_calcium(fb.resting_ph) + 0.10
    assert fb.arterial_ionized_calcium(fb.resting_ph) == pytest.approx(fb.calcium_set_point, abs=0.05)

    # Hypothermic awake startup equilibrates endogenous NE without a cold-clearance startup overshoot.
    cold_ne = engine_factory(
        config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1), start=True, baseline_temp=32.0,
    )
    ne0 = cold_ne.pk["nore"].state.c1
    advance_time(cold_ne, 60.0, dt=1.0)
    assert cold_ne.pk["nore"].state.c1 == pytest.approx(ne0, rel=0.02)

    # Negative osmotic plasma shifts never reverse sign if blood volume sits below min_bv.
    cold_ne.hemo.blood_volume = 1200.0
    bv_low = cold_ne.hemo.blood_volume
    cold_ne.hemo.apply_osmotic_plasma_shift(-50.0)
    assert cold_ne.hemo.blood_volume <= bv_low
