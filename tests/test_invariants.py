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
    ("term_pregnancy", dict(age=30.0, weight=65.0, height=165.0, sex="female", gestational_age_weeks=39.0, position="left_lateral")),
]

CORE_FINITE_FIELDS = (
    "map", "sbp", "dbp", "hr", "co", "svr", "spo2", "pao2", "pa_co2",
    "ph", "hco3", "base_excess", "lactate", "potassium", "ionized_calcium",
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
    prop0, roc0 = eng.state.propofol_ce, eng.state.roc_ce
    eng.give_saline(1500.0)
    eng.give_fluid(1000.0)
    eng.give_blood(700.0)
    eng.give_calcium_chloride(1000.0)
    eng.give_drug_bolus("epi", 1000.0)
    eng.give_drug_bolus("roc", 70.0)
    eng.give_drug_bolus("propofol", 140.0)
    advance_time(eng, 15.0, dt=0.5)
    abg1 = eng.draw_blood_gas()
    assert abs(abg1.ph - abg0.ph) < 0.005
    assert abs(abg1.pa_co2 - abg0.pa_co2) < 0.25
    assert abg1.hco3 == pytest.approx(abg0.hco3, abs=1e-5)
    assert abs(abg1.sao2 - abg0.sao2) < 0.1
    assert abg1.sodium == pytest.approx(abg0.sodium, abs=1e-6)
    assert abg1.potassium == pytest.approx(abg0.potassium, abs=1e-6)
    assert abg1.chloride == pytest.approx(abg0.chloride, abs=1e-6)
    assert abg1.ionized_calcium == pytest.approx(abg0.ionized_calcium, abs=1e-6)
    assert abg1.lactate == pytest.approx(abg0.lactate, abs=1e-6)
    assert abg1.hb == pytest.approx(abg0.hb, abs=1e-9)
    assert abs(eng.state.propofol_ce - prop0) < 0.05
    assert abs(eng.state.roc_ce - roc0) < 0.05

    # Starting chest compressions restores circulation and delivers queued drugs/calcium.
    eng.set_chest_compressions(True)
    advance_time(eng, 45.0, dt=0.5)
    abg2 = eng.draw_blood_gas()
    assert eng.state.epi_ce > 1.0
    assert eng.state.roc_ce > roc0 + 0.5
    assert abg2.ionized_calcium > abg1.ionized_calcium + 0.05
    assert abg2.lactate > abg1.lactate + 0.05


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
