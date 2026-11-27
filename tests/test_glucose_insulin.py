"""Unit and physiological integration tests for glucose, insulin, and dextrose."""

from __future__ import annotations

import math

import pytest

from anasim.cli import build_models_from_config
from anasim.core.action_log import ACTION_FLUID, ACTION_GLUCOSE
from anasim.core.clinical_actions import check_drug_administration
from anasim.core.engine import SimulationEngine
from anasim.core.runtime import step_simulation
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.patient.pk_models import InsulinPK
from anasim.physiology.glucose import GLUCOSE_ANALYSIS_TIME_S, MG_DL_PER_MMOL_L


def _fast_advance(engine: SimulationEngine, seconds: float, dt: float = 2.0) -> None:
    steps = max(1, round(seconds / dt))
    h = seconds / steps
    engine.start()
    for _ in range(steps):
        step_simulation(engine, h, update_monitors=False)
        engine.state.time = round(engine.state.time + h, 9)
    engine.stop()


def test_patient_baseline_glucose_and_type2_diabetes_validation():
    healthy = Patient()
    assert healthy.type2_diabetes is False
    assert healthy.baseline_glucose == pytest.approx(90.0)

    diabetic = Patient(type2_diabetes=True)
    assert diabetic.type2_diabetes is True
    assert diabetic.baseline_glucose == pytest.approx(150.0)

    custom = Patient(type2_diabetes=True, baseline_glucose=240.0)
    assert custom.baseline_glucose == pytest.approx(240.0)

    cli_patient, _ = build_models_from_config({"type2_diabetes": True, "baseline_glucose": 185.0})
    assert cli_patient.type2_diabetes is True
    assert cli_patient.baseline_glucose == pytest.approx(185.0)

    for bad_glucose in (20.0, 500.0, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="baseline_glucose"):
            Patient(baseline_glucose=bad_glucose)
    with pytest.raises(ValueError, match="type2_diabetes"):
        Patient(type2_diabetes="yes")  # type: ignore[arg-type]


def test_insulin_pk_sherwin_plateau_and_organ_impairment():
    """Sherwin 1974 1 mU/kg/min infusion plateaus near 90-100 uU/mL; CKD and cirrhosis lower clearance."""
    healthy = InsulinPK(Patient(weight=70.0, height=170.0, sex="male"))
    esrd = InsulinPK(Patient(weight=70.0, height=170.0, sex="male", renal_function=0.05))
    cirrhosis = InsulinPK(Patient(weight=70.0, height=170.0, sex="male", hepatic_function=0.5))

    assert esrd.cl1_base < healthy.cl1_base * 0.80
    assert cirrhosis.cl1_base < healthy.cl1_base * 0.80

    # 1 mU/kg/min = 70 mU/min = 70/60 mU/s for 70 kg adult (Sherwin 1974: ~93 uU/mL plateau).
    rate_mu_s = 70.0 / 60.0
    for _ in range(180):
        healthy.step(60.0, rate_mu_s)
        esrd.step(60.0, rate_mu_s)
        cirrhosis.step(60.0, rate_mu_s)
    assert 75.0 <= healthy.state.c1 <= 105.0
    assert esrd.state.c1 > healthy.state.c1 * 1.20
    assert cirrhosis.state.c1 > healthy.state.c1 * 1.20


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"type2_diabetes": True},
        {"renal_function": 0.05},
        {"hepatic_function": 0.5},
        {"type2_diabetes": True, "baseline_glucose": 220.0},
    ],
)
def test_resting_glucose_and_electrolytes_stay_drift_free(kwargs):
    eng = SimulationEngine(Patient(**kwargs), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    g0, k0, na0 = eng.state.glucose, eng.state.potassium, eng.state.sodium
    _fast_advance(eng, 1800.0, dt=2.0)
    assert abs(eng.state.glucose - g0) < 2.0
    assert abs(eng.state.potassium - k0) < 0.05
    assert abs(eng.state.sodium - na0) < 0.3


def test_insulin_potassium_shift_preserved_in_uremia_and_additive_with_epinephrine():
    """Alvestrand 1984 / Allon and Copkney 1990: insulin lowers K+ equally in uremia despite
    blunted glucose uptake, and is additive with beta-2 adrenergic stimulation (modeled here with IV epinephrine)."""
    healthy = SimulationEngine(Patient(), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    esrd = SimulationEngine(Patient(renal_function=0.05), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    combo = SimulationEngine(Patient(renal_function=0.05), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))

    # Clamp at 1 mU/kg/min (4.2 U/hr) for 90 min while holding euglycemia.
    for eng in (healthy, esrd, combo):
        eng.set_drug_rate("insulin", 4.2)
    combo.set_drug_rate("epi", 3.0)

    for _ in range(90 * 30):
        for eng in (healthy, esrd, combo):
            eng.hemo.fluids.glucose_mmol = (90.0 / MG_DL_PER_MMOL_L) * eng.hemo.fluids.ecf_l
            step_simulation(eng, 2.0, update_monitors=False)
            eng.state.time += 2.0

    dk_healthy = healthy.state.potassium - 4.0
    dk_esrd = esrd.state.potassium - 4.0
    dk_combo = combo.state.potassium - 4.0
    # Potassium uptake is preserved in uremia despite impaired glucose disposal (Alvestrand 1984).
    assert dk_healthy <= -0.75
    assert dk_esrd <= -0.75
    assert abs(dk_esrd - dk_healthy) < 0.30
    assert esrd.glucose.state.rd_mg_kg_min < healthy.glucose.state.rd_mg_kg_min * 0.78
    # Beta-2 agonist adds to insulin K+ lowering (albuterol in Allon and Copkney 1990; IV epinephrine here).
    assert dk_combo < dk_esrd - 0.15


def test_dextrose_fluids_osmotic_sodium_shift_and_d5w_free_water_redistribution():
    """Hillier 1999 osmotic Na+ shift and Sjöstrand 2001 D5W ICF shift & diuresis vs balanced crystalloid (LR)."""
    d5w = SimulationEngine(Patient(), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    lr = SimulationEngine(Patient(), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    icf_d5w_0 = d5w.hemo.fluids.icf_l
    icf_lr_0 = lr.hemo.fluids.icf_l
    d5w._queue_infusion(1000.0, 1000.0 / 45.0, hematocrit=0.0, label="d5w")
    lr._queue_infusion(1000.0, 1000.0 / 45.0, hematocrit=0.0, label="crystalloid")

    _fast_advance(d5w, 45.0 * 60.0, dt=2.0)
    _fast_advance(lr, 45.0 * 60.0, dt=2.0)
    assert d5w.state.glucose > 220.0
    assert d5w.state.sodium < lr.state.sodium - 4.0

    _fast_advance(d5w, 75.0 * 60.0, dt=2.0)
    _fast_advance(lr, 75.0 * 60.0, dt=2.0)
    # Glucose enters cells, pulling free water into ICF and yielding more urine than LR.
    assert (d5w.hemo.fluids.icf_l - icf_d5w_0) > (lr.hemo.fluids.icf_l - icf_lr_0) + 0.35
    assert d5w.state.urine_out_ml > lr.state.urine_out_ml * 1.3
    assert d5w.state.glucose < 140.0


def test_renal_glycosuria_and_osmotic_diuresis_healthy_vs_t2d():
    """DeFronzo 2013: T2D has higher TmG (threshold and osmotic diuresis are calibrated in AnaSim)."""
    normo = SimulationEngine(Patient(baseline_glucose=90.0), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    hyper_h = SimulationEngine(Patient(baseline_glucose=320.0), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    hyper_d = SimulationEngine(
        Patient(type2_diabetes=True, baseline_glucose=320.0),
        SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0),
    )

    assert normo.glucose.state.glycosuria_mg_min == 0.0
    assert normo.glucose.state.osmotic_diuresis_ml_min == 0.0
    assert hyper_h.glucose.state.glycosuria_mg_min > 50.0
    assert hyper_h.glucose.state.osmotic_diuresis_ml_min > 1.0
    assert 10.0 < hyper_d.glucose.state.glycosuria_mg_min < hyper_h.glucose.state.glycosuria_mg_min

    _fast_advance(normo, 600.0, dt=2.0)
    _fast_advance(hyper_h, 600.0, dt=2.0)
    assert hyper_h.state.urine_out_ml > normo.state.urine_out_ml + 20.0


def test_neuroglycopenic_dysfunction_from_insulin_overdose_and_d10w_vs_d50w_rescue():
    """Pramming 1988 / Mitrakou 1991 / Moore and Woollard 2005: IV insulin overdose causes severe
    hypoglycemia with neuroglycopenic cortical dysfunction and unresponsiveness that recovers on D10W or D50W."""
    d10 = SimulationEngine(Patient(), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    d50 = SimulationEngine(Patient(), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    for eng in (d10, d50):
        eng.give_drug_bolus("insulin", 25.0)
        _fast_advance(eng, 1500.0, dt=2.0)
        assert eng.state.glucose < 35.0
        assert eng.cerebral.hypoglycemic_dysfunction > 0.40
        assert eng.cerebral.cortical_function < 0.70
        assert eng.cerebral.unresponsiveness > 0.40

    d10.give_d10w(100.0)  # 10 g dextrose
    d50.give_d50w(50.0)   # 25 g dextrose
    assert d10.actions.total(ACTION_FLUID, "d10w") == pytest.approx(100.0)
    assert d50.actions.total(ACTION_FLUID, "d50w") == pytest.approx(50.0)
    _fast_advance(d10, 600.0, dt=2.0)
    _fast_advance(d50, 600.0, dt=2.0)

    # Both reverse neuroglycopenia cleanly, while D10W avoids excessive rebound hyperglycemia (Moore and Woollard 2005).
    assert d10.cerebral.unresponsiveness < 0.25
    assert d50.cerebral.unresponsiveness < 0.25
    assert 55.0 <= d10.state.glucose <= 130.0
    assert d50.state.glucose > d10.state.glucose + 40.0

    # Starting hypoglycemic (40 mg/dL) must not permanently disable beta-cell insulin secretion after rescue.
    hypo_start = SimulationEngine(Patient(baseline_glucose=40.0), SimulationConfig(dt=2.0, maintenance_fluid_ml_hr=0.0))
    assert hypo_start.glucose.state.endogenous_insulin_mu_l == 0.0
    hypo_start.give_d50w(50.0)
    hypo_start.give_d50w(50.0)
    _fast_advance(hypo_start, 900.0, dt=2.0)
    assert hypo_start.glucose.state.endogenous_insulin_mu_l > 40.0


def test_perioperative_modulators_surgery_sepsis_cpb_epinephrine_and_sagm_prbc():
    """Schricker 2004, Revelly 2005, Lehot 1992, Clutter 1980, Kolterman 1981, and Rabcuka 2025."""
    # 1. Schricker 2004: sustained surgical stimulation impairs EGP suppression on dextrose.
    awake = SimulationEngine(Patient(), SimulationConfig(dt=2.0, maintenance_fluid_ml_hr=0.0))
    surg = SimulationEngine(
        Patient(),
        SimulationConfig(
            dt=2.0, mode="steady_state", maint_type="balanced",
            disturbance_profile="stim_sustained_surgery", maintenance_fluid_ml_hr=0.0,
        ),
    )
    for eng in (awake, surg):
        eng._queue_infusion(168.0, 1.4, hematocrit=0.0, label="d10w")
    _fast_advance(awake, 7200.0, dt=2.0)
    _fast_advance(surg, 7200.0, dt=2.0)
    assert surg.state.glucose > awake.state.glucose + 30.0
    assert surg.glucose.state.egp_mg_kg_min > awake.glucose.state.egp_mg_kg_min + 0.4

    # 2. Revelly 2005: sepsis doubles EGP and produces stress hyperglycemia with elevated insulin.
    septic = SimulationEngine(Patient(), SimulationConfig(dt=2.0, maintenance_fluid_ml_hr=0.0))
    egp0 = septic.glucose.state.egp_mg_kg_min
    septic.start_sepsis()
    _fast_advance(septic, 5400.0, dt=2.0)
    assert septic.state.glucose > 120.0
    assert septic.state.insulin_mu_l > 20.0
    assert septic.glucose.state.egp_mg_kg_min > egp0 * 1.25

    # 3. Lehot 1992: hypothermic CPB suppresses insulin secretion, rebounding sharply on rewarming.
    cpb = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", dt=2.0, maintenance_fluid_ml_hr=0.0))
    cpb.give_drug_bolus("heparin", 21000.0)
    cpb.start_cpb(blood_flow_l_min=4.5, sweep_gas_l_min=3.0, water_target_c=25.5, prime_volume_ml=1400.0)
    _fast_advance(cpb, 2700.0, dt=2.0)
    cold_ins = cpb.state.insulin_mu_l
    cold_bg = cpb.state.glucose
    assert cold_ins < 9.5
    cpb.set_ecmo_heater_cooler(True, 37.5)
    _fast_advance(cpb, 2700.0, dt=2.0)
    assert cpb.state.insulin_mu_l > cold_ins * 2.0
    assert cpb.state.glucose > cold_bg + 15.0

    # 4. Clutter 1980 / Marker 1998: epinephrine raises blood glucose.
    epi_eng = SimulationEngine(Patient(), SimulationConfig(dt=2.0, maintenance_fluid_ml_hr=0.0))
    epi_eng.set_drug_rate("epi", 2.5)
    _fast_advance(epi_eng, 3600.0, dt=2.0)
    assert epi_eng.state.glucose > 115.0

    # 5. Kolterman 1981: T2D reduces insulin-stimulated glucose disposal.
    t2d = SimulationEngine(Patient(type2_diabetes=True), SimulationConfig(dt=2.0, maintenance_fluid_ml_hr=0.0))
    ctrl = SimulationEngine(Patient(), SimulationConfig(dt=2.0, maintenance_fluid_ml_hr=0.0))
    t2d.set_drug_rate("insulin", 4.2)
    ctrl.set_drug_rate("insulin", 4.2)
    for _ in range(90 * 30):
        t2d.hemo.fluids.glucose_mmol = (150.0 / MG_DL_PER_MMOL_L) * t2d.hemo.fluids.ecf_l
        ctrl.hemo.fluids.glucose_mmol = (90.0 / MG_DL_PER_MMOL_L) * ctrl.hemo.fluids.ecf_l
        step_simulation(t2d, 2.0, update_monitors=False)
        step_simulation(ctrl, 2.0, update_monitors=False)
    assert t2d.glucose.state.rd_mg_kg_min < ctrl.glucose.state.rd_mg_kg_min * 0.78

    # 6. Rabcuka 2025: day-21 SAGM PRBC supernatant carries 21.8 mmol/L glucose.
    prbc = SimulationEngine(Patient(), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    g_before = prbc.hemo.fluids.glucose_mmol
    prbc.hemo.add_volume(600.0, hematocrit=0.55, label="blood")
    expected_added = 0.600 * (1.0 - 0.55) * 21.8
    assert prbc.hemo.fluids.glucose_mmol - g_before == pytest.approx(expected_added, abs=1e-6)


def test_poc_glucose_abg_glucose_and_insulin_safety_warnings():
    eng = SimulationEngine(Patient(baseline_glucose=185.0), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    poc = eng.measure_glucose()
    abg = eng.draw_blood_gas()
    assert poc.glucose == pytest.approx(185.0, abs=0.5)
    assert poc.ready_at == pytest.approx(poc.drawn_at + GLUCOSE_ANALYSIS_TIME_S)
    assert abg.glucose == pytest.approx(185.0, abs=0.5)
    assert eng.actions.since_case(ACTION_GLUCOSE)[-1].amount == pytest.approx(185.0, abs=0.5)

    # No warning when normoglycemic/hyperglycemic and normokalemic.
    assert check_drug_administration(eng, "insulin") is None

    # Hypoglycemia triggers warning.
    hypo = SimulationEngine(Patient(baseline_glucose=58.0), SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0))
    fb_hypo = check_drug_administration(hypo, "insulin")
    assert fb_hypo is not None and fb_hypo.code == "insulin_in_hypoglycemia" and not fb_hypo.blocked

    # Hypokalemia triggers warning.
    eng.state.potassium = 3.1
    fb_k = check_drug_administration(eng, "insulin")
    assert fb_k is not None and fb_k.code == "insulin_in_hypokalemia" and not fb_k.blocked

    # Giving an insulin bolus while paused updates insulin_mu_l without clobbering active EGP/Rd fluxes.
    eng.start_sepsis()
    _fast_advance(eng, 600.0, dt=2.0)
    egp_before, rd_before = eng.glucose.state.egp_mg_kg_min, eng.glucose.state.rd_mg_kg_min
    eng.give_drug_bolus("insulin", 2.0)
    assert eng.glucose.state.egp_mg_kg_min == pytest.approx(egp_before, abs=1e-9)
    assert eng.glucose.state.rd_mg_kg_min == pytest.approx(rd_before, abs=1e-9)
    assert eng.state.insulin_mu_l > 500.0

    # Case timeline descriptions format dextrose fluids and POC glucose checks without KeyError.
    from anasim.core.action_log import ActionRecord
    from anasim.scenarios.runtime import _describe

    assert _describe(ActionRecord(10.0, ACTION_FLUID, "d5w", 250.0)) == "250 mL D5W started"
    assert _describe(ActionRecord(12.0, ACTION_FLUID, "d10w", 100.0)) == "100 mL D10W (10 g) started"
    assert _describe(ActionRecord(14.0, ACTION_FLUID, "d50w", 50.0)) == "50 mL D50W (25 g) started"
    assert _describe(ActionRecord(16.0, ACTION_GLUCOSE, "poc", 185.0)) == "Point-of-care glucose checked (185 mg/dL)"


def test_osmotic_water_plasma_partitioning_and_hyperglycemic_diuresis_refill():
    """Acute hyperglycemic osmotic water shifts enter plasma without adding fake solute, and diuresis draws from interstitial excess."""
    eng = SimulationEngine(
        Patient(baseline_glucose=90.0),
        SimulationConfig(dt=1.0, maintenance_fluid_ml_hr=0.0, rng_seed=3),
    )
    bv0 = eng.hemo.blood_volume
    na0 = eng.hemo.fluids.amounts["sodium"]
    k0 = eng.hemo.fluids.amounts["potassium"]
    cl0 = eng.hemo.fluids.amounts["chloride"]

    # Raise ECF glucose directly by 25 mmol/L (~450 mg/dL) with no fluid volume added to isolate the osmotic free-water shift.
    eng.hemo.fluids.glucose_mmol += 25.0 * eng.hemo.fluids.ecf_l
    _fast_advance(eng, 120.0, dt=1.0)

    # Free-water translocation into ECF partitions ~plasma_fraction into blood volume and dilutes [Na+] without creating Na/K/Cl solute.
    initial_isf_l = eng.hemo.fluids.initial_ecf_l - eng.hemo.fluids.initial_plasma_l
    isf_excess_after_shift = (eng.hemo.fluids.interstitial_l - initial_isf_l) * 1000.0
    assert eng.hemo.total_osmotic_plasma_shift_ml > 50.0
    assert eng.hemo.blood_volume > bv0 + 30.0
    assert isf_excess_after_shift > 100.0
    assert eng.hemo.fluids.amounts["sodium"] <= na0 + 1e-6
    assert eng.hemo.fluids.amounts["sodium"] + eng.hemo.fluids.amounts["potassium"] <= na0 + k0 + 1e-6
    assert eng.hemo.fluids.amounts["chloride"] <= cl0 + 1e-6
    assert eng.hemo.fluids.concentration("sodium") < 138.0

    # Continued hyperglycemic osmotic diuresis depletes urine while interstitial excess buffers blood volume via Starling refill.
    uo_before = eng.hemo.total_urine_out_ml
    _fast_advance(eng, 1800.0, dt=2.0)
    uo_delta = eng.hemo.total_urine_out_ml - uo_before
    isf_excess_after_diuresis = (eng.hemo.fluids.interstitial_l - initial_isf_l) * 1000.0
    assert uo_delta > 150.0
    assert isf_excess_after_diuresis < isf_excess_after_shift
    assert eng.hemo.blood_volume > 4000.0

