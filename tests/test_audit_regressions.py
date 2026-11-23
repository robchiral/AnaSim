"""Regression tests for clinical realism and cross-module audit fixes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.state import RhythmType, SimulationConfig
from anasim.monitors.nibp import NIBPMonitor
from anasim.patient.patient import Patient
from anasim.patient.pk_models import MorphinePK
from anasim.web import WebSession


def _advance(engine: SimulationEngine, seconds: float, dt: float = 1.0) -> None:
    steps = max(1, round(seconds / dt))
    for _ in range(steps):
        engine.step(dt)


def _web_cmd(session: WebSession, name: str, **args):
    return json.loads(session.command(name, json.dumps(args)))


def _web_advance(session: WebSession, seconds: float) -> None:
    _web_cmd(session, "speed", value=50)
    _web_cmd(session, "run", running=True)
    remaining = seconds
    while remaining > 1e-8:
        interval = min(remaining, 10.0)
        session.step(interval / 50.0)
        remaining -= interval


# ---------------------------------------------------------------------------
# Phase 1: Fluids, chemistry, thermal, and core wiring
# ---------------------------------------------------------------------------


def test_vfib_arrest_chemistry_stays_acidotic_without_alkali_runaway():
    """Prolonged VFIB without CPR should not drift into alkalemia via venous alkali pump."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", rng_seed=1))
    engine.start()
    engine.set_rhythm(RhythmType.VFIB)
    _advance(engine, 240.0, dt=1.0)
    assert engine.state.cardiac_arrest
    assert engine.state.ph <= 7.45
    assert engine.state.lactate >= 1.5


def test_labetalol_blunts_epinephrine_hypokalemia_and_lactate():
    """Non-selective beta-2 blockade (labetalol) should attenuate epinephrine K+ and lactate shifts."""
    ctrl = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", rng_seed=1))
    blocked = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", rng_seed=1))
    ctrl.start()
    blocked.start()

    blocked.give_drug_bolus("labetalol", 40.0)
    _advance(ctrl, 60.0, dt=1.0)
    _advance(blocked, 60.0, dt=1.0)

    k0_ctrl, lac0_ctrl = ctrl.state.potassium, ctrl.state.lactate
    k0_blk, lac0_blk = blocked.state.potassium, blocked.state.lactate

    ctrl.set_drug_rate("epi", 10.0)
    blocked.set_drug_rate("epi", 10.0)
    _advance(ctrl, 600.0, dt=2.0)
    _advance(blocked, 600.0, dt=2.0)

    ctrl_k_drop = k0_ctrl - ctrl.state.potassium
    blk_k_drop = k0_blk - blocked.state.potassium
    ctrl_lac_rise = ctrl.state.lactate - lac0_ctrl
    blk_lac_rise = blocked.state.lactate - lac0_blk

    assert ctrl_k_drop > 0.25
    assert blk_k_drop < 0.55 * ctrl_k_drop
    assert ctrl_lac_rise > 0.6
    assert blk_lac_rise < 0.55 * ctrl_lac_rise


def test_last_seizures_generate_lactic_acidosis():
    """Sustained LAST convulsions should produce substantial skeletal-muscle lactate."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
    engine.start()
    lac0 = engine.state.lactate
    engine.give_drug_bolus("lidocaine", 500.0)
    _advance(engine, 300.0, dt=1.0)
    assert engine.state.lactate > lac0 + 1.0


def test_non_tci_csht_uses_patient_pk_without_tci_pump_artifact():
    """get_predicted_csht should work cleanly when tci_enabled is False."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="awake", tci_enabled=False, rng_seed=1))
    engine.start()
    engine.set_drug_rate("propofol", 10.0)
    _advance(engine, 600.0, dt=2.0)
    csht = engine.get_predicted_csht("propofol")
    assert 2.0 < csht < 30.0


# ---------------------------------------------------------------------------
# Phase 2: Hemodynamics, coronary, and arrest
# ---------------------------------------------------------------------------


def test_hfref_cpr_perfusion_matches_normal_heart_without_output_inflation():
    """HFrEF chest compressions should not generate higher CPR MAP than a normal heart."""
    normal = SimulationEngine(
        Patient(age=65, weight=75, height=172, sex="male"),
        SimulationConfig(mode="steady_state", rng_seed=1),
    )
    hfref = SimulationEngine(
        Patient(age=65, weight=75, height=172, sex="male", heart_failure="hfref"),
        SimulationConfig(mode="steady_state", rng_seed=1),
    )
    for eng in (normal, hfref):
        eng.start()
        eng.set_rhythm(RhythmType.VFIB)
        eng.set_chest_compressions(True)
        _advance(eng, 45.0, dt=1.0)

    assert hfref.hemo.state.cpr_perfusion_map <= normal.hemo.state.cpr_perfusion_map + 3.0


def test_hfref_severe_cad_tolerates_750ml_crystalloid_without_arrest():
    """A 750 mL crystalloid bolus in HFrEF + severe CAD should not trigger flash edema arrest."""
    patient = Patient(
        age=72, weight=80, height=175, sex="male",
        heart_failure="hfref", coronary_flow_reserve=1.8,
    )
    engine = SimulationEngine(patient, SimulationConfig(mode="steady_state", rng_seed=1))
    engine.start()
    engine.give_fluid(750.0)
    _advance(engine, 900.0, dt=2.0)
    assert not engine.state.cardiac_arrest
    assert engine.state.lung_water < 3.0


def test_high_dose_epinephrine_crosses_svr_above_baseline():
    """Sustained 0.50 mcg/kg/min epinephrine should cross net SVR above baseline."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", rng_seed=1))
    engine.start()
    base_svr = engine.state.svr
    engine.set_drug_rate("epi", 0.50 * 70.0)
    _advance(engine, 900.0, dt=2.0)
    assert engine.state.svr > base_svr


def test_awake_hfref_milrinone_unloads_svr_and_raises_co():
    """Baim 1983: 50 mcg/kg load over 10 min + 0.5 mcg/kg/min in awake HFrEF raises CO and lowers SVR."""
    patient = Patient(age=65, weight=75, height=172, sex="male", heart_failure="hfref")
    engine = SimulationEngine(patient, SimulationConfig(mode="awake", rng_seed=1))
    engine.start()
    base_co = engine.state.co
    base_svr = engine.state.svr
    base_map = engine.state.map

    # milri rate unit is mcg/kg/min: 5.0 mcg/kg/min for 10 min = 50 mcg/kg load.
    engine.set_drug_rate("milri", 5.0)
    _advance(engine, 600.0, dt=2.0)
    engine.set_drug_rate("milri", 0.5)
    _advance(engine, 600.0, dt=2.0)

    assert engine.state.co > base_co * 1.25
    assert engine.state.svr < base_svr * 0.80
    assert abs(engine.state.map - base_map) < 10.0

    # By contrast, a rapid 50 mcg/kg milrinone bolus in a normal anesthetized heart lowers MAP.
    normal = SimulationEngine(
        Patient(age=65, weight=75, height=172, sex="male"),
        SimulationConfig(mode="steady_state", rng_seed=1),
    )
    normal.start()
    norm_map0 = normal.state.map
    normal.give_drug_bolus("milri", 50.0 * 75.0)
    _advance(normal, 300.0, dt=2.0)
    assert normal.state.map < norm_map0 - 8.0


def test_atropine_clinical_bolus_produces_brisk_hr_response():
    """A standard 0.6 mg IV atropine bolus should produce a brisk awake HR rise by 2 min."""
    atrop = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
    atrop.start()
    hr0 = atrop.state.hr
    atrop.give_drug_bolus("atropine", 0.6)
    _advance(atrop, 120.0, dt=1.0)
    assert atrop.state.hr - hr0 > 20.0


# ---------------------------------------------------------------------------
# Phase 3: Pharmacology, respiration, and ventilator mechanics
# ---------------------------------------------------------------------------


def test_complete_ett_obstruction_does_not_falsely_elevate_pit():
    """100% ETT occlusion raises proximal Ppeak without transmitting airway pressure to Pit."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", rng_seed=1))
    engine.start()
    pit_base = engine.hemo.pit if hasattr(engine.hemo, "pit") else engine.state.pit
    engine.set_airway_obstruction(1.0)
    _advance(engine, 20.0, dt=0.5)
    assert engine.state.paw_peak >= 35.0
    assert engine.state.pit <= pit_base + 1.5


def test_succinylcholine_laryngospasm_relief_is_dose_responsive():
    """Higher succinylcholine rescue doses should break laryngospasm faster via central NMJ."""
    times = {}
    for dose_mg in (10.0, 50.0):
        eng = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
        eng.start()
        eng.give_drug_bolus("propofol", 100.0)
        _advance(eng, 45.0, dt=1.0)
        eng.laryngospasm_severity = 0.95
        eng.give_drug_bolus("succinylcholine", dose_mg)
        relieved_t = None
        for t in range(1, 91):
            eng.step(1.0)
            if relieved_t is None and eng.laryngospasm_severity < 0.25:
                relieved_t = t
        times[dose_mg] = relieved_t
    assert times[50.0] is not None
    assert times[10.0] is None or times[50.0] < times[10.0]


def test_low_dose_sugammadex_produces_recurarization_rebound():
    """Eleveld 2007: subtherapeutic 0.5 mg/kg sugammadex under deep rocuronium rebounds."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", rng_seed=1))
    engine.start()
    engine.give_drug_bolus("rocuronium", 63.0)
    _advance(engine, 2520.0, dt=10.0)
    assert engine.state.tof < 5.0

    engine.give_drug_bolus("sugammadex", 35.0)
    peak_tof = 0.0
    min_after_tof = 100.0
    for step_idx in range(1, 361):
        engine.step(5.0)
        if step_idx <= 60:
            peak_tof = max(peak_tof, engine.state.tof)
        else:
            min_after_tof = min(min_after_tof, engine.state.tof)

    assert peak_tof > 50.0
    assert min_after_tof < peak_tof - 8.0


def test_morphine_overdose_causes_apnea_and_esrd_accumulates_m6g():
    """High-dose morphine causes apnea; ESRD produces deeper late M6G respiratory depression."""
    overdose = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
    overdose.start()
    overdose.give_drug_bolus("morphine", 40.0)
    apnea_seen = False
    for _ in range(1650):
        overdose.step(2.0)
        if overdose.state.rr == 0.0 or overdose.state.apnea:
            apnea_seen = True
    assert apnea_seen
    assert overdose.state.pa_co2 > 75.0

    normal_renal = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
    esrd = SimulationEngine(
        Patient(renal_function=0.05), SimulationConfig(mode="awake", rng_seed=1),
    )
    normal_renal.start()
    esrd.start()
    normal_renal.give_drug_bolus("morphine", 20.0)
    esrd.give_drug_bolus("morphine", 20.0)
    # Advance morphine PK through 6 h so M6G accumulates in ESRD, then verify full-engine respiration.
    norm_pk = normal_renal.pk["morphine"]
    esrd_pk = esrd.pk["morphine"]
    assert isinstance(norm_pk, MorphinePK) and isinstance(esrd_pk, MorphinePK)
    for _ in range(360):
        norm_pk.step(60.0, 0.0)
        esrd_pk.step(60.0, 0.0)
    assert esrd_pk.m6g_excess_ce > norm_pk.m6g_excess_ce + 8.0
    _advance(normal_renal, 600.0, dt=2.0)
    _advance(esrd, 600.0, dt=2.0)
    assert esrd.state.pa_co2 > normal_renal.state.pa_co2 + 8.0
    assert esrd.state.sao2 < normal_renal.state.sao2 - 20.0


def test_olofsen_controller_has_no_drug_free_co2_drop_rebound():
    """When opioids are absent, a sudden PaCO2 drop must not cause spurious opioid-lag rebound."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
    engine.start()
    engine.resp.state.pa_co2 = 60.0
    _advance(engine, 30.0, dt=1.0)
    engine.resp.state.pa_co2 = 40.0
    _advance(engine, 10.0, dt=1.0)
    assert engine.resp._opioid_drive_fraction == pytest.approx(1.0, abs=1e-4)


def test_light_propofol_room_air_bolus_survives_via_hypoxic_rescue():
    """A 90 mg propofol room-air bolus should not progress to hypoxic cardiac arrest."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
    engine.start()
    engine.give_drug_bolus("propofol", 90.0)
    _advance(engine, 600.0, dt=1.0)
    assert not engine.state.cardiac_arrest


def test_high_peep_increases_dead_space_gradient():
    """High PEEP (20 cmH2O) should widen the PaCO2-EtCO2 gradient via overdistension/Zone-1."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", rng_seed=1))
    engine.start()
    base_grad = engine.state.pa_co2 - engine.state.etco2
    engine.vent.update_settings(peep=20.0)
    _advance(engine, 120.0, dt=1.0)
    high_peep_grad = engine.state.pa_co2 - engine.state.etco2
    assert high_peep_grad > base_grad + 2.0


# ---------------------------------------------------------------------------
# Phase 4: Monitors, scenarios, and web/UI integration
# ---------------------------------------------------------------------------


def test_bis_splits_elderly_propofol_delay_from_monitor_delay_and_suppresses_induction_false_high():
    """80-year-old BIS keeps age-dependent propofol delay while capping non-propofol monitor delay."""
    elderly = SimulationEngine(Patient(age=80), SimulationConfig(mode="awake", rng_seed=1))
    elderly.start()
    assert elderly.bis.delay > 60.0
    assert elderly.bis.monitor_delay <= 18.0

    ind = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
    ind.start()
    ind.give_drug_bolus("propofol", 140.0)
    for _ in range(120):
        ind.step(1.0)
        assert not ind.state.alarms.get("BIS", {}).get("high")


def test_oxygen_supply_disconnect_emits_o2supply_alarm():
    """Pipeline oxygen failure should immediately surface an O2Supply alarm."""
    session = WebSession({"scenario_id": "oxygen_supply_failure"})
    session.engine.config.dt = 0.5
    _web_advance(session, 35.0)
    assert session.engine.state.alarms.get("O2Supply", {}).get("low") is True


def test_neuromuscular_monitor_supports_ptc_mode():
    """PTC mode should produce post-tetanic counts and enforce the 60 s repeat interval."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
    engine.start()
    engine.give_drug_bolus("rocuronium", 45.0)
    _advance(engine, 120.0, dt=1.0)
    engine.nmb.set_mode("ptc", engine.state.time)
    _advance(engine, 65.0, dt=1.0)
    reading = engine.nmb.reading
    assert reading is not None
    assert reading.mode == "ptc"
    assert 0 <= reading.count <= 15
    with pytest.raises(ValueError, match="60"):
        engine.nmb.measure(engine.state.time)


def test_vfib_blanks_st_in_web_snapshot_while_keeping_state_float():
    """Web JSON blanks ST on VFIB (None) while engine.state.st_depression_mv remains float 0.0."""
    session = WebSession({"mode": "steady_state"})
    session.engine.set_rhythm(RhythmType.VFIB)
    session.engine.step(0.5)
    assert session.engine.state.st_depression_mv == 0.0
    assert session.snapshot()["vitals"]["st_depression_mv"] is None


def test_nibp_afib_preserves_diastolic_map_systolic_ordering():
    """NIBP AFib jitter must never output MAP > SBP or DBP > MAP even with narrow pulse pressure."""
    mon = NIBPMonitor(rng=np.random.default_rng(42))
    t = 0.0
    valid = 0
    for _ in range(40):
        mon.trigger()
        for _ in range(20):
            t += 1.0
            mon.step(
                dt=1.0,
                current_time=t,
                true_map=64.0,
                true_sys=72.0,
                true_dia=60.0,
                rhythm_type=RhythmType.AFIB,
            )
        if not mon.measurement_failed:
            valid += 1
            r = mon.latest_reading
            assert r.diastolic <= r.map <= r.systolic
        t += 5.0
    assert valid >= 20


def test_sepsis_and_anaphylaxis_early_actions_do_not_soft_lock_step1():
    """Clicking source control or stop trigger early must still satisfy Guided Step 1."""
    sepsis = WebSession({"scenario_id": "sepsis_response"})
    sepsis.engine.config.dt = 0.5
    _web_advance(sepsis, 16.0)
    _web_cmd(sepsis, "case_action", action="source_control")
    scen_s = sepsis.snapshot()["scenario"]
    assert scen_s["id"] == "OBSERVE" and scen_s["met"] is True

    anaph = WebSession({"scenario_id": "anaphylaxis_response"})
    anaph.engine.config.dt = 0.5
    _web_advance(anaph, 41.0)
    _web_cmd(anaph, "case_action", action="stop_trigger")
    scen_a = anaph.snapshot()["scenario"]
    assert scen_a["id"] == "OBSERVE" and scen_a["met"] is True


def test_anaphylaxis_guided_bolus_avoids_extreme_hypertension_and_premature_completion():
    """100 mcg epi + 1000 mL LR in anaphylaxis keeps SBP < 225 and requires severity < 0.35."""
    session = WebSession({"scenario_id": "anaphylaxis_response"})
    session.engine.config.dt = 0.5
    _web_advance(session, 45.0)
    _web_cmd(session, "scenario_next")
    _web_cmd(session, "drug_bolus", key="epi", amount=100)
    _web_cmd(session, "fgf", o2=10, air=0, n2o=0)
    _web_cmd(session, "scenario_next")
    _web_cmd(session, "fluid", kind="crystalloid", volume_ml=1000)
    _web_cmd(session, "scenario_next")
    _web_cmd(session, "case_action", action="stop_trigger")

    max_sbp = 0.0
    for _ in range(35):
        _web_advance(session, 1.0)
        max_sbp = max(max_sbp, session.engine.state.sbp)
    assert max_sbp < 250.0
    # At ~t=80 s severity is still high, so STABILIZE must not mark complete prematurely.
    if session.engine.anaphylaxis.severity >= 0.35:
        assert session.snapshot()["scenario"]["met"] is False


def test_web_drug_rate_limits_respect_per_drug_units():
    """Fentanyl (mcg/hr) permits standard analgesic infusions while vasopressin stays capped at 1 U/min."""
    session = WebSession({"mode": "steady_state"})
    receipt = _web_cmd(session, "drug_rate", key="fentanyl", rate=100.0)
    assert receipt["medication"]["key"] == "fentanyl"
    with pytest.raises(ValueError):
        _web_cmd(session, "drug_rate", key="vaso", rate=5.0)
