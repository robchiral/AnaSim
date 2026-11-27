"""Sodium bicarbonate (8.4% NaHCO3) acid-base, CO2, hemodynamic, LAST, and safety tests."""

import copy
import math

import pytest

from anasim.core.action_log import ACTION_DRUG_BOLUS, ActionRecord
from anasim.core.clinical_actions import check_drug_administration
from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology.toxicity import LocalAnestheticToxicity
from anasim.scenarios.runtime import _describe


def test_bicarbonate_stoichiometry_conservation_channel_and_validation(engine_factory):
    """8.4% NaHCO3 adds exact Na+, strong-ion base, and CO2 carbon on its own IV channel."""
    eng = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
    )
    for bad in (0.0, -10.0, math.nan, math.inf, 501.0):
        with pytest.raises(ValueError):
            eng.give_sodium_bicarbonate(bad)

    transport = eng.resp.gas_transport
    fluids = eng.hemo.fluids
    na0 = fluids.amounts["sodium"]
    base0 = transport.base_meq
    co2_0 = transport.co2_l
    icf0 = fluids.icf_l

    # Direct 50 mL (50 mEq) infusion conserves Na+, buffer base, and CO2 carbon exactly.
    eng.hemo.add_volume(50.0, label="bicarbonate")
    eng.resp.update_fluid_balance(eng.hemo.hb_conc, eng.hemo.get_hematocrit(), eng.hemo.blood_volume)
    fluids._equilibrate_cell_water()
    assert fluids.amounts["sodium"] - na0 == pytest.approx(50.0, abs=1e-9)
    assert transport.base_meq - base0 == pytest.approx(50.0, abs=1e-9)
    assert (transport.co2_l - co2_0) / 0.022414 == pytest.approx(50.0, rel=1e-5)
    # Hyperosmolar 2000 mOsm/L load pulls intracellular water into ECF.
    assert fluids.icf_l < icf0 - 0.12

    # Dedicated IV channel delivers immediately even while a 1 L saline bag is running.
    eng2 = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
    )
    eng2.give_saline(1000.0)
    eng2.give_sodium_bicarbonate(50.0)
    assert eng2.actions.total(ACTION_DRUG_BOLUS, "sodium_bicarbonate") == pytest.approx(50.0)
    assert _describe(ActionRecord(12.0, ACTION_DRUG_BOLUS, "sodium_bicarbonate", 50.0)) == (
        "Sodium bicarbonate 50 mEq given"
    )
    for _ in range(30):
        eng2.step(1.0)
    # 50 mEq at 100 mL/min finishes within 30 s while saline still has >900 mL queued.
    assert not any(inf.label == "bicarbonate" for inf in eng2.pending_infusions)
    assert any(inf.label == "saline" and inf.remaining_ml > 900.0 for inf in eng2.pending_infusions)


def test_bicarbonate_acute_etco2_spike_and_apneic_co2_retention(engine_factory):
    """Fixed-ventilation 50 mEq push spikes EtCO2 promptly; apnea traps generated CO2."""
    vented = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
    )
    for _ in range(120):
        vented.step(1.0)
    ctrl = copy.deepcopy(vented)
    vented.give_sodium_bicarbonate(50.0)
    peak_delta_etco2 = 0.0
    peak_t = 0
    for sec in range(1, 181):
        vented.step(1.0)
        ctrl.step(1.0)
        delta = vented.state.etco2 - ctrl.state.etco2
        if delta > peak_delta_etco2:
            peak_delta_etco2 = delta
            peak_t = sec
    assert 4.0 <= peak_delta_etco2 <= 12.0
    assert 20 <= peak_t <= 60

    # Without ventilation (apnea), the CO2 load cannot be exhaled and PaCO2/PvCO2 climb sharply.
    apnea_ctrl = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
    )
    apnea_ctrl.set_vent_power(False)
    apnea_ctrl.give_drug_bolus("roc", 50.0)
    for _ in range(30):
        apnea_ctrl.step(1.0)
    apnea_bicarb = copy.deepcopy(apnea_ctrl)
    apnea_bicarb.give_sodium_bicarbonate(100.0)
    for _ in range(180):
        apnea_ctrl.step(1.0)
        apnea_bicarb.step(1.0)
    assert apnea_bicarb.state.pa_co2 > apnea_ctrl.state.pa_co2 + 4.0
    assert apnea_bicarb.resp._blood_pressures[2] > apnea_ctrl.resp._blood_pressures[2] + 4.0


def test_bicarbonate_rate_dependent_hyperosmolar_and_alkalemic_vasodilation(engine_factory):
    """Rapid hypertonic bolus briefly dips SVR and MAP (canine data in Huseby and Gumprecht 1981)
    while slow infusion is mild; calibrated alkalemia above pH 7.50 reduces SVR (anchored to
    halothane-volunteer data in Kaplan 1975)."""
    base = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
    )
    for _ in range(120):
        base.step(1.0)
    rapid = copy.deepcopy(base)
    slow = copy.deepcopy(base)

    # Rapid 1 mEq/kg (70 mEq) bolus over 5 s vs slow 60 s push.
    rapid_min_svr = rapid.state.svr
    rapid_min_map = rapid.state.map
    slow_min_factor = 1.0
    for sec in range(120):
        if sec < 5:
            rapid.hemo.add_volume(70.0 / 5.0, label="bicarbonate")
        if sec < 60:
            slow.hemo.add_volume(70.0 / 60.0, label="bicarbonate")
        rapid.step(1.0)
        slow.step(1.0)
        if sec < 30:
            rapid_min_svr = min(rapid_min_svr, rapid.state.svr)
            rapid_min_map = min(rapid_min_map, rapid.state.map)
        slow_min_factor = min(slow_min_factor, slow.hemo._hyperosmolar_svr_factor)
    assert rapid_min_svr < base.state.svr * 0.90
    assert rapid_min_map < base.state.map * 0.92
    # Transient hyperosmolar vasodilation recovers by 2 min and stays much milder on a 60 s push.
    assert rapid.state.svr > rapid_min_svr + 0.08 * base.state.svr
    assert slow_min_factor > 0.92

    # Calibrated alkalemia above pH 7.50 produces direct alkalemic vasodilation.
    alk = copy.deepcopy(base)
    alk.hemo.step(1.0, 0.0, 0.0, 0.0, pit=-1.4, paco2=40.0, pao2=95.0, ph=7.40, ph_37=7.40)
    svr_norm = alk.hemo._alkalosis_vasodilation
    alk.hemo.step(1.0, 0.0, 0.0, 0.0, pit=-1.4, paco2=40.0, pao2=95.0, ph=7.65, ph_37=7.65)
    assert svr_norm == pytest.approx(1.0)
    assert alk.hemo._alkalosis_vasodilation < 0.85


def test_bicarbonate_va_ecmo_zero_pulm_flow_floor_acidosis_zero_flow_rosc_and_reset(engine_factory):
    """Edge-case safety checks: zero pulmonary flow on VA-ECMO/CPB isolates pool 0,
    direct NaHCO3 bolus during floor-clamped acidosis stays inside pH bounds, immediate VFIB
    isolates arterial Na/glucose on step 1 without post-ROSC shock, and reset clears pending queues."""
    from anasim.core.enums import RhythmType
    from anasim.scenarios.base import preroll_scenario

    # 1. VA-ECMO during VFIB (zero native pulmonary flow, high ventilation) + 500 mEq NaHCO3 stays stable.
    ecmo = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        start=True,
    )
    ecmo.set_rhythm(RhythmType.VFIB)
    ecmo.start_va_ecmo(blood_flow_l_min=4.5, sweep_gas_l_min=4.0, sweep_fio2=1.0, prime_volume_ml=0.0)
    ecmo.set_vent_settings(rr=30.0, vt=0.60, peep=5.0, ie="1:2", mode="VCV")
    base0_before = ecmo.resp.gas_transport.base[0]
    ecmo.give_sodium_bicarbonate(500.0)
    for _ in range(180):
        ecmo.step(1.0)
    assert ecmo.resp.gas_transport.base[0] == pytest.approx(base0_before, abs=0.05)
    assert 6.8 < ecmo.state.ph < 8.0 and ecmo.resp._blood_ph[0] < 9.5

    # 2. Direct single-step 50 mEq NaHCO3 during floor-clamped severe acidosis does not crash solver.
    acid = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        start=True,
    )
    for _ in range(25):
        acid.resp.remove_buffer_base(40.0)
        acid.step(1.0)
    assert acid.resp._unbuffered_acid_meq > 20.0
    acid.hemo.add_volume(50.0, label="bicarbonate")
    acid.step(1.0)
    assert math.isfinite(acid.state.ph) and 5.0 <= acid.state.ph <= 7.8

    # 3. Immediate VFIB + NaHCO3 + D50W has zero first-step arterial leak and no post-ROSC shock.
    arrest = engine_factory(
        config=SimulationConfig(mode="steady_state", dt=2.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        start=True,
    )
    abg0 = arrest.draw_blood_gas()
    poc0 = arrest.measure_glucose()
    arrest.set_rhythm(RhythmType.VFIB)
    arrest.give_sodium_bicarbonate(100.0)
    arrest.give_d50w(50.0)
    for _ in range(30):
        arrest.step(2.0)
    abg1 = arrest.draw_blood_gas()
    poc1 = arrest.measure_glucose()
    assert abg1.sodium == pytest.approx(abg0.sodium, abs=1e-6)
    assert abg1.glucose == pytest.approx(abg0.glucose, abs=1e-6)
    assert poc1.glucose == pytest.approx(poc0.glucose, abs=1e-6)
    arrest.set_rhythm(RhythmType.SINUS)
    arrest.step(1.0)
    assert arrest.hemo._hyperosmolar_svr_factor > 0.88

    # 4. initialize_state() and preroll_scenario(..., reset_fluid_totals=True) clear pending infusions.
    reset_eng = engine_factory(
        config=SimulationConfig(mode="awake", dt=1.0, rng_seed=1, maintenance_fluid_ml_hr=0.0),
        start=True,
    )
    reset_eng.give_sodium_bicarbonate(100.0)
    reset_eng.give_d50w(50.0)
    reset_eng.initialize_state()
    assert reset_eng.pending_infusions == []
    reset_eng.give_sodium_bicarbonate(100.0)
    preroll_scenario(reset_eng, 10.0, reset_fluid_totals=True)
    assert reset_eng.pending_infusions == []
    assert reset_eng.hemo._hyperosmolar_load == 0.0
    assert reset_eng.hemo._hyperosmolar_svr_factor == 1.0


def test_bicarbonate_reverses_local_anesthetic_conduction_block():
    """Alkalinization and extracellular Na+ loading attenuate lidocaine conduction block
    without replacing lipid emulsion for myocardial contractility recovery."""
    tox_base = LocalAnestheticToxicity(70.0)
    tox_bicarb = LocalAnestheticToxicity(70.0)
    common = dict(
        neural_ce=0.0,
        cardiac_ce=32.0,
        propofol_ce=0.0,
        midazolam_ce=0.0,
        mac_sevo=0.0,
        motor_strength=1.0,
        cortical_function=1.0,
    )
    tox_base.step(1.0, ph=7.40, sodium=140.0, **common)
    tox_bicarb.step(1.0, ph=7.56, sodium=146.0, **common)
    assert tox_base.conduction_block > 0.35
    assert tox_bicarb.conduction_block < tox_base.conduction_block * 0.55
    assert tox_bicarb.cardiac_stress < tox_base.cardiac_stress
    # Contractility depression above pH 7.40 still requires lipid emulsion.
    assert tox_bicarb.contractility_loss == pytest.approx(tox_base.contractility_loss)


def test_bicarbonate_clinical_safety_warnings():
    """Non-blocking warnings fire when giving NaHCO3 during alkalemia or severe hypernatremia."""
    eng = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
    assert check_drug_administration(eng, "sodium_bicarbonate") is None

    eng.state.ph = 7.55
    fb_alk = check_drug_administration(eng, "sodium_bicarbonate")
    assert fb_alk is not None and fb_alk.code == "bicarbonate_in_alkalemia" and not fb_alk.blocked

    eng.state.ph = 7.38
    eng.state.sodium = 158.0
    fb_na = check_drug_administration(eng, "sodium_bicarbonate")
    assert fb_na is not None and fb_na.code == "bicarbonate_in_hypernatremia" and not fb_na.blocked
