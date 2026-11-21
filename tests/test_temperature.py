import copy

import pytest

from anasim.core.projection import sync_monitor_baselines, thermal_hypnotic_ce
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology.thermal import ThermalModel


def _advance(engine, seconds: float, dt: float = 1.0) -> None:
    for _ in range(int(seconds / dt)):
        engine.step(dt)


def test_anesthetized_patient_cools_and_forced_air_rewarms(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state"), start=True)
    # Redistribution belongs to the hidden settle, not visible time.
    engine.step(0.1)
    assert engine.state.temp_c == pytest.approx(37.0, abs=0.005)

    _advance(engine, 20 * 60, dt=2.0)
    cooled = engine.state.temp_c
    assert cooled < 37.0

    engine.set_bair_hugger(43.0)
    _advance(engine, 20 * 60, dt=2.0)
    assert engine.state.temp_c > cooled


def test_hypothermia_lowers_paco2_at_fixed_ventilation(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state"))
    engine.vent.is_on = True
    engine.start()

    engine.state.temp_c = 37.0
    _advance(engine, 60, dt=2.0)
    normothermic = engine.state.pa_co2

    engine.state.temp_c = 30.0
    _advance(engine, 600, dt=2.0)
    assert engine.state.pa_co2 < normothermic


def test_hypothermia_slows_apneic_co2_rise(engine_factory):
    rises = []
    for temperature in (30.0, 37.0):
        engine = engine_factory(start=True, baseline_temp=temperature)
        engine.set_airway_mode("ETT")
        engine.set_fgf(10.0, 0.0)
        engine.set_vent_power(True)
        engine.give_drug_bolus("roc", engine.patient.weight)
        _advance(engine, 180, dt=2.0)
        # Apneic oxygenation keeps SaO2 up. Prior shivering and tissue stores
        # also affect the transient, so it need not scale directly with VCO2.
        engine.set_vent_power(False)
        start = engine.state.pa_co2
        _advance(engine, 120, dt=2.0)
        rises.append(engine.state.pa_co2 - start)
    assert rises[0] < 0.9 * rises[1]


def test_core_drop_after_induction(patient):
    """Matsukawa 1995: core temperature fell 1.6 ± 0.3 °C in the first hour and 2.8 ± 0.5 °C by 3 h."""
    thermal = ThermalModel(patient)
    temp_c = 37.0
    # Awake thermoregulation holds core temperature before induction.
    for _ in range(30):
        thermal.update_depth(temp_c, 0.0, 0.0)
        temp_c = thermal.step_temperature(60.0, temp_c, 0.0)
    assert temp_c == pytest.approx(37.0, abs=0.02)
    for _ in range(60):
        thermal.update_depth(temp_c, 3.5, 0.0)
        temp_c = thermal.step_temperature(60.0, temp_c, 0.0)
    assert 37.0 - temp_c == pytest.approx(1.6, abs=0.3)
    for _ in range(120):
        thermal.update_depth(temp_c, 3.5, 0.0)
        temp_c = thermal.step_temperature(60.0, temp_c, 0.0)
    assert 37.0 - temp_c == pytest.approx(2.8, abs=0.5)


def test_obese_redistribution_cooling_sepsis_fever_and_hypothermic_bradycardia(engine_factory):
    """Kurz 1995: obesity blunts first-hour redistribution cooling (~0.4-0.6 °C vs 1.6 °C);
    sepsis elevates core temperature; severe hypothermia (<33 °C) depresses resting sinus HR.
    """
    obese = ThermalModel(Patient(age=40, weight=150.0, height=165.0, sex="female"))
    temp_ob = 37.0
    for _ in range(60):
        obese.update_depth(temp_ob, 3.5, 0.0)
        temp_ob = obese.step_temperature(60.0, temp_ob, 0.0)
    assert 0.30 < 37.0 - temp_ob < 0.80

    sepsis = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    sepsis.start_sepsis()
    _advance(sepsis, 1200, dt=2.0)
    assert sepsis.state.temp_c > 37.12

    cold = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, baseline_temp=27.0)
    assert cold.state.hr < 55.0
    cold.step(1.0)
    assert cold.state.hr < 55.0


def test_balanced_mac_temp_parity_abg_warming_to_37c_and_awake_febrile_setpoint(engine_factory):
    """Balanced steady-state starts at uniform MAC across baseline temperatures;
    draw_blood_gas() anaerobically warms hypothermic samples to 37 °C without double-warming;
    awake febrile baseline_temp holds its setpoint without cooling toward 37 °C.
    """
    engines = [
        engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1),
            start=True,
            baseline_temp=temp,
        )
        for temp in (32.0, 37.0, 39.0)
    ]
    macs = [eng.state.mac for eng in engines]
    assert macs[0] == pytest.approx(macs[1], abs=0.02)
    assert macs[2] == pytest.approx(macs[1], abs=0.02)

    hypo = engines[0]
    abg = hypo.draw_blood_gas()
    assert abg.pa_co2 == pytest.approx(hypo.resp._blood_pressures[1], rel=1e-6)
    assert abg.ph == pytest.approx(hypo.resp._blood_ph[1], abs=1e-6)
    assert abg.pa_co2 > hypo.state.pa_co2 + 3.0
    assert abg.pao2 > hypo.state.pao2
    assert abg.ph < hypo.state.ph - 0.04

    febrile = ThermalModel(Patient(baseline_temp=38.5))
    temp_f = 38.5
    for _ in range(30):
        febrile.update_depth(temp_f, 0.0, 0.0)
        temp_f = febrile.step_temperature(60.0, temp_f, 0.0)
    assert temp_f == pytest.approx(38.5, abs=0.03)


def test_zero_flow_arrest_accelerates_passive_core_cooling(engine_factory):
    """Core cooling continues faster during zero-flow arrest than perfused normothermia."""
    perfused = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    arrested = copy.deepcopy(perfused)
    arrested.set_rhythm("VFIB")
    _advance(perfused, 600, dt=2.0)
    _advance(arrested, 600, dt=2.0)
    assert arrested.state.temp_c < perfused.state.temp_c - 0.05


def test_thermal_etomidate_age_invariance_and_ischemic_bis_baseline_seed(engine_factory):
    """Etomidate thermal redistribution equivalent does not double-scale with BIS age_factor,
    and sync_monitor_baselines seeds BIS cortical_function from cerebral ischemia.
    """
    young = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), age=25)
    elderly = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), age=80)
    young.state.etomidate_ce = elderly.state.etomidate_ce = 0.4
    assert thermal_hypnotic_ce(young) == pytest.approx(thermal_hypnotic_ce(elderly), rel=1e-9)

    young.cerebral.ischemia = 0.8
    sync_monitor_baselines(young)
    assert young.bis.cortical_function == pytest.approx(young.cerebral.cortical_function, rel=1e-9)


def test_hypothermia_preserves_etco2_gradient_and_alpha_stat_ionized_calcium(engine_factory):
    """During hypothermia, published EtCO2 stays at or below in-vivo PaCO2 and ionized calcium
    tracks 37 C alpha-stat pH rather than shifting artificially with in-vivo pH."""
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    ica_37 = engine.state.ionized_calcium
    engine.state.temp_c = 30.0
    _advance(engine, 10, dt=1.0)
    assert engine.state.etco2 <= engine.state.pa_co2
    assert engine.state.ionized_calcium == pytest.approx(ica_37, abs=0.05)


def test_hypothermia_cold_narcosis_and_bradypnea(engine_factory, advance_time):
    """Severe hypothermia (27.5 C) induces cold narcosis and hypothermic bradypnea."""
    cold = engine_factory(
        config=SimulationConfig(mode="awake", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
        baseline_temp=27.5,
    )
    advance_time(cold, 120.0, dt=1.0)
    assert cold.state.loc > 0.85
    assert cold.state.bis < 65.0
    assert cold.state.rr < 9.0
    assert cold.state.pa_co2 > 28.0


def test_iv_fluid_and_prbc_thermal_load_and_fluid_warmer(engine_factory, advance_time):
    """Unwarmed crystalloid and cold PRBC cool core temperature; enabling the fluid warmer prevents cooling."""
    unwarmed = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
    )
    warmed = copy.deepcopy(unwarmed)
    warmed.set_fluid_warmer(True)
    for eng in (unwarmed, warmed):
        eng.give_fluid(2000.0)
        eng.give_blood(600.0)
    advance_time(unwarmed, 1200.0, dt=2.0)
    advance_time(warmed, 1200.0, dt=2.0)
    assert unwarmed.state.temp_c < warmed.state.temp_c - 0.55


def test_hypothermia_slows_propofol_pk_and_rocuronium_recovery(engine_factory):
    """Leslie 1995 & Smeulers 1995: Mild hypothermia (33 C vs 37 C) raises propofol Cp and prolongs rocuronium."""
    warm = engine_factory(
        config=SimulationConfig(mode="awake", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
        baseline_temp=37.0,
    )
    cold = engine_factory(
        config=SimulationConfig(mode="awake", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
        baseline_temp=33.0,
    )
    for eng in (warm, cold):
        eng.set_airway_mode("ETT")
        eng.set_vent_settings(rr=12, vt=0.5, peep=5, ie="1:2", mode="VCV")
        eng.set_vent_power(True)
        eng.set_drug_rate("propofol", 100.0)
        eng.give_drug_bolus("roc", 0.6 * eng.patient.weight)

    for _ in range(1800):
        warm.state.temp_c = 37.0
        cold.state.temp_c = 33.0
        warm.step(2.0)
        cold.step(2.0)
    assert cold.state.propofol_cp > warm.state.propofol_cp * 1.15
    assert cold.state.tof < warm.state.tof - 15.0


def test_awake_sepsis_fever_plateaus_with_active_sweating(engine_factory, advance_time):
    """Active sweating and cutaneous vasodilation regulate awake sepsis fever near 38.5-39.8 C."""
    sepsis = engine_factory(
        config=SimulationConfig(mode="awake", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
    )
    sepsis.start_sepsis()
    advance_time(sepsis, 6.0 * 3600.0, dt=10.0)
    assert 38.0 < sepsis.state.temp_c < 39.8
