"""Tests for extracorporeal support: membrane oxygenator, VV ECMO, VA ECMO, and CPB."""

import math

import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.extracorporeal import (
    CannulationType,
    MembraneOxygenator,
    MembraneType,
    recirculation_fraction,
)
from anasim.patient.patient import Patient
from anasim.physiology.blood_chemistry import BloodChemistry
from anasim.physiology.blood_gas_math import oxygen_content


def _step_for(engine: SimulationEngine, seconds: float, dt: float = 0.25) -> None:
    for _ in range(round(seconds / dt)):
        engine.step(dt)


def _vent_engine(
    *,
    patient_kw: dict | None = None,
    mode: str = "steady_state",
    dt: float = 0.25,
    seed: int = 42,
    sevo: bool = False,
    arterial_line: bool = False,
    rr: float = 12.0,
    vt: float = 0.50,
    peep: float = 5.0,
    fio2: float = 0.50,
    ards_shunt: float = 0.0,
) -> SimulationEngine:
    patient = Patient(**(patient_kw or {"age": 50, "weight": 75, "height": 175, "sex": "male"}))
    config = SimulationConfig(
        mode=mode, dt=dt, rng_seed=seed,
        sevoflurane_enabled=sevo, arterial_line_enabled=arterial_line,
    )
    engine = SimulationEngine(patient, config)
    engine.start()
    engine.set_airway_mode("ETT")
    engine.set_vent_power(True)
    engine.set_vent_settings(rr=rr, vt=vt, peep=peep, ie="1:2", mode="VCV", fio2=fio2)
    if ards_shunt > 0.0:
        engine.set_ards_shunt(ards_shunt)
    return engine


def _ards_engine(*, dt: float = 0.25) -> SimulationEngine:
    return _vent_engine(
        patient_kw={"age": 45, "weight": 75, "height": 175, "sex": "male", "baseline_hb": 10.5},
        dt=dt, seed=42, rr=10.0, vt=0.28, peep=10.0, fio2=0.50, ards_shunt=0.65,
    )


def _cardiogenic_shock_engine(*, dt: float = 0.25) -> SimulationEngine:
    return _vent_engine(
        patient_kw={"age": 65, "weight": 75, "height": 175, "sex": "male", "heart_failure": "hfref", "baseline_hb": 11.0},
        dt=dt, seed=21, arterial_line=True, rr=12.0, vt=0.45, peep=5.0, fio2=0.50,
    )


def _cpb_engine(*, dt: float = 0.5, sevo: bool = False) -> SimulationEngine:
    return _vent_engine(
        patient_kw={"age": 60, "weight": 75, "height": 175, "sex": "male", "baseline_hb": 13.5},
        dt=dt, seed=51, sevo=sevo, arterial_line=True, rr=12.0, vt=0.50, peep=5.0, fio2=0.50,
    )


# ---------------------------------------------------------------------------
# Oxygenator and VV ECMO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["vv_ecmo", "va_ecmo"])
def test_ecmo_gas_conservation_and_vv_step_invariance(mode: str):
    """Exact O2, CO2, and buffer-base conservation during VV and VA ECMO, plus VV dt-invariance."""
    if mode == "vv_ecmo":
        eng = _ards_engine(dt=0.25)
        eng.start_vv_ecmo(blood_flow_l_min=4.5, sweep_gas_l_min=5.0, sweep_fio2=1.0, prime_volume_ml=0.0)
    else:
        eng = _vent_engine(
            patient_kw={"age": 55, "weight": 75, "height": 175, "sex": "male"},
            dt=0.25, seed=19, rr=10.0, vt=0.40, peep=5.0, fio2=0.50,
        )
        eng.start_va_ecmo(blood_flow_l_min=3.5, sweep_gas_l_min=4.0, sweep_fio2=1.0, prime_volume_ml=0.0)

    eng.set_continuous_fluid_rate(0.0)
    resp = eng.resp
    tr = resp.gas_transport

    def _totals() -> tuple[float, float, float]:
        cap = resp.gas_volume_l * resp._gas_capacity_per_l
        lung_co2 = cap * resp.state.p_alveolar_co2
        lung_o2 = cap * resp.state.p_alveolar_o2
        co2_tot = tr.co2_l + lung_co2 + tr.exhaled_l + tr.co2_removed_ecmo_l - tr.produced_l
        o2_tot = tr.oxygen_l + lung_o2 + tr.consumed_o2_l - tr.oxygen_from_air_l - tr.oxygen_from_ecmo_l
        return co2_tot, o2_tot, tr.base_meq

    _step_for(eng, 15.0, dt=0.25)
    co2_0, o2_0, base_0 = _totals()
    _step_for(eng, 120.0, dt=0.25)
    co2_1, o2_1, base_1 = _totals()
    assert tr.oxygen_from_ecmo_l > 0.15
    assert tr.co2_removed_ecmo_l > 0.15
    assert abs(co2_1 - co2_0) < 0.015
    assert abs(o2_1 - o2_0) < 0.005
    assert abs(base_1 - base_0) < 1.0

    if mode == "vv_ecmo":
        eng_a = _ards_engine(dt=0.05)
        eng_b = _ards_engine(dt=0.25)
        for e in (eng_a, eng_b):
            e.start_vv_ecmo(blood_flow_l_min=4.0, sweep_gas_l_min=4.5, sweep_fio2=1.0, prime_volume_ml=600.0)
        _step_for(eng_a, 60.0, dt=0.05)
        _step_for(eng_b, 60.0, dt=0.25)
        assert eng_a.state.pao2 == pytest.approx(eng_b.state.pao2, abs=1.5)
        assert eng_a.state.sao2 == pytest.approx(eng_b.state.sao2, abs=0.5)
        assert eng_a.state.pa_co2 == pytest.approx(eng_b.state.pa_co2, abs=0.8)
        assert eng_a.state.ecmo_co2_removal == pytest.approx(eng_b.state.ecmo_co2_removal, rel=0.04)


def test_schmidt_2013_vv_ecmo_determinants():
    """Schmidt 2013: Qb and FiO2_ecmo govern PaO2/SaO2; sweep governs PaCO2."""
    base = _ards_engine()
    _step_for(base, 180.0)
    assert base.state.sao2 < 88.0
    assert base.state.pao2 < 58.0
    assert base.state.pa_co2 > 48.0

    # Full VV ECMO rescues oxygenation and ventilation.
    base.start_vv_ecmo(blood_flow_l_min=4.5, sweep_gas_l_min=5.0, sweep_fio2=1.0, cannulation="fem_jug")
    _step_for(base, 240.0)
    s_high = base.get_latest_state()
    assert s_high.sao2 > 93.0
    assert s_high.pao2 > 68.0
    assert 30.0 <= s_high.pa_co2 <= 45.0
    assert s_high.ecmo_o2_transfer > 100.0
    assert s_high.ecmo_co2_removal > 120.0

    # Drop blood flow (4.5 -> 2.0 L/min): PaO2/SaO2 fall sharply, PaCO2 changes little.
    base.set_ecmo_flow(2.0)
    _step_for(base, 180.0)
    s_low_qb = base.get_latest_state()
    assert s_low_qb.sao2 < s_high.sao2 - 5.0
    assert s_low_qb.pao2 < s_high.pao2 - 15.0
    assert abs(s_low_qb.pa_co2 - s_high.pa_co2) < 8.0

    # Restore Qb, drop circuit FiO2 (1.0 -> 0.40): PaO2 drops, PaCO2 virtually unchanged.
    base.set_ecmo_flow(4.5)
    _step_for(base, 120.0)
    s_ref = base.get_latest_state()
    base.set_ecmo_fio2(0.40)
    _step_for(base, 180.0)
    s_low_fio2 = base.get_latest_state()
    assert s_low_fio2.pao2 < s_ref.pao2 - 10.0
    assert abs(s_low_fio2.pa_co2 - s_ref.pa_co2) < 3.0

    # Restore FiO2, drop sweep gas (5.0 -> 1.0 L/min): PaCO2 rises strongly, SaO2 barely changes.
    base.set_ecmo_fio2(1.0)
    _step_for(base, 120.0)
    s_ref2 = base.get_latest_state()
    base.set_ecmo_sweep(1.0)
    _step_for(base, 300.0)
    s_low_sweep = base.get_latest_state()
    assert s_low_sweep.pa_co2 > s_ref2.pa_co2 + 8.0
    assert abs(s_low_sweep.sao2 - s_ref2.sao2) < 5.0


def test_karagiannidis_2017_membrane_area_co2_ceilings():
    """Karagiannidis 2017 (porcine): 0.4 m^2 (~72 mL/min) vs 1.3 m^2 (~151 mL/min) at Qb=1.0, PvCO2=85, sweep=10."""
    hb = 10.0
    chem = BloodChemistry(hb / 1.61145, 0.30)
    base_bb = chem.evaluate(7.40, 40.0, 0.98).buffer_base - 8.0
    ven = chem.solve(85.0, 0.70, base_bb)
    o2_in = oxygen_content(40.0, hb, 27.0)[0]

    vco2_small = MembraneOxygenator(0.4).solve_pass(1.0, 10.0, 1.0, ven.total_co2, o2_in, base_bb, hb, 37.0, chem).co2_removal_l_min * 1000.0
    vco2_mid = MembraneOxygenator(1.3).solve_pass(1.0, 10.0, 1.0, ven.total_co2, o2_in, base_bb, hb, 37.0, chem).co2_removal_l_min * 1000.0
    vco2_large = MembraneOxygenator(1.8).solve_pass(1.0, 10.0, 1.0, ven.total_co2, o2_in, base_bb, hb, 37.0, chem).co2_removal_l_min * 1000.0
    assert 55.0 <= vco2_small <= 90.0
    assert 130.0 <= vco2_mid <= 175.0
    assert vco2_mid > vco2_small * 1.6
    assert vco2_large - vco2_mid < 25.0


def test_palmer_2016_recirculation_and_arrest_zero_delivery():
    """Palmér 2016: dual_lumen < fem_jug (~18-20%) < fem_fem; R -> 1 in arrest."""
    r_dl = recirculation_fraction(3.5, 5.0, CannulationType.DUAL_LUMEN)
    r_fj = recirculation_fraction(3.5, 5.0, CannulationType.FEM_JUG)
    r_ff = recirculation_fraction(3.5, 5.0, CannulationType.FEM_FEM)
    assert r_dl < r_fj < r_ff
    assert 0.12 <= r_fj <= 0.25
    assert recirculation_fraction(5.5, 5.0, CannulationType.FEM_JUG) > r_fj
    assert recirculation_fraction(4.0, 0.0, CannulationType.FEM_JUG) == pytest.approx(1.0)

    e_dl = _ards_engine()
    e_ff = _ards_engine()
    e_dl.start_vv_ecmo(blood_flow_l_min=4.0, sweep_gas_l_min=4.0, sweep_fio2=1.0, cannulation="dual_lumen")
    e_ff.start_vv_ecmo(blood_flow_l_min=4.0, sweep_gas_l_min=4.0, sweep_fio2=1.0, cannulation="fem_fem")
    _step_for(e_dl, 180.0)
    _step_for(e_ff, 180.0)
    assert e_ff.state.ecmo_recirculation > e_dl.state.ecmo_recirculation + 0.12
    assert e_dl.state.pao2 > e_ff.state.pao2 + 8.0

    # Cardiac arrest (CO=0): VV ECMO delivers zero net systemic O2.
    e_dl.set_rhythm("ASYSTOLE")
    _step_for(e_dl, 5.0)
    assert e_dl.state.ecmo_recirculation == pytest.approx(1.0)
    assert e_dl.state.ecmo_o2_transfer == pytest.approx(0.0, abs=1e-6)


def test_mauri_2016_respiratory_drive_unloading():
    """Mauri 2016: raising VV ECMO sweep lowers PaCO2 and reduces spontaneous RR/MV."""
    engine = _vent_engine(
        patient_kw={"age": 50, "weight": 70, "height": 175, "sex": "male"},
        mode="awake", dt=0.25, seed=7, ards_shunt=0.45,
    )
    engine.set_vent_settings(rr=6.0, vt=0.45, peep=8.0, ie="1:2", mode="PSV", p_support=8.0, fio2=0.60)
    engine.start_vv_ecmo(blood_flow_l_min=3.5, sweep_gas_l_min=0.5, sweep_fio2=1.0)
    _step_for(engine, 180.0)
    paco2_low = engine.state.pa_co2
    rr_low = engine.resp.state.neural_rr
    mv_low = engine.resp.state.rr * engine.resp.state.vt / 1000.0

    engine.set_ecmo_sweep(7.0)
    _step_for(engine, 240.0)
    mv_high = engine.resp.state.rr * engine.resp.state.vt / 1000.0
    assert engine.state.pa_co2 < paco2_low - 3.0
    assert mv_high < mv_low * 0.75
    assert engine.resp.state.neural_rr <= rr_low


def test_shekar_morales_drug_sequestration():
    """Shekar 2012 (ex vivo); Morales Castro 2025: lipophilic drugs sequester; morphine spared; passivates."""
    def _make(ecmo: bool, age_s: float = 0.0) -> SimulationEngine:
        eng = _vent_engine(
            patient_kw={"age": 40, "weight": 70, "height": 170, "sex": "male"},
            mode="awake", dt=0.5, seed=9,
        )
        if ecmo:
            eng.start_vv_ecmo(blood_flow_l_min=4.0, sweep_gas_l_min=4.0, prime_volume_ml=0.0)
            if age_s > 0.0:
                eng.extracorporeal.sequestration.circuit_age_s = age_s
        eng.give_drug_bolus("fentanyl", 200.0)
        eng.give_drug_bolus("propofol", 100.0)
        eng.give_drug_bolus("midazolam", 5.0)
        eng.give_drug_bolus("morphine", 10.0)
        _step_for(eng, 1200.0, dt=0.5)
        return eng

    ctrl = _make(ecmo=False)
    fresh = _make(ecmo=True, age_s=0.0)
    aged = _make(ecmo=True, age_s=8.0 * 3600.0)

    assert fresh.state.fentanyl_cp < ctrl.state.fentanyl_cp * 0.80
    assert fresh.state.propofol_cp < ctrl.state.propofol_cp * 0.85
    assert fresh.pk["midazolam"].state.c1 < ctrl.pk["midazolam"].state.c1 * 0.85
    assert fresh.state.midazolam_ce < ctrl.state.midazolam_ce * 0.90
    assert fresh.state.morphine_cp == pytest.approx(ctrl.state.morphine_cp, rel=0.03)
    assert aged.state.fentanyl_cp > fresh.state.fentanyl_cp
    assert aged.state.propofol_cp > fresh.state.propofol_cp


def test_prasser_2008_pmp_vs_microporous_sevoflurane_washout():
    """Prasser 2008: microporous PPL oxygenator clears sevoflurane far faster than plasma-tight PMP."""
    def _washout(membrane: MembraneType) -> tuple[float, float]:
        eng = _vent_engine(dt=0.5, seed=11, sevo=True)
        eng.set_fgf(2.0, 2.0)
        eng.set_vaporizer(2.5)
        _step_for(eng, 600.0, dt=0.5)
        eng.set_vaporizer(0.0)
        eng.set_vent_power(False)
        eng.start_vv_ecmo(
            blood_flow_l_min=4.5, sweep_gas_l_min=4.0, sweep_fio2=1.0,
            membrane_type=membrane, prime_volume_ml=0.0,
        )
        _step_for(eng, 30.0, dt=0.5)
        early_exhaust = eng.extracorporeal.state.exhaust_sevo_percent
        _step_for(eng, 1770.0, dt=0.5)
        return eng.pk_sevo.state.p_art * 100.0, early_exhaust

    art_pmp, exh_pmp = _washout(MembraneType.PMP)
    art_ppl, exh_ppl = _washout(MembraneType.MICROPOROUS)
    assert exh_ppl > exh_pmp * 3.0
    assert art_ppl < art_pmp * 0.65


def test_prime_hemodilution_without_preload_surge():
    """Priming an ECMO circuit dilutes Hb/Hct while keeping patient vascular volume steady."""
    eng = SimulationEngine(Patient(age=45, weight=70, height=170, sex="male"), SimulationConfig(mode="steady_state", dt=0.25, rng_seed=5))
    eng.start()
    _step_for(eng, 10.0)
    hb_before = eng.state.hb_g_dl
    pat_bv_before = eng.hemo.patient_blood_volume
    map_before = eng.state.map

    eng.start_vv_ecmo(blood_flow_l_min=4.0, sweep_gas_l_min=4.0, prime_volume_ml=600.0)
    assert eng.hemo.hb_conc < hb_before - 1.0
    assert eng.hemo.patient_blood_volume == pytest.approx(pat_bv_before, abs=1.0)
    _step_for(eng, 15.0)
    assert abs(eng.state.map - map_before) < 10.0

    pat_bv_on = eng.hemo.patient_blood_volume
    eng.stop_extracorporeal(reinfuse=False)
    assert eng.state.ecmo_mode == "none"
    assert eng.hemo.extracorporeal_volume_ml == 0.0
    assert eng.hemo.patient_blood_volume == pytest.approx(pat_bv_on, abs=1.0)


# ---------------------------------------------------------------------------
# VA ECMO hemodynamics, Harlequin syndrome, and nonpulsatile monitoring
# ---------------------------------------------------------------------------


def test_ostadal_donker_va_ecmo_flow_step_and_lv_unloading():
    """Ostadal 2015 (porcine); Donker 2019 (simulation); Aissaoui 2012:
    Raising VA ECMO flow increases MAP while reducing native LV SV/CO and raising LAP;
    LV venting, inotropes, and volume unloading reduce LAP and relieve LV distension.
    """
    eng = _cardiogenic_shock_engine()
    _step_for(eng, 30.0)
    base_map = eng.state.map
    base_sv = eng.state.sv

    eng.start_va_ecmo(blood_flow_l_min=1.5, sweep_gas_l_min=3.0, sweep_fio2=1.0, prime_volume_ml=0.0)
    _step_for(eng, 90.0)
    s_low = eng.get_latest_state()

    eng.set_ecmo_flow(4.5)
    _step_for(eng, 180.0)
    s_high = eng.get_latest_state()
    assert s_high.map > s_low.map + 12.0
    assert s_high.map > base_map + 10.0
    assert s_high.co == pytest.approx(s_high.intrinsic_co + s_high.pump_flow, abs=0.05)
    assert s_high.pulmonary_flow == pytest.approx(s_high.intrinsic_co, abs=0.05)
    assert s_high.sv < s_low.sv - 4.0
    assert s_high.sv < base_sv
    assert s_high.intrinsic_co < s_low.intrinsic_co
    assert s_high.lap > s_low.lap + 2.0

    # Weaning ramp down (Aissaoui 2012): stepping flow back to 1.5 L/min increases native SV.
    eng_wean = _cardiogenic_shock_engine()
    eng_wean.start_va_ecmo(blood_flow_l_min=4.5, sweep_gas_l_min=3.0, sweep_fio2=1.0, prime_volume_ml=0.0)
    _step_for(eng_wean, 150.0)
    sv_at_45 = eng_wean.state.sv
    eng_wean.set_ecmo_flow(1.5)
    _step_for(eng_wean, 120.0)
    assert eng_wean.state.sv > sv_at_45 + 4.0

    # LV venting (Donker 2019) directly unloads LAP at constant VA flow.
    lap_before_vent = eng.state.lap
    eng.set_ecmo_lv_vent(1.5)
    _step_for(eng, 120.0)
    assert eng.state.lap < lap_before_vent - 3.0

    # Inotrope unloading (dobutamine) improves native SV and lowers LAP at high VA flow.
    eng_ino = _cardiogenic_shock_engine()
    eng_ino.start_va_ecmo(blood_flow_l_min=4.5, sweep_gas_l_min=3.0, sweep_fio2=1.0, prime_volume_ml=0.0)
    _step_for(eng_ino, 180.0)
    lap_pre_ino, sv_pre_ino = eng_ino.state.lap, eng_ino.state.sv
    eng_ino.set_drug_rate("dobu", 10.0)
    _step_for(eng_ino, 240.0)
    assert eng_ino.state.sv > sv_pre_ino + 3.0
    assert eng_ino.state.lap < lap_pre_ino - 1.5

    # Volume unloading (-700 mL) lowers LAP (Donker 2019).
    eng_vol = _cardiogenic_shock_engine()
    eng_vol.start_va_ecmo(blood_flow_l_min=4.5, sweep_gas_l_min=3.0, sweep_fio2=1.0, prime_volume_ml=0.0)
    _step_for(eng_vol, 180.0)
    lap_pre_vol = eng_vol.state.lap
    eng_vol.hemo.add_volume(-700.0)
    _step_for(eng_vol, 120.0)
    assert eng_vol.state.lap < lap_pre_vol - 2.0


def test_stevens_2017_harlequin_differential_oxygenation():
    """Stevens 2017 (CFD simulation): peripheral femoral VA ECMO with recovering LV and hypoxic lungs
    causes right-radial, cerebral, and coronary desaturation while femoral blood stays hyperoxic;
    central VA ECMO eliminates the upper-lower gradient.
    """
    def _run(cannulation: str) -> SimulationEngine:
        eng = _vent_engine(
            patient_kw={"age": 50, "weight": 75, "height": 175, "sex": "male", "baseline_hb": 11.0},
            dt=0.25, seed=33, arterial_line=True, rr=10.0, vt=0.35, peep=5.0, fio2=0.25, ards_shunt=0.75,
        )
        eng.start_va_ecmo(
            blood_flow_l_min=3.2, sweep_gas_l_min=4.0, sweep_fio2=1.0,
            cannulation=cannulation, prime_volume_ml=0.0,
        )
        _step_for(eng, 180.0)
        return eng

    periph = _run("peripheral_fem")
    central = _run("central")

    assert periph.state.lower_sao2 > 98.0
    assert periph.state.upper_sao2 < periph.state.lower_sao2 - 12.0
    assert periph.extracorporeal.state.coronary_sao2 < periph.state.lower_sao2 - 15.0

    abg_radial = periph.draw_blood_gas("right_radial")
    abg_femoral = periph.draw_blood_gas("femoral")
    abg_post = periph.draw_blood_gas("post_oxy")
    assert abg_radial.site == "right_radial"
    assert abg_femoral.site == "femoral"
    assert abg_femoral.pao2 > abg_radial.pao2 + 80.0
    assert abg_femoral.sao2 > abg_radial.sao2 + 12.0
    assert abg_post.pao2 > 250.0
    assert abs(central.state.upper_sao2 - central.state.lower_sao2) < 1e-6


def test_va_ecmo_nonpulsatile_vfib_perfusion_coronary_and_monitors():
    """VA ECMO maintains systemic and coronary perfusion during VF without false cardiac arrest,
    renders a flat arterial line at pump MAP, fails NIBP, drops SpO2 pleth, and lowers EtCO2.
    """
    eng = _vent_engine(
        patient_kw={"age": 55, "weight": 75, "height": 175, "sex": "male"},
        dt=0.25, seed=17, arterial_line=True,
    )
    _step_for(eng, 15.0)
    etco2_baseline = eng.state.etco2

    eng.start_va_ecmo(blood_flow_l_min=4.5, sweep_gas_l_min=4.0, sweep_fio2=1.0, prime_volume_ml=0.0)
    eng.set_rhythm(RhythmType.VFIB)
    _step_for(eng, 45.0)

    s = eng.get_latest_state()
    assert not s.cardiac_arrest
    assert s.pump_flow == pytest.approx(4.5, abs=0.05)
    assert s.intrinsic_co == pytest.approx(0.0, abs=1e-6)
    assert s.co == pytest.approx(4.5, abs=0.05)
    assert s.map > 50.0
    assert s.art_map == pytest.approx(s.map, abs=4.0)
    assert math.isnan(s.art_sbp) and math.isnan(s.art_dbp)
    assert not s.spo2_signal_valid
    assert eng.hemo.coronary.energy_reserve > 0.85
    assert eng.cerebral.cortical_function > 0.90
    assert s.etco2 < etco2_baseline * 0.45

    eng.measure_nibp()
    _step_for(eng, 35.0)
    assert eng.state.nibp_measurement_failed

    eng.charge_defibrillator()
    _step_for(eng, 6.0)
    eng.defibrillate(synchronized=False)
    _step_for(eng, 15.0)
    assert eng.state.rhythm_type == RhythmType.SINUS
    assert eng.state.intrinsic_co > 0.5
    assert eng.state.etco2 > s.etco2 + 3.0


# ---------------------------------------------------------------------------
# Cardiopulmonary bypass
# ---------------------------------------------------------------------------


def test_magder_1998_cpb_venous_drainage_cross_clamp_alarms_and_weaning():
    """Magder and De Varennes 1998 passive stressed-volume drainage, cardioplegia cross-clamp,
    bypass alarm suppression, low-reservoir safety, and weaning off CPB.
    """
    eng = _cpb_engine(dt=0.25)
    eng.set_continuous_fluid_rate(0.0)
    _step_for(eng, 10.0, dt=0.25)
    pat_bv_pre = eng.hemo.patient_blood_volume
    rbc_pre = eng.hemo.red_cell_volume_ml

    eng.start_cpb(
        blood_flow_l_min=4.5, sweep_gas_l_min=3.0, sweep_fio2=0.80,
        prime_volume_ml=1400.0, initial_reservoir_ml=800.0,
        venous_clamp_open=1.0, bypass_alarm_mode=True,
    )
    eng.set_vent_power(False)
    _step_for(eng, 90.0, dt=0.25)

    drained_ml = pat_bv_pre - eng.hemo.patient_blood_volume
    assert 800.0 <= drained_ml <= 2000.0
    assert eng.state.cpb_reservoir_ml > 1500.0
    assert eng.hemo.red_cell_volume_ml == pytest.approx(rbc_pre, rel=1e-4)
    assert eng.hemo.patient_blood_volume + eng.extracorporeal.total_circuit_mixing_volume_ml == pytest.approx(
        eng.hemo.blood_volume, abs=1e-6,
    )

    eng.set_cpb_cardioplegia(True)
    eng.set_ecmo_heater_cooler(True, 28.0)
    _step_for(eng, 60.0, dt=0.25)
    s_clamp = eng.get_latest_state()
    assert s_clamp.cpb_cross_clamp
    assert s_clamp.hr == 0.0 and s_clamp.sv == 0.0 and s_clamp.intrinsic_co == 0.0
    assert s_clamp.pump_flow == pytest.approx(4.5, abs=0.1)
    assert s_clamp.map > 45.0
    assert math.isnan(s_clamp.art_sbp) and math.isnan(s_clamp.art_dbp)
    assert not s_clamp.cardiac_arrest
    for silenced in ("Asystole", "EtCO2", "SpO2", "HR"):
        assert silenced not in s_clamp.alarms
    assert eng.hemo.coronary.energy_reserve > 0.85

    # Weaning: release cross-clamp, restart ventilation, clamp venous line to refill patient.
    eng.set_cpb_cross_clamp(False)
    eng.set_rhythm(RhythmType.SINUS)
    eng.set_ecmo_heater_cooler(True, 37.0)
    eng.set_vent_power(True)
    eng.set_cpb_venous_clamp(0.0)
    eng.set_ecmo_flow(2.0)
    _step_for(eng, 45.0, dt=0.25)
    assert eng.hemo.patient_blood_volume > pat_bv_pre - 500.0
    assert eng.state.cpb_reservoir_ml < 1100.0
    assert eng.state.intrinsic_co > 1.5

    # Low-reservoir safety alarm and reservoir fluid replenishment.
    _step_for(eng, 45.0, dt=0.25)
    assert eng.extracorporeal.state.reservoir_low_alarm
    assert "ReservoirLow" in eng.state.alarms
    eng.add_cpb_reservoir_fluid(600.0, kind="crystalloid")
    _step_for(eng, 2.0, dt=0.25)
    assert eng.state.cpb_reservoir_ml > 450.0
    assert not eng.extracorporeal.state.reservoir_low_alarm

    eng.set_bypass_alarm_mode(False)
    eng.stop_extracorporeal(reinfuse=False)
    _step_for(eng, 20.0, dt=0.25)
    assert eng.state.ecmo_mode == "none"
    assert eng.state.pump_flow == 0.0
    assert eng.state.co > 2.5 and eng.state.map > 50.0


def test_gordon_1975_and_eckmann_2000_hemodilution_and_hypothermic_viscosity():
    """Gordon 1975: crystalloid CPB prime drops Hct, blood viscosity, and MAP vs blood prime;
    Eckmann 2000 (in vitro): cooling at fixed Hct increases blood viscosity and SVR.
    """
    cryst = _cpb_engine(dt=0.5)
    blood = _cpb_engine(dt=0.5)
    cryst.start_cpb(blood_flow_l_min=4.5, prime_volume_ml=1600.0, prime_hematocrit=0.0, water_target_c=37.0)
    blood.start_cpb(blood_flow_l_min=4.5, prime_volume_ml=1600.0, prime_hematocrit=0.42, prime_label="blood", water_target_c=37.0)
    cryst.set_cpb_cardioplegia(True)
    blood.set_cpb_cardioplegia(True)
    _step_for(cryst, 30.0, dt=0.5)
    _step_for(blood, 30.0, dt=0.5)

    assert cryst.state.hct < blood.state.hct - 0.07
    assert cryst.hemo._hct_viscosity_tpr < blood.hemo._hct_viscosity_tpr - 0.12
    assert cryst.state.map < blood.state.map - 6.0

    visc_warm = cryst.hemo._hct_viscosity_tpr
    cryst.set_ecmo_heater_cooler(True, 18.0)
    _step_for(cryst, 1200.0, dt=1.0)
    assert cryst.state.temp_c < 26.0
    assert cryst.hemo._hct_viscosity_tpr > visc_warm * 1.25


def test_rajek_2000_cpb_cooling_rewarming_and_postbypass_afterdrop():
    """Rajek 2000: cooling and rewarming on CPB leaves a core-to-periphery gradient
    causing post-bypass core afterdrop (~1.0-1.5 °C passive vs much smaller with Bair Hugger).
    """
    def _run_cpb_thermal(use_bair_hugger: bool) -> tuple[float, float, float]:
        eng = _cpb_engine(dt=1.0)
        eng.set_continuous_fluid_rate(0.0)
        eng.start_cpb(
            blood_flow_l_min=4.5, sweep_gas_l_min=3.0, prime_volume_ml=1400.0,
            heater_cooler_on=True, water_target_c=28.0,
        )
        eng.set_cpb_cardioplegia(True)
        _step_for(eng, 1200.0, dt=1.0)
        t_cooled = eng.state.temp_c
        if use_bair_hugger:
            eng.set_bair_hugger(43.0)
        eng.set_ecmo_heater_cooler(True, 38.0)
        for _ in range(2100):
            eng.step(1.0)
            if eng.state.temp_c >= 37.0:
                break
        t_wean = eng.state.temp_c
        grad_wean = t_wean - eng.thermal.peripheral_temp_c
        eng.set_cpb_cross_clamp(False)
        eng.set_rhythm(RhythmType.SINUS)
        eng.set_vent_power(True)
        eng.stop_extracorporeal(reinfuse=True)
        t_nadir = t_wean
        for _ in range(3000):
            eng.step(1.0)
            t_nadir = min(t_nadir, eng.state.temp_c)
        return t_cooled, grad_wean, t_wean - t_nadir

    cooled_p, grad_p, drop_passive = _run_cpb_thermal(use_bair_hugger=False)
    _, grad_w, drop_warmed = _run_cpb_thermal(use_bair_hugger=True)
    assert cooled_p < 32.5
    assert grad_p > 1.0
    assert 0.6 <= drop_passive <= 2.2
    assert grad_w < grad_p
    assert drop_warmed < drop_passive * 0.70


def test_murkin_1987_alpha_stat_vs_ph_stat_cerebral_blood_flow():
    """Murkin 1987: at ~26-27 °C hypothermic CPB, alpha-stat preserves low CMRO2-coupled,
    pressure-autoregulated CBF, whereas pH-stat raises CBF (~1.5-2.2x) and makes CBF pressure-passive.
    """
    def _run_strategy(strategy: str, sweep_fco2: float) -> tuple[float, float]:
        eng = _cpb_engine(dt=0.5)
        eng.start_cpb(
            blood_flow_l_min=4.2, sweep_gas_l_min=2.5, sweep_fio2=0.80,
            sweep_fco2=sweep_fco2, blood_gas_strategy=strategy,
            heater_cooler_on=True, water_target_c=27.0,
        )
        eng.set_cpb_cardioplegia(True)
        eng.set_vent_power(False)
        eng.state.temp_c = 27.0
        eng.thermal.peripheral_temp_c = 27.0
        _step_for(eng, 180.0, dt=0.5)
        eng.set_ecmo_flow(3.5)
        _step_for(eng, 30.0, dt=0.5)
        cbf_low_map = eng.cerebral.last_cbf_ratio
        eng.set_ecmo_flow(5.5)
        _step_for(eng, 30.0, dt=0.5)
        return cbf_low_map, eng.cerebral.last_cbf_ratio

    cbf_alpha_low, cbf_alpha_high = _run_strategy("alpha_stat", sweep_fco2=0.0)
    cbf_ph_low, cbf_ph_high = _run_strategy("ph_stat", sweep_fco2=0.055)
    assert abs(cbf_alpha_high - cbf_alpha_low) < 0.05
    assert cbf_ph_high > cbf_alpha_high * 1.55
    assert cbf_ph_high > cbf_ph_low + 0.12


def test_cpb_pk_pd_and_sweep_sevoflurane():
    """Hiraoka 2004 (propofol unbound fraction), Michelsen 2001 (remifentanil hypothermia),
    Mathew 2001 (hypothermic BIS drop), and microporous CPB sweep-gas sevoflurane delivery.
    """
    warm = _cpb_engine(dt=0.5, sevo=False)
    cold = _cpb_engine(dt=0.5, sevo=False)
    for eng in (warm, cold):
        eng.set_drug_rate("propofol", 100.0)
        eng.set_drug_rate("remi", 0.15)
        eng.start_cpb(blood_flow_l_min=4.5, sweep_gas_l_min=3.0, prime_volume_ml=1600.0, water_target_c=37.0)
        eng.set_cpb_cardioplegia(True)
    warm.state.temp_c = 37.0
    cold.state.temp_c = 28.0
    cold.set_ecmo_heater_cooler(True, 28.0)
    _step_for(warm, 600.0, dt=0.5)
    _step_for(cold, 600.0, dt=0.5)

    assert warm.state.albumin_g_l < 36.0
    assert warm.pk["propofol"].effect_input_fraction > 1.15
    assert cold.state.remi_cp > warm.state.remi_cp * 1.35
    assert cold.state.bis < warm.state.bis - 6.0

    sevo_eng = SimulationEngine(
        Patient(age=60, weight=75, height=175, sex="male"),
        SimulationConfig(mode="awake", dt=0.5, rng_seed=61, sevoflurane_enabled=True, arterial_line_enabled=True),
    )
    sevo_eng.start()
    sevo_eng.set_airway_mode("ETT")
    sevo_eng.set_vent_power(False)
    sevo_eng.set_vaporizer(0.0)
    sevo_eng.start_cpb(
        blood_flow_l_min=4.5, sweep_gas_l_min=4.0, sweep_fio2=0.80,
        sweep_sevo_percent=2.2, membrane_type="microporous",
    )
    sevo_eng.set_cpb_cardioplegia(True)
    _step_for(sevo_eng, 600.0, dt=0.5)
    assert sevo_eng.state.mac_sevo > 0.65
    assert sevo_eng.state.bis < 60.0
    assert sevo_eng.extracorporeal.state.exhaust_sevo_percent > 0.5
