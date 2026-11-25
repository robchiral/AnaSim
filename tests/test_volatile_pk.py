"""Sevoflurane uptake and washout (Carpenter 1986; Yasuda 1991)."""

import copy

import pytest

from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.patient.volatile_pk import VolatilePK


def _sevo(patient_factory):
    return VolatilePK(patient_factory(), name="Sevoflurane", lambda_b_g=0.65, mac_40=2.0)


def _breathe(pk, seconds, fi, va=5.0, co=5.0):
    for _ in range(seconds):
        pk.step(1.0, fi, va, co)


def test_brain_equilibrates_in_minutes_and_fat_lags(patient_factory):
    pk = _sevo(patient_factory)
    _breathe(pk, 600, 0.04)
    assert pk.state.p_vrg > 0.4 * 0.04
    assert pk.state.p_fat < 0.3 * pk.state.p_vrg

    _breathe(pk, 1200, 0.04)
    assert 1.5 < pk.state.mac < 2.5


def test_washout_clears_brain_faster_than_fat(patient_factory):
    pk = _sevo(patient_factory)
    _breathe(pk, 3600, 0.02)
    vrg_start, fat_start = pk.state.p_vrg, pk.state.p_fat

    _breathe(pk, 600, 0.0)
    assert pk.state.p_vrg < 0.6 * vrg_start

    _breathe(pk, 1200, 0.0)
    assert pk.state.p_fat / fat_start > pk.state.p_vrg / vrg_start


@pytest.mark.parametrize("shunt", [0.0, 0.3])
def test_apnea_redistributes_volatile_without_losing_agent(patient_factory, shunt):
    pk = _sevo(patient_factory)
    pk.state.p_alv = pk.state.p_art = 0.02
    initial_amount = pk.frc * pk.state.p_alv
    for _ in range(10):
        pk.step(60.0, fi_agent=0.0, alveolar_vent_l=0.0, cardiac_output_l=5.0, shunt_fraction=shunt)
        state = pk.state
        pressures = (state.p_alv, state.p_vrg, state.p_mus, state.p_fat)
        assert all(0.0 <= value <= 0.02 for value in pressures)
        amount = pk.frc * state.p_alv + pk.lambda_b_g * (
            pk.v_vrg * pk.lambda_t_b_vrg * state.p_vrg
            + pk.v_mus * pk.lambda_t_b_mus * state.p_mus
            + pk.v_fat * pk.lambda_t_b_fat * state.p_fat
        )
        assert amount == pytest.approx(initial_amount, abs=1e-12)
    assert pk.state.p_vrg > 0.0


def test_gas_monitor_mac_follows_end_tidal_during_washin(engine_factory):
    engine = engine_factory(start=True)
    engine.set_airway_mode("Mask")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(6.0, 0.0)
    engine.set_vaporizer(6.0)
    for _ in range(60):
        engine.step(1.0)
    expected = engine.state.et_sevo / engine.pk_sevo.mac_age
    assert engine.state.et_mac == pytest.approx(expected, rel=1e-6)
    assert engine.state.et_mac > engine.state.mac


def test_high_ventilation_and_flow_substepping_preserves_volatile_uptake(patient):
    """VolatilePK substeps large dt under high VA and CO to match fine integration."""
    fine = VolatilePK(patient, name="N2O", lambda_b_g=0.47, mac_40=104.0)
    coarse = VolatilePK(patient, name="N2O", lambda_b_g=0.47, mac_40=104.0)
    for _ in range(1200):
        fine.step(0.05, 0.70, alveolar_vent_l=18.0, cardiac_output_l=12.0)
    for _ in range(60):
        coarse.step(1.0, 0.70, alveolar_vent_l=18.0, cardiac_output_l=12.0)
    assert coarse.state.p_alv == pytest.approx(fine.state.p_alv, rel=0.005)
    assert coarse.state.p_vrg == pytest.approx(fine.state.p_vrg, rel=0.005)


def test_volatile_pk_sex_ffm_scaling_and_zero_lung_volume_guard(patient_factory):
    """Male fat-free mass yields larger vessel-rich volume than a matched female,
    and zero lung_volume_l integrates without ZeroDivisionError.
    """
    male = VolatilePK(patient_factory(age=40, weight=70, height=170, sex="male"), name="Sevo", lambda_b_g=0.65, mac_40=2.0)
    female = VolatilePK(patient_factory(age=40, weight=70, height=170, sex="female"), name="Sevo", lambda_b_g=0.65, mac_40=2.0)
    assert male.v_vrg > female.v_vrg * 1.15
    assert female.v_fat > male.v_fat

    male.step(0.1, 0.02, alveolar_vent_l=4.0, cardiac_output_l=5.0, lung_volume_l=0.0)
    assert male.state.mac >= 0.0


def test_low_cardiac_output_accelerates_brain_volatile_equilibration(engine_factory, advance_time):
    """Low cardiac output / hemorrhagic shock accelerates alveolar wash-in and brain MAC equilibration."""
    p = Patient()
    pk_norm = VolatilePK(p, "Sevoflurane", 0.65, mac_40=2.0)
    pk_low = VolatilePK(p, "Sevoflurane", 0.65, mac_40=2.0)
    for _ in range(300):
        pk_norm.step(1.0, 0.02, 4.0, 5.0)
        pk_low.step(1.0, 0.02, 4.0, 2.0)
    assert pk_low.state.p_alv > pk_norm.state.p_alv
    assert pk_low.state.mac > pk_norm.state.mac + 0.05

    healthy = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    shocked = copy.deepcopy(healthy)
    shocked.hemo.add_volume(-1800.0)
    for eng in (healthy, shocked):
        eng.set_airway_mode("ETT")
        eng.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        eng.set_vent_power(True)
        eng.set_fgf(6.0, 0.0)
        eng.set_vaporizer(6.0)
    advance_time(healthy, 180.0, dt=1.0)
    advance_time(shocked, 180.0, dt=1.0)
    assert shocked.state.co < healthy.state.co * 0.65
    assert shocked.state.mac_sevo > healthy.state.mac_sevo


def test_cooling_conserves_dissolved_volatile_mass():
    """Cooling increases blood-gas solubility, lowering tissue partial pressure for conserved mass."""
    vpk = VolatilePK(Patient(), "Sevoflurane", lambda_b_g=0.65, mac_40=1.80)
    for _ in range(180):
        vpk.step(10.0, 0.02, 4.5, 5.0, temp_c=37.0)
    p_vrg_37 = vpk.state.p_vrg
    vpk.step(1.0, 0.0, 0.0, 5.0, temp_c=32.0)
    assert 0.70 * p_vrg_37 < vpk.state.p_vrg < p_vrg_37
