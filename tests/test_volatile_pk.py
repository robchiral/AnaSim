"""Sevoflurane uptake and washout (Carpenter 1986; Yasuda 1991)."""

import pytest

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
    engine.set_vaporizer("sevo", 6.0)
    for _ in range(600):
        engine.step(0.1)
    expected = engine.state.et_sevo / engine.pk_sevo.mac_age
    assert engine.state.et_mac == pytest.approx(expected, rel=1e-6)
    assert engine.state.et_mac > engine.state.mac
