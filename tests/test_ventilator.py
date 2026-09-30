import math

import pytest

from anasim.physiology.resp_mech import RespiratoryMechanics


def test_bag_mask_ventilates_paralysis_only_through_an_airway(awake_engine, advance_time):
    engine = awake_engine
    engine.give_drug_bolus("Rocuronium", 0.8 * engine.patient.weight)
    engine.set_bag_mask_ventilation(True, rr=12.0, vt=0.6)
    advance_time(engine, 60.0, dt=0.25)
    assert engine.state.mv < 1.0
    apneic_pao2 = engine.state.pao2

    engine.set_airway_mode("Mask")
    advance_time(engine, 60.0, dt=0.25)
    assert engine.state.mv > 5.0
    assert engine.state.vt > 400.0
    assert engine.state.pao2 > apneic_pao2 + 20.0
    # Machine monitors report the same bagged breaths that drive gas exchange.
    assert engine.vent.monitors.rr_total == pytest.approx(12.0)
    assert engine.vent.monitors.mv_exp == pytest.approx(engine.state.mv)


def test_pressure_support_augments_spontaneous_breaths(awake_engine, advance_time):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=0.0, vt=0.0, peep=8.0, ie="1:2", mode="CPAP", p_insp=0.0)
    engine.set_vent_power(True)
    advance_time(engine, 20.0, dt=0.1)
    vt_cpap = engine.state.vt
    assert vt_cpap > 200.0
    assert engine.resp_mech.state.paw_peak <= 10.0

    engine.set_vent_settings(rr=0.0, vt=0.0, peep=8.0, ie="1:2", mode="PSV", p_insp=10.0)
    advance_time(engine, 20.0, dt=0.1)
    assert engine.state.vt > vt_cpap + 50.0


@pytest.mark.parametrize("mode", ["VCV", "PCV"])
def test_breath_volumes_and_pressures_match_analytic_steady_state(mode):
    """Exact single-compartment solutions, including substantial trapped gas."""
    compliance, resistance = 0.05, 30.0
    rr, tidal_volume, peep, pressure = 20.0, 0.5, 5.0, 15.0
    period = 60.0 / rr
    ti, te = period / 3.0, period * 2.0 / 3.0
    tau = resistance * compliance
    exhalation = math.exp(-te / tau)
    if mode == "VCV":
        peak_volume = tidal_volume / (1.0 - exhalation)
        end_volume = peak_volume * exhalation
        expected_peak = peep + peak_volume / compliance + resistance * tidal_volume / ti
        expected_mean = peep + ti / period * (
            (end_volume + tidal_volume / 2.0) / compliance + resistance * tidal_volume / ti
        )
    else:
        peak_volume = compliance * pressure * (1.0 - math.exp(-ti / tau)) / (1.0 - math.exp(-period / tau))
        end_volume = peak_volume * exhalation
        expected_peak = peep + pressure
        expected_mean = peep + pressure * ti / period

    mech = RespiratoryMechanics(compliance=compliance, resistance=resistance)
    mech.set_settings(rr=rr, vt=tidal_volume, peep=peep, ie="1:2", mode=mode, p_insp=pressure)
    for _ in range(15):
        state = mech.step(7.0)  # Coarse steps span several breaths.
    assert state.delivered_vt == pytest.approx((peak_volume - end_volume) * 1000.0, abs=1e-7)
    assert state.auto_peep == pytest.approx(end_volume / compliance, abs=1e-8)
    assert state.paw_peak == pytest.approx(expected_peak, abs=1e-8)
    assert state.paw_plat == pytest.approx(peep + peak_volume / compliance, abs=1e-8)
    assert state.paw_mean == pytest.approx(expected_mean, abs=1e-8)


def test_passive_expiration_matches_time_constant_without_clipping():
    mech = RespiratoryMechanics(compliance=0.05, resistance=10.0)
    mech.set_rr = 0.0
    mech.state.volume = 0.6
    state = mech.step(1.0)  # Two RC time constants.
    expected_volume = 0.6 * math.exp(-2.0)
    assert state.volume == pytest.approx(expected_volume)
    assert state.flow == pytest.approx(-expected_volume / 0.5 * 60.0)
    assert state.auto_peep == pytest.approx(expected_volume / mech.compliance)


@pytest.mark.parametrize("mode", ["VCV", "PCV"])
def test_engine_uses_measured_tidal_volume_for_gas_exchange(awake_engine, mode):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12.0, vt=0.5, peep=5.0, ie="1:2", mode=mode, p_insp=15.0)
    engine.set_vent_power(True)
    engine.give_drug_bolus("Rocuronium", 0.8 * engine.patient.weight)
    engine.resp_mech.compliance = 0.025
    engine.step(0.1)
    assert engine.state.vt == engine.vent.monitors.tv_exp == 0.0
    for _ in range(120):
        engine.step(0.5)
    measured_vt_l = engine.resp_mech.state.delivered_vt / 1000.0
    assert engine.resp.state.apnea
    assert engine.state.vt == pytest.approx(measured_vt_l * 1000.0)
    assert engine.state.mv == pytest.approx(measured_vt_l * 12.0)
    assert engine.state.va == pytest.approx(max(0.0, measured_vt_l - engine.resp.vd_deadspace) * 12.0)
    assert engine.vent.monitors.tv_exp == pytest.approx(engine.state.vt)
