import math

import pytest
from scipy.integrate import solve_ivp

import anasim.machine.ventilator as ventilator
from anasim.machine.ventilator import CIRCUIT_RESISTANCE, AnesthesiaVentilator
from anasim.physiology.resp_mech import RespiratoryMechanics


def breaths(engine, seconds, dt=0.01):
    """Step the engine and return (kind, Ppeak, VTe) for each breath that starts."""
    started, count = [], engine.vent.breath_count
    for _ in range(round(seconds / dt)):
        engine.step(dt)
        if engine.vent.breath_count != count:
            count = engine.vent.breath_count
            spirometry = engine.vent.monitors
            started.append((engine.vent._breath.kind, spirometry.paw_peak, spirometry.tv_exp))
    return started


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
    # Machine spirometry reports the same bagged breaths that drive gas exchange.
    assert engine.vent.monitors.rr_total == pytest.approx(12.0)
    assert engine.vent.monitors.mv_exp == pytest.approx(engine.state.mv)


@pytest.mark.parametrize("mode", ["VCV", "MANUAL", "PCV"])
def test_breath_volumes_and_pressures_match_analytic_steady_state(mode, monkeypatch):
    """Exact single-compartment solutions with substantial trapped gas.

    VCV holds the delivered volume for its inspiratory pause; a bag breath is a
    half-sine flow whose pressure peaks before flow stops. Exhaled gas crosses
    the expiratory limb, which slows expiration and holds the Y-piece above PEEP.
    """
    monkeypatch.setattr(ventilator, "RISE_TIME", 0.0)
    compliance, resistance = 0.05, 30.0
    rr, tidal_volume, pressure = 20.0, 0.5, 15.0
    peep = 0.0 if mode == "MANUAL" else 5.0
    period = 60.0 / rr
    ti, te = period / 3.0, period * 2.0 / 3.0
    exhalation = math.exp(-te / ((resistance + CIRCUIT_RESISTANCE) * compliance))
    lung = RespiratoryMechanics(compliance=compliance, resistance=resistance)
    lung.viscoelastic_ratio = 0.0
    vent = AnesthesiaVentilator()
    vent.update_settings(rr=rr, tv=tidal_volume * 1000.0, peep=peep, ie="1:2",
                         mode="PCV" if mode == "PCV" else "VCV", p_insp=pressure)
    if mode in ("VCV", "MANUAL"):
        peak_volume = tidal_volume / (1.0 - exhalation)
        end_volume = peak_volume * exhalation
        # Volume x time over inspiration; both flows put R x VT of resistive pressure-time in.
        volume_time = ti * (end_volume + tidal_volume / 2.0)
        if mode == "VCV":
            t_flow = ti * (1.0 - vent.settings.pause / 100.0)
            volume_time = t_flow * (end_volume + tidal_volume / 2.0) + (ti - t_flow) * peak_volume
            expected_peak = peep + peak_volume / compliance + resistance * tidal_volume / t_flow
        else:
            w = math.pi / ti
            theta = math.pi - math.atan(resistance * compliance * w)
            expected_peak = peep + (end_volume + tidal_volume / 2.0 * (1.0 - math.cos(theta))) / compliance + (
                resistance * tidal_volume / 2.0 * w * math.sin(theta)
            )
        inspiratory_area = volume_time / compliance + resistance * tidal_volume
        expected_plateau = math.nan if mode == "MANUAL" else peep + peak_volume / compliance
    else:
        filling = math.exp(-ti / (resistance * compliance))
        peak_volume = compliance * pressure * (1.0 - filling) / (1.0 - filling * exhalation)
        end_volume = peak_volume * exhalation
        expected_peak = expected_plateau = peep + pressure
        inspiratory_area = pressure * ti
    exhaled = peak_volume - end_volume
    expected_mean = peep + (inspiratory_area + CIRCUIT_RESISTANCE * exhaled) / period

    for _ in range(15):
        vent.step(7.0, lung, "bag" if mode == "MANUAL" else "vent", bag=(rr, tidal_volume))  # Steps span breaths.
    spirometry = vent.monitors
    assert spirometry.tv_exp == pytest.approx(exhaled * 1000.0, abs=1e-7)
    assert spirometry.auto_peep == pytest.approx(end_volume / compliance, abs=1e-8)
    assert spirometry.paw_peak == pytest.approx(expected_peak, abs=1e-8)
    assert spirometry.paw_plat == pytest.approx(expected_plateau, abs=1e-8, nan_ok=True)
    assert spirometry.paw_mean == pytest.approx(expected_mean, abs=1e-8)
    end_flow = end_volume / compliance / (resistance + CIRCUIT_RESISTANCE)
    assert spirometry.peep == pytest.approx(peep + CIRCUIT_RESISTANCE * end_flow, abs=1e-8)


@pytest.mark.parametrize("mode", ["VCV", "PCV"])
def test_viscoelastic_breaths_match_numerical_integration(mode):
    """Exact segments with tissue stress relaxation agree with a fine numerical solution."""
    lung = RespiratoryMechanics(compliance=0.045, resistance=12.0)
    vent = AnesthesiaVentilator()
    vent.update_settings(rr=15.0, tv=500.0, peep=5.0, ie="1:2", mode=mode, p_insp=14.0, pause=20.0)
    for _ in range(20):
        vent.step(5.0, lung, "vent")  # 25 breaths

    elastance, tau = 1.0 / lung.compliance, lung.viscoelastic_tau
    e2 = lung.viscoelastic_ratio * elastance
    period = 4.0
    ti = period / 3.0
    t_flow = ti * 0.8
    flow = 0.5 / t_flow

    def rhs(t, x):
        phase = t % period
        if phase >= ti:
            vdot = -(elastance * x[0] + x[1]) / (lung.resistance + CIRCUIT_RESISTANCE)
        elif mode == "VCV":
            vdot = flow if phase < t_flow else 0.0
        else:
            drive = 14.0 * min(1.0, phase / ventilator.RISE_TIME)
            vdot = (drive - elastance * x[0] - x[1]) / lung.resistance
        return [vdot, e2 * vdot - x[1] / tau]

    split = t_flow if mode == "VCV" else ventilator.RISE_TIME
    bounds = [b for k in range(25) for b in (k * period, k * period + split, k * period + ti)]
    x = [0.0, 0.0]
    for start, end in zip(bounds, bounds[1:] + [25 * period]):
        x = solve_ivp(rhs, (start, end), x, method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1]
    # Both end at the start of an inspiration.
    assert lung.volume == pytest.approx(x[0], abs=1e-7)
    assert lung.p2 == pytest.approx(x[1], abs=1e-7)


def test_short_inspiratory_pause_overestimates_static_recoil():
    """Tissue stress relaxes during an end-inspiratory hold, so a longer pause
    measures a lower plateau, approaching static recoil (Jonson 1993)."""
    plateaus = {}
    for pause in (10.0, 50.0):
        lung = RespiratoryMechanics(compliance=0.05, resistance=10.0)
        vent = AnesthesiaVentilator()
        vent.update_settings(rr=12.0, tv=500.0, peep=5.0, ie="1:2", mode="VCV", pause=pause)
        for _ in range(10):
            vent.step(6.0, lung, "vent")
        plateaus[pause] = vent.monitors.paw_plat - vent.monitors.peep
    static = 0.5 / 0.05
    assert plateaus[10.0] > plateaus[50.0] > static
    # Measured compliance VT/(Pplat - PEEP) after the usual 10% pause is 5-15% low.
    assert 0.85 < static / plateaus[10.0] < 0.95


def test_pressure_support_follows_patient_triggers(awake_engine, advance_time):
    """Each patient effort triggers a breath that ends when flow falls to 25% of its peak."""
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=0.0, vt=0.0, peep=8.0, ie="1:2", mode="CPAP")
    engine.set_vent_power(True)
    advance_time(engine, 30.0, dt=0.1)
    cpap = engine.get_latest_state()
    assert cpap.vt > 400.0 and cpap.paw_peak < 10.0

    engine.set_vent_settings(rr=0.0, vt=0.0, peep=8.0, ie="1:2", mode="PSV", p_support=10.0)
    advance_time(engine, 30.0, dt=0.1)
    assert engine.state.rr == pytest.approx(engine.resp.state.rr, abs=0.5)
    assert engine.state.vt > cpap.vt + 200.0
    assert engine.state.paw_peak == pytest.approx(18.0, abs=0.1)
    assert math.isnan(engine.state.paw_plat)

    peak = last = 0.0
    ratios = []
    for _ in range(1500):
        inspiring = engine.vent.inspiring
        engine.step(0.01)
        if engine.vent.inspiring:
            peak, last = max(peak, engine.vent.flow), engine.vent.flow
        elif inspiring and peak > 0.0:
            ratios.append(last / peak)
            peak = 0.0
    assert ratios and all(0.25 <= ratio < 0.3 for ratio in ratios)


def test_simv_synchronizes_mandatory_breaths_and_supports_the_rest(awake_engine, advance_time):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=6.0, vt=0.5, peep=5.0, ie="1:2", mode="SIMV-VC", p_support=5.0)
    engine.set_vent_power(True)
    advance_time(engine, 30.0, dt=0.1)

    started = breaths(engine, 60.0)
    kinds = [kind for kind, _, _ in started]
    # The patient breathes at 12/min; every other effort falls in a trigger window.
    assert len(started) == pytest.approx(engine.resp.state.rr, abs=1)
    assert kinds.count("VC") == pytest.approx(6, abs=1)
    assert set(kinds) == {"VC", "PS"}
    assert engine.state.rr == pytest.approx(engine.resp.state.rr, abs=0.5)


def test_volume_guarantee_restores_tidal_volume_three_cmh2o_per_breath(awake_engine, advance_time):
    """GE PCV-VG changes inspiratory pressure by at most 3 cmH2O per breath."""
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.give_drug_bolus("Rocuronium", 0.8 * engine.patient.weight)
    engine.set_vent_settings(rr=12.0, vt=0.45, peep=5.0, ie="1:2", mode="PCV-VG")
    engine.set_vent_power(True)
    advance_time(engine, 60.0, dt=0.1)
    assert engine.state.vt == pytest.approx(450.0, rel=0.02)

    engine.resp_mech.compliance /= 2.0
    started = breaths(engine, 60.0)
    peaks = [peak for _, peak, _ in started]
    assert started[1][2] < 300.0
    assert max(b - a for a, b in zip(peaks, peaks[1:])) <= 3.0 + 1e-9
    assert started[-1][2] == pytest.approx(450.0, rel=0.02)


def test_kinked_tube_reaches_pmax_and_alarms_low_volume(awake_engine, advance_time):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.give_drug_bolus("Rocuronium", 0.8 * engine.patient.weight)
    engine.set_vent_settings(rr=12.0, vt=0.5, peep=5.0, ie="1:2", mode="VCV", p_max=30.0)
    engine.set_vent_power(True)
    advance_time(engine, 60.0, dt=0.1)
    assert engine.state.vt == pytest.approx(500.0, rel=0.01)
    assert "VTe" not in engine.state.alarms

    engine.set_airway_obstruction(1.0)
    engine.set_bronchospasm(1.0)
    advance_time(engine, 40.0, dt=0.1)
    assert engine.state.paw_peak == pytest.approx(30.0, abs=0.01)
    assert engine.state.vt < 450.0
    assert engine.vent.pressure_limited
    assert engine.state.alarms["VTe"]["low"]


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
    # Waning efforts vary the first breaths until block is complete.
    for _ in range(360):
        engine.step(0.5)
    measured_vt_l = engine.vent.monitors.tv_exp / 1000.0
    assert engine.resp.state.apnea
    assert engine.state.vt == pytest.approx(measured_vt_l * 1000.0)
    assert engine.state.mv == pytest.approx(measured_vt_l * 12.0)
    assert engine.state.va == pytest.approx(max(0.0, measured_vt_l - engine.resp.vd_deadspace) * 12.0)


def test_plateau_compliance_scales_with_lung_size_and_bmi(engine_factory):
    """VT/(Pplat - PEEP) at PEEP 5 for a 170 cm, 70 kg man lies in the interquartile
    range of Dräger Primus recordings at PEEP 4-6 (VitalDB, 181 adults: 40.5-53.9
    mL/cmH2O). It rises with lung size and falls with BMI (Pelosi 1998)."""
    from anasim.core.state import SimulationConfig

    def plateau_compliance(**patient):
        engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True, **patient)
        for _ in range(300):
            engine.step(0.1)
        s = engine.state
        return s.vt / (s.paw_plat - s.peep)

    typical = plateau_compliance()
    assert 40.5 < typical < 53.9
    assert plateau_compliance(sex="female", height=160, weight=56) < typical
    assert plateau_compliance(height=170, weight=90) < typical < plateau_compliance(height=190, weight=80)
