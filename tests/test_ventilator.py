import math
from dataclasses import asdict

import pytest
from scipy.integrate import solve_ivp

import anasim.machine.ventilator as ventilator
from anasim.core.state import SimulationConfig
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


def test_rejected_ventilator_updates_preserve_settings_and_gas_flows(awake_engine):
    engine = awake_engine
    engine.set_fgf(1.0, 1.0, 0.0)
    settings = asdict(engine.vent.settings)
    gas_flows = (engine.circuit.fgf_o2, engine.circuit.fgf_air, engine.circuit.fgf_n2o)
    changes = dict(rr=18, vt=0.6, peep=8, ie="1:3", mode="PCV", fio2=0.5)
    invalid_changes = (
        {"ie": "invalid"},
        {"ie": "1:0"},
        {"vt": math.nan},
        {"fio2": math.inf},
        {"rr": -1},
        {"rr": None},
        {"mode": "SIMV"},
        {"mode": None},
        {"t_insp": 0},
        {"p_max": 8},
        {"unknown": 1},
    )
    for invalid in invalid_changes:
        with pytest.raises(ValueError):
            engine.set_vent_settings(**(changes | invalid))
        assert asdict(engine.vent.settings) == settings
        assert (engine.circuit.fgf_o2, engine.circuit.fgf_air, engine.circuit.fgf_n2o) == gas_flows

    engine.set_vent_settings(**changes)
    assert (engine.vent.settings.mode, engine.vent.settings.rr, engine.vent.settings.tv) == ("PCV", 18, 600)
    assert engine.circuit.fgf_o2 + engine.circuit.fgf_air == pytest.approx(2.0)
    assert (engine.circuit.fgf_o2 + 0.21 * engine.circuit.fgf_air) / 2.0 == pytest.approx(0.5)


def test_bag_mask_ventilates_paralysis_only_through_an_airway(awake_engine, advance_time):
    engine = awake_engine
    engine.give_drug_bolus("Rocuronium", 0.8 * engine.patient.weight)
    engine.set_bag_mask_ventilation(True, rr=12.0, vt=0.6)
    advance_time(engine, 60.0, dt=0.25)
    assert engine.state.mv < 1.0
    assert engine.resp.state.apnea
    assert engine.resp.state.drive_central > 0.95
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
    The nonlinear PEEP valve is tested separately.
    """
    monkeypatch.setattr(ventilator, "RISE_TIME", 0.0)
    monkeypatch.setattr(ventilator, "PEEP_VALVE", 0.0)
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
        expected_peak = peep + pressure
        expected_plateau = math.nan  # Short Ti and high resistance leave flow at end-inspiration.
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
def test_viscoelastic_breaths_match_numerical_integration(mode, monkeypatch):
    """Exact segments with tissue stress relaxation agree with a fine numerical solution."""
    monkeypatch.setattr(ventilator, "PEEP_VALVE", 0.0)
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


@pytest.mark.parametrize("compliance, resistance, rr", [(0.045, 12.0, 15.0), (0.02, 20.0, 20.0)])
def test_peep_valve_slows_early_exhalation_as_a_quadratic_drop(compliance, resistance, rr):
    """The linearized valve tracks a numerical solution with drop k Q^2 above PEEP.

    Early in exhalation the valve holds the airway well above PEEP, as Primus
    recordings show; with PEEP off it adds nothing.
    """
    peep, period = 5.0, 60.0 / rr
    ti = period / 3.0
    t_flow = ti * 0.8
    paws = {}
    for set_peep in (0.0, peep):
        lung = RespiratoryMechanics(compliance=compliance, resistance=resistance)
        vent = AnesthesiaVentilator()
        vent.update_settings(rr=rr, tv=500.0, peep=set_peep, ie="1:2", mode="VCV", pause=20.0)
        for _ in range(25):
            vent.step(period, lung, "vent")
        vent.step(ti + 0.1, lung, "vent")
        paws[set_peep] = vent.paw - set_peep

    elastance, tau = 1.0 / compliance, lung.viscoelastic_tau
    e2 = lung.viscoelastic_ratio * elastance
    r, k = resistance + CIRCUIT_RESISTANCE, ventilator.PEEP_VALVE

    def exhaled_flow(x):
        recoil = elastance * x[0] + x[1]
        return 2.0 * recoil / (r + math.sqrt(r * r + 4.0 * k * recoil))

    def rhs(t, x):
        phase = t % period
        vdot = -exhaled_flow(x) if phase >= ti else (0.5 / t_flow if phase < t_flow else 0.0)
        return [vdot, e2 * vdot - x[1] / tau]

    bounds = [b for n in range(25) for b in (n * period, n * period + t_flow, n * period + ti)]
    bounds += [25 * period, 25 * period + t_flow, 25 * period + ti, 25 * period + ti + 0.1]
    x = [0.0, 0.0]
    for start, end in zip(bounds, bounds[1:]):
        x = solve_ivp(rhs, (start, end), x, method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1]
    q = exhaled_flow(x)
    assert lung.volume == pytest.approx(x[0], abs=1e-3)
    assert paws[peep] == pytest.approx(CIRCUIT_RESISTANCE * q + k * q * q, abs=0.05)
    assert paws[peep] > paws[0.0] + 1.0


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
    engine.set_vent_settings(rr=0.0, vt=0.0, peep=8.0, ie="1:2", mode="CPAP", t_insp=0.2)
    engine.set_vent_power(True)
    advance_time(engine, 30.0, dt=0.1)
    cpap = engine.get_latest_state()
    unassisted_effort = engine.resp_mech.effort.amplitude
    unassisted_ti = engine.resp_mech.effort.ti
    # Unsupported: Ppeak comes from exhaling through the PEEP valve.
    assert cpap.vt > 400.0 and cpap.paw_peak < 12.0
    # Inspiration is the near-sinusoidal flow of quiet breathing, so it reverses
    # smoothly at the end of Ti instead of stopping abruptly. Gas left from the
    # last exhalation ends it slightly early through the inflation reflex.
    flows = []
    for _ in range(6000):
        engine.step(0.001)
        flows.append(engine.vent.flow)
    start = next(i for i in range(1, len(flows)) if flows[i - 1] <= 0.0 < flows[i])
    end = next(i for i in range(start, len(flows)) if flows[i] <= 0.0)
    ti, peak = (end - start) * 0.001, math.pi / 2.0 * sum(flows[start:end]) / (end - start)
    half_sine = [peak * math.sin(math.pi * (i - start) / (end - start)) for i in range(start, end)]
    effort = engine.resp_mech.effort
    assert ti == pytest.approx(min(effort.TI_FRACTION * effort.period, effort.TI_MAX), rel=0.05)
    assert max(abs(f - s) for f, s in zip(flows[start:end], half_sine)) < 0.1 * peak

    engine.set_vent_settings(rr=0.0, vt=0.0, peep=8.0, ie="1:2", mode="PSV", p_support=10.0, t_insp=0.2)
    advance_time(engine, 30.0, dt=0.1)
    assert engine.state.rr == pytest.approx(engine.resp.state.rr, abs=0.5)
    # Support unloads the muscles and raises VT. An awake patient's inflation
    # reflex must not cap the assisted breath at their unassisted VT.
    assert engine.state.vt > 1.1 * cpap.vt
    assert engine.resp_mech.effort.amplitude < unassisted_effort
    assert engine.resp_mech.effort.ti < unassisted_ti
    assert engine.state.paw_peak == pytest.approx(18.0, abs=0.1)
    assert math.isnan(engine.state.paw_plat)

    peak = last = 0.0
    complete = False
    ratios = []
    for _ in range(15000):
        inspiring = engine.vent.inspiring
        engine.step(0.001)
        if engine.vent.inspiring and not inspiring:
            complete = True
        if engine.vent.inspiring and complete:
            peak, last = max(peak, engine.vent.flow), engine.vent.flow
        elif inspiring and complete and peak > 0.0:
            ratios.append(last / peak)
            peak = 0.0
    assert ratios and all(0.25 <= ratio < 0.3 for ratio in ratios)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_cpap_and_untriggered_psv_preserve_spontaneous_circuit_mechanics(dt):
    traces = []
    for mode, trigger in (("CPAP", 0.2), ("CPAP", 20), ("PSV", 20)):
        lung = RespiratoryMechanics()
        lung.effort.set_drive(12, 0.25)
        vent = AnesthesiaVentilator()
        vent.update_settings(mode=mode, rr=0, peep=5, p_support=5, trigger=trigger)
        for _ in range(round(25 / dt)):
            vent.step(dt, lung, "vent")
        trace = []
        for _ in range(round(5 / dt)):
            vent.step(dt, lung, "vent")
            trace.append((vent.paw, vent.flow, vent.volume))
            assert vent._breath is None or vent._breath.kind == "SPONT"
        assert vent.monitors.rr_total == pytest.approx(12, abs=0.1)
        assert 230 < vent.monitors.tv_exp < 260
        # Exhaling through the circuit raises Paw above PEEP even when the
        # effort never reaches the pressure-support trigger threshold.
        assert max(paw for paw, _, _ in trace) > 5.4
        traces.append(trace)
    for trace in traces[1:]:
        for point, reference in zip(trace, traces[0]):
            assert point == pytest.approx(reference, abs=1e-9)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_spontaneous_inflow_does_not_retain_the_expiratory_valve_load(dt):
    lung = RespiratoryMechanics()
    lung.effort.set_drive(30, 0.5)
    vent = AnesthesiaVentilator()
    vent.update_settings(mode="CPAP", peep=5)
    for _ in range(round(12 / dt)):
        vent.step(dt, lung, "vent")
    checked = 0
    for _ in range(round(4 / dt)):
        vent.step(dt, lung, "vent", collect_samples=True)
        for _, _, paw, flow, _ in vent.samples:
            if flow > 0:
                # Before detection as well as afterward, inspired gas crosses
                # the inspiratory limb, not the loaded expiratory PEEP valve.
                assert paw == pytest.approx(5 - CIRCUIT_RESISTANCE * flow / 60, abs=0.05)
                checked += 1
    assert checked > 100


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_pressure_support_cycles_at_zero_flow_during_pressure_rise(dt):
    class ObservedVentilator(AnesthesiaVentilator):
        def _end_inspiration(self, segment, t):
            ends.append((self._breath.t, segment.flow(t)))
            super()._end_inspiration(segment, t)

    # Short, weak efforts reach an insensitive trigger late. If flow reverses
    # during pressurization, PSV must release pressure rather than keep rising.
    lung = RespiratoryMechanics()
    lung.effort.set_drive(60, 0.1)
    vent = ObservedVentilator()
    vent.update_settings(mode="PSV", rr=0, peep=5, p_support=1, trigger=20)
    ends = []
    for _ in range(round(5 / dt)):
        vent.step(dt, lung, "vent")
    assert len(ends) == 3
    for duration, flow in ends[1:]:
        assert 0.2 < duration < ventilator.RISE_TIME
        assert flow == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("mode", ["VCV", "PCV"])
@pytest.mark.parametrize("dt", [0.1, 1.0])
def test_mixed_breaths_use_all_expired_gas(awake_engine, mode, dt):
    engine = awake_engine
    # Fix neural frequency while testing gas accounting, independent of CO2 feedback.
    engine.resp.hcvr_slope_baseline = 0.0
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=6, vt=0.5, peep=5, ie="1:2", mode=mode, p_insp=12)
    engine.set_vent_power(True)
    for _ in range(round(40 / dt)):
        engine.step(dt)
    expired = 0.0
    mandatory_starts = 0
    previous = engine.vent._since_mandatory
    for _ in range(round(20 / dt)):
        engine.step(dt)
        expired += sum(max(0.0, -sample[1]) for sample in engine.vent.samples)
        mandatory_starts += engine.vent._since_mandatory < previous
        previous = engine.vent._since_mandatory
    assert mandatory_starts == 2  # Counting spontaneous breaths does not change mandatory timing.
    assert engine.state.rr == pytest.approx(12.0, abs=0.1)
    assert engine.state.mv == pytest.approx(expired * 3, rel=0.01)
    assert engine.state.va == pytest.approx(engine.state.mv - engine.state.rr * engine.resp.vd_deadspace, abs=0.01)

    # At 9/min some inflations start just as a spontaneous inspiration ends.
    # They join that breath, so no measured breath reports a near-zero VTe.
    engine.set_vent_settings(rr=9, vt=0.5, peep=5, ie="1:2", mode=mode, p_insp=12)
    started = breaths(engine, 60.0, dt)
    assert min(vte for _, _, vte in started) > 50.0


def test_inflation_ends_effort_at_tidal_volume_under_anesthesia_but_not_awake():
    durations = []
    for unconscious in (0.0, 1.0):
        lung = RespiratoryMechanics()
        lung.effort.unconscious = unconscious
        lung.effort.set_drive(12, 0.5)
        vent = AnesthesiaVentilator()
        vent.update_settings(mode="VCV", rr=12, tv=500, peep=5, ie="1:6", pause=10)
        for _ in range(100):
            vent.step(0.01, lung, "vent")
        durations.append(lung.effort.ti)
    assert durations[0] == pytest.approx(1.6)
    assert durations[1] < 0.7


def test_simv_synchronizes_mandatory_breaths_and_supports_the_rest(awake_engine, advance_time):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=6.0, vt=0.5, peep=5.0, ie="1:2", mode="SIMV-VC", p_support=5.0)
    engine.set_vent_power(True)
    advance_time(engine, 30.0, dt=0.1)

    started = breaths(engine, 60.0)
    kinds = [kind for kind, _, _ in started]
    # Synchronization preserves the set mandatory rate. Timed breaths outside
    # the trigger window can add to the patient's own breaths.
    assert kinds.count("VC") == pytest.approx(6, abs=1)
    assert set(kinds) == {"VC", "PS"}
    assert engine.state.rr == pytest.approx(engine.resp.state.rr, abs=0.5)


@pytest.mark.parametrize("mode", ["SIMV-VC", "SIMV-PC", "SIMV-VG"])
@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_simv_early_triggers_preserve_the_set_mandatory_rate(mode, dt):
    lung = RespiratoryMechanics()
    lung.effort.set_drive(16.0, 0.5)
    vent = AnesthesiaVentilator()
    vent.update_settings(mode=mode, rr=12, tv=500, peep=5, p_insp=10, p_support=5, t_insp=1)
    previous, mandatory, kinds = None, [], set()
    for i in range(round(150 / dt)):
        vent.step(dt, lung, "vent")
        if vent._breath is not previous:
            previous = vent._breath
            kinds.add(previous.kind)
            if previous.mandatory:
                mandatory.append((i + 1) * dt - previous.t)
    # A patient breathing faster than RR must not turn SIMV into assist-control.
    recent = mandatory[-10:]
    assert 60.0 * (len(recent) - 1) / (recent[-1] - recent[0]) == pytest.approx(12.0, abs=0.05)
    assert mandatory[1] < 5.0  # The first effort really did advance a mandatory breath.
    assert "PS" in kinds  # Compensation must leave room for supported spontaneous breaths.


def test_simv_effort_follows_delivered_pressure_and_recovers_when_assistance_stops(engine_factory):
    endpoints = []
    for dt in (0.01, 0.1, 1.0):
        for ps in (0, 20):
            engine = engine_factory(config=SimulationConfig(mode="awake", dt=dt, rng_seed=1), start=True)
            engine.resp.hcvr_slope_baseline = 0.0
            engine.set_airway_mode("ETT")
            engine.set_vent_settings(mode="SIMV-PC", rr=12, vt=0.5, ie="1:2", peep=5,
                                     p_insp=5, p_support=ps, t_insp=1.7)
            engine.set_vent_power(True)
            kinds = set()
            for _ in range(round(80 / dt)):
                engine.step(dt)
                kinds.add(engine.vent._breath.kind)
            assert kinds == {"PC"}  # PS is configured, but never delivered.
            effort = engine.resp_mech.effort
            assert 2.0 < effort.support_pressure < 5.0
            endpoints.append((engine.state.vt, effort.ti, effort.amplitude, effort.support_pressure))
            support, amplitude = effort.support_pressure, effort.amplitude
            engine.set_vent_settings(mode="SIMV-PC", rr=12, vt=0.5, ie="1:2", peep=5,
                                     p_insp=10, p_support=ps, t_insp=1.7)
            for _ in range(round(20 / dt)):
                engine.step(dt)
            assert effort.support_pressure > 1.5 * support
            assert effort.amplitude < amplitude
            engine.set_vent_power(False)
            for _ in range(round(20 / dt)):
                engine.step(dt)
            assert effort.support_pressure == pytest.approx(0.0, abs=1e-12)
            assert effort.ti == pytest.approx(1.6)
    for endpoint in endpoints[1:]:
        assert endpoint == pytest.approx(endpoints[0], abs=1e-7)


@pytest.mark.parametrize("mode", ["SIMV-VC", "SIMV-PC", "SIMV-VG"])
@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_awake_simv_synchronizes_existing_efforts_without_adding_a_second_tidal_volume(engine_factory, mode, dt):
    # Start ventilation partway through the awake patient's respiratory cycle.
    # A narrow trigger window used to split each later inspiration into PS
    # followed by a mandatory stroke, producing two pressure/flow peaks.
    engine = engine_factory(config=SimulationConfig(mode="awake", dt=dt, rng_seed=1), start=True)
    for _ in range(round(2 / dt)):
        engine.step(dt)
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(mode=mode, rr=12, vt=0.5, ie="1:2", peep=5,
                             p_insp=15, p_support=5)
    engine.set_vent_power(True)
    for _ in range(round(60 / dt)):
        engine.step(dt)
    expired = 0.0
    kinds = set()
    for _ in range(round(20 / dt)):
        engine.step(dt)
        expired += sum(max(0.0, -sample[1]) for sample in engine.vent.samples)
        b = engine.vent._breath
        kinds.add(b.kind)
        if b.inspiring and b.t > 0.5:
            assert engine.state.paw >= 4.5
            assert engine.state.flow >= -0.05
    assert kinds == {mode.removeprefix("SIMV-")}
    assert engine.state.rr == pytest.approx(12.0, abs=0.1)
    if mode != "SIMV-PC":
        assert engine.state.vt == pytest.approx(500.0, abs=30.0)
        assert engine.state.mv == pytest.approx(6.0, abs=0.3)
    assert engine.state.mv == pytest.approx(expired * 3.0, rel=0.02)
    assert 25.0 < engine.state.etco2 < 40.0


@pytest.mark.parametrize("mode", ["SIMV-VC", "SIMV-PC", "SIMV-VG"])
def test_simv_handover_preserves_an_ongoing_supported_inspiration(engine_factory, mode):
    engine = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=1), start=True, baseline_rr=18.75
    )
    # An 8.75 s mandatory period opens its 5 s trigger window at 3.75 s,
    # during the supported breath that began near 3.2 s.
    engine.resp.hcvr_slope_baseline = 0.0
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(mode=mode, rr=60 / 8.75, vt=0.5, peep=5, ie="1:2",
                             p_insp=12, p_support=5, t_insp=1.7)
    engine.set_vent_power(True)
    handover_pressure = []
    previous, handovers = None, 0
    for _ in range(700):
        engine.step(0.01)
        current = engine.vent._breath
        if current is not previous:
            if previous is not None and previous.kind == "PS" and current.mandatory:
                handovers += 1
            previous = current
        if 3.7 <= engine.state.time <= 3.9:
            handover_pressure.append(engine.state.paw)

    # The first complete synchronized breath includes gas from the patient's effort.
    assert handovers == 1
    assert engine.state.vt > 350.0
    if mode == "SIMV-VC":
        assert engine.state.vt <= 550.0  # Adding a second set VT overinflates this breath.
    else:
        assert engine.state.vt < 650.0
        assert min(handover_pressure) > 8.0  # Preserve pressure while changing controllers.


def test_volume_guarantee_restores_tidal_volume_three_cmh2o_per_breath(awake_engine, advance_time):
    """GE PCV-VG changes inspiratory pressure by at most 3 cmH2O per breath."""
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.give_drug_bolus("Rocuronium", 0.8 * engine.patient.weight)
    engine.set_vent_settings(rr=12.0, vt=0.45, peep=5.0, ie="1:2", mode="PCV-VG")
    engine.set_vent_power(True)
    advance_time(engine, 75.0, dt=0.1)
    assert engine.state.vt == pytest.approx(450.0, rel=0.02)

    engine.resp_mech.reference_compliance /= 2.0
    started = breaths(engine, 60.0)
    peaks = [peak for _, peak, _ in started]
    assert started[1][2] < 300.0
    assert max(b - a for a, b in zip(peaks, peaks[1:])) <= 3.0 + 1e-9
    assert started[-1][2] == pytest.approx(450.0, rel=0.02)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_vcv_pressure_limit_remains_active_during_the_pause(dt):
    lung = RespiratoryMechanics(compliance=0.03)
    lung.effort.set_drive(12, 0.5)
    vent = AnesthesiaVentilator()
    vent.update_settings(mode="VCV", rr=12, tv=500, peep=5, ie="1:1", pause=60, p_max=18)
    samples = []
    for _ in range(round(3 / dt)):
        vent.step(dt, lung, "vent", collect_samples=True)
        samples.extend(vent.samples)
    # Flow delivery ends at 1 s; as muscle pressure relaxes during the pause,
    # the pressure limit releases gas rather than holding a rising Paw.
    elapsed, pause = 0.0, []
    for sample in samples:
        elapsed += sample[0]
        if 1 < elapsed < 2.5:
            pause.append(sample)
    assert max(s[2] for s in samples) <= 18 + 1e-9
    assert any(s[2] == pytest.approx(18) and s[3] < -1 for s in pause)
    assert sum(s[1] for s in samples) == pytest.approx(lung.volume, abs=1e-10)


@pytest.mark.parametrize("edit", [{"p_max": 20}, {"peep": 20}])
@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_volume_guarantee_rechecks_pressure_headroom_before_the_next_breath(edit, dt):
    lung = RespiratoryMechanics(compliance=0.02)
    vent = AnesthesiaVentilator()
    vent.update_settings(mode="PCV-VG", rr=12, tv=500, peep=5, p_max=40)
    for _ in range(round(32 / dt)):
        vent.step(dt, lung, "vent")
    assert vent._vg_pressure > 25
    vent.update_settings(**edit)  # Edit during expiration, after pressure adaptation.
    for _ in range(round(4 / dt)):
        vent.step(dt, lung, "vent")
    assert vent._breath.kind == "VG" and vent.inspiring
    assert vent.paw == pytest.approx(vent.settings.p_max - 5, abs=1e-9)
    assert vent._breath.target == pytest.approx(vent.settings.p_max - 5 - vent.settings.peep)


def test_volume_guarantee_feedback_survives_a_mode_change_during_inspiration():
    lung = RespiratoryMechanics()
    vent = AnesthesiaVentilator()
    vent.update_settings(mode="PCV-VG")
    vent.step(5.5, lung, "vent")
    delivered_pressure = vent._breath.target
    vent.update_settings(mode="VCV")
    vent.step(0.1, lung, "vent")
    vent.update_settings(mode="PCV-VG")
    vent.step(1.1, lung, "vent")
    assert abs(vent._vg_pressure - delivered_pressure) <= 3
    vent.step(4, lung, "vent")
    assert vent._breath.kind == "VG"
    assert math.isfinite(vent.paw)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_zero_bag_rate_allows_spontaneous_breathing_without_a_manual_stroke(dt):
    lung = RespiratoryMechanics()
    vent = AnesthesiaVentilator()
    vent.step(5, lung, "bag", bag=(0, 0.5))
    assert vent._breath is None and vent.flow == 0
    lung.effort.set_drive(12, 0.5)
    for _ in range(round(20 / dt)):
        vent.step(dt, lung, "bag", bag=(0, 0.5))
    assert vent._breath is None
    assert vent.monitors.rr_total == pytest.approx(12, abs=0.1)
    assert 450 < vent.monitors.tv_exp < 550


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
    engine.resp_mech.reference_compliance = 0.025
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


@pytest.mark.parametrize("mode", ["VCV", "PCV", "PSV", "CPAP"])
def test_peep_changes_preserve_gas_volume_and_breath_timing_across_step_sizes(mode):
    endpoints = []
    for dt in (0.01, 0.1, 1.0):
        lung = RespiratoryMechanics()
        if mode in ("PSV", "CPAP"):
            lung.effort.set_drive(12, 0.5)
        vent = AnesthesiaVentilator()
        vent.update_settings(mode=mode, rr=12, tv=500, peep=5, p_insp=15, p_support=8)
        vent.step(0.5, lung, "vent", collect_samples=True)
        assert vent.inspiring
        pressure = vent.paw
        vent.update_settings(peep=15)
        vent.step(0.01, lung, "vent", collect_samples=True)
        # A setting edit must not translate the inspiratory pressure trace.
        assert abs(vent.paw - pressure) < 0.5
        assert lung.peep == 5
        for peep in (15, 0, 8):
            vent.update_settings(peep=peep)
            for _ in range(round(10 / dt)):
                before = vent.volume
                vent.step(dt, lung, "vent", collect_samples=True)
                assert vent.volume - before == pytest.approx(sum(row[1] for row in vent.samples), abs=1e-11)
                assert vent._peep() == lung.peep
            endpoints.append((dt, peep, vent.volume, lung.p2, vent.monitors.tv_exp, vent.breath_count))
    for i in range(3):
        reference = endpoints[i][2:]
        for j in (3, 6):
            assert endpoints[i + j][2:] == pytest.approx(reference, abs=1e-7)


def test_plateau_requires_settled_flow_and_relaxed_patient():
    plateaus = []
    for resistance in (5.0, 80.0):
        lung = RespiratoryMechanics(resistance=resistance)
        vent = AnesthesiaVentilator()
        vent.update_settings(mode="PCV", rr=6, peep=5, p_insp=15, ie="1:1")
        vent.step(30, lung, "vent")
        plateaus.append(vent.monitors.paw_plat)
        assert vent.monitors.paw_peak == pytest.approx(20)
    assert plateaus[0] == pytest.approx(20)
    assert math.isnan(plateaus[1])

    # A VCV pause stops flow, but ongoing inspiratory effort lowers Paw.
    lung = RespiratoryMechanics()
    lung.effort.set_drive(12, 0.5)
    vent = AnesthesiaVentilator()
    vent.update_settings(mode="VCV", rr=12, peep=5, tv=300, ie="1:4", pause=20)
    vent.step(5.01, lung, "vent")
    assert math.isnan(vent.monitors.paw_plat)


@pytest.mark.parametrize("mode", ["SIMV-VC", "SIMV-PC", "SIMV-VG", "PSV", "CPAP"])
def test_untriggered_breaths_are_measured_without_receiving_support(mode):
    endpoints = []
    for dt in (0.01, 0.1, 1.0):
        lung = RespiratoryMechanics()
        lung.effort.set_drive(18, 0.25)
        vent = AnesthesiaVentilator()
        vent.update_settings(mode=mode, rr=3 if mode.startswith("SIMV") else 0,
                             tv=500, p_insp=8, p_support=5, trigger=20, t_insp=0.5)
        for _ in range(round(58 / dt)):
            vent.step(dt, lung, "vent")
            assert vent._breath is None or vent._breath.kind != "PS"
        m = vent.monitors
        assert m.rr_total == pytest.approx(18, abs=0.1)
        assert 200 < m.tv_exp < 260
        assert vent.breath_count == 17
        endpoints.append((m.rr_total, m.tv_exp, m.mv_exp, lung.volume))
    for endpoint in endpoints[1:]:
        assert endpoint == pytest.approx(endpoints[0], abs=1e-7)


@pytest.mark.parametrize("mode", ["SIMV-VC", "SIMV-PC", "SIMV-VG"])
def test_simv_zero_rate_allows_only_patient_triggered_breaths(mode):
    lung = RespiratoryMechanics()
    vent = AnesthesiaVentilator()
    vent.update_settings(mode=mode, rr=0, tv=500, peep=5, p_support=8)
    vent.step(10, lung, "vent")
    assert vent._breath is None
    assert vent.flow == pytest.approx(0)
    assert vent.monitors.tv_exp == 0
    lung.effort.set_drive(18, 0.5)
    for _ in range(200):
        vent.step(0.1, lung, "vent")
        assert vent._breath is None or (vent._breath.kind == "PS" and not vent._breath.mandatory)
    assert vent.monitors.rr_total == pytest.approx(18, abs=0.1)
    assert vent.monitors.tv_exp > 400


@pytest.mark.parametrize("mode", ["SIMV-VC", "SIMV-PC", "SIMV-VG", "PSV"])
def test_sensitive_trigger_catches_first_effort_within_the_same_step(mode):
    endpoints = []
    for dt in (0.001, 0.01, 0.1, 1.0):
        lung = RespiratoryMechanics()
        lung.effort.set_drive(12, 0.5)
        vent = AnesthesiaVentilator()
        vent.update_settings(mode=mode, rr=0, peep=5, p_support=8, trigger=0.2)
        for _ in range(round(1 / dt)):
            vent.step(dt, lung, "vent")
        assert vent._breath is not None and vent._breath.kind == "PS"
        assert vent.paw == pytest.approx(13)
        endpoints.append((vent._breath.t, lung.volume, lung.p2, vent.flow))
    for endpoint in endpoints[1:]:
        assert endpoint == pytest.approx(endpoints[0], abs=1e-10)
