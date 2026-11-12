from itertools import pairwise

import pytest

from anasim.machine.ventilator import AnesthesiaVentilator
from anasim.physiology.fatigue import RespiratoryFatigue
from anasim.physiology.resp_mech import RespiratoryMechanics


def _endurance_minutes(
    p_fraction: float,
    duty: float,
    period_s: float = 5.0,
    dt: float = 0.1,
    max_minutes: float = 75.0,
    muscle: RespiratoryFatigue | None = None,
) -> float:
    """Minutes of square-wave breathing until three consecutive breaths miss the target pressure."""
    muscle = muscle or RespiratoryFatigue()
    target = p_fraction * muscle.MAX_PRESSURE
    failed_breaths = 0
    breath_failed = was_inspiring = False
    for i in range(round(max_minutes * 60.0 / dt)):
        t = i * dt
        inspiring = t % period_s < duty * period_s - 1e-12
        if was_inspiring and not inspiring:
            failed_breaths = failed_breaths + 1 if breath_failed else 0
            if failed_breaths == 3:
                return t / 60.0
            breath_failed = False
        if inspiring:
            pressure = min(target, muscle.available_pressure)
            breath_failed = breath_failed or pressure < target - 1e-6
        else:
            pressure = 0.0
        muscle.step(dt, pressure * dt)
        was_inspiring = inspiring
    return max_minutes


def _rest(muscle: RespiratoryFatigue, seconds: int) -> None:
    for _ in range(seconds):
        muscle.step(1.0, 0.0)


def test_subthreshold_load_and_paralysis_do_not_fatigue():
    # Bellemare and Grassino 1982: TTdi below about 0.15 is sustained for over 45 min.
    muscle = RespiratoryFatigue()
    assert _endurance_minutes(0.28, 0.50, max_minutes=60.0, muscle=muscle) == 60.0
    assert muscle.fatigue < 0.001

    paralyzed = RespiratoryFatigue()
    paralyzed.muscle_strength = 0.0
    _rest(paralyzed, 600)
    assert paralyzed.fatigue == 0.0 and paralyzed.available_pressure == 0.0


def test_endurance_time_tracks_bellemare_tension_time_scaling():
    """Bellemare and Grassino 1982: endurance above the threshold falls steeply
    with TTdi = (Pdi / Pdi,max) * (Ti / Ttot); their regression is
    Tlim = 0.1 * TTdi^-3.6 min.
    """
    # Pooled regression gives 32.8 min at TTdi 0.20, between the two duty cycles:
    # about 45 min at 0.4 Pmax, Ti/Ttot 0.5 (p. 1193; Fig. 6) and 13-14 min at
    # 0.8 Pmax, Ti/Ttot 0.25 (Fig. 3).
    assert 40.0 < _endurance_minutes(0.40, 0.50) < 60.0
    assert 9.5 < _endurance_minutes(0.80, 0.25) < 16.0
    assert 12.0 < _endurance_minutes(0.50, 0.50) < 20.0  # Regression 14.7 min
    assert 5.5 < _endurance_minutes(0.60, 0.50) < 9.5    # Regression 7.6 min
    assert 1.5 < _endurance_minutes(0.80, 0.50) < 3.2    # Fig. 6: 1-3 min (regression 2.7 min)
    assert 1.3 < _endurance_minutes(0.75, 0.60) < 2.5    # Regression 1.8 min (Fig. 2B trace 1.1 min)


@pytest.mark.parametrize("dt", [0.02, 0.1, 0.5])
def test_endurance_and_recovery_are_step_size_invariant(dt):
    assert _endurance_minutes(0.60, 0.50, dt=dt) == pytest.approx(6.96, abs=0.15)

    muscle = RespiratoryFatigue()
    for _ in range(round(300.0 / dt)):
        muscle.step(dt, 30.0 * dt)
    assert muscle.fatigue == pytest.approx(0.3114, abs=0.001)
    for _ in range(round(3600.0 / dt)):
        muscle.step(dt, 0.0)
    assert muscle.fatigue == pytest.approx(0.0826, abs=0.001)


def test_recovery_has_fast_and_persistent_phases():
    """Aubier 1981, Laghi 1995, and Travaline 1997: high-frequency fatigue resolves
    within 10-15 min, leaving about 20-25% Pdi,max loss at 30 min, 13-17% at 3 h,
    and a small low-frequency deficit at 24 h.
    """
    muscle = RespiratoryFatigue()
    # Laghi 1995: 30 min of resistive breathing at 60% Pmax, duty 0.5, 15 breaths/min.
    for i in range(18000):
        target = 60.0 if (i * 0.1) % 4.0 < 2.0 else 0.0
        muscle.step(0.1, min(target, muscle.available_pressure) * 0.1)
    assert muscle.available_pressure < 40.0

    _rest(muscle, 60)
    assert muscle.fatigue > 0.5

    _rest(muscle, 14 * 60)
    # Aubier 1981: 100-Hz capacity recovers within 10-15 min while low-frequency fatigue persists.
    assert muscle.fast_fatigue < 0.05 and muscle.slow_fatigue > 0.5

    _rest(muscle, 15 * 60)
    # Travaline 1997: twitch-occlusion Pdi,max is 75 +/- 6% of baseline at 30 min.
    assert 75.0 <= muscle.available_pressure <= 85.0
    at_30_min = muscle.fatigue

    _rest(muscle, 30 * 60)
    # Laghi 1995 and Travaline 1997: little recovery between 30 min and 1 h.
    assert muscle.fatigue > 0.9 * at_30_min

    _rest(muscle, 2 * 3600)
    # Travaline 1997: Pdi,max recovers to 87 +/- 8% by 2-3 h.
    assert 80.0 <= muscle.available_pressure <= 90.0

    _rest(muscle, 21 * 3600)
    # Laghi 1995 and Travaline 1997: small low-frequency deficit at 24 h.
    assert 0.03 < muscle.fatigue < 0.08


def test_fifteen_minute_rest_restores_repeat_endurance():
    """Rohrbach 2003: after a run to failure at about 67% MIP (duty 0.5, 18 breaths/min),
    15 min of rest leaves persistent twitch depression yet restores nearly full repeat
    endurance.
    """
    muscle = RespiratoryFatigue()
    first = _endurance_minutes(0.67, 0.50, period_s=60.0 / 18.0, dt=0.05, muscle=muscle)
    slow_after_first = muscle.slow_fatigue
    assert 3.0 < first < 9.0

    _rest(muscle, 900)
    assert muscle.fast_fatigue < 0.03
    assert 0.08 < muscle.fatigue < 0.15

    second = _endurance_minutes(0.67, 0.50, period_s=60.0 / 18.0, dt=0.05, muscle=muscle)
    assert 0.7 * first < second < first
    assert muscle.slow_fatigue > 1.4 * slow_after_first


def test_hypoxia_limits_endurance_and_prevents_recovery_without_supply():
    fatigues = []
    for oxygen in (1.0, 0.5):
        muscle = RespiratoryFatigue()
        muscle.oxygen_fraction = oxygen
        for _ in range(600):
            muscle.step(1.0, 15.0)
        fatigues.append(muscle.fatigue)
    assert fatigues[0] == 0.0 and fatigues[1] > 0.2
    muscle.oxygen_fraction = 0.0
    before = muscle.fatigue
    _rest(muscle, 600)
    assert muscle.fatigue >= before


def test_measured_muscle_pressure_drives_fatigue_and_assistance_unloads_it():
    fatigues = []
    for support in (False, True):
        lung = RespiratoryMechanics(compliance=0.01, resistance=30)
        lung.effort.set_drive(30, 1.0)
        vent = AnesthesiaVentilator()
        vent.update_settings(mode="PCV", rr=30, peep=5, p_insp=40, ie="1:1")
        for _ in range(600):
            vent.step(1, lung, "vent" if support else "spontaneous")
        assert 0 <= lung.effort.amplitude <= lung.effort.muscle.available_pressure + 0.2
        fatigues.append(lung.effort.muscle.fatigue)
    # Partial assistance reduces work, but does not fully rest these muscles.
    assert fatigues[0] > 0.2 and fatigues[1] < 0.85 * fatigues[0]


def test_ventilator_modes_unload_effort_in_leung_order(engine_factory):
    """Leung 1997: PSV progressively unloads PTP/min; SIMV unloads less than matched PSV
    unless combined with pressure support, and controlled ventilation unloads >75%.
    """
    # Listed from least to most support.
    modes = {
        "cpap": dict(mode="CPAP", rr=0, vt=0.0),
        "simv40": dict(mode="SIMV-VC", rr=6, vt=0.60, p_support=0.0),
        "ps40": dict(mode="PSV", rr=0, vt=0.0, p_support=6.8),
        "ps60": dict(mode="PSV", rr=0, vt=0.0, p_support=10.1),
        "simv40_ps10": dict(mode="SIMV-VC", rr=6, vt=0.60, p_support=10.0),
        "vcv": dict(mode="VCV", rr=17, vt=0.60),
    }
    ptp_per_min = {}
    for name, settings in modes.items():
        # Table 1: 8 of 11 patients had COPD; mean resistance was 18.0 cmH2O/(L/s).
        engine = engine_factory(start=True, age=68, weight=65, height=170, copd_severity=2 / 3)
        engine.set_airway_mode("ETT")
        engine.set_fgf(4.0, 6.0)
        engine.resp_mech.reference_compliance = 1.0 / 23.5
        engine.set_vent_settings(peep=0.0, ie="1:2", **settings)
        engine.set_vent_power(True)
        for _ in range(240):
            engine.step(1.0)
        ptp_per_min[name] = 0.0
        for _ in range(60):
            engine.step(1.0)
            ptp_per_min[name] += engine.vent.muscle_pressure_area

    assert all(less > more for less, more in pairwise(ptp_per_min.values()))
    assert 0.30 * ptp_per_min["cpap"] < ptp_per_min["ps60"] < 0.50 * ptp_per_min["cpap"]
    assert ptp_per_min["vcv"] < 0.25 * ptp_per_min["cpap"]


def test_obstructed_breathing_fatigues_and_pressure_support_restores_acute_capacity(awake_engine):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.set_fgf(10.0, 0.0)
    engine.set_vent_settings(rr=0.0, vt=0.0, peep=0.0, ie="1:2", mode="CPAP")
    engine.set_vent_power(True)
    for _ in range(120):
        engine.step(0.5)

    engine.set_airway_obstruction(0.62)
    for _ in range(3000):  # 25 min obstructed hypercapnic effort on O2
        engine.step(0.5)
    muscle = engine.resp_mech.effort.muscle
    tired = muscle.fatigue
    assert muscle.load > 0.15 and tired > 0.07

    engine.set_airway_obstruction(0.0)
    engine.set_vent_settings(rr=0.0, vt=0.0, peep=5.0, p_support=12.0, ie="1:2", mode="PSV")
    for _ in range(1800):  # 15 min PSV rescue
        engine.step(0.5)
    assert muscle.load < 0.05
    assert muscle.fast_fatigue < 0.01
    assert 0.01 < muscle.fatigue < 0.4 * tired


def test_fatigue_reduces_unconnected_breathing_without_changing_tof_or_central_drive(awake_engine):
    engine = awake_engine
    engine.step(1)
    baseline_vt = engine.resp.state.effort_vt
    baseline_drive = engine.resp.state.drive_central
    muscle = engine.resp_mech.effort.muscle
    muscle.fast_fatigue = muscle.slow_fatigue = 0.6
    engine.step(0.01)
    assert engine.resp.state.effort_vt < 0.5 * baseline_vt
    assert engine.resp.state.drive_central == pytest.approx(baseline_drive, rel=0.01)
    assert engine.state.tof > 99
