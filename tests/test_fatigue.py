import pytest

from anasim.machine.ventilator import AnesthesiaVentilator
from anasim.physiology.fatigue import RespiratoryFatigue
from anasim.physiology.resp_mech import RespiratoryMechanics


def test_quiet_breathing_and_paralysis_do_not_accumulate_fatigue():
    muscle = RespiratoryFatigue()
    for _ in range(10800):
        muscle.step(1, 5)
    assert muscle.fatigue == 0
    muscle.muscle_strength = 0
    for _ in range(600):
        muscle.step(1, 0)
    assert muscle.fatigue == 0 and muscle.available_pressure == 0


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_sustained_pressure_lowers_capacity_and_rest_recovers_it_gradually(dt):
    muscle = RespiratoryFatigue()
    for _ in range(round(300 / dt)):
        muscle.step(dt, 30 * dt)
    assert 0.5 < muscle.fatigue < 0.6
    tired = muscle.fatigue
    for _ in range(round(60 / dt)):
        muscle.step(dt, 0)
    assert muscle.fatigue > 0.9 * tired
    for _ in range(round(3600 / dt)):
        muscle.step(dt, 0)
    # Laghi 1995: twitch pressure remains depressed at 1 h and 24 h.
    assert 0.7 * tired < muscle.fatigue < 0.9 * tired


def test_severe_fatigue_can_persist_for_a_day_after_load_is_removed():
    muscle = RespiratoryFatigue()
    for _ in range(300):
        muscle.step(1, 30)
    tired = muscle.fatigue
    for _ in range(24 * 3600):
        muscle.step(1, 0)
    assert 0.35 * tired < muscle.fatigue < 0.5 * tired


def test_hypoxia_limits_endurance_and_prevents_recovery_without_supply():
    fatigues = []
    for oxygen in (1, 0.5):
        muscle = RespiratoryFatigue()
        muscle.oxygen_fraction = oxygen
        for _ in range(600):
            muscle.step(1, 15)
        fatigues.append(muscle.fatigue)
    assert fatigues[0] == 0 and fatigues[1] > 0.2
    muscle.oxygen_fraction = 0
    before = muscle.fatigue
    for _ in range(600):
        muscle.step(1, 0)
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
    assert fatigues[0] > 0.2 and fatigues[1] < 0.9 * fatigues[0]


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
