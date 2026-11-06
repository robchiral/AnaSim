import pytest

from anasim.core import runtime as runtime_core
from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.ecg import displayed_rhythm


def _hold(engine, seconds, hr=None, map_value=None, co=None):
    for _ in range(int(seconds / 0.1)):
        if hr is not None:
            engine.state.hr = hr
        if map_value is not None:
            engine.state.map = map_value
        if co is not None:
            engine.state.co = co
            engine.state.sbp = engine.state.dbp = 0.0 if co == 0 else 80.0
        runtime_core.check_cardiac_arrest(engine, 0.1)


@pytest.fixture
def arrest_engine(engine_factory):
    return engine_factory()


def test_arrest_needs_fifteen_seconds_without_pulses_and_pea_needs_negligible_flow(arrest_engine):
    _hold(arrest_engine, 30.0, map_value=15.0)
    assert not arrest_engine.state.cardiac_arrest
    _hold(arrest_engine, 10.0, map_value=5.0, co=0.0)
    _hold(arrest_engine, 1.0, map_value=70.0, co=5.0)
    _hold(arrest_engine, 10.0, map_value=5.0, co=0.0)
    assert not arrest_engine.state.cardiac_arrest
    _hold(arrest_engine, 6.0, map_value=5.0, co=0.0)
    assert arrest_engine.state.cardiac_arrest
    assert "Pulseless electrical activity" in arrest_engine.state.arrest_reason


def _ecg_range(engine):
    ecg = [sample.ecg_voltage for sample in engine.output_buffer]
    return max(ecg) - min(ecg)


def test_untreated_vf_becomes_fine_then_asystole_and_ends_the_session(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=7), start=True)
    engine.set_rhythm("VFIB")
    for _ in range(600):
        engine.step(0.1)
    assert engine.state.cardiac_arrest and engine.state.arrest_reason == RhythmType.VFIB.value
    # Without organized beats the monitor shows no pulse, arterial pressure, or pleth.
    state = engine.state
    assert state.display_hr == 0.0 and state.co == 0.0 and state.oxygen_delivery_ratio == 0.0
    assert state.art_sbp == state.art_dbp == state.art_map == 0.0
    assert state.pleth_voltage == pytest.approx(0.0) and not state.spo2_signal_valid
    coarse = _ecg_range(engine)
    while engine.hemo.coronary.energy_failure < 0.9:
        engine.step(0.1)
    assert engine.hemo.state.rhythm_type == RhythmType.VFIB
    assert _ecg_range(engine) < 0.4 * coarse
    while not engine.irreversible_arrest and engine.state.time < 900:
        engine.step(0.1)
    assert engine.state.arrest_reason == RhythmType.ASYSTOLE.value
    # About 11 minutes without circulation, beyond the metabolic phase of arrest.
    assert 600 < engine.state.time < 780


@pytest.mark.parametrize(("vf_seconds", "recovers"), [(240, True), (360, False)])
def test_restoring_rhythm_recovers_circulation_and_consciousness_only_early(engine_factory, vf_seconds, recovers):
    """Weisfeldt 2002: in the first ~4 minutes of VF, defibrillation alone usually restores circulation."""
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=7), start=True)
    engine.set_rhythm("VFIB")
    for _ in range(vf_seconds * 10):
        engine.step(0.1)
    # Without coronary flow the myocardium is ischemic despite oxygenated arterial blood.
    assert engine.state.myocardial_ischemia > 0.9 and engine.state.sao2 > 98
    engine.set_rhythm("SINUS")
    for _ in range(3000):
        engine.step(0.1)
    assert (engine.state.map > 60 and not engine.state.cardiac_arrest) == recovers
    assert (engine.state.myocardial_ischemia < 0.1) == recovers
    assert (engine.cerebral.unresponsiveness < 0.05) == recovers


def test_untreated_hemorrhage_progresses_through_pea_to_asystole(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=7), start=True)
    engine.start_hemorrhage(800)
    for _ in range(3000):
        engine.step(0.1)
        if engine.state.cardiac_arrest:
            break
    state = engine.state
    assert state.cardiac_arrest and not engine.irreversible_arrest
    assert "Pulseless electrical activity" in state.arrest_reason
    assert state.co < 0.3 and state.sbp - state.dbp < 5
    assert state.hr > 20 and engine.hemo.state.rhythm_type == RhythmType.SINUS
    assert not state.spo2_signal_valid
    ecg = [sample.ecg_voltage for sample in engine.output_buffer]
    assert max(ecg) - min(ecg) > 0.5
    pea_time = state.time
    stages = ["Sinus rhythm"]
    for _ in range(9000):
        engine.step(0.1)
        stage = displayed_rhythm(engine.hemo.state.rhythm_type, engine.hemo.coronary.energy_failure)
        if stage != stages[-1]:
            stages.append(stage)
        if engine.hemo.state.rhythm_type == RhythmType.ASYSTOLE:
            break
    assert stages == ["Sinus rhythm", "Junctional rhythm", "Idioventricular rhythm", "Agonal rhythm", "Asystole"]
    assert engine.state.arrest_reason == RhythmType.ASYSTOLE.value
    # The session ends about 11 minutes after circulation stops.
    assert engine.irreversible_arrest
    assert 600 < engine.state.time - pea_time + runtime_core.ARREST_CONFIRM_S < 780


def test_early_hemorrhage_control_and_volume_replacement_prevent_arrest(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=7), start=True)
    engine.start_hemorrhage(800)
    for _ in range(900):
        engine.step(0.1)
    engine.stop_hemorrhage()
    engine.give_fluid(1000)
    for _ in range(6000):
        engine.step(0.1)
        assert not engine.state.cardiac_arrest
    assert engine.state.map > 55 and engine.state.co > 3
    assert engine.state.myocardial_ischemia < 0.1
