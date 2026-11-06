import pytest

from anasim.core import runtime as runtime_core
from anasim.core.action_log import ACTION_SHOCK
from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.ecg import displayed_rhythm
from anasim.physiology.resuscitation import CHARGE_TIME_S, ShockResult


def _hold(engine, seconds, hr=None, map_value=None, co=None):
    for _ in range(int(seconds / 0.1)):
        if hr is not None:
            engine.state.hr = hr
        if map_value is not None:
            engine.state.map = map_value
        if co is not None:
            engine.state.co = engine.state.intrinsic_co = co
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
    # Without organized beats the monitor shows no pulse or pleth, and arterial
    # pressure settles at the mean systemic filling pressure.
    state = engine.state
    assert state.display_hr == 0.0 and state.co == 0.0 and state.oxygen_delivery_ratio == 0.0
    assert state.art_sbp == state.art_dbp == state.art_map == pytest.approx(engine.hemo.state.mcfp, abs=0.1)
    assert 5 < state.art_map < 15
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


class _ShockSucceeds:
    """Stands in for the shock generator so termination is certain."""

    def random(self):
        return 0.0


class _NoRecurrence:
    """Isolate myocardial recovery from the separately tested rhythm hazards."""

    def exponential(self):
        return float("inf")


def _vf_engine(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=7), start=True)
    engine._shock_rng = _ShockSucceeds()
    engine.arrhythmias.rng = _NoRecurrence()
    engine.set_rhythm("VFIB")
    return engine


def _run(engine, seconds):
    for _ in range(round(seconds * 10)):
        engine.step(0.1)


@pytest.mark.parametrize(("vf_seconds", "recovers"), [(240, True), (360, False)])
def test_shock_alone_recovers_circulation_and_consciousness_only_early(engine_factory, vf_seconds, recovers):
    """Weisfeldt 2002: in the first ~4 minutes of VF, defibrillation alone usually restores circulation."""
    engine = _vf_engine(engine_factory)
    _run(engine, vf_seconds - CHARGE_TIME_S)
    engine.charge_defibrillator()
    _run(engine, CHARGE_TIME_S)
    # Without coronary flow the myocardium is ischemic despite oxygenated arterial blood.
    assert engine.state.myocardial_ischemia > 0.9 and engine.state.sao2 > 98
    engine.defibrillate()
    assert engine.hemo.rhythm_type == RhythmType.SINUS_BRADY
    _run(engine, 300)
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
    # Injury becomes irreversible with electrical silence, about 11 minutes after circulation stops.
    for _ in range(50):
        engine.step(0.1)
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


def test_compressions_perfuse_the_heart_so_a_shock_restores_circulation(engine_factory):
    engine = _vf_engine(engine_factory)
    _run(engine, 15)
    engine.set_chest_compressions(True)
    _run(engine, 120)
    state, coronary = engine.state, engine.hemo.coronary
    assert state.cardiac_arrest and state.chest_compressions
    # Meaney 2013 targets: diastolic pressure near 25 mmHg and EtCO2 above 10 mmHg.
    assert 60 < state.art_sbp < 80 and 20 < state.art_dbp < 30
    assert 10 < state.display_etco2 < 25
    # The heart keeps its energy, and low brain flow only slowly adds coma exposure.
    assert coronary.energy_reserve > 0.8 and coronary.energy_failure < 0.05
    assert engine.cerebral.anoxic_s < 60
    # Compression artifact covers the VF.
    assert _ecg_range(engine) > 1.0
    cpr_etco2 = state.display_etco2
    engine.charge_defibrillator()
    _run(engine, CHARGE_TIME_S)
    engine.defibrillate()
    _run(engine, 5)
    engine.set_chest_compressions(False)
    _run(engine, 120)
    assert not state.cardiac_arrest and state.map > 55
    # Washout of tissue CO2 raises EtCO2 abruptly once the heart pumps.
    assert state.display_etco2 > cpr_etco2 + 10


def test_compressions_make_fine_vf_coarse_before_a_late_shock(engine_factory):
    engine = _vf_engine(engine_factory)
    _run(engine, 480)
    coronary = engine.hemo.coronary
    assert coronary.energy_failure > 0.85
    engine.set_chest_compressions(True)
    _run(engine, 120)
    # Reperfusion restores excitation but not injured contraction.
    injury = coronary.injury
    assert coronary.energy_failure < 0.3 and 0.3 < injury < 0.7
    engine.charge_defibrillator()
    _run(engine, CHARGE_TIME_S)
    engine.defibrillate()
    _run(engine, 10)
    engine.set_chest_compressions(False)
    _run(engine, 180)
    # Stunned myocardium leaves hypotension that needs pressor support.
    assert not engine.state.cardiac_arrest and 30 < engine.state.map < 60
    assert coronary.injury == injury


def test_epinephrine_raises_diastolic_pressure_once_compressions_circulate_it(engine_factory):
    engine = _vf_engine(engine_factory)
    _run(engine, 15)
    engine.give_drug_bolus("epi", 1000)
    _run(engine, 30)
    # Without flow, the dose does not reach its effect site.
    assert engine.state.epi_ce < 1.0
    engine.set_chest_compressions(True)
    _run(engine, 5)
    before = engine.state.art_dbp
    _run(engine, 90)
    assert engine.state.art_dbp > before + 8


@pytest.mark.parametrize("dt", [0.01, 0.1])
def test_doses_wait_for_circulation_before_clearance_and_neuromuscular_or_cardiac_effects(engine_factory, dt):
    engine = engine_factory(config=SimulationConfig(rng_seed=7, dt=dt), start=True)
    engine.set_rhythm("VFIB")
    for _ in range(round(15 / dt)):
        engine.step(dt)
    for drug, dose in (("epi", 1000), ("roc", 70), ("propofol", 140)):
        engine.give_drug_bolus(drug, dose)
    epinephrine = engine.pk["epi"]
    dose = epinephrine.v1 * epinephrine.state.c1
    for _ in range(round(120 / dt)):
        engine.step(dt)
    assert engine.state.co == 0
    assert epinephrine.v1 * epinephrine.state.c1 == pytest.approx(dose)
    assert engine.state.epi_ce == engine.state.roc_ce == engine.tof_pd.ce_central == 0
    assert engine.hemo.ce_prop_cv == 0
    assert engine.state.tof == 100

    engine.set_chest_compressions(True)
    for _ in range(round(120 / dt)):
        engine.step(dt)
    assert engine.state.epi_ce > 10
    assert engine.state.roc_ce > 1 and engine.tof_pd.ce_central > 1
    assert engine.hemo.ce_prop_cv > 0
    assert engine.state.tof < 50
    assert epinephrine.v1 * epinephrine.state.c1 < dose


@pytest.mark.parametrize("dt", [0.01, 0.1])
def test_primary_asystole_can_recover_with_coronary_perfusion(engine_factory, dt):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=7, dt=dt), start=True)
    engine.set_rhythm("ASYSTOLE")
    for _ in range(round(16 / dt)):
        engine.step(dt)
    assert engine.state.cardiac_arrest and engine.state.intrinsic_co == 0
    engine.set_chest_compressions(True)
    engine.give_drug_bolus("epi", 1000)
    for _ in range(round(180 / dt)):
        engine.step(dt)
    engine.set_chest_compressions(False)
    for _ in range(round(30 / dt)):
        engine.step(dt)
    assert engine.hemo.rhythm_type == RhythmType.SINUS
    assert not engine.state.cardiac_arrest
    assert engine.state.intrinsic_co > 2 and engine.state.map > 50


@pytest.mark.parametrize("cause", ["no_compressions", "empty_heart", "irreversible_injury"])
def test_primary_asystole_needs_perfusion_and_viable_myocardium(engine_factory, cause):
    engine = engine_factory(config=SimulationConfig(rng_seed=7), start=True)
    engine.set_rhythm("ASYSTOLE")
    if cause == "empty_heart":
        engine.hemo.add_volume(-engine.hemo.blood_volume)
    if cause == "irreversible_injury":
        engine.hemo.coronary.injury = 1.0
    engine.set_chest_compressions(cause != "no_compressions")
    engine.give_drug_bolus("epi", 1000)
    _run(engine, 180)
    assert engine.hemo.rhythm_type == RhythmType.ASYSTOLE
    assert engine.state.intrinsic_co == 0 and engine.state.cardiac_arrest


def test_primary_asystole_recovery_requires_sustained_coronary_support(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=7), start=True)
    engine.set_rhythm("ASYSTOLE")
    _run(engine, 16)
    engine.set_chest_compressions(True)
    engine.give_drug_bolus("epi", 1000)
    _run(engine, 20)
    assert engine.hemo.rhythm_type == RhythmType.ASYSTOLE
    engine.set_chest_compressions(False)
    _run(engine, 2)
    engine.set_chest_compressions(True)
    _run(engine, 20)
    assert engine.hemo.rhythm_type == RhythmType.ASYSTOLE
    _run(engine, 120)
    assert engine.hemo.rhythm_type == RhythmType.SINUS
    assert engine.state.intrinsic_co > 2


def test_compressions_cannot_perfuse_an_empty_heart_until_volume_returns(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=7), start=True)
    engine.start_hemorrhage(800)
    while not engine.state.cardiac_arrest:
        engine.step(0.1)
    engine.stop_hemorrhage()
    engine.set_chest_compressions(True)
    _run(engine, 60)
    assert engine.state.co < 0.1 and engine.state.display_etco2 < 5
    engine.give_fluid(2000)
    engine.give_blood(600)
    _run(engine, 300)
    engine.set_chest_compressions(False)
    _run(engine, 30)
    assert not engine.state.cardiac_arrest and engine.state.map > 60


def test_synchronized_shocks_wait_for_r_waves_and_cannot_fire_twice(engine_factory):
    engine = engine_factory(start=True)
    engine._shock_rng = _ShockSucceeds()
    assert engine.defibrillate() == ShockResult.NOT_READY
    engine.set_rhythm("VFIB")
    engine.charge_defibrillator()
    _run(engine, CHARGE_TIME_S)
    assert engine.defibrillate(synchronized=True) == ShockResult.NOT_DELIVERED
    assert engine.defibrillator.status(engine.state.time) == "ready"
    assert not engine.actions.since_case(ACTION_SHOCK)
    for rhythm in ("SVT", "AFIB", "VTACH"):
        engine.set_rhythm(rhythm)
        engine.charge_defibrillator()
        _run(engine, CHARGE_TIME_S)
        requested = engine.state.time
        count = len(engine.actions.since_case(ACTION_SHOCK))
        assert engine.defibrillate(synchronized=True) == ShockResult.PENDING
        assert engine.defibrillate() == ShockResult.PENDING
        assert engine.hemo.rhythm_type != RhythmType.SINUS
        assert len(engine.actions.since_case(ACTION_SHOCK)) == count
        for _ in range(200):
            engine.step(0.01)
            if engine.defibrillator.last_shock and engine.defibrillator.last_shock[0] > requested:
                break
        assert engine.hemo.rhythm_type == RhythmType.SINUS
        assert len(engine.actions.since_case(ACTION_SHOCK)) == count + 1
        assert requested <= engine.defibrillator.last_shock[0] <= requested + 2
        assert engine.defibrillate() == ShockResult.NOT_READY
    shocks = engine.actions.since_case(ACTION_SHOCK)
    assert [record.label for record in shocks] == ["synchronized"] * 3
