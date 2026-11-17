from itertools import pairwise

import pytest

from anasim.core.clinical_actions import check_intubation
from anasim.core.state import SimulationConfig
from anasim.physiology.cerebral import CerebralModel


def _run(engine, seconds):
    for _ in range(round(seconds * 10)):
        engine.step(0.1)


@pytest.mark.parametrize(("map_mmhg", "saturation"), [(0, 98), (90, 10)])
def test_severe_supply_failure_suppresses_responsiveness_bis_and_breathing_until_it_returns(
        awake_engine, monkeypatch, map_mmhg, saturation):
    brain = awake_engine.cerebral
    # Moderate hypotension and desaturation leave cerebral function intact.
    brain.step(600, 60, 13.5, 90, 40, 7.4)
    assert brain.dysfunction == 0
    step = CerebralModel.step
    monkeypatch.setattr(brain, "step", lambda dt, *_: step(brain, dt, map_mmhg, brain.baseline_hb, saturation, 40, 7.4))
    # Function starts to fail after a few seconds; BIS shows it after the monitor's processing delay.
    _run(awake_engine, 4)
    assert brain.dysfunction < 0.1
    _run(awake_engine, 8)
    assert brain.dysfunction > 0.85 and awake_engine.state.bis > 80
    _run(awake_engine, 28)
    assert awake_engine.state.bis < 5 and awake_engine.state.loc > 0.95
    assert awake_engine.resp.state.effort_rr < 5
    assert check_intubation(awake_engine) is None
    monkeypatch.undo()
    _run(awake_engine, 180)
    assert awake_engine.state.loc < 0.01 and awake_engine.state.bis > 80
    feedback = check_intubation(awake_engine)
    assert feedback is not None and feedback.blocked


def test_persistent_coma_does_not_force_flat_bis_after_oxygen_and_breathing_return(awake_engine):
    brain = awake_engine.cerebral
    brain.step(CerebralModel.COMA_EXPOSURE_S, 0, 13.5, 98, 40, 7.4)
    brain.step(600, 90, 13.5, 98, 40, 7.4)
    _run(awake_engine, 30)
    assert awake_engine.state.bis > 80 and awake_engine.state.loc > 0.95
    assert awake_engine.resp.state.effort_rr > 5


@pytest.mark.parametrize(("pa_co2", "ph", "low", "high"), [
    (40, 7.4, 0, 0.01), (40, 7.0, 0, 0.01),
    (60, 7.3, 0, 0.01), (80, 7.25, 0.1, 0.2),
    (100, 7.2, 0.45, 0.55), (100, 7.1, 0.99, 1.0),
    (100, 7.38, 0, 0.01), (140, 7.38, 0.99, 1.0),
])
def test_co2_response_depends_on_acidemia_and_retention_without_anoxic_injury(pa_co2, ph, low, high):
    brain = CerebralModel(13.5)
    brain.step(300, 90, 13.5, 98, pa_co2, ph)
    assert low <= brain.unresponsiveness <= high
    assert brain.cortical_function == 1 and brain.anoxic_s == 0


def test_co2_narcosis_is_delayed_reversible_and_separate_from_bis(awake_engine, monkeypatch):
    brain = awake_engine.cerebral
    step = CerebralModel.step
    monkeypatch.setattr(brain, "step", lambda dt, *_: step(brain, dt, 90, 13.5, 98, 110, 7.1))
    _run(awake_engine, 1)
    assert 0 < brain.co2_narcosis < 0.1
    _run(awake_engine, 179)
    assert awake_engine.state.loc > 0.99
    assert awake_engine.state.bis > 80
    assert brain.anoxic_s == 0
    monkeypatch.undo()
    _run(awake_engine, 180)
    assert brain.co2_narcosis < 0.01
    assert awake_engine.state.loc < 0.01


def test_oxygenated_hypoventilation_impairs_response_and_ventilation_reverses_it(awake_engine):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.set_fgf(10, 0)
    engine.give_drug_bolus("roc", 70)
    engine.set_drug_rate("roc", 8)
    engine.set_vent_settings(mode="VCV", rr=6, vt=0.2, peep=5, ie="1:2")
    engine.set_vent_power(True)
    for _ in range(1200):
        engine.step(1)
    state = engine.state
    assert state.pa_co2 > 90 and state.ph < 7.15
    assert state.sao2 > 98 and state.map > 65 and state.twitch < 1
    assert state.loc > 0.6 and state.bis > 80
    assert engine.cerebral.anoxic_s == 0
    engine.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5, ie="1:2")
    for _ in range(1200):
        engine.step(1)
    assert state.pa_co2 < 50 and state.loc < 0.01


def test_elderly_arrest_bis_delay_and_anemic_hypotension_dysfunction(engine_factory):
    """Elderly BIS smoothing still reflects arrest ischemia promptly; severe anemia
    compounds cerebral oxygen delivery failure during moderate hypotension.
    """
    elderly = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True, age=85)
    elderly.set_rhythm("VFIB")
    for _ in range(70):
        elderly.step(0.5)
    assert elderly.state.bis < 10.0

    anemic, normal_hb = CerebralModel(6.0), CerebralModel(14.0)
    anemic.step(120.0, 34.0, 6.0, 95.0, 40.0, 7.4)
    normal_hb.step(120.0, 34.0, 14.0, 95.0, 40.0, 7.4)
    assert anemic.dysfunction > normal_hb.dysfunction + 0.15


def test_cerebrovascular_co2_reactivity_hypoxia_protection_and_csf_adaptation():
    """Brian 1998, Battisti-Charbonney 2011, and Willie 2015: hypercapnia augments
    hypoxic cerebral vasodilation up to ~2x baseline CBF to preserve O2 delivery
    (Willie 2015), while normotensive hypocapnia lowers CBF reserve without causing
    normoxic neuronal failure, yields to pressure-limited flow during severe hypotension,
    and adapts over hours (Brian 1998).
    """
    normoxic_hypocapnic = CerebralModel(13.5)
    normoxic_hypocapnic.step(120.0, 90.0, 13.5, 98.0, 18.0, 7.58)
    assert 0.60 < normoxic_hypocapnic.tissue_oxygen < 0.70
    assert normoxic_hypocapnic.dysfunction == 0.0 and normoxic_hypocapnic.anoxic_s == 0.0

    # Willie 2015 end-clamp blood gases (SaO2 ~48%, PaCO2 ~54 mmHg from pre-clamp ~37 mmHg)
    # preserve global cerebral O2 delivery, whereas hypocapnia at the same SaO2 causes dysfunction.
    hypercapnic_hypox = CerebralModel(14.5, baseline_paco2=37.0)
    hypocapnic_hypox = CerebralModel(14.5, baseline_paco2=40.0)
    hypercapnic_hypox.step(120.0, 111.0, 14.5, 48.3, 54.3, 7.35)
    hypocapnic_hypox.step(120.0, 90.0, 14.5, 48.3, 22.0, 7.54)
    assert hypercapnic_hypox.tissue_oxygen > 0.85 and hypercapnic_hypox.dysfunction == 0.0
    assert hypocapnic_hypox.dysfunction > 0.35

    # Severe hypotension makes flow pressure-limited despite hypocapnia (Brian 1998 Table 1).
    low_map_hypocap, low_map_normocap = CerebralModel(13.5), CerebralModel(13.5)
    low_map_hypocap.step(120.0, 34.0, 13.5, 98.0, 20.0, 7.55)
    low_map_normocap.step(120.0, 34.0, 13.5, 98.0, 40.0, 7.40)
    assert low_map_hypocap.dysfunction == pytest.approx(low_map_normocap.dysfunction, abs=1e-4)

    # Sustained hypocapnia adapts CSF bicarbonate over hours, and chronic retainers start adapted.
    adapted = CerebralModel(13.5)
    for _ in range(720):
        adapted.step(60.0, 90.0, 13.5, 98.0, 22.0, 7.48)
    assert adapted.tissue_oxygen > 0.90
    chronic_retainer = CerebralModel(14.0, baseline_paco2=55.0)
    chronic_retainer.step(120.0, 90.0, 14.0, 96.0, 55.0, 7.36)
    assert chronic_retainer.tissue_oxygen == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(("hb", "saturation"), [(14.5, 98.0), (14.5, 48.3), (8.0, 98.0)])
@pytest.mark.parametrize("dt", [1.0, 10.0])
def test_pressure_recovery_during_hypocapnia_preserves_cerebral_function(hb, saturation, dt):
    outcomes = []
    for map_mmhg in (20.0, 30.0, 34.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 75.0, 90.0):
        brain = CerebralModel(14.5)
        for _ in range(round(120.0 / dt)):
            brain.step(dt, map_mmhg, hb, saturation, 22.0, 7.54)
        outcomes.append((brain.tissue_oxygen, brain.unresponsiveness))

    for lower, higher in pairwise(outcomes):
        assert higher[0] >= lower[0] - 1e-12
        assert higher[1] <= lower[1] + 1e-12


def test_extreme_hypercapnia_depresses_medullary_drive_and_prolonged_coma_lags_recovery(
    engine_factory, monkeypatch
):
    """Westlake 1955 & Sieker and Hickam 1956: medullary respiratory depression sets in
    at deeper hypercapnic coma than drowsiness, and coma >1 h lags blood-gas normalization.
    """
    moderate, metabolic, compensated, alkali_extreme = (CerebralModel(13.5) for _ in range(4))
    moderate.step(300.0, 90.0, 13.5, 98.0, 80.0, 7.22)
    metabolic.step(300.0, 90.0, 13.5, 98.0, 35.0, 7.00)
    compensated.step(300.0, 90.0, 13.5, 98.0, 100.0, 7.34)
    alkali_extreme.step(300.0, 90.0, 13.5, 98.0, 145.0, 7.26)
    assert moderate.respiratory_drive == pytest.approx(1.0, abs=1e-3)
    assert metabolic.respiratory_drive == pytest.approx(1.0, abs=1e-3)
    assert compensated.respiratory_drive == pytest.approx(1.0, abs=1e-3)
    assert alkali_extreme.respiratory_drive < 0.10

    # Brief (<1 h) vs prolonged (2 h) hypercapnic coma recovery lag, checked across step sizes.
    brief = CerebralModel(13.5)
    brief.step(900.0, 90.0, 13.5, 98.0, 135.0, 7.05)
    brief.step(180.0, 90.0, 13.5, 98.0, 42.0, 7.39)
    assert brief.unresponsiveness < 0.01

    fine, coarse = CerebralModel(13.5), CerebralModel(13.5)
    for _ in range(7200):
        fine.step(1.0, 90.0, 13.5, 98.0, 135.0, 7.05)
    for _ in range(720):
        coarse.step(10.0, 90.0, 13.5, 98.0, 135.0, 7.05)
    for _ in range(600):
        fine.step(1.0, 90.0, 13.5, 98.0, 45.0, 7.38)
    for _ in range(60):
        coarse.step(10.0, 90.0, 13.5, 98.0, 45.0, 7.38)
    assert fine.unresponsiveness == pytest.approx(coarse.unresponsiveness, abs=0.02)
    assert fine.unresponsiveness > 0.50 and fine.anoxic_s == 0.0
    for _ in range(180):
        fine.step(60.0, 90.0, 13.5, 98.0, 42.0, 7.40)
    assert fine.unresponsiveness < 0.01 and fine.respiratory_drive > 0.99

    # Engine check: with patent intubated airway on O2, extreme hypercapnic coma
    # depresses medullary effort and causes CO2 retention while BIS stays normal.
    engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    engine.set_airway_mode("ETT")
    engine.set_fgf(6.0, 0.0)
    step = CerebralModel.step
    monkeypatch.setattr(
        engine.cerebral, "step", lambda dt, *_: step(engine.cerebral, dt, 90.0, 13.5, 99.0, 90.0, 7.22)
    )
    for _ in range(180):
        engine.step(1.0)
    stim_mv, stim_paco2 = engine.resp.state.mv, engine.state.pa_co2
    monkeypatch.setattr(
        engine.cerebral, "step", lambda dt, *_: step(engine.cerebral, dt, 90.0, 13.5, 99.0, 140.0, 7.04)
    )
    for _ in range(240):
        engine.step(1.0)
    assert engine.state.loc > 0.95 and engine.state.bis > 80.0
    assert engine.resp.state.mv < 0.3 * stim_mv
    assert engine.state.pa_co2 > stim_paco2 + 12.0
