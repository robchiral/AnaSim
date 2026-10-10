"""Fixed anesthetic model endpoints and their integrated opioid behavior."""

import json

import pytest

from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.patient.pd.anesthesia import BISModel, ClinicalResponseModel
from anasim.web import WebSession, catalog


def test_published_response_thresholds_and_bis_delay():
    response = ClinicalResponseModel()
    # Bouillon 2004 Bayesian table: 50% no response with propofol alone.
    assert response.loss_of_response(3.2064, 0.0) == pytest.approx(0.5)
    assert response.tolerance(5.5444, 0.0) == pytest.approx(0.5)
    assert response.tolerance(3.0, 4.0) > 0.9
    bis = BISModel(Patient(age=35, weight=70, height=170, sex="male"))
    assert bis.compute_bis(3.07944689) == pytest.approx(46.4912, abs=0.001)
    assert bis.compute_bis(1.0) > 80.0
    assert bis.compute_bis(6.0) < 30.0
    assert bis.delay == pytest.approx(21.116, abs=0.01)
    # Eleveld's delay applies before the displayed BIS responds.
    assert bis.step(20.0, bis.c50) == bis.baseline
    assert bis.step(30.0, bis.c50) == pytest.approx(bis.baseline / 2.0)


def test_remifentanil_blunts_arousal_without_changing_baseline_bis(engine_factory):
    outcomes = []
    for remi_target in (0.0, 4.0):
        engine = engine_factory(
            config=SimulationConfig(mode="awake", tci_enabled=True, rng_seed=2), start=True,
        )
        engine.set_airway_mode("ETT")
        engine.set_vent_power(True)
        engine.enable_tci("propofol", 2.3)
        if remi_target:
            engine.enable_tci("remi", remi_target)
        for _ in range(900):
            engine.step(1.0)
        baseline = engine.state.bis
        loc, tolerance = engine.state.loc, engine.state.tol
        breathing = engine.resp.state.mv
        engine.start_disturbance("stim_intubation_pulse")
        peak = baseline
        for _ in range(500):
            engine.step(0.1)
            peak = max(peak, engine.state.bis)
        outcomes.append((baseline, loc, tolerance, breathing, peak - baseline))
    alone, combined = outcomes
    assert combined[0] == pytest.approx(alone[0], abs=0.5)
    assert combined[1] > alone[1] + 0.5
    assert combined[2] > alone[2] + 0.5
    assert combined[3] < alone[3]
    assert combined[4] < 0.2 * alone[4]


def test_induction_uses_separate_clinical_and_bis_sites(awake_engine):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.set_vent_power(True)
    engine.give_drug_bolus("propofol", 2.0 * engine.patient.weight)
    clinical_onset = bis_onset = None
    for second in range(720):
        engine.step(1.0)
        if clinical_onset is None and engine.state.loc > 0.95:
            clinical_onset = second
            assert engine.pk_prop.state.ce_response > engine.pk_prop.state.ce
        if bis_onset is None and engine.state.bis < 60.0:
            bis_onset = second
    assert clinical_onset is not None and bis_onset is not None
    assert clinical_onset < bis_onset
    assert engine.state.loc < 0.5


def test_browser_has_one_anesthetic_model_set_and_rejects_removed_options():
    assert "models" not in json.loads(catalog())
    for name in ("pk_model_propofol", "bis_model", "loc_model", "pk_model_nore", "pk_model_epi"):
        with pytest.raises(ValueError, match=name):
            WebSession({name: "Eleveld"})
    with pytest.raises(ValueError, match="maintenance initialization includes opioids"):
        WebSession({"mode": "steady_state", "concomitant_opioids": False})
