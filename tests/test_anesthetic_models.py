"""Fixed anesthetic model endpoints and their integrated opioid behavior."""

import json

import numpy as np
import pytest

from anasim.core.projection import stimulus_tolerance
from anasim.core.runtime import step_disturbances
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.patient.pd.anesthesia import BISModel, ClinicalResponseModel
from anasim.web import WebSession, catalog


def test_published_response_thresholds_and_bis_delay():
    # Bouillon 2004 Bayesian table at the cohort's median age: 50% no response
    # with propofol alone.
    response = ClinicalResponseModel(age=33.5)
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


def test_deeper_iv_hypnosis_preserves_cardiovascular_stimulation(engine_factory):
    """Kazama 1997: propofol alone does not change the BP or HR rise to
    stimulation; opioids, volatiles, and lidocaine still blunt it.
    """
    engine = engine_factory(start=True)
    engine.start_disturbance("stim_intubation_pulse")
    engine.state.time += 40.0
    responses = []
    for ce in (3.4, 12.0):
        engine.state.hypnotic_response_ce = engine.state.hypnotic_ce = ce
        engine._tol_current = stimulus_tolerance(engine)
        responses.append(step_disturbances(engine, 0.1))
    lighter, deeper = responses
    assert deeper.bis < 0.01 * lighter.bis
    assert deeper.hr == pytest.approx(lighter.hr)
    assert deeper.svr == pytest.approx(lighter.svr)
    assert deeper.sv == pytest.approx(lighter.sv)
    engine.pk["remi"].state.ce = 5.0
    engine.state.opioid_ce = 5.0
    opioid = step_disturbances(engine, 0.1)
    assert opioid.hr < 0.1 * deeper.hr
    assert opioid.svr < 0.1 * deeper.svr

    engine.pk["remi"].state.ce = engine.state.opioid_ce = 0.0
    engine.state.mac_sevo = 1.5
    volatile = step_disturbances(engine, 0.1)
    engine.state.mac_sevo = 0.0
    engine.state.lidocaine_ce = 4.0
    lidocaine = step_disturbances(engine, 0.1)
    assert 0.0 < volatile.svr < deeper.svr
    assert 0.0 < lidocaine.svr < deeper.svr


def test_remifentanil_blunts_arousal_without_changing_baseline_bis(engine_factory):
    outcomes = []
    for remi_target in (0.0, 4.0):
        engine = engine_factory(
            config=SimulationConfig(mode="awake", tci_enabled=True, rng_seed=2), start=True,
        )
        engine.set_airway_mode("ETT")
        engine.set_vent_power(True)
        # Keep propofol below the calibrated LOC midpoint so the added opioid
        # response can be resolved without saturating both probabilities.
        engine.enable_tci("propofol", 2.0)
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


def test_older_patients_lose_responsiveness_at_lower_propofol_and_higher_bis(engine_factory):
    """Lysakowski 2009: with 0.5 mcg/mL Ce steps held 5 min, patients aged 78
    lost responsiveness to mild prodding/shaking at median BIS 70 (IQR 65-79),
    and those aged 35 at 58. Hold after Cp/Ce equilibration and average the
    last 30 s before each clinical assessment.

    Age 70 is within the older cohort's 67-96 range and AnaSim's supported
    domain. Use male patients of each cohort's mean size, controlled
    ventilation, and Eleveld TCI in place of the study's Schnider TCI.
    """
    at_loss = {}
    for age, weight, height in ((35, 76, 173), (70, 71, 166)):
        engine = engine_factory(
            config=SimulationConfig(mode="awake", tci_enabled=True, concomitant_opioids=False, rng_seed=2),
            start=True, age=age, weight=weight, height=height,
        )
        engine.set_airway_mode("ETT")
        engine.set_vent_power(True)
        steps = []
        for target in np.arange(0.5, 4.6, 0.5):
            engine.enable_tci("propofol", target)
            for _ in range(900):
                engine.step(1.0)
                if abs(engine.state.propofol_cp - engine.state.propofol_ce) < 0.05:
                    break
            else:
                pytest.fail("propofol Cp/Ce did not equilibrate before the 5-min hold")
            readings = []
            for second in range(300):
                engine.step(1.0)
                if second >= 270:
                    readings.append((engine.state.loc, engine.state.propofol_ce, engine.state.bis))
            steps.append(np.mean(readings, axis=0))
            if steps[-1][0] > 0.5:
                break
        loc, ce, bis = np.asarray(steps).T
        assert loc[0] < 0.5 < loc[-1]
        at_loss[age] = np.interp(0.5, loc, ce), np.interp(0.5, loc, bis)
    assert at_loss[35][1] == pytest.approx(58.0, abs=5.0)
    assert 65.0 < at_loss[70][1] < 79.0
    assert at_loss[70][0] < 0.75 * at_loss[35][0]
    assert at_loss[70][1] > at_loss[35][1] + 5.0


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
            assert engine.pk["propofol"].state.ce_response > engine.pk["propofol"].state.ce
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
