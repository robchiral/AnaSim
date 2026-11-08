"""Clinical courses for MH recognition, trigger withdrawal, and specific rescue."""

import json

import pytest

from anasim.core.state import SimulationConfig
from anasim.web import WebSession


def _run(engine, seconds, dt=1.0):
    for _ in range(round(seconds / dt)):
        engine.step(dt)


@pytest.mark.parametrize("dt", [0.1, 1.0])
def test_untreated_mh_raises_co2_before_fever_and_persists_under_paralysis(engine_factory, dt):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="balanced"), start=True)
    engine.start_malignant_hyperthermia()
    engine.give_drug_bolus("roc", 100)
    _run(engine, 180, dt)
    assert engine.state.pa_co2 > 50 and engine.state.temp_c < 38
    assert engine.state.hr > 95 and engine.state.muscle_rigidity
    assert engine.state.twitch < 5
    _run(engine, 600, dt)
    assert engine.state.temp_c > 37.5 and engine.state.hr > 120
    assert engine.state.ph < 7.25 and engine.state.lactate > 2


@pytest.mark.parametrize("trigger", ["none", "n2o", "succinylcholine", "sevoflurane"])
def test_mh_starts_only_with_a_triggering_agent(engine_factory, trigger):
    engine = engine_factory(config=SimulationConfig(mode="steady_state"), start=True)
    if trigger == "succinylcholine":
        engine.give_drug_bolus(trigger, 100)
    elif trigger == "sevoflurane":
        engine.set_vaporizer(3)
    elif trigger == "n2o":
        engine.set_fgf(3, 0, 3)
    _run(engine, 60)
    if trigger in {"none", "n2o"}:
        with pytest.raises(ValueError, match="requires sevoflurane"):
            engine.start_malignant_hyperthermia()
        return
    engine.start_malignant_hyperthermia()
    _run(engine, 120)
    assert engine.state.muscle_rigidity


@pytest.mark.parametrize("weight", [55, 100])
@pytest.mark.parametrize("treatment", ["trigger_removal", "cooling", "dantrolene"])
def test_cooling_and_hyperventilation_do_not_replace_dantrolene(engine_factory, weight, treatment):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="balanced"),
                            weight=weight, start=True)
    engine.start_malignant_hyperthermia()
    _run(engine, 180)
    peak_co2, peak_hr = engine.state.pa_co2, engine.state.hr
    engine.set_vaporizer(0)
    engine.set_fgf(10, 0)
    engine.set_drug_rate("propofol", 100)
    engine.set_vent_settings(rr=28, vt=0.5, peep=5, ie="1:2", mode="VCV")
    if treatment == "cooling":
        engine.set_surface_cooling(True)
    if treatment == "dantrolene":
        engine.give_drug_bolus("dantrolene", 2.5 * weight)
    _run(engine, 600)
    if treatment == "dantrolene":
        assert engine.malignant_hyperthermia.severity < 0.05
        assert engine.state.pa_co2 < peak_co2 and engine.state.hr < peak_hr
        assert not engine.state.muscle_rigidity
    else:
        assert engine.malignant_hyperthermia.severity > 0.9
        assert engine.state.lactate > 1 and engine.state.ph < 7.3


def test_repeat_dantrolene_controls_an_undertreated_crisis(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="balanced"), start=True)
    engine.start_malignant_hyperthermia()
    _run(engine, 180)
    engine.give_drug_bolus("dantrolene", 0.25 * engine.patient.weight)
    _run(engine, 180)
    undertreated = engine.malignant_hyperthermia.severity
    assert undertreated > 0.4
    engine.give_drug_bolus("dantrolene", 2.5 * engine.patient.weight)
    _run(engine, 300)
    assert engine.malignant_hyperthermia.severity < 0.1 * undertreated


def test_dantrolene_depresses_twitch_without_fade_or_hypoventilation(engine_factory):
    """Flewellen 1983: in awake volunteers, twitch depression reached a 75%
    plateau at about 2.4 mg/kg, and ventilation was unchanged."""
    engine = engine_factory(start=True)
    _run(engine, 60)
    paco2 = engine.state.pa_co2
    engine.give_drug_bolus("dantrolene", 2.4 * engine.patient.weight)
    _run(engine, 600)
    assert 20 < engine.state.twitch < 35
    assert engine.state.tof == pytest.approx(100)
    assert engine.state.pa_co2 == pytest.approx(paco2, abs=1)


def test_dantrolene_stops_hypermetabolism_while_cooling_removes_accumulated_heat(engine_factory):
    temperatures = []
    for cooling in (False, True):
        engine = engine_factory(config=SimulationConfig(mode="steady_state", maint_type="balanced"), start=True)
        engine.start_malignant_hyperthermia()
        _run(engine, 180)
        # A later presentation with accumulated heat, after the early CO2 rise.
        engine.state.temp_c = 40.0
        engine.set_vaporizer(0)
        engine.set_fgf(10, 0)
        engine.give_drug_bolus("dantrolene", 175)
        engine.set_surface_cooling(cooling)
        _run(engine, 300)
        assert engine.malignant_hyperthermia.severity < 0.05
        temperatures.append(engine.state.temp_c)
    assert temperatures[0] > 39.5
    assert temperatures[1] < temperatures[0] - 0.3


def test_independent_mh_case_supports_treatment_and_counts_weight_based_dantrolene():
    session = WebSession({"scenario_id": "malignant_hyperthermia", "practice_mode": "independent", "weight": 100})
    session.engine.config.dt = 0.1
    info = json.loads(session.info())
    assert next(d for d in info["drugs"] if d["key"] == "dantrolene")["default_bolus"] == 250
    session.command("run", '{"running":true}')
    session.command("speed", '{"value":50}')
    for _ in range(40):
        session.step(0.1)
    with pytest.raises(ValueError, match="controlled by the case"):
        session.command("malignant_hyperthermia", '{"active":false}')
    for name, args in [
        ("vaporizer", {"percent": 0}), ("fgf", {"o2": 10, "air": 0, "n2o": 0}),
        ("vent", {"rr": 28}), ("drug_rate", {"key": "propofol", "rate": 100}),
        ("drug_bolus", {"key": "dantrolene", "amount": 250}),
        ("bair_hugger", {"target_c": 0}),
    ]:
        session.command(name, json.dumps(args))
    for _ in range(100):
        session.step(0.1)
    assert session.snapshot()["controls"]["dantrolene_total_mg"] == 250
    report = json.loads(session.command("case_finish"))["debrief"]
    assert all(o["achieved"] for o in report["outcomes"])
    assert any("Dantrolene 250 mg" in item["text"] for item in report["timeline"])


def test_untreated_mh_case_earns_no_treatment_credit():
    """Spontaneous hyperpnea raises minute ventilation without a ventilator change."""
    session = WebSession({"scenario_id": "malignant_hyperthermia", "practice_mode": "independent"})
    session.engine.config.dt = 0.1
    session.command("run", '{"running":true}')
    session.command("speed", '{"value":50}')
    for _ in range(180):
        session.step(0.1)
    assert session.engine.state.mv > 10
    report = json.loads(session.command("case_finish"))["debrief"]
    assert not any(o["achieved"] for o in report["outcomes"] if o["id"] != "stability")
