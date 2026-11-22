"""Warm septic shock scenario."""

from .base import (
    CRISIS_BRIEFING,
    CRISIS_ONSET_S,
    CaseAction,
    CaseEvent,
    CrisisStartMAP,
    Scenario,
    ScenarioStep,
    create_observe_step,
    create_reassess_step,
    crisis_outcomes,
    require_crisis_stopped,
    require_fluid_bolus,
    require_hypotension_or_treatment,
    require_vasopressor_if_hypotensive,
)


def create_sepsis_response() -> Scenario:
    start_map = CrisisStartMAP()
    fluids = require_fluid_bolus(500, labels=("crystalloid", "saline"), event_name="sepsis")

    def start(engine) -> None:
        start_map.record(engine)
        engine.sepsis.onset_rate = 1.0 / 60.0
        engine.start_sepsis()

    steps = [
        create_observe_step("sepsis"),
        ScenarioStep(
            id="RECOGNIZE_WARM_SHOCK",
            title="Recognize septic shock",
            instruction=(
                "Falling blood pressure with active infection suggests septic shock; treat before severe hypotension develops. "
                "Vasodilation drives the drop here, and tachycardia is variable.<br><br>"
                "<i>Also consider blood loss, anesthetic depth, and other causes of hypotension.</i>"
            ),
            check_requirements=require_hypotension_or_treatment(
                start_map, "sepsis", "begin fluid resuscitation and support pressure",
            ),
        ),
        ScenarioStep(
            id="GIVE_FLUIDS",
            title="Give fluids",
            instruction=(
                "Give a <b>500 mL crystalloid bolus</b> (lactated Ringer's is preferred over 0.9% saline) "
                "and reassess the response.<br><br>"
                "<i>Initial sepsis resuscitation starts around 30 mL/kg over 3 hours, with further fluid guided by perfusion.</i>"
            ),
            check_requirements=fluids,
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Start vasopressor",
            instruction=(
                "If <b>MAP < 65 mmHg</b>, start or adjust <b>norepinephrine</b> in mcg/min "
                "(0.05-0.1 mcg/kg/min = 3.5-7 mcg/min for 70 kg) and titrate to pressure and perfusion. "
                "Start pressors alongside fluids in severe hypotension; skip this if pressure has already recovered."
            ),
            check_requirements=require_vasopressor_if_hypotensive(
                "nore", fail_message="Start or adjust norepinephrine for persistent hypotension",
                event_name="sepsis",
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="SOURCE_CONTROL",
            title="Request source control",
            instruction=(
                "Select <b>Request surgical source control</b> and continue hemodynamic support as the patient stabilizes.<br><br>"
                "<i>Draw blood cultures without delaying broad-spectrum antibiotics, ideally within 1 hour; "
                "shock can persist after both.</i>"
            ),
            check_requirements=require_crisis_stopped("sepsis", "Request surgical source control"),
        ),
        create_reassess_step(
            "sepsis",
            "Request surgical source control",
            "source control is underway",
            "<i>Reassess response before giving more fluid and wean vasopressors as perfusion improves. "
            "Follow lactate, urine output, exam, and bedside ultrasound.</i>",
        ),
    ]

    return Scenario(
        id="sepsis_response",
        name="Septic shock response",
        briefing=f"{CRISIS_BRIEFING} The patient has an intra-abdominal infection.",
        steps=steps,
        events=(CaseEvent(
            "sepsis", CRISIS_ONSET_S,
            "The surgeon reports purulent contamination. Blood pressure is falling.", start,
        ),),
        actions=(CaseAction(
            "source_control", "Request surgical source control",
            lambda engine: engine.stop_sepsis(), lambda engine: engine.sepsis.active,
            completed_label="Source control requested",
        ),),
        outcomes=crisis_outcomes("sepsis", fluids),
    )
