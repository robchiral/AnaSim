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
    fluids = require_fluid_bolus(500, labels=("crystalloid",), event_name="sepsis")

    def start(engine) -> None:
        start_map.record(engine)
        engine.start_sepsis()
        engine.sepsis.severity = 1.0

    steps = [
        create_observe_step("sepsis"),
        ScenarioStep(
            id="RECOGNIZE_WARM_SHOCK",
            title="Recognize septic shock",
            instruction=(
                "Recognize falling pressure and impaired perfusion in the setting of infection; "
                "begin treatment before profound hypotension develops. "
                "Vasodilation is the main mechanism in this case; HR may rise, but tachycardia is variable. "
                "In practice, assess perfusion and consider blood loss, anesthetic effects, and other causes of hypotension."
            ),
            check_requirements=require_hypotension_or_treatment(
                start_map, "sepsis", "begin fluid resuscitation and support pressure",
            ),
        ),
        ScenarioStep(
            id="GIVE_FLUIDS",
            title="Initial fluid resuscitation",
            instruction=(
                "Start a <b>500 mL crystalloid bolus</b> in Circulation and reassess "
                "the response. Balanced crystalloid is preferred in practice; the simulator's crystalloid is lactated Ringer's. "
                "An initial 30 mL/kg over the first 3 hours is a guideline starting point, "
                "with further fluid guided by perfusion and response."
            ),
            check_requirements=fluids,
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Start vasopressor",
            instruction=(
                "If <b>MAP remains below 65 mmHg</b>, start or adjust norepinephrine. "
                "A starting rate of 0.05-0.1 mcg/kg/min is 3.5-7 mcg/min for 70 kg; "
                "enter the absolute rate and titrate to pressure and perfusion. "
                "Start vasopressors alongside fluids in severe hypotension. If pressure has recovered, continue without adding one."
            ),
            check_requirements=require_vasopressor_if_hypotensive(
                "nore", fail_message="Start or adjust norepinephrine for persistent hypotension",
                event_name="sepsis",
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="SOURCE_CONTROL",
            title="Source control",
            instruction=(
                "Select <b>Request surgical source control</b> to let the simulated inflammatory "
                "effects subside. Continue hemodynamic support as the patient recovers. "
                "In practice, obtain blood cultures without delaying antibiotics, and give antibiotics "
                "immediately (ideally within 1 hour for shock). Shock can persist after both."
            ),
            check_requirements=require_crisis_stopped("sepsis", "Request surgical source control"),
        ),
        create_reassess_step(
            "sepsis",
            "Request surgical source control",
            "Infection treatment underway",
            "Reassess the response before giving more fluid and wean vasopressors as perfusion improves. "
            "Lactate, urine output, examination, and bedside ultrasound inform reassessment in practice."
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
