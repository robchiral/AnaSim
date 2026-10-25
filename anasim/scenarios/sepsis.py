"""Warm septic shock scenario."""

from .base import (
    CrisisStartMAP,
    Scenario,
    ScenarioStep,
    create_observe_baseline_step,
    create_reassess_step,
    require_crisis_started,
    require_crisis_stopped,
    require_fluid_bolus,
    require_hypotension_or_treatment,
    require_vasopressor_if_hypotensive,
)


def create_sepsis_response() -> Scenario:
    start_map = CrisisStartMAP()
    steps = [
        create_observe_baseline_step("sepsis"),
        ScenarioStep(
            id="START_SEPSIS",
            title="Sepsis begins",
            instruction=(
                "This patient has an intra-abdominal infection. Select <b>Start sepsis</b> "
                "in Events to model developing vasodilatory shock. "
                "Call for help and resuscitate promptly. In practice, obtain blood cultures without delaying "
                "antibiotics, give antibiotics immediately (ideally within 1 hour for shock), "
                "and arrange source control."
            ),
            check_requirements=require_crisis_started(
                "sepsis",
                "Start sepsis event (Events tab)",
            ),
            target_tab="Events",
            target_section="circulation",
            on_met=start_map.record,
        ),
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
                "Start a <b>500 mL crystalloid bolus</b> in Fluids and reassess "
                "the response. Balanced crystalloid is preferred in practice; the simulator uses one crystalloid model. "
                "An initial 30 mL/kg over the first 3 hours is a guideline starting point, "
                "with further fluid guided by perfusion and response."
            ),
            check_requirements=require_fluid_bolus(
                500,
                labels=("crystalloid",),
                event_name="sepsis",
            ),
            target_tab="Fluids",
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
                "Select <b>Stop sepsis</b> to let the simulated inflammatory effects subside. "
                "Continue hemodynamic support as the patient recovers. "
                "This control represents recovery after antibiotics and source control; real shock can persist after both."
            ),
            check_requirements=require_crisis_stopped(
                "sepsis",
                "Select Stop sepsis to model recovery after infection treatment",
            ),
            target_tab="Events",
            target_section="circulation",
        ),
        create_reassess_step(
            "sepsis",
            "Select Stop sepsis to model recovery after infection treatment",
            "Infection treatment underway",
            "Reassess the response before giving more fluid and wean vasopressors as perfusion improves. "
            "Lactate, urine output, examination, and bedside ultrasound inform reassessment in practice."
        ),
    ]

    return Scenario(
        id="sepsis_response",
        name="Septic shock response",
        description="Recognize and manage early distributive septic shock.",
        steps=steps,
    )
