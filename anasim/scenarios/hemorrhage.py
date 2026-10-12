"""Intraoperative hemorrhage and hypovolemic shock scenario."""

from .base import (
    CrisisStartMAP,
    Scenario,
    ScenarioStep,
    create_observe_baseline_step,
    create_reassess_step,
    require_crisis_started,
    require_crisis_stopped,
    require_fluid_bolus,
    require_hypotension_after_start,
    require_vasopressor_if_hypotensive,
)


def create_hemorrhage_response() -> Scenario:
    start_map = CrisisStartMAP()
    steps = [
        create_observe_baseline_step("hemorrhage"),
        ScenarioStep(
            id="START_HEMORRHAGE",
            title="Hemorrhage begins",
            instruction=(
                "The surgeon reports brisk blood loss. Set the bleeding rate to "
                "<b>500-800 mL/min</b>, then select Start bleeding in Events. "
                "Call for help and ask the surgeon to control the source while resuscitation begins."
            ),
            check_requirements=require_crisis_started(
                "hemorrhage",
                "Start hemorrhage event (Events tab)",
            ),
            target_tab="Events",
            target_section="circulation",
            on_met=start_map.record,
        ),
        ScenarioStep(
            id="RECOGNIZE_SHOCK",
            title="Recognize hypovolemic shock",
            instruction=(
                "Watch for falling blood pressure during bleeding: <b>MAP < 65 mmHg and at least 20% "
                "below its value when bleeding began</b>. HR may rise and pulse pressure may narrow. "
                "Anesthesia and beta blockade can blunt tachycardia; begin treatment as the situation develops."
            ),
            check_requirements=require_hypotension_after_start(
                start_map, "observe the response to blood loss",
            ),
        ),
        ScenarioStep(
            id="STOP_BLEEDING",
            title="Control the bleeding source",
            instruction=(
                "Select <b>Stop bleeding</b> to represent surgical hemostasis. "
                "In practice, hemorrhage control and resuscitation proceed together."
            ),
            check_requirements=require_crisis_stopped(
                "hemorrhage", "Stop bleeding to represent surgical hemostasis",
            ),
            target_tab="Events",
            target_section="circulation",
        ),
        ScenarioStep(
            id="GIVE_FLUIDS",
            title="Replace circulating volume",
            instruction=(
                "Start <b>500 mL crystalloid or blood</b> in Fluids, "
                "then reassess blood loss and perfusion. Use blood products early for major "
                "hemorrhage and activate the local major hemorrhage protocol when needed. "
                "Coagulation testing, component therapy, and calcium replacement are clinical actions outside this model."
            ),
            check_requirements=require_fluid_bolus(
                500,
                labels=("crystalloid", "blood"),
            ),
            target_tab="Fluids",
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Vasopressor support",
            instruction=(
                "If <b>MAP remains below 65 mmHg</b>, start or adjust norepinephrine "
                "while replacing volume. A starting rate of 0.05-0.1 mcg/kg/min is "
                "3.5-7 mcg/min for 70 kg; enter the absolute rate in the drug control and titrate to response. "
                "Continue without adding a vasopressor if pressure has recovered."
            ),
            check_requirements=require_vasopressor_if_hypotensive(
                "nore", "phenyl",
                fail_message="Start or adjust vasopressor support for persistent hypotension",
            ),
            target_tab="Medications",
        ),
        create_reassess_step(
            "hemorrhage",
            "Stop hemorrhage first",
            "Hemorrhage controlled",
            "Persistent tachycardia or hypotension warrants reassessment for continued bleeding, "
            "inadequate replacement, or another cause. In practice, also assess hemoglobin, "
            "coagulation, temperature, calcium, and tissue perfusion."
        ),
    ]

    return Scenario(
        id="hemorrhage_response",
        name="Hemorrhage response",
        description="Learn to recognize and manage intraoperative hemorrhage and hypovolemic shock.",
        steps=steps,
    )
