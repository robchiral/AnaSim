"""Intraoperative hemorrhage and hypovolemic shock scenario."""

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


def _require_volume_replacement(engine) -> tuple[bool, str]:
    """Accept an initial crystalloid bolus or one modeled red-cell unit."""
    for volume, labels in ((500, ("crystalloid", "saline")), (300, ("blood",))):
        met, _ = require_fluid_bolus(volume, labels=labels, event_name="hemorrhage")(engine)
        if met:
            return True, ""
    return False, "Give 500 mL crystalloid or 1 unit PRBC (300 mL)"


def create_hemorrhage_response() -> Scenario:
    start_map = CrisisStartMAP()

    def start(engine) -> None:
        start_map.record(engine)
        engine.start_hemorrhage(800)

    steps = [
        create_observe_step("hemorrhage"),
        ScenarioStep(
            id="RECOGNIZE_SHOCK",
            title="Recognize hypovolemic shock",
            instruction=(
                "Treat brisk blood loss early without waiting for hypotension. "
                "Falling MAP, narrowing pulse pressure, and dropping EtCO₂ signal low cardiac output; "
                "anesthesia and beta blockers can blunt tachycardia.<br><br>"
                "<i>Begin volume replacement or hemostasis right away, or watch pressure fall.</i>"
            ),
            check_requirements=require_hypotension_or_treatment(
                start_map, "hemorrhage", "replace volume and control bleeding", pressors=("nore", "phenyl"),
            ),
        ),
        ScenarioStep(
            id="STOP_BLEEDING",
            title="Control the bleeding source",
            instruction=(
                "Call for help and select <b>Request surgical hemostasis</b>.<br><br>"
                "<i>Control bleeding and resuscitate in parallel.</i>"
            ),
            check_requirements=require_crisis_stopped("hemorrhage", "Request surgical hemostasis"),
        ),
        ScenarioStep(
            id="GIVE_FLUIDS",
            title="Replace circulating volume",
            instruction=(
                "Give <b>500 mL crystalloid</b> as a bridge or start <b>PRBC (300 mL)</b>. "
                "Use blood products early in major hemorrhage and activate the massive transfusion protocol when needed.<br><br>"
                "<i>Guide component therapy and calcium replacement with coagulation tests and ionized calcium.</i>"
            ),
            check_requirements=_require_volume_replacement,
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Start vasopressor",
            instruction=(
                "If <b>MAP < 65 mmHg</b>, start or adjust <b>norepinephrine</b> in mcg/min while replacing volume "
                "(0.05-0.1 mcg/kg/min = 3.5-7 mcg/min for 70 kg) and titrate to response. "
                "Skip this if pressure has already recovered."
            ),
            check_requirements=require_vasopressor_if_hypotensive(
                "nore", "phenyl",
                fail_message="Start or adjust vasopressor support for persistent hypotension",
                event_name="hemorrhage",
            ),
            target_tab="Medications",
        ),
        create_reassess_step(
            "hemorrhage",
            "Request surgical hemostasis",
            "hemorrhage is controlled",
            "<i>If tachycardia or hypotension persists, look for ongoing bleeding, inadequate volume replacement, "
            "or another cause. Follow hemoglobin, coagulation, temperature, calcium, and perfusion.</i>",
        ),
    ]

    return Scenario(
        id="hemorrhage_response",
        name="Hemorrhage response",
        briefing=CRISIS_BRIEFING,
        steps=steps,
        events=(CaseEvent("hemorrhage", CRISIS_ONSET_S, "The surgeon reports brisk blood loss.", start),),
        actions=(CaseAction(
            "hemostasis", "Request surgical hemostasis",
            lambda engine: engine.stop_hemorrhage(), lambda engine: engine.active_hemorrhage,
            completed_label="Surgical hemostasis requested",
        ),),
        outcomes=crisis_outcomes("hemorrhage", _require_volume_replacement),
    )
