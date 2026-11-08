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
    for volume, label in ((500, "crystalloid"), (300, "blood")):
        met, _ = require_fluid_bolus(volume, labels=(label,), event_name="hemorrhage")(engine)
        if met:
            return True, ""
    return False, "Start 500 mL crystalloid or one 300 mL red-cell unit"


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
                "Treat brisk blood loss promptly; do not wait for hypotension. "
                "A falling MAP, narrowing pulse pressure, and falling EtCO₂ can indicate reduced flow. "
                "Anesthesia and beta blockade can blunt tachycardia. "
                "This objective accepts early volume replacement or bleeding control, or the developing pressure fall."
            ),
            check_requirements=require_hypotension_or_treatment(
                start_map, "hemorrhage", "replace volume and control bleeding", pressors=("nore", "phenyl"),
            ),
        ),
        ScenarioStep(
            id="STOP_BLEEDING",
            title="Control the bleeding source",
            instruction=(
                "Call for help and select <b>Request surgical hemostasis</b>. "
                "In practice, hemorrhage control and resuscitation proceed together."
            ),
            check_requirements=require_crisis_stopped("hemorrhage", "Request surgical hemostasis"),
        ),
        ScenarioStep(
            id="GIVE_FLUIDS",
            title="Replace circulating volume",
            instruction=(
                "Start <b>500 mL crystalloid</b> as a bridge while blood is obtained, "
                "or give a red-cell unit (<b>300 mL</b> in this simulator). "
                "Reassess blood loss and perfusion. Use blood products early for major "
                "hemorrhage and activate the local massive transfusion protocol when needed. "
                "In practice, use coagulation tests and ionized calcium to guide component therapy and calcium replacement."
            ),
            check_requirements=_require_volume_replacement,
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Start vasopressor",
            instruction=(
                "If <b>MAP remains below 65 mmHg</b>, start or adjust norepinephrine "
                "while replacing volume. A starting rate of 0.05-0.1 mcg/kg/min is "
                "3.5-7 mcg/min for 70 kg; enter the absolute rate in the drug control and titrate to response. "
                "Continue without adding a vasopressor if pressure has recovered."
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
            "Hemorrhage controlled",
            "Persistent tachycardia or hypotension warrants reassessment for continued bleeding, "
            "inadequate replacement, or another cause. In practice, also assess hemoglobin, "
            "coagulation, temperature, calcium, and tissue perfusion."
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
