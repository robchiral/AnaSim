"""Intraoperative hemorrhage and hypovolemic shock scenario."""

from anasim.core.action_log import ACTION_FLUID

from .base import (
    CRISIS_BRIEFING,
    CaseAction,
    CaseEvent,
    CrisisStartMAP,
    Scenario,
    ScenarioStep,
    create_observe_step,
    create_reassess_step,
    crisis_outcomes,
    format_unmet_int,
    preroll_scenario,
    require_crisis_stopped,
    require_hypotension_or_treatment,
    require_vasopressor_if_hypotensive,
)

_OCCULT_LOSS_FRACTION = 0.14
_PREROLL_S = 210.0
_BLEED_ONSET_S = 20.0
_BLEED_RATE_ML_MIN = 450.0


def _prepare_hemorrhage(engine) -> None:
    """Seed moderate occult surgical blood loss (~14% BV) so pulse pressure is narrowing at t = 0."""
    rate_ml_min = _OCCULT_LOSS_FRACTION * engine.hemo.blood_volume_0 / (_PREROLL_S / 60.0)
    engine.start_hemorrhage(rate_ml_min)
    preroll_scenario(
        engine, _PREROLL_S, dt=2.0, settle_s=15.0, sample_interval_s=5.0, preserve_blood_out=True,
    )
    engine.stop_hemorrhage()


def _require_volume_replacement(engine) -> tuple[bool, str]:
    """Require 1000 mL crystalloid, 2 units PRBC (600 mL), or an equivalent combination."""
    records = engine.actions.since_event(
        "hemorrhage", ACTION_FLUID, labels=("crystalloid", "saline", "colloid", "blood"),
    )
    cryst_ml = sum(r.amount for r in records if r.label in ("crystalloid", "saline"))
    colloid_ml = sum(r.amount for r in records if r.label == "colloid")
    blood_ml = sum(r.amount for r in records if r.label == "blood")
    progress = cryst_ml / 1000.0 + colloid_ml / 500.0 + blood_ml / 600.0
    if progress >= 1.0 - 1e-9:
        return True, ""
    if cryst_ml > 0.0 or colloid_ml > 0.0 or blood_ml > 0.0:
        pct = format_unmet_int(progress * 100.0, low=100.0)
        return False, f"Continue volume replacement ({pct}% of 1000 mL crystalloid or 2 U PRBC)"
    return False, "Give 1000 mL crystalloid or 2 units PRBC (600 mL)"


def create_hemorrhage_response() -> Scenario:
    start_map = CrisisStartMAP()

    def start(engine) -> None:
        start_map.record(engine)
        engine.start_hemorrhage(_BLEED_RATE_ML_MIN)

    steps = [
        create_observe_step("hemorrhage"),
        ScenarioStep(
            id="RECOGNIZE_SHOCK",
            title="Recognize hypovolemic shock",
            instruction=(
                "Treat brisk blood loss early without waiting for severe hypotension. "
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
                "Give <b>1000 mL crystalloid</b> as a bridge or start <b>PRBC (2 units = 600 mL)</b>, "
                "or combine crystalloid and blood products. Use blood early in major hemorrhage and "
                "activate the massive transfusion protocol when needed.<br><br>"
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
        setup_engine=_prepare_hemorrhage,
        briefing=f"{CRISIS_BRIEFING} Moderate surgical blood loss has already accumulated.",
        steps=steps,
        events=(CaseEvent("hemorrhage", _BLEED_ONSET_S, "The surgeon reports brisk blood loss.", start),),
        actions=(CaseAction(
            "hemostasis", "Request surgical hemostasis",
            lambda engine: engine.stop_hemorrhage(), lambda engine: engine.active_hemorrhage,
            completed_label="Surgical hemostasis requested",
        ),),
        outcomes=crisis_outcomes("hemorrhage", _require_volume_replacement),
    )
