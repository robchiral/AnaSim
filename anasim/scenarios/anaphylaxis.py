"""Anaphylaxis response scenario."""

from .base import (
    CRISIS_BRIEFING,
    CRISIS_ONSET_S,
    CaseAction,
    CaseEvent,
    CaseOutcome,
    Requirement,
    Scenario,
    ScenarioStep,
    bolus_given_in_case,
    create_observe_step,
    crisis_outcomes,
    require_all,
    require_crisis_resolved_with_map,
    require_fluid_bolus,
    require_infusion_started,
    require_oxygen_flow,
    require_tracheal_ventilation,
)


def _prepare(engine) -> None:
    # Brown 2004 gave 1000 mL saline over 1-3 min.
    engine.fluid_infusion_rate_ml_min = 500.0


def _start(engine) -> None:
    engine.start_anaphylaxis()
    engine.anaphylaxis.severity = 1.0


def _require_bronchospasm_improved(engine) -> tuple[bool, str]:
    improved = engine.state.bronchospasm < 0.5
    return improved, "" if improved else "Continue treating bronchospasm and reassess airway pressure"


def _require_epinephrine_started() -> Requirement:
    """Check that epinephrine was bolused or started after anaphylaxis began."""
    infusion_check = require_infusion_started(
        "epi", fail_message="Give epinephrine (bolus or infusion)", event_name="anaphylaxis",
    )

    def check(engine) -> tuple[bool, str]:
        if bolus_given_in_case(engine, "epi", event_name="anaphylaxis"):
            return True, ""
        return infusion_check(engine)
    return check


def create_anaphylaxis_scenario() -> Scenario:
    epinephrine = _require_epinephrine_started()
    fluids = require_fluid_bolus(
        1000.0,
        labels=("crystalloid",),
        fail_prefix="Start crystalloid resuscitation",
        event_name="anaphylaxis",
    )
    steps = [
        create_observe_step("anaphylaxis"),
        ScenarioStep(
            id="EPINEPHRINE",
            title="Give epinephrine",
            instruction=(
                "Abrupt hypotension and bronchospasm with rising airway pressure suggest anaphylaxis; "
                "HR and skin signs vary. Call for help and pause the procedure.<br><br>"
                "Give <b>epinephrine 50-100 mcg IV</b> promptly for this severe reaction "
                "with IV access and continuous monitoring. Repeat and titrate to response; "
                "start an infusion if repeated boluses are needed (for example, "
                "0.05 mcg/kg/min = 3.5 mcg/min for 70 kg).<br><br>"
                "Give <b>100% oxygen</b>: set O₂ 10 L/min, air and N₂O 0 L/min. "
                "Maintain ventilation and assess the airway while treating circulation. "
                "Select <b>Stop suspected trigger</b> promptly, alongside resuscitation. "
                "If hypotensive, turn off the volatile anesthetic and vasodilating infusions, "
                "and consider an amnestic agent such as midazolam."
            ),
            check_requirements=require_all(
                epinephrine,
                require_oxygen_flow(),
                require_tracheal_ventilation,
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="FLUIDS",
            title="Give fluids",
            instruction=(
                "Start a rapid <b>1000 mL crystalloid bolus</b> for this severe adult "
                "reaction, alongside epinephrine. Reassess and repeat as needed; "
                "vasodilation and capillary leak can require substantial volume."
            ),
            check_requirements=fluids,
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="STABILIZE",
            title="Stabilize and reassess",
            instruction=(
                "Select <b>Stop suspected trigger</b> to model the reaction subsiding after "
                "trigger removal and treatment. Continue support and confirm <b>MAP ≥ 65 mmHg</b>; "
                "reassess oxygenation, capnography, and airway pressure. "
                "Persistent hypotension or bronchospasm requires further epinephrine, fluids, and escalation. "
                "In practice, arrange tryptase sampling and allergy follow-up."
            ),
            check_requirements=require_all(
                require_crisis_resolved_with_map("anaphylaxis", fail_crisis="Stop the suspected trigger"),
                _require_bronchospasm_improved,
            ),
        ),
    ]

    return Scenario(
        id="anaphylaxis_response",
        name="Anaphylaxis management",
        briefing=CRISIS_BRIEFING,
        setup_engine=_prepare,
        steps=steps,
        events=(CaseEvent(
            "anaphylaxis", CRISIS_ONSET_S,
            "Shortly after an antibiotic dose, blood pressure falls and airway pressure rises.",
            _start,
        ),),
        actions=(CaseAction(
            "stop_trigger", "Stop suspected trigger",
            lambda engine: engine.stop_anaphylaxis(), lambda engine: engine.anaphylaxis.active,
            completed_label="Suspected trigger stopped",
        ),),
        outcomes=(
            CaseOutcome("epinephrine", "Epinephrine started", lambda engine: epinephrine(engine)[0],
                        "Give epinephrine promptly for severe perioperative anaphylaxis.", milestone=True),
            CaseOutcome("bronchospasm", "Bronchospasm improved",
                        lambda engine: _require_bronchospasm_improved(engine)[0],
                        "Reassess airway pressure and ventilation after treating bronchospasm."),
            *crisis_outcomes("anaphylaxis", fluids),
        ),
    )
