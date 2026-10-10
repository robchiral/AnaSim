"""Anaphylaxis response scenario."""

from .base import (
    Requirement,
    Scenario,
    ScenarioStep,
    bolus_given_this_step,
    require_all,
    require_crisis_started,
    require_crisis_stopped,
    require_fluid_bolus,
    require_infusion_started,
    require_map_at_least,
    require_oxygen_flow,
    require_oxygenation,
    require_tracheal_ventilation,
)


def _require_epinephrine_started() -> Requirement:
    """Check that epinephrine was bolused or started during this objective."""
    infusion_check = require_infusion_started(
        "epi", fail_message="Give epinephrine (bolus or infusion)"
    )

    def check(engine) -> tuple[bool, str]:
        if bolus_given_this_step(engine, "epi"):
            return True, ""
        return infusion_check(engine)
    return check


def create_anaphylaxis_scenario() -> Scenario:
    steps = [
        ScenarioStep(
            id="RECOGNIZE",
            title="Recognize anaphylaxis",
            instruction=(
                "Select <b>Start anaphylaxis</b> to model a severe perioperative reaction. "
                "Look for abrupt hypotension and bronchospasm with rising airway pressure; "
                "HR and skin signs vary. Call for help, stop suspected triggers, "
                "and pause the procedure while treatment begins."
            ),
            check_requirements=require_crisis_started(
                "anaphylaxis",
                "Start anaphylaxis in Events",
            ),
            target_tab="Events",
        ),
        ScenarioStep(
            id="EPINEPHRINE",
            title="Give epinephrine",
            instruction=(
                "Give <b>epinephrine 50-100 mcg IV</b> promptly for this severe reaction "
                "with IV access and continuous monitoring. Repeat and titrate to response; "
                "start an infusion if repeated boluses are needed (for example, "
                "0.05 mcg/kg/min = 3.5 mcg/min for 70 kg).<br><br>"
                "Give <b>100% oxygen</b>: set O₂ 10 L/min, air and N₂O 0 L/min. "
                "Maintain ventilation and assess the airway while treating circulation."
            ),
            check_requirements=require_all(
                _require_epinephrine_started(),
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
            check_requirements=require_fluid_bolus(
                1000.0,
                labels=("crystalloid",),
                fail_prefix="Start crystalloid resuscitation",
            ),
            target_tab="Events",
        ),
        ScenarioStep(
            id="STABILIZE",
            title="Stabilize and reassess",
            instruction=(
                "Select <b>Stop anaphylaxis</b> to model the reaction subsiding after "
                "trigger removal and treatment. Continue support and confirm <b>MAP ≥ 65 mmHg</b>; "
                "reassess oxygenation, capnography, and airway pressure. "
                "Persistent hypotension or bronchospasm requires further epinephrine, fluids, and escalation. "
                "In practice, arrange tryptase sampling and allergy follow-up."
            ),
            check_requirements=require_all(
                require_crisis_stopped(
                    "anaphylaxis",
                    "Stop anaphylaxis event",
                ),
                require_map_at_least(),
                require_oxygenation(),
                require_tracheal_ventilation,
            ),
            target_tab="Events",
        ),
    ]

    return Scenario(
        id="anaphylaxis_response",
        name="Anaphylaxis management",
        description="Recognize and manage intraoperative anaphylaxis.",
        steps=steps,
    )
