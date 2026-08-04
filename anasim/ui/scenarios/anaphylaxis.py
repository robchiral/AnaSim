"""
Anaphylaxis response scenario.
"""

from typing import Tuple

from .base import (
    Scenario,
    ScenarioStep,
    bolus_given_this_step,
    require_crisis_started,
    require_crisis_stopped_with_map,
    require_fluid_given,
    require_infusion_started,
)


def _require_epinephrine_started() -> callable:
    """Check that epinephrine was bolused or started during this objective."""
    infusion_check = require_infusion_started(
        "epi", fail_message="Give epinephrine (bolus or infusion)"
    )

    def check(engine) -> Tuple[bool, str]:
        if bolus_given_this_step(engine, "epi"):
            return True, ""
        return infusion_check(engine)
    return check


def create_anaphylaxis_scenario() -> Scenario:
    """Create a guided scenario for managing intraoperative anaphylaxis."""
    steps = [
        ScenarioStep(
            id="RECOGNIZE",
            title="Recognize anaphylaxis",
            instruction=(
                "Start the event and identify sudden hypotension, tachycardia, "
                "and bronchospasm.<br><br>"
                "<i>Think distributive shock with airway involvement.</i>"
            ),
            check_requirements=require_crisis_started(
                "anaphylaxis",
                "active_anaphylaxis",
                "Start anaphylaxis in Events",
            ),
            target_tab="Events",
        ),
        ScenarioStep(
            id="EPINEPHRINE",
            title="Give epinephrine",
            instruction=(
                "Start epinephrine support promptly.<br><br>"
                "<i>Epinephrine is first-line because it treats vasoplegia, bronchospasm, and cardiovascular collapse.</i>"
            ),
            check_requirements=_require_epinephrine_started(),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="FLUIDS",
            title="Give fluids",
            instruction=(
                "Administer at least <b>500 mL</b> of fluid to support circulation.<br><br>"
                "<i>Capillary leak and vasodilation often require aggressive volume support.</i>"
            ),
            check_requirements=require_fluid_given(
                500.0,
                labels=("crystalloid", "colloid"),
                fail_prefix="Give fluids",
            ),
            target_tab="Events",
        ),
        ScenarioStep(
            id="STABILIZE",
            title="Stabilize and reassess",
            instruction=(
                "Stop the event after treatment and confirm <b>MAP > 65 mmHg</b>.<br><br>"
                "<i>Persistent hypotension after epinephrine and fluids should prompt escalation.</i>"
            ),
            check_requirements=require_crisis_stopped_with_map(
                "anaphylaxis",
                "active_anaphylaxis",
                fail_crisis="Stop anaphylaxis event",
            ),
            target_tab="Events",
        ),
    ]

    return Scenario(
        id="anaphylaxis_response",
        name="Anaphylaxis management",
        icon="",
        description="Recognize and manage intraoperative anaphylaxis.",
        steps=steps,
    )
