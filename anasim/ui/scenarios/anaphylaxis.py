"""
Anaphylaxis response scenario.
"""

from typing import Tuple

from .base import (
    Scenario,
    ScenarioStep,
    cumulative_fluid_given,
    join_messages,
    monitor_value,
)


def _require_anaphylaxis_active() -> callable:
    """Check that the crisis event has started."""
    def check(engine) -> Tuple[bool, str]:
        if getattr(engine, "active_anaphylaxis", False):
            return True, ""
        return False, "Start anaphylaxis in Events"
    return check


def _require_epinephrine_started() -> callable:
    """Check that epinephrine support has been started."""
    def check(engine) -> Tuple[bool, str]:
        epi_running = engine.epi_rate_ug_sec > 0 or (engine.tci_epi and engine.tci_epi.target > 0)
        if epi_running:
            return True, ""
        return False, "Give epinephrine"
    return check


def _require_fluids_given(min_ml: float = 500.0) -> callable:
    """Check that at least a modest fluid bolus has been given."""
    def check(engine) -> Tuple[bool, str]:
        total = cumulative_fluid_given(engine) + getattr(getattr(engine, "hemo", None), "total_blood_in_ml", 0.0)
        if total >= min_ml:
            return True, ""
        return False, f"Give fluids ({total:.0f}/{min_ml:.0f} mL)"
    return check


def _require_stability() -> callable:
    """Check that MAP has recovered and the crisis is no longer active."""
    def check(engine) -> Tuple[bool, str]:
        map_val = monitor_value(engine, "map")
        crisis_resolved = not getattr(engine, "active_anaphylaxis", False)
        if map_val > 65 and crisis_resolved:
            return True, ""
        msgs = []
        if map_val <= 65:
            msgs.append(f"MAP: {map_val:.0f}/65+")
        if not crisis_resolved:
            msgs.append("Stop anaphylaxis event")
        return False, join_messages(msgs)
    return check


def create_anaphylaxis_scenario() -> Scenario:
    """Create a guided scenario for managing intraoperative anaphylaxis."""
    steps = [
        ScenarioStep(
            id="RECOGNIZE",
            title="Recognize anaphylaxis",
            instruction=(
                "<b>Step 1/4: Recognize anaphylaxis</b><br>"
                "Start the event and identify the pattern of sudden hypotension, tachycardia, and bronchospasm.<br><br>"
                "<i>Think distributive shock with airway involvement.</i>"
            ),
            check_requirements=_require_anaphylaxis_active(),
        ),
        ScenarioStep(
            id="EPINEPHRINE",
            title="Give epinephrine",
            instruction=(
                "<b>Step 2/4: Treat with epinephrine</b><br>"
                "Start epinephrine support promptly.<br><br>"
                "<i>Epinephrine is first-line because it treats vasoplegia, bronchospasm, and cardiovascular collapse.</i>"
            ),
            check_requirements=_require_epinephrine_started(),
        ),
        ScenarioStep(
            id="FLUIDS",
            title="Give fluids",
            instruction=(
                "<b>Step 3/4: Give IV fluids</b><br>"
                "Administer at least <b>500 mL</b> of fluid to support circulation.<br><br>"
                "<i>Capillary leak and vasodilation often require aggressive volume support.</i>"
            ),
            check_requirements=_require_fluids_given(500.0),
        ),
        ScenarioStep(
            id="STABILIZE",
            title="Stabilize and reassess",
            instruction=(
                "<b>Step 4/4: Reassess hemodynamics</b><br>"
                "Stop the event after treatment and confirm <b>MAP > 65 mmHg</b>.<br><br>"
                "<i>Ongoing hypotension after epinephrine and fluids should prompt escalation.</i>"
            ),
            check_requirements=_require_stability(),
        ),
    ]

    return Scenario(
        id="anaphylaxis_response",
        name="Anaphylaxis management",
        icon="",
        description="Recognize and manage intraoperative anaphylaxis.",
        steps=steps,
    )
