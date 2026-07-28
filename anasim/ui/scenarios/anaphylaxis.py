"""
Anaphylaxis response scenario.
"""

from typing import Tuple

from .base import (
    Scenario,
    ScenarioStep,
    cumulative_fluid_given,
    monitor_value,
    require_crisis_active,
    require_crisis_resolved_with_map,
)


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
        total = cumulative_fluid_given(engine) + engine.hemo.total_blood_in_ml
        if total >= min_ml:
            return True, ""
        return False, f"Give fluids ({total:.0f}/{min_ml:.0f} mL)"
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
            check_requirements=require_crisis_active("active_anaphylaxis", "Start anaphylaxis in Events"),
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
            check_requirements=_require_fluids_given(500.0),
            target_tab="Events",
        ),
        ScenarioStep(
            id="STABILIZE",
            title="Stabilize and reassess",
            instruction=(
                "Stop the event after treatment and confirm <b>MAP > 65 mmHg</b>.<br><br>"
                "<i>Persistent hypotension after epinephrine and fluids should prompt escalation.</i>"
            ),
            check_requirements=require_crisis_resolved_with_map("active_anaphylaxis", fail_crisis="Stop anaphylaxis event"),
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
