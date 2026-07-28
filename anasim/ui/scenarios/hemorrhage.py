"""
Hemorrhage Response Scenario.

Teaches recognition and management of intraoperative hemorrhage/hypovolemic shock.
"""

from typing import Tuple
from .base import (
    Scenario,
    ScenarioStep,
    join_messages,
    monitor_value,
    require_crisis_active,
    require_crisis_resolved_with_map,
    require_crisis_stopped,
    require_fluid_given,
    require_vasopressor_running,
    create_observe_baseline_step,
    create_reassess_step,
)


def _require_shock_recognition() -> callable:
    """
    Check that shock signs are present (tachycardia, hypotension).
    User should recognize these signs.
    """
    def check(engine) -> Tuple[bool, str]:
        hr = monitor_value(engine, "hr")
        map_val = monitor_value(engine, "map")
        tachycardia = hr > 100
        hypotension = map_val < 65

        if tachycardia and hypotension:
            return True, ""

        msgs = []
        if not tachycardia: msgs.append(f"HR: {hr:.0f} (watch for ↑)")
        if not hypotension: msgs.append(f"MAP: {map_val:.0f} (watch for ↓)")
        return False, join_messages(msgs)
    return check


def create_hemorrhage_response() -> Scenario:
    """Create hemorrhage response scenario."""
    
    steps = [
        create_observe_baseline_step("hemorrhage"),
        ScenarioStep(
            id="START_HEMORRHAGE",
            title="Hemorrhage begins",
            instruction=(
                "Select a severity, then choose <b>Start bleeding</b> on the Events tab.<br><br>"
                "<i>Intraoperative hemorrhage can occur suddenly during surgery.</i>"
            ),
            check_requirements=require_crisis_active("active_hemorrhage", "Start hemorrhage event (Events tab)"),
            target_tab="Events",
        ),
        ScenarioStep(
            id="RECOGNIZE_SHOCK",
            title="Recognize hypovolemic shock",
            instruction=(
                "Observe the developing shock state:<br>"
                "• <b>Tachycardia</b> (HR > 100) - compensatory response<br>"
                "• <b>Hypotension</b> (MAP < 65) - volume depletion<br>"
                "• Narrowed pulse pressure (SBP-DBP)<br><br>"
                "<i>ATLS Class III hemorrhage (30-40% loss): tachycardia, hypotension, confusion.</i>"
            ),
            check_requirements=_require_shock_recognition(),
        ),
        ScenarioStep(
            id="GIVE_FLUIDS",
            title="Fluid resuscitation",
            instruction=(
                "Give <b>500-1000 mL</b> crystalloid from the Events tab. "
                "Use blood products for major ongoing hemorrhage.<br><br>"
                "<i>Goal: restore intravascular volume while awaiting surgical hemostasis.</i>"
            ),
            check_requirements=require_fluid_given(500),
            target_tab="Events",
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Vasopressor support",
            instruction=(
                "If MAP remains low despite fluids, start vasopressor:<br>"
                "• <b>Norepinephrine</b>: 0.05-0.1 mcg/kg/min<br>"
                "• <b>Phenylephrine</b>: 50-100 mcg/min<br><br>"
                "<i>Vasopressors bridge until volume is restored; not a substitute for blood.</i>"
            ),
            check_requirements=require_vasopressor_running("Start vasopressor (norepinephrine, phenylephrine, or epinephrine)"),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="STOP_BLEEDING",
            title="Surgical hemostasis",
            instruction=(
                "Select <b>Stop bleeding</b> to simulate definitive surgical control.<br><br>"
                "<i>Definitive hemorrhage control is the priority over resuscitation.</i>"
            ),
            check_requirements=require_crisis_stopped("active_hemorrhage", "Stop hemorrhage (simulate surgical hemostasis)"),
            target_tab="Events",
        ),
        create_reassess_step(
            "active_hemorrhage",
            "Stop hemorrhage first",
            "Hemorrhage controlled",
            "HR may remain elevated initially - this is normal after volume loss.<br><br>"
            "<i>Post-hemorrhage: watch for coagulopathy, acidosis, hypothermia.</i>"
        ),
    ]
    
    return Scenario(
        id="hemorrhage_response",
        name="Hemorrhage response",
        icon="",
        description="Learn to recognize and manage intraoperative hemorrhage and hypovolemic shock.",
        steps=steps,
    )
