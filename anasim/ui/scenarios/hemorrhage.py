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
    require_fluid_given,
    require_stable_baseline_vitals,
    require_vasopressor_running,
)


def _require_simulation_running() -> callable:
    """Check simulation is running with stable vitals."""
    return require_stable_baseline_vitals()


def _require_hemorrhage_active() -> callable:
    """Check that hemorrhage event is active."""
    def check(engine) -> Tuple[bool, str]:
        if hasattr(engine, 'active_hemorrhage') and engine.active_hemorrhage:
            return True, ""
        return False, "Start hemorrhage event (Events tab)"
    return check


def _require_shock_recognition() -> callable:
    """
    Check that shock signs are present (tachycardia, hypotension).
    User should recognize these signs.
    """
    def check(engine) -> Tuple[bool, str]:
        # Look for shock signs
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


def _require_vasopressor() -> callable:
    """Check that vasopressor support started."""
    return require_vasopressor_running(
        "Start vasopressor (norepinephrine, phenylephrine, or epinephrine)"
    )


def _require_hemorrhage_stopped() -> callable:
    """Check that hemorrhage event has been stopped."""
    def check(engine) -> Tuple[bool, str]:
        if hasattr(engine, 'active_hemorrhage') and not engine.active_hemorrhage:
            return True, ""
        return False, "Stop hemorrhage (simulate surgical hemostasis)"
    return check


def _require_hemodynamic_stability() -> callable:
    """Check that hemodynamics have stabilized post-resuscitation."""
    def check(engine) -> Tuple[bool, str]:
        # Focus on MAP restoration as primary goal of resuscitation
        # HR recovery takes time even with successful treatment
        map_val = monitor_value(engine, "map")
        map_ok = map_val > 65
        hemorrhage_stopped = not getattr(engine, 'active_hemorrhage', True)
        
        if map_ok and hemorrhage_stopped:
            return True, ""
        
        msgs = []
        if not map_ok: msgs.append(f"MAP: {map_val:.0f}/65+ mmHg")
        if not hemorrhage_stopped: msgs.append("Stop hemorrhage first")
        return False, join_messages(msgs)
    return check


def create_hemorrhage_response() -> Scenario:
    """Create hemorrhage response scenario."""
    
    steps = [
        ScenarioStep(
            id="OBSERVE_BASELINE",
            title="Observe baseline",
            instruction=(
                "<b>Step 1/7: Observe baseline vitals</b><br>"
                "Before the hemorrhage begins, note the patient's baseline vitals:<br>"
                "• Heart rate: 60-80 bpm<br>"
                "• MAP: > 65 mmHg<br>"
                "• SpO₂: > 94%<br><br>"
                "<i>Recognition of abnormal values requires knowing normal baseline.</i>"
            ),
            check_requirements=_require_simulation_running(),
        ),
        ScenarioStep(
            id="START_HEMORRHAGE",
            title="Hemorrhage begins",
            instruction=(
                "<b>Step 2/7: Initiate hemorrhage event</b><br>"
                "Go to the <b>Events</b> tab and click <b>'Start bleeding'</b>.<br>"
                "Select severity (500-2000 mL/min simulates Class II-IV hemorrhage).<br><br>"
                "<i>Intraoperative hemorrhage can occur suddenly during surgery.</i>"
            ),
            check_requirements=_require_hemorrhage_active(),
        ),
        ScenarioStep(
            id="RECOGNIZE_SHOCK",
            title="Recognize hypovolemic shock",
            instruction=(
                "<b>Step 3/7: Recognize shock signs</b><br>"
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
                "<b>Step 4/7: Administer IV fluids</b><br>"
                "Give crystalloid bolus: <b>500-1000 mL</b> rapidly.<br>"
                "Use the <b>Events</b> tab -> <b>Fluid bolus</b>.<br><br>"
                "In practice: use blood products for Class III-IV hemorrhage.<br><br>"
                "<i>Goal: restore intravascular volume while awaiting surgical hemostasis.</i>"
            ),
                check_requirements=require_fluid_given(500),
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Vasopressor support",
            instruction=(
                "<b>Step 5/7: Start vasopressor</b><br>"
                "If MAP remains low despite fluids, start vasopressor:<br>"
                "• <b>Norepinephrine</b>: 0.05-0.1 mcg/kg/min<br>"
                "• <b>Phenylephrine</b>: 50-100 mcg/min<br><br>"
                "<i>Vasopressors bridge until volume is restored; not a substitute for blood.</i>"
            ),
            check_requirements=_require_vasopressor(),
        ),
        ScenarioStep(
            id="STOP_BLEEDING",
            title="Surgical hemostasis",
            instruction=(
                "<b>Step 6/7: Achieve hemostasis</b><br>"
                "Click <b>'Stop bleeding'</b> to simulate surgical control.<br><br>"
                "In reality, this requires:<br>"
                "• Communication with surgical team<br>"
                "• Pressure, cautery, sutures, clips<br>"
                "• Possible interventional radiology<br><br>"
                "<i>Definitive hemorrhage control is the priority over resuscitation.</i>"
            ),
            check_requirements=_require_hemorrhage_stopped(),
        ),
        ScenarioStep(
            id="REASSESS",
            title="Reassess hemodynamics",
            instruction=(
                "<b>Step 7/7: Reassess and stabilize</b><br>"
                "Confirm hemodynamic improvement:<br>"
                "• <b>MAP > 65 mmHg</b> (primary resuscitation goal)<br>"
                "• Hemorrhage controlled<br><br>"
                "HR may remain elevated initially - this is normal after volume loss.<br><br>"
                "<i>Post-hemorrhage: watch for coagulopathy, acidosis, hypothermia.</i>"
            ),
            check_requirements=_require_hemodynamic_stability(),
        ),
    ]
    
    return Scenario(
        id="hemorrhage_response",
        name="Hemorrhage response",
        icon="",
        description="Learn to recognize and manage intraoperative hemorrhage and hypovolemic shock.",
        steps=steps,
    )
