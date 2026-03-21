"""
Septic Shock Response Scenario.

Teaches recognition and initial management of distributive (warm) septic shock.
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


def _require_sepsis_active() -> callable:
    """Check that sepsis event is active."""
    def check(engine) -> Tuple[bool, str]:
        if getattr(engine, "active_sepsis", False):
            return True, ""
        return False, "Start sepsis event (Events tab)"
    return check


def _require_warm_shock_recognition() -> callable:
    """Check for septic shock pattern (vasoplegia/hypotension ± tachycardia)."""
    def check(engine) -> Tuple[bool, str]:
        hr = monitor_value(engine, "hr")
        map_val = monitor_value(engine, "map")
        tachycardia = hr > 90
        hypotension = map_val < 65
        low_svr = engine.state.svr < 12

        if (hypotension and low_svr) or (tachycardia and (hypotension or low_svr)):
            return True, ""

        msgs = []
        if not tachycardia:
            msgs.append(f"HR: {hr:.0f} (watch for ↑)")
        if not hypotension and not low_svr:
            msgs.append(f"MAP: {map_val:.0f} or SVR: {engine.state.svr:.0f} (watch for ↓)")
        return False, join_messages(msgs)
    return check


def _require_vasopressor() -> callable:
    """Check that vasopressor support started."""
    return require_vasopressor_running("Start vasopressor (norepinephrine preferred)")


def _require_source_control() -> callable:
    """Check that sepsis event has been stopped (source control)."""
    def check(engine) -> Tuple[bool, str]:
        if getattr(engine, "active_sepsis", False):
            return False, "Stop sepsis (source control + antibiotics)"
        return True, ""
    return check


def _require_hemodynamic_stability() -> callable:
    """Check that MAP has been restored to a safe range."""
    def check(engine) -> Tuple[bool, str]:
        map_val = monitor_value(engine, "map")
        map_ok = map_val >= 65
        if map_ok:
            return True, ""
        return False, f"MAP: {map_val:.0f}/65+ mmHg"
    return check


def create_sepsis_response() -> Scenario:
    """Create septic shock response scenario."""
    steps = [
        ScenarioStep(
            id="OBSERVE_BASELINE",
            title="Observe baseline",
            instruction=(
                "<b>Step 1/7: Observe baseline vitals</b><br>"
                "Confirm stable baseline values before sepsis begins:<br>"
                "• HR 60–90 bpm<br>"
                "• MAP > 65 mmHg<br>"
                "• SpO₂ > 94%<br><br>"
                "<i>Baseline is essential for recognizing distributive changes.</i>"
            ),
            check_requirements=_require_simulation_running(),
        ),
        ScenarioStep(
            id="START_SEPSIS",
            title="Sepsis begins",
            instruction=(
                "<b>Step 2/7: Initiate sepsis event</b><br>"
                "Go to the <b>Events</b> tab and click <b>'Start sepsis'</b>.<br><br>"
                "<i>Sepsis can evolve rapidly from infection or intra-abdominal sources.</i>"
            ),
            check_requirements=_require_sepsis_active(),
        ),
        ScenarioStep(
            id="RECOGNIZE_WARM_SHOCK",
            title="Recognize septic shock",
            instruction=(
                "<b>Step 3/7: Recognize warm septic shock</b><br>"
                "Look for the classic pattern:<br>"
                "• <b>Tachycardia</b> (HR > 90) – may lag under anesthesia<br>"
                "• <b>Low SVR</b> and/or <b>MAP < 65</b><br>"
                "• Often normal/high CO early (“warm shock”)<br><br>"
                "<i>Vasoplegia drives hypotension despite preserved flow.</i>"
            ),
            check_requirements=_require_warm_shock_recognition(),
        ),
        ScenarioStep(
            id="GIVE_FLUIDS",
            title="Initial fluid resuscitation",
            instruction=(
                "<b>Step 4/7: Give fluids</b><br>"
                "Administer a <b>500–1000 mL</b> crystalloid bolus.<br>"
                "Use the <b>Events</b> tab -> <b>Fluid bolus</b>.<br><br>"
                "<i>Goal: improve preload and support perfusion.</i>"
            ),
            check_requirements=require_fluid_given(500),
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Start vasopressor",
            instruction=(
                "<b>Step 5/7: Start norepinephrine</b><br>"
                "If MAP remains low after fluids, start a vasopressor.<br>"
                "• <b>Norepinephrine</b> is first-line (0.05–0.1 mcg/kg/min).<br><br>"
                "<i>Pressor resistance may require higher doses.</i>"
            ),
            check_requirements=_require_vasopressor(),
        ),
        ScenarioStep(
            id="SOURCE_CONTROL",
            title="Source control",
            instruction=(
                "<b>Step 6/7: Source control</b><br>"
                "Stop the sepsis event to simulate antibiotics and source control.<br>"
                "Click <b>'Stop sepsis'</b> in the <b>Events</b> tab.<br><br>"
                "<i>Without source control, shock will persist.</i>"
            ),
            check_requirements=_require_source_control(),
        ),
        ScenarioStep(
            id="REASSESS",
            title="Reassess hemodynamics",
            instruction=(
                "<b>Step 7/7: Reassess and stabilize</b><br>"
                "Confirm stabilization:<br>"
                "• <b>MAP ≥ 65 mmHg</b><br>"
                "• Sepsis controlled<br><br>"
                "Wean vasopressors as perfusion improves.<br>"
                "<i>Monitor closely for relapse or ongoing fluid needs.</i>"
            ),
            check_requirements=_require_hemodynamic_stability(),
        ),
    ]

    return Scenario(
        id="sepsis_response",
        name="Septic shock response",
        icon="",
        description="Recognize and manage early distributive septic shock.",
        steps=steps,
    )
