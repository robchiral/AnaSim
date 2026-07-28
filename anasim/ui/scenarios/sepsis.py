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
    require_crisis_active,
    require_crisis_resolved_with_map,
    require_crisis_stopped,
    require_fluid_given,
    require_vasopressor_running,
    create_observe_baseline_step,
    create_reassess_step,
)


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


def create_sepsis_response() -> Scenario:
    """Create septic shock response scenario."""
    steps = [
        create_observe_baseline_step("sepsis", hr_range="60-90 bpm"),
        ScenarioStep(
            id="START_SEPSIS",
            title="Sepsis begins",
            instruction=(
                "Select <b>Start sepsis</b> on the Events tab.<br><br>"
                "<i>Sepsis can evolve rapidly from infection or intra-abdominal sources.</i>"
            ),
            check_requirements=require_crisis_active("active_sepsis", "Start sepsis event (Events tab)"),
            target_tab="Events",
        ),
        ScenarioStep(
            id="RECOGNIZE_WARM_SHOCK",
            title="Recognize septic shock",
            instruction=(
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
                "Administer a <b>500–1000 mL</b> crystalloid bolus.<br>"
                "Use the fluid controls on the Events tab.<br><br>"
                "<i>Goal: improve preload and support perfusion.</i>"
            ),
            check_requirements=require_fluid_given(500),
            target_tab="Events",
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Start vasopressor",
            instruction=(
                "If MAP remains low after fluids, start a vasopressor.<br>"
                "• <b>Norepinephrine</b> is first-line (0.05–0.1 mcg/kg/min).<br><br>"
                "<i>Pressor resistance may require higher doses.</i>"
            ),
            check_requirements=require_vasopressor_running("Start vasopressor (norepinephrine preferred)"),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="SOURCE_CONTROL",
            title="Source control",
            instruction=(
                "Stop the sepsis event to simulate antibiotics and source control.<br>"
                "Select <b>Stop sepsis</b> on the Events tab.<br><br>"
                "<i>Without source control, shock will persist.</i>"
            ),
            check_requirements=require_crisis_stopped("active_sepsis", "Stop sepsis (source control + antibiotics)"),
            target_tab="Events",
        ),
        create_reassess_step(
            "active_sepsis",
            "Stop sepsis (source control + antibiotics)",
            "Sepsis controlled",
            "Wean vasopressors as perfusion improves.<br>"
            "<i>Monitor closely for relapse or ongoing fluid needs.</i>"
        ),
    ]

    return Scenario(
        id="sepsis_response",
        name="Septic shock response",
        icon="",
        description="Recognize and manage early distributive septic shock.",
        steps=steps,
    )
