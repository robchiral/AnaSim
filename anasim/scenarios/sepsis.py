"""Warm septic shock scenario."""

from .base import (
    Scenario,
    ScenarioStep,
    create_observe_baseline_step,
    create_reassess_step,
    monitor_value,
    require_crisis_started,
    require_crisis_stopped,
    require_fluid_bolus,
    require_vasopressor_if_hypotensive,
)


def _require_warm_shock_recognition(engine) -> tuple[bool, str]:
    """Recognize hypotension in the infection scenario using the visible monitor."""
    map_val = monitor_value(engine, "map")
    met = map_val < 65
    return met, "" if met else f"MAP: {map_val:.0f} mmHg; observe the developing hypotension"


def create_sepsis_response() -> Scenario:
    steps = [
        create_observe_baseline_step("sepsis"),
        ScenarioStep(
            id="START_SEPSIS",
            title="Sepsis begins",
            instruction=(
                "This patient has an intra-abdominal infection. Select <b>Start sepsis</b> "
                "in Events and fluids to model developing vasodilatory shock. "
                "In practice, call for help, give prompt antibiotics, and arrange source control while resuscitating."
            ),
            check_requirements=require_crisis_started(
                "sepsis",
                "active_sepsis",
                "Start sepsis event (Events tab)",
            ),
            target_tab="Events",
        ),
        ScenarioStep(
            id="RECOGNIZE_WARM_SHOCK",
            title="Recognize septic shock",
            instruction=(
                "Recognize <b>MAP < 65 mmHg</b> in the setting of infection. "
                "Vasodilation is the main mechanism in this case; HR may rise, but tachycardia is variable. "
                "In practice, assess perfusion and consider blood loss, anesthetic effects, and other causes of hypotension."
            ),
            check_requirements=_require_warm_shock_recognition,
        ),
        ScenarioStep(
            id="GIVE_FLUIDS",
            title="Initial fluid resuscitation",
            instruction=(
                "Start a <b>500 mL crystalloid bolus</b> in Events and fluids and reassess "
                "the response. Balanced crystalloid is preferred in practice; the simulator uses one crystalloid model. "
                "An initial 30 mL/kg over the first 3 hours is a guideline starting point, "
                "with further fluid guided by perfusion and response."
            ),
            check_requirements=require_fluid_bolus(
                500,
                labels=("crystalloid",),
            ),
            target_tab="Events",
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Start vasopressor",
            instruction=(
                "If <b>MAP remains below 65 mmHg</b>, start or adjust norepinephrine. "
                "A starting rate of 0.05-0.1 mcg/kg/min is 3.5-7 mcg/min for 70 kg; "
                "enter the absolute rate and titrate to pressure and perfusion. "
                "Start vasopressors alongside fluids in severe hypotension. If pressure has recovered, continue without adding one."
            ),
            check_requirements=require_vasopressor_if_hypotensive(
                "nore", fail_message="Start or adjust norepinephrine for persistent hypotension",
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="SOURCE_CONTROL",
            title="Source control",
            instruction=(
                "Select <b>Stop sepsis</b> to let the simulated inflammatory effects subside. "
                "Continue hemodynamic support as the patient recovers. "
                "This control represents recovery after antibiotics and source control; real shock can persist after both."
            ),
            check_requirements=require_crisis_stopped(
                "sepsis",
                "active_sepsis",
                "Select Stop sepsis to model recovery after infection treatment",
            ),
            target_tab="Events",
        ),
        create_reassess_step(
            "active_sepsis",
            "Select Stop sepsis to model recovery after infection treatment",
            "Infection treatment underway",
            "Reassess the response before giving more fluid and wean vasopressors as perfusion improves. "
            "Lactate, urine output, examination, and bedside ultrasound inform reassessment in practice."
        ),
    ]

    return Scenario(
        id="sepsis_response",
        name="Septic shock response",
        description="Recognize and manage early distributive septic shock.",
        steps=steps,
    )
