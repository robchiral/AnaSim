"""
Emergence scenario definitions (Balanced and TIVA variants).
"""

from typing import Tuple

from anasim.core.action_log import ACTION_AIRWAY, ACTION_FGF, ACTION_VAPORIZER

from .base import (
    Scenario,
    ScenarioStep,
    action_taken_this_step,
    join_messages,
    monitor_value,
    require_infusions_stopped,
)


def _require_assess() -> callable:
    """Check stable maintenance conditions."""
    def check(engine) -> Tuple[bool, str]:
        bis = monitor_value(engine, "bis")
        map_val = monitor_value(engine, "map")
        bis_ok = 35 < bis < 65
        map_ok = map_val > 60
        if bis_ok and map_ok:
            return True, ""
        msgs = []
        if not bis_ok:
            msgs.append(f"BIS: {bis:.0f}")
        if not map_ok:
            msgs.append(f"MAP: {map_val:.0f}")
        return False, join_messages(msgs)
    return check


def _require_agents_stopped_balanced() -> callable:
    """Check volatile agent stopped and high flow gas."""
    def check(engine) -> Tuple[bool, str]:
        gas_off = not engine.circuit.vaporizer_on or engine.circuit.vaporizer_setting < 0.1
        high_flow = engine.circuit.fgf_total() > 6.0
        vaporizer_records = engine.actions.since_step(ACTION_VAPORIZER)
        vaporizer_stopped = any(record.amount < 0.1 for record in vaporizer_records)
        flow_changed = action_taken_this_step(engine, ACTION_FGF)
        if gas_off and high_flow and vaporizer_stopped and flow_changed:
            return True, ""
        msgs = []
        if not gas_off or not vaporizer_stopped:
            msgs.append("Turn vaporizer off for this objective")
        if not high_flow or not flow_changed:
            msgs.append("Set FGF above 6 L/min for this objective")
        return False, join_messages(msgs)
    return check


def _require_agents_stopped_tiva() -> callable:
    """Check TIVA infusions stopped."""
    return require_infusions_stopped(
        "propofol",
        "remi",
        fail_message="Stop propofol and remifentanil for this objective",
    )


def _require_awakening() -> callable:
    """Check patient emerging (BIS > 70, spontaneous breathing)."""
    def check(engine) -> Tuple[bool, str]:
        bis = monitor_value(engine, "bis")
        bis_ok = bis > 70
        rr_ok = engine.state.rr > 6
        if bis_ok and rr_ok:
            return True, ""
        msgs = []
        if not bis_ok:
            msgs.append(f"BIS: {bis:.0f}/70+")
        if not rr_ok:
            msgs.append(f"RR: {engine.state.rr:.0f}/6+")
        return False, join_messages(msgs)
    return check


def _require_extubation_criteria() -> callable:
    """Check extubation criteria met."""
    from anasim.core.state import AirwayType
    def check(engine) -> Tuple[bool, str]:
        breathing = engine.state.rr > 8 and not engine.state.apnea
        bis = monitor_value(engine, "bis")
        awake = bis > 80
        extubated = engine.state.airway_mode != AirwayType.ETT
        airway_action = action_taken_this_step(
            engine,
            ACTION_AIRWAY,
            AirwayType.MASK.value,
            AirwayType.NONE.value,
        )
        if breathing and awake and extubated and airway_action:
            return True, ""
        msgs = []
        if not awake:
            msgs.append(f"BIS: {bis:.0f}/80+")
        if not breathing:
            msgs.append(f"RR: {engine.state.rr:.0f}/8+")
        if not extubated or not airway_action:
            msgs.append("Select Mask or No airway for this objective")
        return False, join_messages(msgs)
    return check


def _require_recovery() -> callable:
    """Check recovery room criteria."""
    def check(engine) -> Tuple[bool, str]:
        spo2 = monitor_value(engine, "spo2")
        spo2_ok = spo2 > 95
        rr_ok = engine.state.rr > 10
        not_apneic = not engine.state.apnea
        if spo2_ok and rr_ok and not_apneic:
            return True, ""
        msgs = []
        if not spo2_ok:
            msgs.append(f"SpO₂: {spo2:.0f}/95+")
        if not rr_ok:
            msgs.append(f"RR: {engine.state.rr:.0f}/10+")
        return False, join_messages(msgs)
    return check


def create_emergence(maint_type: str = "balanced") -> Scenario:
    """
    Create emergence scenario.
    
    Args:
        maint_type: "balanced" or "tiva"
    """
    is_balanced = "balanced" in maint_type.lower()
    
    if is_balanced:
        stop_agents_instruction = (
            "Turn vaporizer <b>OFF</b>. Increase FGF to <b>8-10 L/min</b>.<br><br>"
            "<i>High flow accelerates volatile agent washout.</i>"
        )
        stop_agents_check = _require_agents_stopped_balanced()
        stop_agents_tab = "Machine"
    else:
        stop_agents_instruction = (
            "Turn <b>OFF</b> propofol and remifentanil infusions.<br><br>"
            "<i>Remi t½ is ~3-4 min. Propofol emergence in 5-10 min.</i>"
        )
        stop_agents_check = _require_agents_stopped_tiva()
        stop_agents_tab = "Medications"
    
    steps = [
        ScenarioStep(
            id="ASSESS",
            title="Assess hemodynamic stability",
            instruction=(
                "Verify: <b>BIS 40-60</b>, <b>MAP > 65</b>, <b>EtCO₂ 35-45</b>.<br><br>"
                "<i>Ensure surgery complete and patient is warm before emergence.</i>"
            ),
            check_requirements=_require_assess(),
        ),
        ScenarioStep(
            id="STOP_AGENTS",
            title="Discontinue anesthetics" if is_balanced else "Stop infusions",
            instruction=stop_agents_instruction,
            check_requirements=stop_agents_check,
            target_tab=stop_agents_tab,
        ),
        ScenarioStep(
            id="AWAIT_EMERGENCE",
            title="Await emergence",
            instruction=(
                "Monitor <b>BIS rising</b> toward >70. Patient will start breathing.<br>"
                "Watch for: movement, coughing, eye opening.<br><br>"
                "<i>Avoid premature stimulation at BIS 60-70 (risk of laryngospasm).</i>"
            ),
            check_requirements=_require_awakening(),
        ),
        ScenarioStep(
            id="EXTUBATE",
            title="Extubation",
            instruction=(
                "Criteria: <b>BIS > 80</b>, following commands, <b>RR > 8</b>, <b>Vt > 5 mL/kg</b>.<br>"
                "Remove ETT -> select 'Mask' or 'None'.<br><br>"
                "<i>Suction oropharynx, deflate cuff, remove on inspiration/expiration.</i>"
            ),
            check_requirements=_require_extubation_criteria(),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="RECOVERY",
            title="Post-anesthesia care",
            instruction=(
                "Apply supplemental O₂. Monitor: <b>SpO₂ > 95%</b>, hemodynamic stability.<br><br>"
                "<i>PACU handoff: procedure, anesthetics, airway, blood loss, concerns.</i>"
            ),
            check_requirements=_require_recovery(),
            target_tab="Machine",
        ),
    ]
    
    scenario_id = "emergence_balanced" if is_balanced else "emergence_tiva"
    return Scenario(
        id=scenario_id,
        name="Emergence sequence",
        icon="",
        description="Learn the emergence and extubation sequence.",
        steps=steps,
    )
