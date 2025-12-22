"""
Emergence scenario definitions (Balanced and TIVA variants).
"""

from typing import Tuple
from .base import Scenario, ScenarioStep, require_bis_above, require_all


def _require_assess() -> callable:
    """Check stable maintenance conditions."""
    def check(engine) -> Tuple[bool, str]:
        bis_ok = 35 < engine.state.bis < 65
        map_ok = engine.state.map > 60
        if bis_ok and map_ok:
            return True, ""
        msgs = []
        if not bis_ok: msgs.append(f"BIS: {engine.state.bis:.0f}")
        if not map_ok: msgs.append(f"MAP: {engine.state.map:.0f}")
        return False, "" + ", ".join(msgs)
    return check


def _require_agents_stopped_balanced() -> callable:
    """Check volatile agent stopped and high flow gas."""
    def check(engine) -> Tuple[bool, str]:
        gas_off = not engine.circuit.vaporizer_on or engine.circuit.vaporizer_setting < 0.1
        high_flow = engine.circuit.fgf_total() > 6.0
        if gas_off and high_flow:
            return True, ""
        msgs = []
        if not gas_off: msgs.append("Vaporizer still on")
        if not high_flow: msgs.append(f"FGF: {engine.circuit.fgf_total():.1f}/6+ L/min")
        return False, "" + ", ".join(msgs)
    return check


def _require_agents_stopped_tiva() -> callable:
    """Check TIVA infusions stopped."""
    def check(engine) -> Tuple[bool, str]:
        prop_off = (engine.propofol_rate_mg_sec == 0) and (not engine.tci_prop or engine.tci_prop.target == 0)
        remi_off = (engine.remi_rate_ug_sec == 0) and (not engine.tci_remi or engine.tci_remi.target == 0)
        if prop_off and remi_off:
            return True, ""
        msgs = []
        if not prop_off: msgs.append("Propofol running")
        if not remi_off: msgs.append("Remi running")
        return False, "" + ", ".join(msgs)
    return check


def _require_awakening() -> callable:
    """Check patient emerging (BIS > 70, spontaneous breathing)."""
    def check(engine) -> Tuple[bool, str]:
        bis_ok = engine.state.bis > 70
        rr_ok = engine.state.rr > 6
        if bis_ok and rr_ok:
            return True, ""
        msgs = []
        if not bis_ok: msgs.append(f"BIS: {engine.state.bis:.0f}/70+")
        if not rr_ok: msgs.append(f"RR: {engine.state.rr:.0f}/6+")
        return False, "" + ", ".join(msgs)
    return check


def _require_extubation_criteria() -> callable:
    """Check extubation criteria met."""
    from core.state import AirwayType
    def check(engine) -> Tuple[bool, str]:
        breathing = engine.state.rr > 8 and not engine.state.apnea
        awake = engine.state.bis > 80
        extubated = engine.state.airway_mode != AirwayType.ETT
        if breathing and awake and extubated:
            return True, ""
        msgs = []
        if not awake: msgs.append(f"BIS: {engine.state.bis:.0f}/80+")
        if not breathing: msgs.append(f"RR: {engine.state.rr:.0f}/8+")
        if not extubated: msgs.append("Still intubated")
        return False, "" + ", ".join(msgs)
    return check


def _require_recovery() -> callable:
    """Check recovery room criteria."""
    def check(engine) -> Tuple[bool, str]:
        spo2_ok = engine.state.spo2 > 95
        rr_ok = engine.state.rr > 10
        not_apneic = not engine.state.apnea
        if spo2_ok and rr_ok and not_apneic:
            return True, ""
        msgs = []
        if not spo2_ok: msgs.append(f"SpO₂: {engine.state.spo2:.0f}/95+")
        if not rr_ok: msgs.append(f"RR: {engine.state.rr:.0f}/10+")
        return False, "" + ", ".join(msgs)
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
            "<b>Step 2/5: Discontinue anesthetics</b><br>"
            "Turn vaporizer <b>OFF</b>. Increase FGF to <b>8-10 L/min</b>.<br><br>"
            "<i>High flow accelerates volatile agent washout.</i>"
        )
        stop_agents_check = _require_agents_stopped_balanced()
    else:
        stop_agents_instruction = (
            "<b>Step 2/5: Stop infusions</b><br>"
            "Turn <b>OFF</b> propofol and remifentanil infusions.<br><br>"
            "<i>Remi t½ is ~3-4 min. Propofol emergence in 5-10 min.</i>"
        )
        stop_agents_check = _require_agents_stopped_tiva()
    
    steps = [
        ScenarioStep(
            id="ASSESS",
            title="Assess hemodynamic stability",
            instruction=(
                "<b>Step 1/5: Assess hemodynamic stability</b><br>"
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
        ),
        ScenarioStep(
            id="AWAIT_EMERGENCE",
            title="Await emergence",
            instruction=(
                "<b>Step 3/5: Await emergence</b><br>"
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
                "<b>Step 4/5: Extubation</b><br>"
                "Criteria: <b>BIS > 80</b>, following commands, <b>RR > 8</b>, <b>Vt > 5 mL/kg</b>.<br>"
                "Remove ETT -> select 'Mask' or 'None'.<br><br>"
                "<i>Suction oropharynx, deflate cuff, remove on inspiration/expiration.</i>"
            ),
            check_requirements=_require_extubation_criteria(),
        ),
        ScenarioStep(
            id="RECOVERY",
            title="Post-anesthesia care",
            instruction=(
                "<b>Step 5/5: Post-anesthesia care</b><br>"
                "Apply supplemental O₂. Monitor: <b>SpO₂ > 95%</b>, hemodynamic stability.<br><br>"
                "<i>PACU handoff: procedure, anesthetics, airway, blood loss, concerns.</i>"
            ),
            check_requirements=_require_recovery(),
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
