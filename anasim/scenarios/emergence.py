"""Emergence scenarios after balanced anesthesia or TIVA."""

from typing import Literal

from anasim.core.action_log import ACTION_AIRWAY, ACTION_FGF, ACTION_VAPORIZER
from anasim.core.state import AirwayType

from .base import (
    Requirement,
    Scenario,
    ScenarioStep,
    action_taken_this_step,
    join_messages,
    monitor_value,
    require_all,
    require_infusions_stopped,
    require_maintenance_depth,
    require_map_at_least,
    require_oxygen_flow,
)


def _require_normocapnia(engine) -> tuple[bool, str]:
    if not engine.state.etco2_signal_valid:
        return False, "Awaiting exhaled CO₂"
    etco2 = monitor_value(engine, "etco2")
    met = 35 <= etco2 <= 45
    return met, "" if met else f"EtCO₂: {etco2:.0f}/35-45 mmHg"


def _require_agents_stopped_balanced() -> Requirement:
    """Check volatile agent and remifentanil stopped, with high fresh gas flow."""
    opioid_check = require_infusions_stopped(
        "remi", fail_message="Stop remifentanil for this objective"
    )

    def check(engine) -> tuple[bool, str]:
        gas_off = not engine.circuit.vaporizer_on or engine.circuit.vaporizer_setting < 0.1
        high_flow = engine.circuit.fgf_total() > 6.0
        vaporizer_records = engine.actions.since_step(ACTION_VAPORIZER)
        vaporizer_stopped = any(record.amount < 0.1 for record in vaporizer_records)
        flow_changed = action_taken_this_step(engine, ACTION_FGF)
        opioid_stopped, opioid_message = opioid_check(engine)
        if gas_off and high_flow and vaporizer_stopped and flow_changed and opioid_stopped:
            return True, ""
        msgs = []
        if not gas_off or not vaporizer_stopped:
            msgs.append("Turn vaporizer off for this objective")
        if not high_flow or not flow_changed:
            msgs.append("Set FGF above 6 L/min for this objective")
        if not opioid_stopped:
            msgs.append(opioid_message)
        return False, join_messages(msgs)
    return check


def _require_awakening(engine) -> tuple[bool, str]:
    """Check patient emerging (BIS > 70, spontaneous breathing)."""
    bis = monitor_value(engine, "bis")
    rr = engine.resp.state.rr
    msgs = []
    if not bis > 70:
        msgs.append(f"BIS: {bis:.0f}/70+")
    if not (rr > 6 and not engine.resp.state.apnea):
        msgs.append(f"Spontaneous RR: {rr:.0f}/6+")
    return not msgs, join_messages(msgs)


def _require_extubation_readiness(engine) -> tuple[bool, str]:
    """Check wakefulness, measured breathing, oxygenation, and block recovery."""
    resp = engine.resp.state
    rr = engine.state.rr
    breathing = rr > 8 and resp.rr > 8 and not resp.apnea
    minimum_vt = 5.0 * engine.patient.predicted_body_weight()
    volume_ok = engine.state.vt > minimum_vt
    block_recovered = engine.state.tof >= 90.0
    spo2 = monitor_value(engine, "spo2")
    oxygenated = engine.state.spo2_signal_valid and spo2 > 95.0
    bis = monitor_value(engine, "bis")
    awake = bis > 80
    msgs = []
    if not awake:
        msgs.append(f"BIS: {bis:.0f}/80+")
    if not breathing:
        msgs.append(f"Spontaneous RR: {min(rr, resp.rr):.0f}/8+")
    if not volume_ok:
        msgs.append(f"Exhaled VT: {engine.state.vt:.0f}/{minimum_vt:.0f}+ mL")
    if not block_recovered:
        msgs.append(f"TOF ratio: {engine.state.tof:.0f}%/90%+")
    if not oxygenated:
        msgs.append(f"SpO₂: {spo2:.0f}%/95%+" if engine.state.spo2_signal_valid else "Awaiting valid SpO₂")
    if engine.vent.is_on or engine.bag_mask_active:
        msgs.append("Stop assisted ventilation")
    return not msgs, join_messages(msgs)


def _require_tube_removed(engine) -> tuple[bool, str]:
    """Require tube removal during this objective."""
    extubated = engine.state.airway_mode in (AirwayType.MASK, AirwayType.NONE)
    airway_action = action_taken_this_step(
        engine, ACTION_AIRWAY, AirwayType.MASK.value, AirwayType.NONE.value,
    )
    met = extubated and airway_action
    return met, "" if met else "Select Facemask or Disconnected for this objective"


def create_emergence(maint_type: Literal["balanced", "tiva"] = "balanced") -> Scenario:
    """Create the emergence scenario for "balanced" or "tiva" maintenance."""
    if maint_type not in ("balanced", "tiva"):
        raise ValueError(f"Unsupported maintenance type: {maint_type!r}")
    is_balanced = maint_type == "balanced"

    if is_balanced:
        stop_agents_instruction = (
            "Turn vaporizer <b>OFF</b>. Stop remifentanil in Medications. "
            "Increase FGF to <b>8-10 L/min</b>.<br><br>"
            "<i>High flow accelerates volatile agent washout.</i>"
        )
        stop_agents_check = _require_agents_stopped_balanced()
        stop_agents_tab = "Machine"
    else:
        stop_agents_instruction = (
            "Turn <b>OFF</b> propofol and remifentanil infusions.<br><br>"
            "Observe the return of spontaneous breathing as the drug effect decreases."
        )
        stop_agents_check = require_infusions_stopped(
            "propofol", "remi", fail_message="Stop propofol and remifentanil for this objective",
        )
        stop_agents_tab = "Medications"

    steps = [
        ScenarioStep(
            id="ASSESS",
            title="Assess hemodynamic stability",
            instruction=(
                "Confirm <b>BIS 40-60</b>, <b>MAP ≥ 65 mmHg</b>, and valid <b>EtCO₂ 35-45 mmHg</b>. "
                "Treat hypotension and adjust ventilation as needed. In practice, confirm surgery is complete, "
                "maintain warmth, and arrange postoperative analgesia before stopping remifentanil. "
                "Its analgesic effect wears off quickly."
            ),
            check_requirements=require_all(
                require_maintenance_depth, require_map_at_least(), _require_normocapnia,
            ),
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
                "Wait for <b>BIS > 70</b> and <b>spontaneous RR > 6/min</b>.<br>"
                "Ventilator-delivered breaths do not establish spontaneous breathing."
            ),
            check_requirements=_require_awakening,
        ),
        ScenarioStep(
            id="EXTUBATION_READINESS",
            title="Assess readiness for extubation",
            instruction=(
                "Stop controlled ventilation, keep the tracheal tube in place, and assess "
                "spontaneous breathing. Be ready to assist if ventilation is inadequate. "
                "Wait for <b>BIS > 80</b>, spontaneous "
                "<b>RR > 8/min</b> and exhaled <b>VT > 5 mL/kg predicted body weight</b>, "
                "<b>TOF ratio ≥ 90%</b>, and <b>SpO₂ > 95%</b>.<br><br>"
                "<i>These are simulator criteria. In practice, assess command following, "
                "airway protection, adequate ventilation, and the airway plan; BIS alone does not establish readiness.</i>"
            ),
            check_requirements=require_all(
                lambda engine: (
                    engine.state.airway_mode == AirwayType.ETT,
                    "Keep the tracheal tube in place while assessing readiness",
                ),
                _require_extubation_readiness,
            ),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="EXTUBATE",
            title="Extubation",
            instruction=(
                "With readiness confirmed and the ventilator stopped, select "
                "<b>Facemask</b> to simulate removing the tracheal tube and applying oxygen. "
                "Reassess breathing and airway patency immediately."
            ),
            check_requirements=require_all(_require_extubation_readiness, _require_tube_removed),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="RECOVERY",
            title="Post-anesthesia care",
            instruction=(
                "Select <b>Facemask</b> and set <b>O₂ 5 L/min</b>, air and N₂O 0 L/min "
                "to represent supplemental oxygen. Confirm adequate spontaneous breathing, "
                "<b>SpO₂ > 95%</b>, and <b>MAP ≥ 65 mmHg</b>. A separate PACU mask is not modeled.<br><br>"
                "<i>Handoff should cover the procedure, anesthetics, analgesia, airway, blood loss, and current concerns.</i>"
            ),
            check_requirements=require_all(
                _require_extubation_readiness,
                require_map_at_least(),
                require_oxygen_flow(5.0),
                lambda engine: (
                    engine.state.airway_mode == AirwayType.MASK,
                    "Select Facemask for supplemental oxygen",
                ),
            ),
            target_tab="Machine",
        ),
    ]

    scenario_id = "emergence_balanced" if is_balanced else "emergence_tiva"
    return Scenario(
        id=scenario_id,
        name="Emergence sequence",
        description="Learn the emergence and extubation sequence.",
        steps=steps,
    )
