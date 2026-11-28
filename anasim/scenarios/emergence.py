"""Emergence scenarios after balanced anesthesia or TIVA."""

from typing import Literal

from anasim.core.action_log import (
    ACTION_AIRWAY,
    ACTION_CASE_START,
    ACTION_DRUG_BOLUS,
    ACTION_FGF,
    ACTION_INFUSION_RATE,
    ACTION_TCI_TARGET,
    ACTION_VAPORIZER,
)
from anasim.core.monitors import display_int
from anasim.core.projection import assisted_positive_pressure, sync_pk_state
from anasim.core.state import AirwayType

from .base import (
    CaseOutcome,
    ControlTab,
    Requirement,
    Scenario,
    ScenarioStep,
    action_taken_in_case,
    check_etco2_range,
    format_unmet_int,
    infusion_running,
    join_messages,
    monitor_value,
    neuromuscular_response,
    normocapnia_bounds,
    preroll_scenario,
    require_all,
    require_infusions_stopped,
    require_maintenance_depth,
    require_map_at_least,
    require_oxygen_flow,
    stability_outcome,
)

# Rocuronium 0.6 mg/kg at induction leaves a TOF count of 2 at the start.
# Sevoflurane potentiates rocuronium and slows recovery (Kleijn 2011; Lowry 1998).
ROCURONIUM_DOSE_MG_KG = 0.6
ROCURONIUM_MINUTES = {"balanced": 55, "tiva": 35}


def _prepare_emergence(maint_type: Literal["balanced", "tiva"], minutes: int):
    """Taper maintenance into late surgical closure and seed residual rocuronium block."""
    def prepare(engine) -> None:
        # Taper anesthetics and ease respiratory rate slightly as during skin closure so
        # emergence starts near upper maintenance depth (BIS ~55-57) with mid-normocapnic EtCO2.
        closure_rr = max(8.0, round(engine.vent.settings.rr * 0.88, 1))
        engine.vent.update_settings(rr=closure_rr)
        if maint_type == "balanced":
            engine.set_fgf(1.0, 1.0, 0.0)
            engine.set_vaporizer(round(engine.vaporizer.state.setting * 0.62, 2))
            preroll_scenario(engine, 480.0, dt=5.0, settle_s=15.0, sample_interval_s=10.0)
        else:
            engine.set_fgf(1.0, 1.0, 0.0)
            if engine.config.tci_enabled:
                engine.set_drug_target("propofol", round(engine.get_drug_state("propofol")["target"] * 0.78, 2))
                engine.set_drug_target("remi", round(engine.get_drug_state("remi")["target"] * 0.60, 2))
            else:
                engine.infusion_rates["propofol"] *= 0.58
                engine.infusion_rates["remi"] *= 0.60
            preroll_scenario(engine, 480.0, dt=5.0, settle_s=15.0, sample_interval_s=10.0)

        roc, tof = engine.pk["roc"], engine.tof_pd
        mac_sevo, mac_n2o = engine.pk_sevo.state.mac, engine.pk_n2o.state.mac
        roc.state.c1 += ROCURONIUM_DOSE_MG_KG * engine.patient.weight / roc.v1
        for _ in range(minutes * 60):
            roc.step(1.0, 0.0)
            tof.step_recovery(1.0, roc, mac_sevo, mac_n2o)
        sync_pk_state(engine)
    return prepare


def _assess_instruction_for(is_balanced: bool):
    def instruction(engine) -> str:
        low, high = normocapnia_bounds(engine)
        if is_balanced:
            analgesia_note = (
                "Confirm surgery is finished, keep the patient warm, and verify adequate "
                "multimodal or opioid analgesia (such as fentanyl or hydromorphone) before emergence."
            )
        else:
            analgesia_note = (
                "Before stopping remifentanil, confirm surgery is finished, keep the patient warm, "
                "and give longer-acting analgesia because remifentanil wears off quickly."
            )
        return (
            f"Confirm <b>BIS 40-60</b>, <b>MAP ≥ 65 mmHg</b>, and <b>EtCO₂ {low}-{high} mmHg</b>; "
            f"adjust vasopressors and ventilation as needed.<br><br><i>{analgesia_note}</i>"
        )
    return instruction


def _require_normocapnia(engine) -> tuple[bool, str]:
    return check_etco2_range(engine, *normocapnia_bounds(engine))


def _require_agents_stopped_balanced() -> Requirement:
    """Check that the volatile agent is stopped and fresh gas flow is high 100% O₂."""
    def check(engine) -> tuple[bool, str]:
        gas_off = not engine.circuit.vaporizer_on or engine.circuit.vaporizer_setting < 0.1
        high_o2_flow = (
            engine.circuit.fgf_total() > 6.0
            and engine.circuit.delivered_o2_flow() >= 6.0
            and engine.circuit.fgf_n2o <= 0.1
        )
        vaporizer_records = engine.actions.since_case(ACTION_VAPORIZER)
        vaporizer_stopped = any(record.amount < 0.1 for record in vaporizer_records)
        flow_changed = action_taken_in_case(engine, ACTION_FGF)
        if gas_off and high_o2_flow and vaporizer_stopped and flow_changed:
            return True, ""
        msgs = []
        if not gas_off or not vaporizer_stopped:
            msgs.append("Turn off sevoflurane")
        if not high_o2_flow or not flow_changed:
            msgs.append("Set high O₂ flow (> 6 L/min, N₂O off)")
        return False, join_messages(msgs)
    return check


def _require_awakening(engine) -> tuple[bool, str]:
    """Check that the patient is emerging: BIS above 70 and breathing spontaneously."""
    bis = monitor_value(engine, "bis")
    rr = engine.resp.state.rr
    msgs = []
    if not bis > 70:
        msgs.append(f"BIS: {format_unmet_int(bis, low=70)}/70+")
    if not (rr > 6 and not engine.resp.state.apnea):
        msgs.append(f"Spontaneous RR: {format_unmet_int(rr, low=6)}/6+")
    return not msgs, join_messages(msgs)


def _nmb_valid_since(engine) -> float:
    """Earliest timestamp at which a TOF recovery reading remains valid."""
    valid_since = 0.0
    for record in engine.actions.records:
        if record.action == ACTION_CASE_START:
            valid_since = max(valid_since, record.time)
        elif (
            record.action in (ACTION_DRUG_BOLUS, ACTION_INFUSION_RATE, ACTION_TCI_TARGET)
            and record.label in ("roc", "succinylcholine")
        ):
            valid_since = max(valid_since, record.time)
    if infusion_running(engine, "roc"):
        return float("inf")
    return valid_since


def _check_block_recovery(engine, *, require_recent: bool) -> tuple[bool, str]:
    """Require quantitative recovery and a valid TOF measurement."""
    reading = engine.nmb.reading
    if (
        reading is None
        or reading.mode != "tof"
        or reading.time < _nmb_valid_since(engine)
        or (require_recent and engine.state.time - reading.time > 60.0)
    ):
        return False, "Measure TOF and confirm a ratio of at least 90%"
    if reading.low_signal:
        return False, "Confirm a reliable measured TOF ratio of at least 90%"
    if reading.ratio is None:
        return False, f"Measured TOF count: {reading.count}/4; assess reversal"
    if reading.ratio < 90.0:
        return False, f"Measured TOF ratio: {format_unmet_int(reading.ratio, low=90.0)}%/90%+; assess reversal"
    label, response = neuromuscular_response(engine)
    if response < 90.0:
        return False, f"{label}: {format_unmet_int(response, low=90.0)}%/90%+; assess reversal"
    return True, ""


def _require_block_recovery(engine) -> tuple[bool, str]:
    return _check_block_recovery(engine, require_recent=True)


def _require_confirmed_block_recovery(engine) -> tuple[bool, str]:
    return _check_block_recovery(engine, require_recent=False)


def _require_extubation_readiness(engine) -> tuple[bool, str]:
    """Check wakefulness, measured breathing, oxygenation, and block recovery."""
    resp = engine.resp.state
    rr = engine.state.rr
    breathing = rr > 8 and resp.rr > 8 and not resp.apnea
    minimum_vt = 5.0 * engine.patient.predicted_body_weight()
    volume_ok = engine.state.vt > minimum_vt
    block_recovered, block_message = _require_confirmed_block_recovery(engine)
    spo2 = monitor_value(engine, "spo2")
    oxygenated = engine.state.spo2_signal_valid and spo2 >= 94.0
    bis = monitor_value(engine, "bis")
    awake = bis > 80 and engine.state.loc < 0.20
    msgs = []
    if not awake:
        msgs.append(
            f"BIS: {format_unmet_int(bis, low=80)}/80+"
            if bis <= 80
            else "Continue anesthetic washout (Stage II risk)"
        )
    if not breathing:
        msgs.append(f"Spontaneous RR: {format_unmet_int(min(rr, resp.rr), low=8)}/8+")
    if not volume_ok:
        shown_min_vt = int(display_int(minimum_vt))
        msgs.append(f"Exhaled VT: {format_unmet_int(engine.state.vt, low=shown_min_vt)}/{shown_min_vt}+ mL")
    if not block_recovered:
        msgs.append(block_message)
    if not oxygenated:
        msgs.append(
            f"SpO₂: {format_unmet_int(spo2, low=94.0)}%/94%+"
            if engine.state.spo2_signal_valid
            else "Awaiting valid SpO₂"
        )
    if assisted_positive_pressure(engine) or engine.bag_mask_active:
        msgs.append("Stop assisted ventilation")
    return not msgs, join_messages(msgs)


def _require_tube_removed(engine) -> tuple[bool, str]:
    """Require tube removal during this case."""
    extubated = engine.state.airway_mode in (AirwayType.MASK, AirwayType.NONE)
    airway_action = action_taken_in_case(
        engine, ACTION_AIRWAY, AirwayType.MASK.value, AirwayType.NONE.value,
    )
    met = extubated and airway_action
    return met, "" if met else "Select Facemask or Disconnected"


def _extubated_and_breathing(engine) -> bool:
    return (
        engine.state.airway_mode != AirwayType.ETT
        and not engine.resp.state.apnea
        and not (assisted_positive_pressure(engine) or engine.bag_mask_active)
        and monitor_value(engine, "bis") > 80
        and engine.state.loc < 0.20
    )


# Readiness counts at the moment of tube removal, not at an earlier favorable moment.
OUTCOMES = (
    CaseOutcome("block", "Block recovery confirmed at tube removal",
                lambda engine: _require_confirmed_block_recovery(engine)[0],
                "Confirm quantitative neuromuscular recovery before extubation.", on_action="extubation"),
    CaseOutcome("readiness", "Readiness established at tube removal",
                lambda engine: _require_extubation_readiness(engine)[0],
                "Assess spontaneous ventilation, wakefulness, and oxygenation before removing the tube.",
                on_action="extubation"),
    CaseOutcome("extubation", "Tube removed with spontaneous ventilation", _extubated_and_breathing,
                "Establish adequate spontaneous breathing and wakefulness before completing emergence."),
    stability_outcome("Continue respiratory and circulatory support during recovery."),
)


def create_emergence(maint_type: Literal["balanced", "tiva"] = "balanced") -> Scenario:
    """Create the emergence scenario for "balanced" or "tiva" maintenance."""
    if maint_type not in ("balanced", "tiva"):
        raise ValueError(f"Unsupported maintenance type: {maint_type!r}")
    is_balanced = maint_type == "balanced"
    rocuronium_minutes = ROCURONIUM_MINUTES[maint_type]

    if is_balanced:
        stop_agents_instruction = (
            "Turn off <b>sevoflurane</b> and increase fresh gas flow to <b>8-10 L/min of 100% O₂</b> "
            "(air and N₂O 0 L/min).<br><br>"
            "<i>High oxygen flow speeds volatile washout and preoxygenates the patient before extubation.</i>"
        )
        stop_agents_check = _require_agents_stopped_balanced()
        stop_agents_tab: ControlTab = "Machine"
    else:
        stop_agents_instruction = (
            "Stop the <b>propofol and remifentanil</b> infusions in Drugs, and switch to "
            "<b>100% O₂</b> in Machine to preoxygenate before extubation.<br><br>"
            "<i>Watch for spontaneous breathing as drug levels fall.</i>"
        )
        stop_agents_check = require_infusions_stopped(
            "propofol", "remi", fail_message="Stop propofol and remifentanil",
        )
        stop_agents_tab = "Medications"

    steps = [
        ScenarioStep(
            id="ASSESS",
            title="Assess hemodynamic stability",
            instruction=_assess_instruction_for(is_balanced),
            check_requirements=require_all(
                require_maintenance_depth, require_map_at_least(), _require_normocapnia,
            ),
        ),
        ScenarioStep(
            id="ASSESS_BLOCK",
            title="Assess block and reversal",
            instruction=(
                f"Rocuronium {ROCURONIUM_DOSE_MG_KG:g} mg/kg was given {rocuronium_minutes} minutes ago. "
                "Check <b>TOF</b> and confirm a <b>TOF ratio ≥ 90%</b> before stopping anesthesia: give "
                "<b>sugammadex 2 mg/kg</b> actual body weight for moderate block (TOF ≥ 2), or <b>4 mg/kg</b> "
                "for deep block (TOF 0-1 with PTC ≥ 1-2; 16 mg/kg for immediate reversal at PTC 0).<br><br>"
                "<i>Keep anesthesia and ventilation running until recovery is confirmed.</i>"
            ),
            check_requirements=_require_block_recovery,
            target_tab="Medications",
        ),
        ScenarioStep(
            id="STOP_AGENTS",
            title="Stop anesthetics",
            instruction=stop_agents_instruction,
            check_requirements=stop_agents_check,
            target_tab=stop_agents_tab,
        ),
        ScenarioStep(
            id="AWAIT_EMERGENCE",
            title="Await emergence",
            instruction=(
                "Wait for <b>BIS > 70</b> and <b>spontaneous RR > 6/min</b>. "
                "You can bridge through <b>PSV</b> or lower the ventilator rate to let CO₂ rise gently "
                "and trigger spontaneous breathing.<br><br>"
                "<i>Ventilator-delivered mandatory breaths do not count as spontaneous breathing.</i>"
            ),
            check_requirements=_require_awakening,
        ),
        ScenarioStep(
            id="EXTUBATION_READINESS",
            title="Assess readiness for extubation",
            instruction=(
                "Stop the ventilator (or switch to <b>CPAP</b>) while keeping the tracheal tube in place, "
                "and be ready to assist breathing. Confirm <b>BIS > 80</b>, spontaneous <b>RR > 8/min</b>, exhaled "
                "<b>VT > 5 mL/kg predicted body weight</b>, <b>TOF ratio ≥ 90%</b>, and <b>SpO₂ ≥ 94%</b>.<br><br>"
                "<i>At the bedside, also confirm command following, airway reflexes, and the extubation plan; "
                "BIS alone does not establish readiness.</i>"
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
            title="Extubate",
            instruction=(
                "With readiness confirmed, suction the oropharynx, deflate the cuff at peak inspiration, "
                "and select <b>Facemask</b> to extubate and give oxygen. Reassess breathing and airway "
                "patency immediately."
            ),
            check_requirements=require_all(_require_extubation_readiness, _require_tube_removed),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="RECOVERY",
            title="Begin recovery care",
            instruction=(
                "Keep <b>Facemask</b> selected with <b>O₂ 5 L/min</b> (air and N₂O 0 L/min) for supplemental oxygen. "
                "Confirm spontaneous breathing, <b>SpO₂ ≥ 94%</b>, and <b>MAP ≥ 65 mmHg</b>.<br><br>"
                "<i>Transition to transport oxygen and include the procedure, anesthetics, analgesia, airway, "
                "blood loss, and active concerns in PACU handoff.</i>"
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
        briefing=(
            "Skin closure is finishing and anesthetics have been lightened. "
            "Reverse residual block, awaken the patient, and extubate when ready."
        ),
        steps=steps,
        outcomes=OUTCOMES,
        setup_engine=_prepare_emergence(maint_type, rocuronium_minutes),
    )
