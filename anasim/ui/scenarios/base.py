"""
Base classes for the scenario/tutorial system.

Scenarios are data-driven definitions of guided learning sequences.
Each scenario contains a list of steps with instructions and requirements.

Requirement checks come in two kinds:

- Action objectives ("give fluids", "select the ETT") read `engine.actions`
  and are satisfied only by an action taken while the objective is active, so
  an intervention performed before the objective appeared cannot complete it.
- State and physiologic objectives ("confirm EtCO2", "MAP > 65") inspect
  current engine state, because they describe a condition to reach or hold
  rather than a dose to give.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Tuple

from anasim.core.action_log import (
    ACTION_AIRWAY,
    ACTION_BAG_MASK,
    ACTION_DRUG_BOLUS,
    ACTION_EVENT_START,
    ACTION_EVENT_STOP,
    ACTION_FGF,
    ACTION_FLUID,
    ACTION_INFUSION_RATE,
    ACTION_OXYGEN_SUPPLY,
    ACTION_TCI_TARGET,
    ACTION_VAPORIZER,
)


ControlTab = Literal["Machine", "Medications", "Events"]

VASOPRESSOR_KEYS = ("nore", "phenyl", "epi")


@dataclass
class ScenarioStep:
    """A single step definition in a scenario sequence."""
    id: str
    title: str
    instruction: str
    check_requirements: Callable[[object], Tuple[bool, str]]
    target_tab: ControlTab | None = None


@dataclass
class Scenario:
    """A complete scenario definition containing metadata and ordered steps."""
    id: str
    name: str
    icon: str
    description: str
    steps: List[ScenarioStep] = field(default_factory=list)
    setup_engine: Callable[[object], None] | None = None
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def __getitem__(self, idx: int) -> ScenarioStep:
        return self.steps[idx]

    def prepare(self, engine) -> None:
        """Apply scenario-specific starting conditions."""
        if self.setup_engine is not None:
            self.setup_engine(engine)


# Requirement check helper functions
def action_taken_this_step(engine, action: str, *labels: str) -> bool:
    """Return whether a matching control action occurred during this objective."""
    records = engine.actions.since_step(action, labels=labels or None)
    return bool(records)


def require_airway_selected(airway_type: str) -> Callable:
    """Require selection of an airway type during this objective."""
    from anasim.core.state import AirwayType
    target, label = {
        "None": (AirwayType.NONE, "No airway"),
        "Mask": (AirwayType.MASK, "Facemask"),
        "ETT": (AirwayType.ETT, "ET tube"),
    }[airway_type]
    
    def check(engine) -> Tuple[bool, str]:
        selected = action_taken_this_step(engine, ACTION_AIRWAY, target.value)
        met = selected and engine.state.airway_mode == target
        return met, "" if met else f"Select {label}"
    return check


def _fgf_preox_state(engine) -> Tuple[bool, str]:
    """Return whether current fresh gas flow is adequate for preoxygenation."""
    o2_ok = engine.circuit.fgf_o2 >= 8.0
    air_ok = engine.circuit.fgf_air < 1.0
    n2o_ok = engine.circuit.fgf_n2o <= 0.1
    if o2_ok and air_ok and n2o_ok:
        return True, ""
    msgs = []
    if not o2_ok:
        msgs.append(f"O₂: {engine.circuit.fgf_o2:.1f}/8+ L/min")
    if not air_ok:
        msgs.append(f"Air: {engine.circuit.fgf_air:.1f}/0 L/min")
    if not n2o_ok:
        msgs.append(f"N₂O: {engine.circuit.fgf_n2o:.1f}/0 L/min")
    return False, join_messages(msgs)


def require_fgf_set_for_preox() -> Callable:
    """Require a fresh gas flow action and adequate current preoxygenation flow."""
    def check(engine) -> Tuple[bool, str]:
        state_met, message = _fgf_preox_state(engine)
        action_met = action_taken_this_step(engine, ACTION_FGF)
        if state_met and action_met:
            return True, ""
        if state_met:
            return False, "Set fresh gas flow for this objective"
        return False, message
    return check


def require_preoxygenation_flow() -> Callable:
    """Check that current fresh gas flow remains adequate for preoxygenation."""
    return _fgf_preox_state


def require_propofol_cp(threshold: float = 2.0) -> Callable:
    """Check propofol plasma concentration."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.propofol_cp > threshold
        return met, "" if met else f"Propofol Cp: {engine.state.propofol_cp:.1f}/{threshold}+ µg/mL"
    return check


def monitor_value(engine, attr: str):
    """Return the learner-facing monitor value for a vital sign."""
    return engine.state.display_value(attr)


def join_messages(messages) -> str:
    """Join non-empty requirement messages consistently."""
    return ", ".join(message for message in messages if message)


def bolus_given_this_step(engine, drug_key: str) -> bool:
    """Return True when the drug was bolused during this objective."""
    records = engine.actions.since_step(ACTION_DRUG_BOLUS, labels=(drug_key,))
    return any(record.amount > 0 for record in records)


def infusion_set_this_step(engine, *drug_keys: str) -> bool:
    """Return True when an infusion rate or TCI target was set during this objective."""
    records = engine.actions.since_step(
        ACTION_INFUSION_RATE, ACTION_TCI_TARGET, labels=drug_keys
    )
    return any(record.amount > 0 for record in records)


def infusion_running(engine, *drug_keys: str) -> bool:
    """Return True when a drug is running by manual rate or TCI target."""
    states = (engine.get_drug_state(key) for key in drug_keys)
    return any(state["rate"] > 0 or state["target"] > 0 for state in states)


def require_stable_baseline_vitals(
    hr_min: float = 60.0,
    hr_max: float = 100.0,
    map_min: float = 60.0,
    spo2_min: float = 94.0,
    fail_message: str = "Wait for stable baseline vitals",
) -> Callable:
    """Check that monitor-facing baseline vitals are in a reasonable range."""
    def check(engine) -> Tuple[bool, str]:
        hr = monitor_value(engine, "hr")
        map_val = monitor_value(engine, "map")
        spo2 = monitor_value(engine, "spo2")
        stable = hr_min < hr < hr_max and map_val > map_min and spo2 > spo2_min
        return (True, "") if stable else (False, fail_message)
    return check


def require_fluid_given(
    min_ml: float,
    labels: Tuple[str, ...],
    fail_prefix: str = "Give fluid bolus",
) -> Callable:
    """Check that fluid was given while this objective was active."""
    def check(engine) -> Tuple[bool, str]:
        fluid_given = engine.actions.total_since_step(ACTION_FLUID, labels=labels)
        if fluid_given >= min_ml:
            return True, ""
        if fluid_given > 0.0:
            return False, f"{fail_prefix} ({fluid_given:.0f}/{min_ml:.0f} mL given)"
        return False, f"{fail_prefix} ({min_ml:.0f} mL via Events tab)"
    return check


def require_infusion_started(*drug_keys: str, fail_message: str) -> Callable:
    """Check that one of the infusions was set during this objective and is running."""
    def check(engine) -> Tuple[bool, str]:
        running = infusion_running(engine, *drug_keys)
        if infusion_set_this_step(engine, *drug_keys) and running:
            return True, ""
        if running:
            return False, "Set the infusion rate or TCI target for this objective"
        return False, fail_message
    return check


def require_infusion_running(drug_key: str, fail_message: str) -> Callable:
    """Check that an infusion is running, whatever started it."""
    def check(engine) -> Tuple[bool, str]:
        running = infusion_running(engine, drug_key)
        return running, "" if running else fail_message
    return check


def require_infusions_stopped(*drug_keys: str, fail_message: str) -> Callable:
    """Require each named infusion to be stopped during this objective."""
    def check(engine) -> Tuple[bool, str]:
        missing = []
        for drug_key in drug_keys:
            stopped = not infusion_running(engine, drug_key)
            records = engine.actions.since_step(
                ACTION_INFUSION_RATE,
                ACTION_TCI_TARGET,
                labels=(drug_key,),
            )
            stopped_this_step = any(record.amount <= 0 for record in records)
            if not (stopped and stopped_this_step):
                missing.append(drug_key)
        if not missing:
            return True, ""
        return False, fail_message
    return check


def require_drug_bolus(drug_key: str, fail_message: str) -> Callable:
    """Check that a drug bolus was given during this objective."""
    def check(engine) -> Tuple[bool, str]:
        met = bolus_given_this_step(engine, drug_key)
        return (True, "") if met else (False, fail_message)
    return check


def require_crisis_started(event_name: str, attr: str, fail_message: str) -> Callable:
    """Require a crisis to be started during this objective and remain active."""
    def check(engine) -> Tuple[bool, str]:
        started = action_taken_this_step(engine, ACTION_EVENT_START, event_name)
        if started and getattr(engine, attr):
            return True, ""
        return False, fail_message
    return check


def require_crisis_stopped(event_name: str, attr: str, fail_message: str) -> Callable:
    """Require a crisis to be stopped during this objective and remain inactive."""
    def check(engine) -> Tuple[bool, str]:
        stopped = action_taken_this_step(engine, ACTION_EVENT_STOP, event_name)
        if stopped and not getattr(engine, attr):
            return True, ""
        return False, fail_message
    return check


def require_crisis_stopped_with_map(
    event_name: str,
    attr: str,
    map_threshold: float = 65,
    fail_crisis: str = "Stop crisis event",
) -> Callable:
    """Require crisis stop action plus the current MAP target."""
    stopped_check = require_crisis_stopped(event_name, attr, fail_crisis)

    def check(engine) -> Tuple[bool, str]:
        stopped, stop_message = stopped_check(engine)
        map_val = monitor_value(engine, "map")
        map_met = map_val > map_threshold
        if stopped and map_met:
            return True, ""
        messages = []
        if not map_met:
            messages.append(f"MAP: {map_val:.0f}/{map_threshold:.0f}+")
        if not stopped:
            messages.append(stop_message)
        return False, join_messages(messages)
    return check


def require_crisis_resolved_with_map(
    attr: str, map_threshold: float = 65, fail_crisis: str = "Stop crisis event",
) -> Callable:
    """Check that a crisis flag is False and MAP exceeds a threshold."""
    def check(engine) -> Tuple[bool, str]:
        map_val = monitor_value(engine, "map")
        resolved = not getattr(engine, attr)
        if resolved and map_val > map_threshold:
            return True, ""
        msgs = []
        if map_val <= map_threshold:
            msgs.append(f"MAP: {map_val:.0f}/{map_threshold:.0f}+")
        if not resolved:
            msgs.append(fail_crisis)
        return False, join_messages(msgs)
    return check


def require_bis_below(threshold: float) -> Callable:
    """Check BIS below threshold."""
    def check(engine) -> Tuple[bool, str]:
        bis = monitor_value(engine, "bis")
        met = bis < threshold
        return met, "" if met else f"BIS: {bis:.0f}/<{threshold:.0f}"
    return check


def require_bag_mask_started() -> Callable:
    """Require bag-mask ventilation to be started during this objective."""
    def check(engine) -> Tuple[bool, str]:
        started = action_taken_this_step(engine, ACTION_BAG_MASK, "on")
        met = started and engine.bag_mask_active
        return met, "" if met else "Turn ON Bag-Mask ventilation"
    return check


def require_rocuronium_cp(threshold: float = 0.5) -> Callable:
    """Check rocuronium plasma concentration."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.roc_cp > threshold
        return met, "" if met else f"Rocuronium Cp: {engine.state.roc_cp:.2f}/{threshold}+ µg/mL"
    return check


def require_tof_below(threshold: float = 25) -> Callable:
    """Check TOF below threshold."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.tof <= threshold
        return met, "" if met else f"TOF: {engine.state.tof:.0f}%/≤{threshold:.0f}%"
    return check


def require_etco2_above(threshold: float = 20) -> Callable:
    """Check EtCO2 above threshold."""
    def check(engine) -> Tuple[bool, str]:
        etco2 = monitor_value(engine, "etco2")
        met = etco2 > threshold
        return met, "" if met else f"EtCO₂: {etco2:.0f}/>{threshold:.0f} mmHg"
    return check


def require_mac_above(threshold: float = 0.5) -> Callable:
    """Check MAC above threshold."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.mac > threshold
        return met, "" if met else f"MAC: {engine.state.mac:.2f}/{threshold}+"
    return check


def require_vaporizer_started() -> Callable:
    """Require the vaporizer to be turned on during this objective."""
    def check(engine) -> Tuple[bool, str]:
        records = engine.actions.since_step(ACTION_VAPORIZER)
        started = any(record.amount > 0 for record in records)
        running = engine.circuit.vaporizer_on and engine.circuit.vaporizer_setting > 0
        return (True, "") if started and running else (False, "Turn on vaporizer")
    return check


def require_fgf_reduced(max_total_l_min: float) -> Callable:
    """Require a fresh gas flow reduction during this objective."""
    def check(engine) -> Tuple[bool, str]:
        changed = action_taken_this_step(engine, ACTION_FGF)
        total = engine.circuit.fgf_total()
        met = changed and total <= max_total_l_min
        return met, "" if met else f"Reduce FGF to {max_total_l_min:g} L/min"
    return check


def require_oxygen_supply_action(connected: bool, fail_message: str) -> Callable:
    """Require an oxygen supply action and the corresponding current state."""
    label = "connected" if connected else "disconnected"

    def check(engine) -> Tuple[bool, str]:
        acted = action_taken_this_step(engine, ACTION_OXYGEN_SUPPLY, label)
        met = acted and engine.circuit.oxygen_supply_connected == connected
        return met, "" if met else fail_message
    return check


def require_all(*checks) -> Callable:
    """Combine multiple requirement checks (all must pass)."""
    def combined(engine) -> Tuple[bool, str]:
        all_msgs = []
        all_met = True
        for check in checks:
            met, msg = check(engine)
            if not met:
                all_met = False
                if msg:
                    all_msgs.append(msg)
        if all_met:
            return True, ""
        return False, join_messages(all_msgs)
    return combined


def create_observe_baseline_step(
    crisis_name: str,
    hr_range: str = "60-80 bpm"
) -> ScenarioStep:
    """Create a standard baseline observation step."""
    return ScenarioStep(
        id="OBSERVE_BASELINE",
        title="Observe baseline",
        instruction=(
            f"Before the {crisis_name} begins, note the patient's baseline vitals:<br>"
            f"• Heart rate: {hr_range}<br>"
            "• MAP: > 65 mmHg<br>"
            "• SpO₂: > 94%<br><br>"
            "<i>Recognition of abnormal values requires knowing normal baseline.</i>"
        ),
        check_requirements=require_stable_baseline_vitals(),
    )


def create_reassess_step(
    crisis_attr: str,
    fail_crisis_msg: str,
    controlled_text: str,
    extra_text: str,
) -> ScenarioStep:
    """Create a standard reassessment step."""
    return ScenarioStep(
        id="REASSESS",
        title="Reassess hemodynamics",
        instruction=(
            "Confirm stabilization:<br>"
            "• <b>MAP > 65 mmHg</b> (primary resuscitation goal)<br>"
            f"• {controlled_text}<br><br>"
            f"{extra_text}"
        ),
        check_requirements=require_crisis_resolved_with_map(crisis_attr, fail_crisis=fail_crisis_msg),
    )
