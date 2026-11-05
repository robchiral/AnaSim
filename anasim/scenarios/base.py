"""Case definitions and the requirement helpers for guided steps."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

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
from anasim.core.state import AirwayType

if TYPE_CHECKING:
    from anasim.core.engine import SimulationEngine

ControlTab = Literal["Machine", "Medications", "Fluids"]
Requirement = Callable[["SimulationEngine"], tuple[bool, str]]
Condition = Callable[["SimulationEngine"], bool]

MONITOR_BP_INDEX = {"sbp": 0, "dbp": 1, "map": 2}
MONITOR_DISPLAY_FIELD = {
    "hr": "display_hr",
    "bis": "display_bis",
    "etco2": "display_etco2",
    "spo2": "display_spo2",
}


@dataclass
class ScenarioStep:
    id: str
    title: str
    instruction: str
    check_requirements: Requirement
    target_tab: ControlTab | None = None
    # Runs once, the first time the requirements are met.
    on_met: Callable[["SimulationEngine"], None] | None = None
    # Steps with dosing guidance can provide a variant for sessions allowing TCI.
    tci_instruction: str | None = None

    def instruction_for(self, engine) -> str:
        if engine.config.tci_enabled and self.tci_instruction is not None:
            return self.tci_instruction
        return self.instruction


@dataclass(frozen=True)
class CaseOutcome:
    id: str
    label: str
    check: Condition
    review: str  # Debrief point and hint when the outcome is missing
    milestone: bool = False  # Stays credited once achieved
    on_action: str | None = None  # Checked only when this action occurs


@dataclass(frozen=True)
class CaseEvent:
    id: str
    at: float  # Seconds of simulated time after the case starts
    cue: str
    apply: Callable[["SimulationEngine"], None]


@dataclass(frozen=True)
class CaseAction:
    id: str
    label: str
    apply: Callable[["SimulationEngine"], None]
    available: Condition
    completed_label: str


@dataclass
class Scenario:
    id: str
    name: str
    steps: list[ScenarioStep] = field(default_factory=list)
    setup_engine: Callable[["SimulationEngine"], None] | None = None
    briefing: str = ""
    outcomes: tuple[CaseOutcome, ...] = ()
    events: tuple[CaseEvent, ...] = ()
    actions: tuple[CaseAction, ...] = ()

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[ScenarioStep]:
        return iter(self.steps)

    def __getitem__(self, idx: int) -> ScenarioStep:
        return self.steps[idx]

    def prepare(self, engine) -> None:
        """Apply scenario-specific starting conditions."""
        if self.setup_engine is not None:
            self.setup_engine(engine)


def action_taken_in_case(engine, action: str, *labels: str) -> bool:
    """Return whether a matching learner action occurred during this case."""
    records = engine.actions.since_case(action, labels=labels or None)
    return bool(records)


def require_airway_selected(airway_type: str) -> Requirement:
    """Require selection of an airway type during this case."""
    target, label = {
        "None": (AirwayType.NONE, "Disconnected"),
        "Mask": (AirwayType.MASK, "Facemask"),
        "ETT": (AirwayType.ETT, "Tracheal tube"),
    }[airway_type]

    def check(engine) -> tuple[bool, str]:
        selected = action_taken_in_case(engine, ACTION_AIRWAY, target.value)
        met = selected and engine.state.airway_mode == target
        return met, "" if met else f"Select {label}"
    return check


def _fgf_preox_state(engine) -> tuple[bool, str]:
    """Return whether current fresh gas flow is adequate for preoxygenation."""
    delivered_o2 = engine.circuit.delivered_o2_flow()
    o2_ok = delivered_o2 >= 8.0
    air_ok = engine.circuit.fgf_air < 1.0
    n2o_ok = engine.circuit.fgf_n2o <= 0.1
    if o2_ok and air_ok and n2o_ok:
        return True, ""
    msgs = []
    if not o2_ok:
        msgs.append(f"Delivered O₂: {round(delivered_o2, 1):g}/8+ L/min")
    if not air_ok:
        msgs.append(f"Air: {round(engine.circuit.fgf_air, 1):g}/0 L/min")
    if not n2o_ok:
        msgs.append(f"N₂O: {round(engine.circuit.fgf_n2o, 1):g}/0 L/min")
    return False, join_messages(msgs)


def require_oxygen_flow(min_o2: float = 8.0) -> Requirement:
    """Require supplied oxygen flow with air and N₂O off."""
    def check(engine) -> tuple[bool, str]:
        circuit = engine.circuit
        delivered_o2 = circuit.delivered_o2_flow()
        messages = []
        if not delivered_o2 >= min_o2:
            messages.append(f"O₂ flow: {round(delivered_o2, 1):g}/{min_o2:g}+ L/min")
        if not circuit.fgf_air <= 0.1:
            messages.append("Set air to 0 L/min")
        if not circuit.fgf_n2o <= 0.1:
            messages.append("Set N₂O to 0 L/min")
        return not messages, join_messages(messages)
    return check


def require_fgf_set_for_preox(engine) -> tuple[bool, str]:
    """Require a fresh gas flow action and adequate current preoxygenation flow."""
    state_met, message = _fgf_preox_state(engine)
    action_met = action_taken_in_case(engine, ACTION_FGF)
    if not state_met:
        return False, message
    return action_met, "" if action_met else "Set fresh gas flow for this case"


def require_preoxygenation(engine) -> tuple[bool, str]:
    """Require oxygen by mask until the gas monitor shows end-tidal O₂ ≥ 90%."""
    flow_ok, message = _fgf_preox_state(engine)
    if not flow_ok:
        return False, message
    if engine.state.airway_mode != AirwayType.MASK:
        return False, "Apply the facemask"
    ready = engine.state.etco2_signal_valid and engine.state.et_o2 >= 90.0
    return ready, "" if ready else f"Continue preoxygenation (EtO₂ {engine.state.et_o2:.0f}%/90%+)"


def monitor_value(engine, attr: str):
    """Return the learner-facing monitor value for a vital sign."""
    bp_index = MONITOR_BP_INDEX.get(attr)
    if bp_index is not None:
        pressure = engine.state.monitored_blood_pressure(
            engine.config.arterial_line_enabled
        )
        return pressure[bp_index]
    try:
        return getattr(engine.state, MONITOR_DISPLAY_FIELD[attr])
    except KeyError as exc:
        raise ValueError(f"Unsupported monitor value: {attr!r}") from exc


def join_messages(messages) -> str:
    """Join non-empty requirement messages consistently."""
    return ", ".join(message for message in messages if message)


def _action_records(engine, *actions: str, labels: tuple[str, ...], event_name: str | None = None):
    if event_name is not None:
        return engine.actions.since_event(event_name, *actions, labels=labels)
    return engine.actions.since_case(*actions, labels=labels)


def bolus_given_in_case(engine, drug_key: str, *, event_name: str | None = None) -> bool:
    """Check a bolus since the case or the named crisis began."""
    records = _action_records(engine, ACTION_DRUG_BOLUS, labels=(drug_key,), event_name=event_name)
    return any(record.amount > 0 for record in records)


def infusion_set_in_case(engine, *drug_keys: str, event_name: str | None = None) -> bool:
    """Check an infusion action since the case or the named crisis began."""
    records = _action_records(
        engine, ACTION_INFUSION_RATE, ACTION_TCI_TARGET, labels=drug_keys, event_name=event_name,
    )
    return any(record.amount > 0 for record in records)


def infusion_running(engine, *drug_keys: str) -> bool:
    """Return True when a drug is running by manual rate or TCI target."""
    states = (engine.get_drug_state(key) for key in drug_keys)
    return any(state["rate"] > 0 or state["target"] > 0 for state in states)


def require_fluid_bolus(
    min_ml: float,
    labels: tuple[str, ...],
    fail_prefix: str = "Give fluid bolus",
    event_name: str | None = None,
) -> Requirement:
    """Check requested bolus volume; delivery continues at the pump rate."""
    def check(engine) -> tuple[bool, str]:
        volume_requested = sum(record.amount for record in _action_records(
            engine, ACTION_FLUID, labels=labels, event_name=event_name,
        ))
        if volume_requested >= min_ml:
            return True, ""
        if volume_requested > 0.0:
            return False, f"{fail_prefix} ({volume_requested:.0f}/{min_ml:.0f} mL requested)"
        return False, f"{fail_prefix} ({min_ml:.0f} mL via Fluids tab)"
    return check


def require_infusion_started(*drug_keys: str, fail_message: str, event_name: str | None = None) -> Requirement:
    """Require a running infusion set during this case or crisis."""
    def check(engine) -> tuple[bool, str]:
        if any(
            infusion_set_in_case(engine, key, event_name=event_name) and infusion_running(engine, key)
            for key in drug_keys
        ):
            return True, ""
        if infusion_running(engine, *drug_keys):
            control = "infusion rate or TCI target" if engine.config.tci_enabled else "infusion rate"
            return False, f"Set the {control} for this case"
        return False, fail_message
    return check


def require_infusion_running(drug_key: str, fail_message: str) -> Requirement:
    """Check that an infusion is running, whatever started it."""
    def check(engine) -> tuple[bool, str]:
        running = infusion_running(engine, drug_key)
        return running, "" if running else fail_message
    return check


def require_vasopressor_if_hypotensive(
    *drug_keys: str, fail_message: str, event_name: str | None = None,
) -> Requirement:
    """Accept recovered pressure, or require pressor treatment for persistent hypotension."""
    infusion_check = require_infusion_started(*drug_keys, fail_message=fail_message, event_name=event_name)

    def check(engine) -> tuple[bool, str]:
        if monitor_value(engine, "map") >= 65.0:
            return True, ""
        return infusion_check(engine)
    return check


def require_infusions_stopped(*drug_keys: str, fail_message: str) -> Requirement:
    """Require each named infusion to be stopped during this case."""
    def check(engine) -> tuple[bool, str]:
        for drug_key in drug_keys:
            stopped = not infusion_running(engine, drug_key)
            records = engine.actions.since_case(
                ACTION_INFUSION_RATE,
                ACTION_TCI_TARGET,
                labels=(drug_key,),
            )
            stopped_in_case = any(record.amount <= 0 for record in records)
            if not (stopped and stopped_in_case):
                return False, fail_message
        return True, ""
    return check


def require_drug_bolus(drug_key: str, fail_message: str) -> Requirement:
    """Check that a drug bolus was given during this case."""
    def check(engine) -> tuple[bool, str]:
        met = bolus_given_in_case(engine, drug_key)
        return (True, "") if met else (False, fail_message)
    return check


def require_crisis_started(event_name: str, fail_message: str) -> Requirement:
    """Require a crisis start, including one the learner has already controlled."""
    def check(engine) -> tuple[bool, str]:
        started = action_taken_in_case(engine, ACTION_EVENT_START, event_name)
        if started:
            return True, ""
        return False, fail_message
    return check


class CrisisStartMAP:
    """Monitored MAP when a crisis starts, recorded by the case event."""

    def __init__(self) -> None:
        self.value: float | None = None

    def record(self, engine) -> None:
        self.value = monitor_value(engine, "map")


def require_hypotension_or_treatment(
    start: CrisisStartMAP, event_name: str, prompt: str, pressors: tuple[str, ...] = ("nore",),
) -> Requirement:
    """Accept the observed pressure fall or prompt treatment that may prevent it.

    Maintenance can begin near 65 mmHg, so the absolute threshold alone can be
    met before the crisis has changed anything.
    """
    def check(engine) -> tuple[bool, str]:
        map_val = monitor_value(engine, "map")
        threshold = 65.0 if start.value is None else min(65.0, 0.8 * start.value)
        fluids = engine.actions.since_event(event_name, ACTION_FLUID, labels=("crystalloid", "blood"))
        infusions = engine.actions.since_event(event_name, ACTION_INFUSION_RATE, labels=pressors)
        controlled = engine.actions.since_event(event_name, ACTION_EVENT_STOP, labels=(event_name,))
        treated = any(record.amount > 0 for record in (*fluids, *infusions)) or bool(controlled)
        met = treated or map_val < threshold
        return met, "" if met else f"MAP: {map_val:.0f}/<{threshold:.0f} mmHg; {prompt}"
    return check


def require_crisis_stopped(event_name: str, fail_message: str) -> Requirement:
    """Accept source control at any point after this crisis began."""
    def check(engine) -> tuple[bool, str]:
        stopped = bool(engine.actions.since_event(event_name, ACTION_EVENT_STOP, labels=(event_name,)))
        if stopped and not engine.event_active(event_name):
            return True, ""
        return False, fail_message
    return check


def require_crisis_resolved_with_map(
    event_name: str, map_threshold: float = 65, fail_crisis: str = "Stop crisis event",
) -> Requirement:
    """Check that a crisis has stopped, MAP meets a threshold, and ventilation and oxygenation are adequate."""
    return require_all(
        require_map_at_least(map_threshold),
        require_oxygenation(),
        require_tracheal_ventilation,
        lambda engine: (not engine.event_active(event_name), fail_crisis),
        lambda engine: (patient_stable(engine), ""),
    )


def require_map_at_least(threshold: float = 65) -> Requirement:
    def check(engine) -> tuple[bool, str]:
        map_val = monitor_value(engine, "map")
        met = map_val >= threshold
        return met, "" if met else f"MAP: {map_val:.0f}/{threshold:.0f}+"
    return check


def require_bis_below(threshold: float) -> Requirement:
    def check(engine) -> tuple[bool, str]:
        bis = monitor_value(engine, "bis")
        met = bis < threshold
        return met, "" if met else f"BIS: {bis:.0f}/<{threshold:.0f}"
    return check


def require_maintenance_depth(engine) -> tuple[bool, str]:
    bis = monitor_value(engine, "bis")
    met = 40 <= bis <= 60
    return met, "" if met else f"BIS: {bis:.0f}/40-60"


def require_oxygenation(min_spo2: float = 94.0) -> Requirement:
    def check(engine) -> tuple[bool, str]:
        if not engine.state.spo2_signal_valid:
            return False, "Awaiting valid SpO₂"
        spo2 = monitor_value(engine, "spo2")
        met = spo2 >= min_spo2
        return met, "" if met else f"SpO₂: {spo2:.0f}%/{min_spo2:.0f}%+"
    return check


def require_bag_mask_started(engine) -> tuple[bool, str]:
    """Require bag-mask ventilation to be started during this case."""
    started = action_taken_in_case(engine, ACTION_BAG_MASK, "on")
    met = started and engine.bag_mask_active
    return met, "" if met else "Turn ON Bag-Mask ventilation"


def neuromuscular_response(engine) -> tuple[str, float]:
    """The more depressed of T1 height (phase I block) and TOF ratio, %."""
    state = engine.state
    return ("T1", state.twitch) if state.twitch < state.tof else ("TOF ratio", state.tof)


def require_tof_count_at_most(threshold: int) -> Requirement:
    """Require a recent measured TOF count during block onset."""
    def check(engine) -> tuple[bool, str]:
        reading = engine.nmb.reading
        if reading is None or reading.mode != "tof" or engine.state.time - reading.time > 60.0:
            return False, "Measure TOF at the adductor pollicis"
        met = reading.count <= threshold
        return met, "" if met else f"TOF count: {reading.count}/{threshold} or fewer"
    return check


def require_etco2_above(threshold: float = 20) -> Requirement:
    def check(engine) -> tuple[bool, str]:
        etco2 = monitor_value(engine, "etco2")
        met = engine.state.etco2_signal_valid and etco2 > threshold
        return met, "" if met else f"EtCO₂: {etco2:.0f}/>{threshold:.0f} mmHg"
    return check


def require_tracheal_ventilation(engine) -> tuple[bool, str]:
    """Require controlled ventilation through the tracheal tube."""
    if engine.state.airway_mode != AirwayType.ETT:
        return False, "Select Tracheal tube"
    return engine.vent.is_on, "" if engine.vent.is_on else "Start the ventilator"


def require_mac_above(threshold: float = 0.5) -> Requirement:
    """Check end-tidal MAC as the gas monitor shows it, to one decimal."""
    def check(engine) -> tuple[bool, str]:
        mac = round(engine.state.et_mac, 1)
        met = mac >= threshold
        return met, "" if met else f"MAC: {mac:.1f}/{threshold}+"
    return check


def require_vaporizer_started(engine) -> tuple[bool, str]:
    """Require the vaporizer to be turned on during this case."""
    records = engine.actions.since_case(ACTION_VAPORIZER)
    started = any(record.amount > 0 for record in records)
    running = engine.circuit.vaporizer_on and engine.circuit.vaporizer_setting > 0
    return (True, "") if started and running else (False, "Turn on vaporizer")


def require_fgf_reduced(max_total_l_min: float) -> Requirement:
    """Require reduced flow with continued oxygen delivery during this case."""
    def check(engine) -> tuple[bool, str]:
        changed = action_taken_in_case(engine, ACTION_FGF)
        total = engine.circuit.fgf_total()
        met = changed and total <= max_total_l_min and engine.circuit.delivered_o2_flow() >= 1.0
        return met, "" if met else f"Set FGF ≤ {max_total_l_min:g} L/min with O₂ ≥ 1 L/min"
    return check


def require_oxygen_supply_action(connected: bool, fail_message: str) -> Requirement:
    """Require an oxygen supply action and the corresponding current state."""
    label = "connected" if connected else "disconnected"

    def check(engine) -> tuple[bool, str]:
        acted = action_taken_in_case(engine, ACTION_OXYGEN_SUPPLY, label)
        met = acted and engine.circuit.oxygen_supply_connected == connected
        return met, "" if met else fail_message
    return check


def require_all(*checks: Requirement) -> Requirement:
    def combined(engine) -> tuple[bool, str]:
        failures = []
        for check in checks:
            met, msg = check(engine)
            if not met:
                failures.append(msg)
        return not failures, join_messages(failures)
    return combined


def create_observe_step(
    label: str,
    action: str = ACTION_EVENT_START,
    instruction: str = "Follow blood pressure, oxygenation, capnography, and airway pressure.",
) -> ScenarioStep:
    """Wait for the case event; the learner cannot start it."""
    return ScenarioStep(
        id="OBSERVE",
        title="Observe the patient",
        instruction=f"{instruction} The case progresses automatically while the simulation runs.",
        check_requirements=lambda engine: (action_taken_in_case(engine, action, label), ""),
    )


def create_reassess_step(
    event_name: str,
    fail_crisis_msg: str,
    controlled_text: str,
    extra_text: str,
) -> ScenarioStep:
    return ScenarioStep(
        id="REASSESS",
        title="Reassess hemodynamics",
        instruction=(
            f"Confirm {controlled_text.lower()} and <b>MAP ≥ 65 mmHg</b>. "
            "Reassess HR, oxygenation, and ventilation as treatment takes effect.<br><br>"
            f"{extra_text}"
        ),
        check_requirements=require_crisis_resolved_with_map(event_name, fail_crisis=fail_crisis_msg),
    )


def patient_stable(engine) -> bool:
    # Model values, so deterioration between cuff readings counts.
    return engine.state.sao2 >= 94 and engine.state.map >= 65


def tracheal_ventilation_confirmed(engine) -> bool:
    return (
        engine.state.airway_mode == AirwayType.ETT
        and (engine.vent.is_on or engine.bag_mask_active)
        and engine.state.etco2_signal_valid
        and monitor_value(engine, "etco2") > 20
    )


def stability_outcome(review: str) -> CaseOutcome:
    return CaseOutcome("stability", "Oxygenation and circulation stable", patient_stable, review)


CRISIS_BRIEFING = (
    "Anesthesia is established and surgery is in progress. "
    "Monitor the patient and manage any deterioration."
)
CRISIS_ONSET_S = 30.0


def crisis_outcomes(event_name: str, volume: Requirement) -> tuple[CaseOutcome, ...]:
    stopped = require_crisis_stopped(event_name, "")
    return (
        CaseOutcome("volume", "Volume replacement started", lambda engine: volume(engine)[0],
                    "Replace circulating volume while treating the cause of shock.", milestone=True),
        CaseOutcome("source", "Cause addressed", lambda engine: stopped(engine)[0],
                    "Address the cause of deterioration alongside resuscitation."),
        CaseOutcome("ventilation", "Tracheal ventilation maintained", tracheal_ventilation_confirmed,
                    "Maintain ventilation and assess the airway during circulatory deterioration."),
        stability_outcome("Reassess the response to treatment and continue support until stable."),
    )
