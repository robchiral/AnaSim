"""Case definitions and the requirement helpers for guided steps."""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from anasim.core import monitors as monitor_core
from anasim.core import runtime as runtime_core
from anasim.core.action_log import (
    ACTION_AIRWAY,
    ACTION_BAG_MASK,
    ACTION_DRUG_BOLUS,
    ACTION_EVENT_START,
    ACTION_EVENT_STOP,
    ACTION_FGF,
    ACTION_FLUID,
    ACTION_INFUSION_RATE,
    ACTION_TCI_TARGET,
    ACTION_VAPORIZER,
    ActionLog,
)
from anasim.core.monitors import display_int
from anasim.core.projection import (
    dantrolene_twitch_fraction,
    seed_arterial_monitors,
    sync_state_from_models,
)
from anasim.core.state import AirwayType, WaveformSample
from anasim.monitors.neuromuscular import NeuromuscularMonitor

if TYPE_CHECKING:
    from anasim.core.engine import SimulationEngine

ControlTab = Literal["Machine", "Medications", "Circulation", "Labs"]
Requirement = Callable[["SimulationEngine"], tuple[bool, str]]
Condition = Callable[["SimulationEngine"], bool]
Instruction = str | Callable[["SimulationEngine"], str]

MONITOR_BP_INDEX = {"sbp": 0, "dbp": 1, "map": 2}
MONITOR_DISPLAY_FIELD = {
    "hr": "display_hr",
    "bis": "display_bis",
    "etco2": "display_etco2",
    "spo2": "display_spo2",
}


_STALE_BP_MESSAGE = "Measure blood pressure again"


@dataclass
class ScenarioStep:
    id: str
    title: str
    instruction: Instruction
    check_requirements: Requirement
    target_tab: ControlTab | None = None
    # Runs once, the first time the requirements are met.
    on_met: Callable[["SimulationEngine"], None] | None = None
    # Steps with dosing guidance can provide a variant for sessions allowing TCI.
    tci_instruction: str | None = None

    def instruction_for(self, engine) -> str:
        if callable(self.instruction):
            return self.instruction(engine)
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
    at: float  # Earliest simulated seconds after the case starts
    cue: str  # Empty string for silent physiological onset events
    apply: Callable[["SimulationEngine"], None]
    when: Condition | None = None  # Optional physiological gate after `at`


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
        "SGA": (AirwayType.SGA, "SGA"),
        "ETT": (AirwayType.ETT, "Tracheal tube"),
        "FONA": (AirwayType.FONA, "Emergency surgical airway (eFONA)"),
    }[airway_type]

    def check(engine) -> tuple[bool, str]:
        selected = action_taken_in_case(engine, ACTION_AIRWAY, target.value)
        met = selected and engine.state.airway_mode == target
        return met, "" if met else f"Select {label}"
    return check


def _fgf_preox_state(engine, *, min_o2: float = 8.0) -> tuple[bool, str]:
    """Return whether current fresh gas flow is adequate for preoxygenation."""
    delivered_o2 = engine.circuit.delivered_o2_flow()
    o2_ok = delivered_o2 >= min_o2
    air_ok = engine.circuit.fgf_air < 1.0
    n2o_ok = engine.circuit.fgf_n2o <= 0.1
    if o2_ok and air_ok and n2o_ok:
        return True, ""
    msgs = []
    if not o2_ok:
        msgs.append(f"Delivered O₂: {round(delivered_o2, 1):g}/{min_o2:g}+ L/min")
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
    return action_met, "" if action_met else "Set fresh gas flow"


def require_preoxygenation(engine) -> tuple[bool, str]:
    """Require oxygen by mask until the gas monitor shows end-tidal O₂ ≥ 90%."""
    flow_ok, message = _fgf_preox_state(engine, min_o2=6.0)
    if not flow_ok:
        return False, message
    if engine.state.airway_mode != AirwayType.MASK:
        return False, "Apply the facemask"
    ready = engine.state.etco2_signal_valid and engine.state.et_o2 >= 90.0
    shown_eto2 = format_unmet_int(engine.state.et_o2, low=90.0)
    return ready, "" if ready else f"Continue preoxygenation (EtO₂ {shown_eto2}%/90%+)"


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


def format_unmet_int(
    value: float,
    *,
    low: float | None = None,
    high: float | None = None,
) -> int:
    """Format an unmet monitor value as an integer that stays outside passing bounds."""
    shown = int(display_int(value))
    if low is not None and (high is None or value <= low):
        shown = min(shown, int(display_int(low)) - 1)
    if high is not None and (low is None or value >= high):
        shown = max(shown, int(display_int(high)) + 1)
    return shown


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
            shown_vol = format_unmet_int(volume_requested, low=min_ml)
            return False, f"{fail_prefix} ({shown_vol}/{min_ml:.0f} mL requested)"
        return False, f"{fail_prefix} ({min_ml:.0f} mL)"
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
            return False, f"Adjust the {control}"
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
    map_check = require_map_at_least(65)
    infusion_check = require_infusion_started(*drug_keys, fail_message=fail_message, event_name=event_name)

    def check(engine) -> tuple[bool, str]:
        map_ok, map_message = map_check(engine)
        if map_ok:
            return True, ""
        infusion_ok, infusion_message = infusion_check(engine)
        if infusion_ok:
            return True, ""
        if map_message == _STALE_BP_MESSAGE:
            return False, f"{infusion_message}, or measure blood pressure again"
        return False, infusion_message
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


class CrisisStartMAP:
    """Monitored MAP when a crisis starts, recorded by the case event."""

    def __init__(self) -> None:
        self.value: float | None = None

    def record(self, engine) -> None:
        val = monitor_value(engine, "map")
        self.value = val if math.isfinite(val) and val > 0.0 else None


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
        fluids = engine.actions.since_event(
            event_name, ACTION_FLUID, labels=("crystalloid", "saline", "colloid", "blood"),
        )
        pressor_actions = engine.actions.since_event(
            event_name, ACTION_INFUSION_RATE, ACTION_TCI_TARGET, ACTION_DRUG_BOLUS, labels=pressors,
        )
        controlled = engine.actions.since_event(event_name, ACTION_EVENT_STOP, labels=(event_name,))
        treated = any(record.amount > 0 for record in (*fluids, *pressor_actions)) or bool(controlled)
        failed_cuff_in_shock = (
            not engine.config.arterial_line_enabled
            and engine.state.nibp_measurement_failed
            and engine.state.map < threshold
        )
        stale_cuff = (
            not engine.config.arterial_line_enabled
            and not engine.state.nibp_measurement_failed
            and (
                engine.state.nibp_timestamp is None
                or engine.state.time - engine.state.nibp_timestamp > 60.0
            )
        )
        met = treated or failed_cuff_in_shock or (
            not stale_cuff and math.isfinite(map_val) and 0.0 < map_val < threshold
        )
        if met:
            return True, ""
        if stale_cuff or not math.isfinite(map_val) or map_val <= 0.0:
            return False, f"{_STALE_BP_MESSAGE}; {prompt}"
        shown_map = max(int(display_int(map_val)), round(threshold))
        return False, f"MAP: {shown_map}/<{threshold:.0f} mmHg; {prompt}"
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
        lambda engine: (patient_stable(engine), "Reassess oxygenation and perfusion"),
    )


def require_map_at_least(threshold: float = 65) -> Requirement:
    def check(engine) -> tuple[bool, str]:
        state = engine.state
        if (
            state.cardiac_arrest
            or state.chest_compressions
            or engine.hemo.compressions
            or not engine.spontaneous_circulation
        ):
            return False, "Confirm spontaneous perfusing rhythm"
        if not engine.config.arterial_line_enabled and (
            state.nibp_measurement_failed
            or state.nibp_timestamp is None
            or state.time - state.nibp_timestamp > 60.0
        ):
            return False, _STALE_BP_MESSAGE
        map_val = monitor_value(engine, "map")
        if map_val < threshold:
            shown_map = format_unmet_int(map_val, low=threshold)
            return False, f"MAP: {shown_map}/{threshold:.0f}+"
        if state.map < threshold:
            return False, "Reassess blood pressure and perfusion"
        return True, ""
    return check


def require_bis_below(threshold: float) -> Requirement:
    def check(engine) -> tuple[bool, str]:
        bis = monitor_value(engine, "bis")
        met = bis < threshold
        shown_bis = max(int(display_int(bis)), round(threshold))
        return met, "" if met else f"BIS: {shown_bis}/<{threshold:.0f}"
    return check


def require_maintenance_depth(engine) -> tuple[bool, str]:
    bis = monitor_value(engine, "bis")
    met = 40 <= bis <= 60
    shown_bis = format_unmet_int(bis, low=40, high=60)
    return met, "" if met else f"BIS: {shown_bis}/40-60"


def require_oxygenation(min_spo2: float = 94.0) -> Requirement:
    def check(engine) -> tuple[bool, str]:
        if not engine.state.spo2_signal_valid:
            return False, "Awaiting valid SpO₂"
        spo2 = monitor_value(engine, "spo2")
        met = spo2 >= min_spo2
        shown_spo2 = format_unmet_int(spo2, low=min_spo2)
        return met, "" if met else f"SpO₂: {shown_spo2}%/{min_spo2:.0f}%+"
    return check


def require_bag_mask_started(engine) -> tuple[bool, str]:
    """Require bag-mask ventilation to be started during this case."""
    if engine.apl_cmh2o == 0.0:
        return False, "Set the APL valve above 0"
    started = action_taken_in_case(engine, ACTION_BAG_MASK, "on")
    met = started and engine.bag_mask_active
    return met, "" if met else "Start bag ventilation"


def neuromuscular_response(engine) -> tuple[str, float]:
    """The more depressed of NMBA T1 height (phase I block) and TOF ratio, %."""
    state = engine.state
    dantrolene_scale = max(0.01, dantrolene_twitch_fraction(state.dantrolene_ce))
    nmba_t1 = min(100.0, state.twitch / dantrolene_scale)
    return ("T1", nmba_t1) if nmba_t1 < state.tof else ("TOF ratio", state.tof)


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
        shown_etco2 = min(int(display_int(etco2)), round(threshold))
        return met, "" if met else f"EtCO₂: {shown_etco2}/>{threshold:.0f} mmHg"
    return check


def require_tracheal_ventilation(engine) -> tuple[bool, str]:
    """Require the ventilator to be running through the tracheal tube."""
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
    return (True, "") if started and running else (False, "Start sevoflurane")


def require_fgf_reduced(max_total_l_min: float) -> Requirement:
    """Require reduced flow with continued oxygen delivery during this case."""
    def check(engine) -> tuple[bool, str]:
        changed = action_taken_in_case(engine, ACTION_FGF)
        total = engine.circuit.fgf_total()
        met = changed and total <= max_total_l_min and engine.circuit.delivered_o2_flow() >= 1.0
        return met, "" if met else f"Set FGF ≤ {max_total_l_min:g} L/min with O₂ ≥ 1 L/min"
    return check


def normocapnia_bounds(engine, *, baseline_low: float = 35.0) -> tuple[int, int]:
    """Expected displayed EtCO2 range (mmHg) adjusted for resting PaCO2 (pregnancy) and COPD dead space."""
    paco2_drop = max(0.0, 40.0 - engine.resp.paco2_setpoint)
    copd_shift = 4.0 * max(0.0, engine.patient.copd_severity)
    low = max(20, round(baseline_low - paco2_drop - copd_shift))
    high = max(low + 8, round(45.0 - paco2_drop))
    return low, high


def check_etco2_range(engine, low: int, high: int) -> tuple[bool, str]:
    """Require a valid displayed EtCO2 within the inclusive [low, high] mmHg bounds."""
    if not engine.state.etco2_signal_valid:
        return False, "Awaiting exhaled CO₂"
    etco2 = monitor_value(engine, "etco2")
    rounded = int(display_int(etco2))
    met = low <= rounded <= high
    shown_etco2 = format_unmet_int(etco2, low=low, high=high)
    return met, "" if met else f"EtCO₂: {shown_etco2}/{low}-{high} mmHg"


def require_all(*checks: Requirement) -> Requirement:
    def combined(engine) -> tuple[bool, str]:
        failures = []
        for check in checks:
            met, msg = check(engine)
            if not met:
                failures.append(msg)
        return not failures, join_messages(failures)
    return combined


def equilibrate_maintenance_fio2(
    engine, *, target_fio2: float, min_air_l_min: float, air_vo2_factor: float,
) -> None:
    """Set air/O2 fresh-gas flows for target_fio2 given patient VO2 and equilibrate circuit and lung."""
    vo2 = max(0.15, engine.resp._pulmonary_o2_uptake)
    fgf_air = max(min_air_l_min, air_vo2_factor * vo2)
    fgf_o2 = vo2 + fgf_air * (target_fio2 - 0.21) / (1.0 - target_fio2)
    engine.set_fgf(round(fgf_o2, 3), round(fgf_air, 3), 0.0)
    circuit = engine.circuit
    circuit.equilibrate(engine.resp._pulmonary_o2_uptake, circuit.composition.fi_agent)
    baseline_fio2 = circuit.composition.fio2
    engine.resp.equilibrate_oxygen(
        baseline_fio2, 1.0, engine.vent.settings.peep, fin2=circuit.composition.fin2,
    )
    sync_state_from_models(engine)


def _preroll_trend_point(
    engine,
    rel_time: float,
    *,
    last_nibp: tuple[float, float] | None,
) -> dict:
    """Format one pre-case trend point matching WebSession._sample_trends()."""
    state = engine.state
    arterial = engine.config.arterial_line_enabled
    if arterial:
        map_val = round(float(state.art_map), 1) if math.isfinite(state.art_map) else None
        map_time: float | None = round(rel_time, 3)
    elif last_nibp is not None:
        map_val = round(last_nibp[0], 1) if math.isfinite(last_nibp[0]) else None
        map_time = round(last_nibp[1], 3)
    else:
        map_val = None
        map_time = None
    etco2 = round(float(state.display_etco2), 1) if (state.etco2_signal_valid and math.isfinite(state.display_etco2)) else None
    hr = round(float(state.display_hr), 1) if math.isfinite(state.display_hr) else None
    bis = round(float(state.display_bis), 1) if math.isfinite(state.display_bis) else None
    return {
        "time": round(rel_time, 3),
        "hr": hr,
        "etco2": etco2,
        "bis": bis,
        "map": map_val,
        "map_time": map_time,
    }


def preroll_scenario(
    engine,
    duration_s: float,
    *,
    dt: float = 2.0,
    settle_s: float = 15.0,
    sample_interval_s: float = 5.0,
    reset_fluid_totals: bool = True,
    preserve_blood_out: bool = False,
    record_trends: bool = True,
    on_step: Callable[["SimulationEngine", float], None] | None = None,
) -> None:
    """Advance an engine through pre-case pathology, seed Trends, and reset case time to 0."""
    if duration_s <= 0.0:
        return
    was_running = engine.running
    engine.running = True
    settle = min(max(0.0, settle_s), duration_s)
    fast_duration = max(0.0, duration_s - settle)
    nibp_interval = max(60.0, float(engine.nibp.interval))

    trends: list[dict] = []
    last_nibp: tuple[float, float] | None = (float(engine.state.map), -duration_s)
    next_nibp_elapsed = nibp_interval
    next_sample_elapsed = 0.0

    def maybe_record(elapsed: float) -> None:
        nonlocal last_nibp, next_nibp_elapsed, next_sample_elapsed
        if not record_trends or elapsed >= duration_s - 1e-6:
            return
        rel_time = elapsed - duration_s
        if elapsed >= next_nibp_elapsed - 1e-6:
            last_nibp = (float(engine.state.map), rel_time)
            next_nibp_elapsed += nibp_interval
        if elapsed >= next_sample_elapsed - 1e-6:
            trends.append(_preroll_trend_point(engine, rel_time, last_nibp=last_nibp))
            next_sample_elapsed += sample_interval_s

    elapsed = 0.0
    maybe_record(0.0)
    while elapsed < fast_duration - 1e-9:
        step_dt = min(dt, fast_duration - elapsed)
        runtime_core.step_simulation(engine, step_dt, update_monitors=False)
        engine.state.time = round(engine.state.time + step_dt, 9)
        elapsed += step_dt
        if on_step is not None:
            on_step(engine, elapsed)
        maybe_record(elapsed)

    # Re-seed beat-synchronous monitors to the pre-rolled hemodynamics before the
    # short 100 Hz settle that latches capnography, arterial line, SpO2, and alarms.
    seed_arterial_monitors(engine, engine.hemo.state)
    engine.alarms._violation_s.clear()
    engine.state.alarms = {}

    while elapsed < duration_s - 1e-9:
        step_dt = min(1.0, duration_s - elapsed)
        engine.step(step_dt)
        elapsed += step_dt
        if on_step is not None:
            on_step(engine, elapsed)
        maybe_record(elapsed)

    # Reset visible clock, ephemeral monitor buffers, and setup actions back to t = 0.
    state = engine.state
    state.time = 0.0
    engine._next_nibp_time = 0.0
    engine.nibp.reset()
    monitor_core.seed_nibp_reading(engine)
    nmb_baseline = engine.nmb.baseline
    engine.nmb = NeuromuscularMonitor()
    engine.nmb.baseline = nmb_baseline
    for key in engine.tci:
        engine._recompute_tci(key, 0.0)
    engine._tci_accumulators.clear()
    spec = engine.disturbances.spec
    if engine.disturbance_active and spec is not None and not spec.finite:
        engine.disturbance_start_time = -spec.established_s
    engine.intubation_start_time = None
    engine._extubation_stimulus_until = 0.0
    engine._last_succinylcholine_time = None
    engine.compressions_started_at = None
    engine._pulseless_s = 0.0
    engine.vent.breath_count = 0
    engine.blood_gases.clear()
    engine.acts.clear()
    engine.glucoses.clear()
    engine.coagulation.cumulative_protamine_mg = 0.0
    if reset_fluid_totals:
        engine.pending_infusions.clear()
        hemo = engine.hemo
        hemo._hyperosmolar_load = 0.0
        hemo._hyperosmolar_svr_factor = 1.0
        hemo.total_crystalloid_in_ml = 0.0
        hemo.total_colloid_in_ml = 0.0
        hemo.total_blood_in_ml = 0.0
        hemo.total_urine_out_ml = 0.0
        if not preserve_blood_out:
            hemo.total_blood_out_ml = 0.0
            engine._pk_blood_out_ml = 0.0
        state.fluid_in_ml = 0.0
        state.colloid_in_ml = 0.0
        state.blood_in_ml = 0.0
        state.urine_out_ml = 0.0
        state.blood_out_ml = float(hemo.total_blood_out_ml)
        state.net_fluid_ml = -state.blood_out_ml
    transport = engine.resp.gas_transport
    transport.consumed_o2_l = 0.0
    transport.produced_l = 0.0
    transport.exhaled_l = 0.0
    transport.oxygen_from_air_l = 0.0
    engine.output_buffer.clear()
    engine.output_buffer.append(WaveformSample(
        0.0, state.ecg_voltage, state.pleth_voltage, state.capno_co2, state.art_pressure,
        state.paw, state.flow, state.volume, 0,
    ))
    engine.actions = ActionLog()
    engine.running = was_running
    if record_trends:
        engine._preroll_trends = trends


def create_observe_step(
    label: str,
    action: str = ACTION_EVENT_START,
    instruction: Instruction = (
        "Follow blood pressure, oxygenation, capnography, and airway pressure."
    ),
    *,
    check: Requirement | None = None,
) -> ScenarioStep:
    """Wait for the case event; the learner cannot start it."""
    suffix = " The case progresses automatically while the simulation runs."
    full_instruction: Instruction = (
        (lambda engine: f"{instruction(engine)}{suffix}")
        if callable(instruction)
        else f"{instruction}{suffix}"
    )
    return ScenarioStep(
        id="OBSERVE",
        title="Observe the patient",
        instruction=full_instruction,
        check_requirements=check or (lambda engine: (action_taken_in_case(engine, action, label), "")),
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
    state = engine.state
    return (
        not state.cardiac_arrest
        and not state.chest_compressions
        and not engine.hemo.compressions
        and engine.spontaneous_circulation
        and state.sao2 >= 94
        and state.map >= 65
    )


def tracheal_ventilation_confirmed(engine) -> bool:
    return (
        engine.state.airway_mode in (AirwayType.ETT, AirwayType.FONA)
        and not engine.airway_crisis.esophageal_intubation
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
