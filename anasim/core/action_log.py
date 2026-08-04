"""
Timestamped log of learner control actions and scenario step transitions.

Scenario objectives must distinguish "the learner did this now" from "this was
already true", so control actions are stored in order with the simulation time
at which they were taken.

Step scoping uses record positions rather than timestamps. A paused simulation
holds `state.time` constant, so timestamps alone cannot separate actions taken
before an objective from actions taken while it is active. Step-scoped queries
require an active objective so missing activation cannot silently fall back to
cumulative behavior.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# Action names recorded by the engine.
ACTION_AIRWAY = "airway"
ACTION_BAG_MASK = "bag_mask"
ACTION_DRUG_BOLUS = "drug_bolus"
ACTION_EVENT_START = "event_start"
ACTION_EVENT_STOP = "event_stop"
ACTION_FGF = "fgf"
ACTION_FLUID = "fluid"
ACTION_INFUSION_RATE = "infusion_rate"
ACTION_OXYGEN_SUPPLY = "oxygen_supply"
ACTION_SCENARIO_STEP = "scenario_step"
ACTION_TCI_TARGET = "tci_target"
ACTION_VAPORIZER = "vaporizer"


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One control action and the simulation time at which it was taken."""

    time: float
    action: str
    label: str = ""
    amount: float = 0.0


class ActionLog:
    """Ordered control actions with queries scoped to the active objective."""

    def __init__(self) -> None:
        self._records: List[ActionRecord] = []
        self._step_start_index: Optional[int] = None

    @property
    def records(self) -> Tuple[ActionRecord, ...]:
        """Return every recorded action in order."""
        return tuple(self._records)

    @property
    def current_step(self) -> Optional[ActionRecord]:
        """Return the record that activated the objective, None outside a scenario."""
        if self._step_start_index is None:
            return None
        return self._records[self._step_start_index - 1]

    def record(
        self,
        time: float,
        action: str,
        label: str = "",
        amount: float = 0.0,
    ) -> None:
        """Append one control action to the log."""
        self._records.append(ActionRecord(float(time), action, label, float(amount)))

    def begin_step(self, step_id: str, time: float) -> None:
        """Activate a scenario objective; later actions are scoped to it."""
        self.record(time, ACTION_SCENARIO_STEP, label=step_id)
        self._step_start_index = len(self._records)

    def since_step(
        self,
        *actions: str,
        labels: Optional[Iterable[str]] = None,
    ) -> Tuple[ActionRecord, ...]:
        """Return matching actions taken since the active objective began."""
        if self._step_start_index is None:
            raise RuntimeError("No scenario objective is active")
        return tuple(
            record
            for record in self._records[self._step_start_index:]
            if record.action in actions
            and (labels is None or record.label in labels)
        )

    def total_since_step(
        self,
        *actions: str,
        labels: Optional[Iterable[str]] = None,
    ) -> float:
        """Return the summed amount of matching actions in the active objective."""
        return sum(record.amount for record in self.since_step(*actions, labels=labels))
