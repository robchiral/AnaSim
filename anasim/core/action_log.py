"""Ordered log of learner actions within a case.

Requirements and outcomes count actions from the case start, or from the
latest crisis onset. Record positions define both scopes because paused
actions share a timestamp.
"""

from collections.abc import Iterable
from dataclasses import dataclass

ACTION_AIRWAY = "airway"
ACTION_BAG_MASK = "bag_mask"
ACTION_BLOOD_GAS = "blood_gas"
ACTION_CASE_ACTION = "case_action"
ACTION_CASE_START = "case_start"
ACTION_COMPRESSIONS = "compressions"
ACTION_COOLING = "cooling"
ACTION_DRUG_BOLUS = "drug_bolus"
ACTION_EVENT_START = "event_start"
ACTION_EVENT_STOP = "event_stop"
ACTION_FGF = "fgf"
ACTION_FLUID = "fluid"
ACTION_LIPID_INFUSION = "lipid_infusion"
ACTION_INFUSION_RATE = "infusion_rate"
ACTION_INTUBATION_ATTEMPT = "intubation_attempt"
ACTION_OXYGEN_SUPPLY = "oxygen_supply"
ACTION_POSITION = "position"
ACTION_SHOCK = "shock"
ACTION_TCI_TARGET = "tci_target"
ACTION_VAPORIZER = "vaporizer"
ACTION_VENTILATION = "ventilation"


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One control action and the simulation time at which it was taken."""

    time: float
    action: str
    label: str = ""
    amount: float = 0.0


class ActionLog:
    """Ordered control actions with queries scoped to the case or a crisis."""

    def __init__(self) -> None:
        self._records: list[ActionRecord] = []
        self._case_start_index = 0

    @property
    def records(self) -> tuple[ActionRecord, ...]:
        return tuple(self._records)

    def record(
        self,
        time: float,
        action: str,
        label: str = "",
        amount: float = 0.0,
    ) -> None:
        self._records.append(ActionRecord(float(time), action, label, float(amount)))

    def total(self, action: str, label: str) -> float:
        """Sum the amounts of all matching actions, including setup."""
        return sum(r.amount for r in self._records if r.action == action and r.label == label)

    def begin_case(self, case_id: str, time: float) -> None:
        """Exclude setup actions from later case queries."""
        self.record(time, ACTION_CASE_START, label=case_id)
        self._case_start_index = len(self._records)

    def since_case(self, *actions: str, labels: Iterable[str] | None = None) -> tuple[ActionRecord, ...]:
        """Return matching actions since the case began; no actions means all."""
        return tuple(
            record for record in self._records[self._case_start_index:]
            if (not actions or record.action in actions) and (labels is None or record.label in labels)
        )

    def since_event(
        self,
        event_name: str,
        *actions: str,
        labels: Iterable[str] | None = None,
    ) -> tuple[ActionRecord, ...]:
        """Return actions after the latest crisis start in this case."""
        for index in range(len(self._records) - 1, self._case_start_index - 1, -1):
            record = self._records[index]
            if record.action == ACTION_EVENT_START and record.label == event_name:
                return tuple(
                    item for item in self._records[index + 1:]
                    if item.action in actions and (labels is None or item.label in labels)
                )
        return ()
