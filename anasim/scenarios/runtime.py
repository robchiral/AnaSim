"""Case events, outcome tracking, hints, and debriefs for both practice modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anasim.core.action_log import (
    ACTION_AIRWAY,
    ACTION_BAG_MASK,
    ACTION_BLOOD_GAS,
    ACTION_CASE_ACTION,
    ACTION_COMPRESSIONS,
    ACTION_DRUG_BOLUS,
    ACTION_FGF,
    ACTION_FLUID,
    ACTION_INFUSION_RATE,
    ACTION_INTUBATION_ATTEMPT,
    ACTION_SHOCK,
    ACTION_TCI_TARGET,
    ACTION_VAPORIZER,
    ACTION_VENTILATION,
    ActionRecord,
)
from anasim.core.drug_registry import get_drug_spec

from .base import Scenario

if TYPE_CHECKING:
    from anasim.core.clinical_actions import ActionFeedback
    from anasim.core.engine import SimulationEngine

PracticeMode = Literal["guided", "independent"]

SAFETY_REVIEW = {
    "awake_intubation": "Establish appropriate intubating conditions before airway instrumentation.",
    "intubation_awareness": "Neuromuscular block does not establish hypnosis.",
    "awake_neuromuscular_block": "Establish hypnosis before administering a neuromuscular blocker.",
    "unsafe_airway_removal": "Confirm ventilation and neuromuscular recovery before removing the tube.",
    "ventilation_stopped": "Maintain assisted ventilation while the patient is apneic.",
    "shock_while_responsive": "Give sedation or anesthesia before cardioversion in a responsive patient.",
}


class CaseRun:
    def __init__(self, scenario: Scenario, engine: SimulationEngine, mode: PracticeMode):
        self.scenario, self.engine, self.mode = scenario, engine, mode
        self.started_at = engine.state.time
        engine.actions.begin_case(scenario.id, self.started_at)
        self.fired: set[str] = set()
        self.used_actions: set[str] = set()
        self.achieved_at: dict[str, float] = {}
        self.observations: list[dict] = []
        self.safety: list[dict] = []
        self.hints: list[dict] = []
        self.hint: str | None = None
        self.lowest_map = float(engine.state.map)
        self.lowest_spo2 = float(engine.state.sao2)
        self.hypotension_s = self.hypoxemia_s = 0.0
        self._next_check = self.started_at
        self.report: dict | None = None
        self.observe(force=True)

    @property
    def elapsed(self) -> float:
        return self.engine.state.time - self.started_at

    def before_step(self) -> None:
        if self.report is not None:
            return
        for event in self.scenario.events:
            if event.id not in self.fired and self.elapsed >= event.at - 1e-9:
                self.fired.add(event.id)
                event.apply(self.engine)
                self.observations.append({"time": self.elapsed, "text": event.cue})

    def observe(self, dt: float = 0.0, *, force: bool = False) -> None:
        if self.report is not None:
            return
        state = self.engine.state
        self.lowest_map = min(self.lowest_map, float(state.map))
        self.lowest_spo2 = min(self.lowest_spo2, float(state.sao2))
        self.hypotension_s += dt if state.map < 65.0 else 0.0
        self.hypoxemia_s += dt if state.sao2 < 90.0 else 0.0
        if not force and state.time < self._next_check:
            return
        self._next_check = state.time + 0.5
        for outcome in self.scenario.outcomes:
            if outcome.on_action is not None:
                continue
            if outcome.milestone and outcome.id in self.achieved_at:
                continue
            if outcome.check(self.engine):
                self.achieved_at.setdefault(outcome.id, self.elapsed)
            else:
                self.achieved_at.pop(outcome.id, None)

    def record_transition(self, action: str) -> None:
        """Evaluate outcomes tied to this action at the moment it occurs."""
        for outcome in self.scenario.outcomes:
            if outcome.on_action == action:
                if outcome.check(self.engine):
                    self.achieved_at[outcome.id] = self.elapsed
                else:
                    self.achieved_at.pop(outcome.id, None)

    def record_safety(self, feedback: ActionFeedback) -> None:
        self.safety.append({"time": self.elapsed, "code": feedback.code, "text": feedback.message})

    def act(self, action_id: str) -> None:
        action = next((a for a in self.scenario.actions if a.id == action_id), None)
        if action is None:
            raise ValueError(f"Unknown case action: {action_id!r}")
        if action_id in self.used_actions or not action.available(self.engine):
            raise ValueError("This case action is not currently available")
        self.engine.actions.record(self.engine.state.time, ACTION_CASE_ACTION, label=action_id)
        action.apply(self.engine)
        self.used_actions.add(action_id)
        self.observe(force=True)

    def request_hint(self) -> str:
        self.observe(force=True)
        if self.scenario.events and not self.fired:
            text = "Observe the monitors and clinical information as the case progresses."
        else:
            pending = next((o for o in self.scenario.outcomes if o.id not in self.achieved_at), None)
            text = pending.review if pending is not None else "Reassess the patient and decide whether the case is ready to end."
        self.hint = text
        self.hints.append({"time": self.elapsed, "text": text})
        return text

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "complete": self.report is not None,
            "briefing": self.scenario.briefing,
            "observations": self.observations,
            "actions": self._listed_actions(),
            "hint": self.hint,
            "debrief": self.report,
        }

    def _listed_actions(self) -> list[dict]:
        """Show available actions, and keep used ones as disabled confirmations."""
        listed = []
        for action in self.scenario.actions:
            used = action.id in self.used_actions
            if used or action.available(self.engine):
                listed.append({"id": action.id, "label": action.completed_label if used else action.label,
                               "enabled": not used})
        return listed

    def finish(self, reason: str) -> dict:
        if self.report is not None:
            return self.report
        self.observe(force=True)
        outcomes = [
            {"id": o.id, "label": o.label, "achieved": o.id in self.achieved_at,
             "time": self.achieved_at.get(o.id)} for o in self.scenario.outcomes
        ]
        points = list(dict.fromkeys(
            [SAFETY_REVIEW[event["code"]] for event in self.safety]
            + [o.review for o in self.scenario.outcomes if o.id not in self.achieved_at]
        ))[:3]
        self.report = {
            "name": self.scenario.name, "mode": self.mode, "reason": reason,
            "duration": self.elapsed, "hints_used": len(self.hints),
            "outcomes": outcomes, "safety_events": list(self.safety), "learning_points": points,
            "physiology": {
                "lowest_map": round(self.lowest_map), "lowest_spo2": round(self.lowest_spo2),
                "hypotension_s": round(self.hypotension_s), "hypoxemia_s": round(self.hypoxemia_s),
            },
            "timeline": self._timeline(),
        }
        return self.report

    def _timeline(self) -> list[dict]:
        # Cues describe case events; case actions describe the learner's response to them.
        timeline = [{**item, "kind": "observation"} for item in self.observations]
        completed = {action.id: action.completed_label for action in self.scenario.actions}
        for record in self.engine.actions.since_case():
            text = completed[record.label] if record.action == ACTION_CASE_ACTION else _describe(record)
            if text is not None:
                timeline.append({"time": record.time - self.started_at, "text": text, "kind": "action"})
        timeline.extend({**item, "kind": "hint"} for item in self.hints)
        timeline.extend({**item, "kind": "safety"} for item in self.safety)
        return sorted(timeline, key=lambda item: item["time"])


def _describe(record: ActionRecord) -> str | None:
    """Return timeline text for a learner control action."""
    action, label, amount = record.action, record.label, record.amount
    if action in {ACTION_DRUG_BOLUS, ACTION_INFUSION_RATE, ACTION_TCI_TARGET}:
        if label == "sugammadex":
            return f"Sugammadex {amount:.0f} mg given"
        spec = get_drug_spec(label)
        if action == ACTION_DRUG_BOLUS:
            return f"{spec.generic_name} {amount:g} {spec.bolus_unit} given"
        delivery = "infusion" if action == ACTION_INFUSION_RATE else "TCI"
        if amount <= 0:
            return f"{spec.generic_name} {delivery} stopped"
        unit = spec.rate_unit if action == ACTION_INFUSION_RATE else spec.tci_unit
        return f"{spec.generic_name} {delivery} {amount:g} {unit}"
    if action == ACTION_FLUID:
        fluid = {"crystalloid": "lactated Ringer's", "colloid": "albumin", "blood": "PRBC"}[label]
        return f"{amount:g} mL {fluid} started"
    if action == ACTION_BLOOD_GAS:
        return "Arterial blood gas drawn"
    if action == ACTION_VAPORIZER:
        return f"{label} {amount:g}%" if amount > 0 else f"{label} off"
    if action == ACTION_INTUBATION_ATTEMPT:
        return "Intubation unsuccessful" if label == "failed" else "Tracheal tube placed"
    if action == ACTION_AIRWAY and label != "ETT":  # Placement appears as the intubation attempt.
        return f"Airway connection: {label}"
    if action in {ACTION_BAG_MASK, ACTION_VENTILATION}:
        source = "Bag ventilation" if action == ACTION_BAG_MASK else "Ventilator"
        return f"{source} {'started' if label == 'on' else 'stopped'}"
    if action == ACTION_FGF:
        return f"{label.upper()} flow {amount:g} L/min"
    if action == ACTION_COMPRESSIONS:
        return f"Chest compressions {'started' if label == 'on' else 'stopped'}"
    if action == ACTION_SHOCK:
        return f"{'Synchronized shock' if label == 'synchronized' else 'Shock'} {amount:g} J delivered"
    return None
