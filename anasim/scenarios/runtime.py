"""Case events, outcome tracking, hints, and debriefs for both practice modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anasim.core.action_log import (
    ACTION_ACT,
    ACTION_AIRWAY,
    ACTION_BAG_MASK,
    ACTION_BLOOD_GAS,
    ACTION_CASE_ACTION,
    ACTION_COMPRESSIONS,
    ACTION_COOLING,
    ACTION_DRUG_BOLUS,
    ACTION_EXTRACORPOREAL,
    ACTION_FGF,
    ACTION_FLUID,
    ACTION_INFUSION_RATE,
    ACTION_INTUBATION_ATTEMPT,
    ACTION_LIPID_INFUSION,
    ACTION_O2_FLUSH,
    ACTION_POSITION,
    ACTION_SHOCK,
    ACTION_TCI_TARGET,
    ACTION_VAPORIZER,
    ACTION_VENTILATION,
    ActionRecord,
)
from anasim.core.drug_registry import get_drug_spec
from anasim.core.state import AirwayType

from .base import CaseOutcome, Scenario

if TYPE_CHECKING:
    from anasim.core.clinical_actions import ActionFeedback
    from anasim.core.engine import SimulationEngine

PracticeMode = Literal["guided", "independent"]

SAFETY_REVIEW = {
    "awake_intubation": "Establish adequate anesthesia before attempting intubation.",
    "laryngospasm_blocked": "Treat severe laryngospasm before attempting tracheal intubation.",
    "intubation_awareness": "Neuromuscular block does not establish hypnosis.",
    "awake_neuromuscular_block": "Establish hypnosis before administering a neuromuscular blocker.",
    "unsafe_airway_removal": "Confirm ventilation, neuromuscular recovery, and a safe anesthetic plane before removing the tube.",
    "ventilation_stopped": "Maintain assisted ventilation while the patient is apneic.",
    "shock_while_responsive": "Give sedation or anesthesia before cardioversion in a responsive patient.",
    "succinylcholine_in_mh": "Avoid succinylcholine in malignant hyperthermia.",
    "succinylcholine_in_hyperkalemia": "Avoid succinylcholine when serum potassium is elevated.",
    "protamine_on_bypass": "Avoid administering protamine while cardiopulmonary bypass or ECMO is actively running.",
    "cpb_low_act": "Verify therapeutic heparin anticoagulation (ACT ≥ 400–480 s) before initiating cardiopulmonary bypass.",
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
        init_sao2 = engine.state.upper_sao2 if engine.extracorporeal.is_arterial_support else engine.state.sao2
        self.lowest_spo2 = float(init_sao2)
        self.hypotension_s = self.hypoxemia_s = 0.0
        self._next_check = self.started_at
        self.report: dict | None = None
        self.before_step()
        self.observe(force=True)

    @property
    def elapsed(self) -> float:
        return self.engine.state.time - self.started_at

    def before_step(self) -> None:
        if self.report is not None:
            return
        for event in self.scenario.events:
            if (
                event.id not in self.fired
                and self.elapsed >= event.at - 1e-9
                and (event.when is None or event.when(self.engine))
            ):
                self.fired.add(event.id)
                event.apply(self.engine)
                if event.cue:
                    self.observations.append({"time": self.elapsed, "text": event.cue})

    def observe(self, dt: float = 0.0, *, force: bool = False) -> None:
        if self.report is not None:
            return
        state = self.engine.state
        ec = self.engine.extracorporeal
        eff_sao2 = state.upper_sao2 if ec.is_arterial_support else state.sao2
        self.lowest_map = min(self.lowest_map, float(state.map))
        self.lowest_spo2 = min(self.lowest_spo2, float(eff_sao2))
        unperfused = (
            state.map < 65.0
            or not (self.engine.spontaneous_circulation or ((ec.is_va or ec.is_cpb) and ec.state.blood_flow_l_min >= 1.0))
        )
        self.hypotension_s += dt if unperfused else 0.0
        self.hypoxemia_s += dt if eff_sao2 < 90.0 else 0.0
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

    def _needs_hint(self, outcome: CaseOutcome) -> bool:
        if outcome.id in self.achieved_at:
            return False
        if outcome.on_action is None:
            return True
        if outcome.on_action == "extubation" and self.engine.state.airway_mode != AirwayType.ETT:
            return False
        return not outcome.check(self.engine)

    def request_hint(self) -> str:
        self.observe(force=True)
        if any(event.cue for event in self.scenario.events) and not self.observations:
            text = "Watch the monitors as the case progresses."
        else:
            pending = next((o for o in self.scenario.outcomes if self._needs_hint(o)), None)
            text = pending.review if pending is not None else "Reassess the patient, then end the case when ready."
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
            [SAFETY_REVIEW.get(event["code"], event["text"]) for event in self.safety]
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
        if label == "calcium_chloride":
            return f"Calcium chloride {amount:g} mg given"
        spec = get_drug_spec(label)
        if action == ACTION_DRUG_BOLUS:
            return f"{spec.generic_name} {amount:g} {spec.bolus_unit} given"
        delivery = "infusion" if action == ACTION_INFUSION_RATE else "TCI"
        if amount <= 0:
            return f"{spec.generic_name} {delivery} stopped"
        unit = spec.rate_unit if action == ACTION_INFUSION_RATE else spec.tci_unit
        return f"{spec.generic_name} {delivery} {amount:g} {unit}"
    if action == ACTION_FLUID:
        fluid = {"crystalloid": "lactated Ringer's", "saline": "0.9% saline", "colloid": "5% albumin",
                 "blood": "PRBC", "lipid": "20% lipid emulsion"}[label]
        return f"{amount:g} mL {fluid} started"
    if action == ACTION_LIPID_INFUSION:
        return f"20% lipid emulsion infusion {amount:g} mL/min" if amount > 0 else "Lipid emulsion infusion stopped"
    if action == ACTION_BLOOD_GAS:
        site_labels = {
            "right_radial": "right radial",
            "femoral": "femoral",
            "pre_oxy": "pre-oxygenator",
            "post_oxy": "post-oxygenator",
        }
        site = site_labels.get(label)
        return f"Blood gas drawn ({site})" if site else "Arterial blood gas drawn"
    if action == ACTION_ACT:
        return f"ACT drawn ({amount:.0f} s)"
    if action == ACTION_EXTRACORPOREAL:
        labels = {
            "start_vv_ecmo": f"VV ECMO started ({amount:g} L/min)",
            "start_va_ecmo": f"VA ECMO started ({amount:g} L/min)",
            "start_cpb": f"Cardiopulmonary bypass started ({amount:g} L/min)",
            "stop": "Extracorporeal support stopped",
            "flow": f"Pump blood flow {amount:g} L/min",
            "sweep": f"Sweep gas {amount:g} L/min",
            "fio2": f"Sweep FiO₂ {amount * 100.0:.0f}%",
            "fco2": f"Sweep CO₂ {amount * 100.0:g}%",
            "sevo": f"Oxygenator sevoflurane {amount:g}%",
            "heater_cooler": f"Heater-cooler {amount:g} °C" if amount > 0 else "Heater-cooler off",
            "venous_clamp": f"CPB venous line {amount * 100.0:.0f}% open",
            "cross_clamp_on": "Aortic cross-clamp applied",
            "cross_clamp_off": "Aortic cross-clamp released",
            "cardioplegia_on": "Cardioplegia arrest started",
            "cardioplegia_off": "Cardioplegia arrest stopped",
            "lv_vent": f"LV vent {amount:g} L/min",
            "bg_alpha_stat": "Blood-gas strategy: alpha-stat",
            "bg_ph_stat": "Blood-gas strategy: pH-stat",
        }
        if label in labels:
            return labels[label]
        if label.startswith("reservoir_"):
            kind = label.removeprefix("reservoir_")
            return f"Added {amount:g} mL {kind} to CPB reservoir"
        return f"Extracorporeal: {label}"
    if action == ACTION_VAPORIZER:
        return f"{label} {amount:g}%" if amount > 0 else f"{label} off"
    if action == ACTION_INTUBATION_ATTEMPT:
        return "Intubation unsuccessful" if label == "failed" else "Tracheal tube placed"
    if action == ACTION_AIRWAY and label != "ETT":  # Placement appears as the intubation attempt.
        return f"Airway connection: {label}"
    if action == ACTION_BAG_MASK and label == "breath":
        return "Manual breath delivered"
    if action in {ACTION_BAG_MASK, ACTION_VENTILATION}:
        source = "Bag ventilation" if action == ACTION_BAG_MASK else "Ventilator"
        return f"{source} {'started' if label == 'on' else 'stopped'}"
    if action == ACTION_O2_FLUSH:
        return "O₂ flush delivered"
    if action == ACTION_FGF:
        return f"{label.upper()} flow {amount:g} L/min"
    if action == ACTION_COMPRESSIONS:
        return f"Chest compressions {'started' if label == 'on' else 'stopped'}"
    if action == ACTION_COOLING:
        return f"Surface cooling {'started' if label == 'on' else 'stopped'}"
    if action == ACTION_POSITION:
        return f"Position: {label.replace('_', ' ')}"
    if action == ACTION_SHOCK:
        return f"{'Synchronized shock' if label == 'synchronized' else 'Shock'} {amount:g} J delivered"
    return None
