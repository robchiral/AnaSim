"""Hazard checks for learner actions; only an awake, reactive intubation attempt fails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from anasim.patient.pd import hypnotic_equivalent
from anasim.patient.pd.anesthesia import PROPOFOL_LOC_TRANSFER

from .projection import (
    central_muscle_strength,
    loss_of_response,
    stimulus_tolerance,
    sync_inhaled_agents,
    sync_pk_state,
)

if TYPE_CHECKING:
    from .engine import SimulationEngine


@dataclass(frozen=True, slots=True)
class ActionFeedback:
    code: str
    message: str
    blocked: bool = False


def _responsiveness(engine: SimulationEngine) -> tuple[float, float]:
    # Commands can arrive while paused or immediately after another drug action.
    sync_pk_state(engine)
    sync_inhaled_agents(engine)
    return loss_of_response(engine), stimulus_tolerance(engine)


def check_intubation(engine: SimulationEngine) -> ActionFeedback | None:
    unconscious, tolerance = _responsiveness(engine)
    strength = central_muscle_strength(engine)
    if unconscious < 0.1 and tolerance < 0.1 and strength > 0.5:
        return ActionFeedback(
            "awake_intubation",
            "Patient gags and resists; tube not placed.",
            blocked=True,
        )
    if unconscious < 0.1 and strength <= 0.5:
        return ActionFeedback(
            "intubation_awareness",
            "Tube placed without adequate hypnosis.",
        )
    return None


def check_airway_removal(engine: SimulationEngine) -> ActionFeedback | None:
    # Recompute TOF and T1 from the PD model; a paused command can follow a dose.
    sync_pk_state(engine)
    reasons = []
    if engine.resp.state.apnea:
        reasons.append("the patient is apneic")
    if min(engine.state.tof, engine.state.twitch) < 90.0:
        reasons.append("neuromuscular recovery is incomplete")
    if reasons:
        return ActionFeedback(
            "unsafe_airway_removal",
            f"Tube removed while {' and '.join(reasons)}. "
            "Support ventilation.",
        )
    return None


def check_ventilation_withdrawal(engine: SimulationEngine) -> ActionFeedback | None:
    # The ventilator can display normal RR/VT during complete apnea.
    if engine.resp.state.apnea:
        return ActionFeedback(
            "ventilation_stopped",
            "Assisted ventilation stopped during apnea.",
        )
    return None


def check_drug_administration(engine: SimulationEngine, key: str) -> ActionFeedback | None:
    if key in {"roc", "succinylcholine"}:
        unconscious, _ = _responsiveness(engine)
        if unconscious < 0.1 and not _induction_in_progress(engine):
            return ActionFeedback(
                "awake_neuromuscular_block",
                "Neuromuscular blocker given without adequate hypnosis.",
            )
    return None


def _induction_in_progress(engine: SimulationEngine) -> bool:
    """Recognize an induction bolus before its effect-site concentration rises.

    Used only to delay the awake-block warning; intubation checks ignore it.
    """
    hypnotic = hypnotic_equivalent(
        PROPOFOL_LOC_TRANSFER * engine.pk["propofol"].state.c1,
        engine.pk["etomidate"].state.c1,
        engine.pk["midazolam"].state.c1,
        engine.midazolam_c50,
        engine.response.age_factor,
    )
    return engine.response.loss_of_response(
        hypnotic, engine.state.opioid_ce,
        mac_sevo=engine.state.mac_sevo, mac_n2o=engine.state.mac_n2o,
        ce_ketamine=engine.pk["ketamine"].state.c1,
    ) > 0.9
