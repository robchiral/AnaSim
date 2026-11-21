"""Hazard checks for learner actions; awake reactive or severe-laryngospasm intubation attempts fail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from anasim.core.enums import PULSELESS_RHYTHMS
from anasim.core.runtime import is_perfusing
from anasim.patient.pd import hypnotic_equivalent
from anasim.patient.pd.anesthesia import PROPOFOL_LOC_TRANSFER

from .projection import (
    cardiovascular_hypnotic_ce,
    central_muscle_strength,
    loss_of_response,
    nmba_twitch_fraction,
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


def _in_arrest(engine: SimulationEngine) -> bool:
    hemo = engine.hemo.state
    return (
        engine.state.cardiac_arrest
        or hemo.rhythm_type in PULSELESS_RHYTHMS
        or not is_perfusing(engine, hemo)
    )


def check_intubation(engine: SimulationEngine) -> ActionFeedback | None:
    unconscious, tolerance = _responsiveness(engine)
    strength = central_muscle_strength(engine)
    s = engine.state
    in_arrest = _in_arrest(engine)
    if engine.laryngospasm_severity >= 0.5 and strength > 0.5:
        return ActionFeedback(
            "laryngospasm_blocked",
            "Severe laryngospasm closes the vocal cords; tube not placed.",
            blocked=True,
        )
    sustained_hypnosis = in_arrest or (
        unconscious >= 0.1
        and (
            cardiovascular_hypnotic_ce(engine) >= 1.65
            or s.mac >= 0.5
            or s.ketamine_ce >= 0.8
            or engine.cerebral.unresponsiveness >= 0.8
            or engine.toxicity.unresponsiveness >= 0.8
        )
    )
    if not sustained_hypnosis and (unconscious < 0.1 or tolerance < 0.1) and strength > 0.5:
        return ActionFeedback(
            "awake_intubation",
            "Patient gags and resists; tube not placed.",
            blocked=True,
        )
    if not in_arrest and unconscious < 0.1 and strength <= 0.5:
        return ActionFeedback(
            "intubation_awareness",
            "Tube placed without adequate hypnosis.",
        )
    return None


def check_airway_removal(engine: SimulationEngine) -> ActionFeedback | None:
    # Recompute TOF and T1 from the PD model; a paused command can follow a dose.
    sync_inhaled_agents(engine)
    sync_pk_state(engine)
    unconscious = loss_of_response(engine)
    tolerance = stimulus_tolerance(engine)
    reasons = []
    if engine.resp.state.apnea:
        reasons.append("the patient is apneic")
    if min(engine.state.tof, 100.0 * nmba_twitch_fraction(engine)) < 90.0:
        reasons.append("neuromuscular recovery is incomplete")
    if unconscious >= 0.20 and tolerance < 0.65:
        reasons.append("airway reflexes are reactive (Stage II)")
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


def check_shock(engine: SimulationEngine) -> ActionFeedback | None:
    hemo = engine.hemo.state
    if hemo.rhythm_type in PULSELESS_RHYTHMS or not is_perfusing(engine, hemo):
        return None
    unconscious, _ = _responsiveness(engine)
    if unconscious < 0.5:
        return ActionFeedback(
            "shock_while_responsive",
            "Shock delivered to a responsive patient.",
        )
    return None


def check_drug_administration(engine: SimulationEngine, key: str) -> ActionFeedback | None:
    if key == "succinylcholine":
        if engine.malignant_hyperthermia.active or engine.malignant_hyperthermia.susceptible:
            return ActionFeedback(
                "succinylcholine_in_mh",
                "Succinylcholine is contraindicated in malignant hyperthermia.",
            )
        if engine.state.potassium >= 5.5:
            return ActionFeedback(
                "succinylcholine_in_hyperkalemia",
                "Succinylcholine risks hyperkalemic cardiac arrest when serum potassium is elevated.",
            )
    if key in {"roc", "succinylcholine"}:
        if _in_arrest(engine):
            return None
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
