"""Hazard checks for learner actions; awake reactive or severe-laryngospasm intubation attempts fail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from anasim.core.enums import PULSELESS_RHYTHMS
from anasim.core.runtime import is_perfusing
from anasim.patient.pd import hypnotic_equivalent
from anasim.patient.pd.anesthesia import (
    PROPOFOL_LOC_TRANSFER,
    hypothermia_narcosis_fraction,
)

from .projection import (
    cardiovascular_hypnotic_ce,
    central_muscle_strength,
    extubation_tolerance,
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
    ec = engine.extracorporeal
    if (ec.is_va or ec.is_cpb) and ec.state.blood_flow_l_min >= 1.0:
        return False
    hemo = engine.hemo.state
    return (
        engine.state.cardiac_arrest
        or hemo.rhythm_type in PULSELESS_RHYTHMS
        or not is_perfusing(engine, hemo)
    )


def _drug_responsiveness(engine: SimulationEngine) -> tuple[float, float]:
    """Return drug-only (loss_of_response, stimulus_tolerance), excluding mild non-drug impairment."""
    sync_pk_state(engine)
    sync_inhaled_agents(engine)
    return (
        loss_of_response(engine, include_non_drug=False),
        stimulus_tolerance(engine, include_non_drug=False),
    )


def _deep_non_drug_coma(engine: SimulationEngine) -> bool:
    return (
        engine.cerebral.unresponsiveness >= 0.8
        or engine.toxicity.unresponsiveness >= 0.8
        or engine.neuraxial.unconsciousness >= 0.8
        or hypothermia_narcosis_fraction(engine.state.temp_c) >= 0.8
    )


def _has_sustained_hypnosis(engine: SimulationEngine, unconscious: float, tolerance: float) -> bool:
    s = engine.state
    return _in_arrest(engine) or _deep_non_drug_coma(engine) or (
        unconscious >= 0.1
        and (
            cardiovascular_hypnotic_ce(engine) >= 1.65
            or s.mac >= 1.5
            or (s.mac >= 0.85 and tolerance >= 0.25)
            or s.ketamine_ce >= 0.8
        )
    )


def check_intubation(engine: SimulationEngine) -> ActionFeedback | None:
    unconscious, _ = _responsiveness(engine)
    drug_unconscious, drug_tolerance = _drug_responsiveness(engine)
    strength = central_muscle_strength(engine)
    in_arrest = _in_arrest(engine)
    if engine.laryngospasm_severity >= 0.5 and strength > 0.5:
        return ActionFeedback(
            "laryngospasm_blocked",
            "Severe laryngospasm closes the vocal cords; tube not placed.",
            blocked=True,
        )
    sustained_hypnosis = _has_sustained_hypnosis(engine, drug_unconscious, drug_tolerance)
    if not sustained_hypnosis and (drug_unconscious < 0.1 or drug_tolerance < 0.1) and strength > 0.5:
        return ActionFeedback(
            "awake_intubation",
            "Patient gags and resists; tube not placed.",
            blocked=True,
        )
    if not in_arrest and unconscious < 0.5 and strength <= 0.5:
        return ActionFeedback(
            "intubation_awareness",
            "Tube placed without adequate hypnosis.",
        )
    return None


def check_sga_placement(engine: SimulationEngine) -> ActionFeedback | None:
    """Check responsiveness before placing a supraglottic airway."""
    unconscious, _ = _responsiveness(engine)
    drug_unconscious, drug_tolerance = _drug_responsiveness(engine)
    strength = central_muscle_strength(engine)
    in_arrest = _in_arrest(engine)
    sustained_hypnosis = _has_sustained_hypnosis(engine, drug_unconscious, drug_tolerance)
    if not sustained_hypnosis and (drug_unconscious < 0.1 or drug_tolerance < 0.1) and strength > 0.5:
        return ActionFeedback(
            "awake_sga",
            "Patient gags and resists; SGA not placed.",
            blocked=True,
        )
    if not in_arrest and unconscious < 0.5 and strength <= 0.5:
        return ActionFeedback(
            "intubation_awareness",
            "SGA placed without adequate hypnosis.",
        )
    return None


def check_oral_airway(engine: SimulationEngine) -> ActionFeedback | None:
    """Block oropharyngeal airway placement in an awake reactive patient."""
    unconscious, _ = _responsiveness(engine)
    strength = central_muscle_strength(engine)
    if not _in_arrest(engine) and unconscious < 0.15 and strength > 0.5:
        return ActionFeedback(
            "awake_opa",
            "Patient gags; oral airway requires unconsciousness.",
            blocked=True,
        )
    return None


def check_fona(engine: SimulationEngine) -> ActionFeedback | None:
    """Warn if emergency cricothyrotomy is performed without a difficult or failing airway."""
    crisis = engine.airway_crisis
    if (
        crisis.difficult_profile == "none"
        and crisis.laryngoscopy_attempts == 0
        and not crisis.esophageal_intubation
        and engine.state.airway_obstruction < 0.40
        and engine.laryngospasm_severity < 0.50
        and engine.state.sao2 >= 90.0
    ):
        return ActionFeedback(
            "unnecessary_fona",
            "Emergency surgical airway performed without a difficult, obstructed, or failing airway.",
        )
    return None


def check_airway_removal(engine: SimulationEngine) -> ActionFeedback | None:
    # Removing an esophageal tube is always appropriate and never penalized.
    if engine.airway_crisis.esophageal_intubation:
        return None
    # Recompute TOF and T1 from the PD model; a paused command can follow a dose.
    sync_inhaled_agents(engine)
    sync_pk_state(engine)
    unconscious = loss_of_response(engine)
    extub_tol = extubation_tolerance(engine)
    reasons = []
    if engine.resp.state.apnea:
        reasons.append("the patient is apneic")
    if min(engine.state.tof, 100.0 * nmba_twitch_fraction(engine)) < 90.0:
        reasons.append("neuromuscular recovery is incomplete")
    if unconscious >= 0.20 and extub_tol < 0.65:
        reasons.append("airway reflexes are reactive (Stage II)")
    if reasons:
        return ActionFeedback(
            "unsafe_airway_removal",
            f"Tube removed while {' and '.join(reasons)}. "
            "Support ventilation.",
        )
    return None


def check_ventilation_withdrawal(engine: SimulationEngine) -> ActionFeedback | None:
    # Extracorporeal gas exchange supports apnea intentionally during CPB or ECMO lung rest.
    ec = engine.extracorporeal
    if ec.is_active and ec.state.blood_flow_l_min >= 1.5 and ec.state.sweep_gas_l_min > 0.0:
        return None
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
    if key == "protamine" and engine.extracorporeal.is_active:
        return ActionFeedback(
            "protamine_on_bypass",
            "Protamine given during active extracorporeal circulation risks catastrophic circuit thrombosis.",
        )
    if key == "insulin":
        if engine.glucose.arterial_glucose_mg_dl < 70.0:
            return ActionFeedback(
                "insulin_in_hypoglycemia",
                "Insulin given during hypoglycemia (glucose < 70 mg/dL) risks severe neuroglycopenia.",
            )
        if engine.state.potassium < 3.5:
            return ActionFeedback(
                "insulin_in_hypokalemia",
                "Insulin shifts potassium intracellularly and can worsen hypokalemia (K⁺ < 3.5 mmol/L).",
            )
    if key == "sodium_bicarbonate":
        ph_check = (
            engine.resp.state.ph_37
            if abs(engine.state.ph - engine.resp.state.ph) < 1e-4
            else engine.state.ph
        )
        if ph_check > 7.50:
            return ActionFeedback(
                "bicarbonate_in_alkalemia",
                "Sodium bicarbonate given during alkalemia (pH > 7.50) worsens metabolic alkalosis and hypocalcemia.",
            )
        if engine.state.sodium > 155.0:
            return ActionFeedback(
                "bicarbonate_in_hypernatremia",
                "Hypertonic 8.4% sodium bicarbonate (1000 mEq/L Na⁺) can worsen severe hypernatremia (Na⁺ > 155 mmol/L).",
            )
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


def check_extracorporeal_start(engine: SimulationEngine, mode: str) -> ActionFeedback | None:
    """Warn if CPB is started without therapeutic heparin anticoagulation (ACT < 400 s)."""
    if mode == "cpb" and engine.coagulation.state.act_s < 400.0:
        return ActionFeedback(
            "cpb_low_act",
            "Cardiopulmonary bypass initiated without therapeutic heparinization (ACT < 400 s); verify heparin dosing and ACT.",
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
