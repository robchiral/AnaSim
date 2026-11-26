"""Typed registry of cases for guided and independent practice."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .anaphylaxis import create_anaphylaxis_scenario
from .base import Scenario, ScenarioStep
from .cardiac_arrest import create_cardiac_arrest
from .emergence import create_emergence
from .extracorporeal import (
    create_cpb_weaning,
    create_protamine_reversal,
    create_va_ecmo_harlequin,
    create_vv_ecmo_ards,
)
from .hemorrhage import create_hemorrhage_response
from .induction import create_induction_balanced, create_induction_tiva
from .malignant_hyperthermia import create_malignant_hyperthermia
from .oxygen_supply import create_oxygen_supply_failure
from .sepsis import create_sepsis_response


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Scenario builder and the session configuration it requires."""

    id: str
    label: str
    summary: str  # Shown under the setup screen's case picker
    start_mode: Literal["awake", "steady_state"]
    maint_type: Literal["tiva", "balanced"]
    builder: Callable[[], Scenario]
    surgery: bool = False  # Starts with sustained surgical stimulation in progress


def _create_emergence_tiva() -> Scenario:
    return create_emergence("tiva")


def _create_emergence_balanced() -> Scenario:
    return create_emergence("balanced")


SCENARIO_REGISTRY = (
    ScenarioSpec(
        "induction_tiva",
        "TIVA induction",
        "Preoxygenate, induce with propofol and rocuronium, intubate, and start propofol-remifentanil maintenance.",
        "awake",
        "tiva",
        create_induction_tiva,
    ),
    ScenarioSpec(
        "induction_balanced",
        "Balanced induction",
        "Preoxygenate, induce with fentanyl, propofol, and rocuronium, intubate, and start sevoflurane maintenance.",
        "awake",
        "balanced",
        create_induction_balanced,
    ),
    ScenarioSpec(
        "emergence_tiva",
        "TIVA emergence",
        "End of surgery: reverse residual neuromuscular block, wake the patient, and extubate.",
        "steady_state",
        "tiva",
        _create_emergence_tiva,
    ),
    ScenarioSpec(
        "emergence_balanced",
        "Inhalational emergence",
        "End of surgery: reverse residual neuromuscular block, wash out sevoflurane, and extubate.",
        "steady_state",
        "balanced",
        _create_emergence_balanced,
    ),
    ScenarioSpec(
        "oxygen_supply_failure",
        "O₂ supply failure",
        "Pipeline oxygen supply fails during surgery.",
        "steady_state",
        "tiva",
        create_oxygen_supply_failure,
        surgery=True,
    ),
    ScenarioSpec(
        "hemorrhage_response",
        "Hemorrhage response",
        "Brisk surgical bleeding causes hypovolemic shock.",
        "steady_state",
        "tiva",
        create_hemorrhage_response,
        surgery=True,
    ),
    ScenarioSpec(
        "anaphylaxis_response",
        "Anaphylaxis response",
        "Severe anaphylaxis after an intraoperative antibiotic dose.",
        "steady_state",
        "tiva",
        create_anaphylaxis_scenario,
        surgery=True,
    ),
    ScenarioSpec(
        "sepsis_response",
        "Septic shock response",
        "Intraoperative septic shock from an intra-abdominal infection.",
        "steady_state",
        "tiva",
        create_sepsis_response,
        surgery=True,
    ),
    ScenarioSpec(
        "malignant_hyperthermia",
        "Malignant hyperthermia",
        "Rising EtCO₂, tachycardia, and muscle rigidity under sevoflurane.",
        "steady_state",
        "balanced",
        create_malignant_hyperthermia,
        surgery=True,
    ),
    ScenarioSpec(
        "cardiac_arrest",
        "Cardiac arrest",
        "Sudden ventricular fibrillation during surgery.",
        "steady_state",
        "tiva",
        create_cardiac_arrest,
        surgery=True,
    ),
    ScenarioSpec(
        "vv_ecmo_ards",
        "VV ECMO for severe ARDS",
        "Initiate and titrate venovenous ECMO for refractory ARDS hypoxemia.",
        "steady_state",
        "tiva",
        create_vv_ecmo_ards,
    ),
    ScenarioSpec(
        "va_ecmo_harlequin",
        "VA ECMO differential hypoxia",
        "Recognize and treat upper-body (Harlequin) hypoxemia on peripheral femoral VA ECMO.",
        "steady_state",
        "tiva",
        create_va_ecmo_harlequin,
    ),
    ScenarioSpec(
        "cpb_weaning",
        "CPB rewarming and weaning",
        "Rewarm on bypass, release the aortic cross-clamp, resume ventilation, refill the heart, and separate from CPB.",
        "steady_state",
        "tiva",
        create_cpb_weaning,
    ),
    ScenarioSpec(
        "protamine_reversal",
        "Post-CPB protamine reversal",
        "Measure post-bypass ACT, reverse residual heparin slowly with protamine, and confirm ACT normalization.",
        "steady_state",
        "tiva",
        create_protamine_reversal,
    ),
)

__all__ = [
    "SCENARIO_REGISTRY",
    "Scenario",
    "ScenarioSpec",
    "ScenarioStep",
    "create_anaphylaxis_scenario",
    "create_cardiac_arrest",
    "create_cpb_weaning",
    "create_emergence",
    "create_hemorrhage_response",
    "create_induction_balanced",
    "create_induction_tiva",
    "create_malignant_hyperthermia",
    "create_oxygen_supply_failure",
    "create_protamine_reversal",
    "create_sepsis_response",
    "create_va_ecmo_harlequin",
    "create_vv_ecmo_ards",
]
