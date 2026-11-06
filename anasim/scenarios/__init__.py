"""Typed registry of cases for guided and independent practice."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .anaphylaxis import create_anaphylaxis_scenario
from .base import Scenario, ScenarioStep
from .cardiac_arrest import create_cardiac_arrest
from .emergence import create_emergence
from .hemorrhage import create_hemorrhage_response
from .induction import create_induction_balanced, create_induction_tiva
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
        "Awake before induction: propofol and rocuronium induction, intubation, and propofol-remifentanil maintenance.",
        "awake",
        "tiva",
        create_induction_tiva,
    ),
    ScenarioSpec(
        "induction_balanced",
        "Balanced induction",
        "Awake before induction: fentanyl, propofol, and rocuronium induction, intubation, and sevoflurane maintenance.",
        "awake",
        "balanced",
        create_induction_balanced,
    ),
    ScenarioSpec(
        "emergence_tiva",
        "TIVA emergence",
        "End of surgery under TIVA: reverse residual block, wake the patient, and extubate.",
        "steady_state",
        "tiva",
        _create_emergence_tiva,
    ),
    ScenarioSpec(
        "emergence_balanced",
        "Inhalational emergence",
        "End of surgery under sevoflurane: reverse residual block, wake the patient, and extubate.",
        "steady_state",
        "balanced",
        _create_emergence_balanced,
    ),
    ScenarioSpec(
        "oxygen_supply_failure",
        "O₂ supply failure",
        "During surgery under TIVA: the pipeline oxygen supply fails.",
        "steady_state",
        "tiva",
        create_oxygen_supply_failure,
        surgery=True,
    ),
    ScenarioSpec(
        "hemorrhage_response",
        "Hemorrhage response",
        "During surgery under TIVA: brisk surgical bleeding.",
        "steady_state",
        "tiva",
        create_hemorrhage_response,
        surgery=True,
    ),
    ScenarioSpec(
        "anaphylaxis_response",
        "Anaphylaxis response",
        "During surgery under TIVA: severe anaphylaxis after an antibiotic dose.",
        "steady_state",
        "tiva",
        create_anaphylaxis_scenario,
        surgery=True,
    ),
    ScenarioSpec(
        "sepsis_response",
        "Septic shock response",
        "During surgery under TIVA: septic shock from an intra-abdominal infection.",
        "steady_state",
        "tiva",
        create_sepsis_response,
        surgery=True,
    ),
    ScenarioSpec(
        "cardiac_arrest",
        "Cardiac arrest",
        "During surgery under TIVA: ventricular fibrillation.",
        "steady_state",
        "tiva",
        create_cardiac_arrest,
        surgery=True,
    ),
)

__all__ = [
    "SCENARIO_REGISTRY",
    "Scenario",
    "ScenarioSpec",
    "ScenarioStep",
    "create_anaphylaxis_scenario",
    "create_cardiac_arrest",
    "create_emergence",
    "create_hemorrhage_response",
    "create_induction_balanced",
    "create_induction_tiva",
    "create_oxygen_supply_failure",
    "create_sepsis_response",
]
