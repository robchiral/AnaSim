"""Typed registry of guided simulation scenarios."""

from dataclasses import dataclass
from typing import Callable, Literal

from .anaphylaxis import create_anaphylaxis_scenario
from .base import Scenario, ScenarioStep
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
    start_mode: Literal["awake", "steady_state"]
    maint_type: Literal["tiva", "balanced"]
    builder: Callable[[], Scenario]


def _create_emergence_tiva() -> Scenario:
    return create_emergence("tiva")


def _create_emergence_balanced() -> Scenario:
    return create_emergence("balanced")


SCENARIO_REGISTRY = (
    ScenarioSpec(
        "induction_tiva",
        "TIVA induction",
        "awake",
        "tiva",
        create_induction_tiva,
    ),
    ScenarioSpec(
        "induction_balanced",
        "Balanced induction",
        "awake",
        "balanced",
        create_induction_balanced,
    ),
    ScenarioSpec(
        "emergence_tiva",
        "TIVA emergence",
        "steady_state",
        "tiva",
        _create_emergence_tiva,
    ),
    ScenarioSpec(
        "emergence_balanced",
        "Inhalational emergence",
        "steady_state",
        "balanced",
        _create_emergence_balanced,
    ),
    ScenarioSpec(
        "oxygen_supply_failure",
        "O₂ supply failure",
        "steady_state",
        "tiva",
        create_oxygen_supply_failure,
    ),
    ScenarioSpec(
        "hemorrhage_response",
        "Hemorrhage response",
        "steady_state",
        "tiva",
        create_hemorrhage_response,
    ),
    ScenarioSpec(
        "anaphylaxis_response",
        "Anaphylaxis response",
        "steady_state",
        "tiva",
        create_anaphylaxis_scenario,
    ),
    ScenarioSpec(
        "sepsis_response",
        "Septic shock response",
        "steady_state",
        "tiva",
        create_sepsis_response,
    ),
)

SCENARIO_BUILDERS = {spec.id: spec.builder for spec in SCENARIO_REGISTRY}

__all__ = [
    "Scenario",
    "ScenarioStep",
    "ScenarioSpec",
    "SCENARIO_REGISTRY",
    "SCENARIO_BUILDERS",
    "create_induction_balanced",
    "create_induction_tiva",
    "create_emergence",
    "create_hemorrhage_response",
    "create_anaphylaxis_scenario",
    "create_sepsis_response",
    "create_oxygen_supply_failure",
]
