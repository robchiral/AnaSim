# Scenarios package
from .base import ScenarioStep, Scenario
from .induction import create_induction_balanced, create_induction_tiva
from .emergence import create_emergence
from .hemorrhage import create_hemorrhage_response
from .anaphylaxis import create_anaphylaxis_scenario
from .sepsis import create_sepsis_response

__all__ = [
    'ScenarioStep',
    'Scenario',
    'create_induction_balanced',
    'create_induction_tiva', 
    'create_emergence',
    'create_hemorrhage_response',
    'create_anaphylaxis_scenario',
    'create_sepsis_response',
    'SCENARIO_BUILDERS',
]

SCENARIO_BUILDERS = {
    "hemorrhage_response": create_hemorrhage_response,
    "anaphylaxis_response": create_anaphylaxis_scenario,
    "sepsis_response": create_sepsis_response,
}
