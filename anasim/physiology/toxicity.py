"""Lidocaine systemic toxicity and lipid rescue.

Concentrations are mcg/mL. The lidocaine label and overdose reports (Armstrong
1988) set the CNS and cardiac ranges; the response curves and treatment effects
are calibrations. Lipid reduces effect-site exposure and leaves the plasma
concentration unchanged.
"""

import math

from anasim.core.utils import clamp01, hill_function

# Sodium-channel block at which no beats are conducted.
SILENT_BLOCK = 0.99


def _response(concentration: float, onset: float, full: float) -> float:
    fraction = clamp01((concentration - onset) / (full - onset))
    return fraction * fraction * (3.0 - 2.0 * fraction)


class LocalAnestheticToxicity:
    LIPID_HALF_TIME_S = 1200.0
    LIPID_DELIVERY_TAU_S = 60.0

    def __init__(self, weight: float):
        self.weight = weight
        self.lipid_ml = 0.0
        self.total_lipid_ml = 0.0
        self.lipid_effect = 0.0
        self.seizure_activity = 0.0
        self.convulsion_intensity = 0.0
        self.neural_depression = 0.0
        self.conduction_block = 0.0
        self.contractility_loss = 0.0
        self.cardiac_stress = 0.0

    def add_lipid(self, amount_ml: float) -> None:
        self.lipid_ml += amount_ml
        self.total_lipid_ml += amount_ml

    def step_lipid(self, dt: float, circulation: float) -> None:
        # Lipid acts and clears only as blood circulates.
        self.lipid_ml *= math.exp(-math.log(2.0) * dt * circulation / self.LIPID_HALF_TIME_S)
        target = self.lipid_ml / (self.lipid_ml + 1.5 * self.weight)
        self.lipid_effect += (target - self.lipid_effect) * -math.expm1(
            -dt * circulation / self.LIPID_DELIVERY_TAU_S,
        )

    @property
    def effect_input_fraction(self) -> float:
        # Heinonen 2015 found about 23% lower unentrapped AUC in volunteers.
        return 1.0 - 0.25 * self.lipid_effect

    def step(self, dt: float, *, neural_ce: float, cardiac_ce: float, ph: float,
             propofol_ce: float, midazolam_ce: float, mac_sevo: float,
             motor_strength: float, cortical_function: float) -> None:
        acidemia = 1.0 + 0.5 * clamp01((7.35 - ph) / 0.35)
        neural = max(0.0, neural_ce) * acidemia
        cardiac = max(0.0, cardiac_ce) * acidemia
        self.neural_depression = _response(neural, 12.0, 24.0)
        excitation = _response(neural, 6.0, 10.0) * (1.0 - self.neural_depression)
        # Calibrated anticonvulsant effects at each drug's effect site.
        remaining_excitation = ((1.0 - hill_function(midazolam_ce, 30.0, 2.0))
                                * (1.0 - hill_function(propofol_ce, 1.5, 2.0))
                                * (1.0 - hill_function(mac_sevo, 0.75, 2.0)))
        target = excitation * remaining_excitation * clamp01(cortical_function)
        self.seizure_activity += (target - self.seizure_activity) * -math.expm1(-dt / 2.0)
        self.convulsion_intensity = self.seizure_activity * clamp01(motor_strength)
        self.conduction_block = _response(cardiac, 10.0, 30.0)
        self.contractility_loss = _response(cardiac, 10.0, 25.0) * (1.0 - 0.35 * self.lipid_effect)
        self.cardiac_stress = _response(cardiac, 8.0, 20.0)

    @property
    def electrically_silent(self) -> bool:
        return self.conduction_block >= SILENT_BLOCK

    @property
    def unresponsiveness(self) -> float:
        return max(self.neural_depression, clamp01(self.seizure_activity / 0.5))

    @property
    def respiratory_drive(self) -> float:
        return (1.0 - self.neural_depression) * (1.0 - 0.6 * self.seizure_activity)

    @property
    def metabolic_factor(self) -> float:
        # Cerebral seizure activity and muscular convulsions both raise metabolism.
        return 1.0 + 0.3 * self.seizure_activity + 1.2 * self.convulsion_intensity
