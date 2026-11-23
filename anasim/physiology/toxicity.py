"""Lidocaine systemic toxicity and lipid rescue.

Concentrations are mcg/mL. Volunteer infusions and overdose reports set the
concentration ranges for sympathetic stimulation, seizures, coma, and cardiac
depression (Foldes 1960; Jorfeldt 1968; Tucker and Boas 1971; Armstrong 1988).
Lipid emulsion lowers unentrapped exposure (Heinonen 2015), represented here as
reduced effect-site delivery without altering central PK, and lessens
contractile depression based on rat bupivacaine resuscitation (Fettiplace 2014).
The response curves, conduction and arrhythmia relief, acidemia effect,
anticonvulsant suppression, and arrest thresholds are calibrations.
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
        self.sympathetic_drive = 0.0
        self.heart_rate_increase = 0.0
        self.svr_increase = 0.0
        self.conduction_block = 0.0
        self.contractility_loss = 0.0
        self.cardiac_stress = 0.0

    def add_lipid(self, amount_ml: float) -> None:
        self.lipid_ml += amount_ml
        self.total_lipid_ml += amount_ml

    def step_lipid(self, dt: float, circulation: float) -> None:
        if self.lipid_ml == 0.0 and self.lipid_effect == 0.0:
            return
        # Lipid acts and clears only as blood circulates.
        self.lipid_ml *= math.exp(-math.log(2.0) * dt * circulation / self.LIPID_HALF_TIME_S)
        target = self.lipid_ml / (self.lipid_ml + 1.5 * self.weight)
        self.lipid_effect += (target - self.lipid_effect) * -math.expm1(
            -dt * circulation / self.LIPID_DELIVERY_TAU_S,
        )

    @property
    def effect_input_fraction(self) -> float:
        # Calibrated to match Heinonen 2015's 23% lower 30-min unentrapped
        # volunteer lidocaine AUC (mostly from larger distribution volume).
        return 1.0 - 0.35 * self.lipid_effect

    def step(self, dt: float, *, neural_ce: float, cardiac_ce: float, ph: float,
             propofol_ce: float, midazolam_ce: float, mac_sevo: float,
             motor_strength: float, cortical_function: float,
             mac_n2o: float = 0.0) -> None:
        if (
            neural_ce <= 0.0
            and cardiac_ce <= 0.0
            and self.seizure_activity == 0.0
            and self.sympathetic_drive == 0.0
            and self.neural_depression == 0.0
            and self.conduction_block == 0.0
            and self.contractility_loss == 0.0
            and self.cardiac_stress == 0.0
        ):
            return
        acidemia = 1.0 + 0.5 * clamp01((7.35 - ph) / 0.35)
        neural = max(0.0, neural_ce) * acidemia
        cardiac = max(0.0, cardiac_ce) * acidemia
        self.neural_depression = _response(neural, 10.5, 20.0)
        excitation = _response(neural, 5.0, 9.5) * (1.0 - self.neural_depression)
        inhaled_mac = max(0.0, mac_sevo) + 0.5 * max(0.0, mac_n2o)
        remaining_excitation = ((1.0 - hill_function(midazolam_ce, 30.0, 2.0))
                                * (1.0 - hill_function(propofol_ce, 1.5, 2.0))
                                * (1.0 - hill_function(inhaled_mac, 0.75, 2.0)))
        cortical = clamp01(cortical_function)
        target = excitation * remaining_excitation * cortical
        self.seizure_activity += (target - self.seizure_activity) * -math.expm1(-dt / 2.0)
        self.convulsion_intensity = self.seizure_activity * clamp01(motor_strength)
        # Early CNS excitation raises HR and pressure until depression sets in.
        sympathetic_target = (
            _response(neural, 4.2, 9.0)
            * (1.0 - _response(neural, 9.5, 13.5))
            * remaining_excitation
            * cortical
        )
        self.sympathetic_drive += (sympathetic_target - self.sympathetic_drive) * -math.expm1(-dt / 5.0)
        self.heart_rate_increase = 16.0 * self.sympathetic_drive
        self.svr_increase = 2.4 * self.sympathetic_drive
        eff_cardiac = cardiac * (1.0 - 0.22 * self.lipid_effect)
        self.conduction_block = _response(eff_cardiac, 18.0, 48.0)
        self.contractility_loss = _response(cardiac, 13.0, 42.0) * (1.0 - 0.35 * self.lipid_effect)
        self.cardiac_stress = _response(cardiac, 22.0, 44.0) * (1.0 - 0.35 * self.lipid_effect)

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
