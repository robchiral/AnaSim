"""Core temperature with anesthetic heat redistribution and shivering."""

from anasim.core.constants import (
    SHIVER_BASE_THRESHOLD,
    SHIVER_BIS_FULL,
    SHIVER_BIS_ON,
    SHIVER_DELTA_FULL,
    SHIVER_DEPTH_DROP_MAX,
    SHIVER_MAX_MULTIPLIER,
    SHIVER_REMI_DROP_MAX,
    SHIVER_TAU_OFF,
    SHIVER_TAU_ON,
    TEMP_METABOLIC_COEFFICIENT,
    ThermalTuning,
)
from anasim.core.utils import clamp, clamp01
from anasim.patient.patient import Patient


class ThermalModel:
    """Core heat balance, anesthetic depth, and the metabolic rate they set."""

    def __init__(self, patient: Patient, tuning: ThermalTuning | None = None):
        self.tuning = tuning or ThermalTuning()
        self.heat_production_basal = patient.weight * 1.0  # W
        self.heat_capacity = patient.weight * self.tuning.specific_heat_j_kg_k  # J/K
        self.surface_area = patient.bsa  # m^2
        self.redistributed_heat_j = 0.0  # Core heat moved to the periphery
        self.depth_index = 0.0
        self.shiver_level = 0.0
        self.metabolic_factor = 1.0

    def depth_and_metabolism(
        self, temp_c: float, prop_ce: float, mac: float, shiver_level: float = 0.0,
    ) -> tuple[float, float]:
        """Return the anesthetic depth index and metabolic rate relative to baseline."""
        tuning = self.tuning
        depth_index = mac + prop_ce / tuning.depth_propofol_scale
        metabolic_factor = TEMP_METABOLIC_COEFFICIENT ** (37.0 - temp_c)
        metabolic_factor *= 1.0 - tuning.metabolic_reduction_max * clamp01(depth_index)
        metabolic_factor = max(0.5, metabolic_factor) * (1.0 + SHIVER_MAX_MULTIPLIER * shiver_level)
        return depth_index, metabolic_factor

    def update_depth(self, temp_c: float, prop_ce: float, mac: float) -> None:
        self.depth_index, self.metabolic_factor = self.depth_and_metabolism(
            temp_c, prop_ce, mac, self.shiver_level
        )

    def redistribution_target_j(self, depth_index: float) -> float:
        """Core heat moved to the periphery at steady anesthetic depth."""
        return self.tuning.redistribution_core_drop_c * clamp01(depth_index) * self.heat_capacity

    def settle_at_depth(self) -> None:
        """Start as if the current depth had been held: no shivering, redistribution complete."""
        self.metabolic_factor = 1.0
        self.shiver_level = 0.0
        self.redistributed_heat_j = self.redistribution_target_j(self.depth_index)

    def step_shivering(
        self, dt: float, temp_c: float, bis: float, opioid_effect: float, muscle_factor: float,
    ) -> float:
        """Advance shivering intensity (0-1) from cold, emergence, opioid effect, and muscle strength."""
        depth_factor = clamp01(self.depth_index)
        threshold = SHIVER_BASE_THRESHOLD - SHIVER_DEPTH_DROP_MAX * depth_factor - SHIVER_REMI_DROP_MAX * opioid_effect
        temp_deficit = max(0.0, threshold - temp_c)
        cold_drive = clamp01(temp_deficit / SHIVER_DELTA_FULL)

        emergence = clamp01((bis - SHIVER_BIS_ON) / (SHIVER_BIS_FULL - SHIVER_BIS_ON))

        target = cold_drive * emergence * muscle_factor
        tau = SHIVER_TAU_ON if target > self.shiver_level else SHIVER_TAU_OFF
        self.shiver_level += (target - self.shiver_level) * (dt / tau)
        self.shiver_level = clamp01(self.shiver_level)
        return self.shiver_level

    def step_temperature(self, dt: float, temp_c: float, warmer_target_c: float) -> float:
        """Return core temperature after metabolic heat, environmental loss, and redistribution.

        A forced-air warmer target of 0 °C is off.
        """
        tuning = self.tuning
        depth_factor = clamp01(self.depth_index)

        production = self.heat_production_basal * max(0.5, self.metabolic_factor)
        conductance = (
            tuning.base_conductance_w_per_c
            * (1.0 + tuning.anesthetic_conductance_gain * depth_factor)
            * (self.surface_area / 1.9)
        )
        heat_loss = conductance * (temp_c - tuning.ambient_temp_c)

        # Anesthetic vasodilation moves core heat to the periphery over the first
        # hour (Matsukawa 1995). Vasoconstriction on lightening traps it peripherally,
        # so the core deficit does not reverse.
        deficit_gap = self.redistribution_target_j(self.depth_index) - self.redistributed_heat_j
        redistribution_w = max(0.0, deficit_gap) / tuning.redistribution_tau_s
        self.redistributed_heat_j += redistribution_w * dt

        warming = 0.0
        if warmer_target_c > 0:
            warming = tuning.bair_hugger_gain_w_per_c * max(0.0, warmer_target_c - temp_c)

        net_heat_w = production + warming - heat_loss - redistribution_w
        return clamp(temp_c + net_heat_w * dt / self.heat_capacity, tuning.temp_min_c, tuning.temp_max_c)
