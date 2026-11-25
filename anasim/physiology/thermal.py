"""Core temperature with anesthetic heat redistribution and shivering."""

import math

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
        self.heat_production_basal = 70.0 * (patient.weight / 70.0) ** 0.75 * patient.pregnancy.metabolism  # W
        # Excess adipose tissue above BMI 25 has lower specific heat (~2300 J/kg/K)
        # than normal body composition (~3470 J/kg/K).
        normal_weight = 25.0 * (patient.height / 100.0) ** 2
        excess_fat = max(0.0, patient.weight - normal_weight)
        base_mass = patient.weight - excess_fat
        self.heat_capacity = (
            base_mass * self.tuning.specific_heat_j_kg_k + excess_fat * 2300.0
        )  # J/K
        self.surface_area = patient.bsa  # m^2
        # Kurz 1995 (Anesth Analg 80:562): first-hour redistribution core cooling
        # falls linearly with weight-to-surface-area ratio (~1.6 °C near 38 kg/m²
        # vs ~0.5 °C above 55 kg/m²) because obese periphery is warmer preoperatively.
        wt_per_bsa = patient.weight / max(0.5, self.surface_area)
        self.redistribution_morphometric_factor = clamp(1.0 - 0.042 * (wt_per_bsa - 38.7), 0.22, 1.15)
        # Awake vasomotor tone holds core temperature at baseline: heat loss
        # matches basal production there. Anesthesia replaces it with the
        # passive vasodilated conductance in proportion to depth.
        basal_temp_factor = max(0.25, TEMP_METABOLIC_COEFFICIENT ** (37.0 - patient.baseline_temp))
        self.awake_conductance = (
            self.heat_production_basal * basal_temp_factor
            / max(1.0, patient.baseline_temp - self.tuning.ambient_temp_c)
        )
        self.baseline_temp = patient.baseline_temp
        self.redistributed_heat_j = 0.0  # Core heat moved to the periphery
        self.depth_index = 0.0
        self.shiver_level = 0.0
        self.metabolic_factor = 1.0
        self.extra_heat_w = 0.0
        self._pending_fluid_heat_j = 0.0

    def add_fluid_heat_j(
        self, volume_ml: float, infusate_temp_c: float, core_temp_c: float, *, is_blood: bool = False,
    ) -> None:
        """Accumulate sensible heat transfer (J) from infused fluids or blood products."""
        if volume_ml <= 0.0:
            return
        specific_heat_j_per_ml_c = 3.85 if is_blood else 4.186
        self._pending_fluid_heat_j += volume_ml * specific_heat_j_per_ml_c * (infusate_temp_c - core_temp_c)

    def target_shivering(
        self, temp_c: float, bis: float, opioid_effect: float, muscle_factor: float,
        unconscious: float = 0.0, *, perfusing: bool = True,
    ) -> float:
        """Equilibrium shivering intensity (0-1) from cold, emergence, opioids, muscle strength, and perfusion."""
        if not perfusing or temp_c <= 31.0:
            return 0.0
        depth_factor = clamp01(self.depth_index)
        threshold = SHIVER_BASE_THRESHOLD - SHIVER_DEPTH_DROP_MAX * depth_factor - SHIVER_REMI_DROP_MAX * opioid_effect
        temp_deficit = max(0.0, threshold - temp_c)
        cold_drive = clamp01(temp_deficit / SHIVER_DELTA_FULL)
        if temp_c < 33.0:
            # Brown 2012 (NEJM): shivering peaks in mild hypothermia and ceases near 32 °C (HT I -> II).
            cold_drive *= clamp01((temp_c - 31.0) / 2.0)

        emergence_bis = clamp01((bis - SHIVER_BIS_ON) / (SHIVER_BIS_FULL - SHIVER_BIS_ON))
        emergence = min(emergence_bis, clamp01(1.0 - unconscious))
        # Anesthetics lower the shivering threshold above and attenuate gain rather than
        # imposing a hard zero gate when core temperature falls below the anesthetized threshold.
        anesthetic_gain = 0.35 + 0.65 * emergence
        return clamp01(cold_drive * anesthetic_gain * muscle_factor)

    def depth_and_metabolism(
        self, temp_c: float, prop_ce: float, mac: float, shiver_level: float = 0.0,
    ) -> tuple[float, float]:
        """Return the anesthetic depth index and metabolic rate relative to baseline."""
        tuning = self.tuning
        depth_index = mac + prop_ce / tuning.depth_propofol_scale
        metabolic_factor = TEMP_METABOLIC_COEFFICIENT ** (37.0 - temp_c)
        metabolic_factor *= 1.0 - tuning.metabolic_reduction_max * clamp01(depth_index)
        metabolic_factor = max(0.25, metabolic_factor) * (1.0 + SHIVER_MAX_MULTIPLIER * shiver_level)
        return depth_index, metabolic_factor

    def update_depth(self, temp_c: float, prop_ce: float, mac: float) -> None:
        self.depth_index, self.metabolic_factor = self.depth_and_metabolism(
            temp_c, prop_ce, mac, self.shiver_level
        )

    def redistribution_target_j(self, depth_index: float) -> float:
        """Core heat moved to the periphery at steady anesthetic depth."""
        return (
            self.tuning.redistribution_core_drop_c
            * self.redistribution_morphometric_factor
            * clamp01(depth_index)
            * self.heat_capacity
        )

    def settle_at_depth(self) -> None:
        """Start as if the current depth had been held: no shivering, redistribution complete."""
        self.shiver_level = 0.0
        self.redistributed_heat_j = self.redistribution_target_j(self.depth_index)

    def step_shivering(
        self, dt: float, temp_c: float, bis: float, opioid_effect: float, muscle_factor: float,
        unconscious: float = 0.0, *, perfusing: bool = True,
    ) -> float:
        """Advance shivering intensity (0-1) from cold, emergence, opioid effect, and muscle strength."""
        if not perfusing:
            self.shiver_level = 0.0
            return 0.0
        target = self.target_shivering(
            temp_c, bis, opioid_effect, muscle_factor, unconscious=unconscious, perfusing=perfusing,
        )
        tau = SHIVER_TAU_ON if target > self.shiver_level else SHIVER_TAU_OFF
        self.shiver_level += (target - self.shiver_level) * -math.expm1(-dt / tau)
        self.shiver_level = clamp01(self.shiver_level)
        return self.shiver_level

    def step_temperature(self, dt: float, temp_c: float, warmer_target_c: float,
                         surface_cooling: bool = False, *,
                         oxygen_supply_fraction: float = 1.0,
                         sepsis_severity: float = 0.0,
                         perfusing: bool = True,
                         opioid_effect: float = 0.0) -> float:
        """Return core temperature after metabolic heat, environmental loss, and redistribution.

        A forced-air warmer target of 0 °C is off.
        """
        tuning = self.tuning
        depth_factor = clamp01(self.depth_index)

        aerobic_gate = clamp(oxygen_supply_fraction, 0.12, 1.0)
        production = (
            self.heat_production_basal * max(0.25, self.metabolic_factor) * aerobic_gate
            + self.extra_heat_w
        )
        anesthetized = max(
            1.12 * self.awake_conductance,
            tuning.anesthetized_conductance_w_per_c * self.surface_area / 1.9,
        )
        conductance = self.awake_conductance + (anesthetized - self.awake_conductance) * depth_factor
        # Phase-3 thermoregulatory vasoconstriction (Sessler 2000; Kurz 1995):
        # once core temperature falls below the anesthetized vasoconstriction
        # threshold (~34.3-34.6 °C at 1 MAC), cutaneous constriction limits
        # further passive heat loss to establish a thermal plateau.
        if perfusing:
            vasoconst_threshold = (
                self.baseline_temp - 0.3 - 2.4 * depth_factor - SHIVER_REMI_DROP_MAX * clamp01(opioid_effect)
            )
            if temp_c < vasoconst_threshold:
                vasoparesis = clamp(1.0 - 0.18 * max(0.0, 32.5 - temp_c), 0.2, 1.0)
                vasoconst = clamp01((vasoconst_threshold - temp_c) / 1.0) * vasoparesis
                plateau_conductance = 0.72 * self.awake_conductance
                conductance = conductance + (plateau_conductance - conductance) * vasoconst
        # Forced-air warming blankets shield ~60% of BSA from ambient room air
        # while delivering convective heat transfer (Giesbrecht 1994).
        warmer_active = warmer_target_c > 0.0
        exposed_fraction = 0.40 if warmer_active else 1.0
        heat_loss = conductance * exposed_fraction * (temp_c - tuning.ambient_temp_c)

        # Active sweating and cutaneous vasodilation regulate fever above the
        # hypothalamus's setpoint (shifted upward by sepsis and anesthesia; Sessler 2000).
        sweating_w = 0.0
        if perfusing:
            febrile_setpoint = max(self.baseline_temp, 37.0) + 1.8 * clamp01(sepsis_severity)
            sweat_threshold = febrile_setpoint + 0.25 + 1.2 * depth_factor
            if temp_c > sweat_threshold:
                sweating_w = (
                    140.0
                    * (self.surface_area / 1.9)
                    * (1.0 - 0.45 * depth_factor)
                    * (temp_c - sweat_threshold)
                )

        # Anesthetic vasodilation moves core heat to the periphery over the first
        # hour (Matsukawa 1995). Vasoconstriction on lightening traps it peripherally,
        # and zero-flow arrest halts convective redistribution.
        if perfusing:
            deficit_gap = self.redistribution_target_j(self.depth_index) - self.redistributed_heat_j
            redistribution_w = max(0.0, deficit_gap) / tuning.redistribution_tau_s
            self.redistributed_heat_j += redistribution_w * dt
        else:
            redistribution_w = 0.0

        warming = 0.0
        if warmer_active:
            warming = (
                tuning.bair_hugger_gain_w_per_c
                * (self.surface_area / 1.9)
                * (warmer_target_c - temp_c)
            )

        cooling = 0.0
        if surface_cooling:
            cooling = (tuning.surface_cooling_w_per_c * self.surface_area / 1.9
                       * max(0.0, temp_c - tuning.temp_min_c))
        fluid_heat_j = self._pending_fluid_heat_j
        self._pending_fluid_heat_j = 0.0
        net_heat_w = production + warming - heat_loss - sweating_w - redistribution_w - cooling
        return clamp(
            temp_c + (net_heat_w * dt + fluid_heat_j) / self.heat_capacity,
            tuning.temp_min_c,
            tuning.temp_max_c,
        )
