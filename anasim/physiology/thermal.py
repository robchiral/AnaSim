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


def age_vasoconstriction_shift(age: float) -> float:
    """Ozaki 1997: additional anesthetized vasoconstriction threshold drop (°C) with age."""
    return 0.55 * clamp01((age - 45.0) / 30.0)


def cold_vasoparesis(temp_c: float) -> float:
    """Smooth-muscle vasoconstriction attenuation during moderate-to-severe hypothermia."""
    return clamp(1.0 - 0.18 * max(0.0, 32.5 - temp_c), 0.2, 1.0)


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
        self._inv_heat_capacity = 1.0 / self.heat_capacity
        self.surface_area = patient.bsa  # m^2
        self._bsa_scale = self.surface_area / 1.9
        # Kurz 1995a (Anesth Analg 80:562): first-hour redistribution core cooling
        # falls linearly with weight-to-surface-area ratio (~1.4 °C near 38 kg/m²
        # vs ~0.5 °C near 55 kg/m²) because obese periphery is warmer preoperatively.
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
        self._anesthetized_conductance = max(
            1.12 * self.awake_conductance,
            self.tuning.anesthetized_conductance_w_per_c * self._bsa_scale,
        )
        self._plateau_conductance = 0.72 * self.awake_conductance
        self.baseline_temp = patient.baseline_temp
        self._base_ref_temp = max(patient.baseline_temp, 37.0)
        self.patient_age = patient.age
        self._age_vc_shift = age_vasoconstriction_shift(patient.age)
        self.redistributed_heat_j = 0.0  # Core heat moved to the periphery
        self.depth_index = 0.0
        self.shiver_level = 0.0
        self.metabolic_factor = 1.0
        self.extra_heat_w = 0.0
        self._pending_fluid_heat_j = 0.0
        # Two-compartment peripheral tissue temperature engaged during extracorporeal
        # cooling/rewarming to model post-CPB afterdrop (Rajek 1998; Rajek 2000).
        self.peripheral_temp_c = float(patient.baseline_temp)
        self._extracorporeal_thermal_engaged = False

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
        neuraxial_shiver_shift_c: float = 0.0,
        neuraxial_muscle_keep: float = 1.0,
    ) -> float:
        """Equilibrium shivering intensity (0-1) from cold, emergence, opioids, muscle strength, and perfusion."""
        if not perfusing or temp_c <= 31.0:
            return 0.0
        depth_factor = clamp01(self.depth_index)
        threshold = (
            SHIVER_BASE_THRESHOLD
            - SHIVER_DEPTH_DROP_MAX * depth_factor
            - SHIVER_REMI_DROP_MAX * opioid_effect
            - max(0.0, neuraxial_shiver_shift_c)
        )
        temp_deficit = max(0.0, threshold - temp_c)
        cold_drive = clamp01(temp_deficit / SHIVER_DELTA_FULL)
        if temp_c < 33.0:
            # Brown 2012: shivering ceases below 32 °C (modeled as tapering from 33 °C to zero at 31 °C).
            cold_drive *= clamp01((temp_c - 31.0) / 2.0)

        emergence_bis = clamp01((bis - SHIVER_BIS_ON) / (SHIVER_BIS_FULL - SHIVER_BIS_ON))
        emergence = min(emergence_bis, clamp01(1.0 - unconscious))
        # Anesthetics lower the shivering threshold above and attenuate gain rather than
        # imposing a hard zero gate when core temperature falls below the anesthetized threshold.
        anesthetic_gain = 0.35 + 0.65 * emergence
        return clamp01(cold_drive * anesthetic_gain * muscle_factor * clamp01(neuraxial_muscle_keep))

    def depth_and_metabolism(
        self, temp_c: float, prop_ce: float, mac: float, shiver_level: float = 0.0,
    ) -> tuple[float, float]:
        """Return the anesthetic depth index and metabolic rate relative to baseline."""
        tuning = self.tuning
        depth_index = mac + prop_ce / tuning.depth_propofol_scale
        metabolic_factor = TEMP_METABOLIC_COEFFICIENT ** (37.0 - temp_c) if temp_c != 37.0 else 1.0
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
        neuraxial_shiver_shift_c: float = 0.0,
        neuraxial_muscle_keep: float = 1.0,
    ) -> float:
        """Advance shivering intensity (0-1) from cold, emergence, opioid effect, and muscle strength."""
        if not perfusing:
            self.shiver_level = 0.0
            return 0.0
        target = self.target_shivering(
            temp_c, bis, opioid_effect, muscle_factor, unconscious=unconscious, perfusing=perfusing,
            neuraxial_shiver_shift_c=neuraxial_shiver_shift_c,
            neuraxial_muscle_keep=neuraxial_muscle_keep,
        )
        if target != self.shiver_level:
            tau = SHIVER_TAU_ON if target > self.shiver_level else SHIVER_TAU_OFF
            self.shiver_level = clamp01(self.shiver_level + (target - self.shiver_level) * -math.expm1(-dt / tau))
        return self.shiver_level

    def step_temperature(self, dt: float, temp_c: float, warmer_target_c: float,
                         surface_cooling: bool = False, *,
                         oxygen_supply_fraction: float = 1.0,
                         sepsis_severity: float = 0.0,
                         perfusing: bool = True,
                         opioid_effect: float = 0.0,
                         extracorporeal_heat_w: float = 0.0,
                         extracorporeal_active: bool = False,
                         neuraxial_redistribution_depth: float = 0.0) -> float:
        """Return core temperature after metabolic heat, environmental loss, and redistribution.

        A forced-air warmer target of 0 °C is off. When extracorporeal support
        (ECMO or CPB) has been engaged, a coupled peripheral thermal compartment
        tracks limb/subcutaneous heat storage so CPB cooling, rewarming, and
        post-bypass core afterdrop (Rajek 1998; Rajek 2000) emerge naturally.
        """
        tuning = self.tuning
        depth_factor = clamp01(max(self.depth_index, neuraxial_redistribution_depth))

        aerobic_gate = clamp(oxygen_supply_fraction, 0.12, 1.0)
        production = (
            self.heat_production_basal * max(0.25, self.metabolic_factor) * aerobic_gate
            + self.extra_heat_w
        )
        conductance = self.awake_conductance + (self._anesthetized_conductance - self.awake_conductance) * depth_factor
        # Phase-3 thermoregulatory vasoconstriction (Sessler 2000; opioid shift in
        # Kurz 1995c; age shift in Ozaki 1997): below the vasoconstriction threshold
        # (~34.3 °C at 1 MAC in young adults without opioids), cutaneous constriction
        # limits further passive heat loss to establish a thermal plateau.
        vasoconst = 0.0
        if perfusing:
            vasoconst_threshold = (
                self._base_ref_temp
                - 0.3
                - (2.4 + self._age_vc_shift) * depth_factor
                - SHIVER_REMI_DROP_MAX * clamp01(opioid_effect)
            )
            if temp_c < vasoconst_threshold:
                vasoconst = clamp01(vasoconst_threshold - temp_c) * cold_vasoparesis(temp_c)
                conductance += (self._plateau_conductance - conductance) * vasoconst
        # Forced-air warming is modeled to shield 60% of BSA from room air while
        # delivering convective heat (calibrated to full-body transfer in Giesbrecht 1994).
        warmer_active = warmer_target_c > 0.0
        exposed_fraction = 0.40 if warmer_active else 1.0
        heat_loss = conductance * exposed_fraction * (temp_c - tuning.ambient_temp_c)

        # Active sweating and cutaneous vasodilation limit fever above the sweating
        # threshold (setpoint raised by sepsis; sweating threshold raised by anesthesia, Sessler 2000).
        sweating_w = 0.0
        if perfusing:
            febrile_setpoint = self._base_ref_temp + 1.8 * clamp01(sepsis_severity)
            sweat_threshold = febrile_setpoint + 0.25 + 1.2 * depth_factor
            if temp_c > sweat_threshold:
                sweating_w = (
                    140.0
                    * self._bsa_scale
                    * (1.0 - 0.45 * depth_factor)
                    * (temp_c - sweat_threshold)
                )

        # Anesthetic vasodilation moves core heat to the periphery over the first
        # hour (Matsukawa 1995a; epidural redistribution in Matsukawa 1995b).
        # Vasoconstriction on lightening traps it peripherally until normothermic awake recovery,
        # and zero-flow arrest halts convective redistribution.
        if perfusing:
            deficit_gap = self.redistribution_target_j(depth_factor) - self.redistributed_heat_j
            if deficit_gap > 0.0:
                redistribution_w = deficit_gap / tuning.redistribution_tau_s
                self.redistributed_heat_j += redistribution_w * dt
            else:
                redistribution_w = 0.0
                if depth_factor < 0.1 and temp_c >= self.baseline_temp - 0.5 and self.redistributed_heat_j > 0.0:
                    self.redistributed_heat_j *= math.exp(-dt / (2.0 * tuning.redistribution_tau_s))
                    if self.redistributed_heat_j < 1.0:
                        self.redistributed_heat_j = 0.0
        else:
            redistribution_w = 0.0

        warming = (
            tuning.bair_hugger_gain_w_per_c * self._bsa_scale * (warmer_target_c - temp_c)
            if warmer_active
            else 0.0
        )
        cooling = (
            tuning.surface_cooling_w_per_c * self._bsa_scale * max(0.0, temp_c - tuning.temp_min_c)
            if surface_cooling
            else 0.0
        )
        if perfusing:
            fluid_heat_j = self._pending_fluid_heat_j
            self._pending_fluid_heat_j = 0.0
        else:
            fluid_heat_j = 0.0

        if extracorporeal_active and not self._extracorporeal_thermal_engaged:
            # Initialize peripheral temperature slightly below core when anesthetized (~1 °C gradient).
            self.peripheral_temp_c = float(temp_c - 0.8 * depth_factor)
            self._extracorporeal_thermal_engaged = True

        if not self._extracorporeal_thermal_engaged:
            net_heat_w = production + warming - heat_loss - sweating_w - redistribution_w - cooling
            new_temp = clamp(
                temp_c + (net_heat_w * dt + fluid_heat_j) * self._inv_heat_capacity,
                tuning.temp_min_c,
                tuning.temp_max_c,
            )
            self.peripheral_temp_c = new_temp
            return new_temp

        # Two-compartment core (56%) and peripheral (44%) thermal balance during and after
        # extracorporeal heat exchange (model split informed by Rajek 1998 and Rajek 2000).
        c_core = 0.56 * self.heat_capacity
        c_periph = 0.44 * self.heat_capacity
        # Core-to-periphery conductance (W/°C): cold peripheral vasoconstriction limits
        # heat transfer during hypothermic CPB rewarming, while warm post-bypass
        # anesthetic vasodilation drives convective core-to-periphery afterdrop.
        if perfusing:
            periph_cold_vc = clamp01((35.5 - self.peripheral_temp_c) / 4.5)
            eff_vc = max(
                vasoconst,
                0.75 * periph_cold_vc if extracorporeal_active else 0.20 * periph_cold_vc,
            )
            k_cp = (24.0 + 28.0 * depth_factor * (1.0 - 0.65 * eff_vc)) * self._bsa_scale
        else:
            k_cp = 10.0 * self._bsa_scale
        min_temp = 18.0
        t_core = float(temp_c)
        t_periph = float(self.peripheral_temp_c)
        n_sub = max(1, math.ceil(dt / 2.0))
        sub_dt = dt / n_sub
        fluid_w = fluid_heat_j / dt if dt > 0.0 else 0.0
        for _ in range(n_sub):
            q_cp = k_cp * (t_core - t_periph)
            core_env_loss = 0.20 * conductance * exposed_fraction * (t_core - tuning.ambient_temp_c)
            periph_env_loss = 0.80 * conductance * exposed_fraction * (t_periph - tuning.ambient_temp_c)
            # Forced-air warming heats peripheral tissues directly and strongly reduces
            # post-CPB afterdrop (Rajek 2000: 0.5 °C vs 1.2 °C passive insulation).
            periph_warming = (
                1.45 * tuning.bair_hugger_gain_w_per_c * self._bsa_scale * (warmer_target_c - t_periph)
                if warmer_active
                else 0.0
            )
            core_w = (
                production
                + extracorporeal_heat_w
                + fluid_w
                - core_env_loss
                - sweating_w
                - q_cp
                - 0.30 * cooling
            )
            periph_w = q_cp + periph_warming - periph_env_loss - 0.70 * cooling
            t_core = clamp(t_core + (core_w * sub_dt) / c_core, min_temp, tuning.temp_max_c)
            t_periph = clamp(t_periph + (periph_w * sub_dt) / c_periph, 15.0, tuning.temp_max_c)
        self.peripheral_temp_c = t_periph
        return t_core
