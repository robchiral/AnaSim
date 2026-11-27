"""Massive pulmonary thromboembolism (PE) and venous air embolism (VAE).

Pulmonary thromboembolism obstruction thresholds and parallel alveolar dead
space are informed by McIntyre and Sasahara 1971 (angiographic obstruction vs
mPAP/RAP in men without prior cardiopulmonary disease) and Eriksson 1989
(synchronous dead space widens PaCO2-EtCO2 without increasing capnogram phase
III slope). Venous air embolism lethal volume, N2O bubble expansion
(1 / max(0.24, 1 - 0.94 * F_A_N2O)), 100% O2 nitrogen washout, Durant
positioning, and multiorifice CVC aspiration are calibrated from Mirski 2007
and Munson and Merrick 1966 (rabbit lethal air dose under 72-76% N2O).
"""

from __future__ import annotations

import math

from anasim.core.utils import clamp, clamp01

PE_ONSET_TAU_S = 25.0
PE_THROMBOLYSIS_TAU_S = 150.0
PE_HEPARIN_LYSIS_TAU_S = 1800.0
PE_RESIDUAL_AFTER_LYSIS = 0.08

VAE_N2O_TAU_S = 20.0
VAE_ASPIRATE_FRACTION = 0.45  # Multiorifice right-atrial catheter range 30-60% (Mirski 2007)
# Calibrated partial RVOT air-lock relief for traditional Durant positioning
# (Mirski 2007 notes mixed/negative hemodynamic evidence in canine studies).
VAE_DURANT_RELIEF_MULT = 0.78


class PulmonaryEmbolismModel:
    """Combined model for massive PE and venous air embolism (VAE)."""

    def __init__(self, reference_weight_kg: float = 70.0) -> None:
        # Scale central RV/PA air capacity with lean/reference weight rather than adipose mass.
        self.weight_kg = clamp(float(reference_weight_kg), 40.0, 110.0)
        # PE state
        self.pe_active: bool = False
        self.pe_target_obstruction: float = 0.0
        self.pe_obstruction: float = 0.0
        self.thrombolysis_active: bool = False
        self._thrombolysis_bleed_level: float = 0.0
        # VAE state
        self.vae_active: bool = False
        self.vae_entrainment_ml_min: float = 0.0
        self.vae_air_ml: float = 0.0
        self._vae_distal_ml: float = 0.0
        self._vae_aspirate_cooldown_s: float = 0.0
        self.vae_n2o_factor: float = 1.0
        self._durant_active: bool = False

    @property
    def thrombolysis_bleeding_multiplier(self) -> float:
        """Surgical hemorrhage amplification from systemic thrombolysis (calibrated from Marti 2015 OR 2.91)."""
        return 1.0 + 0.65 * clamp01(self._thrombolysis_bleed_level)

    @property
    def vae_effective_ml(self) -> float:
        """Effective intravascular gas volume after N2O bubble expansion (Munson and Merrick 1966)."""
        return self.vae_air_ml * self.vae_n2o_factor

    @property
    def vae_obstruction(self) -> float:
        """Effective pulmonary vascular / RVOT obstruction from intravascular air (0-0.90)."""
        if self.vae_air_ml <= 1e-4:
            return 0.0
        # Mirski 2007: ~3-5 mL/kg (~200-300 mL adult) causes severe RV air-lock / arrest.
        ref_lethal_ml = 3.5 * self.weight_kg
        pos_mult = VAE_DURANT_RELIEF_MULT if self._durant_active else 1.0
        return clamp((self.vae_effective_ml / ref_lethal_ml) * pos_mult * 0.92, 0.0, 0.90)

    @property
    def total_obstruction(self) -> float:
        """Combined pulmonary vascular obstruction from PE and VAE (0-0.90)."""
        pe = clamp(self.pe_obstruction, 0.0, 0.88)
        vae = self.vae_obstruction
        if pe <= 0.0 and vae <= 0.0:
            return 0.0
        return clamp(1.0 - (1.0 - pe) * (1.0 - vae), 0.0, 0.90)

    @property
    def pvr_factor(self) -> float:
        """Calibrated PVR multiplier (1.0-11.0); sharp rise above 25-40% obstruction (McIntyre and Sasahara 1971)."""
        obs = self.total_obstruction
        if obs <= 0.0:
            return 1.0
        excess = max(0.0, obs - 0.25)
        return clamp(1.0 + 1.5 * obs + 16.0 * (excess ** 1.8), 1.0, 11.0)

    @property
    def alveolar_deadspace(self) -> float:
        """Synchronous parallel alveolar dead space fraction (0-0.76; Eriksson 1989)."""
        obs = self.total_obstruction
        if obs <= 0.0:
            return 0.0
        return clamp(0.82 * obs, 0.0, 0.76)

    @property
    def extra_shunt(self) -> float:
        """Reflex V/Q redistribution and microvascular edema shunt (0-0.30)."""
        obs = self.total_obstruction
        if obs <= 0.0:
            return 0.0
        return clamp(0.32 * (obs ** 1.3), 0.0, 0.30)

    def start_pe(self, obstruction: float = 0.60) -> None:
        """Trigger acute pulmonary thromboembolism (obstruction 0.15-0.85)."""
        self.pe_active = True
        self.thrombolysis_active = False
        self.pe_target_obstruction = clamp(float(obstruction), 0.15, 0.85)

    def stop_pe(self) -> None:
        """Clear PE event and resolve clot."""
        self.pe_active = False
        self.thrombolysis_active = False
        self._thrombolysis_bleed_level = 0.0
        self.pe_target_obstruction = 0.0
        self.pe_obstruction = 0.0

    def give_thrombolysis(self) -> None:
        """Administer systemic thrombolysis (e.g. alteplase), accelerating PE clot lysis."""
        self._thrombolysis_bleed_level = 1.0
        if self.pe_obstruction > 0.0 or self.pe_active:
            self.thrombolysis_active = True
            self.pe_target_obstruction = min(
                self.pe_obstruction if self.pe_obstruction > 0.0 else self.pe_target_obstruction,
                PE_RESIDUAL_AFTER_LYSIS,
            )

    def start_vae(self, rate_ml_min: float = 55.0, *, initial_bolus_ml: float = 45.0) -> None:
        """Start continuous surgical venous air entrainment with an optional initial bolus."""
        self.vae_active = True
        self.vae_entrainment_ml_min = clamp(float(rate_ml_min), 0.0, 300.0)
        if initial_bolus_ml > 0.0:
            self.vae_air_ml = clamp(self.vae_air_ml + float(initial_bolus_ml), 0.0, 600.0)

    def stop_vae(self) -> None:
        """Stop surgical venous air entrainment (surgeon floods field / lowers surgical site)."""
        self.vae_active = False
        self.vae_entrainment_ml_min = 0.0

    def clear_vae(self) -> None:
        """Clear all intravascular air immediately."""
        self.vae_active = False
        self.vae_entrainment_ml_min = 0.0
        self.vae_air_ml = 0.0
        self._vae_distal_ml = 0.0
        self._vae_aspirate_cooldown_s = 0.0
        self.vae_n2o_factor = 1.0

    def aspirate_vae(self) -> float:
        """Aspirate right-atrial air via multiorifice CVC; returns mL of air removed."""
        if self._vae_aspirate_cooldown_s > 0.0 or self.vae_air_ml <= 0.1:
            return 0.0
        aspirable_ml = max(0.0, self.vae_air_ml - self._vae_distal_ml)
        if aspirable_ml <= 0.1:
            return 0.0
        removed = aspirable_ml * VAE_ASPIRATE_FRACTION
        self.vae_air_ml = max(0.0, self.vae_air_ml - removed)
        # Unrecovered fraction of the bolus lies distally in the RVOT/PA beyond the RA catheter tip.
        self._vae_distal_ml = min(self.vae_air_ml, self._vae_distal_ml + (aspirable_ml - removed))
        self._vae_aspirate_cooldown_s = 10.0
        return removed

    def step(
        self,
        dt: float,
        *,
        p_alv_n2o: float = 0.0,
        alveolar_o2_fraction: float = 0.21,
        position: str = "supine",
        heparin_u_ml: float = 0.0,
        act_s: float = 120.0,
        circulation: float = 1.0,
        pulmonary_circulation: float | None = None,
        pulmonary_ventilation: bool = True,
    ) -> None:
        """Advance PE clot kinetics and VAE entrainment, N2O expansion, and washout."""
        if dt <= 0.0:
            return
        self._durant_active = position == "left_lateral"
        circ = clamp01(circulation)
        pulm_circ = circ if pulmonary_circulation is None else clamp01(pulmonary_circulation)
        if self._vae_aspirate_cooldown_s > 0.0:
            self._vae_aspirate_cooldown_s = max(0.0, self._vae_aspirate_cooldown_s - dt)
        if self._thrombolysis_bleed_level > 0.0 and circ > 0.0:
            self._thrombolysis_bleed_level *= math.exp(-dt * circ / 1800.0)
            if self._thrombolysis_bleed_level < 1e-4:
                self._thrombolysis_bleed_level = 0.0

        # Pulmonary thromboembolism kinetics (drug delivery and clot lysis require circulation).
        if self.thrombolysis_active:
            if circ > 0.0:
                target = min(self.pe_obstruction, PE_RESIDUAL_AFTER_LYSIS)
                self.pe_obstruction += (target - self.pe_obstruction) * -math.expm1(
                    -dt * circ / PE_THROMBOLYSIS_TAU_S
                )
                if self.pe_obstruction <= PE_RESIDUAL_AFTER_LYSIS + 0.015:
                    self.pe_active = False
                    self.thrombolysis_active = False
        elif self.pe_active:
            therapeutic_hep = heparin_u_ml >= 0.25
            if therapeutic_hep and circ > 0.0 and self.pe_target_obstruction > PE_RESIDUAL_AFTER_LYSIS:
                # Therapeutic anticoagulation halts propagation and permits slow endogenous lysis.
                self.pe_target_obstruction += (
                    (PE_RESIDUAL_AFTER_LYSIS - self.pe_target_obstruction)
                    * -math.expm1(-dt * circ / PE_HEPARIN_LYSIS_TAU_S)
                )
            if self.pe_obstruction < self.pe_target_obstruction and circ > 0.0:
                self.pe_obstruction += (
                    (self.pe_target_obstruction - self.pe_obstruction) * -math.expm1(-dt * circ / PE_ONSET_TAU_S)
                )
            elif self.pe_obstruction > self.pe_target_obstruction and circ > 0.0:
                self.pe_obstruction += (
                    (self.pe_target_obstruction - self.pe_obstruction)
                    * -math.expm1(-dt * circ / PE_HEPARIN_LYSIS_TAU_S)
                )
        elif self.pe_obstruction > 0.0 and circ > 0.0:
            self.pe_obstruction *= math.exp(-dt * circ / 600.0)
            if self.pe_obstruction < 1e-4:
                self.pe_obstruction = 0.0

        # Venous air embolism kinetics (Munson and Merrick 1966; Mirski 2007).
        f_n2o = clamp(p_alv_n2o, 0.0, 0.80)
        # Munson and Merrick 1966 (rabbit, 72-76% N2O): ~3.4x lethal-dose reduction -> 1/(1 - 0.94*F_N2O).
        target_n2o_factor = 1.0 / max(0.24, 1.0 - 0.94 * f_n2o)
        if pulm_circ > 0.0:
            self.vae_n2o_factor += (target_n2o_factor - self.vae_n2o_factor) * -math.expm1(
                -dt * pulm_circ / VAE_N2O_TAU_S
            )

        if self.vae_active or self.vae_air_ml > 0.0:
            inflow_ml_s = (self.vae_entrainment_ml_min / 60.0) if (self.vae_active and circ > 0.0) else 0.0
            if pulm_circ > 0.0 and pulmonary_ventilation:
                # 100% O2 widens the N2 washout gradient across the alveolar membrane (Mirski 2007);
                # tau_clear = 330 s on 100% O2 vs 720 s on air (calibrated to canine recovery times).
                tau_clear = 720.0 - 390.0 * clamp01((alveolar_o2_fraction - 0.21) / 0.79)
                k_clear = pulm_circ / max(60.0, tau_clear)
                decay = math.exp(-k_clear * dt)
                inflow_integral = -math.expm1(-k_clear * dt) / k_clear
                self.vae_air_ml = clamp(self.vae_air_ml * decay + inflow_ml_s * inflow_integral, 0.0, 600.0)
                self._vae_distal_ml = min(self.vae_air_ml, self._vae_distal_ml * decay)
            elif inflow_ml_s > 0.0:
                self.vae_air_ml = clamp(self.vae_air_ml + inflow_ml_s * dt, 0.0, 600.0)
            if not self.vae_active and self.vae_air_ml < 0.05:
                self.vae_air_ml = 0.0
                self._vae_distal_ml = 0.0
