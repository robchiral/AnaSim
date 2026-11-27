"""Heparin-protamine anticoagulation, ACT, heparin rebound, and protamine hemodynamic reactions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from anasim.core.utils import clamp, clamp01, hill_function
from anasim.patient.pk_models import HeparinPK, ProtaminePK

# Stoichiometry: 1 mg protamine neutralizes ~100 U unfractionated heparin in plasma.
HEPARIN_UNITS_PER_MG_PROTAMINE = 100.0
BASELINE_ACT_S = 120.0
MAX_ACT_S = 999.0
ACT_ANALYSIS_TIME_S = 60.0


@dataclass(frozen=True, slots=True)
class ACTResult:
    """Point-of-care activated clotting time result."""

    drawn_at: float
    act_s: float

    @property
    def ready_at(self) -> float:
        return self.drawn_at + ACT_ANALYSIS_TIME_S


@dataclass(slots=True)
class CoagulationState:
    """Live anticoagulation and protamine hemodynamic-reaction state."""

    heparin_u_ml: float = 0.0  # Free circulating plasma heparin (U/mL)
    protamine_ug_ml: float = 0.0  # Free circulating plasma protamine (mcg/mL = mg/L)
    act_s: float = BASELINE_ACT_S  # Activated clotting time (s)
    protamine_svr_factor: float = 1.0  # Type I systemic vasodilation multiplier (<= 1.0)
    protamine_venous_pool_frac: float = 0.0  # Type I unstressed venous capacitance shift (0-0.18)
    protamine_pvr_factor: float = 1.0  # Pulmonary vascular resistance multiplier (>= 1.0)
    protamine_reaction_active: bool = False  # Catastrophic Type III pulmonary vasoconstriction event
    bleeding_multiplier: float = 1.0  # Surgical hemorrhage amplification from anticoagulation (>= 1.0)


class CoagulationModel:
    """Coupled heparin-protamine plasma neutralization, ACT, and protamine hemodynamic effects.

    References:
    - Bull 1975: celite ACT dose-response framework (~480 s CPB target, early hemodilution shift).
    - Bjornsson 1982: saturable dose-dependent UFH clearance across 25-75 U/kg in healthy adults.
    - Butterworth 2002: rapid protamine elimination (median t1/2 4.5 min post-CPB).
    - Teoh 2004: post-reversal heparin rebound from peripheral redistribution.
    - Mochizuki 1998: in-vitro excess free protamine (>1.3:1) paradoxically prolongs ACT.
    - Horrow 1985: Type I-III protamine reaction taxonomy and rate-dependent Type I vasodilation.
    - Lowenstein 1983: clinical catastrophic pulmonary vasoconstriction and acute RV failure.
    """

    # Second-order plasma binding rate: L / (mg * min). At typical concentrations
    # (~20-40 mg/L), neutralization completes within ~1-2 min while leaving a brief
    # acute exposure window when protamine is pushed as a rapid IV bolus.
    BINDING_RATE_L_MG_MIN = 0.45

    def __init__(self, *, act_sensitivity_scale: float = 1.0) -> None:
        self.baseline_act_s = BASELINE_ACT_S
        self.act_sensitivity_scale = max(0.35, float(act_sensitivity_scale))
        self.state = CoagulationState(act_s=self.baseline_act_s)
        # Transient exposure tracker for Type I rate-dependent vasodilation: captures
        # acute bolus peaks of protamine entering plasma even before heparin binds it.
        self._vaso_exposure_ug_ml: float = 0.0
        self._pending_bolus_vaso_ug_ml: float = 0.0
        self._complex_rate_mg_min: float = 0.0
        self.cumulative_protamine_mg: float = 0.0
        # Type III catastrophic reaction can be triggered directly or armed for next protamine dose.
        self.reaction_armed: bool = False
        self.reaction_active: bool = False
        self.reaction_severity: float = 0.0

    def reset(self) -> None:
        self.state = CoagulationState(act_s=self.baseline_act_s)
        self._vaso_exposure_ug_ml = 0.0
        self._pending_bolus_vaso_ug_ml = 0.0
        self._complex_rate_mg_min = 0.0
        self.cumulative_protamine_mg = 0.0
        self.reaction_armed = False
        self.reaction_active = False
        self.reaction_severity = 0.0

    @classmethod
    def neutralize_step_mg(
        cls,
        m_hep_mg_eq: float,
        m_prot_mg: float,
        v_mix_l: float,
        dt_min: float,
        circulation: float = 1.0,
    ) -> float:
        """Exact second-order heparin-protamine mass neutralized (mg protamine eq) over dt_min."""
        if m_hep_mg_eq <= 0.0 or m_prot_mg <= 0.0 or dt_min <= 0.0 or circulation <= 0.0:
            return 0.0
        rate = cls.BINDING_RATE_L_MG_MIN * clamp01(circulation) / max(0.5, v_mix_l)
        x = min(m_hep_mg_eq, m_prot_mg)
        y = max(m_hep_mg_eq, m_prot_mg)
        delta = y - x
        if delta <= 1e-9 * y:
            rxh = rate * x * dt_min
            return x * rxh / (1.0 + rxh)
        om_e = -math.expm1(-rate * delta * dt_min)
        return (x * y * om_e) / (delta + x * om_e)

    def record_protamine_bolus(self, amount_mg: float, v1_l: float, *, circulation: float = 1.0) -> None:
        """Register an acute IV protamine bolus into the Type I rate-dependent exposure pool."""
        if amount_mg <= 0.0:
            return
        self.cumulative_protamine_mg += amount_mg
        delta_conc = amount_mg / max(1.0, v1_l)
        circ = clamp01(circulation)
        self._vaso_exposure_ug_ml += delta_conc * circ
        self._pending_bolus_vaso_ug_ml += delta_conc * (1.0 - circ)
        if self.reaction_armed and circ >= 0.05:
            self.reaction_active = True
            self.reaction_armed = False

    def compute_act(
        self,
        heparin_u_ml: float,
        free_protamine_ug_ml: float,
        *,
        temp_c: float = 37.0,
        hematocrit: float = 0.42,
        baseline_hct: float = 0.42,
    ) -> float:
        """Compute activated clotting time (s) from heparin, excess protamine, dilution, and temp."""
        hep = max(0.0, heparin_u_ml)
        prot = max(0.0, free_protamine_ug_ml)

        # Heparin prolongation (calibrated to Bull 1975 and Delavenne 2017): ~105 s per U/mL
        # at clinical concentrations, gently saturating toward the 999 s analyzer cap.
        hep_act = (
            105.0 * self.act_sensitivity_scale * hep / (1.0 + 0.025 * hep)
            if hep > 0.0
            else 0.0
        )

        # Excess free protamine paradox (in-vitro blood data in Mochizuki 1998): only protamine
        # above stoichiometric heparin neutralization independently prolongs ACT.
        excess_prot = max(0.0, prot - hep * (1000.0 / HEPARIN_UNITS_PER_MG_PROTAMINE))
        prot_act = 95.0 * hill_function(excess_prot, 26.0, 1.8) if excess_prot > 0.5 else 0.0

        # Early hemodilution shift (Bull 1975) and calibrated mild hypothermia ACT prolongation,
        # blended smoothly so un-heparinized baseline ACT stays exact without a step jump.
        anticoag_blend = clamp01(max(hep / 0.15, excess_prot / 1.5))
        hct_drop = max(0.0, min(0.32, baseline_hct) - hematocrit)
        dilution_act = 120.0 * hct_drop * anticoag_blend if hct_drop > 0.0 and anticoag_blend > 0.0 else 0.0
        cold_dt = max(0.0, 35.0 - temp_c)
        temp_act = 3.0 * cold_dt * anticoag_blend if cold_dt > 0.0 and anticoag_blend > 0.0 else 0.0

        return clamp(
            self.baseline_act_s + hep_act + prot_act + dilution_act + temp_act,
            self.baseline_act_s,
            MAX_ACT_S,
        )

    def step(
        self,
        dt: float,
        heparin_pk: HeparinPK,
        protamine_pk: ProtaminePK,
        *,
        heparin_rate_u_s: float = 0.0,
        protamine_rate_mg_s: float = 0.0,
        temp_c: float = 37.0,
        hematocrit: float = 0.42,
        baseline_hct: float = 0.42,
        circulation: float = 1.0,
    ) -> CoagulationState:
        """Advance heparin PK, protamine PK, plasma neutralization, ACT, and protamine hemodynamics."""
        circ = clamp01(circulation)
        if circ <= 0.0:
            # Zero-flow arrest isolates arterial coagulation readings from peripheral IV doses
            # until circulation resumes; queued infusions still accumulate in C1 and the
            # pending Type I vasodilation exposure buffer.
            if dt > 0.0:
                if heparin_rate_u_s > 0.0:
                    heparin_pk.step(dt, heparin_rate_u_s)
                if protamine_rate_mg_s > 0.0:
                    self.cumulative_protamine_mg += protamine_rate_mg_s * dt
                    protamine_pk.step(dt, protamine_rate_mg_s)
                    infusion_delta = (protamine_rate_mg_s * dt) / max(1.0, protamine_pk.v1)
                    self._pending_bolus_vaso_ug_ml += 0.35 * infusion_delta
            return self.state

        if self._pending_bolus_vaso_ug_ml > 0.0 and dt > 0.0:
            # Flush queued arrest/low-flow bolus exposure continuously with circulation
            # so low-flow ROSC or CPB weaning neither deletes dose nor diverges with dt.
            flush_frac = -math.expm1(-dt * circ / 10.0)
            transfer = self._pending_bolus_vaso_ug_ml * flush_frac
            self._vaso_exposure_ug_ml += transfer
            self._pending_bolus_vaso_ug_ml = max(0.0, self._pending_bolus_vaso_ug_ml - transfer)
            if self._pending_bolus_vaso_ug_ml < 1e-6:
                self._pending_bolus_vaso_ug_ml = 0.0
            if self.reaction_armed and self._vaso_exposure_ug_ml > 0.25:
                self.reaction_active = True
                self.reaction_armed = False

        has_hep = heparin_rate_u_s > 0.0 or heparin_pk.has_mass()
        has_prot = protamine_rate_mg_s > 0.0 or protamine_pk.has_mass()
        has_activity = (
            has_hep
            or has_prot
            or self._vaso_exposure_ug_ml > 1e-4
            or self._complex_rate_mg_min > 1e-4
            or self.reaction_active
            or self.reaction_severity > 1e-4
        )
        if not has_activity:
            st = self.state
            st.heparin_u_ml = 0.0
            st.protamine_ug_ml = 0.0
            st.act_s = self.baseline_act_s
            st.protamine_svr_factor = 1.0
            st.protamine_venous_pool_frac = 0.0
            st.protamine_pvr_factor = 1.0
            st.protamine_reaction_active = False
            st.bleeding_multiplier = 1.0
            return st

        if self.reaction_armed and (
            (protamine_rate_mg_s > 0.0 and circ >= 0.05)
            or protamine_pk.state.c1 * circ > 0.5
        ):
            self.reaction_active = True
            self.reaction_armed = False

        # Substep when both heparin and protamine are interacting so PK and second-order binding stay dt-invariant.
        interacting = has_hep and has_prot
        substeps = max(1, math.ceil(dt / 0.25)) if (interacting and dt > 0.25) else 1
        h = dt / substeps
        h_min = h / 60.0

        for _ in range(substeps):
            if has_hep:
                heparin_pk.update_nonlinear_clearance()
                heparin_pk.step(h, heparin_rate_u_s)
            if has_prot:
                protamine_pk.step(h, protamine_rate_mg_s)

            # Track acute protamine input exposure (Type I rate-dependent vasodilation).
            # Slow infusions (e.g., <= 15-25 mg/min) stay below the vasodilation knee
            # because endothelial/histamine clearance (tau ~ 110 s) dissipates input,
            # whereas rapid boluses spike _vaso_exposure_ug_ml sharply.
            if protamine_rate_mg_s > 0.0:
                self.cumulative_protamine_mg += protamine_rate_mg_s * h
                infusion_delta = 0.35 * (protamine_rate_mg_s * h) / max(1.0, protamine_pk.v1)
                self._vaso_exposure_ug_ml += infusion_delta * circ
                self._pending_bolus_vaso_ug_ml += infusion_delta * (1.0 - circ)
            if h > 0.0 and self._vaso_exposure_ug_ml > 0.0:
                self._vaso_exposure_ug_ml *= math.exp(-h * circ / 110.0)
                if self._vaso_exposure_ug_ml < 1e-5:
                    self._vaso_exposure_ug_ml = 0.0

            # Exact second-order mass-conserving heparin-protamine neutralization in plasma.
            bound_this_substep = 0.0
            if heparin_pk.state.c1 > 0.0 and protamine_pk.state.c1 > 0.0 and h_min > 0.0:
                v_hep = max(0.5, heparin_pk.v1)
                v_prot = max(0.5, protamine_pk.v1)
                m_hep_mg_eq = (heparin_pk.state.c1 * v_hep) / HEPARIN_UNITS_PER_MG_PROTAMINE
                m_prot_mg = protamine_pk.state.c1 * v_prot
                v_mix = 0.5 * (v_hep + v_prot)
                d_mg = self.neutralize_step_mg(m_hep_mg_eq, m_prot_mg, v_mix, h_min, circ)
                if d_mg > 0.0:
                    heparin_pk.state.c1 = max(
                        0.0, heparin_pk.state.c1 - (d_mg * HEPARIN_UNITS_PER_MG_PROTAMINE) / v_hep
                    )
                    protamine_pk.state.c1 = max(0.0, protamine_pk.state.c1 - d_mg / v_prot)
                    bound_this_substep = d_mg

            if h > 0.0:
                inst_complex_rate = bound_this_substep / h_min
                tau_c = 15.0 if inst_complex_rate > self._complex_rate_mg_min else 90.0
                self._complex_rate_mg_min += (inst_complex_rate - self._complex_rate_mg_min) * (
                    -math.expm1(-h * circ / tau_c)
                )
                if self._complex_rate_mg_min < 1e-5:
                    self._complex_rate_mg_min = 0.0

        # Catastrophic Type III pulmonary vasoconstriction dynamics (Lowenstein 1983).
        if dt > 0.0:
            target_sev = 1.0 if self.reaction_active else 0.0
            if target_sev != self.reaction_severity:
                tau_r = 25.0 if target_sev > self.reaction_severity else 180.0
                self.reaction_severity += (target_sev - self.reaction_severity) * (
                    -math.expm1(-dt * circ / tau_r)
                )
                if self.reaction_severity < 1e-4 and not self.reaction_active:
                    self.reaction_severity = 0.0

        hep_u_ml = max(0.0, heparin_pk.state.c1 / 1000.0)
        prot_ug_ml = max(0.0, protamine_pk.state.c1)
        act_s = self.compute_act(
            hep_u_ml,
            prot_ug_ml,
            temp_c=temp_c,
            hematocrit=hematocrit,
            baseline_hct=baseline_hct,
        )

        # Type I rate-dependent vasodilation (Horrow 1985):
        # Slow infusions produce minimal SVR loss, while rapid boluses drop SVR
        # 25-45% and pool venous capacitance.
        vaso_drive = max(self._vaso_exposure_ug_ml, 0.55 * protamine_pk.state.ce)
        vaso_hill = hill_function(max(0.0, vaso_drive - 4.0), 16.0, 2.0) if vaso_drive > 4.0 else 0.0
        svr_factor = 1.0 - 0.45 * vaso_hill
        venous_pool_frac = 0.14 * vaso_hill

        # Pulmonary vascular resistance: mild transient complex-rate bump + severe Type III crisis.
        mild_pvr = 0.45 * hill_function(self._complex_rate_mg_min, 55.0, 1.8) if self._complex_rate_mg_min > 5.0 else 0.0
        catastrophic_pvr = 3.2 * self.reaction_severity
        pvr_factor = 1.0 + mild_pvr + catastrophic_pvr

        # Therapeutic anticoagulation or severe protamine excess amplifies active surgical bleeding.
        bleeding_mult = 1.0 + 0.40 * clamp01((act_s - 150.0) / 330.0)

        st = self.state
        st.heparin_u_ml = hep_u_ml
        st.protamine_ug_ml = prot_ug_ml
        st.act_s = act_s
        st.protamine_svr_factor = clamp(svr_factor, 0.45, 1.0)
        st.protamine_venous_pool_frac = clamp(venous_pool_frac, 0.0, 0.18)
        st.protamine_pvr_factor = clamp(pvr_factor, 1.0, 4.8)
        st.protamine_reaction_active = self.reaction_active
        st.bleeding_multiplier = bleeding_mult
        return st
