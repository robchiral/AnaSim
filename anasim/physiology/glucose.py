"""Coupled glucose homeostasis, endogenous insulin, counterregulation, and renal glycosuria.

Extracellular glucose mass (mmol) lives in FluidBalance so dextrose infusions,
packed red cell transfusions, hemorrhage, osmotic water shifts, and metabolic
fluxes share one conserved pool. Exogenous regular insulin PK is handled by
InsulinPK; this module integrates endogenous pancreatic insulin, fast hepatic
and slow remote interstitial insulin action, hepatic glucose production,
insulin-independent and insulin-mediated peripheral disposal, hypoglycemic
counterregulation (glucagon and endogenous epinephrine), renal glycosuria,
and insulin-mediated potassium shift.

Primary references are documented in docs/REFERENCES.md#glucose-and-insulin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from anasim.core.utils import clamp, clamp01, hill_function

MG_PER_MMOL = 180.16
MG_DL_PER_MMOL_L = MG_PER_MMOL / 10.0  # 18.016 mg/dL per mmol/L
GLUCOSE_ANALYSIS_TIME_S = 10.0  # Bedside US OR point-of-care glucometer turnaround

# Minimum extracellular glucose concentration floor (mg/dL) to prevent negative pools.
_MIN_GLUCOSE_MG_DL = 5.0
# Hypoglycemic counterregulatory threshold (Mitrakou 1991: 68 +/- 1 mg/dL in healthy volunteers).
_COUNTERREG_THRESHOLD_MG_DL = 68.0
# Fast hepatic insulin equilibration (Rizza 1981; Sherwin 1974).
_HEPATIC_INSULIN_TAU_MIN = 6.0
# Calibrated remote interstitial peripheral insulin action (Sherwin 1974 Model A: L13 = 0.020 min^-1, tau = 50 min).
_REMOTE_INSULIN_TAU_MIN = 40.0
# Calibrated transcellular potassium insulin equilibration (Alvestrand 1984; Allon and Copkney 1990).
_K_INSULIN_TAU_MIN = 22.0


@dataclass(frozen=True, slots=True)
class GlucoseResult:
    """Point-of-care blood glucose meter reading (mg/dL)."""

    drawn_at: float
    glucose: float  # mg/dL

    @property
    def ready_at(self) -> float:
        return self.drawn_at + GLUCOSE_ANALYSIS_TIME_S


@dataclass(slots=True)
class GlucoseState:
    """Live metabolic glucose and insulin state."""

    glucose_mg_dl: float = 90.0  # Whole-body extracellular glucose (mg/dL)
    arterial_glucose_mg_dl: float = 90.0  # Arterial/POC glucose; frozen during zero-flow arrest
    insulin_mu_l: float = 10.0  # Arterial plasma insulin (endogenous + exogenous), mU/L = µU/mL
    endogenous_insulin_mu_l: float = 10.0  # Endogenous plasma insulin, mU/L
    hepatic_insulin_mu_l: float = 10.0  # Fast hepatic insulin pool, mU/L
    remote_insulin_mu_l: float = 10.0  # Remote interstitial insulin pool, mU/L
    glucagon_drive: float = 0.0  # Counterregulatory glucagon response above basal (0-1)
    hypoglycemic_epi_ng_ml: float = 0.0  # Endogenous hypoglycemic epinephrine surge (ng/mL)
    insulin_k_uptake: float = 0.0  # Dimensionless additive K+ uptake drive for FluidBalance
    glycosuria_mg_min: float = 0.0  # Urinary glucose excretion (mg/min)
    osmotic_diuresis_ml_min: float = 0.0  # Excess urine flow from glycosuria (mL/min)
    egp_mg_kg_min: float = 0.0  # Endogenous glucose production (mg/kg lean/min)
    rd_mg_kg_min: float = 0.0  # Total peripheral + CNS glucose disposal (mg/kg lean/min)


class GlucoseModel:
    """Whole-body glucose-insulin-counterregulation model coupled to extracellular fluid.

    Key quantitative anchors (model calibrations separated from primary trial values):
    - Sherwin 1974 / Rabkin 1970 / Letiexhe 1993: insulin MCR (~11.3 mL/kg/min) and
      remote interstitial compartment, with a calibrated 55% hepatic / 27% renal /
      18% peripheral clearance split.
    - Rizza 1981: fast hepatic EGP suppression (paper ED50 29 +/- 2 µU/mL) vs slower
      peripheral Rd (paper ED50 55 +/- 7 µU/mL, total Rmax 9.6 +/- 0.7 mg/kg/min).
    - Kolterman 1981: T2D hepatic/peripheral insulin resistance (-48% to -60% Rd_max,
      elevated basal HGO; blunted beta-cell secretion is a model calibration).
    - Alvestrand 1984 / Allon and Copkney 1990: insulin-mediated K+ uptake preserved
      in chronic uremia despite -44% glucose disposal.
    - Rizza 1979 / Mitrakou 1991: counterregulatory glucagon and epinephrine onset
      near 68 mg/dL rescuing modest insulin hypoglycemia.
    - Clutter 1980 / Marker 1998: epinephrine glycogenolysis (>150-200 pg/mL),
      clearance suppression, insulin suppression (>400 pg/mL), and age attenuation.
    - DeFronzo 2013: higher renal TmG in T2D (420 vs 317 mg/min excluding one
      outlier; 160/185 mg/dL onset and osmotic diuresis are model calibrations).
    - Tanaka 2005 / Schricker 2004 / Revelly 2005 / Lehot 1992: volatile suppression
      of stimulated insulin secretion across 0.5-1.5 MAC, surgical hepatic EGP
      resistance, septic 2x basal Ra with insulin resistance, and hypothermic CPB
      cooling suppression with rewarming rebound.
    """

    def __init__(
        self,
        *,
        lean_mass_kg: float,
        ecf_l: float,
        fluids=None,
        baseline_glucose_mg_dl: float = 90.0,
        type2_diabetes: bool = False,
        age: float = 40.0,
        bmi: float = 24.2,
        renal_function: float = 1.0,
        renal_clearance_factor: float = 1.0,
        hepatic_function: float = 1.0,
    ) -> None:
        self.lean_mass_kg = max(20.0, float(lean_mass_kg))
        self.baseline_glucose_mg_dl = float(baseline_glucose_mg_dl)
        self.type2_diabetes = bool(type2_diabetes)
        self.age = float(age)
        self.bmi = float(bmi)
        self.renal_function = clamp(float(renal_function), 0.05, 1.0)
        self.renal_clearance_factor = max(0.05, float(renal_clearance_factor))
        self.hepatic_function = clamp(float(hepatic_function), 0.5, 1.0)
        self._fluids = fluids

        # Standalone fallback pool when FluidBalance is not attached.
        self._standalone_ecf_l = max(1.0, float(ecf_l))
        self._standalone_glucose_mmol = (self.baseline_glucose_mg_dl / MG_DL_PER_MMOL_L) * self._standalone_ecf_l

        # Calibrated organ insulin clearance split informed by Sherwin 1974, Rabkin 1970, and Letiexhe 1993.
        self._organ_cl_scale = (
            0.55 * self.hepatic_function + 0.27 * self.renal_clearance_factor + 0.18
        )
        self._insulin_v1_l = 0.045 * self.lean_mass_kg
        self._insulin_mcr_base_l_min = 0.251 * self._insulin_v1_l * self._organ_cl_scale

        # Insulin sensitivity parameters.
        # Kolterman 1981: nonobese T2D -48% Rd_max, obese T2D -60% Rd_max; T2D ED50 shift is calibrated.
        obese_frac = clamp01((self.bmi - 30.0) / 15.0)
        t2d_vmax = (0.52 - 0.10 * obese_frac) if self.type2_diabetes else (1.0 - 0.12 * obese_frac)
        # Alvestrand 1984: chronic uremia lowers glucose uptake M by 44% while sparing K+.
        uremic_vmax = 1.0 - 0.44 * clamp01((0.50 - self.renal_function) / 0.45)
        # Letiexhe 1993: Child-Pugh A/B cirrhosis lowers SI by ~52%.
        cirrhotic_vmax = 1.0 - 0.48 * clamp01((1.0 - self.hepatic_function) / 0.50)
        self._periph_vmax_scale = max(0.20, t2d_vmax * uremic_vmax * cirrhotic_vmax)
        self._rd_vmax = 8.2  # mg/kg/min insulin-stimulated peripheral Vmax (calibrated to Rizza 1981)
        self._rd_ed50 = 128.0 if self.type2_diabetes else 72.0  # µU/mL (calibrated from Rizza 1981)
        self._egp_ed50 = 64.0 if self.type2_diabetes else 25.0  # µU/mL (calibrated from Rizza 1981 / Kolterman 1981)

        # Calibrated from Marker 1998 Figs. 1-2: reduced glycemic sensitivity per unit epinephrine with age.
        self._age_epi_sensitivity = 1.0 - 0.35 * clamp01((self.age - 30.0) / 45.0)

        # Normoglycemic basal insulin (µU/mL = mU/L) and full pancreatic secretion capacity.
        # Kept independent of starting hypoglycemia so dextrose rescue restores normal beta-cell response.
        base_i = (18.0 + 10.0 * obese_frac) if self.type2_diabetes else 10.0
        cl_compensation = clamp(1.0 / max(0.55, self._organ_cl_scale), 1.0, 1.8)
        self.basal_insulin_mu_l = base_i * cl_compensation
        self._basal_secretion_mu_min = self.basal_insulin_mu_l * self._insulin_mcr_base_l_min
        # Initial dynamic insulin is gated when starting hypoglycemic (matching dynamic shutoff below 82 mg/dL).
        hypo_init_gate = clamp01((self.baseline_glucose_mg_dl - 52.0) / 30.0) if self.baseline_glucose_mg_dl < 82.0 else 1.0
        self._init_insulin_mu_l = self.basal_insulin_mu_l * hypo_init_gate
        # Calibrated beta-cell glucose-stimulated secretion ceiling (blunted in T2D).
        self._beta_stim_max = 2.2 if self.type2_diabetes else 8.5

        # Calibrate resting hepatic EGP so normoglycemic/diabetic baselines start in exact equilibrium.
        self._g_setpoint_mg_dl = max(70.0, self.baseline_glucose_mg_dl)
        rest_rd = self._compute_rd_mg_kg_min(
            self._g_setpoint_mg_dl,
            self.basal_insulin_mu_l,
            epi_met=0.0,
            sepsis_severity=0.0,
            surg_stress=0.0,
            temp_q10=1.0,
        )
        rest_uge_mg_kg_min = self._compute_glycosuria_mg_min(
            self._g_setpoint_mg_dl, renal_perfusion=1.0,
        ) / self.lean_mass_kg
        rest_egp_needed = rest_rd + rest_uge_mg_kg_min
        rest_supp = self._hepatic_insulin_suppression(self.basal_insulin_mu_l)
        self._egp_uninf_cap = rest_egp_needed / max(0.08, 1.0 - rest_supp)
        self._rest_egp_mg_kg_min = rest_egp_needed

        # Dynamic internal states.
        self._endo_insulin_c1 = self._init_insulin_mu_l
        self._hepatic_insulin = self._init_insulin_mu_l
        self._remote_insulin = self._init_insulin_mu_l
        self._k_insulin = self._init_insulin_mu_l
        self._arterial_insulin_mu_l = self._init_insulin_mu_l
        self._glucagon_drive = 0.0
        self._hypo_epi_ng_ml = 0.0
        self._epi_glycogen_adapt = 0.0
        self._surg_stress = 0.0
        self._circulation = 1.0
        self._arterial_glucose_mg_dl = self.baseline_glucose_mg_dl

        init_uge = self._compute_glycosuria_mg_min(self.baseline_glucose_mg_dl, renal_perfusion=1.0)
        init_egp = self._compute_egp_mg_kg_min(
            self.baseline_glucose_mg_dl,
            self._init_insulin_mu_l,
            glucagon_drive=0.0,
            epi_transient=0.0,
            sepsis_severity=0.0,
            surg_stress=0.0,
            oxygen_supply=1.0,
            temp_q10=1.0,
        )
        init_rd = self._compute_rd_mg_kg_min(
            self.baseline_glucose_mg_dl,
            self._init_insulin_mu_l,
            epi_met=0.0,
            sepsis_severity=0.0,
            surg_stress=0.0,
            temp_q10=1.0,
        )
        self.state = GlucoseState(
            glucose_mg_dl=self.baseline_glucose_mg_dl,
            arterial_glucose_mg_dl=self.baseline_glucose_mg_dl,
            insulin_mu_l=self._init_insulin_mu_l,
            endogenous_insulin_mu_l=self._init_insulin_mu_l,
            hepatic_insulin_mu_l=self._init_insulin_mu_l,
            remote_insulin_mu_l=self._init_insulin_mu_l,
            egp_mg_kg_min=init_egp,
            rd_mg_kg_min=init_rd,
            glycosuria_mg_min=init_uge,
            osmotic_diuresis_ml_min=self._osmotic_diuresis_ml_min(init_uge),
        )

    @staticmethod
    def _osmotic_diuresis_ml_min(uge_mg_min: float) -> float:
        """Calibrated excess urine flow from glycosuria (~6 mL water per mmol excreted glucose)."""
        if uge_mg_min <= 0.0:
            return 0.0
        return min(15.0, 6.0 * (uge_mg_min / MG_PER_MMOL))

    def reset_to_baseline(self, fluids=None) -> None:
        """Reset glucose pool and hormonal states back to configured baseline (for hidden settle)."""
        target_fluids = self._fluids if fluids is None else fluids
        if target_fluids is not None:
            self._fluids = target_fluids
            target_fluids.glucose_mmol = (self.baseline_glucose_mg_dl / MG_DL_PER_MMOL_L) * target_fluids.ecf_l
        else:
            self._standalone_glucose_mmol = (
                (self.baseline_glucose_mg_dl / MG_DL_PER_MMOL_L) * self._standalone_ecf_l
            )
        self._endo_insulin_c1 = self._init_insulin_mu_l
        self._hepatic_insulin = self._init_insulin_mu_l
        self._remote_insulin = self._init_insulin_mu_l
        self._k_insulin = self._init_insulin_mu_l
        self._arterial_insulin_mu_l = self._init_insulin_mu_l
        self._glucagon_drive = 0.0
        self._hypo_epi_ng_ml = 0.0
        self._epi_glycogen_adapt = 0.0
        self._surg_stress = 0.0
        self._circulation = 1.0
        self._arterial_glucose_mg_dl = self.baseline_glucose_mg_dl
        init_uge = self._compute_glycosuria_mg_min(self.baseline_glucose_mg_dl, renal_perfusion=1.0)
        st = self.state
        st.glucose_mg_dl = self.baseline_glucose_mg_dl
        st.arterial_glucose_mg_dl = self.baseline_glucose_mg_dl
        st.insulin_mu_l = self._init_insulin_mu_l
        st.endogenous_insulin_mu_l = self._init_insulin_mu_l
        st.hepatic_insulin_mu_l = self._init_insulin_mu_l
        st.remote_insulin_mu_l = self._init_insulin_mu_l
        st.glucagon_drive = 0.0
        st.hypoglycemic_epi_ng_ml = 0.0
        st.insulin_k_uptake = 0.0
        st.glycosuria_mg_min = init_uge
        st.osmotic_diuresis_ml_min = self._osmotic_diuresis_ml_min(init_uge)
        st.egp_mg_kg_min = self._compute_egp_mg_kg_min(
            self.baseline_glucose_mg_dl,
            self._init_insulin_mu_l,
            glucagon_drive=0.0,
            epi_transient=0.0,
            sepsis_severity=0.0,
            surg_stress=0.0,
            oxygen_supply=1.0,
            temp_q10=1.0,
        )
        st.rd_mg_kg_min = self._compute_rd_mg_kg_min(
            self.baseline_glucose_mg_dl,
            self._init_insulin_mu_l,
            epi_met=0.0,
            sepsis_severity=0.0,
            surg_stress=0.0,
            temp_q10=1.0,
        )

    def sync_exogenous_insulin(self, exo_insulin_cp: float, *, circulation: float | None = None) -> None:
        """Reflect an immediate exogenous insulin bolus without overwriting metabolic flux state."""
        circ = self._circulation if circulation is None else clamp01(circulation)
        if circ >= 0.99:
            total_ins = self._endo_insulin_c1 + max(0.0, float(exo_insulin_cp))
            self._arterial_insulin_mu_l = total_ins
            self.state.insulin_mu_l = total_ins

    @property
    def arterial_glucose_mg_dl(self) -> float:
        """Arterial/POC blood glucose (mg/dL); reads live ECF when perfusing and freezes in arrest."""
        if self._circulation >= 0.99:
            if self._fluids is not None:
                return (max(0.0, self._fluids.glucose_mmol) / max(1.0, self._fluids.ecf_l)) * MG_DL_PER_MMOL_L
            return self.state.glucose_mg_dl
        return self._arterial_glucose_mg_dl

    def _hepatic_insulin_suppression(self, hepatic_insulin_mu_l: float) -> float:
        ins = max(0.0, hepatic_insulin_mu_l)
        if ins <= 0.0:
            return 0.0
        return hill_function(ins, self._egp_ed50, 2.3)

    def _compute_rd_mg_kg_min(
        self,
        glucose_mg_dl: float,
        remote_insulin_mu_l: float,
        *,
        epi_met: float,
        sepsis_severity: float,
        surg_stress: float,
        temp_q10: float,
    ) -> float:
        g = max(0.0, glucose_mg_dl)
        # Obligate CNS + non-insulin uptake (~1.05 mg/kg/min at 90 mg/dL, with calibrated
        # GLUT1/3 low-glucose saturation Km ~18 mg/dL) plus modest glucose effectiveness.
        rc = 1.26 * g / (g + 18.0)
        sg = 0.0050 * g
        # Insulin-mediated peripheral disposal (Rizza 1981; Kolterman 1981; Alvestrand 1984; Letiexhe 1993).
        ins = max(0.0, remote_insulin_mu_l)
        ins_hill = hill_function(ins, self._rd_ed50, 2.0) if ins > 0.0 else 0.0
        # Michaelis-Menten GLUT4 substrate dependence below 90 mg/dL preserves disposal during insulin overdose.
        g_substrate = (2.0 * g / (g + 90.0)) if g < 90.0 else ((g / 90.0) / (1.0 + 0.0022 * (g - 90.0)))
        # Calibrated epinephrine clearance suppression above 0.15-0.20 ng/mL (Clutter 1980).
        epi_excess = max(0.0, epi_met - 0.12)
        epi_rc_factor = (
            1.0 - 0.38 * self._age_epi_sensitivity * hill_function(epi_excess, 0.25, 1.5)
            if epi_excess > 0.0
            else 1.0
        )
        # Sepsis (Revelly 2005) and surgical/CPB stress (model peripheral resistance alongside Schricker 2004 / Lehot 1992).
        stress_rc_factor = (1.0 - 0.50 * clamp01(sepsis_severity)) * (1.0 - 0.30 * clamp01(surg_stress))
        rd_ins = (
            self._rd_vmax
            * self._periph_vmax_scale
            * ins_hill
            * g_substrate
            * epi_rc_factor
            * stress_rc_factor
        )
        return (rc + sg + rd_ins) * temp_q10

    def _compute_glycosuria_mg_min(self, glucose_mg_dl: float, *, renal_perfusion: float) -> float:
        """C1-smooth splay renal glucose excretion (DeFronzo 2013)."""
        g = max(0.0, glucose_mg_dl)
        thresh_onset = 185.0 if self.type2_diabetes else 160.0
        if g <= thresh_onset or renal_perfusion <= 0.0:
            return 0.0
        size = self.lean_mass_kg / 70.0
        gfr_dl_min = 1.28 * size * self.renal_clearance_factor * clamp(renal_perfusion, 0.0, 2.0)
        if gfr_dl_min <= 0.0:
            return 0.0
        tmg = (420.0 if self.type2_diabetes else 317.0) * size * self.renal_clearance_factor
        filtered = gfr_dl_min * g
        f_th = gfr_dl_min * thresh_onset
        if filtered <= f_th:
            return 0.0
        if tmg <= f_th + 1e-6:
            return max(0.0, filtered - tmg)
        f_hi = 2.0 * tmg - f_th
        if filtered < f_hi:
            diff = filtered - f_th
            return (diff * diff) / (4.0 * (tmg - f_th))
        return filtered - tmg

    def _compute_egp_mg_kg_min(
        self,
        glucose_mg_dl: float,
        hepatic_insulin_mu_l: float,
        *,
        glucagon_drive: float,
        epi_transient: float,
        sepsis_severity: float,
        surg_stress: float,
        oxygen_supply: float,
        temp_q10: float,
    ) -> float:
        # Hepatic suppression by fast hepatic insulin (Rizza 1981; Insel 1975).
        ins_supp = self._hepatic_insulin_suppression(hepatic_insulin_mu_l)
        # Surgical stress blunts insulin/glucose suppression of EGP (Schricker 2004).
        eff_ins_supp = ins_supp * (1.0 - 0.35 * clamp01(surg_stress))
        # Direct glucose autoregulation suppresses EGP during hyperglycemia unless stress is high.
        g_excess = max(0.0, glucose_mg_dl - self._g_setpoint_mg_dl)
        g_autoreg = (
            0.35 * clamp01(g_excess / 100.0) * (1.0 - 0.70 * clamp01(surg_stress))
            if g_excess > 0.0
            else 0.0
        )
        basal_egp = self._egp_uninf_cap * max(0.0, 1.0 - eff_ins_supp) * (1.0 - g_autoreg)

        # Hepatic glycogenolytic reserve scales down in severe liver dysfunction (hepatic_function=0.5).
        hep_reserve = 0.35 + 0.65 * ((self.hepatic_function - 0.5) / 0.5)
        oxy_gate = 0.25 + 0.75 * clamp01(oxygen_supply)

        # Counterregulatory glucagon overrides moderate insulin (Rizza 1979).
        glucagon_egp = 2.65 * clamp01(glucagon_drive) * hep_reserve
        # Transient epinephrine glycogenolysis (Clutter 1980: peaks 20-30 min, >150-200 pg/mL).
        epi_excess = max(0.0, epi_transient - 0.12)
        epi_egp = (
            2.40 * self._age_epi_sensitivity * hill_function(epi_excess, 0.30, 1.6) * hep_reserve
            if epi_excess > 0.0
            else 0.0
        )
        # Supraphysiological hepatic insulin (>80 µU/mL) progressively overrides counterregulatory EGP,
        # preserving spontaneous recovery after modest boluses (Rizza 1979) while allowing severe
        # neuroglycopenia during large insulin overdoses.
        ins_over = max(0.0, hepatic_insulin_mu_l - 80.0)
        counterreg_scale = (
            1.0 - 0.75 * hill_function(ins_over, 130.0, 1.8)
            if ins_over > 0.0
            else 1.0
        )
        # Sepsis doubles basal Ra (Revelly 2005); surgical/CPB stress raises EGP (Schricker 2004; Lehot 1992).
        sepsis_egp = 1.25 * self._rest_egp_mg_kg_min * clamp01(sepsis_severity) * hep_reserve
        surg_egp = 0.85 * clamp01(surg_stress) * hep_reserve

        return (basal_egp + (glucagon_egp + epi_egp) * counterreg_scale + sepsis_egp + surg_egp) * temp_q10 * oxy_gate

    def sync_circulation(self, circulation: float, fluids=None) -> None:
        """Snapshot arterial glucose and insulin when perfusion drops before peripheral infusions drain."""
        circ = clamp01(circulation)
        if fluids is not None:
            self._fluids = fluids
        active_fluids = fluids if fluids is not None else self._fluids
        if self._circulation >= 0.99 and circ < 0.99:
            ecf_l = max(1.0, active_fluids.ecf_l if active_fluids is not None else self._standalone_ecf_l)
            glucose_mmol = max(
                0.0,
                active_fluids.glucose_mmol if active_fluids is not None else self._standalone_glucose_mmol,
            )
            self._arterial_glucose_mg_dl = (glucose_mmol / ecf_l) * MG_DL_PER_MMOL_L
            self._arterial_insulin_mu_l = self.state.insulin_mu_l
        self._circulation = circ

    def step(
        self,
        dt: float,
        *,
        fluids=None,
        exo_insulin_cp: float = 0.0,
        exo_epinephrine_ng_ml: float = 0.0,
        beta2_dose_ratio: float = 1.0,
        mac_sevo: float = 0.0,
        surgical_stimulation: float = 0.0,
        sepsis_severity: float = 0.0,
        cpb_active: bool = False,
        temp_c: float = 37.0,
        co_ratio: float = 1.0,
        renal_perfusion: float = 1.0,
        oxygen_supply: float = 1.0,
        circulation: float = 1.0,
    ) -> GlucoseState:
        """Advance coupled glucose, endogenous insulin, counterregulation, and glycosuria."""
        circ = clamp01(circulation)
        self.sync_circulation(circ, fluids)
        active_fluids = fluids if fluids is not None else self._fluids
        ecf_l = max(1.0, active_fluids.ecf_l if active_fluids is not None else self._standalone_ecf_l)
        glucose_mmol = max(
            0.0,
            active_fluids.glucose_mmol if active_fluids is not None else self._standalone_glucose_mmol,
        )
        g_now = (glucose_mmol / ecf_l) * MG_DL_PER_MMOL_L

        if dt <= 0.0:
            self.sync_exogenous_insulin(exo_insulin_cp, circulation=circ)
            return self.state

        # Substep large dt (> 2 s) so fast bolus/glucose excursions stay step-size invariant.
        substeps = max(1, math.ceil(dt / 2.0)) if dt > 2.0 else 1
        h = dt / substeps
        h_min = h / 60.0
        clamped_t = clamp(temp_c, 22.0, 42.0)
        temp_q10 = 2.0 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0
        beta2_dr = max(1.0, float(beta2_dose_ratio))
        exo_ins = max(0.0, float(exo_insulin_cp))
        exo_epi = max(0.0, float(exo_epinephrine_ng_ml))
        sepsis = clamp01(sepsis_severity)
        stress_target = clamp01(max(surgical_stimulation, 1.0 if cpb_active else 0.0))
        cpb_stress_boost = 0.95 if cpb_active else 0.0

        # Volatile suppression of glucose-stimulated insulin secretion (Tanaka 2005:
        # -77% across 0.5-1.5 MAC sevoflurane, concentration-independent above 0.5 MAC).
        sevo_beta_supp = (
            0.78 * hill_function(max(0.0, mac_sevo), 0.20, 2.5)
            if mac_sevo > 0.0
            else 0.0
        )
        # Hypothermia suppresses insulin secretion during hypothermic CPB (Lehot 1992:
        # -44% to -60% at 25.6 °C), rebounding promptly on rewarming.
        hypo_beta_scale = (
            0.40 + 0.60 * clamp01((clamped_t - 24.0) / 11.0)
            if clamped_t < 35.0
            else 1.0
        )

        # Endogenous insulin clearance scales with organ function, perfusion, and temperature.
        nonrenal_share = 1.0 - (0.27 * self.renal_clearance_factor / self._organ_cl_scale)
        renal_share = 1.0 - nonrenal_share
        cl_flow = nonrenal_share * (max(0.1, co_ratio) ** 0.5) * temp_q10 + renal_share * clamp(renal_perfusion, 0.0, 2.0)
        mcr_now = self._insulin_mcr_base_l_min * cl_flow * circ

        egp_rate = 0.0
        rd_rate = 0.0
        uge_mg_min = 0.0
        epi_met = 0.0

        for _ in range(substeps):
            g_now = max(_MIN_GLUCOSE_MG_DL, (glucose_mmol / ecf_l) * MG_DL_PER_MMOL_L)

            # Stress-hormone dynamics (cortisol/catecholamine surgical/CPB stress, Schricker 2004).
            if stress_target != self._surg_stress:
                tau_s = 20.0 if stress_target > self._surg_stress else 45.0
                self._surg_stress += (stress_target - self._surg_stress) * -math.expm1(-h_min / tau_s)

            # Hypoglycemic counterregulation below 68 mg/dL (Mitrakou 1991; Rizza 1979).
            if g_now < _COUNTERREG_THRESHOLD_MG_DL:
                glucagon_target = clamp01((_COUNTERREG_THRESHOLD_MG_DL - g_now) / 22.0)
                hypo_epi_target = 0.95 * (clamp01((_COUNTERREG_THRESHOLD_MG_DL - g_now) / 25.0) ** 1.3)
            else:
                glucagon_target = 0.0
                hypo_epi_target = 0.0

            if glucagon_target != self._glucagon_drive:
                # Rizza 1979 Fig. 1: glucagon rises briskly and remains elevated during 40-90 min recovery.
                tau_g = 5.0 if glucagon_target > self._glucagon_drive else 24.0
                self._glucagon_drive += (glucagon_target - self._glucagon_drive) * -math.expm1(-h_min / tau_g)
            if hypo_epi_target != self._hypo_epi_ng_ml:
                tau_e = 4.5 if hypo_epi_target > self._hypo_epi_ng_ml else 10.0
                self._hypo_epi_ng_ml += (hypo_epi_target - self._hypo_epi_ng_ml) * -math.expm1(-h_min / tau_e)

            # Effective metabolic epinephrine (exogenous + endogenous surge, beta-2 gated).
            total_epi = exo_epi + self._hypo_epi_ng_ml
            epi_met = total_epi / beta2_dr
            # Slow glycogenolytic tachyphylaxis during sustained exogenous epinephrine (Clutter 1980).
            if epi_met != self._epi_glycogen_adapt:
                self._epi_glycogen_adapt += (epi_met - self._epi_glycogen_adapt) * -math.expm1(-h_min / 38.0)
            epi_transient = max(0.0, epi_met - 0.55 * self._epi_glycogen_adapt) + 0.25 * self._hypo_epi_ng_ml / beta2_dr

            # Calibrated progressive endogenous beta-cell shutoff below ~82 mg/dL.
            if g_now < 82.0:
                hypo_shutoff = clamp01((g_now - 52.0) / 30.0)
                stim_factor = 0.0
            else:
                hypo_shutoff = 1.0
                g_above = max(0.0, g_now - self._g_setpoint_mg_dl)
                raw_stim = self._beta_stim_max * hill_function(g_above, 85.0, 2.0) if g_above > 0.0 else 0.0
                # Epinephrine suppresses insulin secretion above ~0.40 ng/mL (basal in Clutter 1980, extended to stimulated).
                epi_beta_supp = (
                    0.75 * hill_function(max(0.0, total_epi - 0.40), 0.30, 1.5)
                    if total_epi > 0.40
                    else 0.0
                )
                stim_factor = raw_stim * (1.0 - sevo_beta_supp) * (1.0 - epi_beta_supp)

            # In sepsis, compensatory basal hyperinsulinemia tracks insulin resistance (Revelly 2005).
            # Multiply by temp_q10 * hypo_beta_scale so hypothermic beta-cell suppression drops
            # plasma insulin ~50% despite reduced clearance, rebounding on rewarming (Lehot 1992).
            sepsis_sec = 1.0 + 0.8 * sepsis
            secretion_mu_min = (
                self._basal_secretion_mu_min
                * (hypo_shutoff * sepsis_sec + stim_factor)
                * temp_q10
                * hypo_beta_scale
                * circ
            )
            if circ > 0.0 and mcr_now > 1e-9:
                k_ins = mcr_now / self._insulin_v1_l
                target_endo = secretion_mu_min / mcr_now
                self._endo_insulin_c1 += (target_endo - self._endo_insulin_c1) * -math.expm1(-k_ins * h_min)

            total_plasma_ins = self._endo_insulin_c1 + exo_ins
            if circ > 0.0:
                self._hepatic_insulin += (total_plasma_ins - self._hepatic_insulin) * (
                    -math.expm1(-h_min * circ / _HEPATIC_INSULIN_TAU_MIN)
                )
                self._remote_insulin += (total_plasma_ins - self._remote_insulin) * (
                    -math.expm1(-h_min * circ / _REMOTE_INSULIN_TAU_MIN)
                )
                self._k_insulin += (total_plasma_ins - self._k_insulin) * (
                    -math.expm1(-h_min * circ / _K_INSULIN_TAU_MIN)
                )

            # Metabolic fluxes (mg/kg lean/min) and glycosuria (mg/min).
            egp_rate = self._compute_egp_mg_kg_min(
                g_now,
                self._hepatic_insulin,
                glucagon_drive=self._glucagon_drive,
                epi_transient=epi_transient,
                sepsis_severity=sepsis,
                surg_stress=self._surg_stress,
                oxygen_supply=oxygen_supply,
                temp_q10=temp_q10,
            ) + cpb_stress_boost * self._surg_stress * temp_q10 * (0.25 + 0.75 * clamp01(oxygen_supply))
            rd_rate = self._compute_rd_mg_kg_min(
                g_now,
                self._remote_insulin,
                epi_met=epi_met,
                sepsis_severity=sepsis,
                surg_stress=self._surg_stress,
                temp_q10=temp_q10,
            )
            uge_mg_min = self._compute_glycosuria_mg_min(g_now, renal_perfusion=renal_perfusion) * circ

            # During circulatory arrest, tissue/hepatic flux slows sharply with perfusion;
            # uge_mg_min is already scaled by circ above so it is not attenuated twice.
            flux_scale = 0.15 + 0.85 * circ
            net_mg_min = (egp_rate - rd_rate) * self.lean_mass_kg * flux_scale - uge_mg_min
            delta_mmol = (net_mg_min / MG_PER_MMOL) * h_min
            min_mmol = (_MIN_GLUCOSE_MG_DL / MG_DL_PER_MMOL_L) * ecf_l
            glucose_mmol = max(min_mmol, glucose_mmol + delta_mmol)

        if active_fluids is not None:
            active_fluids.glucose_mmol = glucose_mmol
        else:
            self._standalone_glucose_mmol = glucose_mmol

        g_final = (glucose_mmol / ecf_l) * MG_DL_PER_MMOL_L
        total_ins_final = self._endo_insulin_c1 + exo_ins
        if circ >= 0.99:
            self._arterial_glucose_mg_dl = g_final
            self._arterial_insulin_mu_l = total_ins_final
        elif circ > 0.0:
            alpha = -math.expm1(-dt * circ / 20.0)
            self._arterial_glucose_mg_dl += (g_final - self._arterial_glucose_mg_dl) * alpha
            self._arterial_insulin_mu_l += (total_ins_final - self._arterial_insulin_mu_l) * alpha

        # Insulin-mediated transcellular potassium uptake (mmol/L equilibrium shift;
        # DeFronzo 1980; Alvestrand 1984; Allon and Copkney 1990; Chothia 2014).
        # Independent of uremic/diabetic glucose resistance and baseline K+.
        excess_k_ins = max(0.0, self._k_insulin - self.basal_insulin_mu_l)
        if excess_k_ins > 0.0:
            e_pow = excess_k_ins ** 1.75
            insulin_k_uptake = 1.14 * e_pow / (e_pow + 52.0 ** 1.75)
        else:
            insulin_k_uptake = 0.0

        osmotic_diuresis = self._osmotic_diuresis_ml_min(uge_mg_min)

        st = self.state
        st.glucose_mg_dl = g_final
        st.arterial_glucose_mg_dl = self.arterial_glucose_mg_dl
        st.endogenous_insulin_mu_l = self._endo_insulin_c1
        st.insulin_mu_l = self._arterial_insulin_mu_l
        st.hepatic_insulin_mu_l = self._hepatic_insulin
        st.remote_insulin_mu_l = self._remote_insulin
        st.glucagon_drive = self._glucagon_drive
        st.hypoglycemic_epi_ng_ml = self._hypo_epi_ng_ml
        st.insulin_k_uptake = insulin_k_uptake
        st.glycosuria_mg_min = uge_mg_min
        st.osmotic_diuresis_ml_min = osmotic_diuresis
        st.egp_mg_kg_min = egp_rate
        st.rd_mg_kg_min = rd_rate
        return st
