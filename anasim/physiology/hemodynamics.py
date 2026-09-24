import math
from typing import Optional

from scipy.optimize import root_scalar

from anasim.core.constants import BLOOD_VOLUME_MIN, HR_MAX, HR_MIN, TEMP_TPR_COEFFICIENT
from anasim.core.enums import RhythmType
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.patient.patient import Patient

from .hemo_config import HemodynamicConfig
from .hemo_types import HemoState, HemoStateExtended


class HemodynamicModel:
    """Su et al. Br J Anaesth. 2023 turnover model with volume, pulmonary
    circulation, reflexes, vasoactive drugs, and distributive shock.

    Units: MAP mmHg, HR bpm, SV mL, CO L/min, SVR Wood units (mmHg·min/L),
    internal TPR mmHg·min/mL. Sources are listed in docs/REFERENCES.md.
    """
    def __init__(self, patient: Patient, config: Optional[HemodynamicConfig] = None):
        self.patient = patient
        self.config = config or HemodynamicConfig()
        self._cached_state: Optional[HemoStateExtended] = None
        for name, value in vars(self.config).items():
            setattr(self, name, value)

        self.baseline_hb = patient.baseline_hb
        self.baseline_hct = patient.baseline_hct

        self.base_hr = patient.baseline_hr if patient.baseline_hr > 0 else self.base_hr

        # Baseline SV from cardiac index and BSA.
        ci_0 = self.ci_elderly if patient.age > self.ci_elderly_age else self.ci_adult
        co_0 = ci_0 * patient.bsa
        self.base_sv = (co_0 * 1000.0) / self.base_hr if self.base_hr > 0 else self.base_sv

        flow_ml_min = self.base_hr * self.base_sv
        self.base_tpr = patient.baseline_map / flow_ml_min if flow_ml_min > 0 else self.base_tpr
        self.base_co_l_min = flow_ml_min / 1000.0
        self.baseline_do2 = self.calc_oxygen_content(self.baseline_hb, 0.98, 95.0) * self.base_co_l_min * 10.0
        self._emax_prop_sv_age = self.emax_prop_sv_typ * math.exp(self.age_emax_sv * (patient.age - 35.0))
        self._inv_base_rmap_denom = 1.0 / (self.base_hr * self.base_sv * self.base_tpr)
        self._inv_base_co_l_min = 1.0 / max(0.1, self.base_co_l_min)
        self._sepsis_hr_gain = self.sepsis_hr_increase / max(self.base_hr, 1.0)

        self.delta_tpr_vasopressors = 0.0
        self.dist_svr = 0.0
        self.dist_hr = 0.0
        self.dist_sv = 0.0
        self._sepsis_severity = 0.0
        self._anaphylaxis_severity = 0.0

        # Fast HR effects (chemoreflex, baroreflex, vasoactive chronotropy).
        self.smoothed_epi_hr = 0.0
        self._epi_pressor_ce = 0.0
        self._epi_chrono_effect = 0.0
        self.smoothed_chemo_hr = 0.0
        self.smoothed_baro_hr = 0.0
        self._baro_setpoint = patient.baseline_map
        self._stim_map = 0.0  # MAP attributable to noxious stimulation
        self.myocardial_hypoxia = 0.0  # 0-1 depression of rate and contractility

        self.sv_star = self.base_sv
        self.hr_star = self.base_hr
        self.tpr = self.base_tpr
        self.tde_sv = 0.0
        self.tde_hr = 0.0
        self._frank_starling_baseline_raw = 1.0 - math.exp(-2.0)
        # Production rates that give equilibrium at RMAP = 1.
        self.kin_tpr = self.kout * self.base_tpr
        self.kin_sv = self.kout * self.base_sv
        self.kin_hr = self.kout * self.base_hr
        self.ce_sevo = 0.0

        self.blood_volume = patient.estimate_blood_volume()
        self.blood_volume_0 = self.blood_volume
        self.hb_mass = self.baseline_hb * (self.blood_volume / 100.0)  # g
        self.hb_conc = self.baseline_hb
        # Hemorrhage depletes stressed volume first, lowering MCFP and preload.
        self.unstressed_volume = self.blood_volume * self.unstressed_volume_fraction
        self.cv = self.venous_compliance  # mL/mmHg
        # Guyton MCFP = stressed volume / venous compliance, about 10-15 mmHg.
        stressed_vol_0 = max(0.0, self.blood_volume - self.unstressed_volume)
        self.mcfp_0 = max(self.mcfp_floor, stressed_vol_0 / self.cv)

        # Venous return resistance (Wood units) calibrated to baseline CO.
        delta_p_vr = max(0.1, self.mcfp_0 - self.rap_baseline)
        self.venous_return_resistance = delta_p_vr / max(0.1, self.base_co_l_min)
        self._pulm_flow_factor = 1.0
        self._last_mcfp = self.mcfp_0
        self._last_rap = self.rap_baseline
        self._last_pvr_factor = 1.0
        self._last_pvr = self.pvr_wood_baseline
        self._last_rv_co = self.base_co_l_min
        self._last_lv_inflow = self.base_co_l_min
        self._last_preload_factor = 1.0
        self._last_preload_sv_factor = 1.0

        self.total_crystalloid_in_ml = 0.0
        self.total_colloid_in_ml = 0.0
        self.total_blood_in_ml = 0.0
        self.total_urine_out_ml = 0.0
        self.total_blood_out_ml = 0.0
        self.total_leak_out_ml = 0.0
        self.total_third_space_ml = 0.0

        self._rhythm_type = RhythmType.SINUS
        self.f_preload_pit = 1.0
        self.vasopressor_sv_factor = 1.0
        self._prev_map = patient.baseline_map

    def invalidate_state_cache(self) -> None:
        """Drop the cached state after an external change."""
        self._cached_state = None

    @property
    def sepsis_severity(self) -> float:
        return self._sepsis_severity

    @sepsis_severity.setter
    def sepsis_severity(self, value: float) -> None:
        value = clamp01(value)
        if value != self._sepsis_severity:
            self._sepsis_severity = value
            self.invalidate_state_cache()

    @property
    def anaphylaxis_severity(self) -> float:
        return self._anaphylaxis_severity

    @anaphylaxis_severity.setter
    def anaphylaxis_severity(self, value: float) -> None:
        value = clamp01(value)
        if value != self._anaphylaxis_severity:
            self._anaphylaxis_severity = value
            self.invalidate_state_cache()

    @property
    def rhythm_type(self) -> RhythmType:
        return self._rhythm_type

    @rhythm_type.setter
    def rhythm_type(self, value: RhythmType) -> None:
        if value != self._rhythm_type:
            self._rhythm_type = value
            self.invalidate_state_cache()

    def set_nore_pd(self, c50: float, emax: float = 98.7, gamma: float = 1.8):
        self.nore_c50 = c50
        self.nore_emax_map = emax
        self.nore_gamma = gamma
        self.invalidate_state_cache()

    def add_volume(
        self,
        amount_ml: float,
        hematocrit: float = 0.0,
        retention_fraction: Optional[float] = None,
        label: str = "crystalloid",
    ):
        """Add fluid or blood (positive) or remove blood (negative).

        Retained volume also gives a transient Frank-Starling SV rise via tde_sv.
        """
        self._cached_state = None
        if amount_ml == 0:
            return

        retained_ml = 0.0
        if amount_ml < 0:
            loss_ml = -amount_ml
            available = max(0.0, self.blood_volume - BLOOD_VOLUME_MIN)
            actual_loss = min(loss_ml, available)
            if actual_loss > 0:
                self.blood_volume -= actual_loss
                self.total_blood_out_ml += actual_loss
                hb_loss = self.hb_conc * actual_loss / 100.0
                self.hb_mass = max(0.0, self.hb_mass - hb_loss)
        else:
            if hematocrit > 0:
                self.total_blood_in_ml += amount_ml
                retained_ml = amount_ml * self.blood_retention_fraction
                self.total_third_space_ml += max(0.0, amount_ml - retained_ml)
                hct_ref = max(self.baseline_hct, 0.01)
                hb_gain = self.baseline_hb * (hematocrit / hct_ref) * retained_ml / 100.0
                self.hb_mass += hb_gain
            else:
                is_colloid = (label == "colloid")
                if is_colloid:
                    self.total_colloid_in_ml += amount_ml
                    retention = self.colloid_retention_fraction
                else:
                    self.total_crystalloid_in_ml += amount_ml
                    retention = self.crystalloid_retention_fraction
                if retention_fraction is not None:
                    retention = retention_fraction
                retention = clamp01(retention)
                retained_ml = amount_ml * retention
                self.total_third_space_ml += max(0.0, amount_ml - retained_ml)

            self.blood_volume += retained_ml

            if retained_ml > 0:
                gain = 0.02 * (self.base_sv / 80.0)
                self.tde_sv += retained_ml * gain

        self.blood_volume = max(BLOOD_VOLUME_MIN, self.blood_volume)
        self._update_hb_conc()

    def _update_hb_conc(self):
        if self.blood_volume <= 0:
            self.hb_conc = 0.0
        else:
            self.hb_conc = self.hb_mass / (self.blood_volume / 100.0)

    def _calc_stressed_volume(self, sepsis_sev: Optional[float] = None) -> float:
        """Stressed volume after sepsis shifts part of baseline volume to unstressed."""
        sev = clamp01(self.sepsis_severity) if sepsis_sev is None else sepsis_sev
        pooling = self.sepsis_pooling_fraction * self.blood_volume_0 * sev
        return max(0.0, self.blood_volume - self.unstressed_volume - pooling)

    def get_hematocrit(self) -> float:
        if self.baseline_hb <= 0:
            return self.baseline_hct
        ratio = self.hb_conc / self.baseline_hb
        return clamp(self.baseline_hct * ratio, 0.0, 0.7)

    @staticmethod
    def calc_oxygen_content(hb_g_dl: float, sao2_frac: float, pao2: float) -> float:
        sao2_frac = clamp01(sao2_frac)
        return hb_g_dl * 1.34 * sao2_frac + 0.003 * pao2

    def compute_do2_ratio(self, sao2_frac: float, pao2: float, co_l_min: float) -> float:
        caO2 = self.calc_oxygen_content(self.hb_conc, sao2_frac, pao2)
        do2 = caO2 * max(0.0, co_l_min) * 10.0
        if self.baseline_do2 <= 0:
            return 1.0
        ratio = do2 / self.baseline_do2
        return clamp(ratio, 0.0, 2.0)

    def _frank_starling(self, preload_factor: float, inotropy: float = 1.0) -> float:
        """SV factor 1 - exp(-2 x preload), normalized to 1 at baseline preload.

        Floored at 5% of baseline SV.
        """
        pf = clamp(preload_factor, 0.01, 2.5)
        normalized = (1.0 - math.exp(-2.0 * pf)) / self._frank_starling_baseline_raw
        return max(0.05, inotropy * normalized)

    def _calc_hemorrhage_response(self) -> tuple:
        """HR and TPR multipliers by ATLS hemorrhage class, added to the baroreflex."""
        if self.blood_volume_0 <= 0:
            return 1.0, 1.0

        vol_deficit = (self.blood_volume_0 - self.blood_volume) / self.blood_volume_0
        vol_deficit = max(0.0, vol_deficit)

        if vol_deficit < 0.15:
            # Class I
            hr_mult = 1.0 + 0.15 * (vol_deficit / 0.15)
            tpr_mult = 1.0 + 0.08 * (vol_deficit / 0.15)
        elif vol_deficit < 0.30:
            # Class II
            progress = (vol_deficit - 0.15) / 0.15
            hr_mult = 1.15 + 0.20 * progress
            tpr_mult = 1.08 + 0.12 * progress
        elif vol_deficit < 0.45:
            # Class III: peak compensation
            progress = (vol_deficit - 0.30) / 0.15
            hr_mult = 1.35 + 0.25 * progress
            tpr_mult = 1.20 + 0.10 * progress
        else:
            # Class IV: decompensation
            progress = min(1.0, (vol_deficit - 0.45) / 0.25)
            hr_mult = max(0.8, 1.60 - 0.80 * progress)
            tpr_mult = max(0.5, 1.30 - 0.80 * progress)

        return hr_mult, tpr_mult

    def _calc_epi_effects(self, ce_epi: float, ce_pressor: float) -> tuple[float, float, float]:
        """Return direct chronotropy, SV factor, and baseline-relative SVR factor.

        The lagged pressor concentration separates the HR and SBP peaks after a
        bolus; at steady exposure both concentrations are equal. Beta-2 dilation
        scales with current vascular tone; alpha tone adds relative to baseline.
        """
        delta_hr = self.epi_emax_hr * hill_function(ce_epi, self.epi_c50_hr, self.epi_gamma_hr)
        delta_hr *= 1.0 - self.epi_volatile_hr_depression * clamp01(self.ce_sevo)
        sv_factor = 1.0 + self.epi_emax_sv * hill_function(ce_pressor, self.epi_c50_sv, 1.0)
        beta2 = self.epi_emax_svr_beta * hill_function(ce_epi, self.epi_c50_beta2, 1.0)
        alpha = self.epi_emax_svr_alpha * hill_function(ce_pressor, self.epi_c50_alpha, self.epi_gamma_alpha)
        svr_factor = 1.0 + beta2 * self.tpr / self.base_tpr + alpha
        return delta_hr, sv_factor, max(0.2, svr_factor)

    def _calc_phenyl_effects(self, ce_phenyl: float) -> float:
        """Phenylephrine SVR effect (pure alpha-1)."""
        if ce_phenyl <= 0:
            return 1.0
        return 1.0 + self.phenyl_emax_svr * hill_function(ce_phenyl, self.phenyl_c50, self.phenyl_gamma)

    @staticmethod
    def _calc_hr_sv_svr_effects(ce: float, c50: float, gamma: float,
                                emax_hr: float, emax_sv: float, emax_svr: float) -> tuple:
        """Return (delta HR, SV factor, SVR factor) from one Hill effect."""
        if ce <= 0:
            return 0.0, 1.0, 1.0
        hill = hill_function(ce, c50, gamma)
        return emax_hr * hill, 1.0 + emax_sv * hill, max(0.5, 1.0 + emax_svr * hill)

    def _calc_pvr_factor(self, pao2: float, peep_cmH2O: Optional[float] = None) -> float:
        """PVR multiplier from hypoxic vasoconstriction and PEEP."""
        pvr_factor = 1.0
        if pao2 > 0 and pao2 < self.pvr_o2_threshold:
            denom = max(1.0, self.pvr_o2_threshold - self.pvr_o2_floor)
            frac = clamp01((self.pvr_o2_threshold - pao2) / denom)
            pvr_factor *= 1.0 + (self.pvr_o2_max_factor - 1.0) * frac
        if peep_cmH2O is not None:
            peep_excess = max(0.0, peep_cmH2O - self.pvr_peep_ref)
            pvr_factor *= 1.0 + self.pvr_peep_slope * peep_excess

        return clamp(pvr_factor, 0.2, self.pvr_max_factor)

    def _update_pulmonary_coupling(
        self,
        dt: float,
        pao2: float,
        peep_cmH2O: Optional[float],
        f_preload_pit: float,
        sepsis_sev: float,
    ) -> float:
        """Return the Frank-Starling factor after venous return and pulmonary transit.

        Venous return follows MCFP - RAP, is attenuated by PVR, and reaches the
        left heart through a first-order transit delay.
        """
        stressed_vol = self._calc_stressed_volume(sepsis_sev)
        mcfp = stressed_vol / self.cv if self.cv > 0 else self.mcfp_0

        rap = self.rap_baseline
        delta_p = max(0.0, mcfp - rap)
        vr_flow_l_min = delta_p / max(0.1, self.venous_return_resistance)
        vr_flow_factor = vr_flow_l_min * self._inv_base_co_l_min

        pvr_factor = self._calc_pvr_factor(pao2, peep_cmH2O)
        rv_out_target_factor = vr_flow_factor * (pvr_factor ** (-self.pvr_flow_exponent))

        if self.pulmonary_transit_time_s > 0 and dt > 0:
            alpha = min(1.0, dt / self.pulmonary_transit_time_s)
            self._pulm_flow_factor += (rv_out_target_factor - self._pulm_flow_factor) * alpha
        elif self._pulm_flow_factor <= 0:
            self._pulm_flow_factor = rv_out_target_factor

        self._pulm_flow_factor = clamp(self._pulm_flow_factor, 0.05, 3.0)

        f_preload = self._pulm_flow_factor * f_preload_pit
        f_frank_starling = self._frank_starling(f_preload)

        self._last_mcfp = mcfp
        self._last_rap = rap
        self._last_pvr = self.pvr_wood_baseline * pvr_factor
        self._last_rv_co = rv_out_target_factor * self.base_co_l_min
        self._last_lv_inflow = self._pulm_flow_factor * self.base_co_l_min
        self._last_preload_factor = f_preload
        self._last_preload_sv_factor = f_frank_starling

        return f_frank_starling

    def _calc_nore_effects(self, ce_nore: float) -> tuple:
        """Return (delta HR, SV factor, SVR factor); alpha-1 constriction dominates.

        Reflex bradycardia comes from the baroreflex.
        """
        if ce_nore <= 0:
            return 0.0, 1.0, 1.0
        nore_hill = hill_function(ce_nore, self.nore_c50, self.nore_gamma)
        delta_hr = self.nore_emax_hr * nore_hill
        sv_factor = 1.0 + self.nore_emax_sv * nore_hill
        # Emax is a MAP rise; express it relative to an 80 mmHg baseline.
        svr_factor = 1.0 + (self.nore_emax_map * nore_hill) / 80.0
        return delta_hr, sv_factor, svr_factor

    def _calc_anesthetic_effects(self, cp_prop: float, cp_remi: float, ce_sevo: float) -> tuple:
        """Combined anesthetic effects on TPR, SV, and HR (Su et al. 2023).

        Propofol and remifentanil use plasma concentrations.
        Returns (total_eff_tpr, total_eff_sv, total_eff_hr, eff_remi_tpr,
        eff_remi_sv, eff_remi_hr).
        """
        cp = max(0.0, cp_prop)
        cr = max(0.0, cp_remi)
        emax_prop_sv = self._emax_prop_sv_age

        # Propofol TPR effect, with remifentanil interaction.
        remi_int_term = self.int_tpr * (cr / (self.ec50_remi_tpr + cr + 1e-9))
        eff_prop_tpr = (self.emax_prop_tpr + remi_int_term) * hill_function(cp, self.ec50_prop_tpr, self.gamma_prop)
        eff_prop_sv = emax_prop_sv * hill_function(cp, self.ec50_prop_sv, 1.0)

        remi_shift_factor = self.vol_remi_shift_max * (cr / (self.vol_remi_ec50 + cr + 1e-9))
        vol_ec50_mult = 1.0 - remi_shift_factor

        # Sevoflurane (no HR effect); remifentanil shifts only the TPR EC50.
        eff_sevo_tpr = self.sevo_emax_tpr * hill_function(
            ce_sevo, self.sevo_ec50_tpr * vol_ec50_mult, self.sevo_gamma_tpr
        )
        eff_sevo_sv = self.sevo_emax_sv * hill_function(ce_sevo, self.sevo_ec50_sv, 1.0)

        total_eff_tpr = max(-0.95, eff_prop_tpr + eff_sevo_tpr)
        total_eff_sv = max(-0.95, eff_prop_sv + eff_sevo_sv)
        total_eff_hr = 0.0

        # Remifentanil acts on dissipation; propofol modulates the SV and HR slopes.
        eff_remi_tpr = self.emax_remi_tpr * hill_function(cr, self.ec50_remi_tpr, self.gamma_remi_tpr)
        slope_sv = self.sl_remi_sv + self.int_sv * (cp / (self.ec50_prop_sv + cp + 1e-9))
        eff_remi_sv = slope_sv * cr
        slope_hr = self.sl_remi_hr + self.int_hr * (cp / (self.ec50_int_hr + cp + 1e-9))
        eff_remi_hr = max(-0.9, min(0.9, slope_hr * cr))

        return total_eff_tpr, total_eff_sv, total_eff_hr, eff_remi_tpr, eff_remi_sv, eff_remi_hr

    def _compute_state(
        self,
        preload_sv_factor: Optional[float] = None,
        sepsis_sev: Optional[float] = None,
        anaph_sev: Optional[float] = None,
        distributive_tpr_offset: Optional[float] = None,
        hr_base: Optional[float] = None,
    ) -> HemoStateExtended:
        if sepsis_sev is None:
            sepsis_sev = clamp01(self.sepsis_severity)
        if anaph_sev is None:
            anaph_sev = clamp01(self.anaphylaxis_severity)

        if hr_base is None:
            hr_base = self._calc_hr()
        current_hr = hr_base + self.dist_hr
        current_hr = max(HR_MIN, current_hr)
        current_hr = min(HR_MAX, current_hr)

        # Rhythm rates: SVT 150-220, VT 150-250, untreated AF with RVR 110-150 bpm.
        if self.rhythm_type == RhythmType.SVT:
            current_hr = 160.0
        elif self.rhythm_type == RhythmType.VTACH:
            current_hr = 180.0
        elif self.rhythm_type in (RhythmType.VFIB, RhythmType.ASYSTOLE):
            current_hr = 0.0
        elif self.rhythm_type == RhythmType.AFIB:
            current_hr = max(current_hr, 110.0)
        elif self.rhythm_type == RhythmType.SINUS_BRADY:
            current_hr = min(current_hr, 50.0)
        current_hr *= 1.0 - self.hypoxia_hr_depression * self.myocardial_hypoxia

        term = 1.0 - self.hr_sv_coupling * math.log(max(1.0, current_hr / self.base_hr))
        raw_sv = (self.sv_star + self.tde_sv) * term + self.dist_sv

        if preload_sv_factor is None:
            preload_sv_factor = self._last_preload_sv_factor if self._last_preload_sv_factor > 0 else 1.0
        current_sv = raw_sv * preload_sv_factor * self.vasopressor_sv_factor
        current_sv = max(1.0, current_sv) * (1.0 - self.hypoxia_sv_depression * self.myocardial_hypoxia)

        if self.rhythm_type == RhythmType.AFIB:
            current_sv *= 0.80  # Loss of atrial kick
        elif self.rhythm_type == RhythmType.VTACH:
            current_sv *= 0.25  # CO about 40% of baseline
        elif self.rhythm_type in (RhythmType.VFIB, RhythmType.ASYSTOLE):
            current_sv = 0.0
        elif self.rhythm_type == RhythmType.SVT:
            current_sv *= 0.50  # CO about 80-90% of baseline

        if self.rhythm_type in (RhythmType.VFIB, RhythmType.ASYSTOLE) or current_hr <= 0.0 or current_sv <= 0.0:
            current_hr = current_sv = co = map_val = svr_val = self._stim_map = 0.0
        else:
            co = current_hr * current_sv / 1000.0
            if distributive_tpr_offset is None:
                distributive_tpr_offset = -(
                    self.sepsis_svr_drop_wood * sepsis_sev + self.anaphylaxis_svr_drop_wood * anaph_sev
                ) / 1000.0
            # TPR floor of about 6 Wood units.
            eff_tpr = max(
                0.006,
                self.tpr + self.delta_tpr_vasopressors + self.dist_svr / 1000.0 + distributive_tpr_offset,
            )
            map_val = clamp(current_hr * current_sv * eff_tpr, 5.0, 300.0)
            svr_val = map_val / co
            # First-order share of MAP from stimulation; nociception resets the baroreflex.
            self._stim_map = map_val * (
                self.dist_hr / current_hr + self.dist_sv / max(raw_sv, 1.0) + self.dist_svr / 1000.0 / eff_tpr
            )

        self._cached_state = HemoStateExtended(
            map=map_val,
            hr=current_hr,
            sv=current_sv,
            svr=svr_val,
            co=co,
            tpr=self.tpr,
            sv_star=self.sv_star,
            hr_star=self.hr_star,
            tde_sv=self.tde_sv,
            tde_hr=self.tde_hr,
            ce_sevo=self.ce_sevo,
            mcfp=self._last_mcfp,
            rap=self._last_rap,
            pvr=self._last_pvr,
            rv_co=self._last_rv_co,
            lv_inflow=self._last_lv_inflow,
            preload_factor=self._last_preload_factor,
            rhythm_type=self.rhythm_type,
        )
        return self._cached_state

    @property
    def state(self) -> HemoStateExtended:
        if self._cached_state is not None:
            return self._cached_state
        return self._compute_state()

    @state.setter
    def state(self, new_state: HemoState):
        self._cached_state = None

        if not isinstance(new_state, HemoStateExtended):
            raise TypeError("Hemodynamic state must be HemoStateExtended")
        self.tpr = new_state.tpr
        self.sv_star = new_state.sv_star
        self.hr_star = new_state.hr_star
        self.tde_sv = new_state.tde_sv
        self.tde_hr = new_state.tde_hr
        self.ce_sevo = new_state.ce_sevo
        self.rhythm_type = new_state.rhythm_type
        self._prev_map = self._baro_setpoint = new_state.map
        self.smoothed_baro_hr = 0.0

    def _calc_hr(self):
        return self.hr_star + self.tde_hr + self.smoothed_chemo_hr + self.smoothed_baro_hr + self.smoothed_epi_hr

    def step(self, dt: float, cp_prop: float, cp_remi: float, ce_nore: float, pit: float, paco2: float, pao2: float,
             dist_hr: float = 0.0, dist_sv: float = 0.0, dist_svr: float = 0.0,
             mac_sevo: float = 0.0, ce_epi: float = 0.0, ce_phenyl: float = 0.0,
             ce_vaso: float = 0.0, ce_dobu: float = 0.0, ce_mil: float = 0.0,
             temp_c: float = 37.0, peep_cmH2O: Optional[float] = None, sao2: float = 98.0) -> HemoState:
        """Advance the model by dt seconds and return the new state.

        Args:
            cp_prop: Propofol plasma concentration (mcg/mL).
            cp_remi: Remifentanil plasma concentration (ng/mL).
            ce_nore: Norepinephrine effect-site concentration (ng/mL).
            pit: Intrathoracic pressure (mmHg).
            paco2, pao2: Arterial gas tensions (mmHg).
            dist_hr, dist_sv, dist_svr: Stimulation deltas (bpm, mL, Wood units).
            mac_sevo: End-tidal sevoflurane MAC.
            ce_epi, ce_phenyl, ce_dobu, ce_mil: Effect-site concentrations (ng/mL).
            ce_vaso: Vasopressin effect-site concentration (mU/L).
            temp_c: Core temperature (°C).
            peep_cmH2O: Total PEEP (cmH2O).
            sao2: Arterial saturation (%).
        """
        self._cached_state = None
        dt_min = dt / 60.0
        sepsis_sev = clamp01(self.sepsis_severity)
        anaph_sev = clamp01(self.anaphylaxis_severity)

        # Sevoflurane cardiovascular effect site, driven by end-tidal MAC.
        self.ce_sevo += self.ke0_sevo * (mac_sevo - self.ce_sevo) * dt_min

        # Urine output, scaled by renal perfusion and function.
        map_prev = self._prev_map
        map_denom = max(1e-3, self.renal_map_norm - self.renal_map_min)
        renal_factor = clamp01((map_prev - self.renal_map_min) / map_denom)
        renal_factor *= max(0.0, self.patient.renal_function)
        if self.vol_clearance is not None:
            base_clearance_ml_min = max(0.0, float(self.vol_clearance))
        else:
            base_clearance_ml_min = max(
                0.0,
                self.uop_ml_kg_hr * self.patient.weight / 60.0,
            )
        urine_out_ml = min(
            base_clearance_ml_min * renal_factor * dt_min,
            max(0.0, self.blood_volume - BLOOD_VOLUME_MIN),
        )
        blood_volume = self.blood_volume - urine_out_ml
        if urine_out_ml > 0:
            self.total_urine_out_ml += urine_out_ml
        if sepsis_sev > 0.0:
            # Capillary leak loses plasma, not red cells.
            dt_hr = dt / 3600.0
            leak_fraction = self.sepsis_leak_fraction_per_hr * sepsis_sev
            leak_ml = blood_volume * leak_fraction * dt_hr
            available = max(0.0, blood_volume - BLOOD_VOLUME_MIN)
            actual_leak = min(leak_ml, available)
            if actual_leak > 0:
                blood_volume -= actual_leak
                self.total_leak_out_ml += actual_leak
                self.total_third_space_ml += actual_leak
        if self.total_third_space_ml > 0 and self.third_space_refill_tau_hr > 0:
            tau_s = self.third_space_refill_tau_hr * 3600.0
            frac = 1.0 - math.exp(-dt / max(tau_s, 1e-6))
            refill_ml = self.total_third_space_ml * frac
            if refill_ml > 0:
                self.total_third_space_ml -= refill_ml
                blood_volume += refill_ml
        self.blood_volume = max(BLOOD_VOLUME_MIN, blood_volume)
        self._update_hb_conc()

        (total_eff_tpr, total_eff_sv, total_eff_hr_prod,
         eff_remi_tpr, eff_remi_sv, eff_remi_hr) = self._calc_anesthetic_effects(cp_prop, cp_remi, self.ce_sevo)

        # Positive intrathoracic pressure reduces venous return; spontaneous
        # negative pressure modestly augments it.
        delta_pit = pit - self.pit_0
        if delta_pit >= 0.0:
            f_preload_pit = 1.0 / (1.0 + self.alpha_peep * delta_pit)
        else:
            f_preload_pit = 1.0 + self.alpha_peep * (-delta_pit)
        self.f_preload_pit = clamp(f_preload_pit, 0.4, 1.4)

        f_frank_starling = self._update_pulmonary_coupling(
            dt=dt,
            pao2=pao2,
            peep_cmH2O=peep_cmH2O,
            f_preload_pit=self.f_preload_pit,
            sepsis_sev=sepsis_sev,
        )

        # Chemoreflex: hypercapnia raises HR and TPR; hypoxemia raises HR.
        e_co2 = max(0.0, (paco2 - self.paco2_set) / self.paco2_set)
        e_o2 = max(0.0, (self.pao2_set - pao2) / self.pao2_set)
        chemo_hr_boost = self.g_hr_co2 * e_co2 + self.g_hr_o2 * e_o2
        chemo_tpr_factor = 1.0 + self.k_tpr_co2 * e_co2

        # Fast baroreflex. Propofol depresses both limbs (Sato 2005).
        sensed_error = map_prev - self._stim_map - self._baro_setpoint
        baro_gain = self.baro_gain_brady if sensed_error > 0.0 else self.baro_gain_tachy
        baro_gain *= 1.0 - self.baro_anesthetic_depression * clamp01(self.ce_sevo + cp_prop / 4.0)
        baro_hr = clamp(-baro_gain * sensed_error, -self.baro_max_hr_change, self.baro_max_hr_change)
        self._baro_setpoint += sensed_error * min(1.0, dt / self.baro_reset_tau_s)

        # Myocardial hypoxia; recovery after reoxygenation needs coronary perfusion.
        hypoxia_target = clamp01(
            (self.hypoxia_sao2_onset - sao2) / (self.hypoxia_sao2_onset - self.hypoxia_sao2_full)
        )
        if hypoxia_target > self.myocardial_hypoxia:
            hypoxia_tau = self.hypoxia_tau_on_s
        else:
            hypoxia_tau = self.hypoxia_tau_off_s / clamp(map_prev / self.patient.baseline_map, 0.1, 1.0)
        self.myocardial_hypoxia += (hypoxia_target - self.myocardial_hypoxia) * min(1.0, dt / hypoxia_tau)

        self.hemorrhage_hr_mult, self.hemorrhage_tpr_mult = self._calc_hemorrhage_response()
        sepsis_hr_mult = 1.0 + self._sepsis_hr_gain * sepsis_sev

        self._epi_pressor_ce += (max(0.0, ce_epi) - self._epi_pressor_ce) * (
            -math.expm1(-dt / self.epi_tau_pressor_s)
        )
        epi_delta_hr, epi_sv_factor, epi_svr_factor = self._calc_epi_effects(ce_epi, self._epi_pressor_ce)
        self._epi_chrono_effect += (epi_delta_hr - self._epi_chrono_effect) * (
            -math.expm1(-dt / self.epi_tau_hr_s)
        )
        nore_delta_hr, nore_sv_factor, nore_svr_factor = self._calc_nore_effects(ce_nore)
        phenyl_svr_factor = self._calc_phenyl_effects(ce_phenyl)
        vaso_delta_hr, _, vaso_svr_factor = self._calc_hr_sv_svr_effects(
            ce_vaso, self.vaso_c50, self.vaso_gamma, self.vaso_emax_hr, 0.0, self.vaso_emax_svr
        )
        dobu_delta_hr, dobu_sv_factor, dobu_svr_factor = self._calc_hr_sv_svr_effects(
            ce_dobu, self.dobu_c50, self.dobu_gamma, self.dobu_emax_hr, self.dobu_emax_sv, self.dobu_emax_svr
        )
        mil_delta_hr, mil_sv_factor, mil_svr_factor = self._calc_hr_sv_svr_effects(
            ce_mil, self.mil_c50, self.mil_gamma, self.mil_emax_hr, self.mil_emax_sv, self.mil_emax_svr
        )

        # SVR and SV factors multiply; chronotropy adds. Sepsis blunts catecholamines.
        catechol_svr_factor = epi_svr_factor * nore_svr_factor * phenyl_svr_factor
        pressor_resistance = clamp01(self.sepsis_pressor_resistance * sepsis_sev)
        if pressor_resistance > 0:
            catechol_svr_factor = 1.0 + (catechol_svr_factor - 1.0) * (1.0 - pressor_resistance)

        combined_svr_factor = (
            catechol_svr_factor *
            vaso_svr_factor *
            dobu_svr_factor *
            mil_svr_factor
        )
        combined_sv_factor = epi_sv_factor * nore_sv_factor * dobu_sv_factor * mil_sv_factor
        combined_delta_hr = self._epi_chrono_effect + nore_delta_hr + vaso_delta_hr + dobu_delta_hr + mil_delta_hr
        self.vasopressor_sv_factor = combined_sv_factor
        self.delta_tpr_vasopressors = self.base_tpr * (combined_svr_factor - 1.0)

        # Su feedback senses RMAP = HR * SV * TPR relative to baseline, including
        # drug, preload, and this step's stimulation effects.
        current_hr = self._calc_hr()
        term = max(0.1, 1.0 - self.hr_sv_coupling * math.log(max(1.0, current_hr / self.base_hr)))
        raw_sv = (self.sv_star + self.tde_sv) * term
        current_sv = raw_sv * f_frank_starling * combined_sv_factor
        distributive_svr_drop = (self.sepsis_svr_drop_wood * sepsis_sev +
                                 self.anaphylaxis_svr_drop_wood * anaph_sev)
        distributive_tpr_offset = -distributive_svr_drop / 1000.0
        effective_tpr = self.tpr + self.delta_tpr_vasopressors + (dist_svr / 1000.0) + distributive_tpr_offset
        effective_tpr = max(0.006, effective_tpr)

        rmap = clamp((current_hr * current_sv * effective_tpr) * self._inv_base_rmap_denom, 0.1, 5.0)
        rmap_fb = rmap ** self.fb

        # Thermoregulatory vasoconstriction below 36.5 °C. Anesthesia lowers the
        # threshold by up to 2.5 °C at 1 MAC or propofol 4 mcg/mL (Sessler 2000).
        depth_metric = mac_sevo + (cp_prop / 4.0)
        threshold_drop = 2.5 * min(1.0, depth_metric)
        vasoconstriction_threshold = 36.5 - threshold_drop

        thermo_tpr_mult = 1.0
        if temp_c < vasoconstriction_threshold:
            thermo_tpr_mult = 1.0 + TEMP_TPR_COEFFICIENT * (vasoconstriction_threshold - temp_c)
        thermo_tpr_mult = min(2.0, thermo_tpr_mult)

        # Turnover equations; inotropy and preload act on output SV only.
        tpr_production = self.kin_tpr * rmap_fb * (1.0 + total_eff_tpr) * chemo_tpr_factor * self.hemorrhage_tpr_mult * thermo_tpr_mult
        tpr_dissipation = self.kout * self.tpr * (1.0 - eff_remi_tpr)
        d_tpr = tpr_production - tpr_dissipation
        sv_production = self.kin_sv * rmap_fb * (1.0 + total_eff_sv)
        sv_dissipation = self.kout * self.sv_star * (1.0 - eff_remi_sv)
        d_sv_star = sv_production - sv_dissipation
        # eff_remi_hr can be negative with propofol, which speeds HR dissipation.
        hr_production = self.kin_hr * rmap_fb * (1.0 + total_eff_hr_prod) * self.hemorrhage_hr_mult * sepsis_hr_mult
        hr_dissipation = self.kout * self.hr_star * (1.0 - eff_remi_hr)
        d_hr_star = hr_production - hr_dissipation

        alpha_fast = min(1.0, dt / self.tau_hr_fast)
        self.smoothed_chemo_hr += (chemo_hr_boost - self.smoothed_chemo_hr) * alpha_fast
        self.smoothed_baro_hr += (baro_hr - self.smoothed_baro_hr) * alpha_fast
        self.smoothed_epi_hr += (combined_delta_hr - self.smoothed_epi_hr) * alpha_fast

        self.tpr += d_tpr * dt_min
        self.sv_star += d_sv_star * dt_min
        self.hr_star += d_hr_star * dt_min
        self.tde_hr -= self.k_drift * self.tde_hr * dt_min
        self.tde_sv -= self.k_drift * self.tde_sv * dt_min

        self.dist_hr = dist_hr
        self.dist_sv = dist_sv
        self.dist_svr = dist_svr

        hr_base_for_state = self._calc_hr()
        computed_state = self._compute_state(
            preload_sv_factor=f_frank_starling,
            sepsis_sev=sepsis_sev,
            anaph_sev=anaph_sev,
            distributive_tpr_offset=distributive_tpr_offset,
            hr_base=hr_base_for_state,
        )
        self._prev_map = computed_state.map
        return computed_state

    def calculate_steady_state(self, cp_prop: float, cp_remi: float, ce_nore: float, mac_sevo: float = 0.0) -> HemoState:
        """Return the steady state for constant plasma propofol and remifentanil.

        At steady state the turnover equations reduce to one unknown, RMAP,
        solved as a root of Z_calc(Z) - Z. The model's own state is restored.
        """
        saved_tpr = self.tpr
        saved_sv = self.sv_star
        saved_hr = self.hr_star
        saved_dist = (self.tde_sv, self.tde_hr)
        self.tde_sv = 0
        self.tde_hr = 0

        cn = max(0.0, ce_nore)
        (total_eff_tpr, total_eff_sv, total_eff_hr_prod,
         eff_remi_tpr, eff_remi_sv, eff_remi_hr) = self._calc_anesthetic_effects(
            cp_prop, cp_remi, mac_sevo
        )
        nore_delta_hr, nore_sv_factor, nore_svr_factor = self._calc_nore_effects(cn)

        def residual(z):
            if z <= 0.01:
                z = 0.01
            z_fb = z ** self.fb
            hr_z = self.base_hr * z_fb * (1.0 + total_eff_hr_prod) / (1.0 - eff_remi_hr)
            hr_z += nore_delta_hr
            sv_star_z = self.base_sv * z_fb * (1.0 + total_eff_sv) / (1.0 - eff_remi_sv)
            tpr_z = self.base_tpr * z_fb * (1.0 + total_eff_tpr) / (1.0 - eff_remi_tpr)
            term = 1.0 - self.hr_sv_coupling * math.log(max(1.0, hr_z / self.base_hr))
            sv_z = sv_star_z * term * nore_sv_factor
            eff_tpr_z = tpr_z + self.base_tpr * (nore_svr_factor - 1.0)
            z_new = (hr_z * sv_z * eff_tpr_z) / (self.base_hr * self.base_sv * self.base_tpr)
            return z_new - z

        try:
            sol = root_scalar(residual, bracket=[0.01, 5.0], method='brentq')
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError("Unable to solve hemodynamic steady state") from exc
        if not sol.converged:
            raise RuntimeError("Hemodynamic steady-state solver did not converge")
        z_ss = sol.root

        z_fb = z_ss ** self.fb
        self.hr_star = self.base_hr * z_fb * (1.0 + total_eff_hr_prod) / (1.0 - eff_remi_hr)
        self.sv_star = self.base_sv * z_fb * (1.0 + total_eff_sv) / (1.0 - eff_remi_sv)
        self.tpr = self.base_tpr * z_fb * (1.0 + total_eff_tpr) / (1.0 - eff_remi_tpr)
        self.tde_hr = 0
        self.tde_sv = 0
        self.ce_sevo = mac_sevo

        saved_epi = (self._epi_pressor_ce, self._epi_chrono_effect)
        self._epi_pressor_ce = self._epi_chrono_effect = 0.0
        saved_pressor = (self.smoothed_epi_hr, self.vasopressor_sv_factor, self.delta_tpr_vasopressors)
        self.smoothed_epi_hr = nore_delta_hr
        self.vasopressor_sv_factor = nore_sv_factor
        self.delta_tpr_vasopressors = self.base_tpr * (nore_svr_factor - 1.0)

        ret = self.step(0.0, cp_prop, cp_remi, ce_nore, -2.0, 40.0, 95.0, 0, 0, 0, mac_sevo=mac_sevo)

        self.tpr = saved_tpr
        self.sv_star = saved_sv
        self.hr_star = saved_hr
        self.tde_sv = saved_dist[0]
        self.tde_hr = saved_dist[1]
        self.smoothed_epi_hr, self.vasopressor_sv_factor, self.delta_tpr_vasopressors = saved_pressor
        self._epi_pressor_ce, self._epi_chrono_effect = saved_epi
        self._cached_state = None
        return ret
