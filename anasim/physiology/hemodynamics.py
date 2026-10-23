import math
from itertools import pairwise

from scipy.optimize import root_scalar

from anasim.core.constants import BLOOD_VOLUME_MIN, HR_MAX, HR_MIN, TEMP_TPR_COEFFICIENT
from anasim.core.enums import RhythmType
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.patient.conditions import HEART_FAILURE_PROFILES
from anasim.patient.patient import Patient
from anasim.patient.pd.anesthesia import KETAMINE_MAC_EQUIVALENT

from .coronary import CoronaryModel
from .fluids import CrystalloidKinetics
from .hemo_config import HemodynamicConfig
from .hemo_types import HemoState

# Ventricular rates set by the rhythm: SVT 150-220, VT 150-250 bpm.
FIXED_RHYTHM_RATE = {
    RhythmType.SVT: 160.0,
    RhythmType.VTACH: 180.0,
    RhythmType.VFIB: 0.0,
    RhythmType.ASYSTOLE: 0.0,
}
PULSELESS_RHYTHMS = (RhythmType.VFIB, RhythmType.ASYSTOLE)
# Stroke volume relative to sinus rhythm at the same filling.
RHYTHM_SV_FACTOR = {
    RhythmType.AFIB: 0.68,  # Lost atrial kick (~20%) and R-R irregularity (-15% CO; Clark 1997)
    RhythmType.SVT: 0.50,  # CO about 80-90% of baseline
    RhythmType.VTACH: 0.25,  # CO about 40% of baseline
    RhythmType.VFIB: 0.0,
    RhythmType.ASYSTOLE: 0.0,
}


def _interpolate(x: float, points: tuple[tuple[float, float], ...]) -> float:
    """Piecewise-linear interpolation, constant beyond the end points."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in pairwise(points):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


class HemodynamicModel:
    """Su et al. Br J Anaesth. 2023 drug effects from hemodynamic effect sites
    on slow turnover tone, with volume, pulmonary circulation, reflexes,
    vasoactive drugs, and distributive shock.

    Units: MAP mmHg, HR bpm, SV mL, CO L/min, SVR Wood units (mmHg·min/L),
    internal TPR mmHg·min/mL. Sources are listed in docs/REFERENCES.md.
    """
    def __init__(self, patient: Patient, config: HemodynamicConfig | None = None):
        self.patient = patient
        self.config = cfg = config or HemodynamicConfig()
        self.failure = HEART_FAILURE_PROFILES[patient.heart_failure]
        # Heart failure applies all afterload through ventricular-arterial coupling.
        self._phenyl_afterload_gain = cfg.phenyl_afterload_gain if patient.heart_failure == "none" else 0.0
        self.position = patient.position
        # Supine preload that gives the pregnancy's positional CO loss on the
        # Frank-Starling curve at normal filling.
        loss = patient.pregnancy.supine_output_loss
        self._supine_preload = -math.log(1.0 - (1.0 - loss) * (1.0 - math.exp(-2.0))) / 2.0 if loss else 1.0
        self._position_preload = self._supine_preload if self.position == "supine" else 1.0
        self._cached_state: HemoState | None = None

        self.baseline_hb = patient.baseline_hb
        assert patient.baseline_hct is not None  # Patient derives it from hemoglobin when omitted.
        self.baseline_hct = patient.baseline_hct

        self.base_hr = patient.baseline_hr * patient.pregnancy.heart_rate
        # Pregnancy lowers the PaCO2 the chemoreflex defends.
        self._paco2_set = cfg.paco2_set + patient.pregnancy.paco2 - 40.0
        years = patient.age - cfg.age_reference
        self._age_chronotropy = 1.0 + cfg.age_sympathetic_hr * years
        self._age_vagal_baroreflex = math.exp(cfg.age_vagal_baroreflex * years)

        # Baseline SV from cardiac index and BSA.
        co_0 = cfg.cardiac_index * patient.bsa * patient.cardiac_output_factor
        self.base_sv = (co_0 * 1000.0) / self.base_hr

        flow_ml_min = self.base_hr * self.base_sv
        self.base_tpr = patient.baseline_map / flow_ml_min
        self.base_co_l_min = flow_ml_min / 1000.0
        self.baseline_do2 = self.calc_oxygen_content(self.baseline_hb, 0.98, 95.0) * self.base_co_l_min * 10.0
        self._emax_prop_sv_age = cfg.emax_prop_sv_typ * math.exp(cfg.age_emax_sv * (patient.age - 35.0))
        # Share of propofol's sympathetic withdrawal relative to a patient at the reference pressure.
        self._sympathetic_gain = clamp(
            1.0 + cfg.propofol_baseline_sensitivity * (patient.baseline_map - cfg.propofol_baseline_map_ref),
            *cfg.propofol_baseline_gain_range,
        )
        self._inv_base_rmap_denom = 1.0 / (self.base_hr * self.base_sv * self.base_tpr)
        self._inv_base_co_l_min = 1.0 / max(0.1, self.base_co_l_min)

        self.delta_tpr_vasopressors = 0.0
        self.dist_svr = 0.0
        self.dist_hr = 0.0
        self.dist_sv = 0.0
        self._sepsis_severity = 0.0
        self._anaphylaxis_severity = 0.0
        # Adrenoceptor occupancy and muscarinic block, 0-1.
        self._beta_occupancy = 0.0
        self._vagal_block = 0.0
        self._ephedrine_tolerance_ce = 0.0

        # Fast HR effects (chemoreflex, baroreflex, vasoactive chronotropy).
        self.smoothed_epi_hr = 0.0
        self._epi_pressor_ce = 0.0
        self._epi_chrono_effect = 0.0
        self._epi_beta2_effect = 0.0
        self.smoothed_chemo_hr = 0.0
        self.smoothed_baro_hr = 0.0
        self._baro_setpoint = patient.baseline_map
        self._stim_map = 0.0  # MAP attributable to noxious stimulation
        self._rmap_fb = 1.0  # Turnover feedback, (MAP / baseline) ** fb
        self.myocardial_hypoxia = 0.0  # 0-1 depression of rate and contractility

        # Slow tone: baselines adjusted by MAP feedback, chemoreflex, blood
        # loss, and temperature. Anesthetic effects multiply it from their own
        # effect sites, so drug onset and homeostatic recovery keep separate rates.
        self.sv_tone = self.base_sv
        self.hr_tone = self.base_hr
        self.tpr_tone = self.base_tpr
        self._drug_tpr = self._drug_sv = self._drug_hr = 1.0
        self.ce_prop_cv = self.ce_opioid_cv = self.ce_remi_vagal = 0.0
        self._sympathetic_venous_depth = 0.0
        self._epi_venous_alpha = 0.0  # Share of propofol's venous pooling reversed by epinephrine
        self._ke0_prop_cv = math.log(2.0) / _interpolate(patient.age, cfg.prop_cv_half_time_by_age)
        self._ke0_opioid_cv = math.log(2.0) / cfg.opioid_cv_half_time
        # Baselines are resting values, so Su's pre-infusion anxiety transient
        # starts at zero.
        self.tde_sv = 0.0
        self.tde_hr = 0.0
        self._frank_starling_baseline_raw = 1.0 - math.exp(-2.0)
        # Production rates that give equilibrium at RMAP = 1.
        self.kin_tpr = cfg.kout * self.base_tpr
        self.kin_sv = cfg.kout * self.base_sv
        self.kin_hr = cfg.kout * self.base_hr
        self.ce_sevo = 0.0

        self.blood_volume = patient.estimate_blood_volume()
        self.blood_volume_0 = self.blood_volume
        self.hb_mass = self.baseline_hb * (self.blood_volume / 100.0)  # g
        self.red_cell_volume_ml = self.baseline_hct * self.blood_volume
        self.hb_conc = self.baseline_hb
        # Hemorrhage depletes stressed volume first, lowering MCFP and preload.
        self.unstressed_volume = self.blood_volume * cfg.unstressed_volume_fraction
        self._venous_tone = 1.0  # Share of reflex venoconstriction left after alpha block
        self.cv = cfg.venous_compliance  # mL/mmHg
        # Guyton MCFP = stressed volume / venous compliance, about 10-15 mmHg.
        stressed_vol_0 = max(0.0, self.blood_volume - self.unstressed_volume)
        self._baseline_stressed_volume = stressed_vol_0
        self._venous_stressed_fraction = 1.0
        self._pressor_recruitment_ml = 0.0
        self._venous_reference_mcfp = stressed_vol_0 / self.cv
        self.mcfp_0 = max(cfg.mcfp_floor, stressed_vol_0 / self.cv)

        # Venous return resistance (Wood units) calibrated to baseline CO.
        delta_p_vr = max(0.1, self.mcfp_0 - cfg.rap_baseline)
        self.venous_return_resistance = delta_p_vr / max(0.1, self.base_co_l_min)
        self._baseline_venous_resistance = self.venous_return_resistance
        self._pulm_flow_factor = 1.0
        self._pulm_reference_factor = 1.0
        self._last_mcfp = self.mcfp_0
        self._last_rap = cfg.rap_baseline
        self._last_pvr = cfg.pvr_wood_baseline
        self._last_rv_co = self.base_co_l_min
        self._last_lv_inflow = self.base_co_l_min
        self._last_preload_factor = 1.0
        self._last_preload_sv_factor = 1.0
        self._last_lap = self._lap(1.0)
        # Chronic left atrial hypertension raises lymph flow, so edema begins
        # the same margin above resting LAP as with a normal heart.
        normal_lap = cfg.lap_scale * math.expm1(cfg.lap_preload_gain)
        self._lung_water_threshold = cfg.lung_water_lap_threshold + self._last_lap - normal_lap
        self.coronary = CoronaryModel(patient.coronary_flow_reserve, self.base_hr,
                                      patient.baseline_map, self._last_lap)
        self._pbw = patient.predicted_body_weight()
        self.lung_water_ml = 0.0  # Extravascular lung water above normal

        self.total_crystalloid_in_ml = 0.0
        self.total_colloid_in_ml = 0.0
        self.total_blood_in_ml = 0.0
        self.total_urine_out_ml = 0.0
        self.total_blood_out_ml = 0.0
        self.total_third_space_ml = 0.0
        self.crystalloid = CrystalloidKinetics()

        self._rhythm_type = RhythmType.SINUS
        self.f_preload_pit = 1.0
        self.renal_perfusion = 1.0  # Share of normal renal blood flow, from MAP
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

    def add_volume(
        self,
        amount_ml: float,
        hematocrit: float = 0.0,
        retention_fraction: float | None = None,
        label: str = "crystalloid",
    ):
        """Add fluid or blood (positive) or remove blood (negative).

        Retained volume also gives a transient Frank-Starling SV rise via tde_sv.
        """
        cfg = self.config
        self._cached_state = None
        if amount_ml == 0:
            return

        retained_ml = 0.0
        if amount_ml < 0:
            loss_ml = -amount_ml
            available = max(0.0, self.blood_volume - BLOOD_VOLUME_MIN)
            actual_loss = min(loss_ml, available)
            if actual_loss > 0:
                self.crystalloid.central_ml *= 1.0 - actual_loss / self.blood_volume
                self.red_cell_volume_ml *= 1.0 - actual_loss / self.blood_volume
                self.blood_volume -= actual_loss
                self.total_blood_out_ml += actual_loss
                hb_loss = self.hb_conc * actual_loss / 100.0
                self.hb_mass = max(0.0, self.hb_mass - hb_loss)
        else:
            if hematocrit > 0:
                self.total_blood_in_ml += amount_ml
                retained_ml = amount_ml * cfg.blood_retention_fraction
                self.total_third_space_ml += max(0.0, amount_ml - retained_ml)
                donor_red_cells_ml = hematocrit * retained_ml
                self.red_cell_volume_ml += donor_red_cells_ml
                # Donor MCHC is independent of the recipient's Hb/Hct ratio.
                self.hb_mass += donor_red_cells_ml * cfg.donor_mchc_g_dl / 100.0
            else:
                is_colloid = (label == "colloid")
                if is_colloid:
                    self.total_colloid_in_ml += amount_ml
                    retention = cfg.colloid_retention_fraction
                else:
                    self.total_crystalloid_in_ml += amount_ml
                    retention = 1.0
                if retention_fraction is not None:
                    retention = retention_fraction
                retention = clamp01(retention)
                retained_ml = amount_ml * retention
                self.total_third_space_ml += max(0.0, amount_ml - retained_ml)
                if not is_colloid:
                    self.crystalloid.central_ml += retained_ml

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

    def _calc_stressed_volume(self, sepsis_sev: float | None = None) -> float:
        """Stressed volume after drug effects, sepsis pooling, and venous reflexes.

        Sepsis shifts part of baseline volume to unstressed. Propofol's loss of
        sympathetic venous tone pools a share of stressed blood, up to that
        share of the baseline stressed volume, so blood loss mobilizes it and
        fluid loading does not refill it; epinephrine's alpha tone restores it.
        Venoconstriction returns part of any blood-volume deficit to the
        stressed pool.
        """
        cfg = self.config
        sev = clamp01(self.sepsis_severity) if sepsis_sev is None else sepsis_sev
        pooling = cfg.sepsis_pooling_fraction * self.blood_volume_0 * sev
        deficit = max(0.0, self.blood_volume_0 - self.blood_volume)
        recruited = cfg.venous_recruitment_fraction * deficit * self._venous_tone
        available = max(0.0, self.blood_volume - self.unstressed_volume + recruited - pooling)
        available -= (cfg.propofol_sympathetic_pooling * self._sympathetic_gain * self._sympathetic_venous_depth
                      * (1.0 - self._epi_venous_alpha) * min(available, self._baseline_stressed_volume))
        return min(self.blood_volume, available * self._venous_stressed_fraction + self._pressor_recruitment_ml)

    def _update_venous_drug_effects(self, ce_prop: float, ce_phenyl: float, pressor_gain: float = 1.0) -> None:
        """Venous drug effects, normalized to preserve Su's normal-volume response."""
        cfg = self.config
        propofol = clamp(ce_prop, 0.0, 6.5)
        stressed_fraction = math.exp(-cfg.propofol_venous_pooling_slope * propofol)
        self._venous_stressed_fraction = stressed_fraction
        self._venous_reference_mcfp = self._baseline_stressed_volume * stressed_fraction / self.cv
        self.venous_return_resistance = self._baseline_venous_resistance * math.exp(
            -cfg.propofol_venous_resistance_slope * propofol
        )
        recruitment = cfg.phenyl_venous_recruitment_fraction * self._baseline_stressed_volume * hill_function(
            max(0.0, ce_phenyl), cfg.phenyl_c50, cfg.phenyl_gamma,
        )
        self._pressor_recruitment_ml = min(
            recruitment * pressor_gain,
            self.unstressed_volume + self._baseline_stressed_volume * (1.0 - stressed_fraction),
        )

    def get_hematocrit(self) -> float:
        return clamp(self.red_cell_volume_ml / self.blood_volume, 0.0, 0.7)

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

    def _lap(self, preload_factor: float) -> float:
        """Left atrial pressure (mmHg) from LV filling relative to baseline."""
        if self.patient.heart_failure != "none":
            gain = self.config.lap_preload_gain * self.failure.stiffness_factor
            return self.failure.filling_pressure * math.expm1(gain * preload_factor) / math.expm1(gain)
        return self.config.lap_scale * math.expm1(self.config.lap_preload_gain * preload_factor)

    @property
    def lung_water_ml_kg(self) -> float:
        """Extravascular lung water above normal, mL/kg predicted body weight."""
        return self.lung_water_ml / self._pbw

    def _frank_starling(self, preload_factor: float, inotropy: float = 1.0) -> float:
        """Preload response with a plateau at normal filling (Rebet 2016)."""
        pf = clamp(preload_factor, 0.01, 1.0)
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

    def _calc_epi_effects(
        self, ce_epi: float, ce_pressor: float, beta1_dr: float = 1.0, alpha_dr: float = 1.0
    ) -> tuple[float, float, float]:
        """Return direct chronotropy, SV factor, and baseline-relative SVR factor.

        The lagged pressor concentration separates the HR and SBP peaks after a
        bolus; at steady exposure both concentrations are equal. Beta-2 dilation
        scales with current vascular tone; alpha tone adds relative to baseline.
        Antagonist dose ratios divide the concentration seen by each receptor.
        """
        cfg = self.config
        delta_hr = cfg.epi_emax_hr * hill_function(ce_epi / beta1_dr, cfg.epi_c50_hr, cfg.epi_gamma_hr)
        delta_hr *= (1.0 - cfg.epi_volatile_hr_depression * clamp01(self.ce_sevo)) * self._age_chronotropy
        sv_factor = 1.0 + cfg.epi_emax_sv * hill_function(ce_pressor / beta1_dr, cfg.epi_c50_sv, 1.0)
        alpha = cfg.epi_emax_svr_alpha * hill_function(
            ce_pressor / alpha_dr, cfg.epi_c50_alpha, cfg.epi_gamma_alpha
        )
        svr_factor = 1.0 + self._epi_beta2_effect * self.tpr / self.base_tpr + alpha
        return delta_hr, sv_factor, max(0.2, svr_factor)

    def _calc_phenyl_effects(self, ce_phenyl: float) -> float:
        """Phenylephrine SVR effect (pure alpha-1)."""
        cfg = self.config
        if ce_phenyl <= 0:
            return 1.0
        return 1.0 + cfg.phenyl_emax_svr * hill_function(ce_phenyl, cfg.phenyl_c50, cfg.phenyl_gamma)

    @staticmethod
    def _calc_hr_sv_svr_effects(ce: float, c50: float, gamma: float,
                                emax_hr: float, emax_sv: float, emax_svr: float) -> tuple:
        """Return (delta HR, SV factor, SVR factor) from one Hill effect."""
        if ce <= 0:
            return 0.0, 1.0, 1.0
        hill = hill_function(ce, c50, gamma)
        return emax_hr * hill, 1.0 + emax_sv * hill, max(0.5, 1.0 + emax_svr * hill)

    def _calc_pvr_factor(self, pao2: float, peep_cmH2O: float | None = None) -> float:
        """PVR multiplier from hypoxic vasoconstriction and PEEP."""
        cfg = self.config
        pvr_factor = 1.0
        if pao2 < cfg.pvr_o2_threshold:
            denom = max(1.0, cfg.pvr_o2_threshold - cfg.pvr_o2_floor)
            frac = clamp01((cfg.pvr_o2_threshold - pao2) / denom)
            pvr_factor *= 1.0 + (cfg.pvr_o2_max_factor - 1.0) * frac
        if peep_cmH2O is not None:
            peep_excess = max(0.0, peep_cmH2O - cfg.pvr_peep_ref)
            pvr_factor *= 1.0 + cfg.pvr_peep_slope * peep_excess

        return clamp(pvr_factor, 0.2, cfg.pvr_max_factor)

    def _update_pulmonary_coupling(
        self,
        dt: float,
        pao2: float,
        peep_cmH2O: float | None,
        f_preload_pit: float,
        sepsis_sev: float,
    ) -> float:
        """Return the Frank-Starling factor after venous return and pulmonary transit.

        Venous return follows MCFP - RAP, is attenuated by PVR, and reaches the
        left heart through a first-order transit delay. Below baseline filling,
        RAP falls in proportion to MCFP, the intersection with a cardiac
        function curve that is linear at low filling pressures.
        """
        cfg = self.config
        stressed_vol = self._calc_stressed_volume(sepsis_sev)
        mcfp = stressed_vol / self.cv if self.cv > 0 else self.mcfp_0

        rap = cfg.rap_baseline * min(1.0, mcfp / self.mcfp_0)
        delta_p = max(0.0, mcfp - rap)
        vr_flow_l_min = delta_p / max(0.1, self.venous_return_resistance)
        vr_flow_factor = vr_flow_l_min * self._inv_base_co_l_min

        pvr_factor = self._calc_pvr_factor(pao2, peep_cmH2O)
        rv_out_target_factor = vr_flow_factor * (pvr_factor ** (-cfg.pvr_flow_exponent))
        reference_rap = cfg.rap_baseline * min(1.0, self._venous_reference_mcfp / self.mcfp_0)
        reference_flow_factor = (
            (self._venous_reference_mcfp - reference_rap) / self.venous_return_resistance
            * self._inv_base_co_l_min
        )

        if cfg.pulmonary_transit_time_s > 0 and dt > 0:
            alpha = -math.expm1(-dt / cfg.pulmonary_transit_time_s)
            self._pulm_flow_factor += (rv_out_target_factor - self._pulm_flow_factor) * alpha
            self._pulm_reference_factor += (reference_flow_factor - self._pulm_reference_factor) * alpha
        elif cfg.pulmonary_transit_time_s <= 0:
            self._pulm_flow_factor = rv_out_target_factor
            self._pulm_reference_factor = reference_flow_factor

        self._pulm_flow_factor = clamp(self._pulm_flow_factor, 0.05, 3.0)

        position_target = self._supine_preload if self.position == "supine" else 1.0
        if self._position_preload != position_target:
            self._position_preload += (position_target - self._position_preload) * -math.expm1(-dt / 10.0)
        f_preload = self._pulm_flow_factor / self._pulm_reference_factor * f_preload_pit * self._position_preload
        f_frank_starling = self._frank_starling(f_preload)

        self._last_lap = self._lap(f_preload)
        self._last_mcfp = mcfp
        self._last_rap = rap
        self._last_pvr = cfg.pvr_wood_baseline * pvr_factor
        self._last_rv_co = rv_out_target_factor * self.base_co_l_min
        self._last_lv_inflow = self._pulm_flow_factor * self.base_co_l_min
        self._last_preload_factor = f_preload
        self._last_preload_sv_factor = f_frank_starling

        return f_frank_starling

    def _calc_nore_effects(
        self, ce_nore: float, beta1_dr: float = 1.0, alpha_dr: float = 1.0,
        anesthetic_depth: float = 0.0,
    ) -> tuple:
        """Return direct chronotropy, stroke-volume factor, and vascular-resistance factor.

        Reflex bradycardia comes from the baroreflex.
        """
        cfg = self.config
        if ce_nore <= 0:
            return 0.0, 1.0, 1.0
        beta_hill = hill_function(ce_nore / beta1_dr, cfg.nore_c50, 1.0)
        delta_hr = cfg.nore_emax_hr * beta_hill * self._age_chronotropy
        sv_gain = cfg.nore_emax_sv + cfg.nore_anesthetic_sv_gain * anesthetic_depth
        sv_factor = 1.0 + sv_gain * beta_hill
        alpha_hill = hill_function(ce_nore / alpha_dr, cfg.nore_c50, 1.0)
        alpha_gain = 1.0 + cfg.nore_anesthetic_alpha_gain * anesthetic_depth
        svr_factor = 1.0 + cfg.nore_emax_svr * alpha_gain * alpha_hill
        return delta_hr, sv_factor, svr_factor

    def _hr_feedback(self, map_feedback: float, tachycardia_depth: float) -> float:
        # Anesthesia removes the tachycardic side of the slow MAP feedback;
        # pressure-induced slowing remains (Takahashi 2002's late epinephrine fall).
        return map_feedback - max(0.0, map_feedback - 1.0) * (
            self.config.hr_feedback_anesthetic_depression * tachycardia_depth
        )

    def _calc_anesthetic_effects(self, ce_prop: float, ce_opioid: float, ce_sevo: float) -> tuple:
        """Combined anesthetic effects on TPR, SV, and HR (Su et al. 2023).

        Propofol and remifentanil-equivalent concentrations are at the
        hemodynamic effect sites. Returns (total_eff_tpr, total_eff_sv,
        total_eff_hr, eff_remi_tpr, eff_remi_sv, eff_remi_hr).
        """
        cfg = self.config
        cp = max(0.0, ce_prop)
        cr = max(0.0, ce_opioid)
        if cfg.opioid_cv_saturation > 0.0:
            cr = cfg.opioid_cv_saturation * math.tanh(cr / cfg.opioid_cv_saturation)
        emax_prop_sv = self._emax_prop_sv_age

        # Propofol TPR effect, with remifentanil interaction.
        remi_int_term = cfg.int_tpr * (cr / (cfg.ec50_remi_tpr + cr + 1e-9))
        eff_prop_tpr = (cfg.emax_prop_tpr * self._sympathetic_gain + remi_int_term) * hill_function(
            cp, cfg.ec50_prop_tpr, cfg.gamma_prop)
        eff_prop_sv = emax_prop_sv * hill_function(cp, cfg.ec50_prop_sv, 1.0)

        eff_sevo_tpr = cfg.sevo_emax_tpr * hill_function(ce_sevo, cfg.sevo_ec50_tpr, cfg.sevo_gamma_tpr)
        eff_sevo_sv = cfg.sevo_emax_sv * hill_function(ce_sevo, cfg.sevo_ec50_sv, 1.0)

        # Propofol and sevoflurane act as independent fractions.
        total_eff_tpr = max(-0.95, (1.0 + eff_prop_tpr) * (1.0 + eff_sevo_tpr) - 1.0)
        total_eff_sv = max(-0.95, (1.0 + eff_prop_sv) * (1.0 + eff_sevo_sv) - 1.0)
        total_eff_hr = cfg.sevo_emax_hr * hill_function(ce_sevo, cfg.sevo_ec50_hr, 1.0)

        # Remifentanil effects divide tone, as Su's dissipation terms do at
        # steady state; propofol modulates the SV and HR slopes.
        eff_remi_tpr = cfg.emax_remi_tpr * hill_function(cr, cfg.ec50_remi_tpr, cfg.gamma_remi_tpr)
        slope_sv = cfg.sl_remi_sv + cfg.int_sv * (cp / (cfg.ec50_int_sv + cp + 1e-9))
        eff_remi_sv = slope_sv * cr
        slope_hr = cfg.sl_remi_hr + cfg.int_hr * (cp / (cfg.ec50_int_hr + cp + 1e-9))
        eff_remi_hr = max(-0.9, min(0.9, slope_hr * cr))

        return total_eff_tpr, total_eff_sv, total_eff_hr, eff_remi_tpr, eff_remi_sv, eff_remi_hr

    def _compute_state(
        self,
        preload_sv_factor: float | None = None,
        sepsis_sev: float | None = None,
        anaph_sev: float | None = None,
        distributive_tpr_offset: float | None = None,
        hr_base: float | None = None,
    ) -> HemoState:
        cfg = self.config
        if sepsis_sev is None:
            sepsis_sev = clamp01(self.sepsis_severity)
        if anaph_sev is None:
            anaph_sev = clamp01(self.anaphylaxis_severity)
        if distributive_tpr_offset is None:
            distributive_tpr_offset = -(
                cfg.sepsis_svr_drop_wood * sepsis_sev + cfg.anaphylaxis_svr_drop_wood * anaph_sev
            ) / 1000.0

        if hr_base is None:
            hr_base = self._calc_hr()
        sinus_hr = clamp(hr_base + self.dist_hr, HR_MIN, HR_MAX)
        current_hr = self._rhythm_rate(sinus_hr)
        reflex_tpr_factor = 1.0
        if self.rhythm_type != RhythmType.SINUS:
            # When the rhythm sets the rate, the baroreflex acts through its vascular limb.
            reflex_hr = self.smoothed_baro_hr
            blocked_hr = reflex_hr - (current_hr - self._rhythm_rate(sinus_hr - reflex_hr))
            reflex_tpr_factor = 1.0 + blocked_hr / max(sinus_hr - reflex_hr, HR_MIN)
        current_hr *= 1.0 - cfg.hypoxia_hr_depression * self.myocardial_hypoxia

        term = self._hr_sv_factor(current_hr)
        raw_sv = (self.sv_star + self.tde_sv) * term + self.dist_sv

        if preload_sv_factor is None:
            preload_sv_factor = self._last_preload_sv_factor if self._last_preload_sv_factor > 0 else 1.0
        # TPR floor of about 6 Wood units.
        eff_tpr = max(
            0.006,
            self.tpr + self.delta_tpr_vasopressors + self.dist_svr / 1000.0 + distributive_tpr_offset,
        ) * reflex_tpr_factor

        current_sv = raw_sv * preload_sv_factor * self.vasopressor_sv_factor
        coupling = self.failure.coupling_ratio
        if coupling:
            # SV scales with 1 / (1 + Ea/Ees), with Ea proportional to arterial resistance.
            load = max(0.2, eff_tpr / self.base_tpr)
            current_sv *= (1.0 + coupling) / (1.0 + coupling * load)
        # Both terms describe oxygen-limited myocardium; use the larger so
        # global hypoxemia is not counted twice.
        depression = max(1.0 - self.coronary.contractility_factor,
                         cfg.hypoxia_sv_depression * self.myocardial_hypoxia)
        current_sv = max(1.0, current_sv) * (1.0 - depression)

        current_sv *= RHYTHM_SV_FACTOR.get(self.rhythm_type, 1.0)

        if self.rhythm_type in PULSELESS_RHYTHMS or current_hr <= 0.0 or current_sv <= 0.0:
            current_hr = current_sv = co = map_val = svr_val = self._stim_map = 0.0
        else:
            co = current_hr * current_sv / 1000.0
            map_val = clamp(current_hr * current_sv * eff_tpr, 5.0, 300.0)
            svr_val = map_val / co
            # First-order share of MAP from stimulation; nociception resets the baroreflex.
            self._stim_map = map_val * (
                self.dist_hr / current_hr + self.dist_sv / max(raw_sv, 1.0) + self.dist_svr / 1000.0 / eff_tpr
            )

        self._cached_state = HemoState(
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
            lap=self._last_lap,
            pvr=self._last_pvr,
            rv_co=self._last_rv_co,
            lv_inflow=self._last_lv_inflow,
            preload_factor=self._last_preload_factor,
            rhythm_type=self.rhythm_type,
        )
        return self._cached_state

    def _rhythm_rate(self, sinus_hr: float) -> float:
        """Ventricular rate from the underlying sinus rate; untreated AF with RVR is 110-150 bpm."""
        fixed = FIXED_RHYTHM_RATE.get(self.rhythm_type)
        if fixed is not None:
            return fixed
        if self.rhythm_type == RhythmType.AFIB:
            return max(sinus_hr, 110.0 * (1.0 - self.config.beta_block_af_rate * self._beta_occupancy))
        if self.rhythm_type == RhythmType.SINUS_BRADY:
            return min(sinus_hr, 50.0 + self.config.glyco_brady_relief * self._vagal_block)
        return sinus_hr

    @property
    def state(self) -> HemoState:
        if self._cached_state is not None:
            return self._cached_state
        return self._compute_state()

    @property
    def tpr(self) -> float:
        return self.tpr_tone * self._drug_tpr

    @property
    def sv_star(self) -> float:
        return self.sv_tone * self._drug_sv

    @property
    def hr_star(self) -> float:
        return self.hr_tone * self._drug_hr

    def _set_drug_factors(self, effects: tuple, vagal: float = 0.0) -> None:
        total_eff_tpr, total_eff_sv, total_eff_hr, eff_remi_tpr, eff_remi_sv, eff_remi_hr = effects
        if eff_remi_hr < 0.0:
            eff_remi_hr *= 1.0 - vagal  # Opioid bradycardia is vagal.
        self._drug_tpr = (1.0 + total_eff_tpr) / max(0.05, 1.0 - eff_remi_tpr)
        self._drug_sv = (1.0 + total_eff_sv) / max(0.05, 1.0 - eff_remi_sv)
        cfg = self.config
        remi_vagal = cfg.remi_vagal_hr * self.ce_remi_vagal / (self.ce_remi_vagal + cfg.remi_vagal_c50)
        self._drug_hr = (1.0 + total_eff_hr) / max(0.05, 1.0 - eff_remi_hr) * (1.0 - remi_vagal * (1.0 - vagal))

    def _calc_hr(self):
        return self.hr_star + self.tde_hr + self.smoothed_chemo_hr + self.smoothed_baro_hr + self.smoothed_epi_hr

    def _hr_sv_factor(self, hr: float) -> float:
        # Su Eq. 9 also increases filling when HR falls below baseline. Heart
        # failure, especially HFpEF, loses more filling during tachycardia.
        factor = max(0.1, 1.0 - self.config.hr_sv_coupling * math.log(max(1.0, hr) / self.base_hr))
        return factor / (1.0 + self.failure.tachycardia_gain * max(0.0, hr / self.base_hr - 1.0))

    def step(self, dt: float, cp_prop: float, cp_remi: float, ce_nore: float, pit: float, paco2: float, pao2: float,
             dist_hr: float = 0.0, dist_sv: float = 0.0, dist_svr: float = 0.0,
             mac_sevo: float = 0.0, ce_epi: float = 0.0, ce_phenyl: float = 0.0,
             ce_vaso: float = 0.0, ce_dobu: float = 0.0, ce_mil: float = 0.0,
             ce_esmolol: float = 0.0, ce_labetalol: float = 0.0, ce_glyco: float = 0.0,
             ce_ketamine: float = 0.0, ce_ketamine_hr: float = 0.0,
             ce_atropine: float = 0.0, ce_ephedrine: float = 0.0,
             temp_c: float = 37.0, peep_cmH2O: float | None = None, sao2: float = 98.0,
             remifentanil_cp: float | None = None) -> HemoState:
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
            ce_esmolol: Esmolol effect-site concentration (mcg/mL).
            ce_labetalol, ce_glyco, ce_atropine, ce_ephedrine: Effect-site concentrations (ng/mL).
            ce_ketamine, ce_ketamine_hr: Ketamine cardiac-output and HR effect sites (mcg/mL).
            temp_c: Core temperature (°C).
            peep_cmH2O: Total PEEP (cmH2O).
            sao2: Arterial saturation (%).
            remifentanil_cp: Remifentanil alone (ng/mL), for its vagal
                bradycardia; defaults to cp_remi.
        """
        cfg = self.config
        self._cached_state = None
        dt_min = dt / 60.0
        sepsis_sev = clamp01(self.sepsis_severity)
        anaph_sev = clamp01(self.anaphylaxis_severity)

        # Hemodynamic effect sites: propofol's arterial, cardiac, and venous
        # effects follow Kazama 1999's SBP equilibration. Reflex depression
        # follows plasma propofol, as consciousness does within a minute.
        self.ce_prop_cv += (max(0.0, cp_prop) - self.ce_prop_cv) * -math.expm1(-self._ke0_prop_cv * dt_min)
        self.ce_opioid_cv += (max(0.0, cp_remi) - self.ce_opioid_cv) * -math.expm1(-self._ke0_opioid_cv * dt_min)
        remifentanil = max(0.0, cp_remi if remifentanil_cp is None else remifentanil_cp)
        self.ce_remi_vagal += (remifentanil - self.ce_remi_vagal) * -math.expm1(-self._ke0_opioid_cv * dt_min)
        self.ce_sevo += (mac_sevo - self.ce_sevo) * -math.expm1(-cfg.ke0_sevo * dt_min)
        ce_prop = self.ce_prop_cv
        anesthetic_depth = clamp01(self.ce_sevo + cp_prop / 4.0)
        self._sympathetic_venous_depth += (
            clamp01(max(0.0, cp_prop) / 3.0) - self._sympathetic_venous_depth
        ) * -math.expm1(-math.log(2.0) / cfg.propofol_sympathetic_half_time * dt_min)

        # Antagonist dose ratios; esmolol is beta-1 selective.
        labetalol_beta = max(0.0, ce_labetalol) / cfg.labetalol_kb_beta
        beta1_dr = 1.0 + max(0.0, ce_esmolol) / cfg.esmolol_kb + labetalol_beta
        beta2_dr = 1.0 + labetalol_beta
        alpha_dr = 1.0 + max(0.0, ce_labetalol) / cfg.labetalol_kb_alpha
        self._beta_occupancy = beta_occ = 1.0 - 1.0 / beta1_dr
        alpha_occ = 1.0 - 1.0 / alpha_dr
        glyco = hill_function(max(0.0, ce_glyco), cfg.glyco_c50, cfg.glyco_gamma)
        atropine = hill_function(max(0.0, ce_atropine), cfg.atropine_c50, cfg.atropine_gamma)
        self._vagal_block = vagal = 1.0 - (1.0 - glyco) * (1.0 - atropine)
        atropine_hr = cfg.atropine_emax_hr * atropine / (1.0 + max(0.0, cp_prop) / cfg.atropine_propofol_cp50)
        vagal_hr = cfg.glyco_emax_hr * glyco * (1.0 - atropine) + atropine_hr
        resting_tone = 1.0 - cfg.sympathetic_anesthetic_depression * anesthetic_depth
        # Shares of sympathetic reflex and stimulation responses left after
        # block; age reduces the HR share only, as for beta agonists.
        beta_response = 1.0 - cfg.beta_block_reflex * beta_occ
        sympathetic_hr_response = beta_response * self._age_chronotropy
        alpha_response = 1.0 - alpha_occ
        dist_hr *= sympathetic_hr_response
        dist_sv *= beta_response
        # Septic shock removes stimulation-driven vasoconstriction, which depends
        # partly on the vasopressin that septic shock depletes.
        dist_svr *= alpha_response * (1.0 - sepsis_sev)
        self._venous_tone = alpha_response
        # Severe anaphylaxis blunts adrenergic agonists; vasopressin acts on V1 receptors.
        agonist_shift = 1.0 + cfg.anaphylaxis_agonist_shift * anaph_sev
        ce_epi /= agonist_shift
        ce_nore /= agonist_shift
        ce_phenyl /= agonist_shift
        ce_dobu /= agonist_shift

        # Urine output, scaled by renal perfusion and function.
        map_prev = self._prev_map
        map_denom = max(1e-3, cfg.renal_map_norm - cfg.renal_map_min)
        self.renal_perfusion = clamp01((map_prev - cfg.renal_map_min) / map_denom)
        renal_factor = self.renal_perfusion * self.patient.renal_clearance_factor
        fluid_change, fluid_urine, fluid_sequestration = self.crystalloid.step(
            dt, map_prev, self.patient.age, self.patient.renal_clearance_factor * cfg.crystalloid_renal_scale,
            self.blood_volume - max(BLOOD_VOLUME_MIN, self.red_cell_volume_ml),
            urinary_elimination=cfg.vol_clearance is None,
        )
        self.blood_volume += fluid_change
        self.total_urine_out_ml += fluid_urine
        self.total_third_space_ml += fluid_sequestration
        if cfg.vol_clearance is not None:
            urine_ml_min = max(0.0, float(cfg.vol_clearance))
        else:
            urine_ml_min = cfg.uop_ml_kg_hr * self.patient.weight / 60.0
        urine_out_ml = min(
            urine_ml_min * renal_factor * dt_min,
            max(0.0, self.blood_volume - max(BLOOD_VOLUME_MIN, self.red_cell_volume_ml)),
        )
        blood_volume = self.blood_volume - urine_out_ml
        if urine_out_ml > 0:
            plasma_volume = self.blood_volume - self.red_cell_volume_ml
            self.crystalloid.central_ml *= max(0.0, 1.0 - urine_out_ml / plasma_volume)
            self.total_urine_out_ml += urine_out_ml
        if sepsis_sev > 0.0:
            # Capillary leak loses plasma, not red cells.
            dt_hr = dt / 3600.0
            leak_fraction = cfg.sepsis_leak_fraction_per_hr * sepsis_sev
            leak_ml = blood_volume * leak_fraction * dt_hr
            available = max(0.0, blood_volume - max(BLOOD_VOLUME_MIN, self.red_cell_volume_ml))
            actual_leak = min(leak_ml, available)
            if actual_leak > 0:
                self.crystalloid.central_ml *= max(0.0, 1.0 - actual_leak / (blood_volume - self.red_cell_volume_ml))
                blood_volume -= actual_leak
                self.total_third_space_ml += actual_leak
        if self.total_third_space_ml > 0 and cfg.third_space_refill_tau_hr > 0:
            intravascular_excess = blood_volume - self.blood_volume_0
            shortfall = cfg.interstitial_refill_fraction * (
                intravascular_excess + self.total_third_space_ml
            ) - intravascular_excess
            frac = -math.expm1(-dt / (cfg.third_space_refill_tau_hr * 3600.0))
            refill_ml = min(self.total_third_space_ml, shortfall * frac)
            if refill_ml > 0:
                self.total_third_space_ml -= refill_ml
                blood_volume += refill_ml

        # Lung water filters from blood above the LAP threshold and returns over hours.
        lap_excess = max(0.0, self._last_lap - self._lung_water_threshold)
        filtration_ml = min(
            cfg.lung_water_filtration * self._pbw * lap_excess * dt_min,
            max(0.0, blood_volume - max(BLOOD_VOLUME_MIN, self.red_cell_volume_ml)),
        )
        lung_clearance_ml = self.lung_water_ml * -math.expm1(-dt / (cfg.lung_water_clearance_tau_hr * 3600.0))
        self.lung_water_ml += filtration_ml - lung_clearance_ml
        if filtration_ml > 0.0:
            self.crystalloid.central_ml *= 1.0 - filtration_ml / (blood_volume - self.red_cell_volume_ml)
        blood_volume += lung_clearance_ml - filtration_ml
        self.blood_volume = max(BLOOD_VOLUME_MIN, blood_volume)
        self._update_hb_conc()

        self._set_drug_factors(self._calc_anesthetic_effects(ce_prop, self.ce_opioid_cv, self.ce_sevo), vagal)

        # Positive intrathoracic pressure reduces venous return; spontaneous
        # negative pressure modestly augments it.
        delta_pit = pit - cfg.pit_0
        if delta_pit >= 0.0:
            f_preload_pit = 1.0 / (1.0 + cfg.alpha_peep * delta_pit)
        else:
            f_preload_pit = 1.0 + cfg.alpha_peep * (-delta_pit)
        self.f_preload_pit = clamp(f_preload_pit, 0.4, 1.4)

        self._epi_pressor_ce += (max(0.0, ce_epi) - self._epi_pressor_ce) * (
            -math.expm1(-dt / cfg.epi_tau_pressor_s)
        )
        self._epi_venous_alpha = hill_function(self._epi_pressor_ce / alpha_dr, cfg.epi_c50_alpha, cfg.epi_gamma_alpha)
        pressor_gain = 1.0 - clamp01(cfg.sepsis_pressor_resistance * sepsis_sev)
        self._update_venous_drug_effects(ce_prop, ce_phenyl / alpha_dr, pressor_gain)

        f_frank_starling = self._update_pulmonary_coupling(
            dt=dt,
            pao2=pao2,
            peep_cmH2O=peep_cmH2O,
            f_preload_pit=self.f_preload_pit,
            sepsis_sev=sepsis_sev,
        )

        # Chemoreflex: hypercapnia raises HR and TPR; hypoxemia raises HR.
        e_co2 = max(0.0, (paco2 - self._paco2_set) / self._paco2_set)
        e_o2 = max(0.0, (cfg.pao2_set - pao2) / cfg.pao2_set)
        chemo_hr_boost = (cfg.g_hr_co2 * e_co2 + cfg.g_hr_o2 * e_o2) * sympathetic_hr_response
        chemo_tpr_factor = 1.0 + cfg.k_tpr_co2 * e_co2

        # Fast baroreflex. Propofol depresses both limbs (Sato 2005). Anesthetic
        # ketamine depresses it as much (Akine 2001), but analgesic doses do not
        # (Huang 2020); it counts in MAC equivalents for this reflex only.
        # Opioids raise vagal tone, suppressing reflex tachycardia but not slowing.
        reflex_depth = clamp01(anesthetic_depth + max(0.0, ce_ketamine) / KETAMINE_MAC_EQUIVALENT)
        tachycardia_depth = clamp01(reflex_depth + self.ce_opioid_cv / cfg.opioid_reflex_ce)
        sensed_error = map_prev - self._stim_map - self._baro_setpoint
        if sensed_error > 0.0:
            baro_gain = cfg.baro_gain_brady * (1.0 - vagal) * self._age_vagal_baroreflex
            baro_gain *= 1.0 - cfg.baro_anesthetic_depression * reflex_depth
        else:
            baro_gain = cfg.baro_gain_tachy * sympathetic_hr_response
            baro_gain *= 1.0 - cfg.baro_anesthetic_depression * tachycardia_depth
        baro_hr = clamp(-baro_gain * sensed_error, -cfg.baro_max_hr_change, cfg.baro_max_hr_change)
        self._baro_setpoint += sensed_error * min(1.0, dt / cfg.baro_reset_tau_s)

        # Myocardial hypoxia; recovery after reoxygenation needs coronary perfusion.
        hypoxia_target = clamp01(
            (cfg.hypoxia_sao2_onset - sao2) / (cfg.hypoxia_sao2_onset - cfg.hypoxia_sao2_full)
        )
        if hypoxia_target > self.myocardial_hypoxia:
            hypoxia_tau = cfg.hypoxia_tau_on_s
        else:
            hypoxia_tau = cfg.hypoxia_tau_off_s / clamp(map_prev / self.patient.baseline_map, 0.1, 1.0)
        self.myocardial_hypoxia += (hypoxia_target - self.myocardial_hypoxia) * min(1.0, dt / hypoxia_tau)

        hemorrhage_hr_mult, hemorrhage_tpr_mult = self._calc_hemorrhage_response()
        # Anesthesia already depresses the fast and slow reflex tachycardia, so
        # the volume-loss drive keeps full strength; HR still rises under TIVA (Kurita 2011).
        hemorrhage_hr_mult = 1.0 + (hemorrhage_hr_mult - 1.0) * sympathetic_hr_response
        hemorrhage_tpr_mult = 1.0 + (hemorrhage_tpr_mult - 1.0) * alpha_response

        # Vascular relaxation develops alongside the cardiac response. An
        # instantaneous beta-2 effect otherwise produces a spurious early dip.
        beta2_target = cfg.epi_emax_svr_beta * hill_function(ce_epi / beta2_dr, cfg.epi_c50_beta2, 1.0)
        self._epi_beta2_effect += (beta2_target - self._epi_beta2_effect) * (
            -math.expm1(-dt / cfg.epi_tau_beta2_s)
        )
        epi_delta_hr, epi_sv_factor, epi_svr_factor = self._calc_epi_effects(
            ce_epi, self._epi_pressor_ce, beta1_dr, alpha_dr
        )
        self._epi_chrono_effect += (epi_delta_hr - self._epi_chrono_effect) * (
            -math.expm1(-dt / cfg.epi_tau_hr_s)
        )
        nore_delta_hr, nore_sv_factor, nore_svr_factor = self._calc_nore_effects(
            ce_nore, beta1_dr, alpha_dr, anesthetic_depth,
        )
        self._ephedrine_tolerance_ce += (max(0.0, ce_ephedrine) - self._ephedrine_tolerance_ce) * (
            -math.expm1(-math.log(2.0) * dt_min / cfg.ephedrine_tolerance_half_min)
        )
        ephedrine = max(0.0, ce_ephedrine) / agonist_shift
        ephedrine_effect = max(0.0,
            hill_function(ephedrine, cfg.ephedrine_c50, cfg.ephedrine_gamma)
            - cfg.ephedrine_tolerance_max * hill_function(
                self._ephedrine_tolerance_ce / agonist_shift,
                cfg.ephedrine_tolerance_c50, cfg.ephedrine_tolerance_gamma,
            )
        )
        ephedrine_svr = 1.0 + cfg.ephedrine_emax_svr * ephedrine_effect / alpha_dr
        ephedrine_sv = 1.0 + cfg.ephedrine_emax_sv * ephedrine_effect / beta1_dr
        ephedrine_hr = (
            cfg.ephedrine_hr_slope * ephedrine + cfg.ephedrine_emax_hr * ephedrine_effect
        ) * self.base_hr * sympathetic_hr_response
        phenyl_svr_factor = self._calc_phenyl_effects(ce_phenyl / alpha_dr)
        vaso_delta_hr, _, vaso_svr_factor = self._calc_hr_sv_svr_effects(
            ce_vaso, cfg.vaso_c50, cfg.vaso_gamma, cfg.vaso_emax_hr, 0.0, cfg.vaso_emax_svr
        )
        dobu_delta_hr, dobu_sv_factor, _ = self._calc_hr_sv_svr_effects(
            ce_dobu / beta1_dr, cfg.dobu_c50, cfg.dobu_gamma, cfg.dobu_emax_hr * self._age_chronotropy,
            cfg.dobu_emax_sv, 0.0,
        )
        _, _, dobu_svr_factor = self._calc_hr_sv_svr_effects(
            ce_dobu / beta2_dr, cfg.dobu_c50, cfg.dobu_gamma, 0.0, 0.0, cfg.dobu_emax_svr
        )
        mil_delta_hr, mil_sv_factor, mil_svr_factor = self._calc_hr_sv_svr_effects(
            ce_mil, cfg.mil_c50, cfg.mil_gamma, cfg.mil_emax_hr, cfg.mil_emax_sv, cfg.mil_emax_svr
        )

        # SVR and SV factors multiply; chronotropy adds. Sepsis blunts catecholamines.
        catechol_svr_factor = epi_svr_factor * nore_svr_factor * phenyl_svr_factor * ephedrine_svr
        pressor_resistance = clamp01(cfg.sepsis_pressor_resistance * sepsis_sev)
        if pressor_resistance > 0:
            catechol_svr_factor = 1.0 + (catechol_svr_factor - 1.0) * (1.0 - pressor_resistance)

        # Blockade removes resting sympathetic tone; vagal block removes vagal
        # tone; ketamine adds sympathetic tone.
        ketamine = hill_function(max(0.0, ce_ketamine), cfg.ketamine_c50, cfg.ketamine_gamma) * resting_tone
        ketamine_hr = hill_function(max(0.0, ce_ketamine_hr), cfg.ketamine_c50, cfg.ketamine_gamma) * resting_tone
        combined_svr_factor = (
            catechol_svr_factor *
            vaso_svr_factor *
            dobu_svr_factor *
            mil_svr_factor *
            (1.0 - cfg.alpha_block_tpr * alpha_occ * resting_tone)
        )
        combined_sv_factor = (
            epi_sv_factor * nore_sv_factor * dobu_sv_factor * mil_sv_factor * ephedrine_sv
            * (1.0 - cfg.beta_block_sv * beta_occ * resting_tone)
            * (1.0 + cfg.ketamine_emax_sv * ketamine * beta_response)
            / (1.0 + self._phenyl_afterload_gain * (phenyl_svr_factor - 1.0) * pressor_gain)
        )
        combined_delta_hr = (
            self._epi_chrono_effect + nore_delta_hr + vaso_delta_hr + dobu_delta_hr + mil_delta_hr + ephedrine_hr
            + (cfg.ketamine_emax_hr * ketamine_hr + cfg.sepsis_hr_increase * sepsis_sev) * sympathetic_hr_response
            - cfg.beta_block_hr * beta_occ * resting_tone * self.hr_star
            + vagal_hr
        )
        self.vasopressor_sv_factor = combined_sv_factor
        self.delta_tpr_vasopressors = self.base_tpr * (combined_svr_factor - 1.0)

        # Feedback senses the same circulation as the outputs, including
        # rhythm, hypoxia, and preload. Nociception resets it as it does the
        # fast baroreflex, so sustained surgery keeps MAP raised. Without
        # output it holds its last value instead of winding up during arrest.
        self.dist_hr, self.dist_sv, self.dist_svr = dist_hr, dist_sv, dist_svr
        distributive_svr_drop = (cfg.sepsis_svr_drop_wood * sepsis_sev +
                                 cfg.anaphylaxis_svr_drop_wood * anaph_sev)
        distributive_tpr_offset = -distributive_svr_drop / 1000.0
        feedback_state = self._compute_state(
            preload_sv_factor=f_frank_starling,
            sepsis_sev=sepsis_sev,
            anaph_sev=anaph_sev,
            distributive_tpr_offset=distributive_tpr_offset,
        )
        if feedback_state.map > 0.0:
            rmap = clamp((feedback_state.map - self._stim_map) * self._inv_base_rmap_denom, 0.1, 5.0)
            self._rmap_fb = rmap ** cfg.fb
        rmap_fb = self._rmap_fb

        # Thermoregulatory vasoconstriction below 36.5 °C. Anesthesia lowers the
        # threshold by up to 2.5 °C at 1 MAC or propofol 4 mcg/mL (Sessler 2000).
        vasoconstriction_threshold = 36.5 - 2.5 * anesthetic_depth

        thermo_tpr_mult = 1.0
        if temp_c < vasoconstriction_threshold:
            thermo_tpr_mult = 1.0 + TEMP_TPR_COEFFICIENT * (vasoconstriction_threshold - temp_c)
        thermo_tpr_mult = min(2.0, thermo_tpr_mult)

        # Turnover equations; inotropy and preload act on output SV only.
        tpr_production = self.kin_tpr * rmap_fb * chemo_tpr_factor * hemorrhage_tpr_mult * thermo_tpr_mult
        d_tpr_tone = tpr_production - cfg.kout * self.tpr_tone
        d_sv_tone = self.kin_sv * rmap_fb - cfg.kout * self.sv_tone
        hr_feedback = self._hr_feedback(rmap_fb, tachycardia_depth)
        d_hr_tone = self.kin_hr * hr_feedback * hemorrhage_hr_mult - cfg.kout * self.hr_tone

        alpha_fast = min(1.0, dt / cfg.tau_hr_fast)
        self.smoothed_chemo_hr += (chemo_hr_boost - self.smoothed_chemo_hr) * alpha_fast
        self.smoothed_baro_hr += (baro_hr - self.smoothed_baro_hr) * alpha_fast
        self.smoothed_epi_hr += (combined_delta_hr - self.smoothed_epi_hr) * alpha_fast

        self.tpr_tone += d_tpr_tone * dt_min
        self.sv_tone += d_sv_tone * dt_min
        self.hr_tone += d_hr_tone * dt_min
        self.tde_hr -= cfg.k_drift * self.tde_hr * dt_min
        self.tde_sv -= cfg.k_drift * self.tde_sv * dt_min

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

    def initialize_steady_state(self, cp_prop: float, cp_remi: float, ce_nore: float, mac_sevo: float = 0.0,
                                remifentanil_cp: float | None = None) -> HemoState:
        """Seed tone, drug effects, and the baroreflex at fixed concentrations."""
        cn = max(0.0, ce_nore)
        self.ce_remi_vagal = max(0.0, cp_remi if remifentanil_cp is None else remifentanil_cp)
        self._sympathetic_venous_depth = clamp01(max(0.0, cp_prop) / 3.0)
        self._epi_venous_alpha = 0.0
        self._set_drug_factors(self._calc_anesthetic_effects(cp_prop, cp_remi, mac_sevo))
        anesthetic_depth = clamp01(mac_sevo + cp_prop / 4.0)
        tachycardia_depth = clamp01(anesthetic_depth + max(0.0, cp_remi) / self.config.opioid_reflex_ce)
        nore_delta_hr, nore_sv_factor, nore_svr_factor = self._calc_nore_effects(cn, anesthetic_depth=anesthetic_depth)
        # Venous return at the drug's stressed volume, relative to the reference
        # that keeps Su's response, sets the steady-state preload.
        self._update_venous_drug_effects(cp_prop, 0.0)

        def flow_factor(mcfp: float) -> float:
            rap = self.config.rap_baseline * min(1.0, mcfp / self.mcfp_0)
            return (mcfp - rap) / self.venous_return_resistance * self._inv_base_co_l_min

        self._pulm_reference_factor = flow_factor(self._venous_reference_mcfp)
        self._pulm_flow_factor = flow_factor(self._calc_stressed_volume() / self.cv)
        preload_sv = self._frank_starling(self._pulm_flow_factor / self._pulm_reference_factor * self._position_preload)

        def residual(z):
            z_fb = z ** self.config.fb
            hr_z = self.base_hr * self._hr_feedback(z_fb, tachycardia_depth) * self._drug_hr + nore_delta_hr
            sv_z = self.base_sv * z_fb * self._drug_sv * self._hr_sv_factor(hr_z) * nore_sv_factor * preload_sv
            tpr_z = self.base_tpr * (z_fb * self._drug_tpr + nore_svr_factor - 1.0)
            return hr_z * sv_z * tpr_z * self._inv_base_rmap_denom - z

        try:
            sol = root_scalar(residual, bracket=[0.01, 5.0], method='brentq')
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError("Unable to solve hemodynamic steady state") from exc
        if not sol.converged:
            raise RuntimeError("Hemodynamic steady-state solver did not converge")
        z_ss = sol.root

        z_fb = z_ss ** self.config.fb
        self.hr_tone = self.base_hr * self._hr_feedback(z_fb, tachycardia_depth)
        self.sv_tone = self.base_sv * z_fb
        self.tpr_tone = self.base_tpr * z_fb
        self.ce_prop_cv, self.ce_opioid_cv = max(0.0, cp_prop), max(0.0, cp_remi)
        self.tde_hr = 0
        self.tde_sv = 0
        self.ce_sevo = mac_sevo

        self._epi_pressor_ce = self._epi_chrono_effect = self._epi_beta2_effect = 0.0
        self.smoothed_chemo_hr = self.smoothed_baro_hr = 0.0
        self.smoothed_epi_hr = nore_delta_hr
        self.vasopressor_sv_factor = nore_sv_factor
        self.delta_tpr_vasopressors = self.base_tpr * (nore_svr_factor - 1.0)

        state = self.step(0.0, cp_prop, cp_remi, ce_nore, -2.0, 40.0, 95.0, 0, 0, 0, mac_sevo=mac_sevo,
                          remifentanil_cp=remifentanil_cp)
        self._prev_map = self._baro_setpoint = state.map
        return state
