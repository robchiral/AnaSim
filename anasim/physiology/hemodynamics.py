import math
from dataclasses import replace
from itertools import pairwise

from scipy.optimize import root_scalar

from anasim.core.constants import (
    ARREST_CO_L_MIN,
    BLOOD_VOLUME_MIN,
    CIRCULATION_LIMIT_FRACTION,
    HR_MAX,
    HR_MIN,
    TEMP_TPR_COEFFICIENT,
)
from anasim.core.enums import RhythmType
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.patient.conditions import HEART_FAILURE_PROFILES
from anasim.patient.domain import bounded_number, finite_number
from anasim.patient.patient import Patient
from anasim.patient.pd.anesthesia import KETAMINE_MAC_EQUIVALENT
from anasim.patient.pk_models import NorepinephrinePK
from anasim.patient.variability import HemodynamicVariability

from .coronary import CoronaryModel
from .fluid_balance import FluidBalance
from .fluids import CrystalloidKinetics
from .hemo_config import HemodynamicConfig
from .hemo_types import HemoState
from .resuscitation import (
    COMPRESSION_AFTERLOAD_COUPLING,
    COMPRESSION_DUTY,
    COMPRESSION_OUTPUT_FRACTION,
    compression_pressures,
)
from .toxicity import SILENT_BLOCK

MAX_HEMATOCRIT = 0.7
_REFERENCE_BLOOD_VOLUME_ML = Patient().estimate_blood_volume()

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
    RhythmType.AFIB: 0.74,  # Lost atrial kick and R-R irregularity (~15% CO loss at controlled rate; Clark 1997)
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
        cfg = config or HemodynamicConfig()
        eta = patient.variability.hemodynamics
        if eta != HemodynamicVariability():
            cfg = replace(
                cfg,
                ec50_prop_tpr=cfg.ec50_prop_tpr * eta.propofol_tpr_c50,
                propofol_sympathetic_cp=cfg.propofol_sympathetic_cp * eta.propofol_tpr_c50,
                emax_remi_tpr=cfg.emax_remi_tpr * eta.opioid_tpr_emax,
                sevo_ec50_tpr=cfg.sevo_ec50_tpr * eta.sevoflurane_tpr_c50,
            )
        self.config = cfg
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
        # Resting endogenous norepinephrine (Li 2024 Eq. 2) is already embodied in
        # baseline MAP/HR and resting sympathetic tone; exogenous increments act above it.
        self._basal_nore_ce = NorepinephrinePK.resting_concentration(patient.age)

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
        self.crisis_hr = 0.0
        self.crisis_svr = 0.0
        self._sepsis_severity = 0.0
        self._anaphylaxis_severity = 0.0
        # Adrenoceptor occupancy and muscarinic block, 0-1.
        self._beta_occupancy = 0.0
        self._vagal_block = 0.0
        self._ephedrine_tolerance_ce = 0.0

        # Fast HR effects (chemoreflex, baroreflex, vasoactive chronotropy).
        self.smoothed_epi_hr = 0.0
        self._epi_pressor_ce = 0.0
        self._epi_high_pressor_ce = 0.0
        self._epi_chrono_effect = 0.0
        self._adrenergic_chrono = 0.0
        self._epi_pulse_factor = 1.0
        self._epi_beta2_effect = 0.0
        self.smoothed_chemo_hr = 0.0
        self.smoothed_baro_hr = 0.0
        self.smoothed_baro_vasc = 0.0
        self._unstim_rate = 0.0
        self._unstim_rate_unloaded = 0.0
        self._baro_setpoint = patient.baseline_map
        self._stim_map = 0.0  # MAP attributable to noxious stimulation
        self._rmap_fb = 1.0  # Turnover feedback, (MAP / baseline) ** fb
        self.myocardial_hypoxia = 0.0  # 0-1 depression of rate and contractility
        self.acidosis_factor = 1.0

        # Slow tone: baselines adjusted by MAP feedback, chemoreflex, blood
        # loss, and temperature. Anesthetic effects multiply it from their own
        # effect sites, so drug onset and homeostatic recovery keep separate rates.
        self.sv_tone = self.base_sv
        self.hr_tone = self.base_hr
        self.tpr_tone = self.base_tpr
        self._drug_tpr = self._drug_sv = self._drug_hr = 1.0
        self._amiodarone_ees_factor = 1.0
        self._amiodarone_vasc_adapted = 0.0
        self.ce_prop_cv = self.ce_opioid_cv = self.ce_remi_vagal = 0.0
        self._sympathetic_venous_depth = 0.0
        self._epi_venous_alpha = 0.0  # Share of propofol's venous pooling reversed by epinephrine
        self._ke0_prop_cv = math.log(2.0) / _interpolate(patient.age, cfg.prop_cv_half_time_by_age)
        self._prop_cv_potency = _interpolate(patient.age, cfg.prop_cv_potency_by_age)
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
        self.cv = cfg.venous_compliance * (self.blood_volume_0 / _REFERENCE_BLOOD_VOLUME_ML)  # mL/mmHg
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
        self._rv_pit_preload = 1.0
        self._lv_pit_preload = 1.0
        self._pulmonary_congestion_ml = 0.0
        self._pulm_arrest_blend = 0.0
        self._last_mcfp = self.mcfp_0
        self._last_rap = cfg.rap_baseline
        self._last_pvr = cfg.pvr_wood_baseline
        self._last_rv_co = self.base_co_l_min
        self._last_lv_inflow = self.base_co_l_min
        self._last_intrinsic_co = self.base_co_l_min
        self._last_venous_preload = self._position_preload
        self._last_preload_factor = self._position_preload
        self._last_preload_sv_factor = self._frank_starling(self._position_preload)
        self._last_lap = self._lap(self._position_preload)
        resting_lap = self._lap(1.0)
        self._resting_lap = resting_lap
        # Chronic left atrial hypertension raises lymph flow, partially offsetting
        # elevated resting LAP while leaving less reserve before acute edema.
        normal_lap = cfg.lap_scale * math.expm1(cfg.lap_preload_gain)
        self._lung_water_threshold = cfg.lung_water_lap_threshold + 0.62 * max(0.0, resting_lap - normal_lap)
        self.coronary = CoronaryModel(
            patient.coronary_flow_reserve,
            self.base_hr,
            patient.baseline_map,
            resting_lap,
            baseline_hb=self.baseline_hb,
        )
        self._pbw = patient.predicted_body_weight()
        self.lung_water_ml = 0.0  # Extravascular lung water above normal

        self.total_crystalloid_in_ml = 0.0
        self.total_colloid_in_ml = 0.0
        self.total_blood_in_ml = 0.0
        self.total_urine_out_ml = 0.0
        self.total_blood_out_ml = 0.0
        self.total_third_space_ml = 0.0
        self._protein_third_space_ml = 0.0
        self.crystalloid = CrystalloidKinetics()
        self.fluids = FluidBalance(patient.weight, self.blood_volume, self.red_cell_volume_ml,
                                   patient.pregnancy.ph, patient.pregnancy.extracellular_water)

        self._rhythm_type = RhythmType.SINUS
        self.sodium_block = 0.0
        self.toxic_contractility_loss = 0.0
        self.hyperkalemia_block = 0.0
        self.hyperkalemia_arrhythmia = 0.0
        self._ca_sv_factor = 1.0
        self._ca_tpr_factor = 1.0
        self._hypothermia_hr_factor, self._hypothermia_sv_factor = self._hypothermia_factors(
            patient.baseline_temp
        )
        self._asystole_recovery_s = 0.0
        self.post_shock_asystole = False
        self._compressions = False
        self.f_preload_pit = 1.0
        self.renal_perfusion = 1.0  # Share of normal renal blood flow, from MAP
        self.vasopressor_sv_factor = 1.0
        self._phenyl_svr_factor = 1.0
        self._non_phenyl_sv_factor = 1.0
        self._prev_map = self.state.map

    def calibrate_coronary(self, sbp_mmhg: float, dbp_mmhg: float) -> None:
        """Calibrate resting coronary reserve against steady-state arterial SBP and DBP."""
        self.coronary = CoronaryModel(
            self.patient.coronary_flow_reserve,
            self.base_hr,
            self.patient.baseline_map,
            self._resting_lap,
            sbp_mmhg=sbp_mmhg,
            dbp_mmhg=dbp_mmhg,
            baseline_hb=self.baseline_hb,
        )

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
    def compressions(self) -> bool:
        """Whether chest compressions are in progress."""
        return self._compressions

    @compressions.setter
    def compressions(self, value: bool) -> None:
        if value != self._compressions:
            self._compressions = value
            self.invalidate_state_cache()

    @property
    def rhythm_type(self) -> RhythmType:
        return self._rhythm_type

    @rhythm_type.setter
    def rhythm_type(self, value: RhythmType) -> None:
        if value != self._rhythm_type:
            self._rhythm_type = value
            self._asystole_recovery_s = 0.0
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
        amount_ml = finite_number("amount_ml", amount_ml)
        hematocrit = bounded_number("hematocrit", hematocrit, 0.0, MAX_HEMATOCRIT)
        self._cached_state = None
        if amount_ml == 0:
            return

        gain = 0.02 * (self.base_sv / 80.0)
        retained_ml = 0.0
        if amount_ml < 0:
            loss_ml = -amount_ml
            available = max(0.0, self.blood_volume - BLOOD_VOLUME_MIN)
            actual_loss = min(loss_ml, available)
            if actual_loss > 0:
                self.fluids.bleed(actual_loss, self.red_cell_volume_ml / self.blood_volume)
                self.crystalloid.central_ml *= 1.0 - actual_loss / self.blood_volume
                self.red_cell_volume_ml *= 1.0 - actual_loss / self.blood_volume
                self.blood_volume -= actual_loss
                self.total_blood_out_ml += actual_loss
                hb_loss = self.hb_conc * actual_loss / 100.0
                self.hb_mass = max(0.0, self.hb_mass - hb_loss)
                self.tde_sv = max(0.0, self.tde_sv - actual_loss * gain)
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
                is_colloid = label in ("colloid", "lipid")
                if is_colloid:
                    self.total_colloid_in_ml += amount_ml
                    retention = 0.25 if label == "lipid" else cfg.colloid_retention_fraction
                else:
                    self.total_crystalloid_in_ml += amount_ml
                    retention = 1.0
                if retention_fraction is not None:
                    retention = retention_fraction
                retention = clamp01(retention)
                retained_ml = amount_ml * retention
                if label == "lipid":
                    total_water_l = self.fluids.ecf_l + self.fluids.icf_l
                    ecf_share = self.fluids.ecf_l / total_water_l if total_water_l > 0.0 else 1.0 / 3.0
                    ecf_input_ml = amount_ml * ecf_share
                else:
                    ecf_input_ml = amount_ml
                third_space_gain_ml = max(0.0, ecf_input_ml - retained_ml)
                self.total_third_space_ml += third_space_gain_ml
                if label == "colloid":
                    self._protein_third_space_ml += third_space_gain_ml
                if not is_colloid:
                    self.crystalloid.infuse(retained_ml, saline=label == "saline")

            self.blood_volume += retained_ml
            self.fluids.infuse(amount_ml, retained_ml, hematocrit, label)

            if retained_ml > 0 and self.patient.heart_failure != "hfref":
                reserve = 0.35 if self.patient.heart_failure == "hfpef" else 1.0
                self.tde_sv += retained_ml * gain * reserve

        self.blood_volume = max(BLOOD_VOLUME_MIN, self.blood_volume)
        self._update_hb_conc()

    def _update_hb_conc(self):
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
        early_deficit = min(deficit, 0.3 * self.blood_volume_0)
        recruited = (cfg.venous_recruitment_fraction * early_deficit
                     + cfg.severe_venous_recruitment_fraction * (deficit - early_deficit)) * self._venous_tone
        available = max(0.0, self.blood_volume - self.unstressed_volume + recruited - pooling)
        available -= (cfg.propofol_sympathetic_pooling * self._sympathetic_gain * self._sympathetic_venous_depth
                      * (1.0 - self._epi_venous_alpha) * min(available, self._baseline_stressed_volume))
        return min(self.blood_volume, available * self._venous_stressed_fraction + self._pressor_recruitment_ml)

    def _update_venous_drug_effects(
        self,
        ce_prop: float,
        ce_phenyl: float,
        pressor_gain: float = 1.0,
        ce_nore: float = 0.0,
        *,
        nore_scale: float = 1.0,
    ) -> None:
        """Venous drug effects, normalized to preserve Su's normal-volume response."""
        cfg = self.config
        propofol = clamp(ce_prop, 0.0, 6.5)
        stressed_fraction = math.exp(-cfg.propofol_venous_pooling_slope * propofol)
        self._venous_stressed_fraction = stressed_fraction
        self._venous_reference_mcfp = self._baseline_stressed_volume * stressed_fraction / self.cv
        self.venous_return_resistance = self._baseline_venous_resistance * math.exp(
            -cfg.propofol_venous_resistance_slope * propofol
        )
        # Only exogenous norepinephrine above the propofol-shifted baseline recruits veins.
        eff_nore = max(0.0, ce_nore - 1.5 * self._basal_nore_ce) * nore_scale
        nore_recruitment = cfg.nore_venous_recruitment_fraction * self._baseline_stressed_volume * hill_function(
            eff_nore, cfg.nore_c50 * cfg.nore_venous_potency_ratio, 1.0,
        )
        phenyl_recruitment = cfg.phenyl_venous_recruitment_fraction * self._baseline_stressed_volume * hill_function(
            max(0.0, ce_phenyl), cfg.phenyl_c50, cfg.phenyl_gamma,
        )
        recruitment = nore_recruitment + phenyl_recruitment
        self._pressor_recruitment_ml = min(
            recruitment * pressor_gain,
            self.unstressed_volume + self._baseline_stressed_volume * (1.0 - stressed_fraction),
        )

    def get_hematocrit(self) -> float:
        return clamp(self.red_cell_volume_ml / self.blood_volume, 0.0, MAX_HEMATOCRIT)

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
            raw = self.failure.filling_pressure * math.expm1(gain * preload_factor) / math.expm1(gain)
        else:
            raw = self.config.lap_scale * math.expm1(self.config.lap_preload_gain * preload_factor)
        if raw > 36.0:
            return 36.0 + 12.0 * math.tanh((raw - 36.0) / 12.0)
        return raw

    @property
    def lung_water_ml_kg(self) -> float:
        """Extravascular lung water above normal, mL/kg predicted body weight."""
        return self.lung_water_ml / self._pbw

    def _frank_starling(self, preload_factor: float, inotropy: float = 1.0) -> float:
        """Preload response with a plateau at normal filling (Rebet 2016; Sherry 1995)."""
        pf = clamp(preload_factor, 0.0, 1.0)
        if self.patient.heart_failure == "hfref" and pf > 0.0:
            # HFrEF starts on a flatter upper Starling limb (Sherry 1995).
            pf += pf * pf * (1.0 - pf)
        normalized = (1.0 - math.exp(-2.0 * pf)) / self._frank_starling_baseline_raw
        return max(0.0, inotropy * normalized)

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
        self, ce_epi: float, ce_pressor: float, ce_high_pressor: float,
        beta1_dr: float = 1.0, alpha_dr: float = 1.0
    ) -> tuple[float, float, float]:
        """Return direct chronotropy, SV factor, and baseline-relative SVR factor.

        The pressor concentrations lag the effect-site concentration, with a
        separate lag for high alpha. All three are equal at steady exposure.
        Beta-2 dilation scales with current vascular tone; alpha tone adds
        relative to baseline.
        Cardiac and primary vascular sensitivity decline with age alongside
        reduced clearance (Leenen 2005; Pan 1986; Hogikyan 1994). Antagonist dose
        ratios divide the concentration seen by each receptor.
        """
        cfg = self.config
        age_scale = self._age_chronotropy
        delta_hr = cfg.epi_emax_hr * hill_function(ce_epi / beta1_dr, cfg.epi_c50_hr, cfg.epi_gamma_hr)
        delta_hr *= (1.0 - cfg.epi_volatile_hr_depression * clamp01(self.ce_sevo)) * age_scale
        sv_factor = 1.0 + cfg.epi_emax_sv * (1 + cfg.epi_age_sv * (age_scale - 1)) * hill_function(
            ce_pressor / beta1_dr, cfg.epi_c50_sv, 1.0
        )
        alpha_age_scale = 1 + cfg.epi_age_alpha * (age_scale - 1)
        alpha = cfg.epi_emax_svr_alpha * hill_function(
            ce_pressor * alpha_age_scale / alpha_dr, cfg.epi_c50_alpha, cfg.epi_gamma_alpha
        ) + cfg.epi_emax_svr_alpha_high * hill_function(
            ce_high_pressor / alpha_dr, cfg.epi_c50_alpha_high, cfg.epi_gamma_alpha_high
        )
        svr_factor = (
            1.0 + self._epi_beta2_effect * (1 + cfg.epi_age_beta2 * (age_scale - 1)) * self.tpr / self.base_tpr
            + alpha
        )
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

    @property
    def _electrically_silent(self) -> bool:
        return (
            self.coronary.electrically_silent
            or self.sodium_block >= SILENT_BLOCK
            or self.hyperkalemia_block >= SILENT_BLOCK
        )

    def _effective_lap(self) -> float:
        return self._last_lap + self._pulm_arrest_blend * (self._last_mcfp - self._last_lap)

    def _pit_preload_factor(self, pit: float) -> float:
        delta_pit = pit - self.config.pit_0
        raw = (
            1.0 / (1.0 + self.config.alpha_peep * delta_pit)
            if delta_pit >= 0.0
            else 1.0 + self.config.alpha_peep * (-delta_pit)
        )
        return clamp(raw, 0.0, 1.4)

    def _update_pulmonary_coupling(
        self,
        dt: float,
        pao2: float,
        peep_cmH2O: float | None,
        f_preload_pit: float,
        sepsis_sev: float,
        pit: float = -2.0,
    ) -> float:
        """Return the Frank-Starling factor after venous return and pulmonary transit.

        Venous return follows MCFP - RAP, is attenuated by PVR, and reaches the
        left heart through a first-order transit delay. Below baseline filling,
        RAP falls in proportion to MCFP, the intersection with a cardiac
        function curve that is linear at low filling pressures.
        """
        cfg = self.config
        rhythm = self._rhythm_type
        rhythm_beating = (
            not self._electrically_silent
            and rhythm not in PULSELESS_RHYTHMS
            and not self._compressions
        )
        flow_activity = clamp01(self._last_intrinsic_co / ARREST_CO_L_MIN) if rhythm_beating else 0.0
        beating = rhythm_beating and self._last_intrinsic_co > 0.0
        normal_base_co = max(0.1, self.base_co_l_min / self.failure.output_factor)
        lv_output_ratio = self._last_intrinsic_co / normal_base_co
        structural_factor = (
            self.failure.output_factor
            * (1.0 - self.coronary.injury)
            * (1.0 - self.toxic_contractility_loss)
            * self.acidosis_factor
        )
        lv_pump_factor = min(1.0, structural_factor, lv_output_ratio)
        lap_congested = self._last_lap > self._lung_water_threshold
        unthrottled_rv_co = self._pulm_flow_factor * self.base_co_l_min
        flow_mismatch = (
            clamp01((unthrottled_rv_co - self._last_intrinsic_co) / max(0.5, unthrottled_rv_co))
            if beating and structural_factor < 0.90
            else 0.0
        )
        pump_deficit = max(
            clamp01((0.45 - lv_pump_factor) / 0.20),
            0.55 if lap_congested else 0.0,
            clamp01((flow_mismatch - 0.10) / 0.25),
        )
        filling_gate = clamp01((self._last_venous_preload - 0.45) / 0.20)
        congestion_cap = 600.0 * (pump_deficit * pump_deficit) * filling_gate * flow_activity
        if dt > 0.0:
            arrest_target = 0.0 if self._compressions else 1.0 - flow_activity
            blend_tau = 2.5 if arrest_target < self._pulm_arrest_blend else 5.0
            self._pulm_arrest_blend += (arrest_target - self._pulm_arrest_blend) * -math.expm1(-dt / blend_tau)
            if beating and congestion_cap > 0.0:
                imbalance = self._last_rv_co * flow_activity - self._last_intrinsic_co
                self._pulmonary_congestion_ml = clamp(
                    self._pulmonary_congestion_ml + imbalance * 1000.0 * (dt / 60.0),
                    0.0,
                    congestion_cap,
                )
            else:
                decay_tau = 12.0 if beating else (15.0 if self._compressions else 20.0)
                self._pulmonary_congestion_ml *= math.exp(-dt / decay_tau)

        stressed_vol = self._calc_stressed_volume(sepsis_sev)
        mcfp = stressed_vol / self.cv if self.cv > 0 else self.mcfp_0

        pvr_factor = self._calc_pvr_factor(pao2, peep_cmH2O)
        congestion_lap = self._pulmonary_congestion_ml / 10.0
        rv_backpressure_factor = 1.0 / (1.0 + 0.14 * max(0.0, congestion_lap - 2.0) ** 1.25)
        vr_rap = cfg.rap_baseline * min(1.0, mcfp / self.mcfp_0)
        starling_keep = (
            clamp01((mcfp - pit) / max(1.0, 0.35 * mcfp))
            if pit > 4.0 and pit > 0.65 * mcfp
            else 1.0
        )
        delta_p = max(0.0, mcfp - vr_rap) * starling_keep
        vr_flow_l_min = delta_p / max(0.1, self.venous_return_resistance)
        vr_flow_factor = vr_flow_l_min * self._inv_base_co_l_min
        rap = max(
            pit,
            vr_rap
            + 0.45 * max(0.0, mcfp - self.mcfp_0)
            + 2.0 * max(0.0, pvr_factor - 1.15)
            + 0.35 * max(0.0, congestion_lap - 2.0)
            + max(0.0, (1.0 - f_preload_pit) * 10.0),
        )
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
            pit_delay_tau = min(1.1, 0.35 * cfg.pulmonary_transit_time_s)
            pit_alpha = -math.expm1(-dt / pit_delay_tau)
            self._rv_pit_preload += (f_preload_pit - self._rv_pit_preload) * pit_alpha
            self._lv_pit_preload = clamp(
                self._rv_pit_preload + 0.25 * (self._rv_pit_preload - f_preload_pit), 0.0, 1.4
            )
        elif cfg.pulmonary_transit_time_s <= 0:
            self._pulm_flow_factor = rv_out_target_factor
            self._pulm_reference_factor = reference_flow_factor
            self._rv_pit_preload = self._lv_pit_preload = f_preload_pit

        self._pulm_flow_factor = clamp(self._pulm_flow_factor, 0.0, 3.0)

        position_target = self._supine_preload if self.position == "supine" else 1.0
        if dt > 0.0 and self._position_preload != position_target:
            self._position_preload += (position_target - self._position_preload) * -math.expm1(-dt / 10.0)
        venous_preload = self._pulm_flow_factor / self._pulm_reference_factor * self._position_preload
        f_preload = venous_preload * self._lv_pit_preload
        lv_preload = f_preload + self._pulmonary_congestion_ml / 400.0
        f_frank_starling = self._frank_starling(lv_preload)

        base_lap = self._lap(f_preload)
        raw_lap = base_lap + congestion_lap * clamp01((0.45 - structural_factor) / 0.15)
        self._last_lap = 36.0 + 12.0 * math.tanh((raw_lap - 36.0) / 12.0) if raw_lap > 36.0 else raw_lap
        self._last_mcfp = mcfp
        self._last_rap = rap
        self._last_pvr = cfg.pvr_wood_baseline * pvr_factor
        self._last_rv_co = rv_out_target_factor * rv_backpressure_factor * self.base_co_l_min
        self._last_lv_inflow = self._pulm_flow_factor * rv_backpressure_factor * self.base_co_l_min
        self._last_venous_preload = venous_preload
        self._last_preload_factor = f_preload
        self._last_preload_sv_factor = f_frank_starling

        return f_frank_starling

    def _calc_nore_effects(
        self, ce_nore: float, beta1_dr: float = 1.0, alpha_dr: float = 1.0,
        anesthetic_depth: float = 0.0, *, nore_scale: float = 1.0,
    ) -> tuple:
        """Return direct chronotropy, stroke-volume factor, and vascular-resistance factor.

        Reflex bradycardia comes from the baroreflex.
        """
        cfg = self.config
        eff_ce = max(0.0, ce_nore - self._basal_nore_ce * (1.0 - anesthetic_depth)) * nore_scale
        if eff_ce <= 0:
            return 0.0, 1.0, 1.0
        beta_hill = hill_function(eff_ce / beta1_dr, cfg.nore_c50, 1.0)
        delta_hr = cfg.nore_emax_hr * beta_hill * self._age_chronotropy
        sv_gain = cfg.nore_emax_sv + cfg.nore_anesthetic_sv_gain * anesthetic_depth
        sv_factor = 1.0 + sv_gain * beta_hill
        alpha_hill = hill_function(eff_ce / alpha_dr, cfg.nore_c50, 1.0)
        alpha_gain = 1.0 + cfg.nore_anesthetic_alpha_gain * anesthetic_depth
        svr_factor = 1.0 + cfg.nore_emax_svr * alpha_gain * alpha_hill
        return delta_hr, sv_factor, svr_factor

    def _rhythm_sv_factor(self, current_hr: float) -> float:
        rhythm = self._rhythm_type
        if rhythm is RhythmType.SINUS:
            return 1.0
        factor = RHYTHM_SV_FACTOR.get(rhythm, 1.0)
        if rhythm is RhythmType.AFIB:
            excess = max(0.0, current_hr - 90.0)
            eff_excess = min(excess, 80.0) + 0.25 * max(0.0, excess - 80.0)
            return factor / (1.0 + 0.006 * eff_excess + 0.00018 * eff_excess * eff_excess)
        return factor

    def _hr_feedback(self, map_feedback: float, tachycardia_depth: float) -> float:
        # Anesthesia removes the tachycardic side of the slow MAP feedback;
        # pressure-induced slowing remains (Takahashi 2002's late epinephrine fall).
        return map_feedback - max(0.0, map_feedback - 1.0) * (
            self.config.hr_feedback_anesthetic_depression * tachycardia_depth
        )

    def _update_anesthetic_factors(
        self, ce_prop: float, ce_opioid: float, ce_sevo: float, vagal: float = 0.0,
    ) -> None:
        """Apply combined anesthetic effects on TPR, SV, and HR (Su et al. 2023).

        Propofol and remifentanil-equivalent concentrations are at the
        hemodynamic effect sites.
        """
        cfg = self.config
        cp = max(0.0, ce_prop)
        cr = max(0.0, ce_opioid)
        if cfg.opioid_cv_saturation > 0.0:
            cr = cfg.opioid_cv_saturation * math.tanh(cr / cfg.opioid_cv_saturation)

        # Propofol TPR effect, with remifentanil interaction.
        remi_int_term = cfg.int_tpr * hill_function(cr, cfg.ec50_remi_tpr, 1.0)
        eff_prop_tpr = (cfg.emax_prop_tpr * self._sympathetic_gain + remi_int_term) * hill_function(
            cp, cfg.ec50_prop_tpr, cfg.gamma_prop)
        eff_prop_sv = self._emax_prop_sv_age * hill_function(cp, cfg.ec50_prop_sv, 1.0)

        eff_sevo_tpr = cfg.sevo_emax_tpr * hill_function(ce_sevo, cfg.sevo_ec50_tpr, cfg.sevo_gamma_tpr)
        eff_sevo_sv = cfg.sevo_emax_sv * hill_function(ce_sevo, cfg.sevo_ec50_sv, 1.0)

        # Propofol and sevoflurane act as independent fractions.
        total_eff_tpr = max(-0.95, (1.0 + eff_prop_tpr) * (1.0 + eff_sevo_tpr) - 1.0)
        total_eff_sv = max(-0.95, (1.0 + eff_prop_sv) * (1.0 + eff_sevo_sv) - 1.0)
        total_eff_hr = cfg.sevo_emax_hr * hill_function(ce_sevo, cfg.sevo_ec50_hr, 1.0)

        # Remifentanil effects divide tone, as Su's dissipation terms do at
        # steady state; propofol modulates the SV and HR slopes.
        eff_remi_tpr = cfg.emax_remi_tpr * hill_function(cr, cfg.ec50_remi_tpr, cfg.gamma_remi_tpr)
        slope_sv = cfg.sl_remi_sv + cfg.int_sv * hill_function(cp, cfg.ec50_int_sv, 1.0)
        eff_remi_sv = slope_sv * cr
        slope_hr = cfg.sl_remi_hr + cfg.int_hr * hill_function(cp, cfg.ec50_int_hr, 1.0)
        eff_remi_hr = max(-0.9, min(0.9, slope_hr * cr))
        if eff_remi_hr < 0.0:
            eff_remi_hr *= 1.0 - vagal  # Opioid bradycardia is vagal.
        self._drug_tpr = (1.0 + total_eff_tpr) / max(0.05, 1.0 - eff_remi_tpr)
        self._drug_sv = (1.0 + total_eff_sv) / max(0.05, 1.0 - eff_remi_sv)
        remi_vagal = cfg.remi_vagal_hr * hill_function(self.ce_remi_vagal, cfg.remi_vagal_c50, 1.0)
        self._drug_hr = (1.0 + total_eff_hr) / max(0.05, 1.0 - eff_remi_hr) * (1.0 - remi_vagal * (1.0 - vagal))

    @staticmethod
    def _hypothermia_factors(temp_c: float) -> tuple[float, float]:
        return (
            clamp(1.0 - 0.045 * max(0.0, 33.0 - temp_c), 0.35, 1.0),
            clamp(1.0 - 0.05 * max(0.0, 30.0 - temp_c), 0.45, 1.0),
        )

    @staticmethod
    def _thermo_tpr_mult(temp_c: float, vasoconstriction_threshold: float) -> float:
        if temp_c >= vasoconstriction_threshold:
            return 1.0
        cold_drop = vasoconstriction_threshold - temp_c
        vasoparesis = clamp(1.0 - 0.18 * max(0.0, 32.5 - temp_c), 0.2, 1.0)
        return min(2.0, 1.0 + TEMP_TPR_COEFFICIENT * min(cold_drop, 2.5) * vasoparesis)

    def _effective_beating_hr(self, rhythm_hr: float) -> float:
        return (
            rhythm_hr
            * (1.0 - self.config.hypoxia_hr_depression * self.myocardial_hypoxia)
            * self.coronary.electrical_factor
            * self._hypothermia_hr_factor
            * (1.0 - self.sodium_block)
            * (1.0 - 0.75 * self.hyperkalemia_block)
        )

    def _apply_afterload_coupling(
        self,
        sv: float,
        eff_tpr: float,
        hr: float,
        non_phenyl_sv_factor: float,
        *,
        phenyl_svr_factor: float = 1.0,
        dist_sv: float = 0.0,
        beta_occupancy: float = 0.0,
        amiodarone_ees_factor: float = 1.0,
    ) -> float:
        """Apply ventricular-arterial coupling (Asanoi 1989; Chen 1998)."""
        coupling = self.failure.coupling_ratio
        if coupling:
            sepsis_ees = 1.0 - 0.35 * clamp01(self.sepsis_severity)
            ees_support = 1.0 + 0.75 * max(0.0, non_phenyl_sv_factor - 1.0)
            load = max(0.2, eff_tpr / self.base_tpr) / (amiodarone_ees_factor * sepsis_ees * ees_support)
            return sv * ((1.0 + coupling) / (1.0 + coupling * load))
        r0 = self.config.normal_coupling_ratio
        if r0 <= 0.0:
            return sv
        phenyl_tpr_delta = self.base_tpr * max(0.0, phenyl_svr_factor - 1.0)
        non_phenyl_tpr_ratio = max(0.2, (eff_tpr - phenyl_tpr_delta) / self.base_tpr)
        if non_phenyl_tpr_ratio <= 1.0:
            return sv
        dist_ees = max(0.0, dist_sv) / max(1.0, self.base_sv)
        ea_excess = (non_phenyl_tpr_ratio - 1.0) * max(1.0, hr / self.base_hr)
        physio_match = (0.45 + dist_ees) * (1.0 - 0.5 * beta_occupancy)
        eff_ea_ratio = 1.0 + max(0.0, ea_excess - physio_match)
        if eff_ea_ratio <= 1.0:
            return sv
        ees_ratio = max(0.25, min(1.65, non_phenyl_sv_factor + dist_ees) * amiodarone_ees_factor)
        return sv * ((ees_ratio + r0) / (ees_ratio + r0 * eff_ea_ratio))

    def _cap_sv_by_venous_return(
        self, sv: float, eff_tpr: float, hr: float, vaso_sv_factor: float,
    ) -> float:
        if sv <= 0.0 or hr <= 0.0:
            return sv
        tpr_ratio_vr = eff_tpr / self.base_tpr
        hyperdynamic_gate = clamp01((vaso_sv_factor - 1.10) / 0.20)
        vaso_vr_gate = hyperdynamic_gate * clamp01((tpr_ratio_vr - 1.15) / 0.25)
        hr_vr_gate = clamp01((hr - 130.0) / 35.0)
        if vaso_vr_gate > 0.0 or hr_vr_gate > 0.0:
            eff_rvr = max(0.1, self.venous_return_resistance) * (1.0 + 0.35 * max(0.0, tpr_ratio_vr - 1.0))
            vr_headroom = 1.0 + 0.25 * clamp01(1.15 - tpr_ratio_vr)
            ref_scale = 1.0 / max(0.55, self._pulm_reference_factor)
            max_vr_co = max(
                0.0,
                self._last_mcfp / eff_rvr * ref_scale * max(0.0, self.f_preload_pit) * vr_headroom,
            )
            vr_sv_cap = max_vr_co * 1000.0 / hr
            if vaso_vr_gate > 0.0:
                vaso_floor = sv * 1.10 / max(1.10, vaso_sv_factor)
                vaso_target = max(vr_sv_cap, vaso_floor)
                if vaso_target < sv:
                    sv += vaso_vr_gate * (vaso_target - sv)
            if hr_vr_gate > 0.0 and vr_sv_cap < sv:
                sv += hr_vr_gate * (vr_sv_cap - sv)
        return sv

    def _compute_intrinsic(
        self,
        preload_sv_factor: float,
        sepsis_sev: float,
        distributive_tpr_offset: float,
        hr_base: float,
        *,
        include_stim: bool = True,
    ) -> tuple[float, float, float, float, float, bool]:
        cfg = self.config
        rhythm = self._rhythm_type
        eff_dist_hr = self.dist_hr if include_stim else 0.0
        eff_dist_sv = self.dist_sv if include_stim else 0.0
        eff_dist_svr = self.dist_svr if include_stim else 0.0
        eff_crisis_hr = self.crisis_hr if include_stim else 0.0
        eff_crisis_svr = self.crisis_svr if include_stim else 0.0
        sinus_hr = clamp(hr_base + eff_crisis_hr + eff_dist_hr, HR_MIN, HR_MAX)
        current_hr = self._rhythm_rate(sinus_hr)
        reflex_tpr_factor = 1.0
        if rhythm is not RhythmType.SINUS and self.smoothed_baro_hr > 0.0:
            # When the rhythm holds rate low during hypotension, vasoconstriction compensates.
            reflex_hr = self.smoothed_baro_hr
            blocked_hr = max(0.0, reflex_hr - (current_hr - self._rhythm_rate(sinus_hr - reflex_hr)))
            # Blocked drug chronotropy must not dilute the vascular reflex.
            reflex_hr_base = max(hr_base + eff_crisis_hr + eff_dist_hr - reflex_hr - self.smoothed_epi_hr, HR_MIN)
            reflex_tpr_gain = self._venous_tone * (1.0 - sepsis_sev) * self.acidosis_factor
            reflex_tpr_factor = 1.0 + (blocked_hr / reflex_hr_base) * reflex_tpr_gain
        current_hr = self._effective_beating_hr(current_hr)

        term = self._hr_sv_factor(current_hr)
        if (
            (rhythm is RhythmType.SINUS or rhythm not in RHYTHM_SV_FACTOR)
            and (self._unstim_rate > 0.0 or self._unstim_rate_unloaded > 0.0)
        ):
            preload_avail = (
                clamp01((self._last_venous_preload - 0.75) / 0.25)
                if self.patient.heart_failure == "none"
                else 0.0
            )
            depletion = max(0.0, 1.0 - self._last_venous_preload)
            symp_sv = (
                max(0.0, self.vasopressor_sv_factor - 1.0)
                + max(0.0, eff_dist_sv) / max(1.0, self.base_sv)
            )
            unloaded = min(self._unstim_rate_unloaded, self._unstim_rate)
            eff_unloaded = max(0.0, unloaded - 2.5 * symp_sv)
            eff_unstim = max(0.0, self._unstim_rate - 2.5 * symp_sv)
            reservoir_support = cfg.unstim_sv_reservoir * max(0.0, eff_unstim - eff_unloaded) * preload_avail
            unload_loss = cfg.unstim_sv_unload * (eff_unloaded + 4.4 * depletion * eff_unstim)
            term *= (1.0 + reservoir_support) / (1.0 + unload_loss)
        if self.patient.heart_failure == "hfref" or self.tde_sv <= 0.0:
            eff_tde_sv = 0.0
        else:
            tde_cap = (0.10 if self.patient.heart_failure == "hfpef" else 0.35) * self.base_sv
            eff_tde_sv = tde_cap * math.tanh(self.tde_sv / max(1e-6, tde_cap))
        raw_sv = (self.sv_star + eff_tde_sv) * term + eff_dist_sv

        # TPR floor of about 6 Wood units.
        endogenous_tpr = self.tpr * (1.0 - self.smoothed_baro_vasc) * reflex_tpr_factor
        eff_tpr = max(
            0.006,
            (
                endogenous_tpr
                + self.delta_tpr_vasopressors
                + (eff_crisis_svr + eff_dist_svr) / 1000.0
                + distributive_tpr_offset
            )
            * self._ca_tpr_factor,
        )

        current_sv = raw_sv * preload_sv_factor * self.vasopressor_sv_factor
        if self.patient.heart_failure != "none" and self._last_lap > 30.0:
            current_sv *= 1.0 - 0.22 * clamp01((self._last_lap - 30.0) / 18.0)
        current_sv = self._apply_afterload_coupling(
            current_sv,
            eff_tpr,
            current_hr,
            self._non_phenyl_sv_factor,
            phenyl_svr_factor=self._phenyl_svr_factor,
            dist_sv=eff_dist_sv,
            beta_occupancy=self._beta_occupancy,
            amiodarone_ees_factor=self._amiodarone_ees_factor,
        )
        # Reversible O2-supply shortfalls share one ceiling so global hypoxemia
        # is not counted twice; countershock stunning, permanent injury, and
        # electrical failure apply multiplicatively on top.
        coronary = self.coronary
        reversible_o2_loss = max(
            0.50 * coronary.ischemia,
            cfg.hypoxia_sv_depression * self.myocardial_hypoxia,
        )
        coronary_sv_keep = (
            (1.0 - reversible_o2_loss)
            * (1.0 - coronary.shock_stunning)
            * (1.0 - coronary.injury)
            * coronary.electrical_factor
        )
        current_sv = max(0.0, current_sv) * coronary_sv_keep * self.acidosis_factor

        if rhythm is not RhythmType.SINUS:
            current_sv *= self._rhythm_sv_factor(current_hr)
        current_sv *= (
            (1.0 - self.toxic_contractility_loss)
            * self._ca_sv_factor
            * self._hypothermia_sv_factor
            * (1.0 - 0.55 * self.hyperkalemia_block)
        )

        electrically_silent = self._electrically_silent
        if (
            electrically_silent
            or (rhythm is not RhythmType.SINUS and rhythm in PULSELESS_RHYTHMS)
            or current_hr <= 0.0
        ):
            current_hr = current_sv = 0.0
        elif current_sv > 0.0:
            current_sv = self._cap_sv_by_venous_return(
                current_sv, eff_tpr, current_hr, self.vasopressor_sv_factor,
            )
        if current_hr <= 0.0 or current_sv <= 0.0:
            intrinsic_co = intrinsic_map = 0.0
        else:
            intrinsic_co = current_hr * current_sv / 1000.0
            intrinsic_map = clamp(current_hr * current_sv * eff_tpr, 0.0, 300.0)
        return current_hr, current_sv, eff_tpr, intrinsic_co, intrinsic_map, electrically_silent

    def _compute_state(
        self,
        preload_sv_factor: float | None = None,
        sepsis_sev: float | None = None,
        anaph_sev: float | None = None,
        distributive_tpr_offset: float | None = None,
        hr_base: float | None = None,
        *,
        _include_stim: bool = True,
        _update_stim: bool = False,
        _cache: bool = True,
    ) -> HemoState:
        cfg = self.config
        if sepsis_sev is None:
            sepsis_sev = clamp01(self._sepsis_severity)
        if anaph_sev is None:
            anaph_sev = clamp01(self._anaphylaxis_severity)
        if distributive_tpr_offset is None:
            sepsis_drop_wood = cfg.sepsis_svr_drop_wood / self.failure.output_factor
            distributive_tpr_offset = -(
                sepsis_drop_wood * sepsis_sev + cfg.anaphylaxis_svr_drop_wood * anaph_sev
            ) / 1000.0
        if hr_base is None:
            hr_base = self._calc_hr()
        if preload_sv_factor is None:
            preload_sv_factor = self._last_preload_sv_factor

        current_hr, current_sv, eff_tpr, intrinsic_co, intrinsic_map, electrically_silent = self._compute_intrinsic(
            preload_sv_factor,
            sepsis_sev,
            distributive_tpr_offset,
            hr_base,
            include_stim=_include_stim,
        )
        co = intrinsic_co
        map_val = perfusion_map = intrinsic_map
        svr_val = map_val / co if co > 0.0 else 0.0
        if _include_stim and _update_stim:
            if (
                current_hr <= 0.0
                or current_sv <= 0.0
                or (
                    self.dist_hr == 0.0
                    and self.dist_sv == 0.0
                    and self.dist_svr == 0.0
                    and self.crisis_hr == 0.0
                    and self.crisis_svr == 0.0
                )
            ):
                self._stim_map = 0.0
            else:
                _, _, _, _, unstim_map, _ = self._compute_intrinsic(
                    preload_sv_factor,
                    sepsis_sev,
                    distributive_tpr_offset,
                    hr_base,
                    include_stim=False,
                )
                self._stim_map = clamp(intrinsic_map - unstim_map, 0.0, intrinsic_map)
        venous_pressure = self._last_rap
        rap_val = self._last_rap
        lap_val = self._effective_lap()
        rv_co_val = self._last_rv_co
        lv_inflow_val = self._last_lv_inflow
        cpr_perfusion_map = 0.0
        if self._compressions:
            # Compressions pump what fills the heart against arterial resistance;
            # perfusion_map drives organ flow while map tracks arterial lumen mean.
            load = max(0.2, eff_tpr / self.base_tpr)
            afterload = (1.0 + COMPRESSION_AFTERLOAD_COUPLING) / (1.0 + COMPRESSION_AFTERLOAD_COUPLING * load)
            base_co_ref = self.base_co_l_min / self.failure.output_factor
            cpr_co = COMPRESSION_OUTPUT_FRACTION * base_co_ref * preload_sv_factor * afterload
            cpr_perfusion_map = clamp(cpr_co * 1000.0 * eff_tpr, 0.0, 300.0)
            co = max(intrinsic_co, cpr_co)
            perfusion_map = max(intrinsic_map, cpr_perfusion_map)
            svr_val = 1000.0 * eff_tpr if co > 0.0 else 0.0
            # Venous return, (MCFP - RAP) / resistance, sets the atrial pressure.
            venous_pressure = max(0.0, self._last_mcfp - co * self.venous_return_resistance)
            cpr_sbp, cpr_dbp = compression_pressures(cpr_perfusion_map, venous_pressure)
            cpr_art_map = clamp(
                cpr_dbp + (cpr_sbp - cpr_dbp) * 2.0 * COMPRESSION_DUTY / math.pi,
                0.0,
                300.0,
            )
            map_val = max(intrinsic_map, cpr_art_map)
            if intrinsic_co < cpr_co:
                rap_val = venous_pressure
                lap_val = max(lap_val, rap_val)
                rv_co_val = lv_inflow_val = co
        elif co <= 0.0:
            # Without flow, pressures equilibrate at mean systemic filling pressure
            # (or intrathoracic pleural pressure when positive ITP collapses the venae cavae).
            venous_pressure = rap_val = max(self._last_rap, self._last_mcfp)
            rv_co_val = lv_inflow_val = 0.0
        elif co < ARREST_CO_L_MIN:
            arrest_share = 1.0 - clamp01(co / ARREST_CO_L_MIN)
            guyton_vp = max(
                self._last_rap,
                clamp(self._last_mcfp - co * self.venous_return_resistance, 0.0, self._last_mcfp),
            )
            venous_pressure = rap_val = self._last_rap + arrest_share * (guyton_vp - self._last_rap)
            rv_co_val = self._last_rv_co + arrest_share * (co - self._last_rv_co)
            lv_inflow_val = self._last_lv_inflow + arrest_share * (co - self._last_lv_inflow)

        rhythm = self._rhythm_type
        result = HemoState(
            map=map_val,
            hr=current_hr,
            sv=current_sv,
            pulse_pressure_factor=self._epi_pulse_factor,
            svr=svr_val,
            co=co,
            tpr=self.tpr,
            sv_star=self.sv_star,
            hr_star=self.hr_star,
            tde_sv=self.tde_sv,
            tde_hr=self.tde_hr,
            ce_sevo=self.ce_sevo,
            mcfp=self._last_mcfp,
            rap=rap_val,
            lap=lap_val,
            pvr=self._last_pvr,
            rv_co=rv_co_val,
            lv_inflow=lv_inflow_val,
            preload_factor=self._last_preload_factor,
            rhythm_type=RhythmType.ASYSTOLE
            if (electrically_silent or (current_hr <= 0.0 and rhythm is not RhythmType.VFIB))
            else rhythm,
            compressions=self._compressions,
            intrinsic_co=intrinsic_co,
            intrinsic_map=intrinsic_map,
            perfusion_map=perfusion_map,
            cpr_perfusion_map=cpr_perfusion_map,
            venous_pressure=venous_pressure,
        )
        if _cache:
            self._cached_state = result
        return result

    def _rhythm_rate(self, sinus_hr: float) -> float:
        """Ventricular rate from the underlying sinus rate; untreated AF with RVR is 110-180+ bpm."""
        rhythm = self._rhythm_type
        if rhythm is RhythmType.SINUS:
            return sinus_hr
        fixed = FIXED_RHYTHM_RATE.get(rhythm)
        if fixed is not None:
            return fixed
        if rhythm is RhythmType.AFIB:
            excess_sinus = max(0.0, sinus_hr - 76.0)
            rvr_drive = 110.0 + 95.0 * math.tanh(1.25 * excess_sinus / 95.0)
            return min(HR_MAX, max(sinus_hr, rvr_drive * (1.0 - self.config.beta_block_af_rate * self._beta_occupancy)))
        if rhythm is RhythmType.SINUS_BRADY:
            return min(sinus_hr, 50.0 + self.config.glyco_brady_relief * self._vagal_block + self._adrenergic_chrono)
        return sinus_hr

    @property
    def state(self) -> HemoState:
        if self._cached_state is not None:
            return self._cached_state
        return self._compute_state()

    def evaluate_state(self, *, include_stim: bool = True) -> HemoState:
        """Evaluate HemoState without updating the state cache."""
        return self._compute_state(_include_stim=include_stim, _cache=False)

    @property
    def tpr(self) -> float:
        return self.tpr_tone * self._drug_tpr

    @property
    def sv_star(self) -> float:
        return self.sv_tone * self._drug_sv

    @property
    def hr_star(self) -> float:
        return self.hr_tone * self._drug_hr

    def _calc_hr(self):
        return self.hr_star + self.tde_hr + self.smoothed_chemo_hr + self.smoothed_baro_hr + self.smoothed_epi_hr

    def _hr_sv_factor(self, hr: float) -> float:
        # Su Eq. 9 also increases filling when HR falls below baseline. Heart
        # failure, especially HFpEF, loses more filling during tachycardia, and
        # extreme rates (>170 bpm) curtail diastolic ventricular filling.
        factor = max(0.1, 1.0 - self.config.hr_sv_coupling * math.log(max(35.0, hr) / self.base_hr))
        severe_tachy = max(0.0, (hr - 170.0) / 45.0)
        diastolic_limit = 1.0 + 1.8 * severe_tachy * severe_tachy
        return factor / ((1.0 + self.failure.tachycardia_gain * max(0.0, hr / self.base_hr - 1.0)) * diastolic_limit)

    def step(self, dt: float, cp_prop: float, cp_remi: float, ce_nore: float, pit: float, paco2: float, pao2: float,
             dist_hr: float = 0.0, dist_sv: float = 0.0, dist_svr: float = 0.0,
             mac_sevo: float = 0.0, ce_epi: float = 0.0, ce_phenyl: float = 0.0,
             ce_vaso: float = 0.0, ce_dobu: float = 0.0, ce_mil: float = 0.0,
             ce_esmolol: float = 0.0, ce_labetalol: float = 0.0, ce_glyco: float = 0.0,
             ce_ketamine: float = 0.0, ce_ketamine_hr: float | None = None,
             ce_atropine: float = 0.0, ce_ephedrine: float = 0.0,
             ce_amiodarone: float = 0.0, ce_amiodarone_vasc: float = 0.0, ce_amiodarone_hr: float = 0.0,
             sodium_block: float = 0.0, toxic_contractility_loss: float = 0.0,
             temp_c: float = 37.0, peep_cmH2O: float | None = None, sao2: float = 98.0,
             remifentanil_cp: float | None = None, ph: float = 7.4,
             ce_midazolam: float = 0.0, ce_etomidate: float = 0.0,
             mac_n2o: float = 0.0,
             crisis_hr: float = 0.0, crisis_svr: float = 0.0) -> HemoState:
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
            ce_ketamine, ce_ketamine_hr: Ketamine cardiac-output and HR effect
                sites (mcg/mL); ce_ketamine_hr defaults to ce_ketamine.
            ce_amiodarone, ce_amiodarone_vasc, ce_amiodarone_hr: Amiodarone contractility,
                vascular, and HR effect sites (mcg/mL).
            temp_c: Core temperature (°C).
            peep_cmH2O: Total PEEP (cmH2O).
            sao2: Arterial saturation (%).
            remifentanil_cp: Remifentanil alone (ng/mL), for its vagal
                bradycardia; defaults to cp_remi.
        """
        if dt > 1.0:
            count = math.ceil(dt / 0.5)
            sub_dt = dt / count
            state = self.state
            for _ in range(count):
                state = self.step(
                    sub_dt, cp_prop, cp_remi, ce_nore, pit, paco2, pao2,
                    dist_hr=dist_hr, dist_sv=dist_sv, dist_svr=dist_svr,
                    mac_sevo=mac_sevo, ce_epi=ce_epi, ce_phenyl=ce_phenyl,
                    ce_vaso=ce_vaso, ce_dobu=ce_dobu, ce_mil=ce_mil,
                    ce_esmolol=ce_esmolol, ce_labetalol=ce_labetalol, ce_glyco=ce_glyco,
                    ce_ketamine=ce_ketamine, ce_ketamine_hr=ce_ketamine_hr,
                    ce_atropine=ce_atropine, ce_ephedrine=ce_ephedrine,
                    ce_amiodarone=ce_amiodarone, ce_amiodarone_vasc=ce_amiodarone_vasc,
                    ce_amiodarone_hr=ce_amiodarone_hr,
                    sodium_block=sodium_block, toxic_contractility_loss=toxic_contractility_loss,
                    temp_c=temp_c, peep_cmH2O=peep_cmH2O, sao2=sao2,
                    remifentanil_cp=remifentanil_cp, ph=ph,
                    ce_midazolam=ce_midazolam, ce_etomidate=ce_etomidate,
                    mac_n2o=mac_n2o,
                    crisis_hr=crisis_hr, crisis_svr=crisis_svr,
                )
            return state
        cfg = self.config
        base_co = self.base_co_l_min / self.failure.output_factor
        co_prev = self.state.co
        circulation = clamp01(co_prev / (CIRCULATION_LIMIT_FRACTION * base_co))
        self._cached_state = None
        self.sodium_block = clamp01(sodium_block)
        self.toxic_contractility_loss = clamp01(toxic_contractility_loss)
        dt_min = dt / 60.0
        delivery_time = dt_min * circulation
        sepsis_sev = clamp01(self._sepsis_severity)
        anaph_sev = clamp01(self._anaphylaxis_severity)

        # Hemodynamic effect sites: propofol's arterial, cardiac, and venous
        # effects follow Kazama 1999's SBP equilibration. Reflex depression
        # follows plasma propofol, as consciousness does within a minute.
        self.ce_prop_cv += (max(0.0, cp_prop) - self.ce_prop_cv) * -math.expm1(-self._ke0_prop_cv * delivery_time)
        self.ce_opioid_cv += (max(0.0, cp_remi) - self.ce_opioid_cv) * -math.expm1(-self._ke0_opioid_cv * delivery_time)
        remifentanil = max(0.0, cp_remi if remifentanil_cp is None else remifentanil_cp)
        self.ce_remi_vagal += (remifentanil - self.ce_remi_vagal) * -math.expm1(-self._ke0_opioid_cv * delivery_time)
        self.ce_sevo += (mac_sevo - self.ce_sevo) * -math.expm1(-cfg.ke0_sevo * delivery_time)
        ce_prop = self.ce_prop_cv * self._prop_cv_potency
        adjunct_midaz = hill_function(ce_midazolam, 100.0, 1.2) if ce_midazolam > 0.0 else 0.0
        adjunct_etom = hill_function(ce_etomidate, 0.35, 1.5) if ce_etomidate > 0.0 else 0.0
        adjunct_depth = adjunct_midaz + adjunct_etom
        anesthetic_depth = clamp01(self.ce_sevo + cp_prop / 4.0 + adjunct_depth)
        midaz_venous = (
            1.35
            * cfg.propofol_sympathetic_cp
            * adjunct_midaz
            * (0.45 + 0.55 * hill_function(self.ce_opioid_cv, 1.0, 1.0))
            if adjunct_midaz > 0.0
            else 0.0
        )
        self._sympathetic_venous_depth += (
            clamp01(max(0.0, cp_prop + midaz_venous) / cfg.propofol_sympathetic_cp)
            - self._sympathetic_venous_depth
        ) * -math.expm1(-math.log(2.0) / cfg.propofol_sympathetic_half_time * delivery_time)

        # Severe acidemia depresses myocardium and adrenergic responsiveness.
        # The curve below pH 7.2 is a teaching calibration (Schotola 2012).
        self.acidosis_factor = 1.0 - 0.7 * clamp01((7.2 - ph) / 0.4)
        self._hypothermia_hr_factor, self._hypothermia_sv_factor = self._hypothermia_factors(temp_c)
        base_ica = self.fluids.calcium_set_point
        ica = self.fluids.ionized_calcium(ph)
        potassium = self.fluids.concentration("potassium")
        # Lang 1988 & Denlinger 1975: blood ionized calcium drives myocardial
        # contractility directly without hypercalcemic vasoconstriction.
        hypocalcemia = clamp01((base_ica - ica) / 0.70)
        hypercalcemia = clamp01((ica - base_ica) / 0.50)
        self._ca_sv_factor = 1.0 - 0.42 * hypocalcemia + 0.22 * hypercalcemia
        self._ca_tpr_factor = 1.0 - 0.26 * hypocalcemia
        ca_protection = 1.5 * clamp((ica - base_ica) / 0.50, -0.4, 1.0)
        eff_k = potassium - ca_protection
        self.hyperkalemia_block = clamp01((eff_k - 6.2) / 3.0)
        self.hyperkalemia_arrhythmia = clamp01((eff_k - 6.8) / 2.2)

        if self._rhythm_type is RhythmType.ASYSTOLE:
            coronary = self.coronary
            # Primary asystole uses a calibrated CPR recovery timer;
            # post-shock asystole escapes in ArrhythmiaModel.
            can_restart = (
                self._compressions
                and not self.post_shock_asystole
                and coronary.supply_demand_ratio >= 1.0
                and coronary.energy_reserve > 0.0
                and not coronary.electrically_silent
                and not coronary.irreversible
                and self.sodium_block < 0.70
                and self.hyperkalemia_block < 0.70
            )
            self._asystole_recovery_s = self._asystole_recovery_s + dt if can_restart else 0.0
            if self._asystole_recovery_s >= cfg.asystole_recovery_time_s:
                self.rhythm_type = RhythmType.SINUS
        ce_ephedrine_raw = max(0.0, ce_ephedrine)
        ce_nore_raw = max(0.0, ce_nore)
        ce_nore = ce_nore_raw * self.acidosis_factor
        ce_epi *= self.acidosis_factor
        ce_phenyl *= self.acidosis_factor
        ce_dobu *= self.acidosis_factor
        ce_ephedrine = ce_ephedrine_raw * self.acidosis_factor

        # Antagonist dose ratios; esmolol is beta-1 selective.
        if ce_esmolol > 0.0 or ce_labetalol > 0.0:
            labetalol_beta = max(0.0, ce_labetalol) / cfg.labetalol_kb_beta
            beta1_dr = 1.0 + max(0.0, ce_esmolol) / cfg.esmolol_kb + labetalol_beta
            beta2_dr = 1.0 + labetalol_beta
            alpha_dr = 1.0 + max(0.0, ce_labetalol) / cfg.labetalol_kb_alpha
            self._beta_occupancy = beta_occ = 1.0 - 1.0 / beta1_dr
            alpha_occ = 1.0 - 1.0 / alpha_dr
        else:
            beta1_dr = beta2_dr = alpha_dr = 1.0
            self._beta_occupancy = beta_occ = alpha_occ = 0.0
        if ce_glyco > 0.0 or ce_atropine > 0.0:
            glyco = hill_function(ce_glyco, cfg.glyco_c50, cfg.glyco_gamma) if ce_glyco > 0.0 else 0.0
            atropine = hill_function(ce_atropine, cfg.atropine_c50, cfg.atropine_gamma) if ce_atropine > 0.0 else 0.0
            self._vagal_block = vagal = 1.0 - (1.0 - glyco) * (1.0 - atropine)
            atropine_hr = cfg.atropine_emax_hr * atropine / (1.0 + max(0.0, cp_prop) / cfg.atropine_propofol_cp50)
            glyco_hr = cfg.glyco_emax_hr * glyco
            vagal_hr = max(glyco_hr, glyco_hr * (1.0 - atropine) + atropine_hr)
        else:
            self._vagal_block = vagal = vagal_hr = 0.0
        resting_tone = 1.0 - cfg.sympathetic_anesthetic_depression * anesthetic_depth
        amiodarone = hill_function(ce_amiodarone, cfg.amiodarone_c50, 1.0) if ce_amiodarone > 0.0 else 0.0
        amiodarone_hr = hill_function(ce_amiodarone_hr, cfg.amiodarone_c50, 1.0) if ce_amiodarone_hr > 0.0 else 0.0
        # Shares of sympathetic reflex and stimulation responses left after
        # block. Age and amiodarone reduce the HR share.
        beta_response = 1.0 - cfg.beta_block_reflex * beta_occ
        sympathetic_hr_response = (
            beta_response * self._age_chronotropy * (1.0 - cfg.amiodarone_beta_reflex_block * amiodarone_hr)
        )
        alpha_response = 1.0 - alpha_occ
        dist_hr *= sympathetic_hr_response
        dist_sv *= beta_response
        # Septic shock removes stimulation-driven vasoconstriction, which depends
        # partly on the vasopressin that septic shock depletes.
        dist_svr *= alpha_response * (1.0 - sepsis_sev)
        crisis_hr *= sympathetic_hr_response
        crisis_svr *= alpha_response * (1.0 - sepsis_sev)
        self._venous_tone = alpha_response
        # Severe anaphylaxis blunts adrenergic agonists; vasopressin acts on V1 receptors.
        agonist_shift = 1.0 + cfg.anaphylaxis_agonist_shift * anaph_sev
        ce_epi /= agonist_shift
        ce_nore /= agonist_shift
        ce_phenyl /= agonist_shift
        ce_dobu /= agonist_shift

        # Urine output, scaled by renal perfusion and function; septic efferent
        # vasodilation lowers glomerular filtration pressure at a given MAP.
        map_prev = self._prev_map
        map_denom = max(1e-3, cfg.renal_map_norm - cfg.renal_map_min)
        self.renal_perfusion = (
            clamp01((map_prev - cfg.renal_map_min) / map_denom) * (1.0 - 0.12 * sepsis_sev)
        )
        renal_factor = self.renal_perfusion * self.patient.renal_clearance_factor
        # Keep plasma losses within the hematocrit represented by the chemistry model.
        minimum_blood_volume = max(BLOOD_VOLUME_MIN, self.red_cell_volume_ml / MAX_HEMATOCRIT)
        fluid_change, fluid_urine, fluid_sequestration = self.crystalloid.step(
            dt, map_prev, self.patient.age, self.patient.renal_clearance_factor * cfg.crystalloid_renal_scale,
            self.blood_volume - minimum_blood_volume,
            urinary_elimination=cfg.vol_clearance is None,
            circulation=circulation,
        )
        self.blood_volume += fluid_change
        self.fluids.shift_water(fluid_change + fluid_urine)
        self.fluids.lose_urine(fluid_urine)
        self.total_urine_out_ml += fluid_urine
        self.total_third_space_ml += fluid_sequestration
        if cfg.vol_clearance is not None:
            urine_ml_min = max(0.0, float(cfg.vol_clearance))
        else:
            urine_ml_min = cfg.uop_ml_kg_hr * self.patient.weight / 60.0
        urine_out_ml = min(
            urine_ml_min * renal_factor * dt_min,
            max(0.0, self.blood_volume - minimum_blood_volume),
        )
        blood_volume = self.blood_volume - urine_out_ml
        if urine_out_ml > 0:
            self.fluids.lose_urine(urine_out_ml)
            plasma_volume = self.blood_volume - self.red_cell_volume_ml
            self.crystalloid.central_ml *= max(0.0, 1.0 - urine_out_ml / plasma_volume)
            self.total_urine_out_ml += urine_out_ml
        leak_rate_per_s = (cfg.sepsis_leak_fraction_per_hr * sepsis_sev / 3600.0
                           + cfg.anaphylaxis_leak_fraction_per_min * anaph_sev / 60.0)
        if leak_rate_per_s > 0.0:
            # Capillary leak loses plasma, not red cells.
            leak_ml = blood_volume * -math.expm1(-leak_rate_per_s * dt)
            available = max(0.0, blood_volume - minimum_blood_volume)
            actual_leak = min(leak_ml, available)
            if actual_leak > 0:
                self.fluids.shift_water(-actual_leak, protein_fraction=1.0)
                self.crystalloid.central_ml *= max(0.0, 1.0 - actual_leak / (blood_volume - self.red_cell_volume_ml))
                blood_volume -= actual_leak
                self.total_third_space_ml += actual_leak
                self._protein_third_space_ml += actual_leak
        if self.total_third_space_ml > 0 and cfg.third_space_refill_tau_hr > 0:
            intravascular_excess = blood_volume - self.blood_volume_0
            shortfall = cfg.interstitial_refill_fraction * (
                intravascular_excess + self.total_third_space_ml
            ) - intravascular_excess
            frac = -math.expm1(-dt / (cfg.third_space_refill_tau_hr * 3600.0))
            refill_ml = min(self.total_third_space_ml, shortfall * frac)
            if refill_ml > 0:
                prot_frac = clamp01(self._protein_third_space_ml / self.total_third_space_ml)
                self.fluids.shift_water(refill_ml, protein_fraction=prot_frac)
                self._protein_third_space_ml = max(0.0, self._protein_third_space_ml - refill_ml * prot_frac)
                self.total_third_space_ml -= refill_ml
                blood_volume += refill_ml

        vol_deficit_ml = self.blood_volume_0 - blood_volume
        if cfg.vol_clearance is None and leak_rate_per_s <= 0.0 and vol_deficit_ml > 50.0:
            # Falling capillary hydrostatic pressure recruits protein-free interstitial
            # water (Starling transcapillary refill, ~200-270 mL/h peak).
            interstitial_avail_ml = max(0.0, (self.fluids.interstitial_l - 6.0) * 1000.0)
            refill_drive = clamp01((vol_deficit_ml - 50.0) / 950.0) * (
                0.45 + 0.55 * clamp01((85.0 - map_prev) / 35.0)
            )
            autotransfusion_ml = min(interstitial_avail_ml, vol_deficit_ml * 0.65, 4.5 * refill_drive * dt_min)
            if autotransfusion_ml > 0.0:
                self.fluids.shift_water(autotransfusion_ml)
                blood_volume += autotransfusion_ml

        # Lung water filters from blood above the LAP threshold and returns over hours;
        # once lymphatic clearance saturates, filtration accelerates steeply.
        eff_lap = self._effective_lap()
        lap_excess = max(0.0, eff_lap - self._lung_water_threshold)
        eff_lap_excess = lap_excess + 0.14 * max(0.0, lap_excess - 6.0) ** 2
        pulm_flow_gate = clamp01(co_prev / max(0.3, 0.20 * base_co))
        filtration_ml = min(
            cfg.lung_water_filtration * self._pbw * eff_lap_excess * pulm_flow_gate * dt_min,
            max(0.0, blood_volume - minimum_blood_volume),
        )
        lung_clearance_ml = self.lung_water_ml * -math.expm1(-dt / (cfg.lung_water_clearance_tau_hr * 3600.0))
        self.lung_water_ml += filtration_ml - lung_clearance_ml
        self.fluids.shift_water(lung_clearance_ml - filtration_ml)
        if filtration_ml > 0.0:
            self.crystalloid.central_ml *= 1.0 - filtration_ml / (blood_volume - self.red_cell_volume_ml)
        blood_volume += lung_clearance_ml - filtration_ml
        self.blood_volume = max(BLOOD_VOLUME_MIN, blood_volume)
        self._update_hb_conc()

        self._update_anesthetic_factors(ce_prop, self.ce_opioid_cv, self.ce_sevo, vagal)
        if ce_midazolam > 0.0 or ce_etomidate > 0.0:
            midaz_occ = hill_function(max(0.0, ce_midazolam), 120.0, 1.2)
            opioid_midaz = hill_function(self.ce_opioid_cv, 1.2, 1.0)
            etom_occ = hill_function(max(0.0, ce_etomidate), 0.5, 1.5)
            self._drug_tpr *= (1.0 - (0.14 + 0.24 * opioid_midaz) * midaz_occ) * (1.0 - 0.025 * etom_occ)
            self._drug_sv *= 1.0 - (0.06 + 0.08 * opioid_midaz) * midaz_occ
        if mac_n2o > 0.0:
            n2o_level = clamp(mac_n2o, 0.0, 1.5)
            self._drug_tpr *= 1.0 + 0.09 * n2o_level * resting_tone * alpha_response
            self._drug_sv *= 1.0 - 0.08 * n2o_level
        ce_amio_vasc = max(0.0, ce_amiodarone_vasc)
        if ce_amio_vasc > 0.0 or self._amiodarone_vasc_adapted != 0.0 or amiodarone > 0.0 or amiodarone_hr > 0.0:
            self._amiodarone_vasc_adapted += (ce_amio_vasc - self._amiodarone_vasc_adapted) * (
                -math.expm1(-60.0 * delivery_time / cfg.amiodarone_vasc_adaptation_s)
            )
            amiodarone_vasc = hill_function(
                max(0.0, ce_amio_vasc - self._amiodarone_vasc_adapted),
                cfg.amiodarone_vasc_c50, cfg.amiodarone_vasc_gamma,
            )
            self._amiodarone_ees_factor = 1.0 - cfg.amiodarone_ees_reduction * amiodarone
            self._drug_hr *= 1.0 - cfg.amiodarone_hr_reduction * amiodarone_hr
            self._drug_sv *= 1.0 - cfg.amiodarone_sv_reduction * amiodarone
            self._drug_tpr *= 1.0 - cfg.amiodarone_tpr_reduction * amiodarone_vasc
        else:
            self._amiodarone_ees_factor = 1.0

        # Positive intrathoracic pressure reduces venous return; spontaneous
        # negative pressure modestly augments it.
        self.f_preload_pit = self._pit_preload_factor(pit)

        epi_active = (
            ce_epi > 0.0
            or self._epi_pressor_ce != 0.0
            or self._epi_high_pressor_ce != 0.0
            or self._epi_beta2_effect != 0.0
            or self._epi_chrono_effect != 0.0
        )
        if epi_active:
            self._epi_pressor_ce += (max(0.0, ce_epi) - self._epi_pressor_ce) * (
                -math.expm1(-dt / cfg.epi_tau_pressor_s)
            )
            self._epi_high_pressor_ce += (max(0.0, ce_epi) - self._epi_high_pressor_ce) * (
                -math.expm1(-dt / cfg.epi_tau_alpha_high_s)
            )
            self._epi_venous_alpha = hill_function(
                self._epi_pressor_ce / alpha_dr, cfg.epi_c50_alpha * cfg.epi_venous_potency_ratio, cfg.epi_gamma_alpha
            )
        else:
            self._epi_venous_alpha = 0.0
        pressor_gain = 1.0 - clamp01(cfg.sepsis_pressor_resistance * sepsis_sev)
        nore_scale = self.acidosis_factor / agonist_shift
        self._update_venous_drug_effects(
            ce_prop, ce_phenyl / alpha_dr, pressor_gain, ce_nore_raw,
            nore_scale=nore_scale / alpha_dr,
        )

        f_frank_starling = self._update_pulmonary_coupling(
            dt=dt,
            pao2=pao2,
            peep_cmH2O=peep_cmH2O,
            f_preload_pit=self.f_preload_pit,
            sepsis_sev=sepsis_sev,
            pit=pit,
        )

        # Chemoreflex: hypercapnia raises HR and TPR; hypoxemia raises HR.
        e_co2 = max(0.0, (paco2 - self._paco2_set) / self._paco2_set)
        e_o2 = max(0.0, (cfg.pao2_set - pao2) / cfg.pao2_set)
        chemo_hr_boost = (cfg.g_hr_co2 * e_co2 + cfg.g_hr_o2 * e_o2) * sympathetic_hr_response
        chemo_tpr_factor = 1.0 + cfg.k_tpr_co2 * e_co2

        ce_ket_hr = ce_ketamine if ce_ketamine_hr is None else ce_ketamine_hr
        if ce_ketamine > 0.0 or ce_ket_hr > 0.0:
            vol_deficit_ket = max(0.0, (self.blood_volume_0 - self.blood_volume) / max(1.0, self.blood_volume_0))
            ket_exhaustion = max(
                clamp01((vol_deficit_ket - 0.28) / 0.12),
                clamp01((sepsis_sev - 0.30) / 0.35),
                clamp01((45.0 - map_prev) / 20.0),
                clamp01(self.coronary.energy_failure / 0.40),
                clamp01((7.15 - ph) / 0.20),
                clamp01(1.0 - circulation),
            )
            ket_scale = resting_tone * (1.0 - ket_exhaustion)
            ket_raw = hill_function(max(0.0, ce_ketamine), cfg.ketamine_c50, cfg.ketamine_gamma)
            ketamine = ket_raw * ket_scale
            ketamine_hr = (
                ketamine
                if ce_ketamine_hr is None
                else hill_function(max(0.0, ce_ket_hr), cfg.ketamine_c50, cfg.ketamine_gamma) * ket_scale
            )
        else:
            ket_exhaustion = ket_raw = ketamine = ketamine_hr = 0.0
        sinus_est = clamp(self._calc_hr() + crisis_hr + dist_hr, HR_MIN, HR_MAX)

        # Fast baroreflex. Propofol depresses both limbs (Sato 2005). Anesthetic
        # ketamine depresses gain as much (Akine 2001), whereas analgesic doses
        # do not (Huang 2020); a calibrated upward shift preserves its pressor
        # rise when vagal block or pacing engages vascular withdrawal. Opioids
        # raise vagal tone, suppressing reflex tachycardia but not slowing.
        reflex_depth = clamp01(anesthetic_depth + max(0.0, ce_ket_hr) / KETAMINE_MAC_EQUIVALENT)
        tachycardia_depth = clamp01(reflex_depth + self.ce_opioid_cv / cfg.opioid_reflex_ce)
        sensed_error = map_prev - self._stim_map - self._baro_setpoint
        baro_vasc = 0.0
        rhythm = self._rhythm_type
        if sensed_error > 0.0:
            sensed_error = max(0.0, sensed_error - cfg.ketamine_baro_shift * ketamine_hr)
            reflex_intact = 1.0 - cfg.baro_anesthetic_depression * reflex_depth
            baro_gain = cfg.baro_gain_brady * (1.0 - vagal) * self._age_vagal_baroreflex * reflex_intact
            if (rhythm is RhythmType.SINUS or rhythm not in PULSELESS_RHYTHMS) and not self._compressions:
                rhythm_holds_rate_up = self._rhythm_rate(sinus_est - 1.0) > sinus_est - 1.0
                rhythm_slowing_blocked = (
                    clamp01(1.0 - (self._rhythm_rate(sinus_est) - self._rhythm_rate(sinus_est - 1.0)))
                    if rhythm_holds_rate_up
                    else 0.0
                )
                adrenergic_override = clamp01((self._adrenergic_chrono - 55.0) / 30.0) * clamp01(
                    (sensed_error - 65.0) / 30.0
                )
                blocked_share = max(rhythm_slowing_blocked, vagal, adrenergic_override)
                max_drop = cfg.baro_vasc_max * resting_tone * alpha_response
                if blocked_share > 0.0 and max_drop > 1e-6 and cfg.baro_gain_vasc > 0.0 and sensed_error > 0.0:
                    vasc_gain = cfg.baro_gain_vasc * reflex_intact
                    baro_vasc = max_drop * math.tanh(vasc_gain * sensed_error / max_drop) * blocked_share
        else:
            baro_gain = cfg.baro_gain_tachy * sympathetic_hr_response
            baro_gain *= 1.0 - cfg.baro_anesthetic_depression * tachycardia_depth
        baro_hr = clamp(-baro_gain * sensed_error, -cfg.baro_max_hr_change, cfg.baro_max_hr_change)
        # Baroreceptor firing ceases below ~50 mmHg, so arrest does not reset the setpoint toward zero.
        reset_target = max(min(50.0, self._baro_setpoint), map_prev - self._stim_map)
        awake_homeostatic_return = (
            anesthetic_depth < 0.05
            and abs(reset_target - self.patient.baseline_map) < 2.5
            and (reset_target - self._baro_setpoint) * (self.patient.baseline_map - self._baro_setpoint) > 0.0
        )
        reset_tau = min(cfg.baro_reset_tau_s, 450.0) if awake_homeostatic_return else cfg.baro_reset_tau_s
        self._baro_setpoint += (reset_target - self._baro_setpoint) * -math.expm1(-dt / reset_tau)

        # Recovery follows reoxygenation; the coronary model applies the perfusion limit.
        hypoxia_target = clamp01(
            (cfg.hypoxia_sao2_onset - sao2) / (cfg.hypoxia_sao2_onset - cfg.hypoxia_sao2_full)
        )
        if hypoxia_target > self.myocardial_hypoxia:
            hypoxia_tau = cfg.hypoxia_tau_on_s
        else:
            hypoxia_tau = cfg.hypoxia_tau_off_s
        self.myocardial_hypoxia += (hypoxia_target - self.myocardial_hypoxia) * -math.expm1(-dt / hypoxia_tau)

        hemorrhage_hr_mult, hemorrhage_tpr_mult = self._calc_hemorrhage_response()
        # Anesthesia already depresses the fast and slow reflex tachycardia, so
        # the volume-loss drive keeps full strength; HR still rises under TIVA (Kurita 2011).
        hemorrhage_hr_mult = 1.0 + (hemorrhage_hr_mult - 1.0) * sympathetic_hr_response
        hemorrhage_tpr_mult = 1.0 + (hemorrhage_tpr_mult - 1.0) * alpha_response

        # Vascular relaxation develops alongside the cardiac response. An
        # instantaneous beta-2 effect otherwise produces a spurious early dip.
        if epi_active:
            beta2_target = cfg.epi_emax_svr_beta * hill_function(ce_epi / beta2_dr, cfg.epi_c50_beta2, 1.0)
            self._epi_beta2_effect += (beta2_target - self._epi_beta2_effect) * (
                -math.expm1(-dt / cfg.epi_tau_beta2_s)
            )
            epi_delta_hr, epi_sv_factor, epi_svr_factor = self._calc_epi_effects(
                ce_epi, self._epi_pressor_ce, self._epi_high_pressor_ce, beta1_dr, alpha_dr
            )
            self._epi_pulse_factor = 1.0 + cfg.epi_pulse_gain * (epi_sv_factor - 1.0)
            self._epi_chrono_effect += (epi_delta_hr - self._epi_chrono_effect) * (
                -math.expm1(-dt / cfg.epi_tau_hr_s)
            )
        else:
            epi_sv_factor = epi_svr_factor = self._epi_pulse_factor = 1.0
        nore_delta_hr, nore_sv_factor, nore_svr_factor = self._calc_nore_effects(
            ce_nore_raw, beta1_dr, alpha_dr, anesthetic_depth, nore_scale=nore_scale,
        )
        if ce_ephedrine_raw > 0.0 or self._ephedrine_tolerance_ce != 0.0:
            self._ephedrine_tolerance_ce += (ce_ephedrine_raw - self._ephedrine_tolerance_ce) * (
                -math.expm1(-math.log(2.0) * dt_min / cfg.ephedrine_tolerance_half_min)
            )
            ephedrine_raw = ce_ephedrine_raw / agonist_shift
            ephedrine = max(0.0, ce_ephedrine) / agonist_shift
            tol_raw = self._ephedrine_tolerance_ce / agonist_shift
            tol_alpha = cfg.ephedrine_tolerance_max * hill_function(
                tol_raw / alpha_dr,
                cfg.ephedrine_tolerance_c50, cfg.ephedrine_tolerance_gamma,
            )
            tol_beta = cfg.ephedrine_tolerance_max * hill_function(
                tol_raw / beta1_dr,
                cfg.ephedrine_tolerance_c50, cfg.ephedrine_tolerance_gamma,
            )
            ephedrine_alpha_effect = self.acidosis_factor * max(
                0.0,
                hill_function(ephedrine_raw / alpha_dr, cfg.ephedrine_c50, cfg.ephedrine_gamma) - tol_alpha,
            )
            ephedrine_beta_effect = self.acidosis_factor * max(
                0.0,
                hill_function(ephedrine_raw / beta1_dr, cfg.ephedrine_c50, cfg.ephedrine_gamma) - tol_beta,
            )
            ephedrine_svr = 1.0 + cfg.ephedrine_emax_svr * ephedrine_alpha_effect
            ephedrine_sv = 1.0 + cfg.ephedrine_emax_sv * ephedrine_beta_effect
            ephedrine_chrono_gate = (
                self._age_chronotropy * (1.0 - cfg.amiodarone_beta_reflex_block * amiodarone_hr)
            )
            ephedrine_beta_conc = ephedrine / beta1_dr
            sustained_hr_max = 0.40
            sustained_hr_term = (
                sustained_hr_max * ephedrine_beta_conc / (ephedrine_beta_conc + sustained_hr_max / cfg.ephedrine_hr_slope)
                if cfg.ephedrine_hr_slope > 0.0 and ephedrine_beta_conc > 0.0
                else 0.0
            )
            ephedrine_hr = (
                sustained_hr_term + cfg.ephedrine_emax_hr * ephedrine_beta_effect
            ) * self.base_hr * ephedrine_chrono_gate
        else:
            ephedrine_svr = ephedrine_sv = 1.0
            ephedrine_hr = 0.0
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
        catechol_chrono = self._epi_chrono_effect + nore_delta_hr + dobu_delta_hr + ephedrine_hr
        catechol_sv = epi_sv_factor * nore_sv_factor * dobu_sv_factor * ephedrine_sv
        # SVR, SV, and HR adrenoceptor responses share septic blunting (Bellissant 2000; Hernández 2013).
        catechol_svr_factor = epi_svr_factor * nore_svr_factor * phenyl_svr_factor * ephedrine_svr
        if pressor_gain < 1.0:
            catechol_svr_factor = 1.0 + (catechol_svr_factor - 1.0) * pressor_gain
            catechol_sv = 1.0 + (catechol_sv - 1.0) * pressor_gain
            catechol_chrono *= pressor_gain
        self._adrenergic_chrono = max(0.0, catechol_chrono + mil_delta_hr)

        # Blockade removes resting sympathetic tone; vagal block removes vagal
        # tone; ketamine adds sympathetic tone.
        combined_svr_factor = (
            catechol_svr_factor *
            vaso_svr_factor *
            dobu_svr_factor *
            mil_svr_factor *
            (1.0 + cfg.ketamine_emax_svr * ketamine * alpha_response) *
            (1.0 - 0.15 * ket_raw * ket_exhaustion)
        )
        agonist_sv = catechol_sv * mil_sv_factor
        ket_sv = 1.0 + cfg.ketamine_emax_sv * ketamine * beta_response
        ket_loss = 1.0 - 0.28 * ket_raw * ket_exhaustion
        combined_sv_factor = (
            agonist_sv
            * (1.0 - cfg.beta_block_sv * beta_occ * resting_tone)
            * ket_sv
            * ket_loss
            / (1.0 + self._phenyl_afterload_gain * (phenyl_svr_factor - 1.0) * pressor_gain)
        )
        combined_delta_hr = (
            catechol_chrono + vaso_delta_hr + mil_delta_hr
            + (cfg.ketamine_emax_hr * ketamine_hr + cfg.sepsis_hr_increase * sepsis_sev) * sympathetic_hr_response
            - cfg.beta_block_hr * beta_occ * resting_tone * self.hr_star
            + vagal_hr
        )
        self.vasopressor_sv_factor = combined_sv_factor
        self._phenyl_svr_factor = 1.0 + (phenyl_svr_factor - 1.0) * pressor_gain
        self._non_phenyl_sv_factor = agonist_sv * ket_sv * ket_loss
        self.delta_tpr_vasopressors = self.base_tpr * (
            combined_svr_factor - 1.0 - cfg.alpha_block_tpr * alpha_occ * resting_tone
        )

        if (rhythm is not RhythmType.SINUS and rhythm in RHYTHM_SV_FACTOR) or self._compressions:
            self._unstim_rate = 0.0
        else:
            rhythm_hr = self._rhythm_rate(sinus_est)
            paced_excess = max(0.0, rhythm_hr - sinus_est)
            beating_hr = self._effective_beating_hr(rhythm_hr)
            actual_tachy = max(0.0, beating_hr - self.base_hr)
            self._unstim_rate = min(max(0.0, vagal_hr + paced_excess), actual_tachy) / self.base_hr
        if cfg.unstim_sv_tau_s > 0.0 and dt > 0.0:
            unload_tau = (
                cfg.unstim_sv_tau_s
                if self._unstim_rate >= self._unstim_rate_unloaded
                else min(15.0, cfg.unstim_sv_tau_s)
            )
            self._unstim_rate_unloaded += (
                self._unstim_rate - self._unstim_rate_unloaded
            ) * -math.expm1(-dt / unload_tau)
        elif cfg.unstim_sv_tau_s <= 0.0:
            self._unstim_rate_unloaded = self._unstim_rate

        # Feedback senses the same circulation as the outputs, including
        # rhythm, hypoxia, and preload. Nociception resets it as it does the
        # fast baroreflex, so sustained surgery keeps MAP raised. Without the
        # heart's own output it holds its last value instead of winding up during arrest.
        self.dist_hr, self.dist_sv, self.dist_svr = dist_hr, dist_sv, dist_svr
        self.crisis_hr, self.crisis_svr = crisis_hr, crisis_svr
        sepsis_drop_wood = cfg.sepsis_svr_drop_wood / self.failure.output_factor
        distributive_svr_drop = (sepsis_drop_wood * sepsis_sev +
                                 cfg.anaphylaxis_svr_drop_wood * anaph_sev)
        distributive_tpr_offset = -distributive_svr_drop / 1000.0
        _, _, _, fb_co, fb_map, _ = self._compute_intrinsic(
            f_frank_starling,
            sepsis_sev,
            distributive_tpr_offset,
            self._calc_hr(),
        )
        if fb_co > 0.0:
            rmap = clamp((fb_map - self._stim_map) * self._inv_base_rmap_denom, 0.1, 5.0)
            self._rmap_fb = rmap ** cfg.fb
        rmap_fb = self._rmap_fb

        # Thermoregulatory vasoconstriction below 36.5 °C. Anesthesia lowers the
        # threshold by up to 2.5 °C at 1 MAC or propofol 4 mcg/mL (Sessler 2000),
        # whereas 300 ng/mL midazolam lowers it by 0.8 °C (37.1 to 36.3 °C; Kurz 1995).
        thermo_depth = clamp01(
            self.ce_sevo
            + cp_prop / 4.0
            + max(0.0, ce_ketamine) / KETAMINE_MAC_EQUIVALENT
            + adjunct_etom
            + 0.40 * adjunct_midaz
        )
        base_vc_threshold = min(36.5, self.patient.baseline_temp - 0.5)
        vasoconstriction_threshold = base_vc_threshold - 2.5 * thermo_depth
        thermo_tpr_mult = self._thermo_tpr_mult(temp_c, vasoconstriction_threshold)

        # Turnover equations; inotropy and preload act on output SV only.
        # Terminal anoxia and receptor blockade prevent compensatory sympathetic wind-up.
        anoxic_intact = (1.0 - self.myocardial_hypoxia) * (1.0 - self.coronary.energy_failure)
        symp_vasc_gate = anoxic_intact * alpha_response
        tpr_fb = 1.0 + (rmap_fb - 1.0) * symp_vasc_gate if rmap_fb > 1.0 else rmap_fb
        sv_fb = 1.0 + (rmap_fb - 1.0) * anoxic_intact * beta_response if rmap_fb > 1.0 else rmap_fb
        eff_chemo_tpr = 1.0 + (chemo_tpr_factor - 1.0) * symp_vasc_gate
        eff_thermo_tpr = 1.0 + (thermo_tpr_mult - 1.0) * symp_vasc_gate
        eff_hemorrhage_tpr = 1.0 + (hemorrhage_tpr_mult - 1.0) * anoxic_intact
        tpr_production = self.kin_tpr * tpr_fb * eff_chemo_tpr * eff_hemorrhage_tpr * eff_thermo_tpr
        d_tpr_tone = tpr_production - cfg.kout * self.tpr_tone
        d_sv_tone = self.kin_sv * sv_fb - cfg.kout * self.sv_tone
        eff_tachy_depth = max(tachycardia_depth, clamp01((32.0 - temp_c) / 6.0))
        hr_feedback = self._hr_feedback(rmap_fb, eff_tachy_depth)
        eff_hemorrhage_hr = 1.0 + (hemorrhage_hr_mult - 1.0) * anoxic_intact
        d_hr_tone = self.kin_hr * hr_feedback * eff_hemorrhage_hr - cfg.kout * self.hr_tone

        alpha_fast = -math.expm1(-dt / cfg.tau_hr_fast) if dt > 0.0 else 0.0
        self.smoothed_chemo_hr += (chemo_hr_boost - self.smoothed_chemo_hr) * alpha_fast
        self.smoothed_baro_hr += (baro_hr - self.smoothed_baro_hr) * alpha_fast
        self.smoothed_baro_vasc += (baro_vasc - self.smoothed_baro_vasc) * alpha_fast
        self.smoothed_epi_hr += (combined_delta_hr - self.smoothed_epi_hr) * alpha_fast

        self.tpr_tone += d_tpr_tone * dt_min
        self.sv_tone += d_sv_tone * dt_min
        self.hr_tone += d_hr_tone * dt_min
        tde_decay = math.exp(-cfg.k_drift * dt_min)
        self.tde_hr *= tde_decay
        self.tde_sv *= tde_decay

        hr_base_for_state = self._calc_hr()
        computed_state = self._compute_state(
            preload_sv_factor=f_frank_starling,
            sepsis_sev=sepsis_sev,
            anaph_sev=anaph_sev,
            distributive_tpr_offset=distributive_tpr_offset,
            hr_base=hr_base_for_state,
            _update_stim=True,
        )
        self._last_intrinsic_co = computed_state.intrinsic_co
        self._prev_map = computed_state.perfusion_map
        return computed_state

    def initialize_steady_state(
        self,
        cp_prop: float,
        cp_remi: float,
        ce_nore: float,
        mac_sevo: float = 0.0,
        remifentanil_cp: float | None = None,
        *,
        pit: float = -2.0,
        peep_cmH2O: float | None = None,
    ) -> HemoState:
        """Seed tone, drug effects, and the baroreflex at fixed concentrations."""
        cn = max(0.0, ce_nore)
        self.crisis_hr = self.crisis_svr = 0.0
        self._pulmonary_congestion_ml = 0.0
        self._pulm_arrest_blend = 0.0
        self.ce_remi_vagal = max(0.0, cp_remi if remifentanil_cp is None else remifentanil_cp)
        self._sympathetic_venous_depth = clamp01(max(0.0, cp_prop) / self.config.propofol_sympathetic_cp)
        self._epi_venous_alpha = 0.0
        self._amiodarone_ees_factor = 1.0
        self._amiodarone_vasc_adapted = 0.0
        ce_prop = max(0.0, cp_prop) * self._prop_cv_potency
        self._update_anesthetic_factors(ce_prop, cp_remi, mac_sevo)
        anesthetic_depth = clamp01(mac_sevo + cp_prop / 4.0)
        tachycardia_depth = clamp01(anesthetic_depth + max(0.0, cp_remi) / self.config.opioid_reflex_ce)
        nore_delta_hr, nore_sv_factor, nore_svr_factor = self._calc_nore_effects(cn, anesthetic_depth=anesthetic_depth)
        # Venous return at the drug's stressed volume, relative to the reference
        # that keeps Su's response, sets the steady-state preload.
        self._update_venous_drug_effects(ce_prop, 0.0, 1.0, cn)

        self.f_preload_pit = self._pit_preload_factor(pit)
        self._rv_pit_preload = self.f_preload_pit
        self._lv_pit_preload = self.f_preload_pit
        pvr_factor = self._calc_pvr_factor(95.0, peep_cmH2O)

        def flow_factor(mcfp: float) -> float:
            rap = self.config.rap_baseline * min(1.0, mcfp / self.mcfp_0)
            return (mcfp - rap) / self.venous_return_resistance * self._inv_base_co_l_min

        self._position_preload = self._supine_preload if self.position == "supine" else 1.0
        self._last_mcfp = self._calc_stressed_volume() / self.cv
        self._pulm_reference_factor = flow_factor(self._venous_reference_mcfp)
        self._pulm_flow_factor = flow_factor(self._last_mcfp) * (
            pvr_factor ** (-self.config.pvr_flow_exponent)
        )
        self._last_venous_preload = self._pulm_flow_factor / self._pulm_reference_factor * self._position_preload
        f_preload = self._last_venous_preload * self.f_preload_pit
        preload_sv = self._frank_starling(f_preload)
        last_lap = self._lap(f_preload)
        lap_factor = (
            1.0 - 0.22 * clamp01((last_lap - 30.0) / 18.0)
            if self.patient.heart_failure != "none" and last_lap > 30.0
            else 1.0
        )

        base_temp = self.patient.baseline_temp
        eff_tachy_depth = max(tachycardia_depth, clamp01((32.0 - base_temp) / 6.0))
        hypo_hr, hypo_sv = self._hypothermia_factors(base_temp)
        base_vc_threshold = min(36.5, base_temp - 0.5)
        vasoconstriction_threshold = base_vc_threshold - 2.5 * anesthetic_depth
        thermo_tpr_mult = self._thermo_tpr_mult(base_temp, vasoconstriction_threshold)
        coronary_keep = 1.0 - 0.50 * self.coronary.ischemia

        def residual(z):
            z_fb = z ** self.config.fb
            sinus_z = clamp(
                self.base_hr * self._hr_feedback(z_fb, eff_tachy_depth) * self._drug_hr + nore_delta_hr,
                HR_MIN,
                HR_MAX,
            )
            hr_z = sinus_z * hypo_hr
            tpr_z = max(0.006, self.base_tpr * (z_fb * thermo_tpr_mult * self._drug_tpr + nore_svr_factor - 1.0))
            sv_z = (
                self.base_sv
                * z_fb
                * self._drug_sv
                * self._hr_sv_factor(hr_z)
                * nore_sv_factor
                * preload_sv
                * lap_factor
                * hypo_sv
                * coronary_keep
            )
            sv_z = self._apply_afterload_coupling(sv_z, tpr_z, hr_z, nore_sv_factor)
            sv_z = self._cap_sv_by_venous_return(sv_z, tpr_z, hr_z, nore_sv_factor)
            return hr_z * sv_z * tpr_z * self._inv_base_rmap_denom - z

        try:
            sol = root_scalar(residual, bracket=[0.01, 5.0], method='brentq')
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError("Unable to solve hemodynamic steady state") from exc
        if not sol.converged:
            raise RuntimeError("Hemodynamic steady-state solver did not converge")
        z_ss = sol.root

        z_fb = z_ss ** self.config.fb
        self.hr_tone = self.base_hr * self._hr_feedback(z_fb, eff_tachy_depth)
        self.sv_tone = self.base_sv * z_fb
        self.tpr_tone = self.base_tpr * z_fb * thermo_tpr_mult
        self.ce_prop_cv, self.ce_opioid_cv = max(0.0, cp_prop), max(0.0, cp_remi)
        self.tde_hr = 0
        self.tde_sv = 0
        self.ce_sevo = mac_sevo

        self._epi_pressor_ce = self._epi_high_pressor_ce = self._epi_chrono_effect = self._epi_beta2_effect = 0.0
        self._adrenergic_chrono = max(0.0, nore_delta_hr)
        self.smoothed_chemo_hr = self.smoothed_baro_hr = self.smoothed_baro_vasc = 0.0
        self._unstim_rate = self._unstim_rate_unloaded = 0.0
        self.smoothed_epi_hr = nore_delta_hr
        self.vasopressor_sv_factor = nore_sv_factor
        self._phenyl_svr_factor = 1.0
        self._non_phenyl_sv_factor = nore_sv_factor
        self.delta_tpr_vasopressors = self.base_tpr * (nore_svr_factor - 1.0)

        state = self.step(
            0.0, cp_prop, cp_remi, ce_nore, pit, self._paco2_set, 95.0, 0, 0, 0,
            mac_sevo=mac_sevo, remifentanil_cp=remifentanil_cp,
            temp_c=base_temp, peep_cmH2O=peep_cmH2O, ph=self.patient.pregnancy.ph,
        )
        self._baro_setpoint = state.map
        return state
