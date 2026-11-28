"""Local anesthetic systemic toxicity (lidocaine, bupivacaine, ropivacaine),
lipid emulsion rescue, ASRA resuscitation pharmacology, and opioid rigidity /
reversal dynamics.

Concentrations are mcg/mL for local anesthetics and ng/mL for opioids,
naloxone, flumazenil, and epinephrine; vasopressin is mU/L.

Sources:
- Lidocaine CNS/CV thresholds: Foldes 1960; Jorfeldt 1968; Tucker and Boas 1971;
  Armstrong 1988.
- Bupivacaine and ropivacaine comparative CNS/CV potency, CC/CNS ratios, and
  pregnancy sensitivity: Knudsen 1997 (human volunteers); Feldman 1989 (beagles);
  Reiz 1989 (porcine LAD); Morishima 1985 and Santos 1995 (pregnant vs
  nonpregnant ewes; note Santos 1995 found no pregnancy shift in a blinded design).
- Slow bupivacaine cardiac Na+-channel recovery and rate-dependent conduction
  block: Clarkson and Hondeghem 1985 (guinea pig ventricular muscle).
- Lipophilicity-scaled lipid emulsion partitioning, acidosis attenuation, shuttle,
  and cardiotonic rescue: Mazoit 2009 (in vitro); Zausig 2009 (rat heart);
  Litonius 2012, Heinonen 2015, and Dureau 2016 (human volunteers); Kuo and Akpa
  2013 (human PBPK); Fettiplace 2014 (rat bupivacaine); Neal 2018 (ASRA).
- ASRA LAST resuscitation pharmacology: Hiller 2009 (rat high-dose epinephrine
  threshold >= 10 mcg/kg); Di Gregorio 2009 (rat vasopressin ROSC failure,
  acidosis, and pulmonary edema); Zaballos 2019 (porcine NaHCO3 QRS relief).
- Opioid rigidity and glottic closure: Benthuysen 1986; Bennett 1997.
- Competitive opioid and benzodiazepine antagonism: Yassen 2007b; Moss 2020;
  Klotz and Kanto 1988.
"""

from __future__ import annotations

import math

from anasim.core.utils import clamp, clamp01, hill_function

# Sodium-channel block at which no beats are conducted.
SILENT_BLOCK = 0.99

# Competitive antagonist equilibrium dissociation constants (ng/mL at effect site).
# Naloxone Kb is anchored between Moss 2020 (Ki 1.115 nM ~ 0.37 ng/mL free),
# Kaufman 1981 (0.55 mg -> morphine dose ratio ~5, cited in Dowling 2008), and
# Yassen 2007b (KD 1.87 ng/mL vs high-affinity buprenorphine).
NALOXONE_KB_NG_ML = 0.85
# Flumazenil Kb calibrated to Klotz and Kanto 1988 (10-20 ng/mL plasma reverses
# midazolam sedation; 0.2 mg IV wakes light-moderate sedation within 1-2 min).
FLUMAZENIL_KB_NG_ML = 1.8

# Comparative local anesthetic toxic potency relative to lidocaine (mcg/mL basis).
# Lidocaine = 1.0 across all endpoints so lidocaine-only kinetics are unchanged.
_BUPI_CNS_POTENCY = 2.0          # Seizures ~2.5-4.75 mcg/mL (Knudsen 1997; Feldman 1989; Morishima 1985)
_BUPI_CONTRACTILITY_POTENCY = 4.2  # 4:1 vs lidocaine (Reiz 1989)
_BUPI_CONDUCTION_POTENCY = 6.2   # Narrow CC/CNS ~1.4-1.8 with rate-dependent block (Clarkson 1985; Reiz 1989)
_BUPI_ARRHYTHMIA_POTENCY = 7.0   # High VT/VF incidence (Feldman 1989; Reiz 1989)

_ROPI_CNS_POTENCY = 1.55         # Higher free/total CNS threshold than bupi (Knudsen 1997; Feldman 1989)
_ROPI_CONTRACTILITY_POTENCY = 3.0  # 3:1 vs lidocaine (Reiz 1989)
_ROPI_CONDUCTION_POTENCY = 3.2   # ~2x wider electrophysiologic margin vs bupi (Reiz 1989; Knudsen 1997)
_ROPI_ARRHYTHMIA_POTENCY = 3.4   # Lower malignant arrhythmia propensity vs bupi (Feldman 1989)


def _response(concentration: float, onset: float, full: float) -> float:
    fraction = clamp01((concentration - onset) / (full - onset))
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _diastolic_block_rate_factor(hr_bpm: float, tau_recovery_s: float) -> float:
    """Use-dependent cardiac Na+-channel block multiplier vs 60 bpm (Clarkson 1985).

    Higher heart rates shorten diastolic recovery time T_d relative to channel
    unbinding tau_recovery_s, raising steady-state block for slow-unbinding
    bupivacaine (tau ~ 1.56 s) while leaving fast-unbinding lidocaine (~0.15 s)
    essentially flat at normal rates.
    """
    hr = clamp(hr_bpm, 25.0, 220.0)
    rr_s = 60.0 / hr
    systole_s = min(0.32, 0.35 * math.sqrt(rr_s))
    diastole_s = max(0.06, rr_s - systole_s)
    unrecovered = math.exp(-diastole_s / tau_recovery_s)
    ref_unrecovered = math.exp(-(1.0 - 0.32) / tau_recovery_s)
    # Map channel residual fraction into a bounded conduction potency multiplier (1.0 at 60 bpm).
    return clamp(0.65 + 0.35 * (unrecovered / max(1e-6, ref_unrecovered)), 0.65, 1.55)


class LocalAnestheticToxicity:
    LIPID_HALF_TIME_S = 1200.0
    LIPID_DELIVERY_TAU_S = 60.0

    def __init__(self, weight: float, *, pregnant: bool = False):
        self.weight = weight
        self.pregnant = bool(pregnant)
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
        # Slow-unbinding myocardial Na+-channel pools (mcg/mL equivalent, Clarkson 1985).
        self._bupi_channel_ce = 0.0
        self._ropi_channel_ce = 0.0
        # Multi-beat filtered HR for use-dependent Na+-channel equilibration (~5-10 beats,
        # Clarkson 1985), preventing discrete-step feedback oscillation with bradycardia.
        self._rate_hr: float | None = None
        # ASRA resuscitation pharmacology harm states (Hiller 2009; Di Gregorio 2009; Neal 2018).
        self.epi_last_harm = 0.0
        self.vaso_last_harm = 0.0
        self.vaso_lung_water_ml_min = 0.0
        self.vaso_lactate_mmol_kg_min = 0.0
        # Track active LAST burden for clinical safety checks and scenario evaluation.
        self.last_severity = 0.0
        self.dominant_agent = "none"

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
        """Lidocaine effect-site input fraction under lipid emulsion (Heinonen 2015)."""
        return self.effect_input_fraction_for("lidocaine")

    def effect_input_fraction_for(self, drug: str, *, ph: float = 7.4) -> float:
        """Lipophilicity- and pH-scaled effect-site input reduction under 20% lipid emulsion.

        - Lidocaine: 0.35 max reduction calibrated to Heinonen 2015 (23% lower 30-min AUC).
        - Ropivacaine: 0.48 max reduction (Dureau 2016: V_IL 5.71 L, -26% Cmax; Mazoit 2009).
        - Bupivacaine: 0.68 max reduction (Dureau 2016: V_IL 12.2 L, -30% Cmax; Mazoit 2009
          Kd 665 vs 936 uM; Litonius 2012 shorter context-sensitive t1/2).
        Acidemia lowers lipophilic LA lipid binding (Mazoit 2009: 1.68x higher Kd at pH 7.0).
        """
        if self.lipid_effect <= 0.0:
            return 1.0
        if drug == "lidocaine":
            return 1.0 - 0.35 * self.lipid_effect
        acid_keep = 1.0 - 0.42 * clamp01((7.35 - ph) / 0.35)
        eff_lipid = self.lipid_effect * acid_keep
        if drug == "bupivacaine":
            return 1.0 - 0.68 * eff_lipid
        if drug == "ropivacaine":
            return 1.0 - 0.48 * eff_lipid
        return 1.0 - 0.35 * self.lipid_effect

    @property
    def active(self) -> bool:
        """Whether clinically meaningful local anesthetic systemic toxicity is present."""
        return (
            self.last_severity >= 0.15
            or self.seizure_activity >= 0.10
            or self.conduction_block >= 0.10
            or self.contractility_loss >= 0.10
        )

    def step(
        self,
        dt: float,
        *,
        neural_ce: float,
        cardiac_ce: float,
        ph: float,
        propofol_ce: float,
        midazolam_ce: float,
        mac_sevo: float,
        motor_strength: float,
        cortical_function: float,
        mac_n2o: float = 0.0,
        etomidate_ce: float = 0.0,
        sodium: float = 140.0,
        bupivacaine_neuro_ce: float = 0.0,
        bupivacaine_cv_ce: float = 0.0,
        ropivacaine_neuro_ce: float = 0.0,
        ropivacaine_cv_ce: float = 0.0,
        hr: float = 70.0,
        epi_ce: float = 0.0,
        vaso_ce: float = 0.0,
        flumazenil_ce: float = 0.0,
    ) -> None:
        lido_neuro = max(0.0, neural_ce)
        lido_cv = max(0.0, cardiac_ce)
        bupi_neuro = max(0.0, bupivacaine_neuro_ce)
        bupi_cv = max(0.0, bupivacaine_cv_ce)
        ropi_neuro = max(0.0, ropivacaine_neuro_ce)
        ropi_cv = max(0.0, ropivacaine_cv_ce)
        if self._rate_hr is None or dt <= 0.0:
            self._rate_hr = hr
        else:
            self._rate_hr += (hr - self._rate_hr) * -math.expm1(-dt / 8.0)
        if (
            lido_neuro <= 0.0
            and lido_cv <= 0.0
            and bupi_neuro <= 0.0
            and bupi_cv <= 0.0
            and ropi_neuro <= 0.0
            and ropi_cv <= 0.0
            and self._bupi_channel_ce <= 0.0
            and self._ropi_channel_ce <= 0.0
            and self.seizure_activity == 0.0
            and self.sympathetic_drive == 0.0
            and self.neural_depression == 0.0
            and self.conduction_block == 0.0
            and self.contractility_loss == 0.0
            and self.cardiac_stress == 0.0
            and self.epi_last_harm == 0.0
            and self.vaso_last_harm == 0.0
        ):
            self.vaso_lung_water_ml_min = 0.0
            self.vaso_lactate_mmol_kg_min = 0.0
            self.last_severity = 0.0
            self.dominant_agent = "none"
            return

        acidemia = 1.0 + 0.5 * clamp01((7.35 - ph) / 0.35)
        # Alkalinization and extracellular Na+ loading accelerate Na+-channel
        # recovery and QRS narrowing (porcine bupivacaine in Zaballos 2019,
        # extrapolated conservatively to lidocaine and ropivacaine).
        alkalemia_relief = 0.22 * clamp01((ph - 7.40) / 0.25)
        sodium_relief = 0.18 * clamp01((sodium - 140.0) / 10.0)
        bicarb_conduction_factor = 1.0 - clamp01(alkalemia_relief + sodium_relief)

        # Mazoit 2009: acidosis (pH 7.0) raises Kd 1.68x, cutting lipophilic LA lipid
        # binding ~40%; lidocaine-only path preserves exact existing Heinonen calibration.
        acid_lipid_keep = 1.0 - 0.42 * clamp01((7.35 - ph) / 0.35)
        has_long_acting = (
            bupi_neuro > 0.0
            or bupi_cv > 0.0
            or ropi_neuro > 0.0
            or ropi_cv > 0.0
            or self._bupi_channel_ce > 0.0
            or self._ropi_channel_ce > 0.0
        )

        # Slow myocardial Na+-channel unbinding for bupivacaine ("fast-in, slow-out",
        # Clarkson 1985) and intermediate kinetics for ropivacaine. Lipid shuttle
        # (Kuo 2013; Neal 2018) and alkalinization/Na+ loading accelerate washout.
        if has_long_acting and dt > 0.0:
            # Preliminary pressor harm from previous step modulates shuttle speed.
            shuttle_keep = max(0.25, 1.0 - 0.55 * self.epi_last_harm) * max(0.20, 1.0 - 0.60 * self.vaso_last_harm)
            bupi_shuttle = 1.0 + 3.2 * self.lipid_effect * acid_lipid_keep * shuttle_keep + 0.8 * (
                1.0 - bicarb_conduction_factor
            )
            ropi_shuttle = 1.0 + 1.6 * self.lipid_effect * acid_lipid_keep * shuttle_keep + 0.6 * (
                1.0 - bicarb_conduction_factor
            )
            bupi_tau = 15.0 if bupi_cv >= self._bupi_channel_ce else (220.0 / bupi_shuttle)
            self._bupi_channel_ce += (bupi_cv - self._bupi_channel_ce) * -math.expm1(-dt / bupi_tau)
            if self._bupi_channel_ce < 1e-6 and bupi_cv == 0.0:
                self._bupi_channel_ce = 0.0
            ropi_tau = 15.0 if ropi_cv >= self._ropi_channel_ce else (75.0 / ropi_shuttle)
            self._ropi_channel_ce += (ropi_cv - self._ropi_channel_ce) * -math.expm1(-dt / ropi_tau)
            if self._ropi_channel_ce < 1e-6 and ropi_cv == 0.0:
                self._ropi_channel_ce = 0.0
        elif has_long_acting:
            self._bupi_channel_ce = max(self._bupi_channel_ce, bupi_cv)
            self._ropi_channel_ce = max(self._ropi_channel_ce, ropi_cv)

        # Pregnancy cardiotoxicity sensitivity for long-acting amide LAs
        # (Morishima 1985 and ASRA Neal 2018; conservative because blinded
        # Santos 1995 did not replicate a pregnancy shift in ewes).
        preg_bupi = 1.22 if self.pregnant else 1.0
        preg_ropi = 1.08 if self.pregnant else 1.0

        # Rate-dependent Na+-channel block (Clarkson 1985: tau_rec = 1.56 s for bupivacaine).
        rate_hr = self._rate_hr if self._rate_hr is not None else hr
        bupi_rate_mult = _diastolic_block_rate_factor(rate_hr, 1.56) if self._bupi_channel_ce > 0.0 else 1.0
        ropi_rate_mult = _diastolic_block_rate_factor(rate_hr, 0.55) if self._ropi_channel_ce > 0.0 else 1.0

        # Combine agents in lidocaine-equivalent mcg/mL per endpoint.
        neural_raw = (
            lido_neuro
            + _BUPI_CNS_POTENCY * bupi_neuro
            + _ROPI_CNS_POTENCY * ropi_neuro
        )
        # Flumazenil removes tonic GABA-A restraint and slightly lowers LAST seizure threshold
        # (Klotz 1988 citing Scollo-Lavizzari 1983), in addition to antagonizing midazolam_ce.
        flum_proconvulsant = (
            1.0 + 0.18 * hill_function(flumazenil_ce, 5.0, 1.5)
            if flumazenil_ce > 0.0 and neural_raw > 0.0
            else 1.0
        )
        neural = neural_raw * acidemia * flum_proconvulsant

        lido_cardiac = lido_cv * acidemia
        bupi_eff_cv = max(bupi_cv, self._bupi_channel_ce) * acidemia * preg_bupi
        ropi_eff_cv = max(ropi_cv, self._ropi_channel_ce) * acidemia * preg_ropi

        cardiac_contract = (
            lido_cardiac
            + _BUPI_CONTRACTILITY_POTENCY * bupi_eff_cv
            + _ROPI_CONTRACTILITY_POTENCY * ropi_eff_cv
        )
        cardiac_cond = (
            lido_cardiac
            + _BUPI_CONDUCTION_POTENCY * bupi_eff_cv * bupi_rate_mult
            + _ROPI_CONDUCTION_POTENCY * ropi_eff_cv * ropi_rate_mult
        )
        cardiac_arr = (
            lido_cardiac
            + _BUPI_ARRHYTHMIA_POTENCY * bupi_eff_cv * (0.7 + 0.3 * bupi_rate_mult)
            + _ROPI_ARRHYTHMIA_POTENCY * ropi_eff_cv
        )

        # Track dominant toxic agent for telemetry/UI/clinical checks.
        bupi_score = max(_BUPI_CNS_POTENCY * bupi_neuro, _BUPI_CONDUCTION_POTENCY * bupi_eff_cv)
        ropi_score = max(_ROPI_CNS_POTENCY * ropi_neuro, _ROPI_CONDUCTION_POTENCY * ropi_eff_cv)
        lido_score = max(lido_neuro, lido_cardiac)
        if bupi_score >= ropi_score and bupi_score >= lido_score and bupi_score > 0.5:
            self.dominant_agent = "bupivacaine"
        elif ropi_score >= lido_score and ropi_score > 0.5:
            self.dominant_agent = "ropivacaine"
        elif lido_score > 0.5:
            self.dominant_agent = "lidocaine"
        else:
            self.dominant_agent = "none"

        # ASRA resuscitation pharmacology: high-dose epinephrine (>= 10 mcg/kg -> epi_ce > 15 ng/mL,
        # Hiller 2009) and vasopressin (Di Gregorio 2009; Neal 2018) impair LAST resuscitation.
        cardiac_toxic_present = cardiac_cond > 10.0 or self.conduction_block > 0.10 or self.contractility_loss > 0.10
        epi_harm_target = _response(max(0.0, epi_ce), 15.0, 60.0) if cardiac_toxic_present else 0.0
        vaso_harm_target = _response(max(0.0, vaso_ce), 10.0, 45.0) if cardiac_toxic_present else 0.0
        if dt > 0.0:
            self.epi_last_harm += (epi_harm_target - self.epi_last_harm) * -math.expm1(
                -dt / (20.0 if epi_harm_target > self.epi_last_harm else 90.0),
            )
            self.vaso_last_harm += (vaso_harm_target - self.vaso_last_harm) * -math.expm1(
                -dt / (25.0 if vaso_harm_target > self.vaso_last_harm else 120.0),
            )
            if self.epi_last_harm < 1e-4 and epi_harm_target == 0.0:
                self.epi_last_harm = 0.0
            if self.vaso_last_harm < 1e-4 and vaso_harm_target == 0.0:
                self.vaso_last_harm = 0.0
        else:
            self.epi_last_harm = epi_harm_target
            self.vaso_last_harm = vaso_harm_target
        # Di Gregorio 2009: vasopressin during LAST arrest causes hemorrhagic pulmonary
        # edema (6/6 rats) and severe lactic acidosis (7.5 vs 2.2 mmol/L).
        self.vaso_lung_water_ml_min = 35.0 * self.vaso_last_harm
        self.vaso_lactate_mmol_kg_min = 0.06 * self.vaso_last_harm

        # CNS toxicity: excitation/seizures -> depression/coma.
        self.neural_depression = _response(neural, 10.5, 20.0)
        excitation = _response(neural, 5.0, 9.5) * (1.0 - self.neural_depression)
        inhaled_mac = max(0.0, mac_sevo) + 0.5 * max(0.0, mac_n2o)
        remaining_excitation = (
            (1.0 - hill_function(midazolam_ce, 30.0, 2.0))
            * (1.0 - hill_function(propofol_ce, 1.5, 2.0))
            * (1.0 - hill_function(etomidate_ce, 0.25, 2.0))
            * (1.0 - hill_function(inhaled_mac, 0.75, 2.0))
        )
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

        # Drug-specific lipophilicity-weighted lipid rescue (Mazoit 2009; Zausig 2009;
        # Kuo 2013; Fettiplace 2014; Neal 2018), preserving exact lidocaine behavior
        # when bupivacaine and ropivacaine are absent.
        if not has_long_acting:
            eff_lipid = self.lipid_effect * (1.0 - 0.55 * self.epi_last_harm) * (1.0 - 0.60 * self.vaso_last_harm)
            cond_lipid_gain = 0.22
            cont_lipid_gain = 0.35
            arr_lipid_gain = 0.35
        else:
            total_c = max(1e-9, cardiac_cond)
            w_bupi = (_BUPI_CONDUCTION_POTENCY * bupi_eff_cv * bupi_rate_mult) / total_c
            w_ropi = (_ROPI_CONDUCTION_POTENCY * ropi_eff_cv * ropi_rate_mult) / total_c
            w_lido = lido_cardiac / total_c
            # Acidosis impairs lipid binding for lipophilic LAs; high-dose epi and vasopressin
            # blunt lipid resuscitation efficacy (Hiller 2009; Di Gregorio 2009).
            pressors_keep = (1.0 - 0.55 * self.epi_last_harm) * (1.0 - 0.60 * self.vaso_last_harm)
            eff_lipid = self.lipid_effect * (w_lido + (1.0 - w_lido) * acid_lipid_keep) * pressors_keep
            # Zausig 2009: lipid markedly improves bupivacaine cardiac recovery but has smaller
            # effect on less lipophilic ropivacaine and lidocaine.
            cond_lipid_gain = 0.22 * w_lido + 0.30 * w_ropi + 0.52 * w_bupi
            cont_lipid_gain = 0.35 * w_lido + 0.42 * w_ropi + 0.64 * w_bupi
            arr_lipid_gain = 0.35 * w_lido + 0.40 * w_ropi + 0.60 * w_bupi

        eff_cardiac_cond = cardiac_cond * (1.0 - cond_lipid_gain * eff_lipid) * bicarb_conduction_factor
        self.conduction_block = _response(eff_cardiac_cond, 18.0, 48.0)

        # Direct cardiotonic lipid rescue strengthens once Na+-channel block drops below
        # complete channel paralysis (Neal 2018).
        cardiotonic_gate = 1.0 - 0.45 * clamp01((self.conduction_block - 0.75) / 0.25) if has_long_acting else 1.0
        raw_contract = _response(cardiac_contract, 13.0, 42.0) * (
            1.0 - cont_lipid_gain * eff_lipid * cardiotonic_gate
        )
        self.contractility_loss = clamp01(raw_contract + 0.20 * self.vaso_last_harm * (1.0 - raw_contract))

        raw_stress = (
            _response(cardiac_arr, 22.0, 44.0)
            * (1.0 - arr_lipid_gain * eff_lipid)
            * (1.0 - 0.50 * (1.0 - bicarb_conduction_factor))
        )
        pressor_stress = 0.25 * self.epi_last_harm + 0.30 * self.vaso_last_harm
        self.cardiac_stress = clamp01(raw_stress + pressor_stress * (1.0 - raw_stress))
        self.last_severity = clamp01(
            max(self.seizure_activity, self.neural_depression, self.conduction_block, self.contractility_loss)
        )

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


class OpioidToxicityModel:
    """Rapid-bolus opioid chest-wall/glottic rigidity and acute naloxone withdrawal surge.

    Rigidity ("wooden chest"): rapid high-dose lipophilic opioid administration
    (calibrated for fentanyl >= ~5-7 mcg/kg or remifentanil >= ~2 mcg/kg IV
    bolus, extrapolated from alfentanil 175 mcg/kg in Benthuysen 1986 and
    sufentanil 3 mcg/kg in Bennett 1997) causes supraspinal skeletal-muscle
    rigidity and predominantly glottic vocal-cord closure during mask
    ventilation. Both are abolished by neuromuscular blockade or competitive
    naloxone reversal, and attenuated by concurrent GABAergic induction agents.

    Acute naloxone withdrawal surge: abrupt competitive displacement of high
    mu-opioid occupancy during surgical nociception or hypercapnia precipitates
    a transient catecholamine surge (tachycardia and hypertension).
    """

    def __init__(self) -> None:
        self._adapted_fast_opioid = 0.0
        self._prior_opioid_occ = 0.0
        self.central_rigidity = 0.0     # Supraspinal motor rigidity drive, 0-1 (pre-NMBA)
        self.rigidity = 0.0             # Effective skeletal/laryngeal rigidity, 0-1 (post-NMBA)
        self.glottic_obstruction = 0.0  # Upper-airway vocal cord closure, 0-1
        self.chest_wall_stiffness_add = 0.0  # Added chest-wall elastance ratio
        self.withdrawal_surge = 0.0     # Abrupt naloxone reversal sympathetic surge, 0-1
        self.heart_rate_increase = 0.0  # bpm
        self.svr_increase = 0.0         # Wood units

    def reset(self) -> None:
        self._adapted_fast_opioid = 0.0
        self._prior_opioid_occ = 0.0
        self.central_rigidity = 0.0
        self.rigidity = 0.0
        self.glottic_obstruction = 0.0
        self.chest_wall_stiffness_add = 0.0
        self.withdrawal_surge = 0.0
        self.heart_rate_increase = 0.0
        self.svr_increase = 0.0

    def step(
        self,
        dt: float,
        *,
        fentanyl_ce: float,
        remi_ce: float,
        opioid_ce_raw: float,
        naloxone_ce: float,
        muscle_strength: float,
        propofol_ce: float = 0.0,
        midazolam_ce: float = 0.0,
        mac_sevo: float = 0.0,
        surgical_stress: float = 0.0,
        paco2: float = 40.0,
    ) -> None:
        nal_dr = 1.0 + max(0.0, naloxone_ce) / NALOXONE_KB_NG_ML if naloxone_ce > 0.0 else 1.0
        eff_fast = (1.4 * max(0.0, fentanyl_ce) + max(0.0, remi_ce)) / nal_dr
        eff_total = max(0.0, opioid_ce_raw) / nal_dr
        if (
            eff_fast <= 0.0
            and eff_total <= 0.0
            and self._adapted_fast_opioid <= 1e-6
            and self._prior_opioid_occ <= 1e-6
            and self.central_rigidity <= 1e-6
            and self.withdrawal_surge <= 1e-6
        ):
            self.central_rigidity = self.rigidity = 0.0
            self.glottic_obstruction = self.chest_wall_stiffness_add = 0.0
            self.withdrawal_surge = self.heart_rate_increase = self.svr_increase = 0.0
            return

        # Slow adaptation prevents steady maintenance TIVA infusions from triggering
        # wooden-chest rigidity while preserving rapid-bolus onset (~45-90 s, Benthuysen 1986).
        if dt > 0.0:
            adapt_tau = 420.0 if eff_fast >= self._adapted_fast_opioid else 180.0
            self._adapted_fast_opioid += (eff_fast - self._adapted_fast_opioid) * -math.expm1(-dt / adapt_tau)
        surge = max(0.0, eff_fast - 0.65 * self._adapted_fast_opioid)
        # GABAergic hypnotics modestly attenuate opioid rigidity without blocking massive boluses.
        gaba_attenuation = (
            1.0
            - 0.35 * hill_function(max(0.0, propofol_ce), 3.0, 2.0)
        ) * (
            1.0
            - 0.30 * hill_function(max(0.0, midazolam_ce), 80.0, 2.0)
        ) * (
            1.0
            - 0.30 * hill_function(max(0.0, mac_sevo), 1.0, 2.0)
        )
        target_rigidity = _response(surge, 7.5, 16.0) * gaba_attenuation
        if dt > 0.0:
            tau_rig = 20.0 if target_rigidity > self.central_rigidity else 15.0
            self.central_rigidity += (target_rigidity - self.central_rigidity) * -math.expm1(-dt / tau_rig)
            if self.central_rigidity < 1e-4 and target_rigidity == 0.0:
                self.central_rigidity = 0.0
        else:
            self.central_rigidity = target_rigidity

        strength = clamp01(muscle_strength)
        self.rigidity = clamp01(self.central_rigidity * strength)
        # Bennett 1997: closed vocal cords dominate difficult mask ventilation (28/30),
        # while Benthuysen 1986 / Scamman 1983 show concurrent thoracoabdominal stiffness.
        self.glottic_obstruction = 0.90 * self.rigidity
        self.chest_wall_stiffness_add = 0.85 * self.rigidity

        # Acute naloxone-precipitated sympathetic surge when high opioid occupancy
        # is abruptly displaced by competitive antagonism.
        curr_occ = hill_function(eff_total, 1.5, 1.5) if eff_total > 0.0 else 0.0
        if naloxone_ce > 0.25 and self._prior_opioid_occ > curr_occ + 0.05:
            displaced = clamp01((self._prior_opioid_occ - curr_occ) / 0.55)
            hypercapnia_stress = clamp01((paco2 - 46.0) / 18.0)
            stress_mult = 0.35 + 0.65 * max(clamp01(surgical_stress), hypercapnia_stress)
            surge_target = clamp01(displaced * stress_mult)
            if surge_target > self.withdrawal_surge:
                self.withdrawal_surge += (surge_target - self.withdrawal_surge) * -math.expm1(-dt / 12.0)
        elif dt > 0.0 and self.withdrawal_surge > 0.0:
            self.withdrawal_surge *= math.exp(-dt / 180.0)
            if self.withdrawal_surge < 1e-4:
                self.withdrawal_surge = 0.0
        if dt > 0.0:
            occ_tau = 30.0 if curr_occ >= self._prior_opioid_occ else 120.0
            self._prior_opioid_occ += (curr_occ - self._prior_opioid_occ) * -math.expm1(-dt / occ_tau)
        else:
            self._prior_opioid_occ = curr_occ
        self.heart_rate_increase = 28.0 * self.withdrawal_surge
        self.svr_increase = 4.5 * self.withdrawal_surge

    @property
    def metabolic_factor_add(self) -> float:
        """Excess skeletal-muscle O2 consumption during tonic opioid rigidity (Benthuysen 1986)."""
        return 0.25 * self.rigidity
