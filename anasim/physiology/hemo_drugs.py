"""Anesthetic, autonomic, and vasoactive pharmacodynamic helpers for HemodynamicModel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from anasim.core.utils import clamp01, hill_function

from .hemo_config import HemodynamicConfig


def interpolate_piecewise(x: float, points: tuple[tuple[float, float], ...]) -> float:
    """Piecewise-linear interpolation, constant beyond the end points."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in pairwise(points):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def saturate_alpha_svr(factor: float) -> float:
    """Shared alpha-1 smooth-muscle ceiling preventing unbounded stacked pressor SVR."""
    if factor <= 1.0:
        return factor
    return 1.0 + 2.4 * math.tanh((factor - 1.0) / 2.4)


def calc_hr_sv_svr_effects(
    ce: float,
    c50: float,
    gamma: float,
    emax_hr: float,
    emax_sv: float,
    emax_svr: float,
) -> tuple[float, float, float]:
    """Return (delta HR, SV factor, SVR factor) from one Hill effect."""
    if ce <= 0:
        return 0.0, 1.0, 1.0
    hill = hill_function(ce, c50, gamma)
    return emax_hr * hill, 1.0 + emax_sv * hill, max(0.5, 1.0 + emax_svr * hill)


def calc_phenyl_effects(cfg: HemodynamicConfig, ce_phenyl: float) -> float:
    """Phenylephrine SVR effect (pure alpha-1)."""
    if ce_phenyl <= 0:
        return 1.0
    return 1.0 + cfg.phenyl_emax_svr * hill_function(ce_phenyl, cfg.phenyl_c50, cfg.phenyl_gamma)


def calc_nore_effects(
    cfg: HemodynamicConfig,
    ce_nore: float,
    basal_nore_ce: float,
    age_chronotropy: float,
    beta1_dr: float = 1.0,
    alpha_dr: float = 1.0,
    anesthetic_depth: float = 0.0,
    *,
    nore_scale: float = 1.0,
    hf_beta_hr: float = 1.0,
    hf_beta_sv: float = 1.0,
    preload_reserve: float = 1.0,
) -> tuple[float, float, float]:
    """Return direct chronotropy, stroke-volume factor, and vascular-resistance factor.

    Reflex bradycardia comes from the baroreflex. Anesthetic SV restoration scales
    with pre-pressor preload reserve (strong in unpreloaded venodilated states as
    in de Keijzer 2026 / Maas 2013, attenuated on the flat Starling plateau as in
    volume-preloaded Poterman 2015).
    """
    eff_ce = max(0.0, ce_nore - basal_nore_ce * (1.0 - anesthetic_depth)) * nore_scale
    if eff_ce <= 0:
        return 0.0, 1.0, 1.0
    beta_hill = hill_function(eff_ce / beta1_dr, cfg.nore_c50, 1.0)
    # Apply HFrEF beta-1 receptor downregulation to exogenous NE above resting
    # endogenous tone, preserving baseline sympathetic support and venous-return coupling.
    exo_share = clamp01((ce_nore - basal_nore_ce) / max(1e-6, eff_ce / max(1e-6, nore_scale)))
    eff_hf_hr = 1.0 - exo_share * (1.0 - hf_beta_hr)
    eff_hf_sv = 1.0 - exo_share * 0.5 * (1.0 - hf_beta_sv)
    delta_hr = cfg.nore_emax_hr * eff_hf_hr * beta_hill * age_chronotropy
    # Anesthetic SV restoration saturates at high doses to avoid supraphysiologic SV.
    anest_hill = (
        beta_hill
        if beta_hill <= 0.42
        else 0.42 + 0.18 * math.tanh((beta_hill - 0.42) / 0.18)
    )
    sv_factor = (
        1.0
        + cfg.nore_emax_sv * eff_hf_sv * beta_hill
        + cfg.nore_anesthetic_sv_gain * preload_reserve * anesthetic_depth * anest_hill
    )
    alpha_hill = hill_function(eff_ce / alpha_dr, cfg.nore_c50_alpha, 1.0)
    alpha_gain = 1.0 + cfg.nore_anesthetic_alpha_gain * anesthetic_depth
    svr_factor = 1.0 + cfg.nore_emax_svr * alpha_gain * alpha_hill
    return delta_hr, sv_factor, svr_factor


def calc_epi_effects(
    cfg: HemodynamicConfig,
    ce_epi: float,
    ce_pressor: float,
    ce_high_pressor: float,
    epi_beta2_effect: float,
    ce_sevo: float,
    age_chronotropy: float,
    hf_beta_hr: float,
    hf_beta_sv: float,
    vasc_tone_ratio: float,
    volume_deficit_fraction: float,
    beta1_dr: float = 1.0,
    alpha_dr: float = 1.0,
) -> tuple[float, float, float]:
    """Return direct chronotropy, SV factor, and baseline-relative SVR factor."""
    age_scale = age_chronotropy
    delta_hr = cfg.epi_emax_hr * hf_beta_hr * hill_function(ce_epi / beta1_dr, cfg.epi_c50_hr, cfg.epi_gamma_hr)
    delta_hr *= (1.0 - cfg.epi_volatile_hr_depression * clamp01(ce_sevo)) * age_scale
    sv_factor = 1.0 + cfg.epi_emax_sv * hf_beta_sv * (1 + cfg.epi_age_sv * (age_scale - 1)) * hill_function(
        ce_pressor / beta1_dr, cfg.epi_c50_sv, 1.0
    )
    alpha_age_scale = 1 + cfg.epi_age_alpha * (age_scale - 1)
    alpha = cfg.epi_emax_svr_alpha * hill_function(
        ce_pressor * alpha_age_scale / alpha_dr, cfg.epi_c50_alpha, cfg.epi_gamma_alpha
    ) + cfg.epi_emax_svr_alpha_high * hill_function(
        ce_high_pressor / alpha_dr, cfg.epi_c50_alpha_high, cfg.epi_gamma_alpha_high
    )
    shock_beta2_keep = 1.0 - 0.75 * clamp01((volume_deficit_fraction - 0.15) / 0.20)
    svr_factor = (
        1.0
        + epi_beta2_effect * (1 + cfg.epi_age_beta2 * (age_scale - 1)) * vasc_tone_ratio * shock_beta2_keep
        + alpha
    )
    return delta_hr, sv_factor, max(0.2, svr_factor)


def compute_anesthetic_factors(
    cfg: HemodynamicConfig,
    ce_prop: float,
    ce_opioid: float,
    ce_sevo: float,
    ce_remi_vagal: float,
    sympathetic_gain: float,
    emax_prop_sv_age: float,
    vagal: float = 0.0,
    *,
    adjunct_midaz: float = 0.0,
    adjunct_etom: float = 0.0,
) -> tuple[float, float, float]:
    """Compute combined anesthetic multipliers on TPR, SV, and HR (Su 2023)."""
    cp = max(0.0, ce_prop)
    cr = max(0.0, ce_opioid)
    if cr > 0.0 and cfg.opioid_cv_saturation > 0.0:
        cr = cfg.opioid_cv_saturation * math.tanh(cr / cfg.opioid_cv_saturation)

    if cp > 0.0:
        remi_int_term = cfg.int_tpr * hill_function(cr, cfg.ec50_remi_tpr, 1.0) if cr > 0.0 else 0.0
        eff_prop_tpr = (cfg.emax_prop_tpr * sympathetic_gain + remi_int_term) * hill_function(
            cp, cfg.ec50_prop_tpr, cfg.gamma_prop
        )
        eff_prop_sv = emax_prop_sv_age * hill_function(cp, cfg.ec50_prop_sv, 1.0)
    else:
        eff_prop_tpr = eff_prop_sv = 0.0

    if ce_sevo > 0.0:
        eff_sevo_tpr = cfg.sevo_emax_tpr * hill_function(ce_sevo, cfg.sevo_ec50_tpr, cfg.sevo_gamma_tpr)
        eff_sevo_sv = cfg.sevo_emax_sv * hill_function(ce_sevo, cfg.sevo_ec50_sv, 1.0)
        total_eff_hr = cfg.sevo_emax_hr * hill_function(ce_sevo, cfg.sevo_ec50_hr, 1.0)
    else:
        eff_sevo_tpr = eff_sevo_sv = total_eff_hr = 0.0

    total_eff_tpr = max(-0.95, (1.0 + eff_prop_tpr) * (1.0 + eff_sevo_tpr) - 1.0)
    total_eff_sv = max(-0.95, (1.0 + eff_prop_sv) * (1.0 + eff_sevo_sv) - 1.0)

    if cr > 0.0:
        eff_remi_tpr = cfg.emax_remi_tpr * hill_function(cr, cfg.ec50_remi_tpr, cfg.gamma_remi_tpr)
        hypnotic_sv_cp = cp + 1.5 * adjunct_midaz + 0.3 * adjunct_etom
        hypnotic_hr_cp = cp + 1.5 * adjunct_midaz + 0.8 * adjunct_etom
        slope_sv = cfg.sl_remi_sv + (
            cfg.int_sv * hill_function(hypnotic_sv_cp, cfg.ec50_int_sv, 1.0) if hypnotic_sv_cp > 0.0 else 0.0
        )
        eff_remi_sv = min(0.0, slope_sv * cr)
        slope_hr = cfg.sl_remi_hr + (
            cfg.int_hr * hill_function(hypnotic_hr_cp, cfg.ec50_int_hr, 1.0) if hypnotic_hr_cp > 0.0 else 0.0
        )
        eff_remi_hr = max(-0.9, min(0.0, slope_hr * cr))
        if eff_remi_hr < 0.0:
            eff_remi_hr *= 1.0 - vagal
    else:
        eff_remi_tpr = eff_remi_sv = eff_remi_hr = 0.0

    drug_tpr = (1.0 + total_eff_tpr) / max(0.05, 1.0 - eff_remi_tpr)
    drug_sv = (1.0 + total_eff_sv) / max(0.05, 1.0 - eff_remi_sv)
    remi_vagal = (
        cfg.remi_vagal_hr * hill_function(ce_remi_vagal, cfg.remi_vagal_c50, 1.0)
        if ce_remi_vagal > 0.0
        else 0.0
    )
    drug_hr = (1.0 + total_eff_hr) / max(0.05, 1.0 - eff_remi_hr) * (1.0 - remi_vagal * (1.0 - vagal))
    return drug_tpr, drug_sv, drug_hr


@dataclass(frozen=True, slots=True)
class ReceptorBlockade:
    beta1_dr: float
    beta2_dr: float
    alpha_dr: float
    beta_occ: float
    alpha_occ: float
    vagal: float
    vagal_hr: float
    amiodarone: float
    amiodarone_hr: float
    beta_response: float
    sympathetic_hr_response: float
    alpha_response: float


def compute_receptor_blockade(
    cfg: HemodynamicConfig,
    ce_esmolol: float,
    ce_labetalol: float,
    ce_glyco: float,
    ce_atropine: float,
    ce_amiodarone: float,
    ce_amiodarone_hr: float,
    ce_prop_cv: float,
    age_chronotropy: float,
    age_antimuscarinic: float,
    ce_neostigmine: float = 0.0,
) -> ReceptorBlockade:
    """Compute competitive antagonist dose ratios, vagal block, and reflex shares."""
    if ce_esmolol > 0.0 or ce_labetalol > 0.0:
        labetalol_beta = max(0.0, ce_labetalol) / cfg.labetalol_kb_beta
        beta1_dr = 1.0 + max(0.0, ce_esmolol) / cfg.esmolol_kb + labetalol_beta
        beta2_dr = 1.0 + labetalol_beta
        alpha_dr = 1.0 + max(0.0, ce_labetalol) / cfg.labetalol_kb_alpha
        beta_occ = 1.0 - 1.0 / beta1_dr
        alpha_occ = 1.0 - 1.0 / alpha_dr
    else:
        beta1_dr = beta2_dr = alpha_dr = 1.0
        beta_occ = alpha_occ = 0.0

    neo_m2 = (
        hill_function(ce_neostigmine, cfg.neostigmine_hr_c50, cfg.neostigmine_hr_gamma)
        if ce_neostigmine > 0.0
        else 0.0
    )
    # Saturable synaptic ACh accumulation from cardiac AChE inhibition competes with
    # antimuscarinics at M2 receptors (Mirakhur 1977), preserving 5:1
    # neostigmine:glycopyrrolate balance across clinical reversal doses (2.5-5 mg)
    # while keeping atropine's brief 1-2 min tachycardia followed by mild slowing.
    m2_ach_dr = 1.0 + cfg.neostigmine_m2_shift_max * neo_m2 if neo_m2 > 0.0 else 1.0
    if ce_glyco > 0.0 or ce_atropine > 0.0:
        glyco = hill_function(ce_glyco, cfg.glyco_c50 * m2_ach_dr, cfg.glyco_gamma) if ce_glyco > 0.0 else 0.0
        glyco_chrono = (
            hill_function(ce_glyco, cfg.glyco_hr_c50 * m2_ach_dr, cfg.glyco_hr_gamma)
            if ce_glyco > 0.0
            else 0.0
        )
        atropine = hill_function(ce_atropine, cfg.atropine_c50 * m2_ach_dr, cfg.atropine_gamma) if ce_atropine > 0.0 else 0.0
        vagal = 1.0 - (1.0 - glyco) * (1.0 - atropine)
        prop_vagal_scale = 1.0 / (1.0 + max(0.0, ce_prop_cv) / cfg.atropine_propofol_cp50)
        atropine_hr = cfg.atropine_emax_hr * atropine * prop_vagal_scale * age_antimuscarinic
        glyco_hr = cfg.glyco_emax_hr * glyco_chrono * prop_vagal_scale * age_antimuscarinic
        vagal_hr = max(glyco_hr, glyco_hr * (1.0 - atropine) + atropine_hr)
    else:
        vagal = vagal_hr = 0.0

    if neo_m2 > 0.0:
        vagal_hr -= cfg.neostigmine_emax_hr * neo_m2 * (1.0 - cfg.neostigmine_vagal_block_fraction * vagal)

    amiodarone = hill_function(ce_amiodarone, cfg.amiodarone_c50, 1.0) if ce_amiodarone > 0.0 else 0.0
    amiodarone_hr = hill_function(ce_amiodarone_hr, cfg.amiodarone_c50, 1.0) if ce_amiodarone_hr > 0.0 else 0.0
    beta_response = 1.0 - cfg.beta_block_reflex * beta_occ
    sympathetic_hr_response = (
        beta_response * age_chronotropy * (1.0 - cfg.amiodarone_beta_reflex_block * amiodarone_hr)
    )
    alpha_response = 1.0 - alpha_occ
    return ReceptorBlockade(
        beta1_dr=beta1_dr,
        beta2_dr=beta2_dr,
        alpha_dr=alpha_dr,
        beta_occ=beta_occ,
        alpha_occ=alpha_occ,
        vagal=vagal,
        vagal_hr=vagal_hr,
        amiodarone=amiodarone,
        amiodarone_hr=amiodarone_hr,
        beta_response=beta_response,
        sympathetic_hr_response=sympathetic_hr_response,
        alpha_response=alpha_response,
    )


def compute_step_ephedrine(
    cfg: HemodynamicConfig,
    ce_ephedrine_raw: float,
    eff_ce_ephedrine: float,
    tolerance_ce: float,
    dt_min: float,
    circulation: float,
    agonist_shift: float,
    acidosis_factor: float,
    alpha_dr: float,
    beta1_dr: float,
    base_hr: float,
    age_chronotropy: float,
    amiodarone_hr: float,
    *,
    hf_beta_hr: float = 1.0,
    hf_beta_sv: float = 1.0,
    ce_prop: float = 0.0,
    ce_sevo: float = 0.0,
) -> tuple[float, float, float, float]:
    """Update ephedrine tolerance state and return (new_tol_ce, svr, sv, delta_hr)."""
    if ce_ephedrine_raw <= 0.0 and tolerance_ce == 0.0:
        return tolerance_ce, 1.0, 1.0, 0.0
    new_tol_ce = tolerance_ce + (ce_ephedrine_raw - tolerance_ce) * (
        -math.expm1(-math.log(2.0) * dt_min * circulation / cfg.ephedrine_tolerance_half_min)
    )
    ephedrine_raw = ce_ephedrine_raw / agonist_shift
    ephedrine = max(0.0, eff_ce_ephedrine) / agonist_shift
    tol = cfg.ephedrine_tolerance_max * hill_function(
        new_tol_ce, cfg.ephedrine_tolerance_c50, cfg.ephedrine_tolerance_gamma,
    )
    ephedrine_alpha_effect = acidosis_factor * max(
        0.0, hill_function(ephedrine_raw / alpha_dr, cfg.ephedrine_c50, cfg.ephedrine_gamma) - tol,
    )
    ephedrine_beta_effect = acidosis_factor * max(
        0.0, hill_function(ephedrine_raw / beta1_dr, cfg.ephedrine_c50, cfg.ephedrine_gamma) - tol,
    )
    prop_svr_gain = (
        1.0 + cfg.ephedrine_propofol_svr_gain * hill_function(ce_prop, 2.0, 1.5)
        if ce_prop > 0.0
        else 1.0
    )
    sevo_depth = clamp01(ce_sevo)
    sevo_svr_keep = 1.0 - cfg.ephedrine_sevo_svr_depression * sevo_depth
    sevo_hr_keep = 1.0 - cfg.ephedrine_sevo_hr_depression * sevo_depth
    ephedrine_svr = 1.0 + cfg.ephedrine_emax_svr * prop_svr_gain * sevo_svr_keep * ephedrine_alpha_effect
    ephedrine_sv = 1.0 + cfg.ephedrine_emax_sv * hf_beta_sv * ephedrine_beta_effect
    ephedrine_chrono_gate = (
        age_chronotropy * hf_beta_hr * (1.0 - cfg.amiodarone_beta_reflex_block * amiodarone_hr) * sevo_hr_keep
    )
    ephedrine_beta_conc = ephedrine / beta1_dr
    sustained_hr_max = 0.40
    sustained_hr_term = (
        sustained_hr_max * ephedrine_beta_conc / (ephedrine_beta_conc + sustained_hr_max / cfg.ephedrine_hr_slope)
        if cfg.ephedrine_hr_slope > 0.0 and ephedrine_beta_conc > 0.0
        else 0.0
    )
    ephedrine_hr = (sustained_hr_term + cfg.ephedrine_emax_hr * ephedrine_beta_effect) * base_hr * ephedrine_chrono_gate
    return new_tol_ce, ephedrine_svr, ephedrine_sv, ephedrine_hr
