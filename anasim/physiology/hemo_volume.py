"""Intravascular volume, crystalloid shifts, capillary leak, and lung water accounting."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from anasim.core.constants import BLOOD_VOLUME_MIN
from anasim.core.utils import clamp01
from anasim.patient.domain import bounded_number, finite_number

if TYPE_CHECKING:
    from .hemodynamics import HemodynamicModel

MAX_HEMATOCRIT = 0.7


def add_volume(
    model: HemodynamicModel,
    amount_ml: float,
    hematocrit: float = 0.0,
    retention_fraction: float | None = None,
    label: str = "crystalloid",
    *,
    suppress_tde: bool = False,
) -> None:
    """Add fluid or blood (positive) or remove blood (negative).

    Retained volume also gives a transient Frank-Starling SV rise via tde_sv.
    """
    cfg = model.config
    amount_ml = finite_number("amount_ml", amount_ml)
    hematocrit = bounded_number("hematocrit", hematocrit, 0.0, MAX_HEMATOCRIT)
    if amount_ml == 0:
        return
    prev_co = model._prev_co
    model._cached_state = None
    if abs(amount_ml) > 10.0:
        model._prev_co = None

    gain = 0.02 * (model.base_sv / 80.0)
    retained_ml = 0.0
    if amount_ml < 0:
        loss_ml = -amount_ml
        min_bv = BLOOD_VOLUME_MIN + max(0.0, model.extracorporeal_volume_ml)
        available = max(0.0, model.blood_volume - min_bv)
        actual_loss = min(loss_ml, available)
        if actual_loss > 0:
            model.fluids.bleed(actual_loss, model.red_cell_volume_ml / model.blood_volume)
            model.crystalloid.central_ml *= 1.0 - actual_loss / model.blood_volume
            model.red_cell_volume_ml *= 1.0 - actual_loss / model.blood_volume
            model.blood_volume -= actual_loss
            model.total_blood_out_ml += actual_loss
            hb_loss = model.hb_conc * actual_loss / 100.0
            model.hb_mass = max(0.0, model.hb_mass - hb_loss)
            if not suppress_tde:
                model.tde_sv = max(0.0, model.tde_sv - actual_loss * gain)
    else:
        if hematocrit == 0.0 and label == "blood":
            raise ValueError("Blood administration requires hematocrit > 0")
        if hematocrit > 0:
            model.total_blood_in_ml += amount_ml
            retained_ml = amount_ml * cfg.blood_retention_fraction
            model.total_third_space_ml += max(0.0, amount_ml - retained_ml)
            donor_red_cells_ml = hematocrit * retained_ml
            model.red_cell_volume_ml += donor_red_cells_ml
            # Donor MCHC is independent of the recipient's Hb/Hct ratio.
            model.hb_mass += donor_red_cells_ml * cfg.donor_mchc_g_dl / 100.0
            # Route protein-free SAGM supernatant through crystalloid kinetics while
            # keeping donor red cells 100% intravascular.
            supernatant_ml = max(0.0, retained_ml - donor_red_cells_ml)
            if supernatant_ml > 0.0:
                model.crystalloid.infuse(supernatant_ml, saline=False)
        else:
            is_colloid = label in ("colloid", "lipid")
            if is_colloid:
                model.total_colloid_in_ml += amount_ml
                retention = 0.25 if label == "lipid" else cfg.colloid_retention_fraction
            else:
                model.total_crystalloid_in_ml += amount_ml
                retention = 1.0
            if retention_fraction is not None:
                retention = retention_fraction
            retention = clamp01(retention)
            retained_ml = amount_ml * retention
            if label == "lipid":
                total_water_l = model.fluids.ecf_l + model.fluids.icf_l
                ecf_share = model.fluids.ecf_l / total_water_l if total_water_l > 0.0 else 1.0 / 3.0
                ecf_input_ml = amount_ml * ecf_share
            else:
                ecf_input_ml = amount_ml
            third_space_gain_ml = max(0.0, ecf_input_ml - retained_ml)
            model.total_third_space_ml += third_space_gain_ml
            if label == "colloid":
                model._protein_third_space_ml += third_space_gain_ml
            if not is_colloid:
                model.crystalloid.infuse(
                    retained_ml,
                    saline=label in ("saline", "bicarbonate"),
                    dextrose=label in ("d5w", "d10w"),
                )

        model.blood_volume += retained_ml
        model.fluids.infuse(amount_ml, retained_ml, hematocrit, label)
        if label == "bicarbonate" and amount_ml > 0.0:
            # 8.4% NaHCO3 (2000 mOsm/L) causes brief rate-dependent hyperosmolar
            # arteriolar vasodilation when pushed rapidly (canine Huseby and Gumprecht 1981;
            # transient 1-min dip in porcine Zaballos 2019).
            delta_load = amount_ml / max(1000.0, model.blood_volume_0)
            circ = model.circulation_fraction(prev_co)
            model._hyperosmolar_load += delta_load * circ
            model._pending_hyperosmolar_load += delta_load * (1.0 - circ)

        if retained_ml > 0 and not suppress_tde and model.patient.heart_failure != "hfref":
            reserve = 0.35 if model.patient.heart_failure == "hfpef" else 1.0
            model.tde_sv += retained_ml * gain * reserve

    model.blood_volume = max(BLOOD_VOLUME_MIN + max(0.0, model.extracorporeal_volume_ml), model.blood_volume)
    model._update_hb_conc()
    model._cached_state = None


def step_intravascular_volume(
    model: HemodynamicModel,
    sub_dt: float,
    dt_min: float,
    map_prev: float,
    co_prev: float,
    circulation: float,
    sepsis_sev: float,
    anaph_sev: float,
    leak_rate_per_s: float,
    base_co: float,
    *,
    pulmonary_flow_prev: float | None = None,
    osmotic_diuresis_ml_min: float = 0.0,
) -> None:
    """Advance crystalloid kinetics, urine, capillary leak, refill, and lung water."""
    cfg = model.config
    model.renal_perfusion = model._compute_renal_perfusion(map_prev, co_prev, sepsis_sev)
    renal_factor = model.renal_perfusion * model.patient.renal_clearance_factor
    # Keep plasma losses within the hematocrit represented by the chemistry model.
    ext_vol = max(0.0, model.extracorporeal_volume_ml)
    minimum_blood_volume = max(BLOOD_VOLUME_MIN + ext_vol, model.red_cell_volume_ml / MAX_HEMATOCRIT)
    cryst_elim_scale = (
        model.patient.renal_clearance_factor
        * cfg.crystalloid_renal_scale
        * model._crystalloid_elimination_factor(co_prev, sepsis_sev)
    )
    fluid_change, fluid_urine, fluid_sequestration = model.crystalloid.step(
        sub_dt,
        map_prev,
        model.patient.age,
        cryst_elim_scale,
        model.blood_volume - minimum_blood_volume,
        urinary_elimination=cfg.vol_clearance is None,
        circulation=circulation,
        mcfp_mmHg=model._last_mcfp,
    )
    model.blood_volume += fluid_change
    model.fluids.shift_water(fluid_change + fluid_urine)
    model.fluids.lose_urine(fluid_urine)
    model.total_urine_out_ml += fluid_urine
    model.total_third_space_ml += fluid_sequestration
    # On open-reservoir CPB, drained blood in the venous reservoir remains part of
    # the active systemic pool rather than hemorrhagic loss, while fixed circuit
    # tubing prime sits outside patient circulating volume.
    circ_ext_vol = (
        max(0.0, ext_vol - model._cpb_reservoir_ml) if model._cpb_open_drainage else ext_vol
    )
    if cfg.vol_clearance is not None:
        urine_ml_min = max(0.0, float(cfg.vol_clearance))
    else:
        vol_excess_r = max(0.0, (model.blood_volume - circ_ext_vol - model.blood_volume_0) / max(1.0, model.blood_volume_0))
        hypervol_diuresis = 1.0 + 1.8 * clamp01((vol_excess_r - 0.12) / 0.25)
        urine_ml_min = cfg.uop_ml_kg_hr * model.fluids.lean_mass_kg / 60.0 * hypervol_diuresis
    osmotic_ml_min = max(0.0, osmotic_diuresis_ml_min) if cfg.vol_clearance is None else 0.0
    urine_out_ml = min(
        (urine_ml_min * renal_factor + osmotic_ml_min) * dt_min,
        max(0.0, model.blood_volume - minimum_blood_volume),
    )
    blood_volume = model.blood_volume - urine_out_ml
    if urine_out_ml > 0:
        model.fluids.lose_urine(urine_out_ml)
        plasma_volume = model.blood_volume - model.red_cell_volume_ml
        model.crystalloid.central_ml *= max(0.0, 1.0 - urine_out_ml / plasma_volume)
        model.total_urine_out_ml += urine_out_ml
    if leak_rate_per_s > 0.0:
        # Capillary leak occurs across patient microvasculature, not circuit tubing/reservoir.
        patient_bv = max(0.0, blood_volume - ext_vol)
        leak_ml = patient_bv * -math.expm1(-leak_rate_per_s * sub_dt)
        available = max(0.0, blood_volume - minimum_blood_volume)
        actual_leak = min(leak_ml, available)
        if actual_leak > 0:
            model.fluids.shift_water(-actual_leak, protein_fraction=1.0)
            model.crystalloid.central_ml *= max(0.0, 1.0 - actual_leak / (blood_volume - model.red_cell_volume_ml))
            blood_volume -= actual_leak
            model.total_third_space_ml += actual_leak
            model._protein_third_space_ml += actual_leak
    if model.total_third_space_ml > 0 and cfg.third_space_refill_tau_hr > 0:
        intravascular_excess = (blood_volume - circ_ext_vol) - model.blood_volume_0
        shortfall = cfg.interstitial_refill_fraction * (
            intravascular_excess + model.total_third_space_ml
        ) - intravascular_excess
        frac = -math.expm1(-sub_dt / (cfg.third_space_refill_tau_hr * 3600.0))
        refill_ml = min(model.total_third_space_ml, shortfall * frac)
        if refill_ml > 0:
            prot_frac = clamp01(model._protein_third_space_ml / model.total_third_space_ml)
            model.fluids.shift_water(refill_ml, protein_fraction=prot_frac)
            model._protein_third_space_ml = max(0.0, model._protein_third_space_ml - refill_ml * prot_frac)
            model.total_third_space_ml -= refill_ml
            blood_volume += refill_ml

    vol_deficit_ml = model.blood_volume_0 - (blood_volume - circ_ext_vol)
    if cfg.vol_clearance is None and leak_rate_per_s <= 0.0 and vol_deficit_ml > 50.0 and circulation > 0.0:
        # Falling capillary hydrostatic pressure recruits protein-free interstitial
        # water (Starling transcapillary refill, ~200-270 mL/h peak) while perfusing;
        # expanded interstitium (e.g. during osmotic diuresis) equilibrates across
        # capillaries faster via raised interstitial pressure and plasma oncotic rise.
        interstitial_avail_ml = max(0.0, (model.fluids.interstitial_l - 6.0) * 1000.0)
        refill_drive = (
            clamp01((vol_deficit_ml - 50.0) / 950.0)
            * (0.45 + 0.55 * clamp01((85.0 - map_prev) / 35.0))
            * circulation
        )
        initial_isf_l = model.fluids.initial_ecf_l - model.fluids.initial_plasma_l
        isf_excess_ml = max(0.0, (model.fluids.interstitial_l - initial_isf_l) * 1000.0)
        excess_rate_ml_min = 0.015 * isf_excess_ml * circulation
        autotransfusion_ml = min(
            interstitial_avail_ml,
            vol_deficit_ml * 0.65,
            (4.5 * refill_drive + excess_rate_ml_min) * dt_min,
        )
        if autotransfusion_ml > 0.0:
            model.fluids.shift_water(autotransfusion_ml)
            blood_volume += autotransfusion_ml

    # Lung water filters from blood above the Starling/oncotic LAP threshold and
    # returns over hours; hypoalbuminemia (<34 g/L) lowers colloid osmotic pressure
    # and inflammatory states (sepsis/anaphylaxis) increase alveolar-capillary permeability.
    eff_lap = model._effective_lap_tm()
    plasma_alb_g_l = (
        model.fluids.plasma_albumin_g / model.fluids.plasma_l if model.fluids.plasma_l > 0.0 else 42.0
    )
    oncotic_drop = min(8.0, 0.28 * max(0.0, 34.0 - plasma_alb_g_l))
    inflam = max(sepsis_sev, 0.7 * anaph_sev)
    eff_threshold = max(8.0, model._lung_water_threshold - oncotic_drop - 5.0 * inflam)
    lap_excess = max(0.0, eff_lap - eff_threshold)
    eff_lap_excess = lap_excess + 0.14 * max(0.0, lap_excess - 6.0) ** 2
    pulm_flow = co_prev if pulmonary_flow_prev is None else pulmonary_flow_prev
    pulm_flow_gate = clamp01(pulm_flow / max(0.3, 0.20 * base_co))
    perm_filtration_rate = 0.004 * model._pbw * inflam * max(0.0, eff_lap - 8.0) * pulm_flow_gate
    max_lung_water_ml = 28.0 * model._pbw
    filtration_ml = min(
        (cfg.lung_water_filtration * model._pbw * eff_lap_excess * pulm_flow_gate + perm_filtration_rate) * dt_min,
        max(0.0, blood_volume - minimum_blood_volume),
        max(0.0, max_lung_water_ml - model.lung_water_ml),
    )
    lung_clearance_ml = (
        model.lung_water_ml * -math.expm1(-sub_dt / (cfg.lung_water_clearance_tau_hr * 3600.0))
        if model.lung_water_ml > 0.0
        else 0.0
    )
    model.lung_water_ml += filtration_ml - lung_clearance_ml
    model.fluids.shift_water(lung_clearance_ml - filtration_ml)
    if filtration_ml > 0.0:
        model.crystalloid.central_ml *= max(0.0, 1.0 - filtration_ml / (blood_volume - model.red_cell_volume_ml))
    blood_volume += lung_clearance_ml - filtration_ml
    model.blood_volume = max(BLOOD_VOLUME_MIN + ext_vol, blood_volume)
    model._update_hb_conc()
