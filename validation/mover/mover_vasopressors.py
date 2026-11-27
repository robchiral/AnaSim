"""External MOVER validation of acute isolated vasopressor, beta-blocker, and reversal responses.

Uses 1-minute invasive arterial MAP (`MAP-ART A-line`) and 1-minute ECG HR during settled
maintenance anesthesia to evaluate isolated boluses of phenylephrine, ephedrine,
norepinephrine, esmolol, labetalol, glycopyrrolate alone, and neostigmine + glycopyrrolate,
plus isolated norepinephrine infusion starts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from validation.mover.mover_data import (
    CACHE_DIR,
    DEFAULT_MOVER_DIR,
    DEFAULT_WORKERS,
    RESULTS_DIR,
    SOURCE,
    VASOACTIVE_DRUGS,
    fmt_mean_ci,
    fmt_med_iqr,
    load_cohort,
    load_flowsheet_subset,
    load_meds_for_logs,
    mean_ci,
    median_iqr,
    patient_from_mover,
    utc_date,
    write_results,
)
from validation.vitaldb.vitaldb_replay import step_headless

_COMPETING_DRUGS = VASOACTIVE_DRUGS | frozenset(
    {"propofol", "etomidate", "ketamine", "neostigmine", "remifentanil", "fentanyl"}
)
_INFUSION_PRESSORS = frozenset({"phenylephrine", "norepinephrine", "epinephrine", "vasopressin"})


def _sec_from(ts: datetime, ref: datetime) -> float:
    return (ts - ref).total_seconds()


def _settle_anesthetized_engine(
    row: dict[str, Any],
    *,
    target_map: float,
    target_hr: float,
    et_sevo_pct: float,
) -> SimulationEngine:
    """Create a sevoflurane-anesthetized engine settled at `(target_map, target_hr)` without unrecorded remifentanil."""
    mac_est = max(0.2, min(1.5, et_sevo_pct / 2.0))
    awake_map = float(np.clip(target_map / max(0.68, 1.0 - 0.18 * mac_est), 70.0, 150.0))
    awake_hr = float(np.clip(target_hr + 2.0 * mac_est, 42.0, 125.0))

    def build_and_settle(b_map: float, b_hr: float) -> SimulationEngine:
        patient = patient_from_mover(row, baseline_map=b_map, baseline_hr=b_hr, use_preop_labs=True)
        engine = SimulationEngine(
            patient,
            SimulationConfig(
                mode="steady_state",
                maint_type="balanced",
                concomitant_opioids=True,
                rng_seed=1,
                maintenance_fluid_ml_hr=150.0,
            ),
        )
        engine.start()
        # Clear steady_state's default remifentanil infusion/state so volatile-only cases stay pure sevoflurane.
        engine.infusion_rates["remi"] = 0.0
        remi_pk = engine.pk["remi"]
        remi_pk.state.c1 = remi_pk.state.c2 = remi_pk.state.c3 = remi_pk.state.ce = 0.0
        engine.set_airway_mode("ETT")
        vt = min(0.6, 0.008 * patient.predicted_body_weight())
        engine.set_vent_settings(rr=12, vt=vt, peep=5.0, ie="1:2", mode="VCV")
        engine.set_vent_power(True)
        engine.set_fgf(2.0, 2.0)
        engine.set_vaporizer(max(0.3, min(3.5, et_sevo_pct * 1.15)))
        for _ in range(72):
            step_headless(engine, 5.0)
        return engine

    eng = build_and_settle(awake_map, awake_hr)
    map_err = target_map - eng.state.map
    hr_err = target_hr - eng.state.hr
    if abs(map_err) > 2.5 or abs(hr_err) > 2.5:
        awake_map2 = float(np.clip(awake_map + map_err * 1.25, 65.0, 170.0))
        awake_hr2 = float(np.clip(awake_hr + hr_err * 1.05, 40.0, 130.0))
        eng = build_and_settle(awake_map2, awake_hr2)
    return eng


def _simulate_bolus_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    row = payload["row"]
    arm = payload["arm"]
    base_map = float(payload["base_map"])
    base_hr = float(payload["base_hr"])
    et_sevo = float(payload["et_sevo"])
    boluses: list[tuple[str, float]] = payload["boluses"]
    inf_step_nore_ug_min = float(payload.get("inf_step_nore_ug_min", 0.0))

    engine = _settle_anesthetized_engine(
        row,
        target_map=base_map,
        target_hr=base_hr,
        et_sevo_pct=et_sevo,
    )
    m0 = float(engine.state.map)
    h0 = float(engine.state.hr)

    # Administer bolus(es) or infusion step at t = 0 and step 1 s for 8 min (480 s).
    for generic_name, amount in boluses:
        engine.give_drug_bolus(generic_name, amount)
    if inf_step_nore_ug_min > 0.0:
        engine.infusion_rates["nore"] = inf_step_nore_ug_min / 60.0

    # Sample at 1-minute marks (+1..+8 min) to match 1-minute MOVER resolution.
    mod_dmap = np.empty(8, dtype=float)
    mod_dhr = np.empty(8, dtype=float)
    for minute in range(1, 9):
        for _ in range(60):
            step_headless(engine, 1.0)
        mod_dmap[minute - 1] = float(engine.state.map) - m0
        mod_dhr[minute - 1] = float(engine.state.hr) - h0

    rec_dmap = np.asarray(payload["rec_dmap"], dtype=float)
    rec_dhr = np.asarray(payload["rec_dhr"], dtype=float)

    def _safe_peak(arr: np.ndarray, sl: slice, mode: str) -> float:
        part = arr[sl]
        part = part[np.isfinite(part)]
        if len(part) == 0:
            return float("nan")
        if mode == "max":
            return float(part.max())
        if mode == "min":
            return float(part.min())
        return float(part.mean())

    if arm in ("esmolol", "labetalol"):
        rec_p_map = _safe_peak(rec_dmap, slice(0, 6), "min")
        mod_p_map = _safe_peak(mod_dmap, slice(0, 6), "min")
        rec_p_hr = _safe_peak(rec_dhr, slice(0, 6), "min")
        mod_p_hr = _safe_peak(mod_dhr, slice(0, 6), "min")
    elif arm in ("neo_glyco", "glyco_alone"):
        rec_p_map = _safe_peak(rec_dmap, slice(0, 5), "mean")
        mod_p_map = _safe_peak(mod_dmap, slice(0, 5), "mean")
        rec_p_hr = _safe_peak(rec_dhr, slice(0, 3), "mean")  # early +1..3 min
        mod_p_hr = _safe_peak(mod_dhr, slice(0, 3), "mean")
    else:
        rec_p_map = _safe_peak(rec_dmap, slice(0, 4), "max")
        mod_p_map = _safe_peak(mod_dmap, slice(0, 4), "max")
        rec_p_hr = _safe_peak(rec_dhr, slice(0, 5), "max" if arm == "ephedrine" else "min")
        mod_p_hr = _safe_peak(mod_dhr, slice(0, 5), "max" if arm == "ephedrine" else "min")

    rec_late_map = _safe_peak(rec_dmap, slice(5, 8), "mean")
    mod_late_map = _safe_peak(mod_dmap, slice(5, 8), "mean")
    rec_late_hr = _safe_peak(rec_dhr, slice(3, 8), "mean")
    mod_late_hr = _safe_peak(mod_dhr, slice(3, 8), "mean")

    return {
        "log_id": row["log_id"],
        "arm": arm,
        "dose_label": payload["dose_label"],
        "dose_val": float(payload["dose_val"]),
        "is_first_dose": bool(payload.get("is_first_dose", True)),
        "base_map": base_map,
        "base_hr": base_hr,
        "rec_peak_dmap": rec_p_map,
        "mod_peak_dmap": mod_p_map,
        "rec_peak_dhr": rec_p_hr,
        "mod_peak_dhr": mod_p_hr,
        "rec_late_dmap": rec_late_map,
        "mod_late_dmap": mod_late_map,
        "rec_late_dhr": rec_late_hr,
        "mod_late_dhr": mod_late_hr,
    }


def _has_active_pressor_infusion(meds: list[dict[str, Any]], t0: datetime) -> bool:
    for drug in _INFUSION_PRESSORS:
        inf_ev = [m for m in meds if m["drug"] == drug and m["kind"] == "infusion"]
        if not inf_ev:
            continue
        inf_ev.sort(key=lambda x: x["ts"])
        rate_before = 0.0
        for m in inf_ev:
            dt_s = _sec_from(m["ts"], t0)
            if dt_s <= -900.0:
                rate_before = float(m["val"])
            elif -900.0 < dt_s <= 480.0:
                if float(m["val"]) > 0.0:
                    return True
        if rate_before > 0.0:
            return True
    return False


def _extract_minute_series(
    rows: list[tuple[datetime, str, str]],
    t0: datetime,
    *,
    lo: float,
    hi: float,
    prefer_name: str | None = None,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for ts, name, val in rows:
        dt_s = _sec_from(ts, t0)
        if not (-240.0 <= dt_s <= 540.0):
            continue
        minute = round(dt_s / 60.0)
        try:
            v = float(val)
        except ValueError:
            continue
        if lo <= v <= hi and (minute not in out or (prefer_name is not None and name == prefer_name)):
            out[minute] = v
    return out


def run(
    *,
    mover_dir: Path = DEFAULT_MOVER_DIR,
    cache_dir: Path = CACHE_DIR,
    output_dir: Path = RESULTS_DIR,
    max_per_arm: int = 180,
    workers: int = DEFAULT_WORKERS,
    seed: int = 42,
) -> dict[str, Any]:
    cohort = load_cohort(
        mover_dir,
        cache_dir,
        where="has_cpb = 0 AND has_olv = 0 AND t_induction IS NOT NULL",
    )
    row_by_id = {r["log_id"]: r for r in cohort}
    meds_by_log = load_meds_for_logs([r["log_id"] for r in cohort], mover_dir, cache_dir)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cohort))

    # Candidate selection: at most ONE isolated event per unique patient per drug arm.
    cands_by_arm: dict[str, list[dict[str, Any]]] = {
        "phenylephrine": [],
        "ephedrine": [],
        "norepi_bolus": [],
        "norepi_inf": [],
        "esmolol": [],
        "labetalol": [],
        "neo_glyco": [],
        "glyco_alone": [],
    }
    arm_overfetch = max_per_arm * 3

    for idx in order:
        row = cohort[int(idx)]
        lid = row["log_id"]
        t_ind = row["t_induction"]
        t_emerg = row["t_emergence"] or row["t_an_stop"]
        t_extub = row["t_extubation"] or row["t_an_stop"]
        meds = meds_by_log.get(lid, [])
        if not meds:
            continue
        has_aline = bool(row["has_aline"])

        # Index timestamps for fast isolation check.
        competing_ts = [
            (m["ts"], m["drug"], m["kind"], float(m["val"]))
            for m in meds
            if m["drug"] in _COMPETING_DRUGS and (m["kind"] == "bolus" or m["val"] > 0)
        ]

        def isolated(
            t0: datetime,
            allow_drugs: set[str],
            window_s: float = 300.0,
            *,
            _competing: list[tuple[datetime, str, str, float]] = competing_ts,
        ) -> bool:
            for ts, d, k, _v in _competing:
                dt = abs(_sec_from(ts, t0))
                if dt > window_s:
                    continue
                if dt <= 60.0 and d in allow_drugs and k == "bolus":
                    continue
                return False
            return True

        # 1. Arterial-line vasopressor / beta-blocker boluses
        if has_aline:
            for m in meds:
                if m["kind"] != "bolus":
                    continue
                t0 = m["ts"]
                if _sec_from(t0, t_ind) < 900.0 or _sec_from(t_emerg, t0) < 600.0:
                    continue
                d, v = m["drug"], float(m["val"])
                if d == "phenylephrine" and len(cands_by_arm["phenylephrine"]) < arm_overfetch:
                    if v in (50.0, 100.0, 200.0) and not _has_active_pressor_infusion(meds, t0) and isolated(t0, {"phenylephrine"}):
                        cands_by_arm["phenylephrine"].append(
                            {
                                "row": row,
                                "arm": "phenylephrine",
                                "t0": t0,
                                "boluses": [("Phenylephrine", v)],
                                "dose_val": v,
                                "dose_label": f"{int(v)} mcg",
                                "require_aline": True,
                            }
                        )
                        break
                elif d == "ephedrine" and len(cands_by_arm["ephedrine"]) < arm_overfetch:
                    if v in (5.0, 10.0) and not _has_active_pressor_infusion(meds, t0) and isolated(t0, {"ephedrine"}):
                        prior_eph = any(
                            pm["drug"] == "ephedrine" and pm["kind"] == "bolus" and pm["ts"] < t0
                            for pm in meds
                        )
                        cands_by_arm["ephedrine"].append(
                            {
                                "row": row,
                                "arm": "ephedrine",
                                "t0": t0,
                                "boluses": [("Ephedrine", v)],
                                "dose_val": v,
                                "dose_label": f"{int(v)} mg",
                                "is_first_dose": not prior_eph,
                                "require_aline": True,
                            }
                        )
                        break
                elif d == "norepinephrine" and len(cands_by_arm["norepi_bolus"]) < arm_overfetch:
                    if 4.0 <= v <= 20.0 and not _has_active_pressor_infusion(meds, t0) and isolated(t0, {"norepinephrine"}):
                        cands_by_arm["norepi_bolus"].append(
                            {
                                "row": row,
                                "arm": "norepi_bolus",
                                "t0": t0,
                                "boluses": [("Norepinephrine", v)],
                                "dose_val": v,
                                "dose_label": f"{v:.0f} mcg",
                                "require_aline": True,
                            }
                        )
                        break
                elif d == "esmolol" and len(cands_by_arm["esmolol"]) < arm_overfetch:
                    if 10.0 <= v <= 50.0 and not _has_active_pressor_infusion(meds, t0) and isolated(t0, {"esmolol"}):
                        cands_by_arm["esmolol"].append(
                            {
                                "row": row,
                                "arm": "esmolol",
                                "t0": t0,
                                "boluses": [("Esmolol", v)],
                                "dose_val": v,
                                "dose_label": f"{v:.0f} mg",
                                "require_aline": True,
                            }
                        )
                        break
                elif d == "labetalol" and len(cands_by_arm["labetalol"]) < arm_overfetch:
                    if 5.0 <= v <= 20.0 and not _has_active_pressor_infusion(meds, t0) and isolated(t0, {"labetalol"}):
                        cands_by_arm["labetalol"].append(
                            {
                                "row": row,
                                "arm": "labetalol",
                                "t0": t0,
                                "boluses": [("Labetalol", v)],
                                "dose_val": v,
                                "dose_label": f"{v:.0f} mg",
                                "require_aline": True,
                            }
                        )
                        break

            # Isolated norepinephrine infusion starts (`New Bag` or first positive rate from 0 to 2-12 mcg/min)
            if len(cands_by_arm["norepi_inf"]) < arm_overfetch:
                nore_inf = [
                    m for m in meds
                    if m["drug"] == "norepinephrine" and m["kind"] == "infusion" and float(m["val"]) > 0
                ]
                if nore_inf:
                    m0 = nore_inf[0]
                    t0 = m0["ts"]
                    rate = float(m0["val"])
                    if m0["unit"] == "mcg/kg/min":
                        rate *= float(row["weight_kg"])
                    # Allow competing boluses outside [-3, +3] min (and prior infusions) so real ICU/high-risk starts are captured.
                    no_near_bolus = all(
                        abs(_sec_from(cm["ts"], t0)) > 180.0
                        for cm in meds
                        if cm["kind"] == "bolus" and cm["drug"] in _COMPETING_DRUGS
                    )
                    if (
                        2.0 <= rate <= 12.0
                        and _sec_from(t0, t_ind) >= 900.0
                        and _sec_from(t_emerg, t0) >= 480.0
                        and no_near_bolus
                    ):
                        cands_by_arm["norepi_inf"].append(
                            {
                                "row": row,
                                "arm": "norepi_inf",
                                "t0": t0,
                                "boluses": [],
                                "inf_step_nore_ug_min": rate,
                                "dose_val": rate,
                                "dose_label": f"{rate:.1f} mcg/min",
                                "require_aline": True,
                            }
                        )

        # 2. Neostigmine + Glycopyrrolate co-administration vs Glycopyrrolate alone (HR across all GA)
        if len(cands_by_arm["neo_glyco"]) < arm_overfetch:
            neos = [m for m in meds if m["drug"] == "neostigmine" and m["kind"] == "bolus"]
            glycos = [m for m in meds if m["drug"] == "glycopyrrolate" and m["kind"] == "bolus"]
            for nm in neos:
                t0 = nm["ts"]
                nv = float(nm["val"])
                if not (2.0 <= nv <= 5.0) or _sec_from(t0, t_ind) < 1200.0 or _sec_from(t_extub, t0) < 300.0:
                    continue
                paired = [gm for gm in glycos if abs(_sec_from(gm["ts"], t0)) <= 60.0 and 0.3 <= float(gm["val"]) <= 1.0]
                if len(paired) == 1 and isolated(t0, {"neostigmine", "glycopyrrolate"}):
                    gv = float(paired[0]["val"])
                    cands_by_arm["neo_glyco"].append(
                        {
                            "row": row,
                            "arm": "neo_glyco",
                            "t0": t0,
                            "boluses": [("Neostigmine", nv), ("Glycopyrrolate", gv)],
                            "dose_val": nv,
                            "dose_label": f"neo {nv:.1f} + glyco {gv:.1f} mg",
                            "require_aline": False,
                        }
                    )
                    break

        if len(cands_by_arm["glyco_alone"]) < arm_overfetch:
            glycos = [m for m in meds if m["drug"] == "glycopyrrolate" and m["kind"] == "bolus"]
            for gm in glycos:
                t0 = gm["ts"]
                gv = float(gm["val"])
                if (
                    0.15 <= gv <= 0.45
                    and _sec_from(t0, t_ind) >= 900.0
                    and _sec_from(t_emerg, t0) >= 600.0
                    and isolated(t0, {"glycopyrrolate"})
                ):
                    cands_by_arm["glyco_alone"].append(
                        {
                            "row": row,
                            "arm": "glyco_alone",
                            "t0": t0,
                            "boluses": [("Glycopyrrolate", gv)],
                            "dose_val": gv,
                            "dose_label": f"{gv:.1f} mg",
                            "require_aline": False,
                        }
                    )
                    break

    all_cands = [c for lst in cands_by_arm.values() for c in lst]
    cand_ids = sorted({c["row"]["log_id"] for c in all_cands})
    bp_rows = load_flowsheet_subset(
        "BP",
        cand_ids,
        ("MAP-ART A-line", "Arterial Line MAP (ART)", "NIBP - MAP"),
        mover_dir=mover_dir,
    )
    hr_rows = load_flowsheet_subset("HR", cand_ids, ("Heart Rate", "Pulse"), mover_dir=mover_dir)
    gas_rows = load_flowsheet_subset("Gas", cand_ids, ("SEV ET",), mover_dir=mover_dir)

    tasks: list[dict[str, Any]] = []
    arm_counts: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()

    for c in all_cands:
        arm = c["arm"]
        if arm_counts[arm] >= max_per_arm:
            continue
        lid = c["row"]["log_id"]
        t0 = c["t0"]
        hr_min = _extract_minute_series(hr_rows.get(lid, []), t0, lo=35.0, hi=170.0, prefer_name="Heart Rate")
        pre_hr_vals = [hr_min[m] for m in (-3, -2, -1) if m in hr_min]
        if len(pre_hr_vals) < 2 or max(pre_hr_vals) - min(pre_hr_vals) > 10.0:
            exclusions["unstable or missing pre-event HR"] += 1
            continue
        post_hr = [hr_min.get(m, float("nan")) for m in range(1, 9)]
        if sum(np.isfinite(post_hr[:6])) < 5:
            exclusions["incomplete post-event HR"] += 1
            continue
        base_hr = float(np.median(pre_hr_vals))

        if c["require_aline"]:
            bp_min = _extract_minute_series(
                [r for r in bp_rows.get(lid, []) if "ART" in r[1] or "A-line" in r[1]],
                t0,
                lo=35.0,
                hi=170.0,
                prefer_name="MAP-ART A-line",
            )
            pre_map_vals = [bp_min[m] for m in (-3, -2, -1) if m in bp_min]
            if len(pre_map_vals) < 2 or max(pre_map_vals) - min(pre_map_vals) > 10.0:
                exclusions["unstable or missing pre-event arterial MAP"] += 1
                continue
            post_map = [bp_min.get(m, float("nan")) for m in range(1, 9)]
            if sum(np.isfinite(post_map[:6])) < 5:
                exclusions["incomplete post-event arterial MAP"] += 1
                continue
            base_map = float(np.median(pre_map_vals))
        else:
            bp_min = _extract_minute_series(bp_rows.get(lid, []), t0, lo=35.0, hi=170.0)
            pre_map_vals = [bp_min[m] for m in (-3, -2, -1, 0) if m in bp_min]
            base_map = float(np.median(pre_map_vals)) if pre_map_vals else 80.0
            post_map = [bp_min.get(m, float("nan")) for m in range(1, 9)]

        sev_min = _extract_minute_series(gas_rows.get(lid, []), t0, lo=0.0, hi=8.0)
        sev_vals = [sev_min[m] for m in (-3, -2, -1, 0) if m in sev_min]
        et_sevo = float(np.median(sev_vals)) if sev_vals else 1.5
        # For neostigmine + glycopyrrolate, exclude active emergence washout (>0.4% EtSevo drop)
        # so lightening-induced tachycardia does not confound the cardiac muscarinic comparison.
        if arm == "neo_glyco" and sev_vals:
            post_sev = [sev_min[m] for m in (4, 5, 6) if m in sev_min]
            if post_sev and (et_sevo - float(np.median(post_sev))) > 0.4:
                exclusions["active volatile washout during neostigmine reversal"] += 1
                continue
        if et_sevo < 0.2:
            et_sevo = 1.2

        tasks.append(
            {
                "row": row_by_id[lid],
                "arm": arm,
                "boluses": c["boluses"],
                "inf_step_nore_ug_min": c.get("inf_step_nore_ug_min", 0.0),
                "dose_val": c["dose_val"],
                "dose_label": c["dose_label"],
                "is_first_dose": c.get("is_first_dose", True),
                "base_map": base_map,
                "base_hr": base_hr,
                "et_sevo": et_sevo,
                "rec_dmap": [v - base_map if np.isfinite(v) else float("nan") for v in post_map],
                "rec_dhr": [v - base_hr if np.isfinite(v) else float("nan") for v in post_hr],
            }
        )
        arm_counts[arm] += 1

    results: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_simulate_bolus_event, tasks, chunksize=4):
                if res is not None:
                    results.append(res)
    else:
        for t in tasks:
            res = _simulate_bolus_event(t)
            if res is not None:
                results.append(res)

    lines = [
        "# MOVER external validation: acute vasopressor, beta-blocker, and reversal responses",
        "",
        f"Generated {utc_date()}. Frozen AnaSim parameters (no refitting on MOVER).",
        "",
        "## Methods",
        "",
        "Screens settled intraoperative maintenance episodes (>= 15 min after Induction, no competing",
        "vasoactive, autonomic, or anesthetic bolus within ±5 min, stable pre-event baseline over",
        "-3 to -1 min) with 1-minute invasive arterial MAP (`MAP-ART A-line`) for vasopressors and",
        "beta-blockers, and 1-minute ECG HR across all General Anesthesia cases for neostigmine +",
        "glycopyrrolate versus glycopyrrolate alone. At most one isolated event per unique patient per",
        "drug arm is evaluated.",
        "",
        "## Acute response summary (recorded vs model)",
        "",
        "| Arm | n | Median dose | Pre-event MAP / HR | Peak dMAP rec / mod (mmHg) | dMAP err (95% CI) | Peak/early dHR rec / mod (bpm) | Late dHR +4-8m rec / mod (bpm) |",
        "|---|---:|---|---|---|---|---|---|",
    ]

    arm_meta = [
        ("phenylephrine", "Phenylephrine bolus (all)"),
        ("ephedrine", "Ephedrine bolus (all)"),
        ("norepi_bolus", "Norepinephrine bolus"),
        ("norepi_inf", "Norepinephrine infusion start"),
        ("esmolol", "Esmolol bolus"),
        ("labetalol", "Labetalol bolus"),
        ("neo_glyco", "Neostigmine + Glycopyrrolate"),
        ("glyco_alone", "Glycopyrrolate alone"),
    ]
    summary_arms: dict[str, Any] = {}

    def emit_row(label: str, sub: list[dict[str, Any]], key: str) -> None:
        if len(sub) < 3:
            return
        d_med = float(np.median([r["dose_val"] for r in sub]))
        b_map = float(np.median([r["base_map"] for r in sub]))
        b_hr = float(np.median([r["base_hr"] for r in sub]))
        rec_m = [r["rec_peak_dmap"] for r in sub if np.isfinite(r["rec_peak_dmap"])]
        mod_m = [r["mod_peak_dmap"] for r in sub if np.isfinite(r["rec_peak_dmap"])]
        err_m = [r["mod_peak_dmap"] - r["rec_peak_dmap"] for r in sub if np.isfinite(r["rec_peak_dmap"])]
        rec_h = [r["rec_peak_dhr"] for r in sub if np.isfinite(r["rec_peak_dhr"])]
        mod_h = [r["mod_peak_dhr"] for r in sub if np.isfinite(r["rec_peak_dhr"])]
        rec_lh = [r["rec_late_dhr"] for r in sub if np.isfinite(r["rec_late_dhr"])]
        mod_lh = [r["mod_late_dhr"] for r in sub if np.isfinite(r["rec_late_dhr"])]
        m_cell = f"{np.median(rec_m):+.1f} / {np.median(mod_m):+.1f}" if rec_m else "-"
        e_cell = fmt_mean_ci(mean_ci(err_m)) if err_m else "-"
        h_cell = f"{np.median(rec_h):+.1f} / {np.median(mod_h):+.1f}" if rec_h else "-"
        lh_cell = f"{np.median(rec_lh):+.1f} / {np.median(mod_lh):+.1f}" if rec_lh else "-"
        lines.append(
            f"| {label} | {len(sub)} | {d_med:.1f} | {b_map:.0f} / {b_hr:.0f} | "
            f"{m_cell} | {e_cell} | {h_cell} | {lh_cell} |"
        )
        summary_arms[key] = {
            "n": len(sub),
            "median_dose": d_med,
            "rec_peak_dmap_med": float(np.median(rec_m)) if rec_m else None,
            "mod_peak_dmap_med": float(np.median(mod_m)) if mod_m else None,
            "dmap_err_ci": mean_ci(err_m) if err_m else None,
            "rec_peak_dhr_med": float(np.median(rec_h)) if rec_h else None,
            "mod_peak_dhr_med": float(np.median(mod_h)) if mod_h else None,
            "rec_late_dhr_med": float(np.median(rec_lh)) if rec_lh else None,
            "mod_late_dhr_med": float(np.median(mod_lh)) if mod_lh else None,
        }

    for arm_key, title in arm_meta:
        sub = [r for r in results if r["arm"] == arm_key]
        emit_row(title, sub, arm_key)
        if arm_key == "phenylephrine":
            for dose in (50.0, 100.0, 200.0):
                emit_row(
                    f"  Phenylephrine {int(dose)} mcg",
                    [r for r in sub if r["dose_val"] == dose],
                    f"phenylephrine_{int(dose)}",
                )
        elif arm_key == "ephedrine":
            for dose in (5.0, 10.0):
                emit_row(
                    f"  Ephedrine {int(dose)} mg (first dose)",
                    [r for r in sub if r["dose_val"] == dose and r["is_first_dose"]],
                    f"ephedrine_{int(dose)}_first",
                )
                emit_row(
                    f"  Ephedrine {int(dose)} mg (repeat dose)",
                    [r for r in sub if r["dose_val"] == dose and not r["is_first_dose"]],
                    f"ephedrine_{int(dose)}_repeat",
                )

    # Sustained +6-8 min MAP comparison for phenylephrine vs ephedrine
    lines.extend(["", "## Sustained +6 to +8 min dMAP (phenylephrine vs ephedrine)", ""])
    for arm_key, label in (("phenylephrine", "Phenylephrine"), ("ephedrine", "Ephedrine")):
        sub = [r for r in results if r["arm"] == arm_key and np.isfinite(r["rec_late_dmap"])]
        if sub:
            rec_s = fmt_med_iqr(median_iqr([r["rec_late_dmap"] for r in sub]), suffix=" mmHg")
            mod_s = fmt_med_iqr(median_iqr([r["mod_late_dmap"] for r in sub]), suffix=" mmHg")
            lines.append(f"- **{label}** (n={len(sub)}): recorded {rec_s} vs model {mod_s}")

    lines.extend(["", "## Exclusions", ""])
    for reason, cnt in sorted(exclusions.items()):
        lines.append(f"- {reason}: {cnt}")
    lines.extend(["", SOURCE])

    payload = {
        "generated": utc_date(),
        "arms": summary_arms,
        "exclusions": dict(exclusions),
    }
    write_results("mover_vasopressors", lines, payload, output_dir=output_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="MOVER external acute vasopressor/reversal validation.")
    parser.add_argument("--mover-dir", type=Path, default=DEFAULT_MOVER_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-per-arm", type=int, default=180)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = run(
        mover_dir=args.mover_dir,
        cache_dir=args.cache,
        output_dir=args.output,
        max_per_arm=args.max_per_arm,
        workers=args.workers,
        seed=args.seed,
    )
    print(f"Wrote mover_vasopressors report: arms={list(out['arms'])}")


if __name__ == "__main__":
    main()
