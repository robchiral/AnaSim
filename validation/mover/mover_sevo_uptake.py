"""External MOVER validation of sevoflurane uptake and washout (SEV In -> SEV ET).

Replays 1-minute recorded inspired sevoflurane (`SEV In`) through AnaSim's frozen
`VolatilePK` model without fitting parameters, evaluating uptake, maintenance, and
emergence washout across BMI (including severe obesity), age, and sex strata.
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
    fmt_mean_ci,
    load_cohort,
    load_flowsheet_subset,
    mean_ci,
    patient_from_mover,
    utc_date,
    write_results,
)

_GAS_NAMES = ("SEV In", "SEV ET", "ISO In", "DES In", "ETN2O", "FIN20 (%)")


def _sec_from(ts: datetime, ref: datetime) -> float:
    return (ts - ref).total_seconds()


def _simulate_sevo_case(payload: dict[str, Any]) -> dict[str, Any] | None:
    row = payload["row"]
    t_s = np.asarray(payload["t_s"], dtype=float)
    fi_pct = np.asarray(payload["fi_pct"], dtype=float)
    et_pct = np.asarray(payload["et_pct"], dtype=float)

    # Identify first sustained sevoflurane onset (Fi >= 0.3%).
    on_idx = np.flatnonzero(fi_pct >= 0.3)
    if len(on_idx) == 0:
        return None
    t_on = float(t_s[on_idx[0]])
    t_end = float(t_s[-1])
    if t_end - t_on < 1800.0:
        return None

    # Replay on a 5-s step grid with piecewise-linear 1-min FiSevo.
    grid = np.arange(t_on, t_end + 1.0, 5.0)
    fi_grid = np.interp(grid, t_s, fi_pct, left=0.0, right=float(fi_pct[-1]))

    patient = patient_from_mover(row, baseline_map=90.0, baseline_hr=70.0, use_preop_labs=True)
    engine = SimulationEngine(patient, SimulationConfig(mode="awake"))
    pk = engine.pk_sevo
    # Standard intraoperative mechanical ventilation (VT = 8 mL/kg PBW, RR = 12) and
    # anesthetized cardiac index (~2.6 L/min/m²), since TV/RR are absent from MOVER Subset.
    vt_l = min(0.60, 0.008 * patient.predicted_body_weight())
    va_l_min = 12.0 * max(0.15, vt_l - engine.resp.vd_deadspace)
    co_l_min = max(2.5, 2.6 * patient.bsa)

    pred_grid = np.zeros(len(grid), dtype=float)
    for i, dt in enumerate(np.diff(grid), start=1):
        pk.step(float(dt), float(fi_grid[i - 1] / 100.0), va_l_min, co_l_min)
        pred_grid[i] = 100.0 * pk.state.p_alv

    # Score at recorded 1-min timestamps from onset onward.
    valid = (t_s >= t_on) & np.isfinite(et_pct) & (et_pct >= 0.0) & (et_pct <= 8.0)
    t_v = t_s[valid]
    obs_v = et_pct[valid]
    fi_v = fi_pct[valid]
    pred_v = np.interp(t_v, grid, pred_grid)
    diff = pred_v - obs_v

    uptake = (t_v >= t_on) & (t_v < t_on + 900.0) & (obs_v >= 0.1)
    maint = (t_v >= t_on + 900.0) & (fi_v >= 0.4) & (obs_v >= 0.4)
    # Washout: after maintenance peak, Fi drops below 0.2% while EtSevo washes out (>= 0.1%).
    last_hi = np.flatnonzero(fi_v >= 0.5)
    t_wash_start = float(t_v[last_hi[-1]]) if len(last_hi) else t_end
    washout = (t_v > t_wash_start) & (fi_v < 0.2) & (obs_v >= 0.1)

    if maint.sum() < 15:
        return None

    return {
        "log_id": row["log_id"],
        "age": float(row["age"]),
        "bmi": float(row["bmi"]),
        "sex": str(row["sex"]),
        "all_bias": float(diff[obs_v >= 0.1].mean()),
        "all_mae": float(np.abs(diff[obs_v >= 0.1]).mean()),
        "uptake_bias": float(diff[uptake].mean()) if uptake.sum() >= 3 else float("nan"),
        "uptake_mae": float(np.abs(diff[uptake]).mean()) if uptake.sum() >= 3 else float("nan"),
        "maint_bias": float(diff[maint].mean()),
        "maint_mae": float(np.abs(diff[maint]).mean()),
        "maint_rec_med": float(np.median(obs_v[maint])),
        "maint_mod_med": float(np.median(pred_v[maint])),
        "washout_bias": float(diff[washout].mean()) if washout.sum() >= 3 else float("nan"),
        "washout_mae": float(np.abs(diff[washout]).mean()) if washout.sum() >= 3 else float("nan"),
    }


def run(
    *,
    mover_dir: Path = DEFAULT_MOVER_DIR,
    cache_dir: Path = CACHE_DIR,
    output_dir: Path = RESULTS_DIR,
    max_cases: int = 500,
    workers: int = DEFAULT_WORKERS,
    seed: int = 42,
) -> dict[str, Any]:
    cohort = load_cohort(
        mover_dir,
        cache_dir,
        where="has_cpb = 0 AND has_olv = 0 AND has_lma = 0 AND asa <= 3 AND t_induction IS NOT NULL",
    )
    rng = np.random.default_rng(seed)
    # Stratified oversampling so BMI >= 35 is heavily represented alongside lean/normal/overweight.
    buckets: dict[str, list[dict[str, Any]]] = {"lt25": [], "25_30": [], "30_35": [], "ge35": []}
    for r in cohort:
        b = float(r["bmi"])
        if b < 25.0:
            buckets["lt25"].append(r)
        elif b < 30.0:
            buckets["25_30"].append(r)
        elif b < 35.0:
            buckets["30_35"].append(r)
        else:
            buckets["ge35"].append(r)

    per_bucket = max(40, (max_cases * 2) // 4)
    sampled: list[dict[str, Any]] = []
    for key in ("lt25", "25_30", "30_35", "ge35"):
        arr = buckets[key]
        perm = rng.permutation(len(arr))
        sampled.extend(arr[int(i)] for i in perm[:per_bucket])

    gas_rows = load_flowsheet_subset("Gas", [r["log_id"] for r in sampled], _GAS_NAMES, mover_dir=mover_dir)

    tasks: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for row in sampled:
        if len(tasks) >= max_cases:
            break
        lid = row["log_id"]
        t_ref = row["t_induction"]
        fi_map: dict[float, float] = {}
        et_map: dict[float, float] = {}
        bad_gas = False
        for ts, name, val in gas_rows.get(lid, []):
            try:
                v = float(val)
            except ValueError:
                continue
            if not np.isfinite(v) or v < 0:
                continue
            if name in ("ISO In", "DES In") and v >= 0.1:
                bad_gas = True
                break
            if name in ("ETN2O", "FIN20 (%)") and v >= 2.0:
                bad_gas = True
                break
            t_s = _sec_from(ts, t_ref)
            if not (-300.0 <= t_s <= 14400.0):
                continue
            if name == "SEV In" and v <= 10.0:
                fi_map[t_s] = v
            elif name == "SEV ET" and v <= 10.0:
                et_map[t_s] = v
        if bad_gas:
            exclusions["isoflurane, desflurane, or nitrous oxide exposure"] += 1
            continue
        common_t = sorted(set(fi_map) & set(et_map))
        if len(common_t) < 30:
            exclusions["less than 30 paired 1-min SEV In/ET samples"] += 1
            continue
        t_arr = np.array(common_t, dtype=float)
        if np.max(np.diff(t_arr)) > 300.0:
            exclusions["gas recording gap over 5 min"] += 1
            continue
        fi_arr = np.array([fi_map[t] for t in common_t], dtype=float)
        et_arr = np.array([et_map[t] for t in common_t], dtype=float)
        if float(fi_arr.max()) < 0.6 or float(et_arr.max()) < 0.4:
            exclusions["insufficient sevoflurane administration"] += 1
            continue
        tasks.append({"row": row, "t_s": t_arr.tolist(), "fi_pct": fi_arr.tolist(), "et_pct": et_arr.tolist()})

    results: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_simulate_sevo_case, tasks, chunksize=8):
                if res is not None:
                    results.append(res)
    else:
        for t in tasks:
            res = _simulate_sevo_case(t)
            if res is not None:
                results.append(res)

    lines = [
        "# MOVER external validation: sevoflurane uptake and washout",
        "",
        f"Generated {utc_date()}. Frozen AnaSim parameters (no refitting on MOVER).",
        "",
        "## Methods",
        "",
        "Replays 1-minute recorded inspired sevoflurane (`SEV In`) through AnaSim's `VolatilePK`",
        "model against recorded end-tidal sevoflurane (`SEV ET`) across BMI-stratified unique-MRN",
        "ASA 1-3 General Anesthesia cases without isoflurane, desflurane, or N₂O. Because tidal",
        "volume and respiratory rate are absent from the MOVER flowsheet subset, alveolar ventilation",
        "uses standard intraoperative settings (VT = 8 mL/kg PBW, RR = 12/min, PBW-scaled dead space)",
        "and anesthetized cardiac index 2.6 L/min/m². Errors are predicted minus recorded (% atm).",
        "",
        f"Screened candidates: {len(sampled)}. Retained: {len(results)}.",
        "",
        "## Maintenance and overall EtSevo error (% atm)",
        "",
        "| Subgroup | n | Recorded / Model EtSevo (%) | Maintenance bias (95% CI) | Maintenance MAE (95% CI) | Uptake 0-15m bias | Washout bias |",
        "|---|---:|---|---|---|---|---|",
    ]

    def emit(label: str, sub: list[dict[str, Any]]) -> dict[str, Any] | None:
        if len(sub) < 5:
            return None
        rec_m = float(np.median([r["maint_rec_med"] for r in sub]))
        mod_m = float(np.median([r["maint_mod_med"] for r in sub]))
        m_bias = mean_ci([r["maint_bias"] for r in sub], seed=seed)
        m_mae = mean_ci([r["maint_mae"] for r in sub], seed=seed)
        up_vals = [r["uptake_bias"] for r in sub if np.isfinite(r["uptake_bias"])]
        wa_vals = [r["washout_bias"] for r in sub if np.isfinite(r["washout_bias"])]
        up_s = fmt_mean_ci(mean_ci(up_vals, seed=seed), digits=2) if len(up_vals) >= 5 else "-"
        wa_s = fmt_mean_ci(mean_ci(wa_vals, seed=seed), digits=2) if len(wa_vals) >= 5 else "-"
        lines.append(
            f"| {label} | {len(sub)} | {rec_m:.2f} / {mod_m:.2f} | "
            f"{fmt_mean_ci(m_bias, digits=2)} | {fmt_mean_ci(m_mae, digits=2)} | {up_s} | {wa_s} |"
        )
        return {
            "n": len(sub),
            "rec_med": rec_m,
            "mod_med": mod_m,
            "maint_bias": m_bias,
            "maint_mae": m_mae,
            "uptake_bias": mean_ci(up_vals, seed=seed) if up_vals else None,
            "washout_bias": mean_ci(wa_vals, seed=seed) if wa_vals else None,
        }

    summary_groups: dict[str, Any] = {}
    for label, sub in (
        ("All patients", results),
        ("BMI under 25", [r for r in results if r["bmi"] < 25]),
        ("BMI 25-29.9", [r for r in results if 25 <= r["bmi"] < 30]),
        ("BMI 30-34.9", [r for r in results if 30 <= r["bmi"] < 35]),
        ("BMI 35 and over", [r for r in results if r["bmi"] >= 35]),
        ("Age under 45", [r for r in results if r["age"] < 45]),
        ("Age 45-64", [r for r in results if 45 <= r["age"] < 65]),
        ("Age 65 and over", [r for r in results if r["age"] >= 65]),
        ("Female", [r for r in results if r["sex"] == "female"]),
        ("Male", [r for r in results if r["sex"] == "male"]),
    ):
        stat = emit(label, sub)
        if stat is not None:
            summary_groups[label] = stat

    lines.extend(["", "## Exclusions", ""])
    for reason, cnt in sorted(exclusions.items()):
        lines.append(f"- {reason}: {cnt}")
    lines.extend(["", SOURCE])

    payload = {"generated": utc_date(), "retained": len(results), "groups": summary_groups, "exclusions": dict(exclusions)}
    write_results("mover_sevo_uptake", lines, payload, output_dir=output_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="MOVER external sevoflurane uptake validation.")
    parser.add_argument("--mover-dir", type=Path, default=DEFAULT_MOVER_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = run(
        mover_dir=args.mover_dir,
        cache_dir=args.cache,
        output_dir=args.output,
        max_cases=args.max_cases,
        workers=args.workers,
        seed=args.seed,
    )
    print(f"Wrote mover_sevo_uptake report: retained={out['retained']}")


if __name__ == "__main__":
    main()
