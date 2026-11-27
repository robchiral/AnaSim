"""External MOVER validation of intraoperative core temperature trajectories.

Mirrors `vitaldb_temperature.py` on UCI Health EPIC general anesthesia cases (converting
Fahrenheit flowsheet probes to Celsius), testing AnaSim's frozen `ThermalModel` across
30-, 60-, and 120-minute post-equilibration windows and wide US BMI strata (<22 to >=40 kg/m²).
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from anasim.physiology.thermal import ThermalModel
from validation.mover.mover_data import (
    CACHE_DIR,
    DEFAULT_MOVER_DIR,
    DEFAULT_WORKERS,
    RESULTS_DIR,
    SOURCE,
    f_to_c,
    fmt_mean_ci,
    load_cohort,
    load_flowsheet_subset,
    mean_ci,
    patient_from_mover,
    utc_date,
    write_results,
)


def _sec_from(ts: datetime, ref: datetime) -> float:
    return (ts - ref).total_seconds()


def _simulate_thermal(
    patient: Any,
    *,
    t0: float,
    elapsed_since_ind_s: float,
    fluid_ml_hr: float,
    active_warming: bool,
) -> tuple[float, float, float]:
    """Return modeled `(d30, d60, d120)` °C change from equilibrated starting temperature `t0`."""
    patient.baseline_temp = t0
    model = ThermalModel(patient)
    warmer_target = 38.0 if active_warming else 0.0
    dt = 10.0
    temp = t0
    for _ in range(max(0, round(elapsed_since_ind_s / dt))):
        model.update_depth(temp, prop_ce=0.0, mac=1.0)
        if fluid_ml_hr > 0.0:
            model.add_fluid_heat_j(fluid_ml_hr * dt / 3600.0, 22.0, temp)
        temp = model.step_temperature(dt, temp, warmer_target, perfusing=True)

    # Re-anchor at recorded equilibrated temperature while retaining redistribution state.
    temp = t0
    d30 = d60 = d120 = 0.0
    for step in range(1, 721):  # 720 * 10 s = 120 min
        model.update_depth(temp, prop_ce=0.0, mac=1.0)
        if fluid_ml_hr > 0.0:
            model.add_fluid_heat_j(fluid_ml_hr * dt / 3600.0, 22.0, temp)
        temp = model.step_temperature(dt, temp, warmer_target, perfusing=True)
        if step == 180:
            d30 = temp - t0
        elif step == 360:
            d60 = temp - t0
        elif step == 720:
            d120 = temp - t0
    return float(d30), float(d60), float(d120)


def _evaluate_case(payload: dict[str, Any]) -> dict[str, Any] | None:
    row = payload["row"]
    t_s = np.asarray(payload["t_s"], dtype=float)
    temp_c = np.asarray(payload["temp_c"], dtype=float)
    fluid_ml_hr = float(payload["fluid_ml_hr"])

    ok = np.flatnonzero(temp_c >= 35.5)
    if len(ok) == 0:
        return None
    t_eq = float(t_s[ok[0]]) + 300.0
    w0 = temp_c[(t_s >= t_eq) & (t_s < t_eq + 300.0)]
    w30 = temp_c[(t_s >= t_eq + 1800.0) & (t_s < t_eq + 2100.0)]
    w60 = temp_c[(t_s >= t_eq + 3600.0) & (t_s < t_eq + 3900.0)]
    w120 = temp_c[(t_s >= t_eq + 7200.0) & (t_s < t_eq + 7500.0)]
    if len(w0) < 3 or len(w30) < 3 or len(w60) < 3:
        return None
    t0 = float(np.median(w0))
    if not (35.5 <= t0 <= 38.0):
        return None

    obs_d30 = float(np.median(w30)) - t0
    obs_d60 = float(np.median(w60)) - t0
    obs_d120 = float(np.median(w120)) - t0 if len(w120) >= 3 else float("nan")
    elapsed_ind = float(np.clip(t_eq, 300.0, 2400.0))

    patient = patient_from_mover(row, baseline_temp=t0, use_preop_labs=False)
    unw_d30, unw_d60, unw_d120 = _simulate_thermal(
        patient, t0=t0, elapsed_since_ind_s=elapsed_ind, fluid_ml_hr=fluid_ml_hr, active_warming=False
    )
    wrm_d30, wrm_d60, wrm_d120 = _simulate_thermal(
        patient, t0=t0, elapsed_since_ind_s=elapsed_ind, fluid_ml_hr=fluid_ml_hr, active_warming=True
    )
    return {
        "log_id": row["log_id"],
        "age": float(row["age"]),
        "bmi": float(row["bmi"]),
        "approach": str(row["approach"]),
        "t0": round(t0, 2),
        "obs_d30": obs_d30,
        "obs_d60": obs_d60,
        "obs_d120": obs_d120,
        "unw_d30": unw_d30,
        "unw_d60": unw_d60,
        "unw_d120": unw_d120,
        "wrm_d30": wrm_d30,
        "wrm_d60": wrm_d60,
        "wrm_d120": wrm_d120,
    }


def run(
    *,
    mover_dir: Path = DEFAULT_MOVER_DIR,
    cache_dir: Path = CACHE_DIR,
    output_dir: Path = RESULTS_DIR,
    max_cases: int = 800,
    workers: int = DEFAULT_WORKERS,
    seed: int = 42,
) -> dict[str, Any]:
    cohort = load_cohort(
        mover_dir,
        cache_dir,
        where="has_cpb = 0 AND asa <= 3 AND t_induction IS NOT NULL",
    )
    # Require procedure duration >= 80 min so 60-min post-equilibration window is likely present.
    long_cases = [r for r in cohort if _sec_from(r["t_an_stop"], r["t_induction"]) >= 4800.0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(long_cases))
    sampled = [long_cases[int(i)] for i in perm[: max_cases * 2]]
    ids = [r["log_id"] for r in sampled]

    temp_rows = load_flowsheet_subset("Temp", ids, ("Temp",), mover_dir=mover_dir)
    vol_rows = load_flowsheet_subset("Volume", ids, ("Volume",), mover_dir=mover_dir)

    tasks: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for row in sampled:
        if len(tasks) >= max_cases:
            break
        lid = row["log_id"]
        t_ind = row["t_induction"]
        pts: list[tuple[float, float]] = []
        for ts, _name, val in temp_rows.get(lid, []):
            try:
                raw_v = float(val)
            except ValueError:
                continue
            c = f_to_c(raw_v)
            if c is None or not (34.0 <= c <= 39.0):
                continue
            t_s = _sec_from(ts, t_ind)
            if 0.0 <= t_s <= 14400.0:
                pts.append((t_s, c))
        if len(pts) < 40:
            exclusions["insufficient valid temperature readings"] += 1
            continue
        pts.sort()
        t_arr = np.array([p[0] for p in pts], dtype=float)
        c_arr = np.array([p[1] for p in pts], dtype=float)
        if float(t_arr[-1] - t_arr[0]) < 4200.0:
            exclusions["temperature span under 70 min"] += 1
            continue

        # Approximate crystalloid infusion rate from intraop Volume records (capped to plausible range).
        vol_sum = 0.0
        for _ts, _name, val in vol_rows.get(lid, []):
            try:
                v = float(val)
            except ValueError:
                continue
            if 0.0 < v <= 2000.0:
                vol_sum += v
        dur_h = max(1.0, _sec_from(row["t_an_stop"], t_ind) / 3600.0)
        fluid_ml_hr = float(np.clip(vol_sum / dur_h, 0.0, 2000.0)) if vol_sum > 0 else 400.0
        tasks.append({"row": row, "t_s": t_arr.tolist(), "temp_c": c_arr.tolist(), "fluid_ml_hr": fluid_ml_hr})

    results: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_evaluate_case, tasks, chunksize=8):
                if res is not None:
                    results.append(res)
    else:
        for t in tasks:
            res = _evaluate_case(t)
            if res is not None:
                results.append(res)

    lines = [
        "# MOVER external validation: intraoperative core temperature",
        "",
        f"Generated {utc_date()}. Frozen AnaSim parameters (no refitting on MOVER).",
        "",
        "## Methods",
        "",
        "Compares recorded `Temp` (converted from °F to °C) changes 30, 60, and 120 minutes after",
        "probe equilibration (5 min after first reading >= 35.5 °C) across unique-MRN non-CPB",
        "General Anesthesia cases against AnaSim's `ThermalModel` both without and with active",
        "forced-air warming (38 °C).",
        "",
        f"Screened long-case candidates: {len(sampled)}. Retained with >= 60 min curve: {len(results)}.",
        "",
        "## 60-minute post-equilibration core temperature change (°C)",
        "",
        "| Subgroup | n | Recorded d60 (95% CI) | Unwarmed model d60 | Forced-air model d60 | Recorded d30 | Recorded d120 (n) |",
        "|---|---:|---|---|---|---|---|",
    ]

    summary_groups: dict[str, Any] = {}
    groups = [
        ("All cases", results),
        ("BMI under 22", [r for r in results if r["bmi"] < 22.0]),
        ("BMI 22-26.9", [r for r in results if 22.0 <= r["bmi"] < 27.0]),
        ("BMI 27-31.9", [r for r in results if 27.0 <= r["bmi"] < 32.0]),
        ("BMI 32-39.9", [r for r in results if 32.0 <= r["bmi"] < 40.0]),
        ("BMI 40 and over", [r for r in results if r["bmi"] >= 40.0]),
        ("Age under 50", [r for r in results if r["age"] < 50.0]),
        ("Age 50-69", [r for r in results if 50.0 <= r["age"] < 70.0]),
        ("Age 70 and over", [r for r in results if r["age"] >= 70.0]),
        ("Open", [r for r in results if r["approach"] == "Open"]),
        ("Laparoscopic or robotic", [r for r in results if r["approach"] == "Laparoscopic or robotic"]),
    ]
    for label, sub in groups:
        if len(sub) < 5:
            continue
        ci60 = mean_ci([r["obs_d60"] for r in sub], seed=seed)
        unw60 = float(np.mean([r["unw_d60"] for r in sub]))
        wrm60 = float(np.mean([r["wrm_d60"] for r in sub]))
        obs30 = float(np.mean([r["obs_d30"] for r in sub]))
        d120_vals = [r["obs_d120"] for r in sub if np.isfinite(r["obs_d120"])]
        d120_cell = f"{np.mean(d120_vals):+.2f} ({len(d120_vals)})" if len(d120_vals) >= 5 else "-"
        lines.append(
            f"| {label} | {len(sub)} | {fmt_mean_ci(ci60, digits=2)} | "
            f"{unw60:+.2f} | {wrm60:+.2f} | {obs30:+.2f} | {d120_cell} |"
        )
        summary_groups[label] = {
            "n": len(sub),
            "obs_d60_ci": ci60,
            "unwarmed_d60": unw60,
            "warmed_d60": wrm60,
            "obs_d30": obs30,
            "obs_d120": float(np.mean(d120_vals)) if len(d120_vals) >= 5 else None,
            "n_d120": len(d120_vals),
        }

    if results:
        inside = sum(
            1
            for r in results
            if min(r["unw_d60"], r["wrm_d60"]) - 0.1 <= r["obs_d60"] <= max(r["unw_d60"], r["wrm_d60"]) + 0.1
        )
        lines.extend(
            [
                "",
                f"Median equilibrated starting core temperature: {np.median([r['t0'] for r in results]):.2f} °C. "
                f"Share within [unwarmed, forced-air warmed] ± 0.1 °C at 60 min: "
                f"{100.0 * inside / len(results):.1f}% ({inside}/{len(results)}).",
            ]
        )

    lines.extend(["", "## Exclusions", ""])
    for reason, cnt in sorted(exclusions.items()):
        lines.append(f"- {reason}: {cnt}")
    lines.extend(["", SOURCE])

    payload = {"generated": utc_date(), "retained": len(results), "groups": summary_groups, "exclusions": dict(exclusions)}
    write_results("mover_temperature", lines, payload, output_dir=output_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="MOVER external core temperature validation.")
    parser.add_argument("--mover-dir", type=Path, default=DEFAULT_MOVER_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-cases", type=int, default=800)
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
    print(f"Wrote mover_temperature report: retained={out['retained']}")


if __name__ == "__main__":
    main()
