"""Compare intraoperative core temperature trajectories against VitalDB Solar8000/BT tracks.

Because SNUH places esophageal/nasopharyngeal probes after induction and routinely uses
unrecorded forced-air warmers, each retained TIVA/sevoflurane case is anchored after probe
equilibration (5 minutes after first reading >= 35.5 C) and compared over the next 30 and
60 minutes against AnaSim's ThermalModel both without and with forced-air warming.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anasim.physiology.thermal import ThermalModel  # noqa: E402
from validation.vitaldb import vitaldb_data  # noqa: E402
from validation.vitaldb.vitaldb_data import (  # noqa: E402
    CACHE,
    DEFAULT_WORKERS,
    Excluded,
    amount,
    domain_cases,
    numeric_track,
    patient_from_case,
)
from validation.vitaldb.vitaldb_replay import (  # noqa: E402
    mean_ci,
    report_footer,
    report_header,
    run_hemo_pool,
)


def _simulate_thermal_curve(
    patient: Any,
    *,
    t0: float,
    elapsed_since_ind_s: float,
    fluid_ml_hr: float,
    active_warming: bool,
) -> tuple[float, float]:
    """Advance ThermalModel from induction through t_eq + 60 min; return (d30, d60) from t_eq."""
    patient.baseline_temp = t0
    model = ThermalModel(patient)
    warmer_target = 38.0 if active_warming else 0.0
    dt = 10.0
    pre_steps = max(0, round(elapsed_since_ind_s / dt))
    temp = t0
    for _ in range(pre_steps):
        model.update_depth(temp, prop_ce=0.0, mac=1.0)
        if fluid_ml_hr > 0.0:
            model.add_fluid_heat_j(fluid_ml_hr * dt / 3600.0, 22.0, temp)
        temp = model.step_temperature(dt, temp, warmer_target, perfusing=True)
    # Re-anchor at recorded equilibrated probe temperature while preserving redistribution progress.
    temp = t0
    d30 = d60 = 0.0
    for step in range(1, 361):  # 360 * 10 s = 60 min
        model.update_depth(temp, prop_ce=0.0, mac=1.0)
        if fluid_ml_hr > 0.0:
            model.add_fluid_heat_j(fluid_ml_hr * dt / 3600.0, 22.0, temp)
        temp = model.step_temperature(dt, temp, warmer_target, perfusing=True)
        if step == 180:
            d30 = temp - t0
        elif step == 360:
            d60 = temp - t0
    return float(d30), float(d60)


def evaluate(candidate: tuple[dict[str, str], dict[str, str]], dt: float) -> dict[str, Any]:
    del dt
    case, tracks = candidate
    row: dict[str, Any] = dict(
        caseid=int(case["caseid"]),
        subjectid=int(case["subjectid"]),
        approach=case.get("approach", ""),
    )
    load = functools.cache(numeric_track)
    try:
        bt, bv = load(tracks["Solar8000/BT"])
        valid = (bv >= 34.0) & (bv <= 39.0)
        bt, bv = bt[valid], bv[valid]
        if len(bt) < 100:
            raise Excluded("insufficient valid temperature readings")
        ok = np.flatnonzero(bv >= 35.5)
        if not len(ok):
            raise Excluded("probe never reached 35.5 C")
        t_eq = float(bt[ok[0]]) + 300.0
        w0 = bv[(bt >= t_eq) & (bt < t_eq + 300.0)]
        w30 = bv[(bt >= t_eq + 1800.0) & (bt < t_eq + 2100.0)]
        w60 = bv[(bt >= t_eq + 3600.0) & (bt < t_eq + 3900.0)]
        if len(w0) < 10 or len(w30) < 10 or len(w60) < 10:
            raise Excluded("less than 60 min post-equilibration temperature data")
        t0 = float(np.median(w0))
        if not (35.5 <= t0 <= 38.0):
            raise Excluded("initial core temperature outside 35.5-38.0 C")
        obs_d30 = float(np.median(w30)) - t0
        obs_d60 = float(np.median(w60)) - t0
        patient = patient_from_case(case)
        duration_h = max(1.0, (float(case["aneend"]) - float(case["anestart"])) / 3600.0)
        fluid_ml_hr = (amount(case, "intraop_crystalloid") + amount(case, "intraop_colloid")) / duration_h
        elapsed_ind = max(300.0, min(2400.0, t_eq - float(case["anestart"])))
        unw_d30, unw_d60 = _simulate_thermal_curve(
            patient, t0=t0, elapsed_since_ind_s=elapsed_ind,
            fluid_ml_hr=fluid_ml_hr, active_warming=False,
        )
        wrm_d30, wrm_d60 = _simulate_thermal_curve(
            patient, t0=t0, elapsed_since_ind_s=elapsed_ind,
            fluid_ml_hr=fluid_ml_hr, active_warming=True,
        )
    except Excluded as error:
        return {**row, "exclusion": str(error)}

    return {
        **row,
        "age": float(patient.age),
        "bmi": round(float(patient.bmi), 2),
        "t0": round(t0, 2),
        "obs_d30": obs_d30,
        "obs_d60": obs_d60,
        "unwarmed_d30": unw_d30,
        "unwarmed_d60": unw_d60,
        "warmed_d30": wrm_d30,
        "warmed_d60": wrm_d60,
    }


def report(data: dict[str, Any], settings: dict[str, Any]) -> str:
    rng = np.random.default_rng(settings["seed"])
    rows = data["rows"]
    lines = report_header("VitalDB intraoperative core temperature", settings) + [
        "## Methods", "",
        "Compares recorded Solar8000/BT changes 30 and 60 minutes after probe equilibration",
        "(5 min after first reading >= 35.5 C) across retained TIVA and sevoflurane cases",
        "against AnaSim's ThermalModel both without and with active forced-air warming",
        "(warming status is not logged in VitalDB).", "",
        f"Eligible retained cases with BT track: {data['eligible']}. Retained with 60-min curve: {len(rows)}.", "",
        "| Subgroup | n | Recorded d60, C (95% CI) | Unwarmed model d60 (C) | Forced-air model d60 (C) | Recorded d30 (C) |",
        "|---|---:|---|---|---|---|",
    ]
    groups = [
        ("All cases", rows),
        ("BMI under 22", [r for r in rows if r["bmi"] < 22.0]),
        ("BMI 22-26.9", [r for r in rows if 22.0 <= r["bmi"] < 27.0]),
        ("BMI 27 and over", [r for r in rows if r["bmi"] >= 27.0]),
        ("Age under 50", [r for r in rows if r["age"] < 50.0]),
        ("Age 50-69", [r for r in rows if 50.0 <= r["age"] < 70.0]),
        ("Age 70 and over", [r for r in rows if r["age"] >= 70.0]),
        ("Open", [r for r in rows if r["approach"] == "Open"]),
        ("Videoscopic or robotic", [r for r in rows if r["approach"] != "Open"]),
    ]
    for label, g in groups:
        if len(g) < 5:
            continue
        lines.append(
            f"| {label} | {len(g)} | {mean_ci([r['obs_d60'] for r in g], rng, digits=2, include_n=False)} | "
            f"{np.mean([r['unwarmed_d60'] for r in g]):+.2f} | "
            f"{np.mean([r['warmed_d60'] for r in g]):+.2f} | "
            f"{np.mean([r['obs_d30'] for r in g]):+.2f} |"
        )
    if rows:
        inside = sum(
            1 for r in rows
            if min(r["unwarmed_d60"], r["warmed_d60"]) - 0.1 <= r["obs_d60"] <= max(r["unwarmed_d60"], r["warmed_d60"]) + 0.1
        )
        lines += [
            "",
            f"Median equilibrated starting core temperature: {np.median([r['t0'] for r in rows]):.2f} C. "
            f"Share of patients within [unwarmed, forced-air warmed] +/- 0.1 C at 60 min: "
            f"{inside}/{len(rows)} ({100.0 * inside / len(rows):.0f}%).",
        ]
    lines += report_footer(data["exclusions"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    vitaldb_data.CACHE = args.cache

    keep_ids: set[int] = set()
    for name in ("vitaldb_tiva.json", "vitaldb_sevo.json"):
        p = args.output / name
        if p.exists():
            keep_ids.update(int(r["caseid"]) for r in json.loads(p.read_text()).get("rows", []))

    pool = [
        (case, trk)
        for case, trk in domain_cases(["Solar8000/BT"], open_supine_only=False, cache=args.cache)
        if not keep_ids or int(case["caseid"]) in keep_ids
    ]
    results = run_hemo_pool(evaluate, pool, 1.0, args.workers, {}, args.cache)
    rows = [r for r in results if "exclusion" not in r]
    exclusions = [r for r in results if "exclusion" in r]
    data = dict(eligible=len(pool), rows=rows, exclusions=exclusions)
    settings = dict(generated=datetime.now(timezone.utc).date().isoformat(), seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "vitaldb_temperature"
    stem.with_suffix(".json").write_text(json.dumps(dict(settings=settings, **data), indent=2, allow_nan=False) + "\n")
    stem.with_suffix(".md").write_text(report(data, settings))
    print(f"Report: {stem.with_suffix('.md')} (retained {len(rows)}/{len(pool)}, excl {dict(Counter(r['exclusion'] for r in exclusions))})", flush=True)


if __name__ == "__main__":
    main()
