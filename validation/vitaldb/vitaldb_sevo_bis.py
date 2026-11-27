"""Validate AnaSim's sevoflurane-to-BIS response curve against VitalDB recordings.

Replays recorded end-tidal sevoflurane (Primus/EXP_SEVO) through VolatilePK brain
equilibration and optional remifentanil infusion (Orchestra/RFTN*_RATE) through
RemifentanilPKEleveld into BISModel without fitting any parameters or time shifts.
Primary evaluation uses surgical maintenance (>= 15 min after incision, after any
propofol induction bolus has washed out) and emergence/washout.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
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

from anasim.core.monitors import BIS_DISPLAY_TAU_S  # noqa: E402
from anasim.patient.pd.anesthesia import BISModel  # noqa: E402
from anasim.patient.pk_models import RemifentanilPKEleveld  # noqa: E402
from anasim.patient.volatile_pk import VolatilePK  # noqa: E402
from validation.vitaldb import vitaldb_data  # noqa: E402
from validation.vitaldb.vitaldb_data import (  # noqa: E402
    CACHE,
    DEFAULT_WORKERS,
    REMI_RATES,
    REMI_UG_PER_ML,
    Excluded,
    amount,
    domain_cases,
    held,
    numeric_track,
    patient_from_case,
    time_grid,
)
from validation.vitaldb.vitaldb_replay import (  # noqa: E402
    SAMPLE_SECONDS,
    mean_ci,
    report_footer,
    report_header,
    run_hemo_pool,
)

SQI_MIN = 90.0


def eligible(case: dict[str, str], tracks: dict[str, str]) -> bool:
    if (
        "Primus/EXP_SEVO" not in tracks
        or "BIS/BIS" not in tracks
        or "BIS/SQI" not in tracks
        or "Primus/FIN2O" not in tracks
    ):
        return False
    if "Orchestra/PPF20_RATE" in tracks:
        return False
    if any(f"Orchestra/{d}_RATE" in tracks for d in ("DEX2", "DEX4")):
        return False
    return amount(case, "intraop_mdz") == 0.0


def replay_sevo_bis(
    patient: Any,
    grid: np.ndarray,
    exp_sevo_pct: np.ndarray,
    remi_ug_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay EtSevo (%) into VolatilePK brain MAC and BISModel; return (bis_smooth, brain_mac)."""
    vpk = VolatilePK(patient, "Sevoflurane", 0.65, 1.7, 2.05)
    bis_model = BISModel(patient)
    remi_pk = RemifentanilPKEleveld(patient) if remi_ug_s is not None else None
    # Match VolatilePK resting cerebral equilibration rate (1/s).
    k_brain_s = ((vpk._base_co * vpk.f_vrg_frac / vpk.v_vrg) / vpk.lambda_t_b_vrg) / 60.0
    p_brain = max(0.0, float(exp_sevo_pct[0]) / 100.0)
    mac0 = 100.0 * p_brain / vpk.mac_age
    bis_model.initialize(bis_model.compute_bis(0.0, mac0), mac_sevo=mac0)
    smooth = float(bis_model.compute_bis(0.0, mac0))

    n = len(grid)
    bis_out = np.empty(n, dtype=float)
    mac_out = np.empty(n, dtype=float)
    bis_out[0] = smooth
    mac_out[0] = mac0
    for i, dt in enumerate(np.diff(grid), start=1):
        dt_f = float(dt)
        p_alv = max(0.0, float(exp_sevo_pct[i - 1]) / 100.0)
        p_brain += (p_alv - p_brain) * (-math.expm1(-k_brain_s * dt_f))
        mac = 100.0 * p_brain / vpk.mac_age
        ce_remi = 0.0
        if remi_pk is not None and remi_ug_s is not None:
            remi_pk.step(dt_f, float(remi_ug_s[i - 1]))
            ce_remi = remi_pk.state.ce
        target = bis_model.step(dt_f, 0.0, mac_sevo=mac, ce_opioid=ce_remi)
        smooth += -math.expm1(-dt_f / BIS_DISPLAY_TAU_S) * (target - smooth)
        bis_out[i] = smooth
        mac_out[i] = mac
    return bis_out, mac_out


def evaluate(candidate: tuple[dict[str, str], dict[str, str]], dt: float) -> dict[str, Any]:
    case, tracks = candidate
    row: dict[str, Any] = dict(caseid=int(case["caseid"]), subjectid=int(case["subjectid"]))
    load = functools.cache(numeric_track)
    try:
        patient = patient_from_case(case)
        opstart, opend = float(case["opstart"]), float(case["opend"])
        anestart, aneend = float(case["anestart"]), float(case["aneend"])
        if opend - opstart < 1800.0:
            raise Excluded("short surgery under 30 min")

        exp_track = load(tracks["Primus/EXP_SEVO"])
        bis_track = load(tracks["BIS/BIS"])
        sqi_track = load(tracks["BIS/SQI"])
        n2o_track = load(tracks["Primus/FIN2O"])

        start = max(anestart, float(exp_track[0][0]))
        end = min(aneend, float(exp_track[0][-1]), float(bis_track[0][-1]), float(sqi_track[0][-1]))
        if end < opstart + 1500.0:
            raise Excluded("short usable BIS/sevoflurane overlap")

        grid = time_grid(start, end, dt)
        exp_sevo = held(exp_track, grid, 30.0)
        if float(np.mean(np.isfinite(exp_sevo))) < 0.85:
            raise Excluded("incomplete EtSevo track")
        exp_sevo = np.nan_to_num(exp_sevo, nan=0.0)
        exp_sevo = np.clip(exp_sevo, 0.0, 10.0)

        n2o_vals = held(n2o_track, grid, 30.0)
        if np.any(np.nan_to_num(n2o_vals, nan=0.0) > 5.0):
            raise Excluded("nitrous oxide exposure")

        remi_ug_s = None
        has_remi = False
        for rname in REMI_RATES:
            if rname in tracks:
                rtrack = load(tracks[rname])
                rrate = np.nan_to_num(held(rtrack, grid[:-1], 60.0), nan=0.0)
                if float(np.max(rrate)) > 0.5:
                    remi_ug_s = np.clip(rrate, 0.0, 100.0) * REMI_UG_PER_ML[rname] / 3600.0
                    has_remi = True
                break

        pred_bis, brain_mac = replay_sevo_bis(patient, grid, exp_sevo, remi_ug_s=remi_ug_s)

        sample = np.arange(opstart + 900.0, end, SAMPLE_SECONDS)
        if len(sample) < 120:
            raise Excluded("short maintenance window")
        obs_bis = held(bis_track, sample, 5.0)
        obs_sqi = held(sqi_track, sample, 5.0)
        s_exp = np.interp(sample, grid, exp_sevo)
        s_pred = np.interp(sample, grid, pred_bis)
        s_mac = np.interp(sample, grid, brain_mac)

        maint_mask = (
            (sample <= opend)
            & np.isfinite(obs_bis) & (obs_bis > 5.0) & (obs_bis <= 100.0)
            & np.isfinite(obs_sqi) & (obs_sqi >= SQI_MIN)
            & (s_exp >= 0.4)
        )
        if maint_mask.sum() * SAMPLE_SECONDS < 600.0:
            raise Excluded("less than 10 min usable surgical maintenance BIS")

        diff_m = s_pred[maint_mask] - obs_bis[maint_mask]
        comp_m = 40.0 - obs_bis[maint_mask]
        med_mac = float(np.median(s_mac[maint_mask]))
        med_sevo = float(np.median(s_exp[maint_mask]))

        # Optional emergence/washout window (after surgery ends while EtSevo washes out).
        emerge_mask = (
            (sample > opend)
            & np.isfinite(obs_bis) & (obs_bis > 5.0) & (obs_bis <= 100.0)
            & np.isfinite(obs_sqi) & (obs_sqi >= SQI_MIN)
        )
        emerge = None
        if emerge_mask.sum() * SAMPLE_SECONDS >= 180.0 and float(np.min(s_exp[emerge_mask])) < 0.7 * med_sevo:
            diff_e = s_pred[emerge_mask] - obs_bis[emerge_mask]
            comp_e = 40.0 - obs_bis[emerge_mask]
            emerge = dict(
                minutes=round(float(emerge_mask.sum() * SAMPLE_SECONDS / 60.0), 1),
                bias=float(np.mean(diff_e)),
                mae=float(np.mean(np.abs(diff_e))),
                rmse=float(np.sqrt(np.mean(diff_e**2))),
                comp_mae=float(np.mean(np.abs(comp_e))),
            )
    except Excluded as error:
        return {**row, "exclusion": str(error)}

    return {
        **row,
        "age": float(patient.age),
        "bmi": round(float(patient.bmi), 2),
        "has_remi": has_remi,
        "median_et_sevo": round(med_sevo, 3),
        "median_mac": round(med_mac, 3),
        "maint_minutes": round(float(maint_mask.sum() * SAMPLE_SECONDS / 60.0), 1),
        "obs_bis_median": round(float(np.median(obs_bis[maint_mask])), 1),
        "pred_bis_median": round(float(np.median(s_pred[maint_mask])), 1),
        "bias": float(np.mean(diff_m)),
        "mae": float(np.mean(np.abs(diff_m))),
        "rmse": float(np.sqrt(np.mean(diff_m**2))),
        "comp_mae": float(np.mean(np.abs(comp_m))),
        "emerge": emerge,
    }


def report(data: dict[str, Any], settings: dict[str, Any]) -> str:
    rng = np.random.default_rng(settings["seed"])
    rows = data["rows"]
    lines = report_header("VitalDB sevoflurane to BIS validation", settings) + [
        "## Methods", "",
        "Replays recorded end-tidal sevoflurane (Primus/EXP_SEVO) through VolatilePK cerebral equilibration",
        "and optional remifentanil infusions through RemifentanilPKEleveld into BISModel without fitting",
        "parameters or offsets. Evaluates surgical maintenance (>= 15 min after incision to opend, SQI >= 90,",
        "EtSevo >= 0.4%) and post-surgery washout against recorded BIS/BIS and a constant BIS=40 comparator.", "",
        f"Eligible candidates: {data['eligible']}. Retained: {len(rows)} "
        f"({sum(1 for r in rows if r['has_remi'])} with remifentanil, {sum(1 for r in rows if not r['has_remi'])} without).", "",
        "## Surgical maintenance BIS (predicted minus recorded)", "",
        "| Subgroup | n | Median EtMAC | Recorded / Model BIS | Bias (95% CI) | MAE (95% CI) | Comparator MAE (BIS=40) |",
        "|---|---:|---:|---|---|---|---|",
    ]
    groups = [
        ("All maintenance", rows),
        ("Without remifentanil", [r for r in rows if not r["has_remi"]]),
        ("With remifentanil", [r for r in rows if r["has_remi"]]),
        ("Age under 45", [r for r in rows if r["age"] < 45]),
        ("Age 45-59", [r for r in rows if 45 <= r["age"] < 60]),
        ("Age 60-69", [r for r in rows if 60 <= r["age"] < 70]),
        ("Age 70 and over", [r for r in rows if r["age"] >= 70]),
        ("BMI under 30", [r for r in rows if r["bmi"] < 30]),
        ("BMI 30 and over", [r for r in rows if r["bmi"] >= 30]),
        ("EtMAC under 0.85", [r for r in rows if r["median_mac"] < 0.85]),
        ("EtMAC 0.85-1.05", [r for r in rows if 0.85 <= r["median_mac"] < 1.05]),
        ("EtMAC 1.05 and over", [r for r in rows if r["median_mac"] >= 1.05]),
    ]
    for label, g in groups:
        if len(g) < 3:
            continue
        med_mac = float(np.median([r["median_mac"] for r in g]))
        obs_b = float(np.median([r["obs_bis_median"] for r in g]))
        pred_b = float(np.median([r["pred_bis_median"] for r in g]))
        lines.append(
            f"| {label} | {len(g)} | {med_mac:.2f} | {obs_b:.1f} / {pred_b:.1f} | "
            f"{mean_ci([r['bias'] for r in g], rng, include_n=False)} | "
            f"{mean_ci([r['mae'] for r in g], rng, include_n=False)} | "
            f"{mean_ci([r['comp_mae'] for r in g], rng, include_n=False)} |"
        )

    em = [r["emerge"] for r in rows if r.get("emerge")]
    if em:
        lines += [
            "", "## Emergence / washout BIS", "",
            f"Retained emergence traces: n={len(em)}. "
            f"Bias: {mean_ci([e['bias'] for e in em], rng, include_n=False)}; "
            f"MAE: {mean_ci([e['mae'] for e in em], rng, include_n=False)} "
            f"(comparator BIS=40 MAE: {mean_ci([e['comp_mae'] for e in em], rng, include_n=False)}).",
        ]

    return "\n".join(lines + report_footer(data["exclusions"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    vitaldb_data.CACHE = args.cache

    candidates = [c for c in domain_cases((), open_supine_only=False) if eligible(*c)]
    results = run_hemo_pool(evaluate, candidates, args.dt, args.workers, {}, args.cache)
    rows = [r for r in results if "exclusion" not in r]
    exclusions = [r for r in results if "exclusion" in r]
    data = dict(eligible=len(candidates), rows=rows, exclusions=exclusions)
    settings = dict(generated=datetime.now(timezone.utc).date().isoformat(), seed=args.seed, dt=args.dt)
    args.output.mkdir(parents=True, exist_ok=True)
    md_path = args.output / "vitaldb_sevo_bis.md"
    json_path = args.output / "vitaldb_sevo_bis.json"
    md_path.write_text(report(data, settings) + "\n")
    json_path.write_text(json.dumps({"settings": settings, **data}, indent=2, allow_nan=False) + "\n")
    print(f"Report: {md_path} (retained {len(rows)}/{len(candidates)}, excl {dict(Counter(r['exclusion'] for r in exclusions))})", flush=True)


if __name__ == "__main__":
    main()
