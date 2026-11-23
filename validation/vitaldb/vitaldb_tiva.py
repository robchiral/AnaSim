"""Replay propofol-remifentanil TIVA through the AnaSim engine; compare MAP and HR from induction through surgery.

Covers all eligible unique-subject oral single-lumen general anesthetics inside
AnaSim's supported domain (ASA 1-3, normotensive and hypertensive, all surgical
departments without vasoactive infusions or transfusion), passing recorded preop
hemoglobin and creatinine clearance into ``Patient``. Reports both full-cohort
and legacy-core breakdowns, plus optional k-fold cross-validated override sweeps.
Only deidentified case summaries are saved in Git.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.vitaldb import vitaldb_data  # noqa: E402
from validation.vitaldb.vitaldb_data import (  # noqa: E402
    CACHE,
    REMI_RATES,
    REMI_UG_PER_ML,
    VASOACTIVE,
    Excluded,
    amount,
    delivered_volume,
    domain_cases,
    is_legacy_hemo_core,
    numeric_track,
    optional_float,
    patient_from_case,
    time_grid,
)
from validation.vitaldb.vitaldb_replay import (  # noqa: E402
    BOOTSTRAP,
    CUFF_WINDOW_S,
    SENSITIVITY,
    STOP,
    apply_hemo_override,
    course,
    course_table,
    cuff_model,
    cv_grid_lines,
    cv_hemodynamic_grid,
    cv_hemodynamic_stability,
    cv_stability_lines,
    delivery,
    estimate,
    expand_override_grid,
    hemo_subgroup_rows,
    mean_ci,
    pre_induction,
    procedure,
    readings,
    report_footer,
    report_header,
    rule_stats,
    run_hemodynamic_replay,
    treated,
    window_medians,
)

REQUIRED = [
    "Orchestra/PPF20_RATE", "Orchestra/PPF20_VOL", "Primus/MAC", "Primus/FIN2O",
    "Solar8000/NIBP_MBP", "Solar8000/NIBP_SBP", "Solar8000/HR",
]
INTUBATION_S = 240.0


def eligible(case: dict[str, str], tracks: dict[str, str]) -> bool:
    """Adults in AnaSim's domain under TIVA (ASA 1-3, +/- HTN) without vasoactive infusions or transfusion."""
    return (
        any(name in tracks for name in REMI_RATES)
        and not any(f"Orchestra/{drug}_RATE" in tracks for drug in (*VASOACTIVE, "DEX2", "DEX4"))
        and all(
            amount(case, key) == 0
            for key in ("intraop_ppf", "intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_rbc", "intraop_ffp")
        )
        and amount(case, "intraop_ebl") <= 500
        and case["asa"] in ("1", "2", "3")
        and case["preop_htn"] in ("0", "1")
    )


def peak(series: np.ndarray, samples: int = 15) -> float:
    """Highest rolling median over ``samples`` consecutive values (30 s of 2-s HR)."""
    return float(np.median(np.lib.stride_tricks.sliding_window_view(series, samples), axis=1).max())


def _worker_init(override: dict[str, Any], cache: Path) -> None:
    vitaldb_data.CACHE = cache
    apply_hemo_override(override)


def evaluate(candidate, dt: float) -> dict[str, Any]:
    case, tracks = candidate
    row: dict[str, Any] = dict(
        caseid=int(case["caseid"]),
        subjectid=int(case["subjectid"]),
        treated=treated(case),
        procedure=procedure(case),
        department=case["department"],
        asa=case["asa"],
        hypertensive=case["preop_htn"] == "1",
        legacy_core=is_legacy_hemo_core(case),
    )
    try:
        rate, volume, remi, remi_name, start, end, onset, stop = delivery(tracks)
        opstart, opend = float(case["opstart"]), float(case["opend"])
        incision = min(opstart, stop, end)
        if incision - onset < 300:
            raise Excluded("less than five minutes from induction to incision")
        times, cuff_map, cuff_sbp, cuff_hr = readings(tracks, onset, incision)
        if not len(times):
            raise Excluded("no cuff reading between induction and incision")
        baseline_map, baseline_hr = pre_induction(tracks, onset)
        patient = patient_from_case(case, baseline_map=baseline_map, baseline_hr=baseline_hr)
        maintenance = (opstart + 900.0, min(opend, stop, end))
        has_maintenance = incision == opstart and maintenance[1] - maintenance[0] >= 600
        grid = time_grid(max(start, onset - 60.0), maintenance[1] if has_maintenance else incision, dt)
        propofol_mg = delivered_volume(rate, grid) * 20.0
        measured = np.interp(grid[-1], *volume) - np.interp(grid[0], *volume)
        if abs(propofol_mg[-1] - propofol_mg[0] - 20.0 * measured) > 20.0 * max(0.5, 0.05 * measured):
            raise Excluded("infusion rate and volume disagree by more than five percent or 0.5 mL")
        remi_ug = delivered_volume(remi, grid) * REMI_UG_PER_ML[remi_name]
        step_dt = np.diff(grid)
        propofol_mg_s = np.diff(propofol_mg) / step_dt
        remi_ug_s = np.diff(remi_ug) / step_dt
        ebl = amount(case, "intraop_ebl")
        bleed_rate = ebl / max(1.0, (opend - opstart) / 60.0)
        events = [
            (onset + INTUBATION_S, lambda e: e.start_disturbance("stim_intubation_pulse")),
            (opstart, lambda e: e.start_disturbance("stim_sustained_surgery")),
            (opend, lambda e: e.stop_disturbance()),
        ]
        if ebl:
            events += [(opstart, lambda e: e.start_hemorrhage(bleed_rate)), (opend, lambda e: e.stop_hemorrhage())]
        predicted = run_hemodynamic_replay(
            case, patient, grid, onset=onset, events=events, propofol_mg_s=propofol_mg_s, remi_ug_s=remi_ug_s,
        )
    except Excluded as error:
        return {**row, "exclusion": str(error)}

    model_map = cuff_model(times, grid, predicted[:, 0])
    errors, seconds = model_map - cuff_map, times - onset
    summary: dict[str, Any] = {
        **row,
        "age": float(patient.age),
        "bmi": round(float(patient.bmi), 2),
        "baseline_map": baseline_map,
        "has_preop_hb": optional_float(case, "preop_hb") is not None,
        "has_preop_cr": patient.creatinine_clearance_ml_min is not None,
        "readings": len(times),
        "median_error": float(np.median(errors)),
    }
    for rule, limits in (("primary", STOP), ("sensitivity", SENSITIVITY)):
        summary[rule] = rule_stats(errors, seconds, cuff_map, cuff_sbp, cuff_hr, limits)
    summary["posting"] = rule_stats(
        np.interp(times, grid, predicted[:, 0]) - cuff_map, seconds, cuff_map, cuff_sbp, cuff_hr, STOP,
    )
    summary["course"] = course(seconds, cuff_map, model_map, baseline_map, incision - onset)
    hr_time, hr_value = numeric_track(tracks["Solar8000/HR"])
    window = (hr_time >= onset + 180.0) & (hr_time < min(onset + 600.0, incision)) & (hr_value >= 40) & (hr_value <= 160)
    if window.sum() >= 30:
        model_hr = np.interp(hr_time[window], grid, predicted[:, 1])
        summary["hr"] = dict(
            median_error=float(np.median(model_hr - hr_value[window])),
            peak_error=peak(model_hr) - peak(hr_value[window]),
        )
    if has_maintenance:
        summary["before_incision"] = window_medians(tracks, grid, predicted, onset + 600.0, incision)
        summary["maintenance"] = window_medians(tracks, grid, predicted, *maintenance)
    return summary


def report(data: dict[str, Any], settings: dict[str, Any]) -> str:
    rng = np.random.default_rng(settings["seed"])
    rows = data["rows"]
    untreated = [r for r in rows if not r["treated"]]
    groups = hemo_subgroup_rows(rows, legacy_label="Legacy core (GenSurg, ASA 1-2, normotensive, BMI ≤ 32)")
    override = [f"Model override: `{json.dumps(settings['override'])}`.", ""] if settings.get("override") else []
    n_hb = sum(1 for r in rows if r.get("has_preop_hb"))
    n_cr = sum(1 for r in rows if r.get("has_preop_cr"))
    lines = report_header("VitalDB TIVA hemodynamics", settings) + override + [
        "## Methods", "",
        "Recorded propofol and remifentanil pump delivery drives the full engine across all eligible unique-subject",
        "oral single-lumen general anesthetics inside AnaSim's patient domain (ASA 1-3, normotensive and hypertensive).",
        "Each patient starts from their last cuff MAP within 15 minutes before propofol, preceding 2-minute median HR,",
        f"and recorded preoperative hemoglobin ({n_hb}/{len(rows)}) and Cockcroft-Gault CrCl ({n_cr}/{len(rows)}).",
        "Controlled ventilation (VT 8 mL/kg PBW up to 600 mL, RR 12, PEEP 5) starts 90 s after propofol, and",
        "intubation stimulation occurs at 4 minutes. Recorded crystalloid, colloid, and blood loss run over surgery",
        "with sustained stimulation; positioning and pneumoperitoneum are not modeled.", "",
        f"From propofol to incision, each cuff reading is compared with model MAP averaged over the {CUFF_WINDOW_S:.0f} s",
        "before the reading appears. A patient's readings stop at the first one with MAP below 65 mmHg, SBP below 90, or",
        "HR below 45 (sensitivity analysis: 70, 100, and 45), so later treatment cannot bias the fit. A likelihood fit",
        "with a patient offset and reading noise gives the mean error (model minus recorded), with 95% intervals from",
        f"{BOOTSTRAP} patient bootstrap resamples.", "",
        f"Eligible candidates: {data['eligible']}. Retained: {len(rows)}.", "",
        "## From propofol to incision", "",
        "| Patients | n | Readings | Mean error, mmHg (95% CI) | Patient SD | Reading SD |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for label, group in groups:
        if len(group) < 5:
            continue
        [(mean, low, high)], between, within, used = estimate(group, "primary", False, rng)
        lines.append(
            f"| {label} | {len(group)} | {used} | {mean:.1f} ({low:.1f} to {high:.1f}) | {between:.1f} | {within:.1f} |"
        )
    if rows:
        [(mean, low, high)], _, _, _ = estimate(rows, "sensitivity", False, rng)
        lines += ["", f"Sensitivity stopping rule, all patients: {mean:.1f} ({low:.1f} to {high:.1f}) mmHg."]
        by_period, _, _, _ = estimate(rows, "primary", True, rng)
        lines += [
            "",
            "By time after propofol, all patients: "
            + "; ".join(
                f"{name} {m:.1f} ({lo:.1f} to {hi:.1f})"
                for name, (m, lo, hi) in zip(("before 10 min", "from 10 min"), by_period, strict=True)
            )
            + " mmHg.",
        ]
        [(mean, low, high)], _, _, _ = estimate(rows, "posting", False, rng)
        lines += ["", f"Compared with model MAP when each reading appears, all patients: {mean:.1f} ({low:.1f} to {high:.1f}) mmHg."]
        lines += cv_stability_lines(data.get("cv_stability"))
        lines += course_table(rows)
        lines += course_table([r for r in rows if r["age"] >= 70], ", patients aged 70 and over")
    hr = [r["hr"] for r in untreated if "hr" in r]
    if hr:
        lines += [
            "",
            f"ECG HR error 3-10 minutes after propofol, untreated: per-patient median "
            f"{mean_ci([h['median_error'] for h in hr], rng)}; highest 30-s median "
            f"{mean_ci([h['peak_error'] for h in hr], rng)} bpm.",
        ]
    paired = [r for r in untreated if r.get("maintenance") and r.get("before_incision")]
    if paired:
        lines += [
            "",
            "## Surgery",
            "",
            f"Untreated patients, from 15 minutes after incision to the end of surgery: MAP error "
            f"{mean_ci([r['maintenance']['map']['predicted'] - r['maintenance']['map']['observed'] for r in paired], rng)}"
            f" mmHg; HR error "
            f"{mean_ci([r['maintenance']['hr']['predicted'] - r['maintenance']['hr']['observed'] for r in paired], rng)}"
            " bpm. Arterial MAP is used when recorded.",
            "",
            "MAP rise from the last 10 minutes before incision to surgery: recorded "
            f"{mean_ci([r['maintenance']['map']['observed'] - r['before_incision']['map']['observed'] for r in paired], rng)}"
            ", model "
            f"{mean_ci([r['maintenance']['map']['predicted'] - r['before_incision']['map']['predicted'] for r in paired], rng)}"
            " mmHg.",
        ]
    lines += cv_grid_lines(data.get("cv_grid"))
    lines += report_footer(data["exclusions"])
    return "\n".join(lines)


def _run_pool(pool, dt: float, workers: int, override: dict[str, Any], cache: Path) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(override, cache)) as executor:
        return list(executor.map(evaluate, pool, [dt] * len(pool), chunksize=4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--override", default="{}", help="JSON HemodynamicConfig fields for single-config comparison")
    parser.add_argument("--cv-grid", default="{}", help="JSON dict of {field: [values]} for k-fold CV parameter sweep")
    parser.add_argument("--folds", type=int, default=5, help="Patient-grouped stratified CV folds")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    vitaldb_data.CACHE = args.cache
    override = json.loads(args.override)
    grid_spec = json.loads(args.cv_grid)

    pool = [
        (case, trk)
        for case, trk in domain_cases(REQUIRED, open_supine_only=False, cache=args.cache)
        if eligible(case, trk)
    ]
    results = _run_pool(pool, args.dt, args.workers, override, args.cache)
    rows = [r for r in results if "exclusion" not in r]
    exclusions = [r for r in results if "exclusion" in r]
    data: dict[str, Any] = dict(
        eligible=len(pool),
        rows=rows,
        exclusions=exclusions,
        cv_stability=cv_hemodynamic_stability(rows, n_folds=args.folds, seed=args.seed),
    )
    if grid_spec:
        retained_ids = {r["caseid"] for r in rows}
        retained_pool = [item for item in pool if int(item[0]["caseid"]) in retained_ids]
        candidates = expand_override_grid(grid_spec)
        candidate_results = []
        for cand in candidates:
            merged = {**override, **cand}
            if merged == override:
                candidate_results.append((merged, rows))
            else:
                cand_res = _run_pool(retained_pool, args.dt, args.workers, merged, args.cache)
                candidate_results.append((merged, [r for r in cand_res if "exclusion" not in r]))
        data["cv_grid"] = cv_hemodynamic_grid(candidate_results, n_folds=args.folds, seed=args.seed)

    settings = dict(
        generated=datetime.now(timezone.utc).date().isoformat(),
        dt=args.dt,
        seed=args.seed,
        folds=args.folds,
        override=override,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "vitaldb_tiva"
    stem.with_suffix(".json").write_text(json.dumps(dict(settings=settings, **data), indent=2, allow_nan=False) + "\n")
    stem.with_suffix(".md").write_text(report(data, settings))
    print(f"Report: {stem.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
