"""Replay bolus propofol induction with sevoflurane maintenance; compare MAP and HR with VitalDB.

Covers all eligible unique-subject oral single-lumen general anesthetics inside
AnaSim's supported domain (ASA 1-3, normotensive and hypertensive, all surgical
departments and positions without vasoactive infusions or transfusion), passing
recorded preop hemoglobin and creatinine clearance into ``Patient``. Reports
full-cohort and legacy-core breakdowns, plus optional k-fold cross-validated
override sweeps. Only deidentified case summaries are saved in Git.
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

from anasim.core.engine import SimulationEngine  # noqa: E402
from anasim.core.state import SimulationConfig  # noqa: E402
from validation.vitaldb import vitaldb_data  # noqa: E402
from validation.vitaldb.vitaldb_data import (  # noqa: E402
    CACHE,
    REMI_RATES,
    REMI_UG_PER_ML,
    VASOACTIVE,
    Excluded,
    amount,
    covered,
    delivered_volume,
    domain_cases,
    held,
    is_legacy_core,
    numeric_track,
    optional_float,
    patient_from_case,
    time_grid,
)
from validation.vitaldb.vitaldb_replay import (  # noqa: E402
    BOOTSTRAP,
    CUFF_WINDOW_S,
    SAMPLE_SECONDS,
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
    "Solar8000/NIBP_MBP", "Solar8000/NIBP_SBP", "Solar8000/HR",
    "BIS/BIS", "Primus/INSP_SEVO", "Primus/FIN2O",
]
HR_WINDOWS = ((0.0, 300.0), (300.0, 600.0), (600.0, np.inf))  # Seconds after propofol, to incision
# In TIVA cases with exact pump timing, model BIS falls below 60 a median 34 s after recorded BIS
# (22 s with the fastest loading), so propofol starts this much after the model latency implies.
BIS_LAG_S = 30.0
# The ventilator tidal volume setting first appears a median 200 s after propofol.
INTUBATION_S = 150.0


def eligible(case: dict[str, str], tracks: dict[str, str]) -> bool:
    weight = float(case["weight"])
    return (
        0.0 < amount(case, "intraop_ppf") <= 3.0 * weight
        and amount(case, "intraop_ftn") <= 3.0 * weight
        and "Orchestra/PPF20_RATE" not in tracks
        and not any(f"Orchestra/{drug}_RATE" in tracks for drug in (*VASOACTIVE, "DEX2", "DEX4"))
        and all(amount(case, key) == 0 for key in ("intraop_mdz", "intraop_epi", "intraop_rbc", "intraop_ffp"))
        and amount(case, "intraop_ebl") <= 500
        and case["asa"] in ("1", "2", "3")
        and case["preop_htn"] in ("0", "1")
    )


def bis_drop_time(tracks: dict[str, str]) -> float:
    """When BIS first stays below 60 for 30 s after at least 2 min at 70 or more; zero BIS is missing."""
    time, value = numeric_track(tracks["BIS/BIS"])
    bis = time[value > 0], value[value > 0]
    if len(bis[0]) < 2:
        raise Excluded("no valid BIS")
    grid = np.arange(bis[0][0], bis[0][-1], SAMPLE_SECONDS)
    values = held(bis, grid, 30.0)
    awake = np.isfinite(values) & (values >= 70)
    low = np.isfinite(values) & (values < 60)
    run = round(30.0 / SAMPLE_SECONDS)
    for index in range(round(120.0 / SAMPLE_SECONDS), len(grid) - run):
        if low[index:index + run].all() and awake[index - round(120.0 / SAMPLE_SECONDS):index].sum() >= 18:
            return float(grid[index])
    raise Excluded("no BIS fall from awake values")


def bis_latency(case: dict[str, str], patient) -> float:
    """Seconds from the start of the propofol bolus until model BIS first falls below 60."""
    engine = SimulationEngine(patient, SimulationConfig(mode="awake", concomitant_opioids=True, rng_seed=1))
    engine.start()
    engine.set_airway_mode("Mask")
    engine.set_fgf(6.0, 0.0)
    if amount(case, "intraop_ftn"):
        engine.give_drug_bolus("fentanyl", amount(case, "intraop_ftn"))
    for _ in range(120):
        engine.step(1.0)
    for second in range(300):
        engine.infusion_rates["propofol"] = amount(case, "intraop_ppf") / 20.0 if second < 20 else 0.0
        engine.step(1.0)
        if engine.state.bis < 60.0:
            return second + 1.0
    raise Excluded("model BIS does not fall below 60")


def _worker_init(override: dict[str, Any], cache: Path) -> None:
    vitaldb_data.CACHE = cache
    apply_hemo_override(override)


def evaluate(candidate, dt: float) -> dict[str, Any]:
    case, tracks = candidate
    legacy_sevo_core = is_legacy_core(case) and case["asa"] in ("1", "2") and case["preop_htn"] == "0"
    row: dict[str, Any] = dict(
        caseid=int(case["caseid"]),
        subjectid=int(case["subjectid"]),
        treated=treated(case),
        procedure=procedure(case),
        department=case["department"],
        asa=case["asa"],
        hypertensive=case["preop_htn"] == "1",
        legacy_core=legacy_sevo_core,
    )
    try:
        drop = bis_drop_time(tracks)
        sizing = patient_from_case(case)
        onset = drop - bis_latency(case, sizing) + BIS_LAG_S
        incision, opend = float(case["opstart"]), float(case["opend"])
        if incision - onset < 300:
            raise Excluded("less than five minutes from induction to incision")
        gases = {name: numeric_track(tracks[name]) for name in ("Primus/INSP_SEVO", "Primus/FIN2O")}
        if np.max(gases["Primus/FIN2O"][1]) > 1:
            raise Excluded("nitrous oxide exposure")
        sevo_t, sevo_v = gases["Primus/INSP_SEVO"]
        if np.any(sevo_v[sevo_t < onset] > 0.5):
            raise Excluded("sevoflurane before induction")
        if np.max(sevo_v) < 0.5 or np.any((sevo_v < 0) | (sevo_v > 8)):
            raise Excluded("insufficient or invalid sevoflurane")
        end = min(opend, float(sevo_t[-1]))
        start = onset - 180.0
        covered(gases["Primus/INSP_SEVO"], onset, end, 30.0)
        remi_name = next((name for name in REMI_RATES if name in tracks), None)
        grid = time_grid(start, max(incision, end), dt)
        step_dt = np.diff(grid)
        if remi_name:
            remi = numeric_track(tracks[remi_name])
            if np.any((remi[1] < 0) | (remi[1] > 1500)):
                raise Excluded("invalid remifentanil delivery")
            remi_ug = delivered_volume(remi, grid) * REMI_UG_PER_ML[remi_name]
            remi_ug_s = np.diff(remi_ug) / step_dt
        else:
            remi_ug_s = np.zeros(len(step_dt), dtype=float)
        inspired = np.nan_to_num(held(gases["Primus/INSP_SEVO"], grid, 30.0))
        propofol_rate = amount(case, "intraop_ppf") / 20.0
        propofol_mg_s = np.where((grid[:-1] >= onset) & (grid[:-1] < onset + 20.0), propofol_rate, 0.0)
        # Onset is uncertain by about this lag, so the baseline reading must post this much earlier.
        baseline_map, baseline_hr = pre_induction(tracks, onset - BIS_LAG_S)
        patient = patient_from_case(case, baseline_map=baseline_map, baseline_hr=baseline_hr)
        times, cuff_map, cuff_sbp, cuff_hr = readings(tracks, onset, incision)
        if not len(times):
            raise Excluded("no cuff reading between induction and incision")
        ebl = amount(case, "intraop_ebl")
        bleed = ebl / max(1.0, (opend - incision) / 60.0)
        events = [
            (incision, lambda e: e.start_disturbance("stim_sustained_surgery")),
            (opend, lambda e: e.stop_disturbance()),
        ]
        if amount(case, "intraop_ftn"):
            events.append((onset - 120.0, lambda e: e.give_drug_bolus("fentanyl", amount(case, "intraop_ftn"))))
        events.append((onset + INTUBATION_S, lambda e: e.start_disturbance("stim_intubation_pulse")))
        if ebl:
            events += [(incision, lambda e: e.start_hemorrhage(bleed)), (opend, lambda e: e.stop_hemorrhage())]
        predicted = run_hemodynamic_replay(
            case, patient, grid, onset=onset, events=events,
            propofol_mg_s=propofol_mg_s, remi_ug_s=remi_ug_s, inspired_sevo=inspired,
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
        "fentanyl": amount(case, "intraop_ftn") > 0,
        "remifentanil": remi_name is not None,
        "median_error": float(np.median(errors)),
    }
    for rule, limits in (("primary", STOP), ("sensitivity", SENSITIVITY)):
        summary[rule] = rule_stats(errors, seconds, cuff_map, cuff_sbp, cuff_hr, limits)
    summary["posting"] = rule_stats(
        np.interp(times, grid, predicted[:, 0]) - cuff_map, seconds, cuff_map, cuff_sbp, cuff_hr, STOP,
    )
    summary["course"] = course(seconds, cuff_map, model_map, baseline_map, incision - onset)
    expired = numeric_track(tracks["Primus/EXP_SEVO"]) if "Primus/EXP_SEVO" in tracks else None
    summary["end_tidal"] = {}
    for name, t0, t1 in (("before incision", incision - 600.0, incision), ("surgery", incision + 900.0, end)):
        sample = np.arange(t0, t1, SAMPLE_SECONDS)
        if expired is None or len(sample) < 60:
            continue
        recorded = held(expired, sample, 30.0)
        valid = np.isfinite(recorded) & (recorded > 0.1)
        if valid.sum() >= 60:
            summary["end_tidal"][name] = (
                float(np.median(recorded[valid])),
                float(np.median(np.interp(sample[valid], grid, predicted[:, 2]))),
            )
    hr_track = numeric_track(tracks["Solar8000/HR"])
    summary["hr_errors"] = []
    for low, high in HR_WINDOWS:
        sample = np.arange(onset + low, min(onset + high, incision), SAMPLE_SECONDS)
        recorded = held(hr_track, sample, 30.0) if len(sample) else np.array([])
        valid = np.isfinite(recorded) & (recorded > 30) & (recorded < 160)
        summary["hr_errors"].append(
            float(np.median(np.interp(sample[valid], grid, predicted[:, 1]) - recorded[valid]))
            if valid.sum() >= 12 else None
        )
    summary["maintenance"] = None
    if end - (incision + 900.0) >= 600.0:
        maint = window_medians(tracks, grid, predicted, incision + 900.0, end)
        if maint:
            summary["maintenance"] = {
                k: maint[k]["predicted"] - maint[k]["observed"] for k in ("map", "hr")
            }
    return summary


def report(data: dict[str, Any], settings: dict[str, Any]) -> str:
    rng = np.random.default_rng(settings["seed"])
    rows = data["rows"]
    untreated = [r for r in rows if not r["treated"]]
    groups = hemo_subgroup_rows(
        rows, legacy_label="Legacy core (open supine, age ≤ 70, ASA 1-2, normotensive, BMI ≤ 32)",
    )
    override = [f"Model override: `{json.dumps(settings['override'])}`.", ""] if settings.get("override") else []
    n_hb = sum(1 for r in rows if r.get("has_preop_hb"))
    n_cr = sum(1 for r in rows if r.get("has_preop_cr"))
    lines = report_header("VitalDB bolus induction with sevoflurane", settings) + override + [
        "## Methods", "",
        f"Propofol runs over 20 s, starting {BIS_LAG_S:.0f} s after the time at which model BIS would first fall below",
        "60 when recorded BIS does. Fentanyl is given 2 minutes earlier. Recorded inspired sevoflurane and remifentanil",
        f"pumps are replayed, with intubation stimulation {INTUBATION_S / 60:.1f} minutes after propofol and surgery",
        "from incision. Eligibility covers all unique-subject oral single-lumen general anesthetics inside AnaSim's",
        f"patient domain (ASA 1-3, normotensive and hypertensive), wiring recorded preop hemoglobin ({n_hb}/{len(rows)})",
        f"and Cockcroft-Gault CrCl ({n_cr}/{len(rows)}).", "",
        f"From propofol to incision, each cuff reading is compared with model MAP averaged over the {CUFF_WINDOW_S:.0f} s",
        "before the reading appears. A patient's readings stop at the first one with MAP below 65 mmHg, SBP below 90, or",
        "HR below 45. A likelihood fit with a patient offset and reading noise gives the mean error (model minus recorded),",
        f"with 95% intervals from {BOOTSTRAP} patient bootstrap resamples.", "",
        f"Eligible candidates: {data['eligible']}. Retained: {len(rows)} ({sum(r['fentanyl'] for r in rows)} with fentanyl, "
        f"{sum(r['remifentanil'] for r in rows)} with remifentanil).", "",
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
        lines += cv_stability_lines(data.get("cv_stability"))
        lines += course_table(rows)
        for name in ("before incision", "surgery"):
            pairs = np.array([r["end_tidal"][name] for r in rows if name in r["end_tidal"]])
            if len(pairs):
                lines += [
                    "",
                    f"End-tidal sevoflurane {name}, median: recorded {np.median(pairs[:, 0]):.2f}%, model "
                    f"{np.median(pairs[:, 1]):.2f}% (n={len(pairs)}).",
                ]
    hr_lines = []
    for index, name in enumerate(("0-5 min", "5-10 min", "10 min to incision")):
        values = [r["hr_errors"][index] for r in untreated if r["hr_errors"][index] is not None]
        if values:
            hr_lines.append(f"{name} {mean_ci(values, rng)}")
    if hr_lines:
        lines += ["", "ECG HR error (model minus recorded, per-patient medians, untreated): " + "; ".join(hr_lines) + " bpm."]
    maintenance = [r["maintenance"] for r in untreated if r["maintenance"]]
    if maintenance:
        lines += [
            "",
            "Surgery from 15 min after incision, untreated: MAP "
            f"{mean_ci([m['map'] for m in maintenance], rng)} mmHg; HR {mean_ci([m['hr'] for m in maintenance], rng)} bpm.",
        ]
    lines += cv_grid_lines(data.get("cv_grid"))
    lines += report_footer(data["exclusions"])
    return "\n".join(lines)


def _run_pool(pool, dt: float, workers: int, override: dict[str, Any], cache: Path) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(override, cache)) as executor:
        return list(executor.map(evaluate, pool, [dt] * len(pool), chunksize=4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--override", default="{}", help="JSON HemodynamicConfig fields, for model comparisons")
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
    stem = args.output / "vitaldb_sevo"
    stem.with_suffix(".json").write_text(json.dumps(dict(settings=settings, **data), indent=2, allow_nan=False) + "\n")
    stem.with_suffix(".md").write_text(report(data, settings))
    print(f"Report: {stem.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
