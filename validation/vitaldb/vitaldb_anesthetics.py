"""Compare fixed AnaSim propofol/BIS and sevoflurane uptake models with VitalDB.

Replays recorded drug delivery rather than pump-predicted concentrations across
VitalDB patients inside AnaSim's supported domain, passing available preop
hemoglobin and creatinine clearance into ``Patient``. No parameters, offsets, or
time shifts are fitted. Only deidentified case summaries are saved in Git.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anasim.core.engine import SimulationEngine  # noqa: E402
from anasim.core.monitors import BIS_DISPLAY_TAU_S  # noqa: E402
from anasim.core.state import SimulationConfig  # noqa: E402
from validation.vitaldb import vitaldb_data  # noqa: E402
from validation.vitaldb.vitaldb_data import (  # noqa: E402
    CACHE,
    REMI_RATES,
    SOURCE,
    Excluded,
    amount,
    covered,
    delivered_volume,
    domain_cases,
    has_plausible_creatinine,
    held,
    is_legacy_core,
    numeric_track,
    optional_float,
    patient_from_case,
    time_grid,
)
from validation.vitaldb.vitaldb_replay import delivery  # noqa: E402

TIVA_TRACKS = [
    "Orchestra/PPF20_RATE", "Orchestra/PPF20_VOL", "BIS/BIS",
    "BIS/SQI", "Primus/MAC", "Primus/FIN2O",
]
SEVO_TRACKS = [
    "Primus/INSP_SEVO", "Primus/EXP_SEVO", "Primus/TV",
    "Primus/RR_CO2", "Primus/ETCO2", "Primus/FIN2O",
]
SAMPLE_SECONDS = 5.0
SQI_MIN = 90.0
PHASE_MIN_SECONDS = 60.0

__all__ = [
    "REMI_RATES",
    "SEVO_TRACKS",
    "TIVA_TRACKS",
    "Excluded",
    "covered",
    "delivered_volume",
    "evaluate_sevo",
    "evaluate_tiva",
    "held",
    "numeric_track",
    "patient_for",
    "replay_propofol",
    "replay_sevo",
    "time_grid",
]


def patient_for(case: dict[str, Any]):
    return patient_from_case(case)


def replay_propofol(patient, grid: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """PPF20 is 20 mg/mL; model input is mg/s and effect-site output is mcg/mL."""
    engine = SimulationEngine(patient, SimulationConfig(mode="awake", concomitant_opioids=True))
    pk, bis = engine.pk["propofol"], engine.bis
    predicted = np.empty(len(grid))
    smooth = predicted[0] = bis.baseline
    for i, dt in enumerate(np.diff(grid), start=1):
        pk.step(float(dt), float((volume[i] - volume[i - 1]) * 20.0 / dt))
        target = bis.step(float(dt), pk.state.ce)
        smooth += -math.expm1(-dt / BIS_DISPLAY_TAU_S) * (target - smooth)
        predicted[i] = smooth
    return predicted


def replay_sevo(patient, grid: np.ndarray, inspired: np.ndarray, tidal_volume: np.ndarray, rr: np.ndarray, cardiac_output: float) -> np.ndarray:
    """Use production uptake parameters, patient FRC, and PBW-based dead space."""
    engine = SimulationEngine(patient, SimulationConfig(mode="awake"))
    pk = engine.pk_sevo
    dead_space = 1000.0 * engine.resp.vd_deadspace  # mL.
    va = rr * np.maximum(0.0, tidal_volume - dead_space) / 1000.0
    predicted = np.zeros(len(grid))
    for i, dt in enumerate(np.diff(grid), start=1):
        pk.step(float(dt), float(inspired[i - 1] / 100.0), float(va[i - 1]), cardiac_output)
        predicted[i] = 100.0 * pk.state.p_alv
    return predicted


def phase_metrics(time, observed, predicted, baseline, valid, intervals):
    out = {}
    for phase, (start, end) in {"all": (time[0], time[-1] + SAMPLE_SECONDS), **intervals}.items():
        mask = valid & (time >= start) & (time < end)
        if mask.sum() * SAMPLE_SECONDS < PHASE_MIN_SECONDS:
            continue
        difference = predicted[mask] - observed[mask]
        out[phase] = dict(
            n=int(mask.sum()),
            minutes=float(mask.sum() * SAMPLE_SECONDS / 60.0),
            bias=float(np.mean(difference)),
            mae=float(np.mean(np.abs(difference))),
            rmse=float(np.sqrt(np.mean(difference**2))),
            baseline_mae=float(np.mean(np.abs(baseline[mask] - observed[mask]))),
        )
    if "all" not in out or out["all"]["minutes"] < 10.0:
        raise Excluded("less than ten minutes of usable endpoint measurements")
    return out


def evaluate_tiva(case: dict[str, Any], tracks: dict[str, str], dt: float) -> dict[str, Any]:
    rate, volume, _remi, _remi_name, start, end, onset, stop = delivery(
        tracks,
        max_gap_s=5.0,
        require_gas_before_onset=True,
        track_loader=numeric_track,
    )
    bis_series = numeric_track(tracks["BIS/BIS"])
    sqi_series = numeric_track(tracks["BIS/SQI"])
    end = min(end, float(bis_series[0][-1]), float(sqi_series[0][-1]))
    if end - onset < 900:
        raise Excluded("short replay interval")
    grid = time_grid(start, end, dt)
    delivered = delivered_volume(rate, grid)
    measured = np.interp(grid[-1], *volume) - np.interp(grid[0], *volume)
    error = float((delivered[-1] - delivered[0]) - measured)
    if abs(error) > max(0.5, 0.05 * measured):
        raise Excluded("infusion rate and volume disagree by more than five percent or 0.5 mL")
    patient = patient_for(case)
    prediction = replay_propofol(patient, grid, delivered)
    sample = np.arange(onset, end, SAMPLE_SECONDS)
    observed = held(bis_series, sample, 2.0)
    sqi = held(sqi_series, sample, 2.0)
    predicted = np.interp(sample, grid, prediction)
    valid = np.isfinite(observed) & (observed >= 0) & (observed <= 100) & (sqi >= SQI_MIN) & (sqi <= 100)
    opstart, opend = float(case["opstart"]), float(case["opend"])
    intervals = dict(
        induction=(onset, min(onset + 600, opstart)),
        maintenance=(max(onset + 600, opstart), min(stop, opend)),
        emergence=(max(stop, opend), end),
    )
    metrics = phase_metrics(sample, observed, predicted, np.full(len(sample), 40.0), valid, intervals)
    return dict(
        metrics=metrics,
        dose_mg=float((delivered[-1] - delivered[0]) * 20.0),
        volume_disagreement_ml=error,
        valid_fraction=float(np.mean(valid)),
        _trace=(sample, observed, predicted, valid, onset),
        replay_start=float(start),
        replay_end=float(end),
        onset=onset,
        stop=stop,
    )


def evaluate_sevo(case: dict[str, Any], tracks: dict[str, str], dt: float) -> dict[str, Any]:
    series = {"Primus/INSP_SEVO": numeric_track(tracks["Primus/INSP_SEVO"])}
    fi = series["Primus/INSP_SEVO"]
    if fi[1][0] > 0.1 or np.max(fi[1]) < 1:
        raise Excluded("sevoflurane exposure already present or insufficient exposure")
    if np.any((fi[1] < 0) | (fi[1] > 8)):
        raise Excluded("inspired sevoflurane outside zero to eight percent")
    for name in SEVO_TRACKS[1:]:
        series[name] = numeric_track(tracks[name])
    if series["Primus/EXP_SEVO"][1][0] > 0.1:
        raise Excluded("expired sevoflurane already present at recording start")
    if np.max(series["Primus/FIN2O"][1]) > 1:
        raise Excluded("recorded nitrous oxide exposure")
    start = max(s[0][0] for s in series.values())
    end = min(s[0][-1] for s in series.values())
    exposed = np.flatnonzero((fi[1] >= 0.5) & (fi[0] >= start) & (fi[0] <= end))
    if len(exposed) < 2:
        raise Excluded("insufficient sevoflurane exposure in common interval")
    onset, stop = float(fi[0][exposed[0]]), float(fi[0][exposed[-1]])
    if onset - start < 30 or stop - onset < 900:
        raise Excluded("short zero-agent baseline or exposure interval")
    if np.any(fi[1][fi[0] < start] > 0.1):
        raise Excluded("ventilation recording began after sevoflurane exposure")
    for name in ("Primus/INSP_SEVO", "Primus/TV", "Primus/RR_CO2", "Primus/FIN2O"):
        covered(series[name], start, end, 30.0)
    grid = time_grid(start, end, dt)
    inputs = [held(series[name], grid, 30.0) for name in ("Primus/INSP_SEVO", "Primus/TV", "Primus/RR_CO2")]
    if not all(np.all(np.isfinite(v)) for v in inputs):
        raise Excluded("incomplete input history")
    inspired, vt, rr = inputs
    if np.any((vt < 0) | (vt > 1500) | (rr < 0) | (rr > 50)):
        raise Excluded("ventilation inputs outside comparison range")
    sample = np.arange(onset, end, SAMPLE_SECONDS)
    observed = held(series["Primus/EXP_SEVO"], sample, 15.0)
    co2 = held(series["Primus/ETCO2"], sample, 15.0)
    measured_vt = held(series["Primus/TV"], sample, 15.0)
    measured_rr = held(series["Primus/RR_CO2"], sample, 15.0)
    valid = (
        np.isfinite(observed) & (observed >= 0) & (observed <= 8)
        & (co2 >= 15) & (co2 <= 60) & (measured_vt >= 200) & (measured_rr >= 5)
    )
    intervals = dict(washin=(onset, onset + 600), maintenance=(onset + 600, stop), washout=(stop, end))
    baseline = held(fi, sample, 15.0)
    valid &= np.isfinite(baseline)
    patient = patient_for(case)
    predictions = {co: np.interp(sample, grid, replay_sevo(patient, grid, inspired, vt, rr, co)) for co in (3.0, 5.0, 7.0)}
    metrics = phase_metrics(sample, observed, predictions[5.0], baseline, valid, intervals)
    sensitivity = {
        str(co): phase_metrics(sample, observed, prediction, baseline, valid, intervals)
        for co, prediction in predictions.items()
    }
    return dict(
        metrics=metrics,
        cardiac_output_sensitivity=sensitivity,
        valid_fraction=float(np.mean(valid)),
        _trace=(sample, observed, predictions[5.0], valid, onset),
        replay_start=float(start),
        replay_end=float(end),
        onset=onset,
        stop=stop,
    )


def evaluate(candidate, kind: str, dt: float) -> dict[str, Any]:
    case, tracks = candidate
    try:
        result = (evaluate_tiva if kind == "tiva" else evaluate_sevo)(case, tracks, dt)
    except Excluded as error:
        return dict(caseid=int(case["caseid"]), subjectid=int(case["subjectid"]), exclusion=str(error))
    weight, height = float(case["weight"]), float(case["height"])
    result.update(
        caseid=int(case["caseid"]),
        subjectid=int(case["subjectid"]),
        age=float(case["age"]),
        weight=weight,
        height=height,
        bmi=round(weight / (height / 100.0) ** 2, 2),
        sex=case["sex"],
        legacy_core=is_legacy_core(case),
        has_preop_hb=optional_float(case, "preop_hb") is not None,
        has_preop_cr=has_plausible_creatinine(case),
        tracks={name: tracks[name] for name in (TIVA_TRACKS if kind == "tiva" else SEVO_TRACKS)},
    )
    if kind == "tiva":
        result["tracks"].update({name: tracks[name] for name in REMI_RATES if name in tracks})
    return result


def cohort(kind: str, requested: int, seed: int, workers: int, dt: float) -> dict[str, Any]:
    candidates = domain_cases(TIVA_TRACKS if kind == "tiva" else SEVO_TRACKS, open_supine_only=False)
    initial = len(candidates)
    if kind == "tiva":
        candidates = [
            (c, t) for c, t in candidates
            if any(n in t for n in REMI_RATES)
            and all(amount(c, n) == 0 for n in ("intraop_ppf", "intraop_mdz", "intraop_ftn"))
            and not any(n in t for n in ("Orchestra/DEX2_RATE", "Orchestra/DEX4_RATE"))
        ]
    random.Random(seed).shuffle(candidates)
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(candidates), workers):
            if requested and len(selected) >= requested:
                break
            batch = candidates[offset:offset + workers]
            results = list(pool.map(lambda candidate: evaluate(candidate, kind, dt), batch))
            for result in results:
                if requested and len(selected) >= requested:
                    break
                if "exclusion" in result:
                    exclusions.append(result)
                else:
                    selected.append(result)
            target = str(requested) if requested else "all"
            print(f"{kind}: retained {len(selected)}/{target}; screened {len(selected) + len(exclusions)}", flush=True)
    return dict(
        requested=requested,
        track_eligible=initial,
        metadata_eligible=len(candidates),
        rows=selected,
        exclusions=exclusions,
    )


def mean_ci(values, rng) -> str:
    arr = np.asarray(values, dtype=float)
    means = np.mean(rng.choice(arr, size=(2000, len(arr)), replace=True), axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return f"{np.mean(arr):.2f} ({low:.2f} to {high:.2f})"


def _subgroup_table(rows: list[dict[str, Any]], rng) -> list[str]:
    groups = [
        ("Legacy core (open supine, age ≤ 70, BMI ≤ 32)", [r for r in rows if r.get("legacy_core")]),
        ("Age under 45", [r for r in rows if r["age"] < 45]),
        ("Age 45-59", [r for r in rows if 45 <= r["age"] < 60]),
        ("Age 60-69", [r for r in rows if 60 <= r["age"] < 70]),
        ("Age 70 and over", [r for r in rows if r["age"] >= 70]),
        ("BMI under 30", [r for r in rows if r["bmi"] < 30.0]),
        ("BMI 30 and over", [r for r in rows if r["bmi"] >= 30.0]),
    ]
    lines = [
        "",
        "| Subgroup (all phases) | Patients | Bias (95% CI) | MAE (95% CI) | RMSE (95% CI) |",
        "|---|---:|---|---|---|",
    ]
    added = 0
    for label, group in groups:
        if len(group) < 5:
            continue
        added += 1
        metrics = [r["metrics"]["all"] for r in group]
        lines.append(
            f"| {label} | {len(group)} | {mean_ci([m['bias'] for m in metrics], rng)} "
            f"| {mean_ci([m['mae'] for m in metrics], rng)} | {mean_ci([m['rmse'] for m in metrics], rng)} |"
        )
    return lines if added else []


def report(data: dict[str, Any], settings: dict[str, Any], seed: int) -> str:
    lines = [
        "# VitalDB anesthetic comparisons",
        "",
        f"Generated {settings['generated']}.",
        "",
        "## Methods",
        "",
        "These comparisons test the anesthetic models in isolation across all eligible unique-subject recordings",
        "inside AnaSim's supported domain (general anesthesia, oral single-lumen tube, age 18 to 85 years, weight",
        "50 to 200 kg, height 150 to 200 cm, BMI 18 to 60 kg/m²), passing preoperative hemoglobin and Cockcroft-Gault",
        "creatinine clearance when recorded. The models use AnaSim's standard parameters, which were not tuned to",
        "these recordings. The replay fits no individual calibrations or time shifts, and predicted BIS does not",
        "start from the observed BIS.",
        "",
        f"Each cohort includes at most one case per patient, and seed {seed} fixes candidate ordering.",
        f"The replay step is {settings['dt']:g} s, and endpoints are sampled every {SAMPLE_SECONDS:g} s. Each reported phase",
        "needs at least one usable minute, and each case needs ten usable minutes in total. Errors are predicted minus",
        "measured, with 95% intervals from 2,000 patient bootstrap resamples.",
    ]
    rng = np.random.default_rng(seed)

    sections = (
        (
            "tiva",
            "Propofol and BIS",
            ("all", "induction", "maintenance", "emergence"),
            [
                "Propofol delivery comes from `PPF20_RATE` (20 mg/mL), integrated in mL and checked against cumulative",
                "`PPF20_VOL` (with syringe-change counter resets unwrapped) within 5% or 0.5 mL. Delivery gaps over 5 s,",
                "supplemental propofol/midazolam/fentanyl boluses, dexmedetomidine pumps, volatile or N₂O exposure, and",
                "BIS samples with SQI below 90 are excluded. The comparator is a constant BIS of 40.",
            ],
        ),
        (
            "sevo",
            "Sevoflurane uptake",
            ("all", "washin", "maintenance", "washout"),
            [
                "Recorded inspired sevoflurane, tidal volume, and respiratory rate drive `VolatilePK` with PBW-based",
                "dead space (2.2 mL/kg PBW), assumed cardiac output 5 L/min (with 3 and 7 L/min sensitivity runs), and",
                "fixed patient FRC. The comparator sets end-tidal concentration equal to recorded inspired concentration.",
            ],
        ),
    )

    for kind, heading, phases, intro in sections:
        group = data.get(kind)
        if not group:
            continue
        rows = group["rows"]
        lines += ["", f"## {heading}", "", *intro, ""]
        target = f"of {group['requested']} requested" if group["requested"] else "(all eligible)"
        lines += [
            f"Cases with the required tracks: {group['track_eligible']}. After metadata rules: {group['metadata_eligible']}.",
            f"Screened: {len(rows) + len(group['exclusions'])}. Retained: {len(rows)} {target}.",
            "",
        ]
        if rows:
            lines += [
                "| Phase | Patients | Usable minutes | Bias (95% CI) | MAE (95% CI) | RMSE (95% CI) | Comparator MAE (95% CI) |",
                "|---|---:|---:|---|---|---|---|",
            ]
            for phase in phases:
                phase_rows = [r["metrics"][phase] for r in rows if phase in r["metrics"]]
                if not phase_rows:
                    continue
                minutes = sum(m["minutes"] for m in phase_rows)
                lines.append(
                    f"| {phase} | {len(phase_rows)} | {minutes:.1f} "
                    f"| {mean_ci([m['bias'] for m in phase_rows], rng)} "
                    f"| {mean_ci([m['mae'] for m in phase_rows], rng)} "
                    f"| {mean_ci([m['rmse'] for m in phase_rows], rng)} "
                    f"| {mean_ci([m['baseline_mae'] for m in phase_rows], rng)} |"
                )
            diff = [r["metrics"]["all"]["mae"] - r["metrics"]["all"]["baseline_mae"] for r in rows]
            better = sum(d < 0 for d in diff)
            cov = np.percentile([100.0 * r["valid_fraction"] for r in rows], [25, 50, 75])
            lines += [
                "",
                f"Paired all-phase MAE difference, AnaSim minus comparator: {mean_ci(diff, rng)}.",
                f"AnaSim has lower MAE in {better} of {len(rows)} patients.",
                f"Usable endpoint coverage: median {cov[1]:.1f}% (IQR {cov[0]:.1f}% to {cov[2]:.1f}%) of the replay sampling interval.",
            ]
            lines += _subgroup_table(rows, rng)
            if kind == "sevo":
                lines += ["", "Cardiac output sensitivity (all-phase mean error, 95% CI):", ""]
                lines += ["| Assumed CO (L/min) | Bias | MAE | RMSE |", "|---:|---|---|---|"]
                for co in ("3.0", "5.0", "7.0"):
                    sens = [r["cardiac_output_sensitivity"][co]["all"] for r in rows]
                    lines.append(
                        f"| {co} | {mean_ci([m['bias'] for m in sens], rng)} "
                        f"| {mean_ci([m['mae'] for m in sens], rng)} | {mean_ci([m['rmse'] for m in sens], rng)} |"
                    )
        if group["exclusions"]:
            lines += ["", "Exclusions among screened cases:", ""]
            lines += [
                f"- {reason}: {count}"
                for reason, count in sorted(Counter(r["exclusion"] for r in group["exclusions"]).items())
            ]
        if group["requested"] and len(rows) < group["requested"]:
            lines += ["", "The requested sample was not reached, so the table reports all eligible patients found."]

    lines += [
        "",
        "## Scope and artifacts",
        "",
        "These comparisons evaluate individual anesthetic models against VitalDB development recordings.",
        "Per-patient errors, track identifiers, and exclusions are in `vitaldb_anesthetics.json` and",
        "`vitaldb_anesthetics_cases.csv`.",
        "",
        f"Source: [Lee et al., Scientific Data, 2022]({SOURCE}). VitalDB data are used under CC BY-NC-SA 4.0",
        "and its data use agreement.",
        "",
    ]
    return "\n".join(lines)


def overlays(group: dict[str, Any], kind: str, out: Path) -> None:
    if not group["rows"]:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(group["rows"], key=lambda row: row["metrics"]["all"]["mae"])
    indices = np.unique(np.rint(np.linspace(0, len(ordered) - 1, min(5, len(ordered)))).astype(int))
    fig, axes = plt.subplots(len(indices), 1, figsize=(11, 2.6 * len(indices)), squeeze=False)
    for axis, index in zip(axes[:, 0], indices, strict=True):
        row = ordered[index]
        time, observed, predicted, valid, onset = row["_trace"]
        minutes = (time - onset) / 60.0
        axis.plot(minutes, np.where(valid, observed, np.nan), color="black", lw=0.8, label="Recorded")
        axis.plot(minutes, predicted, color="#c4442f", lw=1.0, label="AnaSim")
        axis.set_title(f"Case {row['caseid']}; MAE {row['metrics']['all']['mae']:.2f}; error rank {index + 1}/{len(ordered)}", fontsize=10)
        axis.set_ylabel("BIS" if kind == "tiva" else "End-tidal sevo (%)")
        axis.set_xlabel("Minutes after first delivery" if kind == "tiva" else "Minutes after inspired sevo reaches 0.5%")
        axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiva-cases", type=int, default=0, help="Max TIVA patients (0 = all eligible)")
    parser.add_argument("--sevo-cases", type=int, default=0, help="Max sevoflurane patients (0 = all eligible)")
    parser.add_argument("--cohorts", choices=("both", "tiva", "sevo"), default="both")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", action="store_true", help="Write representative overlays (requires Matplotlib).")
    args = parser.parse_args()
    if min(args.tiva_cases, args.sevo_cases) < 0 or args.workers < 1 or not 0 < args.dt <= 1:
        parser.error("request non-negative case limits, positive workers, and a replay step in (0, 1] seconds")
    vitaldb_data.CACHE = args.cache
    args.cache.mkdir(parents=True, exist_ok=True)
    selected_kinds: tuple[tuple[str, int], ...] = (("tiva", args.tiva_cases), ("sevo", args.sevo_cases))
    if args.cohorts != "both":
        selected_kinds = tuple(item for item in selected_kinds if item[0] == args.cohorts)
    data = {
        kind: cohort(kind, count, args.seed + offset, args.workers, args.dt)
        for offset, (kind, count) in enumerate(selected_kinds)
    }
    settings = dict(generated=datetime.now(timezone.utc).date().isoformat(), dt=args.dt, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "vitaldb_anesthetics.md").write_text(report(data, settings, args.seed))
    if args.figures:
        for kind, group in data.items():
            overlays(group, kind, args.output / f"vitaldb_{kind}_overlay.png")
    for group in data.values():
        for row in group["rows"]:
            del row["_trace"]
    (args.output / "vitaldb_anesthetics.json").write_text(json.dumps(dict(settings=settings, **data), indent=2, allow_nan=False) + "\n")
    with (args.output / "vitaldb_anesthetics_cases.csv").open("w", newline="") as handle:
        fields = ["cohort", "caseid", "subjectid", "phase", "n", "minutes", "bias", "mae", "rmse", "baseline_mae"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for kind, group in data.items():
            for row in group["rows"]:
                for phase, metrics in row["metrics"].items():
                    writer.writerow(dict(cohort=kind, caseid=row["caseid"], subjectid=row["subjectid"], phase=phase, **metrics))
    print(f"Report: {args.output / 'vitaldb_anesthetics.md'}", flush=True)


if __name__ == "__main__":
    main()
