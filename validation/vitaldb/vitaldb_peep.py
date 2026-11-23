"""Audit VitalDB PEEP steps before incision for recruitment calibration.

Only recorded setting changes with stable volume-controlled ventilation on
both sides are retained. Numeric eligibility is followed by pressure-waveform
coverage, breath-shape, and PEEP agreement checks. By default only cached
tracks are read; pass ``--fetch`` for two-stage downloading (screening tiny
``SET_INTER_PEEP`` tracks first and fetching remaining tracks only for cases
with a candidate pre-incision PEEP step).

Data: VitalDB, Lee et al. 2022, CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    CM_PER_MBAR,
    WAVE_DT,
    domain_cases,
    load_track,
    pft_group,
    prefetch_tracks,
)
from validation.vitaldb.vitaldb_ventilation import (  # noqa: E402
    COMPLIANCE_TRACKS,
    template,
    window_features,
)

SETTINGS = [
    "Primus/SET_INTER_PEEP", "Primus/SET_RR_IPPV", "Primus/SET_TV_L",
    "Primus/SET_INSP_TM", "Primus/SET_INSP_PAUSE",
]
NUMERICS = [*COMPLIANCE_TRACKS, "Primus/RR_CO2"]


def window(series: dict[str, tuple[np.ndarray, np.ndarray]], start: float, end: float):
    values: dict[str, float] = {}
    for name, (time, data) in series.items():
        inside = (time >= start) & (time <= end)
        t, v = time[inside], data[inside]
        if len(v) < 10 or t[-1] - t[0] < 75 or np.max(np.diff(t)) > 15 or not np.all(np.isfinite(v)):
            return None, f"{name}: insufficient coverage or nonfinite values"
        median = np.median(v)
        if name.startswith("Primus/SET_"):
            if np.ptp(v) > 1e-6:
                return None, f"{name}: setting changed in window"
        elif name in COMPLIANCE_TRACKS and np.ptp(v) > max(1.0, 0.15 * abs(median)):
            return None, f"{name}: numeric unstable in window"
        values[name.split("/")[1]] = float(median)
    driving = (values["PPLAT_MBAR"] - values["PEEP_MBAR"]) * CM_PER_MBAR
    if (
        driving < 3 or values["TV"] < 200 or values["SET_RR_IPPV"] < 5
        or abs(values["RR_CO2"] - values["SET_RR_IPPV"]) > 1
        or abs(values["PEEP_MBAR"] * CM_PER_MBAR - values["SET_INTER_PEEP"]) > 0.75
    ):
        return None, "inadequate driving pressure, volume, or agreement with settings"
    values["compliance"] = values["TV"] / driving
    return values, None


def waveform_window(series: tuple[np.ndarray, np.ndarray], start: float, end: float, settings: dict[str, float]):
    time, pressure = series
    inside = (time >= start) & (time <= end)
    t, p = time[inside], pressure[inside] * CM_PER_MBAR
    if (
        len(p) < 0.9 * (end - start) / WAVE_DT or t[-1] - t[0] < 85
        or np.max(np.diff(t)) > 2 * WAVE_DT or not np.all(np.isfinite(p))
    ):
        return None, "pressure waveform: insufficient coverage or nonfinite values"
    breath = template(p, settings["SET_RR_IPPV"])
    if breath is None:
        return None, "pressure waveform: insufficient aligned breaths"
    features = window_features(settings, breath)
    out = {
        key: bool(value) if isinstance(value, np.bool_) else float(value)
        for key, value in features.items() if key in ("peep", "r", "c", "volume_control")
    }
    if not out["volume_control"]:
        return out, "pressure waveform: control shape or mechanics outside comparison range"
    # The pressure trace can have an approximately 1 cmH2O baseline offset at
    # nominal zero PEEP; larger disagreement makes the paired step unusable.
    if abs(out["peep"] - settings["SET_INTER_PEEP"]) > 1.25:
        return out, "pressure waveform: baseline disagrees with set PEEP"
    return out, None


def audit(cache: Path = CACHE, *, fetch_missing: bool = False, workers: int = 16, seed: int = 20261024) -> dict[str, Any]:
    if not fetch_missing:
        for name in ("cases.csv", "trks.csv"):
            if not (cache / name).exists():
                raise FileNotFoundError(f"Cached VitalDB metadata required: {cache / name}")
    vitaldb_data.CACHE = cache
    counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    required = [*SETTINGS, *NUMERICS]

    eligible = domain_cases(
        required,
        open_supine_only=False,
        exclude_tracks=("Primus/SET_INSP_PRES",),
        cache=cache,
    )
    if fetch_missing:
        counts["fetched_peep_tracks"] = prefetch_tracks(
            (tracks["Primus/SET_INTER_PEEP"] for _, tracks in eligible),
            workers=workers,
            cache=cache,
        )
    for case, tracks in eligible:
        counts["eligible_cases"] += 1
        peep_tid = tracks["Primus/SET_INTER_PEEP"]
        if not fetch_missing and not all((cache / (tracks[name] + ".csv")).exists() for name in required):
            continue
        if not fetch_missing:
            counts["cached_complete_cases"] += 1
        times, peep = load_track(peep_tid, cache=cache)
        try:
            settled, incision = float(case["anestart"]) + 600, float(case["opstart"])
        except ValueError:
            continue
        changes = np.flatnonzero(np.abs(np.diff(peep)) >= 2) + 1
        candidates = [i for i in changes if settled <= times[i] - 120 and times[i] + 120 <= incision]
        if not candidates:
            continue
        counts["cases_with_candidate_step"] += 1
        series = {
            name: (times, peep) if name == "Primus/SET_INTER_PEEP" else load_track(tracks[name], cache=cache)
            for name in required
        }
        wave = None
        for i in candidates:
            counts["candidate_steps"] += 1
            stamp = times[i]
            if times[i] - times[i - 1] > 15:
                rejected["setting_gap"] += 1
                continue
            before, before_reason = window(series, stamp - 120, stamp - 30)
            after, after_reason = window(series, stamp + 30, stamp + 120)
            if before is None or after is None:
                rejected["unstable_or_missing_window"] += 1
                exclusions.append(dict(
                    case=int(case["caseid"]), time=float(stamp),
                    before=before_reason, after=after_reason,
                ))
                continue
            if any(before[key] != after[key] for key in ("SET_RR_IPPV", "SET_TV_L", "SET_INSP_TM", "SET_INSP_PAUSE")):
                rejected["other_setting_changed"] += 1
                continue
            if abs(after["TV"] / before["TV"] - 1) > 0.05:
                rejected["tidal_volume_changed"] += 1
                continue
            if abs(after["SET_INTER_PEEP"] - before["SET_INTER_PEEP"]) < 2:
                rejected["no_sustained_step"] += 1
                continue
            counts["numeric_eligible_steps"] += 1
            wave_name = "Primus/AWP"
            if wave_name not in tracks:
                rejected["pressure_waveform_missing"] += 1
                continue
            if not fetch_missing and not (cache / (tracks[wave_name] + ".csv")).exists():
                rejected["pressure_waveform_not_cached"] += 1
                continue
            if wave is None:
                wave = load_track(tracks[wave_name], wave=True, cache=cache)
            before_wave, before_reason = waveform_window(wave, stamp - 120, stamp - 30, before)
            after_wave, after_reason = waveform_window(wave, stamp + 30, stamp + 120, after)
            if before_reason or after_reason:
                rejected["pressure_waveform_failed"] += 1
                exclusions.append(dict(
                    case=int(case["caseid"]), time=float(stamp),
                    before=before_reason, after=after_reason,
                    waveforms=dict(before=before_wave, after=after_wave),
                ))
                continue
            rows.append(dict(
                case=int(case["caseid"]),
                subjectid=case.get("subjectid", case["caseid"]),
                bmi=round(float(case["weight"]) / (float(case["height"]) / 100.0) ** 2, 2),
                pft=pft_group(case),
                time=float(stamp),
                before=before,
                after=after,
                waveforms=dict(before=before_wave, after=after_wave),
                log_gain_per_cmh2o=float(
                    np.log(after["compliance"] / before["compliance"])
                    / (after["SET_INTER_PEEP"] - before["SET_INTER_PEEP"])
                ),
            ))
    counts["retained_steps"] = len(rows)
    counts["retained_patients"] = len({r["subjectid"] for r in rows})
    summary: dict[str, Any] = {}
    if rows:
        by_patient: dict[str, list[float]] = {}
        for r in rows:
            by_patient.setdefault(r["subjectid"], []).append(r["log_gain_per_cmh2o"])
        patient_means = np.array([np.mean(v) for v in by_patient.values()], dtype=float)
        summary["mean_log_gain_per_cmh2o"] = float(np.mean(patient_means))
        summary["median_log_gain_per_cmh2o"] = float(np.median(patient_means))
        if len(patient_means) >= 3:
            rng = np.random.default_rng(seed)
            boot = np.mean(rng.choice(patient_means, size=(2000, len(patient_means)), replace=True), axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            summary["ci_95"] = [float(lo), float(hi)]
    return dict(counts=dict(counts), rejected=dict(rejected), summary=summary, exclusions=exclusions, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--fetch", action="store_true", help="Download missing PEEP tracks in two stages")
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    result = audit(args.cache, fetch_missing=args.fetch, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "vitaldb_peep.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    for row in result["rows"]:
        print(
            row["case"], row["time"],
            [round(row[key][field], 3) for key in ("before", "after") for field in ("SET_INTER_PEEP", "TV", "compliance")],
        )
