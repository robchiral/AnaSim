"""Audit cached VitalDB PEEP steps before incision for recruitment calibration.

Only recorded setting changes with stable volume-controlled ventilation on
both sides are retained. Numeric eligibility is followed by pressure-waveform
coverage, breath-shape, and PEEP agreement checks. These observations measure
an effective compliance response; they do not identify recruitment thresholds or rates by themselves.
Data: VitalDB, Lee et al. 2022, CC BY-NC-SA 4.0. No tracks are downloaded.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import vitaldb_ventilation
from vitaldb_ventilation import (
    CACHE,
    CM_PER_MBAR,
    COMPLIANCE_TRACKS,
    WAVE_DT,
    domain_cases,
    load_track,
    template,
    window_features,
)

SETTINGS = ["Primus/SET_INTER_PEEP", "Primus/SET_RR_IPPV", "Primus/SET_TV_L",
            "Primus/SET_INSP_TM", "Primus/SET_INSP_PAUSE"]
NUMERICS = [*COMPLIANCE_TRACKS, "Primus/RR_CO2"]


def window(series, start, end):
    values = {}
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
    if (driving < 3 or values["TV"] < 200 or values["SET_RR_IPPV"] < 5
            or abs(values["RR_CO2"] - values["SET_RR_IPPV"]) > 1
            or abs(values["PEEP_MBAR"] * CM_PER_MBAR - values["SET_INTER_PEEP"]) > 0.75):
        return None, "inadequate driving pressure, volume, or agreement with settings"
    values["compliance"] = values["TV"] / driving
    return values, None


def waveform_window(series, start, end, settings):
    time, pressure = series
    inside = (time >= start) & (time <= end)
    t, p = time[inside], pressure[inside] * CM_PER_MBAR
    if (len(p) < 0.9 * (end - start) / WAVE_DT or t[-1] - t[0] < 85
            or np.max(np.diff(t)) > 2 * WAVE_DT or not np.all(np.isfinite(p))):
        return None, "pressure waveform: insufficient coverage or nonfinite values"
    breath = template(p, settings["SET_RR_IPPV"])
    if breath is None:
        return None, "pressure waveform: insufficient aligned breaths"
    features = window_features(settings, breath)
    # Keep the measured mechanics range and constant-flow/pause shape checks
    # used in the main comparison. Pressure alone does not prove a mode.
    out = {key: bool(value) if isinstance(value, np.bool_) else float(value)
           for key, value in features.items() if key in ("peep", "r", "c", "volume_control")}
    if not out["volume_control"]:
        return out, "pressure waveform: control shape or mechanics outside comparison range"
    # The pressure trace can have an approximately 1 cmH2O baseline offset at
    # nominal zero PEEP; larger disagreement makes the paired step unusable.
    if abs(out["peep"] - settings["SET_INTER_PEEP"]) > 1.25:
        return out, "pressure waveform: baseline disagrees with set PEEP"
    return out, None


def audit(cache=CACHE):
    for name in ("cases.csv", "trks.csv"):
        if not (cache / name).exists():
            raise FileNotFoundError(f"Cached VitalDB metadata required: {cache / name}")
    vitaldb_ventilation.CACHE = cache
    counts, rejected, rows, exclusions = Counter(), Counter(), [], []
    for case, tracks in domain_cases(COMPLIANCE_TRACKS):
        counts["eligible_cases"] += 1
        if "Primus/SET_INSP_PRES" in tracks:
            continue
        required = [*SETTINGS, *NUMERICS]
        if not all(name in tracks and (cache / (tracks[name] + ".csv")).exists() for name in required):
            continue
        counts["cached_complete_cases"] += 1
        times, peep = load_track(tracks["Primus/SET_INTER_PEEP"])
        settled, incision = float(case["anestart"]) + 600, float(case["opstart"])
        changes = np.flatnonzero(np.abs(np.diff(peep)) >= 2) + 1
        candidates = [i for i in changes if settled <= times[i] - 120 and times[i] + 120 <= incision]
        if not candidates:
            continue
        series = {name: load_track(tracks[name]) for name in required}
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
                exclusions.append(dict(case=int(case["caseid"]), time=float(stamp),
                                       before=before_reason, after=after_reason))
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
            if wave_name not in tracks or not (cache / (tracks[wave_name] + ".csv")).exists():
                rejected["pressure_waveform_not_cached"] += 1
                continue
            wave = load_track(tracks[wave_name], wave=True)
            before_wave, before_reason = waveform_window(wave, stamp - 120, stamp - 30, before)
            after_wave, after_reason = waveform_window(wave, stamp + 30, stamp + 120, after)
            if before_reason or after_reason:
                rejected["pressure_waveform_failed"] += 1
                exclusions.append(dict(case=int(case["caseid"]), time=float(stamp),
                                       before=before_reason, after=after_reason,
                                       waveforms=dict(before=before_wave, after=after_wave)))
                continue
            rows.append(dict(case=int(case["caseid"]),time=float(stamp),before=before,after=after,
                             waveforms=dict(before=before_wave, after=after_wave),
                             log_gain_per_cmh2o=float(np.log(after["compliance"] / before["compliance"])
                                                      / (after["SET_INTER_PEEP"] - before["SET_INTER_PEEP"]))))
    counts["retained_steps"], counts["retained_patients"] = len(rows), len({r["case"] for r in rows})
    return dict(counts=counts, rejected=rejected, exclusions=exclusions, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    result = audit(args.cache)
    out = Path(__file__).parent / "results" / "vitaldb_peep.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2) + "\n")
    print(json.dumps({key:value for key,value in result.items() if key != "rows"},indent=2))
    for row in result["rows"]:
        print(row["case"],row["time"],[(round(row[key][field],3)) for key in ("before","after")
                                       for field in ("SET_INTER_PEEP","TV","compliance")])
