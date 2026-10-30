"""Compare fixed AnaSim propofol/BIS and sevoflurane uptake models with VitalDB.

Replay recorded drug delivery rather than pump-predicted concentrations. No
parameters, offsets, or time shifts are fitted. Downloads use the ventilation
script's existing cache. Only deidentified case summaries are saved in Git.
"""

import argparse
import csv
import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from vitaldb_ventilation import CACHE, ROOT, domain_cases, load_track

from anasim.core.engine import SimulationEngine
from anasim.core.monitors import BIS_DISPLAY_TAU_S
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient

TIVA_TRACKS = ["Orchestra/PPF20_RATE", "Orchestra/PPF20_VOL", "BIS/BIS",
               "BIS/SQI", "Primus/MAC", "Primus/FIN2O"]
SEVO_TRACKS = ["Primus/INSP_SEVO", "Primus/EXP_SEVO", "Primus/TV",
               "Primus/RR_CO2", "Primus/ETCO2", "Primus/FIN2O"]
REMI_RATES = ("Orchestra/RFTN20_RATE", "Orchestra/RFTN50_RATE")
SAMPLE_SECONDS = 5.0
SQI_MIN = 90.0
PHASE_MIN_SECONDS = 60.0
SOURCE = "https://doi.org/10.1038/s41597-022-01411-5"


class Excluded(ValueError):
    """A recording fails a prespecified input or measurement quality rule."""


def numeric_track(tid):
    time, values = load_track(tid)
    finite = np.isfinite(time) & np.isfinite(values)
    time, values = time[finite], values[finite]
    order = np.argsort(time, kind="stable")
    time, values = time[order], values[order]
    # Keep the last packet at an identical timestamp.
    unique = np.r_[np.diff(time) > 0, True] if len(time) else np.array([], dtype=bool)
    time, values = time[unique], values[unique]
    if len(time) < 2:
        raise Excluded("insufficient finite track data")
    return time, values


def held(series, grid, max_age):
    """Use the last recorded value; never bridge a gap beyond max_age seconds."""
    time, values = series
    index = np.searchsorted(time, grid, side="right") - 1
    clipped = np.maximum(index, 0)
    valid = (index >= 0) & (grid <= time[-1]) & (grid - time[clipped] <= max_age)
    return np.where(valid, values[clipped], np.nan)


def covered(series, start, end, max_gap):
    time, _ = series
    inside = time[(time >= start) & (time <= end)]
    if (len(inside) < 2 or inside[0] - start > max_gap or end - inside[-1] > max_gap
            or np.max(np.diff(inside)) > max_gap):
        raise Excluded("incomplete input history")


def patient_for(case):
    if case["sex"] not in ("M", "F"):
        raise Excluded("missing patient sex")
    patient = Patient(age=float(case["age"]), weight=float(case["weight"]),
                      height=float(case["height"]), sex="male" if case["sex"] == "M" else "female")
    # Evaluation eligibility stays fixed when the simulator's input domain grows.
    if patient.weight > 100.0 or patient.bmi > 32.0:
        raise Excluded("outside prespecified evaluation domain")
    return patient


def time_grid(start, end, dt):
    return np.r_[np.arange(start, end, dt), end]


def delivered_volume(rate, grid):
    """Integrate recorded mL/h exactly between packets, then sample cumulative mL."""
    time, values = rate
    cumulative = np.r_[0.0, np.cumsum(values[:-1] * np.diff(time) / 3600.0)]
    if grid[0] < time[0] or grid[-1] > time[-1]:
        raise Excluded("delivery replay outside recorded interval")
    return np.interp(grid, time, cumulative)


def replay_propofol(patient, grid, volume):
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


def replay_sevo(patient, grid, inspired, tidal_volume, rr, cardiac_output):
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
        out[phase] = dict(n=int(mask.sum()), minutes=float(mask.sum() * SAMPLE_SECONDS / 60.0),
                          bias=float(np.mean(difference)), mae=float(np.mean(np.abs(difference))),
                          rmse=float(np.sqrt(np.mean(difference ** 2))),
                          baseline_mae=float(np.mean(np.abs(baseline[mask] - observed[mask]))))
    if "all" not in out or out["all"]["minutes"] < 10.0:
        raise Excluded("less than ten minutes of usable endpoint measurements")
    return out


def evaluate_tiva(case, tracks, dt):
    series = {name: numeric_track(tracks[name]) for name in ("Orchestra/PPF20_RATE", "Orchestra/PPF20_VOL")}
    rate, volume = series["Orchestra/PPF20_RATE"], series["Orchestra/PPF20_VOL"]
    if volume[1][0] > 0.05 or volume[1][0] < 0 or rate[1][0] != 0:
        raise Excluded("propofol delivery began before recording")
    if np.any((rate[1] < 0) | (rate[1] > 1500)) or volume[1][-1] < 5:
        raise Excluded("invalid or insufficient propofol delivery")
    if np.max(np.maximum.accumulate(volume[1]) - volume[1]) > 0.25:
        raise Excluded("propofol volume reset or inconsistent counter")
    active = np.flatnonzero(rate[1] > 0.1)
    if not len(active):
        raise Excluded("no recorded propofol infusion")
    onset, stop = float(rate[0][active[0]]), float(rate[0][active[-1] + 1]) if active[-1] + 1 < len(rate[0]) else float(rate[0][-1])
    if onset - rate[0][0] < 30:
        raise Excluded("less than thirty seconds of zero delivery before induction")
    for name in TIVA_TRACKS[2:]:
        series[name] = numeric_track(tracks[name])
    remi_name = next(name for name in REMI_RATES if name in tracks)
    series[remi_name] = numeric_track(tracks[remi_name])
    if np.max(series[remi_name][1]) <= 0:
        raise Excluded("no recorded remifentanil delivery")
    if np.max(series["Primus/MAC"][1]) > 0.05 or np.max(series["Primus/FIN2O"][1]) > 1:
        raise Excluded("recorded volatile or nitrous oxide exposure")
    start = max(rate[0][0], volume[0][0])
    if onset - start < 30:
        raise Excluded("common delivery history starts too late for induction")
    end = min(s[0][-1] for s in series.values())
    if end - onset < 900:
        raise Excluded("short replay interval")
    for name in ("Orchestra/PPF20_RATE", "Orchestra/PPF20_VOL", remi_name):
        covered(series[name], start, end, 5.0)
    for name in ("Primus/MAC", "Primus/FIN2O"):
        covered(series[name], onset, end, 30.0)
        if series[name][0][0] > onset:
            raise Excluded("gas monitoring began after propofol induction")
    grid = time_grid(start, end, dt)
    delivered = delivered_volume(rate, grid)
    measured = np.interp(grid[-1], *volume) - np.interp(grid[0], *volume)
    error = float((delivered[-1] - delivered[0]) - measured)
    if abs(error) > max(0.5, 0.05 * measured):
        raise Excluded("infusion rate and volume disagree by more than five percent or 0.5 mL")
    prediction = replay_propofol(patient_for(case), grid, delivered)
    sample = np.arange(onset, end, SAMPLE_SECONDS)
    observed = held(series["BIS/BIS"], sample, 2.0)
    sqi = held(series["BIS/SQI"], sample, 2.0)
    predicted = np.interp(sample, grid, prediction)
    valid = np.isfinite(observed) & (observed >= 0) & (observed <= 100) & (sqi >= SQI_MIN) & (sqi <= 100)
    intervals = dict(induction=(onset, min(onset + 600, float(case["opstart"]))),
                     maintenance=(max(onset + 600, float(case["opstart"])), min(stop, float(case["opend"]))),
                     emergence=(max(stop, float(case["opend"])), end))
    metrics = phase_metrics(sample, observed, predicted, np.full(len(sample), 40.0), valid, intervals)
    return dict(metrics=metrics, dose_mg=float((delivered[-1] - delivered[0]) * 20.0),
                volume_disagreement_ml=error, valid_fraction=float(np.mean(valid)),
                _trace=(sample, observed, predicted, valid, onset),
                replay_start=float(start), replay_end=float(end), onset=onset, stop=stop)


def evaluate_sevo(case, tracks, dt):
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
    _inspired, vt, rr = inputs
    if np.any((vt < 0) | (vt > 1500) | (rr < 0) | (rr > 50)):
        raise Excluded("ventilation inputs outside comparison range")
    sample = np.arange(onset, end, SAMPLE_SECONDS)
    observed = held(series["Primus/EXP_SEVO"], sample, 15.0)
    co2 = held(series["Primus/ETCO2"], sample, 15.0)
    measured_vt = held(series["Primus/TV"], sample, 15.0)
    measured_rr = held(series["Primus/RR_CO2"], sample, 15.0)
    valid = (np.isfinite(observed) & (observed >= 0) & (observed <= 8)
             & (co2 >= 15) & (co2 <= 60) & (measured_vt >= 200) & (measured_rr >= 5))
    intervals = dict(washin=(onset, onset + 600), maintenance=(onset + 600, stop), washout=(stop, end))
    baseline = held(fi, sample, 15.0)
    valid &= np.isfinite(baseline)
    predictions = {co: np.interp(sample, grid, replay_sevo(patient_for(case), grid, *inputs, co)) for co in (3.0, 5.0, 7.0)}
    metrics = phase_metrics(sample, observed, predictions[5.0], baseline, valid, intervals)
    sensitivity = {str(co): phase_metrics(sample, observed, prediction, baseline, valid, intervals)
                   for co, prediction in predictions.items()}
    return dict(metrics=metrics, cardiac_output_sensitivity=sensitivity, valid_fraction=float(np.mean(valid)),
                _trace=(sample, observed, predictions[5.0], valid, onset),
                replay_start=float(start), replay_end=float(end), onset=onset, stop=stop)


def evaluate(candidate, kind, dt):
    case, tracks = candidate
    try:
        result = (evaluate_tiva if kind == "tiva" else evaluate_sevo)(case, tracks, dt)
    except Excluded as error:
        return dict(caseid=int(case["caseid"]), subjectid=int(case["subjectid"]), exclusion=str(error))
    result.update(caseid=int(case["caseid"]), subjectid=int(case["subjectid"]),
                  age=float(case["age"]), weight=float(case["weight"]), height=float(case["height"]), sex=case["sex"],
                  tracks={name: tracks[name] for name in (TIVA_TRACKS if kind == "tiva" else SEVO_TRACKS)})
    if kind == "tiva":
        result["tracks"].update({name: tracks[name] for name in REMI_RATES if name in tracks})
    return result


def cohort(kind, requested, seed, workers, dt):
    candidates = domain_cases(TIVA_TRACKS if kind == "tiva" else SEVO_TRACKS)
    initial = len(candidates)
    if kind == "tiva":
        candidates = [(c, t) for c, t in candidates if any(n in t for n in REMI_RATES)
                      and all(c[n] and float(c[n]) == 0 for n in ("intraop_ppf", "intraop_mdz", "intraop_ftn"))
                      and not any(n in t for n in ("Orchestra/DEX2_RATE", "Orchestra/DEX4_RATE"))]
    random.Random(seed).shuffle(candidates)
    selected, exclusions, subjects = [], [], set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(candidates), workers):
            if len(selected) >= requested:
                break
            batch = candidates[offset:offset + workers]
            results = list(pool.map(lambda candidate: evaluate(candidate, kind, dt), batch))
            for result in results:
                if len(selected) >= requested:
                    break
                if "exclusion" in result:
                    exclusions.append(result)
                elif result["subjectid"] in subjects:
                    exclusions.append(dict(caseid=result["caseid"], subjectid=result["subjectid"], exclusion="patient already included"))
                else:
                    subjects.add(result["subjectid"])
                    selected.append(result)
            print(f"{kind}: retained {len(selected)}/{requested}; screened {len(selected) + len(exclusions)}", flush=True)
    return dict(requested=requested, track_eligible=initial, metadata_eligible=len(candidates), rows=selected, exclusions=exclusions)


def mean_ci(values, rng):
    values = np.asarray(values)
    means = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return f"{np.mean(values):.2f} ({low:.2f} to {high:.2f})"


def report(data, settings, seed):
    lines = ["# VitalDB anesthetic comparisons", "", f"Generated {settings['generated']}.",
             "", "## Methods", "",
             "These comparisons test the anesthetic models in isolation, with normal organ function and unchanged blood volume.",
             "The models use AnaSim's standard parameters, which were not tuned to these recordings. The replay fits no",
             "individual calibrations or time shifts, and predicted BIS does not start from the observed BIS.", "",
             "Patients meet the same rules as the ventilation cohort: general anesthesia, oral single-lumen tube, open surgery,",
             "supine position, age 18 to 70 years, weight 50 to 100 kg, height 150 to 200 cm, and BMI 18 to 32 kg/m².",
             f"Each cohort includes at most one case per patient, and seed {seed} fixes the order of candidates.",
             "Eligibility uses only recorded inputs and signal quality.", "",
             f"The replay step is {settings['dt']:g} s, and endpoints are sampled every {SAMPLE_SECONDS:g} s. Each reported phase",
             "needs at least one usable minute, and each case needs ten usable minutes in total. The number of samples in a",
             "phase depends on which measurements are available. The all-phase score includes usable intervals between the",
             "named phases.", "",
             "Errors are predicted minus measured. Tables show the mean of the per-patient metrics, with 95% intervals from",
             "2,000 patient bootstrap resamples, so long cases have the same weight as short cases. The comparator is scored",
             "on the same samples.",
             "", "## Propofol and BIS", "",
             "Propofol delivery comes from the recorded pump rate (PPF20_RATE), which is held between recorded packets and",
             "integrated in mL. A concentration of 20 mg/mL converts it to mg/s for the model. Each case must meet these rules:", "",
             "- The cumulative pump volume must start at zero (at most 0.05 mL), and both rate and volume must stay at zero for",
             "  at least 30 seconds before induction.",
             "- Rates must be between 0 and 1,500 mL/h, at least 5 mL must be delivered, and the replay must last at least 15",
             "  minutes after induction.",
             "- The integrated rate must agree with the recorded volume within 5% or 0.5 mL, whichever is larger.",
             "- Counter reversals over 0.25 mL and delivery gaps over 5 s are excluded. Smaller counter artifacts do not affect",
             "  the replay, which uses the rate.",
             "- Recorded supplemental boluses of propofol, midazolam, and fentanyl must total zero, and dexmedetomidine pump",
             "  tracks are excluded.",
             "- Remifentanil delivery must be recorded and nonzero, and the propofol model uses its opioid-regimen covariate.",
             "- MAC must stay at or below 0.05, and inspired N₂O at or below 1%. Gas recording must start before propofol",
             "  delivery and have no gaps over 30 s during the replay.",
             f"- BIS samples need a signal quality index (SQI) of at least {SQI_MIN:g}, with no more than 2 s since the latest BIS",
             "  and SQI packets.", "",
             "The model includes the Eleveld effect site, the published BIS processing delay, and the display smoothing, but",
             "no display noise. Remifentanil analgesia and stimulus responses are not evaluated, and the replay does not",
             "include hemorrhage or surgical arousal for individual patients.", "",
             "Induction ends at incision or ten minutes after first delivery, whichever is earlier, and maintenance begins",
             "once both have passed. Emergence begins once surgery has ended and the last recorded nonzero infusion has",
             "stopped. These phases come from the recording and do not mark observed loss or return of consciousness. The",
             "comparator is a constant BIS of 40. Concentrations estimated by the pump (CP and CE) are never treated as",
             "measured values.", ""]
    rng = np.random.default_rng(seed)
    for kind, title, phases in (("tiva", None, ("all", "induction", "maintenance", "emergence")),
                                 ("sevo", "Sevoflurane uptake", ("all", "washin", "maintenance", "washout"))):
        group = data[kind]
        if title:
            lines += [f"## {title}", "",
                      "Recorded inspired sevoflurane is the input to the uptake model (`VolatilePK`), which starts with no agent in",
                      "any compartment. Recorded tidal volume and the respiratory rate from the CO₂ trace give alveolar ventilation,",
                      "using AnaSim's dead space of 2.2 mL/kg PBW. Each case must meet these rules:", "",
                      "- Inspired and expired concentrations must start at or below 0.1%, with 30 s before exposure reaches 0.5%.",
                      "- Inspired concentration must reach at least 1%, and the first and last readings at or above 0.5% must be at",
                      "  least 15 minutes apart.",
                      "- Cases are excluded for input gaps over 30 s, inspired concentrations outside 0 to 8%, VT outside 0 to 1,500 mL,",
                      "  RR outside 0 to 50/min, or N₂O exposure over 1%.",
                      "- Endpoint samples need VT of at least 200 mL, RR of at least 5/min, EtCO₂ of 15 to 60 mmHg, and expired gas and",
                      "  ventilation packets within 15 s.", "",
                      "No agent is carried across an unknown input gap. Cardiac output is assumed to be 5 L/min, with sensitivity runs",
                      "at 3 and 7 L/min. FRC comes from the patient's body size and stays fixed, shunt is zero, and temperature is",
                      "37 °C. These values are neither measured nor fitted.", "",
                      "Wash-in is the first ten minutes after inspired agent reaches 0.5%, and washout follows the final reading at or",
                      "above 0.5%. The comparator sets the end-tidal concentration equal to the recorded inspired concentration.",
                      "Concentration errors are in percentage points. The comparison takes measured inspired gas and ventilation as",
                      "inputs, so it does not test vaporizer or circuit delivery, lung recruitment, cardiovascular effects, or the",
                      "effects of sevoflurane on BIS.", ""]
        lines += [f"Cases with the required tracks: {group['track_eligible']}. After the metadata rules: {group['metadata_eligible']}.",
                  f"Screened: {len(group['rows']) + len(group['exclusions'])}. Retained: {len(group['rows'])} of {group['requested']} requested patients.", "",
                  "| Phase | Patients | Usable minutes | Bias (95% CI) | MAE (95% CI) | RMSE (95% CI) | Comparator MAE (95% CI) |",
                  "|---|---:|---:|---|---|---|---|"]
        for phase in phases:
            metrics = [r["metrics"][phase] for r in group["rows"] if phase in r["metrics"]]
            if not metrics:
                continue
            lines.append(f"| {phase} | {len(metrics)} | {sum(m['minutes'] for m in metrics):.1f} | "
                         + " | ".join(mean_ci([m[key] for m in metrics], rng) for key in ("bias", "mae", "rmse", "baseline_mae")) + " |")
        if group["rows"]:
            metrics = [r["metrics"]["all"] for r in group["rows"]]
            differences = [m["mae"] - m["baseline_mae"] for m in metrics]
            coverage = np.percentile([100.0 * r["valid_fraction"] for r in group["rows"]], [25, 50, 75])
            lines += ["", f"Paired all-phase MAE difference, AnaSim minus comparator: {mean_ci(differences, rng)}.",
                      f"AnaSim has lower MAE in {sum(d < 0 for d in differences)} of {len(differences)} patients.",
                      f"Usable endpoint coverage: median {coverage[1]:.1f}% (IQR {coverage[0]:.1f}% to {coverage[2]:.1f}%) of the replay sampling interval."]
        if kind == "sevo" and group["rows"]:
            lines += ["", "| Assumed CO (L/min) | All-phase MAE (95% CI) |", "|---:|---|"]
            for co in ("3.0", "5.0", "7.0"):
                lines.append(f"| {co} | {mean_ci([r['cardiac_output_sensitivity'][co]['all']['mae'] for r in group['rows']], rng)} |")
        lines += ["", "Exclusions among screened cases:", ""]
        counts = Counter(r["exclusion"] for r in group["exclusions"])
        lines.extend(f"- {reason}: {count}" for reason, count in sorted(counts.items()))
        if len(group["rows"]) < group["requested"]:
            lines += ["", "The requested sample was not reached, so the table reports all eligible patients found."]
        lines += [""]
    lines += ["## Interpretation", "",
              "These comparisons test individual anesthetic models against clinical recordings. They do not establish the",
              "accuracy of the whole simulator. BIS reflects EEG and monitor behavior and is not a direct measure of",
              "consciousness. Clinicians chose doses in response to the patient's state, and unrecorded interventions and",
              "delays in delivery or monitoring can contribute to errors. A population model is also not expected to reproduce",
              "each patient's course. Sevoflurane results depend on unmeasured cardiac output, FRC, and ventilation history.",
              "Conclusions about induction, emergence, wash-in, and washout should use the results for that phase, especially",
              "where few minutes of emergence or washout are available.", "",
              "Per-patient errors, track identifiers, and exclusions are in `vitaldb_anesthetics.json` and",
              "`vitaldb_anesthetics_cases.csv`. Overlay plots show the patients at the minimum, quartiles, and maximum of",
              "all-phase MAE, with their case IDs.", "",
              f"Source: [Lee et al., Scientific Data, 2022]({SOURCE}); [VitalDB documentation](https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md).",
              "VitalDB data are used under CC BY-NC-SA 4.0 and its data use agreement. Downloaded tracks and overlay plots are",
              "kept locally and are not in the repository.", ""]
    return "\n".join(lines)


def overlays(group, kind, out):
    if not group["rows"]:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(group["rows"], key=lambda row: row["metrics"]["all"]["mae"])
    indices = np.unique(np.rint(np.linspace(0, len(ordered) - 1, 5)).astype(int))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiva-cases", type=int, default=100)
    parser.add_argument("--sevo-cases", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", action="store_true", help="Write representative overlays (requires Matplotlib).")
    args = parser.parse_args()
    if min(args.tiva_cases, args.sevo_cases) < 0 or max(args.tiva_cases, args.sevo_cases) == 0 or args.workers < 1 or not 0 < args.dt <= 1:
        parser.error("request at least one cohort, positive workers, and a replay step in (0, 1] seconds")
    import vitaldb_ventilation

    vitaldb_ventilation.CACHE = args.cache
    args.cache.mkdir(parents=True, exist_ok=True)
    data = {kind: cohort(kind, count, args.seed + offset, args.workers, args.dt)
            for offset, (kind, count) in enumerate((("tiva", args.tiva_cases), ("sevo", args.sevo_cases)))}
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
