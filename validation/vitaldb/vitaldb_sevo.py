"""Replay bolus propofol induction with sevoflurane maintenance; compare MAP and HR with VitalDB.

Induction as taught in the United States: propofol bolus, fentanyl, and
sevoflurane, without target-controlled infusion. Only deidentified case
summaries are saved in Git.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from vitaldb_anesthetics import (
    REMI_RATES,
    Excluded,
    covered,
    delivered_volume,
    held,
    numeric_track,
    time_grid,
)
from vitaldb_replay import (
    BOOTSTRAP,
    CUFF_WINDOW_S,
    REMI_UG_PER_ML,
    SAMPLE_SECONDS,
    SENSITIVITY,
    STOP,
    VASOACTIVE,
    amount,
    course,
    course_table,
    cuff_model,
    estimate,
    mean_ci,
    pre_induction,
    readings,
    report_footer,
    report_header,
    rule_stats,
    treated,
)
from vitaldb_ventilation import ROOT, domain_cases

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology import hemodynamics

REQUIRED = ["Solar8000/NIBP_MBP", "Solar8000/NIBP_SBP", "Solar8000/HR", "BIS/BIS",
            "Primus/INSP_SEVO", "Primus/FIN2O", "Primus/PPLAT_MBAR"]
HR_WINDOWS = ((0.0, 300.0), (300.0, 600.0), (600.0, np.inf))  # Seconds after propofol, to incision
# In TIVA cases with exact pump timing, model BIS falls below 60 a median 34 s after recorded BIS
# (22 s with the fastest loading), so propofol starts this much after the model latency implies.
BIS_LAG_S = 30.0
# The ventilator tidal volume setting first appears a median 200 s after propofol.
INTUBATION_S = 150.0


def eligible(case, tracks):
    return (
        0.0 < amount(case, "intraop_ppf") <= 3.0 * float(case["weight"])
        and amount(case, "intraop_ftn") <= 3.0 * float(case["weight"])
        and "Orchestra/PPF20_RATE" not in tracks
        and not any(f"Orchestra/{drug}_RATE" in tracks for drug in (*VASOACTIVE, "DEX2", "DEX4"))
        and all(amount(case, key) == 0 for key in ("intraop_mdz", "intraop_epi", "intraop_rbc", "intraop_ffp"))
        and amount(case, "intraop_ebl") <= 500
        and case["asa"] in ("1", "2") and case["preop_htn"] == "0"
    )


def bis_drop_time(tracks):
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


def bis_latency(case, patient):
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


def replay(case, patient, grid, onset, remi_ug, inspired, events, end_tidal=None):
    """Full engine with bolus fentanyl and propofol, recorded sevoflurane and remifentanil.

    With end_tidal, recorded end-tidal sevoflurane (%) sets alveolar sevoflurane where recorded.
    """
    duration_h = max(1.0, (float(case["aneend"]) - float(case["anestart"])) / 3600.0)
    engine = SimulationEngine(patient, SimulationConfig(
        mode="awake", concomitant_opioids=True, rng_seed=1,
        maintenance_fluid_ml_hr=amount(case, "intraop_crystalloid") / duration_h,
    ))
    engine.start()
    engine.set_airway_mode("Mask")
    engine.set_fgf(6.0, 0.0)
    pk_step = engine.pk_sevo.step
    fi = {"now": 0.0, "end_tidal": np.nan}

    def sevo_step(dt, _fi, va, co, **kwargs):
        state = pk_step(dt, fi["now"], va, co, **kwargs)
        if np.isfinite(fi["end_tidal"]):
            engine.pk_sevo.state.p_alv = fi["end_tidal"] / 100.0
        return state
    engine.pk_sevo.step = sevo_step
    colloid_ml_min = amount(case, "intraop_colloid") / duration_h / 60.0
    vt = min(0.6, 0.008 * patient.predicted_body_weight())
    propofol_rate = amount(case, "intraop_ppf") / 20.0  # mg/s over 20 s
    pending = sorted(events, key=lambda event: event[0])
    ventilated, next_colloid = False, onset
    out = np.empty((len(grid), 3))
    out[0] = engine.state.map, engine.state.hr, engine.state.et_sevo
    for i, dt in enumerate(np.diff(grid), start=1):
        now = grid[i - 1]
        while pending and pending[0][0] <= now:
            pending.pop(0)[1](engine)
        if not ventilated and now >= onset + 90.0:
            engine.set_airway_mode("ETT")
            engine.set_vent_settings(rr=12, vt=vt, peep=5.0, ie="1:2", mode="VCV")
            engine.set_vent_power(True)
            engine.set_fgf(1.0, 1.0)
            ventilated = True
        if colloid_ml_min and now >= next_colloid:
            engine.give_albumin(colloid_ml_min * 5.0)
            next_colloid += 300.0
        fi["now"] = float(inspired[i - 1]) / 100.0
        if end_tidal is not None:
            fi["end_tidal"] = float(end_tidal[i - 1])
        engine.infusion_rates["propofol"] = propofol_rate if onset <= now < onset + 20.0 else 0.0
        engine.infusion_rates["remi"] = (remi_ug[i] - remi_ug[i - 1]) / dt
        engine.step(float(dt))
        out[i] = engine.state.map, engine.state.hr, engine.state.et_sevo
    return out


def evaluate(candidate, dt):
    case, tracks = candidate
    row = dict(caseid=int(case["caseid"]), treated=treated(case))
    try:
        drop = bis_drop_time(tracks)
        if case["sex"] not in ("M", "F"):
            raise Excluded("missing patient sex")
        sizing = Patient(age=float(case["age"]), weight=float(case["weight"]), height=float(case["height"]),
                         sex="male" if case["sex"] == "M" else "female")
        onset = drop - bis_latency(case, sizing) + BIS_LAG_S
        incision = float(case["opstart"])
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
        end = min(float(case["opend"]), float(sevo_t[-1]))
        start = onset - 180.0
        covered(gases["Primus/INSP_SEVO"], onset, end, 30.0)
        remi_name = next((name for name in REMI_RATES if name in tracks), None)
        grid = time_grid(start, max(incision, end), dt)
        if remi_name:
            remi = numeric_track(tracks[remi_name])
            if np.any((remi[1] < 0) | (remi[1] > 1500)):
                raise Excluded("invalid remifentanil delivery")
            remi_ug = delivered_volume(remi, grid) * REMI_UG_PER_ML[remi_name]
        else:
            remi_ug = np.zeros(len(grid))
        inspired = np.nan_to_num(held(gases["Primus/INSP_SEVO"], grid, 30.0))
        # Onset is uncertain by about this lag, so the baseline reading must post this much earlier.
        baseline_map, baseline_hr = pre_induction(tracks, onset - BIS_LAG_S)
        patient = Patient(age=float(case["age"]), weight=float(case["weight"]), height=float(case["height"]),
                          sex="male" if case["sex"] == "M" else "female",
                          baseline_map=baseline_map, baseline_hr=baseline_hr)
        times, cuff_map, cuff_sbp, cuff_hr = readings(tracks, onset, incision)
        if not len(times):
            raise Excluded("no cuff reading between induction and incision")
        ebl = amount(case, "intraop_ebl")
        bleed = ebl / max(1.0, (float(case["opend"]) - incision) / 60.0)
        events = [(incision, lambda e: e.start_disturbance("stim_sustained_surgery")),
                  (float(case["opend"]), lambda e: e.stop_disturbance())]
        if amount(case, "intraop_ftn"):
            events.append((onset - 120.0, lambda e: e.give_drug_bolus("fentanyl", amount(case, "intraop_ftn"))))
        events.append((onset + INTUBATION_S, lambda e: e.start_disturbance("stim_intubation_pulse")))
        if ebl:
            events += [(incision, lambda e: e.start_hemorrhage(bleed)), (float(case["opend"]), lambda e: e.stop_hemorrhage())]
        predicted = replay(case, patient, grid, onset, remi_ug, inspired, events)
    except Excluded as error:
        return {**row, "exclusion": str(error)}

    model_map = cuff_model(times, grid, predicted[:, 0])
    errors, seconds = model_map - cuff_map, times - onset
    summary = {**row, "age": float(case["age"]), "baseline_map": baseline_map, "readings": len(times),
               "fentanyl": amount(case, "intraop_ftn") > 0, "remifentanil": remi_name is not None,
               "median_error": float(np.median(errors))}
    for rule, limits in (("primary", STOP), ("sensitivity", SENSITIVITY)):
        summary[rule] = rule_stats(errors, seconds, cuff_map, cuff_sbp, cuff_hr, limits)
    summary["posting"] = rule_stats(np.interp(times, grid, predicted[:, 0]) - cuff_map, seconds,
                                    cuff_map, cuff_sbp, cuff_hr, STOP)
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
            summary["end_tidal"][name] = (float(np.median(recorded[valid])),
                                          float(np.median(np.interp(sample[valid], grid, predicted[:, 2]))))
    hr_track = numeric_track(tracks["Solar8000/HR"])
    summary["hr_errors"] = []
    for low, high in HR_WINDOWS:
        sample = np.arange(onset + low, min(onset + high, incision), SAMPLE_SECONDS)
        recorded = held(hr_track, sample, 30.0) if len(sample) else np.array([])
        valid = np.isfinite(recorded) & (recorded > 30) & (recorded < 160)
        summary["hr_errors"].append(float(np.median(np.interp(sample[valid], grid, predicted[:, 1]) - recorded[valid]))
                                    if valid.sum() >= 12 else None)
    maintenance = np.arange(incision + 900.0, end, SAMPLE_SECONDS)
    pressure = "Solar8000/ART_MBP" if "Solar8000/ART_MBP" in tracks else "Solar8000/NIBP_MBP"
    summary["maintenance"] = None
    if len(maintenance) * SAMPLE_SECONDS >= 600:
        recorded = {"map": held(numeric_track(tracks[pressure]), maintenance, 30.0), "hr": held(hr_track, maintenance, 30.0)}
        valid = {k: np.isfinite(x) & (x > 30) & (x < 160) for k, x in recorded.items()}
        if min(v.sum() for v in valid.values()) * SAMPLE_SECONDS >= 300:
            summary["maintenance"] = {
                k: float(np.median(np.interp(maintenance[valid[k]], grid, predicted[:, column]) - recorded[k][valid[k]]))
                for column, k in enumerate(("map", "hr"))}
    return summary


def apply_override(override):
    if override:
        original = hemodynamics.HemodynamicModel.__init__

        def init(self, patient, config=None):
            original(self, patient, replace(config or hemodynamics.HemodynamicConfig(), **override))
        hemodynamics.HemodynamicModel.__init__ = init


def report(data, settings):
    rng = np.random.default_rng(settings["seed"])
    rows = data["rows"]
    untreated = [r for r in rows if not r["treated"]]
    groups = (("All patients", rows), ("Untreated", untreated), ("Pressor-treated", [r for r in rows if r["treated"]]))
    override = [f"Model override: `{json.dumps(settings['override'])}`.", ""] if settings["override"] else []
    lines = report_header("VitalDB bolus induction with sevoflurane", settings) + override + [
        "## Methods", "",
        f"Propofol runs over 20 s. It starts {BIS_LAG_S:.0f} s after the time at which model BIS would first fall below 60",
        "when recorded BIS does. Fentanyl is given 2 minutes earlier. Recorded inspired sevoflurane and remifentanil pumps",
        f"are replayed, with intubation stimulation {INTUBATION_S / 60:.1f} minutes after propofol and surgery from incision.", "",
        f"From propofol to incision, each cuff reading is compared with model MAP averaged over the {CUFF_WINDOW_S:.0f} s",
        "before the reading appears. A patient's readings stop at the first one with MAP below 65 mmHg, SBP below 90, or",
        "HR below 45. A likelihood fit with a patient offset and reading noise gives the mean error (model minus recorded),",
        f"with 95% intervals from {BOOTSTRAP} patient bootstrap resamples.", "",
        f"Eligible candidates: {data['eligible']}. Retained: {len(rows)} ({sum(r['fentanyl'] for r in rows)} with fentanyl, "
        f"{sum(r['remifentanil'] for r in rows)} with remifentanil).", "",
        "| Patients | n | Readings | Mean error, mmHg (95% CI) | Patient SD | Reading SD |",
        "|---|---:|---:|---|---:|---:|"]
    for label, group in groups:
        if len(group) < 3:
            continue
        [(mean, low, high)], between, within, used = estimate(group, "primary", False, rng)
        lines.append(f"| {label} | {len(group)} | {used} | {mean:.1f} ({low:.1f} to {high:.1f}) | {between:.1f} | "
                     f"{within:.1f} |")
    by_period, _, _, _ = estimate(rows, "primary", True, rng)
    lines += ["", "By time after propofol, all patients: "
              + "; ".join(f"{name} {m:.1f} ({lo:.1f} to {hi:.1f})"
                          for name, (m, lo, hi) in zip(("before 10 min", "from 10 min"), by_period, strict=True))
              + " mmHg."]
    lines += course_table(rows)
    for name in ("before incision", "surgery"):
        pairs = np.array([r["end_tidal"][name] for r in rows if name in r["end_tidal"]])
        if len(pairs):
            lines += ["", f"End-tidal sevoflurane {name}, median: recorded {np.median(pairs[:, 0]):.2f}%, model "
                      f"{np.median(pairs[:, 1]):.2f}% (n={len(pairs)})."]
    hr_lines = []
    for index, name in enumerate(("0-5 min", "5-10 min", "10 min to incision")):
        values = [r["hr_errors"][index] for r in untreated if r["hr_errors"][index] is not None]
        if values:
            hr_lines.append(f"{name} {mean_ci(values, rng)}")
    if hr_lines:
        lines += ["", "ECG HR error (model minus recorded, per-patient medians, untreated): " + "; ".join(hr_lines) + " bpm."]
    maintenance = [r["maintenance"] for r in untreated if r["maintenance"]]
    if maintenance:
        lines += ["", "Surgery from 15 min after incision, untreated: MAP "
                  f"{mean_ci([m['map'] for m in maintenance], rng)} mmHg; HR {mean_ci([m['hr'] for m in maintenance], rng)} bpm."]
    lines += report_footer(data["exclusions"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--override", default="{}", help="JSON HemodynamicConfig fields, for model comparisons")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    override = json.loads(args.override)
    pool = sorted(((case, trk) for case, trk in domain_cases(REQUIRED) if eligible(case, trk)),
                  key=lambda candidate: int(candidate[0]["caseid"]))
    with ProcessPoolExecutor(args.workers, initializer=apply_override, initargs=(override,)) as executor:
        results = list(executor.map(evaluate, pool, [args.dt] * len(pool), chunksize=4))
    data = dict(eligible=len(pool), rows=[r for r in results if "exclusion" not in r],
                exclusions=[r for r in results if "exclusion" in r])
    settings = dict(generated=datetime.now(timezone.utc).date().isoformat(), dt=args.dt, seed=args.seed, override=override)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "vitaldb_sevo"
    stem.with_suffix(".json").write_text(json.dumps(dict(settings=settings, **data), indent=2, allow_nan=False) + "\n")
    stem.with_suffix(".md").write_text(report(data, settings))
    print(f"Report: {stem.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
