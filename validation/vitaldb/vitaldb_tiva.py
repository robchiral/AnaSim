"""Replay propofol-remifentanil TIVA through the AnaSim engine; compare MAP and HR from induction through surgery.

Each patient starts from their own pre-induction cuff MAP and HR. Patients
given ephedrine or phenylephrine are included; their induction readings stop at
the first reading that could prompt treatment. Only deidentified case summaries
are saved in Git.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from vitaldb_anesthetics import (
    REMI_RATES,
    Excluded,
    delivered_volume,
    numeric_track,
    time_grid,
)
from vitaldb_replay import (
    BOOTSTRAP,
    CUFF_WINDOW_S,
    REMI_UG_PER_ML,
    SENSITIVITY,
    STOP,
    VASOACTIVE,
    amount,
    course,
    course_table,
    cuff_model,
    delivery,
    estimate,
    mean_ci,
    pre_induction,
    provenance,
    readings,
    report_footer,
    report_header,
    rule_stats,
    treated,
    window_medians,
)
from vitaldb_ventilation import ROOT, domain_cases

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient

REQUIRED = ["Orchestra/PPF20_RATE", "Orchestra/PPF20_VOL", "Primus/MAC", "Primus/FIN2O",
            "Solar8000/NIBP_MBP", "Solar8000/NIBP_SBP", "Solar8000/HR"]
INTUBATION_S = 240.0


def eligible(case, tracks):
    """Adults in the simulator's domain under general-surgery TIVA without vasoactive infusions or transfusion."""
    try:
        age, weight, height = float(case["age"]), float(case["weight"]), float(case["height"])
    except ValueError:
        return False
    return (
        18 <= age <= 70 and 50 <= weight <= 100 and 150 <= height <= 200 and 18 <= weight / (height / 100) ** 2 <= 32
        and case["department"] == "General surgery" and not case["dltubesize"]
        and any(name in tracks for name in REMI_RATES)
        and not any(f"Orchestra/{drug}_RATE" in tracks for drug in (*VASOACTIVE, "DEX2", "DEX4"))
        and all(amount(case, key) == 0 for key in ("intraop_ppf", "intraop_mdz", "intraop_ftn", "intraop_epi",
                                                    "intraop_rbc", "intraop_ffp"))
        and amount(case, "intraop_ebl") <= 500 and case["asa"] in ("1", "2") and case["preop_htn"] == "0"
    )


def procedure(case):
    if case["position"] != "Supine":
        return "Other positions"
    return "Open, supine" if case["approach"] == "Open" else "Laparoscopic or robotic, supine"


def replay(case, patient, grid, propofol_mg, remi_ug, onset, events):
    """Drive the full engine with recorded delivery, fluids, blood loss, and surgery."""
    duration_h = max(1.0, (float(case["aneend"]) - float(case["anestart"])) / 3600.0)
    engine = SimulationEngine(patient, SimulationConfig(
        mode="awake", concomitant_opioids=True, rng_seed=1,
        maintenance_fluid_ml_hr=amount(case, "intraop_crystalloid") / duration_h,
    ))
    engine.start()
    engine.set_airway_mode("Mask")
    engine.set_fgf(6.0, 0.0)
    colloid_ml_min = amount(case, "intraop_colloid") / duration_h / 60.0
    vt = min(0.6, 0.008 * patient.predicted_body_weight())
    pending = sorted(events, key=lambda event: event[0])
    intubated, next_colloid = False, onset
    out = np.empty((len(grid), 2))
    out[0] = engine.state.map, engine.state.hr
    for i, dt in enumerate(np.diff(grid), start=1):
        now = grid[i - 1]
        while pending and pending[0][0] <= now:
            pending.pop(0)[1](engine)
        if not intubated and now >= onset + 90.0:
            engine.set_airway_mode("ETT")
            engine.set_vent_settings(rr=12, vt=vt, peep=5.0, ie="1:2", mode="VCV")
            engine.set_vent_power(True)
            engine.set_fgf(1.0, 1.0)
            intubated = True
        if colloid_ml_min and now >= next_colloid:
            engine.give_albumin(colloid_ml_min * 5.0)
            next_colloid += 300.0
        engine.infusion_rates["propofol"] = (propofol_mg[i] - propofol_mg[i - 1]) / dt
        engine.infusion_rates["remi"] = (remi_ug[i] - remi_ug[i - 1]) / dt
        engine.step(float(dt))
        out[i] = engine.state.map, engine.state.hr
    return out


def peak(series, samples=15):
    """Highest rolling median over `samples` consecutive values (30 s of 2-s HR)."""
    return float(np.median(np.lib.stride_tricks.sliding_window_view(series, samples), axis=1).max())


def evaluate(candidate, dt):
    case, tracks = candidate
    row = dict(caseid=int(case["caseid"]), treated=treated(case), procedure=procedure(case))
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
        if case["sex"] not in ("M", "F"):
            raise Excluded("missing patient sex")
        patient = Patient(age=float(case["age"]), weight=float(case["weight"]), height=float(case["height"]),
                          sex="male" if case["sex"] == "M" else "female",
                          baseline_map=baseline_map, baseline_hr=baseline_hr)
        maintenance = (opstart + 900.0, min(opend, stop, end))
        has_maintenance = incision == opstart and maintenance[1] - maintenance[0] >= 600
        grid = time_grid(max(start, onset - 60.0), maintenance[1] if has_maintenance else incision, dt)
        propofol_mg = delivered_volume(rate, grid) * 20.0
        measured = np.interp(grid[-1], *volume) - np.interp(grid[0], *volume)
        if abs(propofol_mg[-1] - propofol_mg[0] - 20.0 * measured) > 20.0 * max(0.5, 0.05 * measured):
            raise Excluded("infusion rate and volume disagree by more than five percent or 0.5 mL")
        remi_ug = delivered_volume(remi, grid) * REMI_UG_PER_ML[remi_name]
        ebl = amount(case, "intraop_ebl")
        bleed_rate = ebl / max(1.0, (opend - opstart) / 60.0)
        events = [(onset + INTUBATION_S, lambda e: e.start_disturbance("stim_intubation_pulse")),
                  (opstart, lambda e: e.start_disturbance("stim_sustained_surgery")),
                  (opend, lambda e: e.stop_disturbance())]
        if ebl:
            events += [(opstart, lambda e: e.start_hemorrhage(bleed_rate)), (opend, lambda e: e.stop_hemorrhage())]
        predicted = replay(case, patient, grid, propofol_mg, remi_ug, onset, events)
    except Excluded as error:
        return {**row, "exclusion": str(error)}

    model_map = cuff_model(times, grid, predicted[:, 0])
    errors, seconds = model_map - cuff_map, times - onset
    summary = {**row, "age": float(case["age"]), "baseline_map": baseline_map, "readings": len(times),
               "median_error": float(np.median(errors))}
    for rule, limits in (("primary", STOP), ("sensitivity", SENSITIVITY)):
        summary[rule] = rule_stats(errors, seconds, cuff_map, cuff_sbp, cuff_hr, limits)
    summary["posting"] = rule_stats(np.interp(times, grid, predicted[:, 0]) - cuff_map, seconds,
                                    cuff_map, cuff_sbp, cuff_hr, STOP)
    summary["course"] = course(seconds, cuff_map, model_map, baseline_map, incision - onset)
    hr_time, hr_value = numeric_track(tracks["Solar8000/HR"])
    window = (hr_time >= onset + 180.0) & (hr_time < min(onset + 600.0, incision)) & (hr_value >= 40) & (hr_value <= 160)
    if window.sum() >= 30:
        model_hr = np.interp(hr_time[window], grid, predicted[:, 1])
        summary["hr"] = dict(median_error=float(np.median(model_hr - hr_value[window])),
                             peak_error=peak(model_hr) - peak(hr_value[window]))
    if has_maintenance:
        summary["before_incision"] = window_medians(tracks, grid, predicted, onset + 600.0, incision)
        summary["maintenance"] = window_medians(tracks, grid, predicted, *maintenance)
    return summary


def report(data, prov):
    rng = np.random.default_rng(prov["seed"])
    rows = data["rows"]
    untreated = [r for r in rows if not r["treated"]]
    groups = [("All patients", rows), ("Untreated", untreated), ("Pressor-treated", [r for r in rows if r["treated"]])]
    groups += [(name, [r for r in rows if r["procedure"] == name])
               for name in ("Open, supine", "Laparoscopic or robotic, supine", "Other positions")]
    groups += [(name, [r for r in rows if low <= r["age"] < high])
               for name, low, high in (("Age under 45", 0, 45), ("Age 45-59", 45, 60), ("Age 60 and over", 60, 200))]
    lines = report_header("VitalDB TIVA hemodynamics", prov) + [
        "## Methods", "",
        "Recorded propofol and remifentanil pump delivery drives the full engine at production parameters, from each",
        "patient's last cuff MAP within 15 minutes before propofol and median HR over the preceding 2 minutes.",
        "Controlled ventilation (VT 8 mL/kg PBW, at most 600 mL; RR 12; PEEP 5) starts 90 s after propofol and",
        "intubation stimulation at 4 minutes. Recorded crystalloid runs at a constant rate, colloid is given every 5",
        "minutes, and blood loss bleeds at a constant rate during surgery, with sustained surgical stimulation.",
        "Positioning and pneumoperitoneum are not replayed.", "",
        f"From propofol to incision, each cuff reading is compared with model MAP over the {CUFF_WINDOW_S:.0f} s before it",
        "posts. A patient's readings stop at the first with MAP below 65 mmHg, SBP below 90, or HR below 45",
        "(sensitivity: 70, 100, 45), so later treatment cannot bias the fit; a likelihood fit with a patient offset and",
        f"reading noise gives the mean error (model minus recorded) with {BOOTSTRAP}-sample patient bootstrap intervals.", "",
        f"Eligible candidates {data['eligible']}; retained {len(rows)}.", "",
        "## From propofol to incision", "",
        "| Patients | n | Readings | Mean error, mmHg (95% CI) | Patient SD | Reading SD |",
        "|---|---:|---:|---|---:|---:|"]
    for label, group in groups:
        if len(group) < 5:
            continue
        [(mean, low, high)], between, within, used = estimate(group, "primary", False, rng)
        lines.append(f"| {label} | {len(group)} | {used} | {mean:.1f} ({low:.1f} to {high:.1f}) | {between:.1f} | "
                     f"{within:.1f} |")
    [(mean, low, high)], _, _, _ = estimate(rows, "sensitivity", False, rng)
    lines += ["", f"Sensitivity stopping rule, all patients: {mean:.1f} ({low:.1f} to {high:.1f}) mmHg."]
    by_period, _, _, _ = estimate(rows, "primary", True, rng)
    lines += ["", "By time after propofol, all patients: "
              + "; ".join(f"{name} {m:.1f} ({lo:.1f} to {hi:.1f})"
                          for name, (m, lo, hi) in zip(("before 10 min", "from 10 min"), by_period, strict=True))
              + " mmHg."]
    [(mean, low, high)], _, _, _ = estimate(rows, "posting", False, rng)
    lines += ["", f"Compared with model MAP when each reading posts, all patients: {mean:.1f} ({low:.1f} to {high:.1f}) mmHg."]
    lines += course_table(rows)
    hr = [r["hr"] for r in untreated if "hr" in r]
    if hr:
        lines += ["", f"ECG HR error 3-10 minutes after propofol, untreated: per-patient median "
                  f"{mean_ci([h['median_error'] for h in hr], rng)}; highest 30-s median "
                  f"{mean_ci([h['peak_error'] for h in hr], rng)} bpm."]
    paired = [r for r in untreated if r.get("maintenance") and r.get("before_incision")]
    if paired:
        lines += ["", "## Surgery", "",
                  f"Untreated patients, from 15 minutes after incision to the end of surgery: MAP error "
                  f"{mean_ci([r['maintenance']['map']['predicted'] - r['maintenance']['map']['observed'] for r in paired], rng)}"
                  f" mmHg; HR error "
                  f"{mean_ci([r['maintenance']['hr']['predicted'] - r['maintenance']['hr']['observed'] for r in paired], rng)}"
                  " bpm. Arterial MAP is used when recorded.", "",
                  "MAP rise from the last 10 minutes before incision to surgery: recorded "
                  f"{mean_ci([r['maintenance']['map']['observed'] - r['before_incision']['map']['observed'] for r in paired], rng)}"
                  ", model "
                  f"{mean_ci([r['maintenance']['map']['predicted'] - r['before_incision']['map']['predicted'] for r in paired], rng)}"
                  " mmHg."]
    lines += report_footer(data["exclusions"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    prov = provenance(__file__, dt=args.dt, seed=args.seed)
    pool = sorted(((case, trk) for case, trk in domain_cases(REQUIRED, typical_only=False) if eligible(case, trk)),
                  key=lambda candidate: int(candidate[0]["caseid"]))
    with ProcessPoolExecutor(args.workers) as executor:
        results = list(executor.map(evaluate, pool, [args.dt] * len(pool), chunksize=4))
    data = dict(eligible=len(pool), rows=[r for r in results if "exclusion" not in r],
                exclusions=[r for r in results if "exclusion" in r])
    prov["generated_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "vitaldb_tiva"
    stem.with_suffix(".json").write_text(json.dumps(dict(provenance=prov, **data), indent=2, allow_nan=False) + "\n")
    stem.with_suffix(".md").write_text(report(data, prov))
    print(f"Report: {stem.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
