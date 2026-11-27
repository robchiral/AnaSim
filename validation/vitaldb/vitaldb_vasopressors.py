"""Validate continuous norepinephrine and phenylephrine infusions against VitalDB arterial-line tracks.

Replays eligible unique-subject domain cases (ASA 1-4, no transfusion, EBL <= 1500 mL) that
record Orchestra/NEPI_RATE (20 mcg/mL) or Orchestra/PHEN_RATE (100 mcg/mL) alongside
Solar8000/ART_MBP and HR. Compares:
1. Active-infusion steady-state MAP and HR error with recorded pressor vs a zero-pressor
   counterfactual replay of the exact same case.
2. Acute rate step-ups (infusion onset or >= 2x increase after >= 3 min at stable lower rate),
   comparing recorded vs model delta MAP and delta HR over +2 to +5 min.
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

from validation.vitaldb import vitaldb_data  # noqa: E402
from validation.vitaldb.vitaldb_data import (  # noqa: E402
    CACHE,
    DEFAULT_WORKERS,
    REMI_RATES,
    REMI_UG_PER_ML,
    Excluded,
    amount,
    covered,
    delivered_volume,
    domain_cases,
    held,
    numeric_track,
    patient_from_case,
    time_grid,
)
from validation.vitaldb.vitaldb_replay import (  # noqa: E402
    SAMPLE_SECONDS,
    delivery,
    mean_ci,
    pre_induction,
    procedure,
    report_footer,
    report_header,
    run_hemo_pool,
    run_hemodynamic_replay,
)

# VitalDB Orchestra standard syringe concentrations (Lee et al., Scientific Data, 2022).
PRESSOR_SPECS = {
    "nore": ("Orchestra/NEPI_RATE", "Orchestra/NEPI_VOL", 20.0),    # mcg/mL
    "phenyl": ("Orchestra/PHEN_RATE", "Orchestra/PHEN_VOL", 100.0),  # mcg/mL
}
OTHER_VASO = ("EPI", "DOPA", "DOBU", "VASO", "NTG", "NPS", "MRN", "DTZ", "AMD", "PGE1", "DEX2", "DEX4")


def _active_pressor(tracks: dict[str, str]) -> str | None:
    has_ne = "Orchestra/NEPI_RATE" in tracks
    has_pe = "Orchestra/PHEN_RATE" in tracks
    if has_ne == has_pe:
        return None  # Require single continuous pressor for clean attribution
    return "nore" if has_ne else "phenyl"


def eligible(case: dict[str, str], tracks: dict[str, str]) -> bool:
    if _active_pressor(tracks) is None:
        return False
    if "Solar8000/ART_MBP" not in tracks or "Solar8000/HR" not in tracks or "Solar8000/NIBP_MBP" not in tracks:
        return False
    if any(f"Orchestra/{drug}_RATE" in tracks for drug in OTHER_VASO):
        return False
    if amount(case, "intraop_rbc") > 0 or amount(case, "intraop_ffp") > 0 or amount(case, "intraop_ebl") > 1500:
        return False
    if case["asa"] not in ("1", "2", "3", "4") or case["preop_htn"] not in ("0", "1"):
        return False
    has_tiva = (
        "Orchestra/PPF20_RATE" in tracks
        and "Orchestra/PPF20_VOL" in tracks
        and "Primus/MAC" in tracks
        and "Primus/FIN2O" in tracks
        and any(name in tracks for name in REMI_RATES)
        and all(amount(case, k) == 0 for k in ("intraop_ppf", "intraop_mdz", "intraop_ftn", "intraop_epi"))
    )
    has_sevo = (
        "Primus/INSP_SEVO" in tracks
        and "Primus/FIN2O" in tracks
        and "Orchestra/PPF20_RATE" not in tracks
        and 0.0 < amount(case, "intraop_ppf") <= 3.0 * float(case["weight"])
        and amount(case, "intraop_ftn") <= 3.0 * float(case["weight"])
        and all(amount(case, k) == 0 for k in ("intraop_mdz", "intraop_epi"))
    )
    return has_tiva or has_sevo


def _find_step_ups(
    grid: np.ndarray,
    rate_ml_h: np.ndarray,
    *,
    min_step_ml_h: float,
    t_min: float,
    t_max: float,
) -> list[tuple[float, float, float]]:
    """Find isolated vasopressor infusion starts or >= 2x step-ups with 3-min quiet pre-window."""
    events: list[tuple[float, float, float]] = []
    sample = np.arange(t_min, t_max, 10.0)
    if len(sample) < 60:
        return events
    sampled_rate = np.interp(sample, grid[:-1], rate_ml_h)
    last_t = -1e9
    for idx in range(18, len(sample) - 30):
        t_now = float(sample[idx])
        if t_now - last_t < 600.0:
            continue
        pre = sampled_rate[idx - 18:idx]
        post = sampled_rate[idx + 6:idx + 30]
        if float(np.ptp(pre)) > 1.0:
            continue
        pre_r = float(np.median(pre))
        post_r = float(np.median(post))
        if post_r - pre_r >= min_step_ml_h and (pre_r <= 0.5 or post_r >= 2.0 * pre_r):
            if float(np.min(post)) >= pre_r + 0.5 * min_step_ml_h:
                events.append((t_now, pre_r, post_r))
                last_t = t_now
    return events


def evaluate(candidate: tuple[dict[str, str], dict[str, str]], dt: float) -> dict[str, Any]:
    case, tracks = candidate
    drug = _active_pressor(tracks)
    assert drug is not None
    rate_tname, vol_tname, ug_per_ml = PRESSOR_SPECS[drug]
    is_tiva = "Orchestra/PPF20_RATE" in tracks
    row: dict[str, Any] = dict(
        caseid=int(case["caseid"]),
        subjectid=int(case["subjectid"]),
        drug=drug,
        modality="TIVA" if is_tiva else "Sevoflurane",
        procedure=procedure(case),
        asa=case["asa"],
        hypertensive=case["preop_htn"] == "1",
        bolus_phe=amount(case, "intraop_phe"),
        bolus_eph=amount(case, "intraop_eph"),
    )
    load = functools.cache(numeric_track)
    try:
        opstart, opend = float(case["opstart"]), float(case["opend"])
        if opend - opstart < 900.0:
            raise Excluded("short surgery")
        p_rate_track = load(tracks[rate_tname])
        if np.max(p_rate_track[1]) < 1.0 or np.any((p_rate_track[1] < 0) | (p_rate_track[1] > 200)):
            raise Excluded("inactive or implausible pressor rate")
        if is_tiva:
            rate, volume, remi, remi_name, start, end, onset, stop = delivery(tracks, track_loader=load)
            full_end = min(opend, stop, end, float(p_rate_track[0][-1]))
            if opstart - onset < 300.0 or full_end - opstart < 600.0:
                raise Excluded("insufficient duration")
            baseline_map, baseline_hr = pre_induction(tracks, onset, track_loader=load)
            patient = patient_from_case(case, baseline_map=baseline_map, baseline_hr=baseline_hr)
            grid = time_grid(max(start, onset - 60.0), full_end, dt)
            propofol_mg = delivered_volume(rate, grid) * 20.0
            measured = np.interp(grid[-1], *volume) - np.interp(grid[0], *volume)
            if abs(propofol_mg[-1] - propofol_mg[0] - 20.0 * measured) > 20.0 * max(0.5, 0.05 * measured):
                raise Excluded("propofol rate and volume disagree")
            remi_ug = delivered_volume(remi, grid) * REMI_UG_PER_ML[remi_name]
            step_dt = np.diff(grid)
            propofol_mg_s = np.diff(propofol_mg) / step_dt
            remi_ug_s = np.diff(remi_ug) / step_dt
            inspired = None
            events: list[tuple[float, Any]] = [
                (onset + 240.0, lambda e: e.start_disturbance("stim_intubation_pulse")),
                (opstart, lambda e: e.start_disturbance("stim_sustained_surgery")),
                (opend, lambda e: e.stop_disturbance()),
            ]
        else:
            gases = {name: load(tracks[name]) for name in ("Primus/INSP_SEVO", "Primus/FIN2O")}
            if np.max(gases["Primus/FIN2O"][1]) > 1:
                raise Excluded("nitrous oxide exposure")
            sevo_t, sevo_v = gases["Primus/INSP_SEVO"]
            pos = sevo_t[sevo_v > 0.5]
            if not len(pos) or np.any((sevo_v < 0) | (sevo_v > 8)):
                raise Excluded("insufficient sevoflurane")
            first_sevo = float(pos[0])
            onset = min(first_sevo - 60.0, opstart - 300.0)
            full_end = min(opend, float(sevo_t[-1]), float(p_rate_track[0][-1]))
            if opstart - onset < 300.0 or full_end - opstart < 600.0:
                raise Excluded("insufficient duration")
            covered(gases["Primus/INSP_SEVO"], first_sevo, full_end, 30.0)
            baseline_map, baseline_hr = pre_induction(tracks, onset, track_loader=load)
            patient = patient_from_case(case, baseline_map=baseline_map, baseline_hr=baseline_hr)
            grid = time_grid(onset - 180.0, full_end, dt)
            step_dt = np.diff(grid)
            remi_name = next((name for name in REMI_RATES if name in tracks), None)
            if remi_name:
                remi = load(tracks[remi_name])
                if np.any((remi[1] < 0) | (remi[1] > 1500)):
                    raise Excluded("invalid remifentanil delivery")
                remi_ml_h = np.nan_to_num(held(remi, grid[:-1], 60.0), nan=0.0)
                remi_ug_s = remi_ml_h * REMI_UG_PER_ML[remi_name] / 3600.0
            else:
                remi_ug_s = np.zeros(len(step_dt), dtype=float)
            inspired = np.nan_to_num(held(gases["Primus/INSP_SEVO"], grid, 30.0))
            propofol_rate = amount(case, "intraop_ppf") / 20.0
            propofol_mg_s = np.where((grid[:-1] >= onset) & (grid[:-1] < onset + 20.0), propofol_rate, 0.0)
            events = [
                (onset + 150.0, lambda e: e.start_disturbance("stim_intubation_pulse")),
                (opstart, lambda e: e.start_disturbance("stim_sustained_surgery")),
                (opend, lambda e: e.stop_disturbance()),
            ]
            if amount(case, "intraop_ftn"):
                events.append((onset - 120.0, lambda e: e.give_drug_bolus("fentanyl", amount(case, "intraop_ftn"))))

        # Hold pressor rate across grid (zero before pump starts).
        p_rate_ml_h = np.nan_to_num(held(p_rate_track, grid[:-1], 60.0), nan=0.0)
        p_rate_ml_h = np.clip(p_rate_ml_h, 0.0, 200.0)
        p_ug_s = p_rate_ml_h * ug_per_ml / 3600.0
        if vol_tname in tracks:
            vol_track = load(tracks[vol_tname])
            int_vol = float(np.sum(p_rate_ml_h * step_dt) / 3600.0)
            rec_vol = float(np.interp(grid[-1], *vol_track) - np.interp(grid[0], *vol_track))
            if rec_vol > 1.0 and abs(int_vol - rec_vol) > max(1.0, 0.15 * rec_vol):
                raise Excluded("pressor rate and volume disagree")

        ebl = amount(case, "intraop_ebl")
        if ebl:
            bleed = ebl / max(1.0, (opend - opstart) / 60.0)
            events += [(opstart, lambda e: e.start_hemorrhage(bleed)), (opend, lambda e: e.stop_hemorrhage())]

        pred_with = run_hemodynamic_replay(
            case, patient, grid, onset=onset, events=events,
            propofol_mg_s=propofol_mg_s, remi_ug_s=remi_ug_s,
            inspired_sevo=inspired, vasopressor_ug_s={drug: p_ug_s},
        )
        pred_without = run_hemodynamic_replay(
            case, patient, grid, onset=onset, events=events,
            propofol_mg_s=propofol_mg_s, remi_ug_s=remi_ug_s,
            inspired_sevo=inspired,
        )

        # Evaluate active-infusion samples during surgery (from opstart + 600 s to full_end).
        art_track = load(tracks["Solar8000/ART_MBP"])
        hr_track = load(tracks["Solar8000/HR"])
        sample = np.arange(max(opstart + 600.0, onset + 600.0), full_end, SAMPLE_SECONDS)
        if len(sample) < 60:
            raise Excluded("short active window")
        s_rate = np.interp(sample, grid[:-1], p_rate_ml_h)
        s_map = held(art_track, sample, 30.0)
        s_hr = held(hr_track, sample, 30.0)
        active = (
            (s_rate >= 1.0)
            & np.isfinite(s_map) & (s_map > 35) & (s_map < 150)
            & np.isfinite(s_hr) & (s_hr > 35) & (s_hr < 150)
        )
        if active.sum() * SAMPLE_SECONDS < 300.0:
            raise Excluded("less than 5 min active infusion with arterial line")

        ts = sample[active]
        mod_w_map = np.interp(ts, grid, pred_with[:, 0])
        mod_w_hr = np.interp(ts, grid, pred_with[:, 1])
        mod_0_map = np.interp(ts, grid, pred_without[:, 0])
        mod_0_hr = np.interp(ts, grid, pred_without[:, 1])
        med_rate_ug_kg_min = float(np.median(s_rate[active]) * ug_per_ml / 60.0 / patient.weight)

        # Acute step-ups inside the arterial-line surgery window (excluding incision transition).
        min_step = 2.0 if drug == "nore" else 4.0
        step_results = []
        for t_step, pre_r, post_r in _find_step_ups(
            grid, p_rate_ml_h, min_step_ml_h=min_step, t_min=opstart + 600.0, t_max=full_end - 360.0,
        ):
            pre_grid = np.arange(t_step - 180.0, t_step, SAMPLE_SECONDS)
            post_grid = np.arange(t_step + 120.0, t_step + 300.0, SAMPLE_SECONDS)
            pre_m = held(art_track, pre_grid, 30.0)
            post_m = held(art_track, post_grid, 30.0)
            pre_h = held(hr_track, pre_grid, 30.0)
            post_h = held(hr_track, post_grid, 30.0)
            vp = np.isfinite(pre_m) & (pre_m > 35) & (pre_m < 150) & np.isfinite(pre_h)
            vq = np.isfinite(post_m) & (post_m > 35) & (post_m < 150) & np.isfinite(post_h)
            if vp.sum() < 12 or vq.sum() < 12:
                continue
            obs_dmap = float(np.median(post_m[vq]) - np.median(pre_m[vp]))
            obs_dhr = float(np.median(post_h[vq]) - np.median(pre_h[vp]))
            mod_dmap = float(
                np.median(np.interp(post_grid[vq], grid, pred_with[:, 0]))
                - np.median(np.interp(pre_grid[vp], grid, pred_with[:, 0]))
            )
            mod_dhr = float(
                np.median(np.interp(post_grid[vq], grid, pred_with[:, 1]))
                - np.median(np.interp(pre_grid[vp], grid, pred_with[:, 1]))
            )
            step_results.append(dict(
                delta_ug_kg_min=round((post_r - pre_r) * ug_per_ml / 60.0 / patient.weight, 4),
                obs_dmap=obs_dmap,
                mod_dmap=mod_dmap,
                obs_dhr=obs_dhr,
                mod_dhr=mod_dhr,
            ))
    except Excluded as error:
        return {**row, "exclusion": str(error)}

    return {
        **row,
        "age": float(patient.age),
        "weight": float(patient.weight),
        "active_min": round(float(active.sum() * SAMPLE_SECONDS / 60.0), 1),
        "median_rate_ug_kg_min": round(med_rate_ug_kg_min, 4),
        "obs_map": float(np.median(s_map[active])),
        "obs_hr": float(np.median(s_hr[active])),
        "with_map": float(np.median(mod_w_map)),
        "with_hr": float(np.median(mod_w_hr)),
        "without_map": float(np.median(mod_0_map)),
        "without_hr": float(np.median(mod_0_hr)),
        "steps": step_results,
    }


def report(data: dict[str, Any], settings: dict[str, Any]) -> str:
    rng = np.random.default_rng(settings["seed"])
    rows = data["rows"]
    lines = report_header("VitalDB continuous vasopressor infusions", settings) + [
        "## Methods", "",
        "Replays eligible unique-subject domain cases (ASA 1-4, no RBC/FFP transfusion, EBL <= 1500 mL)",
        "receiving a single continuous Orchestra norepinephrine (20 mcg/mL) or phenylephrine (100 mcg/mL)",
        "infusion with invasive arterial MAP (Solar8000/ART_MBP) and ECG HR. Each case is run both with",
        "the recorded pump infusion and as a zero-pressor counterfactual. Steady-state errors reflect",
        "confounding by indication because pressors are started and titrated upward in patients who are",
        "already more hypotensive than typical population predictions.", "",
        f"Eligible candidates: {data['eligible']}. Retained: {len(rows)}.", "",
        "## Active infusion steady state (model minus recorded)", "",
        "| Drug | n | Median rate (mcg/kg/min) | MAP err WITH pressor (mmHg) | MAP err WITHOUT pressor (mmHg) | Modeled MAP gain (mmHg) | HR err WITH (bpm) | HR delta WITH-WITHOUT (bpm) |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for drug, label in (("nore", "Norepinephrine"), ("phenyl", "Phenylephrine")):
        g = [r for r in rows if r["drug"] == drug]
        if not g:
            continue
        med_r = float(np.median([r["median_rate_ug_kg_min"] for r in g]))
        err_w = [r["with_map"] - r["obs_map"] for r in g]
        err_0 = [r["without_map"] - r["obs_map"] for r in g]
        gain = [r["with_map"] - r["without_map"] for r in g]
        hr_w = [r["with_hr"] - r["obs_hr"] for r in g]
        hr_d = [r["with_hr"] - r["without_hr"] for r in g]
        lines.append(
            f"| {label} | {len(g)} | {med_r:.3f} | {mean_ci(err_w, rng, include_n=False)} | "
            f"{mean_ci(err_0, rng, include_n=False)} | {mean_ci(gain, rng, include_n=False)} | "
            f"{mean_ci(hr_w, rng, include_n=False)} | {mean_ci(hr_d, rng, include_n=False)} |"
        )

    lines += ["", "## Acute infusion step-ups (+2 to +5 min vs preceding 3 min)", ""]
    any_steps = False
    for drug, label in (("nore", "Norepinephrine"), ("phenyl", "Phenylephrine")):
        steps = [s for r in rows if r["drug"] == drug for s in r.get("steps", [])]
        if not steps:
            continue
        any_steps = True
        med_d = float(np.median([s["delta_ug_kg_min"] for s in steps]))
        lines.append(
            f"- **{label}** ({len(steps)} step-ups, median +{med_d:.3f} mcg/kg/min): "
            f"recorded dMAP {mean_ci([s['obs_dmap'] for s in steps], rng, include_n=False)} vs "
            f"model dMAP {mean_ci([s['mod_dmap'] for s in steps], rng, include_n=False)} mmHg; "
            f"recorded dHR {mean_ci([s['obs_dhr'] for s in steps], rng, include_n=False)} vs "
            f"model dHR {mean_ci([s['mod_dhr'] for s in steps], rng, include_n=False)} bpm."
        )
    if not any_steps:
        lines.append("No isolated step-ups met the 3-minute quiet-baseline criteria.")

    lines += report_footer(data["exclusions"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--override", default="{}", help="JSON HemodynamicConfig fields")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    vitaldb_data.CACHE = args.cache
    override = json.loads(args.override)

    pool = [
        (case, trk)
        for case, trk in domain_cases([], open_supine_only=False, cache=args.cache)
        if eligible(case, trk)
    ]
    results = run_hemo_pool(evaluate, pool, args.dt, args.workers, override, args.cache)
    rows = [r for r in results if "exclusion" not in r]
    exclusions = [r for r in results if "exclusion" in r]
    data = dict(eligible=len(pool), rows=rows, exclusions=exclusions)
    settings = dict(generated=datetime.now(timezone.utc).date().isoformat(), dt=args.dt, seed=args.seed, override=override)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "vitaldb_vasopressors"
    stem.with_suffix(".json").write_text(json.dumps(dict(settings=settings, **data), indent=2, allow_nan=False) + "\n")
    stem.with_suffix(".md").write_text(report(data, settings))
    print(f"Report: {stem.with_suffix('.md')} (retained {len(rows)}/{len(pool)}, excl {dict(Counter(r['exclusion'] for r in exclusions))})", flush=True)


if __name__ == "__main__":
    main()
