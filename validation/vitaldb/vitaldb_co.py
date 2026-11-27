"""Validate cardiac output, stroke volume, and SVR against VitalDB EV1000 and Vigileo tracks.

Covers all eligible unique-subject oral single-lumen general anesthetics inside AnaSim's
adult domain (ASA 1-3, normotensive and hypertensive, without vasoactive infusions or
transfusion, EBL <= 500 mL) that record continuous pulse-contour CO and SV (EV1000 or
Vigileo) under TIVA or sevoflurane maintenance. Because EV1000/Vigileo is uncalibrated
pulse-contour monitoring placed after induction in almost all cases, the primary checks
compare paired pre-incision to surgery relative changes in CO, SV, MAP, HR, and SVR
across surgical approach/position strata, alongside absolute steady-state medians.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections import Counter
from datetime import UTC, datetime
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
    VASOACTIVE,
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
    treated,
)

INTUBATION_TIVA_S = 240.0
INTUBATION_SEVO_S = 150.0


def _co_sv_tracks(tracks: dict[str, str]) -> tuple[str, str, str | None] | None:
    if "EV1000/CO" in tracks and "EV1000/SV" in tracks:
        return "EV1000/CO", "EV1000/SV", ("EV1000/SVR" if "EV1000/SVR" in tracks else None)
    if "Vigileo/CO" in tracks and "Vigileo/SV" in tracks:
        return "Vigileo/CO", "Vigileo/SV", None
    return None


def eligible(case: dict[str, str], tracks: dict[str, str]) -> bool:
    if _co_sv_tracks(tracks) is None:
        return False
    if "Solar8000/HR" not in tracks or "Solar8000/NIBP_MBP" not in tracks:
        return False
    if any(f"Orchestra/{drug}_RATE" in tracks for drug in (*VASOACTIVE, "DEX2", "DEX4")):
        return False
    if amount(case, "intraop_rbc") > 0 or amount(case, "intraop_ffp") > 0 or amount(case, "intraop_ebl") > 500:
        return False
    if case["asa"] not in ("1", "2", "3") or case["preop_htn"] not in ("0", "1"):
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
        and "Primus/EXP_SEVO" in tracks
        and "Primus/FIN2O" in tracks
        and "Orchestra/PPF20_RATE" not in tracks
        and 0.0 < amount(case, "intraop_ppf") <= 3.0 * float(case["weight"])
        and amount(case, "intraop_ftn") <= 3.0 * float(case["weight"])
        and all(amount(case, k) == 0 for k in ("intraop_mdz", "intraop_epi"))
    )
    return has_tiva or has_sevo


def _window_co_medians(
    tracks: dict[str, str],
    co_name: str,
    sv_name: str,
    svr_name: str | None,
    grid: np.ndarray,
    predicted: np.ndarray,
    t0: float,
    t1: float,
    *,
    load: Any,
    min_seconds: float = 180.0,
) -> dict[str, dict[str, float]] | None:
    if t1 - t0 < min_seconds:
        return None
    sample = np.arange(t0, t1, SAMPLE_SECONDS)
    if len(sample) < int(min_seconds / SAMPLE_SECONDS):
        return None
    p_track = "Solar8000/ART_MBP" if "Solar8000/ART_MBP" in tracks else "Solar8000/NIBP_MBP"
    obs_map = held(load(tracks[p_track]), sample, 30.0)
    obs_hr = held(load(tracks["Solar8000/HR"]), sample, 30.0)
    obs_co = held(load(tracks[co_name]), sample, 30.0)
    obs_sv = held(load(tracks[sv_name]), sample, 30.0)
    valid = (
        np.isfinite(obs_map) & (obs_map > 30) & (obs_map < 160)
        & np.isfinite(obs_hr) & (obs_hr > 30) & (obs_hr < 160)
        & np.isfinite(obs_co) & (obs_co > 1.2) & (obs_co < 15.0)
        & np.isfinite(obs_sv) & (obs_sv > 15.0) & (obs_sv < 180.0)
    )
    if valid.sum() * SAMPLE_SECONDS < min_seconds:
        return None
    ts = sample[valid]
    # Derived SVR (Wood units, MAP/CO) avoids missing CVP offsets on Vigileo tracks.
    obs_svr_w = obs_map[valid] / obs_co[valid]
    mod_map = np.interp(ts, grid, predicted[:, 0])
    mod_hr = np.interp(ts, grid, predicted[:, 1])
    mod_co = np.interp(ts, grid, predicted[:, 3])
    mod_sv = np.interp(ts, grid, predicted[:, 4])
    mod_svr_w = mod_map / np.maximum(mod_co, 0.2)
    out: dict[str, dict[str, float]] = {
        "map": {"observed": float(np.median(obs_map[valid])), "predicted": float(np.median(mod_map))},
        "hr": {"observed": float(np.median(obs_hr[valid])), "predicted": float(np.median(mod_hr))},
        "co": {"observed": float(np.median(obs_co[valid])), "predicted": float(np.median(mod_co))},
        "sv": {"observed": float(np.median(obs_sv[valid])), "predicted": float(np.median(mod_sv))},
        "svr_wood": {"observed": float(np.median(obs_svr_w)), "predicted": float(np.median(mod_svr_w))},
    }
    if svr_name is not None:
        obs_svr_dyn = held(load(tracks[svr_name]), ts, 30.0)
        v_svr = np.isfinite(obs_svr_dyn) & (obs_svr_dyn > 250) & (obs_svr_dyn < 4000)
        if v_svr.sum() * SAMPLE_SECONDS >= min_seconds:
            out["ev1000_svr_dyn"] = {
                "observed": float(np.median(obs_svr_dyn[v_svr])),
                "predicted": float(np.median(np.interp(ts[v_svr], grid, predicted[:, 5]) * 80.0)),
            }
    return out


def evaluate(candidate: tuple[dict[str, str], dict[str, str]], dt: float) -> dict[str, Any]:
    case, tracks = candidate
    co_tracks = _co_sv_tracks(tracks)
    assert co_tracks is not None
    co_name, sv_name, svr_name = co_tracks
    is_tiva = "Orchestra/PPF20_RATE" in tracks
    row: dict[str, Any] = dict(
        caseid=int(case["caseid"]),
        subjectid=int(case["subjectid"]),
        modality="TIVA" if is_tiva else "Sevoflurane",
        device="EV1000" if co_name.startswith("EV1000") else "Vigileo",
        treated=treated(case),
        procedure=procedure(case),
        asa=case["asa"],
        hypertensive=case["preop_htn"] == "1",
    )
    load = functools.cache(numeric_track)
    try:
        opstart, opend = float(case["opstart"]), float(case["opend"])
        if opend - (opstart + 900.0) < 600.0:
            raise Excluded("short surgery window")
        if is_tiva:
            rate, volume, remi, remi_name, start, end, onset, stop = delivery(tracks, track_loader=load)
            full_end = min(opend, stop, end)
            if opstart - onset < 300.0 or full_end - (opstart + 900.0) < 600.0:
                raise Excluded("insufficient induction or surgery duration")
            baseline_map, baseline_hr = pre_induction(tracks, onset, track_loader=load)
            patient = patient_from_case(case, baseline_map=baseline_map, baseline_hr=baseline_hr)
            grid = time_grid(max(start, onset - 60.0), full_end, dt)
            propofol_mg = delivered_volume(rate, grid) * 20.0
            measured = np.interp(grid[-1], *volume) - np.interp(grid[0], *volume)
            if abs(propofol_mg[-1] - propofol_mg[0] - 20.0 * measured) > 20.0 * max(0.5, 0.05 * measured):
                raise Excluded("infusion rate and volume disagree")
            remi_ug = delivered_volume(remi, grid) * REMI_UG_PER_ML[remi_name]
            step_dt = np.diff(grid)
            propofol_mg_s = np.diff(propofol_mg) / step_dt
            remi_ug_s = np.diff(remi_ug) / step_dt
            inspired = None
            events: list[tuple[float, Any]] = [
                (onset + INTUBATION_TIVA_S, lambda e: e.start_disturbance("stim_intubation_pulse")),
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
                raise Excluded("insufficient or invalid sevoflurane")
            first_sevo = float(pos[0])
            onset = min(first_sevo - 60.0, opstart - 300.0)
            full_end = min(opend, float(sevo_t[-1]))
            if opstart - onset < 300.0 or full_end - (opstart + 900.0) < 600.0:
                raise Excluded("insufficient induction or surgery duration")
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
                (onset + INTUBATION_SEVO_S, lambda e: e.start_disturbance("stim_intubation_pulse")),
                (opstart, lambda e: e.start_disturbance("stim_sustained_surgery")),
                (opend, lambda e: e.stop_disturbance()),
            ]
            if amount(case, "intraop_ftn"):
                events.append((onset - 120.0, lambda e: e.give_drug_bolus("fentanyl", amount(case, "intraop_ftn"))))

        ebl = amount(case, "intraop_ebl")
        if ebl:
            bleed = ebl / max(1.0, (opend - opstart) / 60.0)
            events += [(opstart, lambda e: e.start_hemorrhage(bleed)), (opend, lambda e: e.stop_hemorrhage())]
        predicted = run_hemodynamic_replay(
            case, patient, grid, onset=onset, events=events,
            propofol_mg_s=propofol_mg_s, remi_ug_s=remi_ug_s,
            inspired_sevo=inspired, extra_outputs=True,
        )
        pre_inc = _window_co_medians(
            tracks, co_name, sv_name, svr_name, grid, predicted,
            max(onset + 300.0, opstart - 600.0), opstart, load=load, min_seconds=120.0,
        )
        surgery = _window_co_medians(
            tracks, co_name, sv_name, svr_name, grid, predicted,
            opstart + 900.0, full_end, load=load, min_seconds=300.0,
        )
        if surgery is None:
            raise Excluded("insufficient valid surgery CO/SV readings")
    except Excluded as error:
        return {**row, "exclusion": str(error)}

    return {
        **row,
        "age": float(patient.age),
        "bmi": round(float(patient.bmi), 2),
        "bsa": round(float(patient.bsa), 3),
        "before_incision": pre_inc,
        "surgery": surgery,
    }


def _pct(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def report(data: dict[str, Any], settings: dict[str, Any]) -> str:
    rng = np.random.default_rng(settings["seed"])
    rows = data["rows"]
    untreated = [r for r in rows if not r["treated"]]
    paired = [r for r in untreated if r.get("before_incision") and r.get("surgery")]
    lines = report_header("VitalDB cardiac output, stroke volume, and SVR", settings) + [
        "## Methods", "",
        "Replays all eligible unique-subject domain cases (ASA 1-3, no vasoactive infusions or transfusion,",
        "EBL <= 500 mL) recording EV1000 or Vigileo continuous CO and SV. Because pulse-contour CO monitors",
        "are uncalibrated and typically placed after induction, the primary endpoint is the paired percentage",
        "change from the pre-incision window to surgery (>= 15 min after incision) in untreated patients.",
        "Derived SVR uses MAP / CO (Wood units) alongside EV1000 SVR when recorded.", "",
        f"Eligible candidates: {data['eligible']}. Retained: {len(rows)} "
        f"(TIVA: {sum(r['modality'] == 'TIVA' for r in rows)}, "
        f"Sevoflurane: {sum(r['modality'] == 'Sevoflurane' for r in rows)}; "
        f"untreated: {len(untreated)}, paired pre-incision + surgery: {len(paired)}).", "",
    ]

    if paired:
        lines += [
            "## Pre-incision to surgery relative changes (untreated, paired)", "",
            "| Subgroup | n | dMAP rec / mod (%) | dHR rec / mod (%) | dCO rec / mod (%) | dSV rec / mod (%) | dSVR rec / mod (%) |",
            "|---|---:|---|---|---|---|---|",
        ]
        groups = [
            ("All paired untreated", paired),
            ("TIVA, all", [r for r in paired if r["modality"] == "TIVA"]),
            ("TIVA, Open supine", [r for r in paired if r["modality"] == "TIVA" and r["procedure"] == "Open, supine"]),
            ("TIVA, Laparoscopic/robotic supine", [r for r in paired if r["modality"] == "TIVA" and r["procedure"] == "Laparoscopic or robotic, supine"]),
            ("TIVA, Other positions", [r for r in paired if r["modality"] == "TIVA" and r["procedure"] == "Other positions"]),
            ("Sevoflurane, all", [r for r in paired if r["modality"] == "Sevoflurane"]),
        ]
        for label, g in groups:
            if len(g) < 3:
                continue
            cells = []
            for key in ("map", "hr", "co", "sv", "svr_wood"):
                rec = [_pct(r["surgery"][key]["observed"], r["before_incision"][key]["observed"]) for r in g]
                mod = [_pct(r["surgery"][key]["predicted"], r["before_incision"][key]["predicted"]) for r in g]
                cells.append(f"{np.mean(rec):+.1f}% / {np.mean(mod):+.1f}%")
            lines.append(f"| {label} | {len(g)} | " + " | ".join(cells) + " |")

        open_tiva = [r for r in paired if r["modality"] == "TIVA" and r["procedure"] == "Open, supine"]
        if len(open_tiva) >= 5:
            lines += [""]
            for key, unit_label in (("map", "MAP"), ("co", "CO"), ("sv", "SV"), ("svr_wood", "SVR (MAP/CO)")):
                err = [
                    _pct(r["surgery"][key]["predicted"], r["before_incision"][key]["predicted"])
                    - _pct(r["surgery"][key]["observed"], r["before_incision"][key]["observed"])
                    for r in open_tiva
                ]
                lines.append(
                    f"Open supine TIVA {unit_label} % change error (model minus recorded): {mean_ci(err, rng)} percentage points."
                )

    if untreated:
        lines += [
            "",
            "## Steady-state surgery medians (untreated; uncalibrated pulse-contour absolute values)", "",
            "| Subgroup | n | CO rec / mod (L/min) | SV rec / mod (mL) | MAP/CO rec / mod (Wood) | MAP err (mmHg) |",
            "|---|---:|---|---|---|---|",
        ]
        for label, g in [
            ("All untreated", untreated),
            ("TIVA, Open supine", [r for r in untreated if r["modality"] == "TIVA" and r["procedure"] == "Open, supine"]),
            ("TIVA, Laparoscopic/robotic supine", [r for r in untreated if r["modality"] == "TIVA" and r["procedure"] == "Laparoscopic or robotic, supine"]),
            ("TIVA, Other positions", [r for r in untreated if r["modality"] == "TIVA" and r["procedure"] == "Other positions"]),
            ("Sevoflurane, all", [r for r in untreated if r["modality"] == "Sevoflurane"]),
        ]:
            if len(g) < 3:
                continue
            co_r = np.median([r["surgery"]["co"]["observed"] for r in g])
            co_m = np.median([r["surgery"]["co"]["predicted"] for r in g])
            sv_r = np.median([r["surgery"]["sv"]["observed"] for r in g])
            sv_m = np.median([r["surgery"]["sv"]["predicted"] for r in g])
            svr_r = np.median([r["surgery"]["svr_wood"]["observed"] for r in g])
            svr_m = np.median([r["surgery"]["svr_wood"]["predicted"] for r in g])
            map_err = [r["surgery"]["map"]["predicted"] - r["surgery"]["map"]["observed"] for r in g]
            lines.append(
                f"| {label} | {len(g)} | {co_r:.2f} / {co_m:.2f} | {sv_r:.0f} / {sv_m:.0f} | "
                f"{svr_r:.1f} / {svr_m:.1f} | {mean_ci(map_err, rng, include_n=False)} |"
            )

    ev_svr = [r for r in untreated if "ev1000_svr_dyn" in r["surgery"]]
    if ev_svr:
        rec_svr = [r["surgery"]["ev1000_svr_dyn"]["observed"] for r in ev_svr]
        mod_svr = [r["surgery"]["ev1000_svr_dyn"]["predicted"] for r in ev_svr]
        lines += [
            "",
            f"EV1000 direct SVR during surgery (untreated, n={len(ev_svr)}): median recorded "
            f"{np.median(rec_svr):.0f} vs model {np.median(mod_svr):.0f} dyn*s/cm^5.",
        ]

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
    settings = dict(generated=datetime.now(UTC).date().isoformat(), dt=args.dt, seed=args.seed, override=override)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "vitaldb_co"
    stem.with_suffix(".json").write_text(json.dumps(dict(settings=settings, **data), indent=2, allow_nan=False) + "\n")
    stem.with_suffix(".md").write_text(report(data, settings))
    print(f"Report: {stem.with_suffix('.md')} (retained {len(rows)}/{len(pool)}, excl {dict(Counter(r['exclusion'] for r in exclusions))})", flush=True)


if __name__ == "__main__":
    main()
