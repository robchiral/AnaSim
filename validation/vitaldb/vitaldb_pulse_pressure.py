"""Validate systolic, diastolic, pulse pressure (SBP - DBP), and CVP against VitalDB.

Evaluates:
1. Awake pre-induction cuff pulse pressure (NIBP_SBP - NIBP_DBP) across age and
   hypertension strata on the retained TIVA + Sevoflurane domain cohort against
   AnaSim's HemodynamicModel resting Windkessel/arterial-compliance curve
   (Franklin 1997; Langewouters 1984).
2. Intraoperative arterial-line SBP, DBP, pulse pressure, and CVP during untreated
   surgical maintenance replayed through the full engine.
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

from anasim.core.engine import SimulationEngine  # noqa: E402
from anasim.core.state import SimulationConfig  # noqa: E402
from validation.vitaldb import vitaldb_data  # noqa: E402
from validation.vitaldb.vitaldb_data import (  # noqa: E402
    CACHE,
    DEFAULT_WORKERS,
    Excluded,
    domain_cases,
    held,
    numeric_track,
    patient_from_case,
)
from validation.vitaldb.vitaldb_replay import (  # noqa: E402
    SAMPLE_SECONDS,
    delivery,
    mean_ci,
    report_footer,
    report_header,
    run_hemo_pool,
    treated,
)


def _pre_induction_bp(
    tracks: dict[str, str],
    onset: float,
    load: Any,
) -> tuple[float, float, float, float] | None:
    """Return last valid pre-induction (sbp, dbp, map, hr) within 15 min before onset."""
    if "Solar8000/NIBP_SBP" not in tracks or "Solar8000/NIBP_DBP" not in tracks:
        return None
    st, sv = load(tracks["Solar8000/NIBP_SBP"])
    dt_arr, dv = load(tracks["Solar8000/NIBP_DBP"])
    mt, mv = load(tracks["Solar8000/NIBP_MBP"])
    ht, hv = load(tracks["Solar8000/HR"])
    sm = (st >= onset - 900.0) & (st <= onset) & (sv > 60) & (sv < 240)
    dm = (dt_arr >= onset - 900.0) & (dt_arr <= onset) & (dv > 30) & (dv < 150)
    mm = (mt >= onset - 900.0) & (mt <= onset) & (mv > 45) & (mv < 180)
    hm = (ht >= onset - 120.0) & (ht <= onset) & (hv > 35) & (hv < 160)
    if not (np.any(sm) and np.any(dm) and np.any(mm) and np.any(hm)):
        return None
    sbp = float(sv[sm][-1])
    dbp = float(dv[dm][-1])
    mbp = float(mv[mm][-1])
    hr = float(np.median(hv[hm]))
    if sbp <= dbp + 10.0:
        return None
    return sbp, dbp, mbp, hr


def evaluate(candidate: tuple[dict[str, str], dict[str, str]], dt: float) -> dict[str, Any]:
    case, tracks = candidate
    row: dict[str, Any] = dict(
        caseid=int(case["caseid"]),
        subjectid=int(case["subjectid"]),
        hypertensive=case.get("preop_htn") == "1",
        treated=treated(case),
    )
    load = functools.cache(numeric_track)
    try:
        anestart = float(case["anestart"])
        opstart, opend = float(case["opstart"]), float(case["opend"])
        is_tiva = "Orchestra/PPF20_RATE" in tracks and "Orchestra/PPF20_VOL" in tracks
        onset = anestart
        if is_tiva:
            try:
                _r, _v, _rm, _rn, _s, _e, onset, _st = delivery(tracks, track_loader=load)
            except Excluded:
                onset = anestart
        bp0 = _pre_induction_bp(tracks, onset, load)
        if bp0 is None:
            raise Excluded("missing pre-induction NIBP SBP/DBP")
        sbp0, dbp0, map0, hr0 = bp0
        patient = patient_from_case(case, baseline_map=map0, baseline_hr=hr0)
        eng0 = SimulationEngine(patient, SimulationConfig(mode="awake", rng_seed=1))
        eng0.start()
        mod_sbp0 = float(eng0.state.sbp)
        mod_dbp0 = float(eng0.state.dbp)

        # Intraoperative arterial SBP/DBP/PP and CVP during surgical maintenance (when recorded).
        intraop: dict[str, Any] | None = None
        if (
            not row["treated"]
            and "Solar8000/ART_SBP" in tracks
            and "Solar8000/ART_DBP" in tracks
            and "Solar8000/ART_MBP" in tracks
            and opend - opstart >= 1500.0
        ):
            sample = np.arange(opstart + 900.0, opend, SAMPLE_SECONDS)
            a_sbp = held(load(tracks["Solar8000/ART_SBP"]), sample, 30.0)
            a_dbp = held(load(tracks["Solar8000/ART_DBP"]), sample, 30.0)
            a_map = held(load(tracks["Solar8000/ART_MBP"]), sample, 30.0)
            a_hr = held(load(tracks["Solar8000/HR"]), sample, 30.0)
            valid = (
                np.isfinite(a_sbp) & np.isfinite(a_dbp) & np.isfinite(a_map) & np.isfinite(a_hr)
                & (a_sbp > a_dbp + 10.0) & (a_sbp < 240.0) & (a_dbp > 25.0)
                & (a_map > 40.0) & (a_map < 160.0) & (a_hr > 35.0) & (a_hr < 150.0)
            )
            if valid.sum() * SAMPLE_SECONDS >= 600.0:
                obs_sbp = float(np.median(a_sbp[valid]))
                obs_dbp = float(np.median(a_dbp[valid]))
                obs_map = float(np.median(a_map[valid]))
                obs_hr = float(np.median(a_hr[valid]))
                obs_pp = float(np.median(a_sbp[valid] - a_dbp[valid]))
                # Evaluate AnaSim's arterial compliance model at recorded intraoperative MAP/HR
                # and anesthetized SV (isolating Windkessel PP from MAP bias).
                eng_s = SimulationEngine(
                    patient,
                    SimulationConfig(mode="steady_state", maint_type="tiva" if is_tiva else "balanced", rng_seed=1),
                )
                eng_s.start()
                mod_sbp_matched, mod_dbp_matched = eng_s.arterial_waveform.reference_pressures(
                    obs_map,
                    eng_s.hemo.state.sv,
                    obs_hr,
                    pulse_pressure_factor=eng_s.hemo.state.pulse_pressure_factor,
                    static_pressure=eng_s.hemo.state.venous_pressure,
                )
                mod_pp_matched = float(mod_sbp_matched - mod_dbp_matched)
                cvp_val = None
                if "Solar8000/CVP" in tracks:
                    cvp_arr = held(load(tracks["Solar8000/CVP"]), sample[valid], 30.0)
                    v_cvp = np.isfinite(cvp_arr) & (cvp_arr >= -2.0) & (cvp_arr <= 25.0)
                    if v_cvp.sum() * SAMPLE_SECONDS >= 300.0:
                        cvp_val = {
                            "observed": float(np.median(cvp_arr[v_cvp])),
                            "predicted": float(eng_s.hemo.state.rap),
                        }
                intraop = dict(
                    obs_map=obs_map,
                    obs_sbp=obs_sbp,
                    obs_dbp=obs_dbp,
                    obs_pp=obs_pp,
                    mod_sbp=mod_sbp_matched,
                    mod_dbp=mod_dbp_matched,
                    mod_pp=mod_pp_matched,
                    cvp=cvp_val,
                )
    except Excluded as error:
        return {**row, "exclusion": str(error)}

    del dt
    return {
        **row,
        "age": float(patient.age),
        "bmi": round(float(patient.bmi), 2),
        "awake_map": map0,
        "awake_sbp_obs": sbp0,
        "awake_dbp_obs": dbp0,
        "awake_pp_obs": sbp0 - dbp0,
        "awake_sbp_mod": mod_sbp0,
        "awake_dbp_mod": mod_dbp0,
        "awake_pp_mod": mod_sbp0 - mod_dbp0,
        "intraop": intraop,
    }


def report(data: dict[str, Any], settings: dict[str, Any]) -> str:
    rng = np.random.default_rng(settings["seed"])
    rows = data["rows"]
    lines = report_header("VitalDB pulse pressure, SBP/DBP, and CVP", settings) + [
        "## Methods", "",
        "Compares awake pre-induction cuff SBP, DBP, and pulse pressure (NIBP_SBP - NIBP_DBP) across",
        "retained TIVA and sevoflurane cases against AnaSim's HemodynamicModel resting Windkessel/arterial",
        "compliance curve (Franklin 1997; Langewouters 1984), and evaluates intraoperative invasive arterial",
        "pulse pressure (at matched MAP) and CVP during untreated surgery.", "",
        f"Candidate retained cases: {data['eligible']}. With valid pre-induction SBP/DBP: {len(rows)}.", "",
        "## Awake pre-induction pulse pressure (model minus recorded)", "",
        "| Subgroup | n | Recorded PP (mmHg) | Model PP (mmHg) | PP error, mmHg (95% CI) | SBP error (mmHg) | DBP error (mmHg) |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    groups = [
        ("All patients", rows),
        ("Age under 45", [r for r in rows if r["age"] < 45]),
        ("Age 45-59", [r for r in rows if 45 <= r["age"] < 60]),
        ("Age 60-69", [r for r in rows if 60 <= r["age"] < 70]),
        ("Age 70 and over", [r for r in rows if r["age"] >= 70]),
        ("Normotensive", [r for r in rows if not r["hypertensive"]]),
        ("Hypertensive", [r for r in rows if r["hypertensive"]]),
    ]
    for label, g in groups:
        if len(g) < 3:
            continue
        obs_pp = float(np.median([r["awake_pp_obs"] for r in g]))
        mod_pp = float(np.median([r["awake_pp_mod"] for r in g]))
        pp_err = [r["awake_pp_mod"] - r["awake_pp_obs"] for r in g]
        sbp_err = [r["awake_sbp_mod"] - r["awake_sbp_obs"] for r in g]
        dbp_err = [r["awake_dbp_mod"] - r["awake_dbp_obs"] for r in g]
        lines.append(
            f"| {label} | {len(g)} | {obs_pp:.1f} | {mod_pp:.1f} | "
            f"{mean_ci(pp_err, rng, include_n=False)} | "
            f"{mean_ci(sbp_err, rng, include_n=False)} | "
            f"{mean_ci(dbp_err, rng, include_n=False)} |"
        )

    art = [r for r in rows if r.get("intraop")]
    if art:
        lines += [
            "", "## Intraoperative invasive arterial pulse pressure (untreated surgery, MAP-matched)", "",
            "| Subgroup | n | Recorded PP (mmHg) | Model PP (mmHg) | PP error, mmHg (95% CI) |",
            "|---|---:|---:|---:|---|",
        ]
        for label, g in [
            ("All untreated arterial", art),
            ("Age under 60", [r for r in art if r["age"] < 60]),
            ("Age 60 and over", [r for r in art if r["age"] >= 60]),
            ("MAP under 75 mmHg", [r for r in art if r["intraop"]["obs_map"] < 75.0]),
            ("MAP 75 mmHg and over", [r for r in art if r["intraop"]["obs_map"] >= 75.0]),
        ]:
            if len(g) < 3:
                continue
            obs_pp = float(np.median([r["intraop"]["obs_pp"] for r in g]))
            mod_pp = float(np.median([r["intraop"]["mod_pp"] for r in g]))
            pp_err = [r["intraop"]["mod_pp"] - r["intraop"]["obs_pp"] for r in g]
            lines.append(
                f"| {label} | {len(g)} | {obs_pp:.1f} | {mod_pp:.1f} | {mean_ci(pp_err, rng, include_n=False)} |"
            )

    cvps = [r["intraop"]["cvp"] for r in art if r["intraop"].get("cvp")]
    if cvps:
        obs_c = float(np.median([c["observed"] for c in cvps]))
        mod_c = float(np.median([c["predicted"] for c in cvps]))
        err_c = [c["predicted"] - c["observed"] for c in cvps]
        lines += [
            "", "## Intraoperative CVP (untreated surgery)", "",
            f"n={len(cvps)}: median recorded {obs_c:.1f} vs model RAP {mod_c:.1f} mmHg "
            f"(error {mean_ci(err_c, rng, include_n=False)} mmHg).",
        ]

    return "\n".join(lines + report_footer(data["exclusions"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    vitaldb_data.CACHE = args.cache

    tiva_ids = {r["caseid"] for r in json.loads((ROOT / "results/vitaldb_tiva.json").read_text())["rows"]}
    sevo_ids = {r["caseid"] for r in json.loads((ROOT / "results/vitaldb_sevo.json").read_text())["rows"]}
    keep = tiva_ids | sevo_ids
    candidates = [(c, tr) for c, tr in domain_cases((), open_supine_only=False) if int(c["caseid"]) in keep]
    results = run_hemo_pool(evaluate, candidates, 1.0, args.workers, {}, args.cache)
    rows = [r for r in results if "exclusion" not in r]
    exclusions = [r for r in results if "exclusion" in r]
    data = dict(eligible=len(candidates), rows=rows, exclusions=exclusions)
    settings = dict(generated=datetime.now(timezone.utc).date().isoformat(), seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    md_path = args.output / "vitaldb_pulse_pressure.md"
    json_path = args.output / "vitaldb_pulse_pressure.json"
    md_path.write_text(report(data, settings) + "\n")
    json_path.write_text(json.dumps({"settings": settings, **data}, indent=2, allow_nan=False) + "\n")
    print(f"Report: {md_path} (retained {len(rows)}/{len(candidates)}, excl {dict(Counter(r['exclusion'] for r in exclusions))})", flush=True)


if __name__ == "__main__":
    main()
