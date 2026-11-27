"""Quantify laparoscopic CO2 pneumoperitoneum effects in VitalDB (Open vs Videoscopic/Robotic).

Compares paired pre-incision to surgical maintenance changes in respiratory compliance
(Primus/COMPLIANCE), end-tidal CO2, minute ventilation, CO2 elimination proxy
(VA * EtCO2), and untreated MAP between supine open and supine laparoscopic/robotic
surgeries in the retained TIVA + Sevoflurane domain cohort.
Provides empirical VitalDB targets alongside Joris 1993 and Putensen-Himmer 1992.
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
    Excluded,
    domain_cases,
    held,
    numeric_track,
    patient_from_case,
)
from validation.vitaldb.vitaldb_replay import (  # noqa: E402
    SAMPLE_SECONDS,
    mean_ci,
    report_footer,
    report_header,
    run_hemo_pool,
    treated,
)


def _window_stats(tracks: dict[str, str], t0: float, t1: float, dead_space_ml: float, load: Any) -> dict[str, float] | None:
    if t1 - t0 < 180.0:
        return None
    sample = np.arange(t0, t1, SAMPLE_SECONDS)
    if len(sample) < 24:
        return None
    comp = held(load(tracks["Primus/COMPLIANCE"]), sample, 30.0)
    etco2 = held(load(tracks["Primus/ETCO2"]), sample, 30.0)
    tv = held(load(tracks["Primus/TV"]), sample, 30.0)
    rr = held(load(tracks["Primus/RR_CO2"]), sample, 30.0)
    peep = held(load(tracks["Primus/PEEP_MBAR"]), sample, 30.0)
    p_name = "Solar8000/ART_MBP" if "Solar8000/ART_MBP" in tracks else "Solar8000/NIBP_MBP"
    mbp = held(load(tracks[p_name]), sample, 180.0)
    hr = held(load(tracks["Solar8000/HR"]), sample, 30.0)

    valid = (
        np.isfinite(comp) & (comp > 10.0) & (comp < 150.0)
        & np.isfinite(etco2) & (etco2 > 18.0) & (etco2 < 65.0)
        & np.isfinite(tv) & (tv > 150.0) & (tv < 1200.0)
        & np.isfinite(rr) & (rr > 5.0) & (rr < 35.0)
        & np.isfinite(peep) & (peep >= 0.0) & (peep <= 18.0)
        & np.isfinite(mbp) & (mbp > 40.0) & (mbp < 160.0)
        & np.isfinite(hr) & (hr > 35.0) & (hr < 150.0)
    )
    if valid.sum() * SAMPLE_SECONDS < 120.0:
        return None
    mv = tv[valid] * rr[valid] / 1000.0
    va = np.maximum(0.5, (tv[valid] - dead_space_ml) * rr[valid] / 1000.0)
    # Alveolar CO2 elimination proxy: VA (L/min) * EtCO2 (mmHg) / 0.863 ~ VCO2 (mL/min).
    vco2_proxy = va * etco2[valid] / 0.863
    out = dict(
        comp=float(np.median(comp[valid])),
        etco2=float(np.median(etco2[valid])),
        mv=float(np.median(mv)),
        va=float(np.median(va)),
        vco2=float(np.median(vco2_proxy)),
        peep=float(np.median(peep[valid])),
        map=float(np.median(mbp[valid])),
        hr=float(np.median(hr[valid])),
    )
    if "Primus/PPLAT_MBAR" in tracks:
        pplat = held(load(tracks["Primus/PPLAT_MBAR"]), sample[valid], 30.0)
        vp = np.isfinite(pplat) & (pplat > peep[valid] + 2.0) & (pplat < 50.0)
        if vp.sum() * SAMPLE_SECONDS >= 120.0:
            out["pplat"] = float(np.median(pplat[vp]))
    return out


def evaluate(candidate: tuple[dict[str, str], dict[str, str]], dt: float) -> dict[str, Any]:
    del dt
    case, tracks = candidate
    row: dict[str, Any] = dict(
        caseid=int(case["caseid"]),
        subjectid=int(case["subjectid"]),
        approach=case.get("approach", ""),
        treated=treated(case),
    )
    load = functools.cache(numeric_track)
    try:
        needed = ("Primus/COMPLIANCE", "Primus/ETCO2", "Primus/TV", "Primus/RR_CO2", "Primus/PEEP_MBAR", "Solar8000/HR")
        if any(t not in tracks for t in needed):
            raise Excluded("missing ventilator tracks")
        patient = patient_from_case(case)
        dead_space_ml = 2.2 * patient.predicted_body_weight()
        opstart, opend = float(case["opstart"]), float(case["opend"])
        if opend - opstart < 1800.0:
            raise Excluded("short surgery under 30 min")
        pre = _window_stats(tracks, opstart - 600.0, opstart, dead_space_ml, load)
        if pre is None:
            raise Excluded("insufficient pre-incision ventilation data")
        # Evaluate steady pneumoperitoneum window (20-60 min after incision, capped at opend - 300 s).
        surg_end = min(opend - 300.0, opstart + 3600.0)
        surg = _window_stats(tracks, opstart + 1200.0, surg_end, dead_space_ml, load)
        if surg is None:
            raise Excluded("insufficient surgical ventilation data")
    except Excluded as error:
        return {**row, "exclusion": str(error)}

    return {
        **row,
        "age": float(patient.age),
        "bmi": round(float(patient.bmi), 2),
        "pre": pre,
        "surg": surg,
    }


def report(data: dict[str, Any], settings: dict[str, Any]) -> str:
    rng = np.random.default_rng(settings["seed"])
    rows = data["rows"]
    open_r = [r for r in rows if r["approach"] == "Open"]
    lap_r = [r for r in rows if r["approach"] in ("Videoscopic", "Robotic")]
    lines = report_header("VitalDB supine open versus laparoscopic pneumoperitoneum", settings) + [
        "## Methods", "",
        "Compares paired pre-incision (10 min before opstart) to mid-surgery (20-60 min after opstart)",
        "changes in Primus compliance, EtCO2, minute ventilation, alveolar CO2 elimination",
        "proxy (VA * EtCO2 / 0.863), and untreated MAP between supine Open and supine Videoscopic/Robotic cases.", "",
        f"Supine retained candidates: {data['eligible']}. Paired complete: {len(rows)} "
        f"(Open: {len(open_r)}, Laparoscopic/Robotic: {len(lap_r)}).", "",
        "## Paired pre-incision to surgery changes", "",
        "| Group | n | dCompliance % (95% CI) | dEtCO2 mmHg (95% CI) | dVCO2 % (95% CI) | dMV % (95% CI) | dMAP mmHg (untreated) |",
        "|---|---:|---|---|---|---|---|",
    ]
    for label, g in [("Open, supine", open_r), ("Laparoscopic/Robotic, supine", lap_r)]:
        if len(g) < 3:
            continue
        dc = [100.0 * (r["surg"]["comp"] / r["pre"]["comp"] - 1.0) for r in g]
        det = [r["surg"]["etco2"] - r["pre"]["etco2"] for r in g]
        dvco2 = [100.0 * (r["surg"]["vco2"] / r["pre"]["vco2"] - 1.0) for r in g]
        dmv = [100.0 * (r["surg"]["mv"] / r["pre"]["mv"] - 1.0) for r in g]
        untr = [r for r in g if not r["treated"]]
        dmap = [r["surg"]["map"] - r["pre"]["map"] for r in untr]
        dmap_str = mean_ci(dmap, rng, include_n=True) if len(dmap) >= 3 else "n/a"
        lines.append(
            f"| {label} | {len(g)} | {mean_ci(dc, rng, include_n=False)} | "
            f"{mean_ci(det, rng, include_n=False)} | {mean_ci(dvco2, rng, include_n=False)} | "
            f"{mean_ci(dmv, rng, include_n=False)} | {dmap_str} |"
        )

    if len(open_r) >= 3 and len(lap_r) >= 3:
        diff_c = float(np.mean([100.0 * (r["surg"]["comp"] / r["pre"]["comp"] - 1.0) for r in lap_r])) - float(
            np.mean([100.0 * (r["surg"]["comp"] / r["pre"]["comp"] - 1.0) for r in open_r])
        )
        diff_vco2 = float(np.mean([100.0 * (r["surg"]["vco2"] / r["pre"]["vco2"] - 1.0) for r in lap_r])) - float(
            np.mean([100.0 * (r["surg"]["vco2"] / r["pre"]["vco2"] - 1.0) for r in open_r])
        )
        diff_et = float(np.mean([r["surg"]["etco2"] - r["pre"]["etco2"] for r in lap_r])) - float(
            np.mean([r["surg"]["etco2"] - r["pre"]["etco2"] for r in open_r])
        )
        open_u = [r["surg"]["map"] - r["pre"]["map"] for r in open_r if not r["treated"]]
        lap_u = [r["surg"]["map"] - r["pre"]["map"] for r in lap_r if not r["treated"]]
        diff_map = float(np.mean(lap_u)) - float(np.mean(open_u)) if open_u and lap_u else 0.0
        lines += [
            "", "## Net laparoscopic minus open pneumoperitoneum effect", "",
            f"- Compliance change: **{diff_c:+.1f}%** (laparoscopic minus open)",
            f"- Alveolar CO2 elimination proxy (VCO2): **{diff_vco2:+.1f}%**",
            f"- EtCO2 rise: **{diff_et:+.1f} mmHg**",
            f"- Untreated MAP rise: **{diff_map:+.1f} mmHg**",
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
    candidates = [
        (c, tr)
        for c, tr in domain_cases((), open_supine_only=False)
        if int(c["caseid"]) in keep
        and c.get("position") == "Supine"
        and c.get("approach") in ("Open", "Videoscopic", "Robotic")
    ]
    results = run_hemo_pool(evaluate, candidates, 1.0, args.workers, {}, args.cache)
    rows = [r for r in results if "exclusion" not in r]
    exclusions = [r for r in results if "exclusion" in r]
    data = dict(eligible=len(candidates), rows=rows, exclusions=exclusions)
    settings = dict(generated=datetime.now(UTC).date().isoformat(), seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    md_path = args.output / "vitaldb_laparoscopy.md"
    json_path = args.output / "vitaldb_laparoscopy.json"
    md_path.write_text(report(data, settings) + "\n")
    json_path.write_text(json.dumps({"settings": settings, **data}, indent=2, allow_nan=False) + "\n")
    print(f"Report: {md_path} (retained {len(rows)}/{len(candidates)}, excl {dict(Counter(r['exclusion'] for r in exclusions))})", flush=True)


if __name__ == "__main__":
    main()
