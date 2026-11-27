"""Validate intraoperative ABGs, PaCO2-EtCO2 gradient, hemodilution Hct, and glucose against VitalDB labs.

Evaluates intraoperative arterial blood gas and chemistry panels from VitalDB labs.csv
across retained TIVA and sevoflurane domain cases (no transfusion, EBL <= 500 mL):
1. Arterial PaO2 at recorded FiO2, PEEP, TV, RR, age, BMI, and preop Hb.
2. Arterial-to-end-tidal CO2 gradient (PaCO2 - Primus/ETCO2) across age and surgical approach.
3. Hemodilution Hb/Hct drop from preoperative Hb under recorded crystalloid, colloid, and EBL.
4. Intraoperative acid-base (pH, HCO3-, lactate, iCa) and surgical stress glucose rise.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anasim.core.engine import SimulationEngine  # noqa: E402
from anasim.core.runtime import step_simulation  # noqa: E402
from anasim.core.state import SimulationConfig  # noqa: E402
from validation.vitaldb import vitaldb_data  # noqa: E402
from validation.vitaldb.vitaldb_data import (  # noqa: E402
    CACHE,
    DEFAULT_WORKERS,
    Excluded,
    amount,
    domain_cases,
    held,
    numeric_track,
    optional_float,
    patient_from_case,
)
from validation.vitaldb.vitaldb_replay import (  # noqa: E402
    mean_ci,
    report_footer,
    report_header,
    run_hemo_pool,
)


def load_labs(cache_dir: Path, keep_caseids: set[int]) -> dict[int, list[tuple[float, str, float]]]:
    """Load lab entries (dt, name, value) for selected caseids from VitalDB labs.csv."""
    path = cache_dir / "labs.csv"
    raw = path.read_bytes()
    text = (
        gzip.decompress(raw).decode("utf-8-sig", errors="ignore")
        if raw[:2] == b"\x1f\x8b"
        else raw.decode("utf-8-sig", errors="ignore")
    )
    out: dict[int, list[tuple[float, str, float]]] = defaultdict(list)
    for row in csv.DictReader(text.splitlines()):
        try:
            cid = int(row["caseid"])
            if cid not in keep_caseids:
                continue
            dt = float(row["dt"])
            val = float(row["result"])
        except (ValueError, TypeError):
            continue
        if np.isfinite(dt) and np.isfinite(val):
            out[cid].append((dt, row["name"], val))
    return out


def _group_abg_panels(
    entries: list[tuple[float, str, float]],
    opstart: float,
    opend: float,
) -> list[tuple[float, dict[str, float]]]:
    """Group intraoperative blood-gas entries sharing the same timestamp."""
    by_dt: dict[float, dict[str, float]] = defaultdict(dict)
    for dt, name, val in entries:
        if opstart + 300.0 <= dt <= opend:
            by_dt[dt][name] = val
    return sorted(by_dt.items())


def evaluate(candidate: tuple[dict[str, str], dict[str, str], list[tuple[float, str, float]]], dt_step: float) -> dict[str, Any]:
    del dt_step
    case, tracks, lab_entries = candidate
    row: dict[str, Any] = dict(
        caseid=int(case["caseid"]),
        subjectid=int(case["subjectid"]),
        approach=case.get("approach", ""),
    )
    load = functools.cache(numeric_track)
    try:
        opstart, opend = float(case["opstart"]), float(case["opend"])
        anestart, aneend = float(case["anestart"]), float(case["aneend"])
        panels = _group_abg_panels(lab_entries, opstart, opend)
        abgs = [(t, p) for t, p in panels if "po2" in p and "pco2" in p and "ph" in p]
        if not abgs:
            raise Excluded("no intraoperative ABG panel")
        # Pick the first well-settled intraoperative ABG panel (typically 30-90 min into surgery).
        t_abg, panel = abgs[0]
        obs_po2 = panel["po2"]
        obs_pco2 = panel["pco2"]
        obs_ph = panel["ph"]
        obs_hco3 = panel.get("hco3")
        obs_hct = panel.get("hct")
        obs_glu = panel.get("gluc")
        obs_lac = panel.get("lac")
        obs_ica = panel.get("ica")
        if not (40.0 <= obs_po2 <= 650.0 and 20.0 <= obs_pco2 <= 75.0 and 7.10 <= obs_ph <= 7.60):
            raise Excluded("implausible ABG values")

        needed = ("Primus/FIO2", "Primus/ETCO2", "Primus/TV", "Primus/RR_CO2", "Primus/PEEP_MBAR")
        if any(t not in tracks for t in needed):
            raise Excluded("missing ventilator gas tracks")

        sample = np.arange(t_abg - 180.0, t_abg + 1.0, 5.0)
        fio2 = float(np.nanmedian(held(load(tracks["Primus/FIO2"]), sample, 30.0)))
        etco2 = float(np.nanmedian(held(load(tracks["Primus/ETCO2"]), sample, 30.0)))
        tv = float(np.nanmedian(held(load(tracks["Primus/TV"]), sample, 30.0)))
        rr = float(np.nanmedian(held(load(tracks["Primus/RR_CO2"]), sample, 30.0)))
        peep = float(np.nanmedian(held(load(tracks["Primus/PEEP_MBAR"]), sample, 30.0)))
        if not (
            np.isfinite(fio2) and 21.0 <= fio2 <= 100.0
            and np.isfinite(etco2) and 18.0 <= etco2 <= 65.0
            and np.isfinite(tv) and 200.0 <= tv <= 1000.0
            and np.isfinite(rr) and 6.0 <= rr <= 30.0
            and np.isfinite(peep) and 0.0 <= peep <= 16.0
        ):
            raise Excluded("incomplete ventilator readings around ABG")

        # Determine preoperative glucose from preop labs when available.
        preop_glus = [v for t, nm, v in lab_entries if nm == "gluc" and t < anestart and 55.0 <= v <= 300.0]
        preop_glu = float(preop_glus[-1]) if preop_glus else None
        preop_hb = optional_float(case, "preop_hb")

        patient = patient_from_case(case)
        if preop_glu is not None:
            patient.baseline_glucose = preop_glu

        duration_h = max(0.5, (aneend - anestart) / 3600.0)
        cryst_rate = amount(case, "intraop_crystalloid") / duration_h
        colloid_rate_min = amount(case, "intraop_colloid") / duration_h / 60.0
        ebl = amount(case, "intraop_ebl")
        bleed_rate = ebl / max(1.0, (opend - opstart) / 60.0) if ebl > 0 else 0.0

        maint_type = "tiva" if "Orchestra/PPF20_RATE" in tracks else "balanced"
        eng = SimulationEngine(
            patient,
            SimulationConfig(
                mode="steady_state",
                maint_type=maint_type,
                dt=2.0,
                rng_seed=1,
                maintenance_fluid_ml_hr=cryst_rate,
            ),
        )
        eng.start()
        eng.set_vent_settings(rr=rr, vt=tv / 1000.0, peep=peep, ie="1:2", mode="VCV")
        # Use high FGF (12 L/min) so circuit inspired FiO2 matches recorded Primus/FIO2
        # without circle O2-uptake dilution confounding the PaO2 comparison.
        f_o2 = np.clip(fio2 / 100.0, 0.21, 1.0)
        air_l = 12.0 * max(0.0, (1.0 - f_o2) / 0.79)
        o2_l = max(0.2, 12.0 - air_l)
        eng.set_fgf(o2_l, air_l, 0.0)
        if bleed_rate > 0.0:
            eng.start_hemorrhage(bleed_rate)

        elapsed_s = float(np.clip(t_abg - anestart, 600.0, 7200.0))
        steps = max(1, round(elapsed_s / 2.0))
        for _ in range(steps):
            if colloid_rate_min > 0.0:
                eng.hemo.add_volume(colloid_rate_min * (2.0 / 60.0), label="colloid")
            step_simulation(eng, 2.0, update_monitors=False)
            eng.state.time += 2.0

        mod_po2 = float(eng.state.pao2)
        mod_pco2 = float(eng.state.pa_co2)
        mod_etco2 = float(eng.state.etco2)
        mod_ph = float(eng.state.ph)
        mod_hco3 = float(eng.state.hco3)
        mod_hb = float(eng.state.hb_g_dl)
        mod_glu = float(eng.state.glucose)
        mod_ica = float(eng.state.ionized_calcium)
        mod_lac = float(eng.state.lactate)
    except Excluded as error:
        return {**row, "exclusion": str(error)}

    return {
        **row,
        "age": float(patient.age),
        "bmi": round(float(patient.bmi), 2),
        "fio2": round(fio2, 1),
        "peep": round(peep, 1),
        "elapsed_min": round(elapsed_s / 60.0, 1),
        "obs_po2": obs_po2,
        "mod_po2": mod_po2,
        "obs_grad_co2": obs_pco2 - etco2,
        "mod_grad_co2": mod_pco2 - mod_etco2,
        "obs_ph": obs_ph,
        "mod_ph": mod_ph,
        "obs_hco3": obs_hco3,
        "mod_hco3": mod_hco3,
        "preop_hb": preop_hb,
        "obs_hb_est": (obs_hct / 3.0) if (obs_hct is not None and 15.0 <= obs_hct <= 60.0) else None,
        "mod_hb": mod_hb,
        "preop_glu": preop_glu,
        "obs_glu": obs_glu if (obs_glu is not None and 40.0 <= obs_glu <= 400.0) else None,
        "mod_glu": mod_glu,
        "obs_ica": obs_ica if (obs_ica is not None and 0.6 <= obs_ica <= 1.6) else None,
        "mod_ica": mod_ica,
        "obs_lac": obs_lac if (obs_lac is not None and 0.1 <= obs_lac <= 15.0) else None,
        "mod_lac": mod_lac,
    }


def report(data: dict[str, Any], settings: dict[str, Any]) -> str:
    rng = np.random.default_rng(settings["seed"])
    rows = data["rows"]
    lines = report_header("VitalDB intraoperative ABGs, hemodilution, and glucose", settings) + [
        "## Methods", "",
        "Evaluates the first settled intraoperative ABG/chemistry panel (>= 5 min after incision)",
        "across retained TIVA and sevoflurane cases (no RBC/FFP transfusion, EBL <= 500 mL) using",
        "recorded FiO2, PEEP, TV, RR, preop Hb/glucose, crystalloid/colloid, and EBL.", "",
        f"Retained TIVA/Sevo candidates with lab rows: {data['eligible']}. Analyzed with valid ABG + vent: {len(rows)}.", "",
        "## Oxygenation (PaO2, mmHg; model minus recorded)", "",
        "| Subgroup | n | Median FiO2 (%) | Recorded PaO2 | Model PaO2 | Mean error, mmHg (95% CI) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, g in [
        ("All ABGs", rows),
        ("Age under 50", [r for r in rows if r["age"] < 50]),
        ("Age 50-69", [r for r in rows if 50 <= r["age"] < 70]),
        ("Age 70 and over", [r for r in rows if r["age"] >= 70]),
        ("BMI under 25", [r for r in rows if r["bmi"] < 25]),
        ("BMI 25 and over", [r for r in rows if r["bmi"] >= 25]),
        ("PEEP <= 2 cmH2O", [r for r in rows if r["peep"] <= 2.0]),
        ("PEEP >= 4 cmH2O", [r for r in rows if r["peep"] >= 4.0]),
    ]:
        if len(g) < 3:
            continue
        lines.append(
            f"| {label} | {len(g)} | {float(np.median([r['fio2'] for r in g])):.0f} | "
            f"{float(np.median([r['obs_po2'] for r in g])):.0f} | "
            f"{float(np.median([r['mod_po2'] for r in g])):.0f} | "
            f"{mean_ci([r['mod_po2'] - r['obs_po2'] for r in g], rng, include_n=False)} |"
        )

    lines += [
        "", "## PaCO2 - EtCO2 gradient (mmHg)", "",
        "| Subgroup | n | Recorded gradient | Model gradient | Error (95% CI) |",
        "|---|---:|---:|---:|---|",
    ]
    for label, g in [
        ("All ABGs", rows),
        ("Open", [r for r in rows if r["approach"] == "Open"]),
        ("Videoscopic/Robotic", [r for r in rows if r["approach"] in ("Videoscopic", "Robotic")]),
        ("Age under 60", [r for r in rows if r["age"] < 60]),
        ("Age 60 and over", [r for r in rows if r["age"] >= 60]),
    ]:
        if len(g) < 3:
            continue
        lines.append(
            f"| {label} | {len(g)} | {float(np.median([r['obs_grad_co2'] for r in g])):.1f} | "
            f"{float(np.median([r['mod_grad_co2'] for r in g])):.1f} | "
            f"{mean_ci([r['mod_grad_co2'] - r['obs_grad_co2'] for r in g], rng, include_n=False)} |"
        )

    lines += ["", "## Acid-base, hemodilution Hb, and glucose (median recorded vs model)", ""]
    lines.append(
        f"- **pH** (n={len(rows)}): recorded {float(np.median([r['obs_ph'] for r in rows])):.3f} vs "
        f"model {float(np.median([r['mod_ph'] for r in rows])):.3f} "
        f"(error {mean_ci([r['mod_ph'] - r['obs_ph'] for r in rows], rng, include_n=False)})."
    )
    hco3_r = [r for r in rows if r.get("obs_hco3") is not None]
    if hco3_r:
        lines.append(
            f"- **HCO3-** (n={len(hco3_r)}): recorded {float(np.median([r['obs_hco3'] for r in hco3_r])):.1f} vs "
            f"model {float(np.median([r['mod_hco3'] for r in hco3_r])):.1f} mmol/L "
            f"(error {mean_ci([r['mod_hco3'] - r['obs_hco3'] for r in hco3_r], rng, include_n=False)})."
        )
    hb_r = [r for r in rows if r.get("preop_hb") is not None and r.get("obs_hb_est") is not None]
    if hb_r:
        obs_dhb = [r["obs_hb_est"] - r["preop_hb"] for r in hb_r]
        mod_dhb = [r["mod_hb"] - r["preop_hb"] for r in hb_r]
        lines.append(
            f"- **Hemodilution dHb** (Hct/3 minus preop Hb, n={len(hb_r)}): "
            f"recorded {mean_ci(obs_dhb, rng, include_n=False)} vs "
            f"model {mean_ci(mod_dhb, rng, include_n=False)} g/dL."
        )
    glu_r = [r for r in rows if r.get("preop_glu") is not None and r.get("obs_glu") is not None]
    if glu_r:
        obs_dg = [r["obs_glu"] - r["preop_glu"] for r in glu_r]
        mod_dg = [r["mod_glu"] - r["preop_glu"] for r in glu_r]
        lines.append(
            f"- **Intraoperative dGlucose** (intraop minus preop, n={len(glu_r)}): "
            f"recorded {mean_ci(obs_dg, rng, include_n=False)} vs "
            f"model {mean_ci(mod_dg, rng, include_n=False)} mg/dL."
        )
    ica_r = [r for r in rows if r.get("obs_ica") is not None]
    if ica_r:
        lines.append(
            f"- **Ionized Ca2+** (n={len(ica_r)}): recorded {float(np.median([r['obs_ica'] for r in ica_r])):.2f} vs "
            f"model {float(np.median([r['mod_ica'] for r in ica_r])):.2f} mmol/L."
        )
    lac_r = [r for r in rows if r.get("obs_lac") is not None]
    if lac_r:
        lines.append(
            f"- **Lactate** (n={len(lac_r)}): recorded {float(np.median([r['obs_lac'] for r in lac_r])):.2f} vs "
            f"model {float(np.median([r['mod_lac'] for r in lac_r])):.2f} mmol/L."
        )

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
    labs_by_case = load_labs(args.cache, keep)
    candidates = [
        (c, tr, labs_by_case[int(c["caseid"])])
        for c, tr in domain_cases((), open_supine_only=False)
        if int(c["caseid"]) in labs_by_case
    ]
    results = run_hemo_pool(evaluate, candidates, 1.0, args.workers, {}, args.cache)
    rows = [r for r in results if "exclusion" not in r]
    exclusions = [r for r in results if "exclusion" in r]
    data = dict(eligible=len(candidates), rows=rows, exclusions=exclusions)
    settings = dict(generated=datetime.now(timezone.utc).date().isoformat(), seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    md_path = args.output / "vitaldb_labs.md"
    json_path = args.output / "vitaldb_labs.json"
    md_path.write_text(report(data, settings) + "\n")
    json_path.write_text(json.dumps({"settings": settings, **data}, indent=2, allow_nan=False) + "\n")
    print(f"Report: {md_path} (retained {len(rows)}/{len(candidates)}, excl {dict(Counter(r['exclusion'] for r in exclusions))})", flush=True)


if __name__ == "__main__":
    main()
