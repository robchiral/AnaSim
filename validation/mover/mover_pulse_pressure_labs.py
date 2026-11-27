"""External MOVER validation of pulse pressure (awake NIBP and intraop A-line) and intraop ABGs/labs.

Mirrors `vitaldb_pulse_pressure.py` and `vitaldb_labs.py` on unique-MRN UCI Health EPIC
General Anesthesia cases with frozen AnaSim parameters:
1. Awake pre-induction NIBP SBP, DBP, and pulse pressure across age, BMI, and hypertension.
2. Intraoperative arterial-line pulse pressure matched to recorded MAP.
3. Settled intraoperative arterial blood gases (PaO2 across FiO2, age, and BMI; pH, HCO3-,
   iCa2+, lactate) plus hemodilution dHb and intraoperative dGlucose.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from validation.mover.mover_data import (
    CACHE_DIR,
    DEFAULT_MOVER_DIR,
    DEFAULT_WORKERS,
    RESULTS_DIR,
    SOURCE,
    _duckdb,
    fmt_mean_ci,
    load_cohort,
    load_flowsheet_subset,
    mean_ci,
    parse_bp_pair,
    patient_from_mover,
    utc_date,
    write_results,
)
from validation.vitaldb.vitaldb_replay import step_headless


def _sec_from(ts: datetime, ref: datetime) -> float:
    return (ts - ref).total_seconds()


def _eval_abg_case(payload: dict[str, Any]) -> dict[str, Any] | None:
    row = payload["row"]
    obs = payload["obs"]
    fio2 = float(payload["fio2"])
    elapsed_s = float(payload["elapsed_s"])
    cryst_rate = float(payload["cryst_rate"])
    qbl_ml = float(payload["qbl_ml"])

    patient = patient_from_mover(row, baseline_map=90.0, baseline_hr=70.0, use_preop_labs=True)
    eng = SimulationEngine(
        patient,
        SimulationConfig(
            mode="steady_state",
            maint_type="balanced",
            dt=2.0,
            rng_seed=1,
            maintenance_fluid_ml_hr=cryst_rate,
        ),
    )
    eng.start()
    vt = min(0.60, 0.008 * patient.predicted_body_weight())
    eng.set_vent_settings(rr=12, vt=vt, peep=5.0, ie="1:2", mode="VCV")
    # High FGF so circuit inspired O2 matches recorded FiO2 (%) without circle dilution.
    f_o2 = float(np.clip(fio2 / 100.0, 0.21, 1.0))
    air_l = 12.0 * max(0.0, (1.0 - f_o2) / 0.79)
    o2_l = max(0.2, 12.0 - air_l)
    eng.set_fgf(o2_l, air_l, 0.0)
    if qbl_ml > 0.0:
        eng.start_hemorrhage(qbl_ml / max(10.0, elapsed_s / 60.0))

    steps = max(1, round(min(7200.0, max(600.0, elapsed_s)) / 2.0))
    for _ in range(steps):
        step_headless(eng, 2.0)

    st = eng.state
    pre_hb = row.get("preop_hb")
    pre_glu = row.get("preop_glucose")
    obs_hb = obs.get("hb")
    obs_glu = obs.get("glu")

    return {
        "log_id": row["log_id"],
        "age": float(row["age"]),
        "bmi": float(row["bmi"]),
        "approach": str(row["approach"]),
        "fio2": fio2,
        "obs_po2": float(obs["po2"]),
        "mod_po2": float(st.pao2),
        "po2_err": float(st.pao2) - float(obs["po2"]),
        "obs_pco2": float(obs["pco2"]),
        "mod_pco2": float(st.pa_co2),
        "obs_ph": float(obs["ph"]),
        "mod_ph": float(st.ph),
        "obs_hco3": float(obs["hco3"]) if "hco3" in obs else float("nan"),
        "mod_hco3": float(st.hco3),
        "obs_ica": float(obs["ica"]) if "ica" in obs else float("nan"),
        "mod_ica": float(st.ionized_calcium),
        "obs_lac": float(obs["lac"]) if "lac" in obs else float("nan"),
        "mod_lac": float(st.lactate),
        "obs_dhb": (float(obs_hb) - float(pre_hb)) if (obs_hb is not None and pre_hb is not None) else float("nan"),
        "mod_dhb": (float(st.hb_g_dl) - float(pre_hb)) if (obs_hb is not None and pre_hb is not None) else float("nan"),
        "obs_dglu": (float(obs_glu) - float(pre_glu)) if (obs_glu is not None and pre_glu is not None) else float("nan"),
        "mod_dglu": (float(st.glucose) - float(pre_glu)) if (obs_glu is not None and pre_glu is not None) else float("nan"),
    }


def _evaluate_pulse_pressure(
    cohort: list[dict[str, Any]],
    *,
    mover_dir: Path,
    max_nibp: int,
    max_aline: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cohort))
    nibp_cands = [cohort[int(i)] for i in perm[: max_nibp * 2]]
    aline_pool = [r for r in cohort if r["has_aline"] == 1]
    aline_perm = rng.permutation(len(aline_pool))
    aline_cands = [aline_pool[int(i)] for i in aline_perm[: max_aline * 2]]

    all_ids = sorted({r["log_id"] for r in nibp_cands} | {r["log_id"] for r in aline_cands})
    bp_rows = load_flowsheet_subset(
        "BP",
        all_ids,
        ("NIBP", "BP", "NIBP - MAP", "MAP (mmHg)", "BP-ART A-line", "MAP-ART A-line"),
        mover_dir=mover_dir,
    )
    hr_rows = load_flowsheet_subset("HR", all_ids, ("Heart Rate", "Pulse"), mover_dir=mover_dir)

    awake_res: list[dict[str, Any]] = []
    for row in nibp_cands:
        if len(awake_res) >= max_nibp:
            break
        lid = row["log_id"]
        t_ind = row["t_induction"]
        pairs: list[tuple[float, float, float]] = []
        for ts, name, val in bp_rows.get(lid, []):
            if name not in ("NIBP", "BP"):
                continue
            dt_s = _sec_from(ts, t_ind)
            if not (-900.0 <= dt_s < 0.0):
                continue
            p = parse_bp_pair(val)
            if p is not None:
                pairs.append((dt_s, p[0], p[1]))
        if not pairs:
            continue
        pairs.sort()
        _, obs_sbp, obs_dbp = pairs[-1]
        obs_map = (obs_sbp + 2.0 * obs_dbp) / 3.0
        if not (55.0 <= obs_map <= 150.0):
            continue
        hrs = []
        for ts, _name, val in hr_rows.get(lid, []):
            dt_s = _sec_from(ts, t_ind)
            if -300.0 <= dt_s < 0.0:
                try:
                    v = float(val)
                    if 40.0 <= v <= 140.0:
                        hrs.append(v)
                except ValueError:
                    pass
        base_hr = float(np.median(hrs)) if hrs else 72.0
        patient = patient_from_mover(row, baseline_map=obs_map, baseline_hr=base_hr, use_preop_labs=False)
        eng = SimulationEngine(patient, SimulationConfig(mode="awake", rng_seed=1))
        eng.start()
        st = eng.state
        mod_pp = float(st.sbp - st.dbp)
        obs_pp = float(obs_sbp - obs_dbp)
        awake_res.append(
            {
                "age": float(row["age"]),
                "bmi": float(row["bmi"]),
                "htn": int(row["htn"]),
                "obs_sbp": obs_sbp,
                "obs_dbp": obs_dbp,
                "obs_pp": obs_pp,
                "mod_sbp": float(st.sbp),
                "mod_dbp": float(st.dbp),
                "mod_pp": mod_pp,
                "pp_err": mod_pp - obs_pp,
                "sbp_err": float(st.sbp) - obs_sbp,
                "dbp_err": float(st.dbp) - obs_dbp,
            }
        )

    aline_res: list[dict[str, Any]] = []
    for row in aline_cands:
        if len(aline_res) >= max_aline:
            break
        lid = row["log_id"]
        t_ind = row["t_induction"]
        sbps, dbps, maps = [], [], []
        for ts, name, val in bp_rows.get(lid, []):
            dt_s = _sec_from(ts, t_ind)
            if not (1200.0 <= dt_s <= 3600.0):
                continue
            if name == "BP-ART A-line":
                p = parse_bp_pair(val)
                if p is not None:
                    sbps.append(p[0])
                    dbps.append(p[1])
            elif name == "MAP-ART A-line":
                try:
                    v = float(val)
                    if 45.0 <= v <= 140.0:
                        maps.append(v)
                except ValueError:
                    pass
        if len(sbps) < 10:
            continue
        obs_sbp = float(np.median(sbps))
        obs_dbp = float(np.median(dbps))
        obs_map = float(np.median(maps)) if maps else (obs_sbp + 2.0 * obs_dbp) / 3.0
        obs_pp = obs_sbp - obs_dbp
        if obs_pp < 15.0:
            continue
        # Compare against anesthetized steady-state arterial pulse pressure at matched MAP.
        awake_est = float(np.clip(obs_map / 0.80, 70.0, 150.0))
        patient = patient_from_mover(row, baseline_map=awake_est, baseline_hr=68.0, use_preop_labs=False)
        eng = SimulationEngine(patient, SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1))
        eng.start()
        # Scale model PP proportionally if settled MAP differs slightly from recorded MAP.
        st = eng.state
        mod_pp = float((st.sbp - st.dbp) * (obs_map / max(40.0, st.map)))
        aline_res.append(
            {
                "age": float(row["age"]),
                "obs_map": obs_map,
                "obs_pp": obs_pp,
                "mod_pp": mod_pp,
                "pp_err": mod_pp - obs_pp,
            }
        )
    return awake_res, aline_res


def run(
    *,
    mover_dir: Path = DEFAULT_MOVER_DIR,
    cache_dir: Path = CACHE_DIR,
    output_dir: Path = RESULTS_DIR,
    max_nibp: int = 1500,
    max_aline: int = 500,
    max_abgs: int = 400,
    workers: int = DEFAULT_WORKERS,
    seed: int = 42,
) -> dict[str, Any]:
    cohort = load_cohort(
        mover_dir,
        cache_dir,
        where="has_cpb = 0 AND has_olv = 0 AND asa <= 3 AND t_induction IS NOT NULL",
    )
    awake_pp, aline_pp = _evaluate_pulse_pressure(
        cohort, mover_dir=mover_dir, max_nibp=max_nibp, max_aline=max_aline, seed=seed
    )

    # Part B: Intraoperative arterial blood gases (A-line non-CPB non-OLV cases)
    abg_cohort = [r for r in cohort if r["has_aline"] == 1]
    abg_by_id = {r["log_id"]: r for r in abg_cohort}

    con = _duckdb().connect()
    con.execute("CREATE TEMP TABLE ids (LOG_ID VARCHAR, t_ind TIMESTAMP, t_stop TIMESTAMP)")
    con.executemany(
        "INSERT INTO ids VALUES (?, ?, ?)",
        [(r["log_id"], r["t_induction"], r["t_an_stop"]) for r in abg_cohort],
    )
    from validation.mover.mover_data import _emr_csv

    labs_sql = str(_emr_csv(mover_dir, "patient_labs.csv")).replace("'", "''")
    lab_rows = con.execute(
        f"""
        SELECT l.LOG_ID, TRY_CAST(l."Collection Datetime" AS TIMESTAMP) AS ts,
               l."Lab Code", TRY_CAST(l."Observation Value" AS DOUBLE) AS val
        FROM read_csv('{labs_sql}', header=true, all_varchar=true, ignore_errors=true, strict_mode=false) l
        JOIN ids i USING (LOG_ID)
        WHERE l."Lab Code" IN ('11556-8', '2703-7', '11557-6', '2019-8', '11558-4', '2744-1',
                               '1959-6', '1960-4', '1994-3', '2520-5', '14045-9', '718-7', '30350-3', '2339-0', '2345-7')
          AND TRY_CAST(l."Collection Datetime" AS TIMESTAMP) BETWEEN i.t_ind + INTERVAL 20 MINUTE AND i.t_stop
          AND TRY_CAST(l."Observation Value" AS DOUBLE) IS NOT NULL
        ORDER BY l.LOG_ID, ts
        """
    ).fetchall()

    code_map = {
        "11556-8": "po2",
        "2703-7": "po2",
        "11557-6": "pco2",
        "2019-8": "pco2",
        "11558-4": "ph",
        "2744-1": "ph",
        "1959-6": "hco3",
        "1960-4": "hco3",
        "1994-3": "ica",
        "2520-5": "lac",
        "14045-9": "lac",
        "718-7": "hb",
        "30350-3": "hb",
        "2339-0": "glu",
        "2345-7": "glu",
    }
    panels_by_log: dict[str, dict[datetime, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for lid, ts, code, val in lab_rows:
        if ts is None or not np.isfinite(val):
            continue
        key = code_map[code]
        if key == "po2" and not (60.0 <= val <= 650.0):
            continue
        if key == "pco2" and not (20.0 <= val <= 75.0):
            continue
        if key == "ph" and not (7.10 <= val <= 7.60):
            continue
        if key == "hco3" and not (10.0 <= val <= 45.0):
            continue
        if key == "ica" and not (0.6 <= val <= 1.8):
            continue
        if key == "lac" and not (0.2 <= val <= 15.0):
            continue
        if key == "hb" and not (5.0 <= val <= 20.0):
            continue
        if key == "glu" and not (45.0 <= val <= 450.0):
            continue
        panels_by_log[lid][ts][key] = float(val)

    # Pick first complete arterial panel per unique patient.
    first_panels: list[tuple[str, datetime, dict[str, float]]] = []
    for lid, by_ts in panels_by_log.items():
        for ts in sorted(by_ts):
            p = by_ts[ts]
            if "po2" in p and "pco2" in p and "ph" in p:
                first_panels.append((lid, ts, p))
                break

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(first_panels))
    sampled_panels = [first_panels[int(i)] for i in perm[: max_abgs * 2]]
    abg_ids = [x[0] for x in sampled_panels]

    o2_rows = load_flowsheet_subset("O2", abg_ids, ("FiO2 (%)",), mover_dir=mover_dir)
    blood_rows = load_flowsheet_subset("Blood", abg_ids, ("QBL", "PRBC"), mover_dir=mover_dir)
    vol_rows = load_flowsheet_subset("Volume", abg_ids, ("Volume",), mover_dir=mover_dir)

    abg_tasks: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for lid, t_abg, panel in sampled_panels:
        if len(abg_tasks) >= max_abgs:
            break
        row = abg_by_id[lid]
        # Exclude RBC-transfused or large-hemorrhage cases (EBL > 500 mL) for clean hemodilution/ABG check.
        qbl = 0.0
        prbc = 0.0
        for ts, name, val in blood_rows.get(lid, []):
            if ts > t_abg:
                continue
            try:
                v = float(val)
            except ValueError:
                continue
            if name == "PRBC" and v > 0:
                prbc += v
            elif name == "QBL" and 0 < v <= 10000.0:
                qbl += v
        if prbc > 0 or qbl > 500.0:
            exclusions["transfusion or QBL over 500 mL before ABG"] += 1
            continue

        fio2_vals = []
        for ts, _name, val in o2_rows.get(lid, []):
            dt_s = _sec_from(ts, t_abg)
            if -600.0 <= dt_s <= 60.0:
                try:
                    v = float(val)
                    if 21.0 <= v <= 100.0:
                        fio2_vals.append(v)
                except ValueError:
                    pass
        if len(fio2_vals) < 2 or max(fio2_vals) - min(fio2_vals) > 15.0:
            exclusions["missing or unstable FiO2 before ABG"] += 1
            continue
        fio2 = float(np.median(fio2_vals))

        vol_before = 0.0
        for ts, _name, val in vol_rows.get(lid, []):
            if ts <= t_abg:
                try:
                    v = float(val)
                    if 0.0 < v <= 2000.0:
                        vol_before += v
                except ValueError:
                    pass
        elapsed_s = _sec_from(t_abg, row["t_induction"])
        cryst_rate = float(np.clip(vol_before / max(0.5, elapsed_s / 3600.0), 0.0, 2000.0)) if vol_before > 0 else 400.0

        abg_tasks.append(
            {
                "row": row,
                "obs": panel,
                "fio2": fio2,
                "elapsed_s": elapsed_s,
                "cryst_rate": cryst_rate,
                "qbl_ml": qbl,
            }
        )

    abg_res: list[dict[str, Any]] = []
    if workers > 1 and len(abg_tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_eval_abg_case, abg_tasks, chunksize=8):
                if res is not None:
                    abg_res.append(res)
    else:
        for t in abg_tasks:
            res = _eval_abg_case(t)
            if res is not None:
                abg_res.append(res)

    lines = [
        "# MOVER external validation: pulse pressure and intraoperative ABGs/labs",
        "",
        f"Generated {utc_date()}. Frozen AnaSim parameters (no refitting on MOVER).",
        "",
        "## Awake pre-induction NIBP pulse pressure (model minus recorded)",
        "",
        "| Subgroup | n | Recorded PP (mmHg) | Model PP (mmHg) | PP error (95% CI) | SBP error | DBP error |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    awake_groups: dict[str, Any] = {}
    for label, sub in (
        ("All patients", awake_pp),
        ("Age under 45", [r for r in awake_pp if r["age"] < 45]),
        ("Age 45-59", [r for r in awake_pp if 45 <= r["age"] < 60]),
        ("Age 60-69", [r for r in awake_pp if 60 <= r["age"] < 70]),
        ("Age 70 and over", [r for r in awake_pp if r["age"] >= 70]),
        ("Normotensive", [r for r in awake_pp if not r["htn"]]),
        ("Hypertensive", [r for r in awake_pp if r["htn"]]),
        ("BMI under 30", [r for r in awake_pp if r["bmi"] < 30]),
        ("BMI 30 and over", [r for r in awake_pp if r["bmi"] >= 30]),
    ):
        if len(sub) < 5:
            continue
        rec_pp = float(np.median([r["obs_pp"] for r in sub]))
        mod_pp = float(np.median([r["mod_pp"] for r in sub]))
        pp_ci = mean_ci([r["pp_err"] for r in sub], seed=seed)
        sbp_ci = mean_ci([r["sbp_err"] for r in sub], seed=seed)
        dbp_ci = mean_ci([r["dbp_err"] for r in sub], seed=seed)
        lines.append(
            f"| {label} | {len(sub)} | {rec_pp:.1f} | {mod_pp:.1f} | "
            f"{fmt_mean_ci(pp_ci)} | {fmt_mean_ci(sbp_ci)} | {fmt_mean_ci(dbp_ci)} |"
        )
        awake_groups[label] = {"n": len(sub), "rec_pp": rec_pp, "mod_pp": mod_pp, "pp_err_ci": pp_ci}

    lines.extend(
        [
            "",
            "## Intraoperative invasive arterial pulse pressure (MAP-matched)",
            "",
            "| Subgroup | n | Recorded PP (mmHg) | Model PP (mmHg) | PP error (95% CI) |",
            "|---|---:|---:|---:|---|",
        ]
    )
    aline_groups: dict[str, Any] = {}
    for label, sub in (
        ("All A-line", aline_pp),
        ("Age under 60", [r for r in aline_pp if r["age"] < 60]),
        ("Age 60 and over", [r for r in aline_pp if r["age"] >= 60]),
        ("MAP under 75 mmHg", [r for r in aline_pp if r["obs_map"] < 75]),
        ("MAP 75 mmHg and over", [r for r in aline_pp if r["obs_map"] >= 75]),
    ):
        if len(sub) < 5:
            continue
        rec_pp = float(np.median([r["obs_pp"] for r in sub]))
        mod_pp = float(np.median([r["mod_pp"] for r in sub]))
        pp_ci = mean_ci([r["pp_err"] for r in sub], seed=seed)
        lines.append(f"| {label} | {len(sub)} | {rec_pp:.1f} | {mod_pp:.1f} | {fmt_mean_ci(pp_ci)} |")
        aline_groups[label] = {"n": len(sub), "rec_pp": rec_pp, "mod_pp": mod_pp, "pp_err_ci": pp_ci}

    lines.extend(
        [
            "",
            "## Intraoperative arterial oxygenation (PaO2, mmHg; model minus recorded)",
            "",
            "Standard ventilator settings assumed (VT = 8 mL/kg PBW, RR = 12/min, PEEP = 5 cmH2O)",
            "because PEEP and TV are absent from the MOVER flowsheet subset; FiO2 (%) is recorded.",
            "",
            "| Subgroup | n | Median FiO2 (%) | Recorded PaO2 | Model PaO2 | Mean error (95% CI) |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    abg_groups: dict[str, Any] = {}
    for label, sub in (
        ("All ABGs", abg_res),
        ("Age under 50", [r for r in abg_res if r["age"] < 50]),
        ("Age 50-69", [r for r in abg_res if 50 <= r["age"] < 70]),
        ("Age 70 and over", [r for r in abg_res if r["age"] >= 70]),
        ("BMI under 25", [r for r in abg_res if r["bmi"] < 25]),
        ("BMI 25-29.9", [r for r in abg_res if 25 <= r["bmi"] < 30]),
        ("BMI 30-34.9", [r for r in abg_res if 30 <= r["bmi"] < 35]),
        ("BMI 35 and over", [r for r in abg_res if r["bmi"] >= 35]),
        ("Open", [r for r in abg_res if r["approach"] == "Open"]),
        ("Laparoscopic or robotic", [r for r in abg_res if r["approach"] == "Laparoscopic or robotic"]),
    ):
        if len(sub) < 5:
            continue
        f_med = float(np.median([r["fio2"] for r in sub]))
        rec_p = float(np.median([r["obs_po2"] for r in sub]))
        mod_p = float(np.median([r["mod_po2"] for r in sub]))
        err_ci = mean_ci([r["po2_err"] for r in sub], seed=seed)
        lines.append(f"| {label} | {len(sub)} | {f_med:.0f} | {rec_p:.0f} | {mod_p:.0f} | {fmt_mean_ci(err_ci)} |")
        abg_groups[label] = {"n": len(sub), "rec_po2": rec_p, "mod_po2": mod_p, "po2_err_ci": err_ci}

    lines.extend(["", "## Intraoperative acid-base, hemodilution Hb, and glucose (median recorded vs model)", ""])
    chem_summary: dict[str, Any] = {}
    if abg_res:
        for key, label, obs_k, mod_k, unit, digits in (
            ("pco2", "PaCO2", "obs_pco2", "mod_pco2", " mmHg", 1),
            ("ph", "pH", "obs_ph", "mod_ph", "", 3),
            ("hco3", "HCO3-", "obs_hco3", "mod_hco3", " mmol/L", 1),
            ("ica", "Ionized Ca2+", "obs_ica", "mod_ica", " mmol/L", 2),
            ("lac", "Lactate", "obs_lac", "mod_lac", " mmol/L", 2),
            ("dhb", "Hemodilution dHb", "obs_dhb", "mod_dhb", " g/dL", 2),
            ("dglu", "Intraoperative dGlucose", "obs_dglu", "mod_dglu", " mg/dL", 1),
        ):
            sub = [r for r in abg_res if np.isfinite(r[obs_k]) and np.isfinite(r[mod_k])]
            if len(sub) < 5:
                continue
            r_med = float(np.median([r[obs_k] for r in sub]))
            m_med = float(np.median([r[mod_k] for r in sub]))
            e_ci = mean_ci([r[mod_k] - r[obs_k] for r in sub], seed=seed)
            lines.append(
                f"- **{label}** (n={len(sub)}): recorded {r_med:.{digits}f}{unit} vs model "
                f"{m_med:.{digits}f}{unit} (err {fmt_mean_ci(e_ci, digits=digits)})"
            )
            chem_summary[key] = {"n": len(sub), "rec_med": r_med, "mod_med": m_med, "err_ci": e_ci}

    lines.extend(["", "## ABG Exclusions", ""])
    for reason, cnt in sorted(exclusions.items()):
        lines.append(f"- {reason}: {cnt}")
    lines.extend(["", SOURCE])

    payload = {
        "generated": utc_date(),
        "awake_pp": awake_groups,
        "aline_pp": aline_groups,
        "abg_po2": abg_groups,
        "chemistry": chem_summary,
        "exclusions": dict(exclusions),
    }
    write_results("mover_pulse_pressure_labs", lines, payload, output_dir=output_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="MOVER external pulse pressure and ABG/lab validation.")
    parser.add_argument("--mover-dir", type=Path, default=DEFAULT_MOVER_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-nibp", type=int, default=1500)
    parser.add_argument("--max-aline", type=int, default=500)
    parser.add_argument("--max-abgs", type=int, default=400)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = run(
        mover_dir=args.mover_dir,
        cache_dir=args.cache,
        output_dir=args.output,
        max_nibp=args.max_nibp,
        max_aline=args.max_aline,
        max_abgs=args.max_abgs,
        workers=args.workers,
        seed=args.seed,
    )
    print(f"Wrote mover_pulse_pressure_labs report: abgs={out['abg_po2'].get('All ABGs', {}).get('n', 0)}")


if __name__ == "__main__":
    main()
