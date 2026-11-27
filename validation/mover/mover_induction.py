"""External MOVER validation of induction hemodynamics (Sevoflurane, TIVA, and Etomidate).

Replays frozen AnaSim parameters (no refitting) across unique-patient UCI Health EPIC
encounters with exact MAR bolus/infusion timestamps, Induction/Intubation events, and
1-minute arterial or cuff blood pressure and heart rate tracks.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    VASOACTIVE_DRUGS,
    fmt_mean_ci,
    fmt_med_iqr,
    load_cohort,
    load_flowsheet_subset,
    load_meds_for_logs,
    mean_ci,
    median_iqr,
    parse_bp_pair,
    patient_from_mover,
    utc_date,
    write_results,
)
from validation.vitaldb.vitaldb_replay import CUFF_WINDOW_S, STOP, step_headless

_BOLUS_MAP = {
    "propofol": "Propofol",
    "fentanyl": "Fentanyl",
    "remifentanil": "Remifentanil",
    "midazolam": "Midazolam",
    "etomidate": "Etomidate",
    "ketamine": "Ketamine",
    "lidocaine": "Lidocaine",
    "hydromorphone": "Hydromorphone",
    "morphine": "Morphine",
    "rocuronium": "Rocuronium",
    "succinylcholine": "Succinylcholine",
}

_EXCLUDED_NON_VASO = frozenset({"ketamine", "dexmedetomidine"})
_SEVO_NAMES = ("SEV In", "SEV ET", "ISO In", "DES In", "ETN2O", "FIN20 (%)")
_BP_NAMES = (
    "MAP-ART A-line",
    "BP-ART A-line",
    "Arterial Line MAP (ART)",
    "Arterial Line BP (ART)",
    "NIBP - MAP",
    "NIBP",
    "MAP (mmHg)",
    "BP",
)
_HR_NAMES = ("Heart Rate", "Pulse")


def _sec_from(ts: datetime, ref: datetime) -> float:
    return (ts - ref).total_seconds()


def _extract_bp_series(
    rows: list[tuple[datetime, str, str]],
    t_ind: datetime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return `(t_s, map_mmhg, sbp_mmhg, is_aline)` sorted by time relative to `t_ind`."""
    aline_map: dict[float, float] = {}
    aline_sbp: dict[float, float] = {}
    nibp_map: dict[float, float] = {}
    nibp_sbp: dict[float, float] = {}

    for ts, name, val in rows:
        t_s = _sec_from(ts, t_ind)
        if not (-1200.0 <= t_s <= 1200.0):
            continue
        is_art = "ART" in name or "A-line" in name
        target_map = aline_map if is_art else nibp_map
        target_sbp = aline_sbp if is_art else nibp_sbp
        if "MAP" in name:
            try:
                v = float(val)
            except ValueError:
                continue
            if 30.0 <= v <= 180.0:
                target_map[t_s] = v
        else:
            pair = parse_bp_pair(val)
            if pair is None:
                continue
            sbp, dbp = pair
            target_sbp[t_s] = sbp
            if t_s not in target_map:
                target_map[t_s] = (sbp + 2.0 * dbp) / 3.0

    # Prefer arterial line only if it has BOTH pre-induction baseline AND post-induction samples.
    art_pre = [t for t in aline_map if -900.0 <= t < 0.0]
    art_post = [t for t in aline_map if 60.0 <= t <= 600.0]
    use_art = len(art_pre) >= 1 and len(art_post) >= 3
    chosen_map = aline_map if use_art else nibp_map
    chosen_sbp = aline_sbp if use_art else nibp_sbp
    times = np.array(sorted(chosen_map), dtype=float)
    maps = np.array([chosen_map[t] for t in times], dtype=float)
    sbps = np.array([chosen_sbp.get(t, np.nan) for t in times], dtype=float)
    return times, maps, sbps, use_art


def _extract_hr_series(
    rows: list[tuple[datetime, str, str]],
    t_ind: datetime,
) -> tuple[np.ndarray, np.ndarray]:
    hr_by_t: dict[float, float] = {}
    for ts, name, val in rows:
        t_s = _sec_from(ts, t_ind)
        if not (-900.0 <= t_s <= 1200.0):
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if 30.0 <= v <= 180.0 and (t_s not in hr_by_t or name == "Heart Rate"):
            hr_by_t[t_s] = v
    times = np.array(sorted(hr_by_t), dtype=float)
    vals = np.array([hr_by_t[t] for t in times], dtype=float)
    return times, vals


def _extract_gas_series(
    rows: list[tuple[datetime, str, str]],
    t_ind: datetime,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return `(t_s, sev_in_pct, max_other_vol_pct, max_n2o_pct)` over `[-300, +900] s`."""
    sev_in: dict[float, float] = {}
    max_other = 0.0
    max_n2o = 0.0
    for ts, name, val in rows:
        t_s = _sec_from(ts, t_ind)
        if not (-300.0 <= t_s <= 900.0):
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if not np.isfinite(v) or v < 0:
            continue
        if name == "SEV In" and v <= 10.0:
            sev_in[t_s] = v
        elif name in ("ISO In", "DES In") and v <= 20.0:
            max_other = max(max_other, v)
        elif name in ("ETN2O", "FIN20 (%)") and v <= 100.0:
            max_n2o = max(max_n2o, v)
    times = np.array(sorted(sev_in), dtype=float)
    vals = np.array([sev_in[t] for t in times], dtype=float)
    return times, vals, max_other, max_n2o


def _piecewise_rate(
    events: list[tuple[float, float]],
    grid: np.ndarray,
) -> np.ndarray:
    """Build piecewise-constant rate array on `grid` from sorted `(t_s, rate)` transitions."""
    out = np.zeros(len(grid), dtype=float)
    if not events:
        return out
    events = sorted(events, key=lambda x: x[0])
    for idx, (t_ev, rate) in enumerate(events):
        t_next = events[idx + 1][0] if idx + 1 < len(events) else np.inf
        mask = (grid >= t_ev) & (grid < t_next)
        out[mask] = rate
    return out


def _simulate_case(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Worker: run headless frozen-parameter AnaSim induction replay for one MOVER case."""
    row = payload["row"]
    arm = payload["arm"]
    meds = payload["meds"]
    bp_t = np.asarray(payload["bp_t"], dtype=float)
    bp_map = np.asarray(payload["bp_map"], dtype=float)
    bp_sbp = np.asarray(payload["bp_sbp"], dtype=float)
    is_aline = bool(payload["is_aline"])
    hr_t = np.asarray(payload["hr_t"], dtype=float)
    hr_v = np.asarray(payload["hr_v"], dtype=float)
    sev_t = np.asarray(payload["sev_t"], dtype=float)
    sev_in = np.asarray(payload["sev_in"], dtype=float)
    t_intub_s = float(payload["t_intub_s"])
    t_eval_end = float(payload["t_eval_end"])
    first_pressor_s = payload["first_pressor_s"]
    baseline_map = float(payload["baseline_map"])
    baseline_hr = float(payload["baseline_hr"])
    weight_kg = float(row["weight_kg"])

    # Determine simulation start to include pre-op midazolam/fentanyl (up to 60 min prior).
    med_boluses: list[tuple[float, str, float]] = []
    prop_inf_ev: list[tuple[float, float]] = []
    remi_inf_ev: list[tuple[float, float]] = []
    earliest_s = -120.0

    for m in meds:
        t_s = float(m["t_s"])
        if t_s > t_eval_end + 30.0 or t_s < -3600.0:
            continue
        drug = m["drug"]
        kind = m["kind"]
        val = float(m["val"])
        if kind == "bolus" and drug in _BOLUS_MAP:
            # Cap implausible charting outliers
            if drug == "propofol" and val > 500.0:
                return None
            if drug == "fentanyl" and val > 1000.0:
                return None
            if drug == "midazolam" and val > 15.0:
                return None
            if drug == "etomidate" and val > 60.0:
                return None
            med_boluses.append((t_s, _BOLUS_MAP[drug], val))
            if t_s < earliest_s:
                earliest_s = t_s - 60.0
        elif kind == "infusion":
            if drug == "propofol":
                if m["unit"] == "mcg/kg/min":
                    rate_mg_s = val * weight_kg / 60000.0
                elif m["unit"] == "mcg/min":
                    rate_mg_s = val / 60000.0
                else:
                    rate_mg_s = 0.0
                if rate_mg_s > 2.5:
                    return None
                prop_inf_ev.append((t_s, rate_mg_s))
                if t_s < earliest_s:
                    earliest_s = t_s - 60.0
            elif drug == "remifentanil":
                if m["unit"] == "mcg/kg/min":
                    rate_ug_s = val * weight_kg / 60.0
                elif m["unit"] == "mcg/min":
                    rate_ug_s = val / 60.0
                else:
                    rate_ug_s = 0.0
                if rate_ug_s > 5.0:
                    return None
                remi_inf_ev.append((t_s, rate_ug_s))
                if t_s < earliest_s:
                    earliest_s = t_s - 60.0

    earliest_s = max(-3600.0, np.floor(earliest_s / 5.0) * 5.0)
    t_stop = float(np.ceil(t_eval_end / 5.0) * 5.0 + 10.0)
    # Use 5-s steps during quiet pre-op equilibration (< -60 s) and 1-s steps during induction.
    pre_grid = np.arange(earliest_s, -60.0, 5.0) if earliest_s < -60.0 else np.array([], dtype=float)
    ind_grid = np.arange(-60.0, t_stop + 1.0, 1.0)
    grid = np.concatenate([pre_grid, ind_grid]) if len(pre_grid) else ind_grid

    prop_rate = _piecewise_rate(prop_inf_ev, grid)
    remi_rate = _piecewise_rate(remi_inf_ev, grid)

    # Interpolate/hold 1-min SEV In onto grid (0 before first positive reading).
    if len(sev_t) >= 2 and arm != "tiva":
        fi_sevo_pct = np.interp(grid, sev_t, sev_in, left=0.0, right=float(sev_in[-1]))
    else:
        fi_sevo_pct = np.zeros_like(grid)

    has_opioid = any(d in ("Fentanyl", "Remifentanil", "Hydromorphone", "Morphine") for _, d, _ in med_boluses) or bool(remi_inf_ev)
    t_pre_bp_s = float(payload.get("t_pre_bp_s", -60.0))

    def _run_sim(b_map: float, b_hr: float, stop_at_s: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        pat = patient_from_mover(row, baseline_map=b_map, baseline_hr=b_hr, use_preop_labs=True)
        eng = SimulationEngine(
            pat,
            SimulationConfig(
                mode="awake",
                concomitant_opioids=has_opioid,
                rng_seed=1,
                maintenance_fluid_ml_hr=150.0,
            ),
        )
        eng.start()
        eng.set_airway_mode("Mask")
        eng.set_fgf(6.0, 0.0)

        fi_state = {"now": 0.0}
        if arm != "tiva":
            pk_preview = eng.pk_sevo.preview_exchange
            pk_step = eng.pk_sevo.step

            def sevo_preview(dt: float, _fi: float, va: float, co: float, *a: Any, **kw: Any) -> Any:
                if "circuit_fi_agent" in kw:
                    kw["circuit_fi_agent"] = fi_state["now"]
                return pk_preview(dt, fi_state["now"], va, co, *a, **kw)

            def sevo_step(dt: float, _fi: float, va: float, co: float, **kw: Any) -> Any:
                return pk_step(dt, fi_state["now"], va, co, **kw)

            eng.pk_sevo.preview_exchange = sevo_preview  # type: ignore[assignment,method-assign]
            eng.pk_sevo.step = sevo_step  # type: ignore[assignment,method-assign]

        vt = min(0.6, 0.008 * pat.predicted_body_weight())
        pending = sorted(med_boluses, key=lambda x: x[0])
        intubated = False
        m_arr = np.empty(len(grid), dtype=float)
        h_arr = np.empty(len(grid), dtype=float)
        m_arr[0] = eng.state.map
        h_arr[0] = eng.state.hr

        for i, dt in enumerate(np.diff(grid), start=1):
            now = float(grid[i - 1])
            while pending and pending[0][0] <= now:
                _, generic_name, dose = pending.pop(0)
                eng.give_drug_bolus(generic_name, dose)
            if not intubated and now >= t_intub_s:
                eng.set_airway_mode("ETT")
                eng.set_vent_settings(rr=12, vt=vt, peep=5.0, ie="1:2", mode="VCV")
                eng.set_vent_power(True)
                eng.set_fgf(1.0, 1.0)
                eng.start_disturbance("stim_intubation_pulse")
                intubated = True
            fi_state["now"] = float(fi_sevo_pct[i - 1]) / 100.0
            eng.infusion_rates["propofol"] = float(prop_rate[i - 1])
            eng.infusion_rates["remi"] = float(remi_rate[i - 1])
            step_headless(eng, float(dt))
            m_arr[i] = eng.state.map
            h_arr[i] = eng.state.hr
            if stop_at_s is not None and grid[i] >= stop_at_s:
                m_arr[i + 1 :] = m_arr[i]
                h_arr[i + 1 :] = h_arr[i]
                break
        return m_arr, h_arr

    # Anchor pre-induction baseline at the actual time the pre-induction cuff reading was taken
    # so holding-area midazolam/fentanyl given earlier is not double-counted against `baseline_map`.
    b_map_use, b_hr_use = baseline_map, baseline_hr
    if any(t_b < t_pre_bp_s for t_b, _, _ in med_boluses):
        pre_m, pre_h = _run_sim(baseline_map, baseline_hr, stop_at_s=t_pre_bp_s)
        m_at_pre = float(np.interp(t_pre_bp_s, grid, pre_m))
        h_at_pre = float(np.interp(t_pre_bp_s, grid, pre_h))
        if abs(baseline_map - m_at_pre) > 1.0 or abs(baseline_hr - h_at_pre) > 1.0:
            b_map_use = float(np.clip(baseline_map + (baseline_map - m_at_pre) * 1.10, 55.0, 175.0))
            b_hr_use = float(np.clip(baseline_hr + (baseline_hr - h_at_pre) * 1.05, 40.0, 145.0))

    mod_map, mod_hr = _run_sim(b_map_use, b_hr_use)

    # Evaluate readings in [60 s, t_eval_end] (before any pressor if censored).
    eval_mask = (bp_t >= 60.0) & (bp_t <= t_eval_end)
    if first_pressor_s is not None:
        eval_mask &= bp_t < float(first_pressor_s)
    if eval_mask.sum() < 2:
        return None
    t_eval = bp_t[eval_mask]
    rec_map = bp_map[eval_mask]
    rec_sbp = bp_sbp[eval_mask]

    if is_aline:
        pred_map = np.interp(t_eval[:, None] + np.linspace(-10.0, 0.0, 5), grid, mod_map).mean(axis=1)
    else:
        pred_map = np.interp(t_eval[:, None] + np.linspace(-CUFF_WINDOW_S, 0.0, 9), grid, mod_map).mean(axis=1)

    err_map = pred_map - rec_map

    # VitalDB-equivalent stop rule censoring (first reading with MAP<65, SBP<90, or HR<45).
    rec_hr_at_bp = (
        np.interp(t_eval, hr_t, hr_v, left=np.nan, right=np.nan)
        if len(hr_t) >= 2
        else np.full_like(t_eval, np.nan)
    )
    trigger = np.flatnonzero(
        (rec_map < STOP[0])
        | (np.isfinite(rec_sbp) & (rec_sbp < STOP[1]))
        | (np.isfinite(rec_hr_at_bp) & (rec_hr_at_bp < STOP[2]))
    )
    stop_end = int(trigger[0]) + 1 if len(trigger) else len(err_map)
    err_map_stopped = err_map[:stop_end]

    # HR evaluation over 1-min HR samples in [60 s, t_eval_end] before any pressor.
    hr_mask = (hr_t >= 60.0) & (hr_t <= t_eval_end)
    if first_pressor_s is not None:
        hr_mask &= hr_t < float(first_pressor_s)
    err_hr = np.interp(hr_t[hr_mask], grid, mod_hr) - hr_v[hr_mask] if hr_mask.sum() >= 2 else np.array([])

    # Time-course relative MAP changes (%) and window mean errors.
    rec_pct = 100.0 * (rec_map / baseline_map - 1.0)
    mod_pct = 100.0 * (pred_map / baseline_map - 1.0)
    course: dict[str, tuple[float, float]] = {}
    window_err: dict[str, float] = {}
    for label, lo, hi in (
        ("1-2 min", 60.0, 120.0),
        ("2-5 min", 120.0, 300.0),
        ("5-10 min", 300.0, 600.0),
    ):
        m = (t_eval >= lo) & (t_eval <= hi)
        if m.any():
            course[label] = (float(rec_pct[m].mean()), float(mod_pct[m].mean()))
            window_err[label] = float(err_map[m].mean())
    course["lowest in 10 min"] = (float(rec_pct.min()), float(mod_pct.min()))
    post_intub = (t_eval >= t_intub_s) & (t_eval <= t_intub_s + 180.0)
    if post_intub.any():
        course["post-intubation peak"] = (float(rec_pct[post_intub].max()), float(mod_pct[post_intub].max()))
        window_err["post-intubation peak"] = float(pred_map[post_intub].max() - rec_map[post_intub].max())

    t_ready_s = float(payload["t_ready_s"])
    pre_ready = t_eval <= t_ready_s
    pre_ready_err = float(err_map[pre_ready].mean()) if pre_ready.any() else float("nan")
    has_midaz = any(d == "Midazolam" for _, d, _ in med_boluses)

    return {
        "log_id": row["log_id"],
        "arm": arm,
        "pressor_free": first_pressor_s is None,
        "has_midaz": has_midaz,
        "age": float(row["age"]),
        "bmi": float(row["bmi"]),
        "asa": int(row["asa"]),
        "htn": int(row["htn"]),
        "is_aline": is_aline,
        "baseline_map": baseline_map,
        "baseline_hr": baseline_hr,
        "n_bp": len(err_map),
        "mean_map_err": float(err_map.mean()),
        "mean_map_err_stopped": float(err_map_stopped.mean()),
        "mean_map_err_pre_ready": pre_ready_err,
        "mean_hr_err": float(err_hr.mean()) if len(err_hr) else float("nan"),
        "course": course,
        "window_err": window_err,
    }


def _arm_summary_lines(title: str, results: list[dict[str, Any]]) -> list[str]:
    if not results:
        return [f"## {title}", "", "No eligible cases retained.", ""]
    lines = [
        f"## {title} (n={len(results)})",
        "",
        "| Subgroup | n | MAP err, all (mmHg) | MAP err, stop-rule (mmHg) | HR err (bpm) |",
        "|---|---:|---|---|---|",
    ]

    def add_row(label: str, sub: list[dict[str, Any]]) -> None:
        if len(sub) < 3:
            return
        m_all = fmt_mean_ci(mean_ci([r["mean_map_err"] for r in sub]))
        m_stp = fmt_mean_ci(mean_ci([r["mean_map_err_stopped"] for r in sub]))
        h_err = fmt_mean_ci(mean_ci([r["mean_hr_err"] for r in sub if np.isfinite(r["mean_hr_err"])]))
        lines.append(f"| {label} | {len(sub)} | {m_all} | {m_stp} | {h_err} |")

    add_row("All", results)
    add_row("Age under 45", [r for r in results if r["age"] < 45])
    add_row("Age 45-59", [r for r in results if 45 <= r["age"] < 60])
    add_row("Age 60-69", [r for r in results if 60 <= r["age"] < 70])
    add_row("Age 70 and over", [r for r in results if r["age"] >= 70])
    add_row("BMI under 25", [r for r in results if r["bmi"] < 25])
    add_row("BMI 25-29.9", [r for r in results if 25 <= r["bmi"] < 30])
    add_row("BMI 30-34.9", [r for r in results if 30 <= r["bmi"] < 35])
    add_row("BMI 35 and over", [r for r in results if r["bmi"] >= 35])
    add_row("Normotensive (no HTN dx)", [r for r in results if not r["htn"]])
    add_row("Hypertensive (HTN dx)", [r for r in results if r["htn"]])
    add_row("ASA 1-2", [r for r in results if r["asa"] <= 2])
    add_row("ASA 3+", [r for r in results if r["asa"] >= 3])
    add_row("No pre-op midazolam", [r for r in results if not r["has_midaz"]])
    add_row("With pre-op midazolam", [r for r in results if r["has_midaz"]])
    add_row("Invasive A-line", [r for r in results if r["is_aline"]])
    add_row("Non-invasive NIBP", [r for r in results if not r["is_aline"]])

    pre_r_vals = [r["mean_map_err_pre_ready"] for r in results if np.isfinite(r["mean_map_err_pre_ready"])]
    if len(pre_r_vals) >= 3:
        lines.extend(
            [
                "",
                f"Strictly before `Anesthesia Ready` (n={len(pre_r_vals)}): MAP error "
                f"{fmt_mean_ci(mean_ci(pre_r_vals))} mmHg.",
            ]
        )

    lines.extend(
        [
            "",
            "Time-course change in MAP from pre-induction (median, IQR) and window MAP error (model minus recorded):",
            "",
            "| Window | n | Recorded dMAP (%) | Model dMAP (%) | MAP error (mmHg, 95% CI) |",
            "|---|---:|---|---|---|",
        ]
    )
    for key in ("1-2 min", "2-5 min", "5-10 min", "lowest in 10 min", "post-intubation peak"):
        pairs = [r["course"][key] for r in results if key in r["course"]]
        errs = [r["window_err"][key] for r in results if key in r["window_err"]]
        if not pairs:
            continue
        arr = np.asarray(pairs, dtype=float)
        rec_s = fmt_med_iqr(median_iqr(arr[:, 0]), digits=0, suffix="%")
        mod_s = fmt_med_iqr(median_iqr(arr[:, 1]), digits=0, suffix="%")
        err_s = fmt_mean_ci(mean_ci(errs)) if errs else "-"
        lines.append(f"| {key} | {len(pairs)} | {rec_s} | {mod_s} | {err_s} |")
    lines.append("")
    return lines


def run(
    *,
    mover_dir: Path = DEFAULT_MOVER_DIR,
    cache_dir: Path = CACHE_DIR,
    output_dir: Path = RESULTS_DIR,
    sevo_cases: int = 400,
    sevo_treated_cases: int = 200,
    tiva_cases: int = 300,
    workers: int = DEFAULT_WORKERS,
    seed: int = 42,
) -> dict[str, Any]:
    cohort = load_cohort(
        mover_dir,
        cache_dir,
        where="has_cpb = 0 AND has_olv = 0 AND has_lma = 0 AND t_induction IS NOT NULL AND t_intubation IS NOT NULL",
    )
    eligible: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for row in cohort:
        dt_intub = _sec_from(row["t_intubation"], row["t_induction"])
        if not (0.0 <= dt_intub <= 600.0):
            exclusions["intubation outside 0-10 min of induction"] += 1
            continue
        eligible.append(row)

    meds_by_log = load_meds_for_logs([r["log_id"] for r in eligible], mover_dir, cache_dir)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))
    sevo_free_cands: list[tuple[dict[str, Any], list[dict[str, Any]], float | None]] = []
    sevo_treated_cands: list[tuple[dict[str, Any], list[dict[str, Any]], float | None]] = []
    tiva_cands: list[tuple[dict[str, Any], list[dict[str, Any]], float | None]] = []
    etom_cands: list[tuple[dict[str, Any], list[dict[str, Any]], float | None]] = []

    for idx in order:
        row = eligible[int(idx)]
        t_ind = row["t_induction"]
        raw_meds = meds_by_log.get(row["log_id"], [])
        rel_meds = [{**m, "t_s": _sec_from(m["ts"], t_ind)} for m in raw_meds]
        win_meds = [m for m in rel_meds if -900.0 <= m["t_s"] <= 660.0]

        if any(m["drug"] in _EXCLUDED_NON_VASO for m in win_meds):
            exclusions["ketamine or dexmedetomidine during induction"] += 1
            continue

        # Check earliest vasoactive drug in [-15 min, +10 min].
        pressor_times = [
            m["t_s"]
            for m in win_meds
            if m["drug"] in VASOACTIVE_DRUGS and (m["kind"] == "bolus" or m["val"] > 0)
        ]
        if any(t < 60.0 for t in pressor_times):
            exclusions["vasoactive drug before +1 min"] += 1
            continue
        first_pressor = min(pressor_times) if pressor_times else None

        prop_bolus = sum(
            m["val"] for m in rel_meds if m["drug"] == "propofol" and m["kind"] == "bolus" and -120.0 <= m["t_s"] <= 120.0
        )
        etom_bolus = sum(
            m["val"] for m in rel_meds if m["drug"] == "etomidate" and m["kind"] == "bolus" and -120.0 <= m["t_s"] <= 120.0
        )
        has_prop_inf = any(
            m["drug"] == "propofol" and m["kind"] == "infusion" and m["val"] > 0 and -120.0 <= m["t_s"] <= 300.0
            for m in rel_meds
        )
        has_remi_inf = any(
            m["drug"] == "remifentanil" and m["kind"] == "infusion" and m["val"] > 0 and -120.0 <= m["t_s"] <= 300.0
            for m in rel_meds
        )
        wt = float(row["weight_kg"])

        if etom_bolus > 0 and prop_bolus == 0 and 0.1 * wt <= etom_bolus <= 0.6 * wt and not has_prop_inf:
            etom_cands.append((row, rel_meds, first_pressor))
        elif row["asa"] <= 3 and has_prop_inf and has_remi_inf:
            tiva_cands.append((row, rel_meds, first_pressor))
        elif row["asa"] <= 3 and 0.5 * wt <= prop_bolus <= 4.0 * wt and not has_prop_inf and not has_remi_inf:
            if first_pressor is None:
                if len(sevo_free_cands) < sevo_cases * 2:
                    sevo_free_cands.append((row, rel_meds, None))
            else:
                if len(sevo_treated_cands) < sevo_treated_cases * 2:
                    sevo_treated_cands.append((row, rel_meds, first_pressor))

    tiva_cands = tiva_cands[: tiva_cases * 2]
    all_cands: list[tuple[str, dict[str, Any], list[dict[str, Any]], float | None]] = (
        [("sevo_free", r, m, p) for r, m, p in sevo_free_cands]
        + [("sevo_treated", r, m, p) for r, m, p in sevo_treated_cands]
        + [("tiva", r, m, p) for r, m, p in tiva_cands]
        + [("etomidate", r, m, p) for r, m, p in etom_cands]
    )
    cand_ids = [r["log_id"] for _, r, _, _ in all_cands]

    bp_rows = load_flowsheet_subset("BP", cand_ids, _BP_NAMES, mover_dir=mover_dir)
    hr_rows = load_flowsheet_subset("HR", cand_ids, _HR_NAMES, mover_dir=mover_dir)
    gas_rows = load_flowsheet_subset("Gas", cand_ids, _SEVO_NAMES, mover_dir=mover_dir)

    tasks: list[dict[str, Any]] = []
    arm_caps = {
        "sevo_free": sevo_cases,
        "sevo_treated": sevo_treated_cases,
        "tiva": tiva_cases,
        "etomidate": 1000,
    }
    arm_counts: Counter[str] = Counter()

    for arm, row, rel_meds, first_pressor in all_cands:
        if arm_counts[arm] >= arm_caps[arm]:
            continue
        lid = row["log_id"]
        t_ind = row["t_induction"]
        sev_t, sev_in, max_other, max_n2o = _extract_gas_series(gas_rows.get(lid, []), t_ind)
        if max_other >= 0.2 or max_n2o >= 5.0:
            exclusions["isoflurane, desflurane, or nitrous oxide exposure"] += 1
            continue
        if arm == "tiva" and len(sev_in) and float(sev_in.max()) >= 0.2:
            exclusions["volatile exposure in TIVA candidate"] += 1
            continue
        if arm in ("sevo_free", "sevo_treated") and (len(sev_in) < 3 or float(sev_in.max()) < 0.4):
            exclusions["no inspired sevoflurane recorded"] += 1
            continue

        bp_t, bp_map, bp_sbp, is_aline = _extract_bp_series(bp_rows.get(lid, []), t_ind)
        hr_t, hr_v = _extract_hr_series(hr_rows.get(lid, []), t_ind)
        pre_bp = (bp_t >= -900.0) & (bp_t < 0.0) & (bp_map >= 50.0) & (bp_map <= 150.0)
        pre_hr = (hr_t >= -300.0) & (hr_t < 0.0) & (hr_v >= 40.0) & (hr_v <= 140.0)
        if not pre_bp.any():
            exclusions["no pre-induction MAP within 15 min"] += 1
            continue
        if pre_hr.sum() < 1:
            exclusions["no pre-induction HR within 5 min"] += 1
            continue
        baseline_map = float(bp_map[pre_bp][-1])
        t_pre_bp_s = float(bp_t[pre_bp][-1])
        baseline_hr = float(np.median(hr_v[pre_hr]))

        t_intub_s = _sec_from(row["t_intubation"], t_ind)
        t_ready_s = _sec_from(row["t_ready"], t_ind) if row["t_ready"] is not None else 600.0
        t_eval_end = min(600.0, max(300.0, t_ready_s + 180.0))

        tasks.append(
            {
                "row": row,
                "arm": arm,
                "meds": rel_meds,
                "bp_t": bp_t.tolist(),
                "bp_map": bp_map.tolist(),
                "bp_sbp": bp_sbp.tolist(),
                "is_aline": is_aline,
                "hr_t": hr_t.tolist(),
                "hr_v": hr_v.tolist(),
                "sev_t": sev_t.tolist(),
                "sev_in": sev_in.tolist(),
                "t_intub_s": t_intub_s,
                "t_ready_s": t_ready_s,
                "t_pre_bp_s": t_pre_bp_s,
                "t_eval_end": t_eval_end,
                "first_pressor_s": first_pressor,
                "baseline_map": baseline_map,
                "baseline_hr": baseline_hr,
            }
        )
        arm_counts[arm] += 1

    results: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_simulate_case, tasks, chunksize=4):
                if res is not None:
                    results.append(res)
    else:
        for t in tasks:
            res = _simulate_case(t)
            if res is not None:
                results.append(res)

    by_arm: dict[str, list[dict[str, Any]]] = {
        "sevo_free": [r for r in results if r["arm"] == "sevo_free"],
        "sevo_treated": [r for r in results if r["arm"] == "sevo_treated"],
        "tiva": [r for r in results if r["arm"] == "tiva"],
        "etomidate": [r for r in results if r["arm"] == "etomidate"],
    }
    sevo_combined = by_arm["sevo_free"] + by_arm["sevo_treated"]

    lines = [
        "# MOVER external validation: induction hemodynamics",
        "",
        f"Generated {utc_date()}. Frozen AnaSim parameters (no refitting on MOVER).",
        "",
        "## Methods",
        "",
        "Replays unique-patient adult General Anesthesia encounters (age 18-85, weight 50-200 kg,",
        "height 150-200 cm, BMI 18-60 kg/m², no CPB, OLV, LMA, ketamine, dexmedetomidine, isoflurane,",
        "desflurane, or N₂O) using exact MAR bolus/infusion timestamps (including pre-op midazolam),",
        "recorded `Induction` and `Intubation` events, 1-minute `SEV In`, pre-op hemoglobin and",
        "Cockcroft-Gault CrCl, and 1-minute arterial line or NIBP MAP and ECG HR. Errors are",
        "model minus recorded over `+1` to `+10 min` post-induction (capped at `Anesthesia Ready + 3 min`",
        "and censored before any first vasoactive drug).",
        "",
        f"Screened unique-MRN GA encounters with Induction + Intubation: {len(eligible)}. "
        f"Evaluated: {len(results)} (Sevo pressor-free: {len(by_arm['sevo_free'])}, "
        f"Sevo pressor-censored: {len(by_arm['sevo_treated'])}, TIVA: {len(by_arm['tiva'])}, "
        f"Etomidate: {len(by_arm['etomidate'])}).",
        "",
    ]
    lines.extend(_arm_summary_lines("Propofol + Sevoflurane induction (strictly pressor-free)", by_arm["sevo_free"]))
    lines.extend(
        _arm_summary_lines(
            "Propofol + Sevoflurane induction (all: pressor-free + censored at first pressor)",
            sevo_combined,
        )
    )
    lines.extend(_arm_summary_lines("TIVA (Propofol + Remifentanil infusion) induction", by_arm["tiva"]))
    lines.extend(_arm_summary_lines("Etomidate bolus induction (ASA 1-4)", by_arm["etomidate"]))
    lines.extend(["## Exclusions during candidate screening", ""])
    for reason, cnt in sorted(exclusions.items()):
        lines.append(f"- {reason}: {cnt}")
    lines.extend(["", SOURCE])

    summary_payload = {
        "generated": utc_date(),
        "counts": {k: len(v) for k, v in by_arm.items()},
        "arms": {
            name: {
                "n": len(sub),
                "map_err_all": mean_ci([r["mean_map_err"] for r in sub]),
                "map_err_stopped": mean_ci([r["mean_map_err_stopped"] for r in sub]),
                "hr_err": mean_ci([r["mean_hr_err"] for r in sub if np.isfinite(r["mean_hr_err"])]),
            }
            for name, sub in (
                ("sevo_free", by_arm["sevo_free"]),
                ("sevo_combined", sevo_combined),
                ("tiva", by_arm["tiva"]),
                ("etomidate", by_arm["etomidate"]),
            )
            if sub
        },
        "exclusions": dict(exclusions),
    }
    write_results("mover_induction", lines, summary_payload, output_dir=output_dir)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="MOVER external induction hemodynamics validation.")
    parser.add_argument("--mover-dir", type=Path, default=DEFAULT_MOVER_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--sevo-cases", type=int, default=400)
    parser.add_argument("--sevo-treated-cases", type=int, default=200)
    parser.add_argument("--tiva-cases", type=int, default=300)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = run(
        mover_dir=args.mover_dir,
        cache_dir=args.cache,
        output_dir=args.output,
        sevo_cases=args.sevo_cases,
        sevo_treated_cases=args.sevo_treated_cases,
        tiva_cases=args.tiva_cases,
        workers=args.workers,
        seed=args.seed,
    )
    print(f"Wrote mover_induction report: counts={out['counts']}")


if __name__ == "__main__":
    main()
