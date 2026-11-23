"""Shared VitalDB delivery validation, engine replay, cuff scoring, and CV helpers."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.optimize import minimize

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology import hemodynamics
from validation.vitaldb.vitaldb_data import (
    REMI_RATES,
    REMI_UG_PER_ML,
    SOURCE,
    VASOACTIVE,
    Excluded,
    amount,
    covered,
    held,
    kfold_splits,
    numeric_track,
)

SAMPLE_SECONDS = 5.0
CUFF_WINDOW_S = 40.0  # An oscillometric reading posts at the end of cuff deflation
STOP = (65.0, 90.0, 45.0)  # MAP, SBP, and HR below which treatment could follow
SENSITIVITY = (70.0, 100.0, 45.0)
PERIODS = ((0.0, 600.0), (600.0, np.inf))  # Seconds after propofol onset
BOOTSTRAP = 500
COURSE = (("1-2 min", 60.0, 120.0), ("2-5 min", 120.0, 300.0))

__all__ = [
    "BOOTSTRAP",
    "COURSE",
    "CUFF_WINDOW_S",
    "PERIODS",
    "REMI_RATES",
    "REMI_UG_PER_ML",
    "SAMPLE_SECONDS",
    "SENSITIVITY",
    "SOURCE",
    "STOP",
    "VASOACTIVE",
    "amount",
    "apply_hemo_override",
    "course",
    "course_table",
    "cuff_model",
    "cv_grid_lines",
    "cv_hemodynamic_grid",
    "cv_hemodynamic_stability",
    "cv_stability_lines",
    "delivery",
    "estimate",
    "expand_override_grid",
    "fit",
    "hemo_subgroup_rows",
    "mean_ci",
    "pre_induction",
    "procedure",
    "readings",
    "report_footer",
    "report_header",
    "rule_stats",
    "run_hemodynamic_replay",
    "treated",
    "window_medians",
]


def treated(case: dict[str, Any]) -> bool:
    return amount(case, "intraop_eph") > 0 or amount(case, "intraop_phe") > 0


def procedure(case: dict[str, Any]) -> str:
    if case.get("position") != "Supine":
        return "Other positions"
    return "Open, supine" if case.get("approach") == "Open" else "Laparoscopic or robotic, supine"


def delivery(
    tracks: dict[str, str],
    *,
    max_gap_s: float = 30.0,
    require_gas_before_onset: bool = False,
    track_loader: Callable[[str], tuple[np.ndarray, np.ndarray]] = numeric_track,
):
    """Validated propofol and remifentanil pump histories.

    Rates are held across gaps up to ``max_gap_s``, and a volume counter that restarts
    after a syringe change continues from its previous total.
    """
    rate, volume = track_loader(tracks["Orchestra/PPF20_RATE"]), track_loader(tracks["Orchestra/PPF20_VOL"])
    if volume[1][0] > 0.05 or volume[1][0] < 0 or rate[1][0] != 0:
        raise Excluded("propofol delivery began before recording")
    raw, total, offset = volume[1], volume[1].copy(), 0.0
    for index in range(1, len(raw)):
        if raw[index] < raw[index - 1] - 0.25:
            offset += raw[index - 1]
        total[index] = raw[index] + offset
    volume = (volume[0], total)
    if np.any((rate[1] < 0) | (rate[1] > 1500)) or volume[1][-1] < 5:
        raise Excluded("invalid or insufficient propofol delivery")
    active = np.flatnonzero(rate[1] > 0.1)
    if not len(active):
        raise Excluded("no recorded propofol infusion")
    onset = float(rate[0][active[0]])
    stop = float(rate[0][active[-1] + 1]) if active[-1] + 1 < len(rate[0]) else float(rate[0][-1])
    if onset - rate[0][0] < 30:
        raise Excluded("less than thirty seconds of zero delivery before induction")
    remi_name = next(name for name in REMI_RATES if name in tracks)
    remi = track_loader(tracks[remi_name])
    if np.any((remi[1] < 0) | (remi[1] > 1500)) or np.max(remi[1]) <= 0:
        raise Excluded("invalid or absent remifentanil delivery")
    gases = {name: track_loader(tracks[name]) for name in ("Primus/MAC", "Primus/FIN2O")}
    if np.max(gases["Primus/MAC"][1]) > 0.05 or np.max(gases["Primus/FIN2O"][1]) > 1:
        raise Excluded("recorded volatile or nitrous oxide exposure")
    start = max(rate[0][0], volume[0][0], remi[0][0])
    if onset - start < 30:
        raise Excluded("delivery history starts too late for induction")
    end = min(rate[0][-1], volume[0][-1], remi[0][-1])
    if require_gas_before_onset:
        end = min(end, gases["Primus/MAC"][0][-1], gases["Primus/FIN2O"][0][-1])
        for name in ("Primus/MAC", "Primus/FIN2O"):
            if gases[name][0][0] > onset:
                raise Excluded("gas monitoring began after propofol induction")
            covered(gases[name], onset, end, 30.0)
    for series in (rate, volume, remi):
        covered(series, start, end, max_gap_s)
    return rate, volume, remi, remi_name, start, end, onset, stop


def pre_induction(
    tracks: dict[str, str],
    onset: float,
    *,
    track_loader: Callable[[str], tuple[np.ndarray, np.ndarray]] = numeric_track,
) -> tuple[float, float]:
    """Last cuff MAP within 15 min before propofol, and median HR in the preceding 2 min."""
    nibp, hr = track_loader(tracks["Solar8000/NIBP_MBP"]), track_loader(tracks["Solar8000/HR"])
    sel = (nibp[0] >= onset - 900) & (nibp[0] < onset) & (nibp[1] >= 50) & (nibp[1] <= 150)
    if not sel.any():
        raise Excluded("no cuff MAP within 15 minutes before induction")
    rates = hr[1][(hr[0] >= onset - 120) & (hr[0] < onset) & (hr[1] >= 40) & (hr[1] <= 140)]
    if len(rates) < 5:
        raise Excluded("insufficient HR before induction")
    return float(nibp[1][sel][-1]), float(np.median(rates))


def readings(
    tracks: dict[str, str],
    start: float,
    end: float,
    *,
    track_loader: Callable[[str], tuple[np.ndarray, np.ndarray]] = numeric_track,
):
    """Cuff readings in [start, end); the held 2-s MAP series changes value at each new reading."""
    time, value = track_loader(tracks["Solar8000/NIBP_MBP"])
    sel = np.r_[False, np.diff(value) != 0] & (time >= start) & (time < end) & (value > 30) & (value < 160)
    times = time[sel]
    sbp = held(track_loader(tracks["Solar8000/NIBP_SBP"]), times + 2.0, 10.0)
    hr = held(track_loader(tracks["Solar8000/HR"]), times, 30.0)
    return times, value[sel], sbp, hr


def cuff_model(times: np.ndarray, grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Model values averaged over the measurement window before each cuff reading posts."""
    return np.interp(times[:, None] + np.linspace(-CUFF_WINDOW_S, 0.0, 9), grid, values).mean(axis=1)


def rule_stats(errors, seconds, cuff_map, cuff_sbp, cuff_hr, limits):
    """Error counts, sums, and squares by period, ending at the first reading that could prompt treatment."""
    # NaN SBP or HR never triggers.
    trigger = np.flatnonzero((cuff_map < limits[0]) | (cuff_sbp < limits[1]) | (cuff_hr < limits[2]))
    end = trigger[0] + 1 if len(trigger) else len(errors)
    periods = [errors[:end][(seconds[:end] >= low) & (seconds[:end] < high)] for low, high in PERIODS]
    return dict(
        stopped=bool(len(trigger)),
        stats=[[len(part), float(part.sum()), float(np.square(part).sum())] for part in periods],
    )


def course(seconds, cuff_map, model_map, baseline_map, incision_s):
    """Recorded and model change from pre-induction MAP (%) at standard points, without stopping."""
    recorded, model = 100.0 * (cuff_map / baseline_map - 1.0), 100.0 * (model_map / baseline_map - 1.0)
    out = {}
    for name, low, high in COURSE:
        sel = (seconds >= low) & (seconds < high)
        if sel.any():
            out[name] = (float(recorded[sel].mean()), float(model[sel].mean()))
    early = seconds < 600.0
    if early.any():
        out["lowest in 10 min"] = (float(recorded[early].min()), float(model[early].min()))
    late = seconds >= max(300.0, incision_s - 600.0)
    if late.any():
        out["10 min before incision"] = (float(recorded[late].mean()), float(model[late].mean()))
    return out


def course_table(rows, group=""):
    lines = [
        "",
        f"Change in MAP from pre-induction, all readings without stopping{group}, median (IQR):",
        "",
        "| Measure | Patients | Recorded | Model |",
        "|---|---:|---|---|",
    ]
    for name in ("1-2 min", "2-5 min", "lowest in 10 min", "10 min before incision"):
        pairs = np.array([r["course"][name] for r in rows if name in r["course"]])
        if len(pairs):
            cells = [f"{np.median(x):.0f}% ({np.percentile(x, 25):.0f} to {np.percentile(x, 75):.0f})" for x in pairs.T]
            lines.append(f"| {name} | {len(pairs)} | {cells[0]} | {cells[1]} |")
    return lines


def window_medians(
    tracks: dict[str, str],
    grid: np.ndarray,
    predicted: np.ndarray,
    t0: float,
    t1: float,
    *,
    track_loader: Callable[[str], tuple[np.ndarray, np.ndarray]] = numeric_track,
):
    """Recorded and model medians of MAP (arterial when recorded) and HR over 5-s samples in [t0, t1)."""
    pressure = "Solar8000/ART_MBP" if "Solar8000/ART_MBP" in tracks else "Solar8000/NIBP_MBP"
    sample = np.arange(t0, t1, SAMPLE_SECONDS)
    out = {}
    for column, (key, track) in enumerate((("map", pressure), ("hr", "Solar8000/HR"))):
        values = held(track_loader(tracks[track]), sample, 30.0) if len(sample) else np.array([])
        valid = np.isfinite(values) & (values > 30) & (values < 160)
        if valid.sum() * SAMPLE_SECONDS >= 300:
            out[key] = dict(
                observed=float(np.median(values[valid])),
                predicted=float(np.median(np.interp(sample[valid], grid, predicted[:, column]))),
            )
    return out if len(out) == 2 else None


def run_hemodynamic_replay(
    case: dict[str, Any],
    patient: Patient,
    grid: np.ndarray,
    *,
    onset: float,
    events: Sequence[tuple[float, Callable[[SimulationEngine], Any]]],
    propofol_mg_s: np.ndarray,
    remi_ug_s: np.ndarray,
    inspired_sevo: np.ndarray | None = None,
    end_tidal_sevo: np.ndarray | None = None,
) -> np.ndarray:
    """Drive the full AnaSim engine with recorded anesthetic delivery, fluids, blood loss, and surgery.

    Returns an array of shape ``(len(grid), 3)`` with columns ``[map, hr, et_sevo]``.
    """
    duration_h = max(1.0, (float(case["aneend"]) - float(case["anestart"])) / 3600.0)
    engine = SimulationEngine(
        patient,
        SimulationConfig(
            mode="awake",
            concomitant_opioids=True,
            rng_seed=1,
            maintenance_fluid_ml_hr=amount(case, "intraop_crystalloid") / duration_h,
        ),
    )
    engine.start()
    engine.set_airway_mode("Mask")
    engine.set_fgf(6.0, 0.0)

    fi_state = {"now": 0.0, "end_tidal": np.nan}
    if inspired_sevo is not None:
        pk_preview = engine.pk_sevo.preview_exchange
        pk_commit = engine.pk_sevo.commit_integration
        pk_step = engine.pk_sevo.step

        def sevo_preview(dt, _fi, va, co, *args, **kwargs):
            if "circuit_fi_agent" in kwargs:
                kwargs["circuit_fi_agent"] = fi_state["now"]
            return pk_preview(dt, fi_state["now"], va, co, *args, **kwargs)

        def sevo_commit(result, dt, *args, **kwargs):
            pk_commit(result, dt, *args, **kwargs)
            if np.isfinite(fi_state["end_tidal"]):
                engine.pk_sevo.state.p_alv = fi_state["end_tidal"] / 100.0

        def sevo_step(dt, _fi, va, co, **kwargs):
            state = pk_step(dt, fi_state["now"], va, co, **kwargs)
            if np.isfinite(fi_state["end_tidal"]):
                engine.pk_sevo.state.p_alv = fi_state["end_tidal"] / 100.0
            return state

        engine.pk_sevo.preview_exchange = sevo_preview  # type: ignore[assignment,method-assign]
        engine.pk_sevo.commit_integration = sevo_commit  # type: ignore[method-assign]
        engine.pk_sevo.step = sevo_step  # type: ignore[assignment,method-assign]

    colloid_ml_min = amount(case, "intraop_colloid") / duration_h / 60.0
    vt = min(0.6, 0.008 * patient.predicted_body_weight())
    pending = sorted(events, key=lambda event: event[0])
    ventilated, next_colloid = False, onset
    out = np.empty((len(grid), 3), dtype=float)
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
        if inspired_sevo is not None:
            fi_state["now"] = float(inspired_sevo[i - 1]) / 100.0
        if end_tidal_sevo is not None:
            fi_state["end_tidal"] = float(end_tidal_sevo[i - 1])
        engine.infusion_rates["propofol"] = float(propofol_mg_s[i - 1])
        engine.infusion_rates["remi"] = float(remi_ug_s[i - 1])
        engine.step(float(dt))
        out[i] = engine.state.map, engine.state.hr, engine.state.et_sevo
    return out


_ORIG_HEMO_INIT = hemodynamics.HemodynamicModel.__init__


def apply_hemo_override(override: dict[str, Any] | None) -> None:
    """Configure a process-local HemodynamicConfig override for replay sweeps."""
    if not override:
        hemodynamics.HemodynamicModel.__init__ = _ORIG_HEMO_INIT  # type: ignore[method-assign]
        return

    def init(self, patient, config=None):
        _ORIG_HEMO_INIT(self, patient, replace(config or hemodynamics.HemodynamicConfig(), **override))

    hemodynamics.HemodynamicModel.__init__ = init  # type: ignore[method-assign]


def expand_override_grid(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a JSON dict of {field: [v1, v2, ...]} into a Cartesian list of override dicts."""
    if not spec:
        return []
    keys = list(spec.keys())
    values = [v if isinstance(v, list) else [v] for v in spec.values()]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def fit(stats: np.ndarray):
    """Maximum-likelihood mean errors by period, patient SD, and reading SD.

    Errors are normal with a patient-level offset and independent reading
    noise, so a patient's covariance is compound symmetric.
    """
    count, total, squares = stats[..., 0], stats[..., 1], stats[..., 2]
    n = count.sum(axis=1)
    periods = stats.shape[1]

    def negative_log_likelihood(params):
        mean, within, between = params[:periods], np.exp(2.0 * params[-2]), np.exp(2.0 * params[-1])
        residual = (total - count * mean).sum(axis=1)
        residual_squares = (squares - 2.0 * mean * total + count * mean**2).sum(axis=1)
        quadratic = (residual_squares - between / (within + n * between) * residual**2) / within
        return 0.5 * np.sum((n - 1.0) * np.log(within) + np.log(within + n * between) + quadratic)

    start = [*(total.sum(axis=0) / np.maximum(count.sum(axis=0), 1.0)), np.log(5.0), np.log(10.0)]
    result = minimize(negative_log_likelihood, start, method="L-BFGS-B")
    return result.x[:periods], float(np.exp(result.x[-1])), float(np.exp(result.x[-2]))


def estimate(rows, rule, by_period, rng):
    """Fitted mean errors with patient-bootstrap 95% intervals."""
    stats = np.array([row[rule]["stats"] for row in rows])
    if not by_period:
        stats = stats.sum(axis=1, keepdims=True)
    means, between, within = fit(stats)
    boot = np.array([fit(stats[rng.integers(0, len(stats), len(stats))])[0] for _ in range(BOOTSTRAP)])
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return [(m, lo, hi) for m, lo, hi in zip(means, low, high, strict=True)], between, within, int(stats[..., 0].sum())


def _hemo_stratum(row: dict[str, Any]) -> tuple[bool, bool, int]:
    age_bin = 0 if row.get("age", 40.0) < 45 else (1 if row.get("age", 40.0) < 60 else (2 if row.get("age", 40.0) < 70 else 3))
    return (bool(row.get("treated")), bool(row.get("hypertensive")), age_bin)


def cv_hemodynamic_stability(rows: list[dict[str, Any]], *, n_folds: int = 5, seed: int = 20261024, rule: str = "primary") -> dict[str, Any] | None:
    """Fold-by-fold mixed-effects mean error across patient-grouped stratified folds."""
    if len(rows) < max(n_folds * 2, 10):
        return None
    splits = kfold_splits(
        rows,
        n_folds=n_folds,
        seed=seed,
        group_key=lambda r: r.get("subjectid", r["caseid"]),
        strata=_hemo_stratum,
    )
    fold_means = []
    for _, val_idx in splits:
        val_stats = np.array([rows[i][rule]["stats"] for i in val_idx]).sum(axis=1, keepdims=True)
        mean, _, _ = fit(val_stats)
        fold_means.append(float(mean[0]))
    return {
        "n_folds": n_folds,
        "fold_means": fold_means,
        "mean": float(np.mean(fold_means)),
        "sd": float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else 0.0,
        "min": float(np.min(fold_means)),
        "max": float(np.max(fold_means)),
    }


def cv_hemodynamic_grid(
    candidate_results: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    n_folds: int = 5,
    seed: int = 20261024,
    rule: str = "primary",
) -> dict[str, Any]:
    """Select the lowest-|bias| hemodynamic override inside each training fold and score out-of-fold."""
    if not candidate_results:
        raise ValueError("candidate_results must be non-empty")
    common_ids = sorted(set.intersection(*(set(r["caseid"] for r in rows) for _, rows in candidate_results)))
    indexed = [
        (override, {r["caseid"]: r for r in rows if r["caseid"] in common_ids})
        for override, rows in candidate_results
    ]
    ref_rows = [indexed[0][1][cid] for cid in common_ids]
    splits = kfold_splits(
        ref_rows,
        n_folds=n_folds,
        seed=seed,
        group_key=lambda r: r.get("subjectid", r["caseid"]),
        strata=_hemo_stratum,
    )
    oof_rows: list[dict[str, Any]] = []
    fold_choices: list[dict[str, Any]] = []
    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_ids = [ref_rows[i]["caseid"] for i in train_idx]
        val_ids = [ref_rows[i]["caseid"] for i in val_idx]
        best_score = float("inf")
        best_override: dict[str, Any] = {}
        best_map: dict[int, dict[str, Any]] = {}
        best_train_mean = 0.0
        for override, row_map in indexed:
            tr_stats = np.array([row_map[cid][rule]["stats"] for cid in train_ids]).sum(axis=1, keepdims=True)
            tr_mean = float(fit(tr_stats)[0][0])
            if abs(tr_mean) < best_score:
                best_score = abs(tr_mean)
                best_override = override
                best_map = row_map
                best_train_mean = tr_mean
        val_stats = np.array([best_map[cid][rule]["stats"] for cid in val_ids]).sum(axis=1, keepdims=True)
        val_mean = float(fit(val_stats)[0][0])
        oof_rows.extend(best_map[cid] for cid in val_ids)
        fold_choices.append(
            dict(fold=fold_num, override=best_override, train_mean=best_train_mean, val_mean=val_mean, n_val=len(val_ids))
        )
    rng = np.random.default_rng(seed)
    [(oof_mean, oof_lo, oof_hi)], oof_between, oof_within, oof_readings = estimate(oof_rows, rule, False, rng)
    full_scores: list[dict[str, Any]] = []
    for override, row_map in indexed:
        all_stats = np.array([row_map[cid][rule]["stats"] for cid in common_ids]).sum(axis=1, keepdims=True)
        mean_err = float(fit(all_stats)[0][0])
        full_scores.append(dict(override=override, mean_error=mean_err))
    best_full = min(full_scores, key=lambda s: abs(float(s["mean_error"])))
    return {
        "n_folds": n_folds,
        "n_cases": len(common_ids),
        "oof_mean": float(oof_mean),
        "oof_ci": [float(oof_lo), float(oof_hi)],
        "oof_patient_sd": oof_between,
        "oof_reading_sd": oof_within,
        "oof_readings": oof_readings,
        "fold_choices": fold_choices,
        "full_scores": full_scores,
        "best_full": best_full,
    }


def hemo_subgroup_rows(rows: list[dict[str, Any]], *, legacy_label: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """Standard stratified subgroup breakdown for expanded hemodynamic reports."""
    return [
        ("All patients", rows),
        (legacy_label, [r for r in rows if r.get("legacy_core")]),
        ("Untreated", [r for r in rows if not r["treated"]]),
        ("Pressor-treated", [r for r in rows if r["treated"]]),
        ("Normotensive", [r for r in rows if not r.get("hypertensive")]),
        ("Hypertensive", [r for r in rows if r.get("hypertensive")]),
        ("ASA 1-2", [r for r in rows if r.get("asa") in ("1", "2")]),
        ("ASA 3", [r for r in rows if r.get("asa") == "3"]),
        ("BMI under 30", [r for r in rows if r.get("bmi", 0.0) < 30.0]),
        ("BMI 30 and over", [r for r in rows if r.get("bmi", 0.0) >= 30.0]),
        ("Open, supine", [r for r in rows if r.get("procedure") == "Open, supine"]),
        ("Laparoscopic or robotic, supine", [r for r in rows if r.get("procedure") == "Laparoscopic or robotic, supine"]),
        ("Other positions", [r for r in rows if r.get("procedure") == "Other positions"]),
        ("Age under 45", [r for r in rows if r["age"] < 45]),
        ("Age 45-59", [r for r in rows if 45 <= r["age"] < 60]),
        ("Age 60-69", [r for r in rows if 60 <= r["age"] < 70]),
        ("Age 70 and over", [r for r in rows if r["age"] >= 70]),
    ]


def mean_ci(values, rng, *, n_boot: int = BOOTSTRAP, digits: int = 1, include_n: bool = True) -> str:
    arr = np.asarray(values, dtype=float)
    boot = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    text = f"{arr.mean():.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"
    return f"{text}, n={len(arr)}" if include_n else text


def report_header(title: str, settings: dict[str, Any]) -> list[str]:
    return [f"# {title}", "", f"Generated {settings['generated']}.", ""]


def cv_stability_lines(stability: dict[str, Any] | None) -> list[str]:
    if not stability:
        return []
    fold_str = ", ".join(f"{m:.1f}" for m in stability["fold_means"])
    return [
        "",
        f"{stability['n_folds']}-fold stratified fold means: {fold_str} mmHg "
        f"(mean {stability['mean']:.1f}, SD {stability['sd']:.1f}).",
    ]


def cv_grid_lines(cv_grid: dict[str, Any] | None) -> list[str]:
    if not cv_grid:
        return []
    return [
        "",
        "## Cross-validated parameter grid",
        "",
        f"{cv_grid['n_folds']}-fold out-of-fold MAP error across {cv_grid['n_cases']} patients: "
        f"{cv_grid['oof_mean']:.1f} ({cv_grid['oof_ci'][0]:.1f} to {cv_grid['oof_ci'][1]:.1f}) mmHg "
        f"(patient SD {cv_grid['oof_patient_sd']:.1f}, reading SD {cv_grid['oof_reading_sd']:.1f}).",
        f"Best full-cohort candidate: `{json.dumps(cv_grid['best_full']['override'])}` "
        f"(mean error {cv_grid['best_full']['mean_error']:.2f} mmHg).",
        "",
        "| Fold | Selected override | Train mean (mmHg) | Val mean (mmHg) | n_val |",
        "|---:|---|---:|---:|---:|",
        *(
            f"| {fc['fold']} | `{json.dumps(fc['override'])}` | {fc['train_mean']:.2f} | {fc['val_mean']:.2f} | {fc['n_val']} |"
            for fc in cv_grid["fold_choices"]
        ),
    ]


def report_footer(exclusions: list[dict[str, Any]]) -> list[str]:
    lines = ["", "Exclusions:", ""]
    lines.extend(
        f"- {reason}: {count}" for reason, count in sorted(Counter(r["exclusion"] for r in exclusions).items())
    )
    lines += [
        "",
        f"Source: [Lee et al., Scientific Data, 2022]({SOURCE}). VitalDB data are used under CC BY-NC-SA 4.0",
        "and its data use agreement.",
        "",
    ]
    return lines
