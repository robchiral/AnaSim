"""Shared VitalDB track loading, cohort filtering, preop-lab mapping, and patient-grouped CV."""

from __future__ import annotations

import csv
import gzip
import io
import math
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anasim.patient.domain import (  # noqa: E402
    AGE_RANGE_YEARS,
    BMI_RANGE_KG_M2,
    CREATININE_CLEARANCE_RANGE_ML_MIN,
    HEIGHT_RANGE_CM,
    HEMOGLOBIN_RANGE_G_DL,
    WEIGHT_RANGE_KG,
)
from anasim.patient.patient import Patient  # noqa: E402

CACHE = ROOT / ".vitaldb_cache"
API = "https://api.vitaldb.net"
SOURCE = "https://doi.org/10.1038/s41597-022-01411-5"
WAVE_DT = 1.0 / 62.5  # Dräger Primus waveform interval (s)
CM_PER_MBAR = 1.0197
REMI_RATES = ("Orchestra/RFTN20_RATE", "Orchestra/RFTN50_RATE")
REMI_UG_PER_ML = {"Orchestra/RFTN20_RATE": 20.0, "Orchestra/RFTN50_RATE": 50.0}
VASOACTIVE = ("NEPI", "PHEN", "EPI", "DOPA", "DOBU", "VASO", "NTG", "NPS", "MRN", "DTZ", "AMD", "PGE1")

_PLAUSIBLE_CREATININE_MG_DL = (0.2, 20.0)


class Excluded(ValueError):
    """A recording fails a prespecified input or measurement quality rule."""


def fetch(name: str, url: str, *, cache: Path | None = None) -> str:
    """Read a VitalDB CSV from the local cache, downloading atomically with retries on first access."""
    cache_dir = CACHE if cache is None else cache
    path = cache_dir / name
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{name}.{os.getpid()}.{threading.get_ident()}.tmp")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                try:
                    with urllib.request.urlopen(url, timeout=180) as response:
                        tmp.write_bytes(response.read())
                except urllib.error.URLError:
                    # Some Python installs lack CA certificates; curl uses the system store.
                    subprocess.run(["curl", "-sf", "-m", "180", "-o", str(tmp), url], check=True)
                os.replace(tmp, path)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                tmp.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(1.0 + attempt)
        if last_error is not None:
            raise last_error
    raw = path.read_bytes()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    return raw.decode("utf-8-sig")


def prefetch_tracks(tids: Iterable[str], *, workers: int = 16, cache: Path | None = None) -> int:
    """Download any missing VitalDB track CSVs in parallel; returns count fetched."""
    cache_dir = CACHE if cache is None else cache
    missing = sorted({tid for tid in tids if not (cache_dir / f"{tid}.csv").exists()})
    if not missing:
        return 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda tid: fetch(f"{tid}.csv", f"{API}/{tid}", cache=cache_dir), missing))
    return len(missing)


def load_track(tid: str, wave: bool = False, *, cache: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Waveform rows carry a time only at segment starts; numeric rows always do."""
    times, values = [], []
    t = None
    for line in fetch(f"{tid}.csv", f"{API}/{tid}", cache=cache).splitlines()[1:]:
        stamp, _, value = line.partition(",")
        if stamp:
            t = float(stamp)
        elif wave and t is not None:
            t += WAVE_DT
        else:
            continue
        if value:
            times.append(t)
            values.append(float(value))
    return np.array(times, dtype=float), np.array(values, dtype=float)


def numeric_track(tid: str, *, cache: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a finite, time-ordered, deduplicated numeric track."""
    time, values = load_track(tid, cache=cache)
    finite = np.isfinite(time) & np.isfinite(values)
    time, values = time[finite], values[finite]
    order = np.argsort(time, kind="stable")
    time, values = time[order], values[order]
    # Keep the last packet at an identical timestamp.
    unique = np.r_[np.diff(time) > 0, True] if len(time) else np.array([], dtype=bool)
    time, values = time[unique], values[unique]
    if len(time) < 2:
        raise Excluded("insufficient finite track data")
    return time, values


def held(series: tuple[np.ndarray, np.ndarray], grid: np.ndarray, max_age: float) -> np.ndarray:
    """Use the last recorded value; never bridge a gap beyond max_age seconds."""
    time, values = series
    index = np.searchsorted(time, grid, side="right") - 1
    clipped = np.maximum(index, 0)
    valid = (index >= 0) & (grid <= time[-1]) & (grid - time[clipped] <= max_age)
    return np.where(valid, values[clipped], np.nan)


def covered(series: tuple[np.ndarray, np.ndarray], start: float, end: float, max_gap: float) -> None:
    """Require dense track coverage across [start, end] with no gap exceeding max_gap."""
    time, _ = series
    inside = time[(time >= start) & (time <= end)]
    if (
        len(inside) < 2
        or inside[0] - start > max_gap
        or end - inside[-1] > max_gap
        or np.max(np.diff(inside)) > max_gap
    ):
        raise Excluded("incomplete input history")


def time_grid(start: float, end: float, dt: float) -> np.ndarray:
    return np.r_[np.arange(start, end, dt), end]


def delivered_volume(rate: tuple[np.ndarray, np.ndarray], grid: np.ndarray) -> np.ndarray:
    """Integrate recorded mL/h exactly between packets, then sample cumulative mL."""
    time, values = rate
    cumulative = np.r_[0.0, np.cumsum(values[:-1] * np.diff(time) / 3600.0)]
    if grid[0] < time[0] or grid[-1] > time[-1]:
        raise Excluded("delivery replay outside recorded interval")
    return np.interp(grid, time, cumulative)


def optional_float(case: dict[str, Any], key: str) -> float | None:
    raw = case.get(key)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def amount(case: dict[str, Any], key: str) -> float:
    value = optional_float(case, key)
    return value if value is not None else 0.0


def cockcroft_gault_crcl(age: float, weight: float, height: float, sex: str, scr_mg_dl: float) -> float:
    """Cockcroft-Gault CrCl (mL/min) using adjusted body weight above IBW in obesity."""
    height_in = height / 2.54
    ibw = (50.0 if sex == "male" else 45.5) + 2.3 * (height_in - 60.0)
    ibw = max(40.0, ibw)
    cg_weight = ibw + 0.4 * (weight - ibw) if weight > ibw else weight
    crcl = ((140.0 - age) * cg_weight) / (72.0 * scr_mg_dl)
    if sex == "female":
        crcl *= 0.85
    lo, hi = CREATININE_CLEARANCE_RANGE_ML_MIN
    return float(min(hi, max(lo, crcl)))


def _demographics_in_domain(age: float, weight: float, height: float) -> bool:
    bmi = weight / (height / 100.0) ** 2
    return (
        AGE_RANGE_YEARS[0] <= age <= AGE_RANGE_YEARS[1]
        and WEIGHT_RANGE_KG[0] <= weight <= WEIGHT_RANGE_KG[1]
        and HEIGHT_RANGE_CM[0] <= height <= HEIGHT_RANGE_CM[1]
        and BMI_RANGE_KG_M2[0] <= bmi <= BMI_RANGE_KG_M2[1]
    )


def has_plausible_creatinine(case: dict[str, Any]) -> bool:
    """True when recorded preop creatinine lies inside the plausible mg/dL bounds."""
    scr = optional_float(case, "preop_cr")
    return scr is not None and _PLAUSIBLE_CREATININE_MG_DL[0] <= scr <= _PLAUSIBLE_CREATININE_MG_DL[1]


def in_anasim_domain(case: dict[str, Any]) -> bool:
    """True if demographics (and recorded preop Hb, if present) lie inside AnaSim's domain."""
    if case.get("sex") not in ("M", "F"):
        return False
    age = optional_float(case, "age")
    weight = optional_float(case, "weight")
    height = optional_float(case, "height")
    if age is None or weight is None or height is None or height <= 0:
        return False
    if not _demographics_in_domain(age, weight, height):
        return False
    hb = optional_float(case, "preop_hb")
    if hb is not None and not (HEMOGLOBIN_RANGE_G_DL[0] <= hb <= HEMOGLOBIN_RANGE_G_DL[1]):
        return False
    return True


def patient_from_case(
    case: dict[str, Any],
    *,
    baseline_map: float = 90.0,
    baseline_hr: float = 70.0,
    use_preop_labs: bool = True,
) -> Patient:
    """Construct an AnaSim Patient from a VitalDB case row, wiring preop Hb and CrCl when valid."""
    if case.get("sex") not in ("M", "F"):
        raise Excluded("missing patient sex")
    age = optional_float(case, "age")
    weight = optional_float(case, "weight")
    height = optional_float(case, "height")
    if age is None or weight is None or height is None or height <= 0:
        raise Excluded("missing patient demographics")
    if not _demographics_in_domain(age, weight, height):
        raise Excluded("outside simulator patient domain")
    sex = "male" if case["sex"] == "M" else "female"
    kwargs: dict[str, Any] = {
        "age": age,
        "weight": weight,
        "height": height,
        "sex": sex,
        "baseline_map": baseline_map,
        "baseline_hr": baseline_hr,
    }
    if use_preop_labs:
        hb = optional_float(case, "preop_hb")
        if hb is not None:
            if not (HEMOGLOBIN_RANGE_G_DL[0] <= hb <= HEMOGLOBIN_RANGE_G_DL[1]):
                raise Excluded("preop hemoglobin outside simulator domain")
            kwargs["baseline_hb"] = hb
        if has_plausible_creatinine(case):
            scr = optional_float(case, "preop_cr")
            assert scr is not None
            kwargs["creatinine_clearance_ml_min"] = cockcroft_gault_crcl(age, weight, height, sex, scr)
    return Patient(**kwargs)


def pft_group(case: dict[str, Any]) -> str:
    """Classify VitalDB preop_pft text into Normal, Obstructive, Restrictive, or Unknown."""
    label = (case.get("preop_pft") or "").strip().lower()
    if not label:
        return "Unknown"
    if label == "normal":
        return "Normal"
    if "obstructive" in label:
        return "Obstructive"
    if "restrictive" in label:
        return "Restrictive"
    return "Unknown"


def is_legacy_core(case: dict[str, Any]) -> bool:
    """Prior narrow reference subset: age <= 70, weight <= 100, BMI <= 32, open supine."""
    age = optional_float(case, "age")
    weight = optional_float(case, "weight")
    height = optional_float(case, "height")
    if age is None or weight is None or height is None or height <= 0:
        return False
    bmi = weight / (height / 100.0) ** 2
    return (
        18.0 <= age <= 70.0
        and 50.0 <= weight <= 100.0
        and 150.0 <= height <= 200.0
        and 18.0 <= bmi <= 32.0
        and case.get("approach") == "Open"
        and case.get("position") == "Supine"
    )


def is_legacy_hemo_core(case: dict[str, Any]) -> bool:
    """Prior TIVA hemodynamic reference subset: GenSurg, ASA 1-2, normotensive, weight <= 100, BMI <= 32."""
    weight = optional_float(case, "weight")
    height = optional_float(case, "height")
    if weight is None or height is None or height <= 0:
        return False
    bmi = weight / (height / 100.0) ** 2
    return (
        weight <= 100.0
        and bmi <= 32.0
        and case.get("department") == "General surgery"
        and case.get("asa") in ("1", "2")
        and case.get("preop_htn") == "0"
    )


def domain_cases(
    required: Sequence[str],
    *,
    typical_only: bool = True,
    open_supine_only: bool | None = None,
    supine_only: bool = False,
    exclude_tracks: Sequence[str] = (),
    deduplicate_subjects: bool = True,
    cache: Path | None = None,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Adults in AnaSim's domain under general anesthesia with a single-lumen oral tube.

    When ``open_supine_only`` is explicitly given it controls approach/position filtering;
    otherwise ``typical_only=True`` restricts to open supine cases and ``typical_only=False``
    retains all oral single-lumen general anesthetics inside AnaSim's patient domain.
    """
    restrict_open_supine = typical_only if open_supine_only is None else open_supine_only
    tracks: dict[int, dict[str, str]] = {}
    for row in csv.DictReader(io.StringIO(fetch("trks.csv", f"{API}/trks", cache=cache))):
        tracks.setdefault(int(row["caseid"]), {})[row["tname"]] = row["tid"]
    all_cases = sorted(
        csv.DictReader(io.StringIO(fetch("cases.csv", f"{API}/cases", cache=cache))),
        key=lambda row: int(row["caseid"]),
    )
    out: list[tuple[dict[str, str], dict[str, str]]] = []
    seen_subjects: set[str] = set()
    for case in all_cases:
        if case["ane_type"] != "General" or case["dltubesize"] or case["airway"] != "Oral":
            continue
        if not in_anasim_domain(case):
            continue
        if restrict_open_supine and (case["approach"] != "Open" or case["position"] != "Supine"):
            continue
        if supine_only and case["position"] != "Supine":
            continue
        trk = tracks.get(int(case["caseid"]), {})
        if not all(name in trk for name in required):
            continue
        if any(name in trk for name in exclude_tracks):
            continue
        subject = case.get("subjectid", "")
        if deduplicate_subjects and subject:
            if subject in seen_subjects:
                continue
            seen_subjects.add(subject)
        out.append((case, trk))
    return out


def kfold_splits(
    items: Sequence[Any],
    *,
    n_folds: int = 5,
    seed: int = 20261024,
    group_key: Callable[[Any], Any] | None = None,
    strata: Callable[[Any], Any] | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Deterministic patient-grouped, optionally stratified k-fold index splits.

    All items sharing ``group_key(item)`` stay in the same fold. Stratification assigns
    groups within each stratum round-robin across folds after seeded shuffling.
    """
    n = len(items)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    groups: dict[Any, list[int]] = {}
    for idx, item in enumerate(items):
        key = idx if group_key is None else group_key(item)
        groups.setdefault(key, []).append(idx)
    if len(groups) < n_folds:
        raise ValueError(f"cannot build {n_folds} folds from {len(groups)} groups")

    rng = np.random.default_rng(seed)
    by_stratum: dict[Any, list[list[int]]] = {}
    for member_indices in groups.values():
        stratum = None if strata is None else strata(items[member_indices[0]])
        by_stratum.setdefault(stratum, []).append(member_indices)

    fold_val: list[list[int]] = [[] for _ in range(n_folds)]
    cursor = 0
    for stratum in sorted(by_stratum, key=str):
        stratum_groups = by_stratum[stratum]
        order = rng.permutation(len(stratum_groups))
        for pos in order:
            fold_val[cursor % n_folds].extend(stratum_groups[int(pos)])
            cursor += 1

    all_indices = set(range(n))
    splits: list[tuple[list[int], list[int]]] = []
    for val in fold_val:
        val_sorted = sorted(val)
        train_sorted = sorted(all_indices.difference(val_sorted))
        splits.append((train_sorted, val_sorted))
    return splits
