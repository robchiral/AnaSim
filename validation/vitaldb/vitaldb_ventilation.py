"""Compare AnaSim ventilation and capnography with Dräger Primus recordings from VitalDB.

Volume-control windows are steady minutes of volume-controlled ventilation in
adults inside AnaSim's patient domain (general anesthesia, oral single-lumen
tube, open surgery, supine). The script measures resistance, compliance, and
waveform shape from recorded airway pressure and CO2, then runs each breath
through AnaSim's ventilator, lung, sensor, and capnograph with that patient's
settings. It also checks pressure-control rise time, pre-incision plateau
compliance dependence on body size across supine patients (including
pre-pneumoperitoneum laparoscopic cases), and FiO2 - EtO2 against FiO2 - FeO2.

Calibration parameters are evaluated with patient-grouped k-fold cross-validation
(reporting out-of-fold metrics and per-fold stability) and fitted on 100% of
collected VitalDB windows.

Data: VitalDB (Lee et al., Sci Data 2022; CC BY-NC-SA 4.0 with a data use
agreement). Tracks download on first use into validation/vitaldb/.vitaldb_cache.

    python -m validation.vitaldb.vitaldb_ventilation [--cases 60] [--compliance-cases 0] [--folds 5]

Writes validation/vitaldb/results/vitaldb_ventilation.md.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anasim.machine.ventilator as ventilator  # noqa: E402
from anasim.core.engine import SimulationEngine  # noqa: E402
from anasim.core.state import SimulationConfig  # noqa: E402
from anasim.machine.ventilator import AnesthesiaVentilator  # noqa: E402
from anasim.monitors.airway import AirwaySensor  # noqa: E402
from anasim.monitors.capno import Capnograph  # noqa: E402
from anasim.patient.patient import Patient  # noqa: E402
from anasim.physiology.lung import LungAeration  # noqa: E402
from anasim.physiology.resp_mech import RespiratoryMechanics  # noqa: E402
from validation.vitaldb import vitaldb_data  # noqa: E402
from validation.vitaldb.vitaldb_data import (  # noqa: E402
    CACHE,
    CM_PER_MBAR,
    DEFAULT_WORKERS,
    WAVE_DT,
    domain_cases,
    is_legacy_core,
    kfold_splits,
    load_track,
    patient_from_case,
    pft_group,
)

__all__ = [
    "COMPLIANCE_TRACKS",
    "template",
    "window_features",
]

LEAD = 5  # Template samples before inspiratory onset
TRACKS = [
    "Primus/AWP", "Primus/CO2", "Primus/SET_RR_IPPV", "Primus/SET_TV_L", "Primus/SET_INSP_TM",
    "Primus/SET_INSP_PAUSE", "Primus/SET_INTER_PEEP", "Primus/SET_PIP", "Primus/TV", "Primus/RR_CO2",
    "Primus/ETCO2", "Primus/FIO2", "Primus/FEO2",
]
WAVES = ("Primus/AWP", "Primus/CO2")
COMPLIANCE_TRACKS = ["Primus/TV", "Primus/PPLAT_MBAR", "Primus/PEEP_MBAR"]
PCV_TRACKS = ["Primus/AWP", "Primus/SET_INSP_PRES", "Primus/SET_RR_IPPV", "Primus/SET_INSP_TM"]
SEGMENTS = ("onset", "ramp", "end_flow", "exp_fall", "exp_tail")
TRANSITIONS = ("onset", "end_flow", "exp_fall")
SENSOR_TAUS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04)
LIMB_RESISTANCES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
CAPNOGRAPH_GRIDS = (
    ("ANALYZER_TAU", (0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16), "fall"),
    ("PHASE_II_VOLUME", (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25), "rise"),
    ("PHASE_III_SLOPE", (5.0, 10.0, 15.0, 20.0, 25.0, 30.0), "slope"),
)
RISE_TIMES = (0.15, 0.2, 0.25, 0.3, 0.35)


@contextmanager
def _override_attr(obj: Any, name: str, value: Any):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


def _make_capnograph(dead_space_l: float, params: dict[str, float] | None = None) -> Capnograph:
    if not params:
        return Capnograph(dead_space_l)
    cls = type("_CalibratedCapnograph", (Capnograph,), params)
    return cls(dead_space_l)


# --- Window extraction ------------------------------------------------------

def window_settings(trk: dict[str, str], names: list[str], t0: float, t1: float) -> dict[str, float] | None:
    """Median of each numeric track in [t0, t1]; None if missing or a setting changed."""
    window: dict[str, float] = {}
    for name in names:
        t, v = load_track(trk[name])
        v = v[(t >= t0) & (t <= t1)]
        if len(v) == 0 or (name.startswith("Primus/SET_") and np.ptp(v) > 1e-6):
            return None
        window[name.split("/")[1]] = float(np.median(v))
    return window


def steady_window(case: dict[str, str], trk: dict[str, str], half_s: float = 60.0) -> dict[str, Any] | None:
    """Two minutes at mid-surgery with fixed settings and only machine breaths."""
    try:
        t_mid = (float(case["opstart"]) + float(case["opend"])) / 2
    except ValueError:
        return None
    t0, t1 = t_mid - half_s, t_mid + half_s
    settings = window_settings(trk, [name for name in TRACKS if name not in WAVES], t0, t1)
    if settings is None or abs(settings["RR_CO2"] - settings["SET_RR_IPPV"]) > 1.0:
        return None
    window: dict[str, Any] = dict(settings)
    for name in WAVES:
        t, v = load_track(trk[name], wave=True)
        inside = (t >= t0) & (t <= t1)
        if inside.sum() < 0.9 * (t1 - t0) / WAVE_DT:
            return None
        window[name.split("/")[1]] = v[inside] * (CM_PER_MBAR if name == "Primus/AWP" else 1.0)
    window["case"] = case
    window["patient"] = patient_from_case(case)
    return window


# --- Breath features --------------------------------------------------------

def template(signal: np.ndarray, rr: float) -> np.ndarray | None:
    """Median breath aligned where pressure leaves PEEP."""
    n = round(60.0 / rr / WAVE_DT)
    peep, peak = np.percentile(signal, [10, 99])
    level = peep + 0.3 * (peak - peep)
    breaths = []
    for i in np.flatnonzero((signal[:-1] < level) & (signal[1:] >= level)):
        j = i
        while j > 0 and signal[j - 1] > peep + 0.3 and i - j < 20:
            j -= 1
        if j >= LEAD and j - LEAD + n <= len(signal):
            breaths.append(signal[j - LEAD:j - LEAD + n])
    return np.median(np.array(breaths), axis=0) if len(breaths) >= 5 else None


def features(tpl: np.ndarray, ti: float, vt_ml: float, pause: float) -> dict[str, Any]:
    """Resistance from the end-of-flow drop, compliance from the plateau, and timing."""
    t = (np.arange(len(tpl)) - LEAD) * WAVE_DT
    t_flow = ti * (1 - pause)
    peep = np.median(tpl[t > ti + 0.6 * (t[-1] - ti)])
    insp = (t >= 0) & (t <= ti)
    ppeak = tpl[insp].max()
    pplat = np.mean(tpl[(t >= t_flow + 0.5 * pause * ti) & (t <= ti - WAVE_DT / 2)])
    flow = vt_ml / 1000 / t_flow  # L/s
    r = (ppeak - pplat) / flow
    c = vt_ml / (pplat - peep)  # mL/cmH2O
    window = (t >= 0) & (t <= min(0.4, 0.8 * t_flow))
    onset_time, onset_pressure = t[window], tpl[window]

    def onset_error(params):
        tau, shift = params
        s = np.clip(onset_time - shift, 0, None)
        return peep + r * flow * (1 - np.exp(-s / tau)) + flow * s * 1000 / c - onset_pressure

    tau_knee = least_squares(onset_error, [0.05, 0.0], bounds=([0.002, -0.03], [0.5, 0.05])).x[0]
    tail = (t >= ti + 0.2) & (t <= ti + 2.0)
    excess = tpl[tail] - peep
    good = excess > 0.05
    tau_e = amplitude = np.nan
    if good.sum() >= 5:
        slope, intercept = np.polyfit(t[tail][good] - ti, np.log(excess[good]), 1)
        if slope < 0 and np.exp(intercept) >= 0.5:
            tau_e, amplitude = -1 / slope, np.exp(intercept)
    after_ti = tpl[np.searchsorted(t, ti + 0.2)]
    volume_control = (
        abs(t[insp][np.argmax(tpl[insp])] - t_flow) <= 0.1
        and after_ti < peep + 0.5 * (pplat - peep)
        and 1.0 <= r <= 40.0 and 10.0 <= c <= 150.0
    )
    return {
        "peep": peep, "r": r, "c": c, "tau_knee": tau_knee, "tau_e": tau_e,
        "tau_e_ratio": tau_e / (r * c / 1000) if r > 0 and c > 0 else np.nan,
        "r_exp": amplitude * r * c / vt_ml, "t_flow": t_flow,
        "volume_control": volume_control,
    }


def window_features(w: dict[str, Any], tpl: np.ndarray) -> dict[str, Any]:
    return features(tpl, w["SET_INSP_TM"], w["TV"], w["SET_INSP_PAUSE"] / 100)


def segment_rmse(sim: np.ndarray, real: np.ndarray, ti: float, t_flow: float) -> dict[str, float]:
    t = (np.arange(len(real)) - LEAD) * WAVE_DT
    masks = {
        "onset": (t >= -0.05) & (t <= 0.35), "ramp": (t > 0.35) & (t < t_flow - 0.05),
        "end_flow": (t >= t_flow - 0.05) & (t <= ti), "exp_fall": (t > ti) & (t <= ti + 0.4),
        "exp_tail": (t > ti + 0.4) & (t <= ti + 2.0),
    }
    n = min(len(sim), len(real))
    squared_error = (sim[:n] - real[:n]) ** 2
    return {k: float(np.sqrt(np.mean(squared_error[m[:n]]))) for k, m in masks.items()}


def capnogram_shape(co2: np.ndarray, period: float) -> dict[str, float]:
    """Features of one cyclic CO2 breath measured within the CO2 track alone."""
    n = len(co2)
    lo, hi = np.percentile(co2, [3, 97])
    amplitude = hi - lo
    above = co2 > lo + 0.5 * amplitude
    fall_end = np.flatnonzero(above & ~np.roll(above, -1))[0]
    y = np.roll(co2, -(fall_end + 1) + 40)
    t = np.arange(n) * WAVE_DT

    def cross(fraction, rising, start=0):
        level = lo + fraction * amplitude
        for i in range(start, n - 1):
            if (y[i] < level <= y[i + 1]) if rising else (y[i] > level >= y[i + 1]):
                return t[i] + (level - y[i]) / (y[i + 1] - y[i]) * WAVE_DT
        return np.nan

    fall90, fall50, fall10 = cross(0.9, False), cross(0.5, False), cross(0.1, False)
    after = int(fall10 / WAVE_DT) + 1
    rise10, rise50, rise90 = cross(0.1, True, after), cross(0.5, True, after), cross(0.9, True, after)
    plateau = (t > rise90 + 0.25) & (t < period + fall90 - 0.15)
    slope = np.polyfit(t[plateau], y[plateau], 1)[0] / amplitude if plateau.sum() > 8 else np.nan
    return {
        "fall": fall10 - fall90, "rise": rise90 - rise10, "duty": 1.0 - (rise50 - fall50) / period,
        "slope": 100 * slope,
    }


def cyclic_breath(signal: np.ndarray, rr: float) -> np.ndarray:
    """Median of consecutive breath-length windows: one breath, starting anywhere."""
    n = round(60.0 / rr / WAVE_DT)
    return np.median(signal[: len(signal) // n * n].reshape(-1, n), axis=0)


# --- AnaSim simulation ------------------------------------------------------

def simulate(
    w: dict[str, Any],
    resistance: float,
    compliance: float,
    *,
    viscoelastic: bool = True,
    sensor_tau: float | None = None,
    capnograph: Capnograph | None = None,
    settle: float = 20.0,
    substeps: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Paw and CO2 at the Primus rate for six breaths after settle seconds."""
    lung = RespiratoryMechanics(compliance=compliance, resistance=resistance)
    if not viscoelastic:
        lung.viscoelastic_ratio = 0.0
    vent = AnesthesiaVentilator()
    rr = w["SET_RR_IPPV"]
    ti_fraction = w["SET_INSP_TM"] * rr / 60
    vent.update_settings(
        rr=rr, tv=w["TV"], peep=w["features"]["peep"], ie=ti_fraction / (1 - ti_fraction),
        mode="VCV", pause=w["SET_INSP_PAUSE"],
    )
    sensor = AirwaySensor(sensor_tau) if sensor_tau else None
    dt = WAVE_DT / substeps
    paw, co2 = [], []
    volume = None
    for k in range(int((settle + 6.5 * 60 / rr) / dt)):
        before = vent.volume
        vent.step(dt, lung, "vent")
        p = sensor.step(dt, vent.paw, vent.volume, vent.volume - before)[0] if sensor else vent.paw
        c = 0.0
        if capnograph is not None:
            c = capnograph.step(dt, 0.0 if volume is None else vent.volume - volume, w["ETCO2"])
            volume = vent.volume
        if k % substeps == substeps - 1:
            paw.append(p)
            co2.append(c)
    start = int(settle / WAVE_DT)
    return np.array(paw[start:]), np.array(co2[start:])


def matched_mechanics(w: dict[str, Any], viscoelastic: bool, sensor_tau: float) -> tuple[float, float]:
    """Resistance and compliance whose simulated breath has the case's end-of-flow drop and plateau."""
    f = w["features"]
    target = np.log([f["r"], f["c"]])

    def residual(x):
        paw, _ = simulate(
            w, math.exp(x[0]), math.exp(x[1]) / 1000, viscoelastic=viscoelastic,
            sensor_tau=sensor_tau, settle=12.0,
        )
        tpl_sim = template(paw, w["SET_RR_IPPV"])
        assert tpl_sim is not None
        g = window_features(w, tpl_sim)
        return np.log([max(g["r"], 0.1), max(g["c"], 1.0)]) - target

    x = least_squares(residual, target, diff_step=1e-3, xtol=1e-4, ftol=1e-4).x
    return math.exp(x[0]), math.exp(x[1]) / 1000


def compare_paw_case(w: dict[str, Any], viscoelastic: bool, sensor_tau: float) -> dict[str, float]:
    """Single-case airway pressure errors and predicted features with matched mechanics."""
    resistance, compliance = matched_mechanics(w, viscoelastic, sensor_tau)
    sim = template(
        simulate(w, resistance, compliance, viscoelastic=viscoelastic, sensor_tau=sensor_tau)[0],
        w["SET_RR_IPPV"],
    )
    assert sim is not None
    g = window_features(w, sim)
    seg = segment_rmse(sim, w["template"], w["SET_INSP_TM"], w["features"]["t_flow"])
    n = min(len(sim), len(w["template"]))
    return {
        **seg,
        "whole": float(np.sqrt(np.mean((sim[:n] - w["template"][:n]) ** 2))),
        "transitions": float(np.sqrt(np.mean([seg[k] ** 2 for k in TRANSITIONS]))),
        "knee": float(g["tau_knee"]),
        "tau_e_ratio": float(g["tau_e_ratio"]),
        "r_exp": float(g["r_exp"]),
    }


def _paw_payload(w: dict[str, Any]) -> dict[str, Any]:
    return {
        "template": w["template"],
        "features": w["features"],
        "SET_RR_IPPV": w["SET_RR_IPPV"],
        "SET_INSP_TM": w["SET_INSP_TM"],
        "TV": w["TV"],
        "SET_INSP_PAUSE": w["SET_INSP_PAUSE"],
    }


def _capno_payload(w: dict[str, Any]) -> dict[str, Any]:
    return {
        "features": w["features"],
        "patient": w["patient"],
        "SET_RR_IPPV": w["SET_RR_IPPV"],
        "SET_INSP_TM": w["SET_INSP_TM"],
        "TV": w["TV"],
        "SET_INSP_PAUSE": w["SET_INSP_PAUSE"],
        "ETCO2": w["ETCO2"],
    }


def _eto2_payload(w: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient": w["patient"],
        "SET_RR_IPPV": w["SET_RR_IPPV"],
        "SET_INSP_TM": w["SET_INSP_TM"],
        "TV": w["TV"],
        "SET_INTER_PEEP": w["SET_INTER_PEEP"],
        "FIO2": w["FIO2"],
    }


def _compare_paw_task(args: tuple[dict[str, Any], bool, float, float]) -> dict[str, float]:
    w, viscoelastic, sensor_tau, circuit_resistance = args
    with _override_attr(ventilator, "CIRCUIT_RESISTANCE", circuit_resistance):
        return compare_paw_case(w, viscoelastic, sensor_tau)


def _capnogram_task(args: tuple[dict[str, Any], dict[str, float]]) -> dict[str, float]:
    w, params = args
    return capnogram_case_features(w, params)


def _pcv_ramp_task(ramp: float) -> tuple[float, float]:
    with _override_attr(ventilator, "RISE_TIME", ramp):
        lung = RespiratoryMechanics(compliance=0.045, resistance=10.0)
        vent = AnesthesiaVentilator()
        vent.update_settings(rr=12, peep=5, p_insp=18, ie="1:2", mode="PCV")
        sensor = AirwaySensor()
        dt = WAVE_DT / 4
        paw: list[float] = []
        for k in range(int(45 / dt)):
            before = vent.volume
            vent.step(dt, lung, "vent")
            p_meas = sensor.step(dt, vent.paw, vent.volume, vent.volume - before)[0]
            if k % 4 == 3:
                paw.append(p_meas)
        tpl_sim = template(np.array(paw[int(10 / WAVE_DT):]), 12)
        assert tpl_sim is not None
        return ramp, rise_10_90(tpl_sim, 5 / 3)


def _compliance_sim_task(args: tuple[Patient, float]) -> float:
    patient, tv_ml = args
    return anasim_plateau_compliance(patient, tv_ml)


def _aeration_mechanics_task(args: tuple[Patient, float, float]) -> tuple[float, float]:
    patient, tv_ml, peep = args
    return anasim_aeration_mechanics(patient, tv_ml, peep)


_PAW_KEYS = (*SEGMENTS, "whole", "transitions", "knee", "tau_e_ratio", "r_exp")


def _pool_map(
    executor: ProcessPoolExecutor | None,
    fn,
    tasks,
    *,
    chunksize: int = 1,
) -> list[Any]:
    if executor is None:
        return [fn(task) for task in tasks]
    return list(executor.map(fn, tasks, chunksize=chunksize))


def _pack_paw_rows(rows: list[dict[str, float]]) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {k: [] for k in _PAW_KEYS}
    for row in rows:
        for k in _PAW_KEYS:
            results[k].append(row[k])
    return results


def _compare_paw_batch(
    cases: list[dict[str, Any]],
    configs: list[tuple[Any, bool, float, float]],
    executor: ProcessPoolExecutor | None,
) -> dict[Any, dict[str, list[float]]]:
    """Run multiple (viscoelastic, sensor_tau, circuit_resistance) evaluations across one process pool batch."""
    payloads = [_paw_payload(w) for w in cases]
    tasks = [
        (payload, viscoelastic, sensor_tau, circuit_resistance)
        for _tag, viscoelastic, sensor_tau, circuit_resistance in configs
        for payload in payloads
    ]
    all_rows = _pool_map(executor, _compare_paw_task, tasks, chunksize=2)
    n = len(cases)
    return {
        tag: _pack_paw_rows(all_rows[idx * n:(idx + 1) * n])
        for idx, (tag, *_rest) in enumerate(configs)
    }


def capnogram_case_features(w: dict[str, Any], params: dict[str, float]) -> dict[str, float]:
    """Simulated capnogram features for one case using PBW-based series dead space."""
    f = w["features"]
    patient: Patient = w["patient"]
    dead_space_l = 2.2 * patient.predicted_body_weight() / 1000.0
    capnograph = _make_capnograph(dead_space_l, params)
    _, co2 = simulate(w, f["r"], f["c"] / 1000, capnograph=capnograph, substeps=2)
    return capnogram_shape(cyclic_breath(co2, w["SET_RR_IPPV"]), 60 / w["SET_RR_IPPV"])


def _populate_capno_cache(
    cases: list[dict[str, Any]],
    missing: list[tuple[int, dict[str, float], tuple[int, tuple[tuple[str, float], ...]]]],
    cache: dict[tuple[int, tuple[tuple[str, float], ...]], dict[str, float]],
    executor: ProcessPoolExecutor | None,
) -> None:
    if not missing:
        return
    dedup: dict[tuple[int, tuple[tuple[str, float], ...]], tuple[int, dict[str, float]]] = {}
    for i, trial, key in missing:
        if key not in cache and key not in dedup:
            dedup[key] = (i, trial)
    if not dedup:
        return
    keys = list(dedup)
    tasks = [(_capno_payload(cases[i]), trial) for i, trial in dedup.values()]
    chunksize = max(1, min(4, len(tasks) // 16 or 1))
    results = _pool_map(executor, _capnogram_task, tasks, chunksize=chunksize)
    for key, res in zip(keys, results, strict=True):
        cache[key] = res


def fit_capnograph_subset(
    cases: list[dict[str, Any]],
    indices: list[int],
    base_params: dict[str, float],
    cache: dict[tuple[int, tuple[tuple[str, float], ...]], dict[str, float]],
    *,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[dict[str, float], dict[str, dict[float, float]]]:
    """Fit each capnograph parameter sequentially on ``indices``, reusing cached per-case sims."""
    current = dict(base_params)
    grids: dict[str, dict[float, float]] = {}
    for name, values, feature in CAPNOGRAPH_GRIDS:
        recorded = np.array([cases[i]["capnogram"][feature] for i in indices])
        trials = [(value, {**current, name: value}) for value in values]
        missing = [
            (i, trial, (i, tuple(sorted(trial.items()))))
            for _value, trial in trials
            for i in indices
            if (i, tuple(sorted(trial.items()))) not in cache
        ]
        _populate_capno_cache(cases, missing, cache, executor)
        grid: dict[float, float] = {}
        for value, trial in trials:
            key_tail = tuple(sorted(trial.items()))
            sims = [cache[(i, key_tail)][feature] for i in indices]
            grid[value] = float(np.nanmedian(np.abs(np.array(sims) - recorded)))
        best = min(grid, key=lambda k: grid[k])
        current[name] = best
        grids[name] = grid
    return current, grids


def rise_10_90(tpl: np.ndarray, ti: float) -> float:
    t = (np.arange(len(tpl)) - LEAD) * WAVE_DT
    base = np.median(tpl[:LEAD])
    plateau = np.median(tpl[(t > 0.5 * ti) & (t < 0.9 * ti)])
    if plateau - base < 5:
        return np.nan

    def cross(fraction):
        level = base + fraction * (plateau - base)
        i = np.flatnonzero(tpl >= level)[0]
        return t[i - 1] + (level - tpl[i - 1]) / (tpl[i] - tpl[i - 1]) * WAVE_DT

    return cross(0.9) - cross(0.1)


def pcv_rise_time(
    max_cases: int, *, seed: int = 4, executor: ProcessPoolExecutor | None = None,
) -> tuple[list[float], dict[float, float]]:
    """10-90% Paw rise of recorded pressure-control breaths, and AnaSim's for each pressure ramp."""
    candidates = domain_cases(PCV_TRACKS, open_supine_only=False)
    random.Random(seed).shuffle(candidates)
    recorded: list[float] = []
    for case, trk in candidates:
        try:
            t_mid = (float(case["opstart"]) + float(case["opend"])) / 2
        except ValueError:
            continue
        settings = window_settings(trk, PCV_TRACKS[1:], t_mid - 60, t_mid + 60)
        if settings is None:
            continue
        t_wave, p_wave = load_track(trk["Primus/AWP"], wave=True)
        p_win = p_wave[(t_wave >= t_mid - 60) & (t_wave <= t_mid + 60)] * CM_PER_MBAR
        tpl = template(p_win, settings["SET_RR_IPPV"]) if len(p_win) > 3000 else None
        rise = np.nan if tpl is None else rise_10_90(tpl, settings["SET_INSP_TM"])
        if np.isfinite(rise):
            recorded.append(float(rise))
            print(f"pressure control case {case['caseid']} ({len(recorded)}/{max_cases or 'all'})", flush=True)
        if max_cases and len(recorded) == max_cases:
            break
    simulated = dict(_pool_map(executor, _pcv_ramp_task, RISE_TIMES))
    return recorded, simulated


def anasim_plateau_compliance(patient: Patient, tv_ml: float) -> float:
    """VT/(Pplat - PEEP) from AnaSim at PEEP 5, RR 12, I:E 1:2, and a 10% pause."""
    lung = RespiratoryMechanics(compliance=patient.respiratory_compliance())
    vent = AnesthesiaVentilator()
    vent.update_settings(rr=12, tv=tv_ml, peep=5, ie="1:2", mode="VCV", pause=10)
    for _ in range(6):
        vent.step(5.0, lung, "vent")
    return vent.monitors.tv_exp / (vent.monitors.paw_plat - vent.monitors.peep)


def anasim_aeration_mechanics(patient: Patient, tv_ml: float, peep: float) -> tuple[float, float]:
    """Settled (plateau compliance mL/cmH2O, driving pressure cmH2O) with LungAeration at recorded PEEP."""
    aeration = LungAeration(patient, recruited=0.9, atelectatic=LungAeration.INDUCTION_ATELECTASIS)
    aeration.unconscious = 1.0
    aeration.inert_fraction = 0.5
    lung = RespiratoryMechanics(resistance=patient.airway_resistance(), aeration=aeration)
    vent = AnesthesiaVentilator()
    vent.update_settings(rr=12, tv=tv_ml, peep=max(0.0, peep), ie="1:2", mode="VCV", pause=10)
    for _ in range(60):
        vent.step(5.0, lung, "vent")
    driving = max(0.1, vent.monitors.paw_plat - vent.monitors.peep)
    return vent.monitors.tv_exp / driving, driving


def compliance_by_patient(
    max_cases: int,
    *,
    n_folds: int = 5,
    seed: int = 3,
    workers: int = DEFAULT_WORKERS,
    executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    """Pre-incision plateau compliance across supine patients with patient-grouped k-fold CV."""
    candidates = domain_cases(
        COMPLIANCE_TRACKS,
        open_supine_only=False,
        supine_only=True,
        exclude_tracks=("Primus/SET_INSP_PRES",),
    )
    random.Random(seed).shuffle(candidates)
    pool_items = candidates[:max_cases] if max_cases else candidates

    def measure(item):
        case, trk = item
        try:
            t0, t1 = float(case["anestart"]) + 600, float(case["opstart"])
        except ValueError:
            return None
        values = window_settings(trk, COMPLIANCE_TRACKS, t0, t1) if t1 - t0 >= 180 else None
        if values is None:
            return None
        driving = (values["PPLAT_MBAR"] - values["PEEP_MBAR"]) * CM_PER_MBAR
        if driving < 3 or values["TV"] < 200:
            return None
        patient = patient_from_case(case)
        return {
            "caseid": int(case["caseid"]),
            "subjectid": case.get("subjectid", case["caseid"]),
            "c": values["TV"] / driving,
            "driving": driving,
            "peep": values["PEEP_MBAR"] * CM_PER_MBAR,
            "tv": values["TV"],
            "pbw": patient.predicted_body_weight(),
            "bmi": patient.bmi,
            "age": patient.age,
            "pft": pft_group(case),
            "legacy_core": is_legacy_core(case),
            "patient": patient,
        }

    with ThreadPoolExecutor(min(16, max(8, workers))) as pool:
        rows = [r for r in pool.map(measure, pool_items) if r is not None]

    def design(rs, peep=None):
        return np.column_stack([
            np.ones(len(rs)),
            [np.log(r["pbw"] / 60) for r in rs],
            [r["bmi"] - 23 for r in rs],
            [r["peep"] if peep is None else peep for r in rs],
        ])

    def log_c(rs):
        return np.log([r["c"] for r in rs])

    def _stratum(r):
        peep_bin = 0 if r["peep"] < 2 else (1 if r["peep"] < 4 else (2 if r["peep"] < 6 else 3))
        return (peep_bin, r["bmi"] >= 30.0)

    splits = kfold_splits(rows, n_folds=n_folds, seed=seed, group_key=lambda r: r["subjectid"], strata=_stratum)
    oof_measured: list[float] = []
    oof_constant: list[float] = []
    oof_scaled: list[float] = []
    oof_anasim: list[float] = []
    oof_rows: list[dict[str, Any]] = []
    for train_idx, val_idx in splits:
        train_rows = [rows[i] for i in train_idx]
        val_peep5 = [rows[i] for i in val_idx if 4 <= rows[i]["peep"] < 6]
        if not val_peep5:
            continue
        beta_fold = np.linalg.lstsq(design(train_rows), log_c(train_rows), rcond=None)[0]
        const_pool = [r["c"] for r in train_rows if 4 <= r["peep"] < 6]
        constant_fold = float(np.median(const_pool)) if const_pool else float(np.median([r["c"] for r in train_rows]))
        scaled_fold = np.exp(design(val_peep5, 5.0) @ beta_fold)
        oof_measured.extend(r["c"] for r in val_peep5)
        oof_constant.extend([constant_fold] * len(val_peep5))
        oof_scaled.extend(float(v) for v in scaled_fold)
        oof_rows.extend(val_peep5)

    sim_tasks = [(r["patient"], r["tv"]) for r in oof_rows]
    oof_anasim = _pool_map(executor, _compliance_sim_task, sim_tasks, chunksize=16)

    aer_tasks = [(r["patient"], r["tv"], r["peep"]) for r in rows]
    aer_results = _pool_map(executor, _aeration_mechanics_task, aer_tasks, chunksize=16)
    aer_c = np.array([c for c, _ in aer_results], dtype=float)
    aer_dp = np.array([dp for _, dp in aer_results], dtype=float)
    all_c = np.array([r["c"] for r in rows], dtype=float)
    all_dp = np.array([r["driving"] for r in rows], dtype=float)

    peep_aeration = {}
    for label, pred in (
        ("PEEP 0-2 cmH₂O (all)", lambda r: 0 <= r["peep"] < 2),
        ("PEEP 4-6 cmH₂O (all)", lambda r: 4 <= r["peep"] < 6),
        ("PEEP 0-2, BMI >= 30", lambda r: 0 <= r["peep"] < 2 and r["bmi"] >= 30.0),
        ("PEEP 4-6, BMI >= 30", lambda r: 4 <= r["peep"] < 6 and r["bmi"] >= 30.0),
        ("PEEP 4-6, PFT Normal", lambda r: 4 <= r["peep"] < 6 and r["pft"] == "Normal"),
        ("PEEP 4-6, PFT Obstructive", lambda r: 4 <= r["peep"] < 6 and r["pft"] == "Obstructive"),
    ):
        idx = [i for i, r in enumerate(rows) if pred(r)]
        if len(idx) >= 5:
            peep_aeration[label] = {
                "n": len(idx),
                "rec_c": [float(all_c[i]) for i in idx],
                "sim_c": [float(aer_c[i]) for i in idx],
                "c_mae": float(np.mean(np.abs(aer_c[idx] - all_c[idx]))),
                "c_bias": float(np.median(aer_c[idx] - all_c[idx])),
                "rec_dp": [float(all_dp[i]) for i in idx],
                "sim_dp": [float(aer_dp[i]) for i in idx],
                "dp_bias": float(np.median(aer_dp[idx] - all_dp[idx])),
            }

    measured = np.array(oof_measured)
    constant = np.array(oof_constant)
    scaled = np.array(oof_scaled)
    anasim = np.array(oof_anasim)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(500):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        boot.append(np.linalg.lstsq(design(sample), log_c(sample), rcond=None)[0])
    full = np.linalg.lstsq(design(rows), log_c(rows), rcond=None)[0]
    with_age = np.column_stack([design(rows), [r["age"] for r in rows]])
    age_beta = float(np.linalg.lstsq(with_age, log_c(rows), rcond=None)[0][-1])

    subgroups = {}
    for label, pred in (
        ("Legacy core (open supine, age ≤ 70, BMI ≤ 32)", lambda r: r["legacy_core"]),
        ("BMI under 30", lambda r: r["bmi"] < 30.0),
        ("BMI 30 and over", lambda r: r["bmi"] >= 30.0),
        ("PFT Normal", lambda r: r["pft"] == "Normal"),
        ("PFT Obstructive", lambda r: r["pft"] == "Obstructive"),
    ):
        idx = [i for i, r in enumerate(oof_rows) if pred(r)]
        if len(idx) >= 5:
            subgroups[label] = {
                "n": len(idx),
                "recorded": [float(measured[i]) for i in idx],
                "anasim_mae": float(np.mean(np.abs(anasim[idx] - measured[idx]))),
                "anasim_bias": float(np.median(anasim[idx] - measured[idx])),
            }

    return {
        "n": len(rows),
        "n_folds": n_folds,
        "beta": full,
        "ci": np.percentile(boot, [2.5, 97.5], axis=0),
        "age_beta": age_beta,
        "by_peep": {
            label: [r["c"] for r in rows if lo <= r["peep"] < hi]
            for label, (lo, hi) in {"0-2": (0, 2), "4-6": (4, 6)}.items()
        },
        "oof_peep5_n": len(measured),
        "mae": {
            "constant": float(np.mean(np.abs(constant - measured))),
            "scaled": float(np.mean(np.abs(scaled - measured))),
            "anasim": float(np.mean(np.abs(anasim - measured))),
        },
        "anasim_bias": float(np.median(anasim - measured)),
        "subgroups": subgroups,
        "peep_aeration": peep_aeration,
    }


def engine_eto2_gap(w: dict[str, Any]) -> tuple[float, float]:
    """Return AnaSim (FiO2, FiO2 - EtO2) in % after 10 minutes at the case's settings."""
    patient: Patient = w["patient"]
    engine = SimulationEngine(patient, SimulationConfig(mode="steady_state", rng_seed=1, dt=0.1))
    rr = w["SET_RR_IPPV"]
    ti_fraction = w["SET_INSP_TM"] * rr / 60
    engine.set_vent_settings(
        rr=rr, vt=w["TV"] / 1000, peep=w["SET_INTER_PEEP"],
        ie=ti_fraction / (1 - ti_fraction), mode="VCV",
    )
    engine.start()
    uptake = engine.resp.vco2 * max(0.5, engine.thermal.metabolic_factor) / engine.resp.rq / 1000.0
    recorded_fio2 = w["FIO2"] / 100
    fresh = recorded_fio2 + (1.0 - recorded_fio2) * uptake / 2.0
    o2 = min(2.0, max(0.0, (2.0 * fresh - 0.42) / 0.79))
    engine.set_fgf(o2, 2.0 - o2, 0.0)
    engine.circuit.equilibrate(uptake)
    for _ in range(6000):
        engine.step(0.1)
    return engine.state.fio2 * 100, engine.state.fio2 * 100 - engine.state.et_o2


# --- Report -----------------------------------------------------------------

def quartiles(values) -> str:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f"{q2:.3g} ({q1:.3g}–{q3:.3g})"


def median(values) -> float:
    return float(np.nanmedian(values))


def _fold_stability(choices: list[Any], unit: str = "") -> str:
    counts = Counter(choices)
    suffix = f" {unit}" if unit else ""
    parts = [f"{val:g}{suffix} ({n}/{len(choices)} folds)" for val, n in counts.most_common()]
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=60, help="VCV waveform windows (0 = all eligible)")
    parser.add_argument("--compliance-cases", type=int, default=0, help="Pre-incision compliance cases (0 = all)")
    parser.add_argument("--pcv-cases", type=int, default=40, help="PCV rise-time cases (0 = all)")
    parser.add_argument("--folds", type=int, default=5, help="Patient-grouped cross-validation folds")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--figure", type=Path, help="Write real and simulated breath overlays to this PNG")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    vitaldb_data.CACHE = args.cache

    production_capno = {name: getattr(Capnograph, name) for name, _, _ in CAPNOGRAPH_GRIDS}
    production_rise = ventilator.RISE_TIME
    production_tau = AirwaySensor().tau_s
    production_limb = ventilator.CIRCUIT_RESISTANCE

    candidates = domain_cases(TRACKS, open_supine_only=True, exclude_tracks=("Primus/SET_INSP_PRES",))
    random.Random(args.seed).shuffle(candidates)
    cases: list[dict[str, Any]] = []
    rejected = 0
    for case, trk in candidates:
        w = steady_window(case, trk)
        tpl = None if w is None else template(w["AWP"], w["SET_RR_IPPV"])
        if tpl is None:
            continue
        assert w is not None
        w["template"] = tpl
        w["features"] = window_features(w, tpl)
        if not w["features"]["volume_control"]:
            rejected += 1
            continue
        w["capnogram"] = capnogram_shape(cyclic_breath(w["CO2"], w["SET_RR_IPPV"]), 60 / w["SET_RR_IPPV"])
        cases.append(w)
        print(f"case {case['caseid']} ({len(cases)}/{args.cases or 'all'})", flush=True)
        if args.cases and len(cases) == args.cases:
            break

    splits = kfold_splits(
        cases,
        n_folds=args.folds,
        seed=args.seed,
        group_key=lambda w: w["case"].get("subjectid", w["case"]["caseid"]),
        strata=lambda w: w["patient"].bmi >= 30.0,
    )

    pool_ctx = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else nullcontext(None)
    with pool_ctx as executor:
        print("circuit limb", flush=True)
        limb_full = _compare_paw_batch(
            cases,
            [(r, True, production_tau, r) for r in LIMB_RESISTANCES],
            executor,
        )
        limb_per_case: dict[float, list[float]] = {r: limb_full[r]["whole"] for r in LIMB_RESISTANCES}
        limb_grid = {r: median(vals) for r, vals in limb_per_case.items()}
        limb_fold_choices = [
            min(LIMB_RESISTANCES, key=lambda r: median([limb_per_case[r][i] for i in train_idx]))
            for train_idx, _ in splits
        ]

        print("airway sensor & breath shape", flush=True)
        sensor_configs: list[tuple[Any, bool, float, float]] = [
            (("tau", tau), True, tau, production_limb)
            for tau in SENSOR_TAUS
            if not (tau == production_tau and production_limb in limb_full)
        ]
        sensor_configs.append(("single", False, production_tau, production_limb))
        sensor_batch = _compare_paw_batch(cases, sensor_configs, executor)
        sensor_per_case: dict[float, dict[str, list[float]]] = {}
        for tau in SENSOR_TAUS:
            if tau == production_tau and production_limb in limb_full:
                sensor_per_case[tau] = limb_full[production_limb]
            else:
                sensor_per_case[tau] = sensor_batch[("tau", tau)]
        sensor_grid = {tau: median(res["transitions"]) for tau, res in sensor_per_case.items()}
        sensor_fold_choices = [
            min(SENSOR_TAUS, key=lambda tau: median([sensor_per_case[tau]["transitions"][i] for i in train_idx]))
            for train_idx, _ in splits
        ]

        single_all = sensor_batch["single"]
        viscous_prod = sensor_per_case[production_tau]
        viscous_oof: dict[str, list[float]] = {k: [0.0] * len(cases) for k in _PAW_KEYS}
        for (_, val_idx), chosen_tau in zip(splits, sensor_fold_choices, strict=True):
            for i in val_idx:
                for k in _PAW_KEYS:
                    viscous_oof[k][i] = sensor_per_case[chosen_tau][k][i]

        print("capnograph (5-fold CV + full fit)", flush=True)
        capno_cache: dict[tuple[int, tuple[tuple[str, float], ...]], dict[str, float]] = {}
        capno_oof: list[dict[str, float]] = [{} for _ in cases]
        capno_fold_choices: dict[str, list[float]] = {name: [] for name, _, _ in CAPNOGRAPH_GRIDS}
        oof_pending: list[tuple[int, dict[str, float], tuple[int, tuple[tuple[str, float], ...]]]] = []
        for train_idx, val_idx in splits:
            fold_params, _ = fit_capnograph_subset(cases, train_idx, production_capno, capno_cache, executor=executor)
            for name in capno_fold_choices:
                capno_fold_choices[name].append(fold_params[name])
            key_tail = tuple(sorted(fold_params.items()))
            for i in val_idx:
                oof_pending.append((i, fold_params, (i, key_tail)))
        _populate_capno_cache(cases, oof_pending, capno_cache, executor)
        for i, _params, cache_key in oof_pending:
            capno_oof[i] = capno_cache[cache_key]
        full_capno_params, capno_grids = fit_capnograph_subset(
            cases, list(range(len(cases))), production_capno, capno_cache, executor=executor,
        )

        print("pressure control rise", flush=True)
        pcv_recorded, pcv_simulated = pcv_rise_time(args.pcv_cases, executor=executor)

        print("compliance by patient", flush=True)
        compliance = compliance_by_patient(
            args.compliance_cases, n_folds=args.folds, workers=args.workers, executor=executor,
        )

        print("end-tidal O2", flush=True)
        eto2_inputs = [_eto2_payload(w) for w in cases]
        eto2_results = _pool_map(executor, engine_eto2_gap, eto2_inputs, chunksize=2)
        fio2_sim, gap_sim = zip(*eto2_results, strict=True)
    gap_real = [w["FIO2"] - w["FEO2"] for w in cases]
    paired = np.array(gap_sim) - np.array(gap_real)

    default = Patient()
    feats = [w["features"] for w in cases]
    beta, ci = compliance["beta"], compliance["ci"]
    improved = sum(v < s for s, v in zip(single_all["whole"], viscous_oof["whole"], strict=True))
    settings = ventilator.VentSettings()
    n_legacy = sum(1 for w in cases if is_legacy_core(w["case"]))
    lines = [
        "# VitalDB ventilation comparisons",
        "",
        f"Generated {datetime.now(timezone.utc).date().isoformat()}.",
        "",
        f"These comparisons use {len(cases)} volume-control windows recorded on the Dräger Primus, from {len(candidates)} eligible",
        "unique-subject cases inside AnaSim's patient domain (general anesthesia, oral single-lumen tube, open surgery,",
        f"supine; {n_legacy} fall inside the prior age ≤ 70 / BMI ≤ 32 subset). Each case contributes one steady window",
        f"during surgery. Another {rejected} windows were skipped because breaths did not peak at end-flow and exhale at",
        "the set inspiratory time (pressure-regulated breaths) or gave implausible mechanics.",
        f"Fitted values use all {len(cases)} windows and are evaluated out-of-fold via {args.folds}-fold patient-grouped",
        "cross-validation. Values are median (IQR).",
        "",
        "Source: VitalDB (Lee et al., *Scientific Data*, 2022), used under CC BY-NC-SA 4.0 and its data use agreement.",
        "",
        "## Settings",
        "",
        "| Setting | Recordings | AnaSim default |",
        "|---|---|---|",
        f"| Inspiratory pause (% of Ti) | {quartiles([w['SET_INSP_PAUSE'] for w in cases])} | {settings.pause:g} |",
        f"| Pmax (cmH₂O) | {quartiles([w['SET_PIP'] * CM_PER_MBAR for w in cases])} | {settings.p_max:g} |",
        "",
        "## Mechanics from the recorded airway pressure",
        "",
        "Each case's resistance and compliance are set so that AnaSim's breath has the recorded end-of-flow",
        "drop and plateau. The other quantities are then predictions. Tissue viscoelasticity uses Jonson 1993",
        "(viscoelastic compliance four times static compliance, time constant 0.82 s) without fitting.",
        "",
        "| Quantity | Recordings | Single compartment | With tissue viscoelasticity |",
        "|---|---|---|---|",
        f"| Resistance from end-of-flow drop (cmH₂O/(L/s)) | {quartiles([f['r'] for f in feats])} | matched | matched |",
        f"| Plateau compliance (mL/cmH₂O) | {quartiles([f['c'] for f in feats])} | matched | matched |",
        f"| Expiratory tail decay / RC | {quartiles([f['tau_e_ratio'] for f in feats])} | "
        f"{quartiles(single_all['tau_e_ratio'])} | {quartiles(viscous_prod['tau_e_ratio'])} |",
        f"| Expiratory limb resistance (cmH₂O/(L/s)) | {quartiles([f['r_exp'] for f in feats])} | "
        f"{quartiles(single_all['r_exp'])} | {quartiles(viscous_prod['r_exp'])} |",
        f"| Inspiratory knee time constant (ms) | {quartiles([1000 * f['tau_knee'] for f in feats])} | "
        f"{quartiles(1000 * np.array(single_all['knee']))} | {quartiles(1000 * np.array(viscous_prod['knee']))} |",
        "",
        f"Tail values use windows whose expiratory tail reached 0.5 cmH₂O ({sum(np.isfinite(f['tau_e']) for f in feats)}"
        " recorded). In the other windows, Paw returned to PEEP within 0.2 s.",
        "",
        "Circuit limb resistance fitted on all windows (whole-breath RMSE): "
        + ", ".join(f"{k:g} {v:.3f}" for k, v in limb_grid.items())
        + f". Best {min(limb_grid, key=lambda k: limb_grid[k]):g}; CV stability: "
        + f"{_fold_stability(limb_fold_choices, 'cmH₂O/(L/s)')}; AnaSim uses {production_limb:g} cmH₂O/(L/s).",
        "",
        "## Breath shape (RMSE, cmH₂O)",
        "",
        "Airway sensor time constant fitted on all windows (transition RMSE): "
        + ", ".join(f"{1000 * k:g} ms {v:.2f}" for k, v in sensor_grid.items())
        + f". Best {1000 * min(sensor_grid, key=lambda k: sensor_grid[k]):g} ms; CV stability: "
        + f"{_fold_stability([1000 * t for t in sensor_fold_choices], 'ms')}; AnaSim uses {1000 * production_tau:g} ms.",
        "",
        f"| Segment ({args.folds}-fold out-of-fold, n={len(cases)}) | Single compartment | With tissue viscoelasticity |",
        "|---|---|---|",
        *(f"| {k} | {median(single_all[k]):.2f} | {median(viscous_oof[k]):.2f} |" for k in (*SEGMENTS, "whole")),
        "",
        f"Tissue viscoelasticity lowered whole-breath RMSE in {improved} of {len(cases)} out-of-fold windows.",
        "",
        "## Pressure-control rise",
        "",
        f"10-90% Paw rise of {len(pcv_recorded)} recorded pressure-control breaths: {quartiles(pcv_recorded)} s.",
        "AnaSim through the airway sensor: "
        + ", ".join(f"{k:g} s ramp {v:.3f} s" for k, v in pcv_simulated.items())
        + f". AnaSim uses a {production_rise:g} s ramp.",
        "",
        "## Capnogram shape",
        "",
        "Features come from the CO₂ track alone. Each capnograph parameter was fitted in turn on all windows",
        "to the one feature that it sets (median absolute error): "
        + "; ".join(
            f"{name} " + ", ".join(f"{k:g}: {v:.3f}" for k, v in grid.items()) + f" (best {full_capno_params[name]:g})"
            for name, grid in capno_grids.items()
        )
        + ".",
        f"CV stability ({args.folds} folds): "
        + "; ".join(f"{name}: {_fold_stability(choices)}" for name, choices in capno_fold_choices.items())
        + ".",
        f"AnaSim uses ANALYZER_TAU {production_capno['ANALYZER_TAU']:g} s, PHASE_II_VOLUME {production_capno['PHASE_II_VOLUME']:g} L,",
        f"and PHASE_III_SLOPE {production_capno['PHASE_III_SLOPE']:g} mmHg/L. The out-of-fold check below uses each",
        "fold's training-fitted values and AnaSim's PBW-based series dead space (2.2 mL/kg PBW, unfitted).",
        "",
        f"| Feature ({args.folds}-fold out-of-fold, n={len(cases)}) | Recordings | AnaSim | Paired difference |",
        "|---|---|---|---|",
        *(
            f"| {label} | {quartiles([w['capnogram'][key] for w in cases])} | "
            f"{quartiles([s[key] for s in capno_oof])} | "
            f"{quartiles([s[key] - w['capnogram'][key] for s, w in zip(capno_oof, cases, strict=True)])} |"
            for key, label in (
                ("fall", "Fall 90-10% (s)"),
                ("rise", "Rise 10-90% (s)"),
                ("duty", "Share of breath above 50%"),
                ("slope", "Phase III slope (% of plateau/s)"),
            )
        ),
        "",
        "## Compliance by patient",
        "",
        f"Plateau compliance TV/(Pplat - PEEP) before incision in {compliance['n']} supine cases from numeric tracks:",
        f"PEEP 0-2 {quartiles(compliance['by_peep']['0-2'])}, PEEP 4-6 {quartiles(compliance['by_peep']['4-6'])}"
        " mL/cmH₂O.",
        "",
        "log C = a + b log(PBW/60) + c (BMI - 23) + d PEEP over all cases (bootstrap 95% CI):",
        f"b = {beta[1]:.2f} ({ci[0][1]:.2f} to {ci[1][1]:.2f}), c = {beta[2]:.4f} ({ci[0][2]:.4f} to {ci[1][2]:.4f}) per kg/m²,",
        f"d = {beta[3]:.3f} ({ci[0][3]:.3f} to {ci[1][3]:.3f}) per cmH₂O. C at PBW 60 kg, BMI 23, PEEP 5:",
        f"{math.exp(beta[0] + 5 * beta[3]):.1f} mL/cmH₂O. Age adds {compliance['age_beta']:.4f} per year.",
        "",
        f"{args.folds}-fold out-of-fold cases at PEEP 4-6 (n = {compliance['oof_peep5_n']}), mean absolute error in mL/cmH₂O:",
        f"constant {compliance['mae']['constant']:.2f}, body-size fit {compliance['mae']['scaled']:.2f},",
        f"AnaSim {compliance['mae']['anasim']:.2f} (median bias {compliance['anasim_bias']:+.2f}).",
        f"AnaSim's default patient: static {1000 * default.respiratory_compliance():.1f}, plateau "
        f"{anasim_plateau_compliance(default, 500):.1f} mL/cmH₂O.",
        "",
    ]
    if compliance["subgroups"]:
        lines += [
            "| Compliance subgroup (PEEP 4-6) | n | Recorded (mL/cmH₂O) | AnaSim MAE | AnaSim median bias |",
            "|---|---:|---|---:|---:|",
            *(
                f"| {label} | {sg['n']} | {quartiles(sg['recorded'])} | {sg['anasim_mae']:.2f} | {sg['anasim_bias']:+.2f} |"
                for label, sg in compliance["subgroups"].items()
            ),
            "",
        ]
    if compliance.get("peep_aeration"):
        lines += [
            "### Settled `LungAeration` compliance and driving pressure across PEEP",
            "",
            "Each supine case is settled for 5 min at its recorded pre-incision PEEP and tidal volume through",
            "`LungAeration` and `RespiratoryMechanics` (unfitted against PEEP 0-2).",
            "",
            "| Subgroup | n | Recorded C (mL/cmH₂O) | AnaSim C (mL/cmH₂O) | C MAE | C bias | Recorded ΔP (cmH₂O) | AnaSim ΔP (cmH₂O) | ΔP bias |",
            "|---|---:|---|---|---:|---:|---|---|---:|",
            *(
                f"| {label} | {sg['n']} | {quartiles(sg['rec_c'])} | {quartiles(sg['sim_c'])} | "
                f"{sg['c_mae']:.2f} | {sg['c_bias']:+.2f} | {quartiles(sg['rec_dp'])} | "
                f"{quartiles(sg['sim_dp'])} | {sg['dp_bias']:+.2f} |"
                for label, sg in compliance["peep_aeration"].items()
            ),
            "",
        ]
    r_rec = np.array([w["features"]["r"] for w in cases], dtype=float)
    r_sim = np.array([w["patient"].airway_resistance() for w in cases], dtype=float)
    r_subgroups = []
    for label, pred in (
        ("All VCV windows", lambda w: True),
        ("BMI under 30", lambda w: w["patient"].bmi < 30.0),
        ("BMI 30 and over", lambda w: w["patient"].bmi >= 30.0),
        ("PFT Normal", lambda w: pft_group(w["case"]) == "Normal"),
        ("PFT Obstructive", lambda w: pft_group(w["case"]) == "Obstructive"),
    ):
        idx = [i for i, w in enumerate(cases) if pred(w)]
        if len(idx) >= 3:
            r_subgroups.append((
                label,
                len(idx),
                quartiles(r_rec[idx]),
                quartiles(r_sim[idx]),
                float(np.mean(np.abs(r_sim[idx] - r_rec[idx]))),
                float(np.median(r_sim[idx] - r_rec[idx])),
            ))
    if r_subgroups:
        lines += [
            "## Airway resistance by patient (VCV waveform cohort)",
            "",
            "Recorded inspiratory resistance from the waveform end-of-flow drop versus `Patient.airway_resistance()`.",
            "",
            "| Resistance subgroup | n | Recorded R (cmH₂O/(L/s)) | AnaSim R (cmH₂O/(L/s)) | MAE | Median bias |",
            "|---|---:|---|---|---:|---:|",
            *(
                f"| {label} | {n} | {rec_q} | {sim_q} | {mae:.2f} | {bias:+.2f} |"
                for label, n, rec_q, sim_q, mae, bias in r_subgroups
            ),
            "",
        ]
    lines += [
        "## End-tidal O₂",
        "",
        "| | Recordings | AnaSim |",
        "|---|---|---|",
        f"| FiO₂ (%) | {quartiles([w['FIO2'] for w in cases])} | {quartiles(fio2_sim)} |",
        f"| FiO₂ - EtO₂ (%) | {quartiles(gap_real)} | {quartiles(gap_sim)} |",
        "",
        f"Paired difference, AnaSim minus recorded: {quartiles(paired)}; mean absolute {np.mean(np.abs(paired)):.2f}",
        "percentage points (the Primus reports whole percent).",
        "",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "vitaldb_ventilation.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))

    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 4, figsize=(16, 9))
        for ax, w in zip(axes.flat, cases, strict=False):
            t = (np.arange(len(w["template"])) - LEAD) * WAVE_DT
            ax.plot(t, w["template"], "k", lw=1.5, label="Primus")
            for viscoelastic, color, label in ((False, "C0", "Single compartment"), (True, "C3", "Viscoelastic")):
                resistance, compliance_l = matched_mechanics(w, viscoelastic, production_tau)
                sim = template(
                    simulate(w, resistance, compliance_l, viscoelastic=viscoelastic, sensor_tau=production_tau)[0],
                    w["SET_RR_IPPV"],
                )
                assert sim is not None
                ax.plot((np.arange(len(sim)) - LEAD) * WAVE_DT, sim, color, lw=1.0, label=label)
            ax.set_xlim(-0.1, w["SET_INSP_TM"] + 1.2)
            f = w["features"]
            ax.set_title(f"R {f['r']:.1f}  C {f['c']:.0f}  Ti {w['SET_INSP_TM']} s", fontsize=9)
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=80)


if __name__ == "__main__":
    main()
