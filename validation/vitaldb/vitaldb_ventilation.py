"""Compare AnaSim ventilation and capnography with Dräger Primus recordings from VitalDB.

Volume-control windows are steady minutes of volume-controlled ventilation in
adults inside AnaSim's patient domain (general anesthesia, oral tube, open
surgery, supine). The script measures resistance, compliance, and waveform
shape from the recorded airway pressure and CO2, then runs each breath through
AnaSim's ventilator, lung, sensor, and capnograph code with that patient's
settings. It fits the pressure rise of pressure-control breaths, the dependence
of plateau compliance on body size from numeric tracks of several hundred
cases, and compares FiO2 - EtO2 from the full engine with the recorded
FiO2 - FeO2.

Fitted values come from a calibration half and are checked on the held-out half.

Data: VitalDB (Lee et al., Sci Data 2022; CC BY-NC-SA 4.0 with a data use
agreement). Tracks download on first use into validation/vitaldb/.vitaldb_cache.

    python validation/vitaldb/vitaldb_ventilation.py [--cases 40] [--compliance-cases 600] [--figure overlay.png]

Writes validation/vitaldb/results/vitaldb_ventilation.md.
"""

import argparse
import csv
import gzip
import io
import math
import random
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))

import anasim.machine.ventilator as ventilator  # noqa: E402
from anasim.core.engine import SimulationEngine  # noqa: E402
from anasim.core.state import SimulationConfig  # noqa: E402
from anasim.machine.ventilator import AnesthesiaVentilator  # noqa: E402
from anasim.monitors.airway import AirwaySensor  # noqa: E402
from anasim.monitors.capno import Capnograph  # noqa: E402
from anasim.patient.patient import Patient  # noqa: E402
from anasim.physiology.resp_mech import RespiratoryMechanics  # noqa: E402

CACHE = ROOT / ".vitaldb_cache"
API = "https://api.vitaldb.net"
WAVE_DT = 1 / 62.5  # Primus waveform interval (s)
CM_PER_MBAR = 1.0197
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


# --- VitalDB ------------------------------------------------------------

def fetch(name: str, url: str) -> str:
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                path.write_bytes(response.read())
        except urllib.error.URLError:
            # Some Python installs lack CA certificates; curl uses the system store.
            subprocess.run(["curl", "-sf", "-m", "180", "-o", str(path), url], check=True)
    raw = path.read_bytes()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    return raw.decode("utf-8-sig")


def load_track(tid: str, wave: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Waveform rows carry a time only at segment starts; numeric rows always do."""
    times, values = [], []
    t = None
    for line in fetch(f"{tid}.csv", f"{API}/{tid}").splitlines()[1:]:
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
    return np.array(times), np.array(values)


def domain_cases(required: list[str], typical_only: bool = True) -> list[tuple[dict, dict]]:
    """Adults in AnaSim's domain under general anesthesia with an oral tube, open surgery, supine.

    typical_only=False keeps every general anesthetic with an oral tube, for device behavior.
    """
    tracks: dict[int, dict] = {}
    for row in csv.DictReader(io.StringIO(fetch("trks.csv", f"{API}/trks"))):
        tracks.setdefault(int(row["caseid"]), {})[row["tname"]] = row["tid"]
    out = []
    for case in csv.DictReader(io.StringIO(fetch("cases.csv", f"{API}/cases"))):
        trk = tracks.get(int(case["caseid"]), {})
        if not typical_only:
            if case["ane_type"] == "General" and case["airway"] == "Oral" and all(n in trk for n in required):
                out.append((case, trk))
            continue
        try:
            age, weight, height = float(case["age"]), float(case["weight"]), float(case["height"])
        except ValueError:
            continue
        in_domain = 18 <= age <= 70 and 50 <= weight <= 100 and 150 <= height <= 200
        in_domain = in_domain and 18 <= weight / (height / 100) ** 2 <= 32
        typical = (
            case["ane_type"] == "General" and not case["dltubesize"] and case["airway"] == "Oral"
            and case["approach"] == "Open" and case["position"] == "Supine"
        )
        if in_domain and typical and all(name in trk for name in required):
            out.append((case, trk))
    return out


def window_settings(trk: dict, names: list[str], t0: float, t1: float) -> dict | None:
    """Median of each numeric track in [t0, t1]; None if missing or a setting changed."""
    window = {}
    for name in names:
        t, v = load_track(trk[name])
        v = v[(t >= t0) & (t <= t1)]
        if len(v) == 0 or (name.startswith("Primus/SET_") and np.ptp(v) > 1e-6):
            return None
        window[name.split("/")[1]] = float(np.median(v))
    return window


def steady_window(case: dict, trk: dict, half_s: float = 60.0) -> dict | None:
    """Two minutes at mid-surgery with fixed settings and only machine breaths."""
    t_mid = (float(case["opstart"]) + float(case["opend"])) / 2
    t0, t1 = t_mid - half_s, t_mid + half_s
    window = window_settings(trk, [name for name in TRACKS if name not in WAVES], t0, t1)
    if window is None or abs(window["RR_CO2"] - window["SET_RR_IPPV"]) > 1.0:
        return None
    for name in WAVES:
        t, v = load_track(trk[name], wave=True)
        inside = (t >= t0) & (t <= t1)
        if inside.sum() < 0.9 * (t1 - t0) / WAVE_DT:
            return None
        window[name.split("/")[1]] = v[inside] * (CM_PER_MBAR if name == "Primus/AWP" else 1.0)
    window["case"] = case
    return window


# --- Breath features ------------------------------------------------------

def template(signal: np.ndarray, rr: float) -> np.ndarray | None:
    """Median breath aligned where pressure leaves PEEP."""
    n = round(60.0 / rr / WAVE_DT)
    peep, peak = np.percentile(signal, 10), np.percentile(signal, 99)
    level = peep + 0.3 * (peak - peep)
    breaths = []
    for i in np.flatnonzero((signal[:-1] < level) & (signal[1:] >= level)):
        j = i
        while j > 0 and signal[j - 1] > peep + 0.3 and i - j < 20:
            j -= 1
        if j >= LEAD and j - LEAD + n <= len(signal):
            breaths.append(signal[j - LEAD:j - LEAD + n])
    return np.median(np.array(breaths), axis=0) if len(breaths) >= 5 else None


def features(tpl: np.ndarray, ti: float, vt_ml: float, pause: float) -> dict:
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

    def onset_error(params):
        tau, shift = params
        s = np.clip(t[window] - shift, 0, None)
        return peep + r * flow * (1 - np.exp(-s / tau)) + flow * s * 1000 / c - tpl[window]

    tau_knee = least_squares(onset_error, [0.05, 0.0], bounds=([0.002, -0.03], [0.5, 0.05])).x[0]
    # Expiratory tail: P - PEEP = A exp(-t / tau_e), with A = R_exp x initial expiratory flow.
    tail = (t >= ti + 0.2) & (t <= ti + 2.0)
    excess = tpl[tail] - peep
    good = excess > 0.05
    tau_e = amplitude = np.nan
    if good.sum() >= 5:
        slope, intercept = np.polyfit(t[tail][good] - ti, np.log(excess[good]), 1)
        # Only a tail of at least 0.5 cmH2O gives usable values.
        if slope < 0 and np.exp(intercept) >= 0.5:
            tau_e, amplitude = -1 / slope, np.exp(intercept)
    after_ti = tpl[np.searchsorted(t, ti + 0.2)]
    # Constant-flow breaths peak as flow stops and exhale at the set Ti. Dräger
    # AutoFlow (pressure-regulated) breaths record no pressure setting and fail this.
    volume_control = (
        abs(t[insp][np.argmax(tpl[insp])] - t_flow) <= 0.1
        and after_ti < peep + 0.5 * (pplat - peep)
        and 1.0 <= r <= 40.0 and 10.0 <= c <= 150.0
    )
    return {
        "peep": peep, "r": r, "c": c, "tau_knee": tau_knee, "tau_e": tau_e,
        "tau_e_ratio": tau_e / (r * c / 1000) if r > 0 and c > 0 else np.nan, "r_exp": amplitude * r * c / vt_ml, "t_flow": t_flow,
        "volume_control": volume_control,
    }


def window_features(w: dict, tpl: np.ndarray) -> dict:
    return features(tpl, w["SET_INSP_TM"], w["TV"], w["SET_INSP_PAUSE"] / 100)


def segment_rmse(sim: np.ndarray, real: np.ndarray, ti: float, t_flow: float) -> dict:
    t = (np.arange(len(real)) - LEAD) * WAVE_DT
    masks = {
        "onset": (t >= -0.05) & (t <= 0.35), "ramp": (t > 0.35) & (t < t_flow - 0.05),
        "end_flow": (t >= t_flow - 0.05) & (t <= ti), "exp_fall": (t > ti) & (t <= ti + 0.4),
        "exp_tail": (t > ti + 0.4) & (t <= ti + 2.0),
    }
    n = min(len(sim), len(real))
    return {k: float(np.sqrt(np.mean((sim[:n] - real[:n])[m[:n]] ** 2))) for k, m in masks.items()}


def capnogram_shape(co2: np.ndarray, period: float) -> dict:
    """Features of one cyclic CO2 breath measured within the CO2 track alone.

    VitalDB stores AWP and CO2 as separate tracks whose relative timing varies
    by case, so nothing here refers to airway pressure.
    """
    n = len(co2)
    lo, hi = np.percentile(co2, 3), np.percentile(co2, 97)
    amplitude = hi - lo
    above = co2 > lo + 0.5 * amplitude
    fall_end = np.flatnonzero(above & ~np.roll(above, -1))[0]
    y = np.roll(co2, -(fall_end + 1) + 40)  # Start 40 samples before the fall crosses 50%
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
        "slope": 100 * slope,  # % of the plateau per second
    }


def cyclic_breath(signal: np.ndarray, rr: float) -> np.ndarray:
    """Median of consecutive breath-length windows: one breath, starting anywhere."""
    n = round(60.0 / rr / WAVE_DT)
    return np.median(signal[: len(signal) // n * n].reshape(-1, n), axis=0)


# --- AnaSim -----------------------------------------------------------------

def simulate(w: dict, resistance: float, compliance: float, *, viscoelastic: bool = True,
             sensor_tau: float | None = None, capnograph: Capnograph | None = None,
             settle: float = 20.0, substeps: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Paw and CO2 at the Primus rate for six breaths after settle seconds, from AnaSim's
    ventilator, lung, sensor, and capnograph."""
    lung = RespiratoryMechanics(compliance=compliance, resistance=resistance)
    if not viscoelastic:
        lung.viscoelastic_ratio = 0.0
    vent = AnesthesiaVentilator()
    rr = w["SET_RR_IPPV"]
    ti_fraction = w["SET_INSP_TM"] * rr / 60
    vent.update_settings(rr=rr, tv=w["TV"], peep=w["features"]["peep"], ie=ti_fraction / (1 - ti_fraction),
                         mode="VCV", pause=w["SET_INSP_PAUSE"])
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


def matched_mechanics(w: dict, viscoelastic: bool, sensor_tau: float) -> tuple[float, float]:
    """Resistance and compliance whose simulated breath has the case's end-of-flow drop and plateau.

    Both lung models get the same two measured numbers per case.
    """
    f = w["features"]
    target = np.log([f["r"], f["c"]])

    def residual(x):
        paw, _ = simulate(w, math.exp(x[0]), math.exp(x[1]) / 1000, viscoelastic=viscoelastic,
                          sensor_tau=sensor_tau, settle=12.0)
        g = window_features(w, template(paw, w["SET_RR_IPPV"]))
        return np.log([max(g["r"], 0.1), max(g["c"], 1.0)]) - target

    x = least_squares(residual, target, diff_step=1e-3, xtol=1e-4, ftol=1e-4).x
    return math.exp(x[0]), math.exp(x[1]) / 1000


def compare_paw(cases: list[dict], viscoelastic: bool, sensor_tau: float) -> dict:
    """Per-case errors and predicted features with matched mechanics."""
    results: dict[str, list] = {k: [] for k in (*SEGMENTS, "whole", "transitions", "knee", "tau_e_ratio", "r_exp")}
    for w in cases:
        resistance, compliance = matched_mechanics(w, viscoelastic, sensor_tau)
        sim = template(simulate(w, resistance, compliance, viscoelastic=viscoelastic, sensor_tau=sensor_tau)[0],
                       w["SET_RR_IPPV"])
        g = window_features(w, sim)
        results["knee"].append(g["tau_knee"])
        results["tau_e_ratio"].append(g["tau_e_ratio"])
        results["r_exp"].append(g["r_exp"])
        seg = segment_rmse(sim, w["template"], w["SET_INSP_TM"], w["features"]["t_flow"])
        for k, v in seg.items():
            results[k].append(v)
        n = min(len(sim), len(w["template"]))
        results["whole"].append(float(np.sqrt(np.mean((sim[:n] - w["template"][:n]) ** 2))))
        results["transitions"].append(float(np.sqrt(np.mean([seg[k] ** 2 for k in TRANSITIONS]))))
    return results


def capnogram_features(cases: list[dict]) -> list[dict]:
    """AnaSim capnogram features for each case with its settings, mechanics, and EtCO2."""
    out = []
    for w in cases:
        f = w["features"]
        capnograph = Capnograph(2.2 * float(w["case"]["weight"]) / 1000)
        _, co2 = simulate(w, f["r"], f["c"] / 1000, capnograph=capnograph, substeps=2)
        out.append(capnogram_shape(cyclic_breath(co2, w["SET_RR_IPPV"]), 60 / w["SET_RR_IPPV"]))
    return out


def fit_capnograph(calibration: list[dict]) -> dict:
    """Fit each capnograph parameter to the feature it alone sets, in turn: median absolute error."""
    grids = {}
    for name, values, feature in CAPNOGRAPH_GRIDS:
        recorded = np.array([w["capnogram"][feature] for w in calibration])
        grid = {}
        for value in values:
            setattr(Capnograph, name, value)
            simulated = np.array([s[feature] for s in capnogram_features(calibration)])
            grid[value] = float(np.nanmedian(np.abs(simulated - recorded)))
        setattr(Capnograph, name, min(grid, key=grid.get))
        grids[name] = grid
    return grids


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


def pcv_rise_time(cases: int) -> tuple[list, dict]:
    """10-90% Paw rise of recorded pressure-control breaths, and AnaSim's for each pressure ramp."""
    candidates = domain_cases(PCV_TRACKS, typical_only=False)
    random.Random(4).shuffle(candidates)
    recorded = []
    for case, trk in candidates:
        t_mid = (float(case["opstart"]) + float(case["opend"])) / 2
        settings = window_settings(trk, PCV_TRACKS[1:], t_mid - 60, t_mid + 60)
        if settings is None:
            continue
        t, p = load_track(trk["Primus/AWP"], wave=True)
        p = p[(t >= t_mid - 60) & (t <= t_mid + 60)] * CM_PER_MBAR
        tpl = template(p, settings["SET_RR_IPPV"]) if len(p) > 3000 else None
        rise = np.nan if tpl is None else rise_10_90(tpl, settings["SET_INSP_TM"])
        if np.isfinite(rise):
            recorded.append(rise)
            print(f"pressure control case {case['caseid']} ({len(recorded)}/{cases})", flush=True)
        if len(recorded) == cases:
            break
    simulated = {}
    for ramp in RISE_TIMES:
        ventilator.RISE_TIME = ramp
        lung = RespiratoryMechanics(compliance=0.045, resistance=10.0)
        vent = AnesthesiaVentilator()
        vent.update_settings(rr=12, peep=5, p_insp=18, ie="1:2", mode="PCV")
        sensor = AirwaySensor()
        dt = WAVE_DT / 4
        paw = []
        for k in range(int(45 / dt)):
            before = vent.volume
            vent.step(dt, lung, "vent")
            p = sensor.step(dt, vent.paw, vent.volume, vent.volume - before)[0]
            if k % 4 == 3:
                paw.append(p)
        simulated[ramp] = rise_10_90(template(np.array(paw[int(10 / WAVE_DT):]), 12), 5 / 3)
    return recorded, simulated


def anasim_plateau_compliance(patient: Patient, tv_ml: float) -> float:
    """VT/(Pplat - PEEP) from AnaSim at PEEP 5, RR 12, I:E 1:2, and a 10% pause."""
    lung = RespiratoryMechanics(compliance=patient.respiratory_compliance())
    vent = AnesthesiaVentilator()
    vent.update_settings(rr=12, tv=tv_ml, peep=5, ie="1:2", mode="VCV", pause=10)
    for _ in range(6):
        vent.step(5.0, lung, "vent")
    return vent.monitors.tv_exp / (vent.monitors.paw_plat - vent.monitors.peep)


def compliance_by_patient(cases: int) -> dict:
    """Plateau compliance before incision from numeric tracks, and its dependence on body size."""
    candidates = domain_cases(COMPLIANCE_TRACKS)
    random.Random(2).shuffle(candidates)

    def measure(item):
        case, trk = item
        if "Primus/SET_INSP_PRES" in trk:
            return None
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
        patient = Patient(age=float(case["age"]), weight=float(case["weight"]), height=float(case["height"]),
                          sex="male" if case["sex"] == "M" else "female")
        return {"c": values["TV"] / driving, "peep": values["PEEP_MBAR"] * CM_PER_MBAR, "tv": values["TV"],
                "pbw": patient.predicted_body_weight(), "bmi": patient.bmi, "age": patient.age, "patient": patient}

    with ThreadPoolExecutor(8) as pool:
        rows = [r for r in pool.map(measure, candidates[:cases]) if r is not None]
    rng = np.random.default_rng(3)
    order = rng.permutation(len(rows))
    calibration, held_out = [rows[i] for i in order[::2]], [rows[i] for i in order[1::2]]

    def design(rs, peep=None):
        return np.column_stack([np.ones(len(rs)), [np.log(r["pbw"] / 60) for r in rs], [r["bmi"] - 23 for r in rs],
                                [r["peep"] if peep is None else peep for r in rs]])

    def log_c(rs):
        return np.log([r["c"] for r in rs])

    beta = np.linalg.lstsq(design(calibration), log_c(calibration), rcond=None)[0]
    boot = []
    for _ in range(500):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        boot.append(np.linalg.lstsq(design(sample), log_c(sample), rcond=None)[0])
    full = np.linalg.lstsq(design(rows), log_c(rows), rcond=None)[0]
    with_age = np.column_stack([design(rows), [r["age"] for r in rows]])
    age_beta = np.linalg.lstsq(with_age, log_c(rows), rcond=None)[0][-1]
    peep5 = [r for r in held_out if 4 <= r["peep"] < 6]
    measured = np.array([r["c"] for r in peep5])
    constant = np.median([r["c"] for r in calibration if 4 <= r["peep"] < 6])
    scaled = np.exp(design(peep5, 5.0) @ beta)
    anasim = np.array([anasim_plateau_compliance(r["patient"], r["tv"]) for r in peep5])
    return {
        "n": len(rows), "beta": full, "ci": np.percentile(boot, [2.5, 97.5], axis=0), "age_beta": age_beta,
        "by_peep": {label: [r["c"] for r in rows if lo <= r["peep"] < hi]
                    for label, (lo, hi) in {"0-2": (0, 2), "4-6": (4, 6)}.items()},
        "held_out": len(peep5),
        "mae": {"constant": float(np.mean(np.abs(constant - measured))),
                "scaled": float(np.mean(np.abs(scaled - measured))),
                "anasim": float(np.mean(np.abs(anasim - measured)))},
        "anasim_bias": float(np.median(anasim - measured)),
    }


def engine_eto2_gap(w: dict) -> tuple[float, float]:
    """Return AnaSim (FiO2, FiO2 - EtO2) in % after 10 minutes at the case's settings."""
    case = w["case"]
    patient = Patient(
        age=float(case["age"]), weight=float(case["weight"]), height=float(case["height"]),
        sex="male" if case["sex"] == "M" else "female",
    )
    engine = SimulationEngine(patient, SimulationConfig(mode="steady_state", rng_seed=1, dt=0.1))
    rr = w["SET_RR_IPPV"]
    ti_fraction = w["SET_INSP_TM"] * rr / 60
    engine.set_vent_settings(rr=rr, vt=w["TV"] / 1000, peep=w["SET_INTER_PEEP"], ie=ti_fraction / (1 - ti_fraction),
                             mode="VCV")
    engine.start()
    # 2 L/min of O2 and air whose steady circle composition matches the recorded
    # FiO2: fresh O2 fraction = FiO2 + O2 uptake / fresh gas flow.
    uptake = engine.resp.vco2 * max(0.5, engine.thermal.metabolic_factor) / engine.resp.rq / 1000.0
    fresh = w["FIO2"] / 100 + uptake / 2.0
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--compliance-cases", type=int, default=600)
    parser.add_argument("--pcv-cases", type=int, default=20)
    parser.add_argument("--figure", type=Path, help="Write real and simulated breath overlays to this PNG")
    args = parser.parse_args()
    production = {name: getattr(Capnograph, name) for name, _, _ in CAPNOGRAPH_GRIDS}
    production_rise = ventilator.RISE_TIME
    production_tau = AirwaySensor().tau_s

    candidates = [(c, t) for c, t in domain_cases(TRACKS) if "Primus/SET_INSP_PRES" not in t]
    random.Random(1).shuffle(candidates)
    cases = []
    rejected = 0
    for case, trk in candidates:
        w = steady_window(case, trk)
        tpl = None if w is None else template(w["AWP"], w["SET_RR_IPPV"])
        if tpl is None:
            continue
        w["template"] = tpl
        w["features"] = window_features(w, tpl)
        if not w["features"]["volume_control"]:
            rejected += 1
            continue
        w["capnogram"] = capnogram_shape(cyclic_breath(w["CO2"], w["SET_RR_IPPV"]), 60 / w["SET_RR_IPPV"])
        cases.append(w)
        print(f"case {case['caseid']} ({len(cases)}/{args.cases})", flush=True)
        if len(cases) == args.cases:
            break
    calibration, held_out = cases[0::2], cases[1::2]

    production_limb = ventilator.CIRCUIT_RESISTANCE
    print("circuit limb", flush=True)
    limb_grid = {}
    for resistance in LIMB_RESISTANCES:
        ventilator.CIRCUIT_RESISTANCE = resistance
        limb_grid[resistance] = median(compare_paw(calibration, True, production_tau)["whole"])
    ventilator.CIRCUIT_RESISTANCE = production_limb
    print("airway sensor", flush=True)
    sensor_grid = {tau: median(compare_paw(calibration, True, tau)["transitions"]) for tau in SENSOR_TAUS}
    print("breath shape", flush=True)
    single = compare_paw(held_out, False, production_tau)
    viscous = compare_paw(held_out, True, production_tau)

    print("capnograph", flush=True)
    capno_grids = fit_capnograph(calibration)
    fitted = {name: getattr(Capnograph, name) for name in production}
    simulated_capno = capnogram_features(held_out)  # With the fitted values
    for name, value in production.items():
        setattr(Capnograph, name, value)

    print("pressure control rise", flush=True)
    pcv_recorded, pcv_simulated = pcv_rise_time(args.pcv_cases)
    ventilator.RISE_TIME = production_rise

    print("compliance by patient", flush=True)
    compliance = compliance_by_patient(args.compliance_cases)

    print("end-tidal O2", flush=True)
    fio2_sim, gap_sim = zip(*(engine_eto2_gap(w) for w in cases), strict=True)
    gap_real = [w["FIO2"] - w["FEO2"] for w in cases]
    paired = np.array(gap_sim) - np.array(gap_real)

    default = Patient()
    feats = [w["features"] for w in cases]
    beta, ci = compliance["beta"], compliance["ci"]
    improved = sum(v < s for s, v in zip(single["whole"], viscous["whole"], strict=True))
    settings = ventilator.VentSettings()
    lines = [
        "# AnaSim ventilation vs Dräger Primus recordings (VitalDB)",
        "",
        f"{len(cases)} volume-control windows from {len(candidates)} eligible cases: adults in AnaSim's domain,",
        "general anesthesia, oral tube, open surgery, supine, one steady mid-surgery window each.",
        f"{rejected} further windows were skipped because their breaths did not peak at the end of flow and",
        "exhale at the set Ti (pressure-regulated breaths) or gave implausible mechanics. Fitted values use",
        f"the {len(calibration)} calibration windows and are checked on the {len(held_out)} held-out windows.",
        "Source: VitalDB (Lee et al., Sci Data 2022), CC BY-NC-SA 4.0. Median (IQR).",
        "",
        "## Settings",
        "",
        "| Setting | Recordings | AnaSim default |",
        "|---|---|---|",
        f"| Inspiratory pause (% of Ti) | {quartiles([w['SET_INSP_PAUSE'] for w in cases])} | {settings.pause:g} |",
        f"| Pmax (cmH2O) | {quartiles([w['SET_PIP'] * CM_PER_MBAR for w in cases])} | {settings.p_max:g} |",
        "",
        "## Mechanics from the recorded airway pressure",
        "",
        "Each case's resistance and compliance are set so that AnaSim's breath has the recorded end-of-flow",
        "drop and plateau; the other quantities are then predictions. Tissue viscoelasticity uses Jonson 1993",
        "(viscoelastic compliance 4x static, time constant 0.82 s) without fitting.",
        "",
        "| Quantity | Recordings | Single compartment | With tissue viscoelasticity |",
        "|---|---|---|---|",
        f"| Resistance from end-of-flow drop (cmH2O/(L/s)) | {quartiles([f['r'] for f in feats])} | matched | matched |",
        f"| Plateau compliance (mL/cmH2O) | {quartiles([f['c'] for f in feats])} | matched | matched |",
        f"| Expiratory tail decay / RC | {quartiles([f['tau_e_ratio'] for f in feats])} | "
        f"{quartiles(single['tau_e_ratio'])} | {quartiles(viscous['tau_e_ratio'])} |",
        f"| Expiratory limb resistance (cmH2O/(L/s)) | {quartiles([f['r_exp'] for f in feats])} | "
        f"{quartiles(single['r_exp'])} | {quartiles(viscous['r_exp'])} |",
        f"| Inspiratory knee time constant (ms) | {quartiles([1000 * f['tau_knee'] for f in feats])} | "
        f"{quartiles(1000 * np.array(single['knee']))} | {quartiles(1000 * np.array(viscous['knee']))} |",
        "",
        f"Tail values use windows whose expiratory tail reached 0.5 cmH2O ({sum(np.isfinite(f['tau_e']) for f in feats)}"
        " recorded); in the rest Paw returned to PEEP within 0.2 s.",
        "",
        "Circuit limb resistance fitted on the calibration windows (whole-breath RMSE): "
        + ", ".join(f"{k:g} {v:.3f}" for k, v in limb_grid.items())
        + f". Best {min(limb_grid, key=limb_grid.get):g}; AnaSim uses {production_limb:g} cmH2O/(L/s).",
        "",
        "## Breath shape (RMSE, cmH2O)",
        "",
        "Airway sensor time constant fitted on the calibration windows (transition RMSE): "
        + ", ".join(f"{1000 * k:g} ms {v:.2f}" for k, v in sensor_grid.items())
        + f". Best {1000 * min(sensor_grid, key=sensor_grid.get):g} ms; AnaSim uses {1000 * production_tau:g} ms.",
        "",
        "| Segment (held-out) | Single compartment | With tissue viscoelasticity |",
        "|---|---|---|",
        *(f"| {k} | {median(single[k]):.2f} | {median(viscous[k]):.2f} |" for k in (*SEGMENTS, "whole")),
        "",
        f"Tissue viscoelasticity lowered whole-breath RMSE in {improved} of {len(held_out)} held-out windows.",
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
        "Features come from the CO2 track alone. Each capnograph parameter was fitted on the calibration",
        "windows to the feature it alone sets, in turn (median absolute error): "
        + "; ".join(f"{name} " + ", ".join(f"{k:g}: {v:.3f}" for k, v in grid.items()) + f" (best {fitted[name]:g})"
                    for name, grid in capno_grids.items()) + ".",
        f"AnaSim uses ANALYZER_TAU {production['ANALYZER_TAU']:g} s, PHASE_II_VOLUME {production['PHASE_II_VOLUME']:g} L,",
        f"and PHASE_III_SLOPE {production['PHASE_III_SLOPE']:g} mmHg/L; the held-out check below uses the fitted",
        "values. The series dead space is the respiratory model's 2.2 mL/kg and was not fitted, so the share",
        "of each breath above 50% tests it.",
        "",
        "| Feature (held-out) | Recordings | AnaSim | Paired difference |",
        "|---|---|---|---|",
        *(f"| {label} | {quartiles([w['capnogram'][key] for w in held_out])} | "
          f"{quartiles([s[key] for s in simulated_capno])} | "
          f"{quartiles([s[key] - w['capnogram'][key] for s, w in zip(simulated_capno, held_out, strict=True)])} |"
          for key, label in (("fall", "Fall 90-10% (s)"), ("rise", "Rise 10-90% (s)"),
                             ("duty", "Share of breath above 50%"), ("slope", "Phase III slope (% of plateau/s)"))),
        "",
        "## Compliance by patient",
        "",
        f"Plateau compliance TV/(Pplat - PEEP) before incision in {compliance['n']} cases from numeric tracks:",
        f"PEEP 0-2 {quartiles(compliance['by_peep']['0-2'])}, PEEP 4-6 {quartiles(compliance['by_peep']['4-6'])}"
        " mL/cmH2O.",
        "",
        "log C = a + b log(PBW/60) + c (BMI - 23) + d PEEP over all cases (bootstrap 95% CI):",
        f"b = {beta[1]:.2f} ({ci[0][1]:.2f}-{ci[1][1]:.2f}), c = {beta[2]:.4f} ({ci[0][2]:.4f}-{ci[1][2]:.4f}) per kg/m2,",
        f"d = {beta[3]:.3f} ({ci[0][3]:.3f}-{ci[1][3]:.3f}) per cmH2O. C at PBW 60 kg, BMI 23, PEEP 5:",
        f"{math.exp(beta[0] + 5 * beta[3]):.1f} mL/cmH2O. Age adds {compliance['age_beta']:.4f} per year.",
        "",
        f"Held-out cases at PEEP 4-6 (n = {compliance['held_out']}), mean absolute error in mL/cmH2O:",
        f"constant {compliance['mae']['constant']:.2f}, body-size fit {compliance['mae']['scaled']:.2f},",
        f"AnaSim {compliance['mae']['anasim']:.2f} (median bias {compliance['anasim_bias']:+.2f}).",
        f"AnaSim's default patient: static {1000 * default.respiratory_compliance():.1f}, plateau "
        f"{anasim_plateau_compliance(default, 500):.1f} mL/cmH2O.",
        "",
        "## End-tidal O2",
        "",
        "| | Recordings | AnaSim |",
        "|---|---|---|",
        f"| FiO2 (%) | {quartiles([w['FIO2'] for w in cases])} | {quartiles(fio2_sim)} |",
        f"| FiO2 - EtO2 (%) | {quartiles(gap_real)} | {quartiles(gap_sim)} |",
        "",
        f"Paired AnaSim - recorded difference: {quartiles(paired)}; mean absolute {np.mean(np.abs(paired)):.2f}",
        "percentage points (the Primus reports whole percent).",
        "",
    ]
    out = ROOT / "results" / "vitaldb_ventilation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))

    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 4, figsize=(16, 9))
        for ax, w in zip(axes.flat, held_out, strict=False):
            t = (np.arange(len(w["template"])) - LEAD) * WAVE_DT
            ax.plot(t, w["template"], "k", lw=1.5, label="Primus")
            for viscoelastic, color, label in ((False, "C0", "Single compartment"), (True, "C3", "Viscoelastic")):
                resistance, compliance_l = matched_mechanics(w, viscoelastic, production_tau)
                sim = template(simulate(w, resistance, compliance_l, viscoelastic=viscoelastic,
                                        sensor_tau=production_tau)[0], w["SET_RR_IPPV"])
                ax.plot((np.arange(len(sim)) - LEAD) * WAVE_DT, sim, color, lw=1.0, label=label)
            ax.set_xlim(-0.1, w["SET_INSP_TM"] + 1.2)
            f = w["features"]
            ax.set_title(f"R {f['r']:.1f}  C {f['c']:.0f}  Ti {w['SET_INSP_TM']} s", fontsize=9)
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=80)


if __name__ == "__main__":
    main()
