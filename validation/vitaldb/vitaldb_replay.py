"""Shared pieces of the VitalDB hemodynamic replays: delivery checks, cuff scoring, and summaries."""

from collections import Counter

import numpy as np
from scipy.optimize import minimize
from vitaldb_anesthetics import REMI_RATES, Excluded, covered, held, numeric_track

VASOACTIVE = ("NEPI", "PHEN", "EPI", "DOPA", "DOBU", "VASO", "NTG", "NPS", "MRN", "DTZ", "AMD", "PGE1")
REMI_UG_PER_ML = {"Orchestra/RFTN20_RATE": 20.0, "Orchestra/RFTN50_RATE": 50.0}
SAMPLE_SECONDS = 5.0
SOURCE = "https://doi.org/10.1038/s41597-022-01411-5"
CUFF_WINDOW_S = 40.0  # An oscillometric reading posts at the end of cuff deflation
STOP = (65.0, 90.0, 45.0)  # MAP, SBP, and HR below which treatment could follow
SENSITIVITY = (70.0, 100.0, 45.0)
PERIODS = ((0.0, 600.0), (600.0, np.inf))  # Seconds after propofol onset
BOOTSTRAP = 500


def amount(case, key):
    return float(case[key]) if case[key] else 0.0


def treated(case):
    return amount(case, "intraop_eph") > 0 or amount(case, "intraop_phe") > 0


def delivery(tracks):
    """Validated propofol and remifentanil pump histories.

    Rates are held across gaps up to 30 s, and a volume counter that restarts
    after a syringe change continues from its previous total.
    """
    rate, volume = numeric_track(tracks["Orchestra/PPF20_RATE"]), numeric_track(tracks["Orchestra/PPF20_VOL"])
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
    onset = float(rate[0][active[0]])
    stop = float(rate[0][active[-1] + 1]) if active[-1] + 1 < len(rate[0]) else float(rate[0][-1])
    remi_name = next(name for name in REMI_RATES if name in tracks)
    remi = numeric_track(tracks[remi_name])
    if np.any((remi[1] < 0) | (remi[1] > 1500)) or np.max(remi[1]) <= 0:
        raise Excluded("invalid or absent remifentanil delivery")
    gases = {name: numeric_track(tracks[name]) for name in ("Primus/MAC", "Primus/FIN2O")}
    if np.max(gases["Primus/MAC"][1]) > 0.05 or np.max(gases["Primus/FIN2O"][1]) > 1:
        raise Excluded("recorded volatile or nitrous oxide exposure")
    start = max(rate[0][0], volume[0][0], remi[0][0])
    if onset - start < 30:
        raise Excluded("delivery history starts too late for induction")
    end = min(rate[0][-1], volume[0][-1], remi[0][-1])
    for series in (rate, volume, remi):
        covered(series, start, end, 30.0)
    return rate, volume, remi, remi_name, start, end, onset, stop


def pre_induction(tracks, onset):
    """Last cuff MAP within 15 min before propofol, and median HR in the preceding 2 min."""
    nibp, hr = numeric_track(tracks["Solar8000/NIBP_MBP"]), numeric_track(tracks["Solar8000/HR"])
    sel = (nibp[0] >= onset - 900) & (nibp[0] < onset) & (nibp[1] >= 50) & (nibp[1] <= 150)
    if not sel.any():
        raise Excluded("no cuff MAP within 15 minutes before induction")
    rates = hr[1][(hr[0] >= onset - 120) & (hr[0] < onset) & (hr[1] >= 40) & (hr[1] <= 140)]
    if len(rates) < 5:
        raise Excluded("insufficient HR before induction")
    return float(nibp[1][sel][-1]), float(np.median(rates))


def readings(tracks, start, end):
    """Cuff readings in [start, end); the held 2-s MAP series changes value at each new reading."""
    time, value = numeric_track(tracks["Solar8000/NIBP_MBP"])
    sel = np.r_[False, np.diff(value) != 0] & (time >= start) & (time < end) & (value > 30) & (value < 160)
    times = time[sel]
    sbp = held(numeric_track(tracks["Solar8000/NIBP_SBP"]), times + 2.0, 10.0)
    hr = held(numeric_track(tracks["Solar8000/HR"]), times, 30.0)
    return times, value[sel], sbp, hr


def cuff_model(times, grid, values):
    """Model values averaged over the measurement window before each cuff reading posts."""
    return np.interp(times[:, None] + np.linspace(-CUFF_WINDOW_S, 0.0, 9), grid, values).mean(axis=1)


def rule_stats(errors, seconds, cuff_map, cuff_sbp, cuff_hr, limits):
    """Error counts, sums, and squares by period, ending at the first reading that could prompt treatment."""
    # NaN SBP or HR never triggers.
    trigger = np.flatnonzero((cuff_map < limits[0]) | (cuff_sbp < limits[1]) | (cuff_hr < limits[2]))
    end = trigger[0] + 1 if len(trigger) else len(errors)
    periods = [errors[:end][(seconds[:end] >= low) & (seconds[:end] < high)] for low, high in PERIODS]
    return dict(stopped=bool(len(trigger)),
                stats=[[len(part), float(part.sum()), float(np.square(part).sum())] for part in periods])


COURSE = (("1-2 min", 60.0, 120.0), ("2-5 min", 120.0, 300.0))


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


def course_table(rows):
    lines = ["", "Change in MAP from pre-induction, all readings without stopping, median (IQR):", "",
             "| Measure | Patients | Recorded | Model |", "|---|---:|---|---|"]
    for name in ("1-2 min", "2-5 min", "lowest in 10 min", "10 min before incision"):
        pairs = np.array([r["course"][name] for r in rows if name in r["course"]])
        if len(pairs):
            cells = [f"{np.median(x):.0f}% ({np.percentile(x, 25):.0f} to {np.percentile(x, 75):.0f})" for x in pairs.T]
            lines.append(f"| {name} | {len(pairs)} | {cells[0]} | {cells[1]} |")
    return lines


def window_medians(tracks, grid, predicted, t0, t1):
    """Recorded and model medians of MAP (arterial when recorded) and HR over 5-s samples in [t0, t1)."""
    pressure = "Solar8000/ART_MBP" if "Solar8000/ART_MBP" in tracks else "Solar8000/NIBP_MBP"
    sample = np.arange(t0, t1, SAMPLE_SECONDS)
    out = {}
    for column, (key, track) in enumerate((("map", pressure), ("hr", "Solar8000/HR"))):
        values = held(numeric_track(tracks[track]), sample, 30.0) if len(sample) else np.array([])
        valid = np.isfinite(values) & (values > 30) & (values < 160)
        if valid.sum() * SAMPLE_SECONDS >= 300:
            out[key] = dict(observed=float(np.median(values[valid])),
                            predicted=float(np.median(np.interp(sample[valid], grid, predicted[:, column]))))
    return out if len(out) == 2 else None


def fit(stats):
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


def mean_ci(values, rng):
    values = np.asarray(values, dtype=float)
    boot = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(BOOTSTRAP)]
    low, high = np.percentile(boot, [2.5, 97.5])
    return f"{values.mean():.1f} ({low:.1f} to {high:.1f}), n={len(values)}"


def report_header(title, settings):
    return [f"# {title}", "", f"Generated {settings['generated']}.", ""]


def report_footer(exclusions):
    lines = ["", "Exclusions:", ""]
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(Counter(r["exclusion"] for r in exclusions).items()))
    lines += ["", f"Source: [Lee et al., Scientific Data, 2022]({SOURCE}). VitalDB data are used under CC BY-NC-SA 4.0",
              "and its data use agreement.", ""]
    return lines
