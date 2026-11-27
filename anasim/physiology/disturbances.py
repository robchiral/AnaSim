import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DisturbanceEffects:
    bis: float = 0.0
    svr: float = 0.0
    sv: float = 0.0
    hr: float = 0.0


@dataclass(frozen=True)
class DisturbanceProfileSpec:
    """One time-varying stimulus and its lifecycle.

    `sympathetic_intensity` is the Bouillon 2004a pre-opioid intensity that sets
    how much drug blocks the HR and MAP response. At propofol 3.4 mcg/mL (BIS
    40-50), remifentanil blocked a 15% rise in half of patients at 5.0 ng/mL
    for intubation and 2.1 ng/mL for skin incision (Albertin 2005), giving 1.64
    and 1.25. Laryngoscopy movement is 0.83.
    """

    label: str
    points: np.ndarray
    sympathetic_intensity: float
    tonic_svr_fraction: float = 0.0  # Plateau rise in resistance that anesthetics do not block
    tonic_onset_s: float = 0.0  # Time constant of that rise
    finite: bool = False
    airway_stimulus_s: float = 0.0  # Instrumentation that can provoke laryngospasm

    @property
    def duration_s(self) -> float | None:
        """Return the endpoint for a finite profile."""
        return float(self.points[-1, 0]) if self.finite else None

    @property
    def established_s(self) -> float:
        """Elapsed time after which a sustained profile has reached its plateau."""
        return max(float(self.points[-1, 0]), 5.0 * self.tonic_onset_s)

    def tonic_svr(self, elapsed_s: float, tpr_wood: float) -> float:
        """Unblocked vasoconstriction (Wood units) `elapsed_s` into the profile."""
        progress = -math.expm1(-elapsed_s / self.tonic_onset_s) if self.tonic_onset_s > 0.0 else 1.0
        return self.tonic_svr_fraction * tpr_wood * progress


# Without opioid, intubation after propofol or at 1 MAC sevoflurane raised HR
# 36-38 bpm and MAP 35-40 mmHg (Ebert 1992; Ebert 1995). HR recovers within
# 3 min and pressure within 5; HR recovery and the SV/SVR split are calibrated
# to Billard 1994 and Sawano 2013.
PROFILE_SPECS = {
    "stim_intubation_pulse": DisturbanceProfileSpec(
        label="Intubation or brief stimulus",
        sympathetic_intensity=1.64,
        finite=True,
        airway_stimulus_s=40.0,
        points=np.array([
            # time, BIS, HR (bpm), SVR (Wood units), SV (mL)
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [15.0, 7.2, 36.0, 0.28, 8.8],
            [40.0, 12.0, 24.0, 0.46, 14.7],
            [90.0, 7.2, 10.0, 0.28, 8.8],
            [180.0, 3.0, 0.0, 0.115, 3.7],
            [300.0, 0.0, 0.0, 0.0, 0.0],
        ]),
    ),
    # Surgery also adds vasoconstriction that anesthetics do not block. In open
    # supine VitalDB TIVA cases, MAP rose 10.1 mmHg from before incision to
    # surgery while remifentanil was increased (validation/vitaldb/vitaldb_tiva.py);
    # laparoscopic/robotic cases rose further (+16.9 mmHg) due to pneumoperitoneum.
    # It develops over minutes, so MAP rises about 15% within 5 minutes of
    # incision at the sevoflurane-fentanyl MAC-BAR concentrations (Katoh 1999).
    "stim_sustained_surgery": DisturbanceProfileSpec(
        label="Sustained surgical stimulation",
        sympathetic_intensity=1.25,
        tonic_svr_fraction=0.16,
        tonic_onset_s=300.0,
        points=np.array(
            [
                # time, BIS delta, HR delta, SVR delta, SV delta
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 4.0, 4.0, 0.8, 0.5],
                [60.0, 8.0, 6.0, 1.6, 1.0],
                [600.0, 8.0, 6.0, 1.6, 1.0],
            ]
        ),
    ),
}


def list_disturbance_profiles() -> list[tuple[str, str]]:
    """Return (label, key) pairs for UI/selection."""
    return [(spec.label, key) for key, spec in PROFILE_SPECS.items()]


class Disturbances:
    """Surgical stimulation profile, timed in seconds from when it starts."""

    def __init__(self, dist_profile: str | None = None):
        self.spec = None
        self._plateau = DisturbanceEffects()
        self._times: tuple[float, ...] = ()
        self._rows: tuple[tuple[float, float, float, float], ...] = ()

        if dist_profile is None:
            return
        try:
            self.spec = PROFILE_SPECS[dist_profile]
        except KeyError as exc:
            choices = ", ".join(PROFILE_SPECS)
            raise ValueError(f"dist_profile must be one of {choices} or None") from exc
        pts = self.spec.points
        self._times = tuple(float(row[0]) for row in pts)
        self._rows = tuple((float(row[1]), float(row[2]), float(row[3]), float(row[4])) for row in pts)
        self._plateau = self._effects(self._rows[-1])

    @staticmethod
    def _effects(values) -> DisturbanceEffects:
        bis, hr, svr, sv = values
        return DisturbanceEffects(bis=float(bis), hr=float(hr), svr=float(svr), sv=float(sv))

    def _interp_row(self, time: float) -> tuple[float, float, float, float]:
        times = self._times
        rows = self._rows
        if time <= times[0]:
            return rows[0]
        if time >= times[-1]:
            return rows[-1]
        idx = bisect_right(times, time)
        t0 = times[idx - 1]
        frac = (time - t0) / (times[idx] - t0)
        r0 = rows[idx - 1]
        r1 = rows[idx]
        return (
            r0[0] + frac * (r1[0] - r0[0]),
            r0[1] + frac * (r1[1] - r0[1]),
            r0[2] + frac * (r1[2] - r0[2]),
            r0[3] + frac * (r1[3] - r0[3]),
        )

    def stimulates_airway(self, elapsed_s: float) -> bool:
        """Return whether airway instrumentation is still under way."""
        return self.spec is not None and elapsed_s < self.spec.airway_stimulus_s

    def is_complete(self, elapsed_s: float) -> bool:
        """Return whether a finite profile has reached its configured end."""
        duration_s = self.spec.duration_s if self.spec else None
        return duration_s is not None and elapsed_s >= duration_s

    def compute_dist(self, time: float) -> DisturbanceEffects:
        """Return the BIS, HR, SVR, and SV deltas `time` seconds into the profile."""
        if self.spec is None or time >= self._times[-1]:
            return self._plateau
        return self._effects(self._interp_row(time))

    def compute_average(self, start_s: float, end_s: float) -> DisturbanceEffects:
        """Return the time-weighted average effect over one simulation step."""
        if self.spec is None or start_s >= self._times[-1]:
            return self._plateau
        if end_s <= start_s:
            return self.compute_dist(start_s)

        times = self._times
        first = bisect_right(times, start_s)
        last = bisect_left(times, end_s)
        if first >= last:
            return self._effects(self._interp_row(0.5 * (start_s + end_s)))
        rows = self._rows
        inv_span = 0.5 / (end_s - start_s)
        prev_t = start_s
        prev_r = self._interp_row(start_s)
        a0 = a1 = a2 = a3 = 0.0
        for k in range(first, last):
            t = times[k]
            r = rows[k]
            w = t - prev_t
            a0 += (prev_r[0] + r[0]) * w
            a1 += (prev_r[1] + r[1]) * w
            a2 += (prev_r[2] + r[2]) * w
            a3 += (prev_r[3] + r[3]) * w
            prev_t = t
            prev_r = r
        end_r = self._interp_row(end_s)
        w = end_s - prev_t
        a0 += (prev_r[0] + end_r[0]) * w
        a1 += (prev_r[1] + end_r[1]) * w
        a2 += (prev_r[2] + end_r[2]) * w
        a3 += (prev_r[3] + end_r[3]) * w
        return DisturbanceEffects(bis=a0 * inv_span, hr=a1 * inv_span, svr=a2 * inv_span, sv=a3 * inv_span)
