import math
from dataclasses import dataclass

import numpy as np

_PROFILE_COL_TIME = 0
_PROFILE_COL_BIS = 1
_PROFILE_COL_HR = 2
_PROFILE_COL_SVR = 3
_PROFILE_COL_SV = 4


@dataclass(frozen=True)
class DisturbanceEffects:
    bis: float = 0.0
    svr: float = 0.0
    sv: float = 0.0
    hr: float = 0.0


@dataclass(frozen=True)
class DisturbanceProfileSpec:
    """One time-varying stimulus and its lifecycle.

    `sympathetic_intensity` is the Bouillon 2004 pre-opioid intensity that sets
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
        return float(self.points[-1, _PROFILE_COL_TIME]) if self.finite else None

    @property
    def established_s(self) -> float:
        """Elapsed time after which a sustained profile has reached its plateau."""
        return max(float(self.points[-1, _PROFILE_COL_TIME]), 5.0 * self.tonic_onset_s)

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
        label="Intubation / Short Stimulus",
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
    # Surgery also adds vasoconstriction that anesthetics do not block. In VitalDB
    # TIVA cases, MAP rose 12 mmHg from before incision to surgery while
    # remifentanil was increased (validation/vitaldb/vitaldb_tiva.py).
    # It develops over minutes, so MAP rises about 15% within 5 minutes of
    # incision at the sevoflurane-fentanyl MAC-BAR concentrations (Katoh 1999).
    "stim_sustained_surgery": DisturbanceProfileSpec(
        label="Sustained Surgical Stimulation",
        sympathetic_intensity=1.25,
        tonic_svr_fraction=0.15,
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

        if dist_profile is None:
            return
        try:
            self.spec = PROFILE_SPECS[dist_profile]
        except KeyError as exc:
            choices = ", ".join(PROFILE_SPECS)
            raise ValueError(f"dist_profile must be one of {choices} or None") from exc

    def stimulates_airway(self, elapsed_s: float) -> bool:
        """Return whether airway instrumentation is still under way."""
        return self.spec is not None and elapsed_s < self.spec.airway_stimulus_s

    def is_complete(self, elapsed_s: float) -> bool:
        """Return whether a finite profile has reached its configured end."""
        duration_s = self.spec.duration_s if self.spec else None
        return duration_s is not None and elapsed_s >= duration_s

    def compute_dist(self, time: float) -> DisturbanceEffects:
        """Return the BIS, HR, SVR, and SV deltas `time` seconds into the profile."""
        if self.spec is None:
            return DisturbanceEffects()

        points = self.spec.points
        profile_time = points[:, _PROFILE_COL_TIME]
        return DisturbanceEffects(
            bis=float(np.interp(time, profile_time, points[:, _PROFILE_COL_BIS])),
            svr=float(np.interp(time, profile_time, points[:, _PROFILE_COL_SVR])),
            sv=float(np.interp(time, profile_time, points[:, _PROFILE_COL_SV])),
            hr=float(np.interp(time, profile_time, points[:, _PROFILE_COL_HR])),
        )

    def compute_average(self, start_s: float, end_s: float) -> DisturbanceEffects:
        """Return the time-weighted average effect over one simulation step."""
        if self.spec is None or end_s <= start_s:
            return self.compute_dist(start_s)

        points = self.spec.points
        profile_time = points[:, _PROFILE_COL_TIME]
        internal_points = profile_time[(profile_time > start_s) & (profile_time < end_s)]
        sample_times = np.concatenate(([start_s], internal_points, [end_s]))
        widths = np.diff(sample_times)

        def average(column: int) -> float:
            values = np.interp(sample_times, profile_time, points[:, column])
            area = np.sum((values[:-1] + values[1:]) * widths * 0.5)
            return float(area / (end_s - start_s))

        return DisturbanceEffects(
            bis=average(_PROFILE_COL_BIS),
            svr=average(_PROFILE_COL_SVR),
            sv=average(_PROFILE_COL_SV),
            hr=average(_PROFILE_COL_HR),
        )
