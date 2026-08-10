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
    """One time-varying stimulus and its lifecycle."""

    label: str
    points: np.ndarray
    finite: bool = False

    @property
    def duration_s(self) -> float | None:
        """Return the endpoint for a finite profile."""
        return float(self.points[-1, _PROFILE_COL_TIME]) if self.finite else None


PROFILE_SPECS = {
    "stim_intubation_pulse": DisturbanceProfileSpec(
        label="Intubation / Short Stimulus",
        finite=True,
        points=np.array(
            [
                # time, BIS delta, HR delta, SVR delta, SV delta
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [8.0, 6.0, 6.0, 1.5, 2.0],
                [20.0, 12.0, 12.0, 3.0, 3.0],
                [35.0, 4.0, 4.0, 1.0, 1.0],
                [50.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    ),
    "stim_sustained_surgery": DisturbanceProfileSpec(
        label="Sustained Surgical Stimulation",
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
    """
    Time-dependent disturbance signal to mimic exogenous stimulation.

    Profiles are defined in seconds from the moment stimulation starts so
    the effect is immediate (no long delays).
    """

    def __init__(self, dist_profile: str = None):
        self.dist_profile = dist_profile
        self.spec = None

        if dist_profile is None:
            return
        try:
            self.spec = PROFILE_SPECS[dist_profile]
        except KeyError as exc:
            choices = ", ".join(PROFILE_SPECS)
            raise ValueError(f"dist_profile must be one of {choices} or None") from exc

    def is_complete(self, elapsed_s: float) -> bool:
        """Return whether a finite profile has reached its configured end."""
        duration_s = self.spec.duration_s if self.spec else None
        return duration_s is not None and elapsed_s >= duration_s

    def compute_dist(self, time: float) -> DisturbanceEffects:
        """
        Interpolate the disturbance profile for the given time (seconds).

        Returns DisturbanceEffects with BIS/HR/SVR/SV deltas.
        """
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
