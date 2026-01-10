from dataclasses import dataclass
import numpy as np

_PROFILE_COL_TIME = 0
_PROFILE_COL_BIS = 1
_PROFILE_COL_HR = 2
_PROFILE_COL_SVR = 3
_PROFILE_COL_SV = 4

@dataclass(frozen=True)
class DisturbanceVector:
    bis: float = 0.0
    map: float = 0.0
    co: float = 0.0
    svr: float = 0.0
    sv: float = 0.0
    hr: float = 0.0

    def as_tuple(self) -> tuple:
        return (self.bis, self.map, self.co, self.svr, self.sv, self.hr)

    def __iter__(self):
        return iter(self.as_tuple())

    def __len__(self) -> int:
        return 6

    def __getitem__(self, idx: int) -> float:
        return self.as_tuple()[idx]

    def __eq__(self, other) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(other) == self.as_tuple()
        if isinstance(other, DisturbanceVector):
            return other.as_tuple() == self.as_tuple()
        return False


PROFILE_META = [
    ("stim_intubation_pulse", "Intubation / Short Stimulus"),
    ("stim_sustained_surgery", "Sustained Surgical Stimulation"),
]


def list_disturbance_profiles() -> list[tuple[str, str]]:
    """Return (label, key) pairs for UI/selection."""
    return [(label, key) for key, label in PROFILE_META]


PROFILE_TABLES = {
    # Time in seconds from disturbance start.
    "stim_intubation_pulse": np.array([
        # time, BIS delta, HR delta (bpm), SVR delta (Wood Units), SV delta (mL)
        [0.0,  0.0,  0.0, 0.0, 0.0],
        [8.0,  6.0,  6.0, 1.5, 2.0],
        [20.0, 12.0, 12.0, 3.0, 3.0],
        [35.0, 4.0,  4.0, 1.0, 1.0],
        [50.0, 0.0,  0.0, 0.0, 0.0],
    ]),
    "stim_sustained_surgery": np.array([
        # time, BIS delta, HR delta (bpm), SVR delta (Wood Units), SV delta (mL)
        [0.0,   0.0, 0.0, 0.0, 0.0],
        [10.0,  4.0, 4.0, 0.8, 0.5],
        [60.0,  8.0, 6.0, 1.6, 1.0],
        [600.0, 8.0, 6.0, 1.6, 1.0],
    ]),
}


class Disturbances:
    """
    Time-dependent disturbance signal to mimic exogenous stimulation.

    Profiles are defined in seconds from the moment stimulation starts so
    the effect is immediate (no long delays).
    """

    def __init__(
        self,
        dist_profil: str = None,
        dist_profile: str = None,
        **_kwargs,  # Ignore extra args for compatibility
    ):
        if dist_profile is not None and dist_profil is None:
            dist_profil = dist_profile
        self.dist_profil = dist_profil
        self.disturb_point = PROFILE_TABLES.get(dist_profil)
        self._profile_time = None
        self._profile_bis = None
        self._profile_hr = None
        self._profile_svr = None
        self._profile_sv = None

        if dist_profil is None:
            return
        if self.disturb_point is None:
            raise ValueError(
                "dist_profil should be: stim_intubation_pulse, stim_sustained_surgery or None"
            )
        self._profile_time = self.disturb_point[:, _PROFILE_COL_TIME]
        self._profile_bis = self.disturb_point[:, _PROFILE_COL_BIS]
        self._profile_hr = self.disturb_point[:, _PROFILE_COL_HR]
        self._profile_svr = self.disturb_point[:, _PROFILE_COL_SVR]
        self._profile_sv = self.disturb_point[:, _PROFILE_COL_SV]

    def compute_dist(self, time: float) -> DisturbanceVector:
        """
        Interpolate the disturbance profile for the given time (seconds).

        Returns DisturbanceVector with BIS/HR/SVR/SV deltas.
        """
        if self.disturb_point is None:
            return DisturbanceVector()

        dist_bis = np.interp(time, self._profile_time, self._profile_bis)
        dist_hr = np.interp(time, self._profile_time, self._profile_hr)
        dist_svr = np.interp(time, self._profile_time, self._profile_svr)
        dist_sv = np.interp(time, self._profile_time, self._profile_sv)
        return DisturbanceVector(dist_bis, 0.0, 0.0, dist_svr, dist_sv, dist_hr)
