from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from anasim.core.enums import RhythmType

ORGANIZED_RHYTHMS = frozenset(
    {
        RhythmType.SINUS,
        RhythmType.SINUS_BRADY,
        RhythmType.AFIB,
        RhythmType.SVT,
        RhythmType.VTACH,
    }
)

# Bedside monitors average recent R-R intervals rather than showing each beat's rate.
DISPLAY_HR_BEATS = 12


# AF R-R intervals are uniform within +/-35% of the mean (rMSSD ~0.29 x mean R-R,
# as measured by Corino 2015), narrowed so none is shorter than the AV node passes
# (about 250 ms, 240/min).
AF_RR_SPREAD = 0.35
AF_MIN_RR_S = 0.25
_SYSTOLE_S = 0.25
_FILLING_TAU_S = 0.25


def _af_spread(mean_rr_s: float) -> float:
    return min(AF_RR_SPREAD, max(0.0, 1.0 - AF_MIN_RR_S / mean_rr_s))


def _filling(rr_s: float) -> float:
    """Relative ventricular filling after about 250 ms of systole."""
    return 1.0 - math.exp(-max(rr_s - _SYSTOLE_S, 0.0) / _FILLING_TAU_S)


def _mean_af_filling(mean_rr_s: float) -> float:
    """Average filling over AF R-R intervals uniform within the spread."""
    spread = _af_spread(mean_rr_s)
    if spread <= 0.0:
        return _filling(mean_rr_s)
    low = max(mean_rr_s * (1.0 - spread), _SYSTOLE_S)
    high = max(mean_rr_s * (1.0 + spread), _SYSTOLE_S)
    decay = math.exp(-(low - _SYSTOLE_S) / _FILLING_TAU_S) - math.exp(-(high - _SYSTOLE_S) / _FILLING_TAU_S)
    return (high - low - _FILLING_TAU_S * decay) / (2.0 * spread * mean_rr_s)


@dataclass(frozen=True, slots=True)
class CardiacCycleSample:
    """Timing information shared by beat-synchronous monitor renderers."""

    phase: float
    beat_started: bool
    rr_interval_s: float
    mean_rr_s: float
    stroke_fraction: float
    measured_hr: float
    display_hr: float
    organized: bool
    rhythm_type: RhythmType

    @property
    def elapsed_s(self) -> float:
        return self.phase * self.rr_interval_s

    def delayed_phase(self, delay_s: float) -> float:
        """Return beat phase after applying a fixed signal transit delay."""
        if not self.organized:
            return 0.0
        return ((self.elapsed_s - delay_s) / self.rr_interval_s) % 1.0


class CardiacCycle:
    """Own the organized beat clock used by ECG, ART, and pleth renderers."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self._initialized = False
        self._organized = False
        self._rhythm_type = RhythmType.SINUS
        self._elapsed_s = 0.0
        self._rr_interval_s = 60.0 / 70.0
        self._mean_rr_s = self._rr_interval_s
        self._stroke_fraction = 1.0
        self._measured_hr = 70.0
        self._recent_rr: deque[float] = deque(maxlen=DISPLAY_HR_BEATS)

    @staticmethod
    def _mean_rr(hr: float) -> float:
        return 60.0 / float(hr)

    def _next_rr(self, hr: float, rhythm_type: RhythmType) -> float:
        mean_rr = self._mean_rr(hr)
        if rhythm_type == RhythmType.AFIB:
            spread = _af_spread(mean_rr)
            factor = float(self.rng.uniform(1.0 - spread, 1.0 + spread))
            return mean_rr * factor
        return mean_rr

    def _beat_stroke_fraction(self, preceding_rr_s: float, rhythm_type: RhythmType) -> float:
        """Stroke volume relative to the rhythm's average beat.

        SV rises curvilinearly with the preceding R-R in AF (Hardman 1998), and more
        steeply at faster rates (Kerr 1998).
        """
        if preceding_rr_s == self._mean_rr_s:
            return 1.0
        if rhythm_type == RhythmType.AFIB:
            average = _mean_af_filling(self._mean_rr_s)
        else:
            average = _filling(self._mean_rr_s)
        return min(1.5, _filling(preceding_rr_s) / max(average, 0.05))

    def _sample(self, *, beat_started: bool) -> CardiacCycleSample:
        phase = 0.0
        display_hr = 0.0
        if self._organized:
            phase = self._elapsed_s / self._rr_interval_s
            display_hr = self._measured_hr
            if self._recent_rr:
                display_hr = 60.0 * len(self._recent_rr) / sum(self._recent_rr)
        return CardiacCycleSample(
            phase=float(phase % 1.0),
            beat_started=beat_started,
            rr_interval_s=float(self._rr_interval_s),
            mean_rr_s=float(self._mean_rr_s),
            stroke_fraction=float(self._stroke_fraction if self._organized else 0.0),
            measured_hr=float(self._measured_hr if self._organized else 0.0),
            display_hr=float(display_hr),
            organized=self._organized,
            rhythm_type=self._rhythm_type,
        )

    def _restart(self, hr: float, rhythm_type: RhythmType) -> CardiacCycleSample:
        self._initialized = True
        self._rhythm_type = rhythm_type
        self._organized = rhythm_type in ORGANIZED_RHYTHMS and hr > 0.0
        self._elapsed_s = 0.0
        self._stroke_fraction = 1.0
        if self._organized:
            self._mean_rr_s = self._mean_rr(hr)
            self._rr_interval_s = self._next_rr(hr, rhythm_type)
            self._measured_hr = 60.0 / self._rr_interval_s
        else:
            self._measured_hr = 0.0
            self._recent_rr.clear()
        return self._sample(beat_started=self._organized)

    def seed(self, hr: float, rhythm_type: RhythmType) -> CardiacCycleSample:
        """Initialize timing at a ventricular depolarization."""
        self._recent_rr.clear()
        return self._restart(hr, rhythm_type)

    def step(self, dt: float, hr: float, rhythm_type: RhythmType) -> CardiacCycleSample:
        """Advance the beat clock and report any beat boundaries crossed."""
        if dt <= 0.0:
            raise ValueError("cardiac cycle dt must be greater than zero")

        organized = rhythm_type in ORGANIZED_RHYTHMS and hr > 0.0
        seeded_beat = False
        if not self._initialized or rhythm_type != self._rhythm_type or organized != self._organized:
            # A rhythm change keeps the monitor's rate history.
            sample = self._restart(hr, rhythm_type)
            if not sample.organized:
                return sample
            seeded_beat = sample.beat_started

        if not organized:
            self._rhythm_type = rhythm_type
            self._organized = False
            self._elapsed_s = 0.0
            self._measured_hr = 0.0
            self._recent_rr.clear()
            return self._sample(beat_started=False)

        self._rhythm_type = rhythm_type
        self._organized = True
        self._mean_rr_s = self._mean_rr(hr)
        self._elapsed_s += dt
        beat_started = False
        while self._elapsed_s >= self._rr_interval_s:
            self._elapsed_s -= self._rr_interval_s
            self._measured_hr = 60.0 / self._rr_interval_s
            self._recent_rr.append(self._rr_interval_s)
            self._stroke_fraction = self._beat_stroke_fraction(self._rr_interval_s, rhythm_type)
            self._rr_interval_s = self._next_rr(hr, rhythm_type)
            beat_started = True

        return self._sample(beat_started=seeded_beat or beat_started)
