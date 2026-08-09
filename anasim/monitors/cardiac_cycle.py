from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class CardiacCycleSample:
    """Timing information shared by beat-synchronous monitor renderers."""

    phase: float
    beat_started: bool
    rr_interval_s: float
    measured_hr: float
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
        self._measured_hr = 70.0

    @staticmethod
    def _mean_rr(hr: float) -> float:
        return 60.0 / float(hr)

    def _next_rr(self, hr: float, rhythm_type: RhythmType) -> float:
        mean_rr = self._mean_rr(hr)
        if rhythm_type == RhythmType.AFIB:
            factor = float(self.rng.uniform(0.65, 1.35))
            return mean_rr * factor
        return mean_rr

    def _sample(self, *, beat_started: bool) -> CardiacCycleSample:
        phase = 0.0
        if self._organized:
            phase = self._elapsed_s / self._rr_interval_s
        return CardiacCycleSample(
            phase=float(phase % 1.0),
            beat_started=beat_started,
            rr_interval_s=float(self._rr_interval_s),
            measured_hr=float(self._measured_hr if self._organized else 0.0),
            organized=self._organized,
            rhythm_type=self._rhythm_type,
        )

    def seed(self, hr: float, rhythm_type: RhythmType) -> CardiacCycleSample:
        """Initialize timing at a ventricular depolarization."""
        self._initialized = True
        self._rhythm_type = rhythm_type
        self._organized = rhythm_type in ORGANIZED_RHYTHMS and hr > 0.0
        self._elapsed_s = 0.0
        if self._organized:
            self._rr_interval_s = self._next_rr(hr, rhythm_type)
            self._measured_hr = 60.0 / self._rr_interval_s
        else:
            self._measured_hr = 0.0
        return self._sample(beat_started=self._organized)

    def step(self, dt: float, hr: float, rhythm_type: RhythmType) -> CardiacCycleSample:
        """Advance the beat clock and report any beat boundaries crossed."""
        if dt <= 0.0:
            raise ValueError("cardiac cycle dt must be greater than zero")

        organized = rhythm_type in ORGANIZED_RHYTHMS and hr > 0.0
        seeded_beat = False
        if not self._initialized or rhythm_type != self._rhythm_type or organized != self._organized:
            sample = self.seed(hr, rhythm_type)
            if not sample.organized:
                return sample
            seeded_beat = sample.beat_started

        if not organized:
            self._rhythm_type = rhythm_type
            self._organized = False
            self._elapsed_s = 0.0
            self._measured_hr = 0.0
            return self._sample(beat_started=False)

        self._rhythm_type = rhythm_type
        self._organized = True
        self._elapsed_s += dt
        beat_started = False
        while self._elapsed_s >= self._rr_interval_s:
            self._elapsed_s -= self._rr_interval_s
            self._measured_hr = 60.0 / self._rr_interval_s
            self._rr_interval_s = self._next_rr(hr, rhythm_type)
            beat_started = True

        return self._sample(beat_started=seeded_beat or beat_started)
