import math
from dataclasses import dataclass

import numpy as np

from anasim.core.enums import RhythmType

from .cardiac_cycle import CardiacCycleSample

# Gaussian waves as (offset from R peak s, amplitude mV, width s). Durations stay
# fixed with rate; only the T wave follows the preceding R-R interval.
_P_WAVE = ((-0.165, 0.08, 0.030),)
_NARROW_QRS = ((-0.036, -0.12, 0.010), (0.0, 1.0, 0.013), (0.036, -0.20, 0.010))
_QTC_S = 0.40
_WAVE_CUTOFF_SIGMAS = 5.0


@dataclass(frozen=True, slots=True)
class _Morphology:
    p: tuple[tuple[float, float, float], ...]
    qrs: tuple[tuple[float, float, float], ...]
    t_amplitude: float

    def waves(self, preceding_rr_s: float) -> tuple[tuple[float, float, float], ...]:
        if not self.t_amplitude:
            return self.p + self.qrs
        # Fridericia QT; the T peak sits about 60% of the way through it.
        qt = _QTC_S * preceding_rr_s ** (1.0 / 3.0)
        return self.p + self.qrs + ((0.62 * qt, self.t_amplitude, 0.12 * qt),)


_MORPHOLOGIES = {
    RhythmType.SINUS: _Morphology(_P_WAVE, _NARROW_QRS, 0.15),
    RhythmType.SINUS_BRADY: _Morphology(_P_WAVE, _NARROW_QRS, 0.15),
    # No P wave; step() adds fibrillatory baseline.
    RhythmType.AFIB: _Morphology((), _NARROW_QRS, 0.15),
    # Narrow QRS with the P wave buried.
    RhythmType.SVT: _Morphology((), ((-0.036, -0.10, 0.010), (0.0, 0.9, 0.013), (0.036, -0.15, 0.010)), 0.12),
    # Wide (>120 ms) monomorphic QRS with a discordant ST-T.
    RhythmType.VTACH: _Morphology((), ((0.0, 0.85, 0.035), (0.13, -0.35, 0.055)), 0.0),
}


class _DriftingOscillator:
    """Sine whose frequency wanders around a mean, so the trace never repeats."""

    def __init__(self, mean_hz: float, sd_hz: float, tau_s: float = 0.5):
        self.mean_hz = mean_hz
        self.sd_hz = sd_hz
        self.tau_s = tau_s
        self.freq_hz = mean_hz
        self.phase = 0.0

    def step(self, dt: float, rng: np.random.Generator) -> float:
        a = min(1.0, dt / self.tau_s)
        self.freq_hz += a * (self.mean_hz - self.freq_hz) + self.sd_hz * math.sqrt(2.0 * a) * rng.standard_normal()
        self.phase += 2.0 * math.pi * max(0.1, self.freq_hz) * dt
        return math.sin(self.phase)


class ECGMonitor:
    """ECG from per-rhythm Gaussian waves placed in time around each R peak."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()
        # VF dominant frequency is about 5 Hz; f-waves run at 350-450/min.
        self._vf_main = _DriftingOscillator(5.0, 1.5, tau_s=0.3)
        self._vf_slow = _DriftingOscillator(2.8, 0.8, tau_s=0.3)
        self._vf_envelope = _DriftingOscillator(0.4, 0.15, tau_s=2.0)
        self._f_wave = _DriftingOscillator(6.5, 1.0, tau_s=0.3)
        self._f_envelope = _DriftingOscillator(0.3, 0.1, tau_s=2.0)
        self._rhythm: RhythmType | None = None
        self._rr_s = (0.0, 0.0, 0.0)
        self._waves: list[tuple[float, float, float, float]] = []

    def _place_beats(self, cycle: CardiacCycleSample) -> None:
        """Lay out the previous, current and next beat around the current R peak."""
        if cycle.beat_started and self._rhythm is not None:
            self._rr_s = (self._rr_s[1], self._rr_s[2], cycle.rr_interval_s)
        else:
            self._rr_s = (cycle.rr_interval_s,) * 3
        self._rhythm = cycle.rhythm_type

        morphology = _MORPHOLOGIES[cycle.rhythm_type]
        rr_into_prev, rr_into_current, rr_to_next = self._rr_s
        beats = ((-rr_into_current, rr_into_prev), (0.0, rr_into_current), (rr_to_next, rr_to_next))
        self._waves = [
            (r_time + offset, amplitude, width, _WAVE_CUTOFF_SIGMAS * width)
            for r_time, preceding_rr in beats
            for offset, amplitude, width in morphology.waves(preceding_rr)
        ]

    def step(self, dt: float, cycle: CardiacCycleSample) -> float:
        """Return the next ECG voltage (mV)."""
        rhythm_type = cycle.rhythm_type
        if rhythm_type == RhythmType.ASYSTOLE:
            self._rhythm = None
            return self.rng.random() * 0.02 - 0.01

        if rhythm_type == RhythmType.VFIB:
            self._rhythm = None
            amplitude = 0.3 * (1.0 + 0.35 * self._vf_envelope.step(dt, self.rng))
            val = amplitude * (self._vf_main.step(dt, self.rng) + 0.5 * self._vf_slow.step(dt, self.rng))
            return val + self.rng.random() * 0.10 - 0.05

        if cycle.beat_started or rhythm_type != self._rhythm:
            self._place_beats(cycle)

        t = cycle.elapsed_s
        val = 0.0
        for center, amplitude, width, cutoff in self._waves:
            offset = t - center
            if -cutoff < offset < cutoff:
                val += amplitude * math.exp(-0.5 * (offset / width) ** 2)

        if rhythm_type == RhythmType.AFIB:
            f_amplitude = 0.05 * (1.0 + 0.4 * self._f_envelope.step(dt, self.rng))
            val += f_amplitude * self._f_wave.step(dt, self.rng)

        return val + self.rng.random() * 0.03 - 0.015
