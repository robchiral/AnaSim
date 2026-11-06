import math
from dataclasses import dataclass

import numpy as np

from anasim.core.enums import RhythmType
from anasim.core.utils import clamp01

from .cardiac_cycle import AGONAL_FAILURE, CardiacCycleSample

# Gaussian waves as (offset from R peak s, amplitude mV, width s). Durations stay
# fixed with rate; only the T wave follows the preceding R-R interval.
_P_WAVE = ((-0.165, 0.08, 0.030),)
_NARROW_QRS = ((-0.036, -0.12, 0.010), (0.0, 1.0, 0.013), (0.036, -0.20, 0.010))
_J_POINT_S = 0.05  # End of the narrow QRS after the R peak
_QTC_S = 0.40
_WAVE_CUTOFF_SIGMAS = 5.0

# Ischemic conduction failure is the coronary model's energy failure (0-1). P
# waves fade, then complexes widen and shrink and the T wave turns discordant,
# through junctional, idioventricular, and agonal rhythms. The ranges are
# teaching calibrations.
_P_FADE = (0.2, 0.5)
_QRS_WIDENING = (0.5, 0.95)
_MAX_STRETCH = 2.5  # QRS near 250 ms in agonal rhythm
_MIN_QRS_SCALE = 0.4
_DISCORDANT_T_MV = -0.4
JUNCTIONAL_FAILURE = 0.5
IDIOVENTRICULAR_FAILURE = 0.65  # QRS reaches 120 ms


def _ramp(value: float, bounds: tuple[float, float]) -> float:
    x = clamp01((value - bounds[0]) / (bounds[1] - bounds[0]))
    return x * x * (3.0 - 2.0 * x)


def _stretch(wide: float) -> float:
    return 1.0 + (_MAX_STRETCH - 1.0) * wide


def _t_wave(preceding_rr_s: float, wide: float) -> tuple[float, float]:
    """T-wave peak after the R peak and its width, s."""
    # Fridericia QT; the T peak sits about 60% of the way through it, after the QRS.
    qt = _QTC_S * preceding_rr_s ** (1.0 / 3.0)
    return 0.62 * qt + (_stretch(wide) - 1.0) * _J_POINT_S, 0.12 * qt * (1.0 + 0.5 * wide)


def displayed_rhythm(rhythm: RhythmType, conduction_failure: float) -> str:
    """Rhythm on the ECG, including escape rhythms during ischemic conduction failure."""
    if rhythm in (RhythmType.VTACH, RhythmType.VFIB, RhythmType.ASYSTOLE):
        return rhythm.value
    if conduction_failure >= AGONAL_FAILURE:
        return "Agonal rhythm"
    if conduction_failure >= IDIOVENTRICULAR_FAILURE:
        return "Idioventricular rhythm"
    if conduction_failure >= JUNCTIONAL_FAILURE and rhythm != RhythmType.AFIB:
        return "Junctional rhythm"
    return rhythm.value


@dataclass(frozen=True, slots=True)
class _Morphology:
    p: tuple[tuple[float, float, float], ...]
    qrs: tuple[tuple[float, float, float], ...]
    t_amplitude: float

    def waves(self, preceding_rr_s: float, p_scale: float = 1.0,
              wide: float = 0.0) -> tuple[tuple[float, float, float], ...]:
        stretch = _stretch(wide)
        size = 1.0 - (1.0 - _MIN_QRS_SCALE) * wide
        waves = tuple((offset, amplitude * p_scale, width) for offset, amplitude, width in self.p)
        waves += tuple((offset * stretch, amplitude * size, width * stretch) for offset, amplitude, width in self.qrs)
        if not self.t_amplitude:
            return waves
        peak, width = _t_wave(preceding_rr_s, wide)
        amplitude = self.t_amplitude * (1.0 - wide) + _DISCORDANT_T_MV * size * wide
        return waves + ((peak, amplitude, width),)


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
        self._st_window: tuple[float, float, float] | None = None
        self._st_scale = 1.0

    def _place_beats(self, cycle: CardiacCycleSample, conduction_failure: float) -> None:
        """Lay out the previous, current and next beat around the current R peak."""
        if cycle.beat_started and self._rhythm is not None:
            self._rr_s = (self._rr_s[1], self._rr_s[2], cycle.rr_interval_s)
        else:
            self._rr_s = (cycle.rr_interval_s,) * 3
        self._rhythm = cycle.rhythm_type

        morphology = _MORPHOLOGIES[cycle.rhythm_type]
        p_scale = 1.0 - _ramp(conduction_failure, _P_FADE)
        wide = _ramp(conduction_failure, _QRS_WIDENING)
        rr_into_prev, rr_into_current, rr_to_next = self._rr_s
        # A wide complex carries its own discordant ST-T in place of ischemic ST depression.
        self._st_scale = 1.0 - wide
        self._st_window = None
        if morphology.t_amplitude:
            peak, width = _t_wave(rr_into_current, wide)
            self._st_window = (_J_POINT_S * _stretch(wide), peak - 2.0 * width, peak + 2.5 * width)
        beats = ((-rr_into_current, rr_into_prev), (0.0, rr_into_current), (rr_to_next, rr_to_next))
        self._waves = [
            (r_time + offset, amplitude, width, _WAVE_CUTOFF_SIGMAS * width)
            for r_time, preceding_rr in beats
            for offset, amplitude, width in morphology.waves(preceding_rr, p_scale, wide)
        ]

    def _st_profile(self, t: float) -> float:
        """Horizontal from the J point, then back to baseline through the T wave."""
        if self._st_window is None:
            return 0.0
        j_point, t_onset, t_end = self._st_window
        return _ramp(t, (j_point - 0.01, j_point + 0.01)) * (1.0 - _ramp(t, (t_onset, t_end)))

    def step(self, dt: float, cycle: CardiacCycleSample, st_depression_mv: float = 0.0,
             *, conduction_failure: float = 0.0, render: bool = True) -> float:
        """Advance timing and noise; return voltage (mV), or zero when render is false."""
        rhythm_type = cycle.rhythm_type
        if rhythm_type == RhythmType.ASYSTOLE:
            self._rhythm = None
            voltage = self.rng.random() * 0.02 - 0.01
            return voltage if render else 0.0

        if rhythm_type == RhythmType.VFIB:
            self._rhythm = None
            # Coarse VF becomes fine as the myocardium exhausts its energy.
            amplitude = 0.3 * (1.0 - 0.85 * conduction_failure) * (1.0 + 0.35 * self._vf_envelope.step(dt, self.rng))
            val = amplitude * (self._vf_main.step(dt, self.rng) + 0.5 * self._vf_slow.step(dt, self.rng))
            voltage = val + self.rng.random() * 0.10 - 0.05
            return voltage if render else 0.0

        if cycle.beat_started or rhythm_type != self._rhythm:
            self._place_beats(cycle, conduction_failure)

        val = 0.0
        if render:
            t = cycle.elapsed_s
            for center, amplitude, width, cutoff in self._waves:
                offset = t - center
                if -cutoff < offset < cutoff:
                    val += amplitude * math.exp(-0.5 * (offset / width) ** 2)
            if st_depression_mv > 0.0:
                val -= st_depression_mv * self._st_scale * self._st_profile(t)

        if rhythm_type == RhythmType.AFIB:
            f_amplitude = 0.05 * (1.0 + 0.4 * self._f_envelope.step(dt, self.rng))
            val += f_amplitude * self._f_wave.step(dt, self.rng)

        voltage = val + self.rng.random() * 0.03 - 0.015
        return voltage if render else 0.0
