import math
from dataclasses import dataclass

import numpy as np

from anasim.core.enums import RhythmType
from anasim.core.utils import clamp01
from anasim.physiology.toxicity import SILENT_BLOCK

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
_COMPRESSION_ARTIFACT_MV = 0.6


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


def displayed_rhythm(rhythm: RhythmType, conduction_failure: float, sodium_block: float = 0.0) -> str:
    """Observed rhythm with ischemic slowing or sodium-channel block."""
    if rhythm in (RhythmType.VTACH, RhythmType.VFIB, RhythmType.ASYSTOLE):
        return rhythm.value
    if sodium_block >= SILENT_BLOCK:
        return RhythmType.ASYSTOLE.value
    if conduction_failure >= AGONAL_FAILURE:
        return "Agonal rhythm"
    if conduction_failure >= IDIOVENTRICULAR_FAILURE:
        return "Idioventricular rhythm"
    if sodium_block >= 0.2:
        return "Wide-complex rhythm"
    if conduction_failure >= JUNCTIONAL_FAILURE and rhythm != RhythmType.AFIB:
        return "Junctional rhythm"
    return rhythm.value


@dataclass(frozen=True, slots=True)
class _Morphology:
    p: tuple[tuple[float, float, float], ...]
    qrs: tuple[tuple[float, float, float], ...]
    t_amplitude: float

    def waves(
        self,
        preceding_rr_s: float,
        p_scale: float = 1.0,
        wide: float = 0.0,
        peaked_t: float = 0.0,
        *,
        hyperk: float = 0.0,
        pr_shift_s: float = 0.0,
    ) -> tuple[tuple[float, float, float], ...]:
        qrs_wide = max(wide, hyperk)
        stretch = _stretch(qrs_wide)
        # Ischemic/sodium-block wide complexes shrink and invert the T wave;
        # hyperkalemia widens the QRS into a tall positive sine-wave T complex.
        size = 1.0 - (1.0 - _MIN_QRS_SCALE) * wide - 0.25 * hyperk * (1.0 - wide)
        waves = tuple((offset - pr_shift_s, amplitude * p_scale, width) for offset, amplitude, width in self.p)
        waves += tuple((offset * stretch, amplitude * size, width * stretch) for offset, amplitude, width in self.qrs)
        if not self.t_amplitude:
            return waves
        peak, width = _t_wave(preceding_rr_s, qrs_wide)
        width *= 1.0 - 0.35 * peaked_t * (1.0 - qrs_wide) + 0.55 * hyperk * (1.0 - wide)
        base_t = self.t_amplitude + 0.45 * peaked_t
        amplitude = base_t * (1.0 - wide) + _DISCORDANT_T_MV * size * wide
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


def st_analyzable_scale(
    rhythm: RhythmType,
    conduction_failure: float,
    sodium_block: float = 0.0,
    hyperkalemia_block: float = 0.0,
) -> float:
    """Fraction (0-1) of ischemic ST deviation analyzable on the surface ECG."""
    morphology = _MORPHOLOGIES.get(rhythm)
    if morphology is None or not morphology.t_amplitude:
        return 0.0
    wide = max(_ramp(conduction_failure, _QRS_WIDENING), clamp01(sodium_block), clamp01(hyperkalemia_block))
    return 1.0 - wide


class _DriftingOscillator:
    """Sine whose frequency wanders around a mean, so the trace never repeats."""

    __slots__ = ("freq_hz", "mean_hz", "phase", "sd_hz", "tau_s")

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
        self._waves: list[tuple[float, float, float, float, float]] = []
        self._st_window: tuple[float, float, float] | None = None
        self._st_scale = 1.0
        self._vulnerable_window: tuple[float, float] | None = None
        self._shock_artifact_s = 0.0

    def trigger_shock_artifact(self) -> None:
        """Start a brief defibrillation discharge artifact and amplifier recovery transient."""
        self._shock_artifact_s = 0.25

    def _place_beats(
        self, cycle: CardiacCycleSample, conduction_failure: float, sodium_block: float,
        potassium: float = 4.0, hyperkalemia_block: float = 0.0,
    ) -> None:
        """Lay out the previous, current and next beat around the current R peak."""
        if cycle.beat_started and self._rhythm is not None:
            self._rr_s = (self._rr_s[1], self._rr_s[2], cycle.rr_interval_s)
        else:
            self._rr_s = (cycle.rr_interval_s,) * 3
        self._rhythm = cycle.rhythm_type

        morphology = _MORPHOLOGIES[cycle.rhythm_type]
        hyperk = 0.0 if cycle.rhythm_type == RhythmType.VTACH else clamp01(hyperkalemia_block)
        p_scale = (1.0 - _ramp(conduction_failure, _P_FADE)) * (1.0 - clamp01(hyperk / 0.4))
        pr_shift_s = 0.10 * clamp01(hyperk / 0.4)
        ischemic_wide = max(_ramp(conduction_failure, _QRS_WIDENING), clamp01(sodium_block))
        qrs_wide = max(ischemic_wide, hyperk)
        peaked_t = clamp01((potassium - 5.3) / 2.2)
        rr_into_prev, rr_into_current, rr_to_next = self._rr_s
        # A wide complex carries its own repolarization changes in place of ischemic ST depression.
        self._st_scale = 1.0 - qrs_wide
        self._st_window = None
        self._vulnerable_window = None
        if morphology.t_amplitude:
            peak, width = _t_wave(rr_into_current, qrs_wide)
            self._st_window = (_J_POINT_S * _stretch(qrs_wide), peak - 2.0 * width, peak + 2.5 * width)
            # The vulnerable period's width is a teaching calibration.
            self._vulnerable_window = (peak - width, peak + width)
        elif cycle.rhythm_type == RhythmType.VTACH:
            self._vulnerable_window = (0.075, 0.185)
        beats = ((-rr_into_current, rr_into_prev), (0.0, rr_into_current), (rr_to_next, rr_to_next))
        waves: list[tuple[float, float, float, float, float]] = []
        for r_time, preceding_rr in beats:
            for offset, amplitude, width in morphology.waves(
                preceding_rr, p_scale, ischemic_wide, peaked_t, hyperk=hyperk, pr_shift_s=pr_shift_s,
            ):
                center = r_time + offset
                cutoff = _WAVE_CUTOFF_SIGMAS * width
                waves.append((center - cutoff, center + cutoff, center, amplitude, -0.5 / (width * width)))
        self._waves = waves

    def vulnerable(self, cycle: CardiacCycleSample) -> bool:
        """Whether an unsynchronized discharge falls around this beat's T peak."""
        return (cycle.organized and cycle.rhythm_type == self._rhythm
                and self._vulnerable_window is not None
                and self._vulnerable_window[0] <= cycle.elapsed_s <= self._vulnerable_window[1])

    def _st_profile(self, t: float) -> float:
        """Horizontal from the J point, then back to baseline through the T wave."""
        if self._st_window is None:
            return 0.0
        j_point, t_onset, t_end = self._st_window
        return _ramp(t, (j_point - 0.01, j_point + 0.01)) * (1.0 - _ramp(t, (t_onset, t_end)))

    def step(self, dt: float, cycle: CardiacCycleSample, st_depression_mv: float = 0.0,
             *, conduction_failure: float = 0.0, sodium_block: float = 0.0,
             potassium: float = 4.0, hyperkalemia_block: float = 0.0,
             compression_phase: float | None = None) -> float:
        """Advance timing and noise; return voltage (mV).

        compression_phase, 0-1 through each chest compression, adds compression artifact.
        """
        artifact = 0.0
        shock_spike: float | None = None
        if self._shock_artifact_s > 0.0:
            elapsed_shock = 0.25 - self._shock_artifact_s
            self._shock_artifact_s = max(0.0, self._shock_artifact_s - dt)
            if elapsed_shock < 0.03:
                shock_spike = 2.5 if elapsed_shock < 0.015 else -1.8
            else:
                artifact += -0.8 * math.exp(-(elapsed_shock - 0.03) / 0.06)
        if compression_phase is not None:
            angle = 2.0 * math.pi * compression_phase
            artifact += _COMPRESSION_ARTIFACT_MV * (math.sin(angle) + 0.3 * math.sin(2.0 * angle + 0.5))
        rhythm_type = cycle.rhythm_type
        if rhythm_type == RhythmType.ASYSTOLE or (rhythm_type != RhythmType.VFIB and not cycle.organized):
            self._rhythm = None
            return shock_spike if shock_spike is not None else self.rng.random() * 0.02 - 0.01 + artifact

        if rhythm_type == RhythmType.VFIB:
            self._rhythm = None
            # Coarse VF becomes fine as the myocardium exhausts its energy.
            amplitude = 0.3 * (1.0 - 0.85 * conduction_failure) * (1.0 + 0.35 * self._vf_envelope.step(dt, self.rng))
            val = amplitude * (self._vf_main.step(dt, self.rng) + 0.5 * self._vf_slow.step(dt, self.rng))
            return shock_spike if shock_spike is not None else val + self.rng.random() * 0.10 - 0.05 + artifact

        if cycle.beat_started or rhythm_type != self._rhythm:
            self._place_beats(cycle, conduction_failure, sodium_block, potassium, hyperkalemia_block)

        if shock_spike is not None:
            return shock_spike

        val = 0.0
        t = cycle.elapsed_s
        for low, high, center, amplitude, neg_half_inv_var in self._waves:
            if low < t < high:
                offset = t - center
                val += amplitude * math.exp(neg_half_inv_var * offset * offset)
        if st_depression_mv > 0.0:
            val -= st_depression_mv * self._st_scale * self._st_profile(t)

        if rhythm_type == RhythmType.AFIB:
            f_amplitude = 0.05 * (1.0 + 0.4 * self._f_envelope.step(dt, self.rng))
            val += f_amplitude * self._f_wave.step(dt, self.rng)

        return val + self.rng.random() * 0.03 - 0.015 + artifact
