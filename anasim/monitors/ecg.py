import numpy as np

from anasim.core.enums import RhythmType

from .cardiac_cycle import CardiacCycleSample

_ECG_TEMPLATE_RESOLUTION = 500


def _build_ecg_template(mode: str, resolution: int = _ECG_TEMPLATE_RESOLUTION) -> np.ndarray:
    """
    Generate ECG template using Gaussian functions.
    Waves: (center_phase, amplitude, width)
    """
    # Base: P(0.306), Q(0.458), R(0.5), S(0.542), T(0.778)
    if mode == "sinus":
        waves = [
            (0.306, 0.08, 0.035),   # P wave
            (0.458, -0.12, 0.012),  # Q wave
            (0.500, 1.0, 0.015),    # R wave
            (0.542, -0.20, 0.012),  # S wave
            (0.778, 0.15, 0.055),   # T wave
        ]
    elif mode == "afib":
        # No P wave, irregular baseline (f-waves) handled in step()
        waves = [
            (0.458, -0.12, 0.012),  # Q wave
            (0.500, 1.0, 0.015),    # R wave
            (0.542, -0.20, 0.012),  # S wave
            (0.778, 0.15, 0.055),   # T wave
        ]
    elif mode == "svt":
        # Narrow QRS, P wave buried (removed)
        waves = [
            (0.458, -0.10, 0.010),  # Q wave (narrower)
            (0.500, 0.9, 0.012),    # R wave (narrower)
            (0.542, -0.15, 0.010),  # S wave (narrower)
            (0.750, 0.12, 0.050),   # T wave
        ]
    elif mode == "vtach":
        # Wide QRS, T wave often opposite polarity to QRS
        # "Monomorphic VT" appearance
        waves = [
            (0.400, 0.0, 0.1),      # Wide base
            (0.500, 0.8, 0.06),     # Wide R wave
            (0.650, -0.3, 0.08),    # Deep/Wide S/T transition
        ]
    else:
        raise ValueError(f"Unsupported ECG template mode: {mode!r}")

    phase_arr = np.linspace(0.0, 1.0, resolution)
    template = np.zeros(resolution)
    for center, amplitude, width in waves:
        gaussian = amplitude * np.exp(-0.5 * ((phase_arr - center) / width) ** 2)
        template += gaussian
    return template


_ECG_TEMPLATES = {
    RhythmType.SINUS: _build_ecg_template("sinus"),
    RhythmType.SINUS_BRADY: _build_ecg_template("sinus"),
    RhythmType.AFIB: _build_ecg_template("afib"),
    RhythmType.SVT: _build_ecg_template("svt"),
    RhythmType.VTACH: _build_ecg_template("vtach"),
}


class ECGMonitor:
    """
    ECG Monitor using Gaussian-based PQRST synthesis.
    Now supports multiple rhythm types with distinct waveforms.
    """
    def __init__(self, rng: np.random.Generator = None):
        self.vfib_phase = 0.0
        self.rng = rng if rng is not None else np.random.default_rng()
        
        self._templates = _ECG_TEMPLATES
        self._template_max_index = _ECG_TEMPLATE_RESOLUTION - 1
        
    def step(self, dt: float, cycle: CardiacCycleSample) -> float:
        """
        Return the next voltage sample logic.
        """
        rhythm_type = cycle.rhythm_type
        # 1. Asystole
        if rhythm_type == RhythmType.ASYSTOLE:
            return float(self.rng.uniform(-0.01, 0.01))
            
        # 2. VFib (Chaotic)
        if rhythm_type == RhythmType.VFIB:
            self.vfib_phase += dt
            # Sum of non-harmonic sines
            val = 0.2 * np.sin(self.vfib_phase * 20) + \
                  0.15 * np.sin(self.vfib_phase * 35) + \
                  0.1 * np.sin(self.vfib_phase * 12)
            val += float(self.rng.uniform(-0.05, 0.05))
            return val

        # Structured rhythms share the central beat clock. The existing ECG
        # template places the R wave at 0.5, so shift the lookup by half a beat.
        template_phase = (cycle.phase + 0.5) % 1.0
        template = self._templates[rhythm_type]
        idx = int(template_phase * self._template_max_index)
        val = template[idx]
        
        # Add baseline noise (fibrillatory waves for AFib)
        if rhythm_type == RhythmType.AFIB:
            # Coarse f-waves
            val += 0.02 * np.sin(cycle.phase * 50)
            val += float(self.rng.uniform(-0.02, 0.02))
        else:
            val += float(self.rng.uniform(-0.015, 0.015))
            
        return val
