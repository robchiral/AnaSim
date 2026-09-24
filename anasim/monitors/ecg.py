import numpy as np

from anasim.core.enums import RhythmType

from .cardiac_cycle import CardiacCycleSample

_ECG_TEMPLATE_RESOLUTION = 500


def _build_ecg_template(mode: str, resolution: int = _ECG_TEMPLATE_RESOLUTION) -> np.ndarray:
    """One beat as a sum of Gaussian waves (center phase, amplitude, width).

    The R wave sits at phase 0.5.
    """
    if mode == "sinus":
        # P, Q, R, S, T
        waves = [
            (0.306, 0.08, 0.035),
            (0.458, -0.12, 0.012),
            (0.500, 1.0, 0.015),
            (0.542, -0.20, 0.012),
            (0.778, 0.15, 0.055),
        ]
    elif mode == "afib":
        # No P wave; step() adds fibrillatory baseline.
        waves = [
            (0.458, -0.12, 0.012),
            (0.500, 1.0, 0.015),
            (0.542, -0.20, 0.012),
            (0.778, 0.15, 0.055),
        ]
    elif mode == "svt":
        # Narrow QRS with the P wave buried.
        waves = [
            (0.458, -0.10, 0.010),
            (0.500, 0.9, 0.012),
            (0.542, -0.15, 0.010),
            (0.750, 0.12, 0.050),
        ]
    elif mode == "vtach":
        # Wide monomorphic QRS with a discordant ST-T.
        waves = [
            (0.400, 0.0, 0.1),
            (0.500, 0.8, 0.06),
            (0.650, -0.3, 0.08),
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
    """ECG from per-rhythm Gaussian beat templates on the shared beat clock."""

    def __init__(self, rng: np.random.Generator = None):
        self.vfib_phase = 0.0
        self.rng = rng if rng is not None else np.random.default_rng()
        self._templates = _ECG_TEMPLATES
        self._template_max_index = _ECG_TEMPLATE_RESOLUTION - 1

    def step(self, dt: float, cycle: CardiacCycleSample) -> float:
        """Return the next ECG voltage (mV)."""
        rhythm_type = cycle.rhythm_type
        if rhythm_type == RhythmType.ASYSTOLE:
            return float(self.rng.uniform(-0.01, 0.01))

        if rhythm_type == RhythmType.VFIB:
            # Non-harmonic sines give a chaotic trace.
            self.vfib_phase += dt
            val = 0.2 * np.sin(self.vfib_phase * 20) + \
                  0.15 * np.sin(self.vfib_phase * 35) + \
                  0.1 * np.sin(self.vfib_phase * 12)
            val += float(self.rng.uniform(-0.05, 0.05))
            return val

        # Beat phase 0 is depolarization; the template R wave is at 0.5.
        template_phase = (cycle.phase + 0.5) % 1.0
        template = self._templates[rhythm_type]
        idx = int(template_phase * self._template_max_index)
        val = template[idx]

        if rhythm_type == RhythmType.AFIB:
            # Coarse fibrillatory waves.
            val += 0.02 * np.sin(cycle.phase * 50)
            val += float(self.rng.uniform(-0.02, 0.02))
        else:
            val += float(self.rng.uniform(-0.015, 0.015))

        return val
