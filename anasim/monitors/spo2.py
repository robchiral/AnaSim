import numpy as np

_PPG_TEMPLATE_RESOLUTION = 200
_PPG_LANDMARKS_X = np.array([0.0, 0.175, 0.40, 0.45, 1.0])
_PPG_LANDMARKS_Y = np.array([0.0, 1.0, 0.49, 0.51, 0.0])


def _build_ppg_template(resolution: int = _PPG_TEMPLATE_RESOLUTION) -> np.ndarray:
    """
    Generate a smooth PPG waveform template using landmark interpolation.
    Based on neurokit2's approach with 4 landmarks per cycle.
    """
    phase_hr = np.linspace(0.0, 1.0, resolution)
    template = np.zeros(resolution)

    for i in range(len(_PPG_LANDMARKS_X) - 1):
        x0, x1 = _PPG_LANDMARKS_X[i], _PPG_LANDMARKS_X[i + 1]
        y0, y1 = _PPG_LANDMARKS_Y[i], _PPG_LANDMARKS_Y[i + 1]
        mask = (phase_hr >= x0) & (phase_hr <= x1)
        if not np.any(mask):
            continue

        t = (phase_hr[mask] - x0) / (x1 - x0)
        smooth_t = t * t * (3 - 2 * t)  # smoothstep for natural curvature
        template[mask] = y0 + (y1 - y0) * smooth_t

    return template


_PPG_TEMPLATE = _build_ppg_template()


class SpO2Monitor:
    """
    SpO2 Monitor using landmark-based waveform synthesis.
    Inspired by neurokit2's PPG simulation approach.
    """
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate 
        self.phase = 0.0
        
        # Pre-compute a smooth PPG/pleth template using landmarks
        # This avoids scipy dependency while achieving smooth curves
        self._template = _PPG_TEMPLATE
        self._template_max_index = _PPG_TEMPLATE.size - 1
        
    def step(self, dt: float, hr: float, saturation: float) -> tuple[float, float]:
        """
        Return (Pleth Voltage, Saturation Display Value).
        Uses pre-computed smooth template for realistic waveform.
        """
        freq = hr / 60.0
        self.phase = (self.phase + dt * freq) % 1.0
        
        # Look up value from smooth template
        idx = int(self.phase * self._template_max_index)
        pleth_voltage = self._template[idx]
        
        return pleth_voltage, saturation
