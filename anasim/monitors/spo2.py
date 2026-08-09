import math

import numpy as np

from .cardiac_cycle import CardiacCycleSample

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
    def __init__(self, response_tau_s: float = 4.0, peripheral_delay_s: float = 0.18):
        self.response_tau_s = response_tau_s
        self.peripheral_delay_s = peripheral_delay_s
        if self.response_tau_s <= 0.0:
            raise ValueError("response_tau_s must be greater than zero")
        if self.peripheral_delay_s < 0.0:
            raise ValueError("peripheral_delay_s cannot be negative")
        self.display_saturation = None
        self.signal_valid = True
        
        # Pre-compute a smooth PPG/pleth template using landmarks
        # This avoids scipy dependency while achieving smooth curves
        self._template = _PPG_TEMPLATE
        self._template_max_index = _PPG_TEMPLATE.size - 1
        
    def step(
        self,
        dt: float,
        cycle: CardiacCycleSample,
        saturation: float,
        perfusion: float = 1.0,
    ) -> tuple[float, float]:
        """
        Return (Pleth Voltage, Saturation Display Value).
        Uses pre-computed smooth template for realistic waveform.
        """
        if dt <= 0.0:
            raise ValueError("SpO2 monitor dt must be greater than zero")

        perf = max(0.0, min(1.0, perfusion))
        if cycle.organized:
            phase = cycle.delayed_phase(self.peripheral_delay_s)
            idx = int(phase * self._template_max_index)
            pleth_voltage = self._template[idx] * (0.2 + 0.8 * perf)
        else:
            pleth_voltage = 0.0

        target = max(40.0, min(100.0, saturation))
        if self.display_saturation is None:
            self.display_saturation = target

        # Finger probes trail arterial saturation, and low perfusion slows the
        # response instead of deterministically manufacturing hypoxaemia.
        tau = self.response_tau_s * (1.0 + 2.0 * (1.0 - perf))
        alpha = 1.0 - math.exp(-dt / tau)
        self.display_saturation += alpha * (target - self.display_saturation)
        self.signal_valid = perf >= 0.08
        return pleth_voltage, self.display_saturation
