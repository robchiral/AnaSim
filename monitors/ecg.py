
import numpy as np
import random


class ECGMonitor:
    """
    ECG Monitor using Gaussian-based PQRST synthesis.
    Inspired by neurokit2's ECGSYN dynamical model parameters.
    Uses pre-computed template for efficiency.
    """
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate 
        self.phase = 0.0
        
        # Pre-compute a smooth ECG template using Gaussian waves
        self._template = self._generate_template()
        
    def _generate_template(self, resolution=500):
        """
        Generate ECG template using Gaussian functions for PQRST waves.
        
        Based on neurokit2's ECGSYN parameters:
        - ti: angles of extrema in degrees -> P(-70), Q(-15), R(0), S(15), T(100)
        - ai: amplitudes -> P(1.2), Q(-5), R(30), S(-7.5), T(0.75)
        - bi: Gaussian width -> P(0.25), Q(0.1), R(0.1), S(0.1), T(0.4)
        
        Converted to phase [0,1] where 0.5 corresponds to R peak (angle 0).
        """
        # Wave parameters: (center_phase, amplitude, width)
        # Center phase converted from degrees: phase = (degrees + 180) / 360
        # Angles: P=-70°, Q=-15°, R=0°, S=15°, T=100°
        # -> Phases: P=0.306, Q=0.458, R=0.5, S=0.542, T=0.778
        waves = [
            # (center, amplitude, width) - width in phase units
            (0.306, 0.08, 0.035),   # P wave: small positive
            (0.458, -0.12, 0.012),  # Q wave: small negative
            (0.500, 1.0, 0.015),    # R wave: large positive (main spike)
            (0.542, -0.20, 0.012),  # S wave: medium negative
            (0.778, 0.15, 0.055),   # T wave: medium positive, broader
        ]
        
        # Create high-resolution phase array
        phase_arr = np.linspace(0, 1, resolution)
        template = np.zeros(resolution)
        
        # Sum Gaussian contributions from each wave
        for center, amplitude, width in waves:
            # Gaussian function: a * exp(-0.5 * ((x - c) / w)^2)
            gaussian = amplitude * np.exp(-0.5 * ((phase_arr - center) / width) ** 2)
            template += gaussian
        
        return template
        
    def step(self, dt: float, state_hr: float) -> float:
        """
        Return the next voltage sample.
        Uses pre-computed Gaussian template for realistic PQRST complex.
        """
        # Update phase using modulo for clean wrapping
        freq = state_hr / 60.0
        self.phase += dt * freq
        self.phase = self.phase % 1.0
        
        # Look up value from smooth template
        idx = int(self.phase * (len(self._template) - 1))
        idx = min(idx, len(self._template) - 1)
        val = self._template[idx]
        
        # Add subtle baseline noise for realism
        val += random.uniform(-0.015, 0.015)
        
        return val
