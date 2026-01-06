
import numpy as np


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
        self._template = self._generate_template()
        
    def _generate_template(self, resolution=200):
        """
        Generate a smooth PPG waveform template using landmark interpolation.
        Based on neurokit2's approach with 4 landmarks per cycle.
        
        Landmarks (as fraction of period):
        - Onset: 0.0, amplitude ~0
        - Systolic peak: ~0.175, amplitude ~1.0
        - Dicrotic notch: ~0.4, amplitude ~0.49
        - Diastolic peak: ~0.45, amplitude ~0.51
        - End (next onset): 1.0, amplitude ~0
        """
        # Define landmarks: (phase, amplitude)
        landmarks_x = np.array([0.0, 0.175, 0.40, 0.45, 1.0])
        landmarks_y = np.array([0.0, 1.0, 0.49, 0.51, 0.0])
        
        # Create high-resolution phase array
        phase_hr = np.linspace(0, 1, resolution)
        
        # Use cubic interpolation via numpy's polynomial fitting per segment
        # A piecewise cubic approach is used for smooth curves.
        template = np.zeros(resolution)
        
        for i in range(len(landmarks_x) - 1):
            x0, x1 = landmarks_x[i], landmarks_x[i + 1]
            y0, y1 = landmarks_y[i], landmarks_y[i + 1]
            
            # Find indices in this segment
            mask = (phase_hr >= x0) & (phase_hr <= x1)
            if not np.any(mask):
                continue
                
            # Normalized position within segment [0, 1]
            t = (phase_hr[mask] - x0) / (x1 - x0)
            
            # Smoothstep interpolation for natural curve
            # smoothstep: 3t² - 2t³ gives smooth acceleration/deceleration
            smooth_t = t * t * (3 - 2 * t)
            
            template[mask] = y0 + (y1 - y0) * smooth_t
        
        return template
        
    def step(self, dt: float, hr: float, saturation: float) -> tuple[float, float]:
        """
        Return (Pleth Voltage, Saturation Display Value).
        Uses pre-computed smooth template for realistic waveform.
        """
        freq = hr / 60.0
        self.phase += dt * freq
        if self.phase > 1.0: 
            self.phase -= 1.0
        
        # Look up value from smooth template
        idx = int(self.phase * (len(self._template) - 1))
        idx = min(idx, len(self._template) - 1)
        pleth_voltage = self._template[idx]
        
        return pleth_voltage, saturation

