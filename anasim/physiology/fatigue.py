"""Inspiratory muscle fatigue from sustained pressure generation.

Mean muscle pressure as a fraction of maximum approximates the diaphragm
tension-time index, which is sustainable up to about 0.15 (Bellemare and
Grassino 1982). Recovery has fast and persistent components (Laghi 1995);
their weights and time constants, the maximum, and the loss rate are calibrations.
"""

import math

from anasim.core.utils import clamp01


class RespiratoryFatigue:
    MAX_PRESSURE = 100.0  # cmH2O, representative unfatigued inspiratory capacity
    SUSTAINABLE_LOAD = 0.15
    LOAD_TAU_S = 10.0
    LOSS_TIME_S = 360.0
    FAST_RECOVERY_S = 1800.0
    SLOW_RECOVERY_S = 40.0 * 3600.0
    PERSISTENT_SHARE = 0.8

    def __init__(self):
        self.fast_fatigue = 0.0
        self.slow_fatigue = 0.0
        self.load = 0.0
        self.muscle_strength = 1.0  # Available motor units after neuromuscular block
        self.oxygen_fraction = 1.0

    @property
    def fatigue(self) -> float:
        return (1.0 - self.PERSISTENT_SHARE) * self.fast_fatigue + self.PERSISTENT_SHARE * self.slow_fatigue

    @property
    def available_pressure(self) -> float:
        return self.MAX_PRESSURE * self.muscle_strength * (1.0 - self.fatigue)

    def step(self, dt: float, pressure_area: float) -> None:
        """Integrate actual positive muscle pressure, including isometric effort."""
        if dt <= 0.0:  # Mechanics also advances zero-duration boundary events.
            return
        reference = self.MAX_PRESSURE * self.muscle_strength
        load = max(0.0, pressure_area) / (dt * reference) if reference > 0.0 else 0.0
        self.load += (load - self.load) * -math.expm1(-dt / self.LOAD_TAU_S)
        oxygen = clamp01(self.oxygen_fraction)
        sustainable = self.SUSTAINABLE_LOAD * oxygen
        excess = max(0.0, self.load - sustainable) / self.SUSTAINABLE_LOAD
        loss = excess * excess / self.LOSS_TIME_S
        rest = clamp01(1.0 - self.load / self.SUSTAINABLE_LOAD)
        self.fast_fatigue = self._advance(self.fast_fatigue, dt, loss, oxygen * rest / self.FAST_RECOVERY_S)
        self.slow_fatigue = self._advance(self.slow_fatigue, dt, loss, oxygen * rest / self.SLOW_RECOVERY_S)

    @staticmethod
    def _advance(fatigue: float, dt: float, loss: float, recovery: float) -> float:
        rate = loss + recovery
        if rate > 0.0:
            target = loss / rate
            return fatigue + (target - fatigue) * -math.expm1(-dt * rate)
        return fatigue
