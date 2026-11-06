"""Cerebral dysfunction during severe oxygen supply failure.

Pressure below the autoregulatory range and low arterial oxygen content reduce
supply. Dysfunction develops over seconds and recovers after supply returns,
unless complete failure has lasted long enough to leave coma. The thresholds
are teaching calibrations.
"""

import math

from anasim.core.utils import clamp01


class CerebralModel:
    # Total time of complete supply failure after which coma persists, within
    # the 4-6 minute window for anoxic brain injury.
    COMA_EXPOSURE_S = 300.0

    def __init__(self, baseline_hb: float):
        self.baseline_hb = baseline_hb
        self.dysfunction = 0.0
        self.anoxic_s = 0.0

    def step(self, dt: float, map_mmhg: float, hb: float, saturation: float) -> None:
        # Normal supply is maintained above MAP 50. A modest extraction/flow
        # reserve compensates for reduced oxygen content before function fails.
        pressure = clamp01((map_mmhg - 10.0) / 40.0)
        oxygen = max(0.0, hb * saturation) / (self.baseline_hb * 98.0)
        supply = pressure * min(1.0, 1.5 * oxygen)
        target = clamp01((0.6 - supply) / 0.4)
        if target >= 1.0:
            self.anoxic_s += dt
        tau = 5.0 if target > self.dysfunction else 30.0
        self.dysfunction += (target - self.dysfunction) * -math.expm1(-dt / tau)

    @property
    def unresponsiveness(self) -> float:
        """Cortical dysfunction, including persistent coma; breathing follows dysfunction."""
        return 1.0 if self.anoxic_s >= self.COMA_EXPOSURE_S else self.dysfunction
