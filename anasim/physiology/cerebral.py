"""Cerebral oxygen failure, persistent coma, and reversible CO2 narcosis.

Pressure below the autoregulatory range and low arterial oxygen content reduce
supply. Electrical function fails as tissue oxygen falls from 60% to 35% of
normal, and neurons are injured below 20% (Astrup 1981), more slowly the more
supply remains (Jones 1981). Oxygen stores delay failure: after circulatory
arrest, it starts at about 4 s and the EEG is flat at about 11 s (de Vries
1998). Acute electrical suppression can recover while coma persists.
"""

import math

from anasim.core.utils import clamp01


class CerebralModel:
    # Time without supply after which coma persists, a teaching threshold.
    # Supply of 10-20% counts partially.
    COMA_EXPOSURE_S = 300.0
    TISSUE_OXYGEN_TAU_S = 6.0
    # The PaCO2 and pH ramps follow Westlake 1955 (Table III); the time
    # constant is a teaching calibration.
    CO2_RESPONSE_TAU_S = 30.0

    def __init__(self, baseline_hb: float):
        self.baseline_hb = baseline_hb
        self.tissue_oxygen = 1.0  # Share of normal
        self.dysfunction = 0.0
        self.anoxic_s = 0.0
        self.co2_narcosis = 0.0

    def step(self, dt: float, map_mmhg: float, hb: float, saturation: float,
             pa_co2: float, ph: float) -> None:
        # Normal supply is maintained above MAP 50. A modest extraction/flow
        # reserve compensates for reduced oxygen content before function fails.
        pressure = clamp01((map_mmhg - 10.0) / 40.0)
        oxygen = max(0.0, hb * saturation) / (self.baseline_hb * 98.0)
        supply = pressure * min(1.0, 1.5 * oxygen)
        self.tissue_oxygen += (supply - self.tissue_oxygen) * -math.expm1(-dt / self.TISSUE_OXYGEN_TAU_S)
        target = clamp01((0.6 - self.tissue_oxygen) / 0.25)
        self.anoxic_s += dt * clamp01(2.0 - self.tissue_oxygen / 0.1)
        tau = 3.0 if target > self.dysfunction else 30.0
        self.dysfunction += (target - self.dysfunction) * -math.expm1(-dt / tau)
        # Acute hypercapnia impairs response more with acidemia; compensated
        # retention does so only above 100 mmHg. Gating pH by PaCO2 keeps
        # metabolic acidosis from causing narcosis.
        hypercapnia = clamp01((pa_co2 - 60.0) / 40.0)
        acidemia = clamp01((7.3 - ph) / 0.2)
        target_co2 = max(clamp01((pa_co2 - 100.0) / 40.0), hypercapnia * acidemia)
        self.co2_narcosis += (target_co2 - self.co2_narcosis) * -math.expm1(-dt / self.CO2_RESPONSE_TAU_S)

    @property
    def cortical_function(self) -> float:
        """Cortical electrical function for BIS; coma and CO2 leave it unchanged (Turnbull 2008)."""
        return 1.0 - self.dysfunction

    @property
    def unresponsiveness(self) -> float:
        """Loss of response from oxygen failure, persistent coma, or CO2 narcosis."""
        if self.anoxic_s >= self.COMA_EXPOSURE_S:
            return 1.0
        return 1.0 - (1.0 - self.dysfunction) * (1.0 - self.co2_narcosis)
