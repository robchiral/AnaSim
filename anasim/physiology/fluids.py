"""Two-compartment buffered crystalloid kinetics (Hahn 2017, Table 3; Hahn 2021)."""

import math
from dataclasses import dataclass


@dataclass
class CrystalloidKinetics:
    """Infused volume in plasma and exchanging interstitium (mL)."""

    central_ml: float = 0.0
    interstitial_ml: float = 0.0

    def step(
        self, dt_s: float, map_mmHg: float, age: float, elimination_scale: float,
        available_ml: float, *, urinary_elimination: bool = True,
    ) -> tuple[float, float, float]:
        """Advance exact kinetics; return vascular change, urine, and sequestration (mL)."""
        if dt_s <= 0.0 or self.central_ml + self.interstitial_ml == 0.0:
            return 0.0, 0.0, 0.0
        # Distribution slows with arterial pressure, as at anesthesia induction
        # (Hahn 2021: exponent 2.34), around Hahn 2017's reference pressure.
        k12 = 0.0956 * (max(0.0, map_mmHg) / 81.3)**2.34
        k21, kb = 0.0506, 0.0052
        k10 = (
            0.0065 * (max(0.0, map_mmHg) / 81.3)**5.2 * (age / 40.0)**-1.5 * elimination_scale
        ) if urinary_elimination else 0.0
        central = min(self.central_ml, max(0.0, available_ml))
        # Preserve the minimum blood volume and remaining red cells.
        protected = self.central_ml - central
        peripheral = self.interstitial_ml
        a, d = k12 + k10 + kb, k21
        discriminant = math.sqrt((a - d)**2 + 4.0 * k12 * k21)
        fast = (a + d + discriminant) / 2.0
        slow = (k10 + kb) * k21 / fast
        gap = fast - slow
        central_fast = ((a - slow) * central - k21 * peripheral) / gap
        central_slow = central - central_fast
        peripheral_fast = ((d - slow) * peripheral - k12 * central) / gap
        peripheral_slow = peripheral - peripheral_fast
        minutes = dt_s / 60.0
        fast_decay, slow_decay = math.exp(-fast * minutes), math.exp(-slow * minutes)
        self.central_ml = protected + central_slow * slow_decay + central_fast * fast_decay
        self.interstitial_ml = peripheral_slow * slow_decay + peripheral_fast * fast_decay
        central_integral = (
            central_slow * -math.expm1(-slow * minutes) / slow
            + central_fast * -math.expm1(-fast * minutes) / fast
        )
        urine, sequestration = k10 * central_integral, kb * central_integral
        return self.central_ml - protected - central, urine, sequestration
