"""Two-compartment buffered crystalloid kinetics (Hahn 2017, Table 3; Hahn 2021)."""

import math
from dataclasses import dataclass

from anasim.core.utils import clamp, clamp01

# Saline's urinary elimination relative to Ringer's, fitted to Reid 2003: 2 L over 1 h gave
# median 6 h urine of 450 mL after saline and 1000 mL after Hartmann's.
SALINE_ELIMINATION = 0.25
# Renal autoregulation limits glomerular filtration pressure above normotension;
# cap Hahn's power laws near the upper end of the fitted perioperative range.
_HAHN_MAP_MAX_MMHG = 105.0
_MAX_CRYSTALLOID_K10_PER_MIN = 0.025


def _decay_integral(rate: float, minutes: float) -> float:
    return minutes if rate <= 1e-12 else -math.expm1(-rate * minutes) / rate


@dataclass(slots=True)
class CrystalloidKinetics:
    """Infused volume in plasma and exchanging interstitium (mL)."""

    central_ml: float = 0.0
    interstitial_ml: float = 0.0
    saline_fraction: float = 0.0
    dextrose_fraction: float = 0.0

    def infuse(self, volume_ml: float, saline: bool, *, dextrose: bool = False) -> None:
        total = self.central_ml + self.interstitial_ml + volume_ml
        if total > 0.0:
            self.saline_fraction += (float(saline) - self.saline_fraction) * volume_ml / total
            self.dextrose_fraction += (float(dextrose) - self.dextrose_fraction) * volume_ml / total
        self.central_ml += volume_ml

    def step(
        self, dt_s: float, map_mmHg: float, age: float, elimination_scale: float,
        available_ml: float, *, urinary_elimination: bool = True,
        circulation: float = 1.0, mcfp_mmHg: float = 0.0,
    ) -> tuple[float, float, float]:
        """Advance exact kinetics; return vascular change, urine, and sequestration (mL)."""
        if dt_s <= 0.0 or self.central_ml + self.interstitial_ml == 0.0:
            return 0.0, 0.0, 0.0
        # Distribution slows with arterial pressure, as at anesthesia induction
        # (Hahn 2021: exponent 2.34), around Hahn 2017's reference pressure.
        # During low-flow/arrest states with elevated venous filling pressure (MCFP),
        # venous/capillary hydrostatic filtration continues to extravasate crystalloid.
        flow_scale = clamp01(circulation)
        eff_map = clamp(map_mmHg, 0.0, _HAHN_MAP_MAX_MMHG)
        k12_venous = (
            0.018 * clamp01((mcfp_mmHg - 8.0) / 10.0) * clamp01((60.0 - eff_map) / 40.0)
            if mcfp_mmHg > 8.0 and eff_map < 60.0
            else 0.0
        )
        # Dextrose free-water ICF shift is handled thermodynamically by FluidBalance
        # osmotic equilibration as glucose clears (Sjöstrand 2001; Sjöstrand and Hahn 2004); hypotonic
        # free water adds ADH-suppressed water diuresis above balanced crystalloid.
        dex = self.dextrose_fraction
        k12 = 0.0956 * (eff_map / 81.3)**2.34 + k12_venous
        k21 = 0.0506 * flow_scale
        kb = 0.0052 * flow_scale
        raw_k10 = (
            0.0065 * (eff_map / 81.3)**5.2 * (age / 40.0)**-1.5 * elimination_scale * flow_scale
            * (1.0 - (1.0 - SALINE_ELIMINATION) * self.saline_fraction)
            * (1.0 + 0.55 * dex)
        ) if urinary_elimination else 0.0
        k10 = min(raw_k10, _MAX_CRYSTALLOID_K10_PER_MIN)
        central = min(self.central_ml, max(0.0, available_ml))
        # Preserve the minimum blood volume and remaining red cells.
        protected = self.central_ml - central
        peripheral = self.interstitial_ml
        a, d = k12 + k10 + kb, k21
        if a + d <= 1e-15:
            return 0.0, 0.0, 0.0
        minutes = dt_s / 60.0
        if abs(a - d) < 1e-9 and k12 * k21 == 0.0:
            decay = math.exp(-a * minutes)
            int_exp = _decay_integral(a, minutes)
            if k21 > 0.0:
                c_new = (central + d * peripheral * minutes) * decay
                ramp_int = (
                    0.5 * minutes * minutes
                    if a <= 1e-6
                    else (1.0 - (1.0 + a * minutes) * decay) / (a * a)
                )
                c_int = central * int_exp + d * peripheral * ramp_int
                p_new = peripheral * math.exp(-d * minutes)
            else:
                c_new = central * decay
                c_int = central * int_exp
                p_new = peripheral + k12 * c_int
            self.central_ml = protected + c_new
            self.interstitial_ml = p_new
            return c_new - central, k10 * c_int, kb * c_int
        discriminant = math.sqrt((a - d)**2 + 4.0 * k12 * k21)
        fast = (a + d + discriminant) / 2.0
        slow = (k10 + kb) * k21 / fast
        gap = fast - slow
        central_fast = ((a - slow) * central - k21 * peripheral) / gap
        central_slow = central - central_fast
        peripheral_fast = ((d - slow) * peripheral - k12 * central) / gap
        peripheral_slow = peripheral - peripheral_fast
        fast_decay, slow_decay = math.exp(-fast * minutes), math.exp(-slow * minutes)
        self.central_ml = protected + central_slow * slow_decay + central_fast * fast_decay
        self.interstitial_ml = peripheral_slow * slow_decay + peripheral_fast * fast_decay
        slow_int = _decay_integral(slow, minutes)
        fast_int = _decay_integral(fast, minutes)
        central_integral = central_slow * slow_int + central_fast * fast_int
        urine, sequestration = k10 * central_integral, kb * central_integral
        return self.central_ml - protected - central, urine, sequestration
