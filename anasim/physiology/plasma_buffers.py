"""Albumin and phosphate charge at 37 °C (Figge 2018, Tables A1-A2)."""

import math
from dataclasses import dataclass
from functools import lru_cache

_LN10 = math.log(10.0)
_CELLULAR_PK = 6.96
_ACID_SITES = tuple((n, 10.0**pk) for n, pk in ((1, 8.5), (98, 3.9), (18, 11.7), (1, 3.1)))
_BASIC_SITES = tuple((n, 10.0**pk) for n, pk in (
    (24, 12.5), (2, 5.8), (2, 6.15), (2, 7.51), (2, 7.685), (1, 7.86), (50, 10.3),
    (1, 7.31), (1, 6.75), (1, 6.36), (1, 4.85), (1, 5.76), (1, 6.17),
    (1, 6.73), (1, 5.82), (1, 5.10), (1, 6.70), (1, 6.20), (1, 8.0),
))
_TRANSITION_SITES = tuple(10.0**pk for pk in (7.12, 7.22, 7.10, 7.49, 7.01))


@lru_cache(maxsize=512)
def buffer_coefficients(ph: float) -> tuple[float, float, float, float]:
    """Albumin charge per g and phosphate charge per mmol, with pH derivatives."""
    x = 10.0**ph
    charge = derivative = 0.0
    for count, constant in _ACID_SITES:
        fraction = x / (x + constant)
        charge += count * fraction
        derivative += count * _LN10 * fraction * (1.0 - fraction)
    for count, constant in _BASIC_SITES:
        fraction = constant / (x + constant)
        charge -= count * fraction
        derivative += count * _LN10 * fraction * (1.0 - fraction)
    transition = x / (x + 10.0**6.9)
    shift = 10.0**(-0.4 * transition)
    slope = 1.0 + 0.4 * _LN10 * transition * (1.0 - transition)
    for constant in _TRANSITION_SITES:
        constant *= shift
        fraction = constant / (x + constant)
        charge -= fraction
        derivative += _LN10 * slope * fraction * (1.0 - fraction)
    q1 = 1.22e-2 * x
    q2 = q1 * 2.19e-7 * x
    q3 = q2 * 1.66e-12 * x
    denominator = 1.0 + q1 + q2 + q3
    phosphate = (q1 + 2.0 * q2 + 3.0 * q3) / denominator
    phosphate_derivative = _LN10 * (
        (q1 + 4.0 * q2 + 9.0 * q3) / denominator - phosphate * phosphate
    )
    return charge / 66.5, derivative / 66.5, phosphate, phosphate_derivative


@dataclass(frozen=True, slots=True)
class PlasmaBuffers:
    albumin_g_l: float
    phosphate_mmol_l: float
    cellular_mmol_l: float = 0.0  # Weak acid at pK 6.96, Wooten 2003's extracellular buffer.

    def contents(self, ph: float) -> tuple[float, float]:
        albumin, albumin_ph, phosphate, phosphate_ph = buffer_coefficients(ph)
        cellular = 1.0 / (1.0 + 10.0**(_CELLULAR_PK - ph))
        return (self.albumin_g_l * albumin + self.phosphate_mmol_l * phosphate
                + self.cellular_mmol_l * cellular,
                self.albumin_g_l * albumin_ph + self.phosphate_mmol_l * phosphate_ph
                + self.cellular_mmol_l * _LN10 * cellular * (1.0 - cellular))
