"""Albumin and phosphate charge at 37 °C (Figge 2018, Tables A1-A2)."""

import math
from dataclasses import dataclass

_LN10 = math.log(10.0)
_CELLULAR_PK = 6.96
_TRANSITION_K = 10.0**6.9
_ACID_RAW = ((1, 8.5), (98, 3.9), (18, 11.7), (1, 3.1))
_BASIC_RAW = (
    (24, 12.5), (2, 5.8), (2, 6.15), (2, 7.51), (2, 7.685), (1, 7.86), (50, 10.3),
    (1, 7.31), (1, 6.75), (1, 6.36), (1, 4.85), (1, 5.76), (1, 6.17),
    (1, 6.73), (1, 5.82), (1, 5.10), (1, 6.70), (1, 6.20), (1, 8.0),
)
_ACID_TOTAL = float(sum(n for n, _ in _ACID_RAW))
_FIXED_SITES = tuple((float(n), 10.0**pk) for n, pk in (*_ACID_RAW, *_BASIC_RAW))
_TRANSITION_SITES = tuple(10.0**pk for pk in (7.12, 7.22, 7.10, 7.49, 7.01))

_TABLE_MIN_PH = 5.0
_TABLE_MAX_PH = 10.0
_TABLE_STEPS = 1250  # h = 0.004 pH units; cubic Hermite error < 2e-12 (values) and < 2e-9 (derivatives).
_TABLE_H = (_TABLE_MAX_PH - _TABLE_MIN_PH) / _TABLE_STEPS
_TABLE_INV_H = _TABLE_STEPS / (_TABLE_MAX_PH - _TABLE_MIN_PH)


def _exact_buffer_coefficients(ph: float) -> tuple[float, float, float, float]:
    """Exact albumin charge per g and phosphate charge per mmol, with pH derivatives."""
    x = 10.0**ph
    charge = _ACID_TOTAL
    deriv = 0.0
    for count, constant in _FIXED_SITES:
        fraction = constant / (x + constant)
        charge -= count * fraction
        deriv += count * fraction * (1.0 - fraction)
    transition = x / (x + _TRANSITION_K)
    shift = 10.0**(-0.4 * transition)
    slope = 1.0 + 0.4 * _LN10 * transition * (1.0 - transition)
    trans_deriv = 0.0
    for constant in _TRANSITION_SITES:
        constant *= shift
        fraction = constant / (x + constant)
        charge -= fraction
        trans_deriv += fraction * (1.0 - fraction)
    derivative = _LN10 * (deriv + slope * trans_deriv)
    q1 = 1.22e-2 * x
    q2 = q1 * 2.19e-7 * x
    q3 = q2 * 1.66e-12 * x
    denominator = 1.0 + q1 + q2 + q3
    phosphate = (q1 + 2.0 * q2 + 3.0 * q3) / denominator
    phosphate_derivative = _LN10 * (
        (q1 + 4.0 * q2 + 9.0 * q3) / denominator - phosphate * phosphate
    )
    return charge / 66.5, derivative / 66.5, phosphate, phosphate_derivative


def _build_hermite_table() -> tuple[tuple[float, float, float, float, float, float, float, float], ...]:
    knots = [_exact_buffer_coefficients(_TABLE_MIN_PH + i * _TABLE_H) for i in range(_TABLE_STEPS + 1)]
    rows = []
    h = _TABLE_H
    for i in range(_TABLE_STEPS):
        a0, da0, p0, dp0 = knots[i]
        a1, da1, p1, dp1 = knots[i + 1]
        m0, m1 = da0 * h, da1 * h
        ac2 = 3.0 * (a1 - a0) - 2.0 * m0 - m1
        ac3 = 2.0 * (a0 - a1) + m0 + m1
        n0, n1 = dp0 * h, dp1 * h
        pc2 = 3.0 * (p1 - p0) - 2.0 * n0 - n1
        pc3 = 2.0 * (p0 - p1) + n0 + n1
        rows.append((a0, m0, ac2, ac3, p0, n0, pc2, pc3))
    return tuple(rows)


_HERMITE_TABLE = _build_hermite_table()


def buffer_coefficients(ph: float) -> tuple[float, float, float, float]:
    """Albumin charge per g and phosphate charge per mmol, with pH derivatives."""
    if ph < _TABLE_MIN_PH or ph >= _TABLE_MAX_PH:
        return _exact_buffer_coefficients(ph)
    scaled = (ph - _TABLE_MIN_PH) * _TABLE_INV_H
    idx = int(scaled)
    t = scaled - idx
    a0, a1, a2, a3, p0, p1, p2, p3 = _HERMITE_TABLE[idx]
    alb = ((a3 * t + a2) * t + a1) * t + a0
    alb_ph = ((3.0 * a3 * t + 2.0 * a2) * t + a1) * _TABLE_INV_H
    pi = ((p3 * t + p2) * t + p1) * t + p0
    pi_ph = ((3.0 * p3 * t + 2.0 * p2) * t + p1) * _TABLE_INV_H
    return alb, alb_ph, pi, pi_ph


@dataclass(frozen=True, slots=True)
class PlasmaBuffers:
    albumin_g_l: float
    phosphate_mmol_l: float
    cellular_mmol_l: float = 0.0  # Weak acid at pK 6.96 (Rees 2010), for extracellular buffering (Wooten 2010).

    def contents(self, ph: float) -> tuple[float, float]:
        albumin, albumin_ph, phosphate, phosphate_ph = buffer_coefficients(ph)
        if self.cellular_mmol_l == 0.0:
            return (
                self.albumin_g_l * albumin + self.phosphate_mmol_l * phosphate,
                self.albumin_g_l * albumin_ph + self.phosphate_mmol_l * phosphate_ph,
            )
        cellular = 1.0 / (1.0 + 10.0**(_CELLULAR_PK - ph))
        return (self.albumin_g_l * albumin + self.phosphate_mmol_l * phosphate
                + self.cellular_mmol_l * cellular,
                self.albumin_g_l * albumin_ph + self.phosphate_mmol_l * phosphate_ph
                + self.cellular_mmol_l * _LN10 * cellular * (1.0 - cellular))
