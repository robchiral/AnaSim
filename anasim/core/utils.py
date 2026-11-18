"""Shared numeric helpers."""

import math

import numpy as np

from anasim.core.constants import (
    CONCENTRATION_RATIO_SATURATION,
    GAMMA_MAX,
    HILL_EPSILON,
)


def clamp(value: float, low: float, high: float) -> float:
    # Fast path for finite interior values; NaN or inverted bounds evaluate False.
    if low < value < high:
        return value
    # Ordered comparisons retain min/max behavior for equal or non-finite bounds.
    if math.isnan(value):
        return value
    if low > high:
        low, high = high, low
    if value < high:
        return value if value > low else low
    return high if high > low else low


def clamp01(value: float) -> float:
    if value > 0.0:
        return value if value < 1.0 else 1.0
    return 0.0  # Includes NaN.


def fill_response_rows(rows: np.ndarray, transition: np.ndarray, target: int) -> None:
    """Fill row k with the target row of transition raised to power k + 1."""
    if len(rows) == 0:
        return
    rows[0] = transition[target]
    advance = transition
    filled = 1
    while filled < len(rows):
        count = min(filled, len(rows) - filled)
        rows[filled:filled + count] = rows[:count] @ advance
        filled += count
        if filled < len(rows):
            advance = advance @ advance


def hill_function(c: float, c50: float, gamma: float) -> float:
    """Fractional effect (c/c50)^gamma / (1 + (c/c50)^gamma), in [0, 1).

    Returns 0 for non-positive inputs and caps gamma and c/c50 to avoid overflow.
    """
    if c <= 0 or c50 <= 0 or gamma <= 0:
        return 0.0
    gamma = min(gamma, GAMMA_MAX)
    ratio = c / c50
    if ratio > CONCENTRATION_RATIO_SATURATION:
        return 1.0 - 1e-6
    ratio_g = ratio ** gamma
    return ratio_g / (1.0 + ratio_g + HILL_EPSILON)
