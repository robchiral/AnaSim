"""Shared numeric helpers."""

from anasim.core.constants import (
    CONCENTRATION_RATIO_SATURATION,
    GAMMA_MAX,
    HILL_EPSILON,
)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


def clamp01(value: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, value))


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
