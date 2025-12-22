"""
Shared utility functions for AnaSim.
"""

from core.constants import GAMMA_MAX, HILL_EPSILON, CONCENTRATION_RATIO_SATURATION


def clamp(value: float, low: float, high: float) -> float:
    """
    Clamp value to the inclusive range [low, high].
    """
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


def clamp01(value: float) -> float:
    """
    Clamp value to the inclusive range [0.0, 1.0].
    """
    return clamp(value, 0.0, 1.0)


def hill_function(c: float, c50: float, gamma: float) -> float:
    """
    Generic Hill/sigmoidal Emax function with numerical safeguards.
    
    Returns value between 0.0 and 1.0.
    
    Args:
        c: Drug concentration or effect value
        c50: Half-maximal effect concentration (EC50/IC50)
        gamma: Hill coefficient (steepness)
    
    Returns:
        Effect fraction (0 to 1)
        
    Numerical safeguards:
        - c <= 0: returns 0.0 (no effect)
        - c50 <= 0: returns 0.0 (invalid parameter)
        - gamma <= 0: returns 0.0 (invalid parameter)
        - Overflow protection: caps gamma at GAMMA_MAX to prevent numerical issues
    """
    # Guard against invalid inputs
    if c <= 0 or c50 <= 0 or gamma <= 0:
        return 0.0
    
    # Cap gamma to prevent overflow with very steep curves
    gamma = min(gamma, GAMMA_MAX)
    
    # Safe exponentiation using ratio to avoid overflow
    ratio = c / c50
    if ratio > CONCENTRATION_RATIO_SATURATION:  # Very high concentrations - near saturation
        return 1.0 - 1e-6
    
    ratio_g = ratio ** gamma
    
    # Hill equation: c^γ / (c50^γ + c^γ) = ratio^γ / (1 + ratio^γ)
    return ratio_g / (1.0 + ratio_g + HILL_EPSILON)


def pao2_to_sao2(pao2: float) -> float:
    """
    Convert PaO2 (mmHg) to SaO2 (%) using a Severinghaus-style fit.
    """
    pao2 = max(0.0, pao2)
    p3 = pao2 ** 3
    return 100.0 * (p3 + 150.0 * pao2) / (p3 + 150.0 * pao2 + 23400.0)
