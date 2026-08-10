"""Supported input domain for the integrated patient model."""

import math
from typing import Any

AGE_RANGE_YEARS = (18.0, 70.0)
WEIGHT_RANGE_KG = (50.0, 100.0)
HEIGHT_RANGE_CM = (150.0, 200.0)
BMI_RANGE_KG_M2 = (18.0, 32.0)
HEMOGLOBIN_RANGE_G_DL = (6.0, 20.0)
HEMATOCRIT_RANGE = (0.18, 0.60)
RENAL_FUNCTION_RANGE = (0.4, 1.0)
HEPATIC_FUNCTION_RANGE = (0.5, 1.0)


def finite_number(name: str, value: Any) -> float:
    """Return a finite float or reject the input with a field-specific error."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def bounded_number(
    name: str,
    value: Any,
    minimum: float,
    maximum: float | None = None,
    *,
    unit: str = "",
) -> float:
    """Return a finite number inside an inclusive supported range."""
    number = finite_number(name, value)
    unit_suffix = f" {unit}" if unit else ""
    if maximum is None:
        if number < minimum:
            raise ValueError(f"{name} must be at least {minimum:g}{unit_suffix}")
    elif not minimum <= number <= maximum:
        raise ValueError(
            f"{name} must be between {minimum:g} and {maximum:g}{unit_suffix}"
        )
    return number
