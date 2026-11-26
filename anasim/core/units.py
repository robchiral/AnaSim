"""Infusion rate unit conversion.

Model units: mg/sec for propofol, ketamine, lidocaine, rocuronium, amiodarone,
esmolol, and protamine, mU/sec for vasopressin, U/sec for heparin, and ug/sec
for the other drugs.
"""

import math
from functools import lru_cache

_RATE_UNIT_ALIASES: dict[str, str] = {
    "mg/s": "mg/sec",
    "mg/second": "mg/sec",
    "mg/m": "mg/min",
    "mg/minute": "mg/min",
    "mg/h": "mg/hr",
    "mg/hour": "mg/hr",
    "ug/s": "ug/sec",
    "ug/second": "ug/sec",
    "ug/m": "ug/min",
    "ug/minute": "ug/min",
    "ug/h": "ug/hr",
    "ug/hour": "ug/hr",
    "u/s": "u/sec",
    "u/second": "u/sec",
    "u/m": "u/min",
    "u/minute": "u/min",
    "u/h": "u/hr",
    "u/hour": "u/hr",
    "mu/s": "mu/sec",
    "mu/second": "mu/sec",
    "mu/m": "mu/min",
    "mu/minute": "mu/min",
    "mu/h": "mu/hr",
    "mu/hour": "mu/hr",
}

_RATE_CONVERSIONS: dict[tuple[str, str], float] = {
    ("mg/hr", "mg/sec"): 1.0 / 3600.0,
    ("mg/min", "mg/sec"): 1.0 / 60.0,
    ("mg/hr", "ug/sec"): 1000.0 / 3600.0,
    ("mg/min", "ug/sec"): 1000.0 / 60.0,
    ("mg/sec", "ug/sec"): 1000.0,
    ("ug/hr", "ug/sec"): 1.0 / 3600.0,
    ("ug/min", "ug/sec"): 1.0 / 60.0,
    ("ug/hr", "mg/sec"): 1.0 / 3600000.0,
    ("ug/min", "mg/sec"): 1.0 / 60000.0,
    ("u/hr", "mu/sec"): 1000.0 / 3600.0,
    ("u/min", "mu/sec"): 1000.0 / 60.0,
    ("u/sec", "mu/sec"): 1000.0,
    ("u/hr", "u/sec"): 1.0 / 3600.0,
    ("u/min", "u/sec"): 1.0 / 60.0,
    ("mu/hr", "mu/sec"): 1.0 / 3600.0,
    ("mu/min", "mu/sec"): 1.0 / 60.0,
}


@lru_cache(maxsize=64)
def normalize_rate_unit(unit: str) -> str:
    """Normalize a rate unit string, e.g. "mcg/min" to "ug/min"."""
    if not unit:
        return ""
    u = unit.strip().lower()
    u = u.replace("µ", "u").replace("μ", "u")
    u = u.replace("mcg", "ug")
    u = u.replace("per", "/")
    u = u.replace(" ", "")
    if "/kg/" in u:
        base = u.replace("/kg/", "/")
        norm_base = _RATE_UNIT_ALIASES.get(base, base)
        if "/" in norm_base:
            num, den = norm_base.split("/", 1)
            return f"{num}/kg/{den}"
    return _RATE_UNIT_ALIASES.get(u, u)


def convert_rate(
    value: float, from_unit: str, to_unit: str, *, weight_kg: float | None = None,
) -> float:
    """Convert rate units using patient weight for per-kg rates."""
    from_norm = normalize_rate_unit(from_unit)
    to_norm = normalize_rate_unit(to_unit)
    if from_norm == to_norm:
        return value
    if "/kg/" in from_norm or "/kg/" in to_norm:
        if weight_kg is None or not math.isfinite(weight_kg) or weight_kg <= 0:
            raise ValueError("Weight-based rate conversion requires a positive finite weight_kg")
        if "/kg/" in from_norm:
            value *= weight_kg
            from_norm = from_norm.replace("/kg/", "/")
        if "/kg/" in to_norm:
            value /= weight_kg
            to_norm = to_norm.replace("/kg/", "/")
        if from_norm == to_norm:
            return value
    key = (from_norm, to_norm)
    if key in _RATE_CONVERSIONS:
        return value * _RATE_CONVERSIONS[key]
    reverse = (to_norm, from_norm)
    if reverse in _RATE_CONVERSIONS:
        return value / _RATE_CONVERSIONS[reverse]
    raise ValueError(f"Unsupported rate conversion: {from_unit} -> {to_unit}")
