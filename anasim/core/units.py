"""
Unit conversion helpers for rate handling.

Internal convention:
- Propofol/Rocuronium: mg/sec
- Remifentanil/Catecholamines/Inotropes: ug/sec
- Vasopressin: mU/sec
"""

from typing import Dict, Tuple


_RATE_UNIT_ALIASES: Dict[str, str] = {
    "mg/s": "mg/sec",
    "mg/second": "mg/sec",
    "mg/h": "mg/hr",
    "mcg/s": "ug/sec",
    "mcg/sec": "ug/sec",
    "mcg/min": "ug/min",
    "mcg/hr": "ug/hr",
    "u/s": "u/sec",
    "u/min": "u/min",
    "u/hr": "u/hr",
    "mu/s": "mu/sec",
    "mu/min": "mu/min",
    "mu/hr": "mu/hr",
}

# Conversion factors between rate units (multiplicative).
_RATE_CONVERSIONS: Dict[Tuple[str, str], float] = {
    ("mg/hr", "mg/sec"): 1.0 / 3600.0,
    ("mg/min", "mg/sec"): 1.0 / 60.0,
    ("mg/sec", "mg/sec"): 1.0,
    ("ug/hr", "ug/sec"): 1.0 / 3600.0,
    ("ug/min", "ug/sec"): 1.0 / 60.0,
    ("ug/sec", "ug/sec"): 1.0,
    ("u/hr", "mu/sec"): 1000.0 / 3600.0,
    ("u/min", "mu/sec"): 1000.0 / 60.0,
    ("u/sec", "mu/sec"): 1000.0,
    ("mu/hr", "mu/sec"): 1.0 / 3600.0,
    ("mu/min", "mu/sec"): 1.0 / 60.0,
    ("mu/sec", "mu/sec"): 1.0,
}


def normalize_rate_unit(unit: str) -> str:
    """Normalize rate unit strings to canonical lowercase form."""
    if not unit:
        return ""
    u = unit.strip()
    u = u.replace("µ", "u").replace("μ", "u")
    u = u.replace("mcg", "ug")
    u = u.replace("per", "/")
    u = u.replace(" ", "")
    u = u.lower()
    return _RATE_UNIT_ALIASES.get(u, u)


def convert_rate(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert rate value between supported units.

    Raises ValueError if conversion is unsupported.
    """
    from_norm = normalize_rate_unit(from_unit)
    to_norm = normalize_rate_unit(to_unit)
    if from_norm == to_norm:
        return value
    key = (from_norm, to_norm)
    if key in _RATE_CONVERSIONS:
        return value * _RATE_CONVERSIONS[key]
    reverse = (to_norm, from_norm)
    if reverse in _RATE_CONVERSIONS:
        return value / _RATE_CONVERSIONS[reverse]
    raise ValueError(f"Unsupported rate conversion: {from_unit} -> {to_unit}")
