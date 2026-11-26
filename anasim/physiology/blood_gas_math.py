"""Shared blood-gas oxygen dissociation and temperature-scaling math.

Used by both the native lung (`RespiratoryModel`) and extracorporeal membrane
oxygenators so O2/CO2 equilibrium and conservation laws remain identical across
both exchange sites.
"""

from __future__ import annotations

import math

from anasim.core.utils import clamp01

O2_PER_HB = 0.0134  # STPD L O2 per L blood per g/dL Hb (1.34 mL/g).
O2_SOLUBILITY = 0.000031  # STPD L O2 per L blood per mmHg at 37 °C.
O2_EXP_DECAY = math.log(10.0) / 29.25


def temperature_pco2_scale(temp_c: float) -> float:
    """Anaerobic Rosenthal/Bradley PCO2 ratio PCO2(temp_c) / PCO2(37 °C)."""
    return 10.0 ** (-0.019 * (37.0 - temp_c))


def temperature_o2_solubility(temp_c: float) -> float:
    """Dissolved O2 solubility in blood (STPD L / L / mmHg) at temp_c."""
    return O2_SOLUBILITY * max(0.5, 1.0 + 0.013 * (37.0 - temp_c))


def temperature_p50(temp_c: float) -> float:
    """Dash 2016 Eq. 9d; pH, PCO2, and 2,3-DPG held at reference values."""
    delta = temp_c - 37.0
    return 26.8 + delta * (1.435 + delta * (0.04163 + delta * 0.000686))


def acid_base_p50(
    ph_37: float,
    pco2_37: float,
    temp_c: float = 37.0,
    *,
    temp_p50: float | None = None,
    pco2_scale: float | None = None,
    at_37: bool = False,
) -> float:
    """Dash 2016 Eqs. 9-10 at in-vivo temperature (or 37 °C for ABG warming).

    Internal chemistry tracks 37 °C alpha-stat pH and PCO2; converting to
    in-vivo pH and PCO2 applies the Bohr/CO2 shifts consistently with
    temperature-dependent P50. Dash uses pH_RBC = 0.795 pH_plasma + 1.357.
    """
    if at_37:
        ph, pco2, t_p50 = ph_37, pco2_37, 26.8
    else:
        scale = temperature_pco2_scale(temp_c) if pco2_scale is None else pco2_scale
        ph = ph_37 + 0.0147 * (37.0 - temp_c)
        pco2 = pco2_37 * scale
        t_p50 = temperature_p50(temp_c) if temp_p50 is None else temp_p50
    delta_ph = 0.795 * (ph - 7.4)
    delta_co2 = pco2 - 40.0
    ph_factor = 26.8 + delta_ph * (-25.535 + delta_ph * (10.646 - 1.764 * delta_ph))
    co2_factor = 26.8 + delta_co2 * (0.1273 + 0.0001083 * delta_co2)
    return t_p50 * ph_factor * co2_factor / 26.8**2


def saturation_at_p50(pao2: float, p50: float) -> float:
    """Dash 2016 Eqs. 6 and 11, including low-saturation cooperativity."""
    if pao2 <= 0.0:
        return 0.0
    exponent = 2.82 - 1.20 * 10.0 ** (-pao2 / 29.25)
    return 1.0 / (1.0 + (p50 / pao2) ** exponent)


def saturation_gradient(pao2: float, p50: float) -> tuple[float, float]:
    """S and dS/dPO2 share the pressure-dependent Hill exponent."""
    if pao2 <= 0.0:
        return 0.0, 0.0
    correction = 1.20 * 10.0 ** (-pao2 / 29.25)
    exponent = 2.82 - correction
    exponent_derivative = O2_EXP_DECAY * correction
    sat = 1.0 / (1.0 + (p50 / pao2) ** exponent)
    derivative = sat * (1.0 - sat) * (
        exponent / pao2 + exponent_derivative * math.log(pao2 / p50)
    )
    return sat, derivative


def oxygen_content(
    pressure: float,
    hb: float,
    p50: float,
    solubility: float = O2_SOLUBILITY,
) -> tuple[float, float]:
    """STPD L O2 per L blood and derivative per mmHg."""
    saturation, derivative = saturation_gradient(pressure, p50)
    return O2_PER_HB * hb * saturation + solubility * pressure, O2_PER_HB * hb * derivative + solubility


def po2_and_sat_from_content(
    content: float,
    hb: float,
    p50: float,
    previous: float,
    solubility: float = O2_SOLUBILITY,
) -> tuple[float, float]:
    """Invert total blood O2 content (STPD L/L) to (PO2 mmHg, saturation fraction)."""
    if content <= 0.0:
        return 0.0, 0.0
    bound_capacity = O2_PER_HB * hb
    lower, upper = 0.0, max(760.0, content / solubility)
    pressure = max(1e-6, min(upper, previous))
    for _ in range(24):
        correction = 1.20 * 10.0 ** (-pressure / 29.25)
        exponent = 2.82 - correction
        saturation = 1.0 / (1.0 + (p50 / pressure) ** exponent)
        value = bound_capacity * saturation + solubility * pressure
        residual = value - content
        if abs(residual) < 1e-12:
            return pressure, saturation
        saturation_derivative = saturation * (1.0 - saturation) * (
            exponent / pressure + O2_EXP_DECAY * correction * math.log(pressure / p50)
        )
        derivative = bound_capacity * saturation_derivative + solubility
        if residual > 0.0:
            upper = pressure
        else:
            lower = pressure
        step = residual / derivative
        candidate = pressure - step
        if abs(residual) < 1e-7 and lower < candidate < upper:
            return candidate, clamp01(saturation - saturation_derivative * step)
        pressure = candidate if lower < candidate < upper else (lower + upper) / 2.0
    raise RuntimeError("Blood O2 content solve did not converge")


def resolve_blood_contents(
    chemistry,
    co2_mmol_l: float,
    o2_l_l: float,
    base_meq_l: float,
    hb_g_dl: float,
    temp_c: float,
    pco2_guess: float = 40.0,
    ph_guess: float = 7.4,
    po2_guess: float = 95.0,
    *,
    temp_p50: float | None = None,
    pco2_scale: float | None = None,
    solubility: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Solve coupled CO2/pH/O2 equilibrium for a blood sample.

    Returns (pco2_37, ph_37, po2_in_vivo, saturation, p50, plasma_hco3).
    """
    sol = temperature_o2_solubility(temp_c) if solubility is None else solubility
    t_p50 = temperature_p50(temp_c) if temp_p50 is None else temp_p50
    c_scale = temperature_pco2_scale(temp_c) if pco2_scale is None else pco2_scale
    pco2, ph = pco2_guess, ph_guess
    p50 = acid_base_p50(ph, pco2, temp_c, temp_p50=t_p50, pco2_scale=c_scale)
    po2, sat = po2_and_sat_from_content(o2_l_l, hb_g_dl, p50, po2_guess, solubility=sol)
    hco3 = 24.5
    for _ in range(24):
        iterate = chemistry.iterate_contents(co2_mmol_l, base_meq_l, sat, pco2, ph)
        changed = iterate.ph != ph or iterate.pco2 != pco2
        ph, pco2 = iterate.ph, iterate.pco2
        hco3 = iterate.blood.plasma_bicarbonate
        if changed:
            p50 = acid_base_p50(ph, pco2, temp_c, temp_p50=t_p50, pco2_scale=c_scale)
            po2, sat = po2_and_sat_from_content(o2_l_l, hb_g_dl, p50, po2, solubility=sol)
        if iterate.converged:
            break
    else:
        raise RuntimeError("Blood O2/CO2 equilibrium did not converge")
    return pco2, ph, po2, sat, p50, hco3
