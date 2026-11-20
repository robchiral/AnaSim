"""Chest compressions and electrical therapy.

Compressions are calibrated to adult CPR targets: forward flow near a fifth of
resting output, and diastolic pressure near the 25 mmHg needed for coronary
perfusion (Paradis 1990; Meaney 2013). First-shock biphasic VF termination and
post-shock asystole follow out-of-hospital arrest recordings (Stiell 2007;
Koster 2008; Pierce 2015). Pulseless VT responses, penalties for fine VF or
hypoxia, synchronized cardioversion, and R-on-T risk are calibrations.
"""

from enum import Enum

import numpy as np

from anasim.core.enums import RhythmType
from anasim.core.utils import clamp01

COMPRESSION_RATE = 110.0  # /min, within the guideline 100-120
# Share of resting cardiac output at normal filling and vascular resistance;
# less when underfilled or when pressor afterload impedes passive forward flow
# (Paradis 1990; Cantineau 1994; Kim 2025). Calibrated so anesthetized adult
# compressions produce ~20% resting output and ~70/25 mmHg arterial pressure.
COMPRESSION_OUTPUT_FRACTION = 0.25
COMPRESSION_AFTERLOAD_COUPLING = 0.45
# Each compression raises arterial and venous pressure together, so the pulse
# adds to arterial pressure without driving flow.
COMPRESSION_PULSE_MMHG = 45.0
COMPRESSION_DUTY = 0.5  # Share of each cycle spent compressing

SHOCK_ENERGY_J = 200.0  # Biphasic
# The ZOLL R Series charges to 200 J in under 7 s and disarms 60 s after charging.
CHARGE_TIME_S = 5.0
CHARGE_HOLD_S = 60.0
# First 200 J biphasic shocks terminate ~90% of VF (Stiell 2007; Koster 2008).
# Penalties for fine VF or hypoxia are calibrations.
DEFIBRILLATION_SUCCESS = 0.9
FINE_VF_SUCCESS_LOSS = 0.6
# Synchronized shocks.
CARDIOVERSION_SUCCESS = {
    RhythmType.AFIB: 0.85,
    RhythmType.SVT: 0.95,
    RhythmType.VTACH: 0.95,
}
# VF risk from an unsynchronized shock near the T-wave peak, a teaching calibration.
R_ON_T_PROBABILITY = 0.15


def compression_pressures(perfusion_pressure: float, venous_pressure: float) -> tuple[float, float]:
    """Arterial pressure at the compression peak and between compressions (mmHg)."""
    relaxation = venous_pressure + perfusion_pressure
    return relaxation + COMPRESSION_PULSE_MMHG, relaxation


class ShockResult(Enum):
    NOT_READY = "not_ready"
    PENDING = "pending"  # Charged; waiting for the next R wave
    NOT_DELIVERED = "not_delivered"  # Synchronized mode found no R wave
    DELIVERED = "delivered"


class Defibrillator:
    """One 200 J charge per shock."""

    def __init__(self):
        self.ready_at: float | None = None
        self.waiting_for_r = False
        self.last_shock: tuple[float, bool] | None = None

    def status(self, time: float) -> str:
        if self.ready_at is None:
            return "idle"
        eps = 1e-9 * max(1.0, abs(time))
        if time >= self.ready_at + CHARGE_HOLD_S - eps:
            self.disarm()
            return "idle"
        if time < self.ready_at - eps:
            return "charging"
        return "waiting_for_r" if self.waiting_for_r else "ready"

    def charge(self, time: float) -> None:
        if self.status(time) == "idle":
            self.ready_at = time + CHARGE_TIME_S

    def disarm(self) -> None:
        self.ready_at = None
        self.waiting_for_r = False

    def consume(self, time: float, synchronized: bool) -> None:
        self.disarm()
        self.last_shock = time, synchronized


def shock_outcome(rhythm: RhythmType, *, synchronized: bool, energy_failure: float,
                  electrically_silent: bool, vulnerable: bool, antiarrhythmic: float,
                  rng: np.random.Generator, hypoxia: float = 0.0,
                  perfusing: bool = True, energy_reserve: float = 1.0) -> RhythmType:
    """Return the rhythm after a discharge.

    A terminated arrhythmia becomes sinus rhythm, or transient post-shock
    asystole after prolonged pulseless arrest (Pierce 2015); whether an
    organized rhythm perfuses depends on the myocardium's energy state.
    """
    if electrically_silent:
        return RhythmType.ASYSTOLE
    if rhythm == RhythmType.ASYSTOLE:
        return rhythm
    depletion = max(energy_failure, 0.8 * clamp01(hypoxia))
    if not synchronized and rhythm != RhythmType.VFIB and perfusing and vulnerable and rng.random() < R_ON_T_PROBABILITY:
        return RhythmType.VFIB
    if rhythm == RhythmType.VFIB or (rhythm == RhythmType.VTACH and (not perfusing or not synchronized)):
        success = DEFIBRILLATION_SUCCESS * (1.0 - FINE_VF_SUCCESS_LOSS * depletion)
        # Either drug reduced the shocks needed (Kudenchuk 2016). The effect size is a calibration.
        success += 0.5 * (1.0 - success) * clamp01(antiarrhythmic)
        u = rng.random()
        if u >= success:
            return rhythm
        if rhythm == RhythmType.VFIB or not perfusing:
            arrest_depth = clamp01((max(1.0 - energy_reserve, energy_failure, clamp01(hypoxia)) - 0.25) / 0.75)
            asystole_share = min(0.78, 0.60 * arrest_depth + 0.25 * energy_failure)
            if u >= success * (1.0 - asystole_share):
                return RhythmType.ASYSTOLE
        return RhythmType.SINUS
    success = CARDIOVERSION_SUCCESS.get(rhythm, 0.0) * (1.0 - 0.5 * depletion)
    return RhythmType.SINUS if success and rng.random() < success else rhythm
