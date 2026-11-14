"""Chest compressions and electrical therapy.

Compressions are calibrated to adult CPR targets: forward flow near a fifth of
resting output, and diastolic pressure near the 25 mmHg needed for coronary
perfusion (Paradis 1990; Meaney 2013). Shock success rates are teaching
calibrations.
"""

from enum import Enum

import numpy as np

from anasim.core.enums import RhythmType
from anasim.core.utils import clamp01

COMPRESSION_RATE = 110.0  # /min, within the guideline 100-120
# Share of resting cardiac output at normal filling and vascular resistance;
# less when underfilled or when pressor afterload impedes passive forward flow
# (Paradis 1990; Cantineau 1994; Kim 2025).
COMPRESSION_OUTPUT_FRACTION = 0.2
COMPRESSION_AFTERLOAD_COUPLING = 0.45
# Each compression raises arterial and venous pressure together, so the pulse
# adds to arterial pressure without driving flow.
COMPRESSION_PULSE_MMHG = 45.0
COMPRESSION_DUTY = 0.5  # Share of each cycle spent compressing

SHOCK_ENERGY_J = 200.0  # Biphasic
# The ZOLL R Series charges to 200 J in under 7 s and disarms 60 s after charging.
CHARGE_TIME_S = 5.0
CHARGE_HOLD_S = 60.0
# First biphasic shocks terminate about 90% of VF; fine VF responds less often.
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
        if self.ready_at is not None and time >= self.ready_at + CHARGE_HOLD_S - 1e-9:
            self.disarm()
        if self.ready_at is None:
            return "idle"
        if time < self.ready_at - 1e-9:
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
                  rng: np.random.Generator) -> RhythmType:
    """Return the rhythm after a discharge.

    A terminated arrhythmia becomes sinus rhythm; whether it perfuses depends on
    the myocardium's energy state.
    """
    if electrically_silent:
        return RhythmType.ASYSTOLE
    if rhythm == RhythmType.ASYSTOLE:
        return rhythm
    if rhythm == RhythmType.VFIB or (rhythm == RhythmType.VTACH and not synchronized):
        success = DEFIBRILLATION_SUCCESS * (1.0 - FINE_VF_SUCCESS_LOSS * energy_failure)
        # Either drug reduced the shocks needed (Kudenchuk 2016). The effect size is a calibration.
        success += 0.5 * (1.0 - success) * clamp01(antiarrhythmic)
        return RhythmType.SINUS if rng.random() < success else rhythm
    if not synchronized and vulnerable and rng.random() < R_ON_T_PROBABILITY:
        return RhythmType.VFIB
    success = CARDIOVERSION_SUCCESS.get(rhythm, 0.0)
    return RhythmType.SINUS if success and rng.random() < success else rhythm
