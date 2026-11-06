"""Ventricular rhythm transitions during ischemia and resuscitation.

VF often recurs after a successful shock (Koster 2008; van Alem 2003), and
amiodarone converts sustained VT more often than lidocaine (Somberg 2002).
Event rates and concentration-response curves are calibrations.
"""

import math

import numpy as np

from anasim.core.enums import RhythmType
from anasim.core.utils import clamp01

VENTRICULAR_RHYTHMS = frozenset({RhythmType.VTACH, RhythmType.VFIB})
# mcg/mL. Lidocaine's label gives a therapeutic range of 1.5-6 mcg/mL.
LIDOCAINE_C50 = 1.5
AMIODARONE_C50 = 1.0
ISCHEMIC_ONSET_RATE = 1.0 / 600.0
VF_ONSET_SHARE = 0.15
VT_DEGENERATION_RATE = 1.0 / 45.0
REFIBRILLATION_RATE = 1.0 / 50.0
REFIBRILLATION_TAU_S = 90.0
POST_SHOCK_BRADY_S = 30.0


def antiarrhythmic_effects(lidocaine: float, amiodarone: float) -> tuple[float, float]:
    """Antiarrhythmic effects, 0-1; lidocaine's is lost between 6 and 15 mcg/mL."""
    lido, amio = max(0.0, lidocaine), max(0.0, amiodarone)
    therapeutic_lido = lido / (lido + LIDOCAINE_C50) * (1.0 - clamp01((lido - 6.0) / 9.0))
    return therapeutic_lido, amio / (amio + AMIODARONE_C50)


class ArrhythmiaModel:
    """Integrates each transition hazard to an exponential threshold, independent of step size."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._remaining: dict[str, float] = {}
        self._rhythm: RhythmType | None = None
        self._post_shock_s = math.inf
        self._post_shock_stress = 0.0
        self._recovering_brady = False

    def reset(self) -> None:
        """Clear transient shock effects when the instructor selects a rhythm."""
        self._remaining.clear()
        self._rhythm = None
        self._post_shock_s = math.inf
        self._post_shock_stress = 0.0
        self._recovering_brady = False

    def after_shock(self, before: RhythmType, after: RhythmType, *,
                    ischemia: float, hypoxia: float, energy_failure: float) -> RhythmType:
        """Return the rhythm after a terminating shock; the myocardial substrate persists."""
        if before not in VENTRICULAR_RHYTHMS or after in VENTRICULAR_RHYTHMS:
            return after
        stress = clamp01(max(ischemia, hypoxia))
        if before == RhythmType.VFIB:
            self._post_shock_stress = stress
        else:
            self._post_shock_stress *= math.exp(-self._post_shock_s / REFIBRILLATION_TAU_S)
        self._post_shock_s = 0.0
        self._recovering_brady = after == RhythmType.SINUS and (stress > 0.5 or energy_failure > 0.1)
        return RhythmType.SINUS_BRADY if self._recovering_brady else after

    def step(self, dt: float, rhythm: RhythmType, *, ischemia: float, hypoxia: float,
             energy_failure: float, electrically_silent: bool, perfusing: bool,
             lidocaine: float, amiodarone: float) -> RhythmType:
        if rhythm != self._rhythm:
            self._rhythm = rhythm
            self._remaining.clear()
            if rhythm != RhythmType.SINUS_BRADY:
                self._recovering_brady = False
        self._post_shock_s += dt
        stress = clamp01(max(ischemia, hypoxia))
        if (self._recovering_brady and self._post_shock_s >= POST_SHOCK_BRADY_S
                and stress < 0.5 and energy_failure < 0.2):
            self._recovering_brady = False
            return RhythmType.SINUS
        if electrically_silent or rhythm in (RhythmType.ASYSTOLE, RhythmType.VFIB):
            return rhythm

        # A nearly exhausted myocardium slows to asystole instead.
        viable = clamp01((0.8 - energy_failure) / 0.3)
        substrate = clamp01((stress - 0.6) / 0.4) * viable
        lido, amio = antiarrhythmic_effects(lidocaine, amiodarone)
        if rhythm == RhythmType.VTACH:
            degeneration = VT_DEGENERATION_RATE * substrate
            # Drugs convert only perfusing VT; the stronger drug effect applies.
            conversion = max(lido / 600.0, amio / 120.0) if perfusing else 0.0
            events = (("degeneration", degeneration, RhythmType.VFIB),
                      ("conversion", conversion, RhythmType.SINUS))
        else:
            recurrence = (REFIBRILLATION_RATE * self._post_shock_stress * stress * viable
                          * math.exp(-self._post_shock_s / REFIBRILLATION_TAU_S))
            # Without the heart's own output, as in PEA, ischemia leads to asystole.
            onset = ISCHEMIC_ONSET_RATE * substrate if perfusing else 0.0
            events = (("recurrence", recurrence, RhythmType.VFIB),
                      ("onset", onset, RhythmType.VTACH))
        earliest = math.inf
        following: RhythmType = rhythm
        for name, rate, target in events:
            if rate <= 0.0:
                continue
            if name not in self._remaining:
                self._remaining[name] = float(self.rng.exponential())
            time_to_event = self._remaining[name] / rate
            self._remaining[name] -= rate * dt
            if time_to_event <= dt and time_to_event < earliest:
                earliest, following = time_to_event, target
                if name == "onset" and self.rng.random() < VF_ONSET_SHARE:
                    following = RhythmType.VFIB
        return following
