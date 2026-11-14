"""Ventricular rhythm transitions during ischemia and resuscitation.

VF often recurs within the first two minutes after a successful shock (Koster
2008; van Alem 2003). Amiodarone converts sustained VT with a pulse more often
than lidocaine (Somberg 2002; Scheinman 1995), and neither drug prevents
rearrest (Salcido 2018). Spontaneous onset, VT degeneration, and post-shock
bradycardia recovery are calibrations.
"""

import math

import numpy as np

from anasim.core.enums import RhythmType
from anasim.core.utils import clamp01

VENTRICULAR_RHYTHMS = frozenset({RhythmType.VTACH, RhythmType.VFIB})
# mcg/mL. Lidocaine's label gives 1.5-6 mcg/mL. Amiodarone's C50 sets the gap
# in VT conversion time between the pooled 125 and 1000 mg/24 h arms in
# Scheinman 1995.
LIDOCAINE_C50 = 1.5
AMIODARONE_C50 = 0.4
LIDOCAINE_VT_CONVERSION_RATE = 1.0 / 8500.0
AMIODARONE_VT_CONVERSION_RATE = 1.0 / 3200.0
ISCHEMIC_ONSET_RATE = 1.0 / 600.0
VF_ONSET_SHARE = 0.15
VT_DEGENERATION_RATE = 1.0 / 45.0
REFIBRILLATION_RATE = 1.0 / 75.0
REFIBRILLATION_TAU_S = 150.0
POST_SHOCK_BRADY_S = 30.0


def antiarrhythmic_effects(lidocaine: float, amiodarone: float) -> tuple[float, float]:
    """Antiarrhythmic effects, 0-1. Lidocaine loses its benefit between 6 and 15 mcg/mL."""
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
                    ischemia: float, hypoxia: float, energy_failure: float,
                    energy_reserve: float = 0.0) -> RhythmType:
        """Return the rhythm after a terminating shock. The myocardial substrate persists."""
        if before not in VENTRICULAR_RHYTHMS or after in VENTRICULAR_RHYTHMS:
            return after
        # Brief VF before the energy reserve falls leaves little arrest substrate
        # for post-shock bradycardia or early refibrillation (Park 1994; Tokano 1998).
        depletion = clamp01((0.80 - energy_reserve) / 0.50)
        stress = clamp01(max(ischemia * depletion, hypoxia))
        if before == RhythmType.VFIB:
            self._post_shock_stress = clamp01((stress - 0.25) / 0.75)
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
            # Drugs convert only perfusing VT. The stronger drug effect applies.
            conversion = (
                max(lido * LIDOCAINE_VT_CONVERSION_RATE, amio * AMIODARONE_VT_CONVERSION_RATE)
                if perfusing else 0.0
            )
            events = (("degeneration", degeneration, RhythmType.VFIB),
                      ("conversion", conversion, RhythmType.SINUS))
        else:
            recurrence = (REFIBRILLATION_RATE * self._post_shock_stress * math.sqrt(stress) * viable
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
