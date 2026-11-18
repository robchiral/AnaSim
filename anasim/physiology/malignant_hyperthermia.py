"""Triggered skeletal muscle hypermetabolism and its suppression by dantrolene.

The crisis time course and the dantrolene concentration for crisis control
are calibrated to a teaching course.
"""

import math

from anasim.core.utils import hill_function


def dantrolene_twitch_fraction(concentration: float) -> float:
    """Remaining twitch height without TOF fade (mcg/mL).

    Flewellen 1983: twitch depression in volunteers reached a 75% plateau at
    about 2.4 mg/kg, while ventilation was unchanged, so respiratory muscles
    are spared.
    """
    return 1.0 - 0.75 * hill_function(concentration, 1.0, 2.0)


def triggering_exposure(sevo_mac: float, succinylcholine_cp: float) -> bool:
    return sevo_mac > 0.001 or succinylcholine_cp > 0.01


class MalignantHyperthermia:
    """Crisis severity, 0-1."""

    def __init__(self):
        self._active = False
        self._susceptible = False
        self.severity = 0.0
        self._controlled = False

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = bool(value)
        self._susceptible = self._active
        self._controlled = False

    def step(self, dt: float, *, sevo_mac: float, succinylcholine_cp: float,
             dantrolene_ce: float) -> None:
        if not self._active and not self._susceptible and self.severity == 0.0:
            self._controlled = False
            return
        if not self._active and self._susceptible and (sevo_mac > 0.05 or succinylcholine_cp > 0.01):
            self._active = True
            self._controlled = False
        if self._active:
            inhibition = hill_function(dantrolene_ce, 1.5, 3.0)
            if (
                inhibition > 0.5
                and self.severity < 0.02
                and sevo_mac <= 0.05
                and succinylcholine_cp <= 0.01
            ):
                self._controlled = True
            # Trigger removal alone does not abort an established crisis, whereas
            # full suppression after trigger elimination resolves without relapse
            # despite trace post-washout adipose sevoflurane.
            sevo_threshold = 0.05 if self._controlled else 0.001
            exposed = sevo_mac > sevo_threshold or succinylcholine_cp > 0.01
            if exposed:
                self._controlled = False
            recovery = inhibition / 45.0
            if self._controlled:
                growth = 0.0
                recovery = max(recovery, 1.0 / 180.0)
            else:
                tau_on = 180.0 if exposed else 600.0
                growth = (1.0 / tau_on) * (1.0 - inhibition)
        else:
            self._controlled = False
            exposed = False
            growth, recovery = 0.0, 1.0 / 90.0
        rate = growth + recovery
        if rate > 0.0:
            self.severity += (growth / rate - self.severity) * -math.expm1(-dt * rate)
        if self._active and self._controlled and not exposed and self.severity < 1e-3:
            self.severity = 0.0
            self._active = False
            self._controlled = False

    @property
    def excess_metabolism(self) -> float:
        return 4.0 * self.severity

    @property
    def extra_heat_w_per_kg(self) -> float:
        """Anaerobic glycolysis and SR calcium-cycling heat beyond aerobic metabolism."""
        return 6.5 * self.severity**2

    @property
    def chest_wall_stiffness(self) -> float:
        """Chest-wall elastance multiplier from generalized skeletal muscle contracture."""
        return 1.0 + 0.90 * self.severity

    @property
    def respiratory_excursion_fraction(self) -> float:
        """Remaining spontaneous tidal excursion against generalized muscle rigidity."""
        return max(0.35, 1.0 - 0.62 * self.severity)

    @property
    def heart_rate_increase(self) -> float:
        return 55.0 * self.severity

    @property
    def rigidity(self) -> bool:
        # Sarcoplasmic calcium release persists despite neuromuscular block.
        return self.severity > 0.15
