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
        self.active = False
        self.severity = 0.0

    def step(self, dt: float, *, sevo_mac: float, succinylcholine_cp: float,
             dantrolene_ce: float) -> None:
        if self.active:
            inhibition = hill_function(dantrolene_ce, 1.5, 3.0)
            # Trigger removal alone does not abort an established crisis.
            exposed = triggering_exposure(sevo_mac, succinylcholine_cp)
            growth = (1.0 / 180.0 if exposed else 1.0 / 600.0) * (1.0 - inhibition)
            recovery = inhibition / 45.0
        else:
            growth, recovery = 0.0, 1.0 / 90.0
        rate = growth + recovery
        self.severity += (growth / rate - self.severity) * -math.expm1(-dt * rate)

    @property
    def excess_metabolism(self) -> float:
        return 4.0 * self.severity

    @property
    def heart_rate_increase(self) -> float:
        return 55.0 * self.severity

    @property
    def rigidity(self) -> bool:
        # Sarcoplasmic calcium release persists despite neuromuscular block.
        return self.severity > 0.15
