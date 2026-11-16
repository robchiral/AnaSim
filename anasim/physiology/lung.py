"""Pressure history, aeration, and the static pressure-volume relation."""

import math
from typing import ClassVar

from anasim.core.utils import clamp01
from anasim.patient.patient import Patient


class LungAeration:
    """Aeration, atelectatic shunt, and alveolar flooding.

    Pressure and timing follow Rothen 1993/1999 and Bates and Irvin 2002.
    BMI shifts, perfusion, and timing away from 40 cmH2O are approximations.
    Most FRC loss reflects diaphragm displacement (Hedenstierna 2015).
    After a preoxygenated induction without PEEP, shunt averages about 5%
    (Gunnarsson 1991 4.8%, Tokics 1996 5.0%, Rothen 1995 6.5%). Atelectasis
    persists early after emergence (Benoit 2002) and clears over hours to days
    of awake unintubated breathing (Strandberg 1986; Eichenberger 2002). Lung
    water above 3 mL/kg floods alveoli. PEEP 13 vs 3 cmH2O reduced flooding by
    about 70% (Malo 1984).
    """

    INTERVAL = 0.1  # s; independent of the engine's outer step
    REFERENCE_RECRUITED = 0.9  # Compliance calibration at PEEP 5 under anesthesia
    MIN_RECRUITED = 0.65  # Maximum modeled aeration loss under anesthesia
    ATELECTASIS_SHUNT = 0.2  # Perfusion per unit of atelectatic fraction
    INDUCTION_ATELECTASIS = 0.25  # Left by preoxygenation and apnea at induction
    OPENING_WIDTH = 4.5  # cmH2O; logistic pressure response, Rothen 1993
    OPENING_RATE = 1.0 / 2.6  # /s; CT reopening time constant at 40 cmH2O, Rothen 1999
    # Gas-dependent closure calibrated to Rothen 1995.
    ABSORPTION_RATE = 0.02  # /s at zero distending pressure without inert gas
    INERT_GAS_PROTECTION = 9.0  # Aggregate gas effect
    FLOOD_ONSET = 3.0  # mL/kg lung water above normal
    FLOOD_SLOPE = 0.05  # Flooded fraction per mL/kg
    FLOOD_MAX = 0.6
    FLOOD_RELIEF_MAX = 0.7
    FLOOD_RELIEF_PRESSURES = (6.0, 16.0)  # Mean distending pressure (PEEP + ~3), cmH2O
    FLOOD_RATE = 1.0 / 60.0  # /s
    # Postoperative awake clearance and persistent basal share by surgical site,
    # calibrated to Strandberg 1986, Lindberg 1992, Putensen-Himmer 1992, and
    # Eichenberger 2002.
    AWAKE_CLEARANCE_TAU_S: ClassVar[dict[str, float]] = {
        "other": 3000.0,
        "lower_abdominal": 6000.0,
        "upper_abdominal": 7200.0,
    }
    PERSISTENT_ATELECTASIS_SHARE: ClassVar[dict[str, float]] = {
        "other": 0.15,
        "lower_abdominal": 0.21,
        "upper_abdominal": 0.265,
    }
    PERSISTENT_CLEARANCE_TAU_S = 259200.0  # s

    def __init__(self, patient: Patient, recruited: float = 1.0, atelectatic: float | None = None):
        self.recruited = recruited
        self.atelectatic = 1.0 - recruited if atelectatic is None else atelectatic
        self.persistent_share = self.PERSISTENT_ATELECTASIS_SHARE[patient.surgical_site]
        self.persistent_atelectatic = self.persistent_share * self.atelectatic
        self.inert_fraction = 0.75
        self.unconscious = 0.0
        self.spontaneous_breathing = False
        self.glottic_intact = True
        obesity_clearance = clamp01((40.0 - patient.bmi) / 12.0)
        self.awake_clearance_rate = obesity_clearance / self.AWAKE_CLEARANCE_TAU_S[patient.surgical_site]
        self.persistent_clearance_rate = obesity_clearance / self.PERSISTENT_CLEARANCE_TAU_S
        self.reference_volume = patient.functional_residual_capacity()
        self.frc = self.reference_volume * recruited
        self.reference_compliance = patient.respiratory_compliance()
        self.opening_midpoint = 31.0 + 0.35 * (patient.bmi - 24.0)
        self.closing_pressure = 5.0
        self.lung_water = 0.0  # mL/kg predicted body weight above normal
        self.flooded = 0.0
        self.chest_wall_stiffness = 1.0
        self.pressure_transmission = 1.0
        self.elapsed = 0.0
        self.pressure_area = 0.0

    @property
    def aerated(self) -> float:
        return self.recruited * (1.0 - self.flooded)

    @property
    def relaxed_volume(self) -> float:
        return self.reference_volume * self.aerated

    @property
    def shunt_fraction(self) -> float:
        # Atelectatic and flooded units retain perfusion; no instantaneous oxygen benefit from PEEP.
        return self.ATELECTASIS_SHUNT * self.atelectatic + (1.0 - self.atelectatic) * self.flooded

    def advance(self, dt: float, pressure_area: float) -> bool:
        """Accumulate distending pressure; change aeration at fixed physical times."""
        self.elapsed += dt
        self.pressure_area += pressure_area * clamp01(self.pressure_transmission)
        if self.elapsed < self.INTERVAL - 1e-12:
            return False
        pressure = self.pressure_area / self.elapsed
        fraction = 1.0 / (1.0 + math.exp((self.opening_midpoint - pressure) / self.OPENING_WIDTH))
        target = self.MIN_RECRUITED + (1.0 - self.MIN_RECRUITED) * fraction
        opening = self.OPENING_RATE if target > self.recruited else 0.0
        spontaneous = 0.02 * (1.0 - self.unconscious) if self.spontaneous_breathing else 0.0
        closing = max(0.0, self.closing_pressure - pressure) / 300.0
        minimum = 1.0 - (1.0 - self.MIN_RECRUITED) * self.unconscious
        if self.recruited <= minimum:
            closing = 0.0
        rate = opening + spontaneous + closing
        if rate > 0.0:
            equilibrium = (opening * target + spontaneous + closing * minimum) / rate
            self.recruited = equilibrium + (self.recruited - equilibrium) * math.exp(-rate * self.elapsed)
        # Gas composition changes shunt separately from bulk compliance (Rothen 1995).
        closed_target = 1.0 - target
        s = self.persistent_share
        reopening = self.OPENING_RATE if self.atelectatic > closed_target else 0.0
        reopening_p = self.OPENING_RATE if self.persistent_atelectatic > s * closed_target else 0.0
        awake_active = self.spontaneous_breathing and self.glottic_intact
        awake_reopening = self.awake_clearance_rate * (1.0 - self.unconscious) if awake_active else 0.0
        persistent_rate = self.persistent_clearance_rate * (1.0 - self.unconscious) if awake_active else 0.0
        closed_max = (1.0 - self.MIN_RECRUITED) * self.unconscious
        raw_absorption = self.ABSORPTION_RATE * math.exp(-self.INERT_GAS_PROTECTION * self.inert_fraction)
        raw_absorption *= max(0.0, self.closing_pressure - pressure) / self.closing_pressure
        absorption = raw_absorption if self.atelectatic < closed_max else 0.0
        absorption_p = raw_absorption if self.persistent_atelectatic < s * closed_max else 0.0

        p0 = min(self.persistent_atelectatic, self.atelectatic)
        p_rate = reopening_p + persistent_rate + absorption_p
        if p_rate > 0.0:
            p_eq = (reopening_p * s * closed_target + absorption_p * s * closed_max) / p_rate
            dp0 = p0 - p_eq
            p_new = p_eq + dp0 * math.exp(-p_rate * self.elapsed)
        else:
            p_eq = p0
            dp0 = 0.0
            p_new = p0

        total = reopening + awake_reopening + absorption
        if total > 0.0:
            exp_total = math.exp(-total * self.elapsed)
            eq_const = (reopening * closed_target + absorption * closed_max + awake_reopening * p_eq) / total
            if awake_reopening > 0.0 and dp0 != 0.0:
                exp_pers = math.exp(-p_rate * self.elapsed)
                if abs(total - p_rate) > 1e-12:
                    pers_term = dp0 * (awake_reopening / (total - p_rate)) * (exp_pers - exp_total)
                else:
                    pers_term = dp0 * awake_reopening * self.elapsed * exp_total
            else:
                pers_term = 0.0
            self.atelectatic = eq_const + (self.atelectatic - eq_const) * exp_total + pers_term
        self.persistent_atelectatic = max(0.0, min(p_new, self.atelectatic))
        flood = min(self.FLOOD_MAX, self.FLOOD_SLOPE * max(0.0, self.lung_water - self.FLOOD_ONSET))
        low, high = self.FLOOD_RELIEF_PRESSURES
        relief = self.FLOOD_RELIEF_MAX * clamp01((pressure - low) / (high - low))
        flood *= 1.0 - relief
        self.flooded = flood + (self.flooded - flood) * math.exp(-self.FLOOD_RATE * self.elapsed)
        self.elapsed = self.pressure_area = 0.0
        return True

    @property
    def static_compliance(self) -> float:
        """Low-inflation compliance at the current aeration, L/cmH2O."""
        return self.reference_compliance * self.aerated / (
            self.REFERENCE_RECRUITED * max(0.2, self.chest_wall_stiffness)
        )

    def static_mechanics(self, volume: float) -> tuple[float, float]:
        """Static recoil and tangent compliance at an absolute gas volume (L)."""
        compliance = self.static_compliance
        inflation = (volume - self.relaxed_volume) / compliance
        excess = max(0.0, inflation - 25.0)
        recoil = inflation + excess ** 3 / 1200.0
        tangent = compliance / (1.0 + excess * excess / 400.0)
        return recoil, tangent
