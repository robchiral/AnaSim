"""Pressure history, aeration, and the static pressure-volume relation."""

import math

from anasim.patient.patient import Patient


class LungAeration:
    """Aeration, atelectatic shunt, and alveolar flooding.

    Pressure and timing follow Rothen 1993/1999 and Bates and Irvin 2002.
    BMI shifts, perfusion, and timing away from 40 cmH2O are approximations.
    Most FRC loss reflects diaphragm displacement (Hedenstierna 2015);
    atelectatic units shunt about 1-12% of output (Tokics 1996).
    Lung water above 3 mL/kg floods alveoli. PEEP 13 vs 3 cmH2O reduced
    flooding by about 70% (Malo 1984).
    """

    INTERVAL = 0.1  # s; independent of the engine's outer step
    REFERENCE_RECRUITED = 0.9  # Compliance calibration at PEEP 5 under anesthesia
    MIN_RECRUITED = 0.65  # Maximum modeled aeration loss under anesthesia
    ATELECTASIS_SHUNT = 0.3  # Perfusion per unit of atelectatic fraction
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

    def __init__(self, patient: Patient, recruited: float = 1.0):
        self.recruited = recruited
        self.atelectatic = 1.0 - recruited
        self.inert_fraction = 0.75
        self.absorbed_volume = 0.0  # L of gas absorbed behind a closed airway
        self.unconscious = 0.0
        self.spontaneous_breathing = False
        self.reference_volume = patient.functional_residual_capacity()
        self.frc = self.reference_volume * recruited
        self.reference_compliance = patient.respiratory_compliance()
        self.opening_midpoint = 31.0 + 0.35 * (patient.bmi - 24.0)
        self.closing_pressure = 5.0
        self.lung_water = 0.0  # mL/kg predicted body weight above normal
        self.flooded = 0.0
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
        self.pressure_area += pressure_area
        if self.elapsed < self.INTERVAL - 1e-12:
            return False
        # Absorbed gas shrinks the lung below its mechanical volume and lowers alveolar pressure.
        pressure = self.pressure_area / self.elapsed - self.absorbed_volume / self.static_compliance
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
        reopening = self.OPENING_RATE if self.atelectatic > closed_target else 0.0
        absorption = self.ABSORPTION_RATE * math.exp(-self.INERT_GAS_PROTECTION * self.inert_fraction)
        absorption *= max(0.0, self.closing_pressure - pressure) / self.closing_pressure
        closed_max = (1.0 - self.MIN_RECRUITED) * self.unconscious
        total = reopening + spontaneous + absorption
        if total > 0.0:
            equilibrium = (reopening * closed_target + absorption * closed_max) / total
            self.atelectatic = equilibrium + (self.atelectatic - equilibrium) * math.exp(-total * self.elapsed)
        flood = min(self.FLOOD_MAX, self.FLOOD_SLOPE * max(0.0, self.lung_water - self.FLOOD_ONSET))
        low, high = self.FLOOD_RELIEF_PRESSURES
        relief = self.FLOOD_RELIEF_MAX * min(1.0, max(0.0, (pressure - low) / (high - low)))
        flood *= 1.0 - relief
        self.flooded = flood + (self.flooded - flood) * math.exp(-self.FLOOD_RATE * self.elapsed)
        self.elapsed = self.pressure_area = 0.0
        return True

    @property
    def static_compliance(self) -> float:
        """Low-inflation compliance at the current aeration, L/cmH2O."""
        return self.reference_compliance * self.aerated / self.REFERENCE_RECRUITED

    def static_mechanics(self, volume: float) -> tuple[float, float]:
        """Static recoil and tangent compliance at an absolute gas volume (L)."""
        compliance = self.static_compliance
        inflation = (volume - self.relaxed_volume) / compliance
        excess = max(0.0, inflation - 25.0)
        recoil = inflation + excess ** 3 / 1200.0
        tangent = compliance / (1.0 + excess * excess / 400.0)
        return recoil, tangent
