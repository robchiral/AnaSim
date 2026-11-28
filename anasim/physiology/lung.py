"""Pressure history, aeration, and the static pressure-volume relation."""

import math
from typing import ClassVar

from anasim.core.utils import clamp01
from anasim.patient.patient import Patient


class LungAeration:
    """Aeration, atelectatic shunt, and alveolar flooding.

    Pressure and timing follow Rothen 1993/1999 and Bates and Irvin 2002.
    BMI shifts, perfusion, and timing away from 40 cmH2O are approximations.
    Most FRC loss reflects diaphragm displacement (Hedenstierna and Edmark 2015).
    After a preoxygenated induction without PEEP, shunt averages about 5%
    (Gunnarsson 1991 4.8%, Tokics 1996 5.0%, Rothen 1995 6.5%). Atelectasis
    persists early after emergence (Benoît 2002) and clears over hours to days
    of awake unintubated breathing (Strandberg 1986; Eichenberger 2002). Lung
    water above 3 mL/kg floods alveoli. PEEP 13 vs 3 cmH2O reduced shunt and
    alveolar flooding by about 70% in canine pulmonary edema (Malo 1984).
    """

    INTERVAL = 0.1  # s; independent of the engine's outer step
    REFERENCE_RECRUITED = 0.9  # Compliance calibration at PEEP 5 under anesthesia
    MIN_RECRUITED = 0.65  # Maximum modeled aeration loss under anesthesia
    ATELECTASIS_SHUNT = 0.2  # Perfusion per unit of atelectatic fraction
    INDUCTION_ATELECTASIS = 0.25  # Left by preoxygenation and apnea at induction
    OPENING_WIDTH = 4.5  # cmH2O; logistic pressure response, Rothen 1993
    OPENING_RATE = 1.0 / 2.6  # /s; CT reopening time constant at 40 cmH2O, Rothen 1999
    # Low-pressure reversible small-airway / poorly-aerated compartment
    # (Pelosi 1999 P_flex,L; Maisch 2008 pre-RM PEEP compliance gain; Reinius 2009
    # poorly-aerated -500..-100 HU shift without clearing dense CT atelectasis;
    # Bates and Irvin 2002 separate static opening > closing pressures).
    SMALL_AIRWAY_OPEN_PRESSURE = 14.5  # cmH2O; inspiratory reopening threshold
    SMALL_AIRWAY_OPEN_WIDTH = 1.1  # cmH2O; logistic slope across ZEEP-to-PEEP5 plateau
    SMALL_AIRWAY_CLOSE_PRESSURE = 5.0  # cmH2O; end-expiratory closure threshold
    SMALL_AIRWAY_OPEN_RATE = 0.30  # /s above opening pressure
    SMALL_AIRWAY_CLOSE_RATE = 0.019  # /s at zero distending pressure
    SMALL_AIRWAY_COMPLIANCE_WEIGHT = 0.21  # Compliance share in normal-BMI adults
    # Gas-dependent closure calibrated to Rothen 1995.
    ABSORPTION_RATE = 0.02  # /s at zero distending pressure without inert gas
    INERT_GAS_PROTECTION = 9.0  # Aggregate gas effect
    FLOOD_ONSET = 3.0  # mL/kg lung water above normal
    FLOOD_SLOPE = 0.05  # Flooded fraction per mL/kg
    FLOOD_MAX = 0.6
    FLOOD_RELIEF_MAX = 0.7
    FLOOD_RELIEF_PRESSURES = (6.0, 16.0)  # Mean distending pressure (PEEP + ~3), cmH2O
    FLOOD_RATE = 1.0 / 60.0  # /s
    # High-FiO2 absorption shunt behind severely obstructed bronchospastic airways
    # (Ballester 1989: 0.8 -> 2.1% in moderate asthma; Rodríguez-Roisin 1989: 1.5 -> 8.3% in status asthmaticus).
    BRONCH_SHUNT_MAX = 0.075
    BRONCH_SHUNT_TAU_ON = 900.0  # s
    BRONCH_SHUNT_TAU_OFF = 300.0  # s
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
        self.surgical_site = patient.surgical_site
        self._obesity_clearance = clamp01((40.0 - patient.bmi) / 12.0)
        self.awake_clearance_rate = self._obesity_clearance / self.AWAKE_CLEARANCE_TAU_S[patient.surgical_site]
        self.persistent_clearance_rate = self._obesity_clearance / self.PERSISTENT_CLEARANCE_TAU_S
        self.reference_volume = patient.functional_residual_capacity()
        self.frc = self.reference_volume * recruited
        self.reference_compliance = patient.respiratory_compliance()
        self.opening_midpoint = 31.0 + 0.35 * (patient.bmi - 24.0)
        # Dependent airway closing pressure for gas absorption rises steeply with BMI
        # (Hedenstierna and Edmark 2015), while bulk compliance recruitment uses a
        # gentler threshold because patient.respiratory_compliance() already embeds
        # Pelosi 1998's supine BMI curve (Behazin 2010).
        self.closing_pressure = 5.0 + 0.30 * max(0.0, patient.bmi - 25.0)
        self.bulk_closing_pressure = 5.0
        bmi_factor = clamp01((patient.bmi - 30.0) / 20.0)
        self.small_airway_open = 1.0
        self.small_airway_open_pressure = self.SMALL_AIRWAY_OPEN_PRESSURE + 0.14 * max(0.0, patient.bmi - 25.0)
        self.small_airway_close_pressure = self.SMALL_AIRWAY_CLOSE_PRESSURE + 0.06 * max(0.0, patient.bmi - 25.0)
        self.small_airway_compliance_weight = self.SMALL_AIRWAY_COMPLIANCE_WEIGHT + 0.08 * bmi_factor
        self.min_recruited = self.MIN_RECRUITED
        self.max_atelectatic = (1.0 - self.MIN_RECRUITED) + 0.22 * bmi_factor
        self.atelectasis_shunt = self.ATELECTASIS_SHUNT + 0.10 * bmi_factor
        self.collapse_deficit = 0.0
        self.bronchospasm = 0.0
        self.bronch_absorption_shunt = 0.0
        self.lung_water = 0.0  # mL/kg predicted body weight above normal
        self.flooded = 0.0
        self.extra_shunt = 0.0  # Optional parenchymal shunt (e.g. severe ARDS), 0-0.95
        self.crisis_shunt = 0.0  # Acute crisis shunt (pneumothorax / embolism), 0-0.80
        self.pleural_collapse = 0.0  # Unilateral lung collapse from pneumothorax, 0-0.50
        self.chest_wall_stiffness = 1.0
        self.pressure_transmission = 1.0
        self.elapsed = 0.0
        self.pressure_area = 0.0

    @property
    def aerated(self) -> float:
        return self.recruited * (1.0 - self.flooded) * (1.0 - clamp01(self.pleural_collapse))

    @property
    def relaxed_volume(self) -> float:
        return self.reference_volume * self.aerated

    @property
    def shunt_fraction(self) -> float:
        # Atelectatic and flooded units retain perfusion; closed-airway O2 absorption
        # collapse adds shunt once gas volume falls past closing capacity (~55% of FRC).
        base_shunt = self.atelectasis_shunt * self.atelectatic + (1.0 - self.atelectatic) * self.flooded
        if self.collapse_deficit > 0.36:
            eff_collapse = clamp01((clamp01(self.collapse_deficit) - 0.36) / 0.54)
            collapse_shunt = 0.55 * (eff_collapse ** 1.5)
            base_shunt = min(0.70, base_shunt + (1.0 - base_shunt) * collapse_shunt)
        else:
            base_shunt = min(0.70, base_shunt)
        if self.bronch_absorption_shunt > 0.0:
            bronch_shunt = clamp01(self.bronch_absorption_shunt)
            base_shunt = min(0.75, base_shunt + (1.0 - base_shunt) * bronch_shunt)
        if self.crisis_shunt > 0.0:
            crisis = min(0.80, max(0.0, self.crisis_shunt))
            base_shunt = min(0.90, base_shunt + (1.0 - base_shunt) * crisis)
        if self.extra_shunt > 0.0:
            extra = min(0.95, max(0.0, self.extra_shunt))
            return min(0.95, base_shunt + (1.0 - base_shunt) * extra)
        return base_shunt

    def advance(self, dt: float, pressure_area: float) -> bool:
        """Accumulate distending pressure; change aeration at fixed physical times."""
        self.elapsed += dt
        self.pressure_area += pressure_area
        if self.elapsed < self.INTERVAL - 1e-12:
            return False
        pressure = self.pressure_area / self.elapsed
        fraction = 1.0 / (1.0 + math.exp((self.opening_midpoint - pressure) / self.OPENING_WIDTH))
        target = self.min_recruited + (1.0 - self.min_recruited) * fraction
        opening = self.OPENING_RATE if target > self.recruited else 0.0
        awake_active = self.spontaneous_breathing and self.glottic_intact
        # Postoperative FRC recovery tracks site- and BMI-dependent atelectasis
        # clearance (Wahba 1975; Putensen-Himmer 1992; Eichenberger 2002).
        spontaneous_cap = max(self.recruited, 1.0 - self.atelectatic)
        if awake_active and self.recruited < spontaneous_cap - 1e-12:
            spont_base_rate = (
                max(self.awake_clearance_rate, 0.002 * self._obesity_clearance)
                if self.surgical_site == "other"
                else self.awake_clearance_rate
            )
            spontaneous = spont_base_rate * (1.0 - self.unconscious)
        else:
            spontaneous = 0.0
        closing = max(0.0, self.bulk_closing_pressure - pressure) / 300.0
        minimum = 1.0 - (1.0 - self.min_recruited) * self.unconscious
        if self.recruited <= minimum:
            closing = 0.0
        rate = opening + spontaneous + closing
        if rate > 0.0:
            equilibrium = (opening * target + spontaneous * spontaneous_cap + closing * minimum) / rate
            self.recruited = equilibrium + (self.recruited - equilibrium) * math.exp(-rate * self.elapsed)
        # Low-pressure small-airway reopening/closure responds to PEEP 0-10 cmH2O
        # without clearing dense CT atelectatic shunt (Maisch 2008; Reinius 2009).
        s_open_frac = 1.0 / (
            1.0 + math.exp((self.small_airway_open_pressure - pressure) / self.SMALL_AIRWAY_OPEN_WIDTH)
        )
        s_open = self.SMALL_AIRWAY_OPEN_RATE * s_open_frac if s_open_frac > 0.02 else 0.0
        s_awake = 0.015 * (1.0 - self.unconscious) if awake_active else 0.0
        if pressure < self.small_airway_close_pressure and self.unconscious > 0.0:
            s_close = (
                self.SMALL_AIRWAY_CLOSE_RATE
                * self.unconscious
                * (self.small_airway_close_pressure - pressure)
                / self.small_airway_close_pressure
            )
        else:
            s_close = 0.0
        s_min = 1.0 - self.unconscious
        if self.small_airway_open <= s_min:
            s_close = 0.0
        s_rate = s_open + s_awake + s_close
        if s_rate > 0.0:
            s_eq = (s_open * 1.0 + s_awake * 1.0 + s_close * s_min) / s_rate
            self.small_airway_open = clamp01(
                s_eq + (self.small_airway_open - s_eq) * math.exp(-s_rate * self.elapsed)
            )
        # Gas composition changes shunt separately from bulk compliance (Rothen 1995).
        closed_target = self.max_atelectatic * (1.0 - fraction)
        s = self.persistent_share
        reopening = self.OPENING_RATE if self.atelectatic > closed_target else 0.0
        reopening_p = self.OPENING_RATE if self.persistent_atelectatic > s * closed_target else 0.0
        awake_reopening = self.awake_clearance_rate * (1.0 - self.unconscious) if awake_active else 0.0
        persistent_rate = self.persistent_clearance_rate * (1.0 - self.unconscious) if awake_active else 0.0
        closed_max = self.max_atelectatic * self.unconscious
        if pressure < self.closing_pressure and closed_max > 0.0:
            raw_absorption = (
                self.ABSORPTION_RATE
                * math.exp(-self.INERT_GAS_PROTECTION * self.inert_fraction)
                * (self.closing_pressure - pressure) / self.closing_pressure
            )
            absorption = raw_absorption if self.atelectatic < closed_max else 0.0
            absorption_p = raw_absorption if self.persistent_atelectatic < s * closed_max else 0.0
        else:
            absorption = absorption_p = 0.0

        p0 = min(self.persistent_atelectatic, self.atelectatic)
        p_rate = reopening_p + persistent_rate + absorption_p
        if p_rate > 0.0:
            p_eq = (reopening_p * s * closed_target + absorption_p * s * closed_max) / p_rate
            dp0 = p0 - p_eq
            exp_pers = math.exp(-p_rate * self.elapsed)
            p_new = p_eq + dp0 * exp_pers
        else:
            p_eq = p0
            dp0 = 0.0
            exp_pers = 1.0
            p_new = p0

        total = reopening + awake_reopening + absorption
        if total > 0.0:
            exp_total = math.exp(-total * self.elapsed)
            eq_const = (reopening * closed_target + absorption * closed_max + awake_reopening * p_eq) / total
            if awake_reopening > 0.0 and dp0 != 0.0:
                if abs(total - p_rate) > 1e-12:
                    pers_term = dp0 * (awake_reopening / (total - p_rate)) * (exp_pers - exp_total)
                else:
                    pers_term = dp0 * awake_reopening * self.elapsed * exp_total
            else:
                pers_term = 0.0
            self.atelectatic = eq_const + (self.atelectatic - eq_const) * exp_total + pers_term
        self.persistent_atelectatic = max(0.0, min(p_new, self.atelectatic))
        if self.lung_water > self.FLOOD_ONSET:
            flood = min(self.FLOOD_MAX, self.FLOOD_SLOPE * (self.lung_water - self.FLOOD_ONSET))
            low, high = self.FLOOD_RELIEF_PRESSURES
            relief = self.FLOOD_RELIEF_MAX * clamp01((pressure - low) / (high - low))
            flood *= 1.0 - relief
        else:
            flood = 0.0
        if flood != self.flooded:
            self.flooded = flood + (self.flooded - flood) * math.exp(-self.FLOOD_RATE * self.elapsed)
        if self.bronchospasm > 0.25:
            sev = clamp01((self.bronchospasm - 0.25) / 0.75)
            bronch_shunt_target = self.BRONCH_SHUNT_MAX * (sev ** 1.8) * math.exp(-5.0 * self.inert_fraction)
        else:
            bronch_shunt_target = 0.0
        if bronch_shunt_target != self.bronch_absorption_shunt:
            tau = self.BRONCH_SHUNT_TAU_ON if bronch_shunt_target > self.bronch_absorption_shunt else self.BRONCH_SHUNT_TAU_OFF
            self.bronch_absorption_shunt = (
                bronch_shunt_target
                + (self.bronch_absorption_shunt - bronch_shunt_target) * math.exp(-self.elapsed / tau)
            )
        self.elapsed = self.pressure_area = 0.0
        return True

    @property
    def static_compliance(self) -> float:
        """Low-inflation compliance at the current aeration, L/cmH2O."""
        non_dense = (1.0 - self.flooded) * (1.0 - clamp01(self.pleural_collapse))
        if non_dense <= 0.0:
            return 0.0
        dense_scale = ((self.recruited / self.REFERENCE_RECRUITED) ** 0.45) * non_dense
        low_p_scale = 1.0 - self.small_airway_compliance_weight * (1.0 - self.small_airway_open)
        return self.reference_compliance * dense_scale * low_p_scale / max(0.2, self.chest_wall_stiffness)

    def static_mechanics(self, volume: float) -> tuple[float, float]:
        """Static recoil and tangent compliance at an absolute gas volume (L)."""
        compliance = self.static_compliance
        inflation = (volume - self.relaxed_volume) / compliance
        excess = max(0.0, inflation - 16.0)
        recoil = inflation + excess ** 3 / 1400.0
        tangent = compliance / (1.0 + 3.0 * excess * excess / 1400.0)
        return recoil, tangent
