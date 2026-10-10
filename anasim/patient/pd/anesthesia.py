from __future__ import annotations

import math
from collections import deque

from anasim.patient.patient import Patient

# Sevoflurane potency on the BIS response surface is anchored to BIS ~41 at
# 1 MAC (Kanazawa 2017; Ryu 2018), which gives BIS ~30 at 1.5 MAC (Paraskeva 2005).
SEVO_BIS_AT_1_MAC = 41.0

# Fentanyl counts as remifentanil at 1.37/1.67 of its concentration: a 50%
# isoflurane MAC reduction needs 1.37 ng/mL remifentanil (Lang 1996) and 1.67
# ng/mL fentanyl (McEwan 1993).
FENTANYL_REMI_POTENCY = 1.37 / 1.67

# Midazolam converts to a saturating propofol equivalent (mcg/mL). At its
# loss-of-response Ce (Albrecht 1999: 499 ng/mL at 26 years, 210 at 71) it
# equals the Bouillon Bayesian shaking/shouting Ce50 of 3.21 mcg/mL, and
# half the maximum. AnaSim calibrates the maximum.
PROPOFOL_LOSS_OF_RESPONSE = 6.68 * 0.48
MIDAZOLAM_MAX_PROPOFOL_EQUIVALENT = 2.0 * PROPOFOL_LOSS_OF_RESPONSE
# Short 1992: midazolam with propofol needs 37% less than additive doses for
# hypnosis, an interaction coefficient of 3.7 at equal shares. The propofol
# share in the interaction is capped at loss of response, the endpoint Short
# measured.
MIDAZOLAM_PROPOFOL_SYNERGY = 3.7

# Etomidate (mcg/mL whole blood) converts to propofol at equal
# loss-of-response concentrations: the plasma OAA/S Ce50 is 0.554 mcg/mL
# (Kaneda 2011), about 0.50 in whole blood (Arden 1986 plasma:blood ratio).
# Its respiratory conversion is calibrated separately: apnea after induction lasts
# about 20 s, and CO2-response depression is smaller than with other
# hypnotics (Valk 2021).
ETOMIDATE_LOSS_OF_RESPONSE = 0.50
ETOMIDATE_RESPIRATORY_PROPOFOL_RATIO = 2.0  # Calibrated independently of clinical response.

# Ketamine (mcg/mL): patients woke at 0.64 mcg/mL, and 2.2 mcg/mL with about
# 0.6 MAC nitrous oxide maintained surgical anesthesia (Idvall 1979), so 3.5
# mcg/mL counts as 1 MAC for laryngoscopy tolerance. BIS excludes ketamine.
KETAMINE_LOSS_OF_RESPONSE = 0.64
KETAMINE_MAC_EQUIVALENT = 3.5

# Lidocaine blocks 4.2% of responses to stimulation per mcg/mL, apart from
# other drugs: 3-6 mcg/mL lowered anesthetic requirement 10-28% (Himes 1977).
# It has no hypnotic or BIS effect.
LIDOCAINE_BLOCK_PER_MCG_ML = 0.042


def midazolam_loss_of_response(age: float) -> float:
    """Midazolam Ce (ng/mL) for loss of response, falling 1.9% per year (Albrecht 1999)."""
    return 499.0 * math.exp(-0.0189 * (age - 26.0))


def hypnotic_equivalent(
    ce_prop: float,
    ce_etomidate: float,
    ce_midazolam: float,
    midazolam_c50: float,
    ventilation: bool = False,
) -> float:
    """Propofol-equivalent Ce (mcg/mL) of propofol, etomidate, and midazolam (ng/mL).

    With `ventilation`, etomidate uses its separate respiratory conversion.
    """
    etomidate = max(0.0, ce_etomidate) * PROPOFOL_LOSS_OF_RESPONSE / ETOMIDATE_LOSS_OF_RESPONSE
    if ventilation:
        etomidate = max(0.0, ce_etomidate) * ETOMIDATE_RESPIRATORY_PROPOFOL_RATIO
    ce_prop = max(0.0, ce_prop) + etomidate
    ce_midazolam = max(0.0, ce_midazolam)
    midazolam = MIDAZOLAM_MAX_PROPOFOL_EQUIVALENT * ce_midazolam / (ce_midazolam + midazolam_c50)
    propofol_share = min(ce_prop / PROPOFOL_LOSS_OF_RESPONSE, 1.0)
    return ce_prop + midazolam * (1.0 + MIDAZOLAM_PROPOFOL_SYNERGY * propofol_share)


def opioid_equivalent(remi: float, fentanyl: float) -> float:
    """Remifentanil-equivalent concentration (ng/mL)."""
    return max(0.0, remi) + FENTANYL_REMI_POTENCY * max(0.0, fentanyl)


class BISModel:
    """Eleveld 2018 BIS, with a calibrated additive sevoflurane contribution.

    Opioids affect stimulation-related arousal outside this baseline curve.
    The published age-dependent delay includes monitor processing.
    """

    def __init__(self, patient: Patient):
        self.c50 = 3.079446890954557 * math.exp(-0.00634837 * (patient.age - 35.0))
        self.baseline = 92.9824
        self.gamma_above = 1.4736185873429664
        self.gamma_below = 1.8939053122841638
        self.delay = 15.0 + math.exp(0.0517399 * patient.age)
        sevo_units = (self.baseline / SEVO_BIS_AT_1_MAC - 1.0) ** (1.0 / self.gamma_above)
        self.mac50 = 1.0 / sevo_units
        self.initialize(self.baseline)

    def initialize(self, bis: float) -> None:
        self._clock = 0.0
        self._history = deque([(0.0, bis)])

    def compute_bis(self, ce_prop: float, mac_sevo: float = 0.0) -> float:
        ce_equivalent = max(0.0, ce_prop) + self.c50 * max(0.0, mac_sevo) / self.mac50
        # NONMEM's smooth slope transition is on the concentration scale.
        switch = 1.0 / (1.0 + math.exp(-30.0 * max(-20.0, ce_equivalent - self.c50)))
        gamma = switch * self.gamma_above + (1.0 - switch) * self.gamma_below
        return self.baseline / (1.0 + (ce_equivalent / self.c50)**gamma)

    def step(self, dt: float, ce_prop: float, mac_sevo: float = 0.0) -> float:
        """Return the delayed BIS after dt seconds."""
        self._clock += dt
        history = self._history
        history.append((self._clock, self.compute_bis(ce_prop, mac_sevo)))
        target = self._clock - self.delay
        while len(history) > 1 and history[1][0] <= target:
            history.popleft()
        t0, y0 = history[0]
        if target <= t0 or len(history) == 1:
            return y0
        t1, y1 = history[1]
        return y0 + (y1 - y0) * (target - t0) / (t1 - t0)


class ClinicalResponseModel:
    """Bouillon 2004 Bayesian hierarchy for shaking/shouting and laryngoscopy.

    Propofol and remifentanil inputs represent clinical effect-site concentrations.
    Their transfer to Eleveld PK is an assumption. Sevoflurane potency is anchored
    to Kuizenga 2019 shaking/shouting and Hannivoort 2016 laryngoscopy; the
    combined volatile surface and shared slope are simulator approximations.
    """

    def __init__(self):
        self.c50_prop = 6.68
        self.c50_remi = 1.01
        self.gamma_prop = 6.9
        self.gamma_remi = 0.72
        # The input already includes age-adjusted MAC. Convert the measured
        # thresholds at a chosen reference age to MAC once, rather than
        # applying the patient's age correction twice.
        self.mac_reference = 1.80 * 10.0 ** (-0.00269 * (35.0 - 40.0))

    def _probability(self, hypnotic_units: float, ce_remi: float, intensity: float) -> float:
        opioid_units = max(0.0, ce_remi) / (self.c50_remi * intensity)
        post_opioid = intensity / (1.0 + opioid_units**self.gamma_remi)
        effect = (max(0.0, hypnotic_units) / post_opioid)**self.gamma_prop
        # Increasing nonresponse matches the paper's surfaces. Its printed
        # equation 4 appears to reverse the probability with a complement.
        return effect / (1.0 + effect)

    def loss_of_response(
        self, ce_prop: float, ce_remi: float, mac_sevo: float = 0.0,
        mac_n2o: float = 0.0, ce_ketamine: float = 0.0,
    ) -> float:
        intensity = 0.48
        units = max(0.0, ce_prop) / self.c50_prop
        units += intensity * max(0.0, mac_sevo) * self.mac_reference / 0.90
        n2o_units = intensity * max(0.0, mac_n2o) / 0.61
        units += n2o_units * (0.7 if mac_sevo > 0.0 else 1.0)
        units += intensity * max(0.0, ce_ketamine) / KETAMINE_LOSS_OF_RESPONSE
        return self._probability(units, ce_remi, intensity)

    def tolerance(
        self, ce_prop: float, ce_remi: float, mac_sevo: float = 0.0,
        mac_n2o: float = 0.0, ce_ketamine: float = 0.0, ce_lidocaine: float = 0.0,
    ) -> float:
        intensity = 0.83
        units = max(0.0, ce_prop) / self.c50_prop
        units += intensity * max(0.0, mac_sevo) * self.mac_reference / 2.59
        units += intensity * max(0.0, mac_n2o)
        units += intensity * max(0.0, ce_ketamine) / KETAMINE_MAC_EQUIVALENT
        tolerance = self._probability(units, ce_remi, intensity)
        lidocaine_block = min(1.0, LIDOCAINE_BLOCK_PER_MCG_ML * max(0.0, ce_lidocaine))
        return 1.0 - (1.0 - tolerance) * (1.0 - lidocaine_block)


__all__ = [
    "BISModel", "ClinicalResponseModel", "hypnotic_equivalent",
    "midazolam_loss_of_response", "opioid_equivalent",
]
