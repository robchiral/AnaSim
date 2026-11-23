from __future__ import annotations

import math
from collections import deque

from anasim.core.constants import CONCENTRATION_RATIO_SATURATION
from anasim.core.utils import clamp01, hill_function
from anasim.patient.patient import Patient
from anasim.patient.variability import TYPICAL, OpioidRateVariability

# Sevoflurane potency on the BIS response surface is anchored to BIS ~41 at
# 1 MAC (Kanazawa 2017; Ryu 2018), which gives BIS ~30 at 1.5 MAC (Paraskeva 2005).
SEVO_BIS_AT_1_MAC = 41.0
OPIOID_NARCOSIS_THRESHOLD = 4.0  # ng/mL remifentanil-equivalent above analgesic range

# Fentanyl counts as remifentanil at 1.37/1.67 of its concentration: a 50%
# isoflurane MAC reduction needs 1.37 ng/mL remifentanil (Lang 1996) and 1.67
# ng/mL fentanyl (McEwan 1993).
FENTANYL_REMI_POTENCY = 1.37 / 1.67

# Fentanyl attenuation of the pressure response to stimulation, fitted to
# Kazama 1997 Tables 2-3 (measured fentanyl concentrations, absolute intubation
# SBP rises). The effect scales with propofol-equivalent depth; the C50 applies
# at propofol 3.4 mcg/mL.
FENTANYL_PRESSURE_C50 = 6.7
FENTANYL_PRESSURE_GAMMA = 1.9

# Fentanyl attenuation of the HR response, fitted to Kazama 1997 Table 3
# intubation HR rises relative to the no-fentanyl rises at the motor Cp50 and
# Cp95. A propofol interaction did not improve the fit.
FENTANYL_HR_C50 = 1.15  # ng/mL
FENTANYL_HR_GAMMA = 1.32

# Fentanyl lowers the sevoflurane concentration that blocks HR and MAP rises
# to incision (MAC-BAR) by 50, 57, 83, and 92% at 0.78, 1, 3, and 6 ng/mL
# (Katoh 1999), which 1 / (1 + (C / 0.78)**1.14) reproduces.
FENTANYL_MAC_BAR_C50 = 0.78  # ng/mL
FENTANYL_MAC_BAR_GAMMA = 1.14

# Eleveld-to-Bouillon propofol scale for loss of response, calibrated to BIS at
# loss of response during stepped induction (Lysakowski 2009).
PROPOFOL_LOC_TRANSFER = 1.3

# Midazolam converts to a saturating propofol equivalent (mcg/mL). At its
# loss-of-response Ce (Albrecht 1999: 499 ng/mL at 26 years, 210 at 71) it
# equals the Bouillon Bayesian shaking/shouting Ce50 of 3.21 mcg/mL, and
# half the maximum. AnaSim calibrates the maximum.
PROPOFOL_LOSS_OF_RESPONSE = 6.68 * 0.48
MIDAZOLAM_MAX_PROPOFOL_EQUIVALENT = 2.0 * PROPOFOL_LOSS_OF_RESPONSE
MIDAZOLAM_BIS_MAX_PROPOFOL_EQUIVALENT = 0.85 * PROPOFOL_LOSS_OF_RESPONSE
# Marty 1986: 0.3 mg/kg IV midazolam with low-flow nasal O2 raised PaCO2 from
# 38 to 45-46 mmHg while preserving spontaneous breathing; the ceiling is a calibration.
MIDAZOLAM_RESPIRATORY_MAX_PROPOFOL_EQUIVALENT = 1.25
# Short 1992: midazolam with propofol needs 37% less than additive doses for
# hypnosis (u_p = u_m = 0.315 at the 50% isobole). Because midazolam converts
# through a saturating Hill-1 curve (2 u_m / (1 + u_m) = 0.479 at u_m = 0.315),
# an interaction coefficient of 1.364 hits P(LOC) = 0.50 at equal shares. The
# propofol share is capped at loss of response, the endpoint Short measured.
MIDAZOLAM_PROPOFOL_SYNERGY = 1.364

# Etomidate (mcg/mL whole blood) converts to propofol at equal
# loss-of-response concentrations: the plasma OAA/S Ce50 is 0.554 mcg/mL
# (Kaneda 2011), about 0.50 in whole blood (Arden 1986 plasma:blood ratio).
# Its respiratory conversion is calibrated separately: apnea after induction lasts
# about 20 s, and CO2-response depression is smaller than with other
# hypnotics (Valk 2021).
ETOMIDATE_LOSS_OF_RESPONSE = 0.50
ETOMIDATE_RESPIRATORY_PROPOFOL_RATIO = 2.0  # Calibrated independently of clinical response.

# Ketamine (mcg/mL; Idvall 1979 reports umol/L): patients woke at 0.64 mcg/mL,
# and 2.2 mcg/mL with 65% nitrous oxide (about 0.65 MAC) gave adequate surgical
# conditions under neuromuscular block. Taking that combination as about 1.3
# MAC makes 3.5 mcg/mL 1 MAC; the study measured no MAC endpoint. Ketamine
# raises BIS paradoxically during GABAergic anesthesia via high-frequency EEG.
KETAMINE_LOSS_OF_RESPONSE = 0.64
KETAMINE_MAC_EQUIVALENT = 3.5

# Bouillon 2004 pre-opioid intensity of laryngoscopy for movement.
LARYNGOSCOPY_INTENSITY = 0.83

# Lidocaine blunts stimulation without a baseline hypnotic or BIS effect
# (Hans 2010; Altermatt 2012). The 4.2% attenuation per mcg/mL is calibrated
# from Himes 1977 and intubation-response data.
LIDOCAINE_BLOCK_PER_MCG_ML = 0.042

# Nitrous oxide is less than additive with sevoflurane for response to command.
# Katoh 1997's logistic fit (Table 1) gives the 50% isobole s + n - k s n = 1
# in MAC-awake fractions, with k = b12 x (N2O MAC-awake) / b_sevo. Fractions in
# the product are capped near the studied range so each agent always adds effect.
N2O_SEVO_AWAKE_INTERACTION = 0.135 * (11.06 / 0.1677) / 16.653


def hypothermia_narcosis_fraction(temp_c: float) -> float:
    """Loss of consciousness (0-1) from direct hypothermic cerebral depression below 33 °C."""
    if temp_c >= 33.0:
        return 0.0
    return clamp01((33.0 - temp_c) / 5.5) ** 1.6


def midazolam_loss_of_response(age: float) -> float:
    """Midazolam Ce (ng/mL) for loss of response, log-linear in age (Albrecht 1999)."""
    return 499.0 * (210.0 / 499.0) ** ((age - 26.0) / 45.0)


def hypnotic_equivalent(
    ce_prop: float,
    ce_etomidate: float,
    ce_midazolam: float,
    midazolam_c50: float,
    age_factor: float = 1.0,
    ventilation: bool = False,
    bis: bool = False,
) -> float:
    """Propofol-equivalent Ce (mcg/mL) of propofol, etomidate, and midazolam (ng/mL).

    Etomidate and midazolam convert at the patient's propofol potency for the
    endpoint (`age_factor`), so they keep only their own age effects (Arden
    1986; Albrecht 1999). With `ventilation`, etomidate uses its separate
    respiratory conversion. With `bis`, midazolam reflects its EEG beta ceiling.
    """
    loss_of_response = PROPOFOL_LOSS_OF_RESPONSE * age_factor
    etomidate = max(0.0, ce_etomidate) * loss_of_response / ETOMIDATE_LOSS_OF_RESPONSE
    if ventilation:
        etomidate = max(0.0, ce_etomidate) * ETOMIDATE_RESPIRATORY_PROPOFOL_RATIO
    ce_prop = max(0.0, ce_prop) + etomidate
    ce_midazolam = max(0.0, ce_midazolam)
    if bis:
        midaz_max = MIDAZOLAM_BIS_MAX_PROPOFOL_EQUIVALENT
    elif ventilation:
        midaz_max = MIDAZOLAM_RESPIRATORY_MAX_PROPOFOL_EQUIVALENT
    else:
        midaz_max = MIDAZOLAM_MAX_PROPOFOL_EQUIVALENT
    midazolam = midaz_max * age_factor * ce_midazolam / (ce_midazolam + midazolam_c50)
    propofol_share = min(ce_prop / loss_of_response, 1.0)
    return ce_prop + midazolam * (1.0 + MIDAZOLAM_PROPOFOL_SYNERGY * propofol_share)


def opioid_equivalent(remi: float, fentanyl: float, morphine: float = 0.0, hydromorphone: float = 0.0) -> float:
    """Remifentanil equivalent (ng/mL).

    Morphine and hydromorphone add their Olofsen 2026 Table 1 limit-temperature
    shifts at a calibrated 1.01 ng/mL per degree C, plus supratherapeutic
    overdose narcosis scaling above normal analgesic peaks (>45 ng/mL morphine,
    >12 ng/mL hydromorphone).
    """
    m = max(0.0, morphine)
    h = max(0.0, hydromorphone)
    morphine_shift = (m / (14.9 / 0.316)) ** (0.83 / 0.863)
    hydromorphone_shift = (h / (14.9 * 0.316)) ** (0.83 * 0.863)
    overdose_bonus = 0.045 * max(0.0, m - 45.0) + 0.20 * max(0.0, h - 12.0)
    return (max(0.0, remi) + FENTANYL_REMI_POTENCY * max(0.0, fentanyl)
            + 1.01 * (morphine_shift + hydromorphone_shift) + overdose_bonus)


def opioid_controller_fraction(morphine: float, hydromorphone: float) -> float:
    """Share of ventilatory controller output left (Olofsen 2026 supplement Table 4).

    Morphine and hydromorphone potency units are assumed to add.
    """
    morphine_units = (max(0.0, morphine) / (5.5 / 0.616)) ** (0.72 / 1.04)
    hydromorphone_units = (max(0.0, hydromorphone) / (5.5 * 0.616)) ** (0.72 * 1.04)
    return 1.0 / (1.0 + morphine_units + hydromorphone_units)


def opioid_rate_fraction(
    morphine: float, hydromorphone: float, *, variability: OpioidRateVariability = TYPICAL.opioid_rate,
) -> float:
    """Fraction of baseline RR (Olofsen 2026 Eq. 8, supplement Table 3).

    C25 reduces rate by 25%; the two drugs' potency units are assumed to add.
    """
    c25, gamma = 7.7 * variability.c25, 0.43 * variability.gamma
    morphine_units = (max(0.0, morphine) / (c25 / 0.518)) ** (gamma / 1.08)
    hydromorphone_units = (max(0.0, hydromorphone) / (c25 * 0.518)) ** (gamma * 1.08)
    return 1.0 / (1.0 + (morphine_units + hydromorphone_units) / 3.0)


class BISModel:
    """Eleveld 2018 BIS with calibrated sevoflurane, ketamine, high-dose opioid,
    and hypothermia adjustments.

    The published age-dependent delay includes monitor processing, so cortical
    failure from ischemia shows after the same delay.
    """

    def __init__(self, patient: Patient):
        self.age_factor = math.exp(-0.00634837 * (patient.age - 35.0))
        self.c50_typical = 3.079446890954557 * self.age_factor
        self.c50 = self.c50_typical * patient.variability.propofol.ce50
        self.baseline = 92.9824
        self.gamma_above = 1.4736185873429664
        self.gamma_below = 1.8939053122841638
        self.delay = 15.0 + math.exp(0.0517399 * patient.age)
        self.monitor_delay = min(self.delay, 18.0)
        self.ischemia_delay = min(self.delay, 20.0)
        sevo_units = (self.baseline / SEVO_BIS_AT_1_MAC - 1.0) ** (1.0 / self.gamma_above)
        self.mac50 = 1.0 / sevo_units
        self.initialize(self.baseline)

    def initialize(
        self,
        bis: float,
        cortical_function: float = 1.0,
        *,
        ce_prop: float = 0.0,
        mac_sevo: float = 0.0,
        ce_ketamine: float = 0.0,
        ce_opioid: float = 0.0,
        temp_c: float = 37.0,
    ) -> None:
        self._clock = 0.0
        self._prop_history: deque[tuple[float, float]] = deque([(0.0, max(0.0, ce_prop))])
        self._sevo_history: deque[tuple[float, float]] = deque([(0.0, max(0.0, mac_sevo))])
        self._ket_history: deque[tuple[float, float]] = deque([(0.0, max(0.0, ce_ketamine))])
        self._opioid_history: deque[tuple[float, float]] = deque([(0.0, max(0.0, ce_opioid))])
        self._temp_history: deque[tuple[float, float]] = deque([(0.0, float(temp_c))])
        self._arousal_history: deque[tuple[float, float]] = deque([(0.0, 0.0)])
        init_base = self.compute_bis(ce_prop, mac_sevo, ce_ketamine, ce_opioid, temp_c=temp_c)
        self._init_offset_history: deque[tuple[float, float]] = deque([(0.0, float(bis) - init_base)])
        cortical = clamp01(cortical_function)
        self._cortical_history: deque[tuple[float, float]] = deque([(0.0, cortical)])
        self.cortical_function = cortical

    def compute_bis(
        self,
        ce_prop: float,
        mac_sevo: float = 0.0,
        ce_ketamine: float = 0.0,
        ce_opioid: float = 0.0,
        temp_c: float = 37.0,
    ) -> float:
        ce_equivalent = max(0.0, ce_prop) + self.c50_typical * max(0.0, mac_sevo) / self.mac50
        # NONMEM's smooth slope transition is on the concentration scale.
        switch = 1.0 / (1.0 + math.exp(-30.0 * max(-20.0, min(20.0, ce_equivalent - self.c50))))
        gamma = switch * self.gamma_above + (1.0 - switch) * self.gamma_below
        ratio = ce_equivalent / self.c50 if self.c50 > 0.0 else 0.0
        drug_bis = (
            0.0
            if ratio > CONCENTRATION_RATIO_SATURATION * 1e6
            else self.baseline / (1.0 + ratio**gamma)
        )
        # Pure high-dose opioid EEG slowing above normal analgesic exposure.
        opioid_excess = max(0.0, ce_opioid - OPIOID_NARCOSIS_THRESHOLD)
        if opioid_excess > 0.0:
            drug_bis *= 1.0 - 0.38 * hill_function(opioid_excess, 8.0, 2.0)
        if ce_ketamine > 0.0:
            ket_effect = hill_function(ce_ketamine, 0.4, 1.5)
            ket_shift = (
                12.0
                * ket_effect
                * clamp01(drug_bis / 25.0)
                * clamp01((self.baseline - drug_bis) / 25.0)
            )
            drug_bis = min(self.baseline, drug_bis + ket_shift)
        if temp_c < 34.0:
            # Hypothermia progressively slows cortical EEG activity below 34 °C.
            drug_bis *= 1.0 - 0.55 * (clamp01((34.0 - temp_c) / 9.0) ** 1.3)
        return drug_bis

    @staticmethod
    def _advance_delay_queue(
        history: deque[tuple[float, float]], clock: float, value: float, delay: float,
    ) -> float:
        history.append((clock, value))
        target = clock - delay
        while len(history) > 1 and history[1][0] <= target:
            history.popleft()
        if target <= history[0][0] or len(history) == 1:
            return history[0][1]
        if target >= history[-1][0]:
            return history[-1][1]
        t0, y0 = history[0]
        t1, y1 = history[1]
        weight = (target - t0) / (t1 - t0) if t1 > t0 else 0.0
        return y0 + (y1 - y0) * weight

    def step(
        self, dt: float, ce_prop: float, mac_sevo: float = 0.0, cortical_function: float = 1.0,
        ce_ketamine: float = 0.0, ce_opioid: float = 0.0, temp_c: float = 37.0,
        disturbance_bis: float = 0.0,
    ) -> float:
        """Return the delayed BIS after dt seconds; cortical_function (0-1) scales it.

        Propofol uses Eleveld's age-dependent delay; volatile, adjunct, temperature,
        and noxious-arousal effects use the age-independent monitor processing delay
        (Pilge 2006). The delayed cortical function is kept in self.cortical_function.
        """
        self._clock += dt
        d_prop = self._advance_delay_queue(self._prop_history, self._clock, max(0.0, ce_prop), self.delay)
        d_sevo = self._advance_delay_queue(self._sevo_history, self._clock, max(0.0, mac_sevo), self.monitor_delay)
        d_ket = self._advance_delay_queue(self._ket_history, self._clock, max(0.0, ce_ketamine), self.monitor_delay)
        d_opioid = self._advance_delay_queue(self._opioid_history, self._clock, max(0.0, ce_opioid), self.monitor_delay)
        d_temp = self._advance_delay_queue(self._temp_history, self._clock, float(temp_c), self.monitor_delay)
        d_arousal = self._advance_delay_queue(self._arousal_history, self._clock, float(disturbance_bis), self.monitor_delay)
        d_init_offset = self._advance_delay_queue(self._init_offset_history, self._clock, 0.0, self.monitor_delay)
        bis = self.compute_bis(d_prop, d_sevo, d_ket, d_opioid, temp_c=d_temp) + d_init_offset
        self.cortical_function = self._advance_delay_queue(
            self._cortical_history, self._clock, cortical_function, self.ischemia_delay,
        )
        return max(0.0, min(100.0, (bis + d_arousal) * self.cortical_function))


class ClinicalResponseModel:
    """Bouillon 2004 Bayesian hierarchy for shaking/shouting and laryngoscopy.

    Propofol and remifentanil inputs represent clinical effect-site concentrations.
    The engine calibrates Eleveld propofol's LOC input to Lysakowski 2009;
    the other PK transfers remain assumptions. Sevoflurane potency is
    anchored to Kuizenga 2019 shaking/shouting and Hannivoort 2016
    laryngoscopy; the combined volatile surface and shared slope are simulator
    approximations.
    """

    def __init__(self, age: float, sevoflurane_mac: float = 1.0):
        # Heyse 2012 found one sevoflurane C50 shift for all clinical endpoints.
        self.sevoflurane_mac = sevoflurane_mac
        # Schnider 1999 LOC Ce50 = 2.9 - 0.022 x age mcg/mL, scaled relative
        # to Bouillon's median age of 33.5. Noxious tolerance uses Eleveld's
        # gentler hypnotic age slope so elderly TIVA setup does not raise opioid targets.
        self.age_factor = (2.9 - 0.022 * age) / (2.9 - 0.022 * 33.5)
        self.noxious_age_factor = math.exp(-0.00634837 * (age - 33.5))
        self.c50_prop = 6.68
        self.c50_remi = 1.01
        self.gamma_prop = 6.9
        self.gamma_remi = 0.72
        # Convert thresholds to MAC at reference age 35; inputs already include
        # the patient's age adjustment.
        self.mac_reference = 1.80 * 10.0 ** (-0.00269 * (35.0 - 40.0))

    def _effect(
        self, hypnotic_units: float, ce_remi: float, intensity: float, ce_ketamine: float = 0.0,
    ) -> float:
        opioid_units = max(0.0, ce_remi) / (self.c50_remi * intensity)
        ketamine_analgesia = 0.25 * (max(0.0, ce_ketamine) / 0.65) ** 1.1
        post_opioid = intensity / ((1.0 + opioid_units**self.gamma_remi) * (1.0 + ketamine_analgesia))
        return (max(0.0, hypnotic_units) / post_opioid)**self.gamma_prop

    def _probability(
        self, hypnotic_units: float, ce_remi: float, intensity: float, ce_ketamine: float = 0.0,
    ) -> float:
        effect = self._effect(hypnotic_units, ce_remi, intensity, ce_ketamine=ce_ketamine)
        # Increasing nonresponse matches the paper's surfaces. Its printed
        # equation 4 appears to reverse the probability with a complement.
        return effect / (1.0 + effect)

    def loss_of_response(
        self, ce_prop: float, ce_remi: float, mac_sevo: float = 0.0,
        mac_n2o: float = 0.0, ce_ketamine: float = 0.0,
    ) -> float:
        """Shaking/shouting nonresponse probability with endpoint-specific adjunct potency."""
        intensity = 0.48
        units = max(0.0, ce_prop) / (self.c50_prop * self.age_factor)
        sevo_awake = max(0.0, mac_sevo) / self.sevoflurane_mac * self.mac_reference / 0.90
        n2o_awake = max(0.0, mac_n2o) / 0.61
        interaction = N2O_SEVO_AWAKE_INTERACTION * min(sevo_awake, 1.0) * min(n2o_awake, 1.0)
        units += intensity * (sevo_awake + n2o_awake - interaction)
        units += intensity * max(0.0, ce_ketamine) / KETAMINE_LOSS_OF_RESPONSE
        p_base = self._probability(units, ce_remi, intensity)
        # High-dose opioid narcosis above analgesic range (EC50 ~10 ng/mL remi-equiv).
        opioid_excess = max(0.0, ce_remi - OPIOID_NARCOSIS_THRESHOLD)
        p_opioid = hill_function(opioid_excess, 6.0, 2.2) if opioid_excess > 0.0 else 0.0
        return 1.0 - (1.0 - p_base) * (1.0 - p_opioid)

    def _noxious_units(self, ce_prop: float, mac_sevo: float, mac_n2o: float,
                       ce_ketamine: float) -> float:
        # MAC fractions add at the laryngoscopy endpoint.
        mac = (max(0.0, mac_sevo) / self.sevoflurane_mac + max(0.0, mac_n2o)
               + max(0.0, ce_ketamine) / KETAMINE_MAC_EQUIVALENT)
        return (
            max(0.0, ce_prop) / (self.c50_prop * self.noxious_age_factor)
            + LARYNGOSCOPY_INTENSITY * mac * self.mac_reference / 2.59
        )

    def tolerance(
        self, ce_prop: float, ce_remi: float, mac_sevo: float = 0.0,
        mac_n2o: float = 0.0, ce_ketamine: float = 0.0, ce_lidocaine: float = 0.0,
        intensity: float = LARYNGOSCOPY_INTENSITY,
    ) -> float:
        """Probability of no response to a stimulus; laryngoscopy by default."""
        units = self._noxious_units(ce_prop, mac_sevo, mac_n2o, ce_ketamine)
        lidocaine_sparing = min(0.30, 0.030 * max(0.0, ce_lidocaine))
        tolerance = self._probability(
            units, ce_remi, intensity * (1.0 - lidocaine_sparing), ce_ketamine=ce_ketamine,
        )
        lidocaine_block = min(0.25, LIDOCAINE_BLOCK_PER_MCG_ML * max(0.0, ce_lidocaine))
        return 1.0 - (1.0 - tolerance) * (1.0 - lidocaine_block)

    def cardiovascular_response_fractions(
        self, ce_prop: float, ce_remi: float, mac_sevo: float = 0.0,
        mac_n2o: float = 0.0, ce_ketamine: float = 0.0, ce_lidocaine: float = 0.0,
        *, intensity: float, ce_fentanyl: float = 0.0,
    ) -> tuple[float, float]:
        """Pressure and HR fractions of the sympathetic stimulus left after drugs.

        Kazama 1997: propofol alone lowers baseline BP without blunting the
        stimulus-related rise, so IV hypnotics only potentiate the other drugs.
        ce_remi excludes fentanyl, which has separate fits with propofol and
        lowers the inhaled agents' requirement; other opioids use the Bouillon
        surface.
        """
        ce_fentanyl = max(0.0, ce_fentanyl)
        ce_remi = max(0.0, ce_remi)
        inhaled_mac = max(0.0, mac_sevo) / self.sevoflurane_mac + max(0.0, mac_n2o)
        prop_units = max(0.0, ce_prop) / self.c50_prop
        ket_units = LARYNGOSCOPY_INTENSITY * max(0.0, ce_ketamine) / KETAMINE_MAC_EQUIVALENT * self.mac_reference / 2.59
        effective_fent_mac_bar = ce_fentanyl + (ce_remi / FENTANYL_REMI_POTENCY if inhaled_mac > 0.0 else 0.0)
        inhaled_shift = 1.0 + (effective_fent_mac_bar / FENTANYL_MAC_BAR_C50)**FENTANYL_MAC_BAR_GAMMA
        opioid_boost = 1.0 + (ce_remi / (self.c50_remi * intensity)) ** self.gamma_remi
        inhaled_units = self._noxious_units(0.0, mac_sevo * inhaled_shift, mac_n2o * inhaled_shift, 0.0)
        ket_analgesia = 1.0 + 0.25 * (max(0.0, ce_ketamine) / 0.65) ** 1.1
        additive_scaled = (ket_units * opioid_boost + inhaled_units) * ket_analgesia / intensity
        slope_boost = opioid_boost * ket_analgesia
        prop_scaled = prop_units / intensity
        if additive_scaled > 0.0 and prop_scaled > 0.0:
            gm1 = self.gamma_prop - 1.0

            def _ratio_slope(x: float) -> float:
                xg1 = x**gm1
                ux = slope_boost * x + additive_scaled
                ug1 = ux**gm1
                return xg1 / (1.0 + xg1 * x) - slope_boost * ug1 / (1.0 + ug1 * ux)

            if _ratio_slope(prop_scaled) > 0.0:
                lo, hi = 0.0, prop_scaled
                for _ in range(28):
                    mid = 0.5 * (lo + hi)
                    if _ratio_slope(mid) < 0.0:
                        lo = mid
                    else:
                        hi = mid
                prop_scaled = 0.5 * (lo + hi)
                prop_units = prop_scaled * intensity
        units = (prop_units + ket_units) * opioid_boost + inhaled_units
        effect = self._effect(units, 0.0, intensity, ce_ketamine=ce_ketamine)
        baseline = self._effect(prop_units, 0.0, intensity)
        lidocaine_block = min(1.0, LIDOCAINE_BLOCK_PER_MCG_ML * max(0.0, ce_lidocaine))
        # Ratio of survival probabilities, evaluated without subtracting near-one values.
        remaining = (1.0 + baseline) / (1.0 + effect) * (1.0 - lidocaine_block)
        pressure_exposure = (ce_fentanyl / FENTANYL_PRESSURE_C50)**FENTANYL_PRESSURE_GAMMA
        # Fentanyl blunts the laryngoscopy pressor response directly and alongside
        # propofol or ketamine; inhaled agents already incorporate fentanyl via MAC_BAR.
        direct_fent_floor = 0.65 * (1.0 - clamp01(inhaled_mac / 0.5))
        fent_pressure_scale = max(
            direct_fent_floor,
            max(0.0, ce_prop) / 3.4 + 1.2 * max(0.0, ce_ketamine) / KETAMINE_MAC_EQUIVALENT,
        )
        pressure = remaining * math.exp(-math.log(2.0) * pressure_exposure * fent_pressure_scale)
        inhaled_hr_weight = clamp01(inhaled_mac / 0.25) if inhaled_mac > 0.0 else 0.0
        hr_fent = ce_fentanyl + inhaled_hr_weight * (ce_remi / FENTANYL_REMI_POTENCY)
        hr = remaining / (1.0 + (hr_fent / FENTANYL_HR_C50)**FENTANYL_HR_GAMMA)
        return pressure, hr


__all__ = [
    "LARYNGOSCOPY_INTENSITY", "BISModel", "ClinicalResponseModel", "hypnotic_equivalent",
    "hypothermia_narcosis_fraction", "midazolam_loss_of_response", "opioid_equivalent",
]
