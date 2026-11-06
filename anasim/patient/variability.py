"""Interindividual variability as multipliers on typical model parameters."""

import math
from dataclasses import dataclass

import numpy as np

# Eleveld 2018 control-stream OMEGA, log-scale variances. No correlations were
# estimated. The published tables print the square roots as variances.
ELEVELD_PROPOFOL_OMEGA = {
    "v1": 0.372531,
    "v2": 0.319203,
    "v3": 0.356362,
    "cl": 0.0702216,
    "q2": 0.11998,
    "q3": 0.0436006,
    "ce50": 0.0584127,
    "ke0": 0.492493,
}
# Eleveld pooled 30 studies of children and adults with arterial and venous
# sampling. Half its SDs match the spread of BIS onset and depth in VitalDB
# adult propofol-remifentanil inductions with recorded pump timing.
ELEVELD_SD_SCALE = 0.5
# Eleveld 2017 Tables 3 and 5, log-scale variances. The spectral-edge C50 is not
# used.
ELEVELD_REMIFENTANIL_OMEGA = {
    "v1": 0.104,
    "v2": 0.115,
    "v3": 0.810,
    "cl": 0.0197,
    "q2": 0.0547,
    "q3": 0.285,
    "ke0": 0.379,
}
# Bae 2020 Table 2 CVs. The paper does not define CV; this assumes
# CV = sqrt(exp(variance) - 1). ke0 is from another study and is not sampled.
BAE_FENTANYL_OMEGA = {
    name: math.log1p(cv**2)
    for name, cv in {"v1": 0.675, "v2": 0.716, "v3": 0.367, "cl": 0.308, "q2": 0.672, "q3": 0.494}.items()
}
# Heyse 2012 Tables 2-3: the sevoflurane C50 has a log-SD of 0.20 and is the same
# for shaking and shouting and for noxious stimuli. Heyse's slope of 7.41 is
# close to AnaSim's 6.9. With this spread, the population slope for laryngoscopy
# is 5.2, as in Hannivoort 2016's pooled fit.
HEYSE_SEVOFLURANE_MAC_OMEGA = 0.20**2
# Su 2023 OMEGA: log-normal on the propofol resistance C50, additive on the
# remifentanil resistance Emax of -1. The sevoflurane resistance C50 is drawn
# separately with the propofol variance.
SU_PROPOFOL_TPR_C50_OMEGA = 0.44
SU_REMIFENTANIL_TPR_EMAX_OMEGA = 0.449
# Olofsen 2026 supplement Table 3, log-scale variances. One draw applies to both
# morphine and hydromorphone; correlations were not reported, so draws are independent.
OLOFSEN_OPIOID_RATE_OMEGA = {"half_time": 0.594, "c25": 0.552, "gamma": 0.264}


@dataclass(frozen=True)
class KineticVariability:
    """Volumes, clearances, and the ke0 of the main effect site. In Eleveld's
    models, Q2 and Q3 also scale with V2 and V3."""

    v1: float = 1.0
    v2: float = 1.0
    v3: float = 1.0
    cl: float = 1.0
    q2: float = 1.0
    q3: float = 1.0
    ke0: float = 1.0


@dataclass(frozen=True)
class PropofolVariability(KineticVariability):
    """Eleveld 2018 PK and BIS random effects."""

    ce50: float = 1.0


@dataclass(frozen=True)
class HemodynamicVariability:
    """Random effects on vascular resistance: Su 2023 for propofol and opioids, and
    the sevoflurane C50. The propofol C50 also sets sympathetic venous pooling,
    and the opioid Emax multiplier stops at zero."""

    propofol_tpr_c50: float = 1.0
    opioid_tpr_emax: float = 1.0
    sevoflurane_tpr_c50: float = 1.0


@dataclass(frozen=True)
class OpioidRateVariability:
    """Morphine and hydromorphone respiratory-rate timing, sensitivity, and shape."""

    half_time: float = 1.0
    c25: float = 1.0
    gamma: float = 1.0


@dataclass(frozen=True)
class PatientVariability:
    propofol: PropofolVariability = PropofolVariability()
    hemodynamics: HemodynamicVariability = HemodynamicVariability()
    opioid_rate: OpioidRateVariability = OpioidRateVariability()
    remifentanil: KineticVariability = KineticVariability()
    fentanyl: KineticVariability = KineticVariability()
    # Multiplier on sevoflurane MAC and MAC-awake. Its BIS effect varies with the
    # propofol BIS C50.
    sevoflurane_mac: float = 1.0


TYPICAL = PatientVariability()


def _log_normal(rng: np.random.Generator, omega: dict[str, float], sd_scale: float = 1.0) -> dict[str, float]:
    return {name: math.exp(rng.normal(0.0, sd_scale * math.sqrt(variance))) for name, variance in omega.items()}


def sample_variability(seed: int | None) -> PatientVariability:
    """Draw one patient; a given seed always gives the same patient."""
    rng = np.random.default_rng(None if seed is None else [seed, 1])
    propofol = PropofolVariability(**_log_normal(rng, ELEVELD_PROPOFOL_OMEGA, ELEVELD_SD_SCALE))
    propofol_tpr_c50 = math.exp(rng.normal(0.0, math.sqrt(SU_PROPOFOL_TPR_C50_OMEGA)))
    opioid_tpr_emax = max(0.0, 1.0 + rng.normal(0.0, math.sqrt(SU_REMIFENTANIL_TPR_EMAX_OMEGA)))
    opioid_rate = OpioidRateVariability(**_log_normal(rng, OLOFSEN_OPIOID_RATE_OMEGA))
    # New parameters are drawn last so that existing seeds give the same values.
    remifentanil = KineticVariability(**_log_normal(rng, ELEVELD_REMIFENTANIL_OMEGA))
    fentanyl = KineticVariability(**_log_normal(rng, BAE_FENTANYL_OMEGA))
    sevoflurane_mac = math.exp(rng.normal(0.0, math.sqrt(HEYSE_SEVOFLURANE_MAC_OMEGA)))
    hemodynamics = HemodynamicVariability(
        propofol_tpr_c50=propofol_tpr_c50,
        opioid_tpr_emax=opioid_tpr_emax,
        sevoflurane_tpr_c50=math.exp(rng.normal(0.0, math.sqrt(SU_PROPOFOL_TPR_C50_OMEGA))),
    )
    return PatientVariability(
        propofol=propofol, hemodynamics=hemodynamics, opioid_rate=opioid_rate,
        remifentanil=remifentanil, fentanyl=fentanyl, sevoflurane_mac=sevoflurane_mac,
    )
