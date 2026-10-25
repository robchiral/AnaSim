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
# Su 2023 OMEGA: log-normal on the propofol resistance C50, additive on the
# remifentanil resistance Emax of -1.
SU_PROPOFOL_TPR_C50_OMEGA = 0.44
SU_REMIFENTANIL_TPR_EMAX_OMEGA = 0.449


@dataclass(frozen=True)
class PropofolVariability:
    """Eleveld 2018 PK and BIS random effects; Q2 and Q3 also follow V2 and V3."""

    v1: float = 1.0
    v2: float = 1.0
    v3: float = 1.0
    cl: float = 1.0
    q2: float = 1.0
    q3: float = 1.0
    ce50: float = 1.0
    ke0: float = 1.0


@dataclass(frozen=True)
class HemodynamicVariability:
    """Su 2023 random effects. The propofol C50 also sets sympathetic venous pooling,
    and the opioid Emax multiplier stops at zero."""

    propofol_tpr_c50: float = 1.0
    opioid_tpr_emax: float = 1.0


@dataclass(frozen=True)
class PatientVariability:
    propofol: PropofolVariability = PropofolVariability()
    hemodynamics: HemodynamicVariability = HemodynamicVariability()


TYPICAL = PatientVariability()


def sample_variability(seed: int | None) -> PatientVariability:
    """Draw one patient; a given seed always gives the same patient."""
    rng = np.random.default_rng(None if seed is None else [seed, 1])
    propofol = PropofolVariability(**{
        name: math.exp(rng.normal(0.0, ELEVELD_SD_SCALE * math.sqrt(omega)))
        for name, omega in ELEVELD_PROPOFOL_OMEGA.items()
    })
    hemodynamics = HemodynamicVariability(
        propofol_tpr_c50=math.exp(rng.normal(0.0, math.sqrt(SU_PROPOFOL_TPR_C50_OMEGA))),
        opioid_tpr_emax=max(0.0, 1.0 + rng.normal(0.0, math.sqrt(SU_REMIFENTANIL_TPR_EMAX_OMEGA))),
    )
    return PatientVariability(propofol=propofol, hemodynamics=hemodynamics)
