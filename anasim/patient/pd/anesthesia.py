from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from anasim.core.state import SUPPORTED_MODEL_OPTIONS
from anasim.patient.patient import Patient

# Sevoflurane potency on the BIS response surface is anchored to BIS ~41 at
# 1 MAC (Kanazawa 2017; Ryu 2018), which gives BIS ~30 at 1.5 MAC (Paraskeva 2005).
SEVO_BIS_AT_1_MAC = 41.0


@dataclass(frozen=True)
class BISModelParams:
    c50p: float
    c50r: float
    gamma: float
    beta: float
    e0: float
    emax: float
    delay: float = 0.0
    gamma_above_c50: float | None = None  # Eleveld uses a second slope above Ce50.


BIS_MODEL_PARAMS = {
    "Bouillon": BISModelParams(c50p=4.47, c50r=19.3, gamma=1.43, beta=0.0, e0=97.4, emax=97.4),
    "Fuentes": BISModelParams(c50p=2.99, c50r=21.0, gamma=2.69, beta=0.0, e0=94.0, emax=94.0 * 0.81),
    "Yumuk": BISModelParams(c50p=7.66, c50r=149.62, gamma=4.07, beta=15.03, e0=93.97, emax=93.97),
}


class BISModel:
    """Propofol-remifentanil BIS response surface with additive sevoflurane.

    Sevoflurane adds MAC / mac50 to the model's interaction term, where mac50
    places 1 MAC at SEVO_BIS_AT_1_MAC. Propofol and sevoflurane therefore add
    on one continuous surface (Schumacher 2009) for every IV model choice.
    """

    def __init__(self, patient: Patient, model_name: str = "Bouillon"):
        if model_name not in SUPPORTED_MODEL_OPTIONS["bis_model"]:
            raise ValueError(f"Unsupported BIS model: {model_name!r}")
        self.model_name = model_name
        if model_name == "Eleveld":
            # Eleveld 2018: arterial Ce50 with age, no opioid term, and age-dependent delay (s).
            self.params = BISModelParams(
                c50p=3.08 * math.exp(-0.00635 * (patient.age - 35)),
                c50r=0.0,
                gamma=1.89,
                gamma_above_c50=1.47,
                beta=0.0,
                e0=93.0,
                emax=93.0,
                delay=15.0 + math.exp(0.0517 * patient.age),
            )
        else:
            self.params = BIS_MODEL_PARAMS[model_name]
        self.mac50 = 1.0 / self._interaction_for_bis(SEVO_BIS_AT_1_MAC)
        # Monitor processing: 10 s smoothing plus the model's transport delay.
        self.tau_smooth = 10.0
        self.initialize(self.params.e0)

    def _gamma(self, interaction: float) -> float:
        p = self.params
        return p.gamma_above_c50 if p.gamma_above_c50 and interaction > 1.0 else p.gamma

    def _interaction_for_bis(self, bis: float) -> float:
        """Invert the Hill surface for a single-drug interaction term."""
        p = self.params
        odds = (p.e0 - bis) / (p.emax - p.e0 + bis)
        if odds <= 0.0:
            raise ValueError(f"{self.model_name} BIS model cannot reach BIS {bis:g}")
        return odds ** (1.0 / self._gamma(odds))

    def initialize(self, bis: float) -> None:
        self.bis_smoothed = bis
        self._clock = 0.0
        self._history = deque([(0.0, bis)])

    def compute_bis(self, ce_prop: float, ce_remi: float = 0.0, mac_sevo: float = 0.0) -> float:
        p = self.params
        u_prop = max(0.0, ce_prop) / p.c50p
        u_remi = max(0.0, ce_remi) / p.c50r if p.c50r > 0 else 0.0
        interaction = u_prop + u_remi + p.beta * u_prop * u_remi + max(0.0, mac_sevo) / self.mac50
        term = interaction ** self._gamma(interaction)
        return max(0.0, p.e0 - p.emax * term / (1.0 + term))

    def step(self, dt: float, ce_prop: float, ce_remi: float = 0.0, mac_sevo: float = 0.0) -> float:
        """Return the smoothed, delayed BIS after dt seconds."""
        alpha = 1.0 - math.exp(-dt / self.tau_smooth)
        self.bis_smoothed += alpha * (self.compute_bis(ce_prop, ce_remi, mac_sevo) - self.bis_smoothed)
        if self.params.delay <= 0.0:
            return self.bis_smoothed
        self._clock += dt
        history = self._history
        history.append((self._clock, self.bis_smoothed))
        while len(history) > 1 and history[1][0] <= self._clock - self.params.delay:
            history.popleft()
        return history[0][1]


class LOCModel:
    """Loss-of-consciousness probability model."""

    def __init__(self, model_name: str = "Kern"):
        if model_name not in SUPPORTED_MODEL_OPTIONS["loc_model"]:
            raise ValueError(f"Unsupported LOC model: {model_name!r}")
        self.model_name = model_name
        self.c50p = 1.80
        self.c50r = 12.5
        self.gamma = 3.76
        self.beta = 5.1

        if model_name == "Mertens":
            self.c50p = 2.92
            self.c50r = 5.15
            self.gamma = 3.88
            self.beta = 0.0
        elif model_name == "Johnson":
            self.c50p = 2.20
            self.c50r = 33.1
            self.gamma = 5.00
            self.beta = 3.60

        self.mac_awake_sevo = 0.30
        self.mac_awake_n2o = 0.61
        self.n2o_sevo_awake_interaction = 0.7

    def compute_probability(
        self,
        ce_prop: float,
        ce_remi: float,
        mac_sevo: float = 0.0,
        mac_n2o: float = 0.0,
    ) -> float:
        awake_units = 0.0
        if mac_sevo > 0:
            awake_units += mac_sevo / self.mac_awake_sevo
        if mac_n2o > 0:
            n2o_units = mac_n2o / self.mac_awake_n2o
            if mac_sevo > 0:
                n2o_units *= self.n2o_sevo_awake_interaction
            awake_units += n2o_units

        ce_effective = ce_prop + awake_units * self.c50p
        up = ce_effective / self.c50p
        ur = ce_remi / self.c50r
        interaction = up + ur + self.beta * up * ur
        term = interaction ** self.gamma
        return term / (1 + term)


class TOLModel:
    """Tolerance-of-laryngoscopy probability model."""

    def __init__(self):
        self.c50p = 8.04
        self.c50r = 1.07
        self.gamma_p = 5.1
        self.gamma_r = 0.97
        self.pre_intensity = 1.05

    def compute_probability(self, ce_prop: float, ce_remi: float, mac: float = 0.0) -> float:
        c50r_scaled = self.c50r * self.pre_intensity
        fsig_r = 0.0 if c50r_scaled == 0 else (ce_remi**self.gamma_r) / (c50r_scaled**self.gamma_r + ce_remi**self.gamma_r)
        post_opioid = self.pre_intensity * (1.0 - fsig_r)
        c50p_scaled = self.c50p * post_opioid
        if c50p_scaled <= 1e-6:
            return 1.0
        ce_effective = ce_prop + (mac * self.c50p)
        return (ce_effective**self.gamma_p) / (c50p_scaled**self.gamma_p + ce_effective**self.gamma_p)


__all__ = ["BISModel", "BISModelParams", "LOCModel", "TOLModel"]
