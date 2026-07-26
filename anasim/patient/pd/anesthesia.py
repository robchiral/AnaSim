from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from anasim.core.state import SUPPORTED_MODEL_OPTIONS
from anasim.core.utils import clamp
from anasim.patient.patient import Patient


def compute_bis_from_mac(m_eff: float, agent_offset: float = 0.0) -> float:
    """Core BIS-MAC relationship fitted to sevoflurane BIS data."""
    if m_eff < 0:
        m_eff = 0.0
    bis_base = 25.0 + 70.0 / (1.0 + (m_eff / 0.694) ** 3.326)
    return clamp(bis_base + agent_offset, 30.0, 98.0)


def compute_mac_equivalent_from_drugs(ce_prop: float, remi_rate_ug_kg_min: float) -> float:
    """Convert IV drug effect to a MAC-equivalent hypnotic contribution."""
    delta_m_prop = 0.18 * ce_prop
    delta_m_remi = 0.0
    return min(delta_m_prop + delta_m_remi, 2.0)


@dataclass
class BISModelParams:
    c50p: float
    c50r: float
    gamma: float
    beta: float
    e0: float
    emax: float
    delay: float


class BISModel:
    """BIS PD model for IV agents and volatile anesthetics."""

    def __init__(self, patient: Patient, model_name: str = "Bouillon"):
        if model_name not in SUPPORTED_MODEL_OPTIONS["bis_model"]:
            raise ValueError(f"Unsupported BIS model: {model_name!r}")
        self.model_name = model_name
        self.patient = patient
        self.bis_smoothed = 98.0
        self.tau_smooth = 10.0
        self.alpha_smooth = 0.1
        self.params = BISModelParams(
            c50p=4.47, c50r=19.3, gamma=1.43, beta=0.0, e0=97.4, emax=97.4, delay=0.0
        )

        if self.model_name == "Eleveld":
            age = patient.age

            def faging(x):
                return np.exp(x * (age - 35))

            def fdelay(x):
                return 15 + np.exp(x * age)

            self.params.c50p = 3.08 * faging(-0.00635)
            self.params.c50r = 0.0
            self.params.gamma = 1.89
            self.params.gamma2 = 1.47
            self.params.e0 = 93.0
            self.params.emax = 93.0
            self.params.delay = fdelay(0.0517)
        elif self.model_name == "Fuentes":
            self.params.c50p = 2.99
            self.params.c50r = 21.0
            self.params.gamma = 2.69
            self.params.beta = 0.0
            self.params.e0 = 94.0
            self.params.emax = 94.0 * 0.81
        elif self.model_name == "Yumuk":
            self.params.c50p = 7.66
            self.params.c50r = 149.62
            self.params.gamma = 4.07
            self.params.beta = 15.03
            self.params.e0 = 93.97
            self.params.emax = 93.97

        self.output_buffer = deque([self.params.e0])

    def initialize(self, bis_target: float, dt: float = 0.01):
        self.bis_smoothed = bis_target
        steps_delay = int(np.ceil(self.params.delay / dt)) if self.params.delay > 0 else 10
        self.output_buffer = deque([bis_target] * max(1, steps_delay))

    def step(
        self,
        dt: float,
        ce_prop: float,
        ce_remi: float = 0.0,
        mac_sevo: float = 0.0,
        remi_rate_ug_kg_min: float = 0.0,
    ) -> float:
        total_mac = mac_sevo
        if total_mac > 0.01:
            bis_raw = self._compute_bis_volatile(ce_prop, mac_sevo, remi_rate_ug_kg_min)
        else:
            bis_raw = self._compute_bis_iv_only(ce_prop, ce_remi)

        alpha = 1.0 if dt <= 0 else 1.0 - np.exp(-dt / self.tau_smooth)
        self.bis_smoothed = (1.0 - alpha) * self.bis_smoothed + alpha * bis_raw
        bis_out = self.bis_smoothed

        steps_delay = int(np.ceil(self.params.delay / dt)) if self.params.delay > 0 else 0
        if steps_delay > 0:
            if len(self.output_buffer) != steps_delay:
                self.output_buffer = deque([self.bis_smoothed] * steps_delay, maxlen=steps_delay)
            self.output_buffer.append(bis_out)
            return self.output_buffer.popleft()
        return bis_out

    def compute_bis(self, ce_prop: float, ce_remi: float = 0.0, u_volatile: float = 0.0) -> float:
        if u_volatile > 0.01:
            remi_rate_ug_kg_min = ce_remi * 0.04
            return self._compute_bis_volatile(ce_prop, mac_sevo=u_volatile, remi_rate_ug_kg_min=remi_rate_ug_kg_min)
        return self._compute_bis_iv_only(ce_prop, ce_remi)

    def _compute_bis_volatile(self, ce_prop: float, mac_sevo: float, remi_rate_ug_kg_min: float) -> float:
        delta_m_iv = compute_mac_equivalent_from_drugs(ce_prop, remi_rate_ug_kg_min)
        bis_values = []
        if mac_sevo > 0.01:
            m_eff_sevo = mac_sevo + delta_m_iv
            bis_values.append((mac_sevo, compute_bis_from_mac(m_eff_sevo, agent_offset=0.0)))
        total_mac = sum(mac for mac, _ in bis_values)
        if total_mac == 0:
            return 98.0
        return sum(mac * bis for mac, bis in bis_values) / total_mac

    def _compute_bis_iv_only(self, ce_prop: float, ce_remi: float) -> float:
        p = self.params
        u_prop = ce_prop / p.c50p if p.c50p > 1e-6 else 0.0
        u_remi = ce_remi / p.c50r if p.c50r > 1e-6 else 0.0
        gamma = p.gamma

        if self.model_name == "Bouillon":
            phi = u_prop / (u_prop + u_remi) if (u_prop + u_remi) > 0 else 0.0
            u50 = 1 - p.beta * (phi - phi**2)
            interaction = (u_prop + u_remi) / u50
        else:
            interaction = u_prop + u_remi + p.beta * u_prop * u_remi
            if self.model_name == "Eleveld" and hasattr(p, "gamma2") and u_prop > 1.0:
                gamma = p.gamma2

        term = interaction ** gamma
        effect = p.emax * (term / (1 + term))
        return max(0.0, p.e0 - effect)

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


__all__ = [
    "BISModel",
    "BISModelParams",
    "LOCModel",
    "TOLModel",
    "compute_bis_from_mac",
    "compute_mac_equivalent_from_drugs",
]
