from __future__ import annotations

import numpy as np

from anasim.core.utils import clamp, clamp01
from anasim.patient.patient import Patient


class TOFModel:
    """Train-of-four model for rocuronium with spontaneous recovery and sugammadex reversal."""

    MW_ROCURONIUM = 609.7
    MW_SUGAMMADEX = 2178.0

    def __init__(self, patient: Patient, model_name: str = "Wierda", anesthesia_type: str = "TIVA"):
        self.patient = patient
        self.model_name = model_name
        self.anesthesia_type = anesthesia_type

        age = patient.age
        sex = 1 if patient.sex.lower() == "female" else 0
        age_term = age - 50.0
        theta7 = None

        if model_name == "Szenohradszky":
            theta2 = 1.44
            theta3 = 8.30
            theta5 = -0.00862
            theta6 = -0.0981
        elif model_name == "Cooper":
            theta2 = 0.980
            theta3 = 6.18
            theta5 = -0.00557
            theta6 = -0.0341
            theta7 = -1.32
        elif model_name == "Alvarez-Gomez":
            theta2 = 0.900
            theta3 = 5.99
            theta5 = -0.00539
            theta6 = -0.0443
            theta7 = -1.14
        elif model_name == "McCoy":
            theta2 = 1.08
            theta3 = 4.20
            theta5 = -0.00770
            theta6 = -0.0283
        else:
            theta2 = 1.08
            theta3 = 6.41
            theta5 = -0.00605
            theta6 = -0.0494
            theta7 = -1.24

        self.ce50_base = max(theta2 + theta5 * age_term, 0.01)
        self.gamma = max(theta3 + theta6 * age_term + ((theta7 or 0.0) * sex), 0.5)
        self.ke0_onset = 0.16
        self.recovery_ke0 = 0.12
        self.Ce50_T1 = 0.8
        self.gamma_T1 = 3.0
        self.beta_TOF = 1.5
        self.reversal_ke0 = 1.0
        self.f_sevo = 0.75
        self.f_n2o = 0.6
        self.Vs_L_kg = 0.18
        self.Cl_s_mL_min = 88.0
        self.Vs = self.Vs_L_kg * patient.weight
        self.kel_s = (self.Cl_s_mL_min / 1000.0) / self.Vs
        self.Ka = 1.79e7

        self.ce = 0.0
        self.prev_cp = 0.0
        self.sugammadex_amount_umol = 0.0

    def step_recovery(self, dt_sec: float, cp_roc_mg_l: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        dt_min = dt_sec / 60.0
        if self.sugammadex_amount_umol > 0:
            self.sugammadex_amount_umol *= np.exp(-self.kel_s * dt_min)

        cp_free = self._compute_free_rocuronium(cp_roc_mg_l)
        if cp_free >= self.prev_cp:
            ke0 = self.ke0_onset
        elif self.ce > cp_free * 3.0 and self.sugammadex_amount_umol > 0:
            ke0 = self.reversal_ke0
        else:
            ke0 = self.recovery_ke0
        self.prev_cp = cp_free

        self.ce += ke0 * (cp_free - self.ce) * dt_min
        self.ce = max(0.0, self.ce)
        return self._compute_tof_from_ce(self.ce, mac_sevo, mac_n2o)

    def _compute_free_rocuronium(self, cp_total_mg_l: float) -> float:
        if self.sugammadex_amount_umol <= 0 or cp_total_mg_l <= 0:
            return cp_total_mg_l

        r_tot = (cp_total_mg_l / self.MW_ROCURONIUM) / 1000.0
        s_tot = (self.sugammadex_amount_umol / self.Vs) / 1e6
        if r_tot <= 0 or s_tot <= 0:
            return cp_total_mg_l

        a = self.Ka
        b = -(self.Ka * (r_tot + s_tot) + 1.0)
        c_coef = self.Ka * r_tot * s_tot
        discriminant = max(0.0, b * b - 4.0 * a * c_coef)
        complex_conc = (-b - np.sqrt(discriminant)) / (2.0 * a)
        r_free = max(0.0, r_tot - complex_conc)
        return r_free * self.MW_ROCURONIUM * 1000.0

    def _compute_tof_from_ce(self, ce: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        f_effective = 1.0
        mac_sevo = max(0.0, mac_sevo)
        mac_n2o = max(0.0, mac_n2o)
        if mac_sevo > 0.0:
            f_effective -= (1.0 - self.f_sevo) * min(mac_sevo, 1.0)
        if mac_n2o > 0.0:
            f_effective -= (1.0 - self.f_n2o) * min(mac_n2o, 1.0)
        ce50_eff = self.Ce50_T1 * clamp(f_effective, 0.2, 1.0)

        if ce <= 1e-9:
            return 100.0

        ce50_g = ce50_eff ** self.gamma_T1
        ce_g = max(0.0, ce) ** self.gamma_T1
        block = ce_g / (ce50_g + ce_g)
        twitch = clamp01(1.0 - block)
        return (twitch ** self.beta_TOF) * 100.0

    def give_sugammadex(self, dose_mg: float, weight_kg: float = None):
        total_dose_mg = dose_mg * weight_kg if weight_kg is not None else dose_mg
        dose_umol = (total_dose_mg / self.MW_SUGAMMADEX) * 1000.0
        self.sugammadex_amount_umol += dose_umol

    def compute_tof_from_ce(self, ce_roc: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        return self._compute_tof_from_ce(ce_roc, mac_sevo, mac_n2o)

    def reset(self):
        self.ce = 0.0
        self.prev_cp = 0.0
        self.sugammadex_amount_umol = 0.0


__all__ = ["TOFModel"]
