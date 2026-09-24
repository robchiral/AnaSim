from __future__ import annotations

import numpy as np

from anasim.core.utils import clamp, clamp01
from anasim.patient.patient import Patient


class TOFModel:
    """Train-of-four model for rocuronium with spontaneous recovery and sugammadex reversal."""

    MW_ROCURONIUM = 609.7
    MW_SUGAMMADEX = 2178.0

    def __init__(self, patient: Patient):
        # Adductor pollicis effect site with faster onset than offset
        # (Plaud 1995: t1/2 ke0 4.4 min, Ce50 823 ug/L).
        self.ke0_onset = 0.16
        self.recovery_ke0 = 0.12
        self.reversal_ke0 = 1.0
        self.Ce50_T1 = 0.8
        self.gamma_T1 = 3.0
        self.beta_TOF = 1.5
        # The central site (diaphragm and laryngeal adductors) uses laryngeal
        # kinetics, which equilibrate faster than the adductor pollicis
        # (t1/2 ke0 2.7 vs 4.4 min; Plaud 1995) but need more drug.
        self.central_ke0_ratio = 4.4 / 2.7
        # Volatile potentiation: EC50 multiplier at 1 MAC.
        self.f_sevo = 0.75
        self.f_n2o = 0.6
        # Sugammadex PK (Vd 0.18 L/kg, CL 88 mL/min) and binding constant (M^-1).
        self.Vs = 0.18 * patient.weight
        self.kel_s = 0.088 / self.Vs
        self.Ka = 1.79e7

        self.ce = 0.0
        self.ce_central = 0.0
        self.prev_cp = 0.0
        self.sugammadex_amount_umol = 0.0

    def step_recovery(self, dt_sec: float, cp_roc_mg_l: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        dt_min = dt_sec / 60.0
        if self.sugammadex_amount_umol > 0:
            self.sugammadex_amount_umol *= np.exp(-self.kel_s * dt_min)

        cp_free = self._compute_free_rocuronium(cp_roc_mg_l)
        rising = cp_free >= self.prev_cp
        self.prev_cp = cp_free

        ke0 = self._ke0(self.ce, cp_free, rising)
        self.ce = max(0.0, self.ce + ke0 * (cp_free - self.ce) * dt_min)
        ke0_central = self._ke0(self.ce_central, cp_free, rising) * self.central_ke0_ratio
        self.ce_central = max(0.0, self.ce_central + ke0_central * (cp_free - self.ce_central) * dt_min)
        return self.compute_tof_from_ce(self.ce, mac_sevo, mac_n2o)

    def _ke0(self, ce: float, cp_free: float, rising: bool) -> float:
        if rising:
            return self.ke0_onset
        if ce > cp_free * 3.0 and self.sugammadex_amount_umol > 0:
            return self.reversal_ke0
        return self.recovery_ke0

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

    def twitch_block(self, ce: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        """Fractional T1 depression of peripheral (adductor pollicis) muscle."""
        if ce <= 1e-9:
            return 0.0
        f_effective = 1.0
        f_effective -= (1.0 - self.f_sevo) * clamp01(mac_sevo)
        f_effective -= (1.0 - self.f_n2o) * clamp01(mac_n2o)
        ce50_eff = self.Ce50_T1 * clamp(f_effective, 0.2, 1.0)
        ce_g = ce**self.gamma_T1
        return ce_g / (ce50_eff**self.gamma_T1 + ce_g)

    def compute_tof_from_ce(self, ce: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        twitch = 1.0 - self.twitch_block(ce, mac_sevo, mac_n2o)
        return (twitch ** self.beta_TOF) * 100.0

    def give_sugammadex(self, dose_mg: float) -> None:
        self.sugammadex_amount_umol += dose_mg / self.MW_SUGAMMADEX * 1000.0

    def reset(self):
        self.ce = 0.0
        self.ce_central = 0.0
        self.prev_cp = 0.0
        self.sugammadex_amount_umol = 0.0


__all__ = ["TOFModel"]
