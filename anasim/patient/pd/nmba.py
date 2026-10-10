from __future__ import annotations

import math

import numpy as np

from anasim.core.utils import clamp, clamp01
from anasim.patient.patient import Patient
from anasim.patient.pk_models import MammillaryPK, organ_clearance_scaler

MW_ROCURONIUM = 609.7
MW_SUGAMMADEX = 2178.0
# Ploeger 2009: Kd 0.1 umol/L (in vitro) and dissociation 0.00216 /min.
SUGAMMADEX_KD_UM = 0.1
SUGAMMADEX_K2_PER_MIN = 0.00216
SUGAMMADEX_K1_PER_UM_MIN = SUGAMMADEX_K2_PER_MIN / SUGAMMADEX_KD_UM
# Ploeger links the effect site to the first peripheral compartment (ke0 0.655 /min).
SUGAMMADEX_KE0 = 0.655


def _sugammadex_pk(patient: Patient) -> MammillaryPK:
    """Ploeger 2009 sugammadex PK with rocuronium under anesthesia (umol, umol/L).

    The complex shares these parameters. Parameters are scaled from 80 kg;
    clearance is close to GFR. The effect site is unused.
    """
    size = patient.weight / 80.0
    renal = 0.95
    return MammillaryPK(
        v1=3.47 * size,
        cl1=0.109 * size**0.75 * organ_clearance_scaler(patient, renal_fraction=renal),
        v2=2.3 * size,
        cl2=0.427 * size**0.75,
        v3=8.93 * size,
        cl3=0.21 * size**0.75,
        ke0=1.0,
        renal_fraction=renal,
    )


def free_concentration(total: float, binder: float, kd: float) -> float:
    """Free ligand at equilibrium with a 1:1 binder; all in the same molar units."""
    if total <= 0.0:
        return 0.0
    b = binder - total + kd
    root = math.sqrt(b * b + 4.0 * kd * total)
    # Two algebraically equal forms; each avoids cancellation on its side.
    return 2.0 * kd * total / (b + root) if b > 0.0 else 0.5 * (root - b)


class TOFModel:
    """Train-of-four response to rocuronium, with sugammadex binding after Ploeger 2009.

    Rocuronium PK holds free drug. Sugammadex binds it in plasma at finite
    rates; the complex is cleared like sugammadex. Sugammadex reaches the
    neuromuscular effect sites from its first peripheral compartment and binds
    rocuronium there at equilibrium.
    """

    def __init__(self, patient: Patient):
        # Adductor pollicis effect site with faster onset than offset
        # (Plaud 1995: t1/2 ke0 4.4 min, Ce50 823 ug/L).
        self.ke0_onset = 0.16
        self.recovery_ke0 = 0.12
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
        self.patient = patient
        self.reset()

    def reset(self) -> None:
        self.sugammadex = _sugammadex_pk(self.patient)
        self.complex = _sugammadex_pk(self.patient)
        # Total rocuronium (mg/L) and sugammadex (umol/L) at each effect site.
        self.ce_total = self.ce_central_total = 0.0
        self.ce_sugammadex = self.ce_central_sugammadex = 0.0
        # Free rocuronium (mg/L), which sets the block.
        self.ce = self.ce_central = 0.0

    def update_hemodynamics(
        self, v_ratio: float, co_ratio: float, renal_perfusion: float, blood_volume_l: float | None = None,
    ) -> None:
        for model in (self.sugammadex, self.complex):
            model.update_hemodynamics(v_ratio, co_ratio, renal_perfusion, blood_volume_l=blood_volume_l)

    def give_sugammadex(self, dose_mg: float) -> None:
        self.sugammadex.state.c1 += dose_mg / MW_SUGAMMADEX * 1000.0 / self.sugammadex.v1

    def bound_rocuronium(self) -> float:
        """Plasma rocuronium held in the complex, mg/L of complex central volume."""
        return self.complex.state.c1 * MW_ROCURONIUM / 1000.0

    def step_recovery(self, dt_sec: float, roc: MammillaryPK, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        """Advance binding and the effect sites after the rocuronium PK step; return TOF %."""
        dt_min = dt_sec / 60.0
        sugammadex, complex_ = self.sugammadex, self.complex
        if sugammadex.state.c1 > 0.0 or complex_.state.c1 > 0.0:
            self._bind_in_plasma(dt_min, roc)
        sugammadex.step(dt_sec, 0.0)
        complex_.step(dt_sec, 0.0)

        cp_free = roc.state.c1
        self.ce_total = self._approach(self.ce_total, cp_free, dt_min, 1.0)
        self.ce_central_total = self._approach(self.ce_central_total, cp_free, dt_min, self.central_ke0_ratio)
        c_peripheral = sugammadex.state.c2
        self.ce_sugammadex += (c_peripheral - self.ce_sugammadex) * -math.expm1(-SUGAMMADEX_KE0 * dt_min)
        self.ce_central_sugammadex += (c_peripheral - self.ce_central_sugammadex) * -math.expm1(
            -SUGAMMADEX_KE0 * self.central_ke0_ratio * dt_min
        )
        self.ce = self._free_at_site(self.ce_total, self.ce_sugammadex)
        self.ce_central = self._free_at_site(self.ce_central_total, self.ce_central_sugammadex)
        return self.compute_tof_from_ce(self.ce, mac_sevo, mac_n2o)

    def _bind_in_plasma(self, dt_min: float, roc: MammillaryPK) -> None:
        """Ploeger Eqs 3-4: association K1 Croc Csug and dissociation K2 Ccplx in the central volume."""
        sugammadex, complex_ = self.sugammadex, self.complex
        v_sug = sugammadex.v1
        to_um = 1000.0 / MW_ROCURONIUM
        fastest = SUGAMMADEX_K1_PER_UM_MIN * max(
            sugammadex.state.c1, roc.state.c1 * to_um * v_sug / roc.v1
        ) + SUGAMMADEX_K2_PER_MIN
        substeps = max(1, math.ceil(dt_min * fastest / 0.05))
        h = dt_min / substeps
        for _ in range(substeps):
            free_roc_um = roc.state.c1 * to_um
            rate = (SUGAMMADEX_K1_PER_UM_MIN * free_roc_um * sugammadex.state.c1
                    - SUGAMMADEX_K2_PER_MIN * complex_.state.c1)
            flux = rate * v_sug * h  # umol bound this substep
            flux = min(flux, free_roc_um * roc.v1, sugammadex.state.c1 * v_sug)
            flux = max(flux, -complex_.state.c1 * v_sug)
            roc.state.c1 -= flux / roc.v1 / to_um
            sugammadex.state.c1 -= flux / v_sug
            complex_.state.c1 += flux / v_sug

    def _approach(self, ce: float, cp: float, dt_min: float, ratio: float) -> float:
        ke0 = (self.ke0_onset if cp >= ce else self.recovery_ke0) * ratio
        return ce + (cp - ce) * -np.expm1(-ke0 * dt_min)

    @staticmethod
    def _free_at_site(roc_mg_l: float, sugammadex_um: float) -> float:
        to_um = 1000.0 / MW_ROCURONIUM
        return free_concentration(roc_mg_l * to_um, sugammadex_um, SUGAMMADEX_KD_UM) / to_um

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


__all__ = ["TOFModel", "free_concentration"]
