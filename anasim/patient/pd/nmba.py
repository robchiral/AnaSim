from __future__ import annotations

import copy
import math

from anasim.core.utils import clamp, clamp01
from anasim.patient.patient import Patient
from anasim.patient.pk_models import MammillaryPK

MW_ROCURONIUM = 609.7
MW_SUGAMMADEX = 2178.0

# Kleijn 2011 integrated rocuronium-sugammadex model (446 subjects).
# Complexation in plasma: Kd 0.0559 umol/L (in vitro), dissociation exp(-3.38) /min.
SUGAMMADEX_KD_UM = 0.0559
SUGAMMADEX_K2_PER_MIN = math.exp(-3.38)
SUGAMMADEX_K1_PER_UM_MIN = SUGAMMADEX_K2_PER_MIN / SUGAMMADEX_KD_UM
# Adductor pollicis TOF ratio: EC50 1.62 umol/L, Hill 7.52, effect site linked to plasma.
ROCURONIUM_TOF_EC50 = 1.62 * MW_ROCURONIUM / 1000.0  # mg/L
ROCURONIUM_TOF_HILL = 7.52
# T1 twitch height follows TOF = T1^1.5; its EC50 sets the central muscle potency.
TOF_FADE_EXPONENT = 1.5
ROCURONIUM_T1_EC50 = ROCURONIUM_TOF_EC50 * (0.5**-TOF_FADE_EXPONENT - 1.0) ** (1.0 / ROCURONIUM_TOF_HILL)


def _sugammadex_pk(patient: Patient) -> MammillaryPK:
    """Kleijn 2011 sugammadex PK (umol, umol/L); the complex shares it.

    Clearance follows creatinine clearance; the effect site is unused.
    """
    w = patient.weight
    size = w / 70.0
    crcl = patient.creatinine_clearance()
    return MammillaryPK(
        v1=4.42 * size * (1.0 - 0.00354 * (w - 74.5)),
        cl1=0.093 * (2.0 * crcl / (crcl + 119.0)) ** 1.29 * (1.0 + 0.00378 * (w - 74.5)),
        v2=6.35 * size * math.exp(-0.00305 * (crcl - 119.0)),
        cl2=0.206 * size**0.75,
        ke0=1.0,
        renal_fraction=0.95,
        cl1_co_exponent=0.0,
        distribution_co_exponent=0.0,
    )


class TOFModel:
    """Train-of-four response to rocuronium and sugammadex reversal (Kleijn 2011).

    Rocuronium PK holds free drug. Sugammadex binds it in plasma at finite
    rates; the complex shares sugammadex clearance. Free plasma sugammadex
    increases effect-site efflux and controls reversal speed. Kleijn's efflux
    rate is empirical; binding within the neuromuscular junction is unmodeled.
    """

    def __init__(self, patient: Patient):
        size = patient.weight / 70.0
        self.ke0 = 0.134 * size**-0.25  # /min
        self.ks = math.exp(-3.43) * size**-0.25  # /min per umol/L free sugammadex
        # Kleijn's sevoflurane covariate, applied in proportion to brain MAC.
        self.sevo_ke0_reduction = 0.567
        self.sevo_ec50_reduction = 0.395
        # Nitrous oxide EC50 multiplier at 1 MAC (Fiset 1991).
        self.f_n2o = 0.6
        # The central site (diaphragm and laryngeal adductors) equilibrates
        # faster than the adductor pollicis (t1/2 ke0 2.7 vs 4.4 min; Plaud 1995).
        self.central_ke0_ratio = 4.4 / 2.7
        self.patient = patient
        self.reset()

    def reset(self) -> None:
        self.sugammadex = _sugammadex_pk(self.patient)
        self.complex = _sugammadex_pk(self.patient)
        # Rocuronium (mg/L) at the adductor pollicis and central effect sites.
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

    def simulate_decay(
        self, roc: MammillaryPK, *, max_seconds: int = 7200,
        mac_sevo: float = 0.0, mac_n2o: float = 0.0,
    ) -> float:
        """Minutes for adductor pollicis rocuronium to halve with infusions stopped.

        Hemodynamic covariates and inhaled MAC are held at their current values.
        """
        target = self.ce * 0.5
        if target <= 0.0:
            return 0.0
        rocuronium = copy.deepcopy(roc)
        neuromuscular = copy.deepcopy(self)
        for second in range(1, max_seconds + 1):
            rocuronium.step(1.0, 0.0)
            neuromuscular.step_recovery(1.0, rocuronium, mac_sevo, mac_n2o)
            if neuromuscular.ce <= target:
                return second / 60.0
        return max_seconds / 60.0

    def step_recovery(self, dt_sec: float, roc: MammillaryPK, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        """Advance binding and the effect sites after the rocuronium PK step; return TOF %."""
        dt_min = dt_sec / 60.0
        sugammadex, complex_ = self.sugammadex, self.complex
        if sugammadex.state.c1 > 0.0 or complex_.state.c1 > 0.0:
            self._bind_in_plasma(dt_min, roc)
        sugammadex.step(dt_sec, 0.0)
        complex_.step(dt_sec, 0.0)

        ke0 = self.ke0 * (1.0 - self.sevo_ke0_reduction * clamp01(mac_sevo))
        removal = self.ks * sugammadex.state.c1
        cp = roc.state.c1
        self.ce = self._effect_site(self.ce, cp, ke0, removal, dt_min)
        ratio = self.central_ke0_ratio
        self.ce_central = self._effect_site(self.ce_central, cp, ke0 * ratio, removal * ratio, dt_min)
        return self.compute_tof_from_ce(self.ce, mac_sevo, mac_n2o)

    @staticmethod
    def _effect_site(ce: float, cp: float, ke0: float, removal: float, dt_min: float) -> float:
        """Exact step of dCe/dt = ke0 (Cp - Ce) - removal Ce at constant inputs."""
        rate = ke0 + removal
        if rate <= 0.0:
            return ce
        target = ke0 * cp / rate
        return target + (ce - target) * math.exp(-rate * dt_min)

    def _bind_in_plasma(self, dt_min: float, roc: MammillaryPK) -> None:
        """Association k1 Croc Csug and dissociation k2 Ccplx in the sugammadex central volume."""
        sugammadex, complex_ = self.sugammadex, self.complex
        v_sug = sugammadex.v1
        to_um = 1000.0 / MW_ROCURONIUM
        fastest = SUGAMMADEX_K1_PER_UM_MIN * max(
            sugammadex.state.c1 * v_sug / roc.v1, roc.state.c1 * to_um
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

    def _ec50(self, mac_sevo: float, mac_n2o: float) -> float:
        factor = (1.0 - self.sevo_ec50_reduction * clamp01(mac_sevo)) * (
            1.0 - (1.0 - self.f_n2o) * clamp01(mac_n2o)
        )
        return ROCURONIUM_TOF_EC50 * clamp(factor, 0.2, 1.0)

    def compute_tof_from_ce(self, ce: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        """TOF ratio (%) at the adductor pollicis."""
        if ce <= 1e-9:
            return 100.0
        return 100.0 / (1.0 + (ce / self._ec50(mac_sevo, mac_n2o)) ** ROCURONIUM_TOF_HILL)

    def twitch_block(self, ce: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        """Fractional T1 depression at the adductor pollicis."""
        tof = self.compute_tof_from_ce(ce, mac_sevo, mac_n2o) / 100.0
        return 1.0 - tof ** (1.0 / TOF_FADE_EXPONENT)

    def pharyngeal_tone_fraction(self, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        """Closing-pressure reserve (Herbstreit 2009: 37% loss at TOF 0.8, 54% at 0.5).

        Interpolation and coupling to hypnotic collapse are calibrations.
        """
        tof = self.compute_tof_from_ce(self.ce, mac_sevo, mac_n2o) / 100.0
        if tof >= 0.8:
            reduction = 0.37 * (1.0 - tof) / 0.2
        else:
            reduction = 0.37 + 0.17 * clamp01((0.8 - tof) / 0.3)
        return 1.0 - reduction


__all__ = ["ROCURONIUM_T1_EC50", "TOFModel"]
