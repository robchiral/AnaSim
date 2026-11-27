from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

from anasim.core.utils import clamp, clamp01, fill_response_rows, hill_function
from anasim.patient.patient import Patient
from anasim.patient.pk_models import (
    MammillaryPK,
    _pregnancy_v1,
    rocuronium_adjusted_weight,
)

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

# Competitive anticholinesterase reversal (neostigmine, ng/mL; Kirkegaard 2002;
# Cronnelly 1979; Herbstreit 2010): saturable rightward shift of rocuronium EC50
# creates a pharmacological ceiling at deep block, while unopposed neostigmine
# after full recovery mildly impairs genioglossus/pharyngeal dilator tone.
NEOSTIGMINE_REVERSAL_C50 = 26.0  # ng/mL at NMJ effect site
NEOSTIGMINE_REVERSAL_GAMMA = 1.6
NEOSTIGMINE_EC50_MAX_SHIFT = 0.68  # Up to 1.68x rightward shift of rocuronium EC50


def succinylcholine_twitch_fraction(ce: float, *, central: bool = False) -> float:
    """Phase I twitch height, calibrated to Kopman 2003 Table 2 (mcg/mL).

    A twofold central EC50 calibrates earlier diaphragm/larynx recovery
    (Dhonneur 1999). Phase I suppresses all four twitches without fade.
    """
    ec50 = 0.84 * (2.0 if central else 1.0)
    return 1.0 / (1.0 + (max(0.0, ce) / ec50) ** 4.0)


def succinylcholine_laryngeal_spasm_factor(ce_central: float) -> float:
    """Residual laryngeal adductor spasm strength (0-1) under succinylcholine.

    Sub-intubating IV doses (0.1-0.3 mg/kg; Chung and Rowbottom 1993) relax glottic spasm
    without paralyzing the diaphragm.
    """
    return 1.0 / (1.0 + (max(0.0, ce_central) / 0.55) ** 2.2)


def succinylcholine_phase2_tof_fraction(phase2_exposure: float) -> float:
    """Phase II fade (TOF fraction 0-1) from cumulative succinylcholine exposure.

    Single intubating doses (<=1.5 mg/kg; exposure < 32 mcg*min/mL) remain pure
    Phase I (TOF 100%); cumulative 3-5 mg/kg produces Phase II fade with
    TOF <= 0.30 (Lee 1975).
    """
    excess = max(0.0, phase2_exposure - 32.0)
    if excess <= 0.0:
        return 1.0
    return 1.0 / (1.0 + (excess / 20.0) ** 2.2)


def _sugammadex_pk(patient: Patient) -> MammillaryPK:
    """Kleijn 2011 sugammadex PK (umol, umol/L); the complex shares it.

    Clearance follows creatinine clearance; the effect site is unused.
    """
    w = patient.weight
    size = w / 70.0
    # Kleijn's linear covariate on weight/70 has a quadratic peak near 178.5 kg;
    # hold the peak above that weight so V1 does not shrink at high body mass.
    w_v1 = min(w, (1.0 + 0.00354 * 74.5) / (2.0 * 0.00354))
    crcl = patient.creatinine_clearance()
    return MammillaryPK(
        v1=_pregnancy_v1(patient, 4.42 * (w_v1 / 70.0) * (1.0 - 0.00354 * (w_v1 - 74.5))),
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
        size = rocuronium_adjusted_weight(patient) / 70.0
        self.ke0 = 0.134 * size**-0.25  # /min
        self.ks = math.exp(-3.43) * size**-0.25  # /min per umol/L free sugammadex
        # Kleijn's sevoflurane covariate, applied in proportion to brain MAC.
        self.sevo_ke0_reduction = 0.567
        self.sevo_ec50_reduction = 0.395
        # Calibrated nitrous oxide EC50 multiplier at 1 MAC (70% N2O = 0.67 MAC
        # lowered vecuronium ED50 13% in Fiset 1991).
        self.f_n2o = 0.81
        # Central muscle equilibration (applied to diaphragm and larynx) uses
        # Plaud 1995 laryngeal adductor vs adductor pollicis t1/2 ke0 (2.7 vs 4.4 min).
        self.central_ke0_ratio = 4.4 / 2.7
        self.patient = patient
        self.reset()

    def reset(self) -> None:
        self.sugammadex = _sugammadex_pk(self.patient)
        self.complex = _sugammadex_pk(self.patient)
        # Rocuronium (mg/L) at the adductor pollicis and central effect sites,
        # plus perijunctional buffer for sub-therapeutic sugammadex rebound (Eleveld 2007).
        self.ce = self.ce_central = 0.0
        self._ce_peri = 0.0
        self.neostigmine_ce = 0.0
        self._sugammadex_given = False
        self._temp_c = self.patient.baseline_temp

    def update_hemodynamics(
        self, v_ratio: float, co_ratio: float, renal_perfusion: float,
        blood_volume_l: float | None = None, temp_c: float = 37.0,
    ) -> None:
        self._temp_c = temp_c
        for model in (self.sugammadex, self.complex):
            model.update_hemodynamics(
                v_ratio, co_ratio, renal_perfusion, blood_volume_l=blood_volume_l, temp_c=temp_c,
            )

    def give_sugammadex(self, dose_mg: float) -> None:
        if dose_mg > 0.0:
            self._sugammadex_given = True
        self.sugammadex.state.c1 += dose_mg / MW_SUGAMMADEX * 1000.0 / self.sugammadex.v1

    def bound_rocuronium(self, roc_v1: float) -> float:
        """Plasma rocuronium held in the complex, mg/L on the rocuronium V1 basis."""
        volume_scale = 1.0 if roc_v1 <= 0.0 else self.complex.v1 / roc_v1
        return self.complex.state.c1 * volume_scale * MW_ROCURONIUM / 1000.0

    def _effective_ke0(self, mac_sevo: float) -> float:
        # Extrapolated from vecuronium in Caldwell 2000 (34.0-38.0 °C): ke0 slows
        # ~10%/°C with core cooling (Q10 2.6) while neuromuscular C50 is unchanged.
        temp_ke0 = 2.6 ** ((clamp(self._temp_c, 18.0, 37.0) - 37.0) / 10.0) if self._temp_c < 37.0 else 1.0
        return self.ke0 * (1.0 - self.sevo_ke0_reduction * clamp01(mac_sevo)) * temp_ke0

    def simulate_decay(
        self, roc: MammillaryPK, *, max_seconds: int = 7200,
        mac_sevo: float = 0.0,
    ) -> float:
        """Minutes for adductor pollicis rocuronium to halve with infusions stopped.

        Hemodynamic covariates and inhaled MAC are held at their current values.
        """
        target = self.ce * 0.5
        if target <= 0.0:
            return 0.0
        if roc.circulation <= 0.0 or max_seconds <= 0:
            return max_seconds / 60.0
        ke0 = self._effective_ke0(mac_sevo)
        delivery_time = roc.circulation / 60.0
        roc_n = 3 if roc.v3 > 0 else (2 if roc.v2 > 0 else 1)
        t_roc = expm(roc.get_ss_matrices()[0][:roc_n, :roc_n] / 60.0)
        x_roc = np.array([roc.state.c1, roc.state.c2, roc.state.c3][:roc_n], dtype=float)
        ce = self.ce

        sug, cplx = self.sugammadex, self.complex
        has_sug = (
            sug.state.c1 > 0.0
            or sug.state.c2 > 0.0
            or cplx.state.c1 > 0.0
            or cplx.state.c2 > 0.0
        )
        if not has_sug:
            exp_ke0 = math.exp(-ke0 * delivery_time)
            transition = np.zeros((roc_n + 1, roc_n + 1))
            transition[:roc_n, :roc_n] = t_roc
            transition[-1, :roc_n] = (1.0 - exp_ke0) * t_roc[0]
            transition[-1, -1] = exp_ke0
            x = np.empty(roc_n + 1)
            x[:roc_n] = x_roc
            x[-1] = ce
            block = min(128, max_seconds)
            response = np.empty((block, roc_n + 1))
            fill_response_rows(response, transition, -1)
            advance = np.linalg.matrix_power(transition, block)
            for start in range(0, max_seconds, block):
                count = min(block, max_seconds - start)
                crossed = np.flatnonzero(response[:count] @ x <= target)
                if crossed.size:
                    return (start + int(crossed[0]) + 1) / 60.0
                x = advance @ x
            return max_seconds / 60.0

        t_sug = expm(sug.get_ss_matrices()[0][:2, :2] / 60.0)
        t_cplx = expm(cplx.get_ss_matrices()[0][:2, :2] / 60.0)
        x_sug = np.array([sug.state.c1, sug.state.c2], dtype=float)
        x_cplx = np.array([cplx.state.c1, cplx.state.c2], dtype=float)
        v_roc, v_sug = roc.v1, sug.v1
        dt_min = 1.0 / 60.0
        ce_peri = self._ce_peri
        sug_given = self._sugammadex_given
        for second in range(1, max_seconds + 1):
            x_roc = t_roc @ x_roc
            r_c1, s_c1, c_c1 = float(x_roc[0]), float(x_sug[0]), float(x_cplx[0])
            if s_c1 > 0.0 or c_c1 > 0.0:
                x_roc[0], x_sug[0], x_cplx[0] = self._step_plasma_binding(
                    r_c1, s_c1, c_c1, v_roc, v_sug, dt_min,
                )
            x_sug = t_sug @ x_sug
            x_cplx = t_cplx @ x_cplx
            sug_c1 = float(x_sug[0])
            removal = self._sugammadex_removal(sug_c1)
            roc_c2 = float(x_roc[1]) if roc_n >= 2 else 0.0
            ce_peri, back_flux = self._step_peri_and_backflux(
                ce, ce_peri, roc_c2, sug_c1, delivery_time, sug_given=sug_given, update_peri=roc_n >= 2,
            )
            ce = self._effect_site(ce, float(x_roc[0]), ke0, removal, delivery_time, back_flux=back_flux, peri_ce=ce_peri)
            if ce <= target:
                return second / 60.0
        return max_seconds / 60.0

    def _sugammadex_removal(self, s_c1: float) -> float:
        """Junctional rocuronium encapsulation rate (min^-1) for plasma sugammadex s_c1.

        Calibrated to Kleijn 2011 moderate/deep/immediate reversal, Pühringer 2008
        RSI dose-response (2-16 mg/kg at 3 min after 1.2 mg/kg rocuronium), and
        Eleveld 2007 sub-therapeutic rebound.
        """
        if s_c1 <= 0.0:
            return 0.0
        s2 = s_c1 * s_c1
        return 0.60 * self.ks * s_c1 * (s2 / (s2 + 13.0 * 13.0)) + 0.10 * s_c1 / (s_c1 + 0.35)

    def _step_peri_and_backflux(
        self,
        ce: float,
        ce_peri: float,
        roc_c2: float,
        s_c1: float,
        delivery_time: float,
        *,
        sug_given: bool,
        update_peri: bool = True,
    ) -> tuple[float, float]:
        """Advance the perijunctional rocuronium buffer and return (new_ce_peri, back_flux)."""
        k_peri_in = 0.12
        peri_rate = k_peri_in + 0.25 * self.ks * s_c1
        if update_peri and delivery_time > 0.0 and peri_rate > 0.0:
            peri_target = k_peri_in * roc_c2 / peri_rate
            ce_peri = peri_target + (ce_peri - peri_target) * math.exp(-peri_rate * delivery_time)
        back_flux = 0.0
        if sug_given and ce_peri > ce:
            rebound_gate = 1.0 / (1.0 + (s_c1 / 0.20) ** 2) if s_c1 > 0.0 else 1.0
            back_flux = 0.14 * rebound_gate
        return ce_peri, back_flux

    def step_recovery(
        self,
        dt_sec: float,
        roc: MammillaryPK,
        mac_sevo: float = 0.0,
        mac_n2o: float = 0.0,
        *,
        neostigmine_ce: float | None = None,
    ) -> float:
        """Advance binding and the effect sites after the rocuronium PK step; return TOF %."""
        if neostigmine_ce is not None:
            self.neostigmine_ce = max(0.0, float(neostigmine_ce))
        sugammadex, complex_ = self.sugammadex, self.complex
        cp = roc.state.c1
        sug_active = sugammadex.state.c1 > 0.0 or sugammadex.state.c2 > 0.0
        cplx_active = complex_.state.c1 > 0.0 or complex_.state.c2 > 0.0
        if cp == 0.0 and self.ce == 0.0 and self.ce_central == 0.0 and not sug_active and not cplx_active:
            return self.compute_tof_from_ce(0.0, mac_sevo, mac_n2o)
        dt_min = dt_sec / 60.0
        if sugammadex.state.c1 > 0.0 or complex_.state.c1 > 0.0:
            self._bind_in_plasma(dt_min, roc)
            cp = roc.state.c1
            sug_active = sug_active or sugammadex.state.c1 > 0.0
            cplx_active = cplx_active or complex_.state.c1 > 0.0
        if sug_active:
            sugammadex.step(dt_sec, 0.0)
        if cplx_active:
            complex_.step(dt_sec, 0.0)

        ke0_out = self._effective_ke0(mac_sevo)
        # Lowry 1998: sevoflurane lowers EC50 and prolongs recovery (Kleijn 2011)
        # while leaving 0.6 mg/kg bolus onset time essentially unchanged (0.96 vs 1.02 min).
        ke0_in = ke0_out if mac_sevo <= 0.0 else self._effective_ke0(0.15 * mac_sevo)
        s_c1 = sugammadex.state.c1
        # Rapid saturable junctional encapsulation plus sigmoidally gated Kleijn efflux
        # reproduces rapid partial reversal and redistribution rebound after
        # sub-therapeutic sugammadex doses (Eleveld 2007) while preserving
        # Kleijn 2011 2/4/16 mg/kg reversal times.
        removal = self._sugammadex_removal(s_c1)
        delivery_time = dt_min * roc.circulation
        self._ce_peri, back_flux = self._step_peri_and_backflux(
            self.ce, self._ce_peri, roc.state.c2, s_c1, delivery_time,
            sug_given=self._sugammadex_given,
        )
        ke0_peri = ke0_in if cp > self.ce else ke0_out
        self.ce = self._effect_site(self.ce, cp, ke0_peri, removal, delivery_time, back_flux=back_flux, peri_ce=self._ce_peri)
        ratio = self.central_ke0_ratio
        central_back = back_flux * ratio if self._sugammadex_given and self._ce_peri > self.ce_central else 0.0
        ke0_cent = ke0_in if cp > self.ce_central else ke0_out
        self.ce_central = self._effect_site(
            self.ce_central, cp, ke0_cent * ratio, removal * ratio, delivery_time,
            back_flux=central_back, peri_ce=self._ce_peri,
        )
        # Clear prior sugammadex rebound flag once drug, complex, and perijunctional
        # rocuronium have washed out so much later doses are unaffected.
        if (
            self._sugammadex_given
            and max(s_c1, sugammadex.state.c2, complex_.state.c1, complex_.state.c2) < 1e-3
            and max(cp, self.ce, self.ce_central, self._ce_peri) < 0.05
        ):
            self._sugammadex_given = False
        return self.compute_tof_from_ce(self.ce, mac_sevo, mac_n2o)

    @staticmethod
    def _effect_site(
        ce: float, cp: float, ke0: float, removal: float, dt_min: float,
        *, back_flux: float = 0.0, peri_ce: float = 0.0,
    ) -> float:
        """Exact step of dCe/dt = ke0 (Cp - Ce) - removal Ce + back_flux (peri - Ce)."""
        rate = ke0 + removal + back_flux
        if rate <= 0.0:
            return ce
        target = (ke0 * cp + back_flux * peri_ce) / rate
        return target + (ce - target) * math.exp(-rate * dt_min)

    @staticmethod
    def _step_plasma_binding(
        r_c1: float, s_c1: float, c_c1: float, v_roc: float, v_sug: float, dt_min: float,
    ) -> tuple[float, float, float]:
        """Association k1 Croc Csug and dissociation k2 Ccplx in the sugammadex central volume."""
        to_um = 1000.0 / MW_ROCURONIUM
        fastest = SUGAMMADEX_K1_PER_UM_MIN * max(s_c1 * v_sug / v_roc, r_c1 * to_um) + SUGAMMADEX_K2_PER_MIN
        substeps = max(1, math.ceil(dt_min * fastest / 0.05))
        h = dt_min / substeps
        for _ in range(substeps):
            free_roc_um = r_c1 * to_um
            rate = SUGAMMADEX_K1_PER_UM_MIN * free_roc_um * s_c1 - SUGAMMADEX_K2_PER_MIN * c_c1
            flux = rate * v_sug * h  # umol bound this substep
            flux = min(flux, free_roc_um * v_roc, s_c1 * v_sug)
            flux = max(flux, -c_c1 * v_sug)
            r_c1 -= flux / v_roc / to_um
            s_c1 -= flux / v_sug
            c_c1 += flux / v_sug
        return r_c1, s_c1, c_c1

    def _bind_in_plasma(self, dt_min: float, roc: MammillaryPK) -> None:
        sugammadex, complex_ = self.sugammadex, self.complex
        roc.state.c1, sugammadex.state.c1, complex_.state.c1 = self._step_plasma_binding(
            roc.state.c1, sugammadex.state.c1, complex_.state.c1, roc.v1, sugammadex.v1, dt_min,
        )

    def _neostigmine_occupancy(self, neostigmine_ce: float | None = None) -> float:
        ce = self.neostigmine_ce if neostigmine_ce is None else max(0.0, float(neostigmine_ce))
        if ce <= 1e-9:
            return 0.0
        return hill_function(ce, NEOSTIGMINE_REVERSAL_C50, NEOSTIGMINE_REVERSAL_GAMMA)

    def potentiation_factor(
        self, mac_sevo: float = 0.0, mac_n2o: float = 0.0, *, neostigmine_ce: float | None = None,
    ) -> float:
        """Relative NMBA EC50 multiplier under volatile, N2O, hypothermia, and neostigmine."""
        t = self._temp_c
        # Extrapolated from vecuronium (Caldwell 2000, 34.0-38.0 °C): C50 is unchanged
        # in mild hypothermia (prolongation comes from slower clearance and ke0).
        # Below 32.5 °C, calibrated from 25-28 °C hypothermic CPB data (Smeulers 1995).
        temp_ec50 = 1.0 - 0.35 * clamp01((32.5 - t) / 5.5) if t < 32.5 else 1.0
        sevo_mac = max(0.0, mac_sevo)
        sevo_eff = sevo_mac if sevo_mac <= 1.0 else 1.0 + 0.55 * (sevo_mac - 1.0) / (1.0 + 0.35 * (sevo_mac - 1.0))
        factor = (
            max(0.25, 1.0 - self.sevo_ec50_reduction * sevo_eff)
            * (1.0 - (1.0 - self.f_n2o) * clamp01(mac_n2o))
            * temp_ec50
        )
        base_pot = clamp(factor, 0.15, 1.0)
        neo_occ = self._neostigmine_occupancy(neostigmine_ce)
        if neo_occ > 0.0:
            return base_pot * (1.0 + NEOSTIGMINE_EC50_MAX_SHIFT * neo_occ)
        return base_pot

    def _ec50(self, mac_sevo: float, mac_n2o: float, *, neostigmine_ce: float | None = None) -> float:
        return ROCURONIUM_TOF_EC50 * self.potentiation_factor(mac_sevo, mac_n2o, neostigmine_ce=neostigmine_ce)

    def _base_ec50(self, mac_sevo: float, mac_n2o: float) -> float:
        return ROCURONIUM_TOF_EC50 * self.potentiation_factor(mac_sevo, mac_n2o, neostigmine_ce=0.0)

    def _unopposed_fraction(self, ce: float, mac_sevo: float, mac_n2o: float) -> float:
        """Fraction of postsynaptic nAChRs free of rocuronium (vulnerable to excess ACh)."""
        if ce <= 1e-9:
            return 1.0
        ratio = ce / max(1e-9, self._base_ec50(mac_sevo, mac_n2o))
        return 1.0 / (1.0 + (ratio / 0.35) ** 4.0)

    def compute_tof_from_ce(
        self, ce: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0, *, neostigmine_ce: float | None = None,
    ) -> float:
        """TOF ratio (%) at the adductor pollicis."""
        neo_occ = self._neostigmine_occupancy(neostigmine_ce)
        if ce <= 1e-9:
            roc_tof = 1.0
        else:
            roc_tof = 1.0 / (1.0 + (ce / self._ec50(mac_sevo, mac_n2o, neostigmine_ce=neostigmine_ce)) ** ROCURONIUM_TOF_HILL)
        if neo_occ > 0.0:
            # Excess ACh without residual competitive NMBA occupancy produces mild
            # desensitizing weakness (Herbstreit 2010; Eikermann 2003).
            unopposed = self._unopposed_fraction(ce, mac_sevo, mac_n2o)
            roc_tof *= 1.0 - 0.06 * neo_occ * unopposed
        return 100.0 * roc_tof

    def twitch_block(
        self, ce: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0, *, neostigmine_ce: float | None = None,
    ) -> float:
        """Fractional T1 depression at the adductor pollicis."""
        tof = self.compute_tof_from_ce(ce, mac_sevo, mac_n2o, neostigmine_ce=neostigmine_ce) / 100.0
        return 1.0 - tof ** (1.0 / TOF_FADE_EXPONENT)

    def pharyngeal_tone_fraction(self, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        """Closing-pressure reserve (Herbstreit 2009), using normalized TOF.

        Reserve loss was 54%, 37%, and 16% at normalized TOF 0.48, 0.79,
        and 0.98. Interpolation to baseline at TOF 1 and coupling to hypnotic
        collapse are calibrations. Separate recovery kinetics are unmodeled.
        Unopposed neostigmine after full recovery further reduces upper-airway
        dilator reserve (Herbstreit 2010).
        """
        tof = self.compute_tof_from_ce(self.ce, mac_sevo, mac_n2o) / 100.0
        if tof >= 0.98:
            reduction = 0.16 * (1.0 - tof) / 0.02
        elif tof >= 0.79:
            reduction = 0.16 + 0.21 * (0.98 - tof) / 0.19
        else:
            reduction = 0.37 + 0.17 * clamp01((0.79 - tof) / 0.31)
        tone = 1.0 - reduction
        neo_occ = self._neostigmine_occupancy()
        if neo_occ > 0.0:
            unopposed = self._unopposed_fraction(self.ce, mac_sevo, mac_n2o)
            tone *= 1.0 - 0.20 * neo_occ * unopposed
        return tone


__all__ = [
    "ROCURONIUM_T1_EC50",
    "TOFModel",
    "succinylcholine_laryngeal_spasm_factor",
    "succinylcholine_phase2_tof_fraction",
    "succinylcholine_twitch_fraction",
]
