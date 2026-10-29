"""Linear mammillary pharmacokinetic models.

Sources are cited on each model and in docs/REFERENCES.md.

Units: volumes L, clearances L/min, ke0 1/min, inputs in model units per
second. Concentrations are µg/mL for propofol, etomidate, ketamine,
lidocaine, rocuronium, succinylcholine, and esmolol, ng/mL for opioids,
midazolam, catecholamines, ephedrine, atropine, labetalol, and glycopyrrolate,
and mU/L for vasopressin.
"""

import math
from dataclasses import dataclass
from operator import attrgetter
from typing import TypedDict

import numpy as np
from scipy.linalg import expm

from anasim.core.utils import clamp

from .patient import Patient


@dataclass
class PKState:
    """Compartment concentrations. Unused peripheral compartments stay at zero."""

    c1: float = 0.0  # Central (plasma)
    c2: float = 0.0  # Fast peripheral
    c3: float = 0.0  # Slow peripheral
    ce: float = 0.0  # Effect site
    ce_response: float = 0.0  # Propofol clinical response
    ce_resp: float = 0.0  # Respiratory depression
    ce_rr: float = 0.0  # Morphine/hydromorphone respiratory rate
    ce_cv: float = 0.0  # Ketamine cardiac-output effect
    ce_hr: float = 0.0  # Ketamine chronotropy
    ce_transit: float = 0.0  # Transit before the primary effect site


class MammillaryPK:
    """Central compartment with up to two peripheral compartments and effect sites.

    Mass is V1*c1 + V2*c2 + V3*c3. Hemodynamic scaling changes the vascular
    part of the central volume with blood volume, renal clearance with renal
    perfusion, and other clearances with cardiac output using per-model
    exponents. Peripheral volumes stay fixed so redistribution conserves mass.
    """

    def __init__(
        self,
        v1: float,
        cl1: float,
        ke0: float,
        v2: float = 0.0,
        cl2: float = 0.0,
        v3: float = 0.0,
        cl3: float = 0.0,
        cl1_co_exponent: float = 1.0,
        distribution_co_exponent: float = 1.0,
        additional_effect_sites: tuple[tuple[str, float], ...] = (),
        basal_input_rate: float = 0.0,
        renal_fraction: float = 0.0,
        cl1_deficit_exponent: float = 0.0,
        distribution_deficit_exponent: float = 0.0,
        effect_site_transit_rate: float | None = None,
    ):
        self.v1_base = v1
        self.v1 = v1
        self.cl1_base = cl1
        self.cl1_scale = 1.0
        self.v1_scale = 1.0
        self.cl2_scale = self.cl3_scale = 1.0
        self.basal_input_rate = basal_input_rate  # Model mass units per second.
        self.cl2_base = cl2
        self.cl3_base = cl3
        self.v2 = v2
        self.v3 = v3
        self.ke0 = ke0
        self.cl1_co_exponent = cl1_co_exponent
        self.renal_fraction = renal_fraction
        # Exponents on the blood-volume ratio, applied only below baseline volume.
        self.cl1_deficit_exponent = cl1_deficit_exponent
        self.distribution_deficit_exponent = distribution_deficit_exponent
        self.distribution_co_exponent = distribution_co_exponent
        self.ce_input = "ce_transit" if effect_site_transit_rate is not None else "c1"
        transit = (("ce_transit", effect_site_transit_rate),) if effect_site_transit_rate is not None else ()
        # The primary site is last so TCI and decay predictions target it.
        self.effect_sites = (*transit, *additional_effect_sites, ("ce", ke0))
        # Downstream sites advance first so every site reads the old source.
        self._effect_updates = tuple(
            (name, rate, self.ce_input if name == "ce" else "c1")
            for name, rate in reversed(self.effect_sites)
        )
        self._max_effect_rate = max(rate for _, rate in self.effect_sites)
        self.state_fields = (
            ("c1",) + (("c2",) if v2 > 0 else ()) + (("c3",) if v3 > 0 else ())
            + tuple(name for name, _ in self.effect_sites)
        )
        self._state_values = attrgetter(*self.state_fields)
        self.state = PKState()
        self.update_hemodynamics(1.0, 1.0)

    @property
    def k10(self) -> float:
        return self.elimination_clearance / self.v1

    @property
    def elimination_clearance(self) -> float:
        return self.cl1 * self.cl1_scale

    @property
    def k12(self) -> float:
        return self.cl2 / self.v1

    @property
    def k21(self) -> float:
        return self.cl2 / self.v2 if self.v2 > 0 else 0.0

    @property
    def k13(self) -> float:
        return self.cl3 / self.v1

    @property
    def k31(self) -> float:
        return self.cl3 / self.v3 if self.v3 > 0 else 0.0

    def update_hemodynamics(
        self, v_ratio: float, co_ratio: float, renal_perfusion: float = 1.0,
        blood_volume_l: float | None = None,
    ) -> None:
        """Rescale effective V1 without creating or removing central drug mass.

        Only the part of V1 up to the baseline blood volume changes with blood
        volume; the rest is tissue. Without blood_volume_l, all of V1 does.
        """
        vascular = self.v1_base if blood_volume_l is None else min(self.v1_base, blood_volume_l)
        new_v1 = (self.v1_base + vascular * (v_ratio - 1.0)) * self.v1_scale
        self.state.c1 *= self.v1 / new_v1
        self.v1 = new_v1
        deficit = min(1.0, v_ratio)
        nonrenal = (1.0 - self.renal_fraction) * co_ratio ** self.cl1_co_exponent
        self.cl1 = self.cl1_base * (nonrenal + self.renal_fraction * renal_perfusion)
        self.cl1 *= deficit ** self.cl1_deficit_exponent
        distribution_scale = co_ratio ** self.distribution_co_exponent
        distribution_scale *= deficit ** self.distribution_deficit_exponent
        self.cl2 = self.cl2_base * distribution_scale * self.cl2_scale
        self.cl3 = self.cl3_base * distribution_scale * self.cl3_scale

    def _set_interaction_scales(
        self, *, v1: float = 1.0, cl1: float = 1.0, cl2: float = 1.0, cl3: float = 1.0,
    ) -> bool:
        """Apply drug covariates without changing central mass or losing flow scaling."""
        scales = (v1, cl1, cl2, cl3)
        old = (self.v1_scale, self.cl1_scale, self.cl2_scale, self.cl3_scale)
        if scales == old or all(math.isclose(a, b, rel_tol=1e-3) for a, b in zip(scales, old, strict=True)):
            return False
        self.state.c1 *= self.v1_scale / v1
        self.v1 *= v1 / self.v1_scale
        self.cl2 *= cl2 / self.cl2_scale
        self.cl3 *= cl3 / self.cl3_scale
        self.v1_scale, self.cl1_scale, self.cl2_scale, self.cl3_scale = scales
        return True

    def step(self, dt_sec: float, input_rate_per_sec: float) -> PKState:
        """Advance with simultaneous Euler steps that conserve compartment transfers."""
        input_rate_per_sec += self.basal_input_rate
        s = self.state
        if not input_rate_per_sec and not any(self._state_values(s)):
            return s  # Skip drugs with no input or residual concentration.
        v1, v2, v3 = self.v1, self.v2, self.v3
        cl2 = self.cl2 if v2 > 0 else 0.0
        cl3 = self.cl3 if v3 > 0 else 0.0
        elimination_cl = self.elimination_clearance
        fastest_rate = max((elimination_cl + cl2 + cl3) / v1,
                           cl2 / v2 if v2 > 0 else 0.0, cl3 / v3 if v3 > 0 else 0.0,
                           self._max_effect_rate)
        # At most one second and 2.5% turnover per interval, including after
        # blood-volume scaling. Clipping an overshoot would create drug mass.
        substeps = max(1, math.ceil(dt_sec * max(1.0, fastest_rate / 1.5)))
        dt_min = dt_sec / (60.0 * substeps)
        input_per_min = input_rate_per_sec * 60.0
        for _ in range(substeps):
            flux2 = cl2 * (s.c1 - s.c2)
            flux3 = cl3 * (s.c1 - s.c3)
            elimination = elimination_cl * s.c1
            for name, rate, source_name in self._effect_updates:
                ce = getattr(s, name)
                source = getattr(s, source_name)
                setattr(s, name, ce + rate * (source - ce) * dt_min)
            s.c1 += (input_per_min - elimination - flux2 - flux3) / v1 * dt_min
            if v2 > 0:
                s.c2 += flux2 / v2 * dt_min
            if v3 > 0:
                s.c3 += flux3 / v3 * dt_min
        return s

    def get_ss_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return continuous A (1/min) and B for the state order in `state_fields`."""
        peripherals = [(v, cl) for v, cl in ((self.v2, self.cl2), (self.v3, self.cl3)) if v > 0]
        n = len(peripherals) + 1 + len(self.effect_sites)
        A = np.zeros((n, n))
        B = np.zeros((n, 1))
        A[0, 0] = -(self.elimination_clearance + sum(cl for _, cl in peripherals)) / self.v1
        for i, (v, cl) in enumerate(peripherals, start=1):
            A[0, i] = cl / self.v1
            A[i, 0] = cl / v
            A[i, i] = -cl / v
        for i, (name, rate) in enumerate(self.effect_sites, start=len(peripherals) + 1):
            source = self.state_fields.index(self.ce_input) if name == "ce" else 0
            A[i, source] = rate
            A[i, i] = -rate
        B[0, 0] = 1.0 / self.v1
        return A, B

    def state_vector(self) -> np.ndarray:
        return np.array(self._state_values(self.state))

    def set_state_vector(self, values) -> None:
        for name, value in zip(self.state_fields, values, strict=True):
            setattr(self.state, name, max(0.0, float(value)))

    def reset(self) -> None:
        self.state = PKState()

    def simulate_decay(self, target_fraction: float = 0.5, max_seconds: int = 3600) -> float:
        """Return minutes for Ce to fall to `target_fraction` with no further input."""
        ce_target = self.state.ce * target_fraction
        if ce_target <= 0.0:
            return 0.0
        A, B = self.get_ss_matrices()
        transition = expm(A / 60.0)
        equilibrium = (np.linalg.solve(-A, B[:, 0] * self.basal_input_rate * 60.0)
                       if self.basal_input_rate else np.zeros(A.shape[0]))
        x = self.state_vector()
        for second in range(1, max_seconds + 1):
            x = transition @ (x - equilibrium) + equilibrium
            if x[-1] <= ce_target:
                return second / 60.0
        return max_seconds / 60.0


def _from_rate_constants(v1, k10, k12, k21, k13, k31):
    """Convert published micro-rate constants to volumes and clearances."""
    cl2 = k12 * v1
    cl3 = k13 * v1
    return {
        "v1": v1,
        "cl1": k10 * v1,
        "v2": cl2 / k21 if k21 > 0 else 0.0,
        "cl2": cl2,
        "v3": cl3 / k31 if k31 > 0 else 0.0,
        "cl3": cl3,
    }


def organ_clearance_scaler(patient: Patient, hepatic_fraction: float = 0.0, renal_fraction: float = 0.0) -> float:
    """Weighted clearance scaler; clearance outside the named fractions is unaffected."""
    hepatic = patient.hepatic_function
    renal = patient.renal_clearance_factor
    other = max(0.0, 1.0 - hepatic_fraction - renal_fraction)
    return hepatic_fraction * hepatic + renal_fraction * renal + other


def renal_capacity_scaling(patient: Patient, renal_fraction: float) -> tuple[float, float]:
    """Clearance scale for renal capacity, and the renal share of the scaled clearance.

    Renal perfusion acts only on the renal share left after the capacity change.
    """
    scale = organ_clearance_scaler(patient, renal_fraction=renal_fraction)
    return scale, renal_fraction * patient.renal_clearance_factor / scale


def hepatic_vd_multiplier(patient: Patient, severe_multiplier: float) -> float:
    """Scale volumes with hepatic impairment; severe impairment is hepatic_function 0.5."""
    severity = (1.0 - patient.hepatic_function) / 0.5
    return 1.0 + (severe_multiplier - 1.0) * severity


def _scale_volumes(params: dict, factor: float) -> dict:
    """Expand distribution volumes while preserving clearances."""
    return {key: value * factor if key.startswith("v") else value for key, value in params.items()}


class _CoExponents(TypedDict):
    cl1_co_exponent: float
    distribution_co_exponent: float


# Population parameters without an additional cardiac-output covariate.
_NO_CO_COVARIATE: _CoExponents = {"cl1_co_exponent": 0.0, "distribution_co_exponent": 0.0}


def _sigmoid(x: float, x50: float, gamma: float) -> float:
    return x**gamma / (x50**gamma + x**gamma)


def _eleveld_ffm(age: float, weight: float, height: float, male: bool) -> float:
    """Al-Sallami fat-free mass used by both Eleveld models."""
    bmi = weight / (height / 100.0) ** 2
    if male:
        maturation = 0.88 + 0.12 / (1.0 + (age / 13.4) ** -12.7)
        return maturation * 9270.0 * weight / (6680.0 + 216.0 * bmi)
    maturation = 1.11 - 0.11 / (1.0 + (age / 7.1) ** -1.1)
    return maturation * 9270.0 * weight / (8780.0 + 244.0 * bmi)


# Propofol ---------------------------------------------------------------------

class PropofolPKEleveld(MammillaryPK):
    """Eleveld 2018 arterial PK and BIS effect site, with separate clinical PD sites."""

    def __init__(self, patient: Patient, concomitant_opioids: bool = True):
        age, w = patient.age, patient.weight
        male = patient.sex == "male"
        age_ref, w_ref = 35.0, 70.0
        eta = patient.variability.propofol

        pma, pma_ref = age * 52 + 40, age_ref * 52 + 40
        cl_maturation = _sigmoid(pma, 42.2760190602615, 9.0548452392807) / _sigmoid(pma_ref, 42.2760190602615, 9.0548452392807)
        q3_maturation = _sigmoid(pma, 68.2767978846832, 1) / _sigmoid(pma_ref, 68.2767978846832, 1)
        v2_ref, v3_ref = 25.5013145036879, 272.8166615043603

        v1 = 6.2830780766822 * _sigmoid(w, 33.5531248778544, 1) / _sigmoid(w_ref, 33.5531248778544, 1) * eta.v1
        v2 = v2_ref * (w / w_ref) * math.exp(-0.015633 * (age - age_ref)) * eta.v2
        v3 = v3_ref * _eleveld_ffm(age, w, patient.height, male) / _eleveld_ffm(35.0, 70.0, 170.0, True) * eta.v3
        if concomitant_opioids:
            v3 *= math.exp(-0.0138166 * age)
        cl_ref = 1.7895836588902 if male else 2.1002218877899
        if concomitant_opioids:
            cl_ref *= math.exp(-0.00285709 * age)
        params = {
            "v1": v1,
            "cl1": cl_ref * (w / w_ref) ** 0.75 * cl_maturation * eta.cl,
            "v2": v2,
            "cl2": 1.7500983738779 * (v2 / v2_ref) ** 0.75 * (1.0 + 1.3042680471360 * (1.0 - _sigmoid(pma, 68.2767978846832, 1)))
            * eta.q2,
            "v3": v3,
            "cl3": 1.1085424008536 * (v3 / v3_ref) ** 0.75 * q3_maturation * eta.q3,
        }
        params = _scale_volumes(params, hepatic_vd_multiplier(patient, severe_multiplier=1.6))
        # The clinical response rate is a timing assumption (Kuizenga 2019 study
        # design); respiratory equilibration half-time is 2.6 min (Bouillon 2004).
        # Eleveld's anesthetized cohort already reflects propofol's fall in cardiac
        # output, so blood loss scales clearance instead. In swine bled 30 mL/kg
        # (about 43%), clearance halved and distribution clearances fell 52-78%,
        # doubling plasma propofol; shed blood volume predicted the change best
        # (Johnson 2003). Crystalloid resuscitation restored the kinetics as
        # volume returned (Johnson 2004).
        super().__init__(
            ke0=0.14636965112551278 * (w / 70.0) ** -0.25 * eta.ke0,
            additional_effect_sites=(("ce_response", 0.456), ("ce_resp", math.log(2.0) / 2.6)),
            cl1_deficit_exponent=1.0, distribution_deficit_exponent=2.0,
            **_NO_CO_COVARIATE, **params,
        )

    def update_midazolam(self, concentration_ng_ml: float) -> bool:
        """Vuyk 2009 Table 2, midazolam-only clearance covariates.

        Apply relative changes to Eleveld, centered on no midazolam, and bound
        the covariate at the studied 225 ng/mL. This transfer is an assumption.
        """
        cp = clamp(concentration_ng_ml, 0.0, 225.0)
        return self._set_interaction_scales(
            cl1=math.exp(-0.000820 * cp),
            cl2=math.exp(-0.00274 * cp),
            cl3=math.exp(-0.00142 * cp),
        )


# Remifentanil -----------------------------------------------------------------

class RemifentanilPKEleveld(MammillaryPK):
    """Eleveld 2017 arterial PK, SEF effect site, and Olofsen 2010 respiratory site."""

    def __init__(self, patient: Patient):
        age, weight = patient.age, patient.weight
        size = _eleveld_ffm(age, weight, patient.height, patient.sex == "male") / _eleveld_ffm(35.0, 70.0, 170.0, True)
        aging_fast = math.exp(-0.00554 * (age - 35.0))
        aging_slow = math.exp(-0.00327 * (age - 35.0))
        sex_factor = 1.0
        if patient.sex == "female":
            sex_factor += 0.470 * _sigmoid(age, 12.0, 6.0) * (1.0 - _sigmoid(age, 45.0, 6.0))
        v2 = 8.82 * size * aging_slow * sex_factor
        v3 = 5.03 * size * math.exp(-0.0315 * (age - 35.0) - 0.0260 * (weight - 70.0))
        super().__init__(
            v1=5.81 * size * aging_fast,
            v2=v2,
            v3=v3,
            cl1=2.58 * size**0.75 * _sigmoid(weight, 2.88, 2.0) / _sigmoid(70.0, 2.88, 2.0) * sex_factor * aging_slow,
            cl2=1.72 * (v2 / 8.82)**0.75 * aging_fast * sex_factor,
            cl3=0.124 * (v3 / 5.03)**0.75 * aging_fast,
            ke0=1.09 * math.exp(-0.0289 * (age - 35.0)),
            additional_effect_sites=(("ce_resp", math.log(2.0) / 0.53),),
            # Swine bled 48 mL/kg had 58% lower central clearance and half the
            # V1, with unchanged distribution clearances (Johnson 2001).
            cl1_deficit_exponent=0.8,
            **_NO_CO_COVARIATE,
        )


# Fentanyl, midazolam, etomidate, ketamine, and lidocaine ----------------------

class FentanylPK(MammillaryPK):
    """Bae et al. 2020 allometric three-compartment model (ng/mL).

    Volumes scale with (weight/70)^1.23 and clearances with (weight/70)^0.313;
    ke0 0.147 min^-1 is from Scott and Stanski 1987.
    """

    def __init__(self, patient: Patient):
        volume = (patient.weight / 70.0) ** 1.23
        flow = (patient.weight / 70.0) ** 0.313
        super().__init__(
            v1=10.1 * volume,
            cl1=0.704 * flow,
            v2=26.5 * volume,
            cl2=2.38 * flow,
            v3=206.0 * volume,
            cl3=1.49 * flow,
            ke0=0.147,
        )


class MorphinePK(MammillaryPK):
    """Meissner 2025 Table 1 PK and Olofsen 2026 effect sites, ng/mL (input mcg).

    Olofsen's shared parameters are divided by each drug factor for morphine
    and multiplied by it for hydromorphone.
    """

    def __init__(self, patient: Patient):
        params = _from_rate_constants(20.0, 5.1 / 60, 4.3 / 60, 1.3 / 60, 1.2 / 60, 0.306 / 60)
        super().__init__(
            **params, **_NO_CO_COVARIATE,
            ke0=math.log(2.0) / (0.73 / 1.02 * 60.0),
            additional_effect_sites=(
                ("ce_resp", math.log(2.0) / (3.3 / 0.694 * 60.0)),
                ("ce_rr", math.log(2.0) / (2.2 / 0.736 * 60.0)),
            ),
        )


class HydromorphonePK(MammillaryPK):
    """Meissner 2025 Table 1 PK with its sex factor and Olofsen 2026 sites, ng/mL."""

    def __init__(self, patient: Patient):
        v1 = 13.6 / 0.908 if patient.sex == "male" else 13.6 * 0.908
        params = _from_rate_constants(v1, 6.1 / 60, 6.6 / 60, 1.6 / 60, 1.6 / 60, 0.41 / 60)
        super().__init__(
            **params, **_NO_CO_COVARIATE,
            ke0=math.log(2.0) / (0.73 * 1.02 * 60.0),
            additional_effect_sites=(
                ("ce_resp", math.log(2.0) / (3.3 * 0.694 * 60.0)),
                ("ce_rr", math.log(2.0) / (2.2 * 0.736 * 60.0)),
            ),
        )


class AtropinePK(MammillaryPK):
    """Hinderling 1985 I Table I means (CL 1.175 L/min, V1 19.1 L, k12 0.56 and
    k21 0.0557 /min), in ng/mL.

    ke0 = k21 links HR to the peripheral compartment (part II). Weight scaling
    from the cohort's 72.3 kg mean and proportional renal-capacity scaling of
    the 56.8% renal fraction are simulator assumptions.
    """

    def __init__(self, patient: Patient):
        v1 = 19.1 * patient.weight / 72.3
        params = _from_rate_constants(v1, 1.175 / 19.1, 0.56, 0.0557, 0.0, 0.0)
        clearance_scale, renal_fraction = renal_capacity_scaling(patient, 0.568)
        params["cl1"] *= clearance_scale
        super().__init__(**params, ke0=0.0557, renal_fraction=renal_fraction, **_NO_CO_COVARIATE)


class EphedrinePK(MammillaryPK):
    """Persky 2004 apparent oral disposition, transferred to IV use (ng/mL).

    Bioavailability of one and weight-linear V/CL are assumptions.
    The one-minute effect-site time constant is calibrated.
    """

    def __init__(self, patient: Patient):
        super().__init__(
            v1=4.0 * patient.weight, cl1=0.47 * patient.weight / 60.0,
            ke0=1.0, **_NO_CO_COVARIATE,
        )


class SuccinylcholinePK(MammillaryPK):
    """Roy 2002 plasma PK with effect-site timing calibrated to Kopman 2003.

    Transit and effect-site rates (1/min) fit Table 2 twitch time courses at
    0.4-1 mg/kg. Concentrations are mcg/mL; hydrolysis has no organ/flow covariate.
    """

    def __init__(self, patient: Patient):
        params = _from_rate_constants(0.009 * patient.weight, 5.0, 2.9, 1.6, 0.0, 0.0)
        super().__init__(**params, ke0=0.4, effect_site_transit_rate=1.3, **_NO_CO_COVARIATE)


class MidazolamPK(MammillaryPK):
    """Albrecht et al. 1999 three-compartment model (ng/mL).

    Young volunteers (24-28 y, 66-89 kg) gave Vc 7.9 L and CL 399 mL/min;
    k12 fell from 0.19 to 0.10 min^-1 by 71 years, and ke0 from 0.11 to 0.08
    min^-1. Volumes scale with weight from 78 kg; clearance is mostly hepatic.
    """

    def __init__(self, patient: Patient):
        size = patient.weight / 78.0
        years = clamp((patient.age - 26.0) / 45.0, 0.0, 1.0)
        v1 = 7.9 * size
        k12 = 0.19 - 0.09 * years
        super().__init__(
            v1=v1,
            cl1=0.399 * size**0.75 * organ_clearance_scaler(patient, hepatic_fraction=0.9),
            v2=v1 * k12 / 0.060,
            cl2=v1 * k12,
            v3=v1 * 0.065 / 0.0083,
            cl3=v1 * 0.065,
            ke0=0.11 - 0.03 * years,
        )

    def update_propofol(self, concentration_ug_ml: float) -> bool:
        """Lichtenbelt 2010 Table 2, propofol-only covariates on V1, CL and Q2.

        Apply relative changes to Albrecht, centered on no propofol, and bound
        the covariate at the studied 1.2 mcg/mL. This transfer is an assumption.
        """
        cp_ng_ml = 1000.0 * clamp(concentration_ug_ml, 0.0, 1.2)
        return self._set_interaction_scales(
            v1=math.exp(-0.000453 * cp_ng_ml),
            cl1=math.exp(-0.000205 * cp_ng_ml),
            cl2=math.exp(-0.000232 * cp_ng_ml),
        )


class EtomidatePK(MammillaryPK):
    """Arden et al. 1986 three-compartment whole-blood model (mcg/mL).

    At age 57: V1 0.090 L/kg, CL 20.3, Q2 25.5, and Q3 18.8 mL/kg/min. V1
    falls 42% from 22 to 80 years and CL about 2 mL/kg/min per decade.
    Peripheral volumes 0.259 and 4.36 L/kg reproduce the reported Vss (4.7
    L/kg) and half-lives (0.93, 12.1, and 324 min). t1/2 ke0 1.6 min.
    Clearance is mostly hepatic (Van Hamme 1978).
    """

    def __init__(self, patient: Patient):
        w, age = patient.weight, patient.age
        cl_ml_kg_min = max(8.0, 20.3 - 0.2 * (age - 56.7))
        super().__init__(
            v1=0.120 * (1.0 - 0.00724 * (age - 22.0)) * w,
            cl1=cl_ml_kg_min / 1000.0 * w * organ_clearance_scaler(patient, hepatic_fraction=0.9),
            v2=0.259 * w,
            cl2=0.0255 * w,
            v3=4.36 * w,
            cl3=0.0188 * w,
            ke0=0.433,
        )


class KetaminePK(MammillaryPK):
    """Kamp et al. 2020 meta-analytical three-compartment model (mcg/mL).

    CL 84, Q2 161, and Q3 79 L/h and V1 25, V2 56, and V3 157 L at 70 kg;
    clearance is hepatic. AnaSim sets ke0 for loss of consciousness within
    about a minute. The cardiac-output site uses Kamp 2021's
    equilibration half-time of 2.28 min; its transfer to total racemic
    ketamine concentrations is an assumption. The separate HR half-time of
    0.7 min is calibrated to Bovill 1972's peak within the first 3 min.
    """

    def __init__(self, patient: Patient):
        size = patient.weight / 70.0
        cl1 = 84.0 / 60.0 * size**0.75 * organ_clearance_scaler(patient, hepatic_fraction=0.9)
        super().__init__(
            v1=25.0 * size,
            cl1=cl1,
            v2=56.0 * size,
            cl2=161.0 / 60.0 * size**0.75,
            v3=157.0 * size,
            cl3=79.0 / 60.0 * size**0.75,
            ke0=0.5,
            additional_effect_sites=(("ce_cv", math.log(2.0) / 2.28), ("ce_hr", math.log(2.0) / 0.7)),
        )


class LidocainePK(MammillaryPK):
    """Foong et al. 2025 three-compartment model from surgical patients (mcg/mL).

    CL 45.9, Q2 142, and Q3 5.81 L/h and V1 25.2, V2 44.4, and V3 29.3 L,
    scaled from 70 kg; clearance is hepatic. AnaSim sets ke0 so the effect
    site peaks 3 min after a bolus, the best time to give it before
    intubation (Tam 1987).
    """

    def __init__(self, patient: Patient):
        size = patient.weight / 70.0
        super().__init__(
            v1=25.2 * size,
            cl1=45.9 / 60.0 * size**0.75 * organ_clearance_scaler(patient, hepatic_fraction=0.9),
            v2=44.4 * size,
            cl2=142.0 / 60.0 * size**0.75,
            v3=29.3 * size,
            cl3=5.81 / 60.0 * size**0.75,
            ke0=0.6,
        )


# Rocuronium -------------------------------------------------------------------

class RocuroniumPK(MammillaryPK):
    """Kleijn 2011 two-compartment population PK (446 subjects, 1-91 years).

    Clearance falls 0.68% per year of age from 43, V1 falls with creatinine
    clearance, and V2 rises with age. Of six rocuronium models, this one and
    Magorian's predicted measured concentrations within accepted limits
    (Sasakawa 2016); clearance at 65-88 years was 25% lower (Varrique 2016).
    Clearance is mainly hepatic uptake and biliary excretion: 12-22% of 0.6
    mg/kg appears unchanged in urine, renal failure changes clearance little,
    and cirrhosis enlarges the volumes 30-40% (Khuenl-Brady 1996). The effect
    site is Kleijn's, without sevoflurane.
    """

    RENAL_FRACTION = 0.2

    def __init__(self, patient: Patient):
        size = patient.weight / 70.0
        age = patient.age
        crcl = patient.creatinine_clearance()
        params = {
            "v1": 4.73 * size * math.exp(-0.00143 * (crcl - 119.0)),
            "cl1": 0.269 * size**0.75 * (1.0 - 0.00678 * (age - 43.0)),
            "v2": 6.76 * size * math.exp(0.00613 * (age - 43.0)),
            "cl2": 0.279 * size**0.75,
        }
        params = _scale_volumes(params, hepatic_vd_multiplier(patient, severe_multiplier=1.4))
        clearance_scale, renal_fraction = renal_capacity_scaling(patient, self.RENAL_FRACTION)
        params["cl1"] *= clearance_scale
        # Kleijn's patients were anesthetized, so his parameters already reflect
        # anesthetic cardiac output.
        super().__init__(
            ke0=0.134 * size**-0.25, renal_fraction=renal_fraction, **_NO_CO_COVARIATE, **params,
        )


# Vasoactive drugs -------------------------------------------------------------

class NorepinephrinePK(MammillaryPK):
    """Li 2024 arterial PK with secretion and propofol-dependent clearance.

    The clinical effect-site rate includes circulation and response timing.
    """

    # Li et al. 2024: clearance multiplier exp(theta * Cp / 100), about 12% lower at 3.5 µg/mL.
    PROPOFOL_CL_THETA = -3.57

    def __init__(self, patient: Patient):
        w, age = patient.weight, patient.age
        self.endogenous_ug_min = 0.4977 * (w / 70) ** 0.75
        super().__init__(
            v1=2.4 * (w / 70),
            # Li 2024 Eq. 2: the age covariate is centered on 35 years.
            cl1=2.1 * math.exp(-0.344 / 100 * (age - 35)) * (w / 70) ** 0.75,
            v2=3.6 * (w / 70),
            cl2=0.6 * (w / 70) ** 0.75,
            ke0=0.4,
            basal_input_rate=self.endogenous_ug_min / 60.0,
            **_NO_CO_COVARIATE,
        )
        self.equilibrate_endogenous()

    def equilibrate_endogenous(self) -> None:
        """Seed secretion without an exogenous infusion at the current clearance."""
        endogenous_conc = self.endogenous_ug_min / self.elimination_clearance
        self.set_state_vector([endogenous_conc] * len(self.state_fields))

    def update_propofol(self, propofol_conc_ug_ml: float) -> None:
        """Set the clearance covariate used by both integration and prediction."""
        self.cl1_scale = math.exp(self.PROPOFOL_CL_THETA * max(0.0, propofol_conc_ug_ml) / 100.0)


class EpinephrinePK(MammillaryPK):
    """Exogenous epinephrine, in ng/mL above the patient's endogenous baseline.

    Arterial infusion clearance is from Ensinger 1992. Mixing volume and
    effect-site timing are calibrated against adult infusion and bolus data.
    """

    def __init__(self, patient: Patient):
        w = patient.weight
        # Ensinger: 0.2 mcg/kg/min / (4.349 - 0.053) ng/mL.
        super().__init__(v1=0.035 * w, cl1=0.046 * w, ke0=2.2, **_NO_CO_COVARIATE)


class PhenylephrinePK(MammillaryPK):
    """FDA NDA 203826 two-compartment PK (t1/2 alpha 2.3 min, beta 53 min)."""

    def __init__(self, patient: Patient):
        params = _from_rate_constants(20.4 * patient.weight / 70.0, 0.124, 0.155, 0.0314, 0.0, 0.0)
        # Minimal hysteresis: bolus pressor effect peaks near 1 min.
        super().__init__(ke0=2.0, **params)


class VasopressinPK(MammillaryPK):
    """DailyMed label: Vd 0.14 L/kg, CL 9-25 mL/min/kg (midpoint 17)."""

    def __init__(self, patient: Patient):
        w = patient.weight
        super().__init__(v1=0.14 * w, cl1=0.017 * w, ke0=0.25, cl1_co_exponent=0.5)


class DobutaminePK(MammillaryPK):
    """Kates and Leier 1978: CL 2.35 L/min/m², Vd 0.20 L/kg, t1/2 about 2 min."""

    def __init__(self, patient: Patient):
        super().__init__(v1=0.20 * patient.weight, cl1=2.35 * patient.bsa, ke0=0.7, cl1_co_exponent=0.3)


class MilrinonePK(MammillaryPK):
    """DailyMed label: Vd 0.38-0.45 L/kg, CL 0.13 L/kg/h, mostly renal."""

    def __init__(self, patient: Patient):
        w = patient.weight
        clearance_scale, renal_fraction = renal_capacity_scaling(patient, 0.9)
        super().__init__(v1=0.45 * w, cl1=0.13 * w / 60.0 * clearance_scale, ke0=0.12, renal_fraction=renal_fraction)


# Adrenergic and muscarinic antagonists ----------------------------------------

class EsmololPK(MammillaryPK):
    """Sum et al. 1983 at 400 mcg/kg/min (concentrations in mcg/mL).

    Vc 0.867 L/kg, distribution and elimination t1/2 2.03 and 9.19 min, and
    k21/k12 2.66 give CL 258 mL/min/kg and Varea 3.42 L/kg. Blood esterases
    clear esmolol, so clearance is independent of cardiac output and organ
    function.
    """

    def __init__(self, patient: Patient):
        params = _from_rate_constants(0.867 * patient.weight, 0.2977, 0.0325, 0.0865, 0.0, 0.0)
        super().__init__(ke0=0.5, cl1_co_exponent=0.0, **params)


class LabetalolPK(MammillaryPK):
    """Two-compartment labetalol PK (concentrations in ng/mL).

    Clearance falls from 19.4 mL/min/kg at age 32 to 13.9 at 67 (Abernethy
    1987) and is mostly hepatic glucuronidation (Hafsa 2022). Distribution
    t1/2 is about 2 min, so 0.5 mg/kg gives about 130 ng/mL at 10 min (Hafsa
    2022), with terminal t1/2 about 3 h at age 32.
    """

    def __init__(self, patient: Patient):
        w = patient.weight
        cl_ml_min_kg = max(8.0, 19.4 - 0.157 * (patient.age - 32.0))
        cl1 = cl_ml_min_kg / 1000.0 * w * organ_clearance_scaler(patient, hepatic_fraction=0.9)
        super().__init__(v1=0.4 * w, cl1=cl1, v2=4.0 * w, cl2=0.1 * w, ke0=0.3)


class GlycopyrrolatePK(MammillaryPK):
    """Du et al. 2025 three-compartment model at 61.3 kg (concentrations in ng/mL).

    About 70% of a dose is excreted unchanged in urine (Ali-Melkkilä 1993).
    The published ke0 had 98.5% shrinkage; AnaSim uses 0.4 min^-1, which
    puts the HR peak a few minutes after a bolus.
    """

    def __init__(self, patient: Patient):
        size = patient.weight / 61.3
        clearance_scale, renal_fraction = renal_capacity_scaling(patient, 0.7)
        super().__init__(
            v1=10.38 * size,
            cl1=49.76 / 60.0 * size**0.75 * clearance_scale,
            renal_fraction=renal_fraction,
            v2=21.41 * size,
            cl2=6.54 / 60.0 * size**0.75,
            v3=11.0 * size,
            cl3=31.62 / 60.0 * size**0.75,
            ke0=0.4,
        )
