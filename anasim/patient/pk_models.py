"""Linear mammillary pharmacokinetic models.

Sources are cited on each model and in docs/REFERENCES.md.

Units: volumes L, clearances L/min, ke0 1/min, inputs in model units per
second. Concentrations are µg/mL for propofol, etomidate, ketamine,
lidocaine, rocuronium, and esmolol, ng/mL for opioids, midazolam,
catecholamines, labetalol, and glycopyrrolate, and mU/L for vasopressin.
"""

import math
from dataclasses import dataclass
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


class MammillaryPK:
    """Central compartment with up to two peripheral compartments and effect sites.

    Mass is V1*c1 + V2*c2 + V3*c3. Hemodynamic scaling changes the central
    volume with blood volume and clearances with cardiac output, using per-model
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
    ):
        self.v1_base = v1
        self.v1 = v1
        self.cl1_base = cl1
        self.cl1_scale = 1.0
        self.basal_input_rate = basal_input_rate  # Model mass units per second.
        self.cl2_base = cl2
        self.cl3_base = cl3
        self.v2 = v2
        self.v3 = v3
        self.ke0 = ke0
        self.cl1_co_exponent = cl1_co_exponent
        self.distribution_co_exponent = distribution_co_exponent
        # The primary site is last so TCI and decay predictions target it.
        self.effect_sites = (*additional_effect_sites, ("ce", ke0))
        self.state_fields = (
            ("c1",) + (("c2",) if v2 > 0 else ()) + (("c3",) if v3 > 0 else ())
            + tuple(name for name, _ in self.effect_sites)
        )
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

    def update_hemodynamics(self, v_ratio: float, co_ratio: float) -> None:
        """Rescale effective V1 without creating or removing central drug mass."""
        new_v1 = self.v1_base * v_ratio
        self.state.c1 *= self.v1 / new_v1
        self.v1 = new_v1
        self.cl1 = self.cl1_base * co_ratio ** self.cl1_co_exponent
        distribution_scale = co_ratio ** self.distribution_co_exponent
        self.cl2 = self.cl2_base * distribution_scale
        self.cl3 = self.cl3_base * distribution_scale

    def step(self, dt_sec: float, input_rate_per_sec: float) -> PKState:
        """Advance with simultaneous Euler steps that conserve compartment transfers."""
        input_rate_per_sec += self.basal_input_rate
        s = self.state
        if not input_rate_per_sec and not (s.c1 or s.c2 or s.c3 or s.ce or s.ce_response or s.ce_resp):
            return s  # Skip drugs with no input or residual concentration.
        cl2 = self.cl2 if self.v2 > 0 else 0.0
        cl3 = self.cl3 if self.v3 > 0 else 0.0
        elimination_cl = self.elimination_clearance
        fastest_rate = max((elimination_cl + cl2 + cl3) / self.v1,
                           self.k21, self.k31, *(rate for _, rate in self.effect_sites))
        # At most one second and 2.5% turnover per interval, including after
        # blood-volume scaling. Clipping an overshoot would create drug mass.
        substeps = max(1, math.ceil(dt_sec * max(1.0, fastest_rate / 1.5)))
        dt_min = dt_sec / (60.0 * substeps)
        for _ in range(substeps):
            flux2 = cl2 * (s.c1 - s.c2)
            flux3 = cl3 * (s.c1 - s.c3)
            elimination = elimination_cl * s.c1
            for name, rate in self.effect_sites:
                ce = getattr(s, name)
                setattr(s, name, ce + rate * (s.c1 - ce) * dt_min)
            s.c1 += (input_rate_per_sec * 60.0 - elimination - flux2 - flux3) / self.v1 * dt_min
            if self.v2 > 0:
                s.c2 += flux2 / self.v2 * dt_min
            if self.v3 > 0:
                s.c3 += flux3 / self.v3 * dt_min
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
        for i, (_, rate) in enumerate(self.effect_sites, start=len(peripherals) + 1):
            A[i, 0] = rate
            A[i, i] = -rate
        B[0, 0] = 1.0 / self.v1
        return A, B

    def state_vector(self) -> np.ndarray:
        return np.array([getattr(self.state, name) for name in self.state_fields])

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


def _organ_function(patient: Patient, attr: str) -> float:
    return clamp(float(getattr(patient, attr)), 0.1, 1.0)


def organ_clearance_scaler(patient: Patient, hepatic_fraction: float = 0.0, renal_fraction: float = 0.0) -> float:
    """Weighted clearance scaler; clearance outside the named fractions is unaffected."""
    hepatic = _organ_function(patient, "hepatic_function")
    renal = _organ_function(patient, "renal_function")
    other = max(0.0, 1.0 - hepatic_fraction - renal_fraction)
    return hepatic_fraction * hepatic + renal_fraction * renal + other


def hepatic_vd_multiplier(patient: Patient, severe_multiplier: float) -> float:
    """Scale volumes with hepatic impairment; severe impairment is hepatic_function 0.5."""
    severity = clamp((1.0 - _organ_function(patient, "hepatic_function")) / 0.5, 0.0, 1.0)
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

        pma, pma_ref = age * 52 + 40, age_ref * 52 + 40
        cl_maturation = _sigmoid(pma, 42.2760190602615, 9.0548452392807) / _sigmoid(pma_ref, 42.2760190602615, 9.0548452392807)
        q3_maturation = _sigmoid(pma, 68.2767978846832, 1) / _sigmoid(pma_ref, 68.2767978846832, 1)
        v2_ref, v3_ref = 25.5013145036879, 272.8166615043603

        v1 = 6.2830780766822 * _sigmoid(w, 33.5531248778544, 1) / _sigmoid(w_ref, 33.5531248778544, 1)
        v2 = v2_ref * (w / w_ref) * np.exp(-0.015633 * (age - age_ref))
        v3 = v3_ref * _eleveld_ffm(age, w, patient.height, male) / _eleveld_ffm(35.0, 70.0, 170.0, True)
        if concomitant_opioids:
            v3 *= math.exp(-0.0138166 * age)
        cl_ref = 1.7895836588902 if male else 2.1002218877899
        if concomitant_opioids:
            cl_ref *= math.exp(-0.00285709 * age)
        params = {
            "v1": v1,
            "cl1": cl_ref * (w / w_ref) ** 0.75 * cl_maturation,
            "v2": v2,
            "cl2": 1.7500983738779 * (v2 / v2_ref) ** 0.75 * (1.0 + 1.3042680471360 * (1.0 - _sigmoid(pma, 68.2767978846832, 1))),
            "v3": v3,
            "cl3": 1.1085424008536 * (v3 / v3_ref) ** 0.75 * q3_maturation,
        }
        params = _scale_volumes(params, hepatic_vd_multiplier(patient, severe_multiplier=1.6))
        # The clinical response rate is a timing assumption (Kuizenga 2019 study
        # design); respiratory equilibration half-time is 2.6 min (Bouillon 2004).
        super().__init__(
            ke0=0.14636965112551278 * (w / 70.0) ** -0.25,
            additional_effect_sites=(("ce_response", 0.456), ("ce_resp", math.log(2.0) / 2.6)),
            **_NO_CO_COVARIATE, **params,
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
    about a minute.
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
    """Wierda et al. 1991 with the Masui age-dependent ke0."""

    def __init__(self, patient: Patient):
        params = _from_rate_constants(0.044 * patient.weight, 0.100, 0.210, 0.130, 0.028, 0.010)
        params = _scale_volumes(params, hepatic_vd_multiplier(patient, severe_multiplier=1.4))
        params["cl1"] *= organ_clearance_scaler(patient, renal_fraction=0.6)
        super().__init__(ke0=max(0.01, 0.100 - 0.00138 * (patient.age - 50.0)), **params)


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
        cl1 = 0.13 * w / 60.0 * organ_clearance_scaler(patient, renal_fraction=0.9)
        super().__init__(v1=0.45 * w, cl1=cl1, ke0=0.12)


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
        super().__init__(
            v1=10.38 * size,
            cl1=49.76 / 60.0 * size**0.75 * organ_clearance_scaler(patient, renal_fraction=0.7),
            v2=21.41 * size,
            cl2=6.54 / 60.0 * size**0.75,
            v3=11.0 * size,
            cl3=31.62 / 60.0 * size**0.75,
            ke0=0.4,
        )
