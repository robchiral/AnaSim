"""
Linear mammillary pharmacokinetic models.

References:
- Propofol: Marsh et al. Br J Anaesth. 1991; Schnider et al. Anesthesiology.
  1998; Eleveld et al. Br J Anaesth. 2018 (arterial fixed effects without
  opiate covariates).
- Remifentanil: Minto et al. Anesthesiology. 1997.
- Rocuronium: Wierda et al. Can J Anaesth. 1991, with the Masui age-dependent
  ke0.
- Norepinephrine: Beloeil et al. Br J Anaesth. 2005; Oualha et al. Br J Clin
  Pharmacol. 2014 (children); Li et al. Clin Pharmacokinet. 2024 (healthy
  volunteers, propofol covariate on clearance).
- Epinephrine: Clutter et al. J Clin Invest. 1980; Abboud et al. Crit Care.
  2009; Oualha et al. Br J Clin Pharmacol. 2014 (children).
- Phenylephrine: FDA NDA 203826 Clinical Pharmacology Review. 2012.
- Vasopressin, milrinone: DailyMed labels. Dobutamine: Kates and Leier. Clin
  Pharmacol Ther. 1978.

Units: volumes L, clearances L/min, ke0 1/min, inputs in model units per
second. Concentrations are µg/mL for propofol and rocuronium, ng/mL for
remifentanil and catecholamines, and mU/L for vasopressin.
"""

from dataclasses import dataclass

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


class MammillaryPK:
    """Central compartment with up to two peripheral compartments and an effect site.

    Mass is V1*c1 + V2*c2 + V3*c3. Hemodynamic scaling changes the central
    volume with blood volume and the clearances with cardiac output; peripheral
    tissue volumes stay fixed so redistribution conserves drug mass.
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
    ):
        self.v1_base = v1
        self.cl1_base = cl1
        self.cl2_base = cl2
        self.cl3_base = cl3
        self.v2 = v2
        self.v3 = v3
        self.ke0 = ke0
        self.cl1_co_exponent = cl1_co_exponent
        self.state_fields = (
            ("c1",) + (("c2",) if v2 > 0 else ()) + (("c3",) if v3 > 0 else ()) + ("ce",)
        )
        self.state = PKState()
        self.update_hemodynamics(1.0, 1.0)

    @property
    def k10(self) -> float:
        return self.cl1 / self.v1

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
        """Scale V1 by blood volume ratio and clearances by cardiac output ratio."""
        self.v1 = self.v1_base * v_ratio
        self.cl1 = self.cl1_base * co_ratio ** self.cl1_co_exponent
        self.cl2 = self.cl2_base * co_ratio
        self.cl3 = self.cl3_base * co_ratio

    def step(self, dt_sec: float, input_rate_per_sec: float, cl1_scale: float = 1.0) -> PKState:
        """Advance concentrations by one explicit Euler step."""
        s = self.state
        flux2 = self.cl2 * (s.c1 - s.c2) if self.v2 > 0 else 0.0
        flux3 = self.cl3 * (s.c1 - s.c3) if self.v3 > 0 else 0.0
        elimination = self.cl1 * cl1_scale * s.c1
        dt_min = dt_sec / 60.0
        dce = self.ke0 * (s.c1 - s.ce)
        s.c1 = max(0.0, s.c1 + (input_rate_per_sec * 60.0 - elimination - flux2 - flux3) / self.v1 * dt_min)
        if self.v2 > 0:
            s.c2 = max(0.0, s.c2 + flux2 / self.v2 * dt_min)
        if self.v3 > 0:
            s.c3 = max(0.0, s.c3 + flux3 / self.v3 * dt_min)
        s.ce = max(0.0, s.ce + dce * dt_min)
        return s

    def get_ss_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return continuous A (1/min) and B for the state order in `state_fields`."""
        peripherals = [(v, cl) for v, cl in ((self.v2, self.cl2), (self.v3, self.cl3)) if v > 0]
        n = len(peripherals) + 2
        A = np.zeros((n, n))
        B = np.zeros((n, 1))
        A[0, 0] = -(self.cl1 + sum(cl for _, cl in peripherals)) / self.v1
        for i, (v, cl) in enumerate(peripherals, start=1):
            A[0, i] = cl / self.v1
            A[i, 0] = cl / v
            A[i, i] = -cl / v
        A[-1, 0] = self.ke0
        A[-1, -1] = -self.ke0
        B[0, 0] = 1.0 / self.v1
        return A, B

    def state_vector(self) -> np.ndarray:
        return np.array([getattr(self.state, name) for name in self.state_fields])

    def set_state_vector(self, values) -> None:
        for name, value in zip(self.state_fields, values):
            setattr(self.state, name, max(0.0, float(value)))

    def reset(self) -> None:
        self.state = PKState()

    def simulate_decay(self, target_fraction: float = 0.5, max_seconds: int = 3600) -> float:
        """Return minutes for Ce to fall to `target_fraction` with no further input."""
        ce_target = self.state.ce * target_fraction
        if ce_target <= 0.0:
            return 0.0
        A, _ = self.get_ss_matrices()
        transition = expm(A / 60.0)
        x = self.state_vector()
        for second in range(1, max_seconds + 1):
            x = transition @ x
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


# Propofol ---------------------------------------------------------------------

class PropofolPKMarsh(MammillaryPK):
    """Marsh et al. 1991 with the modified effect-site ke0 of 1.2 min^-1."""

    def __init__(self, patient: Patient):
        w = patient.weight
        params = {
            "v1": 0.228 * w,
            "cl1": 0.119 * 0.228 * w,
            "v2": 0.463 * w,
            "cl2": 0.112 * 0.228 * w,
            "v3": 2.893 * w,
            "cl3": 0.042 * 0.228 * w,
        }
        params = _scale_volumes(params, hepatic_vd_multiplier(patient, severe_multiplier=1.6))
        super().__init__(ke0=1.2, **params)


class PropofolPKSchnider(MammillaryPK):
    """Schnider et al. 1998 with its James lean-body-mass covariate."""

    def __init__(self, patient: Patient):
        age, w, h, lbm = patient.age, patient.weight, patient.height, patient.james_lbm()
        params = {
            "v1": 4.27,
            "cl1": max(0.01, 1.89 + 0.0456 * (w - 77) - 0.0681 * (lbm - 59) + 0.0264 * (h - 177)),
            "v2": max(0.5, 18.9 - 0.391 * (age - 53)),
            "cl2": max(0.01, 1.29 - 0.024 * (age - 53)),
            "v3": 238.0,
            "cl3": 0.836,
        }
        params = _scale_volumes(params, hepatic_vd_multiplier(patient, severe_multiplier=1.6))
        super().__init__(ke0=0.456, **params)


class PropofolPKEleveld(MammillaryPK):
    """Eleveld et al. 2018 arterial fixed effects without opiate covariates."""

    def __init__(self, patient: Patient):
        age = patient.age
        w = patient.weight
        male = patient.sex == "male"
        bmi = w / (patient.height / 100.0) ** 2
        age_ref, w_ref = 35.0, 70.0
        bmi_ref = w_ref / 1.7**2

        def sigmoid(x, x50, gamma):
            return x**gamma / (x50**gamma + x**gamma)

        def fat_free_mass(is_male, weight, years, body_mass_index):
            if is_male:
                return (0.88 + 0.12 / (1 + (years / 13.4) ** -12.7)) * 9270 * weight / (6680 + 216 * body_mass_index)
            return (1.11 - 0.11 / (1 + (years / 7.1) ** -1.1)) * 9270 * weight / (8780 + 244 * body_mass_index)

        pma, pma_ref = age * 52 + 40, age_ref * 52 + 40
        cl_maturation = sigmoid(pma, 42.2760190602615, 9.0548452392807) / sigmoid(pma_ref, 42.2760190602615, 9.0548452392807)
        q3_maturation = sigmoid(pma, 68.2767978846832, 1) / sigmoid(pma_ref, 68.2767978846832, 1)
        v2_ref, v3_ref = 25.5013145036879, 272.8166615043603

        v1 = 6.2830780766822 * sigmoid(w, 33.5531248778544, 1) / sigmoid(w_ref, 33.5531248778544, 1)
        v2 = v2_ref * (w / w_ref) * np.exp(-0.015633 * (age - age_ref))
        v3 = v3_ref * fat_free_mass(male, w, age, bmi) / fat_free_mass(True, w_ref, age_ref, bmi_ref)
        cl_ref = 1.7895836588902 if male else 2.1002218877899
        params = {
            "v1": v1,
            "cl1": cl_ref * (w / w_ref) ** 0.75 * cl_maturation,
            "v2": v2,
            "cl2": 1.7500983738779 * (v2 / v2_ref) ** 0.75 * (1.0 + 1.3042680471360 * (1.0 - sigmoid(pma, 68.2767978846832, 1))),
            "v3": v3,
            "cl3": 1.1085424008536 * (v3 / v3_ref) ** 0.75 * q3_maturation,
        }
        params = _scale_volumes(params, hepatic_vd_multiplier(patient, severe_multiplier=1.6))
        super().__init__(ke0=0.146 * (w / 70.0) ** -0.25, **params)


# Remifentanil -----------------------------------------------------------------

class RemifentanilPKMinto(MammillaryPK):
    """Minto et al. 1997 with its James lean-body-mass covariate."""

    def __init__(self, patient: Patient):
        age, lbm = patient.age, patient.james_lbm()
        super().__init__(
            v1=max(1.0, 5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55)),
            cl1=max(0.01, 2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55)),
            v2=max(1.0, 9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55)),
            cl2=max(0.01, 2.05 - 0.0301 * (age - 40)),
            v3=5.42,
            cl3=max(0.005, 0.076 - 0.00113 * (age - 40)),
            ke0=max(0.01, 0.595 - 0.007 * (age - 40)),
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
    """Norepinephrine PK with endogenous secretion and the Li propofol covariate."""

    # Li et al. 2024: clearance multiplier exp(theta * Cp / 100), about 12% lower at 3.5 µg/mL.
    PROPOFOL_CL_THETA = -3.57

    def __init__(self, patient: Patient, model: str = "Beloeil"):
        w, age = patient.weight, patient.age
        self.model = model
        self.endogenous_ug_min = 0.0
        if model == "Beloeil":
            # CL = 59.6 / SAPS II with SAPS II = 30 (moderate severity).
            params = {"v1": 8.840, "cl1": 59.6 / 30.0}
        elif model == "Oualha":
            params = {"v1": 0.08 * w, "cl1": 0.11 * w**0.75}
            self.endogenous_ug_min = 0.052 * w**0.75
        elif model == "Li":
            params = {
                "v1": 2.4 * (w / 70),
                "cl1": 2.1 * np.exp(-0.377 / 100 * (age - 35)) * (w / 70) ** 0.75,
                "v2": 3.6 * (w / 70),
                "cl2": 0.6 * (w / 70) ** 0.75,
            }
            self.endogenous_ug_min = 0.4977 * (w / 70) ** 0.75
        else:
            raise ValueError(f"Unsupported norepinephrine PK model: {model!r}")
        super().__init__(ke0=0.4, **params)
        endogenous_conc = self.endogenous_ug_min / self.cl1
        self.set_state_vector([endogenous_conc] * len(self.state_fields))

    def step(self, dt_sec: float, infusion_rate_ug_sec: float, propofol_conc_ug_ml: float = 0.0) -> PKState:
        cl1_scale = 1.0
        if self.model == "Li":
            cl1_scale = float(np.exp(self.PROPOFOL_CL_THETA * max(0.0, propofol_conc_ug_ml) / 100.0))
        return super().step(dt_sec, infusion_rate_ug_sec + self.endogenous_ug_min / 60.0, cl1_scale)


class EpinephrinePK(MammillaryPK):
    """Epinephrine one-compartment PK with an effect site (ke0 0.5 min^-1)."""

    def __init__(self, patient: Patient, model: str = "Clutter"):
        w = patient.weight
        self.model = model
        if model == "Clutter":
            # Healthy adults: clearance 52-89 mL/kg/min; use 70 mL/kg/min.
            v1, cl_l_hr = 0.15 * w, 0.07 * w * 60.0
        elif model == "Abboud":
            # Adult septic shock: CL = 127 (BW/70)^0.60 (SAPS II/50)^-0.67 L/h at the
            # cohort reference SAPS II of 50; V about 7.9 L.
            v1, cl_l_hr = 7.9, 127.0 * (w / 70.0) ** 0.60
        elif model == "Oualha":
            # Children: V = 0.08 L/kg; CL = 2.00 BW^0.75 L/h.
            v1, cl_l_hr = 0.08 * w, 2.00 * w**0.75
        else:
            raise ValueError(f"Unsupported epinephrine PK model: {model!r}")
        super().__init__(v1=v1, cl1=cl_l_hr / 60.0, ke0=0.5)


class PhenylephrinePK(MammillaryPK):
    """FDA NDA 203826 two-compartment PK (t1/2 alpha 2.3 min, beta 53 min)."""

    def __init__(self, patient: Patient):
        params = _from_rate_constants(20.4 * patient.weight / 70.0, 0.124, 0.155, 0.0314, 0.0, 0.0)
        # Minimal hysteresis; ke0 gives an effect-site half-time near 1 min.
        super().__init__(ke0=0.7, **params)


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
