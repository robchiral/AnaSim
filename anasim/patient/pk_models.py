"""Linear mammillary pharmacokinetic models.

Sources are cited on each model and in docs/REFERENCES.md.

Units: volumes L, clearances L/min, ke0 1/min, inputs in model units per
second. Concentrations are µg/mL for propofol, etomidate, ketamine,
lidocaine, bupivacaine, ropivacaine, amiodarone, dantrolene, rocuronium,
succinylcholine, esmolol, protamine, and methylene blue, ng/mL for opioids,
midazolam, naloxone, flumazenil, vasoactive drugs other than vasopressin,
atropine, labetalol, glycopyrrolate, and neostigmine, mU/L for vasopressin
and insulin, and U/L for heparin.
"""

import math
from dataclasses import dataclass
from operator import attrgetter
from typing import TypedDict

import numpy as np
from scipy.linalg import expm

from anasim.core.utils import clamp, fill_response_rows

from .patient import Patient


@dataclass(slots=True)
class PKState:
    """Compartment concentrations. Unused peripheral compartments stay at zero."""

    c1: float = 0.0  # Central (plasma)
    c2: float = 0.0  # Fast peripheral
    c3: float = 0.0  # Slow peripheral
    ce: float = 0.0  # Effect site
    ce_response: float = 0.0  # Propofol clinical response
    ce_resp: float = 0.0  # Respiratory depression
    ce_rr: float = 0.0  # Morphine/hydromorphone respiratory rate
    ce_cv: float = 0.0  # Cardiovascular effect
    ce_hr: float = 0.0  # Ketamine and amiodarone chronotropy
    ce_transit: float = 0.0  # Transit before the primary effect site
    ce_neuro: float = 0.0  # Local anesthetic CNS exposure
    ce_central: float = 0.0  # Faster central/laryngeal effect site (succinylcholine)
    ce_nmj: float = 0.0  # Neuromuscular junction effect site (neostigmine)


def _pow(base: float, exp: float) -> float:
    if exp == 0.0:
        return 1.0
    if exp == 1.0:
        return base
    if exp == 2.0:
        return base * base
    return base ** exp


class MammillaryPK:
    """Central compartment with up to two peripheral compartments and effect sites.

    Mass is V1*c1 + V2*c2 + V3*c3. Hemodynamic scaling changes the vascular
    part of the central volume with blood volume, renal clearance with renal
    perfusion, and other clearances with cardiac output using per-model
    exponents. Peripheral volumes stay fixed so redistribution conserves mass.
    Below 30% of resting output, transport to organs, peripheral compartments,
    and effect sites slows with flow. Intravascular esterase clearance continues.
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
        intravascular_clearance: bool = False,
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
        self.intravascular_clearance = intravascular_clearance
        self.effect_input_fraction = 1.0  # Fraction of plasma exposure reaching effect sites
        # Exponents on the blood-volume ratio, applied only below baseline volume.
        self.cl1_deficit_exponent = cl1_deficit_exponent
        self.distribution_deficit_exponent = distribution_deficit_exponent
        self.distribution_co_exponent = distribution_co_exponent
        self.ce_input = "ce_transit" if effect_site_transit_rate is not None else "c1"
        self._has_transit = effect_site_transit_rate is not None
        transit = (("ce_transit", effect_site_transit_rate),) if effect_site_transit_rate is not None else ()
        # The primary site is last so TCI and decay predictions target it.
        self.effect_sites = (*transit, *additional_effect_sites, ("ce", ke0))
        self._single_site = len(self.effect_sites) == 1
        self._primary_ke0 = self.effect_sites[0][1]
        self._site_names = tuple(name for name, _ in self.effect_sites)
        self._site_rates = tuple(rate for _, rate in self.effect_sites)
        self._inv_v2 = 1.0 / v2 if v2 > 0 else 0.0
        self._inv_v3 = 1.0 / v3 if v3 > 0 else 0.0
        self._extra_site_names = tuple(name for name, _ in self.effect_sites[:-1])
        self._extra_site_0 = (
            attrgetter(self._extra_site_names[0]) if len(self._extra_site_names) == 1 else None
        )
        self._extra_getter = (
            attrgetter(*self._extra_site_names) if len(self._extra_site_names) > 1 else None
        )
        self._max_effect_rate = max(rate for _, rate in self.effect_sites)
        self.state_fields = (
            ("c1",) + (("c2",) if v2 > 0 else ()) + (("c3",) if v3 > 0 else ())
            + tuple(name for name, _ in self.effect_sites)
        )
        self._state_values = attrgetter(*self.state_fields)
        self.state = PKState()
        self.circulation = 1.0  # Share of normal transport
        self.temp_dist_scale = 1.0
        self.update_hemodynamics(1.0, 1.0)

    def has_mass(self) -> bool:
        """Whether any compartment or effect site holds a nonzero concentration."""
        s = self.state
        if s.c1 or s.ce or s.c2 or s.c3:
            return True
        if self._extra_site_0 is not None:
            return bool(self._extra_site_0(s))
        if self._extra_getter is not None:
            return any(self._extra_getter(s))
        return False

    @property
    def k10(self) -> float:
        return self.elimination_clearance / self.v1

    @property
    def elimination_clearance(self) -> float:
        transport = 1.0 if self.intravascular_clearance else self.circulation
        return self.cl1 * self.cl1_scale * transport

    @property
    def effective_basal_input_rate(self) -> float:
        transport = 1.0 if self.intravascular_clearance else self.circulation
        return self.basal_input_rate * transport

    @property
    def k12(self) -> float:
        return self.cl2 * self.circulation / self.v1

    @property
    def k21(self) -> float:
        return self.cl2 * self.circulation / self.v2 if self.v2 > 0 else 0.0

    @property
    def k13(self) -> float:
        return self.cl3 * self.circulation / self.v1

    @property
    def k31(self) -> float:
        return self.cl3 * self.circulation / self.v3 if self.v3 > 0 else 0.0

    def _blood_loss_keep_fraction(self, volume_l: float, baseline_blood_volume_l: float) -> float:
        vascular_share = min(self.v1_base, baseline_blood_volume_l) / baseline_blood_volume_l
        removed_volume = volume_l * vascular_share * self.v1_scale
        return max(0.0, 1.0 - removed_volume / self.v1)

    def remove_blood(self, volume_l: float, baseline_blood_volume_l: float) -> None:
        """Remove drug with shed blood from the vascular share of V1, at unit blood:plasma partitioning."""
        self.state.c1 *= self._blood_loss_keep_fraction(volume_l, baseline_blood_volume_l)

    def update_hemodynamics(
        self, v_ratio: float, co_ratio: float, renal_perfusion: float = 1.0,
        blood_volume_l: float | None = None, temp_c: float = 37.0,
        *, temp_met: float | None = None, temp_dist: float | None = None,
    ) -> None:
        """Rescale effective V1 without creating or removing central drug mass.

        Only the part of V1 up to the baseline blood volume changes with blood
        volume; the rest is tissue. Without blood_volume_l, all of V1 does.
        Core temperature scales non-renal metabolic and intercompartmental
        clearances (Leslie 1995; rocuronium override extrapolated from Caldwell 2000
        and Smeulers 1995).
        """
        vascular = self.v1_base if blood_volume_l is None else min(self.v1_base, blood_volume_l)
        new_v1 = (self.v1_base + vascular * (v_ratio - 1.0)) * self.v1_scale
        if self.state.c1:
            self.state.c1 *= self.v1 / new_v1
        self.v1 = new_v1
        deficit = min(1.0, v_ratio)
        if temp_met is None or temp_dist is None:
            clamped_t = clamp(temp_c, 18.0, 38.5)
            temp_met = 2.0 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0
            temp_dist = 1.6 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0
        self.temp_dist_scale = temp_dist
        nonrenal = (1.0 - self.renal_fraction) * _pow(co_ratio, self.cl1_co_exponent) * temp_met
        self.cl1 = self.cl1_base * (nonrenal + self.renal_fraction * renal_perfusion)
        if deficit < 1.0 and self.cl1_deficit_exponent:
            self.cl1 *= _pow(deficit, self.cl1_deficit_exponent)
        if self.cl2_base or self.cl3_base:
            distribution_scale = _pow(co_ratio, self.distribution_co_exponent) * temp_dist
            if deficit < 1.0 and self.distribution_deficit_exponent:
                distribution_scale *= _pow(deficit, self.distribution_deficit_exponent)
            self.cl2 = self.cl2_base * distribution_scale * self.cl2_scale
            self.cl3 = self.cl3_base * distribution_scale * self.cl3_scale
        else:
            self.cl2 = self.cl3 = 0.0

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
        """Advance with conserved midpoint steps across central, peripheral, and effect compartments."""
        if self.basal_input_rate > 0.0:
            input_rate_per_sec += self.effective_basal_input_rate
        s = self.state
        if not (input_rate_per_sec or self.has_mass()):
            return s  # Skip drugs with no input or residual concentration.
        v1, v2, v3 = self.v1, self.v2, self.v3
        inv_v1, inv_v2, inv_v3 = 1.0 / v1, self._inv_v2, self._inv_v3
        circulation = self.circulation
        effect_transport = circulation * self.temp_dist_scale
        cl2 = self.cl2 * circulation if v2 > 0 else 0.0
        cl3 = self.cl3 * circulation if v3 > 0 else 0.0
        elimination_cl = self.elimination_clearance
        fastest_rate = max((elimination_cl + cl2 + cl3) * inv_v1,
                           cl2 * inv_v2, cl3 * inv_v3,
                           self._max_effect_rate * effect_transport)
        # At most 0.25 s and 2% turnover per midpoint step, including after
        # blood-volume scaling. Equal flux terms preserve compartment mass.
        substeps = max(1, math.ceil(dt_sec * max(4.0, fastest_rate / 1.2)))
        dt_min = dt_sec / (60.0 * substeps)
        half_dt_min = 0.5 * dt_min
        input_per_min = input_rate_per_sec * 60.0
        effect_frac = self.effect_input_fraction
        c1, c2, c3 = s.c1, s.c2, s.c3
        if self._single_site:
            ce = s.ce
            rc_half = self._primary_ke0 * effect_transport * half_dt_min
            rc_full = self._primary_ke0 * effect_transport * dt_min
            for _ in range(substeps):
                flux2_0 = cl2 * (c1 - c2)
                flux3_0 = cl3 * (c1 - c3)
                elim_0 = elimination_cl * c1
                c1_mid = c1 + (input_per_min - elim_0 - flux2_0 - flux3_0) * inv_v1 * half_dt_min
                c2_mid = c2 + flux2_0 * inv_v2 * half_dt_min if v2 > 0 else 0.0
                c3_mid = c3 + flux3_0 * inv_v3 * half_dt_min if v3 > 0 else 0.0
                ce_mid = ce + rc_half * (c1 * effect_frac - ce)
                ce = ce + rc_full * (c1_mid * effect_frac - ce_mid)
                flux2_mid = cl2 * (c1_mid - c2_mid)
                flux3_mid = cl3 * (c1_mid - c3_mid)
                elim_mid = elimination_cl * c1_mid
                c1 = c1 + (input_per_min - elim_mid - flux2_mid - flux3_mid) * inv_v1 * dt_min
                if v2 > 0:
                    c2 = c2 + flux2_mid * inv_v2 * dt_min
                if v3 > 0:
                    c3 = c3 + flux3_mid * inv_v3 * dt_min
            s.ce = ce
        else:
            site_names = self._site_names
            site_vals = [getattr(s, name) for name in site_names]
            site_rc_half = [rate * effect_transport * half_dt_min for rate in self._site_rates]
            site_rc_full = [rate * effect_transport * dt_min for rate in self._site_rates]
            n_sites = len(site_names)
            last_idx = n_sites - 1
            has_transit = self._has_transit
            for _ in range(substeps):
                flux2_0 = cl2 * (c1 - c2)
                flux3_0 = cl3 * (c1 - c3)
                elim_0 = elimination_cl * c1
                c1_mid = c1 + (input_per_min - elim_0 - flux2_0 - flux3_0) * inv_v1 * half_dt_min
                c2_mid = c2 + flux2_0 * inv_v2 * half_dt_min if v2 > 0 else 0.0
                c3_mid = c3 + flux3_0 * inv_v3 * half_dt_min if v3 > 0 else 0.0
                c1_eff_0 = c1 * effect_frac
                c1_eff_mid = c1_mid * effect_frac
                if not has_transit:
                    for i in range(n_sites):
                        ce_i = site_vals[i]
                        ce_mid = ce_i + site_rc_half[i] * (c1_eff_0 - ce_i)
                        site_vals[i] = ce_i + site_rc_full[i] * (c1_eff_mid - ce_mid)
                else:
                    transit_0 = site_vals[0]
                    transit_mid = transit_0 + site_rc_half[0] * (c1_eff_0 - transit_0)
                    site_vals[0] = transit_0 + site_rc_full[0] * (c1_eff_mid - transit_mid)
                    for i in range(1, last_idx):
                        ce_i = site_vals[i]
                        ce_mid = ce_i + site_rc_half[i] * (c1_eff_0 - ce_i)
                        site_vals[i] = ce_i + site_rc_full[i] * (c1_eff_mid - ce_mid)
                    ce_last = site_vals[last_idx]
                    ce_mid = ce_last + site_rc_half[last_idx] * (transit_0 - ce_last)
                    site_vals[last_idx] = ce_last + site_rc_full[last_idx] * (transit_mid - ce_mid)
                flux2_mid = cl2 * (c1_mid - c2_mid)
                flux3_mid = cl3 * (c1_mid - c3_mid)
                elim_mid = elimination_cl * c1_mid
                c1 = c1 + (input_per_min - elim_mid - flux2_mid - flux3_mid) * inv_v1 * dt_min
                if v2 > 0:
                    c2 = c2 + flux2_mid * inv_v2 * dt_min
                if v3 > 0:
                    c3 = c3 + flux3_mid * inv_v3 * dt_min
            for name, val in zip(site_names, site_vals, strict=True):
                setattr(s, name, val)
        s.c1 = c1
        if v2 > 0:
            s.c2 = c2
        if v3 > 0:
            s.c3 = c3
        return s

    def get_ss_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return continuous A (1/min) and B for the state order in `state_fields`."""
        peripherals = [(v, cl * self.circulation)
                       for v, cl in ((self.v2, self.cl2), (self.v3, self.cl3)) if v > 0]
        n = len(peripherals) + 1 + len(self.effect_sites)
        A = np.zeros((n, n))
        B = np.zeros((n, 1))
        A[0, 0] = -(self.elimination_clearance + sum(cl for _, cl in peripherals)) / self.v1
        for i, (v, cl) in enumerate(peripherals, start=1):
            A[0, i] = cl / self.v1
            A[i, 0] = cl / v
            A[i, i] = -cl / v
        for i, (name, rate) in enumerate(self.effect_sites, start=len(peripherals) + 1):
            rate *= self.circulation * self.temp_dist_scale
            source = self.state_fields.index(self.ce_input) if name == "ce" else 0
            A[i, source] = rate * (self.effect_input_fraction if source == 0 else 1.0)
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
        self.v1 = self.v1_base
        self.cl1 = self.cl1_base
        self.cl2 = self.cl2_base
        self.cl3 = self.cl3_base
        self.v1_scale = 1.0
        self.cl1_scale = 1.0
        self.cl2_scale = 1.0
        self.cl3_scale = 1.0
        self.effect_input_fraction = 1.0
        self.circulation = 1.0
        self.temp_dist_scale = 1.0
        self.update_hemodynamics(1.0, 1.0)

    def simulate_decay(self, target_fraction: float = 0.5, max_seconds: int = 3600) -> float:
        """Return minutes for Ce to fall to `target_fraction` with no further input."""
        ce_target = self.state.ce * target_fraction
        if ce_target <= 0.0:
            return 0.0
        if self.circulation == 0.0 or max_seconds <= 0:
            return max_seconds / 60.0
        A, B = self.get_ss_matrices()
        transition = expm(A / 60.0)
        basal_rate = self.effective_basal_input_rate
        equilibrium = (np.linalg.solve(-A, B[:, 0] * basal_rate * 60.0)
                       if basal_rate else np.zeros(A.shape[0]))
        x = self.state_vector() - equilibrium
        # Check every second in each block, including when Ce initially rises.
        block = min(128, max_seconds)
        response = np.empty((block, len(x)))
        fill_response_rows(response, transition, -1)
        advance = np.linalg.matrix_power(transition, block)
        for start in range(0, max_seconds, block):
            count = min(block, max_seconds - start)
            crossed = np.flatnonzero(response[:count] @ x + equilibrium[-1] <= ce_target)
            if crossed.size:
                return (start + int(crossed[0]) + 1) / 60.0
            x = advance @ x
        return max_seconds / 60.0


class _CompartmentParams(TypedDict):
    v1: float
    cl1: float
    v2: float
    cl2: float
    v3: float
    cl3: float


def _from_rate_constants(
    v1: float, k10: float, k12: float, k21: float, k13: float, k31: float,
) -> _CompartmentParams:
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


def renal_capacity_scaling(
    patient: Patient, renal_fraction: float, hepatic_fraction: float = 0.0,
) -> tuple[float, float]:
    """Clearance scale for organ capacity, and the renal share of the scaled clearance.

    Renal perfusion acts only on the renal share left after the capacity change.
    """
    scale = organ_clearance_scaler(patient, hepatic_fraction=hepatic_fraction, renal_fraction=renal_fraction)
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
# Cardiac output during surgery under anesthesia, as a fraction of awake output
# (0.82 with TIVA, 0.85 with sevoflurane). Clearances from studies in
# anesthetized patients apply at this output.
ANESTHETIZED_OUTPUT = 0.82


def _sigmoid(x: float, x50: float, gamma: float) -> float:
    return x**gamma / (x50**gamma + x**gamma)


def janmahasatian_ffm(weight: float, height: float, male: bool) -> float:
    """Adult fat-free mass, kg; height is in cm."""
    bmi = weight / (height / 100.0) ** 2
    denominator = 6680.0 + 216.0 * bmi if male else 8780.0 + 244.0 * bmi
    return 9270.0 * weight / denominator


def _eleveld_ffm(age: float, weight: float, height: float, male: bool) -> float:
    """Al-Sallami maturation of adult fat-free mass for both Eleveld models."""
    if male:
        maturation = 0.88 + 0.12 / (1.0 + (age / 13.4) ** -12.7)
    else:
        maturation = 1.11 - 0.11 / (1.0 + (age / 7.1) ** -1.1)
    return maturation * janmahasatian_ffm(weight, height, male)


def _lean_weight_kg(patient: Patient) -> float:
    """Lean-scaled equivalent body weight (kg) from Janmahasatian FFM (obesity LBW principle: Ingrande and Lemmens 2010).

    Equals actual weight in non-obese adults (BMI <= 27 kg/m^2) so published
    normal-weight PK parameters match their source cohorts; above BMI 27 it adds
    only the extra Janmahasatian fat-free mass so central volume and organ
    clearances do not scale with excess adipose tissue.
    """
    ref_w = 27.0 * (patient.height / 100.0) ** 2
    if patient.weight <= ref_w:
        return patient.weight
    male = patient.sex == "male"
    ref_ffm = janmahasatian_ffm(ref_w, patient.height, male)
    actual_ffm = janmahasatian_ffm(patient.weight, patient.height, male)
    return ref_w + max(0.0, actual_ffm - ref_ffm)


def _pregnancy_v1(patient: Patient, v1: float) -> float:
    """Expand central volume V1 for gestational blood-volume and ECF expansion (Dallmann 2017)."""
    bv = patient.pregnancy.blood_volume
    ecf = patient.pregnancy.extracellular_water
    if bv == 1.0 and ecf == 1.0:
        return v1
    nonpreg_bv_l = (patient.estimate_blood_volume() / 1000.0) / bv
    vascular_share = min(1.0, nonpreg_bv_l / max(0.5, v1))
    return v1 * (vascular_share * bv + (1.0 - vascular_share) * ecf)


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
            "v1": _pregnancy_v1(patient, v1),
            "cl1": cl_ref * (w / w_ref) ** 0.75 * cl_maturation * eta.cl,
            "v2": v2,
            "cl2": 1.7500983738779 * (v2 / v2_ref) ** 0.75 * (1.0 + 1.3042680471360 * (1.0 - _sigmoid(pma, 68.2767978846832, 1)))
            * eta.q2,
            "v3": v3,
            "cl3": 1.1085424008536 * (v3 / v3_ref) ** 0.75 * q3_maturation * eta.q3,
        }
        params = _scale_volumes(params, hepatic_vd_multiplier(patient, severe_multiplier=1.6))
        # The clinical response rate is a timing assumption (Kuizenga 2019 study
        # design); respiratory equilibration half-time is 2.6 min (Bouillon 2004b).
        # Eleveld's anesthetized cohort already reflects propofol's fall in cardiac
        # output (around 0.75-0.82 of awake baseline), so update_hemodynamics
        # scales downward only below 0.75 while keeping Johnson 2003/2004
        # hemorrhagic deficit scaling intact.
        super().__init__(
            ke0=0.14636965112551278 * (w / 70.0) ** -0.25 * eta.ke0,
            additional_effect_sites=(("ce_response", 0.456), ("ce_resp", math.log(2.0) / 2.6)),
            cl1_co_exponent=0.65,
            distribution_co_exponent=0.35,
            cl1_deficit_exponent=0.45,
            distribution_deficit_exponent=1.65,
            **params,
        )

    def update_hemodynamics(
        self, v_ratio: float, co_ratio: float, renal_perfusion: float = 1.0,
        blood_volume_l: float | None = None, temp_c: float = 37.0,
        *, temp_met: float | None = None, temp_dist: float | None = None,
    ) -> None:
        # Eleveld parameters apply at anesthetized output (including opioid-TIVA
        # CO ~0.75-0.82); scale downward for normovolemic low-CO states (HFrEF,
        # cardiogenic shock) without shifting normodynamic TIVA clearance.
        low_co_ratio = min(1.0, co_ratio / 0.75)
        super().update_hemodynamics(
            v_ratio, low_co_ratio, renal_perfusion, blood_volume_l=blood_volume_l,
            temp_c=temp_c, temp_met=temp_met, temp_dist=temp_dist,
        )

    def update_midazolam(self, concentration_ng_ml: float) -> bool:
        """Vuyk 2009 Table 2, midazolam-only clearance covariates.

        Apply relative changes to Eleveld, centered on no midazolam, and bound
        the covariate at the studied 225 ng/mL. This transfer is an assumption.
        """
        cp = clamp(concentration_ng_ml, 0.0, 225.0)
        if cp == 0.0 and self.cl1_scale == 1.0 and self.cl2_scale == 1.0 and self.cl3_scale == 1.0:
            return False
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
        eta = patient.variability.remifentanil
        size = _eleveld_ffm(age, weight, patient.height, patient.sex == "male") / _eleveld_ffm(35.0, 70.0, 170.0, True)
        aging_fast = math.exp(-0.00554 * (age - 35.0))
        aging_slow = math.exp(-0.00327 * (age - 35.0))
        sex_factor = 1.0
        if patient.sex == "female":
            sex_factor += 0.470 * _sigmoid(age, 12.0, 6.0) * (1.0 - _sigmoid(age, 45.0, 6.0))
        v2 = 8.82 * size * aging_slow * sex_factor * eta.v2
        # Cap weight in Eleveld's V3 negative exponential at its adult training ceiling
        # (106 kg) so extreme obesity retains lean/FFM-scaled peripheral volume (Egan 1998).
        v3 = 5.03 * size * math.exp(-0.0315 * (age - 35.0) - 0.0260 * (min(weight, 106.0) - 70.0)) * eta.v3
        super().__init__(
            v1=_pregnancy_v1(patient, 5.81 * size * aging_fast * eta.v1),
            v2=v2,
            v3=v3,
            cl1=2.58 * size**0.75 * _sigmoid(weight, 2.88, 2.0) / _sigmoid(70.0, 2.88, 2.0) * sex_factor * aging_slow
            * eta.cl,
            cl2=1.72 * (v2 / 8.82)**0.75 * aging_fast * sex_factor * eta.q2,
            cl3=0.124 * (v3 / 5.03)**0.75 * aging_fast * eta.q3,
            ke0=1.09 * math.exp(-0.0289 * (age - 35.0)) * eta.ke0,
            additional_effect_sites=(("ce_resp", math.log(2.0) / 0.53),),
            # Swine bled 48 mL/kg had 58% lower central clearance, half the V1,
            # and 62% lower V2, with unchanged distribution clearances (Johnson 2001).
            cl1_deficit_exponent=0.8,
            intravascular_clearance=True,
            **_NO_CO_COVARIATE,
        )


# Fentanyl, midazolam, etomidate, ketamine, and lidocaine ----------------------

def _fentanyl_pk_mass(weight_kg: float) -> float:
    """Shibutani 2004 pharmacokinetic mass (kg) for fentanyl in lean and obese adults."""
    return 52.0 / (1.0 + (196.4 * math.exp(-0.025 * weight_kg) - 53.66) / 100.0)


class FentanylPK(MammillaryPK):
    """Bae et al. 2020 allometric three-compartment model (ng/mL).

    Volumes scale with (weight/70)^1.23 up to 70 kg and Shibutani 2004
    pharmacokinetic mass above 70 kg; clearances scale with (weight/70)^0.313
    up to 70 kg, pharmacokinetic mass in overweight/obesity (Shibutani 2004),
    and hepatic function. ke0 0.147 min^-1 is from Scott and Stanski 1987.
    """

    def __init__(self, patient: Patient):
        if patient.weight > 70.0:
            pk_size = _fentanyl_pk_mass(patient.weight) / _fentanyl_pk_mass(70.0)
            volume = pk_size ** 1.23
            flow = pk_size / ANESTHETIZED_OUTPUT
        else:
            volume = (patient.weight / 70.0) ** 1.23
            flow = (patient.weight / 70.0) ** 0.313 / ANESTHETIZED_OUTPUT
        hepatic_scale = organ_clearance_scaler(patient, hepatic_fraction=0.75)
        eta = patient.variability.fentanyl
        super().__init__(
            v1=_pregnancy_v1(patient, 10.1 * volume * eta.v1),
            cl1=0.704 * flow * hepatic_scale * eta.cl,
            v2=26.5 * volume * eta.v2,
            cl2=2.38 * flow * eta.q2,
            v3=206.0 * volume * eta.v3,
            cl3=1.49 * flow * eta.q3,
            ke0=0.147,
        )


def _meissner_opioid_params(
    patient: Patient,
    v1_ref: float,
    k10: float,
    k12: float,
    k21: float,
    k13: float,
    k31: float,
    *,
    hepatic_fraction: float,
    renal_fraction: float,
) -> tuple[_CompartmentParams, float]:
    male = patient.sex == "male"
    size = janmahasatian_ffm(patient.weight, patient.height, male) / janmahasatian_ffm(70.0, 170.0, male)
    rate_scale = size ** -0.25
    age_scale = max(0.5, 1.0 - 0.006 * max(0.0, patient.age - 40.0))
    organ_scale, active_renal = renal_capacity_scaling(
        patient, renal_fraction, hepatic_fraction=hepatic_fraction
    )
    params = _from_rate_constants(
        v1_ref * size,
        k10 / 60 * rate_scale,
        k12 / 60 * rate_scale,
        k21 / 60 * rate_scale,
        k13 / 60 * rate_scale,
        k31 / 60 * rate_scale,
    )
    params["v1"] = _pregnancy_v1(patient, params["v1"])
    params["cl1"] *= age_scale * organ_scale
    return params, active_renal


class MorphinePK(MammillaryPK):
    """Meissner 2025 Table 1 PK and Olofsen 2026 effect sites, ng/mL (input mcg).

    Olofsen's shared parameters are divided by each drug factor for morphine
    and multiplied by it for hydromorphone. Volumes scale with Janmahasatian
    fat-free mass and rate constants with FFM**-0.25 around a 70 kg, 170 cm
    reference (Meissner 2025), with age and hepatic/renal clearance scaling.
    Active morphine-6-glucuronide (M6G) accumulates in renal impairment
    (Osborne 1993: M6G AUC(0-12) ~5x normal in nondialysis kidney failure),
    adding excess CNS/respiratory activity above the normal-GFR baseline
    already folded into Olofsen's morphine PD.
    """

    def __init__(self, patient: Patient):
        params, renal_fraction = _meissner_opioid_params(
            patient, 20.0, 5.1, 4.3, 1.3, 1.2, 0.306,
            hepatic_fraction=0.65, renal_fraction=0.35,
        )
        super().__init__(
            **params,
            cl1_co_exponent=0.7,
            distribution_co_exponent=0.0,
            renal_fraction=renal_fraction,
            ke0=math.log(2.0) / (0.73 / 1.02 * 60.0),
            additional_effect_sites=(
                ("ce_resp", math.log(2.0) / (3.3 / 0.694 * 60.0)),
                ("ce_rr", math.log(2.0) / (2.2 / 0.736 * patient.variability.opioid_rate.half_time * 60.0)),
            ),
        )
        self._hepatic_function = patient.hepatic_function
        self._renal_capacity = patient.renal_clearance_factor
        self._renal_perfusion = 1.0
        self._baseline_temp = patient.baseline_temp
        self._temp_c = patient.baseline_temp
        self._m6g_actual = 0.0
        self._m6g_normal = 0.0
        self._m6g_ce_actual = 0.0
        self._m6g_ce_normal = 0.0

    @property
    def m6g_excess_ce(self) -> float:
        """Excess active M6G effect-site concentration (morphine ng/mL equiv) in renal failure."""
        return max(0.0, self._m6g_ce_actual - self._m6g_ce_normal)

    def has_mass(self) -> bool:
        return (
            super().has_mass()
            or self._m6g_actual > 0.0
            or self._m6g_normal > 0.0
            or self._m6g_ce_actual > 0.0
            or self._m6g_ce_normal > 0.0
        )

    def remove_blood(self, volume_l: float, baseline_blood_volume_l: float) -> None:
        keep = self._blood_loss_keep_fraction(volume_l, baseline_blood_volume_l)
        self.state.c1 *= keep
        if self._m6g_actual > 0.0 or self._m6g_normal > 0.0:
            self._m6g_actual *= keep
            self._m6g_normal *= keep

    def reset(self) -> None:
        super().reset()
        self._renal_perfusion = 1.0
        self._temp_c = self._baseline_temp
        self._m6g_actual = 0.0
        self._m6g_normal = 0.0
        self._m6g_ce_actual = 0.0
        self._m6g_ce_normal = 0.0

    def update_hemodynamics(
        self, v_ratio: float, co_ratio: float, renal_perfusion: float = 1.0,
        blood_volume_l: float | None = None, temp_c: float = 37.0,
        *, temp_met: float | None = None, temp_dist: float | None = None,
    ) -> None:
        self._renal_perfusion = max(0.0, renal_perfusion)
        self._temp_c = temp_c
        super().update_hemodynamics(
            v_ratio, co_ratio, renal_perfusion,
            blood_volume_l=blood_volume_l, temp_c=temp_c,
            temp_met=temp_met, temp_dist=temp_dist,
        )

    _M6G_K_FORM_BASE = 0.011
    _M6G_POTENCY_WEIGHT = 0.28
    _M6G_ELIM_T_HALF_MIN = 120.0
    _M6G_CE_T_HALF_MIN = 210.0
    _M6G_NONRENAL_SHARE = 0.18

    def _m6g_step_factors(self, dt_min: float) -> tuple[float, float, float, float, float, float]:
        """Return (k_form, k_elim_norm, k_elim_act, decay_norm, decay_act, alpha_ce) for dt_min."""
        circ = self.circulation
        clamped_t = clamp(self._temp_c, 18.0, 38.5)
        # Favié 2019 (human neonatal HIE hypothermia, extrapolated to adults): parent morphine
        # glucuronidation falls ~6.9%/°C (Q10 ~2.0) and M6G clearance falls ~4.9%/°C (Q10 ~1.65).
        temp_form = 2.0 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0
        temp_elim = 1.65 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0
        k_form = self._M6G_K_FORM_BASE * self._hepatic_function * temp_form * circ
        k_elim_ref = (math.log(2.0) / self._M6G_ELIM_T_HALF_MIN) * temp_elim * circ
        eff_gfr = self._M6G_NONRENAL_SHARE + (1.0 - self._M6G_NONRENAL_SHARE) * clamp(
            self._renal_capacity * self._renal_perfusion, 0.0, 1.0,
        )
        k_elim_act = k_elim_ref * eff_gfr
        decay_norm = math.exp(-k_elim_ref * dt_min)
        decay_act = math.exp(-k_elim_act * dt_min)
        ke0_m6g = (math.log(2.0) / self._M6G_CE_T_HALF_MIN) * circ
        alpha_ce = -math.expm1(-ke0_m6g * dt_min)
        return k_form, k_elim_ref, k_elim_act, decay_norm, decay_act, alpha_ce

    @classmethod
    def _advance_m6g(
        cls,
        c1: float,
        m6g_n: float,
        m6g_a: float,
        m6g_ce_n: float,
        m6g_ce_a: float,
        factors: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float]:
        k_form, k_elim_norm, k_elim_act, decay_norm, decay_act, alpha_ce = factors
        inflow = cls._M6G_POTENCY_WEIGHT * k_form * c1
        if k_elim_norm > 0.0:
            target_n = inflow / k_elim_norm
            m6g_n = target_n + (m6g_n - target_n) * decay_norm
        if k_elim_act > 0.0:
            target_a = inflow / k_elim_act
            m6g_a = target_a + (m6g_a - target_a) * decay_act
        m6g_ce_n += (m6g_n - m6g_ce_n) * alpha_ce
        m6g_ce_a += (m6g_a - m6g_ce_a) * alpha_ce
        return m6g_n, m6g_a, m6g_ce_n, m6g_ce_a

    def step(self, dt_sec: float, input_rate_per_sec: float) -> PKState:
        state = super().step(dt_sec, input_rate_per_sec)
        if dt_sec > 0.0 and (state.c1 > 0.0 or self._m6g_actual > 0.0 or self._m6g_normal > 0.0
                             or self._m6g_ce_actual > 0.0 or self._m6g_ce_normal > 0.0):
            # Hepatic glucuronidation produces M6G proportional to hepatic clearance;
            # M6G clears almost entirely by GFR (Osborne 1993: normal t1/2 2.6 +/- 0.7 h SD;
            # modeled with a calibrated 2 h reference half-life and ~3.5 h BBB t1/2).
            factors = self._m6g_step_factors(dt_sec / 60.0)
            self._m6g_normal, self._m6g_actual, self._m6g_ce_normal, self._m6g_ce_actual = self._advance_m6g(
                state.c1, self._m6g_normal, self._m6g_actual, self._m6g_ce_normal, self._m6g_ce_actual, factors,
            )
        return state

    def simulate_decay(self, target_fraction: float = 0.5, max_seconds: int = 3600) -> float:
        """Return minutes for total active morphine + excess M6G effect-site concentration to decay."""
        if (
            self._m6g_actual <= 1e-12
            and self._m6g_normal <= 1e-12
            and self._m6g_ce_actual <= 1e-12
            and self._m6g_ce_normal <= 1e-12
            and self._renal_capacity * self._renal_perfusion >= 0.999
        ):
            return super().simulate_decay(target_fraction=target_fraction, max_seconds=max_seconds)
        total_ce = self.state.ce + self.m6g_excess_ce
        ce_target = total_ce * target_fraction
        if ce_target <= 0.0:
            return 0.0
        if self.circulation == 0.0 or max_seconds <= 0:
            return max_seconds / 60.0
        A, _ = self.get_ss_matrices()
        transition = expm(A / 60.0)
        x = self.state_vector()
        factors = self._m6g_step_factors(1.0 / 60.0)
        m6g_n = self._m6g_normal
        m6g_a = self._m6g_actual
        m6g_ce_n = self._m6g_ce_normal
        m6g_ce_a = self._m6g_ce_actual
        ce_idx = self.state_fields.index("ce")
        for sec in range(1, max_seconds + 1):
            x = transition @ x
            m6g_n, m6g_a, m6g_ce_n, m6g_ce_a = self._advance_m6g(
                max(0.0, float(x[0])), m6g_n, m6g_a, m6g_ce_n, m6g_ce_a, factors,
            )
            if float(x[ce_idx]) + max(0.0, m6g_ce_a - m6g_ce_n) <= ce_target:
                return sec / 60.0
        return max_seconds / 60.0


class HydromorphonePK(MammillaryPK):
    """Meissner 2025 Table 1 PK with its sex factor and Olofsen 2026 sites, ng/mL."""

    def __init__(self, patient: Patient):
        v1_ref = 13.6 / 0.908 if patient.sex == "male" else 13.6 * 0.908
        params, renal_fraction = _meissner_opioid_params(
            patient, v1_ref, 6.1, 6.6, 1.6, 1.6, 0.41,
            hepatic_fraction=0.85, renal_fraction=0.15,
        )
        super().__init__(
            **params,
            cl1_co_exponent=0.7,
            distribution_co_exponent=0.0,
            renal_fraction=renal_fraction,
            ke0=math.log(2.0) / (0.73 * 1.02 * 60.0),
            additional_effect_sites=(
                ("ce_resp", math.log(2.0) / (3.3 * 0.694 * 60.0)),
                ("ce_rr", math.log(2.0) / (2.2 * 0.736 * patient.variability.opioid_rate.half_time * 60.0)),
            ),
        )


class AtropinePK(MammillaryPK):
    """Hinderling 1985 I Table I means (CL 1.175 L/min, V1 19.1 L, k12 0.56 and
    k21 0.0557 /min), in ng/mL.

    ke0 = k21 links HR to the peripheral compartment (part II). Lean-weight
    scaling from the cohort's 72.3 kg mean and proportional
    renal-capacity scaling of the 56.8% renal fraction are simulator assumptions.
    """

    def __init__(self, patient: Patient):
        lean_size = _lean_weight_kg(patient) / 72.3
        params = _from_rate_constants(19.1 * lean_size, 1.175 / 19.1, 0.56, 0.0557, 0.0, 0.0)
        params["v1"] = _pregnancy_v1(patient, params["v1"])
        clearance_scale, renal_fraction = renal_capacity_scaling(patient, 0.568)
        params["cl1"] = 1.175 * lean_size**0.75 * clearance_scale
        super().__init__(**params, ke0=0.0557, renal_fraction=renal_fraction, **_NO_CO_COVARIATE)


class EphedrinePK(MammillaryPK):
    """Calibrated two-compartment IV ephedrine model (ng/mL) preserving Persky 2004
    oral Vss (4.0 L/kg) and CL while matching IV bolus onset and 60-min duration
    (Rådström 1995; Michelsen 1998).

    V1 1.15 L/kg and V2 2.85 L/kg preserve Vss 4.0 L/kg (terminal t1/2 ~6.3 h)
    with a ~9 min alpha distribution phase so a single 5-10 mg IV bolus peaks early
    and declines over 15-30 min while retaining pressor effect at 60 min.
    """

    def __init__(self, patient: Patient):
        lean_w = _lean_weight_kg(patient)
        lean_size = lean_w / 70.0
        clearance_scale, renal_fraction = renal_capacity_scaling(patient, 0.70)
        super().__init__(
            v1=_pregnancy_v1(patient, 1.15 * lean_w),
            cl1=0.47 * 70.0 / 60.0 * lean_size**0.75 * clearance_scale,
            v2=2.85 * lean_w,
            cl2=4.0 * lean_size**0.75,
            ke0=0.9,
            renal_fraction=renal_fraction,
            **_NO_CO_COVARIATE,
        )


class SuccinylcholinePK(MammillaryPK):
    """Roy 2002 plasma PK with effect-site timing calibrated to Kopman 2003.

    Transit and effect-site rates (1/min) fit Table 2 twitch time courses at
    0.4-1 mg/kg. Concentrations are mcg/mL; hydrolysis has no organ/flow covariate.
    A faster central/laryngeal site (`ce_central`, Dhonneur 1999; Chung and Rowbottom 1993)
    mediates rapid glottic/diaphragmatic onset, and cumulative effect-site
    exposure tracks Phase II desensitization (Lee 1975).
    """

    def __init__(self, patient: Patient):
        params = _from_rate_constants(0.009 * patient.weight, 5.0, 2.9, 1.6, 0.0, 0.0)
        params["v1"] = _pregnancy_v1(patient, params["v1"])
        super().__init__(
            **params, ke0=0.4, effect_site_transit_rate=1.3,
            additional_effect_sites=(("ce_central", 0.50),),
            intravascular_clearance=True, **_NO_CO_COVARIATE,
        )
        self.phase2_exposure = 0.0  # Cumulative mcg*min/mL exposure driving Phase II fade

    def reset(self) -> None:
        super().reset()
        self.phase2_exposure = 0.0

    def has_mass(self) -> bool:
        return super().has_mass() or self.phase2_exposure > 1e-9

    def step(self, dt_sec: float, input_rate_per_sec: float) -> PKState:
        state = super().step(dt_sec, input_rate_per_sec)
        if dt_sec > 0.0 and (state.ce > 0.0 or self.phase2_exposure > 0.0):
            dt_min = dt_sec / 60.0
            decay = math.exp(-math.log(2.0) * dt_min / 30.0)
            self.phase2_exposure = self.phase2_exposure * decay + state.ce * dt_min
        return state


class MidazolamPK(MammillaryPK):
    """Albrecht et al. 1999 three-compartment model (ng/mL).

    Young volunteers (24-28 y, 66-89 kg) gave Vc 7.9 L and CL 399 mL/min;
    k12 fell from 0.19 to 0.10 min^-1 by 71 years, and ke0 from 0.11 to 0.08
    min^-1. Central volume and clearances scale with lean weight from 78 kg,
    while the lipophilic slow depot V3 scales with total weight.
    """

    def __init__(self, patient: Patient):
        lean_size = _lean_weight_kg(patient) / 78.0
        tbw_size = patient.weight / 78.0
        years = clamp((patient.age - 26.0) / 45.0, 0.0, 1.0)
        v1_lean = 7.9 * lean_size
        k12 = 0.19 - 0.09 * years
        super().__init__(
            v1=_pregnancy_v1(patient, v1_lean),
            cl1=0.399 * lean_size**0.75 * organ_clearance_scaler(patient, hepatic_fraction=0.9),
            v2=v1_lean * k12 / 0.060,
            cl2=7.9 * k12 * lean_size**0.75,
            v3=7.9 * tbw_size * 0.065 / 0.0083,
            cl3=7.9 * 0.065 * lean_size**0.75,
            ke0=0.11 - 0.03 * years,
            cl1_co_exponent=0.2,
        )
        self._propofol_cp_ng_ml = 0.0

    def reset(self) -> None:
        super().reset()
        self._propofol_cp_ng_ml = 0.0

    def update_propofol(self, concentration_ug_ml: float) -> bool:
        """Lichtenbelt 2010 Table 2, propofol-only covariates on V1, CL and Q2.

        Apply relative changes to Albrecht, centered on no propofol, and bound
        the covariate at the studied 1.2 mcg/mL. This transfer is an assumption.
        """
        cp_ng_ml = 1000.0 * clamp(concentration_ug_ml, 0.0, 1.2)
        if cp_ng_ml == self._propofol_cp_ng_ml:
            return False
        changed = self._set_interaction_scales(
            v1=math.exp(-0.000453 * cp_ng_ml),
            cl1=math.exp(-0.000205 * cp_ng_ml),
            cl2=math.exp(-0.000232 * cp_ng_ml),
        )
        if changed:
            self._propofol_cp_ng_ml = cp_ng_ml
        return changed


class EtomidatePK(MammillaryPK):
    """Arden et al. 1986 three-compartment whole-blood model (mcg/mL).

    At age 57: V1 0.090 L/kg, CL 20.3, Q2 25.5, and Q3 18.8 mL/kg/min. V1
    falls 42% from 22 to 80 years and CL about 2 mL/kg/min per decade.
    Peripheral volumes 0.259 and 4.36 L/kg reproduce the reported Vss (4.7
    L/kg) and half-lives (0.93, 12.1, and 324 min). t1/2 ke0 1.6 min.
    Clearance is mostly hepatic (Van Hamme 1978); central volume and clearances
    scale with lean weight, while deep lipophilic V3 scales
    with total weight. Arden gave etomidate before surgery, so its clearances
    apply at awake cardiac output.
    """

    def __init__(self, patient: Patient):
        w, age = patient.weight, patient.age
        lean_w = _lean_weight_kg(patient)
        lean_flow = (lean_w / 70.0) ** 0.75
        cl_ml_kg_min = max(8.0, 20.3 - 0.2 * (age - 56.7))
        super().__init__(
            v1=_pregnancy_v1(patient, 0.120 * (1.0 - 0.00724 * (age - 22.0)) * lean_w),
            cl1=cl_ml_kg_min / 1000.0 * 70.0 * lean_flow * organ_clearance_scaler(patient, hepatic_fraction=0.9),
            v2=0.259 * lean_w,
            cl2=0.0255 * 70.0 * lean_flow,
            v3=4.36 * w,
            cl3=0.0188 * 70.0 * lean_flow,
            ke0=0.433,
        )


class KetaminePK(MammillaryPK):
    """Kamp et al. 2020 meta-analytical three-compartment model (mcg/mL).

    CL 84, Q2 161, and Q3 79 L/h and V1 25, V2 56, and V3 157 L at 70 kg;
    clearance is hepatic. Central volume and clearances scale with lean weight,
    while lipophilic V3 scales with total weight. AnaSim sets
    ke0 for loss of consciousness within about a minute. The cardiac-output site
    uses Kamp 2021's S-ketamine equilibration half-time of 2.28 min; its transfer
    to total racemic ketamine concentrations is an assumption. The separate HR
    half-time of 0.7 min is calibrated to peak HR within the first 3 min
    (Bovill and Dundee 1972). Most
    source studies were in awake volunteers, so its clearances apply at awake
    cardiac output.
    """

    def __init__(self, patient: Patient):
        lean_size = _lean_weight_kg(patient) / 70.0
        tbw_size = patient.weight / 70.0
        cl1 = 84.0 / 60.0 * lean_size**0.75 * organ_clearance_scaler(patient, hepatic_fraction=0.9)
        super().__init__(
            v1=_pregnancy_v1(patient, 25.0 * lean_size),
            cl1=cl1,
            v2=56.0 * lean_size,
            cl2=161.0 / 60.0 * lean_size**0.75,
            v3=157.0 * tbw_size,
            cl3=79.0 / 60.0 * lean_size**0.75,
            ke0=0.5,
            additional_effect_sites=(("ce_cv", math.log(2.0) / 2.28), ("ce_hr", math.log(2.0) / 0.7)),
        )


class DantrolenePK(MammillaryPK):
    """Two-compartment model from the Ryanodex label (DailyMed 2024; mcg/mL).

    At 70 kg, CL is 2.5 L/h and total volume is 36.4 L; V1 gives the label's
    9 mcg/mL peak after 2.5 mg/kg. Q and the 45 s effect-site half-time are
    assumed.
    """

    def __init__(self, patient: Patient):
        size = patient.weight / 70.0
        super().__init__(
            v1=_pregnancy_v1(patient, 19.45 * size), v2=16.95 * size,
            cl1=2.5 / 60.0 * size**0.75 * organ_clearance_scaler(patient, hepatic_fraction=1.0),
            cl2=0.4 * size**0.75, ke0=math.log(2.0) / 0.75,
            cl1_co_exponent=0.2,
        )


class AmiodaronePK(MammillaryPK):
    """Korth-Bradley 1996 two-compartment IV model (245 VT/VF patients), mcg/mL.

    CL 0.22 and Q 0.71 L/h/kg and V1 0.30 L/kg scale with lean weight,
    while lipophilic V2 (10 L/kg) scales with total weight. Calibrated effect
    half-times are 15 s for solvent-mediated bolus vasodilation and 18 min for
    sinus slowing (reproducing the time courses in Munoz 1988), plus 2.5 min
    for contractility and antiarrhythmic action.
    """

    def __init__(self, patient: Patient):
        w = patient.weight
        lean_w = _lean_weight_kg(patient)
        lean_flow = (lean_w / 70.0) ** 0.75
        super().__init__(
            v1=_pregnancy_v1(patient, 0.30 * lean_w), v2=10.0 * w,
            cl1=0.22 * 70.0 / 60.0 * lean_flow * organ_clearance_scaler(patient, hepatic_fraction=1.0),
            cl2=0.71 * 70.0 / 60.0 * lean_flow, ke0=math.log(2.0) / 2.5,
            additional_effect_sites=(("ce_cv", math.log(2.0) / 0.25), ("ce_hr", math.log(2.0) / 18.0)),
            cl1_co_exponent=0.2,
        )


class _EpiduralAbsorptionPK(MammillaryPK):
    """Mammillary local anesthetic PK with dual fast/slow extravascular epidural depots.

    Parallel biphasic epidural absorption follows Burm 1987, Emanuelsson 1997,
    and Olofsen 2008. Absorption requires systemic perfusion (`circulation`) so
    zero-flow arrest halts extravascular wash-in while conserving total drug mass.
    """

    _EPI_F_FAST: float = 0.30
    _EPI_F_SLOW: float = 0.65
    _EPI_T_HALF_FAST_MIN: float = 8.0
    _EPI_T_HALF_SLOW_MIN: float = 240.0
    _EPI_VASOCONSTRICTOR_SCALE: float = 0.65

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epi_fast_mg = 0.0
        self._epi_fast_epi_mg = 0.0  # Fast-pool fraction co-administered with epinephrine
        self._epi_slow_mg = 0.0

    @property
    def epidural_depot_mg(self) -> float:
        """Unabsorbed epidural local anesthetic mass remaining in mg."""
        return self._epi_fast_mg + self._epi_fast_epi_mg + self._epi_slow_mg

    def add_epidural_dose(self, dose_mg: float, *, with_epinephrine: bool = False) -> None:
        """Deposit an extravascular epidural dose (mg) into the biphasic absorption pools."""
        if dose_mg <= 0.0:
            return
        fast = dose_mg * self._EPI_F_FAST
        slow = dose_mg * self._EPI_F_SLOW
        if with_epinephrine:
            self._epi_fast_epi_mg += fast
        else:
            self._epi_fast_mg += fast
        self._epi_slow_mg += slow

    def has_mass(self) -> bool:
        return (
            super().has_mass()
            or self._epi_fast_mg > 1e-12
            or self._epi_fast_epi_mg > 1e-12
            or self._epi_slow_mg > 1e-12
        )

    def reset(self) -> None:
        super().reset()
        self._epi_fast_mg = 0.0
        self._epi_fast_epi_mg = 0.0
        self._epi_slow_mg = 0.0

    def step(self, dt_sec: float, input_rate_per_sec: float) -> PKState:
        absorbed_rate = 0.0
        if dt_sec > 0.0 and self.circulation > 0.0 and (
            self._epi_fast_mg > 0.0 or self._epi_fast_epi_mg > 0.0 or self._epi_slow_mg > 0.0
        ):
            dt_eff_min = (dt_sec / 60.0) * self.circulation
            k_fast = math.log(2.0) / self._EPI_T_HALF_FAST_MIN
            k_slow = math.log(2.0) / self._EPI_T_HALF_SLOW_MIN
            absorbed_mg = 0.0
            if self._epi_fast_mg > 0.0:
                df = self._epi_fast_mg * -math.expm1(-k_fast * dt_eff_min)
                self._epi_fast_mg = max(0.0, self._epi_fast_mg - df)
                if self._epi_fast_mg < 1e-12:
                    self._epi_fast_mg = 0.0
                absorbed_mg += df
            if self._epi_fast_epi_mg > 0.0:
                dfe = self._epi_fast_epi_mg * -math.expm1(-k_fast * self._EPI_VASOCONSTRICTOR_SCALE * dt_eff_min)
                self._epi_fast_epi_mg = max(0.0, self._epi_fast_epi_mg - dfe)
                if self._epi_fast_epi_mg < 1e-12:
                    self._epi_fast_epi_mg = 0.0
                absorbed_mg += dfe
            if self._epi_slow_mg > 0.0:
                ds = self._epi_slow_mg * -math.expm1(-k_slow * dt_eff_min)
                self._epi_slow_mg = max(0.0, self._epi_slow_mg - ds)
                if self._epi_slow_mg < 1e-12:
                    self._epi_slow_mg = 0.0
                absorbed_mg += ds
            absorbed_rate = absorbed_mg / dt_sec
        return super().step(dt_sec, input_rate_per_sec + absorbed_rate)


class LidocainePK(_EpiduralAbsorptionPK):
    """Foong et al. 2025 three-compartment model from surgical patients (mcg/mL).

    CL 45.9, Q2 142, and Q3 5.81 L/h and V1 25.2, V2 44.4, and V3 29.3 L for
    70 kg total and 56.1 kg fat-free mass (Table IV). Fat-free mass scales V1
    and the clearances, and total mass scales V2 and V3. Clearance is hepatic
    and was measured during surgery. Flow scaling lowers clearance and slows
    distribution in heart failure (Stenson 1971; Thomson 1973); peripheral
    volumes remain fixed as a model approximation. Renal impairment leaves
    parent-drug disposition unchanged (Thomson 1973). Cirrhosis expands
    steady-state volume by ~75% and lowers plasma clearance (Thomson 1973).
    Awake clearance and post-distribution concentrations match Rowland 1971
    and Tucker and Boas 1971, and V1 includes first-pass lung uptake. Foong's
    fluid effect on V2, postoperative clearance change, and metabolites are
    omitted. AnaSim sets ke0 so the analgesic site peaks 3 min after a bolus,
    the best time to give it before intubation (Tam 1987). Biphasic epidural
    absorption parameters follow Burm 1987 (F1 0.38, t1/2 9.3 min; F2 0.58,
    t1/2 82 min).
    """

    _EPI_F_FAST = 0.38
    _EPI_F_SLOW = 0.58
    _EPI_T_HALF_FAST_MIN = 9.3
    _EPI_T_HALF_SLOW_MIN = 82.0
    _EPI_VASOCONSTRICTOR_SCALE = 0.60

    def __init__(self, patient: Patient):
        size = patient.weight / 70.0
        lean_size = janmahasatian_ffm(patient.weight, patient.height, patient.sex == "male") / 56.1
        flow = lean_size**0.75 / ANESTHETIZED_OUTPUT
        peripheral_vd = hepatic_vd_multiplier(patient, 1.95)
        super().__init__(
            v1=_pregnancy_v1(patient, 25.2 * lean_size * hepatic_vd_multiplier(patient, 1.16)),
            cl1=45.9 / 60.0 * flow * organ_clearance_scaler(patient, hepatic_fraction=0.9),
            v2=44.4 * size * peripheral_vd,
            cl2=142.0 / 60.0 * flow,
            v3=29.3 * size * peripheral_vd,
            cl3=5.81 / 60.0 * flow,
            ke0=0.6,
            # Cardiac and CNS sites with calibrated 30 s half-times.
            additional_effect_sites=(("ce_cv", math.log(2.0) / 0.5), ("ce_neuro", math.log(2.0) / 0.5)),
        )


class BupivacainePK(_EpiduralAbsorptionPK):
    """Olofsen et al. 2008 arterial three-compartment bupivacaine PK (mcg/mL, input mg).

    Anchored to the Olofsen 2008 arterial levobupivacaine/ropivacaine joint model
    at age 50 (V1 6.85 L, CL 0.358 L/min, V2 11.8 L, Q2 1.04 L/min, V3 48.6 L,
    Q3 0.356 L/min, Vdss 67.2 L, age exponents -0.193 on V1 and +0.579 on k12),
    consistent with racemic bupivacaine disposition in Burm 1987 and Burm 1994
    (Vdss 68-70 L, CL 0.36-0.52 L/min) and volunteer arterial peaks in Knudsen 1997.
    Intermediate hepatic extraction scales with hepatic function and cardiac output
    (cl1_co_exponent 0.45). Biphasic epidural absorption follows Burm 1987
    (F1 0.28, t1/2 7.0 min; F2 0.66, t1/2 362 min).
    """

    _EPI_F_FAST = 0.28
    _EPI_F_SLOW = 0.66
    _EPI_T_HALF_FAST_MIN = 7.0
    _EPI_T_HALF_SLOW_MIN = 362.0
    _EPI_VASOCONSTRICTOR_SCALE = 0.68

    def __init__(self, patient: Patient):
        size = patient.weight / 70.0
        lean_size = janmahasatian_ffm(patient.weight, patient.height, patient.sex == "male") / 56.1
        age_ratio = clamp(patient.age, 20.0, 85.0) / 50.0
        v1_age = age_ratio ** -0.193
        k12_age = age_ratio ** 0.579
        v1_ref = 6.85 * lean_size * v1_age
        cl1 = 0.358 * (lean_size ** 0.75) * organ_clearance_scaler(patient, hepatic_fraction=0.90)
        q2 = 6.85 * 0.152 * k12_age * (lean_size ** 0.75)
        v2 = (6.85 * 0.152 / 0.0884) * lean_size
        q3 = 6.85 * 0.0519 * (lean_size ** 0.75)
        v3 = (6.85 * 0.0519 / 0.00732) * size
        super().__init__(
            v1=_pregnancy_v1(patient, v1_ref),
            cl1=cl1,
            v2=v2,
            cl2=q2,
            v3=v3,
            cl3=q3,
            ke0=math.log(2.0) / 0.5,
            cl1_co_exponent=0.45,
            additional_effect_sites=(("ce_cv", math.log(2.0) / 0.5), ("ce_neuro", math.log(2.0) / 0.5)),
        )


class RopivacainePK(_EpiduralAbsorptionPK):
    """Olofsen et al. 2008 arterial three-compartment ropivacaine PK (mcg/mL, input mg).

    Uses the Olofsen 2008 arterial joint model at age 50 (V1 6.85 L, CL 0.358 L/min,
    V2 11.8 L, Q2 1.04 L/min, V3 26.0 L, Q3 0.356 L/min, Vdss 44.6 L), matching
    Lee 1989 (Vss 42 L) and Emanuelsson 1997 (arterial CL 0.338 L/min, Vss 36 L)
    and Knudsen 1997 arterial concentrations. Biphasic epidural absorption is
    calibrated between Emanuelsson 1997 (f1 0.52, t1/2 14 min; f2 0.48, t1/2 252 min)
    and Olofsen 2008 (t1/2,a1 9.7 min, t1/2,a2 237 min).
    """

    _EPI_F_FAST = 0.44
    _EPI_F_SLOW = 0.50
    _EPI_T_HALF_FAST_MIN = 12.0
    _EPI_T_HALF_SLOW_MIN = 240.0
    _EPI_VASOCONSTRICTOR_SCALE = 0.72

    def __init__(self, patient: Patient):
        size = patient.weight / 70.0
        lean_size = janmahasatian_ffm(patient.weight, patient.height, patient.sex == "male") / 56.1
        age_ratio = clamp(patient.age, 20.0, 85.0) / 50.0
        v1_age = age_ratio ** -0.193
        k12_age = age_ratio ** 0.579
        v1_ref = 6.85 * lean_size * v1_age
        cl1 = 0.358 * (lean_size ** 0.75) * organ_clearance_scaler(patient, hepatic_fraction=0.90)
        q2 = 6.85 * 0.152 * k12_age * (lean_size ** 0.75)
        v2 = (6.85 * 0.152 / 0.0884) * lean_size
        q3 = 6.85 * 0.0519 * (lean_size ** 0.75)
        v3 = (6.85 * 0.0519 / 0.0137) * size
        super().__init__(
            v1=_pregnancy_v1(patient, v1_ref),
            cl1=cl1,
            v2=v2,
            cl2=q2,
            v3=v3,
            cl3=q3,
            ke0=math.log(2.0) / 0.5,
            cl1_co_exponent=0.45,
            additional_effect_sites=(("ce_cv", math.log(2.0) / 0.5), ("ce_neuro", math.log(2.0) / 0.5)),
        )


class NaloxonePK(MammillaryPK):
    """Yassen et al. 2007b arterial two-compartment naloxone PK (ng/mL, input mcg).

    From 44 healthy volunteers (Table III): V1 12.1 L, V2 102 L (Vdss 114.1 L),
    CL 3.45 L/min, Q2 4.74 L/min (elimination t1/2 ~33 min), and biophase ke0
    0.106 min^-1 (t1/2,ke0 6.5 min). Rapid brain entry and short plasma half-life
    match Ngai 1976 and Dowling 2008, producing fast opioid reversal with risk
    of renarcotization after long-acting agonists. Clearance is predominantly
    hepatic glucuronidation and high-extraction flow-dependent.
    """

    def __init__(self, patient: Patient):
        lean_size = _lean_weight_kg(patient) / 70.0
        super().__init__(
            v1=_pregnancy_v1(patient, 12.1 * lean_size),
            v2=102.0 * lean_size,
            cl1=3.45 * (lean_size ** 0.75) * organ_clearance_scaler(patient, hepatic_fraction=0.95),
            cl2=4.74 * (lean_size ** 0.75),
            ke0=0.106,
            cl1_co_exponent=0.7,
        )


class FlumazenilPK(MammillaryPK):
    """Two-compartment IV flumazenil PK grounded in Roncari 1986 and Klotz 1988 (ng/mL, input mcg).

    Matches Roncari 1986 pooled IV estimates in healthy men (CLp 15.5 mL/min/kg =
    1.085 L/min at 70 kg, Vss 1.06 L/kg = 74 L, t1/2,z 53 min, <0.1% unchanged
    renal excretion) with rapid central equilibration (ke0 1.4 min^-1, t1/2 ~30 s)
    reproducing the 30-48 s clinical arousal and 1-2 h duration of 2.5 mg IV
    reported in Klotz and Kanto 1988.
    """

    def __init__(self, patient: Patient):
        lean_size = _lean_weight_kg(patient) / 70.0
        super().__init__(
            v1=_pregnancy_v1(patient, 22.0 * lean_size),
            v2=52.0 * lean_size,
            cl1=1.085 * (lean_size ** 0.75) * organ_clearance_scaler(patient, hepatic_fraction=0.98),
            cl2=2.0 * (lean_size ** 0.75),
            ke0=1.4,
            cl1_co_exponent=0.7,
        )


class MethyleneBluePK(MammillaryPK):
    """Calibrated two-compartment IV methylene blue PK (mcg/mL, input mg).

    Calibrated so a 1-2 mg/kg IV dose (Guay 2009) drives erythrocyte
    NADPH-methemoglobin reductase clearance substantially over 1-2 h while
    capturing transient plasma dye peak and multi-hour elimination.
    """

    def __init__(self, patient: Patient):
        lean_size = _lean_weight_kg(patient) / 70.0
        clearance_scale, renal_fraction = renal_capacity_scaling(
            patient, 0.35, hepatic_fraction=0.65,
        )
        super().__init__(
            v1=_pregnancy_v1(patient, 18.0 * lean_size),
            v2=90.0 * lean_size,
            cl1=0.35 * (lean_size ** 0.75) * clearance_scale,
            cl2=1.1 * (lean_size ** 0.75),
            ke0=0.35,
            renal_fraction=renal_fraction,
            cl1_co_exponent=0.4,
        )


# Rocuronium -------------------------------------------------------------------

def rocuronium_adjusted_weight(patient: Patient) -> float:
    """Adjusted dosing/distribution weight (kg) for rocuronium in obesity (Meyhoff 2009)."""
    pbw = patient.predicted_body_weight()
    threshold = max(pbw, 28.0 * (patient.height / 100.0) ** 2)
    return threshold + 0.30 * (patient.weight - threshold) if patient.weight > threshold else patient.weight


class RocuroniumPK(MammillaryPK):
    """Kleijn 2011 two-compartment population PK (446 subjects, 1-91 years).

    Clearance falls 0.68% per year of age from 43, V1 falls with creatinine
    clearance, and V2 rises with age. Of six rocuronium models, this one and
    Magorian's predicted measured concentrations within accepted limits
    (Sasakawa 2016); clearance at 65-88 years was 25% lower (Varrique 2016).
    Clearance is mainly hepatic uptake and biliary excretion: 12-22% of 0.6
    mg/kg appears unchanged in urine, renal failure changes clearance little,
    and cirrhosis enlarges the volumes 30-40% (Khuenl-Brady and Sparr 1996). Obesity
    scaling follows ideal/adjusted body weight (Meyhoff 2009). The effect site
    is Kleijn's, without sevoflurane.
    """

    RENAL_FRACTION = 0.2

    def __init__(self, patient: Patient):
        size = rocuronium_adjusted_weight(patient) / 70.0
        age = patient.age
        crcl = patient.creatinine_clearance()
        params = {
            "v1": _pregnancy_v1(patient, 4.73 * size * math.exp(-0.00143 * (crcl - 119.0))),
            "cl1": 0.269 * size**0.75 * (1.0 - 0.00678 * (age - 43.0)),
            "v2": 6.76 * size * math.exp(0.00613 * (age - 43.0)),
            "cl2": 0.279 * size**0.75,
        }
        # Cirrhosis enlarges distribution volumes (Khuenl-Brady and Sparr 1996;
        # Magorian 1995) and prolongs elimination and recovery with lower clearance
        # (~28% in van Miert 1997); V1/V2 multipliers and hepatic scaling are calibrated.
        params["v1"] *= hepatic_vd_multiplier(patient, severe_multiplier=1.18)
        params["v2"] *= hepatic_vd_multiplier(patient, severe_multiplier=1.35)
        clearance_scale, renal_fraction = renal_capacity_scaling(
            patient, self.RENAL_FRACTION, hepatic_fraction=0.7
        )
        params["cl1"] *= clearance_scale
        # Kleijn's patients were anesthetized, so his parameters already reflect
        # anesthetic cardiac output.
        super().__init__(
            ke0=0.134 * size**-0.25, renal_fraction=renal_fraction, **_NO_CO_COVARIATE, **params,
        )

    def update_hemodynamics(
        self, v_ratio: float, co_ratio: float, renal_perfusion: float = 1.0,
        blood_volume_l: float | None = None, temp_c: float = 37.0,
        *, temp_met: float | None = None, temp_dist: float | None = None,
    ) -> None:
        # Extrapolated from vecuronium in Caldwell 2000: clearance falls 11.3%/°C
        # during hypothermia (Q10 ~3.0), steeper than general hepatic Q10 = 2.0.
        clamped_t = clamp(temp_c, 18.0, 38.5)
        nmba_temp_met = 3.0 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0
        super().update_hemodynamics(
            v_ratio, co_ratio, renal_perfusion,
            blood_volume_l=blood_volume_l, temp_c=temp_c,
            temp_met=nmba_temp_met, temp_dist=temp_dist,
        )


# Vasoactive drugs -------------------------------------------------------------

class NorepinephrinePK(MammillaryPK):
    """Li 2024 arterial PK with secretion and propofol-dependent clearance.

    The clinical effect-site rate includes circulation and response timing.
    """

    # Li et al. 2024: clearance multiplier exp(theta * Cp / 100), about 12% lower at 3.5 µg/mL.
    PROPOFOL_CL_THETA = -3.57

    def __init__(self, patient: Patient):
        lean_size = _lean_weight_kg(patient) / 70.0
        age = patient.age
        self.endogenous_ug_min = 0.4977 * lean_size**0.75
        super().__init__(
            v1=_pregnancy_v1(patient, 2.4 * lean_size),
            # Li 2024 Eq. 2: the age covariate is centered on 35 years.
            cl1=2.1 * math.exp(-0.344 / 100 * (age - 35)) * lean_size**0.75,
            v2=3.6 * lean_size,
            cl2=0.6 * lean_size**0.75,
            ke0=0.55,
            basal_input_rate=self.endogenous_ug_min / 60.0,
            **_NO_CO_COVARIATE,
        )
        self.equilibrate_endogenous()

    @staticmethod
    def resting_concentration(age: float) -> float:
        """Resting endogenous arterial concentration (ng/mL) before drug covariates."""
        return 0.4977 / (2.1 * math.exp(-0.344 / 100 * (age - 35)))

    def equilibrate_endogenous(self) -> None:
        """Seed secretion without an exogenous infusion at the current clearance."""
        endogenous_conc = self.endogenous_ug_min / (self.cl1_base * self.cl1_scale)
        self.set_state_vector([endogenous_conc] * len(self.state_fields))

    def reset(self) -> None:
        super().reset()
        self.basal_input_rate = self.endogenous_ug_min / 60.0
        self.equilibrate_endogenous()

    def update_hemodynamics(
        self, v_ratio: float, co_ratio: float, renal_perfusion: float = 1.0,
        blood_volume_l: float | None = None, temp_c: float = 37.0,
        *, temp_met: float | None = None, temp_dist: float | None = None,
    ) -> None:
        super().update_hemodynamics(
            v_ratio, co_ratio, renal_perfusion,
            blood_volume_l=blood_volume_l, temp_c=temp_c,
            temp_met=temp_met, temp_dist=temp_dist,
        )
        # Keep resting endogenous NE invariant to hypothermic/hemodynamic clearance
        # scaling while preserving Li 2024 propofol clearance attenuation (cl1_scale).
        cl_ratio = self.cl1 / self.cl1_base if self.cl1_base > 0.0 else 1.0
        self.basal_input_rate = (self.endogenous_ug_min / 60.0) * cl_ratio

    def update_propofol(self, propofol_conc_ug_ml: float) -> None:
        """Set the clearance covariate, capping bolus peaks at 8 µg/mL."""
        cp = clamp(propofol_conc_ug_ml, 0.0, 8.0)
        self.cl1_scale = 1.0 if cp == 0.0 else math.exp(self.PROPOFOL_CL_THETA * cp / 100.0)


class EpinephrinePK(MammillaryPK):
    """Exogenous epinephrine, in ng/mL above the patient's endogenous baseline.

    Young-adult arterial clearance is calibrated between Esler 1995 and
    Ensinger 1992a, and its decline with age is from Esler 1995. Mixing volume
    and effect-site timing are calibrated against adult infusion and bolus data.
    """

    AGE_CL_SLOPE = -0.0053  # Esler 1995: 2.24 vs 2.80 L/min from 20-30 to 60-75 yr.

    def __init__(self, patient: Patient):
        lean_w = _lean_weight_kg(patient)
        # Esler: 2.80 L/min (~40 mL/kg/min at 70 kg); Ensinger 1992a Table 2: 51 mL/kg/min.
        # Keep linear per-kg scaling up to 70 kg (matching Takahashi 2002) and
        # allometric lean scaling above 70 kg to prevent supraphysiologic obesity CL.
        cl_mass = lean_w if lean_w <= 70.0 else 70.0 * (lean_w / 70.0) ** 0.75
        cl1 = 0.046 * cl_mass * math.exp(self.AGE_CL_SLOPE * (patient.age - 25.0))
        super().__init__(
            v1=_pregnancy_v1(patient, 0.021055536419796666 * lean_w), cl1=cl1, ke0=2.2, **_NO_CO_COVARIATE,
        )


class PhenylephrinePK(MammillaryPK):
    """FDA NDA 203826 two-compartment PK (t1/2 alpha 2.3 min, beta 53 min)."""

    def __init__(self, patient: Patient):
        lean_size = _lean_weight_kg(patient) / 70.0
        lean_flow = lean_size**0.75
        v1_ref = 20.4
        # Minimal hysteresis: bolus pressor effect peaks near 1 min.
        super().__init__(
            v1=_pregnancy_v1(patient, v1_ref * lean_size),
            cl1=0.124 * v1_ref * lean_flow,
            v2=(0.155 / 0.0314) * v1_ref * lean_size,
            cl2=0.155 * v1_ref * lean_flow,
            ke0=2.0,
        )


class VasopressinPK(MammillaryPK):
    """DailyMed 2025 vasopressin label: Vd 0.14 L/kg, CL 9-25 mL/min/kg (midpoint 17)."""

    def __init__(self, patient: Patient):
        lean_w = _lean_weight_kg(patient)
        lean_flow = (lean_w / 70.0) ** 0.75
        super().__init__(
            v1=_pregnancy_v1(patient, 0.14 * lean_w), cl1=0.017 * 70.0 * lean_flow, ke0=0.25, cl1_co_exponent=0.5,
        )


class DobutaminePK(MammillaryPK):
    """Kates and Leier 1978: CL 2.35 L/min/m², Vd 0.20 L/kg, t1/2 about 2 min."""

    def __init__(self, patient: Patient):
        super().__init__(
            v1=_pregnancy_v1(patient, 0.20 * _lean_weight_kg(patient)),
            cl1=2.35 * patient.bsa,
            ke0=0.7,
            cl1_co_exponent=0.3,
        )


class MilrinonePK(MammillaryPK):
    """Calibrated two-compartment IV milrinone PK (ng/mL) matching DailyMed 2026
    Vdss (0.45 L/kg) and CL (0.13 L/kg/h) and Baim 1983 bolus levels (mean +/- SEM:
    392 +/- 23 ng/mL at 5 min, 180 +/- 19 ng/mL at 30 min; reported t1/2 136 min).
    """

    def __init__(self, patient: Patient):
        lean_w = _lean_weight_kg(patient)
        lean_flow = (lean_w / 70.0) ** 0.75
        clearance_scale, renal_fraction = renal_capacity_scaling(patient, 0.9)
        super().__init__(
            v1=_pregnancy_v1(patient, 0.155 * lean_w),
            cl1=0.13 * 70.0 / 60.0 * lean_flow * clearance_scale,
            v2=0.295 * lean_w,
            cl2=0.28 * lean_flow,
            ke0=0.25,
            renal_fraction=renal_fraction,
        )


# Adrenergic and muscarinic antagonists ----------------------------------------

class EsmololPK(MammillaryPK):
    """Sum et al. 1983 at 400 mcg/kg/min (concentrations in mcg/mL).

    Vc 0.867 L/kg, distribution and elimination t1/2 2.03 and 9.19 min, and
    k21/k12 2.66 give CL 258 mL/min/kg and Varea 3.42 L/kg. Blood esterases
    clear esmolol, so clearance is independent of cardiac output and organ
    function, scaling with lean body mass in obesity.
    """

    def __init__(self, patient: Patient):
        lean_w = _lean_weight_kg(patient)
        params = _from_rate_constants(0.867 * lean_w, 0.2977, 0.0325, 0.0865, 0.0, 0.0)
        params["v1"] = _pregnancy_v1(patient, params["v1"])
        super().__init__(ke0=0.5, cl1_co_exponent=0.0, intravascular_clearance=True, **params)


class LabetalolPK(MammillaryPK):
    """Two-compartment labetalol PK (concentrations in ng/mL).

    Clearance declines with age (19.4 mL/min/kg at age 32 to 13.9 at 67;
    Abernethy 1987) and is mainly hepatic glucuronidation (organ-impairment
    kinetics informed by Hafsa 2022). Calibrated two-compartment distribution
    (t1/2 ~2 min, terminal t1/2 ~3 h at age 32) yields ~130 ng/mL 10 min after
    0.5 mg/kg IV, matching the clinical profiles modeled in Hafsa 2022.
    """

    def __init__(self, patient: Patient):
        w = patient.weight
        lean_w = _lean_weight_kg(patient)
        lean_flow = (lean_w / 70.0) ** 0.75
        cl_ml_min_kg = max(8.0, 19.4 - 0.157 * (patient.age - 32.0))
        cl1 = cl_ml_min_kg / 1000.0 * 70.0 * lean_flow * organ_clearance_scaler(patient, hepatic_fraction=0.9)
        super().__init__(
            v1=_pregnancy_v1(patient, 0.4 * lean_w), cl1=cl1, v2=4.0 * w, cl2=0.1 * 70.0 * lean_flow, ke0=0.3,
        )


class GlycopyrrolatePK(MammillaryPK):
    """Du et al. 2025 three-compartment model at 61.3 kg (concentrations in ng/mL).

    About 70% of a dose is excreted unchanged in urine (Ali-Melkkilä 1993).
    The published ke0 had 98.5% shrinkage; AnaSim uses 0.14 min^-1 so the HR
    peak trails atropine's onset while reaching therapeutic effect within minutes.
    """

    def __init__(self, patient: Patient):
        lean_size = _lean_weight_kg(patient) / 61.3
        clearance_scale, renal_fraction = renal_capacity_scaling(patient, 0.7)
        super().__init__(
            v1=_pregnancy_v1(patient, 10.38 * lean_size),
            cl1=49.76 / 60.0 * lean_size**0.75 * clearance_scale,
            renal_fraction=renal_fraction,
            v2=21.41 * lean_size,
            cl2=6.54 / 60.0 * lean_size**0.75,
            v3=11.0 * lean_size,
            cl3=31.62 / 60.0 * lean_size**0.75,
            ke0=0.14,
        )


class NeostigminePK(MammillaryPK):
    """Cronnelly 1979 two-compartment model (concentrations in ng/mL).

    Normal-renal Table 1 means: V1 0.37 L/kg, Vdss 1.40 L/kg (V2 1.03 L/kg),
    CL 16.7 mL/kg/min, t1/2_alpha 3.5 min, t1/2_beta 79.8 min. Anephric CL
    fell to 7.8 mL/kg/min (53% renal elimination; t1/2_beta 181 min).
    Cardiac muscarinic ke0 0.16 min^-1 aligns onset (~7 min) with glycopyrrolate
    (Mirakhur 1977), while slower ce_nmj 0.085 min^-1 governs neuromuscular reversal
    (Kirkegaard 2002).
    """

    def __init__(self, patient: Patient):
        lean_w = _lean_weight_kg(patient)
        lean_flow = (lean_w / 70.0) ** 0.75
        clearance_scale, renal_fraction = renal_capacity_scaling(patient, 0.53)
        super().__init__(
            v1=_pregnancy_v1(patient, 0.37 * lean_w),
            cl1=0.0167 * 70.0 * lean_flow * clearance_scale,
            renal_fraction=renal_fraction,
            v2=1.03 * lean_w,
            cl2=0.03924 * 70.0 * lean_flow,
            ke0=0.16,
            additional_effect_sites=(("ce_nmj", 0.085),),
            **_NO_CO_COVARIATE,
        )


# Anticoagulation and reversal -------------------------------------------------

class HeparinPK(MammillaryPK):
    """Two-compartment unfractionated heparin PK (concentrations in U/L; 1000 U/L = 1 U/mL).

    Two-compartment structure follows Delavenne 2017 (linear CPB model, Vc = 3.1 L),
    with central volume calibrated to ~62 mL/kg lean mass and expanding dynamically
    with blood-volume hemodilution via update_hemodynamics(). A reversible peripheral
    compartment (V2) stores endothelial/protein-bound heparin and drives post-reversal
    heparin rebound (Teoh 2004). Saturable endothelial/reticuloendothelial clearance
    (demonstrated across 25-75 U/kg in healthy adults in Bjornsson 1982) combined with
    linear renal elimination is calibrated so apparent half-life lengthens from ~30 min
    at 25 U/kg to ~60 min at 100 U/kg and ~105-150 min at 300 U/kg.
    """

    KM_SAT_U_L = 160.0  # 0.16 U/mL Michaelis-Menten constant for saturable clearance

    def __init__(self, patient: Patient):
        lean_w = _lean_weight_kg(patient)
        lean_flow = (lean_w / 70.0) ** 0.75
        lin_scale, self._lin_renal_frac = renal_capacity_scaling(patient, 0.7)
        self._cl_linear_base = 0.006 * lean_flow * lin_scale
        self._vmax_sat_u_min = 30.0 * lean_flow
        self._cl_zero_base = self._cl_linear_base + self._vmax_sat_u_min / self.KM_SAT_U_L
        self._co_scale = 1.0
        self._temp_met = 1.0
        self._renal_perf = 1.0
        self._deficit_scale = 1.0
        super().__init__(
            v1=_pregnancy_v1(patient, 0.062 * lean_w),
            cl1=self._cl_zero_base,
            v2=0.032 * lean_w,
            cl2=0.036 * lean_flow,
            ke0=1.2,
            renal_fraction=0.0,
            cl1_co_exponent=0.4,
            distribution_co_exponent=0.5,
        )

    def reset(self) -> None:
        super().reset()
        self._co_scale = 1.0
        self._temp_met = 1.0
        self._renal_perf = 1.0
        self._deficit_scale = 1.0
        self.update_nonlinear_clearance()

    def update_hemodynamics(
        self, v_ratio: float, co_ratio: float, renal_perfusion: float = 1.0,
        blood_volume_l: float | None = None, temp_c: float = 37.0,
        *, temp_met: float | None = None, temp_dist: float | None = None,
    ) -> None:
        super().update_hemodynamics(
            v_ratio, co_ratio, renal_perfusion,
            blood_volume_l=blood_volume_l, temp_c=temp_c,
            temp_met=temp_met, temp_dist=temp_dist,
        )
        clamped_t = clamp(temp_c, 18.0, 38.5)
        self._temp_met = (
            temp_met
            if temp_met is not None
            else (2.0 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0)
        )
        self._co_scale = _pow(co_ratio, self.cl1_co_exponent)
        self._renal_perf = renal_perfusion
        deficit = min(1.0, v_ratio)
        self._deficit_scale = _pow(deficit, self.cl1_deficit_exponent) if deficit < 1.0 and self.cl1_deficit_exponent else 1.0
        self.update_nonlinear_clearance()

    def update_nonlinear_clearance(self) -> None:
        """Apply saturable endothelial/RES clearance (Bjornsson 1982; Cohen 1977)
        separately from linear predominantly renal elimination."""
        c1 = max(0.0, self.state.c1)
        sat_cl = (self._vmax_sat_u_min / (self.KM_SAT_U_L + c1)) * self._co_scale * self._temp_met
        lin_nonrenal = (1.0 - self._lin_renal_frac) * self._co_scale * self._temp_met
        lin_cl = self._cl_linear_base * (lin_nonrenal + self._lin_renal_frac * self._renal_perf)
        self.cl1 = (sat_cl + lin_cl) * self._deficit_scale
        self.cl1_scale = 1.0


class ProtaminePK(MammillaryPK):
    """One-compartment protamine sulfate PK (concentrations in mg/L = mcg/mL).

    One-compartment approximation calibrated to Butterworth 2002 post-CPB
    median Vd (5.4 L, ~77 mL/kg lean mass) and rapid elimination (median t1/2
    4.5 min post-CPB), clearing >95% within 20 minutes and exposing late
    peripheral heparin rebound.
    """

    def __init__(self, patient: Patient):
        lean_w = _lean_weight_kg(patient)
        lean_flow = (lean_w / 70.0) ** 0.75
        v1 = _pregnancy_v1(patient, 0.077 * lean_w)
        # 0.693 * 5.4 / 0.83 ~= 4.5 min elimination half-life at 70 kg.
        cl1 = 0.83 * lean_flow
        super().__init__(
            v1=v1,
            cl1=cl1,
            ke0=1.4,
            cl1_co_exponent=0.0,
            distribution_co_exponent=0.0,
        )


# Endocrine and metabolic ------------------------------------------------------

class InsulinPK(MammillaryPK):
    """Three-compartment regular IV insulin PK (concentrations in mU/L = µU/mL).

    Adapts Sherwin 1974 Fig. 1 Model A (porcine insulin in healthy adults,
    total-body-weight volumes V1 = 45, V2 = 17, V3 = 95 mL/kg; L01 = 0.251,
    L21 = 0.142, L31 = 0.042, L13 = 0.020 min^-1; MCR ~11.3 mL/kg/min) to lean
    body mass (keeping exogenous MCR per BSA roughly steady in obesity,
    Polonsky 1988) and scales clearance across pregnancy (Catalano 1998).
    Total clearance uses a calibrated 55% hepatic, 27% renal, and 18%
    peripheral split informed by Rabkin 1970 and Letiexhe 1993. Tracks exogenous
    regular insulin only; endogenous pancreatic secretion is integrated in
    GlucoseModel.
    """

    def __init__(self, patient: Patient):
        lean_w = _lean_weight_kg(patient)
        v1_base = 0.045 * lean_w
        v2 = 0.017 * lean_w
        v3 = 0.095 * lean_w
        cl2 = 0.142 * v1_base
        cl3 = 0.042 * v1_base
        clearance_scale, renal_fraction = renal_capacity_scaling(
            patient, 0.27, hepatic_fraction=0.55,
        )
        cl1 = 0.251 * v1_base * clearance_scale * patient.pregnancy.insulin_clearance
        super().__init__(
            v1=_pregnancy_v1(patient, v1_base),
            cl1=cl1,
            v2=v2,
            cl2=cl2,
            v3=v3,
            cl3=cl3,
            ke0=0.020,
            renal_fraction=renal_fraction,
            cl1_co_exponent=0.5,
            distribution_co_exponent=0.5,
        )
