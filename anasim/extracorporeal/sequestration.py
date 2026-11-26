"""Saturable extracorporeal circuit drug sequestration (adsorption).

Parameterized against:
- Shekar 2012: 24 h ex-vivo whole-blood ECMO recovery (fentanyl 3%,
  midazolam 13%, morphine 103%).
- Lemaitre 2015: ex-vivo PMP circuit propofol (30% left at 30 min,
  11% at 5 h) and midazolam (54% at 30 min, 11% at 24 h).
- Morales Castro 2025: in-vivo adult VV ECMO (n = 11) propofol and fentanyl
  clearance increases on ECMO initiation and plateaus after ~8 hours as binding
  sites saturate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from anasim.core.utils import clamp01


@dataclass(frozen=True, slots=True)
class SequestrationSpec:
    """First-order saturable circuit adsorption parameters for one IV drug.

    adsorption_cl_l_min: initial adsorption clearance (L/min) at 4 L/min circuit flow
        on a fresh adult circuit (scales sublinearly with blood flow).
    capacity_mass: finite mass binding capacity in the drug's PK model mass units
        (c1 * L: mg for mcg/mL drugs; mcg for ng/mL drugs) so adsorption saturates
        naturally during sustained infusions.
    coat_tau_s: time constant (s) over which plasma proteins/lipids passivate
        circuit surfaces even without drug exposure (~8 h plateau per Morales Castro 2025).
    """

    adsorption_cl_l_min: float
    capacity_mass: float
    coat_tau_s: float = 14400.0  # 4 h tau -> ~86% passivated by 8 h


# Model mass units (c1 * V1):
# - mcg/mL * L = mg (propofol, ketamine, etomidate, lidocaine, amiodarone, roc)
# - ng/mL * L = mcg (fentanyl, remi, midazolam, morphine, hydromorphone)
SEQUESTRATION_SPECS: dict[str, SequestrationSpec] = {
    # Lemaitre 2015: rapid propofol adsorption (70% ex-vivo loss in 30 min);
    # Morales Castro 2025: transient in-vivo clearance surge plateauing ~8 h.
    "propofol": SequestrationSpec(adsorption_cl_l_min=0.42, capacity_mass=180.0, coat_tau_s=12000.0),
    # Shekar 2012: fentanyl 3% recovery at 24 h ex vivo; Morales Castro 2025 in-vivo surge.
    "fentanyl": SequestrationSpec(adsorption_cl_l_min=0.45, capacity_mass=350.0, coat_tau_s=14400.0),
    # Shekar 2012; Lemaitre 2015: midazolam 54% at 30 min, 11-13% at 24 h ex vivo.
    "midazolam": SequestrationSpec(adsorption_cl_l_min=0.28, capacity_mass=18000.0, coat_tau_s=14400.0),
    # Lipophilic cardiac/anesthetic agents: calibrated extrapolations for PVC/PMP binding.
    "amiodarone": SequestrationSpec(adsorption_cl_l_min=0.18, capacity_mass=120.0, coat_tau_s=14400.0),
    "ketamine": SequestrationSpec(adsorption_cl_l_min=0.10, capacity_mass=60.0, coat_tau_s=14400.0),
    "etomidate": SequestrationSpec(adsorption_cl_l_min=0.08, capacity_mass=15.0, coat_tau_s=14400.0),
    "lidocaine": SequestrationSpec(adsorption_cl_l_min=0.06, capacity_mass=80.0, coat_tau_s=14400.0),
    # Morphine (Shekar 2012: 103% recovery), remifentanil (hydrophilic ester), NMBAs, pressors:
    # negligible circuit sequestration (omitted -> 0.0).
}


class CircuitSequestration:
    """Tracks adsorbed drug mass per fresh extracorporeal circuit."""

    def __init__(self) -> None:
        self.adsorbed_mass: dict[str, float] = {}
        self.circuit_age_s: float = 0.0

    def reset(self) -> None:
        """Reset binding state for a newly primed circuit."""
        self.adsorbed_mass.clear()
        self.circuit_age_s = 0.0

    def step(
        self,
        dt: float,
        blood_flow_l_min: float,
        pk_models: dict,
        *,
        membrane_area_m2: float = 1.8,
        active: bool = True,
    ) -> None:
        """Remove adsorbed drug from central PK compartments while conserving circuit mass."""
        if not active or dt <= 0.0 or blood_flow_l_min <= 1e-6:
            return
        self.circuit_age_s += dt
        area_scale = clamp01(membrane_area_m2 / 1.8) if membrane_area_m2 < 1.8 else (membrane_area_m2 / 1.8) ** 0.7
        flow_scale = (min(blood_flow_l_min, 7.0) / 4.0) ** 0.35
        dt_min = dt / 60.0
        for key, spec in SEQUESTRATION_SPECS.items():
            model = pk_models.get(key)
            if model is None:
                continue
            c1 = model.state.c1
            if c1 <= 0.0 or model.v1 <= 0.0:
                continue
            bound = self.adsorbed_mass.get(key, 0.0)
            cap = spec.capacity_mass * max(0.3, area_scale)
            headroom = clamp01(1.0 - bound / cap)
            if headroom <= 1e-6:
                continue
            passivation = math.exp(-self.circuit_age_s / spec.coat_tau_s)
            eff_cl = spec.adsorption_cl_l_min * area_scale * flow_scale * headroom * passivation
            if eff_cl <= 1e-9:
                continue
            # Exact exponential step on C1 so sequestration is step-size invariant and never negative.
            removed_frac = -math.expm1(-eff_cl * dt_min / model.v1)
            mass_c1 = c1 * model.v1
            removed_mass = min(mass_c1 * removed_frac, cap - bound)
            if removed_mass > 0.0:
                model.state.c1 = max(0.0, c1 - removed_mass / model.v1)
                self.adsorbed_mass[key] = bound + removed_mass
