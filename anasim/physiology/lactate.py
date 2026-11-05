"""Whole-body lactate kinetics and the associated acid load.

Lactate has a volume of distribution of 0.26 L/kg (Levraut 1998) and a resting
clearance of 12 mL/kg/min. With a resting concentration of 0.9 mmol/L, basal
production is 10.8 umol/kg/min (Revelly 2005; Grip 2015). Elimination follows
Michaelis-Menten kinetics, so apparent clearance falls as lactate rises.
Hepatic elimination capacity is proportional to liver function and to the
fraction of oxygen demand that is met.

Each mmol of endogenous lactate above the resting amount removes 1 mmol of
buffer base. Lactated Ringer's solution contains sodium lactate, which raises
lactate without adding acid and returns base when it is metabolized. The model
includes only the L-isomer (14 mmol/L), which blood gas analyzers measure.
D-lactate and the other ions in the fluid are omitted.
"""

import math

O2_L_PER_MMOL = 0.022414
DISTRIBUTION_L_PER_KG = 0.26
CLEARANCE_L_KG_MIN = 0.012
RESTING_MMOL_L = 0.9
CLEARANCE_KM_MMOL_L = 3.0  # Approximates the fall in apparent clearance in Grip 2015.
HEPATIC_SHARE = 0.6  # Assumed hepatic share of elimination capacity; the rest is extrahepatic.
LACTATED_RINGERS_MMOL_L = 14.0
# Lactate produced per mmol of unmet O2 demand, fitted to the relation between
# oxygen debt and lactate in canine hemorrhage (Dunham 1991).
ANAEROBIC_PER_O2 = 0.56
# Aerobic production rises with exogenous plasma epinephrine. The Hill curve
# is assumed; it gives threefold production at 1.9 ng/mL (Grip 2015) and a
# threshold near 0.2 ng/mL (Clutter 1980).
EPINEPHRINE_EMAX = 2.4
EPINEPHRINE_EC50_NG_ML = 0.6
# Sepsis raises production 2.4-fold (Revelly 2005). At the resulting
# concentration, saturable elimination alone lowers apparent clearance to the
# value in hyperlactatemic septic patients (Levraut 1998), so sepsis does not
# change elimination capacity.
SEPSIS_PRODUCTION = 1.4


class LactateModel:
    """Endogenous and infused lactate in one volume, with associated buffer base."""

    def __init__(self, weight_kg: float, hepatic_function: float = 1.0):
        self.volume_l = DISTRIBUTION_L_PER_KG * weight_kg
        self.hepatic_function = hepatic_function
        self.km_mmol = CLEARANCE_KM_MMOL_L * self.volume_l
        self.vmax_mmol_min = CLEARANCE_L_KG_MIN * weight_kg * (CLEARANCE_KM_MMOL_L + RESTING_MMOL_L)
        self.basal_mmol_min = CLEARANCE_L_KG_MIN * weight_kg * RESTING_MMOL_L
        capacity = self.vmax_mmol_min * (1.0 - HEPATIC_SHARE + HEPATIC_SHARE * hepatic_function)
        self.resting_mmol = self.basal_mmol_min * self.km_mmol / (capacity - self.basal_mmol_min)
        self.endogenous_mmol = self.resting_mmol
        self.infused_mmol = 0.0
        self.cleared_infused_mmol = 0.0
        self._crystalloid_ml = 0.0

    @property
    def concentration(self) -> float:
        """Plasma lactate, mmol/L."""
        return (self.endogenous_mmol + self.infused_mmol) / self.volume_l

    @property
    def base_removed_mmol(self) -> float:
        """Net buffer base removed since rest, including infused L-lactate metabolism."""
        return self.endogenous_mmol - self.resting_mmol - self.cleared_infused_mmol

    def step(self, dt: float, oxygen_deficit_l: float, supply_fraction: float,
             epinephrine_ng_ml: float = 0.0, sepsis_severity: float = 0.0,
             crystalloid_in_ml: float = 0.0) -> float:
        """Advance dt seconds and return net buffer base removed, mmol.

        Args:
            oxygen_deficit_l: Unmet O2 demand over the step, STPD L.
            supply_fraction: Share of O2 demand met, 0-1.
            epinephrine_ng_ml: Exogenous plasma epinephrine.
            sepsis_severity: 0-1.
            crystalloid_in_ml: Cumulative crystalloid given; the increase since
                the last step is lactated Ringer's.
        """
        if dt <= 0.0:
            return 0.0
        infused = max(0.0, crystalloid_in_ml - self._crystalloid_ml) * LACTATED_RINGERS_MMOL_L / 1000.0
        self._crystalloid_ml = crystalloid_in_ml
        minutes = dt / 60.0
        epinephrine = max(0.0, epinephrine_ng_ml) ** 2
        aerobic = self.basal_mmol_min * (
            1.0 + EPINEPHRINE_EMAX * epinephrine / (EPINEPHRINE_EC50_NG_ML ** 2 + epinephrine)
            + SEPSIS_PRODUCTION * sepsis_severity
        )
        anaerobic = ANAEROBIC_PER_O2 * max(0.0, oxygen_deficit_l) / O2_L_PER_MMOL
        supply = min(1.0, max(0.0, supply_fraction))
        capacity = self.vmax_mmol_min * (1.0 - HEPATIC_SHARE + HEPATIC_SHARE * self.hepatic_function * supply)
        endogenous_source = aerobic + anaerobic / minutes
        infused_source = infused / minutes
        total = self.endogenous_mmol + self.infused_mmol
        # Predict the midpoint amount, then freeze Michaelis-Menten clearance
        # there. Exponential updates preserve positive amounts and source balance.
        # Each expm1 result supplies both the eliminated fraction and its decay.
        rate = capacity / (self.km_mmol + total)
        half_loss = -math.expm1(-rate * minutes / 2.0)
        half_retained = half_loss / rate
        midpoint = total * (1.0 - half_loss) + (endogenous_source + infused_source) * half_retained
        rate = capacity / (self.km_mmol + midpoint)
        loss = -math.expm1(-rate * minutes)
        retained = loss / rate
        decay = 1.0 - loss
        before = self.endogenous_mmol
        infused_before = self.infused_mmol
        self.endogenous_mmol = self.endogenous_mmol * decay + endogenous_source * retained
        self.infused_mmol = infused_before * decay + infused_source * retained
        cleared_infused = infused_before + infused - self.infused_mmol
        self.cleared_infused_mmol += cleared_infused
        return self.endogenous_mmol - before - cleared_infused
