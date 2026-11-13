"""Body water, electrolytes, protein, calcium binding, and renal net acid balance."""

import math
from dataclasses import dataclass

from .blood_chemistry import plasma_bicarbonate, standard_base_excess
from .plasma_buffers import PlasmaBuffers


@dataclass(frozen=True, slots=True)
class FluidComposition:
    """Concentrations per liter of fluid or packed-cell supernatant."""

    sodium: float
    potassium: float
    chloride: float
    calcium: float = 0.0
    magnesium: float = 0.0
    phosphate: float = 0.0
    albumin: float = 0.0  # g/L
    lactate: float = 0.0
    citrate: float = 0.0
    l_lactate_fraction: float = 1.0
    carbon: float = 0.0  # Bicarbonate plus dissolved CO2

    @property
    def sid(self) -> float:
        return (self.sodium + self.potassium + 2.0 * (self.calcium + self.magnesium)
                - self.chloride - self.lactate - 3.0 * self.citrate)


# A 3-week-old SAGM unit (Rabcuka 2025, Table 3): pH 6.5 and CO2 13.0 kPa, with
# supernatant potassium 38 and lactate 22.5 mmol/L. Sodium is the additive's
# 150 mmol/L less the potassium gained, and chloride balances the bicarbonate.
# Citrate is 1.32 mmol per 300 mL unit.
STORED_BLOOD_PH = 6.5
STORED_BLOOD_PCO2 = 13.0 * 7.5006168270417
_STORED_BICARBONATE = plasma_bicarbonate(STORED_BLOOD_PH, STORED_BLOOD_PCO2)
_STORED_CITRATE = 1.32 / 0.135

FLUID_COMPOSITIONS = {
    "crystalloid": FluidComposition(130.0, 4.0, 109.0, calcium=1.5, lactate=28.0, l_lactate_fraction=0.5),
    "saline": FluidComposition(154.0, 0.0, 154.0),
    "colloid": FluidComposition(145.0, 0.0, 132.0, albumin=50.0),
    "blood": FluidComposition(
        116.0, 38.0, 116.0 + 38.0 + 1.0 - 22.5 - 3.0 * _STORED_CITRATE - _STORED_BICARBONATE,
        magnesium=0.5, phosphate=1.0, lactate=22.5, citrate=_STORED_CITRATE,
        carbon=_STORED_BICARBONATE * (1.0 + 10.0**(6.1 - STORED_BLOOD_PH)),
    ),
    "lipid": FluidComposition(0.0, 0.0, 0.0),
}
_IONS = ("sodium", "potassium", "chloride", "calcium", "magnesium", "phosphate")
_IONIZED_CALCIUM = 1.2  # mmol/L at pH 7.4
# Fitted to serum citrate against ionized calcium in Bolan 2001 (Fig. 2A); the
# effective constant includes magnesium competition for citrate.
_CITRATE_KD = 1.74  # mmol/L
# Ansari 2025: V1 10.9 L, V2 16.5 L, distribution 1.25 L/min, clearance 0.183 L/min.
# Their fractions and rate constants are applied to the model's extracellular volume.
_CALCIUM_CENTRAL_FRACTION = 10.9 / (10.9 + 16.5)
_CALCIUM_MIXING_PER_MIN = 1.25 * (1.0 / 10.9 + 1.0 / 16.5)
_CALCIUM_CLEARANCE_PER_MIN = 0.183 / (10.9 + 16.5)


def _albumin_binding(albumin_g_l: float, ph: float) -> float:
    """Albumin-bound per free calcium; Thode 1983 bounds the pH dependence."""
    return albumin_g_l / 42.0 * 10.0**(0.44 * (ph - 7.4))


class FluidBalance:
    """Well-mixed extracellular solutes, with albumin tracked separately in plasma and interstitium.

    Buffer base is the conserved charge budget shared with gas transport. Cells
    exchange water to match tonicity and exchange potassium for an equal amount
    of sodium.
    """

    def __init__(self, weight: float, blood_volume_ml: float, red_cell_volume_ml: float,
                 resting_ph: float = 7.4, extracellular_water: float = 1.0):
        self.weight = weight
        self.plasma_l = (blood_volume_ml - red_cell_volume_ml) / 1000.0
        self.red_cells_l = red_cell_volume_ml / 1000.0
        self.ecf_l = max(0.2 * weight * extracellular_water, self.plasma_l + 0.1 * weight)
        self.initial_ecf_l = self.ecf_l
        self.amounts = {"sodium": 140.0 * self.ecf_l, "potassium": 4.0 * self.ecf_l,
                        "chloride": 104.0 * self.ecf_l, "calcium": 0.0,
                        "magnesium": 0.8 * self.ecf_l, "phosphate": self.ecf_l}
        self.plasma_albumin_g = 42.0 * self.plasma_l
        self.interstitial_albumin_g = 14.0 * self.interstitial_l
        # Wooten 2003's plasma-equivalent buffer, 23.5 mmol/L at twice the blood volume,
        # stays as a fixed cellular pool; interstitial protein alone is too weak for the
        # whole-body CO2 responses of Brackett 1965 and Arbus 1969.
        self.cell_buffer_mmol = 23.5 * 2.0 * blood_volume_ml / 1000.0
        self.citrate_mmol = 0.0
        self.calcium_unmixed_mmol = 0.0
        self.amounts["calcium"] = self.ecf_l * self._total_calcium(_IONIZED_CALCIUM, 7.4)
        self.ph = self.resting_ph = resting_ph
        self.calcium_set_point = self.ionized_calcium(resting_ph)
        self.icf_l = 0.4 * weight
        self.cell_k_mmol = 140.0 * self.icf_l
        self.cell_na_mmol = 10.0 * self.icf_l
        self._cell_solute = self.icf_l * self._extracellular_solute() / self.ecf_l
        self.d_lactate_mmol = 0.0
        self.infused_l_lactate_mmol = 0.0
        self.base_meq = 0.0
        self.donor_rbc_base_meq_l = 0.0
        self.blood_base_meq_l = 0.0
        self.renal_base_meq = 0.0
        self.net_acid_excretion_mmol = 0.0
        self.fixed_acid_mmol = 0.0
        self.bicarbonate_excreted_mmol = 0.0
        self.urine_pending_l = 0.0
        self.blood_loss_pending_l = 0.0
        self.red_cell_input_pending_l = 0.0
        self.carbon_pending_mmol = 0.0
        self.lactate_loss_pending_l = 0.0
        self._elapsed = 0.0

    @property
    def interstitial_l(self) -> float:
        return self.ecf_l - self.plasma_l

    @property
    def citrate_volume_l(self) -> float:
        return self.ecf_l + 0.1 * self.weight

    @property
    def lactate_volume_l(self) -> float:
        return 0.26 * self.weight + self.ecf_l - self.initial_ecf_l

    def concentration(self, ion: str) -> float:
        return self.amounts[ion] / self.ecf_l

    def _extracellular_solute(self) -> float:
        return self.amounts["sodium"] + self.amounts["potassium"]

    def _equilibrate_cell_water(self) -> None:
        # Water crosses cell membranes until tonicity matches. It moves through
        # the interstitium, as the circulation sets plasma volume.
        solute = self._extracellular_solute()
        water = self.ecf_l + self.icf_l
        moved = water * solute / (solute + self._cell_solute) - self.ecf_l
        self.ecf_l += moved
        self.icf_l -= moved

    @property
    def plasma_buffers(self) -> PlasmaBuffers:
        return PlasmaBuffers(self.plasma_albumin_g / self.plasma_l, self.concentration("phosphate"))

    @property
    def interstitial_buffers(self) -> PlasmaBuffers:
        return PlasmaBuffers(self.interstitial_albumin_g / self.interstitial_l,
                             self.concentration("phosphate"), self.cell_buffer_mmol / self.interstitial_l)

    def seed_base(self, base_meq: float, donor_rbc_base_meq_l: float, blood_base_meq_l: float) -> None:
        self.base_meq = base_meq
        self.donor_rbc_base_meq_l = donor_rbc_base_meq_l
        self.blood_base_meq_l = blood_base_meq_l

    def infuse(self, amount_ml: float, retained_ml: float, hematocrit: float, label: str) -> None:
        fluid = FLUID_COMPOSITIONS["blood" if hematocrit > 0.0 else label]
        volume = amount_ml / 1000.0
        cells = retained_ml * hematocrit / 1000.0
        water = volume - cells
        for ion in _IONS:
            self.amounts[ion] += water * getattr(fluid, ion)
        self.ecf_l += water
        self.red_cells_l += cells
        self.red_cell_input_pending_l += cells
        self.carbon_pending_mmol += water * fluid.carbon
        self.base_meq += water * fluid.sid + cells * self.donor_rbc_base_meq_l
        self.citrate_mmol += water * fluid.citrate
        self.infused_l_lactate_mmol += water * fluid.lactate * fluid.l_lactate_fraction
        self.d_lactate_mmol += water * fluid.lactate * (1.0 - fluid.l_lactate_fraction)
        plasma_water = retained_ml / 1000.0 - cells
        self.plasma_l += plasma_water
        self.plasma_albumin_g += plasma_water * fluid.albumin
        self.interstitial_albumin_g += (water - plasma_water) * fluid.albumin

    def bleed(self, amount_ml: float, hematocrit: float) -> None:
        volume = amount_ml / 1000.0
        cells = volume * hematocrit
        water = volume - cells
        fraction = water / self.ecf_l
        self.base_meq -= volume * self.blood_base_meq_l
        free_calcium = self.ionized_calcium(self.ph)
        plasma_albumin = self.plasma_albumin_g / self.plasma_l
        plasma_calcium = (free_calcium * (1.0 + _albumin_binding(plasma_albumin, self.ph))
                          + self._citrate_calcium(free_calcium))
        for ion in _IONS:
            if ion == "calcium":
                self.amounts[ion] -= water * plasma_calcium
            else:
                self.amounts[ion] *= 1.0 - fraction
        self.citrate_mmol *= 1.0 - water / self.citrate_volume_l
        self.d_lactate_mmol *= 1.0 - water / self.lactate_volume_l
        self.plasma_albumin_g *= 1.0 - water / self.plasma_l
        self.ecf_l -= water
        self.plasma_l -= water
        self.red_cells_l -= cells
        self.blood_loss_pending_l += volume
        self.lactate_loss_pending_l += water

    def shift_water(self, vascular_change_ml: float, protein_fraction: float = 0.0) -> None:
        volume = vascular_change_ml / 1000.0
        if volume < 0.0:
            protein = -volume * self.plasma_albumin_g / self.plasma_l * protein_fraction
        else:
            protein = -volume * self.interstitial_albumin_g / self.interstitial_l
        self.plasma_albumin_g -= protein
        self.interstitial_albumin_g += protein
        self.plasma_l += volume

    def lose_urine(self, amount_ml: float) -> None:
        volume = amount_ml / 1000.0
        self.ecf_l -= volume
        self.plasma_l -= volume
        self.urine_pending_l += volume

    def step(self, dt: float, *, ph: float, paco2: float, resting_paco2: float,
             renal_function: float, renal_perfusion: float, hepatic_function: float,
             oxygen_supply: float, epinephrine: float) -> None:
        self.ph = ph
        self._equilibrate_cell_water()
        self._elapsed += dt
        if self._elapsed < 1.0 - 1e-12:
            return
        minutes = self._elapsed / 60.0
        self._elapsed = 0.0
        self._clear_calcium(minutes, ph)
        self._metabolize_buffers(minutes, hepatic_function, oxygen_supply)
        self._renal_balance(minutes, ph, paco2, resting_paco2, renal_function, renal_perfusion)
        self._exchange_potassium(minutes, epinephrine)

    def _metabolize_buffers(self, minutes: float, hepatic_function: float, oxygen_supply: float) -> None:
        # Kramer 2003: control Vdss 21 L and clearance 710 mL/min.
        # The liver/perfusion scaling and D-isomer kinetics are reduced-model assumptions.
        citrate_rate = (0.710 * self.weight / 70.0 / self.citrate_volume_l
                        * (0.25 + 0.75 * hepatic_function * oxygen_supply))
        citrate_removed = self.citrate_mmol * -math.expm1(-citrate_rate * minutes)
        self.citrate_mmol -= citrate_removed
        d_removed = self.d_lactate_mmol * -math.expm1(
            -0.012 * self.weight / self.lactate_volume_l * hepatic_function * minutes,
        )
        self.d_lactate_mmol -= d_removed
        self.base_meq += 3.0 * citrate_removed + d_removed

    def _renal_balance(self, minutes: float, ph: float, paco2: float, resting_paco2: float,
                       renal_function: float, renal_perfusion: float) -> None:
        basal = 0.7 * self.weight / 1440.0
        target_pco2 = max(15.0, min(70.0, paco2))
        delta = target_pco2 - resting_paco2
        resting_hco3 = plasma_bicarbonate(self.resting_ph, resting_paco2)
        target_hco3 = resting_hco3 + (0.48 if delta >= 0.0 else 0.40) * delta
        target_ph = 6.1 + math.log10(target_hco3 / (0.230 / 7.5006168270417 * target_pco2))
        target_be = standard_base_excess(target_ph, target_hco3)
        current_be = standard_base_excess(ph, plasma_bicarbonate(ph, paco2))
        # Endpoints follow Gonzalez 2018 and Krapf 1991, which measured steady
        # states; the 24 and 12 h adaptation times are teaching values.
        tau = 1440.0 if delta >= 0.0 else 720.0
        extra = (target_be - current_be) * self.ecf_l / tau
        capacity = min(renal_function * renal_perfusion,
                       self.urine_pending_l / (0.5 * self.weight / 1000.0 / 60.0 * minutes))
        net_acid = max(-4.0 * basal, min(4.0 * basal, basal + extra)) * capacity * minutes
        urine = self.urine_pending_l
        sodium = urine * self.concentration("sodium")
        potassium = urine * 40.0 * (self.concentration("potassium") / 4.0)**2
        magnesium = urine * self.concentration("magnesium")
        calcium = urine * 0.5
        net_acid = max(-sodium - potassium - 2.0 * (calcium + magnesium), net_acid)
        compensation = net_acid - basal * capacity * minutes
        self.renal_base_meq += compensation
        self.net_acid_excretion_mmol += net_acid
        self.fixed_acid_mmol += basal * minutes
        self.base_meq += net_acid - basal * minutes
        self.urine_pending_l = 0.0
        # Fixed acid leaves with its own unmeasured anions. Compensation excretes
        # ammonium chloride or retains chloride, so chloride moves opposite to bicarbonate.
        chloride = max(0.0, urine * self.concentration("chloride") + compensation)
        bicarbonate = max(0.0, -net_acid)
        self.bicarbonate_excreted_mmol += bicarbonate
        for ion, amount in (("sodium", sodium), ("potassium", potassium), ("chloride", chloride),
                            ("calcium", calcium), ("magnesium", magnesium),
                            ("phosphate", urine * self.concentration("phosphate"))):
            self.amounts[ion] -= amount

    def _exchange_potassium(self, minutes: float, epinephrine: float) -> None:
        # Exchange with a finite cellular pool; no universal potassium/pH correction.
        total = self.cell_k_mmol + self.amounts["potassium"]
        beta = 0.20 * epinephrine**2 / (epinephrine**2 + 0.6**2)
        target_k = total * self.ecf_l / (self.ecf_l + self.icf_l * 35.0 * (1.0 + beta))
        moved = (self.amounts["potassium"] - target_k) * -math.expm1(-minutes / 5.0)
        moved = min(self.cell_na_mmol, max(-self.cell_k_mmol, moved))
        self.amounts["potassium"] -= moved
        self.cell_k_mmol += moved
        self.amounts["sodium"] += moved
        self.cell_na_mmol -= moved

    def _clear_calcium(self, minutes: float, ph: float) -> None:
        self.calcium_unmixed_mmol *= math.exp(-_CALCIUM_MIXING_PER_MIN * minutes)
        # Bolan 2001: total calcium held steady through 90 min of citrate-induced
        # hypocalcemia, so only calcium above the normal ionized level is cleared.
        excess = (self.amounts["calcium"] + self.calcium_unmixed_mmol / _CALCIUM_CENTRAL_FRACTION
                  - self.ecf_l * self._total_calcium(self.calcium_set_point, ph))
        if excess > 0.0:
            self.amounts["calcium"] -= excess * -math.expm1(-_CALCIUM_CLEARANCE_PER_MIN * minutes)

    def _protein_binding(self, ph: float) -> float:
        return _albumin_binding((self.plasma_albumin_g + self.interstitial_albumin_g) / self.ecf_l, ph)

    def _citrate_calcium(self, free: float) -> float:
        return self.citrate_mmol / self.citrate_volume_l * free / (_CITRATE_KD + free)

    def _total_calcium(self, free: float, ph: float) -> float:
        return free * (1.0 + self._protein_binding(ph)) + self._citrate_calcium(free)

    def ionized_calcium(self, ph: float) -> float:
        """Plasma ionized calcium, including injected calcium not yet mixed through extracellular fluid."""
        total = (self.concentration("calcium")
                 + self.calcium_unmixed_mmol / (_CALCIUM_CENTRAL_FRACTION * self.ecf_l))
        a = 1.0 + self._protein_binding(ph)
        b = _CITRATE_KD * a + self.citrate_mmol / self.citrate_volume_l - total
        return 2.0 * _CITRATE_KD * total / (b + math.sqrt(b * b + 4.0 * a * _CITRATE_KD * total))

    def give_calcium_chloride(self, mmol: float) -> None:
        self.amounts["calcium"] += mmol
        self.amounts["chloride"] += 2.0 * mmol
        self.calcium_unmixed_mmol += mmol * (1.0 - _CALCIUM_CENTRAL_FRACTION)
