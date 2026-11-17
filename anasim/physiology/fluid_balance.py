"""Body water, electrolytes, protein, calcium binding, and renal net acid balance."""

import math
from dataclasses import dataclass

from .blood_chemistry import (
    _MMHG_PER_KPA,
    _PLASMA_CO2_SOLUBILITY,
    plasma_bicarbonate,
    standard_base_excess,
)
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


@dataclass(slots=True)
class PendingBloodSync:
    """Accumulated vascular, carbon, and buffer transfers awaiting respiratory sync."""

    retention_factor: float = 1.0
    transfused_cells_l: float = 0.0
    retained_cells_l: float = 0.0
    carbon_mmol: float = 0.0
    blood_carbon_in_mmol: float = 0.0
    blood_carbon_retained_mmol: float = 0.0
    blood_base_in_meq: float = 0.0
    blood_base_retained_meq: float = 0.0
    infuse_base_meq: float = 0.0
    water_shift_total_l: float = 0.0
    water_shift_retained_l: float = 0.0
    albumin_shift_total_g: float = 0.0
    albumin_shift_retained_g: float = 0.0
    blood_urine_l: float = 0.0
    bled_base_est_meq: float = 0.0

    def bleed(self, keep: float, bled_base_est: float) -> None:
        self.bled_base_est_meq += bled_base_est
        self.retention_factor *= keep
        self.retained_cells_l *= keep
        self.blood_carbon_retained_mmol *= keep
        self.blood_base_retained_meq *= keep
        self.water_shift_retained_l *= keep
        self.albumin_shift_retained_g *= keep
        self.blood_urine_l *= keep


# A 3-week-old SAGM unit (Rabcuka 2025, Table 3): pH 6.5 and CO2 13.0 kPa, with
# supernatant potassium 38 and lactate 22.5 mmol/L. Sodium is the additive's
# 150 mmol/L less the potassium gained, and chloride balances the bicarbonate.
# Citrate is 1.32 mmol per 300 mL unit.
STORED_BLOOD_PH = 6.5
STORED_BLOOD_PCO2 = 13.0 * _MMHG_PER_KPA
STORED_BLOOD_SO2 = 0.60
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
        # Plasma-equivalent buffer (23.5 mmol/L, Rees 2010) across twice the blood
        # volume (Wooten 2010) stays as a fixed cellular pool; interstitial protein
        # alone is too weak for the whole-body CO2 responses of Brackett 1965 and Arbus 1969.
        self.cell_buffer_mmol = 23.5 * 2.0 * blood_volume_ml / 1000.0
        self.citrate_mmol = 0.0
        self.calcium_unmixed_mmol = 0.0
        self.amounts["calcium"] = self.ecf_l * self._total_calcium(_IONIZED_CALCIUM, 7.4)
        self.ph = self.resting_ph = resting_ph
        self.calcium_set_point = self.ionized_calcium(resting_ph)
        self.icf_l = 0.4 * weight
        self.cell_k_mmol = 140.0 * self.icf_l
        self.cell_na_mmol = 10.0 * self.icf_l
        self.exogenous_calcium_mmol = 0.0
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
        self._blood_sync = PendingBloodSync()
        self._running_blood_base_meq: float | None = None
        self.lactate_loss_pending_l = 0.0

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

    def _ensure_running_blood_base(self) -> float:
        if self._running_blood_base_meq is None:
            self._running_blood_base_meq = (self.plasma_l + self.red_cells_l) * self.blood_base_meq_l
        return self._running_blood_base_meq

    def seed_base(self, base_meq: float, donor_rbc_base_meq_l: float, blood_base_meq_l: float) -> None:
        self.base_meq = base_meq
        self.donor_rbc_base_meq_l = donor_rbc_base_meq_l
        self.blood_base_meq_l = blood_base_meq_l
        self._running_blood_base_meq = None

    def consume_blood_sync(self) -> PendingBloodSync:
        """Return and reset accumulated vascular transfers for respiratory synchronization."""
        sync = self._blood_sync
        self._blood_sync = PendingBloodSync()
        self._running_blood_base_meq = None
        return sync

    def consume_lactate_loss_l(self) -> float:
        """Return and clear shed plasma water (L) for lactate pool adjustment."""
        lost = self.lactate_loss_pending_l
        self.lactate_loss_pending_l = 0.0
        return lost

    def infuse(self, amount_ml: float, retained_ml: float, hematocrit: float, label: str) -> None:
        running_blood_base = self._ensure_running_blood_base()
        sync = self._blood_sync
        fluid = FLUID_COMPOSITIONS["blood" if hematocrit > 0.0 else label]
        volume = amount_ml / 1000.0
        cells = retained_ml * hematocrit / 1000.0
        water = volume - cells
        for ion in _IONS:
            self.amounts[ion] += water * getattr(fluid, ion)
        self.ecf_l += water
        self.red_cells_l += cells
        sync.transfused_cells_l += cells
        sync.retained_cells_l += cells
        sync.carbon_mmol += water * fluid.carbon
        total_base_in = water * fluid.sid + cells * self.donor_rbc_base_meq_l
        self.base_meq += total_base_in
        sync.infuse_base_meq += total_base_in
        self.citrate_mmol += water * fluid.citrate
        self.infused_l_lactate_mmol += water * fluid.lactate * fluid.l_lactate_fraction
        self.d_lactate_mmol += water * fluid.lactate * (1.0 - fluid.l_lactate_fraction)
        plasma_water = retained_ml / 1000.0 - cells
        plasma_carbon = plasma_water * fluid.carbon
        sync.blood_carbon_in_mmol += plasma_carbon
        sync.blood_carbon_retained_mmol += plasma_carbon
        organic_acid_per_l = fluid.lactate + 3.0 * fluid.citrate
        inorganic_sid = fluid.sid + organic_acid_per_l
        plasma_share = (self.plasma_l + plasma_water) / self.ecf_l if self.ecf_l > 0.0 else 0.0
        blood_inc = (
            plasma_water * inorganic_sid
            - water * organic_acid_per_l * plasma_share
            + cells * self.donor_rbc_base_meq_l
        )
        sync.blood_base_in_meq += blood_inc
        sync.blood_base_retained_meq += blood_inc
        self._running_blood_base_meq = running_blood_base + blood_inc
        self.plasma_l += plasma_water
        self.plasma_albumin_g += plasma_water * fluid.albumin
        self.interstitial_albumin_g += (water - plasma_water) * fluid.albumin

    def bleed(self, amount_ml: float, hematocrit: float) -> None:
        running_blood_base = self._ensure_running_blood_base()
        curr_blood_l = self.plasma_l + self.red_cells_l
        volume = amount_ml / 1000.0
        cells = volume * hematocrit
        water = volume - cells
        fraction = water / self.ecf_l
        step_bleed_frac = min(1.0, max(0.0, volume / curr_blood_l)) if curr_blood_l > 0.0 else 0.0
        keep = 1.0 - step_bleed_frac
        bled_base_est = step_bleed_frac * running_blood_base
        self._running_blood_base_meq = running_blood_base - bled_base_est
        self.base_meq -= bled_base_est
        self._blood_sync.bleed(keep, bled_base_est)
        free_calcium = self.ionized_calcium(self.ph)
        plasma_albumin = self.plasma_albumin_g / self.plasma_l
        plasma_calcium = (free_calcium * (1.0 + _albumin_binding(plasma_albumin, self.ph))
                          + self._citrate_calcium(free_calcium))
        for ion in _IONS:
            if ion == "calcium":
                self.amounts[ion] -= water * plasma_calcium
            else:
                self.amounts[ion] *= 1.0 - fraction
        self.exogenous_calcium_mmol *= max(0.0, 1.0 - fraction)
        self.citrate_mmol *= 1.0 - water / self.citrate_volume_l
        self.d_lactate_mmol *= 1.0 - water / self.lactate_volume_l
        self.plasma_albumin_g *= 1.0 - water / self.plasma_l
        self.ecf_l -= water
        self.plasma_l -= water
        self.red_cells_l -= cells
        self.lactate_loss_pending_l += water

    def shift_water(self, vascular_change_ml: float, protein_fraction: float = 0.0) -> None:
        running_blood_base = self._ensure_running_blood_base()
        sync = self._blood_sync
        volume = vascular_change_ml / 1000.0
        frac = max(0.0, min(1.0, protein_fraction))
        if volume < 0.0:
            protein = -volume * self.plasma_albumin_g / self.plasma_l * frac
        else:
            protein = -volume * self.interstitial_albumin_g / self.interstitial_l * frac
        self.plasma_albumin_g -= protein
        self.interstitial_albumin_g += protein
        self.plasma_l += volume
        sync.water_shift_total_l += volume
        sync.water_shift_retained_l += volume
        sync.albumin_shift_total_g -= protein
        sync.albumin_shift_retained_g -= protein
        self._running_blood_base_meq = running_blood_base + volume * self.blood_base_meq_l

    def lose_urine(self, amount_ml: float) -> None:
        self._ensure_running_blood_base()
        volume = amount_ml / 1000.0
        self.ecf_l -= volume
        self.plasma_l -= volume
        self.urine_pending_l += volume
        self._blood_sync.blood_urine_l += volume

    def step(self, dt: float, *, ph: float, paco2: float, resting_paco2: float,
             renal_function: float, renal_perfusion: float, hepatic_function: float,
             oxygen_supply: float, epinephrine: float,
             succinylcholine_ce: float = 0.0, mh_severity: float = 0.0) -> None:
        self.ph = ph
        self._equilibrate_cell_water()
        if dt <= 0.0:
            return
        minutes = dt / 60.0
        self._clear_calcium(minutes)
        self._metabolize_buffers(minutes, hepatic_function, oxygen_supply)
        self._renal_balance(minutes, ph, paco2, resting_paco2, renal_function, renal_perfusion)
        self._exchange_potassium(
            minutes, epinephrine, ph=ph,
            succinylcholine_ce=succinylcholine_ce, mh_severity=mh_severity,
        )

    def _metabolize_buffers(self, minutes: float, hepatic_function: float, oxygen_supply: float) -> None:
        # Kramer 2003: control Vdss 21 L and clearance 710 mL/min, consistent with
        # intraoperative massive-transfusion citrate in Kahn 1979.
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
        target_ph = 6.1 + math.log10(target_hco3 / (_PLASMA_CO2_SOLUBILITY * target_pco2))
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
        ecf_before = self.ecf_l + urine
        if urine > 0.0 and ecf_before > 0.0:
            conc_na = self.amounts["sodium"] / ecf_before
            conc_k = self.amounts["potassium"] / ecf_before
            conc_cl = self.amounts["chloride"] / ecf_before
            conc_mg = self.amounts["magnesium"] / ecf_before
            conc_pi = self.amounts["phosphate"] / ecf_before
            max_na = min(self.amounts["sodium"], urine * conc_na)
            potassium = min(self.amounts["potassium"], urine * 25.0 * (conc_k / 4.0) ** 2)
            magnesium = min(self.amounts["magnesium"], urine * conc_mg)
            calcium = min(self.amounts["calcium"], urine * 0.5)
            phosphate = min(self.amounts["phosphate"], urine * conc_pi)
        else:
            conc_cl = max_na = potassium = magnesium = calcium = phosphate = 0.0
        non_na_cations = potassium + 2.0 * (calcium + magnesium)
        net_acid = max(-max_na - non_na_cations, net_acid)
        compensation = net_acid - basal * capacity * minutes
        self.renal_base_meq += compensation
        self.net_acid_excretion_mmol += net_acid
        self.fixed_acid_mmol += basal * minutes
        self.urine_pending_l = 0.0
        # Fixed acid leaves with its own unmeasured anions. Compensation excretes
        # ammonium chloride or retains chloride, while urinary strong cations match
        # uncompensated chloride so urine SID tracks -compensation.
        chloride = min(self.amounts["chloride"], max(0.0, urine * conc_cl + compensation))
        target_na = chloride - compensation - non_na_cations
        if target_na >= 0.0:
            sodium = min(self.amounts["sodium"], target_na)
        else:
            sodium = 0.0
            chloride = min(self.amounts["chloride"], max(0.0, non_na_cations + compensation))
        sid_excreted = sodium + non_na_cations - chloride
        self.base_meq += -sid_excreted - basal * (1.0 - capacity) * minutes
        bicarbonate = max(0.0, -net_acid)
        self.bicarbonate_excreted_mmol += bicarbonate
        for ion, amount in (("sodium", sodium), ("potassium", potassium), ("chloride", chloride),
                            ("calcium", calcium), ("magnesium", magnesium),
                            ("phosphate", phosphate)):
            self.amounts[ion] -= amount

    def _exchange_potassium(
        self, minutes: float, epinephrine: float, *, ph: float = 7.4,
        succinylcholine_ce: float = 0.0, mh_severity: float = 0.0,
    ) -> None:
        # Exchange with a finite cellular pool; mild pH changes do not shift K+.
        total = self.cell_k_mmol + self.amounts["potassium"]
        beta = 0.20 * epinephrine**2 / (epinephrine**2 + 0.6**2)
        sux_ce = max(0.0, succinylcholine_ce)
        sux = 0.22 * sux_ce**2 / (sux_ce**2 + 1.5**2)
        acid = 0.80 * max(0.0, 7.15 - ph)
        efflux = 1.0 + sux + 0.70 * max(0.0, mh_severity) ** 2 + acid
        partition = 35.0 * (1.0 + beta) / efflux
        target_k = total * self.ecf_l / (self.ecf_l + self.icf_l * partition)
        moved = (self.amounts["potassium"] - target_k) * -math.expm1(-minutes / 5.0)
        moved = min(self.cell_na_mmol, max(-self.cell_k_mmol, moved))
        self.amounts["potassium"] -= moved
        self.cell_k_mmol += moved
        self.amounts["sodium"] += moved
        self.cell_na_mmol -= moved

    def _clear_calcium(self, minutes: float) -> None:
        self.calcium_unmixed_mmol *= math.exp(-_CALCIUM_MIXING_PER_MIN * minutes)
        # Bolan 2001: elevated PTH did not restore citrate-induced calcium loss
        # within 90 min; clearing only calcium above the resting pool is a
        # reduced-model simplification (omitting citrate-bound calciuria).
        excess = (self.amounts["calcium"] + self.calcium_unmixed_mmol / _CALCIUM_CENTRAL_FRACTION
                  - self.ecf_l * self._total_calcium(self.calcium_set_point, self.resting_ph))
        if excess > 0.0:
            cleared = excess * -math.expm1(-_CALCIUM_CLEARANCE_PER_MIN * minutes)
            self.amounts["calcium"] -= cleared
            exo_cleared = min(cleared, self.exogenous_calcium_mmol)
            self.exogenous_calcium_mmol -= exo_cleared
            self.base_meq -= 2.0 * exo_cleared
            endo_cleared = cleared - exo_cleared
            if endo_cleared > 0.0:
                self.amounts["chloride"] = max(0.0, self.amounts["chloride"] - 2.0 * endo_cleared)

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
        self.exogenous_calcium_mmol += mmol
        self.calcium_unmixed_mmol += mmol * (1.0 - _CALCIUM_CENTRAL_FRACTION)
