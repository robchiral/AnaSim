"""Body water, electrolytes, protein, calcium binding, and renal net acid balance."""

import math
from dataclasses import dataclass

from anasim.core.utils import clamp01

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

    def reset(self) -> None:
        self.retention_factor = 1.0
        self.transfused_cells_l = 0.0
        self.retained_cells_l = 0.0
        self.carbon_mmol = 0.0
        self.blood_carbon_in_mmol = 0.0
        self.blood_carbon_retained_mmol = 0.0
        self.blood_base_in_meq = 0.0
        self.blood_base_retained_meq = 0.0
        self.infuse_base_meq = 0.0
        self.water_shift_total_l = 0.0
        self.water_shift_retained_l = 0.0
        self.albumin_shift_total_g = 0.0
        self.albumin_shift_retained_g = 0.0
        self.blood_urine_l = 0.0
        self.bled_base_est_meq = 0.0


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
                 resting_ph: float = 7.4, extracellular_water: float = 1.0,
                 excess_fat_kg: float = 0.0):
        self.weight = weight
        self.plasma_l = (blood_volume_ml - red_cell_volume_ml) / 1000.0
        self.red_cells_l = red_cell_volume_ml / 1000.0
        # Adipose tissue has a much higher ECW/ICW ratio and lower total water
        # than lean mass (Waki 1991); parameterize excess adipose above BMI 25 at
        # 11% ECW and 3% ICW so normal-BMI patients (excess_fat_kg = 0) are unchanged.
        fat_kg = max(0.0, min(weight * 0.75, excess_fat_kg))
        base_mass = weight - fat_kg
        base_ecf = (0.2 * base_mass + 0.11 * fat_kg) * extracellular_water
        self.ecf_l = max(base_ecf, self.plasma_l + 0.1 * base_mass + 0.05 * fat_kg)
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
        self.icf_l = 0.4 * base_mass + 0.03 * fat_kg
        self.cell_k_mmol = 140.0 * self.icf_l
        self.cell_na_mmol = 10.0 * self.icf_l
        self.exogenous_calcium_mmol = 0.0
        self.pending_calcium_mmol = 0.0
        self._circulation = 1.0
        self._arterial_k = 4.0
        self._cell_solute = self.icf_l * self._extracellular_solute() / self.ecf_l
        self.d_lactate_mmol = 0.0
        self.infused_l_lactate_mmol = 0.0
        self.stored_rbc_acid_mmol = 0.0
        self.metabolized_organic_carbon_mmol = 0.0
        self.base_meq = 0.0
        self.donor_rbc_base_meq_l = 0.0
        self.donor_rbc_normal_base_meq_l = 0.0
        self.blood_base_meq_l = 0.0
        self.renal_base_meq = 0.0
        self.net_acid_excretion_mmol = 0.0
        self.fixed_acid_mmol = 0.0
        self.bicarbonate_excreted_mmol = 0.0
        self.urine_pending_l = 0.0
        self._blood_sync = PendingBloodSync()
        self._spare_blood_sync = PendingBloodSync()
        self._running_blood_base_meq: float | None = None
        self.lactate_loss_pending_l = 0.0
        self._lysed_k_mmol = 0.0
        self._sux_k_efflux_mmol = 0.0
        self._ca_push_tau_s = 5.0

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

    @property
    def arterial_potassium(self) -> float:
        """Arterial-sample potassium (mmol/L); lags tissue efflux during zero-flow arrest."""
        if self._circulation >= 0.99:
            return self.concentration("potassium")
        return self._arterial_k

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
    def plasma_albumin_g_l(self) -> float:
        return self.plasma_albumin_g / self.plasma_l

    @property
    def plasma_buffers(self) -> PlasmaBuffers:
        return PlasmaBuffers(self.plasma_albumin_g_l, self.concentration("phosphate"))

    @property
    def interstitial_buffers(self) -> PlasmaBuffers:
        return PlasmaBuffers(self.interstitial_albumin_g / self.interstitial_l,
                             self.concentration("phosphate"), self.cell_buffer_mmol / self.interstitial_l)

    def adjust_running_blood_base(self, delta_meq: float) -> None:
        """Apply an external correction to the running blood base estimate if initialized."""
        if self._running_blood_base_meq is not None:
            self._running_blood_base_meq += delta_meq

    def _ensure_running_blood_base(self) -> float:
        if self._running_blood_base_meq is None:
            self._running_blood_base_meq = (self.plasma_l + self.red_cells_l) * self.blood_base_meq_l
        return self._running_blood_base_meq

    def seed_base(
        self,
        base_meq: float,
        donor_rbc_base_meq_l: float,
        blood_base_meq_l: float,
        donor_rbc_normal_base_meq_l: float | None = None,
    ) -> None:
        self.base_meq = base_meq
        self.donor_rbc_base_meq_l = donor_rbc_base_meq_l
        self.donor_rbc_normal_base_meq_l = (
            donor_rbc_base_meq_l if donor_rbc_normal_base_meq_l is None else donor_rbc_normal_base_meq_l
        )
        self.blood_base_meq_l = blood_base_meq_l
        self._running_blood_base_meq = None

    def consume_blood_sync(self) -> PendingBloodSync:
        """Return and reset accumulated vascular transfers for respiratory synchronization."""
        sync = self._blood_sync
        spare = self._spare_blood_sync
        spare.reset()
        self._blood_sync = spare
        self._spare_blood_sync = sync
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
        if cells > 0.0:
            # Stored RBCs carry intracellular L-lactate equilibrated with supernatant
            # plus a reversible storage-acid deficit relative to pH 7.4 RBCs; clearing
            # these with citrate restores Stewart electroneutrality and reproduces
            # post-transfusion metabolic alkalosis (Kahn 1979) without losing the
            # immediate Rabcuka 2025 acid load.
            rbc_l_lactate = cells * fluid.lactate * fluid.l_lactate_fraction
            self.infused_l_lactate_mmol += rbc_l_lactate
            rbc_deficit = max(0.0, self.donor_rbc_normal_base_meq_l - self.donor_rbc_base_meq_l)
            self.stored_rbc_acid_mmol += max(0.0, cells * rbc_deficit - rbc_l_lactate)
        self.d_lactate_mmol += water * fluid.lactate * (1.0 - fluid.l_lactate_fraction)
        plasma_water = retained_ml / 1000.0 - cells
        plasma_carbon = plasma_water * fluid.carbon
        sync.blood_carbon_in_mmol += plasma_carbon
        sync.blood_carbon_retained_mmol += plasma_carbon
        blood_inc = plasma_water * fluid.sid + cells * self.donor_rbc_base_meq_l
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
        central_ca_l = _CALCIUM_CENTRAL_FRACTION * self.ecf_l
        central_ca_frac = min(1.0, max(0.0, water / central_ca_l)) if central_ca_l > 0.0 else 1.0
        resting_plasma_ca = self.calcium_set_point * (1.0 + _albumin_binding(plasma_albumin, self.resting_ph))
        shed_ca = min(self.amounts["calcium"], water * plasma_calcium)
        for ion in _IONS:
            if ion == "calcium":
                self.amounts[ion] = max(0.0, self.amounts[ion] - shed_ca)
            else:
                self.amounts[ion] *= 1.0 - fraction
        self.calcium_unmixed_mmol *= 1.0 - central_ca_frac
        shed_excess_ca = max(0.0, shed_ca - water * resting_plasma_ca)
        self.exogenous_calcium_mmol = max(
            0.0,
            min(self.exogenous_calcium_mmol * max(0.0, 1.0 - fraction), self.exogenous_calcium_mmol - shed_excess_ca),
        )
        self.citrate_mmol *= 1.0 - water / self.citrate_volume_l
        self.d_lactate_mmol *= 1.0 - water / self.lactate_volume_l
        self.stored_rbc_acid_mmol *= keep
        self._lysed_k_mmol *= 1.0 - fraction
        self._sux_k_efflux_mmol *= 1.0 - fraction
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
             succinylcholine_ce: float = 0.0, mh_severity: float = 0.0,
             circulation: float = 1.0, lactate_acid_meq_l: float = 0.0) -> None:
        self.ph = ph
        # Preserve pre-arrest arterial K+ equilibrated under normal perfusion before
        # switching into low/zero-flow arrest where peripheral efflux lags central blood.
        if self._circulation >= 0.99:
            self._arterial_k = self.concentration("potassium")
        self._circulation = clamp01(circulation)
        self._equilibrate_cell_water()
        if dt <= 0.0:
            if self._circulation >= 0.99:
                self._arterial_k = self.concentration("potassium")
            return
        if self.pending_calcium_mmol > 0.0 and self._circulation > 0.0:
            delivered = self.pending_calcium_mmol * -math.expm1(-self._circulation * dt / self._ca_push_tau_s)
            self.pending_calcium_mmol -= delivered
            self._deliver_calcium(delivered)
        minutes = dt / 60.0
        self._clear_calcium(minutes * self._circulation, renal_function=renal_function)
        self._metabolize_buffers(minutes, hepatic_function, oxygen_supply)
        self._renal_balance(minutes, ph, paco2, resting_paco2, renal_function, renal_perfusion)
        self._exchange_potassium(
            minutes, epinephrine, ph=ph, paco2=paco2, resting_paco2=resting_paco2,
            succinylcholine_ce=succinylcholine_ce, mh_severity=mh_severity,
            oxygen_supply=oxygen_supply, lactate_acid_meq_l=lactate_acid_meq_l,
        )
        ecf_k = self.concentration("potassium")
        if self._circulation >= 0.99:
            self._arterial_k = ecf_k
        elif self._circulation > 0.0:
            self._arterial_k += (ecf_k - self._arterial_k) * -math.expm1(-dt * self._circulation / 20.0)

    def _metabolize_buffers(self, minutes: float, hepatic_function: float, oxygen_supply: float) -> None:
        # Kramer 2003: control Vdss 21 L and clearance 710 mL/min, with cirrhotic
        # clearance ~48% of control (matched at hepatic_function = 0.5).
        # The perfusion scaling and D-isomer kinetics are reduced-model assumptions.
        oxy = clamp01(oxygen_supply)
        hepatic_citrate = 0.05 + 0.95 * (max(0.0, hepatic_function) ** 1.15)
        citrate_rate = 0.710 * self.weight / 70.0 / self.citrate_volume_l * hepatic_citrate * oxy
        citrate_removed = self.citrate_mmol * -math.expm1(-citrate_rate * minutes)
        self.citrate_mmol -= citrate_removed
        d_removed = self.d_lactate_mmol * -math.expm1(
            -0.012 * self.weight / self.lactate_volume_l * hepatic_function * oxy * minutes,
        )
        self.d_lactate_mmol -= d_removed
        rbc_acid_rate = 0.045 * (0.25 + 0.75 * hepatic_function) * oxy
        rbc_acid_removed = self.stored_rbc_acid_mmol * -math.expm1(-rbc_acid_rate * minutes)
        self.stored_rbc_acid_mmol -= rbc_acid_removed
        organic_base = 3.0 * citrate_removed + d_removed
        self.base_meq += organic_base + rbc_acid_removed
        # Chioléro 1993: aerobic oxidation of exogenous sodium lactate/citrate
        # generates bicarbonate carbon directly (RER 0.85 -> 0.75, VCO2 -5.3%)
        # rather than scavenging 1 mmol endogenous CO2 per mEq base formed.
        self.metabolized_organic_carbon_mmol += 0.85 * organic_base

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
        net_acid = max(-10.0 * basal, min(4.0 * basal, basal + extra)) * capacity * minutes
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
            conc_na = conc_cl = max_na = potassium = magnesium = calcium = phosphate = 0.0
        non_na_cations = potassium + 2.0 * (calcium + magnesium)
        net_acid = max(-max_na - non_na_cations, net_acid)
        compensation = net_acid - basal * capacity * minutes
        self.renal_base_meq += compensation
        self.net_acid_excretion_mmol += net_acid
        self.fixed_acid_mmol += basal * minutes
        self.urine_pending_l = 0.0
        # Fixed acid leaves with its own unmeasured anions. Compensation excretes
        # ammonium chloride or retains chloride, while volume-expansion natriuresis
        # excretes additional isotonic NaCl (Reid 2003; Chowdhury 2012) so urine SID
        # remains -compensation without causing hypernatremia.
        natriuresis_share = (
            clamp01((ecf_before - self.initial_ecf_l) / 1.5)
            if urine > 0.0 and ecf_before > self.initial_ecf_l
            else 0.0
        )
        base_na_conc = max(0.0, conc_cl - non_na_cations / urine) if urine > 0.0 else 0.0
        extra_nacl = (
            urine * min(max(0.0, conc_na - base_na_conc), 45.0) * natriuresis_share
            if natriuresis_share > 0.0
            else 0.0
        )
        chloride = min(self.amounts["chloride"], max(0.0, urine * conc_cl + compensation + extra_nacl))
        target_na = chloride - compensation - non_na_cations
        if target_na >= 0.0:
            sodium = min(self.amounts["sodium"], target_na)
            if sodium < target_na:
                chloride = max(0.0, chloride - (target_na - sodium))
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
        if self._lysed_k_mmol > 0.0 and potassium > 0.0:
            basal_k_urine = urine * 25.0
            self._lysed_k_mmol = max(0.0, self._lysed_k_mmol - max(0.0, potassium - basal_k_urine))

    def _exchange_potassium(
        self, minutes: float, epinephrine: float, *, ph: float = 7.4,
        paco2: float = 40.0, resting_paco2: float = 40.0,
        succinylcholine_ce: float = 0.0, mh_severity: float = 0.0,
        oxygen_supply: float = 1.0, lactate_acid_meq_l: float = 0.0,
    ) -> None:
        # Reversible transcellular K+ shifts (beta-2, succinylcholine, bidirectional pH)
        # plus cumulative rhabdomyolytic (MH) and severe hypoxic cellular K+ leak that
        # persists after crisis arrest until cleared renally or shifted by beta-2 agonists.
        if mh_severity > 0.0:
            self._lysed_k_mmol += 0.075 * self.weight * (mh_severity ** 2) * minutes
        if oxygen_supply < 0.50:
            hypox = clamp01((0.50 - oxygen_supply) / 0.50)
            self._lysed_k_mmol += 0.015 * self.weight * hypox * minutes
        # Slow passive redistribution/cellular recovery over hours once lysis stops.
        if self._lysed_k_mmol > 0.0 and mh_severity <= 0.05 and oxygen_supply >= 0.50:
            self._lysed_k_mmol *= math.exp(-minutes / 240.0)
        self._lysed_k_mmol = min(self._lysed_k_mmol, 0.5 * self.cell_k_mmol)

        total = self.cell_k_mmol + self.amounts["potassium"]
        beta = 0.20 * epinephrine**2 / (epinephrine**2 + 0.6**2)
        sux_ce = max(0.0, succinylcholine_ce)
        # Martyn 2006: depolarizing nicotinic K+ efflux rises +0.5 mEq/L, peaking at 2-5 min
        # faster than slower metabolic/beta-2 transcellular equilibration.
        sux_target_mmol = 0.22 * sux_ce**2 / (sux_ce**2 + 1.5**2) * (0.65 * self.ecf_l * 4.0)
        sux_tau = 1.2 if sux_target_mmol > self._sux_k_efflux_mmol else 3.5
        sux_delta = (sux_target_mmol - self._sux_k_efflux_mmol) * -math.expm1(-minutes / sux_tau)
        if sux_delta < 0.0:
            sux_delta *= self._circulation
        sux_delta = min(self.cell_k_mmol, max(-self.cell_na_mmol, sux_delta))
        self._sux_k_efflux_mmol = max(0.0, self._sux_k_efflux_mmol + sux_delta)
        self.amounts["potassium"] += sux_delta
        self.cell_k_mmol -= sux_delta
        self.amounts["sodium"] -= sux_delta
        self.cell_na_mmol += sux_delta

        delta_ph = ph - self.resting_ph
        if delta_ph < 0.0:
            # Adrogue & Madias 1981; Orringer 1977: mineral metabolic acidemia
            # drives substantial K+ efflux (~0.40 mEq/L per 0.1 pH), acute
            # respiratory acidemia causes a modest shift (~0.10-0.13 mEq/L per
            # 0.1 pH), and organic/lactic acidemia causes negligible direct shift.
            eff_paco2 = max(10.0, paco2)
            ref_paco2 = max(10.0, resting_paco2)
            resting_hco3 = plasma_bicarbonate(self.resting_ph, ref_paco2)
            acute_hco3 = max(1.0, resting_hco3 + 0.10 * max(0.0, eff_paco2 - ref_paco2))
            resp_ph = 6.1 + math.log10(acute_hco3 / (_PLASMA_CO2_SOLUBILITY * eff_paco2))
            resp_drop = min(-delta_ph, max(0.0, self.resting_ph - resp_ph))
            metab_drop = max(0.0, -delta_ph - resp_drop)
            organic_drop = min(metab_drop, 0.015 * max(0.0, lactate_acid_meq_l))
            mineral_drop = max(0.0, metab_drop - organic_drop)
            ph_efflux = (
                1.05 * mineral_drop
                + 0.45 * max(0.0, mineral_drop - 0.25)
                + 0.28 * resp_drop
            )
            ph_uptake = 0.0
        else:
            ph_efflux = 0.0
            ph_uptake = 0.50 * delta_ph
        efflux = 1.0 + ph_efflux
        partition = 35.0 * (1.0 + beta + ph_uptake) / efflux
        intact_total = max(0.0, total - self._lysed_k_mmol - self._sux_k_efflux_mmol)
        lysed_ecf_share = self._lysed_k_mmol / (1.0 + beta + ph_uptake)
        target_k = intact_total * self.ecf_l / (self.ecf_l + self.icf_l * partition) + lysed_ecf_share + self._sux_k_efflux_mmol
        moved = (self.amounts["potassium"] - target_k) * -math.expm1(-minutes / 5.0)
        if moved > 0.0:
            moved *= self._circulation
        moved = min(self.cell_na_mmol, max(-self.cell_k_mmol, moved))
        self.amounts["potassium"] -= moved
        self.cell_k_mmol += moved
        self.amounts["sodium"] += moved
        self.cell_na_mmol -= moved

    def _clear_calcium(self, minutes: float, *, renal_function: float = 1.0) -> None:
        self.calcium_unmixed_mmol *= math.exp(-_CALCIUM_MIXING_PER_MIN * minutes)
        # Bolan 2001: elevated PTH did not restore citrate-induced calcium loss
        # within 90 min; clearing only calcium above the resting pool is a
        # reduced-model simplification (omitting citrate-bound calciuria).
        excess = (self.amounts["calcium"] + self.calcium_unmixed_mmol / _CALCIUM_CENTRAL_FRACTION
                  - self.ecf_l * self._total_calcium(self.calcium_set_point, self.resting_ph))
        if excess > 0.0:
            clearance_rate = _CALCIUM_CLEARANCE_PER_MIN * (0.25 + 0.75 * clamp01(renal_function))
            cleared = excess * -math.expm1(-clearance_rate * minutes)
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

    def _deliver_calcium(self, mmol: float) -> None:
        self.amounts["calcium"] += mmol
        self.amounts["chloride"] += 2.0 * mmol
        self.exogenous_calcium_mmol += mmol
        self.calcium_unmixed_mmol += mmol * (1.0 - _CALCIUM_CENTRAL_FRACTION)

    def give_calcium_chloride(
        self, mmol: float, circulation: float | None = None, *, bolus_push_s: float = 0.0,
    ) -> None:
        circ = self._circulation if circulation is None else clamp01(circulation)
        if circ >= 1.0 and bolus_push_s <= 0.0:
            self._deliver_calcium(mmol)
            return
        # Deliver an initial central pass smoothly scaled by circulation squared
        # (zero at arrest, 40% at full flow) and push the remainder over bolus_push_s.
        immediate = 0.40 * (circ * circ) * mmol
        if immediate > 0.0:
            self._deliver_calcium(immediate)
        added = mmol - immediate
        tau = max(5.0, bolus_push_s)
        prev = self.pending_calcium_mmol
        if prev + added > 0.0:
            self._ca_push_tau_s = (prev * self._ca_push_tau_s + added * tau) / (prev + added)
        else:
            self._ca_push_tau_s = tau
        self.pending_calcium_mmol = prev + added
