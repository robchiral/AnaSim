"""Blood buffering at 37 °C from Rees et al. (2010), Fig. 1 and Table 2.

Oxygen saturation is supplied independently. Oxygen affinity, gas transport,
and electrolyte exchange with other body fluids belong to the coupled model.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

from anasim.patient.domain import bounded_number, finite_number

_MMHG_PER_KPA = 7.5006168270417
_PLASMA_CO2_SOLUBILITY = 0.230 / _MMHG_PER_KPA  # mmol/L/mmHg
_RBC_CO2_SOLUBILITY = 0.195 / _MMHG_PER_KPA  # mmol/L/mmHg
_PH_RANGE = (6.0, 10.0)  # Numerical limits, wide enough for CO2-washed-out pulmonary blood.
_LN_10 = math.log(10.0)
# mmol/L or mEq/L; leaves pH within about 1e-9 and PCO2 within 1e-7 mmHg.
_CONTENT_TOLERANCE = 1e-8
# Reaction ratios relative to the bicarbonate pK 6.1 (Rees Fig. 1).
_PLASMA_BUFFER_RATIO = 10.0**(6.96 - 6.1)
_OXY_AMINO_RATIO = 10.0**(7.14 - 6.1)
_DEOXY_AMINO_RATIO = 10.0**(7.32 - 6.1)
_OXY_CARBAMINO_RATIO = 10.0**(6.1 - 5.46)
_DEOXY_CARBAMINO_RATIO = 10.0**(6.1 - 4.80)
_RBC_PH_GRADIENT = 0.77 * _LN_10  # d(rbc pH)/d(plasma pH) times ln 10 (Rees Eq. 24r).


@dataclass(frozen=True, slots=True)
class BloodChemistryState:
    """Equilibrium concentrations per liter of the named fluid or whole blood.

    Oxygen saturation is a fraction. Buffer base is in mEq/L; bicarbonate and
    total CO2 are in mmol/L. Buffer base is distinct from standard base excess.
    """

    ph: float
    rbc_ph: float
    pco2_mmhg: float
    oxygen_saturation: float
    plasma_bicarbonate_mmol_l: float
    rbc_bicarbonate_mmol_l: float
    plasma_buffer_base_meq_l: float
    rbc_buffer_base_meq_l: float
    buffer_base_meq_l: float
    total_co2_mmol_l: float


class Equilibrium(NamedTuple):
    """Concentrations and analytic pH and PCO2 derivatives at one equilibrium."""

    ph: float
    rbc_ph: float
    plasma_bicarbonate: float
    rbc_bicarbonate: float
    plasma_base: float
    rbc_base: float
    buffer_base: float
    total_co2: float
    base_ph: float
    base_co2: float
    content_ph: float
    content_co2: float

    @property
    def co2_capacity(self) -> float:
        """d(total CO2)/dPCO2 at fixed buffer base, mmol/L/mmHg."""
        return self.content_co2 - self.content_ph * self.base_co2 / self.base_ph

    def state(self, pco2: float, saturation: float) -> BloodChemistryState:
        return BloodChemistryState(
            self.ph, self.rbc_ph, pco2, saturation, self.plasma_bicarbonate,
            self.rbc_bicarbonate, self.plasma_base, self.rbc_base,
            self.buffer_base, self.total_co2,
        )


class FluidEquilibrium(NamedTuple):
    """Blood and adjacent extracellular fluid at a common pH and PCO2.

    The adjacent fluid uses the plasma-equivalent extracellular_contents model.
    """

    blood: Equilibrium
    pco2: float


class ContentIterate(NamedTuple):
    """Equilibrium at one iterate and the next damped Newton iterate."""

    blood: Equilibrium
    converged: bool
    ph: float
    pco2: float


def step_fraction(ph: float, ph_change: float, pco2: float, pressure_change: float) -> float:
    """Halve a Newton step until pH stays in the solver range and PCO2 stays positive."""
    fraction = 1.0
    while not (_PH_RANGE[0] <= ph - fraction * ph_change <= _PH_RANGE[1]
               and pco2 - fraction * pressure_change > 0.0):
        fraction *= 0.5
        if fraction < 1e-8:
            raise ValueError("Blood gas contents require pH outside the chemistry range")
    return fraction


def extracellular_contents(ph: float, pco2: float) -> tuple[float, float, float, float]:
    """CO2, buffer base, and base derivatives with respect to pH and PCO2."""
    dissolved = _PLASMA_CO2_SOLUBILITY * pco2
    bicarbonate = dissolved * 10.0**(ph - 6.1)
    ratio = 10.0**(6.96 - ph)
    weak_base = 23.5 / (1.0 + ratio)
    base_ph = _LN_10 * (bicarbonate + weak_base * ratio / (1.0 + ratio))
    return dissolved + bicarbonate, bicarbonate + weak_base, base_ph, bicarbonate / pco2


@dataclass(frozen=True, slots=True)
class BloodChemistry:
    """Plasma and red-cell equilibrium for a fixed blood composition.

    Hemoglobin is in mmol heme equivalents/L whole blood, hematocrit is a
    fraction, and plasma_buffer_mmol_l is the lumped non-bicarbonate buffer.
    The source uses 9.3 mmol/L hemoglobin, hematocrit 9.3/21, and 23.5 mmol/L
    plasma buffer. Other hemoglobin/hematocrit ratios are an extension of its
    fixed red-cell hemoglobin concentration and need separate validation.
    """

    hemoglobin_mmol_l: float
    hematocrit: float
    plasma_buffer_mmol_l: float = 23.5

    def __post_init__(self) -> None:
        hb = finite_number("hemoglobin_mmol_l", self.hemoglobin_mmol_l)
        hct = finite_number("hematocrit", self.hematocrit)
        buffer = bounded_number("plasma_buffer_mmol_l", self.plasma_buffer_mmol_l, 0.0)
        if hb <= 0.0:
            raise ValueError("hemoglobin_mmol_l must be greater than zero")
        if not 0.0 < hct < 1.0:
            raise ValueError("hematocrit must be greater than zero and less than one")
        object.__setattr__(self, "hemoglobin_mmol_l", hb)
        object.__setattr__(self, "hematocrit", hct)
        object.__setattr__(self, "plasma_buffer_mmol_l", buffer)

    def equilibrate(
        self, *, ph: float, pco2_mmhg: float, oxygen_saturation: float,
    ) -> BloodChemistryState:
        """Calculate contents and buffer base from plasma pH, PCO2, and saturation."""
        ph = bounded_number("ph", ph, *_PH_RANGE)
        pco2, saturation = self._gas_inputs(pco2_mmhg, oxygen_saturation)
        return self.evaluate(ph, pco2, saturation).state(pco2, saturation)

    def solve_ph(
        self, *, pco2_mmhg: float, oxygen_saturation: float, buffer_base_meq_l: float,
        initial_ph: float = 7.4,
    ) -> BloodChemistryState:
        """Solve pH at a specified whole-blood buffer base and gas state.

        Holding buffer base constant describes an acute respiratory disturbance
        for this fixed composition. Strong-acid addition reduces buffer base.
        initial_ph is a numerical starting point and does not change the result.
        """
        pco2, saturation = self._gas_inputs(pco2_mmhg, oxygen_saturation)
        target = finite_number("buffer_base_meq_l", buffer_base_meq_l)
        ph = bounded_number("initial_ph", initial_ph, *_PH_RANGE)
        return self.solve(pco2, saturation, target, ph).state(pco2, saturation)

    def solve(self, pco2_mmhg: float, oxygen_saturation: float, buffer_base_meq_l: float,
              initial_ph: float = 7.4) -> Equilibrium:
        """solve_ph without input checks or public state, for inner loops.

        PCO2 must be positive, saturation a fraction, and initial_ph within 6.0 to 10.0.
        """
        ph = initial_ph
        low, high = _PH_RANGE
        for _ in range(32):
            equilibrium = self.evaluate(ph, pco2_mmhg, oxygen_saturation)
            residual = equilibrium.buffer_base - buffer_base_meq_l
            if abs(residual) < _CONTENT_TOLERANCE:
                return equilibrium
            if residual > 0.0:
                high = ph
            else:
                low = ph
            candidate = ph - residual / equilibrium.base_ph
            # Evaluate a numerical search boundary when Newton reaches it, so
            # inclusive endpoint solutions do not approach it by bisection.
            candidate = max(_PH_RANGE[0], min(_PH_RANGE[1], candidate))
            if candidate == ph:
                break
            ph = candidate if low <= candidate <= high else (low + high) / 2.0
        raise ValueError("buffer_base_meq_l requires a pH outside the solver range 6.0 to 10.0")

    def solve_contents(self, co2: float, base: float, saturation: float,
                       pco2: float, ph: float, extracellular_ratio: float = 0.0) -> FluidEquilibrium:
        """Invert conserved CO2 and base per liter of blood plus adjacent fluid.

        Contacted extracellular fluid shares the aliquot's pH and PCO2 at equilibrium.
        """
        for _ in range(24):
            iterate = self.iterate_contents(co2, base, saturation, pco2, ph, extracellular_ratio)
            if iterate.converged:
                return FluidEquilibrium(iterate.blood, iterate.pco2)
            ph, pco2 = iterate.ph, iterate.pco2
        raise RuntimeError("Blood gas content solve did not converge")

    def iterate_contents(self, co2: float, base: float, saturation: float,
                         pco2: float, ph: float, extracellular_ratio: float = 0.0) -> ContentIterate:
        """One damped Newton step of solve_contents, for solves coupled to saturation."""
        blood = self.evaluate(ph, pco2, saturation)
        if extracellular_ratio:
            fluid_co2, fluid_base, fluid_ph, fluid_pco2 = extracellular_contents(ph, pco2)
        else:
            fluid_co2 = fluid_base = fluid_ph = fluid_pco2 = 0.0
        # Residuals per liter of blood plus fluid keep the tolerance independent of aliquot size.
        blood_weight = 1.0 / (1.0 + extracellular_ratio)
        fluid_weight = extracellular_ratio * blood_weight
        carbon_residual = blood_weight * (blood.total_co2 - co2) + fluid_weight * fluid_co2
        base_residual = blood_weight * (blood.buffer_base - base) + fluid_weight * fluid_base
        if max(abs(carbon_residual), abs(base_residual)) < _CONTENT_TOLERANCE:
            return ContentIterate(blood, True, ph, pco2)
        b_ph = blood_weight * blood.base_ph + fluid_weight * fluid_ph
        b_p = blood_weight * blood.base_co2 + fluid_weight * fluid_pco2
        c_ph = blood_weight * blood.content_ph + fluid_weight * _LN_10 * fluid_pco2 * pco2
        c_p = blood_weight * blood.content_co2 + fluid_weight * fluid_co2 / pco2
        determinant = b_ph * c_p - b_p * c_ph
        ph_change = (base_residual * c_p - carbon_residual * b_p) / determinant
        pressure_change = (carbon_residual * b_ph - base_residual * c_ph) / determinant
        fraction = step_fraction(ph, ph_change, pco2, pressure_change)
        return ContentIterate(blood, False, ph - fraction * ph_change, pco2 - fraction * pressure_change)

    @staticmethod
    def _gas_inputs(pco2_mmhg: float, oxygen_saturation: float) -> tuple[float, float]:
        pco2 = finite_number("pco2_mmhg", pco2_mmhg)
        if pco2 <= 0.0:
            raise ValueError("pco2_mmhg must be greater than zero")
        saturation = bounded_number("oxygen_saturation", oxygen_saturation, 0.0, 1.0)
        return pco2, saturation

    def evaluate(self, ph: float, pco2: float, saturation: float) -> Equilibrium:
        """Concentrations and pH and PCO2 derivatives at a given pH, without input checks."""
        # Fig. 1, Eq. 24r: desaturation shifts the plasma/red-cell pH relation.
        rbc_ph = 7.19 + 0.77 * (ph - 7.4) + 0.031 * (1.0 - saturation)
        co2_plasma = _PLASMA_CO2_SOLUBILITY * pco2
        co2_rbc = _RBC_CO2_SOLUBILITY * pco2
        plasma_ratio = 10.0**(ph - 6.1)
        rbc_ratio = 10.0**(rbc_ph - 6.1)
        bicarbonate_plasma = co2_plasma * plasma_ratio
        bicarbonate_rbc = co2_rbc * rbc_ratio
        weak_ratio = _PLASMA_BUFFER_RATIO / plasma_ratio
        weak_base = self.plasma_buffer_mmol_l / (1.0 + weak_ratio)
        weak_ph = _LN_10 * weak_base * weak_ratio / (1.0 + weak_ratio)

        hb_rbc = self.hemoglobin_mmol_l / self.hematocrit
        gradient = _RBC_PH_GRADIENT
        # Oxygenated and deoxygenated Hb have different H+ and CO2 affinities.
        # The carbamino equilibrium uses dissolved CO2 in mol/L.
        dissolved_rbc = co2_rbc / 1000.0 * rbc_ratio
        oxy_hb = hb_rbc * saturation
        oxy_proton = _OXY_AMINO_RATIO / rbc_ratio
        oxy_carbamino = dissolved_rbc * _OXY_CARBAMINO_RATIO
        oxy_denominator = 1.0 + oxy_proton + oxy_carbamino
        oxy_amino = oxy_hb / oxy_denominator
        oxy_carb = oxy_amino * oxy_carbamino
        oxy_side = oxy_hb / (1.0 + oxy_proton)
        deoxy_hb = hb_rbc * (1.0 - saturation)
        deoxy_proton = _DEOXY_AMINO_RATIO / rbc_ratio
        deoxy_carbamino = dissolved_rbc * _DEOXY_CARBAMINO_RATIO
        deoxy_denominator = 1.0 + deoxy_proton + deoxy_carbamino
        deoxy_amino = deoxy_hb / deoxy_denominator
        deoxy_carb = deoxy_amino * deoxy_carbamino
        deoxy_side = deoxy_hb / (1.0 + deoxy_proton)

        amino_base = oxy_amino + deoxy_amino
        carbamino = oxy_carb + deoxy_carb
        side_chain_base = oxy_side + deoxy_side
        amino_ph = (gradient * oxy_amino * (oxy_proton - oxy_carbamino) / oxy_denominator
                    + gradient * deoxy_amino * (deoxy_proton - deoxy_carbamino) / deoxy_denominator)
        carbamino_ph = (gradient * oxy_carb * (1.0 + 2.0 * oxy_proton) / oxy_denominator
                        + gradient * deoxy_carb * (1.0 + 2.0 * deoxy_proton) / deoxy_denominator)
        side_chain_ph = (gradient * oxy_side * oxy_proton / (1.0 + oxy_proton)
                         + gradient * deoxy_side * deoxy_proton / (1.0 + deoxy_proton))
        amino_co2 = -(oxy_carb / oxy_denominator / pco2 + deoxy_carb / deoxy_denominator / pco2)
        carbamino_co2 = (oxy_carb * (1.0 + oxy_proton) / oxy_denominator / pco2
                         + deoxy_carb * (1.0 + deoxy_proton) / deoxy_denominator / pco2)

        plasma_base = bicarbonate_plasma + weak_base
        # Eq. 4br: each carbamino group contributes two buffer-base equivalents.
        rbc_base = bicarbonate_rbc + 4.27 * side_chain_base + amino_base + 2.0 * carbamino
        hct = self.hematocrit
        plasma = 1.0 - hct
        buffer_base = plasma * plasma_base + hct * rbc_base
        total_co2 = (
            plasma * (co2_plasma + bicarbonate_plasma)
            + hct * (co2_rbc + bicarbonate_rbc + carbamino)
        )
        base_ph = (plasma * (_LN_10 * bicarbonate_plasma + weak_ph)
                   + hct * (gradient * bicarbonate_rbc + 4.27 * side_chain_ph + amino_ph + 2.0 * carbamino_ph))
        base_co2 = (plasma * bicarbonate_plasma / pco2
                    + hct * (bicarbonate_rbc / pco2 + amino_co2 + 2.0 * carbamino_co2))
        content_ph = (plasma * _LN_10 * bicarbonate_plasma
                      + hct * (gradient * bicarbonate_rbc + carbamino_ph))
        content_co2 = (plasma * (co2_plasma + bicarbonate_plasma) / pco2
                       + hct * ((co2_rbc + bicarbonate_rbc) / pco2 + carbamino_co2))
        return Equilibrium(
            ph, rbc_ph, bicarbonate_plasma, bicarbonate_rbc, plasma_base, rbc_base,
            buffer_base, total_co2, base_ph, base_co2, content_ph, content_co2,
        )
