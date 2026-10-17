"""Gas transport among pulmonary, arterial, and venous blood and two tissue groups.

Blood flow carries CO2, O2, and buffer base and conserves each. The tissue
groups hold calibrated effective CO2 capacities reduced from Magor-Elliott 2021
and exchange with blood along the local CO2 dissociation slope.
"""

import math
from typing import NamedTuple

from .blood_chemistry import BloodChemistry, extracellular_contents

CO2_L_PER_MMOL = 0.022414


class TransportPlan(NamedTuple):
    """End-of-step concentrations as affine functions of pulmonary blood content."""

    arterial_base: float
    arterial_gain: float
    venous_base: float
    venous_gain: float
    tissue_bases: tuple[float, ...] = ()
    tissue_gains: tuple[float, ...] = ()
    partition_volumes: tuple[float, ...] = ()
    tissue_offsets: tuple[float, ...] = ()

    def blood(self, pulmonary: float) -> tuple[float, float]:
        return self.arterial_base + self.arterial_gain * pulmonary, self.venous_base + self.venous_gain * pulmonary


class GasTransport:
    """Blood concentrations, tissue amounts, and extracellular buffer stores."""

    PERFUSION = (0.68, 0.32)
    METABOLISM = (0.65, 0.35)
    EFFECTIVE_STORE_FRACTION = 0.5

    def __init__(self, weight: float, blood_volume_l: float):
        if blood_volume_l <= 0.0:
            raise ValueError("Blood gas transport requires a positive blood volume")
        self.capillary_fraction = 0.070 * weight / 70.0 / blood_volume_l
        self.resize_blood(blood_volume_l)
        # Normal pulmonary capillary volume is about 60-75 mL. The arterial
        # share is a reduced-compartment assumption. The extracellular pool is
        # plasma-equivalent fluid at twice the blood volume (Wooten 2003).
        self.extracellular_volume = 2.0 * blood_volume_l
        scale = weight / 70.0 * self.EFFECTIVE_STORE_FRACTION
        gross_capacities = (0.01748986 * scale, 0.10237336 * scale)
        carbon, _, base_ph, base_co2 = extracellular_contents(7.4, 40.0)
        carbon_ph = math.log(10.0) * base_co2 * 40.0
        extracellular_capacity = self.extracellular_volume * CO2_L_PER_MMOL * (
            carbon / 40.0 - carbon_ph * base_co2 / base_ph)
        # The tissue budget already includes extracellular CO2, so subtract
        # the explicit pool's reference capacity.
        remaining = 1.0 - extracellular_capacity / sum(gross_capacities)
        if remaining <= 0.0:
            raise ValueError("Extracellular CO2 capacity exceeds the effective tissue budget")
        self.capacities = tuple(capacity * remaining / CO2_L_PER_MMOL for capacity in gross_capacities)
        self.co2 = [0.0, 0.0, 0.0]  # mmol/L blood, pulmonary/arterial/venous.
        self.oxygen = [0.0, 0.0, 0.0]  # STPD L/L blood.
        self.base = [0.0, 0.0, 0.0]  # mEq/L blood.
        self.tissue_co2 = [0.0, 0.0]  # mmol.
        self._tissue_ph = [7.36, 7.36]
        self.extracellular_co2 = 0.0  # mmol.
        self.extracellular_base = 0.0  # mEq.
        self.produced_l = self.exhaled_l = self.consumed_o2_l = self.oxygen_from_air_l = 0.0

    def resize_blood(self, volume: float) -> None:
        if volume <= 0.0:
            raise ValueError("Blood gas transport requires a positive blood volume")
        pulmonary = self.capillary_fraction * volume
        arterial = 0.2 * volume
        self.volumes = (pulmonary, arterial, volume - pulmonary - arterial)

    @property
    def tissue_co2_l(self) -> float:
        return sum(self.tissue_co2) * CO2_L_PER_MMOL

    @property
    def co2_l(self) -> float:
        return (sum(v * c for v, c in zip(self.volumes, self.co2, strict=True))
                + sum(self.tissue_co2) + self.extracellular_co2) * CO2_L_PER_MMOL

    @property
    def oxygen_l(self) -> float:
        return sum(v * c for v, c in zip(self.volumes, self.oxygen, strict=True))

    @property
    def base_meq(self) -> float:
        return sum(v * c for v, c in zip(self.volumes, self.base, strict=True)) + self.extracellular_base

    def seed_tissues(self, arterial_co2: float, cardiac_output: float, production_l_min: float,
                     chemistry: BloodChemistry, saturation: float) -> None:
        production = production_l_min / CO2_L_PER_MMOL
        for index, (capacity, flow, metabolism) in enumerate(zip(
            self.capacities, self.PERFUSION, self.METABOLISM, strict=True,
        )):
            fluid = chemistry.solve_contents(
                arterial_co2 + production * metabolism / (cardiac_output * flow),
                self.base[2], saturation, 46.0, 7.36,
            )
            self.tissue_co2[index] = capacity * fluid.pco2
            self._tissue_ph[index] = fluid.blood.ph

    def _plan(self, values: list[float], flow: float, shunt: float, source: float,
              relations: tuple[tuple[float, float], ...] = ()) -> TransportPlan:
        _, va, vv = self.volumes
        cp, ca, cv = values
        pulmonary, bypass = flow * (1.0 - shunt), flow * shunt
        rhs_a = va * ca + pulmonary * (cp - ca) + bypass * (cv - ca)
        bases, gains = [], []
        partitions, offsets = [], []
        if relations:
            outgoing = venous_base = venous_gain = 0.0
            for amount, capacity, (content, slope), fraction, metabolism in zip(
                self.tissue_co2, self.capacities, relations, self.PERFUSION, self.METABOLISM, strict=True,
            ):
                volume = capacity / slope
                offset = content - slope * amount / capacity
                exchange = flow * fraction
                base = (amount + volume * offset + exchange * (ca - content) + source * metabolism) / (volume + exchange)
                gain = exchange / (volume + exchange)
                partitions.append(volume)
                offsets.append(offset)
                bases.append(base)
                gains.append(gain)
                outgoing += fraction * content
                venous_base += fraction * base
                venous_gain += fraction * gain
            rhs_v = vv * cv + flow * (outgoing - cv)
            av = (rhs_v + flow * venous_base) / (vv + flow)
            bv = flow * venous_gain / (vv + flow)
        else:
            av = (vv * cv + flow * (ca - cv) + source) / (vv + flow)
            bv = flow / (vv + flow)
        denominator = va + flow - bypass * bv
        aa = (rhs_a + bypass * av) / denominator
        ba = pulmonary / denominator
        return TransportPlan(aa, ba, av + bv * aa, bv * ba, tuple(bases), tuple(gains), tuple(partitions), tuple(offsets))

    def prepare(self, seconds: float, cardiac_output: float, shunt: float,
                production_l_min: float, consumption_l_min: float, chemistry: BloodChemistry,
                saturation: float) -> tuple[TransportPlan, TransportPlan, float]:
        """Crank-Nicolson advection, coupled to the pulmonary gas equilibrium."""
        minutes = seconds / 60.0
        flow = max(0.0, cardiac_output) * minutes / 2.0
        relations = []
        for index, (amount, capacity) in enumerate(zip(self.tissue_co2, self.capacities, strict=True)):
            pco2 = amount / capacity
            blood = chemistry.evaluate(self._tissue_ph[index], pco2, saturation)
            ph_change = (self.base[2] - blood.buffer_base) / blood.base_ph
            # A Newton step this small from the last pH leaves a quadratic error below the solver tolerance.
            if abs(ph_change) > 1e-5:
                blood = chemistry.solve(pco2, saturation, self.base[2], blood.ph)
                ph_change = 0.0
            self._tissue_ph[index] = blood.ph + ph_change
            relations.append((blood.total_co2 + blood.content_ph * ph_change, blood.co2_capacity))
        carbon = self._plan(self.co2, flow, shunt, production_l_min * minutes / CO2_L_PER_MMOL, tuple(relations))
        oxygen = self._plan(self.oxygen, flow, shunt, -consumption_l_min * minutes)
        base = self._plan(self.base, flow, shunt, 0.0)
        vp = self.volumes[0]
        pulmonary = flow * (1.0 - shunt)
        old_p, _, old_v = self.base
        cp = (vp * old_p + pulmonary * (old_v - old_p + base.venous_base)) / (
            vp + pulmonary * (1.0 - base.venous_gain))
        self.base = [cp, *base.blood(cp)]
        return carbon, oxygen, pulmonary

    def oxygen_blood(self, pulmonary_content: float, plan: TransportPlan,
                     flow: float, shunt: float) -> tuple[float, float, float]:
        """Arterial/venous O2 and venous derivative, limited by available O2."""
        arterial, venous = plan.blood(pulmonary_content)
        if venous >= 0.0:
            return arterial, venous, plan.venous_gain
        _, va, _ = self.volumes
        cp, ca, cv = self.oxygen
        pulmonary, bypass = flow * (1.0 - shunt), flow * shunt
        rhs_a = va * ca + pulmonary * (cp - ca) + bypass * (cv - ca)
        return (rhs_a + pulmonary * pulmonary_content) / (va + flow), 0.0, 0.0

    def finish(self, pulmonary_co2: float, pulmonary_o2: float, carbon: TransportPlan,
               oxygen: TransportPlan, seconds: float, cardiac_output: float, shunt: float,
               production_l_min: float) -> tuple[float, float]:
        """Commit amounts and return O2 uptake and CO2 delivery at the lung."""
        minutes = seconds / 60.0
        flow = max(0.0, cardiac_output) * minutes / 2.0
        pulmonary = flow * (1.0 - shunt)
        vp, va, vv = self.volumes
        old_cp, _, old_cv = self.co2
        arterial_co2, venous_co2 = carbon.blood(pulmonary_co2)
        delivered_co2 = (pulmonary * (old_cv + venous_co2 - old_cp - pulmonary_co2)
                         - vp * (pulmonary_co2 - old_cp)) * CO2_L_PER_MMOL
        old_cp, old_ca, old_cv = self.oxygen
        arterial_o2, venous_o2, _ = self.oxygen_blood(pulmonary_o2, oxygen, flow, shunt)
        uptake_o2 = vp * (pulmonary_o2 - old_cp) - pulmonary * (old_cv + venous_o2 - old_cp - pulmonary_o2)
        used_o2 = va * (old_ca - arterial_o2) + vv * (old_cv - venous_o2) + uptake_o2 - vp * (pulmonary_o2 - old_cp)
        self.co2 = [pulmonary_co2, arterial_co2, venous_co2]
        self.oxygen = [pulmonary_o2, arterial_o2, venous_o2]
        self.tissue_co2 = [v * (a + b * arterial_co2 - offset) for v, a, b, offset in zip(
            carbon.partition_volumes, carbon.tissue_bases, carbon.tissue_gains, carbon.tissue_offsets, strict=True)]
        self.produced_l += production_l_min * minutes
        self.consumed_o2_l += max(0.0, used_o2)
        return uptake_o2, delivered_co2
