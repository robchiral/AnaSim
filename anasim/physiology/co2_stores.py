"""Reduced CO2 tissue stores informed by Magor-Elliott 2021, Tables 1 and 2.

Fast tissues combine brain, viscera, heart, lungs, and remaining organs; slow
tissues combine muscle, skin, bone, and fat. The reduction omits transit delays
and regional perfusion changes. CO2 amounts are in STPD liters. Half the source
tissue capacity is used as an effective short-term store, calibrated against
Stock 1989's five-minute apneic rise. This is not the full published model.
"""

import math
from collections.abc import Callable


class CarbonDioxideStores:
    """Conservative exchange between nonlinear central blood/lung and two tissues."""

    ACCESSIBLE_TISSUE_FRACTION = 0.5
    # A 20% central blood share approximates pulmonary plus arterial blood.
    CENTRAL_BLOOD_FRACTION = 0.2

    def __init__(self, weight_kg: float, pressure: float, cardiac_output: float,
                 blood_capacity_l_l_mmhg: float, production_l_min: float, blood_volume_l: float):
        # Grouped tissue volume times apparent CO2 solubility, L STPD/mmHg at 70 kg.
        scale = weight_kg / 70.0 * self.ACCESSIBLE_TISSUE_FRACTION
        self.perfusion_fractions = (0.68, 0.32)
        # Remaining blood storage is distributed with tissue perfusion.
        peripheral = (1.0 - self.CENTRAL_BLOOD_FRACTION) * blood_volume_l * blood_capacity_l_l_mmhg
        self.capacities = (0.01748986 * scale + 0.68 * peripheral,
                           0.10237336 * scale + 0.32 * peripheral)
        self.production_fractions = (0.65, 0.35)  # Rounded sums of Table 2 metabolism.
        self.blood_capacity = blood_capacity_l_l_mmhg
        self.equilibrate(pressure, cardiac_output, production_l_min)
        self.produced_l = 0.0
        self.exhaled_l = 0.0

    def equilibrate(self, pressure: float, cardiac_output: float, production_l_min: float) -> None:
        """Seed tissue gradients for a specified steady arterial gas state."""
        self.pressures = [
            pressure + production_l_min * metabolic / (cardiac_output * flow * self.blood_capacity)
            for flow, metabolic in zip(self.perfusion_fractions, self.production_fractions, strict=True)
        ]

    @property
    def tissue_co2_l(self) -> float:
        return sum(capacity * pressure for capacity, pressure in zip(self.capacities, self.pressures, strict=True))

    def step(self, dt_s: float, pressure: float, *, cardiac_output: float,
             production_l_min: float, ventilation_l_min_mmhg: float,
             initial_content_l: float,
             central: Callable[[float], tuple[float, float]]) -> tuple[float, float]:
        """Advance conserved contents with exact tissues for a linear pressure path.

        Central content is solved nonlinearly. Exhalation uses the mean pressure
        of the interval, which callers keep to about one second or less.
        Return pressure and its converged central content.
        """
        minutes = dt_s / 60.0
        flow_capacity = max(0.0, cardiac_output) * self.blood_capacity
        # Each tissue ends at base + gain * (end - start pressure).
        tissue_change_l = tissue_slope = 0.0
        bases, gains = [], []
        for capacity, old, flow, metabolic in zip(
            self.capacities, self.pressures, self.perfusion_fractions,
            self.production_fractions, strict=True,
        ):
            conductance = flow_capacity * flow
            production = production_l_min * metabolic
            if conductance == 0.0:
                base, gain = old + production * minutes / capacity, 0.0
            else:
                rate_time = conductance * minutes / capacity
                exchanged = -math.expm1(-rate_time)
                base = old + exchanged * (pressure - old + production / conductance)
                # Series avoids cancellation at small simulation time steps.
                gain = (rate_time / 2.0 - rate_time**2 / 6.0 + rate_time**3 / 24.0
                        if rate_time < 1e-3 else 1.0 - exchanged / rate_time)
            bases.append(base)
            gains.append(gain)
            tissue_change_l += capacity * (base - old)
            tissue_slope += capacity * gain
        produced = production_l_min * minutes
        ventilation = ventilation_l_min_mmhg * minutes
        candidate = pressure
        for _iteration in range(16):
            content, slope = central(candidate)
            residual = (content - initial_content_l + tissue_change_l
                        + tissue_slope * (candidate - pressure)
                        - produced + ventilation * (pressure + candidate) / 2.0)
            if abs(residual) < 1e-11:
                break
            candidate -= residual / (slope + tissue_slope + ventilation / 2.0)
            if candidate <= 0.0:
                raise RuntimeError("CO2 store step produced a nonpositive pressure")
        else:
            raise RuntimeError("CO2 store solve did not converge")
        self.pressures = [base + gain * (candidate - pressure) for base, gain in zip(bases, gains, strict=True)]
        self.produced_l += produced
        self.exhaled_l += ventilation * (pressure + candidate) / 2.0
        return candidate, content
