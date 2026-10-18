"""Coronary supply-demand balance and reversible ischemia."""

import math

from anasim.core.utils import clamp


class CoronaryModel:
    """Flow reserve limits supply; rate-pressure product estimates demand.

    Wilson 1987 informs reserve, Gobel 1978 informs demand, and Arthurs 2016
    motivates tracking oxygen imbalance. Pressure/time scaling, response times,
    contractile depression, and the illustrative ST signal are calibrations.
    """

    def __init__(self, reserve: float, hr: float, map_mmhg: float, lap: float):
        # Normalize reserve at estimated resting SBP (1.35 x MAP), diastolic
        # perfusion pressure (0.8 x MAP - LAP), diastolic time, and Hb 13.5 g/dL.
        self._reference_rpp = hr * 1.35 * map_mmhg
        reference_cpp = max(10.0, 0.8 * map_mmhg - lap)
        self._supply_scale = reserve / (reference_cpp * self._diastolic_fraction(hr) * 13.5)
        self.supply_demand_ratio = reserve
        self.ischemia = 0.0

    @staticmethod
    def _diastolic_fraction(hr: float) -> float:
        return clamp(1.0 - 0.30 * (max(1.0, hr) / 60.0) ** 0.65, 0.15, 0.85)

    def step(self, dt: float, hr: float, sbp: float, dbp: float, lap: float,
             hb: float, saturation: float) -> None:
        """Advance ischemia toward the supply shortfall; saturation is in percent."""
        demand = max(0.1, hr * max(0.0, sbp) / self._reference_rpp)
        pressure = max(0.0, dbp - lap)
        oxygen = hb * clamp(saturation / 98.0, 0.0, 1.05)
        self.supply_demand_ratio = self._supply_scale * pressure * self._diastolic_fraction(hr) * oxygen / demand
        target = clamp(1.0 - self.supply_demand_ratio, 0.0, 1.0)
        tau = 15.0 if target > self.ischemia else 60.0
        self.ischemia += (target - self.ischemia) * -math.expm1(-dt / tau)

    @property
    def contractility_factor(self) -> float:
        return 1.0 - 0.50 * self.ischemia

    @property
    def st_depression_mv(self) -> float:
        return 0.30 * self.ischemia
