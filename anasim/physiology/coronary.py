"""Coronary supply-demand balance, ischemia, and ischemic electrical failure."""

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
        self._reference_hr = hr
        # Coronary zero-flow pressure exceeds venous pressure (Nanto 2001). At
        # rest it is calibrated to 20 mmHg.
        reference_cpp = max(10.0, 0.8 * map_mmhg - max(20.0, lap))
        self._supply_scale = reserve / (reference_cpp * self._diastolic_fraction(hr) * 13.5)
        self.supply_demand_ratio = reserve
        self.ischemia = 0.0
        self.energy_reserve = 1.0
        self.energy_failure = 0.0

    @staticmethod
    def _diastolic_fraction(hr: float) -> float:
        return clamp(1.0 - 0.30 * (max(1.0, hr) / 60.0) ** 0.65, 0.15, 0.85)

    def step(self, dt: float, hr: float, sbp: float, dbp: float, lap: float,
             hb: float, saturation: float, hypoxia: float = 0.0) -> None:
        """Advance ischemia toward the supply shortfall; saturation is in percent.

        hypoxia is the hemodynamic model's hypoxemic myocardial depression, 0-1.
        """
        # Basal and activation metabolism persist when mechanical work falls (Harasawa 1992).
        work = hr * max(0.0, sbp) / self._reference_rpp
        demand = max(work, 0.25 + 0.25 * hr / self._reference_hr)
        # The pressure-flow relation is curvilinear at low pressure (Jeremy
        # 1986); a zero-flow pressure that falls with SBP approximates it.
        back_pressure = max(lap, min(20.0, 0.2 * max(0.0, sbp)))
        pressure = max(0.0, dbp - back_pressure)
        oxygen = hb * clamp(saturation / 98.0, 0.0, 1.05)
        self.supply_demand_ratio = self._supply_scale * pressure * self._diastolic_fraction(hr) * oxygen / demand
        target = clamp(1.0 - self.supply_demand_ratio, 0.0, 1.0)
        tau = 15.0 if target > self.ischemia else 60.0
        self.ischemia += (target - self.ischemia) * -math.expm1(-dt / tau)
        # Severe oxygen limitation from either cause first depletes an energy
        # reserve, so a perfusing rhythm can return within about 4 minutes of
        # no flow, as in the electrical phase of arrest (Weisfeldt 2002).
        # Excitation then fails. The timing is a teaching calibration.
        severity = clamp((max(self.ischemia, hypoxia) - 0.5) / 0.5, 0.0, 1.0)
        self.energy_reserve = clamp(self.energy_reserve + dt * (
            (1.0 - severity) * (1.0 - self.energy_reserve) / 120.0 - severity / 240.0), 0.0, 1.0)
        failure_target = severity if self.energy_reserve <= 0.0 else 0.0
        failure_tau = 90.0 if failure_target > self.energy_failure else 60.0
        self.energy_failure += (failure_target - self.energy_failure) * -math.expm1(-dt / failure_tau)

    @property
    def contractility_factor(self) -> float:
        return (1.0 - 0.50 * self.ischemia) * self.electrical_factor

    @property
    def electrically_silent(self) -> bool:
        # Permanent: without a heartbeat or chest compressions, coronary flow cannot return.
        return self.energy_failure >= 0.99

    @property
    def electrical_factor(self) -> float:
        return 0.0 if self.electrically_silent else 1.0 - self.energy_failure

    @property
    def st_depression_mv(self) -> float:
        return 0.30 * self.ischemia
