"""Coronary supply-demand balance, ischemia, and ischemic electrical failure."""

import math

from anasim.core.utils import clamp, clamp01

from .resuscitation import COMPRESSION_DUTY

# Reversible oxygen debt after brief VF clears over seconds once beats return,
# whereas arrest-level depletion recovers over about a minute (Park 1994; Weisfeldt 2002).
REPERFUSION_FAST_TAU_S = 3.5
REPERFUSION_SLOW_TAU_S = 60.0
# High-energy countershocks transiently depress stroke volume by ~10-12%,
# resolving over a few minutes of coronary perfusion (Tokano 1998).
SHOCK_STUN_PER_200J = 0.125
SHOCK_STUN_MAX = 0.25
SHOCK_STUN_BASE_TAU_S = 90.0
SHOCK_STUN_ISCHEMIC_TAU_S = 300.0
# Ischemic injury that reperfusion does not reverse within a session builds up
# over about 7 minutes once the energy reserve is exhausted (Laurent 2002).
INJURY_TIME_S = 420.0


class CoronaryModel:
    """Flow reserve limits supply; rate-pressure product estimates demand.

    Reserve follows Wilson 1987 and demand Gobel 1978; Arthurs 2016 is the basis
    for tracking oxygen imbalance. Pressure/time scaling, response times,
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
        self.shock_stunning = 0.0
        self._shock_stun_tau_s = SHOCK_STUN_BASE_TAU_S
        self.energy_reserve = 1.0
        self.energy_failure = 0.0
        self.injury = 0.0

    def apply_shock(self, energy_j: float = 200.0) -> None:
        """Apply transient contractile stunning from a countershock (Tokano 1998)."""
        scale = clamp(energy_j / 200.0, 0.0, 2.0)
        if scale <= 0.0:
            return
        tau = SHOCK_STUN_BASE_TAU_S + (SHOCK_STUN_ISCHEMIC_TAU_S - SHOCK_STUN_BASE_TAU_S) * self.ischemia
        prev = self.shock_stunning
        new_stun = min(SHOCK_STUN_MAX, 1.0 - (1.0 - prev) * (1.0 - SHOCK_STUN_PER_200J * scale))
        self._shock_stun_tau_s = (prev * self._shock_stun_tau_s + (new_stun - prev) * tau) / new_stun
        self.shock_stunning = new_stun

    @staticmethod
    def _diastolic_fraction(hr: float) -> float:
        return clamp(1.0 - 0.30 * (max(1.0, hr) / 60.0) ** 0.65, 0.15, 0.85)

    def step(self, dt: float, hr: float, sbp: float, dbp: float, lap: float,
             hb: float, saturation: float, hypoxia: float = 0.0, *,
             fibrillating: bool = False, compressions: bool = False) -> None:
        """Advance ischemia toward the supply shortfall; saturation is in percent.

        hypoxia is the hemodynamic model's hypoxemic myocardial depression, 0-1.
        During compressions, dbp is the relaxation pressure between them.
        """
        if fibrillating:
            # Fibrillating myocardium uses most of the beating heart's resting
            # oxygen, less as its energy fails.
            demand = 0.25 + 0.5 * (1.0 - self.energy_failure)
        else:
            # Basal and activation metabolism persist when mechanical work falls (Harasawa 1992).
            work = hr * max(0.0, sbp) / self._reference_rpp
            demand = max(work, 0.25 + 0.25 * hr / self._reference_hr)
        if compressions:
            # Coronary flow occurs between compressions, against the resting zero-flow pressure.
            back_pressure = max(lap, 20.0)
            diastolic_fraction = 1.0 - COMPRESSION_DUTY
        else:
            # The pressure-flow relation is curvilinear at low pressure (Jeremy
            # 1986); a zero-flow pressure that falls with SBP approximates it.
            back_pressure = max(lap, min(20.0, 0.2 * max(0.0, sbp)))
            diastolic_fraction = self._diastolic_fraction(hr)
        pressure = max(0.0, dbp - back_pressure)
        oxygen = hb * clamp(saturation / 98.0, 0.0, 1.05)
        self.supply_demand_ratio = self._supply_scale * pressure * diastolic_fraction * oxygen / demand
        target = clamp01(1.0 - self.supply_demand_ratio)
        if fibrillating or compressions:
            tau_off = REPERFUSION_SLOW_TAU_S
        else:
            tau_off = REPERFUSION_FAST_TAU_S + (REPERFUSION_SLOW_TAU_S - REPERFUSION_FAST_TAU_S) * (
                1.0 - self.energy_reserve
            )
        tau = 15.0 if target > self.ischemia else tau_off
        self.ischemia += (target - self.ischemia) * -math.expm1(-dt / tau)
        if self.shock_stunning > 0.0:
            reperfusion = clamp01(self.supply_demand_ratio)
            self.shock_stunning *= math.exp(-dt * reperfusion / self._shock_stun_tau_s)
            if self.shock_stunning < 1e-4:
                self.shock_stunning = 0.0
                self._shock_stun_tau_s = SHOCK_STUN_BASE_TAU_S
        # Severe oxygen limitation from either cause first depletes an energy
        # reserve, so a perfusing rhythm can return within about 4 minutes of
        # no flow, as in the electrical phase of arrest (Weisfeldt 2002).
        # Excitation then fails and injury accumulates. The timing is a teaching calibration.
        severity = clamp01((max(self.ischemia, hypoxia) - 0.5) / 0.5)
        self.energy_reserve = clamp01(
            self.energy_reserve + dt * ((1.0 - severity) * (1.0 - self.energy_reserve) / 120.0 - severity / 240.0)
        )
        exhausted = self.energy_reserve <= 0.0
        if exhausted:
            self.injury = min(1.0, self.injury + dt * severity / INJURY_TIME_S)
        failure_target = severity if exhausted else 0.0
        failure_tau = 90.0 if failure_target > self.energy_failure else 60.0
        self.energy_failure += (failure_target - self.energy_failure) * -math.expm1(-dt / failure_tau)

    @property
    def contractility_factor(self) -> float:
        return (
            (1.0 - 0.50 * self.ischemia)
            * (1.0 - self.shock_stunning)
            * (1.0 - self.injury)
            * self.electrical_factor
        )

    @property
    def irreversible(self) -> bool:
        """Injury leaves no contraction to recover."""
        return self.injury >= 0.99

    @property
    def electrically_silent(self) -> bool:
        # Without a heartbeat, only chest compressions can restore coronary flow.
        return self.energy_failure >= 0.99

    @property
    def electrical_factor(self) -> float:
        return 0.0 if self.electrically_silent else 1.0 - self.energy_failure

    @property
    def st_depression_mv(self) -> float:
        return 0.30 * self.ischemia
