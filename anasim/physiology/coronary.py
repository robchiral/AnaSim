"""Coronary supply-demand balance, ischemia, and ischemic electrical failure."""

import math

from anasim.core.constants import TEMP_METABOLIC_COEFFICIENT
from anasim.core.utils import clamp, clamp01

from .resuscitation import COMPRESSION_DUTY

# Calibrated reperfusion time constants: brief-VF recovery takes seconds (Park 1994),
# whereas prolonged arrest-level depletion recovers over about a minute (Weisfeldt and Becker 2002).
REPERFUSION_FAST_TAU_S = 3.5
REPERFUSION_SLOW_TAU_S = 60.0
# Countershocks transiently depress stroke volume by ~10-15%, resolving over a
# few minutes of coronary perfusion (calibrated from 27-34 J ICD shocks in Tokano 1998).
SHOCK_STUN_PER_200J = 0.12
SHOCK_STUN_MAX = 0.25
SHOCK_STUN_BASE_TAU_S = 90.0
SHOCK_STUN_ISCHEMIC_TAU_S = 300.0
# Session-permanent ischemic injury builds up over about 7 minutes once the
# energy reserve is exhausted (calibration; reversible post-arrest stunning in Laurent 2002).
INJURY_TIME_S = 420.0


class CoronaryModel:
    """Flow reserve limits supply; rate-pressure product estimates demand.

    Reserve follows Wilson 1987 and demand Gobel 1978; Arthurs 2016 is the basis
    for tracking oxygen imbalance. Pressure/time scaling, response times,
    contractile depression, and the illustrative ST signal are calibrations.
    """

    def __init__(
        self,
        reserve: float,
        hr: float,
        map_mmhg: float,
        lap: float,
        *,
        sbp_mmhg: float | None = None,
        dbp_mmhg: float | None = None,
        baseline_hb: float = 13.5,
    ):
        # Normalize reserve against the patient's resting SBP, diastolic
        # perfusion pressure (DBP - back pressure), diastolic time, and Hb 13.5 g/dL.
        ref_sbp = 1.35 * map_mmhg if sbp_mmhg is None else max(1.0, sbp_mmhg)
        ref_dbp = 0.8 * map_mmhg if dbp_mmhg is None else max(0.0, dbp_mmhg)
        self._reference_rpp = hr * ref_sbp
        self._reference_hr = hr
        self._reserve = reserve
        self._ref_dbp = ref_dbp
        # Coronary zero-flow pressure exceeds venous pressure (Nanto 2001).
        ref_back = self._zero_flow_pressure(ref_sbp, lap)
        reference_cpp = max(10.0, ref_dbp - ref_back)
        self._supply_scale = reserve / (reference_cpp * self._diastolic_fraction(hr) * 13.5)
        self.supply_demand_ratio = reserve * (max(0.0, baseline_hb) / 13.5)
        self.ischemia = clamp01(1.0 - self.supply_demand_ratio)
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
    def _zero_flow_pressure(sbp_mmhg: float, lap: float) -> float:
        """Coronary zero-flow back-pressure: 15-20 mmHg across perioperative
        hypotension, falling toward zero in extreme shock (canine data in Jeremy 1986;
        Nanto 2001). Above 25 mmHg LAP, a model calibration moderates further whole-wall
        back-pressure rises to reflect preserved subepicardial perfusion.
        """
        eff_lap = 25.0 + 0.60 * (lap - 25.0) if lap > 25.0 else lap
        return max(eff_lap, min(20.0, 0.2 * sbp_mmhg + 6.0 * clamp01((sbp_mmhg - 45.0) / 30.0)))

    @staticmethod
    def _diastolic_fraction(hr: float) -> float:
        return clamp(1.0 - 0.30 * (max(1.0, hr) / 60.0) ** 0.65, 0.15, 0.85)

    @staticmethod
    def _nonneg_pressure(value: float) -> float:
        return 0.0 if math.isnan(value) else max(0.0, value)

    def step(self, dt: float, hr: float, sbp: float, dbp: float, lap: float,
             hb: float, saturation: float, hypoxia: float = 0.0, *,
             fibrillating: bool = False, compressions: bool = False,
             venous_pressure: float = 13.0,
             intrinsic_sbp: float | None = None,
             nonpulsatile_perfusion: bool = False,
             aortic_cross_clamp: bool = False,
             cardioplegia_arrest: bool = False,
             temp_c: float = 37.0) -> None:
        """Advance ischemia toward the supply shortfall; saturation is in percent.

        hypoxia is the hemodynamic model's hypoxemic myocardial depression, 0-1.
        During compressions, dbp is the relaxation pressure between them, while
        active wall-stress demand follows intrinsic_sbp rather than passive chest recoil.
        During unclamped VA ECMO (nonpulsatile_perfusion=True), retrograde aortic root
        pressure perfuses the coronaries continuously even without native pulses.
        During CPB aortic cross-clamp (aortic_cross_clamp=True), coronary inflow ceases
        while cardioplegia_arrest + hypothermia markedly lower metabolic demand.
        """
        sbp_val = self._nonneg_pressure(sbp)
        dbp_val = self._nonneg_pressure(dbp)
        active_sbp = sbp_val if intrinsic_sbp is None else self._nonneg_pressure(intrinsic_sbp)
        temp_q10 = clamp(TEMP_METABOLIC_COEFFICIENT ** (37.0 - clamp(temp_c, 18.0, 38.0)), 0.20, 1.15)
        if cardioplegia_arrest:
            # Cold diastolic cardioplegic arrest lowers basal myocardial VO2 >85-90%
            # and extends safe cross-clamp tolerance well past 60-90 min (Nissinen 2009; Al-Sarraf 2011).
            demand = 0.12 * temp_q10
        elif fibrillating:
            # Fibrillating myocardium uses most of the beating heart's resting
            # oxygen, less as its energy fails; hypothermia slows metabolic demand.
            demand = (0.25 + 0.5 * (1.0 - self.energy_failure)) * temp_q10
        else:
            # Basal and activation metabolism persist when mechanical work falls (isolated canine heart data in Harasawa 1992).
            work = hr * active_sbp / self._reference_rpp
            activation_hr = hr / self._reference_hr
            if compressions and activation_hr > 1.5:
                activation_hr += clamp01((35.0 - active_sbp) / 15.0) * (1.5 - activation_hr)
            demand = max(work, (0.35 + 0.30 * activation_hr) * temp_q10)
        if aortic_cross_clamp:
            pressure = 0.0
            diastolic_fraction = 1.0
        elif compressions:
            # Coronary flow occurs between compressions, against right-atrial/LA
            # back-pressure plus myocardial wall closing margin (~12.8 mmHg above resting CVP).
            back_pressure = max(lap, venous_pressure + 12.8, 20.0)
            diastolic_fraction = 1.0 - COMPRESSION_DUTY
            pressure = max(0.0, dbp_val - back_pressure)
        elif nonpulsatile_perfusion:
            # Continuous aortic-root pressure from VA ECMO perfuses coronaries even when
            # pulse pressure vanishes (e.g., VF or closed aortic valve).
            base_bp = self._zero_flow_pressure(sbp_val, lap)
            back_pressure = max(lap, base_bp)
            diastolic_fraction = 0.85 if (hr <= 0.0 or fibrillating) else self._diastolic_fraction(hr)
            pressure = max(0.0, dbp_val - back_pressure)
        else:
            base_bp = self._zero_flow_pressure(sbp_val, lap)
            pulseless_blend = clamp01((4.95 - (sbp_val - dbp_val)) / 1.65)
            back_pressure = base_bp + pulseless_blend * max(0.0, dbp_val - base_bp)
            diastolic_fraction = self._diastolic_fraction(hr)
            pressure = max(0.0, dbp_val - back_pressure)
        if not compressions and not aortic_cross_clamp and self._reserve < 2.5:
            # Fixed epicardial stenosis exhausts distal autoregulatory dilation, so
            # flow falls steeply as aortic diastolic pressure drops below ~58-60 mmHg.
            stenosis = clamp01((2.5 - self._reserve) / 1.5)
            dbp_knee = min(60.0, 0.85 * self._ref_dbp)
            if dbp_val < dbp_knee:
                dbp_deficit = clamp01((dbp_knee - dbp_val) / 25.0)
                pressure *= 1.0 - 0.55 * stenosis * (dbp_deficit ** 1.3)
        oxygen = hb * clamp(saturation / 98.0, 0.0, 1.05)
        self.supply_demand_ratio = self._supply_scale * pressure * diastolic_fraction * oxygen / demand
        target = clamp01(1.0 - self.supply_demand_ratio)
        if target != self.ischemia:
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
        # no flow, as in the electrical phase of arrest (Weisfeldt and Becker 2002).
        # Excitation then fails and injury accumulates. The timing is a teaching calibration.
        severity = clamp01((max(self.ischemia, hypoxia) - 0.5) / 0.5)
        # Cold cardioplegic protection sustains ~90-120 min cross-clamp tolerance at 28 °C
        # (LCOS rises above 90 min in Al-Sarraf 2011; mortality jumps past 150 min in Nissinen 2009),
        # while hypothermic non-cardioplegic arrest slows ATP depletion with Q10.
        depletion_scale = clamp(0.064 * temp_q10, 0.015, 0.12) if cardioplegia_arrest else temp_q10
        if severity > 0.0 or self.energy_reserve < 1.0:
            self.energy_reserve = clamp01(
                self.energy_reserve
                + dt * ((1.0 - severity) * (1.0 - self.energy_reserve) / 120.0 - severity * depletion_scale / 240.0)
            )
        exhausted = self.energy_reserve <= 0.0
        if exhausted:
            self.injury = min(1.0, self.injury + dt * severity * depletion_scale / INJURY_TIME_S)
        failure_target = severity if exhausted else 0.0
        if failure_target != self.energy_failure:
            failure_tau = 90.0 / max(0.20, temp_q10) if failure_target > self.energy_failure else 60.0
            self.energy_failure += (failure_target - self.energy_failure) * -math.expm1(-dt / failure_tau)

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
