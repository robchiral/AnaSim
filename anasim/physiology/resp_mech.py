"""Single-compartment lung mechanics under VCV, PCV, PSV, and CPAP."""

import math
from dataclasses import dataclass
from enum import Enum


class VentMode(Enum):
    VCV = "VCV"
    PCV = "PCV"
    PSV = "PSV"
    CPAP = "CPAP"


@dataclass
class MechState:
    """Instantaneous and last-breath values. Pressures in cmH2O."""

    paw: float = 0.0
    flow: float = 0.0          # L/min, positive during inspiration
    volume: float = 0.0        # L above passive equilibrium at set PEEP
    phase: str = "EXP"         # "INSP" or "EXP"
    paw_peak: float = 0.0
    paw_plat: float = 0.0
    paw_mean: float = 0.0
    auto_peep: float = 0.0
    delivered_vt: float = 0.0  # mL
    rr: float = 0.0            # Active mechanical breath rate, breaths/min
    eelv: float = 0.0          # Trapped end-expiratory volume above that equilibrium (L)


class RespiratoryMechanics:
    """Equation of motion: Paw = V/C + R x flow + PEEP.

    VCV sets a square-wave flow and computes Paw. PCV, PSV, and CPAP set Paw
    and compute flow, so falling compliance lowers delivered VT.
    """

    def __init__(self, compliance: float = 0.05, resistance: float = 10.0):
        """Compliance in L/cmH2O; resistance in cmH2O/(L/s) (ETT about 5-15)."""
        self.compliance = compliance
        self.resistance = resistance

        self.mode = VentMode.VCV
        self.set_rr = 12.0  # breaths/min
        self.set_vt = 0.5  # L
        self.set_peep = 5.0  # cmH2O
        self.set_p_insp = 15.0  # cmH2O above PEEP
        self.insp_time_fraction = 1.0 / 3.0

        self.state = MechState()
        self.cycle_time = 0.0
        self.patient_effort_cmH2O = 0.0

        self._paw_accumulator = 0.0
        self._paw_elapsed = 0.0
        self._breath_peak = 0.0
        self._breath_peak_volume = 0.0

    def set_mode(self, mode: str):
        """Set ventilator mode (VCV, PCV, PSV, CPAP)."""
        try:
            self.mode = VentMode(mode.upper())
        except (AttributeError, ValueError) as exc:
            choices = ", ".join(mode.value for mode in VentMode)
            raise ValueError(f"Unsupported ventilator mode {mode!r}; choose one of: {choices}") from exc
        if self.mode == VentMode.CPAP:
            self.set_p_insp = 0.0

    def set_settings(self, rr: float, vt: float, peep: float, ie: str = "1:2",
                     mode: str = None, p_insp: float = None):
        """Set rate (breaths/min), VT (L), PEEP (cmH2O), I:E such as "1:2",
        mode, and inspiratory pressure above PEEP (cmH2O)."""
        self.set_rr = rr
        self.set_vt = vt
        self.set_peep = peep
        if mode is not None:
            self.set_mode(mode)
        if p_insp is not None:
            self.set_p_insp = p_insp
        elif self.mode == VentMode.CPAP:
            self.set_p_insp = 0.0

        try:
            i, e = map(float, ie.split(':'))
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid I:E ratio {ie!r}; expected a ratio such as '1:2'") from exc
        if i <= 0.0 or e <= 0.0:
            raise ValueError("I:E ratio components must be greater than zero")
        self.insp_time_fraction = i / (i + e)

    def snapshot_settings(self) -> tuple:
        """Return the settings for a temporary override."""
        return (
            self.set_rr,
            self.set_vt,
            self.set_peep,
            self.mode,
            self.set_p_insp,
            self.insp_time_fraction,
            self.patient_effort_cmH2O,
        )

    def restore_settings(self, snapshot: tuple) -> None:
        (
            self.set_rr,
            self.set_vt,
            self.set_peep,
            self.mode,
            self.set_p_insp,
            self.insp_time_fraction,
            self.patient_effort_cmH2O,
        ) = snapshot

    def step(self, dt: float) -> MechState:
        """Integrate each breath phase exactly for fixed settings and mechanics.

        Split at inspiration and breath boundaries, including when one call
        spans several breaths. Pressure means use elapsed time, not sample count.
        """
        if not math.isfinite(dt):
            raise ValueError("mechanics dt must be finite")
        state = self.state
        if dt <= 0.0:
            return state
        if not math.isfinite(self.set_rr) or not 0.0 < self.insp_time_fraction < 1.0:
            raise ValueError("mechanics requires a finite rate and an inspiratory fraction between 0 and 1")
        state.rr = max(0.0, self.set_rr)
        if self.set_rr <= 0.0:
            # No machine breaths: exhale passively to set PEEP.
            self._advance_expiration(dt)
            state.phase = "EXP"
            state.auto_peep = state.volume / self.compliance
            state.eelv = state.volume
            state.delivered_vt = 0.0
            state.paw_peak = state.paw_plat = state.paw_mean = self.set_peep
            self._reset_breath()
            self.cycle_time = 0.0
            return state

        breath_period = 60.0 / self.set_rr
        insp_duration = breath_period * self.insp_time_fraction
        # A rate change can place the clock beyond the new breath boundary.
        if self.cycle_time >= breath_period:
            self._finish_breath()
            self.cycle_time %= breath_period

        remaining = dt
        while remaining > 0.0:
            in_insp = self.cycle_time < insp_duration
            boundary = insp_duration if in_insp else breath_period
            interval = min(remaining, boundary - self.cycle_time)
            if in_insp:
                self._advance_inspiration(interval, insp_duration)
            else:
                self._advance_expiration(interval)
                self._paw_accumulator += self.set_peep * interval
                self._breath_peak = max(self._breath_peak, self.set_peep)
            self._paw_elapsed += interval
            self.cycle_time += interval
            remaining = max(0.0, remaining - interval)

            # Snap roundoff at boundaries so an exact endpoint completes a breath.
            if self.cycle_time >= boundary - 1e-12:
                self.cycle_time = boundary
                if in_insp:
                    state.paw_plat = state.volume / self.compliance + self.set_peep
                else:
                    self._finish_breath()
                    self.cycle_time = 0.0

        # Instantaneous values describe the phase at the end of the interval.
        state.phase = "INSP" if self.cycle_time < insp_duration else "EXP"
        if state.phase == "EXP":
            state.paw = self.set_peep
            state.flow = -state.volume / (self.resistance * self.compliance) * 60.0
        elif self.mode == VentMode.VCV:
            flow = self.set_vt / insp_duration
            state.paw = state.volume / self.compliance + self.resistance * flow + self.set_peep
            state.flow = flow * 60.0
        else:
            drive = self._inspiratory_drive()
            flow = min((drive - state.volume / self.compliance) / self.resistance, 1.5)
            state.paw = state.volume / self.compliance + self.resistance * flow + self.set_peep - self.patient_effort_cmH2O
            state.flow = flow * 60.0
        return state

    def _reset_breath(self) -> None:
        self._breath_peak = 0.0
        self._breath_peak_volume = self.state.volume
        self._paw_accumulator = 0.0
        self._paw_elapsed = 0.0

    def _finish_breath(self) -> None:
        state = self.state
        state.eelv = state.volume
        state.auto_peep = state.eelv / self.compliance
        state.paw_peak = self._breath_peak
        state.delivered_vt = max(0.0, self._breath_peak_volume - state.eelv) * 1000.0
        if self._paw_elapsed > 0.0:
            state.paw_mean = self._paw_accumulator / self._paw_elapsed
        self._reset_breath()

    def _inspiratory_drive(self) -> float:
        support = 0.0 if self.mode == VentMode.CPAP else self.set_p_insp
        return support + self.patient_effort_cmH2O

    def _advance_inspiration(self, dt: float, insp_duration: float) -> None:
        state = self.state
        initial_volume = state.volume
        if self.mode == VentMode.VCV:
            flow = self.set_vt / insp_duration
            state.volume += flow * dt
            base_pressure = self.set_peep + self.resistance * flow
            peak = base_pressure + state.volume / self.compliance
            pressure_area = (base_pressure + (initial_volume + state.volume) / (2.0 * self.compliance)) * dt
        else:
            # V tends to C*(support + effort) with time constant R*C. Trapped
            # volume is already in V; adding auto-PEEP here would count it twice.
            tau = self.resistance * self.compliance
            drive = self._inspiratory_drive()
            target_volume = self.compliance * drive
            max_flow = 1.5  # L/s
            # Integrate the flow-limited part first, then the exponential part.
            limited_time = min(dt, max(0.0, (target_volume - initial_volume - max_flow * tau) / max_flow))
            after_limit = initial_volume + max_flow * limited_time
            applied_pressure = self.set_peep + drive - self.patient_effort_cmH2O
            pressure_area = (
                self.set_peep - self.patient_effort_cmH2O + self.resistance * max_flow
                + (initial_volume + after_limit) / (2.0 * self.compliance)
            ) * limited_time
            state.volume = after_limit + (target_volume - after_limit) * -math.expm1(-(dt - limited_time) / tau)
            pressure_area += applied_pressure * (dt - limited_time)
            end_flow = min((target_volume - state.volume) / tau, max_flow)
            peak = state.volume / self.compliance + self.resistance * end_flow + self.set_peep - self.patient_effort_cmH2O

        self._breath_peak = max(self._breath_peak, peak)
        self._breath_peak_volume = max(self._breath_peak_volume, state.volume)
        self._paw_accumulator += pressure_area

    def _advance_expiration(self, dt: float) -> None:
        state = self.state
        tau = self.resistance * self.compliance
        state.volume *= math.exp(-dt / tau)
        state.flow = -state.volume / tau * 60.0
        state.paw = self.set_peep

    def get_total_peep(self) -> float:
        """Return total PEEP (set + auto)."""
        return self.set_peep + self.state.auto_peep
