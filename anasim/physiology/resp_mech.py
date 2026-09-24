"""Single-compartment lung mechanics under VCV, PCV, PSV, and CPAP."""

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
    volume: float = 0.0        # L above FRC
    phase: str = "EXP"         # "INSP" or "EXP"
    paw_peak: float = 0.0
    paw_plat: float = 0.0
    paw_mean: float = 0.0
    auto_peep: float = 0.0
    delivered_vt: float = 0.0  # mL
    eelv: float = 0.0          # End-expiratory volume above FRC (L)


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
        self._paw_samples = 0
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
        """Restore ventilator settings from a snapshot."""
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
        """Advance by dt seconds and return the updated state."""
        state = self.state
        if self.set_rr <= 0:
            # No machine breaths: exhale passively to PEEP.
            state.phase = "EXP"
            state.flow = 0.0
            state.paw = self.set_peep
            if state.volume > 0.001:
                time_constant = self.resistance * self.compliance
                if time_constant > 0:
                    decay_flow = -state.volume / time_constant
                    state.volume += decay_flow * dt
                    state.volume = max(0, state.volume)
                    state.flow = decay_flow * 60.0
            state.auto_peep = 0.0
            state.eelv = state.volume
            state.delivered_vt = 0.0
            state.paw_peak = state.paw
            state.paw_plat = state.paw
            state.paw_mean = state.paw
            self._breath_peak = 0.0
            self._breath_peak_volume = state.volume
            self._paw_accumulator = 0.0
            self._paw_samples = 0
            self.cycle_time = 0.0
            return state

        breath_period = 60.0 / self.set_rr
        insp_duration = breath_period * self.insp_time_fraction
        self.cycle_time += dt

        if self.cycle_time >= breath_period:
            # New breath: auto-PEEP is the trapped volume over compliance.
            self.cycle_time -= breath_period
            if state.volume > 0.005:
                state.auto_peep = state.volume / self.compliance
            else:
                state.auto_peep = 0.0
            state.eelv = state.volume

            state.paw_peak = self._breath_peak
            vt_l = max(0.0, self._breath_peak_volume - state.eelv)
            state.delivered_vt = vt_l * 1000.0
            if self._paw_samples > 0:
                state.paw_mean = self._paw_accumulator / self._paw_samples

            self._breath_peak = 0.0
            self._breath_peak_volume = state.volume
            self._paw_accumulator = 0.0
            self._paw_samples = 0

        in_insp = self.cycle_time < insp_duration
        phase_changed = (in_insp and state.phase == "EXP") or \
                       (not in_insp and state.phase == "INSP")
        if phase_changed and state.phase == "INSP":
            state.paw_plat = state.volume / self.compliance + self.set_peep

        if in_insp:
            state.phase = "INSP"
            if self.mode == VentMode.VCV:
                target_flow = self._step_vcv_insp(insp_duration)
            else:
                target_flow = self._step_pcv_insp(dt)
        else:
            state.phase = "EXP"
            target_flow = self._step_expiration()

        state.volume += target_flow * dt
        if state.volume < 0:
            state.volume = 0
            if state.phase == "EXP":
                target_flow = 0
        if in_insp and target_flow > 0:
            self._breath_peak_volume = max(self._breath_peak_volume, state.volume)

        paw = self._calculate_paw(target_flow)
        state.paw = paw
        state.flow = target_flow * 60.0

        self._breath_peak = max(self._breath_peak, paw)
        self._paw_accumulator += paw
        self._paw_samples += 1

        return state

    def _step_vcv_insp(self, insp_duration: float) -> float:
        """Square-wave inspiratory flow (L/s) that delivers the set VT."""
        if insp_duration > 0:
            return self.set_vt / insp_duration
        return 0.0

    def _step_pcv_insp(self, dt: float) -> float:
        """Decelerating flow (L/s) = (set Paw - alveolar pressure) / R, at most 1.5 L/s."""
        total_peep = self.set_peep + self.state.auto_peep
        p_insp_total = self.set_p_insp + self.set_peep
        p_alv = self.state.volume / self.compliance + total_peep - self.patient_effort_cmH2O
        pressure_gradient = p_insp_total - p_alv
        if pressure_gradient > 0 and self.resistance > 0:
            return min(pressure_gradient / self.resistance, 1.5)
        return 0.0

    def _step_expiration(self) -> float:
        """Passive expiratory flow (L/s) = -V / (R x C)."""
        time_constant = self.resistance * self.compliance
        if time_constant > 0 and self.state.volume > 0:
            return -self.state.volume / time_constant
        return 0.0

    def _calculate_paw(self, flow: float) -> float:
        """Airway pressure (cmH2O) at the airway opening for a flow in L/s."""
        total_peep = self.set_peep + self.state.auto_peep
        if self.state.phase == "EXP":
            return total_peep
        if self.mode == VentMode.CPAP:
            return self.set_peep
        if self.mode in (VentMode.PCV, VentMode.PSV):
            return self.set_p_insp + self.set_peep
        return self.state.volume / self.compliance + self.resistance * flow + total_peep

    def get_total_peep(self) -> float:
        """Return total PEEP (set + auto)."""
        return self.set_peep + self.state.auto_peep
