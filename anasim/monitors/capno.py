import math
from dataclasses import dataclass

import numpy as np

from anasim.core.utils import clamp


@dataclass
class CapnoState:
    co2: float = 0.0  # mmHg
    phase: int = 4  # 1 dead space, 2 upstroke, 3 plateau, 4 inspiration


@dataclass
class CapnoContext:
    """Timing and cleft parameters for the next breath."""

    exp_duration: float
    is_spontaneous: bool
    curare_active: bool
    effort_scale: float
    spontaneous_weight: float
    effective_rr: float


class Capnograph:
    """Capnogram waveform from alveolar CO2 and breath timing."""

    def __init__(self, rng: np.random.Generator = None):
        self.state = CapnoState()
        self.last_phase = "EXP"
        self.time_in_phase = 0.0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.val_at_change = 0.0
        self.deadspace_fraction = 0.15  # Phase I share of expiration

    @staticmethod
    def build_context(resp_state, vent_rr: float, insp_fraction: float, vent_active: bool) -> CapnoContext:
        """Choose breath timing and whether a curare cleft appears."""
        insp_fraction = clamp(insp_fraction, 0.05, 0.8)
        if vent_active and vent_rr > 0.1:
            drive = max(0.0, resp_state.drive_central)
            muscle = max(0.0, resp_state.muscle_factor)
            spont_rr = max(0.0, resp_state.rr)
            delta_rr = max(0.0, spont_rr - vent_rr)
            effort_signal = drive * muscle

            # Curare cleft: returning drive with partial block, so the patient
            # makes efforts against controlled breaths.
            curare_active = (
                drive > 0.3 and
                0.1 < muscle < 0.9 and
                (effort_signal > 0.15 or delta_rr > 2.0)
            )
            effort_scale = 0.0
            if curare_active:
                effort_scale = min(3.0,
                                   0.2 * delta_rr +
                                   1.5 * effort_signal +
                                   0.3 * (drive - 0.3))

            # Blend controlled and spontaneous timing as the patient takes over.
            rr_ratio = 0.0
            if vent_rr > 0.1:
                rr_ratio = max(0.0, (spont_rr - vent_rr) / max(vent_rr, 1.0))
            rr_weight = clamp(rr_ratio / 0.5, 0.0, 1.0)
            effort_weight = clamp((effort_signal - 0.2) / 0.6, 0.0, 1.0)
            spontaneous_weight = clamp(0.6 * rr_weight + 0.4 * effort_weight, 0.0, 1.0)

            effective_rr = (1.0 - spontaneous_weight) * vent_rr + spontaneous_weight * spont_rr
            exp_fraction = (1.0 - spontaneous_weight) * max(0.1, 1.0 - insp_fraction) + spontaneous_weight * 0.65
            cycle_time = 60.0 / max(effective_rr, 0.1)
            exp_duration = cycle_time * exp_fraction

            is_spontaneous = spontaneous_weight >= 0.6
            if spontaneous_weight >= 0.5:
                curare_active = False
                effort_scale = 0.0

            return CapnoContext(exp_duration, is_spontaneous, curare_active, effort_scale, spontaneous_weight, effective_rr)
        else:
            spont_rr = resp_state.rr
            if spont_rr <= 0.5:
                spont_rr = 12.0
            cycle_time = 60.0 / spont_rr
            exp_duration = cycle_time * 0.65
            return CapnoContext(exp_duration, True, False, 0.0, 1.0, spont_rr)

    def step(self, dt: float, phase: str, p_alv: float, is_spontaneous: bool = False, curare_cleft: bool = False,
             exp_duration: float = 3.0, effort_scale: float = 1.0, airway_obstruction: float = 0.0) -> float:
        """Return instantaneous CO2 (mmHg).

        Args:
            phase: "INSP" or "EXP" from the mechanics.
            p_alv: Alveolar PCO2 (mmHg).
            is_spontaneous: Spontaneous breathing adds plateau noise.
            curare_cleft: Draw a cleft in a controlled-breath plateau.
            exp_duration: Expiratory time (s).
            effort_scale: Cleft depth multiplier.
            airway_obstruction: 0-1; slows the upstroke into a shark fin.
        """
        if phase != self.last_phase:
            self.val_at_change = self.state.co2
            self.time_in_phase = 0.0
            self.last_phase = phase
        else:
            self.time_in_phase += dt

        co2 = 0.0

        if phase == "INSP":
            self.state.phase = 4
            k = 10.0
            co2 = self.val_at_change * math.exp(-k * self.time_in_phase)
        else:
            base_tau = 0.08
            base_slope = 0.1
            tau = base_tau * (1 + 4 * airway_obstruction)
            slope_scale = 2.0 / max(exp_duration, 0.5)
            plateau_slope = base_slope * slope_scale * (1 + 5 * airway_obstruction)
            deadspace_time = self.deadspace_fraction * exp_duration
            if deadspace_time < 0.05:
                deadspace_time = 0.05

            if self.time_in_phase < deadspace_time:
                self.state.phase = 1
                co2 = 0.0
            else:
                t_exp = self.time_in_phase - deadspace_time
                rise_component = 1.0 - math.exp(-t_exp / tau)
                slope_component = plateau_slope * t_exp
                co2 = (p_alv * rise_component) + (slope_component * rise_component)

                if is_spontaneous:
                    noise = self.rng.normal(0, 0.2) * rise_component
                    co2 += noise

                # Curare cleft in the mid-to-late plateau (Bissinger 1993).
                if curare_cleft and not is_spontaneous:
                    exp_effective = max(0.2, exp_duration - deadspace_time)
                    rel_cleft = exp_effective * (0.55 + 0.1 * min(1.0, effort_scale / 2.0))
                    t_cleft = min(exp_effective * 0.85, max(exp_effective * 0.30, rel_cleft))
                    depth = min(p_alv * 0.6, 4.0 + 8.0 * effort_scale)
                    width = max(0.08, 0.12 * (exp_effective / 2.0))
                    dist = abs(t_exp - t_cleft)
                    if dist < 0.5:
                        dynamic_depth = min(p_alv * 0.7, depth + 12.0 * effort_scale)
                        dip = dynamic_depth * math.exp(-(dist**2) / (2 * width**2))
                        co2 -= dip

                plateau_cap = p_alv + 5.0 + 10.0 * airway_obstruction
                co2 = min(co2, plateau_cap)
                if co2 < 0:
                    co2 = 0
                if rise_component < 0.95:
                    self.state.phase = 2
                else:
                    self.state.phase = 3

        if co2 < 0:
            co2 = 0

        self.state.co2 = co2
        return co2
