import math

import numpy as np

from anasim.core.utils import clamp, clamp01

from .cardiac_cycle import CardiacCycleSample

_PPG_TEMPLATE_RESOLUTION = 200
_PPG_LANDMARKS_X = np.array([0.0, 0.175, 0.40, 0.45, 1.0])
_PPG_LANDMARKS_Y = np.array([0.0, 1.0, 0.49, 0.51, 0.0])
_REACQUIRE_WINDOW_S = 0.12
_REACQUIRE_TAU_S = 0.012


def _build_ppg_template(resolution: int = _PPG_TEMPLATE_RESOLUTION) -> np.ndarray:
    """One pleth beat by smoothstep interpolation between landmarks (after neurokit2)."""
    phase_hr = np.linspace(0.0, 1.0, resolution)
    template = np.zeros(resolution)

    for i in range(len(_PPG_LANDMARKS_X) - 1):
        x0, x1 = _PPG_LANDMARKS_X[i], _PPG_LANDMARKS_X[i + 1]
        y0, y1 = _PPG_LANDMARKS_Y[i], _PPG_LANDMARKS_Y[i + 1]
        mask = (phase_hr >= x0) & (phase_hr <= x1)
        if not np.any(mask):
            continue

        t = (phase_hr[mask] - x0) / (x1 - x0)
        smooth_t = t * t * (3 - 2 * t)
        template[mask] = y0 + (y1 - y0) * smooth_t

    return template


_PPG_TEMPLATE = _build_ppg_template()


def pulse_ox_methemoglobin_spo2(
    functional_sao2_pct: float,
    methb_fraction: float,
    *,
    mb_optical_dip: float = 0.0,
) -> float:
    """Two-wavelength (660/940 nm) pulse oximeter optical reading under methemoglobinemia.

    Grounded in Barker et al. 1989 (Anesthesiology 70:112-117): MetHb absorbs
    strongly at both 660 and 940 nm with an extinction ratio near 1.0, pulling
    SpO2 toward an ~85% plateau at MetHb >= 30-35% despite normal PaO2 and
    progressively flattening sensitivity to true hypoxemia (slope ~0.10-0.12
    above 45% MetHb). IV methylene blue dye also causes a brief false optical
    dip (Guay 2009). Reduces identically to functional_sao2_pct when MetHb and
    dye dip are zero.
    """
    sat = clamp(functional_sao2_pct, 0.0, 100.0)
    m = clamp(methb_fraction, 0.0, 0.90)
    dip = max(0.0, mb_optical_dip)
    if m <= 0.0 and dip <= 0.0:
        return sat
    if m <= 0.0:
        return clamp(sat - dip, 0.0, 100.0)
    s = sat / 100.0
    # Standard two-wavelength calibration SpO2 = 110 - 25 * (A_660 / A_940) is exact
    # at m = 0 with R_0(s) = 4.4 - 4.0 * s and IR_0 = 1.0.
    r_0 = 4.4 - 4.0 * s
    a_660 = (1.0 - m) * r_0 + m * 9.5
    a_940 = (1.0 - m) * 1.0 + m * 9.0
    return clamp(110.0 - 25.0 * (a_660 / a_940) - dip, 0.0, 100.0)


class SpO2Monitor:
    """Finger pulse oximeter: delayed pleth, circulatory transit lag, and multi-beat lock."""

    def __init__(
        self,
        response_tau_s: float = 2.0,
        peripheral_delay_s: float = 0.18,
        transport_delay_s: float = 6.5,
    ):
        self.response_tau_s = response_tau_s
        self.peripheral_delay_s = peripheral_delay_s
        self.transport_delay_s = transport_delay_s
        if self.response_tau_s <= 0.0:
            raise ValueError("response_tau_s must be greater than zero")
        if self.peripheral_delay_s < 0.0:
            raise ValueError("peripheral_delay_s cannot be negative")
        if self.transport_delay_s < 0.0:
            raise ValueError("transport_delay_s cannot be negative")
        self.display_saturation: float | None = None
        self.signal_valid = True
        self._reacquire_s = 0.0
        self._valid_beats = 0
        self._valid_s = 0.0
        self._transit: list[float] | None = None
        self._template = _PPG_TEMPLATE.tolist()  # Python floats avoid NumPy scalar arithmetic per sample.
        self._template_max_index = _PPG_TEMPLATE.size - 1
        self._beat_amplitude = 1.0
        self._beat_perf = 1.0
        self._beat_rr_s: float | None = None
        self._beat_pending = False
        self._has_pulse = False
        self._transit_key = (-1.0, -1.0)
        self._transit_alpha = 0.0
        self._resp_key = (-1.0, -1.0)
        self._resp_alpha = 0.0

    def step(
        self,
        dt: float,
        cycle: CardiacCycleSample,
        saturation: float,
        perfusion: float = 1.0,
        *,
        cpr_artifact: bool = False,
        delay_scale: float = 1.0,
    ) -> tuple[float, float]:
        """Return (pleth voltage, displayed saturation %)."""
        if dt <= 0.0:
            raise ValueError("SpO2 monitor dt must be greater than zero")

        perf = clamp01(perfusion)
        target = clamp(saturation, 0.0, 100.0)
        if self._transit is None:
            self._transit = [target, target, target, target]
        if self.display_saturation is None:
            self.display_saturation = target

        if cycle.organized and not cpr_artifact:
            # Each pulse starts at its upstroke with the beat's stroke volume, and the
            # previous pulse keeps its own timing until then. On the first beat after
            # loss of pulse, wait until peripheral_delay_s before rendering the upstroke.
            delay_s = self.peripheral_delay_s
            self._beat_pending |= cycle.beat_started
            if self._beat_rr_s is None or (self._beat_pending and cycle.elapsed_s >= delay_s):
                self._beat_amplitude = cycle.stroke_fraction
                self._beat_perf = perf
                self._beat_rr_s = cycle.rr_interval_s
                self._beat_pending = False
            if not self._has_pulse and cycle.elapsed_s >= delay_s:
                self._has_pulse = True

            if perf <= 0.0 or not self._has_pulse:
                pleth_voltage = 0.0
            else:
                if self._beat_pending:
                    phase = min((self._beat_rr_s - delay_s + cycle.elapsed_s) / self._beat_rr_s, 1.0)
                else:
                    phase = ((cycle.elapsed_s - delay_s) / self._beat_rr_s) % 1.0
                idx = int(phase * self._template_max_index)
                pleth_voltage = self._template[idx] * self._beat_perf * self._beat_amplitude
        else:
            self._beat_amplitude = 1.0
            self._beat_perf = 0.0
            self._beat_rr_s = None
            self._beat_pending = False
            self._has_pulse = False
            pleth_voltage = 0.0

        # Advance lung-to-finger arterial O2 transit while forward perfusion exists;
        # delay lengthens with hypothermia and vasoconstriction (MacLeod 2005) as well as low CO.
        if cycle.organized and perf > 0.0 and not cpr_artifact:
            eff_delay = clamp(self.transport_delay_s * max(0.5, delay_scale), 0.0, 120.0)
            if eff_delay <= 0.0:
                self._transit = [target, target, target, target]
            else:
                stage_tau = eff_delay / 4.0
                key = (dt, stage_tau)
                if key != self._transit_key:
                    self._transit_key = key
                    self._transit_alpha = -math.expm1(-dt / stage_tau)
                step_alpha = self._transit_alpha
                prev = target
                transit = self._transit
                for i in range(4):
                    transit[i] += step_alpha * (prev - transit[i])
                    prev = transit[i]

        usable = cycle.organized and perf >= 0.08 and not cpr_artifact
        if not usable:
            self.signal_valid = False
            self._valid_beats = 0
            self._valid_s = 0.0
            self._reacquire_s = _REACQUIRE_WINDOW_S
            return pleth_voltage, self.display_saturation

        if not self.signal_valid:
            self._valid_s += dt
            if cycle.beat_started:
                self._valid_beats += 1
            if (self._valid_beats >= 2 and self._valid_s >= 1.2) or self._valid_s >= 2.5:
                self.signal_valid = True
                # Stagnant peripheral blood during pulse loss does not hold stale
                # pre-arrest saturation above current arterial saturation on return.
                self._transit = [min(val, target) for val in self._transit]
            else:
                return pleth_voltage, self.display_saturation

        finger_sat = self._transit[-1]
        # Re-acquired pulses lock onto current peripheral saturation rapidly;
        # steady tracking trails finger arterial saturation with perfusion-dependent averaging.
        if self._reacquire_s > 0.0:
            tau = _REACQUIRE_TAU_S
            self._reacquire_s = max(0.0, self._reacquire_s - dt)
        else:
            tau = self.response_tau_s * (1.0 + 2.0 * (1.0 - perf))
        resp_key = (dt, tau)
        if resp_key != self._resp_key:
            self._resp_key = resp_key
            self._resp_alpha = -math.expm1(-dt / tau)
        self.display_saturation += self._resp_alpha * (finger_sat - self.display_saturation)
        return pleth_voltage, self.display_saturation
