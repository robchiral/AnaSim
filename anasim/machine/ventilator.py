"""Ventilator settings and breath monitors."""

from dataclasses import dataclass


@dataclass
class VentSettings:
    """Settings: VT mL, rate breaths/min, pressures cmH2O above atmosphere."""

    mode: str = "VCV"
    tv: float = 500.0
    rr: float = 12.0
    peep: float = 5.0
    ie_ratio: float = 0.5  # I / E
    fio2: float = 0.21
    p_insp: float = 15.0  # Above PEEP


@dataclass
class VentMonitors:
    """Last-breath monitors: pressures cmH2O, VT mL, MV L/min, compliance mL/cmH2O."""

    paw_peak: float = 0.0
    paw_plat: float = 0.0
    paw_mean: float = 0.0
    auto_peep: float = 0.0
    mv_exp: float = 0.0
    tv_exp: float = 0.0
    rr_total: float = 0.0
    compliance: float = 50.0


class AnesthesiaVentilator:
    """Stores settings; RespiratoryMechanics produces the breaths."""

    def __init__(self):
        self.settings = VentSettings()
        self.monitors = VentMonitors()
        self.is_on = True

    def set_mode(self, mode: str):
        mode_upper = mode.upper()
        if mode_upper in ["VCV", "PCV", "PSV", "CPAP"]:
            self.settings.mode = mode_upper

    def update_settings(self, rr=None, tv=None, peep=None, fio2=None,
                       ie=None, p_insp=None, mode=None):
        """Update any given setting; VT in mL, I:E as "1:2" or a float."""
        if rr is not None:
            self.settings.rr = rr
        if tv is not None:
            self.settings.tv = tv
        if peep is not None:
            self.settings.peep = peep
        if fio2 is not None:
            self.settings.fio2 = fio2
        if p_insp is not None:
            self.settings.p_insp = p_insp
        if mode is not None:
            self.set_mode(mode)

        if ie is not None:
            if isinstance(ie, str) and ':' in ie:
                parts = ie.split(':')
                i, e = float(parts[0]), float(parts[1])
                if i <= 0.0 or e <= 0.0:
                    raise ValueError("I:E ratio components must be greater than zero")
                self.settings.ie_ratio = i / e
            else:
                ratio = float(ie)
                if ratio <= 0.0:
                    raise ValueError("I:E ratio must be greater than zero")
                self.settings.ie_ratio = ratio

    def step(self, dt: float, mech_state, rr_total: float = None):
        """Copy breath monitors from the mechanics state."""
        self.monitors.paw_peak = mech_state.paw_peak
        self.monitors.paw_plat = mech_state.paw_plat
        self.monitors.paw_mean = mech_state.paw_mean
        self.monitors.auto_peep = mech_state.auto_peep
        self.monitors.tv_exp = mech_state.delivered_vt

        rr_eff = rr_total if rr_total is not None else self.settings.rr
        if rr_eff > 0:
            self.monitors.mv_exp = (self.monitors.tv_exp / 1000.0) * rr_eff
        else:
            self.monitors.mv_exp = 0.0

        # Compliance = VT / driving pressure (plateau - total PEEP in VCV).
        if self.settings.mode == "VCV":
            delta_p = mech_state.paw_plat - self.settings.peep - mech_state.auto_peep
            if delta_p > 0.5:
                self.monitors.compliance = self.monitors.tv_exp / delta_p
            else:
                self.monitors.compliance = 50.0
        elif self.settings.p_insp and self.settings.p_insp > 0:
            self.monitors.compliance = self.monitors.tv_exp / self.settings.p_insp

        self.monitors.rr_total = rr_total if rr_total is not None else self.settings.rr

