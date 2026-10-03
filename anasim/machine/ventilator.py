"""Anesthesia workstation ventilation and spirometry."""

import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

from anasim.physiology.resp_mech import RespiratoryMechanics, find_root

MODES = ("VCV", "PCV", "PCV-VG", "SIMV-VC", "SIMV-PC", "SIMV-VG", "PSV", "CPAP")
# Mandatory breath type of each mode. SIMV, PSV, and CPAP respond to patient triggers.
MANDATORY = {"VCV": "VC", "PCV": "PC", "PCV-VG": "VG", "SIMV-VC": "VC", "SIMV-PC": "PC", "SIMV-VG": "VG"}
TRIGGERED = ("SIMV-VC", "SIMV-PC", "SIMV-VG", "PSV", "CPAP")

# Circle-system limb resistance between the Y-piece and the bag or expiratory
# valve (cmH2O/(L/s)), fitted with the PEEP valve to Primus recordings. See
# docs/REFERENCES.md#ventilator-waveforms.
CIRCUIT_RESISTANCE = 1.0
# Exhaled gas leaves through the PEEP valve, whose pressure drop above PEEP
# grows with the square of flow, as through an orifice. Fitted to Primus
# recordings at PEEP 5 cmH2O; recordings at zero PEEP show no added drop, so it
# is scaled down linearly below 5 cmH2O.
PEEP_VALVE = 7.0  # cmH2O/(L/s)^2
PEEP_VALVE_FULL = 5.0  # cmH2O
# Seconds after exhalation begins at which the valve is linearized about the
# flow it then carries, keeping segments exact; the last ends its effect.
VALVE_KNOTS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8,
               0.9, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0)
RISE_TIME = 0.28  # s, pressure ramp of pressure-controlled and supported breaths (Primus recordings)
TRIGGER_WINDOW = 0.25  # SIMV synchronizes to efforts in the last quarter of each period
CYCLE_FRACTION = 0.25  # Supported breaths end when flow falls to this share of its peak
APNEA_BACKUP_S = 20.0  # PSV starts backup breaths after this long without a breath
VG_STEP = 3.0  # Largest VG pressure change per breath, cmH2O
VG_MIN = 2.0  # cmH2O above PEEP
EVENT_STEP = 0.02  # s; longest interval searched for one trigger or cycling crossing
RECENT_BREATHS = 4  # Breaths averaged for RR and MV
APNEA_S = 15.0  # RR and MV read zero after this long without a breath
MEASURE_FLOW = 0.05  # L/s; detects inspiration independently of ventilator trigger eligibility.
MIN_EXPIRATION_L = 0.05  # A breath exhaling less is still in progress; brief reversals neither end nor start one.


@dataclass
class VentSettings:
    """Settings: VT mL, rate breaths/min, pressures cmH2O (Pinsp and PS above PEEP), times s."""

    mode: str = "VCV"
    tv: float = 500.0
    rr: float = 12.0  # Mandatory rate; PSV backup rate
    peep: float = 5.0
    ie_ratio: float = 0.5  # I / E for VCV, PCV, and PCV-VG
    t_insp: float = 1.7  # SIMV mandatory and PSV backup breaths; longest supported breath
    p_insp: float = 15.0
    p_support: float = 5.0
    pause: float = 10.0  # % of Ti in volume-controlled breaths
    p_max: float = 40.0  # Pressure limit of volume-targeted breaths
    trigger: float = 3.0  # L/min
    fio2: float = 0.21

    @property
    def ie(self) -> str:
        """I:E ratio as offered by the controls, such as "1:2"."""
        return f"1:{max(1, round(1.0 / self.ie_ratio))}" if self.ie_ratio > 0 else "1:2"


@dataclass
class VentMonitors:
    """Spirometry at the airway: pressures cmH2O, volumes mL, MV L/min, compliance mL/cmH2O."""

    paw_peak: float = 0.0
    paw_plat: float = math.nan  # End-inspiratory pressure of the last mandatory breath with no flow
    paw_mean: float = 0.0
    peep: float = 0.0  # End-expiratory pressure
    auto_peep: float = 0.0  # Static recoil left at end-expiration
    tv_insp: float = 0.0
    tv_exp: float = 0.0
    tv_exp_mandatory: float = math.nan
    mv_exp: float = 0.0  # Recent breaths
    rr_total: float = 0.0
    compliance_dyn: float = math.nan  # VTe / (Ppeak - PEEP)


@dataclass
class Breath:
    kind: str  # VC, PC, VG, PS, SPONT (detected, unsupported), or MANUAL
    mandatory: bool
    ti: float  # Inspiratory time; the longest allowed for PS and SPONT
    flow_end: float  # End of VC flow; equals ti otherwise
    target: float  # VC and MANUAL: VT (L); pressure breaths: cmH2O above PEEP
    peep: float
    v_start: float
    t: float = 0.0
    inspiring: bool = True  # Includes the inspiratory pause
    limited: bool = False  # VC flow handed to the pressure limit
    delivered: bool = False  # VC volume reached under the limit
    vg_test: bool = False  # First PCV-VG breath: volume control, to measure compliance
    peak_flow: float = 0.0
    v_max: float = -math.inf  # Largest volume during inspiration
    plat: float = math.nan


@dataclass
class MeasuredBreath:
    kind: str
    mandatory: bool
    t: float = 0.0
    inspiring: bool = True
    inspired: float = 0.0
    expired: float = 0.0
    paw_peak: float = -math.inf
    area: float = 0.0
    plat: float = math.nan


class AnesthesiaVentilator:
    """Mechanical, manual, and spontaneous breathing through the circle system.

    The ventilator applies flow or pressure to RespiratoryMechanics in exact
    segments, split where breaths trigger, cycle, or reach Pmax. Spirometry
    measures every breath through the circuit, as workstation flow sensors do.
    Settings take effect at the next breath and persist while it is off.
    """

    def __init__(self):
        self.settings = VentSettings()
        self.monitors = VentMonitors()
        self.is_on = False
        self.apnea_backup_s = APNEA_BACKUP_S
        self.paw = 0.0  # cmH2O at the Y-piece
        self.flow = 0.0  # L/min into the patient
        self.volume = 0.0  # L above relaxed volume at zero PEEP
        self.inspiring = False
        self.breath_count = 0
        self.pressure_limited = False  # Last volume-targeted breath fell short at its limit
        self._source = None
        self._bag = (12.0, 0.5)
        self._mode = None
        self._baseline = None  # PEEP the lung volume is referenced to; the first one applied
        self._breath = None
        self._measured = None
        self._since_measured = 0.0
        self.samples = []  # Resolved (dt, dV L, Paw, flow L/min, absolute volume L).
        self._collect_samples = False
        self._paw_end = 0.0
        self._current = None  # (segment, time) valid at the end of the step
        self._since_mandatory = 0.0
        self._since_breath = 0.0
        self._since_exhalation = 0.0
        self._valve = None  # (knot, offset, resistance) linearizing the PEEP valve
        self._backup = False
        self._vg_pressure = None
        self._armed = False
        self._recent = deque(maxlen=RECENT_BREATHS)
        self._recent_totals = (0.0, 0.0)  # Duration s and exhaled volume L

    def set_mode(self, mode: str):
        if mode.upper() in MODES:
            self.settings.mode = mode.upper()

    @property
    def has_measured_breath(self) -> bool:
        return bool(self._recent)

    def update_settings(self, rr=None, tv=None, peep=None, fio2=None, ie=None, p_insp=None, mode=None, **extra):
        """Update any given setting; VT in mL, I:E as "1:2" or a float."""
        for name, value in dict(rr=rr, tv=tv, peep=peep, fio2=fio2, p_insp=p_insp, **extra).items():
            if value is not None:
                if not hasattr(self.settings, name):
                    raise ValueError(f"Unknown ventilator setting {name!r}")
                setattr(self.settings, name, float(value))
        if mode is not None:
            self.set_mode(mode)
        if ie is not None:
            if isinstance(ie, str) and ':' in ie:
                i, e = map(float, ie.split(':'))
                if i <= 0.0 or e <= 0.0:
                    raise ValueError("I:E ratio components must be greater than zero")
                self.settings.ie_ratio = i / e
            else:
                ratio = float(ie)
                if ratio <= 0.0:
                    raise ValueError("I:E ratio must be greater than zero")
                self.settings.ie_ratio = ratio

    # --- Configuration of the active source --------------------------------

    def _peep(self) -> float:
        return self.settings.peep if self._source == "vent" else 0.0

    def _series_resistance(self) -> float:
        """Resistance between the Y-piece and the baseline pressure source."""
        return 0.0 if self._source is None else CIRCUIT_RESISTANCE

    def _mandatory_period(self) -> float:
        rr = self._bag[0] if self._source == "bag" else self.settings.rr
        return 60.0 / rr if rr > 0.0 else math.inf

    def _triggering(self) -> bool:
        return self._source == "spontaneous" or (self._source == "vent" and self.settings.mode in TRIGGERED)

    def _pressure_cap(self, peep: float) -> float:
        """Pmax above PEEP for volume-targeted breaths."""
        return self.settings.p_max - peep if self._source == "vent" else math.inf

    # --- Stepping ------------------------------------------------------------

    def step(self, dt: float, lung: RespiratoryMechanics, source: str | None, bag: tuple = (12.0, 0.5),
             collect_samples: bool = False) -> None:
        """Advance dt seconds.

        source is "vent", "bag", "spontaneous" (breathing through the circuit),
        or None when the airway is disconnected and the machine measures nothing.
        """
        if dt <= 0.0:
            return
        self.samples.clear()
        self._collect_samples = collect_samples
        self._bag = bag
        if source != self._source:
            self._switch(source, lung)
        if self._source == "vent" and self.settings.mode != self._mode:
            if MANDATORY.get(self.settings.mode) == "VG" and MANDATORY.get(self._mode) != "VG":
                self._vg_pressure = None
            self._mode = self.settings.mode
        if not self.inspiring:
            self._rebase(lung, self._peep())
        lung.effort.start(lung, self._series_resistance())
        remaining = dt
        while remaining > 1e-12:
            remaining -= self._advance(remaining, lung)
        self._update_rates()
        self._sample(lung)

    def _switch(self, source, lung) -> None:
        self._source = source
        self._mode = self.settings.mode if source == "vent" else None
        self._vg_pressure = None
        self._backup = False
        self._breath = None
        self._measured = None
        self._since_measured = 0.0
        self._recent.clear()
        self.monitors = VentMonitors()
        self.monitors.peep = self._peep()
        self.pressure_limited = False
        self._since_breath = 0.0
        self._since_exhalation, self._valve = 0.0, None
        self._rebase(lung, self._peep())
        mandatory = self._mandatory_type()
        if mandatory is not None:
            self._start(lung, mandatory, True)
        else:
            self.inspiring = False
            self._armed = False

    def _mandatory_type(self):
        if self._source == "bag":
            return "MANUAL"
        if self._source != "vent":
            return None
        if self.settings.mode == "PSV":
            return "PC" if self._backup else None
        return MANDATORY.get(self.settings.mode)

    def _rebase(self, lung, peep: float) -> None:
        """Keep absolute lung volume continuous when the set PEEP changes."""
        if self._baseline is not None and peep != self._baseline:
            lung.volume -= lung.compliance * (peep - self._baseline)
        self._baseline = peep

    def _start(self, lung, kind: str, mandatory: bool) -> None:
        self._begin_measurement(lung, kind, mandatory)
        s = self.settings
        peep = self._peep()
        self._rebase(lung, peep)
        period = self._mandatory_period()
        if self._source == "bag":
            ti = period / 3.0
        elif s.mode in MANDATORY and not s.mode.startswith("SIMV"):
            ti = period * s.ie_ratio / (1.0 + s.ie_ratio)
        else:
            ti = min(s.t_insp, 0.8 * period)
        flow_end, target, vg_test = ti, 0.0, False
        if kind == "VG" and self._vg_pressure is None:
            kind, vg_test = "VC", True
        if kind == "VC":
            pause = max(s.pause, 10.0) if vg_test else s.pause
            flow_end = ti * (1.0 - min(max(pause, 0.0), 60.0) / 100.0)
            target = s.tv / 1000.0
        elif kind == "MANUAL":
            target = self._bag[1]
        elif kind == "PC":
            target = s.p_insp
        elif kind == "VG":
            target = self._vg_pressure
        elif kind == "PS":
            target, ti = s.p_support, s.t_insp
            flow_end = ti
        else:  # SPONT
            ti = flow_end = s.t_insp
        self._breath = Breath(kind, mandatory, ti, flow_end, target, peep, lung.volume, vg_test=vg_test,
                              v_max=lung.volume)
        self.inspiring = True
        self._armed = False
        self._since_breath = 0.0
        if mandatory:
            self._since_mandatory = 0.0

    def _finish(self, lung) -> None:
        """Record the breath that ends as the next one starts."""
        b = self._measured
        if self._source is None or b is None or b.t <= 0.0:
            return
        m = self.monitors
        vte = b.expired
        m.paw_peak = b.paw_peak
        m.paw_mean = b.area / b.t
        if self._breath is None or not self._breath.inspiring:
            m.peep = self._paw_end
            m.auto_peep = lung.elastic_pressure
        m.tv_insp = b.inspired * 1000.0
        m.tv_exp = vte * 1000.0
        if b.kind != "SPONT":
            driving = b.paw_peak - self._paw_end
            m.compliance_dyn = m.tv_exp / driving if driving > 0.5 else math.nan
        if b.mandatory:
            m.tv_exp_mandatory = m.tv_exp
            m.paw_plat = b.plat
        self._recent.append((b.t, vte))
        self._recent_totals = (sum(d for d, _ in self._recent), sum(v for _, v in self._recent))
        self.breath_count += 1

    def _begin_measurement(self, lung, kind: str, mandatory: bool) -> None:
        # A scheduled inflation before a spontaneous breath has exhaled belongs
        # to the same measured breath; controller timing stays separate.
        b = self._measured
        if b is not None and b.expired < MIN_EXPIRATION_L and not b.mandatory and mandatory:
            b.kind, b.mandatory = kind, True
            return
        self._finish(lung)
        self._measured = MeasuredBreath(kind, mandatory)
        self._since_measured = 0.0

    def _update_rates(self) -> None:
        if not self._recent or self._since_measured >= APNEA_S:
            self.monitors.rr_total = self.monitors.mv_exp = 0.0
            return
        total, volume = self._recent_totals
        # A breath that runs long, as in apnea, lowers the rate before it ends.
        span = max(total, total - self._recent[0][0] + self._since_measured)
        self.monitors.rr_total = 60.0 * len(self._recent) / span
        self.monitors.mv_exp = 60.0 * volume / span

    # --- Segments ------------------------------------------------------------

    def _segment(self, lung, pmus):
        b = self._breath
        rs = self._series_resistance()
        if b is None or not b.inspiring:
            offset, valve = self._valve_tangent(lung, pmus[0])
            return lung.pressure_segment((offset, 0.0), pmus, rs + valve)
        if b.kind == "SPONT":
            return lung.pressure_segment((0.0, 0.0), pmus, rs)
        if b.kind == "MANUAL":
            return lung.sine_segment(b.target, b.ti, b.t, pmus)
        if b.kind == "VC":
            if b.limited:
                return lung.pressure_segment((self._pressure_cap(b.peep), 0.0), pmus)
            if b.t < b.flow_end - 1e-12 and not b.delivered:
                return lung.flow_segment(b.target / b.flow_end, pmus)
            return lung.flow_segment(0.0, pmus)
        if b.t < RISE_TIME - 1e-12:
            slope = b.target / RISE_TIME
            return lung.pressure_segment((slope * b.t, slope), pmus)
        return lung.pressure_segment((b.target, 0.0), pmus)

    def _valve_coefficient(self) -> float:
        return PEEP_VALVE * min(self._peep(), PEEP_VALVE_FULL) / PEEP_VALVE_FULL

    def _valve_tangent(self, lung, pmus: float) -> tuple[float, float]:
        """(Pressure offset, resistance) of the PEEP valve's drop k Q^2, linearized at the latest knot.

        Q is the exhaled flow the valve carries at the knot, solved from the
        lung's recoil and muscle pressure: (R + Rs) Q + k Q^2 = recoil.
        """
        k = self._valve_coefficient()
        knot = bisect_right(VALVE_KNOTS, self._since_exhalation + 1e-12)
        if k <= 0.0 or knot >= len(VALVE_KNOTS):
            return 0.0, 0.0
        if self._valve is None or self._valve[0] != knot:
            r = lung.resistance + self._series_resistance()
            recoil = lung.elastic_pressure + lung.p2 - pmus
            q = 2.0 * recoil / (r + math.sqrt(r * r + 4.0 * k * recoil)) if recoil > 0.0 else 0.0
            self._valve = (knot, -k * q * q, 2.0 * k * q)
        return self._valve[1], self._valve[2]

    def _boundary(self) -> float:
        """Time to the next scheduled change of segment."""
        b = self._breath
        if b is not None and b.inspiring:
            ends = [b.ti]
            if b.kind == "VC" and not b.limited and not b.delivered:
                ends.append(b.flow_end)
            if b.kind in ("PC", "VG", "PS"):
                ends.append(RISE_TIME)
            return min((end - b.t for end in ends if end > b.t + 1e-12), default=0.0)
        period = self._mandatory_period()
        mode = self.settings.mode
        times = []
        if self._mandatory_type() is not None:
            times.append(period - self._since_mandatory)
            if self._source == "vent" and mode.startswith("SIMV"):
                times.append((1.0 - TRIGGER_WINDOW) * period - self._since_mandatory)
        elif self._source == "vent" and mode == "PSV" and self.settings.rr > 0.0:
            times.append(self.apnea_backup_s - self._since_breath)
        knot = bisect_right(VALVE_KNOTS, self._since_exhalation + 1e-12)
        if knot < len(VALVE_KNOTS) and self._valve_coefficient() > 0.0:
            times.append(VALVE_KNOTS[knot] - self._since_exhalation)
        return min((t for t in times if t > 1e-12), default=math.inf)

    def _advance(self, available: float, lung) -> float:
        effort = lung.effort
        if not self.inspiring and self._due():
            self._start(lung, self._mandatory_type(), True)
        b = self._breath
        boundary, effort_break = self._boundary(), effort.time_to_break()
        h = min(available, effort_break, boundary)
        segment = self._segment(lung, effort.pmus())
        watch = self._watch()
        observe = (self._source is not None and effort.pmus()[1] > 0.0 and watch != "trigger"
                   and (self._measured is None or (not self._measured.inspiring
                        and self._measured.expired >= MIN_EXPIRATION_L)))
        onset = effort.onset_volume()
        if watch is not None or observe or onset is not None:
            h = min(h, EVENT_STEP)
        t, event = (h, None) if watch is None else self._search(segment, h, watch)
        if observe:
            measured_t, measured_event = self._search(segment, t, "measure")
            if measured_event is not None and (event is None or measured_t < t - 1e-12):
                t, event = measured_t, measured_event
        # Lung volume ends the patient's contraction, or releases a breath waiting for deflation.
        for level, sign, name in ((effort.off_switch_volume(), 1.0, "inflated"), (onset, -1.0, "deflated")):
            reached = None if level is None else self._volume_reached(segment, t, level, sign)
            if reached is not None and (event is None or reached < t - 1e-12):
                t, event = reached, name
        self._measure(segment, t)
        segment.commit(t)
        self._paw_end = self._peep() + segment.paw(t)
        effort.advance(t, lung, self._series_resistance())
        self._since_mandatory += t
        self._since_breath += t
        self._since_measured += t
        self._since_exhalation += t
        if b is not None:
            b.t += t
        # The segment still describes the present unless something changed at its end.
        changed = event is not None or t >= min(boundary, effort_break) - 1e-12
        self._current = None if changed else (segment, t)
        # A zero-length step only ends in an event, and every event changes state.
        self._transition(lung, segment, t, event)
        return t

    @staticmethod
    def _volume_reached(segment, t: float, level: float, sign: float) -> float | None:
        """First time in [0, t] at which volume rises (sign 1) or falls (sign -1) to level."""
        f = lambda u: sign * (segment.volume(u) - level)  # noqa: E731
        f0, ft = f(0.0), f(t)
        if f0 >= 0.0:
            return 0.0
        return find_root(f, 0.0, t, f0, ft) if ft >= 0.0 else None

    def _due(self) -> bool:
        """Whether a mandatory breath starts now, entering PSV apnea backup first if due."""
        s = self.settings
        if (self._source == "vent" and s.mode == "PSV" and s.rr > 0.0 and not self._backup
                and self._since_breath >= self.apnea_backup_s - 1e-12):
            self._backup = True
        return self._mandatory_type() is not None and self._since_mandatory >= self._mandatory_period() - 1e-12

    # --- Events --------------------------------------------------------------

    def _watch(self):
        """Return the crossing to search for in the current segment, if any."""
        b = self._breath
        if b is not None and b.inspiring:
            if b.kind == "VC" and not b.delivered:
                if b.limited:
                    return "volume"
                if b.t < b.flow_end - 1e-12 and math.isfinite(self._pressure_cap(b.peep)):
                    return "limit"
            if b.kind == "PS":
                return "cycle"
            if b.kind == "SPONT":
                return "zero_flow"
            return None
        return "trigger" if self._triggering() and self._source is not None else None

    def _search(self, segment, h: float, watch: str):
        """Return (t, event) for the first crossing in [0, h], or (h, None)."""
        b = self._breath
        if watch == "limit":
            cap = self._pressure_cap(b.peep)
            f = lambda t: segment.paw(t) - cap  # noqa: E731
        elif watch == "volume":
            goal = b.v_start + b.target
            f = lambda t: segment.volume(t) - goal  # noqa: E731
        elif watch == "zero_flow":
            f = lambda t: -segment.flow(t)  # noqa: E731
        elif watch == "cycle":
            start = 0.0
            if segment.dflow(0.0) > 0.0 > segment.dflow(h):
                start = find_root(segment.dflow, 0.0, h)
            b.peak_flow = max(b.peak_flow, segment.flow(0.0), segment.flow(start), segment.flow(h))
            if b.t + h < RISE_TIME + 1e-12:
                return h, None
            level = CYCLE_FRACTION * b.peak_flow
            g = lambda t: level - segment.flow(t)  # noqa: E731
            if g(start) < 0.0 <= g(h):
                return find_root(g, start, h), "cycle"
            if g(0.0) >= 0.0 and b.t >= RISE_TIME:
                return 0.0, "cycle"
            return h, None
        else:  # trigger or an unsupported breath measured between mandatory breaths
            # Flow must first fall below the trigger level, so the end of a
            # supported breath cannot trigger the next one.
            level = MEASURE_FLOW if watch == "measure" else self.settings.trigger / 60.0
            f = lambda t: segment.flow(t) - level  # noqa: E731
            if watch == "trigger" and not self._armed:
                self._armed = f(h) < 0.0
                return h, None
        f0, fh = f(0.0), f(h)
        if f0 >= 0.0:
            return 0.0, watch
        if fh >= 0.0:
            return find_root(f, 0.0, h, f0, fh), watch
        return h, None

    def _transition(self, lung, segment, t, event) -> None:
        b = self._breath
        if event == "limit":
            b.limited = True
            return
        if event == "volume":
            b.limited, b.delivered = False, True
            return
        if event == "trigger":
            self._on_trigger(lung)
            return
        if event == "measure":
            self._begin_measurement(lung, "SPONT", False)
            return
        if event == "inflated":
            lung.effort.end_contraction()
            return
        if event == "deflated":
            lung.effort.resume(lung, self._series_resistance())
            return
        if b is None or not b.inspiring:
            return
        if event in ("cycle", "zero_flow") or b.t >= b.ti - 1e-12:
            self._end_inspiration(segment, t)

    def _on_trigger(self, lung) -> None:
        mode = self.settings.mode
        if self._source == "vent" and mode.startswith("SIMV"):
            window = (1.0 - TRIGGER_WINDOW) * self._mandatory_period()
            if self._since_mandatory >= window - 1e-12:
                self._start(lung, MANDATORY[mode], True)
                return
        self._backup = False
        supported = self._source == "vent" and mode != "CPAP" and self.settings.p_support > 0.0
        self._start(lung, "PS" if supported else "SPONT", False)

    def _end_inspiration(self, segment, t) -> None:
        b = self._breath
        b.inspiring = False
        self.inspiring = False
        self._since_exhalation, self._valve = 0.0, None
        end_paw = b.peep + segment.paw(t)
        volume = max(0.0, b.v_max - b.v_start)
        if b.kind == "VC":
            if not b.limited and (b.delivered or b.flow_end < b.ti):
                b.plat = end_paw
            self.pressure_limited = b.limited and not b.delivered
            if b.vg_test:
                self._vg_pressure = min(max(end_paw - b.peep, VG_MIN), self._pressure_cap(b.peep))
        elif b.kind in ("PC", "VG"):
            b.plat = end_paw
            if b.kind == "VG":
                self._adjust_vg(volume, b)
        if b.mandatory and self._measured.mandatory:
            self._measured.plat = b.plat

    def _adjust_vg(self, volume: float, b: Breath) -> None:
        """Move Pinsp toward the set VT by at most VG_STEP per breath, within Pmax."""
        goal = self.settings.tv / 1000.0
        pressure = self._vg_pressure
        if volume > 0.0:
            pressure += min(max(pressure * (goal / volume - 1.0), -VG_STEP), VG_STEP)
        ceiling = self._pressure_cap(b.peep)
        self._vg_pressure = min(max(pressure, VG_MIN), ceiling)
        self.pressure_limited = self._vg_pressure >= ceiling and volume < 0.9 * goal

    # --- Measurement ---------------------------------------------------------

    def _measure(self, segment, t: float) -> None:
        b = self._breath
        if t <= 0.0:
            return
        # Split gas movement at a reversal even when it occurs within one outer
        # step. Subsamples keep analyzer and airway-sensor timing independent of it.
        f0, f1 = segment.flow(0.0), segment.flow(t)
        bounds = [0.0]
        if f0 * f1 < 0.0:
            bounds.append(find_root(segment.flow, 0.0, t, f0, f1))
        bounds.append(t)
        # Volume is monotonic between bounds; the patient's breathing tracks its lowest value.
        segment.lung.effort.note_volume(min(segment.volume(u) for u in bounds))
        measured = self._measured
        for start, end in zip(bounds, bounds[1:]):
            change = segment.volume(end) - segment.volume(start)
            if measured is not None:
                if change >= 0.0:
                    measured.inspired += change
                else:
                    measured.expired -= change
                    measured.inspiring = False
            if self._collect_samples:
                n = max(1, math.ceil((end - start) / 0.01))
                previous = start
                for i in range(1, n + 1):
                    u = start + (end - start) * i / n
                    dv = segment.volume(u) - segment.volume(previous)
                    connected = self._source is not None
                    self.samples.append((u - previous, dv,
                                         self._peep() + segment.paw(u) if connected else 0.0,
                                         segment.flow(u) * 60.0 if connected else 0.0,
                                         segment.volume(u) + segment.lung.compliance * self._baseline
                                         if connected else 0.0))
                    previous = u
        paw = [segment.paw(0.0), segment.paw(t)]
        d0, d1 = segment.dpaw(0.0), segment.dpaw(t)
        if d0 > 0.0 > d1:
            paw.append(segment.paw(find_root(segment.dpaw, 0.0, t, d0, d1)))
        if measured is not None:
            measured.paw_peak = max(measured.paw_peak, self._peep() + max(paw))
            measured.area += segment.area(t) + self._peep() * t
            measured.t += t
        if b is None or not b.inspiring:
            return
        volume = [segment.volume(t)]
        f0, f1 = segment.flow(0.0), segment.flow(t)
        if f0 > 0.0 > f1:
            volume.append(segment.volume(find_root(segment.flow, 0.0, t, f0, f1)))
        b.v_max = max(b.v_max, *volume)

    def _sample(self, lung) -> None:
        """Airway pressure and flow at the end of the step."""
        segment, t = self._current or (self._segment(lung, lung.effort.pmus()), 0.0)
        self.paw = self._peep() + segment.paw(t)
        self.flow = segment.flow(t) * 60.0
        self.volume = lung.volume + lung.compliance * (self._baseline or 0.0)
