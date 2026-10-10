"""Anesthesia workstation ventilation and spirometry."""

import itertools
import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, fields, replace

from anasim.patient.domain import finite_number
from anasim.physiology.resp_mech import (
    ExpiratorySegment,
    RespiratoryMechanics,
    find_root,
)

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
TRIGGER_WINDOW_S = 5.0  # Adult SIMV window, capped at the scheduled expiratory time (Dräger)
CYCLE_FRACTION = 0.25  # Supported breaths end when flow falls to this share of its peak
SUPPORT_MAX_S = 4.0  # Adult pressure support safety limit (Dräger ASB); independent of mandatory Ti
APNEA_BACKUP_S = 20.0  # PSV starts backup breaths after this long without a breath
VG_STEP = 3.0  # Largest VG pressure change per breath, cmH2O
VG_MIN = 2.0  # cmH2O above PEEP
VG_MARGIN = 5.0  # GE volume-guarantee pressure targets stay this far below Pmax
EVENT_STEP = 0.02  # s; longest interval searched for one trigger or cycling crossing
RECENT_BREATHS = 4  # Breaths averaged for RR and MV
APNEA_S = 15.0  # RR and MV read zero after this long without a breath
MEASURE_FLOW = 0.05  # L/s; detects inspiration independently of ventilator trigger eligibility.
MIN_EXPIRATION_L = 0.05  # A breath exhaling less is still in progress; brief reversals neither end nor start one.
PLATEAU_FLOW = 0.5 / 60.0  # L/s; negligible end-inspiratory flow for a pressure breath
PLATEAU_PRESSURE = 0.5  # cmH2O; maximum residual resistive drop or muscle pressure


@dataclass
class VentSettings:
    """Settings: VT mL, rate breaths/min, pressures cmH2O (Pinsp and PS above PEEP), times s."""

    mode: str = "VCV"
    tv: float = 500.0
    rr: float = 12.0  # Mandatory rate; PSV backup rate
    peep: float = 5.0
    ie_ratio: float = 0.5  # I / E for VCV, PCV, and PCV-VG
    t_insp: float = 1.0  # SIMV mandatory and PSV backup breaths; 500 mL gives 33 L/min with a 10% pause
    p_insp: float = 15.0
    p_support: float = 5.0
    pause: float = 10.0  # % of Ti in volume-controlled breaths
    p_max: float = 40.0  # Pressure limit of volume-targeted breaths
    trigger: float = 3.0  # L/min
    fio2: float = 0.21

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode.upper() not in MODES:
            raise ValueError(f"Unsupported ventilator mode {self.mode!r}; choose one of: {', '.join(MODES)}")
        self.mode = self.mode.upper()
        for setting in fields(self):
            if setting.name != "mode":
                setattr(self, setting.name, finite_number(setting.name, getattr(self, setting.name)))
        if self.rr < 0.0:
            raise ValueError("ventilator rate must not be negative")
        if self.ie_ratio <= 0.0:
            raise ValueError("I:E ratio must be greater than zero")
        limits = {"t_insp": (0.2, 5.0), "trigger": (0.2, 20.0), "p_support": (0.0, 60.0), "pause": (0.0, 60.0)}
        for name, (low, high) in limits.items():
            if not low <= getattr(self, name) <= high:
                raise ValueError(f"{name} must be between {low:g} and {high:g}")
        if self.p_max <= self.peep:
            raise ValueError("Pmax must exceed PEEP")

    @property
    def ie(self) -> str:
        """I:E ratio as offered by the controls, such as "1:2"."""
        return f"1:{max(1, round(1.0 / self.ie_ratio))}"


@dataclass
class VentMonitors:
    """Spirometry at the airway: pressures cmH2O, volumes mL, MV L/min, compliance mL/cmH2O."""

    paw_peak: float = 0.0
    paw_plat: float = math.nan  # End-inspiratory pressure of the last mandatory breath with no flow
    paw_mean: float = 0.0
    peep: float = 0.0  # End-expiratory pressure
    auto_peep: float = 0.0  # Static recoil left at end-expiration
    tv_exp: float = 0.0
    tv_exp_mandatory: float = math.nan
    mv_exp: float = 0.0  # Recent breaths
    rr_total: float = 0.0
    compliance_dyn: float = math.nan  # VTe / (Ppeak - PEEP)


@dataclass
class Breath:
    kind: str  # VC, PC, VG, PS, SPONT (detected, unsupported), or MANUAL
    mandatory: bool
    ti: float  # Mandatory duration, PS safety limit, or infinity for unsupported breaths
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
    p_start: float = 0.0  # Pressure above PEEP at a PS-to-mandatory handoff


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
    Inspiratory targets take effect at the next breath; PEEP changes at the
    next expiration, or immediately during expiration. Settings persist while off.
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
        self._source: str | None = None
        self._bag = (12.0, 0.5)
        self._mode: str | None = None
        self._baseline: float | None = None  # PEEP the lung volume is referenced to; the first one applied
        self._breath: Breath | None = None
        self._measured: MeasuredBreath | None = None
        self._since_measured = 0.0
        self.samples = []  # Resolved (dt, dV L, Paw, flow L/min, absolute volume L).
        self.muscle_pressure_area = 0.0
        self.airway_pressure_area = 0.0
        self._collect_samples = False
        self._paw_end = 0.0
        self._current = None  # (segment, time) valid at the end of the step
        self._since_mandatory = 0.0  # SIMV retains time advanced by an early trigger
        self._since_breath = 0.0
        self._since_exhalation = 0.0
        self._valve = None  # (knot, offset, resistance) linearizing the PEEP valve
        self._expiration = None  # (physical knot, airway parameters) for obstructed expiration
        self._backup = False
        self._vg_pressure: float | None = None
        self._armed = False
        self._recent: deque[tuple[float, float]] = deque(maxlen=RECENT_BREATHS)  # (duration s, VTe L)
        self._recent_totals = (0.0, 0.0)  # Duration s and exhaled volume L

    @property
    def has_measured_breath(self) -> bool:
        return bool(self._recent)

    @property
    def _active_breath(self) -> Breath:
        """The controlled breath, for code that runs only while one exists."""
        if self._breath is None:
            raise RuntimeError("No ventilator breath in progress")
        return self._breath

    @property
    def _active_measurement(self) -> MeasuredBreath:
        """The breath spirometry is measuring, for code that runs only while one exists."""
        if self._measured is None:
            raise RuntimeError("No breath is being measured")
        return self._measured

    def update_settings(self, **values):
        """Validate and apply settings together; VT in mL, I:E as "1:2" or a float."""
        unknown = set(values) - {setting.name for setting in fields(self.settings)} - {"ie"}
        if unknown:
            raise ValueError(f"Unknown ventilator setting(s): {', '.join(sorted(unknown))}")
        if "ie" in values:
            ie = values.pop("ie")
            if isinstance(ie, str) and ':' in ie:
                i, e = (finite_number("I:E ratio", value) for value in ie.split(':'))
                if i <= 0.0 or e <= 0.0:
                    raise ValueError("I:E ratio components must be greater than zero")
                values["ie_ratio"] = i / e
            else:
                values["ie_ratio"] = ie
        self.settings = replace(self.settings, **values)

    # --- Configuration of the active source --------------------------------

    def _peep(self) -> float:
        """Applied PEEP, shared by the pressure source and lung coordinates."""
        return self._baseline or 0.0

    def _configured_peep(self) -> float:
        return self.settings.peep if self._source == "vent" else 0.0

    def _series_resistance(self) -> float:
        """Resistance between the Y-piece and the baseline pressure source."""
        return 0.0 if self._source is None else CIRCUIT_RESISTANCE

    def _mandatory_period(self) -> float:
        rr = self._bag[0] if self._source == "bag" else self.settings.rr
        return 60.0 / rr if rr > 0.0 else math.inf

    def _sync_window_start(self) -> float:
        """Opening on the compensated mandatory clock, after the scheduled Ti.

        The adult window spans the final 5 s, or all of expiration when shorter.
        Time borrowed by an early breath remains outside the next window.
        """
        period = self._mandatory_period()
        return max(min(self.settings.t_insp, 0.8 * period), period - TRIGGER_WINDOW_S)

    def _triggering(self) -> bool:
        return self._source == "spontaneous" or (self._source == "vent" and self.settings.mode in TRIGGERED)

    def _pressure_cap(self, peep: float) -> float:
        """Pmax above PEEP for volume-targeted breaths."""
        return self.settings.p_max - peep if self._source == "vent" else math.inf

    def _vg_ceiling(self, peep: float) -> float:
        return max(0.0, self._pressure_cap(peep) - VG_MARGIN)

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
        self.muscle_pressure_area = 0.0
        self.airway_pressure_area = 0.0
        self._collect_samples = collect_samples
        self._bag = bag
        if source != self._source:
            self._switch(source, lung)
        if self._source == "vent" and self.settings.mode != self._mode:
            if MANDATORY.get(self.settings.mode) == "VG" and (self._mode is None or MANDATORY.get(self._mode) != "VG"):
                self._vg_pressure = None
            self._mode = self.settings.mode
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
        self.monitors.peep = self._configured_peep()
        self.pressure_limited = False
        self._since_breath = 0.0
        self._since_exhalation, self._valve = 0.0, None
        self._expiration = None
        self._rebase(lung, self._configured_peep())
        mandatory = self._mandatory_type()
        if mandatory is not None:
            self._start(lung, mandatory, True)
        else:
            self.inspiring = False
            self._armed = False

    def _mandatory_type(self):
        if self._source == "bag":
            return "MANUAL" if self._bag[0] > 0.0 else None
        if self._source != "vent" or self.settings.rr <= 0.0:
            return None
        if self.settings.mode == "PSV":
            return "PC" if self._backup else None
        return MANDATORY.get(self.settings.mode)

    def _rebase(self, lung, peep: float) -> None:
        """Keep absolute lung volume continuous when the set PEEP changes."""
        if self._baseline is not None and peep != self._baseline:
            shift = -lung.compliance * (peep - self._baseline)
            lung.volume += shift
            lung.effort.shift_volume_reference(shift)
            self._current = self._valve = None
            self._expiration = None
        elif self._baseline is None and lung.aeration is not None:
            lung.volume -= lung.compliance * peep
        self._baseline = peep
        lung.peep = peep

    def _start(self, lung, kind: str, mandatory: bool) -> None:
        previous = self._breath
        continuing = previous is not None
        lung.note_end_expiration(self._baseline or 0.0)
        self._begin_measurement(lung, kind, mandatory)
        # A mandatory stroke can join an ongoing spontaneous inflation. Its
        # volume target includes the gas already delivered in that same breath.
        measured = self._active_measurement
        inspired = measured.inspired
        retained = max(0.0, inspired - measured.expired)
        s = self.settings
        peep = self._peep()
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
            goal = s.tv / 1000.0
            target = max(0.0, goal - retained)
            if retained > 0.0:
                ti *= target / goal if goal > 0.0 else 0.0
            flow_end = ti * (1.0 - min(max(pause, 0.0), 60.0) / 100.0)
        elif kind == "MANUAL":
            target = self._bag[1]
        elif kind == "PC":
            target = s.p_insp
        elif kind == "VG":
            assert self._vg_pressure is not None  # A VC test breath sets it first.
            # A Pmax or PEEP edit during expiration must constrain the very
            # next stroke, before the next volume-feedback update.
            target = self._vg_pressure = min(max(self._vg_pressure, VG_MIN), self._vg_ceiling(peep))
        elif kind == "PS":
            target, ti = s.p_support, SUPPORT_MAX_S
            flow_end = ti
        else:  # SPONT
            ti = flow_end = math.inf  # Unsupported inspiration ends only when flow reverses.
        v_start, v_max = lung.volume, lung.volume
        if kind == "VG":
            # Pressure adaptation sees the whole inflation, including its peak
            # before the mandatory stroke, rather than only the added volume.
            v_start -= retained
            v_max = max(v_max, v_start + inspired)
        p_start = 0.0
        if kind in ("PC", "VG") and previous is not None and previous.inspiring and previous.kind == "PS":
            # Continue the pressure already supplied to this inspiration.
            p_start = (previous.target if previous.t >= RISE_TIME else previous.p_start
                       + (previous.target - previous.p_start) * previous.t / RISE_TIME)
        self._breath = Breath(kind, mandatory, ti, flow_end, target, peep, v_start, vg_test=vg_test,
                              v_max=v_max, p_start=p_start)
        self.inspiring = True
        self._armed = False
        self._since_breath = 0.0
        if mandatory:
            if continuing and self._source == "vent" and s.mode.startswith("SIMV") and math.isfinite(period):
                # An early synchronized breath replaces the next scheduled one.
                # Carry its advance forward so repeated triggers do not raise RR.
                self._since_mandatory = min(0.0, self._since_mandatory - period)
            else:
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
        if b is not None and b.expired < MIN_EXPIRATION_L and not b.mandatory:
            # Detection can precede the support trigger. Keep gas already
            # inspired in this breath when PS or a mandatory stroke takes over.
            b.kind, b.mandatory = kind, mandatory
            return
        self._finish(lung)
        lung.note_end_expiration(self._baseline or 0.0)
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
            if self._measured is not None and self._measured.kind == "SPONT" and self._measured.inspiring:
                # An effort below the support trigger still draws gas through
                # the inspiratory limb, then exhales through the PEEP valve.
                return self._driven_segment(lung, (0.0, 0.0), pmus, rs)
            if lung.bronchospasm > 0.0:
                knot = math.floor((self._since_exhalation + 1e-12) / lung.EXPIRATORY_INTERVAL)
                if self._expiration is None or self._expiration[0] != knot:
                    self._valve = None
                    offset, valve = self._valve_tangent(lung, pmus[0])
                    parameters = lung.expiration_parameters((offset, 0.0), pmus, rs + valve)
                    self._expiration = (knot, offset, valve, parameters)
                _, offset, valve, parameters = self._expiration
                return ExpiratorySegment(lung, (offset, 0.0), pmus, rs + valve, parameters)
            offset, valve = self._valve_tangent(lung, pmus[0])
            return lung.pressure_segment((offset, 0.0), pmus, rs + valve)
        if b.kind == "SPONT":
            return self._driven_segment(lung, (0.0, 0.0), pmus, rs)
        if b.kind == "MANUAL":
            return lung.sine_segment(b.target, b.ti, b.t, pmus)
        if b.kind == "VC":
            if b.limited:
                return self._driven_segment(lung, (self._pressure_cap(b.peep), 0.0), pmus)
            if b.t < b.flow_end - 1e-12 and not b.delivered:
                return lung.flow_segment(b.target / b.flow_end, pmus)
            return lung.flow_segment(0.0, pmus)
        if b.t < RISE_TIME - 1e-12:
            slope = (b.target - b.p_start) / RISE_TIME
            return self._driven_segment(lung, (b.p_start + slope * b.t, slope), pmus)
        return self._driven_segment(lung, (b.target, 0.0), pmus)

    def _driven_segment(self, lung, pressure, pmus, series_resistance=0.0):
        if lung.bronchospasm <= 0.0:
            return lung.pressure_segment(pressure, pmus, series_resistance)
        # A relaxing patient can exhale during a mandatory pressure plateau.
        # Airway mechanics follow gas direction, including before triggering.
        knot = ("drive", math.floor((self._since_exhalation + 1e-12) / lung.EXPIRATORY_INTERVAL))
        if self._expiration is None or self._expiration[0] != knot:
            parameters = lung.expiration_parameters(pressure, pmus, series_resistance)
            self._expiration = (knot, 0.0, 0.0, parameters)
        return ExpiratorySegment(lung, pressure, pmus, series_resistance, self._expiration[3])

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
            ceiling = math.inf
            if lung.bronchospasm > 0.0:
                resistance, ceiling_resistance, _ = lung.expiration_parameters((0.0, 0.0), (pmus, 0.0), self._series_resistance())
                r = resistance + self._series_resistance()
                ceiling = max(0.0, (lung.peep + lung.elastic_pressure + lung.p2 + 5.0) / ceiling_resistance)
            q = 2.0 * recoil / (r + math.sqrt(r * r + 4.0 * k * recoil)) if recoil > 0.0 else 0.0
            q = min(q, ceiling)
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
            if not b.mandatory and self._source == "vent" and self.settings.mode.startswith("SIMV"):
                window = self._sync_window_start()
                ends.append(b.t + window - self._since_mandatory)
            return min((end - b.t for end in ends if end > b.t + 1e-12), default=0.0)
        period = self._mandatory_period()
        mode = self.settings.mode
        times = []
        if self._mandatory_type() is not None:
            times.append(period - self._since_mandatory)
            if self._source == "vent" and mode.startswith("SIMV"):
                times.append(self._sync_window_start() - self._since_mandatory)
        elif self._source == "vent" and mode == "PSV" and self.settings.rr > 0.0:
            times.append(self.apnea_backup_s - self._since_breath)
        knot = bisect_right(VALVE_KNOTS, self._since_exhalation + 1e-12)
        if knot < len(VALVE_KNOTS) and self._valve_coefficient() > 0.0:
            times.append(VALVE_KNOTS[knot] - self._since_exhalation)
        return min((t for t in times if t > 1e-12), default=math.inf)

    def _advance(self, available: float, lung) -> float:
        effort = lung.effort
        if not self.inspiring:
            self._rebase(lung, self._configured_peep())
        if not self.inspiring and self._due():
            self._start(lung, self._mandatory_type(), True)
        b = self._active_breath if self.inspiring else None
        boundary, effort_break = self._boundary(), effort.time_to_break()
        h = min(available, effort_break, boundary)
        if lung.aeration is not None:
            h = min(h, lung.aeration.INTERVAL - lung.aeration.elapsed)
        if lung.bronchospasm > 0.0 and (b is None or b.kind in ("PC", "VG", "PS", "SPONT")
                                      or (b.kind == "VC" and b.limited)):
            interval = lung.EXPIRATORY_INTERVAL
            knot = math.floor((self._since_exhalation + 1e-12) / interval)
            h = min(h, (knot + 1) * interval - self._since_exhalation)
        pmus = effort.pmus()
        segment = self._segment(lung, pmus)
        watch = self._watch()
        observe = (self._source is not None and pmus[1] > 0.0
                   and (self._measured is None or (not self._measured.inspiring
                        and self._measured.expired >= MIN_EXPIRATION_L)))
        passive_inspiration = (not self.inspiring and self._measured is not None
                               and self._measured.kind == "SPONT" and self._measured.inspiring)
        onset = effort.onset_volume()
        if watch is not None or observe or passive_inspiration or onset is not None:
            h = min(h, EVENT_STEP)
        t, event = (h, None) if watch is None else self._search(segment, h, watch)
        if observe:
            measured_t, measured_event = self._search(segment, t, "measure")
            if measured_event is not None and (event is None or measured_t < t - 1e-12):
                t, event = measured_t, measured_event
        if passive_inspiration:
            exhaled_t, exhaled_event = self._search(segment, t, "zero_flow")
            if exhaled_event is not None and (event is None or exhaled_t < t - 1e-12):
                t, event = exhaled_t, "passive_exhale"
        # Lung volume ends the patient's contraction, or releases a breath waiting for deflation.
        for level, sign, name in ((effort.off_switch_volume(), 1.0, "inflated"), (onset, -1.0, "deflated")):
            reached = None if level is None else self._volume_reached(segment, t, level, sign)
            if reached is not None and (event is None or reached < t - 1e-12):
                t, event = reached, name
        self._measure(segment, t)
        self.muscle_pressure_area += pmus[0] * t + 0.5 * pmus[1] * t * t
        if self._source is not None:
            self.airway_pressure_area += self._peep() * t + segment.area(t)
        segment.commit(t)
        self._paw_end = self._peep() + segment.paw(t)
        if effort.clock < effort.ti:
            assistance = segment.area(t) if b is not None and b.kind != "SPONT" else 0.0
            effort.note_assistance(t, assistance)
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
        if changed:
            self._expiration = None
            if not self.inspiring and t >= effort_break - 1e-12:
                # Renew the valve tangent as muscle pressure changes. Keeping
                # an earlier expiratory load until breath detection creates a
                # false pressure dip and flow jump at the inspiratory trigger.
                self._valve = None
        # A zero-length step only ends in an event, and every event changes state.
        self._transition(lung, segment, t, event)
        if lung.aeration is not None:
            # Recoil plus tissue pressure excludes airway and circuit drops,
            # including the extra drop when expiratory flow is limited.
            pressure_area = self._peep() * t + segment.distending_area(t)
            shift = lung.update_aeration(t, pressure_area, self._baseline or 0.0)
            if lung.aeration.elapsed == 0.0:
                if self._breath is not None:
                    self._breath.v_start += shift
                    self._breath.v_max += shift
                self._current = self._valve = None
                self._expiration = None
        if not self.inspiring:
            self._rebase(lung, self._configured_peep())
        return t

    @staticmethod
    def _volume_reached(segment, t: float, level: float, sign: float) -> float | None:
        """First time in [0, t] at which volume rises (sign 1) or falls (sign -1) to level."""
        # A constant-flow breath can land on the threshold to roundoff and
        # then hold there. Treat that as reaching it rather than missing it.
        f = lambda u: sign * (segment.volume(u) - level) + 1e-12  # noqa: E731
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
            if b.kind == "VC":
                if b.limited:
                    return None if b.delivered else "volume"
                # Relaxing inspiratory muscles can raise pressure during the
                # pause too, including after the set volume has been delivered.
                if math.isfinite(self._pressure_cap(b.peep)):
                    return "limit"
            if b.kind == "PS":
                return "cycle"
            if b.kind == "SPONT":
                return "zero_flow"
            return None
        return "trigger" if self._triggering() and self._source is not None else None

    def _search(self, segment, h: float, watch: str):
        """Return (t, event) for the first crossing in [0, h], or (h, None)."""
        if watch == "limit":
            cap = self._pressure_cap(self._active_breath.peep)
            f = lambda t: segment.paw(t) - cap  # noqa: E731
        elif watch == "volume":
            b = self._active_breath
            goal = b.v_start + b.target
            f = lambda t: segment.volume(t) - goal  # noqa: E731
        elif watch == "zero_flow":
            f = lambda t: -segment.flow(t)  # noqa: E731
        elif watch == "cycle":
            b = self._active_breath
            start = 0.0
            if segment.dflow(0.0) > 0.0 > segment.dflow(h):
                start = find_root(segment.dflow, 0.0, h)
            b.peak_flow = max(b.peak_flow, segment.flow(0.0), segment.flow(start), segment.flow(h))
            # During pressurization, exhalation aborts support. Once the
            # target is reached, cycle at the configured share of peak flow.
            level = 0.0 if b.t < RISE_TIME - 1e-12 else CYCLE_FRACTION * b.peak_flow
            g = lambda t: level - segment.flow(t)  # noqa: E731
            if g(start) < 0.0 <= g(h):
                return find_root(g, start, h), "cycle"
            if g(0.0) >= 0.0:
                return 0.0, "cycle"
            return h, None
        else:  # trigger or an unsupported breath measured between mandatory breaths
            # Flow must first fall below the trigger level, so the end of a
            # supported breath cannot trigger the next one.
            detect_only = self._source == "spontaneous" or (self._source == "vent" and self.settings.mode == "CPAP")
            level = MEASURE_FLOW if watch == "measure" or detect_only else self.settings.trigger / 60.0
            f = lambda t: segment.flow(t) - level  # noqa: E731
            if watch == "trigger" and not self._armed:
                # Arm at the segment start when flow is already below the
                # threshold, then catch a crossing within this same segment.
                if f(0.0) < 0.0:
                    self._armed = True
                else:
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
            self._active_breath.limited = True
            return
        if event == "volume":
            self._active_breath.limited, self._active_breath.delivered = False, True
            return
        if event == "trigger":
            self._on_trigger(lung)
            return
        if event == "measure":
            self._begin_measurement(lung, "SPONT", False)
            return
        if event == "passive_exhale":
            self._active_measurement.inspiring = False
            self._since_exhalation, self._valve = 0.0, None
            self._expiration = None
            return
        if event == "inflated":
            lung.effort.end_contraction()
            return
        if event == "deflated":
            lung.effort.resume(lung, self._series_resistance())
            return
        if b is None or not b.inspiring:
            return
        if not b.mandatory and self._source == "vent" and self.settings.mode.startswith("SIMV"):
            window = self._sync_window_start()
            if self._since_mandatory >= window - 1e-12 and segment.flow(t) >= self.settings.trigger / 60.0:
                # An effort already in progress when the window opens is
                # eligible too; waiting for a fresh flow crossing stacks breaths.
                self._start(lung, MANDATORY[self.settings.mode], True)
                return
        if event in ("cycle", "zero_flow") or b.t >= b.ti - 1e-12:
            self._end_inspiration(segment, t)

    def _on_trigger(self, lung) -> None:
        mode = self.settings.mode
        if self._source == "vent" and mode.startswith("SIMV"):
            window = self._sync_window_start()
            if self._since_mandatory >= window - 1e-12:
                self._start(lung, MANDATORY[mode], True)
                return
        self._backup = False
        supported = self._source == "vent" and mode != "CPAP" and self.settings.p_support > 0.0
        self._start(lung, "PS" if supported else "SPONT", False)

    def _end_inspiration(self, segment, t) -> None:
        b = self._active_breath
        b.inspiring = False
        self.inspiring = False
        self._since_exhalation, self._valve = 0.0, None
        self._expiration = None
        end_paw = b.peep + segment.paw(t)
        volume = max(0.0, b.v_max - b.v_start)
        if b.kind == "VC":
            if not b.limited and (b.delivered or b.flow_end < b.ti):
                b.plat = end_paw
            self.pressure_limited = b.limited and not b.delivered
            if b.vg_test:
                self._vg_pressure = min(max(end_paw - b.peep, VG_MIN), self._vg_ceiling(b.peep))
        elif b.kind in ("PC", "VG"):
            # A flat pressure trace still includes airway resistance while
            # gas flows. Only approximate a plateau after flow has settled.
            flow = abs(segment.flow(t))
            if flow <= PLATEAU_FLOW and flow * segment.lung.resistance <= PLATEAU_PRESSURE:
                b.plat = end_paw
            if b.kind == "VG":
                self._adjust_vg(volume, b)
        if segment.lung.effort.pmus()[0] > PLATEAU_PRESSURE:
            b.plat = math.nan
        if b.mandatory and self._active_measurement.mandatory:
            self._active_measurement.plat = b.plat

    def _adjust_vg(self, volume: float, b: Breath) -> None:
        """Move Pinsp toward the set VT by at most VG_STEP, retaining the Pmax margin."""
        goal = self.settings.tv / 1000.0
        pressure = b.target
        if volume > 0.0:
            pressure += min(max(pressure * (goal / volume - 1.0), -VG_STEP), VG_STEP)
        ceiling = self._vg_ceiling(b.peep)
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
        volumes = [segment.volume(u) for u in bounds]
        # Flow reversals bound monotonic volume intervals.
        segment.lung.effort.note_volume(min(volumes))
        measured = self._measured
        peep = self._peep()
        if self._collect_samples:
            connected = self._source is not None
            lung = segment.lung
            peep_volume = lung.compliance * self._baseline if connected else 0.0
            volume_offset = lung.volume_offset
        for index, (start, end) in enumerate(itertools.pairwise(bounds)):
            if end <= start:
                continue  # A reversal rounded to an existing boundary moves no gas.
            change = volumes[index + 1] - volumes[index]
            if measured is not None:
                if change >= 0.0:
                    measured.inspired += change
                else:
                    measured.expired -= change
                    measured.inspiring = False
            if self._collect_samples:
                n = max(1, math.ceil((end - start) / 0.01))
                previous, previous_volume = start, volumes[index]
                for i in range(1, n + 1):
                    u = start + (end - start) * i / n
                    volume = segment.volume(u)
                    dv = volume - previous_volume
                    self.samples.append((u - previous, dv,
                                         peep + segment.paw(u) if connected else 0.0,
                                         segment.flow(u) * 60.0 if connected else 0.0,
                                         volume + peep_volume + volume_offset if connected else 0.0))
                    previous, previous_volume = u, volume
        if measured is not None:
            paw = [segment.paw(0.0), segment.paw(t)]
            d0, d1 = segment.dpaw(0.0), segment.dpaw(t)
            if d0 > 0.0 > d1:
                paw.append(segment.paw(find_root(segment.dpaw, 0.0, t, d0, d1)))
            measured.paw_peak = max(measured.paw_peak, peep + max(paw))
            measured.area += segment.area(t) + peep * t
            measured.t += t
        if b is None or not b.inspiring:
            return
        volume = volumes[-1]
        if f0 > 0.0 > f1:
            volume = max(volume, volumes[1])
        b.v_max = max(b.v_max, volume)

    def _sample(self, lung) -> None:
        """Airway pressure and flow at the end of the step."""
        segment, t = self._current or (self._segment(lung, lung.effort.pmus()), 0.0)
        self.paw = self._peep() + segment.paw(t)
        self.flow = segment.flow(t) * 60.0
        self.volume = lung.volume + lung.compliance * (self._baseline or 0.0) + lung.volume_offset
