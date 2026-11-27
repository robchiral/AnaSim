"""Anesthesia workstation ventilation and spirometry."""

import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, fields, replace

from anasim.core.utils import clamp, clamp01
from anasim.patient.domain import finite_number
from anasim.physiology.resp_mech import (
    ExpiratorySegment,
    FlowSegment,
    RespiratoryMechanics,
    find_root,
)
from anasim.physiology.resuscitation import (
    COMPRESSION_AIRWAY_CLOSURE_R,
    COMPRESSION_BAG_IN_R,
    COMPRESSION_BAG_OUT_R,
    COMPRESSION_CLOSURE_OFFSET_CMH2O,
    COMPRESSION_CLOSURE_WIDTH_CMH2O,
    COMPRESSION_PERIOD_S,
    COMPRESSION_SUB_FRC_COMPLIANCE_FRAC,
    COMPRESSION_SUB_FRC_KNEE_L,
    COMPRESSION_VALVE_EXP_IN_R,
    COMPRESSION_VALVE_EXP_OUT_R,
    COMPRESSION_VALVE_INSP_REV_R,
    compression_pressure_drive,
)

MODES = ("VCV", "PCV", "PCV-VG", "SIMV-VC", "SIMV-PC", "SIMV-VG", "PSV", "CPAP")
# Mandatory breath type of each mode. SIMV, PSV, and CPAP respond to patient triggers.
MANDATORY = {"VCV": "VC", "PCV": "PC", "PCV-VG": "VG", "SIMV-VC": "VC", "SIMV-PC": "PC", "SIMV-VG": "VG"}
TRIGGERED = ("SIMV-VC", "SIMV-PC", "SIMV-VG", "PSV", "CPAP")
VOLUME_TARGETED = ("VCV", "PCV-VG", "SIMV-VC", "SIMV-VG")

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
# Times (s) after exhalation begins at which the valve's quadratic drop is
# relinearized at the current flow, so each segment stays exact. The valve adds
# no drop after the last knot.
VALVE_KNOTS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8,
               0.9, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0)
RISE_TIME = 0.28  # s, pressure ramp of pressure-controlled and supported breaths (Primus recordings)
BAG_TI_MAX_S = 1.6  # s; caps slow-rate manual squeezes at ~1.6 s (matches PatientEffort.TI_MAX)
TRIGGER_WINDOW_S = 5.0  # Adult SIMV window, capped at the scheduled expiratory time (Dräger)
CYCLE_FRACTION = 0.25  # Supported breaths end when flow falls to this share of its peak
SUPPORT_MAX_S = 4.0  # Adult pressure support safety limit (Dräger ASB); independent of mandatory Ti
APNEA_BACKUP_S = 20.0  # PSV starts backup breaths after this long without a breath
VG_STEP = 3.0  # Largest VG pressure change per breath, cmH2O
VG_MIN = 2.0  # cmH2O above PEEP
VG_MARGIN = 5.0  # GE volume-guarantee pressure targets stay this far below Pmax
EVENT_STEP = 0.02  # s; longest interval searched for one trigger or cycling crossing
RECENT_BREATHS = 4  # Minimum breaths retained when rolling window is satisfied
RECENT_WINDOW_S = 15.0  # Minimum rolling time window for RR, MV, and VA
APNEA_S = 15.0  # Minimum seconds without a breath before RR and MV read zero
MEASURE_FLOW = 0.05  # L/s; detects inspiration independently of ventilator trigger eligibility.
MIN_EXPIRATION_L = 0.05  # A breath exhaling less is still in progress; brief reversals neither end nor start one.
PLATEAU_FLOW = 0.5 / 60.0  # L/s; negligible end-inspiratory flow for a pressure breath
PLATEAU_PRESSURE = 0.5  # cmH2O; maximum residual resistive drop or muscle pressure
MIN_CIRCUIT_PAW = -10.0  # cmH2O; sub-atmospheric relief valve floor (ISO 80601-2-13)


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
        if self.tv < 0.0:
            raise ValueError("tidal volume must not be negative")
        if self.peep < 0.0:
            raise ValueError("PEEP must not be negative")
        if self.p_insp < 0.0:
            raise ValueError("inspiratory pressure must not be negative")
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


@dataclass(slots=True)
class VentMonitors:
    """Spirometry at the airway: pressures cmH2O, volumes mL, MV L/min, compliance mL/cmH2O."""

    paw_peak: float = 0.0
    paw_plat: float = math.nan  # End-inspiratory pressure of the last mandatory breath with no flow
    paw_mean: float = 0.0
    paw_mean_recent: float = 0.0  # Time-weighted over recent breaths, as for RR and MV
    peep: float = 0.0  # End-expiratory pressure
    auto_peep: float = 0.0  # Static recoil left at end-expiration
    tv_exp: float = 0.0
    tv_exp_mandatory: float = math.nan
    mv_exp: float = 0.0  # Recent breaths
    mv_supported: float = 0.0  # Recent breaths with positive-pressure support
    rr_total: float = 0.0
    compliance_dyn: float = math.nan  # VTe / (Ppeak - PEEP)


@dataclass(slots=True)
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
    limited: bool = False  # VC or MANUAL flow handed to the pressure limit
    clipped: bool = False  # Hit the pressure limit (Pmax or APL) during this breath
    delivered: bool = False  # VC or MANUAL volume reached under the limit
    vg_test: bool = False  # First PCV-VG breath: volume control, to measure compliance
    peak_flow: float = 0.0
    v_max: float = -math.inf  # Largest volume during inspiration
    plat: float = math.nan
    p_start: float = 0.0  # Pressure above PEEP at a PS-to-mandatory handoff
    depleted: bool = False  # Reservoir gas exhausted during this inspiration


@dataclass(slots=True)
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
    pmus_area: float = 0.0
    support_area: float = 0.0
    cpr_oscillating: bool = False
    net_volume: float = 0.0
    net_insp_end: float = 0.0


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
        self.pressure_limited = False  # Last volume-targeted or manual breath fell short at its limit
        self._source: str | None = None
        self._bag = (12.0, 0.5)
        self._bag_p_max = math.inf
        self._spontaneous_peep = 0.0
        self._bellows_fraction = 1.0
        self._peep_collapsed = False
        self._available_volume = math.inf
        self._fresh_flow_l_s = 0.0
        self._mask_leak_gap = 0.0
        self._bag_ref_volume = 0.0
        self._volume_delivery = 1.0
        self._mode: str | None = None
        self._baseline: float | None = None  # Applied PEEP (cmH2O) that lung volume is referenced to
        self._eff_baseline: float | None = None
        self._breath: Breath | None = None
        self._measured: MeasuredBreath | None = None
        self._since_measured = 0.0
        self.samples = []  # Resolved (dt, dV L, Paw, flow L/min, absolute volume L).
        self.muscle_pressure_area = 0.0
        self.capped_muscle_pressure_area = 0.0
        self.alveolar_pressure_area = 0.0
        self._effort_cap = 20.0
        self._collect_samples = False
        self._paw_end = 0.0
        self._current = None  # (segment, time) valid at the end of the step
        self._since_mandatory = 0.0  # Synchronized modes (SIMV, bag) retain time advanced by an early trigger
        self._since_breath = 0.0
        self._since_exhalation = 0.0
        self._valve = None  # (knot, offset, resistance) linearizing the PEEP valve
        self._expiration = None  # (knot, offset, valve, airway parameters) for obstructed expiration
        self._backup = False
        self._vg_pressure: float | None = None
        self._armed = False
        # (duration s, VTe L, pressure area cmH2O·s, supported VTe L, mandatory VTe L)
        self._recent: deque[tuple[float, float, float, float, float]] = deque()
        self._recent_totals = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._mandatory_paw_peak = 0.0
        self._mandatory_peak_age_s = math.inf
        self._attempted_breath = False
        self._manual_settled = False
        self._recent_provisional = False
        self._cpr_active = False
        self._cpr_clock = 0.0
        self._cpr_elapsed = 0.0

    def alveolar_ventilation(
        self,
        dead_space_l: float,
        *,
        upper_eff: float = 1.0,
        leak_fraction: float = 0.0,
        deliverable_mv_l_min: float | None = None,
        assisted: bool = False,
        sealed_airway: bool = True,
    ) -> float:
        """Per-breath alveolar ventilation (L/min) over recent measured breaths."""
        if not self._recent or self.monitors.rr_total <= 0.0:
            return 0.0
        total, _vol, _area, supported, _mand = self._recent_totals
        if self._recent_provisional and self._measured is not None:
            total = max(total, total - self._recent[-1][0] + self._measured.t)
        span = total if self._recent_provisional else max(total, total - self._recent[0][0] + self._since_measured)
        if span <= 0.0:
            return 0.0
        supp_mv = 60.0 * supported / span
        unsupp_mv = max(0.0, self.monitors.mv_exp - supp_mv)
        supp_scale = 1.0
        unsupp_scale = 1.0
        if deliverable_mv_l_min is not None:
            if assisted and supp_mv > deliverable_mv_l_min and supp_mv > 1e-9:
                supp_scale = max(0.0, deliverable_mv_l_min) / supp_mv
            rem_mv = max(0.0, deliverable_mv_l_min - supp_mv * supp_scale)
            if sealed_airway and unsupp_mv > rem_mv and unsupp_mv > 1e-9:
                unsupp_scale = rem_mv / unsupp_mv
        keep = 1.0 - clamp01(leak_fraction)
        va_total = 0.0
        for _dur, vte, _b_area, supp_vte, _mand_vte in self._recent:
            unsupp_vte = max(0.0, vte - supp_vte)
            vt_lung = supp_vte * supp_scale * keep + unsupp_vte * unsupp_scale
            va_total += max(0.0, vt_lung * upper_eff - dead_space_l)
        return 60.0 * va_total / span

    @property
    def mandatory_share(self) -> float:
        """Fraction of recent expired volume delivered by mandatory or manual breaths."""
        volume = self._recent_totals[1]
        return min(1.0, self._recent_totals[4] / volume) if volume > 0.0 else 0.0

    @property
    def mandatory_period_s(self) -> float | None:
        """Configured mandatory breath period (s), or None in spontaneous-only modes."""
        return self._mandatory_period() if self._mandatory_type() is not None else None

    @property
    def has_measured_breath(self) -> bool:
        return bool(self._recent) or self._attempted_breath

    @property
    def breath_inspiring(self) -> bool:
        return self._measured is not None and self._measured.inspiring

    @property
    def manual_breath_active(self) -> bool:
        return self._breath is not None and self._breath.inspiring and self._breath.kind == "MANUAL"

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
        target_mode = values.get("mode", self.settings.mode)
        new_peep = values.get("peep", self.settings.peep)
        if (
            "p_max" not in values
            and target_mode not in VOLUME_TARGETED
            and isinstance(new_peep, (int, float))
            and math.isfinite(new_peep)
            and self.settings.p_max <= new_peep
        ):
            values["p_max"] = max(40.0, float(new_peep) + 10.0)
        prev_mode = self.settings.mode
        self.settings = replace(self.settings, **values)
        if self.settings.mode != prev_mode:
            self._backup = False
            self._since_breath = 0.0
            period = self._mandatory_period()
            if math.isfinite(period):
                self._since_mandatory = min(self._since_mandatory, period)

    # --- Configuration of the active source --------------------------------

    def _peep(self) -> float:
        """Applied PEEP, shared by the pressure source and lung coordinates."""
        return self._baseline or 0.0

    def _configured_peep(self) -> float:
        if self._source == "vent":
            if self._peep_collapsed or (self._bellows_fraction <= 0.05 and self._fresh_flow_l_s <= 1e-6):
                return 0.0
            return self.settings.peep
        if self._source in ("spontaneous", "bag"):
            return self._spontaneous_peep
        return 0.0

    def _series_resistance(self) -> float:
        """Resistance between the Y-piece and the baseline pressure source."""
        return 0.0 if self._source is None else CIRCUIT_RESISTANCE

    def _effective_available_volume(self) -> float:
        if not math.isfinite(self._available_volume):
            return math.inf
        if self._fresh_flow_l_s <= 0.0:
            return self._available_volume
        b = self._breath
        ti = min(b.ti, 5.0) if b is not None and b.inspiring and math.isfinite(b.ti) else 1.0
        return self._available_volume + self._fresh_flow_l_s * ti

    def _bag_exhausted(self, lung: RespiratoryMechanics) -> bool:
        avail = self._effective_available_volume()
        if not math.isfinite(avail):
            return False
        if avail <= 1e-6:
            return True
        return max(0.0, lung.volume - self._bag_ref_volume) >= avail - 1e-9

    def _inspiratory_resistance(self, lung: RespiratoryMechanics | None = None) -> float:
        """Inspiratory limb resistance, closing when the reservoir is empty."""
        if self._source is None:
            return 0.0
        depleted = self._breath is not None and self._breath.inspiring and self._breath.depleted
        avail = self._effective_available_volume()
        starved = avail <= 1e-6 or depleted or (lung is not None and self._bag_exhausted(lung))
        r_circuit = CIRCUIT_RESISTANCE + 1e6 if starved else CIRCUIT_RESISTANCE
        if self._mask_leak_gap > 1e-6 and (self._breath is None or self._breath.kind == "SPONT"):
            r_room = 2.0 / self._mask_leak_gap
            return (r_circuit * r_room) / (r_circuit + r_room)
        return r_circuit

    def _mandatory_period(self) -> float:
        rr = self._bag[0] if self._source == "bag" else self.settings.rr
        return 60.0 / rr if rr > 0.0 else math.inf

    def _synchronized(self) -> bool:
        """Whether mandatory breaths synchronize with patient efforts in a trigger window."""
        return (
            (self._source == "vent" and self.settings.mode.startswith("SIMV"))
            or (self._source == "bag" and self._mandatory_type() == "MANUAL")
        )

    def _sync_window_start(self) -> float:
        """Start of the trigger window on the compensated mandatory clock.

        The adult window spans the final 5 s, or all of expiration when shorter.
        Time borrowed by an early breath remains outside the next window.
        """
        period = self._mandatory_period()
        t_insp = min(BAG_TI_MAX_S, period / 3.0) if self._source == "bag" else self.settings.t_insp
        return max(min(t_insp, 0.8 * period), period - TRIGGER_WINDOW_S)

    def _triggering(self) -> bool:
        return (
            self._source == "spontaneous"
            or (self._source == "vent" and self.settings.mode in TRIGGERED)
            or (self._source == "bag" and self._mandatory_type() == "MANUAL")
        )

    def _pressure_cap(self, peep: float) -> float:
        """Pressure limit (Pmax or APL) above PEEP for volume-targeted and manual breaths."""
        if self._source == "vent":
            return self.settings.p_max - peep
        if self._source == "bag" or (self._breath is not None and self._breath.kind == "MANUAL"):
            return max(0.0, self._bag_p_max - peep)
        return math.inf

    def _vg_ceiling(self, peep: float) -> float:
        return max(0.0, self._pressure_cap(peep) - VG_MARGIN)

    # --- Stepping ------------------------------------------------------------

    def _shift_breath_reference(self, shift: float) -> None:
        self._bag_ref_volume += shift
        if self._breath is not None:
            self._breath.v_start += shift
            self._breath.v_max += shift
        self._current = self._valve = None
        self._expiration = None

    def absorb_lung_gas(self, volume_l: float, lung: RespiratoryMechanics) -> None:
        """Adjust lung and breath coordinates when gas is absorbed behind an obstructed airway."""
        if volume_l <= 0.0:
            return
        applied = lung.absorb_gas(volume_l)
        if applied > 0.0:
            self._shift_breath_reference(-applied)

    def step(self, dt: float, lung: RespiratoryMechanics, source: str | None, bag: tuple = (12.0, 0.5),
             collect_samples: bool = False, bag_p_max: float = math.inf,
             spontaneous_peep: float = 0.0, bellows_fraction: float = 1.0,
             reservoir_volume: float | None = None, volume_delivery: float = 1.0,
             effort_cap: float = 20.0, fresh_flow_l_s: float = 0.0,
             mask_leak_gap: float = 0.0, trigger_manual_breath: bool = False,
             cpr_active: bool = False) -> None:
        """Advance dt seconds.

        source is "vent", "bag", "spontaneous" (breathing through the circuit),
        or None when the airway is disconnected and the machine measures nothing.
        """
        if bag_p_max != self._bag_p_max and bag_p_max != math.inf:
            bag_p_max = finite_number("bag_p_max", bag_p_max)
            if bag_p_max < 0.0:
                raise ValueError("bag_p_max must be nonnegative")
        if dt <= 0.0:
            return
        self.samples.clear()
        self.muscle_pressure_area = 0.0
        self.capped_muscle_pressure_area = 0.0
        self.alveolar_pressure_area = 0.0
        self._effort_cap = max(0.0, float(effort_cap))
        self._collect_samples = collect_samples
        new_cpr = bool(cpr_active)
        if new_cpr != self._cpr_active:
            self._cpr_active = new_cpr
            if new_cpr:
                self._cpr_clock = 0.0
                self._cpr_elapsed = 0.0
            self._current = self._valve = self._expiration = None
        self._bag = bag
        prev_bag_p_max = self._bag_p_max
        if source in ("bag", "spontaneous") and bag_p_max != prev_bag_p_max:
            self._current = None
            if source == "bag" and prev_bag_p_max <= 0.0 and bag_p_max > 0.0 and not self.inspiring:
                period = self._mandatory_period()
                if math.isfinite(period):
                    self._since_mandatory = max(self._since_mandatory, period)
            if bag_p_max <= 0.0:
                self.pressure_limited = False
        self._bag_p_max = bag_p_max
        self._spontaneous_peep = max(0.0, float(spontaneous_peep))
        self._volume_delivery = clamp01(float(volume_delivery))
        new_fresh = max(0.0, float(fresh_flow_l_s))
        new_gap = clamp01(float(mask_leak_gap))
        new_bellows = clamp01(float(bellows_fraction))
        avail_vol = (
            max(0.0, float(reservoir_volume))
            if reservoir_volume is not None
            else (math.inf if new_bellows >= 1.0 - 1e-9 else 0.5 * new_bellows)
        )
        avail_changed = (
            math.isfinite(avail_vol) != math.isfinite(self._available_volume)
            or (math.isfinite(avail_vol) and abs(avail_vol - self._available_volume) > 1e-9)
        )
        if (
            abs(new_bellows - self._bellows_fraction) > 1e-6
            or abs(new_fresh - self._fresh_flow_l_s) > 1e-9
            or abs(new_gap - self._mask_leak_gap) > 1e-6
            or avail_changed
        ):
            self._bellows_fraction = new_bellows
            self._available_volume = avail_vol
            self._fresh_flow_l_s = new_fresh
            self._mask_leak_gap = new_gap
            self._current = None
        if source == "vent":
            if new_fresh >= 0.006 or avail_vol >= 1.5:
                self._peep_collapsed = False
            elif new_bellows <= 0.05:
                self._peep_collapsed = True
        if source != self._source:
            self._switch(source, lung)
        elif source == "spontaneous" and not self.manual_breath_active and (
            abs(self._spontaneous_peep - (self._baseline or 0.0)) > 1e-6
            or abs(lung._transmitted_peep(self._spontaneous_peep) - (self._eff_baseline or 0.0)) > 1e-6
        ):
            self._rebase(lung, self._spontaneous_peep)
            self.monitors.peep = self._spontaneous_peep
        if self._source == "vent" and self.settings.mode != self._mode:
            if MANDATORY.get(self.settings.mode) == "VG" and (self._mode is None or MANDATORY.get(self._mode) != "VG"):
                self._vg_pressure = None
            if MANDATORY.get(self.settings.mode) is None:
                self.monitors.paw_plat = math.nan
                self.monitors.tv_exp_mandatory = math.nan
                self._mandatory_paw_peak = 0.0
                self._mandatory_peak_age_s = math.inf
            elif self.settings.mode in VOLUME_TARGETED and self._mode not in VOLUME_TARGETED:
                self.monitors.tv_exp_mandatory = math.nan
            if self.settings.mode == "CPAP":
                self.monitors.compliance_dyn = math.nan
            self.pressure_limited = False
            self._backup = False
            self._since_breath = 0.0
            period = self._mandatory_period()
            if (
                self._mandatory_type() is not None
                and math.isfinite(period)
                and (self.inspiring or lung.volume > 0.05)
            ):
                # Wait for passive exhalation to finish instead of stacking a full
                # mandatory stroke onto an unexhaled breath after PSV -> VCV switch.
                self._since_mandatory = max(0.0, min(self._since_mandatory, period - 1.5))
            else:
                self._since_mandatory = max(0.0, self._since_mandatory)
            self._mode = self.settings.mode
        elif self._source in ("vent", "bag") and self._since_mandatory < 0.0:
            period = self._mandatory_period()
            if math.isfinite(period):
                self._since_mandatory = max(-period, self._since_mandatory)
        self._set_lung_mandatory_flags(lung)
        lung.effort.start(lung, self._series_resistance())
        if not self.inspiring:
            self._sync_expiratory_baseline(lung)
        if (
            trigger_manual_breath
            and self._source == "spontaneous"
            and self._bag_p_max > 0.0
            and not self.manual_breath_active
        ):
            self._start(lung, "MANUAL", True)
        remaining = dt
        while remaining > 1e-12:
            remaining -= self._advance(remaining, lung)
        self._update_rates(lung)
        self._sample(lung)
        if (
            self._source is not None
            and not self.inspiring
            and not self.breath_inspiring
            and lung.effort.pmus() == (0.0, 0.0)
            and abs(self.flow) < 0.05
            and abs(lung.elastic_pressure) < 0.05
        ):
            lung.note_end_expiration(self._eff_baseline or 0.0, settled=True)

    def _switch(self, source, lung) -> None:
        prev_source = self._source
        self._source = source
        self._mode = self.settings.mode if source == "vent" else None
        self._vg_pressure = None
        self._backup = False
        self._peep_collapsed = False
        self._breath = None
        self._measured = None
        self._since_measured = 0.0
        self._recent.clear()
        self._recent_totals = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._mandatory_paw_peak = 0.0
        self._mandatory_peak_age_s = math.inf
        self._attempted_breath = False
        self._manual_settled = False
        self._recent_provisional = False
        self.monitors = VentMonitors()
        self.monitors.peep = self._configured_peep()
        self.pressure_limited = False
        self._since_breath = 0.0
        self._since_exhalation, self._valve = 0.0, None
        self._expiration = None
        lung._exp_trough_vol = None
        self._rebase(lung, self._configured_peep())
        self._bag_ref_volume = lung.volume
        mandatory = self._mandatory_type()
        self._set_lung_mandatory_flags(lung)
        if mandatory is not None:
            if prev_source in ("bag", "vent") and lung.volume > 0.15:
                # Let an in-progress mandatory/bag breath finish exhaling before
                # delivering the next mandatory stroke (prevents breath stacking).
                period = self._mandatory_period()
                self._since_mandatory = max(0.0, period - 1.5) if math.isfinite(period) else 0.0
                self.inspiring = False
                self._armed = False
            else:
                self._start(lung, mandatory, True)
        else:
            self.inspiring = False
            self._armed = False

    def _mandatory_type(self):
        if self._source == "bag":
            return "MANUAL" if self._bag[0] > 0.0 and self._bag_p_max > 0.0 else None
        if self._source != "vent" or self.settings.rr <= 0.0:
            return None
        if self.settings.mode == "PSV":
            return "PC" if self._backup else None
        return MANDATORY.get(self.settings.mode)

    def _set_lung_mandatory_flags(self, lung) -> None:
        has_mandatory = self._mandatory_type() is not None
        lung._mandatory_ventilating = has_mandatory
        if not has_mandatory:
            lung._time_triggered_fraction = 0.0
        elif self._recent_totals[0] > 0.0:
            lung._time_triggered_fraction = clamp01(self.mandatory_share / 0.35)
        else:
            period = self._mandatory_period()
            mand_rr = 60.0 / period if math.isfinite(period) and period > 0.0 else 0.0
            share = mand_rr / (mand_rr + max(1e-6, lung.effort.rr)) if mand_rr > 0.0 else 0.0
            lung._time_triggered_fraction = clamp01(share / 0.35)

    def _rebase(self, lung, peep: float) -> None:
        """Keep absolute lung volume continuous when the set PEEP changes."""
        eff_peep = lung._transmitted_peep(peep)
        prev_lung = getattr(self, "_lung", None)
        if prev_lung is not None and prev_lung is not lung:
            prev_eff: float | None = lung.peep
            self._bag_ref_volume = lung.volume
        else:
            prev_eff = self._eff_baseline
        if prev_eff is not None and eff_peep != prev_eff:
            shift = -lung.compliance * (eff_peep - prev_eff)
            lung.volume += shift
            self._bag_ref_volume += shift
            if lung._exp_trough_vol is not None:
                lung._exp_trough_vol += shift
            lung.effort.shift_peep_reference(shift)
            self._current = self._valve = None
            self._expiration = None
        elif prev_eff is None:
            if lung.aeration is not None:
                lung.volume -= lung.compliance * eff_peep
            self._bag_ref_volume = lung.volume
        self._lung = lung
        prev_baseline = self._baseline
        self._baseline = peep
        self._eff_baseline = eff_peep
        lung.peep = eff_peep
        if self._source is not None and (prev_baseline is None or abs(peep - prev_baseline) > 1e-9):
            self.monitors.peep = peep

    def _sync_expiratory_baseline(self, lung: RespiratoryMechanics) -> None:
        self._rebase(lung, self._configured_peep())
        if not self._cpr_active and not self.breath_inspiring and lung.volume <= 1e-3 and lung.effort.pmus() == (0.0, 0.0):
            lung.note_end_expiration(self._eff_baseline or 0.0, settled=True)

    def _start(self, lung, kind: str, mandatory: bool) -> None:
        previous = self._breath
        continuing = previous is not None
        self._begin_measurement(lung, kind, mandatory)
        # A mandatory stroke can join an ongoing spontaneous inflation. Its
        # volume target includes the gas already delivered in that same breath.
        measured = self._active_measurement
        inspired = measured.inspired
        retained = max(0.0, inspired - measured.expired)
        s = self.settings
        peep = self._peep()
        period = self._mandatory_period()
        if kind == "MANUAL":
            ti = (
                min(BAG_TI_MAX_S, period / 3.0)
                if self._source == "bag" and math.isfinite(period)
                else BAG_TI_MAX_S
            )
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
            goal = self._bag[1]
            target = max(0.0, goal - retained)
            if retained > 0.0:
                ti *= target / goal if goal > 0.0 else 0.0
            flow_end = ti
        elif kind == "PC":
            target = s.p_insp
        elif kind == "VG":
            if s.tv <= 0.0:
                target, ti, flow_end = 0.0, 0.0, 0.0
            else:
                assert self._vg_pressure is not None  # A VC test breath sets it first.
                # A Pmax or PEEP edit during expiration must constrain the very
                # next stroke, before the next volume-feedback update.
                self._vg_pressure = min(max(self._vg_pressure, VG_MIN), self._vg_ceiling(peep))
                target = self._vg_pressure
        elif kind == "PS":
            target, ti = s.p_support, SUPPORT_MAX_S
            flow_end = ti
        else:  # SPONT
            ti = flow_end = math.inf  # Unsupported inspiration ends only when flow reverses.
        v_start, v_max = lung.volume, lung.volume
        if kind == "VG":
            # Pressure adaptation uses the full inflation, including the peak before the mandatory stroke.
            v_start -= retained
            v_max = max(v_max, v_start + inspired)
        p_start = 0.0
        if kind in ("PC", "VG") and previous is not None and previous.inspiring and previous.kind == "PS":
            # Continue the pressure already supplied to this inspiration.
            p_start = (previous.target if previous.t >= RISE_TIME else previous.p_start
                       + (previous.target - previous.p_start) * previous.t / RISE_TIME)
        self._breath = Breath(
            kind, mandatory, ti, flow_end, target, peep, v_start, vg_test=vg_test,
            v_max=v_max, p_start=p_start, depleted=self._bag_exhausted(lung),
        )
        self.inspiring = True
        self._armed = False
        self._since_breath = 0.0
        if mandatory:
            if continuing and self._synchronized() and math.isfinite(period):
                # An early synchronized breath replaces the next scheduled one,
                # preserving the set RR across repeated triggers.
                self._since_mandatory = min(0.0, self._since_mandatory - period)
            else:
                self._since_mandatory = 0.0

    def _finish(
        self,
        lung,
        *,
        push_recent: bool = True,
        replace_last: bool = False,
        count_breath: bool = True,
    ) -> bool:
        """Record the breath that ends as the next one starts; return True if exhaled volume reached MIN_EXPIRATION_L."""
        b = self._measured
        if self._source is None or b is None or b.t <= 0.0:
            return False
        self._attempted_breath = True
        m = self.monitors
        # During CPR, oscillatory back-and-forth flow inflates cumulative
        # expired sums; use net excursion from end of inspiration (Orlob 2021).
        vte = max(0.0, b.net_insp_end - b.net_volume) if b.cpr_oscillating else b.expired
        period = self.mandatory_period_s
        if b.mandatory:
            self._mandatory_paw_peak = b.paw_peak
            self._mandatory_peak_age_s = 0.0
            m.paw_peak = b.paw_peak
        elif (
            period is not None
            and math.isfinite(period)
            and self._mandatory_peak_age_s <= max(APNEA_S, 1.25 * period)
        ):
            # Interposed spontaneous breaths must not overwrite a recent mandatory
            # peak below the latched plateau pressure or drop active Ppeak alarms.
            m.paw_peak = max(b.paw_peak, self._mandatory_paw_peak)
        else:
            m.paw_peak = b.paw_peak
        m.paw_mean = b.area / b.t
        peep_end = self._peep() if b.cpr_oscillating else self._paw_end
        if self._breath is None or not self._breath.inspiring:
            m.peep = peep_end
            m.auto_peep = max(0.0, lung.elastic_pressure)
        full_failed_mandatory = (
            b.mandatory
            and (
                b.inspired < MIN_EXPIRATION_L
                or (period is not None and math.isfinite(period) and b.t >= 0.8 * period)
            )
        )
        if vte >= MIN_EXPIRATION_L or full_failed_mandatory:
            m.tv_exp = vte * 1000.0
            if b.kind != "SPONT":
                driving = b.paw_peak - peep_end
                m.compliance_dyn = m.tv_exp / driving if driving > 0.5 and vte >= MIN_EXPIRATION_L else math.nan
            if b.mandatory:
                m.tv_exp_mandatory = m.tv_exp
                m.paw_plat = b.plat if vte >= MIN_EXPIRATION_L else math.nan
        if vte < MIN_EXPIRATION_L:
            return False
        if b.kind == "SPONT":
            supported_vte = 0.0
        elif not b.mandatory and b.pmus_area > 0.0:
            total_drive = b.support_area + b.pmus_area
            supported_vte = vte * (clamp01(b.support_area / total_drive) if total_drive > 1e-9 else 1.0)
        else:
            supported_vte = vte
        mandatory_vte = vte if b.mandatory else 0.0
        if push_recent:
            entry = (b.t, vte, b.area, supported_vte, mandatory_vte)
            if replace_last and self._recent:
                self._recent[-1] = entry
            else:
                self._recent.append(entry)
            window_s = max(RECENT_WINDOW_S, period) if period is not None and math.isfinite(period) else RECENT_WINDOW_S
            total_dur = sum(item[0] for item in self._recent)
            while len(self._recent) > RECENT_BREATHS and total_dur - self._recent[0][0] >= window_s - 1e-6:
                total_dur -= self._recent.popleft()[0]
            durations, volumes, areas, supported, mandatory = zip(*self._recent, strict=True)
            self._recent_totals = (sum(durations), sum(volumes), sum(areas), sum(supported), sum(mandatory))
        if count_breath:
            self.breath_count += 1
        return True

    def _begin_measurement(self, lung, kind: str, mandatory: bool) -> None:
        # A scheduled inflation before a spontaneous breath has exhaled belongs
        # to the same measured breath; controller timing stays separate.
        b = self._measured
        if b is not None and not b.mandatory and (b.inspiring or (b.expired < MIN_EXPIRATION_L and kind != "SPONT")):
            # Detection can precede the support trigger. Keep gas already
            # inspired in this breath when PS or a mandatory stroke takes over.
            b.kind, b.mandatory = kind, mandatory
            return
        had_prior = b is not None
        if self._manual_settled:
            counted = self._finish(
                lung, push_recent=True, replace_last=self._recent_provisional, count_breath=False,
            )
            self._manual_settled = False
            self._recent_provisional = False
        else:
            counted = self._finish(lung)
        if (
            not self.inspiring
            and (mandatory or not lung._mandatory_ventilating)
            and (b is None or b.expired >= MIN_EXPIRATION_L or lung.volume <= 1e-3)
        ):
            lung.note_end_expiration(self._eff_baseline or 0.0, cpr_active=self._cpr_active)
        self._bag_ref_volume = lung.volume
        self._measured = MeasuredBreath(kind, mandatory)
        if not had_prior or counted:
            self._since_measured = 0.0

    def _update_rates(self, lung=None) -> None:
        if lung is not None and not self.inspiring and not self.breath_inspiring:
            self.monitors.auto_peep = min(self.monitors.auto_peep, max(0.0, lung.elastic_pressure))
        total, volume, area, supported, _ = self._recent_totals
        current = self._measured
        if self._recent_provisional and self._recent and current is not None:
            total = max(total, total - self._recent[-1][0] + current.t)
            area = area - self._recent[-1][2] + current.area
        window_t, window_area = total, area
        if (
            current is not None
            and not self._recent_provisional
            and (not self._recent or current.t > self._recent[0][0])
        ):
            # A breath that outlasts the oldest one replaces it, so pauses lower the mean.
            oldest_t, _, oldest_area, _, _ = self._recent[0] if self._recent else (0.0, 0.0, 0.0, 0.0, 0.0)
            window_t += current.t - oldest_t
            window_area += current.area - oldest_area
        if window_t > 0.0:
            self.monitors.paw_mean_recent = window_area / window_t
        period = self.mandatory_period_s
        if period is not None and math.isfinite(period):
            apnea_timeout = max(APNEA_S, 1.25 * period)
        elif self._recent:
            recent_period = min(25.0, self._recent[-1][0])
            apnea_timeout = max(APNEA_S, 1.25 * recent_period)
        else:
            apnea_timeout = APNEA_S
        if not self._recent or self._since_measured >= apnea_timeout:
            self.monitors.rr_total = self.monitors.tv_exp = self.monitors.mv_exp = self.monitors.mv_supported = 0.0
            if self._since_measured >= apnea_timeout:
                self._recent.clear()
                self._recent_totals = (0.0, 0.0, 0.0, 0.0, 0.0)
                self._manual_settled = False
                self._recent_provisional = False
                if self._measured is not None and not self.inspiring and not self._measured.inspiring:
                    self._measured = None
                baseline_paw = 0.0 if (self._bellows_fraction <= 0.05 or self._peep_collapsed) else self._configured_peep()
                self.monitors.paw_mean_recent = baseline_paw
                self.monitors.paw_mean = baseline_paw
                if self._since_breath >= apnea_timeout:
                    self.monitors.paw_peak = baseline_paw
                    self._mandatory_paw_peak = 0.0
                    self._mandatory_peak_age_s = math.inf
                self.monitors.paw_plat = math.nan
                self.monitors.compliance_dyn = math.nan
                self.monitors.tv_exp_mandatory = 0.0 if self._mandatory_type() is not None else math.nan
            return
        # A breath that runs long, as in apnea, lowers the rate before it ends.
        span = total if self._recent_provisional else max(total, total - self._recent[0][0] + self._since_measured)
        self.monitors.rr_total = 60.0 * len(self._recent) / span
        self.monitors.mv_exp = 60.0 * volume / span
        self.monitors.mv_supported = 60.0 * supported / span

    # --- Segments ------------------------------------------------------------

    def _cpr_terms(
        self, lung: RespiratoryMechanics, *, inspiring_positive: bool,
    ) -> tuple[tuple[float, float], float, float, float, float]:
        """Return (cpr_pressure, elastance, recoil_offset, r_close, static_recoil) during CPR."""
        cpr_p, cpr_slope, _ = compression_pressure_drive(self._cpr_clock)
        e_base = 1.0 / lung.compliance
        if lung.volume < 0.0:
            x = -lung.volume
            e_sub = e_base / COMPRESSION_SUB_FRC_COMPLIANCE_FRAC
            knee = COMPRESSION_SUB_FRC_KNEE_L
            p_el = -e_base * x - (e_sub - e_base) * (x * x) / (x + knee)
            e_tan = e_base + (e_sub - e_base) * (x * x + 2.0 * x * knee) / ((x + knee) ** 2)
            p_el_off = p_el - e_tan * lung.volume
        else:
            p_el = lung.volume * e_base
            e_tan = e_base
            p_el_off = 0.0
        if inspiring_positive:
            r_close = 0.0
        else:
            aop = lung.aeration.closing_pressure if lung.aeration is not None else 5.0
            p_stent = self._peep() + max(0.0, lung.volume * e_base)
            z = (aop - COMPRESSION_CLOSURE_OFFSET_CMH2O - p_stent) / COMPRESSION_CLOSURE_WIDTH_CMH2O
            closure = 1.0 / (1.0 + math.exp(-clamp(z, -20.0, 20.0)))
            r_close = COMPRESSION_AIRWAY_CLOSURE_R * closure
        return (cpr_p, cpr_slope), e_tan, p_el_off, r_close, p_el

    def _cpr_segment(self, lung: RespiratoryMechanics, pmus):
        """Build an exact respiratory segment coupled to active 110/min chest compressions."""
        b = self._breath
        rs = self._series_resistance()
        r_insp = self._inspiratory_resistance(lung)
        starvation = r_insp - rs
        if b is None or not b.inspiring or b.kind == "SPONT":
            cpr_p, e_tan, p_el_off, r_close, p_el = self._cpr_terms(lung, inspiring_positive=False)
            recoil = p_el + lung.p2
            r_aw = lung.resistance + r_close
            if self._source is None:
                rs_eff = 0.0
            else:
                free_flow = (pmus[0] - cpr_p[0] - recoil) / (r_aw + rs)
                if self._source == "vent":
                    target_rs = COMPRESSION_VALVE_EXP_OUT_R if free_flow < 0.0 else COMPRESSION_VALVE_EXP_IN_R
                else:
                    target_rs = COMPRESSION_BAG_OUT_R if free_flow < 0.0 else COMPRESSION_BAG_IN_R
                rs_eff = max(rs, target_rs)
                if (
                    (b is not None and b.inspiring and b.kind == "SPONT")
                    or (self._measured is not None and self._measured.kind == "SPONT" and self._measured.inspiring)
                    or (
                        self._bag_exhausted(lung)
                        and (pmus[0] > recoil + 1e-9 or (pmus[0] >= recoil - 1e-9 and pmus[1] > 0.0))
                    )
                ):
                    rs_eff += starvation
            if lung.expiratory_obstruction > 0.0:
                parameters = lung.expiration_parameters(
                    (0.0, 0.0), pmus, rs_eff,
                    extra_resistance=r_close, cpr_pressure=cpr_p[0], static_recoil=p_el,
                )
                return ExpiratorySegment(
                    lung, (0.0, 0.0), pmus, rs_eff, parameters,
                    elastance=e_tan, recoil_offset=p_el_off,
                    cpr_pressure=cpr_p,
                )
            return lung.pressure_segment(
                (0.0, 0.0), pmus, rs_eff,
                elastance=e_tan, airway_resistance=r_aw,
                recoil_offset=p_el_off, cpr_pressure=cpr_p,
            )

        cpr_p, e_tan, p_el_off, _, p_el = self._cpr_terms(lung, inspiring_positive=True)
        recoil = p_el + lung.p2

        def _cpr_driven(pressure, base_rs: float = 0.0, manual: bool = False):
            free_flow = (pressure[0] + pmus[0] - cpr_p[0] - recoil) / lung.resistance
            cpr_valve_r = (
                (COMPRESSION_BAG_OUT_R if manual else COMPRESSION_VALVE_INSP_REV_R)
                if free_flow < 0.0
                else 0.0
            )
            tot_rs = base_rs + cpr_valve_r
            if lung.expiratory_obstruction > 0.0:
                parameters = lung.expiration_parameters(
                    pressure, pmus, tot_rs, cpr_pressure=cpr_p[0], static_recoil=p_el,
                )
                return ExpiratorySegment(
                    lung, pressure, pmus, tot_rs, parameters,
                    elastance=e_tan, recoil_offset=p_el_off,
                    cpr_pressure=cpr_p,
                )
            return lung.pressure_segment(
                pressure, pmus, tot_rs,
                elastance=e_tan, recoil_offset=p_el_off,
                cpr_pressure=cpr_p,
            )

        if b.kind == "MANUAL":
            cap = self._pressure_cap(b.peep)
            if b.depleted or b.delivered or b.target <= 1e-9 or b.ti <= 1e-12:
                if recoil + cpr_p[0] - pmus[0] > cap:
                    return _cpr_driven((cap, 0.0), manual=True)
                return lung.flow_segment(0.0, pmus, elastance=e_tan, recoil_offset=p_el_off, cpr_pressure=cpr_p)
            if b.limited:
                sine_q = 0.5 * b.target * (math.pi / b.ti) * math.sin(math.pi * b.t / b.ti)
                if lung.resistance * sine_q + recoil + cpr_p[0] - pmus[0] < cap - 1e-3:
                    b.limited = False
                else:
                    return _cpr_driven((cap, 0.0), manual=True)
            return lung.sine_segment(
                b.target, b.ti, b.t, pmus, elastance=e_tan, recoil_offset=p_el_off, cpr_pressure=cpr_p,
            )
        if b.kind == "VC":
            cap = self._pressure_cap(b.peep)
            if b.depleted or b.delivered or b.t >= b.flow_end - 1e-12:
                if recoil + cpr_p[0] - pmus[0] > cap:
                    return _cpr_driven((cap, 0.0))
                return lung.flow_segment(0.0, pmus, elastance=e_tan, recoil_offset=p_el_off, cpr_pressure=cpr_p)
            q_set = b.target / b.flow_end if b.flow_end > 1e-12 else 0.0
            if b.limited:
                # Resume constant VCV flow once decompression drops Paw below Pmax
                # rather than over-inflating above the set flow rate.
                if lung.resistance * q_set + recoil + cpr_p[0] - pmus[0] < cap - 1e-3:
                    b.limited = False
                else:
                    return _cpr_driven((cap, 0.0))
            return lung.flow_segment(
                q_set, pmus, elastance=e_tan, recoil_offset=p_el_off, cpr_pressure=cpr_p,
            )
        if b.depleted and b.kind in ("PC", "VG"):
            if recoil + cpr_p[0] - pmus[0] > b.target:
                return _cpr_driven((b.target, 0.0))
            return lung.flow_segment(0.0, pmus, elastance=e_tan, recoil_offset=p_el_off, cpr_pressure=cpr_p)
        if b.t < RISE_TIME - 1e-12:
            slope = (b.target - b.p_start) / RISE_TIME
            return _cpr_driven((b.p_start + slope * b.t, slope), starvation)
        return _cpr_driven((b.target, 0.0), starvation)

    def _segment(self, lung, pmus):
        if self._cpr_active:
            return self._cpr_segment(lung, pmus)
        b = self._breath
        rs = self._series_resistance()
        r_insp = self._inspiratory_resistance(lung)
        starvation = r_insp - rs
        recoil = lung.elastic_pressure + lung.p2
        if b is None or not b.inspiring:
            if self._measured is not None and self._measured.kind == "SPONT" and self._measured.inspiring:
                # An effort below the support trigger still draws gas through
                # the inspiratory limb, then exhales through the PEEP valve.
                return self._driven_segment(lung, (0.0, 0.0), pmus, r_insp)
            if self._bag_exhausted(lung) and (
                pmus[0] > recoil + 1e-9 or (pmus[0] >= recoil - 1e-9 and pmus[1] > 0.0)
            ):
                return self._driven_segment(lung, (0.0, 0.0), pmus, r_insp)
            if lung.expiratory_obstruction > 0.0:
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
            return self._driven_segment(lung, (0.0, 0.0), pmus, r_insp)
        if b.kind == "MANUAL":
            cap = self._pressure_cap(b.peep)
            if b.depleted:
                if b.limited and recoil - pmus[0] > cap:
                    return self._driven_segment(lung, (cap, 0.0), pmus)
                return lung.flow_segment(0.0, pmus)
            if b.limited:
                return self._driven_segment(lung, (cap, 0.0), pmus)
            if not b.delivered and b.target > 1e-9 and b.ti > 1e-12:
                return lung.sine_segment(b.target, b.ti, b.t, pmus)
            return lung.flow_segment(0.0, pmus)
        if b.kind == "VC":
            if b.depleted:
                if b.limited and recoil - pmus[0] > self._pressure_cap(b.peep):
                    return self._driven_segment(lung, (self._pressure_cap(b.peep), 0.0), pmus)
                return lung.flow_segment(0.0, pmus)
            if b.limited:
                return self._driven_segment(lung, (self._pressure_cap(b.peep), 0.0), pmus)
            if b.t < b.flow_end - 1e-12 and not b.delivered:
                return lung.flow_segment(b.target / b.flow_end, pmus)
            return lung.flow_segment(0.0, pmus)
        if b.depleted and b.kind in ("PC", "VG"):
            if recoil - pmus[0] > b.target:
                return self._driven_segment(lung, (b.target, 0.0), pmus)
            return lung.flow_segment(0.0, pmus)
        if b.t < RISE_TIME - 1e-12:
            slope = (b.target - b.p_start) / RISE_TIME
            return self._driven_segment(lung, (b.p_start + slope * b.t, slope), pmus, starvation)
        return self._driven_segment(lung, (b.target, 0.0), pmus, starvation)

    def _driven_segment(self, lung, pressure, pmus, series_resistance=0.0):
        if lung.expiratory_obstruction <= 0.0:
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
            if lung.expiratory_obstruction > 0.0:
                resistance, ceiling_resistance, _ = lung.expiration_parameters((0.0, 0.0), (pmus, 0.0), self._series_resistance())
                r = resistance + self._series_resistance()
                ceiling = max(0.0, (lung.peep + lung.elastic_pressure + lung.p2 + lung._expiratory_collapse_offset()) / ceiling_resistance)
            q = 2.0 * recoil / (r + math.sqrt(r * r + 4.0 * k * recoil)) if recoil > 0.0 else 0.0
            q = min(q, ceiling)
            self._valve = (knot, -k * q * q, 2.0 * k * q)
        return self._valve[1], self._valve[2]

    def _boundary(self) -> float:
        """Time to the next scheduled change of segment."""
        b = self._breath
        if b is not None and b.inspiring:
            bt = b.t
            best: float | None = b.ti - bt if b.ti > bt + 1e-12 else None
            if b.kind == "VC" and (not b.limited or self._cpr_active) and not b.delivered and not b.depleted and b.flow_end > bt + 1e-12:
                dt = b.flow_end - bt
                best = dt if best is None or dt < best else best
            if b.kind in ("PC", "VG", "PS") and RISE_TIME > bt + 1e-12:
                dt = RISE_TIME - bt
                best = dt if best is None or dt < best else best
            if not b.mandatory and self._synchronized():
                end = bt + self._sync_window_start() - self._since_mandatory
                if end > bt + 1e-12:
                    dt = end - bt
                    best = dt if best is None or dt < best else best
            return 0.0 if best is None else best
        period = self._mandatory_period()
        mode = self.settings.mode
        best = math.inf
        if self._mandatory_type() is not None:
            dt = period - self._since_mandatory
            if dt > 1e-12:
                best = dt
            if self._synchronized():
                dt = self._sync_window_start() - self._since_mandatory
                if dt > 1e-12 and dt < best:
                    best = dt
        elif self._source == "vent" and mode == "PSV" and self.settings.rr > 0.0:
            dt = self.apnea_backup_s - self._since_breath
            if dt > 1e-12:
                best = dt
        knot = bisect_right(VALVE_KNOTS, self._since_exhalation + 1e-12)
        if knot < len(VALVE_KNOTS) and self._valve_coefficient() > 0.0:
            dt = VALVE_KNOTS[knot] - self._since_exhalation
            if dt > 1e-12 and dt < best:
                best = dt
        if (
            self._source == "spontaneous"
            and not self._manual_settled
            and self._measured is not None
            and self._measured.kind == "MANUAL"
            and not self._measured.inspiring
        ):
            for settle_t in (2.0, 4.0):
                dt = settle_t - self._since_exhalation
                if dt > 1e-12 and dt < best:
                    best = dt
        return best

    def _advance(self, available: float, lung) -> float:
        effort = lung.effort
        if not self.inspiring and self._due():
            self._set_lung_mandatory_flags(lung)
            self._start(lung, self._mandatory_type(), True)
        b = self._active_breath if self.inspiring else None
        boundary, effort_break = self._boundary(), effort.time_to_break()
        h = min(available, effort_break, boundary)
        cpr_break = math.inf
        if self._cpr_active:
            _, _, cpr_knot_dt = compression_pressure_drive(self._cpr_clock)
            cpr_interval = lung.EXPIRATORY_INTERVAL
            cpr_tick = math.floor((self._cpr_elapsed + 1e-12) / cpr_interval)
            cpr_break = min(cpr_knot_dt, (cpr_tick + 1) * cpr_interval - self._cpr_elapsed)
            h = min(h, cpr_break)
        if lung.aeration is not None:
            h = min(h, lung.aeration.INTERVAL - lung.aeration.elapsed)
        if lung.expiratory_obstruction > 0.0 and (b is None or b.kind in ("PC", "VG", "PS", "SPONT")
                                      or (b.kind in ("VC", "MANUAL") and b.limited)):
            interval = lung.EXPIRATORY_INTERVAL
            knot = math.floor((self._since_exhalation + 1e-12) / interval)
            h = min(h, (knot + 1) * interval - self._since_exhalation)
        pmus = effort.pmus()
        # A fresh spontaneous effort discards any already-exhaled sub-threshold
        # unassisted breath so obstructed shallow efforts never merge into one VTe.
        if (
            self._source is not None
            and not self.inspiring
            and effort.clock <= 1e-9
            and pmus[1] > 0.0
            and self._measured is not None
            and not self._measured.inspiring
            and not self._measured.mandatory
            and 1e-6 < self._measured.expired < MIN_EXPIRATION_L
        ):
            self._measured = None
        segment = self._segment(lung, pmus)
        watch = self._watch()
        if watch == "zero_flow" and self._cpr_active and pmus[0] > 0.0:
            watch = None
        mandatory_exhaling = (
            self._measured is not None
            and self._measured.mandatory
            and not self._measured.inspiring
            and self._measured.expired < max(MIN_EXPIRATION_L, 0.75 * self._measured.inspired)
            and lung.volume > 0.10
        )
        observe = (
            self._source is not None
            and pmus[1] > 0.0
            and not mandatory_exhaling
            and (
                self._measured is None
                or (not self._measured.inspiring and self._measured.expired >= MIN_EXPIRATION_L)
            )
        )
        passive_inspiration = (not self.inspiring and self._measured is not None
                               and self._measured.kind == "SPONT" and self._measured.inspiring)
        onset = effort.onset_volume()
        avail_vol = self._effective_available_volume()
        can_deplete = (
            b is not None and b.inspiring and not b.depleted and math.isfinite(avail_vol)
        )
        can_passive_deplete = (
            not can_deplete
            and (b is None or not b.inspiring)
            and self._source is not None
            and math.isfinite(avail_vol)
            and not self._bag_exhausted(lung)
            and (segment.flow(0.0) > 1e-9 or segment.flow(h) > 1e-9)
        )
        # Analytical FlowSegment/PressureSegment searches for "limit" and
        # passive-lung depletion stay exact over full segment spans; keep
        # EVENT_STEP for trigger/cycle/zero_flow and active-effort watches.
        if (
            observe
            or passive_inspiration
            or onset is not None
            or watch in ("trigger", "cycle", "zero_flow")
            or ((can_deplete or can_passive_deplete) and pmus != (0.0, 0.0))
        ):
            h = min(h, EVENT_STEP)
        t, event = (h, None) if watch is None else self._search(segment, h, watch)
        if can_deplete:
            deplete_t = self._volume_reached(segment, t, self._bag_ref_volume + avail_vol, 1.0)
            if deplete_t is not None and (event is None or deplete_t < t - 1e-12):
                t, event = deplete_t, "deplete"
        elif can_passive_deplete:
            deplete_t = self._volume_reached(segment, t, self._bag_ref_volume + avail_vol, 1.0)
            if deplete_t is not None and deplete_t > 1e-12 and (event is None or deplete_t < t - 1e-12):
                t, event = deplete_t, "passive_deplete"
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
        seg_area = segment.area(t)
        self._measure(segment, t, seg_area)
        p0, slope = pmus
        pmus_seg_area = p0 * t + 0.5 * slope * t * t
        self.muscle_pressure_area += pmus_seg_area
        self.capped_muscle_pressure_area += self._capped_pmus_area(
            p0, slope, t, self._effort_cap, pmus_seg_area
        )
        if self._measured is not None and self._measured.inspiring:
            self._measured.pmus_area += max(0.0, pmus_seg_area)
            if b is not None and b.kind != "SPONT":
                self._measured.support_area += max(0.0, seg_area)
        peep_eff = self._eff_baseline or 0.0
        aeration = lung.aeration
        need_alveolar = self._source in ("vent", "bag") or peep_eff > 0.0 or self.manual_breath_active
        transmission = clamp01(aeration.pressure_transmission) if aeration is not None else 1.0
        raw_distending = segment.distending_area(t) if (need_alveolar or aeration is not None) else 0.0
        if need_alveolar:
            dist_palv = raw_distending * transmission if (self._breath is None or self._breath.kind != "SPONT") else raw_distending
            self.alveolar_pressure_area += max(0.0, peep_eff * t + dist_palv - pmus_seg_area)
        elif (
            lung.expiratory_obstruction > 0.0
            and lung._prev_ee_excess_l is not None
            and lung._prev_ee_excess_l > 0.5 * lung.compliance
        ):
            self.alveolar_pressure_area += (lung._prev_ee_excess_l / lung.compliance) * t
        segment.commit(t)
        raw_paw_end = self._peep() + segment.paw(t)
        self._paw_end = max(MIN_CIRCUIT_PAW, raw_paw_end) if self._source is not None else raw_paw_end
        if effort.clock < effort.ti:
            if b is not None and b.kind != "SPONT":
                assistance = seg_area * self._volume_delivery
            elif (
                self._source in ("vent", "bag")
                and self._mandatory_type() is not None
                and not self._triggering()
                and self._recent_totals[0] > 0.0
                and self._recent_totals[3] > 0.0
            ):
                # In non-triggered mandatory modes (VCV/PCV/PCV-VG), spontaneous
                # efforts are asynchronous with mandatory inflation; weight the recent
                # mandatory inspiratory pressure above PEEP by mandatory_share so
                # controlled ventilation unloads effort while breakthrough hyperpnea
                # (e.g. malignant hyperthermia) is not suppressed (dt-invariant).
                recent_mean_above_peep = max(0.0, self._recent_totals[2] / self._recent_totals[0] - self._peep())
                if self._source == "bag":
                    period = self._mandatory_period()
                    duty = (
                        clamp(min(BAG_TI_MAX_S, period / 3.0) / period, 0.05, 1.0 / 3.0)
                        if math.isfinite(period)
                        else 1.0 / 3.0
                    )
                else:
                    duty = clamp(self.settings.ie_ratio / (1.0 + self.settings.ie_ratio), 0.2, 0.8)
                assistance = (recent_mean_above_peep / duty) * self.mandatory_share * t * self._volume_delivery
            else:
                assistance = 0.0
            effort.note_assistance(t, assistance)
        effort.advance(t, lung, self._series_resistance(), pmus_seg_area)
        self._since_mandatory += t
        self._since_breath += t
        self._since_measured += t
        self._since_exhalation += t
        self._mandatory_peak_age_s += t
        if self._cpr_active:
            self._cpr_clock = (self._cpr_clock + t) % COMPRESSION_PERIOD_S
            self._cpr_elapsed += t
        if b is not None:
            b.t += t
        # The segment still describes the present unless something changed at its end.
        changed = event is not None or t >= min(boundary, effort_break, cpr_break) - 1e-12
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
        if aeration is not None:
            # Recoil plus tissue pressure excludes airway and circuit drops,
            # including the extra drop when expiratory flow is limited.
            peep_area = peep_eff * t
            distending = raw_distending
            if (
                self._source in ("vent", "bag")
                or (self._breath is not None and self._breath.kind == "MANUAL")
            ) and (self._breath is None or self._breath.kind != "SPONT"):
                distending *= transmission
            shift = lung.update_aeration(
                t, peep_area + distending, peep_eff, cpr_active=self._cpr_active,
            )
            if aeration.elapsed == 0.0:
                self._shift_breath_reference(shift)
        if not self.inspiring:
            self._sync_expiratory_baseline(lung)
        return t

    @staticmethod
    def _capped_pmus_area(p0: float, slope: float, t: float, cap: float, area: float) -> float:
        p_end = p0 + slope * t
        if p0 <= cap and p_end <= cap:
            return area
        if p0 >= cap and p_end >= cap:
            return cap * t
        if abs(slope) > 1e-12:
            u_c = clamp((cap - p0) / slope, 0.0, t)
            if p0 < cap:
                return 0.5 * (p0 + cap) * u_c + cap * (t - u_c)
            return cap * u_c + 0.5 * (cap + p_end) * (t - u_c)
        return min(p0, cap) * t

    @staticmethod
    def _volume_reached(segment, t: float, level: float, sign: float) -> float | None:
        """First time in [0, t] at which volume rises (sign 1) or falls (sign -1) to level."""
        if type(segment) is FlowSegment:
            f0 = sign * (segment.v0 - level) + 1e-12
            if f0 >= 0.0:
                return 0.0
            ft = sign * (segment.v0 + segment.q * t - level) + 1e-12
            if ft < 0.0 or sign * segment.q <= 0.0:
                return None
            return clamp((level - segment.v0 - sign * 1e-12) / segment.q, 0.0, t)
        # Count thresholds reached within roundoff, including constant-flow plateaus.
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
            self._since_mandatory = max(self._since_mandatory, self._mandatory_period())
        return self._mandatory_type() is not None and self._since_mandatory >= self._mandatory_period() - 1e-12

    # --- Events --------------------------------------------------------------

    def _watch(self):
        """Return the crossing to search for in the current segment, if any."""
        b = self._breath
        if b is not None and b.inspiring:
            if b.ti <= 1e-12:
                return None
            if b.kind in ("VC", "MANUAL"):
                if b.limited:
                    return None if (b.delivered or b.depleted) else "volume"
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
            f0 = f(0.0)
            if f0 >= 0.0:
                return 0.0, watch
            d0, dh = segment.dpaw(0.0), segment.dpaw(h)
            if d0 > 0.0 > dh:
                peak = find_root(segment.dpaw, 0.0, h, d0, dh)
                fp = f(peak)
                if fp >= 0.0:
                    return find_root(f, 0.0, peak, f0, fp), watch
            fh = f(h)
            if fh >= 0.0:
                return find_root(f, 0.0, h, f0, fh), watch
            return h, None
        if watch == "volume":
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
            level = (
                MEASURE_FLOW
                if watch == "measure" or detect_only or self._source == "bag"
                else self.settings.trigger / 60.0
            )
            f = lambda t: segment.flow(t) - level  # noqa: E731
            if watch == "trigger":
                pmus = segment.lung.effort.pmus()
                # Chest-recoil inflow can auto-trigger sensitive flow triggers during CPR
                # (bench manikin study in Neumamm 2021, extended here to PSV/SIMV) when
                # recoil exceeds settings.trigger in supported or synchronized modes (excluding apnea and CPAP).
                cpr_auto_trigger = (
                    self._cpr_active
                    and self._source == "vent"
                    and self.settings.mode in TRIGGERED
                    and self.settings.mode != "CPAP"
                    and (
                        self.settings.p_support > 0.0
                        or (self._synchronized() and self._since_mandatory >= self._sync_window_start() - 1e-12)
                    )
                )
                if pmus[0] <= 0.0 and pmus[1] <= 0.0 and not cpr_auto_trigger:
                    self._armed = f(h) < 0.0
                    return h, None
                if not self._armed:
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
            self._active_breath.clipped = True
            return
        if event == "volume":
            b = self._active_breath
            b.limited, b.delivered = False, True
            if b.kind == "MANUAL":
                self._end_inspiration(segment, t)
            return
        if event == "deplete":
            self._active_breath.depleted = True
            if self._active_breath.kind in ("PS", "SPONT"):
                self._end_inspiration(segment, t)
            return
        if event == "passive_deplete":
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
            segment.lung._exp_trough_vol = segment.lung.volume
            return
        if event == "inflated":
            lung.effort.end_contraction()
            return
        if event == "deflated":
            lung.effort.resume(lung, self._series_resistance())
            return
        if b is not None and b.inspiring:
            if not b.mandatory and self._synchronized():
                window = self._sync_window_start()
                pmus = lung.effort.pmus()
                trigger_flow = MEASURE_FLOW if self._source == "bag" else self.settings.trigger / 60.0
                if (
                    self._since_mandatory >= window - 1e-12
                    and segment.flow(t) >= trigger_flow
                    and (pmus[0] > 0.0 or pmus[1] > 0.0)
                ):
                    # An effort already in progress when the window opens is
                    # eligible too; waiting for a fresh flow crossing stacks breaths.
                    mandatory_kind = self._mandatory_type()
                    if mandatory_kind is not None:
                        self._start(lung, mandatory_kind, True)
                        return
            if event in ("cycle", "zero_flow") or b.t >= b.ti - 1e-12:
                self._end_inspiration(segment, t)
        if (
            self._source == "spontaneous"
            and not self.inspiring
            and not self._manual_settled
            and self._measured is not None
            and self._measured.kind == "MANUAL"
            and not self._measured.inspiring
            and self._since_exhalation >= 2.0 - 1e-12
            and self._measured.expired >= MIN_EXPIRATION_L
            and (
                (lung.elastic_pressure < 0.2 if self._cpr_active else abs(lung.elastic_pressure) < 0.2)
                or (self._since_exhalation >= 4.0 - 1e-12 and abs(segment.flow(t)) * 60.0 < 0.05)
            )
        ):
            provisional = not self._recent
            counted = self._finish(lung, push_recent=provisional, count_breath=True)
            self._manual_settled = counted
            self._recent_provisional = counted and provisional

    def _on_trigger(self, lung) -> None:
        if self._synchronized():
            window = self._sync_window_start()
            if self._since_mandatory >= window - 1e-12:
                mandatory_kind = self._mandatory_type()
                if mandatory_kind is not None:
                    self._start(lung, mandatory_kind, True)
                    return
        self._backup = False
        supported = self._source == "vent" and self.settings.mode != "CPAP" and self.settings.p_support > 0.0
        self._start(lung, "PS" if supported else "SPONT", False)

    def _end_inspiration(self, segment, t) -> None:
        b = self._active_breath
        b.inspiring = False
        self.inspiring = False
        if self._measured is not None:
            self._measured.inspiring = False
            if b.kind != "SPONT":
                self._measured.net_insp_end = self._measured.net_volume
        self._since_exhalation, self._valve = 0.0, None
        self._expiration = None
        segment.lung._exp_trough_vol = segment.lung.volume
        end_paw = b.peep + segment.paw(t)
        volume = (
            max(0.0, segment.volume(t) - b.v_start)
            if self._cpr_active and b.kind in ("PC", "VG")
            else max(0.0, b.v_max - b.v_start)
        )
        delivered_volume = volume * self._volume_delivery
        if b.kind == "VC":
            if not b.limited and (b.delivered or b.depleted or b.flow_end < b.ti):
                b.plat = end_paw
            short_cpr = self._cpr_active and b.clipped and b.target > 1e-9 and volume < 0.95 * b.target
            self.pressure_limited = (b.limited and not b.delivered) or short_cpr
            if b.vg_test:
                goal = self.settings.tv / 1000.0
                ceiling = self._vg_ceiling(b.peep)
                cpr_end_p = compression_pressure_drive(self._cpr_clock)[0] if self._cpr_active else 0.0
                est_end_paw = max(b.peep, end_paw - cpr_end_p)
                if not b.depleted and delivered_volume >= 0.9 * goal:
                    self._vg_pressure = min(max(est_end_paw - b.peep, VG_MIN), ceiling)
                elif not b.depleted and b.limited:
                    self._vg_pressure = ceiling
                elif not b.depleted and self._volume_delivery < 1.0 - 1e-6 and delivered_volume > 0.0:
                    eff_vol = max(delivered_volume, volume / 1.45)
                    base_p = max(est_end_paw - b.peep, VG_MIN)
                    step_up = clamp(base_p * (goal / eff_vol - 1.0), -VG_STEP, VG_STEP)
                    self._vg_pressure = min(max(base_p + step_up, VG_MIN), ceiling)
                elif self._vg_pressure is None:
                    self._vg_pressure = min(max(self.settings.p_insp, VG_MIN), ceiling)
        elif b.kind == "MANUAL":
            short_cpr = self._cpr_active and b.clipped and b.target > 1e-9 and volume < 0.95 * b.target
            self.pressure_limited = (b.limited and not b.delivered) or short_cpr
        elif b.kind in ("PC", "VG"):
            # A flat pressure trace still includes airway resistance while
            # gas flows. Only approximate a plateau after flow has settled.
            flow = abs(segment.flow(t))
            if flow <= PLATEAU_FLOW and flow * segment.lung.resistance <= PLATEAU_PRESSURE:
                b.plat = end_paw
            if b.kind == "VG":
                # Cap leak compensation against runaway ghost inflation on severe
                # facemask leaks while still compensating moderate leaks (<= 1.45x VT).
                vg_vol = max(delivered_volume, volume / 1.45) if self._volume_delivery < 1.0 - 1e-6 else delivered_volume
                self._adjust_vg(vg_vol, b, delivered_volume=delivered_volume)
        if segment.lung.effort.pmus()[0] > PLATEAU_PRESSURE or self._cpr_active:
            b.plat = math.nan
        if b.mandatory and self._active_measurement.mandatory:
            self._active_measurement.plat = b.plat

    def _adjust_vg(self, volume: float, b: Breath, *, delivered_volume: float | None = None) -> None:
        """Move Pinsp toward the set VT by at most VG_STEP, retaining the Pmax margin."""
        goal = self.settings.tv / 1000.0
        actual_delivered = volume if delivered_volume is None else delivered_volume
        if b.depleted:
            self.pressure_limited = actual_delivered < 0.9 * goal
            return
        pressure = self._vg_pressure if self._vg_pressure is not None else b.target
        if volume > 0.0:
            pressure += min(max(pressure * (goal / volume - 1.0), -VG_STEP), VG_STEP)
        ceiling = self._vg_ceiling(b.peep)
        self._vg_pressure = min(max(pressure, VG_MIN), ceiling)
        self.pressure_limited = (
            (self._vg_pressure >= ceiling or actual_delivered < volume - 1e-6)
            and actual_delivered < 0.9 * goal
        )

    # --- Measurement ---------------------------------------------------------

    def _measure(self, segment, t: float, seg_area: float) -> None:
        b = self._breath
        if t <= 0.0:
            return
        # Split gas movement at a reversal even when it occurs within one outer
        # step. Subsamples keep analyzer and airway-sensor timing independent of it.
        f0, f1 = segment.flow(0.0), segment.flow(t)
        v0, vt = segment.volume(0.0), segment.volume(t)
        v_rev = vt
        if (f0 > 0.0 > f1) or (f0 < 0.0 < f1):
            t_rev = find_root(segment.flow, 0.0, t, f0, f1)
            v_rev = segment.volume(t_rev)
            intervals: tuple[tuple[float, float, float, float], ...] = (
                (0.0, t_rev, v0, v_rev),
                (t_rev, t, v_rev, vt),
            )
            min_vol = min(v0, v_rev, vt)
        else:
            intervals = ((0.0, t, v0, vt),)
            min_vol = v0 if v0 <= vt else vt
        # Flow reversals bound monotonic volume intervals.
        segment.lung.note_volume(min_vol)
        segment.lung.effort.note_volume(min_vol)
        measured = self._measured
        peep = self._peep()
        if self._collect_samples:
            connected = self._source is not None
            lung = segment.lung
            peep_volume = lung.compliance * (self._eff_baseline or 0.0) if connected else 0.0
            volume_offset = lung.volume_offset
        for start, end, vol_start, vol_end in intervals:
            if end <= start:
                continue  # A reversal rounded to an existing boundary moves no gas.
            change = vol_end - vol_start
            if measured is not None:
                if self._cpr_active:
                    measured.cpr_oscillating = True
                measured.net_volume += change
                if change >= 0.0:
                    measured.inspired += change
                else:
                    measured.expired -= change
                    if not (self._cpr_active and b is not None and b.inspiring):
                        measured.inspiring = False
                if b is not None and b.inspiring and b.kind != "SPONT":
                    measured.net_insp_end = measured.net_volume
                elif measured.kind == "SPONT" and measured.net_volume > measured.net_insp_end:
                    measured.net_insp_end = measured.net_volume
            if self._collect_samples:
                n = max(1, math.ceil((end - start) / 0.01))
                previous, previous_volume = start, vol_start
                for i in range(1, n + 1):
                    u = start + (end - start) * i / n
                    volume, flow, paw = segment.sample(u)
                    dv = volume - previous_volume
                    self.samples.append((u - previous, dv,
                                         max(MIN_CIRCUIT_PAW, peep + paw) if connected else 0.0,
                                         flow * 60.0 if connected else 0.0,
                                         volume + peep_volume + volume_offset if connected else 0.0))
                    previous, previous_volume = u, volume
        if measured is not None:
            peak_paw = max(segment.paw(0.0), segment.paw(t))
            d0, d1 = segment.dpaw(0.0), segment.dpaw(t)
            if d0 > 0.0 > d1:
                peak_paw = max(peak_paw, segment.paw(find_root(segment.dpaw, 0.0, t, d0, d1)))
            measured.paw_peak = max(measured.paw_peak, peep + peak_paw)
            measured.area += seg_area + peep * t
            measured.t += t
        if b is None or not b.inspiring:
            return
        volume = vt
        if f0 > 0.0 > f1:
            volume = max(volume, v_rev)
        b.v_max = max(b.v_max, volume)

    def _sample(self, lung) -> None:
        """Airway pressure, flow, and volume at the end of the step."""
        segment, t = self._current or (self._segment(lung, lung.effort.pmus()), 0.0)
        self.paw = self._peep() + segment.paw(t)
        if self._source is not None:
            self.paw = max(MIN_CIRCUIT_PAW, self.paw)
        self.flow = segment.flow(t) * 60.0
        self.volume = lung.volume + lung.compliance * (self._eff_baseline or 0.0) + lung.volume_offset
