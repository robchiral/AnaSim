"""Respiratory system mechanics and inspiratory muscle effort."""

import math
from bisect import bisect_right
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lung import LungAeration


class RespiratoryMechanics:
    """Equation of motion: Paw + Pmus = R x flow + V/C + P2, pressures above PEEP.

    P2 is tissue stress adaptation, a spring in series with a dashpot that lies
    parallel to the static elastance: dP2/dt = E2 x flow - P2/tau2. Volume is
    measured from the current reference at the set PEEP. Each segment is
    integrated exactly; aeration and expiratory coefficients update on fixed
    physical clocks.
    """

    EXPIRATORY_INTERVAL = 0.02  # s; volume-dependent coefficients use a physical clock

    def __init__(self, compliance: float = 0.05, resistance: float = 10.0,
                 aeration: "LungAeration | None" = None):
        """Static compliance in L/cmH2O; airway and tube resistance in cmH2O/(L/s)."""
        self.compliance = compliance
        self.resistance = resistance
        self.bronchospasm = 0.0
        self.bronch_resistance = 0.0
        self.aeration = aeration
        self.peep = 0.0
        self.volume_offset = 0.0  # Shift of the static curve relative to the initial relaxed volume
        if aeration is not None:
            self.compliance = aeration.reference_compliance * aeration.recruited / aeration.REFERENCE_RECRUITED
            self.volume_offset = aeration.relaxed_volume - aeration.reference_volume
        # Healthy anesthetized adults: viscoelastic compliance 4x static
        # compliance, time constant 0.82 s (Jonson 1993).
        self.viscoelastic_ratio = 0.25  # E2 / static elastance
        self.viscoelastic_tau = 0.82  # s
        self.volume = 0.0  # L
        self.p2 = 0.0  # cmH2O
        self.effort = PatientEffort()

    def update_aeration(self, dt: float, pressure_area: float, peep: float) -> float:
        """Return the coordinate shift after updating the static curve, conserving gas volume."""
        aeration = self.aeration
        absolute = aeration.reference_volume + self.volume_offset + self.compliance * peep + self.volume
        aeration.frc = min(aeration.frc, absolute)
        if not aeration.advance(dt, pressure_area):
            return 0.0
        recoil, compliance = aeration.static_mechanics(absolute)
        volume = compliance * (recoil - peep)
        shift = volume - self.volume
        self.compliance, self.volume = compliance, volume
        self.volume_offset = absolute - aeration.reference_volume - compliance * peep - volume
        self.effort.shift_volume_reference(shift)
        return shift

    def note_end_expiration(self, peep: float) -> None:
        if self.aeration is not None:
            self.aeration.frc = (self.aeration.reference_volume + self.volume_offset
                                 + self.compliance * peep + self.volume)

    @property
    def reference_compliance(self) -> float:
        """Low-inflation compliance at the reference aeration, L/cmH2O.

        Integrated patients use this to change stiffness; compliance is the
        current tangent of their changing pressure-volume curve.
        """
        return self.compliance if self.aeration is None else self.aeration.reference_compliance

    @reference_compliance.setter
    def reference_compliance(self, value: float) -> None:
        if self.aeration is None:
            self.compliance = value
        else:
            self.aeration.reference_compliance = value

    @property
    def elastic_pressure(self) -> float:
        """Static recoil pressure above PEEP, as an end-expiratory hold measures it."""
        return self.volume / self.compliance

    def flow_segment(self, flow: float, pmus: tuple[float, float] = (0.0, 0.0)) -> "FlowSegment":
        """Constant inspiratory flow (L/s) set by the ventilator, or a pause at zero flow."""
        return FlowSegment(self, flow, pmus)

    def sine_segment(self, volume: float, duration: float, start: float,
                     pmus: tuple[float, float] = (0.0, 0.0)) -> "SineSegment":
        """Half-sine flow delivering volume (L) over duration, from start seconds in."""
        return SineSegment(self, volume, duration, start, pmus)

    def pressure_segment(self, pressure: tuple[float, float], pmus: tuple[float, float] = (0.0, 0.0),
                         series_resistance: float = 0.0) -> "PressureSegment":
        """Pressure p0 + p1 t applied through series_resistance, such as a circuit limb."""
        return PressureSegment(self, pressure, pmus, series_resistance)

    def expiration_parameters(self, pressure, pmus, series_resistance) -> tuple[float, float, bool]:
        """Expiratory resistance and a recoil-dependent flow ceiling.

        Narrowed airways lose radial traction as the lung empties. A simplified
        equal-pressure-point model limits flow independently of downstream
        pressure (Mead 1967). The resistance curve, severity scaling, and 5 cmH2O
        resting transpulmonary offset are teaching estimates, not patient fits.
        """
        inflation = self.volume + self.compliance * self.peep
        if self.aeration is not None:
            inflation += self.volume_offset + self.aeration.reference_volume - self.aeration.relaxed_volume
        scale = 10.0 * self.reference_compliance
        resistance = self.resistance + 2.0 * self.bronch_resistance / (1.0 + max(0.0, inflation) / scale) ** 2
        ceiling_resistance = resistance * (1.0 + 0.75 * self.bronchospasm ** 2)
        recoil = self.elastic_pressure + self.p2
        free_flow = (pressure[0] + pmus[0] - recoil) / (resistance + series_resistance)
        if free_flow >= 0.0:
            # Effort can draw gas through the open circuit before a trigger.
            return self.resistance, ceiling_resistance, False
        ceiling = (self.peep + recoil + 5.0) / ceiling_resistance
        limited = ceiling > 0.0 and free_flow < -ceiling
        return resistance, ceiling_resistance, limited


class FlowSegment:
    """Flow is set; Paw = R x flow + V/C + P2 - Pmus. Times are seconds from the segment start."""

    __slots__ = ("lung", "v0", "p20", "q", "m0", "m1", "elastance", "tau", "asymptote")

    def __init__(self, lung: RespiratoryMechanics, q: float, pmus: tuple[float, float]):
        self.lung = lung
        self.v0, self.p20 = lung.volume, lung.p2
        self.q = q
        self.m0, self.m1 = pmus
        self.elastance = 1.0 / lung.compliance
        self.tau = lung.viscoelastic_tau
        self.asymptote = lung.viscoelastic_ratio * self.elastance * q * self.tau

    def volume(self, t: float) -> float:
        return self.v0 + self.q * t

    def flow(self, t: float) -> float:
        return self.q

    def p2(self, t: float) -> float:
        return self.asymptote + (self.p20 - self.asymptote) * math.exp(-t / self.tau)

    def paw(self, t: float) -> float:
        return (self.lung.resistance * self.flow(t) + self.elastance * self.volume(t) + self.p2(t)
                - self.m0 - self.m1 * t)

    def dpaw(self, t: float) -> float:
        return self.elastance * self.q + (self.asymptote - self.p20) / self.tau * math.exp(-t / self.tau) - self.m1

    def area(self, t: float) -> float:
        """Integral of Paw over [0, t]."""
        volume_area = self.v0 * t + 0.5 * self.q * t * t
        return self._paw_area(t, volume_area)

    def _paw_area(self, t: float, volume_area: float) -> float:
        # dP2/dt = E2 x flow - P2/tau integrates to tau (E2 dV - dP2).
        dv = self.volume(t) - self.v0
        p2_area = self.tau * (self.lung.viscoelastic_ratio * self.elastance * dv - (self.p2(t) - self.p20))
        return (self.lung.resistance * dv + self.elastance * volume_area + p2_area
                - self.m0 * t - 0.5 * self.m1 * t * t)

    def distending_area(self, t: float) -> float:
        """Integral of static recoil plus tissue pressure, above PEEP."""
        return (self.area(t) + self.m0 * t + 0.5 * self.m1 * t * t
                - self.lung.resistance * (self.volume(t) - self.v0))

    def commit(self, t: float) -> None:
        self.lung.volume, self.lung.p2 = self.volume(t), self.p2(t)


class SineSegment(FlowSegment):
    """Flow (V/2) w sin(w u), w = pi/duration, where u is time since the flow began."""

    __slots__ = ("w", "amplitude", "u0", "gain", "offset")

    def __init__(self, lung, volume, duration, start, pmus):
        super().__init__(lung, 0.0, pmus)
        self.w = math.pi / duration
        self.amplitude = 0.5 * volume
        self.u0 = start
        # P2 = offset exp(-t/tau) + gain F(u), with F the particular solution for sin(w u).
        self.gain = lung.viscoelastic_ratio * self.elastance * self.amplitude * self.w
        self.offset = self.p20 - self.gain * self._forced(start)

    def _forced(self, u: float) -> float:
        rate = 1.0 / self.tau
        return (rate * math.sin(self.w * u) - self.w * math.cos(self.w * u)) / (rate * rate + self.w * self.w)

    def volume(self, t: float) -> float:
        u = self.u0 + t
        return self.v0 + self.amplitude * (math.cos(self.w * self.u0) - math.cos(self.w * u))

    def flow(self, t: float) -> float:
        return self.amplitude * self.w * math.sin(self.w * (self.u0 + t))

    def p2(self, t: float) -> float:
        return self.offset * math.exp(-t / self.tau) + self.gain * self._forced(self.u0 + t)

    def dpaw(self, t: float) -> float:
        flow = self.flow(t)
        dflow = self.amplitude * self.w * self.w * math.cos(self.w * (self.u0 + t))
        dp2 = self.lung.viscoelastic_ratio * self.elastance * flow - self.p2(t) / self.tau
        return self.lung.resistance * dflow + self.elastance * flow + dp2 - self.m1

    def area(self, t: float) -> float:
        u0, u1 = self.u0, self.u0 + t
        volume_area = (self.v0 + self.amplitude * math.cos(self.w * u0)) * t - self.amplitude * (
            math.sin(self.w * u1) - math.sin(self.w * u0)) / self.w
        return self._paw_area(t, volume_area)


class PressureSegment:
    """Pressure P(t) = p0 + p1 t drives flow through the series and airway resistances.

    With x = (V, P2) and drive D = P + Pmus, x' = A x + b D. For a linear
    drive the solution is exp(At)(x0 - p) + p + q t, with the particular
    solution p + q t below. Paw at the airway is P - series resistance x flow.
    The ventilator queries the same few times repeatedly, so the start and the
    latest time are kept.
    """

    __slots__ = ("lung", "v0", "p20", "p0", "p1", "d0", "d1", "rs", "elastance", "s", "w",
                 "a11", "a12", "a21", "a22", "y0", "z0", "az", "slope", "pv", "pp",
                 "start", "last_t", "last", "rt")

    def __init__(self, lung: RespiratoryMechanics, pressure, pmus, series_resistance):
        self.lung = lung
        self.v0, self.p20 = lung.volume, lung.p2
        self.p0, self.p1 = pressure
        self.d0, self.d1 = self.p0 + pmus[0], self.p1 + pmus[1]
        self.rs = series_resistance
        e = self.elastance = 1.0 / lung.compliance
        tau = lung.viscoelastic_tau
        e2 = lung.viscoelastic_ratio * e
        rt = self.rt = lung.resistance + series_resistance
        self.a11, self.a12 = -e / rt, -1.0 / rt
        self.a21, self.a22 = -e2 * e / rt, -e2 / rt - 1.0 / tau
        self.s = 0.5 * (self.a11 + self.a22)
        determinant = e / (rt * tau)
        self.w = math.sqrt(max(0.0, self.s * self.s - determinant))
        # Slow drive changes leave V lagging by (R + R2) / E behind D/E.
        self.slope = self.d1 / e
        self.pv = self.d0 / e - (rt + e2 * tau) * self.d1 / (e * e)
        self.pp = e2 * tau * self.d1 / e
        self.y0 = (self.v0 - self.pv, self.p20 - self.pp)
        self.z0 = self._apply(self.y0)
        self.az = self._apply(self.z0)
        self.start = (self.v0, self.p20, self.z0[0] + self.slope, self.az[0])
        self.last_t = self.last = None

    def _apply(self, x):
        return self.a11 * x[0] + self.a12 * x[1], self.a21 * x[0] + self.a22 * x[1]

    def _evaluate(self, t: float):
        """(V, P2, flow, d flow/dt) at t, using exp(At) x = e^{st} (cosh(wt) x + sinh(wt)/w (A - sI) x)."""
        if t == 0.0:
            return self.start
        if t == self.last_t:
            return self.last
        s, wt = self.s, self.w * t
        if wt < 1e-4:
            decay = math.exp(s * t)
            c, g = decay * (1.0 + 0.5 * wt * wt), decay * t * (1.0 + wt * wt / 6.0)
        else:
            fast, slow = math.exp((s + self.w) * t), math.exp((s - self.w) * t)
            c, g = 0.5 * (fast + slow), 0.5 * (fast - slow) / self.w
        (y1, y2), (z1, z2), (az1, az2) = self.y0, self.z0, self.az
        self.last_t = t
        self.last = (c * y1 + g * (z1 - s * y1) + self.pv + self.slope * t,
                     c * y2 + g * (z2 - s * y2) + self.pp,
                     c * z1 + g * (az1 - s * z1) + self.slope,
                     c * az1 + g * (self.a11 * az1 + self.a12 * az2 - s * az1))
        return self.last

    def state(self, t: float):
        return self._evaluate(t)[:2]

    def volume(self, t: float) -> float:
        return self._evaluate(t)[0]

    def flow(self, t: float) -> float:
        return self._evaluate(t)[2]

    def dflow(self, t: float) -> float:
        return self._evaluate(t)[3]

    def paw(self, t: float) -> float:
        return self.p0 + self.p1 * t - self.rs * self.flow(t)

    def dpaw(self, t: float) -> float:
        return self.p1 - self.rs * self.dflow(t)

    def area(self, t: float) -> float:
        return self.p0 * t + 0.5 * self.p1 * t * t - self.rs * (self.volume(t) - self.v0)

    def distending_area(self, t: float) -> float:
        return self.d0 * t + 0.5 * self.d1 * t * t - self.rt * (self.volume(t) - self.v0)

    def commit(self, t: float) -> None:
        self.lung.volume, self.lung.p2 = self.state(t)


class ExpiratorySegment(PressureSegment):
    """Exact segment with frozen expiratory resistance or an active flow ceiling.

    When limited, recoil rather than downstream pressure drives flow through
    an upstream resistance. The virtual boundary is 5 cmH2O below the relaxed
    pressure. Physical Paw still includes the circuit's actual pressure drop.
    """

    __slots__ = ()

    def __init__(self, lung, pressure, pmus, series_resistance, parameters):
        resistance, ceiling_resistance, limited = parameters
        if limited:
            super().__init__(lung, (-lung.peep - 5.0, 0.0), (0.0, 0.0), ceiling_resistance - lung.resistance)
        else:
            super().__init__(lung, pressure, pmus, resistance - lung.resistance + series_resistance)
        self.p0, self.p1 = pressure
        self.rs = series_resistance


class PatientEffort:
    """Inspiratory muscle pressure with gradual relaxation and support unloading.

    Ti/Ttot is about 0.4 in quiet breathing (Tobin 1983). Opioids slow the rate
    mainly by lengthening expiration, so Ti stops growing at 1.6 s. Unassisted,
    the muscles draw the half-sine flow of quiet breathing, then relax over half
    of Ti. Delivered assistance shortens and weakens contraction; the unloading
    fraction is a teaching approximation based on delivered pressure during
    the previous contraction relative to the unassisted elastic pressure.

    Inflation feedback strengthens with unconsciousness. Its awake threshold
    is twice resting VT (Clark 1972); under anesthesia, reaching resting VT
    ends contraction (Polacheck 1980). A breath due while a mechanical breath
    keeps the lungs inflated can wait up to one period for deflation, allowing
    synchronization under anesthesia (Graves 1986).

    Fixed knots approximate the curve with linear drives, retaining exact
    mechanics integration and the same profile at every outer step size.
    """

    TI_FRACTION = 0.4
    TI_MAX = 1.6  # s
    RELEASE_FRACTION = 0.5
    PROFILE_POINTS = 64
    ONSET_FRACTION = 0.1  # Of VT above the lowest volume since the last breath, holding off the next
    AWAKE_INFLATION_VT = 2.0

    def __init__(self):
        self.rr = 0.0
        self.target_vt = 0.0  # L
        self.amplitude = 0.0  # cmH2O
        self.clock = 0.0  # s since this breath began
        self.period = math.inf
        self.ti = self.release = 0.0
        self.support_pressure = 0.0  # Mean delivered pressure above PEEP during the previous contraction
        self._assistance_area = self._assistance_time = 0.0
        self.unconscious = 0.0  # 0 awake, 1 unconscious; interpolates inflation feedback
        self._waiting = False  # The next breath is due, but the lungs are still inflated
        self._wait_end = math.inf  # Clock time at which it begins regardless
        self._trough = 0.0  # L, lowest lung volume since this breath began
        self._reference_shift = 0.0
        self._release_slope = 0.0
        self._knots = (math.inf,)
        self._values = (0.0, 0.0)

    def set_drive(self, rr: float, vt_l: float) -> None:
        """Rate and unassisted volume for the breaths that follow."""
        self.rr, self.target_vt = max(0.0, rr), max(0.0, vt_l)

    def note_assistance(self, dt: float, pressure_area: float) -> None:
        """Accumulate delivered airway pressure above PEEP during contraction."""
        self._assistance_area += pressure_area
        self._assistance_time += dt

    def time_to_break(self) -> float:
        """Seconds to the next change of slope, where segments must split."""
        if self._waiting:
            return self._wait_end - self.clock
        i = bisect_right(self._knots, self.clock + 1e-12)
        return self._knots[i] - self.clock if i < len(self._knots) else math.inf

    def pmus(self) -> tuple[float, float]:
        """(value, slope) at the current time until the next breakpoint."""
        value, slope = self._unit_drive(self.clock)
        return self.amplitude * value, self.amplitude * slope

    def advance(self, dt: float, lung: RespiratoryMechanics, series_resistance: float) -> None:
        self.clock += dt
        if self._waiting:
            if self.clock >= self._wait_end - 1e-12:
                self.resume(lung, series_resistance)
        elif self.clock >= self.period - 1e-12:
            if self.rr > 0.0 and lung.volume > self._onset_threshold():
                self._waiting, self._wait_end = True, self.clock + self.period
            else:
                self._begin(lung, series_resistance)

    def start(self, lung: RespiratoryMechanics, series_resistance: float) -> None:
        """Begin a breath now if none is in progress, as when breathing resumes after apnea."""
        if self.period == math.inf:
            self._begin(lung, series_resistance)

    def off_switch_volume(self) -> float | None:
        """Inflation above relaxed volume that ends contraction, with stronger feedback under anesthesia."""
        fraction = self.AWAKE_INFLATION_VT + (1.0 - self.AWAKE_INFLATION_VT) * self.unconscious
        return (fraction * self.target_vt + self._reference_shift
                if self.amplitude > 0.0 and self.clock < self.ti - 1e-12 else None)

    def shift_volume_reference(self, shift: float) -> None:
        self._trough += shift
        self._reference_shift += shift

    def end_contraction(self) -> None:
        """Relax from the present pressure."""
        value = self._unit_drive(self.clock)[0]
        kept = bisect_right(self._knots, self.clock - 1e-12)
        self.ti = self.clock
        self._knots = self._knots[:kept] + (self.clock,)
        self._values = self._values[:kept + 1] + (value,)
        self._relax(value)

    def onset_volume(self) -> float | None:
        """Lung volume (L above relaxed volume) below which a waiting breath begins."""
        return self._onset_threshold() if self._waiting else None

    def note_volume(self, volume: float) -> None:
        """Record the lowest lung volume reached in the latest interval."""
        self._trough = min(self._trough, volume)

    def resume(self, lung: RespiratoryMechanics, series_resistance: float) -> None:
        self._waiting = False
        self._begin(lung, series_resistance)

    def _onset_threshold(self) -> float:
        # Filling to a new relaxed volume after PEEP rises is not inflation above it.
        fraction = self.AWAKE_INFLATION_VT + (self.ONSET_FRACTION - self.AWAKE_INFLATION_VT) * self.unconscious
        return max(self._trough, self._reference_shift) + fraction * self.target_vt

    def _unit_drive(self, t: float) -> tuple[float, float]:
        i = bisect_right(self._knots, t + 1e-12)
        if i >= len(self._knots) or self._knots[i] == math.inf:
            return 0.0, 0.0
        start = self._knots[i - 1] if i else 0.0
        slope = (self._values[i + 1] - self._values[i]) / (self._knots[i] - start)
        return self._values[i] + slope * (t - start), slope

    def _begin(self, lung: RespiratoryMechanics, series_resistance: float) -> None:
        self.support_pressure = (max(0.0, self._assistance_area / self._assistance_time)
                                 if self._assistance_time > 0.0 else 0.0)
        self._assistance_area = self._assistance_time = 0.0
        if self.rr <= 0.0 or self.target_vt <= 0.0:
            self.clock, self.period, self.amplitude = 0.0, math.inf, 0.0
            self.ti = self.release = 0.0
            self._knots, self._values = (math.inf,), (0.0, 0.0)
            self.support_pressure = 0.0
            return
        self.clock = 0.0
        self.period = 60.0 / self.rr
        lung.note_end_expiration(lung.peep)
        self._trough = lung.volume
        self._reference_shift = 0.0
        recoil = self.target_vt / lung.compliance
        unloading = recoil / (recoil + self.support_pressure)
        self.ti = min(self.TI_FRACTION * self.period, self.TI_MAX) * math.sqrt(unloading)
        self.release = self.RELEASE_FRACTION * self.ti
        n = self.PROFILE_POINTS
        rising = [self._sine_flow_pressure(lung, series_resistance, self.ti * i / n) for i in range(n + 1)]
        peak = max(rising)
        self._knots = tuple(self.ti * i / n for i in range(1, n + 1))
        self._values = tuple(value / peak for value in rising)
        # Relaxation continues the final slope, so flow reverses smoothly.
        self._release_slope = max(-3.0, n * self.RELEASE_FRACTION * (rising[-1] - rising[-2]) / rising[-1])
        self._relax(self._values[-1])
        self.amplitude = unloading * self.target_vt / self._unit_volume(lung, series_resistance)

    def _relax(self, value: float) -> None:
        """Append relaxation from a unit value at Ti to zero over the release time.

        A cubic Hermite with the breath's release slope, which levels off at zero
        and stays positive for normalized slopes above -3.
        """
        n, m = self.PROFILE_POINTS, self._release_slope
        steps = [i / n for i in range(1, n + 1)]
        self._knots += tuple(self.ti + self.release * s for s in steps) + (self.period,)
        self._values += tuple(value * (1.0 - s) ** 2 * (1.0 + (2.0 + m) * s) for s in steps) + (0.0,)

    def _sine_flow_pressure(self, lung: RespiratoryMechanics, series_resistance: float, t: float) -> float:
        """Muscle pressure that draws a unit volume as half-sine flow over Ti, starting at rest."""
        w = math.pi / self.ti
        flow = 0.5 * w * math.sin(w * t)
        volume = 0.5 * (1.0 - math.cos(w * t))
        elastance = 1.0 / lung.compliance
        rate = 1.0 / lung.viscoelastic_tau
        # Tissue stress solves dP2/dt = E2 x flow - P2/tau2 from P2 = 0.
        gain = lung.viscoelastic_ratio * elastance * 0.5 * w / (rate * rate + w * w)
        p2 = gain * (rate * math.sin(w * t) - w * math.cos(w * t) + w * math.exp(-rate * t))
        return (lung.resistance + series_resistance) * flow + elastance * volume + p2

    def _unit_volume(self, lung: RespiratoryMechanics, series_resistance: float) -> float:
        """Peak volume from rest for a unit amplitude, breathing at the baseline pressure."""
        scratch = RespiratoryMechanics(lung.compliance, lung.resistance)
        scratch.viscoelastic_ratio, scratch.viscoelastic_tau = lung.viscoelastic_ratio, lung.viscoelastic_tau
        start = 0.0
        for end in self._knots:
            duration = end - start
            pmus = self._unit_drive(start)
            segment = scratch.pressure_segment((0.0, 0.0), pmus, series_resistance)
            if segment.flow(0.0) > 0.0 >= segment.flow(duration):
                return segment.volume(find_root(segment.flow, 0.0, duration))
            segment.commit(duration)
            start = end
        return scratch.volume


def find_root(f, a: float, b: float, fa: float | None = None, fb: float | None = None) -> float:
    """Return t in [a, b] where f changes sign (Illinois false position)."""
    fa = f(a) if fa is None else fa
    fb = f(b) if fb is None else fb
    if fa == 0.0:
        return a
    side = 0
    for _ in range(100):
        if fb == fa:
            break
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        if fc * fb > 0.0:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        elif fc * fa > 0.0:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
        else:
            return c
        if abs(b - a) < 1e-13:
            break
    return b
