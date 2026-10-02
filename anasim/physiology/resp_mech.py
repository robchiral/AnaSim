"""Respiratory system mechanics and inspiratory muscle effort."""

import math


class RespiratoryMechanics:
    """Equation of motion: Paw + Pmus = R x flow + V/C + P2, pressures above PEEP.

    P2 is tissue stress adaptation, a spring in series with a dashpot that lies
    parallel to the static elastance: dP2/dt = E2 x flow - P2/tau2. Volume is
    measured from the relaxed volume at the set PEEP. Segments are integrated
    exactly, so results do not depend on step size.
    """

    def __init__(self, compliance: float = 0.05, resistance: float = 10.0):
        """Static compliance in L/cmH2O; airway and tube resistance in cmH2O/(L/s)."""
        self.compliance = compliance
        self.resistance = resistance
        # Healthy anesthetized adults: viscoelastic compliance 4x static
        # compliance, time constant 0.82 s (Jonson 1993).
        self.viscoelastic_ratio = 0.25  # E2 / static elastance
        self.viscoelastic_tau = 0.82  # s
        self.volume = 0.0  # L
        self.p2 = 0.0  # cmH2O
        self.effort = PatientEffort()

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
                 "start", "last_t", "last")

    def __init__(self, lung: RespiratoryMechanics, pressure, pmus, series_resistance):
        self.lung = lung
        self.v0, self.p20 = lung.volume, lung.p2
        self.p0, self.p1 = pressure
        self.d0, self.d1 = self.p0 + pmus[0], self.p1 + pmus[1]
        self.rs = series_resistance
        e = self.elastance = 1.0 / lung.compliance
        tau = lung.viscoelastic_tau
        e2 = lung.viscoelastic_ratio * e
        rt = lung.resistance + series_resistance
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

    def commit(self, t: float) -> None:
        self.lung.volume, self.lung.p2 = self.state(t)


class PatientEffort:
    """Inspiratory muscle pressure: a linear rise over the neural inspiratory time, then release.

    Ti/Ttot is about 0.4 in quiet breathing (Tobin 1983). Opioids slow the rate
    mainly by lengthening expiration, so Ti stops growing at 1.6 s. The muscles
    relax over a third of Ti. The amplitude gives the patient's unassisted
    tidal volume against the current mechanics.
    """

    TI_FRACTION = 0.4
    TI_MAX = 1.6  # s
    RELEASE_FRACTION = 1.0 / 3.0

    def __init__(self):
        self.rr = 0.0
        self.target_vt = 0.0  # L
        self.amplitude = 0.0  # cmH2O
        self.clock = 0.0  # s since this breath began
        self.period = math.inf
        self.ti = self.release = 0.0

    def set_drive(self, rr: float, vt_l: float) -> None:
        """Rate and unassisted volume for the breaths that follow."""
        self.rr, self.target_vt = max(0.0, rr), max(0.0, vt_l)

    def _breakpoints(self):
        return (self.ti, self.ti + self.release, self.period)

    def time_to_break(self) -> float:
        """Seconds to the next change of slope, where segments must split."""
        return min((b - self.clock for b in self._breakpoints() if b > self.clock + 1e-12), default=math.inf)

    def pmus(self) -> tuple[float, float]:
        """(value, slope) at the current time until the next breakpoint."""
        a, t = self.amplitude, self.clock
        if self.ti > 0.0 and t < self.ti:
            return a * t / self.ti, a / self.ti
        if self.release > 0.0 and t < self.ti + self.release:
            return a * (1.0 - (t - self.ti) / self.release), -a / self.release
        return 0.0, 0.0

    def advance(self, dt: float, lung: RespiratoryMechanics, series_resistance: float) -> None:
        self.clock += dt
        if self.clock >= self.period - 1e-12:
            self._begin(lung, series_resistance)

    def _begin(self, lung: RespiratoryMechanics, series_resistance: float) -> None:
        if self.rr <= 0.0 or self.target_vt <= 0.0:
            self.clock, self.period, self.amplitude = 0.0, math.inf, 0.0
            self.ti = self.release = 0.0
            return
        self.clock = 0.0
        self.period = 60.0 / self.rr
        self.ti = min(self.TI_FRACTION * self.period, self.TI_MAX)
        self.release = self.RELEASE_FRACTION * self.ti
        self.amplitude = self.target_vt / self._unit_volume(lung, series_resistance)

    def start(self, lung: RespiratoryMechanics, series_resistance: float) -> None:
        """Begin a breath now if none is in progress, as when breathing resumes after apnea."""
        if self.period == math.inf:
            self._begin(lung, series_resistance)

    def _unit_volume(self, lung: RespiratoryMechanics, series_resistance: float) -> float:
        """Peak volume from rest for a unit amplitude, breathing at the baseline pressure."""
        scratch = RespiratoryMechanics(lung.compliance, lung.resistance)
        scratch.viscoelastic_ratio, scratch.viscoelastic_tau = lung.viscoelastic_ratio, lung.viscoelastic_tau
        for duration, pmus in ((self.ti, (0.0, 1.0 / self.ti)), (self.release, (1.0, -1.0 / self.release)),
                               (10.0 * self.period, (0.0, 0.0))):
            segment = scratch.pressure_segment((0.0, 0.0), pmus, series_resistance)
            if segment.flow(duration) <= 0.0:
                return segment.volume(find_root(segment.flow, 0.0, duration))
            segment.commit(duration)
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
