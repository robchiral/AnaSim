"""Sidestream capnography at the Y-piece."""

import math
from collections import deque


class Capnograph:
    """CO2 at the Y-piece after dead-space washout and analyzer lag.

    Exhalation displaces inspired gas, then alveolar gas with a phase III slope.
    Unequal airway paths spread phase II around the Fowler dead space.
    Inspiration draws fresh gas past the sampling port.
    """

    # Fitted to Dräger Primus recordings; see docs/REFERENCES.md#ventilator-waveforms.
    ANALYZER_TAU = 0.12  # s, each of two first-order stages
    PHASE_II_VOLUME = 0.10  # L, spread of path dead spaces
    PHASE_III_SLOPE = 20.0  # mmHg/L
    MIN_FLOW_L_S = 0.002  # L/s (0.12 L/min); analyzer no-flow floor
    PASSIVE_CREEP_L_S = 0.015  # L/s (0.9 L/min); post-expiratory apneic drift

    def __init__(self, dead_space_l: float):
        self.dead_space = dead_space_l
        self.reset()

    def reset(self) -> None:
        """Discard the airway gas and analyzer history when exhaled gas is unavailable."""
        # (volume L, PCO2 mmHg) slugs from the Y-piece inward; the column is
        # shorter than the dead space by half the phase II spread.
        self._column = deque([[max(1e-3, self.dead_space - 0.5 * self.PHASE_II_VOLUME), 0.0]])
        self._window: deque[list[float]] = deque()  # Recently exhaled gas, Y-piece end first
        self._window_volume = self._window_co2 = 0.0  # L and L x mmHg
        self._exhaled_alveolar = 0.0
        self._breath_volume = 0.0  # Inspired volume of the current breath, L.
        self._inhaled = 0.0
        self.breath_index = 0
        self._no_flow_s = 0.0
        self._stages = [0.0, 0.0]
        self.co2 = 0.0
        self.exhaling = False

    def step(self, dt: float, volume_change: float, end_tidal: float, obstruction: float = 0.0) -> float:
        """Advance by dt with net lung volume change volume_change (L); return displayed PCO2.

        end_tidal is the PCO2 reached by the current breath. Obstruction (0-1)
        steepens phase III; previous breath sizes do not change its endpoint.
        """
        exhaled = -volume_change
        passive_creep = (
            0.0 < exhaled <= self.PASSIVE_CREEP_L_S * dt
            and self._inhaled <= self.dead_space
            and self._exhaled_alveolar >= self._breath_volume
        )
        flow_floor = self.PASSIVE_CREEP_L_S * dt if passive_creep else self.MIN_FLOW_L_S * dt
        active_flow = abs(exhaled) > flow_floor
        self.exhaling = active_flow and exhaled > 0.0
        if active_flow:
            self._no_flow_s = 0.0
        else:
            self._no_flow_s += dt
        if self.exhaling:
            if self._inhaled > self.dead_space:
                # Fresh gas reached the alveoli, so this is a new exhalation.
                self._breath_volume = self._inhaled
                self._exhaled_alveolar = 0.0
                self.breath_index += 1
            else:
                # A small effort interrupts this expiration. Its returning gas
                # must not advance the alveolar concentration profile twice.
                self._exhaled_alveolar = max(0.0, self._exhaled_alveolar - self._inhaled)
            self._inhaled = 0.0
            slope = self.PHASE_III_SLOPE * (1.0 + 3.0 * obstruction)
            # Allow for transit through dead space, using this inspiration's
            # volume. Bound the source concentration at the current endpoint.
            leaving = self._exhaled_alveolar + 0.5 * exhaled
            remaining = max(0.0, self._breath_volume - leaving - self.dead_space)
            pco2 = max(0.0, end_tidal - slope * remaining)
            self._exhaled_alveolar += exhaled
            col = self._column
            win = self._window
            self._push(col, exhaled, pco2, right=True)
            rem = exhaled
            while rem > 1e-12 and col:
                slug = col[0]
                part = rem if rem < slug[0] else slug[0]
                slug_co2 = slug[1]
                self._push(win, part, slug_co2, right=True)
                self._window_volume += part
                self._window_co2 += part * slug_co2
                slug[0] -= part
                rem -= part
                if slug[0] <= 1e-12:
                    col.popleft()
            excess = self._window_volume - self.PHASE_II_VOLUME
            if excess > 0.0:
                while excess > 1e-12 and win:
                    slug = win[0]
                    part = excess if excess < slug[0] else slug[0]
                    self._window_volume -= part
                    self._window_co2 -= part * slug[1]
                    slug[0] -= part
                    excess -= part
                    if slug[0] <= 1e-12:
                        win.popleft()
            sample = self._window_co2 / self._window_volume if self._window_volume > 0.0 else 0.0
        elif active_flow:
            inhaled = -exhaled
            self._inhaled += inhaled
            # Gas moves into the alveoli and fresh gas refills the column from the Y-piece.
            col = self._column
            if not (len(col) == 1 and col[0][1] == 0.0):
                rem = inhaled
                moved = 0.0
                while rem > 1e-12 and col:
                    slug = col[-1]
                    part = rem if rem < slug[0] else slug[0]
                    moved += part
                    slug[0] -= part
                    rem -= part
                    if slug[0] <= 1e-12:
                        col.pop()
                self._push(col, moved, 0.0, right=False)
            if self._window:
                self._window.clear()
            self._window_volume = self._window_co2 = 0.0
            sample = 0.0
        else:
            if self._no_flow_s > 1.5:
                self._decay_window(math.exp(-dt / 1.8))
                sample = self._window_co2 / self._window_volume if self._window_volume > 0.0 else 0.0
            else:
                sample = self._window_co2 / self._window_volume if self._window_volume > 0.0 else self._stages[0]
        # Exact coupled response of two identical first-order stages to a
        # constant sample, using both stages' values at the interval start.
        z = dt / self.ANALYZER_TAU
        decay = math.exp(-z)
        first, second = self._stages
        self._stages[0] = sample + (first - sample) * decay
        self._stages[1] = sample + (second - sample + z * (first - sample)) * decay
        if self._no_flow_s > 1.5 and self._stages[1] < 0.25 and sample < 0.25:
            self._decay_window(0.0)
            self._stages[0] = self._stages[1] = 0.0
        self.co2 = self._stages[1]
        return self.co2

    def _decay_window(self, factor: float) -> None:
        self._window_co2 = 0.0
        for slug in self._window:
            slug[1] *= factor
            self._window_co2 += slug[0] * slug[1]

    @staticmethod
    def _push(column: deque, volume: float, pco2: float, right: bool) -> None:
        end = column[-1] if right and column else column[0] if column else None
        if end is not None and abs(end[1] - pco2) < 0.05:
            end[0] += volume
        elif right:
            column.append([volume, pco2])
        else:
            column.appendleft([volume, pco2])


class EtCO2Readout:
    """End-tidal CO2 numeric from the peak of each completed exhalation.

    The value holds between breaths and blanks once no breath has completed
    for the timeout, or when the sampling line is disconnected.
    """

    def __init__(self, timeout_s: float = 15.0, peak_window_s: float = 11.0):
        self.timeout_s = timeout_s
        self.peak_window_s = peak_window_s
        self.value = 0.0  # mmHg
        self.et_gases: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._peak = 0.0
        self._peak_gases: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._age_s = 0.0
        self._clock = 0.0
        self._last_interval_s = 0.0
        self._effective_timeout_s = timeout_s
        self._recent_peaks: deque[tuple[float, float, tuple[float, float, float, float]]] = deque()
        self._exhaling = True
        self._has_sample = False
        self._ever_sampled = False
        self._latched = False
        self._last_breath_index = 0

    @property
    def valid(self) -> bool:
        return self._has_sample and self._age_s <= self._effective_timeout_s

    @property
    def awaiting_first_breath(self) -> bool:
        return not self._ever_sampled and not self._has_sample and self._age_s < self._effective_timeout_s

    def seed(self, value: float, *, et_gases: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        self.value = value
        self._age_s = 0.0
        self._clock = 0.0
        self._last_interval_s = 0.0
        self._effective_timeout_s = self.timeout_s
        self._peak = 0.0
        self._has_sample = value > 0.0
        self._ever_sampled = self._has_sample
        self._latched = False
        self.et_gases = et_gases if self._has_sample else (0.0, 0.0, 0.0, 0.0)
        self._peak_gases = self.et_gases
        self._recent_peaks.clear()
        if self._has_sample:
            self._recent_peaks.append((0.0, value, self.et_gases))

    def interrupt(self, dt: float, disconnected: bool, *, idle_airway: bool = False) -> None:
        """Advance without exhaled gas at the sampling port."""
        self._clock += dt
        self._peak = 0.0
        self._exhaling = False
        self._latched = False
        if disconnected:
            self._has_sample = False
        if idle_airway:
            self._age_s = 0.0
            self._ever_sampled = False
        else:
            self._age_s += dt
        if disconnected or idle_airway:
            self._recent_peaks.clear()
            self._last_interval_s = 0.0
            self._effective_timeout_s = self.timeout_s
        self._expire()

    def _latch_peak(self) -> None:
        if self._has_sample and self._age_s > 0.5:
            self._last_interval_s = min(25.0, self._age_s)
        self._recent_peaks.append((self._clock, self._peak, self._peak_gases))
        while len(self._recent_peaks) > 1 and self._clock - self._recent_peaks[0][0] > self.peak_window_s:
            self._recent_peaks.popleft()
        best = max(self._recent_peaks, key=lambda entry: entry[1])
        # Track the newest valid breath unless it is a shallow interposed breath
        # (<70% of a recent full breath in the rolling window).
        if self._peak >= 0.70 * best[1]:
            best = self._recent_peaks[-1]
        self.value = best[1]
        self.et_gases = best[2]
        self._age_s = 0.0
        self._has_sample = True
        self._ever_sampled = True

    def update(
        self,
        dt: float,
        exhaling: bool,
        co2: float,
        *,
        inhaled_l: float = 0.0,
        dead_space_l: float = 0.0,
        breath_index: int | None = None,
        et_gases: tuple[float, float, float, float] | None = None,
        expected_period_s: float | None = None,
    ) -> None:
        """Advance with one capnograph sample; ignore mid-expiration curare clefts."""
        self._clock += dt
        self._age_s += dt
        period_hint = (
            min(25.0, expected_period_s)
            if expected_period_s is not None and math.isfinite(expected_period_s) and expected_period_s > 0.0
            else self._last_interval_s
        )
        self._effective_timeout_s = (
            max(self.timeout_s, min(32.0, 1.25 * period_hint))
            if period_hint > 0.0
            else self.timeout_s
        )
        if breath_index is not None and breath_index != self._last_breath_index:
            if self._peak > 1.0 and not self._latched:
                self._latch_peak()
            self._last_breath_index = breath_index
            self._peak = 0.0
            self._latched = False
        new_inspiration = not exhaling and dead_space_l > 0.0 and inhaled_l > dead_space_l
        if exhaling and co2 > self._peak:
            self._peak = co2
            if et_gases is not None:
                self._peak_gases = et_gases
                if self._has_sample and co2 > 1.0 and co2 >= self.value - 1.0:
                    self.et_gases = et_gases
            self._latched = False
        end_pause = self._exhaling and not exhaling and inhaled_l <= 0.0
        if (end_pause or new_inspiration) and self._peak > 1.0 and not self._latched:
            self._latch_peak()
            self._latched = True
        if new_inspiration or (end_pause and dead_space_l <= 0.0):
            self._peak = 0.0
            self._latched = False
        self._exhaling = exhaling
        self._expire()

    def _expire(self) -> None:
        if not self.valid:
            self.value = 0.0
            self.et_gases = (0.0, 0.0, 0.0, 0.0)
            self._recent_peaks.clear()
