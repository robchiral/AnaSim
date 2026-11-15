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
            self._push(self._column, exhaled, pco2, right=True)
            for volume, slug_co2 in self._take(self._column, exhaled, left=True):
                self._push(self._window, volume, slug_co2, right=True)
                self._window_volume += volume
                self._window_co2 += volume * slug_co2
            excess = self._window_volume - self.PHASE_II_VOLUME
            if excess > 0.0:
                for volume, slug_co2 in self._take(self._window, excess, left=True):
                    self._window_volume -= volume
                    self._window_co2 -= volume * slug_co2
            sample = self._window_co2 / self._window_volume if self._window_volume > 0.0 else 0.0
        elif active_flow:
            inhaled = -exhaled
            self._inhaled += inhaled
            # Gas moves into the alveoli and fresh gas refills the column from the Y-piece.
            moved = sum(part for part, _ in self._take(self._column, inhaled, left=False))
            self._push(self._column, moved, 0.0, right=False)
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

    @staticmethod
    def _take(column: deque, volume: float, left: bool) -> list:
        """Remove volume from one end and return the removed slugs in order."""
        taken = []
        while volume > 1e-12 and column:
            slug = column[0] if left else column[-1]
            part = min(volume, slug[0])
            taken.append((part, slug[1]))
            slug[0] -= part
            volume -= part
            if slug[0] <= 1e-12:
                column.popleft() if left else column.pop()
        return taken


class EtCO2Readout:
    """End-tidal CO2 numeric from the peak of each completed exhalation.

    The value holds between breaths and blanks once no breath has completed
    for the timeout, or when the sampling line is disconnected.
    """

    def __init__(self, timeout_s: float = 15.0):
        self.timeout_s = timeout_s
        self.value = 0.0  # mmHg
        self._peak = 0.0
        self._age_s = 0.0
        self._exhaling = True
        self._has_sample = False
        self._ever_sampled = False
        self._latched = False
        self._last_breath_index = 0

    @property
    def valid(self) -> bool:
        return self._has_sample and self._age_s <= self.timeout_s

    @property
    def awaiting_first_breath(self) -> bool:
        return not self._ever_sampled and not self._has_sample and self._age_s < self.timeout_s

    def seed(self, value: float) -> None:
        self.value = value
        self._age_s = 0.0
        self._peak = 0.0
        self._has_sample = value > 0.0
        self._ever_sampled = self._has_sample
        self._latched = False

    def interrupt(self, dt: float, disconnected: bool, *, idle_airway: bool = False) -> None:
        """Advance without exhaled gas at the sampling port."""
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
        self._expire()

    def update(
        self,
        dt: float,
        exhaling: bool,
        co2: float,
        *,
        inhaled_l: float = 0.0,
        dead_space_l: float = 0.0,
        breath_index: int | None = None,
    ) -> None:
        """Advance with one capnograph sample; ignore mid-expiration curare clefts."""
        self._age_s += dt
        if breath_index is not None and breath_index != self._last_breath_index:
            if self._peak > 1.0 and not self._latched:
                self.value = self._peak
                self._age_s = 0.0
                self._has_sample = True
                self._ever_sampled = True
            self._last_breath_index = breath_index
            self._peak = 0.0
            self._latched = False
        new_inspiration = not exhaling and dead_space_l > 0.0 and inhaled_l > dead_space_l
        if exhaling and co2 > self._peak:
            self._peak = co2
            self._latched = False
        end_pause = self._exhaling and not exhaling and inhaled_l <= 0.0
        if (end_pause or new_inspiration) and self._peak > 1.0 and not self._latched:
            self.value = self._peak
            self._age_s = 0.0
            self._has_sample = True
            self._ever_sampled = True
            self._latched = True
        if new_inspiration or (end_pause and dead_space_l <= 0.0):
            self._peak = 0.0
            self._latched = False
        self._exhaling = exhaling
        self._expire()

    def _expire(self) -> None:
        if not self.valid:
            self.value = 0.0
