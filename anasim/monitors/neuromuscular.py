from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

from anasim.core.utils import clamp01

# Simulator signal thresholds, as fractions of the unblocked response.
DETECTION_THRESHOLD = 0.05
LOW_SIGNAL_THRESHOLD = 0.20
MIN_REPEAT_S = 10.0
PULSE_INTERVAL_S = 0.5  # 2 Hz train
DEFAULT_REPEAT_S = {"tof": 20.0, "single": 10.0}


@dataclass(frozen=True)
class TwitchReading:
    time: float
    mode: str
    responses: tuple[float, ...]
    count: int
    ratio: float | None
    t1: float | None
    low_signal: bool


class NeuromuscularMonitor:
    """TOF and single-twitch monitor that samples the modeled adductor pollicis response."""

    def __init__(self):
        self.mode = "tof"
        self.interval = DEFAULT_REPEAT_S["tof"]
        self.automatic = True
        self.baseline: float | None = None
        self.reading: TwitchReading | None = None
        self.history: deque[TwitchReading] = deque(maxlen=90)
        self._last_start = -math.inf
        self._next_start = 0.0
        self._started: float | None = None
        self._responses: list[float] = []

    def establish_baseline(self, t1: float) -> None:
        """Record the reference response only before appreciable block."""
        if t1 >= 0.95:
            self.baseline = t1

    def set_mode(self, mode: str, time: float) -> None:
        if mode not in DEFAULT_REPEAT_S:
            raise ValueError("Choose TOF or single twitch")
        if self._started is not None:
            raise ValueError("Wait for stimulation to finish")
        if mode == self.mode:
            return
        self.mode = mode
        self.interval = DEFAULT_REPEAT_S[mode]
        self.reading = None
        self._next_start = max(time, self._last_start + MIN_REPEAT_S)

    def set_repeat(self, seconds: float, time: float) -> None:
        if seconds not in (0, 10, 20, 30, 60):
            raise ValueError("Repeat must be off, 10, 20, 30, or 60 seconds")
        self.automatic = seconds != 0
        if self.automatic:
            self.interval = seconds
            self._next_start = max(time, self._last_start + seconds)

    def measure(self, time: float) -> None:
        if self._started is not None or time < self._last_start + MIN_REPEAT_S:
            raise ValueError(f"Wait at least {MIN_REPEAT_S:.0f} seconds between stimulations")
        self._begin(time)

    def _begin(self, time: float) -> None:
        self._started = self._last_start = time
        self._responses = []
        self._next_start = time + self.interval

    def step(self, time: float, t1: float, ratio: float) -> None:
        if self._started is None:
            if not self.automatic or time < self._next_start:
                return
            self._begin(time)
        assert self._started is not None
        pulses = 4 if self.mode == "tof" else 1
        while len(self._responses) < pulses:
            index = len(self._responses)
            if time + 1e-9 < self._started + PULSE_INTERVAL_S * index:
                return
            # Geometric interpolation supplies T2/T3 between the modeled T1/T4.
            self._responses.append(clamp01(t1) * clamp01(ratio) ** (index / 3.0))
        self._finish(time)

    def _finish(self, time: float) -> None:
        responses = tuple(self._responses)
        count = sum(value >= DETECTION_THRESHOLD for value in responses)
        complete_train = self.mode == "tof" and count == 4
        self.reading = TwitchReading(
            time, self.mode, responses, count,
            ratio=100.0 * responses[3] / responses[0] if complete_train else None,
            t1=None if self.baseline is None else 100.0 * responses[0] / self.baseline,
            low_signal=complete_train and max(responses) < LOW_SIGNAL_THRESHOLD,
        )
        self.history.append(self.reading)
        self._started = None

    def snapshot(self, time: float) -> dict:
        return {
            "mode": self.mode,
            "repeat": self.interval if self.automatic else 0,
            "baseline": self.baseline,
            "measuring": self._started is not None,
            "pulses": len(self._responses) if self._started is not None else 0,
            "wait": max(0.0, self._last_start + MIN_REPEAT_S - time),
            "next": max(0.0, self._next_start - time) if self.automatic else None,
            "reading": None if self.reading is None else asdict(self.reading),
        }

    def trend_history(self, after: float = -math.inf) -> list[list]:
        """Rows of [time, mode, TOF ratio or T1 %, low signal] newer than `after`."""
        return [
            [r.time, r.mode, r.ratio if r.mode == "tof" else r.t1, r.low_signal]
            for r in self.history if r.time > after
        ]
