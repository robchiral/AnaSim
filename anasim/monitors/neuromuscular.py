import math
from collections import deque
from dataclasses import dataclass

from anasim.core.utils import clamp01

# Simulator signal thresholds, as fractions of the unblocked response.
DETECTION_THRESHOLD = 0.05
LOW_SIGNAL_THRESHOLD = 0.20
MIN_REPEAT_S = 10.0
PULSE_INTERVAL_S = 0.5  # 2 Hz train
DEFAULT_REPEAT_S = {"tof": 20.0, "single": 10.0, "ptc": 60.0}
PTC_MAX_COUNT = 15
PTC_MIN_REPEAT_S = 60.0


@dataclass(frozen=True, slots=True)
class TwitchReading:
    time: float
    mode: str
    responses: tuple[float, ...]
    count: int
    ratio: float | None
    t1: float | None
    low_signal: bool

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "mode": self.mode,
            "responses": self.responses,
            "count": self.count,
            "ratio": self.ratio,
            "t1": self.t1,
            "low_signal": self.low_signal,
        }


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
        self._last_mode: str | None = None
        self._next_start = 0.0
        self._started: float | None = None
        self._responses: list[float] = []
        self._train_t1: float = 0.0
        self._train_ratio: float = 1.0
        self._deep_block_ratio: float = 0.0

    def _min_repeat_s(self) -> float:
        """Minimum quiet period after the previous stimulation (Fuchs-Buder 2007)."""
        return PTC_MIN_REPEAT_S if self._last_mode == "ptc" else MIN_REPEAT_S

    def establish_baseline(self, t1: float) -> None:
        """Record the reference response only before appreciable block."""
        if t1 >= 0.95:
            self.baseline = t1

    def set_mode(self, mode: str, time: float) -> None:
        if mode not in DEFAULT_REPEAT_S:
            raise ValueError("Choose TOF, single twitch, or PTC")
        if self._started is not None:
            raise ValueError("Wait for stimulation to finish")
        if mode == self.mode:
            return
        self.mode = mode
        self.interval = DEFAULT_REPEAT_S[mode]
        self.reading = None
        self._next_start = max(time, self._last_start + self._min_repeat_s())

    def set_repeat(self, seconds: float, time: float) -> None:
        if seconds not in (0, 10, 20, 30, 60):
            raise ValueError("Repeat must be off, 10, 20, 30, or 60 seconds")
        self.automatic = seconds != 0
        if self.automatic:
            self.interval = max(PTC_MIN_REPEAT_S, seconds) if self.mode == "ptc" else seconds
            self._next_start = max(time, self._last_start + max(self.interval, self._min_repeat_s()))

    def measure(self, time: float) -> None:
        min_gap = self._min_repeat_s()
        if self._started is not None or time < self._last_start + min_gap:
            raise ValueError(f"Wait at least {min_gap:.0f} seconds between stimulations")
        self._begin(time)

    def _begin(self, time: float) -> None:
        self._started = self._last_start = time
        self._last_mode = self.mode
        self._responses = []
        self._next_start = time + self.interval

    def step(self, time: float, t1: float, ratio: float, *, deep_block_ratio: float = 0.0) -> None:
        time_eps = 1e-9 * max(1.0, abs(time))
        if self._started is None:
            if not self.automatic or time + time_eps < self._next_start:
                return
            self._begin(time)
        assert self._started is not None
        if self.mode == "ptc":
            # 5 s 50 Hz tetanus + 3 s pause + 15 single 1 Hz twitches; complete in 2 s sim time
            # for responsive bedside interaction while enforcing the 60 s PTC repeat interval.
            if not self._responses:
                self._train_t1 = clamp01(t1)
                self._train_ratio = clamp01(ratio)
                self._deep_block_ratio = max(0.0, float(deep_block_ratio))
                self._responses.append(self._train_t1)
            if time + time_eps < self._started + 2.0:
                return
            self._finish(time)
            return
        pulses = 4 if self.mode == "tof" else 1
        while len(self._responses) < pulses:
            index = len(self._responses)
            if time + time_eps < self._started + PULSE_INTERVAL_S * index:
                return
            if index == 0:
                self._train_t1 = clamp01(t1)
                self._train_ratio = clamp01(ratio)
                self._deep_block_ratio = max(0.0, float(deep_block_ratio))
            # Geometric interpolation supplies T2/T3 between the modeled T1/T4.
            self._responses.append(self._train_t1 * self._train_ratio ** (index / 3.0))
        self._finish(time)

    def _finish(self, time: float) -> None:
        if self.mode == "ptc":
            # Post-tetanic count (0-15) reflects deep non-depolarizing receptor occupancy
            # when TOF count is 0 (Viby-Mogensen 1981; Thilen 2023).
            if self._train_t1 >= DETECTION_THRESHOLD:
                ptc = PTC_MAX_COUNT
            elif self._deep_block_ratio > 0.0:
                # Around Ce/EC50 ~ 1.6 (T1 just disappearing) PTC is ~12-15;
                # above ~3.5x EC50 (intense block) PTC reaches 0.
                frac = clamp01((3.5 - self._deep_block_ratio) / 2.0)
                ptc = round(PTC_MAX_COUNT * frac)
            else:
                ptc = 0
            responses = tuple(1.0 if i < ptc else 0.0 for i in range(4))
            self.reading = TwitchReading(
                time, "ptc", responses, ptc,
                ratio=None,
                t1=None if self.baseline is None else 100.0 * self._train_t1 / self.baseline,
                low_signal=False,
            )
            self.history.append(self.reading)
            self._started = None
            return
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
        min_gap = self._min_repeat_s()
        return {
            "mode": self.mode,
            "repeat": self.interval if self.automatic else 0,
            "baseline": self.baseline,
            "measuring": self._started is not None,
            "pulses": len(self._responses) if self._started is not None else 0,
            "wait": max(0.0, self._last_start + min_gap - time),
            "next": max(0.0, self._next_start - time) if self.automatic else None,
            "reading": None if self.reading is None else self.reading.to_dict(),
        }

    def trend_history(self, after: float = -math.inf) -> list[list]:
        """Rows of [time, mode, TOF ratio / T1 % / PTC count, low signal] newer than `after`."""
        if not self.history or self.history[-1].time <= after:
            return []
        return [
            [
                r.time,
                r.mode,
                r.ratio if r.mode == "tof" else (r.count if r.mode == "ptc" else r.t1),
                r.low_signal,
            ]
            for r in self.history if r.time > after
        ]
