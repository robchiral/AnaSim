DEFAULT_THRESHOLDS = {
    "BIS_min": 20, "BIS_max": 70,
    "MAP_min": 60, "MAP_max": 110,
    "HR_min": 45, "HR_max": 120,
    "SpO2_min": 90, "SpO2_max": 100,
    "EtCO2_min": 30, "EtCO2_max": 45,
}

# Seconds a limit must stay violated before the alarm sounds.
DEFAULT_DELAYS = {"BIS": 0, "MAP": 0, "HR": 0, "SpO2": 5, "EtCO2": 0}


class AlarmSystem:
    """Threshold alarms that sound once a limit violation persists for its delay."""

    def __init__(self, thresholds: dict = None, delays: dict = None, dt: float = 1.0):
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.delays = delays or dict(DEFAULT_DELAYS)
        self.dt = dt
        self._violation_s: dict[tuple[str, str], float] = {}
        self.active_alarms = {}

    def update(self, values: dict, dt: float = None) -> dict:
        """Update with current values, e.g. {'HR': 60, 'MAP': 80}; return active alarms."""
        if dt is not None and dt > 0:
            self.dt = dt
        alarms = {}
        for name, delay_s in self.delays.items():
            if name not in values:
                continue
            value = values[name]
            limit_low = self.thresholds.get(f"{name}_min")
            limit_high = self.thresholds.get(f"{name}_max")
            flags = {}
            for side, violated in (
                ("low", limit_low is not None and value < limit_low),
                ("high", limit_high is not None and value > limit_high),
            ):
                elapsed = self._violation_s.get((name, side), 0.0) + self.dt if violated else 0.0
                self._violation_s[(name, side)] = elapsed
                flags[side] = violated and elapsed >= delay_s - 1e-9
            if flags["low"] or flags["high"]:
                alarms[name] = flags
        self.active_alarms = alarms
        return alarms
