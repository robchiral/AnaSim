THRESHOLDS = {
    "BIS_min": 20, "BIS_max": 70,
    "MAP_min": 60, "MAP_max": 110,
    "HR_min": 45, "HR_max": 120,
    "SpO2_min": 90, "SpO2_max": 100,
    "EtCO2_min": 30, "EtCO2_max": 45,
    "RR_min": 5,
    "Temp_min": 35.5, "Temp_max": 38.0,
    # Ventilator and gas monitor: cmH2O, L/min, and %.
    "Ppeak_max": 40,
    "MV_min": 2.0,
    "VTe_min": 90,  # % of the set VT in volume-targeted modes
    "FiO2_min": 21,
}

# Seconds a limit must stay violated before the alarm sounds. The MV delay
# covers the first breath after the ventilator starts; the VTe delay spans
# a few breaths.
DEFAULT_DELAYS = {
    "BIS": 0,
    "MAP": 0,
    "HR": 0,
    "SpO2": 5,
    "EtCO2": 0,
    "RR": 15,
    "Temp": 0,
    "Ppeak": 0,
    "MV": 15,
    "VTe": 10,
    "FiO2": 0,
}


_KEY_CACHE: dict[str, tuple[str, str, tuple[str, str], tuple[str, str]]] = {
    name: (f"{name}_min", f"{name}_max", (name, "low"), (name, "high"))
    for name in DEFAULT_DELAYS
}


class AlarmSystem:
    """Threshold alarms that sound once a limit violation persists for its delay."""

    def __init__(self, delays: dict | None = None, thresholds: dict | None = None):
        self.delays = delays or dict(DEFAULT_DELAYS)
        self.thresholds = {**THRESHOLDS, **(thresholds or {})}
        self._violation_s: dict[tuple[str, str], float] = {}

    def update(self, values: dict, dt: float) -> dict:
        """Update with values held for dt seconds, e.g. {'HR': 60, 'MAP': 80}; return active alarms."""
        alarms = {}
        thresholds = self.thresholds
        violations = self._violation_s
        for name, delay_s in self.delays.items():
            keys = _KEY_CACHE.get(name)
            if keys is None:
                keys = (f"{name}_min", f"{name}_max", (name, "low"), (name, "high"))
                _KEY_CACHE[name] = keys
            min_key, max_key, low_key, high_key = keys
            value = values.get(name)
            if value is None:
                if violations:
                    violations.pop(low_key, None)
                    violations.pop(high_key, None)
                continue
            limit_low = thresholds.get(min_key)
            limit_high = thresholds.get(max_key)
            low_violated = limit_low is not None and value < limit_low
            high_violated = limit_high is not None and value > limit_high
            if low_violated:
                low_elapsed = violations.get(low_key, 0.0) + dt
                violations[low_key] = low_elapsed
                low_active = low_elapsed >= delay_s - 1e-9
            else:
                if low_key in violations:
                    del violations[low_key]
                low_active = False
            if high_violated:
                high_elapsed = violations.get(high_key, 0.0) + dt
                violations[high_key] = high_elapsed
                high_active = high_elapsed >= delay_s - 1e-9
            else:
                if high_key in violations:
                    del violations[high_key]
                high_active = False
            if low_active or high_active:
                alarms[name] = {"low": low_active, "high": high_active}
        return alarms
