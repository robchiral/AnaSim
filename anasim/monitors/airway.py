import math


class AirwaySensor:
    """Airway pressure, flow, and volume as the ventilator display shows them.

    Two first-order lags fit the roughly 70 ms response in Dräger Primus
    recordings. They preserve area, so flow integrates to delivered volume.
    """

    def __init__(self, tau_s: float = 0.02):
        self.tau_s = tau_s
        self.seed(0.0, 0.0, 0.0)

    def seed(self, paw: float, flow: float, volume: float) -> None:
        self._stage1 = [paw, flow, volume]
        self._stage2 = [paw, flow, volume]

    def step(self, dt: float, paw: float, volume: float, volume_change: float) -> tuple[float, float, float]:
        """Filter pressure, interval-mean flow, and a volume ramp with the same gas movement."""
        z = dt / self.tau_s
        decay = math.exp(-z)
        flow = 60.0 * volume_change / dt
        for i, source in enumerate((paw, flow)):
            first, second = self._stage1[i], self._stage2[i]
            self._stage1[i] = source + (first - source) * decay
            self._stage2[i] = source + (second - source + z * (first - source)) * decay
        # Volume is the integral of the interval-mean flow. Filter its ramp
        # exactly rather than holding the end volume through the whole interval.
        first, second = self._stage1[2], self._stage2[2]
        start, slope = volume - volume_change, volume_change / dt
        ramp = slope * self.tau_s
        self._stage1[2] = volume - ramp + (first - start + ramp) * decay
        self._stage2[2] = volume - 2.0 * ramp + (second - start + 2.0 * ramp
                                              + z * (first - start + ramp)) * decay
        paw, flow, volume = self._stage2
        return paw, flow, volume
