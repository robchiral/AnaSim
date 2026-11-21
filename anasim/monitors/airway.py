import math


class AirwaySensor:
    """Airway pressure, flow, and volume as the ventilator display shows them.

    Two first-order lags fit the roughly 70 ms response in Dräger Primus
    recordings. They preserve area, so flow integrates to delivered volume.
    """

    def __init__(self, tau_s: float = 0.02):
        self.tau_s = tau_s
        self._last_key = (-1.0, -1.0)
        self._last_z = 0.0
        self._last_decay = 1.0
        self.seed(0.0, 0.0, 0.0)

    def seed(self, paw: float, flow: float, volume: float) -> None:
        self._stage1 = [paw, flow, volume]
        self._stage2 = [paw, flow, volume]

    @property
    def reading(self) -> tuple[float, float, float]:
        paw, flow, volume = self._stage2
        return paw, flow, volume

    def step(self, dt: float, paw: float, volume: float, volume_change: float) -> tuple[float, float, float]:
        """Filter pressure, interval-mean flow, and a volume ramp with the same gas movement."""
        tau_s = self.tau_s
        key = (dt, tau_s)
        if key != self._last_key:
            self._last_key = key
            self._last_z = dt / tau_s
            self._last_decay = math.exp(-self._last_z)
        z = self._last_z
        decay = self._last_decay
        flow = 60.0 * volume_change / dt
        s1 = self._stage1
        s2 = self._stage2
        f0 = s1[0] - paw
        s1[0] = paw + f0 * decay
        s2[0] = paw + (s2[0] - paw + z * f0) * decay
        f1 = s1[1] - flow
        s1[1] = flow + f1 * decay
        s2[1] = flow + (s2[1] - flow + z * f1) * decay
        # Filter the volume ramp exactly from the interval-mean flow.
        first, second = s1[2], s2[2]
        start, slope = volume - volume_change, volume_change / dt
        ramp = slope * tau_s
        diff = first - start + ramp
        s1[2] = volume - ramp + diff * decay
        s2[2] = volume - 2.0 * ramp + (second - start + 2.0 * ramp + z * diff) * decay
        return s2[0], s2[1], s2[2]
