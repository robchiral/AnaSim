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

    def step(self, dt: float, paw: float, flow: float, volume: float) -> tuple[float, float, float]:
        z = dt / self.tau_s
        decay = math.exp(-z)
        for i, source in enumerate((paw, flow, volume)):
            first, second = self._stage1[i], self._stage2[i]
            self._stage1[i] = source + (first - source) * decay
            self._stage2[i] = source + (second - source + z * (first - source)) * decay
        return tuple(self._stage2)
