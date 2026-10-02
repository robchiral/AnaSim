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
        alpha = -math.expm1(-dt / self.tau_s)
        for stage, source in ((self._stage1, (paw, flow, volume)), (self._stage2, self._stage1)):
            for i in range(3):
                stage[i] += alpha * (source[i] - stage[i])
        return tuple(self._stage2)
