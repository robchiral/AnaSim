import math


class AirwaySensor:
    """Airway pressure and flow as the ventilator display shows them.

    Two first-order lags stand in for ventilator and transducer response, so
    step changes rise over about 70 ms. The time constant gave the smallest
    transition error against Dräger Primus recordings. The lags preserve area,
    so the flow trace still integrates to the delivered volume.
    """

    def __init__(self, tau_s: float = 0.02):
        self.tau_s = tau_s
        self.seed(0.0, 0.0)

    def seed(self, paw: float, flow: float) -> None:
        self._stage1 = [paw, flow]
        self._stage2 = [paw, flow]

    def step(self, dt: float, paw: float, flow: float) -> tuple[float, float]:
        """Return measured (Paw cmH2O, flow L/min) after dt seconds."""
        alpha = -math.expm1(-dt / self.tau_s)
        for stage, source in ((self._stage1, (paw, flow)), (self._stage2, self._stage1)):
            stage[0] += alpha * (source[0] - stage[0])
            stage[1] += alpha * (source[1] - stage[1])
        return self._stage2[0], self._stage2[1]
