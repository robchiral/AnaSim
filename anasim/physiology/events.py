"""Clinical events whose severity develops and resolves over minutes."""

from dataclasses import dataclass


@dataclass(slots=True)
class SeverityRamp:
    """Severity (0-1) that rises while the event is active and falls after it stops.

    Rates are per second.
    """

    onset_rate: float
    decay_rate: float
    active: bool = False
    severity: float = 0.0

    def step(self, dt: float) -> float:
        if self.active:
            self.severity = min(1.0, self.severity + self.onset_rate * dt)
        else:
            self.severity = max(0.0, self.severity - self.decay_rate * dt)
        return self.severity
