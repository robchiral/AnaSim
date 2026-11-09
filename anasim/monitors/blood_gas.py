"""Arterial blood gas samples reported by a point-of-care analyzer."""

from dataclasses import dataclass

ANALYSIS_TIME_S = 120.0


@dataclass(frozen=True, slots=True)
class BloodGasResult:
    """Arterial values at the time of the draw, reported at 37 °C."""

    drawn_at: float
    ph: float
    pa_co2: float  # mmHg
    pao2: float  # mmHg
    hco3: float  # Plasma bicarbonate, mmol/L.
    base_excess: float  # Standard base excess, mmol/L.
    lactate: float  # mmol/L
    sodium: float  # mmol/L
    potassium: float  # mmol/L
    chloride: float  # mmol/L
    ionized_calcium: float  # mmol/L
    sao2: float  # %
    hb: float  # g/dL
    fio2: float  # Inspired O2 fraction at the draw.

    @property
    def ready_at(self) -> float:
        return self.drawn_at + ANALYSIS_TIME_S
