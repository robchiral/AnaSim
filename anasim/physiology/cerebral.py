"""Cerebral oxygen failure, persistent coma, and reversible CO2 narcosis.

Pressure below the autoregulatory range and low arterial oxygen content reduce
supply. Electrical function ramps down from 60% to complete failure at 35% of
normal supply, and neurons are injured below 20% (Astrup 1981), more slowly when
more supply remains (Jones 1981). Oxygen stores delay failure: after circulatory
arrest, it starts at about 4 s and the EEG is flat at about 11 s (de Vries
1998, 11 +/- 2 s to isoelectric EEG). Acute electrical suppression can recover
while coma persists.
"""

import math

from anasim.core.utils import clamp01


class CerebralModel:
    # Time without supply after which coma persists, a teaching threshold.
    # Supply of 10-20% counts partially.
    COMA_EXPOSURE_S = 300.0
    TISSUE_OXYGEN_TAU_S = 6.0
    # CSF bicarbonate adapts over 4-12 h of sustained PaCO2 change (Brian 1998);
    # tau = 6 h is a model time constant.
    CSF_ADAPTATION_TAU_S = 6.0 * 3600.0
    # The PaCO2 and pH ramps follow Westlake 1955 (Table III) and Sieker and
    # Hickam 1956, where coma lasting over an hour lags blood-gas correction
    # (p. 397); the 30 s onset and up to 30 min recovery time constants are
    # teaching calibrations.
    CO2_RESPONSE_TAU_S = 30.0
    PROLONGED_NARCOSIS_ONSET_S = 1800.0
    PROLONGED_NARCOSIS_SPAN_S = 5400.0
    CO2_RECOVERY_MAX_TAU_S = 1800.0

    def __init__(self, baseline_hb: float, baseline_paco2: float = 40.0):
        self.baseline_hb = baseline_hb
        self.csf_paco2 = max(15.0, baseline_paco2)
        self.tissue_oxygen = 1.0  # Share of normal
        self.dysfunction = 0.0
        self.anoxic_s = 0.0
        self.co2_narcosis = 0.0
        self.co2_respiratory_depression = 0.0
        self.hypercapnic_coma_s = 0.0

    def step(self, dt: float, map_mmhg: float, hb: float, saturation: float,
             pa_co2: float, ph: float) -> None:
        if dt <= 0.0:
            return
        # Normal supply is maintained above MAP 50. Partial chronic CBF adaptation
        # mitigates mild anemia, while severe baseline anemia reduces reserve.
        self.csf_paco2 += (max(15.0, pa_co2) - self.csf_paco2) * -math.expm1(-dt / self.CSF_ADAPTATION_TAU_S)
        pressure = clamp01((map_mmhg - 10.0) / 40.0)
        reference_hb = max(self.baseline_hb, 13.0 - 0.35 * max(0.0, 13.0 - self.baseline_hb))
        oxygen = max(0.0, hb * saturation) / (reference_hb * 98.0)
        # Normocapnic hypoxia adds up to 50% flow (calibrated); acute hypercapnia
        # augments hypoxic vasodilation (Battisti-Charbonney 2011) up to twice
        # baseline flow (Willie 2015), and hypocapnia lowers the flow ceiling
        # until hypotension limits perfusion (Brian 1998).
        effective_paco2 = max(10.0, pa_co2) * 40.0 / self.csf_paco2
        base_cbf = min(1.5, 1.0 + 1.5 * max(0.0, 1.0 - oxygen))
        if effective_paco2 >= 40.0:
            cbf = min(2.0, base_cbf + 0.5 * clamp01((effective_paco2 - 40.0) / 30.0))
            supply = pressure * min(1.0, cbf * oxygen)
        else:
            hypocapnia = clamp01((40.0 - effective_paco2) / 20.0)
            cbf = base_cbf * (1.0 - 0.38 * hypocapnia)
            supply = min(pressure * min(1.0, base_cbf * oxygen), cbf * oxygen)
        self.tissue_oxygen += (supply - self.tissue_oxygen) * -math.expm1(-dt / self.TISSUE_OXYGEN_TAU_S)
        target = clamp01((0.6 - self.tissue_oxygen) / 0.25)
        self.anoxic_s += dt * clamp01(2.0 - self.tissue_oxygen / 0.1)
        tau = 3.0 if target > self.dysfunction else 30.0
        self.dysfunction += (target - self.dysfunction) * -math.expm1(-dt / tau)
        # Acute hypercapnia impairs response more with acidemia; compensated
        # retention does so only above 100 mmHg. Gating pH by PaCO2 keeps
        # metabolic acidosis from causing narcosis.
        hypercapnia = clamp01((pa_co2 - 60.0) / 40.0)
        acidemia = clamp01((7.3 - ph) / 0.2)
        target_co2 = max(clamp01((pa_co2 - 100.0) / 40.0), hypercapnia * acidemia)
        prolonged = clamp01(
            (self.hypercapnic_coma_s - self.PROLONGED_NARCOSIS_ONSET_S) / self.PROLONGED_NARCOSIS_SPAN_S
        )
        recovery_tau = self.CO2_RESPONSE_TAU_S + (self.CO2_RECOVERY_MAX_TAU_S - self.CO2_RESPONSE_TAU_S) * prolonged
        co2_tau = self.CO2_RESPONSE_TAU_S if target_co2 >= self.co2_narcosis else recovery_tau
        self.co2_narcosis += (target_co2 - self.co2_narcosis) * -math.expm1(-dt / co2_tau)
        if self.co2_narcosis >= 0.5 and target_co2 >= 0.5:
            self.hypercapnic_coma_s = min(
                self.PROLONGED_NARCOSIS_ONSET_S + self.PROLONGED_NARCOSIS_SPAN_S,
                self.hypercapnic_coma_s + dt * clamp01((self.co2_narcosis - 0.5) / 0.3),
            )
        elif target_co2 < 0.5:
            self.hypercapnic_coma_s *= math.exp(-dt / (2.0 * self.CO2_RECOVERY_MAX_TAU_S))
        # Deeper hypercapnic coma depresses medullary breathing toward shallow
        # hypoventilation (Westlake 1955; Sieker and Hickam 1956).
        medullary_co2 = max(
            clamp01((pa_co2 - 110.0) / 35.0),
            clamp01((pa_co2 - 90.0) / 40.0) * clamp01((7.22 - ph) / 0.14),
        )
        target_resp_dep = 0.95 * medullary_co2 * medullary_co2
        resp_tau = self.CO2_RESPONSE_TAU_S if target_resp_dep >= self.co2_respiratory_depression else recovery_tau
        self.co2_respiratory_depression += (
            (target_resp_dep - self.co2_respiratory_depression) * -math.expm1(-dt / resp_tau)
        )

    @property
    def cortical_function(self) -> float:
        """Cortical electrical function for BIS; persistent coma and CO2 leave it unchanged (Turnbull 2008 for EtCO2 3-12 kPa under propofol)."""
        return 1.0 - self.dysfunction

    @property
    def respiratory_drive(self) -> float:
        """Medullary drive remaining after severe hypoxia/ischemia or extreme CO2 narcosis."""
        return (1.0 - self.dysfunction ** 8) * (1.0 - self.co2_respiratory_depression)

    @property
    def unresponsiveness(self) -> float:
        """Loss of response from oxygen failure, persistent coma, or CO2 narcosis."""
        if self.anoxic_s >= self.COMA_EXPOSURE_S:
            return 1.0
        return 1.0 - (1.0 - self.dysfunction) * (1.0 - self.co2_narcosis)
