"""Laparoscopic CO2 pneumoperitoneum mechanics, gas absorption, and hemodynamics.

Calibrated to invasive human laparoscopy hemodynamics (Joris 1993, 14 mmHg CO2
pneumoperitoneum: MAP +35%, SVR +65%, PVR +90%, CI -20%, RAP +6 mmHg, plateau
Paw +6-7 cmH2O), respiratory mechanics (Fahy 1995), intraperitoneal CO2
absorption kinetics (Tan 1992; Mullett 1993: ~30% VCO2 rise plateauing over
15-20 min with gradual washout after desufflation), and paired VitalDB
laparoscopic vs open supine cohorts.
"""

from __future__ import annotations

import math

from anasim.core.utils import clamp, clamp01

DEFAULT_IAP_MMHG = 14.0
MIN_IAP_MMHG = 4.0
MAX_IAP_MMHG = 25.0

INSUFFLATION_TAU_S = 45.0
DESUFFLATION_TAU_S = 12.0
HUMORAL_ON_TAU_S = 75.0
HUMORAL_OFF_TAU_S = 240.0
ADAPTATION_ON_TAU_S = 600.0
ADAPTATION_OFF_TAU_S = 300.0
CO2_ABSORB_ON_TAU_S = 240.0
CO2_ABSORB_OFF_TAU_S = 600.0


class PneumoperitoneumModel:
    """Intra-abdominal CO2 insufflation state and coupled cardiopulmonary effects."""

    def __init__(self) -> None:
        self.active: bool = False
        self.target_iap_mmhg: float = 0.0
        self.iap_mmhg: float = 0.0
        self._humoral_tone: float = 0.0
        self._adaptation: float = 0.0
        self._co2_load_fraction: float = 0.0

    @property
    def _iap_norm(self) -> float:
        return self.iap_mmhg / DEFAULT_IAP_MMHG

    @property
    def svr_factor(self) -> float:
        """Systemic vascular resistance multiplier from abdominal compression and vasopressin."""
        drive = 0.30 * self._iap_norm + 0.70 * self._humoral_tone
        if drive <= 1e-6:
            return 1.0
        return 1.0 + 0.72 * drive * (1.0 - 0.18 * self._adaptation)

    @property
    def pvr_factor(self) -> float:
        """Pulmonary vascular resistance multiplier from diaphragmatic elevation and humoral tone."""
        if self.iap_mmhg <= 1e-6 and self._humoral_tone <= 1e-6:
            return 1.0
        return 1.0 + 0.75 * (0.65 * self._iap_norm + 0.35 * self._humoral_tone)

    @property
    def pit_add_mmhg(self) -> float:
        """Direct transdiaphragmatic intrathoracic pressure rise (mmHg)."""
        return 0.14 * self.iap_mmhg

    @property
    def mcfp_add_mmhg(self) -> float:
        """Splanchnic venous compression raising abdominal MCFP up to the IVC-collapse threshold (mmHg)."""
        return 0.18 * min(self.iap_mmhg, DEFAULT_IAP_MMHG)

    @property
    def intracavitary_pressure_add_mmhg(self) -> float:
        """Extracardiac thoracic pressure transmitted equally to measured RAP and PCWP/LAP (mmHg)."""
        return 0.32 * self.iap_mmhg

    @property
    def rvr_factor(self) -> float:
        """Inferior vena caval / abdominal venous return resistance multiplier."""
        if self.iap_mmhg <= 1e-6:
            return 1.0
        high_iap = max(0.0, self.iap_mmhg - 12.0) / 10.0
        return 1.0 + 0.08 * self._iap_norm + 0.42 * (high_iap ** 1.5)

    @property
    def chest_wall_stiffness_add(self) -> float:
        """Additional chest-wall/diaphragmatic elastance ratio from abdominal pressurization."""
        return 0.42 * self._iap_norm

    @property
    def alveolar_deadspace(self) -> float:
        """Parallel alveolar dead-space fraction widening the PaCO2-EtCO2 gradient."""
        return 0.055 * clamp01(self._iap_norm)

    def co2_absorption_ml_min(self, baseline_vco2_ml_min: float) -> float:
        """Exogenous peritoneal CO2 load entering venous blood (mL/min STPD)."""
        if self._co2_load_fraction <= 1e-6 or baseline_vco2_ml_min <= 0.0:
            return 0.0
        return 0.30 * baseline_vco2_ml_min * self._co2_load_fraction

    def start(self, iap_mmhg: float = DEFAULT_IAP_MMHG) -> None:
        """Begin or update laparoscopic CO2 insufflation at the target IAP (mmHg)."""
        self.active = True
        self.target_iap_mmhg = clamp(float(iap_mmhg), MIN_IAP_MMHG, MAX_IAP_MMHG)

    def stop(self) -> None:
        """Desufflate the abdomen (target IAP -> 0; humoral tone and tissue CO2 wash out)."""
        self.active = False
        self.target_iap_mmhg = 0.0

    def reset(self) -> None:
        """Immediately clear pneumoperitoneum and all residual humoral/CO2 state."""
        self.active = False
        self.target_iap_mmhg = 0.0
        self.iap_mmhg = 0.0
        self._humoral_tone = 0.0
        self._adaptation = 0.0
        self._co2_load_fraction = 0.0

    def step(self, dt: float, *, circulation: float = 1.0) -> None:
        """Advance insufflation pressure, humoral SVR surge, and peritoneal CO2 absorption."""
        if dt <= 0.0:
            return
        target = self.target_iap_mmhg if self.active else 0.0
        if target != self.iap_mmhg:
            tau = INSUFFLATION_TAU_S if target > self.iap_mmhg else DESUFFLATION_TAU_S
            self.iap_mmhg += (target - self.iap_mmhg) * -math.expm1(-dt / tau)
            if not self.active and self.iap_mmhg < 1e-3:
                self.iap_mmhg = 0.0

        circ = clamp01(circulation)
        iap_norm = self._iap_norm
        if circ > 0.0:
            h_tau = HUMORAL_ON_TAU_S if iap_norm > self._humoral_tone else HUMORAL_OFF_TAU_S
            self._humoral_tone += (iap_norm - self._humoral_tone) * -math.expm1(-dt * circ / h_tau)
            if iap_norm == 0.0 and self._humoral_tone < 1e-4:
                self._humoral_tone = 0.0

            adapt_target = clamp01(iap_norm)
            a_tau = ADAPTATION_ON_TAU_S if adapt_target > self._adaptation else ADAPTATION_OFF_TAU_S
            self._adaptation += (adapt_target - self._adaptation) * -math.expm1(-dt * circ / a_tau)
            if adapt_target == 0.0 and self._adaptation < 1e-4:
                self._adaptation = 0.0

            co2_target = min(1.35, iap_norm ** 0.70) if iap_norm > 0.0 else 0.0
            c_tau = CO2_ABSORB_ON_TAU_S if co2_target > self._co2_load_fraction else CO2_ABSORB_OFF_TAU_S
            self._co2_load_fraction += (co2_target - self._co2_load_fraction) * -math.expm1(-dt * circ / c_tau)
            if co2_target == 0.0 and self._co2_load_fraction < 1e-4:
                self._co2_load_fraction = 0.0
