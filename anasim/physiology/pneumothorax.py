"""Tension pneumothorax with positive-pressure progression and N2O cavity expansion.

Intrapleural air-to-pressure and hemodynamic collapse kinetics are calibrated to
ventilated porcine data (Nelson 2013) and the human systematic review of
spontaneous vs positive-pressure tension pneumothorax (Roberts 2015). Nitrous
oxide diffusion into the closed pleural gas space follows canine pleural cavity
kinetics (Eger and Saidman 1965): volume doubles by ~10 min and approaches
1 / (1 - F_A_N2O) over 1-2 h.
"""

from __future__ import annotations

import math

from anasim.core.utils import clamp, clamp01

# Eger and Saidman 1965 (canine pleural cavity, 68-78% N2O): 50% end-tidal N2O
# reached in ~5 min, volume doubled at 10 min, tripled at 45 min, ~4x at 2 h.
N2O_PLEURAL_TAU_S = 1300.0
DECOMPRESSION_TAU_S = 8.0  # Calibrated chest-tube venting time constant
MAX_PLEURAL_PRESSURE_MMHG = 24.0
MAX_PLEURAL_VOLUME_L = 4.5
MED_TRANSMISSION_FRACTION = 0.90


class PneumothoraxModel:
    """Unilateral tension pneumothorax state, N2O diffusion, and chest decompression."""

    def __init__(self, reference_frc_l: float = 2.4) -> None:
        self.reference_hemithorax_l = clamp(2.0 * (reference_frc_l / 2.4), 1.2, 3.0)
        self.active: bool = False
        self.decompressed: bool = False
        self.air_volume_l: float = 0.0
        self.n2o_volume_l: float = 0.0
        self.severity_scale: float = 1.0

    @property
    def total_volume_l(self) -> float:
        return self.air_volume_l + self.n2o_volume_l

    @property
    def collapse_fraction(self) -> float:
        """Ipsilateral lung collapse fraction (0-1)."""
        return clamp01(self.total_volume_l / max(0.5, 0.85 * self.reference_hemithorax_l))

    @property
    def pleural_pressure_mmhg(self) -> float:
        """Intrapleural pressure above resting baseline (mmHg, 0-24).

        Low-pressure lung collapse phase (<45% hemithorax volume) transitions to
        steep mediastinal shift and vena caval compression above 45% (Nelson
        2013 ventilated swine: CO -67% near 10 mmHg, PEA near mean 20 mmHg).
        """
        vol_ratio = self.total_volume_l / max(0.5, self.reference_hemithorax_l)
        if vol_ratio <= 0.0:
            return 0.0
        if vol_ratio <= 0.45:
            return 10.0 * vol_ratio  # 0 to 4.5 mmHg during initial ipsilateral collapse
        return min(MAX_PLEURAL_PRESSURE_MMHG, 4.5 + 22.0 * ((vol_ratio - 0.45) ** 1.35))

    @property
    def mediastinal_pit_mmhg(self) -> float:
        """Additional intrathoracic pressure transmitted to the venae cavae and heart (mmHg)."""
        return MED_TRANSMISSION_FRACTION * self.pleural_pressure_mmhg

    @property
    def lung_collapse_fraction(self) -> float:
        """Fraction of total bilateral lung volume lost to unilateral collapse (0-0.50)."""
        return 0.50 * self.collapse_fraction

    @property
    def chest_wall_stiffness_add(self) -> float:
        """Extra effective thoracic elastance under high pleural tension (collapses PCV VT)."""
        return 2.0 * clamp01((self.pleural_pressure_mmhg - 2.0) / 13.0)

    @property
    def shunt_fraction(self) -> float:
        """Calibrated ipsilateral unventilated perfusion shunt (0-0.38)."""
        return 0.38 * self.collapse_fraction

    def start(self, *, severity: float = 1.0, initial_volume_l: float = 0.25) -> None:
        """Start an active one-way pleural air leak."""
        self.active = True
        self.decompressed = False
        self.severity_scale = clamp(float(severity), 0.2, 3.0)
        if self.air_volume_l < initial_volume_l:
            self.air_volume_l = clamp(float(initial_volume_l), 0.0, MAX_PLEURAL_VOLUME_L)

    def set_closed_volume(self, air_volume_l: float) -> None:
        """Seed a closed (non-leaking) pneumothorax volume (e.g., to test N2O expansion)."""
        self.active = False
        self.decompressed = False
        self.air_volume_l = clamp(float(air_volume_l), 0.0, MAX_PLEURAL_VOLUME_L)
        self.n2o_volume_l = 0.0

    def stop(self) -> None:
        """Seal the active pleural leak without venting trapped gas."""
        self.active = False

    def decompress(self) -> None:
        """Perform needle thoracostomy / chest tube decompression, venting pleural gas."""
        self.decompressed = True
        self.active = False

    def reset(self) -> None:
        self.active = False
        self.decompressed = False
        self.air_volume_l = 0.0
        self.n2o_volume_l = 0.0
        self.severity_scale = 1.0

    def step(
        self,
        dt: float,
        *,
        ppv_active: bool = False,
        spontaneous_vent_active: bool = True,
        mean_paw_cmh2o: float = 0.0,
        peep_cmh2o: float = 0.0,
        p_alv_n2o: float = 0.0,
        circulation: float = 1.0,
    ) -> None:
        """Advance pleural gas volume and blood-borne N2O diffusion over dt seconds."""
        if dt <= 0.0:
            return
        if self.decompressed:
            decay = math.exp(-dt / DECOMPRESSION_TAU_S)
            self.air_volume_l *= decay
            self.n2o_volume_l *= decay
            if self.air_volume_l < 1e-4:
                self.air_volume_l = 0.0
            if self.n2o_volume_l < 1e-4:
                self.n2o_volume_l = 0.0
            return

        f_n2o = clamp(p_alv_n2o, 0.0, 0.85)
        if self.active:
            substeps = max(1, math.ceil(dt / 0.5))
            sub_dt = dt / substeps
            for _ in range(substeps):
                p_now = self.pleural_pressure_mmhg
                driving = max(0.0, mean_paw_cmh2o + 0.5 * max(0.0, peep_cmh2o))
                if ppv_active and driving > 1.0:
                    pressure_head = max(0.06, 1.0 - p_now / MAX_PLEURAL_PRESSURE_MMHG)
                    leak_l = self.severity_scale * 0.011 * (driving / 10.0) * pressure_head * sub_dt
                elif spontaneous_vent_active:
                    # Spontaneous breathing progresses slowly and rarely reaches lethal tension alone (Roberts 2015).
                    pressure_head = max(0.0, 1.0 - p_now / 9.0)
                    leak_l = self.severity_scale * 0.0012 * pressure_head * sub_dt
                else:
                    leak_l = 0.0
                if leak_l > 0.0:
                    headroom = max(0.0, MAX_PLEURAL_VOLUME_L - (self.air_volume_l + self.n2o_volume_l))
                    added = min(leak_l, headroom)
                    self.air_volume_l += added * (1.0 - f_n2o)
                    self.n2o_volume_l += added * f_n2o

        # Blood-borne N2O diffuses into or out of closed pleural spaces toward Dalton equilibrium
        # V_n2o_eq = V_air * F_A_N2O / (1 - F_A_N2O) (Eger and Saidman 1965); halts at zero circulation.
        circ = clamp01(circulation)
        if circ > 0.0 and (self.air_volume_l > 1e-5 or self.n2o_volume_l > 1e-5):
            target_n2o = self.air_volume_l * f_n2o / max(0.15, 1.0 - f_n2o)
            alpha = -math.expm1(-dt * circ / N2O_PLEURAL_TAU_S)
            self.n2o_volume_l = clamp(
                self.n2o_volume_l + (target_n2o - self.n2o_volume_l) * alpha,
                0.0,
                max(0.0, MAX_PLEURAL_VOLUME_L - self.air_volume_l),
            )
            if self.n2o_volume_l < 1e-5 and f_n2o <= 1e-4:
                self.n2o_volume_l = 0.0
