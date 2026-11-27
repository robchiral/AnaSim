from dataclasses import dataclass

from anasim.core.utils import clamp


@dataclass
class VaporizerState:
    setting: float = 0.0  # %
    is_on: bool = False
    level: float = 250.0  # mL of liquid


class Vaporizer:
    """Sevoflurane vaporizer with liquid depletion."""

    CAPACITY_ML: float = 250.0

    def __init__(self):
        self.state = VaporizerState()

    def set_concentration(self, conc: float):
        self.state.setting = clamp(conc, 0.0, 8.0)  # Sevoflurane dial maximum
        self.state.is_on = self.state.setting > 0 and self.state.level > 0.0

    def refill(self, level_ml: float = CAPACITY_ML) -> None:
        """Refill the liquid sevoflurane reservoir up to CAPACITY_ML."""
        self.state.level = clamp(float(level_ml), 0.0, self.CAPACITY_ML)
        if self.state.level > 0.0 and self.state.setting > 0.0:
            self.state.is_on = True
        elif self.state.level <= 0.0:
            self.state.is_on = False

    def step(self, dt: float, fgf_l_min: float) -> float:
        """Consume liquid and return mean delivered vapor concentration (%)."""
        if not self.state.is_on or fgf_l_min <= 0 or dt <= 0:
            return 0.0

        # 1 mL of liquid sevoflurane gives about 170-180 mL of vapor.
        vapor_to_liquid_expansion = 180.0
        fraction = self.state.setting / 100.0
        vapor_vol_l_min = fgf_l_min * fraction / (1.0 - fraction)
        liquid_ml = vapor_vol_l_min * 1000.0 / vapor_to_liquid_expansion * (dt / 60.0)
        if liquid_ml < self.state.level:
            self.state.level -= liquid_ml
            return self.state.setting

        # The remaining liquid is delivered over this step; keep the dial setting
        # so refilling resumes delivery at the dialed concentration.
        vapor_vol_l_min *= self.state.level / liquid_ml
        self.state.level = 0.0
        self.state.is_on = False
        return 100.0 * vapor_vol_l_min / (fgf_l_min + vapor_vol_l_min)
