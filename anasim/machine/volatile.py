from dataclasses import dataclass

from anasim.core.utils import clamp

@dataclass
class VaporizerState:
    agent: str = "Sevo"
    setting: float = 0.0 # %
    is_on: bool = False
    level: float = 250.0 # mL liquid remaining

class Vaporizer:
    """
    Anesthetic Vaporizer.
    """
    def __init__(self, agent: str = "Sevo"):
        self.state = VaporizerState(agent=agent)
        
    def set_concentration(self, conc: float):
        self.state.setting = clamp(conc, 0.0, 8.0)  # Cap at 8% (Sevoflurane max)
        self.state.is_on = self.state.setting > 0
            
    def step(self, dt: float, fgf_l_min: float):
        """
        Consume liquid agent.
        Rule: 3 * Flow * Vol% ~ mL/hr liquid.
        """
        if not self.state.is_on or fgf_l_min <= 0:
            return

        # 1 mL liquid sevoflurane yields ~170 mL vapor; use 180 as a conservative round value.
        vapor_to_liquid_expansion = 180.0
        vapor_vol_l_min = fgf_l_min * (self.state.setting / 100.0)
        liquid_vol_ml_min = (vapor_vol_l_min * 1000.0) / vapor_to_liquid_expansion

        self.state.level -= liquid_vol_ml_min * (dt / 60.0)
        self.state.level = max(0.0, self.state.level)
        if self.state.level == 0:
            self.state.is_on = False
            self.state.setting = 0
