from dataclasses import dataclass

from anasim.core.utils import clamp01

@dataclass
class GasComposition:
    fio2: float = 0.21
    fin2: float = 0.79
    fin2o: float = 0.0
    fi_agent: float = 0.0  # Volatile fraction (0-1)
    
class CircleSystem:
    """
    Simulates the breathing circuit gas composition (Wash-in/Wash-out).
    """
    def __init__(self, volume_l: float = 6.0):
        self.volume = volume_l  # Circuit volume (inc. bag)
        self.composition = GasComposition()
        
        # FGF settings (L/min)
        self.fgf_o2 = 2.0
        self.fgf_air = 0.0
        self.fgf_n2o = 0.0
        
        # Vaporizer
        self.vaporizer_agent = "Sevo"
        self.vaporizer_setting = 0.0  # %
        self.vaporizer_on = False
        
    def fgf_total(self) -> float:
        return self.fgf_o2 + self.fgf_air + self.fgf_n2o

    def step(self, dt: float, uptake_o2: float, uptake_agent: float):
        """
        dt: seconds
        uptake_o2: L/min (Metabolic)
        uptake_agent: L/min (Volatile uptake, can be negative if excreting)
        """
        # 1. Total FGF
        total_fgf = self.fgf_o2 + self.fgf_air + self.fgf_n2o
        if total_fgf < 0.1:
            total_fgf = 0.1  # Min flow check
        
        # 2. Fresh Gas Composition
        # Air is 21% O2, 79% N2
        fg_o2 = (self.fgf_o2 + 0.21 * self.fgf_air) / total_fgf
        fg_n2o = self.fgf_n2o / total_fgf
        
        # Vaporizer output
        fg_agent = (self.vaporizer_setting / 100.0) if self.vaporizer_on else 0.0
        
        # Dilution effect of agent on other gases
        fg_o2 *= (1.0 - fg_agent)
        fg_n2o *= (1.0 - fg_agent)
        
        # 3. Mixing in Circuit (1-compartment model)
        # V dF/dt = FGF*(F_fg - F_circ) - Uptake
        
        dt_min = dt / 60.0
        
        # O2
        d_fio2 = (total_fgf * (fg_o2 - self.composition.fio2) - uptake_o2) / self.volume * dt_min
        self.composition.fio2 += d_fio2
        
        # Agent
        d_fiagent = (total_fgf * (fg_agent - self.composition.fi_agent) - uptake_agent) / self.volume * dt_min
        self.composition.fi_agent += d_fiagent
        
        # N2O
        d_fin2o = (total_fgf * (fg_n2o - self.composition.fin2o)) / self.volume * dt_min
        self.composition.fin2o += d_fin2o

        # Normalize fractions and recompute N2 balance
        self.composition.fi_agent = clamp01(self.composition.fi_agent)
        self.composition.fio2 = clamp01(self.composition.fio2)
        self.composition.fin2o = clamp01(self.composition.fin2o)

        available = max(0.0, 1.0 - self.composition.fi_agent)
        non_n2 = self.composition.fio2 + self.composition.fin2o
        if non_n2 > available and non_n2 > 0:
            scale = available / non_n2
            self.composition.fio2 *= scale
            self.composition.fin2o *= scale
            non_n2 = self.composition.fio2 + self.composition.fin2o
        self.composition.fin2 = max(0.0, available - non_n2)
