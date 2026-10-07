import math
from dataclasses import dataclass

from anasim.core.utils import clamp01


@dataclass
class GasComposition:
    fio2: float = 0.21
    fin2: float = 0.79
    fin2o: float = 0.0
    fi_agent: float = 0.0


class CircleSystem:
    """Well-mixed circle with patient uptake and spill of surplus fresh gas.
    """

    def __init__(self, volume_l: float = 6.0):
        self.volume = volume_l  # Including the bag
        self.composition = GasComposition()
        self.fgf_o2 = 2.0  # L/min
        self.fgf_air = 0.0
        self.fgf_n2o = 0.0
        self.oxygen_supply_connected = True
        self.vaporizer_agent = "Sevo"
        self.vaporizer_setting = 0.0  # %
        self.vaporizer_on = False

    def delivered_o2_flow(self) -> float:
        """Return oxygen flow reaching the circuit from the supply."""
        return self.fgf_o2 if self.oxygen_supply_connected else 0.0

    def delivered_n2o_flow(self) -> float:
        """Return N2O flow after the oxygen-pressure fail-safe."""
        return self.fgf_n2o if self.oxygen_supply_connected else 0.0

    def fgf_total(self) -> float:
        """Return delivered carrier flow before added vapor, L/min."""
        return self.delivered_o2_flow() + self.fgf_air + self.delivered_n2o_flow()

    def equilibrate(self, uptake_o2: float, fi_agent: float = 0.0) -> None:
        """Set steady composition at the supplied agent fraction and zero N2O uptake."""
        total_fgf = self.fgf_total()
        net_o2 = self.delivered_o2_flow() + 0.21 * self.fgf_air - uptake_o2
        if total_fgf <= uptake_o2 or net_o2 < 0.0:
            raise ValueError("Circuit equilibrium requires enough oxygen flow and positive spill")
        # Carrier gases occupy 1-Fi_agent of the spill flow. Agent uptake is
        # implicit in the requested Fi_agent, which may differ from the dial.
        spill = (total_fgf - uptake_o2) / (1.0 - fi_agent)
        composition = self.composition
        composition.fi_agent = fi_agent
        composition.fin2o = self.delivered_n2o_flow() / spill
        composition.fio2 = net_o2 / spill
        composition.fin2 = max(0.0, 1.0 - fi_agent - composition.fin2o - composition.fio2)

    def step(self, dt: float, uptake_o2: float, uptake_agent: float, uptake_n2o: float = 0.0):
        """Advance dt seconds. Uptakes are L/min and negative during washout.

        V dF/dt = supplied gas - uptake - spill*F for each gas.
        Spill is total inflow minus net uptake, including agent washout.
        """
        delivered_o2 = self.delivered_o2_flow()
        delivered_n2o = self.delivered_n2o_flow()
        total_fgf = max(0.0, delivered_o2 + self.fgf_air + delivered_n2o)

        # The dial specifies vapor as a fraction of carrier plus added vapor.
        fg_agent = self.vaporizer_setting / 100.0 if self.vaporizer_on else 0.0
        vapor_flow = total_fgf * fg_agent / (1.0 - fg_agent)
        net_flow = total_fgf + vapor_flow - uptake_o2 - uptake_agent - uptake_n2o
        spill = max(0.0, net_flow)
        entrained_air = max(0.0, -net_flow)
        o2_flow = delivered_o2 + 0.21 * (self.fgf_air + entrained_air)

        dt_min = dt / 60.0
        # Exact mixing for constant flows over the interval, including zero spill.
        mixing = -math.expm1(-spill * dt_min / self.volume) / spill if spill > 0.0 else dt_min / self.volume
        self.composition.fio2 += (o2_flow - uptake_o2 - spill * self.composition.fio2) * mixing
        self.composition.fi_agent += (vapor_flow - uptake_agent - spill * self.composition.fi_agent) * mixing
        self.composition.fin2o += (delivered_n2o - uptake_n2o - spill * self.composition.fin2o) * mixing

        # N2 is the balance gas.
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
