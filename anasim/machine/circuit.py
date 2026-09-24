from dataclasses import dataclass

from anasim.core.utils import clamp01


@dataclass
class GasComposition:
    fio2: float = 0.21
    fin2: float = 0.79
    fin2o: float = 0.0
    fi_agent: float = 0.0


class CircleSystem:
    """Well-mixed circle system gas: fresh gas in, patient uptake out."""

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
        """Return total fresh gas flow actually reaching the circuit."""
        return self.delivered_o2_flow() + self.fgf_air + self.delivered_n2o_flow()

    def equilibrate(self, uptake_o2: float, fi_agent: float = 0.0) -> None:
        """Set the steady-state composition for the current fresh gas flows."""
        total_fgf = self.fgf_total()
        if total_fgf <= 0.0:
            return
        fg_agent = self.vaporizer_setting / 100.0 if self.vaporizer_on else 0.0
        fg_o2 = (self.delivered_o2_flow() + 0.21 * self.fgf_air) / total_fgf * (1.0 - fg_agent)
        composition = self.composition
        composition.fi_agent = fi_agent
        composition.fin2o = self.delivered_n2o_flow() / total_fgf * (1.0 - fg_agent)
        composition.fio2 = clamp01(fg_o2 - uptake_o2 / total_fgf)
        composition.fin2 = max(0.0, 1.0 - fi_agent - composition.fin2o - composition.fio2)

    def step(self, dt: float, uptake_o2: float, uptake_agent: float, uptake_n2o: float = 0.0):
        """Advance dt seconds. Uptakes are L/min and negative during washout.

        V dF/dt = FGF (F_fresh - F) - uptake for each gas.
        """
        delivered_o2 = self.delivered_o2_flow()
        delivered_n2o = self.delivered_n2o_flow()
        total_fgf = max(0.0, delivered_o2 + self.fgf_air + delivered_n2o)

        if total_fgf > 0.0:
            # Vapor dilutes the other fresh gases.
            fg_agent = (self.vaporizer_setting / 100.0) if self.vaporizer_on else 0.0
            fg_o2 = (delivered_o2 + 0.21 * self.fgf_air) / total_fgf * (1.0 - fg_agent)
            fg_n2o = delivered_n2o / total_fgf * (1.0 - fg_agent)
        else:
            fg_o2 = self.composition.fio2
            fg_n2o = self.composition.fin2o
            fg_agent = 0.0

        dt_min = dt / 60.0
        d_fio2 = (total_fgf * (fg_o2 - self.composition.fio2) - uptake_o2) / self.volume * dt_min
        self.composition.fio2 += d_fio2
        d_fiagent = (total_fgf * (fg_agent - self.composition.fi_agent) - uptake_agent) / self.volume * dt_min
        self.composition.fi_agent += d_fiagent
        d_fin2o = (total_fgf * (fg_n2o - self.composition.fin2o) - uptake_n2o) / self.volume * dt_min
        self.composition.fin2o += d_fin2o

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
