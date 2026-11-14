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
        self.reservoir_capacity = min(3.0, 0.5 * volume_l)
        self.reservoir_volume = self.reservoir_capacity
        self.deliverable_flow_l_min = math.inf
        self.last_net_flow = 0.0  # L/min
        self.composition = GasComposition()
        self.fgf_o2 = 2.0  # L/min
        self.fgf_air = 0.0
        self.fgf_n2o = 0.0
        self.oxygen_supply_connected = True
        self.vaporizer_setting = 0.0  # %
        self.vaporizer_on = False

    @property
    def bellows_fraction(self) -> float:
        """Deliverable tidal fraction remaining before the reservoir bag/bellows collapses."""
        return clamp01(self.reservoir_volume / 0.5)

    def delivered_o2_flow(self) -> float:
        """Return oxygen flow reaching the circuit from the supply."""
        return self.fgf_o2 if self.oxygen_supply_connected else 0.0

    def delivered_n2o_flow(self) -> float:
        """Return N2O flow after the oxygen-pressure fail-safe and 25% O2 guard."""
        if not self.oxygen_supply_connected:
            return 0.0
        return min(self.fgf_n2o, 3.0 * self.delivered_o2_flow())

    def fgf_total(self) -> float:
        """Return delivered carrier flow before added vapor, L/min."""
        return self.delivered_o2_flow() + self.fgf_air + self.delivered_n2o_flow()

    def deliverable_flow(self, dt: float, vapor_flow: float = 0.0) -> float:
        """Maximum gas flow (L/min) available from fresh gas plus the reservoir over dt."""
        reservoir_rate = (
            self.reservoir_volume * 60.0 / dt
            if dt > 0.0
            else (math.inf if self.reservoir_volume > 0.0 else 0.0)
        )
        return max(0.0, self.fgf_total()) + vapor_flow + reservoir_rate

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

    def step(
        self,
        dt: float,
        uptake_o2: float,
        uptake_agent: float,
        uptake_n2o: float = 0.0,
        uptake_n2: float = 0.0,
        rebreathed_n2: float | None = None,
    ):
        """Advance dt seconds. Uptakes are L/min and negative during washout.

        V dF/dt = supplied gas - uptake - spill*F for each gas.
        Spill is total inflow minus net uptake, including agent and rebreathed N2.
        """
        delivered_o2 = self.delivered_o2_flow()
        delivered_n2o = self.delivered_n2o_flow()
        total_fgf = max(0.0, delivered_o2 + self.fgf_air + delivered_n2o)

        # The dial specifies vapor as a fraction of carrier plus added vapor.
        fg_agent = self.vaporizer_setting / 100.0 if self.vaporizer_on else 0.0
        vapor_flow = total_fgf * fg_agent / (1.0 - fg_agent)
        bag_flow = total_fgf + vapor_flow - uptake_o2 - uptake_agent - uptake_n2o - uptake_n2
        self.last_net_flow = bag_flow
        dt_min = dt / 60.0
        prev_reservoir = self.reservoir_volume
        # Record pre-debit capacity so downstream lung exchange receives this step's gas.
        self.deliverable_flow_l_min = self.deliverable_flow(dt, vapor_flow)
        self.reservoir_volume = min(
            self.reservoir_capacity,
            max(0.0, self.reservoir_volume + bag_flow * dt_min),
        )
        bag_change_rate = (self.reservoir_volume - prev_reservoir) / dt_min if dt_min > 0.0 else 0.0
        n2_mix = uptake_n2 if rebreathed_n2 is None else rebreathed_n2
        mix_flow = total_fgf + vapor_flow - uptake_o2 - uptake_agent - uptake_n2o - n2_mix
        spill = max(0.0, mix_flow - max(0.0, bag_change_rate))
        o2_flow = delivered_o2 + 0.21 * self.fgf_air
        eff_volume = max(1.0, self.volume - (self.reservoir_capacity - prev_reservoir))
        next_eff_volume = max(1.0, self.volume - (self.reservoir_capacity - self.reservoir_volume))
        eff_change_rate = (next_eff_volume - eff_volume) / dt_min if dt_min > 0.0 else 0.0
        outflow = spill + eff_change_rate
        # Exact constant-rate integration of d(V F)/dt = inflow - uptake - spill * F.
        if abs(eff_change_rate) <= 1e-9:
            mixing = -math.expm1(-spill * dt_min / eff_volume) / spill if spill > 1e-9 else dt_min / eff_volume
        elif spill <= 1e-9:
            mixing = dt_min / next_eff_volume
        elif abs(outflow) <= 1e-9:
            mixing = -math.log(eff_volume / next_eff_volume) / eff_change_rate
        else:
            mixing = -math.expm1(outflow * math.log(eff_volume / next_eff_volume) / eff_change_rate) / outflow
        self.composition.fio2 += (o2_flow - uptake_o2 - outflow * self.composition.fio2) * mixing
        self.composition.fi_agent += (vapor_flow - uptake_agent - outflow * self.composition.fi_agent) * mixing
        self.composition.fin2o += (delivered_n2o - uptake_n2o - outflow * self.composition.fin2o) * mixing

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
