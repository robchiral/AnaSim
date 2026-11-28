"""Medical gas pipelines, backup E-cylinders, and ventilator drive-gas consumption.

Pipeline pressures, secondary cylinder regulator switchover, liquefied N2O
vapor-pressure plateau, and oxygen-pressure fail-safe follow NFPA 99 / ISO
7396-1 anesthesia workstation architecture. Pneumatic bellows ventilator
drive-gas consumption follows Szpisjak et al. (Anesth Analg 2008; 107:1907-14).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from anasim.core.utils import clamp, clamp01
from anasim.patient.domain import finite_number

GAS_NAMES = ("o2", "air", "n2o")
DRIVE_GAS_MODES = ("o2", "o2_only", "air", "electric")

PIPELINE_NOMINAL_PSIG = 50.0
PIPELINE_LOW_PSIG = 35.0
CYLINDER_REGULATOR_PSIG = 45.0
FAILSAFE_CUTOFF_PSIG = 25.0

O2_CYLINDER_CAPACITY_L = 660.0
O2_CYLINDER_FULL_PSIG = 1900.0
O2_CYLINDER_LOW_PSIG = 500.0

AIR_CYLINDER_CAPACITY_L = 625.0
AIR_CYLINDER_FULL_PSIG = 1900.0
AIR_CYLINDER_LOW_PSIG = 500.0

N2O_CYLINDER_CAPACITY_L = 1590.0
N2O_CYLINDER_FULL_PSIG = 745.0
# Liquefied N2O holds ~745 psig at 20 °C until liquid exhausts near 25% remaining (~400 L).
N2O_LIQUID_EXHAUSTION_L = 400.0
N2O_CYLINDER_LOW_PSIG = 500.0

# Combined ventilator-bellows and breathing-circuit internal compliance (Szpisjak 2008).
CIRCUIT_VENT_COMPLIANCE_L_CMH2O = 0.005


@dataclass(slots=True)
class GasSource:
    """Single medical gas supply with central pipeline and backup E-cylinder."""

    gas: str
    cylinder_capacity_l: float
    cylinder_full_psig: float
    cylinder_low_psig: float
    liquid_exhaustion_l: float = 0.0
    pipeline_connected: bool = True
    pipeline_pressure_psig: float = PIPELINE_NOMINAL_PSIG
    cylinder_valve_open: bool = False
    cylinder_volume_l: float = 0.0
    regulator_psig: float = CYLINDER_REGULATOR_PSIG
    last_flow_l_min: float = 0.0
    last_drive_flow_l_min: float = 0.0

    def __post_init__(self) -> None:
        if self.cylinder_volume_l <= 0.0:
            self.cylinder_volume_l = self.cylinder_capacity_l

    @property
    def effective_pipeline_psig(self) -> float:
        """Pipeline pressure seen at the workstation inlet gauge (psig)."""
        return max(0.0, self.pipeline_pressure_psig) if self.pipeline_connected else 0.0

    @property
    def cylinder_pressure_psig(self) -> float:
        """Cylinder contents gauge pressure (psig); flat for liquefied N2O until liquid exhausts."""
        vol = max(0.0, self.cylinder_volume_l)
        if vol <= 0.0:
            return 0.0
        if self.liquid_exhaustion_l > 0.0:
            if vol >= self.liquid_exhaustion_l:
                return self.cylinder_full_psig
            return self.cylinder_full_psig * (vol / self.liquid_exhaustion_l)
        return self.cylinder_full_psig * (vol / max(1e-6, self.cylinder_capacity_l))

    @property
    def cylinder_fraction(self) -> float:
        """Remaining fraction of full E-cylinder gas volume, 0-1."""
        return clamp01(self.cylinder_volume_l / max(1e-6, self.cylinder_capacity_l))

    @property
    def regulated_cylinder_psig(self) -> float:
        """Downstream pressure supplied by the backup cylinder regulator (psig)."""
        if not self.cylinder_valve_open or self.cylinder_volume_l <= 0.0:
            return 0.0
        return min(self.regulator_psig, self.cylinder_pressure_psig)

    @property
    def supply_pressure_psig(self) -> float:
        """Manifold pressure feeding flowmeters and pneumatic subsystems (psig)."""
        return max(self.effective_pipeline_psig, self.regulated_cylinder_psig)

    @property
    def active_source(self) -> str:
        """Currently supplying source: 'pipeline', 'cylinder', or 'none'."""
        pipe_p = self.effective_pipeline_psig
        cyl_p = self.regulated_cylinder_psig
        if pipe_p >= cyl_p and pipe_p > 1e-6:
            return "pipeline"
        if cyl_p > 1e-6:
            return "cylinder"
        return "none"

    @property
    def available(self) -> bool:
        """Whether downstream supply pressure meets the workstation fail-safe threshold."""
        return self.supply_pressure_psig >= FAILSAFE_CUTOFF_PSIG

    @property
    def delivery_scale(self) -> float:
        """Fractional flowmeter delivery at current manifold pressure, 0-1."""
        p = self.supply_pressure_psig
        if p >= FAILSAFE_CUTOFF_PSIG:
            return 1.0
        return clamp01(p / FAILSAFE_CUTOFF_PSIG)

    @property
    def pipeline_low(self) -> bool:
        """Whether the central wall pipeline gauge is below the low-pressure alarm limit."""
        return self.effective_pipeline_psig < PIPELINE_LOW_PSIG

    @property
    def cylinder_low(self) -> bool:
        """Whether an open backup E-cylinder gauge has fallen below its low-reserve limit."""
        return self.cylinder_valve_open and self.cylinder_pressure_psig < self.cylinder_low_psig

    @property
    def total_draw_l_min(self) -> float:
        """Total gas draw during the last step (fresh gas + flush + drive gas), L/min."""
        return self.last_flow_l_min + self.last_drive_flow_l_min

    @property
    def cylinder_minutes_remaining(self) -> float | None:
        """Estimated E-cylinder runtime at current draw rate (min), or None if not drawing."""
        if self.active_source != "cylinder":
            return None
        rate = self.total_draw_l_min
        if rate <= 1e-6:
            return math.inf if self.cylinder_volume_l > 0.0 else 0.0
        return self.cylinder_volume_l / rate

    def set_pipeline(self, connected: bool, pressure_psig: float | None = None) -> None:
        self.pipeline_connected = bool(connected)
        if not self.pipeline_connected:
            self.pipeline_pressure_psig = 0.0
        elif pressure_psig is not None:
            p = finite_number(f"{self.gas} pipeline pressure", pressure_psig)
            self.pipeline_pressure_psig = clamp(p, 0.0, 100.0)
        elif self.pipeline_pressure_psig <= 0.0:
            self.pipeline_pressure_psig = PIPELINE_NOMINAL_PSIG

    def set_pipeline_pressure(self, pressure_psig: float) -> None:
        p = clamp(finite_number(f"{self.gas} pipeline pressure", pressure_psig), 0.0, 100.0)
        self.pipeline_pressure_psig = p
        self.pipeline_connected = p > 0.0

    def set_cylinder_valve(self, open_: bool) -> None:
        self.cylinder_valve_open = bool(open_)

    def replace_cylinder(self, fraction: float = 1.0) -> None:
        frac = clamp01(finite_number(f"{self.gas} cylinder fraction", fraction))
        self.cylinder_volume_l = frac * self.cylinder_capacity_l

    def consume(self, dt: float, flow_l_min: float, drive_l_min: float = 0.0) -> tuple[float, float]:
        """Withdraw gas over dt seconds; return actually delivered (flow_l_min, drive_l_min)."""
        flow = max(0.0, float(flow_l_min))
        drive = max(0.0, float(drive_l_min))
        total = flow + drive
        source = self.active_source
        if source == "none":
            self.last_flow_l_min = 0.0
            self.last_drive_flow_l_min = 0.0
            return 0.0, 0.0
        scale = self.delivery_scale
        if source == "pipeline" or dt <= 0.0 or total <= 1e-12:
            deliv_flow = flow * scale
            deliv_drive = drive * scale
            self.last_flow_l_min = deliv_flow
            self.last_drive_flow_l_min = deliv_drive
            return deliv_flow, deliv_drive
        dt_min = dt / 60.0
        max_rate = self.cylinder_volume_l / dt_min
        actual_scale = min(scale, max_rate / total)
        deliv_flow = flow * actual_scale
        deliv_drive = drive * actual_scale
        self.cylinder_volume_l = max(0.0, self.cylinder_volume_l - (deliv_flow + deliv_drive) * dt_min)
        self.last_flow_l_min = deliv_flow
        self.last_drive_flow_l_min = deliv_drive
        return deliv_flow, deliv_drive


class GasSupplySystem:
    """Workstation O2, Air, and N2O pipelines, E-cylinders, and ventilator drive gas."""

    def __init__(self, drive_gas_mode: str = "o2") -> None:
        self.o2 = GasSource(
            gas="o2",
            cylinder_capacity_l=O2_CYLINDER_CAPACITY_L,
            cylinder_full_psig=O2_CYLINDER_FULL_PSIG,
            cylinder_low_psig=O2_CYLINDER_LOW_PSIG,
        )
        self.air = GasSource(
            gas="air",
            cylinder_capacity_l=AIR_CYLINDER_CAPACITY_L,
            cylinder_full_psig=AIR_CYLINDER_FULL_PSIG,
            cylinder_low_psig=AIR_CYLINDER_LOW_PSIG,
        )
        self.n2o = GasSource(
            gas="n2o",
            cylinder_capacity_l=N2O_CYLINDER_CAPACITY_L,
            cylinder_full_psig=N2O_CYLINDER_FULL_PSIG,
            cylinder_low_psig=N2O_CYLINDER_LOW_PSIG,
            liquid_exhaustion_l=N2O_LIQUID_EXHAUSTION_L,
        )
        self._sources = {"o2": self.o2, "air": self.air, "n2o": self.n2o}
        self.drive_gas_mode: str = "o2"
        self.set_drive_gas_mode(drive_gas_mode)
        self.last_drive_gas_source: str = "none"
        self.last_drive_gas_l_min: float = 0.0

    def source(self, gas: str) -> GasSource:
        norm = str(gas).strip().lower()
        if norm not in self._sources:
            choices = ", ".join(GAS_NAMES)
            raise ValueError(f"Unknown medical gas {gas!r}; choose one of: {choices}")
        return self._sources[norm]

    def set_drive_gas_mode(self, mode: str) -> None:
        norm = str(mode).strip().lower()
        if norm not in DRIVE_GAS_MODES:
            choices = ", ".join(DRIVE_GAS_MODES)
            raise ValueError(f"Unknown ventilator drive gas mode {mode!r}; choose one of: {choices}")
        self.drive_gas_mode = norm

    def resolve_drive_gas(self) -> tuple[str, bool]:
        """Return (active_drive_source, vent_drive_available)."""
        mode = self.drive_gas_mode
        if mode == "electric":
            return "electric", True
        if mode == "o2":
            if self.o2.available:
                return "o2", True
            if self.air.available:
                return "air", True
            return "none", False
        if mode == "o2_only":
            return ("o2", True) if self.o2.available else ("none", False)
        if self.air.available:
            return "air", True
        if self.o2.available:
            return "o2", True
        return "none", False

    @property
    def vent_drive_available(self) -> bool:
        """Whether pneumatic or electric drive power is available to cycle the ventilator."""
        return self.resolve_drive_gas()[1]

    @staticmethod
    def ventilator_drive_gas_demand_l_min(
        *,
        vent_is_on: bool,
        mode: str,
        rr: float,
        vt_l: float,
        ie_ratio: float,
        t_insp: float,
        pip_cmh2o: float,
        fgf_l_min: float,
        mv_supported_l_min: float | None = None,
    ) -> float:
        """Pneumatic bellows drive-gas consumption (L/min), Szpisjak 2008.

        Per breath: DG = max(0, VT - FGF * t_I) + C_circuit_vent * PIP,
        with C_circuit_vent = 5 mL/cmH2O.
        """
        if not vent_is_on or mode == "CPAP" or rr <= 0.0:
            return 0.0
        period = 60.0 / rr
        if mode.startswith("SIMV") or mode == "PSV":
            t_i = min(t_insp, 0.8 * period)
        else:
            t_i = period * ie_ratio / (1.0 + ie_ratio)
        eff_vt_l = max(0.0, vt_l)
        if mv_supported_l_min is not None and mv_supported_l_min > 0.0:
            eff_vt_l = mv_supported_l_min / rr
        fgf_insp_l = max(0.0, fgf_l_min) * (t_i / 60.0)
        dg_breath_l = max(0.0, eff_vt_l - fgf_insp_l) + CIRCUIT_VENT_COMPLIANCE_L_CMH2O * max(0.0, pip_cmh2o)
        return max(0.0, dg_breath_l * rr)

    def step(
        self,
        dt: float,
        *,
        o2_fgf: float,
        o2_flush: float,
        air_fgf: float,
        n2o_fgf: float,
        drive_demand_l_min: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """Consume supplied gases over dt; return delivered (o2_fgf, o2_flush, air_fgf, n2o_fgf)."""
        drive_src, drive_ok = self.resolve_drive_gas()
        drive_req = max(0.0, float(drive_demand_l_min)) if drive_ok and drive_src in ("o2", "air") else 0.0
        o2_drive_req = drive_req if drive_src == "o2" else 0.0
        air_drive_req = drive_req if drive_src == "air" else 0.0

        o2_avail_before = self.o2.available
        req_o2_fgf = max(0.0, float(o2_fgf))
        req_o2_flush = max(0.0, float(o2_flush)) if o2_avail_before else 0.0
        o2_total_flow_req = req_o2_fgf + req_o2_flush
        deliv_o2_total, deliv_o2_drive = self.o2.consume(dt, o2_total_flow_req, o2_drive_req)
        if o2_total_flow_req > 1e-12:
            o2_ratio = deliv_o2_total / o2_total_flow_req
            deliv_o2_fgf = req_o2_fgf * o2_ratio
            deliv_o2_flush = req_o2_flush * o2_ratio
        else:
            deliv_o2_fgf = 0.0
            deliv_o2_flush = 0.0

        deliv_air_fgf, deliv_air_drive = self.air.consume(dt, max(0.0, float(air_fgf)), air_drive_req)

        # Oxygen-pressure fail-safe and 25% O2 hypoxic guard limit N2O delivery.
        req_n2o = min(max(0.0, float(n2o_fgf)), 3.0 * deliv_o2_fgf) if o2_avail_before else 0.0
        deliv_n2o_fgf, _ = self.n2o.consume(dt, req_n2o, 0.0)

        self.last_drive_gas_source = drive_src if (drive_ok and drive_demand_l_min > 0.0) else "none"
        self.last_drive_gas_l_min = deliv_o2_drive + deliv_air_drive
        return deliv_o2_fgf, deliv_o2_flush, deliv_air_fgf, deliv_n2o_fgf
