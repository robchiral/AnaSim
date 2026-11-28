from __future__ import annotations

from dataclasses import dataclass

from anasim.core.enums import RhythmType


@dataclass(slots=True)
class HemoState:
    """Hemodynamic outputs and the internal model variables behind them."""

    map: float = 80.0
    hr: float = 75.0
    sv: float = 70.0
    # Peripheral pulse amplitude, independent of mean pressure and flow.
    pulse_pressure_factor: float = 1.0
    svr: float = 16.0
    co: float = 5.25
    rhythm_type: RhythmType = RhythmType.SINUS
    tpr: float = 0.016
    sv_star: float = 82.2
    hr_star: float = 56.0
    tde_sv: float = 7.4
    tde_hr: float = 6.7
    ce_sevo: float = 0.0
    mcfp: float = 0.0
    rap: float = 0.0
    lap: float = 0.0
    pvr: float = 0.0
    rv_co: float = 0.0
    lv_inflow: float = 0.0
    preload_factor: float = 1.0
    # During chest compressions, co and map include compression flow/pulses; hr,
    # sv, intrinsic_co, and intrinsic_map are the heart's own, while perfusion_map
    # is the forward trans-circuit pressure driving organ perfusion.
    compressions: bool = False
    intrinsic_co: float = 5.25
    intrinsic_map: float = 80.0
    perfusion_map: float = 80.0
    cpr_perfusion_map: float = 0.0
    # Right atrial pressure, or mean systemic filling pressure without flow.
    venous_pressure: float = 0.0
    # Separated flows for parallel extracorporeal support (VA ECMO, CPB):
    # pulmonary_flow crosses the native lungs; pump_flow is extracorporeal
    # arterial return; co (= systemic_flow) is total systemic perfusion.
    pulmonary_flow: float = 5.25
    pump_flow: float = 0.0
    nonpulsatile_map: float = 0.0
    paced_capture: bool = False
    paced_mode: str = "off"
