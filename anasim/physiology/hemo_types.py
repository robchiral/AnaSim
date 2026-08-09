from __future__ import annotations

from dataclasses import dataclass

from anasim.core.enums import RhythmType


@dataclass
class HemoState:
    """User-facing hemodynamic snapshot."""

    map: float = 80.0
    hr: float = 75.0
    sv: float = 70.0
    svr: float = 16.0
    co: float = 5.25
    rhythm_type: RhythmType = RhythmType.SINUS


@dataclass
class HemoStateExtended(HemoState):
    """Extended hemodynamic state with internal model variables."""

    tpr: float = 0.016
    sv_star: float = 82.2
    hr_star: float = 56.0
    tde_sv: float = 7.4
    tde_hr: float = 6.7
    ce_sevo: float = 0.0
    mcfp: float = 0.0
    rap: float = 0.0
    pvr: float = 0.0
    rv_co: float = 0.0
    lv_inflow: float = 0.0
    preload_factor: float = 1.0
