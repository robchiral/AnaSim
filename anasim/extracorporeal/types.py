"""Types, enums, configuration, and public telemetry for extracorporeal support."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExtracorporealMode(Enum):
    """Active extracorporeal circuit modality."""

    NONE = "none"
    VV_ECMO = "vv_ecmo"
    VA_ECMO = "va_ecmo"
    CPB = "cpb"


class CannulationType(Enum):
    """Vascular access configuration governing recirculation or aortic watershed."""

    FEM_JUG = "fem_jug"  # VV ECMO femoro-jugular (standard lower recirculation)
    FEM_FEM = "fem_fem"  # VV ECMO femoro-femoral (higher recirculation)
    DUAL_LUMEN = "dual_lumen"  # VV ECMO bicaval dual-lumen (lowest recirculation)
    PERIPHERAL_FEM = "peripheral_fem"  # Peripheral femoral VA ECMO (Harlequin watershed)
    CENTRAL = "central"  # Central VA ECMO or CPB ascending aortic return


class MembraneType(Enum):
    """Oxygenator membrane wall construction (Prasser 2008)."""

    PMP = "pmp"  # Plasma-tight polymethylpentene (standard ECMO; minimal volatile transfer)
    MICROPOROUS = "microporous"  # Microporous polypropylene (standard CPB; full volatile transfer)


class BloodGasManagement(Enum):
    """Hypothermic blood-gas management strategy during CPB (Murkin 1987)."""

    ALPHA_STAT = "alpha_stat"  # Maintain 37 °C PaCO2 ~40 mmHg; preserves CBF-CMRO2 coupling
    PH_STAT = "ph_stat"  # Maintain temperature-corrected PaCO2 ~40 mmHg; vasodilates CBF


@dataclass(frozen=True, slots=True)
class ExtracorporealConfig:
    """Physical defaults for ECMO and cardiopulmonary bypass circuits."""

    ecmo_prime_volume_ml: float = 600.0  # Closed adult ECMO circuit prime
    cpb_prime_volume_ml: float = 1400.0  # Adult CPB circuit + initial reservoir prime
    cpb_reservoir_low_ml: float = 300.0  # Low-level safety threshold (mL)
    cpb_reservoir_max_ml: float = 4000.0
    cpb_drain_conductance_l_min_mmhg: float = 0.70  # Gravity/VAVD venous drainage conductance
    cpb_drain_siphon_mmhg: float = 8.0  # Negative siphon pressure head on open venous line
    hx_effectiveness: float = 0.62  # Blood-water heat exchanger thermal effectiveness
    hx_max_gradient_c: float = 10.0  # Clinical safety cap on water-to-blood gradient (°C)
    default_membrane_area_m2: float = 1.8  # Standard adult hollow-fiber PMP/PPL oxygenator


@dataclass(slots=True)
class ExtracorporealState:
    """Public state and telemetry for the extracorporeal circuit."""

    mode: ExtracorporealMode = ExtracorporealMode.NONE
    active: bool = False
    cannulation: CannulationType = CannulationType.FEM_JUG
    membrane_type: MembraneType = MembraneType.PMP
    blood_gas_strategy: BloodGasManagement = BloodGasManagement.ALPHA_STAT

    # Operator settings
    target_blood_flow_l_min: float = 0.0
    blood_flow_l_min: float = 0.0
    sweep_gas_l_min: float = 0.0
    sweep_fio2: float = 1.0
    sweep_fco2: float = 0.0
    sweep_sevo_percent: float = 0.0
    membrane_area_m2: float = 1.8
    heater_cooler_on: bool = False
    water_target_c: float = 37.0
    venous_clamp_open: float = 1.0  # CPB drainage clamp, 0-1
    aortic_cross_clamp: bool = False
    cardioplegia_arrest: bool = False
    lv_vent_l_min: float = 0.0
    bypass_alarm_mode: bool = False

    # Circuit volumes and drainage
    primed_volume_ml: float = 0.0
    reservoir_volume_ml: float = 0.0
    venous_drainage_l_min: float = 0.0
    reservoir_low_alarm: bool = False

    # Gas exchange telemetry
    recirculation_fraction: float = 0.0
    effective_blood_flow_l_min: float = 0.0
    o2_transfer_ml_min: float = 0.0
    co2_removal_ml_min: float = 0.0
    pre_oxy_po2: float = 40.0
    pre_oxy_so2: float = 75.0
    post_oxy_po2: float = 400.0
    post_oxy_so2: float = 100.0
    exhaust_sevo_percent: float = 0.0

    # VA ECMO differential oxygenation / aortic watershed
    upper_body_sao2: float = 98.0
    upper_body_pao2: float = 95.0
    upper_body_paco2_37: float = 40.0
    upper_body_ph_37: float = 7.4
    coronary_sao2: float = 98.0
    lower_body_sao2: float = 98.0
    lower_body_pao2: float = 95.0
    lower_body_paco2_37: float = 40.0
    lower_body_ph_37: float = 7.4


@dataclass(frozen=True, slots=True)
class OxygenatorPassResult:
    """Solved blood and gas outputs for one pass through the membrane lung."""

    co2_out_mmol_l: float
    o2_out_l_l: float
    pco2_out_37: float
    ph_out_37: float
    po2_out: float
    so2_out: float
    o2_transfer_l_min: float  # STPD L/min added to blood
    co2_removal_l_min: float  # STPD L/min removed from blood
    pre_po2: float
    pre_so2: float
    post_po2: float
    post_so2: float
    pre_pco2_37: float = 46.0
    pre_ph_37: float = 7.36
