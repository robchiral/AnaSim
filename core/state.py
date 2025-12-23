from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import numpy as np

@dataclass
class SimulationConfig:
    """Configuration for the simulation engine."""
    dt: float = 0.01  # Time step in seconds
    
    # Models
    pk_model_propofol: str = "Eleveld"
    pk_model_remi: str = "Minto"
    bis_model: str = "GrecoBouillon"
    hemo_model: str = "Su2023"
    resp_model: str = "SingleCompartment"
    pk_model_nore: str = "Li"
    loc_model: str = "Kern"
    mode: str = "awake" # 'awake' or 'steady_state'
    maint_type: str = "tiva" # 'tiva' or 'balanced'
    disturbance_profile: str = None # 'stim_intubation_pulse', 'stim_sustained_surgery'
    baseline_hb: float = 13.5
    fidelity_mode: str = "clinical"  # "clinical" (tuned realism) or "literature"
    
    # Agents
    volatile_agents: List[str] = field(default_factory=lambda: ["sevoflurane"])
    
    # Simulation settings
    simulation_speed: float = 1.0  # Real-time multiplier
    enable_death_detector: bool = False

class AirwayType(Enum):
    NONE = "None"
    MASK = "Mask"
    ETT = "ETT"

@dataclass
class SimulationState:
    """Immutable snapshot of the simulation state at a specific time."""
    time: float = 0.0
    
    # Drug Concentrations (Concentration at effect site Ce, and Plasma Cp)
    is_dead: bool = False
    death_reason: str = ""
    
    # Propofol (ug/mL)
    propofol_ce: float = 0.0
    propofol_cp: float = 0.0
    
    # Remifentanil (ng/mL)
    remi_ce: float = 0.0
    remi_cp: float = 0.0
    
    # Norepinephrine (ng/mL)
    nore_ce: float = 0.0
    
    # Epinephrine (ng/mL)
    epi_ce: float = 0.0
    
    # Phenylephrine (ng/mL)
    phenyl_ce: float = 0.0
    
    # Volatile
    fi_sevo: float = 0.0
    et_sevo: float = 0.0

    
    mac: float = 0.0

    # Physiology
    
    # Rocuronium (ug/mL)
    roc_ce: float = 0.0
    roc_cp: float = 0.0
    
    # Monitor Values
    bis: float = 98.0
    tof: float = 100.0
    loc: float = 0.0 # Probability
    tol: float = 0.0 # Probability
    map: float = 90.0  # Mean Arterial Pressure
    hr: float = 70.0   # Heart Rate
    rr: float = 12.0   # Respiratory Rate
    vt: float = 500.0  # Tidal Volume (mL)
    mv: float = 6.0    # Minute Ventilation (L/min)
    va: float = 4.0    # Alveolar ventilation (L/min)
    apnea: bool = False
    
    etco2: float = 38.0 # End-tidal CO2
    paco2: float = 40.0 # Alveolar CO2 (using paco2 name for clarity with arterial/alveolar)
    pao2: float = 95.0 # Arterial Oxygen (mmHg)
    capno_co2: float = 0.0 # Instantaneous CO2 for waveform
    
    # Mechanics
    paw: float = 0.0
    pit: float = -2.0 # Intrathoracic Pressure (mmHg)
    flow: float = 0.0
    volume: float = 0.0
    
    spo2: float = 99.0 # Saturation
    
    # NIBP
    nibp_sys: float = 120.0
    nibp_dia: float = 80.0
    nibp_map: float = 93.3
    nibp_timestamp: float = 0.0
    nibp_interval_sec: float = 300.0
    nibp_is_cycling: bool = False
    nibp_cuff_pressure: float = 0.0
    
    # Advanced Hemodynamics
    sv: float = 70.0 # Stroke Volume (mL)
    svr: float = 16.0 # Systemic Vascular Resistance (mmHg*min/L). Normal ~16-20. 
    # User specified: mmHg*min/L. Typical = 80 mmHg / 5 L/min = 16.
    # Let's use mmHg*min/L as requested. 1200 dynes is 15-20 units.
    # We'll use 16.0 as baseline.
    co: float = 5.0 # Cardiac Output (L/min)
    sbp: float = 120.0 # Systolic Blood Pressure (mmHg)
    dbp: float = 80.0 # Diastolic Blood Pressure (mmHg)
    blood_volume: float = 5000.0 # Blood volume (mL)
    hb_g_dl: float = 13.5
    hct: float = 0.42
    
    # Ventilator Settings State (Snapshot)
    fio2: float = 0.21
    
    # Temperature
    temp_c: float = 37.0
    bair_hugger_target: float = 0.0 # 0.0 means OFF. Otherwise target temp (32, 38, 43)
    oxygen_delivery_ratio: float = 1.0

    # Waveform snapshots (last N points, or current instantaneous value)
    # For efficiency, waveforms might be handled via a separate ring buffer, 
    # but we can keep instantaneous values here.
    ecg_voltage: float = 0.0
    pleth_voltage: float = 0.0
    
    # Alarms
    alarms: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    
    # Airway
    airway_mode: AirwayType = AirwayType.NONE
