from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, NamedTuple, Optional

from anasim.patient.domain import finite_number

SUPPORTED_MODEL_OPTIONS = {
    "pk_model_propofol": {"Marsh", "Schnider", "Eleveld"},
    "pk_model_remi": {"Minto"},
    "bis_model": {"Bouillon", "Eleveld", "Fuentes", "Yumuk"},
    "hemo_model": {"Su2023"},
    "resp_model": {"SingleCompartment"},
    "pk_model_nore": {"Beloeil", "Li"},
    "pk_model_epi": {"HealthyAdult", "Abboud"},
    "loc_model": {"Kern", "Mertens", "Johnson"},
    "mode": {"awake", "steady_state"},
    "maint_type": {"tiva", "balanced"},
}
SUPPORTED_VOLATILE_AGENTS = {"sevoflurane"}


@dataclass
class SimulationConfig:
    """Configuration for SimulationEngine; see docs/CLI_USAGE.md."""

    dt: float = 0.01  # s

    pk_model_propofol: str = "Eleveld"
    pk_model_remi: str = "Minto"
    bis_model: str = "Bouillon"
    hemo_model: str = "Su2023"
    resp_model: str = "SingleCompartment"
    pk_model_nore: str = "Li"
    pk_model_epi: str = "HealthyAdult"
    loc_model: str = "Kern"
    mode: str = "awake"
    maint_type: str = "tiva"
    disturbance_profile: str = None
    # An empty list disables the vaporizer.
    volatile_agents: List[str] = field(default_factory=lambda: ["sevoflurane"])
    # None gives 1 mL/kg/hr.
    maintenance_fluid_ml_hr: Optional[float] = None

    simulation_speed: float = 1.0  # Multiple of real time
    end_on_cardiac_arrest: bool = False
    arterial_line_enabled: bool = True
    rng_seed: Optional[int] = None

    def __post_init__(self):
        self.dt = finite_number("dt", self.dt)
        if self.dt <= 0:
            raise ValueError("dt must be greater than zero")
        for field_name, supported in SUPPORTED_MODEL_OPTIONS.items():
            value = getattr(self, field_name)
            if not isinstance(value, str) or value not in supported:
                choices = ", ".join(sorted(supported))
                raise ValueError(f"{field_name}={value!r} is unsupported; choose one of: {choices}")
        if not isinstance(self.volatile_agents, list) or any(
            not isinstance(agent, str) for agent in self.volatile_agents
        ):
            raise ValueError("volatile_agents must be a list of agent names")
        unsupported_agents = set(self.volatile_agents) - SUPPORTED_VOLATILE_AGENTS
        if unsupported_agents:
            names = ", ".join(sorted(unsupported_agents))
            raise ValueError(f"Unsupported volatile agent(s): {names}")


class WaveformSample(NamedTuple):
    """One simulation step of the monitor waveforms."""

    time: float
    ecg_voltage: float
    pleth_voltage: float
    capno_co2: float
    art_pressure: float


class AirwayType(Enum):
    NONE = "None"
    MASK = "Mask"
    ETT = "ETT"


@dataclass(slots=True)
class SimulationState:
    """Public snapshot of the simulation at one time. See docs/ARCHITECTURE.md."""

    time: float = 0.0
    cardiac_arrest: bool = False
    arrest_reason: str = ""

    # Effect-site (ce) and plasma (cp) concentrations: propofol mcg/mL,
    # vasopressin mU/L, others ng/mL.
    propofol_ce: float = 0.0
    propofol_cp: float = 0.0
    remi_ce: float = 0.0
    remi_cp: float = 0.0
    nore_ce: float = 0.0
    epi_ce: float = 0.0
    phenyl_ce: float = 0.0
    vaso_ce: float = 0.0
    dobu_ce: float = 0.0
    mil_ce: float = 0.0

    # Inhaled agents: fractions in %, MAC as age-adjusted multiples.
    fi_sevo: float = 0.0
    et_sevo: float = 0.0
    mac_sevo: float = 0.0
    fi_n2o: float = 0.0
    et_n2o: float = 0.0
    mac_n2o: float = 0.0
    mac: float = 0.0  # Brain MAC, summed across agents; drives drug effects
    et_mac: float = 0.0  # End-tidal MAC shown by the gas monitor

    # Rocuronium (mcg/mL): free drug at the adductor pollicis, and total plasma.
    roc_ce: float = 0.0
    roc_cp: float = 0.0

    bis: float = 98.0
    display_bis: float = 98.0
    tof: float = 100.0  # %
    loc: float = 0.0  # Probability of unconsciousness
    tol: float = 0.0  # Probability of tolerating laryngoscopy
    map: float = 90.0
    hr: float = 70.0
    display_hr: float = 70.0

    # Ventilation: VT mL, MV and VA L/min, gas tensions mmHg.
    rr: float = 12.0
    vt: float = 500.0
    mv: float = 6.0
    va: float = 4.0
    apnea: bool = False
    etco2: float = 38.0
    display_etco2: float = 38.0
    etco2_signal_valid: bool = False
    pa_co2: float = 40.0
    alveolar_co2: float = 40.0
    pao2: float = 95.0
    capno_co2: float = 0.0

    # Airway pressure cmH2O, intrathoracic pressure mmHg, flow L/min, volume L.
    paw: float = 0.0
    pit: float = -2.0
    flow: float = 0.0
    volume: float = 0.0

    # spo2 is the pulse oximeter reading; sao2 is arterial saturation.
    spo2: float = 99.0
    display_spo2: float = 99.0
    spo2_signal_valid: bool = True
    sao2: float = 98.0

    nibp_sys: float = 120.0
    nibp_dia: float = 80.0
    nibp_map: float = 93.3
    nibp_timestamp: Optional[float] = None
    nibp_interval_sec: float = 300.0
    nibp_is_cycling: bool = False
    nibp_cuff_pressure: float = 0.0

    # MAP, HR, SV, SVR, and CO come from Su; sbp and dbp are the ideal pulse
    # landmarks. SV mL, SVR Wood units, CO L/min, pressures mmHg.
    sv: float = 70.0
    svr: float = 16.0
    co: float = 5.0
    sbp: float = 120.0
    dbp: float = 80.0
    # Arterial catheter output and completed-beat numerics.
    art_pressure: float = 80.0
    art_sbp: float = 120.0
    art_dbp: float = 80.0
    art_map: float = 93.3
    blood_volume: float = 5000.0  # mL
    hb_g_dl: float = 13.5
    hct: float = 0.42

    # Cumulative fluid balance (mL).
    fluid_in_ml: float = 0.0
    colloid_in_ml: float = 0.0
    blood_in_ml: float = 0.0
    urine_out_ml: float = 0.0
    blood_out_ml: float = 0.0
    net_fluid_ml: float = 0.0

    fio2: float = 0.21
    temp_c: float = 37.0
    bair_hugger_target: float = 0.0  # °C; 0 is off
    oxygen_delivery_ratio: float = 1.0
    shivering: float = 0.0

    # Instantaneous waveform values; history is engine.output_buffer.
    ecg_voltage: float = 0.0
    pleth_voltage: float = 0.0

    alarms: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    # Airway severities, 0-1.
    airway_mode: AirwayType = AirwayType.NONE
    airway_obstruction: float = 0.0
    bronchospasm: float = 0.0
    laryngospasm: float = 0.0

    def monitored_blood_pressure(
        self,
        arterial_line_enabled: bool,
    ) -> tuple[float, float, float]:
        """Return systolic, diastolic, and mean pressure from the active monitor."""
        if arterial_line_enabled:
            return self.art_sbp, self.art_dbp, self.art_map
        return self.nibp_sys, self.nibp_dia, self.nibp_map
