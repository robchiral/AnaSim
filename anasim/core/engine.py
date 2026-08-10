import copy
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from anasim.core.constants import (
    AirwayTuning,
    ThermalTuning,
)
from anasim.core.enums import RhythmType
from anasim.core.recorder import DataRecorder
from anasim.core.utils import clamp
from anasim.machine.circuit import CircleSystem
from anasim.machine.ventilator import AnesthesiaVentilator
from anasim.machine.volatile import Vaporizer
from anasim.monitors.alarms import AlarmSystem
from anasim.monitors.arterial import ArterialLineMonitor, ArterialWaveformRenderer
from anasim.monitors.capno import Capnograph
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.monitors.ecg import ECGMonitor
from anasim.monitors.nibp import NIBPMonitor
from anasim.monitors.spo2 import SpO2Monitor
from anasim.patient.patient import Patient
from anasim.patient.pd import BISModel, LOCModel, TOFModel, TOLModel
from anasim.patient.pk_models import (
    DobutaminePK,
    EpinephrinePK,
    MilrinonePK,
    NorepinephrinePK,
    PhenylephrinePK,
    PropofolPKEleveld,
    PropofolPKMarsh,
    PropofolPKSchnider,
    RemifentanilPKMinto,
    RocuroniumPK,
    VasopressinPK,
)
from anasim.patient.volatile_pk import VolatilePK
from anasim.physiology.disturbances import Disturbances
from anasim.physiology.hemodynamics import HemodynamicModel
from anasim.physiology.resp_mech import RespiratoryMechanics
from anasim.physiology.respiration import RespiratoryModel

from . import monitors as monitor_core
from . import projection as projection_core
from . import runtime as runtime_core
from .action_log import (
    ACTION_AIRWAY,
    ACTION_BAG_MASK,
    ACTION_DRUG_BOLUS,
    ACTION_EVENT_START,
    ACTION_EVENT_STOP,
    ACTION_FGF,
    ACTION_FLUID,
    ACTION_OXYGEN_SUPPLY,
    ACTION_VAPORIZER,
    ActionLog,
)
from .drug_api import DrugControllerMixin
from .drug_registry import DRUG_REGISTRY, resolve_bolus_drug
from .initialization import initialize_engine_state
from .state import AirwayType, SimulationConfig, SimulationState

AIRWAY_MODE_MAP = {
    "None": AirwayType.NONE,
    "Mask": AirwayType.MASK,
    "ETT": AirwayType.ETT,
}

@dataclass(slots=True)
class PendingInfusion:
    remaining_ml: float
    rate_ml_min: float
    hematocrit: float = 0.0
    retention_fraction: Optional[float] = None
    label: str = "crystalloid"


PROPOFOL_MODELS = {
    "Marsh": PropofolPKMarsh,
    "Schnider": PropofolPKSchnider,
    "Eleveld": PropofolPKEleveld,
}

REMI_MODELS = {
    "minto": RemifentanilPKMinto,
}

VOLATILE_AGENT_PARAMS = {
    "sevoflurane": {"name": "Sevoflurane", "lambda_b_g": 0.65, "mac_40": 2.1},
}

VOLATILE_AGENT_ALIASES = {
    "sevoflurane": "sevoflurane",
    "sevo": "sevoflurane",
}

# Nitrous oxide (N2O) parameters (inhaled gas, not vaporizer-controlled).
# References:
# - Table of partition coefficients at 37C: blood:gas 0.47, brain:blood 1.1,
#   muscle:blood 1.2, fat:blood 2.3.
# - Human MAC reported ~1.04 atm (104%) at 1 atm absolute.
N2O_PARAMS = {
    "name": "Nitrous Oxide",
    "lambda_b_g": 0.47,
    "mac_40": 104.0,
    # Tissue:blood partition coefficients are near unity; use conservative defaults.
    "lambda_t_b_vrg": 1.1,
    "lambda_t_b_mus": 1.2,
    "lambda_t_b_fat": 2.3,
}

NORE_PD_PARAMS = {
    "Beloeil": (7.04, 98.7, 1.8),
    "Li": (5.4, 98.7, 1.8),
    "Oualha": (7.04, 98.7, 1.8),
}


def _normalize_model_key(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip().lower()


def _resolve_volatile_agent(agent: Optional[str]) -> Optional[str]:
    if not agent:
        return None
    key = _normalize_model_key(agent)
    return VOLATILE_AGENT_ALIASES.get(key)

PK_MODEL_ATTRS = tuple(spec.pk_attr for spec in DRUG_REGISTRY) + (
    "pk_sevo",
    "pk_n2o",
)
PHYSIO_MODEL_ATTRS = ("hemo", "resp", "resp_mech")
MACHINE_ATTRS = ("circuit", "vaporizer", "vent")
MONITOR_ATTRS = (
    "bis",
    "capno",
    "loc_pd",
    "tol_pd",
    "tof_pd",
    "cardiac_cycle",
    "arterial_waveform",
    "art_line",
    "ecg",
    "spo2_mon",
    "nibp",
)
RATE_ATTRS = tuple(spec.rate_attr for spec in DRUG_REGISTRY)
TCI_ATTRS = tuple(spec.tci_attr for spec in DRUG_REGISTRY)


class SimulationEngine(DrugControllerMixin):
    """
    Main simulation orchestrator.
    Manages time, updates subsystems, and produces state snapshots.
    
    State management:
    - `self.state` is the public snapshot for UI/tests.
    - Subsystems keep their own internal state and project into `self.state`
      through the projection and monitor modules.
    """
    def __init__(self, patient: Patient, config: SimulationConfig):
        self.patient = patient
        self.config = config
        self.state = SimulationState()

        # Timestamped control actions (scenario objectives, debrief).
        self.actions = ActionLog()

        # Tuning knobs (centralized defaults).
        self.airway_tuning = AirwayTuning()
        self.thermal_tuning = ThermalTuning()
        
        # Subsystems.
        for attr in PK_MODEL_ATTRS:
            setattr(self, attr, None)
        self.active_agent = "Sevoflurane"
        self._volatile_enabled = True
        self._volatile_agent_key = "sevoflurane"
        for attr in PHYSIO_MODEL_ATTRS:
            setattr(self, attr, None)
        
        # Machine.
        for attr in MACHINE_ATTRS:
            setattr(self, attr, None)
        
        # Monitors.
        for attr in MONITOR_ATTRS:
            setattr(self, attr, None)
        self._next_nibp_time = 0.0
        
        # Controls (basic TIVA).
        for attr in RATE_ATTRS:
            setattr(self, attr, 0.0)
        
        # TCI controllers.
        for attr in TCI_ATTRS:
            setattr(self, attr, None)
        
        # Disturbances & alarms.
        self.disturbances = Disturbances(config.disturbance_profile)
        self.disturbance_profile = config.disturbance_profile
        self.disturbance_active = bool(config.disturbance_profile)
        self.disturbance_start_time = 0.0
        self.alarms = AlarmSystem(dt=config.dt)
        
        # Output history for the ten-second monitor sweep.
        self._output_window_s = 10.0
        self.output_buffer = deque()

        # Optional CSV recorder.
        self.recorder = None

        # TCI timing accumulators (per-controller).
        self._tci_accumulators = {}
        
        # Control flags.
        self.running = False
        
        # Bag-mask ventilation (manual PPV, separate from mechanical vent).
        self.bag_mask_active = False
        self.bag_mask_rr = 12.0   # breaths per min (typical for manual PPV)
        self.bag_mask_vt = 0.5    # Liters (~500mL)

        # PSV/CPAP apnea backup (if RR configured).
        self.psv_apnea_backup_delay = 20.0  # seconds before backup activates
        self._psv_apnea_timer = 0.0
        
        # Event flags.
        self.active_hemorrhage = False
        self.active_anaphylaxis = False
        self.hemorrhage_rate_ml_min = 500.0
        
        # Pending fluid infusion (realistic timing).
        self.pending_infusions = []
        self.fluid_infusion_rate_ml_min = 150.0  # mL/min (moderate rate with pressure)
        self.blood_infusion_rate_ml_min = 75.0   # mL/min slower for PRBCs
        self._maintenance_override_ml_hr = self.config.maintenance_fluid_ml_hr
        self.maintenance_fluid_rate_ml_min = 0.0
        
        # Anaphylaxis (gradual onset/offset).
        self.anaphylaxis_severity = 0.0  # 0 to 1 (1 = full severity)
        self.anaphylaxis_onset_rate = 0.5 / 60.0  # per second (reaches 1.0 in ~2 min)
        self.anaphylaxis_decay_rate = 0.1 / 60.0  # per second (decays ~10 min)

        # Sepsis / distributive shock (gradual onset/offset).
        self.active_sepsis = False
        self.sepsis_severity = 0.0  # 0 to 1 (1 = full severity)
        self.sepsis_onset_rate = 0.1 / 60.0  # per second (~10 min to full)
        self.sepsis_decay_rate = 0.03 / 60.0  # per second (~30+ min recovery)

        # Airway complications (manual + auto-triggered).
        self.auto_laryngospasm_enabled = True
        self.airway_obstruction_manual = 0.0
        self.bronchospasm_manual = 0.0
        self.laryngospasm_severity = 0.0
        self.laryngospasm_tau_on = self.airway_tuning.laryngospasm_tau_on   # seconds (fast onset)
        self.laryngospasm_tau_off = self.airway_tuning.laryngospasm_tau_off  # seconds (slower relief)
        self._airway_patency = 1.0
        self._ventilation_efficiency = 1.0
        self._capno_obstruction = 0.0
        self._vq_mismatch = 0.0
        self._base_airway_resistance = 10.0
        
        # Monitor state.
        self.smooth_bis = 98.0
        self._monitor_tau_bis_s = 2.0
        self._capno_numeric_peak = 0.0
        self._capno_numeric_age_s = 0.0
        self._capno_numeric_timeout_s = 15.0
        self._capno_last_phase = "EXP"
        self._capno_has_sample = False
        self._mean_paw_tau_s = 0.25
        self._tol_current = None
        self._pk_hemo_scale_cache = None
        
        # Helpers.
        self.current_mean_paw = 5.0
        self._last_patient_effort_cmH2O = 0.0
        
        # Random number generator for noise.
        self.rng = np.random.default_rng(self.config.rng_seed)
        # Dedicated RNG for capnography to keep output reproducible without
        # coupling to monitor noise draws.
        self._capno_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        # Dedicated RNG for ECG to keep rhythm noise reproducible.
        self._ecg_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        # Dedicated RNG for NIBP cycling/failure behavior.
        self._nibp_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        # Dedicated RNG for beat-level rhythm variability.
        self._cardiac_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        self._bis_noise_std = 0.2
        self._monitor_values = {
            "BIS": 0.0,
            "MAP": 0.0,
            "HR": 0.0,
            "EtCO2": 0.0,
            "SpO2": 0.0,
        }
        self._remi_rate_weight = self.patient.weight
        self._remi_rate_scale = 60.0 / self._remi_rate_weight
        
        # Thermal model state.
        self.heat_production_basal = 0.0 # W
        self.specific_heat = self.thermal_tuning.specific_heat_j_kg_k # J/(kg K)
        self.surface_area = 1.9 # m^2 (default, updated in init)
        self._last_depth_index = 0.0 # For temperature redistribution calculation
        self._shiver_level = 0.0
        self._cached_temp_metabolic = 37.0
        self._cached_temp_metabolic_factor = 1.0
        self._metabolic_factor = 1.0
        self._vent_active = False
        self._do2_ratio = 1.0
        
        # Viability timers (death detector).
        self.time_brady = 0.0
        self.time_hypotension = 0.0
        self.time_tachy = 0.0
        self.DEATH_GRACE_PERIOD = 15.0 # seconds
        
        self.initialize_models()
        self._configure_maintenance_fluids()
        self.initialize_state()

    def _append_output_snapshot(self) -> None:
        """Append public state and retain ten seconds by timestamp."""
        self.output_buffer.append(copy.copy(self.state))
        cutoff = self.state.time - self._output_window_s
        while len(self.output_buffer) > 1 and self.output_buffer[0].time < cutoff:
            self.output_buffer.popleft()

    def initialize_state(self):
        """Set initial state from live subsystem state instead of placeholder defaults."""
        self.state.temp_c = self.patient.baseline_temp
        self._tol_current = None
        self._airway_patency = 1.0
        self._ventilation_efficiency = 1.0
        self._capno_obstruction = 0.0
        self._vq_mismatch = 0.0

        initialize_engine_state(self)
        projection_core.sync_state_from_models(self)
        monitor_core.seed_nibp_reading(self)
        self._append_output_snapshot()

    def _configure_maintenance_fluids(self):
        """Set continuous IV fluids to 1 mL/kg/hr unless overridden."""
        override = self._maintenance_override_ml_hr
        if override is None:
            self.maintenance_fluid_rate_ml_min = self.patient.weight / 60.0
        else:
            self.maintenance_fluid_rate_ml_min = max(0.0, float(override) / 60.0)

    def set_continuous_fluid_rate(self, ml_hr: Optional[float]):
        """
        Set continuous IV fluid rate (mL/hr).
        Pass None to revert to default (1 mL/kg/hr).
        """
        if ml_hr is None:
            self._maintenance_override_ml_hr = None
            self._configure_maintenance_fluids()
            return
        val = float(ml_hr)
        self._maintenance_override_ml_hr = max(0.0, val)
        self.maintenance_fluid_rate_ml_min = self._maintenance_override_ml_hr / 60.0

    def get_continuous_fluid_rate(self) -> float:
        """Return current continuous IV fluid rate in mL/hr."""
        return max(0.0, self.maintenance_fluid_rate_ml_min * 60.0)
        
    def initialize_models(self):
        """Initialize PK/PD models based on config."""
        self.pk_prop = PROPOFOL_MODELS[self.config.pk_model_propofol](self.patient)
            
        remi_key = _normalize_model_key(self.config.pk_model_remi)
        self.pk_remi = REMI_MODELS[remi_key](self.patient)
        
        # Machine
        self.circuit = CircleSystem()
        self.vent = AnesthesiaVentilator()

        requested_agents = self.config.volatile_agents or []
        if not requested_agents:
            self._volatile_enabled = False
            agent_key = "sevoflurane"
        else:
            self._volatile_enabled = True
            agent_key = requested_agents[0]

        agent_params = VOLATILE_AGENT_PARAMS[agent_key]
        self._volatile_agent_key = agent_key
        self.active_agent = agent_params["name"]
        self.vaporizer = Vaporizer(agent=agent_params["name"])
        self.pk_sevo = VolatilePK(
            self.patient,
            agent_params["name"],
            lambda_b_g=agent_params["lambda_b_g"],
            mac_40=agent_params["mac_40"],
        )
        # N2O is delivered via fresh gas (not vaporizer); always initialize PK.
        self.pk_n2o = VolatilePK(
            self.patient,
            N2O_PARAMS["name"],
            lambda_b_g=N2O_PARAMS["lambda_b_g"],
            lambda_t_b_vrg=N2O_PARAMS["lambda_t_b_vrg"],
            mac_40=N2O_PARAMS["mac_40"],
            lambda_t_b_mus=N2O_PARAMS["lambda_t_b_mus"],
            lambda_t_b_fat=N2O_PARAMS["lambda_t_b_fat"],
        )

        self.circuit.vaporizer_agent = agent_params["name"]
        self.circuit.vaporizer_setting = 0.0
        self.circuit.vaporizer_on = False
        if not self._volatile_enabled:
            self.vaporizer.set_concentration(0.0)

        # Hemodynamics.
        self.hemo = HemodynamicModel(self.patient)
        # Configure norepinephrine PD.
        c50, emax, gamma = NORE_PD_PARAMS[self.config.pk_model_nore]
        self.hemo.set_nore_pd(c50=c50, emax=emax, gamma=gamma)
        self.resp = RespiratoryModel(self.patient)
        self.resp_mech = RespiratoryMechanics()
        self._base_airway_resistance = self.resp_mech.resistance
        # Keep ventilator settings and is_on state consistent on init.
        self.set_vent_settings(rr=0.0, vt=0.0, peep=0.0, ie="1:2", mode="VCV")
        # Align respiratory baseline CO with hemodynamic baseline to avoid perfusion bias.
        self.resp.baseline_co_l_min = self.hemo.base_co_l_min
        
        # Monitors.
        self.bis = BISModel(self.patient, model_name=self.config.bis_model)
        self.capno = Capnograph(rng=self._capno_rng)
        self.loc_pd = LOCModel(model_name=self.config.loc_model)
        self.tol_pd = TOLModel()
        self.tof_pd = TOFModel(
            self.patient,
            model_name="Wierda",
        ) # Default Rocuronium Model

        self.cardiac_cycle = CardiacCycle(rng=self._cardiac_rng)
        self.arterial_waveform = ArterialWaveformRenderer(self.patient.age)
        self.art_line = ArterialLineMonitor()
        self.ecg = ECGMonitor(rng=self._ecg_rng)
        self.spo2_mon = SpO2Monitor()
        self.nibp = NIBPMonitor(interval_min=5.0, rng=self._nibp_rng)
        self.state.nibp_interval_sec = self.nibp.interval
        
        # Additional PK models.
        self.pk_nore = NorepinephrinePK(self.patient, model=self.config.pk_model_nore)
        self.pk_roc = RocuroniumPK(self.patient, model_name="Wierda")
        self.pk_epi = EpinephrinePK(self.patient, model=self.config.pk_model_epi)

        self.pk_phenyl = PhenylephrinePK(self.patient)
        self.pk_vaso = VasopressinPK(self.patient)
        self.pk_dobu = DobutaminePK(self.patient)
        self.pk_mil = MilrinonePK(self.patient)
        
        # Init thermal params.
        self.heat_production_basal = self.patient.weight * 1.0
        self.surface_area = self.patient.bsa

    def set_fgf(self, o2_l_min: float, air_l_min: float, n2o_l_min: float = 0.0):
        """Set Fresh Gas Flow."""
        for gas, requested in (
            ("o2", o2_l_min),
            ("air", air_l_min),
            ("n2o", n2o_l_min),
        ):
            attr = f"fgf_{gas}"
            flow = max(0.0, requested)
            if getattr(self.circuit, attr) != flow:
                self.actions.record(self.state.time, ACTION_FGF, label=gas, amount=flow)
            setattr(self.circuit, attr, flow)

    def set_oxygen_supply_connected(self, connected: bool):
        """Connect or isolate the oxygen source feeding the flowmeter."""
        connected = bool(connected)
        self.circuit.oxygen_supply_connected = connected
        self.actions.record(
            self.state.time,
            ACTION_OXYGEN_SUPPLY,
            label="connected" if connected else "disconnected",
        )

    def set_vaporizer(self, agent: str, percent: float):
        """Set Vaporizer Agent and Dial."""
        if not self._volatile_enabled:
            self.vaporizer.set_concentration(0.0)
            self.circuit.vaporizer_setting = 0.0
            self.circuit.vaporizer_on = False
            return

        resolved = _resolve_volatile_agent(agent)
        if resolved is None:
            raise ValueError(f"Unsupported volatile agent: {agent!r}")

        agent_params = VOLATILE_AGENT_PARAMS[resolved]
        self._volatile_agent_key = resolved
        self.active_agent = agent_params["name"]
        self.vaporizer.state.agent = agent_params["name"]
        self.vaporizer.set_concentration(percent)
        self.circuit.vaporizer_agent = agent_params["name"]
        self.circuit.vaporizer_setting = self.vaporizer.state.setting
        self.circuit.vaporizer_on = self.vaporizer.state.is_on
        self.actions.record(
            self.state.time,
            ACTION_VAPORIZER,
            label=self.active_agent,
            amount=self.circuit.vaporizer_setting,
        )

    def give_fluid(self, volume_ml: float):
        """
        Queue fluid bolus for infusion.
        
        Fluids are infused over time at a realistic rate rather than
        delivered instantly. A 500mL bolus takes ~3-5 minutes.
        
        Args:
            volume_ml: Volume to infuse (mL)
        """
        self._queue_infusion(volume_ml, self.fluid_infusion_rate_ml_min, hematocrit=0.0)

    def give_blood(self, volume_ml: float = 300.0, hematocrit: float = 0.55):
        """
        Queue packed RBC transfusion.
        Delivered over several minutes similar to rapid infuser.
        """
        self._queue_infusion(
            volume_ml,
            self.blood_infusion_rate_ml_min,
            hematocrit=hematocrit,
            label="blood",
        )

    def give_albumin(self, volume_ml: float):
        """
        Queue albumin (colloid) infusion.
        Retention is higher than crystalloid.
        """
        self._queue_infusion(
            volume_ml,
            self.fluid_infusion_rate_ml_min,
            hematocrit=0.0,
            retention_fraction=self.hemo.colloid_retention_fraction,
            label="colloid",
        )

    def _queue_infusion(
        self,
        volume_ml: float,
        rate_ml_min: float,
        hematocrit: float,
        retention_fraction: float = None,
        label: str = "crystalloid",
    ):
        if volume_ml <= 0 or rate_ml_min <= 0:
            return
        self.pending_infusions.append(
            PendingInfusion(
                remaining_ml=volume_ml,
                rate_ml_min=rate_ml_min,
                hematocrit=hematocrit,
                retention_fraction=retention_fraction,
                label=label,
            )
        )
        self.actions.record(self.state.time, ACTION_FLUID, label=label, amount=volume_ml)

    def give_drug_bolus(self, drug_name: str, amount: float):
        """
        Administer a drug bolus in the unit declared by its registry entry.

        Updates PK state instantaneously: New_C1 = Old_C1 + Dose/V1.
        Sugammadex is a separate bolus-only reversal routed to the TOF model.
        """
        amount = float(amount)
        if amount <= 0:
            return

        if drug_name.strip().casefold() == "sugammadex":
            # Sugammadex bolus (mg) routes to TOF model for binding.
            if self.tof_pd:
                self.tof_pd.give_sugammadex(amount)
            self.actions.record(
                self.state.time, ACTION_DRUG_BOLUS, label="sugammadex", amount=amount
            )
            return

        spec = resolve_bolus_drug(drug_name)
        model = getattr(self, spec.pk_attr)
        dose = amount * spec.bolus_model_scale
        model.state.c1 += dose / model.v1
        self.sync_active_tci_from_pk(spec.key)
        self.actions.record(
            self.state.time, ACTION_DRUG_BOLUS, label=spec.key, amount=amount
        )

    def _set_event(self, name: str, attr: str, active: bool, amount: float = 0.0):
        """Set a clinical event flag and log learner-triggered transitions."""
        if getattr(self, attr) != active:
            action = ACTION_EVENT_START if active else ACTION_EVENT_STOP
            self.actions.record(self.state.time, action, label=name, amount=amount)
        setattr(self, attr, active)

    def start_hemorrhage(self, rate_ml_min: float = 500.0):
        self.hemorrhage_rate_ml_min = rate_ml_min
        self._set_event("hemorrhage", "active_hemorrhage", True, amount=rate_ml_min)

    def stop_hemorrhage(self):
        self._set_event("hemorrhage", "active_hemorrhage", False)

    def start_anaphylaxis(self):
        self._set_event("anaphylaxis", "active_anaphylaxis", True)

    def stop_anaphylaxis(self):
        self._set_event("anaphylaxis", "active_anaphylaxis", False)

    def start_sepsis(self):
        self._set_event("sepsis", "active_sepsis", True)

    def stop_sepsis(self):
        self._set_event("sepsis", "active_sepsis", False)

    def stop_events(self):
        self.stop_hemorrhage()
        self.stop_anaphylaxis()
        self.stop_sepsis()
        self.stop_disturbance()

    def start_disturbance(self, profile: str):
        """Start a scripted stimulation/disturbance profile."""
        if not profile:
            return
        self.set_disturbance_profile(profile)
        self.disturbance_active = True
        self.disturbance_start_time = self.state.time

    def set_disturbance_profile(self, profile: str):
        """Select a disturbance profile without activating it."""
        if profile:
            self.disturbances = Disturbances(profile)
            self.disturbance_profile = profile
            self.config.disturbance_profile = profile
        else:
            self.disturbances = Disturbances(None)
            self.disturbance_profile = None
            self.config.disturbance_profile = None

    def stop_disturbance(self, clear_profile: bool = False):
        """Stop any active disturbance profile."""
        self.disturbance_active = False
        if clear_profile:
            self.disturbance_profile = None
            self.config.disturbance_profile = None
        
    def set_bair_hugger(self, target_c: float):
        """Set Bair Hugger target temperature (0 to disable)."""
        self.state.bair_hugger_target = target_c
        
    def set_airway_mode(self, mode_str: str):
        try:
            self.state.airway_mode = AIRWAY_MODE_MAP[mode_str]
        except (KeyError, TypeError) as exc:
            choices = ", ".join(AIRWAY_MODE_MAP)
            raise ValueError(
                f"Unsupported airway mode {mode_str!r}; choose one of: {choices}"
            ) from exc
        self.actions.record(
            self.state.time, ACTION_AIRWAY, label=self.state.airway_mode.value
        )
        if self.state.airway_mode == AirwayType.NONE:
            self.bag_mask_active = False

    def set_airway_obstruction(self, severity: float):
        self.airway_obstruction_manual = clamp(severity, 0.0, 1.0)

    def get_resp_step_kwargs(self, total_assisted_mv, peep, mean_paw, mech_rr, mech_vt_l, cardiac_output, mac_sevo=None):
        """Construct commonly duplicated arguments for respiratory step."""
        return {
            "ce_prop": self.state.propofol_ce,
            "ce_remi": self.state.remi_ce,
            "mech_vent_mv": total_assisted_mv,
            "fio2": self.state.fio2,
            "ce_roc": self.state.roc_ce,
            "et_sevo": self.state.et_sevo,
            "mac_sevo": mac_sevo if mac_sevo is not None else self.state.mac_sevo,
            "peep": peep,
            "mean_paw": mean_paw,
            "temp_c": self.state.temp_c,
            "mech_rr": mech_rr,
            "mech_vt_l": mech_vt_l,
            "airway_patency": self._airway_patency,
            "ventilation_efficiency": self._ventilation_efficiency,
            "vq_mismatch": self._vq_mismatch,
            "hb_g_dl": self.hemo.hb_conc,
            "oxygen_delivery_ratio": self._do2_ratio,
            "shiver_level": self._shiver_level,
            "cardiac_output": cardiac_output,
            "metabolic_factor": max(0.5, self._metabolic_factor),
        }


    def set_bronchospasm(self, severity: float):
        self.bronchospasm_manual = clamp(severity, 0.0, 1.0)

    def set_auto_laryngospasm(self, enabled: bool):
        self.auto_laryngospasm_enabled = bool(enabled)
            
    def set_rhythm(self, rhythm_name: str):
        """Set cardiac rhythm."""
        normalized = str(rhythm_name).upper()
        rhythm = next(
            (item for item in RhythmType if item.value == rhythm_name or item.name == normalized),
            None,
        )
        if rhythm is None:
            raise ValueError(f"Unknown rhythm: {rhythm_name!r}")
        self.hemo.rhythm_type = rhythm
        self.hemo.invalidate_state_cache()
    
    def set_bag_mask_ventilation(self, active: bool, rr: float = 12.0, vt: float = 0.5):
        """
        Enable/disable manual bag-mask ventilation.
        
        This is separate from mechanical ventilation. Bag-mask provides
        positive pressure ventilation via face mask or ETT using a
        self-inflating bag (Ambu bag).
        
        Args:
            active: True to enable, False to disable
            rr: Respiratory rate (bpm), default 12
            vt: Tidal volume (L), default 0.5 (~500mL)
        """
        self.bag_mask_active = active
        if active:
            self.bag_mask_rr = clamp(rr, 0.0, 40.0)
            self.bag_mask_vt = clamp(vt, 0.05, 1.2)
        self.actions.record(
            self.state.time, ACTION_BAG_MASK, label="on" if active else "off"
        )

    def start(self):
        """Start the simulation loop."""
        self.running = True

    def stop(self):
        """Stop the simulation."""
        self.running = False

    def start_recording(self, output_dir: str = ".", sample_interval_sec: float = 1.0):
        """Start CSV recording to the specified output directory."""
        if self.recorder and self.recorder.is_recording:
            return
        self.recorder = DataRecorder(output_dir=output_dir, sample_interval_sec=sample_interval_sec)
        self.recorder.start()

    def stop_recording(self):
        """Stop CSV recording."""
        if self.recorder:
            self.recorder.stop()

    def step(self, dt: float):
        """
        Advance simulation by dt seconds.
        """
        if dt <= 0 or not self.running:
            return
        runtime_core.step_simulation(self, dt)
        self.state.time += dt
        self._append_output_snapshot()
        if self.recorder and self.recorder.is_recording:
            self.recorder.log(self.state)

    def get_latest_state(self) -> SimulationState:
        """Return the most recent state snapshot."""
        return copy.copy(self.state)

    def get_predicted_csht(self, drug: str) -> float:
        """
        Predict context-sensitive half-time from current PK state.
        
        Simulates drug decay from current effect-site concentration (Ce)
        and measures time to 50% reduction. Returns time in minutes.
        
        Args:
            drug: "propofol" or "remi"
            
        Returns:
            Predicted CSHT in minutes, or 0.0 if drug inactive
        """
        if drug == "propofol" and self.pk_prop:
            return self.pk_prop.simulate_decay(target_fraction=0.5, max_seconds=3600)
        elif drug == "remi" and self.pk_remi:
            return self.pk_remi.simulate_decay(target_fraction=0.5, max_seconds=1200)
        return 0.0


    def set_vent_settings(self, rr: float, vt: float, peep: float, ie: str, 
                          mode: str, p_insp: float = None, fio2: float = None):
        """
        Update mechanical ventilator settings.
        
        Args:
            rr: Respiratory rate (bpm)
            vt: Tidal volume (L)
            peep: PEEP (cmH2O)
            ie: I:E ratio string (e.g., "1:2")
        mode: Ventilator mode ("VCV", "PCV", "PSV", "CPAP")
            p_insp: Optional inspiratory pressure above PEEP (cmH2O) - for PCV
            fio2: Optional FiO2 target (0.21-1.0). If set, adjusts fresh gas blender.
        """
        mode_upper = mode.upper()
        p_insp_effective = p_insp if p_insp is not None else self.vent.settings.p_insp
        if rr <= 0 and vt <= 0 and (p_insp_effective is None or p_insp_effective <= 0) and peep <= 0:
            self.vent.is_on = False
        elif mode_upper == "VCV":
            self.vent.is_on = rr > 0 and vt > 0
        elif mode_upper == "PCV":
            self.vent.is_on = rr > 0 and p_insp_effective > 0
        elif mode_upper in ("PSV", "CPAP"):
            has_support = p_insp_effective > 0 if mode_upper == "PSV" else False
            has_peep = peep > 0
            has_backup = rr > 0
            self.vent.is_on = has_support or has_peep or has_backup
        else:
            raise ValueError(f"Unsupported ventilator mode '{mode}'")
            
        if mode_upper == "CPAP":
            p_insp = 0.0
        self.resp_mech.set_settings(rr, vt, peep, ie, mode=mode, p_insp=p_insp)
        self.vent.update_settings(rr=rr, tv=vt*1000, peep=peep, ie=ie, 
                                  mode=mode, p_insp=p_insp, fio2=fio2)  # vt in mL for vent

        if fio2 is not None:
            self._apply_fio2_blender(fio2)

    def _apply_fio2_blender(self, fio2: float):
        """Blend fresh gas flows to achieve target FiO2 (O2/air only, N2O preserved)."""
        total_non_n2o = self.circuit.fgf_o2 + self.circuit.fgf_air
        if total_non_n2o <= 0:
            return
        target = clamp(fio2, 0.21, 1.0)
        # FiO2 = (O2 + 0.21*Air) / (O2 + Air + N2O)
        # Solve for O2 flow while keeping N2O fixed.
        n2o_flow = max(0.0, self.circuit.fgf_n2o)
        o2_flow = ((target * (total_non_n2o + n2o_flow)) - 0.21 * total_non_n2o) / 0.79
        o2_flow = clamp(o2_flow, 0.0, total_non_n2o)
        air_flow = total_non_n2o - o2_flow
        self.circuit.fgf_o2 = o2_flow
        self.circuit.fgf_air = air_flow
