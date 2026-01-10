
from collections import deque
from typing import Optional
import numpy as np
import copy

from .state import SimulationState, SimulationConfig, AirwayType
from .step_helpers import StepHelpersMixin
from .drug_api import DrugControllerMixin
from anasim.patient.patient import Patient
from anasim.physiology.hemodynamics import HemodynamicModel
from anasim.physiology.respiration import RespiratoryModel
from anasim.physiology.resp_mech import RespiratoryMechanics
from anasim.monitors.capno import Capnograph
from anasim.patient.pk_models import (
    PropofolPKMarsh, PropofolPKSchnider, PropofolPKEleveld,
    RemifentanilPKMinto, NorepinephrinePK, RocuroniumPK,
    EpinephrinePK, PhenylephrinePK
)
from anasim.patient.pd_models import TOFModel, LOCModel, TOLModel, BISModel
from anasim.physiology.disturbances import Disturbances
from anasim.monitors.alarms import AlarmSystem
from anasim.core.tci import TCIController
from anasim.core.enums import RhythmType
from anasim.core.utils import clamp

AIRWAY_MODE_MAP = {
    "None": AirwayType.NONE,
    "Mask": AirwayType.MASK,
    "ETT": AirwayType.ETT,
}

BOLUS_TARGETS = (
    ("prop", "pk_prop"),
    ("remi", "pk_remi"),
    ("nore", "pk_nore"),
    ("epi", "pk_epi"),
    ("phenyl", "pk_phenyl"),
    ("roc", "pk_roc"),
)

PROPOFOL_MODELS = {
    "Marsh": PropofolPKMarsh,
    "Schnider": PropofolPKSchnider,
    "Eleveld": PropofolPKEleveld,
}

NORE_PD_PARAMS = {
    "Li": (5.4, 98.7, 1.8),
    "Oualha": (7.04, 98.7, 1.8),
}

# Machine modules
from anasim.machine.circuit import CircleSystem
from anasim.machine.volatile import Vaporizer
from anasim.machine.ventilator import AnesthesiaVentilator
from anasim.patient.volatile_pk import VolatilePK

from anasim.monitors.ecg import ECGMonitor
from anasim.monitors.spo2 import SpO2Monitor
from anasim.monitors.nibp import NIBPMonitor, NIBPReading

class SimulationEngine(StepHelpersMixin, DrugControllerMixin):
    """
    Main simulation orchestrator.
    Manages time, updates subsystems, and produces state snapshots.
    
    State management:
    - `self.state` is the public snapshot for UI/tests.
    - Subsystems keep their own internal state and sync into `self.state`
      via _sync_pk_state() and inline updates in _step_physiology().
    """
    def __init__(self, patient: Patient, config: SimulationConfig):
        if hasattr(config, 'baseline_hb'):
            patient.baseline_hb = config.baseline_hb
        self.patient = patient
        self.config = config
        self.state = SimulationState()
        
        # Physics constants.
        self.K_REDIST = 50000.0 # Heat loss flux (W) per unit depth/sec increase
        
        # Subsystems.
        self.pk_prop = None
        self.pk_remi = None
        self.pk_nore = None
        self.pk_roc = None
        self.pk_sevo = None
        self.pk_epi = None
        self.pk_phenyl = None

        self.active_agent = "Sevoflurane"

        self.hemo = None
        self.resp = None
        self.resp_mech = None
        
        # Machine.
        self.circuit = None
        self.vaporizer = None
        self.vent = None
        
        # Monitors.
        self.bis = None
        self.capno = None
        self.loc_pd = None
        self.tol_pd = None
        self.tof_pd = None
        
        self.ecg = None
        self.spo2_mon = None
        self.nibp = None
        self._next_nibp_time = 0.0
        
        # Controls (basic TIVA).
        self.propofol_rate_mg_sec = 0.0
        self.remi_rate_ug_sec = 0.0
        self.nore_rate_ug_sec = 0.0
        self.roc_rate_mg_sec = 0.0
        self.epi_rate_ug_sec = 0.0
        self.phenyl_rate_ug_sec = 0.0
        
        # TCI controllers.
        self.tci_prop: Optional[TCIController] = None
        self.tci_remi: Optional[TCIController] = None
        self.tci_nore: Optional[TCIController] = None
        self.tci_epi: Optional[TCIController] = None
        self.tci_phenyl: Optional[TCIController] = None
        self.tci_roc: Optional[TCIController] = None
        
        # Disturbances & alarms.
        self.disturbances = Disturbances(config.disturbance_profile)
        self.disturbance_profile = config.disturbance_profile
        self.disturbance_active = bool(config.disturbance_profile)
        self.disturbance_start_time = 0.0
        self.alarms = AlarmSystem(dt=config.dt)
        
        # Output buffer (ring buffer for UI).
        self.output_buffer = deque(maxlen=1000)
        
        # Control flags.
        self.running = False
        
        # Bag-mask ventilation (manual PPV, separate from mechanical vent).
        self.bag_mask_active = False
        self.bag_mask_rr = 12.0   # breaths per min (typical for manual PPV)
        self.bag_mask_vt = 0.5    # Liters (~500mL)
        
        # Event flags.
        self.active_hemorrhage = False
        self.active_anaphylaxis = False
        self.hemorrhage_rate_ml_min = 500.0
        
        # Pending fluid infusion (realistic timing).
        self.pending_infusions = []
        self.fluid_infusion_rate_ml_min = 150.0  # mL/min (moderate rate with pressure)
        self.blood_infusion_rate_ml_min = 75.0   # mL/min slower for PRBCs
        
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
        self.laryngospasm_tau_on = 1.0   # seconds (fast onset)
        self.laryngospasm_tau_off = 8.0  # seconds (slower relief)
        self._airway_patency = 1.0
        self._ventilation_efficiency = 1.0
        self._capno_obstruction = 0.0
        self._vq_mismatch = 0.0
        self._base_airway_resistance = 10.0
        
        # Filtering/smoothing state.
        self.smooth_map = 80.0
        self.smooth_hr = 75.0
        self.smooth_bis = 98.0
        self._tol_current = None
        
        # Helpers.
        self.current_mean_paw = 5.0
        
        # Random number generator for noise.
        self.rng = np.random.default_rng()
        self._monitor_noise_std = np.array([0.5, 0.5, 0.2])
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
        self.specific_heat = 3470.0 # J/(kg K)
        self.surface_area = 1.9 # m^2 (default, updated in init)
        self._last_depth_index = 0.0 # For temperature redistribution calculation
        self._shiver_level = 0.0
        self._cached_temp_metabolic = 37.0
        self._cached_temp_metabolic_factor = 1.0
        self._vent_active = False
        self._do2_ratio = 1.0
        
        # Viability timers (death detector).
        self.time_brady = 0.0
        self.time_hypotension = 0.0
        self.time_tachy = 0.0
        self.DEATH_GRACE_PERIOD = 15.0 # seconds
        
        self.initialize_models()
        self.initialize_state()
        
    def initialize_state(self):
        """Set initial state based on config (Steady State)."""
        if self.config.mode == 'steady_state':
            self.state.airway_mode = AirwayType.ETT  # Default to intubated for steady state
            self.vent.is_on = True  # Ventilator ON for steady state
            
            # Initialize patient physiology for steady state (MAC 1.0).
            ce_prop = 0.0
            ce_remi = 0.0
            ce_nore = 0.0
            mac = 0.0
            
            if "tiva" in self.config.maint_type:
                # Propofol ~3.0 ug/mL.
                p_target = 3.0
                ce_prop = p_target
                self._prime_tci_drug("propofol", self.pk_prop, p_target)
                
                # Remi ~2.0 ng/mL.
                r_target = 2.0
                ce_remi = r_target
                self._prime_tci_drug("remi", self.pk_remi, r_target)
                
            elif "balanced" in self.config.maint_type:
                # Sevo ~2.0% (target MAC 1.0).
                target_pct = 2.0
                target_frac = target_pct / 100.0
                
                self.set_vaporizer("Sevoflurane", target_pct)
                mac = 1.0
                
                # Pre-fill volatile PK (fraction 0-1).
                if self.pk_sevo:
                    self._prime_volatile_state(target_frac)
                
                # Remi ~2.0 ng/mL (analgesia).
                r_target = 2.0
                ce_remi = r_target
                self._prime_tci_drug("remi", self.pk_remi, r_target)
            
            # Ventilator setup.
            self.resp.state.apnea = True
            self.resp_mech.set_rr = 12
            self.resp_mech.set_vt = 0.5 # L
            self.resp_mech.peep = 5.0
            
            # Force monitors to show "asleep" values initially.
            self.state.bis = 45.0
            self.smooth_bis = 45.0
            if self.bis:
                # Initialize BIS with target (no longer needs volatile Et percentages)
                self.bis.initialize(45.0)
            
            # Calculate and apply hemodynamic steady state.
            hemo_ss = self.hemo.calculate_steady_state(ce_prop, ce_remi, ce_nore, mac)
            self.hemo.state = hemo_ss
            
            # Sync engine state.
            self.state.map = hemo_ss.map
            self.state.hr = hemo_ss.hr
            self.state.sv = hemo_ss.sv
            self.state.svr = hemo_ss.svr
            self.state.co = hemo_ss.co
            self.state.mac = mac
            
            # Sync smoothers.
            self.smooth_map = hemo_ss.map
            self.smooth_hr = hemo_ss.hr
            
            # Seed NIBP.
            self.state.nibp_map = hemo_ss.map
            self.state.nibp_sys = hemo_ss.map + 15
            self.state.nibp_dia = hemo_ss.map - 10
            
            # Initialize temperature.
            if hasattr(self.patient, 'baseline_temp'):
                self.state.temp_c = self.patient.baseline_temp
            else:
                self.state.temp_c = 37.0

            # Sync PK state to engine state for consistency.
            self._sync_pk_state()

        self._seed_nibp_reading()

    def _seed_nibp_reading(self):
        """Seed NIBP with an initial reading and start cycling immediately."""
        if not self.nibp or not self.hemo:
            return
        hemo_state = self.hemo.state
        map_val = hemo_state.map
        sbp_val = getattr(hemo_state, "sbp", map_val + 20.0)
        dbp_val = getattr(hemo_state, "dbp", map_val - 20.0)
        ts = self.state.time if self.state.time > 0.0 else 1e-3
        self.nibp.latest_reading = NIBPReading(sbp_val, dbp_val, map_val, ts)
        self.state.nibp_sys = sbp_val
        self.state.nibp_dia = dbp_val
        self.state.nibp_map = map_val
        self.state.nibp_timestamp = ts
        self._next_nibp_time = self.state.time + self.nibp.interval
        self.nibp.trigger()
        
    def initialize_models(self):
        """Initialize PK/PD models based on config."""
        # Propofol PK
        propofol_model = PROPOFOL_MODELS.get(self.config.pk_model_propofol, PropofolPKMarsh)
        self.pk_prop = propofol_model(self.patient)
            
        # Remi PK
        self.pk_remi = RemifentanilPKMinto(self.patient)
        
        # Machine
        self.circuit = CircleSystem()
        self.vaporizer = Vaporizer()
        self.vent = AnesthesiaVentilator()
        self.vent.is_on = False # Default OFF for induction/awake

        
        self.pk_sevo = VolatilePK(self.patient, "Sevoflurane", lambda_b_g=0.65, mac_40=2.1)

        # Hemodynamics.
        self.hemo = HemodynamicModel(self.patient, fidelity_mode=self.config.fidelity_mode)
        if self.config.hemo_model not in ["Su2023", "Su", "Advanced", None, ""]:
            print(f"Note: hemo_model '{self.config.hemo_model}' not recognized, using default")
            
        # Configure norepinephrine PD.
        c50, emax, gamma = NORE_PD_PARAMS.get(self.config.pk_model_nore, (7.04, 98.7, 1.8))
        self.hemo.set_nore_pd(c50=c50, emax=emax, gamma=gamma)
        self.resp = RespiratoryModel(self.patient, fidelity_mode=self.config.fidelity_mode)
        self.resp_mech = RespiratoryMechanics()
        self._base_airway_resistance = self.resp_mech.resistance
        
        # Monitors.
        self.bis = BISModel(self.patient, model_name=self.config.bis_model)
        self.capno = Capnograph()
        self.loc_pd = LOCModel(model_name=self.config.loc_model)
        self.tol_pd = TOLModel()
        self.tof_pd = TOFModel(
            self.patient,
            model_name="Wierda",
            fidelity_mode=self.config.fidelity_mode
        ) # Default Rocuronium Model
        
        self.ecg = ECGMonitor()
        self.spo2_mon = SpO2Monitor()
        self.nibp = NIBPMonitor(interval_min=5.0)
        self.state.nibp_interval_sec = self.nibp.interval
        
        # Additional PK models.
        self.pk_nore = NorepinephrinePK(self.patient, model=self.config.pk_model_nore)
        self.pk_roc = RocuroniumPK(self.patient, model_name="Wierda")
        self.pk_epi = EpinephrinePK(self.patient, model=self.config.pk_model_epi)

        self.pk_phenyl = PhenylephrinePK(self.patient)
        
        # Init thermal params.
        self.heat_production_basal = self.patient.weight * 1.0
        if self.patient.bsa > 0:
            self.surface_area = self.patient.bsa
        else:
            self.surface_area = 1.9

    def set_fgf(self, o2_l_min: float, air_l_min: float):
        """Set Fresh Gas Flow."""
        if self.circuit:
            self.circuit.fgf_o2 = o2_l_min
            self.circuit.fgf_air = air_l_min
            
    def set_vaporizer(self, agent: str, percent: float):
        """Set Vaporizer Agent and Dial."""
        self.active_agent = agent
        if self.circuit:
            self.circuit.vaporizer_setting = percent
            self.circuit.vaporizer_on = (percent > 0)

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
        self._queue_infusion(volume_ml, self.blood_infusion_rate_ml_min, hematocrit=hematocrit)

    def _queue_infusion(self, volume_ml: float, rate_ml_min: float, hematocrit: float):
        if volume_ml <= 0 or rate_ml_min <= 0:
            return
        self.pending_infusions.append({
            'remaining': volume_ml,
            'rate': rate_ml_min,
            'hematocrit': hematocrit
        })

    def give_drug_bolus(self, drug_name: str, amount: float):
        """
        Administer drug bolus.
        Updates PK state instantaneously: New_C1 = Old_C1 + Dose/V1.
        
        amounts:
        - Propofol: mg
        - Remifentanil: mcg
        - Norepinephrine: mcg
        - Epinephrine: mcg
        - Phenylephrine: mcg
        - Rocuronium: mg
        """
        drug = drug_name.lower()
        
        if "sug" in drug:
            # Sugammadex bolus (mg) routes to TOF model for binding.
            if self.tof_pd:
                self.tof_pd.give_sugammadex(amount)
            return

        for token, pk_attr in BOLUS_TARGETS:
            if token in drug:
                model = getattr(self, pk_attr, None)
                if model:
                    conc_delta = amount / model.v1
                    model.state.c1 += conc_delta
                break

    def start_hemorrhage(self, rate_ml_min: float = 500.0):
        self.active_hemorrhage = True
        self.hemorrhage_rate_ml_min = rate_ml_min

    def stop_hemorrhage(self):
        self.active_hemorrhage = False
        
    def start_anaphylaxis(self):
        self.active_anaphylaxis = True

    def stop_anaphylaxis(self):
        self.active_anaphylaxis = False

    def start_sepsis(self):
        self.active_sepsis = True

    def stop_sepsis(self):
        self.active_sepsis = False
        
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
        self.state.airway_mode = AIRWAY_MODE_MAP.get(mode_str, AirwayType.NONE)
        if self.state.airway_mode == AirwayType.NONE:
            self.bag_mask_active = False

    def set_airway_obstruction(self, severity: float):
        self.airway_obstruction_manual = clamp(severity, 0.0, 1.0)

    def set_bronchospasm(self, severity: float):
        self.bronchospasm_manual = clamp(severity, 0.0, 1.0)

    def set_auto_laryngospasm(self, enabled: bool):
        self.auto_laryngospasm_enabled = bool(enabled)
            
    def set_rhythm(self, rhythm_name: str):
        """Set cardiac rhythm."""
        try:
             # Search by value (e.g. "Sinus Rhythm") or name (e.g. "SINUS")
             found = False
             for r in RhythmType:
                 if r.value == rhythm_name or r.name == rhythm_name.upper():
                     if self.hemo:
                         self.hemo.rhythm_type = r
                         if hasattr(self.hemo, "invalidate_state_cache"):
                             self.hemo.invalidate_state_cache()
                         found = True
                     break
             
             if not found:
                 print(f"Unknown rhythm: {rhythm_name}")
                 
        except Exception as e:
             print(f"Error setting rhythm: {e}")
    
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
            self.bag_mask_rr = rr
            self.bag_mask_vt = vt 

    def start(self):
        """Start the simulation loop."""
        self.running = True

    def stop(self):
        """Stop the simulation."""
        self.running = False

    def step(self, dt: float):
        """
        Advance simulation by dt seconds.
        """
        if dt <= 0:
            return
        if not self.running:
            return

        # Cache depth_index for use in temperature and disturbances.
        self._depth_index = self.state.mac + (self.state.propofol_ce / 4.0)

        dist_vec = self._step_disturbances(dt)

        self._step_tci(dt)
        
        fi_sevo = self._step_machine(dt)
        
        self._step_pk(dt, fi_sevo, self.state.co)
        
        hemo_state, resp_state, phase = self._step_physiology(dt, dist_vec)
        
        self._step_monitors(dt, phase, hemo_state, resp_state, dist_vec)

        self._update_shivering(dt)

        self._step_temperature(dt)

        self._check_patient_viability(dt)
        
        self.state.time += dt
        self.output_buffer.append(copy.copy(self.state))

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
                          mode: str = None, p_insp: float = None):
        """
        Update mechanical ventilator settings.
        
        Args:
            rr: Respiratory rate (bpm)
            vt: Tidal volume (L)
            peep: PEEP (cmH2O)
            ie: I:E ratio string (e.g., "1:2")
            mode: Optional ventilator mode ("VCV" or "PCV")
            p_insp: Optional inspiratory pressure above PEEP (cmH2O) - for PCV
        """
        mode_upper = mode.upper() if mode else None
        if mode_upper == "VCV":
            self.vent.is_on = rr > 0 and vt > 0
        elif mode_upper == "PCV":
            self.vent.is_on = rr > 0 and p_insp and p_insp > 0
        elif mode_upper in ("PSV", "CPAP"):
            has_support = (p_insp is not None and p_insp > 0)
            has_peep = (peep is not None and peep > 0)
            has_backup = rr > 0
            self.vent.is_on = has_support or has_peep or has_backup
        else:
            # Fallback for legacy callers (mode not specified).
            self.vent.is_on = rr > 0 and vt > 0
            
        self.resp_mech.set_settings(rr, vt, peep, ie, mode=mode, p_insp=p_insp)
        self.vent.update_settings(rr=rr, tv=vt*1000, peep=peep, ie=ie, 
                                  mode=mode, p_insp=p_insp)  # vt in mL for vent

    def _prime_pk_state(self, pk_model, target: float):
        """Initialize PK compartments to a target concentration."""
        if not pk_model or not hasattr(pk_model, "state"):
            return
        for attr in ("c1", "c2", "c3"):
            if hasattr(pk_model.state, attr):
                setattr(pk_model.state, attr, target)
        for attr in ("ce", "ce2"):
            if hasattr(pk_model.state, attr):
                setattr(pk_model.state, attr, target)

    def _prime_tci_drug(self, drug: str, pk_model, target: float):
        """Seed PK state and enable TCI at the given target."""
        if not pk_model:
            return
        self._prime_pk_state(pk_model, target)
        self.enable_tci(drug, target)

    def _prime_volatile_state(self, target_frac: float):
        """Seed volatile circuit and tissue compartments to a target fraction."""
        if self.circuit:
            self.circuit.composition.fi_agent = target_frac
        if not self.pk_sevo:
            return
        for attr in ("p_alv", "p_art", "p_ven", "p_vrg", "p_mus", "p_fat"):
            if hasattr(self.pk_sevo.state, attr):
                setattr(self.pk_sevo.state, attr, target_frac)
        self.pk_sevo.state.mac = 1.0
