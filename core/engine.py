
from collections import deque
from typing import Optional
import numpy as np
from dataclasses import replace

from .state import SimulationState, SimulationConfig, AirwayType
from .step_helpers import StepHelpersMixin
from .drug_api import DrugControllerMixin
from patient.patient import Patient
from physiology.hemodynamics import HemodynamicModel
from physiology.respiration import RespiratoryModel
from physiology.resp_mech import RespiratoryMechanics
from monitors.capno import Capnograph
from patient.pk_models import (
    PropofolPKMarsh, PropofolPKSchnider, PropofolPKEleveld,
    RemifentanilPKMinto, NorepinephrinePK, RocuroniumPK,
    EpinephrinePK, PhenylephrinePK
)
from patient.pd_models import TOFModel, LOCModel, TOLModel, BISModel
from physiology.disturbances import Disturbances
from monitors.alarms import AlarmSystem
from core.tci import TCIController

# Machine modules
from machine.circuit import CircleSystem
from machine.volatile import Vaporizer
from machine.ventilator import AnesthesiaVentilator
from patient.volatile_pk import VolatilePK

from monitors.ecg import ECGMonitor
from monitors.spo2 import SpO2Monitor
from monitors.nibp import NIBPMonitor, NIBPReading

class SimulationEngine(StepHelpersMixin, DrugControllerMixin):
    """
    Main simulation orchestrator.
    Manages time, updates subsystems, and produces state snapshots.
    
    STATE MANAGEMENT ARCHITECTURE:
    ==============================
    - `self.state` (SimulationState): Public API snapshot, read by UI/tests
    - Subsystem states (pk_prop.state, hemo.state, etc.): Internal working states
    
    State flows from subsystems → public state via sync methods:
    - _sync_pk_state(): PK concentrations (called in _step_pk)
    - Physiology sync: Done inline in _step_physiology (timing-dependent)
    
    This is intentional: the public state is a flat, consistent snapshot
    while subsystems maintain their own internal representations.
    """
    def __init__(self, patient: Patient, config: SimulationConfig):
        if hasattr(config, 'baseline_hb'):
            patient.baseline_hb = config.baseline_hb
        self.patient = patient
        self.config = config
        self.state = SimulationState()
        
        # Physics Constants
        self.K_REDIST = 50000.0 # Heat loss flux (W) per unit depth/sec increase
        
        # Subsystems
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
        
        # Machine
        self.circuit = None
        self.vaporizer = None
        self.vent = None
        
        # Monitors
        self.bis = None
        self.capno = None
        self.loc_pd = None
        self.tol_pd = None
        self.tof_pd = None
        
        self.ecg = None
        self.spo2_mon = None
        self.nibp = None
        self._next_nibp_time = 0.0
        
        # Controls (Basic TIVA controls)
        self.propofol_rate_mg_sec = 0.0
        self.remi_rate_ug_sec = 0.0
        self.nore_rate_ug_sec = 0.0
        self.roc_rate_mg_sec = 0.0
        self.epi_rate_ug_sec = 0.0
        self.phenyl_rate_ug_sec = 0.0
        
        # TCI Controllers
        self.tci_prop: Optional[TCIController] = None
        self.tci_remi: Optional[TCIController] = None
        self.tci_nore: Optional[TCIController] = None
        self.tci_epi: Optional[TCIController] = None
        self.tci_phenyl: Optional[TCIController] = None
        self.tci_roc: Optional[TCIController] = None
        
        # Disturbances & Alarms
        self.disturbances = Disturbances(config.disturbance_profile)
        self.disturbance_profile = config.disturbance_profile
        self.disturbance_active = bool(config.disturbance_profile)
        self.disturbance_start_time = 0.0
        self.alarms = AlarmSystem(dt=config.dt)
        
        # Output buffer (ring buffer for UI)
        self.output_buffer = deque(maxlen=1000)
        
        # Control flags
        self.running = False
        
        # Bag-mask ventilation (manual PPV, separate from mechanical vent)
        self.bag_mask_active = False
        self.bag_mask_rr = 12.0   # breaths per min (typical for manual PPV)
        self.bag_mask_vt = 0.5    # Liters (~500mL)
        
        # Event Flags
        self.active_hemorrhage = False
        self.active_anaphylaxis = False
        self.hemorrhage_rate_ml_min = 500.0
        
        # Pending fluid infusion (realistic timing)
        # Crystalloid: ~100 mL/min wide open (gravity) to 300 mL/min (pressure bag)
        # Blood products: ~50-100 mL/min
        self.pending_infusions = []
        self.fluid_infusion_rate_ml_min = 150.0  # mL/min (moderate rate with pressure)
        self.blood_infusion_rate_ml_min = 75.0   # mL/min slower for PRBCs
        
        # Anaphylaxis (gradual onset/offset)
        # Onset: ~2 min to peak (histamine/cytokine cascade)
        # Resolution: ~10-30 min with treatment
        self.anaphylaxis_severity = 0.0  # 0 to 1 (1 = full severity)
        self.anaphylaxis_onset_rate = 0.5 / 60.0  # per second (reaches 1.0 in ~2 min)
        self.anaphylaxis_decay_rate = 0.1 / 60.0  # per second (decays ~10 min)
        
        # Filtering/Smoothing state
        self.smooth_map = 80.0
        self.smooth_hr = 75.0
        self.smooth_bis = 98.0
        
        # Helpers
        self.current_mean_paw = 5.0
        
        # Random number generator for noise
        self.rng = np.random.default_rng()
        
        # Thermal Model State
        self.heat_production_basal = 0.0 # W
        self.specific_heat = 3470.0 # J/(kg K)
        self.surface_area = 1.9 # m^2 (default, updated in init)
        self._last_depth_index = 0.0 # For temperature redistribution calculation
        self._vent_active = False
        self._do2_ratio = 1.0
        
        # Viability Timers (Death Detector)
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
            
            # 1. Initialize Patient Physiology to Anesthetized State (MAC 1.0)
            # Pre-fill compartments for steady state
            ce_prop = 0.0
            ce_remi = 0.0
            ce_nore = 0.0
            mac = 0.0
            
            if "tiva" in self.config.maint_type:
                # Propofol ~3.0 ug/mL
                p_target = 3.0
                ce_prop = p_target
                self._prime_tci_drug("propofol", self.pk_prop, p_target)
                
                # Remi ~2.0 ng/mL
                r_target = 2.0
                ce_remi = r_target
                self._prime_tci_drug("remi", self.pk_remi, r_target)
                
            elif "balanced" in self.config.maint_type:
                # Sevo ~2.0% (target MAC 1.0)
                target_pct = 2.0
                target_frac = target_pct / 100.0
                
                self.set_vaporizer("Sevoflurane", target_pct)
                mac = 1.0
                
                # Pre-fill Volatile PK
                # Units: Fraction (0-1)
                if self.pk_sevo:
                    self._prime_volatile_state(target_frac)
                
                # Remi ~2.0 ng/mL (Analgesia)
                r_target = 2.0
                ce_remi = r_target
                self._prime_tci_drug("remi", self.pk_remi, r_target)
            
            # Ventilator Setup
            self.resp.state.apnea = True
            self.resp_mech.set_rr = 12
            self.resp_mech.set_vt = 0.5 # L
            self.resp_mech.peep = 5.0
            
            # Force Monitors to show "Asleep" values initially
            self.state.bis = 45.0
            self.smooth_bis = 45.0
            if self.bis:
                # Initialize BIS with target (no longer needs volatile Et percentages)
                self.bis.initialize(45.0)
            
            # Calculate and Apply Hemodynamic Steady State
            hemo_ss = self.hemo.calculate_steady_state(ce_prop, ce_remi, ce_nore, mac)
            self.hemo.state = hemo_ss
            
            # Sync Engine State
            self.state.map = hemo_ss.map
            self.state.hr = hemo_ss.hr
            self.state.sv = hemo_ss.sv
            self.state.svr = hemo_ss.svr
            self.state.co = hemo_ss.co
            self.state.mac = mac
            
            # Sync Smoothers
            self.smooth_map = hemo_ss.map
            self.smooth_hr = hemo_ss.hr
            
            # Update NIBP instant
            self.state.nibp_map = hemo_ss.map
            self.state.nibp_sys = hemo_ss.map + 15
            self.state.nibp_dia = hemo_ss.map - 10
            
            # Initialize Temperature
            if hasattr(self.patient, 'baseline_temp'):
                self.state.temp_c = self.patient.baseline_temp
            else:
                self.state.temp_c = 37.0

            # Sync PK state to Engine state so initial values are consistent
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
        propofol_models = {
            "Marsh": PropofolPKMarsh,
            "Schnider": PropofolPKSchnider,
            "Eleveld": PropofolPKEleveld,
        }
        propofol_model = propofol_models.get(self.config.pk_model_propofol, PropofolPKMarsh)
        self.pk_prop = propofol_model(self.patient)
            
        # Remi PK
        self.pk_remi = RemifentanilPKMinto(self.patient)
        
        # Machine
        self.circuit = CircleSystem()
        self.vaporizer = Vaporizer()
        self.vent = AnesthesiaVentilator()
        self.vent.is_on = False # Default OFF for induction/awake

        
        # Volatile PK
        self.pk_sevo = VolatilePK(self.patient, "Sevoflurane", lambda_b_g=0.65, mac_40=2.1)

        # Hemodynamics - Integrated hemodynamic model
        self.hemo = HemodynamicModel(self.patient, fidelity_mode=self.config.fidelity_mode)
        if self.config.hemo_model not in ["Su2023", "Su", "Advanced", None, ""]:
            print(f"Note: hemo_model '{self.config.hemo_model}' not recognized, using default")
            
        # Configure Norepinephrine PD if supported
        nore_pd_params = {
            "Li": (5.4, 98.7, 1.8),
            "Oualha": (7.04, 98.7, 1.8),
        }
        c50, emax, gamma = nore_pd_params.get(self.config.pk_model_nore, (7.04, 98.7, 1.8))
        self.hemo.set_nore_pd(c50=c50, emax=emax, gamma=gamma)
        self.resp = RespiratoryModel(self.patient, fidelity_mode=self.config.fidelity_mode)
        self.resp_mech = RespiratoryMechanics()
        
        # Monitors
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
        
        # Additional PK models
        self.pk_nore = NorepinephrinePK(self.patient, model=self.config.pk_model_nore)
        self.pk_roc = RocuroniumPK(self.patient, model_name="Wierda")
        self.pk_epi = EpinephrinePK(self.patient)

        self.pk_phenyl = PhenylephrinePK(self.patient)
        
        # Init thermal params
        self.heat_production_basal = self.patient.weight * 1.0 # ~1 W/kg basal
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
            # Sugammadex bolus (mg) - routes to TOF model for binding
            if self.tof_pd:
                self.tof_pd.give_sugammadex(amount)
            return

        bolus_targets = (
            ("prop", "pk_prop"),
            ("remi", "pk_remi"),
            ("nore", "pk_nore"),
            ("epi", "pk_epi"),
            ("phenyl", "pk_phenyl"),
            ("roc", "pk_roc"),
        )
        for token, pk_attr in bolus_targets:
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
        
    def stop_events(self):
        self.stop_hemorrhage()
        self.stop_anaphylaxis() 
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
            if hasattr(self, "config"):
                self.config.disturbance_profile = profile
        else:
            self.disturbances = Disturbances(None)
            self.disturbance_profile = None
            if hasattr(self, "config"):
                self.config.disturbance_profile = None

    def stop_disturbance(self, clear_profile: bool = False):
        """Stop any active disturbance profile."""
        self.disturbance_active = False
        if clear_profile:
            self.disturbance_profile = None
            if hasattr(self, "config"):
                self.config.disturbance_profile = None
        
    def set_bair_hugger(self, target_c: float):
        """Set Bair Hugger target temperature (0 to disable)."""
        self.state.bair_hugger_target = target_c
        
    def set_airway_mode(self, mode_str: str):
        """
        Set airway mode: "None", "Mask", "ETT".
        """
        if mode_str == "None":
            self.state.airway_mode = AirwayType.NONE
            # Disconnecting airway stops all ventilation
            self.bag_mask_active = False
        elif mode_str == "Mask":
            self.state.airway_mode = AirwayType.MASK
        elif mode_str == "ETT":
            self.state.airway_mode = AirwayType.ETT
            # ETT placement typically means switching from bag-mask to vent
            # (don't auto-disable bag-mask here - let UI handle it)
    
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

        # 1. Disturbances & Events
        dist_vec = self._step_disturbances(dt)
        
        # 2. TCI Controllers
        self._step_tci(dt)
        
        # 3. Machine (Vaporizer, Circuit)
        fi_sevo = self._step_machine(dt)
        
        # 4. PK Models
        self._step_pk(dt, fi_sevo, self.state.co)
        
        # 5. Physiology (Hemo, Resp)
        hemo_state, resp_state, phase = self._step_physiology(dt, dist_vec)
        
        # 6. Monitors & History
        self._step_monitors(dt, phase, hemo_state, resp_state, dist_vec)
        
        # 7. Temperature Model
        self._step_temperature(dt)

        # 8. Death Detector
        self._check_patient_viability(dt)
        
        # 9. Update Time & Buffer
        self.state.time += dt
        self.output_buffer.append(replace(self.state))

    def get_latest_state(self) -> SimulationState:
        """Return the most recent state snapshot."""
        # Return a copy to safely store/consume
        return replace(self.state)

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
        self.vent.is_on = rr > 0 and (vt > 0 or (mode == "PCV" and p_insp and p_insp > 0))
            
        self.resp_mech.set_settings(rr, vt, peep, ie, mode=mode, p_insp=p_insp)
        # Also update vent.settings for UI sync
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
