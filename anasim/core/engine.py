import copy
from collections import deque
from dataclasses import dataclass

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
from anasim.monitors.airway import AirwaySensor
from anasim.monitors.alarms import AlarmSystem
from anasim.monitors.arterial import ArterialLineMonitor, ArterialWaveformRenderer
from anasim.monitors.capno import Capnograph
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.monitors.ecg import ECGMonitor
from anasim.monitors.nibp import NIBPMonitor
from anasim.monitors.spo2 import SpO2Monitor
from anasim.patient.patient import Patient
from anasim.patient.pd import (
    BISModel,
    ClinicalResponseModel,
    TOFModel,
    hypnotic_equivalent,
    midazolam_loss_of_response,
    opioid_equivalent,
)
from anasim.patient.pk_models import (
    DobutaminePK,
    EpinephrinePK,
    EsmololPK,
    EtomidatePK,
    FentanylPK,
    GlycopyrrolatePK,
    KetaminePK,
    LabetalolPK,
    LidocainePK,
    MidazolamPK,
    MilrinonePK,
    NorepinephrinePK,
    PhenylephrinePK,
    PropofolPKEleveld,
    RemifentanilPKEleveld,
    RocuroniumPK,
    VasopressinPK,
)
from anasim.patient.volatile_pk import VolatilePK
from anasim.physiology.disturbances import Disturbances
from anasim.physiology.hemodynamics import HemodynamicModel
from anasim.physiology.lung import LungAeration
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
from .state import AirwayType, SimulationConfig, SimulationState, WaveformSample
from .tci import TCIController

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
    retention_fraction: float | None = None
    label: str = "crystalloid"


class SimulationEngine(DrugControllerMixin):
    """Owns the subsystems and learner controls and advances them in time.

    Subsystems keep their own state; projection and monitor modules copy it
    into the public `self.state` snapshot.
    """

    def __init__(self, patient: Patient, config: SimulationConfig):
        self.patient = patient
        self.config = config
        self.state = SimulationState()
        self.actions = ActionLog()
        self.airway_tuning = AirwayTuning()
        self.thermal_tuning = ThermalTuning()

        # Infusion rates (model units/s) and active TCI controllers, by drug key.
        self.infusion_rates = {spec.key: 0.0 for spec in DRUG_REGISTRY if spec.has_infusion}
        self.tci: dict[str, TCIController] = {}
        self._next_nibp_time = 0.0

        self.disturbances = Disturbances(config.disturbance_profile)
        self.disturbance_profile = config.disturbance_profile
        self.disturbance_active = bool(config.disturbance_profile)
        self.disturbance_start_time = 0.0
        self.alarms = AlarmSystem()

        self._output_window_s = 20.0
        self.output_buffer: deque[WaveformSample] = deque()
        self.recorder: DataRecorder | None = None
        self._tci_accumulators: dict[str, float] = {}
        self.running = False

        self.bag_mask_active = False
        self.bag_mask_rr = 12.0  # breaths/min
        self.bag_mask_vt = 0.5  # L

        self.active_hemorrhage = False
        self.active_anaphylaxis = False
        self.hemorrhage_rate_ml_min = 500.0

        # Boluses run in over minutes rather than instantly.
        self.pending_infusions: list[PendingInfusion] = []
        self.fluid_infusion_rate_ml_min = 150.0
        self.blood_infusion_rate_ml_min = 75.0
        self.maintenance_fluid_rate_ml_min = 0.0

        # Event severities (0-1) ramp at these rates per second.
        self.anaphylaxis_severity = 0.0
        self.anaphylaxis_onset_rate = 0.5 / 60.0  # full in about 2 min
        self.anaphylaxis_decay_rate = 0.1 / 60.0  # resolves in about 10 min
        self.active_sepsis = False
        self.sepsis_severity = 0.0
        self.sepsis_onset_rate = 0.1 / 60.0  # full in about 10 min
        self.sepsis_decay_rate = 0.03 / 60.0  # resolves in 30+ min

        self.auto_laryngospasm_enabled = True
        self.airway_obstruction_manual = 0.0
        self.bronchospasm_manual = 0.0
        self.laryngospasm_severity = 0.0
        self._airway_patency = 1.0
        self._ventilation_efficiency = 1.0
        self._capno_obstruction = 0.0
        self._vq_mismatch = 0.0

        self.smooth_bis = 98.0
        self._monitor_tau_bis_s = 2.0
        self._capno_numeric_peak = 0.0
        self._capno_numeric_age_s = 0.0
        self._capno_numeric_timeout_s = 15.0
        self._capno_last_phase = "EXP"
        self._capno_has_sample = False
        self._mean_paw_tau_s = 0.25
        self._tol_current = 0.0
        self._pk_hemo_scale_cache: tuple[float, float] | None = None
        self.current_mean_paw = 0.0

        # Separate generators keep each monitor's noise reproducible regardless
        # of how often the others draw.
        self.rng = np.random.default_rng(self.config.rng_seed)
        self._ecg_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        self._nibp_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        self._cardiac_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        self._bis_noise_std = 0.2

        self.heat_production_basal = self.patient.weight * 1.0  # W
        self.specific_heat = self.thermal_tuning.specific_heat_j_kg_k  # J/(kg K)
        self.surface_area = self.patient.bsa  # m^2
        self._redistributed_heat_j = 0.0  # Core heat moved to the periphery
        self._depth_index = 0.0
        self._shiver_level = 0.0
        self._metabolic_factor = 1.0
        self._pulseless_s = 0.0

        self.initialize_models()
        self.set_continuous_fluid_rate(self.config.maintenance_fluid_ml_hr)
        self.initialize_state()

    def _append_output_snapshot(self) -> None:
        """Append this step's waveform samples and retain twenty seconds by timestamp."""
        state = self.state
        self.output_buffer.append(
            WaveformSample(
                state.time, state.ecg_voltage, state.pleth_voltage, state.capno_co2, state.art_pressure,
                state.paw, state.flow, state.volume, self.vent.breath_count,
            )
        )
        cutoff = self.state.time - self._output_window_s
        while len(self.output_buffer) > 1 and self.output_buffer[0].time < cutoff:
            self.output_buffer.popleft()

    def initialize_state(self):
        """Seed the public state from the initialized subsystems."""
        self.state.temp_c = self.patient.baseline_temp
        self._airway_patency = 1.0
        self._ventilation_efficiency = 1.0
        self._capno_obstruction = 0.0
        self._vq_mismatch = 0.0

        initialize_engine_state(self)
        projection_core.sync_state_from_models(self)
        monitor_core.seed_nibp_reading(self)
        self._append_output_snapshot()

    def set_continuous_fluid_rate(self, ml_hr: float | None):
        """Set continuous IV fluids in mL/hr; None restores 1 mL/kg/hr."""
        rate = self.patient.weight if ml_hr is None else max(0.0, float(ml_hr))
        self.maintenance_fluid_rate_ml_min = rate / 60.0

    def get_continuous_fluid_rate(self) -> float:
        """Return current continuous IV fluid rate in mL/hr."""
        return self.maintenance_fluid_rate_ml_min * 60.0

    def initialize_models(self):
        """Build the subsystem models selected by the configuration."""
        self.pk_prop = PropofolPKEleveld(self.patient, concomitant_opioids=self.config.concomitant_opioids)
        self.pk_remi = RemifentanilPKEleveld(self.patient)
        self.pk_fentanyl = FentanylPK(self.patient)
        self.pk_midazolam = MidazolamPK(self.patient)
        self.pk_etomidate = EtomidatePK(self.patient)
        self.pk_ketamine = KetaminePK(self.patient)
        self.pk_lidocaine = LidocainePK(self.patient)
        self.midazolam_c50 = midazolam_loss_of_response(self.patient.age)

        self.circuit = CircleSystem()
        self.vent = AnesthesiaVentilator()

        self.vaporizer = Vaporizer()
        # MAC40 from Mapleson 1996 / Nickalls and Mapleson 2003.
        self.pk_sevo = VolatilePK(
            self.patient,
            "Sevoflurane",
            lambda_b_g=0.65,
            mac_40=1.80,
        )
        # N2O comes from fresh gas, not the vaporizer, so it is always modeled.
        # Partition coefficients at 37 °C (blood:gas 0.47; brain, muscle, fat:blood
        # 1.1, 1.2, 2.3) and MAC about 104% at 1 atm.
        self.pk_n2o = VolatilePK(
            self.patient,
            "Nitrous Oxide",
            lambda_b_g=0.47,
            lambda_t_b_vrg=1.1,
            mac_40=104.0,
            lambda_t_b_mus=1.2,
            lambda_t_b_fat=2.3,
        )

        self.hemo = HemodynamicModel(self.patient)
        self.resp = RespiratoryModel(self.patient)
        self.aeration = LungAeration(self.patient, recruited=0.9 if self.config.mode == "steady_state" else 1.0)
        self.resp_mech = RespiratoryMechanics(aeration=self.aeration)
        self._base_airway_resistance = self.resp_mech.resistance
        self.set_vent_settings(rr=12.0, vt=0.5, peep=5.0, ie="1:2", mode="VCV", p_insp=15.0)
        self.resp.baseline_co_l_min = self.hemo.base_co_l_min

        self.bis = BISModel(self.patient)
        self.capno = Capnograph(self.resp.vd_deadspace)
        self.airway_sensor = AirwaySensor()
        self.response = ClinicalResponseModel()
        self.tof_pd = TOFModel(self.patient)

        self.cardiac_cycle = CardiacCycle(rng=self._cardiac_rng)
        self.arterial_waveform = ArterialWaveformRenderer(self.patient.age)
        self.art_line = ArterialLineMonitor()
        self.ecg = ECGMonitor(rng=self._ecg_rng)
        self.spo2_mon = SpO2Monitor()
        self.nibp = NIBPMonitor(interval_min=5.0, rng=self._nibp_rng)
        self.state.nibp_interval_sec = self.nibp.interval

        self.pk_nore = NorepinephrinePK(self.patient)
        self.pk_roc = RocuroniumPK(self.patient)
        self.pk_epi = EpinephrinePK(self.patient)
        self.pk_phenyl = PhenylephrinePK(self.patient)
        self.pk_vaso = VasopressinPK(self.patient)
        self.pk_dobu = DobutaminePK(self.patient)
        self.pk_mil = MilrinonePK(self.patient)
        self.pk_esmolol = EsmololPK(self.patient)
        self.pk_labetalol = LabetalolPK(self.patient)
        self.pk_glyco = GlycopyrrolatePK(self.patient)

    def set_fgf(self, o2_l_min: float, air_l_min: float, n2o_l_min: float = 0.0):
        """Set fresh gas flows in L/min."""
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

    def set_vaporizer(self, percent: float):
        """Set the sevoflurane vaporizer dial (%); it stays off when sevoflurane is disabled."""
        if not self.config.sevoflurane_enabled:
            return
        self.vaporizer.set_concentration(percent)
        self.circuit.vaporizer_setting = self.vaporizer.state.setting
        self.circuit.vaporizer_on = self.vaporizer.state.is_on
        self.actions.record(
            self.state.time,
            ACTION_VAPORIZER,
            label="Sevoflurane",
            amount=self.circuit.vaporizer_setting,
        )

    def give_fluid(self, volume_ml: float):
        """Queue a crystalloid bolus at 150 mL/min."""
        self._queue_infusion(volume_ml, self.fluid_infusion_rate_ml_min, hematocrit=0.0)

    def give_blood(self, volume_ml: float = 300.0, hematocrit: float = 0.55):
        """Queue packed red cells at 75 mL/min."""
        self._queue_infusion(
            volume_ml,
            self.blood_infusion_rate_ml_min,
            hematocrit=hematocrit,
            label="blood",
        )

    def give_albumin(self, volume_ml: float):
        """Queue albumin, which stays intravascular longer than crystalloid."""
        self._queue_infusion(
            volume_ml,
            self.fluid_infusion_rate_ml_min,
            hematocrit=0.0,
            retention_fraction=self.hemo.config.colloid_retention_fraction,
            label="colloid",
        )

    def _queue_infusion(
        self,
        volume_ml: float,
        rate_ml_min: float,
        hematocrit: float,
        retention_fraction: float | None = None,
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
        """Give a bolus in the drug's registry unit; it adds dose / V1 to C1.

        Sugammadex goes to the TOF model for binding.
        """
        amount = float(amount)
        if amount <= 0:
            return

        if drug_name.strip().casefold() == "sugammadex":
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
        """Start a stimulation profile."""
        if not profile:
            return
        self.set_disturbance_profile(profile)
        self.disturbance_active = True
        self.disturbance_start_time = self.state.time

    def set_disturbance_profile(self, profile: str | None):
        """Select a disturbance profile without activating it; None clears it."""
        profile = profile or None
        self.disturbances = Disturbances(profile)
        self.disturbance_profile = self.config.disturbance_profile = profile

    def stop_disturbance(self, clear_profile: bool = False):
        """Stop the active stimulation profile."""
        self.disturbance_active = False
        if clear_profile:
            self.set_disturbance_profile(None)

    def set_bair_hugger(self, target_c: float):
        """Set the forced-air warmer target (°C); 0 turns it off."""
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

    def get_resp_step_kwargs(self, total_assisted_mv, mech_rr, mech_vt_l, cardiac_output):
        """Respiratory-model inputs shared by the runtime and startup projection."""
        return {
            "ce_prop": hypnotic_equivalent(
                self.pk_prop.state.ce_resp,
                self.state.etomidate_ce,
                self.state.midazolam_ce,
                self.midazolam_c50,
                ventilation=True,
            ),
            "ce_remi": opioid_equivalent(self.pk_remi.state.ce_resp, self.state.fentanyl_ce),
            "mech_vent_mv": total_assisted_mv,
            "fio2": self.state.fio2,
            "ce_roc": self.tof_pd.ce_central,
            "mac_sevo": self.state.mac_sevo,
            "mech_rr": mech_rr,
            "mech_vt_l": mech_vt_l,
            "measured_breaths": self.state.airway_mode != AirwayType.NONE and self.vent.has_measured_breath,
            "airway_patency": self._airway_patency,
            "ventilation_efficiency": self._ventilation_efficiency,
            "vq_mismatch": self._vq_mismatch,
            "hb_g_dl": self.hemo.hb_conc,
            "blood_volume_ml": self.hemo.blood_volume,
            "cardiac_output": cardiac_output,
            "metabolic_factor": max(0.5, self._metabolic_factor),
            "unconscious": self.state.loc,
            "lung_volume_l": self.aeration.frc,
            "shunt_fraction": self.aeration.shunt_fraction,
        }

    def set_bronchospasm(self, severity: float):
        self.bronchospasm_manual = clamp(severity, 0.0, 1.0)

    def set_auto_laryngospasm(self, enabled: bool):
        self.auto_laryngospasm_enabled = bool(enabled)

    @property
    def laryngospasm_level(self) -> str:
        """Laryngospasm severity as shown to the learner."""
        severity = self.laryngospasm_severity
        if severity < 0.05:
            return "none"
        if severity < 0.3:
            return "mild"
        if severity < 0.6:
            return "moderate"
        return "severe"

    def set_rhythm(self, rhythm_name: str):
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
        """Start or stop manual bag ventilation at rr breaths/min and vt liters.

        It ventilates only through a mask or ETT and yields to the ventilator.
        """
        self.bag_mask_active = active
        if active:
            self.bag_mask_rr = clamp(rr, 0.0, 40.0)
            self.bag_mask_vt = clamp(vt, 0.05, 1.2)
        self.actions.record(
            self.state.time, ACTION_BAG_MASK, label="on" if active else "off"
        )

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def measure_nibp(self):
        """Start a cuff measurement and schedule the next automatic cycle."""
        if self.nibp.is_cycling:
            return
        self.nibp.trigger()
        self.state.nibp_is_cycling = True
        self.state.nibp_measurement_failed = False
        self.state.nibp_cuff_pressure = 0.0
        self._next_nibp_time = self.state.time + self.nibp.interval

    def start_recording(self, output_dir: str = ".", sample_interval_sec: float = 1.0):
        """Start CSV recording, raising RecordingError on file failures."""
        if self.recorder and self.recorder.is_recording:
            return
        self.recorder = recorder = DataRecorder(output_dir=output_dir, sample_interval_sec=sample_interval_sec)
        recorder.start()

    def stop_recording(self):
        """Flush and close the CSV, raising RecordingError on failure."""
        if self.recorder:
            self.recorder.stop()

    def step(self, dt: float):
        """Advance by dt; a RecordingError occurs after the state has advanced."""
        if dt <= 0 or not self.running:
            return
        runtime_core.step_simulation(self, dt)
        self.state.time += dt
        self._append_output_snapshot()
        if self.recorder and self.recorder.is_recording:
            self.recorder.log(self.state)

    def get_latest_state(self) -> SimulationState:
        """Return a copy of the current state."""
        return copy.copy(self.state)

    def get_predicted_csht(self, drug: str) -> float:
        """Minutes for the "propofol", "remi", or "fentanyl" effect site to halve if stopped now."""
        models = {
            "propofol": (self.pk_prop, 3600),
            "remi": (self.pk_remi, 1200),
            "fentanyl": (self.pk_fentanyl, 7200),
        }
        if drug not in models:
            return 0.0
        model, max_seconds = models[drug]
        return model.simulate_decay(target_fraction=0.5, max_seconds=max_seconds)

    def set_vent_power(self, on: bool):
        """Start or stop the ventilator; its settings persist while it is off."""
        self.vent.is_on = bool(on)

    def set_vent_settings(self, rr: float, vt: float, peep: float, ie: str,
                          mode: str, p_insp: float | None = None, fio2: float | None = None, **extra):
        """Set the ventilator without starting or stopping it.

        Args:
            rr: Mandatory rate, or the PSV apnea backup rate (breaths/min).
            vt: Set or targeted tidal volume (L).
            peep: PEEP (cmH2O).
            ie: I:E ratio such as "1:2" for VCV, PCV, and PCV-VG.
            mode: One of anasim.machine.ventilator.MODES.
            p_insp: Inspiratory pressure above PEEP (cmH2O) for PCV, SIMV-PC,
                and PSV backup breaths.
            fio2: Target FiO2; rebalances O2 and air flows.
            extra: p_support, t_insp, pause, p_max, or trigger; see VentSettings.
        """
        settings = dict(rr=rr, tv=vt * 1000, peep=peep, ie=ie, mode=mode, **extra)
        if p_insp is not None:
            settings["p_insp"] = p_insp
        if fio2 is not None:
            settings["fio2"] = fio2
        self.vent.update_settings(**settings)

        if fio2 is not None:
            self._apply_fio2_blender(self.vent.settings.fio2)

    def _apply_fio2_blender(self, fio2: float):
        """Split the O2 plus air flow to reach the target FiO2, keeping N2O fixed.

        Solves FiO2 = (O2 + 0.21 x air) / (O2 + air + N2O) for O2.
        """
        total_non_n2o = self.circuit.fgf_o2 + self.circuit.fgf_air
        if total_non_n2o <= 0:
            return
        target = clamp(fio2, 0.21, 1.0)
        n2o_flow = max(0.0, self.circuit.fgf_n2o)
        o2_flow = ((target * (total_non_n2o + n2o_flow)) - 0.21 * total_non_n2o) / 0.79
        o2_flow = clamp(o2_flow, 0.0, total_non_n2o)
        air_flow = total_non_n2o - o2_flow
        self.circuit.fgf_o2 = o2_flow
        self.circuit.fgf_air = air_flow
