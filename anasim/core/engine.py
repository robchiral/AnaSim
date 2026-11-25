import copy
from collections import deque
from dataclasses import dataclass

import numpy as np

from anasim.core.constants import (
    APL_DEFAULT_CMH2O,
    APL_RANGE_CMH2O,
    O2_FLUSH_DURATION_S,
    AirwayTuning,
)
from anasim.core.enums import PULSELESS_RHYTHMS, RhythmType
from anasim.core.recorder import DataRecorder
from anasim.core.utils import clamp, clamp01
from anasim.machine.circuit import CircleSystem
from anasim.machine.ventilator import AnesthesiaVentilator
from anasim.machine.volatile import Vaporizer
from anasim.monitors.airway import AirwaySensor
from anasim.monitors.alarms import THRESHOLDS, AlarmSystem
from anasim.monitors.arterial import ArterialLineMonitor, ArterialWaveformRenderer
from anasim.monitors.blood_gas import BloodGasResult
from anasim.monitors.capno import Capnograph, EtCO2Readout
from anasim.monitors.cardiac_cycle import (
    CardiacCycle,
    CardiacCycleSample,
    CompressionClock,
)
from anasim.monitors.ecg import ECGMonitor
from anasim.monitors.neuromuscular import NeuromuscularMonitor
from anasim.monitors.nibp import NIBPMonitor
from anasim.monitors.spo2 import SpO2Monitor
from anasim.patient.conditions import COPD_RESISTANCE, POSITIONS
from anasim.patient.domain import bounded_number, finite_number
from anasim.patient.patient import Patient
from anasim.patient.pd import (
    BISModel,
    ClinicalResponseModel,
    TOFModel,
    midazolam_loss_of_response,
)
from anasim.patient.pk_models import (
    ANESTHETIZED_OUTPUT,
    NorepinephrinePK,
    janmahasatian_ffm,
)
from anasim.patient.variability import sample_variability
from anasim.patient.volatile_pk import VolatilePK
from anasim.physiology.arrhythmia import ArrhythmiaModel, antiarrhythmic_effects
from anasim.physiology.cerebral import CerebralModel
from anasim.physiology.disturbances import Disturbances
from anasim.physiology.events import SeverityRamp
from anasim.physiology.hemodynamics import MAX_HEMATOCRIT, HemodynamicModel
from anasim.physiology.lactate import LactateModel
from anasim.physiology.lung import LungAeration
from anasim.physiology.malignant_hyperthermia import (
    SUCCINYLCHOLINE_TRIGGER_WINDOW_S,
    MalignantHyperthermia,
    triggering_exposure,
)
from anasim.physiology.resp_mech import RespiratoryMechanics
from anasim.physiology.respiration import RespiratoryModel
from anasim.physiology.resuscitation import (
    SHOCK_ENERGY_J,
    Defibrillator,
    ShockResult,
    compression_pressures,
    shock_outcome,
)
from anasim.physiology.thermal import ThermalModel
from anasim.physiology.toxicity import LocalAnestheticToxicity

from . import monitors as monitor_core
from . import projection as projection_core
from . import runtime as runtime_core
from .action_log import (
    ACTION_AIRWAY,
    ACTION_BAG_MASK,
    ACTION_BLOOD_GAS,
    ACTION_COMPRESSIONS,
    ACTION_COOLING,
    ACTION_DRUG_BOLUS,
    ACTION_EVENT_START,
    ACTION_EVENT_STOP,
    ACTION_FGF,
    ACTION_FLUID,
    ACTION_INTUBATION_ATTEMPT,
    ACTION_LIPID_INFUSION,
    ACTION_O2_FLUSH,
    ACTION_OXYGEN_SUPPLY,
    ACTION_POSITION,
    ACTION_SHOCK,
    ACTION_VAPORIZER,
    ACTION_VENTILATION,
    ActionLog,
)
from .clinical_actions import ActionFeedback, check_intubation, check_shock
from .drug_api import POPULATION_TCI_DRUGS, DrugControllerMixin
from .drug_registry import DRUG_REGISTRY, get_drug_spec, resolve_bolus_drug
from .initialization import initialize_engine_state
from .state import (
    AirwayStatus,
    AirwayType,
    SimulationConfig,
    SimulationState,
    WaveformSample,
)
from .tci import TCIController

OUTPUT_WINDOW_S = 20.0  # Waveform history kept in output_buffer
BAG_RR_RANGE = (1.0, 40.0)  # breaths/min
BAG_VT_RANGE_L = (0.05, 1.2)


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
        if config.patient_variability:
            patient = copy.copy(patient)
            patient.variability = sample_variability(config.rng_seed)
        self.patient = patient
        self.config = config
        self.state = SimulationState()
        self.actions = ActionLog()
        self.airway_tuning = AirwayTuning()

        # Infusion rates (model units/s) and active TCI controllers, by drug key.
        self.infusion_rates = {spec.key: 0.0 for spec in DRUG_REGISTRY if spec.has_infusion}
        self.tci: dict[str, TCIController] = {}
        self._pump_needs_reset: set[str] = set()
        self._next_nibp_time = 0.0

        self.disturbances = Disturbances(config.disturbance_profile)
        self.disturbance_active = bool(config.disturbance_profile)
        self.disturbance_start_time = 0.0
        # Learner intubation adds a transient without replacing ongoing surgery.
        self.intubation_stimulus = Disturbances("stim_intubation_pulse")
        self.intubation_start_time: float | None = None
        paco2_drop = max(0.0, 40.0 - self.patient.pregnancy.paco2)
        alarm_thresholds = (
            {"EtCO2_min": max(20, round(THRESHOLDS["EtCO2_min"] - paco2_drop))}
            if paco2_drop > 0.5
            else None
        )
        self.alarms = AlarmSystem(thresholds=alarm_thresholds)

        self.output_buffer: deque[WaveformSample] = deque()
        self._preroll_trends: list[dict] = []
        self.recorder: DataRecorder | None = None
        self._tci_accumulators: dict[str, float] = {}
        self.running = False

        self.bag_mask_active = False
        self.bag_mask_rr = 12.0  # breaths/min
        self.bag_mask_vt = 0.5  # L
        self.apl_cmh2o = APL_DEFAULT_CMH2O
        self._o2_flush_remaining_s = 0.0
        self._manual_breath_requested = False
        self._last_succinylcholine_time: float | None = None

        self.active_hemorrhage = False
        self.hemorrhage_rate_ml_min = 500.0

        # Fluid and blood boluses run in over minutes.
        self.pending_infusions: list[PendingInfusion] = []
        self.fluid_infusion_rate_ml_min = 150.0
        self.blood_infusion_rate_ml_min = 75.0
        self.maintenance_fluid_rate_ml_min = 0.0
        self.lipid_infusion_ml_min = 0.0

        self.auto_laryngospasm_enabled = True
        self.mask_seal = 1.0
        self.airway_obstruction_manual = 0.0
        self.bronchospasm_manual = 0.0
        self.laryngospasm_severity = 0.0
        self.airway_status = AirwayStatus()

        self.smooth_bis = 98.0
        self._bis_noise = 0.0
        self.etco2_readout = EtCO2Readout()
        self._tol_current = 0.0
        self._pk_hemo_scale_cache: tuple[float, float, float, float] | None = None
        self._pk_blood_out_ml = 0.0
        self.current_mean_paw = 0.0
        self.current_mean_effort = 0.0
        self._apl_pressure = 0.0
        self._extubation_stimulus_until = 0.0
        self._tracheal_gas: tuple[float, float, float, float] = (0.21, 0.0, 0.0, 0.79)
        self._step_gas: projection_core.StepGasCache | None = None

        # Independent generators keep each monitor's noise reproducible.
        self.rng = np.random.default_rng(self.config.rng_seed)
        self._ecg_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        self._nibp_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        self._cardiac_rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        # Derived without drawing from self.rng, so the other streams keep their seeds.
        # [seed, 1] is reserved for sample_variability().
        seed = self.config.rng_seed
        self._shock_rng = np.random.default_rng(None if seed is None else [seed, 3])
        self.arrhythmias = ArrhythmiaModel(np.random.default_rng(None if seed is None else [seed, 2]))

        self._pulseless_s = 0.0
        self.compressions_started_at: float | None = None
        self.defibrillator = Defibrillator()
        self._last_shock_feedback: ActionFeedback | None = None

        self.initialize_models()
        self.set_continuous_fluid_rate(self.config.maintenance_fluid_ml_hr)
        self.initialize_state()

    def initialize_state(self):
        """Seed the public state from the initialized subsystems."""
        self._step_gas = None
        self.state.temp_c = self.patient.baseline_temp
        self.state.fluid_warmer = self.config.fluid_warmer_enabled
        initialize_engine_state(self)
        self.actions = ActionLog()
        projection_core.sync_state_from_models(self)
        self._bis_anesthetized = self.config.mode == "steady_state" or self.state.display_bis < 65.0
        monitor_core.seed_nibp_reading(self)
        self.nmb.establish_baseline(self.state.twitch / 100.0)
        state = self.state
        self.output_buffer.append(WaveformSample(
            state.time, state.ecg_voltage, state.pleth_voltage, state.capno_co2, state.art_pressure,
            state.paw, state.flow, state.volume, self.vent.breath_count,
        ))

    def set_continuous_fluid_rate(self, ml_hr: float | None):
        """Set continuous IV fluids in mL/hr; None restores 1 mL/kg/hr."""
        rate = self.patient.weight if ml_hr is None else max(0.0, finite_number("maintenance fluid rate", ml_hr))
        self.maintenance_fluid_rate_ml_min = rate / 60.0

    def get_continuous_fluid_rate(self) -> float:
        """Return current continuous IV fluid rate in mL/hr."""
        return self.maintenance_fluid_rate_ml_min * 60.0

    @property
    def _tracheal_fio2(self) -> float:
        return self._tracheal_gas[0]

    @_tracheal_fio2.setter
    def _tracheal_fio2(self, value: float) -> None:
        o2 = clamp01(float(value))
        _, sevo, n2o, _ = self._tracheal_gas
        rem = max(0.0, 1.0 - o2)
        other = sevo + n2o
        if other > rem and other > 0.0:
            scale = rem / other
            sevo *= scale
            n2o *= scale
        n2 = max(0.0, rem - sevo - n2o)
        self._tracheal_gas = (o2, sevo, n2o, n2)

    def initialize_models(self):
        """Build the subsystem models selected by the configuration."""
        self.thermal = ThermalModel(self.patient)
        # Anaphylaxis is full in about 2 min and resolves in about 10; sepsis
        # is full in about 10 min and resolves over 30 or more.
        self.anaphylaxis = SeverityRamp(onset_rate=0.5 / 60.0, decay_rate=0.1 / 60.0)
        self.sepsis = SeverityRamp(onset_rate=0.1 / 60.0, decay_rate=0.03 / 60.0)
        self.malignant_hyperthermia = MalignantHyperthermia()

        # Intravenous drug PK, keyed like the drug registry.
        self.pk = {spec.key: spec.pk_model(self.patient, self.config) for spec in DRUG_REGISTRY}
        self._pk_sync_entries = tuple(
            (self.pk[spec.key], spec.ce_field, spec.cp_field) for spec in DRUG_REGISTRY
        )
        p = self.patient
        pump_patient = Patient(age=p.age, weight=p.weight, height=p.height, sex=p.sex)
        self.pump_pk = {key: get_drug_spec(key).pk_model(pump_patient, self.config) for key in POPULATION_TCI_DRUGS}
        # Published models were fitted during anesthesia, so they apply at its output.
        for model in self.pump_pk.values():
            model.update_hemodynamics(1.0, ANESTHETIZED_OUTPUT)
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
        # Fresh gas supplies N2O even when sevoflurane is disabled.
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
        self.resp.attach_fluid_balance(self.hemo.fluids, self.hemo.config.donor_mchc_g_dl)
        self.pk_sevo.frc = self.resp.gas_volume_l
        self.pk_n2o.frc = self.resp.gas_volume_l
        self.lactate = LactateModel(self.patient.weight, self.patient.hepatic_function)
        self.cerebral = CerebralModel(
            self.patient.baseline_hb, self.resp.paco2_setpoint, self.patient.baseline_map,
        )
        # Scale lipid rescue volume/EC50 with fat-free mass matching LidocainePK
        # (equals TBW for the 70 kg, 170 cm male reference adult).
        ref_ffm = janmahasatian_ffm(70.0, 170.0, True)
        patient_ffm = janmahasatian_ffm(
            self.patient.weight, self.patient.height, self.patient.sex == "male",
        )
        self.toxicity = LocalAnestheticToxicity(70.0 * patient_ffm / ref_ffm)
        if self.config.mode == "steady_state":
            self.aeration = LungAeration(self.patient, recruited=0.9, atelectatic=LungAeration.INDUCTION_ATELECTASIS)
        else:
            self.aeration = LungAeration(self.patient)
        self.resp_mech = RespiratoryMechanics(resistance=self.patient.airway_resistance(), aeration=self.aeration)
        self.resp_mech.chronic_obstruction = self.patient.copd_severity
        self.resp_mech.chronic_resistance = COPD_RESISTANCE * self.patient.copd_severity
        self._base_airway_resistance = self.resp_mech.resistance
        self.set_vent_settings(rr=12.0, vt=0.5, peep=5.0, ie="1:2", mode="VCV", p_insp=15.0)
        self.resp.baseline_co_l_min = self.hemo.base_co_l_min

        self.bis = BISModel(self.patient)
        self.capno = Capnograph(self.resp.vd_deadspace)
        self.airway_sensor = AirwaySensor()
        self.response = ClinicalResponseModel(self.patient.age, self.patient.variability.sevoflurane_mac)
        self.tof_pd = TOFModel(self.patient)

        self.cardiac_cycle = CardiacCycle(rng=self._cardiac_rng)
        self.compression_clock = CompressionClock()
        self.arterial_waveform = ArterialWaveformRenderer(self.patient.age, bsa=self.patient.bsa)
        resting_sbp, resting_dbp = self.arterial_waveform.steady_pressures(
            self.patient.baseline_map, self.hemo.base_sv, self.hemo.base_hr,
        )
        self.hemo.calibrate_coronary(resting_sbp, resting_dbp)
        self.art_line = ArterialLineMonitor()
        self.ecg = ECGMonitor(rng=self._ecg_rng)
        self.spo2_mon = SpO2Monitor()
        self.nibp = NIBPMonitor(interval_min=5.0, rng=self._nibp_rng)
        self.nmb = NeuromuscularMonitor()
        self.blood_gases: list[BloodGasResult] = []
        self.state.nibp_interval_sec = self.nibp.interval

    @property
    def pk_nore(self) -> NorepinephrinePK:
        """Norepinephrine PK, which also takes a propofol clearance covariate."""
        model = self.pk["nore"]
        assert isinstance(model, NorepinephrinePK)
        return model

    def set_fgf(self, o2_l_min: float, air_l_min: float, n2o_l_min: float = 0.0):
        """Set fresh gas flows in L/min."""
        for gas, requested in (
            ("o2", o2_l_min),
            ("air", air_l_min),
            ("n2o", n2o_l_min),
        ):
            attr = f"fgf_{gas}"
            flow = max(0.0, finite_number(f"{gas} flow", requested))
            if getattr(self.circuit, attr) != flow:
                self.actions.record(self.state.time, ACTION_FGF, label=gas, amount=flow)
            setattr(self.circuit, attr, flow)

    def set_position(self, position: str):
        """Change maternal position; venous return responds over seconds."""
        if position not in POSITIONS:
            raise ValueError("position must be 'supine' or 'left_lateral'")
        self.hemo.position = position
        self.actions.record(self.state.time, ACTION_POSITION, label=position)

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
        self.vaporizer.set_concentration(finite_number("vaporizer setting", percent))
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

    def give_saline(self, volume_ml: float):
        """Queue 0.9% saline at the crystalloid infusion rate."""
        self._queue_infusion(volume_ml, self.fluid_infusion_rate_ml_min, hematocrit=0.0, label="saline")

    def give_calcium_chloride(self, mg: float):
        """Give calcium chloride dihydrate; 1 g contains 6.8 mmol calcium."""
        dose = bounded_number("calcium chloride dose", mg, 0.0, 10000.0, unit="mg")
        if dose == 0.0:
            raise ValueError("calcium chloride dose must be greater than zero")
        circulation = self.hemo.circulation_fraction()
        self.hemo.fluids.give_calcium_chloride(dose * 0.0068, circulation=circulation, bolus_push_s=90.0)
        self.state.ionized_calcium = float(self.hemo.fluids.ionized_calcium(self.resp.state.ph_37))
        self.actions.record(self.state.time, ACTION_DRUG_BOLUS, label="calcium_chloride", amount=dose)

    def give_blood(self, volume_ml: float = 300.0, hematocrit: float = 0.55):
        """Queue packed red cells at 75 mL/min."""
        hct = bounded_number("blood hematocrit", hematocrit, 0.01, MAX_HEMATOCRIT)
        self._queue_infusion(
            volume_ml,
            self.blood_infusion_rate_ml_min,
            hematocrit=hct,
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

    def give_lipid_emulsion(self, volume_ml: float):
        """Give 20% lipid over 2.5 minutes; continued infusion is set separately."""
        volume = bounded_number("lipid volume", volume_ml, 0.0, 3000.0, unit="mL")
        if volume <= 0:
            raise ValueError("Lipid volume must be positive")
        self._queue_infusion(volume, volume / 2.5, hematocrit=0.0,
                             retention_fraction=0.25, label="lipid")

    def set_lipid_infusion(self, ml_min: float):
        self.lipid_infusion_ml_min = bounded_number("lipid infusion", ml_min, 0.0, 200.0, unit="mL/min")
        self.actions.record(self.state.time, ACTION_LIPID_INFUSION, amount=self.lipid_infusion_ml_min)

    def _queue_infusion(
        self,
        volume_ml: float,
        rate_ml_min: float,
        hematocrit: float,
        retention_fraction: float | None = None,
        label: str = "crystalloid",
    ):
        volume_ml = finite_number(f"{label} volume", volume_ml)
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
        """Give a hand bolus in the drug's registry unit; it adds dose / V1 to C1.

        Population TCI pump models omit it. Sugammadex goes to the TOF model for binding.
        """
        amount = finite_number("bolus amount", amount)
        if amount <= 0:
            return

        if drug_name.strip().casefold() == "sugammadex":
            self.tof_pd.give_sugammadex(amount)
            self.actions.record(
                self.state.time, ACTION_DRUG_BOLUS, label="sugammadex", amount=amount
            )
            return

        spec = resolve_bolus_drug(drug_name)
        model = self.pk[spec.key]
        dose = amount * spec.bolus_model_scale
        model.state.c1 += dose / model.v1
        if spec.key == "succinylcholine":
            self._last_succinylcholine_time = self.state.time
        self.sync_active_tci_from_pk(spec.key)
        self.actions.record(
            self.state.time, ACTION_DRUG_BOLUS, label=spec.key, amount=amount
        )

    def event_active(self, name: str) -> bool:
        """Return whether a clinical event is running."""
        if name == "hemorrhage":
            return self.active_hemorrhage
        if name == "anaphylaxis":
            return self.anaphylaxis.active
        if name == "sepsis":
            return self.sepsis.active
        if name == "malignant_hyperthermia":
            return self.malignant_hyperthermia.active
        raise ValueError(f"Unknown clinical event: {name!r}")

    def _log_event(self, name: str, active: bool, amount: float = 0.0):
        """Log a learner-triggered event transition before it is applied."""
        if self.event_active(name) != active:
            action = ACTION_EVENT_START if active else ACTION_EVENT_STOP
            self.actions.record(self.state.time, action, label=name, amount=amount)

    def start_hemorrhage(self, rate_ml_min: float = 500.0):
        rate_ml_min = finite_number("rate_ml_min", rate_ml_min)
        if rate_ml_min < 0.0:
            raise ValueError("rate_ml_min must not be negative")
        self.hemorrhage_rate_ml_min = rate_ml_min
        self._log_event("hemorrhage", True, amount=rate_ml_min)
        self.active_hemorrhage = True

    def stop_hemorrhage(self):
        self._log_event("hemorrhage", False)
        self.active_hemorrhage = False

    def start_anaphylaxis(self):
        self._log_event("anaphylaxis", True)
        self.anaphylaxis.active = True

    def stop_anaphylaxis(self):
        self._log_event("anaphylaxis", False)
        self.anaphylaxis.active = False

    def start_sepsis(self):
        self._log_event("sepsis", True)
        self.sepsis.active = True

    def stop_sepsis(self):
        self._log_event("sepsis", False)
        self.sepsis.active = False

    def stop_events(self):
        self.stop_hemorrhage()
        self.stop_anaphylaxis()
        self.stop_sepsis()
        self.stop_malignant_hyperthermia()
        self.stop_disturbance()
        self.intubation_start_time = None

    def start_malignant_hyperthermia(self):
        """Start the crisis now; onset after a trigger varies, so the instructor or case sets it."""
        recent_sux = (
            self._last_succinylcholine_time is not None
            and 0.0 <= self.state.time - self._last_succinylcholine_time <= SUCCINYLCHOLINE_TRIGGER_WINDOW_S
        )
        if not triggering_exposure(
            self.state.mac_sevo, self.pk["succinylcholine"].state.c1,
            recent_succinylcholine=recent_sux,
        ):
            raise ValueError("Malignant hyperthermia requires sevoflurane anesthesia or recent succinylcholine.")
        self._log_event("malignant_hyperthermia", True)
        self.malignant_hyperthermia.active = True

    def stop_malignant_hyperthermia(self):
        """Stop the crisis; it subsides without treatment."""
        self._log_event("malignant_hyperthermia", False)
        self.malignant_hyperthermia.active = False

    def start_disturbance(self, profile: str):
        """Start a stimulation profile."""
        if not profile:
            return
        self.set_disturbance_profile(profile)
        self.disturbance_active = True
        self.disturbance_start_time = self.state.time

    @property
    def disturbance_profile(self) -> str | None:
        return self.config.disturbance_profile

    @disturbance_profile.setter
    def disturbance_profile(self, value: str | None) -> None:
        self.config.disturbance_profile = value

    def set_disturbance_profile(self, profile: str | None):
        """Select a disturbance profile without activating it; None clears it."""
        profile = profile or None
        self.disturbances = Disturbances(profile)
        self.config.disturbance_profile = profile

    def stop_disturbance(self, clear_profile: bool = False):
        """Stop the active stimulation profile."""
        self.disturbance_active = False
        if clear_profile:
            self.set_disturbance_profile(None)

    def set_bair_hugger(self, target_c: float):
        """Set the forced-air warmer target (°C); 0 turns it off, active range 32-43 °C."""
        val = bounded_number("warmer target", target_c, 0.0, 43.0, unit="°C")
        if 0.0 < val < 32.0:
            raise ValueError("warmer target must be 0 °C (off) or between 32 and 43 °C")
        self.state.bair_hugger_target = val

    def set_fluid_warmer(self, enabled: bool):
        """Turn the IV fluid and blood warmer (37 °C) on or off."""
        self.state.fluid_warmer = bool(enabled)

    def set_surface_cooling(self, on: bool):
        """Start or stop surface cooling; it continues until stopped."""
        if bool(on) != self.state.surface_cooling:
            self.actions.record(self.state.time, ACTION_COOLING, label="on" if on else "off")
        self.state.surface_cooling = bool(on)

    def attempt_intubation(self) -> ActionFeedback | None:
        """Attempt learner intubation, including stimulation even if placement fails."""
        if self.state.airway_mode == AirwayType.ETT:
            return None
        feedback = check_intubation(self)
        failed = feedback is not None and feedback.blocked
        self.intubation_start_time = self.state.time
        self.actions.record(
            self.state.time, ACTION_INTUBATION_ATTEMPT, label="failed" if failed else "placed",
        )
        if not failed:
            self.set_airway_mode("ETT")
            if self.bag_mask_active:
                self.set_bag_mask_ventilation(False)
        return feedback

    def set_airway_mode(self, mode_str: str):
        """Set the connection directly; learner tube placement uses attempt_intubation."""
        previous_mode = self.state.airway_mode
        try:
            self.state.airway_mode = AirwayType(mode_str)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in AirwayType)
            raise ValueError(
                f"Unsupported airway mode {mode_str!r}; choose one of: {choices}"
            ) from exc
        if previous_mode == AirwayType.ETT and self.state.airway_mode != AirwayType.ETT:
            projection_core.sync_inhaled_agents(self)
            projection_core.sync_pk_state(self)
            if (
                projection_core.loss_of_response(self) >= 0.20
                and projection_core.extubation_tolerance(self) < 0.65
            ):
                self._extubation_stimulus_until = self.state.time + 25.0
        self.actions.record(
            self.state.time, ACTION_AIRWAY, label=self.state.airway_mode.value
        )
        if self.state.airway_mode == AirwayType.ETT:
            self.laryngospasm_severity = 0.0
            self.state.laryngospasm = 0.0
        if self.state.airway_mode == AirwayType.NONE and self.bag_mask_active:
            self.set_bag_mask_ventilation(False)
        if self.state.airway_mode == AirwayType.NONE:
            self._manual_breath_requested = False

    def set_mask_seal(self, seal: float):
        """Set the facemask seal fraction, from 0 (open leak) to 1 (tight seal)."""
        self.mask_seal = bounded_number("mask seal", seal, 0.0, 1.0)
        self.state.mask_seal = self.mask_seal

    def set_airway_obstruction(self, severity: float):
        self.airway_obstruction_manual = bounded_number("airway obstruction", severity, 0.0, 1.0)

    def set_bronchospasm(self, severity: float):
        self.bronchospasm_manual = bounded_number("bronchospasm", severity, 0.0, 1.0)

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

    def _sync_cardiac_state(self) -> None:
        """Project immediate rhythm, compression, or post-shock changes into public hemodynamic landmarks."""
        self.hemo.post_shock_asystole = self.arrhythmias.post_shock_asystole
        hemo_state = self.hemo.state
        projection_core.project_hemodynamics(self, hemo_state)
        if hemo_state.compressions:
            cpr_sbp, cpr_dbp = compression_pressures(
                hemo_state.cpr_perfusion_map,
                hemo_state.venous_pressure,
                filling_factor=self.hemo.cpr_filling_factor,
            )
            if hemo_state.intrinsic_co > 0.0 and hemo_state.rhythm_type not in PULSELESS_RHYTHMS:
                i_sbp, i_dbp = self.arterial_waveform.reference_pressures(
                    hemo_state.intrinsic_map,
                    hemo_state.sv,
                    hemo_state.hr,
                    pulse_pressure_factor=hemo_state.pulse_pressure_factor,
                    static_pressure=hemo_state.venous_pressure,
                )
                sbp, dbp = max(i_sbp, cpr_sbp), max(i_dbp, cpr_dbp)
            else:
                sbp, dbp = cpr_sbp, cpr_dbp
        elif hemo_state.rhythm_type in PULSELESS_RHYTHMS or hemo_state.co <= 0.0 or hemo_state.hr <= 0.0:
            sbp = dbp = float(hemo_state.venous_pressure)
        else:
            sbp, dbp = self.arterial_waveform.reference_pressures(
                hemo_state.map,
                hemo_state.sv,
                hemo_state.hr,
                pulse_pressure_factor=hemo_state.pulse_pressure_factor,
                static_pressure=hemo_state.venous_pressure,
            )
        self.state.sbp = float(max(sbp, self.state.map))
        self.state.dbp = float(min(dbp, self.state.map))
        if hemo_state.rhythm_type in PULSELESS_RHYTHMS or hemo_state.hr <= 0.0:
            self.state.display_hr = 0.0

    def set_rhythm(self, rhythm_name: str | RhythmType):
        rhythm: RhythmType | None
        if isinstance(rhythm_name, RhythmType):
            rhythm = rhythm_name
        else:
            normalized = str(rhythm_name).upper()
            rhythm = next(
                (item for item in RhythmType if item.value == rhythm_name or item.name == normalized),
                None,
            )
        if rhythm is None:
            raise ValueError(f"Unknown rhythm: {rhythm_name!r}")
        self.hemo.rhythm_type = rhythm
        self.arrhythmias.reset()
        self.cardiac_cycle.clear_rate_history()
        self.hemo.invalidate_state_cache()
        self._sync_cardiac_state()

    def set_chest_compressions(self, active: bool):
        """Start or stop chest compressions at the guideline rate."""
        active = bool(active)
        if active != self.hemo.compressions:
            self.compressions_started_at = self.state.time if active else None
            self.actions.record(self.state.time, ACTION_COMPRESSIONS, label="on" if active else "off")
        self.hemo.compressions = active
        self._sync_cardiac_state()

    def charge_defibrillator(self) -> None:
        """Start a 5-second charge to 200 J."""
        self.defibrillator.charge(self.state.time)

    def disarm_defibrillator(self) -> None:
        """Cancel charging or a pending synchronized shock."""
        self.defibrillator.disarm()

    def defibrillate(self, synchronized: bool = False) -> ShockResult:
        """Discharge a ready device, or arm it to fire at the next R wave."""
        device = self.defibrillator
        status = device.status(self.state.time)
        if status == "waiting_for_r":
            return ShockResult.PENDING
        if status != "ready":
            return ShockResult.NOT_READY
        if synchronized:
            if self.hemo.state.hr <= 0.0 or self.hemo.state.rhythm_type in PULSELESS_RHYTHMS:
                return ShockResult.NOT_DELIVERED
            device.waiting_for_r = True
            return ShockResult.PENDING
        self._deliver_shock(self.state.time, synchronized=False)
        return ShockResult.DELIVERED

    def step_defibrillator(self, time: float, cycle: CardiacCycleSample) -> bool:
        """Fire a pending synchronized shock at an R wave; return whether it fired."""
        device = self.defibrillator
        if device.status(time) != "waiting_for_r":
            return False
        if not cycle.organized:
            device.waiting_for_r = False
            return False
        if not cycle.beat_started:
            return False
        self._deliver_shock(time - cycle.elapsed_s, synchronized=True)
        return True

    def _deliver_shock(self, time: float, *, synchronized: bool) -> None:
        self._last_shock_feedback = check_shock(self)
        prev_rhythm = self.hemo.rhythm_type
        cycle = self.cardiac_cycle.current
        hypoxia, energy_failure, silent = runtime_core.myocardial_substrate(self)
        rhythm = shock_outcome(
            prev_rhythm, synchronized=synchronized, energy_failure=energy_failure,
            electrically_silent=silent,
            vulnerable=cycle.rhythm_type == self.hemo.state.rhythm_type and self.ecg.vulnerable(cycle),
            antiarrhythmic=max(antiarrhythmic_effects(
                self.pk["lidocaine"].state.ce_cv, self.state.amiodarone_ce,
            )),
            rng=self._shock_rng,
            hypoxia=hypoxia,
            perfusing=runtime_core.is_perfusing(self),
            energy_reserve=self.hemo.coronary.energy_reserve,
            temp_c=self.state.temp_c,
            hyperkalemia_block=self.hemo.hyperkalemia_block,
        )
        self.defibrillator.consume(time, synchronized)
        self.ecg.trigger_shock_artifact()
        self.actions.record(time, ACTION_SHOCK,
                            label="synchronized" if synchronized else "unsynchronized", amount=SHOCK_ENERGY_J)
        self.hemo.coronary.apply_shock(SHOCK_ENERGY_J)
        # An electrically silent heart keeps its underlying rhythm.
        if not silent and prev_rhythm != RhythmType.ASYSTOLE:
            self.hemo.rhythm_type = self.arrhythmias.after_shock(
                prev_rhythm, rhythm, ischemia=self.hemo.coronary.ischemia,
                hypoxia=hypoxia, energy_failure=energy_failure,
                energy_reserve=self.hemo.coronary.energy_reserve,
                perfusing=runtime_core.is_perfusing(self),
            )
        self.hemo.invalidate_state_cache()
        if not synchronized and self.hemo.rhythm_type != prev_rhythm:
            self.cardiac_cycle.clear_rate_history()
        self._sync_cardiac_state()

    def set_bag_mask_ventilation(
        self,
        active: bool,
        rr: float | None = None,
        vt: float | None = None,
        p_max: float | None = None,
    ):
        """Start or stop manual bag ventilation at rr breaths/min and vt liters.

        Pass p_max to set the APL valve in cmH2O. Omitted settings keep their
        values. Bag ventilation requires a mask or ETT and yields to the ventilator.
        """
        if rr is not None or vt is not None or p_max is not None:
            self.set_bag_mask_settings(
                self.bag_mask_rr if rr is None else rr,
                self.bag_mask_vt if vt is None else vt,
                p_max=p_max,
            )
        self.bag_mask_active = bool(active)
        self.actions.record(
            self.state.time, ACTION_BAG_MASK, label="on" if active else "off"
        )

    def set_bag_mask_settings(self, rr: float, vt: float, p_max: float | None = None):
        """Set bag rate and volume, optionally setting the APL valve."""
        validated_rr = bounded_number("bag rate", rr, *BAG_RR_RANGE, unit="breaths/min")
        validated_vt = bounded_number("bag tidal volume", vt, *BAG_VT_RANGE_L, unit="L")
        if p_max is not None:
            self.set_apl_valve(p_max)
        self.bag_mask_rr, self.bag_mask_vt = validated_rr, validated_vt

    def set_apl_valve(self, cmh2o: float):
        """Set the APL valve for bagging and spontaneous breathing. Zero is fully open."""
        self.apl_cmh2o = bounded_number("APL valve", cmh2o, *APL_RANGE_CMH2O, unit="cmH2O")

    def deliver_manual_breath(self):
        """Queue a single manual bag squeeze at the current bag tidal volume and APL setting."""
        if self.state.airway_mode == AirwayType.NONE or self.vent.is_on or self.bag_mask_active:
            return
        self._manual_breath_requested = True
        self.actions.record(self.state.time, ACTION_BAG_MASK, label="breath")

    def trigger_o2_flush(self, duration_s: float = O2_FLUSH_DURATION_S):
        """Deliver a pure oxygen flush bypassing the vaporizer for duration_s seconds."""
        duration = bounded_number("O2 flush duration", duration_s, 0.1, 10.0, unit="s")
        self._o2_flush_remaining_s = max(self._o2_flush_remaining_s, duration)
        self.actions.record(self.state.time, ACTION_O2_FLUSH, amount=duration)

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
        self.state.nibp_cuff_pressure = 0.0
        self._next_nibp_time = self.state.time + self.nibp.interval

    def draw_blood_gas(self) -> BloodGasResult:
        """Draw an arterial sample; its result is ready after the analysis time."""
        state = self.state
        fluids = self.hemo.fluids
        abg_ph = float(self.resp.state.ph_37)
        abg_paco2 = float(self.resp.state.pa_co2_37)
        abg_hco3 = float(state.hco3)
        result = BloodGasResult(
            drawn_at=state.time, ph=abg_ph, pa_co2=abg_paco2, pao2=self.resp.arterial_po2_at_37(),
            hco3=abg_hco3, base_excess=state.base_excess,
            lactate=float(self.lactate.arterial_concentration),
            sodium=float(fluids.concentration("sodium")),
            potassium=float(fluids.arterial_potassium),
            chloride=float(fluids.concentration("chloride")),
            ionized_calcium=float(fluids.ionized_calcium(abg_ph)),
            sao2=state.sao2, hb=float(self.hemo.hb_conc), fio2=state.fio2,
        )
        self.blood_gases.append(result)
        self.actions.record(state.time, ACTION_BLOOD_GAS)
        return result

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
        self.state.time = round(self.state.time + dt, 9)
        output = self.output_buffer
        cutoff = self.state.time - OUTPUT_WINDOW_S
        while len(output) > 1 and output[0].time < cutoff:
            output.popleft()
        if self.recorder and self.recorder.is_recording:
            self.recorder.log(self.state)

    @property
    def waveform_interval(self) -> float:
        """Spacing of output_buffer samples at the configured step, s."""
        dt = self.config.dt
        return dt / monitor_core.waveform_substeps(dt)

    @property
    def spontaneous_circulation(self) -> bool:
        """The heart's own beats perfuse now; arrest is flagged only after 15 s without them."""
        return (
            self._pulseless_s == 0.0
            and self.hemo.state.rhythm_type not in PULSELESS_RHYTHMS
        )

    @property
    def irreversible_arrest(self) -> bool:
        """Arrest after ischemic injury that leaves the heart unable to contract."""
        return self.state.cardiac_arrest and self.hemo.coronary.irreversible

    def get_latest_state(self) -> SimulationState:
        """Return an independent copy of the current state, including alarm flags."""
        return copy.deepcopy(self.state)

    def get_predicted_csht(self, drug: str, max_seconds: int = 7200) -> float:
        """Minutes for a drug's effect site to halve if stopped now, up to max_seconds."""
        key = get_drug_spec(drug).key
        if key == "roc":
            return self.tof_pd.simulate_decay(
                self.pk["roc"], max_seconds=max_seconds,
                mac_sevo=self.pk_sevo.state.mac,
            )
        pump_model = (
            self.pump_pk.get(key)
            if self.config.tci_enabled and key not in self._pump_needs_reset
            else None
        )
        if pump_model is not None and (key in self.tci or pump_model.state.ce > 1e-12):
            model = pump_model
        else:
            model = self.pk[key]
        target_fraction = 0.5
        if model.basal_input_rate > 0.0:
            if model.circulation <= 0.0 or model.elimination_clearance <= 0.0:
                basal_ref = model.basal_input_rate * 60.0 / max(1e-9, model.cl1 * model.cl1_scale)
                if model.state.ce <= basal_ref + 1e-6:
                    return 0.0
            else:
                basal_ce = model.effective_basal_input_rate * 60.0 / model.elimination_clearance
                if model.state.ce <= basal_ce + 1e-9:
                    return 0.0
                target_fraction = (basal_ce + 0.5 * (model.state.ce - basal_ce)) / model.state.ce
        return model.simulate_decay(target_fraction=target_fraction, max_seconds=max_seconds)

    def set_vent_power(self, on: bool):
        """Start or stop the ventilator; its settings persist while it is off."""
        on = bool(on)
        if self.vent.is_on != on:
            self.actions.record(self.state.time, ACTION_VENTILATION, label="on" if on else "off")
        self.vent.is_on = on
        if on:
            self._manual_breath_requested = False

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
        """Split the O2 plus air flow to reach the target FiO2, reducing N2O only when needed.

        Solves FiO2 = (O2 + 0.21 x air) / (O2 + air + N2O) at constant total flow.
        """
        total_non_n2o = self.circuit.fgf_o2 + self.circuit.fgf_air
        n2o_flow = max(0.0, self.circuit.fgf_n2o)
        total_flow = total_non_n2o + n2o_flow
        if total_flow <= 0:
            return
        target = clamp(fio2, 0.21, 1.0)
        if n2o_flow <= 0.0:
            o2_flow = clamp((target - 0.21) / 0.79 * total_flow, 0.0, total_flow)
            air_flow = total_flow - o2_flow
            self.set_fgf(o2_flow, air_flow, 0.0)
            return
        o2_needed = (target * total_flow - 0.21 * total_non_n2o) / 0.79
        o2_min_n2o = n2o_flow / 3.0
        if max(o2_needed, o2_min_n2o) <= total_non_n2o:
            o2_flow = clamp(max(o2_needed, o2_min_n2o), 0.0, total_non_n2o)
            air_flow = total_non_n2o - o2_flow
            self.set_fgf(o2_flow, air_flow, min(n2o_flow, 3.0 * o2_flow))
        else:
            o2_flow = clamp(max(target, 0.25) * total_flow, 0.0, total_flow)
            n2o_new = max(0.0, total_flow - o2_flow)
            self.set_fgf(o2_flow, 0.0, min(n2o_new, 3.0 * o2_flow))
