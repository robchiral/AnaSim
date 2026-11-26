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
from anasim.extracorporeal import (
    BloodGasManagement,
    CannulationType,
    ExtracorporealSystem,
    MembraneType,
)
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
    HeparinPK,
    NorepinephrinePK,
    ProtaminePK,
    janmahasatian_ffm,
)
from anasim.patient.variability import sample_variability
from anasim.patient.volatile_pk import VolatilePK
from anasim.physiology.airway_crisis import (
    ESOPHAGEAL_DEFAULT_CO2_MMHG,
    INTUBATION_TECHNIQUES,
    AirwayCrisisModel,
)
from anasim.physiology.arrhythmia import ArrhythmiaModel, antiarrhythmic_effects
from anasim.physiology.cerebral import CerebralModel
from anasim.physiology.coagulation import ACTResult, CoagulationModel
from anasim.physiology.disturbances import Disturbances
from anasim.physiology.embolism import PulmonaryEmbolismModel
from anasim.physiology.events import SeverityRamp
from anasim.physiology.glucose import GlucoseModel, GlucoseResult
from anasim.physiology.hemodynamics import MAX_HEMATOCRIT, HemodynamicModel
from anasim.physiology.lactate import LactateModel
from anasim.physiology.lung import LungAeration
from anasim.physiology.malignant_hyperthermia import (
    SUCCINYLCHOLINE_TRIGGER_WINDOW_S,
    MalignantHyperthermia,
    triggering_exposure,
)
from anasim.physiology.neuraxial import NeuraxialLevelCheck, NeuraxialModel
from anasim.physiology.pneumothorax import PneumothoraxModel
from anasim.physiology.resp_mech import RespiratoryMechanics
from anasim.physiology.respiration import RespiratoryModel
from anasim.physiology.resuscitation import (
    SHOCK_ENERGY_J,
    Defibrillator,
    ShockResult,
    shock_outcome,
)
from anasim.physiology.thermal import ThermalModel
from anasim.physiology.toxicity import LocalAnestheticToxicity

from . import monitors as monitor_core
from . import projection as projection_core
from . import runtime as runtime_core
from .action_log import (
    ACTION_ACT,
    ACTION_AIRWAY,
    ACTION_AIRWAY_TOOL,
    ACTION_BAG_MASK,
    ACTION_BLOOD_GAS,
    ACTION_CHEST_DECOMPRESS,
    ACTION_COMPRESSIONS,
    ACTION_COOLING,
    ACTION_DRUG_BOLUS,
    ACTION_EMBOLISM_TREAT,
    ACTION_EVENT_START,
    ACTION_EVENT_STOP,
    ACTION_EXTRACORPOREAL,
    ACTION_FGF,
    ACTION_FLUID,
    ACTION_FONA,
    ACTION_GLUCOSE,
    ACTION_INTUBATION_ATTEMPT,
    ACTION_LIPID_INFUSION,
    ACTION_NEURAXIAL,
    ACTION_O2_FLUSH,
    ACTION_OXYGEN_SUPPLY,
    ACTION_POSITION,
    ACTION_SHOCK,
    ACTION_VAPORIZER,
    ACTION_VENTILATION,
    ActionLog,
)
from .clinical_actions import (
    ActionFeedback,
    check_fona,
    check_intubation,
    check_oral_airway,
    check_sga_placement,
    check_shock,
)
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
        self._had_mask_ventilation = False

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
        self.airway_crisis = AirwayCrisisModel()
        self.pneumothorax = PneumothoraxModel(self.patient.functional_residual_capacity())
        ref_ffm = janmahasatian_ffm(70.0, 170.0, True)
        patient_ffm = janmahasatian_ffm(
            self.patient.weight, self.patient.height, self.patient.sex == "male",
        )
        self.embolism = PulmonaryEmbolismModel(70.0 * patient_ffm / ref_ffm)
        self.neuraxial = NeuraxialModel(
            self.patient.age,
            pregnant=self.patient.gestational_age_weeks > 0.0,
        )

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
        # Sevoflurane MAC40 from Nickalls and Mapleson 2003 (age formula: Mapleson 1996).
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
        self.extracorporeal = ExtracorporealSystem()
        self.resp = RespiratoryModel(self.patient)
        self.resp.attach_fluid_balance(self.hemo.fluids, self.hemo.config.donor_mchc_g_dl)
        self.pk_sevo.frc = self.resp.gas_volume_l
        self.pk_n2o.frc = self.resp.gas_volume_l
        self.lactate = LactateModel(
            self.patient.weight,
            self.patient.hepatic_function,
            lean_mass_kg=self.hemo.fluids.lean_mass_kg,
            volume_l=self.hemo.fluids.lactate_volume_l,
        )
        self.glucose = GlucoseModel(
            lean_mass_kg=self.hemo.fluids.lean_mass_kg,
            ecf_l=self.hemo.fluids.ecf_l,
            fluids=self.hemo.fluids,
            baseline_glucose_mg_dl=(
                90.0 if self.patient.baseline_glucose is None else self.patient.baseline_glucose
            ),
            type2_diabetes=self.patient.type2_diabetes,
            age=self.patient.age,
            bmi=self.patient.bmi,
            renal_function=self.patient.renal_function,
            renal_clearance_factor=self.patient.renal_clearance_factor,
            hepatic_function=self.patient.hepatic_function,
        )
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
        self.coagulation = CoagulationModel()

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
        self.acts: list[ACTResult] = []
        self.glucoses: list[GlucoseResult] = []
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

    def give_d5w(self, volume_ml: float = 250.0):
        """Queue 5% dextrose in water (50 g/L) at the crystalloid infusion rate."""
        self._queue_infusion(volume_ml, self.fluid_infusion_rate_ml_min, hematocrit=0.0, label="d5w")

    def give_d10w(self, volume_ml: float = 100.0):
        """Queue 10% dextrose in water (100 g/L) at 50 mL/min."""
        self._queue_infusion(volume_ml, 50.0, hematocrit=0.0, label="d10w")

    def give_d50w(self, volume_ml: float = 50.0):
        """Queue 50% dextrose in water (500 g/L; 50 mL = 25 g) as a 25 mL/min IV push."""
        self._queue_infusion(volume_ml, 25.0, hematocrit=0.0, label="d50w")

    def give_calcium_chloride(self, mg: float):
        """Give calcium chloride dihydrate; 1 g contains 6.8 mmol calcium."""
        dose = bounded_number("calcium chloride dose", mg, 0.0, 10000.0, unit="mg")
        if dose == 0.0:
            raise ValueError("calcium chloride dose must be greater than zero")
        circulation = self.hemo.circulation_fraction()
        self.hemo.fluids.give_calcium_chloride(dose * 0.0068, circulation=circulation, bolus_push_s=90.0)
        self.state.ionized_calcium = float(self.hemo.fluids.ionized_calcium(self.resp.state.ph_37))
        self.actions.record(self.state.time, ACTION_DRUG_BOLUS, label="calcium_chloride", amount=dose)

    def give_sodium_bicarbonate(self, meq: float = 50.0):
        """Queue 8.4% sodium bicarbonate (1 mEq/mL; 50 mEq = 50 mL) as a 100 mL/min IV push."""
        dose = bounded_number("sodium bicarbonate dose", meq, 0.0, 500.0, unit="mEq")
        if dose <= 0.0:
            raise ValueError("sodium bicarbonate dose must be greater than zero")
        self._queue_infusion(dose, 100.0, hematocrit=0.0, label="bicarbonate", record_action=False)
        self.actions.record(self.state.time, ACTION_DRUG_BOLUS, label="sodium_bicarbonate", amount=dose)

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
        *,
        record_action: bool = True,
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
        if record_action:
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
        if spec.key in ("heparin", "protamine"):
            circ = self.hemo.circulation_fraction()
            if spec.key == "protamine":
                self.coagulation.record_protamine_bolus(dose, model.v1, circulation=circ)
            hep_pk = self.pk["heparin"]
            prot_pk = self.pk["protamine"]
            assert isinstance(hep_pk, HeparinPK) and isinstance(prot_pk, ProtaminePK)
            self.coagulation.step(
                0.0,
                hep_pk,
                prot_pk,
                temp_c=self.state.temp_c,
                hematocrit=self.hemo.get_hematocrit(),
                baseline_hct=self.hemo.baseline_hct,
                circulation=circ,
            )
            projection_core.sync_pk_state(self)
        elif spec.key == "insulin":
            self.glucose.sync_exogenous_insulin(
                model.state.c1,
                circulation=self.hemo.circulation_fraction(),
            )
            projection_core.sync_pk_state(self)
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
        if name == "protamine_reaction":
            return self.coagulation.reaction_active
        if name == "difficult_airway":
            return self.airway_crisis.difficult_profile != "none"
        if name == "esophageal_intubation":
            return self.airway_crisis.esophageal_intubation
        if name == "pneumothorax":
            return self.pneumothorax.active
        if name == "pulmonary_embolism":
            return self.embolism.pe_active
        if name == "venous_air_embolism":
            return self.embolism.vae_active
        if name == "high_spinal":
            return self.neuraxial.last_mode == "high_spinal" and self.neuraxial.active
        raise ValueError(f"Unknown clinical event: {name!r}")

    def _log_event(self, name: str, active: bool, amount: float = 0.0):
        """Log a clinical event transition before it is applied."""
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
        self.stop_protamine_reaction()
        self.stop_pneumothorax()
        self.pneumothorax.reset()
        self.state.pneumothorax_pressure = 0.0
        self.state.pneumothorax_decompressed = False
        self.stop_pulmonary_embolism()
        self.stop_venous_air_embolism()
        self.embolism.clear_vae()
        self.state.vae_air_ml = 0.0
        self.set_difficult_airway("none")
        self.set_esophageal_intubation(False)
        self.stop_neuraxial()
        self.neuraxial.place_epidural(False)
        self._sync_neuraxial_state()
        self.stop_disturbance()
        self.intubation_start_time = None

    def start_pneumothorax(self, severity: float = 1.0, *, initial_volume_l: float = 0.25):
        """Start an active unilateral one-way pleural air leak (tension pneumothorax)."""
        sev = bounded_number("pneumothorax severity", severity, 0.2, 3.0)
        self._log_event("pneumothorax", True, amount=sev)
        self.pneumothorax.start(severity=sev, initial_volume_l=initial_volume_l)
        self.state.pneumothorax_pressure = float(self.pneumothorax.pleural_pressure_mmhg)
        self.state.pneumothorax_decompressed = False

    def stop_pneumothorax(self):
        """Seal an active pleural leak without decompressing trapped gas."""
        self._log_event("pneumothorax", False)
        self.pneumothorax.stop()

    def decompress_pneumothorax(self):
        """Perform needle thoracostomy / chest tube decompression."""
        self._log_event("pneumothorax", False)
        self.pneumothorax.decompress()
        self.state.pneumothorax_decompressed = True
        self.actions.record(self.state.time, ACTION_CHEST_DECOMPRESS, label="decompressed")

    def start_pulmonary_embolism(self, obstruction: float = 0.60):
        """Start acute massive pulmonary thromboembolism (obstruction fraction 0.15-0.85)."""
        obs = bounded_number("PE obstruction", obstruction, 0.15, 0.85)
        self._log_event("pulmonary_embolism", True, amount=obs)
        self.embolism.start_pe(obstruction=obs)

    def stop_pulmonary_embolism(self):
        """Clear an active pulmonary thromboembolism."""
        self._log_event("pulmonary_embolism", False)
        self.embolism.stop_pe()
        self.state.pe_obstruction = 0.0

    def give_thrombolysis(self):
        """Administer systemic thrombolysis (e.g. alteplase) for massive PE."""
        self.embolism.give_thrombolysis()
        self.actions.record(self.state.time, ACTION_EMBOLISM_TREAT, label="thrombolysis")

    def start_venous_air_embolism(
        self, rate_ml_min: float = 55.0, *, initial_bolus_ml: float = 45.0,
    ):
        """Start surgical venous air entrainment (VAE)."""
        rate = bounded_number("VAE entrainment rate", rate_ml_min, 0.0, 300.0, unit="mL/min")
        self._log_event("venous_air_embolism", True, amount=rate)
        self.embolism.start_vae(rate_ml_min=rate, initial_bolus_ml=initial_bolus_ml)
        self.state.vae_air_ml = float(self.embolism.vae_air_ml)

    def stop_venous_air_embolism(self):
        """Stop surgical venous air entrainment (flood field / lower surgical site)."""
        self._log_event("venous_air_embolism", False)
        self.embolism.stop_vae()

    def aspirate_vae_air(self) -> float:
        """Aspirate intravascular air from a right-atrial CVC; returns mL removed."""
        removed = self.embolism.aspirate_vae()
        self.state.vae_air_ml = float(self.embolism.vae_air_ml)
        self.actions.record(self.state.time, ACTION_EMBOLISM_TREAT, label="vae_aspirate", amount=removed)
        return removed

    def _sync_neuraxial_state(self) -> None:
        nx = self.neuraxial
        self.state.neuraxial_mode = str(nx.last_mode)
        self.state.neuraxial_sensory_level = float(nx.sensory_level)
        self.state.neuraxial_sensory_label = str(nx.sensory_label)
        self.state.neuraxial_motor_label = str(nx.motor_label)
        self.state.neuraxial_bromage = int(nx.bromage_score)
        self.state.epidural_catheter = bool(nx.epidural_catheter)
        self.state.epidural_infusion_ml_hr = float(nx.epidural_infusion_ml_hr)

    def give_spinal(
        self,
        drug: str = "bupivacaine",
        dose_mg: float = 12.5,
        baricity: str = "hyperbaric",
        *,
        high_spinal: bool = False,
    ) -> None:
        """Administer a subarachnoid (spinal) local anesthetic bolus."""
        dose = bounded_number("spinal dose", dose_mg, 1.0, 150.0, unit="mg")
        self.neuraxial.give_spinal(drug=drug, dose_mg=dose, baricity=baricity, high_spinal=high_spinal)
        if self.neuraxial.last_mode == "high_spinal":
            self._log_event("high_spinal", True)
        self._sync_neuraxial_state()
        self.actions.record(
            self.state.time,
            ACTION_NEURAXIAL,
            label=f"spinal:{drug.strip().lower()}:{baricity.strip().lower()}",
            amount=dose,
        )

    def place_epidural(self, placed: bool = True) -> None:
        """Place or remove an epidural catheter."""
        placed = bool(placed)
        if self.neuraxial.epidural_catheter != placed:
            self.neuraxial.place_epidural(placed)
            self._sync_neuraxial_state()
            self.actions.record(
                self.state.time,
                ACTION_NEURAXIAL,
                label="epidural_catheter_on" if placed else "epidural_catheter_off",
            )

    def give_epidural_bolus(
        self,
        drug: str = "bupivacaine",
        volume_ml: float = 10.0,
        concentration_pct: float = 0.5,
        *,
        with_epinephrine: bool = False,
        accidental_intrathecal: bool = False,
    ) -> None:
        """Administer an epidural bolus (optionally with 1:200,000 epinephrine or intrathecal)."""
        vol = bounded_number("epidural volume", volume_ml, 1.0, 30.0, unit="mL")
        conc = bounded_number("epidural concentration", concentration_pct, 0.05, 2.0, unit="%")
        if accidental_intrathecal:
            self._log_event("high_spinal", True, amount=vol)
        self.neuraxial.give_epidural_bolus(
            drug=drug,
            volume_ml=vol,
            concentration_pct=conc,
            with_epinephrine=with_epinephrine,
            accidental_intrathecal=accidental_intrathecal,
        )
        self._sync_neuraxial_state()
        if not accidental_intrathecal:
            suffix = "_epi" if with_epinephrine else ""
            self.actions.record(
                self.state.time,
                ACTION_NEURAXIAL,
                label=f"epidural_bolus:{drug.strip().lower()}{suffix}",
                amount=vol,
            )

    def set_epidural_infusion(
        self,
        ml_hr: float,
        *,
        drug: str = "bupivacaine",
        concentration_pct: float = 0.125,
    ) -> None:
        """Set continuous epidural infusion rate (mL/hr)."""
        rate = bounded_number("epidural infusion rate", ml_hr, 0.0, 25.0, unit="mL/hr")
        conc = bounded_number("epidural concentration", concentration_pct, 0.05, 0.5, unit="%")
        self.neuraxial.set_epidural_infusion(rate, drug=drug, concentration_pct=conc)
        self._sync_neuraxial_state()
        self.actions.record(
            self.state.time,
            ACTION_NEURAXIAL,
            label=f"epidural_rate:{drug.strip().lower()}",
            amount=rate,
        )

    def check_neuraxial_level(self) -> NeuraxialLevelCheck:
        """Assess current sensory dermatome height and modified Bromage motor score."""
        check = self.neuraxial.check_level(self.state.time)
        self._sync_neuraxial_state()
        self.actions.record(
            self.state.time,
            ACTION_NEURAXIAL,
            label=f"check_level:{check.sensory_label}",
            amount=check.sensory_level,
        )
        return check

    def trigger_high_spinal(self) -> None:
        """Trigger an accidental intrathecal high/total spinal crisis."""
        self.give_epidural_bolus(
            drug="bupivacaine",
            volume_ml=15.0,
            concentration_pct=0.5,
            accidental_intrathecal=True,
        )

    def stop_neuraxial(self) -> None:
        """Clear any active neuraxial block and epidural infusion."""
        if self.event_active("high_spinal"):
            self._log_event("high_spinal", False)
        self.neuraxial.clear()
        self._sync_neuraxial_state()

    def _set_protamine_reaction(self, active: bool) -> None:
        self._log_event("protamine_reaction", active)
        self.coagulation.reaction_armed = False
        self.coagulation.reaction_active = active
        self.coagulation.state.protamine_reaction_active = active
        self.state.protamine_reaction = active

    def start_protamine_reaction(self):
        """Start a catastrophic Type III protamine pulmonary vasoconstriction reaction (Lowenstein 1983)."""
        self._set_protamine_reaction(True)

    def stop_protamine_reaction(self):
        """Stop an active Type III protamine pulmonary vasoconstriction reaction."""
        self._set_protamine_reaction(False)

    def arm_protamine_reaction(self, armed: bool = True):
        """Arm a Type III protamine reaction to fire automatically on the next protamine dose."""
        self.coagulation.reaction_armed = bool(armed)

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

    def attempt_intubation(self, technique: str = "dl") -> ActionFeedback | None:
        """Attempt learner intubation with the selected technique, including stimulation on failure."""
        if not isinstance(technique, str):
            choices = ", ".join(INTUBATION_TECHNIQUES)
            raise ValueError(f"Unsupported intubation technique {technique!r}; choose one of: {choices}")
        tech = technique.strip().lower()
        if tech not in INTUBATION_TECHNIQUES:
            choices = ", ".join(INTUBATION_TECHNIQUES)
            raise ValueError(f"Unsupported intubation technique {technique!r}; choose one of: {choices}")
        was_esophageal = self.airway_crisis.esophageal_intubation
        if self.state.airway_mode == AirwayType.ETT and not was_esophageal:
            return None
        feedback = check_intubation(self)
        failed = feedback is not None and feedback.blocked
        self.intubation_start_time = self.state.time
        if failed:
            self.actions.record(
                self.state.time,
                ACTION_INTUBATION_ATTEMPT,
                label="failed" if tech == "dl" else f"failed:{tech}",
            )
            return feedback
        result = self.airway_crisis.evaluate_intubation(
            tech,
            prior_mask_ventilation=self._had_mask_ventilation,
            vent_breath_count=self.vent.breath_count,
        )
        self.state.laryngoscopy_attempts = self.airway_crisis.laryngoscopy_attempts
        self.state.laryngoscopy_view = self.airway_crisis.last_view
        self.state.esophageal_intubation = self.airway_crisis.esophageal_intubation
        if not result.placed:
            self.actions.record(
                self.state.time,
                ACTION_INTUBATION_ATTEMPT,
                label="failed" if tech == "dl" else f"failed:{tech}",
            )
            if self.airway_crisis.laryngoscopy_attempts > 3:
                return ActionFeedback(
                    "repeated_laryngoscopy",
                    f"{result.message} More than 3 attempts increases edema and hypoxemia risk.",
                    blocked=True,
                )
            return ActionFeedback(
                result.code or "failed_laryngoscopy",
                result.message,
                blocked=True,
            )
        self.actions.record(
            self.state.time,
            ACTION_INTUBATION_ATTEMPT,
            label="placed" if tech == "dl" else f"placed:{tech}",
        )
        if result.esophageal:
            self._log_event("esophageal_intubation", True)
            self._reset_esophageal_monitors()
        elif was_esophageal:
            self._log_event("esophageal_intubation", False)
            self._reset_esophageal_monitors()
        self.set_airway_mode("ETT")
        if self.bag_mask_active:
            self.set_bag_mask_ventilation(False)
        if feedback is not None:
            return feedback
        if result.message:
            return ActionFeedback("intubation_note", result.message, blocked=False)
        return None

    def _reset_esophageal_monitors(self) -> None:
        """Clear latched capnography and ventilator breath phase on esophageal transitions."""
        self.capno.reset()
        self.etco2_readout.reset()
        self.vent._switch(None, self.resp_mech)
        self.state.display_etco2 = 0.0
        self.state.etco2_signal_valid = False
        self.state.vt = 0.0
        self.state.et_o2 = self.state.et_sevo = self.state.et_n2o = self.state.et_mac = 0.0

    def attempt_sga(self) -> ActionFeedback | None:
        """Attempt learner placement of a 2nd-generation supraglottic airway."""
        if self.state.airway_mode == AirwayType.SGA:
            return None
        feedback = check_sga_placement(self)
        if feedback is not None and feedback.blocked:
            self.actions.record(self.state.time, ACTION_AIRWAY_TOOL, label="sga_failed")
            return feedback
        ok, msg = self.airway_crisis.evaluate_sga()
        if not ok:
            self.actions.record(self.state.time, ACTION_AIRWAY_TOOL, label="sga_failed")
            return ActionFeedback("sga_failed", msg, blocked=True)
        self.set_airway_mode("SGA")
        if self.bag_mask_active:
            self.set_bag_mask_ventilation(False)
        return feedback

    def perform_fona(self) -> ActionFeedback | None:
        """Perform emergency scalpel-bougie-tube cricothyrotomy (eFONA)."""
        if self.state.airway_mode == AirwayType.FONA:
            return None
        feedback = check_fona(self)
        self.airway_crisis.fona_performed = True
        self.airway_crisis.clear_esophageal()
        self.state.esophageal_intubation = False
        self.actions.record(self.state.time, ACTION_FONA, label="performed")
        self.set_airway_mode("FONA")
        if self.bag_mask_active:
            self.set_bag_mask_ventilation(False)
        return feedback

    def set_oral_airway(self, placed: bool) -> ActionFeedback | None:
        """Insert or remove an oropharyngeal airway (OPA) for mask ventilation."""
        placed = bool(placed)
        if placed and not self.airway_crisis.oral_airway:
            feedback = check_oral_airway(self)
            if feedback is not None and feedback.blocked:
                return feedback
        if self.airway_crisis.oral_airway != placed:
            self.airway_crisis.oral_airway = placed
            self.state.oral_airway = placed
            self.actions.record(
                self.state.time, ACTION_AIRWAY_TOOL, label="opa_on" if placed else "opa_off",
            )
        return None

    def set_two_hand_mask(self, active: bool) -> None:
        """Enable or disable two-handed jaw-thrust mask ventilation (Joffe 2010)."""
        active = bool(active)
        if self.airway_crisis.two_hand_mask != active:
            self.airway_crisis.two_hand_mask = active
            self.state.two_hand_mask = active
            self.actions.record(
                self.state.time, ACTION_AIRWAY_TOOL, label="two_hand_on" if active else "two_hand_off",
            )

    def call_airway_help(self) -> None:
        """Call for senior help and the difficult airway cart (ASA 2022)."""
        if not self.airway_crisis.help_called:
            self.airway_crisis.help_called = True
            self.state.airway_help_called = True
            self.actions.record(self.state.time, ACTION_AIRWAY_TOOL, label="call_help")

    def set_difficult_airway(self, profile: str) -> None:
        """Configure difficult airway anatomy ('none', 'anterior_dl', or 'severe_cico')."""
        was_active = self.airway_crisis.difficult_profile != "none"
        self.airway_crisis.set_difficult_profile(profile)
        will_be_active = self.airway_crisis.difficult_profile != "none"
        if was_active != will_be_active:
            self._log_event("difficult_airway", will_be_active)
        self.state.difficult_airway = self.airway_crisis.difficult_profile
        self.state.laryngoscopy_attempts = self.airway_crisis.laryngoscopy_attempts
        self.state.laryngoscopy_view = self.airway_crisis.last_view

    def arm_esophageal_intubation(self, armed: bool = True) -> None:
        """Arm unrecognized esophageal placement for the next blind direct laryngoscopy."""
        self.airway_crisis.esophageal_armed = bool(armed)

    def set_esophageal_intubation(
        self, active: bool, *, gastric_co2_mmhg: float = ESOPHAGEAL_DEFAULT_CO2_MMHG,
    ) -> None:
        """Directly activate or clear esophageal intubation."""
        active = bool(active)
        if active:
            self._log_event("esophageal_intubation", True)
            self.airway_crisis.activate_esophageal(
                gastric_co2_mmhg=gastric_co2_mmhg,
                vent_breath_count=self.vent.breath_count,
            )
            self.state.esophageal_intubation = True
            if self.state.airway_mode != AirwayType.ETT:
                self.set_airway_mode("ETT")
            self._reset_esophageal_monitors()
        else:
            was = self.airway_crisis.esophageal_intubation
            if was:
                self._log_event("esophageal_intubation", False)
            self.airway_crisis.clear_esophageal()
            self.state.esophageal_intubation = False
            if was:
                self._reset_esophageal_monitors()

    def set_airway_mode(self, mode_str: str):
        """Set the connection directly; learner tube placement uses attempt_intubation."""
        previous_mode = self.state.airway_mode
        was_esophageal = previous_mode == AirwayType.ETT and self.airway_crisis.esophageal_intubation
        try:
            self.state.airway_mode = AirwayType(mode_str)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in AirwayType)
            raise ValueError(
                f"Unsupported airway mode {mode_str!r}; choose one of: {choices}"
            ) from exc
        if previous_mode == AirwayType.ETT and self.state.airway_mode != AirwayType.ETT:
            if was_esophageal:
                self._log_event("esophageal_intubation", False)
                self.airway_crisis.clear_esophageal()
                self.state.esophageal_intubation = False
                self._reset_esophageal_monitors()
            else:
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
        if projection_core.is_infraglottic_airway(self):
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

    def set_ards_shunt(self, fraction: float):
        """Set additional pulmonary right-to-left shunt (0-0.90) for ARDS/ECMO scenarios."""
        self.aeration.extra_shunt = bounded_number("ARDS shunt", fraction, 0.0, 0.90)

    def _sync_extracorporeal_state(self) -> None:
        self.hemo.extracorporeal_volume_ml = self.extracorporeal.total_circuit_mixing_volume_ml
        if not self.extracorporeal.is_arterial_support:
            self.hemo._ecmo_arterial_flow_l_min = 0.0
            self.hemo._ecmo_venous_drain_l_min = 0.0
            self.hemo._ecmo_lv_vent_l_min = 0.0
            self.hemo._aortic_cross_clamp = False
            self.hemo._cardioplegia_arrest = False
            self.hemo._cpb_open_drainage = False
        self.hemo.invalidate_state_cache()
        self._sync_cardiac_state()
        self.state.glucose = self.glucose.arterial_glucose_mg_dl
        projection_core.project_extracorporeal(self)

    @staticmethod
    def _normalize_prime_label(label: str) -> str:
        norm = str(label).strip().lower()
        if norm == "albumin":
            return "colloid"
        if norm not in ("crystalloid", "saline", "colloid", "blood"):
            raise ValueError(f"Unknown circuit prime fluid: {label!r}")
        return norm

    def _seed_extracorporeal_gases(self) -> None:
        """Seed pre/post/regional extracorporeal gases from current patient gases before stepping."""
        st = self.extracorporeal.state
        r = self.resp.state
        st.upper_body_sao2 = st.lower_body_sao2 = st.coronary_sao2 = float(r.sao2)
        st.upper_body_pao2 = st.lower_body_pao2 = float(r.p_arterial_o2)
        st.upper_body_paco2_37 = st.lower_body_paco2_37 = float(r.pa_co2_37)
        st.upper_body_ph_37 = st.lower_body_ph_37 = float(r.ph_37)
        st.pre_oxy_po2 = float(self.resp._blood_po2[2])
        st.pre_oxy_so2 = float(self.resp._blood_saturations[2] * 100.0)
        st.post_oxy_po2 = float(r.p_arterial_o2)
        st.post_oxy_so2 = float(r.sao2)

    def _finish_extracorporeal_start(
        self,
        action_label: str,
        blood_flow_l_min: float,
        prime_ml: float,
        prime_label: str,
        prime_hematocrit: float,
    ) -> None:
        if prime_ml > 0.0:
            self.hemo.add_volume(
                prime_ml,
                hematocrit=prime_hematocrit,
                retention_fraction=1.0,
                label=self._normalize_prime_label(prime_label),
                suppress_tde=True,
            )
        if self.extracorporeal._last_pass is None:
            self._seed_extracorporeal_gases()
        self._sync_extracorporeal_state()
        self.actions.record(self.state.time, ACTION_EXTRACORPOREAL, label=action_label, amount=blood_flow_l_min)

    def start_vv_ecmo(
        self,
        *,
        blood_flow_l_min: float = 4.0,
        sweep_gas_l_min: float = 4.0,
        sweep_fio2: float = 1.0,
        cannulation: CannulationType | str = CannulationType.FEM_JUG,
        membrane_type: MembraneType | str = MembraneType.PMP,
        membrane_area_m2: float | None = None,
        prime_volume_ml: float | None = None,
        prime_label: str = "crystalloid",
        prime_hematocrit: float = 0.0,
        heater_cooler_on: bool = True,
        water_target_c: float = 37.0,
    ) -> None:
        """Initiate venovenous ECMO with optional circuit prime hemodilution."""
        self._normalize_prime_label(prime_label)
        if self.extracorporeal.is_active and not self.extracorporeal.is_vv:
            self.stop_extracorporeal(reinfuse=False)
        prime_ml = self.extracorporeal.start_vv_ecmo(
            blood_flow_l_min=blood_flow_l_min,
            sweep_gas_l_min=sweep_gas_l_min,
            sweep_fio2=sweep_fio2,
            cannulation=cannulation,
            membrane_type=membrane_type,
            membrane_area_m2=membrane_area_m2,
            prime_volume_ml=prime_volume_ml,
            heater_cooler_on=heater_cooler_on,
            water_target_c=water_target_c,
        )
        self._finish_extracorporeal_start("start_vv_ecmo", blood_flow_l_min, prime_ml, prime_label, prime_hematocrit)

    def start_va_ecmo(
        self,
        *,
        blood_flow_l_min: float = 4.0,
        sweep_gas_l_min: float = 4.0,
        sweep_fio2: float = 1.0,
        cannulation: CannulationType | str = CannulationType.PERIPHERAL_FEM,
        membrane_type: MembraneType | str = MembraneType.PMP,
        membrane_area_m2: float | None = None,
        prime_volume_ml: float | None = None,
        prime_label: str = "crystalloid",
        prime_hematocrit: float = 0.0,
        heater_cooler_on: bool = True,
        water_target_c: float = 37.0,
        lv_vent_l_min: float = 0.0,
    ) -> None:
        """Initiate venoarterial ECMO (peripheral femoral or central)."""
        self._normalize_prime_label(prime_label)
        if self.extracorporeal.is_active and not self.extracorporeal.is_va:
            self.stop_extracorporeal(reinfuse=False)
        prime_ml = self.extracorporeal.start_va_ecmo(
            blood_flow_l_min=blood_flow_l_min,
            sweep_gas_l_min=sweep_gas_l_min,
            sweep_fio2=sweep_fio2,
            cannulation=cannulation,
            membrane_type=membrane_type,
            membrane_area_m2=membrane_area_m2,
            prime_volume_ml=prime_volume_ml,
            heater_cooler_on=heater_cooler_on,
            water_target_c=water_target_c,
            lv_vent_l_min=lv_vent_l_min,
        )
        self._finish_extracorporeal_start("start_va_ecmo", blood_flow_l_min, prime_ml, prime_label, prime_hematocrit)

    def start_cpb(
        self,
        *,
        blood_flow_l_min: float = 4.5,
        sweep_gas_l_min: float = 3.0,
        sweep_fio2: float = 0.80,
        sweep_fco2: float = 0.0,
        sweep_sevo_percent: float = 0.0,
        membrane_type: MembraneType | str = MembraneType.MICROPOROUS,
        membrane_area_m2: float | None = None,
        prime_volume_ml: float | None = None,
        initial_reservoir_ml: float = 800.0,
        prime_label: str = "crystalloid",
        prime_hematocrit: float = 0.0,
        blood_gas_strategy: BloodGasManagement | str = BloodGasManagement.ALPHA_STAT,
        heater_cooler_on: bool = True,
        water_target_c: float = 37.0,
        venous_clamp_open: float = 1.0,
        bypass_alarm_mode: bool = True,
    ) -> None:
        """Initiate cardiopulmonary bypass with an open venous reservoir."""
        self._normalize_prime_label(prime_label)
        if self.extracorporeal.is_active and not self.extracorporeal.is_cpb:
            self.stop_extracorporeal(reinfuse=False)
        prime_ml = self.extracorporeal.start_cpb(
            blood_flow_l_min=blood_flow_l_min,
            sweep_gas_l_min=sweep_gas_l_min,
            sweep_fio2=sweep_fio2,
            sweep_fco2=sweep_fco2,
            sweep_sevo_percent=sweep_sevo_percent,
            membrane_type=membrane_type,
            membrane_area_m2=membrane_area_m2,
            prime_volume_ml=prime_volume_ml,
            initial_reservoir_ml=initial_reservoir_ml,
            blood_gas_strategy=blood_gas_strategy,
            heater_cooler_on=heater_cooler_on,
            water_target_c=water_target_c,
            venous_clamp_open=venous_clamp_open,
            bypass_alarm_mode=bypass_alarm_mode,
        )
        self._finish_extracorporeal_start("start_cpb", blood_flow_l_min, prime_ml, prime_label, prime_hematocrit)

    def stop_extracorporeal(self, *, reinfuse: bool = False) -> None:
        """Stop active ECMO or CPB support."""
        if not self.extracorporeal.is_active:
            return
        discard_ml = self.hemo.extracorporeal_volume_ml
        self.extracorporeal.stop()
        self.hemo.extracorporeal_volume_ml = 0.0
        if not reinfuse and discard_ml > 0.0:
            self.hemo.add_volume(-discard_ml, suppress_tde=True)
        self._sync_extracorporeal_state()
        self.actions.record(self.state.time, ACTION_EXTRACORPOREAL, label="stop")

    def set_ecmo_flow(self, l_min: float) -> None:
        self.extracorporeal.set_blood_flow(l_min)
        self._sync_extracorporeal_state()
        self.actions.record(self.state.time, ACTION_EXTRACORPOREAL, label="flow", amount=l_min)

    def set_ecmo_sweep(self, l_min: float) -> None:
        self.extracorporeal.set_sweep_gas(l_min)
        self._sync_extracorporeal_state()
        self.actions.record(self.state.time, ACTION_EXTRACORPOREAL, label="sweep", amount=l_min)

    def set_ecmo_fio2(self, fio2: float) -> None:
        self.extracorporeal.set_sweep_fio2(fio2)
        self._sync_extracorporeal_state()
        self.actions.record(self.state.time, ACTION_EXTRACORPOREAL, label="fio2", amount=fio2)

    def set_ecmo_fco2(self, fco2: float) -> None:
        self.extracorporeal.set_sweep_fco2(fco2)
        self._sync_extracorporeal_state()
        self.actions.record(self.state.time, ACTION_EXTRACORPOREAL, label="fco2", amount=fco2)

    def set_ecmo_sevo(self, percent: float) -> None:
        self.extracorporeal.set_sweep_sevo(percent)
        self._sync_extracorporeal_state()
        self.actions.record(self.state.time, ACTION_EXTRACORPOREAL, label="sevo", amount=percent)

    def set_ecmo_membrane_area(self, area_m2: float) -> None:
        self.extracorporeal.set_membrane_area(area_m2)
        self._sync_extracorporeal_state()

    def set_ecmo_heater_cooler(self, on: bool, target_c: float | None = None) -> None:
        self.extracorporeal.set_heater_cooler(on, target_c)
        self._sync_extracorporeal_state()
        self.actions.record(
            self.state.time,
            ACTION_EXTRACORPOREAL,
            label="heater_cooler",
            amount=self.extracorporeal.state.water_target_c if on else 0.0,
        )

    def set_cpb_venous_clamp(self, open_fraction: float) -> None:
        self.extracorporeal.set_venous_clamp(open_fraction)
        self._sync_extracorporeal_state()
        self.actions.record(self.state.time, ACTION_EXTRACORPOREAL, label="venous_clamp", amount=open_fraction)

    def set_cpb_cross_clamp(self, clamped: bool) -> None:
        self.extracorporeal.set_aortic_cross_clamp(clamped)
        self._sync_extracorporeal_state()
        self.actions.record(
            self.state.time, ACTION_EXTRACORPOREAL, label="cross_clamp_on" if clamped else "cross_clamp_off",
        )

    def set_cpb_cardioplegia(self, arrested: bool) -> None:
        self.extracorporeal.set_cardioplegia_arrest(arrested)
        self._sync_extracorporeal_state()
        self.actions.record(
            self.state.time, ACTION_EXTRACORPOREAL, label="cardioplegia_on" if arrested else "cardioplegia_off",
        )

    def set_ecmo_lv_vent(self, l_min: float) -> None:
        self.extracorporeal.set_lv_vent(l_min)
        self._sync_extracorporeal_state()
        self.actions.record(self.state.time, ACTION_EXTRACORPOREAL, label="lv_vent", amount=l_min)

    def set_cpb_blood_gas_strategy(self, strategy: BloodGasManagement | str) -> None:
        self.extracorporeal.set_blood_gas_strategy(strategy)
        self._sync_extracorporeal_state()
        self.actions.record(
            self.state.time, ACTION_EXTRACORPOREAL, label=f"bg_{self.extracorporeal.state.blood_gas_strategy.value}",
        )

    def set_bypass_alarm_mode(self, enabled: bool) -> None:
        self.extracorporeal.set_bypass_alarm_mode(enabled)
        self._sync_extracorporeal_state()

    def add_cpb_reservoir_fluid(
        self, volume_ml: float, kind: str = "crystalloid", hematocrit: float = 0.55,
    ) -> float:
        """Add fluid or packed RBCs directly to the CPB venous reservoir; return volume added (mL)."""
        if not self.extracorporeal.is_cpb:
            raise ValueError("Reservoir fluid can only be added during active CPB")
        vol = bounded_number("reservoir fluid volume", volume_ml, 1.0, 2000.0, unit="mL")
        st = self.extracorporeal.state
        cfg = self.extracorporeal.config
        room = max(0.0, cfg.cpb_reservoir_max_ml - st.reservoir_volume_ml)
        actual_vol = min(vol, room)
        if actual_vol <= 0.0:
            return 0.0
        before_bv = self.hemo.blood_volume
        norm_kind = self._normalize_prime_label(kind)
        hct = bounded_number("blood hematocrit", hematocrit, 0.01, MAX_HEMATOCRIT) if norm_kind == "blood" else 0.0
        # Fluid poured into the hard-shell reservoir stays 100% in the reservoir+circuit pool.
        self.hemo.add_volume(actual_vol, hematocrit=hct, retention_fraction=1.0, label=norm_kind, suppress_tde=True)
        added_bv = max(0.0, self.hemo.blood_volume - before_bv)
        st.reservoir_volume_ml = min(cfg.cpb_reservoir_max_ml, st.reservoir_volume_ml + added_bv)
        st.reservoir_low_alarm = st.reservoir_volume_ml < cfg.cpb_reservoir_low_ml
        self._sync_extracorporeal_state()
        self.actions.record(self.state.time, ACTION_EXTRACORPOREAL, label=f"reservoir_{kind}", amount=added_bv)
        return added_bv

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
        sbp, dbp = projection_core.reference_arterial_pressures(self, hemo_state)
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
            perfusing=runtime_core.is_systemic_perfusing(self),
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
                perfusing=runtime_core.is_systemic_perfusing(self),
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
        active = bool(active)
        if active and self.vent.is_on:
            self.set_vent_power(False)
        if self.bag_mask_active != active:
            self.actions.record(
                self.state.time, ACTION_BAG_MASK, label="on" if active else "off"
            )
        self.bag_mask_active = active

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
        if (
            self.state.airway_mode == AirwayType.NONE
            or self.vent.is_on
            or self.bag_mask_active
            or self.apl_cmh2o <= 0.0
        ):
            return
        self._manual_breath_requested = True
        self.actions.record(self.state.time, ACTION_BAG_MASK, label="breath")

    def refill_vaporizer(self, level_ml: float = 250.0):
        """Refill the sevoflurane vaporizer liquid reservoir (mL)."""
        level = bounded_number("vaporizer level", level_ml, 0.0, self.vaporizer.CAPACITY_ML, unit="mL")
        self.vaporizer.refill(level)

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

    def draw_blood_gas(self, site: str = "arterial") -> BloodGasResult:
        """Draw a blood gas sample; its result is ready after the analysis time.

        Supported sites: 'arterial' (right radial / upper body on peripheral VA ECMO),
        'right_radial', 'femoral' (lower body), 'pre_oxy' (venous inlet), 'post_oxy' (outlet).
        """
        from anasim.physiology.blood_chemistry import (
            plasma_bicarbonate,
            standard_base_excess,
        )
        from anasim.physiology.blood_gas_math import (
            O2_SOLUBILITY,
            acid_base_p50,
            oxygen_content,
            po2_and_sat_from_content,
            temperature_o2_solubility,
        )

        site_norm = str(site).strip().lower()
        if site_norm not in ("arterial", "right_radial", "femoral", "pre_oxy", "post_oxy"):
            raise ValueError(f"Unknown blood gas site: {site!r}")
        state = self.state
        fluids = self.hemo.fluids
        ecmo = self.extracorporeal
        if site_norm in ("pre_oxy", "post_oxy") and not ecmo.is_active:
            raise ValueError("Pre- and post-oxygenator blood gases require active extracorporeal support")
        st = ecmo.state
        abg_ph = float(self.resp.state.ph_37)
        abg_paco2 = float(self.resp.state.pa_co2_37)
        abg_pao2 = self.resp.arterial_po2_at_37()
        abg_hco3 = float(state.hco3)
        abg_be = float(state.base_excess)
        abg_sao2 = float(state.sao2)
        abg_fio2 = float(st.sweep_fio2) if (ecmo.is_cpb and not self.vent.is_on) else float(state.fio2)
        hb_sample = float(self.hemo.hb_conc if fluids._circulation >= 0.99 else self.resp._arterial_hb)
        temp_c = float(state.temp_c)

        def _warm_po2_and_sat_to_37(
            po2_invivo: float, ph_37: float, pco2_37: float, sat_invivo_pct: float,
        ) -> tuple[float, float]:
            if abs(temp_c - 37.0) <= 1e-6:
                return float(po2_invivo), float(sat_invivo_pct)
            p50_invivo = acid_base_p50(ph_37, pco2_37, temp_c)
            sol_invivo = temperature_o2_solubility(temp_c)
            o2_c, _ = oxygen_content(max(0.0, po2_invivo), hb_sample, p50_invivo, sol_invivo)
            p50_37 = acid_base_p50(ph_37, pco2_37, 37.0, at_37=True)
            po2_37, sat_37 = po2_and_sat_from_content(
                o2_c, hb_sample, p50_37, max(1.0, po2_invivo), solubility=O2_SOLUBILITY,
            )
            return float(po2_37), float(sat_37 * 100.0)

        if ecmo.is_active:
            peripheral_va = ecmo.is_arterial_support and st.cannulation is CannulationType.PERIPHERAL_FEM
            last_pass = ecmo._last_pass
            site_values: tuple[float, float, float, float, float] | None = None
            if site_norm in ("arterial", "right_radial") and peripheral_va:
                site_values = (
                    st.upper_body_ph_37, st.upper_body_paco2_37,
                    st.upper_body_pao2, st.upper_body_sao2, abg_fio2,
                )
            elif site_norm == "femoral" and peripheral_va:
                site_values = (
                    st.lower_body_ph_37, st.lower_body_paco2_37,
                    st.lower_body_pao2, st.lower_body_sao2, float(st.sweep_fio2),
                )
            elif site_norm == "pre_oxy":
                site_values = (
                    last_pass.pre_ph_37 if last_pass is not None else self.resp._blood_ph[2],
                    last_pass.pre_pco2_37 if last_pass is not None else self.resp._blood_pressures[2],
                    st.pre_oxy_po2, st.pre_oxy_so2, float(st.sweep_fio2),
                )
            elif site_norm == "post_oxy":
                site_values = (
                    last_pass.ph_out_37 if last_pass is not None else self.resp.state.ph_37,
                    last_pass.pco2_out_37 if last_pass is not None else self.resp.state.pa_co2_37,
                    st.post_oxy_po2, st.post_oxy_so2, float(st.sweep_fio2),
                )
            if site_values is not None:
                ph_37, pco2_37, po2_invivo, sat_invivo, abg_fio2 = site_values
                abg_ph, abg_paco2 = float(ph_37), float(pco2_37)
                abg_pao2, abg_sao2 = _warm_po2_and_sat_to_37(po2_invivo, abg_ph, abg_paco2, sat_invivo)
                abg_hco3 = float(plasma_bicarbonate(abg_ph, abg_paco2))
                abg_be = float(standard_base_excess(abg_ph, abg_hco3))

        result = BloodGasResult(
            drawn_at=state.time, ph=abg_ph, pa_co2=abg_paco2, pao2=abg_pao2,
            hco3=abg_hco3, base_excess=abg_be,
            lactate=float(self.lactate.arterial_concentration),
            sodium=float(fluids.arterial_sodium),
            potassium=float(fluids.arterial_potassium),
            chloride=float(fluids.arterial_chloride),
            ionized_calcium=float(fluids.arterial_ionized_calcium(abg_ph)),
            sao2=abg_sao2,
            hb=hb_sample,
            fio2=abg_fio2,
            site=site_norm,
            glucose=float(self.glucose.arterial_glucose_mg_dl),
        )
        self.blood_gases.append(result)
        self.actions.record(state.time, ACTION_BLOOD_GAS, label=site_norm)
        return result

    def measure_act(self) -> ACTResult:
        """Draw a whole-blood sample for point-of-care activated clotting time (ACT)."""
        result = ACTResult(drawn_at=self.state.time, act_s=float(self.coagulation.state.act_s))
        self.acts.append(result)
        self.actions.record(self.state.time, ACTION_ACT, amount=result.act_s)
        return result

    def measure_glucose(self) -> GlucoseResult:
        """Measure point-of-care blood glucose (mg/dL); result posts after 10 s."""
        result = GlucoseResult(
            drawn_at=self.state.time,
            glucose=float(self.glucose.arterial_glucose_mg_dl),
        )
        self.glucoses.append(result)
        self.actions.record(self.state.time, ACTION_GLUCOSE, amount=result.glucose)
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
        """Minutes for a drug's effect site to halve (or fall halfway to endogenous baseline) if stopped now, up to max_seconds."""
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
        if on and self.bag_mask_active:
            self.set_bag_mask_ventilation(False)
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
