import math
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

from anasim.patient.domain import finite_number

from .enums import RhythmType

CONFIG_CHOICES = {
    "mode": {"awake", "steady_state"},
    "maint_type": {"tiva", "balanced"},
}


@dataclass
class SimulationConfig:
    """Configuration for SimulationEngine; see docs/CLI_USAGE.md."""

    dt: float = 0.01  # s

    concomitant_opioids: bool = True  # Opioid regimen; fixed for the session.
    mode: str = "awake"
    maint_type: str = "tiva"
    tci_enabled: bool = False
    disturbance_profile: str | None = None
    sevoflurane_enabled: bool = True
    # None gives 1 mL/kg/hr.
    maintenance_fluid_ml_hr: float | None = None

    arterial_line_enabled: bool = True
    fluid_warmer_enabled: bool = False
    # Sample PK/PD from published variability; otherwise the typical patient.
    patient_variability: bool = False
    rng_seed: int | None = None

    def __post_init__(self):
        for name in (
            "tci_enabled",
            "sevoflurane_enabled",
            "concomitant_opioids",
            "arterial_line_enabled",
            "fluid_warmer_enabled",
            "patient_variability",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        self.dt = finite_number("dt", self.dt)
        if self.dt <= 0 or self.dt > 2.0:
            raise ValueError("dt must be between 0 (exclusive) and 2.0 seconds")
        if self.maintenance_fluid_ml_hr is not None:
            self.maintenance_fluid_ml_hr = finite_number("maintenance_fluid_ml_hr", self.maintenance_fluid_ml_hr)
            if self.maintenance_fluid_ml_hr < 0.0:
                raise ValueError("maintenance_fluid_ml_hr must not be negative")
        for field_name, supported in CONFIG_CHOICES.items():
            value = getattr(self, field_name)
            if not isinstance(value, str) or value not in supported:
                choices = ", ".join(sorted(supported))
                raise ValueError(f"{field_name}={value!r} is unsupported; choose one of: {choices}")
        if self.maint_type == "balanced" and not self.sevoflurane_enabled:
            raise ValueError("balanced maintenance requires sevoflurane_enabled=True")
        if self.mode == "steady_state" and not self.concomitant_opioids:
            raise ValueError("maintenance initialization includes opioids; use concomitant_opioids=True")


@dataclass(frozen=True, slots=True)
class AirwayStatus:
    """Effects of upper-airway obstruction and bronchospasm on gas exchange, 0-1."""

    patency: float = 1.0  # Share of the upper airway left open
    ventilation_efficiency: float = 1.0
    capno_obstruction: float = 0.0  # Distortion of the capnogram
    vq_mismatch: float = 0.0


class WaveformSample(NamedTuple):
    """Monitor waveforms at one instant; a step yields one sample per 10 ms or less."""

    time: float
    ecg_voltage: float
    pleth_voltage: float
    capno_co2: float
    art_pressure: float
    paw: float
    flow: float
    volume: float
    breath: int  # Count of completed breaths, so loops can split at breath boundaries


class AirwayType(Enum):
    NONE = "None"
    MASK = "Mask"
    SGA = "SGA"
    ETT = "ETT"
    FONA = "FONA"


@dataclass(slots=True)
class SimulationState:
    """Public snapshot of the simulation at one time. See docs/ARCHITECTURE.md."""

    time: float = 0.0
    cardiac_arrest: bool = False
    arrest_reason: str = ""
    rhythm_type: RhythmType = RhythmType.SINUS
    chest_compressions: bool = False
    cpr_rate: float = 110.0  # /min
    cpr_depth_cm: float = 5.5  # cm
    cpr_recoil: float = 1.0  # Full decompression fraction, 0-1
    shock_energy_j: float = 200.0  # Selected biphasic defibrillator energy, J
    pacing_mode: str = "off"
    pacing_route: str = "transcutaneous"
    pacing_rate: float = 70.0  # bpm
    pacing_output_ma: float = 0.0  # mA
    pacing_capture: bool = False
    pacing_threshold_ma: float = 55.0  # mA

    # Effect-site (ce) and plasma (cp) concentrations: propofol, etomidate,
    # ketamine, lidocaine, bupivacaine, ropivacaine, methylene_blue, amiodarone,
    # dantrolene, succinylcholine, and esmolol mcg/mL, vasopressin mU/L, others ng/mL.
    propofol_ce: float = 0.0
    propofol_cp: float = 0.0
    remi_ce: float = 0.0
    remi_cp: float = 0.0
    fentanyl_ce: float = 0.0
    fentanyl_cp: float = 0.0
    morphine_ce: float = 0.0
    morphine_cp: float = 0.0
    hydromorphone_ce: float = 0.0
    hydromorphone_cp: float = 0.0
    naloxone_ce: float = 0.0
    flumazenil_ce: float = 0.0
    atropine_ce: float = 0.0
    ephedrine_ce: float = 0.0
    succinylcholine_ce: float = 0.0
    midazolam_ce: float = 0.0
    etomidate_ce: float = 0.0
    ketamine_ce: float = 0.0
    lidocaine_ce: float = 0.0
    bupivacaine_ce: float = 0.0
    ropivacaine_ce: float = 0.0
    methylene_blue_ce: float = 0.0
    amiodarone_ce: float = 0.0
    dantrolene_ce: float = 0.0  # mcg/mL
    # Combined drug effects: remifentanil-equivalent opioid (ng/mL) and
    # propofol-equivalent hypnotic with etomidate and midazolam (mcg/mL).
    opioid_ce: float = 0.0
    opioid_cp: float = 0.0
    hypnotic_ce: float = 0.0  # BIS effect site
    hypnotic_response_ce: float = 0.0  # Clinical response effect site
    nore_ce: float = 0.0
    epi_ce: float = 0.0
    phenyl_ce: float = 0.0
    vaso_ce: float = 0.0
    dobu_ce: float = 0.0
    mil_ce: float = 0.0
    esmolol_ce: float = 0.0
    labetalol_ce: float = 0.0
    glyco_ce: float = 0.0
    neostigmine_ce: float = 0.0

    # Inhaled agents: concentrations in %; brain MAC is age-, pregnancy-, and temperature-adjusted, et_mac is age-adjusted only.
    fi_sevo: float = 0.0
    et_sevo: float = 0.0
    mac_sevo: float = 0.0
    fi_n2o: float = 0.0
    et_n2o: float = 0.0
    mac_n2o: float = 0.0
    mac: float = 0.0  # Brain MAC, summed across agents; drives drug effects
    et_mac: float = 0.0  # End-tidal MAC shown by the gas monitor

    # Rocuronium (mcg/mL): free drug at the adductor pollicis, and total plasma
    # including drug bound to sugammadex.
    roc_ce: float = 0.0
    roc_cp: float = 0.0

    bis: float = 98.0
    display_bis: float = 98.0
    tof: float = 100.0  # %
    twitch: float = 100.0  # T1 height %, including depolarizing block
    loc: float = 0.0  # Probability of no response to shaking/shouting
    respiratory_fatigue: float = 0.0  # Inspiratory muscle capacity lost, 0-1
    seizure_activity: float = 0.0  # Toxic seizure intensity, 0-1; independent of motor block
    convulsions: bool = False
    muscle_rigidity: bool = False
    opioid_rigidity: float = 0.0  # Opioid-induced chest-wall/glottic rigidity intensity, 0-1
    surface_cooling: bool = False
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
    ph: float = 7.4  # In-vivo arterial pH at patient temperature (pa_co2 and pao2 are also in-vivo).
    hco3: float = 24.5  # Plasma bicarbonate, mmol/L.
    base_excess: float = 0.0  # Standard base excess, mmol/L.
    lactate: float = 0.9  # Plasma lactate, mmol/L.
    sodium: float = 140.0  # mmol/L
    potassium: float = 4.0  # mmol/L
    chloride: float = 104.0  # mmol/L
    ionized_calcium: float = 1.2  # mmol/L, at 37 °C alpha-stat pH
    magnesium: float = 0.8  # Serum total magnesium, mmol/L
    methb_pct: float = 0.0  # Methemoglobin fraction, %
    albumin_g_l: float = 42.0
    phosphate: float = 1.0  # mmol/L
    glucose: float = 90.0  # mg/dL
    insulin_mu_l: float = 10.0  # Total plasma insulin, mU/L (µU/mL)
    alveolar_co2: float = 40.0
    pao2: float = 95.0
    capno_co2: float = 0.0
    et_o2: float = 0.0  # %

    # Airway pressure cmH2O, intrathoracic pressure mmHg, flow L/min, volume L.
    paw: float = 0.0
    pit: float = -2.0
    flow: float = 0.0
    volume: float = 0.0
    # Last completed breath at the airway (cmH2O). Pplat is end-inspiratory
    # pressure without flow, NaN when the last mandatory breath had none; PEEP
    # is end-expiratory airway pressure, so auto-PEEP shows only in Pplat and
    # the flow trace. Dynamic compliance is VTe / (Ppeak - PEEP), mL/cmH2O.
    paw_peak: float = 0.0
    paw_plat: float = 0.0
    paw_mean: float = 0.0
    peep: float = 0.0
    compliance_dyn: float = math.nan

    # spo2 is the pulse oximeter reading; sao2 is arterial saturation.
    spo2: float = 99.0
    display_spo2: float = 99.0
    spo2_signal_valid: bool = True
    sao2: float = 98.0

    nibp_sys: float = 120.0
    nibp_dia: float = 80.0
    nibp_map: float = 93.3
    nibp_timestamp: float | None = None
    nibp_interval_sec: float = 300.0
    nibp_is_cycling: bool = False
    nibp_measurement_failed: bool = False
    nibp_cuff_pressure: float = 0.0

    # MAP, HR, SV, SVR, and CO come from HemodynamicModel (built on Su 2023); sbp and dbp are the ideal pulse
    # landmarks. SV mL, SVR Wood units, CO L/min, pressures mmHg.
    sv: float = 70.0
    svr: float = 16.0
    co: float = 5.0
    intrinsic_co: float = 5.0  # From the heart's own beats; co adds chest compressions or arterial pump flow.
    pulmonary_flow: float = 5.0  # Blood flow crossing the native pulmonary vascular bed, L/min.
    pump_flow: float = 0.0  # Extracorporeal arterial pump flow (VA ECMO / CPB), L/min.
    sbp: float = 120.0
    dbp: float = 80.0
    # Arterial catheter output and beat numerics; art_sbp and art_dbp clear to
    # NaN after the pulse timeout.
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
    # Left atrial pressure (mmHg); lung water above normal (mL/kg PBW).
    lap: float = 9.0
    myocardial_ischemia: float = 0.0  # Reversible supply-demand depression, 0-1
    coronary_supply_demand_ratio: float = 5.0  # Available maximal supply / estimated demand
    st_depression_mv: float = 0.0  # Illustrative ECG signal, not a lead-specific prediction
    lung_water: float = 0.0

    fio2: float = 0.21
    # Medical gas supply pressures (psig) and backup E-cylinder volumes (L).
    o2_pipeline_psig: float = 50.0
    air_pipeline_psig: float = 50.0
    n2o_pipeline_psig: float = 50.0
    o2_cylinder_psig: float = 1900.0
    air_cylinder_psig: float = 1900.0
    n2o_cylinder_psig: float = 745.0
    o2_cylinder_l: float = 660.0
    air_cylinder_l: float = 625.0
    n2o_cylinder_l: float = 1590.0
    o2_supply_source: str = "pipeline"
    vent_drive_gas_l_min: float = 0.0
    temp_c: float = 37.0
    bair_hugger_target: float = 0.0  # °C; 0 is off
    fluid_warmer: bool = False
    oxygen_delivery_ratio: float = 1.0
    shivering: float = 0.0

    # Instantaneous waveform values; history is engine.output_buffer.
    ecg_voltage: float = 0.0
    pleth_voltage: float = 0.0

    alarms: dict[str, dict[str, bool]] = field(default_factory=dict)

    # Airway severities and facemask seal, 0-1.
    airway_mode: AirwayType = AirwayType.NONE
    mask_seal: float = 1.0
    airway_obstruction: float = 0.0
    bronchospasm: float = 0.0
    laryngospasm: float = 0.0
    esophageal_intubation: bool = False
    laryngoscopy_attempts: int = 0
    laryngoscopy_view: str = ""
    difficult_airway: str = "none"
    airway_help_called: bool = False
    oral_airway: bool = False
    two_hand_mask: bool = False

    # Extracorporeal support summary (VV ECMO, VA ECMO, CPB).
    ecmo_mode: str = "none"
    ecmo_flow: float = 0.0  # Circuit blood flow, L/min
    ecmo_sweep: float = 0.0  # Sweep gas flow, L/min
    ecmo_recirculation: float = 0.0  # VV ECMO recirculation fraction, 0-1
    ecmo_o2_transfer: float = 0.0  # STPD mL/min
    ecmo_co2_removal: float = 0.0  # STPD mL/min
    cpb_reservoir_ml: float = 0.0  # Open venous reservoir level, mL
    cpb_cross_clamp: bool = False
    cpb_arterial_line_pressure: float = 0.0  # Extracorporeal arterial return line pressure, mmHg
    cpb_cardioplegia_protocol: str = "buckberg"
    cpb_cardioplegia_route: str = "antegrade"
    cpb_cardioplegia_active: bool = False
    cpb_cardioplegia_protection: float = 0.0
    cpb_cardioplegia_redose_due_s: float = 0.0
    cpb_cs_pressure: float = 0.0  # Retrograde coronary sinus pressure, mmHg
    cpb_myocardial_temp_c: float = 37.0
    upper_sao2: float = 98.0  # Upper-body / right-radial arterial saturation, %
    lower_sao2: float = 98.0  # Lower-body / femoral arterial saturation, %

    # Anticoagulation, reversal, and dilutional hemostasis.
    heparin_u_ml: float = 0.0  # Free circulating unfractionated heparin, U/mL
    protamine_ug_ml: float = 0.0  # Free circulating protamine, mcg/mL
    act_s: float = 120.0  # Activated clotting time, s
    protamine_reaction: bool = False  # Catastrophic Type III pulmonary vasoconstriction active
    fibrinogen_g_l: float = 3.2  # Plasma fibrinogen, g/L
    platelets_10e9_l: float = 250.0  # Platelet count, x10^9/L (K/uL)
    inr: float = 1.0  # International normalized ratio
    rbc_23dpg_fraction: float = 1.0  # Circulating RBC 2,3-DPG fraction relative to fresh blood

    # Thoracic and embolic crisis telemetry.
    pneumothorax_pressure: float = 0.0  # Intrapleural tension above baseline, mmHg
    pneumothorax_decompressed: bool = False
    pneumoperitoneum_iap: float = 0.0  # Intra-abdominal CO2 insufflation pressure, mmHg
    pe_obstruction: float = 0.0  # Pulmonary vascular obstruction from PE, 0-1
    vae_air_ml: float = 0.0  # Entrained intravascular air volume, mL

    # Neuraxial anesthesia telemetry.
    neuraxial_mode: str = "none"
    neuraxial_sensory_level: float = 0.0  # 0=None .. 11=T12 .. 22=T1 .. 27=C4 .. 30=Brainstem
    neuraxial_sensory_label: str = "None"
    neuraxial_motor_label: str = "None"
    neuraxial_bromage: int = 0
    epidural_catheter: bool = False
    epidural_infusion_ml_hr: float = 0.0

    def monitored_blood_pressure(
        self,
        arterial_line_enabled: bool,
    ) -> tuple[float, float, float]:
        """Return systolic, diastolic, and mean pressure from the active monitor."""
        if arterial_line_enabled:
            return self.art_sbp, self.art_dbp, self.art_map
        if self.nibp_measurement_failed or self.nibp_timestamp is None:
            return math.nan, math.nan, 0.0
        return self.nibp_sys, self.nibp_dia, self.nibp_map
