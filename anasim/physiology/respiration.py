import math
from dataclasses import dataclass

from anasim.core.constants import RR_APNEA_THRESHOLD, TEMP_METABOLIC_COEFFICIENT, VT_MIN
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.patient.conditions import SURGICAL_SITE_CLOSURE
from anasim.patient.patient import Patient
from anasim.patient.pd.nmba import ROCURONIUM_T1_EC50

from .blood_chemistry import (
    _PLASMA_CO2_SOLUBILITY,
    MOLAR_GAS_VOLUME_STPD_L_MMOL,
    BloodChemistry,
    Equilibrium,
    extracellular_contents,
    plasma_bicarbonate,
    standard_base_excess,
    step_fraction,
)
from .fluid_balance import (
    STORED_BLOOD_PCO2,
    STORED_BLOOD_PH,
    STORED_BLOOD_SO2,
    FluidBalance,
)
from .gas_transport import GasTransport, TransportPlan
from .hemo_config import HemodynamicConfig
from .plasma_buffers import PlasmaBuffers, buffer_coefficients

_HB_G_DL_PER_MMOL_L = 1.61145  # Heme monomer, 16.1145 g/mmol.
_CO2_L_PER_MMOL = MOLAR_GAS_VOLUME_STPD_L_MMOL  # STPD.
_O2_PER_HB = 0.0134  # STPD L O2 per L blood per g/dL Hb (1.34 mL/g).
_O2_SOLUBILITY = 0.000031  # STPD L O2 per L blood per mmHg.


@dataclass
class _LowVQUnit:
    ratio: float
    share: float = 0.0  # Fraction of nonshunted pulmonary blood flow.
    po2: float = 40.0  # Previous solution, mmHg.


@dataclass
class RespState:
    """Respiratory outputs. Gas tensions in mmHg, EtO2 in %, VT in mL, MV and VA in L/min."""

    rr: float = 12.0
    vt: float = 500.0
    mv: float = 6.0
    va: float = 4.0
    apnea: bool = False
    p_alveolar_co2: float = 40.0
    pa_co2: float = 40.0
    etco2: float = 40.0
    eto2: float = 14.0
    p_alveolar_o2: float = 100.0
    p_arterial_o2: float = 95.0
    sao2: float = 98.0  # %
    ph: float = 7.4  # In-vivo blood gas values at patient temperature.
    hco3: float = 24.5  # Plasma bicarbonate, mmol/L.
    base_excess: float = 0.0  # Standard (extracellular) base excess, mmol/L.
    pv_co2: float = 46.0
    pv_o2: float = 40.0
    venous_ph: float = 7.36
    drive_central: float = 1.0
    muscle_factor: float = 1.0
    effort_rr: float = 12.0
    effort_vt: float = 500.0


class RespiratoryModel:
    """Ventilatory control, anesthetic respiratory depression, and CO2 and O2 exchange."""

    # Under anesthesia the apneic threshold is 4-5 mmHg below resting PaCO2,
    # independent of agent and depth (Hickey 1971).
    APNEIC_GAP = 4.5  # mmHg
    LOW_VQ_RATIO = 0.05  # Age, airway closure, and bronchospasm; fitted to Gunnarsson 1991
    COPD_VQ_RATIO = 0.1  # Low V/Q mode in COPD (Wagner 1977, Figs. 3-4)
    LOW_VQ_AIRWAY = 0.2  # Share of output at full airway V/Q mismatch; room-air PaO2 near 64 mmHg at 40
    RESTING_PCO2_MAX = 100.0  # mmHg; search ceiling for the anesthetized resting PaCO2
    RESTING_PCO2_INTERVAL_S = 1.0  # Drug effects change slowly; recompute the threshold this often
    AWAKE_HYPOCAPNIC_GAIN = 0.2  # Fraction of HCVR; nonlinear depth response, a teaching estimate
    RATE_CO2_GAP = 5.0  # mmHg above the set point before hypercapnia raises frequency
    CO2_CONTROLLER_TAU_S = 150.0  # Olofsen 2010, 2.5 min
    # O2 extraction at which consumption starts to depend on delivery, in septic
    # and nonseptic patients alike (Ronco 1993).
    CRITICAL_EXTRACTION = 0.6
    # Lung tissue uses about 5% of whole-body O2 (Loer 1997). It takes O2 from
    # alveolar gas, so its metabolism continues without pulmonary flow.
    LUNG_TISSUE_FRACTION = 0.05
    CO2_RESPONSE_EXPONENT = 4.37  # Bouillon 2004 nonlinear CO2 response.
    MAX_ALVEOLAR_DRIVE = 12.0  # Bound demand above the nonlinear response at PCO2 70.
    RR_MAX = 60.0  # breaths/min; physiological ceiling on adult tachypnea.
    # Nieuwenhuijs 2000: acute hypoxic drive followed by ventilatory decline.
    HYPOXIC_GAIN = 0.10  # Fraction of alveolar demand per percentage-point desaturation
    HYPOXIC_TAU_S = 5.0  # Fast response timing is a simulator approximation.
    HYPOXIC_DECLINE_DELAY_S = 180.0
    HYPOXIC_DECLINE_TAU_S = 180.0
    HYPOXIC_RECOVERY_TAU_S = 1200.0  # Estimated; decline persists beyond 2 min of normoxia.

    def __init__(self, patient: Patient):
        self.patient = patient
        self.baseline_hb = patient.baseline_hb
        # Baseline CO for perfusion effects; the engine sets the hemodynamic model's value.
        self.baseline_co_l_min = HemodynamicConfig().cardiac_index * patient.bsa * patient.cardiac_output_factor

        # Bouillon 2004 respiratory concentration-response curve, using its
        # own effect site (equilibration half-time 2.6 min).
        self.c50_prop = 1.33
        self.gamma_prop = 1.68
        # Olofsen 2010: a linear opioid shift allows apnea at finite Ce.
        self.c50_remi = 1.6
        self.gamma_remi = 1.0  # Used by the separate shivering response.
        self._ventilatory_drive = 1.0
        self._metabolic_compensation = 1.0
        self._central_drive_fraction = 1.0
        self._hypoxic_drive = 0.0
        self._hypoxic_exposure_s = 0.0
        self._hypoxic_decline = 0.0
        # Olofsen's low-dose propofol background reduces opioid-controller gain
        # to 0.46. Interpolation to other concentrations is a transfer assumption.
        prop_keep_at_one = 1.0 - hill_function(1.0, self.c50_prop, self.gamma_prop)
        self.prop_co2_gain_exponent = math.log(0.46) / math.log(prop_keep_at_one)

        # Sevoflurane barely changes resting VE or the CO2 response at 0.1 MAC
        # (Pandit 1999) and depresses them at 1.1-1.4 MAC with PaCO2 ~55 mmHg at
        # 1.4 MAC (Doi and Ikeda 1987).
        self.c50_sevo_mac = 1.02
        self.gamma_sevo = 2.6

        # Rocuronium at the central (diaphragm and larynx) effect site. These
        # muscles need about 1.8x the adductor pollicis concentration for the
        # same twitch depression (Cantineau 1994; Plaud 1995).
        self.c50_nmba = 1.8 * ROCURONIUM_T1_EC50
        self.gamma_nmba = 3.0

        # Awake chemoreflex calibration (Nieuwenhuijs 2000; Nieuwenhuijs 2001;
        # Pandit 1999). Age curves are calibrated to García-Río 2007 and Poulin
        # 1993; other protocols show equal hypoxic and CO2 declines (Peterson
        # 1981) or rising hypoxic responses in older men (Lhuissier 2012).
        self.hcvr_slope_baseline = 2.2 * (1.0 - 0.22 * clamp01((patient.age - 60.0) / 15.0))
        self.hypoxic_gain = self.HYPOXIC_GAIN * (1.0 - 0.60 * clamp01((patient.age - 62.0) / 13.0))
        self.paco2_setpoint = patient.pregnancy.paco2
        self.rq = 0.8

        # Resting VO2 3.6 mL/kg/min at 70 kg, scaled with weight^0.75.
        self.vco2 = 3.6 * 70.0 * (patient.weight / 70.0) ** 0.75 * self.rq * patient.pregnancy.metabolism  # mL/min
        self.frc = patient.functional_residual_capacity()
        self.shunt_fraction = 0.0
        self.oxygen_supply_fraction = 1.0  # Share of O2 demand met
        self.oxygen_deficit_l = 0.0  # Unmet O2 demand awaiting the lactate update, STPD L.
        # Gas absorbed behind a closed airway, STPD L. Opening the airway
        # passively refills this deficit (Laviola 2020).
        self._apneic_gas_deficit_l = 0.0
        self._pending_mech_absorption_btps_l = 0.0
        self.baseline_blood_volume_ml = patient.estimate_blood_volume()

        # COPD adds dead space and, below, low V/Q perfusion that persist without
        # bronchospasm (Wagner 1977). Severity scaling is a calibration.
        self.vd_deadspace = 2.2 * patient.predicted_body_weight() / 1000.0 * (1.0 + 0.6 * patient.copd_severity)  # L

        self.atm_p = 760.0
        self.vapor_p = 47.0
        self._atm_dry = self.atm_p - self.vapor_p
        # Lung ventilation/volume is BTPS; VO2 and Hb-bound O2 are STPD.
        self._btps_to_stpd = self._atm_dry / 760.0 * 273.15 / 310.15
        self._gas_capacity_per_l = self._btps_to_stpd / self._atm_dry
        # PACO2 = K x VCO2 / VA, with VCO2 in STPD mL/min and VA in BTPS L/min (K about 0.863).
        self.co2_constant = self._atm_dry / self._btps_to_stpd / 1000.0

        # Without a configured VT, resting breaths carry the alveolar
        # ventilation that holds 37 °C alpha-stat PaCO2 at the set point.
        awake_metabolism = max(0.5, TEMP_METABOLIC_COEFFICIENT ** (37.0 - patient.baseline_temp))
        self._awake_metabolism = awake_metabolism
        self._temp_c = patient.baseline_temp
        self._pco2_temp_scale_now = self._temperature_pco2_scale(patient.baseline_temp)
        self.rr_0 = patient.baseline_rr
        if patient.baseline_vt is not None:
            self.vt_0 = patient.baseline_vt
        elif self.rr_0 > 0.0:
            resting_va = self.co2_constant * self.vco2 * awake_metabolism / self.paco2_setpoint
            self.vt_0 = (self.vd_deadspace + resting_va / self.rr_0) * 1000.0
        else:
            self.vt_0 = 0.0
        in_vivo_paco2 = self.paco2_setpoint * self._pco2_temp_scale_now
        self._last_published_pa_co2 = in_vivo_paco2
        self.state = RespState(
            rr=self.rr_0, vt=self.vt_0, mv=self.rr_0 * self.vt_0 / 1000.0,
            effort_rr=self.rr_0, effort_vt=self.vt_0,
            p_alveolar_co2=self.paco2_setpoint, pa_co2=in_vivo_paco2,
        )
        # A residual gradient plus perfusion of poorly ventilated units. Awake,
        # their share of output rises with age (Gunnarsson 1991), giving Crapo
        # 1999's sea-level A-a gradient on room air. Unconscious, airways also
        # close at end-expiration after age 30 unless PEEP holds them open (about
        # 6 cmH2O at BMI 25 and 8 at BMI 30; Hedenstierna 2015). Closure is fitted
        # to the A-a gradient's rise with age at FiO2 0.4 without PEEP (Gunnarsson
        # 1991) and partly persists after abdominal surgery. COPD adds low V/Q
        # units with little shunt (Wagner 1977), 13% of output at severity 1 for
        # a room-air PaO2 near 70 mmHg at age 65.
        self.aa_grad_base = 5.0
        self.low_vq_awake = 0.001 * max(0.0, patient.age - 34.0)
        self.low_vq_copd = 0.14 * patient.copd_severity
        self.low_vq_closure = 0.0025 * max(0.0, patient.age - 30.0)
        self.postoperative_closure = SURGICAL_SITE_CLOSURE[patient.surgical_site]
        self._anesthetized = 0.0  # Deepest unconsciousness so far; surgery follows it
        self.airway_opening_pressure = 6.0 + 0.4 * max(0.0, patient.bmi - 25.0)
        self._low_vq_units = (_LowVQUnit(self.LOW_VQ_RATIO), _LowVQUnit(self.COPD_VQ_RATIO))
        self._p50 = self._temperature_p50(patient.baseline_temp)
        self._temperature_p50_now = self._p50
        self._o2_solubility_now = self._temperature_o2_solubility(patient.baseline_temp)
        assert patient.baseline_hct is not None
        self.baseline_hct = patient.baseline_hct
        self._oxygen_hb = self.baseline_hb
        self.fluids: FluidBalance | None = None
        self._plasma_buffers: PlasmaBuffers | None = None
        self._extracellular_buffers: PlasmaBuffers | None = None
        self._donor_rbc_co2: float = 0.0
        self._urine_carbon_applied: float = 0.0
        self._unbuffered_acid_meq: float = 0.0
        self._set_blood_composition(self.baseline_hb, self.baseline_hct)
        self.gas_transport = GasTransport(
            patient.weight, self.baseline_blood_volume_ml / 1000.0,
        )
        self._blood_pressures = [self.paco2_setpoint, self.paco2_setpoint, self.paco2_setpoint + 6.0]
        self._blood_ph = [patient.pregnancy.ph, patient.pregnancy.ph, patient.pregnancy.ph - 0.04]
        self._blood_po2 = [100.0, 95.0, 40.0]
        self._blood_saturations = [0.98, 0.98, 0.75]
        self._blood_p50 = [self._p50] * 3
        self.alveolar_fn2 = 0.79
        self.equilibrate_oxygen(0.21)
        awake_metabolism = max(0.5, TEMP_METABOLIC_COEFFICIENT ** (37.0 - patient.baseline_temp))
        self.equilibrate_blood_gases(self.baseline_co_l_min, self.vco2 / 1000.0 * awake_metabolism)
        self._resting_pco2: float | None = None
        self._rate_pco2_reference = self.paco2_setpoint
        self._resting_age_s = 0.0
        # Perfusion effect on deadspace fraction (low flow increases VD/VT).
        self.perfusion_deadspace_gain = 0.25

    def _set_blood_composition(self, hb_g_dl: float, hematocrit: float) -> Equilibrium:
        """Rebuild buffering around the patient's resting acid-base reference."""
        self._chemistry_composition = (hb_g_dl, hematocrit)
        self.chemistry = BloodChemistry(
            hb_g_dl / _HB_G_DL_PER_MMOL_L, hematocrit,
            plasma_buffers=self._plasma_buffers, extracellular_buffers=self._extracellular_buffers,
        )
        reference = self.chemistry.evaluate(self.patient.pregnancy.ph, self.paco2_setpoint, 0.98)
        self.buffer_base = self._reference_buffer_base = reference.buffer_base
        return reference

    def attach_fluid_balance(self, fluids: FluidBalance, donor_mchc_g_dl: float) -> None:
        """Seed physical fluid compartments before the engine's maintenance history."""
        self.fluids = fluids
        self._donor_mchc_g_dl = donor_mchc_g_dl
        self._plasma_buffers = fluids.plasma_buffers
        self._extracellular_buffers = fluids.interstitial_buffers
        self._set_blood_composition(self.baseline_hb, self.baseline_hct)
        self.gas_transport = GasTransport(
            self.patient.weight, self.baseline_blood_volume_ml / 1000.0,
            extracellular_volume_l=fluids.interstitial_l,
            extracellular_buffers=self._extracellular_buffers,
        )
        self._fluid_composition = self._composition(
            self.baseline_hb, self.baseline_hct, self._plasma_buffers, self._extracellular_buffers,
        )
        metabolism = max(0.5, TEMP_METABOLIC_COEFFICIENT ** (37.0 - self.patient.baseline_temp))
        self.equilibrate_blood_gases(self.baseline_co_l_min, self.vco2 / 1000.0 * metabolism)

    def update_fluid_balance(self, hb: float, hct: float, blood_volume_ml: float) -> None:
        """Apply fluid charge and carbon transfers before gas exchange."""
        fluids = self.fluids
        if fluids is None:
            return
        transport = self.gas_transport
        old_volumes = transport.volumes
        old_carbon = transport.blood_amount(transport.co2)
        old_blood_base = transport.blood_amount(transport.base)
        old_total_co2 = old_carbon + transport.extracellular_co2
        old_total_base = transport.base_meq

        sync = fluids.consume_blood_sync()
        keep_0 = clamp01(sync.retention_factor)
        bleed_frac_0 = 1.0 - keep_0

        bled_donor_cells_l = max(0.0, sync.transfused_cells_l - sync.retained_cells_l)
        added_carbon = sync.transfused_cells_l * self._donor_rbc_co2 + sync.carbon_mmol
        blood_infuse_carbon = sync.retained_cells_l * self._donor_rbc_co2 + sync.blood_carbon_retained_mmol
        bled_infuse_carbon = (
            bled_donor_cells_l * self._donor_rbc_co2
            + max(0.0, sync.blood_carbon_in_mmol - sync.blood_carbon_retained_mmol)
        )
        blood_infuse_base = sync.blood_base_retained_meq
        bled_infuse_base = sync.blood_base_in_meq - blood_infuse_base

        urine_carbon = fluids.bicarbonate_excreted_mmol - self._urine_carbon_applied
        self._urine_carbon_applied = fluids.bicarbonate_excreted_mmol

        pi_conc = fluids.concentration("phosphate")

        def _uf_terms(ph: float, pco2: float) -> tuple[float, float, float]:
            alb, _, pi, _ = buffer_coefficients(ph)
            bicarb = plasma_bicarbonate(ph, pco2)
            return _PLASMA_CO2_SOLUBILITY * pco2 + bicarb, bicarb + pi_conc * pi, alb

        uf_co2_b, uf_base_b, alb_b = _uf_terms(self._blood_ph[2], self._blood_pressures[2])
        uf_co2_e, uf_base_e, alb_e = _uf_terms(self._extracellular_ph, self._extracellular_pco2)

        def _shift_terms(s_l: float, s_alb_g: float) -> tuple[float, float]:
            if s_l < 0.0:
                return s_l * uf_co2_b, s_l * uf_base_b + s_alb_g * alb_b
            if s_l > 0.0 or s_alb_g != 0.0:
                return s_l * uf_co2_e, s_l * uf_base_e + s_alb_g * alb_e
            return 0.0, 0.0

        shift_co2_total, shift_base_total = _shift_terms(sync.water_shift_total_l, sync.albumin_shift_total_g)
        shift_co2, shift_base = _shift_terms(sync.water_shift_retained_l, sync.albumin_shift_retained_g)
        lost_carbon = bleed_frac_0 * old_carbon + bled_infuse_carbon + (shift_co2_total - shift_co2) + urine_carbon
        bleed_base = bleed_frac_0 * old_blood_base + bled_infuse_base + (shift_base_total - shift_base)
        fluids.base_meq += sync.bled_base_est_meq - bleed_base
        urine_base = sync.blood_urine_l * uf_base_b

        old_o2_amounts = [v * c for v, c in zip(old_volumes, transport.oxygen, strict=True)]
        donor_o2_per_l = _O2_PER_HB * self._donor_mchc_g_dl * STORED_BLOOD_SO2 + _O2_SOLUBILITY * 30.0
        donor_o2_retained = sync.retained_cells_l * donor_o2_per_l
        if bleed_frac_0 > 0.0:
            old_o2_amounts = [amt * keep_0 for amt in old_o2_amounts]
        if donor_o2_retained > 0.0:
            old_o2_amounts[2] += donor_o2_retained
        if sync.water_shift_retained_l < 0.0:
            lost_dissolved_o2 = -sync.water_shift_retained_l * self._o2_solubility_now * self._blood_po2[2]
            old_o2_amounts[2] = max(0.0, old_o2_amounts[2] - lost_dissolved_o2)

        new_vol = blood_volume_ml / 1000.0
        transport.resize_blood(new_vol)
        transport.extracellular_volume = fluids.interstitial_l
        transport.oxygen = [amt / v for amt, v in zip(old_o2_amounts, transport.volumes, strict=True)]
        self._oxygen_hb = hb

        total_target_co2 = old_total_co2 - lost_carbon + added_carbon
        target_blood_co2 = keep_0 * old_carbon + blood_infuse_carbon + shift_co2 - urine_carbon
        target_blood_co2 = clamp(
            target_blood_co2,
            0.5 * new_vol,
            max(0.5 * new_vol, total_target_co2 - 0.5 * transport.extracellular_volume),
        )
        raw_blood_co2 = transport.blood_amount(transport.co2)
        if raw_blood_co2 > 0.0:
            co2_scale = target_blood_co2 / raw_blood_co2
            transport.co2 = [c * co2_scale for c in transport.co2]
        transport.extracellular_co2 = total_target_co2 - transport.blood_amount(transport.co2)

        net_global_base_delta = fluids.base_meq - old_total_base
        ecf_metabolic_delta = net_global_base_delta - (sync.infuse_base_meq - bleed_base)
        infuse_base_for_blood = blood_infuse_base
        if self._unbuffered_acid_meq > 0.0 and sync.infuse_base_meq > 0.0:
            repaid = min(self._unbuffered_acid_meq, sync.infuse_base_meq)
            self._unbuffered_acid_meq -= repaid
            fluids.base_meq -= repaid
            blood_share = blood_infuse_base / sync.infuse_base_meq if sync.infuse_base_meq > 0.0 else 0.0
            infuse_base_for_blood -= repaid * blood_share
            if fluids._running_blood_base_meq is not None:
                fluids._running_blood_base_meq -= repaid * blood_share
        plasma_share = clamp01(fluids.plasma_l / fluids.ecf_l) if fluids.ecf_l > 0.0 else 0.0
        target_blood_base = (
            keep_0 * old_blood_base
            + infuse_base_for_blood
            + shift_base
            - urine_base
            + plasma_share * (ecf_metabolic_delta + urine_base)
        )
        base_shift = (target_blood_base - transport.blood_amount(transport.base)) / new_vol
        transport.base = [b + base_shift for b in transport.base]
        blood_base = transport.blood_amount(transport.base)
        transport.extracellular_base = fluids.base_meq - blood_base

        plasma, interstitial = fluids.plasma_buffers, fluids.interstitial_buffers
        composition = self._composition(hb, hct, plasma, interstitial)
        for new, old in zip(composition, self._fluid_composition, strict=True):
            if abs(new - old) > 1e-6 * max(1.0, abs(old)):
                self._fluid_composition = composition
                self._plasma_buffers, self._extracellular_buffers = plasma, interstitial
                self._set_blood_composition(hb, hct)
                break
        self._rebalance_base_above_floor()
        fluids.blood_base_meq_l = transport.blood_amount(transport.base) / transport.blood_volume

    @staticmethod
    def _composition(hb: float, hct: float, plasma: PlasmaBuffers, interstitial: PlasmaBuffers) -> tuple[float, ...]:
        return (hb, hct, plasma.albumin_g_l, plasma.phosphate_mmol_l, plasma.cellular_mmol_l,
                interstitial.albumin_g_l, interstitial.phosphate_mmol_l, interstitial.cellular_mmol_l)

    @property
    def va_baseline(self) -> float:
        """Resting alveolar ventilation derived from baseline breath rate and volume."""
        return max(0.1, (self.vt_0 / 1000.0 - self.vd_deadspace) * self.rr_0)

    def _set_low_vq(self, unconscious: float, airway_pressure: float, vq_mismatch: float = 0.0) -> None:
        unconscious = clamp01(unconscious)
        self._anesthetized = max(self._anesthetized, unconscious)
        closed = max(unconscious, self.postoperative_closure * self._anesthetized)
        closure = closed * clamp01(1.0 - airway_pressure / self.airway_opening_pressure)
        airway = self.low_vq_awake + closure * self.low_vq_closure + self.LOW_VQ_AIRWAY * clamp01(vq_mismatch)
        self.low_vq_fraction = airway + self.low_vq_copd
        self._low_vq_share = min(1.0, self.low_vq_fraction / (1.0 - self.shunt_fraction))
        scale = self._low_vq_share / self.low_vq_fraction if self.low_vq_fraction else 0.0
        self._low_vq_units[0].share = airway * scale
        self._low_vq_units[1].share = self.low_vq_copd * scale

    def _seed_capillary_oxygen(self, extraction: float) -> tuple[float, float, float]:
        """Capillary O2 content, and the PO2 and saturation of the mixed blood, at a venous extraction."""
        hb, alveolar = self._oxygen_hb, self.state.p_alveolar_o2
        content = self._oxygen_content(
            max(0.0, alveolar - self.aa_grad_base), hb, self._p50, solubility=self._o2_solubility_now
        )[0]
        for _ in range(50):
            previous = content
            low = self._low_vq_oxygen(alveolar, max(0.0, content - extraction), hb, self._p50)
            content, _, po2, _ = self._capillary_oxygen(alveolar, hb, self._p50, self.aa_grad_base, low)
            if abs(content - previous) < 1e-12:
                po2 = self._po2_from_content(content, hb, self._p50, po2, solubility=self._o2_solubility_now)
                return content, po2, self._saturation_at_p50(po2, self._p50)
        raise RuntimeError("Capillary O2 seed did not converge")

    def equilibrate_oxygen(
        self, fio2: float, unconscious: float = 0.0, airway_pressure: float = 0.0, fin2: float | None = None
    ) -> None:
        """Seed lung and blood oxygen after an explicitly equilibrated exposure."""
        self.alveolar_fn2 = clamp01(1.0 - fio2) if fin2 is None else clamp01(fin2)
        self._set_low_vq(unconscious, airway_pressure)
        self.state.p_alveolar_o2 = max(
            0.0,
            fio2 * self._atm_dry - self.state.p_alveolar_co2 * (fio2 + (1.0 - fio2) / self.rq),
        )
        hb = self._oxygen_hb
        extraction = self.vco2 / self.rq / 1000.0 / self.baseline_co_l_min
        content = self._seed_capillary_oxygen(extraction)[0]
        self.gas_transport.oxygen = [content, content, max(0.0, content - extraction)]
        for index, oxygen in enumerate(self.gas_transport.oxygen):
            po2 = self._po2_from_content(
                oxygen, hb, self._blood_p50[index], self._blood_po2[index], solubility=self._o2_solubility_now
            )
            self._blood_po2[index] = po2
            self._blood_saturations[index] = self._saturation_at_p50(po2, self._blood_p50[index])
        self.state.p_arterial_o2 = self._blood_po2[1]
        self.state.sao2 = 100.0 * self._blood_saturations[1]
        self._apneic_gas_deficit_l = 0.0
        self._pending_mech_absorption_btps_l = 0.0

    def equilibrate_blood_gases(self, cardiac_output: float, production_l_min: float) -> None:
        """Seed blood and tissues for a specified maintenance history."""
        self._unbuffered_acid_meq = 0.0
        transport = self.gas_transport
        hb = self._oxygen_hb
        reference = self.chemistry.evaluate(self.patient.pregnancy.ph, self.paco2_setpoint, 0.98)
        transport.base = [reference.buffer_base] * 3
        # Van Slyke equation, as reported by blood gas analyzers.
        self._reference_base_excess = standard_base_excess(reference.ph, reference.plasma_bicarbonate)
        systemic = production_l_min * (1.0 - self.LUNG_TISSUE_FRACTION)
        extraction = systemic / self.rq / cardiac_output
        oxygen, capillary, saturation = self._seed_capillary_oxygen(extraction)
        transport.oxygen = [oxygen, oxygen, max(0.0, oxygen - extraction)]
        arterial = self.chemistry.solve(self.state.p_alveolar_co2, saturation, reference.buffer_base)
        venous_co2 = arterial.total_co2 + systemic / _CO2_L_PER_MMOL / cardiac_output
        transport.co2 = [arterial.total_co2, arterial.total_co2, venous_co2]
        venous_po2 = self._po2_from_content(
            transport.oxygen[2], hb, self._p50, 40.0, solubility=self._o2_solubility_now
        )
        venous_saturation = self._saturation_at_p50(venous_po2, self._p50)
        venous = self.chemistry.solve_contents(venous_co2, reference.buffer_base, venous_saturation, 46.0, 7.36)
        co2, base, _, _ = extracellular_contents(
            venous.blood.ph, venous.pco2, self._extracellular_buffers,
        )
        transport.extracellular_co2 = transport.extracellular_volume * co2
        transport.extracellular_base = transport.extracellular_volume * base
        self._reference_extracellular_base = base
        self._extracellular_ph = venous.blood.ph
        self._extracellular_pco2 = venous.pco2
        self._blood_pressures[0], self._blood_ph[0] = self.state.p_alveolar_co2, arterial.ph
        self._blood_po2[0], self._blood_saturations[0] = capillary, saturation
        self._resolve_blood_pool(1, hb)
        self._resolve_blood_pool(2, hb)
        transport.seed_tissues(arterial.total_co2, cardiac_output, systemic,
                               self.chemistry, self._blood_saturations[2])
        self._pulmonary_o2_uptake = production_l_min / self.rq
        self._pulmonary_co2_delivery = production_l_min
        self._publish_blood_gases()
        if self.fluids is not None:
            donor = BloodChemistry(self._donor_mchc_g_dl * 0.55 / _HB_G_DL_PER_MMOL_L, 0.55).evaluate(
                STORED_BLOOD_PH, STORED_BLOOD_PCO2, STORED_BLOOD_SO2,
            )
            self.fluids.seed_base(transport.base_meq, donor.rbc_base, reference.buffer_base)
            plasma_carbon = donor.plasma_bicarbonate * (1.0 + 10.0**(6.1 - STORED_BLOOD_PH))
            self._donor_rbc_co2 = (donor.total_co2 - 0.45 * plasma_carbon) / 0.55
            self._urine_carbon_applied = 0.0

    def _gas_volume_at(self, frc_l: float, deficit_l: float) -> float:
        return max(self.vd_deadspace, frc_l - deficit_l / self._btps_to_stpd)

    @property
    def gas_volume_l(self) -> float:
        """Lung gas volume, L BTPS.

        Absorption behind a closed airway shrinks the lung at atmospheric
        pressure until only the conducting airways hold gas; further absorption
        lowers alveolar pressure.
        """
        return self._gas_volume_at(self.frc, self._apneic_gas_deficit_l)

    def consume_mechanical_absorption_l(self) -> float:
        """Return and clear pending unreplaced alveolar gas absorption (BTPS L)."""
        absorbed = self._pending_mech_absorption_btps_l
        self._pending_mech_absorption_btps_l = 0.0
        return absorbed

    def reconcile_collapse_deficit(
        self, volume_change: float, deficit_l: float | None = None
    ) -> tuple[float, float]:
        """Return (exhaled BTPS volume change L, remaining STPD deficit L)."""
        deficit = self._apneic_gas_deficit_l if deficit_l is None else deficit_l
        if volume_change < 0.0 and deficit > 0.0:
            absorbed = min(-volume_change, deficit / self._btps_to_stpd)
            return volume_change + absorbed, max(0.0, deficit - absorbed * self._btps_to_stpd)
        return volume_change, deficit

    def _washout_stpd_l(
        self,
        airway_patency: float,
        effective_va: float,
        n2o_uptake_l_min: float,
        dt: float,
        deficit_l: float | None = None,
    ) -> float:
        """Net outward tracheal gas evolution (STPD L) over dt when blood evolves more gas than it absorbs."""
        del effective_va
        if dt <= 0.0:
            return 0.0
        patency = clamp01(airway_patency)
        if patency <= 0.0:
            return 0.0
        apneic_patency = clamp01(patency / 0.05)
        metabolic_net = self._pulmonary_o2_uptake - self._pulmonary_co2_delivery
        net_blood_uptake = metabolic_net + n2o_uptake_l_min * self._btps_to_stpd
        if net_blood_uptake >= 0.0:
            return 0.0
        deficit = self._apneic_gas_deficit_l if deficit_l is None else max(0.0, deficit_l)
        evolved_l = -net_blood_uptake * dt / 60.0
        return max(0.0, evolved_l - deficit) * apneic_patency

    def _apneic_inflow_components(
        self,
        airway_patency: float,
        effective_va: float,
        n2o_uptake_l_min: float,
        dt: float,
        remaining_flow_limit: float | None = None,
        deficit_l: float | None = None,
    ) -> tuple[float, float, float]:
        """Return (ongoing_o2_inflow, deficit_refill, n2o_inflow) in STPD L/min."""
        del effective_va
        patency = clamp01(airway_patency)
        apneic_patency = clamp01(patency / 0.05)
        n2o_stpd_l_min = n2o_uptake_l_min * self._btps_to_stpd
        metabolic_net = self._pulmonary_o2_uptake - self._pulmonary_co2_delivery
        net_blood_uptake = metabolic_net + n2o_stpd_l_min
        if net_blood_uptake <= 0.0:
            apneic_inflow = 0.0
            n2o_inflow = 0.0
        elif n2o_stpd_l_min >= 0.0:
            if metabolic_net >= 0.0:
                apneic_inflow = metabolic_net * apneic_patency
                n2o_inflow = n2o_stpd_l_min * apneic_patency
            else:
                apneic_inflow = 0.0
                n2o_inflow = net_blood_uptake * apneic_patency
        else:
            apneic_inflow = net_blood_uptake * apneic_patency
            n2o_inflow = 0.0
        deficit = self._apneic_gas_deficit_l if deficit_l is None else max(0.0, deficit_l)
        refill_conductance = patency * patency * patency
        refill_want = (
            deficit * (-math.expm1(-refill_conductance * dt / 0.5)) * 60.0 / dt
            if dt > 0.0 and deficit > 0.0 and net_blood_uptake >= 0.0
            else 0.0
        )
        if remaining_flow_limit is not None:
            cap = max(0.0, remaining_flow_limit)
            ongoing = apneic_inflow + n2o_inflow
            if ongoing > cap:
                scale = cap / ongoing
                apneic_inflow *= scale
                n2o_inflow *= scale
                refill_l_min = 0.0
            else:
                refill_l_min = min(refill_want, cap - ongoing)
        else:
            refill_l_min = refill_want
        return apneic_inflow, refill_l_min, n2o_inflow

    @classmethod
    def _oxygen_content(
        cls, pressure: float, hb: float, p50: float, solubility: float = _O2_SOLUBILITY
    ) -> tuple[float, float]:
        """STPD L O2 per L blood and derivative per mmHg."""
        saturation, derivative = cls._saturation_gradient(pressure, p50)
        return _O2_PER_HB * hb * saturation + solubility * pressure, _O2_PER_HB * hb * derivative + solubility

    def _low_vq_unit(self, unit: _LowVQUnit, alveolar_po2: float, venous_o2: float, hb: float,
                     p50: float) -> tuple[float, float, float, float, float]:
        """Alveolar PO2, and low V/Q end-capillary PO2 and saturation with their slopes per mmHg of it.

        The units draw alveolar gas, and net gas uptake, O2 uptake less CO2 output, draws in more:
        C(P) - Cv = k (PA - P) / (1 - (1 - RQ) PA / Pdry), with k from the unit's V/Q.
        Whole-body RQ stands in for each unit's exchange ratio.
        """
        solubility = self._o2_solubility_now
        net_uptake = 1.0 - self.rq
        denominator = self._atm_dry - net_uptake * alveolar_po2
        exchange = unit.ratio * self._gas_capacity_per_l * self._atm_dry / denominator
        target = venous_o2 + exchange * alveolar_po2
        bound_capacity = _O2_PER_HB * hb
        lower, upper = 0.0, max(800.0, target / (solubility + exchange))
        pressure = min(max(unit.po2, 1e-6), upper)
        for _ in range(40):
            saturation, slope = self._saturation_gradient(pressure, p50)
            residual = bound_capacity * saturation + (solubility + exchange) * pressure - target
            derivative = bound_capacity * slope + solubility + exchange
            if residual > 0.0:
                upper = pressure
            else:
                lower = pressure
            if abs(residual) < 1e-12:
                break
            candidate = pressure - residual / derivative
            pressure = candidate if lower < candidate < upper else (lower + upper) / 2.0
        else:
            raise RuntimeError("Low V/Q unit O2 solve did not converge")
        unit.po2 = pressure
        rise = exchange * (self._atm_dry - net_uptake * pressure) / denominator / derivative
        return alveolar_po2, pressure, saturation, rise, slope * rise

    def _low_vq_oxygen(self, alveolar_po2: float, venous_o2: float, hb: float,
                       p50: float) -> tuple[float, float, float, float, float] | None:
        """Flow-weighted mix of the low V/Q units, keeping each unit's response to alveolar PO2."""
        if not self._low_vq_share:
            return None
        po2 = saturation = po2_slope = saturation_slope = 0.0
        for unit in self._low_vq_units:
            if unit.share:
                _, pressure, sat, pressure_rise, sat_rise = self._low_vq_unit(unit, alveolar_po2, venous_o2, hb, p50)
                weight = unit.share / self._low_vq_share
                po2 += weight * pressure
                saturation += weight * sat
                po2_slope += weight * pressure_rise
                saturation_slope += weight * sat_rise
        return alveolar_po2, po2, saturation, po2_slope, saturation_slope

    def _capillary_oxygen(self, alveolar_po2: float, hb: float, p50: float, gradient: float,
                          low: tuple[float, float, float, float, float] | None) -> tuple[float, float, float, float]:
        """Mixed end-capillary O2 content, its slope per mmHg of alveolar PO2, PO2, and saturation.

        Low V/Q units follow a _low_vq_oxygen solution, linear in alveolar PO2. The PO2 weights
        each stream's dissolved O2, so it and the saturation reproduce the content.
        """
        solubility = self._o2_solubility_now
        po2 = max(0.0, alveolar_po2 - gradient)
        saturation, slope = self._saturation_gradient(po2, p50)
        slope = _O2_PER_HB * hb * slope + solubility if po2 > 0.0 else 0.0
        if low is not None:
            share = self._low_vq_share
            reference, low_po2, low_saturation, po2_slope, saturation_slope = low
            shift = alveolar_po2 - reference
            po2 += share * (low_po2 + po2_slope * shift - po2)
            saturation += share * (low_saturation + saturation_slope * shift - saturation)
            slope += share * (_O2_PER_HB * hb * saturation_slope + solubility * po2_slope - slope)
        return _O2_PER_HB * hb * saturation + solubility * po2, slope, po2, saturation

    @classmethod
    def _po2_from_content(
        cls, content: float, hb: float, p50: float, previous: float, solubility: float = _O2_SOLUBILITY
    ) -> float:
        if content <= 0.0:
            return 0.0
        bound_capacity = _O2_PER_HB * hb
        exponent_decay = math.log(10.0) / 29.25
        lower, upper = 0.0, max(760.0, content / solubility)
        pressure = max(1e-6, min(upper, previous))
        for _ in range(24):
            # Inlined _saturation_gradient; the derivative is needed only for another step.
            correction = 1.20 * 10.0**(-pressure / 29.25)
            exponent = 2.82 - correction
            saturation = 1.0 / (1.0 + (p50 / pressure)**exponent)
            value = bound_capacity * saturation + solubility * pressure
            residual = value - content
            if abs(residual) < 1e-12:
                return pressure
            saturation_derivative = saturation * (1.0 - saturation) * (
                exponent / pressure + exponent_decay * correction * math.log(pressure / p50))
            derivative = bound_capacity * saturation_derivative + solubility
            if residual > 0.0:
                upper = pressure
            else:
                lower = pressure
            candidate = pressure - residual / derivative
            if abs(residual) < 1e-7 and lower < candidate < upper:
                # The curve's curvature leaves this Newton step a content error below 1e-12.
                return candidate
            pressure = candidate if lower < candidate < upper else (lower + upper) / 2.0
        raise RuntimeError("Blood O2 content solve did not converge")

    def _resolve_blood_pool(self, index: int, hb: float) -> None:
        transport = self.gas_transport
        carbon, base = transport.co2[index], transport.base[index]
        # Start from the saturation that carries the new O2 content at the previous affinity.
        pco2, ph, p50 = self._blood_pressures[index], self._blood_ph[index], self._blood_p50[index]
        po2 = self._po2_from_content(
            transport.oxygen[index], hb, p50, self._blood_po2[index], solubility=self._o2_solubility_now
        )
        saturation = self._saturation_at_p50(po2, p50)
        # Alternate single CO2 and O2 updates; saturation barely moves once O2 content is fixed.
        for iteration in range(24):
            iterate = self.chemistry.iterate_contents(carbon, base, saturation, pco2, ph)
            if iterate.converged and iteration:
                break  # Saturation was last updated at this pH and PCO2.
            ph, pco2 = iterate.ph, iterate.pco2
            p50 = self._acid_base_p50(ph, pco2)
            po2 = self._po2_from_content(
                transport.oxygen[index], hb, p50, po2, solubility=self._o2_solubility_now
            )
            updated = self._saturation_at_p50(po2, p50)
            if iterate.converged and abs(updated - saturation) < 1e-10:
                break
            saturation = updated
        else:
            raise RuntimeError("Blood O2/CO2 equilibrium did not converge")
        self._blood_pressures[index], self._blood_ph[index] = pco2, ph
        self._blood_po2[index], self._blood_saturations[index] = po2, saturation
        self._blood_p50[index] = p50
        if index == 1:
            self.state.hco3 = iterate.blood.plasma_bicarbonate

    def _exchange_extracellular(self, hb: float, contacted_l: float) -> None:
        """Equilibrate venous blood renewed by flow with extracellular fluid and mix it back.

        The fluid is far larger than the renewed blood, so the blood's saturation is
        taken at the fluid's last affinity. The fluid takes the opposite change, so CO2
        and base are conserved at any tolerance.
        """
        if contacted_l <= 0.0:
            return
        transport = self.gas_transport
        ph, pco2 = self._extracellular_ph, self._extracellular_pco2
        p50 = self._acid_base_p50(ph, pco2)
        po2 = self._po2_from_content(
            transport.oxygen[2], hb, p50, self._blood_po2[2], solubility=self._o2_solubility_now
        )
        fluid = self.chemistry.solve_contents(
            transport.co2[2] + transport.extracellular_co2 / contacted_l,
            transport.base[2] + transport.extracellular_base / contacted_l,
            self._saturation_at_p50(po2, p50), pco2, ph, transport.extracellular_volume / contacted_l,
        )
        self._extracellular_ph, self._extracellular_pco2 = fluid.blood.ph, fluid.pco2
        carbon_change = contacted_l * (fluid.blood.total_co2 - transport.co2[2])
        base_change = contacted_l * (fluid.blood.buffer_base - transport.base[2])
        volume = transport.volumes[2]
        transport.co2[2] += carbon_change / volume
        transport.base[2] += base_change / volume
        transport.extracellular_co2 -= carbon_change
        transport.extracellular_base -= base_change

    def _equilibrate_lung(self, carbon: float, oxygen: float, hb: float,
                          gas_capacity: float, aa_gradient: float, carbon_plan: TransportPlan,
                          oxygen_plan: TransportPlan, flow: float) -> tuple[float, float]:
        """Equilibrate lung gas and pulmonary capillary blood at fixed amounts."""
        state, transport = self.state, self.gas_transport
        pulmonary = flow * (1.0 - self.shunt_fraction)
        volume = transport.volumes[0] + pulmonary
        pco2, po2 = state.p_alveolar_co2, state.p_alveolar_o2
        ph, p50 = self._blood_ph[0], self._blood_p50[0]
        base = transport.base[0]
        carbon_volume = _CO2_L_PER_MMOL * (volume - pulmonary * carbon_plan.venous_gain)
        low = self._low_vq_oxygen(po2, transport.oxygen[2], hb, p50)
        for _ in range(24):
            saturation = self._capillary_oxygen(po2, hb, p50, aa_gradient, low)[3]
            chemical = self.chemistry.evaluate(ph, pco2, saturation)
            # Newton step in pH and PCO2 for fixed buffer base and conserved CO2.
            base_residual = chemical.buffer_base - base
            venous_co2 = carbon_plan.venous_base + carbon_plan.venous_gain * chemical.total_co2
            residual = gas_capacity * pco2 + _CO2_L_PER_MMOL * (volume * chemical.total_co2 - pulmonary * venous_co2) - carbon
            carbon_ph = carbon_volume * chemical.content_ph
            carbon_co2 = gas_capacity + carbon_volume * chemical.content_co2
            determinant = chemical.base_ph * carbon_co2 - chemical.base_co2 * carbon_ph
            ph_change = (base_residual * carbon_co2 - residual * chemical.base_co2) / determinant
            change = (residual * chemical.base_ph - base_residual * carbon_ph) / determinant
            fraction = step_fraction(ph, ph_change, pco2, change)
            ph -= fraction * ph_change
            pco2 -= fraction * change
            p50 = self._acid_base_p50(ph, pco2)
            content, derivative, _, _ = self._capillary_oxygen(po2, hb, p50, aa_gradient, low)
            _, venous_o2, venous_gain = transport.oxygen_blood(content, oxygen_plan, flow, self.shunt_fraction)
            o2_residual = gas_capacity * po2 + volume * content - pulmonary * venous_o2 - oxygen
            effective_volume = volume - pulmonary * venous_gain
            o2_change = o2_residual / (gas_capacity + effective_volume * derivative)
            next_po2 = max(0.0, po2 - o2_change)
            o2_step = po2 - next_po2
            po2 = next_po2
            o2_converged = abs(o2_change) < 1e-6 or (po2 == 0.0 and o2_change >= 0.0 and abs(o2_step) < 1e-6)
            # Newton converges quadratically, so the step after a 1e-6 mmHg step is negligible.
            if abs(change) < 1e-6 and o2_converged and abs(base_residual) < 1e-8:
                break
        else:
            raise RuntimeError("Pulmonary blood gas equilibrium did not converge")
        # Carry blood CO2 along the final step, as the conserved residual was linearized.
        capillary_co2 = chemical.total_co2 - fraction * (chemical.content_ph * ph_change + chemical.content_co2 * change)
        state.p_alveolar_co2, state.p_alveolar_o2 = pco2, po2
        self._blood_pressures[0], self._blood_ph[0] = pco2, ph
        content, _, capillary, saturation = self._capillary_oxygen(po2, hb, p50, aa_gradient, low)
        self._blood_po2[0], self._blood_p50[0], self._blood_saturations[0] = capillary, p50, saturation
        return capillary_co2, content

    def _publish_blood_gases(self) -> None:
        state = self.state
        delta_t = 37.0 - self._temp_c
        pco2_scale = self._pco2_temp_scale_now
        ph_shift = 0.0147 * delta_t
        state.pa_co2 = self._blood_pressures[1] * pco2_scale
        self._last_published_pa_co2 = state.pa_co2
        state.ph = self._blood_ph[1] + ph_shift
        state.p_arterial_o2 = self._blood_po2[1]
        state.sao2 = 100.0 * self._blood_saturations[1]
        state.pv_co2 = self._blood_pressures[2] * pco2_scale
        state.pv_o2 = self._blood_po2[2]
        state.venous_ph = self._blood_ph[2] + ph_shift
        self._p50 = self._blood_p50[1]
        self.buffer_base = self.gas_transport.base[1]
        if self.fluids is not None:
            transport = self.gas_transport
            self.fluids.blood_base_meq_l = transport.blood_amount(transport.base) / transport.blood_volume
        self._publish_base_excess()

    def _publish_base_excess(self) -> None:
        if self.fluids is not None:
            self.state.base_excess = standard_base_excess(self._blood_ph[1], self.state.hco3)
            return
        # Respiratory changes move base between blood and extracellular fluid,
        # so base excess uses their combined change from the resting reference.
        transport = self.gas_transport
        blood_volume = transport.blood_volume
        acquired_base = (
            transport.base_meq - blood_volume * self._reference_buffer_base
            - transport.extracellular_volume * self._reference_extracellular_base
        )
        self.state.base_excess = self._reference_base_excess + acquired_base / (
            blood_volume + transport.extracellular_volume
        )

    def arterial_po2_at_37(self) -> float:
        """Arterial PO2 after the analyzer warms the sample to 37 °C, mmHg."""
        p50 = self._blood_p50[1] * self._temperature_p50(37.0) / self._temperature_p50_now
        return self._po2_from_content(
            self.gas_transport.oxygen[1], self._oxygen_hb, p50, self._blood_po2[1], solubility=_O2_SOLUBILITY
        )

    def _rebalance_base_above_floor(self) -> None:
        """Keep blood and interstitial buffer base above the pH 5.05 solver floor, tracking excess acid."""
        transport = self.gas_transport
        pco2_blood = max(200.0, *self._blood_pressures)
        pco2_ext = max(200.0, self._extracellular_pco2)
        min_blood = self.chemistry.evaluate(5.05, pco2_blood, 1.0).buffer_base
        min_ext = extracellular_contents(5.05, pco2_ext, self.chemistry.extracellular_buffers)[1]
        ext_floor = min_ext * transport.extracellular_volume
        if transport.extracellular_base < ext_floor:
            ext_deficit = ext_floor - transport.extracellular_base
            transport.extracellular_base = ext_floor
            blood_avail = (
                max(0.0, (min(transport.base) - min_blood) * transport.blood_volume)
                if transport.blood_volume > 0.0
                else 0.0
            )
            from_blood = min(ext_deficit, blood_avail)
            if from_blood > 0.0:
                shift = from_blood / transport.blood_volume
                transport.base = [b - shift for b in transport.base]
                if self.fluids is not None and self.fluids._running_blood_base_meq is not None:
                    self.fluids._running_blood_base_meq -= from_blood
            unbuffered = ext_deficit - from_blood
            if unbuffered > 0.0:
                self._unbuffered_acid_meq += unbuffered
                if self.fluids is not None:
                    self.fluids.base_meq += unbuffered
        if transport.blood_volume > 0.0 and min(transport.base) < min_blood:
            blood_deficit_per_l = min_blood - min(transport.base)
            blood_deficit = blood_deficit_per_l * transport.blood_volume
            transport.base = [b + blood_deficit_per_l for b in transport.base]
            if self.fluids is not None and self.fluids._running_blood_base_meq is not None:
                self.fluids._running_blood_base_meq += blood_deficit
            ext_avail = max(0.0, transport.extracellular_base - ext_floor)
            from_ext = min(blood_deficit, ext_avail)
            transport.extracellular_base -= from_ext
            unbuffered = blood_deficit - from_ext
            if unbuffered > 0.0:
                self._unbuffered_acid_meq += unbuffered
                if self.fluids is not None:
                    self.fluids.base_meq += unbuffered

    def remove_buffer_base(self, meq: float) -> None:
        """Remove buffer base across extracellular fluid and blood; a negative amount adds base."""
        if meq < 0.0 and self._unbuffered_acid_meq > 0.0:
            repaid = min(self._unbuffered_acid_meq, -meq)
            self._unbuffered_acid_meq -= repaid
            meq += repaid
            if abs(meq) <= 1e-15:
                return
        transport = self.gas_transport
        if self.fluids is not None:
            plasma_share = clamp01(self.fluids.plasma_l / self.fluids.ecf_l) if self.fluids.ecf_l > 0.0 else 0.0
            self.fluids.base_meq -= meq
        else:
            total_vol = transport.blood_volume + transport.extracellular_volume
            plasma_share = clamp01(transport.blood_volume / total_vol) if total_vol > 0.0 else 0.0
        blood_delta = meq * plasma_share
        if transport.blood_volume > 0.0 and blood_delta != 0.0:
            shift = blood_delta / transport.blood_volume
            transport.base = [b - shift for b in transport.base]
        transport.extracellular_base -= meq - blood_delta
        if self.fluids is not None and transport.blood_volume > 0.0:
            if self.fluids._running_blood_base_meq is not None:
                self.fluids._running_blood_base_meq -= blood_delta
        self._rebalance_base_above_floor()
        if self.fluids is not None and transport.blood_volume > 0.0:
            self.fluids.blood_base_meq_l = transport.blood_amount(transport.base) / transport.blood_volume
        self._publish_base_excess()

    def central_nmba_block(self, ce_roc: float, mac_sevo: float = 0.0, mac_n2o: float = 0.0) -> float:
        """Diaphragmatic/central neuromuscular block fraction (0-1) from potentiated free rocuronium."""
        if ce_roc <= 0.0:
            return 0.0
        ratio = ce_roc / self.c50_nmba
        excess = max(0.0, ratio - 1.0)
        mac_pot = clamp01(mac_sevo + 0.5 * mac_n2o)
        gamma = self.gamma_nmba + 3.5 * mac_pot * excess / (0.35 + excess)
        return hill_function(ce_roc, self.c50_nmba, gamma)

    def _substep_count(self, dt: float, cardiac_output: float, shunt_fraction: float) -> int:
        # Crank-Nicolson advection preserves nonnegative contents when
        # no pool exports more than its volume in the explicit half-step.
        max_step = 1.0
        if cardiac_output > 0.0:
            vp, va, vv = self.gas_transport.volumes
            transit_volume = min(vp / (1.0 - clamp(shunt_fraction, 0.0, 0.95)), va, vv)
            max_step = min(max_step, 120.0 * transit_volume / cardiac_output)
        return max(1, math.ceil(dt / max_step)) if dt > max_step else 1

    def _frc_volume_change_l(
        self,
        lung_volume_l: float | None,
        airway_patency: float,
        dt: float,
        flow_limit: float | None = None,
        frc_l: float | None = None,
    ) -> tuple[float, float | None]:
        """Return (BTPS FRC delta L, remaining STPD flow allowance L/min)."""
        remaining = None if flow_limit is None else max(0.0, flow_limit)
        if lung_volume_l is None or dt <= 0.0:
            return 0.0, remaining
        patency = clamp01(airway_patency)
        patency_gate = 1.0 if patency >= 0.05 else -math.expm1(-dt * (patency / 0.05))
        base_frc = self.frc if frc_l is None else frc_l
        volume_change = (lung_volume_l - base_frc) * patency_gate
        if volume_change > 0.0 and remaining is not None:
            volume_change = min(volume_change, remaining * dt / 60.0 / self._btps_to_stpd)
            remaining = max(0.0, remaining - volume_change * self._btps_to_stpd * 60.0 / dt)
        return volume_change, remaining

    def _step_nitrogen_store(
        self,
        lung_n2: float,
        fn2: float,
        target_fn2: float,
        ventilation: float,
        inflow: float,
        washout_l: float,
        gas_stpd_l: float,
        dt: float,
    ) -> tuple[float, float, float]:
        if washout_l <= 0.0:
            washout_n2_l = 0.0
        else:
            washout_n2_l = min(
                max(0.0, 0.85 * lung_n2),
                washout_l * clamp01(lung_n2 / max(1e-9, gas_stpd_l)),
            )
        if dt > 0.0 and gas_stpd_l > 0.0:
            vent_stpd_l = ventilation * self._btps_to_stpd * dt / 60.0
            mixing_n2 = vent_stpd_l / (2.0 * gas_stpd_l)
            passive_n2_l = inflow * target_fn2 * dt / 60.0 - washout_n2_l
            tidal_n2_l = (
                vent_stpd_l * (target_fn2 - (lung_n2 + 0.5 * passive_n2_l) / gas_stpd_l)
                / (1.0 + mixing_n2)
            )
            lung_n2 = max(0.0, lung_n2 + passive_n2_l + tidal_n2_l)
            fn2 = clamp01(lung_n2 / gas_stpd_l)
        return lung_n2, fn2, washout_n2_l

    def oxygen_exchange_l_min(
        self, fio2: float, alveolar_ventilation: float,
        airway_patency: float, vq_mismatch: float, dt: float = 1.0,
        apneic_flow_limit_l_min: float | None = None,
        n2o_uptake_l_min: float = 0.0,
        lung_volume_l: float | None = None,
    ) -> float:
        """Net O2 transport from inspired gas into the lung and blood stores, STPD L/min."""
        if airway_patency <= 0.0:
            return 0.0
        patency = clamp01(airway_patency)
        ventilation = max(0.0, alveolar_ventilation) * (1.0 - 0.6 * clamp01(vq_mismatch))
        volume_change, remaining_flow_limit = self._frc_volume_change_l(
            lung_volume_l, patency, dt, apneic_flow_limit_l_min
        )
        exhaled_vc, deficit_after_collapse = self.reconcile_collapse_deficit(volume_change)
        frc_o2_l_min = 0.0
        if exhaled_vc != 0.0 and dt > 0.0:
            gas_o2 = fio2 * self._atm_dry if exhaled_vc > 0.0 else self.state.p_alveolar_o2
            frc_o2_l_min = exhaled_vc * self._gas_capacity_per_l * gas_o2 * 60.0 / dt
        apneic_inflow, refill_l_min, n2o_inflow = self._apneic_inflow_components(
            patency, ventilation, n2o_uptake_l_min, dt, remaining_flow_limit, deficit_l=deficit_after_collapse
        )
        deficit_after_refill = max(0.0, deficit_after_collapse - refill_l_min * dt / 60.0)
        gas_vol = self._gas_volume_at(self.frc + volume_change, deficit_after_refill)
        passive_flow = apneic_inflow + refill_l_min + n2o_inflow
        washout_l = self._washout_stpd_l(
            patency, ventilation, n2o_uptake_l_min, dt, deficit_l=deficit_after_collapse
        )
        washout_o2_l_min = 0.0
        if washout_l > 0.0 and dt > 0.0:
            lung_o2 = (
                self.gas_volume_l * self._gas_capacity_per_l * self.state.p_alveolar_o2
                + frc_o2_l_min * dt / 60.0
            )
            washout_o2_l_min = (
                min(max(0.0, 0.85 * lung_o2), washout_l * clamp01(self.state.p_alveolar_o2 / self._atm_dry))
                * 60.0
                / dt
            )
        initial_flux = (
            ventilation * self._btps_to_stpd
            * (fio2 - self.state.p_alveolar_o2 / self._atm_dry)
            + passive_flow * fio2
            - washout_o2_l_min
        )
        # Predict the interval mean gas pressure for the circuit's O2 balance.
        # The respiratory solve uses the same midpoint ventilation flux.
        mixing = ventilation * dt / (120.0 * gas_vol)
        flux = frc_o2_l_min + (initial_flux + mixing * self._pulmonary_o2_uptake) / (1.0 + mixing)
        return min(0.0, flux) if fio2 <= 0.0 else flux

    def nitrogen_exchange_l_min(
        self,
        target_fn2: float,
        alveolar_ventilation: float,
        airway_patency: float,
        vq_mismatch: float,
        dt: float = 1.0,
        apneic_flow_limit_l_min: float | None = None,
        n2o_uptake_l_min: float = 0.0,
        lung_volume_l: float | None = None,
        cardiac_output: float = 5.0,
        shunt_fraction: float = 0.0,
    ) -> float:
        """Net N2 transport from inspired gas into the lung gas store, STPD L/min."""
        if airway_patency <= 0.0 or dt <= 0.0:
            return 0.0
        patency = clamp01(airway_patency)
        target_fn2 = clamp01(target_fn2)
        ventilation = max(0.0, alveolar_ventilation) * (1.0 - 0.6 * clamp01(vq_mismatch))
        count = self._substep_count(dt, cardiac_output, shunt_fraction)
        sub_dt = dt / count
        start_frc = self.frc
        sim_frc = self.frc
        sim_deficit = self._apneic_gas_deficit_l
        sim_fn2 = self.alveolar_fn2
        initial_n2_l = self._gas_volume_at(sim_frc, sim_deficit) * self._btps_to_stpd * sim_fn2
        sim_lung_n2 = initial_n2_l
        for i in range(count):
            sub_target_vol = (
                start_frc + (lung_volume_l - start_frc) * (i + 1) / count
                if lung_volume_l is not None and patency >= 0.05
                else lung_volume_l
            )
            vc, remaining = self._frc_volume_change_l(
                sub_target_vol, patency, sub_dt, apneic_flow_limit_l_min, frc_l=sim_frc
            )
            exhaled_vc, sim_deficit = self.reconcile_collapse_deficit(vc, sim_deficit)
            if exhaled_vc != 0.0:
                gas_n2 = target_fn2 if exhaled_vc > 0.0 else sim_fn2
                sim_lung_n2 += exhaled_vc * self._btps_to_stpd * gas_n2
            sim_frc += vc
            apneic_inflow, refill_l_min, n2o_inflow = self._apneic_inflow_components(
                patency, ventilation, n2o_uptake_l_min, sub_dt, remaining, deficit_l=sim_deficit
            )
            washout_l = self._washout_stpd_l(
                patency, ventilation, n2o_uptake_l_min, sub_dt, deficit_l=sim_deficit
            )
            sim_deficit = max(0.0, sim_deficit - refill_l_min * sub_dt / 60.0)
            gas_stpd_l = self._gas_volume_at(sim_frc, sim_deficit) * self._btps_to_stpd
            inflow = apneic_inflow + refill_l_min + n2o_inflow
            sim_lung_n2, sim_fn2, _ = self._step_nitrogen_store(
                sim_lung_n2,
                sim_fn2,
                target_fn2,
                ventilation,
                inflow,
                washout_l,
                gas_stpd_l,
                sub_dt,
            )
        return (sim_lung_n2 - initial_n2_l) * 60.0 / dt

    def step(self, dt: float, ce_prop: float, ce_remi: float, mech_vent_mv: float = 0.0,
             fio2: float = 0.21, ce_roc: float = 0.0, mac_sevo: float = 0.0,
             mech_rr: float = 0.0, mech_vt_l: float = 0.0,
             airway_patency: float = 1.0, ventilation_efficiency: float = 1.0,
             vq_mismatch: float = 0.0,
             hb_g_dl: float | None = None, hematocrit: float | None = None,
             cardiac_output: float = 5.0,
             metabolic_factor: float = 1.0,
             blood_volume_ml: float | None = None,
             measured_breaths: bool = False,
             unconscious: float = 0.0, lung_volume_l: float | None = None,
             shunt_fraction: float = 0.0, temp_c: float = 37.0,
             opioid_controller_fraction: float = 1.0, neuromuscular_strength: float = 1.0,
             opioid_rate_fraction: float = 1.0, airway_pressure: float = 0.0,
             central_drive_fraction: float = 1.0,
             ketamine_cp: float = 0.0,
             mac_n2o: float = 0.0,
             n2o_uptake_l_min: float = 0.0,
             fin2: float | None = None,
             apneic_flow_limit_l_min: float | None = None,
             mech_supported_fraction: float = 1.0) -> RespState:
        """Advance ventilatory control and blood-lung gas exchange by dt seconds."""
        transport = self.gas_transport
        blood_volume = self.baseline_blood_volume_ml if blood_volume_ml is None else blood_volume_ml
        transport.resize_blood(blood_volume / 1000.0)
        count = self._substep_count(dt, cardiac_output, shunt_fraction)
        if count > 1:
            start_frc = self.frc
            for i in range(count):
                sub_lung_vol = (
                    start_frc + (lung_volume_l - start_frc) * (i + 1) / count
                    if lung_volume_l is not None and airway_patency >= 0.05
                    else lung_volume_l
                )
                self.step(
                    dt / count, ce_prop, ce_remi, mech_vent_mv, fio2, ce_roc, mac_sevo,
                    mech_rr, mech_vt_l, airway_patency, ventilation_efficiency, vq_mismatch,
                    hb_g_dl, hematocrit, cardiac_output, metabolic_factor, blood_volume_ml,
                    measured_breaths, unconscious, sub_lung_vol, shunt_fraction, temp_c,
                    opioid_controller_fraction, neuromuscular_strength,
                    opioid_rate_fraction, airway_pressure, central_drive_fraction,
                    ketamine_cp=ketamine_cp, mac_n2o=mac_n2o,
                    n2o_uptake_l_min=n2o_uptake_l_min, fin2=fin2,
                    apneic_flow_limit_l_min=apneic_flow_limit_l_min,
                    mech_supported_fraction=mech_supported_fraction,
                )
            return self.state
        state = self.state

        prop_keep = 1.0 - hill_function(ce_prop, self.c50_prop, self.gamma_prop)
        sevo_apnea = clamp01((2.0 - mac_sevo) / 0.6)
        sevo_keep = (1.0 - hill_function(mac_sevo, self.c50_sevo_mac, self.gamma_sevo)) * sevo_apnea
        sevo_rate = (
            (1.0 + 0.50 * hill_function(mac_sevo, 0.7, 2.0)) * clamp01((2.0 - mac_sevo) / 0.4)
            if mac_sevo > 0.0
            else 1.0
        ) * (1.0 + 0.25 * clamp01(mac_n2o))
        muscle_factor = 1.0 - self.central_nmba_block(ce_roc, mac_sevo, mac_n2o)
        muscle_factor *= clamp01(neuromuscular_strength)
        airway_patency = clamp01(airway_patency)
        ventilation_efficiency = clamp01(ventilation_efficiency)
        unconscious = clamp01(unconscious)
        ket_bolus_keep = 1.0 / (1.0 + max(0.0, ketamine_cp - 10.0) / 8.0)
        self._central_drive_fraction = clamp01(central_drive_fraction * ket_bolus_keep)

        # At Ce_prop=1 the opioid C50 is 0.84 * 1.6 ng/mL (Olofsen).
        # Interpolation to other propofol concentrations is a transfer assumption.
        remi_c50 = self.c50_remi * (1.0 - 0.32 * max(0.0, ce_prop) / (1.0 + max(0.0, ce_prop)))
        # Base deficit lowers the defended PCO2 by the usual 1.2 mmHg per mmol/L
        # over about two minutes. Base excess, unlike bicarbonate, does not
        # fall with respiratory alkalosis.
        deficit = max(0.0, self._reference_base_excess - state.base_excess)
        compensated_pco2 = max(10.0, self.paco2_setpoint - 1.2 * deficit)
        target_compensation = self.paco2_setpoint / compensated_pco2
        self._metabolic_compensation += (target_compensation - self._metabolic_compensation) * -math.expm1(-dt / 120.0)
        baseline_gain = self.hcvr_slope_baseline / 2.2 * (0.42 / 7.2)
        gain = baseline_gain * prop_keep ** (self.prop_co2_gain_exponent - 1.0) * sevo_keep
        sensed_paco2 = (
            self._blood_pressures[1]
            if state.pa_co2 == self._last_published_pa_co2
            else state.pa_co2
        )
        co2_excess = max(0.0, sensed_paco2 * self._metabolic_compensation - self.paco2_setpoint)
        opioid_reference = 1.0 + gain * co2_excess
        long_opioid_keep = clamp01(opioid_controller_fraction)
        long_opioid_rate = clamp01(opioid_rate_fraction)
        target_drive = opioid_reference * long_opioid_keep - 0.5 * max(0.0, ce_remi) / remi_c50
        # Keep the controller signed during apnea. Clipping its internal state
        # discards drug inhibition and makes recovery too early.
        alpha = -math.expm1(-dt / self.CO2_CONTROLLER_TAU_S)
        self._ventilatory_drive += alpha * (target_drive - self._ventilatory_drive)
        self._resting_age_s += dt
        if self._resting_pco2 is None or self._resting_age_s >= self.RESTING_PCO2_INTERVAL_S:
            # The RR curve includes the infusion's CO2 rise. Extra rate feedback
            # starts above the resting PCO2 predicted for the long opioid alone.
            self._rate_pco2_reference = self.paco2_setpoint
            if long_opioid_keep < 1.0:
                self._rate_pco2_reference = self._solve_resting_pco2(
                    1.0, 1.0, 1.0, 0.0, baseline_gain,
                    0.0, self._awake_metabolism, long_opioid_keep, long_opioid_rate,
                )
            self._resting_pco2 = self._solve_resting_pco2(
                prop_keep, sevo_keep, 1.0, unconscious, gain,
                0.5 * max(0.0, ce_remi) / remi_c50,
                min(self._awake_metabolism, max(0.1, float(metabolic_factor))),
                long_opioid_keep, long_opioid_rate, sevo_rate=sevo_rate,
            )
            self._resting_age_s = 0.0
        effort_rr, effort_vt, drive_central = self._unassisted(
            sensed_paco2, prop_keep, sevo_keep, muscle_factor,
            unconscious, opioid_reference, resting_pco2=self._resting_pco2,
            hypoxic_drive=opioid_reference * self._step_hypoxic_control(dt, ce_prop, mac_sevo, mac_n2o),
            opioid_controller_fraction=long_opioid_keep, opioid_rate_fraction=long_opioid_rate,
            sevo_rate=sevo_rate,
        )
        state.effort_rr, state.effort_vt = effort_rr, effort_vt
        current_rr = effort_rr
        resistive_keep = 1.0 / (1.0 + 1.25 * (1.0 - ventilation_efficiency))
        current_vt = effort_vt * airway_patency * resistive_keep
        if current_vt < max(100.0, VT_MIN):
            current_rr = current_vt = 0.0

        state.apnea = current_rr <= 0.0

        vd = self.vd_deadspace
        vt_eff_spont = max(0.0, current_vt / 1000.0 - vd) * ventilation_efficiency
        # Assisted volumes arrive net of any mask leak; upper-airway stenosis
        # increases turbulent dead-space mixing while distal bronchospasm lowers
        # alveolar exchange efficiency after anatomic dead space.
        upper_eff = max(0.1, 1.0 - 0.2 * (1.0 - airway_patency))
        bronch_eff = clamp01(ventilation_efficiency / upper_eff)
        mech_bronch_eff = 1.0 - clamp01(mech_supported_fraction) * (1.0 - bronch_eff)
        alveolar_vt_mech = max(0.0, mech_vt_l * upper_eff - vd) * mech_bronch_eff
        if mech_vt_l <= 0 and mech_vent_mv > 0 and mech_rr > 0:
            inferred_vt = mech_vent_mv / mech_rr
            alveolar_vt_mech = max(0.0, inferred_vt * upper_eff - vd) * mech_bronch_eff

        # Assisted breaths augment or replace spontaneous ones, so each breath
        # gets the larger of the two volumes at the higher of the two rates.
        ref_rr_mech = max(0.0, mech_rr)
        ref_rr_spont = max(0.0, current_rr)
        effective_rate = max(ref_rr_mech, ref_rr_spont)
        vt_mech_avail = alveolar_vt_mech
        vt_spont_avail = vt_eff_spont

        if measured_breaths:
            # Measured ventilation already includes the patient's breaths, so a
            # separate neural rate would count extra gas.
            total_va_l_min = ref_rr_mech * vt_mech_avail
        else:
            total_va_l_min = effective_rate * (max(vt_mech_avail, vt_spont_avail) if mech_rr > 0 else vt_spont_avail)
        if airway_patency <= 0.0:
            total_va_l_min = 0.0
        state.va = total_va_l_min

        vq_mismatch = clamp01(vq_mismatch)
        effective_va = total_va_l_min * (1.0 - 0.6 * vq_mismatch)
        metabolic_factor = max(0.1, float(metabolic_factor))
        # EtCO2 samples diluted alveolar CO2. The PaCO2-EtCO2 gap widens
        # with dead space, V/Q mismatch, obstruction, and low cardiac output
        # (Russell 1990; Lujan 2008; Kim 2019).
        perfusion_ratio = 1.0
        if self.baseline_co_l_min > 0:
            perfusion_ratio = clamp(cardiac_output / self.baseline_co_l_min, 0.05, 1.2)
        if measured_breaths:
            vt_for_gradient_l = mech_vt_l
        elif mech_rr > 0:
            vt_for_gradient_l = max(mech_vt_l, current_vt / 1000.0)
        else:
            vt_for_gradient_l = current_vt / 1000.0
        vt_l = max(0.05, vt_for_gradient_l)
        vd_vt = min(0.95, self.vd_deadspace / vt_l)
        perfusion_excess = self.perfusion_deadspace_gain * clamp01(1.0 - perfusion_ratio)
        vd_vt_excess = max(0.0, vd_vt - 0.30 + perfusion_excess)
        etco2_gradient = 4.0 + 15.0 * vd_vt_excess + 8.0 * vq_mismatch + 6.0 * (1.0 - ventilation_efficiency)
        etco2_gradient = min(20.0, etco2_gradient)
        hb = self.baseline_hb if hb_g_dl is None else hb_g_dl
        hct = self.baseline_hct if hematocrit is None else hematocrit
        new_solubility = self._temperature_o2_solubility(temp_c)
        if hb != self._oxygen_hb:
            # Each pool keeps its PO2 and saturation as Hb changes.
            transport.oxygen = [
                _O2_PER_HB * hb * saturation + self._o2_solubility_now * po2
                for po2, saturation in zip(self._blood_po2, self._blood_saturations, strict=True)
            ]
            self._oxygen_hb = hb
        old_hb, old_hct = self._chemistry_composition
        # Fluid kinetics move Hb slightly every step; rebuild buffering for measurable changes.
        if abs(hb - old_hb) > 1e-6 * old_hb or abs(hct - old_hct) > 1e-6 * old_hct:
            previous_reference = self._reference_buffer_base
            reference = self._set_blood_composition(hb, hct)
            if self.fluids is None:
                shift = reference.buffer_base - previous_reference
                transport.base = [base + shift for base in transport.base]
        self._temp_c = temp_c
        self._pco2_temp_scale_now = self._temperature_pco2_scale(temp_c)
        self._temperature_p50_now = self._temperature_p50(temp_c)
        self._o2_solubility_now = new_solubility
        self.shunt_fraction = clamp(shunt_fraction, 0.0, 0.95)
        self._set_low_vq(unconscious, airway_pressure, vq_mismatch)
        # Above the critical delivery, extraction rises to hold consumption at
        # demand. Below it, consumption and aerobic CO2 production fall together.
        production = self.vco2 * metabolic_factor / 1000.0
        lung_demand = self.LUNG_TISSUE_FRACTION * production
        production -= lung_demand
        # Mitochondrial respiration continues down to a PO2 near 1 mmHg.
        lung_supply = clamp01(state.p_alveolar_o2)
        lung_production = lung_demand * lung_supply
        demand = production / self.rq
        delivery = max(0.0, cardiac_output) * transport.oxygen[1]
        supply = min(1.0, self.CRITICAL_EXTRACTION * delivery / demand)
        self.oxygen_supply_fraction = supply
        self.oxygen_deficit_l += (demand * (1.0 - supply)
                                  + lung_demand / self.rq * (1.0 - lung_supply)) * dt / 60.0
        production *= supply
        consumption = production / self.rq
        carbon_plan, oxygen_plan, pulmonary = transport.prepare(
            dt, cardiac_output, self.shunt_fraction, production, consumption, self.chemistry,
            self._blood_saturations[2],
        )
        gas_capacity = self.gas_volume_l * self._gas_capacity_per_l
        gas_stpd_l = self.gas_volume_l * self._btps_to_stpd
        lung_co2 = gas_capacity * state.p_alveolar_co2
        lung_o2 = gas_capacity * state.p_alveolar_o2
        lung_n2 = gas_stpd_l * self.alveolar_fn2
        target_fn2 = clamp01(1.0 - fio2) if fin2 is None else clamp01(fin2)
        volume_change, remaining_flow_limit = self._frc_volume_change_l(
            lung_volume_l, airway_patency, dt, apneic_flow_limit_l_min
        )
        exhaled_vc, self._apneic_gas_deficit_l = self.reconcile_collapse_deficit(volume_change)
        if exhaled_vc != 0.0:
            gas_o2 = fio2 * self._atm_dry if exhaled_vc > 0.0 else state.p_alveolar_o2
            gas_n2 = target_fn2 if exhaled_vc > 0.0 else self.alveolar_fn2
            delta_o2 = exhaled_vc * self._gas_capacity_per_l * gas_o2
            lung_o2 += delta_o2
            lung_n2 += exhaled_vc * self._btps_to_stpd * gas_n2
            transport.oxygen_from_air_l += delta_o2
            if exhaled_vc < 0.0:
                lost_co2 = -exhaled_vc * self._gas_capacity_per_l * state.p_alveolar_co2
                lung_co2 -= lost_co2
                transport.exhaled_l += lost_co2
        if volume_change != 0.0:
            self.frc += volume_change
        old_co2, old_o2 = state.p_alveolar_co2, state.p_alveolar_o2
        prev_fn2 = self.alveolar_fn2
        deficit_before_refill = self._apneic_gas_deficit_l
        vol_before_absorb = self._gas_volume_at(self.frc, deficit_before_refill)
        apneic_inflow, refill_l_min, n2o_inflow = self._apneic_inflow_components(
            airway_patency, effective_va, n2o_uptake_l_min, dt, remaining_flow_limit,
            deficit_l=deficit_before_refill,
        )
        washout_l = self._washout_stpd_l(
            airway_patency, effective_va, n2o_uptake_l_min, dt, deficit_l=deficit_before_refill
        )
        refill_stpd_l = refill_l_min * dt / 60.0
        # Refill re-expands the lung before exchange, so inflowing gas mixes into the restored volume.
        self._apneic_gas_deficit_l = max(0.0, deficit_before_refill - refill_stpd_l)
        gas_capacity = self.gas_volume_l * self._gas_capacity_per_l
        gas_stpd_l = self.gas_volume_l * self._btps_to_stpd
        gas_pressure = (lung_o2 + lung_co2) / gas_capacity
        avail_lung_o2 = max(
            0.0,
            lung_o2 + transport.volumes[0] * transport.oxygen[0]
            + pulmonary * (transport.oxygen[2] - transport.oxygen[0]),
        )
        want_lung_o2 = lung_production / self.rq * dt / 60.0
        lung_tissue_o2 = min(want_lung_o2, 0.9 * avail_lung_o2)
        if lung_tissue_o2 < want_lung_o2:
            self.oxygen_deficit_l += want_lung_o2 - lung_tissue_o2
        lung_tissue_co2 = lung_tissue_o2 * self.rq
        actual_lung_prod_rate = lung_tissue_co2 * 60.0 / dt if dt > 0.0 else lung_production
        transport.produced_l += lung_tissue_co2
        transport.consumed_o2_l += lung_tissue_o2
        carbon = (lung_co2 + lung_tissue_co2 + transport.volumes[0] * _CO2_L_PER_MMOL * transport.co2[0]
                  + pulmonary * _CO2_L_PER_MMOL * (transport.co2[2] - transport.co2[0]))
        oxygen = (lung_o2 - lung_tissue_o2 + transport.volumes[0] * transport.oxygen[0]
                  + pulmonary * (transport.oxygen[2] - transport.oxygen[0]))
        n2o_stpd_l_min = n2o_uptake_l_min * self._btps_to_stpd
        inflow = apneic_inflow + refill_l_min + n2o_inflow
        if washout_l <= 0.0:
            n2o_washout_o2_l = 0.0
            n2o_co2_loss_l = 0.0
        else:
            n2o_washout_o2_l = min(
                max(0.0, 0.85 * lung_o2),
                washout_l * clamp01(old_o2 / self._atm_dry),
            )
            n2o_co2_loss_l = min(max(0.0, 0.85 * lung_co2), washout_l * clamp01(old_co2 / self._atm_dry))
        lung_n2, self.alveolar_fn2, n2o_washout_n2_l = self._step_nitrogen_store(
            lung_n2,
            self.alveolar_fn2,
            target_fn2,
            effective_va,
            inflow,
            washout_l,
            gas_stpd_l,
            dt,
        )
        inflow_o2_l = inflow * fio2 * dt / 60.0 - n2o_washout_o2_l
        ventilation = effective_va * self._gas_capacity_per_l * dt / 60.0
        inspired_o2_l = ventilation * fio2 * self._atm_dry + inflow_o2_l
        explicit_washout_o2 = min(ventilation * old_o2 / 2.0, max(0.0, oxygen + inspired_o2_l))
        capillary_co2, capillary_o2 = self._equilibrate_lung(
            carbon - ventilation * old_co2 / 2.0 - n2o_co2_loss_l,
            max(0.0, oxygen + inspired_o2_l - explicit_washout_o2),
            hb, gas_capacity + ventilation / 2.0, self.aa_grad_base,
            carbon_plan, oxygen_plan, max(0.0, cardiac_output) * dt / 120.0,
        )
        transport.exhaled_l += ventilation * (old_co2 + state.p_alveolar_co2) / 2.0 + n2o_co2_loss_l
        transport.oxygen_from_air_l += (
            inspired_o2_l - explicit_washout_o2 - ventilation * state.p_alveolar_o2 / 2.0
        )
        uptake_o2, delivered_co2 = transport.finish(
            capillary_co2, capillary_o2, carbon_plan, oxygen_plan, dt, cardiac_output,
            self.shunt_fraction, production,
        )
        self._resolve_blood_pool(1, hb)
        # Venous blood renewed by cardiac output this step exchanges with extracellular fluid.
        self._exchange_extracellular(hb, transport.volumes[2] * -math.expm1(
            -max(0.0, cardiac_output) * dt / (60.0 * transport.volumes[2])))
        self._resolve_blood_pool(2, hb)
        self._publish_blood_gases()
        if dt > 0.0:
            self._pulmonary_o2_uptake = max(0.0, uptake_o2 * 60.0 / dt) + actual_lung_prod_rate / self.rq
            self._pulmonary_co2_delivery = max(0.0, delivered_co2 * 60.0 / dt) + actual_lung_prod_rate
        state.drive_central, state.muscle_factor = drive_central, muscle_factor
        state.rr, state.vt, state.mv = current_rr, current_vt, current_rr * current_vt / 1000.0
        deficit_delta: float | None = None
        if dt > 0.0 and effective_va <= 1e-4:
            loss = gas_pressure - state.p_alveolar_o2 - state.p_alveolar_co2
            agent_fraction = clamp01(1.0 - (old_o2 + old_co2) / self._atm_dry - prev_fn2)
            washout_agent_l = washout_l * agent_fraction
            deficit_delta = (
                refill_stpd_l
                + gas_capacity * loss
                + n2o_stpd_l_min * dt / 60.0
                + n2o_washout_n2_l
                + washout_agent_l
                - inflow * (1.0 - fio2) * dt / 60.0
            )
        elif dt > 0.0 and remaining_flow_limit is not None:
            want_ongoing = max(
                0.0,
                (self._pulmonary_o2_uptake - self._pulmonary_co2_delivery) + n2o_stpd_l_min,
            ) * clamp01(airway_patency / 0.05)
            unmet_ongoing = max(0.0, want_ongoing - (apneic_inflow + n2o_inflow)) * dt / 60.0
            if unmet_ongoing > 0.0:
                deficit_delta = unmet_ongoing
        if deficit_delta is not None:
            # Absorbed gas not replaced from the airway shrinks the lung; partial pressures rise with the same amounts.
            volume = self.gas_volume_l
            self._apneic_gas_deficit_l = max(0.0, self._apneic_gas_deficit_l + deficit_delta)
            mech_shrink_btps = max(0.0, vol_before_absorb - self.gas_volume_l)
            if mech_shrink_btps > 0.0:
                self._pending_mech_absorption_btps_l += mech_shrink_btps
            if self.gas_volume_l > 0.0 and self.gas_volume_l != volume:
                scale = volume / self.gas_volume_l
                state.p_alveolar_o2 *= scale
                state.p_alveolar_co2 *= scale
                self._blood_pressures[0] = state.p_alveolar_co2
                self.alveolar_fn2 = clamp01(self.alveolar_fn2 * scale)
        state.etco2 = max(0.0, state.p_alveolar_co2 - etco2_gradient)
        dilution = clamp01(1.0 - state.etco2 / state.p_alveolar_co2)
        state.eto2 = 100.0 * (state.p_alveolar_o2
                             + dilution * (fio2 * self._atm_dry - state.p_alveolar_o2)) / self._atm_dry
        return state

    def _step_hypoxic_control(
        self, dt: float, ce_prop: float, mac_sevo: float, mac_n2o: float = 0.0
    ) -> float:
        """Hypoxic drive calibrated to Nieuwenhuijs 2000, Sarton 1996, and Knill and Gelb 1978.

        Timing and the surgical propofol cutoff are simulator approximations.
        """
        if dt == 0.0:
            return self._hypoxic_drive
        pressure = self.state.p_arterial_o2
        # Carotid bodies sense PO2, not Hb content or temperature-dependent SaO2.
        deficit = max(0.0, 98.0 - 100.0 * self._saturation_at_p50(pressure, 26.8))
        deficit *= clamp01((85.0 - pressure) / 25.0)
        # Combined with baseline depression, halves the response at Ce_prop 0.6
        # and abolishes it at surgical volatile or propofol anesthesia (Knill and Gelb 1978).
        prop_sedation_keep = 1.0 / (1.0 + max(0.0, ce_prop) / 1.1)
        prop_keep = prop_sedation_keep * clamp01((4.2 - ce_prop) / 1.7)
        sevo_keep = clamp01((0.95 - mac_sevo) / 0.45) / (1.0 + max(0.0, mac_sevo) / 0.23)
        n2o_keep = 1.0 / (1.0 + max(0.0, mac_n2o) / 0.50)
        if pressure < 60.0:
            previous = self._hypoxic_exposure_s
            self._hypoxic_exposure_s += dt
            decline_dt = max(0.0, self._hypoxic_exposure_s - max(previous, self.HYPOXIC_DECLINE_DELAY_S))
            target = 0.35 + 0.55 * (1.0 - prop_sedation_keep)
            self._hypoxic_decline += (target - self._hypoxic_decline) * (
                -math.expm1(-decline_dt / self.HYPOXIC_DECLINE_TAU_S)
            )
        else:
            self._hypoxic_exposure_s = 0.0
            self._hypoxic_decline *= math.exp(-dt / self.HYPOXIC_RECOVERY_TAU_S)
        target = self.hypoxic_gain * deficit * prop_keep * sevo_keep * n2o_keep * (1.0 - self._hypoxic_decline)
        self._hypoxic_drive += (target - self._hypoxic_drive) * -math.expm1(-dt / self.HYPOXIC_TAU_S)
        return self._hypoxic_drive

    def initialize_drug_effects(self, ce_prop: float, ce_remi: float) -> None:
        """Seed the signed ventilatory controller for a maintenance history."""
        remi_c50 = self.c50_remi * (1.0 - 0.32 * ce_prop / (1.0 + ce_prop))
        self._ventilatory_drive = 1.0 - 0.5 * ce_remi / remi_c50

    def _solve_resting_pco2(
        self, prop_keep: float, sevo_keep: float, muscle_factor: float,
        unconscious: float, gain: float, opioid_shift: float, metabolic_factor: float,
        opioid_controller_fraction: float = 1.0, opioid_rate_fraction: float = 1.0,
        sevo_rate: float = 1.0,
    ) -> float:
        """Sensed alveolar PCO2 at which unassisted breathing removes the CO2 produced.

        Sensed PCO2 includes metabolic compensation, as in _unassisted. The
        opioid controller is taken at equilibrium. Without drugs this is the set point.
        """
        produced = self.co2_constant * self.vco2 * metabolic_factor

        def removed(sensed: float) -> float:
            pco2 = sensed / self._metabolic_compensation
            reference = 1.0 + gain * max(0.0, sensed - self.paco2_setpoint)
            rr, vt, _ = self._unassisted(
                pco2, prop_keep, sevo_keep, muscle_factor, unconscious, reference,
                controller=reference * opioid_controller_fraction - opioid_shift,
                opioid_controller_fraction=opioid_controller_fraction,
                opioid_rate_fraction=opioid_rate_fraction,
                sevo_rate=sevo_rate,
            )
            return rr * max(0.0, vt / 1000.0 - self.vd_deadspace) * pco2

        low, high = self.paco2_setpoint, self.RESTING_PCO2_MAX
        if removed(low) >= produced:
            return low
        if removed(high) < produced:
            return high
        for _ in range(20):
            middle = 0.5 * (low + high)
            if removed(middle) < produced:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def _unassisted(
        self, pco2: float, prop_keep: float, sevo_keep: float, muscle_factor: float,
        unconscious: float, opioid_reference: float, controller: float | None = None,
        resting_pco2: float | None = None, hypoxic_drive: float = 0.0,
        opioid_controller_fraction: float = 1.0, opioid_rate_fraction: float = 1.0,
        sevo_rate: float = 1.0,
    ) -> tuple[float, float, float]:
        """Convert normalized alveolar demand to a rate and tidal volume.

        Morphine and hydromorphone have separate rate and controller curves.
        Their combination with CO2 feedback and other drugs is a simulator
        assumption, as is the awake hypocapnic response.
        """
        pco2 *= self._metabolic_compensation
        controller = self._ventilatory_drive if controller is None else controller
        # The controller sets alveolar demand. The rate controller adds back the
        # morphine and hydromorphone share, which their own rate curve applies,
        # so it is not counted twice; remifentanil's rate effect is unchanged.
        rate_controller = max(
            0.0, controller + opioid_reference * (1.0 - opioid_controller_fraction) + hypoxic_drive,
        )
        controller = max(0.0, controller + hypoxic_drive)
        # Transfer the opioid controller as a fraction of its drug-free response
        # at the same CO2, then apply Bouillon's nonlinear alveolar response.
        co2_exponent = self.CO2_RESPONSE_EXPONENT * self.hcvr_slope_baseline / 2.2
        if sevo_keep < 1.0:
            co2_exponent *= 0.25 + 0.75 * sevo_keep
        co2_response = (max(self.paco2_setpoint, pco2) / self.paco2_setpoint) ** co2_exponent
        response_fraction = controller / opioid_reference * co2_response
        drive = min(self.MAX_ALVEOLAR_DRIVE,
                    response_fraction * prop_keep * sevo_keep * self._metabolic_compensation)
        rate_fraction = min(1.0, rate_controller)
        if rate_controller > 1.0:
            rate_fraction += min(
                0.6 * (rate_controller / opioid_reference * co2_response - 1.0),
                (0.06 + 0.04 * unconscious)
                * (0.20 + 0.80 * sevo_keep)
                * max(0.0, pco2 - self._rate_pco2_reference - self.RATE_CO2_GAP)
                + 0.3 * hypoxic_drive,
            )
        rate_fraction *= opioid_rate_fraction * (0.6 + 0.4 * prop_keep) * sevo_rate
        rate_fraction *= math.sqrt(self._metabolic_compensation)
        if controller == 0.0 or prop_keep * sevo_keep <= 1e-4:
            rate_fraction = 0.0
        rr = min(self.RR_MAX, self.rr_0 * rate_fraction)
        # Add dead space after allocating the alveolar demand to breaths.
        # Applying drug depression to total VT would subtract dead space twice.
        vt = (self.vd_deadspace + self.va_baseline * drive / rr) * 1000.0 if rr > 0.0 else 0.0
        vt *= muscle_factor
        central_keep = math.sqrt(self._central_drive_fraction)
        rr *= central_keep
        vt *= central_keep

        # Unconscious breathing stops APNEIC_GAP below the resting PaCO2, which
        # drugs raise; awake, hypocapnia below the set point weakens effort.
        resting = self.paco2_setpoint if resting_pco2 is None else max(self.paco2_setpoint, resting_pco2)
        below_resting = max(0.0, resting - pco2)
        if below_resting > 0.0:
            kept = math.sqrt(1.0 - unconscious * clamp01(below_resting / self.APNEIC_GAP))
            rr *= kept
            vt *= kept
        below = max(0.0, self.paco2_setpoint - pco2)
        if below > 0.0:
            awake_kept = 1.0 / (1.0 + self.AWAKE_HYPOCAPNIC_GAIN * self.hcvr_slope_baseline * below / self.va_baseline)
            vt *= unconscious + (1.0 - unconscious) * awake_kept
        if rr < RR_APNEA_THRESHOLD or vt < max(100.0, VT_MIN):
            rr = vt = 0.0
        return rr, vt, drive

    @staticmethod
    def _temperature_pco2_scale(temp_c: float) -> float:
        """Anaerobic Rosenthal/Bradley PCO2 ratio PCO2(temp_c) / PCO2(37 °C)."""
        return 10.0 ** (-0.019 * (37.0 - temp_c))

    @staticmethod
    def _temperature_o2_solubility(temp_c: float) -> float:
        """Dissolved O2 solubility in blood (STPD L / L / mmHg) at temp_c."""
        return _O2_SOLUBILITY * max(0.5, 1.0 + 0.013 * (37.0 - temp_c))

    @staticmethod
    def _temperature_p50(temp_c: float) -> float:
        """Dash 2016 Eq. 9d; pH, PCO2, and 2,3-DPG held at reference values."""
        delta = temp_c - 37.0
        return 26.8 + delta * (1.435 + delta * (0.04163 + delta * 0.000686))

    def _acid_base_p50(self, ph: float, pco2: float) -> float:
        """Dash 2016 Eqs. 9-10 at the current temperature; its RBC pH convention, with DPG at reference.

        Dash uses pH_RBC = 0.795 pH_plasma + 1.357, distinct from Rees's
        red-cell pH relation. Combining the models is a simulator assumption.
        """
        delta_ph = 0.795 * (ph - 7.4)
        delta_co2 = pco2 - 40.0
        ph_factor = 26.8 + delta_ph * (-25.535 + delta_ph * (10.646 - 1.764 * delta_ph))
        co2_factor = 26.8 + delta_co2 * (0.1273 + 0.0001083 * delta_co2)
        return self._temperature_p50_now * ph_factor * co2_factor / 26.8**2

    @staticmethod
    def _saturation_at_p50(pao2: float, p50: float) -> float:
        """Dash 2016 Eqs. 6 and 11, including low-saturation cooperativity."""
        if pao2 <= 0.0:
            return 0.0
        exponent = 2.82 - 1.20 * 10.0**(-pao2 / 29.25)
        return 1.0 / (1.0 + (p50 / pao2)**exponent)

    @staticmethod
    def _saturation_gradient(pao2: float, p50: float) -> tuple[float, float]:
        """S and dS/dPO2 share the pressure-dependent Hill exponent."""
        if pao2 <= 0.0:
            return 0.0, 0.0
        correction = 1.20 * 10.0**(-pao2 / 29.25)
        exponent = 2.82 - correction
        exponent_derivative = math.log(10.0) * correction / 29.25
        sat = 1.0 / (1.0 + (p50 / pao2)**exponent)
        derivative = sat * (1.0 - sat) * (exponent / pao2 + exponent_derivative * math.log(pao2 / p50))
        return sat, derivative
