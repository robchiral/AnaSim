import math
from dataclasses import dataclass

from anasim.core.constants import RR_APNEA_THRESHOLD, TEMP_METABOLIC_COEFFICIENT, VT_MIN
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.patient.patient import Patient
from anasim.patient.pd.nmba import ROCURONIUM_T1_EC50

from .blood_chemistry import (
    BloodChemistry,
    Equilibrium,
    extracellular_contents,
    step_fraction,
)
from .gas_transport import GasTransport, TransportPlan
from .hemo_config import HemodynamicConfig

_HB_G_DL_PER_MMOL_L = 1.61145  # Heme monomer, 16.1145 g/mmol.
_CO2_L_PER_MMOL = 0.022414  # STPD.
_O2_PER_HB = 0.0134  # STPD L O2 per L blood per g/dL Hb (1.34 mL/g).
_O2_SOLUBILITY = 0.000031  # STPD L O2 per L blood per mmHg.


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
    ph: float = 7.4  # Blood gas values at 37 °C.
    hco3: float = 24.5  # Plasma bicarbonate, mmol/L.
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
    LOW_VQ_RATIO = 0.05  # Ventilation per unit perfusion of poorly ventilated units (MIGET 0.005-0.1)
    LOW_VQ_AIRWAY = 0.2  # Share of output at full airway V/Q mismatch; room-air PaO2 near 64 mmHg at 40
    RESTING_PCO2_MAX = 100.0  # mmHg; search ceiling for the anesthetized resting PaCO2
    RESTING_PCO2_INTERVAL_S = 1.0  # Drug effects change slowly; recompute the threshold this often
    AWAKE_HYPOCAPNIC_GAIN = 0.2  # Fraction of HCVR; nonlinear depth response, a teaching estimate
    RATE_CO2_GAP = 5.0  # mmHg above the set point before hypercapnia raises frequency
    CO2_CONTROLLER_TAU_S = 150.0  # Olofsen 2010, 2.5 min
    CO2_RESPONSE_EXPONENT = 4.37  # Bouillon 2004 nonlinear CO2 response.
    MAX_ALVEOLAR_DRIVE = 12.0  # Bound demand above the nonlinear response at PCO2 70.
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
        self._hypoxic_drive = 0.0
        self._hypoxic_exposure_s = 0.0
        self._hypoxic_decline = 0.0
        # Olofsen's low-dose propofol background reduces opioid-controller gain
        # to 0.46. Interpolation to other concentrations is a transfer assumption.
        prop_keep_at_one = 1.0 - hill_function(1.0, self.c50_prop, self.gamma_prop)
        self.prop_co2_gain_exponent = math.log(0.46) / math.log(prop_keep_at_one)

        # Sevoflurane barely changes the CO2 response at 0.1 MAC (Pandit 1999)
        # and depresses it at 1.1-1.4 MAC (Doi 1987).
        self.c50_sevo_mac = 1.1
        self.gamma_sevo = 2.0

        # Rocuronium at the central (diaphragm and larynx) effect site. These
        # muscles need about 1.8x the adductor pollicis concentration for the
        # same twitch depression (Cantineau 1994; Plaud 1995).
        self.c50_nmba = 1.8 * ROCURONIUM_T1_EC50
        self.gamma_nmba = 3.0

        # Awake hypocapnic calibration (Nieuwenhuijs 2001; Pandit 1999).
        self.hcvr_slope_baseline = 2.2
        self.paco2_setpoint = patient.pregnancy.paco2
        self.rq = 0.8

        # Resting VO2 3.6 mL/kg/min at 70 kg, scaled with weight^0.75.
        self.vco2 = 3.6 * 70.0 * (patient.weight / 70.0) ** 0.75 * self.rq * patient.pregnancy.metabolism  # mL/min
        self.frc = patient.functional_residual_capacity()
        self.shunt_fraction = 0.0
        # Gas absorbed behind a closed airway, STPD L. Opening the airway
        # passively refills this deficit (Laviola 2020).
        self._apneic_gas_deficit_l = 0.0
        self.baseline_blood_volume_ml = patient.estimate_blood_volume()

        # COPD adds dead space and, below, an A-a gradient that persist without
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
        # ventilation that holds PaCO2 at the set point.
        self.rr_0 = patient.baseline_rr
        if patient.baseline_vt is not None:
            self.vt_0 = patient.baseline_vt
        elif self.rr_0 > 0.0:
            resting_va = self.co2_constant * self.vco2 / self.paco2_setpoint
            self.vt_0 = (self.vd_deadspace + resting_va / self.rr_0) * 1000.0
        else:
            self.vt_0 = 0.0
        self.state = RespState(
            rr=self.rr_0, vt=self.vt_0, mv=self.rr_0 * self.vt_0 / 1000.0,
            effort_rr=self.rr_0, effort_vt=self.vt_0,
            p_alveolar_co2=self.paco2_setpoint, pa_co2=self.paco2_setpoint,
        )
        # A residual gradient plus perfusion of poorly ventilated units, a share
        # of cardiac output. Awake, the share rises with age as in Gunnarsson
        # 1991 and gives Crapo 1999's sea-level A-a gradient on room air.
        # Unconscious, airways also close at end-expiration after age 30 unless
        # PEEP holds them open: about 6 cmH2O at BMI 25 and 8 at BMI 30
        # (Hedenstierna 2015). Closure is calibrated to the A-a gradient's rise
        # with age at FiO2 0.4 without PEEP (Gunnarsson 1991). COPD adds low V/Q
        # units with little shunt (Wagner 1977), 12% of output at severity 1 for
        # a room-air PaO2 near 70 mmHg at age 65.
        self.aa_grad_base = 5.0
        self.low_vq_awake = 0.001 * max(0.0, patient.age - 34.0) + 0.12 * patient.copd_severity
        self.low_vq_closure = 0.0025 * max(0.0, patient.age - 30.0)
        self.airway_opening_pressure = 6.0 + 0.4 * max(0.0, patient.bmi - 25.0)
        self.low_vq_fraction = self.low_vq_awake
        self._low_vq_share = self.low_vq_awake
        self._low_vq_po2 = 40.0
        self._p50 = self._temperature_p50(patient.baseline_temp)
        self._temperature_p50_now = self._p50
        assert patient.baseline_hct is not None
        self.baseline_hct = patient.baseline_hct
        self._oxygen_hb = self.baseline_hb
        self._set_blood_composition(self.baseline_hb, self.baseline_hct)
        self.gas_transport = GasTransport(
            patient.weight, self.baseline_blood_volume_ml / 1000.0,
        )
        self._blood_pressures = [self.paco2_setpoint, self.paco2_setpoint, self.paco2_setpoint + 6.0]
        self._blood_ph = [patient.pregnancy.ph, patient.pregnancy.ph, patient.pregnancy.ph - 0.04]
        self._blood_po2 = [100.0, 95.0, 40.0]
        self._blood_saturations = [0.98, 0.98, 0.75]
        self._blood_p50 = [self._p50] * 3
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
        self.chemistry = BloodChemistry(hb_g_dl / _HB_G_DL_PER_MMOL_L, hematocrit)
        reference = self.chemistry.evaluate(self.patient.pregnancy.ph, self.paco2_setpoint, 0.98)
        self.buffer_base = self._reference_buffer_base = reference.buffer_base
        return reference

    @property
    def va_baseline(self) -> float:
        """Resting alveolar ventilation derived from baseline breath rate and volume."""
        return max(0.1, (self.vt_0 / 1000.0 - self.vd_deadspace) * self.rr_0)

    def _set_low_vq(self, unconscious: float, airway_pressure: float, vq_mismatch: float = 0.0) -> None:
        closure = clamp01(unconscious) * clamp01(1.0 - airway_pressure / self.airway_opening_pressure)
        self.low_vq_fraction = (self.low_vq_awake + closure * self.low_vq_closure
                                + self.LOW_VQ_AIRWAY * clamp01(vq_mismatch))
        self._low_vq_share = min(1.0, self.low_vq_fraction / (1.0 - self.shunt_fraction))

    def _seed_capillary_oxygen(self, extraction: float) -> tuple[float, float, float]:
        """Capillary O2 content, and the PO2 and saturation of the mixed blood, at a venous extraction."""
        hb, alveolar = self._oxygen_hb, self.state.p_alveolar_o2
        content = self._oxygen_content(max(0.0, alveolar - self.aa_grad_base), hb, self._p50)[0]
        for _ in range(50):
            previous = content
            low = self._low_vq_unit(alveolar, max(0.0, content - extraction), hb, self._p50) if self._low_vq_share else None
            content, _, po2, _ = self._capillary_oxygen(alveolar, hb, self._p50, self.aa_grad_base, low)
            if abs(content - previous) < 1e-12:
                po2 = self._po2_from_content(content, hb, self._p50, po2)
                return content, po2, self._saturation_at_p50(po2, self._p50)
        raise RuntimeError("Capillary O2 seed did not converge")

    def equilibrate_oxygen(self, fio2: float, unconscious: float = 0.0, airway_pressure: float = 0.0) -> None:
        """Seed lung and blood oxygen after an explicitly equilibrated exposure."""
        self._set_low_vq(unconscious, airway_pressure)
        self.state.p_alveolar_o2 = max(0.0, fio2 * self._atm_dry - self.state.p_alveolar_co2 / self.rq)
        hb = self._oxygen_hb
        extraction = self.vco2 / self.rq / 1000.0 / self.baseline_co_l_min
        content = self._seed_capillary_oxygen(extraction)[0]
        self.gas_transport.oxygen = [content, content, max(0.0, content - extraction)]
        for index, oxygen in enumerate(self.gas_transport.oxygen):
            po2 = self._po2_from_content(oxygen, hb, self._blood_p50[index], self._blood_po2[index])
            self._blood_po2[index] = po2
            self._blood_saturations[index] = self._saturation_at_p50(po2, self._blood_p50[index])
        self.state.p_arterial_o2 = self._blood_po2[1]
        self.state.sao2 = 100.0 * self._blood_saturations[1]
        self._apneic_gas_deficit_l = 0.0

    def equilibrate_blood_gases(self, cardiac_output: float, production_l_min: float) -> None:
        """Seed blood and tissues for a specified maintenance history."""
        transport = self.gas_transport
        hb = self._oxygen_hb
        reference = self.chemistry.evaluate(self.patient.pregnancy.ph, self.paco2_setpoint, 0.98)
        transport.base = [reference.buffer_base] * 3
        extraction = production_l_min / self.rq / cardiac_output
        oxygen, capillary, saturation = self._seed_capillary_oxygen(extraction)
        transport.oxygen = [oxygen, oxygen, max(0.0, oxygen - extraction)]
        arterial = self.chemistry.solve(self.state.p_alveolar_co2, saturation, reference.buffer_base)
        venous_co2 = arterial.total_co2 + production_l_min / _CO2_L_PER_MMOL / cardiac_output
        transport.co2 = [arterial.total_co2, arterial.total_co2, venous_co2]
        venous_po2 = self._po2_from_content(transport.oxygen[2], hb, self._p50, 40.0)
        venous_saturation = self._saturation_at_p50(venous_po2, self._p50)
        venous = self.chemistry.solve_contents(venous_co2, reference.buffer_base, venous_saturation, 46.0, 7.36)
        co2, base, _, _ = extracellular_contents(venous.blood.ph, venous.pco2)
        transport.extracellular_co2 = transport.extracellular_volume * co2
        transport.extracellular_base = transport.extracellular_volume * base
        self._extracellular_ph = venous.blood.ph
        self._extracellular_pco2 = venous.pco2
        self._blood_pressures[0], self._blood_ph[0] = self.state.p_alveolar_co2, arterial.ph
        self._blood_po2[0], self._blood_saturations[0] = capillary, saturation
        self._resolve_blood_pool(1, hb)
        self._resolve_blood_pool(2, hb)
        transport.seed_tissues(arterial.total_co2, cardiac_output, production_l_min,
                               self.chemistry, self._blood_saturations[2])
        self._pulmonary_o2_uptake = production_l_min / self.rq
        self._pulmonary_co2_delivery = production_l_min
        self._publish_blood_gases()

    @property
    def gas_volume_l(self) -> float:
        """Lung gas volume, L BTPS.

        Absorption behind a closed airway shrinks the lung at atmospheric
        pressure until only the conducting airways hold gas; further absorption
        lowers alveolar pressure.
        """
        return max(self.vd_deadspace, self.frc - self._apneic_gas_deficit_l / self._btps_to_stpd)

    def _passive_refill_l_min(self, airway_patency: float, dt: float) -> float:
        """Mean passive inflow this interval, limited to the missing lung gas.

        The 0.5 s refill time is a simulator estimate. Patency slows refill;
        no pressure or lung-volume transient is represented.
        """
        if dt <= 0.0:
            return 0.0
        fraction = -math.expm1(-clamp01(airway_patency) * dt / 0.5)
        return self._apneic_gas_deficit_l * fraction * 60.0 / dt

    @classmethod
    def _oxygen_content(cls, pressure: float, hb: float, p50: float) -> tuple[float, float]:
        """STPD L O2 per L blood and derivative per mmHg."""
        saturation, derivative = cls._saturation_gradient(pressure, p50)
        return _O2_PER_HB * hb * saturation + _O2_SOLUBILITY * pressure, _O2_PER_HB * hb * derivative + _O2_SOLUBILITY

    def _low_vq_unit(self, alveolar_po2: float, venous_o2: float, hb: float,
                     p50: float) -> tuple[float, float, float, float, float]:
        """Alveolar PO2, and low V/Q end-capillary PO2 and saturation with their slopes per mmHg of it.

        The units draw alveolar gas, and absorbed O2 draws in more:
        C(P) - Cv = k (PA - P) / (1 - PA / Pdry), with k from LOW_VQ_RATIO.
        """
        exchange = self.LOW_VQ_RATIO * self._gas_capacity_per_l * self._atm_dry / max(
            0.01 * self._atm_dry, self._atm_dry - alveolar_po2)
        target = venous_o2 + exchange * alveolar_po2
        bound_capacity = _O2_PER_HB * hb
        lower, upper = 0.0, 800.0
        pressure = min(max(self._low_vq_po2, 1e-6), upper)
        for _ in range(40):
            saturation, slope = self._saturation_gradient(pressure, p50)
            residual = bound_capacity * saturation + (_O2_SOLUBILITY + exchange) * pressure - target
            derivative = bound_capacity * slope + _O2_SOLUBILITY + exchange
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
        self._low_vq_po2 = pressure
        if alveolar_po2 < 0.99 * self._atm_dry:
            exchange *= (self._atm_dry - pressure) / (self._atm_dry - alveolar_po2)
        rise = exchange / derivative
        return alveolar_po2, pressure, saturation, rise, slope * rise

    def _capillary_oxygen(self, alveolar_po2: float, hb: float, p50: float, gradient: float,
                          low: tuple[float, float, float, float, float] | None) -> tuple[float, float, float, float]:
        """Mixed end-capillary O2 content, its slope per mmHg of alveolar PO2, PO2, and saturation.

        Low V/Q units follow a _low_vq_unit solution, linear in alveolar PO2. The PO2 weights
        each stream's dissolved O2, so it and the saturation reproduce the content.
        """
        po2 = max(0.0, alveolar_po2 - gradient)
        saturation, slope = self._saturation_gradient(po2, p50)
        slope = _O2_PER_HB * hb * slope + _O2_SOLUBILITY if po2 > 0.0 else 0.0
        if low is not None:
            share = self._low_vq_share
            reference, low_po2, low_saturation, po2_slope, saturation_slope = low
            shift = alveolar_po2 - reference
            po2 += share * (low_po2 + po2_slope * shift - po2)
            saturation += share * (low_saturation + saturation_slope * shift - saturation)
            slope += share * (_O2_PER_HB * hb * saturation_slope + _O2_SOLUBILITY * po2_slope - slope)
        return _O2_PER_HB * hb * saturation + _O2_SOLUBILITY * po2, slope, po2, saturation

    @classmethod
    def _po2_from_content(cls, content: float, hb: float, p50: float, previous: float) -> float:
        if content <= 0.0:
            return 0.0
        bound_capacity = _O2_PER_HB * hb
        exponent_decay = math.log(10.0) / 29.25
        lower, upper = 0.0, max(760.0, content / _O2_SOLUBILITY)
        pressure = max(1e-6, min(upper, previous))
        for _ in range(24):
            # Inlined _saturation_gradient; the derivative is needed only for another step.
            correction = 1.20 * 10.0**(-pressure / 29.25)
            exponent = 2.82 - correction
            saturation = 1.0 / (1.0 + (p50 / pressure)**exponent)
            value = bound_capacity * saturation + _O2_SOLUBILITY * pressure
            residual = value - content
            if abs(residual) < 1e-12:
                return pressure
            saturation_derivative = saturation * (1.0 - saturation) * (
                exponent / pressure + exponent_decay * correction * math.log(pressure / p50))
            derivative = bound_capacity * saturation_derivative + _O2_SOLUBILITY
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
        po2 = self._po2_from_content(transport.oxygen[index], hb, p50, self._blood_po2[index])
        saturation = self._saturation_at_p50(po2, p50)
        # Alternate single CO2 and O2 updates; saturation barely moves once O2 content is fixed.
        for iteration in range(24):
            iterate = self.chemistry.iterate_contents(carbon, base, saturation, pco2, ph)
            if iterate.converged and iteration:
                break  # Saturation was last updated at this pH and PCO2.
            ph, pco2 = iterate.ph, iterate.pco2
            p50 = self._acid_base_p50(ph, pco2)
            po2 = self._po2_from_content(transport.oxygen[index], hb, p50, po2)
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
        po2 = self._po2_from_content(transport.oxygen[2], hb, p50, self._blood_po2[2])
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
        low = self._low_vq_unit(po2, transport.oxygen[2], hb, p50) if self._low_vq_share else None
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
            po2 = max(0.0, po2 - o2_change)
            # Newton converges quadratically, so the step after a 1e-6 mmHg step is negligible.
            if abs(change) < 1e-6 and abs(o2_change) < 1e-6 and abs(base_residual) < 1e-8:
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
        state.pa_co2, state.ph = self._blood_pressures[1], self._blood_ph[1]
        state.p_arterial_o2 = self._blood_po2[1]
        state.sao2 = 100.0 * self._blood_saturations[1]
        state.pv_co2, state.pv_o2, state.venous_ph = self._blood_pressures[2], self._blood_po2[2], self._blood_ph[2]
        self._p50 = self._blood_p50[1]
        self.buffer_base = self.gas_transport.base[1]

    def oxygen_exchange_l_min(
        self, fio2: float, alveolar_ventilation: float,
        airway_patency: float, vq_mismatch: float, dt: float = 1.0,
    ) -> float:
        """Net O2 transport from inspired gas into the lung and blood stores, STPD L/min."""
        if airway_patency <= 0.0:
            return 0.0
        ventilation = max(0.0, alveolar_ventilation * (1.0 - 0.6 * clamp01(vq_mismatch)))
        net_uptake = max(0.0, self._pulmonary_o2_uptake - self._pulmonary_co2_delivery)
        apneic_inflow = net_uptake * clamp01(airway_patency) * clamp01(1.0 - ventilation)
        initial_flux = (
            ventilation * self._btps_to_stpd
            * (fio2 - self.state.p_alveolar_o2 / self._atm_dry)
            + (apneic_inflow + self._passive_refill_l_min(airway_patency, dt)) * fio2
        )
        # Predict the interval mean gas pressure for the circuit's O2 balance.
        # The respiratory solve uses the same midpoint ventilation flux.
        mixing = ventilation * dt / (120.0 * self.gas_volume_l)
        return (initial_flux + mixing * self._pulmonary_o2_uptake) / (1.0 + mixing)

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
             opioid_rate_fraction: float = 1.0, airway_pressure: float = 0.0) -> RespState:
        """Advance respiration by dt seconds.

        Args:
            ce_prop: Propofol effect-site concentration (mcg/mL).
            ce_remi: Remifentanil effect-site concentration (ng/mL).
            mech_vent_mv: Assisted minute ventilation reaching the lungs (L/min).
            fio2: Inspired O2 fraction.
            ce_roc: Free rocuronium at the central effect site (mcg/mL).
            mac_sevo: Brain sevoflurane MAC.
            mech_rr, mech_vt_l: Assisted rate (breaths/min) and tidal volume
                reaching the lungs (L).
            airway_patency: Upper-airway patency, 0-1.
            ventilation_efficiency: Lower-airway efficiency (bronchospasm), 0-1.
            vq_mismatch: V/Q mismatch severity, 0-1.
            hb_g_dl: Hemoglobin (g/dL), which sets the blood O2 store.
            hematocrit: Red-cell volume fraction, supplied independently of Hb.
            cardiac_output: Pulmonary and systemic blood flow, L/min.
            metabolic_factor: VO2 and VCO2 multiplier.
            blood_volume_ml: Blood volume (mL), which with Hb sets the blood O2 store.
            measured_breaths: The supplied rate and volume include all observed
                breaths, including spontaneous breaths during mechanical ventilation.
            unconscious: Probability of loss of consciousness, 0-1.
            lung_volume_l: Measured end-expiratory gas volume, L.
            shunt_fraction: Perfusion through closed lung units, 0-1.
            temp_c: Blood temperature for hemoglobin O2 affinity, degrees C.
            opioid_controller_fraction: Fraction of baseline ventilation and CO2 gain remaining.
            neuromuscular_strength: Muscle strength remaining after depolarizing block, 0-1.
            opioid_rate_fraction: Fraction of baseline RR from morphine and hydromorphone.
            airway_pressure: Applied end-expiratory pressure, cmH2O.
        """
        transport = self.gas_transport
        blood_volume = self.baseline_blood_volume_ml if blood_volume_ml is None else blood_volume_ml
        transport.resize_blood(blood_volume / 1000.0)
        max_step = 1.0
        if cardiac_output > 0.0:
            vp, va, vv = transport.volumes
            # Crank-Nicolson advection preserves nonnegative contents when
            # no pool exports more than its volume in the explicit half-step.
            transit_volume = min(vp / (1.0 - clamp(shunt_fraction, 0.0, 0.95)), va, vv)
            max_step = min(max_step, 120.0 * transit_volume / cardiac_output)
        if dt > max_step:
            count = math.ceil(dt / max_step)
            for _ in range(count):
                self.step(
                    dt / count, ce_prop, ce_remi, mech_vent_mv, fio2, ce_roc, mac_sevo,
                    mech_rr, mech_vt_l, airway_patency, ventilation_efficiency, vq_mismatch,
                    hb_g_dl, hematocrit, cardiac_output, metabolic_factor, blood_volume_ml,
                    measured_breaths, unconscious, lung_volume_l, shunt_fraction, temp_c,
                    opioid_controller_fraction, neuromuscular_strength,
                    opioid_rate_fraction, airway_pressure,
                )
            return self.state
        state = self.state
        hill = hill_function
        clamp01_local = clamp01

        prop_keep = 1.0 - hill(ce_prop, self.c50_prop, self.gamma_prop)
        sevo_keep = 1.0 - hill(mac_sevo, self.c50_sevo_mac, self.gamma_sevo)
        muscle_factor = 1.0 - hill(ce_roc, self.c50_nmba, self.gamma_nmba)
        muscle_factor *= clamp01_local(neuromuscular_strength)
        airway_patency = clamp01_local(airway_patency)
        ventilation_efficiency = clamp01_local(ventilation_efficiency)
        unconscious = clamp01_local(unconscious)

        # At Ce_prop=1 the opioid C50 is 0.84 * 1.6 ng/mL (Olofsen).
        # Interpolation to other propofol concentrations is a teaching assumption.
        remi_c50 = self.c50_remi * (1.0 - 0.32 * max(0.0, ce_prop) / (1.0 + max(0.0, ce_prop)))
        baseline_gain = self.hcvr_slope_baseline / 2.2 * (0.42 / 7.2)
        gain = baseline_gain * prop_keep ** (self.prop_co2_gain_exponent - 1.0) * sevo_keep
        co2_excess = max(0.0, state.pa_co2 - self.paco2_setpoint)
        opioid_reference = 1.0 + gain * co2_excess
        long_opioid_keep = clamp01_local(opioid_controller_fraction)
        long_opioid_rate = clamp01_local(opioid_rate_fraction)
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
                    0.0, 1.0, long_opioid_keep, long_opioid_rate,
                )
            self._resting_pco2 = self._solve_resting_pco2(
                prop_keep, sevo_keep, muscle_factor, unconscious, gain,
                0.5 * max(0.0, ce_remi) / remi_c50, max(0.1, float(metabolic_factor)),
                long_opioid_keep, long_opioid_rate,
            )
            self._resting_age_s = 0.0
        effort_rr, effort_vt, drive_central = self._unassisted(
            state.pa_co2, prop_keep, sevo_keep, muscle_factor,
            unconscious, opioid_reference, resting_pco2=self._resting_pco2,
            hypoxic_drive=opioid_reference * self._step_hypoxic_control(dt, ce_prop, mac_sevo),
            opioid_controller_fraction=long_opioid_keep, opioid_rate_fraction=long_opioid_rate,
        )
        state.effort_rr, state.effort_vt = effort_rr, effort_vt
        current_rr = effort_rr
        current_vt = effort_vt * airway_patency * ventilation_efficiency
        if current_vt < max(100.0, VT_MIN):
            current_rr = current_vt = 0.0

        state.apnea = current_rr <= 0.0

        vd = self.vd_deadspace
        vt_eff_spont = max(0.0, current_vt / 1000.0 - vd)
        # Assisted volumes arrive net of any mask leak; bronchospasm still
        # lowers the alveolar share.
        mech_vt_l *= ventilation_efficiency
        mech_vent_mv *= ventilation_efficiency
        alveolar_vt_mech = max(0.0, mech_vt_l - vd)
        if mech_vt_l <= 0 and mech_vent_mv > 0 and mech_rr > 0:
            inferred_vt = mech_vent_mv / mech_rr
            alveolar_vt_mech = max(0.0, inferred_vt - vd)

        # Assisted breaths augment or replace spontaneous ones, so each breath
        # gets the larger of the two volumes at the higher of the two rates.
        ref_rr_mech = max(0.0, mech_rr)
        ref_rr_spont = max(0.0, current_rr)
        effective_rate = max(ref_rr_mech, ref_rr_spont)
        vt_mech_avail = max(0.0, alveolar_vt_mech)
        vt_spont_avail = max(0.0, vt_eff_spont)

        if measured_breaths:
            # Measured ventilation already contains the patient's contribution.
            # Combining its VT with a separate neural rate invents extra gas.
            total_va_l_min = ref_rr_mech * vt_mech_avail
        else:
            total_va_l_min = effective_rate * (max(vt_mech_avail, vt_spont_avail) if mech_rr > 0 else vt_spont_avail)
        if airway_patency <= 0.0:
            total_va_l_min = 0.0
        state.va = total_va_l_min

        # V/Q mismatch reduces effective CO2 elimination.
        vq_mismatch = clamp01_local(vq_mismatch)
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
        vq_for_etco2 = clamp01(vq_mismatch)
        etco2_gradient = 4.0 + 15.0 * vd_vt_excess + 8.0 * vq_for_etco2 + 6.0 * (1.0 - ventilation_efficiency)
        etco2_gradient = min(20.0, etco2_gradient)
        hb = self.baseline_hb if hb_g_dl is None else hb_g_dl
        hct = self.baseline_hct if hematocrit is None else hematocrit
        if hb != self._oxygen_hb:
            # Each pool keeps its PO2 and saturation as Hb changes.
            transport.oxygen = [_O2_PER_HB * hb * saturation + _O2_SOLUBILITY * po2 for po2, saturation in zip(
                self._blood_po2, self._blood_saturations, strict=True)]
            self._oxygen_hb = hb
        old_hb, old_hct = self._chemistry_composition
        # Fluid kinetics move Hb slightly every step; rebuild buffering for measurable changes.
        if abs(hb - old_hb) > 1e-6 * old_hb or abs(hct - old_hct) > 1e-6 * old_hct:
            previous_reference = self._reference_buffer_base
            reference = self._set_blood_composition(hb, hct)
            shift = reference.buffer_base - previous_reference
            transport.base = [base + shift for base in transport.base]
        self._temperature_p50_now = self._temperature_p50(temp_c)
        self.shunt_fraction = clamp(shunt_fraction, 0.0, 0.95)
        self._set_low_vq(unconscious, airway_pressure, vq_mismatch)
        production = self.vco2 * metabolic_factor / 1000.0
        consumption = production / self.rq
        carbon_plan, oxygen_plan, pulmonary = transport.prepare(
            dt, cardiac_output, self.shunt_fraction, production, consumption, self.chemistry,
            self._blood_saturations[2],
        )
        gas_capacity = self.gas_volume_l * self._gas_capacity_per_l
        lung_co2 = gas_capacity * state.p_alveolar_co2
        lung_o2 = gas_capacity * state.p_alveolar_o2
        if lung_volume_l is not None:
            volume_change = lung_volume_l - self.frc
            gas_o2 = fio2 * self._atm_dry if volume_change > 0.0 else state.p_alveolar_o2
            lung_o2 += volume_change * self._gas_capacity_per_l * gas_o2
            transport.oxygen_from_air_l += volume_change * self._gas_capacity_per_l * gas_o2
            if volume_change < 0.0:
                lost_co2 = -volume_change * self._gas_capacity_per_l * state.p_alveolar_co2
                lung_co2 -= lost_co2
                transport.exhaled_l += lost_co2
            self.frc = lung_volume_l
        old_co2, old_o2 = state.p_alveolar_co2, state.p_alveolar_o2
        refill_l_min = self._passive_refill_l_min(airway_patency, dt)
        # Refill re-expands the lung before exchange, so inflowing gas mixes into the restored volume.
        self._apneic_gas_deficit_l = max(0.0, self._apneic_gas_deficit_l - refill_l_min * dt / 60.0)
        gas_capacity = self.gas_volume_l * self._gas_capacity_per_l
        gas_pressure = (lung_o2 + lung_co2) / gas_capacity
        carbon = (lung_co2 + transport.volumes[0] * _CO2_L_PER_MMOL * transport.co2[0]
                  + pulmonary * _CO2_L_PER_MMOL * (transport.co2[2] - transport.co2[0]))
        oxygen = (lung_o2 + transport.volumes[0] * transport.oxygen[0]
                  + pulmonary * (transport.oxygen[2] - transport.oxygen[0]))
        net_uptake = max(0.0, self._pulmonary_o2_uptake - self._pulmonary_co2_delivery)
        inflow = (net_uptake * airway_patency * clamp01_local(1.0 - effective_va) + refill_l_min)
        ventilation = effective_va * self._gas_capacity_per_l * dt / 60.0
        capillary_co2, capillary_o2 = self._equilibrate_lung(
            carbon - ventilation * old_co2 / 2.0,
            oxygen + ventilation * (fio2 * self._atm_dry - old_o2 / 2.0) + inflow * fio2 * dt / 60.0,
            hb, gas_capacity + ventilation / 2.0, self.aa_grad_base,
            carbon_plan, oxygen_plan, max(0.0, cardiac_output) * dt / 120.0,
        )
        transport.exhaled_l += ventilation * (old_co2 + state.p_alveolar_co2) / 2.0
        transport.oxygen_from_air_l += (
            ventilation * (fio2 * self._atm_dry - (old_o2 + state.p_alveolar_o2) / 2.0)
            + inflow * fio2 * dt / 60.0
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
            self._pulmonary_o2_uptake = max(0.0, uptake_o2 * 60.0 / dt)
            self._pulmonary_co2_delivery = max(0.0, delivered_co2 * 60.0 / dt)
        state.drive_central, state.muscle_factor = drive_central, muscle_factor
        state.rr, state.vt, state.mv = current_rr, current_vt, current_rr * current_vt / 1000.0
        if airway_patency <= 0.0:
            # Absorbed gas shrinks the sealed lung; partial pressures rise with the same amounts.
            volume = self.gas_volume_l
            loss = gas_pressure - state.p_alveolar_o2 - state.p_alveolar_co2
            self._apneic_gas_deficit_l = max(0.0, self._apneic_gas_deficit_l + gas_capacity * loss)
            scale = volume / self.gas_volume_l
            state.p_alveolar_o2 *= scale
            state.p_alveolar_co2 *= scale
        state.etco2 = max(0.0, state.p_alveolar_co2 - etco2_gradient)
        dilution = clamp01_local(1.0 - state.etco2 / state.p_alveolar_co2)
        state.eto2 = 100.0 * (state.p_alveolar_o2
                             + dilution * (fio2 * self._atm_dry - state.p_alveolar_o2)) / self._atm_dry
        return state

    def _step_hypoxic_control(self, dt: float, ce_prop: float, mac_sevo: float) -> float:
        """Hypoxic drive calibrated to Nieuwenhuijs 2000 and Sarton 1996.

        Timing and extrapolation beyond low-dose volunteer data are assumptions.
        """
        if dt == 0.0:
            return self._hypoxic_drive
        pressure = self.state.p_arterial_o2
        # Carotid bodies sense PO2, not Hb content or temperature-dependent SaO2.
        deficit = max(0.0, 98.0 - 100.0 * self._saturation_at_p50(pressure, 26.8))
        deficit *= clamp01((85.0 - pressure) / 25.0)
        # Combined with baseline depression, halves the response at Ce_prop 0.6.
        prop_keep = 1.0 / (1.0 + max(0.0, ce_prop) / 1.1)
        sevo_keep = 1.0 / (1.0 + max(0.0, mac_sevo) / 0.23)
        if pressure < 60.0:
            previous = self._hypoxic_exposure_s
            self._hypoxic_exposure_s += dt
            decline_dt = max(0.0, self._hypoxic_exposure_s - max(previous, self.HYPOXIC_DECLINE_DELAY_S))
            target = 0.35 + 0.55 * (1.0 - prop_keep)
            self._hypoxic_decline += (target - self._hypoxic_decline) * (
                -math.expm1(-decline_dt / self.HYPOXIC_DECLINE_TAU_S)
            )
        else:
            self._hypoxic_exposure_s = 0.0
            self._hypoxic_decline *= math.exp(-dt / self.HYPOXIC_RECOVERY_TAU_S)
        target = self.HYPOXIC_GAIN * deficit * prop_keep * sevo_keep * (1.0 - self._hypoxic_decline)
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
    ) -> float:
        """Alveolar PCO2 at which unassisted breathing removes the CO2 produced.

        The opioid controller is taken at equilibrium. Without drugs this is the set point.
        """
        produced = self.co2_constant * self.vco2 * metabolic_factor

        def removed(pco2: float) -> float:
            reference = 1.0 + gain * max(0.0, pco2 - self.paco2_setpoint)
            rr, vt, _ = self._unassisted(
                pco2, prop_keep, sevo_keep, muscle_factor, unconscious, reference,
                controller=reference * opioid_controller_fraction - opioid_shift,
                opioid_controller_fraction=opioid_controller_fraction,
                opioid_rate_fraction=opioid_rate_fraction,
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
    ) -> tuple[float, float, float]:
        """Convert normalized alveolar demand to a rate and tidal volume.

        Morphine and hydromorphone have separate rate and controller curves.
        Their combination with CO2 feedback and other drugs is a simulator
        assumption, as is the awake hypocapnic response.
        """
        controller = self._ventilatory_drive if controller is None else controller
        # The controller sets alveolar demand. Rate adds back the long-opioid
        # share, which the separate rate curve applies, so it is not applied
        # twice and remifentanil's rate effect is unchanged.
        rate_controller = max(
            0.0, controller + opioid_reference * (1.0 - opioid_controller_fraction) + hypoxic_drive,
        )
        controller = max(0.0, controller + hypoxic_drive)
        # Transfer the opioid controller as a fraction of its drug-free response
        # at the same CO2, then apply Bouillon's nonlinear alveolar response.
        co2_exponent = self.CO2_RESPONSE_EXPONENT * self.hcvr_slope_baseline / 2.2
        co2_response = (max(self.paco2_setpoint, pco2) / self.paco2_setpoint) ** co2_exponent
        response_fraction = controller / opioid_reference * co2_response
        drive = min(self.MAX_ALVEOLAR_DRIVE, response_fraction * prop_keep * sevo_keep)
        rate_fraction = min(1.0, rate_controller)
        if rate_controller > 1.0:
            rate_fraction += min(
                0.6 * (rate_controller / opioid_reference * co2_response - 1.0),
                (0.06 + 0.04 * unconscious) * max(0.0, pco2 - self._rate_pco2_reference - self.RATE_CO2_GAP)
                + 0.3 * hypoxic_drive,
            )
        rate_fraction *= opioid_rate_fraction * (0.6 + 0.4 * prop_keep) * (0.6 + 0.4 * sevo_keep)
        if controller == 0.0:
            rate_fraction = 0.0
        rr = self.rr_0 * rate_fraction
        # Add dead space after allocating the alveolar demand to breaths.
        # Applying drug depression to total VT would subtract dead space twice.
        vt = (self.vd_deadspace + self.va_baseline * drive / rr) * 1000.0 if rr > 0.0 else 0.0
        vt *= muscle_factor

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
