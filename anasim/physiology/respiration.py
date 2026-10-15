import math
from dataclasses import dataclass

from anasim.core.constants import (
    APNEA_PACO2_RISE_FAST_DURATION_SEC,
    APNEA_PACO2_RISE_FAST_MMHG_MIN,
    APNEA_PACO2_RISE_SLOW_MMHG_MIN,
    RR_APNEA_THRESHOLD,
    RR_BRADYPNEA_THRESHOLD,
    VT_MIN,
)
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.patient.patient import Patient
from anasim.patient.pd.nmba import ROCURONIUM_T1_EC50

from .hemo_config import HemodynamicConfig


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
    drive_central: float = 1.0
    muscle_factor: float = 1.0
    effort_rr: float = 12.0
    effort_vt: float = 500.0


class RespiratoryModel:
    """Ventilatory control, anesthetic respiratory depression, and CO2 and O2 exchange."""

    # Under anesthesia the apneic threshold is 4-5 mmHg below resting PaCO2,
    # independent of agent and depth (Hickey 1971).
    APNEIC_GAP = 4.5  # mmHg
    VQ_MISMATCH_GRADIENT = 35.0  # mmHg added to the A-a gradient at full airway V/Q mismatch
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
        self.baseline_co_l_min = HemodynamicConfig().cardiac_index * patient.bsa

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
        self.paco2_setpoint = 40.0
        self.rq = 0.8

        # Resting VO2 3.6 mL/kg/min at 70 kg, scaled with weight^0.75.
        self.vco2 = 3.6 * 70.0 * (patient.weight / 70.0) ** 0.75 * self.rq  # mL/min
        self.frc = patient.functional_residual_capacity()
        self.shunt_fraction = 0.0
        # Gas removed from the sealed lung during obstruction, STPD L. Opening
        # the airway passively refills this deficit (Laviola 2020).
        self._apneic_gas_deficit_l = 0.0
        self.baseline_blood_volume_ml = patient.estimate_blood_volume()

        self.vd_deadspace = 2.2 * patient.predicted_body_weight() / 1000.0  # L

        # Body CO2 stores equilibrate slowly (apneic rise 3-5 mmHg/min).
        self.tau_co2 = 180.0  # s
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
        )
        # Mean sea-level A-a gradient, -4.4 + 0.261 x age mmHg (Crapo 1999).
        self.aa_grad_base = max(5.0, -4.404 + 0.261 * self.patient.age)
        self.equilibrate_oxygen(0.21)
        self._blood_temp_c = patient.baseline_temp
        self._p50 = self._temperature_p50(patient.baseline_temp)
        self._apnea_timer = 0.0
        self._resting_pco2: float | None = None
        self._resting_age_s = 0.0
        # Perfusion effect on deadspace fraction (low flow increases VD/VT).
        self.perfusion_deadspace_gain = 0.25

    @property
    def va_baseline(self) -> float:
        """Resting alveolar ventilation derived from baseline breath rate and volume."""
        return max(0.1, (self.vt_0 / 1000.0 - self.vd_deadspace) * self.rr_0)

    def equilibrate_oxygen(self, fio2: float) -> None:
        """Set alveolar O2 to its alveolar-gas-equation value at the current PACO2."""
        self.state.p_alveolar_o2 = max(0.0, fio2 * self._atm_dry - self.state.p_alveolar_co2 / self.rq)
        self.state.p_arterial_o2 = max(0.0, self.state.p_alveolar_o2 - self.aa_grad_base)
        self._apneic_gas_deficit_l = 0.0

    def _passive_refill_l_min(self, airway_patency: float, dt: float) -> float:
        """Mean passive inflow this interval, limited to the missing lung gas.

        The 0.5 s refill time is a simulator estimate. Patency slows refill;
        no pressure or lung-volume transient is represented.
        """
        if dt <= 0.0:
            return 0.0
        fraction = -math.expm1(-clamp01(airway_patency) * dt / 0.5)
        return self._apneic_gas_deficit_l * fraction * 60.0 / dt

    def _add_oxygen_store(
        self, oxygen_l: float, hb: float, blood_volume_ml: float,
        cardiac_output: float, vo2_l_min: float, vq_mismatch: float,
    ) -> None:
        """Add refill O2 by solving the nonlinear lung and blood content balance.

        A fixed saturation derivative loses O2 during the rapid transition
        between severe hypoxemia and the flat part of the dissociation curve.
        """
        if oxygen_l <= 1e-9:
            return
        target = self._oxygen_store(
            self.state.p_alveolar_o2, hb, blood_volume_ml, cardiac_output, vo2_l_min, vq_mismatch,
        )[0] + oxygen_l
        self.state.p_alveolar_o2 = self._pressure_from_oxygen_store(
            target, hb, blood_volume_ml, cardiac_output, vo2_l_min, vq_mismatch,
        )

    def _oxygen_store(
        self, pressure: float, hb: float, blood_volume_ml: float,
        cardiac_output: float, vo2_l_min: float, vq_mismatch: float,
    ) -> tuple[float, float]:
        """O2 store (STPD L) and its derivative with respect to alveolar PO2."""
        gas_capacity = self.frc * self._gas_capacity_per_l
        blood_dl_to_l = blood_volume_ml / 100000.0
        cap_pressure = max(0.0, pressure - self._aa_gradient(vq_mismatch))
        content = 1.34 * hb * self._saturation(cap_pressure) + 0.0031 * cap_pressure
        derivative = (
            1.34 * hb * self._saturation_derivative(cap_pressure) + 0.0031
        ) if cap_pressure > 0.0 else 0.0
        shunt = self.shunt_fraction
        extraction = 100.0 * vo2_l_min / max(0.1, cardiac_output)
        if extraction >= (1.0 - shunt) * content:
            arterial_content = (1.0 - shunt) * content
            derivative *= 1.0 - shunt
        else:
            arterial_content = content - shunt / (1.0 - shunt) * extraction
        return (
            gas_capacity * pressure + blood_dl_to_l * arterial_content,
            gas_capacity + blood_dl_to_l * derivative,
        )

    def _pressure_from_oxygen_store(
        self, target: float, hb: float, blood_volume_ml: float,
        cardiac_output: float, vo2_l_min: float, vq_mismatch: float,
    ) -> float:
        """Invert the nonlinear O2 content with a bracketed Newton solve."""
        lower, upper = 0.0, max(0.0, self._atm_dry - self.state.p_alveolar_co2)
        if target <= 0.0:
            return lower
        if target >= self._oxygen_store(
            upper, hb, blood_volume_ml, cardiac_output, vo2_l_min, vq_mismatch,
        )[0]:
            return upper
        pressure = clamp(self.state.p_alveolar_o2, lower, upper)
        for _ in range(24):
            content, derivative = self._oxygen_store(
                pressure, hb, blood_volume_ml, cardiac_output, vo2_l_min, vq_mismatch,
            )
            residual = content - target
            if abs(residual) < 1e-10:
                return pressure
            if residual > 0.0:
                upper = pressure
            else:
                lower = pressure
            candidate = pressure - residual / derivative
            pressure = candidate if lower < candidate < upper else 0.5 * (lower + upper)
        return pressure

    def oxygen_exchange_l_min(
        self, fio2: float, alveolar_ventilation: float, metabolic_factor: float,
        airway_patency: float, vq_mismatch: float, dt: float = 1.0,
    ) -> float:
        """Net O2 transport from inspired gas into the lung and blood stores, STPD L/min."""
        if airway_patency <= 0.0:
            return 0.0
        ventilation = max(0.0, alveolar_ventilation * (1.0 - 0.6 * clamp01(vq_mismatch)))
        vo2 = self.vco2 * max(0.1, metabolic_factor) / self.rq / 1000.0
        apneic_inflow = vo2 * clamp01(airway_patency) * clamp01(1.0 - ventilation)
        return (
            ventilation * self._btps_to_stpd
            * (fio2 - self.state.p_alveolar_o2 / self._atm_dry)
            + (apneic_inflow + self._passive_refill_l_min(airway_patency, dt)) * fio2
        )

    def step(self, dt: float, ce_prop: float, ce_remi: float, mech_vent_mv: float = 0.0,
             fio2: float = 0.21, ce_roc: float = 0.0, mac_sevo: float = 0.0,
             mech_rr: float = 0.0, mech_vt_l: float = 0.0,
             airway_patency: float = 1.0, ventilation_efficiency: float = 1.0,
             vq_mismatch: float = 0.0,
             hb_g_dl: float | None = None,
             cardiac_output: float = 5.0,
             metabolic_factor: float = 1.0,
             blood_volume_ml: float | None = None,
             measured_breaths: bool = False,
             unconscious: float = 0.0, lung_volume_l: float | None = None,
             shunt_fraction: float = 0.0, temp_c: float = 37.0) -> RespState:
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
            cardiac_output: L/min, for the PaCO2-EtCO2 gap.
            metabolic_factor: VO2 and VCO2 multiplier.
            blood_volume_ml: Blood volume (mL), which with Hb sets the blood O2 store.
            measured_breaths: The supplied rate and volume include all observed
                breaths, including spontaneous breaths during mechanical ventilation.
            unconscious: Probability of loss of consciousness, 0-1.
            lung_volume_l: Measured end-expiratory gas volume, L.
            shunt_fraction: Perfusion through closed lung units, 0-1.
            temp_c: Blood temperature for hemoglobin O2 affinity, degrees C.
        """
        state = self.state
        old_gas_pressure = state.p_alveolar_o2 + state.p_alveolar_co2
        hill = hill_function
        clamp01_local = clamp01

        prop_keep = 1.0 - hill(ce_prop, self.c50_prop, self.gamma_prop)
        sevo_keep = 1.0 - hill(mac_sevo, self.c50_sevo_mac, self.gamma_sevo)
        muscle_factor = 1.0 - hill(ce_roc, self.c50_nmba, self.gamma_nmba)
        airway_patency = clamp01_local(airway_patency)
        ventilation_efficiency = clamp01_local(ventilation_efficiency)
        unconscious = clamp01_local(unconscious)

        # At Ce_prop=1 the opioid C50 is 0.84 * 1.6 ng/mL (Olofsen).
        # Interpolation to other propofol concentrations is a teaching assumption.
        remi_c50 = self.c50_remi * (1.0 - 0.32 * max(0.0, ce_prop) / (1.0 + max(0.0, ce_prop)))
        gain = self.hcvr_slope_baseline / 2.2 * (0.42 / 7.2)
        gain *= prop_keep ** (self.prop_co2_gain_exponent - 1.0) * sevo_keep
        co2_excess = max(0.0, state.p_alveolar_co2 - self.paco2_setpoint)
        opioid_reference = 1.0 + gain * co2_excess
        target_drive = opioid_reference - 0.5 * max(0.0, ce_remi) / remi_c50
        # Keep the controller signed during apnea. Clipping its internal state
        # discards drug inhibition and makes recovery too early.
        alpha = -math.expm1(-dt / self.CO2_CONTROLLER_TAU_S)
        self._ventilatory_drive += alpha * (target_drive - self._ventilatory_drive)
        self._resting_age_s += dt
        if self._resting_pco2 is None or self._resting_age_s >= self.RESTING_PCO2_INTERVAL_S:
            self._resting_pco2 = self._solve_resting_pco2(
                prop_keep, sevo_keep, muscle_factor, unconscious, gain,
                0.5 * max(0.0, ce_remi) / remi_c50, max(0.1, float(metabolic_factor)),
            )
            self._resting_age_s = 0.0
        effort_rr, effort_vt, drive_central = self._unassisted(
            state.p_alveolar_co2, prop_keep, sevo_keep, muscle_factor,
            unconscious, opioid_reference, resting_pco2=self._resting_pco2,
            hypoxic_drive=opioid_reference * self._step_hypoxic_control(dt, ce_prop, mac_sevo),
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

        # PACO2 relaxes toward K x VCO2 / VA; V/Q mismatch reduces effective
        # CO2 elimination.
        vq_mismatch = clamp01_local(vq_mismatch)
        effective_va = max(0.1, total_va_l_min * (1.0 - 0.6 * vq_mismatch))

        assisted_active = mech_rr > 0.1 or mech_vent_mv > 0.1
        if assisted_active:
            apnea_like = effective_va <= 0.1
        else:
            apnea_like = state.apnea or effective_va <= 0.1
        if apnea_like:
            self._apnea_timer += dt
        else:
            self._apnea_timer = 0.0

        metabolic_factor = max(0.1, float(metabolic_factor))
        paco2_eq = self.co2_constant * self.vco2 * metabolic_factor / effective_va
        d_paco2 = (paco2_eq - state.p_alveolar_co2) * -math.expm1(-dt / self.tau_co2)

        # CO2 continues accumulating without alveolar ventilation, independent
        # of the baseline breath pattern. Production scales the apneic rise.
        fast_seconds = 0.0
        if self._apnea_timer > 0:
            fast_seconds = min(dt, max(0.0, APNEA_PACO2_RISE_FAST_DURATION_SEC - (self._apnea_timer - dt)))
        max_rise = metabolic_factor * (
            APNEA_PACO2_RISE_FAST_MMHG_MIN * fast_seconds
            + APNEA_PACO2_RISE_SLOW_MMHG_MIN * (dt - fast_seconds)
        ) / 60.0
        if apnea_like:
            d_paco2 = max_rise
        elif d_paco2 > 0:
            d_paco2 = min(d_paco2, max_rise)

        state.p_alveolar_co2 += d_paco2
        # PaCO2 and EtCO2 derive from alveolar CO2. The PaCO2-EtCO2 gap widens
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
        pa_co2_gap = (
            0.5
            + 2.5 * vq_for_etco2
            + 2.5 * clamp01(1.0 - perfusion_ratio)
            + 1.5 * (1.0 - ventilation_efficiency)
        )
        pa_co2_gap = min(8.0, pa_co2_gap)
        state.pa_co2 = state.p_alveolar_co2 + pa_co2_gap
        etco2_raw = max(0.0, state.p_alveolar_co2 - etco2_gradient)
        state.etco2 = etco2_raw

        # Alveolar O2 mass balance over the lung and blood stores:
        # C dPAO2/dt = VA_STPD (PIO2 - PAO2)/Pdry + inflow*FiO2 - VO2, where C is FRC gas
        # plus hemoglobin-bound O2 (steep on the dissociation curve). At steady
        # state PAO2 = PIO2 - 863 VO2/VA_BTPS; in apnea stores deplete at VO2, giving the
        # preoxygenation-dependent safe apnea time (Benumof 1997; Farmery 1996).
        # During apnea a patent airway draws gas in to replace absorbed O2
        # (apneic oxygenation).
        vo2 = self.vco2 * metabolic_factor / self.rq / 1000.0  # L/min, same uptake as the circuit
        hb = self.baseline_hb if hb_g_dl is None else max(0.0, hb_g_dl)
        blood_volume = self.baseline_blood_volume_ml if blood_volume_ml is None else max(0.0, blood_volume_ml)
        new_shunt = clamp(shunt_fraction, 0.0, 0.95)
        if temp_c != self._blood_temp_c or new_shunt != self.shunt_fraction:
            # Affinity and shunt redistribute existing O2 between gas and blood.
            stored = self._oxygen_store(
                state.p_alveolar_o2, hb, blood_volume, cardiac_output, vo2, vq_mismatch,
            )[0]
            self._blood_temp_c = temp_c
            self._p50 = self._temperature_p50(temp_c)
            self.shunt_fraction = new_shunt
            state.p_alveolar_o2 = self._pressure_from_oxygen_store(
                stored, hb, blood_volume, cardiac_output, vo2, vq_mismatch,
            )
            cap_pressure = max(0.0, state.p_alveolar_o2 - self._aa_gradient(vq_mismatch))
            state.p_arterial_o2 = self._arterial_po2(
                cap_pressure, hb, cardiac_output, vo2, state.p_arterial_o2,
            )
        refill_l_min = self._passive_refill_l_min(airway_patency, dt)
        o2_flux = self.oxygen_exchange_l_min(
            fio2, total_va_l_min, metabolic_factor, airway_patency, vq_mismatch, dt,
        ) - vo2 - refill_l_min * fio2
        dsat_dpo2 = self._saturation_derivative(state.p_arterial_o2)
        # Circulating Hb mass sets the store, so blood loss lowers it before hemodilution.
        blood_o2_capacity_l = 1.34 * hb * (blood_volume / 100.0) / 1000.0
        if self.shunt_fraction > 0.0:
            # The blood store follows arterial saturation as alveolar PO2
            # changes. Shunt mixing changes that derivative, especially on O2.
            cap_pressure = max(0.0, state.p_alveolar_o2 - self._aa_gradient(vq_mismatch))
            cap_saturation = self._saturation(cap_pressure)
            cap_derivative = 1.34 * hb * self._saturation_derivative(cap_pressure) + 0.0031
            cap_content = 1.34 * hb * cap_saturation + 0.0031 * cap_pressure
            extraction = 100.0 * vo2 / max(0.1, cardiac_output)
            if extraction >= (1.0 - self.shunt_fraction) * cap_content:
                # At zero venous O2, only the ventilated share changes CaO2.
                cap_derivative *= 1.0 - self.shunt_fraction
            art_derivative = 1.34 * hb * dsat_dpo2 + 0.0031
            dsat_dpo2 *= cap_derivative / art_derivative
        if lung_volume_l is not None:
            volume_change = lung_volume_l - self.frc
            self.frc = lung_volume_l
            if volume_change > 0.0:
                # Extra end-expiratory volume came from inspired gas. Loss of
                # gas at the current alveolar composition leaves PO2 unchanged.
                capacity = self.frc * self._gas_capacity_per_l + blood_o2_capacity_l * dsat_dpo2
                state.p_alveolar_o2 += (volume_change * self._gas_capacity_per_l
                                       * (fio2 * self._atm_dry - state.p_alveolar_o2) / capacity)
        o2_capacitance = self.frc * self._gas_capacity_per_l + blood_o2_capacity_l * dsat_dpo2
        state.p_alveolar_o2 += o2_flux / o2_capacitance * dt / 60.0
        state.p_alveolar_o2 = clamp(state.p_alveolar_o2, 0.0, max(0.0, self._atm_dry - state.p_alveolar_co2))
        self._add_oxygen_store(refill_l_min * fio2 * dt / 60.0, hb, blood_volume,
                               cardiac_output, vo2, vq_mismatch)
        if airway_patency <= 0.0:
            # Only loss from the lung gas creates hypobaric inflow. Blood O2
            # buffering does not, and accumulating alveolar CO2 offsets it.
            gas_pressure_loss = old_gas_pressure - state.p_alveolar_o2 - state.p_alveolar_co2
            self._apneic_gas_deficit_l = max(
                0.0, self._apneic_gas_deficit_l + self.frc * self._gas_capacity_per_l * gas_pressure_loss,
            )
        else:
            self._apneic_gas_deficit_l = max(
                0.0, self._apneic_gas_deficit_l - refill_l_min * dt / 60.0,
            )

        # Residual V/Q inequality sets end-capillary PO2. Closed units mix
        # venous blood into arterial blood by O2 content, not partial pressure.
        end_capillary = max(0.0, state.p_alveolar_o2 - self._aa_gradient(vq_mismatch))
        state.p_arterial_o2 = self._arterial_po2(end_capillary, hb, cardiac_output, vo2,
                                              state.p_arterial_o2)

        # End-tidal gas is alveolar gas diluted by alveolar dead-space gas of
        # inspired composition, the dilution that puts EtCO2 below PACO2. Gas
        # analyzers report dry-gas fractions, like FiO2.
        dilution = clamp01_local(1.0 - etco2_raw / state.p_alveolar_co2) if state.p_alveolar_co2 > 0.0 else 0.0
        p_end_tidal_o2 = state.p_alveolar_o2 + dilution * (fio2 * self._atm_dry - state.p_alveolar_o2)
        state.eto2 = 100.0 * p_end_tidal_o2 / self._atm_dry

        state.drive_central = drive_central
        state.muscle_factor = muscle_factor
        state.rr = current_rr
        state.vt = current_vt
        state.mv = (current_rr * current_vt) / 1000.0
        state.sao2 = 100.0 * self._saturation(state.p_arterial_o2)

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

    def _solve_resting_pco2(self, prop_keep: float, sevo_keep: float, muscle_factor: float,
                       unconscious: float, gain: float, opioid_shift: float,
                       metabolic_factor: float) -> float:
        """Alveolar PCO2 at which unassisted breathing removes the CO2 produced.

        The opioid controller is taken at equilibrium. Without drugs this is the set point.
        """
        produced = self.co2_constant * self.vco2 * metabolic_factor

        def removed(pco2: float) -> float:
            reference = 1.0 + gain * max(0.0, pco2 - self.paco2_setpoint)
            rr, vt, _ = self._unassisted(pco2, prop_keep, sevo_keep, muscle_factor, unconscious,
                                         reference, controller=reference - opioid_shift)
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

    def _unassisted(self, pco2: float, prop_keep: float, sevo_keep: float,
                    muscle_factor: float,
                    unconscious: float, opioid_reference: float,
                    controller: float | None = None,
                    resting_pco2: float | None = None,
                    hypoxic_drive: float = 0.0) -> tuple[float, float, float]:
        """Convert normalized alveolar demand to a rate and tidal volume.

        Opioids mainly slow rate; hypnotics mainly reduce depth. This split and
        the awake hypocapnic response are teaching approximations.
        """
        controller = max(0.0, (self._ventilatory_drive if controller is None else controller) + hypoxic_drive)
        # Transfer the opioid controller as a fraction of its drug-free response
        # at the same CO2, then apply Bouillon's nonlinear alveolar response.
        co2_exponent = self.CO2_RESPONSE_EXPONENT * self.hcvr_slope_baseline / 2.2
        co2_response = (max(self.paco2_setpoint, pco2) / self.paco2_setpoint) ** co2_exponent
        response_fraction = controller / opioid_reference * co2_response
        drive = min(self.MAX_ALVEOLAR_DRIVE, response_fraction * prop_keep * sevo_keep)
        rate_fraction = min(1.0, controller)
        if controller > 1.0:
            rate_fraction += min(
                0.6 * (response_fraction - 1.0),
                (0.06 + 0.04 * unconscious) * max(0.0, pco2 - self.paco2_setpoint - self.RATE_CO2_GAP)
                + 0.3 * hypoxic_drive,
            )
        rate_fraction *= (0.6 + 0.4 * prop_keep) * (0.6 + 0.4 * sevo_keep)
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
        elif rr < RR_BRADYPNEA_THRESHOLD:
            rr *= 0.5
        return rr, vt, drive

    def _aa_gradient(self, vq_mismatch: float) -> float:
        """Alveolar to end-capillary PO2 difference (mmHg) from residual V/Q inequality."""
        return self.aa_grad_base + self.VQ_MISMATCH_GRADIENT * clamp01(vq_mismatch)

    @staticmethod
    def _temperature_p50(temp_c: float) -> float:
        """Dash 2016 Eq. 9d; pH, PCO2, and 2,3-DPG held at reference values."""
        delta = temp_c - 37.0
        return 26.8 + delta * (1.435 + delta * (0.04163 + delta * 0.000686))

    def _saturation(self, pao2: float) -> float:
        """Dash 2016 Eqs. 6 and 11, including low-saturation cooperativity."""
        return self._saturation_at_p50(pao2, self._p50)

    @staticmethod
    def _saturation_at_p50(pao2: float, p50: float) -> float:
        if pao2 <= 0.0:
            return 0.0
        exponent = 2.82 - 1.20 * 10.0**(-pao2 / 29.25)
        return 1.0 / (1.0 + (p50 / pao2)**exponent)

    def _saturation_derivative(self, pao2: float) -> float:
        """dS/dPO2 includes the pressure-dependent Hill exponent."""
        if pao2 <= 0.0:
            return 0.0
        correction = 1.20 * 10.0**(-pao2 / 29.25)
        exponent = 2.82 - correction
        exponent_derivative = math.log(10.0) * correction / 29.25
        sat = self._saturation(pao2)
        return sat * (1.0 - sat) * (exponent / pao2 + exponent_derivative * math.log(pao2 / self._p50))

    def _arterial_po2(self, end_capillary: float, hb: float, cardiac_output: float,
                      vo2_l_min: float, previous: float) -> float:
        """Shunt mixing with venous content determined by Fick O2 extraction.

        Ca = (1-s) Cc + s Cv and Cv = Ca - VO2/Q give
        Ca = Cc - s/(1-s) VO2/Q. Contents include dissolved O2 (mL/dL).
        Extraction stops at Cv = 0, so Ca cannot fall below (1-s) Cc.
        """
        shunt = self.shunt_fraction
        if shunt == 0.0:
            return end_capillary
        capacity = 1.34 * hb
        extraction = 100.0 * vo2_l_min / max(0.1, cardiac_output)
        capillary_content = capacity * self._saturation(end_capillary) + 0.0031 * end_capillary
        target = max((1.0 - shunt) * capillary_content,
                     capillary_content - shunt / (1.0 - shunt) * extraction)
        lower, upper = 0.0, end_capillary
        pressure = min(max(previous, lower), upper)
        for _ in range(16):
            saturation = self._saturation(pressure)
            residual = capacity * saturation + 0.0031 * pressure - target
            if abs(residual) < 1e-8:
                return pressure
            if residual > 0.0:
                upper = pressure
            else:
                lower = pressure
            derivative = capacity * self._saturation_derivative(pressure) + 0.0031
            candidate = pressure - residual / derivative
            pressure = candidate if lower < candidate < upper else 0.5 * (lower + upper)
        return pressure
