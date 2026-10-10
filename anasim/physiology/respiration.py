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
    AWAKE_HYPOCAPNIC_GAIN = 0.2  # Fraction of HCVR; nonlinear depth response, a teaching estimate
    RATE_CO2_GAP = 5.0  # mmHg above the set point before hypercapnia raises frequency
    CO2_CONTROLLER_TAU_S = 150.0  # Olofsen 2010, 2.5 min
    CO2_RESPONSE_EXPONENT = 4.37  # Bouillon 2004 nonlinear CO2 response.
    MAX_ALVEOLAR_DRIVE = 12.0  # Bound demand above the nonlinear response at PCO2 70.

    def __init__(self, patient: Patient):
        self.patient = patient
        self.rr_0 = patient.baseline_rr
        self.vt_0 = patient.baseline_vt
        self.baseline_hb = patient.baseline_hb
        # Baseline CO for perfusion effects, matching the hemodynamic defaults.
        ci_adult = 3.0
        ci_elderly = 2.5
        ci = ci_elderly if patient.age > 70 else ci_adult
        self.baseline_co_l_min = max(0.1, ci * patient.bsa)

        # Bouillon 2004 respiratory concentration-response curve, using its
        # own effect site (equilibration half-time 2.6 min).
        self.c50_prop = 1.33
        self.gamma_prop = 1.68
        # Olofsen 2010: a linear opioid shift allows apnea at finite Ce.
        self.c50_remi = 1.6
        self.gamma_remi = 1.0  # Used by the separate shivering response.
        self._ventilatory_drive = 1.0
        # Olofsen's low-dose propofol background reduces opioid-controller gain
        # to 0.46. Interpolation to other concentrations is a transfer assumption.
        prop_keep_at_one = 1.0 - hill_function(1.0, self.c50_prop, self.gamma_prop)
        self.prop_co2_gain_exponent = math.log(0.46) / math.log(prop_keep_at_one)

        # Sevoflurane barely changes the CO2 response at 0.1 MAC (Pandit 1999)
        # and depresses it at 1.1-1.4 MAC (Doi 1987).
        self.c50_sevo_mac = 1.1
        self.gamma_sevo = 2.0

        # Free rocuronium at the central (diaphragm and larynx) effect site. These
        # muscles need about 1.8x the adductor pollicis concentration (Cantineau
        # 1994; Plaud 1995).
        self.c50_nmba = 1.8 * 0.8
        self.gamma_nmba = 3.0

        # Awake hypocapnic calibration (Nieuwenhuijs 2001; Pandit 1999).
        self.hcvr_slope_baseline = 2.2
        self.paco2_setpoint = 40.0
        self.state = RespState(
            rr=self.rr_0, vt=self.vt_0, mv=self.rr_0 * self.vt_0 / 1000.0,
            effort_rr=self.rr_0, effort_vt=self.vt_0,
        )
        self.rq = 0.8

        # Resting VO2 about 3.6 mL/kg/min.
        self.vo2_ml_kg_min = 3.6
        self.vco2 = self.vo2_ml_kg_min * patient.weight * self.rq  # mL/min
        self.frc = patient.functional_residual_capacity()
        self.shunt_fraction = 0.0
        self.baseline_blood_volume_ml = patient.estimate_blood_volume()

        self.vd_deadspace = 2.2 * patient.weight / 1000.0  # L

        # Body CO2 stores equilibrate slowly (apneic rise 3-5 mmHg/min).
        self.tau_co2 = 180.0  # s
        self.atm_p = 760.0
        self.vapor_p = 47.0
        self._atm_dry = self.atm_p - self.vapor_p
        # Lung ventilation/volume is BTPS; VO2 and Hb-bound O2 are STPD.
        self._btps_to_stpd = self._atm_dry / 760.0 * 273.15 / 310.15
        self._gas_capacity_per_l = self._btps_to_stpd / self._atm_dry
        # Age-adjusted A-a gradient, age/4 + 4 mmHg.
        self.aa_grad_base = max(5.0, (self.patient.age / 4.0) + 4.0)
        self.equilibrate_oxygen(0.21)
        self._p50 = 26.6
        self._n_hill = 2.7
        self._p50_pow = self._p50 ** self._n_hill
        self._apnea_timer = 0.0
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

    def oxygen_exchange_l_min(
        self, fio2: float, alveolar_ventilation: float, metabolic_factor: float,
        airway_patency: float, vq_mismatch: float,
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
            + apneic_inflow * fio2
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
             shunt_fraction: float = 0.0) -> RespState:
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
        """
        state = self.state
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
        effort_rr, effort_vt, drive_central = self._unassisted(
            state.p_alveolar_co2, prop_keep, sevo_keep, muscle_factor,
            unconscious, opioid_reference,
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

        # PACO2 relaxes toward 40 x metabolic factor x VA_baseline / VA; V/Q
        # mismatch reduces effective CO2 elimination.
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
        paco2_base = 40.0
        paco2_eq = paco2_base * metabolic_factor * (self.va_baseline / effective_va)
        d_paco2 = (paco2_eq - state.p_alveolar_co2) / self.tau_co2 * dt

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
        o2_flux = self.oxygen_exchange_l_min(
            fio2, total_va_l_min, metabolic_factor, airway_patency, vq_mismatch,
        ) - vo2
        sat = self._saturation(state.p_arterial_o2)
        dsat_dpo2 = self._n_hill * sat * (1.0 - sat) / max(state.p_arterial_o2, 1.0)
        hb = self.baseline_hb if hb_g_dl is None else max(0.0, hb_g_dl)
        blood_volume = self.baseline_blood_volume_ml if blood_volume_ml is None else max(0.0, blood_volume_ml)
        # Circulating Hb mass sets the store, so blood loss lowers it before hemodilution.
        blood_o2_capacity_l = 1.34 * hb * (blood_volume / 100.0) / 1000.0
        self.shunt_fraction = clamp(shunt_fraction, 0.0, 0.95)
        if self.shunt_fraction > 0.0:
            # The blood store follows arterial saturation as alveolar PO2
            # changes. Shunt mixing changes that derivative, especially on O2.
            cap_pressure = max(0.0, state.p_alveolar_o2 - self.aa_grad_base * (1.0 + 2.5 * vq_mismatch))
            cap_saturation = self._saturation(cap_pressure)
            cap_derivative = (1.34 * hb * self._n_hill * cap_saturation * (1.0 - cap_saturation)
                              / max(cap_pressure, 1.0) + 0.0031)
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

        # Residual V/Q inequality sets end-capillary PO2. Closed units mix
        # venous blood into arterial blood by O2 content, not partial pressure.
        end_capillary = max(0.0, state.p_alveolar_o2 - self.aa_grad_base * (1.0 + 2.5 * vq_mismatch))
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

    def initialize_drug_effects(self, ce_prop: float, ce_remi: float) -> None:
        """Seed the signed ventilatory controller for a maintenance history."""
        remi_c50 = self.c50_remi * (1.0 - 0.32 * ce_prop / (1.0 + ce_prop))
        self._ventilatory_drive = 1.0 - 0.5 * ce_remi / remi_c50

    def _unassisted(self, pco2: float, prop_keep: float, sevo_keep: float,
                    muscle_factor: float,
                    unconscious: float, opioid_reference: float) -> tuple[float, float, float]:
        """Convert normalized alveolar demand to a rate and tidal volume.

        Opioids mainly slow rate; hypnotics mainly reduce depth. This split and
        the awake hypocapnic response are teaching approximations.
        """
        controller = max(0.0, self._ventilatory_drive)
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
                (0.06 + 0.04 * unconscious) * max(0.0, pco2 - self.paco2_setpoint - self.RATE_CO2_GAP),
            )
        rate_fraction *= (0.6 + 0.4 * prop_keep) * (0.6 + 0.4 * sevo_keep)
        rr = self.rr_0 * rate_fraction
        # Add dead space after allocating the alveolar demand to breaths.
        # Applying drug depression to total VT would subtract dead space twice.
        vt = (self.vd_deadspace + self.va_baseline * drive / rr) * 1000.0 if rr > 0.0 else 0.0
        vt *= muscle_factor

        below = max(0.0, self.paco2_setpoint - pco2)
        if below > 0.0:
            kept = 1.0 - unconscious * clamp01(below / self.APNEIC_GAP)
            rr *= math.sqrt(kept)
            awake_kept = 1.0 / (1.0 + self.AWAKE_HYPOCAPNIC_GAIN * self.hcvr_slope_baseline * below / self.va_baseline)
            vt *= math.sqrt(kept) * (unconscious + (1.0 - unconscious) * awake_kept)
        if rr < RR_APNEA_THRESHOLD or vt < max(100.0, VT_MIN):
            rr = vt = 0.0
        elif rr < RR_BRADYPNEA_THRESHOLD:
            rr *= 0.5
        return rr, vt, drive

    def _saturation(self, pao2: float) -> float:
        """Hill oxyhemoglobin dissociation (P50 26.6 mmHg, n 2.7) as a fraction."""
        pao2_pow = max(0.0, pao2) ** self._n_hill
        return pao2_pow / (pao2_pow + self._p50_pow)

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
            derivative = (capacity * self._n_hill * saturation * (1.0 - saturation)
                          / max(pressure, 1e-12) + 0.0031)
            candidate = pressure - residual / derivative
            pressure = candidate if lower < candidate < upper else 0.5 * (lower + upper)
        return pressure
