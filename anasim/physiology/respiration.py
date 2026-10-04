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


class RespiratoryModel:
    """Ventilatory control, anesthetic respiratory depression, and CO2 and O2 exchange."""

    # Under anesthesia the apneic threshold is 4-5 mmHg below resting PaCO2,
    # independent of agent and depth (Hickey 1971).
    APNEIC_GAP = 4.5  # mmHg
    AWAKE_HYPOCAPNIC_GAIN = 0.2  # Fraction of HCVR; nonlinear depth response, a teaching estimate
    RATE_CO2_GAP = 5.0  # mmHg above the set point before hypercapnia raises frequency

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

        # Propofol depresses central drive (HCVR) at lower concentrations than
        # rate and depth (Blouin 1993; Nieuwenhuijs 2001; Lee 2011).
        self.c50_prop_hcvr = 2.0
        self.gamma_prop_hcvr = 2.0
        self.c50_prop_mech = 4.0
        self.gamma_prop_mech = 2.0

        # Remifentanil EC50 about 1.1-1.2 ng/mL (Glass 1999; Babenco 2000); it
        # also shifts the CO2 set point right.
        self.c50_remi = 1.2
        self.gamma_remi = 1.7
        self.remi_setpoint_shift_max = 8.0  # mmHg

        # Sevoflurane barely changes the CO2 response at 0.1 MAC (Pandit 1999)
        # and depresses it at 1.1-1.4 MAC (Doi 1987).
        self.c50_sevo_mac = 1.1
        self.gamma_sevo = 2.0

        # Free rocuronium at the central (diaphragm and larynx) effect site. These
        # muscles need about 1.8x the adductor pollicis concentration (Cantineau
        # 1994; Plaud 1995).
        self.c50_nmba = 1.8 * 0.8
        self.gamma_nmba = 3.0

        # HCVR about 2-3 L/min per mmHg (Nieuwenhuijs 2001; Pandit 1999).
        self.hcvr_slope_baseline = 2.2
        self.paco2_setpoint = 40.0
        # Fractional HCVR slope reduction at full drug effect.
        self.hcvr_depression_remi = 0.70
        self.hcvr_depression_prop = 0.40
        self.hcvr_depression_sevo = 0.50

        # Share of each drug's effect on rate vs tidal volume. Opioids slow the
        # rate; propofol and sevoflurane mainly reduce depth.
        self.w_prop_rr = 0.6
        self.w_prop_vt = 0.8
        self.w_remi_rr = 1.0
        self.w_remi_vt = 0.35
        self.w_sevo_rr = 0.4
        self.w_sevo_vt = 0.8

        self.state = RespState(self.rr_0, self.vt_0, (self.rr_0 * self.vt_0)/1000.0)
        self.rq = 0.8

        # Resting VO2 about 3.6 mL/kg/min.
        self.vo2_ml_kg_min = 3.6
        self.vco2 = self.vo2_ml_kg_min * patient.weight * self.rq  # mL/min
        self.frc = patient.functional_residual_capacity()
        self.shunt_fraction = 0.0
        self.baseline_blood_volume_ml = patient.estimate_blood_volume()

        self.vd_deadspace = 2.2 * patient.weight / 1000.0  # L
        self.va_baseline = max(0.1, (self.vt_0/1000.0 - self.vd_deadspace) * self.rr_0)  # L/min

        # Body CO2 stores equilibrate slowly (apneic rise 3-5 mmHg/min).
        self.tau_co2 = 180.0  # s
        self.atm_p = 760.0
        self.vapor_p = 47.0
        self._atm_dry = self.atm_p - self.vapor_p
        # Lung ventilation/volume is BTPS; VO2 and Hb-bound O2 are STPD.
        self._btps_to_stpd = self._atm_dry / 760.0 * 273.15 / 310.15
        self._gas_capacity_per_l = self._btps_to_stpd / self._atm_dry
        # Age-adjusted A-a gradient, age/4 + 4 mmHg (Stein 1995).
        self.aa_grad_base = max(5.0, (self.patient.age / 4.0) + 4.0)
        self.equilibrate_oxygen(0.21)
        self._p50 = 26.6
        self._n_hill = 2.7
        self._p50_pow = self._p50 ** self._n_hill
        self._apnea_timer = 0.0
        # Perfusion effect on deadspace fraction (low flow increases VD/VT).
        self.perfusion_deadspace_gain = 0.25
        self._own_pco2 = self.state.p_alveolar_co2  # PCO2 the patient's own breathing would hold

    def equilibrate_oxygen(self, fio2: float) -> None:
        """Set alveolar O2 to its alveolar-gas-equation value at the current PACO2."""
        self.state.p_alveolar_o2 = max(0.0, fio2 * self._atm_dry - self.state.p_alveolar_co2 / self.rq)
        self.state.p_arterial_o2 = max(0.0, self.state.p_alveolar_o2 - self.aa_grad_base)

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

        eff_prop_hcvr = hill(ce_prop, self.c50_prop_hcvr, self.gamma_prop_hcvr)
        eff_prop_mech = hill(ce_prop, self.c50_prop_mech, self.gamma_prop_mech)
        eff_remi = hill(ce_remi, self.c50_remi, self.gamma_remi)
        eff_sevo = hill(mac_sevo, self.c50_sevo_mac, self.gamma_sevo)
        eff_nmba = hill(ce_roc, self.c50_nmba, self.gamma_nmba)

        # Central drive; multiplicative drug interaction is synergistic.
        drug_drive = (1.0 - eff_prop_hcvr) * (1.0 - eff_remi) * (1.0 - eff_sevo)

        # HCVR: VA boost = slope x (PACO2 - set point). Drugs flatten the slope
        # multiplicatively, and opioids move the set point right.
        factor_remi = max(0.0, 1.0 - self.hcvr_depression_remi * eff_remi)
        factor_prop = max(0.0, 1.0 - self.hcvr_depression_prop * eff_prop_hcvr)
        factor_sevo = max(0.0, 1.0 - self.hcvr_depression_sevo * eff_sevo)
        hcvr_slope = self.hcvr_slope_baseline * factor_remi * factor_prop * factor_sevo
        effective_setpoint = self.paco2_setpoint + (self.remi_setpoint_shift_max * eff_remi)

        # Neuromuscular block weakens the muscles without changing drive.
        muscle_factor = 1.0 - eff_nmba
        rr_inhib_base = (self.w_prop_rr * eff_prop_mech +
                         self.w_remi_rr * eff_remi +
                         self.w_sevo_rr * eff_sevo)
        vt_inhib_base = (self.w_prop_vt * eff_prop_mech +
                         self.w_remi_vt * eff_remi +
                         self.w_sevo_vt * eff_sevo)
        airway_patency = clamp01_local(airway_patency)
        ventilation_efficiency = clamp01_local(ventilation_efficiency)
        unconscious = clamp01_local(unconscious)
        pattern = (drug_drive, hcvr_slope, effective_setpoint, rr_inhib_base, vt_inhib_base, muscle_factor,
                   airway_patency * ventilation_efficiency, unconscious)
        current_rr, current_vt, drive_central = self._unassisted(state.p_alveolar_co2, *pattern)

        # Awake hypocapnia reduces effort intensity while rhythmic breathing
        # persists (Patrick 1995). Under anesthesia, rate and depth fall to the
        # apneic threshold (Hickey 1971).
        below = self._own_pco2 - state.p_alveolar_co2
        if below > 0.0 and current_rr > 0.0:
            kept = 1.0 - unconscious * clamp01_local(below / self.APNEIC_GAP)
            share = math.sqrt(kept)
            dead_space_ml = 1000.0 * self.vd_deadspace
            current_rr *= share
            if current_vt > dead_space_ml:
                awake_kept = 1.0 / (1.0 + self.AWAKE_HYPOCAPNIC_GAIN * hcvr_slope * below / self.va_baseline)
                depth = share * (1.0 - (1.0 - unconscious) * (1.0 - awake_kept))
                current_vt = dead_space_ml + depth * (current_vt - dead_space_ml)
            if current_rr < RR_APNEA_THRESHOLD:
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
        paco2_eq = min(150.0, paco2_eq)
        d_paco2 = (paco2_eq - state.p_alveolar_co2) / self.tau_co2 * dt

        # Cap the rise at the apneic rate (fast for the first minute, then slow)
        # scaled by CO2 production. Washout is not limited.
        if d_paco2 > 0:
            fast_seconds = 0.0
            if self._apnea_timer > 0:
                fast_seconds = min(dt, max(0.0, APNEA_PACO2_RISE_FAST_DURATION_SEC - (self._apnea_timer - dt)))
            max_rise = metabolic_factor * (
                APNEA_PACO2_RISE_FAST_MMHG_MIN * fast_seconds
                + APNEA_PACO2_RISE_SLOW_MMHG_MIN * (dt - fast_seconds)
            ) / 60.0
            d_paco2 = min(d_paco2, max_rise)

        state.p_alveolar_co2 += d_paco2
        # The PCO2 the patient's own breathing would hold follows the same
        # dynamics; only assistance holds the actual value lower.
        own_rr, own_vt, _ = self._unassisted(self._own_pco2, *pattern)
        own_va = max(0.1, own_rr * max(0.0, own_vt / 1000.0 - vd) * (1.0 - 0.6 * vq_mismatch))
        own_eq = min(150.0, paco2_base * metabolic_factor * (self.va_baseline / own_va))
        own_change = min((own_eq - self._own_pco2) / self.tau_co2 * dt,
                         metabolic_factor * APNEA_PACO2_RISE_SLOW_MMHG_MIN * dt / 60.0)
        self._own_pco2 = max(self._own_pco2 + own_change, state.p_alveolar_co2)

        # PaCO2 and EtCO2 derive from alveolar CO2. The PaCO2-EtCO2 gap widens
        # with dead space, V/Q mismatch, obstruction, and low cardiac output
        # (Russell 1990; Lujan 2008; Kim 2019).
        perfusion_ratio = 1.0
        if self.baseline_co_l_min > 0:
            perfusion_ratio = clamp(cardiac_output / self.baseline_co_l_min, 0.05, 1.2)
        if mech_rr > 0:
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
        o2_ventilation = max(0.0, total_va_l_min * (1.0 - 0.6 * vq_mismatch))
        apneic_inflow = vo2 * airway_patency * clamp01_local(1.0 - o2_ventilation)
        o2_flux = (
            o2_ventilation * self._btps_to_stpd * (fio2 * self._atm_dry - state.p_alveolar_o2) / self._atm_dry
            + apneic_inflow * fio2
            - vo2
        )
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

    def _unassisted(self, pco2: float, drug_drive: float, hcvr_slope: float, setpoint: float,
                    rr_inhib: float, vt_inhib: float, muscle_factor: float,
                    vent_factor: float, unconscious: float) -> tuple[float, float, float]:
        """Unassisted rate (/min), VT (mL), and central drive at an alveolar PCO2."""
        boost = hcvr_slope * max(0.0, pco2 - setpoint) / self.va_baseline if self.va_baseline > 0 else 0.0
        # Express the VA boost as drive relative to baseline VA, capped at 2x.
        drive = min(2.0, drug_drive + boost)
        # Hypercapnia partly overcomes drug depression, by up to 50%.
        counteraction = min(0.5, boost * 0.3)
        rr_fraction = (1.0 - clamp01(rr_inhib * (1.0 - counteraction))) * muscle_factor
        vt_fraction = (1.0 - clamp01(vt_inhib * (1.0 - counteraction))) * muscle_factor
        # Modest hypercapnia primarily increases depth; assigning it all to
        # frequency makes tiny CO2 fluctuations shift every assisted breath.
        # At larger rises, frequency contributes too (Georgopoulos 1997).
        frequency = 1.0
        if drive > 1.0:
            awake_rate = min(0.6 * (drive - 1.0), 0.06 * max(0.0, pco2 - setpoint - self.RATE_CO2_GAP))
            frequency += (1.0 - unconscious) * awake_rate + unconscious * 0.6 * (drive - 1.0)
            rr_fraction *= frequency
        rr = self.rr_0 * rr_fraction
        # Below the apnea threshold breathing stops; in the bradypnea range
        # irregular breaths halve the effective rate.
        if rr < RR_APNEA_THRESHOLD:
            rr = 0.0
        elif rr < RR_BRADYPNEA_THRESHOLD:
            rr *= 0.5
        vt = self.vt_0 * vt_fraction
        dead_space_ml = 1000.0 * self.vd_deadspace
        if drive > 1.0 and vt > dead_space_ml:
            depth = 1.0 + (1.0 - unconscious) * (drive / frequency - 1.0)
            vt = dead_space_ml + (vt - dead_space_ml) * depth
        if vt < VT_MIN:
            vt = 0.0
        vt *= vent_factor
        # Breaths under 100 mL do not produce a detectable capnogram.
        if vt < 100.0:
            rr = 0.0
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
        """
        shunt = self.shunt_fraction
        if shunt == 0.0:
            return end_capillary
        capacity = 1.34 * hb
        extraction = 100.0 * vo2_l_min / max(0.1, cardiac_output)
        target = max(0.0, capacity * self._saturation(end_capillary) + 0.0031 * end_capillary
                     - shunt / (1.0 - shunt) * extraction)
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
