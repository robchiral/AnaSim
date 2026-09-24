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
    """Respiratory outputs. Gas tensions in mmHg, VT in mL, MV and VA in L/min."""

    rr: float = 12.0
    vt: float = 500.0
    mv: float = 6.0
    va: float = 4.0
    apnea: bool = False
    p_alveolar_co2: float = 40.0
    pa_co2: float = 40.0
    etco2: float = 40.0
    p_alveolar_o2: float = 100.0
    p_arterial_o2: float = 95.0
    sao2: float = 98.0  # %
    drive_central: float = 1.0
    muscle_factor: float = 1.0


class RespiratoryModel:
    """Ventilatory control, anesthetic respiratory depression, and CO2 and O2 exchange."""

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
        self.frc = 2.5  # L
        # Hemoglobin-bound O2 per unit saturation, L per g/dL.
        self._blood_o2_per_hb = 1.34 * 10.0 * patient.estimate_blood_volume() / 1e6

        self.vd_deadspace = 2.2 * patient.weight / 1000.0  # L
        self.va_baseline = max(0.1, (self.vt_0/1000.0 - self.vd_deadspace) * self.rr_0)  # L/min

        # Body CO2 stores equilibrate slowly (apneic rise 3-5 mmHg/min).
        self.tau_co2 = 180.0  # s
        self.mean_paw_recruit_gain = 0.03
        self.atm_p = 760.0
        self.vapor_p = 47.0
        self._atm_dry = self.atm_p - self.vapor_p
        # Age-adjusted A-a gradient, age/4 + 4 mmHg (Stein 1995).
        self.aa_grad_base = max(5.0, (self.patient.age / 4.0) + 4.0)
        self.equilibrate_oxygen(0.21)
        self._p50 = 26.6
        self._n_hill = 2.7
        self._p50_pow = self._p50 ** self._n_hill
        self._apnea_timer = 0.0
        # Perfusion effect on deadspace fraction (low flow increases VD/VT).
        self.perfusion_deadspace_gain = 0.25

    def equilibrate_oxygen(self, fio2: float) -> None:
        """Set alveolar O2 to its alveolar-gas-equation value at the current PACO2."""
        self.state.p_alveolar_o2 = max(0.0, fio2 * self._atm_dry - self.state.p_alveolar_co2 / self.rq)
        self.state.p_arterial_o2 = max(10.0, self.state.p_alveolar_o2 - self.aa_grad_base)

    def step(self, dt: float, ce_prop: float, ce_remi: float, mech_vent_mv: float = 0.0,
             fio2: float = 0.21, ce_roc: float = 0.0, mac_sevo: float = 0.0,
             peep: float = 0.0, mean_paw: float = 5.0,
             mech_rr: float = 0.0, mech_vt_l: float = 0.0,
             airway_patency: float = 1.0, ventilation_efficiency: float = 1.0,
             vq_mismatch: float = 0.0,
             hb_g_dl: float | None = None,
             cardiac_output: float = 5.0,
             metabolic_factor: float = 1.0) -> RespState:
        """Advance respiration by dt seconds.

        Args:
            ce_prop: Propofol effect-site concentration (mcg/mL).
            ce_remi: Remifentanil effect-site concentration (ng/mL).
            mech_vent_mv: Assisted minute ventilation (L/min).
            fio2: Inspired O2 fraction.
            ce_roc: Free rocuronium at the central effect site (mcg/mL).
            mac_sevo: Brain sevoflurane MAC.
            peep, mean_paw: Airway pressures (cmH2O).
            mech_rr, mech_vt_l: Assisted rate (breaths/min) and tidal volume (L).
            airway_patency: Upper-airway patency, 0-1.
            ventilation_efficiency: Lower-airway efficiency (bronchospasm), 0-1.
            vq_mismatch: V/Q mismatch severity, 0-1.
            hb_g_dl: Hemoglobin (g/dL), which sets the blood O2 store.
            cardiac_output: L/min, for the PaCO2-EtCO2 gap.
            metabolic_factor: VO2 and VCO2 multiplier.
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
        drive_central = (1.0 - eff_prop_hcvr) * (1.0 - eff_remi) * (1.0 - eff_sevo)

        # HCVR: VA boost = slope x (PACO2 - set point). Drugs flatten the slope
        # multiplicatively, and opioids move the set point right.
        factor_remi = max(0.0, 1.0 - self.hcvr_depression_remi * eff_remi)
        factor_prop = max(0.0, 1.0 - self.hcvr_depression_prop * eff_prop_hcvr)
        factor_sevo = max(0.0, 1.0 - self.hcvr_depression_sevo * eff_sevo)

        slope_factor = factor_remi * factor_prop * factor_sevo
        hcvr_slope = self.hcvr_slope_baseline * slope_factor
        effective_setpoint = self.paco2_setpoint + (self.remi_setpoint_shift_max * eff_remi)
        co2_above_setpoint = max(0.0, state.p_alveolar_co2 - effective_setpoint)
        va_boost_from_co2 = hcvr_slope * co2_above_setpoint
        # Express the VA boost as drive relative to baseline VA, capped at 2x.
        if self.va_baseline > 0:
            co2_drive_boost = va_boost_from_co2 / self.va_baseline
        else:
            co2_drive_boost = 0.0
        drive_central = min(2.0, drive_central + co2_drive_boost)

        # Neuromuscular block weakens the muscles without changing drive.
        muscle_factor = 1.0 - eff_nmba

        rr_inhib_base = (self.w_prop_rr * eff_prop_mech +
                         self.w_remi_rr * eff_remi +
                         self.w_sevo_rr * eff_sevo)

        vt_inhib_base = (self.w_prop_vt * eff_prop_mech +
                         self.w_remi_vt * eff_remi +
                         self.w_sevo_vt * eff_sevo)

        # Hypercapnia partly overcomes drug depression, by up to 50%.
        hcvr_counteraction = min(0.5, co2_drive_boost * 0.3)
        rr_fraction = 1.0 - clamp01_local(rr_inhib_base * (1.0 - hcvr_counteraction))
        vt_fraction = 1.0 - clamp01_local(vt_inhib_base * (1.0 - hcvr_counteraction))
        vt_fraction *= muscle_factor
        rr_fraction *= muscle_factor

        # Drive above baseline raises the rate (60% of the excess).
        if drive_central > 1.0:
            rr_fraction *= 1.0 + (drive_central - 1.0) * 0.6

        current_rr = self.rr_0 * rr_fraction
        # Below the apnea threshold breathing stops; in the bradypnea range
        # irregular breaths halve the effective rate.
        if current_rr < RR_APNEA_THRESHOLD:
            current_rr = 0.0
            state.apnea = True
        elif current_rr < RR_BRADYPNEA_THRESHOLD:
            current_rr = current_rr * 0.5
            state.apnea = False
        else:
            state.apnea = False

        current_vt = self.vt_0 * vt_fraction
        if current_vt < VT_MIN:
            current_vt = 0.0

        airway_patency = clamp01_local(airway_patency)
        ventilation_efficiency = clamp01_local(ventilation_efficiency)
        vent_factor = airway_patency * ventilation_efficiency
        current_vt *= vent_factor
        # Breaths under 100 mL do not produce a detectable capnogram.
        if current_vt < 100.0:
            current_rr = 0.0
            state.apnea = True

        vd = self.vd_deadspace
        vt_eff_spont = max(0.0, current_vt / 1000.0 - vd)
        mech_vt_l *= vent_factor
        mech_vent_mv *= vent_factor
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

        if mech_rr > 0:
            effective_vt_alv = max(vt_mech_avail, vt_spont_avail)
        else:
            effective_vt_alv = vt_spont_avail
        total_va_l_min = effective_rate * effective_vt_alv
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

        # Limit the apneic rise: fast for the first minute, then slower. Washout
        # during hyperventilation is not limited.
        if d_paco2 > 0:
            if self._apnea_timer > 0:
                rise_rate = APNEA_PACO2_RISE_SLOW_MMHG_MIN
                if self._apnea_timer <= APNEA_PACO2_RISE_FAST_DURATION_SEC:
                    rise_rate = APNEA_PACO2_RISE_FAST_MMHG_MIN
            else:
                rise_rate = APNEA_PACO2_RISE_SLOW_MMHG_MIN
            max_rise_rate = (rise_rate / 60.0) * dt
            d_paco2 = min(d_paco2, max_rise_rate)

        state.p_alveolar_co2 += d_paco2

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
        # C dPAO2/dt = VA (PIO2 - PAO2)/Pdry + inflow*FiO2 - VO2, where C is FRC gas
        # plus hemoglobin-bound O2 (steep on the dissociation curve). At steady
        # state PAO2 = PIO2 - PACO2/R; in apnea stores deplete at VO2, giving the
        # preoxygenation-dependent safe apnea time (Benumof 1997; Farmery 1996).
        # During apnea a patent airway draws gas in to replace absorbed O2
        # (apneic oxygenation).
        vo2 = paco2_base * metabolic_factor * self.va_baseline / self.rq / self._atm_dry  # L/min
        o2_ventilation = max(0.0, total_va_l_min * (1.0 - 0.6 * vq_mismatch))
        apneic_inflow = vo2 * airway_patency * clamp01_local(1.0 - o2_ventilation)
        o2_flux = (
            o2_ventilation * (fio2 * self._atm_dry - state.p_alveolar_o2) / self._atm_dry
            + apneic_inflow * fio2
            - vo2
        )
        sat = self._saturation(state.p_arterial_o2)
        dsat_dpo2 = self._n_hill * sat * (1.0 - sat) / max(state.p_arterial_o2, 1.0)
        hb = self.baseline_hb if hb_g_dl is None else max(0.0, hb_g_dl)
        o2_capacitance = self.frc / self._atm_dry + self._blood_o2_per_hb * hb * dsat_dpo2
        state.p_alveolar_o2 += o2_flux / o2_capacitance * dt / 60.0
        state.p_alveolar_o2 = clamp(state.p_alveolar_o2, 0.0, max(0.0, self._atm_dry - state.p_alveolar_co2))

        # PEEP, and to a lesser degree mean Paw above PEEP, recruit alveoli and
        # narrow the A-a gradient; V/Q mismatch widens it.
        k_peep_recruit = 0.08
        aa_grad_effective = max(3.0, self.aa_grad_base / (1.0 + k_peep_recruit * peep))
        mean_paw_effect = max(0.0, mean_paw - peep)
        aa_grad_effective = aa_grad_effective / (1.0 + self.mean_paw_recruit_gain * mean_paw_effect)
        aa_grad_effective *= (1.0 + 2.5 * vq_mismatch)
        aa_grad_effective = min(80.0, aa_grad_effective)
        # Anemia and low cardiac output lower O2 content and delivery, not PaO2.
        state.p_arterial_o2 = max(10.0, state.p_alveolar_o2 - aa_grad_effective)

        state.drive_central = drive_central
        state.muscle_factor = muscle_factor
        state.rr = current_rr
        state.vt = current_vt
        state.mv = (current_rr * current_vt) / 1000.0
        state.sao2 = 100.0 * self._saturation(state.p_arterial_o2)

        return state

    def _saturation(self, pao2: float) -> float:
        """Hill oxyhemoglobin dissociation (P50 26.6 mmHg, n 2.7) as a fraction."""
        pao2_pow = max(0.1, pao2) ** self._n_hill
        return pao2_pow / (pao2_pow + self._p50_pow)
