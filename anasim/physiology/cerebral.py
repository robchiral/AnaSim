"""Cerebral oxygen failure, persistent coma, and reversible CO2 narcosis.

Pressure below the autoregulatory range and low arterial oxygen content reduce
supply. Electrical function ramps down from 60% to complete failure at 35% of
normal supply. Severe injury accumulates below 20% supply (primate focal-ischemia
thresholds in Astrup 1981; slower when more supply remains in Jones 1981), with
slow penumbral injury modeled between 20% and 42%. Oxygen stores delay failure: after circulatory
arrest, it starts at about 4 s and the EEG is flat at about 11 s (de Vries
1998, 11 +/- 2 s to isoelectric EEG). Brief arrest allows electrical recovery,
whereas prolonged anoxia causes graded permanent cortical and brainstem injury
alongside permanent coma after 5 equivalent anoxic minutes.
"""

import math

from anasim.core.constants import TEMP_METABOLIC_COEFFICIENT
from anasim.core.utils import clamp, clamp01


class CerebralModel:
    # Time without supply after which coma persists, a teaching threshold.
    # Supply of 10-20% counts partially.
    COMA_EXPOSURE_S = 300.0
    CORTICAL_INJURY_ONSET_S = 245.0
    CORTICAL_INJURY_SPAN_S = 55.0
    BRAINSTEM_INJURY_ONSET_S = 300.0
    BRAINSTEM_INJURY_SPAN_S = 300.0
    TISSUE_OXYGEN_TAU_S = 6.0
    # CSF bicarbonate adapts over 4-12 h of sustained PaCO2 change (Brian 1998);
    # tau = 6 h is a model time constant.
    CSF_ADAPTATION_TAU_S = 6.0 * 3600.0
    # The PaCO2 and pH ramps follow Westlake 1955 (Table III) and Sieker and
    # Hickam 1956, where coma lasting over an hour lags blood-gas correction
    # (p. 397); the 30 s onset and up to 30 min recovery time constants are
    # teaching calibrations.
    CO2_RESPONSE_TAU_S = 30.0
    PROLONGED_NARCOSIS_ONSET_S = 1800.0
    PROLONGED_NARCOSIS_SPAN_S = 5400.0
    CO2_RECOVERY_MAX_TAU_S = 1800.0

    def __init__(self, baseline_hb: float, baseline_paco2: float = 40.0, baseline_map: float = 90.0):
        self.baseline_hb = baseline_hb
        self.baseline_map = baseline_map
        htn_excess = max(0.0, baseline_map - 90.0)
        self._ccp_shift = 0.25 * htn_excess
        self._cpp_span = max(1.0, (40.0 + 0.85 * htn_excess) - self._ccp_shift)
        reference_hb = max(baseline_hb, 13.0 - 0.35 * max(0.0, 13.0 - baseline_hb))
        self._inv_ref_o2 = 1.0 / (reference_hb * 98.0)
        self.csf_paco2 = max(15.0, baseline_paco2)
        self.tissue_oxygen = 1.0  # Share of normal
        self.dysfunction = 0.0
        self.anoxic_s = 0.0
        self.cortical_injury = 0.0
        self.brainstem_injury = 0.0
        self.co2_narcosis = 0.0
        self.co2_respiratory_depression = 0.0
        self.hypercapnic_coma_s = 0.0
        self.hypoglycemic_dysfunction = 0.0
        self.last_cbf_ratio = 1.0

    def step(
        self,
        dt: float,
        map_mmhg: float,
        hb: float,
        saturation: float,
        pa_co2: float,
        ph: float,
        *,
        venous_pressure: float = 10.0,
        flow_ceiling: float | None = None,
        temp_c: float = 37.0,
        anesthetic_depth: float = 0.0,
        metabolic_stress: float = 0.0,
        volatile_luxury: float = 0.0,
        ph_stat: bool = False,
        glucose_mg_dl: float = 90.0,
    ) -> None:
        if dt <= 0.0:
            return
        # Normal supply is maintained above CPP 40 mmHg (MAP 50 at ICP 10 mmHg),
        # shifted rightward in chronic hypertension. Elevated venous outflow
        # pressure reduces CPP without adding phantom gain.
        target_csf = max(15.0, pa_co2)
        if target_csf != self.csf_paco2:
            self.csf_paco2 += (target_csf - self.csf_paco2) * -math.expm1(-dt / self.CSF_ADAPTATION_TAU_S)
        outflow_pressure = max(10.0, venous_pressure)
        cpp = map_mmhg - outflow_pressure
        pressure = clamp01((cpp - self._ccp_shift) / self._cpp_span)
        cpr_cap = clamp01(flow_ceiling) if flow_ceiling is not None else None
        oxygen = max(0.0, hb * saturation) * self._inv_ref_o2
        # CMRO2 scales with temperature Q10 (~7%/°C), anesthetic suppression, and
        # cerebral hypermetabolic stress (seizures). Hypothermia slows both acute
        # oxygen store depletion and ischemic injury; normothermic anesthesia
        # preserves the ~11 s arrest EEG flattening (de Vries 1998) and does not
        # shift the cellular ischemic threshold (barbiturate focal-ischemia data in Astrup 1981).
        temp_q10 = TEMP_METABOLIC_COEFFICIENT ** (37.0 - clamp(temp_c, 22.0, 42.0)) if temp_c != 37.0 else 1.0
        ischemic_rate = max(0.20, temp_q10 * (1.0 + 0.80 * clamp01(metabolic_stress)))
        cmro2 = max(
            0.20,
            ischemic_rate * (1.0 - 0.35 * clamp01(anesthetic_depth)),
        )
        # Kaisti 2003: propofol preserves CBF-CMRO2 coupling (constant OEF),
        # whereas sevoflurane adds modest luxury perfusion (~15-18% higher CBF/CMRO2).
        # Normocapnic hypoxia dilates up to 1.5x resting CBF (calibration), hypercapnia
        # augments hypoxic vasodilation up to 2x (Battisti-Charbonney 2011; Willie 2015),
        # and hypocapnia lowers the flow ceiling until pressure limits perfusion (Brian 1998).
        resting_cbf = min(2.0, cmro2 * (1.0 + 0.18 * clamp01(volatile_luxury)))
        effective_paco2 = max(10.0, pa_co2) * 40.0 / self.csf_paco2
        hypoxic_cbf = min(2.0, resting_cbf * min(1.5, 1.0 + 1.5 * max(0.0, 1.0 - oxygen)))
        base_cbf = max(resting_cbf, hypoxic_cbf)
        if effective_paco2 >= 40.0:
            cbf = min(2.0, base_cbf + 0.5 * resting_cbf * clamp01((effective_paco2 - 40.0) / 30.0))
            supply = pressure * min(resting_cbf, cbf * oxygen)
            cbf_actual = pressure * cbf
        else:
            hypocapnia = clamp01((40.0 - effective_paco2) / 20.0)
            cbf = base_cbf * (1.0 - 0.38 * hypocapnia)
            supply = min(pressure * min(resting_cbf, base_cbf * oxygen), cbf * oxygen)
            cbf_actual = min(pressure * base_cbf, cbf)
        if ph_stat and temp_c < 36.0:
            # Murkin 1987: hypothermic pH-stat CO2 addition (~26-27 °C) uncouples CBF
            # from CMRO2 (~1.5-2.2x higher CBF) and makes CBF pressure-passive.
            hypo_blend = clamp01((36.0 - temp_c) / 6.0)
            pco2_excess = clamp01((max(10.0, pa_co2) - 40.0) / 20.0)
            dilated_cbf = cbf * (1.0 + 1.15 * hypo_blend * (0.4 + 0.6 * pco2_excess))
            passive_pressure = clamp((cpp - 5.0) / 70.0, 0.0, 1.5)
            eff_pressure = pressure + hypo_blend * (passive_pressure - pressure)
            cbf_actual = eff_pressure * dilated_cbf
            supply = min(cbf_actual * oxygen, 2.5)
        self.last_cbf_ratio = cbf_actual
        if cpr_cap is not None:
            # Passive CPR flow ceiling is referenced to normothermic awake resting CBF
            # (1.0), so hypothermic CMRO2 reduction still protects oxygen balance.
            supply = min(supply, cpr_cap * oxygen)
            self.last_cbf_ratio = min(self.last_cbf_ratio, cpr_cap)

        supply_ratio = min(1.0, supply / cmro2) if supply > 0.0 else 0.0
        if supply_ratio != self.tissue_oxygen:
            o2_tau = self.TISSUE_OXYGEN_TAU_S / max(0.35, temp_q10)
            self.tissue_oxygen += (supply_ratio - self.tissue_oxygen) * -math.expm1(-dt / o2_tau)
        # During passive CPR, hypothermia/anesthesia reduce CMRO2 and protect against
        # anoxic injury without restoring cortical wakefulness or spontaneous tidal breathing.
        electrical_o2 = min(self.tissue_oxygen, cpr_cap * oxygen) if cpr_cap is not None else self.tissue_oxygen
        target = clamp01((0.6 - electrical_o2) / 0.25)
        if self.tissue_oxygen < 0.42:
            severe_anoxia = clamp01(2.0 - self.tissue_oxygen / 0.10)
            # Prolonged low-flow CPR (20-42% supply) slowly accumulates penumbral injury
            # while still permitting short resuscitation without permanent deficit.
            penumbra_anoxia = 0.18 * (clamp01((0.42 - self.tissue_oxygen) / 0.22) ** 1.5)
            self.anoxic_s += dt * ischemic_rate * max(severe_anoxia, penumbra_anoxia)
            # Beyond ~4 min of global anoxia (the electrical-to-circulatory boundary in
            # Weisfeldt and Becker 2002), cortical suppression (Hofmeijer 2015;
            # Jørgensen and Malchow-Møller 1981) precedes caudal brainstem failure.
            self.cortical_injury = max(
                self.cortical_injury,
                clamp01((self.anoxic_s - self.CORTICAL_INJURY_ONSET_S) / self.CORTICAL_INJURY_SPAN_S),
            )
            self.brainstem_injury = max(
                self.brainstem_injury,
                clamp01((self.anoxic_s - self.BRAINSTEM_INJURY_ONSET_S) / self.BRAINSTEM_INJURY_SPAN_S),
            )
        if target != self.dysfunction:
            tau = 3.0 if target > self.dysfunction else 30.0
            self.dysfunction += (target - self.dysfunction) * -math.expm1(-dt / tau)
        # Acute hypercapnia impairs response more with acidemia; compensated
        # retention does so only above 100 mmHg. Gating pH by PaCO2 keeps
        # metabolic acidosis from causing narcosis.
        if pa_co2 > 60.0:
            hypercapnia = clamp01((pa_co2 - 60.0) / 40.0)
            acidemia = clamp01((7.3 - ph) / 0.2)
            target_co2 = max(clamp01((pa_co2 - 100.0) / 40.0), hypercapnia * acidemia)
        else:
            target_co2 = 0.0
        prolonged = (
            clamp01((self.hypercapnic_coma_s - self.PROLONGED_NARCOSIS_ONSET_S) / self.PROLONGED_NARCOSIS_SPAN_S)
            if self.hypercapnic_coma_s > self.PROLONGED_NARCOSIS_ONSET_S
            else 0.0
        )
        recovery_tau = self.CO2_RESPONSE_TAU_S + (self.CO2_RECOVERY_MAX_TAU_S - self.CO2_RESPONSE_TAU_S) * prolonged
        if target_co2 != self.co2_narcosis:
            co2_tau = self.CO2_RESPONSE_TAU_S if target_co2 >= self.co2_narcosis else recovery_tau
            self.co2_narcosis += (target_co2 - self.co2_narcosis) * -math.expm1(-dt / co2_tau)
        if self.co2_narcosis >= 0.5 and target_co2 >= 0.5:
            self.hypercapnic_coma_s = min(
                self.PROLONGED_NARCOSIS_ONSET_S + self.PROLONGED_NARCOSIS_SPAN_S,
                self.hypercapnic_coma_s + dt * clamp01((self.co2_narcosis - 0.5) / 0.3),
            )
        elif target_co2 < 0.5 and self.hypercapnic_coma_s > 0.0:
            self.hypercapnic_coma_s *= math.exp(-dt / (2.0 * self.CO2_RECOVERY_MAX_TAU_S))
        # Deeper hypercapnic coma depresses medullary breathing toward shallow
        # hypoventilation (Westlake 1955; Sieker and Hickam 1956).
        if pa_co2 > 90.0:
            medullary_co2 = max(
                clamp01((pa_co2 - 110.0) / 35.0),
                clamp01((pa_co2 - 90.0) / 40.0) * clamp01((7.22 - ph) / 0.14),
            )
            target_resp_dep = 0.95 * medullary_co2 * medullary_co2
        else:
            target_resp_dep = 0.0
        if target_resp_dep != self.co2_respiratory_depression:
            resp_tau = self.CO2_RESPONSE_TAU_S if target_resp_dep >= self.co2_respiratory_depression else recovery_tau
            self.co2_respiratory_depression += (
                (target_resp_dep - self.co2_respiratory_depression) * -math.expm1(-dt / resp_tau)
            )
        # Reversible neuroglycopenic cortical slowing and coma below a calibrated 42 mg/dL
        # threshold (awake T1DM EEG slowing at median 36 mg/dL, nadir 25 mg/dL in Pramming 1988;
        # cognitive dysfunction near 43-49 mg/dL in Mitrakou 1991; 120 s / 300 s onset/recovery
        # time constants calibrated to 8 min prehospital recovery in Moore and Woollard 2005).
        target_hypo = clamp01((42.0 - glucose_mg_dl) / 22.0) if glucose_mg_dl < 42.0 else 0.0
        if target_hypo != self.hypoglycemic_dysfunction:
            hypo_tau = 120.0 if target_hypo > self.hypoglycemic_dysfunction else 300.0
            self.hypoglycemic_dysfunction += (
                (target_hypo - self.hypoglycemic_dysfunction) * -math.expm1(-dt / hypo_tau)
            )
            if self.hypoglycemic_dysfunction < 1e-6 and target_hypo == 0.0:
                self.hypoglycemic_dysfunction = 0.0

    @property
    def cortical_function(self) -> float:
        """Cortical electrical function for BIS; CO2 narcosis leaves it unchanged (Turnbull 2008)."""
        return (
            (1.0 - self.dysfunction)
            * (1.0 - self.cortical_injury)
            * (1.0 - 0.85 * self.hypoglycemic_dysfunction)
        )

    @property
    def respiratory_drive(self) -> float:
        """Medullary drive remaining after severe hypoxia/ischemia, brainstem injury, or extreme CO2 narcosis."""
        return (
            (1.0 - self.dysfunction ** 8)
            * (1.0 - self.brainstem_injury)
            * (1.0 - self.co2_respiratory_depression)
            * (1.0 - 0.35 * (self.hypoglycemic_dysfunction ** 4))
        )

    @property
    def unresponsiveness(self) -> float:
        """Loss of response from oxygen failure, postanoxic injury, CO2 narcosis, or severe hypoglycemia."""
        if self.anoxic_s >= self.COMA_EXPOSURE_S:
            return 1.0
        return 1.0 - (
            (1.0 - self.dysfunction)
            * (1.0 - self.cortical_injury)
            * (1.0 - self.co2_narcosis)
            * (1.0 - self.hypoglycemic_dysfunction)
        )
