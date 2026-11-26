"""Membrane oxygenator O2, CO2, and volatile anesthetic mass transfer.

Parameterized against:
- Joyce 2018 and Zanella 2016 (mathematical models) for coupled membrane/native lung mass balance
- Schmidt 2013 (adult ARDS) and Park 2013 (porcine MODS) for blood-flow, FiO2, and sweep-gas titration
- Karagiannidis 2017 (porcine acidosis) for membrane surface-area and low-blood-flow CO2 ceilings
- Broman 2015 (simulation and clinical) and Palmér 2016 (atrio-femoral cohort) for VV ECMO recirculation
- Prasser 2008 and Nitzschke 2013 for PMP vs. microporous volatile transfer
"""

from __future__ import annotations

import math

from anasim.core.utils import clamp, clamp01
from anasim.physiology.blood_chemistry import (
    MOLAR_GAS_VOLUME_STPD_L_MMOL,
    BloodChemistry,
)
from anasim.physiology.blood_gas_math import (
    acid_base_p50,
    oxygen_content,
    resolve_blood_contents,
    temperature_o2_solubility,
    temperature_p50,
    temperature_pco2_scale,
)

from .types import CannulationType, MembraneType, OxygenatorPassResult

_CO2_L_PER_MMOL = MOLAR_GAS_VOLUME_STPD_L_MMOL
_ATM_DRY_MMHG = 713.0  # 760 - 47 mmHg water vapor at 37 °C
# Dry room-temperature (20 °C) sweep flowmeter L/min converted to STPD L/min.
_SWEEP_STPD_PER_L = 273.15 / 293.15
_SWEEP_GAS_CAP_PER_L_MMHG = _SWEEP_STPD_PER_L / _ATM_DRY_MMHG
# Hollow-fiber diffusing conductances per m^2 membrane surface area (STPD L/min/mmHg/m^2):
# Calibrated so 0.4 m^2 caps CO2 removal near ~72 mL/min and >=0.8-1.3 m^2 reaches ~150 mL/min
# at 1 L/min blood flow in severe hypercapnia (porcine data in Karagiannidis 2017), while adult
# 1.8 m^2 oxygenators are sweep-flow-limited for CO2 and blood-flow-limited for O2 up to rated flow
# (Schmidt 2013; porcine data in Park 2013).
_D_CO2_PER_M2 = 0.0031  # STPD L/min/mmHg/m^2
_D_O2_PER_M2 = 0.0018  # STPD L/min/mmHg/m^2
_RATED_FLOW_PER_M2 = 3.9  # L/min per m^2 (~7.0 L/min for 1.8 m^2 Quadrox/HLS)
_BASE_MEMBRANE_SHUNT = 0.03  # Default intrinsic blood-path shunt (model calibration; structure per Zanella 2016)

# Calibrated cannulation recirculation geometry presets informed by Broman 2015 (simulation)
# and Palmér 2016 (atrio-femoral cohort: multistage 19.0 +/- 12.2% vs conventional 38.0 +/- 13.7%).
_RECIRC_GEOM = {
    CannulationType.DUAL_LUMEN: 0.11,
    CannulationType.FEM_JUG: 0.28,
    CannulationType.FEM_FEM: 0.56,
}


def recirculation_fraction(
    blood_flow_l_min: float,
    native_co_l_min: float,
    cannulation: CannulationType = CannulationType.FEM_JUG,
) -> float:
    """Return VV ECMO recirculation fraction R in [0, 1.0] (Broman 2015; Palmér 2016).

    Ensures effective non-recirculated flow (1 - R) * Qb remains strictly below
    native venous return so mass conservation holds even when pump flow exceeds CO,
    and reaches 1.0 when native cardiac output stops (no net systemic delivery).
    """
    qb = max(0.0, blood_flow_l_min)
    if qb <= 1e-6:
        return 0.0
    if native_co_l_min <= 1e-6:
        return 1.0
    ratio = qb / native_co_l_min
    k = _RECIRC_GEOM.get(cannulation, 0.28)
    raw = 1.0 - 1.0 / (1.0 + k * (ratio ** 1.30) + 0.12 * (ratio ** 3.0))
    # Physical ceiling: non-recirculated flow cannot exceed native venous return.
    if ratio > 0.90:
        min_r = 1.0 - 0.95 / ratio
        if min_r > raw:
            excess = min_r - raw
            raw = min_r - 0.02 * math.exp(-excess / 0.02)
            raw = max(raw, 1.0 - 0.99 / ratio)
    return clamp(raw, 0.0, 1.0)


def volatile_membrane_permeability(membrane_type: MembraneType) -> float:
    """Relative volatile anesthetic transfer permeability (Prasser 2008).

    Microporous polypropylene CPB oxygenators equilibrate volatile agents across
    the gas phase; plasma-tight PMP ECMO oxygenators transmit minimal volatile
    gas (~3-fold higher retained blood sevoflurane at 30-60 min with zero measurable exhaust).
    """
    if membrane_type == MembraneType.MICROPOROUS:
        return 1.0
    return 0.04


class MembraneOxygenator:
    """Two-stage crossflow hollow-fiber membrane lung mass-transfer solver."""

    STAGES = 2

    def __init__(self, area_m2: float = 1.8):
        self.area_m2 = max(0.2, float(area_m2))
        self._last_out_pco2 = 35.0
        self._last_out_ph = 7.42
        self._last_out_po2 = 400.0

    def _solve_stage(
        self,
        qb_eff: float,
        qs_stage: float,
        area_stage: float,
        sweep_fio2: float,
        sweep_fco2: float,
        co2_in: float,
        o2_in: float,
        base_in: float,
        hb: float,
        temp_c: float,
        chemistry: BloodChemistry,
        pco2_guess: float,
        ph_guess: float,
        po2_guess: float,
        temp_p50_now: float,
        pco2_scale: float,
        o2_sol: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Advance blood through one crossflow stage; return (co2, o2, pco2_37, ph_37, po2, sat)."""
        if qb_eff <= 1e-9 or qs_stage <= 1e-9:
            pco2, ph, po2, sat, _, _ = resolve_blood_contents(
                chemistry,
                co2_in,
                o2_in,
                base_in,
                hb,
                temp_c,
                pco2_guess,
                ph_guess,
                po2_guess,
                temp_p50=temp_p50_now,
                pco2_scale=pco2_scale,
                solubility=o2_sol,
            )
            return co2_in, o2_in, pco2, ph, po2, sat

        flow_boundary = (max(0.25, qb_eff) / 2.5) ** 0.10
        d_co2 = _D_CO2_PER_M2 * area_stage * flow_boundary
        d_o2 = _D_O2_PER_M2 * area_stage * flow_boundary
        gas_cap = qs_stage * _SWEEP_GAS_CAP_PER_L_MMHG
        k_co2 = (d_co2 * gas_cap) / (d_co2 + gas_cap)
        pg_co2_in = clamp01(sweep_fco2) * _ATM_DRY_MMHG
        sweep_total_stpd = qs_stage * _SWEEP_STPD_PER_L
        sweep_o2_in_stpd = sweep_total_stpd * clamp01(sweep_fio2)

        pco2_37 = max(1.0, pco2_guess)
        ph_37 = clamp(ph_guess, 6.2, 8.2)
        po2_out = max(1.0, po2_guess)
        sat_out = 0.98
        co2_out = co2_in
        o2_out = o2_in

        # Coupled outer fixed-point iterations (Bohr/Haldane coupling converges in 2-3 steps).
        for _ in range(4):
            # 1) Solve CO2 transfer for exiting PCO2_37 at current O2 saturation.
            # f(P37) = qb_eff * (co2_in - C(P37)) * CO2_L_PER_MMOL - k_co2 * (P37 * pco2_scale - pg_co2_in) = 0
            # Note: C(P37) increases monotonically with P37, so f(P37) decreases monotonically.
            low_p = 1.0
            high_p = max(150.0, pco2_guess * 2.0, (pg_co2_in / max(0.2, pco2_scale)) + 20.0)
            p_curr = clamp(pco2_37, low_p, high_p)
            ph_curr = ph_37
            for _ in range(18):
                blood = chemistry.solve(p_curr, sat_out, base_in, ph_curr)
                ph_curr = blood.ph
                co2_c = blood.total_co2
                # dC/dP37 at fixed buffer base = co2_capacity (mmol/L/mmHg)
                dc_dp = max(1e-6, blood.co2_capacity)
                vco2_blood = qb_eff * (co2_in - co2_c) * _CO2_L_PER_MMOL
                vco2_mem = k_co2 * (p_curr * pco2_scale - pg_co2_in)
                res = vco2_blood - vco2_mem
                if abs(res) < 1e-8:
                    break
                if res > 0.0:
                    low_p = p_curr
                else:
                    high_p = p_curr
                dres_dp = -(qb_eff * dc_dp * _CO2_L_PER_MMOL + k_co2 * pco2_scale)
                cand = p_curr - res / dres_dp
                p_curr = cand if low_p < cand < high_p else 0.5 * (low_p + high_p)
            pco2_37 = p_curr
            ph_37 = ph_curr
            blood_co2 = chemistry.solve(pco2_37, sat_out, base_in, ph_37)
            co2_out = blood_co2.total_co2
            vco2_stage = qb_eff * (co2_in - co2_out) * _CO2_L_PER_MMOL

            # 2) Solve O2 transfer for exiting PO2 at updated (pH, PCO2).
            # g(PO2) = qb_eff * (C_O2(PO2) - o2_in) - d_o2 * (Pg_O2(VO2) - PO2) = 0
            # Strictly increasing with PO2 -> guaranteed unique root in [0, ATM_DRY].
            p50 = acid_base_p50(ph_37, pco2_37, temp_c, temp_p50=temp_p50_now, pco2_scale=pco2_scale)
            lo_o2, hi_o2 = 0.0, _ATM_DRY_MMHG
            po2_curr = clamp(po2_out, 1e-3, hi_o2)
            for _ in range(22):
                c_o2, dc_do2 = oxygen_content(po2_curr, hb, p50, o2_sol)
                vo2 = qb_eff * (c_o2 - o2_in)
                # Gas-phase O2 partial pressure after O2 uptake and CO2 efflux into sweep gas.
                gas_out_total = max(1e-6, sweep_total_stpd - vo2 + vco2_stage)
                pg_o2 = clamp((sweep_o2_in_stpd - vo2) / gas_out_total * _ATM_DRY_MMHG, 0.0, _ATM_DRY_MMHG)
                res_o2 = vo2 - d_o2 * (pg_o2 - po2_curr)
                if abs(res_o2) < 1e-9:
                    break
                if res_o2 > 0.0:
                    hi_o2 = po2_curr
                else:
                    lo_o2 = po2_curr
                # Derivative d(pg_o2)/d(vo2) <= 0, so d(res_o2)/d(po2) > 0 strictly.
                dvo2_dp = qb_eff * dc_do2
                dpg_dvo2 = (
                    -_ATM_DRY_MMHG * (gas_out_total - (sweep_o2_in_stpd - vo2)) / (gas_out_total * gas_out_total)
                    if 0.0 < pg_o2 < _ATM_DRY_MMHG
                    else 0.0
                )
                dres_do2 = dvo2_dp * (1.0 - d_o2 * dpg_dvo2) + d_o2
                cand = po2_curr - res_o2 / max(1e-9, dres_do2)
                po2_curr = cand if lo_o2 < cand < hi_o2 else 0.5 * (lo_o2 + hi_o2)
            po2_out = po2_curr
            o2_out, _ = oxygen_content(po2_out, hb, p50, o2_sol)
            new_sat = clamp01((o2_out - o2_sol * po2_out) / max(1e-9, 0.0134 * hb)) if hb > 0.0 else 0.0
            if abs(new_sat - sat_out) < 1e-5:
                sat_out = new_sat
                break
            sat_out = new_sat

        return co2_out, o2_out, pco2_37, ph_37, po2_out, sat_out

    def solve_pass(
        self,
        blood_flow_l_min: float,
        sweep_gas_l_min: float,
        sweep_fio2: float,
        co2_in_mmol_l: float,
        o2_in_l_l: float,
        base_in_meq_l: float,
        hb_g_dl: float,
        temp_c: float,
        chemistry: BloodChemistry,
        *,
        sweep_fco2: float = 0.0,
        area_m2: float | None = None,
        pco2_in_guess: float = 46.0,
        ph_in_guess: float = 7.36,
        po2_in_guess: float = 40.0,
    ) -> OxygenatorPassResult:
        """Compute exiting blood composition and O2/CO2 transfer across the membrane lung."""
        qb = max(0.0, float(blood_flow_l_min))
        qs = max(0.0, float(sweep_gas_l_min))
        area = self.area_m2 if area_m2 is None else max(0.2, float(area_m2))
        temp_p50_now = temperature_p50(temp_c)
        pco2_scale = temperature_pco2_scale(temp_c)
        o2_sol = temperature_o2_solubility(temp_c)

        pre_pco2_37, pre_ph_37, pre_po2, pre_sat, _, _ = resolve_blood_contents(
            chemistry,
            co2_in_mmol_l,
            o2_in_l_l,
            base_in_meq_l,
            hb_g_dl,
            temp_c,
            pco2_in_guess,
            ph_in_guess,
            po2_in_guess,
            temp_p50=temp_p50_now,
            pco2_scale=pco2_scale,
            solubility=o2_sol,
        )
        if qb <= 1e-6 or qs <= 1e-6:
            return OxygenatorPassResult(
                co2_out_mmol_l=co2_in_mmol_l,
                o2_out_l_l=o2_in_l_l,
                pco2_out_37=pre_pco2_37,
                ph_out_37=pre_ph_37,
                po2_out=pre_po2,
                so2_out=pre_sat,
                o2_transfer_l_min=0.0,
                co2_removal_l_min=0.0,
                pre_po2=pre_po2,
                pre_so2=pre_sat * 100.0,
                post_po2=pre_po2,
                post_so2=pre_sat * 100.0,
                pre_pco2_37=pre_pco2_37,
                pre_ph_37=pre_ph_37,
            )

        # Intrinsic + transit-time shunt rises smoothly above the oxygenator's rated flow.
        rated_flow = _RATED_FLOW_PER_M2 * area
        overload = max(0.0, qb / max(0.5, rated_flow) - 1.0)
        shunt = clamp(_BASE_MEMBRANE_SHUNT + 0.08 * (overload ** 1.4), 0.0, 0.35)
        qb_eff = qb * (1.0 - shunt)

        co2_s = co2_in_mmol_l
        o2_s = o2_in_l_l
        pco2_g = self._last_out_pco2
        ph_g = self._last_out_ph
        po2_g = self._last_out_po2
        area_stage = area / self.STAGES
        qs_stage = qs / self.STAGES
        for _ in range(self.STAGES):
            co2_s, o2_s, pco2_g, ph_g, po2_g, _ = self._solve_stage(
                qb_eff,
                qs_stage,
                area_stage,
                sweep_fio2,
                sweep_fco2,
                co2_s,
                o2_s,
                base_in_meq_l,
                hb_g_dl,
                temp_c,
                chemistry,
                pco2_g,
                ph_g,
                po2_g,
                temp_p50_now,
                pco2_scale,
                o2_sol,
            )

        # Recombine non-shunted and shunted blood streams by exact content conservation.
        co2_out = (1.0 - shunt) * co2_s + shunt * co2_in_mmol_l
        o2_out = (1.0 - shunt) * o2_s + shunt * o2_in_l_l
        pco2_out_37, ph_out_37, po2_out, sat_out, _, _ = resolve_blood_contents(
            chemistry,
            co2_out,
            o2_out,
            base_in_meq_l,
            hb_g_dl,
            temp_c,
            pco2_g,
            ph_g,
            po2_g,
            temp_p50=temp_p50_now,
            pco2_scale=pco2_scale,
            solubility=o2_sol,
        )
        self._last_out_pco2 = pco2_out_37
        self._last_out_ph = ph_out_37
        self._last_out_po2 = po2_out

        o2_transfer_l_min = qb * (o2_out - o2_in_l_l)
        co2_removal_l_min = qb * (co2_in_mmol_l - co2_out) * _CO2_L_PER_MMOL
        return OxygenatorPassResult(
            co2_out_mmol_l=co2_out,
            o2_out_l_l=o2_out,
            pco2_out_37=pco2_out_37,
            ph_out_37=ph_out_37,
            po2_out=po2_out,
            so2_out=sat_out,
            o2_transfer_l_min=o2_transfer_l_min,
            co2_removal_l_min=co2_removal_l_min,
            pre_po2=pre_po2,
            pre_so2=pre_sat * 100.0,
            post_po2=po2_out,
            post_so2=sat_out * 100.0,
            pre_pco2_37=pre_pco2_37,
            pre_ph_37=pre_ph_37,
        )

    def solve_vv_exchange(
        self,
        blood_flow_l_min: float,
        native_co_l_min: float,
        sweep_gas_l_min: float,
        sweep_fio2: float,
        venous_co2_mmol_l: float,
        venous_o2_l_l: float,
        base_meq_l: float,
        hb_g_dl: float,
        temp_c: float,
        chemistry: BloodChemistry,
        *,
        cannulation: CannulationType = CannulationType.FEM_JUG,
        sweep_fco2: float = 0.0,
        area_m2: float | None = None,
        pco2_v_guess: float = 46.0,
        ph_v_guess: float = 7.36,
        po2_v_guess: float = 40.0,
    ) -> tuple[OxygenatorPassResult, float]:
        """Solve VV ECMO oxygenator pass with self-consistent venous recirculation.

        In VV ECMO, recirculated post-oxygenator blood mixes with systemic venous
        blood at the drainage cannula: C_pre = (1 - R) * C_v + R * C_post.
        Returns (pass_result, recirculation_fraction).
        """
        recirc = recirculation_fraction(blood_flow_l_min, native_co_l_min, cannulation)
        if recirc >= 1.0 - 1e-9:
            # Zero native venous return: recirculating blood yields zero net patient transfer.
            result = self.solve_pass(
                0.0,
                sweep_gas_l_min,
                sweep_fio2,
                venous_co2_mmol_l,
                venous_o2_l_l,
                base_meq_l,
                hb_g_dl,
                temp_c,
                chemistry,
                sweep_fco2=sweep_fco2,
                area_m2=area_m2,
                pco2_in_guess=pco2_v_guess,
                ph_in_guess=ph_v_guess,
                po2_in_guess=po2_v_guess,
            )
            return result, 1.0
        if recirc <= 1e-6 or blood_flow_l_min <= 1e-6 or sweep_gas_l_min <= 1e-6:
            result = self.solve_pass(
                blood_flow_l_min,
                sweep_gas_l_min,
                sweep_fio2,
                venous_co2_mmol_l,
                venous_o2_l_l,
                base_meq_l,
                hb_g_dl,
                temp_c,
                chemistry,
                sweep_fco2=sweep_fco2,
                area_m2=area_m2,
                pco2_in_guess=pco2_v_guess,
                ph_in_guess=ph_v_guess,
                po2_in_guess=po2_v_guess,
            )
            return result, recirc

        co2_post = venous_co2_mmol_l
        o2_post = venous_o2_l_l
        pass_res: OxygenatorPassResult | None = None
        for _ in range(40):
            co2_pre = (1.0 - recirc) * venous_co2_mmol_l + recirc * co2_post
            o2_pre = (1.0 - recirc) * venous_o2_l_l + recirc * o2_post
            pass_res = self.solve_pass(
                blood_flow_l_min,
                sweep_gas_l_min,
                sweep_fio2,
                co2_pre,
                o2_pre,
                base_meq_l,
                hb_g_dl,
                temp_c,
                chemistry,
                sweep_fco2=sweep_fco2,
                area_m2=area_m2,
                pco2_in_guess=pco2_v_guess,
                ph_in_guess=ph_v_guess,
                po2_in_guess=po2_v_guess,
            )
            if (
                abs(pass_res.co2_out_mmol_l - co2_post) < 1e-6
                and abs(pass_res.o2_out_l_l - o2_post) < 1e-7
            ):
                break
            co2_post = pass_res.co2_out_mmol_l
            o2_post = pass_res.o2_out_l_l
        assert pass_res is not None
        q_eff = blood_flow_l_min * (1.0 - recirc)
        return OxygenatorPassResult(
            co2_out_mmol_l=pass_res.co2_out_mmol_l,
            o2_out_l_l=pass_res.o2_out_l_l,
            pco2_out_37=pass_res.pco2_out_37,
            ph_out_37=pass_res.ph_out_37,
            po2_out=pass_res.po2_out,
            so2_out=pass_res.so2_out,
            o2_transfer_l_min=q_eff * (pass_res.o2_out_l_l - venous_o2_l_l),
            co2_removal_l_min=q_eff * (venous_co2_mmol_l - pass_res.co2_out_mmol_l) * _CO2_L_PER_MMOL,
            pre_po2=pass_res.pre_po2,
            pre_so2=pass_res.pre_so2,
            post_po2=pass_res.post_po2,
            post_so2=pass_res.post_so2,
            pre_pco2_37=pass_res.pre_pco2_37,
            pre_ph_37=pass_res.pre_ph_37,
        ), recirc
