import math
from dataclasses import dataclass
from typing import NamedTuple

from anasim.core.utils import clamp, clamp01

from .patient import Patient
from .pk_models import janmahasatian_ffm


class _IntegrationResult(NamedTuple):
    p_alv: float
    p_vrg: float
    p_brain: float
    p_mus: float
    p_fat: float
    p_ven: float
    p_art: float
    v_frc: float
    net_btps_l: float
    blood_uptake_btps_l: float
    exhaust_fraction: float = 0.0


@dataclass(slots=True)
class VolatileState:
    """Partial pressures as fractions of an atmosphere (0.02 is 2%).

    At equilibrium every compartment has the arterial partial pressure.
    """

    p_alv: float = 0.0
    p_art: float = 0.0
    p_ven: float = 0.0  # Mixed venous
    p_vrg: float = 0.0  # Vessel-rich group, including brain
    p_brain: float = 0.0  # Autoregulated brain compartment for MAC effect site
    p_mus: float = 0.0
    p_fat: float = 0.0
    mac: float = 0.0  # Brain MAC: cerebral partial pressure over age-, pregnancy-, and temperature-adjusted MAC


class VolatilePK:
    """Physiologic inhaled-agent model with lung, VRG, muscle, and fat
    compartments (Davis and Mapleson 1981; Yasuda 1991)."""

    def __init__(
        self,
        patient: Patient,
        name: str,
        lambda_b_g: float,
        lambda_t_b_vrg: float = 1.6,
        mac_40: float = 2.0,
        lambda_t_b_mus: float | None = None,
        lambda_t_b_fat: float | None = None,
    ):
        self.patient = patient
        self.name = name
        self.lambda_b_g = lambda_b_g
        self.lambda_t_b_vrg = lambda_t_b_vrg
        self.lambda_t_b_mus = 2.5 if lambda_t_b_mus is None else lambda_t_b_mus
        self.lambda_t_b_fat = 50.0 if lambda_t_b_fat is None else lambda_t_b_fat

        # Age- and pregnancy-adjusted MAC (Mapleson 1996; Gin and Chan 1994).
        # Monitor gas analyzers display age-adjusted EtMAC without pregnancy scaling.
        self.mac_40 = mac_40
        self.mac_age_monitor = self.mac_40 * (10 ** (-0.00269 * (self.patient.age - 40)))
        self.mac_age = self.mac_age_monitor * self.patient.pregnancy.mac_factor
        # Temperature dependence of blood:gas solubility (Lockwood 1997: -4.5%/°C sevoflurane).
        self._lambda_temp_slope = 0.045 if name.lower().startswith("sevo") else 0.020

        # Tissue volumes (L) and shares of cardiac output: lean organs scale
        # with fat-free mass, while excess body mass expands the adipose depot.
        male = self.patient.sex == "male"
        ffm_scale = (
            janmahasatian_ffm(self.patient.weight, self.patient.height, male)
            / janmahasatian_ffm(70.0, 170.0, True)
        )
        self._base_co = 5.0 * ffm_scale
        self.v_vrg = 7.0 * ffm_scale
        self.v_mus = 35.0 * ffm_scale
        self.v_fat = max(3.0, 14.0 * ffm_scale + 0.8 * max(0.0, self.patient.weight - 70.0 * ffm_scale))
        self.f_vrg_frac = 0.75
        self.f_mus_frac = 0.19
        self.f_fat_frac = 0.06

        self.state = VolatileState()
        self.frc = patient.functional_residual_capacity()
        self._temp_c = float(patient.baseline_temp)
        self.last_blood_uptake_l_min = 0.0
        self.last_exhaust_fraction = 0.0

    def flow_fractions(self, cardiac_output_l: float) -> tuple[float, float, float]:
        """Organ CO shares; low CO shifts flow from muscle and fat toward VRG."""
        co_ratio = clamp01(cardiac_output_l / max(0.5, self._base_co))
        shift = 0.70 * clamp01((1.0 - co_ratio) / 0.65)
        f_mus = self.f_mus_frac * (1.0 - shift)
        f_fat = self.f_fat_frac * (1.0 - shift)
        return 1.0 - f_mus - f_fat, f_mus, f_fat

    def mac_at_temp(self, temp_c: float = 37.0) -> float:
        """Age-, pregnancy-, and temperature-adjusted MAC (% atm).

        Falls exponentially with cooling (~6%/°C, calibrated to rat halothane/isoflurane
        MAC data in Vitez 1974), remaining strictly monotonic with blood:gas solubility
        down to deep hypothermic CPB (18 °C).
        """
        clamped_t = clamp(temp_c, 18.0, 42.0)
        return self.mac_age * math.exp(-0.060 * (37.0 - clamped_t))

    def lambda_b_g_at_temp(self, temp_c: float = 37.0) -> float:
        """Blood:gas partition coefficient at core temperature (Lockwood 1997)."""
        if temp_c == 37.0:
            return self.lambda_b_g
        return self.lambda_b_g * math.exp(-self._lambda_temp_slope * (temp_c - 37.0))

    def alveolar_uptake_l_min(
        self,
        cardiac_output_l: float,
        shunt_fraction: float = 0.0,
        temp_c: float = 37.0,
        *,
        pulmonary_flow_l: float | None = None,
    ) -> float:
        """Instantaneous pulmonary capillary uptake from alveoli (positive into blood, L/min)."""
        q_pulm = cardiac_output_l if pulmonary_flow_l is None else pulmonary_flow_l
        lambda_q = max(0.0, q_pulm) * (1.0 - shunt_fraction) * self.lambda_b_g_at_temp(temp_c)
        return lambda_q * (self.state.p_alv - self.state.p_ven)

    def _integrate(
        self,
        dt: float,
        fi_agent: float,
        alveolar_vent_l: float,
        cardiac_output_l: float,
        lung_volume_l: float | None,
        shunt_fraction: float,
        companion_uptake_l_min: float,
        passive_inflow_l_min: float,
        exhaled_volume_change_l: float | None,
        circuit_share: float = 1.0,
        circuit_fi_agent: float | None = None,
        temp_c: float = 37.0,
        pulmonary_flow_l: float | None = None,
        ecmo_blood_flow_l: float = 0.0,
        ecmo_sweep_l: float = 0.0,
        ecmo_fi_agent: float = 0.0,
        ecmo_permeability: float = 0.0,
        ecmo_is_arterial: bool = False,
    ) -> _IntegrationResult:
        dt_min = dt / 60.0
        q_co = max(0.0, cardiac_output_l)
        q_pulm = q_co if pulmonary_flow_l is None else min(q_co, max(0.0, pulmonary_flow_l))
        f_vrg, f_mus, f_fat = self.flow_fractions(q_co)
        q_vrg = q_co * f_vrg
        q_mus = q_co * f_mus
        q_fat = q_co * f_fat
        alveolar_vent = max(0.0, alveolar_vent_l)
        branch_fi = fi_agent / circuit_share if circuit_fi_agent is None else circuit_fi_agent
        concentrating_inflow = (
            (max(0.0, companion_uptake_l_min) + max(0.0, passive_inflow_l_min)) if fi_agent > 0.0 else 0.0
        )
        washout_outflow = max(0.0, -companion_uptake_l_min)
        p_alv = self.state.p_alv
        p_art = self.state.p_art
        p_vrg = self.state.p_vrg
        p_brain = self.state.p_brain
        p_mus = self.state.p_mus
        p_fat = self.state.p_fat
        # Conserve dissolved body volatile mass when temperature alters solubility.
        # Lipid (fat) solubility varies roughly half as steeply with temperature as blood.
        lambda_bg_now = self.lambda_b_g_at_temp(temp_c)
        if temp_c != self._temp_c and lambda_bg_now > 0.0:
            sol_scale = self.lambda_b_g_at_temp(self._temp_c) / lambda_bg_now
            p_art = clamp01(p_art * sol_scale)
            p_vrg = clamp01(p_vrg * sol_scale)
            p_brain = clamp01(p_brain * sol_scale)
            p_mus = clamp01(p_mus * sol_scale)
            p_fat = clamp01(p_fat * math.sqrt(sol_scale))
        v_frc = self.frc
        net_btps_l = 0.0
        blood_uptake_btps_l = 0.0
        # FRC dPalv/dt = VA (Fi - Palv) + U_companion Fi + (1-s) Q lambda_bg (Pven - Palv).
        # Arterial blood mixes end-capillary and shunted venous blood.
        if lung_volume_l is not None:
            airway_change = lung_volume_l - v_frc if exhaled_volume_change_l is None else exhaled_volume_change_l
            if airway_change > 0.0:
                net_btps_l += circuit_share * airway_change * branch_fi
                agent_vol = v_frc * p_alv + airway_change * fi_agent
            elif airway_change < 0.0:
                net_btps_l += circuit_share * airway_change * p_alv
                agent_vol = max(0.0, (v_frc + airway_change) * p_alv)
            else:
                agent_vol = v_frc * p_alv
            v_frc = lung_volume_l
            p_alv = clamp01(agent_vol / v_frc) if v_frc > 0.0 else 0.0
        one_minus_shunt = 1.0 - shunt_fraction
        lambda_q = q_pulm * one_minus_shunt * lambda_bg_now
        ecmo_active = (
            ecmo_blood_flow_l > 1e-6
            and ecmo_sweep_l > 1e-6
            and ecmo_permeability > 1e-6
            and lambda_bg_now > 0.0
        )
        q_ecmo = max(0.0, ecmo_blood_flow_l)
        ecmo_equil = 0.0
        exhaust_gain = 0.0
        if ecmo_active:
            # Series blood-membrane-sweep conductance (Prasser 2008; Nitzschke 2013):
            # microporous PPL equilibrates rapidly; plasma-tight PMP blocks volatile gas.
            c_b = q_ecmo * lambda_bg_now
            g_mem = ecmo_permeability * 3.0 * c_b
            eff_g = (g_mem * ecmo_sweep_l) / (g_mem + ecmo_sweep_l)
            k_vol = (c_b * eff_g) / (c_b + eff_g)
            ecmo_equil = clamp01(k_vol / c_b)
            exhaust_gain = k_vol / ecmo_sweep_l
        # Tissue rate constant k = (flow / volume) / lambda (1/min). A high
        # partition coefficient slows equilibration: VRG in minutes, fat in hours.
        # Fat solubility scales with sqrt(lambda_b_g), so fat:blood partition scales
        # inversely with sqrt(lambda_b_g) to conserve dissolved mass across temperature.
        fat_tb_scale = math.sqrt(self.lambda_b_g / lambda_bg_now) if lambda_bg_now > 0.0 else 1.0
        k_vrg = (q_vrg / self.v_vrg) / self.lambda_t_b_vrg
        k_mus = (q_mus / self.v_mus) / self.lambda_t_b_mus
        k_fat = (q_fat / self.v_fat) / (self.lambda_t_b_fat * fat_tb_scale)
        # Cerebral autoregulation preserves brain equilibration rate during
        # moderate low-CO shock so faster alveolar wash-in accelerates brain MAC.
        if q_co <= 0.0:
            k_brain = 0.0
        elif q_co >= self._base_co:
            k_brain = k_vrg
        else:
            co_ratio = q_co / max(0.5, self._base_co)
            autoreg = min(1.0, (co_ratio / 0.35) ** 0.75)
            k_brain = max(k_vrg, (self._base_co * self.f_vrg_frac * autoreg / self.v_vrg) / self.lambda_t_b_vrg)
        # Update all compartments from the same pressures so agent is
        # conserved; substep so no Euler weight exceeds 0.05.
        alveolar_rate = (
            (alveolar_vent + concentrating_inflow + washout_outflow + lambda_q) / v_frc
            if v_frc > 0.0
            else 0.0
        )
        fastest_rate = max(alveolar_rate, k_vrg, k_brain, k_mus, k_fat)
        substeps = max(1, math.ceil(dt_min * fastest_rate / 0.05))
        interval = dt_min / substeps
        k_vrg_dt = k_vrg * interval
        k_brain_dt = k_brain * interval
        k_mus_dt = k_mus * interval
        k_fat_dt = k_fat * interval
        lambda_q_dt = lambda_q * interval
        inv_frc_dt = interval / v_frc if v_frc > 0.0 else 0.0
        full_circuit = circuit_share >= 1.0 - 1e-9
        circuit_share_dt = circuit_share * interval
        standard_path = not ecmo_active and pulmonary_flow_l is None
        exhaust_frac = 0.0
        for _ in range(substeps):
            p_ven = f_vrg * p_vrg + f_mus * p_mus + f_fat * p_fat
            if standard_path:
                if v_frc > 0.0:
                    p_art_target = one_minus_shunt * p_alv + shunt_fraction * p_ven
                    if q_pulm >= 0.25:
                        p_art = p_art_target
                    elif q_pulm > 0.0:
                        p_art += (p_art_target - p_art) * -math.expm1(-interval * q_pulm / 0.5)
                    airway_flux = (
                        alveolar_vent * (fi_agent - p_alv)
                        + concentrating_inflow * fi_agent
                        - washout_outflow * p_alv
                    )
                    step_blood = lambda_q_dt * (p_alv - p_ven)
                    blood_uptake_btps_l += step_blood
                    next_p_alv = clamp01(p_alv + (airway_flux + lambda_q * (p_ven - p_alv)) * inv_frc_dt)
                    if full_circuit:
                        net_btps_l += (next_p_alv - p_alv) * v_frc + step_blood
                    else:
                        net_btps_l += circuit_share_dt * (
                            alveolar_vent * (branch_fi - p_alv)
                            + concentrating_inflow * branch_fi
                            - washout_outflow * p_alv
                        )
                    p_alv = next_p_alv
                elif q_co >= 0.25:
                    p_art = p_ven
                elif q_co > 0.0:
                    p_art += (p_ven - p_art) * -math.expm1(-interval * q_co / 0.5)
            else:
                p_post_oxy = (p_ven + ecmo_equil * (ecmo_fi_agent - p_ven)) if ecmo_active else p_ven
                if ecmo_is_arterial:
                    p_pa = p_ven
                else:
                    f_ecmo = clamp01(q_ecmo / q_co) if (ecmo_active and q_co > 1e-6) else 0.0
                    p_pa = (1.0 - f_ecmo) * p_ven + f_ecmo * p_post_oxy
                if v_frc > 0.0:
                    p_lv = one_minus_shunt * p_alv + shunt_fraction * p_pa
                    airway_flux = (
                        alveolar_vent * (fi_agent - p_alv)
                        + concentrating_inflow * fi_agent
                        - washout_outflow * p_alv
                    )
                    step_blood = lambda_q_dt * (p_alv - p_pa)
                    blood_uptake_btps_l += step_blood
                    next_p_alv = clamp01(p_alv + (airway_flux + lambda_q * (p_pa - p_alv)) * inv_frc_dt)
                    if full_circuit:
                        net_btps_l += (next_p_alv - p_alv) * v_frc + step_blood
                    else:
                        net_btps_l += circuit_share_dt * (
                            alveolar_vent * (branch_fi - p_alv)
                            + concentrating_inflow * branch_fi
                            - washout_outflow * p_alv
                        )
                    p_alv = next_p_alv
                else:
                    p_lv = p_pa
                if ecmo_is_arterial:
                    if q_co >= 0.25:
                        p_art = (q_pulm * p_lv + max(0.0, q_co - q_pulm) * p_post_oxy) / q_co
                    elif q_co > 0.0:
                        p_art_target = (q_pulm * p_lv + max(0.0, q_co - q_pulm) * p_post_oxy) / q_co
                        p_art += (p_art_target - p_art) * -math.expm1(-interval * q_co / 0.5)
                else:
                    if q_pulm >= 0.25:
                        p_art = p_lv
                    elif q_pulm > 0.0:
                        p_art += (p_lv - p_art) * -math.expm1(-interval * q_pulm / 0.5)
            p_vrg += k_vrg_dt * (p_art - p_vrg)
            p_brain += k_brain_dt * (p_art - p_brain)
            p_mus += k_mus_dt * (p_art - p_mus)
            p_fat += k_fat_dt * (p_art - p_fat)
        p_ven = f_vrg * p_vrg + f_mus * p_mus + f_fat * p_fat
        if standard_path:
            if q_pulm >= 0.25:
                p_art = (one_minus_shunt * p_alv + shunt_fraction * p_ven) if v_frc > 0.0 else p_ven
        else:
            p_post_oxy = (p_ven + ecmo_equil * (ecmo_fi_agent - p_ven)) if ecmo_active else p_ven
            exhaust_frac = clamp01(ecmo_fi_agent + exhaust_gain * (p_ven - ecmo_fi_agent)) if ecmo_active else 0.0
            if ecmo_is_arterial:
                if q_co >= 0.25:
                    p_lv = (one_minus_shunt * p_alv + shunt_fraction * p_ven) if v_frc > 0.0 else p_ven
                    p_art = (q_pulm * p_lv + max(0.0, q_co - q_pulm) * p_post_oxy) / q_co
            elif q_pulm >= 0.25:
                f_ecmo = clamp01(q_ecmo / q_co) if (ecmo_active and q_co > 1e-6) else 0.0
                p_pa = (1.0 - f_ecmo) * p_ven + f_ecmo * p_post_oxy
                p_art = (one_minus_shunt * p_alv + shunt_fraction * p_pa) if v_frc > 0.0 else p_pa
        return _IntegrationResult(
            p_alv, p_vrg, p_brain, p_mus, p_fat, p_ven, p_art, v_frc, net_btps_l, blood_uptake_btps_l,
            exhaust_frac,
        )

    def _is_empty(self) -> bool:
        s = self.state
        return (
            s.p_alv == 0.0
            and s.p_vrg == 0.0
            and s.p_brain == 0.0
            and s.p_mus == 0.0
            and s.p_fat == 0.0
        )

    def preview_exchange(
        self,
        dt: float,
        fi_agent: float,
        alveolar_vent_l: float,
        cardiac_output_l: float,
        lung_volume_l: float | None = None,
        shunt_fraction: float = 0.0,
        companion_uptake_l_min: float = 0.0,
        passive_inflow_l_min: float = 0.0,
        circuit_share: float = 1.0,
        circuit_fi_agent: float | None = None,
        exhaled_volume_change_l: float | None = None,
        temp_c: float = 37.0,
        *,
        pulmonary_flow_l: float | None = None,
        ecmo_blood_flow_l: float = 0.0,
        ecmo_sweep_l: float = 0.0,
        ecmo_fi_agent: float = 0.0,
        ecmo_permeability: float = 0.0,
        ecmo_is_arterial: bool = False,
    ) -> tuple[float, _IntegrationResult | None]:
        """Return (circuit uptake BTPS L/min, integration result) without mutating state."""
        if dt <= 0.0 or circuit_share <= 0.0:
            return 0.0, None
        if (
            fi_agent <= 0.0
            and ecmo_fi_agent <= 0.0
            and (circuit_fi_agent is None or circuit_fi_agent <= 0.0)
            and self._is_empty()
        ):
            return 0.0, None
        result = self._integrate(
            dt,
            fi_agent,
            alveolar_vent_l,
            cardiac_output_l,
            lung_volume_l,
            shunt_fraction,
            companion_uptake_l_min,
            passive_inflow_l_min,
            exhaled_volume_change_l,
            circuit_share,
            circuit_fi_agent,
            temp_c=temp_c,
            pulmonary_flow_l=pulmonary_flow_l,
            ecmo_blood_flow_l=ecmo_blood_flow_l,
            ecmo_sweep_l=ecmo_sweep_l,
            ecmo_fi_agent=ecmo_fi_agent,
            ecmo_permeability=ecmo_permeability,
            ecmo_is_arterial=ecmo_is_arterial,
        )
        return result.net_btps_l / (dt / 60.0), result

    def commit_integration(self, result: _IntegrationResult, dt: float, temp_c: float = 37.0) -> None:
        """Apply a previously previewed integration result to state."""
        state = self.state
        state.p_alv = result.p_alv
        state.p_vrg = result.p_vrg
        state.p_brain = result.p_brain
        state.p_mus = result.p_mus
        state.p_fat = result.p_fat
        state.p_ven = result.p_ven
        state.p_art = result.p_art
        self.frc = result.v_frc
        self._temp_c = float(temp_c)
        self.last_blood_uptake_l_min = result.blood_uptake_btps_l / (dt / 60.0) if dt > 0.0 else 0.0
        self.last_exhaust_fraction = result.exhaust_fraction
        state.mac = (state.p_brain * 100.0) / self.mac_at_temp(temp_c)

    def step(self, dt: float, fi_agent: float, alveolar_vent_l: float, cardiac_output_l: float,
             temp_c: float = 37.0, lung_volume_l: float | None = None, shunt_fraction: float = 0.0,
             companion_uptake_l_min: float = 0.0, passive_inflow_l_min: float = 0.0,
             exhaled_volume_change_l: float | None = None,
             *,
             pulmonary_flow_l: float | None = None,
             ecmo_blood_flow_l: float = 0.0,
             ecmo_sweep_l: float = 0.0,
             ecmo_fi_agent: float = 0.0,
             ecmo_permeability: float = 0.0,
             ecmo_is_arterial: bool = False):
        """Advance with inspired fraction, ventilation/output in L/min, and temperature in °C.

        lung_volume_l is end-expiratory gas volume; shunt_fraction is the
        fraction of cardiac output bypassing ventilated alveoli. Positive
        companion_uptake_l_min (e.g. N2O wash-in) and passive_inflow_l_min
        draw additional inspired gas into FRC.
        """
        state = self.state
        if fi_agent <= 0.0 and ecmo_fi_agent <= 0.0 and self._is_empty():
            if lung_volume_l is not None:
                self.frc = lung_volume_l
            state.p_ven = state.p_art = state.mac = 0.0
            self.last_blood_uptake_l_min = 0.0
            self.last_exhaust_fraction = 0.0
            return
        result = self._integrate(
            dt,
            fi_agent,
            alveolar_vent_l,
            cardiac_output_l,
            lung_volume_l,
            shunt_fraction,
            companion_uptake_l_min,
            passive_inflow_l_min,
            exhaled_volume_change_l,
            temp_c=temp_c,
            pulmonary_flow_l=pulmonary_flow_l,
            ecmo_blood_flow_l=ecmo_blood_flow_l,
            ecmo_sweep_l=ecmo_sweep_l,
            ecmo_fi_agent=ecmo_fi_agent,
            ecmo_permeability=ecmo_permeability,
            ecmo_is_arterial=ecmo_is_arterial,
        )
        self.commit_integration(result, dt, temp_c=temp_c)
