import math
from dataclasses import dataclass
from typing import NamedTuple

from anasim.core.utils import clamp01

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
    mac: float = 0.0  # Brain MAC: cerebral pressure over the age-adjusted MAC


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

    def flow_fractions(self, cardiac_output_l: float) -> tuple[float, float, float]:
        """Organ CO shares; low CO shifts flow from muscle and fat toward VRG."""
        co_ratio = clamp01(cardiac_output_l / max(0.5, self._base_co))
        shift = 0.70 * clamp01((1.0 - co_ratio) / 0.65)
        f_mus = self.f_mus_frac * (1.0 - shift)
        f_fat = self.f_fat_frac * (1.0 - shift)
        return 1.0 - f_mus - f_fat, f_mus, f_fat

    def mac_at_temp(self, temp_c: float = 37.0) -> float:
        """Age- and temperature-adjusted MAC (% atm); falls ~5% per °C of hypothermia."""
        temp_factor = max(0.5, 1.0 - 0.05 * (37.0 - temp_c))
        return self.mac_age * temp_factor

    def lambda_b_g_at_temp(self, temp_c: float = 37.0) -> float:
        """Blood:gas partition coefficient at core temperature (Lockwood 1997)."""
        if temp_c == 37.0:
            return self.lambda_b_g
        return self.lambda_b_g * math.exp(-self._lambda_temp_slope * (temp_c - 37.0))

    def alveolar_uptake_l_min(
        self, cardiac_output_l: float, shunt_fraction: float = 0.0, temp_c: float = 37.0,
    ) -> float:
        """Instantaneous pulmonary capillary uptake from alveoli (positive into blood, L/min)."""
        lambda_q = max(0.0, cardiac_output_l) * (1.0 - shunt_fraction) * self.lambda_b_g_at_temp(temp_c)
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
    ) -> _IntegrationResult:
        dt_min = dt / 60.0
        q_co = max(0.0, cardiac_output_l)
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
        p_vrg = self.state.p_vrg
        p_brain = self.state.p_brain
        p_mus = self.state.p_mus
        p_fat = self.state.p_fat
        # Conserve dissolved body volatile mass when temperature alters solubility.
        # Lipid (fat) solubility varies roughly half as steeply with temperature as blood.
        if temp_c != self._temp_c:
            new_lambda = self.lambda_b_g_at_temp(temp_c)
            if new_lambda > 0.0:
                sol_scale = self.lambda_b_g_at_temp(self._temp_c) / new_lambda
                p_vrg = clamp01(p_vrg * sol_scale)
                p_brain = clamp01(p_brain * sol_scale)
                p_mus = clamp01(p_mus * sol_scale)
                p_fat = clamp01(p_fat * (sol_scale ** 0.5))
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
        lambda_q = q_co * (1.0 - shunt_fraction) * self.lambda_b_g_at_temp(temp_c)
        # Tissue rate constant k = (flow / volume) / lambda (1/min). A high
        # partition coefficient slows equilibration: VRG in minutes, fat in hours.
        k_vrg = (q_vrg / self.v_vrg) / self.lambda_t_b_vrg
        k_mus = (q_mus / self.v_mus) / self.lambda_t_b_mus
        k_fat = (q_fat / self.v_fat) / self.lambda_t_b_fat
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
        for _ in range(substeps):
            p_ven = f_vrg * p_vrg + f_mus * p_mus + f_fat * p_fat
            if v_frc > 0.0:
                p_art = (1.0 - shunt_fraction) * p_alv + shunt_fraction * p_ven
                airway_flux = (
                    alveolar_vent * (fi_agent - p_alv)
                    + concentrating_inflow * fi_agent
                    - washout_outflow * p_alv
                )
                step_blood = lambda_q * (p_alv - p_ven) * interval
                blood_uptake_btps_l += step_blood
                next_p_alv = clamp01(p_alv + (airway_flux + lambda_q * (p_ven - p_alv)) / v_frc * interval)
                if circuit_share >= 1.0 - 1e-9:
                    net_btps_l += (next_p_alv - p_alv) * v_frc + step_blood
                else:
                    net_btps_l += circuit_share * (
                        alveolar_vent * (branch_fi - p_alv)
                        + concentrating_inflow * branch_fi
                        - washout_outflow * p_alv
                    ) * interval
                p_alv = next_p_alv
            else:
                p_art = p_ven
            p_vrg += k_vrg * (p_art - p_vrg) * interval
            p_brain += k_brain * (p_art - p_brain) * interval
            p_mus += k_mus * (p_art - p_mus) * interval
            p_fat += k_fat * (p_art - p_fat) * interval
        p_ven = f_vrg * p_vrg + f_mus * p_mus + f_fat * p_fat
        p_art = ((1.0 - shunt_fraction) * p_alv + shunt_fraction * p_ven) if v_frc > 0.0 else p_ven
        return _IntegrationResult(
            p_alv, p_vrg, p_brain, p_mus, p_fat, p_ven, p_art, v_frc, net_btps_l, blood_uptake_btps_l,
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
    ) -> tuple[float, _IntegrationResult | None]:
        """Return (circuit uptake BTPS L/min, integration result) without mutating state."""
        if dt <= 0.0 or circuit_share <= 0.0:
            return 0.0, None
        s = self.state
        if (
            fi_agent <= 0.0
            and (circuit_fi_agent is None or circuit_fi_agent <= 0.0)
            and s.p_alv == 0.0
            and s.p_vrg == 0.0
            and s.p_brain == 0.0
            and s.p_mus == 0.0
            and s.p_fat == 0.0
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
        )
        return result.net_btps_l / (dt / 60.0), result

    def exchange_btps_l_min(
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
    ) -> float:
        """Return net circuit uptake (BTPS L/min) matching step() over dt without mutating state."""
        return self.preview_exchange(
            dt, fi_agent, alveolar_vent_l, cardiac_output_l,
            lung_volume_l=lung_volume_l, shunt_fraction=shunt_fraction,
            companion_uptake_l_min=companion_uptake_l_min,
            passive_inflow_l_min=passive_inflow_l_min,
            circuit_share=circuit_share, circuit_fi_agent=circuit_fi_agent,
            exhaled_volume_change_l=exhaled_volume_change_l,
            temp_c=temp_c,
        )[0]

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
        state.mac = (state.p_brain * 100.0) / self.mac_at_temp(temp_c)

    def step(self, dt: float, fi_agent: float, alveolar_vent_l: float, cardiac_output_l: float,
             temp_c: float = 37.0, lung_volume_l: float | None = None, shunt_fraction: float = 0.0,
             companion_uptake_l_min: float = 0.0, passive_inflow_l_min: float = 0.0,
             exhaled_volume_change_l: float | None = None):
        """Advance with inspired fraction, ventilation/output in L/min, and temperature in °C.

        lung_volume_l is end-expiratory gas volume; shunt_fraction is the
        fraction of cardiac output bypassing ventilated alveoli. Positive
        companion_uptake_l_min (e.g. N2O wash-in) and passive_inflow_l_min
        draw additional inspired gas into FRC.
        """
        state = self.state
        if (
            fi_agent <= 0.0
            and state.p_alv == 0.0
            and state.p_vrg == 0.0
            and state.p_brain == 0.0
            and state.p_mus == 0.0
            and state.p_fat == 0.0
        ):
            if lung_volume_l is not None:
                self.frc = lung_volume_l
            state.p_ven = state.p_art = state.mac = 0.0
            self.last_blood_uptake_l_min = 0.0
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
        )
        self.commit_integration(result, dt, temp_c=temp_c)
