import math
from dataclasses import dataclass

from anasim.core.utils import clamp01

from .patient import Patient
from .pk_models import _janmahasatian_ffm


@dataclass(slots=True)
class VolatileState:
    """Partial pressures as fractions of an atmosphere (0.02 is 2%).

    At equilibrium every compartment has the arterial partial pressure.
    """

    p_alv: float = 0.0
    p_art: float = 0.0
    p_ven: float = 0.0  # Mixed venous
    p_vrg: float = 0.0  # Vessel-rich group, including brain
    p_mus: float = 0.0
    p_fat: float = 0.0
    mac: float = 0.0  # Brain MAC: VRG pressure over the age-adjusted MAC


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

        # Age-adjusted MAC (Mapleson 1996).
        self.mac_40 = mac_40
        self.mac_age = self.mac_40 * (10 ** (-0.00269 * (self.patient.age - 40)))

        # Tissue volumes (L) and shares of cardiac output: lean organs scale
        # with fat-free mass, while excess body mass expands the adipose depot.
        male = self.patient.sex == "male"
        ffm_scale = (
            _janmahasatian_ffm(self.patient.weight, self.patient.height, male)
            / _janmahasatian_ffm(70.0, 170.0, True)
        )
        self.v_vrg = 7.0 * ffm_scale
        self.v_mus = 35.0 * ffm_scale
        self.v_fat = max(3.0, 14.0 * ffm_scale + 0.8 * max(0.0, self.patient.weight - 70.0 * ffm_scale))
        self.f_vrg_frac = 0.75
        self.f_mus_frac = 0.19
        self.f_fat_frac = 0.06

        self.state = VolatileState()
        self.frc = patient.functional_residual_capacity()

    def mac_at_temp(self, temp_c: float = 37.0) -> float:
        """Age- and temperature-adjusted MAC (% atm); falls ~5% per °C of hypothermia."""
        temp_factor = max(0.5, 1.0 - 0.05 * (37.0 - temp_c))
        return self.mac_age * temp_factor

    def alveolar_uptake_l_min(self, cardiac_output_l: float, shunt_fraction: float = 0.0) -> float:
        """Instantaneous pulmonary capillary uptake from alveoli (positive into blood, L/min)."""
        lambda_q = max(0.0, cardiac_output_l) * (1.0 - shunt_fraction) * self.lambda_b_g
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
    ) -> tuple[float, float, float, float, float, float, float, float]:
        dt_min = dt / 60.0
        q_co = cardiac_output_l
        q_vrg = q_co * self.f_vrg_frac
        q_mus = q_co * self.f_mus_frac
        q_fat = q_co * self.f_fat_frac
        alveolar_vent = max(0.0, alveolar_vent_l)
        branch_fi = fi_agent / circuit_share if circuit_fi_agent is None else circuit_fi_agent
        concentrating_inflow = (
            (max(0.0, companion_uptake_l_min) + max(0.0, passive_inflow_l_min)) if fi_agent > 0.0 else 0.0
        )
        washout_outflow = max(0.0, -companion_uptake_l_min)
        p_alv = self.state.p_alv
        p_vrg = self.state.p_vrg
        p_mus = self.state.p_mus
        p_fat = self.state.p_fat
        v_frc = self.frc
        net_btps_l = 0.0
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
        lambda_q = q_co * (1.0 - shunt_fraction) * self.lambda_b_g
        # Tissue rate constant k = (flow / volume) / lambda (1/min). A high
        # partition coefficient slows equilibration: VRG in minutes, fat in hours.
        k_vrg = (q_vrg / self.v_vrg) / self.lambda_t_b_vrg
        k_mus = (q_mus / self.v_mus) / self.lambda_t_b_mus
        k_fat = (q_fat / self.v_fat) / self.lambda_t_b_fat
        # Update all compartments from the same pressures so agent is
        # conserved; substep so no Euler weight exceeds 0.05.
        alveolar_rate = (
            (alveolar_vent + concentrating_inflow + washout_outflow + lambda_q) / v_frc
            if v_frc > 0.0
            else 0.0
        )
        fastest_rate = max(alveolar_rate, k_vrg, k_mus, k_fat)
        substeps = max(1, math.ceil(dt_min * fastest_rate / 0.05))
        interval = dt_min / substeps
        for _ in range(substeps):
            p_ven = self.f_vrg_frac * p_vrg + self.f_mus_frac * p_mus + self.f_fat_frac * p_fat
            if v_frc > 0.0:
                p_art = (1.0 - shunt_fraction) * p_alv + shunt_fraction * p_ven
                airway_flux = (
                    alveolar_vent * (fi_agent - p_alv)
                    + concentrating_inflow * fi_agent
                    - washout_outflow * p_alv
                )
                next_p_alv = clamp01(p_alv + (airway_flux + lambda_q * (p_ven - p_alv)) / v_frc * interval)
                if circuit_share >= 1.0 - 1e-9:
                    net_btps_l += (next_p_alv - p_alv) * v_frc + lambda_q * (p_alv - p_ven) * interval
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
            p_mus += k_mus * (p_art - p_mus) * interval
            p_fat += k_fat * (p_art - p_fat) * interval
        p_ven = self.f_vrg_frac * p_vrg + self.f_mus_frac * p_mus + self.f_fat_frac * p_fat
        p_art = ((1.0 - shunt_fraction) * p_alv + shunt_fraction * p_ven) if v_frc > 0.0 else p_ven
        return p_alv, p_vrg, p_mus, p_fat, p_ven, p_art, v_frc, net_btps_l

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
    ) -> float:
        """Return net circuit uptake (BTPS L/min) matching step() over dt without mutating state."""
        if dt <= 0.0 or circuit_share <= 0.0:
            return 0.0
        s = self.state
        if (
            fi_agent <= 0.0
            and (circuit_fi_agent is None or circuit_fi_agent <= 0.0)
            and s.p_alv == 0.0
            and s.p_vrg == 0.0
            and s.p_mus == 0.0
            and s.p_fat == 0.0
        ):
            return 0.0
        net_btps_l = self._integrate(
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
        )[7]
        return net_btps_l / (dt / 60.0)

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
            and state.p_mus == 0.0
            and state.p_fat == 0.0
        ):
            if lung_volume_l is not None:
                self.frc = lung_volume_l
            state.p_ven = state.p_art = state.mac = 0.0
            return
        (
            state.p_alv,
            state.p_vrg,
            state.p_mus,
            state.p_fat,
            state.p_ven,
            state.p_art,
            self.frc,
            _,
        ) = self._integrate(
            dt,
            fi_agent,
            alveolar_vent_l,
            cardiac_output_l,
            lung_volume_l,
            shunt_fraction,
            companion_uptake_l_min,
            passive_inflow_l_min,
            exhaled_volume_change_l,
        )
        state.mac = (state.p_vrg * 100.0) / self.mac_at_temp(temp_c)
