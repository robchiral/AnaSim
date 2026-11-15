import math
from dataclasses import dataclass

from anasim.core.utils import clamp01

from .patient import Patient
from .pk_models import _janmahasatian_ffm


@dataclass
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
        ffm_scale = _janmahasatian_ffm(self.patient.weight, self.patient.height, male) / _janmahasatian_ffm(70.0, 170.0, male)
        self.v_vrg = 7.0 * ffm_scale
        self.v_mus = 35.0 * ffm_scale
        self.v_fat = max(3.0, 14.0 * ffm_scale + 0.8 * max(0.0, self.patient.weight - 70.0 * ffm_scale))
        self.f_vrg_frac = 0.75
        self.f_mus_frac = 0.19
        self.f_fat_frac = 0.06

        self.state = VolatileState()
        self.frc = patient.functional_residual_capacity()

    def alveolar_uptake_l_min(self, cardiac_output_l: float, shunt_fraction: float = 0.0) -> float:
        """Instantaneous pulmonary capillary uptake from alveoli (positive into blood, L/min)."""
        lambda_q = max(0.0, cardiac_output_l) * (1.0 - shunt_fraction) * self.lambda_b_g
        return lambda_q * (self.state.p_alv - self.state.p_ven)

    def step(self, dt: float, fi_agent: float, alveolar_vent_l: float, cardiac_output_l: float,
             temp_c: float = 37.0, lung_volume_l: float | None = None, shunt_fraction: float = 0.0,
             companion_uptake_l_min: float = 0.0):
        """Advance with inspired fraction, ventilation/output in L/min, and temperature in °C.

        lung_volume_l is end-expiratory gas volume; shunt_fraction is the
        fraction of cardiac output bypassing ventilated alveoli. Positive
        companion_uptake_l_min (e.g. N2O wash-in) draws additional inspired
        gas into FRC, producing the second-gas concentrating effect.
        """
        state = self.state
        dt_min = dt / 60.0

        q_co = cardiac_output_l
        q_vrg = q_co * self.f_vrg_frac
        q_mus = q_co * self.f_mus_frac
        q_fat = q_co * self.f_fat_frac

        alveolar_vent = max(0.0, alveolar_vent_l)
        concentrating_inflow = max(0.0, companion_uptake_l_min) if fi_agent > 0.0 else 0.0

        # FRC dPalv/dt = VA (Fi - Palv) + U_companion Fi + (1-s) Q lambda_bg (Pven - Palv).
        # Arterial blood mixes end-capillary and shunted venous blood.
        if lung_volume_l is not None:
            change = lung_volume_l - self.frc
            if change > 0.0:
                state.p_alv = (self.frc * state.p_alv + change * fi_agent) / lung_volume_l
            self.frc = lung_volume_l
        v_frc = self.frc
        lambda_q = q_co * (1.0 - shunt_fraction) * self.lambda_b_g
        # Tissue rate constant k = (flow / volume) / lambda (1/min). A high
        # partition coefficient slows equilibration: VRG in minutes, fat in hours.
        k_vrg = (q_vrg / self.v_vrg) / self.lambda_t_b_vrg
        k_mus = (q_mus / self.v_mus) / self.lambda_t_b_mus
        k_fat = (q_fat / self.v_fat) / self.lambda_t_b_fat

        # Update all compartments from the same pressures so agent is
        # conserved; substep so no Euler weight exceeds 1.
        fastest_rate = max((alveolar_vent + concentrating_inflow + lambda_q) / v_frc, k_vrg, k_mus, k_fat)
        substeps = max(1, math.ceil(dt_min * fastest_rate))
        interval = dt_min / substeps
        for _ in range(substeps):
            p_alv = state.p_alv
            p_ven = (
                self.f_vrg_frac * state.p_vrg
                + self.f_mus_frac * state.p_mus
                + self.f_fat_frac * state.p_fat
            )
            p_art = (1.0 - shunt_fraction) * p_alv + shunt_fraction * p_ven
            state.p_alv = clamp01(
                p_alv
                + (
                    alveolar_vent * (fi_agent - p_alv)
                    + concentrating_inflow * fi_agent
                    + lambda_q * (p_ven - p_alv)
                )
                / v_frc
                * interval
            )
            state.p_vrg += k_vrg * (p_art - state.p_vrg) * interval
            state.p_mus += k_mus * (p_art - state.p_mus) * interval
            state.p_fat += k_fat * (p_art - state.p_fat) * interval

        # Mixed venous pressure is the flow-weighted tissue pressure.
        state.p_ven = (
            self.f_vrg_frac * state.p_vrg
            + self.f_mus_frac * state.p_mus
            + self.f_fat_frac * state.p_fat
        )
        state.p_art = (1.0 - shunt_fraction) * state.p_alv + shunt_fraction * state.p_ven

        # MAC requirement falls about 5% per °C of hypothermia.
        temp_diff = 37.0 - temp_c
        temp_factor = max(0.5, 1.0 - 0.05 * temp_diff)
        corrected_mac_age = self.mac_age * temp_factor
        state.mac = (state.p_vrg * 100.0) / corrected_mac_age
