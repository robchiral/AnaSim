import math
from dataclasses import dataclass

from .patient import Patient


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
        lambda_t_b_mus: float = None,
        lambda_t_b_fat: float = None,
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

        # Tissue volumes (L) and shares of cardiac output.
        w_kg = self.patient.weight
        self.v_vrg = 0.1 * w_kg
        self.v_mus = 0.5 * w_kg
        self.v_fat = 0.2 * w_kg
        self.f_vrg_frac = 0.75
        self.f_mus_frac = 0.19
        self.f_fat_frac = 0.06

        self.state = VolatileState()

    def step(self, dt: float, fi_agent: float, alveolar_vent_l: float, cardiac_output_l: float, temp_c: float = 37.0):
        """Advance dt seconds with inspired fraction fi_agent, alveolar
        ventilation and cardiac output in L/min, and core temperature (°C)."""
        state = self.state
        dt_min = dt / 60.0

        q_co = cardiac_output_l
        q_vrg = q_co * self.f_vrg_frac
        q_mus = q_co * self.f_mus_frac
        q_fat = q_co * self.f_fat_frac

        alveolar_vent = max(0.0, alveolar_vent_l)

        # FRC dPalv/dt = VA (Fi - Palv) + Q lambda_bg (Pven - Palv); arterial
        # blood leaves at the alveolar pressure.
        v_frc = 2.5  # L
        lambda_q = q_co * self.lambda_b_g
        # Tissue rate constant k = (flow / volume) / lambda (1/min). A high
        # partition coefficient slows equilibration: VRG in minutes, fat in hours.
        k_vrg = (q_vrg / self.v_vrg) / self.lambda_t_b_vrg
        k_mus = (q_mus / self.v_mus) / self.lambda_t_b_mus
        k_fat = (q_fat / self.v_fat) / self.lambda_t_b_fat

        # Update all compartments from the same pressures so agent is
        # conserved; substep so no Euler weight exceeds 1.
        fastest_rate = max((alveolar_vent + lambda_q) / v_frc, k_vrg, k_mus, k_fat)
        substeps = max(1, math.ceil(dt_min * fastest_rate))
        interval = dt_min / substeps
        for _ in range(substeps):
            p_alv = state.p_alv
            p_ven = (
                self.f_vrg_frac * state.p_vrg
                + self.f_mus_frac * state.p_mus
                + self.f_fat_frac * state.p_fat
            )
            state.p_alv += (
                alveolar_vent * (fi_agent - p_alv) + lambda_q * (p_ven - p_alv)
            ) / v_frc * interval
            state.p_vrg += k_vrg * (p_alv - state.p_vrg) * interval
            state.p_mus += k_mus * (p_alv - state.p_mus) * interval
            state.p_fat += k_fat * (p_alv - state.p_fat) * interval

        state.p_art = state.p_alv
        # Mixed venous pressure is the flow-weighted tissue pressure.
        state.p_ven = (
            self.f_vrg_frac * state.p_vrg
            + self.f_mus_frac * state.p_mus
            + self.f_fat_frac * state.p_fat
        )

        # MAC requirement falls about 5% per °C of hypothermia.
        temp_diff = 37.0 - temp_c
        temp_factor = max(0.5, 1.0 - 0.05 * temp_diff)
        corrected_mac_age = self.mac_age * temp_factor
        state.mac = (state.p_vrg * 100.0) / corrected_mac_age
