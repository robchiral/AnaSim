"""Unified extracorporeal support controller for VV ECMO, VA ECMO, and CPB."""

from __future__ import annotations

import math

from anasim.core.utils import clamp, clamp01
from anasim.patient.domain import bounded_number

from .oxygenator import MembraneOxygenator, volatile_membrane_permeability
from .sequestration import CircuitSequestration
from .types import (
    BloodGasManagement,
    CannulationType,
    ExtracorporealConfig,
    ExtracorporealMode,
    ExtracorporealState,
    MembraneType,
    OxygenatorPassResult,
)

# Specific heat of whole blood (~3.85 J / mL / °C -> 3850 J / L / °C).
_BLOOD_HEAT_CAP_J_PER_L_C = 3850.0


def watershed_ecmo_fractions(
    pump_flow_l_min: float,
    native_lv_flow_l_min: float,
    cannulation: CannulationType,
) -> tuple[float, float, float]:
    """Return (upper_body_ecmo_share, coronary_ecmo_share, lower_body_ecmo_share).

    In central VA ECMO and CPB, return enters the ascending aorta so upper,
    coronary, and lower beds receive the flow-weighted mixed arterial stream.
    In peripheral (femoral) VA ECMO, retrograde aortic flow meets antegrade LV
    ejection at a flow-dependent aortic watershed (CFD arch-branch simulation in
    Stevens 2017; clinical bedside CEUS localization in Buchtele 2020): below
    70-75% ECMO support arch branches remain LV-perfused, with 50% brachiocephalic
    mixing near 82% ECMO support. Lower-body (0.25) and coronary-root (0.88)
    sigmoid midpoints are simulator calibrations.
    """
    q_pump = max(0.0, pump_flow_l_min)
    q_lv = max(0.0, native_lv_flow_l_min)
    q_tot = q_pump + q_lv
    if q_tot <= 1e-6:
        return 0.0, 0.0, 0.0
    ecmo_share = clamp01(q_pump / q_tot)
    if cannulation == CannulationType.CENTRAL:
        return ecmo_share, ecmo_share, ecmo_share
    if q_lv <= 0.05:
        return 1.0, 1.0, 1.0
    if q_pump <= 0.05:
        return 0.0, 0.0, 0.0

    # Smooth sigmoidal watershed across aortic arch (arch midpoint informed by Stevens 2017):
    # - Lower body (femoral/renal) is almost completely ECMO-perfused once ecmo_share > 0.35.
    # - Upper body / right radial / carotids cross 50% ECMO near ecmo_share ~0.78.
    # - Coronaries (aortic root) stay LV-perfused until ecmo_share > 0.85-0.90.
    lower = clamp01(1.0 / (1.0 + math.exp(-12.0 * (ecmo_share - 0.25))))
    upper = clamp01(1.0 / (1.0 + math.exp(-14.0 * (ecmo_share - 0.78))))
    coronary = clamp01(1.0 / (1.0 + math.exp(-18.0 * (ecmo_share - 0.88))))
    # Enforce exact endpoints when LV flow vanishes or pump stops.
    lv_zero_blend = clamp01((0.20 - q_lv) / 0.15)
    upper = upper + lv_zero_blend * (1.0 - upper)
    coronary = coronary + lv_zero_blend * (1.0 - coronary)
    lower = max(lower, upper)
    return upper, coronary, lower


class ExtracorporealSystem:
    """Manages VV ECMO, VA ECMO, and CPB circuit state, oxygenator, reservoir, and thermal exchange."""

    def __init__(self, config: ExtracorporealConfig | None = None):
        self.config = config or ExtracorporealConfig()
        self.state = ExtracorporealState(membrane_area_m2=self.config.default_membrane_area_m2)
        self.oxygenator = MembraneOxygenator(area_m2=self.config.default_membrane_area_m2)
        self.sequestration = CircuitSequestration()
        self._last_pass: OxygenatorPassResult | None = None

    @property
    def is_active(self) -> bool:
        return self.state.active and self.state.mode is not ExtracorporealMode.NONE

    @property
    def is_vv(self) -> bool:
        return self.is_active and self.state.mode is ExtracorporealMode.VV_ECMO

    @property
    def is_va(self) -> bool:
        return self.is_active and self.state.mode is ExtracorporealMode.VA_ECMO

    @property
    def is_cpb(self) -> bool:
        return self.is_active and self.state.mode is ExtracorporealMode.CPB

    @property
    def is_arterial_support(self) -> bool:
        """True when extracorporeal blood returns to the systemic arterial tree (VA ECMO or CPB)."""
        return self.is_va or self.is_cpb

    @property
    def volatile_permeability(self) -> float:
        return volatile_membrane_permeability(self.state.membrane_type)

    @property
    def total_circuit_mixing_volume_ml(self) -> float:
        """Extracorporeal tubing + hard-shell reservoir blood volume (mL) held outside patient vessels."""
        if not self.is_active:
            return 0.0
        return max(0.0, self.state.primed_volume_ml + self.state.reservoir_volume_ml)

    def _begin_session(
        self,
        mode: ExtracorporealMode,
        *,
        cannulation: CannulationType,
        membrane_type: MembraneType | str,
        membrane_area_m2: float | None,
        prime_volume_ml: float | None,
        default_prime_ml: float,
        max_prime_ml: float,
        blood_flow_l_min: float,
        sweep_gas_l_min: float,
        sweep_fio2: float,
        heater_cooler_on: bool,
        water_target_c: float,
    ) -> tuple[bool, float]:
        mem = self._parse_membrane(membrane_type)
        area = self.config.default_membrane_area_m2 if membrane_area_m2 is None else bounded_number(
            "membrane area", membrane_area_m2, 0.2, 3.0, unit="m^2",
        )
        prime = default_prime_ml if prime_volume_ml is None else bounded_number(
            "prime volume", prime_volume_ml, 0.0, max_prime_ml, unit="mL",
        )
        fresh = self.state.mode is not mode or not self.state.active
        self.state.mode = mode
        self.state.active = True
        self.state.cannulation = cannulation
        self.state.membrane_type = mem
        self.state.membrane_area_m2 = area
        self.oxygenator.area_m2 = area
        self.set_blood_flow(blood_flow_l_min)
        self.state.blood_flow_l_min = self.state.target_blood_flow_l_min
        self.state.effective_blood_flow_l_min = self.state.blood_flow_l_min
        self.set_sweep_gas(sweep_gas_l_min)
        self.set_sweep_fio2(sweep_fio2)
        self.set_heater_cooler(heater_cooler_on, water_target_c)
        if fresh:
            self._last_pass = None
            self.sequestration.reset()
        return fresh, prime

    def start_vv_ecmo(
        self,
        *,
        blood_flow_l_min: float = 4.0,
        sweep_gas_l_min: float = 4.0,
        sweep_fio2: float = 1.0,
        cannulation: CannulationType | str = CannulationType.FEM_JUG,
        membrane_type: MembraneType | str = MembraneType.PMP,
        membrane_area_m2: float | None = None,
        prime_volume_ml: float | None = None,
        heater_cooler_on: bool = True,
        water_target_c: float = 37.0,
    ) -> float:
        """Initiate VV ECMO and return prime volume (mL) to mix into circulating volume."""
        cann = self._parse_cannulation(cannulation, allowed=(
            CannulationType.FEM_JUG, CannulationType.FEM_FEM, CannulationType.DUAL_LUMEN,
        ))
        fresh, prime = self._begin_session(
            ExtracorporealMode.VV_ECMO,
            cannulation=cann,
            membrane_type=membrane_type,
            membrane_area_m2=membrane_area_m2,
            prime_volume_ml=prime_volume_ml,
            default_prime_ml=self.config.ecmo_prime_volume_ml,
            max_prime_ml=2500.0,
            blood_flow_l_min=blood_flow_l_min,
            sweep_gas_l_min=sweep_gas_l_min,
            sweep_fio2=sweep_fio2,
            heater_cooler_on=heater_cooler_on,
            water_target_c=water_target_c,
        )
        st = self.state
        st.sweep_fco2 = st.sweep_sevo_percent = st.lv_vent_l_min = st.reservoir_volume_ml = 0.0
        st.venous_clamp_open = 1.0
        st.aortic_cross_clamp = st.cardioplegia_arrest = st.bypass_alarm_mode = st.reservoir_low_alarm = False
        if fresh:
            st.primed_volume_ml = prime
            return prime
        return 0.0

    def start_va_ecmo(
        self,
        *,
        blood_flow_l_min: float = 4.0,
        sweep_gas_l_min: float = 4.0,
        sweep_fio2: float = 1.0,
        cannulation: CannulationType | str = CannulationType.PERIPHERAL_FEM,
        membrane_type: MembraneType | str = MembraneType.PMP,
        membrane_area_m2: float | None = None,
        prime_volume_ml: float | None = None,
        heater_cooler_on: bool = True,
        water_target_c: float = 37.0,
        lv_vent_l_min: float = 0.0,
    ) -> float:
        """Initiate VA ECMO and return prime volume (mL) to mix into circulating volume."""
        cann = self._parse_cannulation(cannulation, allowed=(
            CannulationType.PERIPHERAL_FEM, CannulationType.CENTRAL,
        ))
        fresh, prime = self._begin_session(
            ExtracorporealMode.VA_ECMO,
            cannulation=cann,
            membrane_type=membrane_type,
            membrane_area_m2=membrane_area_m2,
            prime_volume_ml=prime_volume_ml,
            default_prime_ml=self.config.ecmo_prime_volume_ml,
            max_prime_ml=2500.0,
            blood_flow_l_min=blood_flow_l_min,
            sweep_gas_l_min=sweep_gas_l_min,
            sweep_fio2=sweep_fio2,
            heater_cooler_on=heater_cooler_on,
            water_target_c=water_target_c,
        )
        self.set_lv_vent(lv_vent_l_min)
        st = self.state
        st.sweep_fco2 = st.sweep_sevo_percent = st.reservoir_volume_ml = 0.0
        st.venous_clamp_open = 1.0
        st.aortic_cross_clamp = st.cardioplegia_arrest = st.bypass_alarm_mode = st.reservoir_low_alarm = False
        if fresh:
            st.primed_volume_ml = prime
            return prime
        return 0.0

    def start_cpb(
        self,
        *,
        blood_flow_l_min: float = 4.5,
        sweep_gas_l_min: float = 3.0,
        sweep_fio2: float = 0.80,
        sweep_fco2: float = 0.0,
        sweep_sevo_percent: float = 0.0,
        membrane_type: MembraneType | str = MembraneType.MICROPOROUS,
        membrane_area_m2: float | None = None,
        prime_volume_ml: float | None = None,
        initial_reservoir_ml: float = 800.0,
        blood_gas_strategy: BloodGasManagement | str = BloodGasManagement.ALPHA_STAT,
        heater_cooler_on: bool = True,
        water_target_c: float = 37.0,
        venous_clamp_open: float = 1.0,
        bypass_alarm_mode: bool = True,
    ) -> float:
        """Initiate cardiopulmonary bypass with open venous reservoir.

        Returns the prime volume (mL) mixed into the patient+circuit pool on initiation.
        """
        strategy = self._parse_bg_strategy(blood_gas_strategy)
        res_init = bounded_number("initial reservoir volume", initial_reservoir_ml, 0.0, self.config.cpb_reservoir_max_ml, unit="mL")
        fresh, prime = self._begin_session(
            ExtracorporealMode.CPB,
            cannulation=CannulationType.CENTRAL,
            membrane_type=membrane_type,
            membrane_area_m2=membrane_area_m2,
            prime_volume_ml=prime_volume_ml,
            default_prime_ml=self.config.cpb_prime_volume_ml,
            max_prime_ml=3000.0,
            blood_flow_l_min=blood_flow_l_min,
            sweep_gas_l_min=sweep_gas_l_min,
            sweep_fio2=sweep_fio2,
            heater_cooler_on=heater_cooler_on,
            water_target_c=water_target_c,
        )
        st = self.state
        st.blood_gas_strategy = strategy
        self.set_sweep_fco2(sweep_fco2)
        self.set_sweep_sevo(sweep_sevo_percent)
        self.set_venous_clamp(venous_clamp_open)
        st.bypass_alarm_mode = bool(bypass_alarm_mode)
        if fresh:
            st.lv_vent_l_min = 0.0
            st.aortic_cross_clamp = st.cardioplegia_arrest = False
            # Total prime enters the shared blood+reservoir pool; res_init sits in
            # the hard-shell reservoir while (prime - res_init) fills tubing/patient.
            st.reservoir_volume_ml = min(prime, res_init)
            st.primed_volume_ml = max(0.0, prime - st.reservoir_volume_ml)
            st.reservoir_low_alarm = st.reservoir_volume_ml < self.config.cpb_reservoir_low_ml
            return prime
        return 0.0

    def stop(self) -> None:
        """Terminate extracorporeal support and reset flow and gas telemetry."""
        self._last_pass = None
        self.sequestration.reset()
        self.state = ExtracorporealState(membrane_area_m2=self.config.default_membrane_area_m2)
        self.oxygenator.area_m2 = self.config.default_membrane_area_m2

    @staticmethod
    def _parse_cannulation(
        value: CannulationType | str,
        *,
        allowed: tuple[CannulationType, ...],
    ) -> CannulationType:
        cann = value if isinstance(value, CannulationType) else CannulationType(str(value))
        if cann not in allowed:
            choices = ", ".join(item.value for item in allowed)
            raise ValueError(f"Unsupported cannulation {cann.value!r}; choose one of: {choices}")
        return cann

    @staticmethod
    def _parse_membrane(value: MembraneType | str) -> MembraneType:
        return value if isinstance(value, MembraneType) else MembraneType(str(value))

    @staticmethod
    def _parse_bg_strategy(value: BloodGasManagement | str) -> BloodGasManagement:
        return value if isinstance(value, BloodGasManagement) else BloodGasManagement(str(value))

    def set_blood_flow(self, l_min: float) -> None:
        if not self.is_active:
            raise ValueError("Extracorporeal blood flow can only be adjusted during active support")
        self.state.target_blood_flow_l_min = bounded_number("extracorporeal blood flow", l_min, 0.0, 8.0, unit="L/min")
        if not self.is_cpb:
            self.state.blood_flow_l_min = self.state.target_blood_flow_l_min
            self.state.effective_blood_flow_l_min = self.state.blood_flow_l_min * (1.0 - self.state.recirculation_fraction)

    def apply_closed_circuit_drainage_limit(
        self,
        patient_stressed_vol_ml: float,
        patient_blood_volume_ml: float,
        min_patient_blood_volume_ml: float = 500.0,
    ) -> float:
        """Limit closed-circuit VV/VA ECMO pump flow when venous drainage collapses in severe hypovolemia."""
        st = self.state
        if not self.is_active or self.is_cpb:
            return st.blood_flow_l_min if self.is_active else 0.0
        vol_margin = max(0.0, patient_blood_volume_ml - min_patient_blood_volume_ml)
        vol_gate = clamp01(vol_margin / 500.0)
        stress_gate = clamp01(max(0.0, patient_stressed_vol_ml) / 200.0)
        drain_gate = min(vol_gate, stress_gate)
        st.blood_flow_l_min = st.target_blood_flow_l_min * drain_gate
        st.venous_drainage_l_min = st.blood_flow_l_min
        if self.is_va:
            st.effective_blood_flow_l_min = st.blood_flow_l_min
        return st.blood_flow_l_min

    def set_sweep_gas(self, l_min: float) -> None:
        if not self.is_active:
            raise ValueError("Sweep gas flow can only be adjusted during active support")
        self.state.sweep_gas_l_min = bounded_number("sweep gas flow", l_min, 0.0, 16.0, unit="L/min")

    def set_sweep_fio2(self, fio2: float) -> None:
        if not self.is_active:
            raise ValueError("Sweep FiO2 can only be adjusted during active support")
        self.state.sweep_fio2 = bounded_number("sweep FiO2", fio2, 0.21, 1.0)

    def set_sweep_fco2(self, fco2: float) -> None:
        if not self.is_active:
            raise ValueError("Sweep FCO2 can only be adjusted during active support")
        self.state.sweep_fco2 = bounded_number("sweep FCO2", fco2, 0.0, 0.10)

    def set_sweep_sevo(self, percent: float) -> None:
        if not self.is_active:
            raise ValueError("Oxygenator sevoflurane can only be adjusted during active support")
        val = bounded_number("oxygenator sevoflurane", percent, 0.0, 8.0, unit="%")
        if val > 0.0 and not self.is_cpb and self.state.membrane_type != MembraneType.MICROPOROUS:
            raise ValueError("Oxygenator sevoflurane vaporizer requires CPB or a microporous oxygenator")
        self.state.sweep_sevo_percent = val

    def set_membrane_area(self, area_m2: float) -> None:
        if not self.is_active:
            raise ValueError("Membrane area can only be adjusted during active support")
        area = bounded_number("membrane area", area_m2, 0.2, 3.0, unit="m^2")
        self.state.membrane_area_m2 = area
        self.oxygenator.area_m2 = area

    def clear_last_pass(self) -> None:
        """Clear cached last oxygenator gas pass (e.g. when circuit flow stops)."""
        self._last_pass = None

    def set_heater_cooler(self, on: bool, target_c: float | None = None) -> None:
        if not self.is_active:
            raise ValueError("Heater-cooler can only be adjusted during active support")
        self.state.heater_cooler_on = bool(on)
        if target_c is not None:
            self.state.water_target_c = bounded_number("heater-cooler target", target_c, 18.0, 38.5, unit="°C")

    def set_venous_clamp(self, open_fraction: float) -> None:
        if not self.is_cpb:
            raise ValueError("Venous line clamp can only be adjusted during active CPB")
        self.state.venous_clamp_open = bounded_number("venous clamp", open_fraction, 0.0, 1.0)

    def set_aortic_cross_clamp(self, clamped: bool) -> None:
        clamped = bool(clamped)
        if clamped and not self.is_cpb:
            raise ValueError("Aortic cross-clamp can only be applied during active CPB")
        self.state.aortic_cross_clamp = clamped
        if not clamped:
            self.state.cardioplegia_arrest = False

    def set_cardioplegia_arrest(self, arrested: bool) -> None:
        arrested = bool(arrested)
        if arrested and not self.is_cpb:
            raise ValueError("Cardioplegia arrest can only be started during active CPB")
        if arrested and not self.state.aortic_cross_clamp:
            self.state.aortic_cross_clamp = True
        self.state.cardioplegia_arrest = arrested

    def set_lv_vent(self, l_min: float) -> None:
        val = bounded_number("LV vent flow", l_min, 0.0, 3.0, unit="L/min")
        if val > 0.0 and not self.is_arterial_support:
            raise ValueError("LV vent requires active VA ECMO or CPB")
        self.state.lv_vent_l_min = val

    def set_blood_gas_strategy(self, strategy: BloodGasManagement | str) -> None:
        if not self.is_cpb:
            raise ValueError("Blood-gas management strategy can only be changed during active CPB")
        self.state.blood_gas_strategy = self._parse_bg_strategy(strategy)

    def set_bypass_alarm_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self.is_arterial_support:
            raise ValueError("Bypass alarm mode requires active VA ECMO or CPB")
        self.state.bypass_alarm_mode = enabled

    def compute_heat_exchanger_watts(self, core_temp_c: float) -> float:
        """Sensible heat rate (W) delivered to blood by the circuit heater-cooler.

        Applies a <=10 °C water-to-blood clinical perfusion safety ceiling.
        """
        st = self.state
        if not self.is_active or not st.heater_cooler_on or st.blood_flow_l_min <= 1e-6:
            return 0.0
        # Water bath approaches target within safe gradient of core blood temperature.
        delta_t = clamp(st.water_target_c - core_temp_c, -self.config.hx_max_gradient_c, self.config.hx_max_gradient_c)
        return (st.blood_flow_l_min / 60.0) * _BLOOD_HEAT_CAP_J_PER_L_C * self.config.hx_effectiveness * delta_t

    def compute_cpb_reservoir_step(
        self,
        dt: float,
        patient_stressed_vol_ml: float,
        patient_mcfp_mmhg: float,
        patient_blood_volume_ml: float,
        min_patient_blood_volume_ml: float,
        *,
        venous_compliance_ml_mmhg: float = 125.0,
    ) -> tuple[float, float]:
        """Compute CPB passive venous drainage and arterial pump return over dt.

        Substeps at <= 0.25 s for step-size invariance. Returns
        (delta_patient_volume_ml, mean_actual_pump_flow_l_min).
        Strict conservation: delta_reservoir_ml = -delta_patient_volume_ml.
        """
        st = self.state
        cfg = self.config
        if not self.is_cpb or dt <= 0.0:
            return 0.0, st.blood_flow_l_min if self.is_active else 0.0

        n_sub = max(1, math.ceil(dt / 0.25))
        sub_dt = dt / n_sub
        res_ml = st.reservoir_volume_ml
        pat_bv = patient_blood_volume_ml
        stressed = max(0.0, patient_stressed_vol_ml)
        cv = max(10.0, venous_compliance_ml_mmhg)
        mcfp_offset = patient_mcfp_mmhg - stressed / cv
        start_res_ml = res_ml
        pump_sum = 0.0
        drain_last = 0.0
        pump_last = 0.0

        for _ in range(n_sub):
            avail_patient_ml = max(0.0, pat_bv - min_patient_blood_volume_ml)
            mcfp = max(0.0, stressed / cv + mcfp_offset)
            # Passive gravity/VAVD drainage from patient stressed venous pool (Magder and De Varennes 1998).
            fill_gate = clamp01(stressed / 250.0)
            driving_head = max(0.0, mcfp + cfg.cpb_drain_siphon_mmhg * fill_gate)
            drain_l_min = cfg.cpb_drain_conductance_l_min_mmhg * st.venous_clamp_open * driving_head * fill_gate
            max_drain_l_min = avail_patient_ml / 1000.0 * (60.0 / sub_dt)
            drain_l_min = min(drain_l_min, max_drain_l_min, 8.0)

            # Pump flow cannot draw reservoir below 0 mL; smooth low-level ramp below 100 mL prevents air entrainment.
            avail_res_ml = res_ml + drain_l_min * 1000.0 * (sub_dt / 60.0)
            max_pump_l_min = max(0.0, avail_res_ml / 1000.0 * (60.0 / sub_dt))
            low_level_gate = clamp01(avail_res_ml / 100.0)
            actual_pump = min(st.target_blood_flow_l_min * low_level_gate, max_pump_l_min)

            net_to_res_ml = (drain_l_min - actual_pump) * 1000.0 * (sub_dt / 60.0)
            next_res = clamp(res_ml + net_to_res_ml, 0.0, cfg.cpb_reservoir_max_ml)
            delta_res = next_res - res_ml
            res_ml = next_res
            pat_bv -= delta_res
            stressed = max(0.0, stressed - delta_res)
            pump_sum += actual_pump
            drain_last = drain_l_min
            pump_last = actual_pump

        st.reservoir_volume_ml = res_ml
        st.venous_drainage_l_min = drain_last
        st.blood_flow_l_min = pump_last
        st.reservoir_low_alarm = res_ml < cfg.cpb_reservoir_low_ml
        return -(res_ml - start_res_ml), pump_sum / n_sub

    def step_gas_exchange(
        self,
        dt: float,
        native_co_l_min: float,
        venous_co2_mmol_l: float,
        venous_o2_l_l: float,
        base_meq_l: float,
        hb_g_dl: float,
        temp_c: float,
        chemistry,
        *,
        pco2_v_guess: float = 46.0,
        ph_v_guess: float = 7.36,
        po2_v_guess: float = 40.0,
    ) -> OxygenatorPassResult | None:
        """Advance oxygenator gas transfer and update telemetry."""
        st = self.state
        if not self.is_active:
            self._last_pass = None
            return None
        if self.is_vv:
            result, recirc = self.oxygenator.solve_vv_exchange(
                st.blood_flow_l_min,
                native_co_l_min,
                st.sweep_gas_l_min,
                st.sweep_fio2,
                venous_co2_mmol_l,
                venous_o2_l_l,
                base_meq_l,
                hb_g_dl,
                temp_c,
                chemistry,
                cannulation=st.cannulation,
                sweep_fco2=st.sweep_fco2,
                area_m2=st.membrane_area_m2,
                pco2_v_guess=pco2_v_guess,
                ph_v_guess=ph_v_guess,
                po2_v_guess=po2_v_guess,
            )
            st.recirculation_fraction = recirc
            st.effective_blood_flow_l_min = st.blood_flow_l_min * (1.0 - recirc)
        else:
            result = self.oxygenator.solve_pass(
                st.blood_flow_l_min,
                st.sweep_gas_l_min,
                st.sweep_fio2,
                venous_co2_mmol_l,
                venous_o2_l_l,
                base_meq_l,
                hb_g_dl,
                temp_c,
                chemistry,
                sweep_fco2=st.sweep_fco2,
                area_m2=st.membrane_area_m2,
                pco2_in_guess=pco2_v_guess,
                ph_in_guess=ph_v_guess,
                po2_in_guess=po2_v_guess,
            )
            st.recirculation_fraction = 0.0
            st.effective_blood_flow_l_min = st.blood_flow_l_min

        self._last_pass = result
        st.o2_transfer_ml_min = result.o2_transfer_l_min * 1000.0
        st.co2_removal_ml_min = result.co2_removal_l_min * 1000.0
        st.pre_oxy_po2 = result.pre_po2
        st.pre_oxy_so2 = result.pre_so2
        st.post_oxy_po2 = result.post_po2
        st.post_oxy_so2 = result.post_so2
        return result
