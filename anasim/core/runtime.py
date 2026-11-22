from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, cast

from anasim.core.constants import (
    ARREST_CO_L_MIN,
    ARREST_CONFIRM_S,
    ARREST_HR_BPM,
    ARREST_MAP_MMHG,
    ARREST_PULSE_PRESSURE_MMHG,
    O2_FLUSH_FLOW_L_MIN,
    PULSELESS_VT_MAP_MMHG,
    VT_MIN,
)
from anasim.core.enums import PULSELESS_RHYTHMS, RhythmType
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.machine.ventilator import BAG_TI_MAX_S, VOLUME_TARGETED
from anasim.monitors.arterial import MIN_PULSE_PRESSURE_MMHG
from anasim.patient.pd.anesthesia import opioid_equivalent
from anasim.patient.pd.nmba import succinylcholine_twitch_fraction
from anasim.patient.pk_models import MidazolamPK, PropofolPKEleveld
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.toxicity import SILENT_BLOCK

from .monitors import step_monitors
from .projection import (
    PhysiologyStepState,
    StepGasCache,
    advance_tracheal_gas,
    assisted_positive_pressure,
    build_snapshot_from_models,
    cardiovascular_response_fractions,
    central_muscle_strength,
    circuit_connected,
    circuit_inspired_mixture,
    compute_pit_estimate,
    delivered_fraction,
    loss_of_response,
    mask_circuit_share,
    mask_leak,
    mask_seal_factor,
    project_runtime_physiology,
    step_coronary_from_state,
    step_respiration,
    stimulus_tolerance,
    sync_inhaled_agents,
    sync_inspired_gas,
    sync_pk_state,
    thermal_hypnotic_ce,
)
from .state import AirwayStatus, AirwayType

if TYPE_CHECKING:
    from .engine import SimulationEngine

logger = logging.getLogger(__name__)

# Mean airway pressure is smoothed before it sets intrathoracic pressure, while
# spontaneous inspiratory effort is breath-averaged across step sizes.
MEAN_PAW_TAU_S = 0.25
MEAN_EFFORT_TAU_S = 1.5
# Typical one-handed bag squeeze pressure that can vent surplus FGF through the
# APL valve; APL settings above 20 cmH2O (or FGF exceeding squeeze rate) trap
# gas and raise baseline PEEP.
BAG_VENT_PRESSURE_CMH2O = 20.0


def step_simulation(engine: SimulationEngine, dt: float) -> None:
    if dt <= 0.0:
        return
    state = engine.state
    update_shivering(engine, dt)
    engine.thermal.update_depth(state.temp_c, thermal_hypnotic_ce(engine), state.mac)
    engine._tol_current = stimulus_tolerance(engine)

    disturbance_complete = _disturbance_completes_during_step(engine, dt)
    # hemo.state includes a rhythm change made since the last step.
    hemo_pre = engine.hemo.state
    cardiac_output = hemo_pre.co
    disturbances = step_disturbances(engine, dt)
    update_pk_covariates(engine, cardiac_output)

    step_tci(engine, dt)
    update_airway_complications(engine, dt, rhythm_type=hemo_pre.rhythm_type)
    fi_sevo, fi_n2o = step_machine(engine, dt, co_now=cardiac_output)
    toxicity = engine.toxicity
    lidocaine = engine.pk["lidocaine"]
    toxicity.step_lipid(dt, lidocaine.circulation)
    lidocaine.effect_input_fraction = toxicity.effect_input_fraction
    step_pk(engine, dt, fi_sevo, fi_n2o, cardiac_output)
    toxicity.step(
        dt, neural_ce=lidocaine.state.ce_neuro, cardiac_ce=lidocaine.state.ce_cv, ph=state.ph,
        propofol_ce=state.propofol_ce, midazolam_ce=state.midazolam_ce,
        mac_sevo=state.mac_sevo, motor_strength=state.twitch / 100.0,
        cortical_function=engine.cerebral.cortical_function,
        mac_n2o=state.mac_n2o,
    )
    engine.thermal.metabolic_factor *= toxicity.metabolic_factor
    mh = engine.malignant_hyperthermia
    mh.step(dt, sevo_mac=state.mac_sevo, succinylcholine_cp=engine.pk["succinylcholine"].state.c1,
            dantrolene_ce=state.dantrolene_ce)
    sepsis_sev = engine.sepsis.severity
    engine.thermal.metabolic_factor += mh.excess_metabolism + 0.25 * sepsis_sev
    engine.thermal.extra_heat_w = engine.patient.weight * (
        mh.extra_heat_w_per_kg + 0.35 * sepsis_sev
    )
    engine.aeration.chest_wall_stiffness = mh.chest_wall_stiffness
    state.muscle_rigidity = mh.rigidity
    state.seizure_activity = toxicity.seizure_activity
    state.convulsions = toxicity.convulsion_intensity > 0.2
    physiology = step_physiology(engine, dt, disturbances)
    project_runtime_physiology(engine, physiology)
    hemo_state = physiology.hemo_state
    if hemo_state.compressions and hemo_state.co > 0.0:
        cpr_share = clamp01((hemo_state.co - hemo_state.intrinsic_co) / hemo_state.co)
        intrinsic_art_map = (
            engine.hemo.lumen_map(hemo_state.intrinsic_map, hemo_state.venous_pressure)
            if hemo_state.intrinsic_co > 0.0
            else hemo_state.venous_pressure
        )
        cpr_driving_map = hemo_state.venous_pressure + hemo_state.cpr_perfusion_map
        cerebral_map = (1.0 - cpr_share) * intrinsic_art_map + cpr_share * cpr_driving_map
        base_co_ref = engine.hemo.normal_base_co_l_min
        # Closed-chest CPR delivers ~20-34% of normal CBF: enough to protect against
        # rapid anoxic injury (Astrup 1981 >20% threshold) without restoring cortical
        # wakefulness (>35% electrical threshold).
        cpr_cbf_cap = clamp(1.35 * (hemo_state.co / max(0.1, base_co_ref)), 0.0, 0.34)
        flow_ceiling: float | None = 1.0 - cpr_share * (1.0 - cpr_cbf_cap)
    else:
        cerebral_map = hemo_state.map
        flow_ceiling = None
    engine.cerebral.step(
        dt,
        cerebral_map,
        state.hb_g_dl,
        state.sao2,
        state.pa_co2,
        state.ph,
        venous_pressure=hemo_state.venous_pressure,
        flow_ceiling=flow_ceiling,
        temp_c=state.temp_c,
        anesthetic_depth=clamp01(engine.thermal.depth_index),
        metabolic_stress=clamp01(state.seizure_activity),
    )
    step_monitors(engine, dt, physiology.hemo_state, physiology.resp_state, disturbances)
    step_temperature(engine, dt)
    check_cardiac_arrest(engine, dt)
    if disturbance_complete and engine.disturbance_active:
        engine.stop_disturbance()
    if engine.intubation_start_time is not None and engine.intubation_stimulus.is_complete(
        state.time + dt - engine.intubation_start_time,
    ):
        engine.intubation_start_time = None


def _manual_breath_delta_p(lung, vt_l: float, ti_s: float) -> float:
    """Peak airway pressure swing (cmH2O above PEEP) for a half-sine manual squeeze."""
    if vt_l <= 0.0 or ti_s <= 1e-6:
        return 0.0
    c = max(0.005, lung.compliance)
    r = max(0.0, lung.resistance)
    w = math.pi / ti_s
    elastance = 1.0 / c
    tau = max(1e-3, lung.viscoelastic_tau)
    rate = 1.0 / tau
    gain = lung.viscoelastic_ratio * elastance * 0.5 * w / (rate * rate + w * w)
    theta = math.pi - math.atan(r * c * w)
    peak = 0.0
    for t in (0.5 * ti_s, theta / w, 0.85 * ti_s, ti_s):
        wt = w * t
        flow = 0.5 * w * math.sin(wt)
        vol = 0.5 * (1.0 - math.cos(wt))
        p2 = gain * (rate * math.sin(wt) - w * math.cos(wt) + w * math.exp(-rate * t))
        paw = r * flow + elastance * vol + p2
        if paw > peak:
            peak = paw
    return vt_l * peak


def _advance_apl_pressure(
    pressure: float,
    dt: float,
    inflow_l_min: float,
    c_sys: float,
    target_ceiling: float,
    *,
    peep_vent: float | None = None,
    vent_max_l_min: float = 0.0,
) -> float:
    """Advance bag/APL baseline pressure over dt with optional squeeze venting above peep_vent."""
    r_in = (inflow_l_min / 60.0) / c_sys
    if peep_vent is None or vent_max_l_min <= 0.0:
        return clamp(pressure + r_in * dt, 0.0, target_ceiling)
    r_vent = (vent_max_l_min / 60.0) / c_sys
    rem = dt
    p = pressure
    if p < peep_vent - 1e-12:
        if r_in <= 0.0:
            return clamp(p + r_in * rem, 0.0, target_ceiling)
        t_hit = (peep_vent - p) / r_in
        if rem <= t_hit:
            return clamp(p + r_in * rem, 0.0, target_ceiling)
        p = peep_vent
        rem -= t_hit
    if p > peep_vent + 1e-12:
        r_net = r_in - r_vent
        if r_net >= 0.0:
            return clamp(p + r_net * rem, 0.0, target_ceiling)
        t_hit = (peep_vent - p) / r_net
        if rem <= t_hit:
            return clamp(p + r_net * rem, 0.0, target_ceiling)
        p = peep_vent
        rem -= t_hit
    if r_in > r_vent:
        p += (r_in - r_vent) * rem
    elif r_in < 0.0:
        p += r_in * rem
    return clamp(p, 0.0, target_ceiling)


def step_mechanics(engine: SimulationEngine, dt: float, connected: bool, vent_active: bool,
                   bag_mask_active: bool, *, collect_samples: bool = True) -> None:
    """Advance breathing through the workstation: ventilator, bag, or the patient's own effort."""
    resp = engine.resp.state
    lung = engine.resp_mech
    aeration = engine.aeration
    seal_scale = mask_seal_factor(engine)
    aeration.pressure_transmission = seal_scale if connected else 0.0
    aeration.unconscious = engine.state.loc
    aeration.lung_water = engine.hemo.lung_water_ml_kg
    # Alveolar nitrogen and argon slow absorption collapse.
    aeration.inert_fraction = clamp01(engine.resp.alveolar_fn2)
    absorbed_btps = engine.resp.consume_mechanical_absorption_l()
    bellows_empty = engine.circuit.bellows_fraction <= 0.05
    if absorbed_btps > 0.0 and (not assisted_positive_pressure(engine) or bellows_empty):
        engine.vent.absorb_lung_gas(absorbed_btps, lung)
    lung.effort.unconscious = engine.state.loc
    lung.effort.muscle.muscle_strength = (
        central_muscle_strength(engine) * engine.malignant_hyperthermia.respiratory_excursion_fraction
    )
    lung.effort.muscle.oxygen_fraction = engine.resp.oxygen_supply_fraction
    aeration.spontaneous_breathing = not resp.apnea and resp.vt > VT_MIN and engine.airway_status.patency > 0.5
    aeration.glottic_intact = engine.state.airway_mode != AirwayType.ETT
    # The patient's unassisted neural demand sets inspiratory effort; muscle strength
    # and fatigue cap mechanical pressure inside PatientEffort._begin().
    lung.effort.set_drive(resp.neural_rr, resp.neural_vt / 1000.0)
    source: str | None
    if vent_active:
        source = "vent"
    elif bag_mask_active:
        source = "bag"
    else:
        source = "spontaneous" if connected else None
    single_breath = (
        engine._manual_breath_requested
        and source == "spontaneous"
        and engine.apl_cmh2o > 0.0
    )
    engine._manual_breath_requested = False
    bag_full = engine.circuit.reservoir_volume >= engine.circuit.reservoir_capacity - 1e-6
    if source in ("spontaneous", "bag") and engine.apl_cmh2o > 0.0:
        apl_ceiling = (
            min(engine.apl_cmh2o, engine.airway_tuning.mask_seal_pressure)
            if engine.state.airway_mode == AirwayType.MASK
            else engine.apl_cmh2o
        )
        half_cap = 0.5 * max(1e-6, engine.circuit.reservoir_capacity)
        bag_tautness = (
            1.0
            if bag_full
            else clamp01((engine.circuit.reservoir_volume - half_cap) / half_cap)
        )
        if bag_tautness > 0.0:
            target_ceiling = apl_ceiling * bag_tautness
            frc_inflow_l_min = (
                max(0.0, aeration.frc - engine.resp.frc)
                * engine.resp._exchangeable_frc_fraction
                * clamp01(engine.airway_status.patency)
                * mask_circuit_share(engine)
                * engine.resp._btps_to_stpd
                * 60.0
                / dt
            )
            inflow_l_min = (
                max(0.0, engine.circuit.last_spill_flow) + frc_inflow_l_min
                if bag_full
                else (
                    engine.circuit.last_net_flow + frc_inflow_l_min
                    if engine.circuit.last_net_flow + frc_inflow_l_min > 0.0
                    else min(engine.circuit.last_net_flow, -0.5 * engine._apl_pressure)
                )
            )
            peep_vent: float | None = None
            vent_max_l_min = 0.0
            if source == "bag" and engine.bag_mask_rr > 0.0:
                period = 60.0 / engine.bag_mask_rr
                ti = min(BAG_TI_MAX_S, period / 3.0)
                delta_p = _manual_breath_delta_p(lung, engine.bag_mask_vt, ti)
                peep_vent = max(0.0, engine.apl_cmh2o - max(delta_p, BAG_VENT_PRESSURE_CMH2O))
                vent_max_l_min = max(engine.bag_mask_vt, 1.0) * engine.bag_mask_rr
            elif single_breath or engine.vent.manual_breath_active:
                delta_p = _manual_breath_delta_p(lung, engine.bag_mask_vt, BAG_TI_MAX_S)
                peep_vent = max(0.0, engine.apl_cmh2o - max(delta_p, BAG_VENT_PRESSURE_CMH2O))
                vent_max_l_min = max(engine.bag_mask_vt, 1.0) * 60.0 / BAG_TI_MAX_S
            engine._apl_pressure = _advance_apl_pressure(
                engine._apl_pressure,
                dt,
                inflow_l_min,
                0.05 + max(0.01, lung.compliance),
                target_ceiling,
                peep_vent=peep_vent,
                vent_max_l_min=vent_max_l_min,
            )
        else:
            engine._apl_pressure = 0.0
    else:
        engine._apl_pressure *= math.exp(-dt / 0.4)
        if engine._apl_pressure < 1e-3:
            engine._apl_pressure = 0.0
    cache = engine._step_gas
    bellows_frac = (
        cache.bellows_fraction
        if cache is not None and cache.bellows_fraction is not None
        else engine.circuit.bellows_fraction
    )
    reservoir_vol = (
        cache.reservoir_volume_l
        if cache is not None and cache.reservoir_volume_l is not None
        else engine.circuit.reservoir_volume
    )
    volume_delivery = (
        1.0 - mask_leak(engine)
        if (assisted_positive_pressure(engine) or single_breath)
        else mask_seal_factor(engine)
    )
    fresh_flow_btps_l_s = (
        engine.circuit.total_inflow()
        / engine.resp._btps_to_stpd
        / 60.0
    )
    mask_leak_gap = 1.0 - seal_scale
    engine.vent.step(
        dt, lung, source, bag=(engine.bag_mask_rr, engine.bag_mask_vt),
        collect_samples=collect_samples, bag_p_max=engine.apl_cmh2o,
        spontaneous_peep=engine._apl_pressure,
        bellows_fraction=bellows_frac,
        reservoir_volume=reservoir_vol / engine.resp._btps_to_stpd,
        volume_delivery=volume_delivery,
        effort_cap=engine.hemo.config.effort_pressure_max,
        fresh_flow_l_s=fresh_flow_btps_l_s,
        mask_leak_gap=mask_leak_gap,
        trigger_manual_breath=single_breath,
        cpr_active=engine.hemo.compressions,
    )


def update_shivering(engine: SimulationEngine, dt: float) -> None:
    """Update shivering intensity based on temperature, perfusion, and anesthetic state."""
    state = engine.state
    remi_effect = hill_function(state.opioid_ce, engine.resp.c50_remi, engine.resp.gamma_remi)
    # Shivering muscle is as sensitive to block as the adductor pollicis.
    muscle_factor = state.twitch / 100.0
    perfusing = is_perfusing(engine) and not state.cardiac_arrest
    state.shivering = float(
        engine.thermal.step_shivering(
            dt, state.temp_c, state.bis, remi_effect, muscle_factor,
            unconscious=state.loc, perfusing=perfusing,
        )
    )


def step_temperature(engine: SimulationEngine, dt: float) -> None:
    """Update core temperature from metabolic heat, environmental loss, fluids, and redistribution."""
    state = engine.state
    state.temp_c = float(engine.thermal.step_temperature(
        dt,
        state.temp_c,
        state.bair_hugger_target,
        state.surface_cooling,
        oxygen_supply_fraction=engine.resp.oxygen_supply_fraction,
        sepsis_severity=engine.sepsis.severity,
        perfusing=is_perfusing(engine),
    ))


def stimulus_effects(engine: SimulationEngine, dt: float) -> DisturbanceEffects:
    """Drug-attenuated effects of surgical and intubation stimulation over one step."""
    if not engine.disturbance_active and engine.intubation_start_time is None:
        return DisturbanceEffects()
    active = []
    if engine.disturbance_active:
        active.append((engine.disturbances, engine.disturbance_start_time))
    if engine.intubation_start_time is not None:
        active.append((engine.intubation_stimulus, engine.intubation_start_time))
    bis = svr = sv = heart_rate = 0.0
    for disturbance, started in active:
        spec = disturbance.spec
        if spec is None:
            continue
        t_rel = max(0.0, engine.state.time - started)
        stimulus = disturbance.compute_average(t_rel, t_rel + dt)
        # Movement/EEG arousal and cardiovascular attenuation are separate endpoints.
        pressure, hr = cardiovascular_response_fractions(engine, spec.sympathetic_intensity)
        bis += stimulus.bis * (1.0 - engine._tol_current)
        svr += stimulus.svr * pressure + spec.tonic_svr(t_rel + 0.5 * dt, 1000.0 * engine.hemo.tpr)
        sv += stimulus.sv * pressure
        heart_rate += stimulus.hr * hr
    return DisturbanceEffects(bis=bis, svr=svr, sv=sv, hr=heart_rate)


def step_disturbances(engine: SimulationEngine, dt: float) -> DisturbanceEffects:
    """Calculate disturbances and update event-driven volume state."""
    effects = stimulus_effects(engine, dt)
    hemo = engine.hemo
    if engine.active_hemorrhage:
        rate_sec = engine.hemorrhage_rate_ml_min / 60.0
        hemo.add_volume(-rate_sec * dt)

    warmed = engine.state.fluid_warmer
    core_temp = engine.state.temp_c
    fluid_temp = 37.0 if warmed else 21.0

    if engine.pending_infusions:
        remaining = []
        for infusion in engine.pending_infusions:
            rate_sec = infusion.rate_ml_min / 60.0
            amount_this_step = min(infusion.remaining_ml, rate_sec * dt)
            infusion.remaining_ml -= amount_this_step
            if amount_this_step > 0:
                if infusion.label == "lipid":
                    engine.toxicity.add_lipid(amount_this_step)
                hemo.add_volume(
                    amount_this_step,
                    hematocrit=infusion.hematocrit,
                    retention_fraction=infusion.retention_fraction,
                    label=infusion.label,
                )
                is_blood = infusion.hematocrit > 0.0 or infusion.label == "blood"
                infusate_temp = 37.0 if warmed else (6.0 if is_blood else 21.0)
                engine.thermal.add_fluid_heat_j(
                    amount_this_step, infusate_temp, core_temp, is_blood=is_blood,
                )
            if infusion.remaining_ml > 1e-3:
                remaining.append(infusion)
        engine.pending_infusions[:] = remaining

    if engine.lipid_infusion_ml_min > 0.0:
        amount = engine.lipid_infusion_ml_min * dt / 60.0
        hemo.add_volume(amount, retention_fraction=0.25, label="lipid")
        engine.toxicity.add_lipid(amount)
        engine.thermal.add_fluid_heat_j(amount, fluid_temp, core_temp)

    if engine.maintenance_fluid_rate_ml_min > 0:
        rate_sec = engine.maintenance_fluid_rate_ml_min / 60.0
        maint_ml = rate_sec * dt
        hemo.add_volume(maint_ml, hematocrit=0.0, label="crystalloid")
        engine.thermal.add_fluid_heat_j(maint_ml, fluid_temp, core_temp)

    hemo.anaphylaxis_severity = engine.anaphylaxis.step(dt)
    hemo.sepsis_severity = engine.sepsis.step(dt)

    return effects


def _disturbance_completes_during_step(engine: SimulationEngine, dt: float) -> bool:
    if not engine.disturbance_active:
        return False
    t_rel = max(0.0, engine.state.time - engine.disturbance_start_time)
    return engine.disturbances.is_complete(t_rel + dt)


def _tci_time_eps(sim_time: float) -> float:
    return 1e-9 * max(1.0, abs(sim_time))


def _flush_tci_to_time(engine: SimulationEngine, key: str, controller, sim_time: float) -> float:
    sampling_time = controller.sampling_time
    eps = _tci_time_eps(sim_time)
    acc = engine._tci_accumulators.get(key, 0.0)
    steps = int((acc + eps) / sampling_time)
    if steps > 0:
        for i in range(steps):
            remaining = acc - (i + 1) * sampling_time
            sample_time = sim_time if abs(remaining) <= eps else sim_time - remaining
            engine.infusion_rates[key] = controller.step(sim_time=sample_time)
        acc = max(0.0, acc - steps * sampling_time)
    if sim_time + eps >= controller.next_update_time:
        engine.infusion_rates[key] = controller.step(sim_time=sim_time)
        acc = 0.0
    return acc


def step_tci(engine: SimulationEngine, dt: float) -> None:
    """Advance TCI controllers on their own sampling clock."""
    sim_time = engine.state.time
    eps = _tci_time_eps(sim_time + dt)
    for key, controller in engine.tci.items():
        acc = _flush_tci_to_time(engine, key, controller, sim_time)
        if controller.next_update_time >= sim_time + dt - eps:
            engine._tci_accumulators[key] = acc + dt
        else:
            # Defer mid-step pulse transitions until step_pk advances PK to next_update_time.
            engine._tci_accumulators[key] = acc


def step_machine(
    engine: SimulationEngine, dt: float, co_now: float | None = None,
) -> tuple[float, float]:
    """Update machine state and return inspired volatile fractions."""
    circuit = engine.circuit
    resp = engine.resp
    btps_to_stpd = resp.btps_to_stpd
    if engine._o2_flush_remaining_s > 0.0 and dt > 0.0:
        flush_dt = min(dt, engine._o2_flush_remaining_s)
        circuit.o2_flush_flow = O2_FLUSH_FLOW_L_MIN * (flush_dt / dt)
        engine._o2_flush_remaining_s = max(0.0, engine._o2_flush_remaining_s - dt)
    else:
        circuit.o2_flush_flow = 0.0
    delivered_setting = (
        engine.vaporizer.step(dt, circuit.fgf_total()) if engine.config.sevoflurane_enabled else 0.0
    )
    circuit.vaporizer_setting = delivered_setting
    circuit.vaporizer_on = bool(
        engine.config.sevoflurane_enabled
        and (engine.vaporizer.state.is_on or delivered_setting > 0.0)
    )

    fi_sevo, fi_n2o = sync_inspired_gas(engine)
    if co_now is None:
        co_now = engine.hemo.state.co
    shunt_now = engine.aeration.shunt_fraction
    n2o_uptake = engine.pk_n2o.alveolar_uptake_l_min(co_now, shunt_now)
    sevo_uptake = engine.pk_sevo.alveolar_uptake_l_min(co_now, shunt_now)
    agent_uptake = n2o_uptake + sevo_uptake
    target_lung_vol = engine.aeration.frc
    circuit_share = mask_circuit_share(engine)
    apneic_limit: float | None = None
    leak_l_min = 0.0
    if engine.state.airway_mode == AirwayType.MASK and circuit_connected(engine):
        if assisted_positive_pressure(engine):
            if engine.vent.has_measured_breath:
                supported_mv = engine.vent.monitors.mv_supported
            elif engine.vent.is_on and engine.vent.settings.mode in VOLUME_TARGETED:
                supported_mv = engine.vent.settings.rr * engine.vent.settings.tv / 1000.0
            elif engine.bag_mask_active and not engine.vent.is_on and engine.apl_cmh2o > 0.0:
                supported_mv = engine.bag_mask_rr * engine.bag_mask_vt
            else:
                supported_mv = 0.0
            leak_l_min = max(0.0, supported_mv) * mask_leak(engine) * btps_to_stpd
        if engine.mask_seal < 0.999:
            cpap_cmh2o = (
                engine.vent.settings.peep
                if engine.vent.is_on and engine.vent.settings.mode == "CPAP"
                else (engine._apl_pressure if not engine.vent.is_on else 0.0)
            )
            if cpap_cmh2o > 0.0:
                exp_share = 1.0
                if engine.bag_mask_active and not engine.vent.is_on and engine.bag_mask_rr > 0.0 and engine.apl_cmh2o > 0.0:
                    period = 60.0 / engine.bag_mask_rr
                    exp_share = max(0.0, 1.0 - min(BAG_TI_MAX_S, period / 3.0) / period)
                leak_l_min += exp_share * 0.45 * cpap_cmh2o * (1.0 - engine.mask_seal) * btps_to_stpd
    rebreathed_share = 1.0
    exchange = None
    va = 0.0
    sevo_result = None
    n2o_result = None
    if circuit_share <= 0.0:
        tracheal_fio2, fi_sevo, fi_n2o, tracheal_fin2 = advance_tracheal_gas(engine, dt)
        uptake_o2 = uptake_sevo = uptake_n2o = uptake_n2 = 0.0
    else:
        airway = engine.airway_status
        patency = clamp01(airway.patency)
        raw_va = max(0.0, engine.state.va) if patency > 0.0 else 0.0
        va = resp.effective_va(raw_va, airway.vq_mismatch)
        va_circuit = va * circuit_share
        fresh_btps = circuit.total_inflow() / btps_to_stpd
        rebreathed_share = (
            clamp01(1.0 - fresh_btps / max(0.1, va_circuit))
            if va_circuit > 0.0
            else 0.0
        )
        composition = circuit.composition
        apneic_limit = circuit.deliverable_flow(dt) if circuit_share >= 0.999 else None
        terms = resp.passive_exchange_terms(
            target_lung_vol, patency, agent_uptake, dt, apneic_limit,
        )
        exchange = terms
        frc_inflow_stpd = max(0.0, terms.exhaled_vc_btps_l) * btps_to_stpd * 60.0 / dt if dt > 0.0 else 0.0
        inflow_stpd = terms.passive_stpd_l_min + frc_inflow_stpd
        tracheal_fio2, tracheal_fi_sevo, tracheal_fin2o, tracheal_fin2 = advance_tracheal_gas(
            engine, dt, inflow_stpd_l_min=inflow_stpd
        )
        if va <= 1e-4:
            circuit_fio2, circuit_fi_sevo, circuit_fin2o, circuit_fin2 = (
                tracheal_fio2, tracheal_fi_sevo, tracheal_fin2o, tracheal_fin2
            )
            fi_sevo = tracheal_fi_sevo
            fi_n2o = tracheal_fin2o
        else:
            circuit_fio2, circuit_fi_sevo, circuit_fin2o, circuit_fin2 = circuit_inspired_mixture(
                composition, tracheal_fio2, circuit_share
            )
            fi_sevo = circuit_share * circuit_fi_sevo
            fi_n2o = circuit_share * circuit_fin2o
        sevo_btps, sevo_result = engine.pk_sevo.preview_exchange(
            dt, fi_sevo, va, co_now, lung_volume_l=terms.gas_vol_l,
            exhaled_volume_change_l=terms.exhaled_vc_btps_l,
            shunt_fraction=shunt_now, companion_uptake_l_min=terms.companion_btps_l_min,
            passive_inflow_l_min=terms.passive_o2_btps_l_min,
            circuit_share=circuit_share, circuit_fi_agent=circuit_fi_sevo,
        )
        uptake_sevo = sevo_btps * btps_to_stpd
        n2o_btps, n2o_result = engine.pk_n2o.preview_exchange(
            dt, fi_n2o, va, co_now, lung_volume_l=terms.gas_vol_l,
            exhaled_volume_change_l=terms.exhaled_vc_btps_l,
            shunt_fraction=shunt_now, companion_uptake_l_min=terms.companion_btps_l_min,
            passive_inflow_l_min=terms.passive_o2_btps_l_min,
            circuit_share=circuit_share, circuit_fi_agent=circuit_fin2o,
        )
        uptake_n2o = n2o_btps * btps_to_stpd
        if circuit_share >= 0.999:
            uptake_n2 = resp.nitrogen_exchange_l_min(
                circuit_fin2, raw_va, airway.patency, airway.vq_mismatch, dt,
                apneic_flow_limit_l_min=apneic_limit, n2o_uptake_l_min=agent_uptake,
                lung_volume_l=target_lung_vol, cardiac_output=co_now, shunt_fraction=shunt_now,
                exchange=terms,
            )
        else:
            mixing = va * dt / (120.0 * terms.gas_vol_l) if terms.gas_vol_l > 0.0 else 0.0
            alveolar_n2 = resp.alveolar_fn2
            frc_stpd_l_min = terms.exhaled_vc_btps_l * btps_to_stpd * 60.0 / dt if dt > 0.0 else 0.0
            frc_n2 = circuit_fin2 if frc_stpd_l_min >= 0.0 else alveolar_n2
            washout_stpd = terms.washout_stpd_l * 60.0 / dt if dt > 0.0 else 0.0
            tidal_scale = va_circuit * btps_to_stpd / (1.0 + mixing)
            uptake_n2 = tidal_scale * (circuit_fin2 - alveolar_n2) + circuit_share * (
                frc_stpd_l_min * frc_n2 + terms.passive_stpd_l_min * circuit_fin2 - washout_stpd * alveolar_n2
            )
        uptake_o2 = circuit_share * resp.oxygen_exchange_l_min(
            circuit_fio2, raw_va,
            airway.patency, airway.vq_mismatch, dt,
            n2o_uptake_l_min=agent_uptake,
            apneic_flow_limit_l_min=apneic_limit,
            lung_volume_l=target_lung_vol,
            exchange=terms,
        )
        # During apneic bulk inflow, gas enters the conserved dead-space buffer at
        # circuit composition and leaves into the alveoli at tracheal composition.
        if va <= 1e-4 and inflow_stpd > 0.0:
            uptake_o2 += circuit_share * inflow_stpd * (composition.fio2 - tracheal_fio2)
            uptake_sevo += circuit_share * inflow_stpd * (composition.fi_agent - tracheal_fi_sevo)
            uptake_n2o += circuit_share * inflow_stpd * (composition.fin2o - tracheal_fin2o)
            uptake_n2 += circuit_share * inflow_stpd * (composition.fin2 - tracheal_fin2)
    pre_reservoir = circuit.reservoir_volume
    pre_bellows = circuit.bellows_fraction
    circuit.step(
        dt,
        uptake_o2,
        uptake_sevo,
        uptake_n2o,
        uptake_n2,
        leak_l_min=leak_l_min,
        rebreathed_share=rebreathed_share,
    )
    sync_inspired_gas(engine)
    engine._step_gas = StepGasCache(
        tracheal_fio2=tracheal_fio2,
        n2o_uptake_l_min=n2o_uptake,
        lung_volume_l=target_lung_vol,
        tracheal_fin2=tracheal_fin2,
        sevo_uptake_l_min=sevo_uptake,
        apneic_limit_l_min=apneic_limit,
        reservoir_volume_l=pre_reservoir,
        bellows_fraction=pre_bellows,
        exchange=exchange,
        va=va,
        co=co_now,
        sevo_result=sevo_result,
        n2o_result=n2o_result,
    )
    return fi_sevo, fi_n2o


def step_pk(engine: SimulationEngine, dt: float, fi_sevo: float, fi_n2o: float, co_curr: float) -> None:
    """Update pharmacokinetic models and synchronize their public state."""
    state = engine.state
    aeration = engine.aeration
    airway = engine.airway_status
    resp = engine.resp
    patency = clamp01(airway.patency)
    va = resp.effective_va(state.va, airway.vq_mismatch) if patency > 0.0 else 0.0
    cache = engine._step_gas
    if cache is not None and cache.exchange is not None:
        terms = cache.exchange
    else:
        if cache is not None:
            agent_uptake = cache.n2o_uptake_l_min + cache.sevo_uptake_l_min
        else:
            agent_uptake = (
                engine.pk_n2o.alveolar_uptake_l_min(co_curr, aeration.shunt_fraction)
                + engine.pk_sevo.alveolar_uptake_l_min(co_curr, aeration.shunt_fraction)
            )
        target_lung_vol = cache.lung_volume_l if cache is not None else aeration.frc
        sealed_fallback = circuit_connected(engine) and mask_circuit_share(engine) >= 0.999
        fallback_flow_cap = max(0.0, engine.circuit.deliverable_flow_l_min) if sealed_fallback else None
        terms = resp.passive_exchange_terms(
            target_lung_vol,
            patency,
            agent_uptake,
            dt,
            cache.apneic_limit_l_min if cache is not None else None,
            fallback_flow_cap=fallback_flow_cap,
        )
    sealed_circuit = circuit_connected(engine) and mask_circuit_share(engine) >= 0.999
    uptake_scale = engine.circuit.last_uptake_scale if sealed_circuit else 1.0
    va_sevo = va * uptake_scale if fi_sevo > engine.pk_sevo.state.p_alv else va
    va_n2o = va * uptake_scale if fi_n2o > engine.pk_n2o.state.p_alv else va
    if (
        cache is not None
        and cache.sevo_result is not None
        and va_sevo == cache.va
        and co_curr == cache.co
    ):
        engine.pk_sevo.commit_integration(cache.sevo_result, dt, temp_c=state.temp_c)
    else:
        engine.pk_sevo.step(
            dt, fi_sevo, va_sevo, co_curr, temp_c=state.temp_c,
            lung_volume_l=terms.gas_vol_l, shunt_fraction=aeration.shunt_fraction,
            companion_uptake_l_min=terms.companion_btps_l_min,
            passive_inflow_l_min=terms.passive_o2_btps_l_min,
            exhaled_volume_change_l=terms.exhaled_vc_btps_l,
        )
    if (
        cache is not None
        and cache.n2o_result is not None
        and va_n2o == cache.va
        and co_curr == cache.co
    ):
        engine.pk_n2o.commit_integration(cache.n2o_result, dt, temp_c=state.temp_c)
    else:
        engine.pk_n2o.step(
            dt, fi_n2o, va_n2o, co_curr, temp_c=state.temp_c,
            lung_volume_l=terms.gas_vol_l, shunt_fraction=aeration.shunt_fraction,
            companion_uptake_l_min=terms.companion_btps_l_min,
            passive_inflow_l_min=terms.passive_o2_btps_l_min,
            exhaled_volume_change_l=terms.exhaled_vc_btps_l,
        )
    sync_inhaled_agents(engine)

    rates = engine.infusion_rates
    sim_time = state.time
    time_eps = _tci_time_eps(sim_time + dt)
    mac_sevo = engine.pk_sevo.state.mac
    mac_n2o = engine.pk_n2o.state.mac
    segmented_keys = (
        {
            key
            for key, ctrl in engine.tci.items()
            if dt > 0.0 and ctrl.next_update_time < sim_time + dt - time_eps
        }
        if engine.tci
        else ()
    )

    for key, model in engine.pk.items():
        if segmented_keys and key in segmented_keys:
            continue
        rate = rates.get(key, 0.0)
        if key == "roc":
            _step_roc_drug(engine, dt, rate, mac_sevo, mac_n2o)
        elif rate or model.basal_input_rate or model.has_mass():
            model.step(dt, rate)
    for key, model in engine.pump_pk.items():
        if segmented_keys and key in segmented_keys:
            continue
        rate = rates.get(key, 0.0)
        if rate or model.basal_input_rate or model.has_mass():
            model.step(dt, rate)
    for key in segmented_keys:
        ctrl = engine.tci[key]
        t_cur = sim_time
        t_end = sim_time + dt
        while t_cur < t_end - time_eps:
            t_next = min(t_end, max(t_cur, ctrl.next_update_time))
            if t_next > t_cur + 1e-12:
                seg_dt = t_next - t_cur
                rate = rates.get(key, 0.0)
                _step_patient_drug(engine, key, seg_dt, rate, mac_sevo, mac_n2o)
                if key in engine.pump_pk:
                    engine.pump_pk[key].step(seg_dt, rate)
                engine._tci_accumulators[key] = engine._tci_accumulators.get(key, 0.0) + seg_dt
                t_cur = t_next
            if t_cur < t_end - time_eps:
                engine._tci_accumulators[key] = _flush_tci_to_time(engine, key, ctrl, t_cur)
                if ctrl.next_update_time <= t_cur + time_eps:
                    rem_dt = t_end - t_cur
                    rate = rates.get(key, 0.0)
                    _step_patient_drug(engine, key, rem_dt, rate, mac_sevo, mac_n2o)
                    if key in engine.pump_pk:
                        engine.pump_pk[key].step(rem_dt, rate)
                    engine._tci_accumulators[key] = engine._tci_accumulators.get(key, 0.0) + rem_dt
                    break
    sync_pk_state(engine)


def _step_roc_drug(
    engine: SimulationEngine, seg_dt: float, rate: float, mac_sevo: float, mac_n2o: float,
) -> None:
    roc = engine.pk["roc"]
    tof = engine.tof_pd
    has_sug = tof.sugammadex.state.c1 > 0.0 or tof.complex.state.c1 > 0.0
    if not (
        rate
        or roc.has_mass()
        or tof.ce
        or tof.ce_central
        or has_sug
        or tof.sugammadex.state.c2
        or tof.complex.state.c2
    ):
        return
    if seg_dt > 0.05 and has_sug:
        substeps = max(1, math.ceil(seg_dt / 0.05))
        h = seg_dt / substeps
        for _ in range(substeps):
            roc.step(h, rate)
            tof.step_recovery(h, roc, mac_sevo=mac_sevo, mac_n2o=mac_n2o)
        return
    roc.step(seg_dt, rate)
    tof.step_recovery(seg_dt, roc, mac_sevo=mac_sevo, mac_n2o=mac_n2o)


def _step_patient_drug(
    engine: SimulationEngine, drug_key: str, seg_dt: float, rate: float, mac_sevo: float, mac_n2o: float,
) -> None:
    if drug_key == "roc":
        _step_roc_drug(engine, seg_dt, rate, mac_sevo, mac_n2o)
        return
    engine.pk[drug_key].step(seg_dt, rate)


def update_pk_covariates(engine: SimulationEngine, co_curr: float) -> None:
    """Update drug-interaction and hemodynamic PK covariates.

    Propofol slows norepinephrine and midazolam disposition; midazolam slows
    propofol clearance. Hemodynamic covariates are blood volume, cardiac
    output, and renal perfusion.
    """
    propofol = cast(PropofolPKEleveld, engine.pk["propofol"])
    midazolam = cast(MidazolamPK, engine.pk["midazolam"])
    engine.pk_nore.update_propofol(propofol.state.c1)
    # Both covariates use the concentrations before either parameter update.
    propofol_cp = propofol.state.c1
    midazolam_cp = midazolam.state.c1
    propofol.update_midazolam(midazolam_cp)
    midazolam.update_propofol(propofol_cp)
    base_bv = engine.hemo.blood_volume_0
    blood_lost = engine.hemo.total_blood_out_ml - engine._pk_blood_out_ml
    if blood_lost > 0.0:
        pre_loss_bv = max(500.0, engine.hemo.blood_volume + blood_lost)
        engine.toxicity.lipid_ml *= max(0.0, 1.0 - blood_lost / pre_loss_bv)
        for model in (*engine.pk.values(), engine.tof_pd.sugammadex, engine.tof_pd.complex):
            model.remove_blood(blood_lost / 1000.0, base_bv / 1000.0)
        engine._pk_blood_out_ml = engine.hemo.total_blood_out_ml
    # Heart failure's low output slows flow-dependent clearance. Pregnancy's
    # added output mostly perfuses the uterus and kidneys, so it is the reference.
    base_co = engine.hemo.normal_base_co_l_min
    if base_bv <= 0.0 or base_co <= 0.0:
        return
    # Below 30% of resting output, as in arrest and CPR, flow limits drug transport.
    circulation = engine.hemo.circulation_fraction(co_curr)
    if circulation != propofol.circulation:
        for model in (*engine.pk.values(), engine.tof_pd.sugammadex, engine.tof_pd.complex):
            model.circulation = circulation
    v_ratio = clamp(engine.hemo.blood_volume / base_bv, 0.1, 2.0)
    co_ratio = clamp(co_curr / base_co, 0.1, 2.0)
    renal_perf = engine.hemo.renal_perfusion
    temp_c = engine.state.temp_c

    last_scale = engine._pk_hemo_scale_cache
    # Changes below 0.1% leave concentrations unchanged at display precision.
    if last_scale is not None:
        lv, lc, lr, lt = last_scale
        if (
            abs(v_ratio - lv) <= max(1e-3, 1e-3 * max(abs(v_ratio), abs(lv)))
            and abs(co_ratio - lc) <= max(1e-3, 1e-3 * max(abs(co_ratio), abs(lc)))
            and abs(renal_perf - lr) <= max(1e-3, 1e-3 * max(abs(renal_perf), abs(lr)))
            and abs(temp_c - lt) <= max(1e-3, 1e-3 * max(abs(temp_c), abs(lt)))
        ):
            return

    bv_l = base_bv / 1000.0
    clamped_t = clamp(temp_c, 25.0, 38.5)
    temp_met = 2.0 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0
    temp_dist = 1.6 ** ((clamped_t - 37.0) / 10.0) if clamped_t != 37.0 else 1.0
    for model in engine.pk.values():
        model.update_hemodynamics(
            v_ratio, co_ratio, renal_perf,
            blood_volume_l=bv_l, temp_c=temp_c, temp_met=temp_met, temp_dist=temp_dist,
        )
    tof = engine.tof_pd
    tof._temp_c = temp_c
    for model in (tof.sugammadex, tof.complex):
        model.update_hemodynamics(
            v_ratio, co_ratio, renal_perf,
            blood_volume_l=bv_l, temp_c=temp_c, temp_met=temp_met, temp_dist=temp_dist,
        )
    engine._pk_hemo_scale_cache = (v_ratio, co_ratio, renal_perf, temp_c)


def update_airway_complications(
    engine: SimulationEngine, dt: float, *, rhythm_type: RhythmType | None = None,
) -> None:
    """Update airway obstruction/bronchospasm/laryngospasm state."""
    state = engine.state
    tol = engine._tol_current
    # Laryngoscopy or light extubation can provoke laryngospasm; an aborted
    # fully awake intubation stops at oropharyngeal gagging before the glottis.
    dist_stim = (
        engine.disturbance_active
        and engine.disturbances.stimulates_airway(state.time - engine.disturbance_start_time)
    )
    intub_stim = (
        engine.intubation_start_time is not None
        and state.loc >= 0.08
        and engine.intubation_stimulus.stimulates_airway(state.time - engine.intubation_start_time)
    )
    extub_stim = state.time < engine._extubation_stimulus_until
    stim_active = engine.auto_laryngospasm_enabled and (dist_stim or intub_stim or extub_stim)

    depolarizing_strength = succinylcholine_twitch_fraction(state.succinylcholine_ce, central=True)
    muscle_factor = central_muscle_strength(engine)

    airway_tuning = engine.airway_tuning
    splint_pressure = 0.0
    if state.airway_mode == AirwayType.MASK:
        vent = engine.vent
        if vent.is_on:
            splint_pressure = vent.settings.peep
            if vent.settings.mode != "CPAP":
                splint_pressure = max(splint_pressure, vent.monitors.paw_mean_recent)
        elif engine.bag_mask_active:
            splint_pressure = max(engine._apl_pressure, vent.monitors.paw_mean_recent)
        else:
            splint_pressure = engine._apl_pressure
        splint_pressure *= engine.mask_seal * clamp01(engine.circuit.bellows_fraction / 0.05)

    rhythm_now = engine.hemo.state.rhythm_type if rhythm_type is None else rhythm_type
    brainstem_viable = (
        engine.cerebral.respiratory_drive > 0.1
        and not state.cardiac_arrest
        and rhythm_now not in PULSELESS_RHYTHMS
    )
    laryng_target = 0.0
    if state.airway_mode != AirwayType.ETT and stim_active and brainstem_viable:
        light_factor = clamp01(1.0 - tol)
        laryng_target = clamp01(light_factor * muscle_factor * engine.cerebral.respiratory_drive)

    stage_ii_sustained = (
        brainstem_viable
        and state.loc >= 0.20
        and muscle_factor > 0.2
        and splint_pressure < 10.0
        and tol < 0.65
    )
    tau_off = 40.0 if stage_ii_sustained else airway_tuning.laryngospasm_tau_off
    tau = airway_tuning.laryngospasm_tau_on if laryng_target > engine.laryngospasm_severity else tau_off
    if tau > 0:
        engine.laryngospasm_severity += (laryng_target - engine.laryngospasm_severity) * -math.expm1(-dt / tau)
    engine.laryngospasm_severity = clamp01(engine.laryngospasm_severity)

    upper_obstruction = engine.airway_obstruction_manual
    if state.airway_mode != AirwayType.ETT:
        # Mask CPAP or positive-pressure breaths hold the pharynx open. Mean
        # pressure over recent breaths includes any unassisted breaths between them.
        relief = clamp01(splint_pressure / airway_tuning.collapse_relief_pressure)
        unconscious = loss_of_response(engine, ketamine=False)
        nondepol_tone = engine.tof_pd.pharyngeal_tone_fraction(state.mac_sevo, state.mac_n2o)
        depol_tone = 0.46 + 0.54 * depolarizing_strength
        tone = min(nondepol_tone, depol_tone)
        # Passive pharyngeal collapse stays below complete airtight glottic occlusion
        # (matching non-depolarizing reserve 0.40 / 0.46 ~= 0.87) so apneic O2 insufflation persists.
        sedated_collapse = min(0.87, airway_tuning.unsupported_collapse_max * unconscious / tone)
        # Severe awake residual neuromuscular block (TOF < 35% or depolarizing weakness)
        # causes upper-airway collapse even while conscious, while TOF 40-85% preserves
        # resting awake patency per Herbstreit 2009.
        tof_frac = engine.tof_pd.compute_tof_from_ce(
            engine.tof_pd.ce, state.mac_sevo, state.mac_n2o
        ) / 100.0
        severe_weakness = clamp01((0.35 - min(tof_frac, depolarizing_strength)) / 0.35)
        awake_nmba_collapse = 0.65 * severe_weakness
        collapse = max(sedated_collapse, awake_nmba_collapse) * (1.0 - relief)
        upper_obstruction = max(upper_obstruction, engine.laryngospasm_severity, collapse)
    upper_obstruction = clamp01(upper_obstruction)

    bronch = 1.0 - (1.0 - engine.bronchospasm_manual) * (1.0 - engine.anaphylaxis.severity)
    # Epinephrine relieves bronchospasm via beta-2 receptors (antagonized by
    # non-selective labetalol); sevoflurane and ketamine add direct bronchodilation.
    beta2_dr = 1.0 + max(0.0, state.labetalol_ce) / engine.hemo.config.labetalol_kb_beta
    epi_dil = airway_tuning.epi_bronchodilation_max * hill_function(
        state.epi_ce, airway_tuning.epi_bronchodilation_c50 * beta2_dr, 1.0
    )
    sevo_dil = 0.28 * hill_function(state.mac_sevo, 0.8, 1.5)
    ket_dil = 0.32 * hill_function(state.ketamine_ce, 1.0, 1.5)
    bronchodilation = 1.0 - (1.0 - epi_dil) * (1.0 - sevo_dil) * (1.0 - ket_dil)
    bronch *= 1.0 - bronchodilation
    bronch = clamp01(bronch)

    base_r = engine._base_airway_resistance
    # Resistance diverges as the airway closes. The finite denominator keeps
    # the exact mechanics solver conditioned at complete occlusion.
    r_upper = airway_tuning.upper_resistance_gain * upper_obstruction / max(1e-6, 1.0 - upper_obstruction)
    r_bronch = airway_tuning.bronch_resistance_gain * bronch * (
        1.0 + 0.10 * bronch * bronch + 0.85 * (engine.anaphylaxis.severity ** 1.5) * bronch * bronch
    )
    engine.resp_mech.reference_resistance = base_r
    engine.resp_mech.resistance = base_r + r_upper + r_bronch
    engine.resp_mech.bronchospasm = bronch
    engine.resp_mech.bronch_resistance = r_bronch

    engine.airway_status = AirwayStatus(
        patency=clamp(1.0 - upper_obstruction, 0.0, 1.0),
        ventilation_efficiency=clamp(
            1.0
            - airway_tuning.vent_efficiency_bronch_weight * bronch
            - airway_tuning.vent_efficiency_upper_weight * upper_obstruction,
            airway_tuning.vent_efficiency_min,
            1.0,
        ),
        capno_obstruction=clamp01(
            airway_tuning.capno_obstruction_upper_weight * upper_obstruction
            + airway_tuning.capno_obstruction_bronch_weight * engine.resp_mech.expiratory_obstruction
        ),
        vq_mismatch=clamp01(
            airway_tuning.vq_mismatch_bronch_weight * bronch
            + airway_tuning.vq_mismatch_upper_weight * upper_obstruction
        ),
    )

    state.airway_obstruction = upper_obstruction
    state.bronchospasm = bronch
    state.laryngospasm = engine.laryngospasm_severity


def step_physiology(
    engine: SimulationEngine, dt: float, disturbances: DisturbanceEffects, *, collect_samples: bool = True,
) -> PhysiologyStepState:
    """Advance physiology models and return a projected runtime snapshot."""
    state = engine.state
    connected = circuit_connected(engine)
    vent_active = connected and engine.vent.is_on
    bag_mask_active = engine.bag_mask_active and connected and not vent_active
    step_mechanics(engine, dt, connected, vent_active, bag_mask_active, collect_samples=collect_samples)
    vent = engine.vent
    spirometry = vent.monitors

    alpha_paw = 1.0 - math.exp(-dt / MEAN_PAW_TAU_S)
    mean_paw = vent.airway_pressure_area / dt * delivered_fraction(engine)
    engine.current_mean_paw = (1 - alpha_paw) * engine.current_mean_paw + alpha_paw * mean_paw
    alpha_effort = -math.expm1(-dt / MEAN_EFFORT_TAU_S)
    step_effort = min(vent.capped_muscle_pressure_area / dt, engine.hemo.config.effort_pressure_max)
    engine.current_mean_effort += alpha_effort * (step_effort - engine.current_mean_effort)

    pit_estimate = compute_pit_estimate(engine, dt)
    hemo_pit = compute_pit_estimate(engine, dt, smoothed=True)

    applied_peep = engine.resp_mech.peep if connected else 0.0
    intrinsic_peep = (
        spirometry.auto_peep
        if connected
        else (
            max(0.0, engine.resp_mech.elastic_pressure)
            if not vent.inspiring and engine.resp_mech.effort.pmus() == (0.0, 0.0)
            else 0.0
        )
    )
    total_peep_effect = applied_peep + intrinsic_peep
    # Sevoflurane cardiovascular effects follow end-tidal MAC through the model's own ke0.
    mac_sevo = engine.pk_sevo.state.p_alv * 100.0 / engine.pk_sevo.mac_age
    mac_n2o = engine.pk_n2o.state.p_alv * 100.0 / engine.pk_n2o.mac_age
    resp_state = step_respiration(engine, dt, engine.hemo.state.co)
    step_lactate(engine, dt)

    hemo = engine.hemo
    hemo_pre = hemo.state
    # Without flow, arterial pressure equals venous pressure, so coronary flow stops.
    step_coronary_from_state(engine, dt, hemo_pre, resp_state.sao2)

    # Plasma remifentanil and propofol reach hemodynamic effect sites inside the model;
    # slower opioids enter at their own effect sites.
    hemo.post_shock_asystole = engine.arrhythmias.post_shock_asystole
    hemo_state = engine.hemo.step(
        dt,
        state.propofol_cp,
        opioid_equivalent(state.remi_cp, state.fentanyl_ce, state.morphine_ce, state.hydromorphone_ce),
        state.nore_ce,
        pit=hemo_pit,
        paco2=resp_state.pa_co2,
        pao2=resp_state.p_arterial_o2,
        dist_hr=disturbances.hr,
        dist_sv=disturbances.sv,
        dist_svr=disturbances.svr,
        crisis_hr=(
            engine.malignant_hyperthermia.heart_rate_increase
            + engine.toxicity.heart_rate_increase
        ),
        crisis_svr=engine.toxicity.svr_increase,
        mac_sevo=mac_sevo,
        ce_epi=state.epi_ce,
        ce_phenyl=state.phenyl_ce,
        ce_vaso=state.vaso_ce,
        ce_dobu=state.dobu_ce,
        ce_mil=state.mil_ce,
        ce_esmolol=state.esmolol_ce,
        ce_labetalol=state.labetalol_ce,
        ce_glyco=state.glyco_ce,
        ce_atropine=state.atropine_ce,
        ce_ephedrine=state.ephedrine_ce,
        ce_amiodarone=state.amiodarone_ce,
        ce_amiodarone_vasc=engine.pk["amiodarone"].state.ce_cv,
        ce_amiodarone_hr=engine.pk["amiodarone"].state.ce_hr,
        sodium_block=engine.toxicity.conduction_block,
        toxic_contractility_loss=engine.toxicity.contractility_loss,
        ce_ketamine=engine.pk["ketamine"].state.ce_cv,
        ce_ketamine_hr=engine.pk["ketamine"].state.ce_hr,
        temp_c=state.temp_c,
        peep_cmH2O=total_peep_effect,
        sao2=resp_state.sao2,
        remifentanil_cp=state.remi_cp,
        ph=resp_state.ph,
        ph_37=resp_state.ph_37,
        ce_midazolam=state.midazolam_ce,
        ce_etomidate=state.etomidate_ce,
        mac_n2o=mac_n2o,
        shivering=state.shivering,
    )
    fluid_circulation = hemo.circulation_fraction(hemo_state.co)
    hemo.fluids.step(
        dt, ph=resp_state.ph_37, paco2=resp_state.pa_co2_37, resting_paco2=engine.resp.paco2_setpoint,
        renal_function=engine.patient.renal_function, renal_perfusion=hemo.renal_perfusion,
        hepatic_function=engine.patient.hepatic_function, oxygen_supply=engine.resp.oxygen_supply_fraction,
        epinephrine=state.epi_ce,
        succinylcholine_ce=state.succinylcholine_ce,
        mh_severity=engine.malignant_hyperthermia.severity,
        circulation=fluid_circulation,
    )
    hypoxia, energy_failure, silent = myocardial_substrate(engine)
    perfusing = is_perfusing(engine, hemo_state)
    rhythm = engine.arrhythmias.step(
        dt, hemo.rhythm_type, ischemia=hemo.coronary.ischemia, hypoxia=hypoxia,
        energy_failure=energy_failure, electrically_silent=silent,
        perfusing=perfusing,
        lidocaine=engine.pk["lidocaine"].state.ce_cv, amiodarone=state.amiodarone_ce,
        hyperkalemia=hemo.hyperkalemia_arrhythmia,
        hypothermia=clamp01((29.5 - state.temp_c) / 3.5),
        energy_reserve=hemo.coronary.energy_reserve,
        compressions=hemo.compressions,
    )
    if rhythm != hemo.rhythm_type:
        hemo.rhythm_type = rhythm
        hemo_state = hemo.state

    return build_snapshot_from_models(
        engine, hemo_state, resp_state, pit_estimate=pit_estimate,
    )


def myocardial_substrate(engine: SimulationEngine) -> tuple[float, float, bool]:
    """Myocardial hypoxia, energy failure, and electrical silence, including lidocaine and K+ toxicity."""
    coronary, toxicity, hemo = engine.hemo.coronary, engine.toxicity, engine.hemo
    return (
        max(hemo.myocardial_hypoxia, toxicity.cardiac_stress),
        coronary.energy_failure,
        coronary.electrically_silent or toxicity.electrically_silent or hemo.hyperkalemia_block >= SILENT_BLOCK,
    )


def _intrinsic_pulse_pressure(engine: SimulationEngine, hemo_state) -> float:
    return engine.arterial_waveform.reference_pulse_pressure(
        hemo_state.intrinsic_map,
        hemo_state.sv,
        hemo_state.hr,
        pulse_pressure_factor=hemo_state.pulse_pressure_factor,
        static_pressure=hemo_state.venous_pressure,
    )


def _is_pulseless(
    engine: SimulationEngine,
    hemo_state,
    *,
    hr: float | None = None,
    intrinsic_co: float | None = None,
    pulse_pressure: float | None = None,
) -> bool:
    """Whether the heart's own beats fail to generate effective systemic perfusion."""
    rhythm = hemo_state.rhythm_type
    eff_hr = hemo_state.hr if hr is None else hr
    eff_co = hemo_state.intrinsic_co if intrinsic_co is None else intrinsic_co
    if rhythm in PULSELESS_RHYTHMS or eff_hr < ARREST_HR_BPM:
        return True
    pp = _intrinsic_pulse_pressure(engine, hemo_state) if pulse_pressure is None else pulse_pressure
    imap = hemo_state.intrinsic_map
    no_pulse = pp < ARREST_PULSE_PRESSURE_MMHG
    vasoplegic_collapse = imap < ARREST_MAP_MMHG and hemo_state.svr < 10.0
    if no_pulse and (eff_co <= ARREST_CO_L_MIN or vasoplegic_collapse):
        return True
    return rhythm == RhythmType.VTACH and (
        pp <= MIN_PULSE_PRESSURE_MMHG or (no_pulse and imap < PULSELESS_VT_MAP_MMHG)
    )


def is_perfusing(engine: SimulationEngine, hemo_state=None) -> bool:
    """Whether the heart's own beats produce effective systemic perfusion."""
    state = engine.hemo.state if hemo_state is None else hemo_state
    return not _is_pulseless(engine, state)


def step_lactate(engine: SimulationEngine, dt: float) -> None:
    """Advance lactate from this step's O2 balance and update extracellular buffer base."""
    if dt <= 0.0:
        return
    resp = engine.resp
    fluids = engine.hemo.fluids
    engine.lactate.update_fluid_volume(fluids.lactate_volume_l, fluids.consume_lactate_loss_l())
    base_removed = engine.lactate.step(
        dt, resp.oxygen_deficit_l, resp.oxygen_supply_fraction,
        epinephrine_ng_ml=engine.state.epi_ce,
        sepsis_severity=engine.hemo.sepsis_severity,
        infused_lactate_mmol=fluids.infused_l_lactate_mmol,
    )
    resp.oxygen_deficit_l = 0.0
    if base_removed != 0.0:
        resp.remove_buffer_base(
            base_removed,
            venous_only=(base_removed > 0.0 and resp.oxygen_supply_fraction < 0.99),
        )


def check_cardiac_arrest(engine: SimulationEngine, dt: float) -> None:
    """Flag arrest after 15 s without effective pulses from the heart itself."""
    state = engine.state
    hemo_state = engine.hemo.state
    rhythm = hemo_state.rhythm_type
    # Compressions mask the heart's own pulse pressure.
    pulse_pressure = (
        _intrinsic_pulse_pressure(engine, hemo_state)
        if state.chest_compressions
        else state.sbp - state.dbp
    )
    pulseless = _is_pulseless(
        engine,
        hemo_state,
        hr=state.hr,
        intrinsic_co=state.intrinsic_co,
        pulse_pressure=pulse_pressure,
    )
    engine._pulseless_s = engine._pulseless_s + dt if pulseless else 0.0
    if not pulseless:
        state.cardiac_arrest = False
        state.arrest_reason = ""
        state.alarms.pop("Arrest", None)
        return
    if engine._pulseless_s < ARREST_CONFIRM_S:
        return

    if rhythm in PULSELESS_RHYTHMS:
        reason = rhythm.value
    elif rhythm == RhythmType.VTACH:
        reason = "Pulseless ventricular tachycardia"
    elif state.hr < ARREST_HR_BPM:
        reason = f"Extreme bradycardia (HR < {ARREST_HR_BPM:g} bpm)"
    else:
        reason = "Pulseless electrical activity (negligible forward flow)"
    changed = not state.cardiac_arrest or state.arrest_reason != reason
    state.cardiac_arrest = True
    state.arrest_reason = reason
    if rhythm not in (RhythmType.ASYSTOLE, RhythmType.VFIB, RhythmType.VTACH):
        state.alarms["Arrest"] = {"low": False, "high": True}
    if changed:
        logger.warning("Cardiac arrest: %s (MAP=%.1f mmHg, HR=%.1f bpm)", reason, state.map, state.hr)
