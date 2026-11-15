from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, cast

from anasim.core.constants import (
    ARREST_CO_L_MIN,
    ARREST_CONFIRM_S,
    ARREST_HR_BPM,
    ARREST_PULSE_PRESSURE_MMHG,
    BAG_P_MAX_DEFAULT,
    CIRCULATION_LIMIT_FRACTION,
    PULSELESS_VT_MAP_MMHG,
)
from anasim.core.enums import RhythmType
from anasim.core.utils import clamp, clamp01, hill_function
from anasim.machine.ventilator import VOLUME_TARGETED
from anasim.patient.pd.anesthesia import opioid_equivalent
from anasim.patient.pd.nmba import succinylcholine_twitch_fraction
from anasim.patient.pk_models import MidazolamPK, PropofolPKEleveld
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.toxicity import SILENT_BLOCK

from .monitors import step_monitors
from .projection import (
    PhysiologyStepState,
    StepGasCache,
    advance_tracheal_fio2,
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


def step_simulation(engine: SimulationEngine, dt: float) -> None:
    if dt <= 0.0:
        return
    state = engine.state
    update_shivering(engine, dt)
    engine.thermal.update_depth(state.temp_c, thermal_hypnotic_ce(engine), state.mac)
    engine._tol_current = stimulus_tolerance(engine)

    disturbance_complete = _disturbance_completes_during_step(engine, dt)
    disturbances = step_disturbances(engine, dt)
    # hemo.state includes a rhythm change made since the last step.
    cardiac_output = engine.hemo.state.co
    update_pk_covariates(engine, cardiac_output)

    step_tci(engine, dt)
    update_airway_complications(engine, dt)
    fi_sevo, fi_n2o = step_machine(engine, dt)
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
    engine.cerebral.step(dt, physiology.hemo_state.perfusion_map, state.hb_g_dl, state.sao2, state.pa_co2, state.ph)
    step_monitors(engine, dt, physiology.hemo_state, physiology.resp_state, disturbances)
    step_temperature(engine, dt)
    check_cardiac_arrest(engine, dt)
    if disturbance_complete and engine.disturbance_active:
        engine.stop_disturbance()
    if engine.intubation_start_time is not None and engine.intubation_stimulus.is_complete(
        state.time + dt - engine.intubation_start_time,
    ):
        engine.intubation_start_time = None


def step_mechanics(engine: SimulationEngine, dt: float, connected: bool, vent_active: bool,
                   bag_mask_active: bool) -> None:
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
    if absorbed_btps > 0.0 and not (vent_active or bag_mask_active):
        engine.vent.absorb_lung_gas(absorbed_btps, lung)
    lung.effort.unconscious = engine.state.loc
    lung.effort.muscle.muscle_strength = (
        central_muscle_strength(engine) * engine.malignant_hyperthermia.respiratory_excursion_fraction
    )
    lung.effort.muscle.oxygen_fraction = engine.resp.oxygen_supply_fraction
    aeration.spontaneous_breathing = not resp.apnea and resp.vt > 100.0 and engine.airway_status.patency > 0.5
    # The patient's unassisted breathing sets inspiratory effort for the breaths that follow.
    lung.effort.set_drive(resp.effort_rr, resp.effort_vt / 1000.0)
    source: str | None
    if vent_active:
        source = "vent"
    elif bag_mask_active:
        source = "bag"
    else:
        source = "spontaneous" if connected else None
    bag_full = engine.circuit.reservoir_volume >= engine.circuit.reservoir_capacity - 1e-6
    if source == "spontaneous" and engine.bag_mask_p_max != BAG_P_MAX_DEFAULT:
        apl_ceiling = (
            min(engine.bag_mask_p_max, engine.airway_tuning.mask_seal_pressure)
            if engine.state.airway_mode == AirwayType.MASK
            else engine.bag_mask_p_max
        )
        half_cap = 0.5 * max(1e-6, engine.circuit.reservoir_capacity)
        bag_tautness = (
            1.0
            if bag_full
            else clamp01((engine.circuit.reservoir_volume - half_cap) / half_cap)
        )
        if bag_tautness > 0.0:
            target_ceiling = apl_ceiling * bag_tautness
            inflow_l_min = (
                max(0.0, engine.circuit.last_spill_flow)
                if bag_full
                else max(0.0, engine.circuit.last_net_flow)
            )
            excess_l_s = inflow_l_min / 60.0
            if engine._apl_pressure <= target_ceiling:
                engine._apl_pressure = min(
                    target_ceiling,
                    engine._apl_pressure + excess_l_s * dt / (0.05 + max(0.01, lung.compliance)),
                )
            else:
                engine._apl_pressure = target_ceiling
        else:
            engine._apl_pressure = 0.0
    else:
        engine._apl_pressure *= math.exp(-dt / 0.4)
        if engine._apl_pressure < 1e-3:
            engine._apl_pressure = 0.0
    volume_delivery = 1.0 - mask_leak(engine) if vent_active else 1.0
    engine.vent.step(
        dt, lung, source, bag=(engine.bag_mask_rr, engine.bag_mask_vt),
        collect_samples=True, bag_p_max=engine.bag_mask_p_max,
        spontaneous_peep=engine._apl_pressure,
        bellows_fraction=engine.circuit.bellows_fraction,
        reservoir_volume=engine.circuit.reservoir_volume,
        volume_delivery=volume_delivery,
        effort_cap=engine.hemo.config.effort_pressure_max,
    )


def update_shivering(engine: SimulationEngine, dt: float) -> None:
    """Update shivering intensity based on temperature and anesthetic state."""
    state = engine.state
    remi_effect = hill_function(state.opioid_ce, engine.resp.c50_remi, engine.resp.gamma_remi)
    # Shivering muscle is as sensitive to block as the adductor pollicis.
    muscle_factor = state.twitch / 100.0
    state.shivering = float(
        engine.thermal.step_shivering(
            dt, state.temp_c, state.bis, remi_effect, muscle_factor, unconscious=state.loc,
        )
    )


def step_temperature(engine: SimulationEngine, dt: float) -> None:
    """Update core temperature from metabolic heat, environmental loss, and redistribution."""
    state = engine.state
    state.temp_c = float(engine.thermal.step_temperature(
        dt,
        state.temp_c,
        state.bair_hugger_target,
        state.surface_cooling,
        oxygen_supply_fraction=engine.resp.oxygen_supply_fraction,
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
            if infusion.remaining_ml > 1e-3:
                remaining.append(infusion)
        engine.pending_infusions[:] = remaining

    if engine.lipid_infusion_ml_min > 0.0:
        amount = engine.lipid_infusion_ml_min * dt / 60.0
        hemo.add_volume(amount, retention_fraction=0.25, label="lipid")
        engine.toxicity.add_lipid(amount)

    if engine.maintenance_fluid_rate_ml_min > 0:
        rate_sec = engine.maintenance_fluid_rate_ml_min / 60.0
        hemo.add_volume(rate_sec * dt, hematocrit=0.0, label="crystalloid")

    hemo.anaphylaxis_severity = engine.anaphylaxis.step(dt)
    hemo.sepsis_severity = engine.sepsis.step(dt)

    return effects


def _disturbance_completes_during_step(engine: SimulationEngine, dt: float) -> bool:
    if not engine.disturbance_active:
        return False
    t_rel = max(0.0, engine.state.time - engine.disturbance_start_time)
    return engine.disturbances.is_complete(t_rel + dt)


def step_tci(engine: SimulationEngine, dt: float) -> None:
    """Advance TCI controllers on their own sampling clock."""
    sim_time = engine.state.time
    for key, controller in engine.tci.items():
        sampling_time = controller.sampling_time
        acc = engine._tci_accumulators.get(key, 0.0)
        steps = int((acc + 1e-9) / sampling_time)
        if steps > 0:
            for i in range(steps):
                remaining = acc - (i + 1) * sampling_time
                sample_time = sim_time if abs(remaining) <= 1e-9 else sim_time - remaining
                engine.infusion_rates[key] = controller.step(sim_time=sample_time)
            acc = max(0.0, acc - steps * sampling_time)
        engine._tci_accumulators[key] = acc + dt


def _alveolar_exchange_terms(
    engine: SimulationEngine,
    target_lung_vol: float,
    patency: float,
    va: float,
    n2o_uptake: float,
    dt: float,
    apneic_limit: float | None,
    *,
    fallback_flow_cap: float | None = None,
) -> tuple[float, float, float, float, float]:
    """Return (gas_vol_l, exhaled_vc_btps_l, passive_o2_btps_l_min, companion_btps_l_min, passive_stpd_l_min)."""
    resp = engine.resp
    frc_change, remaining_limit = resp._frc_volume_change_l(target_lung_vol, patency, dt, apneic_limit)
    if remaining_limit is None:
        remaining_limit = fallback_flow_cap
    exhaled_vc, deficit = resp.reconcile_collapse_deficit(frc_change)
    apneic_inflow, refill_l_min, n2o_inflow = resp._apneic_inflow_components(
        patency, va, n2o_uptake, dt, remaining_limit, deficit_l=deficit
    )
    passive_btps = (apneic_inflow + refill_l_min) / resp._btps_to_stpd
    companion_btps = n2o_inflow / resp._btps_to_stpd if n2o_uptake >= 0.0 else n2o_uptake * patency
    deficit_after_refill = max(0.0, deficit - refill_l_min * dt / 60.0)
    gas_vol = resp._gas_volume_at(resp.frc + frc_change, deficit_after_refill)
    return gas_vol, exhaled_vc, passive_btps, companion_btps, apneic_inflow + refill_l_min + n2o_inflow


def step_machine(engine: SimulationEngine, dt: float) -> tuple[float, float]:
    """Update machine state and return inspired volatile fractions."""
    circuit = engine.circuit
    delivered_setting = engine.vaporizer.step(dt, circuit.fgf_total())
    circuit.vaporizer_setting = delivered_setting
    circuit.vaporizer_on = engine.vaporizer.state.is_on and engine.config.sevoflurane_enabled

    fi_sevo, fi_n2o = sync_inspired_gas(engine)
    tracheal_fio2 = advance_tracheal_fio2(engine, dt)
    n2o_uptake = engine.pk_n2o.alveolar_uptake_l_min(engine.hemo.state.co, engine.aeration.shunt_fraction)
    target_lung_vol = engine.aeration.frc
    circuit_share = mask_circuit_share(engine)
    apneic_limit: float | None = None
    circuit_fin2: float | None = None
    leak_l_min = 0.0
    if engine.state.airway_mode == AirwayType.MASK and circuit_connected(engine):
        if assisted_positive_pressure(engine):
            if engine.vent.has_measured_breath:
                supported_mv = engine.vent.monitors.mv_supported
            elif engine.vent.is_on and engine.vent.settings.mode in VOLUME_TARGETED:
                supported_mv = engine.vent.settings.rr * engine.vent.settings.tv / 1000.0
            elif engine.bag_mask_active and not engine.vent.is_on:
                supported_mv = engine.bag_mask_rr * engine.bag_mask_vt
            else:
                supported_mv = 0.0
            leak_l_min = max(0.0, supported_mv) * mask_leak(engine) * engine.resp._btps_to_stpd
        elif engine.mask_seal < 0.999:
            cpap_cmh2o = (
                engine.vent.settings.peep
                if engine.vent.is_on and engine.vent.settings.mode == "CPAP"
                else (engine._apl_pressure if not engine.vent.is_on and not engine.bag_mask_active else 0.0)
            )
            if cpap_cmh2o > 0.0:
                leak_l_min = 0.45 * cpap_cmh2o * (1.0 - engine.mask_seal) * engine.resp._btps_to_stpd
    rebreathed_share = 1.0
    if circuit_share <= 0.0:
        uptake_o2 = uptake_sevo = uptake_n2o = uptake_n2 = 0.0
    else:
        airway = engine.airway_status
        patency = clamp01(airway.patency)
        raw_va = max(0.0, engine.state.va) if patency > 0.0 else 0.0
        vq_factor = 1.0 - 0.6 * clamp01(airway.vq_mismatch)
        va = raw_va * vq_factor
        va_circuit = va * circuit_share
        rebreathed_share = (
            clamp01(1.0 - circuit.fgf_total() / max(0.1, va_circuit))
            if va_circuit > 0.0
            else 0.0
        )
        composition = circuit.composition
        circuit_fio2, circuit_fi_sevo, circuit_fin2o, circuit_fin2 = circuit_inspired_mixture(
            composition, tracheal_fio2, circuit_share
        )
        fi_sevo = circuit_share * circuit_fi_sevo
        fi_n2o = circuit_share * circuit_fin2o
        resp = engine.resp
        apneic_limit = circuit.deliverable_flow(dt) if circuit_share >= 0.999 else None
        gas_vol, exhaled_vc, passive_o2_btps, companion_btps, passive_flow = _alveolar_exchange_terms(
            engine, target_lung_vol, patency, va, n2o_uptake, dt, apneic_limit
        )
        co_curr = engine.hemo.state.co
        shunt = engine.aeration.shunt_fraction
        uptake_sevo = (
            engine.pk_sevo.exchange_btps_l_min(
                dt, fi_sevo, va, co_curr, lung_volume_l=gas_vol,
                exhaled_volume_change_l=exhaled_vc,
                shunt_fraction=shunt, companion_uptake_l_min=companion_btps,
                passive_inflow_l_min=passive_o2_btps,
                circuit_share=circuit_share, circuit_fi_agent=circuit_fi_sevo,
            )
            * resp._btps_to_stpd
        )
        uptake_n2o = (
            engine.pk_n2o.exchange_btps_l_min(
                dt, fi_n2o, va, co_curr, lung_volume_l=gas_vol,
                exhaled_volume_change_l=exhaled_vc,
                shunt_fraction=shunt, companion_uptake_l_min=companion_btps,
                passive_inflow_l_min=passive_o2_btps,
                circuit_share=circuit_share, circuit_fi_agent=circuit_fin2o,
            )
            * resp._btps_to_stpd
        )
        if circuit_share >= 0.999:
            uptake_n2 = resp.nitrogen_exchange_l_min(
                circuit_fin2, raw_va, airway.patency, airway.vq_mismatch, dt,
                apneic_flow_limit_l_min=apneic_limit, n2o_uptake_l_min=n2o_uptake,
                lung_volume_l=target_lung_vol, cardiac_output=co_curr, shunt_fraction=shunt,
            )
        else:
            mixing = va * dt / (120.0 * gas_vol) if gas_vol > 0.0 else 0.0
            alveolar_n2 = resp.alveolar_fn2
            frc_stpd_l_min = exhaled_vc * resp._btps_to_stpd * 60.0 / dt if dt > 0.0 else 0.0
            frc_n2 = circuit_fin2 if frc_stpd_l_min >= 0.0 else alveolar_n2
            washout_stpd = max(0.0, -n2o_uptake * resp._btps_to_stpd) * patency
            tidal_scale = va_circuit * resp._btps_to_stpd / (1.0 + mixing)
            uptake_n2 = tidal_scale * (circuit_fin2 - alveolar_n2) + circuit_share * (
                frc_stpd_l_min * frc_n2 + passive_flow * circuit_fin2 - washout_stpd * alveolar_n2
            )
        uptake_o2 = circuit_share * resp.oxygen_exchange_l_min(
            circuit_fio2, raw_va,
            airway.patency, airway.vq_mismatch, dt,
            n2o_uptake_l_min=n2o_uptake,
            apneic_flow_limit_l_min=apneic_limit,
            lung_volume_l=target_lung_vol,
        )
    circuit.step(
        dt,
        uptake_o2,
        uptake_sevo,
        uptake_n2o,
        uptake_n2,
        leak_l_min=leak_l_min,
        rebreathed_share=rebreathed_share,
    )
    engine._step_gas = StepGasCache(
        tracheal_fio2=tracheal_fio2,
        n2o_uptake_l_min=n2o_uptake,
        lung_volume_l=target_lung_vol,
        circuit_fin2=circuit_fin2,
        apneic_limit_l_min=None if apneic_limit is None else apneic_limit * circuit.last_uptake_scale,
    )
    return fi_sevo, fi_n2o


def step_pk(engine: SimulationEngine, dt: float, fi_sevo: float, fi_n2o: float, co_curr: float) -> None:
    """Update pharmacokinetic models and synchronize their public state."""
    state = engine.state
    aeration = engine.aeration
    airway = engine.airway_status
    patency = clamp01(airway.patency)
    va = state.va * (1.0 - 0.6 * clamp01(airway.vq_mismatch)) if patency > 0.0 else 0.0
    cache = engine._step_gas
    n2o_uptake = (
        cache.n2o_uptake_l_min
        if cache is not None
        else engine.pk_n2o.alveolar_uptake_l_min(co_curr, aeration.shunt_fraction)
    )
    target_lung_vol = cache.lung_volume_l if cache is not None else aeration.frc
    sealed_circuit = circuit_connected(engine) and mask_circuit_share(engine) >= 0.999
    fallback_flow_cap = max(0.0, engine.circuit.deliverable_flow_l_min) if sealed_circuit else None
    gas_vol, exhaled_vc, passive_btps, companion_btps, _ = _alveolar_exchange_terms(
        engine,
        target_lung_vol,
        patency,
        va,
        n2o_uptake,
        dt,
        cache.apneic_limit_l_min if cache is not None else None,
        fallback_flow_cap=fallback_flow_cap,
    )
    uptake_scale = engine.circuit.last_uptake_scale if sealed_circuit else 1.0
    va_sevo = va * uptake_scale if fi_sevo > engine.pk_sevo.state.p_alv else va
    va_n2o = va * uptake_scale if fi_n2o > engine.pk_n2o.state.p_alv else va
    engine.pk_sevo.step(dt, fi_sevo, va_sevo, co_curr, temp_c=state.temp_c,
                        lung_volume_l=gas_vol, shunt_fraction=aeration.shunt_fraction,
                        companion_uptake_l_min=companion_btps, passive_inflow_l_min=passive_btps,
                        exhaled_volume_change_l=exhaled_vc)
    engine.pk_n2o.step(dt, fi_n2o, va_n2o, co_curr, temp_c=state.temp_c,
                       lung_volume_l=gas_vol, shunt_fraction=aeration.shunt_fraction,
                       companion_uptake_l_min=companion_btps, passive_inflow_l_min=passive_btps,
                       exhaled_volume_change_l=exhaled_vc)
    sync_inhaled_agents(engine)

    rates = engine.infusion_rates
    for models in (engine.pk, engine.pump_pk):
        for key, model in models.items():
            model.step(dt, rates.get(key, 0.0))
    # Plasma binding and sugammadex-dependent effect-site efflux reduce free
    # rocuronium, which drives TOF and every muscle effect.
    engine.tof_pd.step_recovery(
        dt, engine.pk["roc"], mac_sevo=engine.pk_sevo.state.mac, mac_n2o=engine.pk_n2o.state.mac
    )
    sync_pk_state(engine)


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
    base_co = engine.hemo.base_co_l_min / engine.hemo.failure.output_factor
    if base_bv <= 0.0 or base_co <= 0.0:
        return
    # Below 30% of resting output, as in arrest and CPR, flow limits drug transport.
    circulation = clamp01(co_curr / (CIRCULATION_LIMIT_FRACTION * base_co))
    if circulation != propofol.circulation:
        for model in (*engine.pk.values(), engine.tof_pd.sugammadex, engine.tof_pd.complex):
            model.circulation = circulation
    v_ratio = clamp(engine.hemo.blood_volume / base_bv, 0.1, 2.0)
    co_ratio = clamp(co_curr / base_co, 0.1, 2.0)
    scale = (v_ratio, co_ratio, engine.hemo.renal_perfusion)

    last_scale = engine._pk_hemo_scale_cache
    # Changes below 0.1% leave concentrations unchanged at display precision.
    if last_scale is not None and all(
        math.isclose(value, last, rel_tol=1e-3, abs_tol=1e-3) for value, last in zip(scale, last_scale, strict=True)
    ):
        return

    for model in engine.pk.values():
        model.update_hemodynamics(*scale, blood_volume_l=base_bv / 1000.0)
    engine.tof_pd.update_hemodynamics(*scale, blood_volume_l=base_bv / 1000.0)
    engine._pk_hemo_scale_cache = scale


def update_airway_complications(engine: SimulationEngine, dt: float) -> None:
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
            splint_pressure = vent.monitors.paw_mean_recent
        else:
            splint_pressure = engine._apl_pressure
        splint_pressure *= engine.mask_seal

    laryng_target = 0.0
    if state.airway_mode != AirwayType.ETT and stim_active:
        light_factor = clamp01(1.0 - tol)
        laryng_target = clamp01(light_factor * muscle_factor)

    stage_ii_sustained = (
        state.loc >= 0.20
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
        tone = max(0.1, engine.tof_pd.pharyngeal_tone_fraction(state.mac_sevo, state.mac_n2o) * depolarizing_strength)
        collapse = clamp01(airway_tuning.unsupported_collapse_max * unconscious / tone) * (1.0 - relief)
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
    r_bronch = airway_tuning.bronch_resistance_gain * bronch
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


def step_physiology(engine: SimulationEngine, dt: float, disturbances: DisturbanceEffects) -> PhysiologyStepState:
    """Advance physiology models and return a projected runtime snapshot."""
    state = engine.state
    connected = circuit_connected(engine)
    vent_active = connected and engine.vent.is_on
    bag_mask_active = engine.bag_mask_active and connected and not vent_active
    step_mechanics(engine, dt, connected, vent_active, bag_mask_active)
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

    applied_peep = (
        engine.resp_mech.peep
        if vent_active
        else (0.0 if bag_mask_active or not connected else engine._apl_pressure * mask_seal_factor(engine))
    )
    total_peep_effect = applied_peep + spirometry.auto_peep
    # Sevoflurane cardiovascular effects follow end-tidal MAC through the model's own ke0.
    mac_sevo = engine.pk_sevo.state.p_alv * 100.0 / engine.pk_sevo.mac_age
    mac_n2o = engine.pk_n2o.state.p_alv * 100.0 / engine.pk_n2o.mac_age
    resp_state = step_respiration(engine, dt, state.co)
    step_lactate(engine, dt)

    hemo = engine.hemo
    hemo_pre = hemo.state
    # Without flow, arterial pressure equals venous pressure, so coronary flow stops.
    step_coronary_from_state(engine, dt, hemo_pre, resp_state.sao2)

    # Plasma remifentanil and propofol reach hemodynamic effect sites inside the model;
    # slower opioids enter at their own effect sites.
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
        ce_midazolam=state.midazolam_ce,
        ce_etomidate=state.etomidate_ce,
        mac_n2o=mac_n2o,
    )
    hemo.fluids.step(
        dt, ph=resp_state.ph, paco2=resp_state.pa_co2, resting_paco2=engine.resp.paco2_setpoint,
        renal_function=engine.patient.renal_function, renal_perfusion=hemo.renal_perfusion,
        hepatic_function=engine.patient.hepatic_function, oxygen_supply=engine.resp.oxygen_supply_fraction,
        epinephrine=state.epi_ce,
        succinylcholine_ce=state.succinylcholine_ce,
        mh_severity=engine.malignant_hyperthermia.severity,
    )
    hypoxia, energy_failure, silent = myocardial_substrate(engine)
    intrinsic_pp = _intrinsic_pulse_pressure(engine, hemo_state)
    perfusing = hemo_state.intrinsic_co > ARREST_CO_L_MIN and not (
        hemo.rhythm_type == RhythmType.VTACH
        and intrinsic_pp < ARREST_PULSE_PRESSURE_MMHG
        and hemo_state.intrinsic_map < PULSELESS_VT_MAP_MMHG
    )
    rhythm = engine.arrhythmias.step(
        dt, hemo.rhythm_type, ischemia=hemo.coronary.ischemia, hypoxia=hypoxia,
        energy_failure=energy_failure, electrically_silent=silent,
        perfusing=perfusing,
        lidocaine=engine.pk["lidocaine"].state.ce_cv, amiodarone=state.amiodarone_ce,
        hyperkalemia=hemo.hyperkalemia_arrhythmia,
        hypothermia=clamp01((29.5 - state.temp_c) / 3.5),
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
    return hemo_state.sv / engine.arterial_waveform.compliance_at(max(1.0, hemo_state.intrinsic_map))


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
    resp.remove_buffer_base(base_removed)


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
    intrinsic_map = hemo_state.intrinsic_map if state.chest_compressions else state.map
    no_pulse = pulse_pressure < ARREST_PULSE_PRESSURE_MMHG
    pulseless = (
        rhythm in (RhythmType.VFIB, RhythmType.ASYSTOLE)
        or state.hr < ARREST_HR_BPM
        or (state.intrinsic_co < ARREST_CO_L_MIN and no_pulse)
        or (rhythm == RhythmType.VTACH and no_pulse and intrinsic_map < PULSELESS_VT_MAP_MMHG)
    )
    engine._pulseless_s = engine._pulseless_s + dt if pulseless else 0.0
    if not pulseless:
        state.cardiac_arrest = False
        state.arrest_reason = ""
        return
    if engine._pulseless_s < ARREST_CONFIRM_S:
        return

    if rhythm in (RhythmType.VFIB, RhythmType.ASYSTOLE):
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
    if changed:
        logger.warning("Cardiac arrest: %s (MAP=%.1f mmHg, HR=%.1f bpm)", reason, state.map, state.hr)
