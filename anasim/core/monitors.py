from __future__ import annotations

import math
from typing import TYPE_CHECKING

from anasim.monitors.capno import Capnograph
from anasim.monitors.cardiac_cycle import CardiacCycleSample
from anasim.monitors.nibp import NIBPReading
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.resp_mech import VentMode

from .projection import set_state_float_fields
from .state import AirwayType
from .utils import clamp

if TYPE_CHECKING:
    from .engine import SimulationEngine

CARDIAC_MONITOR_MAX_STEP_S = 0.01


def phase_from_rr(engine: "SimulationEngine", rr: float) -> str:
    """Calculate respiratory phase from respiratory rate."""
    if rr <= 0:
        return "EXP"
    cycle_time = 60.0 / rr
    insp_time = cycle_time / 3.0
    t_cycle = engine.state.time % cycle_time
    return "INSP" if t_cycle < insp_time else "EXP"


def seed_nibp_reading(engine: "SimulationEngine") -> None:
    """Seed NIBP with an initial reading and start cycling immediately."""
    state = engine.state
    map_val = state.map
    sbp_val = state.sbp
    dbp_val = state.dbp
    ts = state.time
    engine.nibp.latest_reading = NIBPReading(sbp_val, dbp_val, map_val, ts)
    set_state_float_fields(
        state,
        nibp_sys=sbp_val,
        nibp_dia=dbp_val,
        nibp_map=map_val,
        nibp_timestamp=ts,
    )
    engine._next_nibp_time = state.time + engine.nibp.interval
    engine.nibp.trigger()


def update_nibp(engine: "SimulationEngine", dt: float, hemo_state) -> None:
    """Update NIBP state and trigger cycles when appropriate."""
    state = engine.state
    if state.time >= engine._next_nibp_time and not engine.nibp.is_cycling:
        engine.nibp.trigger()
        engine._next_nibp_time = state.time + engine.nibp.interval

    prev_ts = state.nibp_timestamp
    cuff_p = engine.nibp.step(
        dt,
        state.time,
        state.map,
        true_sys=state.sbp,
        true_dia=state.dbp,
        rhythm_type=hemo_state.rhythm_type,
    )

    state.nibp_is_cycling = engine.nibp.is_cycling
    set_state_float_fields(state, nibp_cuff_pressure=cuff_p)

    latest = engine.nibp.latest_reading
    if latest.timestamp is not None and latest.timestamp != prev_ts:
        set_state_float_fields(
            state,
            nibp_sys=latest.systolic,
            nibp_dia=latest.diastolic,
            nibp_map=latest.map,
            nibp_timestamp=latest.timestamp,
        )


def compute_capno_value(engine: "SimulationEngine", dt: float, phase: str, resp_state) -> float:
    """Compute capnography waveform value for the current step."""
    state = engine.state
    if state.airway_mode == AirwayType.NONE:
        return 0.0
    if engine._airway_patency < 0.05 or state.rr == 0:
        engine.capno.state.co2 = 0.0
        return 0.0

    resp_mech = engine.resp_mech
    bag_mask_active = engine.bag_mask_active and not engine._vent_active
    vent_rr = resp_mech.set_rr if engine._vent_active else (engine.bag_mask_rr if bag_mask_active else 0.0)
    insp_fraction = resp_mech.insp_time_fraction
    if bag_mask_active and not engine._vent_active:
        insp_fraction = 1.0 / 3.0

    capno_context = Capnograph.build_context(
        resp_state,
        vent_rr=vent_rr,
        insp_fraction=insp_fraction,
        vent_active=(engine._vent_active or bag_mask_active),
    )
    capno_phase = phase
    capno_exp_duration = capno_context.exp_duration
    capno_is_spontaneous = capno_context.is_spontaneous
    capno_curare = capno_context.curare_active

    if engine._vent_active or bag_mask_active:
        support_mode = engine._vent_active and resp_mech.mode in (VentMode.PSV, VentMode.CPAP)
        if support_mode:
            capno_is_spontaneous = True
            capno_curare = False
            capno_phase = phase_from_rr(engine, resp_state.rr)
            if resp_state.rr > 0:
                capno_exp_duration = (60.0 / resp_state.rr) * 0.65
        else:
            if capno_context.spontaneous_weight >= 0.6:
                capno_phase = phase_from_rr(engine, capno_context.effective_rr)
                capno_exp_duration = capno_context.exp_duration
            else:
                driven_rr = vent_rr if vent_rr > 0.1 else capno_context.effective_rr
                cycle_time = 60.0 / max(driven_rr, 0.1)
                exp_fraction = max(0.1, 1.0 - insp_fraction)
                capno_exp_duration = cycle_time * exp_fraction
    elif capno_is_spontaneous:
        capno_phase = phase_from_rr(engine, resp_state.rr)

    capno_p_alv = resp_state.etco2 * engine._airway_patency
    return engine.capno.step(
        dt,
        capno_phase,
        capno_p_alv,
        is_spontaneous=capno_is_spontaneous,
        curare_cleft=capno_curare,
        exp_duration=capno_exp_duration,
        effort_scale=capno_context.effort_scale,
        airway_obstruction=engine._capno_obstruction,
    )


def update_capno_numeric(engine: "SimulationEngine", dt: float, phase: str, capno_value: float) -> tuple[float, bool]:
    """Hold breath-derived EtCO2 and invalidate it when exhaled gas is absent."""
    state = engine.state
    sampling_possible = (
        state.airway_mode != AirwayType.NONE
        and engine._airway_patency >= 0.05
        and state.rr > 0.0
    )
    engine._capno_numeric_age_s += dt

    if sampling_possible and phase == "EXP":
        engine._capno_numeric_peak = max(engine._capno_numeric_peak, capno_value)

    completed_breath = engine._capno_last_phase == "EXP" and phase == "INSP"
    if sampling_possible and completed_breath and engine._capno_numeric_peak > 1.0:
        display_value = engine._capno_numeric_peak
        engine._capno_numeric_age_s = 0.0
        engine._capno_numeric_peak = 0.0
        engine._capno_has_sample = True
    else:
        display_value = state.display_etco2

    engine._capno_last_phase = phase
    valid = (
        sampling_possible
        and engine._capno_has_sample
        and engine._capno_numeric_age_s <= engine._capno_numeric_timeout_s
    )
    return (float(display_value) if valid else 0.0), valid


def step_cardiac_monitors(
    engine: "SimulationEngine",
    dt: float,
    hemo_state,
    sao2: float,
) -> CardiacCycleSample:
    """Advance beat-synchronous monitors at waveform resolution."""
    if dt <= 0.0:
        raise ValueError("cardiac monitor dt must be greater than zero")
    substeps = math.ceil(dt / CARDIAC_MONITOR_MAX_STEP_S)
    substep_dt = dt / substeps
    rhythm = hemo_state.rhythm_type
    co_ratio = hemo_state.co / engine.hemo.base_co_l_min
    perfusion = clamp(co_ratio, 0.05, 1.0)

    for _ in range(substeps):
        cardiac_sample = engine.cardiac_cycle.step(
            substep_dt,
            hemo_state.hr,
            rhythm,
        )
        arterial_sample = engine.arterial_waveform.step(
            cardiac_sample,
            hemo_state.map,
            hemo_state.sv,
        )
        art_reading = engine.art_line.step(
            substep_dt,
            cardiac_sample,
            arterial_sample,
        )
        ecg_voltage = engine.ecg.step(substep_dt, cardiac_sample)
        pleth, spo2_val = engine.spo2_mon.step(
            substep_dt,
            cardiac_sample,
            saturation=sao2,
            perfusion=perfusion,
        )

    state = engine.state
    state.spo2_signal_valid = engine.spo2_mon.signal_valid
    set_state_float_fields(
        state,
        ecg_voltage=ecg_voltage,
        pleth_voltage=pleth,
        sbp=arterial_sample.systolic,
        dbp=arterial_sample.diastolic,
        art_pressure=art_reading.pressure,
        art_sbp=art_reading.systolic,
        art_dbp=art_reading.diastolic,
        art_map=art_reading.mean,
        spo2=spo2_val,
    )
    return cardiac_sample


def step_monitors(
    engine: "SimulationEngine",
    dt: float,
    phase: str,
    hemo_state,
    resp_state,
    disturbances: DisturbanceEffects,
) -> None:
    """Update monitor models and learner-facing display values."""
    state = engine.state

    if engine.patient.weight != engine._remi_rate_weight:
        engine._remi_rate_weight = engine.patient.weight
        engine._remi_rate_scale = 60.0 / engine._remi_rate_weight
    remi_rate_ug_kg_min = engine.remi_rate_ug_sec * engine._remi_rate_scale

    mac_sevo = engine.pk_sevo.state.mac
    bis_val = engine.bis.step(
        dt,
        state.propofol_ce,
        state.remi_ce,
        mac_sevo=mac_sevo,
        remi_rate_ug_kg_min=remi_rate_ug_kg_min,
    )
    capno_val = compute_capno_value(engine, dt, phase, resp_state)

    tof_val = engine.tof_pd.step_recovery(
        dt,
        state.roc_cp,
        mac_sevo=mac_sevo,
        mac_n2o=state.mac_n2o,
    )
    loc_val = engine.loc_pd.compute_probability(
        state.propofol_ce,
        state.remi_ce,
        mac_sevo=mac_sevo,
        mac_n2o=state.mac_n2o,
    )
    if engine._tol_current is not None:
        tol_val = engine._tol_current
    else:
        tol_val = engine.tol_pd.compute_probability(state.propofol_ce, state.remi_ce)

    sao2 = state.sao2
    cardiac_sample = step_cardiac_monitors(
        engine,
        dt,
        hemo_state,
        sao2,
    )

    update_nibp(engine, dt, hemo_state)

    bis_display_source = clamp(bis_val + disturbances.bis, 0.0, 100.0)
    set_state_float_fields(state, bis=bis_display_source)
    display_etco2, etco2_signal_valid = update_capno_numeric(
        engine,
        dt,
        engine.capno.last_phase,
        capno_val,
    )
    state.etco2_signal_valid = etco2_signal_valid

    raw_bis = state.bis + float(engine.rng.normal(0.0, engine._bis_noise_std))
    alpha_bis = 1.0 - math.exp(-dt / engine._monitor_tau_bis_s)
    engine.smooth_bis = float((1 - alpha_bis) * engine.smooth_bis + alpha_bis * raw_bis)

    display_hr = max(0.0, cardiac_sample.measured_hr)
    display_bis = clamp(engine.smooth_bis, 0.0, 100.0)
    set_state_float_fields(
        state,
        display_hr=display_hr,
        display_bis=display_bis,
        capno_co2=capno_val,
        display_etco2=display_etco2,
        tof=tof_val,
        loc=loc_val,
        tol=tol_val,
        display_spo2=state.spo2,
    )

    monitor_vals = engine._monitor_values
    monitor_vals["BIS"] = state.display_bis
    monitor_vals["MAP"] = state.monitored_blood_pressure(
        engine.config.arterial_line_enabled
    )[2]
    monitor_vals["HR"] = state.display_hr
    monitor_vals["EtCO2"] = state.display_etco2
    monitor_vals["SpO2"] = state.display_spo2
    state.alarms = engine.alarms.update(monitor_vals, dt=dt)
