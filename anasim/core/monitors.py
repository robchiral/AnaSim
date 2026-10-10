from __future__ import annotations

import math
from typing import TYPE_CHECKING

from anasim.monitors.cardiac_cycle import CardiacCycleSample
from anasim.monitors.nibp import NIBPReading
from anasim.physiology.disturbances import DisturbanceEffects

from .projection import loss_of_response, mask_leak, project_arterial
from .state import AirwayType
from .utils import clamp

if TYPE_CHECKING:
    from .engine import SimulationEngine

CARDIAC_MONITOR_MAX_STEP_S = 0.01
BIS_NOISE_SD = 0.2
BIS_DISPLAY_TAU_S = 2.0
VOLUME_TARGETED = ("VCV", "PCV-VG", "SIMV-VC", "SIMV-VG")


def seed_nibp_reading(engine: SimulationEngine) -> None:
    """Seed NIBP with an initial reading and start cycling immediately."""
    state = engine.state
    map_val = state.map
    sbp_val = state.sbp
    dbp_val = state.dbp
    ts = state.time
    engine.nibp.latest_reading = NIBPReading(sbp_val, dbp_val, map_val, ts)
    state.nibp_sys = float(sbp_val)
    state.nibp_dia = float(dbp_val)
    state.nibp_map = float(map_val)
    state.nibp_timestamp = float(ts)
    engine._next_nibp_time = state.time + engine.nibp.interval
    engine.nibp.trigger()
    state.nibp_is_cycling = True


def update_nibp(engine: SimulationEngine, dt: float, hemo_state) -> None:
    state = engine.state
    if state.time >= engine._next_nibp_time and not engine.nibp.is_cycling:
        engine.nibp.trigger()
        engine._next_nibp_time = state.time + engine.nibp.interval

    prev_ts = state.nibp_timestamp
    cuff_p = engine.nibp.step(
        dt,
        state.time + dt,
        state.map,
        true_sys=state.sbp,
        true_dia=state.dbp,
        rhythm_type=hemo_state.rhythm_type,
    )

    state.nibp_is_cycling = engine.nibp.is_cycling
    state.nibp_measurement_failed = engine.nibp.measurement_failed
    state.nibp_cuff_pressure = float(cuff_p)

    latest = engine.nibp.latest_reading
    if latest.timestamp is not None and latest.timestamp != prev_ts:
        state.nibp_sys = float(latest.systolic)
        state.nibp_dia = float(latest.diastolic)
        state.nibp_map = float(latest.map)
        state.nibp_timestamp = float(latest.timestamp)


def _capno_sampling_possible(engine: SimulationEngine) -> bool:
    state = engine.state
    return (
        state.airway_mode != AirwayType.NONE
        and engine.airway_status.patency >= 0.05
        and state.rr > 0.0
        and state.va > 0.0
    )


def compute_capno_value(engine: SimulationEngine, resp_state, dt: float) -> float:
    """Advance the capnograph with the gas that crossed the Y-piece this step."""
    state = engine.state
    readout = engine.etco2_readout
    if not _capno_sampling_possible(engine):
        engine.capno.reset()
        readout.interrupt(dt, disconnected=state.airway_mode == AirwayType.NONE)
        co2 = 0.0
    else:
        capno = engine.capno
        airway = engine.airway_status
        end_tidal = resp_state.etco2 * airway.patency
        for duration, change, _, _, _ in engine.vent.samples:
            value = capno.step(duration, change, end_tidal, obstruction=airway.capno_obstruction)
            readout.update(duration, capno.exhaling, value)
        co2 = capno.co2
    state.display_etco2 = float(readout.value)
    state.etco2_signal_valid = readout.valid
    return co2


def step_cardiac_monitors(
    engine: SimulationEngine,
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
    perfusion = clamp(co_ratio, 0.0, 1.0)

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
    state.ecg_voltage = float(ecg_voltage)
    state.pleth_voltage = float(pleth)
    project_arterial(state, arterial_sample, art_reading)
    state.spo2 = float(spo2_val)
    return cardiac_sample


def step_monitors(
    engine: SimulationEngine,
    dt: float,
    hemo_state,
    resp_state,
    disturbances: DisturbanceEffects,
) -> None:
    """Update monitor models and learner-facing display values."""
    state = engine.state
    mac_sevo = engine.pk_sevo.state.mac
    bis_val = engine.bis.step(dt, state.hypnotic_ce, mac_sevo=mac_sevo)
    capno_val = compute_capno_value(engine, resp_state, dt)

    loc_val = loss_of_response(engine)
    cardiac_sample = step_cardiac_monitors(engine, dt, hemo_state, state.sao2)

    update_nibp(engine, dt, hemo_state)

    for duration, change, sample_paw, _, sample_volume in engine.vent.samples:
        paw, flow, volume = engine.airway_sensor.step(duration, sample_paw, sample_volume,
                                                   change if state.airway_mode != AirwayType.NONE else 0.0)
        state.paw = float(paw)
        state.flow = float(flow)
        state.volume = float(volume)
    bis_display_source = clamp(bis_val + disturbances.bis, 0.0, 100.0)
    state.bis = float(bis_display_source)

    raw_bis = state.bis + float(engine.rng.normal(0.0, BIS_NOISE_SD))
    alpha_bis = 1.0 - math.exp(-dt / BIS_DISPLAY_TAU_S)
    engine.smooth_bis = float((1 - alpha_bis) * engine.smooth_bis + alpha_bis * raw_bis)

    display_hr = max(0.0, cardiac_sample.display_hr)
    display_bis = clamp(engine.smooth_bis, 0.0, 100.0)
    state.display_hr = float(display_hr)
    state.display_bis = float(display_bis)
    state.capno_co2 = float(capno_val)
    state.loc = float(loc_val)
    state.tol = float(engine._tol_current)
    state.display_spo2 = float(state.spo2)
    connected = state.airway_mode != AirwayType.NONE
    vent = engine.vent
    # Volume-targeted breaths alarm when they deliver less than the set VT.
    vte_percent = None
    if connected and vent.is_on and vent.settings.mode in VOLUME_TARGETED and vent.settings.tv > 0.0:
        delivered = vent.monitors.tv_exp_mandatory * (1.0 - mask_leak(engine))
        vte_percent = None if math.isnan(delivered) else 100.0 * delivered / vent.settings.tv
    state.alarms = engine.alarms.update(
        {
            "BIS": state.display_bis,
            "MAP": state.monitored_blood_pressure(engine.config.arterial_line_enabled)[2],
            "HR": state.display_hr,
            "EtCO2": state.display_etco2,
            "SpO2": state.display_spo2 if state.spo2_signal_valid else None,
            "Ppeak": state.paw_peak,
            # The ventilator measures only gas returning through the circuit.
            "MV": (state.mv if connected else 0.0) if vent.is_on else None,
            "VTe": vte_percent,
            "FiO2": state.fio2 * 100.0,
        },
        dt=dt,
    )
