from __future__ import annotations

import math
from typing import TYPE_CHECKING

from anasim.monitors.cardiac_cycle import CardiacCycleSample
from anasim.monitors.nibp import NIBPReading
from anasim.physiology.disturbances import DisturbanceEffects

from .projection import mask_leak, set_state_float_fields
from .state import AirwayType
from .utils import clamp

if TYPE_CHECKING:
    from .engine import SimulationEngine

CARDIAC_MONITOR_MAX_STEP_S = 0.01
VOLUME_TARGETED = ("VCV", "PCV-VG", "SIMV-VC", "SIMV-VG")


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
    state.nibp_is_cycling = True


def update_nibp(engine: "SimulationEngine", dt: float, hemo_state) -> None:
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


def _capno_sampling_possible(engine: "SimulationEngine") -> bool:
    state = engine.state
    return (
        state.airway_mode != AirwayType.NONE
        and engine._airway_patency >= 0.05
        and state.rr > 0.0
        and state.va > 0.0
    )


def compute_capno_value(engine: "SimulationEngine", dt: float, resp_state) -> float:
    """Advance the capnograph with the gas that crossed the Y-piece this step."""
    volume = engine.vent.volume
    change, engine._capno_volume = volume - engine._capno_volume, volume
    if not _capno_sampling_possible(engine):
        engine.capno.reset()
        return 0.0
    return engine.capno.step(
        dt, change, resp_state.etco2 * engine._airway_patency, obstruction=engine._capno_obstruction
    )


def update_capno_numeric(engine: "SimulationEngine", dt: float, phase: str, capno_value: float) -> tuple[float, bool]:
    """Hold breath-derived EtCO2 and invalidate it when exhaled gas is absent."""
    state = engine.state
    sampling_possible = _capno_sampling_possible(engine)
    if not sampling_possible:
        engine._capno_numeric_peak = 0.0
        engine._capno_numeric_age_s = 0.0
        engine._capno_has_sample = False
        engine._capno_last_phase = phase
        return 0.0, False
    engine._capno_numeric_age_s += dt

    if phase == "EXP":
        engine._capno_numeric_peak = max(engine._capno_numeric_peak, capno_value)

    completed_breath = engine._capno_last_phase == "EXP" and phase == "INSP"
    if completed_breath and engine._capno_numeric_peak > 1.0:
        display_value = engine._capno_numeric_peak
        engine._capno_numeric_age_s = 0.0
        engine._capno_numeric_peak = 0.0
        engine._capno_has_sample = True
    else:
        display_value = state.display_etco2

    engine._capno_last_phase = phase
    valid = (
        engine._capno_has_sample
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
    hemo_state,
    resp_state,
    disturbances: DisturbanceEffects,
) -> None:
    """Update monitor models and learner-facing display values."""
    state = engine.state
    mac_sevo = engine.pk_sevo.state.mac
    bis_val = engine.bis.step(dt, state.hypnotic_ce, state.opioid_ce, mac_sevo=mac_sevo)
    capno_val = compute_capno_value(engine, dt, resp_state)

    loc_val = engine.loc_pd.compute_probability(
        state.hypnotic_ce,
        state.opioid_ce,
        mac_sevo=mac_sevo,
        mac_n2o=state.mac_n2o,
        ce_ketamine=state.ketamine_ce,
    )
    cardiac_sample = step_cardiac_monitors(engine, dt, hemo_state, state.sao2)

    update_nibp(engine, dt, hemo_state)

    paw, flow, volume = engine.airway_sensor.step(dt, state.paw, state.flow, state.volume)
    bis_display_source = clamp(bis_val + disturbances.bis, 0.0, 100.0)
    set_state_float_fields(state, bis=bis_display_source, paw=paw, flow=flow, volume=volume)
    display_etco2, etco2_signal_valid = update_capno_numeric(
        engine,
        dt,
        "EXP" if engine.capno.exhaling else "INSP",
        capno_val,
    )
    state.etco2_signal_valid = etco2_signal_valid

    raw_bis = state.bis + float(engine.rng.normal(0.0, engine._bis_noise_std))
    alpha_bis = 1.0 - math.exp(-dt / engine._monitor_tau_bis_s)
    engine.smooth_bis = float((1 - alpha_bis) * engine.smooth_bis + alpha_bis * raw_bis)

    display_hr = max(0.0, cardiac_sample.display_hr)
    display_bis = clamp(engine.smooth_bis, 0.0, 100.0)
    set_state_float_fields(
        state,
        display_hr=display_hr,
        display_bis=display_bis,
        capno_co2=capno_val,
        display_etco2=display_etco2,
        loc=loc_val,
        tol=engine._tol_current,
        display_spo2=state.spo2,
    )
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
