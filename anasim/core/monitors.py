from __future__ import annotations

import math
from typing import TYPE_CHECKING

from anasim.core.constants import ARREST_CO_L_MIN
from anasim.machine.ventilator import VOLUME_TARGETED
from anasim.monitors.arterial import MIN_PULSE_PRESSURE_MMHG, ArterialPressureSample
from anasim.monitors.cardiac_cycle import CardiacCycleSample
from anasim.monitors.nibp import NIBPReading
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.resuscitation import COMPRESSION_DUTY, compression_pressures

from .projection import (
    central_muscle_strength,
    circuit_connected,
    delivered_fraction,
    loss_of_response,
    mask_leak,
    mask_seal_factor,
    nmba_twitch_fraction,
    project_arterial,
    project_hemodynamics,
    stimulus_tolerance,
)
from .state import AirwayType, WaveformSample
from .utils import clamp, clamp01

if TYPE_CHECKING:
    from .engine import SimulationEngine

WAVEFORM_MAX_INTERVAL_S = 0.01
BIS_NOISE_SD = 0.2
BIS_DISPLAY_TAU_S = 2.0


def display_int(value: float) -> float:
    """Round to the whole-number value shown on the patient monitor."""
    if not math.isfinite(value):
        return value
    return float(math.floor(round(value, 1) + 0.5))


def waveform_substeps(dt: float) -> int:
    """Waveform samples in a step of dt; beat-synchronous monitors advance once per sample."""
    return math.ceil(dt / WAVEFORM_MAX_INTERVAL_S)


def _hold(waves: list, value, elapsed: float, substeps: int, substep_dt: float) -> None:
    """Assign the latest value to each waveform sample time that `elapsed` has reached, except the last."""
    if substeps <= 1:
        return
    while len(waves) < substeps - 1 and elapsed >= (len(waves) + 1) * substep_dt - 1e-9:
        waves.append(value)


def seed_nibp_reading(engine: SimulationEngine) -> None:
    """Seed NIBP with an initial reading and start cycling immediately."""
    state = engine.state
    map_val = state.map
    sbp_val = state.sbp
    dbp_val = state.dbp
    ts = state.time
    engine.nibp.latest_reading = NIBPReading(sbp_val, dbp_val, map_val, ts)
    engine.nibp.measurement_failed = False
    engine.nibp.failed_timestamp = None
    state.nibp_sys = float(sbp_val)
    state.nibp_dia = float(dbp_val)
    state.nibp_map = float(map_val)
    state.nibp_timestamp = float(ts)
    state.nibp_measurement_failed = False
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
        compressions=hemo_state.compressions,
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
        and (state.airway_mode != AirwayType.MASK or engine.mask_seal >= 0.05)
        and engine.airway_status.patency >= 0.05
    )


def compute_capno_value(engine: SimulationEngine, resp_state, dt: float, substeps: int = 1) -> list[float]:
    """Advance the capnograph with the gas that crossed the Y-piece; return CO2 at each waveform sample."""
    state = engine.state
    readout = engine.etco2_readout
    et_gases = (state.et_o2, state.et_sevo, state.et_n2o, state.et_mac)
    if not _capno_sampling_possible(engine):
        engine.capno.reset()
        readout.interrupt(
            dt,
            disconnected=state.airway_mode == AirwayType.NONE,
            idle_airway=state.airway_mode == AirwayType.NONE and not (engine.vent.is_on or engine.bag_mask_active),
        )
        waves = [0.0] * substeps
    else:
        capno = engine.capno
        airway = engine.airway_status
        end_tidal = resp_state.etco2
        # Gas lost around a facemask never passes the sampling port, so a
        # large leak leaves dead-space gas at the port.
        port_share = delivered_fraction(engine)
        waves = []
        elapsed = 0.0
        substep_dt = dt / substeps
        samples = engine.vent.samples or [(dt, 0.0, 0.0, 0.0, 0.0)]
        for duration, change, _, _, _ in samples:
            value = capno.step(duration, change * port_share, end_tidal, obstruction=airway.capno_obstruction)
            readout.update(
                duration,
                capno.exhaling,
                value,
                inhaled_l=capno._inhaled,
                dead_space_l=capno.dead_space,
                breath_index=capno.breath_index,
                et_gases=et_gases,
            )
            elapsed += duration
            _hold(waves, value, elapsed, substeps, substep_dt)
        waves.extend([capno.co2] * (substeps - len(waves)))
    state.display_etco2 = float(readout.value)
    state.etco2_signal_valid = readout.valid
    state.et_o2, state.et_sevo, state.et_n2o, state.et_mac = (
        readout.et_gases if state.etco2_signal_valid else (0.0, 0.0, 0.0, 0.0)
    )
    return waves


def step_cardiac_monitors(
    engine: SimulationEngine,
    dt: float,
    hemo_state,
    sao2: float,
) -> tuple[CardiacCycleSample, list[tuple[float, float, float]]]:
    """Advance beat-synchronous monitors; return the last beat sample and (ECG, pleth, ART) per waveform sample."""
    if dt <= 0.0:
        raise ValueError("cardiac monitor dt must be greater than zero")
    substeps = waveform_substeps(dt)
    substep_dt = dt / substeps
    waves = []
    rhythm = hemo_state.rhythm_type
    co_ratio = hemo_state.co / engine.hemo.base_co_l_min
    perfusion = clamp(co_ratio, 0.0, 1.0)
    st_depression = engine.hemo.coronary.st_depression_mv
    conduction_failure = engine.hemo.coronary.energy_failure
    compressions = hemo_state.compressions
    if compressions:
        compression_sbp, compression_dbp = compression_pressures(
            hemo_state.cpr_perfusion_map, hemo_state.venous_pressure
        )
        compression_mean = (
            compression_dbp + (compression_sbp - compression_dbp) * 2.0 * COMPRESSION_DUTY / math.pi
        )
    else:
        engine.compression_clock.stop()

    cardiac_cycle = engine.cardiac_cycle
    art_waveform = engine.arterial_waveform
    art_line = engine.art_line
    ecg_mon = engine.ecg
    spo2_mon = engine.spo2_mon
    defib = engine.defibrillator
    sodium_block = engine.toxicity.conduction_block
    potassium = engine.state.potassium
    hyperk_block = engine.hemo.hyperkalemia_block
    start_time = engine.state.time

    for index in range(substeps):
        cardiac_sample = cardiac_cycle.step(
            substep_dt,
            hemo_state.hr,
            rhythm,
            conduction_failure,
        )
        if defib.waiting_for_r and engine.step_defibrillator(start_time + (index + 1) * substep_dt, cardiac_sample):
            # A shock can change the rhythm partway through a step; physiology follows on the next step.
            hemo_state = engine.hemo.state
            project_hemodynamics(engine, hemo_state)
            rhythm = hemo_state.rhythm_type
            cardiac_sample = cardiac_cycle.seed(hemo_state.hr, rhythm)
            if compressions:
                compression_sbp, compression_dbp = compression_pressures(
                    hemo_state.cpr_perfusion_map, hemo_state.venous_pressure
                )
                compression_mean = (
                    compression_dbp + (compression_sbp - compression_dbp) * 2.0 * COMPRESSION_DUTY / math.pi
                )
        # Compressions, when running, envelope any weaker intrinsic beat rather than adding onto MAP.
        if compressions:
            comp_sample = engine.compression_clock.step(substep_dt, rhythm)
            if hemo_state.intrinsic_co > 0.0:
                intrinsic_art = art_waveform.step(
                    cardiac_sample,
                    hemo_state.intrinsic_map,
                    hemo_state.sv,
                    dt=substep_dt,
                    static_pressure=hemo_state.venous_pressure,
                    pulse_pressure_factor=hemo_state.pulse_pressure_factor,
                )
                comp_u = comp_sample.phase / COMPRESSION_DUTY
                comp_shape = math.sin(math.pi * comp_u) if comp_u < 1.0 else 0.0
                comp_p = compression_dbp + (compression_sbp - compression_dbp) * comp_shape
                arterial_sample = ArterialPressureSample(
                    pressure=float(max(intrinsic_art.pressure, comp_p)),
                    systolic=float(max(intrinsic_art.systolic, compression_sbp)),
                    diastolic=float(max(intrinsic_art.diastolic, compression_dbp)),
                    mean=float(max(intrinsic_art.mean, compression_mean)),
                )
                pulse_sample = cardiac_sample if intrinsic_art.mean >= compression_mean else comp_sample
            else:
                pulse_sample = comp_sample
                arterial_sample = art_waveform.step_compressions(
                    pulse_sample, compression_sbp, compression_dbp,
                )
        else:
            pulse_sample = cardiac_sample
            arterial_sample = art_waveform.step(
                cardiac_sample,
                hemo_state.map,
                hemo_state.sv,
                dt=substep_dt,
                static_pressure=hemo_state.venous_pressure,
                pulse_pressure_factor=hemo_state.pulse_pressure_factor,
            )
        art_line.advance(
            substep_dt,
            pulse_sample,
            arterial_sample,
        )
        ecg_voltage = ecg_mon.step(substep_dt, cardiac_sample, st_depression,
                                   conduction_failure=conduction_failure,
                                   sodium_block=sodium_block,
                                   potassium=potassium,
                                   hyperkalemia_block=hyperk_block,
                                   compression_phase=comp_sample.phase if compressions else None)
        pulse_pressure = arterial_sample.systolic - arterial_sample.diastolic
        pp_gate = (
            clamp01((pulse_pressure - MIN_PULSE_PRESSURE_MMHG) / 4.0)
            if math.isfinite(pulse_pressure)
            else 0.0
        )
        spo2_perfusion = (
            0.0
            if not compressions and (hemo_state.intrinsic_co < ARREST_CO_L_MIN or pp_gate <= 0.0)
            else (perfusion if compressions else perfusion * pp_gate)
        )
        pleth, spo2_val = spo2_mon.step(
            substep_dt,
            pulse_sample,
            saturation=sao2,
            perfusion=spo2_perfusion,
        )
        waves.append((float(ecg_voltage), float(pleth), art_line.pressure))

    state = engine.state
    state.spo2_signal_valid = spo2_mon.signal_valid
    state.ecg_voltage, state.pleth_voltage, _ = waves[-1]
    project_arterial(state, arterial_sample, art_line.reading)
    state.spo2 = float(spo2_val) if state.spo2_signal_valid else 0.0
    return cardiac_sample, waves


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
    state.respiratory_fatigue = float(engine.resp_mech.effort.muscle.fatigue)
    bis_val = engine.bis.step(
        dt,
        state.hypnotic_ce,
        mac_sevo=mac_sevo,
        cortical_function=engine.cerebral.cortical_function,
        ce_ketamine=state.ketamine_ce,
    )
    substeps = waveform_substeps(dt)
    substep_dt = dt / substeps
    capno_waves = compute_capno_value(engine, resp_state, dt, substeps)

    loc_val = loss_of_response(engine)
    engine._tol_current = stimulus_tolerance(engine)
    cardiac_sample, cardiac_waves = step_cardiac_monitors(engine, dt, hemo_state, state.sao2)

    update_nibp(engine, dt, engine.hemo.state)
    engine.nmb.step(state.time + dt, state.twitch / 100.0, state.tof / 100.0)

    connected = state.airway_mode != AirwayType.NONE
    sealed = circuit_connected(engine)
    seal_scale = mask_seal_factor(engine)
    kept = delivered_fraction(engine) if sealed else 0.0
    airway_wave = engine.airway_sensor.reading
    airway_waves: list[tuple[float, float, float]] = []
    elapsed = 0.0
    for duration, change, sample_paw, _, sample_volume in engine.vent.samples:
        airway_wave = engine.airway_sensor.step(
            duration,
            sample_paw * seal_scale if sealed else 0.0,
            sample_volume * kept,
            change * kept,
        )
        elapsed += duration
        _hold(airway_waves, airway_wave, elapsed, substeps, substep_dt)
    airway_waves.extend([airway_wave] * (substeps - len(airway_waves)))
    state.paw, state.flow, state.volume = (float(value) for value in airway_wave)
    bis_display_source = clamp(bis_val + disturbances.bis * engine.bis.cortical_function, 0.0, 100.0)
    state.bis = float(bis_display_source)

    raw_bis = state.bis + float(engine.rng.normal(0.0, BIS_NOISE_SD))
    alpha_bis = 1.0 - math.exp(-dt / BIS_DISPLAY_TAU_S)
    engine.smooth_bis = float((1 - alpha_bis) * engine.smooth_bis + alpha_bis * raw_bis)

    display_hr = max(0.0, cardiac_sample.display_hr)
    display_bis = clamp(engine.smooth_bis, 0.0, 100.0)
    state.display_hr = float(display_hr)
    state.display_bis = float(display_bis)
    state.capno_co2 = float(capno_waves[-1])
    state.loc = float(loc_val)
    state.tol = float(engine._tol_current)
    state.display_spo2 = float(state.spo2)
    vent = engine.vent
    # Volume-targeted breaths alarm when they deliver less than the set VT.
    vte_percent = None
    if connected and vent.is_on and vent.settings.mode in VOLUME_TARGETED and vent.settings.tv > 0.0:
        if not sealed:
            vte_percent = 0.0
        else:
            delivered = vent.monitors.tv_exp_mandatory * (1.0 - mask_leak(engine))
            vte_percent = None if math.isnan(delivered) else 100.0 * delivered / vent.settings.tv
    if state.display_bis < 65.0:
        engine._bis_anesthetized = True
    elif not connected and state.display_bis >= 80.0:
        engine._bis_anesthetized = False
    paralyzed = (
        state.tof < 90.0
        or nmba_twitch_fraction(engine) < 0.90
        or central_muscle_strength(engine) < 0.85
    )
    maintaining = engine._bis_anesthetized and (
        engine.disturbance_active
        or engine.vaporizer.state.is_on
        or engine.circuit.fgf_n2o > 0.0
        or any(
            engine.infusion_rates.get(k, 0.0) > 0.0 or (k in engine.tci and engine.tci[k].target > 0.0)
            for k in ("propofol", "midazolam", "ketamine")
        )
    )
    bis_int = display_int(state.display_bis)
    bis_alarm = bis_int if (paralyzed or maintaining or bis_int <= engine.alarms.thresholds["BIS_max"]) else None
    map_alarm = (
        display_int(state.art_map)
        if engine.config.arterial_line_enabled
        else (0.0 if state.nibp_measurement_failed else display_int(state.nibp_map))
    )
    etco2_alarm = (
        display_int(state.display_etco2)
        if (connected or vent.is_on or engine.bag_mask_active) and not engine.etco2_readout.awaiting_first_breath
        else None
    )
    state.alarms = engine.alarms.update(
        {
            "BIS": bis_alarm,
            "MAP": map_alarm,
            "HR": display_int(state.display_hr),
            "EtCO2": etco2_alarm,
            "RR": display_int(state.rr) if not (vent.is_on or engine.bag_mask_active) else None,
            "SpO2": display_int(state.display_spo2) if state.spo2_signal_valid else 0.0,
            "Ppeak": display_int(state.paw_peak),
            # The ventilator measures only gas returning through the circuit.
            "MV": (round(state.mv, 1) if connected else 0.0) if vent.is_on else None,
            "VTe": vte_percent,
            "FiO2": display_int(state.fio2 * 100.0),
        },
        dt=dt,
    )
    start, breath = state.time, vent.breath_count
    engine.output_buffer.extend(
        WaveformSample(start + dt if index == substeps - 1 else start + (index + 1) * substep_dt,
                       ecg, pleth, float(co2), art, float(paw), float(flow), float(volume), breath)
        for index, ((ecg, pleth, art), co2, (paw, flow, volume)) in enumerate(
            zip(cardiac_waves, capno_waves, airway_waves, strict=True))
    )
