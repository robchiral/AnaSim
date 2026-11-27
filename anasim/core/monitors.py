from __future__ import annotations

import math
from typing import TYPE_CHECKING

from anasim.core.constants import ARREST_CO_L_MIN
from anasim.core.enums import RhythmType
from anasim.machine.ventilator import VOLUME_TARGETED
from anasim.monitors.arterial import MIN_PULSE_PRESSURE_MMHG, ArterialPressureSample
from anasim.monitors.cardiac_cycle import CardiacCycleSample
from anasim.monitors.nibp import NIBPReading
from anasim.physiology.airway_crisis import ESOPHAGEAL_RETURN_FRACTION
from anasim.physiology.disturbances import DisturbanceEffects
from anasim.physiology.resuscitation import (
    compression_mean_pressure,
    compression_pressures,
    compression_waveform_pressure,
)

from .projection import (
    bis_hypnotic_components,
    central_muscle_strength,
    circuit_connected,
    delivered_fraction,
    effective_mask_seal,
    is_esophageal_intubation,
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
    return max(1, math.ceil(dt / WAVEFORM_MAX_INTERVAL_S - 1e-12))


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
    was_cycling = engine.nibp.is_cycling
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
    # Bedside oscillometric monitors automatically schedule one prompt retry after
    # a failed measurement (without looping faster than 20 s).
    if was_cycling and not engine.nibp.is_cycling and engine.nibp.measurement_failed:
        engine._next_nibp_time = min(engine._next_nibp_time, state.time + dt + 20.0)

    latest = engine.nibp.latest_reading
    if latest.timestamp is not None and latest.timestamp != prev_ts:
        state.nibp_sys = float(latest.systolic)
        state.nibp_dia = float(latest.diastolic)
        state.nibp_map = float(latest.map)
        state.nibp_timestamp = float(latest.timestamp)


def _capno_sampling_possible(engine: SimulationEngine) -> bool:
    if is_esophageal_intubation(engine):
        return True
    state = engine.state
    return (
        state.airway_mode != AirwayType.NONE
        and (state.airway_mode != AirwayType.MASK or effective_mask_seal(engine) >= 0.05)
        and engine.airway_status.patency >= 0.05
    )


def compute_capno_value(
    engine: SimulationEngine,
    resp_state,
    dt: float,
    substeps: int = 1,
    *,
    cpr_oscillating: bool = False,
    port_share: float | None = None,
) -> list[float]:
    """Advance the capnograph with the gas that crossed the Y-piece; return CO2 at each waveform sample."""
    state = engine.state
    readout = engine.etco2_readout
    esophageal = is_esophageal_intubation(engine)
    et_gases = None if esophageal else (state.et_o2, state.et_sevo, state.et_n2o, state.et_mac)
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
        end_tidal = engine.airway_crisis.esophageal_co2_mmhg if esophageal else resp_state.etco2
        obstruction = 0.55 if esophageal else airway.capno_obstruction
        # Gas lost around a facemask never passes the sampling port, so a
        # large leak leaves dead-space gas at the port.
        if esophageal:
            port_share = 1.0
        elif port_share is None:
            port_share = delivered_fraction(engine)
        waves = []
        elapsed = 0.0
        substep_dt = dt / substeps
        mandatory_period = engine.vent.mandatory_period_s
        expected_period_s: float | None = (
            mandatory_period
            if mandatory_period is not None
            else (60.0 / resp_state.rr if resp_state.rr > 0.0 else None)
        )
        samples = engine.vent.samples or [(dt, 0.0, 0.0, 0.0, 0.0)]
        for duration, change, _, _, _ in samples:
            value = capno.step(
                duration,
                change * port_share,
                end_tidal,
                obstruction=obstruction,
                cpr_oscillating=cpr_oscillating,
            )
            readout.update(
                duration,
                capno.exhaling,
                value,
                inhaled_l=capno.inhaled_l,
                dead_space_l=capno.dead_space,
                breath_index=capno.breath_index,
                et_gases=et_gases,
                expected_period_s=expected_period_s,
                cpr_oscillating=cpr_oscillating,
                accept_decaying=esophageal,
            )
            elapsed += duration
            _hold(waves, value, elapsed, substeps, substep_dt)
        waves.extend([capno.co2] * (substeps - len(waves)))
    state.display_etco2 = float(readout.value)
    state.etco2_signal_valid = readout.valid
    state.et_o2, state.et_sevo, state.et_n2o, state.et_mac = (
        readout.et_gases if (readout.gas_valid and not esophageal) else (0.0, 0.0, 0.0, 0.0)
    )
    return waves


def _peripheral_perfusion(
    hemo_state, base_co: float, base_svr: float, temp_factor: float,
    *, cold_delay_factor: float | None = None,
) -> tuple[float, float]:
    """Return (pleth perfusion 0-1, SpO2 circulatory transit delay scale)."""
    co_ratio = hemo_state.co / base_co
    svr_ratio = hemo_state.svr / base_svr
    vaso_factor = 1.0 / (1.0 + max(0.0, svr_ratio - 1.0) ** 1.4 * 1.4)
    map_factor = clamp01((hemo_state.map - 12.0) / 10.0) * (0.35 + 0.65 * clamp01((hemo_state.map - 20.0) / 40.0))
    perfusion = clamp(co_ratio, 0.0, 1.0) * vaso_factor * temp_factor * map_factor
    delay_temp = temp_factor if cold_delay_factor is None else cold_delay_factor
    delay_scale = clamp(
        1.0 / max(0.08, clamp(co_ratio, 0.15, 1.5) * vaso_factor * delay_temp),
        0.75,
        12.0,
    )
    return perfusion, delay_scale


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
    base_co = engine.hemo.base_co_l_min
    base_svr = max(1.0, engine.hemo.base_tpr * 1000.0)
    temp_factor = clamp(1.0 - 0.14 * max(0.0, 36.0 - engine.state.temp_c), 0.12, 1.0)
    # Calibrated cold vasoconstriction delay on finger SpO2 transit (~2.5x transit
    # scale, ~2.1x total lag at 35 °C; delayed finger response in MacLeod 2005).
    cold_delay_factor = max(0.10, 1.0 / (1.0 + 0.85 * max(0.0, 36.5 - engine.state.temp_c) ** 1.35))
    perfusion, spo2_delay_scale = _peripheral_perfusion(
        hemo_state, base_co, base_svr, temp_factor, cold_delay_factor=cold_delay_factor,
    )
    st_depression = engine.hemo.coronary.st_depression_mv
    conduction_failure = engine.hemo.coronary.energy_failure
    ecmo = engine.extracorporeal
    pump_override = hemo_state.pump_flow > 0.0 or (
        ecmo.is_arterial_support and (ecmo.state.aortic_cross_clamp or ecmo.state.cardioplegia_arrest)
    )
    compressions = hemo_state.compressions and not pump_override
    if compressions:
        compression_sbp, compression_dbp = compression_pressures(
            hemo_state.cpr_perfusion_map, hemo_state.venous_pressure, filling_factor=engine.hemo.cpr_filling_factor,
        )
        compression_mean = compression_mean_pressure(compression_sbp, compression_dbp)
    else:
        engine.compression_clock.stop()

    cardiac_cycle = engine.cardiac_cycle
    art_waveform = engine.arterial_waveform
    art_line = engine.art_line
    ecg_mon = engine.ecg
    spo2_mon = engine.spo2_mon
    defib = engine.defibrillator
    sodium_block = engine.toxicity.conduction_block
    potassium = engine.hemo.effective_potassium
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
            new_rhythm = hemo_state.rhythm_type
            perfusion, spo2_delay_scale = _peripheral_perfusion(
                hemo_state, base_co, base_svr, temp_factor, cold_delay_factor=cold_delay_factor,
            )
            if new_rhythm != rhythm:
                rhythm = new_rhythm
                cardiac_sample = cardiac_cycle.seed(hemo_state.hr, rhythm)
            if compressions:
                compression_sbp, compression_dbp = compression_pressures(
                    hemo_state.cpr_perfusion_map, hemo_state.venous_pressure,
                    filling_factor=engine.hemo.cpr_filling_factor,
                )
                compression_mean = compression_mean_pressure(compression_sbp, compression_dbp)
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
                comp_p = compression_waveform_pressure(comp_sample.phase, compression_sbp, compression_dbp)
                if compression_mean > intrinsic_art.mean:
                    art_waveform.note_compression_baseline(compression_dbp)
                art_sbp = (
                    compression_sbp
                    if not math.isfinite(intrinsic_art.systolic)
                    else max(intrinsic_art.systolic, compression_sbp)
                )
                art_dbp = (
                    compression_dbp
                    if not math.isfinite(intrinsic_art.diastolic)
                    else max(intrinsic_art.diastolic, compression_dbp)
                )
                arterial_sample = ArterialPressureSample(
                    pressure=float(max(intrinsic_art.pressure, comp_p)),
                    systolic=float(art_sbp),
                    diastolic=float(art_dbp),
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
            arterial_static = (
                hemo_state.nonpulsatile_map
                if hemo_state.pump_flow > 0.0 and (hemo_state.sv <= 0.0 or not cardiac_sample.organized)
                else hemo_state.venous_pressure
            )
            arterial_sample = art_waveform.step(
                cardiac_sample,
                hemo_state.map,
                hemo_state.sv,
                dt=substep_dt,
                static_pressure=arterial_static,
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
            if compressions or hemo_state.intrinsic_co < ARREST_CO_L_MIN or pp_gate <= 0.0
            else perfusion * pp_gate
        )
        pleth, spo2_val = spo2_mon.step(
            substep_dt,
            pulse_sample,
            saturation=sao2,
            perfusion=spo2_perfusion,
            cpr_artifact=compressions,
            delay_scale=spo2_delay_scale,
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
    prop_bis, adjunct_bis = bis_hypnotic_components(engine)
    bis_val = engine.bis.step(
        dt,
        prop_bis,
        mac_sevo=mac_sevo,
        cortical_function=engine.cerebral.cortical_function * (1.0 - 0.55 * engine.neuraxial.unconsciousness),
        ce_ketamine=state.ketamine_ce,
        ce_opioid=state.opioid_ce,
        temp_c=state.temp_c,
        disturbance_bis=disturbances.bis,
        ce_adjunct=adjunct_bis,
    )
    substeps = waveform_substeps(dt)
    substep_dt = dt / substeps
    loc_val = loss_of_response(engine)
    engine._tol_current = stimulus_tolerance(engine)
    # On peripheral VA ECMO, standard pre-ductal (right-hand/forehead) pulse oximetry
    # monitors upper-body oxygenation so Harlequin desaturation is detected immediately.
    spo2_sao2 = (
        engine.extracorporeal.state.upper_body_sao2
        if engine.extracorporeal.is_arterial_support
        else state.sao2
    )
    cardiac_sample, cardiac_waves = step_cardiac_monitors(engine, dt, hemo_state, spo2_sao2)
    connected = state.airway_mode != AirwayType.NONE
    sealed = circuit_connected(engine)
    seal_scale = mask_seal_factor(engine)
    kept = delivered_fraction(engine) if sealed else 0.0
    capno_waves = compute_capno_value(
        engine, resp_state, dt, substeps, cpr_oscillating=hemo_state.compressions, port_share=kept,
    )

    update_nibp(engine, dt, engine.hemo.state)
    deep_block_ratio = (
        state.roc_ce / max(1e-6, engine.tof_pd._ec50(state.mac_sevo, state.mac_n2o))
        if engine.nmb.mode == "ptc" and state.roc_ce > 0.0
        else 0.0
    )
    engine.nmb.step(
        state.time + dt, state.twitch / 100.0, state.tof / 100.0,
        deep_block_ratio=deep_block_ratio,
    )

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
    paw_w, flow_w, vol_w = airway_wave
    state.paw, state.flow, state.volume = float(paw_w), float(flow_w), float(vol_w)
    bis_display_source = clamp(bis_val, 0.0, 100.0)
    state.bis = float(bis_display_source)

    decay_bis = math.exp(-dt / BIS_DISPLAY_TAU_S)
    bis_signal = engine.smooth_bis - engine._bis_noise
    bis_signal = decay_bis * bis_signal + (1.0 - decay_bis) * state.bis
    engine._bis_noise = (
        decay_bis * engine._bis_noise
        + float(engine.rng.normal(0.0, BIS_NOISE_SD)) * math.sqrt(max(0.0, 1.0 - decay_bis * decay_bis))
    )
    engine.smooth_bis = float(bis_signal + engine._bis_noise)

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
    if vent.is_on and vent.settings.mode in VOLUME_TARGETED and vent.settings.tv > 0.0:
        if not sealed:
            vte_percent = 0.0
        else:
            scale = ESOPHAGEAL_RETURN_FRACTION if is_esophageal_intubation(engine) else (1.0 - mask_leak(engine))
            delivered = vent.monitors.tv_exp_mandatory * scale
            vte_percent = None if math.isnan(delivered) else 100.0 * delivered / vent.settings.tv
    if state.display_bis < 65.0:
        engine._bis_anesthetized = True
    elif not connected and state.display_bis >= 80.0:
        engine._bis_anesthetized = False
    bis_int = display_int(state.display_bis)
    if bis_int <= engine.alarms.thresholds["BIS_max"]:
        bis_alarm: float | None = bis_int
    else:
        maintaining = engine._bis_anesthetized and (
            engine.disturbance_active
            or (engine.vaporizer.state.is_on and engine.circuit.fgf_total() > 0.0)
            or (engine.extracorporeal.is_active and engine.extracorporeal.state.sweep_sevo_percent > 0.0)
            or engine.circuit.fgf_n2o > 0.0
            or any(
                engine.infusion_rates.get(k, 0.0) > 0.0 or (k in engine.tci and engine.tci[k].target > 0.0)
                for k in ("propofol", "midazolam", "ketamine")
            )
        )
        if maintaining:
            bis_alarm = bis_int
        else:
            paralyzed = (
                state.tof < 90.0
                or nmba_twitch_fraction(engine) < 0.90
                or central_muscle_strength(engine) < 0.85
            )
            if paralyzed:
                # During induction/RSI, suppress transient BIS HIGH alarms while the patient is
                # already deeply unconscious (LOC >= 0.80) and undelayed hypnotic BIS is < 65,
                # waiting only for Eleveld/display smoothing to catch up.
                inducing_lag = (
                    not engine._bis_anesthetized
                    and loc_val >= 0.80
                    and engine.bis.compute_bis(
                        state.hypnotic_ce, mac_sevo, state.ketamine_ce, state.opioid_ce, temp_c=state.temp_c,
                    ) < 65.0
                )
                bis_alarm = None if inducing_lag else bis_int
            else:
                bis_alarm = None
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
    ppeak_val = display_int(state.paw_peak)
    ppeak_limit = engine.alarms.thresholds["Ppeak_max"]
    vent_pressure_alarm = (
        connected
        and vent.is_on
        and vent.pressure_limited
        and display_int(vent.monitors.paw_peak) >= min(ppeak_limit, display_int(vent.settings.p_max))
    )
    bag_pressure_alarm = (
        connected
        and not vent.is_on
        and engine.bag_mask_active
        and vent.pressure_limited
        and ppeak_val >= ppeak_limit
    )
    ppeak_alarm = ppeak_limit + 1.0 if (vent_pressure_alarm or bag_pressure_alarm) else ppeak_val
    bypass_silence = (
        engine.extracorporeal.is_arterial_support
        and engine.extracorporeal.state.bypass_alarm_mode
        and engine.hemo.state.pump_flow > ARREST_CO_L_MIN
    )
    state.alarms = engine.alarms.update(
        {
            "BIS": bis_alarm,
            "MAP": map_alarm,
            "HR": None if bypass_silence else display_int(state.display_hr),
            "EtCO2": None if bypass_silence else etco2_alarm,
            "RR": None if bypass_silence else (display_int(state.rr) if not vent.is_on else None),
            "Temp": round(state.temp_c, 1),
            "SpO2": None if bypass_silence else (display_int(state.display_spo2) if state.spo2_signal_valid else 0.0),
            "Ppeak": None if bypass_silence else ppeak_alarm,
            # The ventilator measures only gas returning through the circuit.
            "MV": None if bypass_silence else ((round(state.mv, 1) if sealed else 0.0) if vent.is_on else None),
            "VTe": None if bypass_silence else vte_percent,
            "FiO2": None if bypass_silence else (display_int(state.fio2 * 100.0) if sealed else None),
        },
        dt=dt,
    )
    if engine.extracorporeal.is_cpb and engine.extracorporeal.state.reservoir_low_alarm:
        state.alarms["ReservoirLow"] = {"low": True, "high": False}
    if not engine.circuit.oxygen_supply_connected:
        state.alarms["O2Supply"] = {"low": True, "high": False}
    rhythm = engine.hemo.state.rhythm_type
    if not bypass_silence:
        if rhythm == RhythmType.ASYSTOLE:
            state.alarms["Asystole"] = {"low": False, "high": True}
        elif rhythm == RhythmType.VFIB:
            state.alarms["VFIB"] = {"low": False, "high": True}
        elif rhythm == RhythmType.VTACH:
            state.alarms["VTACH"] = {"low": False, "high": True}
        elif state.cardiac_arrest:
            state.alarms["Arrest"] = {"low": False, "high": True}
    start, breath = state.time, vent.breath_count
    if substeps == 1:
        ecg, pleth, art = cardiac_waves[0]
        paw, flow, volume = airway_waves[0]
        engine.output_buffer.append(WaveformSample(
            start + dt, ecg, pleth, float(capno_waves[0]), art, float(paw), float(flow), float(volume), breath,
        ))
    else:
        engine.output_buffer.extend(
            WaveformSample(start + dt if index == substeps - 1 else start + (index + 1) * substep_dt,
                           ecg, pleth, float(co2), art, float(paw), float(flow), float(volume), breath)
            for index, ((ecg, pleth, art), co2, (paw, flow, volume)) in enumerate(
                zip(cardiac_waves, capno_waves, airway_waves, strict=True))
        )
