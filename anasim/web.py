"""One simulation session for the browser interface, with JSON commands and snapshots.

The Pyodide worker and the local server (`anasim.local`) both use `WebSession`.
"""

import json
import math
import os
import tempfile
from collections import deque

from anasim import __version__
from anasim.core.action_log import ACTION_DRUG_BOLUS
from anasim.core.clinical_actions import (
    ActionFeedback,
    check_airway_removal,
    check_drug_administration,
    check_extracorporeal_start,
    check_ventilation_withdrawal,
)
from anasim.core.drug_registry import get_drug_spec
from anasim.core.engine import SimulationEngine
from anasim.core.enums import RhythmType
from anasim.core.projection import assisted_positive_pressure, circuit_connected
from anasim.core.recorder import RecordingError
from anasim.core.state import AirwayType, SimulationConfig, WaveformSample
from anasim.core.utils import clamp01
from anasim.monitors.blood_gas import BloodGasResult
from anasim.monitors.ecg import displayed_rhythm, st_analyzable_scale
from anasim.patient import domain
from anasim.patient.patient import Patient
from anasim.physiology.coagulation import ACTResult
from anasim.physiology.disturbances import list_disturbance_profiles
from anasim.physiology.glucose import GlucoseResult
from anasim.physiology.resuscitation import SHOCK_ENERGY_J, ShockResult
from anasim.scenarios import SCENARIO_REGISTRY
from anasim.scenarios.base import Scenario
from anasim.scenarios.runtime import CaseRun

PATIENT_FIELDS = (
    "age",
    "weight",
    "height",
    "sex",
    "baseline_hb",
    "baseline_glucose",
    "renal_function",
    "creatinine_clearance_ml_min",
    "hepatic_function",
    "coronary_flow_reserve",
    "heart_failure",
    "copd_severity",
    "type2_diabetes",
    "surgical_site",
    "gestational_age_weeks",
    "position",
)
CONFIG_FIELDS = (
    "concomitant_opioids",
    "mode",
    "maint_type",
    "tci_enabled",
    "arterial_line_enabled",
    "fluid_warmer_enabled",
    "patient_variability",
)

SPEED_RANGE = (0.1, 50.0)
STEP_S = 0.05  # The engine still samples waveforms every 10 ms.
MAX_REAL_DT_S = 0.2
# Average achieved speed over several frames.
SPEED_AVERAGE_S = 2.0
WAVE_WINDOW_S = 20.0  # The slower respiratory sweep
TREND_WINDOW_S = 1800
VENT_FIELDS = ("mode", "rr", "tv", "peep", "ie", "p_insp", "p_support", "t_insp", "pause", "p_max", "trigger")
IE_RATIOS = ("1:1", "1:2", "1:3", "1:4")
LOOP_INTERVAL_S = 0.03  # Spacing of loop points, about 170 per 5-second breath
# Fluid kind: (engine method, receipt label).
FLUIDS = {
    "crystalloid": ("give_fluid", "lactated Ringer's"),
    "saline": ("give_saline", "0.9% saline"),
    "albumin": ("give_albumin", "5% albumin"),
    "d5w": ("give_d5w", "5% dextrose (D5W)"),
    "d10w": ("give_d10w", "10% dextrose (D10W)"),
    "d50w": ("give_d50w", "50% dextrose (D50W)"),
    "blood": ("give_blood", "PRBC"),
    "lipid": ("give_lipid_emulsion", "20% lipid emulsion over 2.5 min"),
}

SCENARIOS = {spec.id: spec for spec in SCENARIO_REGISTRY}
# Instructor event controls; cases start their own events and offer case actions instead.
EVENT_COMMANDS = frozenset({
    "hemorrhage", "anaphylaxis", "sepsis", "malignant_hyperthermia", "protamine_reaction",
    "difficult_airway", "esophageal_intubation", "pneumothorax", "pneumoperitoneum", "pulmonary_embolism",
    "venous_air_embolism", "high_spinal", "neuraxial_clear",
    "stop_events", "rhythm", "obstruction",
    "bronchospasm", "auto_laryngospasm", "ards_shunt", "disturbance", "disturbance_profile", "oxygen_supply",
})
# Commands still accepted after a case ends.
ENDED_CASE_COMMANDS = frozenset({"csht", "dismiss_feedback", "case_finish"})

LoopPoint = tuple[float | None, float | None, float | None]  # Paw, flow, volume above breath start


def catalog() -> str:
    """Return the choices the setup screen offers, as JSON."""
    return json.dumps({
        "version": __version__,
        "scenarios": [
            {
                "id": spec.id,
                "label": spec.label,
                "summary": spec.summary,
                "category": spec.category if spec.category is not None else ("crisis" if spec.surgery else "routine"),
                "start_mode": spec.start_mode,
                "maint_type": spec.maint_type,
            }
            for spec in SCENARIO_REGISTRY
        ],
        "ranges": {
            "age": domain.AGE_RANGE_YEARS,
            "weight": domain.WEIGHT_RANGE_KG,
            "height": domain.HEIGHT_RANGE_CM,
            "baseline_hb": domain.HEMOGLOBIN_RANGE_G_DL,
            "baseline_glucose": domain.BASELINE_GLUCOSE_RANGE_MG_DL,
            "creatinine_clearance_ml_min": domain.CREATININE_CLEARANCE_RANGE_ML_MIN,
            "gestational_age_weeks": domain.GESTATIONAL_AGE_RANGE_WEEKS,
        },
    })


def _num(value, digits=None):
    """Return a JSON-safe number; non-finite values become null."""
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits) if digits is not None else value


def _blood_gas(result: BloodGasResult) -> dict:
    return {
        "drawn_at": _num(result.drawn_at, 3), "ready_at": _num(result.ready_at, 3),
        "ph": _num(result.ph, 3), "pa_co2": _num(result.pa_co2, 1), "pao2": _num(result.pao2, 1),
        "hco3": _num(result.hco3, 2), "base_excess": _num(result.base_excess, 2),
        "lactate": _num(result.lactate, 2), "sao2": _num(result.sao2, 2), "hb": _num(result.hb, 2),
        "sodium": _num(result.sodium, 1), "potassium": _num(result.potassium, 2),
        "chloride": _num(result.chloride, 1), "ionized_calcium": _num(result.ionized_calcium, 2),
        "glucose": _num(result.glucose, 0),
        "fio2": _num(result.fio2, 3), "site": result.site,
    }


def _act(result: ACTResult) -> dict:
    return {
        "drawn_at": _num(result.drawn_at, 3),
        "ready_at": _num(result.ready_at, 3),
        "act_s": _num(result.act_s, 0),
    }


def _glucose(result: GlucoseResult) -> dict:
    return {
        "drawn_at": _num(result.drawn_at, 3),
        "ready_at": _num(result.ready_at, 3),
        "glucose": _num(result.glucose, 0),
    }


_DRUG_RATE_LIMITS: dict[str, float] = {
    "propofol": 500.0,
    "remi": 5.0,
    "fentanyl": 1000.0,
    "morphine": 100.0,
    "hydromorphone": 20.0,
    "midazolam": 50.0,
    "roc": 50.0,
    "nore": 500.0,
    "phenyl": 500.0,
    "epi": 200.0,
    "dobu": 100.0,
    "milri": 5.0,
    "esmolol": 1000.0,
    "vaso": 1.0,
    "heparin": 10000.0,
    "protamine": 100.0,
    "insulin": 50.0,
}
_UNIT_RATE_LIMITS: dict[str, float] = {
    "U/min": 1.0,
    "u/min": 1.0,
    "U/hr": 10000.0,
    "u/hr": 10000.0,
    "mcg/kg/min": 500.0,
    "mcg/min": 1000.0,
    "mcg/kg/hr": 25.0,
    "mg/kg/hr": 25.0,
    "mg/min": 50.0,
    "mg/hr": 500.0,
}


def _max_display_rate(spec) -> float:
    if spec.key in _DRUG_RATE_LIMITS:
        return _DRUG_RATE_LIMITS[spec.key]
    if spec.rate_unit in _UNIT_RATE_LIMITS:
        return _UNIT_RATE_LIMITS[spec.rate_unit]
    return 2000.0


def _target_label(spec) -> str:
    site = spec.fixed_tci_mode.value.replace("_", " ") if spec.fixed_tci_mode else "effect site"
    return f"{site.capitalize()} target"


class WebSession:
    """One simulation session driven by the browser UI."""

    def __init__(self, params: dict, recordings_dir: str | None = None, *, retain_recordings: bool = False):
        params = dict(params)
        unknown = set(params) - set(PATIENT_FIELDS) - set(CONFIG_FIELDS) - {"scenario_id", "practice_mode"}
        if unknown:
            raise ValueError(f"Unknown session setting(s): {', '.join(sorted(unknown))}")

        spec = None
        scenario_id = params.pop("scenario_id", None)
        practice_mode = params.pop("practice_mode", "guided")
        if practice_mode not in {"guided", "independent"}:
            raise ValueError("practice_mode must be 'guided' or 'independent'")
        if not scenario_id and practice_mode != "guided":
            raise ValueError("Independent practice requires a case")
        if scenario_id:
            spec = SCENARIOS[scenario_id]
            params.update(mode=spec.start_mode, maint_type=spec.maint_type)

        patient = Patient(**{key: params[key] for key in PATIENT_FIELDS if key in params})
        config = SimulationConfig(dt=STEP_S, **{key: params[key] for key in CONFIG_FIELDS if key in params})
        # Maintenance starts during surgery unless a scenario, such as emergence, says otherwise.
        surgery = spec.surgery if spec is not None else config.mode == "steady_state"
        if surgery:
            config.disturbance_profile = "stim_sustained_surgery"
        self.engine = SimulationEngine(patient, config)
        self.speed = 1.0
        self.achieved_speed = 1.0
        self.ended = False
        self.lagging = False
        self.notice: str | None = None
        self._clinical_feedback: dict | None = None
        self._observed_shock: tuple[float, bool] | None = None
        self._download: dict[str, str] | None = None
        self._accumulator = 0.0
        self._last_wave_time = -math.inf
        self._last_nmb_time = -math.inf
        # Loops trace the breath in progress over the last completed one.
        self._loop_sample: WaveformSample | None = None
        self._loop_pending: list[WaveformSample] = []  # Flow crossing awaiting confirmation by the breath counter.
        self._loop_breath: int | None = None
        self._loop_vent_breath: int | None = None
        self._loop_seq = 0
        self._loop_has_exhaled = False
        self._loop_start = 0.0
        self._loop_index = 0
        self._loop_current: list[LoopPoint] = []
        self._loop_previous: tuple[int | None, list[LoopPoint]] | None = None
        self._loop_new: list[tuple[int | None, LoopPoint]] = []
        self._loop_replay = False
        self._loop_reset = False
        self._recordings_dir = recordings_dir or os.path.join(tempfile.gettempdir(), "anasim-recordings")
        self.retain_recordings = retain_recordings
        self._medication_history: deque[dict] = deque(maxlen=50)
        self._medication_sequence = 0
        self._intubation_technique = "dl"

        self.scenario: Scenario | None = None
        self.case_run: CaseRun | None = None
        self.step_index = 0
        self.step_met = False
        self.step_status = ""
        self.step_effect_applied = False
        if spec is not None:
            self.scenario = spec.builder()
            self.scenario.prepare(self.engine)
            self.case_run = CaseRun(self.scenario, self.engine, practice_mode)
            if practice_mode == "guided":
                self._begin_step()
        self._trend_history: deque[dict] = deque(maxlen=TREND_WINDOW_S + 1)
        self._trend_pending: deque[dict] = deque(maxlen=TREND_WINDOW_S + 1)
        if self.engine._preroll_trends:
            self._trend_history.extend(self.engine._preroll_trends)
        self._next_trend_time = math.floor(self.engine.state.time) + 1
        self._sample_trends()

    def info(self) -> str:
        """Return session constants the UI needs once, as JSON."""
        engine = self.engine
        patient = engine.patient
        self._trend_pending.clear()
        return json.dumps({
            "version": __version__,
            "patient": {
                "age": patient.age,
                "sex": patient.sex,
                "weight": patient.weight,
                "height": patient.height,
                "renal_status": patient.renal_status,
                "creatinine_clearance_ml_min": patient.creatinine_clearance_ml_min,
                "hepatic_status": patient.hepatic_status,
                "coronary_flow_reserve": patient.coronary_flow_reserve,
                "heart_failure": patient.heart_failure,
                "copd_severity": patient.copd_severity,
                "type2_diabetes": patient.type2_diabetes,
                "baseline_glucose": patient.baseline_glucose,
                "surgical_site": patient.surgical_site,
                "gestational_age_weeks": patient.gestational_age_weeks,
            },
            "arterial_line": engine.config.arterial_line_enabled,
            "alarm_thresholds": dict(engine.alarms.thresholds),
            "recordings_dir": self._recordings_dir if self.retain_recordings else None,
            "sample_interval": engine.waveform_interval,
            "speed_range": SPEED_RANGE,
            "drugs": [
                {
                    "key": spec.key,
                    "name": spec.name,
                    "generic_name": spec.generic_name,
                    "rate_unit": spec.rate_unit,
                    "max_rate": _max_display_rate(spec) if spec.has_infusion else None,
                    "bolus_unit": spec.bolus_unit,
                    "default_bolus": spec.default_bolus_for(patient.weight),
                    "max_bolus": spec.max_bolus,
                    "tci_unit": spec.tci_unit if engine.config.tci_enabled else None,
                    "tci_range": spec.tci_range if engine.config.tci_enabled else None,
                    "target_label": _target_label(spec) if engine.config.tci_enabled and spec.has_tci else None,
                }
                for spec in engine.get_controllable_drugs()
            ],
            "medication_history": list(self._medication_history),
            "blood_gases": [_blood_gas(result) for result in engine.blood_gases],
            "acts": [_act(result) for result in engine.acts],
            "glucoses": [_glucose(result) for result in engine.glucoses],
            "trends": list(self._trend_history),
            "nmb_history": engine.nmb.trend_history(),
            "disturbances": [{"key": key, "label": label} for label, key in list_disturbance_profiles()],
            "rhythms": [rhythm.value for rhythm in RhythmType],
            "scenario": None if self.scenario is None or self.case_run is None else {
                "name": self.scenario.name,
                "mode": self.case_run.mode,
                "total": len(self.scenario),
            },
        })

    # --- Loop -----------------------------------------------------------

    def advance(self, real_dt: float) -> str:
        """Step by real_dt seconds of wall time at the current speed; return a snapshot."""
        self.step(real_dt)
        return json.dumps(self.snapshot())

    def _step_cap(self) -> float:
        """Maximum outer engine step (s) for the current speed; waveforms remain 100 Hz."""
        base = self.engine.config.dt
        recorder = self.engine.recorder
        if recorder is not None and recorder.is_recording and recorder.sample_interval_sec < 0.5:
            return base
        if self.speed > 25.0:
            return max(base, 0.50)
        if self.speed > 10.0:
            return max(base, 0.25)
        if self.speed > 4.0:
            return max(base, 0.10)
        return base

    def _next_step_dt(self, base: float, cap: float) -> float:
        """Choose a step that is a multiple of base without crossing trend or case-event boundaries."""
        if cap <= base + 1e-12:
            return base
        limit = min(cap, self._accumulator + 1e-12)
        engine = self.engine
        now = engine.state.time
        to_trend = self._next_trend_time - now
        if to_trend > base - 1e-12:
            limit = min(limit, to_trend + 1e-12)
        if self.case_run is not None:
            for event in self.case_run.scenario.events:
                if event.id not in self.case_run.fired:
                    to_event = (self.case_run.started_at + event.at) - now
                    if to_event > base - 1e-12:
                        limit = min(limit, to_event + 1e-12)
        recorder = engine.recorder
        if recorder is not None and recorder.is_recording and recorder.sample_interval_sec >= base:
            limit = min(limit, recorder.sample_interval_sec + 1e-12)
        steps = max(1, math.floor(limit / base))
        return round(steps * base, 12)

    def step(self, real_dt: float) -> None:
        """Advance without consuming snapshots, for the native server clock."""
        engine = self.engine
        if engine.running:
            start_time = engine.state.time
            clamped = min(max(float(real_dt), 0.0), MAX_REAL_DT_S)
            self._accumulator += clamped * self.speed
            base_step = engine.config.dt
            cap_step = self._step_cap()
            max_steps = max(100, math.ceil(MAX_REAL_DT_S * SPEED_RANGE[1] / base_step))
            steps = 0
            while (
                self._accumulator >= base_step - 1e-12
                and steps < max_steps
                and not engine.irreversible_arrest
            ):
                sim_step = self._next_step_dt(base_step, cap_step)
                try:
                    if self.case_run is not None:
                        self.case_run.before_step()
                    engine.step(sim_step)
                    self._observe_shock()
                except RecordingError as error:
                    self._pause()
                    self._download = self._take_recording(engine.recorder)
                    saved = "; the recorded part was downloaded" if self._download else ""
                    self.notice = f"{error}. The CSV is incomplete{saved}. The simulation is paused."
                    break
                finally:
                    # The model step completes before a recording write can fail.
                    if self.case_run is not None:
                        self.case_run.observe(sim_step)
                self._accumulator = max(0.0, self._accumulator - sim_step)
                steps += 1
                trend_eps = 1e-9 * max(1.0, abs(engine.state.time))
                if engine.state.time >= self._next_trend_time - trend_eps:
                    self._sample_trends()
                    self._next_trend_time = math.floor(engine.state.time + trend_eps) + 1
            # Discard unsimulated time to prevent a later catch-up.
            if steps >= max_steps:
                self._accumulator = 0.0
            if real_dt > 0 and engine.running:
                weight = min(1.0, real_dt / SPEED_AVERAGE_S)
                rate = (engine.state.time - start_time) / real_dt
                self.achieved_speed += weight * (rate - self.achieved_speed)
                self.lagging = self.achieved_speed < 0.9 * self.speed
            if engine.irreversible_arrest:
                self.ended = True
                self._pause()
                if self.case_run is not None:
                    self._finish_case("cardiac_arrest")
                else:
                    download = self._cmd_record(False)
                    if download is not None:
                        self._download = download

    def close(self) -> None:
        """Pause and finish any recording before disconnect or shutdown."""
        self._pause()
        self.engine.stop_recording()

    def _pause(self):
        self.engine.stop()
        self._accumulator = 0.0
        self._reset_speed_average()

    def _reset_speed_average(self):
        self.achieved_speed = self.speed
        self.lagging = False

    def snapshot(self) -> dict:
        """Return monitor values, new waveform samples, and control state."""
        self._update_scenario()
        notice, self.notice = self.notice, None
        download, self._download = self._download, None
        state = self.engine.state
        trends = list(self._trend_pending)
        self._trend_pending.clear()
        nmb_trend = self.engine.nmb.trend_history(after=self._last_nmb_time)
        if nmb_trend:
            self._last_nmb_time = nmb_trend[-1][0]
        return {
            "time": _num(state.time, 3),
            "running": self.engine.running,
            "speed": self.speed,
            "achieved_speed": round(self.achieved_speed, 1),
            "lagging": self.lagging,
            "ended": self.ended,
            "arrest_reason": state.arrest_reason if state.cardiac_arrest else None,
            "recording": self._recording(),
            "notice": notice,
            "download": download,
            "vitals": self._vitals(),
            "nmb": self.engine.nmb.snapshot(state.time),
            "nmb_trend": nmb_trend,
            "trends": trends,
            "alarms": {
                name: "low" if flags.get("low") else "high"
                for name, flags in state.alarms.items()
                if flags.get("low") or flags.get("high")
            },
            "waves": self._new_waves(),
            "loop": self._take_loop(),
            "controls": self._controls(),
            "scenario": self._scenario_state(),
        }

    def _sample_trends(self) -> None:
        engine = self.engine
        s = engine.state
        arterial = engine.config.arterial_line_enabled
        if arterial:
            pressure = _num(s.art_map, 1)
            pressure_time: float | None = s.time
        elif s.nibp_timestamp is None or s.nibp_measurement_failed:
            pressure = None
            pressure_time = None
        else:
            pressure = _num(s.nibp_map, 1)
            pressure_time = s.nibp_timestamp
        sample = {
            "time": _num(s.time, 3),
            "hr": None if engine.hemo.state.rhythm_type == RhythmType.VFIB else _num(s.display_hr, 1),
            "etco2": _num(s.display_etco2, 1) if s.etco2_signal_valid else None,
            "bis": _num(s.display_bis, 1),
            "map": pressure,
            "map_time": None if pressure_time is None else _num(pressure_time, 3),
        }
        self._trend_history.append(sample)
        self._trend_pending.append(sample)

    def _vitals(self) -> dict:
        engine = self.engine
        s = engine.state
        # Spirometry needs gas through the circuit, Pplat a positive-pressure
        # breath, and end-tidal values an exhaled breath.
        connected = circuit_connected(engine)
        assisted = connected and (engine.vent.is_on or assisted_positive_pressure(engine))
        exhaled = (s.etco2_signal_valid or engine.etco2_readout.gas_valid) and not s.esophageal_intubation
        nibp_ts = (
            engine.nibp.failed_timestamp
            if (s.nibp_measurement_failed and engine.nibp.failed_timestamp is not None)
            else s.nibp_timestamp
        )
        st_scale = st_analyzable_scale(
            engine.hemo.state.rhythm_type,
            engine.hemo.coronary.energy_failure,
            engine.toxicity.conduction_block,
            engine.hemo.hyperkalemia_block,
        )
        return {
            "hr": None if engine.hemo.state.rhythm_type == RhythmType.VFIB else _num(s.display_hr, 1),
            "st_depression_mv": _num(s.st_depression_mv, 2) if st_scale > 0.01 else None,
            "spo2": _num(s.display_spo2, 1) if s.spo2_signal_valid else None,
            "art": [_num(s.art_sbp, 1), _num(s.art_dbp, 1), _num(s.art_map, 1)],
            "nibp": None if (s.nibp_timestamp is None or s.nibp_measurement_failed) else [
                _num(s.nibp_sys, 1), _num(s.nibp_dia, 1), _num(s.nibp_map, 1)
            ],
            "nibp_cuff": _num(s.nibp_cuff_pressure, 1) if s.nibp_is_cycling else None,
            "nibp_age": None if nibp_ts is None else _num(max(0.0, s.time - nibp_ts), 0),
            "nibp_failed": s.nibp_measurement_failed,
            "etco2": _num(s.display_etco2, 1) if s.etco2_signal_valid else None,
            "rr": _num(s.rr, 1),
            # Without an airway, respiratory rate comes from chest impedance.
            "rr_source": "co2" if connected else "impedance",
            "bis": _num(s.display_bis, 1),
            "temp": _num(s.temp_c, 2),
            "fluid_in": _num(s.fluid_in_ml, 1),
            "blood_in": _num(s.blood_in_ml, 1),
            "urine_out": _num(s.urine_out_ml, 1),
            "blood_out": _num(s.blood_out_ml, 1),
            "net_fluid": _num(s.net_fluid_ml, 1),
            "ppeak": _num(s.paw_peak, 1) if connected else None,
            "pplat": _num(s.paw_plat, 1) if assisted else None,
            "peep": _num(s.peep, 1) if connected else None,
            "pmean": _num(s.paw_mean, 1) if connected else None,
            "vte": _num(s.vt, 0) if connected else None,
            "mv": _num(s.mv, 2) if connected else None,
            "cdyn": _num(s.compliance_dyn, 0) if assisted else None,
            "fio2": _num(s.fio2 * 100.0, 1),
            "eto2": _num(s.et_o2, 1) if exhaled else None,
            "fi_n2o": _num(s.fi_n2o, 1),
            "et_n2o": _num(s.et_n2o, 1) if exhaled else None,
            "fi_sevo": _num(s.fi_sevo, 2),
            "et_sevo": _num(s.et_sevo, 2) if exhaled else None,
            "et_mac": _num(s.et_mac, 3) if exhaled else None,
        }

    def replay_waves(self) -> None:
        """Resend the retained sweep and loops in the next snapshot, for a reloaded page."""
        self._last_wave_time = -math.inf
        self._loop_replay = True

    def _new_waves(self) -> dict:
        """Return samples newer than the last snapshot, at most one sweep."""
        buffer = self.engine.output_buffer
        limit = round(WAVE_WINDOW_S / self.engine.waveform_interval)
        samples = []
        for sample in reversed(buffer):
            if sample.time <= self._last_wave_time:
                break
            samples.append(sample)
            if len(samples) >= limit:
                break
        if samples:
            self._last_wave_time = samples[0].time
        samples.reverse()
        self._collect_loop(samples)
        return {
            "ecg": [_num(s.ecg_voltage, 4) for s in samples],
            "pleth": [_num(s.pleth_voltage, 4) for s in samples],
            "co2": [_num(s.capno_co2, 2) for s in samples],
            "art": [_num(s.art_pressure, 2) for s in samples],
            "paw": [_num(s.paw, 2) for s in samples],
            "flow": [_num(s.flow, 1) for s in samples],
        }

    def _reset_loop(self) -> None:
        """Clear retained spirometry loops on airway disconnect."""
        had_loop = bool(
            self._loop_previous is not None
            or self._loop_current
            or self._loop_breath is not None
            or self._loop_pending
            or self._loop_new
        )
        self._loop_previous = None
        self._loop_current = []
        self._loop_breath = None
        self._loop_vent_breath = None
        self._loop_has_exhaled = False
        self._loop_pending = []
        self._loop_new = []
        if had_loop:
            self._loop_reset = True

    def _collect_loop(self, samples) -> None:
        """Add points of the breath in progress; a new breath makes it the previous loop."""
        if not circuit_connected(self.engine):
            self._reset_loop()
            if samples:
                self._loop_sample = samples[-1]
            return
        stride = max(1, round(LOOP_INTERVAL_S / self.engine.waveform_interval))
        for sample in samples:
            previous = self._loop_sample
            if previous is not None and sample.time <= previous.time:
                continue  # Replayed for the sweep
            self._loop_sample = sample
            if previous is None or self._loop_breath is None:
                self._start_loop(sample)
                continue
            # The filtered flow crossing can precede or follow breath detection.
            # Retain it until the counter confirms a breath; a small cleft that
            # returns to expiration stays in the existing loop.
            if previous.flow <= 0 < sample.flow:
                fraction = -previous.flow / (sample.flow - previous.flow)
                boundary = sample._replace(
                    time=previous.time + fraction * (sample.time - previous.time),
                    paw=previous.paw + fraction * (sample.paw - previous.paw),
                    volume=previous.volume + fraction * (sample.volume - previous.volume),
                    flow=0.0,
                )
                self._loop_pending = [boundary]
            if self._loop_pending:
                self._loop_pending.append(sample)
                first_boundary = self._loop_pending[0]
                inspired_l = sample.volume - first_boundary.volume
                vent_advanced = sample.breath != self._loop_vent_breath
                if vent_advanced or (not self._loop_has_exhaled and inspired_l >= 0.015) or inspired_l >= 0.060:
                    boundary = first_boundary._replace(breath=sample.breath)
                    self._start_loop(boundary, promote_previous=self._loop_has_exhaled)
                    pending = self._loop_pending[1:]
                elif sample.flow <= 0:
                    pending = self._loop_pending
                else:
                    continue
                for point in pending:
                    self._sample_loop(point, stride)
                self._loop_pending = []
            elif sample.breath != self._loop_vent_breath and (previous.flow >= 0 or sample.flow >= 0):
                if not self._loop_has_exhaled and len(self._loop_current) > 2:
                    self._loop_vent_breath = sample.breath
                    self._sample_loop(sample, stride)
                else:
                    self._start_loop(sample, promote_previous=self._loop_has_exhaled)
            else:
                self._sample_loop(sample, stride)
        # Apnea leaves one breath open; keep a sweep of it.
        limit = round(WAVE_WINDOW_S / LOOP_INTERVAL_S)
        if len(self._loop_current) > limit:
            self._loop_current = self._loop_current[-limit:]

    def _start_loop(self, boundary, *, promote_previous: bool = True) -> None:
        if promote_previous and len(self._loop_current) > 1:
            # Preserve changes in end-expiratory volume.
            self._append_loop_point(boundary.paw, boundary.flow, boundary.volume)
            self._loop_previous = (self._loop_breath, self._loop_current)
        self._loop_seq += 1
        self._loop_breath = self._loop_seq
        self._loop_vent_breath = boundary.breath
        self._loop_has_exhaled = boundary.flow < -0.5
        self._loop_start = boundary.volume
        self._loop_index, self._loop_current = 0, []
        self._append_loop_point(boundary.paw, boundary.flow, boundary.volume)

    def _sample_loop(self, sample, stride) -> None:
        if sample.flow < -0.5:
            self._loop_has_exhaled = True
        if self._loop_index % stride == 0:
            self._append_loop_point(sample.paw, sample.flow, sample.volume)
        self._loop_index += 1

    def _append_loop_point(self, paw, flow, volume) -> None:
        point = (_num(paw, 2), _num(flow, 1), _num((volume - self._loop_start) * 1000.0, 1))
        self._loop_current.append(point)
        self._loop_new.append((self._loop_breath, point))

    def _take_loop(self) -> list:
        """Return new loop points as runs of (breath, paw, flow, volume); a replay resends both loops."""
        reset = self._loop_reset
        self._loop_reset = False
        if self._loop_replay:
            self._loop_replay = False
            runs = [self._loop_previous] if self._loop_previous else []
            if self._loop_current:
                runs.append((self._loop_breath, self._loop_current))
            points = [(breath, point) for breath, loop in runs for point in loop]
        else:
            points = self._loop_new
        self._loop_new = []
        out: list[dict] = [{"reset": True}] if reset else []
        for breath, (paw, flow, volume) in points:
            if not out or out[-1].get("breath") != breath:
                out.append({"breath": breath, "paw": [], "flow": [], "volume": []})
            run = out[-1]
            run["paw"].append(paw)
            run["flow"].append(flow)
            run["volume"].append(volume)
        return out

    def _reservoir_status(self) -> dict:
        engine = self.engine
        circuit = engine.circuit
        fill = clamp01(circuit.reservoir_volume / max(1e-6, circuit.reservoir_capacity))
        connected = circuit_connected(engine)
        pressure = engine._apl_pressure if connected and not engine.vent.is_on else 0.0
        if circuit.bellows_fraction <= 0.05:
            state = "empty"
        elif fill < 0.35:
            state = "low"
        elif not engine.vent.is_on and connected and pressure >= 25.0:
            state = "high"
        elif not engine.vent.is_on and connected and pressure >= 12.0:
            state = "tense"
        else:
            state = "normal"
        return {
            "fill": round(fill, 3),
            "pressure": round(pressure, 1),
            "state": state,
        }

    def _controls(self) -> dict:
        engine = self.engine
        circuit = engine.circuit
        vent = engine.vent.settings
        return {
            "airway": engine.state.airway_mode.value,
            "mask_seal": engine.mask_seal * 100.0,
            "clinical_feedback": self._clinical_feedback,
            "convulsions": engine.state.convulsions,
            "muscle_rigidity": engine.state.muscle_rigidity,
            "surface_cooling": engine.state.surface_cooling,
            "dantrolene_total_mg": engine.actions.total(ACTION_DRUG_BOLUS, "dantrolene"),
            "lipid": {"rate": engine.lipid_infusion_ml_min, "total_ml": engine.toxicity.total_lipid_ml},
            "position": engine.hemo.position,
            "fgf": {"o2": circuit.fgf_o2, "air": circuit.fgf_air, "n2o": circuit.fgf_n2o},
            "o2_connected": circuit.oxygen_supply_connected,
            "o2_flush": engine._o2_flush_remaining_s > 0.0,
            "circuit_fio2": _num(circuit.composition.fio2, 4),
            "vaporizer": engine.vaporizer.state.setting,
            "vaporizer_level": round(engine.vaporizer.state.level, 1),
            "bag_mask": engine.bag_mask_active,
            "compressions": {"active": engine.hemo.compressions, "since": engine.compressions_started_at},
            "defibrillator": {
                "status": engine.defibrillator.status(engine.state.time),
                "synchronized": engine.defibrillator.waiting_for_r,
                "charge_remaining": max(0.0, engine.defibrillator.ready_at - engine.state.time)
                if engine.defibrillator.ready_at is not None else 0.0,
                "last_shock": self._shock_receipt(),
            },
            "bag": {
                "rr": round(engine.bag_mask_rr),
                "tv": round(engine.bag_mask_vt * 1000.0),
            },
            "reservoir": self._reservoir_status(),
            "apl": engine.apl_cmh2o,
            "vent": {
                "on": engine.vent.is_on,
                "mode": vent.mode,
                "rr": round(vent.rr),
                "tv": round(vent.tv),
                "peep": round(vent.peep),
                "ie": vent.ie,
                "p_insp": round(vent.p_insp),
                "p_support": round(vent.p_support),
                "t_insp": round(vent.t_insp, 1),
                "pause": round(vent.pause),
                "p_max": round(vent.p_max),
                "trigger": round(vent.trigger, 1),
            },
            "drugs": {spec.key: engine._drug_state_for_spec(spec) for spec in engine.get_controllable_drugs()},
            "maintenance_ml_hr": engine.get_continuous_fluid_rate(),
            "disturbance": {"profile": engine.disturbance_profile, "active": bool(
                engine.disturbance_active and engine.disturbance_profile
            )},
            "obstruction": engine.airway_obstruction_manual * 100.0,
            "bronchospasm": engine.bronchospasm_manual * 100.0,
            "ards_shunt": round(engine.aeration.extra_shunt * 100.0, 1),
            "laryngospasm": engine.laryngospasm_level,
            "auto_laryngospasm": engine.auto_laryngospasm_enabled,
            "hemorrhage": {"active": engine.active_hemorrhage, "rate": engine.hemorrhage_rate_ml_min},
            "anaphylaxis": engine.anaphylaxis.active,
            "sepsis": engine.sepsis.active,
            "malignant_hyperthermia": engine.malignant_hyperthermia.active,
            "protamine_reaction": engine.coagulation.reaction_active,
            "ecg_rhythm": displayed_rhythm(
                engine.hemo.state.rhythm_type,
                engine.hemo.coronary.energy_failure,
                max(engine.toxicity.conduction_block, engine.hemo.hyperkalemia_block),
                hr=engine.state.hr,
            ),
            "bair_hugger": engine.state.bair_hugger_target,
            "fluid_warmer": engine.state.fluid_warmer,
            "extracorporeal": self._extracorporeal_controls(),
            "airway_crisis": {
                "difficult_profile": engine.airway_crisis.difficult_profile,
                "attempts": engine.airway_crisis.laryngoscopy_attempts,
                "last_view": engine.airway_crisis.last_view,
                "technique": self._intubation_technique,
                "help_called": engine.airway_crisis.help_called,
                "oral_airway": engine.airway_crisis.oral_airway,
                "two_hand_mask": engine.airway_crisis.two_hand_mask,
                "esophageal_armed": engine.airway_crisis.esophageal_armed,
                "esophageal_active": engine.airway_crisis.esophageal_intubation,
                "fona_performed": engine.airway_crisis.fona_performed,
            },
            "pneumothorax": {
                "active": engine.pneumothorax.active,
                "decompressed": engine.pneumothorax.decompressed,
                "pressure": _num(engine.pneumothorax.pleural_pressure_mmhg, 1),
            },
            "pneumoperitoneum": {
                "active": engine.pneumoperitoneum.active,
                "target_iap": _num(engine.pneumoperitoneum.target_iap_mmhg, 1),
                "iap": _num(engine.pneumoperitoneum.iap_mmhg, 1),
            },
            "embolism": {
                "pe_active": engine.embolism.pe_active,
                "pe_obstruction": _num(engine.embolism.pe_obstruction, 2),
                "thrombolysis": engine.embolism.thrombolysis_active,
                "vae_active": engine.embolism.vae_active,
                "vae_air_ml": _num(engine.embolism.vae_air_ml, 1),
            },
            "neuraxial": {
                "active": engine.neuraxial.active,
                "mode": engine.neuraxial.last_mode,
                "sensory_label": engine.neuraxial.sensory_label,
                "motor_label": engine.neuraxial.motor_label,
                "bromage": engine.neuraxial.bromage_score,
                "epidural_catheter": engine.neuraxial.epidural_catheter,
                "epidural_rate": _num(engine.neuraxial.epidural_infusion_ml_hr, 1),
                "last_check": None if engine.neuraxial.last_check is None else {
                    "time": _num(engine.neuraxial.last_check.time, 1),
                    "sensory_label": engine.neuraxial.last_check.sensory_label,
                    "motor_label": engine.neuraxial.last_check.motor_label,
                    "bromage": engine.neuraxial.last_check.bromage,
                    "note": engine.neuraxial.last_check.note,
                },
            },
        }

    def _extracorporeal_controls(self) -> dict:
        ec = self.engine.extracorporeal
        st = ec.state
        return {
            "mode": st.mode.value,
            "active": ec.is_active,
            "cannulation": st.cannulation.value,
            "blood_gas_strategy": st.blood_gas_strategy.value,
            "target_flow": _num(st.target_blood_flow_l_min, 2),
            "flow": _num(st.blood_flow_l_min, 2),
            "sweep": _num(st.sweep_gas_l_min, 2),
            "fio2": _num(st.sweep_fio2 * 100.0, 0),
            "sevo": _num(st.sweep_sevo_percent, 1),
            "heater_cooler_on": bool(st.heater_cooler_on),
            "water_target_c": _num(st.water_target_c, 1),
            "venous_clamp": _num(st.venous_clamp_open * 100.0, 0),
            "cross_clamp": bool(st.aortic_cross_clamp),
            "cardioplegia": bool(st.cardioplegia_arrest),
            "lv_vent": _num(st.lv_vent_l_min, 2),
            "bypass_alarm_mode": bool(st.bypass_alarm_mode),
            "reservoir_ml": _num(st.reservoir_volume_ml, 0),
            "reservoir_low": bool(st.reservoir_low_alarm),
            "pre_so2": _num(st.pre_oxy_so2, 1),
        }

    # --- Scenario -------------------------------------------------------

    def _begin_step(self):
        self.step_met = False
        self.step_status = ""
        self.step_effect_applied = False

    def _update_scenario(self):
        if self.case_run is not None:
            self.case_run.observe()
            if self.case_run.mode == "independent" or self.case_run.report is not None:
                return
        if self.scenario is None or self.step_index >= len(self.scenario):
            return
        step = self.scenario[self.step_index]
        met, status = step.check_requirements(self.engine)
        self.step_met = bool(met)
        self.step_status = status
        if met and step.on_met is not None and not self.step_effect_applied:
            self.step_effect_applied = True
            step.on_met(self.engine)

    def _scenario_state(self):
        if self.scenario is None or self.case_run is None:
            return None
        state = self.case_run.snapshot()
        if self.case_run.mode == "independent" or state["complete"]:
            return state
        step = self.scenario[self.step_index]
        return {**state,
            "index": self.step_index,
            "complete": False,
            "id": step.id,
            "title": step.title,
            "instruction": step.instruction_for(self.engine),
            "target_tab": step.target_tab,
            "met": self.step_met,
            "status": "" if self.step_met else self.step_status,
        }

    # --- Commands -------------------------------------------------------

    def command(self, name: str, args_json: str = "{}") -> str:
        """JSON command entry point for the Pyodide worker."""
        return json.dumps(self.apply_command(name, json.loads(args_json or "{}")))

    def apply_command(self, name: str, args: dict):
        """Apply one learner control and return its result."""
        if self.case_run is not None:
            if self.case_run.report is not None:
                stopping = (name == "record" and not args.get("active")) or (name == "run" and not args.get("running"))
                if not (name in ENDED_CASE_COMMANDS or stopping):
                    raise ValueError("The case has ended. Start a new session to continue.")
            if name in EVENT_COMMANDS:
                raise ValueError("Case events are controlled by the case. Use the available case actions.")
        elif self.ended:
            stopping = name == "run" or (name == "record" and not args.get("active"))
            if not (name in ENDED_CASE_COMMANDS or stopping):
                raise ValueError("The session has ended. Start a new session to continue.")
        handler = getattr(self, f"_cmd_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown command: {name!r}")
        result = handler(**args)
        if self.case_run is not None:
            self.case_run.observe(force=True)
        return result

    def _cmd_run(self, running: bool):
        if running and not self.ended:
            self.engine.start()
            self._accumulator = 0.0
            self._reset_speed_average()
        else:
            self._pause()

    def _cmd_speed(self, value: float):
        self.speed = min(max(domain.finite_number("speed", value), SPEED_RANGE[0]), SPEED_RANGE[1])
        self._reset_speed_average()

    def _cmd_airway(self, mode: str, technique: str | None = None):
        target = AirwayType(mode)
        if target == AirwayType.ETT:
            tech = self._intubation_technique if technique is None else technique
            if not isinstance(tech, str) or tech.strip().lower() not in {"dl", "vl", "vl_bougie", "fiberoptic"}:
                raise ValueError(f"Unsupported intubation technique: {tech!r}")
            tech = tech.strip().lower()
            self._intubation_technique = tech
            return self._clinical_result(self.engine.attempt_intubation(tech))
        if target == AirwayType.FONA:
            return self._clinical_result(self.engine.perform_fona())
        prev = self.engine.state.airway_mode
        was_esophageal = prev == AirwayType.ETT and self.engine.airway_crisis.esophageal_intubation
        was_assisted = circuit_connected(self.engine) and not was_esophageal and assisted_positive_pressure(self.engine)
        leaving_infraglottic = prev in (AirwayType.ETT, AirwayType.FONA)
        feedback = check_airway_removal(self.engine) if leaving_infraglottic else None
        if target == AirwayType.SGA:
            sga_fb = self.engine.attempt_sga()
            if sga_fb is not None and sga_fb.blocked:
                return self._clinical_result(sga_fb)
            if self.case_run is not None and prev == AirwayType.ETT and not was_esophageal:
                self.case_run.record_transition("extubation")
            if feedback is not None and sga_fb is not None and self.case_run is not None:
                self.case_run.record_safety(sga_fb)
            return self._clinical_result(feedback or sga_fb)
        if self.case_run is not None and prev == AirwayType.ETT and not was_esophageal:
            self.case_run.record_transition("extubation")
        self.engine.set_airway_mode(mode)
        if not circuit_connected(self.engine):
            self._reset_loop()
        if feedback is None and target == AirwayType.NONE and was_assisted:
            feedback = check_ventilation_withdrawal(self.engine)
        return self._clinical_result(feedback)

    def _cmd_airway_technique(self, technique: str):
        if not isinstance(technique, str):
            raise ValueError(f"Unsupported intubation technique: {technique!r}")
        tech = technique.strip().lower()
        if tech not in {"dl", "vl", "vl_bougie", "fiberoptic"}:
            raise ValueError(f"Unsupported intubation technique: {technique!r}")
        self._intubation_technique = tech

    def _cmd_oral_airway(self, placed: bool):
        return self._clinical_result(self.engine.set_oral_airway(bool(placed)))

    def _cmd_two_hand_mask(self, active: bool):
        self.engine.set_two_hand_mask(bool(active))

    def _cmd_call_airway_help(self):
        self.engine.call_airway_help()

    def _cmd_perform_fona(self):
        return self._clinical_result(self.engine.perform_fona())

    def _cmd_chest_decompress(self):
        self.engine.decompress_pneumothorax()
        return {"procedure": {"time": self.engine.state.time, "text": "Chest decompression performed"}}

    def _cmd_vae_aspirate(self):
        removed = self.engine.aspirate_vae_air()
        text = (
            f"Aspirated {removed:.0f} mL air from CVC"
            if removed >= 1.0
            else "CVC aspirated (no significant air returned)"
        )
        return {"procedure": {"time": self.engine.state.time, "text": text}}

    def _cmd_pe_thrombolysis(self):
        self.engine.give_thrombolysis()
        return {"procedure": {"time": self.engine.state.time, "text": "Systemic thrombolysis started"}}

    def _cmd_neuraxial_spinal(
        self, drug: str = "bupivacaine", dose_mg: float = 12.5, baricity: str = "hyperbaric",
    ):
        dose = domain.bounded_number("spinal dose", dose_mg, 1.0, 150.0, unit="mg")
        self.engine.give_spinal(drug=drug, dose_mg=dose, baricity=baricity)
        text = f"Spinal {baricity} {drug} {dose:g} mg given"
        return self._medication_receipt("neuraxial", text)

    def _cmd_neuraxial_epidural_catheter(self, placed: bool = True):
        self.engine.place_epidural(bool(placed))

    def _cmd_neuraxial_epidural_bolus(
        self,
        drug: str = "bupivacaine",
        volume_ml: float = 10.0,
        concentration_pct: float = 0.5,
        with_epinephrine: bool = False,
    ):
        vol = domain.bounded_number("epidural bolus volume", volume_ml, 1.0, 30.0, unit="mL")
        conc = domain.bounded_number("epidural concentration", concentration_pct, 0.05, 2.0, unit="%")
        self.engine.give_epidural_bolus(
            drug=drug,
            volume_ml=vol,
            concentration_pct=conc,
            with_epinephrine=bool(with_epinephrine),
        )
        epi_str = " + 1:200k epi" if with_epinephrine else ""
        text = f"Epidural {vol:g} mL {conc:g}% {drug}{epi_str}"
        return self._medication_receipt("neuraxial", text)

    def _cmd_neuraxial_epidural_rate(
        self, ml_hr: float, drug: str = "bupivacaine", concentration_pct: float = 0.125,
    ):
        rate = domain.bounded_number("epidural infusion rate", ml_hr, 0.0, 25.0, unit="mL/hr")
        conc = domain.bounded_number("epidural concentration", concentration_pct, 0.05, 0.5, unit="%")
        self.engine.set_epidural_infusion(rate, drug=drug, concentration_pct=conc)
        text = (
            f"Epidural infusion {rate:g} mL/hr ({conc:g}% {drug})"
            if rate > 0
            else "Epidural infusion stopped"
        )
        return self._medication_receipt("neuraxial", text)

    def _cmd_neuraxial_check_level(self):
        check = self.engine.check_neuraxial_level()
        return {
            "neuraxial_check": {
                "time": _num(check.time, 1),
                "sensory_label": check.sensory_label,
                "motor_label": check.motor_label,
                "bromage": check.bromage,
                "note": check.note,
            },
        }

    def _cmd_neuraxial_clear(self):
        self.engine.stop_neuraxial()

    def _cmd_mask_seal(self, percent: float):
        self.engine.set_mask_seal(domain.bounded_number("mask seal", percent, 0.0, 100.0, unit="%") / 100.0)
        if not circuit_connected(self.engine):
            self._reset_loop()

    def _clinical_result(self, feedback: ActionFeedback | None, receipt: dict | None = None):
        """Record feedback for the debrief and return what the browser shows now."""
        informational = feedback is not None and feedback.code in {"intubation_note", "failed_laryngoscopy", "sga_failed"}
        if feedback is not None and self.case_run is not None and not informational:
            self.case_run.record_safety(feedback)
            if self.case_run.mode == "independent" and not feedback.blocked:
                feedback = None
        if feedback is None:
            if self._clinical_feedback is not None:
                self._clinical_feedback = None
                return {**(receipt or {}), "feedback": None}
            return receipt
        self._clinical_feedback = {
            "code": feedback.code,
            "level": "blocked" if feedback.blocked else "warning",
            "message": feedback.message,
            "time": self.engine.state.time,
        }
        return {**(receipt or {}), "feedback": self._clinical_feedback}

    def _cmd_dismiss_feedback(self):
        self._clinical_feedback = None
        return {"feedback": None}

    def _cmd_fgf(self, o2: float, air: float, n2o: float):
        self.engine.set_fgf(
            domain.bounded_number("oxygen flow", o2, 0.0, 15.0, unit="L/min"),
            domain.bounded_number("air flow", air, 0.0, 15.0, unit="L/min"),
            domain.bounded_number("nitrous oxide flow", n2o, 0.0, 15.0, unit="L/min"),
        )

    def _cmd_oxygen_supply(self, connected: bool):
        self.engine.set_oxygen_supply_connected(connected)

    def _cmd_vaporizer(self, percent: float):
        self.engine.set_vaporizer(percent)

    def _cmd_refill_vaporizer(self, level_ml: float = 250.0):
        self.engine.refill_vaporizer(level_ml)

    # Bag-mask and the ventilator are mutually exclusive.
    def _cmd_bag_mask(self, active: bool):
        assisted = assisted_positive_pressure(self.engine)
        if active and self.engine.vent.is_on:
            self.engine.set_vent_power(False)
        self.engine.set_bag_mask_ventilation(active)
        return self._assistance_result(assisted)

    def _cmd_bag_settings(self, rr: float, tv: float, p_max: float | None = None):
        assisted = assisted_positive_pressure(self.engine)
        self.engine.set_bag_mask_settings(rr, domain.finite_number("bag tidal volume", tv) / 1000.0, p_max=p_max)
        return self._assistance_result(assisted)

    def _cmd_apl(self, cmh2o: float):
        assisted = assisted_positive_pressure(self.engine)
        self.engine.set_apl_valve(cmh2o)
        return self._assistance_result(assisted)

    def _cmd_bag_breath(self):
        self.engine.deliver_manual_breath()

    def _cmd_o2_flush(self):
        self.engine.trigger_o2_flush()

    def _cmd_vent_power(self, on: bool):
        assisted = assisted_positive_pressure(self.engine)
        if on and self.engine.bag_mask_active:
            self.engine.set_bag_mask_ventilation(False)
        self.engine.set_vent_power(on)
        return self._assistance_result(assisted)

    def _assistance_result(self, was_assisted: bool):
        """Check apnea when the last source of assisted ventilation stops."""
        if self.engine.airway_crisis.esophageal_intubation:
            return self._clinical_result(None)
        stopped = was_assisted and not assisted_positive_pressure(self.engine)
        return self._clinical_result(check_ventilation_withdrawal(self.engine) if stopped else None)

    def _cmd_vent(self, **fields):
        unknown = set(fields) - set(VENT_FIELDS)
        if unknown:
            raise ValueError(f"Unknown ventilator setting(s): {', '.join(sorted(unknown))}")
        if "ie" in fields and fields["ie"] not in IE_RATIOS:
            raise ValueError(f"Unsupported I:E ratio {fields['ie']!r}")
        assisted = assisted_positive_pressure(self.engine)
        self.engine.vent.update_settings(**fields)
        return self._assistance_result(assisted)

    def _cmd_drug_rate(self, key: str, rate: float):
        spec = get_drug_spec(key)
        rate = domain.bounded_number("infusion rate", rate, 0.0, _max_display_rate(spec))
        previous = self.engine.get_drug_state(key)["rate"]
        self.engine.set_drug_rate(key, rate)
        applied = self.engine.get_drug_state(key)["rate"]
        feedback = check_drug_administration(self.engine, spec.key) if applied > previous else None
        text = f"{spec.generic_name} infusion {applied:g} {spec.rate_unit}" if applied else f"{spec.generic_name} infusion stopped"
        return self._clinical_result(feedback, self._medication_receipt(spec.key, text))

    def _cmd_drug_target(self, key: str, target: float | None):
        if not self.engine.config.tci_enabled:
            raise ValueError("TCI is disabled for this session")
        spec = get_drug_spec(key)
        if target is not None:
            upper = spec.tci_range[1] if spec.tci_range is not None else 1000.0
            target = domain.bounded_number("TCI target", target, 0.0, upper)
        previous = self.engine.get_drug_state(key)["target"]
        self.engine.set_drug_target(key, target)
        applied = self.engine.get_drug_state(key)
        feedback = check_drug_administration(self.engine, spec.key) if applied["target"] > previous else None
        text = (
            f"{spec.generic_name} {_target_label(spec).lower()} {applied['target']:g} {spec.tci_unit}"
            if applied["is_tci"] else f"{spec.generic_name} TCI stopped"
        )
        return self._clinical_result(feedback, self._medication_receipt(spec.key, text))

    def _cmd_drug_bolus(self, key: str, amount: float):
        spec = get_drug_spec(key)
        amount = domain.bounded_number("bolus amount", amount, 0.0, spec.max_bolus)
        if amount <= 0:
            raise ValueError("Bolus amount must be positive and finite")
        feedback = check_drug_administration(self.engine, spec.key)
        self.engine.give_drug_bolus(key, amount)
        return self._clinical_result(
            feedback, self._medication_receipt(spec.key, f"{spec.generic_name} {amount:g} {spec.bolus_unit} given"),
        )

    def _cmd_sugammadex(self, mg_per_kg: float):
        mg_per_kg = domain.bounded_number("sugammadex dose", mg_per_kg, 0.0, 32.0, unit="mg/kg")
        if mg_per_kg <= 0:
            raise ValueError("Sugammadex dose must be positive and finite")
        amount = mg_per_kg * self.engine.patient.weight
        self.engine.give_drug_bolus("sugammadex", amount)
        return self._clinical_result(
            None, self._medication_receipt("sugammadex", f"Sugammadex {amount:.0f} mg given ({mg_per_kg:g} mg/kg)"),
        )

    def _cmd_calcium_chloride(self, mg: float):
        dose = domain.bounded_number("calcium chloride dose", mg, 0.0, 10000.0, unit="mg")
        if dose <= 0:
            raise ValueError("Calcium chloride dose must be positive and finite")
        self.engine.give_calcium_chloride(dose)
        return self._clinical_result(
            None, self._medication_receipt("calcium_chloride", f"Calcium chloride {dose:g} mg given"),
        )

    def _cmd_sodium_bicarbonate(self, meq: float):
        dose = domain.bounded_number("sodium bicarbonate dose", meq, 0.0, 500.0, unit="mEq")
        if dose <= 0:
            raise ValueError("Sodium bicarbonate dose must be positive and finite")
        feedback = check_drug_administration(self.engine, "sodium_bicarbonate")
        self.engine.give_sodium_bicarbonate(dose)
        return self._clinical_result(
            feedback,
            self._medication_receipt("sodium_bicarbonate", f"Sodium bicarbonate {dose:g} mEq given"),
        )

    def _medication_receipt(self, key: str, text: str) -> dict:
        """Acknowledge a completed action and retain it across local page reloads."""
        self._medication_sequence += 1
        entry = {"id": self._medication_sequence, "time": self.engine.state.time, "key": key, "text": text}
        self._medication_history.append(entry)
        return {"medication": entry}

    def _cmd_fluid(self, kind: str, volume_ml: float):
        if kind not in FLUIDS:
            raise ValueError(f"Unknown fluid: {kind!r}")
        max_volumes = {"d50w": 500.0, "d10w": 2000.0, "lipid": 3000.0}
        max_volume = max_volumes.get(kind, 5000.0)
        volume = domain.bounded_number("fluid volume", volume_ml, 0.0, max_volume, unit="mL")
        if volume <= 0:
            raise ValueError("Fluid volume must be positive and finite")
        method, label = FLUIDS[kind]
        getattr(self.engine, method)(volume)
        if kind == "d50w":
            text = f"{volume:g} mL 50% dextrose (D50W, {volume * 0.5:g} g) started"
        elif kind == "d10w":
            text = f"{volume:g} mL 10% dextrose (D10W, {volume * 0.1:g} g) started"
        else:
            text = f"{volume:g} mL {label} started"
        receipt = {"time": self.engine.state.time, "text": text}
        if kind in ("lipid", "d50w", "d10w"):
            return self._clinical_result(None, {"fluid": receipt, **self._medication_receipt(kind, text)})
        return {"fluid": receipt}

    def _cmd_lipid_rate(self, ml_min: float):
        self.engine.set_lipid_infusion(ml_min)
        applied = self.engine.lipid_infusion_ml_min
        text = f"20% lipid infusion {applied:g} mL/min" if applied > 0 else "20% lipid infusion stopped"
        return self._medication_receipt("lipid", text)

    def _cmd_compressions(self, active: bool):
        self.engine.set_chest_compressions(bool(active))

    def _cmd_charge_defibrillator(self):
        self.engine.charge_defibrillator()

    def _cmd_disarm_defibrillator(self):
        self.engine.disarm_defibrillator()

    def _shock_receipt(self) -> dict | None:
        event = self.engine.defibrillator.last_shock
        if event is None:
            return None
        time, synchronized = event
        kind = "Synchronized shock" if synchronized else "Shock"
        return {"time": time, "text": f"{kind} {SHOCK_ENERGY_J:g} J delivered"}

    def _observe_shock(self) -> dict:
        event = self.engine.defibrillator.last_shock
        if event is None or event == self._observed_shock:
            return {}
        self._observed_shock = event
        return self._clinical_result(self.engine._last_shock_feedback, {"shock": self._shock_receipt()})

    def _cmd_shock(self, synchronized: bool):
        engine = self.engine
        result = engine.defibrillate(synchronized=bool(synchronized))
        if result == ShockResult.DELIVERED:
            return self._observe_shock()
        text = {
            ShockResult.NOT_READY: "Defibrillator is not charged; no shock delivered",
            ShockResult.NOT_DELIVERED: "No R wave detected; no shock delivered",
            ShockResult.PENDING: "Waiting for an R wave",
        }[result]
        return {"shock": {"time": engine.state.time, "text": text}}

    def _cmd_maintenance_fluid(self, ml_hr: float):
        self.engine.set_continuous_fluid_rate(
            domain.bounded_number("maintenance fluid rate", ml_hr, 0.0, 5000.0, unit="mL/hr")
        )

    def _cmd_disturbance_profile(self, profile: str | None):
        if profile is None:
            self.engine.stop_disturbance(clear_profile=True)
        else:
            self.engine.set_disturbance_profile(profile)

    def _cmd_disturbance(self, active: bool):
        profile = self.engine.disturbance_profile
        if active and profile:
            self.engine.start_disturbance(profile)
        else:
            self.engine.stop_disturbance()

    def _cmd_obstruction(self, percent: float):
        self.engine.set_airway_obstruction(domain.bounded_number("airway obstruction", percent, 0.0, 100.0, unit="%") / 100.0)

    def _cmd_bronchospasm(self, percent: float):
        self.engine.set_bronchospasm(domain.bounded_number("bronchospasm", percent, 0.0, 100.0, unit="%") / 100.0)

    def _cmd_ards_shunt(self, percent: float):
        self.engine.set_ards_shunt(domain.bounded_number("ARDS shunt", percent, 0.0, 90.0, unit="%") / 100.0)

    def _cmd_extracorporeal_start(self, mode: str, **kwargs):
        common = {"blood_flow", "sweep_gas", "sweep_fio2", "fio2_percent", "membrane_type", "prime_volume_ml", "heater_cooler_on", "water_target_c"}
        mode_specs = {
            "vv_ecmo": ("VV ECMO", common | {"cannulation"}),
            "va_ecmo": ("VA ECMO", common | {"cannulation", "lv_vent", "bypass_alarm_mode"}),
            "cpb": (
                "CPB",
                common | {"sweep_fco2", "sweep_sevo", "initial_reservoir_ml", "blood_gas_strategy", "venous_clamp", "bypass_alarm_mode"},
            ),
        }
        if mode not in mode_specs:
            raise ValueError(f"Unknown extracorporeal mode: {mode!r}")
        label, allowed = mode_specs[mode]
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unknown {label} setting(s): {', '.join(sorted(unknown))}")
        rename = {
            "blood_flow": "blood_flow_l_min",
            "sweep_gas": "sweep_gas_l_min",
            "lv_vent": "lv_vent_l_min",
            "sweep_sevo": "sweep_sevo_percent",
        }
        opts: dict = {}
        for key, value in kwargs.items():
            if key == "venous_clamp":
                opts["venous_clamp_open"] = domain.bounded_number("venous clamp", value, 0.0, 100.0, unit="%") / 100.0
            elif key == "fio2_percent":
                opts["sweep_fio2"] = domain.bounded_number("sweep FiO2", value, 21.0, 100.0, unit="%") / 100.0
            else:
                opts[rename.get(key, key)] = value
        feedback = check_extracorporeal_start(self.engine, mode) if mode == "cpb" else None
        if mode == "vv_ecmo":
            self.engine.start_vv_ecmo(**opts)
        elif mode == "va_ecmo":
            self.engine.start_va_ecmo(**opts)
        else:
            self.engine.start_cpb(**opts)
        return self._clinical_result(feedback)

    def _cmd_extracorporeal_stop(self, reinfuse: bool = False, reinfuse_reservoir: bool | None = None):
        reinfuse_flag = bool(reinfuse)
        if reinfuse_reservoir is None:
            reinfuse_res = (not reinfuse_flag) and self.engine.extracorporeal.is_cpb
        else:
            reinfuse_res = bool(reinfuse_reservoir)
        self.engine.stop_extracorporeal(reinfuse=reinfuse_flag, reinfuse_reservoir=reinfuse_res)

    def _cmd_extracorporeal_settings(self, **fields):
        allowed = {
            "blood_flow", "sweep_gas", "sweep_fio2", "fio2_percent", "sweep_fco2", "sweep_sevo",
            "heater_cooler_on", "water_target_c", "venous_clamp", "cross_clamp", "cardioplegia",
            "lv_vent", "blood_gas_strategy", "bypass_alarm_mode",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown extracorporeal setting(s): {', '.join(sorted(unknown))}")
        if "blood_flow" in fields:
            self.engine.set_ecmo_flow(fields["blood_flow"])
        if "sweep_gas" in fields:
            self.engine.set_ecmo_sweep(fields["sweep_gas"])
        if "sweep_fio2" in fields:
            self.engine.set_ecmo_fio2(fields["sweep_fio2"])
        if "fio2_percent" in fields:
            self.engine.set_ecmo_fio2(
                domain.bounded_number("sweep FiO2", fields["fio2_percent"], 21.0, 100.0, unit="%") / 100.0
            )
        if "sweep_fco2" in fields:
            self.engine.set_ecmo_fco2(fields["sweep_fco2"])
        if "sweep_sevo" in fields:
            self.engine.set_ecmo_sevo(fields["sweep_sevo"])
        if "heater_cooler_on" in fields or "water_target_c" in fields:
            on = bool(fields.get("heater_cooler_on", self.engine.extracorporeal.state.heater_cooler_on))
            target = fields.get("water_target_c")
            self.engine.set_ecmo_heater_cooler(on, target)
        if "venous_clamp" in fields:
            self.engine.set_cpb_venous_clamp(
                domain.bounded_number("venous clamp", fields["venous_clamp"], 0.0, 100.0, unit="%") / 100.0
            )
        if "cross_clamp" in fields:
            self.engine.set_cpb_cross_clamp(bool(fields["cross_clamp"]))
        if "cardioplegia" in fields:
            self.engine.set_cpb_cardioplegia(bool(fields["cardioplegia"]))
        if "lv_vent" in fields:
            self.engine.set_ecmo_lv_vent(fields["lv_vent"])
        if "blood_gas_strategy" in fields:
            self.engine.set_cpb_blood_gas_strategy(fields["blood_gas_strategy"])
        if "bypass_alarm_mode" in fields:
            self.engine.set_bypass_alarm_mode(bool(fields["bypass_alarm_mode"]))

    def _cmd_cpb_reservoir_fluid(self, volume_ml: float, kind: str = "crystalloid"):
        if kind not in {"crystalloid", "saline", "albumin", "colloid", "blood"}:
            raise ValueError(f"Unknown reservoir fluid: {kind!r}")
        added = self.engine.add_cpb_reservoir_fluid(volume_ml, kind=kind)
        labels = {
            "crystalloid": "lactated Ringer's",
            "saline": "0.9% saline",
            "albumin": "5% albumin",
            "colloid": "5% albumin",
            "blood": "PRBC",
        }
        text = (
            f"{added:g} mL {labels[kind]} added to CPB reservoir"
            if added > 0.0
            else "CPB reservoir is full"
        )
        return {"fluid": {"time": self.engine.state.time, "text": text}}

    def _cmd_position(self, position: str):
        self.engine.set_position(position)

    def _cmd_auto_laryngospasm(self, enabled: bool):
        self.engine.set_auto_laryngospasm(enabled)

    def _cmd_hemorrhage(self, active: bool, rate_ml_min: float = 500.0):
        if active:
            self.engine.start_hemorrhage(
                domain.bounded_number("hemorrhage rate", rate_ml_min, 0.0, 5000.0, unit="mL/min")
            )
        else:
            self.engine.stop_hemorrhage()

    def _cmd_anaphylaxis(self, active: bool):
        (self.engine.start_anaphylaxis if active else self.engine.stop_anaphylaxis)()

    def _cmd_sepsis(self, active: bool):
        (self.engine.start_sepsis if active else self.engine.stop_sepsis)()

    def _cmd_malignant_hyperthermia(self, active: bool):
        (self.engine.start_malignant_hyperthermia if active else self.engine.stop_malignant_hyperthermia)()

    def _cmd_protamine_reaction(self, active: bool):
        (self.engine.start_protamine_reaction if active else self.engine.stop_protamine_reaction)()

    def _cmd_difficult_airway(self, profile: str = "none"):
        self.engine.set_difficult_airway(profile)

    def _cmd_esophageal_intubation(self, active: bool = True, armed: bool | None = None):
        if armed is not None:
            self.engine.arm_esophageal_intubation(bool(armed))
        else:
            self.engine.set_esophageal_intubation(bool(active))

    def _cmd_pneumothorax(self, active: bool, severity: float = 1.0):
        if active:
            self.engine.start_pneumothorax(severity=severity)
        else:
            self.engine.stop_pneumothorax()

    def _cmd_pneumoperitoneum(self, active: bool, iap_mmhg: float = 14.0):
        if active:
            self.engine.start_pneumoperitoneum(iap_mmhg=iap_mmhg)
        else:
            self.engine.stop_pneumoperitoneum()

    def _cmd_pulmonary_embolism(self, active: bool, obstruction: float = 0.60):
        if active:
            self.engine.start_pulmonary_embolism(obstruction=obstruction)
        else:
            self.engine.stop_pulmonary_embolism()

    def _cmd_venous_air_embolism(self, active: bool, rate_ml_min: float = 55.0):
        if active:
            self.engine.start_venous_air_embolism(rate_ml_min=rate_ml_min)
        else:
            self.engine.stop_venous_air_embolism()

    def _cmd_high_spinal(self, active: bool = True):
        if active:
            self.engine.trigger_high_spinal()
        else:
            self.engine.stop_neuraxial()

    def _cmd_surface_cooling(self, on: bool):
        self.engine.set_surface_cooling(on)

    def _cmd_stop_events(self):
        self.engine.stop_events()

    def _cmd_rhythm(self, name: str):
        self.engine.set_rhythm(name)

    def _cmd_bair_hugger(self, target_c: float):
        self.engine.set_bair_hugger(target_c)

    def _cmd_fluid_warmer(self, enabled: bool):
        self.engine.set_fluid_warmer(enabled)

    def _cmd_scenario_next(self):
        if self.scenario is None or self.case_run is None or self.case_run.mode != "guided":
            raise ValueError("Step progression is available only in guided cases")
        if self.step_index >= len(self.scenario):
            return
        self._update_scenario()
        if self.step_met:
            self.step_index += 1
            if self.step_index == len(self.scenario):
                self._finish_case("completed")
            else:
                self._begin_step()

    def _cmd_case_action(self, action: str):
        if self.case_run is None:
            raise ValueError("No case is active")
        self.case_run.act(action)

    def _cmd_case_hint(self):
        if self.case_run is None or self.case_run.mode != "independent":
            raise ValueError("Hints are available only in independent practice")
        return {"hint": self.case_run.request_hint()}

    def _cmd_case_finish(self):
        if self.case_run is None:
            raise ValueError("No case is active")
        return {"debrief": self._finish_case("learner_finished")}

    def _finish_case(self, reason: str) -> dict:
        assert self.case_run is not None
        report = self.case_run.finish(reason)
        self._pause()
        self.ended = True
        download = self._cmd_record(False)
        if download is not None:
            self._download = download
        return report

    def _cmd_csht(self):
        """Estimated context-sensitive half-times (min) for propofol, remifentanil, and fentanyl."""
        return {key: _num(self.engine.get_predicted_csht(key), 1) for key in ("propofol", "remi", "fentanyl")}

    def _cmd_nibp(self):
        self.engine.measure_nibp()

    def _cmd_blood_gas(self, site: str = "arterial"):
        return {"blood_gas": _blood_gas(self.engine.draw_blood_gas(site=site))}

    def _cmd_act(self):
        return {"act": _act(self.engine.measure_act())}

    def _cmd_glucose(self):
        return {"glucose": _glucose(self.engine.measure_glucose())}

    def _cmd_nmb_mode(self, mode: str):
        self.engine.nmb.set_mode(mode, self.engine.state.time)

    def _cmd_nmb_repeat(self, seconds: float):
        self.engine.nmb.set_repeat(seconds, self.engine.state.time)

    def _cmd_nmb_measure(self):
        self.engine.nmb.measure(self.engine.state.time)

    def _cmd_record(self, active: bool):
        """Start recording, or stop it and return {"filename", "csv"} to download, if any."""
        engine = self.engine
        if active:
            try:
                engine.start_recording(output_dir=self._recordings_dir)
            except RecordingError:
                self._take_recording(engine.recorder)
                raise
            return None
        recorder = engine.recorder
        if recorder is None or not recorder.is_recording:
            return None
        try:
            engine.stop_recording()
        except RecordingError as error:
            self.notice = f"{error}. The CSV may be incomplete."
        return self._take_recording(recorder)

    def _take_recording(self, recorder) -> dict[str, str] | None:
        """Move a finished recording into a download; retained recordings stay on disk."""
        if self.retain_recordings:
            return None
        try:
            with open(recorder.file_path, encoding="utf-8") as file:
                text = file.read()
            os.remove(recorder.file_path)
        except OSError:
            return None
        return {"filename": recorder.filename, "csv": text}

    def _recording(self) -> bool:
        recorder = self.engine.recorder
        return bool(recorder and recorder.is_recording)
