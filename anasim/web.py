"""One simulation session for the browser interface, with JSON commands and snapshots.

The Pyodide worker and the local server (`anasim.local`) both use `WebSession`.
"""

import json
import math
import os
import tempfile
from collections import deque

from anasim import __version__
from anasim.core.drug_registry import get_drug_spec
from anasim.core.engine import SimulationEngine
from anasim.core.enums import RhythmType
from anasim.core.recorder import RecordingError
from anasim.core.state import SUPPORTED_MODEL_OPTIONS, AirwayType, SimulationConfig
from anasim.machine.ventilator import MODES
from anasim.patient import domain
from anasim.patient.patient import Patient
from anasim.physiology.disturbances import list_disturbance_profiles
from anasim.scenarios import SCENARIO_REGISTRY

PATIENT_FIELDS = (
    "age",
    "weight",
    "height",
    "sex",
    "baseline_hb",
    "renal_function",
    "hepatic_function",
)
CONFIG_FIELDS = (
    "pk_model_propofol",
    "pk_model_nore",
    "pk_model_epi",
    "bis_model",
    "loc_model",
    "mode",
    "maint_type",
    "end_on_cardiac_arrest",
    "arterial_line_enabled",
)
MODEL_FIELDS = ("pk_model_propofol", "pk_model_nore", "pk_model_epi", "bis_model", "loc_model")

SPEED_RANGE = (0.1, 50.0)
MAX_REAL_DT_S = 0.2
# Averaging window for the achieved speed, so one slow frame does not flag lag.
SPEED_AVERAGE_S = 2.0
WAVE_WINDOW_S = 20.0  # The slower respiratory sweep
VENT_FIELDS = ("mode", "rr", "tv", "peep", "ie", "p_insp", "p_support", "t_insp", "pause", "p_max", "trigger")
IE_RATIOS = ("1:1", "1:2", "1:3", "1:4")
LOOP_INTERVAL_S = 0.03  # Spacing of loop points, about 170 per 5-second breath
FLUIDS = {"crystalloid": "give_fluid", "albumin": "give_albumin", "blood": "give_blood"}

SCENARIOS = {spec.id: spec for spec in SCENARIO_REGISTRY}


def catalog() -> str:
    """Return the choices the setup screen offers, as JSON."""
    defaults = SimulationConfig()
    return json.dumps({
        "version": __version__,
        "scenarios": [
            {"id": spec.id, "label": spec.label, "mode": spec.start_mode, "maint_type": spec.maint_type}
            for spec in SCENARIO_REGISTRY
        ],
        "models": {
            name: {"options": sorted(SUPPORTED_MODEL_OPTIONS[name]), "default": getattr(defaults, name)}
            for name in MODEL_FIELDS
        },
        "ranges": {
            "age": domain.AGE_RANGE_YEARS,
            "weight": domain.WEIGHT_RANGE_KG,
            "height": domain.HEIGHT_RANGE_CM,
            "baseline_hb": domain.HEMOGLOBIN_RANGE_G_DL,
        },
    })


def _num(value, digits=None):
    """Return a JSON-safe number; non-finite values become null."""
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits) if digits is not None else value


def _target_label(spec) -> str | None:
    if not spec.has_tci:
        return None
    site = spec.fixed_tci_mode.value.replace("_", " ") if spec.fixed_tci_mode else "effect site"
    return f"{site.capitalize()} target"


class WebSession:
    """One simulation session driven by the browser UI."""

    def __init__(self, params: dict, recordings_dir: str | None = None, *, retain_recordings: bool = False):
        params = dict(params)
        unknown = set(params) - set(PATIENT_FIELDS) - set(CONFIG_FIELDS) - {"scenario_id"}
        if unknown:
            raise ValueError(f"Unknown session setting(s): {', '.join(sorted(unknown))}")

        spec = None
        scenario_id = params.pop("scenario_id", None)
        if scenario_id:
            spec = SCENARIOS[scenario_id]
            params.update(mode=spec.start_mode, maint_type=spec.maint_type)

        patient = Patient(**{key: params[key] for key in PATIENT_FIELDS if key in params})
        config = SimulationConfig(**{key: params[key] for key in CONFIG_FIELDS if key in params})
        self.engine = SimulationEngine(patient, config)
        self.speed = 1.0
        self.achieved_speed = 1.0
        self.ended = False
        self.lagging = False
        self.notice = None
        self._download = None
        self._accumulator = 0.0
        self._last_wave_time = -math.inf
        # Loops trace the breath in progress over the last completed one.
        self._loop_sample = None
        self._loop_pending = []  # Flow crossing awaiting confirmation by the breath counter.
        self._loop_breath = None
        self._loop_start = 0.0
        self._loop_index = 0
        self._loop_current = []
        self._loop_previous = None
        self._loop_new = []
        self._loop_replay = False
        self._recordings_dir = recordings_dir or os.path.join(tempfile.gettempdir(), "anasim-recordings")
        self.retain_recordings = retain_recordings
        self._medication_history = deque(maxlen=50)
        self._medication_sequence = 0

        self.scenario = None
        self.step_index = 0
        self.step_met = False
        self.step_status = ""
        if spec is not None:
            self.scenario = spec.builder()
            self.scenario.prepare(self.engine)
            self._begin_step()

    def info(self) -> str:
        """Return session constants the UI needs once, as JSON."""
        engine = self.engine
        patient = engine.patient
        return json.dumps({
            "patient": {
                "age": patient.age,
                "sex": patient.sex,
                "weight": patient.weight,
                "renal_status": patient.renal_status,
                "hepatic_status": patient.hepatic_status,
            },
            "arterial_line": engine.config.arterial_line_enabled,
            "recordings_dir": self._recordings_dir if self.retain_recordings else None,
            "sample_interval": engine.config.dt,
            "speed_range": SPEED_RANGE,
            "drugs": [
                {
                    "key": spec.key,
                    "name": spec.name,
                    "rate_unit": spec.rate_unit,
                    "bolus_unit": spec.bolus_unit,
                    "default_bolus": spec.default_bolus,
                    "tci_unit": spec.tci_unit,
                    "tci_range": spec.tci_range,
                    "target_label": _target_label(spec),
                }
                for spec in engine.get_controllable_drugs()
            ],
            "medication_history": list(self._medication_history),
            "disturbances": [{"key": key, "label": label} for label, key in list_disturbance_profiles()],
            "rhythms": [rhythm.value for rhythm in RhythmType],
            "vent_modes": MODES,
            "ie_ratios": IE_RATIOS,
            "scenario": None if self.scenario is None else {
                "name": self.scenario.name,
                "description": self.scenario.description,
                "total": len(self.scenario),
            },
        })

    # --- Loop -----------------------------------------------------------

    def advance(self, real_dt: float) -> str:
        """Step by real_dt seconds of wall time at the current speed; return a snapshot."""
        self.step(real_dt)
        return json.dumps(self.snapshot())

    def step(self, real_dt: float) -> None:
        """Advance without consuming snapshots, for the native server clock."""
        engine = self.engine
        if engine.running:
            start_time = engine.state.time
            clamped = min(max(float(real_dt), 0.0), MAX_REAL_DT_S)
            self._accumulator += clamped * self.speed
            sim_step = engine.config.dt
            max_steps = max(100, math.ceil(MAX_REAL_DT_S * SPEED_RANGE[1] / sim_step))
            steps = 0
            while self._accumulator >= sim_step and steps < max_steps and not engine.state.cardiac_arrest:
                try:
                    engine.step(sim_step)
                except RecordingError as error:
                    self._pause()
                    self._download = self._take_recording(engine.recorder)
                    saved = "; the recorded part was downloaded" if self._download else ""
                    self.notice = f"{error}. The CSV is incomplete{saved}. The simulation is paused."
                    break
                self._accumulator -= sim_step
                steps += 1
            # Drop time the browser could not simulate instead of catching up later.
            if steps >= max_steps:
                self._accumulator = 0.0
            if real_dt > 0 and engine.running:
                weight = min(1.0, real_dt / SPEED_AVERAGE_S)
                rate = (engine.state.time - start_time) / real_dt
                self.achieved_speed += weight * (rate - self.achieved_speed)
                self.lagging = self.achieved_speed < 0.9 * self.speed
            if engine.state.cardiac_arrest:
                self.ended = True
                self._pause()

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

    def _vitals(self) -> dict:
        engine = self.engine
        s = engine.state
        # Spirometry needs gas through the circuit, Pplat a positive-pressure
        # breath, and end-tidal values an exhaled breath.
        connected = s.airway_mode != AirwayType.NONE
        assisted = connected and (engine.vent.is_on or engine.bag_mask_active)
        exhaled = s.etco2_signal_valid
        return {
            "hr": _num(s.display_hr, 1),
            "spo2": _num(s.display_spo2, 1) if s.spo2_signal_valid else None,
            "art": [_num(s.art_sbp, 1), _num(s.art_dbp, 1), _num(s.art_map, 1)],
            "nibp": None if s.nibp_timestamp is None else [
                _num(s.nibp_sys, 1), _num(s.nibp_dia, 1), _num(s.nibp_map, 1)
            ],
            "nibp_cuff": _num(s.nibp_cuff_pressure, 1) if s.nibp_is_cycling else None,
            "nibp_age": None if s.nibp_timestamp is None else _num(max(0.0, s.time - s.nibp_timestamp), 0),
            "nibp_failed": s.nibp_measurement_failed,
            "etco2": _num(s.display_etco2, 1) if s.etco2_signal_valid else None,
            "rr": _num(s.rr, 1),
            # Without an airway the monitor counts chest movement by impedance, not CO2.
            "rr_source": "co2" if connected else "impedance",
            "bis": _num(s.display_bis, 1),
            "tof": _num(s.tof, 1),
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
        samples = []
        for sample in reversed(buffer):
            if sample.time <= self._last_wave_time:
                break
            samples.append(sample)
            if len(samples) >= round(WAVE_WINDOW_S / self.engine.config.dt):
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

    def _collect_loop(self, samples) -> None:
        """Add points of the breath in progress; a new breath makes it the previous loop."""
        stride = max(1, round(LOOP_INTERVAL_S / self.engine.config.dt))
        for sample in samples:
            previous = self._loop_sample
            if previous is not None and sample.time <= previous.time:
                continue  # Replayed for the sweep
            self._loop_sample = sample
            if self._loop_breath is None:
                self._start_loop(sample)
                continue
            # The filtered flow crossing can precede or follow breath detection.
            # Retain it until the counter confirms a breath; a small cleft that
            # returns to expiration stays in the existing loop.
            if previous is not None and previous.flow <= 0 < sample.flow:
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
                if sample.breath != self._loop_breath:
                    boundary = self._loop_pending[0]._replace(breath=sample.breath)
                    self._start_loop(boundary)
                    pending = self._loop_pending[1:]
                elif sample.flow <= 0:
                    pending = self._loop_pending
                else:
                    continue
                for point in pending:
                    self._sample_loop(point, stride)
                self._loop_pending = []
            elif sample.breath != self._loop_breath and (previous.flow >= 0 or sample.flow >= 0):
                self._start_loop(sample)
            else:
                self._sample_loop(sample, stride)
        # Apnea leaves one breath open; keep a sweep of it.
        limit = round(WAVE_WINDOW_S / LOOP_INTERVAL_S)
        if len(self._loop_current) > limit:
            self._loop_current = self._loop_current[-limit:]

    def _start_loop(self, boundary) -> None:
        if len(self._loop_current) > 1:
            # Keep real changes in end-expiratory volume rather than forcing closure.
            self._append_loop_point(boundary.paw, boundary.flow, boundary.volume)
            self._loop_previous = (self._loop_breath, self._loop_current)
        self._loop_breath, self._loop_start = boundary.breath, boundary.volume
        self._loop_index, self._loop_current = 0, []
        self._append_loop_point(boundary.paw, boundary.flow, boundary.volume)

    def _sample_loop(self, sample, stride) -> None:
        if self._loop_index % stride == 0:
            self._append_loop_point(sample.paw, sample.flow, sample.volume)
        self._loop_index += 1

    def _append_loop_point(self, paw, flow, volume) -> None:
        point = (_num(paw, 2), _num(flow, 1), _num((volume - self._loop_start) * 1000.0, 1))
        self._loop_current.append(point)
        self._loop_new.append((self._loop_breath, point))

    def _take_loop(self) -> list:
        """Return new loop points as runs of (breath, paw, flow, volume); a replay resends both loops."""
        if self._loop_replay:
            self._loop_replay = False
            runs = [self._loop_previous] if self._loop_previous else []
            if self._loop_current:
                runs.append((self._loop_breath, self._loop_current))
            points = [(breath, point) for breath, loop in runs for point in loop]
        else:
            points = self._loop_new
        self._loop_new = []
        out = []
        for breath, (paw, flow, volume) in points:
            if not out or out[-1]["breath"] != breath:
                out.append({"breath": breath, "paw": [], "flow": [], "volume": []})
            run = out[-1]
            run["paw"].append(paw)
            run["flow"].append(flow)
            run["volume"].append(volume)
        return out

    def _controls(self) -> dict:
        engine = self.engine
        circuit = engine.circuit
        vent = engine.vent.settings
        return {
            "airway": engine.state.airway_mode.value,
            "fgf": {"o2": circuit.fgf_o2, "air": circuit.fgf_air, "n2o": circuit.fgf_n2o},
            "o2_connected": circuit.oxygen_supply_connected,
            "circuit_fio2": _num(circuit.composition.fio2, 4),
            "vaporizer": circuit.vaporizer_setting,
            "bag_mask": engine.bag_mask_active,
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
            "drugs": {spec.key: engine.get_drug_state(spec.key) for spec in engine.get_controllable_drugs()},
            "maintenance_ml_hr": engine.get_continuous_fluid_rate(),
            "disturbance": {"profile": engine.disturbance_profile, "active": bool(
                engine.disturbance_active and engine.disturbance_profile
            )},
            "obstruction": engine.airway_obstruction_manual * 100.0,
            "bronchospasm": engine.bronchospasm_manual * 100.0,
            "laryngospasm": engine.laryngospasm_level,
            "auto_laryngospasm": engine.auto_laryngospasm_enabled,
            "hemorrhage": {"active": engine.active_hemorrhage, "rate": engine.hemorrhage_rate_ml_min},
            "anaphylaxis": engine.active_anaphylaxis,
            "sepsis": engine.active_sepsis,
            "rhythm": engine.hemo.rhythm_type.value,
            "bair_hugger": engine.state.bair_hugger_target,
        }

    # --- Scenario -------------------------------------------------------

    def _begin_step(self):
        """Scope action objectives to what the learner does from now on."""
        self.step_met = False
        self.step_status = ""
        if self.step_index < len(self.scenario):
            step = self.scenario[self.step_index]
            self.engine.actions.begin_step(step.id, self.engine.state.time)

    def _update_scenario(self):
        if self.scenario is None or self.step_index >= len(self.scenario):
            return
        met, status = self.scenario[self.step_index].check_requirements(self.engine)
        self.step_met = bool(met)
        self.step_status = status

    def _scenario_state(self):
        if self.scenario is None:
            return None
        if self.step_index >= len(self.scenario):
            return {"index": self.step_index, "complete": True}
        step = self.scenario[self.step_index]
        return {
            "index": self.step_index,
            "complete": False,
            "id": step.id,
            "title": step.title,
            "instruction": step.instruction,
            "target_tab": step.target_tab,
            "met": self.step_met,
            "status": "" if self.step_met else self.step_status,
        }

    # --- Commands -------------------------------------------------------

    def command(self, name: str, args_json: str = "{}") -> str:
        """Apply one learner control; return its JSON result (usually null)."""
        handler = getattr(self, f"_cmd_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown command: {name!r}")
        return json.dumps(handler(**json.loads(args_json or "{}")))

    def _cmd_run(self, running: bool):
        if running and not self.ended:
            self.engine.start()
            self._accumulator = 0.0
            self._reset_speed_average()
        else:
            self._pause()

    def _cmd_speed(self, value: float):
        self.speed = min(max(float(value), SPEED_RANGE[0]), SPEED_RANGE[1])
        self._reset_speed_average()

    def _cmd_airway(self, mode: str):
        if mode == "ETT" and self.engine.bag_mask_active:
            self.engine.set_bag_mask_ventilation(False)
        self.engine.set_airway_mode(mode)

    def _cmd_fgf(self, o2: float, air: float, n2o: float):
        self.engine.set_fgf(float(o2), float(air), float(n2o))

    def _cmd_oxygen_supply(self, connected: bool):
        self.engine.set_oxygen_supply_connected(connected)

    def _cmd_vaporizer(self, percent: float):
        self.engine.set_vaporizer(self.engine.active_agent, float(percent))

    def _cmd_bag_mask(self, active: bool):
        # Bag-mask and the ventilator are mutually exclusive.
        if active and self.engine.vent.is_on:
            self.engine.set_vent_power(False)
        if active:
            self.engine.set_bag_mask_ventilation(True, rr=12.0, vt=0.5)
        else:
            self.engine.set_bag_mask_ventilation(False)

    def _cmd_vent_power(self, on: bool):
        if on and self.engine.bag_mask_active:
            self.engine.set_bag_mask_ventilation(False)
        self.engine.set_vent_power(on)

    def _cmd_vent(self, **fields):
        unknown = set(fields) - set(VENT_FIELDS)
        if unknown:
            raise ValueError(f"Unknown ventilator setting(s): {', '.join(sorted(unknown))}")
        if "mode" in fields and fields["mode"] not in MODES:
            raise ValueError(f"Unsupported ventilator mode {fields['mode']!r}")
        if "ie" in fields and fields["ie"] not in IE_RATIOS:
            raise ValueError(f"Unsupported I:E ratio {fields['ie']!r}")
        current = self.engine.vent.settings
        settings = {name: getattr(current, name) for name in VENT_FIELDS} | fields
        numbers = {name: float(settings[name]) for name in VENT_FIELDS if name not in ("mode", "ie")}
        if not all(map(math.isfinite, numbers.values())):
            raise ValueError("Ventilator settings must be finite numbers")
        self.engine.set_vent_settings(
            numbers.pop("rr"),
            numbers.pop("tv") / 1000.0,
            numbers.pop("peep"),
            settings["ie"],
            mode=settings["mode"],
            **numbers,
        )

    def _cmd_drug_rate(self, key: str, rate: float):
        if not math.isfinite(float(rate)):
            raise ValueError("Infusion rate must be finite")
        self.engine.set_drug_rate(key, float(rate))
        spec = get_drug_spec(key)
        applied = self.engine.get_drug_state(key)["rate"]
        text = f"{spec.generic_name} infusion {applied:g} {spec.rate_unit}" if applied else f"{spec.generic_name} infusion stopped"
        return self._medication_receipt(spec.key, text)

    def _cmd_drug_target(self, key: str, target: float | None):
        if target is not None and not math.isfinite(float(target)):
            raise ValueError("TCI target must be finite")
        self.engine.set_drug_target(key, None if target is None else float(target))
        spec = get_drug_spec(key)
        applied = self.engine.get_drug_state(key)
        text = (
            f"{spec.generic_name} {_target_label(spec).lower()} {applied['target']:g} {spec.tci_unit}"
            if applied["is_tci"] else f"{spec.generic_name} TCI stopped"
        )
        return self._medication_receipt(spec.key, text)

    def _cmd_drug_bolus(self, key: str, amount: float):
        if not math.isfinite(float(amount)) or float(amount) <= 0:
            raise ValueError("Bolus amount must be positive and finite")
        spec = get_drug_spec(key)
        self.engine.give_drug_bolus(key, float(amount))
        return self._medication_receipt(spec.key, f"{spec.generic_name} {float(amount):g} {spec.bolus_unit} given")

    def _cmd_sugammadex(self, mg_per_kg: float):
        if not math.isfinite(float(mg_per_kg)) or float(mg_per_kg) <= 0:
            raise ValueError("Sugammadex dose must be positive and finite")
        amount = float(mg_per_kg) * self.engine.patient.weight
        self.engine.give_drug_bolus("sugammadex", amount)
        return self._medication_receipt("sugammadex", f"Sugammadex {amount:g} mg given ({float(mg_per_kg):g} mg/kg)")

    def _medication_receipt(self, key: str, text: str) -> dict:
        """Acknowledge a completed action and retain it across local page reloads."""
        self._medication_sequence += 1
        entry = {"id": self._medication_sequence, "time": self.engine.state.time, "key": key, "text": text}
        self._medication_history.append(entry)
        return {"medication": entry}

    def _cmd_fluid(self, kind: str, volume_ml: float):
        getattr(self.engine, FLUIDS[kind])(float(volume_ml))

    def _cmd_maintenance_fluid(self, ml_hr: float):
        self.engine.set_continuous_fluid_rate(float(ml_hr))

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
        self.engine.set_airway_obstruction(float(percent) / 100.0)

    def _cmd_bronchospasm(self, percent: float):
        self.engine.set_bronchospasm(float(percent) / 100.0)

    def _cmd_auto_laryngospasm(self, enabled: bool):
        self.engine.set_auto_laryngospasm(enabled)

    def _cmd_hemorrhage(self, active: bool, rate_ml_min: float = 500.0):
        if active:
            self.engine.start_hemorrhage(float(rate_ml_min))
        else:
            self.engine.stop_hemorrhage()

    def _cmd_anaphylaxis(self, active: bool):
        (self.engine.start_anaphylaxis if active else self.engine.stop_anaphylaxis)()

    def _cmd_sepsis(self, active: bool):
        (self.engine.start_sepsis if active else self.engine.stop_sepsis)()

    def _cmd_stop_events(self):
        self.engine.stop_events()

    def _cmd_rhythm(self, name: str):
        self.engine.set_rhythm(name)

    def _cmd_bair_hugger(self, target_c: float):
        self.engine.set_bair_hugger(float(target_c))

    def _cmd_scenario_next(self):
        if self.scenario is None or self.step_index >= len(self.scenario):
            return
        self._update_scenario()
        if self.step_met:
            self.step_index += 1
            self._begin_step()

    def _cmd_csht(self):
        """Estimated context-sensitive half-times (min) for propofol, remifentanil, and fentanyl."""
        return {key: _num(self.engine.get_predicted_csht(key), 1) for key in ("propofol", "remi", "fentanyl")}

    def _cmd_nibp(self):
        self.engine.measure_nibp()

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

    def _take_recording(self, recorder):
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
