# CLI and Python usage

For installation and the browser interface, see the [README](../README.md).

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Run mode, `ui` or `headless` | `ui` |
| `--port` | Local UI port; `0` chooses a free port | `0` |
| `--no-browser` | Print the local URL without opening it | `false` |
| `--duration` | Headless duration in simulated seconds | `10.0` |
| `--config` | Headless JSON configuration path | None |
| `--record` | Write CSV in headless mode | `false` |
| `--record-dir` | CSV directory for headless runs and the local UI | `recordings` |
| `--record-interval` | Headless CSV interval in simulated seconds; `0` records every step | `1.0` |

## Configuration file

Use a flat JSON object; omitted fields use defaults. CLI `--mode` chooses
the interface; JSON `mode` chooses the patient's initial state.

### Minimal example

Save as `patient_config.json`.

```json
{
    "age": 40,
    "weight": 70,
    "rng_seed": 123,
    "mode": "steady_state",
    "maint_type": "balanced"
}
```

### Fields

#### Patient inputs

| Field | Default | Units or accepted values |
|-------|---------|--------------------------|
| `age` | 40 | 18 to 70 years |
| `weight` | 70 | 50 to 100 kg |
| `height` | 170 | 150 to 200 cm; BMI 18 to 32 kg/m² |
| `sex` | `"male"` | `"male"` or `"female"` |
| `asa` | 1 | Integer 1 to 5 |
| `baseline_temp` | 37 | 25 to 42 °C |
| `baseline_hb` | 13.5 | 6 to 20 g/dL |
| `baseline_hct` | `null` | 0.18 to 0.60, as a fraction; `null` derives it from Hb |
| `baseline_hr` | 70 | At least 10 bpm |
| `baseline_map` | 90 | At least 20 mmHg |
| `baseline_rr` | 12 | At least 0 breaths/min |
| `baseline_vt` | 500 | At least 50 mL |
| `renal_function` | 1.0 | 0.4 to 1.0, as a fraction |
| `hepatic_function` | 1.0 | 0.5 to 1.0, as a fraction |

If you set `baseline_hct`, it must be within 0.12 of `0.03 * baseline_hb`.
See [patient limits and source cohorts](ARCHITECTURE.md#supported-patient-domain).

#### Initialization and runtime

| Field | Default | Meaning |
|-------|---------|---------|
| `mode` | `"awake"` | `"awake"` or `"steady_state"`; see [initialization](ARCHITECTURE.md#initialization) |
| `maint_type` | `"tiva"` | `"tiva"` or `"balanced"` for steady-state initialization |
| `tci_enabled` | `false` | Allow TCI controls and attach TCI controllers at maintenance startup |
| `dt` | 0.01 | Positive engine step in seconds |
| `rng_seed` | `null` | Integer for repeatable runs; `null` for a new random sequence |
| `simulation_speed` | 1.0 | Real-time multiplier for UI sessions |
| `arterial_line_enabled` | `true` | Display continuous arterial pressure |
| `end_on_cardiac_arrest` | `false` | End at modeled [cardiac arrest](ARCHITECTURE.md#hemodynamics) |
| `maintenance_fluid_ml_hr` | `null` | mL/hr; `null` uses 1 mL/kg/hr |
| `volatile_agents` | `["sevoflurane"]` | `["sevoflurane"]` or `[]` to disable the vaporizer |
| `disturbance_profile` | `null` | `"stim_intubation_pulse"`, `"stim_sustained_surgery"`, or `null` |

#### Models

| Field | Default | Accepted values |
|-------|---------|-----------------|
| `pk_model_propofol` | `"Eleveld"` | `"Marsh"`, `"Schnider"`, `"Eleveld"` |
| `pk_model_remi` | `"Minto"` | `"Minto"` |
| `bis_model` | `"Bouillon"` | `"Bouillon"`, `"Eleveld"`, `"Fuentes"`, `"Yumuk"` |
| `hemo_model` | `"Su2023"` | `"Su2023"` |
| `resp_model` | `"SingleCompartment"` | `"SingleCompartment"` |
| `pk_model_nore` | `"Li"` | `"Li"`, `"Beloeil"` |
| `pk_model_epi` | `"HealthyAdult"` | `"HealthyAdult"`, `"Abboud"` |
| `loc_model` | `"Kern"` | `"Kern"`, `"Mertens"`, `"Johnson"` |

Patient inputs and `dt` must be finite.

## Headless example

```bash
anasim --mode headless --duration 60 --config patient_config.json \
    --record --record-dir results --record-interval 0.1
```

## Recordings

`anasim_log_<timestamp>.csv` contains one [`SimulationState`](../anasim/core/state.py)
row per sampled step. `time` is elapsed simulation time in seconds; the default
recording interval is 1 s. Each engine step records at most one row.

`pa_co2` is arterial CO2; `display_etco2` includes sensor effects and has an
`etco2_signal_valid` flag. `NaN` in `paw_plat` means an unavailable measurement.
See [field groups](ARCHITECTURE.md#simulation-state) and
[units](../anasim/core/state.py).

Per-step monitor waveforms are in `engine.output_buffer` (last 20 s).
Save the configuration, AnaSim version, seed, `dt`, and intervention times
for replay.

## Python use

This example records a run while introducing and relieving bronchospasm
in 20-second phases.

```python
from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient

config = SimulationConfig(mode="steady_state", dt=0.01, rng_seed=123)
engine = SimulationEngine(Patient(), config)
engine.start_recording(output_dir="results", sample_interval_sec=0.1)
engine.start()
try:
    for severity in (0.0, 0.7, 0.0):
        engine.set_bronchospasm(severity)
        for _ in range(2000):
            engine.step(config.dt)
finally:
    engine.stop()
    engine.stop_recording()

state = engine.get_latest_state()
print(state.time, state.pa_co2, state.display_etco2)
```

`set_vent_settings(vt=...)` takes liters. `Patient.baseline_vt` takes mL.
`set_drug_rate(key, rate)` uses the [registry's units](../anasim/core/drug_registry.py).
Propofol and remifentanil rates are in mcg/kg/min using the patient's weight;
vasopressor rates are absolute. `enable_tci()` requires `tci_enabled=True`.
Read `engine.state` or `get_latest_state()`; apply changes through engine methods.
