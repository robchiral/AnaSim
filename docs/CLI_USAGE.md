# CLI and Python usage

For installation and the browser interface, see the [README](../README.md).

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--help`, `-h` | Show command-line help and exit | `false` |
| `--mode` | Run mode, `ui` or `headless` | `ui` |
| `--port` | Local UI port, 0 to 65535; `0` chooses a free port | `0` |
| `--no-browser` | Print the local URL without opening it | `false` |
| `--duration` | Headless duration in simulated seconds | `10.0` |
| `--config` | Headless JSON configuration path | No file |
| `--record` | Write CSV in headless mode | `false` |
| `--record-dir` | CSV directory for headless runs and the local UI | `recordings` |
| `--record-interval` | Headless CSV interval in simulated seconds; `0` records every step | `1.0` |

## Configuration file

Headless runs accept a flat JSON object. Omitted fields use defaults; unknown
fields are rejected. The CLI argument `--mode` selects the interface, and the
JSON field `mode` selects the patient's initial state.

Save this configuration as `patient_config.json` for balanced maintenance.

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
| `baseline_temp` | 37 | Core temperature, 25 to 42 °C |
| `baseline_hb` | 13.5 | Hemoglobin, 6 to 20 g/dL |
| `baseline_hct` | `null` | Hematocrit, 0.18 to 0.60 as a fraction; `null` derives it from hemoglobin |
| `baseline_hr` | 70 | Heart rate, at least 10 bpm |
| `baseline_map` | 90 | Mean arterial pressure, at least 20 mmHg |
| `baseline_rr` | 12 | Respiratory rate, at least 0 breaths/min |
| `baseline_vt` | 500 | Tidal volume, at least 50 mL |
| `renal_function` | 1.0 | Renal-function factor, 0.4 to 1.0; `1.0` is normal |
| `hepatic_function` | 1.0 | Hepatic-function factor, 0.5 to 1.0; `1.0` is normal |

If you set `baseline_hct`, it must be within 0.12 of `0.03 * baseline_hb`.
Patient ranges are inclusive. Patient inputs and `dt` must be finite.
See [patient limits and source cohorts](ARCHITECTURE.md#supported-patient-domain).

#### Initialization and runtime

| Field | Default | Meaning |
|-------|---------|---------|
| `mode` | `"awake"` | `"awake"` or `"steady_state"`; see [initialization](ARCHITECTURE.md#initialization) |
| `maint_type` | `"tiva"` | Total intravenous anesthesia (`"tiva"`) or `"balanced"` maintenance for steady-state initialization |
| `concomitant_opioids` | `true` | Fixed opioid-regimen covariate for Eleveld propofol PK; maintenance initialization requires `true` |
| `tci_enabled` | `false` | Enable target-controlled infusion (TCI) and use it at maintenance startup |
| `dt` | 0.01 | Positive engine step in seconds |
| `rng_seed` | `null` | Integer for repeatable runs; `null` for a new random sequence |
| `arterial_line_enabled` | `true` | Display continuous arterial pressure |
| `end_on_cardiac_arrest` | `false` | End at modeled [cardiac arrest](ARCHITECTURE.md#hemodynamics) |
| `maintenance_fluid_ml_hr` | `null` | Maintenance fluid rate in mL/hr; `null` uses 1 mL/kg/hr |
| `sevoflurane_enabled` | `true` | Enable the sevoflurane vaporizer; `"balanced"` requires `true` |
| `disturbance_profile` | `null` | `"stim_intubation_pulse"`, `"stim_sustained_surgery"`, or `null` |

## Headless example

```bash
anasim --mode headless --duration 60 --config patient_config.json \
    --record --record-dir results --record-interval 0.1
```

## Recordings

`anasim_log_<timestamp>.csv` contains sampled
[`SimulationState`](../anasim/core/state.py) values. `time` is elapsed simulation
time in seconds. The default interval is 1 second, and each engine step records
at most one row. An interval shorter than `dt` still produces one row per step.

`pa_co2` is arterial CO₂. `display_etco2` is the end-tidal CO₂ measurement with
sensor effects; `etco2_signal_valid` reports its validity. `NaN` in `paw_plat`
means plateau pressure is unavailable.
See [field groups](ARCHITECTURE.md#simulation-state) and
[units](../anasim/core/state.py).

`engine.output_buffer` holds 20 seconds of monitor waveform samples. To
reproduce a run, save the configuration, AnaSim version, random seed, `dt`,
and intervention times.

## Python use

This example records three 20-second phases of anesthetized maintenance,
with bronchospasm introduced in the second phase and relieved in the third.

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

`set_vent_settings(vt=...)` takes liters; `Patient.baseline_vt` takes mL.
Use `set_vent_power(True)` to start the ventilator after applying its settings.
`set_vaporizer(percent)` sets the sevoflurane dial as a percentage.
`set_drug_rate(key, rate)` uses the [registry's units](../anasim/core/drug_registry.py).
Propofol and remifentanil rates are in mcg/kg/min using total body weight;
vasopressor rates are absolute. `enable_tci()` requires `tci_enabled=True`.
Read results through `engine.state` or `get_latest_state()`. Apply controls
through engine methods.
