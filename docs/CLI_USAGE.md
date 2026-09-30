# CLI usage

For installation and the browser interface, see the [README](../README.md).

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Run mode: `ui` or `headless` | `ui` |
| `--port` | Local UI port; `0` chooses a free port | `0` |
| `--no-browser` | Print the local URL without opening it | `false` |
| `--duration` | Headless run time in seconds | `10.0` |
| `--config` | JSON configuration path | None |
| `--record` | Write a CSV recording in headless mode | `false` |
| `--record-dir` | CSV directory for headless runs and the local interface | `recordings` |
| `--record-interval` | Sample interval in seconds for CSV | `1.0` |

## Configuration file

A JSON file sets patient, model, initialization, and runtime fields. Omitted
fields use their defaults.

### Minimal example

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

| Group | Fields |
|-------|--------|
| Patient | `age`, `weight`, `height`, `sex`, `asa` |
| Baseline physiology | `baseline_temp`, `baseline_hb`, `baseline_hct`, `baseline_hr`, `baseline_map`, `baseline_rr`, `baseline_vt` |
| Organ function | `renal_function`, `hepatic_function` |
| Initialization | `mode`: `awake` or `steady_state`; `maint_type`: `tiva` or `balanced` |
| Runtime | `dt`, `rng_seed`, `simulation_speed`, `end_on_cardiac_arrest`, `arterial_line_enabled`, `maintenance_fluid_ml_hr` |
| Events | `disturbance_profile`: `stim_intubation_pulse`, `stim_sustained_surgery`, or `null` |
| Volatile agents | `volatile_agents`: `["sevoflurane"]` or `[]` |

`"baseline_hct": null` derives hematocrit from hemoglobin, and
`"maintenance_fluid_ml_hr": null` uses 1 mL/kg/hr. Steady-state mode runs a
hidden maintenance period before visible time starts.

Patient values must be within these limits:

| Field | Range |
|-------|-------|
| `age` | 18 to 70 years |
| `weight` | 50 to 100 kg |
| `height` | 150 to 200 cm, with BMI 18 to 32 kg/m² |
| `baseline_hb` | 6 to 20 g/dL |
| `baseline_hct` | 0.18 to 0.60 |
| `renal_function` | 0.4 to 1.0 (dimensionless) |
| `hepatic_function` | 0.5 to 1.0 (dimensionless) |

Model fields accept these values:

| Field | Values |
|-------|--------|
| `pk_model_propofol` | `Marsh`, `Schnider`, `Eleveld` |
| `pk_model_remi` | `Minto` |
| `bis_model` | `Bouillon`, `Eleveld`, `Fuentes`, `Yumuk` |
| `hemo_model` | `Su2023` |
| `resp_model` | `SingleCompartment` |
| `pk_model_nore` | `Li`, `Beloeil` |
| `pk_model_epi` | `HealthyAdult`, `Abboud` |
| `loc_model` | `Kern`, `Mertens`, `Johnson` |

AnaSim reports unknown keys, invalid model names, out-of-range values, and
non-finite numbers as configuration errors.

## Examples

Run a 60-second headless simulation with a custom patient:

```bash
anasim --mode headless --duration 60 --config patient_config.json
```
