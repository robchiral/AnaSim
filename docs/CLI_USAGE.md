# Advanced CLI usage

This document details advanced configuration options for the AnaSim command-line interface, particularly for headless mode. For basic installation and usage, see the [README](../README.md).

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Run mode: `ui` or `headless` | `ui` |
| `--duration` | Simulation duration in seconds (Headless only) | `10.0` |
| `--config` | Path to a JSON configuration file | None |
| `--record` | Enable CSV recording (Headless only) | `false` |
| `--record-dir` | Output directory for recordings | `recordings` |
| `--record-interval` | Sample interval in seconds for CSV | `1.0` |

## Configuration file

You can provide a JSON file to customize the patient and simulation parameters.

### Structure

```json
{
    "age": 40,
    "weight": 70,
    "height": 170,
    "sex": "male",
    "asa": 1,
    "baseline_hr": 70,
    "baseline_map": 90,
    "baseline_rr": 12,
    "baseline_vt": 500,
    "baseline_temp": 37.0,
    "baseline_hb": 13.5,
    "baseline_hct": null,
    "renal_function": 1.0,
    "hepatic_function": 1.0,
    "dt": 0.01,
    "mode": "awake",
    "maint_type": "tiva",
    "pk_model_propofol": "Eleveld",
    "pk_model_remi": "Minto",
    "bis_model": "Bouillon",
    "hemo_model": "Su2023",
    "resp_model": "SingleCompartment",
    "pk_model_nore": "Li",
    "pk_model_epi": "Clutter",
    "loc_model": "Kern",
    "disturbance_profile": null,
    "volatile_agents": ["sevoflurane"],
    "rng_seed": 123,
    "maintenance_fluid_ml_hr": null,
    "simulation_speed": 1.0,
    "enable_death_detector": false
}
```

### Options

- **age/weight/height/sex/asa**: patient demographics.
- **baseline_hr/baseline_map/baseline_rr/baseline_vt**: patient baseline vitals (bpm, mmHg, bpm, mL).
- **baseline_temp/baseline_hb/baseline_hct**: baseline temperature (°C), hemoglobin (g/dL), hematocrit (fraction). If `baseline_hct` is omitted or `null`, the engine derives it from hemoglobin.
- **renal_function/hepatic_function**: organ function fractions (0.1–1.0).
- **dt**: simulation time step (seconds).
- **mode**: `awake`, `steady_state`
- **maint_type**: `tiva`, `balanced`. `steady_state` uses an internal managed-maintenance bootstrap before visible time starts; it is not a pure compartment equilibrium preset.
- **pk_model_propofol**: `Marsh`, `Schnider`, `Eleveld`
- **pk_model_remi**: `Minto`
- **bis_model**: `Bouillon`, `Eleveld`, `Fuentes`, `Yumuk`
- **hemo_model**: `Su2023`
- **resp_model**: `SingleCompartment`
- **pk_model_nore**: `Li`, `Oualha`, `Beloeil`
- **pk_model_epi**: `Clutter`, `Abboud`, `Oualha`
- **loc_model**: `Kern`
- **disturbance_profile**: `stim_intubation_pulse`, `stim_sustained_surgery`, or `null`
- **volatile_agents**: list of enabled volatile agents (e.g., `["sevoflurane"]`)
- **rng_seed**: integer seed for deterministic noise
- **maintenance_fluid_ml_hr**: continuous IV fluid rate in mL/hr. `null` (or omitted) uses the default 1 mL/kg/hr.
- **simulation_speed**: real-time multiplier (UI only, informational in headless).
- **enable_death_detector**: `true`/`false` to enable viability checks.

Unknown configuration keys and unsupported model names are rejected with an error.

## Examples

**Run a 60-second headless simulation with custom patient:**

```bash
anasim --mode headless --duration 60 --config patient_config.json
```
