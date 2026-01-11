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
    "mode": "awake",
    "maint_type": "tiva",
    "pk_model_propofol": "Eleveld",
    "pk_model_remi": "Minto",
    "volatile_agents": ["sevoflurane"],
    "baseline_hb": 13.5,
    "fidelity_mode": "clinical",
    "rng_seed": 123,
    "maintenance_fluid_ml_hr": null
}
```

### Options

- **mode**: `awake`, `steady_state`
- **maint_type**: `tiva`, `balanced`
- **pk_model_propofol**: `Marsh`, `Schnider`, `Eleveld`
- **pk_model_remi**: `Minto`
- **volatile_agents**: list of enabled volatile agents (e.g., `["sevoflurane"]`)
- **baseline_hb**: baseline hemoglobin in g/dL
- **fidelity_mode**: `clinical` or `literature`
- **rng_seed**: integer seed for deterministic noise
- **maintenance_fluid_ml_hr**: continuous IV fluid rate in mL/hr. `null` (or omitted) uses the default 1 mL/kg/hr.

## Examples

**Run a 60-second headless simulation with custom patient:**

```bash
anasim --mode headless --duration 60 --config patient_config.json
```
