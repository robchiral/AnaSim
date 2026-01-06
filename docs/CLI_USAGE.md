# Advanced CLI usage

This document details advanced configuration options for the AnaSim command-line interface, particularly for headless mode. For basic installation and usage, see the [README](../README.md).

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Run mode: `ui` or `headless` | `ui` |
| `--duration` | Simulation duration in seconds (Headless only) | `10.0` |
| `--config` | Path to a JSON configuration file | None |

## Configuration file

You can provide a JSON file to customize the patient and simulation parameters.

### Structure

```json
{
    "age": 40,
    "weight": 70,
    "height": 170,
    "sex": "Male",
    "mode": "awake",
    "maint_type": "tiva",
    "pk_model_propofol": "Eleveld",
    "pk_model_remi": "Minto"
}
```

### Options

- **mode**: `awake`, `steady_state`, `induction`
- **maint_type**: `tiva`, `balanced`
- **pk_model_propofol**: `Marsh`, `Schnider`, `Eleveld`
- **pk_model_remi**: `Minto`

## Examples

**Run a 60-second headless simulation with custom patient:**

```bash
anasim --mode headless --duration 60 --config patient_config.json
```
