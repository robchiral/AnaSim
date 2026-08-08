# CLI usage

For installation and the desktop workflow, see the [README](../README.md).

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Run mode: `ui` or `headless` | `ui` |
| `--duration` | Simulation duration in seconds (headless only) | `10.0` |
| `--config` | Path to a JSON configuration file | Not set |
| `--record` | Enable CSV recording (headless only) | `false` |
| `--record-dir` | Output directory for recordings | `recordings` |
| `--record-interval` | Sample interval in seconds for CSV | `1.0` |

## Configuration file

A JSON file can set any patient or simulation field. Omitted fields keep their
defaults.

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

### Options

- Patient fields include `age`, `weight`, `height`, `sex`, `asa`, baseline
  vital signs and hematology, and renal or hepatic function. If `baseline_hct`
  is omitted or `null`, the engine derives it from hemoglobin.
- Runtime fields include `dt`, `rng_seed`, `simulation_speed`,
  `enable_death_detector`, and `maintenance_fluid_ml_hr`. A `null` maintenance
  fluid rate uses 1 mL/kg/hr.
- `mode` accepts `awake` or `steady_state`; `maint_type` accepts `tiva` or
  `balanced`. Steady-state mode runs a managed-maintenance bootstrap before
  visible time starts.
- Model fields accept: propofol PK (`Marsh`, `Schnider`, `Eleveld`),
  remifentanil PK (`Minto`), BIS (`Bouillon`, `Eleveld`, `Fuentes`, `Yumuk`),
  hemodynamics (`Su2023`), respiration (`SingleCompartment`), norepinephrine PK
  (`Li`, `Oualha`, `Beloeil`), epinephrine PK (`Clutter`, `Abboud`, `Oualha`),
  and loss of consciousness (`Kern`, `Mertens`, `Johnson`).
- `disturbance_profile` accepts `stim_intubation_pulse`,
  `stim_sustained_surgery`, or `null`. `volatile_agents` accepts
  `["sevoflurane"]` or an empty list.

AnaSim rejects unknown configuration keys and unsupported model names with an
error.

## Examples

Run a 60-second headless simulation with a custom patient:

```bash
anasim --mode headless --duration 60 --config patient_config.json
```
