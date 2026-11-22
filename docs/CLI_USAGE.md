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

## Headless runs

Headless runs accept a flat JSON object. Omitted fields use their defaults, and
unknown fields are rejected. On the command line, `--mode` selects the
interface; in JSON, `mode` selects the patient's starting state.

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

Then run it and record the output.

```bash
anasim --mode headless --duration 60 --config patient_config.json \
    --record --record-dir results --record-interval 0.1
```

## Fields

### Patient inputs

| Field | Default | Units or accepted values |
|-------|---------|--------------------------|
| `age` | 40 | 18 to 85 years |
| `weight` | 70 | 50 to 200 kg; use pre-pregnancy weight during pregnancy |
| `height` | 170 | 150 to 200 cm; BMI must be 18 to 60 kg/m² |
| `sex` | `"male"` | `"male"` or `"female"` |
| `baseline_temp` | 37 | Core temperature, 25 to 42 °C |
| `baseline_hb` | 13.5 | Hemoglobin, 6 to 20 g/dL |
| `baseline_hct` | `null` | Hematocrit as a fraction, 0.18 to 0.60; `null` derives it from hemoglobin |
| `baseline_hr` | 70 | Heart rate before any rise from pregnancy, at least 10 bpm |
| `baseline_map` | 90 | Mean arterial pressure, at least 20 mmHg |
| `baseline_rr` | 12 | Respiratory rate, at least 0 breaths/min |
| `baseline_vt` | `null` | Tidal volume, at least 50 mL; `null` uses the volume that keeps resting PaCO₂ at 40 mmHg (lower in pregnancy) |
| `renal_function` | 1.0 | Renal function factor, 0.05 to 1.0; `1.0` is normal |
| `creatinine_clearance_ml_min` | `null` | Absolute creatinine clearance for rocuronium and sugammadex, 4.3 to 221 mL/min; `null` estimates it from age and renal function. When `renal_function` is left at `1.0`, supplying this field also derives `renal_function` for other renally cleared drugs |
| `hepatic_function` | 1.0 | Hepatic function factor, 0.5 to 1.0; `1.0` is normal |
| `coronary_flow_reserve` | 5.0 | Ratio of maximal to resting coronary flow, 1 to 5; `5` is normal |
| `heart_failure` | `"none"` | `"none"`, `"hfref"` (reduced EF), or `"hfpef"` (preserved EF) |
| `copd_severity` | 0.0 | Severity of a mixed obstructive and emphysematous profile, 0 to 1; this is not a GOLD stage |
| `surgical_site` | `"other"` | `"other"`, `"lower_abdominal"`, or `"upper_abdominal"`; abdominal surgery keeps part of the anesthetic airway closure after emergence and slows postoperative clearance of atelectasis |
| `gestational_age_weeks` | 0.0 | `0` for no pregnancy, or 4 to 40 weeks for a female patient |
| `position` | `"supine"` | `"supine"` or `"left_lateral"`; affects venous return from 20 weeks of pregnancy |

Patient ranges are inclusive, and patient inputs and `dt` must be finite. If you
set `baseline_hct`, it must be within 0.12 of `0.03 * baseline_hb`. During
pregnancy, hemoglobin and hematocrit are the measured starting values.
`engine.set_position("left_lateral")` changes position during a run. The [model
guide](MODELS.md#supported-patient-domain) describes how the models represent
these patient conditions.

### Initialization and runtime

| Field | Default | Meaning |
|-------|---------|---------|
| `mode` | `"awake"` | `"awake"` or `"steady_state"`; see [initialization](ARCHITECTURE.md#initialization) |
| `maint_type` | `"tiva"` | `"tiva"` (propofol-remifentanil) or `"balanced"` (sevoflurane-remifentanil) maintenance for steady-state initialization |
| `concomitant_opioids` | `true` | Whether the patient receives opioids, a covariate in the Eleveld propofol PK model; steady-state initialization requires `true` |
| `tci_enabled` | `false` | Enable target-controlled infusion (TCI) and use it at maintenance startup |
| `dt` | 0.01 | Engine step in seconds, above 0 and at most 2.0 |
| `rng_seed` | `null` | Integer for repeatable runs; `null` for a new random sequence |
| `patient_variability` | `false` | Sample individual drug kinetics and responses; `rng_seed` fixes the sampled patient |
| `arterial_line_enabled` | `true` | Display continuous arterial pressure |
| `fluid_warmer_enabled` | `false` | Warm infused fluids (crystalloids, albumin, lipid emulsion) and packed red cells to 37 °C |
| `maintenance_fluid_ml_hr` | `null` | Maintenance fluid rate in mL/hr; `null` uses 1 mL/kg/hr |
| `sevoflurane_enabled` | `true` | Enable the sevoflurane vaporizer; `"balanced"` requires `true` |
| `disturbance_profile` | `null` | `"stim_intubation_pulse"`, `"stim_sustained_surgery"`, or `null` |

## Recordings

`anasim_log_<timestamp>.csv` contains sampled
[`SimulationState`](../anasim/core/state.py) values. `time` is the elapsed
simulation time in seconds. The default interval is 1 second, and each engine
step records at most one row. An interval shorter than `dt` still produces one
row per step. Restarting a recording creates a new CSV file.

`pa_co2` is arterial CO₂. `display_etco2` is the end-tidal CO₂ measurement with
sensor effects, and `etco2_signal_valid` reports whether it is valid. When
`spo2_signal_valid` or `etco2_signal_valid` is false, the corresponding
displayed reading is `0`. `NaN` in `paw_plat` means that plateau pressure is
unavailable. The [field groups](ARCHITECTURE.md#simulation-state) and
[units](../anasim/core/state.py) describe the other columns.

To reproduce a run, save the configuration, AnaSim version, random seed, `dt`,
and intervention times.

## Python use

This example records three 20-second phases of anesthetized maintenance. It
introduces bronchospasm in the second phase and relieves it in the third.

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

Read results through `engine.state` or `get_latest_state()`, and apply controls
through engine methods. `engine.output_buffer` holds 20 seconds of monitor
waveform samples, spaced 10 ms apart or less at any `dt`.

| Method | Notes |
|--------|-------|
| `set_vent_settings(rr, vt, peep, ie, mode, ...)` | Takes tidal volume `vt` in liters, whereas `Patient.baseline_vt` takes mL |
| `set_vent_power(True)` | Starts the ventilator after you apply its settings |
| `set_bag_mask_ventilation(True, rr, vt, p_max)` | Starts bag ventilation. Optional settings are rate (1 to 40 breaths/min), tidal volume (0.05 to 1.2 L), and APL pressure limit (0 to 70 cmH₂O). Omitted settings keep their values. |
| `deliver_manual_breath()` | Queues a single manual squeeze at the current bag tidal volume and APL setting while the ventilator and continuous bagging are off. |
| `set_apl_valve(cmh2o)` | Sets the APL valve for bag ventilation and spontaneous breathing. The default is 0 cmH₂O (fully open). |
| `trigger_o2_flush(duration_s=1.5)` | Delivers 45 L/min pure O₂ bypassing the vaporizer for `duration_s` seconds (0.1 to 10 s), requiring an active O₂ supply. |
| `set_mask_seal(seal)` | Takes a fraction from 0 (open) to 1 (tight seal), whereas the browser shows percent |
| `set_vaporizer(percent)` | Sets the sevoflurane dial as a percentage |
| `give_drug_bolus(drug, amount)` | Gives a bolus in the [registry's units](../anasim/core/drug_registry.py), or `"sugammadex"` in mg |
| `set_drug_rate(key, rate)` | Uses the [registry's units](../anasim/core/drug_registry.py). Weight-based rates (propofol, remifentanil, rocuronium, dobutamine, milrinone, esmolol, ketamine, and lidocaine) use total body weight; other infusion rates are not weight-scaled |
| `enable_tci(drug, target, mode="effect_site")` | Requires `tci_enabled=True`; a non-positive target stops TCI |
| `set_drug_target(key, target)` | Preserves the active plasma or effect-site mode; `None` or a non-positive target stops TCI |
| `give_blood(volume_ml, hematocrit=0.55)` | Queues packed red cells at 75 mL/min; donor hematocrit must be between 0.01 and 0.70 |
| `give_lipid_emulsion(volume_ml)` | Gives up to 3,000 mL of 20% lipid emulsion over 2.5 simulated minutes |
| `set_lipid_infusion(ml_min)` | Sets a 20% lipid emulsion infusion from 0 to 200 mL/min |
| `give_calcium_chloride(mg)` | Gives calcium chloride dihydrate up to 10,000 mg; 1 g contains 6.8 mmol calcium |
| `set_fluid_warmer(enabled)` | Turns the 37 °C intravenous fluid and blood warmer on or off |
| `charge_defibrillator()` | Charges to 200 J in 5 simulated seconds and holds the charge for 60 seconds |
| `disarm_defibrillator()` | Cancels a charge or a pending synchronized shock |
| `defibrillate(synchronized=False)` | Returns a `ShockResult`: `NOT_READY` without a charge, `NOT_DELIVERED` in synchronized mode without R waves, `PENDING` until the next R wave, or `DELIVERED`. `rng_seed` fixes outcomes |
