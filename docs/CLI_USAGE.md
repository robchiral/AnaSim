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

Headless runs accept a flat JSON object and print a final monitored-vitals
summary when the run ends. Omitted fields use their defaults, and unknown fields
are rejected. On the command line, `--mode` selects the interface; in JSON,
`mode` selects the patient's starting state.

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
| `baseline_glucose` | `null` | Blood glucose, 40 to 400 mg/dL; `null` uses 90 mg/dL (150 mg/dL when `type2_diabetes` is `true`) |
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
| `type2_diabetes` | `false` | Type 2 diabetes with insulin resistance, blunted beta-cell secretion, higher renal glycosuria threshold, and 150 mg/dL default baseline glucose |
| `surgical_site` | `"other"` | `"other"`, `"lower_abdominal"`, or `"upper_abdominal"`; abdominal surgery keeps part of the anesthetic airway closure after emergence and slows postoperative clearance of atelectasis |
| `gestational_age_weeks` | 0.0 | `0` for no pregnancy, or 4 to 40 weeks for a female patient aged 55 or under |
| `position` | `"supine"` | `"supine"` or `"left_lateral"`; relieves aortocaval compression from 20 weeks of pregnancy and provides Durant left-lateral positioning during venous air embolism |

Patient ranges are inclusive, and patient inputs and `dt` must be finite. If you
set `baseline_hct`, it must be within 0.12 of `0.03 * baseline_hb`, and setting
both `renal_function` and `creatinine_clearance_ml_min` rejects grossly
inconsistent combinations. During pregnancy, hemoglobin and hematocrit are the
measured starting values. `engine.set_position("left_lateral")` changes position
during a run. The [model guide](MODELS.md#supported-patient-domain) describes
how the models represent these patient conditions.

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
| `fluid_warmer_enabled` | `false` | Warm infused intravenous fluids and packed red cells to 37 °C |
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
| `set_bag_mask_ventilation(True, rr, vt, p_max)` | Starts bag ventilation. Optional settings are rate (1 to 40 breaths/min), tidal volume (0.05 to 1.2 L), and APL pressure limit (0 to 70 cmH₂O); omitted settings keep their current values |
| `deliver_manual_breath()` | Queues one manual squeeze at the current bag tidal volume and APL setting; requires a connected airway, APL above 0 cmH₂O, and both the ventilator and continuous bagging off |
| `set_apl_valve(cmh2o)` | Sets the APL valve for bag ventilation and spontaneous breathing (0 to 70 cmH₂O; default 0, fully open) |
| `trigger_o2_flush(duration_s=1.5)` | Delivers 45 L/min pure O₂ bypassing the vaporizer for 0.1 to 10 s; requires an active O₂ supply |
| `set_mask_seal(seal)` | Takes a fraction from 0 (open) to 1 (tight seal), whereas the browser shows percent |
| `set_vaporizer(percent)` | Sets the sevoflurane dial as a percentage |
| `give_drug_bolus(drug_name, amount)` | Gives a bolus in the [registry's units](../anasim/core/drug_registry.py), or `"sugammadex"` in mg |
| `set_drug_rate(key, rate_user_unit)` | Uses the [registry's units](../anasim/core/drug_registry.py). Weight-based rates (propofol, remifentanil, rocuronium, dobutamine, milrinone, esmolol, ketamine, and lidocaine) use total body weight; other infusion rates are not weight-scaled |
| `enable_tci(drug, target, mode="effect_site")` | Requires `tci_enabled=True`; a non-positive target stops TCI |
| `set_drug_target(key, target)` | Preserves the active plasma or effect-site mode; `None` or a non-positive target stops TCI |
| `give_blood(volume_ml=300.0, hematocrit=0.55)` | Queues packed red cells at 75 mL/min on the blood channel; donor hematocrit must be between 0.01 and 0.70. Repeated fluid, blood, lipid, bolus dextrose, or bicarbonate orders drain sequentially on their own channel rather than multiplying delivery rate |
| `give_d5w(volume_ml=250.0)`, `give_d10w(volume_ml=100.0)`, `give_d50w(volume_ml=50.0)` | Queues 5%, 10%, or 50% intravenous dextrose (D5W on the fluid channel; D10W at 50 mL/min and D50W at 25 mL/min on a separate bolus dextrose channel) |
| `give_lipid_emulsion(volume_ml)` | Gives up to 3,000 mL of 20% lipid emulsion over 2.5 simulated minutes |
| `set_lipid_infusion(ml_min)` | Sets a 20% lipid emulsion infusion from 0 to 200 mL/min |
| `give_calcium_chloride(mg)` | Gives calcium chloride dihydrate up to 10,000 mg; 1 g contains 6.8 mmol calcium |
| `give_sodium_bicarbonate(meq=50.0)` | Queues 8.4% sodium bicarbonate (1 mEq/mL, up to 500 mEq) at 100 mL/min on its own intravenous channel |
| `set_fluid_warmer(enabled)` | Turns the 37 °C intravenous fluid and blood warmer on or off |
| `charge_defibrillator()` | Charges to 200 J in 5 simulated seconds and holds the charge for 60 seconds |
| `disarm_defibrillator()` | Cancels a charge or a pending synchronized shock |
| `defibrillate(synchronized=False)` | Returns a `ShockResult`: `NOT_READY` without a charge, `NOT_DELIVERED` in synchronized mode without R waves, `PENDING` until the next R wave, or `DELIVERED`. `rng_seed` fixes outcomes |
| `refill_vaporizer(level_ml=250.0)` | Refills the sevoflurane vaporizer liquid reservoir (0 to 250 mL) |
| `set_ards_shunt(fraction)` | Sets additional right-to-left pulmonary shunt (0 to 0.90) |
| `start_vv_ecmo(...)`, `start_va_ecmo(...)`, `start_cpb(...)` | Starts VV ECMO, VA ECMO, or CPB with optional flow, sweep, cannulation, prime, and thermal settings (see [`engine.py`](../anasim/core/engine.py)); `stop_extracorporeal(reinfuse=False)` stops support |
| `set_ecmo_flow(l_min)`, `set_ecmo_sweep(l_min)`, `set_ecmo_fio2(fio2)`, `set_ecmo_sevo(percent)` | Adjusts active circuit flow (0 to 8 L/min), sweep gas (0 to 16 L/min), sweep FiO₂ (0.21 to 1.0), or oxygenator sevoflurane (0 to 8%); additional setters control FCO₂, membrane area, heater-cooler, LV vent, CPB venous clamp, cross-clamp, cardioplegia, blood-gas strategy, and bypass alarm mode |
| `add_cpb_reservoir_fluid(volume_ml, kind="crystalloid", hematocrit=0.55)` | Adds 1 to 2,000 mL of crystalloid, saline, colloid (`"albumin"`), or blood directly to the CPB reservoir |
| `draw_blood_gas(site="arterial")`, `measure_act()`, `measure_glucose()` | Draws a blood gas sample (ready after 120 s; sites `"arterial"`, `"right_radial"`, `"femoral"`, `"pre_oxy"`, `"post_oxy"`), whole-blood ACT (ready after 60 s), or point-of-care blood glucose (ready after 10 s, mg/dL) |
| `start_protamine_reaction()`, `stop_protamine_reaction()`, `arm_protamine_reaction(armed=True)` | Controls a Type III protamine pulmonary vasoconstriction reaction |
| `attempt_intubation(technique="dl")`, `attempt_sga()`, `perform_fona()`, `set_oral_airway(placed)`, `set_two_hand_mask(active)`, `call_airway_help()` | Airway escalation (`"dl"`, `"vl"`, `"vl_bougie"`, `"fiberoptic"`), SGA rescue, eFONA, and mask adjuncts; instructor triggers `set_difficult_airway(profile)`, `arm_esophageal_intubation(armed=True)`, and `set_esophageal_intubation(active, ...)` (see [`engine.py`](../anasim/core/engine.py)) |
| `start_pneumothorax(...)`, `decompress_pneumothorax()`, `start_pulmonary_embolism(...)`, `give_thrombolysis()`, `start_venous_air_embolism(...)`, `aspirate_vae_air()` | Thoracic and embolic crisis controls, rescue procedures, and matching `stop_*` helpers (see [`engine.py`](../anasim/core/engine.py)) |
| `give_spinal(...)`, `place_epidural(placed=True)`, `give_epidural_bolus(...)`, `set_epidural_infusion(...)`, `check_neuraxial_level()`, `trigger_high_spinal()`, `stop_neuraxial()` | Spinal and epidural neuraxial anesthesia, bedside dermatome and Bromage level checks, and high-spinal controls (see [`engine.py`](../anasim/core/engine.py)) |
