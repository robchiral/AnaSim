# AnaSim architecture

AnaSim brings models of the patient, drugs, physiology, anesthesia machine, and
monitors together inside one simulation engine. Each model keeps its own
internal state. On every step, the runtime updates the models in a fixed order
and copies their outputs into `SimulationState`.

This guide covers the code structure, step order, browser app, and validation
workflow. The [model guide](MODELS.md) explains what each model represents and
which studies it is based on, and [CLI and Python usage](CLI_USAGE.md) covers
configuration, units, and scripted runs.

## Main modules

| Module | Responsibility |
|--------|----------------|
| [`engine.py`](../anasim/core/engine.py) | Builds the models, applies user controls, advances the clock, and records simulation output |
| [`runtime.py`](../anasim/core/runtime.py) | Runs the per-step model sequence |
| [`projection.py`](../anasim/core/projection.py) and [`state.py`](../anasim/core/state.py) | Copy model outputs into `SimulationState` and define its fields |
| [`initialization.py`](../anasim/core/initialization.py) | Initializes awake and steady-state maintenance sessions |
| [`patient/`](../anasim/patient/) and [`drug_registry.py`](../anasim/core/drug_registry.py) | Define patient parameters, drug models, units, pump limits, and UI metadata |
| [`physiology/`](../anasim/physiology/) | Models circulation, resuscitation, fluids, glucose and insulin, anticoagulation, neuraxial anesthesia, respiration, gas exchange, clinical crises, and temperature |
| [`extracorporeal/`](../anasim/extracorporeal/) | Models VV ECMO, VA ECMO, and cardiopulmonary bypass circuits |
| [`machine/`](../anasim/machine/) | Models the breathing circuit, vaporizer, and ventilator |
| [`monitors/`](../anasim/monitors/) and [`core/monitors.py`](../anasim/core/monitors.py) | Produce measurements, waveforms, and alarms |
| [`scenarios/`](../anasim/scenarios/) | Defines cases, events, pre-case setup, guided steps, and debrief outcomes |
| [`web.py`](../anasim/web.py) and [`web_assets/`](../anasim/web_assets/) | Connect the engine and cases to the browser interface |
| [`cli.py`](../anasim/cli.py) and [`local.py`](../anasim/local.py) | Run headless simulations and the local web server |

## Runtime

### Step order

Each call to `SimulationEngine.step()` runs these stages in order:

1. Update shivering and neuraxial block, then compute thermal anesthetic depth,
   metabolic rate, and stimulation tolerance.
2. Advance stimulation, hemorrhage, fluid infusions, anaphylaxis, and sepsis.
3. Update pharmacokinetic (PK) covariates for blood volume, cardiac output,
   renal perfusion, core temperature, and drug interactions, then step
   target-controlled infusion (TCI) controllers.
4. Update airway complications, the vaporizer, and the breathing circuit.
5. Advance lipid emulsion, inhaled anesthetic uptake, intravenous drugs,
   neuromuscular block, anticoagulation, and circuit drug sequestration. Then
   advance toxicity, malignant hyperthermia, pneumothorax, embolism, and
   pneumoperitoneum, and apply their effects on metabolism, heat, chest-wall
   stiffness, lung collapse, and shunt.
6. Advance respiratory mechanics, lung and oxygenator gas exchange, glucose and
   lactate metabolism, CPB reservoir drainage, hemodynamics and fluid balance,
   cardiac rhythm, and regional arterial ECMO/CPB gas mixing.
7. Copy physiology outputs into `SimulationState`, then update cerebral
   oxygenation and function, monitor measurements, and alarms.
8. Update core temperature, check for cardiac arrest, and clear any completed
   finite stimuli.

PK and circuit updates use the cardiac output and ventilation from the start of
the step, and the physiology models compute updated values for the next step.
After the model sequence finishes, the engine advances the clock, trims the
20-second waveform buffer, and logs a CSV row when recording is active.

When a TCI rate update falls inside a step, PK advances up to that boundary
before the controller chooses the next rate.

### Simulation state

`SimulationState` stores both underlying physiological values and processed
monitor measurements.

| Group | Selected fields | Meaning |
|-------|-----------------|---------|
| Physiology | `map`, `hr`, `co`, `intrinsic_co`, `pulmonary_flow`, `pump_flow`, `rhythm_type`, `sao2`, `pa_co2`, `pao2` | Values used by the models and arrest checks; `intrinsic_co` excludes compressions and pump flow, `pulmonary_flow` crosses the native lung, and `pump_flow` is arterial extracorporeal flow |
| Extracorporeal and anticoagulation | `ecmo_mode`, `ecmo_flow`, `cpb_reservoir_ml`, `upper_sao2`, `lower_sao2`, `heparin_u_ml`, `protamine_ug_ml`, `act_s` | Active circuit mode and flow, CPB reservoir level, regional arterial saturations, free heparin and protamine, and ACT |
| Blood chemistry | `ph`, `hco3`, `base_excess`, `lactate`, `glucose`, `insulin_mu_l`, `sodium`, `potassium`, `chloride`, `ionized_calcium`, `albumin_g_l` | Arterial pH at patient temperature, 37 °C plasma bicarbonate and standard base excess, blood glucose and plasma insulin, and plasma lactate, electrolytes, and albumin |
| Measurements | `art_map`, `nibp_map`, `display_spo2`, `display_bis` | Values after sensor dynamics, sampling, and smoothing |
| Ventilation | `paw_peak`, `paw_plat`, `peep`, `et_o2` | Airway pressures and exhaled gas readings |
| Fluids | `blood_volume`, `urine_out_ml`, `net_fluid_ml`, `lap`, `lung_water` | Circulating volume, fluid balance, left atrial pressure, and excess lung water |

Monitors derive displayed readings from physiological values by applying sensor
dynamics, sampling intervals, and averaging. Pressure displays and case
objectives follow whichever blood pressure monitor is configured, arterial line
or noninvasive cuff (NIBP). CSV recordings capture both underlying physiology
and monitor measurements. Every field and its units are defined in `state.py`.

### Initialization

`awake` starts directly from the patient's resting baseline. `steady_state`
builds a maintenance anesthetic from a 30-minute drug history, starts
controlled ventilation, and settles the coupled models with fluid infusion and
clearance paused before resetting the clock, vaporizer level, breath count,
glucose baseline, and gas-transport totals. Sustained surgical stimulation settles
with the maintenance state, whereas finite pulses such as intubation start at
time zero. Startup TCI targets and manual infusion rates use population
kinetics and potency, so sampled variability and organ impairment change the
patient's actual concentrations rather than the initial pump settings, and
vasopressors start at zero.

Slow-onset cases and emergence add a hidden pre-case simulation phase on top of
this baseline (`preroll_scenario` in `scenarios/base.py`). That phase advances
the physiology with lightweight scalar monitor updates, finishes with a brief
full-rate settle so waveforms and alarms latch cleanly, records negative-time
trend history for the Trends display, and resets the visible clock, buffers,
and action log back to 0:00.

## Browser app

### Cases and action checks

Each scenario defines a briefing, timed or monitor-gated events, case actions
such as requesting hemostasis, guided steps, and debrief outcomes. `CaseRun` in
[`scenarios/runtime.py`](../anasim/scenarios/runtime.py) fires events, evaluates
case outcomes, tracks safety events and hints, and compiles the debrief, while
`WebSession` manages guided step progression and disables instructor event
controls during a case. Ending a case, completing the final guided step, or
reaching irreversible cardiac arrest pauses the simulation, stops recording,
and locks treatment controls and the debrief. Open simulations also stop
recording and disable treatment controls at irreversible arrest.

[`clinical_actions.py`](../anasim/core/clinical_actions.py) checks intubation
conditions and clinical safety warnings when learner actions are applied. The
[model guide](MODELS.md#cases-and-clinical-actions) describes case behavior,
objective checks, and action rules.

### Session transport

Both the local and hosted apps share `web_assets/` and
[`WebSession`](../anasim/web.py), which handles JSON commands and returns
snapshots of measurements, waveforms, controls, and case state. Commands run in
order, and the simulation speed multiplies elapsed wall time into simulated
time. `WebSession` steps the engine in 0.05-second increments up to 4x speed
and coarser steps (up to 0.5 seconds) at higher speeds, stopping at scheduled
events, trend boundaries, and recording intervals while monitors continue to
sample waveforms every 10 ms.

The local Python server owns the simulation clock and writes CSV recordings to
disk. It binds to `127.0.0.1` under a random session-specific URL path and
accepts only same-origin requests. Each successful command returns an updated
snapshot, and the browser polls every 50 ms while running and every 5 seconds
while paused.

The hosted app runs Python, NumPy, and SciPy inside a Pyodide web worker, and
stopping a recording triggers a CSV download.
[`build_web.py`](../scripts/build_web.py) assembles the hosted bundle, and
[pages.yml](../.github/workflows/pages.yml) publishes it.

## Validation

Automated tests check published equations, gas and drug conservation, monitor
behavior, cross-module invariants (`tests/test_invariants.py`), published
clinical trial benchmarks (`tests/test_clinical_benchmarks.py`), and teaching
cases in native Python and in Pyodide. The intubation hemodynamic benchmark can
also be run directly with `python -m tests.intubation_replays` (pass
`--sensitivity` for loading-history checks).

External-dataset scripts live under `validation/`:
[VitalDB comparisons](../validation/vitaldb/README.md) evaluate the development
cohorts across anesthetic, respiratory, hemodynamic, laboratory, and thermal
endpoints, and [MOVER external validation](../validation/mover/README.md)
evaluates frozen parameters out of sample.
