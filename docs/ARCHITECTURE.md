# AnaSim architecture

AnaSim combines models of the patient, drugs, physiology, anesthesia machine,
and monitors in one simulation engine. Each model keeps its own state. At every
step, the runtime updates the models in a fixed order and copies their outputs
into `SimulationState`.

This guide covers the code structure, the step order, the browser app, and
validation. The [model guide](MODELS.md) explains what each model does and which
studies it is based on, and [CLI and Python usage](CLI_USAGE.md) covers
configuration, units, and scripted runs.

## Main modules

| Module | Responsibility |
|--------|----------------|
| [`engine.py`](../anasim/core/engine.py) | Creates the models, applies controls, and records simulation output |
| [`runtime.py`](../anasim/core/runtime.py) | Advances the models in step order |
| [`projection.py`](../anasim/core/projection.py) and [`state.py`](../anasim/core/state.py) | Copy model outputs into the shared state and define its fields |
| [`initialization.py`](../anasim/core/initialization.py) | Sets up awake and maintenance starting states |
| [`patient/`](../anasim/patient/) and [`drug_registry.py`](../anasim/core/drug_registry.py) | Define patient characteristics, drug models, delivery units, pump limits, and interface metadata |
| [`physiology/`](../anasim/physiology/) | Models circulation, resuscitation, fluids, respiration, gas exchange, and temperature |
| [`machine/`](../anasim/machine/) | Models the breathing circuit, vaporizer, and ventilator |
| [`monitors/`](../anasim/monitors/) and [`core/monitors.py`](../anasim/core/monitors.py) | Produce measurements, waveforms, and alarms |
| [`scenarios/`](../anasim/scenarios/) | Defines cases, with their events, guided steps, and debrief outcomes |
| [`web.py`](../anasim/web.py) and [`web_assets/`](../anasim/web_assets/) | Connect the engine and cases to the browser interface |
| [`cli.py`](../anasim/cli.py) and [`local.py`](../anasim/local.py) | Run headless simulations and the local server |

## Runtime

### Step order

Each call to `SimulationEngine.step()` runs the following sequence:

1. Update shivering, then calculate thermal anesthetic depth, metabolic rate,
   and stimulation tolerance.
2. Advance stimulation, hemorrhage, fluids, anaphylaxis, and sepsis.
3. Update pharmacokinetic (PK) parameters for blood volume, cardiac output,
   renal perfusion, core temperature, and drug interactions, then calculate
   target-controlled infusion (TCI) rates.
4. Update airway complications, the vaporizer, and the breathing circuit.
5. Update lipid emulsion, intravenous drug concentrations, inhaled anesthetic
   uptake, neuromuscular block, lidocaine toxicity, and malignant hyperthermia,
   then apply toxicity, malignant hyperthermia, and sepsis effects on
   metabolism, heat production, and chest-wall stiffness.
6. Advance breathing mechanics, gas exchange, hemodynamics, and cardiac rhythm.
7. Copy physiology outputs into `SimulationState`, then update cerebral
   oxygenation and function, monitor measurements, and alarms.
8. Update core temperature, check for cardiac arrest, and clear completed finite
   stimuli.

PK and circuit calculations use the cardiac output and ventilation from the
start of the step, and the physiology models update those values for the next
step. The engine then advances the clock, trims the 20-second waveform buffer,
and writes a CSV row when due.

If a TCI rate change falls within a step, PK advances to that time before the
controller selects the next rate.

### Simulation state

`SimulationState` holds physiological values and monitor measurements.

| Group | Selected fields | Meaning |
|-------|-----------------|---------|
| Physiology | `map`, `hr`, `co`, `intrinsic_co`, `rhythm_type`, `sao2`, `pa_co2`, `pao2` | Values that the models and the cardiac arrest check use; `intrinsic_co` excludes chest compressions |
| Blood chemistry | `ph`, `hco3`, `base_excess`, `lactate`, `sodium`, `potassium`, `chloride`, `ionized_calcium`, `albumin_g_l` | Arterial plasma pH and bicarbonate at patient temperature, standard base excess, and plasma lactate, electrolytes, and albumin |
| Measurements | `art_map`, `nibp_map`, `display_spo2`, `display_bis` | Values after measurement and processing effects |
| Ventilation | `paw_peak`, `paw_plat`, `peep`, `et_o2` | Breath pressures and exhaled gas measurements |
| Fluids | `blood_volume`, `urine_out_ml`, `net_fluid_ml`, `lap`, `lung_water` | Circulating volume, fluid balance, filling pressure, and excess lung water |

Monitors take the physiological values and add sensor dynamics, sampling, and
averaging. Pressure displays and scenario objectives use whichever blood
pressure monitor is configured, arterial or noninvasive (NIBP). CSV recordings
include both the physiological values and the measurements. `state.py` defines
every field and its units.

### Initialization

`awake` starts from the patient's baseline values. `steady_state` starts a
maintenance anesthetic from a 30-minute drug history, begins controlled
ventilation, and lets the models settle before resetting the session clock,
recordings, and fluid totals. Sustained surgical stimulation settles with the
maintenance state, while finite pulses such as intubation begin when the
session starts. Startup targets and manual infusion rates use population
kinetics and potency, so sampled variability and organ impairment affect the
patient's concentrations rather than the pump settings, and vasopressors start
at zero.

## Browser app

### Cases and action checks

Each scenario file defines a briefing, timed events, case actions such as
requesting hemostasis, guided steps, and debrief outcomes. `CaseRun` in
[`scenarios/runtime.py`](../anasim/scenarios/runtime.py) starts timed events,
tracks objectives in guided or independent mode, and builds the debrief while
hiding instructor event controls. Ending a case, finishing the last guided
step, or reaching irreversible cardiac arrest pauses the model, stops
recording, and locks the controls and debrief. Open simulations also stop
recording and disable treatment controls at irreversible arrest.

[`clinical_actions.py`](../anasim/core/clinical_actions.py) evaluates
intubation placement and clinical safety warnings before actions run. The
[model guide](MODELS.md#cases-and-clinical-actions) documents case timing,
objective checks, and action rules.

### Session transport

The local and hosted apps share `web_assets/` and
[`WebSession`](../anasim/web.py), which applies commands and returns
measurements, waveforms, controls, and case state. Commands are processed in
order, and the simulation speed setting converts elapsed real time into
simulation time. The session advances the engine in 0.05-second steps, and the
monitors sample waveforms every 10 ms within each step.

The local Python server runs the session on its own clock and saves CSV
recordings to disk. It accepts requests from the same origin under a
session-specific path on 127.0.0.1. Each successful command returns an updated
snapshot. The browser polls every 50 ms while running and every 5 seconds while
paused.

The hosted app runs Python, NumPy, and SciPy in a Pyodide worker, and stopping a
recording downloads the CSV. [`build_web.py`](../scripts/build_web.py) builds
the hosted bundle, and [pages.yml](../.github/workflows/pages.yml) publishes it.

## Validation

Tests cover published equations, conservation of gas and drugs, monitor
behavior, and complete clinical scenarios in native Python and in Pyodide.
Acceptance tests check standard teaching cases (balanced and total intravenous
induction, intubation, incision, and vasopressor boluses) against ranges from
VitalDB recordings and published studies.

Offline scripts compare model outputs against clinical datasets and published
cohorts: [recorded-case comparisons](../validation/vitaldb/README.md) replay
VitalDB cases across anesthetic, respiratory, and hemodynamic components, and
the [intubation replay](../validation/intubation.py) (`python -m
validation.intubation`, with `--sensitivity` for loading-history checks) prints
induction and intubation errors against published trials.
