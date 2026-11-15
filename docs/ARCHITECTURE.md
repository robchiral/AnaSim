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

Each call to `SimulationEngine.step()` does the following:

1. Update shivering, then calculate anesthetic depth, metabolic rate, and
   stimulation tolerance.
2. Apply stimulation, hemorrhage, fluids, and other clinical events.
3. Update pharmacokinetic (PK) parameters for blood volume, cardiac output, and
   drug interactions, then calculate target-controlled infusion (TCI) rates.
4. Update the airway, vaporizer, and breathing circuit.
5. Update drug concentrations, inhaled anesthetic uptake, neuromuscular block,
   lidocaine toxicity, and malignant hyperthermia.
6. Advance breathing mechanics, gas exchange, hemodynamics, and cardiac rhythm.
7. Copy outputs into `SimulationState`, then update brain function,
   measurements, and alarms.
8. Update temperature, then check for cardiac arrest.

PK and circuit calculations use the cardiac output and alveolar ventilation from
the start of the step, and the physiology models update these values for the
next step. The engine then advances the clock, stores waveform samples, and
writes the recording.

### Simulation state

`SimulationState` holds physiological values and monitor measurements.

| Group | Selected fields | Meaning |
|-------|-----------------|---------|
| Physiology | `map`, `hr`, `co`, `intrinsic_co`, `sao2`, `pa_co2`, `pao2` | Values that the models and the cardiac arrest check use; `intrinsic_co` excludes chest compressions |
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
ventilation, and lets the models settle. Vasopressor infusions start at zero, so
mean arterial pressure (MAP) shows the untreated effect of the anesthetic. The
session clock, recordings, and fluid totals start after initialization.

The initial ventilator settings aim for the patient's resting PaCO₂. Tidal
volume is adjusted first, within limits based on predicted body weight, and then
respiratory rate. Patients with COPD have a lower rate limit to allow time for
expiration. With severe obstruction or obesity, PaCO₂ can remain above the
target.

## Browser app

### Cases

Each scenario file defines one case: a briefing, timed events, case actions
such as requesting hemostasis, guided steps, and debrief outcomes. `CaseRun` in
[`scenarios/runtime.py`](../anasim/scenarios/runtime.py) starts the events,
tracks outcomes, and builds the debrief in both practice modes. Case sessions
hide the instructor event controls that open simulation provides.

Crises start 30 simulated seconds into the case. Anaphylaxis and sepsis reach
full severity over 45 and 60 seconds, while hemorrhage and malignant
hyperthermia develop over several minutes. In the anaphylaxis case, crystalloid
boluses run at 500 mL/min.

Guided requirements count actions from the start of the case, or from crisis
onset for crisis treatment, so the guide never asks for a repeat dose. In
independent mode, the browser receives no step text, and safety warnings other
than a failed intubation appear only in the debrief.

Pressure objectives require both the displayed MAP and the current model MAP to
reach the target. Without an arterial line, the cuff reading must be no more
than one minute old. Anaphylaxis recovery also requires improved bronchospasm.

Milestone outcomes, such as preoxygenation, stay credited once achieved.
Extubation outcomes are checked when the tube is removed, and the rest must hold
when the case ends. Ending the case, finishing the last guided step, or
irreversible cardiac arrest pauses the model, stops recording, and freezes the
debrief.

### Action checks

Selecting the tracheal tube calls `SimulationEngine.attempt_intubation`, which
adds a brief intubation stimulus to any surgical stimulation. The tube is not
placed when central muscle strength is above 0.5 and either loss of response or
laryngoscopy tolerance is below 0.1. This check does not apply when hypnosis is
sustained or circulation has stopped.

[`clinical_actions.py`](../anasim/core/clinical_actions.py) shows a warning but
lets the action proceed when:

- the tube is placed in a responsive patient
- a neuromuscular blocker is given to a responsive patient without an induction
  dose in plasma
- the tube is removed during apnea, reactive Stage II anesthesia, or residual neuromuscular block (train-of-four ratio or
  blocker T1 below 90%, excluding weakness from dantrolene)
- assisted ventilation stops during apnea
- a shock is delivered to a responsive patient with spontaneous circulation

### Session transport

The local and hosted apps share `web_assets/` and
[`WebSession`](../anasim/web.py), which applies commands and returns
measurements, waveforms, controls, and case state. Commands are processed in
order, and the simulation speed setting converts elapsed real time into
simulation time. The session advances the engine in 0.05-second steps, and the
monitors sample waveforms every 10 ms within each step.

The local Python server runs the session on its own clock and saves CSV
recordings to disk. It accepts requests from the same origin under a
session-specific path on 127.0.0.1.

The hosted app runs Python, NumPy, and SciPy in a Pyodide worker, and stopping a
recording downloads the CSV. [`build_web.py`](../scripts/build_web.py) builds
the hosted bundle, and [pages.yml](../.github/workflows/pages.yml) publishes it.

## Validation

Tests cover published equations, conservation of gas and drugs, monitor
behavior, and complete clinical scenarios, in native Python and in Pyodide.
Acceptance tests check standard teaching cases (balanced and total intravenous
induction, intubation, incision, and vasopressor boluses) against ranges from
VitalDB recordings and published studies. [Recorded-case
comparisons](../validation/vitaldb/README.md) compare anesthetic, respiratory,
and hemodynamic components with VitalDB data.

The [intubation replay](../validation/intubation.py) (`python -m
validation.intubation`) prints absolute and incremental errors in systolic
pressure, heart rate (HR), MAP, and cardiac output against Billard 1994 and
Sawano 2013. These two studies set the HR recovery and the split of the
intubation response between stroke volume and vascular resistance. Ebert 1992
and 1995 set the peak HR and MAP response, and Hayakawa-Fujii 2001 is an
independent check. The replay also compares induction changes with Sjøen 2023
and 2024 and Poterman 2017. `--sensitivity` compares assumed propofol loading
histories at Sawano's plasma target.