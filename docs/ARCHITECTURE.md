# AnaSim architecture

AnaSim has separate modules for pharmacology, physiology, the anesthesia
machine, and monitors. The runtime updates them in a fixed order and copies
their outputs into `SimulationState`.

## Simulation state

`SimulationState` stores model outputs separately from monitor measurements.

| Group | Fields | Meaning |
|-------|--------|---------|
| Physiology | `map`, `hr`, `co`, `sv`, `svr`, `sao2`, `pa_co2`, `alveolar_co2`, `pao2` | Model values used by physiology, analytics, and endpoints |
| Ideal arterial pulse | `sbp`, `dbp` | Systolic and diastolic pressures derived from Su MAP and stroke volume |
| Arterial catheter | `art_pressure`, `art_sbp`, `art_dbp`, `art_map` | Instantaneous and completed-beat pressures after catheter filtering |
| Other monitors | `nibp_sys`, `nibp_dia`, `nibp_map`, `display_hr`, `display_bis`, `display_etco2`, `display_spo2` | Values shown to the learner |
| Ventilator | `paw_peak`, `paw_plat`, `paw_mean`, `peep`, `compliance_dyn`, `et_o2` | Breath pressures, dynamic compliance, and end-tidal O2 |

- Monitors read model outputs without changing them.
- NIBP measures `sbp`, `dbp`, and `map`.
- The UI, CLI, alarms, and scenarios read `art_*` when the arterial line is
  enabled and `nibp_*` otherwise.
- The recorder writes both physiology and monitor values.
- `engine.output_buffer` holds 20 seconds of per-step `WaveformSample` records
  (ECG, pleth, capnogram, arterial pressure, airway pressure, flow, volume, and
  breath count) for the monitor sweep and loops.
- Public numeric fields are built-in `float` values.

## Main modules

[`engine.py`](../anasim/core/engine.py) holds the subsystems and applies learner controls.
[`runtime.py`](../anasim/core/runtime.py) advances them,
[`projection.py`](../anasim/core/projection.py) writes their outputs to state,
and [`monitors.py`](../anasim/core/monitors.py) updates measurements and alarms.
[`initialization.py`](../anasim/core/initialization.py) sets the starting state.
Drug units, pump limits, and UI metadata are defined in
[`drug_registry.py`](../anasim/core/drug_registry.py).

## Step order

```text
SimulationEngine.step()
 1. Anesthetic depth, metabolic rate, and tolerance of stimulation
 2. Disturbances and clinical events
 3. PK scaling to blood volume and CO; resynchronize active TCI
 4. TCI infusion rates
 5. Machine: vaporizer, circuit
 6. PK: plasma and effect-site concentrations, TOF, volatile uptake
 7. Physiology: ventilator or bag with lung mechanics, gas exchange, hemodynamics
 8. Copy outputs into SimulationState
 9. Monitors: cardiac cycle, arterial pulse and line, ECG, pleth, NIBP,
    capnography, alarms
10. Shivering
11. Temperature
12. Cardiac arrest check
```

## Inputs from the previous step

Update order means that some modules use values from the previous step.

| Value | Used by | Updated by | Reason |
|-------|---------|------------|--------|
| `state.co` | Volatile PK scaling, respiration | Hemodynamics | Hemodynamics runs after PK and respiration |
| `state.va` | Volatile PK | Respiration | Respiration runs after the machine and PK |
| `state.mv` | Circuit and machine | Physiology | Minute ventilation is final after mechanics and respiration |

## Initialization

- `awake` starts from patient baselines.
- `steady_state` simulates a maintenance period starting at MAP 70 mmHg.
  Controllers, drug concentrations, gases, and fluid balance carry into the
  visible session and may continue to change early in the run. Any
  norepinephrine infusion used during initialization remains active and visible.
- The session clock, recording, display history, arrest checks, and visible
  fluid and temperature totals start after initialization.

## Model notes

The [model references](REFERENCES.md) distinguish published models from
AnaSim calibrations.

### Hemodynamics

AnaSim extends the Su et al. 2023 model with blood volume, pulmonary
circulation, vasoactive drugs, a baroreflex, septic shock, and anaphylaxis.
Propofol and opioid cardiovascular effects use plasma concentrations.
Hypnosis, BIS, tolerance of stimulation, and respiratory depression use
effect-site concentrations.

The baroreflex adjusts HR, or vascular resistance when a rhythm fixes the
ventricular rate. Drug effects modify autonomic responses, and severe hypoxia
reduces HR and contractility. With `end_on_cardiac_arrest`, MAP below 20 mmHg
or HR below 10 bpm for 15 seconds ends the session. Resuscitation is not
modeled.

### Respiration

Anesthetics and opioids reduce respiratory drive. Assisted ventilation can
suppress breathing in unconscious patients by lowering PaCO2 below the
apneic threshold; awake patients retain breathing drive. Separate rocuronium
effect sites allow breathing to recover before TOF. Loss of consciousness can
cause upper-airway obstruction, relieved by positive pressure or an ETT.

Gas exchange tracks alveolar gas and blood oxygen stores, so preoxygenation,
apnea, and blood loss affect time to desaturation. Alveolar, arterial, and
end-tidal CO2 are separate; low cardiac output widens the PaCO2-EtCO2 gap.

### Ventilation mechanics

[`RespiratoryMechanics`](../anasim/physiology/resp_mech.py) models airway
resistance, compliance, tissue viscoelasticity, and inspiratory muscle
pressure. Compliance scales with predicted body weight and BMI. Pressure
support reduces muscle effort, and lung inflation changes its timing,
allowing breathing to synchronize with the ventilator. Integration steps
split at breath transitions and pressure limits.

[`AnesthesiaVentilator`](../anasim/machine/ventilator.py) controls VCV, PCV,
PCV-VG, SIMV, PSV with apnea backup, and CPAP. VCV has an adjustable pause
and pressure limit. PCV-VG adjusts pressure to meet the target volume;
SIMV, PSV, and CPAP support patient triggering and flow cycling.

Settings take effect at the next breath. `set_vent_power` starts and stops
the ventilator; `set_vent_settings` changes settings without starting it.
PSV's apnea delay is `vent.apnea_backup_s`.

Spirometry includes ventilated, bagged, and unassisted breaths. VTe is the
last exhaled volume; RR and MV average four breaths and read zero after
15 seconds without a breath. Gas exchange uses the same completed-breath
averages. Upper-airway obstruction causes mask leak or, with a tracheal tube,
increased resistance. Bronchospasm reduces alveolar ventilation without
reducing displayed VT.

### Monitors

ECG, arterial pressure, and pleth use the same beat timing and update at
intervals of 10 ms or less. Arterial pressure depends on model MAP, stroke
volume, arterial compliance, and catheter filtering. Irregular rhythms alter
filling time and beat pressures. Displayed HR averages recent R-R intervals;
NIBP updates after each cuff cycle.

BIS adds sevoflurane to the selected propofol-remifentanil model, then applies
smoothing and processing delay. Poor perfusion delays SpO2 readings and reduces
pleth amplitude. SpO2 requires an organized rhythm and adequate perfusion.

The capnograph models exhaled gas passing through airway dead space and the
analyzer. Inspiratory efforts can produce curare clefts. The gas monitor
shows end-tidal age-adjusted MAC (`et_mac`); brain MAC (`mac`) determines drug
effects. End-tidal values clear 15 seconds after the last valid exhaled CO2
sample.

Airway traces include sensor filtering. PEEP-valve resistance slows the fall
in expiratory airway pressure. Plateau pressure requires a mandatory breath
with no flow at end-inspiration; VCV without a pause leaves it blank.
Displayed PEEP is airway pressure, while auto-PEEP affects plateau pressure
and residual expiratory flow. The monitor also shows dynamic compliance and
pressure-volume and flow-volume loops of the current and previous breaths.
RR uses capnography with an airway or chest impedance ("RR imp") without one.
Alarms detect high pressure, low minute ventilation, low delivered tidal
volume, and low inspired O2. See the
[waveform references](REFERENCES.md#ventilator-waveforms).

### PK and TCI

IV drugs modeled by `MammillaryPK` have up to two peripheral compartments and
an effect site. `state_fields`, `get_ss_matrices()`, and `state_vector()` use
the same ordering for TCI and initialization.

Hemodynamic scaling changes central volume with blood volume and most
clearances with cardiac output. Rescaling central concentration preserves
drug amount. Peripheral and effect-site concentrations stay unchanged;
drug lost in shed blood is not tracked separately. Epinephrine clearance is
independent of cardiac output.

Every 10 seconds, TCI selects an infusion rate that keeps the predicted peak
concentration at or below target over the next ten minutes. Propofol and
remifentanil rates are limited to 1200 mL/h. Controllers are rebuilt after PK
parameter changes and reset from current concentrations after external
boluses. A manual rate disables TCI for that drug; changing the target
compartment replaces its controller.

### Temperature and stimulation

Induction redistributes core heat to the periphery. Responses to noxious
stimulation scale with the modeled probability of responding to laryngoscopy,
so opioids reduce hemodynamic and BIS responses.

## Browser app

Both versions use the page in `anasim/web_assets/` and a
[`WebSession`](../anasim/web.py), which applies learner commands and returns
JSON snapshots of monitor values, new waveform samples, control state, and the
current objective.

- [`local.py`](../anasim/local.py) serves the local app on 127.0.0.1 under a
  random URL path, accepts only same-origin requests, and steps the session on
  its own clock while the page polls. A reload reconnects to the paused
  session. If polling stops for 5 seconds, the session pauses and any recording
  closes. Recordings stay in the recordings directory.
- The hosted app uses a module worker to load Pyodide, numpy, and scipy from
  the CDN and run the session in the browser. Stopping a recording downloads it.
  `scripts/build_web.py` builds the site into `build/web`, and
  `.github/workflows/pages.yml` publishes it.

## Scenario objectives

`engine.actions` records controls and event transitions with their simulation
times. `WebSession` calls `begin_step()` when an objective becomes active.

| Kind | Example | Check reads |
|------|---------|-------------|
| Action | "Give 500 mL", "start the vasopressor", "select the ETT" | `engine.actions` since the objective started, plus current state where relevant |
| State | "MAP > 65", "TOF below 25%", "circuit FiO2 below 30%" | Current engine state |

Only actions taken after an objective starts count toward it. The log uses
its entry position to mark the start because actions taken while paused share
a timestamp. Queries for the current objective raise an error if none is active.
