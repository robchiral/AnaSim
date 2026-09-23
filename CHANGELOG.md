# Changelog

## Unreleased

- Modeled alveolar and blood O2 stores. Preoxygenation now sets the safe apnea
  time (SaO2 < 90% after about 7 min preoxygenated, about 1 min on room air),
  and a patent airway provides apneic oxygenation.
- Replaced the TCI controller with a peak-constrained Shafer-Gregg controller.
  The previous controller ignored its effect-site peak time, which produced
  max-rate pulses after state resynchronization (norepinephrine plasma
  concentration spikes and 10 mmHg MAP swings in steady-state TIVA).
  Propofol and remifentanil TCI now use a 1200 mL/h syringe-pump limit.
- Combined sevoflurane with every propofol-remifentanil BIS model on one
  additive surface. Adding sevoflurane to TIVA previously raised BIS by about
  12 points. The Eleveld BIS delay now runs on simulation time and no longer
  breaks when the step size changes.
- Corrected the sevoflurane MAC40 from 2.1% to 1.80% (Mapleson 1996).
- Scaled noxious-stimulus responses by the Bouillon laryngoscopy-response
  probability so remifentanil blunts the intubation response.
- Replaced the impulse redistribution-hypothermia term with a first-order
  redistribution (about 1.3 °C), removing a 0.17 °C step at steady-state start.
- Set the balanced-anesthesia vaporizer from lung and circuit mass balance and
  pre-equilibrated circuit and alveolar gas at steady-state start.
- Kept peripheral PK volumes fixed during hemodynamic scaling so drug mass is
  conserved, and unified all intravenous PK models in `MammillaryPK`.
- Schnider and Minto now use their published James lean-body-mass covariate
  (valid across the supported BMI range) instead of Janmahasatian.
- Sevoflurane cardiovascular effects now follow end-tidal MAC through the
  hemodynamic ke0, as documented, instead of lagging the brain compartment twice.
- Abboud epinephrine clearance now uses the cohort reference SAPS II instead of
  an estimate from awake vital signs.
- Spontaneous breathing shows circuit pressure and flow only with a connected
  airway, with negative inspiratory pressure and correct flow amplitude.
- Removed unused model variants, parameters, caches, and plumbing (about 1,600
  lines), stored lightweight waveform samples, and made alarms timer-based.

## 1.1 - 2026-08-30

- Enforced the supported patient domain across API, CLI, and desktop setup,
  including finite body-size, hematology, and organ-function inputs. Patient
  data now owns hematology and derives organ status labels.
- Applied finite disturbance effects across each simulation interval and ended
  them after all current-step consumers run.
- Added a dedicated arterial pressure waveform constrained by Su MAP and stroke
  volume, with shared ECG and pleth timing and catheter-transducer dynamics.
- Made cardiac monitor synthesis and waveform history independent of the outer
  simulation step size.
- Removed pleth-derived arterial pressure, duplicate pressure reconstruction,
  separate cardiac phases, and redundant MAP and HR display smoothing.
- Added a clinician-facing guide and a clearer first-session path in the README.
- Revised setup, monitor, and control labels for clinical clarity.
- Fixed clipped content in the setup dialog.
- Moved setup cancellation out of the main window constructor and removed the
  duplicate screenshot launcher path.
- Added Ruff configuration and a CI lint job.

## 1.0 - 2026-07-17

- Integrated cardiovascular, respiratory, pharmacologic, ventilator, fluid, and
  temperature simulation.
- Interactive operating-room monitor, guided clinical scenarios, headless runner,
  and CSV recording.
- Published component models with documented simulator-specific adaptations.
- Realistic pulse-oximeter lag and monitor sample validity.
- Supported adult patient domain of 18 to 70 years.
- Simplified configuration, simulation state, UI, and tests.
- Python package, continuous integration, and automated PyPI publishing.
