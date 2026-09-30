# Changelog

## Unreleased

- Replace the Qt desktop app with a browser interface. `anasim` serves it from
  local Python and saves recordings to `recordings/`, and the same interface
  runs at robche.com/AnaSim through Pyodide. PySide6 and pyqtgraph are no
  longer dependencies.
- Simplify the interface styling. Event buttons are colored only while the
  event is running, and automatic laryngospasm is a checkbox.
- Correct NIBP cycle timing across step sizes, show no SpO₂ value without a
  pulsatile signal, and keep the previous BP reading visible while the cuff
  measures.
- Make ventilator mechanics consistent across step sizes, count trapped-gas
  pressure once, and base gas exchange on completed exhaled breaths.
- Show capnography during PSV apnea backup and require a fresh exhalation after
  reconnection.
- The ventilator can start from an awake session and keeps its settings while
  off. `set_vent_settings` no longer starts or stops it; use `set_vent_power`.
- Setting a manual infusion rate disables TCI, so a stopped infusion stays
  stopped.
- Preserve central drug amount when hemodynamic scaling changes PK volume.
- Report CSV recording failures, keep the part already recorded, and keep
  sampling on schedule when steps cross sample deadlines.
- With `end_on_cardiac_arrest`, a confirmed arrest ends the session at that
  moment.
- Induction scenarios require starting the ventilator after intubation, TIVA
  maintenance checks the fresh gas reduction it asks for, and baseline
  objectives show the ranges they check.
- Apply arterial-line setting changes immediately and reduce waveform overhead.
- Support Python 3.14 and drop the unused pandas dependency.

## 1.2 - 2026-09-24

- Added a cardiac baroreflex, revised epinephrine responses, and corrected the
  Li norepinephrine age covariate. Saved configurations must use
  `HealthyAdult` instead of `Clutter` for epinephrine PK.
- Added myocardial hypoxia and a cardiac arrest endpoint. The configuration
  option is now `end_on_cardiac_arrest`; state fields are `cardiac_arrest` and
  `arrest_reason`.
- Added a central rocuronium effect site, so respiratory muscle recovery can
  precede TOF recovery after sugammadex.
- Added partial upper-airway obstruction after loss of consciousness and
  modeled alveolar and blood oxygen stores, including preoxygenation and
  apneic oxygenation.
- Updated target-controlled infusion to avoid rate spikes after
  resynchronization and capped propofol and remifentanil TCI at 1200 mL/h.
- Corrected sevoflurane MAC40 to 1.80%, its cardiovascular effect timing, and
  its contribution to BIS during TIVA. The gas monitor and scenario objectives
  now use end-tidal MAC.
- Improved steady-state startup, ventilator waveforms, temperature changes,
  and responses to surgical stimulation.
- Conserved drug mass during hemodynamic PK scaling and used the published
  James lean-body-mass covariate for Schnider and Minto.
- Removed the pediatric Oualha norepinephrine and epinephrine models. Saved
  configurations that select them now fail validation.
- Simplified subsystem code, expanded alarm and cardiac arrest tests, and
  shortened the architecture guide.

## 1.1 - 2026-08-30

- Enforced the supported patient domain across API, CLI, and desktop setup,
  including finite body-size, hematology, and organ-function inputs.
  Hemoglobin and hematocrit are patient inputs, and organ status labels come
  from the organ-function factors.
- Averaged stimulation profiles over each simulation step, and ended
  time-limited profiles after the step in which they finish.
- Added an arterial pressure waveform whose mean and pulse pressure come from
  Su MAP and stroke volume, with shared ECG and pleth timing and catheter-transducer dynamics.
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
- Pulse-oximeter lag and monitor signal validity.
- Supported adult patient domain of 18 to 70 years.
- Simplified configuration, simulation state, UI, and tests.
- Python package, continuous integration, and automated PyPI publishing.
