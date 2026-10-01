# Changelog

## Unreleased

- Preoxygenation objectives wait for lung O₂ wash-in (about 3 minutes), and
  the paralysis objective waits for TOF ≤ 5%.
- Extubation requires spontaneous VT > 5 mL/kg, TOF ratio ≥ 90%, and SpO₂ >
  95%. SpO₂ objectives need a valid reading.
- Epinephrine relieves bronchospasm. Mask CPAP or positive-pressure breaths
  relieve pharyngeal collapse in proportion to airway pressure.
- Blood loss shortens safe apnea time, and the apneic CO₂ rise follows
  metabolic rate.
- Cardiac arrest gives zero O₂ delivery, and volatile agent is conserved
  across large time steps.

## 1.3 - 2026-09-30

- Replaced the Qt desktop app with a browser interface. `anasim` opens it from
  local Python and saves recordings to `recordings/`, and the same interface
  runs at robche.com/AnaSim through Pyodide. PySide6 and pyqtgraph are no
  longer dependencies.
- Changed `set_vent_settings` so it no longer starts or stops the ventilator;
  use `set_vent_power`. The ventilator now starts from an awake session and
  keeps its settings while off.
- Made ventilator mechanics consistent across step sizes, counted trapped-gas
  pressure once, based gas exchange on completed exhaled breaths, and showed
  capnography during PSV apnea backup.
- Corrected NIBP cycle timing across step sizes. SpO₂ shows a value only with
  a pulsatile signal, and the previous BP reading stays visible while the cuff
  measures.
- Made a manual infusion rate disable TCI and conserved the central drug
  amount during hemodynamic PK scaling.
- Reported CSV recording failures, kept the part already recorded, and kept
  sampling on schedule when steps cross sample deadlines.
- Ended the session at the moment of a confirmed arrest when
  `end_on_cardiac_arrest` is set.
- Required starting the ventilator after intubation in induction scenarios,
  checked the fresh gas reduction in TIVA maintenance, and showed the ranges
  that baseline objectives check.
- Applied arterial-line setting changes immediately and reduced waveform
  overhead.
- Added Python 3.14 support and dropped the unused pandas dependency.

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
