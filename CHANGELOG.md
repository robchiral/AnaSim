# Changelog

## Unreleased

- Add a browser version, which runs locally through Pyodide.
- Make ventilator mechanics consistent across step sizes, count trapped-gas
  pressure once, and base gas exchange on completed exhaled breaths.
- Show capnography during PSV apnea backup and require a fresh exhalation after
  reconnection.
- Setting a manual infusion rate disables TCI, so a stopped infusion stays
  stopped.
- Preserve central drug amount when hemodynamic scaling changes PK volume.
- Report CSV recording failures and keep sampling on schedule when steps cross
  sample deadlines.
- Apply arterial-line setting changes immediately and reduce waveform overhead.
- Support Python 3.14 and drop the unused pandas dependency.
- Fix the desktop ventilator, which could not start from an awake session and
  restarted with zero settings. Settings can be changed before starting and are
  kept while it is off. `set_vent_settings` no longer starts or stops the
  ventilator; use `set_vent_power`.
- Induction scenarios require starting the ventilator after intubation, TIVA
  maintenance checks the fresh gas reduction it asks for, and baseline
  objectives show the ranges they check.
- End the session at the moment of a confirmed cardiac arrest in the desktop,
  browser, and headless modes. The desktop can no longer resume it.
- The browser app downloads the recorded part of a CSV when recording fails and
  reports commands sent after the simulation stops.
- `build_web.py` refuses to replace a directory that is not a previous build.

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
