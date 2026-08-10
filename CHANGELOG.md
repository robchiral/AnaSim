# Changelog

## Unreleased

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
