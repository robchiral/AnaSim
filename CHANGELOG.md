# Changelog

## Unreleased

- Added a fast cardiac baroreflex, depressed by propofol and sevoflurane, with a
  30-minute set-point reset. Phenylephrine boluses cause reflex bradycardia.
- Recalibrated epinephrine to arterial infusion (Freyschuss 1986) and IV bolus
  (Takahashi 2002) data and stopped applying inotropy twice. Renamed the
  `Clutter` epinephrine PK option to `HealthyAdult`; update saved configurations.
- Raised the maintenance-preset MAP seed from 65 to 70 mmHg so sessions do not
  start below 65 mmHg.
- Added myocardial hypoxia: SaO2 below 70% depresses the heart and progresses to
  pulseless electrical activity unless oxygenation is restored.
- Added a central (diaphragm and larynx) neuromuscular effect site driven by
  free rocuronium. Breathing returns before the TOF recovers, and sugammadex
  restores spontaneous breathing.
- Switched the gas monitor and scenario MAC objectives to end-tidal MAC
  (`et_mac`).
- Added partial upper-airway obstruction at loss of consciousness without an
  ETT; positive pressure splints it open.
- Replaced the death detector with a cardiac arrest endpoint and dropped the
  HR >= 220 criterion. Renamed `enable_death_detector` to
  `end_on_cardiac_arrest`, and the `is_dead` and `death_reason` state fields to
  `cardiac_arrest` and `arrest_reason`.
- Removed the pediatric Oualha norepinephrine and epinephrine models;
  configurations that select them fail validation.
- Allowed the SpO2 display below 40% and corrected the Li norepinephrine age
  covariate from -0.377 to -0.344 per 100 years.
- Modeled alveolar and blood O2 stores. Preoxygenation sets the safe apnea time
  (SaO2 < 90% after about 7 min preoxygenated, about 1 min on room air), and a
  patent airway provides apneic oxygenation.
- Replaced the TCI controller with a peak-constrained Shafer-Gregg controller,
  which removes max-rate pulses after resynchronization (norepinephrine spikes
  and 10 mmHg MAP swings in steady-state TIVA). Limited propofol and
  remifentanil TCI to 1200 mL/h.
- Made sevoflurane additive with every propofol-remifentanil BIS model; adding
  it to TIVA had raised BIS by about 12 points. Ran the Eleveld BIS delay on
  simulation time so it does not depend on step size.
- Corrected the sevoflurane MAC40 from 2.1% to 1.80% (Mapleson 1996).
- Scaled noxious-stimulus responses by the Bouillon laryngoscopy-response
  probability so remifentanil blunts the intubation response.
- Made redistribution hypothermia first-order (about 1.3 °C), removing a
  0.17 °C step at steady-state start.
- Set the balanced-anesthesia vaporizer from lung and circuit mass balance and
  pre-equilibrated circuit and alveolar gas at steady-state start.
- Kept peripheral PK volumes fixed during hemodynamic scaling so drug mass is
  conserved, and unified all intravenous PK models in `MammillaryPK`.
- Switched Schnider and Minto from Janmahasatian to their published James
  lean-body-mass covariate, valid across the supported BMI range.
- Drove sevoflurane cardiovascular effects from end-tidal MAC through the
  hemodynamic ke0 rather than lagging the brain compartment twice.
- Set Abboud epinephrine clearance at the cohort reference SAPS II rather than
  estimating it from awake vital signs.
- Showed spontaneous-breathing circuit pressure and flow only with a connected
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
