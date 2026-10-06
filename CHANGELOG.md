# Changelog

## Unreleased

## 1.4 - 2026-10-06

- Fluid overload causes pulmonary edema, which lowers oxygenation and
  compliance; PEEP partly reverses it. Urine rises with volume expansion.
- Maintenance starts no longer add norepinephrine to reach a preset MAP;
  vasopressors start only when the user gives them.
- Propofol and remifentanil clearances no longer scale with CO. Bradycardia
  now increases stroke volume (Su Eq. 9).
- Renamed the `hemo_model` option from `"Su2023"` to `"Su"`.
- Manual infusion rates are the default. Setup can enable TCI controls,
  maintenance starts, and dosing guidance. Propofol and remifentanil rates
  use mcg/kg/min.
- Added fentanyl with TCI, midazolam, etomidate, ketamine, esmolol, labetalol,
  and glycopyrrolate, including combined anesthetic and autonomic effects.
- Added lidocaine by bolus or infusion. It slightly blunts the response to
  intubation and surgical stimulation.
- Induction scenarios offer lidocaine before propofol, an airway stimulus at
  laryngoscopy, and a MAP target during maintenance. Balanced induction gives
  fentanyl during preoxygenation and starts sevoflurane during rocuronium onset.
- Preoxygenation requires end-tidal O₂ ≥ 90%. Emergence checks unassisted
  breathing and TOF ratio ≥ 90% before tube removal.
- Revised crisis instructions, fluid choices, and oxygen supply checks.
  Vasopressor objectives accept recovered pressure without another infusion.
- Added epinephrine relief of bronchospasm, positive-pressure relief of
  pharyngeal collapse, and reduced oxygen reserve after blood loss.
- Revised AF hemodynamics, ECG timing, and displayed HR. The baroreflex adjusts
  vascular resistance when a rhythm fixes the ventricular rate.
- Added PCV-VG and SIMV, patient triggering and flow cycling, and adjustable
  VCV pause and pressure limits.
- Added tissue viscoelasticity, compliance scaling by body size, and muscle
  pressure for spontaneous and assisted breaths, including synchronization
  with the ventilator, curare clefts, and CO₂-dependent apnea under anesthesia.
- Added airway traces, ventilator measurements and live loops, inspired and
  end-tidal gas values, and corresponding CSV fields. End-tidal values clear
  when exhaled breath detection stops.
- Added alarms for high airway pressure, low minute ventilation, low delivered
  tidal volume, and low inspired O₂.
- Corrected exhaled volume measurements, mask leak, and tracheal-tube
  obstruction. Bronchospasm reduces alveolar ventilation without reducing
  displayed MV.
- Added flow-dependent PEEP-valve resistance during expiration.
- Labeled chest-impedance respiratory rate as "RR imp".
- Separated `RespiratoryMechanics` from `AnesthesiaVentilator`.
  `set_vent_settings` accepts the new modes and settings; PSV's apnea delay is
  now `vent.apnea_backup_s`.

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
  computation time.
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

- Enforced patient limits across API, CLI, and desktop setup,
  including finite body-size, hematology, and organ-function inputs.
  Hemoglobin and hematocrit are patient inputs, and organ status labels come
  from the organ-function factors.
- Averaged stimulation profiles over each simulation step, and ended
  time-limited profiles after the step in which they finish.
- Added an arterial pressure waveform whose mean and pulse pressure come from
  Su MAP and stroke volume, with shared ECG and pleth timing and
  catheter-transducer dynamics.
- Made cardiac waveforms and their history independent of the simulation
  step size.
- Removed pleth-derived arterial pressure, duplicate pressure reconstruction,
  separate cardiac phases, and redundant MAP and HR display smoothing.
- Added a clinical guide and first-session instructions to the README.
- Revised setup, monitor, and control labels for clinical clarity.
- Fixed clipped content in the setup dialog.
- Moved setup cancellation out of the main window constructor and removed the
  duplicate screenshot launcher path.
- Added Ruff configuration and a CI lint job.

## 1.0 - 2026-07-17

- Integrated cardiovascular, respiratory, pharmacologic, ventilator, fluid, and
  temperature simulation.
- Added an interactive operating-room monitor, guided clinical scenarios,
  a headless runner, and CSV recording.
- Used published component models with documented simulator-specific adaptations.
- Modeled pulse-oximeter lag and monitor signal validity.
- Supported adults aged 18 to 70 years.
- Simplified configuration, simulation state, UI, and tests.
- Added a Python package, continuous integration, and automated PyPI publishing.
