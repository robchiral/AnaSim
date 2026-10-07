# Changelog

Ventilation abbreviations follow the [README](README.md#features).

## Unreleased

- Revised the setup, CLI, architecture, and reference guides.

## 1.4 - 2026-10-06

- Added pulmonary edema from fluid overload, with reduced oxygenation and
  compliance and partial relief from positive end-expiratory pressure (PEEP).
  Volume expansion increases urine output.
- Removed automatic norepinephrine from maintenance starts. Mean arterial
  pressure (MAP) now reflects the untreated anesthetic response.
- Removed cardiac-output scaling of propofol and remifentanil clearance.
  Bradycardia now increases stroke volume. Renamed `hemo_model` value
  `"Su2023"` to `"Su"`.
- Made manual infusion rates the default. Setup can enable target-controlled
  infusion (TCI) for concentration targets, maintenance starts, and dosing
  guidance. Propofol and remifentanil rates use mcg/kg/min.
- Added fentanyl with TCI, midazolam, etomidate, ketamine, esmolol, labetalol,
  and glycopyrrolate, including combined anesthetic and autonomic effects.
  Added lidocaine boluses and infusions to partly blunt stimulation responses.
- Revised induction scenarios to offer lidocaine before propofol, apply a
  laryngoscopy stimulus, and check maintenance MAP. Balanced induction gives
  fentanyl during preoxygenation and starts sevoflurane during rocuronium onset.
- Required end-tidal O₂ ≥ 90% for preoxygenation. Emergence checks unassisted
  breathing and train-of-four (TOF) ratio ≥ 90% before tube removal.
- Revised crisis instructions, fluid choices, and oxygen supply checks.
  Vasopressor objectives accept recovered pressure without another infusion.
- Added epinephrine relief of bronchospasm, positive-pressure relief of
  pharyngeal collapse, and reduced oxygen reserve after blood loss.
- Revised atrial-fibrillation hemodynamics, ECG timing, and displayed heart
  rate. The baroreflex adjusts vascular resistance when a rhythm fixes the rate.
- Added PCV-VG and SIMV, patient triggering, flow cycling, adjustable VCV pause
  and pressure limits, and flow-dependent expiratory PEEP-valve resistance.
- Added tissue viscoelasticity, compliance scaling by body size, and muscle
  pressure for spontaneous and assisted breaths. Patient effort synchronizes
  with the ventilator and can produce curare clefts. CO₂ controls apnea under
  anesthesia.
- Added airway traces, ventilator measurements, live loops, inspired and
  end-tidal gas values, and corresponding CSV fields. End-tidal values clear
  when exhaled breath detection stops. Chest-impedance respiratory rate is
  labeled "RR imp".
- Added alarms for high airway pressure, low minute ventilation, low delivered
  tidal volume, and low inspired O₂.
- Corrected expired-volume measurements, mask leak, and tracheal-tube
  obstruction. Bronchospasm reduces alveolar ventilation without reducing
  displayed minute ventilation.
- Separated `RespiratoryMechanics` from `AnesthesiaVentilator`.
  `set_vent_settings` accepts the new modes and settings; PSV's apnea delay is
  now `vent.apnea_backup_s`.

## 1.3 - 2026-09-30

- Replaced the Qt desktop app with a browser interface shared by local Python
  and the hosted Pyodide app. `anasim` opens the local interface and saves CSV
  files to `recordings/`. Removed PySide6 and pyqtgraph dependencies.
- Separated ventilator settings from power. Use `set_vent_settings` to apply
  settings and `set_vent_power` to start or stop ventilation. The ventilator
  works in awake sessions and retains settings while off.
- Corrected step-size effects on ventilator mechanics, double-counted trapped-gas
  pressure, and gas exchange timing. Gas exchange uses completed exhaled
  breaths; capnography works during PSV apnea backup.
- Corrected NIBP cycle timing across step sizes. SpO₂ requires a pulsatile
  signal; the previous blood-pressure reading remains visible during cuff cycles.
- Made manual infusion rates disable TCI and preserved central drug amount
  during hemodynamic pharmacokinetic (PK) scaling.
- Added CSV error reporting that preserves the recorded portion. Sampling
  stays on schedule when steps cross sample deadlines.
- Stopped sessions at confirmed arrest when `end_on_cardiac_arrest` is enabled.
- Required ventilator startup after intubation and fresh gas reduction during
  total intravenous anesthesia (TIVA) maintenance. Baseline objectives show
  their checked ranges.
- Applied arterial-line setting changes immediately and reduced waveform
  computation time.
- Added Python 3.14 support and removed the unused pandas dependency.

## 1.2 - 2026-09-24

- Added a cardiac baroreflex, revised epinephrine responses, and corrected the
  Li norepinephrine age covariate. Renamed epinephrine PK option `"Clutter"` to
  `"HealthyAdult"` in saved configurations.
- Added myocardial hypoxia and a cardiac arrest endpoint. Renamed the
  configuration option to `end_on_cardiac_arrest`; state fields are
  `cardiac_arrest` and `arrest_reason`.
- Added a central rocuronium effect site, allowing respiratory muscle recovery
  before TOF recovery after sugammadex.
- Added upper-airway obstruction after loss of consciousness and alveolar and
  blood oxygen stores for preoxygenation and apneic oxygenation.
- Removed TCI rate spikes after resynchronization and capped propofol and
  remifentanil TCI at 1200 mL/hr.
- Corrected sevoflurane minimum alveolar concentration (MAC) at age 40 to 1.80%,
  cardiovascular effect timing, and its bispectral index (BIS) contribution
  during TIVA. Gas monitoring and scenario objectives use end-tidal MAC.
- Revised steady-state startup, ventilator waveforms, temperature changes,
  and stimulation responses.
- Preserved drug mass during hemodynamic PK scaling and used the published
  James lean body mass covariate for Schnider and Minto.
- Removed pediatric Oualha norepinephrine and epinephrine models.
  Configurations that select them fail validation.
- Simplified subsystem code and expanded alarm and cardiac arrest tests.

## 1.1 - 2026-08-30

- Enforced patient limits across the API, CLI, and desktop setup. Added
  hemoglobin and hematocrit inputs and derived organ status labels from
  organ-function factors.
- Averaged stimulation profiles over each step and ended timed profiles after
  the step in which they finish.
- Added an arterial pressure waveform based on Su MAP and stroke volume,
  with shared ECG and pleth timing and catheter-transducer dynamics.
- Made cardiac waveforms and their history independent of step size. Removed
  pleth-derived arterial pressure, duplicate pressure reconstruction, separate
  cardiac phases, and redundant MAP and heart-rate display smoothing.
- Added a clinical guide and first-session instructions to the README.
- Revised setup, monitor, and control labels and fixed clipped setup content.
- Simplified setup cancellation and removed the duplicate screenshot launcher.
- Added Ruff configuration and CI linting.

## 1.0 - 2026-07-17

- Integrated cardiovascular, respiratory, pharmacologic, ventilator, fluid, and
  temperature simulation using published models with documented adaptations.
- Added an operating-room monitor, guided scenarios, a headless runner, and
  CSV recording, including pulse-oximeter lag and signal validity.
- Supported adults aged 18 to 70 years.
- Simplified configuration, simulation state, interface code, and tests.
- Added Python packaging, continuous integration, and automated PyPI publishing.
