# Changelog

## Unreleased

- Limited atelectatic shunt to about 10% of cardiac output; lost aeration still
  lowers lung volume and compliance. Open-airway apnea after 100, 80, and 60%
  O₂ matches Edmark 2003.
- Separated HR and MAP responses to stimulation from movement tolerance, using
  the remifentanil concentrations that blunt them (Albertin 2005), so surgery
  raises MAP and HR during maintenance. Without opioid, intubation raises HR
  36 bpm and MAP 35-40 mmHg for about 5 minutes (Ebert 1992, 1995);
  laryngospasm follows only the first 40 s.
- Placed the anesthetized apneic threshold 4.5 mmHg below the resting PaCO₂
  set by the drugs (Hickey 1971), so balanced maintenance no longer breathes
  over VCV.
- Blunted hemorrhage tachycardia under anesthesia as for the cardiac baroreflex.
- Held the pharynx open by mean pressure over recent breaths during mask
  ventilation, so obstruction no longer alternates with unassisted breaths.
- Required MAP 20% below its crisis-start value for shock recognition in the
  hemorrhage and sepsis scenarios.
- Set the baseline A-a gradient to sea-level means by age (Crapo 1999) and
  rejected negative or non-finite hemorrhage rates.
- Corrected sevoflurane cardiovascular equilibration, pulmonary transit delays,
  and CO₂ washout across time steps.
- Isolated alarm flags in returned state copies from the live simulation.
- Restricted rocuronium target-controlled infusion (TCI) to free plasma targets.
  Decay predictions include neuromuscular kinetics, sugammadex, and inhaled agents.
- Corrected gas exchange and maintenance tidal volumes for body size. Oxygen
  consumption scales with weight^0.75, CO₂ production sets alveolar CO₂, and
  dead space follows predicted body weight. `baseline_vt` defaults to the volume
  that holds resting PaCO₂ at 40 mmHg.
- Applied the continuous sevoflurane-nitrous oxide interaction at awakening and
  included nitrous oxide in laryngoscopy tolerance.
- Limited blood-volume scaling to the vascular part of each drug's central
  volume. Renal clearance of milrinone, glycopyrrolate, and rocuronium follows
  renal perfusion.
- Updated pharmacokinetic (PK) covariates on 0.1% hemodynamic changes and
  resynchronized active TCI controllers. `get_predicted_csht` accepts any registry drug.
- Replaced rocuronium and sugammadex PK/PD with Kleijn 2011's integrated model
  and set rocuronium's renal clearance share to 20%.
- Added blood-loss scaling of propofol clearances and remifentanil elimination
  clearance, reversed by volume restoration.
- Added reciprocal propofol-midazolam PK covariates with drug mass conservation
  and TCI resynchronization.
- Recalibrated ketamine to raise HR and cardiac output about 30% with SVR
  unchanged (Tweed 1972), added a separate effect site with a 2.28 min
  equilibration half-time, and included ketamine in laryngoscopy tolerance.
  Anesthetic ketamine concentrations depress the baroreflex (Akine 2001).
- Added passive lung-gas refill after relief of complete obstruction during
  apnea, with oxygen drawn from the breathing circuit.
- Stabilized awake core temperature, aligned thermoregulatory vasoconstriction
  with anesthetic depth, and recalibrated anesthetic heat loss and redistribution.

## 1.5 - 2026-10-10

- Corrected circuit gas balance, vaporizer delivery as liquid runs out, and
  shunt mixing when venous oxygen is depleted.
- Added PK substeps to preserve drug mass during larger steps. Corrected
  rocuronium onset and recovery and matched pressure feedback to the displayed circulation.
- Added VitalDB comparison scripts and reports for ventilation, capnography,
  gas exchange, and positive end-expiratory pressure (PEEP).
- Fixed anesthetic PK to Eleveld 2018 propofol and Eleveld 2017 remifentanil. Added
  `concomitant_opioids` for the propofol opioid-regimen covariate.
- Paired Eleveld bispectral index (BIS) with Bouillon clinical responsiveness
  and laryngoscopy tolerance. Added separate clinical and respiratory effect
  sites and opioid suppression of stimulation-related BIS arousal.
- Fixed norepinephrine to Li PK with propofol-dependent clearance and responses
  calibrated to de Keijzer and Joachim data. Removed additional cardiac-output scaling.
- Fixed epinephrine to Ensinger arterial clearance with adult infusion and
  bolus calibration. Delayed beta-2 vasodilation and limited the alpha response
  to reduce early pressure dips and large-bolus pressure clipping.
- Adopted Bouillon propofol depression and an Olofsen signed opioid/CO₂
  controller with nonlinear CO₂ feedback. Corrected depression to act on
  alveolar ventilation and updated maintenance starts and walkthroughs.
- Corrected ongoing CO₂ accumulation during apnea, including a zero baseline
  respiratory rate.
- Preserved inspiratory effort and impedance respiratory rate during
  obstruction, limited airflow as the airway closes, and held end-tidal CO₂
  until its timeout while connected.
- Coupled circuit oxygen removal to lung gas exchange and oxygen stores.
  Complete obstruction stops circuit transport while metabolism uses patient stores.
- Derived intrathoracic pressure from resolved airway pressure and muscle
  effort, removing offsets that exaggerated venous return.
- Matched TCI prediction pulses to rate-update intervals. Norepinephrine TCI
  uses live PK, endogenous secretion, and propofol-dependent clearance;
  maintenance starts seed the corresponding endogenous concentration.
- Initialized hemodynamic turnover and fast drug responses together to
  preserve pressure and cardiac output at maintenance startup.
- Revised urine output to preserve most basal output at mean arterial pressure
  (MAP) 60 and taper toward zero near circulatory collapse.
- Corrected hemorrhage responses through filling-dependent right atrial
  pressure and venous recruitment, preserving pressure through ATLS class II.
- Blunted adrenergic agonists in severe anaphylaxis to prevent excessive
  pressure rises after epinephrine boluses.
- Removed `simulation_speed`, `pk_model_remi`, `hemo_model`, `resp_model`,
  `pk_model_propofol`, `bis_model`, `loc_model`, `pk_model_nore`, `pk_model_epi`,
  and the `asa` patient input, along with model selectors and alternative implementations.
- Organized drug models and controls in `engine.pk`, `engine.infusion_rates`,
  and `engine.tci`; moved temperature, anaphylaxis, and sepsis to `engine.thermal`,
  `engine.anaphylaxis`, and `engine.sepsis`.

## 1.4 - 2026-10-06

- Added pulmonary edema from fluid overload, reduced oxygenation and
  compliance, partial relief from PEEP, and increased urine output after volume expansion.
- Removed automatic norepinephrine from maintenance starts so MAP reflects
  the untreated anesthetic response.
- Removed cardiac-output scaling of propofol and remifentanil clearance and
  increased stroke volume during bradycardia. Renamed `hemo_model` value `"Su2023"` to `"Su"`.
- Made manual infusion rates the default, with optional TCI for targets,
  maintenance starts, and dosing guidance. Propofol and remifentanil rates use mcg/kg/min.
- Added fentanyl with TCI, midazolam, etomidate, ketamine, esmolol, labetalol,
  and glycopyrrolate with combined drug effects. Added lidocaine boluses and
  infusions to partly blunt stimulation.
- Revised induction scenarios for lidocaine, laryngoscopy stimulation, and
  maintenance MAP checks. Balanced induction gives fentanyl during
  preoxygenation and starts sevoflurane during rocuronium onset.
- Required end-tidal O₂ ≥ 90% for preoxygenation and unassisted breathing with
  train-of-four (TOF) ratio ≥ 90% before tube removal.
- Revised crisis instructions, fluid choices, and oxygen supply checks.
  Vasopressor objectives accept recovered pressure without another infusion.
- Added epinephrine relief of bronchospasm, positive-pressure relief of
  pharyngeal collapse, and reduced oxygen reserve after blood loss.
- Revised atrial-fibrillation circulation, ECG timing, and displayed heart
  rate. The baroreflex adjusts vascular resistance when a rhythm fixes the rate.
- Added PCV-VG, SIMV, patient triggering, flow cycling, VCV pause and pressure
  limits, and flow-dependent expiratory PEEP-valve resistance.
- Added tissue viscoelasticity, body-size compliance scaling, and synchronized
  inspiratory muscle effort with curare clefts. CO₂ controls apnea under anesthesia.
- Added airway traces, ventilator measurements, live loops, gas values, CSV
  fields, and alarms for high airway pressure and low ventilation, delivered
  tidal volume, and inspired O₂. End-tidal readings clear when exhaled breath
  detection stops; chest-impedance rate is labeled "RR imp".
- Corrected expired volume, mask leak, and tube obstruction. Bronchospasm
  reduces alveolar ventilation without reducing displayed minute ventilation.
- Separated `RespiratoryMechanics` from `AnesthesiaVentilator`. Extended
  `set_vent_settings` and moved PSV's apnea delay to `vent.apnea_backup_s`.

## 1.3 - 2026-09-30

- Replaced Qt with a browser interface shared by local Python and hosted
  Pyodide. `anasim` opens the local app and saves CSV files to `recordings/`.
  Removed PySide6 and pyqtgraph dependencies.
- Separated `set_vent_settings` from `set_vent_power`. Ventilation works in
  awake sessions and settings persist while power is off.
- Corrected step-size effects, trapped-gas pressure, and gas exchange timing.
  Gas exchange uses completed exhaled breaths; capnography works during PSV backup.
- Corrected NIBP cycle timing and retained the last reading during cuff cycles.
  SpO₂ requires a pulsatile signal.
- Made manual rates disable TCI and preserved central drug mass during PK scaling.
- Added CSV error reporting that preserves recorded data and sampling schedules.
- Stopped sessions at confirmed arrest when `end_on_cardiac_arrest` is enabled.
- Required ventilator startup after intubation and fresh gas reduction during
  total intravenous anesthesia (TIVA) maintenance. Baseline objectives show checked ranges.
- Applied arterial-line settings immediately and reduced waveform computation time.
- Added Python 3.14 support and removed pandas.

## 1.2 - 2026-09-24

- Added a baroreflex, revised epinephrine responses, and corrected the Li
  norepinephrine age covariate. Renamed epinephrine PK option `"Clutter"` to `"HealthyAdult"`.
- Added myocardial hypoxia and cardiac arrest with `end_on_cardiac_arrest`,
  `cardiac_arrest`, and `arrest_reason`.
- Added a central rocuronium effect site for respiratory muscle recovery before TOF recovery.
- Added upper-airway obstruction after loss of consciousness and lung and
  blood oxygen stores for preoxygenation and apneic oxygenation.
- Removed TCI rate spikes after resynchronization and capped propofol and
  remifentanil TCI at 1200 mL/hr.
- Corrected sevoflurane minimum alveolar concentration (MAC) at age 40 to 1.80%,
  cardiovascular timing, and BIS contribution during TIVA. Gas monitoring and
  scenario objectives use end-tidal MAC.
- Revised steady-state startup, ventilator waveforms, temperature, and stimulation responses.
- Preserved drug mass during PK scaling and used the published James lean
  body mass covariate for Schnider and Minto.
- Removed pediatric Oualha norepinephrine and epinephrine models and rejected
  configurations that select them.

## 1.1 - 2026-08-30

- Enforced patient limits across the API, CLI, and desktop setup. Added
  hemoglobin and hematocrit inputs and derived organ status from function factors.
- Averaged stimulation over each step and ended timed profiles after their final step.
- Added arterial waveforms with shared ECG and pleth timing and catheter
  dynamics. Made cardiac waveforms and history independent of step size and
  removed duplicate pressure, timing, and display calculations.
- Added a clinical guide and first-session instructions to the README.
- Revised interface labels, fixed clipped setup content, simplified
  cancellation, and removed the duplicate screenshot launcher.
- Added Ruff configuration and CI linting.

## 1.0 - 2026-07-17

- Integrated cardiovascular, respiratory, pharmacologic, ventilator, fluid,
  and temperature simulation with documented adaptations of published models.
- Added an operating-room monitor, guided scenarios, a headless runner, and
  CSV recording with pulse-oximeter lag and signal validity.
- Supported adults aged 18 to 70 years.
- Added Python packaging, continuous integration, and automated PyPI publishing.
