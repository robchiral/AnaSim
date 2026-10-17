# Changelog

## Unreleased

- Added plasma and red-cell buffering with live pH and bicarbonate, and made
  oxygen affinity depend on pH and PCO₂.
- Replaced prescribed apneic CO₂ slopes with CO₂ transport through lung,
  blood, tissue, and extracellular stores, so blood flow and stored CO₂ shape
  PaCO₂ and EtCO₂.
- Added ephedrine, atropine, succinylcholine, hydromorphone, and morphine.
- Replaced the TOF value with a TOF and single-twitch monitor.
- Added hypoxic breathing feedback, sustained hypoxic ventilatory decline, and
  anesthetic attenuation.
- Made recurrent atelectatic shunt depend on alveolar inert gas separately from
  bulk compliance, with oxygen conserved as shunt changes.
- Made the lung shrink as it absorbs gas behind an obstructed airway, so PaO₂
  falls more slowly after preoxygenation and absorption atelectasis develops.
- Added propofol venous pooling and resistance changes, and preload-dependent
  phenylephrine recruitment and afterload responses.

## 1.6 - 2026-10-14

- Adopted the AnaSim noncommercial license. Personal use and qualifying
  noncommercial institutional use are free. Commercial use and hosting outside
  the institutional exemption require a paid license. Earlier releases retain
  their MIT terms.
- Added recorded-case VitalDB comparisons for TIVA and sevoflurane.
- Changed the validation-guide link to an absolute URL.
- Added residual neuromuscular block to hypnotic airway collapse.
- Added crystalloid distribution and pressure- and age-dependent elimination.
- Updated oxygen dissociation and temperature-dependent affinity while
  conserving oxygen stores.
- Reduced reflex and beta-agonist tachycardia and cardiovagal gain with age.
- Lowered the propofol concentration for loss of response with age, so older
  patients lose responsiveness at a higher BIS. Midazolam and etomidate keep
  only their own age effects.
- Held arterial compliance constant below age 40, matching pulse pressure by age.
- Tracked red-cell volume separately from hemoglobin, so transfused hemoglobin
  no longer depends on the recipient's Hb/Hct ratio.
- Limited atelectatic shunt to about 10% of cardiac output; lost aeration still
  lowers lung volume and compliance.
- Separated HR and MAP responses to stimulation from movement tolerance, so
  surgery raises MAP and HR during maintenance and opioids blunt the intubation
  response. Laryngospasm can follow only the first 40 s of laryngoscopy.
- Placed the apneic threshold below the resting PaCO₂ set by the drugs, so
  patients no longer breathe over VCV during balanced maintenance.
- Blunted hemorrhage tachycardia under anesthesia.
- Held the pharynx open by mean airway pressure over recent breaths during mask
  ventilation, so obstruction no longer alternates with unassisted breaths.
- Required MAP 20% below its crisis-start value for shock recognition in the
  hemorrhage and sepsis scenarios.
- Set the baseline A-a gradient to sea-level means by age.
- Rejected negative or non-finite hemorrhage rates.
- Corrected sevoflurane cardiovascular equilibration, pulmonary transit delays,
  and CO₂ washout across time steps.
- Isolated alarm flags in returned state copies from the live simulation.
- Restricted rocuronium target-controlled infusion (TCI) to free plasma targets.
  Decay predictions include neuromuscular kinetics, sugammadex, and inhaled agents.
- Scaled oxygen consumption, CO₂ production, and dead space with body size.
  `baseline_vt` defaults to the volume that holds resting PaCO₂ at 40 mmHg.
- Applied the sevoflurane-nitrous oxide interaction at awakening and included
  nitrous oxide in laryngoscopy tolerance.
- Limited blood-volume scaling to the vascular part of each drug's central
  volume. Renal clearance of milrinone, glycopyrrolate, and rocuronium follows
  renal perfusion.
- Updated pharmacokinetic (PK) covariates after small hemodynamic changes and
  resynchronized active TCI controllers. `get_predicted_csht` accepts any registry drug.
- Replaced rocuronium and sugammadex PK/PD with Kleijn 2011's integrated model.
- Reduced propofol and remifentanil clearances with blood loss, reversed by
  volume restoration.
- Added reciprocal propofol-midazolam PK interactions.
- Recalibrated ketamine to raise HR and cardiac output about 30% with SVR
  unchanged, added a separate cardiovascular effect site, and included ketamine
  in laryngoscopy tolerance and baroreflex depression.
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
  bolus calibration, reducing early pressure dips and large-bolus clipping.
- Adopted Bouillon propofol and Olofsen opioid ventilatory depression, acting
  on alveolar ventilation.
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
  accounts for endogenous secretion and propofol-dependent clearance.
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
