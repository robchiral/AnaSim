# Changelog

## Unreleased

- Separated the spontaneous-breathing APL valve (0 to 70 cmH₂O, default open)
  from the manual bag peak pressure limit (5 to 70 cmH₂O, default 20 cmH₂O).
- Added an optional 37 °C IV fluid and blood warmer. Unwarmed fluids and
  refrigerated packed red cells now cool core temperature.
- Extended `renal_function` down to 0.05, and derived it from
  `creatinine_clearance_ml_min` when `renal_function` is left at 1.0.
- Pregnancy now lowers volatile MAC by up to 28% and expands central volumes for
  IV drugs. Above BMI 27, central volumes and clearances for most IV drugs now
  scale with lean-equivalent weight.
- Core hypothermia now slows non-renal IV drug clearance and transport,
  strengthens rocuronium block, slows ventilation and BIS, and lowers
  defibrillation success below 30.5 °C. Active sweating limits fever, and
  shivering raises heart rate and cardiac output while stopping during arrest
  or below 29 °C.
- Added graded postanoxic cortical injury (suppressing EEG and BIS after 4
  equivalent anoxic minutes) and brainstem respiratory injury (from 5 to 10
  minutes), while retaining the 5-minute permanent coma cutoff. Cerebral
  perfusion now accounts for venous pressure, chronic hypertension, metabolic
  demand, and CPR flow limits.
- Improved shock and heart failure hemodynamics with a 48% unstressed
  blood-volume floor during hemorrhage, compensation for acute normovolemic
  anemia, acidemic vasodilation, heart failure unloading with inotropes, and
  amiodarone rate control in atrial fibrillation.
- Improved resuscitation hemodynamics with filling-dependent CPR pulses,
  vasoplegic PEA, and stricter requirements to restart primary asystole.
- Switched ephedrine to a two-compartment IV model, removed the positive direct
  opioid stroke-volume slope, and prevented malignant hyperthermia from
  resolving while triggering agents remain active.
- Added steep opioid suppression of hypoxic ventilatory drive and overdose
  apnea, separated trapped gas in COPD from the exchangeable oxygen store, and
  scaled airway closing pressure with BMI.
- Added post-transfusion alkalosis and slowed intravenous calcium chloride
  delivery after its initial central pass.
- Added dedicated arrest and lethal rhythm alarms (`Asystole`, `VFIB`, `VTACH`,
  `Arrest`) that highlight the heart-rate display with a rhythm banner, break
  prior alarm silence, and play a faster, higher-pitched cadence, along with
  core temperature alarms (35.5 to 38.0 °C).
- Added perfusion-dependent finger SpO₂ transit delay, blanked SpO₂ and pleth
  during CPR, averaged NIBP across the deflation sweep with variability in
  atrial fibrillation, added capnography peak hold and slow-rate timeouts, and
  rounded displayed TOF ratios down to whole percentages.
- Updated case starting FiO₂, debrief targets, intubation blocking during severe
  laryngospasm, and succinylcholine warnings in MH or hyperkalemia, and kept the
  Trends panel open after case lock.
- Sped up high-speed web playback (about four times faster at 50x), maintenance
  startup, and per-step execution across blood chemistry, inhaled uptake,
  hemodynamics, and monitors.
- Expanded VitalDB comparisons across the full adult patient domain with
  preoperative labs and 5-fold patient-grouped cross-validation, and
  recalibrated respiratory compliance, propofol venous pooling kinetics,
  direct opioid vasodilation, and sevoflurane heart-rate depression.
- Scaled atropine and glycopyrrolate heart-rate responses with age, removed
  acidemic epinephrine beta-2 inversion, preserved hypothermic CPR cerebral
  protection, and added non-perfusing local anesthetic arrhythmias with
  fat-free-mass lipid scaling.
- Separated smooth extubation airway-reflex tolerance from laryngoscopy,
  combined non-drug coma into airway tolerance, conserved volatile tissue mass
  across temperature changes, and slowed rocuronium washout under sevoflurane
  without delaying bolus onset.
- Added metabolic alkalosis hypoventilation, term-pregnancy compliance
  reduction, exogenous lactate and citrate oxidation CO₂, volume-expansion
  natriuresis, ketamine redistribution sparing, and opioid vasoconstriction
  threshold shifts.
- Kept end-tidal O₂ and agent readings live during ventilated zero-CO₂ arrest,
  enforced the 60-second post-tetanic neuromuscular lockout across modes,
  blanked displayed HR during ventricular fibrillation, and broke active alarm
  silence when lethal rhythm or oxygen-supply alarms recur after clearing.

## 1.9 - 2026-11-19

- Extended supported adult ages to 85 years and sped up high-speed simulation
  by about four times.
- Added sodium, potassium, chloride, and ionized calcium to blood gases, along
  with fluid-specific ion and albumin loads, 0.9% saline (causing hyperchloremic
  acidosis), calcium chloride, transfusion acid, potassium, and citrate loads,
  and renal compensation for sustained PaCO₂ changes.
- Linked electrolytes to cardiac physiology: hypocalcemia lowers contractility
  and vascular resistance, succinylcholine, MH, and acidemia raise plasma
  potassium, and hyperkalemia alters the ECG and can trigger arrhythmias or
  asystole unless protected by calcium.
- Added malignant hyperthermia with dantrolene, surface cooling, chest-wall
  rigidity, and a guided or independent case.
- Added body-composition, age, and organ-function scaling for morphine,
  hydromorphone, fentanyl, lidocaine, rocuronium, and inhaled-agent tissue
  volumes, plus maternal extracellular-water expansion in pregnancy.
- Recalibrated midazolam, etomidate, ketamine, remifentanil under inhaled
  anesthesia, central neuromuscular potentiation by sevoflurane and nitrous
  oxide, and propofol hypotension above age 65.
- Recalibrated epinephrine across age and dose, norepinephrine venous
  recruitment, septic blunting of catecholamines, atropine, ephedrine under
  adrenergic blockade, and intravenous amiodarone.
- Added an adjustable bag pressure limit (APL, default 20 cmH₂O) and facemask
  seal with leaks, finite circuit reservoir and bellows volume without room-air
  entry, nitrous oxide concentration and dilution effects with 3:1 oxygen
  proportioning, sevoflurane progression to apnea, extubation laryngospasm, and
  blocked awake intubation.
- Added age and anesthetic suppression of hypoxic ventilatory drive,
  CO₂-dependent cerebral blood-flow reserve and narcosis recovery, recalibrated
  respiratory muscle fatigue, and postoperative atelectasis clearance by
  surgical site and obesity.
- Updated HFrEF preload and pulmonary edema thresholds, transcapillary refill,
  ventriculo-arterial coupling, hypothermic bradycardia and VF below 30 to
  33 °C, VT and VF resuscitation responses, and lidocaine toxicity with lipid
  rescue.
- Switched propofol, remifentanil, fentanyl, and rocuronium TCI to clinical
  population-only pump models, updated infusion units (`mcg/kg/min`,
  `mg/kg/hr`, `mg/min`), removed the browser `concomitant_opioids` checkbox
  (still available via CLI or Python), capped `dt` at 2 s, and added
  `--record-interval 0`.
- Warmed hypothermic blood gases to 37 °C, rounded displayed numerics, broke
  audio silence on new or changed alarms, improved monitor behavior during
  arrest, apnea, and leaks, added Stage II extubation warnings, updated crisis
  cases, and fixed step-size, browser-throttling, and arrest-locking issues.

## 1.8 - 2026-11-06

- Added a Circulation tab with chest compressions, a 200 J defibrillator and
  synchronized cardioversion, fluids, and 20% lipid emulsion, plus a cardiac
  arrest case.
- Continued simulations through cardiac arrest (VF, pulseless VT, PEA, and
  asystole) until irreversible ischemic injury (about 11 minutes without flow),
  with failing escape rhythms and ischemic ST depression. Removed
  `end_on_cardiac_arrest`.
- Added severe cerebral hypoperfusion and hypoxemia effects (unresponsiveness,
  airway collapse, low BIS, apnea, and postanoxic coma) and CO₂ narcosis.
- Added arterial blood gas sampling in a Labs tab, labeled crystalloid as
  lactated Ringer's, and added lactate, standard base excess, acidemic cardiac
  depression, and metabolic ventilatory compensation.
- Added lidocaine toxicity (seizures, QRS widening, arrhythmias, and collapse)
  with lipid emulsion rescue, lean-mass lidocaine PK scaling, and amiodarone.
- Added poorly ventilated (low V/Q) lung units separate from atelectatic shunt,
  age-dependent airway closure under anesthesia, a `surgical_site` setting for
  postoperative closure, and respiratory muscle fatigue.
- Added bag ventilation rate and tidal volume settings, optional pulse-oximeter
  and alarm audio, independent case mode with debriefs, automatic crisis onset,
  case actions, and clinical safety warnings.
- Updated PK and hemodynamics so shed blood removes central drug, drug
  transport stops without circulation outside esterase metabolism, sevoflurane
  lowers cardiac output mainly through stroke volume, and fentanyl and
  lidocaine clearances match surgical data. Extended optional patient
  variability to remifentanil, fentanyl, sevoflurane, and morphine and
  hydromorphone respiratory rates.
- Started anaphylaxis and sepsis cases in established shock (with anaphylaxis
  plasma leak and bronchospasm recovery checks), required recent cuff readings
  without an arterial line, added intubation stimulation on every attempt, and
  sped up TCI, PK, and monitor updates.

## 1.7 - 2026-10-25

- Added optional `patient_variability` sampling for propofol PK and BIS
  (Eleveld 2018) and propofol and opioid vascular sensitivity (Su 2023).
- Added coronary disease (with ischemic ST depression), HFrEF and HFpEF, COPD,
  pregnancy with maternal positioning, and severe obesity up to 200 kg and
  BMI 60.
- Added ephedrine, atropine, succinylcholine, hydromorphone, and morphine,
  replaced the single TOF reading with a TOF and single-twitch monitor, and
  added `creatinine_clearance_ml_min` for rocuronium and sugammadex.
- Remodeled anesthetic hemodynamics on slow effect sites fitted to VitalDB
  inductions and published trials, adding propofol venous pooling, baseline-MAP
  sensitivity, opioid suppression of reflex tachycardia, remifentanil vagal
  bradycardia, sevoflurane HR effects, pressure-dependent arterial compliance,
  and surgical vasoconstriction.
- Added plasma and red-cell acid-base buffering (live pH and bicarbonate),
  oxygen affinity that depends on pH and PCO₂, and multi-compartment CO₂ stores
  replacing fixed apneic slopes.
- Added hypoxic ventilatory drive and decline, inert-gas-dependent recurrent
  atelectasis, gas-absorption lung shrinkage behind obstructed airways,
  residual-block airway collapse through full TOF recovery, and
  pressure-dependent crystalloid distribution.
- Recalibrated epinephrine, preload-dependent phenylephrine responses, ketamine
  HR, propofol loss of response, and stimulation responses under propofol,
  fentanyl, remifentanil, and sevoflurane.
- Improved setup layout, loading progress, and guided scenarios (prompt crisis
  treatment, TCI induction without an extra bolus, measured TOF, and residual
  block at emergence), and added clinical acceptance and VitalDB hemodynamic
  validation checks.

## 1.6 - 2026-10-14

- Adopted the AnaSim noncommercial license (free for personal and qualifying
  noncommercial institutional use; paid license for commercial use; earlier
  releases retain MIT).
- Added VitalDB recorded-case validation for TIVA and sevoflurane.
- Replaced rocuronium and sugammadex PK/PD with Kleijn 2011, added hemorrhage
  clearance scaling for propofol and remifentanil, added reciprocal
  propofol-midazolam PK interactions, and scaled vascular central volume and
  renal drug clearance with blood volume and renal perfusion.
- Restricted rocuronium TCI to free plasma targets, extended
  `get_predicted_csht` to all registry drugs, and resynchronized active TCI
  controllers after hemodynamic PK updates.
- Added pressure- and age-dependent crystalloid kinetics, separate red-cell
  volume tracking, updated temperature-dependent oxygen dissociation, and
  age-adjusted arterial compliance, reflexes, baseline A-a gradient, and
  propofol loss of response.
- Added residual-block airway collapse under hypnotics, mean-pressure pharyngeal
  splinting during mask ventilation, passive apneic lung refilling after
  obstruction relief, capped atelectatic shunt (~10%), and size-scaled
  metabolism and dead space (`baseline_vt` defaults to PaCO₂ 40 mmHg).
- Separated HR and MAP stimulation responses from movement tolerance, linked
  the apneic threshold to anesthetic resting PaCO₂, blunted hemorrhage
  tachycardia under anesthesia, recalibrated ketamine, included N₂O in
  awakening and laryngoscopy tolerance, and stabilized thermoregulation.
- Updated shock recognition in hemorrhage and sepsis cases (20% MAP drop),
  validated hemorrhage rates, isolated alarm state copies, and fixed sevoflurane
  equilibration, pulmonary transit, and CO₂ washout across step sizes.

## 1.5 - 2026-10-10

- Standardized core drug models on Eleveld 2018 (propofol, with
  `concomitant_opioids`), Eleveld 2017 (remifentanil), Li (norepinephrine),
  Ensinger (epinephrine), and Bouillon and Olofsen (responsiveness, BIS, and
  alveolar ventilatory depression).
- Removed legacy model selectors and alternative implementations
  (`simulation_speed`, `asa`, `pk_model_*`, `hemo_model`, `resp_model`,
  `bis_model`, `loc_model`), and organized subsystems under `engine.pk`,
  `engine.infusion_rates`, `engine.tci`, `engine.thermal`, `engine.anaphylaxis`,
  and `engine.sepsis`.
- Added VitalDB comparison scripts and reports for ventilation, capnography,
  gas exchange, and PEEP.
- Linked circuit oxygen uptake to lung oxygen stores, corrected apneic CO₂
  accumulation and circuit, vaporizer, and shunt gas conservation, preserved
  respiratory effort and impedance rate during obstruction, and derived
  intrathoracic pressure from airway pressure and muscle effort.
- Added PK substeps for mass conservation across larger steps, aligned TCI
  prediction pulses with update intervals, and initialized hemodynamic turnover
  and fast drug responses cleanly at maintenance startup.
- Updated hemorrhage venous recruitment and right atrial pressure, tapered
  urine output at low MAP, and blunted adrenergic responses in severe
  anaphylaxis.

## 1.4 - 2026-10-06

- Added pulmonary edema from fluid overload, with reduced oxygenation and
  compliance, partial relief from PEEP, and increased urine output after volume
  expansion.
- Removed automatic norepinephrine from maintenance starts, so MAP reflects the
  untreated anesthetic response.
- Removed cardiac-output scaling of propofol and remifentanil clearance, and
  increased stroke volume during bradycardia. Renamed the `hemo_model` value
  `"Su2023"` to `"Su"`.
- Made manual infusion rates the default, with optional TCI for targets,
  maintenance starts, and dosing guidance. Propofol and remifentanil rates use
  mcg/kg/min.
- Added fentanyl with TCI, midazolam, etomidate, ketamine, esmolol, labetalol,
  and glycopyrrolate, with combined drug effects. Added lidocaine boluses and
  infusions to partly blunt stimulation.
- Revised induction scenarios for lidocaine, laryngoscopy stimulation, and
  maintenance MAP checks. Balanced induction gives fentanyl during
  preoxygenation and starts sevoflurane during rocuronium onset.
- Required end-tidal O₂ of at least 90% for preoxygenation, and unassisted
  breathing with a TOF ratio of at least 90% before tube removal.
- Revised crisis instructions, fluid choices, and oxygen supply checks.
  Vasopressor objectives accept recovered pressure without another infusion.
- Added epinephrine relief of bronchospasm, relief of pharyngeal collapse by
  positive pressure, and reduced oxygen reserve after blood loss.
- Revised the circulation, ECG timing, and displayed heart rate in atrial
  fibrillation. The baroreflex adjusts vascular resistance when a rhythm fixes
  the rate.
- Added PCV-VG, SIMV, patient triggering, flow cycling, VCV pause and pressure
  limits, and a PEEP valve whose resistance depends on expiratory flow.
- Added tissue viscoelasticity, scaling of compliance with body size, and
  synchronized inspiratory muscle effort with curare clefts. CO₂ controls apnea
  under anesthesia.
- Added airway traces, ventilator measurements, live loops, gas values, CSV
  fields, and alarms for high airway pressure, low ventilation, low delivered
  tidal volume, and low inspired O₂. End-tidal readings clear when detection of
  exhaled breath stops, and the chest-impedance rate is labeled "RR imp".
- Corrected expired volume, mask leak, and tube obstruction. Bronchospasm
  reduces alveolar ventilation without reducing displayed minute ventilation.
- Separated `RespiratoryMechanics` from `AnesthesiaVentilator`. Extended
  `set_vent_settings` and moved PSV's apnea delay to `vent.apnea_backup_s`.

## 1.3 - 2026-09-30

- Replaced Qt with a browser interface shared by local Python and hosted
  Pyodide. `anasim` opens the local app and saves CSV files to `recordings/`.
  Removed the PySide6 and pyqtgraph dependencies.
- Separated `set_vent_settings` from `set_vent_power`. Ventilation works in
  awake sessions, and settings persist while power is off.
- Corrected step-size effects, trapped-gas pressure, and gas exchange timing.
  Gas exchange uses completed exhaled breaths, and capnography works during PSV
  backup.
- Corrected NIBP cycle timing and kept the last reading during cuff cycles. SpO₂
  requires a pulsatile signal.
- Manual rates now disable TCI, and PK scaling preserves central drug mass.
- Added CSV error reporting that preserves recorded data and sampling schedules.
- Stopped sessions at confirmed arrest when `end_on_cardiac_arrest` is enabled.
- Required ventilator startup after intubation and a reduction in fresh gas
  during TIVA maintenance. Baseline objectives show the ranges they check.
- Applied arterial-line settings immediately and reduced waveform computation
  time.
- Added Python 3.14 support and removed pandas.

## 1.2 - 2026-09-24

- Added a baroreflex, revised epinephrine responses, and corrected the age
  covariate in the Li norepinephrine model. Renamed the epinephrine PK option
  `"Clutter"` to `"HealthyAdult"`.
- Added myocardial hypoxia and cardiac arrest with `end_on_cardiac_arrest`,
  `cardiac_arrest`, and `arrest_reason`.
- Added a central rocuronium effect site, so respiratory muscles recover before
  TOF.
- Added upper airway obstruction after loss of consciousness, and oxygen stores
  in the lung and blood for preoxygenation and apneic oxygenation.
- Removed TCI rate spikes after resynchronization and capped propofol and
  remifentanil TCI at 1200 mL/hr.
- Corrected sevoflurane MAC at age 40 to 1.80%, its cardiovascular timing, and
  its BIS contribution during TIVA. Gas monitoring and scenario objectives use
  end-tidal MAC.
- Revised steady-state startup, ventilator waveforms, temperature, and
  stimulation responses.
- Preserved drug mass during PK scaling and used the published James lean body
  mass covariate for Schnider and Minto.
- Removed the pediatric Oualha norepinephrine and epinephrine models and
  rejected configurations that select them.

## 1.1 - 2026-08-30

- Enforced patient limits across the API, CLI, and desktop setup. Added
  hemoglobin and hematocrit inputs and derived organ status from function
  factors.
- Averaged stimulation over each step and ended timed profiles after their final
  step.
- Added arterial waveforms with shared ECG and pleth timing and catheter
  dynamics. Made cardiac waveforms and history independent of step size and
  removed duplicate pressure, timing, and display calculations.
- Added a clinical guide and first-session instructions to the README.
- Revised interface labels, fixed clipped setup content, simplified
  cancellation, and removed the duplicate screenshot launcher.
- Added Ruff configuration and CI linting.

## 1.0 - 2026-07-17

- Integrated cardiovascular, respiratory, pharmacologic, ventilator, fluid, and
  temperature simulation with documented adaptations of published models.
- Added an operating-room monitor, guided scenarios, a headless runner, and CSV
  recording with pulse-oximeter lag and signal validity.
- Supported adults aged 18 to 70 years.
- Added Python packaging, continuous integration, and automated PyPI publishing.
