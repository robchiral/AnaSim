# Changelog

## Unreleased

- Added venovenous ECMO, venoarterial ECMO, and cardiopulmonary bypass support
  with membrane gas exchange, regional differential oxygenation, passive bypass
  reservoir drainage, active heat exchange, circuit drug sequestration, and
  site-specific blood gas sampling.
- Added unfractionated heparin and protamine kinetics, point-of-care activated
  clotting time (ACT), protamine hemodynamic reactions, an ARDS shunt event,
  and guided or independent extracorporeal and anticoagulation cases.
- Added an optional 37 °C IV fluid and blood warmer so unwarmed fluids and
  refrigerated packed red cells cool core temperature.
- Added vaporizer fill-level tracking and refill controls.
- Unified the APL valve (0 to 70 cmH₂O, default open) across manual bag
  ventilation and spontaneous breathing.
- Extended `renal_function` down to 0.05 and derived it from
  `creatinine_clearance_ml_min` when `renal_function` is left at 1.0.
- Updated pregnancy, obesity, and liver-disease scaling across drug kinetics,
  volatile MAC, respiratory compliance, baseline metabolism, and lactate pools.
- Expanded hypothermia and thermoregulation across drug clearance, volatile
  mass conservation, ventilation, BIS, defibrillation success, sweating,
  shivering (stopping at or below 31 °C or during arrest), and vasoconstriction
  thresholds.
- Added graded postanoxic cortical and brainstem respiratory injury alongside
  permanent coma, and updated cerebral perfusion for venous pressure, chronic
  hypertension, metabolic demand, and CPR flow limits.
- Improved shock, anemia, acidemia, and heart-failure hemodynamics, including
  venous-recruitment and pressor limits during severe hemorrhage and an
  age-dependent sinus heart-rate ceiling.
- Recalibrated cardiovascular drugs, including age-scaled antimuscarinics,
  labetalol alpha-blockade, two-compartment IV ephedrine, amiodarone rate
  control in atrial fibrillation, opioid stroke-volume effects, and acidemic
  epinephrine responses.
- Improved cardiac arrest and resuscitation physiology, including filling-dependent
  CPR pulses, vasoplegic PEA, primary asystole recovery, local anesthetic
  arrhythmias and lipid dosing, passive left atrial pressure equilibration, and
  zero-flow arterial blood gas isolation.
- Updated airway reflexes, neuromuscular responses, and malignant hyperthermia
  remission, including separate extubation tolerance, slower rocuronium
  washout under sevoflurane, and succinylcholine Phase II central weakness.
- Improved respiratory control, COPD dead space and trapped-gas modeling,
  BMI-scaled airway closure, empty-bellows PEEP loss without fresh gas flow,
  and step-size coupling between mechanics and gas exchange.
- Updated fluid, electrolyte, and transfusion physiology, including natriuresis,
  hypovolemic suppression of crystalloid excretion, post-transfusion alkalosis,
  lactate and citrate oxidation CO₂, and calcium chloride distribution.
- Added dedicated arrest and lethal-rhythm alarms (`Asystole`, `VFIB`, `VTACH`,
  `Arrest`) that highlight the HR tile, display the rhythm in the alarm banner,
  and break silence on recurrence (along with `O2Supply`).
- Upgraded patient monitors with core temperature alarms, perfusion-dependent
  SpO₂ transit delay, CPR pleth and VF HR blanking, deflation-averaged NIBP,
  capnogram Phase III steepening and slow-rate timeouts, a 60-second PTC
  lockout across modes, whole-percent TOF rounding, and spirometry loop
  clearing on disconnect.
- Refined clinical cases and safety checks for unparalyzed intubation,
  succinylcholine warnings in malignant hyperthermia or hyperkalemia, sepsis
  vasopressor crediting, debrief targets, post-lock Trends viewing, and
  defibrillator sync reset.
- Expanded VitalDB comparisons across the full adult domain with preoperative
  labs, multiprocessing, and 5-fold patient-grouped cross-validation, and
  consolidated clinical benchmarks and invariants under `tests/`.
- Sped up high-speed web playback (about four times faster at 50x), maintenance
  startup, and per-step execution across the engine, monitors, and browser
  rendering.

## 1.9 - 2026-11-19

- Extended the supported adult age range to 85 years.
- Sped up high-speed simulation by about four times.
- Added sodium, potassium, chloride, and ionized calcium to blood gases, along
  with 0.9% saline, calcium chloride, fluid-specific ion and albumin loads,
  transfusion acid and citrate loads, and renal compensation for sustained
  PaCO₂ changes.
- Linked plasma calcium and potassium to cardiac contractility, vascular tone,
  ECG changes, arrhythmias, and hyperkalemic asystole.
- Added malignant hyperthermia with dantrolene, surface cooling, chest-wall
  rigidity, and a guided or independent case.
- Added body-composition, age, and organ-function scaling across intravenous
  and inhaled drug kinetics, plus maternal extracellular-water expansion in
  pregnancy.
- Recalibrated intravenous hypnotics, opioids, neuromuscular potentiation by
  inhaled agents, and elderly propofol hypotension.
- Recalibrated catecholamines, atropine, ephedrine under adrenergic blockade,
  and intravenous amiodarone, including septic blunting of vasopressors.
- Added an adjustable bag pressure limit (default 20 cmH₂O), facemask seal with
  leaks, finite circuit reservoir and bellows volume, nitrous oxide
  concentration and dilution effects with 3:1 oxygen proportioning, extubation
  laryngospasm, and blocked awake intubation.
- Updated respiratory control and atelectasis, including hypoxic drive
  suppression, CO₂ cerebral blood-flow reserve, respiratory muscle fatigue, and
  postoperative atelectasis clearance by surgical site and obesity.
- Updated heart-failure preload and pulmonary edema thresholds, transcapillary
  refill, and ventriculo-arterial coupling.
- Improved hypothermic bradycardia and VF onset, VT and VF resuscitation
  responses, and arrest recovery.
- Refined lidocaine toxicity and lipid emulsion rescue.
- Switched TCI pumps to clinical population-only models and updated infusion
  units (`mcg/kg/min`, `mg/kg/hr`, `mg/min`).
- Removed the browser `concomitant_opioids` checkbox (still available via CLI
  or Python), capped `dt` at 2 s, and added `--record-interval 0`.
- Warmed hypothermic blood gases to 37 °C, rounded displayed numerics, broke
  audio silence on new or changed alarms, and improved monitor behavior during
  arrest, apnea, and leaks.
- Added Stage II extubation warnings and updated crisis cases.
- Fixed step-size, browser-throttling, and arrest-locking issues.

## 1.8 - 2026-11-06

- Added a Circulation tab with chest compressions, a 200 J defibrillator and
  synchronized cardioversion, fluids, and 20% lipid emulsion, along with a
  cardiac arrest case.
- Continued simulations through cardiac arrest (VF, pulseless VT, PEA, and
  asystole) until irreversible ischemic injury, with failing escape rhythms and
  ischemic ST depression; removed `end_on_cardiac_arrest`.
- Added cerebral hypoperfusion, severe hypoxemia, postanoxic coma, and CO₂
  narcosis effects on responsiveness, airway tone, BIS, and breathing.
- Added arterial blood gas sampling in a Labs tab, labeled crystalloid as
  lactated Ringer's, and added lactate, standard base excess, acidemic cardiac
  depression, and metabolic ventilatory compensation.
- Added lidocaine toxicity (seizures, QRS widening, arrhythmias, and collapse)
  with lipid emulsion rescue, lean-mass lidocaine PK scaling, and amiodarone.
- Added low-V/Q lung units separate from atelectatic shunt, age-dependent
  airway closure under anesthesia, a `surgical_site` setting for postoperative
  closure, and respiratory muscle fatigue.
- Added bag ventilation rate and tidal volume settings and optional
  pulse-oximeter and alarm audio.
- Added independent case mode with debriefs, automatic crisis onset, case
  actions, and clinical safety warnings.
- Updated drug kinetics and hemodynamics so shed blood removes central drug,
  non-esterase transport halts without circulation, sevoflurane lowers output
  mainly through stroke volume, and surgical clearances match published data.
- Extended optional patient variability to remifentanil, fentanyl, sevoflurane,
  and morphine and hydromorphone respiratory depression.
- Started the anaphylaxis and sepsis cases in established shock and applied
  intubation stimulation on every attempt.
- Required recent cuff readings in cases when no arterial line is present.
- Sped up TCI, pharmacokinetic, and monitor updates.

## 1.7 - 2026-10-25

- Added optional `patient_variability` sampling for propofol PK and BIS and
  propofol and opioid vascular sensitivity.
- Added coronary disease with ischemic ST depression, HFrEF and HFpEF, COPD,
  pregnancy with maternal positioning, and severe obesity up to 200 kg and
  BMI 60.
- Added ephedrine, atropine, succinylcholine, hydromorphone, and morphine, and
  added `creatinine_clearance_ml_min` for rocuronium and sugammadex.
- Replaced the single TOF reading with a combined TOF and single-twitch
  monitor.
- Remodeled anesthetic hemodynamics on slow effect sites fitted to VitalDB
  inductions and published trials, covering propofol venous pooling,
  baseline-MAP sensitivity, opioid reflex suppression, sevoflurane heart-rate
  effects, arterial compliance, and surgical vasoconstriction.
- Added plasma and red-cell acid-base buffering, pH- and PCO₂-dependent oxygen
  affinity, and multi-compartment CO₂ stores.
- Added hypoxic ventilatory drive and decline, recurrent atelectasis,
  obstructed-airway gas absorption, residual-block airway collapse, and
  pressure-dependent crystalloid distribution.
- Recalibrated epinephrine, preload-dependent phenylephrine responses, ketamine
  heart rate, propofol loss of response, and surgical stimulation responses.
- Improved setup layout, loading progress, and guided scenarios.
- Added clinical acceptance tests and VitalDB hemodynamic validation checks.

## 1.6 - 2026-10-14

- Adopted the AnaSim noncommercial license (free for personal and qualifying
  noncommercial institutional use; paid license for commercial use; earlier
  releases retain MIT).
- Added VitalDB recorded-case validation for TIVA and sevoflurane.
- Replaced rocuronium and sugammadex PK/PD with the Kleijn model, added
  hemorrhage clearance scaling and reciprocal propofol-midazolam interactions,
  and scaled central volume and renal clearance with blood volume and renal
  perfusion.
- Restricted rocuronium TCI to free plasma targets, extended
  `get_predicted_csht` to all registry drugs, and resynchronized active TCI
  controllers after hemodynamic PK updates.
- Added pressure- and age-dependent crystalloid kinetics, separate red-cell
  volume tracking, and temperature-dependent oxygen dissociation.
- Added age scaling for arterial compliance, baroreflexes, baseline A-a
  gradient, and propofol sensitivity.
- Added residual-block airway collapse under hypnotics, positive-pressure
  pharyngeal splinting, passive apneic refilling after obstruction relief,
  capped atelectatic shunt, and size-scaled metabolism and dead space
  (`baseline_vt` defaults to PaCO₂ 40 mmHg).
- Separated HR and MAP stimulation responses from movement tolerance, linked
  the apneic threshold to anesthetic PaCO₂, recalibrated ketamine, and included
  N₂O in awakening and laryngoscopy tolerance.
- Blunted hemorrhage tachycardia under anesthesia.
- Stabilized thermoregulation.
- Updated hemorrhage and sepsis shock thresholds and validated hemorrhage rates.
- Isolated alarm state copies and fixed step-size sensitivity in sevoflurane
  equilibration, pulmonary transit, and CO₂ washout.

## 1.5 - 2026-10-10

- Standardized core drug models on Eleveld (propofol, with
  `concomitant_opioids`, and remifentanil), Li (norepinephrine), Ensinger
  (epinephrine), and Bouillon and Olofsen (responsiveness, BIS, and ventilatory
  depression).
- Removed legacy model selectors (`simulation_speed`, `asa`, `pk_model_*`,
  `hemo_model`, `resp_model`, `bis_model`, `loc_model`) and reorganized engine
  subsystems under dedicated attributes.
- Added VitalDB comparison scripts and reports for ventilation, capnography,
  gas exchange, and PEEP.
- Linked circuit oxygen uptake to lung oxygen stores, corrected apneic CO₂
  accumulation and gas conservation, preserved respiratory effort and impedance
  rate during obstruction, and derived intrathoracic pressure from airway
  pressure and muscle effort.
- Added PK substeps for mass conservation across larger steps, aligned TCI
  prediction pulses with update intervals, and improved steady-state
  initialization for fast cardiovascular responses.
- Updated hemorrhage venous recruitment and right atrial pressure, tapered
  urine output at low MAP, and blunted adrenergic responses in severe
  anaphylaxis.

## 1.4 - 2026-10-06

- Added pulmonary edema from fluid overload, with reduced oxygenation and
  compliance, partial relief from PEEP, and increased urine output after volume
  expansion.
- Removed automatic norepinephrine from maintenance starts so MAP reflects the
  untreated anesthetic response.
- Removed cardiac-output scaling of propofol and remifentanil clearance and
  increased stroke volume during bradycardia.
- Renamed the `hemo_model` value `"Su2023"` to `"Su"`.
- Made manual infusion rates the default (with propofol and remifentanil in
  mcg/kg/min) and kept TCI optional for targets, maintenance starts, and dosing
  guidance.
- Added fentanyl with TCI, midazolam, etomidate, ketamine, esmolol, labetalol,
  glycopyrrolate, and lidocaine with combined drug effects.
- Revised induction, emergence, and crisis scenarios for preoxygenation EtO₂,
  lidocaine and laryngoscopy timing, TOF and breathing checks before
  extubation, fluid selection, oxygen-supply checks, and vasopressor goals.
- Added epinephrine relief of bronchospasm, positive-pressure relief of
  pharyngeal collapse, and reduced oxygen reserve after blood loss.
- Revised atrial fibrillation hemodynamics, ECG timing, and baroreflex vascular
  compensation when a rhythm fixes the heart rate.
- Added PCV-VG, SIMV, patient triggering, flow cycling, VCV pause and pressure
  limits, flow-dependent PEEP valve resistance, tissue viscoelasticity,
  body-size compliance scaling, and synchronized inspiratory effort with curare
  clefts.
- Added airway waveforms, live spirometry loops, ventilator measurements, CSV
  fields, and respiratory alarms, with end-tidal clearing on apnea and
  chest-impedance rate labeled "RR imp".
- Corrected expired volume, mask leak, tube obstruction, and alveolar
  ventilation during bronchospasm.
- Separated `RespiratoryMechanics` from `AnesthesiaVentilator`, extended
  `set_vent_settings`, and moved PSV's apnea delay to `vent.apnea_backup_s`.

## 1.3 - 2026-09-30

- Replaced the Qt desktop interface with a browser interface shared by local
  Python and hosted Pyodide, saving CSV logs to `recordings/` and removing the
  PySide6 and pyqtgraph dependencies.
- Separated `set_vent_settings` from `set_vent_power` so ventilation works in
  awake sessions and settings persist while power is off.
- Corrected step-size effects, trapped-gas pressure, gas exchange timing on
  completed exhaled breaths, and capnography during PSV backup.
- Corrected NIBP cycle timing and reading hold, and required a pulsatile signal
  for SpO₂.
- Disabled TCI automatically when manual infusion rates are set, and preserved
  central drug mass during PK scaling.
- Added CSV error reporting that preserves recorded data and sampling schedules.
- Stopped sessions at confirmed arrest when `end_on_cardiac_arrest` is enabled.
- Updated scenarios to require ventilator startup after intubation, reduced
  fresh gas flow during TIVA maintenance, and visible baseline objective ranges.
- Applied arterial-line settings immediately and sped up waveform generation.
- Added Python 3.14 support and removed the pandas dependency.

## 1.2 - 2026-09-24

- Added a baroreflex, revised epinephrine responses, corrected the age
  covariate in the norepinephrine model, and renamed the epinephrine PK option
  `"Clutter"` to `"HealthyAdult"`.
- Added myocardial hypoxia and cardiac arrest with `end_on_cardiac_arrest`,
  `cardiac_arrest`, and `arrest_reason`.
- Added a central rocuronium effect site so respiratory muscles recover before
  peripheral TOF.
- Added upper airway obstruction after loss of consciousness and lung and blood
  oxygen stores for preoxygenation and apneic oxygenation.
- Removed TCI rate spikes after resynchronization and capped propofol and
  remifentanil TCI at 1200 mL/hr.
- Corrected sevoflurane MAC at age 40 to 1.80%, cardiovascular timing, and BIS
  contribution during TIVA, and switched gas monitoring and objectives to
  end-tidal MAC.
- Revised steady-state startup, ventilator waveforms, temperature, and
  stimulation responses.
- Preserved drug mass during PK scaling and switched Schnider and Minto to the
  published James lean body mass covariate.
- Removed the pediatric Oualha catecholamine models and rejected configurations
  that select them.

## 1.1 - 2026-08-30

- Enforced patient limits across the API, CLI, and desktop setup, added
  hemoglobin and hematocrit inputs, and derived organ status from function
  factors.
- Averaged stimulation over each step and ended timed profiles after their
  final step.
- Added arterial waveforms with shared ECG and pleth timing and catheter
  dynamics, and made cardiac waveforms and history independent of step size.
- Added a clinical guide and first-session instructions to the README.
- Simplified setup and cancellation controls in the interface and removed the
  duplicate screenshot launcher.
- Added Ruff configuration and CI linting.

## 1.0 - 2026-07-17

- Integrated cardiovascular, respiratory, pharmacologic, ventilator, fluid, and
  temperature simulation using documented adaptations of published models.
- Added an operating-room monitor, guided scenarios, a headless runner, and CSV
  recording with pulse-oximeter lag and signal validity.
- Supported adults aged 18 to 70 years.
- Added Python packaging, continuous integration, and automated PyPI publishing.
