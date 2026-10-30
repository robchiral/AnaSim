# Changelog

## Unreleased

- Added optional pulse oximeter tones and alarm audio.
- Sped up TCI prediction rebuilds, drug kinetics, and monitor updates.
- Added poorly ventilated (low V/Q) lung units, separate from atelectatic shunt.
  Under anesthesia, airway closure lowers PaO₂ more in older patients, and PEEP
  of about 6 cmH₂O prevents most of this fall. COPD and bronchospasm now cause
  low V/Q hypoxemia, which responds to added oxygen more than shunt does.
- Lowered atelectatic shunt to about 5% after a preoxygenated induction.
  Maintenance sessions start with this atelectasis, and it persists after
  emergence.

## 1.7 - 2026-10-25

- Added loading progress and made the patient setup controls more compact.
- Added optional patient variability. Each session samples propofol kinetics and
  BIS sensitivity (Eleveld 2018, with standard deviations halved to match VitalDB)
  and blood pressure sensitivity to propofol and opioids (Su 2023).
- Corrected guided scenarios to accept prompt crisis treatment across steps,
  allow TCI induction without an extra propofol bolus, and use measured TOF
  counts and ratios. Emergence starts with residual rocuronium block.
- Moved anesthetic cardiovascular effects to hemodynamic effect sites that act
  on slow tone, so pressure falls within minutes of induction while compensation
  stays slow.
- Slowed propofol's effect on pressure to about 10 minutes, and made propofol
  alone lower pressure about half as much as propofol with an opioid.
- Added venous pooling as propofol removes sympathetic tone, so induction lowers
  cardiac output and filling pressure.
- Made pressure fall more after propofol in patients with a higher pre-induction
  MAP.
- Fitted propofol-opioid and sevoflurane induction to VitalDB recordings of
  routine practice, with published induction trials as limits.
- Added an HR effect for sevoflurane, and opioids no longer strengthen its
  effect on pressure.
- Made opioids suppress reflex tachycardia, and added remifentanil's vagal
  bradycardia.
- Made arterial compliance rise as pressure falls, so pulse pressure narrows in
  hypotension.
- Propofol alone no longer blunts the cardiovascular response to stimulation.
- Routine remifentanil doses now leave most of the intubation response.
- Fentanyl now lowers the sevoflurane concentration that blocks the response to
  incision.
- Added surgical vasoconstriction that builds over minutes. Maintenance sessions
  and intraoperative crises start during surgery, and emergence starts after it.
- Septic tachycardia now follows severity under anesthesia, and septic shock
  removes surgical vasoconstriction.
- Slowed crystalloid distribution at low arterial pressure.
- Recalibrated epinephrine, phenylephrine, the rise in HR with ketamine, and the
  propofol concentration for loss of response.
- Added acceptance tests for standard teaching cases, replays of published
  induction and intubation studies, and VitalDB hemodynamic comparisons.
- Added coronary disease, heart failure with reduced or preserved EF, COPD, and
  pregnancy, with maternal positioning and reversible ischemic ST depression.
- Extended patient inputs to 200 kg and BMI 60, with respiratory mechanics for
  severe obesity.
- Added plasma and red-cell buffering with live pH and bicarbonate, and made
  oxygen affinity depend on pH and PCO₂.
- Replaced prescribed apneic CO₂ slopes with CO₂ transport through lung, blood,
  tissue, and extracellular stores, so blood flow and stored CO₂ determine PaCO₂
  and EtCO₂.
- Added ephedrine, atropine, succinylcholine, hydromorphone, and morphine.
  Morphine and hydromorphone have separate effects on respiratory rate and on
  ventilatory drive and CO₂ response.
- Replaced the TOF value with a TOF and single-twitch monitor.
- Added measured creatinine clearance for rocuronium and sugammadex.
- Applied renal perfusion only to the renal clearance that remains after renal
  impairment.
- Airway collapse from residual block now lasts until TOF fully recovers, which
  matches the change in pharyngeal closing pressure at a normalized TOF of 0.98.
- Removed the extra halving of spontaneous rates between 2 and 4 breaths/min.
- Added hypoxic breathing feedback and hypoxic ventilatory decline, both reduced
  by anesthetics.
- Made recurrent atelectatic shunt depend on alveolar inert gas, separately from
  bulk compliance, with oxygen conserved as shunt changes.
- Made the lung shrink as it absorbs gas behind an obstructed airway, so PaO₂
  falls more slowly after preoxygenation and absorption atelectasis develops.
- Added propofol's effects on stressed volume and venous resistance, and made
  phenylephrine's venous recruitment and afterload effects depend on preload.

## 1.6 - 2026-10-14

- Adopted the AnaSim noncommercial license. Personal use and qualifying
  noncommercial institutional use are free. Commercial use and hosting outside
  the institutional exemption require a paid license. Earlier releases keep
  their MIT terms.
- Added recorded-case VitalDB comparisons for TIVA and sevoflurane.
- Changed the link to the validation guide to an absolute URL.
- Added residual neuromuscular block to airway collapse under hypnotics.
- Added crystalloid distribution and elimination that depends on pressure and
  age.
- Updated the oxygen dissociation curve and its temperature dependence while
  conserving oxygen stores.
- Reduced reflex and beta-agonist tachycardia and cardiovagal gain with age.
- Lowered the propofol concentration for loss of response with age, so older
  patients lose responsiveness at a higher BIS. Midazolam and etomidate keep
  only their own age effects.
- Held arterial compliance constant below age 40, which matches pulse pressure
  by age.
- Tracked red-cell volume separately from hemoglobin, so transfused hemoglobin
  no longer depends on the recipient's Hb/Hct ratio.
- Limited atelectatic shunt to about 10% of cardiac output. Lost aeration still
  lowers lung volume and compliance.
- Separated HR and MAP responses to stimulation from movement tolerance, so
  surgery raises MAP and HR during maintenance and opioids blunt the intubation
  response. Laryngospasm can follow only the first 40 s of laryngoscopy.
- Placed the apneic threshold below the resting PaCO₂ that the drugs produce, so
  patients no longer breathe over VCV during balanced maintenance.
- Blunted hemorrhage tachycardia under anesthesia.
- During mask ventilation, the mean airway pressure over recent breaths now
  holds the pharynx open, so obstruction no longer alternates with unassisted
  breaths.
- In the hemorrhage and sepsis scenarios, shock recognition now requires MAP 20%
  below its value at the start of the crisis.
- Set the baseline A-a gradient to sea-level means by age.
- Rejected negative or non-finite hemorrhage rates.
- Corrected sevoflurane cardiovascular equilibration, pulmonary transit delays,
  and CO₂ washout across time steps.
- State copies returned to callers no longer share alarm flags with the live
  simulation.
- Restricted rocuronium TCI to free plasma targets. Predictions of decay include
  neuromuscular kinetics, sugammadex, and inhaled agents.
- Scaled oxygen consumption, CO₂ production, and dead space with body size.
  `baseline_vt` defaults to the volume that holds resting PaCO₂ at 40 mmHg.
- Applied the sevoflurane-nitrous oxide interaction at awakening and included
  nitrous oxide in laryngoscopy tolerance.
- Limited blood-volume scaling to the vascular part of each drug's central
  volume. Renal clearance of milrinone, glycopyrrolate, and rocuronium follows
  renal perfusion.
- Updated PK covariates after small hemodynamic changes and resynchronized
  active TCI controllers. `get_predicted_csht` accepts any registry drug.
- Replaced rocuronium and sugammadex PK/PD with Kleijn 2011's integrated model.
- Reduced propofol and remifentanil clearances with blood loss, reversed by
  volume restoration.
- Added reciprocal PK interactions between propofol and midazolam.
- Recalibrated ketamine to raise HR and cardiac output about 30% with SVR
  unchanged, added a separate cardiovascular effect site, and included ketamine
  in laryngoscopy tolerance and baroreflex depression.
- After complete obstruction is relieved during apnea, the lungs now refill
  passively, drawing oxygen from the breathing circuit.
- Stabilized awake core temperature, aligned thermoregulatory vasoconstriction
  with anesthetic depth, and recalibrated anesthetic heat loss and
  redistribution.

## 1.5 - 2026-10-10

- Corrected circuit gas balance, vaporizer delivery as liquid runs out, and
  shunt mixing when venous oxygen is depleted.
- Added PK substeps to preserve drug mass during larger steps. Corrected
  rocuronium onset and recovery, and matched pressure feedback to the displayed
  circulation.
- Added VitalDB comparison scripts and reports for ventilation, capnography, gas
  exchange, and PEEP.
- Made Eleveld 2018 the only propofol PK model and Eleveld 2017 the only
  remifentanil PK model. Added `concomitant_opioids` for the propofol
  opioid-regimen covariate.
- Paired Eleveld BIS with Bouillon clinical responsiveness and laryngoscopy
  tolerance. Added separate clinical and respiratory effect sites, and opioids
  now suppress BIS arousal during stimulation.
- Made Li the only norepinephrine PK model, with clearance that depends on
  propofol and responses calibrated to de Keijzer and Joachim data. Removed the
  additional cardiac-output scaling.
- Made Ensinger arterial clearance the only epinephrine PK model, with
  calibration to adult infusions and boluses. This reduces early pressure dips
  and the clipping of large boluses.
- Adopted Bouillon propofol and Olofsen opioid ventilatory depression, both
  acting on alveolar ventilation.
- Corrected ongoing CO₂ accumulation during apnea, including at a baseline
  respiratory rate of zero.
- During obstruction, inspiratory effort and the impedance respiratory rate now
  continue, airflow is limited as the airway closes, and end-tidal CO₂ holds
  until its timeout while connected.
- Linked oxygen removal from the circuit to lung gas exchange and oxygen stores.
  Complete obstruction stops transport from the circuit while metabolism uses
  the patient's stores.
- Derived intrathoracic pressure from airway pressure and muscle effort, which
  removes offsets that exaggerated venous return.
- Matched TCI prediction pulses to the intervals between rate updates.
  Norepinephrine TCI accounts for endogenous secretion and for clearance that
  depends on propofol.
- Initialized hemodynamic turnover and fast drug responses together to preserve
  pressure and cardiac output at maintenance startup.
- Urine output now keeps most of its basal rate at a MAP of 60 and tapers toward
  zero near circulatory collapse.
- Corrected hemorrhage responses by making right atrial pressure depend on
  filling and by adding venous recruitment, which preserves pressure through
  ATLS class II.
- Blunted adrenergic agonists in severe anaphylaxis to prevent excessive
  pressure rises after epinephrine boluses.
- Removed `simulation_speed`, `pk_model_remi`, `hemo_model`, `resp_model`,
  `pk_model_propofol`, `bis_model`, `loc_model`, `pk_model_nore`,
  `pk_model_epi`, and the `asa` patient input, along with the model selectors
  and alternative implementations.
- Organized drug models and controls in `engine.pk`, `engine.infusion_rates`,
  and `engine.tci`. Moved temperature, anaphylaxis, and sepsis to
  `engine.thermal`, `engine.anaphylaxis`, and `engine.sepsis`.

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
