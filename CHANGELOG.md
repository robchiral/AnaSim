# Changelog

## Unreleased

- AnaSim now supports adults up to 85 years old.
- The app now runs about four times faster at high simulation speeds.
- Blood gases now report sodium, potassium, chloride, and ionized calcium.
  Fluids carry their own ions and albumin. Saline causes hyperchloremic
  acidosis, and rapid transfusion adds acid, CO₂, and potassium and lowers
  ionized calcium. Added renal compensation for sustained PaCO₂ changes,
  0.9% saline, and calcium chloride.
- Hypocalcemia now lowers contractility and vascular resistance, and calcium
  chloride restores contractility. Succinylcholine, malignant hyperthermia,
  and severe acidemia move potassium into plasma. Hyperkalemia peaks T waves,
  widens the QRS complex, and can cause ventricular arrhythmias or asystole,
  and calcium protects against these effects.
- Added malignant hyperthermia with dantrolene, surface cooling, chest-wall
  rigidity, and a guided or independent case.
- Morphine, hydromorphone, fentanyl, lidocaine, and rocuronium kinetics now
  depend on body composition and hepatic function, and morphine and
  hydromorphone also on age and renal function. Tissue volumes for inhaled
  agents scale with fat-free mass and fat mass.
- Midazolam alone now lowers BIS only to the sedation range and causes mild
  respiratory depression and vasodilation. Etomidate preserves blood pressure.
  Both affect thermoregulation, midazolam much less than general anesthetics.
- Ketamine now raises BIS under propofol or sevoflurane, adds analgesia, and
  briefly depresses breathing after rapid high-dose boluses. When sympathetic
  reserve is exhausted, it lowers contractility and vascular resistance.
- Remifentanil now lowers sevoflurane MAC-BAR and blunts the HR response to
  stimulation under inhaled anesthesia, as fentanyl does. Sevoflurane and
  nitrous oxide strengthen rocuronium block at central as well as peripheral
  muscles.
- Sevoflurane now causes rapid shallow breathing that progresses to apnea near
  2 MAC. Nitrous oxide uptake concentrates the other alveolar gases, and its
  washout dilutes alveolar oxygen and CO₂. Nitrous oxide flow is limited to
  three times the delivered oxygen flow.
- When patient uptake exceeds fresh gas flow, the reservoir bag and bellows
  now empty, and a collapsed bellows limits delivered breaths. Room air no
  longer enters the sealed circuit. During apnea without earlier
  denitrogenation, oxygen washes through dead space slowly.
- Sevoflurane and ketamine now partly relieve bronchospasm, and labetalol
  antagonizes epinephrine's bronchodilation. Severe hyperinflation weakens
  inspiratory effort.
- Extubation under light anesthesia can now trigger laryngospasm. Intubation
  of a responsive patient fails unless hypnosis is sustained or circulation
  has stopped.
- Fluid loading no longer raises stroke volume in HFrEF, moderate preload loss
  lowers it less, and pulmonary edema starts at a higher filling pressure in
  chronic heart failure. Blood loss draws interstitial fluid into plasma. High
  intrathoracic pressure reduces preload further, and rapid AF and extreme
  tachycardia limit ventricular filling.
- Norepinephrine now recruits venous volume, raising filling pressure and
  preload, and septic shock blunts cardiac as well as vascular responses to
  catecholamines.
- Postoperative atelectasis now clears over hours of unintubated breathing,
  faster after nonabdominal than abdominal surgery. A smaller residual lasts
  past a day, and morbid obesity prevents clearance over 24 hours.
- Hypothermia below 33 °C now slows HR, and below 30 °C it depresses
  contractility and can start VF.
- Alpha and beta blockade now shift ephedrine's vascular and cardiac responses
  separately, and beta agonists relieve sinus bradycardia.
- Epinephrine clearance and sensitivity now fall with age, and its responses
  are recalibrated to infusion and bolus studies. Its inotropic effect widens
  pulse pressure. HR and pressure responses to boluses keep rising at larger
  doses, and higher infusion rates restore vascular resistance.
- Atropine now raises pressure less, and ketamine raises it more after
  atropine or during pacing.
- Above 65, propofol lowers blood pressure more slowly but further.
- Hypoxic ventilatory drive now falls after age 62 to about 40% from age 75
  onward, while CO₂ sensitivity falls much less. Surgical concentrations of
  sevoflurane or propofol now abolish hypoxic hyperventilation while preserving
  the CO₂ response.
- Acute hypercapnia now increases hypoxic cerebral blood-flow reserve, and
  hypocapnia reduces it, with cerebrospinal fluid adaptation over hours.
  Extreme CO₂ narcosis depresses medullary breathing toward shallow
  hypoventilation, and recovery slows after coma lasting longer than an hour.
- Recalibrated respiratory muscle fatigue. Heavy loads take about twice as
  long to exhaust the muscles, most strength returns within 15 minutes of rest
  or pressure support, and a smaller deficit lasts up to a day.
- Recalibrated ventricular arrhythmias and intravenous amiodarone. Sustained
  VT with a pulse takes tens of minutes to convert, and VF recurs after about
  half of successful shocks. Rapid amiodarone boluses lower pressure more than
  slower infusions, and cardiac output falls more in HFrEF.
- Epinephrine during CPR now raises diastolic pressure and lowers cardiac
  output and EtCO₂. Pulseless VT is defibrillated like VF, and an
  unsynchronized shock on a perfusing rhythm can cause VF. Shocks temporarily
  reduce stroke volume, arterial pressure recovers more slowly after longer VF,
  and shocks after prolonged arrest can leave transient post-shock asystole
  before beats return.
- When pulses stop, arterial systolic and diastolic readings clear after a
  brief hold, and MAP follows catheter pressure. NIBP fails during PEA and
  raises a low-MAP alarm. Headless output shows arterial MAP during arrest and
  blanks failed NIBP and invalid SpO₂ readings.
- Recalibrated lidocaine toxicity and lipid rescue. In awake patients, HR and
  blood pressure rise before seizures, and QRS widening and cardiac depression
  require much higher concentrations. Nitrous oxide helps suppress seizures.
  Lipid emulsion reduces effect-site exposure by about a fifth over 30 minutes
  and lessens cardiac toxicity.
- Added an adjustable bag pressure limit (APL, default 20 cmH₂O) and facemask
  seal. Obstruction and bronchospasm limit bag volume at the APL, and an APL
  setting other than 20 cmH₂O during spontaneous breathing builds continuous
  airway pressure as the bag fills. Mask leaks reduce assisted ventilation and
  the displayed airway pressure, flow, and volume, and they admit room air.
  Large leaks flatten the capnogram, and pressures above 20 cmH₂O leak even
  with a tight seal.
- During upper airway obstruction, mask CPAP no longer lowers ventilation
  below that of unassisted breathing.
- Basal heat production now scales with body weight, redistribution cooling is
  smaller in heavier patients, and sepsis raises metabolism and heat
  production. Shivering starts only after the patient regains responsiveness.
- Cerebral ischemia now lowers BIS within 20 seconds. High-BIS alarms apply
  only during paralysis or active maintenance, and EtCO₂ alarms wait for the
  first connected breath.
- Anaphylaxis and sepsis cases now reach full severity over 45 to 60 seconds
  and accept 0.9% saline for initial resuscitation. The oxygen supply failure
  case starts on low-flow maintenance.
- Maternal extracellular water now expands during pregnancy.
- Propofol, remifentanil, fentanyl, and rocuronium TCI now use only age,
  weight, height, and sex, as clinical pumps do. Organ impairment, blood loss,
  midazolam, and sugammadex can move concentrations away from the target. The
  pump model omits hand boluses, and TCI started after a manual rate assumes no
  drug on board. Pump bolus and rate limits are updated, and maintenance with
  TCI starts from a 30-minute TCI history.
- Removed the opioid-containing anesthesia checkbox from the browser setup.
  Propofol kinetics assume opioid co-administration unless
  `concomitant_opioids` is set to false through the CLI or Python API.
- Infusion rates use the units clinical pumps display: rocuronium, dobutamine,
  milrinone, and esmolol in mcg/kg/min, ketamine and lidocaine in mg/kg/hr,
  and amiodarone in mg/min.
- The emergence case no longer counts a patient in Stage II as ready for
  extubation, and removing the tube in Stage II asks for confirmation. A
  recovered train-of-four stays valid until another blocker dose.
- Monitor numerics now round instead of truncating, and alarms and case
  status messages use the rounded values. A failed cuff measurement shows its
  own age and prompts a new reading, and NIBP bias and failure rate rise
  smoothly as MAP falls below 60 mmHg.
- EtCO₂ now falls to zero within 15 seconds of apnea, and a curare cleft no
  longer gives a falsely low reading. Disconnecting the airway clears
  end-tidal agent and nitrous oxide readings, and an unplanned disconnect no
  longer suspends EtCO₂ alarms.
- A partly empty reservoir now limits VCV, PCV, and PCV-VG breaths alike, and
  bagging through a leaky mask drains the bag at low fresh gas flow. Mask
  breathing no longer counts bronchospasm twice, and slow VCV keeps reporting
  exhaled volumes. Breathing pure oxygen clears alveolar nitrogen, and setting
  FiO₂ adjusts nitrous oxide, air, and oxygen flows together.
- During chest compressions, arterial readings stay ordered and organ
  perfusion follows forward flow. VF stops right heart output. Hemorrhage in
  AF no longer inverts the baroreflex, older patients with low coronary
  reserve no longer have ischemia at rest, and HFrEF maintenance starts
  without a pressure jump.
- Blood gases from hypothermic patients are warmed to 37 °C, so reported
  PaCO₂ and PaO₂ rise and pH falls. Febrile patients hold their baseline
  temperature, and balanced maintenance starts at the same MAC at any
  temperature.
- A hand bolus of a vasopressor running by TCI pauses the pump, and manual
  TIVA startup sets a propofol rate that holds the effect-site concentration.
  Lipid emulsion expands volumes equally by bolus or infusion, sugammadex
  volume no longer shrinks in very heavy patients, and calcium chloride leaves
  an acid load as the calcium clears.
- The browser session now survives background-tab throttling, and the app no
  longer drops steps at some speeds. `dt` is limited to 2 s, boolean settings
  and drug rates are validated, and CSV alarm columns are valid JSON.
- In normal ventricles, stroke volume now falls when vasopressors raise
  resistance more than contractility, which limits the pressure overshoot
  after epinephrine at return of circulation. Severe left ventricular failure
  congests the lungs and raises left atrial pressure. Without circulation,
  left atrial pressure settles at venous pressure and pulmonary edema stops
  forming.
- VT with a pulse pressure below 5 mmHg and MAP below 25 mmHg now counts as
  pulseless VT, and compressions restore its coronary supply at once. Asystole
  from hyperkalemia or local anesthetic toxicity restarts only after
  conduction recovers. Compressions during a perfusing rhythm show artifact on
  the ECG.
- Below 33 °C, cold-induced vasoconstriction and reflex tachycardia now
  weaken, and shivering fades below 31 °C. Alpha blockade blunts cold-induced
  vasoconstriction, and the core cools faster during arrest.
- Rapid saline no longer causes a brief alkalemia before the acidosis, and
  volume expansion alone no longer adds metabolic acid. Lipid emulsion no
  longer causes hyponatremia, stored red cells carry less oxygen, and
  crystalloid stops returning from the interstitium without circulation.
- A loose mask now admits room air during PSV and SIMV breaths with little
  support, and PCV-VG raises pressure to make up for a mask leak. An empty
  bellows or apnea on the circuit no longer ventilates the lungs. Gas absorbed
  behind a closed airway lowers resting lung volume, and a partly obstructed
  airway refills the lungs more slowly.
- Effect-site TCI now holds maintenance concentrations with a steady infusion
  rate at any step size. Stopping a TCI infusion in the browser clears its
  target.
- Norepinephrine now raises MAP steadily during TIVA with hypovolemia, and
  awake patients start at their baseline MAP. Sugammadex no longer causes a
  jump in plasma rocuronium.
- The low EtCO₂ alarm limit and the emergence case's EtCO₂ range now follow
  the lower PaCO₂ of pregnancy, and the range also widens in COPD. SpO₂
  returns to the current saturation promptly after signal loss, and a dropout
  no longer clears a low-SpO₂ alarm.
- Malignant hyperthermia no longer recurs after dantrolene has controlled it
  and the triggers have cleared, and the case credits hyperventilation started
  before onset. Case blood pressure checks no longer accept an outdated cuff
  reading, and hints in the independent emergence case skip completed steps.
- `--record-interval 0` now records every step.
- Arterial pressure now falls continuously toward venous pressure at very low
  cardiac output. Maintenance starts without a first-beat pressure overshoot,
  HR falls during prolonged pauses, and weak pulses reduce the SpO₂ signal.
- Occluded inflations no longer count as breaths. Auto-PEEP falls during
  expiratory pauses, and switching out of PSV clears apnea backup.
- Inhaled-agent uptake and washout now share the lung's gas-volume balance
  with oxygen and nitrogen. Washout gas first refills volume lost to absorption
  before flowing out of the lungs.
- TCI target changes now preserve plasma or effect-site mode, and pump rate
  changes keep their timing when simulation steps cross an update.
- Open simulations now stop recording and lock treatment controls at
  irreversible cardiac arrest. Local commands update controls immediately,
  with less frequent polling while paused.
- New or changed alarms now end an active audio silence.

## 1.8 - 2026-11-06

- Anaphylaxis and sepsis cases now begin with established shock. Anaphylaxis
  causes plasma leak, and the case requires improved bronchospasm.
- Without an arterial line, blood pressure objectives require a cuff reading
  from the last minute.
- Sustained high inspiratory effort now fatigues the respiratory muscles.
- Lidocaine overdose now causes seizures, QRS widening, arrhythmias, and
  circulatory collapse. Added 20% lipid emulsion to the Circulation tab.
- Lidocaine's central volume and clearances now scale with fat-free mass.
- Severe myocardial ischemia or hypoxia can now start VT or VF, and VF can
  recur after a successful shock.
- Added amiodarone, listed with lidocaine under Antiarrhythmics.
- Severe circulatory failure, including untreated hemorrhage, now progresses
  through PEA to asystole, and the simulation continues through arrest. A
  perfusing rhythm within about 4 minutes restores circulation; the session
  ends when ischemic injury is complete, after about 11 minutes without
  circulation. Removed the `end_on_cardiac_arrest` option.
- Added chest compressions and a defibrillator to a new Circulation tab, which
  also holds fluids. Compressions move about a fifth of resting cardiac output,
  little when the heart is empty, and diastolic pressure and EtCO₂ show their
  effect. They keep VF shockable, make fine VF coarse again, and can restart
  beats in asystole. The defibrillator charges before each shock. Shocks
  terminate VF, and synchronized shocks fire on the next R wave to convert AF,
  SVT, and VT; afterward, the heart beats only as well as its time without flow
  allows. Epinephrine at arrest doses raises diastolic pressure during
  compressions.
- Added a cardiac arrest case with VF during surgery.
- As the heart fails, the ECG passes through junctional, idioventricular, and
  agonal rhythms, and untreated VF becomes fine and then asystole. Ischemic ST
  depression now starts at the J point and merges into the T wave.
- Severe cerebral hypoperfusion or hypoxemia now causes unresponsiveness,
  airway collapse, low BIS, and apnea within seconds; BIS shows it after the
  monitor's processing delay. After 5 minutes without supply, the patient
  remains comatose, although BIS and breathing can recover.
- Added CO₂ narcosis: severe hypercapnia, especially with acidemia, reduces
  responsiveness and airway tone.
- Without flow, arterial pressure now falls to the mean systemic filling
  pressure instead of zero, and drugs are distributed, cleared, and delivered
  to their effect sites only as blood circulates, apart from esterase
  metabolism.
- Severe acidemia depresses the heart and the response to adrenergic drugs, and
  metabolic acidosis increases ventilation.
- Shed blood now removes drug from the central compartment.
- Fixed a crash during prolonged ventilated arrest after hemodilution.
- Added remifentanil and fentanyl kinetics, sevoflurane potency, and the
  vascular effect of sevoflurane to optional patient variability. TCI then
  predicts from population kinetics, so concentrations differ from the target.
- Lowered cardiac output under sevoflurane below awake output, as measured in
  volunteers. Most of sevoflurane's fall in pressure now comes from stroke
  volume; the size of the fall is unchanged.
- Matched fentanyl and lidocaine clearances to their source studies, which were
  measured during anesthesia and surgery.
- Added arterial blood gas sampling in a new Labs tab.
- Labeled the crystalloid as lactated Ringer's.
- Added lactate and standard base excess. Oxygen consumption falls when oxygen
  delivery drops below a critical level, and the unmet demand raises lactate.
  Epinephrine, sepsis, hepatic impairment, and lactated Ringer's also change
  lactate and base excess.
- Added bag ventilation rate and tidal volume settings.
- Added independent practice with optional hints and a debrief for every case.
- Reorganized the setup screen around the choice of a case or an open
  simulation, with patient details and options collapsed.
- Made case events start on their own and replaced the Events tab in cases with
  case actions, such as requesting hemostasis. Guided steps credit earlier actions.
- Added warnings for unsafe actions, such as extubation with residual block.
  Intubation of an awake, reactive patient fails.
- Made every intubation add a stimulation response to any surgical stimulation.
- Added optional pulse oximeter tones and alarm audio.
- Sped up TCI prediction rebuilds, drug kinetics, and monitor updates.
- Added poorly ventilated (low V/Q) lung units, separate from atelectatic shunt.
  Under anesthesia, airway closure lowers PaO₂ more in older patients, and PEEP
  of about 6 cmH₂O prevents most of this fall. COPD and bronchospasm now cause
  low V/Q hypoxemia, which responds to added oxygen more than shunt does.
- Lowered atelectatic shunt to about 5% after a preoxygenated induction.
  Maintenance sessions start with this atelectasis, and it persists after
  emergence.
- Added morphine and hydromorphone respiratory-rate responses to optional
  patient variability (Olofsen 2026).
- Added a surgical site setting. After abdominal surgery, part of the airway
  closure under anesthesia persists after emergence, so room-air PaO₂ stays
  lower, most after upper abdominal surgery and in older patients.

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
