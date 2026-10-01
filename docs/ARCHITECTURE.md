# AnaSim architecture

AnaSim splits pharmacology, cardiorespiratory physiology, the anesthesia
machine, and monitors into stateful subsystems. The runtime advances them in a
fixed order and copies their outputs into `SimulationState`.

## Simulation state

`SimulationState` keeps physiology, ideal arterial pulse landmarks, and monitor
measurements separate:

| Layer | Fields | Meaning |
|-------|--------|---------|
| Physiology | `map`, `hr`, `co`, `sv`, `svr`, `sao2`, `pa_co2`, `alveolar_co2`, `pao2` | Model values used by physiology, analytics, and endpoints |
| Ideal arterial pulse | `sbp`, `dbp` | Beat landmarks derived from Su MAP and stroke volume |
| Arterial catheter | `art_pressure`, `art_sbp`, `art_dbp`, `art_map` | Instantaneous and completed-beat values after line dynamics |
| Other monitors | `nibp_sys`, `nibp_dia`, `nibp_map`, `display_hr`, `display_bis`, `display_etco2`, `display_spo2` | Values shown to the learner |

- `map`, `hr`, `sv`, `co`, and `svr` are Su model outputs. Monitors only read
  them.
- NIBP measures `sbp`, `dbp`, and `map`.
- The UI, CLI, alarms, and scenarios read `art_*` when the arterial line is
  enabled and `nibp_*` otherwise.
- The recorder writes both physiology and monitor values.
- `engine.output_buffer` holds ten seconds of per-step `WaveformSample` records
  (ECG, pleth, capnogram, arterial pressure) for the monitor sweep.
- Public numeric fields are built-in `float` values.

## Main modules

[`engine.py`](../anasim/core/engine.py) holds the subsystems and applies learner controls.
[`runtime.py`](../anasim/core/runtime.py) advances them,
[`projection.py`](../anasim/core/projection.py) writes their outputs to state,
and [`monitors.py`](../anasim/core/monitors.py) updates measurements and alarms.
[`initialization.py`](../anasim/core/initialization.py) sets the starting state.
Drug units, pump limits, and UI metadata are defined in
[`drug_registry.py`](../anasim/core/drug_registry.py).

## Step order

```text
SimulationEngine.step()
 1. Anesthetic depth, metabolic rate, and tolerance of stimulation
 2. Disturbances and clinical events
 3. PK scaling to blood volume and CO; resynchronize active TCI
 4. TCI infusion rates
 5. Machine: ventilator, bag-mask, vaporizer, circuit
 6. PK: plasma and effect-site concentrations, TOF, volatile uptake
 7. Physiology: respiratory mechanics, gas exchange, hemodynamics
 8. Copy outputs into SimulationState
 9. Monitors: cardiac cycle, arterial pulse and line, ECG, pleth, NIBP,
    capnography, alarms
10. Shivering
11. Temperature
12. Cardiac arrest check
```

## Cross-step inputs

These values come from the previous step:

| Value | Used by | Updated by | Reason |
|-------|---------|------------|--------|
| `state.co` | Volatile PK scaling, respiration | Hemodynamics | Hemodynamics runs after PK and respiration |
| `state.va` | Volatile PK | Respiration | Respiration runs after the machine and PK |
| `state.mv` | Circuit and machine | Physiology | Minute ventilation is final after mechanics and respiration |

## Initialization

- `awake` starts from patient baselines.
- `steady_state` runs a hidden maintenance period to set drug, gas, fluid, and
  physiologic state, starting from MAP 70 mmHg. Recording, display history, arrest
  checks, and visible fluid and temperature totals start after it.
- Visible time starts at zero. Norepinephrine used to reach the initial MAP
  stays running and visible.
- Controllers, gases, and fluid balance continue from the hidden period, so
  early drift is expected.

## Model notes

### Hemodynamics

- The model extends Su et al. 2023 with blood volume, pulmonary circulation,
  vasoactive drugs, septic shock, and anaphylaxis.
- Cardiovascular effects use propofol and opioid plasma concentrations.
  Hypnosis, tolerance, BIS, and respiratory depression use effect-site
  concentrations.
- The opioid terms use remifentanil plus 0.82 times fentanyl (`opioid_ce`,
  `opioid_cp`). The propofol terms of BIS, LOC, tolerance, and ventilation use
  propofol plus etomidate and midazolam equivalents (`hypnotic_ce`). The
  midazolam equivalent saturates and is synergistic with propofol, and
  etomidate counts at half strength for ventilation. Ketamine adds to LOC and
  tolerance and raises sympathetic tone. BIS, ventilatory drive, and
  pharyngeal collapse exclude ketamine.
- A fast baroreflex adjusts HR around a MAP set point that resets over about
  30 minutes. Propofol and sevoflurane reduce its gain, and the sensed pressure
  excludes noxious stimulation. When a rhythm fixes the ventricular rate, the
  baroreflex adjusts TPR. Su's turnover feedback is the slow component.
- Epinephrine has separate curves for chronotropy, inotropy, beta-2 dilation,
  and alpha constriction. The pressor response lags, so a bolus peaks HR
  before SBP.
- Esmolol and labetalol are competitive antagonists. The agonist
  concentration at each adrenoceptor is divided by 1 + C/Kb, so higher agonist
  doses overcome the block. Milrinone and vasopressin act on other receptors.
  Receptor occupancy reduces resting sympathetic tone (less under anesthesia),
  the reflex, chemoreflex, hemorrhage, and stimulation responses, and the AF
  ventricular rate. Glycopyrrolate reduces resting vagal tone, baroreflex and
  opioid bradycardia, and raises the sinus bradycardia rate.
- Below SaO2 70% (full effect at 30%), myocardial hypoxia slows the heart and
  reduces contractility, progressing to pulseless electrical activity.
- With `end_on_cardiac_arrest`, MAP below 20 mmHg or HR below 10 bpm for
  15 seconds ends the session. Resuscitation is not modeled.

### Respiration

- Propofol, remifentanil, and sevoflurane depress central drive.
- Free (sugammadex-unbound) rocuronium acts at two effect sites. The adductor
  pollicis site sets TOF and shivering. The central site (diaphragm and larynx)
  sets respiratory muscle strength and laryngospasm; it equilibrates faster but
  needs about 1.8 times the concentration (Plaud 1995; Cantineau 1994), so
  breathing returns before the TOF recovers.
- Without an ETT or positive pressure, loss of consciousness causes up to 40%
  upper-airway obstruction (Hillman 2009; Eastwood 2005). CPAP or bag-mask
  ventilation keeps it open.
- Alveolar O2 is a mass balance over the FRC gas store and hemoglobin-bound O2.
  At steady state it reduces to the alveolar gas equation. During apnea the
  stores deplete at VO2, so preoxygenation sets the safe apnea time, and a
  patent airway draws in gas to replace absorbed O2 (apneic oxygenation).
- `alveolar_co2`, `pa_co2`, and `etco2` are separate. Low cardiac output widens
  the PaCO2-EtCO2 gap.

### Ventilation mechanics

- The lung is a single resistance-compliance compartment.
  VCV delivers constant flow. Pressure modes and passive expiration use the
  exact exponential solution, and steps split at breath boundaries.
- Trapped volume counts toward auto-PEEP once. Mean airway pressure is
  time-weighted.
- Gas exchange uses the last completed exhaled VT, including zero-volume
  breaths, so it lags one breath. Displayed VT is the raw exhaled volume;
  MV includes modeled airway losses.
- PSV and CPAP have no patient triggering or flow cycling, and VCV has no
  pressure limit.
- `set_vent_power` starts and stops the ventilator. Its settings persist while
  it is off, and `set_vent_settings` changes them without starting it.

### Monitors

- ECG, arterial pressure, and pleth share one beat clock. Arterial ejection
  follows the QRS, and the pleth follows arterial pressure after a peripheral
  delay.
- Beat-synchronous monitors step at 10 ms or less, whatever the outer step.
- The arterial renderer shapes each beat from the systolic, diastolic,
  dicrotic-notch, and dicrotic-peak landmarks of Mahdi et al. At a regular rate,
  beat mean equals Su MAP and pulse pressure comes from Su stroke volume and
  age-adjusted arterial compliance.
- Each beat's stroke volume follows its preceding filling time, and its
  diastole runs off for its own R-R with the Windkessel time constant
  (resistance × compliance). This only changes beats when R-R varies, as in AF.
  The pleth amplitude follows the same stroke volume.
- The arterial catheter is a second-order system (default 20 Hz, damping
  0.65). ART numerics average completed filtered beats over the last 5 seconds.
- NIBP measures the ideal pressures with cuff timing, bias, and failure.
- Monitor HR averages the last 12 R-R intervals. Arrest rhythms without
  organized beats remove the arterial pulse and pleth; the ECG shows the rhythm.
- ECG waves keep fixed durations around each R peak; only QT follows the
  preceding R-R interval (Fridericia, QTc 400 ms). AF R-R intervals are drawn
  independently within ±35% of the mean, narrowed at fast rates so none is
  shorter than 250 ms.
- BIS adds sevoflurane (1 MAC gives BIS 41) to the selected
  propofol-remifentanil model, then applies 10 s smoothing and the model's
  processing delay.
- The gas monitor shows end-tidal age-adjusted MAC (`et_mac`). Brain MAC
  (`mac`) sets drug effects.
- Poor perfusion slows the finger SpO2 response and shrinks the pleth. SpO2
  shows a value only with an organized rhythm and adequate perfusion.
  Arterial saturation (`sao2`) depends only on PaO2 and hemoglobin.
- EtCO2 updates from completed capnogram breaths and clears 15 seconds after the
  last valid exhaled sample.

### PK and TCI

- All intravenous drugs use `MammillaryPK`: up to two peripheral compartments
  and an effect site. `state_fields`, `get_ss_matrices()`, and `state_vector()`
  share one state order, used by TCI and steady-state initialization.
- Hemodynamic scaling changes V1 with blood volume and clearances with cardiac
  output. Changing V1 rescales central concentration by old V1 / new V1 to
  preserve drug amount. Peripheral volumes and concentrations and the effect
  site stay unchanged. Drug lost in shed blood is not tracked separately.
  Epinephrine clearance is independent of cardiac output.
- Every 10 s the TCI controller predicts the zero-input course of the target
  compartment over ten minutes and picks the largest rate that keeps it at or
  below target (Shafer and Gregg 1992). From zero this is the bolus whose
  effect-site peak reaches target; at target it is the maintenance rate.
  Propofol and remifentanil are limited to 1200 mL/h.
- Controllers are rebuilt after PK parameter changes and reset from current
  concentrations after external boluses.
- Setting a manual rate disables that drug's TCI controller. Changing the target
  compartment replaces the controller, starting from current PK concentrations.

### Temperature and stimulation

- Induction moves up to 1.3 °C of core heat to the periphery with a 20-minute
  time constant (Matsukawa 1995). The heat stays peripheral if anesthesia
  lightens.
- Noxious-stimulus responses scale with the probability of responding to
  laryngoscopy on the Bouillon propofol-remifentanil-MAC surface, so opioids
  reduce the hemodynamic and BIS response.

## Browser app

Both versions use the page in `anasim/web_assets/` and a
[`WebSession`](../anasim/web.py), which applies learner commands and returns
JSON snapshots of monitor values, new waveform samples, control state, and the
current objective.

- Local: [`local.py`](../anasim/local.py) serves the page on 127.0.0.1 under a
  random URL path, accepts only same-origin requests, and steps the session on
  its own clock while the page polls. A reload reconnects to the paused
  session. If polling stops for 5 seconds, the session pauses and any recording
  closes. Recordings stay in the recordings directory.
- Hosted: a module worker loads Pyodide, numpy, and scipy from the CDN and runs
  the session in the browser. Stopping a recording downloads it.
  `scripts/build_web.py` builds the site into `build/web`, and
  `.github/workflows/pages.yml` publishes it.

## Scenario objectives

`engine.actions` records controls and event transitions with their simulation
times. `WebSession` calls `begin_step()` when an objective becomes active.

| Kind | Example | Check reads |
|------|---------|-------------|
| Action | "Give 500 mL", "start the vasopressor", "select the ETT" | `engine.actions` since the objective started, plus current state where relevant |
| State | "MAP > 65", "TOF below 25%", "circuit FiO2 below 30%" | Current engine state |

Only actions taken after an objective starts count toward it. The log marks
the start by position, because actions taken while paused share a timestamp. Step-scoped queries raise an error when no objective is active.
