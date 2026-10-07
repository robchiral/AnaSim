# AnaSim architecture

AnaSim separates pharmacology, physiology, the anesthesia machine, and monitors.
The runtime advances them in a fixed order and writes outputs to `SimulationState`.
See [CLI and Python usage](CLI_USAGE.md) for configuration and scripted runs.

## Simulation state

`SimulationState` stores physiology separately from monitor measurements.

| Group | Selected fields | Meaning |
|-------|-----------------|---------|
| Physiology | `map`, `hr`, `co`, `sv`, `svr`, `sao2`, `pa_co2`, `alveolar_co2`, `pao2` | Values used by the models and cardiac arrest check |
| Ideal arterial pulse | `sbp`, `dbp` | Systolic and diastolic pressures derived from the Su model's mean pressure and stroke volume |
| Arterial catheter | `art_pressure`, `art_sbp`, `art_dbp`, `art_map` | Instantaneous and completed-beat pressures after catheter filtering |
| Other monitors | `nibp_sys`, `nibp_dia`, `nibp_map`, `display_hr`, `display_bis`, `display_etco2`, `display_spo2` | Displayed measurements |
| Ventilation | `paw_peak`, `paw_plat`, `paw_mean`, `peep`, `compliance_dyn`, `et_o2` | Breath pressures, dynamic compliance, and end-tidal O₂ |
| Fluids | `blood_volume`, `urine_out_ml`, `net_fluid_ml`, `lap`, `lung_water` | Blood volume, fluid balance, left atrial pressure, and lung water above normal |

Monitors read physiology without changing it. Noninvasive blood pressure (NIBP)
measures `sbp`, `dbp`, and `map`. Displays and scenario pressure checks use
`art_*` with an arterial line and `nibp_*` otherwise. The recorder writes both
sets. `engine.output_buffer` holds 20 seconds of waveform samples.
[`state.py`](../anasim/core/state.py) defines all fields and their units.

## Main modules

| Module | Responsibility |
|--------|----------------|
| [`engine.py`](../anasim/core/engine.py) | Owns the subsystems and applies controls |
| [`runtime.py`](../anasim/core/runtime.py) | Advances the simulation |
| [`projection.py`](../anasim/core/projection.py) | Copies model outputs into public state |
| [`monitors.py`](../anasim/core/monitors.py) | Updates measurements and alarms |
| [`initialization.py`](../anasim/core/initialization.py) | Sets the starting state |
| [`drug_registry.py`](../anasim/core/drug_registry.py) | Defines drug units, pump limits, and interface metadata |

## Step order

Each call to `SimulationEngine.step()` runs these stages in order.

1. Calculate anesthetic depth, metabolic rate, and tolerance of stimulation.
2. Apply disturbances and clinical events.
3. Scale pharmacokinetic (PK) parameters with blood volume and cardiac output
   (CO); resynchronize active target-controlled infusion (TCI).
4. Update TCI infusion rates.
5. Advance the vaporizer and breathing circuit.
6. Update plasma and effect-site concentrations, neuromuscular block, and
   volatile uptake.
7. Advance breathing mechanics, gas exchange, and hemodynamics.
8. Copy outputs into `SimulationState`.
9. Update cardiac timing, waveforms, monitor measurements, and alarms.
10. Update shivering.
11. Update temperature.
12. Check for cardiac arrest.

### Inputs from the previous step

| Value | Used by | Updated by |
|-------|---------|------------|
| `state.co` | Volatile PK scaling, respiration | Hemodynamics |
| `state.va` | Volatile PK | Respiration |
| `state.mv` | Circuit and machine | Physiology |

## Initialization

`awake` starts from patient baselines. `steady_state` seeds a maintenance drug
history, starts controlled ventilation, and settles drugs, gases, and physiology.
Vasopressor infusion rates start at zero. Mean arterial pressure (MAP) reflects
the untreated anesthetic response and may need treatment. The session clock,
recording, display history, arrest checks, and displayed fluid and temperature
totals start after initialization.

## Supported patient domain

AnaSim accepts adults aged 18 to 70 years within the [input limits](CLI_USAGE.md#fields).
Li et al. 2024 studied 36 healthy volunteers aged 18 to 70 years, weighing
51.5 to 94.8 kg, 151 to 196 cm tall, with BMI 18.0 to 31.1 kg/m². AnaSim slightly
extends those body-size ranges. Su et al. 2023 used a cohort of the same size
and age groups; age affects the modeled stroke-volume response to propofol.
Hemoglobin, hematocrit, and organ-function limits are simulator choices.

The optional Fuentes bispectral index (BIS) model was derived in children.
Using it in adult sessions extrapolates beyond its source population; see
[Model references](REFERENCES.md#respiratory-control-and-bis).

## Model notes

AnaSim adapts published models and calibrates additional responses for teaching.
[Model references](REFERENCES.md) lists the sources. The linked tests check
implemented responses against expected ranges.

### Hemodynamics

AnaSim extends Su et al. 2023 with blood volume, pulmonary circulation,
vasoactive drugs, a baroreflex, septic shock, and anaphylaxis. Su drug-effect
parameters retain their published values; tests compare concentration-step
responses with an independent integration of the equations. Baseline heart rate
and MAP are resting values, omitting the study's initial anxiety-related
transients. Pressure feedback includes rhythm, hypoxia, and stimulation effects,
and holds its last value during arrest.

Cardiovascular propofol and opioid effects use plasma concentrations;
hypnosis and respiratory depression use effect-site concentrations. Applying
fentanyl's remifentanil equivalent to cardiovascular effects is a simulator
approximation.

[`HemodynamicConfig`](../anasim/physiology/hemo_config.py) combines published
anesthetic reflex effects and hypoxic arrest thresholds with calibrated reflex
gains, set-point reset, and hypoxia time constants. The baroreflex adjusts heart
rate, or vascular resistance when a rhythm fixes the rate. Severe hypoxia
reduces heart rate and contractility.

Epinephrine, phenylephrine, vasopressin, dobutamine, and milrinone use
concentration-response curves informed by published data, with calibrated
combined effects. Epinephrine fits arterial infusion
(Freyschuss 1986) and IV bolus (Takahashi 2002) data. Esmolol (Sum 1983), labetalol
(Abernethy 1987; Hafsa 2022), and glycopyrrolate (Ali-Melkkilä 1993) use published
antagonist potencies, with calibrated sympathetic and vagal contributions.
During beta blockade, the model's sinus baroreflex adjusts only heart rate.
Part of labetalol's pressure reduction therefore persists beyond the 30-minute
recovery reported by Abernethy.
See [hemodynamics](../tests/test_hemodynamics.py),
[epinephrine](../tests/test_epinephrine.py), and
[autonomic drug](../tests/test_autonomic_drugs.py) tests.

Crystalloid retains 30% of its volume in blood and albumin 80%. The rest enters
an interstitial pool that returns fluid only to replace a blood-volume deficit.
Volume expansion increases urine output, with a smaller response under
anesthesia (Reid 2003; Hahn 2010). Fluid filters into the lung above a left
atrial pressure of 20 mmHg. Lung water above 3 mL/kg of predicted body weight
causes alveolar flooding, increased shunt, and reduced compliance. Positive
end-expiratory pressure (PEEP) can re-aerate these units (Malo 1984). Filtration
and flooding rates are teaching estimates. Oncotic pressure and cardiac
dysfunction are not modeled.

With `end_on_cardiac_arrest`, MAP below 20 mmHg or heart rate below 10 bpm for
15 seconds ends the session. Cardiac arrest resuscitation is not modeled.

### Drug effects

[`anesthesia.py`](../anasim/patient/pd/anesthesia.py) calculates remifentanil
equivalents for fentanyl using isoflurane minimum alveolar concentration (MAC)
reduction (McEwan 1993; Lang 1996), and propofol equivalents for etomidate and
midazolam at equal loss-of-response concentrations (Kaneda 2011; Albrecht 1999).
Midazolam-propofol synergy follows
Short 1992. Etomidate's ventilatory effect (Valk 2021), the maximum midazolam
equivalent, and ketamine's laryngoscopy potency and sympathetic response
(Idvall 1979) are simulator calibrations.

Lidocaine blocks part of the response to stimulation in proportion to its
reduction of anesthetic requirement (Himes 1977). It has no hypnotic,
hemodynamic, or antiarrhythmic effect in AnaSim; toxicity is not modeled.
See [adjunct tests](../tests/test_anesthetic_adjuncts.py).

[`TOFModel`](../anasim/patient/pd/nmba.py) separates train-of-four (TOF) response
at the adductor pollicis from diaphragm and larynx effects, allowing breathing
to recover before TOF. The diaphragm and larynx share laryngeal kinetics.
Plasma-to-muscle gradients select calibrated onset or recovery rates;
sugammadex binding accelerates recovery.
See [pharmacology tests](../tests/test_pharmacology.py).

### Respiration

[`RespiratoryModel`](../anasim/physiology/respiration.py) combines published
ventilatory and hypercapnic responses with calibrated parameters. Anesthetics
and opioids reduce drive. Ventilation can suppress breathing in unconscious
patients below the apneic PaCO₂ threshold. Awake CO₂ feedback changes depth
before frequency; hypocapnia preserves rhythmic breathing. Awake response
gains are teaching estimates. Positive pressure or a tracheal tube relieves
upper-airway obstruction after loss of consciousness.

Gas exchange tracks alveolar gas and blood oxygen stores for preoxygenation,
apnea, and blood loss. Alveolar, arterial, and end-tidal CO₂ are separate;
low CO widens the arterial to end-tidal CO₂ gap. Oxygen, sevoflurane, and nitrous
oxide share lung gas volume and shunt from perfused closed units. Shunt mixing
limits extraction to available oxygen, keeping venous oxygen content
nonnegative. Oxygen calculations use standard gas conditions and hemoglobin
stores. See [respiratory tests](../tests/test_respiration.py).

### Ventilation mechanics

[`RespiratoryMechanics`](../anasim/physiology/resp_mech.py) combines resistance,
published viscoelastic parameters, compliance fitted to VitalDB, and a
simplified muscle-pressure profile. Ventilatory assistance reduces effort on
the next breath; inflation ends effort earlier under anesthesia. Bronchospasm
increases expiratory resistance as volume falls and can limit flow, causing
trapping and intrinsic PEEP.

[`LungAeration`](../anasim/physiology/lung.py) shares aeration and gas volume
with gas exchange and volatile uptake. Recruitment and closure depend on
pressure history; high inflation stiffens the lung. Compliance scales with
predicted body weight and BMI. Pressure-volume curve updates conserve gas
volume. The solver resolves breath events and pressure limits within each step.

[`AnesthesiaVentilator`](../anasim/machine/ventilator.py) uses GE Aisys CS2 modes
and ranges, with defaults informed by Primus recordings and device conventions.
The [README](../README.md#features) lists supported modes.

| Mode | Behavior |
|------|----------|
| VCV | Enforces the pressure limit during flow delivery and the inspiratory pause |
| PCV-VG | Adjusts pressure toward the set volume by at most 3 cmH₂O per breath; the target stays at least 5 cmH₂O below the pressure limit (Pmax), including after Pmax or PEEP changes |
| SIMV | Synchronizes mandatory breaths with patient efforts during the final 5 seconds of each cycle, bounded by expiration; an early trigger delays the next window to preserve the set rate |
| PSV | Ends support at flow reversal during the pressure rise, at 25% of peak flow afterward, or after 4 seconds; `vent.apnea_backup_s` sets the apnea backup delay |
| CPAP | Detects breaths independently of the support trigger setting and marks expiration at flow reversal |

SIMV follows the Dräger adult synchronization convention. Mandatory breaths
account for gas already inspired; transitions from pressure support to
mandatory pressure control preserve pressure. A zero mandatory rate leaves
spontaneous breathing with configured support. Untriggered efforts also use
the inspiratory limb and expiratory PEEP valve. Mandatory pressure breaths end
at the set inspiratory time, which defaults to 1 second for SIMV and PSV backup.

`set_vent_settings` applies inspiratory settings at the next breath and PEEP
at expiration, preserving lung volume and the current inspiratory pressure
trace. `set_vent_power` starts or stops ventilation; settings persist while off.

Spirometry covers ventilated, bagged, and unassisted breaths, including efforts
below the ventilator trigger threshold. Expired tidal volume (VTe) is the last
completed breath's expired volume. Respiratory rate (RR), minute ventilation
(MV), and gas exchange use four-breath averages that clear after 15 seconds
without a breath. Mask leak reduces delivered gas; tracheal-tube obstruction
raises resistance. See [ventilator tests](../tests/test_ventilator.py).

Primus recordings informed compliance and sensor fits. CT atelectasis data
informed recruitment pressure and timing, but do not determine gas volume or
perfusion. Recruitment-to-shunt mapping, assisted effort, and obstructed
expiration remain teaching estimates; closure is independent of gas composition.

### Monitors

The airway-pressure trace includes sensor filtering and expiratory PEEP-valve
resistance. Flow uses gas volume moved during each sampling interval; volume
is the filtered integral of that flow. Pressure-volume and flow-volume loops
use signals from the same sample and show current and previous breaths.
Patient effort can alter loops or reverse flow during a mandatory pressure plateau.

VCV reports plateau pressure after a pause. PCV and PCV-VG report it only when
end-inspiratory flow and its resistive pressure drop are negligible. All modes
leave plateau blank during patient effort. Displayed PEEP is airway pressure;
auto-PEEP affects plateau pressure and residual flow. Alarms cover high airway
pressure, low minute ventilation, low delivered tidal volume, and low inspired O₂.

ECG, arterial pressure, and pleth share beat timing and sample at intervals
of 10 ms or less. [`ArterialWaveformRenderer`](../anasim/monitors/arterial.py)
uses Mahdi 2017 pressure points with Su MAP, stroke volume, and arterial
compliance; `ArterialLineMonitor` adds catheter and transducer dynamics.
Rhythms alter filling; displayed heart rate averages R-R intervals, and NIBP
updates after each cuff cycle. See [waveform](../tests/test_arterial_waveform.py)
and [arterial line](../tests/test_arterial_line.py) tests.

BIS includes sevoflurane in the selected propofol-remifentanil model, with
smoothing and delay. Poor perfusion delays SpO₂ and reduces pleth amplitude;
SpO₂ requires an organized rhythm and adequate perfusion.

[`Capnograph`](../anasim/monitors/capno.py) tracks exhaled gas through series
dead space. Analyzer response, phase II spread, and phase III slope are fitted
to Primus recordings. Inspiratory efforts can produce curare clefts; see
[capnography tests](../tests/test_nibp_capno.py). RR uses capnography with an
airway and chest impedance ("RR imp") without one. End-tidal values clear
15 seconds after valid exhaled CO₂ stops. The monitor shows end-tidal,
age-adjusted MAC (`et_mac`); anesthetic effects use brain MAC (`mac`).

### PK and TCI

[`MammillaryPK`](../anasim/patient/pk_models.py) uses a central compartment,
up to two peripheral compartments, and an effect site. TCI and initialization
use the same state ordering.

Central volume scales with blood volume, preserving drug amount; other
concentrations stay unchanged. Clearances scale with CO by per-drug exponents.
Propofol and remifentanil retain their population clearances, which have no CO
covariate, so low-output effects on their kinetics are not modeled. Epinephrine
clearance is also independent of CO. Shed-blood drug loss is not tracked
separately. Li norepinephrine concentrations include endogenous secretion,
so they stay above zero without an infusion.

TCI updates rates every 10 seconds, within pump limits, to keep predicted
concentrations at or below target over a 10-minute horizon. Material PK changes
rebuild the prediction model; boluses resynchronize concentration estimates.
Setting a manual rate disables TCI for that drug.

`SimulationConfig(tci_enabled=True)` enables TCI. The default is `False`, so
maintenance retains the manual rates used to seed its drug history.
[User rates](CLI_USAGE.md#python-use) convert to absolute model units internally.

### Temperature and stimulation

Induction redistributes core heat to the periphery. Noxious responses scale
with laryngoscopy response probability; opioids reduce hemodynamic and BIS
responses.

## Browser app

The local and hosted apps share `anasim/web_assets/` and
[`WebSession`](../anasim/web.py), which applies commands and returns JSON
snapshots of measurements, waveforms, controls, and scenario objectives.

- [`local.py`](../anasim/local.py) serves on 127.0.0.1 under a random path and
  accepts same-origin requests. The server advances simulation on its own clock
  while the page polls. Reloading reconnects to the paused session. After five
  seconds without polling, the server pauses simulation and closes the recording.
  Files remain in the recording directory.
- The hosted app runs Python in a Pyodide worker with NumPy and SciPy. Stopping
  a recording downloads it. `scripts/build_web.py` builds `build/web`;
  [pages.yml](../.github/workflows/pages.yml) publishes it.

## Scenario objectives

`engine.actions` records controls and event transitions with simulation times.
`WebSession` calls `begin_step()` when an objective becomes active.

| Objective | Example | Checked against |
|-----------|---------|-----------------|
| Action | "Start a 500 mL bolus" | Actions since objective activation, plus state where relevant |
| State | "MAP ≥ 65" | Current monitor values and control state |

Action checks use log positions because actions taken while paused share
timestamps. Queries raise an error when no objective is active. A step's
`on_met` runs once when its check first passes; intubation uses it to start the
laryngoscopy stimulus.

Fluid objectives count requested bolus volume; delivery continues at the pump
rate. Dosing instructions follow the session's TCI setting.
