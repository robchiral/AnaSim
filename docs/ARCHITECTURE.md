# AnaSim architecture

AnaSim separates pharmacology, physiology, the anesthesia machine, and
monitors. The runtime advances them in a fixed order and copies outputs into
`SimulationState`.

## Simulation state

`SimulationState` stores physiology separately from monitor measurements.

| Group | Fields | Meaning |
|-------|--------|---------|
| Physiology | `map`, `hr`, `co`, `sv`, `svr`, `sao2`, `pa_co2`, `alveolar_co2`, `pao2` | Values used by physiology, analytics, and endpoints |
| Ideal arterial pulse | `sbp`, `dbp` | Pressures derived from Su MAP and stroke volume |
| Arterial catheter | `art_pressure`, `art_sbp`, `art_dbp`, `art_map` | Instantaneous and completed-beat pressures after catheter filtering |
| Other monitors | `nibp_sys`, `nibp_dia`, `nibp_map`, `display_hr`, `display_bis`, `display_etco2`, `display_spo2` | Displayed measurements |
| Ventilator | `paw_peak`, `paw_plat`, `paw_mean`, `peep`, `compliance_dyn`, `et_o2` | Breath pressures, dynamic compliance, and end-tidal O2 |
| Fluids | `blood_volume`, `urine_out_ml`, `net_fluid_ml`, `lap`, `lung_water` | Blood volume, fluid balance, LAP, and lung water above normal |

Monitors read physiology without changing it. NIBP measures `sbp`, `dbp`, and
`map`; displays and clinical checks use `art_*` with an arterial line and
`nibp_*` otherwise. The recorder writes both sets. `engine.output_buffer`
holds 20 seconds of waveforms for the sweep and loops.

## Main modules

[`engine.py`](../anasim/core/engine.py) holds subsystems and applies controls.
[`runtime.py`](../anasim/core/runtime.py) advances them,
[`projection.py`](../anasim/core/projection.py) writes state, and
[`monitors.py`](../anasim/core/monitors.py) updates measurements and alarms.
[`initialization.py`](../anasim/core/initialization.py) sets the starting state.
[`drug_registry.py`](../anasim/core/drug_registry.py) defines drug units, pump
limits, and UI metadata.

## Step order

```text
SimulationEngine.step()
 1. Anesthetic depth, metabolic rate, and tolerance of stimulation
 2. Disturbances and clinical events
 3. PK scaling to blood volume and CO; resynchronize active TCI
 4. TCI infusion rates
 5. Machine: vaporizer, circuit
 6. PK: plasma and effect-site concentrations, TOF, volatile uptake
 7. Physiology: ventilator or bag with lung mechanics, gas exchange, hemodynamics
 8. Copy outputs into SimulationState
 9. Monitors: cardiac cycle, arterial pulse and line, ECG, pleth, NIBP,
    capnography, alarms
10. Shivering
11. Temperature
12. Cardiac arrest check
```

## Inputs from the previous step

| Value | Used by | Updated by |
|-------|---------|------------|
| `state.co` | Volatile PK scaling, respiration | Hemodynamics |
| `state.va` | Volatile PK | Respiration |
| `state.mv` | Circuit and machine | Physiology |

These modules use values from the previous step because the listed updates
happen later.

## Initialization

`awake` starts from patient baselines. `steady_state` seeds a maintenance
history and settles drugs, gases, and physiology. No vasopressor is running, so
MAP is the untreated anesthetic response and may need treatment. The session
clock, recording, display history, arrest checks, and visible fluid and
temperature totals start after initialization.

## Supported patient domain

Inputs cover adults aged 18 to 70 years, 50 to 100 kg, 150 to 200 cm, and
BMI 18 to 32 kg/m². Li et al. 2024 studied 36 healthy volunteers aged 18 to
70 years, weighing 51.5 to 94.8 kg, 151 to 196 cm tall, with BMI 18.0 to
31.1 kg/m². AnaSim slightly extends those body-size ranges. Su et al. 2023
used a cohort of the same size and age groups. In their model, age strongly
affects the stroke-volume response to propofol. Hemoglobin, hematocrit, and
organ-function limits are simulator choices. All input limits are in
[CLI fields](CLI_USAGE.md#fields).

## Model notes

Sources are listed in [Model references](REFERENCES.md). AnaSim adapts published
models and calibrates additional responses for teaching. Linked tests define
expected response ranges.

### Hemodynamics

AnaSim extends Su et al. 2023 with blood volume, pulmonary circulation,
vasoactive drugs, a baroreflex, septic shock, and anaphylaxis. Su drug-effect
parameters keep their published values, and concentration-step responses match
an independent integration of the published equations. Baseline HR and MAP are
resting values, so the study's initial anxiety-related HR and SV transients are
omitted. Cardiovascular propofol and opioid effects use plasma concentrations;
hypnosis and respiratory depression use effect-site concentrations. Applying
fentanyl's remifentanil equivalent to cardiovascular effects is a simulator
approximation.

[`HemodynamicConfig`](../anasim/physiology/hemo_config.py) combines published
anesthetic reflex effects and hypoxic arrest thresholds with calibrated reflex
gains, set-point reset, and hypoxia time constants. The baroreflex adjusts HR,
or vascular resistance when a rhythm fixes the rate. Severe hypoxia reduces
HR and contractility.

Epinephrine, phenylephrine, vasopressin, dobutamine, and milrinone use
concentration-response curves. Published data guide response direction and
size; combined effects are calibrated. Epinephrine fits arterial infusion
(Freyschuss 1986) and IV bolus (Takahashi 2002) data.
Esmolol (Sum 1983), labetalol (Abernethy 1987; Hafsa 2022), and glycopyrrolate
(Ali-Melkkilä 1993) use published antagonist potencies, with calibrated
sympathetic and vagal contributions to resting tone and reflexes. In AnaSim,
part of labetalol's BP reduction persists during beta blockade because the
sinus baroreflex adjusts only HR. Abernethy's study reported BP recovery
within 30 minutes.
Checks cover [hemodynamics](../tests/test_hemodynamics.py),
[epinephrine](../tests/test_epinephrine.py), and
[autonomic drugs](../tests/test_autonomic_drugs.py).

Crystalloid keeps 30% of its volume in the blood and albumin 80%; the rest
enters an interstitial pool that returns only to replace a blood-volume
deficit. Volume expansion adds urine, less under anesthesia (Reid 2003; Hahn
2010). Above LAP 20 mmHg, fluid filters into the lung; beyond 3 mL/kg it
floods alveoli, which shunt and stiffen the lung until PEEP re-aerates them
(Malo 1984). Filtration and flooding rates are teaching estimates. Oncotic
pressure and cardiac dysfunction are not modeled.

With `end_on_cardiac_arrest`, MAP below 20 mmHg or HR below 10 bpm for
15 seconds ends the session. Cardiac arrest resuscitation is not modeled.

### Drug effects

[`anesthesia.py`](../anasim/patient/pd/anesthesia.py) calculates remifentanil
equivalents for fentanyl using isoflurane MAC reduction (McEwan 1993; Lang
1996), and propofol equivalents for etomidate and midazolam at equal
loss-of-response concentrations (Kaneda 2011; Albrecht 1999). Midazolam-propofol
synergy follows Short 1992. Etomidate's ventilatory effect (Valk 2021), the
maximum midazolam equivalent, and ketamine's
laryngoscopy potency and sympathetic response (Idvall 1979) are simulator
calibrations. Lidocaine blocks part of the response to stimulation, in
proportion to its reduction of anesthetic requirement (Himes 1977). It has no
hypnotic, hemodynamic, or antiarrhythmic effect, and toxicity is not modeled.
See [adjunct tests](../tests/test_anesthetic_adjuncts.py).

[`TOFModel`](../anasim/patient/pd/nmba.py) separates adductor pollicis from
diaphragm and larynx effects, allowing breathing to recover before TOF. Both
central muscles share laryngeal kinetics. It models spontaneous recovery and
simplified sugammadex binding, with calibrated onset and recovery constants.
See [pharmacology tests](../tests/test_pharmacology.py).

### Respiration

[`RespiratoryModel`](../anasim/physiology/respiration.py) combines published
ventilatory and hypercapnic responses with calibrated parameters. Anesthetics
and opioids reduce drive. Ventilation can suppress breathing in unconscious
patients below the apneic PaCO2 threshold. Awake CO2 feedback changes depth
before frequency; hypocapnia preserves rhythmic breathing.
Awake gains are teaching estimates. Positive pressure or an ETT relieves
upper-airway obstruction after loss of consciousness.

Gas exchange tracks alveolar gas and blood oxygen stores for preoxygenation,
apnea, and blood loss. Alveolar, arterial, and end-tidal CO2 are separate;
low CO widens the PaCO2-EtCO2 gap. Oxygen, sevoflurane, and N2O share lung gas
volume and shunt from perfused closed units. Oxygen calculations use standard
gas conditions and hemoglobin stores. See
[respiratory tests](../tests/test_respiration.py).

### Ventilation mechanics

[`RespiratoryMechanics`](../anasim/physiology/resp_mech.py) combines resistance,
published viscoelastic parameters, compliance fitted to VitalDB, and a
simplified muscle-pressure profile. Delivered assistance reduces
next-breath effort; inflation ends effort earlier under anesthesia.
Bronchospasm increases expiratory resistance as volume falls and can limit
flow, causing trapping and intrinsic PEEP.

[`LungAeration`](../anasim/physiology/lung.py) shares aeration and gas volume
with gas exchange and volatile uptake. Recruitment and closure depend on
pressure history; high inflation stiffens the lung. Compliance scales with
predicted body weight and BMI. Pressure-volume curve updates conserve gas
volume. The mechanics solver splits each step at breath events, pressure
limits, and fixed aeration and expiration update times.

[`AnesthesiaVentilator`](../anasim/machine/ventilator.py) uses GE Aisys CS2 modes
and ranges, with defaults informed by Primus recordings and device conventions.
It supports VCV, PCV, PCV-VG, SIMV, PSV with apnea backup, and CPAP.
VCV enforces its pressure limit during flow delivery and the inspiratory pause.
PCV-VG adapts pressure to volume by at most 3 cmH2O per breath, with pressure
targets capped 5 cmH2O below Pmax, including after Pmax or PEEP edits.

SIMV synchronizes with efforts during the final 5 seconds of each scheduled
cycle, capped at the expiratory duration, following the Dräger adult convention.
An early trigger delays the next synchronization window to preserve the set
rate. Mandatory strokes account for gas already inspired in the same breath;
transitions from pressure support to mandatory pressure control preserve pressure.
A zero mandatory rate leaves spontaneous breathing with configured support.

Pressure support cycles off at flow reversal during the pressure rise, at 25%
of peak flow afterward, or after 4 seconds. CPAP detects breaths independently
of the support trigger setting and marks expiration at flow reversal.
Untriggered efforts also use the inspiratory limb and expiratory PEEP valve.
Mandatory pressure breaths end at the set Ti, which defaults to 1 second for
SIMV and PSV backup. `vent.apnea_backup_s` sets PSV's backup delay.

`set_vent_settings` applies inspiratory settings at the next breath and PEEP
at expiration, preserving lung volume and the current inspiratory pressure
trace. Use `set_vent_power` to start or stop ventilation.

Spirometry covers ventilated, bagged, and unassisted breaths, including efforts
below the ventilator trigger threshold.
VTe is the last expired volume; RR, MV, and gas exchange use four-breath averages
that clear after 15 seconds without a breath. Mask leak reduces delivered gas;
ETT obstruction raises resistance.
See [ventilator tests](../tests/test_ventilator.py).

Primus recordings informed compliance and sensor fits. CT atelectasis data
informed recruitment pressure and timing, but do not determine gas volume or
perfusion. Recruitment-to-shunt mapping, assisted effort, and obstructed
expiration remain teaching estimates; closure is independent of gas composition.

### Monitors

Airway pressure uses sensor filtering and expiratory PEEP-valve resistance.
Flow uses gas volume moved during each sampling interval. Volume is the
filtered integral of that flow. Both loops use signals from the same sample
and show current and previous breaths. Patient effort can bend loops or reverse
flow during a mandatory pressure plateau.

VCV reports plateau pressure after a pause. PCV and volume guarantee report
it only when end-inspiratory flow and its resistive pressure drop are negligible.
All modes leave plateau blank during patient effort. Displayed PEEP is airway
pressure; auto-PEEP affects plateau pressure and residual flow. Alarms cover
pressure, minute ventilation, delivered VT, and inspired O2.

ECG, arterial pressure, and pleth share beat timing and sample at intervals
of 10 ms or less. [`ArterialWaveformRenderer`](../anasim/monitors/arterial.py)
uses Mahdi 2017 pressure points with Su MAP, stroke volume, and arterial
compliance; `ArterialLineMonitor` adds catheter and transducer dynamics.
Rhythms alter filling; HR averages R-R intervals, and NIBP updates after each
cuff cycle.
See [waveform](../tests/test_arterial_waveform.py) and
[arterial line](../tests/test_arterial_line.py) tests.

BIS includes sevoflurane in the selected propofol-remifentanil model, with
smoothing and delay. Poor perfusion delays SpO2 and reduces pleth amplitude;
SpO2 requires an organized rhythm and adequate perfusion.

[`Capnograph`](../anasim/monitors/capno.py) tracks exhaled gas through series
dead space. Analyzer response, phase II spread, and phase III
slope are fitted to Primus recordings. Inspiratory efforts can produce curare
clefts; see [capnography tests](../tests/test_nibp_capno.py). RR uses capnography
with an airway and chest impedance ("RR imp") without one. End-tidal values
clear 15 seconds after valid exhaled CO2 stops. The monitor shows end-tidal
age-adjusted MAC (`et_mac`); anesthetic effects use brain MAC (`mac`).

### PK and TCI

`MammillaryPK` uses up to two peripheral compartments and an effect site,
with consistent state ordering for TCI and initialization.

Hemodynamics scale central volume with blood volume, preserving drug amount;
other concentrations stay unchanged. Clearances scale with CO by per-drug
exponents. Propofol and remifentanil keep their population clearances, which
have no CO covariate, so low-output effects on their kinetics are not modeled.
Epinephrine clearance is also independent of CO. Shed-blood drug loss is not
tracked separately. Li norepinephrine concentrations include
endogenous secretion, so they stay above zero without an infusion.

TCI recalculates every 10 seconds to keep predicted peaks at or below target
for ten minutes, within pump limits. PK changes rebuild controllers; boluses
resynchronize concentrations. Manual rates disable TCI for that drug.
TCI requires `SimulationConfig(tci_enabled=True)`. With the default `False`,
maintenance retains the manual rates used to seed its drug history.
[User rates](CLI_USAGE.md#python-use) convert to absolute model units internally.

### Temperature and stimulation

Induction redistributes core heat to the periphery. Noxious responses scale
with laryngoscopy response probability; opioids reduce hemodynamic and BIS
responses.

## Browser app

Both versions use `anasim/web_assets/` and [`WebSession`](../anasim/web.py),
which applies commands and returns JSON snapshots of measurements, waveforms,
controls, and the current objective.

- [`local.py`](../anasim/local.py) serves on 127.0.0.1 under a random path,
  accepts same-origin requests, and steps on its own clock while the page polls.
  Reloading reconnects to the paused session. Five seconds without polling
  pauses simulation and closes recording; files stay in the recording directory.
- The hosted app runs Python in a Pyodide worker with numpy and scipy.
  Stopping a recording downloads it. `scripts/build_web.py` builds `build/web`;
  `.github/workflows/pages.yml` publishes it.

## Scenario objectives

`engine.actions` records controls and event transitions with simulation times.
`WebSession` calls `begin_step()` when an objective becomes active.

| Kind | Example | Check reads |
|------|---------|-------------|
| Action | "Start a 500 mL bolus" | Actions since objective activation, plus state where relevant |
| State | "MAP ≥ 65" | Current monitor values and control state |

Action checks use log positions because paused actions share timestamps.
Queries raise an error when no objective is active. A step's `on_met` runs
once when its check first passes; intubation uses it to start the
laryngoscopy stimulus.

Fluid objectives count requested bolus volume; delivery continues at the pump
rate. Dosing instructions follow the session's TCI setting.
