# AnaSim architecture

AnaSim combines pharmacology, physiology, the anesthesia machine, and monitor
models. The runtime advances these models and writes their outputs to
`SimulationState`. [CLI and Python usage](CLI_USAGE.md) covers configuration,
units, and scripted runs.

## Simulation state

`SimulationState` contains physiological values and monitor measurements.

| Group | Selected fields | Meaning |
|-------|-----------------|---------|
| Physiology | `map`, `hr`, `co`, `sv`, `sao2`, `pa_co2`, `pao2` | Values used by the models and cardiac arrest check |
| Arterial pulse | `sbp`, `dbp` | Pressures derived from mean pressure, stroke volume, and arterial compliance |
| Arterial catheter | `art_pressure`, `art_sbp`, `art_dbp`, `art_map` | Pressure after catheter and transducer filtering |
| Other monitors | `nibp_map`, `display_hr`, `display_bis`, `display_etco2`, `display_spo2` | Monitor measurements |
| Ventilation | `paw_peak`, `paw_plat`, `paw_mean`, `peep`, `compliance_dyn`, `et_o2` | Breath pressures, compliance, and end-tidal oxygen |
| Fluids | `blood_volume`, `urine_out_ml`, `net_fluid_ml`, `lap`, `lung_water` | Blood volume, fluid balance, left atrial pressure, and excess lung water |

Monitors read physiological values. Pressure displays and scenario objectives
use the configured arterial or noninvasive blood pressure (NIBP) monitor. CSV
recordings include both physiological values and measurements.
[`state.py`](../anasim/core/state.py) defines the fields and units.

## Main modules

| Module | Responsibility |
|--------|----------------|
| [`engine.py`](../anasim/core/engine.py) | Creates the models and applies controls |
| [`runtime.py`](../anasim/core/runtime.py) | Advances the simulation |
| [`projection.py`](../anasim/core/projection.py) | Copies model outputs into the shared state |
| [`monitors.py`](../anasim/core/monitors.py) | Updates measurements and alarms |
| [`initialization.py`](../anasim/core/initialization.py) | Configures starting states |
| [`drug_registry.py`](../anasim/core/drug_registry.py) | Defines drug models, units, pump limits, state fields, and interface metadata |

## Step order

Each `SimulationEngine.step()` call performs these operations:

1. Calculate anesthetic depth, metabolic rate, and stimulation tolerance.
2. Apply stimulation, hemorrhage, fluids, and other clinical events.
3. Update pharmacokinetic (PK) parameters for blood volume, cardiac output, and
   drug interactions, then calculate target-controlled infusion (TCI) rates.
4. Update the airway, vaporizer, and breathing circuit.
5. Update drug concentrations, inhaled anesthetic uptake, and neuromuscular block.
6. Advance breathing mechanics, gas exchange, and hemodynamics.
7. Copy outputs into `SimulationState` and update monitor measurements and alarms.
8. Update shivering and temperature, then check for cardiac arrest.

PK and circuit calculations use cardiac output and alveolar ventilation from
the start of the step. Physiology supplies these values for the next step.

## Initialization

`awake` starts from patient baselines. `steady_state` initializes a maintenance
anesthetic after 30 minutes of drug administration, starts controlled
ventilation, and allows the models to settle. Vasopressor infusion rates start
at zero; mean arterial pressure (MAP) reflects the untreated anesthetic response.
The session clock and recordings start after initialization. Fluid totals start
at zero.

## Supported patient domain

AnaSim supports adults aged 18 to 70 years. The [input limits](CLI_USAGE.md#fields)
define the accepted body size, hemoglobin, and organ-function ranges. These are
simulation boundaries. Published model parameters represent typical responses
within their source cohorts.

## Model notes

[Model references](REFERENCES.md) lists the source studies. Simulator
calibrations define combined drug responses, reflexes, and volume and airway
effects. Tests check equations, conservation, and clinical responses; they do
not establish accuracy for
individual patients.

### Hemodynamics

The Su 2023 turnover model describes propofol-remifentanil cardiovascular
effects using plasma concentrations. The model includes blood volume, pulmonary
circulation, reflexes, vasoactive drugs, and distributive shock. MAP feedback
accounts for rhythm, hypoxia, and stimulation. The baroreflex adjusts heart rate
or vascular resistance when a rhythm fixes the rate. Severe hypoxia reduces
rate and contractility.

| Drug | Sources and implementation |
|------|----------------------------|
| Norepinephrine | Li 2024 PK with endogenous secretion and propofol-dependent clearance; integrated response calibrated to de Keijzer 2026 infusion and Joachim 2024 bolus data |
| Epinephrine | Ensinger 1992 arterial clearance; mixing volume and cardiac and vascular response timing calibrated to Freyschuss 1986 infusion and Takahashi 2002 bolus data |
| Phenylephrine, vasopressin, dobutamine, milrinone | Concentration-response curves informed by published data |
| Esmolol, labetalol, glycopyrrolate | Published antagonist potencies with calibrated sympathetic and vagal effects |

Anesthetic depth increases the norepinephrine vascular and stroke-volume
responses. Applying this calibration to volatile anesthesia and populations
outside the source cohorts is an assumption. Severe anaphylaxis reduces
adrenergic sensitivity.

Blood loss lowers filling pressure and preload; reflex venoconstriction partly
compensates. Crystalloid retains 30% of infused volume in blood and albumin 80%.
The remaining volume enters an interstitial pool that can refill blood-volume
deficits. Urine output depends on pressure, volume expansion, renal function,
and anesthetic state. Elevated left atrial pressure increases lung water, shunt,
and stiffness. Positive end-expiratory pressure (PEEP) can recruit affected lung
units. These volume responses use simulator calibrations.

With `end_on_cardiac_arrest=True`, MAP below 20 mmHg or heart rate below 10 bpm
for 15 seconds ends the session.

### Drug effects

AnaSim uses separate effect sites for bispectral index (BIS), clinical
responsiveness, and breathing.

| Endpoint | Model |
|----------|-------|
| Propofol PK | Eleveld 2018 arterial model with the opioid-regimen covariate |
| Remifentanil PK | Eleveld 2017 arterial model |
| Baseline BIS | Eleveld 2018 curve with age-dependent potency and processing delay |
| Clinical response and laryngoscopy tolerance | Bouillon 2004 Bayesian hierarchy |
| Ventilatory depression | Bouillon 2004 propofol curve with Olofsen 2010 opioid and CO₂ control |

Propofol and sevoflurane determine baseline BIS. Opioids blunt
stimulation-related BIS arousal through tolerance of the stimulus.
`loc` is the probability of no response to shaking or shouting; `tol` is the
probability of no response to laryngoscopy. Sevoflurane thresholds follow
Kuizenga 2019 and Hannivoort 2016 in age-adjusted minimum alveolar concentration
(MAC).

Applying Bouillon's individual concentration estimates to Eleveld population
concentrations, selecting clinical effect-site timing, and extending the
hierarchy to sevoflurane are modeling assumptions. BIS arousal and adjunct drug
combinations use simulator calibrations. Fentanyl contributes a remifentanil
equivalent; etomidate and midazolam contribute propofol equivalents, with
midazolam-propofol synergy. Nitrous oxide and ketamine affect clinical
responsiveness; ketamine and lidocaine also blunt stimulation responses.

Ketamine uses a separate cardiovascular effect site with a 2.28 min
equilibration half-time (Kamp 2021) and response magnitude calibrated to
Idvall 1979. The model represents total ketamine without separate enantiomers
or metabolites. Lidocaine reduces stimulation-related BIS and cardiovascular
responses without a baseline hypnotic effect (Hans 2010); its response
coefficient is a simulator estimate.

The [neuromuscular model](../anasim/patient/pd/nmba.py) uses Kleijn 2011 with
age, weight, and estimated creatinine clearance covariates. Sugammadex binds
rocuronium in plasma and increases its effect-site efflux. The complex clears
like sugammadex; free rocuronium drives muscle effects.

Train-of-four (TOF) represents the adductor pollicis response. Diaphragm and
larynx effects equilibrate faster and require higher rocuronium concentrations,
allowing breathing to recover before TOF. Creatinine clearance is estimated
from age and renal function. Sevoflurane scaling, TOF normalization, and
conversion to muscle effects use simulator calibrations.

### Respiration

[Ventilatory control](../anasim/physiology/respiration.py) combines anesthetic
effects, CO₂ feedback, and muscle strength. Propofol and sevoflurane mainly
reduce breath depth; opioids mainly slow rate. CO₂ accumulation can restore
breathing during opioid administration. Hypocapnia and neuromuscular block
suppress effort. Airway obstruction limits airflow while muscle effort
continues.

The respiratory model applies Olofsen's fractional opioid response to
Bouillon's nonlinear alveolar CO₂ response and extrapolates the low-dose
propofol interaction to other concentrations. Combined responses, breath
patterns, and apnea durations are simulator estimates.

Measured breaths drive gas exchange. Lung and blood oxygen stores support
preoxygenation, apnea, and blood loss responses and determine circuit oxygen
uptake. Opening an obstructed airway allows passive gas refill and
reoxygenation during apnea (Laviola 2020). Refill uses circuit oxygen; its timing
is estimated, and the pressure transient is not modeled.

Alveolar, arterial, and end-tidal CO₂ are separate; low cardiac output widens
the arterial to end-tidal gap. Oxygen and inhaled anesthetics share lung
gas volume and shunt from perfused closed units. Shunt mixing accounts for
hemoglobin-bound and dissolved oxygen, with extraction limited to available
oxygen.

### Ventilation mechanics

The [mechanics model](../anasim/physiology/resp_mech.py) combines airway
resistance, compliance, tissue viscoelasticity, and inspiratory muscle pressure.
Compliance scales with predicted body weight and body mass index (BMI).
[Lung aeration](../anasim/physiology/lung.py) depends on pressure history;
recruitment and closure change gas volume, shunt, and compliance. Volume is
continuous during PEEP and compliance changes.

Ventilatory assistance reduces muscle effort. Inflation ends effort earlier
under anesthesia. Airway and muscle pressures determine the intrathoracic
pressure used for venous return. Bronchospasm increases expiratory resistance
as volume falls and can limit flow, causing gas trapping and intrinsic PEEP.

The [ventilator](../anasim/machine/ventilator.py) uses GE Aisys CS2 modes and
ranges. [Mode names](../README.md#features) are defined in the README.

| Mode | Behavior |
|------|----------|
| VCV | Delivers set volume within the pressure limit |
| PCV | Maintains inspiratory pressure for the set inspiratory time |
| PCV-VG | Adjusts inspiratory pressure toward the target volume within the pressure limit |
| SIMV | Synchronizes mandatory breaths with patient efforts while preserving the set rate |
| PSV | Cycles support according to inspiratory flow, with apnea backup ventilation |
| CPAP | Maintains airway pressure during spontaneous breathing |

Mandatory breaths include gas already inspired during spontaneous effort.
Inspiratory settings apply at the next breath; PEEP applies during expiration.
Stopping ventilation preserves the settings. Spirometry includes ventilated,
bagged, and unassisted breaths. Expired tidal volume is the last completed
breath's volume. Respiratory rate and minute ventilation use four-breath
averages that clear after 15 seconds without a breath. Mask leak reduces
measured gas delivery; tube obstruction raises resistance.

Primus recordings inform compliance and sensor fits; CT atelectasis studies
inform recruitment pressure and timing. Recruitment-to-shunt mapping,
assisted effort, and obstructed expiration use simulator estimates. Lung
closure is independent of gas composition.

### Monitors

Airway pressure includes sensor filtering and expiratory valve resistance.
Flow reflects gas moved during each sampling interval; volume is its filtered
integral. Pressure-volume and flow-volume loops use synchronized samples.
Patient effort can alter loops or reverse flow during a pressure plateau.

VCV reports plateau pressure after an inspiratory pause. Pressure-control modes
report it when end-inspiratory flow and its pressure drop are negligible.
Plateau pressure is unavailable during patient effort. Displayed PEEP is airway
pressure; intrinsic PEEP affects plateau pressure and residual flow.

ECG, arterial pressure, and plethysmography share beat timing. The arterial
waveform uses Mahdi 2017 pressure points with modeled MAP, stroke volume, and
arterial compliance. Catheter and transducer dynamics affect arterial-line
measurements. Displayed heart rate averages beat intervals; NIBP updates after
each cuff cycle. Poor perfusion delays pulse oximetry and reduces pleth
amplitude. Pulse oximetry requires an organized rhythm and adequate perfusion.
BIS includes processing delay and smoothing.

[Capnography](../anasim/monitors/capno.py) tracks exhaled gas through dead space
and analyzer lag. End-tidal values clear 15 seconds after valid exhaled CO₂
stops. Without an airway, impedance respiratory rate counts breathing effort,
including during obstruction. The gas monitor displays end-tidal, age-adjusted
MAC; anesthetic effects use brain MAC.

### PK and TCI

Intravenous drugs use a central compartment, up to two peripheral compartments,
and endpoint-specific effect sites. The vascular part of the central volume, up
to the baseline blood volume, changes with blood volume while preserving drug
amount. Renal clearance follows renal perfusion, the same MAP curve as urine
output. Other clearances scale with cardiac output by drug-specific exponents.
Propofol, remifentanil, norepinephrine, and epinephrine use population clearances
at normal blood volume. Blood loss lowers propofol elimination and distribution
clearances and remifentanil elimination clearance (Johnson 2001, 2003).
Volume replacement restores these PK parameters. This scaling uses swine data;
it does not distinguish compensated and uncompensated shock or model persistent
propofol sensitivity after resuscitation (Kurita 2011; Johnson 2004).

Midazolam reduces propofol elimination and distribution clearances (Vuyk 2009).
Propofol reduces midazolam central volume, elimination clearance, and fast
distribution clearance (Lichtenbelt 2010). Covariate calculations cap midazolam
at 225 ng/mL and propofol at 1.2 mcg/mL. Applying these interaction coefficients
to Eleveld propofol and Albrecht midazolam is a modeling assumption.

TCI uses current PK parameters and concentrations to predict exposure over
10 minutes. Rates update at a nominal 10-second interval within pump limits.
Rocuronium TCI targets free plasma concentration. Its effect-site decay
prediction uses the neuromuscular model with current hemodynamic parameters
and inhaled MAC held constant. Norepinephrine predictions include endogenous
secretion. A manual rate disables TCI for that drug. `tci_enabled` selects TCI
or manual infusion rates for maintenance initialization.

### Temperature and stimulation

Awake thermoregulation holds core temperature at baseline. Anesthesia replaces
it with passive heat loss in proportion to depth, and induction redistributes
core heat to the periphery. Core temperature falls about 1.6 °C in the first
hour and 2.8 °C by 3 hours (Matsukawa 1995). Stimulation responses scale
with the probability of responding to laryngoscopy; opioids reduce the
hemodynamic and BIS responses.

## Browser app

The local and hosted apps share `anasim/web_assets/` and
[`WebSession`](../anasim/web.py), which applies commands and returns
measurements, waveforms, controls, and scenario objectives.

The local server binds to 127.0.0.1 and accepts same-origin requests under a
session-specific path. It advances simulation on its own clock. Reloading
reconnects to the paused session. After five seconds without polling, the
server pauses simulation and closes the recording.

The hosted app runs Python in a Pyodide worker with NumPy and SciPy. Stopping
a recording downloads the CSV. `scripts/build_web.py` creates `build/web`;
[pages.yml](../.github/workflows/pages.yml) publishes the hosted app.

## Scenario objectives

The action log records controls and clinical events with simulation times.
Objectives check actions taken after activation, current monitor values, and
control state. Log positions distinguish actions performed at the same paused
timestamp. Completion callbacks run once; the intubation objective starts the
laryngoscopy stimulus.

Fluid objectives count requested bolus volume while delivery continues at the
pump rate. Dosing instructions follow the session's TCI setting.
