# AnaSim architecture

AnaSim combines pharmacology, physiology, the anesthesia machine, and monitor
models. The runtime advances these models and writes their outputs to
`SimulationState`. [CLI and Python usage](CLI_USAGE.md) covers configuration,
units, and scripted runs.

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
effects.

### PK and TCI

Intravenous drugs use a central compartment, up to two peripheral compartments,
and endpoint-specific effect sites. The vascular part of the central volume, up
to the baseline blood volume, changes with blood volume while preserving drug
amount. Renal clearance follows renal perfusion, the same MAP curve as urine
output. Other clearances scale with cardiac output by drug-specific exponents,
except propofol, remifentanil, norepinephrine, and epinephrine, which use
population clearances. Blood loss lowers propofol clearances and remifentanil
elimination clearance, and volume replacement restores them (Johnson 2001,
2003, 2004). This scaling uses swine data, does not distinguish compensated
from uncompensated shock, and does not model persistent propofol sensitivity
after resuscitation (Kurita 2011).

Midazolam and propofol reduce each other's clearances (Vuyk 2009; Lichtenbelt
2010), with covariates capped at the studied concentrations. Applying these
coefficients to Eleveld propofol and Albrecht midazolam is an assumption.

TCI uses current PK parameters and concentrations to predict exposure over
10 minutes. Rates update at a nominal 10-second interval within pump limits.
Rocuronium TCI targets free plasma concentration. Its effect-site decay
prediction uses the neuromuscular model with current hemodynamic parameters
and inhaled MAC held constant. Norepinephrine predictions include endogenous
secretion. A manual rate disables TCI for that drug. `tci_enabled` selects TCI
or manual infusion rates for maintenance initialization.

### Anesthetic drug effects

AnaSim uses separate effect sites for bispectral index (BIS), clinical
responsiveness, and breathing.

| Endpoint | Model |
|----------|-------|
| Propofol PK | Eleveld 2018 arterial model with the opioid-regimen covariate |
| Remifentanil PK | Eleveld 2017 arterial model |
| Baseline BIS | Eleveld 2018 curve with age-dependent potency and processing delay |
| Clinical response and laryngoscopy tolerance | Bouillon 2004 Bayesian hierarchy; loss of response uses Schnider 1999's age term |
| Ventilatory depression | Bouillon 2004 propofol curve with Olofsen 2010 opioid and CO₂ control |

Propofol and sevoflurane determine baseline BIS. Opioids blunt
stimulation-related BIS arousal through tolerance of the stimulus.
The propofol curve predicts a population response. Prospective Eleveld studies
show substantial individual BIS prediction error (Vellinga 2021; Coetzee 2024).
Monitor processing and early distribution after a rapid bolus affect transition
accuracy (Sahinovic 2020; Struys 2007). The Eleveld processing delay is included;
delay estimates and equilibration constants from other PK models do not transfer
directly to this model.

`loc` is the probability of no response to shaking or shouting; `tol` is the
probability of no response to laryngoscopy. Sevoflurane thresholds follow
Kuizenga 2019 and Hannivoort 2016 in age-adjusted minimum alveolar concentration
(MAC).

Applying Bouillon's individual concentration estimates and Schnider's age term
to Eleveld population concentrations, selecting clinical effect-site timing,
and extending the hierarchy to sevoflurane are modeling assumptions. Older
patients lose responsiveness at a higher BIS (Lysakowski 2009). BIS arousal
and adjunct drug combinations use simulator calibrations. Fentanyl contributes
a remifentanil equivalent; etomidate and midazolam contribute propofol
equivalents at the patient's endpoint-specific potency, with midazolam-propofol
synergy. Nitrous oxide and ketamine affect clinical responsiveness; ketamine
and lidocaine also blunt stimulation responses.

Morphine and hydromorphone use Meissner 2025 PK with separate analgesic and
respiratory effect sites (Olofsen 2026). Analgesia contributes a calibrated
remifentanil equivalent; the respiratory site reduces ventilatory drive and
CO₂ response.

Ketamine has a separate cardiovascular effect site (Kamp 2021) and raises HR
and cardiac output with SVR unchanged (Tweed 1972). Anesthetic concentrations
depress the baroreflex (Akine 2001); analgesic doses do not (Huang 2020).
Enantiomers and metabolites are not modeled. Lidocaine reduces
stimulation-related BIS and cardiovascular responses without a baseline
hypnotic effect (Hans 2010); its response coefficient is a simulator estimate.

### Neuromuscular block

The [neuromuscular model](../anasim/patient/pd/nmba.py) uses Kleijn 2011 for
rocuronium, with age, weight, and estimated creatinine clearance covariates.
Sugammadex binds rocuronium in plasma and increases its effect-site efflux.
The complex clears like sugammadex; free rocuronium drives muscle effects.

Train-of-four (TOF) represents the adductor pollicis response. Diaphragm and
larynx effects equilibrate faster and require higher rocuronium concentrations,
allowing breathing to recover before TOF. Creatinine clearance is estimated
from age and renal function. Sevoflurane scaling, TOF normalization, and
conversion to muscle effects use simulator calibrations.

Succinylcholine uses Roy 2002 PK, with onset and recovery calibrated to Kopman
2003 and central muscle sensitivity to Dhonneur 1999. Its phase I block
depresses T1 without TOF fade.

Residual block increases susceptibility to hypnotic airway collapse
(Herbstreit 2009), with preserved awake patency and pressure relief.
This coupling uses simulator calibration.

### Hemodynamics

The Su 2023 turnover model describes propofol-remifentanil cardiovascular
effects using plasma concentrations. The model includes blood volume, pulmonary
circulation, reflexes, vasoactive drugs, and distributive shock. MAP feedback
accounts for rhythm, hypoxia, and stimulation. The baroreflex adjusts heart rate
or vascular resistance when a rhythm fixes the rate. Severe hypoxia reduces
rate and contractility.

Blood loss lowers filling pressure and preload; reflex venoconstriction partly
compensates. Awake responses to 1 L in 333 mL steps match Johnson 2014.
Anesthesia blunts hemorrhage tachycardia as it does the cardiac baroreflex
(Ebert 1992; Sellgren 1994). With age, cardiovagal baroreflex gain falls
(Kardos 2001), as do reflex and beta-agonist tachycardia, while vascular
responses are preserved (Laitinen 2004; Monahan 2007). Tachycardia scaling and
its transfer to hemorrhage and stimulation use simulator calibrations.

Propofol reduces stressed volume and venous resistance (de Wit 2016), while
the net response at normal blood volume follows Su. Phenylephrine recruits
venous blood and raises afterload, preserving output better at low preload
(Rebet 2016). These couplings use simulator calibrations.

| Drug | Sources and implementation |
|------|----------------------------|
| Norepinephrine | Li 2024 PK with endogenous secretion and propofol-dependent clearance; integrated response calibrated to de Keijzer 2026 infusion and Joachim 2024 bolus data |
| Epinephrine | Ensinger 1992 arterial clearance; mixing volume and cardiac and vascular response timing calibrated to Freyschuss 1986 infusion and Takahashi 2002 bolus data |
| Phenylephrine, vasopressin, dobutamine, milrinone | Concentration-response curves informed by published data |
| Ephedrine | Persky 2004 oral disposition and tolerance transferred to IV use; responses calibrated to Meng 2011 and Sjøen 2026 |
| Esmolol, labetalol, glycopyrrolate | Published antagonist potencies with calibrated sympathetic and vagal effects |
| Atropine | Hinderling 1985 PK and tissue-linked HR response; potency and propofol attenuation calibrated to Horiguchi 2002 |

Anesthetic depth increases the norepinephrine vascular and stroke-volume
responses. Applying this calibration to volatile anesthesia and populations
outside the source cohorts is an assumption. Ephedrine tachyphylaxis reduces
vascular, stroke-volume, and peak HR responses; a smaller HR response persists
(Persky 2004). Severe anaphylaxis reduces adrenergic sensitivity.

With `end_on_cardiac_arrest=True`, MAP below 20 mmHg or heart rate below 10 bpm
for 15 seconds ends the session.

### Fluids

Crystalloid represents lactated Ringer's solution and distributes between
blood and interstitium (Hahn 2017). Urinary elimination depends on MAP, age,
and renal function, with an integrated calibration to Reid 2003. Albumin retains 80% initially.
Non-exchanging fluid and capillary leakage enter a slower pool that can refill
blood-volume deficits. Fluid exchange during hemorrhage and sepsis, albumin
retention, and slow refill use simulator calibrations. Transfused red cells
carry donor hemoglobin (33.3 g/dL of cells) whatever the recipient's Hb/Hct
ratio. Elevated left atrial pressure increases lung water, shunt, and
stiffness. Positive end-expiratory pressure (PEEP) can recruit affected units.

### Respiration

[Ventilatory control](../anasim/physiology/respiration.py) combines anesthetic
effects, CO₂ feedback, and muscle strength. Propofol and sevoflurane mainly
reduce breath depth; opioids mainly slow rate. CO₂ accumulation can restore
breathing during opioid administration. Hypocapnia and neuromuscular block
suppress effort. Under anesthesia, breathing stops 4.5 mmHg below the resting
PaCO₂ that the drugs set (Hickey 1971). Airway obstruction limits airflow
while muscle effort continues.

The respiratory model applies Olofsen's fractional opioid response to
Bouillon's nonlinear alveolar CO₂ response and extrapolates the low-dose
propofol interaction to other concentrations. Combined responses, breath
patterns, and apnea durations are simulator estimates.

Hypoxic drive declines during sustained exposure and is attenuated by propofol
and sevoflurane (Nieuwenhuijs 2000; Sarton 1996). Timing and extrapolation beyond
these volunteer protocols use simulator calibrations.

Measured breaths drive gas exchange. Lung and blood oxygen stores support
preoxygenation, apnea, and blood loss responses and determine circuit oxygen
uptake. Opening an obstructed airway allows passive gas refill and
reoxygenation during apnea (Laviola 2020). Refill uses circuit oxygen; its timing
is estimated, and the pressure transient is not modeled.

Alveolar, arterial, and end-tidal CO₂ are separate; low cardiac output widens
the arterial to end-tidal gap. Oxygen and inhaled anesthetics share lung
gas volume and shunt. The A-a gradient follows sea-level means by age (Crapo
1999). Shunt mixing accounts for hemoglobin-bound and dissolved oxygen, with
extraction limited to available oxygen.

Hemoglobin saturation uses Dash 2016's pressure- and temperature-dependent
curve at reference pH, PCO₂, and 2,3-DPG. Temperature changes preserve the
combined lung and blood oxygen store.

### Ventilation mechanics

The [mechanics model](../anasim/physiology/resp_mech.py) combines airway
resistance, compliance, tissue viscoelasticity, and inspiratory muscle pressure.
Compliance scales with predicted body weight and body mass index (BMI).
[Lung aeration](../anasim/physiology/lung.py) depends on pressure history;
recruitment and closure change gas volume, shunt, and compliance. Most lost
aeration is the anesthetic fall in FRC (Hedenstierna 2015); its atelectatic
part shunts at most about 10% of cardiac output (Tokics 1996). Flooded units
shunt fully. Open-airway apnea times after 100, 80, and 60% O₂ match Edmark
2003. Volume is continuous during PEEP and compliance changes.

Primus recordings inform compliance and sensor fits; CT atelectasis studies
inform recruitment pressure and timing. Alveolar nitrogen slows recurrent
collapse (Rothen 1995); atelectatic shunt evolves separately from bulk FRC and
compliance. Gas-dependent closure, assisted effort, and obstructed expiration
use simulator calibrations.

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
measured gas delivery; tube obstruction raises resistance. Mean airway
pressure over recent breaths, including unassisted ones, holds the pharynx open
during mask ventilation.

### Monitors

ECG, arterial pressure, and plethysmography share beat timing. The arterial
waveform uses Mahdi 2017 pressure points with modeled MAP, stroke volume, and
arterial compliance, which falls after age 40 (Franklin 1997). Catheter and
transducer dynamics affect arterial-line measurements. Displayed heart rate
averages beat intervals; NIBP updates after each cuff cycle. Poor perfusion
delays pulse oximetry and reduces pleth amplitude. Pulse oximetry requires an
organized rhythm and adequate perfusion.

Airway pressure includes sensor filtering and expiratory valve resistance.
Flow reflects gas moved during each sampling interval; volume is its filtered
integral. Pressure-volume and flow-volume loops use synchronized samples.
Patient effort can alter loops or reverse flow during a pressure plateau.

VCV reports plateau pressure after an inspiratory pause. Pressure-control modes
report it when end-inspiratory flow and its pressure drop are negligible.
Plateau pressure is unavailable during patient effort. Displayed PEEP is airway
pressure; intrinsic PEEP affects plateau pressure and residual flow.

[Capnography](../anasim/monitors/capno.py) tracks exhaled gas through dead space
and analyzer lag. End-tidal values clear 15 seconds after valid exhaled CO₂
stops. Without an airway, impedance respiratory rate counts breathing effort,
including during obstruction. The gas monitor displays end-tidal, age-adjusted
MAC; anesthetic effects use brain MAC.

BIS includes processing delay and smoothing. The
[neuromuscular monitor](../anasim/monitors/neuromuscular.py) measures TOF
or single twitch on demand or at a set interval. TOF reports the twitch count
and, with all four twitches, the T4/T1 ratio, qualified for weak trains. Single
twitch reports T1 relative to an unblocked baseline. Display conventions follow
TwitchView; intermediate twitch amplitudes and signal thresholds are simulator
estimates.

### Temperature and stimulation

Awake thermoregulation holds core temperature at baseline. Anesthesia replaces
it with passive heat loss in proportion to depth, and induction redistributes
core heat to the periphery. Core temperature falls about 1.6 °C in the first
hour and 2.8 °C by 3 hours (Matsukawa 1995). Stimulation-related BIS arousal
scales with the probability of responding to laryngoscopy. HR and MAP responses
use stronger Bouillon stimulus intensities, fitted to the remifentanil
concentrations that blunted them in half of patients (Albertin 2005), and
resolve over about 5 minutes (Ebert 1992; Ebert 1995). Laryngospasm can follow
only the first 40 s of laryngoscopy.

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
laryngoscopy stimulus. Shock recognition requires MAP below 65 mmHg and 20%
below its value when the crisis started.

Fluid objectives count requested bolus volume while delivery continues at the
pump rate. Dosing instructions follow the session's TCI setting.
