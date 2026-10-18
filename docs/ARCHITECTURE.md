# AnaSim architecture

AnaSim combines patient, drug, physiology, anesthesia machine, and monitor
models in one simulation engine. Each subsystem keeps its own state. The runtime
updates the models and copies their outputs into `SimulationState`.

This guide describes how the components work and share data. [CLI and Python
usage](CLI_USAGE.md) covers configuration, units, and scripted runs. [Model
references](REFERENCES.md) lists the supporting studies.

## Main modules

| Module | Responsibility |
|--------|----------------|
| [`engine.py`](../anasim/core/engine.py) | Creates the subsystems, applies controls, and records simulation output |
| [`runtime.py`](../anasim/core/runtime.py) | Advances subsystems in step order |
| [`projection.py`](../anasim/core/projection.py) and [`state.py`](../anasim/core/state.py) | Copy model outputs into the shared state and define its fields |
| [`initialization.py`](../anasim/core/initialization.py) | Configures awake and maintenance starting states |
| [`patient/`](../anasim/patient/) and [`drug_registry.py`](../anasim/core/drug_registry.py) | Define patient characteristics, drug models, delivery units, pump limits, and interface metadata |
| [`physiology/`](../anasim/physiology/) | Models circulation, fluids, respiration, gas exchange, and temperature |
| [`machine/`](../anasim/machine/) | Models the breathing circuit, vaporizer, and ventilator |
| [`monitors/`](../anasim/monitors/) and [`core/monitors.py`](../anasim/core/monitors.py) | Produce measurements, waveforms, and alarms |
| [`web.py`](../anasim/web.py) and [`web_assets/`](../anasim/web_assets/) | Connect the engine and guided scenarios to the browser interface |

## Runtime

### Step order

Each `SimulationEngine.step()` call performs these operations:

1. Calculate anesthetic depth, metabolic rate, and stimulation tolerance.
2. Apply stimulation, hemorrhage, fluids, and other clinical events.
3. Update pharmacokinetic (PK) parameters for blood volume, cardiac output, and
   drug interactions, then calculate target-controlled infusion (TCI) rates.
4. Update the airway, vaporizer, and breathing circuit.
5. Update drug concentrations, inhaled anesthetic uptake, and neuromuscular block.
6. Advance breathing mechanics, gas exchange, and hemodynamics.
7. Copy outputs into `SimulationState` and update measurements and alarms.
8. Update shivering and temperature, then check for cardiac arrest.

PK and circuit calculations use cardiac output and alveolar ventilation from the
start of the step. Physiology supplies these values for the next step. The
engine then advances the simulation clock, retains waveform samples, and writes
the recording.

### Simulation state

`SimulationState` holds physiological values and monitor measurements.

| Group | Selected fields | Meaning |
|-------|-----------------|---------|
| Physiology | `map`, `hr`, `co`, `sao2`, `pa_co2`, `pao2` | Values used by the models and cardiac arrest check |
| Blood chemistry | `ph`, `hco3` | Arterial plasma pH and bicarbonate at 37 °C |
| Measurements | `art_map`, `nibp_map`, `display_spo2`, `display_bis` | Values after measurement and processing effects |
| Ventilation | `paw_peak`, `paw_plat`, `peep`, `et_o2` | Breath pressures and exhaled gas measurements |
| Fluids | `blood_volume`, `urine_out_ml`, `net_fluid_ml`, `lap`, `lung_water` | Circulating volume, fluid balance, filling pressure, and excess lung water |

Monitors read physiological values and apply sensor dynamics, sampling, and
averaging. Pressure displays and scenario objectives use the configured arterial
or noninvasive blood pressure (NIBP) monitor. CSV recordings include both
physiological values and measurements. `state.py` defines all fields and their
units.

### Initialization

`awake` starts from patient baselines. `steady_state` initializes a maintenance
anesthetic from a 30-minute drug history, starts controlled ventilation, and
allows the models to settle. Vasopressor infusion rates start at zero, so mean
arterial pressure (MAP) reflects the untreated anesthetic response. The session
clock, recordings, and fluid totals start after initialization.

Starting ventilation targets resting PaCO₂ by adjusting tidal volume within
limits based on predicted body weight, then respiratory rate. COPD uses a lower
rate ceiling to allow expiration; severe obstruction or obesity can leave PaCO₂
above target.

## Supported patient domain

AnaSim supports adults aged 18 to 70 years within the [input
limits](CLI_USAGE.md#fields) for body size, hemoglobin, organ function, and
patient conditions. These ranges define accepted simulation inputs.

Coronary disease limits coronary flow reserve. Myocardial oxygen demand follows
the rate-pressure product; supply depends on diastolic perfusion pressure,
diastolic time, and arterial oxygen content. Sustained imbalance produces
reversible contractile depression and illustrative ST depression.

Heart failure profiles change resting output, filling pressure, and sensitivity
to loading conditions (Schwartzenberg 2012; Westermann 2008). Reduced ejection
fraction increases afterload sensitivity. Preserved ejection fraction increases
stiffness and reduces filling during tachycardia.

COPD combines persistent obstruction, hyperinflation, gas trapping, dead space,
and impaired gas exchange with normal resting PaCO₂. Acute bronchospasm adds to
this baseline. Obesity reduces supine functional residual capacity (FRC) and
respiratory compliance and increases airway resistance (Pelosi 1998).

Pregnancy changes maternal blood volume, cardiac output, renal filtration, FRC,
heart rate, metabolism, and resting PaCO₂ with gestational age (Dallmann 2017;
Contreras 1991). Supine positioning reduces venous return in later pregnancy;
left lateral positioning restores it (Humphries 2019). Weight is pre-pregnancy
weight, entered hemoglobin and hematocrit are starting values, and drug
responses retain their adult calibrations.

## Model notes

Published models describe their study populations. AnaSim uses calibrations to
connect the models and represent patient conditions. The sections below describe
the main model choices and interactions. The implementation documents source
equations and coefficients.

### PK and TCI

Intravenous drugs use a central compartment, up to two peripheral compartments,
and effect sites for their clinical effects. Propofol uses Eleveld 2018 arterial
PK with its opioid-regimen covariate; remifentanil uses Eleveld 2017. Morphine
and hydromorphone use Meissner 2025 PK. Inhaled agents move between lung, blood,
and tissue compartments. Their effects follow brain partial pressures expressed
as age-adjusted minimum alveolar concentration (MAC) (Davis and Mapleson 1981;
Yasuda 1991).

Changes in blood volume rescale the vascular part of each drug's central volume
while conserving drug amount. Clearance responds to organ function, renal
perfusion, and cardiac output according to each model. Renal perfusion acts on
the renal clearance left after renal impairment. Blood loss lowers
propofol and remifentanil clearance; volume replacement restores it. This
scaling comes from swine data (Johnson 2001, 2003, 2004). Midazolam and propofol
reduce each other's clearances within the studied concentration ranges (Vuyk
2009; Lichtenbelt 2010).

TCI predicts exposure from current PK parameters and concentrations, then
adjusts infusion rates within pump limits. Rocuronium TCI targets free plasma
concentration, and its decay prediction includes sugammadex and inhaled-agent
effects. A manual infusion rate disables TCI for that drug. `tci_enabled`
selects TCI or manual rates for maintenance initialization.

### Anesthetic drug effects

Separate effect sites determine bispectral index (BIS), clinical responsiveness,
and breathing.

| Endpoint | Model |
|----------|-------|
| Baseline BIS | Eleveld 2018 propofol response with age-dependent potency and processing delay, plus a calibrated sevoflurane contribution |
| Clinical responsiveness | Bouillon 2004 response curves for shaking or shouting and laryngoscopy, with Schnider 1999's age effect on loss of response |
| Ventilatory depression | Bouillon 2004 propofol response with Olofsen 2010 opioid and CO₂ control |

`loc` is the probability of no response to shaking or shouting; `tol` is the
probability of no response to laryngoscopy. Sevoflurane thresholds follow
Kuizenga 2019 and Hannivoort 2016. Older patients lose responsiveness at a
higher BIS (Lysakowski 2009).

The response curves use remifentanil equivalents for fentanyl, morphine, and
hydromorphone, and propofol equivalents for etomidate and midazolam. Potency
depends on the patient and effect. Midazolam and propofol act synergistically
in the model. Nitrous oxide
and ketamine affect clinical responsiveness. Ketamine and lidocaine also blunt
stimulation responses.

Stimulation-related BIS arousal depends on laryngoscopy tolerance. HR and MAP
responses use separate, stronger stimulus intensities (Albertin 2005). The
laryngoscopy stimulus can trigger laryngospasm.

### Neuromuscular block

The [neuromuscular model](../anasim/patient/pd/nmba.py) uses Kleijn 2011 for
rocuronium and sugammadex. Sugammadex binds free plasma rocuronium and increases
its effect-site efflux; the complex clears like sugammadex. Both drugs use
measured creatinine clearance when supplied, or an estimate from age and renal
function. Other renal effects use the renal-function factor.

Train-of-four (TOF) represents the adductor pollicis response. Diaphragm and
larynx effects equilibrate faster and require higher rocuronium concentrations,
allowing breathing to recover before TOF. Residual block reduces pharyngeal
reserve and increases susceptibility to hypnotic airway collapse (Herbstreit
2009). Awake tone and positive airway pressure preserve patency.

Succinylcholine uses Roy 2002 PK, with twitch onset and recovery calibrated to
Kopman 2003 and central muscle sensitivity to Dhonneur 1999. Its phase I block
depresses single-twitch height without TOF fade.

### Hemodynamics

The [hemodynamic model](../anasim/physiology/hemodynamics.py) uses Su 2023's
turnover model for propofol-remifentanil effects, connected to blood volume,
pulmonary circulation, reflexes, vasoactive drugs, and distributive shock.
Propofol reduces stressed volume and venous resistance (de Wit 2016). Blood loss
lowers filling pressure and preload; reflex venoconstriction partly compensates.

MAP feedback adjusts heart rate or vascular resistance according to rhythm,
anesthetic depth, blood gases, and stimulation. Anesthesia blunts baroreflex and
hemorrhage tachycardia (Ebert 1992; Sellgren 1994). With age, cardiovagal gain
and reflex and beta-agonist tachycardia fall, while vascular responses are
preserved (Kardos 2001; Laitinen 2004; Monahan 2007). Severe hypoxia reduces
rate and contractility.

Vasoactive drugs modify vascular resistance, stroke volume, and heart rate;
adrenergic and muscarinic antagonists modify these responses and reflexes.
Norepinephrine includes endogenous secretion and propofol-dependent clearance
(Li 2024). Anesthetic depth increases its vascular and stroke-volume responses.
Epinephrine uses arterial clearance with calibrated cardiac and vascular
response timing (Ensinger 1992). Phenylephrine recruits venous blood and raises
afterload, making cardiac output depend on preload (Rebet 2016). Anaphylaxis
reduces adrenergic sensitivity.

Ephedrine uses Persky 2004 oral disposition and tolerance transferred to IV use,
with responses calibrated to Meng 2011 and Sjøen 2026. Tachyphylaxis reduces its
pressor and peak HR effects while a smaller HR response persists. Atropine's HR
response follows peripheral concentration (Hinderling 1985), and propofol
reduces this response (Horiguchi 2002). Ketamine raises HR and cardiac output through
sympathetic effects and depresses the baroreflex at anesthetic concentrations
(Tweed 1972; Akine 2001).

With `end_on_cardiac_arrest=True`, MAP below 20 mmHg or heart rate below 10 bpm
for 15 seconds ends the session.

### Fluids

Crystalloid represents lactated Ringer's solution and exchanges between blood
and interstitium (Hahn 2017). Urinary elimination depends on MAP, age, and renal
function. Albumin infusions keep more of their volume in the circulation
initially. Fluid leaves the circulation through capillary leakage and other slow
exchanges. A separate pool holds this fluid and slowly refills blood-volume
deficits.

Transfusion tracks donor red-cell volume and hemoglobin separately. Elevated
left atrial pressure increases lung water, shunt, and stiffness, linking fluid
balance to respiratory mechanics and gas exchange.

### Respiratory control

[Ventilatory control](../anasim/physiology/respiration.py) combines anesthetic
effects, CO₂ feedback, hypoxic drive, and muscle strength. Propofol and
sevoflurane mainly reduce breath depth; opioids mainly slow rate. CO₂
accumulation can restore breathing during opioid administration. Hypocapnia and
neuromuscular block suppress effort. Under anesthesia, the apneic threshold
tracks the drug-dependent resting PaCO₂ (Hickey 1971).

Morphine and hydromorphone rate effects set breath frequency separately from
ventilatory drive (Olofsen 2026). The controller combines Olofsen's fractional
opioid response with Bouillon's nonlinear alveolar CO₂ response. Hypoxic drive
declines during sustained exposure and is attenuated by propofol and sevoflurane
(Nieuwenhuijs 2000; Sarton 1996). Airway obstruction limits airflow while muscle
effort continues.

### Gas exchange and blood chemistry

Ventilation exchanges gas with lung stores, which support preoxygenation, apnea,
and blood loss responses and determine circuit oxygen uptake. Shunt mixes blood
oxygen contents, including hemoglobin-bound and dissolved oxygen. The baseline
A-a gradient follows sea-level means by age (Crapo 1999).

[Gas transport](../anasim/physiology/gas_transport.py) carries oxygen, CO₂, and
buffer base through pulmonary capillary, arterial, and mixed venous blood, with
shunt bypassing the lungs. Two tissue groups and an extracellular pool store CO₂
with calibrated capacities (Magor-Elliott 2021; Wooten 2003). Blood flow
controls exchange with these stores, and respiratory feedback follows arterial
PCO₂. During cardiac arrest, ventilation clears lung CO₂ while arterial blood
retains its contents.

Behind a closed airway, gas absorption shrinks the respiratory gas store and
promotes absorption atelectasis. This absorbed volume is tracked separately from
the bulk lung volume used by mechanics and inhaled agents. Opening the airway
refills the store with circuit gas (Laviola 2020).

Blood-gas chemistry describes respiratory acid-base changes at 37 °C. [Blood
buffering](../anasim/physiology/blood_chemistry.py) follows Rees 2010; oxygen
affinity follows Dash 2016's temperature, pH, and PCO₂ terms at reference
2,3-DPG. Exchange and affinity changes conserve CO₂, oxygen, and buffer base.
Changes in hemoglobin or hematocrit adjust the normal buffer contribution while
preserving respiratory buffer-base changes.

### Respiratory mechanics and aeration

The [mechanics model](../anasim/physiology/resp_mech.py) combines airway
resistance, compliance, tissue viscoelasticity, and inspiratory muscle pressure.
Body size and patient conditions set baseline mechanics. Ventilatory assistance
reduces muscle effort, and inflation ends effort earlier under anesthesia.
Airway and muscle pressures determine the intrathoracic pressure used for venous
return.

[Lung aeration](../anasim/physiology/lung.py) depends on pressure history.
Recruitment and closure change gas volume, shunt, and compliance. Alveolar inert
gas slows recurrent collapse. Atelectatic shunt evolves separately from bulk FRC
and compliance (Rothen 1995). Bronchospasm increases expiratory resistance as
volume falls and can limit flow, causing gas trapping and intrinsic positive
end-expiratory pressure (PEEP).

### Temperature

Awake thermoregulation maintains baseline core temperature. Anesthesia reduces
active regulation, increases heat loss, and redistributes core heat to the
periphery (Matsukawa 1995). Temperature changes metabolic demand, and warming
changes heat input. Opioids and neuromuscular block reduce shivering.

### Anesthesia machine and ventilation

The [circle system](../anasim/machine/circuit.py) mixes fresh gas with circuit
gas and accounts for patient uptake, surplus gas spill, and ambient air
entrainment when uptake exceeds fresh gas flow. Vaporizer output depends on
flow, dial setting, and available agent. Oxygen supply failure stops oxygen and
nitrous oxide delivery.

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

Mandatory breaths include gas already inspired during patient effort.
Inspiratory settings apply at the next breath; PEEP applies during expiration.
Mask leak reduces gas delivery, tube obstruction raises resistance, and mean
airway pressure over recent breaths holds the pharynx open during mask
ventilation.

## Monitors

ECG, arterial pressure, and plethysmography share beat timing. Arterial
waveforms reflect MAP, stroke volume, and age-dependent arterial compliance
(Mahdi 2017; Franklin 1997). Catheter and transducer dynamics filter arterial
pressure measurements. Heart rate averages beat intervals; pulse oximetry
responds to rhythm and perfusion, with poor perfusion delaying readings and
reducing plethysmographic amplitude.

Spirometry measures ventilated, bagged, and unassisted breaths. Expired tidal
volume comes from completed breaths; respiratory rate and minute ventilation use
recent breath averages. Pressure-volume and flow-volume loops use synchronized
pressure, flow, and volume samples. Plateau pressure requires negligible flow
and passive muscles. Displayed PEEP is airway pressure, while intrinsic PEEP
affects plateau pressure and residual flow.

[Capnography](../anasim/monitors/capno.py) includes dead-space transit and
analyzer lag. End-tidal readings clear when valid exhaled gas stops. Without an
airway, impedance respiratory rate counts breathing effort, including during
obstruction. The gas monitor displays end-tidal, age-adjusted MAC; anesthetic
effects use brain MAC.

BIS includes processing delay and smoothing. The [neuromuscular
monitor](../anasim/monitors/neuromuscular.py) samples TOF or single twitch on
demand or at a set interval. TOF reports twitch count and, with all four
twitches, the T4/T1 ratio. It flags ratios from weak twitches. Single twitch
reports T1 relative to an unblocked baseline.

## Browser app

The local and hosted apps share `web_assets/` and
[`WebSession`](../anasim/web.py), which applies commands and returns
measurements, waveforms, controls, and scenario objectives. The transport keeps
commands ordered, and simulation speed maps elapsed wall time to simulation
time.

The local Python server advances the session on its own clock and writes CSV
recordings to disk. It accepts same-origin requests under a session-specific
path on 127.0.0.1. Reloading reconnects to the paused session; loss of polling
pauses simulation and closes the recording.

The hosted app runs Python, NumPy, and SciPy in a Pyodide worker. Stopping a
recording downloads the CSV. [`build_web.py`](../scripts/build_web.py) creates
the hosted bundle, and [pages.yml](../.github/workflows/pages.yml) publishes it.

## Scenario objectives

The action log records controls and clinical events with simulation times.
Objectives combine actions taken since activation, current monitor values, and
control state. Log positions distinguish actions performed at the same paused
timestamp, and completion callbacks run once. In induction scenarios, completing
the intubation objective starts the laryngoscopy stimulus.

Fluid objectives count requested bolus volume while delivery continues at the
pump rate. Dosing instructions follow the session's TCI setting.

## Validation

Tests cover published equations, gas and drug conservation, monitor behavior,
and integrated clinical scenarios in native Python and Pyodide. [Recorded-case
comparisons](../validation/vitaldb/README.md) evaluate selected anesthetic and
respiratory components against VitalDB data. Each comparison applies to the
patients, conditions, and measurements it covers.
