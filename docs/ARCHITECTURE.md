# AnaSim architecture

AnaSim combines models of the patient, drugs, physiology, anesthesia machine,
and monitors in one simulation engine. Each model keeps its own state. At every
step, the runtime updates the models in a fixed order and copies their outputs
into `SimulationState`.

This guide explains how the models work and how they share data. [CLI and Python
usage](CLI_USAGE.md) covers configuration, units, and scripted runs, and [Model
references](REFERENCES.md) lists the supporting studies.

## Main modules

| Module | Responsibility |
|--------|----------------|
| [`engine.py`](../anasim/core/engine.py) | Creates the models, applies controls, and records simulation output |
| [`runtime.py`](../anasim/core/runtime.py) | Advances the models in step order |
| [`projection.py`](../anasim/core/projection.py) and [`state.py`](../anasim/core/state.py) | Copy model outputs into the shared state and define its fields |
| [`initialization.py`](../anasim/core/initialization.py) | Sets up awake and maintenance starting states |
| [`patient/`](../anasim/patient/) and [`drug_registry.py`](../anasim/core/drug_registry.py) | Define patient characteristics, drug models, delivery units, pump limits, and interface metadata |
| [`physiology/`](../anasim/physiology/) | Models circulation, fluids, respiration, gas exchange, and temperature |
| [`machine/`](../anasim/machine/) | Models the breathing circuit, vaporizer, and ventilator |
| [`monitors/`](../anasim/monitors/) and [`core/monitors.py`](../anasim/core/monitors.py) | Produce measurements, waveforms, and alarms |
| [`scenarios/`](../anasim/scenarios/) | Defines guided scenarios, their steps, and their objective checks |
| [`web.py`](../anasim/web.py) and [`web_assets/`](../anasim/web_assets/) | Connect the engine and guided scenarios to the browser interface |

## Runtime

### Step order

Each call to `SimulationEngine.step()` does the following:

1. Calculate anesthetic depth, metabolic rate, and stimulation tolerance.
2. Apply stimulation, hemorrhage, fluids, and other clinical events.
3. Update pharmacokinetic (PK) parameters for blood volume, cardiac output, and
   drug interactions, then calculate target-controlled infusion (TCI) rates.
4. Update the airway, vaporizer, and breathing circuit.
5. Update drug concentrations, inhaled anesthetic uptake, and neuromuscular
   block.
6. Advance breathing mechanics, gas exchange, and hemodynamics.
7. Copy outputs into `SimulationState` and update measurements and alarms.
8. Update shivering and temperature, then check for cardiac arrest.

PK and circuit calculations use the cardiac output and alveolar ventilation from
the start of the step, and the physiology models update these values for the
next step. The engine then advances the clock, stores waveform samples, and
writes the recording.

### Simulation state

`SimulationState` holds physiological values and monitor measurements.

| Group | Selected fields | Meaning |
|-------|-----------------|---------|
| Physiology | `map`, `hr`, `co`, `sao2`, `pa_co2`, `pao2` | Values that the models and the cardiac arrest check use |
| Blood chemistry | `ph`, `hco3` | Arterial plasma pH and bicarbonate at 37 °C |
| Measurements | `art_map`, `nibp_map`, `display_spo2`, `display_bis` | Values after measurement and processing effects |
| Ventilation | `paw_peak`, `paw_plat`, `peep`, `et_o2` | Breath pressures and exhaled gas measurements |
| Fluids | `blood_volume`, `urine_out_ml`, `net_fluid_ml`, `lap`, `lung_water` | Circulating volume, fluid balance, filling pressure, and excess lung water |

Monitors take the physiological values and add sensor dynamics, sampling, and
averaging. Pressure displays and scenario objectives use whichever blood
pressure monitor is configured, arterial or noninvasive (NIBP). CSV recordings
include both the physiological values and the measurements. `state.py` defines
every field and its units.

### Initialization

`awake` starts from the patient's baseline values. `steady_state` starts a
maintenance anesthetic from a 30-minute drug history, begins controlled
ventilation, and lets the models settle. Vasopressor infusions start at zero, so
mean arterial pressure (MAP) shows the untreated effect of the anesthetic. The
session clock, recordings, and fluid totals start after initialization.

The initial ventilator settings aim for the patient's resting PaCO₂. Tidal
volume is adjusted first, within limits based on predicted body weight, and then
respiratory rate. Patients with COPD have a lower rate limit to allow time for
expiration. With severe obstruction or obesity, PaCO₂ can remain above the
target.

## Supported patient domain

AnaSim supports adults aged 18 to 70 years within the [input
limits](CLI_USAGE.md#fields) for body size, hemoglobin, organ function, and
patient conditions.

Coronary disease reduces coronary flow reserve. Myocardial oxygen demand is
based on the rate-pressure product, and supply depends on diastolic perfusion
pressure, diastolic time, and arterial oxygen content. A sustained mismatch
between supply and demand causes reversible contractile depression and ST
depression.

Heart failure changes resting cardiac output, filling pressure, and sensitivity
to loading conditions (Schwartzenberg 2012; Westermann 2008). With reduced
ejection fraction, stroke volume is more sensitive to afterload. With preserved
ejection fraction, the ventricle is stiffer and fills less during tachycardia.

COPD combines persistent obstruction, hyperinflation, gas trapping, dead space,
and impaired gas exchange, with a normal resting PaCO₂. Acute bronchospasm adds
to these changes. Obesity reduces supine functional residual capacity (FRC) and
respiratory compliance and increases airway resistance (Pelosi 1998).

Pregnancy changes maternal blood volume, cardiac output, renal filtration, FRC,
heart rate, metabolism, and resting PaCO₂ according to gestational age (Dallmann
2017; Contreras 1991). In later pregnancy, the supine position reduces venous
return and the left lateral position restores it (Humphries 2019). Drug
responses use the same calibrations as in other adults.

With `patient_variability`, each session samples propofol kinetics and
bispectral index (BIS) sensitivity from Eleveld 2018, using standard deviations
halved to match VitalDB inductions with total intravenous anesthesia (TIVA). It
also samples blood pressure sensitivity to propofol and opioids from Su 2023.
The session seed fixes the sampled patient. Without this option, each patient
has the typical response.

## Model notes

AnaSim links published models, each developed in its own study population, and
uses calibrations to join them and to represent patient conditions. The code
documents the source equations and coefficients.

### PK and TCI

Intravenous drugs have a central compartment, up to two peripheral compartments,
and effect sites for their clinical effects. Propofol uses the Eleveld 2018
arterial PK model with its opioid-regimen covariate, and remifentanil uses
Eleveld 2017. Morphine and hydromorphone use Meissner 2025. Inhaled agents move
between lung, blood, and tissue compartments. Their effects depend on brain
partial pressure, expressed as age-adjusted minimum alveolar concentration (MAC)
(Davis and Mapleson 1981; Yasuda 1991).

When blood volume changes, the vascular part of each drug's central volume is
rescaled and the amount of drug is conserved. Clearance depends on organ
function, renal perfusion, and cardiac output as specified by each model. Renal
perfusion affects only the renal clearance that remains after any renal
impairment. Blood loss reduces propofol and remifentanil clearance, and volume
replacement restores it, based on swine data (Johnson 2001, 2003, 2004).
Midazolam and propofol reduce each other's clearance within the concentration
ranges studied (Vuyk 2009; Lichtenbelt 2010).

TCI predicts concentrations from the current PK parameters and adjusts infusion
rates within pump limits. Rocuronium TCI targets the free plasma concentration,
and its prediction accounts for sugammadex and inhaled agents. Setting a manual
infusion rate turns off TCI for that drug. `tci_enabled` chooses between TCI and
manual rates when maintenance is initialized.

### Anesthetic drug effects

Separate effect sites determine BIS, clinical responsiveness, and breathing.

| Endpoint | Model |
|----------|-------|
| Baseline BIS | Eleveld 2018 propofol response with age-dependent potency and processing delay, plus a calibrated sevoflurane contribution |
| Clinical responsiveness | Bouillon 2004 response curves for shaking or shouting and laryngoscopy, with Schnider 1999's age effect and a calibrated scaling of propofol for loss of response |
| Ventilatory depression | Bouillon 2004 propofol response, with opioid effects and CO₂ control from Olofsen 2010 |

`loc` is the probability of no response to shaking or shouting, and `tol` is the
probability of no response to laryngoscopy. Sevoflurane thresholds come from
Kuizenga 2019 and Hannivoort 2016. The propofol concentration for loss of
response is scaled so that patients become unresponsive at the BIS values seen
in stepped inductions, which are higher in older patients (Lysakowski 2009).

For the clinical response curves, fentanyl, morphine, and hydromorphone are
converted to remifentanil equivalents, and etomidate and midazolam to propofol
equivalents. Potency depends on the patient and on the effect. Midazolam and
propofol are synergistic. Nitrous oxide and ketamine also affect responsiveness,
and ketamine and lidocaine reduce responses to stimulation.

### Stimulation

BIS arousal during stimulation depends on laryngoscopy tolerance. The heart rate
(HR) and MAP responses use separate, stronger stimulus intensities (Albertin
2005). Laryngoscopy can also trigger laryngospasm.

Propofol lowers baseline pressure but does not reduce the cardiovascular
response to stimulation, whereas opioids do reduce it (Kazama 1997). This
interaction uses the propofol concentration at the BIS effect site because it
was calibrated at steady concentrations. In the first minutes after induction,
the faster response site reads about twice as high. Fentanyl's effects on the
pressure and HR responses are fitted separately to Kazama's equilibrated
concentrations and intubation responses. With inhaled agents, fentanyl also
lowers the concentration needed to block the response, in line with its effect
on sevoflurane MAC-BAR (Katoh 1999). Other opioids are converted to remifentanil
equivalents.

Surgery also increases vascular resistance by 15%, and anesthetics do not reduce
this increase. The size comes from the rise in MAP between the period before
incision and surgery in VitalDB TIVA cases. The increase develops with a
5-minute time constant, which gives a MAP rise of about 15% within 5 minutes of
incision at Katoh's MAC-BAR concentrations. Septic shock removes the increase.
Nociception resets the slow MAP feedback in the same way as the baroreflex, so
pressor responses persist while stimulation continues.

In the browser, maintenance sessions and intraoperative guided crises begin
during surgery, and emergence begins after surgery.

### Neuromuscular block

The [neuromuscular model](../anasim/patient/pd/nmba.py) uses Kleijn 2011 for
rocuronium and sugammadex. Sugammadex binds free plasma rocuronium and speeds
its removal from the effect site, and the complex is cleared at the rate of
sugammadex. Both drugs use measured creatinine clearance when it is supplied and
otherwise an estimate from age and renal function. Other renal effects use the
renal function factor.

Train-of-four (TOF) represents the adductor pollicis. The diaphragm and larynx
equilibrate faster and need higher rocuronium concentrations, so breathing
recovers before TOF. Residual block reduces pharyngeal reserve and makes the
airway more likely to collapse under hypnotics (Herbstreit 2009). Awake muscle
tone and positive airway pressure keep the airway open.

Succinylcholine uses Roy 2002 PK. Twitch onset and recovery are calibrated to
Kopman 2003, and central muscle sensitivity to Dhonneur 1999. Its phase I block
lowers single-twitch height without TOF fade.

### Hemodynamics

The [hemodynamic model](../anasim/physiology/hemodynamics.py) uses Su 2023's
model of the effects of propofol and remifentanil on vascular resistance, stroke
volume, and HR. It is linked to blood volume, the pulmonary circulation,
reflexes, vasoactive drugs, and distributive shock. Drug effects act through
hemodynamic effect sites on a slowly adapting tone. This tone uses Su's turnover
rate and also responds to MAP feedback, chemoreflexes, blood loss, and
temperature. As a result, pressure falls within minutes of induction, whereas
compensation is slow.

**Propofol.** The effect of propofol on pressure equilibrates over about 10
minutes, consistent with Jeleazcov 2015's volunteers and Kazama 1999's older age
groups. Its effect on vascular resistance is smaller than Su's estimate, as
Fregolent 2025 also found. Early after induction, propofol alone therefore
lowers pressure about half as much as propofol with an opioid (Ebert 1992 and
1995; Billard 1994). Its effect on stroke volume increases less with age than in
Su's model, consistent with Kazama's finding of similar sensitivity across ages.
Propofol's effects on resistance and venous pooling increase with pre-induction
MAP, so patients with higher starting pressures have larger relative falls
(Morley 2010).

Propofol also reduces stressed volume and the resistance to venous return (de
Wit 2016), using the same effect site as its arterial effects. In addition, the
loss of sympathetic venous tone moves up to 30% of stressed volume into
unstressed volume. This shift follows the plasma propofol concentration over a
few minutes. It is the reason that induction lowers filling pressure and cardiac
output more than resistance (Fairfield 1991; Zucker 2022; Sjøen 2024). Blood
loss draws on this pooled blood first, fluid loading does not refill it, and the
alpha-adrenergic effect of epinephrine reverses it.

**Opioids.** Opioids equilibrate within about a minute and are expressed as
remifentanil equivalents. Fentanyl has its own effect site. Opioid effects level
off above 2.5 ng/mL, because Sjøen 2024 found the same falls in pressure and HR
at remifentanil doses that gave peak effect-site concentrations of 2, 4, and 8
ng/mL. The direct vasodilating effect of remifentanil is a quarter of Su's
volunteer estimate. This value lies between Su and Garraza-Obaldia 2024's
patient model, which found no direct effect. The effect of remifentanil on HR is
half of Su's estimate, because Su's volunteers breathed spontaneously and were
probably hypercapnic.

The interactions between propofol and opioids on resistance, HR, and stroke
volume were fitted to VitalDB propofol-remifentanil inductions, which represent
routine practice. The [validation
guide](../validation/vitaldb/README.md#hemodynamics) gives the recorded and
modeled falls in MAP. Induction trials that used higher doses (Sawano 2013;
Sjøen 2023 and 2024; Poterman 2017) reported larger falls, and the model is
within two standard deviations of their results. The stroke volume interaction
rises over a wider range of propofol concentrations than in Su's model, which
makes it larger at the doses used in these trials.

**Sevoflurane.** Sevoflurane lowers vascular resistance and HR. Both effects
were fitted to routine bolus inductions in VitalDB, in which the falls did not
depend on the opioid dose. Ebert 1995's volunteers had larger falls in pressure
and no change in HR, and the model is within two standard deviations of their
pressures. The vascular effect equilibrates with a 14-minute half-time, which
matches the gradual fall in MAP over 10 minutes of sevoflurane by mask after
propofol in the same study. The effects of propofol and sevoflurane on
resistance and stroke volume combine as independent fractions.

**Reflexes.** MAP feedback changes HR or vascular resistance, depending on
rhythm, anesthetic depth, blood gases, and stimulation. Anesthesia reduces the
baroreflex and the tachycardia of hemorrhage (Ebert 1992; Sellgren 1994). Older
patients have lower cardiovagal gain and less reflex and beta-agonist
tachycardia, while their vascular responses are preserved (Kardos 2001; Laitinen
2004; Monahan 2007). Severe hypoxia lowers HR and contractility. Blood loss
lowers filling pressure and preload, and reflex venoconstriction partly
compensates.

Under propofol, MAP has little effect on HR (Garraza-Obaldia 2024). Slow
compensatory tachycardia therefore decreases with anesthetic depth, while
slowing in response to high pressure remains (calibrated to Sawano 2013 and
Takahashi 2002). Opioids also suppress reflex tachycardia but not reflex
slowing, so HR falls along with pressure after propofol-remifentanil induction
(Sjøen 2024). Remifentanil also causes a vagal bradycardia that does not occur
with fentanyl and that antimuscarinics block (Poterman 2017). Its size matches
the fall in HR after induction in VitalDB TIVA cases. Septic tachycardia depends
directly on sepsis severity and persists under anesthesia.

**Cardiac arrest.** With `end_on_cardiac_arrest=True`, the session ends when MAP
stays below 20 mmHg or HR stays below 10 bpm for 15 seconds.

### Cardiovascular drugs

Vasoactive drugs change vascular resistance, stroke volume, and HR. Adrenergic
and muscarinic antagonists change both these responses and the reflexes.
Anaphylaxis reduces the response to adrenergic drugs.

The norepinephrine model includes endogenous secretion and clearance that
depends on propofol (Li 2024). Its effects on resistance and stroke volume
increase with anesthetic depth. Epinephrine uses arterial clearance. Its cardiac
and vascular responses are calibrated to boluses in awake and anesthetized
patients and to infusions in awake volunteers (Ensinger 1992; Guinard 1995;
Takahashi 2002; Freyschuss 1986). Phenylephrine recruits venous blood and raises
afterload, so its effect on cardiac output depends on preload (Rebet 2016).

Ephedrine uses the oral disposition and tolerance reported by Persky 2004,
applied to intravenous use. Its peak responses are calibrated to Meng 2011, and
its effect on pressure during induction was checked against Sjøen 2026. With
repeated doses, tachyphylaxis reduces the pressor and peak HR responses, while a
smaller HR response remains.

The HR response to atropine follows its peripheral concentration (Hinderling
1985) and is reduced by propofol (Horiguchi 2002). Ketamine increases HR and
cardiac output through sympathetic stimulation and depresses the baroreflex at
anesthetic concentrations (Tweed 1972; Akine 2001). Its effect on HR starts
sooner than its effect on cardiac output and adds a fixed increase, including
after atropine (Bovill 1972).

### Fluids

Crystalloid represents lactated Ringer's solution and moves between blood and
the interstitium (Hahn 2017). Distribution slows when MAP is low, so fluid given
during hypotension after induction stays in the circulation longer (Hahn 2021).
Urine output depends on MAP, age, and renal function. More of an albumin
infusion stays in the circulation at first. Fluid also leaves the circulation
through capillary leak and other slow exchanges into a separate compartment,
which slowly returns fluid when blood volume is low.

Transfused blood tracks donor red-cell volume and hemoglobin separately. A
raised left atrial pressure increases lung water, shunt, and lung stiffness,
which links fluid balance to respiratory mechanics and gas exchange.

### Respiratory control

[Ventilatory control](../anasim/physiology/respiration.py) combines anesthetic
effects, CO₂ feedback, hypoxic drive, and muscle strength. Propofol and
sevoflurane mainly reduce tidal volume, and opioids mainly reduce respiratory
rate. Rising CO₂ can restore breathing during opioid administration. Hypocapnia
and neuromuscular block suppress respiratory effort. Under anesthesia, the
apneic threshold moves with the resting PaCO₂ that the drugs produce (Hickey
1971).

For morphine and hydromorphone, the effect on respiratory rate is modeled
separately from the effect on ventilatory drive (Olofsen 2026). The controller
combines Olofsen's fractional opioid response with Bouillon's nonlinear response
to alveolar CO₂. Hypoxic drive declines during sustained hypoxia and is reduced
by propofol and sevoflurane (Nieuwenhuijs 2000; Sarton 1996). During airway
obstruction, airflow is limited while respiratory effort continues.

### Gas exchange and blood chemistry

Ventilation exchanges gas with lung stores. These stores affect the time course
of preoxygenation, apnea, and blood loss, and they determine oxygen uptake from
the circuit. Shunt mixes blood by oxygen content, which includes oxygen bound to
hemoglobin and dissolved oxygen.

Some pulmonary blood also passes through poorly ventilated (low V/Q) units.
These units lower PaO₂ most at low FiO₂ and little at high FiO₂. Their share of
cardiac output rises with age, which produces the sea-level A-a gradient on room
air (Crapo 1999). In unconscious patients, dependent airways also close at
end-expiration, and this closure increases after age 30. Positive end-expiratory
pressure (PEEP) of about 6 cmH₂O, or more with obesity, holds these airways open
(Gunnarsson 1991; Hedenstierna 2015). Older patients therefore oxygenate worse
under anesthesia, especially without PEEP. COPD and bronchospasm add low V/Q
units with little shunt (Wagner 1977), so added oxygen corrects most of their
hypoxemia.

[Gas transport](../anasim/physiology/gas_transport.py) moves oxygen, CO₂, and
buffer base through pulmonary capillary, arterial, and mixed venous blood, with
shunt bypassing the lungs. Two tissue groups and an extracellular pool store
CO₂, with calibrated capacities (Magor-Elliott 2021; Wooten 2003). Blood flow
determines exchange with these stores, and respiratory feedback responds to
arterial PCO₂. During cardiac arrest, ventilation clears CO₂ from the lungs
while the contents of arterial blood stay the same.

When the airway is closed, absorption of gas shrinks the lung gas volume and
leads to absorption atelectasis. The model tracks this absorbed volume
separately from the lung volume used for mechanics and inhaled agents. Opening
the airway refills the lungs with gas from the circuit (Laviola 2020).

The blood gas chemistry describes respiratory acid-base changes at 37 °C. [Blood
buffering](../anasim/physiology/blood_chemistry.py) is based on Rees 2010, and
oxygen affinity on the temperature, pH, and PCO₂ terms of Dash 2016 at a
reference 2,3-DPG level. Gas exchange and changes in affinity conserve CO₂,
oxygen, and buffer base. Changes in hemoglobin or hematocrit adjust the normal
buffer contribution without changing the respiratory buffer-base changes.

### Respiratory mechanics and aeration

The [mechanics model](../anasim/physiology/resp_mech.py) combines airway
resistance, compliance, tissue viscoelasticity, and inspiratory muscle pressure.
Baseline mechanics depend on body size and patient conditions. Ventilator
assistance reduces muscle effort, and under anesthesia lung inflation ends
effort earlier. Airway and muscle pressures determine the intrathoracic pressure
that affects venous return. Bronchospasm increases expiratory resistance as lung
volume falls and can limit expiratory flow, which causes gas trapping and
intrinsic PEEP.

[Lung aeration](../anasim/physiology/lung.py) depends on the pressure history.
Recruitment and collapse change gas volume, shunt, and compliance. Inert gas in
the alveoli slows repeated collapse. Atelectatic shunt changes separately from
bulk FRC and compliance (Rothen 1995). Preoxygenation and apnea at induction
leave a shunt of about 5%, and maintenance sessions start with this shunt. PEEP
of 5 cmH₂O stops further collapse but does not reopen the lung, and atelectasis
persists after emergence (Benoit 2002).

### Temperature

Awake thermoregulation maintains the baseline core temperature. Anesthesia
reduces active regulation, increases heat loss, and redistributes heat from the
core to the periphery (Matsukawa 1995). Temperature changes metabolic demand,
and warming adds heat. Opioids and neuromuscular block reduce shivering.

### Anesthesia machine and ventilation

The [circle system](../anasim/machine/circuit.py) mixes fresh gas with circuit
gas. It accounts for patient uptake, excess gas spilling from the circuit, and
room air drawn in when uptake exceeds fresh gas flow. Vaporizer output depends
on flow, the dial setting, and the remaining agent. Oxygen supply failure stops
delivery of oxygen and nitrous oxide.

The [ventilator](../anasim/machine/ventilator.py) uses the modes and ranges of
the GE Aisys CS2. The README defines the [mode names](../README.md#features).

| Mode | Behavior |
|------|----------|
| VCV | Delivers set volume within the pressure limit |
| PCV | Maintains inspiratory pressure for the set inspiratory time |
| PCV-VG | Adjusts inspiratory pressure toward the target volume within the pressure limit |
| SIMV | Synchronizes mandatory breaths with patient efforts while preserving the set rate |
| PSV | Cycles support according to inspiratory flow, with apnea backup ventilation |
| CPAP | Maintains airway pressure during spontaneous breathing |

Mandatory breaths include gas already inspired during the patient's own effort.
Changes to inspiratory settings take effect at the next breath, and PEEP applies
during expiration. A mask leak reduces delivered gas, and tube obstruction
raises resistance. During mask ventilation, the mean airway pressure over recent
breaths holds the pharynx open.

## Monitors

ECG, arterial pressure, and plethysmography use the same beat timing. The
arterial waveform depends on MAP, stroke volume, and age-dependent arterial
compliance (Mahdi 2017; Franklin 1997). Compliance increases as pressure falls
(Langewouters 1984), so pulse pressure narrows during hypotension and widens
with vasopressors. Under propofol, systolic pressure falls about twice as much
as diastolic pressure (three times as much in Ebert 1992 and Jeleazcov 2015).
Catheter and transducer dynamics filter the arterial pressure measurements. HR
is averaged over beat intervals. Pulse oximetry responds to rhythm and
perfusion, and poor perfusion delays readings and reduces the plethysmograph
amplitude.

Spirometry measures ventilator, bag, and unassisted breaths. Expired tidal
volume comes from completed breaths, and respiratory rate and minute ventilation
are averaged over recent breaths. Pressure-volume and flow-volume loops use
synchronized pressure, flow, and volume samples. Plateau pressure requires
negligible flow and relaxed respiratory muscles. Displayed PEEP is the airway
pressure, while intrinsic PEEP affects plateau pressure and residual flow.

[Capnography](../anasim/monitors/capno.py) includes dead-space transit time and
analyzer delay. End-tidal values clear when valid exhaled gas stops. Without an
airway device, the impedance respiratory rate counts breathing efforts,
including during obstruction. The gas monitor displays end-tidal age-adjusted
MAC, while anesthetic effects use brain MAC.

BIS includes processing delay and smoothing. The [neuromuscular
monitor](../anasim/monitors/neuromuscular.py) measures TOF or single twitch on
demand or at a set interval. TOF reports the twitch count and, when all four
twitches are present, the T4/T1 ratio. Ratios from weak twitches are flagged.
Single twitch reports T1 relative to an unblocked baseline.

## Browser app

The local and hosted apps share `web_assets/` and
[`WebSession`](../anasim/web.py), which applies commands and returns
measurements, waveforms, controls, and scenario objectives. Commands are
processed in order, and the simulation speed setting converts elapsed real time
into simulation time.

The local Python server runs the session on its own clock and saves CSV
recordings to disk. It accepts requests from the same origin under a
session-specific path on 127.0.0.1.

The hosted app runs Python, NumPy, and SciPy in a Pyodide worker, and stopping a
recording downloads the CSV. [`build_web.py`](../scripts/build_web.py) builds
the hosted bundle, and [pages.yml](../.github/workflows/pages.yml) publishes it.

## Validation

Tests cover published equations, conservation of gas and drugs, monitor
behavior, and complete clinical scenarios, in native Python and in Pyodide.
Acceptance tests check standard teaching cases (balanced and TIVA induction,
intubation, incision, and vasopressor boluses) against ranges from VitalDB
recordings and published studies. [Recorded-case
comparisons](../validation/vitaldb/README.md) compare anesthetic, respiratory,
and hemodynamic components with VitalDB data.

The [intubation replay](../validation/intubation.py) (`python -m
validation.intubation`) prints absolute and incremental errors in systolic
pressure, HR, MAP, and cardiac output against Billard 1994 and Sawano 2013.
These two studies set the HR recovery and the split of the intubation response
between stroke volume and vascular resistance. The replay also compares
induction changes with Sjøen 2023 and 2024 and Poterman 2017. Ebert 1992 and
1995 set the peak HR and MAP response, and Hayakawa-Fujii 2001 is an independent
check. `--sensitivity` compares assumed propofol loading histories at Sawano's
plasma target.
