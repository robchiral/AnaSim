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
| [`physiology/`](../anasim/physiology/) | Models circulation, resuscitation, fluids, respiration, gas exchange, and temperature |
| [`machine/`](../anasim/machine/) | Models the breathing circuit, vaporizer, and ventilator |
| [`monitors/`](../anasim/monitors/) and [`core/monitors.py`](../anasim/core/monitors.py) | Produce measurements, waveforms, and alarms |
| [`scenarios/`](../anasim/scenarios/) | Defines cases, with their events, guided steps, and debrief outcomes |
| [`web.py`](../anasim/web.py) and [`web_assets/`](../anasim/web_assets/) | Connect the engine and cases to the browser interface |

## Runtime

### Step order

Each call to `SimulationEngine.step()` does the following:

1. Calculate anesthetic depth, metabolic rate, and stimulation tolerance.
2. Apply stimulation, hemorrhage, fluids, and other clinical events.
3. Update pharmacokinetic (PK) parameters for blood volume, cardiac output, and
   drug interactions, then calculate target-controlled infusion (TCI) rates.
4. Update the airway, vaporizer, and breathing circuit.
5. Update drug concentrations, inhaled anesthetic uptake, neuromuscular block,
   and lidocaine toxicity.
6. Advance breathing mechanics, gas exchange, hemodynamics, and cardiac rhythm.
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
| Physiology | `map`, `hr`, `co`, `intrinsic_co`, `sao2`, `pa_co2`, `pao2` | Values that the models and the cardiac arrest check use; `intrinsic_co` excludes chest compressions |
| Blood chemistry | `ph`, `hco3`, `base_excess`, `lactate`, `sodium`, `potassium`, `chloride`, `ionized_calcium`, `albumin_g_l` | Arterial plasma pH and bicarbonate at 37 °C, standard base excess, and plasma lactate, electrolytes, and albumin |
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

AnaSim supports adults aged 18 to 85 years within the [input
limits](CLI_USAGE.md#fields) for body size, hemoglobin, organ function, and
patient conditions.

Coronary disease reduces coronary flow reserve. Heart failure changes resting
cardiac output, filling pressure, and sensitivity to loading conditions
(Schwartzenberg 2012; Westermann 2008). With reduced ejection fraction, stroke
volume is more sensitive to afterload. With preserved ejection fraction, the
ventricle is stiffer and fills less during tachycardia.

COPD combines persistent obstruction, hyperinflation, gas trapping, dead space,
and impaired gas exchange, with a normal resting PaCO₂. Acute bronchospasm adds
to these changes. Obesity reduces supine functional residual capacity (FRC) and
respiratory compliance and increases airway resistance (Pelosi 1998).

Pregnancy changes maternal blood volume, extracellular water, cardiac output,
renal filtration, FRC, heart rate, metabolism, and resting PaCO₂ according to
gestational age (Dallmann 2017; Contreras 1991; Ansari 2025). In later
pregnancy, the supine position reduces venous return and the left lateral
position restores it (Humphries 2019). Drug responses use the same calibrations
as in other adults.

With `patient_variability`, each session samples propofol kinetics and
bispectral index (BIS) sensitivity from Eleveld 2018, using standard deviations
halved to match VitalDB inductions with total intravenous anesthesia (TIVA). It
also samples remifentanil kinetics (Eleveld 2017), fentanyl kinetics (Bae 2020),
sevoflurane potency for awakening and noxious stimuli (Heyse 2012), the effects
of propofol and opioids on vascular resistance (Su 2023), and the timing,
sensitivity, and shape of the respiratory-rate response to morphine and
hydromorphone (Olofsen 2026). Sevoflurane's effect on vascular resistance varies
as much as propofol's. The session seed fixes the sampled patient. Without this
option, each patient has the typical response.

## Model notes

AnaSim links published models, each developed in its own study population, and
uses calibrations to join them and to represent patient conditions. The code
documents the source equations and coefficients.

### PK and TCI

Intravenous drugs have a central compartment, up to two peripheral compartments,
and effect sites for their clinical effects. Propofol uses the Eleveld 2018
arterial PK model with its opioid-regimen covariate, and remifentanil uses
Eleveld 2017. Morphine and hydromorphone use Meissner 2025, and lidocaine uses
Foong 2025. Inhaled agents move between lung, blood, and tissue compartments.
Their effects depend on brain partial pressure, expressed as age-adjusted
minimum alveolar concentration (MAC) (Davis and Mapleson 1981; Yasuda 1991).

When blood volume changes, the vascular part of each drug's central volume is
rescaled. Shed blood carries drug away at plasma concentration, so blood loss
alone leaves the concentration unchanged; other volume changes conserve drug
mass. Clearance depends on organ function, renal perfusion, and cardiac output
as specified by each model. Fentanyl and lidocaine clearances were measured
during anesthesia and surgery, so they apply at the model's cardiac output under
anesthesia (Bae 2020; Foong 2025). Renal perfusion affects only the renal
clearance that remains after any renal impairment. Blood loss reduces propofol
and remifentanil clearance, and volume replacement restores it, based on swine
data (Johnson 2001, 2003, 2004). Midazolam and propofol reduce each other's
clearance within the concentration ranges studied (Vuyk 2009; Lichtenbelt 2010).
Below 30% of resting cardiac output, drug transport to organs, peripheral
compartments, and effect sites falls in proportion to flow, so a dose given
during arrest does not reach its effect sites until compressions circulate it.
Esterase metabolism of remifentanil, succinylcholine, and esmolol continues
without flow, and sugammadex still binds rocuronium in plasma.

TCI adjusts infusion rates within pump limits to reach a predicted plasma or
effect-site target. For propofol, remifentanil, and fentanyl, TCI and decrement
times use the published model with only age, weight, height, and sex, as
clinical pumps do. The prediction omits sampled variability, hepatic impairment,
blood volume, cardiac output, and midazolam, so the patient's concentrations can
differ from the target; after blood loss, propofol and remifentanil rise above
it. The pump also omits hand boluses. A TCI program started after a manual rate
assumes no drug on board, and a stopped program resumes from its own history.
TCI for other drugs predicts from the patient's current PK. Rocuronium TCI
targets the free plasma concentration, and its prediction accounts for
sugammadex and inhaled agents. Setting a manual infusion rate turns off TCI for
that drug. When maintenance is initialized, `tci_enabled` chooses between TCI
and manual rates.

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
the concentration at the faster clinical-response site is about twice as high.
Fentanyl's effects on the
pressure and HR responses are fitted separately to Kazama's equilibrated
concentrations and intubation responses. With inhaled agents, fentanyl also
lowers the concentration needed to block the response, in line with its effect
on sevoflurane MAC-BAR (Katoh 1999). Other opioids are converted to remifentanil
equivalents.

Surgery also increases vascular resistance by 15%, and anesthetics do not reduce
this increase. Its size comes from the rise in MAP from before incision to
during surgery in VitalDB TIVA cases. The increase develops with a
5-minute time constant, which gives a MAP rise of about 15% within 5 minutes of
incision at Katoh's MAC-BAR concentrations. Septic shock removes the increase.
Nociception resets the slow MAP feedback in the same way as the baroreflex, so
pressor responses persist while stimulation continues.

In the browser, maintenance sessions and intraoperative crisis cases begin
during surgery, and emergence begins after surgery.

### Learner action checks

In the browser, selecting the tracheal tube calls
`SimulationEngine.attempt_intubation`, which adds a brief intubation stimulus to
any surgical stimulation. The tube is not placed when loss of response and
laryngoscopy tolerance are both below 0.1 and central muscle strength is above
0.5.

[`clinical_actions.py`](../anasim/core/clinical_actions.py) shows a warning but
lets the action proceed when:

- the tube is placed in a responsive patient
- a neuromuscular blocker is given to a responsive patient without an induction
  dose in plasma
- the tube is removed during apnea or residual block (TOF ratio or T1 below 90%)
- assisted ventilation stops during apnea
- a shock is delivered to a responsive patient

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
minutes, consistent with Jeleazcov 2015's volunteers and Kazama 1999's patients
aged 60 to 69. Above 65, the effect develops more slowly and grows larger, as
Kazama found at 70 to 85 years, with about twice the half-time by age 78. Its
effect on vascular resistance is smaller than Su's estimate, as Fregolent 2025
also found. Early after induction, propofol alone therefore lowers pressure
about half as much as propofol with an opioid (Ebert 1992 and 1995; Billard
1994). Its effect on stroke volume increases less with age than in Su's model,
consistent with Kazama's finding of similar sensitivity from 20 to 69 years.
Propofol's effects on resistance and venous pooling increase with pre-induction
MAP, so patients with higher starting pressures have larger relative falls
(Morley 2010).

Propofol also reduces stressed volume and the resistance to venous return (de
Wit 2016), using the same effect site as its arterial effects. In addition, the
loss of sympathetic venous tone moves up to 30% of stressed volume into
unstressed volume. This shift follows the plasma propofol concentration over a
few minutes. This shift is why induction lowers filling pressure and cardiac
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

**Sevoflurane.** Sevoflurane lowers stroke volume, vascular resistance, and HR.
The fall in pressure was fitted to routine bolus inductions in VitalDB, in which
the falls did not depend on the opioid dose. Most of the fall comes from stroke
volume: in unstimulated volunteers, cardiac index fell 28% at 1 and 1.5 MAC and
resistance fell 4% and 16% (Malan 1995). Ebert 1995's volunteers had larger
falls in pressure and no change in HR, and the model is within two standard
deviations of their pressures. The effect equilibrates with a 14-minute
half-time, which matches the gradual fall in MAP over 10 minutes of sevoflurane
by mask after propofol in Ebert's study. The effects of propofol and sevoflurane
on resistance and stroke volume combine as independent fractions.

**Reflexes.** MAP feedback changes HR or vascular resistance, depending on
rhythm, anesthetic depth, blood gases, and stimulation. When muscarinic block
or a fixed rhythm prevents slowing, high MAP lowers endogenous vascular
resistance (Weissler 1957; Tweed 1972). Anesthesia reduces the
baroreflex and the tachycardia of hemorrhage (Ebert 1992; Sellgren 1994). Older
patients have lower cardiovagal gain and less reflex tachycardia, while
baroreflex vascular control is preserved (Kardos 2001; Laitinen 2004; Monahan
2007). Severe hypoxia lowers HR and contractility. Blood loss lowers filling
pressure and preload, and reflex venoconstriction partly compensates. Venous
recruitment increases for losses above 30%, so slow class III and IV hemorrhage
remains compensated until stressed volume is exhausted.

Under propofol, MAP has little effect on HR (Garraza-Obaldia 2024). Slow
compensatory tachycardia therefore decreases with anesthetic depth, while
slowing in response to high pressure remains (calibrated to Sawano 2013 and
Takahashi 2002). Opioids also suppress reflex tachycardia but not reflex
slowing, so HR falls along with pressure after propofol-remifentanil induction
(Sjøen 2024). Remifentanil also causes a vagal bradycardia that does not occur
with fentanyl and that antimuscarinics block (Poterman 2017). Its size matches
the fall in HR after induction in VitalDB TIVA cases. Septic tachycardia depends
directly on sepsis severity and persists under anesthesia.

**Acidemia.** Below pH 7.2, contractility and the response to adrenergic drugs
fall linearly to 30% at pH 6.8, extrapolated from isolated failing human
myocardium (Schotola 2012).

### Cardiovascular drugs

Vasoactive drugs change vascular resistance, stroke volume, and HR. Adrenergic
and muscarinic antagonists change both these responses and the reflexes.
Anaphylaxis reduces the response to adrenergic drugs.

The norepinephrine model includes endogenous secretion and clearance that
depends on propofol (Li 2024). Its effects on resistance and stroke volume
increase with anesthetic depth.

Epinephrine uses arterial clearance (Ensinger 1992), which falls with age
(Esler 1995). Its effective mixing volume is calibrated to bolus responses.
This volume sets the initial concentration after a bolus and, with clearance,
the rate at which plasma concentration falls. Cardiac and vascular responses
are calibrated to boluses in awake and anesthetized patients and to infusions
in awake volunteers (Guinard 1995; Takahashi 2002; Freyschuss 1986; Stratton
1985; Ensinger 1992; Marker 1998; Leenen 2005).

Older patients are less sensitive to epinephrine (Leenen 2005; Pan 1986;
Hogikyan 1994), so the arterial alpha-adrenergic C50 rises with age. During
infusions, higher plasma concentrations and weaker baroreflex buffering offset
this, while boluses raise HR less in older patients. Under propofol, epinephrine
recruits pooled venous blood at lower concentrations than those that constrict
arteries. A second alpha-adrenergic component acts at higher concentrations,
with a gradual concentration response and its own delay. It maintains vascular
resistance at the infusion rates used in shock and supports pressure during
CPR. The concentration-response calibration covers concentrations up to
Ensinger's highest infusion step, 0.22 mcg/kg/min.

Phenylephrine recruits venous blood and raises afterload, so its effect on
cardiac output depends on preload (Rebet 2016).

Ephedrine uses the oral disposition and tolerance reported by Persky 2004,
applied to intravenous use. Its peak responses are calibrated to Meng 2011, and
its effect on pressure during induction was checked against Sjøen 2026. With
repeated doses, tachyphylaxis reduces the pressor and peak HR responses, while a
smaller HR response remains.

The HR response to atropine follows its peripheral concentration (Hinderling
1985) and is reduced by propofol (Horiguchi 2002). Venous unloading lowers
stroke volume over minutes, with an earlier fall at low preload (Weissler
1957; Berry 1959). Adrenergic drive restores stroke volume during pacing
(Ross 1965).

Ketamine raises HR, stroke volume, and vascular resistance through sympathetic
stimulation (Tweed 1972). Its HR effect starts sooner and adds a fixed increase,
including after atropine (Bovill 1972). Baroreflex gain falls at anesthetic
concentrations, based on rat data (Akine 2001). A calibrated offset of up to
20 mmHg reduces reflex slowing and vasodilation to fit pressure rises after
atropine and during pacing. Beta blockade leaves a pressor response, while
alpha blockade reduces it (Bovill 1972; Bålfors 1983).

Rapid boluses of intravenous amiodarone briefly lower vascular resistance and
blood pressure, mostly through the polysorbate 80 solvent, and slower infusions
lower pressure much less (Munoz 1988). Contractility falls over several
minutes, so patients with HFrEF lose more cardiac output and blood pressure
than patients with a normal left ventricle (Kosinski 1984). Sinus rate slows
over 15 to 30 minutes (Munoz 1988), and sympathetic HR reflexes weaken over the
same period.

### Heart and brain oxygen supply

Myocardial oxygen demand is based on the rate-pressure product, with a basal
demand that remains as mechanical work falls (Harasawa 1992). Fibrillating
myocardium uses most of the resting demand of the beating heart. Supply depends
on diastolic pressure above the coronary zero-flow pressure (Nanto 2001),
diastolic time, coronary flow reserve, and arterial oxygen content. A sustained
mismatch between supply and demand causes reversible contractile depression and
ST depression.

Cerebral oxygen supply falls with MAP below 50 mmHg and with arterial oxygen
content below two-thirds of baseline. Brain oxygen stores buffer a fall in
supply for a few seconds. As tissue oxygen falls from 60% to 35% of normal
(Astrup 1981), responsiveness, airway tone, and BIS fail (Nebout 2013), and
near-complete failure stops breathing. After circulation stops, this failure
begins at about 4 seconds, and the EEG is flat at about 11 seconds (de Vries
1998). Function recovers within about two minutes after supply returns. Below
20% of normal, neurons are injured, and injury is slower when more supply
remains (Jones 1981). After the equivalent of 5 minutes without supply, the
patient remains comatose, although BIS and breathing can recover.

CO₂ narcosis reduces responsiveness and airway tone without anoxic injury. Its
severity depends on both PaCO₂ and pH (Westlake 1955). With acidemia, it rises
as PaCO₂ increases from 60 to 100 mmHg and pH falls from 7.3 to 7.1. In
compensated hypercapnia, it rises as PaCO₂ increases from 100 to 140 mmHg.
Onset and recovery have a 30-second time constant. CO₂ does not change BIS
directly; BIS was unchanged at EtCO₂ 3-12 kPa under propofol anesthesia
(Turnbull 2008).

### Cardiac arrest and resuscitation

Arrest is confirmed when VF, asystole, HR below 10 bpm, or PEA lasts 15
seconds. PEA is output from the heart's own beats below 0.3 L/min with pulse
pressure below 5 mmHg while the ECG remains organized. The simulation continues
through arrest, so the learner can resuscitate and treat the cause.

Severe myocardial ischemia or hypoxemia first depletes an energy reserve that
lasts about 4 minutes without circulation. If a perfusing rhythm returns within
this time, as after defibrillation in the electrical phase of arrest (Weisfeldt
2002), circulation is restored. Once the reserve is used up, the heart slows
over several minutes: P waves disappear
(junctional rhythm), and complexes widen and shrink (idioventricular, then
agonal rhythm) until the ECG is flat. Untreated VF becomes fine and then
asystole. Meanwhile, injury that reperfusion does not reverse within a session
builds up and reduces contractility (Laurent 2002). The session ends when this
injury is complete, about 11 minutes after circulation stops.

[Chest compressions](../anasim/physiology/resuscitation.py) at 110/min pump a
fifth of resting cardiac output at normal filling and vascular resistance, and
little after major blood loss. This flow and vascular resistance set arterial
pressure between compressions, and each compression adds a pulse that does not
drive flow. Coronary flow occurs between compressions. The energy reserve
recovers when diastolic pressure exceeds about 25 mmHg. This is the CPR target
in Meaney 2013 and corresponds to the coronary perfusion pressure needed for
return of circulation in Paradis 1990. In an anesthetized adult, compressions
give about 70/25 mmHg and EtCO₂ near 15-20 mmHg, so CPR keeps VF coarse and
makes fine VF coarse again. Epinephrine raises diastolic and coronary perfusion
pressure during compressions (Kim 2025), while the higher afterload reduces
forward flow and lowers EtCO₂ by a few mmHg at constant ventilation (Cantineau
1994). In primary asystole, beats resume after 30 seconds of adequate coronary
oxygen supply during compressions if energy remains and the myocardium is
viable. An interruption or inadequate perfusion restarts this interval.

A 200 J biphasic shock terminates coarse VF 9 times in 10, and fine VF less
often. Each shock also causes mild contractile stunning that reduces stroke
volume by about 10% and resolves over several minutes of coronary perfusion
(Tokano 1998). After brief VF, arterial pressure recovers over 5 to 20 seconds
as beats refill the arterial tree and reversible oxygen debt clears, taking
longer after longer VF (Park 1994). After prolonged arrest, the heart beats only
as well as its energy reserve and accumulated injury allow, so a late shock
without CPR leaves PEA or asystole, and resuscitation after reserve exhaustion
leaves hypotension for the rest of the session (Laurent 2002). Synchronized shocks
fire on the next R wave and usually convert AF, SVT, and VT; they cannot fire
in VF or asystole, and compressions do not trigger them. An unsynchronized
shock that falls near the T-wave peak of an organized rhythm causes VF 15% of
the time.

Each shock needs a new charge, which takes 5 seconds and expires after 60
seconds (ZOLL R Series). Compressions can continue while it charges. Disarming
cancels a charge or a pending synchronized shock.

[Severe ischemia or hypoxia](../anasim/physiology/arrhythmia.py) in a beating
heart can start VT or VF. After a successful shock, VF often recurs over the
next two minutes if myocardial ischemia or hypoxia persists (Koster 2008;
van Alem 2003). Amiodarone (Korth-Bradley 1996) and lidocaine convert sustained
VT with a pulse over tens of minutes, and amiodarone converts it more often
(Somberg 2002; Scheinman 1995). In VF, both drugs make a shock more likely to
succeed after CPR delivers them (Kudenchuk 2016), and neither prevents rearrest
(Salcido 2018).

### Lidocaine toxicity

In awake patients, [lidocaine](../anasim/physiology/toxicity.py) raises HR and
blood pressure above an effect-site concentration of about 4 mcg/mL and causes
seizures above 5 mcg/mL (Foldes 1960; Jorfeldt 1968). Between 11 and 20 mcg/mL,
CNS depression and apnea replace seizures. Contractile depression begins above
13 mcg/mL, and higher concentrations widen the QRS complex and can cause
ventricular arrhythmias or arrest. A 2 g infusion over an hour causes
hypotension and coma in sinus rhythm rather than arrest (Armstrong 1988),
although the model clears the drug faster than in that patient, whose ejection
fraction was 15%. Acidemia worsens both CNS and cardiac toxicity. Midazolam,
propofol, and sevoflurane suppress seizures and the sympathetic rise, and
neuromuscular block hides convulsions without stopping the seizure. Lipid
emulsion reduces lidocaine delivery to its effect sites by about a fifth, as it
reduced unentrapped exposure in volunteers (Heinonen 2015), and lessens
contractile depression (Fettiplace 2014).

### Fluids

Lactated Ringer's and 0.9% saline move between blood and the interstitium
(Hahn 2017). Distribution slows when MAP is low, so fluid given during
hypotension after induction stays in the circulation longer (Hahn 2021). Urine
output depends on MAP, age, and renal function. Saline is excreted more slowly
than lactated Ringer's: after 2 L over an hour, healthy volunteers pass about
half as much urine in 6 hours (Reid 2003). More of a 5% albumin infusion stays
in the circulation at first. Fluid also leaves the circulation through
capillary leak and other slow exchanges into a separate compartment, which
slowly returns fluid when blood volume is low.

[Fluid balance](../anasim/physiology/fluid_balance.py) tracks sodium,
potassium, chloride, calcium, magnesium, and phosphate in one extracellular
volume, and albumin separately in plasma and the interstitium. Water moves
between extracellular fluid and cells until their tonicity is equal, so the
free water in lactated Ringer's and lipid emulsion spreads through total body
water. Each fluid adds its own ions and albumin, and buffer base equal to its
strong ion difference. Saline therefore causes a hyperchloremic acidosis
and lactated Ringer's does not (Scheingraber 1999). Packed red cells have the
composition of a 3-week-old unit, at pH 6.5 and PCO₂ 98 mmHg with supernatant
potassium 38 and lactate 22.5 mmol/L (Rabcuka 2025), so rapid transfusion adds
acid, CO₂, and potassium. Lactate from Ringer's and packed cells, and citrate
from packed cells, return base as they are metabolized (Kramer 2003).

Potassium exchanges with a large cellular pool over minutes, and epinephrine
moves it into cells. Ionized calcium depends on albumin, citrate, and pH
(Thode 1983; Bolan 2001), so dilution and rapid transfusion lower it. Injected
calcium mixes through extracellular fluid, and calcium above the patient's
resting ionized level is then cleared. One gram of calcium chloride therefore
raises ionized calcium by about 0.4 mmol/L, and half of the rise is gone by
30 minutes (Ansari 2025). Calcium lost to dilution or bound by citrate is not
replaced within a session (Bolan 2001).

The kidneys excrete the daily fixed acid load when urine output and renal
perfusion allow. Over about a day, they also move bicarbonate toward the chronic
compensation for the current PaCO₂: 0.48 mmol/L per mmHg in hypercapnia up to
70 mmHg (González 2018) and 0.4 mmol/L per mmHg in hypocapnia (Krapf 1991).
Chloride moves in the opposite direction to bicarbonate.

Anaphylaxis causes rapid plasma leak. Red cells stay in the circulation, so
hemoglobin concentration rises (Fisher 1986). The leak slows as the reaction
subsides after the trigger is removed, and an ongoing reaction can require
repeated crystalloid boluses.

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

[Respiratory muscles fatigue](../anasim/physiology/fatigue.py) when mean
inspiratory pressure, including effort against an obstructed airway, exceeds
about 15% of maximum. Endurance falls steeply as the load rises further
(Bellemare and Grassino 1982). Fatigue reduces spontaneous tidal volume and
develops sooner when oxygen supply is inadequate. Pressure support and
mandatory breaths reduce inspiratory effort (Leung 1997). Most fatigue recovers
within minutes of rest (Aubier 1981; Rohrbach 2003), and a smaller deficit
persists for up to 24 hours after prolonged heavy loading (Laghi 1995;
Travaline 1997).

Metabolic acidosis lowers the PaCO₂ set point by 1.2 mmHg per mmol/L of base
deficit, with a two-minute time constant. Anesthetics and opioids still limit
the ventilatory response.

### Gas exchange and blood chemistry

Ventilation exchanges gas with lung stores. These stores affect the time course
of preoxygenation, apnea, and blood loss, and they determine oxygen uptake from
the circuit. Shunt mixes blood by oxygen content, which includes oxygen bound to
hemoglobin and dissolved oxygen.

Some pulmonary blood also passes through poorly ventilated units with a
ventilation-perfusion ratio (V/Q) of 0.05. These units lower PaO₂ most at low
FiO₂ and little at high FiO₂. Their share of cardiac output rises with age,
which produces the sea-level A-a gradient on room air (Crapo 1999). In
unconscious patients, dependent airways also close at end-expiration, and this
closure increases after age 30. Positive end-expiratory pressure (PEEP) of about
6 cmH₂O, or more with obesity, holds these airways open (Gunnarsson 1991;
Hedenstierna 2015). Older patients therefore oxygenate worse under anesthesia,
especially without PEEP. After upper abdominal surgery, FRC stays at or below
its value under anesthesia (Wahba 1975; Putensen-Himmer 1992), so this closure
persists after emergence. After lower abdominal surgery, about a third of it
persists (Lindberg 1992). Bronchospasm adds more of these units, and COPD adds a
separate group with a V/Q of 0.1 (Wagner 1977). Neither condition adds much
shunt, so added oxygen corrects most of the hypoxemia.

[Gas transport](../anasim/physiology/gas_transport.py) moves oxygen, CO₂, and
buffer base through pulmonary capillary, arterial, and mixed venous blood, with
shunt bypassing the lungs. Two tissue groups and an extracellular pool store
CO₂, with calibrated capacities (Magor-Elliott 2021; Wooten 2003). Blood flow
determines exchange with these stores, and respiratory feedback responds to
arterial PCO₂. Lung tissue accounts for 5% of oxygen consumption and CO₂
production and exchanges directly with alveolar gas (Loer 1997). Without
pulmonary blood flow, ventilation therefore lowers alveolar PCO₂ only to about
2 mmHg.

When the airway is closed, absorption of gas shrinks the lung gas volume and
leads to absorption atelectasis. The model tracks this absorbed volume
separately from the lung volume used for mechanics and inhaled agents. Opening
the airway refills the lungs with gas from the circuit (Laviola 2020).

Oxygen consumption equals metabolic demand until oxygen delivery falls to a
critical level. At this level, the oxygen extraction ratio is 0.6 (Ronco 1993).
The critical delivery is therefore proportional to demand, so it is lower
during hypothermia and anesthesia. Below this level, consumption and aerobic
CO₂ production fall in proportion to delivery, and the unmet demand produces
[lactate](../anasim/physiology/lactate.py). The amount of lactate produced per
unit of oxygen deficit was fitted to data on oxygen debt in dogs with
hemorrhagic shock (Dunham 1991).

Epinephrine and sepsis increase aerobic lactate production (Grip 2015;
Revelly 2005). Lactate has a volume of distribution of 0.26 L/kg
(Levraut 1998). Its elimination follows Michaelis-Menten kinetics, so
clearance falls as the concentration rises. The model assumes that the liver
provides 60% of the elimination capacity. This hepatic portion is reduced in
proportion to liver function and to the fraction of oxygen demand that is met.

Each mmol of endogenous lactate that accumulates above the resting amount
removes 1 mmol of buffer base, and clearance of this lactate returns the base.
Lactated Ringer's solution raises the measured lactate without adding acid,
and metabolism of this lactate adds buffer base (Zitek 2018). Blood gas
analyzers measure only the L-isomer, half of the racemic lactate in Ringer's.

The blood gas model includes respiratory, lactic, and fluid-related acid-base
disturbances at 37 °C. [Blood buffering](../anasim/physiology/blood_chemistry.py)
is based on Rees 2010, with plasma albumin and phosphate charge from
Figge 2018. Oxygen affinity uses the temperature, pH, and PCO₂ terms of
Dash 2016 at a reference 2,3-DPG level. Gas exchange and changes in oxygen
affinity conserve CO₂, oxygen, and buffer base. Bleeding, transfusion, fluids,
and urine change the total buffer base by the amounts that they carry.

The interstitium buffers with its albumin and phosphate and with a fixed
cellular buffer, the extracellular buffer of Wooten 2003. With these buffers,
acute changes in PaCO₂ move pH and bicarbonate within the human ranges of
Brackett 1965 and Arbus 1969, and standard base excess changes little. Standard
base excess is calculated from arterial pH and bicarbonate with the Van Slyke
equation, as blood gas analyzers do. It is near 0 mmol/L in nonpregnant adults
and about −2 mmol/L in late pregnancy.

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
warming adds heat, and surface cooling removes it. Opioids and neuromuscular
block reduce shivering.

### Malignant hyperthermia

[Malignant hyperthermia](../anasim/physiology/malignant_hyperthermia.py) can
start during sevoflurane anesthesia or after succinylcholine. Onset after a
volatile anesthetic varies and often comes in the second or third hour (Visoiu
2014), so the instructor or case sets the time. Skeletal muscle hypermetabolism
continues after the trigger is removed. Untreated, oxygen consumption and
CO₂ production approach five times baseline, and the gas exchange, lactate,
and temperature models produce hypercapnia, lactic acidosis, and a rising
temperature. Tachycardia and rigidity increase with the crisis, and rigidity
persists under neuromuscular block. Dantrolene PK follows the Ryanodex label.
Dantrolene controls the crisis and reduces twitch height without TOF fade, but
leaves ventilation unchanged (Flewellen 1983).

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
during expiration. Manual bag breaths stop at the adjustable pressure limit
(APL, default 20 cmH₂O), so obstruction or bronchospasm lowers the delivered
volume. During mask ventilation, the mean airway pressure over recent breaths
holds the pharynx open. A loose mask loses part of each assisted breath and
lowers this pressure, and pressures above 20 cmH₂O leak even with a tight seal
(Bouvet 2014). A full seal delivers the tidal volume of two-handed mask
ventilation, and an 80% seal that of a one-handed hold (Joffe 2010). Room air
enters around a loose mask only during unsupported breaths, since bag,
ventilator, and pressure-support breaths pressurize the mask. Tube obstruction
raises airway resistance.

## Monitors

ECG, arterial pressure, and plethysmography use the same beat timing, and HR
is averaged over beat intervals. The arterial waveform depends on MAP, stroke
volume, and age-dependent arterial compliance (Mahdi 2017; Franklin 1997).
Compliance increases as pressure falls (Langewouters 1984), so pulse pressure
narrows during hypotension and widens with vasopressors. Epinephrine's
inotropic effect widens pulse pressure more than its effect on stroke volume
alone would, without changing MAP or flow. In human inotrope studies, pulse
pressure rose with contractility even when stroke volume did not (Piccioli
2023). Under
propofol, systolic pressure falls about twice as much as diastolic pressure
(three times as much in Ebert 1992 and Jeleazcov 2015). Catheter and transducer
dynamics filter the arterial pressure measurements. Pulse oximetry responds to
rhythm and perfusion, and poor perfusion delays readings and reduces the
plethysmograph amplitude.

Without flow, arterial pressure falls to the mean systemic filling pressure.
During chest compressions, arterial pressure and plethysmography follow the
compressions, the ECG shows compression artifact, and NIBP fails.

Spirometry measures ventilator, bag, and unassisted breaths. Expired tidal
volume comes from completed breaths, and respiratory rate and minute ventilation
are averaged over recent breaths. Pressure-volume and flow-volume loops use
synchronized pressure, flow, and volume samples. Plateau pressure requires
negligible flow and relaxed respiratory muscles. Displayed PEEP is the airway
pressure, while intrinsic PEEP affects plateau pressure and residual flow.

[Capnography](../anasim/monitors/capno.py) includes dead-space transit time and
analyzer delay. End-tidal values clear when valid exhaled gas stops. Gas lost
around a facemask does not pass the sampling port, so a large leak, especially
with small breaths, leaves dead-space gas at the port and flattens the
capnogram. Without an airway device, the impedance respiratory rate counts
breathing efforts, including during obstruction. The gas monitor displays
end-tidal age-adjusted MAC, while anesthetic effects use brain MAC.

BIS includes processing delay and smoothing, which also delay its response to
cerebral ischemia. The [neuromuscular
monitor](../anasim/monitors/neuromuscular.py) measures TOF or single twitch on
demand or at a set interval. TOF reports the twitch count and, when all four
twitches are present, the T4/T1 ratio. Ratios from weak twitches are flagged.
Single twitch reports T1 relative to an unblocked baseline.

An [arterial blood gas](../anasim/monitors/blood_gas.py) sample records the
arterial values at the time of the draw, and the result appears 2 minutes
later. As in a blood gas analyzer, pH, PaCO₂, and PaO₂ are reported at 37 °C,
so PaO₂ in a hypothermic patient is higher than the value in the body. The
report also includes bicarbonate, standard base excess, lactate, sodium,
potassium, chloride, ionized calcium, SaO₂, hemoglobin, and FiO₂. Glucose is
not modeled.

## Browser app

### Cases

Each scenario file defines one case: a briefing, timed events, case actions
such as requesting hemostasis, guided steps, and debrief outcomes. `CaseRun` in
[`scenarios/runtime.py`](../anasim/scenarios/runtime.py) starts the events,
tracks outcomes, and builds the debrief in both practice modes. Case sessions
hide the instructor event controls that open simulation provides.

Crises start 30 simulated seconds into the case. Anaphylaxis and sepsis start
at full severity, while hemorrhage and MH develop over several minutes. In the
anaphylaxis case, crystalloid boluses run at 500 mL/min.

Guided requirements count actions from the start of the case, or from crisis
onset for crisis treatment, so the guide never asks for a repeat dose.
In independent mode, the browser receives no step text, and safety warnings
other than a failed intubation appear only in the debrief.

Pressure objectives require both the displayed MAP and the current model MAP
to reach the target. Without an arterial line, the cuff reading must be no
more than one minute old. Anaphylaxis recovery also requires improved
bronchospasm.

Milestone outcomes, such as preoxygenation, stay credited once achieved.
Extubation outcomes are checked when the tube is removed, and the rest must
hold when the case ends. Ending the case, finishing the last guided step, or
irreversible cardiac arrest pauses the model, stops recording, and freezes the
debrief.

### Session transport

The local and hosted apps share `web_assets/` and
[`WebSession`](../anasim/web.py), which applies commands and returns
measurements, waveforms, controls, and case state. Commands are
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
