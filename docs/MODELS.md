# AnaSim models

This guide describes how AnaSim models the patient, drugs, physiology,
anesthesia machine, and monitors, and which studies each model is based on.
The [architecture guide](ARCHITECTURE.md) covers the code structure and step
order, and [Model references](REFERENCES.md) lists the studies.

AnaSim links published models, each developed in its own study population, and
uses calibrations to join them and to represent patient conditions. The code
documents the source equations and coefficients.

## Supported patient domain

AnaSim supports adults aged 18 to 85 years within the [input
limits](CLI_USAGE.md#fields) for body size, hemoglobin, organ function, and
patient conditions.

### Patient conditions

Coronary disease reduces coronary flow reserve.

Heart failure changes resting cardiac output, filling pressure, and sensitivity
to loading conditions (Schwartzenberg 2012; Westermann 2008). With reduced
ejection fraction (HFrEF), stroke volume is more sensitive to afterload and does
not rise with fluid loading. With preserved ejection fraction (HFpEF), the
ventricle is stiffer and fills less during tachycardia. Chronic left atrial
hypertension increases lymphatic drainage, so pulmonary edema starts at a higher
filling pressure. The reserve above the resting pressure is still smaller than
in a normal heart.

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

### Patient variability

Without `patient_variability`, each patient has the typical drug response. With
it, each session samples the following, and the session seed fixes the sampled
patient:

- propofol kinetics and bispectral index (BIS) sensitivity from Eleveld 2018,
  with standard deviations halved to match VitalDB inductions with total
  intravenous anesthesia (TIVA)
- remifentanil kinetics (Eleveld 2017) and fentanyl kinetics (Bae 2020)
- sevoflurane potency for awakening and noxious stimuli (Heyse 2012)
- the effects of propofol and opioids on vascular resistance (Su 2023), with the
  same variability for sevoflurane as for propofol
- the timing, sensitivity, and shape of the respiratory-rate response to
  morphine and hydromorphone (Olofsen 2026)

## Pharmacokinetics

The pharmacokinetic (PK) model of each intravenous drug has a central
compartment, up to two peripheral compartments, and effect sites for its
clinical effects.

| Drug | Model |
|------|-------|
| Propofol | Eleveld 2018 arterial model with its opioid-regimen covariate |
| Remifentanil | Eleveld 2017 |
| Fentanyl | Bae 2020 up to 70 kg; at higher weights it scales with pharmacokinetic mass (Shibutani 2004). Clearance scales with hepatic function |
| Morphine and hydromorphone | Meissner 2025, scaled with fat-free mass, age, and hepatic and renal function |
| Lidocaine | Foong 2025 |
| Rocuronium | Scaled with adjusted weight toward ideal body weight in obesity (Meyhoff 2009). Cirrhosis enlarges its volumes and lowers hepatic clearance (Khuenl-Brady 1996; Magorian 1995) |

Inhaled agents move between lung, blood, and tissue compartments. Lean tissue
volumes scale with fat-free mass, and the fat volume grows with excess weight.
The effects of inhaled agents depend on brain partial pressure, expressed as
age-adjusted minimum alveolar concentration (MAC) (Davis and Mapleson 1981;
Yasuda 1991). Uptake of nitrous oxide concentrates the other gases in the
alveoli, and its washout dilutes them.

When blood volume changes, the vascular part of each drug's central volume is
rescaled. Shed blood carries drug away at the plasma concentration, so blood
loss alone leaves the concentration unchanged. Other volume changes conserve
drug mass.

Clearance depends on organ function, renal perfusion, and cardiac output as
specified by each model:

- Fentanyl and lidocaine clearances were measured during anesthesia and surgery,
  so they apply at the model's cardiac output under anesthesia (Bae 2020; Foong
  2025).
- Renal perfusion affects only the renal clearance that remains after any renal
  impairment.
- Blood loss reduces propofol and remifentanil clearance, and volume replacement
  restores it. This is based on swine data (Johnson 2001, 2003, 2004).
- Midazolam and propofol reduce each other's clearance within the concentration
  ranges studied (Vuyk 2009; Lichtenbelt 2010).

Below 30% of resting cardiac output, drug transport to organs, peripheral
compartments, and effect sites falls in proportion to flow. A dose given during
arrest therefore does not reach its effect sites until compressions circulate
it. Esterase metabolism of remifentanil, succinylcholine, and esmolol continues
without flow, and sugammadex still binds rocuronium in plasma.

## Target-controlled infusion

Target-controlled infusion (TCI) adjusts infusion rates within pump limits to
reach a predicted plasma or effect-site target. For propofol, remifentanil,
fentanyl, and rocuronium, TCI uses the published model with only age, weight,
height, and sex, as clinical pumps do. The prediction omits sampled variability,
organ impairment, blood volume, cardiac output, midazolam, and sugammadex, so
the patient's concentrations can differ from the target. After blood loss, for
example, propofol and remifentanil concentrations rise above it. TCI for other
drugs predicts from the patient's current PK.

The pump does not know about hand boluses. A TCI program that starts after a
manual rate assumes no drug on board, and a stopped program resumes from its own
history. Setting a manual infusion rate turns off TCI for that drug. When
maintenance is initialized, `tci_enabled` chooses between a 30-minute TCI
history and manual rates.

Decrement times for propofol, remifentanil, and fentanyl use the pump model. The
decrement time for rocuronium uses the patient's PK and neuromuscular model,
including sugammadex and inhaled agents. For norepinephrine, it is the time for
the concentration to fall halfway to its endogenous baseline.

## Anesthetic drug effects

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
equivalents. Potency depends on the patient and on the effect. Midazolam alone
lowers BIS only to the sedation range and causes little ventilatory depression
in a patient who is breathing unassisted (Marty 1986), but it is synergistic
with propofol (Short 1992). Nitrous oxide and ketamine also affect
responsiveness. Ketamine adds to opioid analgesia and, like lidocaine, reduces
responses to stimulation. During propofol or sevoflurane anesthesia, ketamine
raises BIS because it increases high-frequency EEG activity.

## Stimulation

BIS arousal during stimulation depends on laryngoscopy tolerance. The heart rate
(HR) and mean arterial pressure (MAP) responses use separate, stronger stimulus
intensities (Albertin 2005). Laryngoscopy and tracheal extubation under light
anesthesia can trigger laryngospasm, which resolves more slowly while the
patient remains lightly anesthetized without positive airway pressure.

Propofol lowers baseline pressure but does not reduce the cardiovascular
response to stimulation, whereas opioids do reduce it (Kazama 1997). This
interaction uses the propofol concentration at the BIS effect site because it
was calibrated at steady concentrations. In the first minutes after induction,
the concentration at the faster clinical-response site is about twice as high.

Fentanyl's effects on the pressure and HR responses are fitted separately to
Kazama's equilibrated concentrations and intubation responses. With inhaled
agents, fentanyl and remifentanil also lower the concentration needed to block
the response and blunt the HR rise. This agrees with opioid reductions in
volatile MAC and MAC-BAR (Katoh 1999; Lang 1996). Other opioids are converted to
remifentanil equivalents.

Surgery also increases vascular resistance by 15%, and anesthetics do not reduce
this increase. Its size comes from the rise in MAP from before incision to
during surgery in VitalDB TIVA cases. The increase develops with a 5-minute time
constant, which gives a MAP rise of about 15% within 5 minutes of incision at
Katoh's MAC-BAR concentrations. Septic shock removes the increase. Nociception
resets the slow MAP feedback as it resets the baroreflex, so pressor responses
persist while stimulation continues.

In the browser, maintenance sessions and intraoperative crisis cases begin
during surgery, and emergence begins after surgery.

## Neuromuscular block

The [neuromuscular model](../anasim/patient/pd/nmba.py) uses Kleijn 2011 for
rocuronium and sugammadex. Sevoflurane and nitrous oxide strengthen rocuronium
block at both peripheral and central muscles (Kleijn 2011; Fiset 1991).
Sugammadex binds free plasma rocuronium and speeds its removal from the effect
site, and the complex is cleared at the rate of sugammadex. Both drugs use
measured creatinine clearance when it is supplied and otherwise an estimate from
age and renal function. Other renal effects use the renal function factor.

Train-of-four (TOF) represents the adductor pollicis. The diaphragm and larynx
equilibrate faster and need higher rocuronium concentrations, so breathing
recovers before TOF. Residual block reduces pharyngeal reserve and makes the
airway more likely to collapse under hypnotics (Herbstreit 2009). Awake muscle
tone and positive airway pressure keep the airway open.

Succinylcholine uses Roy 2002 PK. Twitch onset and recovery are calibrated to
Kopman 2003, and central muscle sensitivity to Dhonneur 1999. Its phase I block
lowers single-twitch height without TOF fade and releases potassium from muscle
(Martyn and Richtsfeld 2006).

## Hemodynamics

The [hemodynamic model](../anasim/physiology/hemodynamics.py) uses Su 2023's
model of the effects of propofol and remifentanil on vascular resistance, stroke
volume, and HR. It is linked to blood volume, the pulmonary circulation,
reflexes, vasoactive drugs, and distributive shock.

Drugs act through hemodynamic effect sites on a slowly adapting tone. This tone
uses Su's turnover rate and also responds to MAP feedback, chemoreflexes, blood
loss, and temperature. As a result, pressure falls within minutes of induction,
whereas compensation is slow.

### Propofol

The effect of propofol on pressure equilibrates over about 10 minutes,
consistent with Jeleazcov 2015's volunteers and Kazama 1999's patients aged 60
to 69. Above 65, the effect develops more slowly and becomes larger, as Kazama
found at 70 to 85 years. By age 78, the half-time is about twice as long.

Compared with Su's model:

- The effect on vascular resistance is smaller, as Fregolent 2025 also found.
  Early after induction, propofol alone therefore lowers pressure about half as
  much as propofol with an opioid (Ebert 1992 and 1995; Billard 1994).
- The effect on stroke volume increases less with age, consistent with Kazama's
  finding of similar sensitivity from 20 to 69 years.

Propofol's effects on resistance and venous pooling increase with pre-induction
MAP, so patients with higher starting pressures have larger relative falls
(Morley 2010).

Propofol also reduces stressed volume and the resistance to venous return (de
Wit 2016), through the same effect site as its arterial effects. In addition,
the loss of sympathetic venous tone moves up to 30% of stressed volume into
unstressed volume. This shift follows the plasma propofol concentration over a
few minutes, and it is why induction lowers filling pressure and cardiac output
more than resistance (Fairfield 1991; Zucker 2022; Sjøen 2024). Blood loss draws
on this pooled blood first, fluid loading does not refill it, and the
alpha-adrenergic effect of epinephrine reverses it.

### Opioids

Opioids equilibrate within about a minute and are expressed as remifentanil
equivalents. Fentanyl has its own effect site.

Opioid effects level off above 2.5 ng/mL, because Sjøen 2024 found the same
falls in pressure and HR at remifentanil doses that gave peak effect-site
concentrations of 2, 4, and 8 ng/mL. The direct vasodilating effect of
remifentanil is a quarter of Su's volunteer estimate. This value lies between Su
and Garraza-Obaldia 2024's patient model, which found no direct effect. The
effect of remifentanil on HR is half of Su's estimate, because Su's volunteers
breathed spontaneously and were probably hypercapnic.

The interactions between propofol and opioids on resistance, HR, and stroke
volume were fitted to VitalDB propofol-remifentanil inductions, which represent
routine practice. The [validation
guide](../validation/vitaldb/README.md#hemodynamics) gives the recorded and
modeled falls in MAP. Induction trials that used higher doses (Sawano 2013;
Sjøen 2023 and 2024; Poterman 2017) reported larger falls, and the model is
within two standard deviations of their results. The stroke volume interaction
rises over a wider range of propofol concentrations than in Su's model, which
makes it larger at the doses used in these trials.

### Sevoflurane and adjuncts

Sevoflurane lowers stroke volume, vascular resistance, and HR. The fall in
pressure was fitted to routine bolus inductions in VitalDB, in which the falls
did not depend on the opioid dose. Most of the fall comes from stroke volume. In
unstimulated volunteers, cardiac index fell 28% at 1 and 1.5 MAC, and resistance
fell 4% and 16% (Malan 1995). Ebert 1995's volunteers had larger falls in
pressure and no change in HR, and the model is within two standard deviations of
their pressures.

The sevoflurane effect equilibrates with a 14-minute half-time. This matches the
gradual fall in MAP over 10 minutes of sevoflurane by mask after propofol in
Ebert's study. The effects of propofol and sevoflurane on resistance and stroke
volume combine as independent fractions.

Midazolam causes mild vasodilation and venous pooling, which increase when it is
combined with opioids (Marty 1986; Heikkilä 1984). Etomidate changes vascular
resistance and stroke volume very little. Nitrous oxide causes mild sympathetic
vasoconstriction and slightly depresses contractility.

### Reflexes

MAP feedback changes HR or vascular resistance, depending on rhythm, anesthetic
depth, blood gases, and stimulation. When muscarinic block or a fixed rhythm
prevents slowing, high MAP lowers endogenous vascular resistance (Weissler 1957;
Tweed 1972). Anesthesia reduces the baroreflex and the tachycardia of hemorrhage
(Ebert 1992; Sellgren 1994). Older patients have lower cardiovagal gain and less
reflex tachycardia, while baroreflex control of vascular resistance is preserved
(Kardos 2001; Laitinen 2004; Monahan 2007). Severe hypoxia lowers HR and
contractility.

Under propofol, MAP has little effect on HR (Garraza-Obaldia 2024). Slow
compensatory tachycardia therefore decreases with anesthetic depth, while
slowing in response to high pressure remains (calibrated to Sawano 2013 and
Takahashi 2002). Opioids also suppress reflex tachycardia but not reflex
slowing, so HR falls along with pressure after propofol-remifentanil induction
(Sjøen 2024). Remifentanil also causes a vagal bradycardia that does not occur
with fentanyl and that antimuscarinics block (Poterman 2017). Its size matches
the fall in HR after induction in VitalDB TIVA cases. Septic tachycardia depends
directly on sepsis severity and persists under anesthesia.

### Preload and heart rate

Blood loss lowers filling pressure and preload. Reflex venoconstriction and
transcapillary refill from the interstitium partly compensate. Venous
recruitment increases for losses above 30%, so slow class III and IV hemorrhage
remains compensated until stressed volume is exhausted. High intrathoracic
pressure from positive end-expiratory pressure (PEEP) or a closed adjustable
pressure-limiting (APL) valve raises right atrial pressure and impedes venous
return.

In untreated atrial fibrillation, the ventricular rate rises with sympathetic
drive, and stroke volume falls further at rapid rates. In any rhythm, venous
return limits stroke volume above 145 bpm, and short diastolic filling reduces
it further above 170 bpm.

### Acidemia, electrolytes, and hypothermia

Below pH 7.2, contractility and the response to adrenergic drugs fall linearly
to 30% at pH 6.8. This is extrapolated from isolated failing human myocardium
(Schotola 2012).

Hypocalcemia lowers contractility and vascular resistance. Intravenous calcium
restores contractility without raising systemic vascular resistance above
baseline (Denlinger 1975; Lang 1988). Hyperkalemia slows conduction, depresses
contractility, and can trigger ventricular arrhythmias or asystole. Elevated
ionized calcium raises the potassium threshold for these effects.

Core temperatures below 33 °C slow HR. Temperatures below 30 °C depress
contractility and increase the risk of ventricular fibrillation.

## Cardiovascular drugs

Vasoactive drugs change vascular resistance, stroke volume, and HR. Adrenergic
and muscarinic antagonists change both these responses and the reflexes.
Anaphylaxis reduces the response to adrenergic drugs.

### Norepinephrine

The norepinephrine model includes endogenous secretion and a clearance that
depends on propofol (Li 2024). Its effects on resistance and stroke volume
increase with anesthetic depth.

### Epinephrine

Epinephrine uses arterial clearance (Ensinger 1992), which falls with age (Esler
1995). Its effective mixing volume is calibrated to bolus responses. This volume
sets the initial concentration after a bolus and, with clearance, the rate at
which plasma concentration falls. Cardiac and vascular responses are calibrated
to boluses in awake and anesthetized patients and to infusions in awake
volunteers (Guinard 1995; Takahashi 2002; Freyschuss 1986; Stratton 1985;
Ensinger 1992; Marker 1998; Leenen 2005).

Older patients are less sensitive to epinephrine (Leenen 2005; Pan 1986;
Hogikyan 1994), so the arterial alpha-adrenergic C50 rises with age. During
infusions, higher plasma concentrations and weaker baroreflex buffering offset
this lower sensitivity. Boluses raise HR less in older patients.

Under propofol, epinephrine recruits pooled venous blood at lower concentrations
than those that constrict arteries. A second alpha-adrenergic component acts at
higher concentrations, with a gradual concentration response and its own delay.
It maintains vascular resistance at the infusion rates used in shock and
supports pressure during CPR. The concentration-response calibration covers
concentrations up to Ensinger's highest infusion step, 0.22 mcg/kg/min.

### Phenylephrine and ephedrine

Phenylephrine recruits venous blood and raises afterload, so its effect on
cardiac output depends on preload (Rebet 2016).

Ephedrine uses the oral disposition and tolerance reported by Persky 2004,
applied to intravenous use. Its peak responses are calibrated to Meng 2011, and
its effect on pressure during induction was checked against Sjøen 2026. Alpha
and beta blockade shift its vascular and cardiac concentration-response curves
separately. With repeated doses, tachyphylaxis reduces the pressor and peak HR
responses, while a smaller HR response remains.

### Atropine

The HR response to atropine follows its peripheral concentration (Hinderling
1985) and is reduced by propofol (Horiguchi 2002). Both antimuscarinics and
beta-adrenergic agonists relieve sinus bradycardia. After atropine, venous
unloading lowers stroke volume over minutes, and the fall comes earlier at low
preload (Weissler 1957; Berry 1959). Adrenergic drive restores stroke volume
during pacing (Ross 1965).

### Ketamine

Ketamine raises HR, stroke volume, and vascular resistance through sympathetic
stimulation (Tweed 1972). Its HR effect starts sooner than the others and adds a
fixed increase, including after atropine (Bovill 1972). Baroreflex gain falls at
anesthetic concentrations, based on rat data (Akine 2001). A calibrated offset
of up to 20 mmHg reduces reflex slowing and vasodilation, to fit pressure rises
after atropine and during pacing. Beta blockade leaves a pressor response, and
alpha blockade reduces it (Bovill 1972; Bålfors 1983).

Sympathetic reserve is exhausted by severe blood loss, septic shock, profound
hypotension, severe acidemia, or myocardial energy failure. Ketamine then loses
its sympathetic effect and lowers contractility and vascular resistance directly
(Waxman 1980).

### Amiodarone

Rapid boluses of intravenous amiodarone briefly lower vascular resistance and
blood pressure, mostly through the polysorbate 80 solvent. Slower infusions
lower pressure much less (Munoz 1988). Contractility falls over several minutes,
so patients with HFrEF lose more cardiac output and blood pressure than patients
with a normal left ventricle (Kosinski 1984). Sinus rate slows over 15 to 30
minutes (Munoz 1988), and sympathetic HR reflexes weaken over the same period.

## Heart and brain oxygen supply

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
1998). Function recovers within about two minutes after supply returns.

Below 20% of normal, neurons are injured, and injury is slower when more supply
remains (Jones 1981). After the equivalent of 5 minutes without supply, the
patient remains comatose, although BIS and breathing can recover.

CO₂ narcosis reduces responsiveness and airway tone without anoxic injury. Its
severity depends on both PaCO₂ and pH (Westlake 1955). With acidemia, it rises
as PaCO₂ increases from 60 to 100 mmHg and pH falls from 7.3 to 7.1. In
compensated hypercapnia, it rises as PaCO₂ increases from 100 to 140 mmHg. Onset
and recovery have a 30-second time constant. CO₂ does not change BIS directly:
BIS was unchanged at EtCO₂ 3-12 kPa under propofol anesthesia (Turnbull 2008).

## Cardiac arrest and resuscitation

Arrest is confirmed when ventricular fibrillation (VF), pulseless ventricular
tachycardia (VT), asystole, HR below 10 bpm, or pulseless electrical activity
(PEA) lasts 15 seconds. In PEA, the ECG remains organized while the heart's own
beats produce less than 0.3 L/min with a pulse pressure below 5 mmHg. The
simulation continues through arrest, so the learner can resuscitate and treat
the cause.

### Course of arrest

Severe myocardial ischemia or hypoxemia first depletes an energy reserve that
lasts about 4 minutes without circulation. If a perfusing rhythm returns within
this time, as after defibrillation in the electrical phase of arrest (Weisfeldt
2002), circulation is restored.

Once the reserve is used up, the heart slows over several minutes. P waves
disappear (junctional rhythm), and complexes widen and shrink (idioventricular,
then agonal rhythm) until the ECG is flat. Untreated VF becomes fine and then
asystole. Meanwhile, injury builds up that reperfusion does not reverse within a
session, and it reduces contractility (Laurent 2002). The session ends when this
injury is complete, about 11 minutes after circulation stops.

### Chest compressions

[Chest compressions](../anasim/physiology/resuscitation.py) at 110/min pump a
fifth of resting cardiac output at normal filling and vascular resistance, and
little after major blood loss. This flow and vascular resistance set arterial
pressure between compressions. Each compression adds a pressure pulse that does
not produce flow.

Coronary flow occurs between compressions, against venous and left atrial
back-pressure plus myocardial closing pressure. The energy reserve recovers when
diastolic pressure exceeds about 25 mmHg. This is the CPR target in Meaney 2013,
and it corresponds to the coronary perfusion pressure needed for return of
circulation in Paradis 1990. In an anesthetized adult, compressions give about
70/25 mmHg and EtCO₂ near 15-20 mmHg, so CPR keeps VF coarse and makes fine VF
coarse again.

Epinephrine raises diastolic and coronary perfusion pressure during compressions
(Kim 2025). The higher afterload reduces forward flow and lowers EtCO₂ by a few
mmHg at constant ventilation (Cantineau 1994).

In primary asystole, beats resume after 30 seconds of adequate coronary oxygen
supply during compressions if energy remains and the myocardium is viable. An
interruption or inadequate perfusion restarts this interval.

### Defibrillation and cardioversion

A 200 J biphasic shock terminates coarse VF and pulseless VT 9 times in 10, and
fine VF or hypoxic arrest less often. Each shock also causes mild contractile
stunning that reduces stroke volume by about 10% and resolves over several
minutes of coronary perfusion (Tokano 1998).

After brief VF, arterial pressure recovers over 5 to 20 seconds as beats refill
the arterial tree and reversible oxygen debt clears. Recovery takes longer after
longer VF (Park 1994). After prolonged arrest, the heart beats only as well as
its energy reserve and accumulated injury allow. A late shock without CPR
therefore leaves PEA or asystole, and resuscitation after the reserve is
exhausted leaves hypotension for the rest of the session (Laurent 2002).

Synchronized shocks fire on the next R wave and usually convert atrial
fibrillation, supraventricular tachycardia, and perfusing VT. They cannot fire
in VF or asystole, and compressions do not trigger them. An unsynchronized shock
that falls near the T-wave peak of a perfusing organized rhythm causes VF 15% of
the time.

Each shock needs a new charge, which takes 5 seconds and expires after 60
seconds (ZOLL R Series). Compressions can continue while it charges. Disarming
cancels a charge or a pending synchronized shock.

### Ventricular arrhythmias

[Severe ischemia, hypoxia, hyperkalemia, or severe
hypothermia](../anasim/physiology/arrhythmia.py) in a beating heart can start VT
or VF. After a successful shock, VF often recurs over the next two minutes if
myocardial ischemia or hypoxia persists (Koster 2008; van Alem 2003).

Amiodarone (Korth-Bradley 1996) and lidocaine convert sustained VT with a pulse
over tens of minutes when potassium is not severely elevated, and amiodarone
converts it more often (Somberg 2002; Scheinman 1995). In VF, both drugs make a
shock more likely to succeed after CPR delivers them (Kudenchuk 2016), and
neither prevents rearrest (Salcido 2018).

## Lidocaine toxicity

In awake patients, [lidocaine](../anasim/physiology/toxicity.py) raises HR and
blood pressure above an effect-site concentration of about 4 mcg/mL and causes
seizures above 5 mcg/mL (Foldes 1960; Jorfeldt 1968). Between 11 and 20 mcg/mL,
CNS depression and apnea replace seizures. Contractile depression begins above
13 mcg/mL, and higher concentrations widen the QRS complex and can cause
ventricular arrhythmias or arrest. Acidemia worsens both CNS and cardiac
toxicity.

A 2 g infusion over an hour causes hypotension and coma in sinus rhythm without
arrest, as in Armstrong 1988. The model clears the drug faster than that
patient did, whose ejection fraction was 15%.

Midazolam, propofol, sevoflurane, and nitrous oxide suppress seizures and the
sympathetic rise in HR and pressure. Neuromuscular block hides convulsions
without stopping the seizure.

Lipid emulsion reduces lidocaine delivery to its effect sites by about a fifth,
as it reduced unentrapped exposure in volunteers (Heinonen 2015). It also
lessens conduction block, arrhythmogenic stress, and contractile depression
(Fettiplace 2014).

## Fluids

### Volume kinetics

Lactated Ringer's and 0.9% saline move between blood and the interstitium (Hahn
2017). Distribution slows when MAP is low, so fluid given during hypotension
after induction stays in the circulation longer (Hahn 2021). Urine output
depends on MAP, age, and renal function. Saline is excreted more slowly than
lactated Ringer's: after 2 L over an hour, healthy volunteers pass about half as
much urine in 6 hours (Reid 2003). More of a 5% albumin infusion stays in the
circulation at first.

Fluid also leaves the circulation through capillary leak and other slow
exchanges into a separate compartment, which slowly returns fluid when blood
volume is low. Anaphylaxis causes rapid plasma leak. Red cells stay in the
circulation, so hemoglobin concentration rises (Fisher 1986). The leak slows as
the reaction subsides after the trigger is removed, and an ongoing reaction can
require repeated crystalloid boluses.

Transfused blood tracks donor red-cell volume and hemoglobin separately. A
raised left atrial pressure increases lung water, shunt, and lung stiffness,
which links fluid balance to respiratory mechanics and gas exchange.

### Electrolytes and fluid composition

[Fluid balance](../anasim/physiology/fluid_balance.py) tracks sodium, potassium,
chloride, calcium, magnesium, and phosphate in one extracellular volume, and
albumin separately in plasma and the interstitium. Water moves between
extracellular fluid and cells until their tonicity is equal, so the free water
in lactated Ringer's and lipid emulsion spreads through total body water.

Each fluid adds its own ions and albumin, and buffer base equal to its strong
ion difference. Saline therefore causes a hyperchloremic acidosis, and lactated
Ringer's does not (Scheingraber 1999). Packed red cells have the composition of
a 3-week-old unit, at pH 6.5 and PCO₂ 98 mmHg with supernatant potassium 38 and
lactate 22.5 mmol/L (Rabcuka 2025), so rapid transfusion adds acid, CO₂, and
potassium. Lactate from Ringer's and packed cells, and citrate from packed
cells, return base as they are metabolized (Kramer 2003).

Potassium exchanges with a large cellular pool over minutes. Epinephrine moves
it into cells. Succinylcholine (Martyn and Richtsfeld 2006), malignant
hyperthermia, and severe acidemia below pH 7.15 move it into plasma.

Ionized calcium depends on albumin, citrate, and pH (Thode 1983; Bolan 2001), so
dilution and rapid transfusion lower it. Injected calcium mixes through
extracellular fluid, and calcium above the patient's resting ionized pool is
then cleared. One gram of calcium chloride therefore raises ionized calcium by
about 0.4 mmol/L, and half of the rise is gone by 30 minutes (Ansari 2025).
Calcium lost to dilution or bound by citrate is not replaced within a session
(Bolan 2001).

### Renal acid-base handling

The kidneys excrete the daily fixed acid load when urine output and renal
perfusion allow. Over about a day, they also move bicarbonate toward the chronic
compensation for the current PaCO₂: 0.48 mmol/L per mmHg in hypercapnia up to
70 mmHg (González 2018) and 0.4 mmol/L per mmHg in hypocapnia (Krapf 1991).
Chloride moves in the opposite direction to bicarbonate.

## Respiratory control

[Ventilatory control](../anasim/physiology/respiration.py) combines anesthetic
effects, CO₂ feedback, hypoxic drive, and muscle strength.

Propofol reduces tidal volume, and opioids mainly reduce respiratory rate.
Sevoflurane causes rapid shallow breathing that progresses to apnea near 2 MAC
(Doi and Ikeda 1987). Rapid high-dose ketamine boluses briefly depress central
drive (Bourke 1987). Hypocapnia and neuromuscular block suppress respiratory
effort, and rising CO₂ can restore breathing during opioid administration. Under
anesthesia, the apneic threshold moves with the resting PaCO₂ that the drugs
produce (Hickey 1971).

For morphine and hydromorphone, the effect on respiratory rate is modeled
separately from the effect on ventilatory drive (Olofsen 2026). The controller
combines Olofsen's fractional opioid response with Bouillon's nonlinear response
to alveolar CO₂.

Hypoxic drive declines during sustained hypoxia and is reduced by propofol,
sevoflurane, and nitrous oxide (Nieuwenhuijs 2000; Sarton 1996). During airway
obstruction, airflow is limited while respiratory effort continues.

Metabolic acidosis lowers the PaCO₂ set point by 1.2 mmHg per mmol/L of base
deficit, with a two-minute time constant. Anesthetics and opioids still limit
the ventilatory response.

[Respiratory muscles fatigue](../anasim/physiology/fatigue.py) when mean
inspiratory pressure, including effort against an obstructed airway, exceeds
about 15% of maximum. Endurance falls steeply as the load rises further
(Bellemare and Grassino 1982). Fatigue reduces spontaneous tidal volume and
develops sooner when oxygen supply is inadequate. Pressure support and mandatory
breaths reduce inspiratory effort (Leung 1997). Most fatigue recovers within
minutes of rest (Aubier 1981; Rohrbach 2003), and a smaller deficit persists for
up to 24 hours after prolonged heavy loading (Laghi 1995; Travaline 1997).

## Gas exchange and blood chemistry

### Oxygenation

Ventilation exchanges gas with lung stores. These stores affect the time course
of preoxygenation, apnea, and blood loss, and they determine oxygen uptake from
the circuit. Shunt mixes blood by oxygen content, which includes oxygen bound to
hemoglobin and dissolved oxygen.

Some pulmonary blood also passes through poorly ventilated units with a
ventilation-perfusion ratio (V/Q) of 0.05. These units lower PaO₂ most at low
FiO₂ and little at high FiO₂. Their share of cardiac output rises with age,
which produces the sea-level A-a gradient on room air (Crapo 1999). Bronchospasm
adds more of these units, and COPD adds a separate group with a V/Q of 0.1
(Wagner 1977). Neither condition adds much shunt, so added oxygen corrects most
of the hypoxemia.

In unconscious patients, dependent airways also close at end-expiration, and
this closure increases after age 30. PEEP of about 6 cmH₂O, or more with
obesity, holds these airways open (Gunnarsson 1991; Hedenstierna 2015). Older
patients therefore oxygenate worse under anesthesia, especially without PEEP.
After upper abdominal surgery, FRC stays at or below its value under anesthesia
(Wahba 1975; Putensen-Himmer 1992), so this closure persists after emergence.
After lower abdominal surgery, about a third of it persists (Lindberg 1992).

### Gas transport and stores

[Gas transport](../anasim/physiology/gas_transport.py) moves oxygen, CO₂, and
buffer base through pulmonary capillary, arterial, and mixed venous blood, with
shunt bypassing the lungs. Two tissue groups and an extracellular pool store
CO₂, with calibrated capacities (Magor-Elliott 2021; Wooten 2003). Blood flow
determines exchange with these stores, and respiratory feedback responds to
arterial PCO₂. Lung tissue accounts for 5% of oxygen consumption and CO₂
production and exchanges directly with alveolar gas (Loer 1997). Without
pulmonary blood flow, ventilation therefore lowers alveolar PCO₂ only to about
2 mmHg.

When the airway is closed or circuit gas is exhausted, gas that is absorbed is
not replaced. Lung gas volume shrinks, and absorption atelectasis follows. The
model tracks this absorbed volume separately from the lung volume used for
mechanics and inhaled agents. Opening the airway refills the lungs with gas from
the circuit (Laviola 2020).

During apnea with an open airway, circuit oxygen washes through anatomic dead
space slowly unless earlier breathing removed its nitrogen. The circuit and
lungs draw on one flow-limited supply of gas. Nitrous oxide wash-in adds
convective inflow, which raises the uptake of alveolar oxygen and volatile
agent. Its washout dilutes alveolar oxygen and CO₂ when inspired oxygen is low.

### Oxygen delivery and lactate

Oxygen consumption equals metabolic demand until oxygen delivery falls to a
critical level. At this level, the oxygen extraction ratio is 0.6 (Ronco 1993).
The critical delivery is therefore proportional to demand, so it is lower during
hypothermia and anesthesia. Below this level, consumption and aerobic CO₂
production fall in proportion to delivery, and the unmet demand produces
[lactate](../anasim/physiology/lactate.py). The amount of lactate produced per
unit of oxygen deficit was fitted to data on oxygen debt in dogs with
hemorrhagic shock (Dunham 1991).

Epinephrine and sepsis increase aerobic lactate production (Grip 2015; Revelly
2005). Lactate has a volume of distribution of 0.26 L/kg (Levraut 1998). Its
elimination follows Michaelis-Menten kinetics, so clearance falls as the
concentration rises. The model assumes that the liver provides 60% of the
elimination capacity. This hepatic portion is reduced in proportion to liver
function and to the fraction of oxygen demand that is met.

Each mmol of endogenous lactate that accumulates above the resting amount
removes 1 mmol of buffer base, and clearance of this lactate returns the base.
Lactated Ringer's solution raises the measured lactate without adding acid, and
metabolism of this lactate adds buffer base (Zitek 2018). Blood gas analyzers
measure only the L-isomer, half of the racemic lactate in Ringer's.

### Acid-base

The blood gas model includes respiratory, lactic, and fluid-related acid-base
disturbances at 37 °C. [Blood buffering](../anasim/physiology/blood_chemistry.py)
is based on Rees 2010, with plasma albumin and phosphate charge from Figge 2018.
Oxygen affinity uses the temperature, pH, and PCO₂ terms of Dash 2016 at a
reference 2,3-DPG level. Gas exchange and changes in oxygen affinity conserve
CO₂, oxygen, and buffer base. Bleeding, transfusion, fluids, and urine change
the total buffer base by the amounts that they carry.

The interstitium buffers with its albumin and phosphate and with a fixed
cellular buffer, the extracellular buffer of Wooten 2003. With these buffers,
acute changes in PaCO₂ move pH and bicarbonate within the human ranges of
Brackett 1965 and Arbus 1969, and standard base excess changes little. Standard
base excess is calculated from arterial pH and bicarbonate with the Van Slyke
equation, as blood gas analyzers do. It is near 0 mmol/L in nonpregnant adults
and about −2 mmol/L in late pregnancy.

## Respiratory mechanics and aeration

The [mechanics model](../anasim/physiology/resp_mech.py) combines airway
resistance, compliance, tissue viscoelasticity, and inspiratory muscle pressure.
Baseline mechanics depend on body size and patient conditions. Airway and muscle
pressures determine the intrathoracic pressure that affects venous return.

Ventilator assistance reduces muscle effort. Under anesthesia, lung inflation
ends inspiratory effort earlier. Severe hyperinflation flattens the diaphragm
and lowers the inspiratory pressure that the muscles can generate.

Bronchospasm increases expiratory resistance as lung volume falls and can limit
expiratory flow, which causes gas trapping and intrinsic PEEP. Epinephrine
relieves bronchospasm through beta-2 receptors, an effect that labetalol
antagonizes. Sevoflurane and ketamine add partial bronchodilation.

[Lung aeration](../anasim/physiology/lung.py) depends on the pressure history.
Recruitment and collapse change gas volume, shunt, and compliance. Inert gas in
the alveoli slows repeated collapse. Atelectatic shunt changes separately from
bulk FRC and compliance (Rothen 1995). Preoxygenation and apnea at induction
leave a shunt of about 5%, and maintenance sessions start with this shunt. PEEP
of 5 cmH₂O stops further collapse but does not reopen the lung, and atelectasis
persists after emergence (Benoit 2002).

## Temperature

Awake thermoregulation maintains the baseline core temperature. Basal heat
production scales with body weight to the 0.75 power. Temperature and sepsis
change metabolic demand and heat production, warming adds heat, and surface
cooling removes it.

General anesthetics, including etomidate and ketamine, inhibit active
regulation, increase heat loss, and redistribute heat from the core to the
periphery (Matsukawa 1995). Midazolam shifts thermoregulatory thresholds much
less (Kurz 1995). Cooling from redistribution in the first hour is smaller in
heavier patients, who have higher ratios of weight to surface area (Kurz 1995).

A cold patient shivers only after BIS has risen and responsiveness has
returned. Opioids and neuromuscular block reduce shivering.

## Malignant hyperthermia

[Malignant hyperthermia](../anasim/physiology/malignant_hyperthermia.py) (MH)
can start during sevoflurane anesthesia or after succinylcholine. Onset after a
volatile anesthetic varies and often comes in the second or third hour (Visoiu
2014), so the instructor or case sets the time. Skeletal muscle hypermetabolism
continues after the trigger is removed.

Untreated, oxygen consumption and CO₂ production approach five times baseline,
and anaerobic glycolysis and calcium cycling add extra heat. The gas exchange,
lactate, fluid, and temperature models then produce hypercapnia, lactic
acidosis, hyperkalemia, and a rapidly rising temperature. Tachycardia and
generalized rigidity increase with the crisis. Rigidity stiffens the chest wall,
limits spontaneous tidal volume, and persists under neuromuscular block.

Dantrolene PK follows the Ryanodex label. Dantrolene controls the crisis and
reduces twitch height without TOF fade, but leaves ventilation unchanged
(Flewellen 1983).

## Anesthesia machine and ventilation

### Breathing circuit

The [circle system](../anasim/machine/circuit.py) mixes fresh gas with circuit
gas in a 6 L circuit that includes a 3 L reservoir bag and bellows. It accounts
for the exchange of oxygen, volatile agent, nitrous oxide, and nitrogen, and for
excess gas that spills from the circuit.

When patient uptake exceeds the inflow of fresh gas and vapor, the reservoir
empties. The circuit is sealed, so no room air enters. As the bellows collapses,
delivered breaths become smaller and inspiratory resistance rises.

Vaporizer output depends on flow, the dial setting, and the remaining agent. A
proportioning system limits nitrous oxide flow to three times the delivered
oxygen flow, and oxygen supply failure stops delivery of both oxygen and nitrous
oxide.

### Ventilator

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
during expiration. Tube obstruction raises airway resistance.

### Bag and mask ventilation

Manual bag breaths stop at the APL setting (default 20 cmH₂O), so obstruction or
bronchospasm lowers the delivered volume. During unassisted breathing on the
circuit, an APL above 20 cmH₂O with excess fresh gas builds continuous airway
pressure up to the APL or the pressure at which the mask leaks.

During mask ventilation, the mean airway pressure over recent breaths holds the
pharynx open. A loose mask loses part of each assisted breath, lowers this
pressure, and reduces the measured airway pressure, flow, and volume. Pressures
above 20 cmH₂O leak even with a tight seal (Bouvet 2014). A full seal delivers
the tidal volume of two-handed mask ventilation, and an 80% seal that of a
one-handed hold (Joffe 2010). Room air enters around a loose mask during
unsupported breaths, or when an empty reservoir cannot supply an assisted
breath.

## Monitors

### ECG, arterial pressure, and pulse oximetry

ECG, arterial pressure, and plethysmography use the same beat timing, and HR is
averaged over beat intervals.

The arterial waveform depends on MAP, stroke volume, and age-dependent arterial
compliance (Mahdi 2017; Franklin 1997). Compliance increases as pressure falls
(Langewouters 1984), so pulse pressure narrows during hypotension and widens
with vasopressors. Under propofol, systolic pressure falls about twice as much
as diastolic pressure (three times as much in Ebert 1992 and Jeleazcov 2015).
Epinephrine's inotropic effect widens pulse pressure more than its effect on
stroke volume alone would, without changing MAP or flow. In human inotrope
studies, pulse pressure rose with contractility even when stroke volume did not
(Piccioli 2023). Catheter and transducer dynamics filter the arterial pressure
measurements.

Pulse oximetry responds to rhythm and perfusion. Poor perfusion delays readings
and reduces the plethysmograph amplitude.

Hyperkalemia peaks the T waves, flattens the P waves, widens the QRS complex,
and progresses to asystole.

### Monitoring during arrest

When pulses stop, arterial pressure falls to the mean systemic filling pressure.
The numerics hold briefly. Systolic and diastolic readings then clear, and MAP
follows the catheter pressure. During chest compressions, arterial pressure and
plethysmography follow the compressions, and the ECG shows compression artifact.

Noninvasive blood pressure (NIBP) raises its cuff inflation pressure when
systolic pressure is high. It fails during VF, asystole, PEA, profound
hypotension, or chest compressions. A failed measurement blanks the displayed
values and raises a low-MAP alarm.

### Airway pressure, flow, and gases

Spirometry measures ventilator, bag, and unassisted breaths. Expired tidal
volume comes from completed breaths, and respiratory rate and minute ventilation
are averaged over recent breaths. Pressure-volume and flow-volume loops use
synchronized pressure, flow, and volume samples. Plateau pressure requires
negligible flow and relaxed respiratory muscles. Displayed PEEP is the airway
pressure, while intrinsic PEEP affects plateau pressure and residual flow.

[Capnography](../anasim/monitors/capno.py) includes dead-space transit time and
analyzer delay. End-tidal values clear when valid exhaled gas stops, and gas
that stays at the sampling port decays when flow stops. End-tidal CO₂ alarms
start only after the first breath with the airway connected. Gas lost around a
facemask does not pass the sampling port, so a large leak, especially with small
breaths, leaves dead-space gas at the port and flattens the capnogram.

Without an airway device, the impedance respiratory rate counts breathing
efforts, including during obstruction. The gas monitor displays end-tidal
age-adjusted MAC, while anesthetic effects use brain MAC.

### BIS and neuromuscular monitoring

BIS includes Eleveld's age-dependent drug delay and display smoothing. Cortical
failure from cerebral ischemia appears after a delay of at most 20 seconds.
High-BIS alarms apply only when the patient is paralyzed or under active
anesthetic maintenance.

The [neuromuscular monitor](../anasim/monitors/neuromuscular.py) measures TOF or
single twitch on demand or at a set interval. TOF reports the twitch count and,
when all four twitches are present, the T4/T1 ratio. Ratios from weak twitches
are flagged. Single twitch reports T1 relative to an unblocked baseline.

### Arterial blood gases

An [arterial blood gas](../anasim/monitors/blood_gas.py) sample records the
arterial values at the time of the draw, and the result appears 2 minutes later.
As in a blood gas analyzer, pH, PaCO₂, and PaO₂ are reported at 37 °C, so PaO₂
in a hypothermic patient is higher than the value in the body. The report also
includes bicarbonate, standard base excess, lactate, sodium, potassium,
chloride, ionized calcium, SaO₂, hemoglobin, and FiO₂. Glucose is not modeled.
