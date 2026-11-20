# AnaSim models

This guide describes how AnaSim models the patient, drugs, physiology,
anesthesia machine, monitors, and clinical cases, and which studies each model
is based on.
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
ejection fraction (HFrEF), stroke volume is more sensitive to afterload, does
not rise with fluid loading, and falls less when induction reduces preload,
consistent with low-output cardiac surgical patients in Sherry 1995. With
preserved ejection fraction (HFpEF), the ventricle is stiffer and fills less
during tachycardia. Chronic left atrial hypertension increases lymphatic
drainage, so pulmonary edema starts at a higher filling pressure. The reserve
above the resting pressure is still smaller than in a normal heart.

COPD combines persistent obstruction, hyperinflation, gas trapping, dead space,
and impaired gas exchange, with a normal resting PaCO₂. Acute bronchospasm adds
to these changes. Obesity reduces supine functional residual capacity (FRC) and
respiratory compliance and increases airway resistance (Pelosi 1998).

Pregnancy changes maternal blood volume, extracellular water, cardiac output,
renal filtration, FRC, heart rate, metabolism, and resting PaCO₂ according to
gestational age (Dallmann 2017; Contreras 1991), with late-pregnancy
interstitial water calibrated to cesarean calcium dilution (Ansari 2025). In
later pregnancy, the supine position reduces venous return and the left lateral
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
| Propofol | Eleveld 2018 arterial model with its opioid-regimen covariate. Cirrhosis enlarges its volumes (Servin 1990) |
| Remifentanil | Eleveld 2017 |
| Fentanyl | Bae 2020 up to 70 kg; at higher weights it scales with pharmacokinetic mass (Shibutani 2004). Clearance scales with hepatic function |
| Morphine and hydromorphone | Meissner 2025, scaled with fat-free mass, age, and hepatic and renal function |
| Midazolam | Albrecht 1999, with age-dependent distribution and effect-site equilibration. Clearance scales with hepatic function |
| Etomidate | Arden 1986, with age-dependent central volume and clearance; clearance scales with hepatic function (Van Hamme 1978) |
| Ketamine | Kamp 2020, with hepatic clearance scaling and separate cardiovascular and heart-rate effect sites (Kamp 2021; Bovill and Dundee 1972) |
| Lidocaine | Foong 2025. Cirrhosis enlarges its volumes and lowers hepatic clearance (Thomson 1973) |
| Amiodarone | Korth-Bradley 1996, with hepatic clearance scaling and separate effect sites for bolus vasodilation, contractility and rhythm conversion, and sinus slowing |
| Rocuronium | Kleijn 2011, scaled with adjusted weight toward ideal body weight in obesity (Meyhoff 2009). Cirrhosis enlarges its volumes (Khuenl-Brady and Sparr 1996; Magorian 1995), and hepatic clearance scales with liver function as a model assumption |

Kinetic models for succinylcholine, dantrolene, and cardiovascular drugs are
described in their sections below.

Inhaled agents move between lung, blood, and tissue compartments. Lean tissue
volumes scale with fat-free mass, and the fat volume grows with excess weight.
The effects of inhaled agents depend on brain partial pressure, expressed as
age-adjusted minimum alveolar concentration (MAC) (Davis and Mapleson 1981;
Yasuda 1991; Mapleson 1996; Nickalls and Mapleson 2003). Uptake of nitrous oxide
concentrates the other gases in the alveoli, and its washout dilutes them.

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
it, and endogenous norepinephrine spillover falls with flow and stops during
arrest. Esterase metabolism of remifentanil, succinylcholine, and esmolol
continues without flow, and sugammadex still binds rocuronium in plasma.

## Target-controlled infusion

Target-controlled infusion (TCI) adjusts infusion rates within pump limits to
reach a predicted plasma or effect-site target. Propofol, remifentanil, and
fentanyl support plasma or effect-site targeting, whereas rocuronium and
vasoactive infusions target plasma concentration. For propofol, remifentanil,
fentanyl, and rocuronium, TCI uses the published model with only age, weight,
height, and sex, as clinical pumps do. That prediction omits sampled
variability, organ impairment, blood volume, cardiac output, midazolam, and
sugammadex, so the patient's concentrations can differ from the target. After
blood loss, for example, propofol and remifentanil concentrations rise above it.
TCI for vasoactive drugs predicts from the patient's current PK.

Population pump models omit hand boluses, reset to zero drug on board when
restarted after a manual infusion, and resume from their own history after a
stop. Changing a target preserves the active plasma or effect-site mode, and
setting a manual infusion rate turns off TCI for that drug. When maintenance is
initialized, `tci_enabled` selects either a 30-minute TCI history or manual
infusion rates.

Decrement times for propofol, remifentanil, and fentanyl use the pump model. The
decrement time for rocuronium uses the patient's PK and neuromuscular model,
including sugammadex and inhaled agents. For norepinephrine, it is the time for
the concentration to fall halfway to its endogenous baseline.

## Anesthetic drug effects

Separate effect sites determine BIS, clinical responsiveness, and breathing.

| Endpoint | Model |
|----------|-------|
| Baseline BIS | Eleveld 2018 propofol response with age-dependent potency and processing delay, plus a calibrated sevoflurane contribution |
| Clinical responsiveness | Bouillon 2004 response curves for shaking or shouting and laryngoscopy, with the age effect from Schnider 1999 and a calibrated scaling of propofol for loss of response |
| Ventilatory depression | Bouillon 2004 propofol response, with opioid effects and CO₂ control from Olofsen 2010 |

`loc` is the probability of no response to shaking or shouting, and `tol` is the
probability of no response to laryngoscopy. Sevoflurane thresholds come from
Kuizenga 2019 and Hannivoort 2016. The propofol concentration for loss of
response is scaled so that patients become unresponsive at the BIS values seen
in stepped inductions, which are higher in older patients (Lysakowski 2009).

For the clinical response curves, fentanyl, morphine, and hydromorphone are
converted to remifentanil equivalents, and etomidate and midazolam to propofol
equivalents. Potency depends on the patient and on the effect. Midazolam alone
lowers BIS only to a calibrated sedation floor and preserves spontaneous
breathing with a moderate rise in PaCO₂ (Marty 1986), while acting
synergistically with propofol (Short 1992). Nitrous oxide and ketamine also
affect responsiveness. Ketamine adds to opioid analgesia and, like lidocaine,
reduces responses to stimulation. During propofol or sevoflurane anesthesia,
ketamine raises BIS because it increases high-frequency EEG activity (Hans
2005).

Lidocaine reduces the anesthetic concentration needed to tolerate stimulation
(Himes 1977; Hans 2010), although its effect alone is capped below surgical
unresponsiveness.

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

The effects of fentanyl on the pressure and HR responses are fitted separately
to the equilibrated concentrations and intubation responses in Kazama 1997. With
inhaled agents, fentanyl and remifentanil also lower the concentration needed to
block the response and blunt the HR rise. This agrees with opioid reductions in
volatile MAC and MAC-BAR (Katoh 1999; Lang 1996). Other opioids are converted to
remifentanil equivalents.

Peak HR and MAP responses to intubation follow Ebert 1992 and 1995. Billard
1994 and Sawano 2013 set the HR recovery and the split of the intubation
response between stroke volume and vascular resistance, and Hayakawa-Fujii 2001
is an independent check.

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
rocuronium and sugammadex. Sevoflurane strengthens rocuronium block (Kleijn
2011), and nitrous oxide adds a calibrated potentiation based on vecuronium data
from Fiset 1991; AnaSim applies both at peripheral and central muscles.
Sugammadex binds free plasma rocuronium and speeds its removal from the effect
site, and the complex is cleared at the rate of sugammadex. Both drugs use
measured creatinine clearance when it is supplied and otherwise an estimate from
age and renal function. Other renal effects use the renal function factor.

Train-of-four (TOF) represents the adductor pollicis. The diaphragm and larynx
equilibrate faster and require higher rocuronium concentrations (Cantineau 1994;
Plaud 1995; Dhonneur 1999), so breathing recovers before TOF (Eikermann 2003).
Residual block reduces pharyngeal reserve and makes the airway more likely to
collapse under hypnotics (Herbstreit 2009). Awake muscle tone and positive
airway pressure keep the airway open.

Succinylcholine uses Roy 2002 PK. Twitch onset and recovery are calibrated to
Kopman 2003, and central muscle sensitivity to Dhonneur 1999. Its phase I block
lowers single-twitch height without TOF fade and releases potassium from muscle
(Martyn and Richtsfeld 2006).

## Hemodynamics

The [hemodynamic model](../anasim/physiology/hemodynamics.py) is based on the
model of Su 2023 for the effects of propofol and remifentanil on vascular
resistance, stroke volume, and HR. It is linked to blood volume, the pulmonary
circulation, reflexes, vasoactive drugs, and distributive shock.

Drugs act through hemodynamic effect sites on a slowly adapting tone. This tone
uses the turnover rate from Su 2023 and also responds to MAP feedback,
chemoreflexes, blood loss, and temperature. As a result, pressure falls within
minutes of induction, whereas compensation is slow.

### Propofol

The effect of propofol on pressure equilibrates over about 10 minutes,
consistent with the volunteers in Jeleazcov 2015 and the patients aged 60 to 69
in Kazama 1999. Above 65, the effect develops more slowly and becomes larger, as
Kazama found at 70 to 85 years. By age 78, the half-time is about twice as long.

Compared with the model of Su 2023:

- The effect on vascular resistance is smaller, as Fregolent 2025 also found.
  Early after induction, propofol alone therefore lowers pressure about half as
  much as propofol with an opioid (Ebert 1992 and 1995; Billard 1994).
- The effect on stroke volume increases less with age, consistent with Kazama
  1999, which found similar sensitivity from 20 to 69 years.

The effects of propofol on resistance and venous pooling increase with
pre-induction MAP, so patients with higher starting pressures have larger
relative falls (Morley 2010).

Propofol also reduces stressed volume and the resistance to venous return (de
Wit 2016) through the same effect site as its arterial effects. In addition, the
loss of sympathetic venous tone moves up to 30% of stressed volume into
unstressed volume. This shift follows the plasma propofol concentration over a
few minutes, and it explains why induction lowers filling pressure and cardiac
output more than vascular resistance (Fairfield 1991; Zucker 2022; Sjøen 2024).
Blood loss draws on this pooled blood first, fluid loading does not refill it,
and the alpha-adrenergic effect of epinephrine reverses it.

### Opioids

Opioids equilibrate within about a minute and are expressed as remifentanil
equivalents. Fentanyl has its own effect site.

Opioid effects level off above 2.5 ng/mL, because Sjøen 2024 found the same
falls in pressure and HR at remifentanil doses that gave peak effect-site
concentrations of 2, 4, and 8 ng/mL. The direct vasodilating effect of
remifentanil is a quarter of the volunteer estimate in Su 2023. This value lies
between Su 2023 and the patient model of Garraza-Obaldia 2024, which found no
direct effect. The effect of remifentanil on HR is half of the estimate in Su
2023, because those volunteers breathed spontaneously and were likely
hypercapnic.

The interactions between propofol and opioids on resistance, HR, and stroke
volume were fitted to VitalDB propofol-remifentanil inductions, which represent
routine practice. The [validation
guide](../validation/vitaldb/README.md#hemodynamics) gives the recorded and
modeled falls in MAP. Induction trials that used higher doses (Sawano 2013;
Sjøen 2023 and 2024; Poterman 2017) reported larger falls, and the model is
within two standard deviations of their results. The stroke volume interaction
rises over a wider range of propofol concentrations than in Su 2023, which makes
it larger at the doses used in those trials.

### Sevoflurane and adjuncts

Sevoflurane lowers stroke volume, vascular resistance, and HR. The fall in
pressure was fitted to routine bolus inductions in VitalDB, in which the falls
did not depend on the opioid dose. Most of the fall comes from stroke volume. In
unstimulated volunteers on controlled ventilation, cardiac index fell 29% at 1
and 1.5 MAC, while resistance fell 3% and 15% and decreased significantly only
at 2 MAC (Malan 1995). Volunteers in Ebert 1995 had larger falls in pressure and
no change in HR, and the model is within two standard deviations of their
pressures.

The sevoflurane effect equilibrates with a 14-minute half-time. This matches the
gradual fall in MAP over 10 minutes of sevoflurane by mask after propofol in
Ebert 1995. The effects of propofol and sevoflurane on resistance and stroke
volume combine as independent fractions.

Midazolam causes mild vasodilation and venous pooling based on forearm
vasodilation in Marty 1986, and combining it with opioids lowers stroke volume
and pressure further (Heikkilä 1984). Etomidate changes vascular resistance and
stroke volume very little. Nitrous oxide causes mild sympathetic
vasoconstriction and slightly depresses contractility.

### Reflexes

MAP feedback changes HR or vascular resistance, depending on rhythm, anesthetic
depth, blood gases, and stimulation. When muscarinic block or a fixed rhythm
prevents slowing, high MAP lowers endogenous vascular resistance (Weissler 1957;
Tweed 1972). Vascular compensation for a rate-limited rhythm stops during
pulseless rhythms, electrical silence, chest compressions, and anoxic failure.
Anesthesia reduces the baroreflex and the tachycardia of hemorrhage (Ebert 1992;
Sellgren 1994). Older patients have lower cardiovagal gain and less reflex
tachycardia, while baroreflex control of vascular resistance is preserved
(Kardos 2001; Laitinen 2004; Monahan 2007). Severe hypoxia lowers HR and
contractility and blunts reflex tachycardia.

Under propofol, MAP has little effect on HR (Garraza-Obaldia 2024). Slow
compensatory tachycardia therefore decreases with anesthetic depth, while
slowing in response to high pressure remains, calibrated to Sawano 2013 and
Takahashi 2002. Opioids also suppress reflex tachycardia without suppressing
reflex slowing, so HR falls along with pressure after propofol-remifentanil
induction (Sjøen 2024). Remifentanil also causes a vagal bradycardia that does
not occur with fentanyl and that antimuscarinics block (Poterman 2017). Its size
matches the fall in HR after induction in VitalDB TIVA cases. Septic tachycardia
depends directly on sepsis severity and persists under anesthesia.

### Preload and heart rate

Blood loss lowers filling pressure and preload. Reflex venoconstriction and
transcapillary refill from the interstitium partly compensate. Venous
recruitment increases for losses above 30%, so slow class III and IV hemorrhage
remains compensated until stressed volume is exhausted. Venous compliance scales
with baseline blood volume, so resting filling pressure is consistent across
body sizes. High intrathoracic pressure from positive end-expiratory pressure
(PEEP) or a closed adjustable pressure-limiting (APL) valve raises right atrial
pressure and impedes venous return, and venous return stops when intrathoracic
pressure exceeds mean systemic filling pressure.

In untreated atrial fibrillation, atrioventricular conduction limits the
ventricular rate at high sympathetic drive, and stroke volume falls further at
rapid rates. In normal ventricles, stroke volume falls when arterial resistance
rises more than contractility. This follows ventriculo-arterial coupling with a
resting ratio of arterial to ventricular elastance of 0.7 (Asanoi 1989; Chen
1998). Venous return limits stroke volume above 130 bpm and during combined
inotropic and vasoconstrictor stimulation, and short diastolic filling reduces
it further above 170 bpm.

Severe left ventricular failure, whether chronic or from acute reversible
depression, shifts blood into the pulmonary circulation. Left atrial pressure,
lung water, and right ventricular afterload then rise until the outputs of the
two ventricles match. Without circulation, left atrial pressure equilibrates
with mean systemic filling pressure, and fluid stops filtering into the lungs.

### Acidemia, electrolytes, and hypothermia

Below an in-vivo pH of 7.2, contractility and the response to adrenergic drugs
fall linearly to 30% at pH 6.8. This is extrapolated from isolated failing
human myocardium (Schotola 2012).

Hypocalcemia lowers contractility and vascular resistance. Intravenous calcium
restores contractility without raising systemic vascular resistance above
baseline (Denlinger 1975; Lang 1988). Hyperkalemia slows conduction, depresses
contractility, and can trigger ventricular arrhythmias or asystole. Elevated
ionized calcium raises the potassium threshold for these effects. Prolonged
electrical silence from hyperkalemia or local anesthetic toxicity transitions
to asystole after 15 seconds.

Core temperatures below 33 °C slow HR and weaken cold-induced vasoconstriction
and reflex tachycardia. Temperatures below 30 °C depress contractility and
increase the risk of ventricular fibrillation.

## Cardiovascular drugs

Vasoactive drugs change vascular resistance, stroke volume, and HR. Adrenergic
and muscarinic antagonists modify both drug responses and autonomic reflexes.
Anaphylaxis reduces the response to adrenergic agonists.

### Norepinephrine

The norepinephrine model includes endogenous secretion and a clearance that
falls with plasma propofol up to 8 mcg/mL (Li 2024). Only concentrations above
the resting endogenous level change HR, resistance, venous capacitance, and
stroke volume, and the effects on resistance and stroke volume increase with
anesthetic depth. Norepinephrine recruits unstressed venous blood, which raises
filling pressure and preload (Persichini 2012; Hamzaoui 2010), and supports
stroke volume in reduced-EF septic shock (Hamzaoui 2010; Carelli 2026). Septic
shock halves vascular and cardiac responses to catecholamines (Bellissant and
Annane 2000; Schotola 2012; Hernández 2013).

### Epinephrine

Epinephrine uses arterial clearance (Ensinger 1992), which falls with age (Esler
1995). Its effective mixing volume is calibrated to bolus responses. This volume
sets the initial concentration after a bolus and, with clearance, the rate at
which plasma concentration falls. Cardiac and vascular responses are calibrated
to boluses in awake and anesthetized patients and to infusions in awake
volunteers (Guinard 1995; Takahashi 2002; Freyschuss 1986; Stratton 1985;
Ensinger 1992; Marker 1998; Leenen 2005).

Older patients are less sensitive to epinephrine (Leenen 2005; Pan 1986;
Hogikyan and Supiano 1994), so the arterial alpha-adrenergic C50 rises with age.
During infusions, higher plasma concentrations and weaker baroreflex buffering
offset this lower sensitivity. Boluses raise HR less in older patients.

Under propofol, epinephrine recruits pooled venous blood at lower concentrations
than those that constrict arteries. A second alpha-adrenergic component acts at
higher concentrations, with a gradual concentration response and its own delay.
It maintains vascular resistance at the infusion rates used in shock and
supports pressure during CPR. The concentration-response calibration covers
concentrations up to the highest infusion step in Ensinger 1992,
0.22 mcg/kg/min.

### Phenylephrine and ephedrine

Phenylephrine uses two-compartment PK from US Food and Drug Administration 2012,
with bolus responses calibrated to Meng 2011. It recruits venous blood and
raises afterload, so its effect on cardiac output depends on preload (Rebet
2016).

Ephedrine uses the oral disposition and tolerance reported by Persky 2004,
applied to intravenous use. Its peak responses are calibrated to Meng 2011, and
its effect on pressure during induction was checked against Sjøen 2026. Alpha
and beta blockade shift its vascular and cardiac concentration-response curves
separately. With repeated doses, presynaptic tachyphylaxis builds from ephedrine
exposure regardless of receptor blockade. It reduces the pressor and peak HR
responses, while a smaller HR response remains and saturates at high
concentrations.

### Other vasoactive drugs

Vasopressin uses label PK (DailyMed 2025) and raises vascular resistance with
mild HR slowing through V1 receptors. Because it acts outside adrenoceptors, its
response is preserved during adrenoceptor blockade, acidemia, anaphylaxis, and
septic shock.

Dobutamine (Kates and Leier 1978) and milrinone (DailyMed 2026, with mostly
renal clearance) increase stroke volume and HR and lower vascular resistance
(Daly 1997; Magnani 1977; Baim 1983). Dobutamine acts on beta-1 and beta-2
receptors, so its effects are reduced by beta blockade, acidemia, anaphylaxis,
and septic shock (Hernández 2013). Milrinone acts outside adrenoceptors and
retains its effect in those states.

Esmolol uses blood-esterase PK and selective beta-1 blockade from Sum 1983
(Wiest and Haney 2012), so its clearance does not depend on cardiac output or
organ function. Labetalol has age-dependent hepatic clearance (Abernethy 1987;
Hafsa 2022) and blocks beta-1, beta-2, and alpha-1 receptors, with about
sevenfold higher intravenous potency at beta than alpha receptors (Hafsa 2022).
Competitive antagonists divide effective agonist concentrations by
`1 + sum(C / Kb)`, so higher agonist doses overcome the block. Receptor
occupancy reduces resting sympathetic tone (less as anesthetic depth increases),
reflex, hemorrhage, and stimulation responses, and the ventricular rate in
atrial fibrillation.

### Antimuscarinics

Atropine links its HR response to its peripheral compartment (Hinderling 1985),
with renal scaling on the fraction excreted unchanged. Volatile and opioid
anesthesia leave the HR rise intact (Yamaguchi 1988), whereas propofol lowers
the peak HR increase without reducing vagal block (Horiguchi and Nishikawa
2002). Glycopyrrolate uses three-compartment population PK from Du 2025, with
70% renal clearance and vagolytic potency from Ali-Melkkilä 1993, and produces a
smaller peak HR rise than atropine.

Both antimuscarinics block resting vagal tone, baroreflex slowing, and
remifentanil bradycardia, and both relieve sinus bradycardia alongside
beta-adrenergic agonists and milrinone. During unstimulated vagolytic
tachycardia, venous unloading lowers stroke volume over several minutes, and the
fall occurs earlier when preload is low (Weissler 1957; Berry 1959). Adrenergic
drive restores stroke volume during pacing (Ross 1965).

### Ketamine

Ketamine raises HR, stroke volume, and vascular resistance through sympathetic
stimulation (Tweed 1972). Its HR effect starts sooner than the others and adds a
fixed increase, including after atropine (Bovill and Dundee 1972). Baroreflex
gain falls at anesthetic concentrations, based on rat data (Akine 2001). A
calibrated offset of up to 20 mmHg reduces reflex slowing and vasodilation to
fit pressure rises after atropine and during pacing. Beta blockade leaves a
pressor response, and alpha blockade reduces it, modeled on droperidol
pretreatment (Bovill and Dundee 1972; Bålfors 1983).

Sympathetic reserve is exhausted by severe blood loss, septic shock, profound
hypotension, severe acidemia, or myocardial energy failure. Ketamine then loses
its sympathetic effect and lowers contractility and vascular resistance directly
(Waxman 1980).

### Amiodarone

Rapid boluses of intravenous amiodarone briefly lower vascular resistance and
blood pressure, mostly through the polysorbate 80 solvent. Slower infusions
lower pressure much less (Munoz 1988). Contractility falls over several minutes,
reducing cardiac output in HFrEF (Kosinski 1984) and leaving a small 15- to
30-minute fall after vasodilation wears off in a normal left ventricle (Munoz
1988; acute output rose in the four normal-EF patients in Kosinski 1984). Sinus
rate slows over 15 to 30 minutes (Munoz 1988), and sympathetic HR reflexes
weaken over the same period.

## Heart and brain oxygen supply

Myocardial oxygen demand is based on the rate-pressure product, with a basal
demand that remains as mechanical work falls (Harasawa 1992). Fibrillating
myocardium uses most of the resting demand of the beating heart. Supply depends
on diastolic pressure above the coronary zero-flow pressure (Nanto 2001),
diastolic time, coronary flow reserve, and arterial oxygen content. A sustained
mismatch between supply and demand causes reversible contractile depression and
ST depression.

Cerebral oxygen supply falls with MAP below 50 mmHg and with arterial oxygen
content. Hypoxia increases normocapnic cerebral blood flow by up to half as a
model calibration, and acute hypercapnia augments hypoxic vasodilation
(Battisti-Charbonney 2011) up to twice baseline flow, preserving oxygen
delivery during hypercapnic hypoxemia (Willie 2015). Acute hypocapnia constricts
cerebral vessels, reducing reserve during hypoxemia or anemia without causing
ischemic failure in a normoxic brain (Brian 1998). Cerebrospinal fluid
bicarbonate adapts with a 6-hour time constant, consistent with blood-flow
recovery over 4 to 12 hours of sustained PaCO₂ change. Severe hypotension makes
flow pressure-limited; restoring pressure raises supply up to the hypocapnic
flow ceiling (Brian 1998). Brain oxygen stores buffer a fall in supply for a few
seconds. Electrical function ramps down from 60% of normal supply to failure at
35% (Astrup 1981), so responsiveness, airway tone, and BIS fail (Nebout 2013),
and near-complete failure stops breathing. After circulation stops, this decline
begins at about 4 seconds and reaches a flat EEG at about 11 seconds (11 ± 2 s
in de Vries 1998). Function recovers within about two minutes after supply
returns.

Below 20% of normal, neurons are injured, and injury is slower when more supply
remains (Jones 1981). After the equivalent of 5 minutes without supply, the
patient remains comatose, although BIS and breathing can recover.

CO₂ narcosis reduces responsiveness and airway tone without anoxic injury. Its
severity depends on both PaCO₂ and pH (Westlake 1955; Sieker and Hickam 1956).
With acidemia, it rises as PaCO₂ increases from 60 to 100 mmHg and pH falls from
7.3 to 7.1. In compensated hypercapnia, it rises as PaCO₂ increases from 100 to
140 mmHg. At deeper hypercapnic coma, medullary respiratory drive is also
depressed toward shallow hypoventilation (Westlake 1955; Sieker and Hickam
1956). Prolonged coma lasting longer than an hour lags blood-gas correction
(Sieker and Hickam 1956); as teaching calibrations, onset uses a 30-second time
constant, and recovery slows from 30 seconds after brief narcosis up to 30
minutes after prolonged coma. CO₂ does not change BIS directly: BIS was
unchanged at EtCO₂ 3 to 12 kPa under propofol anesthesia (Turnbull 2008).

## Cardiac arrest and resuscitation

Arrest is confirmed when ventricular fibrillation (VF), pulseless ventricular
tachycardia (VT), asystole, HR below 10 bpm, or pulseless electrical activity
(PEA) lasts 15 seconds. In PEA, the ECG remains organized while the heart's own
beats produce less than 0.3 L/min with a pulse pressure below 5 mmHg. VT is
pulseless when pulse pressure is at most 3 mmHg, or below 5 mmHg with MAP below
25 mmHg. The simulation continues through arrest, so the learner can resuscitate
and treat the cause.

### Course of arrest

Severe myocardial ischemia or hypoxemia first depletes an energy reserve that
lasts about 4 minutes without circulation. If a perfusing rhythm returns within
this time, as after defibrillation in the electrical phase of arrest (Weisfeldt
and Becker 2002), circulation is restored.

Once the reserve is used up, the heart slows over several minutes. P waves
disappear (junctional rhythm), and complexes widen and shrink (idioventricular,
then agonal rhythm) until the ECG is flat. Untreated VF becomes fine and then
asystole. Meanwhile, injury builds up that reperfusion does not reverse within a
session, and it reduces contractility (Laurent 2002). The session ends when this
injury is complete, about 11 minutes after circulation stops.

### Chest compressions

[Chest compressions](../anasim/physiology/resuscitation.py) at 110/min pump a
quarter of resting cardiac output at normal filling and vascular resistance,
about a fifth under anesthesia, and little after major blood loss. This flow and
vascular resistance set arterial pressure between compressions. Each compression
adds a pressure pulse that does not produce flow.

Coronary flow occurs between compressions, against venous and left atrial
back-pressure plus myocardial closing pressure. The energy reserve recovers when
diastolic pressure exceeds about 25 mmHg. This is the CPR target in Meaney 2013,
and it corresponds to the coronary perfusion pressure needed for return of
circulation in Paradis 1990. In an anesthetized adult, compressions give about
70/25 mmHg and EtCO₂ near 15 to 20 mmHg, so CPR keeps VF coarse and makes fine
VF coarse again.

Epinephrine raises diastolic and coronary perfusion pressure during compressions
(Kim 2025). The higher afterload reduces forward flow and lowers EtCO₂ by a few
mmHg at constant ventilation (Cantineau 1994).

After withdrawal of life-sustaining measures, cardiac activity sometimes
resumed transiently without intervention; no patient with resumed activity
regained consciousness or survived (Dhanani 2021). AnaSim requires compressions
for recovery from primary asystole, a teaching calibration. Beats resume after
30 seconds of adequate coronary oxygen supply during compressions if energy
remains, the myocardium is viable, and neither hyperkalemia nor local anesthetic
toxicity blocks conduction. An interruption or inadequate perfusion restarts
this interval.

### Defibrillation and cardioversion

A first 200 J biphasic shock terminates VF about 9 times in 10 (Stiell 2007;
Koster 2008). AnaSim uses this rate for coarse VF and pulseless VT, with lower
calibrated rates for fine VF or hypoxic arrest. Each shock also causes mild
contractile stunning that reduces stroke volume by about 10% and resolves over
several minutes of coronary perfusion, calibrated from internal defibrillator
shocks in Tokano 1998.

After brief VF, conversion is directly to sinus rhythm, and arterial pressure
recovers over 5 to 20 seconds as beats refill the arterial tree and reversible
oxygen debt clears. Recovery takes longer after longer VF (Park 1994). After
prolonged arrest, terminating shocks often leave transient post-shock asystole
before pacemaker escape or refibrillation (Pierce 2015), and an organized rhythm
beats only as well as the heart's energy reserve and accumulated injury allow. A
late shock without CPR therefore leaves PEA or persistent asystole, and
resuscitation after the reserve is exhausted leaves hypotension for the rest of
the session (Laurent 2002).

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

Low cardiac output slows lidocaine distribution and hepatic clearance, while
renal failure leaves parent-drug disposition unchanged (Stenson 1971; Thomson
1973). A 2 g infusion over an hour in HFrEF causes hypotension and coma in
sinus rhythm without arrest, matching the post-infusion concentration in
Armstrong 1988, though linear kinetics clear the drug faster than in that
patient, whose ejection fraction was 15%.

Midazolam, propofol, sevoflurane, and nitrous oxide suppress seizures and the
sympathetic rise in HR and pressure. Neuromuscular block hides convulsions
without stopping the seizure.

Lipid emulsion reduces lidocaine delivery to its effect sites by about a fifth,
calibrated to the 23% reduction in 30-minute unentrapped plasma exposure in
volunteers, which comes mainly from wider tissue distribution (Heinonen 2015).
It also lessens contractile depression, based on rat bupivacaine resuscitation
(Fettiplace 2014), with calibrated reductions in conduction block and
arrhythmogenic stress.

## Fluids

### Volume kinetics

Lactated Ringer's and 0.9% saline move between blood and the interstitium (Hahn
2017; Hahn 2024). Distribution slows when MAP is low, so fluid given during
hypotension after induction stays in the circulation longer (Ewaldsson and Hahn
2001; Hahn 2021). Pressure scaling of distribution and urinary elimination is
capped above 105 mmHg, and the urinary elimination rate has an upper bound.
Urine output depends on MAP, age, and renal function; under volatile or propofol
anesthesia, about 8% of a 25 mL/kg Ringer's load is excreted over 150 minutes
(7% and 11% group medians in Ewaldsson and Hahn 2005), with similar plasma
dilution and extravascular retention across anesthetics.
Saline is excreted more slowly than lactated Ringer's: after 2 L over an hour,
healthy volunteers pass about half as much urine in 6 hours (Reid 2003). More of
a 5% albumin infusion stays in the circulation at first. The model retains a
quarter of infused lipid emulsion volume in the circulation.

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
a 3-week-old unit, at pH 6.5 and PCO₂ 98 mmHg with supernatant potassium
38 mmol/L and lactate 22.5 mmol/L (Rabcuka 2025), so rapid transfusion adds
acid, CO₂, and potassium. Lactate from Ringer's and packed cells, and citrate
from packed cells, return base as they are metabolized (Kramer 2003; Kahn 1979).
Cirrhosis lowers citrate clearance (Kramer 2003); scaling citrate and D-lactate
clearance with liver function and stopping it when oxygen supply fails are model
assumptions.

Potassium exchanges with a large cellular pool over minutes. Epinephrine moves
it into cells when circulation is present. Succinylcholine (Martyn and
Richtsfeld 2006), malignant hyperthermia, and severe acidemia below pH 7.15 move
it into plasma.

Ionized calcium depends on albumin, citrate, and 37 °C pH (Thode 1983; Bolan
2001), so dilution and rapid transfusion lower it without shifting during
hypothermia. When rapid transfusion slows or stops, ionized calcium rebounds
partially as citrate redistributes and clears (Kahn 1979), while unreplaced
dilutional loss remains. Injected calcium mixes through extracellular fluid and
clears in proportion to circulation, so a dose given during arrest waits until
flow returns. Shed blood removes calcium, including the fraction still mixing
after an injection. One gram of calcium chloride therefore raises ionized
calcium by about 0.4 mmol/L, and half of the rise is gone by 30 minutes (Ansari
2025). Calcium lost to dilution or bound by citrate is not replaced within a
session (Bolan 2001).

### Renal acid-base handling

The kidneys excrete the daily fixed acid load when urine output and renal
perfusion allow. Over about a day, they also move bicarbonate toward the chronic
compensation for the current PaCO₂: 0.48 mmol/L per mmHg in hypercapnia up to
70 mmHg (González 2018) and 0.4 mmol/L per mmHg in hypocapnia (Krapf 1991).
Chloride moves in the opposite direction to bicarbonate, while urinary sodium
follows extracellular strong-ion balance without antidiuretic hormone regulation
(Reid 2003; Philbin and Coggins 1978).

## Respiratory control

[Ventilatory control](../anasim/physiology/respiration.py) combines anesthetic
effects, CO₂ feedback, hypoxic drive, and muscle strength.

Propofol reduces tidal volume, and opioids mainly reduce respiratory rate.
Sevoflurane causes rapid shallow breathing at surgical concentrations (Doi and
Ikeda 1987) and progresses to apnea near 2 MAC. Ketamine shifts the
steady-state CO₂ response slightly (Bourke 1987), and rapid high-dose boluses
briefly depress central drive. Hypocapnia and neuromuscular block suppress respiratory
effort, and rising CO₂ can restore breathing during opioid administration. Under
anesthesia, the apneic threshold moves with the resting PaCO₂ that the drugs
produce (Hickey 1971).

For morphine and hydromorphone, the effect on respiratory rate is modeled
separately from the effect on ventilatory drive (Olofsen 2026). The controller
combines Olofsen's fractional opioid response with Bouillon's nonlinear response
to alveolar CO₂.

Hypoxic drive declines during sustained hypoxia. Low doses of propofol,
sevoflurane, and nitrous oxide reduce it, and surgical concentrations of
sevoflurane or propofol abolish it while CO₂ drive persists (Knill and Gelb
1978; Blouin 1993; Sarton 1996; Nieuwenhuijs 2000). Reported age effects vary
by method: García-Río 2007 and Poulin 1993 found a much steeper fall in hypoxic
than in moderate CO₂ sensitivity after 65, whereas Peterson 1981 found about 50%
reductions in both during rebreathing and Lhuissier 2012 found a rising
poikilocapnic hypoxic response in men. As a calibration, acute hypoxic
sensitivity holds through age 62 and falls to 40% from age 75, while CO₂
sensitivity falls much less. During airway obstruction, airflow is limited
while respiratory effort continues.

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
(Wagner 1977). Neither condition adds much shunt, so controlled or high-flow
oxygen corrects most of the hypoxemia while relieving hypoxic drive and
releasing CO₂ from hemoglobin (DeGaute 1981; Robinson 2000).

In unconscious patients, dependent airways also close at end-expiration, and
this closure increases after age 30. PEEP of about 6 cmH₂O, or more with
obesity, holds these airways open (Gunnarsson 1991; Hedenstierna and Edmark
2015). Older patients therefore oxygenate worse under anesthesia, especially
without PEEP. After upper abdominal surgery, FRC stays at or below its value
under anesthesia (Wahba 1975; Putensen-Himmer 1992), so this closure persists
after emergence. After lower abdominal surgery, about a third of it persists
(Lindberg 1992).

### Gas transport and stores

[Gas transport](../anasim/physiology/gas_transport.py) moves oxygen, CO₂, and
buffer base through pulmonary capillary, arterial, and mixed venous blood, with
shunt bypassing the lungs. Two tissue groups and an extracellular pool store
CO₂, with calibrated capacities that equilibrate over 30 to 60 minutes after
changes in ventilation or metabolism (Henneberg 1987; Taskar 1995; Wooten 2003;
Magor-Elliott 2021). Blood flow determines exchange with these stores, and
respiratory feedback responds to arterial PCO₂. Lung tissue accounts for 5% of
oxygen consumption and CO₂ production and exchanges directly with alveolar gas
(Loer 1997). Without pulmonary blood flow, ventilation therefore lowers alveolar
PCO₂ only to about 2 mmHg.

When the airway is closed or circuit gas is exhausted, gas that is absorbed is
not replaced. Lung gas volume shrinks, and absorption atelectasis follows.
Without positive-pressure ventilation, the resting lung volume used for
mechanics falls by the absorbed volume. Opening the airway refills the lungs
with gas from the circuit (Laviola 2020). The lung gas volume has a floor at
anatomic dead space.

During apnea with an open airway, circuit oxygen washes through anatomic dead
space slowly unless earlier breathing removed its nitrogen. The circuit and
lungs draw on one flow-limited supply of gas. Uptake of nitrous oxide and
sevoflurane adds convective inflow, which raises uptake of the other alveolar
gases. During washout, evolved gas first refills any volume lost to absorption,
then flows outward, carrying oxygen, CO₂, and nitrogen with it. Nitrous oxide
washout therefore dilutes alveolar oxygen and CO₂ when inspired oxygen is low.

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
elimination capacity. This hepatic portion scales with liver function and with
the fraction of oxygen demand that is met. The extrahepatic portion holds down
to 15% of oxygen demand and falls linearly to zero when oxygen delivery stops.

Each mmol of endogenous lactate that accumulates above the resting amount
removes 1 mmol of buffer base, and clearance of this lactate returns the base.
Lactated Ringer's solution raises the measured lactate without adding acid, and
metabolism of this lactate adds buffer base (Zitek 2018). Blood gas analyzers
measure only the L-isomer, half of the racemic lactate in Ringer's.

### Acid-base

The blood gas model solves respiratory, lactic, and fluid-related acid-base
disturbances at 37 °C, then shifts arterial pH, PaCO₂, alveolar CO₂, and EtCO₂
to the patient's core temperature. Ionized calcium and renal compensation use
the 37 °C values, while myocardial acidosis uses in-vivo pH. [Blood
buffering](../anasim/physiology/blood_chemistry.py) is based on Rees 2010, with
plasma albumin and phosphate charge from Figge 2018.
Oxygen affinity uses the temperature, pH, and PCO₂ terms of Dash 2016 at a
reference 2,3-DPG level. Gas exchange and changes in oxygen affinity conserve
CO₂, oxygen, and buffer base. Bleeding, transfusion, fluids, and urine change
the total buffer base by the amounts that they carry.

The interstitium buffers with its albumin and phosphate and with a fixed
cellular buffer using the plasma weak-acid parameters of Rees 2010 across twice
the blood volume (Wooten 2003 and 2010). With these buffers, acute changes in
PaCO₂ move pH and bicarbonate within the human ranges of Brackett 1965 and Arbus
1969, and standard base excess changes little. Standard base excess is
calculated from arterial pH and bicarbonate with the Van Slyke equation, as
blood gas analyzers do. It is near 0 mmol/L in nonpregnant adults and about
−2 mmol/L in late pregnancy.

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
of 5 cmH₂O stops further collapse but does not reopen the lung. Atelectasis
persists during emergence and early recovery (Benoît 2002; Östberg 2019). Once
the patient is awake and extubated, most of it clears over several hours,
fastest after nonabdominal surgery and slowest after upper abdominal surgery
(Strandberg 1986; Putensen-Himmer 1992). A smaller dependent fraction lasts
past 24 hours and resolves over days (Strandberg 1986; Lindberg 1992), and
morbid obesity prevents clearance over 24 hours (Eichenberger 2002).

## Temperature

Awake thermoregulation maintains the baseline core temperature. Basal heat
production scales with body weight to the 0.75 power. Temperature and sepsis
change metabolic demand and heat production, warming adds heat, and surface
cooling removes it. Heat production falls with oxygen delivery, so the core
cools faster during arrest.

General anesthetics inhibit active thermoregulation, lower metabolic heat
production, and redistribute heat from the core to the periphery. This heat
redistribution was measured under propofol-fentanyl in Matsukawa 1995 and is
extended to sevoflurane, etomidate, and ketamine as a model assumption.
Midazolam shifts thermoregulatory thresholds much less (Kurz 1995). Cooling from
redistribution in the first hour is smaller in heavier patients, who have higher
ratios of weight to surface area (Kurz 1995).

A cold patient shivers only after BIS has risen and responsiveness has
returned. Opioids, neuromuscular block, and core temperatures below 31 °C
reduce shivering.

## Malignant hyperthermia

[Malignant hyperthermia](../anasim/physiology/malignant_hyperthermia.py) (MH)
can start during sevoflurane anesthesia or after succinylcholine. In clinical
reports, onset after a volatile anesthetic varies from a median of 45 minutes
with sevoflurane alone into the second or third hour with other agents (Visoiu
2014), so the instructor or case sets the onset time. Skeletal muscle
hypermetabolism continues after the trigger is removed.

Untreated, oxygen consumption and CO₂ production approach five times baseline,
and anaerobic glycolysis and calcium cycling add extra heat. The gas exchange,
lactate, fluid, and temperature models then produce hypercapnia, lactic
acidosis, hyperkalemia, and a rapidly rising temperature. Tachycardia and
generalized rigidity increase with the crisis. Rigidity stiffens the chest wall,
limits spontaneous tidal volume, and persists under neuromuscular block.

Dantrolene PK follows the Ryanodex label. Dantrolene controls the crisis and
reduces twitch height without TOF fade, while leaving ventilation unchanged
(Flewellen 1983). Once the crisis is controlled and triggering agents have
cleared, it does not recur as dantrolene washes out. Re-exposure to sevoflurane
or succinylcholine can restart it.

## Anesthesia machine and ventilation

### Breathing circuit

The [circle system](../anasim/machine/circuit.py) mixes fresh gas with circuit
gas in a 6 L circuit that includes a 3 L reservoir bag and bellows. It accounts
for the exchange of oxygen, volatile agent, nitrous oxide, and nitrogen, and for
excess gas that spills from the circuit.

When patient uptake exceeds the inflow of fresh gas and vapor, the reservoir
empties. The circuit is sealed, so no room air enters. As the bellows collapses,
delivered breaths are capped by remaining gas plus fresh gas flow during
inspiration, and CPAP pressure and airway splinting drop to zero once the
bellows is empty without fresh gas flow.

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
during expiration. Flow triggering requires inspiratory muscle effort, so
passive PEEP changes do not trigger breaths. Tube obstruction raises airway
resistance. Inflations without measurable expired volume do not count as
breaths. Auto-PEEP falls as trapped gas escapes during expiration, including
during a pause. Switching out of mandatory modes or an apnea timeout clears
stale pressure and compliance readings. Switching out of PSV clears apnea
backup, and the apnea timeout starts a backup breath immediately.

At maintenance startup, initial ventilator settings aim for the patient's
resting PaCO₂. Tidal volume is adjusted first within limits based on predicted
body weight, followed by respiratory rate. Patients with COPD use a lower rate
limit to allow time for expiration; with severe obstruction or obesity, PaCO₂
can remain above the target.

### Bag and mask ventilation

Manual bag breaths stop at the APL setting (default 20 cmH₂O), so obstruction or
bronchospasm lowers the delivered volume. During unassisted breathing on the
circuit, the default setting adds no airway pressure. At any other setting,
excess fresh gas builds continuous airway pressure once the reservoir bag is
more than half full, capped by the APL setting or the mask leak threshold.

During mask ventilation, the mean airway pressure over recent breaths holds the
pharynx open. A loose mask loses part of each assisted breath, lowers this
pressure, and reduces the measured airway pressure, flow, and volume. Pressures
above 20 cmH₂O leak even with a tight seal (Bouvet 2014). A full seal delivers
the tidal volume of two-handed mask ventilation, and an 80% seal delivers that
of a one-handed hold (Joffe 2010). Room air enters around a loose mask in
proportion to the leak during unsupported breaths, or when assisted breaths
deliver less than the patient's own effort draws. PCV-VG raises inspiratory
pressure to make up for a mask leak.

## Monitors

Optional monitor audio follows plethysmograph pulses and active alarms. Alarm
silence lasts 2 minutes of real time and ends early if a new alarm sounds or
alarm priority changes between high and low.

### ECG, arterial pressure, and pulse oximetry

ECG, arterial pressure, and plethysmography use the same beat timing, and HR is
averaged over recent beat intervals. A long pause lowers the displayed HR and
clears it after 8 seconds without a beat.

The arterial waveform depends on MAP, stroke volume, and age-dependent arterial
compliance (Mahdi 2017; Franklin 1997). Compliance increases as pressure falls
(Langewouters 1984), so pulse pressure narrows during hypotension and widens
with vasopressors. Under propofol, systolic pressure falls about twice as much
as diastolic pressure (about three to four times as much in Ebert 1992 and
Jeleazcov 2015). The inotropic effect of epinephrine widens pulse pressure more
than its effect on stroke volume alone would, without changing MAP or flow. In
human inotrope studies, pulse pressure rose with contractility even when stroke
volume did not (Piccioli 2023). Catheter and transducer dynamics filter the
arterial pressure measurements.

At very low output, diastolic pressure stays at least as high as venous
pressure. Pulse pressure falls with stroke volume, and mean pressure approaches
this floor continuously.

Pulse oximetry responds to rhythm and perfusion. Poor perfusion delays readings
and reduces the plethysmograph amplitude, as does a narrow pulse pressure.

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
analyzer delay. End-tidal oxygen and anesthetic readings hold the last exhaled
sample during brief apnea and clear when valid exhaled gas stops, while gas at
the sampling port decays when flow stops. Once an airway is connected or
assisted ventilation is on, end-tidal CO₂ alarms start after the first breath,
or after the apnea timeout if no breath arrives. In pregnancy, the low EtCO₂
alarm limit falls with the expected PaCO₂. Gas lost around a facemask does not
pass the sampling port, so a large leak, especially with small breaths, leaves
dead-space gas at the port and flattens the capnogram.

Without an airway device, the impedance respiratory rate counts breathing
efforts, including during obstruction. During unassisted breathing, a low
respiratory rate alarm sounds after 15 seconds below 5 breaths/min. The gas
monitor displays end-tidal age-adjusted MAC, while anesthetic effects use brain
MAC.

### BIS and neuromuscular monitoring

BIS includes Eleveld's age-dependent drug delay and display smoothing. Sampled
hypnotic sensitivity affects BIS under both propofol and sevoflurane. Cortical
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
As in a blood gas analyzer, pH, PaCO₂, and PaO₂ are warmed anaerobically to
37 °C, so PaCO₂ and PaO₂ in a hypothermic patient are higher on the 37 °C report
than in vivo and pH is lower. The report also includes bicarbonate, standard
base excess, lactate, sodium, potassium, chloride, ionized calcium, SaO₂,
hemoglobin, and FiO₂. Glucose is not modeled.

## Cases and clinical actions

### Cases

Crises start 30 simulated seconds into a case. In their cases, anaphylaxis and
sepsis reach full severity over 45 and 60 seconds, while hemorrhage and
malignant hyperthermia develop over several minutes. In the anaphylaxis case,
crystalloid boluses run at 500 mL/min.

Guided requirements count actions from the start of the case, or from crisis
onset for crisis treatment, so the guide does not ask for a repeat dose. In
independent mode, step text is hidden, and safety warnings other than a failed
intubation appear only in the debrief.

Pressure objectives require spontaneous circulation without chest compressions,
and both the displayed MAP and the current model MAP must reach the target.
Without an arterial line, the cuff reading must be at most one minute old, and
a missing or failed reading does not satisfy a hypotension check. Anaphylaxis
recovery also requires improved bronchospasm.

Milestone outcomes, such as preoxygenation, stay credited once achieved.
Extubation outcomes are checked when the tube is removed, and the remaining
outcomes must hold when the case ends.

### Clinical action checks

Selecting the tracheal tube triggers a brief intubation stimulus whether or not
placement succeeds. When circulation continues and central muscle strength is
above 0.5, placement fails if loss of response is below 0.1, or if laryngoscopy
tolerance is below 0.1 without sustained hypnosis. Placement succeeds during
cardiac arrest or nonperfusing rhythms.

[`clinical_actions.py`](../anasim/core/clinical_actions.py) warns without
blocking the action when:

- the tube is placed in a responsive patient (loss of response below 0.1) whose
  central muscle strength is 0.5 or lower
- rocuronium or succinylcholine is given to a responsive patient without an
  induction dose already in plasma
- the tube is removed during apnea, reactive Stage II anesthesia, or incomplete
  neuromuscular recovery (TOF ratio or blocker T1 below 90%, excluding weakness
  from dantrolene)
- assisted ventilation stops during apnea
- a shock is delivered to a responsive patient (loss of response below 0.5)
  with spontaneous circulation
