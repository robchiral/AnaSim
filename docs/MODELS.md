# AnaSim models

This guide explains how AnaSim models the patient, drugs, physiology,
anesthesia machine, monitors, and clinical cases, and which published studies
each model is based on. The [architecture guide](ARCHITECTURE.md) covers code
structure and step order, and [Model references](REFERENCES.md) lists the cited
studies.

AnaSim connects published models that were each developed in their own study
populations, using calibrations to join them and to represent patient
conditions. Exact equations, thresholds, and coefficients are documented in the
code.

## Supported patient domain

AnaSim models adults aged 18 to 85 years within the [input
limits](CLI_USAGE.md#fields) for body size, hemoglobin, organ function, and
coexisting conditions.

### Patient conditions

Coronary artery disease lowers coronary flow reserve, which makes myocardial
oxygen delivery fall steeply when diastolic blood pressure drops.

Heart failure alters resting cardiac output, filling pressure, and sensitivity
to preload and afterload (Schwartzenberg 2012; Westermann 2008). In heart
failure with reduced ejection fraction (HFrEF), stroke volume is sensitive to
afterload, does not rise with fluid loading, and falls less when induction
reduces preload, consistent with findings in low-output cardiac surgical
patients (Sherry 1995). Vasodilation raises stroke volume substantially in HFrEF
(Schwartzenberg 2012), capped in the model around a third above baseline. In
heart failure with preserved ejection fraction (HFpEF), the stiffer ventricle
fills poorly during tachycardia (Westermann 2008) and gains little stroke volume
from arterial vasodilation (Schwartzenberg 2012). In both forms of heart
failure, inotropes and inodilators unload high left atrial pressure (Baim 1983).
Beta-receptor downregulation blunts catecholamine inotropy while preserving
calcium responsiveness in failing myocardium (Fowler 1986); AnaSim attenuates
both inotropic and chronotropic catecholamine reserve (consistent with modest
post-bypass catecholamine responses in Steen 1978) while keeping milrinone and
calcium effects intact. Chronic left atrial hypertension increases pulmonary
lymphatic drainage, so pulmonary edema begins at a higher filling pressure,
though the margin above resting pressure remains narrower than in a normal
heart.

COPD combines persistent airflow obstruction, hyperinflation, gas trapping,
increased parallel alveolar dead space, elevated baseline pulmonary vascular
resistance, and impaired gas exchange while keeping resting PaCO₂ normal.
Trapped gas in hyperinflated alveoli is modeled separately from the rapidly
exchangeable alveolar oxygen store, so severe COPD does not prolong apneic
oxygenation beyond that of a healthy lung. Acute bronchospasm adds on top of
baseline obstruction and dead space.

Obesity lowers supine functional residual capacity (FRC), minimum recruited lung
volume, and respiratory compliance, raises airway resistance and atelectatic
shunt, and increases the airway pressures required to keep alveoli open (Pelosi
1998; Hedenstierna and Edmark 2015).

Pregnancy alters maternal blood volume, extracellular water, cardiac output, and
renal filtration (Dallmann 2017), FRC and ventilatory drive (Contreras 1991),
and term static respiratory compliance (Marx 1970), along with heart rate,
metabolism, and resting PaCO₂ across gestational age. Late-pregnancy
interstitial water is calibrated against cesarean calcium dilution data (Ansari
2025). In late pregnancy, the supine position compresses venous return and the
left lateral position restores it (Humphries 2019). Pregnancy lowers the minimum
alveolar concentration (MAC) of inhaled anesthetics, scaled up to a 28%
reduction by 14 weeks from early-pregnancy (8 to 12 weeks) isoflurane data (Gin
and Chan 1994), and expands the central distribution volume of intravenous drugs
in proportion to maternal blood volume and extracellular water (Dallmann 2017).
Other drug responses use the standard adult calibrations.

### Patient variability

When `patient_variability` is off, every patient has the typical response for
their demographics and conditions. When it is on, each session draws a fixed
patient profile from the session seed across these sources of variability:

- propofol kinetics and bispectral index (BIS) sensitivity (Eleveld 2018), with
  standard deviations halved to match VitalDB inductions with total intravenous
  anesthesia (TIVA)
- remifentanil kinetics (Eleveld 2017) and fentanyl kinetics (Bae 2020)
- sevoflurane potency for awakening and noxious stimuli (Heyse 2012)
- the effects of propofol and opioids on vascular resistance (Su 2023), using
  the same spread for sevoflurane as for propofol
- the timing, sensitivity, and shape of the respiratory-rate response to
  morphine and hydromorphone (Olofsen 2026)

## Pharmacokinetics

Each intravenous drug uses a compartmental pharmacokinetic (PK) model with a
central compartment, up to two peripheral compartments, and effect sites for
its clinical actions.

| Drug | Model |
|------|-------|
| Propofol | Eleveld 2018 arterial model with its opioid covariate; cirrhosis enlarges distribution volumes (Servin 1990) |
| Remifentanil | Eleveld 2017, with deep peripheral volume capped toward lean mass in severe obesity (following Egan 1998) |
| Fentanyl | Bae 2020 up to 70 kg, scaled by pharmacokinetic mass at higher weights (Shibutani 2004); clearance scales with hepatic function |
| Morphine and hydromorphone | Meissner 2025, scaled with fat-free mass, age, and hepatic and renal function; renal impairment also accumulates active morphine-6-glucuronide (Osborne 1993) |
| Midazolam | Albrecht 1999, with age-dependent distribution and effect-site equilibration; clearance scales with hepatic function |
| Etomidate | Arden 1986, with age-dependent central volume and clearance; clearance scales with hepatic function (Van Hamme 1978) |
| Ketamine | Kamp 2020, with hepatic clearance scaling, a cardiac-output effect site (Kamp 2021), and a faster calibrated heart-rate effect site |
| Lidocaine | Foong 2025; cirrhosis enlarges volumes and lowers hepatic clearance (Thomson 1973) |
| Amiodarone | Korth-Bradley 1996, with hepatic clearance scaling and separate effect sites for bolus vasodilation, contractility and rhythm conversion, and sinus slowing |
| Rocuronium | Kleijn 2011, scaled toward ideal body weight in obesity (Meyhoff 2009); cirrhosis enlarges early distribution volumes (Khuenl-Brady and Sparr 1996; Magorian 1995; van Miert 1997), and hepatic clearance scales with liver function as a model assumption |

Models for succinylcholine, dantrolene, and cardiovascular drugs are covered in
their own sections below. In obesity, central volumes and clearances for most
intravenous drugs scale with lean body mass, while deep lipophilic peripheral
compartments and succinylcholine (Ingrande and Lemmens 2010), as well as
dantrolene, keep total-body-weight scaling.

Inhaled anesthetics move among lung, blood, and tissue compartments, with lean
tissue scaled to fat-free mass and fat volume growing with excess weight. When
cardiac output is low, blood flow shifts away from muscle and fat toward the
vessel-rich organs while cerebral autoregulation maintains brain uptake, so
hemorrhagic shock speeds up the rise in brain MAC. Clinical effects depend on
brain partial pressure, expressed as age- and pregnancy-adjusted MAC (Davis and
Mapleson 1981; Yasuda 1991; Mapleson 1996; Nickalls and Mapleson 2003; Gin and
Chan 1994). Hypothermia increases volatile blood-gas solubility (Lockwood 1997),
conserving dissolved tissue anesthetic mass as temperature changes and slowing
both uptake and washout. Nitrous oxide uptake concentrates other gases
in the alveoli, and its washout dilutes them.

When blood volume changes, the vascular share of each drug's central volume
scales accordingly. Shed blood removes drug at the current plasma
concentration, so blood loss alone does not dilute or concentrate plasma drug,
whereas fluid shifts conserve drug mass.

Clearance responds to organ function, circulation, temperature, and drug
interactions:

- Fentanyl, lidocaine, morphine, and hydromorphone clearances scale with cardiac
  output, with fentanyl and lidocaine anchored to anesthetized output (Bae 2020;
  Foong 2025). Low-extraction hepatic drugs (midazolam, dantrolene, and
  amiodarone) decline more gently as cardiac output falls.
- Renal perfusion scales only the fraction of renal clearance remaining after
  chronic renal impairment.
- Blood loss lowers propofol and remifentanil clearance based on swine
  hemorrhage studies (Johnson 2001, 2003); volume replacement restores propofol
  clearance (Johnson 2004), and AnaSim applies the same recovery to
  remifentanil. Severe low cardiac output without hypovolemia also reduces
  propofol clearance.
- Core hypothermia slows non-renal metabolic clearance and intercompartmental
  or effect-site transfer across intravenous drugs (Leslie 1995), with
  rocuronium slowing extrapolated from vecuronium (Caldwell 2000) and
  hypothermic cardiopulmonary bypass data (Smeulers 1995).
- Midazolam and propofol slow each other's clearance within clinically studied
  concentrations (Vuyk 2009; Lichtenbelt 2010).

During severe low-flow states and cardiac arrest, drug delivery to organs,
peripheral compartments, and effect sites falls with cardiac output. A drug
given during arrest therefore waits for chest compressions to circulate it, and
endogenous norepinephrine spillover stops without flow. Blood esterase
metabolism of remifentanil, succinylcholine, and esmolol continues without
flow, and sugammadex continues to bind rocuronium in plasma.

## Target-controlled infusion

Target-controlled infusion (TCI) varies infusion rates within pump limits to
reach a predicted plasma or effect-site target. Propofol, remifentanil, and
fentanyl support plasma or effect-site targeting, while rocuronium and
vasoactive infusions target plasma concentration.

As in clinical pumps, TCI for propofol, remifentanil, fentanyl, and rocuronium
uses only age, weight, height, and sex. The pump prediction does not account
for sampled patient variability, organ impairment, blood loss, low cardiac
output, midazolam interactions, or sugammadex, so actual patient concentrations
can drift away from the pump target. After major blood loss, for instance,
actual propofol and remifentanil concentrations rise above the displayed
target. Vasoactive TCI predicts from the patient's current PK state.

Population pump models ignore manual boluses, reset their drug history when
restarted after a manual infusion, and retain their own history across a stop.
Changing a target keeps the active plasma or effect-site mode, and setting a
manual rate turns off TCI for that drug. When a session starts in steady-state
maintenance, `tci_enabled` initializes either a 30-minute TCI pump history or
manual infusions.

Decrement times for propofol, remifentanil, and fentanyl come from the pump
model while TCI or pump history is active, and otherwise from the patient's
live PK model (including sessions with TCI disabled). Rocuronium decrement time
uses the patient's full PK and neuromuscular state, including sugammadex
back-flux and sevoflurane. Norepinephrine decrement time is the time to fall
halfway back to the endogenous baseline.

## Anesthetic drug effects

Separate effect sites govern BIS, clinical responsiveness, and ventilatory
drive.

| Endpoint | Model |
|----------|-------|
| Baseline BIS | Eleveld 2018 propofol response with age-dependent potency and delay, combined with a calibrated sevoflurane response and an age-independent monitor calculation lag (Pilge 2006) |
| Clinical responsiveness | Bouillon 2004a curves for shaking or shouting and for laryngoscopy, with the age effect from Schnider 1999 and calibrated propofol scaling |
| Ventilatory depression | Bouillon 2004b propofol response combined with opioid and CO₂ control from Olofsen 2010 |

Two probabilities track clinical depth: `loc` is the probability of no response
to shaking or shouting, and `tol` is the probability of no response to
laryngoscopy. Sevoflurane thresholds follow Kuizenga 2019 and Hannivoort 2016.
Propofol potency for loss of response is scaled so patients become unresponsive
at the BIS values reported in stepped inductions, which are higher in older
adults (Lysakowski 2009), and propofol tolerance of laryngoscopy also rises
gently with age.

For clinical responsiveness, other opioids convert to remifentanil equivalents,
and etomidate and midazolam convert to propofol equivalents, with potencies
adjusted for the patient and endpoint. Midazolam alone lowers BIS only to a
calibrated sedation floor and preserves spontaneous breathing with a modest rise
in PaCO₂ (Marty 1986), while acting synergistically with propofol (Short 1992).
Nitrous oxide and ketamine also depress responsiveness. Ketamine adds to opioid
analgesia and blunts responses to noxious stimuli, though during sevoflurane
(Hans 2005) or propofol anesthesia it raises BIS by increasing high-frequency
EEG activity.

Lidocaine lowers anesthetic requirements (shown for nitrous oxide in humans and
halothane in dogs in Himes 1977, and for propofol during stimulation in Hans
2010), though lidocaine alone does not produce surgical unresponsiveness.
Opioids well above analgesic concentrations add direct sedation and EEG slowing,
and moderate-to-severe core hypothermia directly depresses consciousness (Brown
2012), modeled across responsiveness, laryngoscopy tolerance, and BIS.

## Stimulation

BIS arousal during noxious stimulation depends on laryngoscopy tolerance,
whereas heart rate (HR) and mean arterial pressure (MAP) responses use
separate, stronger stimulus intensities (Albertin 2005). Laryngoscopy or
extubation under light anesthesia can trigger laryngospasm whenever brainstem
respiratory drive is intact. Laryngospasm does not happen during cardiac arrest
or anoxic brainstem failure, and it resolves slowly while the patient stays
lightly anesthetized without positive airway pressure.

Propofol lowers baseline blood pressure without blunting the cardiovascular
response to laryngoscopy, whereas opioids blunt that response (Kazama 1997).
This interaction uses propofol at the slower BIS effect site because it was
calibrated at steady concentrations; during the first few minutes after an
induction bolus, the faster clinical-response site reaches roughly twice that
concentration.

Fentanyl's effects on intubation pressure and HR are fitted separately to
Kazama 1997, and fentanyl blunts the pressor response alongside both propofol
and ketamine. With inhaled agents, fentanyl lowers the sevoflurane concentration
required to block movement and cardiovascular surges while nitrous oxide adds to
that effect (Katoh 1999; Daniel 1998), and remifentanil converts to fentanyl
equivalents using its isoflurane MAC reduction (Lang 1996). Smooth tracheal
extubation uses a lower airway-reflex intensity than laryngoscopy (Inomata 1999;
Lee 2009), and non-drug coma (cerebral ischemia or CO₂ narcosis) suppresses
airway-reflex reactivity alongside anesthetic drugs. Other opioids convert to
remifentanil equivalents.

Peak HR and MAP surges at intubation follow Ebert 1992 and 1995. Post-intubation
blood pressure and HR recovery follow Billard 1994 and Sawano 2013 (with the
split between stroke volume and vascular resistance taken from Sawano 2013),
checked against Hayakawa-Fujii 2001.

Ongoing surgery adds a persistent rise in vascular resistance that anesthetics
do not abolish, sized to the pre-incision to intraoperative MAP rise in VitalDB
TIVA cases and developing over roughly 5 minutes (matching MAP rises near
Katoh's MAC-BAR concentrations). Septic shock removes this surgical increase.
Nociception resets slow MAP feedback just as it resets the baroreflex, so
pressor responses persist while surgical stimulation continues.

In the browser interface, maintenance sessions and intraoperative crisis cases
start during surgery, whereas emergence cases start after surgery ends.

## Neuromuscular block

The [neuromuscular model](../anasim/patient/pd/nmba.py) uses Kleijn 2011 for
rocuronium and sugammadex. Sevoflurane strengthens rocuronium block and slows
effect-site washout during recovery while preserving bolus onset speed (Kleijn
2011; Lowry 1998), extending gently above 1 MAC, and nitrous oxide adds a
smaller potentiation based on vecuronium data (Fiset 1991); AnaSim applies both
at peripheral and central muscles. Hypothermia slows rocuronium clearance and
equilibration (extrapolated from vecuronium data in Caldwell 2000) and increases
sensitivity below 32.5 °C (calibrated from 25 to 28 °C cardiopulmonary bypass
data in Smeulers 1995). Sugammadex binds free rocuronium in plasma, accelerates
saturable washout from the effect site, and clears the bound complex at the rate
of sugammadex. A perijunctional rocuronium buffer reproduces transient rebound
weakness when a sub-therapeutic sugammadex dose clears unbound plasma drug
before peripheral stores empty (Eleveld 2007). Both drugs use measured
creatinine clearance when provided, or estimate it from age and renal function;
when creatinine clearance is supplied with default renal function, it also
scales renal function for other renally cleared drugs.

Train-of-four (TOF) reflects the adductor pollicis. The diaphragm and larynx are
more resistant to rocuronium and recover earlier than the adductor pollicis
(Cantineau 1994a; Plaud 1995; Dhonneur 1999), with faster laryngeal
equilibration modeled after Plaud 1995. Spontaneous breathing and vital capacity
therefore recover before full TOF recovery, while upper-airway inspiratory flow
remains impaired until TOF nears normal (Eikermann 2003). Residual block reduces
pharyngeal tone and makes upper-airway collapse more likely under hypnotics, and
severe weakness causes upper-airway collapse even in an awake patient
(Herbstreit 2009). Depolarizing and nondepolarizing blocks share the same
passive pharyngeal floor so passive apneic oxygenation remains possible. Awake
muscle tone and positive airway pressure otherwise keep the airway open.

Succinylcholine uses the PK model of Roy 2002, with twitch onset and recovery
calibrated to Kopman 2003 and central muscle sensitivity to Dhonneur 1999. Its
faster central effect site relieves laryngospasm rapidly at sub-intubating doses
(Chung and Rowbottom 1993). Phase I block reduces single-twitch height without
TOF fade and releases potassium from skeletal muscle (Martyn and Richtsfeld
2006), while high cumulative exposure produces Phase II block with TOF fade and
central muscle weakness (Lee 1975).

## Hemodynamics

The [hemodynamic model](../anasim/physiology/hemodynamics.py) builds on Su 2023
for the effects of propofol and remifentanil on vascular resistance, stroke
volume, and HR, coupled to blood volume, the pulmonary circulation, autonomic
reflexes, vasoactive drugs, and distributive shock.

Drugs act through hemodynamic effect sites on a slowly adapting vascular and
cardiac tone that uses the turnover rate from Su 2023 and also responds to MAP
feedback, chemoreflexes, blood loss, and temperature. Blood pressure therefore
falls within minutes of induction, while compensatory recovery is slow.

### Propofol

Propofol's arterial pressure effect equilibrates with a half-time around 6 to 10
minutes in adults up to age 69 (Kazama 1999; Jeleazcov 2015). Above age 65,
equilibration slows further and the pressure drop grows larger, matching
Kazama's data in patients aged 70 to 85.

Two adjustments distinguish this calibration from Su 2023:

- The direct drop in vascular resistance is smaller (consistent with Fregolent
  2025), so propofol alone lowers early post-induction blood pressure roughly
  half as much as propofol combined with an opioid (Ebert 1992, 1995; Billard
  1994).
- Sensitivity of stroke volume increases less steeply with age, reflecting
  similar responses across ages 20 to 69 in Kazama 1999.

Patients with higher pre-induction MAP show larger relative falls in arterial
pressure (Morley 2010), modeled through larger drops in vascular resistance and
venous tone.

Propofol also reduces stressed venous volume and resistance to venous return
(de Wit 2016). Loss of sympathetic venous tone shifts up to 30% of stressed
volume into unstressed capacitance over a few minutes, which explains why
induction lowers filling pressure and cardiac output more than systemic
vascular resistance (Zucker 2022; Sjøen 2024; delayed cardiac output decline in
Fairfield 1991). Acute blood loss draws on this pooled volume first, fluid
loading does not refill it, and alpha-adrenergic stimulation from epinephrine
reverses it.

### Opioids

Opioids equilibrate hemodynamically within about a minute and act as
remifentanil equivalents, with fentanyl using its own effect site.

Opioid hemodynamic effects plateau above 2.5 ng/mL remifentanil equivalent,
reflecting the similar drops in pressure and HR observed across 2, 4, and 8
ng/mL peak targets in Sjøen 2024. Given without a hypnotic, opioids have minimal
direct arterial pressure effect and cause modest central vagal slowing
(Garraza-Obaldia 2024), modeled as mild vasodilation alongside HR slowing.
Combined hypnotic depth from propofol, midazolam, or etomidate determines the
strength of the opioid interaction on resistance, HR, and stroke volume.

Propofol-opioid interactions were fitted to routine inductions in VitalDB, as
summarized in the [validation guide](../validation/vitaldb/README.md#hemodynamics).
Trials with higher induction doses reported larger pressure drops (Sawano 2013;
Sjøen 2023, 2024; Poterman 2017), and the model stays within two standard
deviations of those trials, with the stroke-volume interaction extending across
a wider propofol concentration range than in Su 2023.

### Sevoflurane and adjuncts

Sevoflurane lowers stroke volume, vascular resistance, and HR. Its overall
pressure drop was fitted to routine VitalDB inductions, where the fall did not
depend on opioid dose, and most of the fall comes from reduced stroke volume.
This matches unstimulated volunteer measurements in which cardiac index fell
sharply at 1 to 1.5 MAC while vascular resistance fell substantially only at 2
MAC (Malan 1995). Volunteers in Ebert 1995 showed larger pressure falls without
HR slowing, and the model stays within two standard deviations of their
pressures.

The sevoflurane hemodynamic effect equilibrates with a 14-minute half-time,
matching the gradual fall in MAP during mask sevoflurane after propofol in
Ebert 1995. Propofol and sevoflurane combine as independent fractions on
resistance and stroke volume.

Midazolam lowers blood pressure and sympathetic tone mildly (Marty 1986),
modeled as mild vasodilation and venous pooling, and combining it with opioids
lowers stroke volume and pressure further (Heikkilä 1984). Etomidate changes
vascular resistance and stroke volume very little. Nitrous oxide causes mild
sympathetic vasoconstriction and slight myocardial depression.

### Reflexes

MAP feedback adjusts HR or vascular resistance depending on rhythm, anesthetic
depth, blood gases, and surgical stimulation. When muscarinic block or a fixed
rhythm prevents heart-rate slowing, high MAP lowers endogenous vascular
resistance instead, reflecting the drop in systemic resistance observed when
cardiac output rises at steady pressure under atropine or pacing (Weissler 1957;
Tweed 1972). That vascular compensation stops during pulseless rhythms,
electrical silence, chest compressions, and anoxic failure.

Anesthesia blunts the baroreflex and the tachycardia of blood loss (Ebert 1992;
Sellgren 1994), although severe hypotension partly breaks through until
terminal shock. Older adults have lower cardiovagal gain and weaker reflex
tachycardia while keeping baroreflex control of vascular resistance (Kardos
2001; Laitinen 2004; Monahan 2007). Compensatory tachycardia during hemorrhage
has both a fast cardiopulmonary component and a slower endocrine component.
Acute normovolemic anemia lowers blood viscosity and triggers compensatory
rises in stroke volume and HR, more strongly when awake than under anesthesia
(Ickx 2000). Severe hypoxia slows the heart, depresses contractility, and
blunts reflex tachycardia.

Under propofol, MAP has little effect on HR (Garraza-Obaldia 2024). Slow
compensatory tachycardia therefore weakens with anesthetic depth, while reflex
slowing to high pressure remains intact, calibrated to Sawano 2013 and
Takahashi 2002. Opioids also suppress reflex tachycardia without suppressing
reflex slowing, so HR tends to fall alongside blood pressure during
propofol-remifentanil induction (Sjøen 2024). Remifentanil also adds a vagal
bradycardia that antimuscarinics block (Poterman 2017); AnaSim scales this
effect to VitalDB TIVA inductions and does not apply it to fentanyl. Septic
shock and moderate-to-severe anaphylaxis add direct tachycardia that persists
under anesthesia, and both blunt slow endogenous vascular and stroke-volume
feedback so untreated distributive shock does not rebound on its own.

### Preload and heart rate

Blood loss lowers filling pressure and preload. Reflex venoconstriction and
transcapillary refill from the interstitium compensate initially, though severe
hemorrhage exhausts venous recruitment and blunts adrenergic pressor
responsiveness (informed by porcine hemorrhage data in Meybohm 2007 and
preload-dependent venous return dynamics in Maas 2013). A non-recruitable
unstressed volume floor ensures that venoconstriction or vasopressors cannot
restore cardiac filling during exsanguination beyond about half of blood volume
without fluid or blood replacement. Venous compliance scales with baseline blood
volume so resting filling pressure stays consistent across body sizes. High
intrathoracic pressure from PEEP or a closed adjustable pressure-limiting (APL)
valve raises right atrial pressure and impedes venous return, stopping venous
return altogether if it exceeds mean systemic filling pressure.

In untreated atrial fibrillation, atrioventricular nodal conduction limits peak
ventricular rate and rapid rates further reduce stroke volume. Amiodarone and
beta-blockers slow the ventricular rate in atrial fibrillation and
supraventricular tachycardia, and can convert perfusing supraventricular
tachycardia back to sinus rhythm. In normal hearts, stroke volume falls when
arterial resistance rises faster than contractility, following
ventriculo-arterial coupling (Asanoi 1989; Chen 1998). High heart rates and
strong inotropic or vasoconstrictor stimulation are capped by venous return,
sinus rate follows an age-dependent ceiling, and extreme tachycardia further
reduces diastolic filling.

Severe left ventricular failure, whether chronic or acute, shifts blood into
the pulmonary circulation. Left atrial pressure, lung water, and right
ventricular afterload rise until left and right ventricular outputs match.
Without circulation, left atrial pressure equilibrates with mean systemic
filling pressure and pulmonary filtration and transcapillary refill stop; chest
compressions partially restore left ventricular emptying.

### Acidemia, electrolytes, and hypothermia

At and below a 37 °C alpha-stat pH of 7.2, myocardial contractility, effective
catecholamine activity, and endogenous sympathetic vasoconstriction decline
steeply, extrapolated from isolated failing human myocardium tested at pH 7.2
(Schotola 2012). Profound acidemia also causes direct arteriolar vasodilation.

Hypocalcemia depresses contractility and vascular resistance; intravenous
calcium restores contractility without raising systemic vascular resistance
above baseline, informed by normocalcemic halothane volunteers (Denlinger 1975)
and dialysis studies varying ionized calcium (Lang 1988). Hyperkalemia slows
conduction, weakens contractility, and can trigger ventricular arrhythmias or
asystole. High ionized calcium raises the potassium threshold for toxicity,
evaluated at resting pH so acidemia does not artificially mask hyperkalemic
arrhythmia. Sustained electrical silence from hyperkalemia or local anesthetic
toxicity progresses to asystole.

Moderate-to-severe core hypothermia slows HR (reaching roughly half of baseline
near 28 °C; Brown 2012), blunts cold vasoconstriction and reflex tachycardia,
depresses contractility, and increases the risk of ventricular fibrillation.
Active shivering raises oxygen demand, HR, and cardiac output (Ralley 1988),
modeled through higher sympathetic drive, HR, contractility, and skeletal-muscle
blood flow.

## Cardiovascular drugs

Vasoactive drugs act on vascular resistance, venous capacitance, stroke volume,
and HR, with alpha-1 vasoconstrictors sharing a common smooth-muscle saturation
ceiling so stacked pressors cannot raise vascular resistance without bound.
Adrenergic and muscarinic antagonists modify both drug effects and autonomic
reflexes. Anaphylaxis lowers catecholamine vascular and cardiac efficacy
directly rather than shifting epinephrine into beta-2 dominance.

### Norepinephrine

The norepinephrine model includes endogenous secretion and a clearance that
falls as plasma propofol rises (Li 2024). Only concentrations above the resting
endogenous level change HR, vascular resistance, venous capacitance, and stroke
volume. Its vascular and stroke-volume effects grow stronger under anesthesia,
with the anesthetized stroke-volume gain saturating at high doses.
Norepinephrine recruits unstressed venous volume to raise filling pressure and
preload (Persichini 2012; Hamzaoui 2010), and it supports stroke volume in
septic shock with reduced ejection fraction (Hamzaoui 2010; Carelli 2026).
Septic shock roughly halves both vascular and cardiac responses to catecholamines
(Bellissant and Annane 2000; Hernández 2013).

### Epinephrine

Epinephrine uses arterial clearance (Ensinger 1992a), which decreases with age
(Esler 1995), paired with a calibrated mixing volume that governs initial bolus
peaks and decay. Cardiac and vascular responses are fitted to awake and
anesthetized bolus data and volunteer infusions (Guinard 1995; Takahashi 2002;
Freyschuss 1986; Stratton 1985; Ensinger 1992a, 1992b; Marker 1998; Leenen
2005).

Older patients have lower cardiac and primary vascular sensitivity to
epinephrine (Leenen 2005; Pan 1986; Hogikyan and Supiano 1994). During
infusions, higher plasma levels and weaker baroreflex buffering offset that
lower sensitivity, while boluses produce smaller HR spikes in older patients.

Under propofol, epinephrine recruits pooled venous blood at lower
concentrations than those needed for arterial constriction. A slower,
higher-concentration alpha-adrenergic component raises vascular resistance above
baseline during high-dose shock infusions and supports diastolic pressure during
CPR while keeping brief bolus MAP peaks bounded; it washes out faster during
normal circulation than during low-flow CPR. The concentration-response curve is
calibrated up to the highest volunteer infusion studied in Ensinger 1992b (0.22
mcg/kg/min).

### Phenylephrine and ephedrine

Phenylephrine uses two-compartment PK from the FDA label (US Food and Drug
Administration 2012) with bolus effects calibrated to Meng 2011. Because it both
recruits venous volume and raises afterload, its net effect on cardiac output
depends on the patient's preload reserve (Rebet 2016).

Ephedrine uses a calibrated two-compartment intravenous distribution model that
preserves the distribution volume, terminal half-life, renal elimination, and
tolerance behavior of oral ephedrine (Persky 2004) while matching intravenous
bolus onset and 60-minute duration (Rådström 1995; Michelsen 1998). Peak and
early post-induction responses are checked against Meng 2011 and Sjøen 2026, and
a single 10 mg bolus retains pressor effect an hour later (Michelsen 1998).
Alpha and beta blockade shift its vascular and cardiac curves separately.
Repeated doses build presynaptic tachyphylaxis regardless of receptor blockade,
reducing pressor and peak HR responses while leaving a smaller saturating HR
effect.

### Other vasoactive drugs

Vasopressin follows label PK (DailyMed 2025) and raises vascular resistance
with mild HR slowing via V1 receptors, calibrated to vasodilatory-shock
infusions around 0.04 U/min. Because it bypasses adrenoceptors, its effect is
preserved during adrenergic blockade, acidemia, anaphylaxis, and septic shock.

Dobutamine (Kates and Leier 1978) and milrinone (calibrated two-compartment
kinetics matched to Baim 1983 and DailyMed 2026b, largely renally cleared) raise
stroke volume and HR while lowering systemic vascular resistance (Magnani 1977;
Baim 1983). Dobutamine's heart-rate curve rises progressively across stress-dose
infusions (Daly 1997), while milrinone adds stronger arterial and venous
dilation that lowers high filling pressure. Because dobutamine acts through beta
receptors, beta blockade, acidemia (Schotola 2012), anaphylaxis, and septic
shock (Hernández 2013) blunt its response, whereas milrinone keeps its effect in
those states.

Esmolol uses blood-esterase PK and selective beta-1 blockade from Sum 1983
(Wiest and Haney 2012), so clearance is independent of cardiac output and organ
function. Labetalol uses age-dependent hepatic clearance (Abernethy 1987;
organ-impairment kinetics informed by Hafsa 2022) and blocks beta-1, beta-2, and
alpha-1 receptors, with roughly sevenfold higher intravenous potency at beta
than alpha receptors and alpha-mediated vasodilation calibrated to bolus
responses (Richards 1977). Competitive antagonists scale effective agonist
concentrations according to receptor occupancy, so higher agonist doses can
overcome blockade. Antagonist occupancy also lowers resting sympathetic tone,
reflex and stimulation responses, and ventricular rate in atrial fibrillation
and supraventricular tachycardia. Labetalol's beta-2 blockade also blunts
epinephrine-driven hypokalemia and hyperlactatemia.

### Antimuscarinics

Atropine links its HR effect to its peripheral compartment (Hinderling 1985),
scaling the renally excreted fraction with renal function. Its chronotropic gain
declines with age alongside the fall in intrinsic heart rate (Jose and Collison
1970; Dauchot and Gravenstein 1971). Halothane, enflurane, and opioid anesthesia
leave atropine's HR increase largely intact (Yamaguchi 1988), whereas propofol
blunts the peak HR rise of atropine (Horiguchi and Nishikawa 2002); AnaSim
applies the same propofol attenuation to glycopyrrolate without reducing vagal
blockade. Glycopyrrolate uses three-compartment population PK with predominantly
renal clearance (Du 2025) and vagolytic potency from Ali-Melkkilä 1993, giving a
slower onset and smaller peak HR rise than atropine.

Both drugs block resting vagal tone, baroreflex slowing, and remifentanil
bradycardia, and both treat sinus bradycardia alongside beta-agonists and
milrinone. During unstimulated vagolytic tachycardia, venous unloading lowers
stroke volume over several minutes, sooner when preload is low (Weissler 1957;
Berry 1959); adrenergic drive restores stroke volume during pacing (Ross 1965).

### Ketamine

Ketamine raises HR, stroke volume, and vascular resistance through central
sympathetic stimulation (Tweed 1972). In the model, its HR rise begins earlier
than its vascular effect, and its tachycardia persists after atropine (Bovill
and Dundee 1972). Baroreflex gain falls at anesthetic concentrations, based on
rat data (Akine 2001), and a calibrated pressure offset reduces reflex slowing
and vasodilation to match pressure rises after atropine and during pacing. Beta
blockade leaves a pressor response, whereas alpha blockade reduces it, modeled
on droperidol pretreatment (Bovill and Dundee 1972; Bålfors 1983).

When sympathetic reserve is exhausted by severe hemorrhage, septic shock,
profound hypotension, severe acidemia, or myocardial energy failure, ketamine
loses its sympathetic drive and directly depresses contractility and vascular
resistance (Waxman 1980).

### Amiodarone

Rapid intravenous boluses of amiodarone briefly lower vascular resistance and
blood pressure, largely from the polysorbate 80 solvent (Munoz 1988), whereas
slower infusions are modeled to lower pressure much less. Contractility declines
over several minutes, lowering cardiac output in HFrEF (Kosinski 1984) and
producing a small delayed dip after vasodilation resolves in normal hearts
(Munoz 1988). Over 15 to 30 minutes, amiodarone slows sinus rate (Munoz 1988),
slows ventricular rate in atrial fibrillation and supraventricular tachycardia,
weakens sympathetic HR reflexes, and adds a modest rate of pharmacological
conversion to sinus rhythm.

## Heart and brain oxygen supply

Myocardial oxygen demand follows the rate-pressure product (Gobel 1978) plus a
basal demand that remains even when mechanical work stops (isolated canine heart
data in Harasawa 1992); fibrillating myocardium still consumes most of a beating
heart's resting demand. Coronary supply depends on diastolic pressure above the
coronary zero-flow pressure (Nanto 2001; canine data in Jeremy 1986), diastolic
duration, coronary flow reserve, and arterial oxygen content; above 25 mmHg left
atrial pressure, a model calibration moderates further rises in whole-wall
zero-flow pressure to reflect preserved subepicardial perfusion. When coronary
flow reserve is reduced, flow drops steeply below the autoregulatory pressure
threshold. Persistent mismatch between supply and demand causes reversible
contractile depression and ST depression.

Cerebral oxygen supply falls when cerebral perfusion pressure drops below the
lower autoregulatory limit (about 40 mmHg in normotensive adults, shifted
higher in chronic hypertension) or when arterial oxygen content drops.
Closed-chest CPR caps cerebral blood flow to a fraction of normal (canine CPR
data in Michael 1984), which slows ischemic injury without restoring wakefulness
or spontaneous breathing even when hypothermia lowers cerebral oxygen demand.
Hypoxia raises normocapnic cerebral blood flow as a model calibration, and acute
hypercapnia augments that vasodilation so oxygen delivery is better preserved
during hypercapnic hypoxemia (Battisti-Charbonney 2011; Willie 2015). Acute
hypocapnia constricts cerebral vessels and trims reserve during hypoxemia or
anemia without causing ischemia in a normoxic brain (Brian 1998). Cerebrospinal
fluid bicarbonate adapts over hours of sustained PaCO₂ change. During severe
hypotension, cerebral flow becomes pressure-limited, and restoring pressure
returns flow up to the current CO₂ ceiling (Brian 1998).

Cerebral oxygen demand changes with core temperature (roughly 7% per °C),
anesthetic depth, and seizure activity, with flow-metabolism coupling raising
blood flow when pressure and oxygen content allow. Propofol keeps cerebral blood
flow coupled to metabolic demand, whereas sevoflurane adds modest luxury
perfusion for a similar drop in oxygen consumption (Kaisti 2003); anesthetic
suppression does not shift the cellular ischemic threshold itself (based on
barbiturate focal-ischemia data in Astrup 1981). Hypothermia slows both acute
oxygen depletion and ischemic injury. Brain oxygen stores buffer a sudden loss
of circulation for a few seconds before electrical activity fails (Astrup 1981),
causing loss of consciousness, airway tone, BIS (which also tracks cerebral
hypoperfusion during carotid occlusion in Nebout 2013), and finally respiratory
drive. At normothermia, EEG flattens in about 11 seconds after circulation stops
(de Vries 1998); AnaSim models recovery over roughly 2 minutes once perfusion
returns after sustained arrest (brief defibrillation-testing pauses recovered
within about 20 seconds in de Vries 1998).

Prolonged severe hypoxia or sustained low-flow CPR accumulates ischemic neuronal
injury in proportion to metabolic demand (primate focal-ischemia thresholds in
Jones 1981). After roughly 4 minutes without perfusion (the
electrical-to-circulatory phase boundary in Weisfeldt and Becker 2002), cortical
injury progressively suppresses postanoxic EEG (Hofmeijer 2015; Jørgensen and
Malchow-Møller 1981), mapped onto BIS in AnaSim even after circulation returns.
Around 5 minutes, unresponsiveness progresses to permanent coma, and longer
arrest progressively abolishes brainstem respiratory drive.

CO₂ narcosis depresses responsiveness and airway tone without anoxic injury.
Its severity depends on both PaCO₂ and pH (Westlake 1955; Sieker and Hickam
1956): acidemia causes narcosis at lower PaCO₂ than compensated hypercapnia,
and deep hypercapnic coma depresses medullary respiratory drive into shallow
hypoventilation. Recovery after prolonged hypercapnic coma lags blood-gas
correction (Sieker and Hickam 1956), using teaching calibrations for onset and
recovery times. Hypercapnia does not alter BIS directly under propofol
anesthesia (Turnbull 2008).

## Cardiac arrest and resuscitation

Cardiac arrest is confirmed after 15 seconds of ventricular fibrillation (VF),
pulseless ventricular tachycardia (VT), asystole, extreme bradycardia, or
pulseless electrical activity (PEA, defined by an organized ECG with negligible
pulse pressure from inadequate cardiac output or profound vasoplegic collapse).
The simulation keeps running during arrest so learners can perform
resuscitation and treat the underlying cause.

### Course of arrest

Severe myocardial ischemia or hypoxemia first drains a myocardial energy
reserve lasting about 4 minutes without perfusion. Restoring a perfusing rhythm
within this electrical phase restores circulation promptly (Weisfeldt and Becker
2002).

Once the energy reserve is spent, an organized rhythm slows over several
minutes from junctional through idioventricular and agonal rhythms to
asystole, while untreated VF degrades to fine VF and then asystole. At the same
time, cumulative ischemic injury lowers post-resuscitation contractility
(reversible stunning is documented in cardiac arrest survivors in Laurent 2002;
AnaSim also tracks permanent injury that ends the session around 11 minutes
after circulation stops).

### Chest compressions

[Chest compressions](../anasim/physiology/resuscitation.py) at 110/min generate
roughly a quarter of resting cardiac output when filling and vascular tone are
normal, about a fifth under anesthesia, and very little after severe
hemorrhage, with passive venous filling bypassing heart failure's flat upper
Starling limb. Forward flow and vascular resistance set diastolic pressure
between compressions, and each compression adds a pulse scaled by cardiac
filling.

Coronary perfusion happens between compressions against venous, left atrial,
and myocardial closing pressures. Myocardial energy recovers when arterial
diastolic pressure exceeds about 25 mmHg, matching the CPR diastolic target in
Meaney 2013 and exceeding the 15 mmHg coronary perfusion pressure threshold for
return of spontaneous circulation in Paradis 1990 once right atrial pressure is
subtracted. In an anesthetized adult, compressions produce roughly 70/25 mmHg
and EtCO₂ around 15 to 20 mmHg, keeping VF coarse or converting fine VF back to
coarse VF.

Epinephrine raises arterial diastolic pressure (Kim 2025) and coronary
perfusion pressure (canine CPR data in Michael 1984) during compressions, and
lowers EtCO₂ at fixed ventilation (Cantineau 1994b), modeled in AnaSim through
higher afterload reducing forward flow.

Although transient cardiac activity occasionally resumed without intervention
after withdrawal of life support, no patient regained consciousness or survived
(Dhanani 2021). As a teaching calibration, AnaSim requires sustained chest
compressions with adequate coronary perfusion, viable myocardium, and correction
of severe hypothermia, acidemia, hypoxia, hyperkalemia, or local anesthetic
conduction block before beats return from primary asystole. Most primary
asystole also requires a vasopressor on board, whereas reversible
conduction-block asystole from local anesthetic toxicity (extrapolated from rat
bupivacaine lipid resuscitation in Hiller 2009) or hyperkalemia (after calcium)
can restart with compressions once the block clears. Interrupting compressions
or losing perfusion resets the recovery interval.

### Defibrillation and cardioversion

A first 200 J biphasic shock terminates VF roughly 9 times in 10 (Stiell 2007;
Koster 2008), with lower calibrated success in fine VF, pulseless VT, or
hypoxic arrest. Severe hypothermia and severe hyperkalemia sharply lower shock
success until treated. Each shock causes mild contractile stunning that
resolves over several minutes of coronary perfusion, calibrated from internal
defibrillator shocks (Tokano 1998).

After brief VF, successful defibrillation returns directly to sinus rhythm, and
arterial pressure recovers over 5 to 20 seconds (matching the post-VF pressure
recovery times in Park 1994, modeled here as arterial refilling). After
prolonged arrest, terminating shocks often produce transient post-shock
asystole before escape beats or refibrillation (Pierce 2015), recovering
post-shock bradycardia returns to sinus rhythm once heart rate improves, and
cardiac output remains limited by energy depletion, post-resuscitation stunning
(Laurent 2002), and cumulative injury. A late shock without CPR therefore
leaves PEA or asystole, and delayed resuscitation leaves persistent post-arrest
hypotension.

Synchronized shocks deliver energy on the next R wave, automatically clear sync
mode after delivery or cancellation, and usually convert atrial fibrillation,
supraventricular tachycardia, and perfusing VT. They do not fire during VF,
asystole, or compression artifact. An unsynchronized shock that lands near the
T-wave peak of an organized perfusing rhythm can trigger VF (about 15% of the
time).

Each shock requires a fresh 5-second charge (within the under-7-second
specification) that expires after 60 seconds (ZOLL 2023). Compressions can
continue while charging, and disarming cancels a charge or pending synchronized
shock.

### Ventricular arrhythmias

[Severe ischemia, hypoxia, hyperkalemia, or deep
hypothermia](../anasim/physiology/arrhythmia.py) in a beating heart can trigger
VT or VF. After a successful shock for VF or pulseless VT, arrhythmia frequently
recurs within the first 2 minutes (Koster 2008; van Alem 2003), driven in
AnaSim by persistent myocardial ischemia or hypoxia.

Amiodarone (Korth-Bradley 1996) and lidocaine can convert sustained perfusing
VT over tens of minutes when potassium is not severely elevated, with
amiodarone converting more reliably (Somberg 2002; refractory VT/VF amiodarone
dosing in Scheinman 1995). In VF, both drugs reduced subsequent shock
requirements and improved survival to hospital admission in Kudenchuk 2016,
modeled here as higher defibrillation success once CPR circulates them, though
neither prevents rearrest on its own (Salcido 2018).

## Lidocaine toxicity

In awake patients, [lidocaine](../anasim/physiology/toxicity.py) produces
sympathetic stimulation and seizures at moderate toxic concentrations (Foldes
1960; Jorfeldt 1968). Higher concentrations replace seizures with central
nervous system depression and apnea, then add myocardial depression, QRS
widening, ventricular arrhythmias, and cardiac arrest. Acidemia worsens both
neurological and cardiac toxicity.

Low cardiac output slows lidocaine distribution and hepatic clearance, while
renal failure leaves parent-drug disposition unchanged (Stenson 1971; Thomson
1973). A 2 g infusion over an hour in HFrEF produces hypotension and coma in
sinus rhythm without arrest, matching the post-infusion concentration reported
in Armstrong 1988, though linear clearance eliminates the drug faster than the
prolonged decay observed in that case.

Midazolam, propofol, etomidate, sevoflurane, and nitrous oxide suppress
lidocaine seizures and their sympathetic surge. Neuromuscular blockers mask
visible convulsions without stopping cerebral seizure activity.

Intravenous lipid emulsion lowers lidocaine delivery to effect sites by roughly
a fifth as a heuristic calibration based on the 23% drop in unentrapped venous
plasma exposure in volunteers (Heinonen 2015, which itself saw no reduction in
brain-toxicity symptoms and attributed the lower venous exposure to larger
tissue distribution). It also lessens contractile depression, based on rat
bupivacaine resuscitation (Fettiplace 2014), with calibrated reductions in
conduction block and arrhythmogenic stress.

## Fluids

### Volume kinetics

Lactated Ringer's and 0.9% saline distribute between plasma and the
interstitium (Hahn 2017; Hahn 2024). Distribution slows during hypotension, so
crystalloid given after induction stays intravascular longer (Ewaldsson and
Hahn 2001; Hahn 2021), acute hypovolemia suppresses renal crystalloid
elimination (Drobin and Hahn 1999), adrenergic stimulation modifies fluid
retention and diuresis (ovine data in Vane 2004; human data in Li 2012), and
high venous filling pressure during low-flow or arrest states continues to
drive fluid extravasation. Urine output depends on MAP, cardiac output, blood
volume, renal vasoconstriction, age, and renal function, with hypervolemic
diuresis at high blood volumes. Under anesthesia, roughly 11% of a 25 mL/kg
acetated Ringer's bolus is excreted over 150 minutes (median 11% under
isoflurane and 7% under propofol, with similar dilution and retention across
agents; Ewaldsson and Hahn 2005); AnaSim calibrates lactated Ringer's excretion
near 8% over 150 minutes. Normal saline is excreted more slowly than lactated
Ringer's (Reid 2003). A larger fraction of 5% albumin stays intravascular
initially, and a quarter of infused lipid emulsion volume remains in plasma.
Queued fluid, blood, and lipid boluses drain sequentially on separate
intravenous channels.

Fluid also leaves the circulation through capillary leak into a third-space
compartment that slowly returns fluid when blood volume falls. Anaphylaxis
drives rapid plasma leak while keeping red cells intravascular, raising
hemoglobin concentration (Fisher 1986); the leak slows once the trigger is
removed, though ongoing reactions often need repeated crystalloid boluses.

Transfused packed red cells track donor red-cell volume and hemoglobin
separately from their protein-free supernatant, which follows crystalloid
kinetics. High left atrial pressure increases extravascular lung water, shunt,
and lung stiffness, connecting fluid balance to gas exchange. Hypoalbuminemia
and inflammatory permeability in sepsis or anaphylaxis lower the threshold for
pulmonary edema and accelerate filtration.

### Electrolytes and fluid composition

[Fluid balance](../anasim/physiology/fluid_balance.py) tracks sodium,
potassium, chloride, calcium, magnesium, and phosphate in a single
extracellular compartment, and tracks albumin separately in plasma and the
interstitium. Excess adipose tissue in obesity adds less extracellular and
intracellular water per kilogram than lean mass (Waki 1991, applied above a BMI
of 25), and baseline metabolic acid and buffer loads scale with lean mass
rather than total body weight. Water shifts between extracellular fluid and
cells to equalize tonicity, so free water from lactated Ringer's and lipid
emulsion distributes across total body water.

Each fluid contributes its own ions, albumin, and buffer base according to its
strong ion difference. Large volumes of 0.9% saline therefore produce
hyperchloremic metabolic acidosis, whereas lactated Ringer's does not
(Scheingraber 1999). Stored packed red cells represent 3-week-old blood with
acidic, hyperkalemic, high-lactate supernatant and storage acid (Rabcuka 2025),
so rapid transfusion adds acid, CO₂, and potassium. Hepatic metabolism of
citrate (Kramer 2003; Kahn 1979) and exogenous lactate returns base over time,
which can lead to post-transfusion metabolic alkalosis. Cirrhosis and
hypothermia impair citrate clearance (Kramer 2003; Ludbrook and Wynn 1958);
scaling citrate and D-lactate clearance with liver function and halting
clearance when oxygen delivery fails are model assumptions.

Potassium exchanges with a large intracellular pool over minutes. Mineral
metabolic acidemia shifts potassium into plasma much more strongly than acute
respiratory acidemia, whereas lactic acidemia causes negligible direct potassium
shift (Adrogué and Madias 1981; Orringer 1977). Alkalemia and circulating
epinephrine shift potassium into cells (blunted by labetalol's beta-2 blockade),
and succinylcholine releases potassium into plasma (Martyn and Richtsfeld 2006).
Malignant hyperthermia and severe hypoxia release cumulative cellular potassium
that clears slowly over hours after the insult resolves.

Ionized calcium depends on albumin, citrate, and 37 °C pH (Thode 1983; Bolan
2001), so dilution and rapid citrated transfusion lower it without shifting
during hypothermia. Once rapid transfusion stops, ionized calcium rebounds
partially as citrate redistributes and clears (Kahn 1979), while dilutional
losses persist. Intravenous calcium chloride mixes into extracellular fluid and
clears in proportion to circulation and renal function, so a dose given during
arrest waits for flow; calibrated to 1 g 10-minute infusions during cesarean
delivery, 1 g raises ionized calcium by about 0.4 mmol/L, with half of the rise
gone roughly 30 minutes after infusion start (Ansari 2025). Shed blood removes
calcium, and calcium lost to dilution or bound by citrate is not replaced by
endogenous stores within a session (consistent with progressive ionized calcium
falls during citrate apheresis in Bolan 2001). During zero-flow cardiac arrest,
arterial blood gases, hemoglobin, electrolytes, and lactate remain isolated
from peripheral venous and fluid changes until circulation resumes.

### Renal acid-base handling

When urine output and renal perfusion allow, the kidneys excrete daily fixed
acid, excrete excess sodium and bicarbonate during volume expansion and
alkalemia, and gradually adjust bicarbonate toward chronic compensation for
sustained hypercapnia (González 2018) or hypocapnia (Krapf 1991). Chloride
moves opposite to bicarbonate, while urinary sodium follows extracellular
strong-ion balance and volume load (Reid 2003; Chowdhury 2012); surgical
antidiuretic hormone surges (Philbin and Coggins 1978) are not explicitly
modeled.

## Respiratory control

[Ventilatory control](../anasim/physiology/respiration.py) integrates
anesthetic effects, CO₂ feedback, hypoxic drive, and respiratory muscle
strength.

Propofol mainly lowers tidal volume, whereas opioids mainly lower respiratory
rate, and etomidate blunts the hypercapnic slope while preserving resting
ventilation better than methohexital (Choi 1985) and better than propofol in
the model. Sevoflurane causes rapid shallow breathing at surgical
concentrations (Doi and Ikeda 1987) and leads to apnea near 2 MAC. Ketamine
shifts the steady-state CO₂ response slightly (Bourke 1987), while rapid
high-dose boluses briefly depress central drive. Hypocapnia and neuromuscular
block suppress respiratory effort, and rising CO₂ can restore breathing during
opioid administration. Under anesthesia, the apneic threshold shifts alongside
the drug-elevated resting PaCO₂ (Hickey 1971), while active hypoxic drive
lowers it during light sedation on room air.

For morphine and hydromorphone, respiratory rate is modeled separately from
overall ventilatory drive (Olofsen 2026), combining Olofsen's fractional opioid
response with Bouillon's nonlinear CO₂ curve (Bouillon 2004b). At overdose
concentrations, both drugs flatten CO₂ sensitivity and raise the apneic
threshold enough to cause apnea on room air. Core hypothermia scales
spontaneous ventilation down with metabolic demand and further slows
respiratory rate below about 31 °C.

Hypoxic ventilatory drive fades during sustained hypoxia. Low doses of propofol
(Blouin 1993; Nieuwenhuijs 2000), sevoflurane (Sarton 1996), halothane (Knill
and Gelb 1978), and analgesic morphine (Weil 1975; sex-dependent in Dahan 1998)
depress hypoxic drive steeply, and fentanyl-midazolam sedation also produces
hypoxemia and apnea (Bailey 1990). Extending surgical halothane abolition of
hypoxic drive (Knill and Gelb 1978), AnaSim abolishes hypoxic drive under
surgical sevoflurane, propofol, nitrous oxide, and high-dose opioids while CO₂
drive remains. Published age effects on chemoreflexes vary by method
(García-Río 2007; Poulin 1993; Peterson 1981; Lhuissier 2012); as a
calibration, acute hypoxic sensitivity declines substantially in older adults
while CO₂ sensitivity declines much less. During airway obstruction, airflow is
restricted while inspiratory effort continues, with partial load compensation
during severe acute obstruction.

Metabolic acidosis lowers the PaCO₂ setpoint over a couple of minutes, and
metabolic alkalosis raises it by roughly 0.7 mmHg per mmol/L of excess base
chiefly through smaller tidal volume (calibrated from chronic metabolic
alkalosis in Javaheri 1982), though anesthetics and opioids still limit the
ventilatory response.

[Respiratory muscles fatigue](../anasim/physiology/fatigue.py) when sustained
inspiratory load, such as breathing against an obstructed airway, exceeds a
safe fraction of available strength. Neural demand is compared against
surviving muscle capacity (including any residual neuromuscular block), so
fatigued muscles do not recover while obstruction continues. Endurance drops
steeply at higher loads (Bellemare and Grassino 1982), and inadequate oxygen
delivery accelerates fatigue. Fatigue lowers spontaneous tidal volume, while
pressure support and mandatory breaths unload the muscles (Leung 1997).
High-frequency fatigue (Aubier 1981) and repeat endurance time (Rohrbach 2003)
recover within minutes of rest, whereas low-frequency contractile depression
recovers more slowly over 15 to 60 minutes (Aubier 1981; Rohrbach 2003) and up
to 24 hours after prolonged heavy loading (Laghi 1995; Travaline 1997).

## Gas exchange and blood chemistry

### Oxygenation

Ventilation exchanges gas with alveolar stores that govern preoxygenation,
apneic desaturation, changes during blood loss, and circuit oxygen uptake.
Shunt mixes venous and capillary blood by total oxygen content, including both
hemoglobin-bound and dissolved oxygen.

Part of pulmonary blood flow also passes through poorly ventilated low-V/Q
units, which lower PaO₂ strongly at low FiO₂ and minimally at high FiO₂. Their
share grows with age to reproduce the normal sea-level A-a gradient on room air
(Crapo 1999). COPD (Wagner 1977) and acute bronchospasm (Ballester 1989) add
further low-V/Q mismatch without much true shunt, and COPD adds parallel
alveolar dead space that widens during acute bronchospasm. Supplemental oxygen
therefore corrects most hypoxemia while potentially raising PaCO₂ through
hypoventilation and V/Q redistribution in acute COPD (Robinson 2000; controlled
oxygen in DeGaute 1981) and through Haldane displacement of CO₂ from hemoglobin
(Dash 2016; O'Neill and Robbins 2017). High PEEP can also overdistend aerated
units and add alveolar dead space, widening the PaCO₂-to-EtCO₂ gradient.

In unconscious patients, low-V/Q perfusion increases with age (Gunnarsson 1991)
and dependent airways close at end-expiration, especially after age 30; PEEP of
about 6 cmH₂O (higher in obesity) keeps them open (Hedenstierna and Edmark
2015). Older patients therefore oxygenate worse under anesthesia without PEEP.
After upper abdominal surgery, FRC stays depressed after emergence so this
closure persists (Wahba 1975; Putensen-Himmer 1992). Lower abdominal surgery
causes persistent postoperative hypoxemia, atelectasis, and about a one-third
decline in FVC and FEV₁ (Lindberg 1992), which AnaSim models by retaining
roughly a third of anesthetic airway closure after emergence.

### Gas transport and stores

[Gas transport](../anasim/physiology/gas_transport.py) moves oxygen, CO₂, and
buffer base through pulmonary capillary, arterial, and mixed venous blood while
shunt bypasses the lungs. Two tissue pools and an extracellular pool store CO₂,
equilibrating over 30 to 60 minutes (or longer during controlled ventilation)
after ventilation or metabolic changes (Henneberg 1987; Taskar 1995;
Magor-Elliott 2021). Blood flow governs exchange with tissue stores, and
ventilatory control senses arterial PCO₂. Lung tissue itself accounts for 5% of
oxygen uptake and CO₂ production directly with alveolar gas (Loer 1997), so
ventilating a patient with no pulmonary blood flow still leaves a trace
alveolar PCO₂.

When the airway is closed or circuit gas runs out, gas absorbed from the
alveoli is not replaced, causing lung volume to shrink down toward anatomic
dead space and adding absorption atelectasis and shunt once volume falls below
closing capacity. Opening the airway lets circuit gas refill the lungs
(computational model in Laviola 2020).

During apnea with an open airway, passive convective inflow moves gas through a
shared dead-space buffer so circuit, dead space, and alveoli conserve mass
together. Uptake of nitrous oxide and sevoflurane draws extra gas into the
alveoli, whereas gas washout refills lost volume first and then flows outward,
diluting alveolar oxygen and CO₂ when inspired oxygen is low.

### Oxygen delivery and lactate

Oxygen consumption matches metabolic demand until oxygen delivery falls to a
critical threshold around a 60% extraction ratio (Ronco 1993). Because critical
delivery scales with demand, hypothermia and anesthesia lower it. Below that
threshold, oxygen uptake and aerobic CO₂ production drop in proportion to
delivery, and the oxygen debt produces [lactate](../anasim/physiology/lactate.py)
at a rate fitted to hemorrhagic shock data in dogs (Dunham 1991).

Epinephrine, sepsis, severe vasoplegic hypotension (MAP below 40 mmHg),
malignant hyperthermia, and lidocaine convulsions also raise lactate production
(Grip 2015; Revelly 2005; Orringer 1977). Lactate has a distribution volume
anchored to 0.26 L/kg (measured in stable septic ICU patients in Levraut 1998
and scaled to lean mass in AnaSim) and clears with saturable Michaelis-Menten
kinetics slowed by hypothermia. The liver provides the majority of clearance,
scaled by hepatic function and oxygen delivery, while extrahepatic clearance
holds until severe hypoxia and stops when oxygen delivery ceases.

Endogenous lactate removes buffer base one-for-one as it accumulates and
returns that base when cleared. During hypoxic low-flow states, lactic acid
consumes buffer base primarily in venous blood, raising venous PCO₂ and
reproducing the arterial-to-venous acid-base divergence seen during ventilated
CPR (Weil 1986). Lactated Ringer's raises measured lactate modestly without
acidosis (within-group rise after 30 mL/kg in Zitek 2018, though the
between-group difference versus saline was not significant). Exogenous sodium
lactate generates bicarbonate and undergoes substrate oxidation (Chioléro 1993,
where net pulmonary VCO₂ fell slightly as bicarbonate was retained), and
citrate metabolism likewise generates base (Kahn 1979; Kramer 2003); AnaSim
adds the corresponding oxidized CO₂ load. Blood gas analyzers report only the
L-isomer, which is half of the racemic lactate in Ringer's.

### Acid-base

Acid-base chemistry is solved at 37 °C (alpha-stat PCO₂), alveolar exchange
equilibrates against in-vivo PaCO₂ at core temperature, and arterial pH, PaCO₂,
and EtCO₂ reflect core temperature. Ionized calcium, renal compensation, and
myocardial acidosis use the 37 °C values. [Blood
buffering](../anasim/physiology/blood_chemistry.py) builds on Rees 2010 with
plasma albumin and phosphate charge from Figge 2018, and hemoglobin oxygen
affinity uses Dash 2016 at a reference 2,3-DPG level. Gas exchange and
oxygen-affinity shifts conserve CO₂, oxygen, and buffer base, and bleeding,
fluids, transfusion, and urine add or remove buffer base directly.

Interstitial weak-acid buffering follows Wooten (2003, 2010), extended with
cellular buffering calibrated to keep acute PaCO₂ changes within human pH and
bicarbonate ranges (Brackett 1965; Arbus 1969) with minimal shift in standard
base excess. Standard base excess is computed with the Van Slyke equation as in
clinical analyzers, sitting near 0 mmol/L in nonpregnant adults and around
−2 mmol/L in late pregnancy.

## Respiratory mechanics and aeration

The [mechanics model](../anasim/physiology/resp_mech.py) combines airway
resistance, compliance, tissue viscoelasticity, and inspiratory muscle
pressure, scaled to body size and lung conditions (Pelosi 1998). Alveolar
pressure (including trapped auto-PEEP during severe expiratory obstruction,
rather than proximal pressure behind an obstructed airway) transmits to the
pleural space in proportion to chest-wall-to-total elastance (Jardin 1985),
setting the intrathoracic pressure transmitted to venous return.

Ventilator assistance lowers inspiratory muscle effort, anesthesia ends
inspiratory effort earlier as the lung inflates, and severe hyperinflation
flattens the diaphragm and reduces maximal inspiratory pressure.

Bronchospasm raises expiratory resistance as lung volume falls, increases
steeply in severe anaphylaxis, and can limit expiratory flow enough to cause
gas trapping and intrinsic PEEP. Epinephrine reverses bronchospasm via beta-2
receptors (antagonized by labetalol), while sevoflurane and ketamine provide
partial bronchodilation.

[Lung aeration](../anasim/physiology/lung.py) depends on pressure history, with
recruitment and collapse changing gas volume, shunt, and compliance. Inert gas
in the alveoli slows recollapse, and atelectatic shunt evolves separately from
bulk FRC and compliance (Rothen 1995). Preoxygenation and induction apnea leave
roughly a 5% shunt, which is also the starting point for maintenance sessions.
In normal-weight lungs, 5 cmH₂O PEEP prevents further collapse without
reopening collapsed units, whereas higher closing pressures in overweight and
obese patients still allow atelectasis on 100% oxygen at 5 cmH₂O PEEP
(Hedenstierna and Edmark 2015). Atelectasis carries into emergence and early
recovery (Benoît 2002; Östberg 2019). After extubation, most atelectasis clears
over the first 24 hours, fastest after nonabdominal surgery and slowest after
upper abdominal surgery (Strandberg 1986; Putensen-Himmer 1992). A smaller
dependent fraction resolves over several days (Lindberg 1992), and morbid
obesity prevents clearance over 24 hours (Eichenberger 2002).

## Temperature

Awake thermoregulation defends core temperature, with sweating and cutaneous
vasodilation limiting fever above a hypothalamic setpoint shifted upward by
sepsis, whereas general anesthetics widen the interthreshold range by slightly
raising the sweating threshold while markedly lowering cold-defense thresholds
(Sessler 2000). Basal heat production scales with body weight to the 0.75
power, and excess body fat lowers mean specific heat capacity. Temperature and
sepsis alter metabolic heat production (scaling down toward a quarter of
baseline in deep hypothermia), forced-air warming is modeled to shield 60% of
body surface area from room loss during surgery while delivering convective
heat (calibrated to full-body convective transfer in Giesbrecht 1994), surface
cooling removes heat, and unwarmed intravenous fluids or refrigerated packed
red cells cool the core unless the 37 °C fluid warmer is on. Thermoregulatory
vasoconstriction limits peripheral heat loss below the vasoconstriction
threshold, which shifts lower in older adults (Sessler 2000; Ozaki 1997), and
heat production falls with oxygen delivery so core temperature drops faster
during arrest.

General anesthetics suppress thermoregulation, reduce metabolic heat
production, and redistribute heat from the core to the periphery;
redistribution halts during cardiac arrest. This redistribution follows
propofol-fentanyl data (Matsukawa 1995) and is extended to sevoflurane and
etomidate as a model assumption, whereas ketamine largely preserves peripheral
vasoconstriction and spares redistribution cooling (Ikeda 2001) and midazolam
shifts thermoregulatory thresholds much less (Kurz 1995b). Opioids lower both
vasoconstriction and shivering thresholds (Kurz 1995c). First-hour
redistribution cooling is smaller in heavier patients because of their higher
mass-to-surface ratio (Kurz 1995a).

Cold patients shiver below about 35.8 °C once awake or emerging (matching the
35.9 °C unmedicated volunteer threshold in Kurz 1995b, 1995c), and awake
sessions that begin hypothermic start with active shivering. Opioids,
neuromuscular block, and moderate-to-severe hypothermia (shivering ceases below
32 °C in Brown 2012, modeled here as tapering from 33 °C to zero at 31 °C)
suppress shivering, and it stops during cardiac arrest. Shivering increases
oxygen consumption, CO₂ production, HR, and cardiac output (Ralley 1988).

## Malignant hyperthermia

[Malignant hyperthermia](../anasim/physiology/malignant_hyperthermia.py) (MH)
can be triggered during sevoflurane anesthesia or within an hour of
succinylcholine. Clinical onset varies widely, from a median of 45 minutes
with sevoflurane alone into the second or third hour with other agents (Visoiu
2014), so the case or instructor determines when an episode begins. Once
established, skeletal muscle hypermetabolism continues even after the trigger
is stopped.

Left untreated, oxygen consumption and CO₂ production approach five times
baseline, with extra heat from anaerobic glycolysis and calcium cycling. The
coupled gas exchange, lactate, fluid, and temperature models then generate
hypercapnia, lactic acidosis, hyperkalemia, and rising core temperature.
Tachycardia and generalized muscle rigidity worsen alongside severity,
stiffening the chest wall, limiting spontaneous tidal volume, and persisting
despite neuromuscular block.

Dantrolene PK follows the Ryanodex label (DailyMed 2024). Dantrolene suppresses
the crisis and lowers twitch height without TOF fade while sparing ventilation
(awake volunteer data in Flewellen 1983). Ongoing sevoflurane or
succinylcholine competitively opposes dantrolene and prevents full resolution
until triggers are eliminated. Once controlled by sustained dantrolene
suppression after trigger washout, MH does not relapse as dantrolene clears,
though re-exposure to sevoflurane or succinylcholine can restart it.

## Anesthesia machine and ventilation

### Breathing circuit

The [circle system](../anasim/machine/circuit.py) mixes fresh gas with circuit
gas in a 6 L circuit containing a 3 L reservoir bag and bellows, tracking
oxygen, volatile agent, nitrous oxide, nitrogen, and gas vented through the
scavenger.

When patient uptake exceeds fresh gas and vapor inflow, or when the ventilator
runs into an open circuit or zero mask seal, the reservoir empties without
drawing in room air. As the bellows collapses, delivered breaths are limited to
the remaining circuit gas plus inspiratory fresh gas flow, and PEEP, CPAP, or
airway splinting drops to zero once the bellows is empty without fresh gas flow.

Vaporizer output depends on fresh gas flow, dial setting, and remaining agent
(from a 250 mL fill capacity that can be refilled during a session). A
proportioning system limits nitrous oxide to three times the delivered oxygen
flow, and oxygen supply failure cuts off both oxygen and nitrous oxide. The O₂
flush valve delivers 45 L/min of oxygen directly into the circuit, bypassing the
vaporizer; it refills a collapsed bag, dilutes volatile agent and nitrous
oxide, and raises circuit pressure if the APL valve is closed.

### Ventilator

The [ventilator](../anasim/machine/ventilator.py) supports standard anesthesia
workstation modes (see the [mode names](../README.md#features)), combining GE
Carestation mode behavior with Dräger waveform resistance, rise-time, and
timing parameters.

| Mode | Behavior |
|------|----------|
| VCV | Delivers set tidal volume within the pressure limit |
| PCV | Holds set inspiratory pressure for the set inspiratory time |
| PCV-VG | Adjusts inspiratory pressure toward target volume within the pressure limit |
| SIMV | Synchronizes mandatory breaths with patient efforts at the set rate |
| PSV | Cycles support off inspiratory flow, with apnea backup ventilation |
| CPAP | Maintains constant airway pressure during spontaneous breathing |

Mandatory breaths include volume already drawn by the patient's own effort.
Inspiratory setting changes take effect on the next breath, and PEEP applies
during expiration. Flow triggering requires active muscle effort rather than
passive PEEP shifts. Tracheal tube obstruction raises airway resistance, and
inflations without measurable exhaled volume do not count as breaths and blank
plateau pressure. Auto-PEEP decays as trapped gas empties during expiration or
pauses. Leaving mandatory modes or hitting an apnea timeout clears stale
pressure and compliance readings; leaving PSV clears apnea backup, and apnea
timeout immediately delivers a backup breath.

Steady-state maintenance initializes ventilator settings to target the patient's
resting PaCO₂, adjusting tidal volume within predicted-body-weight limits
before raising respiratory rate. COPD lowers the initial rate cap to preserve
expiratory time, so severe obstruction or obesity can leave PaCO₂ slightly
above baseline.

### Bag and mask ventilation

During manual bagging and unassisted spontaneous breathing, the APL valve sets
both the peak pressure limit and baseline circuit pressure once the reservoir
bag is more than half full. With APL fully open, bag squeezes create no
positive pressure and spontaneous breathing meets no valve resistance. Higher
settings allow positive pressure up to the APL limit, so obstruction or
bronchospasm reduces delivered volume. At routine settings up to 20 cmH₂O,
regular squeezes vent excess fresh gas and keep baseline PEEP low; setting APL
higher, running high fresh gas flow, or stopping bagging lets surplus gas build
PEEP up to the APL or mask-seal ceiling, which can depress arterial pressure.
Manual breaths synchronize with patient efforts, release once target volume is
reached, and can also be delivered one squeeze at a time. Mechanical
ventilation isolates the APL valve.

During mask ventilation, baseline APL pressure and mean airway pressure help
splint the pharynx open. A loose mask leaks assisted breaths and CPAP, lowers
airway pressure, and reduces measured flow and volume. To limit peak mask
pressures near the 15 to 20 cmH₂O gastric-insufflation threshold (Bouvet 2014),
AnaSim models external leakage above 20 cmH₂O even with a tight seal. A full
seal represents two-handed mask ventilation, and an 80% seal approximates the
roughly 20% smaller tidal volume delivered by a one-handed hold (Joffe 2010,
which attributed that difference to upper-airway patency rather than audible
leak). Room air enters around a loose mask during unsupported breaths or when
patient effort exceeds assisted delivery, and PCV-VG raises inspiratory
pressure to compensate for leaks.

## Monitors

Optional monitor audio plays plethysmograph pulse tones and active alarms.
Alarm silence lasts 2 minutes of real time and ends early if a new alarm sounds
or an active alarm flips between low and high limits. Lethal rhythm and arrest
alarms (Asystole, VFIB, VTACH, and Arrest) highlight the HR tile red and show
the rhythm name in the monitor alarm banner (leaving numeric HR blank during
ventricular fibrillation). Along with oxygen pipeline failure (O2Supply), they
break prior silence, re-break silence if they recur after clearing, and sound a
faster, higher-pitched cadence. Core temperature alarms outside 35.5 to
38.0 °C. Peak airway pressure alarms above 40 cmH₂O, when a pressure-limited
ventilator breath hits `min(40, Pmax)` before delivering target volume, or when
continuous manual bagging is pressure-limited at or above 40 cmH₂O.

### ECG, arterial pressure, and pulse oximetry

ECG, arterial pressure, and plethysmography share the same cardiac beat timing,
with HR averaged over recent beats. A prolonged pause lowers displayed HR and
blanks it after 8 seconds without a beat. ECG renders a brief discharge artifact
during defibrillation and preserves ventricular tachycardia amplitude during
hyperkalemia. ST depression blanks during non-analyzable rhythms (VF, asystole,
VT, and wide-complex conduction block).

The arterial waveform reflects MAP, stroke volume, body surface area, and
age-dependent arterial compliance (Mahdi 2017; Franklin 1997), keeping beat-foot
continuity across rhythm transitions. Compliance increases as pressure drops
(Langewouters 1984), so pulse pressure narrows in hypotension and widens with
vasopressors. Under propofol, systolic pressure falls roughly twice as much as
diastolic pressure (compared with three to four times in Ebert 1992 and
Jeleazcov 2015). Epinephrine's inotropic effect widens pulse pressure beyond
stroke volume alone, consistent with human dobutamine data (Piccioli 2023).
Catheter and transducer dynamics filter the displayed trace. At very low output,
diastolic pressure stays at or above venous pressure, and pulse pressure
narrows smoothly toward that floor.

Pulse oximetry depends on rhythm and peripheral perfusion, which reflects
cardiac output, vascular resistance, temperature, MAP, and pulse pressure. Poor
perfusion shrinks plethysmograph amplitude, slows averaging, and delays
displayed readings through a circulatory transit model (MacLeod 2005). After
pulse loss, numeric SpO₂ waits for sustained beats before locking back on.

Hyperkalemia progresses from peaked T waves (scaled to calcium-adjusted
effective potassium) through flattened P waves, prolonged PR interval, and a
widening QRS complex with an upright sine-wave T complex to asystole.

### Monitoring during arrest

When pulses cease, arterial pressure falls toward mean systemic filling
pressure, systolic and diastolic numerics blank after a brief hold, and MAP
follows static catheter pressure. During chest compressions, arterial pressure
tracks each compression, ECG shows compression artifact, and pulse oximetry
remains blank.

Chest compressions couple directly into [respiratory
mechanics](../anasim/physiology/resp_mech.py) and
[ventilation](../anasim/machine/ventilator.py) as a rhythmic thoracic pressure
pulse. Below end-expiratory equilibrium, respiratory stiffness rises (bench and
OHCA model in Cordioli 2016) and intrathoracic airway closure increases below
opening pressure (Grieco 2019). Compressions raise peak pressure and clip
inspiratory flow at `Pmax` in VCV (human cadaver study in Orlob 2021), reverse
inspiratory flow during manual bagging (OHCA recordings in Duchatelet 2018),
lower delivered tidal volume in PCV and PRVC, and trigger premature breaths
when ventilator triggers are left sensitive (bench manikin study in Neumamm
2021, extended in AnaSim to PSV and SIMV). Spirometry reports net tidal
excursion rather than summed oscillatory flow (Orlob 2021) and blanks plateau
pressure, while the intrathoracic pressure passed to hemodynamics excludes
extrapulmonary compression spikes but retains the venous-return cost of PEEP,
hyperventilation, and thoracic distension (porcine distension hemodynamics in
Lesimple 2022).

In [capnography](../anasim/monitors/capno.py), bidirectional flow produces CO₂
oscillations timed with each compression. Airway closure at low PEEP (Grieco
2019) or elevated closing pressure in obesity blunts recoil oscillations until
PEEP exceeds closing pressure, while large tidal volumes delay oscillations
until lung volume nears FRC (Lesimple 2022). Numeric EtCO₂ follows maximal
expired CO₂ between breaths (Grieco 2019); during apneic compression-only CPR,
passive strokes smaller than dead space produce low CO₂ ripples (Deakin 2007),
while AnaSim's spirometry reads zero and apnea timeout blanks numeric EtCO₂.

Noninvasive blood pressure (NIBP) raises cuff inflation pressure when systolic
pressure is high and averages pressure over the deflation sweep. Measurement
fails when VF, asystole, PEA, profound hypotension, or chest compressions
disrupt the sweep, while atrial fibrillation adds scatter and occasional failure
while keeping DBP at or below MAP and SBP. A failed reading blanks displayed
NIBP values, triggers a low-MAP alarm, and automatically retries after 20
seconds.

### Airway pressure, flow, and gases

Spirometry measures ventilator, bag, and unassisted breaths. Expired tidal
volume comes from completed breaths, respiratory rate and minute ventilation
average across a rolling window at least as long as the mandatory breath
period, and mandatory peak pressure holds across interposed spontaneous
breaths. Pressure-volume and flow-volume loops use synchronized pressure, flow,
and volume and clear when the airway disconnects. Plateau pressure is reported
only during an inspiratory pause with negligible flow and relaxed respiratory
muscles. Displayed PEEP reflects airway pressure, while intrinsic PEEP affects
plateau pressure and residual flow.

[Capnography](../anasim/monitors/capno.py) models dead-space transit time and
analyzer delay. Phase III upslope scales with EtCO₂ across breaths and steepens
with obstructive airway disease (Yaron 1996), a rolling peak window holds
full-breath readings across shallow interposed breaths, shallow sub-dead-space
breaths refresh the trace without reporting false full-breath readings, and the
no-breath timeout extends automatically at slow rates. End-tidal oxygen and
anesthetic values hold the last exhaled sample during brief apnea and clear
when exhaled gas stops, while staying live during ventilated zero-flow arrest
even when EtCO₂ reads zero; gas at the sampling port decays without flow. Once
an airway is connected or assisted ventilation runs, EtCO₂ alarms activate
after the first breath or after apnea timeout, with the low EtCO₂ limit lowered
during pregnancy to match expected PaCO₂. Gas leaking around a facemask misses
the sampling port, so large leaks flatten the capnogram.

Without an airway device, impedance respiratory rate counts breathing efforts
even during obstruction. Whenever the ventilator is off (including manual mode
with an open APL valve or failed mask seal), a low respiratory rate alarm
sounds after 15 seconds below 5 breaths/min. A low FiO₂ alarm triggers below
25% while the patient is connected to a sealed breathing circuit and stays
silent when disconnected on room air or with a zero mask seal. The gas monitor
displays end-tidal age-adjusted MAC (without pregnancy scaling), whereas
anesthetic effects depend on brain MAC.

### BIS and neuromuscular monitoring

BIS applies Eleveld's age-dependent delay to propofol while routing sevoflurane,
adjuncts, temperature, and arousal shifts through an age-independent monitor
calculation delay capped at 18 seconds (matching the lower end of the 14 to 66
second delays measured on artificial EEG steps in Pilge 2006), and sampled
hypnotic sensitivity affects BIS under both propofol and sevoflurane. Cortical
failure from cerebral ischemia appears on BIS within 20 seconds. High-BIS
alarms apply only when the patient is paralyzed or under active anesthetic
maintenance, and stay silent during rapid induction once the patient is already
deeply unconscious while the delayed display catches up.

The [neuromuscular monitor](../anasim/monitors/neuromuscular.py) measures TOF,
post-tetanic count (PTC; Viby-Mogensen 1981), or single twitch on demand or on
an interval. TOF reports twitch count and, when all four twitches return, the
T4/T1 ratio rounded down to a whole percentage so values below 90% never display
as 90%, alongside depressed T1 height so recovery is not marked complete while
T1 remains below 90%. Weak-twitch ratios are flagged, PTC counts 0 to 15
post-tetanic twitches during deep block and allows PTC soon after a zero TOF
count (using a 60-second post-tetanic lockout for simulation usability, shorter
than the 3-minute minimum interval recommended in Fuchs-Buder 2007), and single
twitch reports T1 relative to an unblocked baseline.

### Arterial blood gases

An [arterial blood gas](../anasim/monitors/blood_gas.py) records arterial
values at the time of the draw and reports the result 2 minutes later. As in a
clinical blood gas analyzer, pH, PaCO₂, and PaO₂ are warmed anaerobically to
37 °C, so a hypothermic sample reads higher PaCO₂ and PaO₂ and lower pH than in
vivo. Reports include bicarbonate, standard base excess, lactate, sodium,
potassium, chloride, ionized calcium, SaO₂, hemoglobin, and FiO₂. Glucose is
not modeled. During extracorporeal support, samples can also be drawn from the right radial
artery (upper body), femoral artery (lower body), pre-oxygenator venous line, or
post-oxygenator outlet.

## Extracorporeal support and anticoagulation

### ECMO and cardiopulmonary bypass

Venovenous ECMO (VV ECMO), venoarterial ECMO (VA ECMO), and cardiopulmonary
bypass (CPB) live in [`anasim/extracorporeal/`](../anasim/extracorporeal/) and
couple into the shared blood-gas, hemodynamic, thermal, and pharmacokinetic
models.

A two-stage crossflow membrane oxygenator shares AnaSim's blood-gas chemistry
(simulation of Joyce 2018), conserving O₂, CO₂, and buffer base across the
native and membrane lungs. Oxygen uptake rises mainly with blood flow,
hemoglobin, and sweep FiO₂, whereas CO₂ removal scales with sweep gas flow up
to blood-flow and membrane-area limits (Schmidt 2013; porcine data of
Karagiannidis 2017).

- **VV ECMO** operates in series within the venous circulation with
  flow-ratio-dependent venous recirculation calibrated across dual-lumen,
  femoro-jugular, and femoro-femoral cannulation (clinical cohort of Palmér
  2016). Extracorporeal CO₂ clearance reduces spontaneous respiratory drive
  (Mauri 2016), and an optional pulmonary shunt control supports severe ARDS
  cases.
- **VA ECMO** diverts venous blood in parallel into the arterial tree,
  separating native LV output, pulmonary flow, and pump flow (simulation of
  Donker 2019). Higher retrograde flow raises aortic afterload, reduces native
  stroke volume (porcine data of Ostadal 2015), and elevates left atrial
  pressure unless relieved by lower pump flow (Aissaoui 2012), LV venting,
  inotropes, or volume unloading (Donker 2019). In peripheral femoral VA ECMO,
  antegrade LV output and retrograde pump flow meet at a flow-dependent aortic
  watershed (CFD simulation of Stevens 2017): failing native lungs desaturate
  right-radial, cerebral, and coronary blood while femoral blood stays
  hyperoxic (Harlequin syndrome), whereas central cannulation mixes both
  streams uniformly. Nonpulsatile arterial support keeps systemic and coronary
  perfusion at pump MAP, flattens the arterial waveform with blanked systolic
  and diastolic numerics, loses NIBP and pulse oximetry signals, and suppresses
  cardiac arrest.
- **CPB** adds central arterial return, passive venous drainage from stressed
  volume into an open hard-shell reservoir with a low-level ramp and alarm
  (Magder and De Varennes 1998), crystalloid prime hemodilution lowering blood
  viscosity and SVR (Gordon 1975), cold-induced viscosity rise (in vitro data
  of Eckmann 2000), active heat exchange into the core-periphery thermal model
  producing post-bypass afterdrop blunted by forced-air warming (Rajek 2000),
  alpha-stat versus pH-stat cerebral blood flow management (Murkin 1987),
  aortic cross-clamping with cardioplegic arrest, and optional bypass alarm
  suppression.

Circuit priming expands central volume and dilutes albumin, raising propofol's
unbound fraction (CPB study of Hiraoka 2004, extended to ECMO). Hypothermia
slows remifentanil and propofol clearance (Michelsen 2001; Leslie 1995) and
independently lowers BIS (CPB study of Mathew 2001, extended below 34 °C).
Sevoflurane transfers rapidly across microporous polypropylene CPB oxygenators
and minimally across plasma-tight polymethylpentene ECMO membranes (Prasser
2008), while lipophilic drugs undergo saturable, passivating circuit
sequestration that spares morphine (ex vivo data of Shekar 2012; clinical VV
ECMO study of Morales Castro 2025).

### Heparin, protamine, and ACT

[`coagulation.py`](../anasim/physiology/coagulation.py) models two-compartment
unfractionated heparin kinetics (`HeparinPK`) with saturable cellular uptake
plus linear renal elimination so half-life lengthens from roughly 30 minutes at
25 U/kg to 120 to 150 minutes at 300 U/kg (extrapolated from 25 to 75 U/kg
volunteer data in Bjornsson 1982). Activated clotting time (ACT) rises from a
120-second baseline to roughly 250 to 300 seconds at 100 U/kg and above 480
seconds at 300 U/kg, with mild calibrated hemodilution and hypothermia shifts
(Bull 1975) and a 60-second Labs analysis delay.

Protamine (`ProtaminePK`) neutralizes plasma heparin at 1 mg per 100 U and
clears rapidly (about 4.5-minute half-life after CPB in Butterworth 2002), so
peripheral heparin redistribution causes post-reversal rebound unless offset by
a low-dose protamine infusion (Teoh 2004); excess free protamine above roughly
1.3:1 paradoxically prolongs ACT (in vitro data of Mochizuki 1998). Rapid
protamine boluses trigger rate-dependent Type I vasodilation and venous pooling
that slow infusions avoid, while a separate Type III catastrophic pulmonary
vasoconstriction reaction spikes PVR and causes acute RV failure (Horrow 1985;
Lowenstein 1983). Therapeutic anticoagulation amplifies active hemorrhage.

## Cases and clinical actions

### Cases

Cases run in guided mode, which shows step instructions and live requirement
checks, or independent mode, which hides instructions and defers non-blocking
safety warnings to the debrief.

Slow-onset crises (malignant hyperthermia, sepsis, and hemorrhage), emergence,
and the extracorporeal and anticoagulation cases run a hidden pre-case
simulation phase before case time starts at 0:00. Sampled vitals from that
phase populate negative-time trend history, so those cases open with developing
physiological signs, accumulated occult blood loss, lighter anesthesia
alongside residual neuromuscular block, or active extracorporeal support. Acute
intraoperative crises (anaphylaxis, oxygen supply failure, and cardiac arrest)
begin shortly into ongoing surgery. Cues tied to monitor changes wait until
deterioration is visible, and initial ventricular fibrillation shocks are
refractory so learners practice multi-cycle resuscitation.

Crisis cases stay coupled to the underlying physiology rather than using
scripted shortcuts. Requesting surgical source control in sepsis logs the
action without clearing established vasoplegia, so fluid resuscitation and
norepinephrine remain necessary. Malignant hyperthermia and balanced emergence
start on routine low-flow air/oxygen maintenance so high-flow oxygen washout is
mechanically required; malignant hyperthermia accepts non-triggering TIVA
(propofol, midazolam, or ketamine) for maintenance, aligns surface cooling at
38 °C, and allows a brief grace window after dantrolene before flagging
persistent crisis signs. Oxygen supply failure equilibrates baseline FiO₂ so
the circuit low-FiO₂ alarm warns before arterial desaturation.

Guided checks count learner actions from case start, or from crisis onset for
crisis management, so prompt treatment does not require repeat dosing. Both
balanced and TIVA induction (manual or TCI) accept transitional BIS depth during
maintenance handover while induction doses equilibrate. Standard pressure
objectives require spontaneous circulation without compressions and verify both
displayed and true model MAP, requiring a current cuff reading within 60 seconds
when no arterial line is placed (a failed cuff reading during true hypotension
still satisfies checks at crisis onset); extracorporeal cases also accept
arterial pump flow of at least 1.0 L/min as systemic perfusion. Debrief
hypotension time sums periods with MAP below 65 mmHg or without spontaneous
circulation or adequate arterial extracorporeal flow (at least 1.0 L/min), and
debrief hypoxemia time uses upper-body arterial saturation during arterial
extracorporeal support. Milestone outcomes remain credited once achieved,
extubation outcomes are evaluated when the tube is removed, and remaining
outcomes must hold at case end.

### Clinical action checks

Attempting tracheal intubation triggers a brief laryngoscopy stimulus whether
or not placement succeeds. When central muscle strength is more than half
intact, placement fails during severe laryngospasm or, while systemic perfusion
continues, when the patient is responsive or lacks adequate laryngoscopy
tolerance or deep volatile anesthesia (Kimura 1994). During cardiac arrest or
nonperfusing rhythms without arterial extracorporeal support, placement succeeds
unless severe laryngospasm closes the vocal cords.

[`clinical_actions.py`](../anasim/core/clinical_actions.py) raises a warning
without blocking the action when:

- a tracheal tube is placed in a responsive, paralyzed patient with systemic
  perfusion
- a neuromuscular blocker is given to a responsive patient with systemic
  perfusion and without an induction dose in plasma
- succinylcholine is given during active or susceptible malignant hyperthermia,
  or during significant hyperkalemia
- the tracheal tube is removed during apnea, reactive Stage II anesthesia, or
  incomplete neuromuscular recovery (excluding non-fading dantrolene weakness)
- assisted ventilation is stopped during apnea without active extracorporeal
  gas exchange (blood flow at least 1.5 L/min and positive sweep gas)
- a defibrillator shock is delivered to a responsive patient with a perfusing
  native rhythm
- protamine is administered during active ECMO or cardiopulmonary bypass
- cardiopulmonary bypass is started with an activated clotting time below 400
  seconds
