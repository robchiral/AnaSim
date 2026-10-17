# Model references

These sources support model parameters, simulator calibrations, and scenario guidance.
The [architecture guide](ARCHITECTURE.md#model-notes) describes their use and
the model assumptions. PK and PD mean pharmacokinetics and pharmacodynamics.
BIS means bispectral index, MAC means minimum alveolar concentration, and
TCI means target-controlled infusion. Ventilation modes are defined in the
[README](../README.md#features).

## Hemodynamics and physiology

- Su et al. (2023). *Br J Anaesth*. [Mechanistic hemodynamic interaction model](https://pubmed.ncbi.nlm.nih.gov/37355412/).
- de Wit et al. (2016). *Br J Anaesth*. [Propofol, stressed volume, and resistance to venous return](https://pubmed.ncbi.nlm.nih.gov/27199311/).
- Rebet et al. (2016). *Eur J Anaesthesiol*. [Preload-dependent phenylephrine cardiac-output responses](https://pubmed.ncbi.nlm.nih.gov/27164015/).
- Clutter et al. (1980). *J Clin Invest*. [Epinephrine cardiovascular effects](https://pubmed.ncbi.nlm.nih.gov/6995479/).
- Ebert et al. (1995). *Anesth Analg*. [Sevoflurane cardiovascular responses](https://pubmed.ncbi.nlm.nih.gov/7486143/).
- Segeroth et al. (2023). *Eur Heart J Cardiovasc Imaging*. [Pulmonary transit time](https://pubmed.ncbi.nlm.nih.gov/36662127/).
- Koganov et al. (1997). *Crit Care Med*. [PEEP raises pulmonary vascular resistance](https://pubmed.ncbi.nlm.nih.gov/9187594/).
- Carlsson et al. (1985). *Acta Anaesthesiol Scand*. [Hypoxic pulmonary vasoconstriction](https://pubmed.ncbi.nlm.nih.gov/3993324/).
- Sessler (2000). *Anesthesiology*. [Perioperative heat balance](https://pubmed.ncbi.nlm.nih.gov/10691247/).
- Sessler (2016). *Lancet*. [Review of perioperative thermoregulation](https://pubmed.ncbi.nlm.nih.gov/26775126/).
- Matsukawa et al. (1995). *Anesthesiology*. [Heat redistribution after induction](https://pubmed.ncbi.nlm.nih.gov/7879935/).
- Frank et al. (1997). *JAMA*. [Perioperative thermoregulation](https://pubmed.ncbi.nlm.nih.gov/9087467/).
- Anderson et al. (2017). *Paediatr Anaesth*. [Phenylephrine PK](https://pubmed.ncbi.nlm.nih.gov/28868789/).
- Magnani et al. (1977). *J Int Med Res*. [Dobutamine dose response](https://pubmed.ncbi.nlm.nih.gov/838109/).
- Baim et al. (1983). *N Engl J Med*. [Milrinone hemodynamics](https://pubmed.ncbi.nlm.nih.gov/6888453/).
- Martin et al. (1990). *Acta Anaesthesiol Scand*. [Vascular resistance in septic shock](https://pubmed.ncbi.nlm.nih.gov/2389659/).
- Melo et al. (1999). *Crit Care*. [Vascular resistance in septic shock](https://pubmed.ncbi.nlm.nih.gov/11056727/).
- Meng et al. (2011). *Br J Anaesth*. [Phenylephrine and ephedrine boluses under propofol-remifentanil anesthesia](https://pubmed.ncbi.nlm.nih.gov/21642644/).
- Ebert et al. (1994). *Anesth Analg*. [Propofol and baroreflex sensitivity](https://pubmed.ncbi.nlm.nih.gov/8311293/).
- Sato et al. (2005). *Br J Anaesth*. [Baroreflex during propofol infusion](https://pubmed.ncbi.nlm.nih.gov/15722386/).
- Ebert et al. (1992). *Anesthesiology*. [Sympathetic responses and baroreflexes during propofol induction](https://pubmed.ncbi.nlm.nih.gov/1575340/).
- Sellgren et al. (1994). *Anesthesiology*. [Sympathetic nerve activity and baroreflexes during propofol anesthesia](https://pubmed.ncbi.nlm.nih.gov/8141450/).
- Laitinen et al. (2004). *J Appl Physiol*. [Age and autonomic responses to head-up tilt](https://pubmed.ncbi.nlm.nih.gov/14766788/).
- Monahan (2007). *Am J Physiol Regul Integr Comp Physiol*. [Aging and baroreflex function](https://pubmed.ncbi.nlm.nih.gov/17442786/).
- Kardos et al. (2001). *Hypertension*. [Spontaneous baroreflex sensitivity by age in 1134 adults](https://pubmed.ncbi.nlm.nih.gov/11244017/).
- Bullington et al. (1989). *Anesth Analg*. [Age and the sympathetic response to intubation](https://pubmed.ncbi.nlm.nih.gov/2541638/).
- Leenen et al. (2005). *Am J Physiol Heart Circ Physiol*. [Aging and cardiac responses to epinephrine](https://pubmed.ncbi.nlm.nih.gov/15626690/).
- Guinard, Mulroy, and Carpenter (1995). *Reg Anesth*. [Age and HR after IV epinephrine test doses](https://pubmed.ncbi.nlm.nih.gov/7547654/).
- Ebert et al. (1995). *Anesthesiology*. [Sympathetic and hemodynamic responses to sevoflurane and intubation](https://pubmed.ncbi.nlm.nih.gov/7605024/).
- Johnson et al. (2014). *J Appl Physiol*. [Hemodynamic responses to graded blood loss in awake humans](https://pubmed.ncbi.nlm.nih.gov/24876357/).
- Umehara et al. (2006). *Anesth Analg*. [Sevoflurane and baroreflex sensitivity](https://pubmed.ncbi.nlm.nih.gov/16368802/).
- Mort (2004). *J Clin Anesth*. [Hypoxemia and intubation-related cardiac arrest](https://pubmed.ncbi.nlm.nih.gov/15590254/).
- De Jong et al. (2018). *Crit Care Med*. [Intubation-related cardiac arrest](https://pubmed.ncbi.nlm.nih.gov/29261566/).
- Heffner et al. (2013). *Resuscitation*. [Cardiac arrest during emergency airway management](https://pubmed.ncbi.nlm.nih.gov/23911630/).
- de Keijzer et al. (2026). *Eur J Anaesthesiol*. [Norepinephrine response under general anesthesia](https://pubmed.ncbi.nlm.nih.gov/41481868/).
- Joachim et al. (2024). *Br J Clin Pharmacol*. [Peripheral norepinephrine bolus PK/PD during general anesthesia](https://pubmed.ncbi.nlm.nih.gov/39039779/).
- Stratton et al. (1985). *J Appl Physiol*. [Epinephrine infusion hemodynamics](https://pubmed.ncbi.nlm.nih.gov/3988675/).
- Freyschuss et al. (1986). *Clin Sci (Lond)*. [Arterial epinephrine concentrations and hemodynamics](https://pubmed.ncbi.nlm.nih.gov/3956110/).
- Takahashi et al. (2002). *Anesth Analg*. [IV epinephrine bolus response](https://pubmed.ncbi.nlm.nih.gov/11867404/).
- Bellissant et al. (2000). *Clin Pharmacol Ther*. [Pressor hyporesponsiveness in septic shock](https://pubmed.ncbi.nlm.nih.gov/11014411/).
- Margarson et al. (2002). *J Appl Physiol (1985)*. [Albumin leakage in septic shock](https://pubmed.ncbi.nlm.nih.gov/11960967/).
- Persichini et al. (2012). *Crit Care Med*. [Norepinephrine and mean systemic pressure](https://pubmed.ncbi.nlm.nih.gov/22926333/).
- Reid et al. (2003). *Clin Sci (Lond)*. [Urine after 2 L saline or Hartmann's](https://pubmed.ncbi.nlm.nih.gov/12519083/).
- Hahn (2010). *Anesthesiology*. [Crystalloid volume kinetics](https://pubmed.ncbi.nlm.nih.gov/20613481/).
- Hahn (2017). *Anesth Analg*. [Arterial pressure, age, and crystalloid elimination](https://pubmed.ncbi.nlm.nih.gov/28452823/).
- Hahn and Nemme (2020). *BMC Anesthesiol*. [Fluid retention after anesthesia induction](https://doi.org/10.1186/s12871-020-01001-1).
- Clark et al. (1997). *J Am Coll Cardiol*. [Atrial fibrillation and cardiac output](https://pubmed.ncbi.nlm.nih.gov/9316536/).
- Hardman et al. (1998). *Cardiovasc Res*. [Atrial fibrillation and stroke volume](https://pubmed.ncbi.nlm.nih.gov/9683909/).
- Kerr et al. (1998). *Am J Cardiol*. [Stroke-volume variability in atrial fibrillation](https://pubmed.ncbi.nlm.nih.gov/9874054/).
- Corino et al. (2015). *J Cardiovasc Electrophysiol*. [R-R variability in atrial fibrillation](https://pubmed.ncbi.nlm.nih.gov/25367150/).
- Hogue et al. (1996). *Anesthesiology*. [Post-induction bradycardia and atrial pacing](https://pubmed.ncbi.nlm.nih.gov/8694384/).

## Cardiovascular monitor models

- Mahdi, Clifford, and Payne (2017). *Physiol Meas*. [Synthetic arterial pressure waveform](https://pubmed.ncbi.nlm.nih.gov/28176674/).
- Franklin et al. (1997). *Circulation*. [Pulse pressure by age in the Framingham Heart Study](https://pubmed.ncbi.nlm.nih.gov/9236450/).
- Saugel et al. (2020). *Crit Care*. [Arterial catheter frequency response and damping](https://pubmed.ncbi.nlm.nih.gov/32331527/).

## Inhaled anesthetics

- Davis and Mapleson (1981). *Br J Anaesth*. [Physiological model of inhaled anesthetics](https://pubmed.ncbi.nlm.nih.gov/7225273/).
- Mapleson (1996). *Br J Anaesth*. [Age-related MAC formula](https://pubmed.ncbi.nlm.nih.gov/8777094/).
- Nickalls and Mapleson (2003). *Br J Anaesth*. [Sevoflurane MAC](https://pubmed.ncbi.nlm.nih.gov/12878613/).
- Yasuda et al. (1991). *Anesth Analg*. [Sevoflurane and isoflurane uptake and washout](https://pubmed.ncbi.nlm.nih.gov/1994760/).
- Carpenter et al. (1986). *Anesth Analg*. [Inhaled anesthetic kinetics in humans](https://pubmed.ncbi.nlm.nih.gov/3706798/).
- Goto et al. (2000). *Anesthesiology*. [Nitrous oxide MAC at awakening](https://pubmed.ncbi.nlm.nih.gov/11046204/).
- Katoh et al. (1997). *Br J Anaesth*. [Sevoflurane-nitrous oxide interaction at awakening](https://pubmed.ncbi.nlm.nih.gov/9389264/).

## Intravenous drug models

- Eleveld et al. (2017). *Anesthesiology*. [Remifentanil PK/PD](https://doi.org/10.1097/ALN.0000000000001634).
- Eleveld et al. (2018). *Br J Anaesth*. [Propofol PK/PD](https://pubmed.ncbi.nlm.nih.gov/29661412/).
- Eleveld et al. (2018). *Br J Anaesth*. [Propofol model corrigendum](https://doi.org/10.1016/j.bja.2018.05.045).
- Johnson et al. (2003). *Anesthesiology*. [Propofol PK/PD in hemorrhagic shock](https://pubmed.ncbi.nlm.nih.gov/12883414/).
- Johnson et al. (2001). *Anesthesiology*. [Remifentanil PK/PD in hemorrhagic shock](https://pubmed.ncbi.nlm.nih.gov/11176098/).
- Johnson et al. (2004). *Anesthesiology*. [Propofol after hemorrhagic shock and crystalloid resuscitation](https://pubmed.ncbi.nlm.nih.gov/15329589/).
- Kurita et al. (2011). *Br J Anaesth*. [Remifentanil and propofol across stages of hemorrhage](https://pubmed.ncbi.nlm.nih.gov/21862493/).
- Egan and Johnson (2020). *Anesth Analg*. [Intravenous anesthetics in hemorrhagic shock](https://pubmed.ncbi.nlm.nih.gov/32149755/).
- Vellinga et al. (2021). *Br J Anaesth*. [Prospective Eleveld propofol PK/PD validation](https://doi.org/10.1016/j.bja.2020.10.027).
- Coetzee et al. (2024). *Br J Anaesth*. [Agreement between Eleveld-predicted and measured BIS during TIVA](https://doi.org/10.1016/j.bja.2024.06.041).
- Sahinovic et al. (2020). *Anesth Analg*. [EEG monitor delay relative to Schnider-predicted propofol effect-site concentrations](https://doi.org/10.1213/ANE.0000000000004672).
- Dahaba et al. (2020). *J Pharmacokinet Pharmacodyn*. [Age covariates in propofol-BIS pharmacodynamics](https://doi.org/10.1007/s10928-020-09678-0).
- Struys et al. (2007). *Anesthesiology*. [Propofol administration rate and apparent plasma-effect-site equilibration](https://doi.org/10.1097/01.anes.0000278902.15505.f8).
- Doufas et al. (2004). *Anesthesiology*. [Propofol induction rate and individualized pharmacodynamics](https://doi.org/10.1097/00000542-200411000-00010).
- Servin et al. (1990). *Br J Anaesth*. [Propofol PK in cirrhosis](https://pubmed.ncbi.nlm.nih.gov/2223333/).
- Hiraoka et al. (2005). *Br J Clin Pharmacol*. [Renal extraction of propofol](https://pubmed.ncbi.nlm.nih.gov/16042671/).
- Dershwitz et al. (1996). *Anesthesiology*. [Remifentanil PK/PD in liver disease](https://pubmed.ncbi.nlm.nih.gov/8638835/).
- Hoke et al. (1997). *Anesthesiology*. [Remifentanil PK in renal failure](https://pubmed.ncbi.nlm.nih.gov/9316957/).
- Bae et al. (2020). *Br J Anaesth*. [Adult fentanyl PK](https://pubmed.ncbi.nlm.nih.gov/32861508/).
- Meissner et al. (2023). *Anesthesiology*. [Morphine and hydromorphone effects and respiratory time courses](https://pubmed.ncbi.nlm.nih.gov/37014986/).
- Meissner et al. (2025). *Br J Anaesth*. [Morphine and hydromorphone population PK](https://pubmed.ncbi.nlm.nih.gov/39592363/).
- Olofsen et al. (2026). *Br J Anaesth*. [Morphine and hydromorphone endpoint-specific PD](https://pubmed.ncbi.nlm.nih.gov/41656122/), [supplement](https://ars.els-cdn.com/content/image/1-s2.0-S0007091225009134-mmc1.docx), and [corrected doses and pupil equation](https://pubmed.ncbi.nlm.nih.gov/42150960/).
- Hinderling et al. (1985). *J Pharm Sci*. [Atropine PK](https://pubmed.ncbi.nlm.nih.gov/4032240/) and [peripheral-compartment PD](https://pubmed.ncbi.nlm.nih.gov/4032241/).
- Yamaguchi et al. (1988). *Can J Anaesth*. [Atropine HR response during anesthesia](https://pubmed.ncbi.nlm.nih.gov/3168130/).
- Horiguchi and Nishikawa (2002). *Anesth Analg*. [Atropine dose-response awake and during propofol anesthesia](https://pubmed.ncbi.nlm.nih.gov/12145057/).
- Persky et al. (2004). *Br J Clin Pharmacol*. [Oral ephedrine PK and cardiovascular tolerance](https://doi.org/10.1111/j.1365-2125.2003.02062.x).
- Dings et al. (2026). *Pharmaceutics*. [Ephedrine hemodynamic time course during cesarean section](https://doi.org/10.3390/pharmaceutics18030296).
- Sjøen et al. (2026). *Acta Anaesthesiol Scand*. [Vasopressor boluses during propofol-remifentanil induction](https://pubmed.ncbi.nlm.nih.gov/41144827/) and [corrected norepinephrine 0.1 mcg/kg and phenylephrine 1 mcg/kg doses](https://pubmed.ncbi.nlm.nih.gov/41326972/).
- Roy et al. (2002). *Anesthesiology*. [Succinylcholine concentration-effect model](https://pubmed.ncbi.nlm.nih.gov/12411790/).
- Kopman et al. (2003). *Anesthesiology*. [Succinylcholine dose, onset, and twitch recovery](https://pubmed.ncbi.nlm.nih.gov/14576537/).
- Dhonneur et al. (1999). *Anesthesiology*. [Succinylcholine and rocuronium at the larynx, diaphragm, and adductor pollicis](https://pubmed.ncbi.nlm.nih.gov/10201662/).
- Scott and Stanski (1987). *J Pharmacol Exp Ther*. [Fentanyl effect-site equilibration and age](https://pubmed.ncbi.nlm.nih.gov/3100765/).
- McEwan et al. (1993). *Anesthesiology*. [Fentanyl reduction of isoflurane MAC](https://pubmed.ncbi.nlm.nih.gov/8489058/).
- Lang et al. (1996). *Anesthesiology*. [Remifentanil reduction of isoflurane MAC](https://pubmed.ncbi.nlm.nih.gov/8873541/).
- Albrecht et al. (1999). *Clin Pharmacol Ther*. [Midazolam PK/PD and age](https://pubmed.ncbi.nlm.nih.gov/10391668/).
- Vuyk et al. (2009). *Anesth Analg*. [Midazolam reduces propofol clearance](https://pubmed.ncbi.nlm.nih.gov/19372331/).
- Lichtenbelt et al. (2010). *Anesth Analg*. [Propofol reduces midazolam distribution and clearance](https://pubmed.ncbi.nlm.nih.gov/20435936/).
- Short, Plummer, and Chui (1992). *Br J Anaesth*. [Midazolam-propofol hypnotic synergy](https://pubmed.ncbi.nlm.nih.gov/1389820/).
- Arden, Holley, and Stanski (1986). *Anesthesiology*. [Etomidate PK/PD and age](https://pubmed.ncbi.nlm.nih.gov/3729056/).
- Van Hamme et al. (1978). *Anesthesiology*. [Etomidate PK](https://pubmed.ncbi.nlm.nih.gov/697083/).
- Kaneda et al. (2011). *J Clin Pharmacol*. [Etomidate PK/PD](https://pubmed.ncbi.nlm.nih.gov/20498288/).
- Valk and Struys (2021). *Clin Pharmacokinet*. [Etomidate pharmacology review](https://pubmed.ncbi.nlm.nih.gov/34060021/).
- Kamp et al. (2020). *Anesthesiology*. [Ketamine PK meta-analysis and three-compartment model](https://pubmed.ncbi.nlm.nih.gov/32997732/).
- Kamp et al. (2021). *Br J Anaesth*. [Stereoselective ketamine effects on cardiac output](https://pubmed.ncbi.nlm.nih.gov/33896589/).
- Idvall et al. (1979). *Br J Anaesth*. [Ketamine anesthesia and hemodynamics](https://pubmed.ncbi.nlm.nih.gov/526385/).
- Tweed, Minuck, and Mymin (1972). *Anesthesiology*. [Circulatory responses to ketamine](https://pubmed.ncbi.nlm.nih.gov/4652778/).
- Akine et al. (2001). *Auton Neurosci*. [Ketamine and propofol depress baroreflex sensitivity](https://pubmed.ncbi.nlm.nih.gov/11476280/).
- Huang et al. (2020). *J Physiol*. [Low-dose ketamine and sympathetic responses to central hypovolemia](https://pubmed.ncbi.nlm.nih.gov/33084081/).
- Bourke, Malit, and Smith (1987). *Anesthesiology*. [Ketamine and CO₂ response](https://pubmed.ncbi.nlm.nih.gov/3101549/).
- Foong et al. (2025). *Pharm Res*. [Lidocaine population PK in surgical patients](https://pubmed.ncbi.nlm.nih.gov/40021547/).
- Himes, DiFazio, and Burney (1977). *Anesthesiology*. [Lidocaine and anesthetic requirement](https://pubmed.ncbi.nlm.nih.gov/911052/).
- Hans et al. (2010). *Br J Anaesth*. [Lidocaine reduces propofol requirements during stimulation](https://pubmed.ncbi.nlm.nih.gov/20650919/).
- Altermatt et al. (2012). *Br J Anaesth*. [Lidocaine during propofol maintenance and recovery](https://pubmed.ncbi.nlm.nih.gov/22490315/).
- Qin et al. (2025). *Indian J Anaesth*. [Meta-analysis of IV lidocaine and the intubation response](https://pubmed.ncbi.nlm.nih.gov/40800699/).
- Wilson, Meiklejohn, and Smith (1991). *Anaesthesia*. [Lidocaine timing and the intubation response](https://pubmed.ncbi.nlm.nih.gov/2014891/).
- Tam, Chung, and Campbell (1987). *Anesth Analg*. [Lidocaine timing before intubation](https://pubmed.ncbi.nlm.nih.gov/3631567/).
- Okuda et al. (1990). *J Anesth*. [Lidocaine plasma concentration and timing before intubation](https://pubmed.ncbi.nlm.nih.gov/15236000/).
- Wierda et al. (1991). *Can J Anaesth*. [Rocuronium PK](https://pubmed.ncbi.nlm.nih.gov/1829656/).
- Khuenl-Brady and Sparr (1996). *Clin Pharmacokinet*. [Rocuronium clinical PK review](https://pubmed.ncbi.nlm.nih.gov/8877248/).
- Masui et al. (2018). *J Anesth*. [Rocuronium PD covariates](https://pubmed.ncbi.nlm.nih.gov/30099599/).
- Sasakawa et al. (2016). *J Anesth*. [Predictive ability of six rocuronium PK models](https://pubmed.ncbi.nlm.nih.gov/27098255/).
- Varrique et al. (2016). *J Pharm Pharmacol*. [Rocuronium PK/PD in young and elderly patients](https://pubmed.ncbi.nlm.nih.gov/27545305/).
- Magorian et al. (1995). *Anesth Analg*. [Rocuronium PK in liver disease](https://pubmed.ncbi.nlm.nih.gov/7893030/).
- Robertson et al. (2005). *Eur J Anaesthesiol*. [Rocuronium PK in renal failure](https://pubmed.ncbi.nlm.nih.gov/15816565/).
- Ensinger et al. (1992). *Eur J Anaesthesiol*. [Arterial epinephrine clearance](https://pubmed.ncbi.nlm.nih.gov/1425612/).
- Ensinger et al. (1992). *Eur J Clin Pharmacol*. [Arterial and peripheral venous catecholamine concentrations](https://pubmed.ncbi.nlm.nih.gov/1425886/).
- Li et al. (2024). *Clin Pharmacokinet*. [Norepinephrine PK with propofol interaction](https://pubmed.ncbi.nlm.nih.gov/39465453/).
- Ploeger et al. (2009). *Anesthesiology*. [Sugammadex PK/PD modeling](https://pubmed.ncbi.nlm.nih.gov/19104176/).
- Kleijn et al. (2011). *Br J Clin Pharmacol*. [Sugammadex PK/PD](https://pubmed.ncbi.nlm.nih.gov/21535448/).
- Pühringer et al. (2010). *Br J Anaesth*. [Sugammadex reversal times](https://pubmed.ncbi.nlm.nih.gov/20876699/).
- Cantineau et al. (1994). *Anesthesiology*. [Rocuronium effects at the diaphragm](https://pubmed.ncbi.nlm.nih.gov/8092503/).
- Plaud et al. (1995). *Clin Pharmacol Ther*. [Rocuronium laryngeal effect-site kinetics](https://pubmed.ncbi.nlm.nih.gov/7648768/).
- Wright et al. (1994). *Anesthesiology*. [Laryngeal sensitivity to rocuronium](https://pubmed.ncbi.nlm.nih.gov/7978469/).
- Eikermann et al. (2003). *Anesthesiology*. [Respiratory recovery after neuromuscular block](https://pubmed.ncbi.nlm.nih.gov/12766640/).
- Eikermann et al. (2007). *Am J Respir Crit Care Med*. [Airway collapse during partial neuromuscular block](https://pubmed.ncbi.nlm.nih.gov/17023729/).
- Herbstreit, Peters, and Eikermann (2009). *Anesthesiology*. [Residual block and pharyngeal closing pressure](https://pubmed.ncbi.nlm.nih.gov/19417617/).
- Fiset et al. (1991). *Can J Anaesth*. [Nitrous oxide and neuromuscular block](https://pubmed.ncbi.nlm.nih.gov/1683819/).
- Nguyen-Lee et al. (2018). *Curr Anesthesiol Rep*. [Sugammadex PK review](https://doi.org/10.1007/s40140-018-0266-5).
- Bouillon et al. (2004). *Anesthesiology*. [Propofol-remifentanil interaction for hypnosis, BIS, and tolerance of laryngoscopy](https://pubmed.ncbi.nlm.nih.gov/15166553/).
- Hannivoort et al. (2016). *Br J Anaesth*. [Propofol-sevoflurane-remifentanil interaction for laryngoscopy](https://doi.org/10.1093/bja/aew060).
- Kuizenga et al. (2019). *Anesthesiology*. [Propofol and sevoflurane with remifentanil for clinical responses](https://doi.org/10.1097/ALN.0000000000002966).
- Albertin et al. (2005). *Anesth Analg*. [Remifentanil blunting the hemodynamic response to intubation and incision](https://pubmed.ncbi.nlm.nih.gov/15976218/).
- Schnider et al. (1999). *Anesthesiology*. [Age and propofol Ce50 for loss of consciousness](https://pubmed.ncbi.nlm.nih.gov/10360845/).
- Lysakowski et al. (2009). *Br J Anaesth*. [BIS at propofol loss of consciousness in young and elderly patients](https://pubmed.ncbi.nlm.nih.gov/19542103/).
- Kazama et al. (1999). *Anesthesiology*. [Propofol equilibration for SBP and BIS by age](https://pubmed.ncbi.nlm.nih.gov/10360846/).
- Minto et al. (1997). *Anesthesiology*. [Age and remifentanil EEG potency](https://pubmed.ncbi.nlm.nih.gov/9009935/).
- Kim et al. (2018). *Clin Interv Aging*. [Age and remifentanil for emergence cough](https://pubmed.ncbi.nlm.nih.gov/29881261/).
- FDA NDA 203826 Clinical Pharmacology Review (2012). [Phenylephrine PK](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2012/203826_phenylephrine_toc.cfm).
- Hengstmann and Goronzy (1982). *Eur J Clin Pharmacol*. [Phenylephrine PK](https://pubmed.ncbi.nlm.nih.gov/7056280/).
- Vasopressin injection label (DailyMed). [Vasopressin PK](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=971f9b1c-6094-4f80-920b-cb5d7e62950a).
- Kates and Leier (1978). *Clin Pharmacol Ther*. [Dobutamine PK in heart failure](https://pubmed.ncbi.nlm.nih.gov/699477/).
- Daly et al. (1997). *Am J Cardiol*. [Dobutamine dose-concentration relationship](https://pubmed.ncbi.nlm.nih.gov/9165162/).
- Dobutamine injection label (Clinical Pharmacology). [Dobutamine PK](https://www.pfizermedicalinformation.com/patient/dobutamine/clinical-pharmacology).
- Sum et al. (1983). *Clin Pharmacol Ther*. [Esmolol PK and beta blockade](https://pubmed.ncbi.nlm.nih.gov/6617063/).
- Wiest and Haney (2012). *Clin Pharmacokinet*. [Esmolol pharmacology review](https://pubmed.ncbi.nlm.nih.gov/22515557/).
- Abernethy et al. (1987). *Am J Cardiol*. [Labetalol PK/PD and age](https://pubmed.ncbi.nlm.nih.gov/3661438/).
- Hafsa et al. (2022). *Pharmaceutics*. [Labetalol kinetics and antagonist potency](https://pubmed.ncbi.nlm.nih.gov/36365181/).
- Ali-Melkkilä, Kanto, and Iisalo (1993). *Acta Anaesthesiol Scand*. [Anticholinergic PK/PD review](https://pubmed.ncbi.nlm.nih.gov/8249551/).
- Du et al. (2025). *J Drug Deliv Sci Technol*. [Glycopyrrolate population PK](https://doi.org/10.1016/j.jddst.2025.106692).
- Milrinone lactate injection label (DailyMed). [Milrinone PK](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=88f78780-399f-4780-be19-205739db1682&version=21).

## Respiratory control and BIS

- García-Río et al. (2007). *Respir Med*. [Age and hypoxic and hypercapnic inspiratory drive](https://pubmed.ncbi.nlm.nih.gov/17643972/).
- Egan et al. (2004). *Br J Anaesth*. [Remifentanil bolus respiratory effects by age](https://pubmed.ncbi.nlm.nih.gov/14970134/).
- Dash, Korman, and Bassingthwaighte (2016). *Eur J Appl Physiol*. [Hemoglobin dissociation curves across oxygen tension and temperature](https://pubmed.ncbi.nlm.nih.gov/26298270/).
- Bouillon et al. (2004). *Anesthesiology*. [Dynamic propofol ventilatory depression](https://pubmed.ncbi.nlm.nih.gov/14739795/).
- Olofsen et al. (2010). *Anesthesiology*. [Dynamic remifentanil respiratory effects in awake and propofol-sedated volunteers](https://pubmed.ncbi.nlm.nih.gov/20461001/).
- Kanazawa et al. (2017). *J Anesth*. [Volatile anesthetics and BIS](https://pubmed.ncbi.nlm.nih.gov/28791477/).
- Ryu et al. (2018). *Anesthesiology*. [BIS and surgical pleth index during stimulation](https://pubmed.ncbi.nlm.nih.gov/29509579/).
- Ryu et al. (2018). *Br J Anaesth*. [Remifentanil requirements with volatile anesthetics](https://pubmed.ncbi.nlm.nih.gov/30336856/).
- Paraskeva et al. (2005). *J Clin Anesth*. [Sevoflurane and BIS](https://pubmed.ncbi.nlm.nih.gov/16427526/).
- Schwab et al. (2004). *Anesth Analg*. [BIS response to sevoflurane](https://pubmed.ncbi.nlm.nih.gov/15562061/).
- Olofsen et al. (2002). *Anesthesiology*. [Remifentanil-sevoflurane BIS interaction](https://pubmed.ncbi.nlm.nih.gov/11873028/).
- Schumacher et al. (2009). *Anesthesiology*. [Additive propofol-sevoflurane interaction on BIS](https://pubmed.ncbi.nlm.nih.gov/19741484/).
- Barr et al. (1999). *Br J Anaesth*. [Nitrous oxide hypnosis and BIS](https://pubmed.ncbi.nlm.nih.gov/10562773/).
- Hirota et al. (1999). *Eur J Anaesthesiol*. [Nitrous oxide and BIS](https://pubmed.ncbi.nlm.nih.gov/10713872/).
- Ozcan et al. (2010). *J Neurosurg Anesthesiol*. [Nitrous oxide effects on BIS and entropy](https://pubmed.ncbi.nlm.nih.gov/20844378/).
- Babenco et al. (2000). *Anesthesiology*. [Opioid effects on ventilatory control](https://pubmed.ncbi.nlm.nih.gov/10691225/).
- Lee et al. (2011). *Korean J Anesthesiol*. [Propofol effect-site EC₅₀ for respiratory depression](https://pubmed.ncbi.nlm.nih.gov/21927681/).
- Blouin et al. (1993). *Anesthesiology*. [Propofol depresses hypoxic ventilatory response](https://pubmed.ncbi.nlm.nih.gov/8267192/).
- Nieuwenhuijs et al. (2000). *Anesthesiology*. [Acute and sustained hypoxic responses during propofol sedation](https://pubmed.ncbi.nlm.nih.gov/10638898/).
- Sarton et al. (1996). *Anesthesiology*. [Sevoflurane, stimulation, and hypoxic ventilatory response](https://pubmed.ncbi.nlm.nih.gov/8712445/).
- Nieuwenhuijs et al. (2001). *Anesthesiology*. [Propofol effects on ventilatory control](https://pubmed.ncbi.nlm.nih.gov/11605929/).
- Glass et al. (1999). *Anesthesiology*. [Remifentanil ventilatory depression and CO₂](https://pubmed.ncbi.nlm.nih.gov/10360852/).
- Pandit et al. (1999). *Br J Anaesth*. [Sevoflurane and hypercapnic ventilatory response](https://pubmed.ncbi.nlm.nih.gov/10618930/).
- Doi and Ikeda (1987). *Anesth Analg*. [Sevoflurane and CO₂ response](https://pubmed.ncbi.nlm.nih.gov/3826666/).
- Hickey et al. (1971). *Anesthesiology*. [Apneic threshold under anesthesia](https://pubmed.ncbi.nlm.nih.gov/4932620/).
- Patrick et al. (1995). *J Appl Physiol*. [Awake breathing during hypocapnia](https://pubmed.ncbi.nlm.nih.gov/8847274/).
- Georgopoulos et al. (1997). *Am J Respir Crit Care Med*. [CO₂ feedback during awake pressure support](https://pubmed.ncbi.nlm.nih.gov/9196108/).
- Duffin (2011). *Respir Physiol Neurobiol*. [Ventilatory response modeling](https://pubmed.ncbi.nlm.nih.gov/21514404/).
- Bissinger et al. (1993). *Anasthesiol Intensivmed Notfallmed Schmerzther*. [Curare clefts and neuromuscular monitoring](https://pubmed.ncbi.nlm.nih.gov/7902740/).
- Russell et al. (1990). *Can J Anaesth*. [Arterial to end-tidal CO₂ gradient during postoperative support](https://pubmed.ncbi.nlm.nih.gov/2115404/).
- Lujan et al. (2008). *Med Sci Monit*. [Arterial to end-tidal CO₂ gradient in airway obstruction](https://pubmed.ncbi.nlm.nih.gov/18758420/).
- Idris et al. (1994). *Ann Emerg Med*. [End-tidal CO₂ during extremely low cardiac output](https://pubmed.ncbi.nlm.nih.gov/8135436/).
- Kim et al. (2019). *Am J Emerg Med*. [Arterial to end-tidal CO₂ gap after cardiac arrest](https://pubmed.ncbi.nlm.nih.gov/29685358/).
- Poorzargar et al. (2022). *J Clin Monit Comput*. [Pulse oximetry in poor perfusion](https://pubmed.ncbi.nlm.nih.gov/35119597/).
- Broome et al. (1992). *Anaesthesia*. [Finger pulse-oximeter response delay during anesthesia](https://pubmed.ncbi.nlm.nih.gov/1536395/).
- Tanaka et al. (2014). *J Clin Monit Comput*. [Capnographic detection of respiratory pauses](https://pubmed.ncbi.nlm.nih.gov/24420342/).
- Benumof et al. (1997). *Anesthesiology*. [Desaturation after preoxygenation](https://pubmed.ncbi.nlm.nih.gov/9357902/).
- Nimmagadda et al. (2017). *Anesth Analg*. [Preoxygenation endpoints](https://pubmed.ncbi.nlm.nih.gov/28099321/).
- Hardman et al. (2000). *Anesth Analg*. [Hypoxemia during apnea](https://pubmed.ncbi.nlm.nih.gov/10702447/).
- Laviola et al. (2020). *Br J Anaesth*. [Oxygen fraction during passive airway rescue](https://pubmed.ncbi.nlm.nih.gov/32008701/).
- Edmark et al. (2003). *Anesthesiology*. [Inspired O₂ during induction, atelectasis, and apnea time](https://pubmed.ncbi.nlm.nih.gov/12502975/).
- Tokics et al. (1996). *J Appl Physiol*. [Atelectasis and shunt during anesthesia](https://pubmed.ncbi.nlm.nih.gov/8904605/).
- Hedenstierna and Edmark (2015). *Best Pract Res Clin Anaesthesiol*. [FRC, atelectasis, and gas exchange under anesthesia](https://pubmed.ncbi.nlm.nih.gov/26643094/).
- Gunnarsson et al. (1991). *Br J Anaesth*. [Age, low V/Q, and atelectasis during anesthesia](https://pubmed.ncbi.nlm.nih.gov/2025468/).
- Crapo et al. (1999). *Am J Respir Crit Care Med*. [Arterial blood gas and A-a gradient reference values](https://pubmed.ncbi.nlm.nih.gov/10556115/).
- Eastwood et al. (2005). *Anesthesiology*. [Propofol and airway closing pressure](https://pubmed.ncbi.nlm.nih.gov/16129969/).
- Hillman et al. (2009). *Anesthesiology*. [Airway collapse at loss of consciousness](https://pubmed.ncbi.nlm.nih.gov/19512872/).
- Larsson et al. (1985). *J Allergy Clin Immunol*. [Epinephrine relief of bronchoconstriction](https://pubmed.ncbi.nlm.nih.gov/3989143/).
- Farmery and Roe (1996). *Br J Anaesth*. [Model of oxyhemoglobin desaturation during apnea](https://pubmed.ncbi.nlm.nih.gov/8777112/).
- Stock et al. (1989). *J Clin Anesth*. [CO₂ accumulation during airway obstruction](https://pubmed.ncbi.nlm.nih.gov/2516732/).
- Kobayashi et al. (1994). *Masui*. [Blood gas changes during apnea](https://pubmed.ncbi.nlm.nih.gov/7933492/).
- Miyamura et al. (1980). *Jpn J Physiol*. [Hypercapnic ventilatory response variability](https://pubmed.ncbi.nlm.nih.gov/6790801/).
- Christie et al. (1992). *Anesth Analg*. [Pressure support during general anesthesia](https://pubmed.ncbi.nlm.nih.gov/1632530/).
- Lim et al. (2012). *Paediatr Anaesth*. [Pediatric pressure support with an LMA](https://pubmed.ncbi.nlm.nih.gov/22380745/).
- Capdevila et al. (2014). *PLoS One*. [Pressure support and LMA recovery](https://pubmed.ncbi.nlm.nih.gov/25536515/).

## Ventilator waveforms

- GE Healthcare. Avance Carestation 8.X participant guide, p. 7.9. [PCV-VG initialization, pressure range, and breath-to-breath adjustment](https://www.gehealthcare.com/-/jssmedia/60148736cbc34131bbebbec747b696af.pdf).
- Giosa et al. (2024). *J Clin Med*. [Plateau pressure, residual flow, and patient effort](https://pubmed.ncbi.nlm.nih.gov/39685913/).
- Lee et al. (2022). *Sci Data*. [VitalDB waveforms and numerics](https://pubmed.ncbi.nlm.nih.gov/35676300/).
- Jonson et al. (1993). *J Appl Physiol*. [Human tissue viscoelasticity](https://pubmed.ncbi.nlm.nih.gov/8376259/).
- D'Angelo et al. (1989). *J Appl Physiol*. [Respiratory stress adaptation](https://pubmed.ncbi.nlm.nih.gov/2606863/).
- Pelosi et al. (1998). *Anesth Analg*. [Body mass index, functional residual capacity, and compliance under anesthesia](https://pubmed.ncbi.nlm.nih.gov/9728848/).
- Bates and Irvin (2002). *J Appl Physiol*. [Recruitment and derecruitment dynamics](https://pubmed.ncbi.nlm.nih.gov/12133882/).
- Rothen et al. (1993). *Br J Anaesth*. [Inflation pressure and CT atelectasis](https://pubmed.ncbi.nlm.nih.gov/8280539/).
- Rothen et al. (1995). *Anesthesiology*. [Oxygen concentration and recurrent atelectasis](https://pubmed.ncbi.nlm.nih.gov/7717553/).
- Rothen et al. (1999). *Br J Anaesth*. [Time course of recruitment](https://pubmed.ncbi.nlm.nih.gov/10472221/).
- Malo, Ali, and Wood (1984). *J Appl Physiol*. [PEEP and shunt in pulmonary edema](https://pubmed.ncbi.nlm.nih.gov/6389451/).
- Tobin et al. (1983). *Chest*. [Breathing pattern of normal subjects](https://pubmed.ncbi.nlm.nih.gov/6872603/).
- Clark and von Euler (1972). *J Physiol*. [Volume feedback on inspiratory duration](https://pmc.ncbi.nlm.nih.gov/articles/PMC1331381/).
- Leung, Jubran, and Tobin (1997). *Am J Respir Crit Care Med*. [SIMV and pressure-support effort unloading](https://pubmed.ncbi.nlm.nih.gov/9196100/).
- Graves et al. (1986). *Am J Physiol*. [Patient-ventilator synchronization under anesthesia](https://pubmed.ncbi.nlm.nih.gov/3706575/).
- Dräger. Evita 4 SW 4.n instructions for use, pp. 188-191. [SIMV and pressure-support controls](https://www.draeger.com/Content/Documents/Content/IfU_Evita_4_SW_4.n_EN_9039485.pdf).
- Feigenwinter and Zbinden (1991). *Anaesthesist*. [Circle-system resistance](https://pubmed.ncbi.nlm.nih.gov/1952036/).
- Polacheck et al. (1980). *J Appl Physiol*. [Vagal inflation feedback under anesthesia](https://pubmed.ncbi.nlm.nih.gov/7440275/).
- Sydow et al. (1993). *Intensive Care Med*. [Respiratory mechanics in status asthmaticus](https://pubmed.ncbi.nlm.nih.gov/8294630/).
- Okayama et al. (1991). *J Asthma*. [Airway resistance in status asthmaticus](https://pubmed.ncbi.nlm.nih.gov/2010425/).
- Mead et al. (1967). *J Appl Physiol*. [Maximal expiratory flow](https://pubmed.ncbi.nlm.nih.gov/6017658/).
- Boczkowski et al. (1997). *Am J Respir Crit Care Med*. [Tidal expiratory flow limitation in asthma](https://pubmed.ncbi.nlm.nih.gov/9309989/).
- Tuxen and Lane (1987). *Am Rev Respir Dis*. [Expiratory time and obstructive hyperinflation](https://pubmed.ncbi.nlm.nih.gov/3662241/).

## Scenario guidance

- Blink Device Company (2024). [TwitchView operating manual, Rev N, pp. 14-16 and 20](https://www.blinkdc.com/hubfs/PN00211%20TwitchView%20IFU%20Rev%20N.pdf). TOF count/ratio, weak-response qualification, single twitch, baseline reference, and stimulation timing.
- Thilen et al. (2023). *Anesthesiology*. [Quantitative neuromuscular monitoring and train-of-four ratio ≥ 0.9 before extubation](https://pubmed.ncbi.nlm.nih.gov/36520073/).
- ANZAAG and ANZCA (2022). [Perioperative anaphylaxis, adult epinephrine doses, oxygen, and crystalloid resuscitation](https://www.anzca.edu.au/getContentAsset/d38d29ae-74f0-4136-8372-1d342c594e11/80feb437-d24d-46b8-a858-4a2a28b9b970/Anaphylaxis-Card-1-Adult-Immediate-Management-2022.pdf?language=en).
- Surviving Sepsis Campaign (2026). [Adult fluids, norepinephrine, antibiotics, and source control](https://sccm.org/clinical-resources/guidelines/guidelines/surviving-sepsis-campaign-international-guidelines-for-management-of-sepsis-and-septic-shock-2026).
- Kietaibl et al. (2023). *Eur J Anaesthesiol*. [Severe perioperative bleeding, hemostasis, blood products, and reassessment](https://pubmed.ncbi.nlm.nih.gov/36855941/).
- Xing et al. (2018). *Pain Med*. [Lidocaine pretreatment for propofol injection pain](https://pubmed.ncbi.nlm.nih.gov/28525614/).
- APSF (2019). [Oxygen pipeline failure, independent ventilation, and cylinder supply](https://www.apsf.org/article/nitrogen-contamination-of-operating-room-oxygen-pipeline/).

## Drug timing and TCI performance

- Hughes et al. (1992). *Anesthesiology*. [Propofol context-sensitive half-time](https://pubmed.ncbi.nlm.nih.gov/1539843/).
- Egan et al. (1993). *Anesthesiology*. [Remifentanil context-sensitive half-time](https://pubmed.ncbi.nlm.nih.gov/7902032/).
- Kapila et al. (1995). *Anesthesiology*. [Remifentanil offset and ventilation recovery](https://pubmed.ncbi.nlm.nih.gov/7486182/).
- Grundmann et al. (2001). *Acta Anaesthesiol Scand*. [Propofol-remifentanil recovery](https://pubmed.ncbi.nlm.nih.gov/11207468/).
- Kwon et al. (2018). *Korean J Anesthesiol*. [Hypercapnia and propofol emergence](https://pubmed.ncbi.nlm.nih.gov/29690757/).
- Magorian et al. (1993). *Anesthesiology*. [Rocuronium onset](https://pubmed.ncbi.nlm.nih.gov/7902034/).
- Varvel et al. (1992). *J Pharmacokinet Biopharm*. [TCI performance metrics](https://pubmed.ncbi.nlm.nih.gov/1588504/).
- Shafer and Gregg (1992). *J Pharmacokinet Biopharm*. [Effect-site and plasma-targeted TCI](https://pubmed.ncbi.nlm.nih.gov/1629794/).
