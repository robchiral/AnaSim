# Model references

These studies are the sources for the model parameters, calibrations, and
scenario guidance. The sections follow the
[architecture guide](ARCHITECTURE.md#model-notes), which explains how the
studies are used.

Abbreviations: PK, pharmacokinetics; PD, pharmacodynamics; BIS, bispectral
index; MAC, minimum alveolar concentration; TCI, target-controlled infusion;
TOF, train-of-four; SIMV, synchronized intermittent mandatory ventilation;
PCV-VG, pressure control with volume guarantee.

## Patient conditions

- Gobel et al. (1978). *Circulation*. [Rate-pressure product and myocardial oxygen consumption in angina](https://pubmed.ncbi.nlm.nih.gov/624164/).
- Wilson et al. (1987). *Circulation*. [Coronary lesion geometry and flow reserve](https://pubmed.ncbi.nlm.nih.gov/3829334/).
- Arthurs et al. (2016). *Am J Physiol Heart Circ Physiol*. [Coronary blood-flow control and myocardial oxygen balance](https://pubmed.ncbi.nlm.nih.gov/26945076/).
- Harasawa et al. (1992). *Circ Res*. [Basal metabolism and unloaded myocardial oxygen consumption](https://pubmed.ncbi.nlm.nih.gov/1628397/).
- Jeremy et al. (1986). *Cardiovasc Res*. [Left ventricular diastolic pressure and the coronary pressure-flow relation during vasodilation](https://pubmed.ncbi.nlm.nih.gov/3802128/).
- Nanto et al. (2001). *Jpn Circ J*. [Coronary zero-flow pressure in humans](https://pubmed.ncbi.nlm.nih.gov/11548878/).
- Nebout et al. (2013). *Ann Fr Anesth Reanim*. [BIS for early detection of cerebral hypoperfusion](https://pubmed.ncbi.nlm.nih.gov/23953319/).
- Astrup et al. (1981). *Stroke*. [Thresholds in cerebral ischemia: the ischemic penumbra](https://pubmed.ncbi.nlm.nih.gov/6272455/).
- Jones et al. (1981). *J Neurosurg*. [Thresholds of focal cerebral ischemia in awake monkeys](https://pubmed.ncbi.nlm.nih.gov/7241187/).
- de Vries et al. (1998). *Anesth Analg*. [Cerebral oxygen uptake and EEG during defibrillation threshold testing](https://pubmed.ncbi.nlm.nih.gov/9661538/).
- Westlake et al. (1955). *Q J Med*. [Carbon dioxide narcosis in emphysema](https://doi.org/10.1093/oxfordjournals.qjmed.a066724).
- Turnbull et al. (2008). *Anaesthesia*. [The influence of changes in end-tidal carbon dioxide upon the Bispectral Index](https://pubmed.ncbi.nlm.nih.gov/18412642/).
- Schwartzenberg et al. (2012). *J Am Coll Cardiol*. [Hemodynamic responses to vasodilation in reduced- and preserved-EF heart failure](https://pubmed.ncbi.nlm.nih.gov/22281246/).
- Westermann et al. (2008). *Circulation*. [Ventricular stiffness and filling during tachycardia in preserved-EF heart failure](https://pubmed.ncbi.nlm.nih.gov/18413502/).
- Wagner et al. (1977). *J Clin Invest*. [Ventilation-perfusion inequality in COPD](https://pubmed.ncbi.nlm.nih.gov/833271/).
- Dallmann et al. (2017). *Clin Pharmacokinet*. [Gestation-specific anatomical and physiological parameter regressions](https://pubmed.ncbi.nlm.nih.gov/28401479/).
- Contreras et al. (1991). *Am Rev Respir Dis*. [Ventilatory drive, lung volumes, and respiratory muscles during pregnancy](https://pubmed.ncbi.nlm.nih.gov/1928958/).
- Humphries et al. (2019). *J Matern Fetal Neonatal Med*. [Supine positioning and maternal hemodynamics in late pregnancy](https://pubmed.ncbi.nlm.nih.gov/29772936/).

## Intravenous drug models

### Propofol and remifentanil

- Eleveld et al. (2017). *Anesthesiology*. [Remifentanil PK/PD](https://pubmed.ncbi.nlm.nih.gov/28509794/).
- Eleveld et al. (2018). *Br J Anaesth*. [Propofol PK/PD](https://pubmed.ncbi.nlm.nih.gov/29661412/) and [correction](https://pubmed.ncbi.nlm.nih.gov/30032904/).
- Vellinga et al. (2021). *Br J Anaesth*. [Prospective Eleveld propofol PK/PD validation](https://pubmed.ncbi.nlm.nih.gov/33317804/).
- Kim et al. (2023). *J Pharmacokinet Pharmacodyn*. [Propofol PK/PD model and Eleveld validation in Korean elderly subjects](https://pubmed.ncbi.nlm.nih.gov/36522561/).
- Coetzee et al. (2024). *Br J Anaesth*. [Agreement between Eleveld-predicted and measured BIS during TIVA](https://pubmed.ncbi.nlm.nih.gov/39179443/).
- Sørensen et al. (2011). *Acta Anaesthesiol Scand*. [Onset time and hemodynamic response after thiopental versus propofol in the elderly](https://pubmed.ncbi.nlm.nih.gov/21342151/).
- Sahinovic et al. (2020). *Anesth Analg*. [EEG monitor delay relative to Schnider-predicted propofol effect-site concentrations](https://pubmed.ncbi.nlm.nih.gov/32925339/).
- Dahaba et al. (2020). *J Pharmacokinet Pharmacodyn*. [Age covariates in propofol-BIS pharmacodynamics](https://pubmed.ncbi.nlm.nih.gov/32100175/).
- Struys et al. (2007). *Anesthesiology*. [Propofol administration rate and apparent plasma-effect-site equilibration](https://pubmed.ncbi.nlm.nih.gov/17721240/).
- Doufas et al. (2004). *Anesthesiology*. [Propofol induction rate and individualized pharmacodynamics](https://pubmed.ncbi.nlm.nih.gov/15505446/).
- Servin et al. (1990). *Br J Anaesth*. [Propofol PK in cirrhosis](https://pubmed.ncbi.nlm.nih.gov/2223333/).
- Hiraoka et al. (2005). *Br J Clin Pharmacol*. [Renal extraction of propofol](https://pubmed.ncbi.nlm.nih.gov/16042671/).
- Dershwitz et al. (1996). *Anesthesiology*. [Remifentanil PK/PD in liver disease](https://pubmed.ncbi.nlm.nih.gov/8638835/).
- Hoke et al. (1997). *Anesthesiology*. [Remifentanil PK in renal failure](https://pubmed.ncbi.nlm.nih.gov/9316957/).
- Minto et al. (1997). *Anesthesiology*. [Age and remifentanil EEG potency](https://pubmed.ncbi.nlm.nih.gov/9009935/).
- Kim et al. (2018). *Clin Interv Aging*. [Age and remifentanil for emergence cough](https://pubmed.ncbi.nlm.nih.gov/29881261/).

### Hemorrhagic shock

- Johnson et al. (2003). *Anesthesiology*. [Propofol PK/PD in hemorrhagic shock](https://pubmed.ncbi.nlm.nih.gov/12883414/).
- Johnson et al. (2001). *Anesthesiology*. [Remifentanil PK/PD in hemorrhagic shock](https://pubmed.ncbi.nlm.nih.gov/11176098/).
- Johnson et al. (2004). *Anesthesiology*. [Propofol after hemorrhagic shock and crystalloid resuscitation](https://pubmed.ncbi.nlm.nih.gov/15329589/).
- Kurita et al. (2011). *Br J Anaesth*. [Remifentanil and propofol across stages of hemorrhage](https://pubmed.ncbi.nlm.nih.gov/21862493/).
- Egan and Johnson (2020). *Anesth Analg*. [Intravenous anesthetics in hemorrhagic shock](https://pubmed.ncbi.nlm.nih.gov/32149755/).

### Other opioids

- Bae et al. (2020). *Br J Anaesth*. [Adult fentanyl PK](https://pubmed.ncbi.nlm.nih.gov/32861508/).
- Scott and Stanski (1987). *J Pharmacol Exp Ther*. [Fentanyl effect-site equilibration and age](https://pubmed.ncbi.nlm.nih.gov/3100765/).
- McEwan et al. (1993). *Anesthesiology*. [Fentanyl reduction of isoflurane MAC](https://pubmed.ncbi.nlm.nih.gov/8489058/).
- Lang et al. (1996). *Anesthesiology*. [Remifentanil reduction of isoflurane MAC](https://pubmed.ncbi.nlm.nih.gov/8873541/).
- Meissner et al. (2023). *Anesthesiology*. [Morphine and hydromorphone effects and respiratory time courses](https://pubmed.ncbi.nlm.nih.gov/37014986/).
- Meissner et al. (2025). *Br J Anaesth*. [Morphine and hydromorphone population PK](https://pubmed.ncbi.nlm.nih.gov/39592363/).
- Olofsen et al. (2026). *Br J Anaesth*. [Morphine and hydromorphone PD for each endpoint](https://pubmed.ncbi.nlm.nih.gov/41656122/), [supplement](https://ars.els-cdn.com/content/image/1-s2.0-S0007091225009134-mmc1.docx), and [correction](https://pubmed.ncbi.nlm.nih.gov/42150960/).

### Midazolam and etomidate

- Albrecht et al. (1999). *Clin Pharmacol Ther*. [Midazolam PK/PD and age](https://pubmed.ncbi.nlm.nih.gov/10391668/).
- Vuyk et al. (2009). *Anesth Analg*. [Midazolam reduces propofol clearance](https://pubmed.ncbi.nlm.nih.gov/19372331/).
- Lichtenbelt et al. (2010). *Anesth Analg*. [Propofol reduces midazolam distribution and clearance](https://pubmed.ncbi.nlm.nih.gov/20435936/).
- Short et al. (1992). *Br J Anaesth*. [Midazolam-propofol hypnotic synergy](https://pubmed.ncbi.nlm.nih.gov/1389820/).
- Arden et al. (1986). *Anesthesiology*. [Etomidate PK/PD and age](https://pubmed.ncbi.nlm.nih.gov/3729056/).
- Van Hamme et al. (1978). *Anesthesiology*. [Etomidate PK](https://pubmed.ncbi.nlm.nih.gov/697083/).
- Kaneda et al. (2011). *J Clin Pharmacol*. [Etomidate PK/PD](https://pubmed.ncbi.nlm.nih.gov/20498288/).
- Valk and Struys (2021). *Clin Pharmacokinet*. [Etomidate pharmacology review](https://pubmed.ncbi.nlm.nih.gov/34060021/).

### Ketamine

- Kamp et al. (2020). *Anesthesiology*. [Ketamine PK meta-analysis and three-compartment model](https://pubmed.ncbi.nlm.nih.gov/32997732/).
- Kamp et al. (2021). *Br J Anaesth*. [Stereoselective ketamine effects on cardiac output](https://pubmed.ncbi.nlm.nih.gov/33896589/).
- Idvall et al. (1979). *Br J Anaesth*. [Ketamine anesthesia and hemodynamics](https://pubmed.ncbi.nlm.nih.gov/526385/).
- Tweed et al. (1972). *Anesthesiology*. [Circulatory responses to ketamine](https://pubmed.ncbi.nlm.nih.gov/4652778/).
- Bovill and Dundee (1972). *Anaesthesia*. [Ketamine cardiovascular responses after atropine and adrenergic blockade](https://pubmed.ncbi.nlm.nih.gov/5041076/).
- Akine et al. (2001). *Auton Neurosci*. [Ketamine and propofol depress baroreflex sensitivity](https://pubmed.ncbi.nlm.nih.gov/11476280/).
- Huang et al. (2020). *J Physiol*. [Low-dose ketamine and sympathetic responses to central hypovolemia](https://pubmed.ncbi.nlm.nih.gov/33084081/).
- Bourke et al. (1987). *Anesthesiology*. [Ketamine and CO₂ response](https://pubmed.ncbi.nlm.nih.gov/3101549/).

### Lidocaine

- DailyMed (2026). [Lidocaine injection prescribing information](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ad1ef0a8-88db-4648-c098-d009eaeb4e5b).
- Foong et al. (2025). *Pharm Res*. [Lidocaine population PK in surgical patients](https://pubmed.ncbi.nlm.nih.gov/40021547/).
- Tucker and Boas (1971). *Anesthesiology*. [Lidocaine kinetics and intravenous regional anesthesia](https://pubmed.ncbi.nlm.nih.gov/5556244/).
- Rowland et al. (1971). *Ann N Y Acad Sci*. [Lidocaine kinetics in healthy subjects](https://pubmed.ncbi.nlm.nih.gov/5285383/).
- Foldes et al. (1960). *JAMA*. [Intravenous local anesthetic toxicity in man](https://pubmed.ncbi.nlm.nih.gov/13823696/).
- Jorfeldt et al. (1968). *Acta Anaesthesiol Scand*. [Central circulation and respiration during local anesthetic infusion](https://pubmed.ncbi.nlm.nih.gov/4896937/).
- Armstrong et al. (1988). *Ther Drug Monit*. [Clinical response and total and unbound plasma concentrations after lidocaine overdose](https://pubmed.ncbi.nlm.nih.gov/3201537/).
- Heinonen et al. (2015). *Basic Clin Pharmacol Toxicol*. [Lipid emulsion and lidocaine toxicity in volunteers](https://pubmed.ncbi.nlm.nih.gov/25207682/).
- Fettiplace et al. (2014). *Anesthesiology*. [Cardiotonic effect of lipid emulsion in cardiac toxicity](https://pubmed.ncbi.nlm.nih.gov/24496123/).
- Neal et al. (2021). *Reg Anesth Pain Med*. [ASRA practice advisory and 2020 LAST checklist](https://pubmed.ncbi.nlm.nih.gov/33148630/).
- American Society of Regional Anesthesia and Pain Medicine (2020). [LAST checklist](https://asra.com/docs/default-source/guidelines-articles/local-anesthetic-systemic-toxicity-rgb.pdf).
- Himes et al. (1977). *Anesthesiology*. [Lidocaine and anesthetic requirement](https://pubmed.ncbi.nlm.nih.gov/911052/).
- Hans et al. (2010). *Br J Anaesth*. [Lidocaine reduces propofol requirements during stimulation](https://pubmed.ncbi.nlm.nih.gov/20650919/).
- Altermatt et al. (2012). *Br J Anaesth*. [Lidocaine during propofol maintenance and recovery](https://pubmed.ncbi.nlm.nih.gov/22490315/).
- Qin et al. (2025). *Indian J Anaesth*. [Meta-analysis of IV lidocaine and the intubation response](https://pubmed.ncbi.nlm.nih.gov/40800699/).
- Wilson et al. (1991). *Anaesthesia*. [Lidocaine timing and the intubation response](https://pubmed.ncbi.nlm.nih.gov/2014891/).
- Tam et al. (1987). *Anesth Analg*. [Lidocaine timing before intubation](https://pubmed.ncbi.nlm.nih.gov/3631567/).
- Okuda et al. (1990). *J Anesth*. [Lidocaine plasma concentration and timing before intubation](https://pubmed.ncbi.nlm.nih.gov/15236000/).

### Amiodarone

- Korth-Bradley et al. (1996). *J Clin Pharmacol*. [Population PK of IV amiodarone in refractory VT/VF](https://pubmed.ncbi.nlm.nih.gov/8877675/).
- Kosinski et al. (1984). *J Am Coll Cardiol*. [Hemodynamic effects of IV amiodarone](https://pubmed.ncbi.nlm.nih.gov/6470338/).
- Munoz et al. (1988). *Eur Heart J*. [IV amiodarone with and without Tween 80](https://pubmed.ncbi.nlm.nih.gov/3280316/).
- DailyMed (2024). [Amiodarone injection prescribing information](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=51772745-9ea9-42cd-97e6-7266f41ae3ca).

## Drug timing and TCI performance

- Hughes et al. (1992). *Anesthesiology*. [Propofol context-sensitive half-time](https://pubmed.ncbi.nlm.nih.gov/1539843/).
- Egan et al. (1993). *Anesthesiology*. [Remifentanil context-sensitive half-time](https://pubmed.ncbi.nlm.nih.gov/7902032/).
- Kapila et al. (1995). *Anesthesiology*. [Remifentanil offset and ventilation recovery](https://pubmed.ncbi.nlm.nih.gov/7486182/).
- Grundmann et al. (2001). *Acta Anaesthesiol Scand*. [Propofol-remifentanil recovery](https://pubmed.ncbi.nlm.nih.gov/11207468/).
- Kwon et al. (2018). *Korean J Anesthesiol*. [Hypercapnia and propofol emergence](https://pubmed.ncbi.nlm.nih.gov/29690757/).
- Varvel et al. (1992). *J Pharmacokinet Biopharm*. [TCI performance metrics](https://pubmed.ncbi.nlm.nih.gov/1588504/).
- Shafer and Gregg (1992). *J Pharmacokinet Biopharm*. [Effect-site and plasma-targeted TCI](https://pubmed.ncbi.nlm.nih.gov/1629794/).

## Inhaled anesthetics

- Davis and Mapleson (1981). *Br J Anaesth*. [Physiological model of inhaled anesthetics](https://pubmed.ncbi.nlm.nih.gov/7225273/).
- Mapleson (1996). *Br J Anaesth*. [Age-related MAC formula](https://pubmed.ncbi.nlm.nih.gov/8777094/).
- Nickalls and Mapleson (2003). *Br J Anaesth*. [Sevoflurane MAC](https://pubmed.ncbi.nlm.nih.gov/12878613/).
- Yasuda et al. (1991). *Anesth Analg*. [Sevoflurane and isoflurane uptake and washout](https://pubmed.ncbi.nlm.nih.gov/1994760/).
- Carpenter et al. (1986). *Anesth Analg*. [Inhaled anesthetic kinetics in humans](https://pubmed.ncbi.nlm.nih.gov/3706798/).
- Goto et al. (2000). *Anesthesiology*. [Nitrous oxide MAC at awakening](https://pubmed.ncbi.nlm.nih.gov/11046204/).
- Katoh et al. (1997). *Br J Anaesth*. [Sevoflurane-nitrous oxide interaction at awakening](https://pubmed.ncbi.nlm.nih.gov/9389264/).

## Anesthetic depth and responsiveness

- Bouillon et al. (2004). *Anesthesiology*. [Propofol-remifentanil interaction for hypnosis, BIS, and tolerance of laryngoscopy](https://pubmed.ncbi.nlm.nih.gov/15166553/).
- Hannivoort et al. (2016). *Br J Anaesth*. [Propofol-sevoflurane-remifentanil interaction for laryngoscopy](https://pubmed.ncbi.nlm.nih.gov/27106965/).
- Heyse et al. (2012). *Anesthesiology*. [Sevoflurane-remifentanil interaction for clinical responses](https://pubmed.ncbi.nlm.nih.gov/22222473/).
- Kuizenga et al. (2019). *Anesthesiology*. [Propofol and sevoflurane with remifentanil for clinical responses](https://pubmed.ncbi.nlm.nih.gov/31567365/).
- Schnider et al. (1999). *Anesthesiology*. [Age and propofol Ce50 for loss of consciousness](https://pubmed.ncbi.nlm.nih.gov/10360845/).
- Lysakowski et al. (2009). *Br J Anaesth*. [BIS at propofol loss of consciousness in young and elderly patients](https://pubmed.ncbi.nlm.nih.gov/19542103/).
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

## Stimulation responses

- Albertin et al. (2005). *Anesth Analg*. [Remifentanil blunting the hemodynamic response to intubation and incision](https://pubmed.ncbi.nlm.nih.gov/15976218/).
- Kazama et al. (1997). *Anesthesiology*. [Propofol-fentanyl interaction and cardiovascular responses to noxious stimuli](https://pubmed.ncbi.nlm.nih.gov/9286884/).
- Katoh et al. (1999). *Anesthesiology*. [Fentanyl reduction of sevoflurane MAC and MAC-BAR](https://pubmed.ncbi.nlm.nih.gov/9952144/).
- Ebert et al. (1995). *Anesthesiology*. [Sympathetic and hemodynamic responses to sevoflurane and intubation](https://pubmed.ncbi.nlm.nih.gov/7605024/).
- Bullington et al. (1989). *Anesth Analg*. [Age and the sympathetic response to intubation](https://pubmed.ncbi.nlm.nih.gov/2541638/).
- Billard et al. (1994). *Anesthesiology*. [Hemodynamic response to induction and intubation with propofol and fentanyl](https://pubmed.ncbi.nlm.nih.gov/7992907/).
- Hayakawa-Fujii et al. (2001). *J Anesth*. [Hemodynamic stability during induction and intubation with propofol plus fentanyl, ketamine, and fentanyl-ketamine](https://pubmed.ncbi.nlm.nih.gov/14569434/).
- Sawano et al. (2013). *J Anesth*. [Fentanyl dose, hemodynamics, and cardiac output during induction and intubation](https://pubmed.ncbi.nlm.nih.gov/23314694/).

## Neuromuscular block

- Kleijn et al. (2011). *Br J Clin Pharmacol*. [Sugammadex PK/PD](https://pubmed.ncbi.nlm.nih.gov/21535448/).
- Ploeger et al. (2009). *Anesthesiology*. [Sugammadex PK/PD modeling](https://pubmed.ncbi.nlm.nih.gov/19104176/).
- Pühringer et al. (2010). *Br J Anaesth*. [Sugammadex reversal times](https://pubmed.ncbi.nlm.nih.gov/20876699/).
- Nguyen-Lee et al. (2018). *Curr Anesthesiol Rep*. [Sugammadex PK review](https://doi.org/10.1007/s40140-018-0266-5).
- Wierda et al. (1991). *Can J Anaesth*. [Rocuronium PK](https://pubmed.ncbi.nlm.nih.gov/1829656/).
- Khuenl-Brady and Sparr (1996). *Clin Pharmacokinet*. [Rocuronium clinical PK review](https://pubmed.ncbi.nlm.nih.gov/8877248/).
- Masui et al. (2018). *J Anesth*. [Rocuronium PD covariates](https://pubmed.ncbi.nlm.nih.gov/30099599/).
- Sasakawa et al. (2016). *J Anesth*. [Predictive ability of six rocuronium PK models](https://pubmed.ncbi.nlm.nih.gov/27098255/).
- Varrique et al. (2016). *J Pharm Pharmacol*. [Rocuronium PK/PD in young and elderly patients](https://pubmed.ncbi.nlm.nih.gov/27545305/).
- Magorian et al. (1995). *Anesth Analg*. [Rocuronium PK in liver disease](https://pubmed.ncbi.nlm.nih.gov/7893030/).
- Robertson et al. (2005). *Eur J Anaesthesiol*. [Rocuronium PK in renal failure](https://pubmed.ncbi.nlm.nih.gov/15816565/).
- Magorian et al. (1993). *Anesthesiology*. [Rocuronium onset](https://pubmed.ncbi.nlm.nih.gov/7902034/).
- Roy et al. (2002). *Anesthesiology*. [Succinylcholine concentration-effect model](https://pubmed.ncbi.nlm.nih.gov/12411790/).
- Kopman et al. (2003). *Anesthesiology*. [Succinylcholine dose, onset, and twitch recovery](https://pubmed.ncbi.nlm.nih.gov/14576537/).
- Dhonneur et al. (1999). *Anesthesiology*. [Succinylcholine and rocuronium at the larynx, diaphragm, and adductor pollicis](https://pubmed.ncbi.nlm.nih.gov/10201662/).
- Cantineau et al. (1994). *Anesthesiology*. [Rocuronium effects at the diaphragm](https://pubmed.ncbi.nlm.nih.gov/8092503/).
- Plaud et al. (1995). *Clin Pharmacol Ther*. [Rocuronium laryngeal effect-site kinetics](https://pubmed.ncbi.nlm.nih.gov/7648768/).
- Wright et al. (1994). *Anesthesiology*. [Laryngeal sensitivity to rocuronium](https://pubmed.ncbi.nlm.nih.gov/7978469/).
- Eikermann et al. (2003). *Anesthesiology*. [Respiratory recovery after neuromuscular block](https://pubmed.ncbi.nlm.nih.gov/12766640/).
- Eikermann et al. (2007). *Am J Respir Crit Care Med*. [Airway collapse during partial neuromuscular block](https://pubmed.ncbi.nlm.nih.gov/17023729/).
- Herbstreit et al. (2009). *Anesthesiology*. [Residual block and pharyngeal closing pressure](https://pubmed.ncbi.nlm.nih.gov/19417617/).
- Fiset et al. (1991). *Can J Anaesth*. [Nitrous oxide and neuromuscular block](https://pubmed.ncbi.nlm.nih.gov/1683819/).
- Bissinger et al. (1993). *Anasthesiol Intensivmed Notfallmed Schmerzther*. [Curare clefts and neuromuscular monitoring](https://pubmed.ncbi.nlm.nih.gov/7902740/).

## Hemodynamics

### Anesthetic effects

- Su et al. (2023). *Br J Anaesth*. [Mechanistic hemodynamic interaction model](https://pubmed.ncbi.nlm.nih.gov/37355412/).
- Garraza-Obaldia et al. (2024). *Pharmaceutics*. [Propofol-remifentanil hemodynamics and noxious stimulation](https://pubmed.ncbi.nlm.nih.gov/39771593/).
- Fregolent et al. (2025). *IFAC-PapersOnLine*. [Su model calibration in surgical patients](https://doi.org/10.1016/j.ifacol.2025.07.125).
- Jeleazcov et al. (2015). *Br J Anaesth*. [Propofol arterial pressure effect sites in volunteers](https://pubmed.ncbi.nlm.nih.gov/25627394/).
- Kazama et al. (1999). *Anesthesiology*. [Propofol equilibration for SBP and BIS by age](https://pubmed.ncbi.nlm.nih.gov/10360846/).
- Claeys et al. (1988). *Br J Anaesth*. [Hemodynamic changes during anesthesia induced and maintained with propofol](https://pubmed.ncbi.nlm.nih.gov/3257393/).
- He et al. (2024). *J Biomed Res*. [Echocardiographic hemodynamics during remimazolam versus propofol induction in elderly patients](https://pubmed.ncbi.nlm.nih.gov/37907250/).
- Hino et al. (2019). *J Anesth*. [Age-stratified propofol and thiopental induction with remifentanil](https://doi.org/10.1007/s00540-019-02657-x).
- Oh et al. (2025). *Drug Des Devel Ther*. [Blood pressure during propofol TCI induction in patients aged 65 and over](https://doi.org/10.2147/DDDT.S553809).
- van den Berg et al. (2021). *J Clin Monit Comput*. [Hemodynamics at equipotent propofol-remifentanil pairs](https://pubmed.ncbi.nlm.nih.gov/32488680/).
- Sjøen et al. (2024). *Acta Anaesthesiol Scand*. [Remifentanil dose and hemodynamics during propofol induction](https://pubmed.ncbi.nlm.nih.gov/38400761/).
- Sjøen et al. (2023). *Acta Anaesthesiol Scand*. [Propofol dose and hemodynamics during remifentanil-propofol induction](https://pubmed.ncbi.nlm.nih.gov/37291731/).
- Poterman et al. (2017). *Eur J Anaesthesiol*. [Atropine before propofol and high-dose remifentanil induction](https://pubmed.ncbi.nlm.nih.gov/28403018/).
- de Wit et al. (2016). *Br J Anaesth*. [Propofol, stressed volume, and resistance to venous return](https://pubmed.ncbi.nlm.nih.gov/27199311/).
- Zucker et al. (2022). *BMC Anesthesiol*. [Mean systemic filling pressure during propofol induction](https://pubmed.ncbi.nlm.nih.gov/35869445/).
- Fairfield et al. (1991). *Br J Anaesth*. [Echocardiographic hemodynamics after propofol 2.5 mg/kg](https://pubmed.ncbi.nlm.nih.gov/1751277/).
- Morley et al. (2010). *Anesth Analg*. [Fasting, baseline pressure, and hypotension during propofol induction](https://pubmed.ncbi.nlm.nih.gov/20861421/).
- Ebert et al. (1995). *Anesth Analg*. [Sevoflurane cardiovascular responses](https://pubmed.ncbi.nlm.nih.gov/7486143/).
- Malan et al. (1995). *Anesthesiology*. [Sevoflurane cardiac output and vascular resistance in volunteers](https://pubmed.ncbi.nlm.nih.gov/7486177/).

### Reflexes and age

- Ebert et al. (1992). *Anesthesiology*. [Sympathetic responses and baroreflexes during propofol induction](https://pubmed.ncbi.nlm.nih.gov/1575340/).
- Ebert and Muzi (1994). *Anesth Analg*. [Propofol and baroreflex sensitivity](https://pubmed.ncbi.nlm.nih.gov/8311293/).
- Sato et al. (2005). *Br J Anaesth*. [Baroreflex during propofol infusion](https://pubmed.ncbi.nlm.nih.gov/15722386/).
- Sellgren et al. (1994). *Anesthesiology*. [Sympathetic nerve activity and baroreflexes during propofol anesthesia](https://pubmed.ncbi.nlm.nih.gov/8141450/).
- Umehara et al. (2006). *Anesth Analg*. [Sevoflurane and baroreflex sensitivity](https://pubmed.ncbi.nlm.nih.gov/16368802/).
- Laitinen et al. (2004). *J Appl Physiol*. [Age and autonomic responses to head-up tilt](https://pubmed.ncbi.nlm.nih.gov/14766788/).
- Monahan (2007). *Am J Physiol Regul Integr Comp Physiol*. [Aging and baroreflex function](https://pubmed.ncbi.nlm.nih.gov/17442786/).
- Kardos et al. (2001). *Hypertension*. [Spontaneous baroreflex sensitivity by age in 1134 adults](https://pubmed.ncbi.nlm.nih.gov/11244017/).
- Leenen et al. (2005). *Am J Physiol Heart Circ Physiol*. [Aging and cardiac responses to epinephrine](https://pubmed.ncbi.nlm.nih.gov/15626690/).

### Blood loss and shock

- Johnson et al. (2014). *J Appl Physiol*. [Hemodynamic responses to graded blood loss in awake humans](https://pubmed.ncbi.nlm.nih.gov/24876357/).
- Martin et al. (1990). *Acta Anaesthesiol Scand*. [Vascular resistance in septic shock](https://pubmed.ncbi.nlm.nih.gov/2389659/).
- Melo and Peters (1999). *Crit Care*. [Vascular resistance in septic shock](https://pubmed.ncbi.nlm.nih.gov/11056727/).
- Bellissant and Annane (2000). *Clin Pharmacol Ther*. [Pressor hyporesponsiveness in septic shock](https://pubmed.ncbi.nlm.nih.gov/11014411/).
- Margarson and Soni (2002). *J Appl Physiol*. [Albumin leakage in septic shock](https://pubmed.ncbi.nlm.nih.gov/11960967/).
- Fisher (1986). *Anaesth Intensive Care*. [Plasma escape during anaphylaxis](https://pubmed.ncbi.nlm.nih.gov/2869715/).
- Brown et al. (2004). *Emerg Med J*. [IV epinephrine and rapid saline for sting anaphylaxis](https://pubmed.ncbi.nlm.nih.gov/14988337/).
- Schotola et al. (2012). *Crit Care*. [Acidosis and the adrenergic response of human failing myocardium](https://pubmed.ncbi.nlm.nih.gov/22889236/).

### Pulmonary circulation

- Segeroth et al. (2023). *Eur Heart J Cardiovasc Imaging*. [Pulmonary transit time](https://pubmed.ncbi.nlm.nih.gov/36662127/).
- Koganov et al. (1997). *Crit Care Med*. [PEEP raises pulmonary vascular resistance](https://pubmed.ncbi.nlm.nih.gov/9187594/).
- Carlsson et al. (1985). *Acta Anaesthesiol Scand*. [Hypoxic pulmonary vasoconstriction](https://pubmed.ncbi.nlm.nih.gov/3993324/).

### Arrhythmias and cardiac arrest

- Koster et al. (2008). *Resuscitation*. [Recurrent VF during advanced life support](https://pubmed.ncbi.nlm.nih.gov/18556106/).
- van Alem et al. (2003). *Resuscitation*. [VF recurrence and patient outcome](https://pubmed.ncbi.nlm.nih.gov/14625108/).
- Somberg et al. (2002). *Am J Cardiol*. [IV lidocaine versus amiodarone for incessant VT](https://pubmed.ncbi.nlm.nih.gov/12372573/).
- Scheinman et al. (1995). *Circulation*. [Dose-ranging IV amiodarone in life-threatening ventricular tachyarrhythmias](https://pubmed.ncbi.nlm.nih.gov/7586313/).
- Kudenchuk et al. (2016). *N Engl J Med*. [Amiodarone, lidocaine, or placebo in cardiac arrest](https://pubmed.ncbi.nlm.nih.gov/27043165/).
- Salcido et al. (2018). *Resuscitation*. [Antiarrhythmic administration and rearrest in ALPS](https://pubmed.ncbi.nlm.nih.gov/29803703/).
- American Heart Association (2025). [Adult cardiac arrest algorithm](https://cpr.heart.org/-/media/CPR-Files/CPR-Guidelines-Files/2025-Algorithms/Algorithm-ACLS-CA-250527.pdf).
- Clark et al. (1997). *J Am Coll Cardiol*. [Atrial fibrillation and cardiac output](https://pubmed.ncbi.nlm.nih.gov/9316536/).
- Hardman et al. (1998). *Cardiovasc Res*. [Atrial fibrillation and stroke volume](https://pubmed.ncbi.nlm.nih.gov/9683909/).
- Kerr et al. (1998). *Am J Cardiol*. [Stroke-volume variability in atrial fibrillation](https://pubmed.ncbi.nlm.nih.gov/9874054/).
- Corino et al. (2015). *J Cardiovasc Electrophysiol*. [R-R variability in atrial fibrillation](https://pubmed.ncbi.nlm.nih.gov/25367150/).
- Hogue et al. (1996). *Anesthesiology*. [Post-induction bradycardia and atrial pacing](https://pubmed.ncbi.nlm.nih.gov/8694384/).
- Mort (2004). *J Clin Anesth*. [Hypoxemia and intubation-related cardiac arrest](https://pubmed.ncbi.nlm.nih.gov/15590254/).
- De Jong et al. (2018). *Crit Care Med*. [Intubation-related cardiac arrest](https://pubmed.ncbi.nlm.nih.gov/29261566/).
- Heffner et al. (2013). *Resuscitation*. [Cardiac arrest during emergency airway management](https://pubmed.ncbi.nlm.nih.gov/23911630/).
- Weisfeldt and Becker (2002). *JAMA*. [Three-phase time-sensitive model of resuscitation after cardiac arrest](https://pubmed.ncbi.nlm.nih.gov/12479769/).
- Paradis et al. (1990). *JAMA*. [Coronary perfusion pressure and return of spontaneous circulation in human CPR](https://pubmed.ncbi.nlm.nih.gov/2386557/).
- Meaney et al. (2013). *Circulation*. [Cardiopulmonary resuscitation quality](https://pubmed.ncbi.nlm.nih.gov/23801105/).
- Cantineau et al. (1994). *Am J Emerg Med*. [Epinephrine and end-tidal carbon dioxide during prehospital CPR](https://pubmed.ncbi.nlm.nih.gov/8179728/).
- Kim et al. (2025). *Resuscitation*. [Real-time arterial pressure responses to epinephrine during CPR](https://pubmed.ncbi.nlm.nih.gov/40250548/).
- Park et al. (1994). *Pacing Clin Electrophysiol*. [Systolic arterial pressure recovery after ventricular fibrillation in humans](https://pubmed.ncbi.nlm.nih.gov/7521035/).
- Tokano et al. (1998). *J Cardiovasc Electrophysiol*. [Ventricular shock strength and cardiac hemodynamics](https://pubmed.ncbi.nlm.nih.gov/9727657/).
- Laurent et al. (2002). *J Am Coll Cardiol*. [Reversible myocardial dysfunction in survivors of out-of-hospital cardiac arrest](https://pubmed.ncbi.nlm.nih.gov/12505221/).
- ZOLL (2023). [R Series operator's guide](https://www.zoll.com/-/media/product-materials/product-manuals/r-series/05/9650-0912-05-sf_r.ashx).

## Cardiovascular drugs

### Norepinephrine

- Li et al. (2024). *Clin Pharmacokinet*. [Norepinephrine PK with propofol interaction](https://pubmed.ncbi.nlm.nih.gov/39465453/).
- de Keijzer et al. (2026). *Eur J Anaesthesiol*. [Norepinephrine response under general anesthesia](https://pubmed.ncbi.nlm.nih.gov/41481868/).
- Joachim et al. (2024). *Br J Clin Pharmacol*. [Peripheral norepinephrine bolus PK/PD during general anesthesia](https://pubmed.ncbi.nlm.nih.gov/39039779/).
- Persichini et al. (2012). *Crit Care Med*. [Norepinephrine and mean systemic pressure](https://pubmed.ncbi.nlm.nih.gov/22926333/).

### Epinephrine

- Ensinger et al. (1992). *Eur J Anaesthesiol*. [Arterial epinephrine clearance](https://pubmed.ncbi.nlm.nih.gov/1425612/).
- Ensinger et al. (1992). *Eur J Clin Pharmacol*. [Arterial and peripheral venous catecholamine concentrations](https://pubmed.ncbi.nlm.nih.gov/1425886/).
- Esler et al. (1995). *J Clin Endocrinol Metab*. [Aging and arterial epinephrine clearance](https://pubmed.ncbi.nlm.nih.gov/7852502/).
- Marker et al. (1998). *Am J Physiol*. [Epinephrine clearance and infusion responses in older adults](https://pubmed.ncbi.nlm.nih.gov/9814995/).
- Clutter et al. (1980). *J Clin Invest*. [Epinephrine cardiovascular effects](https://pubmed.ncbi.nlm.nih.gov/6995479/).
- Stratton et al. (1985). *J Appl Physiol*. [Epinephrine infusion hemodynamics](https://pubmed.ncbi.nlm.nih.gov/3988675/).
- Freyschuss et al. (1986). *Clin Sci (Lond)*. [Arterial epinephrine concentrations and hemodynamics](https://pubmed.ncbi.nlm.nih.gov/3956110/).
- Piccioli et al. (2023). *Front Cardiovasc Med*. [Contractility and pulse pressure during dobutamine](https://doi.org/10.3389/fcvm.2023.1197842).
- Pan et al. (1986). *J Pharmacol Exp Ther*. [Aging and beta-adrenergic vascular relaxation](https://pubmed.ncbi.nlm.nih.gov/3025419/).
- Hogikyan and Supiano (1994). *Am J Physiol*. [Aging and arterial alpha-adrenergic responsiveness](https://pubmed.ncbi.nlm.nih.gov/8203510/).
- Takahashi et al. (2002). *Anesth Analg*. [IV epinephrine bolus response](https://pubmed.ncbi.nlm.nih.gov/11867404/).
- Guinard et al. (1995). *Reg Anesth*. [Age and HR after IV epinephrine test doses](https://pubmed.ncbi.nlm.nih.gov/7547654/).

### Phenylephrine and ephedrine

- Anderson et al. (2017). *Paediatr Anaesth*. [Phenylephrine PK](https://pubmed.ncbi.nlm.nih.gov/28868789/).
- US Food and Drug Administration (2012). [Clinical pharmacology review of phenylephrine, NDA 203826](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2012/203826_phenylephrine_toc.cfm).
- Hengstmann and Goronzy (1982). *Eur J Clin Pharmacol*. [Phenylephrine PK](https://pubmed.ncbi.nlm.nih.gov/7056280/).
- Rebet et al. (2016). *Eur J Anaesthesiol*. [Preload-dependent phenylephrine cardiac-output responses](https://pubmed.ncbi.nlm.nih.gov/27164015/).
- Meng et al. (2011). *Br J Anaesth*. [Phenylephrine and ephedrine boluses under propofol-remifentanil anesthesia](https://pubmed.ncbi.nlm.nih.gov/21642644/).
- Persky et al. (2004). *Br J Clin Pharmacol*. [Oral ephedrine PK and cardiovascular tolerance](https://pubmed.ncbi.nlm.nih.gov/15089807/).
- Dings et al. (2026). *Pharmaceutics*. [Ephedrine hemodynamic time course during cesarean section](https://pubmed.ncbi.nlm.nih.gov/41900782/).
- Sjøen et al. (2026). *Acta Anaesthesiol Scand*. [Vasopressor boluses during propofol-remifentanil induction](https://pubmed.ncbi.nlm.nih.gov/41144827/) and [correction](https://pubmed.ncbi.nlm.nih.gov/41326972/).

### Other vasoactive drugs

- DailyMed (2025). [Vasopressin injection prescribing information](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=971f9b1c-6094-4f80-920b-cb5d7e62950a).
- Kates and Leier (1978). *Clin Pharmacol Ther*. [Dobutamine PK in heart failure](https://pubmed.ncbi.nlm.nih.gov/699477/).
- Daly et al. (1997). *Am J Cardiol*. [Dobutamine dose-concentration relationship](https://pubmed.ncbi.nlm.nih.gov/9165162/).
- Pfizer. [Dobutamine injection prescribing information](https://www.pfizermedicalinformation.com/patient/dobutamine/clinical-pharmacology).
- Magnani et al. (1977). *J Int Med Res*. [Dobutamine dose response](https://pubmed.ncbi.nlm.nih.gov/838109/).
- Baim et al. (1983). *N Engl J Med*. [Milrinone hemodynamics](https://pubmed.ncbi.nlm.nih.gov/6888453/).
- DailyMed (2026). [Milrinone injection prescribing information](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=88f78780-399f-4780-be19-205739db1682).
- Sum et al. (1983). *Clin Pharmacol Ther*. [Esmolol PK and beta blockade](https://pubmed.ncbi.nlm.nih.gov/6617063/).
- Wiest and Haney (2012). *Clin Pharmacokinet*. [Esmolol pharmacology review](https://pubmed.ncbi.nlm.nih.gov/22515557/).
- Abernethy et al. (1987). *Am J Cardiol*. [Labetalol PK/PD and age](https://pubmed.ncbi.nlm.nih.gov/3661438/).
- Hafsa et al. (2022). *Pharmaceutics*. [Labetalol kinetics and antagonist potency](https://pubmed.ncbi.nlm.nih.gov/36365181/).

### Antimuscarinics

- Hinderling et al. (1985). *J Pharm Sci*. [Atropine PK](https://pubmed.ncbi.nlm.nih.gov/4032240/) and [peripheral-compartment PD](https://pubmed.ncbi.nlm.nih.gov/4032241/).
- Yamaguchi et al. (1988). *Can J Anaesth*. [Atropine HR response during anesthesia](https://pubmed.ncbi.nlm.nih.gov/3168130/).
- Horiguchi and Nishikawa (2002). *Anesth Analg*. [Atropine dose-response awake and during propofol anesthesia](https://pubmed.ncbi.nlm.nih.gov/12145057/).
- Ali-Melkkilä et al. (1993). *Acta Anaesthesiol Scand*. [Anticholinergic PK/PD review](https://pubmed.ncbi.nlm.nih.gov/8249551/).
- Du et al. (2025). *J Drug Deliv Sci Technol*. [Glycopyrrolate population PK](https://doi.org/10.1016/j.jddst.2025.106692).

## Fluids

- Reid et al. (2003). *Clin Sci (Lond)*. [Urine after 2 L saline or Hartmann's](https://pubmed.ncbi.nlm.nih.gov/12519083/).
- Hahn (2010). *Anesthesiology*. [Crystalloid volume kinetics](https://pubmed.ncbi.nlm.nih.gov/20613481/).
- Hahn (2017). *Anesth Analg*. [Arterial pressure, age, and crystalloid elimination](https://pubmed.ncbi.nlm.nih.gov/28452823/).
- Hahn and Nemme (2020). *BMC Anesthesiol*. [Fluid retention after anesthesia induction](https://pubmed.ncbi.nlm.nih.gov/32334513/).
- Hahn (2021). *Perioper Med (Lond)*. [Crystalloid distribution during anesthesia-induced hypotension](https://pubmed.ncbi.nlm.nih.gov/34629113/).
- Scheingraber et al. (1999). *Anesthesiology*. [Hyperchloremic acidosis after rapid saline infusion](https://pubmed.ncbi.nlm.nih.gov/10319771/).
- Kramer et al. (2003). *Crit Care Med*. [Citrate pharmacokinetics in critically ill patients](https://pubmed.ncbi.nlm.nih.gov/14530750/).
- Rabcuka et al. (2025). *Transfusion*. [Acid-base and metabolite changes in stored red cell concentrates](https://pubmed.ncbi.nlm.nih.gov/41122829/).
- Thode et al. (1983). *Scand J Clin Lab Invest Suppl*. [Relation between pH and ionized calcium in vitro and in vivo](https://pubmed.ncbi.nlm.nih.gov/6578581/).
- Bolan et al. (2001). *Transfusion*. [Citrate, ionized calcium, and PTH during plateletpheresis](https://pubmed.ncbi.nlm.nih.gov/11552076/).
- Ansari et al. (2025). *Anesthesiology*. [Pharmacokinetics of intravenous calcium during cesarean delivery](https://pubmed.ncbi.nlm.nih.gov/39361822/).
- González et al. (2018). *Kidney Int Rep*. [Secondary response to chronic respiratory acidosis](https://pubmed.ncbi.nlm.nih.gov/30197983/).
- Krapf et al. (1991). *N Engl J Med*. [Renal response to chronic respiratory alkalosis](https://pubmed.ncbi.nlm.nih.gov/1902283/).

## Respiratory control

- Bellemare and Grassino (1982). *J Appl Physiol*. [Pressure, duty cycle, and diaphragm endurance](https://pubmed.ncbi.nlm.nih.gov/7174413/).
- Aubier et al. (1981). *J Appl Physiol*. [High- and low-frequency diaphragmatic fatigue and recovery](https://pubmed.ncbi.nlm.nih.gov/7251445/).
- Laghi et al. (1995). *J Appl Physiol*. [Recovery from diaphragmatic fatigue over 24 hours](https://pubmed.ncbi.nlm.nih.gov/7592215/).
- Travaline et al. (1997). *Am J Respir Crit Care Med*. [Recovery of twitch and maximal transdiaphragmatic pressure after fatigue](https://pubmed.ncbi.nlm.nih.gov/9372676/).
- Rohrbach et al. (2003). *Eur J Appl Physiol*. [Repeat inspiratory endurance and twitch pressure after 15-minute rest](https://pubmed.ncbi.nlm.nih.gov/12827367/).
- García-Río et al. (2007). *Respir Med*. [Age and hypoxic and hypercapnic inspiratory drive](https://pubmed.ncbi.nlm.nih.gov/17643972/).
- Egan et al. (2004). *Br J Anaesth*. [Remifentanil bolus respiratory effects by age](https://pubmed.ncbi.nlm.nih.gov/14970134/).
- Bouillon et al. (2004). *Anesthesiology*. [Dynamic propofol ventilatory depression](https://pubmed.ncbi.nlm.nih.gov/14739795/).
- Olofsen et al. (2010). *Anesthesiology*. [Dynamic remifentanil respiratory effects in awake and propofol-sedated volunteers](https://pubmed.ncbi.nlm.nih.gov/20461001/).
- Babenco et al. (2000). *Anesthesiology*. [Remifentanil bolus ventilatory control](https://pubmed.ncbi.nlm.nih.gov/10691225/).
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
- Miyamura et al. (1980). *Jpn J Physiol*. [Hypercapnic ventilatory response variability](https://pubmed.ncbi.nlm.nih.gov/6790801/).
- Clark and von Euler (1972). *J Physiol*. [Volume feedback on inspiratory duration](https://pubmed.ncbi.nlm.nih.gov/5033464/).
- Polacheck et al. (1980). *J Appl Physiol*. [Vagal inflation feedback under anesthesia](https://pubmed.ncbi.nlm.nih.gov/7440275/).
- Tobin et al. (1983). *Chest*. [Breathing pattern of normal subjects](https://pubmed.ncbi.nlm.nih.gov/6872603/).
- Leung et al. (1997). *Am J Respir Crit Care Med*. [SIMV and pressure-support effort unloading](https://pubmed.ncbi.nlm.nih.gov/9196100/).
- Graves et al. (1986). *Am J Physiol*. [Patient-ventilator synchronization under anesthesia](https://pubmed.ncbi.nlm.nih.gov/3706575/).
- Christie and Smith (1992). *Anesth Analg*. [Pressure support during general anesthesia](https://pubmed.ncbi.nlm.nih.gov/1632530/).
- Lim et al. (2012). *Paediatr Anaesth*. [Pediatric pressure support with an LMA](https://pubmed.ncbi.nlm.nih.gov/22380745/).
- Capdevila et al. (2014). *PLoS One*. [Pressure support and LMA recovery](https://pubmed.ncbi.nlm.nih.gov/25536515/).

## Gas exchange and blood chemistry

- Ronco et al. (1993). *JAMA*. [Critical oxygen delivery and extraction in septic and nonseptic humans](https://pubmed.ncbi.nlm.nih.gov/8411504/).
- Loer et al. (1997). *Anesthesiology*. [Oxygen consumption of the human lung](https://pubmed.ncbi.nlm.nih.gov/9066318/).
- Dunham et al. (1991). *Crit Care Med*. [Oxygen debt, lactate, and base deficit during canine hemorrhagic shock](https://pubmed.ncbi.nlm.nih.gov/1989759/).
- Ward et al. (2010). *Resuscitation*. [Oxygen transport, lactate, and base excess during simulated hemorrhage in humans](https://pubmed.ncbi.nlm.nih.gov/20418009/).
- Revelly et al. (2005). *Crit Care Med*. [Lactate production and clearance in severe sepsis and cardiogenic shock](https://pubmed.ncbi.nlm.nih.gov/16215376/).
- Levraut et al. (1998). *Am J Respir Crit Care Med*. [Lactate distribution and impaired clearance in stable septic patients](https://pubmed.ncbi.nlm.nih.gov/9563714/).
- Grip et al. (2015). *Clin Sci (Lond)*. [Whole-body and muscle lactate kinetics during epinephrine infusion](https://pubmed.ncbi.nlm.nih.gov/25828264/).
- Zitek et al. (2018). *J Emerg Med*. [Serum lactate after 30 mL/kg of lactated Ringer's](https://pubmed.ncbi.nlm.nih.gov/30037514/).
- Dash et al. (2016). *Eur J Appl Physiol*. [Hemoglobin dissociation curves across oxygen tension, temperature, pH, and PCO₂](https://pubmed.ncbi.nlm.nih.gov/26298270/).
- Rees et al. (2010). *Eur J Appl Physiol*. [Plasma and red-cell acid-base chemistry with oxygen-dependent buffering](https://pubmed.ncbi.nlm.nih.gov/19841930/).
- Figge et al. (2018). *J Crit Care*. [Albumin and phosphate charge in plasma acid-base chemistry](https://pubmed.ncbi.nlm.nih.gov/29080515/).
- O'Neill and Robbins (2017). *J Appl Physiol*. [Blood CO₂ transport and the magnitude of the Haldane effect](https://pubmed.ncbi.nlm.nih.gov/27881667/).
- Wooten (2003). *J Appl Physiol*. [Acid-base balance in multicompartment systems and the extracellular buffer approximation](https://pubmed.ncbi.nlm.nih.gov/12923118/).
- Brackett et al. (1965). *N Engl J Med*. [Whole-body acute hypercapnic buffering](https://pubmed.ncbi.nlm.nih.gov/14219220/).
- Arbus et al. (1969). *N Engl J Med*. [Human acute respiratory alkalosis ranges](https://pubmed.ncbi.nlm.nih.gov/5782512/).
- Crapo et al. (1999). *Am J Respir Crit Care Med*. [Arterial blood gas and A-a gradient reference values](https://pubmed.ncbi.nlm.nih.gov/10556115/).
- Magor-Elliott et al. (2021). *J Appl Physiol*. [Tissue gas storage and circulation delays](https://pubmed.ncbi.nlm.nih.gov/33475459/).
- Lyons et al. (2024). *Anaesthesia*. [Arterial CO₂ accumulation during apnea with and without high-flow oxygen](https://pubmed.ncbi.nlm.nih.gov/38100148/).
- Gentz et al. (1998). *J Clin Anesth*. [Apneic CO₂ accumulation after brief hypocapnia](https://pubmed.ncbi.nlm.nih.gov/9603587/).
- Eger and Severinghaus (1961). *Anesthesiology*. [Apneic CO₂ rise after sustained hyperventilation](https://pubmed.ncbi.nlm.nih.gov/13725901/).
- Frumin et al. (1959). *Anesthesiology*. [Prolonged apneic oxygenation](https://pubmed.ncbi.nlm.nih.gov/13825447/).
- Taskar et al. (1995). *Chest*. [CO₂ elimination after ventilation changes](https://pubmed.ncbi.nlm.nih.gov/7606958/).
- Weil et al. (1986). *N Engl J Med*. [Arterial and mixed venous acid-base differences during CPR](https://pubmed.ncbi.nlm.nih.gov/3088448/).
- Kobayashi and Yokoyama (1994). *Masui*. [Blood gas changes during apnea](https://pubmed.ncbi.nlm.nih.gov/7933492/).
- Benumof et al. (1997). *Anesthesiology*. [Desaturation after preoxygenation](https://pubmed.ncbi.nlm.nih.gov/9357902/).
- Nimmagadda et al. (2017). *Anesth Analg*. [Preoxygenation endpoints](https://pubmed.ncbi.nlm.nih.gov/28099321/).
- Hardman et al. (2000). *Anesth Analg*. [Hypoxemia during apnea](https://pubmed.ncbi.nlm.nih.gov/10702447/).
- Laviola et al. (2020). *Br J Anaesth*. [Oxygen fraction during passive airway rescue](https://pubmed.ncbi.nlm.nih.gov/32008701/).
- Farmery and Roe (1996). *Br J Anaesth*. [Model of oxyhemoglobin desaturation during apnea](https://pubmed.ncbi.nlm.nih.gov/8777112/).
- Stock et al. (1989). *J Clin Anesth*. [Blood gases during airway obstruction](https://pubmed.ncbi.nlm.nih.gov/2516732/).

## Lung aeration and airway patency

- Gunnarsson et al. (1991). *Br J Anaesth*. [Age, low V/Q, and atelectasis during anesthesia](https://pubmed.ncbi.nlm.nih.gov/2025468/).
- Hedenstierna and Edmark (2015). *Best Pract Res Clin Anaesthesiol*. [FRC, airway closure, atelectasis, and gas exchange under anesthesia](https://pubmed.ncbi.nlm.nih.gov/26643094/).
- Tokics et al. (1996). *J Appl Physiol*. [Atelectasis and shunt during anesthesia](https://pubmed.ncbi.nlm.nih.gov/8904605/).
- Ballester et al. (1989). *Thorax*. [V/Q distributions and oxygen response in acute severe asthma](https://pubmed.ncbi.nlm.nih.gov/2763227/).
- Edmark et al. (2003). *Anesthesiology*. [Inspired O₂ during induction, atelectasis, and apnea time](https://pubmed.ncbi.nlm.nih.gov/12502975/).
- Rothen et al. (1993). *Br J Anaesth*. [Inflation pressure and CT atelectasis](https://pubmed.ncbi.nlm.nih.gov/8280539/).
- Rothen et al. (1995). *Anesthesiology*. [Oxygen concentration and recurrent atelectasis](https://pubmed.ncbi.nlm.nih.gov/7717553/).
- Rothen et al. (1999). *Br J Anaesth*. [Time course of recruitment](https://pubmed.ncbi.nlm.nih.gov/10472221/).
- Bates and Irvin (2002). *J Appl Physiol*. [Recruitment and derecruitment dynamics](https://pubmed.ncbi.nlm.nih.gov/12133882/).
- Malo et al. (1984). *J Appl Physiol*. [PEEP and shunt in pulmonary edema](https://pubmed.ncbi.nlm.nih.gov/6389451/).
- Benoit et al. (2002). *Anesth Analg*. [Inspired O₂ before extubation and postoperative atelectasis](https://pubmed.ncbi.nlm.nih.gov/12456458/).
- Wahba et al. (1975). *Can Anaesth Soc J*. [Lung volumes after upper abdominal surgery](https://pubmed.ncbi.nlm.nih.gov/1095163/).
- Lindberg et al. (1992). *Acta Anaesthesiol Scand*. [Atelectasis and A-a gradient after lower abdominal surgery](https://pubmed.ncbi.nlm.nih.gov/1514340/).
- Putensen-Himmer et al. (1992). *Anesthesiology*. [FRC and blood gases after open and laparoscopic cholecystectomy](https://pubmed.ncbi.nlm.nih.gov/1416164/).
- Eastwood et al. (2005). *Anesthesiology*. [Propofol and airway closing pressure](https://pubmed.ncbi.nlm.nih.gov/16129969/).
- Hillman et al. (2009). *Anesthesiology*. [Airway collapse at loss of consciousness](https://pubmed.ncbi.nlm.nih.gov/19512872/).
- Joffe et al. (2010). *Anesthesiology*. [One-handed versus two-handed facemask ventilation in apneic adults](https://pubmed.ncbi.nlm.nih.gov/20808210/).
- Bouvet et al. (2014). *Anesthesiology*. [Inspiratory pressure and gastric insufflation during facemask ventilation](https://pubmed.ncbi.nlm.nih.gov/24317204/).
- Larsson et al. (1985). *J Allergy Clin Immunol*. [Epinephrine relief of bronchoconstriction](https://pubmed.ncbi.nlm.nih.gov/3989143/).
- Sydow et al. (1993). *Intensive Care Med*. [Respiratory mechanics in status asthmaticus](https://pubmed.ncbi.nlm.nih.gov/8294630/).
- Okayama et al. (1991). *J Asthma*. [Airway resistance in status asthmaticus](https://pubmed.ncbi.nlm.nih.gov/2010425/).
- Mead et al. (1967). *J Appl Physiol*. [Maximal expiratory flow](https://pubmed.ncbi.nlm.nih.gov/6017658/).
- Boczkowski et al. (1997). *Am J Respir Crit Care Med*. [Tidal expiratory flow limitation in asthma](https://pubmed.ncbi.nlm.nih.gov/9309989/).
- Tuxen and Lane (1987). *Am Rev Respir Dis*. [Expiratory time and obstructive hyperinflation](https://pubmed.ncbi.nlm.nih.gov/3662241/).

## Temperature

- Sessler (2000). *Anesthesiology*. [Perioperative heat balance](https://pubmed.ncbi.nlm.nih.gov/10691247/).
- Sessler (2016). *Lancet*. [Review of perioperative thermoregulation](https://pubmed.ncbi.nlm.nih.gov/26775126/).
- Matsukawa et al. (1995). *Anesthesiology*. [Heat redistribution after induction](https://pubmed.ncbi.nlm.nih.gov/7879935/).
- Frank et al. (1997). *JAMA*. [Perioperative thermoregulation](https://pubmed.ncbi.nlm.nih.gov/9087467/).

## Malignant hyperthermia

- DailyMed (2024). [Ryanodex (dantrolene) prescribing information](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=8f7b3ac0-604d-4c78-b545-5e0f8ea3d698).
- Flewellen et al. (1983). *Anesthesiology*. [Dantrolene dose response in awake man](https://doi.org/10.1097/00000542-198310000-00002).
- Visoiu et al. (2014). *Anesth Analg*. [Anesthetic drugs and onset of malignant hyperthermia](https://pubmed.ncbi.nlm.nih.gov/24445637/).

## Ventilator waveforms

- GE Healthcare. [Avance Carestation 8.X participant guide](https://www.gehealthcare.com/-/jssmedia/60148736cbc34131bbebbec747b696af.pdf).
- Giosa et al. (2024). *J Clin Med*. [Plateau pressure, residual flow, and patient effort](https://pubmed.ncbi.nlm.nih.gov/39685913/).
- Lee et al. (2022). *Sci Data*. [VitalDB waveforms and numerics](https://pubmed.ncbi.nlm.nih.gov/35676300/).
- Jonson et al. (1993). *J Appl Physiol*. [Human tissue viscoelasticity](https://pubmed.ncbi.nlm.nih.gov/8376259/).
- D'Angelo et al. (1989). *J Appl Physiol*. [Respiratory stress adaptation](https://pubmed.ncbi.nlm.nih.gov/2606863/).
- Pelosi et al. (1998). *Anesth Analg*. [Body mass index, functional residual capacity, and compliance under anesthesia](https://pubmed.ncbi.nlm.nih.gov/9728848/).
- Dräger. [Evita 4 SW 4.n instructions for use](https://www.draeger.com/Content/Documents/Content/IfU_Evita_4_SW_4.n_EN_9039485.pdf).
- Feigenwinter and Zbinden (1991). *Anaesthesist*. [Circle-system resistance](https://pubmed.ncbi.nlm.nih.gov/1952036/).

## Monitors

- Mahdi et al. (2017). *Physiol Meas*. [Synthetic arterial pressure waveform](https://pubmed.ncbi.nlm.nih.gov/28176674/).
- Franklin et al. (1997). *Circulation*. [Pulse pressure by age in the Framingham Heart Study](https://pubmed.ncbi.nlm.nih.gov/9236450/).
- Langewouters et al. (1984). *J Biomech*. [Pressure dependence of human aortic compliance](https://pubmed.ncbi.nlm.nih.gov/6480618/).
- Saugel et al. (2020). *Crit Care*. [Arterial catheter frequency response and damping](https://pubmed.ncbi.nlm.nih.gov/32331527/).
- Russell et al. (1990). *Can J Anaesth*. [Arterial to end-tidal CO₂ gradient during postoperative support](https://pubmed.ncbi.nlm.nih.gov/2115404/).
- Lujan et al. (2008). *Med Sci Monit*. [Arterial to end-tidal CO₂ gradient in airway obstruction](https://pubmed.ncbi.nlm.nih.gov/18758420/).
- Idris et al. (1994). *Ann Emerg Med*. [End-tidal CO₂ during extremely low cardiac output](https://pubmed.ncbi.nlm.nih.gov/8135436/).
- Kim et al. (2019). *Am J Emerg Med*. [Arterial to end-tidal CO₂ gap after cardiac arrest](https://pubmed.ncbi.nlm.nih.gov/29685358/).
- Tanaka et al. (2014). *J Clin Monit Comput*. [Capnographic detection of respiratory pauses](https://pubmed.ncbi.nlm.nih.gov/24420342/).
- Poorzargar et al. (2022). *J Clin Monit Comput*. [Pulse oximetry in poor perfusion](https://pubmed.ncbi.nlm.nih.gov/35119597/).
- Broome et al. (1992). *Anaesthesia*. [Finger pulse-oximeter response delay during anesthesia](https://pubmed.ncbi.nlm.nih.gov/1536395/).

## Scenario guidance

- Stanford Anesthesia Cognitive Aid Program (2022). [Emergency Manual v4.4](https://web.stanford.edu/dept/anesthesia/em/epic-manual.pdf).
- Merck (2026). [Bridion (sugammadex) prescribing information](https://www.merck.com/product/usa/pi_circulars/b/bridion/bridion_pi.pdf).
- Blink Device Company (2024). [TwitchView operating manual](https://www.blinkdc.com/hubfs/PN00211%20TwitchView%20IFU%20Rev%20N.pdf).
- Thilen et al. (2023). *Anesthesiology*. [ASA guidelines for monitoring and antagonism of neuromuscular block](https://pubmed.ncbi.nlm.nih.gov/36520073/).
- ANZAAG and ANZCA (2022). [Immediate management of perioperative anaphylaxis in adults](https://www.anzca.edu.au/getContentAsset/d38d29ae-74f0-4136-8372-1d342c594e11/80feb437-d24d-46b8-a858-4a2a28b9b970/Anaphylaxis-Card-1-Adult-Immediate-Management-2022.pdf?language=en).
- Surviving Sepsis Campaign (2026). [International guidelines for management of sepsis and septic shock](https://sccm.org/clinical-resources/guidelines/guidelines/surviving-sepsis-campaign-international-guidelines-for-management-of-sepsis-and-septic-shock-2026).
- Kietaibl et al. (2023). *Eur J Anaesthesiol*. [ESAIC guidelines for severe perioperative bleeding](https://pubmed.ncbi.nlm.nih.gov/36855941/).
- Xing et al. (2018). *Pain Med*. [Lidocaine pretreatment for propofol injection pain](https://pubmed.ncbi.nlm.nih.gov/28525614/).
- Anesthesia Patient Safety Foundation (2019). [Nitrogen contamination of an operating room oxygen pipeline](https://www.apsf.org/article/nitrogen-contamination-of-operating-room-oxygen-pipeline/).
- Malignant Hyperthermia Association of the United States. [Managing a crisis](https://www.mhaus.org/healthcare-professionals/managing-a-crisis/).
- European Malignant Hyperthermia Group (2024). [Recognising and managing a malignant hyperthermia crisis](https://www.emhg.org/recommendations-1/2024/11/4/recognising-and-managing-a-malignant-hyperthermia-crisis-v2024).
