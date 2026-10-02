# Model references

## Simulator-specific models

AnaSim adapts or calibrates the following models. The linked tests define
expected response ranges.

| Area | Implementation |
|------|----------------|
| Hemodynamic PK scaling | AnaSim scales volume and clearance with hemodynamics. See [PK assumptions and limits](ARCHITECTURE.md#pk-and-tci). |
| Respiratory drug effects | [`RespiratoryModel`](../anasim/physiology/respiration.py) combines published effects on ventilation and the hypercapnic response, with parameters calibrated for AnaSim. [Respiratory tests](../tests/test_respiration.py). |
| Neuromuscular block and reversal | [`TOFModel`](../anasim/patient/pd/nmba.py) uses adductor pollicis and central (diaphragm and larynx) effect sites, spontaneous recovery, and simplified sugammadex binding. Both central muscles use laryngeal kinetics. AnaSim calibrates onset and recovery constants. [Pharmacology tests](../tests/test_pharmacology.py). |
| Baroreflex and myocardial hypoxia | [`HemodynamicConfig`](../anasim/physiology/hemo_config.py) uses published anesthetic effects on reflexes and hypoxic arrest thresholds. AnaSim calibrates reflex gains, set-point reset, and hypoxia time constants. [Hemodynamic tests](../tests/test_hemodynamics.py). |
| Vasoactive drug effects | [`HemodynamicConfig`](../anasim/physiology/hemo_config.py) models concentration-response curves for epinephrine, phenylephrine, vasopressin, dobutamine, and milrinone. Published data set response direction and approximate size; AnaSim calibrates combined effects. Epinephrine fits arterial infusion (Freyschuss 1986) and IV bolus (Takahashi 2002) data. [Epinephrine](../tests/test_epinephrine.py) and [hemodynamic](../tests/test_hemodynamics.py) tests. |
| Fentanyl, midazolam, etomidate, and ketamine | [`anesthesia.py`](../anasim/patient/pd/anesthesia.py) converts fentanyl to remifentanil equivalents using isoflurane MAC reduction (McEwan 1993; Lang 1996). Etomidate (Kaneda 2011) and midazolam (Albrecht 1999) become propofol equivalents at equal loss-of-response concentrations, with midazolam-propofol synergy from Short 1992. AnaSim calibrates etomidate's ventilatory effect (Valk 2021), the maximum midazolam equivalent, and ketamine's laryngoscopy potency and sympathetic response (Idvall 1979). [Adjunct tests](../tests/test_anesthetic_adjuncts.py). |
| Adrenergic and muscarinic blockade | [`HemodynamicConfig`](../anasim/physiology/hemo_config.py) uses antagonist potencies from Sum 1983 (esmolol), Abernethy 1987 and Hafsa 2022 (labetalol), and Ali-Melkkilä 1993 (glycopyrrolate). AnaSim calibrates sympathetic and vagal contributions to resting tone and reflexes. Unlike the BP recovery within 30 minutes reported by Abernethy, part of AnaSim's labetalol BP reduction persists during beta blockade because the sinus-rhythm baroreflex adjusts only HR. [Autonomic drug tests](../tests/test_autonomic_drugs.py). |
| Lung mechanics and ventilator | [`RespiratoryMechanics`](../anasim/physiology/resp_mech.py) uses published viscoelastic parameters, compliance fitted to VitalDB data, and a simplified inspiratory muscle-pressure profile. [`AnesthesiaVentilator`](../anasim/machine/ventilator.py) uses GE Aisys CS2 modes and ranges, with defaults informed by Primus recordings and device conventions. See [waveform sources and limits](#ventilator-waveforms) and [ventilator tests](../tests/test_ventilator.py). |
| Capnography | [`Capnograph`](../anasim/monitors/capno.py) follows exhaled gas through the series dead space (Fowler 1948), with analyzer response, phase II spread, and phase III slope fitted to Primus recordings. [Capnography tests](../tests/test_nibp_capno.py). |
| Arterial pressure waveform | [`ArterialWaveformRenderer`](../anasim/monitors/arterial.py) uses Mahdi et al.'s pressure landmarks with Su MAP and stroke volume. [`ArterialLineMonitor`](../anasim/monitors/arterial.py) models catheter and transducer dynamics. [Waveform](../tests/test_arterial_waveform.py) and [arterial line](../tests/test_arterial_line.py) tests. |

## Supported patient domain

The patient limits (age 18 to 70 years, weight 50 to 100 kg, height 150 to
200 cm, BMI 18 to 32 kg/m²) slightly extend the ranges in Li et al.'s 36-volunteer
healthy-adult cohort: age 18 to 70 years, weight 51.5 to 94.8 kg, height 151 to
196 cm, and BMI 18.0 to 31.1 kg/m². Su et al. built the hemodynamic model from a
cohort of the same size and age strata and found age to be a strong covariate
of the propofol effect on stroke volume.

Hemoglobin, hematocrit, and organ-function limits are simulator choices. See
the [CLI guide](CLI_USAGE.md#fields) for all input limits.

## Hemodynamics and physiology

- Su et al. Br J Anaesth. 2023. (mechanistic hemodynamic interaction model). [PubMed](https://pubmed.ncbi.nlm.nih.gov/37355412/)
- Beloeil et al. Br J Anaesth. 2005. (norepinephrine PK/PD in septic shock/trauma). [PubMed](https://pubmed.ncbi.nlm.nih.gov/16227334/)
- Clutter et al. J Clin Invest. 1980. (epinephrine cardiovascular effects). [PubMed](https://pubmed.ncbi.nlm.nih.gov/6995479/)
- Ebert et al. Anesth Analg. 1995. (sevoflurane cardiovascular responses). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7486143/)
- ESC/ERS Task Force. Eur Heart J. 2022. (RHC normal ranges; RAP 2-6 mmHg, PVR 0.3-2.0 WU). [PubMed](https://pubmed.ncbi.nlm.nih.gov/36017548/)
- Segeroth et al. Eur Heart J Cardiovasc Imaging. 2023. (pulmonary transit time; median ~6.8 s with normal biventricular EF). [PubMed](https://pubmed.ncbi.nlm.nih.gov/36662127/)
- Koganov et al. Crit Care Med. 1997. (PEEP raises pulmonary vascular resistance). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9187594/)
- Carlsson et al. Acta Anaesthesiol Scand. 1985. (hypoxic pulmonary vasoconstriction; PVR increases with unilateral hypoxia). [PubMed](https://pubmed.ncbi.nlm.nih.gov/3993324/)
- Sessler. Anesthesiology. 2000. (perioperative heat balance). [PubMed](https://pubmed.ncbi.nlm.nih.gov/10691247/)
- Sessler. Lancet. 2016. (review of perioperative thermoregulation). [PubMed](https://pubmed.ncbi.nlm.nih.gov/26775126/)
- Matsukawa et al. Anesthesiology. 1995. (core temperature falls 1.6 °C in the first hour, 81% from redistribution). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7879935/)
- Frank et al. JAMA. 1997. (perioperative thermoregulation). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9087467/)
- Anderson et al. Paediatr Anaesth. 2017. (phenylephrine PK; PD here is heuristic). [PubMed](https://pubmed.ncbi.nlm.nih.gov/28868789/)
- Magnani et al. J Int Med Res. 1977. (dobutamine dose-response; ↑CO, low-dose SV increase without HR change). [PubMed](https://pubmed.ncbi.nlm.nih.gov/838109/)
- Baim et al. N Engl J Med. 1983. (milrinone hemodynamics; ↑CI, ↓SVR). [PubMed](https://pubmed.ncbi.nlm.nih.gov/6888453/)
- Martin et al. Acta Anaesthesiol Scand. 1990. (hyperdynamic septic shock: SVR <= 600; target 700-800 dyn·s/cm^5). [PubMed](https://pubmed.ncbi.nlm.nih.gov/2389659/)
- Melo et al. Crit Care. 1999. (septic shock low SVR cohort: mean SVR ~445 dyn·s/cm^5). [PubMed](https://pubmed.ncbi.nlm.nih.gov/11056727/)
- Meng et al. Br J Anaesth. 2011. (phenylephrine 100-200 mcg under propofol-remifentanil, first treatment: MAP +29.5 mmHg, HR -17 bpm, CO -1.7 L/min). [PubMed](https://pubmed.ncbi.nlm.nih.gov/21642644/)
- Ebert et al. Anesth Analg. 1994. (propofol reduces cardiac baroreflex sensitivity to falling pressure by 60%). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8311293/)
- Sato et al. Br J Anaesth. 2005. (baroreflex control of heart rate during propofol infusion). [PubMed](https://pubmed.ncbi.nlm.nih.gov/15722386/)
- Umehara et al. Anesth Analg. 2006. (sevoflurane depresses carotid-cardiac baroreflex gain). [PubMed](https://pubmed.ncbi.nlm.nih.gov/16368802/)
- Mort. J Clin Anesth. 2004. (83% of cardiac arrests during emergency intubation were associated with SpO2 < 70%). [PubMed](https://pubmed.ncbi.nlm.nih.gov/15590254/)
- De Jong et al. Crit Care Med. 2018. (cardiac arrest related to intubation, multicenter cohort). [PubMed](https://pubmed.ncbi.nlm.nih.gov/29261566/)
- Heffner et al. Resuscitation. 2013. (incidence and factors of cardiac arrest complicating emergency airway management). [PubMed](https://pubmed.ncbi.nlm.nih.gov/23911630/)
- de Keijzer et al. Eur J Anaesthesiol. 2026. (norepinephrine dose-MAP slope in healthy volunteers: 103 mmHg per mcg/kg/min awake, about 222 under general anesthesia). [PubMed](https://pubmed.ncbi.nlm.nih.gov/41481868/)
- Stratton et al. J Appl Physiol. 1985. (epinephrine 25-100 ng/kg/min in awake men: HR +8 to +17 bpm, MAP -3 to -9 mmHg, SV +26% to +40%, SVR -31% to -48%; peripheral venous sampling). [PubMed](https://pubmed.ncbi.nlm.nih.gov/3988675/)
- Freyschuss et al. Clin Sci (Lond). 1986. (arterial epinephrine 0.27 to 1.34, 2.30, and 6.02 nmol/L; increased SV and CO, decreased SVR, no significant MAP change). [PubMed](https://pubmed.ncbi.nlm.nih.gov/3956110/)
- Takahashi et al. Anesth Analg. 2002. (5, 10, and 15 mcg IV epinephrine with lidocaine under propofol/N2O; HR peaks before SBP, followed by a late HR fall). [PubMed](https://pubmed.ncbi.nlm.nih.gov/11867404/)
- Bellissant et al. Clin Pharmacol Ther. 2000. (septic shock pressor hyporesponsiveness: phenylephrine Emax ~39 vs 84 mmHg controls). [PubMed](https://pubmed.ncbi.nlm.nih.gov/11014411/)
- Margarson et al. J Appl Physiol (1985). 2002. (septic shock TER of albumin ~6.7%/h). [PubMed](https://pubmed.ncbi.nlm.nih.gov/11960967/)
- Persichini et al. Crit Care Med. 2012. (mean systemic pressure decreases with reduced norepinephrine in septic shock). [PubMed](https://pubmed.ncbi.nlm.nih.gov/22926333/)
- Clark et al. J Am Coll Cardiol. 1997. (irregular AF R-R sequence vs regular pacing at the same rate: CO 4.4 vs 5.2 L/min, higher PCWP). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9316536/)
- Hardman et al. Cardiovasc Res. 1998. (AF stroke volume rises curvilinearly with the preceding R-R interval). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9683909/)
- Kerr et al. Am J Cardiol. 1998. (AF beat-to-beat stroke volume variability increases with ventricular rate). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9874054/)
- Corino et al. J Cardiovasc Electrophysiol. 2015. (permanent AF R-R rMSSD 171 ms at baseline, 226-256 ms on rate control). [PubMed](https://pubmed.ncbi.nlm.nih.gov/25367150/)
- Hogue et al. Anesthesiology. 1996. (post-induction sinus bradycardia lowered CO and blood pressure with unchanged stroke volume; atrial pacing at 65-80 bpm restored them). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8694384/)

## Cardiovascular monitor models

- Mahdi, Clifford, and Payne. Physiol Meas. 2017. (synthetic ABP model based on systolic, diastolic, dicrotic-notch, and dicrotic-peak pressure points). [PubMed](https://pubmed.ncbi.nlm.nih.gov/28176674/)
- Saugel et al. Crit Care. 2020. (arterial catheter waveform quality, natural frequency, and damping). [PubMed](https://pubmed.ncbi.nlm.nih.gov/32331527/)

## Volatile anesthetics (PBPK and MAC)

- Davis and Mapleson. Br J Anaesth. 1981. (physiological model of inhaled anesthetics). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7225273/)
- Mapleson. Br J Anaesth. 1996. (age-related MAC formula). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8777094/)
- Nickalls and Mapleson. Br J Anaesth. 2003. (MAC40 sevoflurane 1.80%). [PubMed](https://pubmed.ncbi.nlm.nih.gov/12878613/)
- Yasuda et al. Anesth Analg. 1991. (sevoflurane vs isoflurane uptake/washout). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1994760/)
- Carpenter et al. Anesth Analg. 1986. (inhaled anesthetic kinetics in humans). [PubMed](https://pubmed.ncbi.nlm.nih.gov/3706798/)
- Hornbein et al. Anesth Analg. 1982. (MAC of nitrous oxide in humans; ~1.04 atm absolute). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7201254/)
- Schuh. Anaesthesist. 1975. (side effects of nitrous oxide; blood:gas partition coefficient of N2O ~0.47). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1190419/)
- Goto et al. Anesthesiology. 2000. (MACawake of nitrous oxide ~63% at 1 atm). [PubMed](https://pubmed.ncbi.nlm.nih.gov/11046204/)
- Katoh et al. Br J Anaesth. 1997. (sevoflurane MACawake ~0.63%; N2O 45% reduces MACawake ~50%, less than additive). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9389264/)

## PK and PD models (IV agents and vasoactive drugs)

- Janmahasatian et al. Clin Pharmacokinet. 2005. (lean bodyweight equation across body sizes). [PubMed](https://pubmed.ncbi.nlm.nih.gov/16176118/)
- James. Research on Obesity. HMSO. 1976. (lean body mass covariate used by the Schnider and Minto models).
- Marsh et al. Br J Anaesth. 1991. (propofol PK). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1859758/)
- Thomson et al. Anaesthesia. 2014. (effect-site TCI with Marsh `ke0` values of 0.6 and 1.2 min⁻¹). [PubMed](https://pubmed.ncbi.nlm.nih.gov/24738800/)
- Schnider et al. Anesthesiology. 1998. (propofol PK). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9605675/)
- Eleveld et al. Br J Anaesth. 2018. (propofol PK/PD). [PubMed](https://pubmed.ncbi.nlm.nih.gov/29661412/)
- Servin et al. Br J Anaesth. 1990. (propofol PK in cirrhosis; Vd increases, clearance not significantly reduced). [PubMed](https://pubmed.ncbi.nlm.nih.gov/2223333/)
- Hiraoka et al. Br J Clin Pharmacol. 2005. (renal extraction of propofol; kidneys ~1/3 of total clearance). [PubMed](https://pubmed.ncbi.nlm.nih.gov/16042671/)
- Minto et al. Anesthesiology. 1997. (remifentanil PK). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9009936/)
- Dershwitz et al. Anesthesiology. 1996. (remifentanil PK/PD in severe liver disease; PK unchanged). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8638835/)
- Hoke et al. Anesthesiology. 1997. (remifentanil PK in renal failure; PK unchanged, metabolite accumulates). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9316957/)
- Bae et al. Br J Anaesth. 2020. (allometric fentanyl PK from 95 adults). [PubMed](https://pubmed.ncbi.nlm.nih.gov/32861508/)
- Scott and Stanski. J Pharmacol Exp Ther. 1987. (fentanyl ke0 0.147 min⁻¹; lower dose requirements with age). [PubMed](https://pubmed.ncbi.nlm.nih.gov/3100765/)
- McEwan et al. Anesthesiology. 1993. (fentanyl 1.67 ng/mL halves isoflurane MAC). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8489058/)
- Lang et al. Anesthesiology. 1996. (remifentanil 1.37 ng/mL halves isoflurane MAC). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8873541/)
- Albrecht et al. Clin Pharmacol Ther. 1999. (midazolam PK/PD by age; loss-of-response Ce 499 ng/mL at 24-28 years and 210 at 67-81). [PubMed](https://pubmed.ncbi.nlm.nih.gov/10391668/)
- Short, Plummer, and Chui. Br J Anaesth. 1992. (midazolam-propofol hypnotic synergy; ED50 37% below additive). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1389820/)
- Arden, Holley, and Stanski. Anesthesiology. 1986. (etomidate PK/PD by age; V1 falls 42% from 22 to 80 years, clearance about 2 mL/kg/min per decade, t1/2 ke0 1.6 min). [PubMed](https://pubmed.ncbi.nlm.nih.gov/3729056/)
- Van Hamme et al. Anesthesiology. 1978. (etomidate PK after 0.3 mg/kg; hepatic clearance about half of hepatic blood flow). [PubMed](https://pubmed.ncbi.nlm.nih.gov/697083/)
- Kaneda et al. J Clin Pharmacol. 2011. (etomidate PK/PD; OAA/S Ce50 0.554 and BIS Ce50 0.526 mcg/mL, t1/2 ke0 1.55 min). [PubMed](https://pubmed.ncbi.nlm.nih.gov/20498288/)
- Valk and Struys. Clin Pharmacokinet. 2021. (etomidate review; 0.3 mg/kg gives 5-10 min of hypnosis with minimal hemodynamic change, brief apnea, and less CO2-response depression). [PubMed](https://pubmed.ncbi.nlm.nih.gov/34060021/)
- Kamp et al. Anesthesiology. 2020. (ketamine PK meta-analysis and three-compartment model). [PubMed](https://pubmed.ncbi.nlm.nih.gov/32997732/)
- Idvall et al. Br J Anaesth. 1979. (ketamine-N2O anesthesia at 2.2 mcg/mL, awakening at 0.64 mcg/mL; pressure, HR, and CO up 15-30%). [PubMed](https://pubmed.ncbi.nlm.nih.gov/526385/)
- Bourke, Malit, and Smith. Anesthesiology. 1987. (ketamine 3 mg/kg shifts the CO2 response 2 mmHg). [PubMed](https://pubmed.ncbi.nlm.nih.gov/3101549/)
- Wierda et al. Can J Anaesth. 1991. (rocuronium PK). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1829656/)
- Masui et al. J Anesth. 2018. (rocuronium PD models; age and sex covariates, age-dependent ke0). [PubMed](https://pubmed.ncbi.nlm.nih.gov/30099599/)
- Magorian et al. Anesth Analg. 1995. (rocuronium PK in liver disease; Vd increases, clearance unchanged). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7893030/)
- Robertson et al. Eur J Anaesthesiol. 2005. (rocuronium PK in renal failure; clearance reduced). [PubMed](https://pubmed.ncbi.nlm.nih.gov/15816565/)
- Ensinger et al. Eur J Anaesthesiol. 1992. (arterial epinephrine at 0.2 mcg/kg/min; clearance about 0.046 L/kg/min). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1425612/)
- Ensinger et al. Eur J Clin Pharmacol. 1992. (arterial vs peripheral venous catecholamine concentrations). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1425886/)
- Abboud et al. Crit Care. 2009. (epinephrine PK in adult septic shock). [PubMed](https://pubmed.ncbi.nlm.nih.gov/19622169/)
- Li et al. Clin Pharmacokinet. 2024. (norepinephrine PK with propofol interaction). [PubMed](https://pubmed.ncbi.nlm.nih.gov/39465453/)
- Ploeger et al. Anesthesiology. 2009. (sugammadex PK/PD modeling). [PubMed](https://pubmed.ncbi.nlm.nih.gov/19104176/)
- Kleijn et al. Br J Clin Pharmacol. 2011. (sugammadex PK/PD). [PubMed](https://pubmed.ncbi.nlm.nih.gov/21535448/)
- Pühringer et al. Br J Anaesth. 2010. (sugammadex reversal times). [PubMed](https://pubmed.ncbi.nlm.nih.gov/20876699/)
- Cantineau et al. Anesthesiology. 1994. (rocuronium ED50 0.26 vs 0.14 mg/kg and ED95 0.50 vs 0.24 mg/kg at the diaphragm vs the adductor pollicis; diaphragm onset slower but recovery faster). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8092503/)
- Plaud et al. Clin Pharmacol Ther. 1995. (rocuronium at the laryngeal adductors vs adductor pollicis: ke0 half-time 2.7 vs 4.4 min, Ce50 1424 vs 823 mcg/L). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7648768/)
- Wright et al. Anesthesiology. 1994. (laryngeal adductors are more resistant to rocuronium than the adductor pollicis). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7978469/)
- Eikermann et al. Anesthesiology. 2003. (TOF ratio vs recovery of respiratory function; upper airway obstruction during partial block). [PubMed](https://pubmed.ncbi.nlm.nih.gov/12766640/)
- Eikermann et al. Am J Respir Crit Care Med. 2007. (predisposition to inspiratory upper airway collapse during partial neuromuscular blockade). [PubMed](https://pubmed.ncbi.nlm.nih.gov/17023729/)
- Fiset et al. Can J Anaesth. 1991. (N2O potentiates vecuronium neuromuscular blockade; ED95 reduction). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1683819/)
- Nguyen-Lee et al. Curr Anesthesiol Rep. 2018. (sugammadex PK review). [DOI](https://doi.org/10.1007/s40140-018-0266-5)
- Bouillon et al. Anesthesiology. 2004. (propofol/remifentanil interaction for tolerance of laryngoscopy). [PubMed](https://pubmed.ncbi.nlm.nih.gov/15166553/)
- Kern et al. Anesthesiology. 2004. (response surface analysis of propofol-remifentanil interaction). [PubMed](https://pubmed.ncbi.nlm.nih.gov/15166554/)
- FDA NDA 203826 Clinical Pharmacology Review. 2012. (phenylephrine PK). [Drugs@FDA](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2012/203826_phenylephrine_toc.cfm)
- Hengstmann and Goronzy. Eur J Clin Pharmacol. 1982. (phenylephrine PK). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7056280/)
- Vasopressin injection label (DailyMed). (Vd 0.14 L/kg; CL 9-25 mL/min/kg; t1/2 ≤10 min). [DailyMed](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=971f9b1c-6094-4f80-920b-cb5d7e62950a)
- Kates and Leier. Clin Pharmacol Ther. 1978. (dobutamine PK in severe CHF; CL 2.35 L/min/m^2; Vd 0.20 L/kg; t1/2 ~2 min). [PubMed](https://pubmed.ncbi.nlm.nih.gov/699477/)
- Daly et al. Am J Cardiol. 1997. (dobutamine plasma concentrations vs dose, 5-30 µg/kg/min). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9165162/)
- Dobutamine injection label (Clinical Pharmacology). (rapid onset/offset; t1/2 ~2 min). [Pfizer Medical Information](https://www.pfizermedicalinformation.com/patient/dobutamine/clinical-pharmacology)
- Sum et al. Clin Pharmacol Ther. 1983. (esmolol two-compartment PK at 400 mcg/kg/min; steady-state 0.569 mcg/mL at 150; isoproterenol tachycardia blocked about 50% at 0.3 and 80% at 1 mcg/mL). [PubMed](https://pubmed.ncbi.nlm.nih.gov/6617063/)
- Wiest and Haney. Clin Pharmacokinet. 2012. (esmolol review; 90% beta block 5 minutes after a load, recovery 18-30 minutes after stopping). [PubMed](https://pubmed.ncbi.nlm.nih.gov/22515557/)
- Abernethy et al. Am J Cardiol. 1987. (labetalol 50 mg IV in young and elderly hypertensive adults; clearance 19.4 vs 13.9 mL/min/kg; HR fell for 3 h and BP for less than 30 minutes). [PubMed](https://pubmed.ncbi.nlm.nih.gov/3661438/)
- Hafsa et al. Pharmaceutics. 2022. (labetalol PBPK; 0.5 mg/kg IV Cmax about 143 ng/mL; IV beta:alpha potency 6.9:1; hepatic glucuronidation). [PubMed](https://pubmed.ncbi.nlm.nih.gov/36365181/)
- Ali-Melkkilä, Kanto, and Iisalo. Acta Anaesthesiol Scand. 1993. (anticholinergic PK/PD review; glycopyrrolate 11.8 ng/mL prevented the oculocardiac reflex and 3.1 ng/mL was insufficient; 80% excreted unchanged in urine). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8249551/)
- Du et al. J Drug Deliv Sci Technol. 2025. (glycopyrrolate three-compartment population PK after general anesthesia). [DOI](https://doi.org/10.1016/j.jddst.2025.106692)
- Milrinone lactate injection label (DailyMed). (Vd 0.38-0.45 L/kg; CL ~0.13 L/kg/hr; t1/2 2.3-2.4 h; hemodynamic improvement within ~5-15 min). [DailyMed](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=88f78780-399f-4780-be19-205739db1682&version=21)

## Respiratory control and BIS

- Kanazawa et al. J Anesth. 2017. (BIS at age-adjusted 1 MAC desflurane vs sevoflurane). [PubMed](https://pubmed.ncbi.nlm.nih.gov/28791477/)
- Ryu et al. Anesthesiology. 2018. (BIS and SPI at 1 MAC desflurane vs sevoflurane under tetanic stimulation). [PubMed](https://pubmed.ncbi.nlm.nih.gov/29509579/)
- Ryu et al. Br J Anaesth. 2018. (remifentanil requirements at 1 MAC desflurane vs sevoflurane). [PubMed](https://pubmed.ncbi.nlm.nih.gov/30336856/)
- Paraskeva et al. J Clin Anesth. 2005. (BIS ~30 at 1.5 MAC sevoflurane; physostigmine no effect). [PubMed](https://pubmed.ncbi.nlm.nih.gov/16427526/)
- Schwab et al. Anesth Analg. 2004. (BIS response to sevoflurane). [PubMed](https://pubmed.ncbi.nlm.nih.gov/15562061/)
- Olofsen et al. Anesthesiology. 2002. (remifentanil does not change the sevoflurane BIS C50). [PubMed](https://pubmed.ncbi.nlm.nih.gov/11873028/)
- Schumacher et al. Anesthesiology. 2009. (additive propofol-sevoflurane interaction on BIS). [PubMed](https://pubmed.ncbi.nlm.nih.gov/19741484/)
- Barr et al. Br J Anaesth. 1999. (N2O can produce LOC without lowering BIS). [PubMed](https://pubmed.ncbi.nlm.nih.gov/10562773/)
- Hirota et al. Eur J Anaesthesiol. 1999. (adding N2O to propofol-fentanyl: BIS changes are small/variable). [PubMed](https://pubmed.ncbi.nlm.nih.gov/10713872/)
- Ozcan et al. J Neurosurg Anesthesiol. 2010. (sevoflurane or propofol + N2O: BIS/entropy effects modest). [PubMed](https://pubmed.ncbi.nlm.nih.gov/20844378/)
- Babenco et al. Anesthesiology. 2000. (opioid effects on ventilatory control). [PubMed](https://pubmed.ncbi.nlm.nih.gov/10691225/)
- Lee et al. Korean J Anesthesiol. 2011. (propofol effect-site EC50 for respiratory depression). [PubMed](https://pubmed.ncbi.nlm.nih.gov/21927681/)
- Blouin et al. Anesthesiology. 1993. (propofol depresses hypoxic ventilatory response). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8267192/)
- Nieuwenhuijs et al. Anesthesiology. 2001. (propofol effects on ventilatory control). [PubMed](https://pubmed.ncbi.nlm.nih.gov/11605929/)
- Glass et al. Anesthesiology. 1999. (remifentanil ventilatory depression vs CO2). [PubMed](https://pubmed.ncbi.nlm.nih.gov/10360852/)
- Pandit et al. Br J Anaesth. 1999. (hypercapnic ventilatory response at 0.1 MAC sevoflurane). [PubMed](https://pubmed.ncbi.nlm.nih.gov/10618930/)
- Doi and Ikeda. Anesth Analg. 1987. (sevoflurane depresses CO2 response at 1.1-1.4 MAC). [PubMed](https://pubmed.ncbi.nlm.nih.gov/3826666/)
- Duffin. Respir Physiol Neurobiol. 2011. (ventilatory response modeling). [PubMed](https://pubmed.ncbi.nlm.nih.gov/21514404/)
- Bissinger et al. Anasthesiol Intensivmed Notfallmed Schmerzther. 1993. (curare cleft; capnography vs relaxometry). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7902740/)
- Russell et al. Can J Anaesth. 1990. (PaCO2-EtCO2 gradient during postoperative support). [PubMed](https://pubmed.ncbi.nlm.nih.gov/2115404/)
- Lujan et al. Med Sci Monit. 2008. (PaCO2-EtCO2 gradient increases with obstruction; n=120). [PubMed](https://pubmed.ncbi.nlm.nih.gov/18758420/)
- Idris et al. Ann Emerg Med. 1994. (EtCO2 during extremely low cardiac output). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8135436/)
- Kim et al. Am J Emerg Med. 2019. (post-arrest PaCO2-EtCO2 gap). [PubMed](https://pubmed.ncbi.nlm.nih.gov/29685358/)
- Poorzargar et al. J Clin Monit Comput. 2022. (pulse oximeter accuracy in poor peripheral perfusion). [PubMed](https://pubmed.ncbi.nlm.nih.gov/35119597/)
- Sinex. Am J Emerg Med. 1999. (pulse oximetry principles and limitations). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9928703/)
- Broome et al. Anaesthesia. 1992. (finger pulse-ox response delay during anesthesia). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1536395/)
- Tanaka et al. J Clin Monit Comput. 2014. (capnographic detection of respiratory pauses during sedation). [PubMed](https://pubmed.ncbi.nlm.nih.gov/24420342/)
- Byrne et al. J Appl Physiol (1985). 2005. (resting VO2/MET reference values; large cohort). [PubMed](https://pubmed.ncbi.nlm.nih.gov/15831804/)
- Stein et al. Chest. 1995. (A-a gradient age adjustment formula in PE assessment). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7632205/)
- Benumof et al. Anesthesiology. 1997. (time to SaO2 < 90% after preoxygenation; about 8 min in healthy adults). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9357902/)
- Nimmagadda et al. Anesth Analg. 2017. (preoxygenation endpoint: EtO2 90%). [PubMed](https://pubmed.ncbi.nlm.nih.gov/28099321/)
- Hardman et al. Anesth Analg. 2000. (physiological model of the onset and course of hypoxemia during apnea). [PubMed](https://pubmed.ncbi.nlm.nih.gov/10702447/)
- Eastwood et al. Anesthesiology. 2005. (upper airway critical closing pressure rises with propofol depth: -0.3, +0.5, +1.4 cmH2O at 2.5, 4.0, 6.0 mcg/mL). [PubMed](https://pubmed.ncbi.nlm.nih.gov/16129969/)
- Hillman et al. Anesthesiology. 2009. (upper airway collapsibility rises abruptly at loss of consciousness). [PubMed](https://pubmed.ncbi.nlm.nih.gov/19512872/)
- Larsson et al. J Allergy Clin Immunol. 1985. (circulating epinephrine relieves allergen-induced bronchoconstriction; AnaSim calibrates the C50 and maximum relief). [PubMed](https://pubmed.ncbi.nlm.nih.gov/3989143/)
- Farmery and Roe. Br J Anaesth. 1996. (model of oxyhemoglobin desaturation during apnea). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8777112/)
- Stock et al. J Clin Anesth. 1989. (PaCO2 rise in anesthetized patients with airway obstruction). [PubMed](https://pubmed.ncbi.nlm.nih.gov/2516732/)
- Kobayashi et al. Masui. 1994. (arterial blood gas changes during apnea under anesthesia). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7933492/)
- Miyamura et al. Jpn J Physiol. 1980. (ventilatory response to CO2 by rebreathing; HCVR slope variability). [PubMed](https://pubmed.ncbi.nlm.nih.gov/6790801/)
- Christie et al. Anesth Analg. 1992. (PSV decreases inspiratory work during GA with spontaneous ventilation). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1632530/)
- Lim et al. Paediatr Anaesth. 2012. (PSV vs spontaneous ventilation via ProSeal LMA in children; improved ventilation). [PubMed](https://pubmed.ncbi.nlm.nih.gov/22380745/)
- Capdevila et al. PLoS One. 2014. (PSV vs CMV/SB with LMA; emergence time/ventilatory function). [PubMed](https://pubmed.ncbi.nlm.nih.gov/25536515/)

## Ventilator waveforms

Volume-control waveforms, capnography, and compliance were compared with
Dräger Primus recordings from VitalDB in adults within the supported range.
Separate recordings were used to fit and check the parameters. These
comparisons informed compliance scaling, monitor response, and ventilator
defaults; published studies and device specifications supplied the remaining
parameters.

Compliance scales with predicted body weight and BMI, using a fit at PEEP
5 cmH2O. The model omits PEEP-dependent compliance and does not reproduce all
recorded inspiratory pressure shapes. Bronchospasm resistance is calibrated
against measurements in ventilated patients with severe asthma.

- Lee et al. Sci Data. 2022. (VitalDB: intraoperative waveforms and numerics from 6,388 surgical patients; CC BY-NC-SA 4.0). [PubMed](https://pubmed.ncbi.nlm.nih.gov/35676300/)
- Jonson et al. J Appl Physiol. 1993. (healthy anesthetized humans: viscoelastic resistance 3.7 cmH2O/(L/s), compliance 4x static, time constant 0.82 s). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8376259/)
- D'Angelo et al. J Appl Physiol. 1989. (stress adaptation adds viscoelastic resistance in anesthetized paralyzed humans). [PubMed](https://pubmed.ncbi.nlm.nih.gov/2606863/)
- Pelosi et al. Anesth Analg. 1998. (respiratory system compliance falls exponentially with BMI during general anesthesia). [PubMed](https://pubmed.ncbi.nlm.nih.gov/9728848/)
- Tobin et al. Chest. 1983. (breathing pattern of normal subjects). [PubMed](https://pubmed.ncbi.nlm.nih.gov/6872603/)
- Fowler. Am J Physiol. 1948. (anatomic dead space from the exhaled CO2 front). [PubMed](https://pubmed.ncbi.nlm.nih.gov/18101134/)
- GE Healthcare. Aisys CS2 specifications. DOC1261976 rev3. 2013. (modes, setting ranges, and circuit resistance). [PDF](https://www.oxygen-care.com/wp-content/uploads/2025/05/SPEC-SHEET-Aisys-CS2.pdf)
- Sydow et al. Intensive Care Med. 1993. (status asthmaticus: inspiratory resistance 22.7 ± 7.0 cmH2O/(L/s), Ppeak 43 cmH2O). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8294630/)
- Okayama et al. J Asthma. 1991. (status asthmaticus: airway resistance 13-17 mmHg/(L/s) by interruption). [PubMed](https://pubmed.ncbi.nlm.nih.gov/2010425/)

## Timing and performance metrics

- Hughes et al. Anesthesiology. 1992. (propofol CSHT). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1539843/)
- Egan et al. Anesthesiology. 1993. (remifentanil CSHT). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7902032/)
- Kapila et al. Anesthesiology. 1995. (remifentanil CSHT; PD offset to MV recovery). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7486182/)
- Grundmann et al. Acta Anaesthesiol Scand. 2001. (remifentanil-based anesthesia recovery profile with propofol). [PubMed](https://pubmed.ncbi.nlm.nih.gov/11207468/)
- Kwon et al. Korean J Anesthesiol. 2018. (hypercapnia does not shorten emergence time from propofol anesthesia). [PubMed](https://pubmed.ncbi.nlm.nih.gov/29690757/)
- Magorian et al. Anesthesiology. 1993. (rocuronium onset). [PubMed](https://pubmed.ncbi.nlm.nih.gov/7902034/)
- Varvel et al. J Pharmacokinet Biopharm. 1992. (TCI performance metrics). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1588504/)
- Shafer and Gregg. J Pharmacokinet Biopharm. 1992. (effect-site/plasma-targeted TCI algorithms). [PubMed](https://pubmed.ncbi.nlm.nih.gov/1629794/)
