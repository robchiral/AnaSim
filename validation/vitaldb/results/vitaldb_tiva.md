# VitalDB TIVA hemodynamics

Generated 2026-11-23.

## Methods

Recorded propofol and remifentanil pump delivery drives the full engine across all eligible unique-subject
oral single-lumen general anesthetics inside AnaSim's patient domain (ASA 1-3, normotensive and hypertensive).
Each patient starts from their last cuff MAP within 15 minutes before propofol, preceding 2-minute median HR,
and recorded preoperative hemoglobin (502/511) and Cockcroft-Gault CrCl (499/511).
Controlled ventilation (VT 8 mL/kg PBW up to 600 mL, RR 12, PEEP 5) starts 90 s after propofol, and
intubation stimulation occurs at 4 minutes. Recorded crystalloid, colloid, and blood loss run over surgery
with sustained stimulation; positioning and pneumoperitoneum are not modeled.

From propofol to incision, each cuff reading is compared with model MAP averaged over the 40 s
before the reading appears. A patient's readings stop at the first one with MAP below 65 mmHg, SBP below 90, or
HR below 45 (sensitivity analysis: 70, 100, and 45), so later treatment cannot bias the fit. A likelihood fit
with a patient offset and reading noise gives the mean error (model minus recorded), with 95% intervals from
500 patient bootstrap resamples.

Eligible candidates: 1664. Retained: 511.

## From propofol to incision

| Patients | n | Readings | Mean error, mmHg (95% CI) | Patient SD | Reading SD |
|---|---:|---:|---|---:|---:|
| All patients | 511 | 2788 | -3.3 (-4.3 to -2.5) | 7.5 | 12.4 |
| Legacy core (GenSurg, ASA 1-2, normotensive, BMI ≤ 32) | 330 | 1733 | -3.0 (-4.0 to -2.1) | 7.0 | 12.1 |
| Untreated | 282 | 1772 | -4.9 (-6.0 to -4.0) | 7.3 | 12.0 |
| Pressor-treated | 229 | 1016 | -1.2 (-2.4 to 0.2) | 7.3 | 13.0 |
| Normotensive | 370 | 1949 | -3.4 (-4.3 to -2.5) | 7.1 | 12.1 |
| Hypertensive | 141 | 839 | -3.3 (-5.0 to -1.4) | 8.6 | 12.9 |
| ASA 1-2 | 470 | 2593 | -3.2 (-4.1 to -2.3) | 7.3 | 12.4 |
| ASA 3 | 41 | 195 | -4.5 (-8.4 to -0.7) | 9.3 | 12.0 |
| BMI under 30 | 486 | 2648 | -3.4 (-4.2 to -2.5) | 7.6 | 12.4 |
| BMI 30 and over | 25 | 140 | -2.1 (-5.2 to 0.4) | 4.8 | 12.3 |
| Open, supine | 265 | 1399 | -3.1 (-4.3 to -1.8) | 7.5 | 12.6 |
| Laparoscopic or robotic, supine | 115 | 607 | -1.8 (-3.5 to -0.2) | 6.6 | 11.7 |
| Other positions | 131 | 782 | -5.1 (-6.8 to -3.5) | 8.0 | 12.5 |
| Age under 45 | 99 | 504 | -2.5 (-4.1 to -0.9) | 6.4 | 11.8 |
| Age 45-59 | 173 | 962 | -3.2 (-4.4 to -1.8) | 6.5 | 12.0 |
| Age 60-69 | 154 | 884 | -3.4 (-4.9 to -1.9) | 7.9 | 12.2 |
| Age 70 and over | 85 | 438 | -4.5 (-7.4 to -2.0) | 9.6 | 14.0 |

Sensitivity stopping rule, all patients: -3.8 (-4.6 to -2.9) mmHg.

By time after propofol, all patients: before 10 min -3.8 (-4.7 to -2.8); from 10 min -0.7 (-1.9 to 0.6) mmHg.

Compared with model MAP when each reading appears, all patients: -4.6 (-5.4 to -3.7) mmHg.

5-fold stratified fold means: -2.8, -3.5, -3.3, -4.0, -2.9 mmHg (mean -3.3, SD 0.5).

Change in MAP from pre-induction, all readings without stopping, median (IQR):

| Measure | Patients | Recorded | Model |
|---|---:|---|---|
| 1-2 min | 331 | -18% (-24 to -9) | -18% (-21 to -16) |
| 2-5 min | 487 | -22% (-30 to -13) | -26% (-29 to -22) |
| lowest in 10 min | 510 | -33% (-40 to -25) | -30% (-33 to -26) |
| 10 min before incision | 441 | -29% (-37 to -21) | -29% (-33 to -25) |

Change in MAP from pre-induction, all readings without stopping, patients aged 70 and over, median (IQR):

| Measure | Patients | Recorded | Model |
|---|---:|---|---|
| 1-2 min | 50 | -17% (-24 to -6) | -21% (-22 to -19) |
| 2-5 min | 81 | -26% (-32 to -16) | -29% (-31 to -25) |
| lowest in 10 min | 85 | -37% (-45 to -26) | -33% (-36 to -31) |
| 10 min before incision | 69 | -32% (-39 to -24) | -33% (-36 to -30) |

ECG HR error 3-10 minutes after propofol, untreated: per-patient median 0.1 (-1.1 to 1.3), n=281; highest 30-s median 0.5 (-1.4 to 2.3), n=281 bpm.

## Surgery

Untreated patients, from 15 minutes after incision to the end of surgery: MAP error -3.5 (-5.2 to -1.7), n=182 mmHg; HR error -0.1 (-1.7 to 1.9), n=182 bpm. Arterial MAP is used when recorded.

MAP rise from the last 10 minutes before incision to surgery: recorded 11.0 (9.4 to 12.8), n=182, model 9.5 (9.0 to 9.9), n=182 mmHg.

Exclusions:

- delivery history starts too late for induction: 6
- incomplete input history: 378
- infusion rate and volume disagree by more than five percent or 0.5 mL: 257
- insufficient HR before induction: 3
- invalid or absent remifentanil delivery: 1
- invalid or insufficient propofol delivery: 1
- less than five minutes from induction to incision: 1
- less than thirty seconds of zero delivery before induction: 63
- no cuff MAP within 15 minutes before induction: 106
- no cuff reading between induction and incision: 9
- propofol delivery began before recording: 120
- recorded volatile or nitrous oxide exposure: 208

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.
