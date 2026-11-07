# VitalDB TIVA hemodynamics

Generated 2026-11-07.

## Methods

Recorded propofol and remifentanil pump delivery is the input to the full engine, at its standard parameters.
Each patient starts from their last cuff MAP within 15 minutes before propofol and their median HR over the
preceding 2 minutes. Controlled ventilation (VT 8 mL/kg PBW up to 600 mL, RR 12, PEEP 5) starts 90 s after
propofol, and intubation stimulation occurs at 4 minutes. Recorded crystalloid runs at a constant rate, colloid
is given every 5 minutes, and recorded blood loss occurs at a constant rate during surgery, with sustained
surgical stimulation. The replay does not include positioning or pneumoperitoneum.

From propofol to incision, each cuff reading is compared with model MAP averaged over the 40 s
before the reading appears. A patient's readings stop at the first one with MAP below 65 mmHg, SBP below 90, or
HR below 45 (sensitivity analysis: 70, 100, and 45), so later treatment cannot bias the fit. A likelihood fit
with a patient offset and reading noise gives the mean error (model minus recorded), with 95% intervals from
500 patient bootstrap resamples.

Eligible candidates: 1037. Retained: 333.

## From propofol to incision

| Patients | n | Readings | Mean error, mmHg (95% CI) | Patient SD | Reading SD |
|---|---:|---:|---|---:|---:|
| All patients | 333 | 1760 | -1.8 (-2.8 to -0.7) | 7.7 | 12.4 |
| Untreated | 196 | 1177 | -3.1 (-4.7 to -1.8) | 7.6 | 12.0 |
| Pressor-treated | 137 | 583 | 0.3 (-1.3 to 2.1) | 7.2 | 13.4 |
| Open, supine | 186 | 952 | -1.4 (-2.9 to -0.0) | 7.6 | 12.7 |
| Laparoscopic or robotic, supine | 67 | 327 | 0.7 (-1.2 to 2.6) | 5.4 | 9.9 |
| Other positions | 80 | 481 | -4.6 (-6.8 to -2.2) | 8.3 | 13.4 |
| Age under 45 | 87 | 441 | -0.5 (-2.3 to 1.2) | 7.6 | 11.1 |
| Age 45-59 | 127 | 685 | -1.2 (-2.7 to 0.3) | 7.1 | 12.1 |
| Age 60-69 | 86 | 476 | -3.4 (-5.6 to -1.3) | 7.9 | 12.9 |
| Age 70 and over | 33 | 158 | -3.1 (-7.5 to 0.9) | 8.3 | 15.6 |

Sensitivity stopping rule, all patients: -1.5 (-2.6 to -0.4) mmHg.

By time after propofol, all patients: before 10 min -1.7 (-2.8 to -0.6); from 10 min -2.2 (-3.8 to -0.4) mmHg.

Compared with model MAP when each reading appears, all patients: -3.5 (-4.6 to -2.5) mmHg.

Change in MAP from pre-induction, all readings without stopping, median (IQR):

| Measure | Patients | Recorded | Model |
|---|---:|---|---|
| 1-2 min | 228 | -17% (-25 to -9) | -15% (-19 to -11) |
| 2-5 min | 321 | -22% (-30 to -12) | -25% (-30 to -21) |
| lowest in 10 min | 332 | -32% (-39 to -25) | -30% (-34 to -26) |
| 10 min before incision | 298 | -28% (-35 to -19) | -28% (-33 to -23) |

Change in MAP from pre-induction, all readings without stopping, patients aged 70 and over, median (IQR):

| Measure | Patients | Recorded | Model |
|---|---:|---|---|
| 1-2 min | 20 | -19% (-26 to -7) | -17% (-20 to -12) |
| 2-5 min | 31 | -27% (-34 to -16) | -31% (-32 to -26) |
| lowest in 10 min | 33 | -39% (-46 to -30) | -35% (-39 to -33) |
| 10 min before incision | 27 | -33% (-42 to -27) | -35% (-39 to -32) |

ECG HR error 3-10 minutes after propofol, untreated: per-patient median 1.3 (-0.1 to 2.7), n=196; highest 30-s median 3.3 (1.1 to 5.8), n=196 bpm.

## Surgery

Untreated patients, from 15 minutes after incision to the end of surgery: MAP error -1.6 (-3.8 to 0.8), n=123 mmHg; HR error 2.5 (0.5 to 4.5), n=123 bpm. Arterial MAP is used when recorded.

MAP rise from the last 10 minutes before incision to surgery: recorded 11.7 (9.8 to 13.6), n=123, model 11.9 (11.0 to 12.7), n=123 mmHg.

Exclusions:

- delivery history starts too late for induction: 43
- incomplete input history: 233
- infusion rate and volume disagree by more than five percent or 0.5 mL: 157
- insufficient HR before induction: 1
- invalid or absent remifentanil delivery: 1
- less than five minutes from induction to incision: 1
- no cuff MAP within 15 minutes before induction: 61
- no cuff reading between induction and incision: 5
- propofol delivery began before recording: 75
- recorded volatile or nitrous oxide exposure: 127

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.
