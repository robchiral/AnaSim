# VitalDB TIVA hemodynamics

Generated 2026-10-24T16:10:50.856915+00:00.
Git revision `7185631994e5f0eb6635e412b6c09420722d84db`. Model source SHA256 `f2186c9cd885b5acfe0e744245ccfa6b6b069bd90905ee423a7249990b53246c`.

## Methods

Recorded propofol and remifentanil pump delivery drives the full engine at production parameters, from each
patient's last cuff MAP within 15 minutes before propofol and median HR over the preceding 2 minutes.
Controlled ventilation (VT 8 mL/kg PBW, at most 600 mL; RR 12; PEEP 5) starts 90 s after propofol and
intubation stimulation at 4 minutes. Recorded crystalloid runs at a constant rate, colloid is given every 5
minutes, and blood loss bleeds at a constant rate during surgery, with sustained surgical stimulation.
Positioning and pneumoperitoneum are not replayed.

From propofol to incision, each cuff reading is compared with model MAP over the 40 s before it
posts. A patient's readings stop at the first with MAP below 65 mmHg, SBP below 90, or HR below 45
(sensitivity: 70, 100, 45), so later treatment cannot bias the fit; a likelihood fit with a patient offset and
reading noise gives the mean error (model minus recorded) with 500-sample patient bootstrap intervals.

Eligible candidates 959; retained 301.

## From propofol to incision

| Patients | n | Readings | Mean error, mmHg (95% CI) | Patient SD | Reading SD |
|---|---:|---:|---|---:|---:|
| All patients | 301 | 1609 | -1.7 (-2.8 to -0.6) | 7.5 | 12.1 |
| Untreated | 187 | 1127 | -2.9 (-4.2 to -1.6) | 7.5 | 11.7 |
| Pressor-treated | 114 | 482 | 0.4 (-1.6 to 2.1) | 6.9 | 13.0 |
| Open, supine | 169 | 872 | -1.1 (-2.6 to 0.5) | 7.2 | 12.0 |
| Laparoscopic or robotic, supine | 65 | 317 | 0.7 (-1.4 to 2.8) | 5.6 | 9.8 |
| Other positions | 67 | 420 | -5.6 (-8.1 to -3.0) | 8.2 | 13.7 |
| Age under 45 | 87 | 441 | -0.5 (-2.4 to 1.5) | 7.6 | 11.1 |
| Age 45-59 | 127 | 685 | -1.2 (-2.8 to 0.4) | 7.1 | 12.1 |
| Age 60 and over | 87 | 483 | -3.5 (-5.7 to -1.5) | 7.8 | 12.9 |

Sensitivity stopping rule, all patients: -1.6 (-2.8 to -0.5) mmHg.

By time after propofol, all patients: before 10 min -1.6 (-2.8 to -0.5); from 10 min -2.3 (-4.2 to -0.5) mmHg.

Compared with model MAP when each reading posts, all patients: -3.4 (-4.5 to -2.4) mmHg.

Change in MAP from pre-induction, all readings without stopping, median (IQR):

| Measure | Patients | Recorded | Model |
|---|---:|---|---|
| 1-2 min | 209 | -17% (-24 to -9) | -15% (-19 to -11) |
| 2-5 min | 291 | -21% (-29 to -12) | -25% (-29 to -21) |
| lowest in 10 min | 300 | -31% (-38 to -25) | -30% (-34 to -25) |
| 10 min before incision | 272 | -27% (-34 to -19) | -28% (-32 to -23) |

ECG HR error 3-10 minutes after propofol, untreated: per-patient median 1.5 (-0.0 to 2.8), n=187; highest 30-s median 3.8 (1.6 to 6.0), n=187 bpm.

## Surgery

Untreated patients, from 15 minutes after incision to the end of surgery: MAP error -1.3 (-3.6 to 0.8), n=114 mmHg; HR error 2.7 (0.7 to 4.8), n=114 bpm. Arterial MAP is used when recorded.

MAP rise from the last 10 minutes before incision to surgery: recorded 11.8 (9.7 to 13.7), n=114, model 12.3 (11.3 to 13.2), n=114 mmHg.

Exclusions:

- delivery history starts too late for induction: 40
- incomplete input history: 215
- infusion rate and volume disagree by more than five percent or 0.5 mL: 148
- insufficient HR before induction: 1
- invalid or absent remifentanil delivery: 1
- less than five minutes from induction to incision: 1
- no cuff MAP within 15 minutes before induction: 60
- no cuff reading between induction and incision: 5
- propofol delivery began before recording: 72
- recorded volatile or nitrous oxide exposure: 115

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.
