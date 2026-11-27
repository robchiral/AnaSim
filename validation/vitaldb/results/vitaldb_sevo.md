# VitalDB bolus induction with sevoflurane

Generated 2026-11-27.

## Methods

Propofol runs over 20 s, starting 30 s after the time at which model BIS would first fall below
60 when recorded BIS does. Fentanyl is given 2 minutes earlier. Recorded inspired sevoflurane and remifentanil
pumps are replayed, with intubation stimulation 2.5 minutes after propofol and surgery
from incision. Eligibility covers all unique-subject oral single-lumen general anesthetics inside AnaSim's
patient domain (ASA 1-3, normotensive and hypertensive), wiring recorded preop hemoglobin (134/136)
and Cockcroft-Gault CrCl (135/136).

From propofol to incision, each cuff reading is compared with model MAP averaged over the 40 s
before the reading appears. A patient's readings stop at the first one with MAP below 65 mmHg, SBP below 90, or
HR below 45. A likelihood fit with a patient offset and reading noise gives the mean error (model minus recorded),
with 95% intervals from 500 patient bootstrap resamples.

Eligible candidates: 993. Retained: 136 (80 with fentanyl, 51 with remifentanil).

| Patients | n | Readings | Mean error, mmHg (95% CI) | Patient SD | Reading SD |
|---|---:|---:|---|---:|---:|
| All patients | 136 | 753 | -6.1 (-8.2 to -4.1) | 8.7 | 18.0 |
| Legacy core (open supine, age ≤ 70, ASA 1-2, normotensive, BMI ≤ 32) | 29 | 168 | -5.5 (-9.5 to -1.5) | 7.5 | 16.5 |
| Untreated | 57 | 388 | -9.2 (-11.9 to -6.2) | 7.5 | 19.1 |
| Pressor-treated | 79 | 365 | -3.4 (-6.2 to -0.6) | 9.2 | 16.7 |
| Normotensive | 100 | 607 | -8.0 (-10.5 to -5.7) | 8.5 | 18.4 |
| Hypertensive | 36 | 146 | -0.3 (-3.9 to 3.8) | 6.5 | 16.2 |
| ASA 1-2 | 128 | 717 | -6.6 (-8.8 to -4.5) | 8.5 | 18.3 |
| ASA 3 | 8 | 36 | 0.6 (-6.1 to 8.9) | 8.2 | 13.0 |
| BMI under 30 | 127 | 702 | -6.0 (-8.4 to -3.7) | 8.9 | 18.0 |
| BMI 30 and over | 9 | 51 | -7.6 (-14.4 to -1.6) | 5.6 | 18.0 |
| Open, supine | 45 | 261 | -4.3 (-7.6 to -1.3) | 6.9 | 16.2 |
| Laparoscopic or robotic, supine | 43 | 215 | -7.6 (-11.8 to -2.9) | 9.0 | 19.2 |
| Other positions | 48 | 277 | -6.8 (-10.4 to -2.5) | 9.4 | 18.8 |
| Age under 45 | 23 | 129 | -4.6 (-9.8 to 1.2) | 9.2 | 17.4 |
| Age 45-59 | 60 | 342 | -6.8 (-10.1 to -3.7) | 9.1 | 18.4 |
| Age 60-69 | 36 | 183 | -8.2 (-12.6 to -3.9) | 7.9 | 19.3 |
| Age 70 and over | 17 | 99 | -2.3 (-6.6 to 1.5) | 5.1 | 15.3 |

By time after propofol, all patients: before 10 min -6.8 (-9.2 to -4.6); from 10 min -2.5 (-6.1 to 0.8) mmHg.

5-fold stratified fold means: -7.8, -2.9, -4.8, -5.7, -9.3 mmHg (mean -6.1, SD 2.5).

Change in MAP from pre-induction, all readings without stopping, median (IQR):

| Measure | Patients | Recorded | Model |
|---|---:|---|---|
| 1-2 min | 86 | -17% (-25 to -6) | -14% (-18 to -10) |
| 2-5 min | 123 | -15% (-25 to -5) | -18% (-27 to -12) |
| lowest in 10 min | 136 | -29% (-36 to -20) | -31% (-36 to -27) |
| 10 min before incision | 82 | -27% (-32 to -18) | -28% (-32 to -24) |

End-tidal sevoflurane before incision, median: recorded 1.60%, model 1.71% (n=127).

End-tidal sevoflurane surgery, median: recorded 1.90%, model 2.04% (n=124).

ECG HR error (model minus recorded, per-patient medians, untreated): 0-5 min 1.0 (-1.1 to 3.0), n=57; 5-10 min -4.2 (-7.6 to -0.4), n=57; 10 min to incision 3.9 (1.2 to 6.7), n=56 bpm.

## Surgery

Primary cohort (untreated patients with BIS-timed induction, from 15 min after incision to end of surgery; arterial MAP used when recorded):

| Procedure | n | MAP error, mmHg (95% CI) | HR error, bpm (95% CI) |
|---|---:|---|---|
| All untreated | 48 | -1.8 (-6.7 to 2.8) | -0.3 (-3.3 to 2.9) |
| Open, supine | 16 | 3.5 (-3.7 to 9.4) | 4.4 (-1.9 to 10.1) |
| Laparoscopic or robotic, supine | 7 | -0.8 (-9.4 to 10.2) | -5.2 (-12.8 to 0.9) |
| Other positions | 25 | -5.4 (-11.7 to 1.3) | -2.0 (-5.9 to 1.6) |

Extended untreated surgery cohort (including cases without pre-induction BIS fall, replayed from pre-sevoflurane baseline; n=171, +123 added):

| Procedure | n | MAP error, mmHg (95% CI) | HR error, bpm (95% CI) |
|---|---:|---|---|
| All untreated | 171 | -1.6 (-4.2 to 0.8) | 0.2 (-1.8 to 2.0) |
| Open, supine | 54 | 0.1 (-3.7 to 3.8) | 1.1 (-2.4 to 4.7) |
| Laparoscopic or robotic, supine | 51 | -2.5 (-7.1 to 2.2) | -3.4 (-6.3 to 0.2) |
| Other positions | 66 | -2.4 (-5.7 to 1.5) | 2.2 (-0.9 to 5.1) |

Exclusions:

- delivery replay outside recorded interval: 84
- incomplete input history: 203
- insufficient HR before induction: 1
- insufficient or invalid sevoflurane: 3
- less than five minutes from induction to incision: 7
- model BIS does not fall below 60: 10
- nitrous oxide exposure: 6
- no BIS fall from awake values: 493
- no cuff MAP within 15 minutes before induction: 20
- no valid BIS: 25
- sevoflurane before induction: 5

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.
