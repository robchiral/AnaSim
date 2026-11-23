# VitalDB bolus induction with sevoflurane

Generated 2026-11-23.

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
| All patients | 136 | 753 | -6.8 (-8.9 to -4.8) | 8.7 | 18.0 |
| Legacy core (open supine, age ≤ 70, ASA 1-2, normotensive, BMI ≤ 32) | 29 | 168 | -6.2 (-10.1 to -2.1) | 7.5 | 16.5 |
| Untreated | 57 | 388 | -9.9 (-12.6 to -7.0) | 7.3 | 19.1 |
| Pressor-treated | 79 | 365 | -4.1 (-6.8 to -1.2) | 9.2 | 16.7 |
| Normotensive | 100 | 607 | -8.7 (-11.1 to -6.3) | 8.5 | 18.4 |
| Hypertensive | 36 | 146 | -1.0 (-4.7 to 3.1) | 6.8 | 16.2 |
| ASA 1-2 | 128 | 717 | -7.2 (-9.5 to -5.2) | 8.5 | 18.2 |
| ASA 3 | 8 | 36 | -0.5 (-7.5 to 8.1) | 8.6 | 13.1 |
| BMI under 30 | 127 | 702 | -6.7 (-9.1 to -4.4) | 8.9 | 18.0 |
| BMI 30 and over | 9 | 51 | -8.4 (-15.2 to -2.4) | 5.6 | 18.0 |
| Open, supine | 45 | 261 | -5.0 (-8.3 to -2.0) | 6.8 | 16.2 |
| Laparoscopic or robotic, supine | 43 | 215 | -8.3 (-12.5 to -3.5) | 9.0 | 19.2 |
| Other positions | 48 | 277 | -7.4 (-11.1 to -3.1) | 9.4 | 18.8 |
| Age under 45 | 23 | 129 | -5.0 (-10.3 to 0.7) | 9.2 | 17.4 |
| Age 45-59 | 60 | 342 | -7.3 (-10.6 to -4.1) | 9.2 | 18.3 |
| Age 60-69 | 36 | 183 | -8.9 (-13.3 to -4.6) | 7.8 | 19.3 |
| Age 70 and over | 17 | 99 | -3.7 (-7.8 to 0.2) | 5.3 | 15.3 |

By time after propofol, all patients: before 10 min -7.5 (-9.9 to -5.2); from 10 min -3.4 (-7.0 to -0.1) mmHg.

5-fold stratified fold means: -8.4, -3.7, -5.5, -6.4, -10.0 mmHg (mean -6.8, SD 2.5).

Change in MAP from pre-induction, all readings without stopping, median (IQR):

| Measure | Patients | Recorded | Model |
|---|---:|---|---|
| 1-2 min | 86 | -17% (-25 to -6) | -14% (-19 to -11) |
| 2-5 min | 123 | -15% (-25 to -5) | -18% (-28 to -12) |
| lowest in 10 min | 136 | -29% (-36 to -20) | -32% (-37 to -27) |
| 10 min before incision | 82 | -27% (-32 to -18) | -29% (-33 to -25) |

End-tidal sevoflurane before incision, median: recorded 1.60%, model 1.72% (n=127).

End-tidal sevoflurane surgery, median: recorded 1.90%, model 2.05% (n=124).

ECG HR error (model minus recorded, per-patient medians, untreated): 0-5 min 0.1 (-1.9 to 2.2), n=57; 5-10 min -5.4 (-8.8 to -1.7), n=57; 10 min to incision 2.2 (-0.5 to 4.9), n=56 bpm.

Surgery from 15 min after incision, untreated: MAP -3.1 (-7.9 to 1.4), n=48 mmHg; HR -2.7 (-5.6 to 0.4), n=48 bpm.

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
