# VitalDB bolus induction with sevoflurane

Generated 2026-11-25.

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
| All patients | 136 | 753 | -7.0 (-9.1 to -4.9) | 8.7 | 18.0 |
| Legacy core (open supine, age ≤ 70, ASA 1-2, normotensive, BMI ≤ 32) | 29 | 168 | -6.3 (-10.3 to -2.3) | 7.5 | 16.5 |
| Untreated | 57 | 388 | -10.1 (-12.8 to -7.2) | 7.4 | 19.1 |
| Pressor-treated | 79 | 365 | -4.2 (-6.9 to -1.3) | 9.3 | 16.7 |
| Normotensive | 100 | 607 | -8.9 (-11.3 to -6.5) | 8.5 | 18.4 |
| Hypertensive | 36 | 146 | -1.2 (-4.9 to 2.9) | 6.7 | 16.2 |
| ASA 1-2 | 128 | 717 | -7.4 (-9.7 to -5.4) | 8.5 | 18.2 |
| ASA 3 | 8 | 36 | -0.6 (-7.6 to 8.0) | 8.7 | 13.1 |
| BMI under 30 | 127 | 702 | -6.8 (-9.3 to -4.6) | 8.9 | 18.0 |
| BMI 30 and over | 9 | 51 | -8.5 (-15.5 to -2.5) | 5.8 | 18.1 |
| Open, supine | 45 | 261 | -5.2 (-8.5 to -2.1) | 6.9 | 16.2 |
| Laparoscopic or robotic, supine | 43 | 215 | -8.4 (-12.6 to -3.7) | 9.1 | 19.2 |
| Other positions | 48 | 277 | -7.6 (-11.3 to -3.3) | 9.4 | 18.8 |
| Age under 45 | 23 | 129 | -5.2 (-10.5 to 0.6) | 9.3 | 17.4 |
| Age 45-59 | 60 | 342 | -7.5 (-10.8 to -4.3) | 9.2 | 18.3 |
| Age 60-69 | 36 | 183 | -9.1 (-13.4 to -4.7) | 7.8 | 19.2 |
| Age 70 and over | 17 | 99 | -3.9 (-8.0 to -0.1) | 5.0 | 15.4 |

By time after propofol, all patients: before 10 min -7.6 (-10.0 to -5.4); from 10 min -3.6 (-7.3 to -0.3) mmHg.

5-fold stratified fold means: -8.6, -3.9, -5.7, -6.5, -10.1 mmHg (mean -7.0, SD 2.5).

Change in MAP from pre-induction, all readings without stopping, median (IQR):

| Measure | Patients | Recorded | Model |
|---|---:|---|---|
| 1-2 min | 86 | -17% (-25 to -6) | -14% (-19 to -11) |
| 2-5 min | 123 | -15% (-25 to -5) | -18% (-28 to -12) |
| lowest in 10 min | 136 | -29% (-36 to -20) | -32% (-37 to -27) |
| 10 min before incision | 82 | -27% (-32 to -18) | -29% (-33 to -25) |

End-tidal sevoflurane before incision, median: recorded 1.60%, model 1.72% (n=127).

End-tidal sevoflurane surgery, median: recorded 1.90%, model 2.05% (n=124).

ECG HR error (model minus recorded, per-patient medians, untreated): 0-5 min 0.1 (-1.9 to 2.1), n=57; 5-10 min -5.4 (-8.8 to -1.7), n=57; 10 min to incision 2.3 (-0.4 to 5.0), n=56 bpm.

Surgery from 15 min after incision, untreated: MAP -4.0 (-8.9 to 0.6), n=48 mmHg; HR -2.7 (-5.6 to 0.4), n=48 bpm.

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
