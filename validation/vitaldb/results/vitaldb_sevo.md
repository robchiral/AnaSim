# VitalDB bolus induction with sevoflurane

Generated 2026-10-30.

## Methods

Propofol runs over 20 s. It starts 30 s after the time at which model BIS would first fall below 60
when recorded BIS does. Fentanyl is given 2 minutes earlier. Recorded inspired sevoflurane and remifentanil pumps
are replayed, with intubation stimulation 2.5 minutes after propofol and surgery from incision.

From propofol to incision, each cuff reading is compared with model MAP averaged over the 40 s
before the reading appears. A patient's readings stop at the first one with MAP below 65 mmHg, SBP below 90, or
HR below 45. A likelihood fit with a patient offset and reading noise gives the mean error (model minus recorded),
with 95% intervals from 500 patient bootstrap resamples.

Eligible candidates: 217. Retained: 29 (20 with fentanyl, 9 with remifentanil).

| Patients | n | Readings | Mean error, mmHg (95% CI) | Patient SD | Reading SD |
|---|---:|---:|---|---:|---:|
| All patients | 29 | 168 | -5.4 (-9.6 to -1.1) | 8.1 | 17.0 |
| Untreated | 13 | 92 | -11.0 (-15.7 to -6.3) | 3.7 | 18.4 |
| Pressor-treated | 16 | 76 | 0.2 (-5.6 to 6.4) | 9.9 | 14.4 |

By time after propofol, all patients: before 10 min -5.4 (-9.8 to -1.2); from 10 min -4.9 (-11.7 to 1.7) mmHg.

Change in MAP from pre-induction, all readings without stopping, median (IQR):

| Measure | Patients | Recorded | Model |
|---|---:|---|---|
| 1-2 min | 21 | -16% (-25 to -8) | -14% (-16 to -12) |
| 2-5 min | 27 | -13% (-24 to 1) | -19% (-26 to -12) |
| lowest in 10 min | 29 | -29% (-34 to -26) | -32% (-37 to -27) |
| 10 min before incision | 24 | -27% (-28 to -19) | -29% (-33 to -27) |

End-tidal sevoflurane before incision, median: recorded 1.60%, model 1.66% (n=27).

End-tidal sevoflurane surgery, median: recorded 2.00%, model 1.97% (n=27).

ECG HR error (model minus recorded, per-patient medians, untreated): 0-5 min -0.7 (-4.7 to 3.5), n=13; 5-10 min -2.6 (-8.9 to 3.5), n=13; 10 min to incision 0.2 (-6.2 to 6.3), n=12 bpm.

Surgery from 15 min after incision, untreated: MAP 6.0 (-1.6 to 13.7), n=12 mmHg; HR 1.9 (-5.6 to 9.6), n=12 bpm.

Exclusions:

- delivery replay outside recorded interval: 13
- incomplete input history: 51
- insufficient or invalid sevoflurane: 1
- less than five minutes from induction to incision: 1
- model BIS does not fall below 60: 3
- no BIS fall from awake values: 109
- no cuff MAP within 15 minutes before induction: 7
- no valid BIS: 3

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.
