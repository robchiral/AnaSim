# VitalDB bolus induction with sevoflurane

Generated 2026-10-24T16:04:30.597428+00:00.
Git revision `7185631994e5f0eb6635e412b6c09420722d84db`. Model source SHA256 `f2186c9cd885b5acfe0e744245ccfa6b6b069bd90905ee423a7249990b53246c`.

Model override: `{}`.

## Methods

Propofol runs over 20 s, starting so that model BIS first falls below 60 when recorded BIS does; fentanyl is
given 2 minutes earlier; recorded inspired sevoflurane and remifentanil pumps are replayed, with intubation
stimulation 3 minutes after propofol and surgery from incision.
Each cuff reading from propofol to incision is compared with model MAP over the 40 s before it
posts, stopping at the first reading with MAP below 65 mmHg, SBP below 90, or HR below 45; a likelihood fit with
a patient offset and reading noise gives the mean error (model minus recorded) with 500-sample patient
bootstrap intervals.

Eligible candidates 217; retained 26 (18 with fentanyl, 8 with remifentanil).

| Patients | n | Readings | Mean error, mmHg (95% CI) | Patient SD | Reading SD |
|---|---:|---:|---|---:|---:|
| All patients | 26 | 151 | -6.4 (-10.0 to -2.0) | 7.1 | 15.8 |
| Untreated | 12 | 82 | -10.7 (-13.8 to -7.4) | 0.0 | 16.6 |
| Pressor-treated | 14 | 69 | -1.4 (-7.6 to 4.7) | 9.5 | 13.9 |

By time after propofol, all patients: before 10 min -6.5 (-10.7 to -2.1); from 10 min -5.1 (-13.5 to 1.7) mmHg.

Change in MAP from pre-induction, all readings without stopping, median (IQR):

| Measure | Patients | Recorded | Model |
|---|---:|---|---|
| 1-2 min | 20 | -12% (-17 to 4) | -16% (-18 to -12) |
| 2-5 min | 24 | -12% (-25 to -3) | -20% (-24 to -13) |
| lowest in 10 min | 26 | -29% (-34 to -27) | -31% (-36 to -27) |
| 10 min before incision | 22 | -26% (-28 to -20) | -29% (-32 to -26) |

End-tidal sevoflurane before incision, median: recorded 1.60%, model 1.68% (n=24).

End-tidal sevoflurane surgery, median: recorded 1.95%, model 1.97% (n=24).

ECG HR error (model minus recorded, per-patient medians, untreated): 0-5 min -0.4 (-3.9 to 3.1), n=12; 5-10 min -3.1 (-10.4 to 3.6), n=12; 10 min to incision -0.6 (-7.8 to 5.4), n=11 bpm.

Surgery from 15 min after incision, untreated: MAP 4.6 (-2.6 to 11.9), n=11 mmHg; HR 1.6 (-6.6 to 9.6), n=11 bpm.

Exclusions:

- delivery replay outside recorded interval: 14
- incomplete input history: 53
- insufficient or invalid sevoflurane: 1
- less than five minutes from induction to incision: 1
- model BIS does not fall below 60: 3
- no BIS fall from awake values: 109
- no cuff MAP within 15 minutes before induction: 7
- no valid BIS: 3

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.
