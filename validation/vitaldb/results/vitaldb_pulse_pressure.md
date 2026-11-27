# VitalDB pulse pressure, SBP/DBP, and CVP

Generated 2026-11-27.

## Methods

Compares awake pre-induction cuff SBP, DBP, and pulse pressure (NIBP_SBP - NIBP_DBP) across
retained TIVA and sevoflurane cases against AnaSim's HemodynamicModel resting Windkessel/arterial
compliance curve (Franklin 1997; Langewouters 1984), and evaluates intraoperative invasive arterial
pulse pressure (at matched MAP) and CVP during untreated surgery.

Candidate retained cases: 642. With valid pre-induction SBP/DBP: 512.

## Awake pre-induction pulse pressure (model minus recorded)

| Subgroup | n | Recorded PP (mmHg) | Model PP (mmHg) | PP error, mmHg (95% CI) | SBP error (mmHg) | DBP error (mmHg) |
|---|---:|---:|---:|---|---|---|
| All patients | 512 | 65.0 | 59.4 | -7.8 (-9.9 to -5.9) | -3.9 (-5.4 to -2.5) | 3.8 (3.2 to 4.6) |
| Age under 45 | 99 | 61.0 | 46.6 | -15.9 (-19.5 to -12.6) | -11.2 (-14.0 to -8.2) | 4.7 (3.5 to 6.0) |
| Age 45-59 | 173 | 62.0 | 55.5 | -8.7 (-12.3 to -5.4) | -4.8 (-7.3 to -2.1) | 3.9 (2.9 to 5.0) |
| Age 60-69 | 155 | 66.0 | 63.6 | -6.0 (-9.4 to -2.5) | -2.2 (-5.1 to 0.6) | 3.8 (2.4 to 5.2) |
| Age 70 and over | 85 | 75.0 | 75.1 | 0.1 (-4.3 to 4.5) | 3.0 (-0.8 to 6.2) | 2.9 (1.0 to 4.7) |
| Normotensive | 371 | 63.0 | 56.0 | -8.3 (-10.8 to -6.0) | -4.5 (-6.1 to -2.9) | 3.8 (3.0 to 4.6) |
| Hypertensive | 141 | 70.0 | 64.8 | -6.4 (-10.1 to -2.3) | -2.4 (-5.8 to 0.8) | 3.9 (2.5 to 5.2) |

## Intraoperative invasive arterial pulse pressure (untreated surgery, MAP-matched)

| Subgroup | n | Recorded PP (mmHg) | Model PP (mmHg) | PP error, mmHg (95% CI) |
|---|---:|---:|---:|---|
| All untreated arterial | 45 | 56.0 | 44.1 | -14.1 (-18.8 to -9.7) |
| Age under 60 | 24 | 52.0 | 39.0 | -14.9 (-20.4 to -9.8) |
| Age 60 and over | 21 | 61.0 | 49.3 | -13.3 (-21.0 to -5.3) |
| MAP under 75 mmHg | 5 | 47.0 | 36.0 | -15.1 (-32.4 to -2.6) |
| MAP 75 mmHg and over | 40 | 58.0 | 44.3 | -14.0 (-19.0 to -9.3) |

## Intraoperative CVP (untreated surgery)

n=7: median recorded 5.0 vs model RAP 3.7 mmHg (error -1.1 (-2.2 to 0.1) mmHg).

Exclusions:

- missing pre-induction NIBP SBP/DBP: 130

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.

