# VitalDB anesthetic comparisons

Generated 2026-11-23.

## Methods

These comparisons test the anesthetic models in isolation across all eligible unique-subject recordings
inside AnaSim's supported domain (general anesthesia, oral single-lumen tube, age 18 to 85 years, weight
50 to 200 kg, height 150 to 200 cm, BMI 18 to 60 kg/m²), passing preoperative hemoglobin and Cockcroft-Gault
creatinine clearance when recorded. The models use AnaSim's standard parameters, which were not tuned to
these recordings. The replay fits no individual calibrations or time shifts, and predicted BIS does not
start from the observed BIS.

Each cohort includes at most one case per patient, and seed 20261013 fixes candidate ordering.
The replay step is 1 s, and endpoints are sampled every 5 s. Each reported phase
needs at least one usable minute, and each case needs ten usable minutes in total. Errors are predicted minus
measured, with 95% intervals from 2,000 patient bootstrap resamples.

## Propofol and BIS

Propofol delivery comes from `PPF20_RATE` (20 mg/mL), integrated in mL and checked against cumulative
`PPF20_VOL` (with syringe-change counter resets unwrapped) within 5% or 0.5 mL. Delivery gaps over 5 s,
supplemental propofol/midazolam/fentanyl boluses, dexmedetomidine pumps, volatile or N₂O exposure, and
BIS samples with SQI below 90 are excluded. The comparator is a constant BIS of 40.

Cases with the required tracks: 1986. After metadata rules: 1969.
Screened: 1969. Retained: 218 (all eligible).

| Phase | Patients | Usable minutes | Bias (95% CI) | MAE (95% CI) | RMSE (95% CI) | Comparator MAE (95% CI) |
|---|---:|---:|---|---|---|---|
| all | 218 | 10914.6 | -6.55 (-7.73 to -5.42) | 11.09 (10.46 to 11.78) | 13.11 (12.44 to 13.75) | 9.68 (9.19 to 10.18) |
| induction | 201 | 858.3 | -9.75 (-11.37 to -8.18) | 15.99 (15.03 to 16.94) | 17.86 (16.90 to 18.80) | 16.41 (15.42 to 17.44) |
| maintenance | 213 | 6753.2 | -5.71 (-6.90 to -4.53) | 9.72 (9.01 to 10.44) | 11.03 (10.37 to 11.74) | 7.10 (6.63 to 7.64) |
| emergence | 124 | 472.8 | -5.75 (-7.72 to -3.58) | 11.71 (10.51 to 12.98) | 12.90 (11.69 to 14.14) | 21.29 (19.68 to 22.84) |

Paired all-phase MAE difference, AnaSim minus comparator: 1.41 (0.79 to 2.01).
AnaSim has lower MAE in 79 of 218 patients.
Usable endpoint coverage: median 47.7% (IQR 38.1% to 56.6%) of the replay sampling interval.

| Subgroup (all phases) | Patients | Bias (95% CI) | MAE (95% CI) | RMSE (95% CI) |
|---|---:|---|---|---|
| Legacy core (open supine, age ≤ 70, BMI ≤ 32) | 96 | -4.77 (-6.52 to -3.10) | 10.48 (9.64 to 11.43) | 12.48 (11.66 to 13.35) |
| Age under 45 | 36 | 0.76 (-1.69 to 3.30) | 9.52 (8.65 to 10.42) | 11.45 (10.53 to 12.34) |
| Age 45-59 | 69 | -5.53 (-7.12 to -3.95) | 9.65 (8.81 to 10.52) | 11.80 (10.90 to 12.74) |
| Age 60-69 | 71 | -10.05 (-12.10 to -7.98) | 12.95 (11.53 to 14.37) | 14.85 (13.62 to 16.25) |
| Age 70 and over | 42 | -8.58 (-10.91 to -6.29) | 11.65 (10.19 to 13.14) | 13.77 (12.36 to 15.20) |
| BMI under 30 | 213 | -6.54 (-7.71 to -5.37) | 11.10 (10.41 to 11.77) | 13.13 (12.50 to 13.80) |
| BMI 30 and over | 5 | -6.91 (-14.99 to 0.10) | 10.88 (7.13 to 17.32) | 12.60 (8.90 to 18.75) |

Exclusions among screened cases:

- delivery history starts too late for induction: 7
- gas monitoring began after propofol induction: 117
- incomplete input history: 1025
- infusion rate and volume disagree by more than five percent or 0.5 mL: 95
- invalid or absent remifentanil delivery: 1
- invalid or insufficient propofol delivery: 1
- less than ten minutes of usable endpoint measurements: 1
- less than thirty seconds of zero delivery before induction: 82
- propofol delivery began before recording: 143
- recorded volatile or nitrous oxide exposure: 279

## Sevoflurane uptake

Recorded inspired sevoflurane, tidal volume, and respiratory rate drive `VolatilePK` with PBW-based
dead space (2.2 mL/kg PBW), assumed cardiac output 5 L/min (with 3 and 7 L/min sensitivity runs), and
fixed patient FRC. The comparator sets end-tidal concentration equal to recorded inspired concentration.

Cases with the required tracks: 2290. After metadata rules: 2290.
Screened: 2290. Retained: 305 (all eligible).

| Phase | Patients | Usable minutes | Bias (95% CI) | MAE (95% CI) | RMSE (95% CI) | Comparator MAE (95% CI) |
|---|---:|---:|---|---|---|---|
| all | 305 | 46436.0 | 0.10 (0.09 to 0.10) | 0.13 (0.13 to 0.14) | 0.18 (0.18 to 0.19) | 0.36 (0.35 to 0.37) |
| washin | 305 | 2776.4 | -0.02 (-0.04 to -0.01) | 0.25 (0.24 to 0.27) | 0.42 (0.39 to 0.45) | 0.66 (0.63 to 0.68) |
| maintenance | 305 | 41617.8 | 0.11 (0.10 to 0.12) | 0.12 (0.12 to 0.13) | 0.14 (0.13 to 0.14) | 0.33 (0.32 to 0.34) |
| washout | 291 | 2036.6 | 0.04 (0.03 to 0.06) | 0.12 (0.11 to 0.13) | 0.15 (0.14 to 0.16) | 0.39 (0.38 to 0.41) |

Paired all-phase MAE difference, AnaSim minus comparator: -0.23 (-0.23 to -0.22).
AnaSim has lower MAE in 304 of 305 patients.
Usable endpoint coverage: median 97.8% (IQR 96.2% to 98.7%) of the replay sampling interval.

| Subgroup (all phases) | Patients | Bias (95% CI) | MAE (95% CI) | RMSE (95% CI) |
|---|---:|---|---|---|
| Legacy core (open supine, age ≤ 70, BMI ≤ 32) | 95 | 0.10 (0.08 to 0.11) | 0.13 (0.12 to 0.14) | 0.18 (0.17 to 0.20) |
| Age under 45 | 64 | 0.09 (0.07 to 0.10) | 0.13 (0.12 to 0.14) | 0.19 (0.17 to 0.21) |
| Age 45-59 | 122 | 0.10 (0.09 to 0.11) | 0.13 (0.12 to 0.14) | 0.18 (0.17 to 0.19) |
| Age 60-69 | 72 | 0.10 (0.09 to 0.11) | 0.13 (0.12 to 0.14) | 0.18 (0.17 to 0.20) |
| Age 70 and over | 47 | 0.10 (0.09 to 0.12) | 0.14 (0.13 to 0.15) | 0.18 (0.17 to 0.20) |
| BMI under 30 | 293 | 0.09 (0.09 to 0.10) | 0.13 (0.12 to 0.14) | 0.18 (0.17 to 0.19) |
| BMI 30 and over | 12 | 0.15 (0.13 to 0.17) | 0.18 (0.16 to 0.20) | 0.23 (0.20 to 0.24) |

Cardiac output sensitivity (all-phase mean error, 95% CI):

| Assumed CO (L/min) | Bias | MAE | RMSE |
|---:|---|---|---|
| 3.0 | 0.18 (0.17 to 0.18) | 0.21 (0.20 to 0.21) | 0.24 (0.23 to 0.25) |
| 5.0 | 0.10 (0.09 to 0.10) | 0.13 (0.13 to 0.14) | 0.18 (0.18 to 0.19) |
| 7.0 | 0.06 (0.05 to 0.06) | 0.11 (0.11 to 0.12) | 0.17 (0.17 to 0.18) |

Exclusions among screened cases:

- expired sevoflurane already present at recording start: 22
- incomplete input history: 316
- insufficient sevoflurane exposure in common interval: 2
- recorded nitrous oxide exposure: 3
- sevoflurane exposure already present or insufficient exposure: 1311
- short zero-agent baseline or exposure interval: 262
- ventilation inputs outside comparison range: 58
- ventilation recording began after sevoflurane exposure: 11

## Scope and artifacts

These comparisons evaluate individual anesthetic models against VitalDB development recordings.
Per-patient errors, track identifiers, and exclusions are in `vitaldb_anesthetics.json` and
`vitaldb_anesthetics_cases.csv`.

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.
