# VitalDB anesthetic comparisons

Generated 2026-10-30.

## Methods

These comparisons test the anesthetic models in isolation, with normal organ function and unchanged blood volume.
The models use AnaSim's standard parameters, which were not tuned to these recordings. The replay fits no
individual calibrations or time shifts, and predicted BIS does not start from the observed BIS.

Patients meet the same rules as the ventilation cohort: general anesthesia, oral single-lumen tube, open surgery,
supine position, age 18 to 70 years, weight 50 to 100 kg, height 150 to 200 cm, and BMI 18 to 32 kg/m².
Each cohort includes at most one case per patient, and seed 20261013 fixes the order of candidates.
Eligibility uses only recorded inputs and signal quality.

The replay step is 1 s, and endpoints are sampled every 5 s. Each reported phase
needs at least one usable minute, and each case needs ten usable minutes in total. The number of samples in a
phase depends on which measurements are available. The all-phase score includes usable intervals between the
named phases.

Errors are predicted minus measured. Tables show the mean of the per-patient metrics, with 95% intervals from
2,000 patient bootstrap resamples, so long cases have the same weight as short cases. The comparator is scored
on the same samples.

## Propofol and BIS

Propofol delivery comes from the recorded pump rate (PPF20_RATE), which is held between recorded packets and
integrated in mL. A concentration of 20 mg/mL converts it to mg/s for the model. Each case must meet these rules:

- The cumulative pump volume must start at zero (at most 0.05 mL), and both rate and volume must stay at zero for
  at least 30 seconds before induction.
- Rates must be between 0 and 1,500 mL/h, at least 5 mL must be delivered, and the replay must last at least 15
  minutes after induction.
- The integrated rate must agree with the recorded volume within 5% or 0.5 mL, whichever is larger.
- Counter reversals over 0.25 mL and delivery gaps over 5 s are excluded. Smaller counter artifacts do not affect
  the replay, which uses the rate.
- Recorded supplemental boluses of propofol, midazolam, and fentanyl must total zero, and dexmedetomidine pump
  tracks are excluded.
- Remifentanil delivery must be recorded and nonzero, and the propofol model uses its opioid-regimen covariate.
- MAC must stay at or below 0.05, and inspired N₂O at or below 1%. Gas recording must start before propofol
  delivery and have no gaps over 30 s during the replay.
- BIS samples need a signal quality index (SQI) of at least 90, with no more than 2 s since the latest BIS
  and SQI packets.

The model includes the Eleveld effect site, the published BIS processing delay, and the display smoothing, but
no display noise. Remifentanil analgesia and stimulus responses are not evaluated, and the replay does not
include hemorrhage or surgical arousal for individual patients.

Induction ends at incision or ten minutes after first delivery, whichever is earlier, and maintenance begins
once both have passed. Emergence begins once surgery has ended and the last recorded nonzero infusion has
stopped. These phases come from the recording and do not mark observed loss or return of consciousness. The
comparator is a constant BIS of 40. Concentrations estimated by the pump (CP and CE) are never treated as
measured values.

Cases with the required tracks: 826. After the metadata rules: 817.
Screened: 817. Retained: 84 of 100 requested patients.

| Phase | Patients | Usable minutes | Bias (95% CI) | MAE (95% CI) | RMSE (95% CI) | Comparator MAE (95% CI) |
|---|---:|---:|---|---|---|---|
| all | 84 | 3937.2 | -5.20 (-7.09 to -3.17) | 11.01 (10.11 to 12.01) | 13.07 (12.17 to 14.08) | 9.68 (8.90 to 10.50) |
| induction | 77 | 312.6 | -8.64 (-11.24 to -6.25) | 14.39 (12.98 to 15.89) | 16.16 (14.64 to 17.64) | 15.92 (14.14 to 17.74) |
| maintenance | 82 | 2546.2 | -4.40 (-6.44 to -2.37) | 9.70 (8.56 to 10.80) | 10.96 (9.90 to 12.01) | 7.09 (6.35 to 7.85) |
| emergence | 49 | 177.8 | -3.85 (-7.39 to -0.11) | 12.10 (10.17 to 14.03) | 13.36 (11.59 to 15.32) | 19.60 (16.99 to 22.06) |

Paired all-phase MAE difference, AnaSim minus comparator: 1.33 (0.45 to 2.29).
AnaSim has lower MAE in 32 of 84 patients.
Usable endpoint coverage: median 45.4% (IQR 34.9% to 56.3%) of the replay sampling interval.

Exclusions among screened cases:

- gas monitoring began after propofol induction: 6
- incomplete input history: 174
- invalid or insufficient propofol delivery: 192
- less than thirty seconds of zero delivery before induction: 13
- propofol delivery began before recording: 56
- propofol volume reset or inconsistent counter: 267
- recorded volatile or nitrous oxide exposure: 25

The requested sample was not reached, so the table reports all eligible patients found.

## Sevoflurane uptake

Recorded inspired sevoflurane is the input to the uptake model (`VolatilePK`), which starts with no agent in
any compartment. Recorded tidal volume and the respiratory rate from the CO₂ trace give alveolar ventilation,
using AnaSim's dead space of 2.2 mL/kg PBW. Each case must meet these rules:

- Inspired and expired concentrations must start at or below 0.1%, with 30 s before exposure reaches 0.5%.
- Inspired concentration must reach at least 1%, and the first and last readings at or above 0.5% must be at
  least 15 minutes apart.
- Cases are excluded for input gaps over 30 s, inspired concentrations outside 0 to 8%, VT outside 0 to 1,500 mL,
  RR outside 0 to 50/min, or N₂O exposure over 1%.
- Endpoint samples need VT of at least 200 mL, RR of at least 5/min, EtCO₂ of 15 to 60 mmHg, and expired gas and
  ventilation packets within 15 s.

No agent is carried across an unknown input gap. Cardiac output is assumed to be 5 L/min, with sensitivity runs
at 3 and 7 L/min. FRC comes from the patient's body size and stays fixed, shunt is zero, and temperature is
37 °C. These values are neither measured nor fitted.

Wash-in is the first ten minutes after inspired agent reaches 0.5%, and washout follows the final reading at or
above 0.5%. The comparator sets the end-tidal concentration equal to the recorded inspired concentration.
Concentration errors are in percentage points. The comparison takes measured inspired gas and ventilation as
inputs, so it does not test vaporizer or circuit delivery, lung recruitment, cardiovascular effects, or the
effects of sevoflurane on BIS.

Cases with the required tracks: 856. After the metadata rules: 856.
Screened: 331. Retained: 40 of 40 requested patients.

| Phase | Patients | Usable minutes | Bias (95% CI) | MAE (95% CI) | RMSE (95% CI) | Comparator MAE (95% CI) |
|---|---:|---:|---|---|---|---|
| all | 40 | 5674.4 | 0.09 (0.07 to 0.11) | 0.13 (0.11 to 0.14) | 0.17 (0.16 to 0.19) | 0.36 (0.33 to 0.39) |
| washin | 40 | 371.9 | -0.03 (-0.08 to 0.02) | 0.24 (0.21 to 0.27) | 0.40 (0.34 to 0.46) | 0.67 (0.61 to 0.74) |
| maintenance | 40 | 5059.0 | 0.10 (0.08 to 0.13) | 0.12 (0.10 to 0.14) | 0.14 (0.12 to 0.15) | 0.33 (0.31 to 0.36) |
| washout | 36 | 240.9 | 0.06 (0.03 to 0.09) | 0.12 (0.10 to 0.13) | 0.14 (0.12 to 0.16) | 0.41 (0.37 to 0.44) |

Paired all-phase MAE difference, AnaSim minus comparator: -0.23 (-0.25 to -0.21).
AnaSim has lower MAE in 39 of 40 patients.
Usable endpoint coverage: median 98.1% (IQR 96.4% to 99.0%) of the replay sampling interval.

| Assumed CO (L/min) | All-phase MAE (95% CI) |
|---:|---|
| 3.0 | 0.17 (0.16 to 0.19) |
| 5.0 | 0.13 (0.11 to 0.14) |
| 7.0 | 0.11 (0.10 to 0.13) |

Exclusions among screened cases:

- expired sevoflurane already present at recording start: 5
- incomplete input history: 41
- insufficient sevoflurane exposure in common interval: 1
- sevoflurane exposure already present or insufficient exposure: 208
- short zero-agent baseline or exposure interval: 30
- ventilation inputs outside comparison range: 6

## Interpretation

These comparisons test individual anesthetic models against clinical recordings. They do not establish the
accuracy of the whole simulator. BIS reflects EEG and monitor behavior and is not a direct measure of
consciousness. Clinicians chose doses in response to the patient's state, and unrecorded interventions and
delays in delivery or monitoring can contribute to errors. A population model is also not expected to reproduce
each patient's course. Sevoflurane results depend on unmeasured cardiac output, FRC, and ventilation history.
Conclusions about induction, emergence, wash-in, and washout should use the results for that phase, especially
where few minutes of emergence or washout are available.

Per-patient errors, track identifiers, and exclusions are in `vitaldb_anesthetics.json` and
`vitaldb_anesthetics_cases.csv`. Overlay plots show the patients at the minimum, quartiles, and maximum of
all-phase MAE, with their case IDs.

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5); [VitalDB documentation](https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md).
VitalDB data are used under CC BY-NC-SA 4.0 and its data use agreement. Downloaded tracks and overlay plots are
kept locally and are not in the repository.
