# VitalDB anesthetic comparisons

Generated 2026-10-13T23:53:42.116599+00:00.
Git revision `2d704b251a1a16957feed608804c7e3bf8147bef`. Fixed model source SHA256 `489a73798ce2521e09984128f34fd215be8bed7cc1cca0428bad54788d0cf88f`.

## Methods

Models use the checkout's production parameters without fitting, individual calibration, observed-BIS initialization,
or fitted time shifts. This evaluates isolated anesthetic components, with normal organ function and unchanged blood volume.
Patients meet the existing ventilation cohort rules: general anesthesia, oral single-lumen tube, open surgery, supine,
age 18 to 70 years, weight 50 to 100 kg, height 150 to 200 cm, and BMI 18 to 32 kg/m².
Seed 20261013 fixes candidate order; each cohort includes at most one case per patient. No validation data are used to tune models.
Replay step 1 s; endpoint sampling every 5 s. Each reported phase needs at least one usable minute;
each case needs ten usable minutes in total. Phase sample counts vary with measurement availability.
The all-phase score includes usable intervals between the named phases. Eligibility uses recorded inputs and signal quality, not prediction errors.
Errors are predicted minus measured. Tables show the mean of per-patient metrics with patient bootstrap 95% intervals
(2,000 resamples). Long cases therefore have the same weight as short cases. Baseline comparisons use identical samples.

## Propofol and BIS

PPF20_RATE is held between recorded packets and integrated in mL; 20 mg/mL converts delivery to model mg/s.
Cumulative pump volume must start at zero (at most 0.05 mL), with at least 30 seconds of common zero-rate/volume history before induction.
Rates must be between 0 and 1,500 mL/h, with at least 5 mL delivered and a replay interval of at least 15 minutes after induction.
The integrated rate must agree with the recorded volume within the larger of 5% or 0.5 mL. Counter reversals over 0.25 mL
are excluded; smaller counter artifacts do not affect rate-based replay. Delivery gaps over 5 s are excluded.
Recorded supplemental propofol, midazolam, and fentanyl bolus totals must be zero. Dexmedetomidine pump tracks are excluded.
Remifentanil delivery must be recorded and nonzero; the production propofol model uses its opioid-regimen covariate.
Remifentanil analgesia and stimulus responses are not evaluated. No hemorrhage or individualized surgical arousal is replayed.
MAC must stay at or below 0.05 and inspired N2O at or below 1%; gas coverage must start before propofol delivery and have no
gaps over 30 s during replay. BIS uses SQI at least 90, with no more than 2 s since the latest BIS/SQI packet.
The Eleveld effect site, published BIS processing delay, and production display smoothing are included; display noise is omitted.
Induction ends at incision or ten minutes after first delivery, whichever is earlier. Maintenance begins after both;
emergence begins after both surgical end and the final recorded nonzero infusion. These are recording phases, not observed
loss or return of consciousness. The comparator is constant BIS 40. Pump CP/CE estimates are never treated as measured concentrations.

Track-eligible cases 826; after metadata rules 817;
screened 817; retained 84 of 100 requested patients.

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

The requested sample was not reached; the table reports all eligible patients found.

## Sevoflurane uptake

Recorded inspired sevoflurane drives production VolatilePK, initialized with no agent in any compartment.
Inspired and expired concentrations must start at or below 0.1%, with 30 s before exposure reaches 0.5%.
Inspired exposure must reach at least 1%, with at least 15 minutes between the first and last readings at or above 0.5%.
Recorded tidal volume and CO2 respiratory rate give alveolar ventilation using AnaSim's 2.2 mL/kg PBW dead space.
Input gaps over 30 s, inspired concentrations outside 0 to 8%, VT outside 0 to 1,500 mL, RR outside 0 to 50/min,
and N2O exposure over 1% are excluded. Endpoint samples need VT at least 200 mL, RR at least 5/min, EtCO2 15 to 60 mmHg,
and expired gas/ventilation packets within 15 s. No agent is carried across an unknown input gap.
Cardiac output is assumed to be 5 L/min, with prespecified 3 and 7 L/min sensitivity runs. FRC uses the patient's production
body-size formula and stays fixed; shunt is zero and temperature is 37 °C. These quantities are not measured or fitted.
Wash-in is the first ten minutes after inspired agent reaches 0.5%; washout follows its final reading at or above 0.5%.
The comparator sets end-tidal equal to recorded inspired sevoflurane. Concentration errors are percentage points.
This conditions on measured inspired gas and ventilation; vaporizer/circuit delivery, lung recruitment, cardiovascular effects,
and sevoflurane BIS effects are not tested.

Track-eligible cases 856; after metadata rules 856;
screened 331; retained 40 of 40 requested patients.

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

These are external clinical comparisons of fixed component models, not proof of accuracy for the whole simulator.
BIS reflects EEG and monitor behavior rather than a direct consciousness label. Doses were selected by clinicians in response
to patient state. Unrecorded interventions and delivery/monitor timing can contribute to errors; population predictions need
not reproduce individual trajectories. A model that does not outperform its comparator should be reported as such.
Sevoflurane results depend on unmeasured cardiac output, FRC, and ventilation history. Conclusions about transition behavior
need the phase-specific results, especially where few emergence or washout minutes are available.

Per-patient errors, track identifiers, and exclusions are in `vitaldb_anesthetics.json` and `vitaldb_anesthetics_cases.csv`.
Overlay patients are selected at the minimum, quartiles, and maximum of all-phase MAE, with their case IDs shown.

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5); [VitalDB documentation](https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md).
VitalDB data are used under CC BY-NC-SA 4.0 and its data use agreement. Downloaded tracks and overlay PNGs stay in the ignored local cache/results paths.
