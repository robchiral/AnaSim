# VitalDB sevoflurane to BIS validation

Generated 2026-11-27.

## Methods

Replays recorded end-tidal sevoflurane (Primus/EXP_SEVO) through VolatilePK cerebral equilibration
and optional remifentanil infusions through RemifentanilPKEleveld into BISModel without fitting
parameters or offsets. Evaluates surgical maintenance (>= 15 min after incision to opend, SQI >= 90,
EtSevo >= 0.4%) and post-surgery washout against recorded BIS/BIS and a constant BIS=40 comparator.

Eligible candidates: 1295. Retained: 1081 (635 with remifentanil, 446 without).

## Surgical maintenance BIS (predicted minus recorded)

| Subgroup | n | Median EtMAC | Recorded / Model BIS | Bias (95% CI) | MAE (95% CI) | Comparator MAE (BIS=40) |
|---|---:|---:|---|---|---|---|
| All maintenance | 1081 | 1.03 | 40.7 / 39.8 | -0.2 (-0.7 to 0.3) | 7.6 (7.3 to 7.9) | 6.7 (6.5 to 6.9) |
| Without remifentanil | 446 | 1.05 | 39.8 / 39.3 | 0.2 (-0.6 to 1.0) | 8.2 (7.8 to 8.7) | 7.1 (6.8 to 7.5) |
| With remifentanil | 635 | 1.03 | 41.2 / 40.0 | -0.5 (-1.1 to 0.1) | 7.2 (6.8 to 7.6) | 6.4 (6.1 to 6.6) |
| Age under 45 | 199 | 0.96 | 37.5 / 42.4 | 5.0 (3.9 to 6.0) | 8.0 (7.3 to 8.8) | 7.4 (7.0 to 7.9) |
| Age 45-59 | 387 | 1.03 | 39.4 / 40.1 | 1.0 (0.1 to 1.7) | 7.1 (6.8 to 7.6) | 6.7 (6.3 to 7.0) |
| Age 60-69 | 293 | 1.06 | 41.7 / 39.0 | -2.7 (-3.5 to -1.7) | 7.5 (6.9 to 8.0) | 6.3 (5.9 to 6.7) |
| Age 70 and over | 202 | 1.10 | 42.2 / 37.8 | -4.0 (-5.1 to -2.9) | 8.3 (7.6 to 9.0) | 6.6 (6.0 to 7.2) |
| BMI under 30 | 1029 | 1.04 | 40.7 / 39.7 | -0.3 (-0.8 to 0.2) | 7.6 (7.3 to 7.9) | 6.6 (6.4 to 6.8) |
| BMI 30 and over | 52 | 1.00 | 39.6 / 40.6 | 1.4 (-0.6 to 3.4) | 7.5 (6.3 to 8.9) | 7.8 (6.7 to 8.9) |
| EtMAC under 0.85 | 120 | 0.77 | 44.0 / 51.0 | 9.3 (7.6 to 10.9) | 11.6 (10.3 to 12.9) | 7.9 (6.9 to 8.7) |
| EtMAC 0.85-1.05 | 457 | 0.97 | 41.1 / 42.1 | 1.5 (0.8 to 2.1) | 6.7 (6.4 to 7.1) | 6.9 (6.6 to 7.2) |
| EtMAC 1.05 and over | 504 | 1.16 | 39.6 / 36.0 | -4.0 (-4.6 to -3.4) | 7.5 (7.1 to 7.9) | 6.2 (6.0 to 6.5) |

## Emergence / washout BIS

Retained emergence traces: n=542. Bias: 5.8 (4.9 to 6.8); MAE: 10.5 (9.9 to 11.2) (comparator BIS=40 MAE: 24.5 (23.9 to 25.3)).

Exclusions:

- incomplete EtSevo track: 15
- less than 10 min usable surgical maintenance BIS: 55
- nitrous oxide exposure: 3
- short surgery under 30 min: 56
- short usable BIS/sevoflurane overlap: 85

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.

