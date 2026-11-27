# VitalDB intraoperative core temperature

Generated 2026-11-27.

## Methods

Compares recorded Solar8000/BT changes 30 and 60 minutes after probe equilibration
(5 min after first reading >= 35.5 C) across retained TIVA and sevoflurane cases
against AnaSim's ThermalModel both without and with active forced-air warming
(warming status is not logged in VitalDB).

Eligible retained cases with BT track: 637. Retained with 60-min curve: 360.

| Subgroup | n | Recorded d60, C (95% CI) | Unwarmed model d60 (C) | Forced-air model d60 (C) | Recorded d30 (C) |
|---|---:|---|---|---|---|
| All cases | 360 | -0.24 (-0.27 to -0.21) | -1.25 | -0.31 | -0.17 |
| BMI under 22 | 104 | -0.27 (-0.34 to -0.22) | -1.41 | -0.40 | -0.19 |
| BMI 22-26.9 | 215 | -0.24 (-0.28 to -0.20) | -1.22 | -0.30 | -0.17 |
| BMI 27 and over | 41 | -0.19 (-0.27 to -0.11) | -1.00 | -0.14 | -0.12 |
| Age under 50 | 87 | -0.22 (-0.29 to -0.16) | -1.28 | -0.34 | -0.17 |
| Age 50-69 | 212 | -0.22 (-0.27 to -0.18) | -1.25 | -0.31 | -0.15 |
| Age 70 and over | 61 | -0.33 (-0.39 to -0.27) | -1.22 | -0.27 | -0.24 |
| Open | 249 | -0.24 (-0.28 to -0.21) | -1.26 | -0.33 | -0.18 |
| Videoscopic or robotic | 111 | -0.24 (-0.30 to -0.18) | -1.23 | -0.28 | -0.14 |

Median equilibrated starting core temperature: 36.30 C. Share of patients within [unwarmed, forced-air warmed] +/- 0.1 C at 60 min: 182/360 (51%).

Exclusions:

- initial core temperature outside 35.5-38.0 C: 9
- less than 60 min post-equilibration temperature data: 251
- probe never reached 35.5 C: 17

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.
