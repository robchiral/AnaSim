# MOVER external validation: intraoperative core temperature

Generated 2026-11-27. Frozen AnaSim parameters (no refitting on MOVER).

## Methods

Compares recorded `Temp` (converted from °F to °C) changes 30, 60, and 120 minutes after
probe equilibration (5 min after first reading >= 35.5 °C) across unique-MRN non-CPB
General Anesthesia cases against AnaSim's `ThermalModel` both without and with active
forced-air warming (38 °C).

Screened long-case candidates: 1600. Retained with >= 60 min curve: 613.

## 60-minute post-equilibration core temperature change (°C)

| Subgroup | n | Recorded d60 (95% CI) | Unwarmed model d60 | Forced-air model d60 | Recorded d30 | Recorded d120 (n) |
|---|---:|---|---|---|---|---|
| All cases | 613 | -0.14 (-0.18 to -0.10) | -1.23 | -0.34 | -0.18 | +0.02 (389) |
| BMI under 22 | 73 | -0.25 (-0.38 to -0.12) | -1.59 | -0.58 | -0.28 | -0.11 (51) |
| BMI 22-26.9 | 238 | -0.20 (-0.26 to -0.14) | -1.39 | -0.48 | -0.22 | -0.04 (154) |
| BMI 27-31.9 | 166 | -0.07 (-0.13 to -0.00) | -1.15 | -0.29 | -0.13 | +0.12 (102) |
| BMI 32-39.9 | 107 | -0.07 (-0.17 to 0.01) | -0.90 | -0.07 | -0.13 | +0.09 (63) |
| BMI 40 and over | 29 | -0.04 (-0.22 to 0.16) | -0.66 | +0.14 | -0.09 | +0.16 (19) |
| Age under 50 | 247 | -0.07 (-0.13 to -0.01) | -1.27 | -0.38 | -0.16 | +0.21 (136) |
| Age 50-69 | 256 | -0.16 (-0.22 to -0.11) | -1.18 | -0.30 | -0.18 | -0.02 (180) |
| Age 70 and over | 110 | -0.25 (-0.33 to -0.15) | -1.25 | -0.35 | -0.24 | -0.22 (73) |
| Open | 423 | -0.17 (-0.21 to -0.13) | -1.24 | -0.35 | -0.21 | +0.04 (286) |
| Laparoscopic or robotic | 161 | -0.06 (-0.14 to 0.01) | -1.19 | -0.31 | -0.12 | -0.01 (91) |

Median equilibrated starting core temperature: 36.11 °C. Share within [unwarmed, forced-air warmed] ± 0.1 °C at 60 min: 39.6% (243/613).

## Exclusions

- insufficient valid temperature readings: 182
- temperature span under 70 min: 22

Source: [Samad et al., JAMIA Open, 2023](https://pubmed.ncbi.nlm.nih.gov/37860605/) (MOVER EPIC dataset, UCI Health, 2017-2022, DOI 10.1093/jamiaopen/ooad084).
