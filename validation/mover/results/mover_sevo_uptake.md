# MOVER external validation: sevoflurane uptake and washout

Generated 2026-11-27. Frozen AnaSim parameters (no refitting on MOVER).

## Methods

Replays 1-minute recorded inspired sevoflurane (`SEV In`) through AnaSim's `VolatilePK`
model against recorded end-tidal sevoflurane (`SEV ET`) across BMI-stratified unique-MRN
ASA 1-3 General Anesthesia cases without isoflurane, desflurane, or N₂O. Because tidal
volume and respiratory rate are absent from the MOVER flowsheet subset, alveolar ventilation
uses standard intraoperative settings (VT = 8 mL/kg PBW, RR = 12/min, PBW-scaled dead space)
and anesthetized cardiac index 2.6 L/min/m². Errors are predicted minus recorded (% atm).

Screened candidates: 1000. Retained: 235.

## Maintenance and overall EtSevo error (% atm)

| Subgroup | n | Recorded / Model EtSevo (%) | Maintenance bias (95% CI) | Maintenance MAE (95% CI) | Uptake 0-15m bias | Washout bias |
|---|---:|---|---|---|---|---|
| All patients | 235 | 1.80 / 1.88 | 0.11 (0.10 to 0.12) | 0.12 (0.11 to 0.13) | -0.04 (-0.05 to -0.02) | 0.01 (-0.01 to 0.04) |
| BMI under 25 | 57 | 1.80 / 1.96 | 0.10 (0.09 to 0.12) | 0.11 (0.10 to 0.13) | -0.02 (-0.05 to 0.01) | -0.08 (-0.13 to -0.03) |
| BMI 25-29.9 | 53 | 1.70 / 1.78 | 0.11 (0.10 to 0.13) | 0.12 (0.10 to 0.14) | -0.04 (-0.07 to -0.02) | -0.01 (-0.04 to 0.03) |
| BMI 30-34.9 | 65 | 1.80 / 1.92 | 0.11 (0.09 to 0.12) | 0.12 (0.10 to 0.13) | -0.02 (-0.06 to 0.02) | 0.04 (-0.05 to 0.12) |
| BMI 35 and over | 60 | 1.80 / 1.90 | 0.11 (0.09 to 0.13) | 0.13 (0.11 to 0.14) | -0.07 (-0.11 to -0.03) | 0.06 (0.02 to 0.09) |
| Age under 45 | 87 | 2.00 / 2.08 | 0.09 (0.08 to 0.11) | 0.11 (0.09 to 0.12) | -0.02 (-0.05 to 0.01) | 0.01 (-0.03 to 0.06) |
| Age 45-64 | 84 | 1.70 / 1.87 | 0.12 (0.10 to 0.13) | 0.12 (0.11 to 0.14) | -0.04 (-0.08 to -0.01) | -0.01 (-0.06 to 0.04) |
| Age 65 and over | 64 | 1.50 / 1.61 | 0.12 (0.11 to 0.14) | 0.13 (0.11 to 0.15) | -0.05 (-0.08 to -0.03) | 0.04 (-0.00 to 0.08) |
| Female | 130 | 1.80 / 1.89 | 0.11 (0.10 to 0.12) | 0.12 (0.11 to 0.13) | -0.04 (-0.06 to -0.01) | -0.00 (-0.04 to 0.03) |
| Male | 105 | 1.80 / 1.87 | 0.11 (0.10 to 0.12) | 0.11 (0.10 to 0.12) | -0.03 (-0.06 to -0.01) | 0.04 (-0.01 to 0.08) |

## Exclusions

- gas recording gap over 5 min: 434
- insufficient sevoflurane administration: 9
- isoflurane, desflurane, or nitrous oxide exposure: 193
- less than 30 paired 1-min SEV In/ET samples: 118

Source: [Samad et al., JAMIA Open, 2023](https://pubmed.ncbi.nlm.nih.gov/37860605/) (MOVER EPIC dataset, UCI Health, 2017-2022, DOI 10.1093/jamiaopen/ooad084).
