# MOVER external validation: pulse pressure and intraoperative ABGs/labs

Generated 2026-11-27. Frozen AnaSim parameters (no refitting on MOVER).

## Awake pre-induction NIBP pulse pressure (model minus recorded)

| Subgroup | n | Recorded PP (mmHg) | Model PP (mmHg) | PP error (95% CI) | SBP error | DBP error |
|---|---:|---:|---:|---|---|---|
| All patients | 1500 | 58.0 | 55.3 | -3.9 (-4.8 to -2.9) | -3.0 (-3.6 to -2.3) | 0.9 (0.6 to 1.2) |
| Age under 45 | 495 | 53.0 | 46.3 | -7.6 (-8.8 to -6.3) | -5.6 (-6.5 to -4.7) | 1.9 (1.6 to 2.3) |
| Age 45-59 | 422 | 56.0 | 54.4 | -3.5 (-5.1 to -1.9) | -2.8 (-3.9 to -1.7) | 0.7 (0.2 to 1.2) |
| Age 60-69 | 301 | 62.0 | 62.8 | -1.6 (-3.7 to 0.5) | -1.4 (-2.9 to 0.1) | 0.3 (-0.4 to 0.8) |
| Age 70 and over | 282 | 78.0 | 76.1 | -0.6 (-3.0 to 1.7) | -0.5 (-2.2 to 1.2) | 0.1 (-0.6 to 0.8) |
| Normotensive | 1220 | 57.0 | 54.2 | -3.3 (-4.3 to -2.3) | -2.6 (-3.3 to -1.9) | 0.7 (0.4 to 1.0) |
| Hypertensive | 280 | 68.5 | 61.8 | -6.5 (-9.2 to -4.0) | -4.7 (-6.6 to -2.9) | 1.8 (1.1 to 2.6) |
| BMI under 30 | 1022 | 58.0 | 56.5 | -2.7 (-3.8 to -1.6) | -2.1 (-2.9 to -1.3) | 0.6 (0.3 to 0.9) |
| BMI 30 and over | 478 | 59.0 | 53.7 | -6.7 (-8.2 to -5.1) | -5.0 (-6.1 to -3.9) | 1.6 (1.2 to 2.1) |

## Intraoperative invasive arterial pulse pressure (MAP-matched)

| Subgroup | n | Recorded PP (mmHg) | Model PP (mmHg) | PP error (95% CI) |
|---|---:|---:|---:|---|
| All A-line | 500 | 53.0 | 51.6 | -3.1 (-4.2 to -2.0) |
| Age under 60 | 220 | 48.0 | 44.7 | -5.0 (-6.5 to -3.3) |
| Age 60 and over | 280 | 57.0 | 56.4 | -1.7 (-3.3 to -0.1) |
| MAP under 75 mmHg | 167 | 48.0 | 46.7 | -3.1 (-5.0 to -1.3) |
| MAP 75 mmHg and over | 333 | 55.0 | 53.7 | -3.2 (-4.5 to -1.8) |

## Intraoperative arterial oxygenation (PaO2, mmHg; model minus recorded)

Standard ventilator settings assumed (VT = 8 mL/kg PBW, RR = 12/min, PEEP = 5 cmH2O)
because PEEP and TV are absent from the MOVER flowsheet subset; FiO2 (%) is recorded.

| Subgroup | n | Median FiO2 (%) | Recorded PaO2 | Model PaO2 | Mean error (95% CI) |
|---|---:|---:|---:|---:|---|
| All ABGs | 400 | 46 | 180 | 176 | 14.4 (5.6 to 23.8) |
| Age under 50 | 104 | 46 | 184 | 193 | 35.7 (13.9 to 59.6) |
| Age 50-69 | 189 | 47 | 185 | 179 | 14.2 (0.6 to 29.1) |
| Age 70 and over | 107 | 46 | 167 | 148 | -6.0 (-21.1 to 8.0) |
| BMI under 25 | 150 | 46 | 212 | 192 | 0.6 (-13.3 to 16.1) |
| BMI 25-29.9 | 134 | 46 | 182 | 174 | 18.6 (1.4 to 36.9) |
| BMI 30-34.9 | 75 | 52 | 141 | 145 | 25.9 (6.1 to 50.3) |
| BMI 35 and over | 41 | 46 | 116 | 125 | 30.3 (3.6 to 61.5) |
| Open | 335 | 47 | 181 | 179 | 15.9 (5.4 to 26.7) |
| Laparoscopic or robotic | 65 | 46 | 166 | 161 | 6.6 (-10.5 to 27.5) |

## Intraoperative acid-base, hemodilution Hb, and glucose (median recorded vs model)

- **PaCO2** (n=400): recorded 41.0 mmHg vs model 33.4 mmHg (err -6.8 (-7.6 to -6.0))
- **pH** (n=400): recorded 7.390 vs model 7.459 (err 0.069 (0.061 to 0.078))
- **HCO3-** (n=400): recorded 24.0 mmol/L vs model 23.6 mmol/L (err -0.8 (-1.1 to -0.5))
- **Ionized Ca2+** (n=400): recorded 1.14 mmol/L vs model 1.19 mmol/L (err 0.06 (0.05 to 0.07))
- **Hemodilution dHb** (n=9): recorded -1.20 g/dL vs model -0.45 g/dL (err 0.64 (0.11 to 1.11))
- **Intraoperative dGlucose** (n=188): recorded 1.5 mg/dL vs model -5.1 mg/dL (err -8.5 (-15.3 to -2.3))

## ABG Exclusions

- missing or unstable FiO2 before ABG: 114
- transfusion or QBL over 500 mL before ABG: 18

Source: [Samad et al., JAMIA Open, 2023](https://pubmed.ncbi.nlm.nih.gov/37860605/) (MOVER EPIC dataset, UCI Health, 2017-2022, DOI 10.1093/jamiaopen/ooad084).
