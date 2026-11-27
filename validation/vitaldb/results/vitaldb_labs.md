# VitalDB intraoperative ABGs, hemodilution, and glucose

Generated 2026-11-27.

## Methods

Evaluates the first settled intraoperative ABG/chemistry panel (>= 5 min after incision)
across retained TIVA and sevoflurane cases (no RBC/FFP transfusion, EBL <= 500 mL) using
recorded FiO2, PEEP, TV, RR, preop Hb/glucose, crystalloid/colloid, and EBL.

Retained TIVA/Sevo candidates with lab rows: 555. Analyzed with valid ABG + vent: 94.

## Oxygenation (PaO2, mmHg; model minus recorded)

| Subgroup | n | Median FiO2 (%) | Recorded PaO2 | Model PaO2 | Mean error, mmHg (95% CI) |
|---|---:|---:|---:|---:|---|
| All ABGs | 94 | 36 | 152 | 118 | -17.2 (-34.6 to 1.3) |
| Age under 50 | 20 | 34 | 166 | 126 | -9.7 (-37.4 to 26.8) |
| Age 50-69 | 53 | 37 | 143 | 117 | -8.8 (-28.9 to 14.0) |
| Age 70 and over | 21 | 37 | 155 | 89 | -45.7 (-83.8 to -6.9) |
| BMI under 25 | 69 | 36 | 160 | 123 | -25.4 (-46.3 to -3.5) |
| BMI 25 and over | 25 | 38 | 109 | 104 | 5.4 (-11.3 to 23.3) |
| PEEP <= 2 cmH2O | 45 | 37 | 143 | 91 | -45.1 (-66.3 to -22.4) |
| PEEP >= 4 cmH2O | 48 | 36 | 156 | 132 | 7.5 (-15.7 to 35.1) |

## PaCO2 - EtCO2 gradient (mmHg)

| Subgroup | n | Recorded gradient | Model gradient | Error (95% CI) |
|---|---:|---:|---:|---|
| All ABGs | 94 | 6.0 | 5.5 | -1.2 (-2.0 to -0.5) |
| Open | 45 | 5.0 | 6.2 | -0.1 (-1.1 to 1.0) |
| Videoscopic/Robotic | 49 | 7.0 | 5.1 | -2.3 (-3.3 to -1.3) |
| Age under 60 | 44 | 4.0 | 5.3 | -0.3 (-1.5 to 0.8) |
| Age 60 and over | 50 | 7.0 | 6.1 | -2.0 (-3.0 to -1.0) |

## Acid-base, hemodilution Hb, and glucose (median recorded vs model)

- **pH** (n=94): recorded 7.410 vs model 7.451 (error 0.0 (0.0 to 0.1)).
- **HCO3-** (n=94): recorded 26.0 vs model 23.8 mmol/L (error -2.6 (-3.2 to -2.1)).
- **Hemodilution dHb** (Hct/3 minus preop Hb, n=88): recorded -1.1 (-1.4 to -0.9) vs model -0.5 (-0.6 to -0.4) g/dL.
- **Intraoperative dGlucose** (intraop minus preop, n=63): recorded -8.4 (-20.4 to 1.7) vs model -5.7 (-6.5 to -4.9) mg/dL.
- **Ionized Ca2+** (n=93): recorded 1.14 vs model 1.20 mmol/L.
- **Lactate** (n=94): recorded 0.90 vs model 1.02 mmol/L.

Exclusions:

- no intraoperative ABG panel: 461

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.

