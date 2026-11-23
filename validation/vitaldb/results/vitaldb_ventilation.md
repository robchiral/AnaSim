# VitalDB ventilation comparisons

Generated 2026-11-23.

These comparisons use 60 volume-control windows recorded on the Dräger Primus, from 1637 eligible
unique-subject cases inside AnaSim's patient domain (general anesthesia, oral single-lumen tube, open surgery,
supine; 53 fall inside the prior age ≤ 70 / BMI ≤ 32 subset). Each case contributes one steady window
during surgery. Another 12 windows were skipped because breaths did not peak at end-flow and exhale at
the set inspiratory time (pressure-regulated breaths) or gave implausible mechanics.
Fitted values use all 60 windows and are evaluated out-of-fold via 5-fold patient-grouped
cross-validation. Values are median (IQR).

Source: VitalDB (Lee et al., *Scientific Data*, 2022), used under CC BY-NC-SA 4.0 and its data use agreement.

## Settings

| Setting | Recordings | AnaSim default |
|---|---|---|
| Inspiratory pause (% of Ti) | 10 (10–10) | 10 |
| Pmax (cmH₂O) | 40.8 (40.8–40.8) | 40 |

## Mechanics from the recorded airway pressure

Each case's resistance and compliance are set so that AnaSim's breath has the recorded end-of-flow
drop and plateau. The other quantities are then predictions. Tissue viscoelasticity uses Jonson 1993
(viscoelastic compliance four times static compliance, time constant 0.82 s) without fitting.

| Quantity | Recordings | Single compartment | With tissue viscoelasticity |
|---|---|---|---|
| Resistance from end-of-flow drop (cmH₂O/(L/s)) | 9.58 (8.59–11.7) | matched | matched |
| Plateau compliance (mL/cmH₂O) | 45.9 (35.2–56) | matched | matched |
| Expiratory tail decay / RC | 0.908 (0.49–1.27) | 0.975 (0.812–1.04) | 0.948 (0.784–1.01) |
| Expiratory limb resistance (cmH₂O/(L/s)) | 3.39 (2.34–3.96) | 1.55 (1.41–3.28) | 1.39 (1.29–2.98) |
| Inspiratory knee time constant (ms) | 68.8 (54.1–86.1) | 31 (30–34.5) | 41.8 (36.8–49.1) |

Tail values use windows whose expiratory tail reached 0.5 cmH₂O (29 recorded). In the other windows, Paw returned to PEEP within 0.2 s.

Circuit limb resistance fitted on all windows (whole-breath RMSE): 1 0.425, 1.5 0.471, 2 0.521, 2.5 0.562, 3 0.613, 3.5 0.654. Best 1; CV stability: 1 cmH₂O/(L/s) (5/5 folds); AnaSim uses 1 cmH₂O/(L/s).

## Breath shape (RMSE, cmH₂O)

Airway sensor time constant fitted on all windows (transition RMSE): 10 ms 0.90, 15 ms 0.76, 20 ms 0.67, 25 ms 0.72, 30 ms 0.75, 40 ms 0.85. Best 20 ms; CV stability: 20 ms (4/5 folds), 25 ms (1/5 folds); AnaSim uses 20 ms.

| Segment (5-fold out-of-fold, n=60) | Single compartment | With tissue viscoelasticity |
|---|---|---|
| onset | 0.59 | 0.59 |
| ramp | 0.43 | 0.34 |
| end_flow | 0.69 | 0.66 |
| exp_fall | 0.67 | 0.67 |
| exp_tail | 0.27 | 0.27 |
| whole | 0.45 | 0.42 |

Tissue viscoelasticity lowered whole-breath RMSE in 41 of 60 out-of-fold windows.

## Pressure-control rise

10-90% Paw rise of 40 recorded pressure-control breaths: 0.249 (0.213–0.283) s.
AnaSim through the airway sensor: 0.15 s ramp 0.136 s, 0.2 s ramp 0.170 s, 0.25 s ramp 0.207 s, 0.3 s ramp 0.244 s, 0.35 s ramp 0.283 s. AnaSim uses a 0.28 s ramp.

## Capnogram shape

Features come from the CO₂ track alone. Each capnograph parameter was fitted in turn on all windows
to the one feature that it sets (median absolute error): ANALYZER_TAU 0.08: 0.154, 0.09: 0.120, 0.1: 0.087, 0.11: 0.053, 0.12: 0.031, 0.13: 0.038, 0.14: 0.065, 0.15: 0.091, 0.16: 0.120 (best 0.12); PHASE_II_VOLUME 0.05: 0.099, 0.075: 0.102, 0.1: 0.098, 0.125: 0.096, 0.15: 0.104, 0.175: 0.107, 0.2: 0.115, 0.25: 0.119 (best 0.125); PHASE_III_SLOPE 5: 1.931, 10: 1.359, 15: 0.872, 20: 0.865, 25: 1.052, 30: 1.220 (best 20).
CV stability (5 folds): ANALYZER_TAU: 0.12 (4/5 folds), 0.13 (1/5 folds); PHASE_II_VOLUME: 0.125 (2/5 folds), 0.05 (2/5 folds), 0.075 (1/5 folds); PHASE_III_SLOPE: 15 (4/5 folds), 20 (1/5 folds).
AnaSim uses ANALYZER_TAU 0.12 s, PHASE_II_VOLUME 0.1 L,
and PHASE_III_SLOPE 20 mmHg/L. The out-of-fold check below uses each
fold's training-fitted values and AnaSim's PBW-based series dead space (2.2 mL/kg PBW, unfitted).

| Feature (5-fold out-of-fold, n=60) | Recordings | AnaSim | Paired difference |
|---|---|---|---|
| Fall 90-10% (s) | 0.423 (0.393–0.466) | 0.403 (0.403–0.403) | -0.012 (-0.0626–0.0177) |
| Rise 10-90% (s) | 0.544 (0.495–0.613) | 0.532 (0.499–0.599) | -0.00412 (-0.0706–0.0712) |
| Share of breath above 50% | 0.595 (0.574–0.614) | 0.605 (0.592–0.618) | 0.00965 (-0.00359–0.0264) |
| Phase III slope (% of plateau/s) | 3.14 (2.38–4.18) | 2.64 (1.79–3.43) | -0.532 (-1.45–0.278) |

## Compliance by patient

Plateau compliance TV/(Pplat - PEEP) before incision in 2602 supine cases from numeric tracks:
PEEP 0-2 32.8 (28–37.8), PEEP 4-6 47.7 (40.6–55.3) mL/cmH₂O.

log C = a + b log(PBW/60) + c (BMI - 23) + d PEEP over all cases (bootstrap 95% CI):
b = 0.69 (0.64 to 0.74), c = -0.0230 (-0.0256 to -0.0209) per kg/m²,
d = 0.072 (0.068 to 0.074) per cmH₂O. C at PBW 60 kg, BMI 23, PEEP 5:
48.4 mL/cmH₂O. Age adds -0.0012 per year.

5-fold out-of-fold cases at PEEP 4-6 (n = 1013), mean absolute error in mL/cmH₂O:
constant 8.68, body-size fit 7.45,
AnaSim 7.58 (median bias -1.91).
AnaSim's default patient: static 54.8, plateau 48.8 mL/cmH₂O.

| Compliance subgroup (PEEP 4-6) | n | Recorded (mL/cmH₂O) | AnaSim MAE | AnaSim median bias |
|---|---:|---|---:|---:|
| Legacy core (open supine, age ≤ 70, BMI ≤ 32) | 397 | 46.3 (39.8–53.9) | 7.64 | -1.07 |
| BMI under 30 | 956 | 48.3 (41.3–55.9) | 7.73 | -2.00 |
| BMI 30 and over | 57 | 36.2 (32.1–42.6) | 4.98 | -1.21 |
| PFT Normal | 838 | 47.8 (40.8–55.4) | 7.37 | -2.48 |
| PFT Obstructive | 126 | 48.7 (42.1–57.7) | 8.76 | -0.99 |

## End-tidal O₂

| | Recordings | AnaSim |
|---|---|---|
| FiO₂ (%) | 35 (34–36) | 37.3 (35.9–38) |
| FiO₂ - EtO₂ (%) | 6 (5–6) | 4.95 (4.58–5.86) |

Paired difference, AnaSim minus recorded: -0.438 (-1.17–0.309); mean absolute 0.87
percentage points (the Primus reports whole percent).
