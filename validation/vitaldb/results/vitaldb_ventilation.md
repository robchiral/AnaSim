# VitalDB ventilation comparisons

Generated 2026-11-27.

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
to the one feature that it sets (median absolute error): ANALYZER_TAU 0.08: 0.153, 0.09: 0.120, 0.1: 0.087, 0.11: 0.053, 0.12: 0.031, 0.13: 0.038, 0.14: 0.065, 0.15: 0.091, 0.16: 0.120 (best 0.12); PHASE_II_VOLUME 0.05: 0.070, 0.075: 0.068, 0.1: 0.071, 0.125: 0.075, 0.15: 0.083, 0.175: 0.084, 0.2: 0.089, 0.25: 0.092 (best 0.075); PHASE_III_SLOPE 5: 1.893, 10: 1.313, 15: 0.805, 20: 1.006, 25: 1.055, 30: 1.302 (best 15).
CV stability (5 folds): ANALYZER_TAU: 0.12 (4/5 folds), 0.13 (1/5 folds); PHASE_II_VOLUME: 0.075 (3/5 folds), 0.05 (2/5 folds); PHASE_III_SLOPE: 15 (5/5 folds).
AnaSim uses ANALYZER_TAU 0.12 s, PHASE_II_VOLUME 0.1 L,
and PHASE_III_SLOPE 15 mmHg/L. The out-of-fold check below uses each
fold's training-fitted values and AnaSim's PBW-based series dead space (2.2 mL/kg PBW, unfitted).

| Feature (5-fold out-of-fold, n=60) | Recordings | AnaSim | Paired difference |
|---|---|---|---|
| Fall 90-10% (s) | 0.423 (0.393–0.466) | 0.403 (0.403–0.403) | -0.012 (-0.0626–0.0179) |
| Rise 10-90% (s) | 0.544 (0.495–0.613) | 0.507 (0.484–0.584) | -0.0176 (-0.0982–0.0522) |
| Share of breath above 50% | 0.595 (0.574–0.614) | 0.606 (0.594–0.62) | 0.011 (-0.00171–0.0256) |
| Phase III slope (% of plateau/s) | 3.14 (2.38–4.18) | 2.61 (1.88–3.35) | -0.569 (-1.55–0.134) |

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

### Settled `LungAeration` compliance and driving pressure across PEEP

Each supine case is settled for 5 min at its recorded pre-incision PEEP and tidal volume through
`LungAeration` and `RespiratoryMechanics` (unfitted against PEEP 0-2).

| Subgroup | n | Recorded C (mL/cmH₂O) | AnaSim C (mL/cmH₂O) | C MAE | C bias | Recorded ΔP (cmH₂O) | AnaSim ΔP (cmH₂O) | ΔP bias |
|---|---:|---|---|---:|---:|---|---|---:|
| PEEP 0-2 cmH₂O (all) | 1533 | 32.8 (28–37.8) | 34.1 (30.6–37.2) | 4.28 | +1.15 | 11.2 (10.2–13.3) | 10.9 (10–12.1) | -0.35 |
| PEEP 4-6 cmH₂O (all) | 1013 | 47.7 (40.6–55.3) | 45.8 (41.2–49.6) | 7.52 | -2.23 | 10.2 (8.16–11.2) | 10.2 (9.3–11.4) | +0.50 |
| PEEP 0-2, BMI >= 30 | 66 | 25.2 (22–29.1) | 26.9 (24.5–30.3) | 3.78 | +2.49 | 15.3 (13.3–16.3) | 13.5 (12.1–15.3) | -1.28 |
| PEEP 4-6, BMI >= 30 | 57 | 36.2 (32.1–42.6) | 34.7 (33.3–38.4) | 4.92 | -2.02 | 13.3 (11.2–15.3) | 14.1 (12.3–15.5) | +0.66 |
| PEEP 4-6, PFT Normal | 838 | 47.8 (40.8–55.4) | 45 (40.5–49.2) | 7.32 | -2.65 | 10.2 (8.16–11.2) | 10.3 (9.42–11.5) | +0.58 |
| PEEP 4-6, PFT Obstructive | 126 | 48.7 (42.1–57.7) | 48.6 (45.3–51.9) | 8.56 | -1.35 | 9.18 (8.16–11.2) | 9.74 (8.84–10.7) | +0.23 |

## Airway resistance by patient (VCV waveform cohort)

Recorded inspiratory resistance from the waveform end-of-flow drop versus `Patient.airway_resistance()`.

| Resistance subgroup | n | Recorded R (cmH₂O/(L/s)) | AnaSim R (cmH₂O/(L/s)) | MAE | Median bias |
|---|---:|---|---|---:|---:|
| All VCV windows | 60 | 9.58 (8.59–11.7) | 10 (10–10) | 1.95 | +0.42 |
| BMI under 30 | 56 | 9.45 (8.56–11.4) | 10 (10–10) | 1.95 | +0.55 |
| BMI 30 and over | 4 | 11.8 (11.2–12.9) | 10.7 (10.4–10.9) | 1.98 | -0.97 |
| PFT Normal | 52 | 9.74 (8.56–11.7) | 10 (10–10) | 2.03 | +0.26 |
| PFT Obstructive | 6 | 9.21 (8.99–9.57) | 10 (10–10) | 1.14 | +0.79 |

## End-tidal O₂

| | Recordings | AnaSim |
|---|---|---|
| FiO₂ (%) | 35 (34–36) | 37.3 (35.9–38) |
| FiO₂ - EtO₂ (%) | 6 (5–6) | 4.95 (4.58–5.86) |

Paired difference, AnaSim minus recorded: -0.437 (-1.17–0.31); mean absolute 0.87
percentage points (the Primus reports whole percent).
