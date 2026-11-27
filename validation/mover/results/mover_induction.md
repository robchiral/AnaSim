# MOVER external validation: induction hemodynamics

Generated 2026-11-27. Frozen AnaSim parameters (no refitting on MOVER).

## Methods

Replays unique-patient adult General Anesthesia encounters (age 18-85, weight 50-200 kg,
height 150-200 cm, BMI 18-60 kg/m², no CPB, OLV, LMA, ketamine, dexmedetomidine, isoflurane,
desflurane, or N₂O) using exact MAR bolus/infusion timestamps (including pre-op midazolam),
recorded `Induction` and `Intubation` events, 1-minute `SEV In`, pre-op hemoglobin and
Cockcroft-Gault CrCl, and 1-minute arterial line or NIBP MAP and ECG HR. Errors are
model minus recorded over `+1` to `+10 min` post-induction (capped at `Anesthesia Ready + 3 min`
and censored before any first vasoactive drug).

Screened unique-MRN GA encounters with Induction + Intubation: 13229. Evaluated: 506 (Sevo pressor-free: 332, Sevo pressor-censored: 102, TIVA: 37, Etomidate: 35).

## Propofol + Sevoflurane induction (strictly pressor-free) (n=332)

| Subgroup | n | MAP err, all (mmHg) | MAP err, stop-rule (mmHg) | HR err (bpm) |
|---|---:|---|---|---|
| All | 332 | -11.3 (-12.8 to -9.8) | -10.4 (-12.0 to -8.9) | -6.0 (-7.2 to -4.7) |
| Age under 45 | 118 | -7.6 (-9.4 to -5.8) | -7.1 (-9.0 to -5.2) | -9.2 (-11.4 to -6.9) |
| Age 45-59 | 89 | -12.7 (-15.2 to -10.3) | -12.5 (-15.0 to -10.1) | -5.2 (-7.9 to -2.9) |
| Age 60-69 | 61 | -17.8 (-21.8 to -13.7) | -15.2 (-19.7 to -10.7) | -4.5 (-7.2 to -1.7) |
| Age 70 and over | 64 | -9.9 (-13.8 to -6.1) | -9.3 (-13.4 to -5.3) | -2.5 (-4.7 to -0.0) |
| BMI under 25 | 102 | -9.7 (-12.3 to -7.0) | -8.8 (-11.7 to -5.9) | -7.6 (-9.6 to -5.3) |
| BMI 25-29.9 | 117 | -11.9 (-14.7 to -9.2) | -10.4 (-13.3 to -7.4) | -5.4 (-7.4 to -3.0) |
| BMI 30-34.9 | 68 | -12.3 (-15.1 to -9.5) | -12.3 (-15.0 to -9.5) | -4.6 (-7.4 to -2.0) |
| BMI 35 and over | 45 | -11.8 (-15.3 to -8.5) | -11.6 (-15.2 to -8.2) | -5.8 (-9.2 to -2.0) |
| Normotensive (no HTN dx) | 272 | -11.3 (-12.9 to -9.8) | -10.4 (-12.0 to -8.8) | -6.9 (-8.4 to -5.5) |
| Hypertensive (HTN dx) | 60 | -11.1 (-15.0 to -7.4) | -10.8 (-14.7 to -7.0) | -1.9 (-4.8 to 0.9) |
| ASA 1-2 | 159 | -9.3 (-11.4 to -7.5) | -8.5 (-10.8 to -6.7) | -7.8 (-9.6 to -5.8) |
| ASA 3+ | 173 | -13.1 (-15.3 to -10.9) | -12.2 (-14.5 to -10.0) | -4.3 (-6.2 to -2.6) |
| No pre-op midazolam | 85 | -9.7 (-13.2 to -6.6) | -9.4 (-13.2 to -6.2) | -3.3 (-5.8 to -1.0) |
| With pre-op midazolam | 247 | -11.8 (-13.2 to -10.3) | -10.8 (-12.4 to -9.2) | -6.9 (-8.3 to -5.4) |
| Non-invasive NIBP | 331 | -11.2 (-12.6 to -9.9) | -10.4 (-11.9 to -8.9) | -6.0 (-7.2 to -4.6) |

Strictly before `Anesthesia Ready` (n=322): MAP error -9.5 (-11.0 to -7.8) mmHg.

Time-course change in MAP from pre-induction (median, IQR) and window MAP error (model minus recorded):

| Window | n | Recorded dMAP (%) | Model dMAP (%) | MAP error (mmHg, 95% CI) |
|---|---:|---|---|---|
| 1-2 min | 193 | -4% (-13 to 2) | -11% (-15 to -7) | -5.3 (-7.5 to -3.0) |
| 2-5 min | 318 | -6% (-16 to 5) | -13% (-18 to -7) | -7.2 (-9.4 to -5.4) |
| 5-10 min | 316 | -5% (-15 to 4) | -21% (-27 to -14) | -16.1 (-18.0 to -14.5) |
| lowest in 10 min | 332 | -15% (-24 to -7) | -23% (-29 to -18) | - |
| post-intubation peak | 309 | 0% (-13 to 14) | -9% (-16 to 1) | -9.0 (-11.6 to -6.4) |

## Propofol + Sevoflurane induction (all: pressor-free + censored at first pressor) (n=434)

| Subgroup | n | MAP err, all (mmHg) | MAP err, stop-rule (mmHg) | HR err (bpm) |
|---|---:|---|---|---|
| All | 434 | -9.4 (-10.9 to -8.1) | -8.7 (-10.3 to -7.2) | -4.9 (-5.9 to -3.7) |
| Age under 45 | 135 | -6.0 (-8.0 to -3.9) | -5.4 (-7.6 to -3.1) | -8.6 (-10.6 to -6.6) |
| Age 45-59 | 110 | -10.5 (-12.9 to -8.1) | -10.3 (-12.8 to -7.8) | -4.4 (-6.5 to -2.3) |
| Age 60-69 | 93 | -13.1 (-17.0 to -9.3) | -11.5 (-15.4 to -7.4) | -2.4 (-4.6 to -0.3) |
| Age 70 and over | 96 | -9.2 (-12.6 to -5.6) | -8.6 (-12.3 to -5.0) | -2.4 (-4.2 to -0.2) |
| BMI under 25 | 149 | -7.5 (-10.1 to -5.1) | -6.7 (-9.3 to -4.0) | -5.8 (-7.6 to -3.7) |
| BMI 25-29.9 | 146 | -10.4 (-13.0 to -7.5) | -9.1 (-11.6 to -6.0) | -4.6 (-6.3 to -2.6) |
| BMI 30-34.9 | 85 | -10.2 (-13.1 to -7.0) | -10.3 (-13.0 to -7.4) | -3.8 (-6.2 to -1.3) |
| BMI 35 and over | 54 | -10.5 (-13.7 to -7.3) | -10.2 (-13.5 to -6.8) | -4.7 (-8.5 to -1.7) |
| Normotensive (no HTN dx) | 354 | -9.3 (-11.0 to -7.7) | -8.4 (-10.4 to -6.7) | -5.5 (-6.7 to -4.4) |
| Hypertensive (HTN dx) | 80 | -9.9 (-13.4 to -6.7) | -9.7 (-13.4 to -6.6) | -2.1 (-4.6 to 0.3) |
| ASA 1-2 | 194 | -8.3 (-10.2 to -6.3) | -7.6 (-9.7 to -5.5) | -6.8 (-8.4 to -5.2) |
| ASA 3+ | 240 | -10.3 (-12.4 to -8.1) | -9.5 (-11.6 to -7.3) | -3.3 (-4.8 to -2.0) |
| No pre-op midazolam | 112 | -9.4 (-12.3 to -6.4) | -9.1 (-12.2 to -6.1) | -3.0 (-5.2 to -1.1) |
| With pre-op midazolam | 322 | -9.4 (-10.8 to -7.9) | -8.5 (-10.0 to -6.9) | -5.5 (-6.8 to -4.2) |
| Invasive A-line | 5 | -6.5 (-15.4 to 3.0) | -8.0 (-16.0 to 2.0) | 0.1 (-7.9 to 6.7) |
| Non-invasive NIBP | 429 | -9.4 (-10.9 to -8.1) | -8.7 (-10.1 to -7.3) | -4.9 (-6.0 to -3.9) |

Strictly before `Anesthesia Ready` (n=420): MAP error -7.9 (-9.5 to -6.3) mmHg.

Time-course change in MAP from pre-induction (median, IQR) and window MAP error (model minus recorded):

| Window | n | Recorded dMAP (%) | Model dMAP (%) | MAP error (mmHg, 95% CI) |
|---|---:|---|---|---|
| 1-2 min | 263 | -4% (-14 to 3) | -11% (-15 to -7) | -4.9 (-6.9 to -2.9) |
| 2-5 min | 415 | -7% (-18 to 5) | -12% (-18 to -6) | -5.8 (-7.5 to -4.1) |
| 5-10 min | 392 | -7% (-18 to 4) | -21% (-27 to -14) | -14.0 (-15.9 to -12.2) |
| lowest in 10 min | 434 | -16% (-27 to -7) | -23% (-29 to -17) | - |
| post-intubation peak | 403 | -2% (-15 to 13) | -8% (-15 to 2) | -6.8 (-9.1 to -4.4) |

## TIVA (Propofol + Remifentanil infusion) induction (n=37)

| Subgroup | n | MAP err, all (mmHg) | MAP err, stop-rule (mmHg) | HR err (bpm) |
|---|---:|---|---|---|
| All | 37 | -14.0 (-19.5 to -7.6) | -13.5 (-19.1 to -6.7) | -3.9 (-7.3 to -0.2) |
| Age under 45 | 8 | -7.0 (-16.7 to 4.8) | -6.4 (-16.1 to 5.3) | -4.7 (-17.0 to 6.9) |
| Age 45-59 | 8 | -20.5 (-29.5 to -10.6) | -18.8 (-29.1 to -7.9) | -2.3 (-7.4 to 2.7) |
| Age 60-69 | 12 | -11.2 (-22.9 to 1.3) | -11.2 (-22.9 to 1.3) | -3.5 (-8.6 to 1.9) |
| Age 70 and over | 9 | -18.2 (-30.3 to -5.7) | -18.2 (-30.3 to -5.7) | -5.5 (-13.1 to 2.8) |
| BMI under 25 | 11 | -13.6 (-22.3 to -2.8) | -12.4 (-22.0 to -1.2) | -8.0 (-13.9 to -2.9) |
| BMI 25-29.9 | 15 | -19.6 (-27.3 to -12.1) | -19.2 (-26.9 to -11.8) | -4.1 (-8.6 to 0.7) |
| BMI 30-34.9 | 9 | -6.1 (-22.0 to 10.1) | -6.2 (-22.1 to 10.0) | 2.6 (-5.8 to 9.6) |
| Normotensive (no HTN dx) | 27 | -13.0 (-20.0 to -5.4) | -12.3 (-19.5 to -4.7) | -2.5 (-6.2 to 1.2) |
| Hypertensive (HTN dx) | 10 | -16.8 (-27.6 to -5.8) | -16.8 (-27.6 to -5.8) | -7.7 (-16.5 to 1.0) |
| ASA 1-2 | 17 | -17.8 (-25.1 to -10.5) | -16.7 (-24.4 to -9.2) | -8.9 (-13.7 to -4.1) |
| ASA 3+ | 20 | -10.8 (-18.9 to -1.1) | -10.8 (-18.9 to -1.1) | 0.0 (-4.4 to 4.6) |
| No pre-op midazolam | 16 | -7.2 (-16.0 to 2.4) | -6.7 (-15.5 to 2.9) | -2.9 (-8.8 to 2.4) |
| With pre-op midazolam | 21 | -19.2 (-26.8 to -12.2) | -18.7 (-26.5 to -11.5) | -4.7 (-10.1 to 0.2) |
| Non-invasive NIBP | 37 | -14.0 (-19.5 to -7.6) | -13.5 (-19.1 to -6.7) | -3.9 (-7.3 to -0.2) |

Strictly before `Anesthesia Ready` (n=37): MAP error -13.9 (-19.7 to -7.1) mmHg.

Time-course change in MAP from pre-induction (median, IQR) and window MAP error (model minus recorded):

| Window | n | Recorded dMAP (%) | Model dMAP (%) | MAP error (mmHg, 95% CI) |
|---|---:|---|---|---|
| 1-2 min | 21 | -8% (-19 to 5) | -12% (-22 to -7) | -7.2 (-17.6 to 2.5) |
| 2-5 min | 35 | -9% (-22 to 5) | -19% (-23 to -14) | -9.1 (-16.8 to -1.7) |
| 5-10 min | 35 | -7% (-16 to -1) | -29% (-37 to -22) | -21.8 (-27.7 to -15.8) |
| lowest in 10 min | 37 | -19% (-29 to -9) | -29% (-38 to -24) | - |
| post-intubation peak | 32 | -6% (-20 to 7) | -17% (-23 to -10) | -11.6 (-20.1 to -4.5) |

## Etomidate bolus induction (ASA 1-4) (n=35)

| Subgroup | n | MAP err, all (mmHg) | MAP err, stop-rule (mmHg) | HR err (bpm) |
|---|---:|---|---|---|
| All | 35 | -3.5 (-7.9 to 1.1) | -2.3 (-6.7 to 2.4) | -4.7 (-8.2 to -1.0) |
| Age under 45 | 9 | -10.0 (-14.7 to -3.9) | -5.1 (-12.1 to 3.8) | -11.0 (-19.8 to -2.0) |
| Age 45-59 | 8 | -2.0 (-11.4 to 9.0) | -2.3 (-11.4 to 8.3) | 0.7 (-2.9 to 4.0) |
| Age 60-69 | 8 | -4.3 (-14.8 to 5.2) | -4.3 (-14.8 to 5.2) | -5.7 (-13.3 to 4.3) |
| Age 70 and over | 10 | 1.9 (-4.9 to 9.2) | 1.8 (-4.9 to 8.9) | -2.6 (-6.7 to 1.8) |
| BMI under 25 | 17 | -2.5 (-9.2 to 4.0) | -0.9 (-7.9 to 6.3) | -8.4 (-13.3 to -3.7) |
| BMI 25-29.9 | 8 | -3.2 (-12.0 to 4.0) | -3.5 (-12.0 to 3.6) | 0.0 (-8.2 to 5.6) |
| BMI 30-34.9 | 5 | -4.7 (-14.1 to 6.3) | -1.6 (-9.4 to 8.6) | -6.0 (-13.6 to -0.0) |
| BMI 35 and over | 5 | -5.9 (-15.9 to 6.3) | -5.9 (-15.9 to 6.3) | 1.8 (-7.0 to 12.3) |
| Normotensive (no HTN dx) | 28 | -4.4 (-9.3 to 0.1) | -2.9 (-7.8 to 2.3) | -5.8 (-10.1 to -1.7) |
| Hypertensive (HTN dx) | 7 | 0.4 (-9.0 to 8.0) | 0.2 (-9.0 to 7.5) | -0.1 (-4.9 to 5.2) |
| ASA 1-2 | 5 | -8.6 (-14.7 to -1.0) | -5.5 (-10.7 to -0.2) | -15.4 (-24.0 to -6.5) |
| ASA 3+ | 30 | -2.6 (-7.6 to 2.7) | -1.8 (-7.1 to 3.6) | -2.9 (-6.4 to 0.5) |
| No pre-op midazolam | 21 | -2.1 (-7.7 to 3.7) | -2.1 (-7.8 to 3.6) | -1.9 (-6.1 to 2.3) |
| With pre-op midazolam | 14 | -5.6 (-11.8 to 0.9) | -2.5 (-9.4 to 4.4) | -9.0 (-15.2 to -3.3) |
| Invasive A-line | 15 | -5.4 (-11.0 to 1.5) | -2.7 (-9.2 to 5.0) | -5.4 (-11.6 to -0.5) |
| Non-invasive NIBP | 20 | -2.0 (-7.9 to 4.6) | -2.0 (-7.9 to 4.6) | -4.1 (-9.8 to 0.9) |

Strictly before `Anesthesia Ready` (n=35): MAP error -1.3 (-6.0 to 3.5) mmHg.

Time-course change in MAP from pre-induction (median, IQR) and window MAP error (model minus recorded):

| Window | n | Recorded dMAP (%) | Model dMAP (%) | MAP error (mmHg, 95% CI) |
|---|---:|---|---|---|
| 1-2 min | 28 | 4% (-1 to 11) | 2% (1 to 4) | 2.0 (-3.1 to 7.8) |
| 2-5 min | 34 | 11% (-1 to 20) | 7% (3 to 14) | -3.3 (-8.3 to 1.8) |
| 5-10 min | 33 | 8% (-9 to 20) | -1% (-8 to 2) | -7.1 (-12.7 to -1.7) |
| lowest in 10 min | 35 | -7% (-24 to -1) | -8% (-11 to -1) | - |
| post-intubation peak | 31 | 19% (9 to 35) | 20% (9 to 30) | -1.0 (-7.5 to 5.8) |

## Exclusions during candidate screening

- intubation outside 0-10 min of induction: 350
- isoflurane, desflurane, or nitrous oxide exposure: 26
- ketamine or dexmedetomidine during induction: 499
- no inspired sevoflurane recorded: 41
- no pre-induction HR within 5 min: 29
- no pre-induction MAP within 15 min: 127
- vasoactive drug before +1 min: 940
- volatile exposure in TIVA candidate: 64

Source: [Samad et al., JAMIA Open, 2023](https://pubmed.ncbi.nlm.nih.gov/37860605/) (MOVER EPIC dataset, UCI Health, 2017-2022, DOI 10.1093/jamiaopen/ooad084).
