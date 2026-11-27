# MOVER external validation: acute vasopressor, beta-blocker, and reversal responses

Generated 2026-11-27. Frozen AnaSim parameters (no refitting on MOVER).

## Methods

Screens settled intraoperative maintenance episodes (>= 15 min after Induction, no competing
vasoactive, autonomic, or anesthetic bolus within ±5 min, stable pre-event baseline over
-3 to -1 min) with 1-minute invasive arterial MAP (`MAP-ART A-line`) for vasopressors and
beta-blockers, and 1-minute ECG HR across all General Anesthesia cases for neostigmine +
glycopyrrolate versus glycopyrrolate alone. At most one isolated event per unique patient per
drug arm is evaluated.

## Acute response summary (recorded vs model)

| Arm | n | Median dose | Pre-event MAP / HR | Peak dMAP rec / mod (mmHg) | dMAP err (95% CI) | Peak/early dHR rec / mod (bpm) | Late dHR +4-8m rec / mod (bpm) |
|---|---:|---|---|---|---|---|---|
| Phenylephrine bolus (all) | 180 | 100.0 | 68 / 79 | +13.0 / +16.2 | 2.0 (-0.0 to 3.9) | -4.5 / -2.9 | -1.3 / -0.2 |
|   Phenylephrine 50 mcg | 24 | 50.0 | 66 / 74 | +12.0 / +9.4 | -2.3 (-6.6 to 1.4) | -5.0 / -1.4 | -2.3 / +0.5 |
|   Phenylephrine 100 mcg | 133 | 100.0 | 69 / 79 | +12.0 / +16.2 | 2.8 (0.5 to 5.0) | -4.0 / -3.0 | -1.2 / -0.2 |
|   Phenylephrine 200 mcg | 23 | 200.0 | 67 / 80 | +15.0 / +21.8 | 2.1 (-5.9 to 9.1) | -5.0 / -4.2 | -2.4 / -1.6 |
| Ephedrine bolus (all) | 180 | 5.0 | 69 / 63 | +12.0 / +18.8 | 8.2 (5.6 to 10.8) | +5.0 / +7.1 | +3.0 / +7.1 |
|   Ephedrine 5 mg (first dose) | 55 | 5.0 | 69 / 61 | +10.0 / +16.5 | 4.4 (0.9 to 7.7) | +5.0 / +5.9 | +5.8 / +5.9 |
|   Ephedrine 5 mg (repeat dose) | 67 | 5.0 | 68 / 63 | +12.0 / +16.4 | 1.7 (-2.9 to 5.7) | +4.0 / +6.2 | +2.0 / +6.2 |
|   Ephedrine 10 mg (first dose) | 39 | 10.0 | 67 / 63 | +19.0 / +34.0 | 17.3 (11.0 to 24.0) | +6.0 / +10.6 | +4.8 / +10.0 |
|   Ephedrine 10 mg (repeat dose) | 19 | 10.0 | 69 / 64 | +10.0 / +33.5 | 23.3 (17.6 to 28.3) | +4.0 / +10.4 | +1.6 / +10.0 |
| Norepinephrine infusion start | 22 | 3.0 | 68 / 90 | +5.5 / +10.5 | 5.2 (1.5 to 9.0) | +0.0 / -0.2 | +0.4 / -0.0 |
| Esmolol bolus | 32 | 17.5 | 90 / 100 | -5.0 / -3.6 | 3.8 (-0.9 to 7.8) | -15.0 / -3.2 | -11.8 / -1.8 |
| Labetalol bolus | 22 | 5.0 | 102 / 82 | -6.0 / -13.3 | -3.4 (-7.9 to 1.6) | -4.5 / -7.0 | -1.7 / -6.7 |
| Neostigmine + Glycopyrrolate | 180 | 3.4 | 80 / 77 | +3.0 / -4.5 | -7.7 (-9.1 to -6.5) | +5.0 / -5.8 | +5.1 / -10.5 |
| Glycopyrrolate alone | 180 | 0.2 | 76 / 56 | +2.0 / +8.5 | 5.5 (3.8 to 7.1) | +1.7 / +12.6 | +8.8 / +23.6 |

## Sustained +6 to +8 min dMAP (phenylephrine vs ephedrine)

- **Phenylephrine** (n=180): recorded 5.5 mmHg (0.0 to 12.7) vs model 5.1 mmHg (4.1 to 6.4)
- **Ephedrine** (n=180): recorded 8.8 mmHg (1.9 to 18.0) vs model 12.2 mmHg (10.0 to 19.0)

## Exclusions

- active volatile washout during neostigmine reversal: 133
- incomplete post-event HR: 40
- incomplete post-event arterial MAP: 6
- unstable or missing pre-event HR: 347
- unstable or missing pre-event arterial MAP: 234

Source: [Samad et al., JAMIA Open, 2023](https://pubmed.ncbi.nlm.nih.gov/37860605/) (MOVER EPIC dataset, UCI Health, 2017-2022, DOI 10.1093/jamiaopen/ooad084).
