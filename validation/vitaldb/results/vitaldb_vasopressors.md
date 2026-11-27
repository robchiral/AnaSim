# VitalDB continuous vasopressor infusions

Generated 2026-11-27.

## Methods

Replays eligible unique-subject domain cases (ASA 1-4, no RBC/FFP transfusion, EBL <= 1500 mL)
receiving a single continuous Orchestra norepinephrine (20 mcg/mL) or phenylephrine (100 mcg/mL)
infusion with invasive arterial MAP (Solar8000/ART_MBP) and ECG HR. Each case is run both with
the recorded pump infusion and as a zero-pressor counterfactual. Steady-state errors reflect
confounding by indication because pressors are started and titrated upward in patients who are
already more hypotensive than typical population predictions.

Eligible candidates: 29. Retained: 11.

## Active infusion steady state (model minus recorded)

| Drug | n | Median rate (mcg/kg/min) | MAP err WITH pressor (mmHg) | MAP err WITHOUT pressor (mmHg) | Modeled MAP gain (mmHg) | HR err WITH (bpm) | HR delta WITH-WITHOUT (bpm) |
|---|---:|---:|---|---|---|---|---|
| Norepinephrine | 2 | 0.036 | 7.6 (-7.5 to 22.7) | -2.2 (-19.6 to 15.2) | 9.8 (7.5 to 12.1) | 2.4 (0.0 to 4.8) | -0.3 (-0.7 to 0.0) |
| Phenylephrine | 9 | 0.274 | 27.0 (14.5 to 39.7) | 15.3 (6.0 to 24.2) | 11.7 (8.2 to 15.8) | -8.3 (-14.1 to -2.1) | -0.7 (-1.3 to -0.2) |

## Acute infusion step-ups (+2 to +5 min vs preceding 3 min)

- **Phenylephrine** (13 step-ups, median +0.237 mcg/kg/min): recorded dMAP 1.0 (-5.3 to 8.3) vs model dMAP 6.6 (3.9 to 9.4) mmHg; recorded dHR 6.5 (-1.4 to 19.0) vs model dHR -0.7 (-1.3 to -0.1) bpm.

Exclusions:

- inactive or implausible pressor rate: 1
- incomplete input history: 4
- insufficient HR before induction: 1
- nitrous oxide exposure: 1
- pressor rate and volume disagree: 2
- propofol rate and volume disagree: 4
- recorded volatile or nitrous oxide exposure: 5

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.
