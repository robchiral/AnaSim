# VitalDB cardiac output, stroke volume, and SVR

Generated 2026-11-27.

## Methods

Replays all eligible unique-subject domain cases (ASA 1-3, no vasoactive infusions or transfusion,
EBL <= 500 mL) recording EV1000 or Vigileo continuous CO and SV. Because pulse-contour CO monitors
are uncalibrated and typically placed after induction, the primary endpoint is the paired percentage
change from the pre-incision window to surgery (>= 15 min after incision) in untreated patients.
Derived SVR uses MAP / CO (Wood units) alongside EV1000 SVR when recorded.

Eligible candidates: 286. Retained: 66 (TIVA: 42, Sevoflurane: 24; untreated: 27, paired pre-incision + surgery: 22).

## Pre-incision to surgery relative changes (untreated, paired)

| Subgroup | n | dMAP rec / mod (%) | dHR rec / mod (%) | dCO rec / mod (%) | dSV rec / mod (%) | dSVR rec / mod (%) |
|---|---:|---|---|---|---|---|
| All paired untreated | 22 | +16.9% / +12.1% | +6.3% / -8.0% | +4.6% / -2.3% | +0.5% / +6.7% | +15.1% / +15.1% |
| TIVA, all | 14 | +18.5% / +12.9% | +6.4% / -5.6% | +3.8% / -4.2% | -0.2% / +1.6% | +18.0% / +17.9% |
| TIVA, Open supine | 7 | +29.6% / +13.3% | +1.0% / -7.2% | +6.1% / -5.3% | +7.5% / +2.1% | +22.6% / +19.4% |
| TIVA, Other positions | 7 | +7.3% / +12.6% | +11.8% / -3.9% | +1.6% / -3.0% | -7.9% / +1.1% | +13.4% / +16.3% |
| Sevoflurane, all | 8 | +14.2% / +10.7% | +6.0% / -12.2% | +6.1% / +0.8% | +1.7% / +15.5% | +10.0% / +10.3% |

Open supine TIVA MAP % change error (model minus recorded): -16.4 (-25.2 to -8.9), n=7 percentage points.
Open supine TIVA CO % change error (model minus recorded): -11.4 (-19.9 to -4.0), n=7 percentage points.
Open supine TIVA SV % change error (model minus recorded): -5.4 (-13.5 to 2.9), n=7 percentage points.
Open supine TIVA SVR (MAP/CO) % change error (model minus recorded): -3.2 (-23.5 to 10.3), n=7 percentage points.

## Steady-state surgery medians (untreated; uncalibrated pulse-contour absolute values)

| Subgroup | n | CO rec / mod (L/min) | SV rec / mod (mL) | MAP/CO rec / mod (Wood) | MAP err (mmHg) |
|---|---:|---|---|---|---|
| All untreated | 27 | 4.20 / 3.42 | 63 / 57 | 19.2 / 24.0 | -2.3 (-7.4 to 3.1) |
| TIVA, Open supine | 9 | 3.70 / 3.22 | 63 / 55 | 25.3 / 25.7 | -5.7 (-12.3 to 0.6) |
| TIVA, Other positions | 10 | 4.40 / 3.55 | 64 / 58 | 18.5 / 24.2 | 0.8 (-6.7 to 9.3) |
| Sevoflurane, all | 8 | 4.60 / 3.51 | 65 / 56 | 18.1 / 21.6 | -2.2 (-13.2 to 8.1) |

EV1000 direct SVR during surgery (untreated, n=2): median recorded 2056 vs model 2064 dyn*s/cm^5.

Exclusions:

- incomplete input history: 39
- infusion rate and volume disagree: 85
- insufficient HR before induction: 1
- insufficient induction or surgery duration: 6
- insufficient valid surgery CO/SV readings: 1
- less than thirty seconds of zero delivery before induction: 8
- no cuff MAP within 15 minutes before induction: 16
- propofol delivery began before recording: 19
- recorded volatile or nitrous oxide exposure: 44
- short surgery window: 1

Source: [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5). VitalDB data are used under CC BY-NC-SA 4.0
and its data use agreement.
