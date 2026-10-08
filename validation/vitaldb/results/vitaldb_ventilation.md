# AnaSim ventilation vs Dräger Primus recordings (VitalDB)

40 volume-control windows from 1384 eligible cases: adults in AnaSim's domain,
general anesthesia, oral tube, open surgery, supine, one steady mid-surgery window each.
6 further windows were skipped because their breaths did not peak at the end of flow and
exhale at the set Ti (pressure-regulated breaths) or gave implausible mechanics. Fitted values use
the 20 calibration windows and are checked on the 20 held-out windows.
Source: VitalDB (Lee et al., Sci Data 2022), CC BY-NC-SA 4.0. Median (IQR).

## Settings

| Setting | Recordings | AnaSim default |
|---|---|---|
| Inspiratory pause (% of Ti) | 10 (10–10) | 10 |
| Pmax (cmH2O) | 40.8 (33.7–40.8) | 40 |

## Mechanics from the recorded airway pressure

Each case's resistance and compliance are set so that AnaSim's breath has the recorded end-of-flow
drop and plateau; the other quantities are then predictions. Tissue viscoelasticity uses Jonson 1993
(viscoelastic compliance 4x static, time constant 0.82 s) without fitting.

| Quantity | Recordings | Single compartment | With tissue viscoelasticity |
|---|---|---|---|
| Resistance from end-of-flow drop (cmH2O/(L/s)) | 9.21 (8.03–11.6) | matched | matched |
| Plateau compliance (mL/cmH2O) | 36.2 (32–48.8) | matched | matched |
| Expiratory tail decay / RC | 0.62 (0.435–0.966) | 1.02 (0.855–1.08) | 0.978 (0.838–1.07) |
| Expiratory limb resistance (cmH2O/(L/s)) | 2.89 (1.94–3.83) | 1.68 (1.45–3.52) | 1.46 (1.33–2.95) |
| Inspiratory knee time constant (ms) | 64.4 (56.3–77.2) | 31.9 (28.9–35.3) | 47.2 (38–51.7) |

Tail values use windows whose expiratory tail reached 0.5 cmH2O (24 recorded); in the rest Paw returned to PEEP within 0.2 s.

Circuit limb resistance fitted on the calibration windows (whole-breath RMSE): 1 0.410, 1.5 0.441, 2 0.489, 2.5 0.528, 3 0.584, 3.5 0.666. Best 1; AnaSim uses 1 cmH2O/(L/s).

## Breath shape (RMSE, cmH2O)

Airway sensor time constant fitted on the calibration windows (transition RMSE): 10 ms 1.02, 15 ms 0.72, 20 ms 0.65, 25 ms 0.71, 30 ms 0.77, 40 ms 0.83. Best 20 ms; AnaSim uses 20 ms.

| Segment (held-out) | Single compartment | With tissue viscoelasticity |
|---|---|---|
| onset | 0.55 | 0.55 |
| ramp | 0.44 | 0.38 |
| end_flow | 0.66 | 0.60 |
| exp_fall | 0.85 | 0.78 |
| exp_tail | 0.22 | 0.20 |
| whole | 0.46 | 0.42 |

Tissue viscoelasticity lowered whole-breath RMSE in 15 of 20 held-out windows.

## Pressure-control rise

10-90% Paw rise of 20 recorded pressure-control breaths: 0.23 (0.207–0.263) s.
AnaSim through the airway sensor: 0.15 s ramp 0.136 s, 0.2 s ramp 0.170 s, 0.25 s ramp 0.207 s, 0.3 s ramp 0.244 s, 0.35 s ramp 0.283 s. AnaSim uses a 0.28 s ramp.

## Capnogram shape

Features come from the CO2 track alone. Each capnograph parameter was fitted on the calibration
windows to the feature it alone sets, in turn (median absolute error): ANALYZER_TAU 0.08: 0.151, 0.09: 0.117, 0.1: 0.084, 0.11: 0.051, 0.12: 0.020, 0.13: 0.023, 0.14: 0.051, 0.15: 0.082, 0.16: 0.113 (best 0.12); PHASE_II_VOLUME 0.05: 0.082, 0.075: 0.081, 0.1: 0.082, 0.125: 0.083, 0.15: 0.088, 0.175: 0.097, 0.2: 0.110, 0.25: 0.127 (best 0.075); PHASE_III_SLOPE 5: 2.771, 10: 2.269, 15: 1.851, 20: 1.456, 25: 1.262, 30: 1.466 (best 25).
AnaSim uses ANALYZER_TAU 0.12 s, PHASE_II_VOLUME 0.1 L,
and PHASE_III_SLOPE 20 mmHg/L; the held-out check below uses the fitted
values. The series dead space is the respiratory model's 2.2 mL/kg and was not fitted, so the share
of each breath above 50% tests it.

| Feature (held-out) | Recordings | AnaSim | Paired difference |
|---|---|---|---|
| Fall 90-10% (s) | 0.416 (0.397–0.446) | 0.403 (0.403–0.403) | -0.0136 (-0.0427–0.00614) |
| Rise 10-90% (s) | 0.565 (0.534–0.608) | 0.564 (0.515–0.693) | -0.0115 (-0.0569–0.171) |
| Share of breath above 50% | 0.598 (0.559–0.618) | 0.596 (0.576–0.621) | 0.0155 (-0.0053–0.0351) |
| Phase III slope (% of plateau/s) | 3.4 (2.2–6.28) | 3.28 (2.4–5.2) | -0.193 (-1.77–0.99) |

## Compliance by patient

Plateau compliance TV/(Pplat - PEEP) before incision in 586 cases from numeric tracks:
PEEP 0-2 32.9 (28–37.8), PEEP 4-6 45.9 (39.4–51.7) mL/cmH2O.

log C = a + b log(PBW/60) + c (BMI - 23) + d PEEP over all cases (bootstrap 95% CI):
b = 0.60 (0.50-0.70), c = -0.0263 (-0.0318--0.0206) per kg/m2,
d = 0.065 (0.058-0.072) per cmH2O. C at PBW 60 kg, BMI 23, PEEP 5:
46.3 mL/cmH2O. Age adds -0.0012 per year.

Held-out cases at PEEP 4-6 (n = 91), mean absolute error in mL/cmH2O:
constant 8.12, body-size fit 7.26,
AnaSim 7.32 (median bias -1.06).
AnaSim's default patient: static 53.1, plateau 47.4 mL/cmH2O.

## End-tidal O2

| | Recordings | AnaSim |
|---|---|---|
| FiO2 (%) | 35 (34–37) | 36.8 (36–39) |
| FiO2 - EtO2 (%) | 6 (5–6) | 5.03 (4.16–6.45) |

Paired AnaSim - recorded difference: -0.617 (-1.38–0.986); mean absolute 1.42
percentage points (the Primus reports whole percent).
