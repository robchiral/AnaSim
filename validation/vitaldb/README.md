# VitalDB comparisons

These scripts compare AnaSim's ventilation, gas exchange, propofol and BIS,
sevoflurane uptake, and hemodynamics with VitalDB recordings used for model
development and calibration. Run them from the repository root after completing
the [contributor setup](../../CONTRIBUTING.md#setup). Uncached tracks download
into `validation/vitaldb/.vitaldb_cache/`.

```bash
python -m validation.vitaldb.vitaldb_ventilation
python -m validation.vitaldb.vitaldb_peep
python -m validation.vitaldb.vitaldb_anesthetics --figures
python -m validation.vitaldb.vitaldb_sevo_bis
python -m validation.vitaldb.vitaldb_tiva
python -m validation.vitaldb.vitaldb_sevo
python -m validation.vitaldb.vitaldb_pulse_pressure
python -m validation.vitaldb.vitaldb_co
python -m validation.vitaldb.vitaldb_vasopressors
python -m validation.vitaldb.vitaldb_laparoscopy
python -m validation.vitaldb.vitaldb_labs
python -m validation.vitaldb.vitaldb_temperature
```

Each script writes its report to `results/` and screens unique subjects inside
AnaSim's adult patient domain (general anesthesia, oral single-lumen tube, age
18 to 85 years, weight 50 to 200 kg, height 150 to 200 cm, BMI 18 to 60 kg/m²).
The replay and compliance scripts pass preoperative hemoglobin and
Cockcroft-Gault creatinine clearance into `Patient` when recorded and retain a
legacy-core subgroup row. Shared data loading, replay, and cross-validation
helpers live in `vitaldb_data.py` and `vitaldb_replay.py`. CPU-bound scripts
accept `--workers N` (defaulting to the available CPU count).

The source data come from VitalDB (Lee et al., *Scientific Data*, 2022) and are
used under CC BY-NC-SA 4.0 and the VitalDB data use agreement.

## Ventilation and PEEP

`vitaldb_ventilation.py` writes `results/vitaldb_ventilation.md`. It evaluates
circuit resistance, airway sensor lag, capnography, VCV waveform airway
resistance by BMI and PFT subgroup, pre-incision supine compliance with 5-fold
patient-grouped cross-validation and full-cohort fits, and settled
`LungAeration` compliance and driving pressure across PEEP 0-2 and PEEP 4-6.
Compliance includes all eligible supine cases by default; pass `--cases 0` or
`--pcv-cases 0` to remove waveform caps, `--folds` to change fold count, or
`--figure validation/vitaldb/results/overlay.png` (requires Matplotlib) for
breath overlays.

`vitaldb_peep.py` audits pre-incision PEEP steps, replays each retained step
through `LungAeration` and `RespiratoryMechanics`, and writes
`results/vitaldb_peep.json`. By default it reads only cached tracks; pass
`--fetch` to screen `Primus/SET_INTER_PEEP` first and download remaining tracks
only for cases with a candidate step. Use `--cache PATH` to read another cache
directory.

## Anesthetic drugs

`vitaldb_anesthetics.py` replays recorded propofol delivery (with syringe-change
volume resets unwrapped) through the PK and BIS models, and recorded inspired
sevoflurane through the uptake model, without fitting parameters. By default it
screens all eligible cases (`--tiva-cases 0 --sevo-cases 0`) and writes
`results/vitaldb_anesthetics.md`, per-patient CSV errors, and JSON metadata.

The report gives each patient's mean absolute BIS error. Published studies
usually report median absolute prediction error (MDAPE) with different BIS
filtering, so those numbers are not directly comparable.

| Option | Effect |
|--------|--------|
| `--tiva-cases`, `--sevo-cases` | Limit retained sample sizes (`0` uses all eligible cases) |
| `--cohorts` | Run `both`, `tiva`, or `sevo` |
| `--seed` | Changes candidate ordering and bootstrap seed |
| `--cache`, `--output` | Change data and report directories |
| `--dt 0.5` | Runs with a smaller time step |
| `--figures` | Writes representative overlays (requires Matplotlib) |

`vitaldb_sevo_bis.py` replays recorded end-tidal sevoflurane (and optional
remifentanil) through cerebral equilibration into `BISModel` across surgical
maintenance and post-surgery emergence without fitting parameters.

## Hemodynamics

The two hemodynamic scripts drive the full engine across eligible ASA 1 to 3
patients (normotensive and hypertensive) with recorded drug delivery, fluids,
blood loss, pre-induction cuff MAP and HR, and intubation and surgical
stimulation. Positioning and pneumoperitoneum are not modeled. Errors are model
minus recorded.

`vitaldb_tiva.py` replays propofol-remifentanil cases, including treated
patients. From propofol to incision, each cuff reading is compared with model
MAP averaged over the preceding 40 seconds and stops at the first reading that
could prompt treatment (MAP below 65 mmHg, SBP below 90, or HR below 45).

`vitaldb_sevo.py` replays bolus propofol induction with fentanyl (when given),
remifentanil infusions (when present), and inspired sevoflurane maintenance.
Propofol onset is inferred from recorded BIS and shifted 30 seconds later than
model BIS latency implies, matching the TIVA BIS lag; intubation is placed 2.5
minutes after propofol. Both `vitaldb_tiva.py` and `vitaldb_sevo.py` stratify
untreated surgical maintenance by procedure and position (`Open, supine`,
`Laparoscopic or robotic, supine`, and `Other positions`), and `vitaldb_sevo.py`
also reports an extended untreated surgery cohort that retains cases without a
pre-induction BIS fall.

Both scripts fit a patient-offset likelihood model with patient bootstrap 95%
intervals, report 5-fold cross-validation stability, and accept `--override '{...}'`
or `--cv-grid '{"field": [v1, v2]}'` (`--folds 5`, evaluating pre-incision
induction windows) for cross-validated sweeps.

`vitaldb_co.py` compares paired pre-incision to surgery relative changes in
cardiac output, stroke volume, and derived/direct SVR (plus steady-state
medians) against uncalibrated EV1000 and Vigileo pulse-contour recordings.
`vitaldb_pulse_pressure.py` compares awake pre-induction cuff SBP, DBP, and
pulse pressure, intraoperative invasive arterial pulse pressure (at matched
MAP), and CVP against `HemodynamicModel`.
`vitaldb_vasopressors.py` replays continuous Orchestra norepinephrine
(20 mcg/mL) and phenylephrine (100 mcg/mL) infusions with invasive arterial MAP
against zero-pressor counterfactuals and acute rate step-ups (noting
confounding by indication during steady-state pressor administration).

## Additional endpoints

`vitaldb_laparoscopy.py` compares paired pre-incision to mid-surgery changes
in compliance, EtCO₂, minute ventilation, CO₂ elimination proxy, and untreated
MAP between supine open and supine laparoscopic or robotic surgeries.
`vitaldb_labs.py` evaluates settled intraoperative arterial blood gases and
chemistry panels (`PaO2`, `PaCO2 - EtCO2`, `pH`, `HCO3-`, hemodilution `dHb`,
`dGlucose`, `iCa2+`, and lactate) using recorded ventilator settings, fluids,
and blood loss.
`vitaldb_temperature.py` compares 30- and 60-minute post-equilibration
`Solar8000/BT` core temperature trajectories across BMI, age, and surgical
approach against `ThermalModel` both without and with forced-air warming.

## Current results

MAP error, in mmHg with 95% CI:

| Report | Patients | Window | MAP error |
|---|---:|---|---|
| TIVA (all) | 511 | Propofol to incision | -2.7 (-3.7 to -1.9) |
| TIVA (legacy core) | 330 | Propofol to incision | -2.4 (-3.4 to -1.6) |
| TIVA, untreated (open supine) | 91 | Surgery | -1.0 (-3.3 to 1.3) |
| TIVA, untreated (all) | 182 | Surgery | -3.2 (-4.9 to -1.5) |
| Sevoflurane (all) | 136 | Propofol to incision | -6.1 (-8.2 to -4.1) |
| Sevoflurane (legacy core) | 29 | Propofol to incision | -5.5 (-9.5 to -1.5) |
| Sevoflurane, extended untreated (open supine) | 54 | Surgery | 0.1 (-3.7 to 3.8) |
| Sevoflurane, extended untreated (all) | 171 | Surgery | -1.6 (-4.2 to 0.8) |

Median fall in MAP from the pre-induction value:

| Report | 1 to 2 minutes | Lowest reading in 10 minutes | Before incision |
|---|---|---|---|
| TIVA, recorded | 18% | 33% | 29% |
| TIVA, model | 17% | 30% | 29% |
| Sevoflurane, recorded | 17% | 29% | 27% |
| Sevoflurane, model | 14% | 31% | 28% |

Untreated HR error was `+0.4` bpm (induction) and `-0.4` bpm (surgery) in TIVA,
and `-4.2` bpm at 5 to 10 minutes and `+0.2` bpm during extended surgery
(`+1.1` bpm open supine) in sevoflurane, where 2 to 5 minute MAP fell `18%` in
the model versus `15%` recorded. In open supine TIVA (`n=91`), the pre-incision
to surgery MAP rise was `+9.8` mmHg in the model versus `+10.1` mmHg recorded,
while laparoscopic/robotic supine cases (`n=28`) rose `+16.9` mmHg due to
unmodeled pneumoperitoneum.

Summary of additional scripts (model minus recorded, or recorded vs model):

| Validation | Cohort | Key outcome |
|---|---|---|
| Sevoflurane BIS (`vitaldb_sevo_bis`) | `n=1,081` maint, `542` emerg | Maintenance bias `-0.2` (`-0.7` to `0.3`), MAE `7.6` (`40.7` vs `39.8`); emergence MAE `10.5` vs `24.5` constant `BIS=40` |
| Cardiac output and SVR (`vitaldb_co`) | `n=66` (`22` paired untreated) | Pre-incision to surgery SVR (`MAP/CO`) `+15.1%` vs `+15.1%` (TIVA `+18.0%` vs `+17.9%`; sevo `+10.0%` vs `+10.3%`) |
| Pulse pressure and CVP (`vitaldb_pulse_pressure`) | `n=512` NIBP, `45` A-line, `7` CVP | Awake NIBP aged `>=70` yr `75.0` vs `75.1` mmHg (`+0.1`); intraop radial A-line `56.0` vs `44.1` mmHg (peripheral amplification); CVP `5.0` vs `3.7` mmHg |
| Vasopressor infusions (`vitaldb_vasopressors`) | `n=11` (`2` NE, `9` PE) | Modeled MAP gain over zero-pressor counterfactual: NE `+9.8` mmHg (HR `-0.3` bpm); PE `+11.7` mmHg (HR `-0.7` bpm) |
| Laparoscopy vs open (`vitaldb_laparoscopy`) | `n=399` (`277` open, `122` lap) | Net lap minus open: compliance `-18.7%` (`-25.2%` total), VCO₂ proxy `+32.6%`, EtCO₂ `+4.3` mmHg, untreated MAP `+8.2` mmHg |
| Intraoperative ABGs (`vitaldb_labs`) | `n=94` ABGs | `PaO2` err `-17.2` (`-34.6` to `1.3`; `+7.5` at PEEP `>=4` vs `-45.1` at PEEP `<=2`); `PaCO2 - EtCO2` err `-0.1` open, widened `+2.3` mmHg in unmodeled lap (`7.0` vs `5.1`) |
| Core temperature (`vitaldb_temperature`) | `n=360` | 60-min `dT` `-0.24 °C` (`-0.27` to `-0.21`) vs `-0.31 °C` warmed and `-1.25 °C` unwarmed; BMI `<22` / `22-26.9` / `>=27`: recorded `-0.27`, `-0.24`, `-0.19 °C` vs warmed model `-0.40`, `-0.30`, `-0.14 °C` |
