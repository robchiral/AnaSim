# MOVER external validation

These scripts evaluate frozen AnaSim parameters out of sample against the EPIC
portion of the Medical Informatics Operating Room Vitals and Events Repository
(MOVER; Samad et al., *JAMIA Open*, 2023, PMID 37860605, DOI
10.1093/jamiaopen/ooad084; UCI Health, 2017 to 2022). No AnaSim parameters,
offsets, or time shifts are fitted to MOVER.

## Data layout and flowsheet subsetting

Raw UCI MOVER EPIC downloads contain the `EMR/` CSV tables and 19 large
`flowsheet_part*.csv` measurement files rather than per-category intraoperative
subsets. To keep repeated queries fast, the validation scripts prefer
pre-subsetted files under `<mover-dir>/Flowsheets/Subset/<Category>/intraop.csv`
(created from raw `patient_information.csv` and `flowsheet_part*.csv`), and
fall back to scanning raw `flowsheet_part*.csv` directly when `Subset/` is
absent.

Set `MOVER_DATA_DIR` (or pass `--mover-dir PATH`) to your local MOVER directory.
Starting from a raw UCI download, regenerate the required intraoperative
flowsheet subsets once with `prepare_flowsheets.py`:

```bash
python -m validation.mover.prepare_flowsheets --mover-dir /path/to/MOVER
```

Pass `--all-splits` to also write `preop.csv`, `postop.csv`, and `unknown.csv`,
or `--categories all` to extract all 15 flowsheet categories instead of the
seven used by these scripts (`BP`, `HR`, `Gas`, `O2`, `Temp`, `Blood`,
`Volume`).

## Running the validation scripts

Running any script builds a compact local Parquet index under
`validation/mover/.mover_cache/` (gitignored) and writes deidentified aggregate
reports to `results/`.

```bash
python -m validation.mover.mover_data
python -m validation.mover.mover_induction
python -m validation.mover.mover_vasopressors
python -m validation.mover.mover_sevo_uptake
python -m validation.mover.mover_temperature
python -m validation.mover.mover_pulse_pressure_labs
```

All scripts screen unique patients (`MRN`, first eligible general anesthesia
encounter) inside AnaSim's supported adult domain (age 18 to 85 years, weight
50 to 200 kg, height 150 to 200 cm, BMI 18 to 60 kg/m²). Drug and ABG replays
pass available preoperative hemoglobin, Cockcroft-Gault creatinine clearance,
glucose, and type 2 diabetes history into `Patient`, and the five analysis
scripts accept `--workers N` and `--seed`.
Errors are model minus recorded with patient-level percentile bootstrap 95% CIs.

## Scripts

- `prepare_flowsheets.py`: builds `patient_timings.csv` (`START_TIME =
  min(IN_OR_DTTM, AN_START_DATETIME)`, `END_TIME = max(OUT_OR_DTTM,
  AN_STOP_DATETIME)`) and splits raw `flowsheet_part*.csv` files into
  `Flowsheets/Subset/<Category>/intraop.csv` via DuckDB.
- `mover_data.py`: builds `.mover_cache/cohort.parquet` (`25,082` unique-patient
  general anesthesia encounters) and `.mover_cache/meds.parquet` (`441,480`
  normalized, deduplicated IV medication events filtered by `MED_ACTION_TIME`
  within `[-90 min, +15 min]` of anesthesia start/stop rather than MAR order
  `RECORD_TYPE`).
- `mover_induction.py`: replays exact MAR bolus and infusion histories
  (including holding-area midazolam and fentanyl at recorded MAR timestamps),
  anchors pre-induction baseline MAP and HR at the pre-induction cuff timestamp,
  and replays `Induction` and `Intubation` events and 1-minute `SEV In` across
  propofol plus sevoflurane, propofol-remifentanil TIVA, and etomidate
  inductions.
- `mover_vasopressors.py`: evaluates isolated maintenance boluses under 1-minute
  invasive arterial MAP (`MAP-ART A-line`) for phenylephrine (`50/100/200 mcg`),
  ephedrine (`5/10 mg`, first vs repeat dose), norepinephrine infusion starts,
  esmolol, and labetalol, plus 1-minute ECG HR across all general anesthesia
  cases for neostigmine plus glycopyrrolate (excluding active volatile washout)
  versus glycopyrrolate alone.
- `mover_sevo_uptake.py`: replays 1-minute `SEV In` against `SEV ET` through
  `VolatilePK` across BMI strata from under 25 to 35+ kg/m² (assuming standard
  ventilation `VT = 8 mL/kg` PBW, `RR = 12/min`, since TV and RR are absent
  from the flowsheet subset).
- `mover_temperature.py`: mirrors `vitaldb_temperature.py` on intraoperative
  core temperature probes (°F converted to °C) across 30-, 60-, and 120-minute
  post-equilibration windows and BMI strata up to 40+ kg/m².
- `mover_pulse_pressure_labs.py`: mirrors `vitaldb_pulse_pressure.py` (awake
  NIBP and anesthetized MAP-matched arterial-line pulse pressure) and
  `vitaldb_labs.py` (settled intraoperative ABGs across FiO₂, age, and BMI,
  plus acid-base, ionized calcium, hemodilution Hb, and glucose).

## Current frozen-parameter results

| Validation | Cohort | Key outcome (model minus recorded, or rec / mod) |
|---|---|---|
| Induction, etomidate (`mover_induction`) | `n=35` (ASA 1-4) | Stop-rule MAP err `-2.3` (`-6.7` to `2.4`) mmHg; pre-ready `-1.3` mmHg; lowest dMAP `-7%` vs `-8%`; post-intubation peak `+19%` vs `+20%` (`-1.0` mmHg) |
| Induction, propofol + sevoflurane (`mover_induction`) | `n=434` (`332` pressor-free) | Stop-rule MAP err `-8.7` (`-10.3` to `-7.2`) mmHg; `1-2 min` `-4.9`, `2-5 min` `-5.8` (`-7.5` to `-4.1`), pre-ready `-7.9`, `5-10 min` `-14.0` mmHg |
| Induction, TIVA propofol + remifentanil (`mover_induction`) | `n=37` | Stop-rule MAP err `-13.5` (`-19.1` to `-6.7`) mmHg; `1-2 min` `-7.2`, `2-5 min` `-9.1`, `5-10 min` `-21.8` mmHg |
| Phenylephrine bolus (`mover_vasopressors`) | `n=180` A-line | Peak dMAP `+13.0` / `+16.2` mmHg (err `+2.0`, `-0.0` to `3.9`); peak dHR `-4.5` / `-2.9` bpm; `+6-8 min` dMAP `+5.5` / `+5.1` mmHg |
| Ephedrine bolus (`mover_vasopressors`) | `n=180` A-line | All `+12.0` / `+12.3` mmHg (err `-0.1`, `-2.4` to `2.2`), dHR `+5.0` / `+5.0` bpm, `+6-8 min` `+8.8` / `+8.5` mmHg; `5 mg` first (`n=55`) `+10.0` / `+11.2`; `10 mg` first (`n=39`) `+19.0` / `+19.7` mmHg |
| Norepinephrine infusion start (`mover_vasopressors`) | `n=22` A-line (`3.0 mcg/min`) | Peak dMAP `+5.5` / `+10.5` mmHg; dHR `+0.0` / `-0.2` bpm |
| Esmolol bolus (`mover_vasopressors`) | `n=32` A-line (`17.5 mg`) | Peak dMAP `-5.0` / `-3.6` mmHg (err `+3.8`, `-0.9` to `7.8`); peak dHR `-15.0` / `-3.2` bpm; late dHR `-11.8` / `-1.8` bpm |
| Labetalol bolus (`mover_vasopressors`) | `n=22` A-line (`5 mg`) | Peak dMAP `-6.0` / `-13.3` mmHg (err `-3.4`, `-7.9` to `1.6`); dHR `-4.5` / `-7.0` bpm |
| Glycopyrrolate alone vs neo + glyco (`mover_vasopressors`) | `n=180` each | Glyco `0.2 mg` alone early dHR `+1.7` / `+2.7` and late `+8.8` / `+5.3` bpm (dMAP `+1.2` / `+1.2`); neo `3.4 mg` + glyco late dHR `+5.1` / `-4.5` bpm |
| Sevoflurane uptake (`mover_sevo_uptake`) | `n=235` (BMI `<25` to `>=35`) | Maintenance `EtSevo` `1.80%` / `1.88%`, bias `+0.11%` (MAE `0.12%`, uniform across BMI); uptake `-0.04%`; washout `+0.01%` |
| Core temperature (`mover_temperature`) | `n=613` (BMI `<22` to `>=40`) | `d60` `-0.14 °C` (`-0.18` to `-0.10`) vs `-0.34 °C` warmed and `-1.23 °C` unwarmed; BMI gradient `-0.25` (`<22`) to `-0.04 °C` (`>=40`) |
| Awake NIBP pulse pressure (`mover_pulse_pressure_labs`) | `n=1,500` | `58.0` / `55.3` mmHg overall (`-3.9`); `60-69` yr `62.0` / `62.8` (`-1.6`); `>=70` yr `78.0` / `76.1` (`-0.6`, `-3.0` to `1.7`) |
| Intraop A-line pulse pressure (`mover_pulse_pressure_labs`) | `n=500` | `53.0` / `51.6` mmHg overall (`-3.1`); `>=60` yr `57.0` / `56.4` (`-1.7`) |
| Intraop ABGs (`mover_pulse_pressure_labs`) | `n=400` (`46%` FiO₂) | `PaO2` `180` / `176` mmHg (`>=70` yr `167` / `148`, err `-6.0`; BMI `<25` `212` / `192`; `30-34.9` `141` / `145`; `>=35` `116` / `125`); `HCO3-` `24.0` / `23.6`; `iCa2+` `1.14` / `1.19` |
