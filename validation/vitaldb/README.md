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
python -m validation.vitaldb.vitaldb_tiva
python -m validation.vitaldb.vitaldb_sevo
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
circuit resistance, airway sensor lag, capnography, and pre-incision supine
compliance with 5-fold patient-grouped cross-validation and full-cohort fits.
Compliance includes all eligible supine cases by default; pass `--cases 0` or
`--pcv-cases 0` to remove waveform caps, `--folds` to change fold count, or
`--figure validation/vitaldb/results/overlay.png` (requires Matplotlib) for
breath overlays.

`vitaldb_peep.py` audits pre-incision PEEP steps and writes
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
minutes after propofol.

Both scripts fit a patient-offset likelihood model with patient bootstrap 95%
intervals, report 5-fold cross-validation stability, and accept `--override '{...}'`
or `--cv-grid '{"field": [v1, v2]}'` (`--folds 5`) for cross-validated sweeps.

### Current results

MAP error, in mmHg with 95% CI:

| Report | Patients | Window | MAP error |
|---|---:|---|---|
| TIVA (all) | 511 | Propofol to incision | -3.4 (-4.4 to -2.6) |
| TIVA (legacy core) | 330 | Propofol to incision | -3.1 (-4.1 to -2.2) |
| TIVA, untreated | 182 | Surgery | -3.5 (-5.2 to -1.8) |
| Sevoflurane (all) | 136 | Propofol to incision | -7.0 (-9.1 to -4.9) |
| Sevoflurane (legacy core) | 29 | Propofol to incision | -6.3 (-10.3 to -2.3) |
| Sevoflurane, untreated | 48 | Surgery | -4.0 (-8.9 to 0.6) |

Median fall in MAP from the pre-induction value:

| Report | 1 to 2 minutes | Lowest reading in 10 minutes | Before incision |
|---|---|---|---|
| TIVA, recorded | 18% | 33% | 29% |
| TIVA, model | 19% | 30% | 29% |
| Sevoflurane, recorded | 17% | 29% | 27% |
| Sevoflurane, model | 14% | 32% | 29% |

Untreated HR error was `+0.1` bpm (induction) and `-0.1` bpm (surgery) in TIVA,
and `-5.4` bpm at 5 to 10 minutes and `-2.7` bpm during surgery in sevoflurane,
where 2 to 5 minute MAP fell `18%` in the model versus `15%` recorded.
