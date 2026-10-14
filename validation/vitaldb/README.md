# VitalDB validation

These scripts compare AnaSim ventilation, gas exchange, propofol/BIS, and
sevoflurane uptake with clinical recordings in VitalDB. Run them from the
repository root after the
[contributor setup](../../CONTRIBUTING.md#setup).

```bash
python validation/vitaldb/vitaldb_ventilation.py
python validation/vitaldb/vitaldb_peep.py
python validation/vitaldb/vitaldb_anesthetics.py --figures
```

The ventilation script downloads missing tracks into `.vitaldb_cache/` and writes
`results/vitaldb_ventilation.md`. It fits parameters on calibration cases and
checks them on held-out cases. Rerun it after mechanics, airway pressure, or
capnography changes.

The PEEP audit uses cached tracks only and writes `results/vitaldb_peep.json`.
Its counts depend on the local cache. It requires cached `cases.csv` and
`trks.csv`; `--cache PATH` selects another cache directory.

The anesthetic script replays recorded propofol delivery through the production
PK/BIS models and inspired sevoflurane through the production uptake model.
It checks input history and signal quality, compares each phase with a simple
baseline, and reports errors per patient. It uses fixed model parameters and
fits no patient-specific offsets or time shifts. Pump-predicted concentrations
are not clinical concentration measurements. The sevoflurane comparison assumes
cardiac output and FRC and includes cardiac-output sensitivity runs.

Treat the anesthetic cohort as evaluation data. Use the primary papers in
[Model references](../../docs/REFERENCES.md#intravenous-drug-models) to resolve
equation or unit defects. Do not fit BIS offsets, age terms, or delays to the
recorded cases. If inspected results guide a model revision, evaluate the revised
model on a separate cohort with eligibility and scoring fixed before inspection.
Published BIS studies often report median absolute prediction error (MDAPE);
this report uses the mean of each patient's mean absolute error (MAE). These
metrics and the studies' signal processing differ, so their values provide
context rather than a direct performance comparison.

The default samples are 100 propofol/remifentanil cases and 40 sevoflurane cases.
`--tiva-cases`, `--sevo-cases`, and `--seed` select sample sizes and candidate
order. `--cache` and `--output` select data and report directories; `--dt 0.5`
allows a finer replay check. Missing tracks download into `.vitaldb_cache/`.
The script writes `results/vitaldb_anesthetics.md`, per-patient CSV errors,
and JSON provenance and exclusions. `--figures` requires Matplotlib and shows
patients across the range of errors. These comparisons support the tested
components; they do not validate the whole simulator, remifentanil analgesia,
consciousness, or vaporizer/circuit delivery.

Reports in `results/` contain the comparison results. Generate a report for the
checkout being evaluated. To create overlays, install Matplotlib and add
`--figure validation/vitaldb/results/overlay.png` to the ventilation command.

Source data are from VitalDB (Lee et al., *Scientific Data*, 2022), under
CC BY-NC-SA 4.0 and its data use agreement.
