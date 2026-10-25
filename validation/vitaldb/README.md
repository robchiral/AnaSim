# VitalDB validation

These scripts compare AnaSim's ventilation, gas exchange, propofol and BIS,
sevoflurane uptake, and hemodynamics with patient recordings from VitalDB. Run
them from the repository root after the [contributor
setup](../../CONTRIBUTING.md#setup). Tracks that are not yet cached are
downloaded into `validation/vitaldb/.vitaldb_cache/`.

```bash
python validation/vitaldb/vitaldb_ventilation.py
python validation/vitaldb/vitaldb_peep.py
python validation/vitaldb/vitaldb_anesthetics.py --figures
python validation/vitaldb/vitaldb_tiva.py
python validation/vitaldb/vitaldb_sevo.py
```

The ventilation script writes `results/vitaldb_ventilation.md`. It fits
parameters on one set of cases and checks them on another. Rerun it after
changing respiratory mechanics, airway pressure, or capnography.

The PEEP audit reads only cached tracks, so its counts depend on what has been
downloaded. It needs cached copies of `cases.csv` and `trks.csv`, and writes
`results/vitaldb_peep.json`. Use `--cache PATH` to read from a different cache
directory.

The anesthetic script passes recorded propofol delivery through the PK and BIS
models, and recorded inspired sevoflurane through the uptake model, without
changing any model parameters. It checks each case's input history and signal
quality, compares each phase of the anesthetic with a simple baseline, and
reports errors for each patient. The sevoflurane comparison assumes values for
cardiac output and FRC and includes runs that vary cardiac output. If the
comparison shows an equation or unit error, correct it from the primary papers
listed in [Model references](../../docs/REFERENCES.md#intravenous-drug-models).
The report gives each patient's mean absolute BIS error. Published studies
usually report the median absolute prediction error (MDAPE) and process the BIS
signal differently, so their numbers cannot be compared directly with these.

By default the script samples 100 propofol-remifentanil cases and 40
sevoflurane cases. `--tiva-cases`, `--sevo-cases`, and `--seed` change the
sample sizes and the order of candidates. `--cache` and `--output` change the
data and report directories, and `--dt 0.5` runs the replay with a smaller time
step. The script writes `results/vitaldb_anesthetics.md`, a CSV of
per-patient errors, and a JSON file with provenance and exclusions. `--figures`
plots patients across the range of errors and requires Matplotlib.

## Hemodynamics

The two hemodynamic scripts run the full engine with each patient's recorded
anesthetic delivery, fluids, and blood loss. Each patient starts from their own
pre-induction cuff MAP and HR, and intubation and surgical stimulation are added
at the corresponding times. Patient positioning and pneumoperitoneum are not
included. All errors are the model value minus the recorded value.

`vitaldb_tiva.py` replays every eligible propofol-remifentanil case in general
surgery, including patients who received ephedrine or phenylephrine. From the
start of propofol to incision, each cuff reading is compared with the model's
average MAP over the 40 seconds before the reading appeared, which is roughly
how long the cuff takes to measure. Each patient's readings are used only up to
the first one that might have led to treatment (MAP below 65 mmHg, SBP below 90,
or HR below 45), so any treatment given afterward does not affect the result.
The mean error is estimated with a model that allows each patient their own
offset, and confidence intervals come from resampling patients. The report
breaks the error down by treatment, type of surgery, and age, compares the
recorded and model course of MAP, and scores MAP and HR during surgery in
untreated patients.

`vitaldb_sevo.py` replays induction as it is usually taught in the United
States: a propofol bolus with fentanyl and sevoflurane. Propofol is timed from
the recorded fall in BIS, 30 s later than the model's BIS latency suggests,
because model BIS falls about that much later than recorded BIS in the TIVA
cases. Intubation is placed at 2.5 minutes, shortly before the ventilator
settings first appear. It is scored in the same way and also reports MAP and HR
during surgery.

Current results (mmHg, 95% CI):

| Report | Patients | Window | MAP error |
|---|---:|---|---|
| TIVA | 301 | Propofol to incision | -1.7 (-2.8 to -0.6) |
| TIVA, untreated | 114 | Surgery | -1.3 (-3.6 to 0.8) |
| Sevoflurane | 29 | Propofol to incision | -5.3 (-9.5 to -1.0) |
| Sevoflurane, untreated | 12 | Surgery | 6.1 (-1.6 to 13.7) |

In the TIVA cases, recorded MAP fell by a median of 17% at 1 to 2 minutes after
propofol, 31% at the lowest reading in the first 10 minutes, and 27% before
incision. The corresponding model values are 15%, 30%, and 28%. In the
sevoflurane cases, recorded MAP fell by 16% at 1 to 2 minutes, 29% at the
lowest reading, and 27% before incision, compared with 14%, 32%, and 29% in the
model, and model HR was within 3 bpm of the recordings. At 2 to 5 minutes the
recorded fall was smaller than the model's (13% compared with 19%).

## Reports and source

Each script writes its report to `results/`. To plot ventilation overlays,
install Matplotlib and add `--figure validation/vitaldb/results/overlay.png` to
the ventilation command.

The source data come from VitalDB (Lee et al., *Scientific Data*, 2022) and are
used under CC BY-NC-SA 4.0 and the VitalDB data use agreement.
