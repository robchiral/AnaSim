# VitalDB validation

These scripts compare AnaSim ventilation, capnography, and gas exchange with
Dräger Primus recordings in VitalDB. Run them from the repository root after the
[contributor setup](../../CONTRIBUTING.md#setup).

```bash
python validation/vitaldb/vitaldb_ventilation.py
python validation/vitaldb/vitaldb_peep.py
```

The ventilation script downloads missing tracks into `.vitaldb_cache/` and writes
`results/vitaldb_ventilation.md`. It fits parameters on calibration cases and
checks them on held-out cases. Rerun it after mechanics, airway pressure, or
capnography changes.

The PEEP audit uses cached tracks only and writes `results/vitaldb_peep.json`.
Its counts depend on the local cache. It requires cached `cases.csv` and
`trks.csv`; `--cache PATH` selects another cache directory.

Reports in `results/` are saved snapshots and may predate current model
changes. To create overlays, install Matplotlib and add
`--figure validation/vitaldb/results/overlay.png` to the ventilation command.

Source data are from VitalDB (Lee et al., *Scientific Data*, 2022), under
CC BY-NC-SA 4.0 and its data use agreement.
