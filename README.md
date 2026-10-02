# AnaSim

AnaSim is an adult anesthesia simulator for teaching and research. It runs
published drug and cardiorespiratory models in real time, with a patient
monitor and anesthesia machine interface.

[![CI](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml/badge.svg)](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/anasim-simulator.svg)](https://pypi.org/project/anasim-simulator/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/robchiral/AnaSim/blob/main/LICENSE)

![AnaSim running the guided TIVA induction](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/anasim_demo.gif)

> [!WARNING]
> AnaSim is for education and research. Do not use it to guide clinical care.

## Run AnaSim

Open [robche.com/AnaSim](https://robche.com/AnaSim/) in your browser.

To run locally, install with Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install anasim-simulator
anasim
```

`anasim` opens the simulator in your browser and runs it in Python on your
computer. It works offline. Stop it with Ctrl+C.

For a first session, choose **Guided scenario**, select **TIVA induction**,
and press **Start simulation**. Press **Start simulation** below the monitor
to start the clock. Each objective links to the controls needed to complete it.

## Features

- **Drugs:** propofol, remifentanil, fentanyl, midazolam, etomidate, ketamine,
  sevoflurane, rocuronium, sugammadex, norepinephrine, epinephrine,
  phenylephrine, vasopressin, dobutamine, milrinone, esmolol, labetalol, and
  glycopyrrolate. Available delivery methods depend on the drug and include
  infusion, bolus, and effect-site target-controlled infusion (TCI).
- **Machine:** facemask or tracheal tube, fresh gas flow, vaporizer, bag-mask
  ventilation, and VCV, PCV, PCV-VG, SIMV, PSV, or CPAP.
- **Monitor:** ECG, SpO₂, arterial line, NIBP, capnography, airway pressure
  and flow, ventilator pressures, volumes, and loops, O₂, N₂O, and sevoflurane
  gas monitoring, BIS, TOF, temperature, fluid balance, and alarms.
- **Events:** fluids and blood, surgical stimulation, airway obstruction,
  bronchospasm, laryngospasm, hemorrhage, anaphylaxis, sepsis, and arrhythmias.
- **Guided scenarios:** TIVA and inhalational induction and emergence,
  hemorrhage, anaphylaxis, septic shock, and oxygen supply failure.

**Record CSV** saves the simulation as a time series. The local version writes
it to `recordings/` in the directory where you ran `anasim`; the browser
version downloads it.

## Limits

Patients must be adults aged 18 to 70 years, weighing 50 to 100 kg, 150 to
200 cm tall, with a BMI of 18 to 32 kg/m². These limits are based on the
volunteer studies used to build the hemodynamic and norepinephrine models.

Respiratory drug effects, neuromuscular block, the baroreflex, and vasoactive
responses use AnaSim calibrations fitted to published data. The
[model references](https://github.com/robchiral/AnaSim/blob/main/docs/REFERENCES.md)
list the sources.

Displayed values include monitor lag and artifact. Acid-base balance, lactate,
tissue oxygen debt, machine pneumatics, and resuscitation are not modeled.

## Headless use

```bash
anasim --mode headless --duration 60 --record
```

This runs 60 simulated seconds and writes a CSV to `recordings/`. Use
`--config` to load patient, model, and starting-state settings from JSON. The
[CLI guide](https://github.com/robchiral/AnaSim/blob/main/docs/CLI_USAGE.md)
lists the fields.

## Documentation

- [CLI usage](https://github.com/robchiral/AnaSim/blob/main/docs/CLI_USAGE.md)
- [Model references](https://github.com/robchiral/AnaSim/blob/main/docs/REFERENCES.md)
- [Architecture](https://github.com/robchiral/AnaSim/blob/main/docs/ARCHITECTURE.md)
- [Contributing](https://github.com/robchiral/AnaSim/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/robchiral/AnaSim/blob/main/CHANGELOG.md)

## Citation and license

Cite AnaSim using [`CITATION.cff`](https://github.com/robchiral/AnaSim/blob/main/CITATION.cff).
AnaSim is released under the
[MIT License](https://github.com/robchiral/AnaSim/blob/main/LICENSE). The first
TIVA implementation was derived from
[Python Anesthesia Simulator](https://github.com/AnesthesiaSimulation/Python_Anesthesia_Simulator).
