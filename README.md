# AnaSim

AnaSim is a real-time adult anesthesia simulator for teaching and model
exploration. It combines published pharmacokinetic, pharmacodynamic, and
cardiorespiratory models with a patient monitor and anesthesia machine.

[![CI](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml/badge.svg)](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/anasim-simulator.svg)](https://pypi.org/project/anasim-simulator/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/robchiral/AnaSim/blob/main/LICENSE)

![AnaSim running the guided TIVA induction](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/anasim_demo.gif)

> [!WARNING]
> AnaSim is for education and research. Do not use it to guide clinical care.

## Run AnaSim

**In the browser:** open [robche.com/AnaSim](https://robche.com/AnaSim/). It
runs on your device through [Pyodide](https://pyodide.org). The first visit
downloads about 23 MB.

**Locally** (Python 3.10 or later):

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install anasim-simulator
anasim
```

`anasim` opens the simulator in your browser and runs it in Python on your
computer. It works offline. Stop it with Ctrl+C.

To begin, choose **Guided scenario** and **TIVA induction**, then **Start
simulation**. The clock runs once you press **Start simulation** again below
the monitor. Each objective has a button that opens the controls it needs.

## What's included

- **Drugs:** propofol, remifentanil, sevoflurane, rocuronium, sugammadex,
  norepinephrine, epinephrine, phenylephrine, vasopressin, dobutamine, and
  milrinone, given by infusion, bolus, or effect-site TCI.
- **Machine:** facemask or tracheal tube, fresh gas flow, vaporizer, bag-mask
  ventilation, and VCV, PCV, PSV, or CPAP.
- **Monitor:** ECG, SpO₂, arterial line, NIBP, capnography, BIS, TOF,
  temperature, fluid balance, and alarms.
- **Events:** fluids and blood, surgical stimulation, airway obstruction,
  bronchospasm, laryngospasm, hemorrhage, anaphylaxis, sepsis, and arrhythmias.
- **Guided scenarios:** TIVA and inhalational induction and emergence,
  hemorrhage, anaphylaxis, septic shock, and oxygen supply failure.

**Record CSV** saves the simulation as a time series. The local version writes
it to `recordings/` in the directory where you ran `anasim`; the browser
version downloads it.

## Limits

Patients must be adults aged 18 to 70 years, weighing 50 to 100 kg, 150 to
200 cm tall, with a BMI of 18 to 32 kg/m². These are the ranges of the
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

This runs 60 simulated seconds without the interface and writes a CSV to
`recordings/`. Use `--config` to set the patient, models, and starting state
from a JSON file; the
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
