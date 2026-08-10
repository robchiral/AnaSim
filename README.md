# AnaSim

AnaSim is an interactive adult anesthesia and physiology simulator for teaching
and model exploration.

[![CI](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml/badge.svg)](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/anasim-simulator.svg)](https://pypi.org/project/anasim-simulator/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/robchiral/AnaSim/blob/main/LICENSE)

![AnaSim guided induction demo](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/anasim_demo.gif)

> [!WARNING]
> **Education and research use**
>
> AnaSim is simulation software, not a medical device. Do not use its output to
> guide clinical care.

## What AnaSim models

AnaSim simulates drug delivery, pharmacokinetics, pharmacodynamics,
cardiorespiratory physiology, ventilation, fluids, temperature, and common
perioperative events in real time. The desktop interface uses standard patient
monitor and anesthesia machine conventions. Headless mode supports scripted
runs and CSV recording.

AnaSim includes propofol, remifentanil, sevoflurane, rocuronium, sugammadex,
norepinephrine, epinephrine, phenylephrine, vasopressin, dobutamine, and
milrinone. Ventilation modes include VCV, PCV, PSV, CPAP, and bag-mask
ventilation.

## Install and start

AnaSim requires Python 3.10 or later. Install it in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install anasim-simulator
anasim
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The package name is `anasim-simulator`. The command and Python module are both
named `anasim`.

## First session

Run the guided TIVA induction to become familiar with the interface:

1. Launch `anasim`.
2. Select **Guided scenario** and **Induction (TIVA)**.
3. Confirm the default patient and select **Start simulation**.
4. Use **Open machine**, **Open medications**, and **Open events** in the
   objective panel to open each control area.
5. Start the clock when you are ready to observe the physiologic response.

## Interface

| Area | Purpose |
|------|---------|
| Monitor | Waveforms, displayed vital signs, anesthetic gas, temperature, neuromuscular function, and fluid balance |
| Machine | Airway connection, fresh gas flow, vaporizer, manual ventilation, and mechanical ventilation |
| Medications | Manual infusions, effect-site TCI, boluses, vasoactive agents, and reversal |
| Events and fluids | Fluids, blood products, surgical stimulation, airway events, hemorrhage, anaphylaxis, and sepsis |

Guided scenarios cover TIVA and inhalational induction, emergence, hemorrhage,
anaphylaxis, septic shock, and oxygen supply failure. Open simulation mode
provides direct control of the same environment.

## Model scope and limits

AnaSim accepts these patient inputs:

| Input | Supported range |
|-------|-----------------|
| Age | 18 to 70 years |
| Weight | 50 to 100 kg |
| Height | 150 to 200 cm |
| BMI derived from weight and height | 18 to 32 kg/m² |
| Hemoglobin | 6 to 20 g/dL |
| Hematocrit | 0.18 to 0.60 |
| Renal function factor | 0.4 to 1.0 |
| Hepatic function factor | 0.5 to 1.0 |

The body-size limits enclose the observed ranges in the healthy-adult cohort used
by the Su hemodynamic and Li norepinephrine models. Weight and height must also
produce a BMI within the supported range. Renal and hepatic factors are
dimensionless model inputs.

AnaSim combines published component models with simulator-specific models for
respiratory drug interaction, neuromuscular block and reversal, vasoactive drug
response, and arterial pressure display. See
[model references](https://github.com/robchiral/AnaSim/blob/main/docs/REFERENCES.md)
for sources and implementation details.

Display values include simulated monitor response and may differ from the
underlying physiologic state. Acid-base balance, lactate, tissue oxygen debt,
and complete anesthesia machine pneumatics are outside the current model scope.

## Headless use

Run a reproducible ten-second simulation:

```bash
anasim --mode headless --duration 10 --config patient.json --record
```

Configuration files set patient characteristics, model choices, initial state,
random seed, and runtime options. See the
[CLI guide](https://github.com/robchiral/AnaSim/blob/main/docs/CLI_USAGE.md) for
the available fields.

## Documentation

- [CLI usage](https://github.com/robchiral/AnaSim/blob/main/docs/CLI_USAGE.md)
- [Model references](https://github.com/robchiral/AnaSim/blob/main/docs/REFERENCES.md)
- [Architecture](https://github.com/robchiral/AnaSim/blob/main/docs/ARCHITECTURE.md)
- [Contribution guide](https://github.com/robchiral/AnaSim/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/robchiral/AnaSim/blob/main/CHANGELOG.md)

## Development

```bash
git clone https://github.com/robchiral/AnaSim.git
cd AnaSim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Regenerate the animated demo with `python scripts/capture_demo.py`.

Initial TIVA implementations were derived from
[Python Anesthesia Simulator](https://github.com/AnesthesiaSimulation/Python_Anesthesia_Simulator).

## Citation and license

For teaching or research use, cite the software and release described in
[`CITATION.cff`](https://github.com/robchiral/AnaSim/blob/main/CITATION.cff).
AnaSim is available under the
[MIT License](https://github.com/robchiral/AnaSim/blob/main/LICENSE).
