# AnaSim

Real-time anesthesia and physiology simulation for medical education.

[![CI](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml/badge.svg)](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/robchiral/AnaSim/blob/main/LICENSE)

![AnaSim Induction Window](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/induction_full_window.png)

> [!WARNING]
> **For educational use only**
>
> AnaSim is not a medical device. Do not use it for clinical care or patient-specific
> prediction.

## Installation

AnaSim requires Python 3.10 or newer. Install the published package in a virtual
environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install anasim-simulator
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The package is distributed as `anasim-simulator`; the command and Python import
are `anasim`.

## Quick start

Launch the interactive operating-room monitor:

```bash
anasim
```

Run a headless simulation:

```bash
anasim --mode headless --duration 10
```

See the [CLI guide](https://github.com/robchiral/AnaSim/blob/main/docs/CLI_USAGE.md)
for configuration files, recording, and additional examples.

## Features

- Cardiovascular and respiratory physiology responding to drugs, ventilation,
  fluids, and surgical stimulation
- PK/PD models for propofol, remifentanil, sevoflurane, rocuronium, and
  vasoactive agents
- VCV, PCV, PSV, CPAP, and manual bag-mask ventilation
- ECG, pulse oximetry, capnography, NIBP, and depth-of-anesthesia monitoring
- Guided induction, hemorrhage, anaphylaxis, septic shock, and emergence scenarios
- Headless execution and CSV recording for reproducible teaching exercises

## Model scope and calibration

AnaSim uses published models where they behave coherently in the integrated
simulation and documents calibrated deviations where literal parameters produce
implausible managed intraoperative states. See the
[model references and deviations](https://github.com/robchiral/AnaSim/blob/main/docs/REFERENCES.md)
for citations and details.

Examples of calibration choices and deliberate deviations:

- hemodynamics uses plasma propofol/remifentanil concentrations in line with Su et al. 2023, while CNS depth, tolerance, BIS, and respiratory depression use effect-site concentrations
- rocuronium spontaneous recovery uses a faster runtime `ke0` than pure literature modeling so recovery timing stays in the clinical range
- maintenance startup adds visible norepinephrine support only when the selected anesthetic state would otherwise begin below MAP 65 mmHg

AnaSim currently accepts adult patients aged 18–70 years. The upper bound avoids unsupported extrapolation of the strongly age-dependent Su et al. hemodynamic term beyond the population used to develop and illustrate that model.

## Development

```bash
git clone https://github.com/robchiral/AnaSim.git
cd AnaSim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

See the [contribution guide](https://github.com/robchiral/AnaSim/blob/main/CONTRIBUTING.md)
and [changelog](https://github.com/robchiral/AnaSim/blob/main/CHANGELOG.md).

Initial TIVA implementations were derived from
[Python Anesthesia Simulator](https://github.com/AnesthesiaSimulation/Python_Anesthesia_Simulator).

## License

AnaSim is released under the
[MIT License](https://github.com/robchiral/AnaSim/blob/main/LICENSE).
