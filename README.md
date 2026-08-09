# AnaSim

Interactive adult anesthesia and physiology simulation for teaching and model
exploration.

[![CI](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml/badge.svg)](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/anasim-simulator.svg)](https://pypi.org/project/anasim-simulator/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/robchiral/AnaSim/blob/main/LICENSE)

![AnaSim guided induction demo](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/anasim_demo.gif)

> [!WARNING]
> **For education and research only**
>
> AnaSim is not a medical device or a patient-specific predictor. Do not use it
> to guide clinical care.

## What AnaSim models

AnaSim runs drug delivery, pharmacokinetics, pharmacodynamics, cardiorespiratory
physiology, ventilation, fluids, temperature, and common perioperative
disturbances together in real time. The desktop interface follows familiar
monitor and anesthesia machine conventions. A headless mode runs scripted
experiments and records to CSV.

The current drug set includes propofol, remifentanil, sevoflurane, rocuronium,
sugammadex, norepinephrine, epinephrine, phenylephrine, vasopressin, dobutamine,
and milrinone. Ventilation modes include VCV, PCV, PSV, CPAP, and bag-mask
ventilation.

## Install and start

AnaSim requires Python 3.10 or newer. Install the published package in a virtual
environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install anasim-simulator
anasim
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The distribution name is `anasim-simulator`. The command and Python import are
both `anasim`.

## First session

For a short guided review of the interface:

1. Launch `anasim`.
2. Select **Guided scenario** and **Induction (TIVA)**.
3. Confirm the default patient and select **Start simulation**.
4. Use **Open machine**, **Open medications**, and **Open events** in the
   objective panel to move between control areas.
5. Start the clock when you are ready to observe the physiologic response.

## Interface

| Area | Purpose |
|------|---------|
| Monitor | Waveforms, displayed vital signs, anesthetic gas, temperature, neuromuscular function, and fluid balance |
| Machine | Airway connection, fresh gas flow, vaporizer, manual ventilation, and mechanical ventilation |
| Medications | Manual infusions, effect-site TCI, boluses, vasoactive agents, and reversal |
| Events and fluids | Fluids, blood products, surgical stimulation, airway events, hemorrhage, anaphylaxis, and sepsis |

Guided scenarios cover TIVA and inhalational induction, emergence, hemorrhage,
anaphylaxis, septic shock, and oxygen supply failure. Open simulation mode gives
direct control of the same environment.

## Model scope and limits

AnaSim accepts adult patients aged 18 to 70 years. That is the population behind
the Su et al. hemodynamic model, whose age term is strong, so ages outside the
range would be extrapolation the source does not support.

AnaSim uses published models wherever they still hold together once combined.
Where the literature offers no compatible joint model, it falls back on stated
heuristic parameters and regression ranges, listed in
[references and implementation choices](https://github.com/robchiral/AnaSim/blob/main/docs/REFERENCES.md).

Important limits include:

- The integrated simulator has not been validated for clinical prediction.
- Displayed values pass through modeled monitor behavior, so they can differ
  from the underlying physiologic state.
- The arterial pressure trace uses a dedicated Su-constrained landmark waveform
  and catheter-transducer model. It has not been clinically validated.
- The model does not currently represent acid-base balance, lactate, tissue
  oxygen debt, or a full anesthesia machine pneumatic system.

## Headless use

Run a reproducible ten-second simulation:

```bash
anasim --mode headless --duration 10 --config patient.json --record
```

Configuration files can set patient characteristics, model choices, initial
state, random seed, and runtime options. See the
[CLI guide](https://github.com/robchiral/AnaSim/blob/main/docs/CLI_USAGE.md) for
the full schema and examples.

## Documentation

- [CLI usage](https://github.com/robchiral/AnaSim/blob/main/docs/CLI_USAGE.md)
- [References and implementation choices](https://github.com/robchiral/AnaSim/blob/main/docs/REFERENCES.md)
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

If you use AnaSim in teaching or research, cite the software and the exact
release described in
[`CITATION.cff`](https://github.com/robchiral/AnaSim/blob/main/CITATION.cff).
AnaSim is available under the
[MIT License](https://github.com/robchiral/AnaSim/blob/main/LICENSE).
