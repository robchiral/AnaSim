# AnaSim

AnaSim is an adult anesthesia simulator for teaching and research. It runs
published drug and cardiorespiratory models in real time, with a patient
monitor and anesthesia machine interface.

[![CI](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml/badge.svg)](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/anasim-simulator.svg)](https://pypi.org/project/anasim-simulator/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/robchiral/AnaSim/blob/main/LICENSE)

[![Run AnaSim in your browser](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/run_anasim.svg)](https://robche.com/AnaSim/)

![AnaSim running the guided TIVA induction](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/anasim_demo.gif)

> [!WARNING]
> AnaSim is for education and research. Do not use it to guide clinical care.

## Run AnaSim

Open [robche.com/AnaSim](https://robche.com/AnaSim/) in your browser.

To run locally, install with Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install anasim-simulator
anasim
```

`anasim` opens a local browser session and works offline. Stop the server with Ctrl+C.

## Use a session

For a first session, choose Guided scenario and TIVA induction. For open
practice, choose Open simulation and either Awake before induction or
Anesthetized maintenance. Choose Start simulation, then Start below the monitor.

Manual infusion rates are the default. Select Allow TCI during setup to use
target controls and TCI maintenance starts. Scenario instructions follow this
setting. Propofol and remifentanil rates use mcg/kg/min.

Use Apply or Enter to save settings and Escape to cancel edits. For mechanical
ventilation, select a facemask or tracheal tube, apply a mode, and choose
Start ventilator.

Local CSV recordings go to `recordings/` where you launched `anasim`;
hosted recordings download when stopped.

## Features

- Drugs include propofol, remifentanil, fentanyl, midazolam, etomidate, ketamine,
  lidocaine, sevoflurane, rocuronium, sugammadex, norepinephrine, epinephrine,
  phenylephrine, vasopressin, dobutamine, milrinone, esmolol, labetalol, and
  glycopyrrolate. Delivery includes bolus, infusion, and effect-site
  target-controlled infusion (TCI) where supported.
- Machine options include a facemask or tracheal tube, fresh gas flow, a
  vaporizer, bag-mask ventilation, and VCV, PCV, PCV-VG, SIMV, PSV, or CPAP.
- The monitor shows ECG, SpO₂, arterial pressure, NIBP, capnography,
  ventilation waveforms and loops, gas values, BIS, TOF, temperature,
  fluid balance, and alarms.
- Events include fluids and blood, surgical stimulation, airway obstruction,
  bronchospasm, laryngospasm, hemorrhage, anaphylaxis, sepsis, and arrhythmias.
- Guided scenarios cover TIVA and inhalational induction and emergence,
  hemorrhage, anaphylaxis, septic shock, and oxygen supply failure.

## Limits

AnaSim supports adults aged 18 to 70 years, with [input limits](https://github.com/robchiral/AnaSim/blob/main/docs/CLI_USAGE.md#fields).
[Architecture](https://github.com/robchiral/AnaSim/blob/main/docs/ARCHITECTURE.md#supported-patient-domain)
describes source populations and teaching calibrations. Acid-base balance,
lactate, tissue oxygen debt, machine pneumatics, and CPR are not modeled.

## Headless use

```bash
anasim --mode headless --duration 60 --record
```

## Documentation

| Task | Guide |
|------|-------|
| Configure patients, run scripts, or interpret CSV | [CLI and Python usage](https://github.com/robchiral/AnaSim/blob/main/docs/CLI_USAGE.md) |
| Understand model behavior and assumptions | [Architecture](https://github.com/robchiral/AnaSim/blob/main/docs/ARCHITECTURE.md) |
| Find the source studies | [Model references](https://github.com/robchiral/AnaSim/blob/main/docs/REFERENCES.md) |
| Change the code and choose checks | [Contributing](https://github.com/robchiral/AnaSim/blob/main/CONTRIBUTING.md) |
| Review release changes | [Changelog](https://github.com/robchiral/AnaSim/blob/main/CHANGELOG.md) |

## Citation and license

Cite using [`CITATION.cff`](https://github.com/robchiral/AnaSim/blob/main/CITATION.cff).
Licensed under [MIT](https://github.com/robchiral/AnaSim/blob/main/LICENSE).
The first TIVA implementation was derived from
[Python Anesthesia Simulator](https://github.com/AnesthesiaSimulation/Python_Anesthesia_Simulator).
