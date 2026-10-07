# AnaSim

AnaSim is an adult anesthesia simulator for education and research. It runs
published pharmacology and cardiorespiratory models in real time, with a patient
monitor and anesthesia machine interface.

[![CI](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml/badge.svg)](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/anasim-simulator.svg)](https://pypi.org/project/anasim-simulator/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/robchiral/AnaSim/blob/main/LICENSE)

[![Run AnaSim in your browser](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/run_anasim.svg)](https://robche.com/AnaSim/)

![AnaSim patient monitor and controls during guided induction](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/anasim_demo.gif)

> [!WARNING]
> AnaSim is for education and research. Do not use it to guide clinical care.

## Run AnaSim

Open [robche.com/AnaSim](https://robche.com/AnaSim/) in your browser.

For local use, install with Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install anasim-simulator
anasim
```

In Windows Command Prompt, use `python` in place of `python3` and activate with
`.venv\Scripts\activate.bat`.
`anasim` opens the interface in your browser and works offline after installation.
Press Ctrl+C in the terminal to stop the server.

## Start a session

For a first session, choose Guided scenario and the total intravenous anesthesia
(TIVA) induction scenario. Choose Start simulation, then Start below the monitor.
For open practice, choose Open simulation and either Awake before induction or
Anesthetized maintenance.

Manual infusion rates are the default. To use target-controlled infusion (TCI),
select Allow TCI during setup. Concentration targets, maintenance starts, and
guided dosing instructions follow this setting. Propofol and remifentanil
infusion rates use mcg/kg/min.

Use Apply or Enter to apply settings and Escape to cancel edits. For mechanical
ventilation, select a facemask or tracheal tube, apply a mode, and choose
Start ventilator.

Choose Record CSV to record a session. Local recordings go to `recordings/` in
the directory where you launched `anasim`. In the hosted app, choose Stop
recording to download the file.

## Features

| Feature | Supported controls and measurements |
|---------|-------------------------------------|
| Anesthesia | Propofol, remifentanil, fentanyl, midazolam, etomidate, ketamine, lidocaine, sevoflurane, nitrous oxide, rocuronium, and sugammadex |
| Cardiovascular drugs | Norepinephrine, epinephrine, phenylephrine, vasopressin, dobutamine, milrinone, esmolol, labetalol, and glycopyrrolate |
| Anesthesia machine | Facemask or tracheal tube, fresh gas flow, vaporizer, and bag-mask ventilation |
| Ventilation | Volume control (VCV), pressure control (PCV), pressure control with volume guarantee (PCV-VG), synchronized intermittent mandatory ventilation (SIMV), pressure support (PSV), and continuous positive airway pressure (CPAP) |
| Monitoring | ECG, pulse oximetry, arterial and noninvasive blood pressure, capnography, ventilation waveforms and loops, gas concentrations, bispectral index (BIS), train-of-four (TOF), temperature, fluid balance, and alarms |
| Events | Fluids and blood, surgical stimulation, airway obstruction, bronchospasm, laryngospasm, hemorrhage, anaphylaxis, sepsis, and arrhythmias |
| Guided scenarios | TIVA and balanced induction, TIVA and inhalational emergence, hemorrhage, anaphylaxis, septic shock, and oxygen supply failure |

Drug delivery supports boluses, manual infusions, and plasma or effect-site TCI,
depending on the drug.

## Limits

AnaSim supports adults aged 18 to 70 years within the
[input limits](https://github.com/robchiral/AnaSim/blob/main/docs/CLI_USAGE.md#fields).
The [architecture guide](https://github.com/robchiral/AnaSim/blob/main/docs/ARCHITECTURE.md#supported-patient-domain)
describes source populations, simulator calibrations, and model limits.
Acid-base balance, lactate, tissue oxygen debt, machine pneumatics, and
cardiopulmonary resuscitation are not modeled.

## Headless use

Run a 60-second simulation and save a CSV recording.

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
