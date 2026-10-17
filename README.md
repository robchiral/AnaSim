# AnaSim

AnaSim simulates adult anesthesia for education and research. It combines
published pharmacology and cardiorespiratory models with a patient monitor and
anesthesia machine interface.

[![CI](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml/badge.svg)](https://github.com/robchiral/AnaSim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/anasim-simulator.svg)](https://pypi.org/project/anasim-simulator/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Noncommercial](https://img.shields.io/badge/license-noncommercial-blue.svg)](https://github.com/robchiral/AnaSim/blob/main/LICENSE)

[![Run AnaSim in your browser](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/run_anasim.svg)](https://robche.com/AnaSim/)

![AnaSim patient monitor and controls during guided induction](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/anasim_demo.gif)

> [!WARNING]
> Do not use AnaSim to guide clinical care.

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

Choose a guided scenario or open simulation, then select Start simulation and
Start. Use Apply to confirm setting changes.

## Features

| Feature | Supported controls and measurements |
|---------|-------------------------------------|
| Anesthesia | Propofol, remifentanil, fentanyl, hydromorphone, morphine, midazolam, etomidate, ketamine, lidocaine, sevoflurane, nitrous oxide, rocuronium, succinylcholine, and sugammadex |
| Cardiovascular drugs | Norepinephrine, epinephrine, phenylephrine, ephedrine, vasopressin, dobutamine, milrinone, esmolol, labetalol, atropine, and glycopyrrolate |
| Anesthesia machine | Facemask or tracheal tube, fresh gas flow, vaporizer, and bag-mask ventilation |
| Ventilation | Volume control (VCV), pressure control (PCV), pressure control with volume guarantee (PCV-VG), synchronized intermittent mandatory ventilation (SIMV), pressure support (PSV), and continuous positive airway pressure (CPAP) |
| Monitoring | ECG, pulse oximetry, arterial and noninvasive blood pressure, capnography, ventilation waveforms and loops, gas concentrations, bispectral index (BIS), train-of-four (TOF) and single-twitch monitoring, temperature, fluid balance, trends, and alarms |
| Fluids and events | Fluids and blood, surgical stimulation, airway obstruction, bronchospasm, laryngospasm, hemorrhage, anaphylaxis, sepsis, and arrhythmias |
| Guided scenarios | Total intravenous anesthesia (TIVA) and balanced induction, TIVA and inhalational emergence, hemorrhage, anaphylaxis, septic shock, and oxygen supply failure |

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
| Compare model outputs with clinical recordings | [VitalDB validation](https://github.com/robchiral/AnaSim/blob/main/validation/vitaldb/README.md) |
| Change the code and choose checks | [Contributing](https://github.com/robchiral/AnaSim/blob/main/CONTRIBUTING.md) |
| Review release changes | [Changelog](https://github.com/robchiral/AnaSim/blob/main/CHANGELOG.md) |

## Citation and license

Cite using [`CITATION.cff`](https://github.com/robchiral/AnaSim/blob/main/CITATION.cff).

AnaSim 1.6 and later use the proprietary, source-available
[AnaSim noncommercial license](https://github.com/robchiral/AnaSim/blob/main/LICENSE).

| Use | Terms |
|-----|-------|
| Individual study, experimentation, and code changes | Free, including for students, residents, and clinicians |
| Noncommercial teaching, assessment, research, and internal deployment at nonprofit or government institutions | Free, including internal hosting for participants |
| Commercial use, other institutional use, or hosting outside that exemption | Paid license |

Contact [Robert Chen](mailto:rob@robche.com) for paid licenses or
written fee waivers. Sharing copies is allowed under the license terms.
Earlier releases retain their MIT terms.

Early AnaSim versions adapted code from
[Python Anesthesia Simulator](https://github.com/AnesthesiaSimulation/Python_Anesthesia_Simulator),
licensed under GPL-3.0. Those implementations have since been substantially
rewritten. Third-party material retains its own terms.
