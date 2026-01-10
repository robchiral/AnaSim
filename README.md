# AnaSim

Real-time anesthesia simulation engine for medical education.

![AnaSim Induction Window](https://raw.githubusercontent.com/robchiral/AnaSim/main/docs/images/induction_full_window.png)

> [!WARNING]
> **For educational use only**
> - Not for clinical use or research.
> - No warranty; models are simplifications and may not accurately predict human responses.

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

## Usage

### UI mode

```bash
anasim
```

### Headless mode

```bash
anasim --mode headless --duration 10
```

See [docs/CLI_USAGE.md](docs/CLI_USAGE.md) for detailed CLI instructions.

## Features

- **Physiological modelßing:** Cardiovascular and respiratory systems responding to drugs, ventilation, and surgical stimulation at 60Hz.
- **Pharmacology:** PK/PD models for propofol, remifentanil, sevoflurane, rocuronium, and vasoactive agents.
- **Patient monitoring:** Simulated ECG, SpO2, capnography, NIBP, and depth of anesthesia.
- **Clinical scenarios/events:** Rapid sequence induction, hemorrhage, anaphylaxis, septic shock, and emergence.
- **Thermoregulation:** Core temperature dynamics with vasoconstriction and shivering.

## Model Fidelity

AnaSim combines published models with heuristics where real-time computational models are sparse. See [docs/REFERENCES.md](docs/REFERENCES.md) for the full bibliography.

**Literature-derived:** hemodynamics, propofol/remifentanil/norepinephrine PK, volatile agent PBPK, rocuronium effect-site dynamics.

**(Heuristic):** epinephrine/phenylephrine PK, respiratory control overrides, thermoregulation.

## Comparison

| Feature | AnaSim | PAS | Gas Man |
|---------|--------|-----|---------|
| IV anesthetics | Yes | Yes | No |
| Volatile anesthetics | Yes | No | Yes |
| Hemodynamics | Advanced | Basic | No |
| Vasoactive drugs | Multiple | Norepinephrine | No |
| Real-time monitor UI | Yes | No | No |
| Clinical scenarios | Yes | No | No |
| Open source | Yes | Yes | No |
| Primary use | Education | Control research | Education |

Initial TIVA implementations derived from [PAS](https://github.com/AnesthesiaSimulation/Python_Anesthesia_Simulator).
