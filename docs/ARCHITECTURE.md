# AnaSim Architecture

## Overview

AnaSim is a real-time anesthesia simulation engine that models patient physiology, pharmacokinetics, pharmacodynamics, and monitoring equipment. This document describes the core architecture and data flow.
Assisted ventilation (ventilator or bag-mask) is simulated through the respiratory mechanics model, while purely spontaneous breathing bypasses mechanics and is handled by the respiratory drive model.
The UI can toggle whether an arterial line is enabled: when enabled, ABP waveforms and continuous SBP/DBP/MAP are displayed; when disabled, the ABP panel is hidden and intermittent NIBP values are displayed.

## Module Structure

```
AnaSim/
├── core/               # Simulation engine and control
│   ├── engine.py       # Main simulation orchestrator
│   ├── tci.py          # Target-Controlled Infusion controllers
│   ├── state.py        # Simulation state dataclasses
│   ├── constants.py    # Shared physiological constants
│   └── utils.py        # Utility functions (hill_function, etc.)
├── patient/            # Patient and drug models
│   ├── patient.py      # Patient demographics
│   ├── pk_models.py    # 3-compartment PK (Propofol, Remifentanil, etc.)
│   ├── pd_models.py    # Pharmacodynamic models (BIS, TOF)
│   └── volatile_pk.py  # PBPK model for volatile anesthetics
├── physiology/         # Physiological models
│   ├── hemodynamics.py # Cardiovascular model (HR, MAP, SV, CO)
│   ├── respiration.py  # Respiratory drive and gas exchange
│   └── resp_mech.py    # Mechanical ventilation
├── monitors/           # Simulated monitoring equipment
│   ├── ecg.py          # ECG waveform
│   ├── capno.py        # Capnography waveforms
│   ├── nibp.py         # Non-invasive blood pressure
│   ├── spo2.py         # Pulse oximetry
│   └── alarms.py       # Alarm evaluation
└── ui/                 # PyQt user interface
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     SimulationEngine.step()                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Disturbances (surgical stimulation, user events)            │
│  2. TCI Controllers → drug infusion rates                       │
│  3. Machine (ventilator, bag-mask, vaporizer)                   │
│  4. PK Models → drug concentrations (Ce, Cp)                    │
│  5. Physiology:                                                 │
│     a. Respiration → RR, VT, PaCO2, PaO2                        │
│     b. Hemodynamics → HR, MAP, SBP, DBP, CO                     │
│  6. Monitors → waveforms, displayed values                      │
│  7. Temperature model                                           │
│  8. Death detector                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Key Models

### Hemodynamics (`physiology/hemodynamics.py`)
- Based on Su et al. 2023 BJA comprehensive cardiovascular model
- Three state variables: TPR, SV*, HR* with TDE regulators
- Drug effects: Propofol, Remifentanil, Vasopressors, Volatiles
- Frank-Starling preload dependence, baroreflex-like feedback

### Respiration (`physiology/respiration.py`)
- **Central Drive**: Neural output inhibited by Propofol (C50 ~4.5 µg/mL), Remifentanil, Sevo.
- **Muscle Factor**: Mechanical ability inhibited by NMBA (Rocuronium).
- **HCVR (Hypercapnic Ventilatory Response)**:
  - Negative feedback loop: Rising PaCO2 stimulates respiratory drive.
  - Baseline slope: ~2.5 L/min/mmHg above 40 mmHg setpoint.
  - **Drug Depression**: Opioids (Remi) strongly depress slope (by ~70%). Volatiles/Propofol cause moderate depression.
  - Critical for realistic emergence timing (accelerates CO2 elimination as drugs fade).
- **Gas Exchange**: Deadspace mixing, alveolar ventilation equations.


### Pharmacokinetics (`patient/pk_models.py`)
- 3-compartment models (central, peripheral, effect-site)
- Implementations: Marsh, Schnider, Eleveld (Propofol), Minto (Remifentanil)
- Hemodynamic-adjusted clearances for hemorrhage scenarios

### Volatile Anesthetics (`patient/volatile_pk.py`)
- 4-compartment PBPK: Lungs, VRG (brain), Muscle, Fat
- Partial pressure-based equilibration
- Age-corrected MAC calculation

### TCI Controller (`core/tci.py`)
- Shafer/Gregg algorithm implementation
- Effect-site or plasma targeting
- Rate-limited target changes for stability
- Controller timebase is synchronized with simulation time

## Units Convention

| Parameter                          | Unit    |
|------------------------------------|---------|
| Drug concentrations (Propofol)     | µg/mL   |
| Drug concentrations (Remifentanil) | ng/mL   |
| Blood pressure                     | mmHg    |
| Heart rate                         | bpm     |
| Cardiac output                     | L/min   |
| Temperature                        | °C      |
| Time step                          | seconds |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_hemodynamics.py -v
```

Tests cover clinical scenarios, PK/PD accuracy, hemodynamics, respiration, temperature, TCI, and long-duration stability.
