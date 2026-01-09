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
├── ui/                 # PyQt user interface
└── scripts/            # Utility and benchmark scripts
    ├── run_benchmarks.py
    └── capture_screenshots.py
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
- Based on Su et al. Br J Anaesth. 2023 comprehensive cardiovascular model (see `docs/REFERENCES.md`)
- Three state variables: TPR, SV*, HR* with TDE regulators
- Drug effects: Propofol, Remifentanil, Vasopressors, Volatiles
- Frank-Starling preload dependence, baroreflex-like feedback
- Septic shock: vasoplegia, capillary leak, pressor resistance
- Anaphylaxis: rapid-onset vasoplegia plus airway effects (bronchospasm/laryngospasm), no capillary leak model

### Respiration (`physiology/respiration.py`)
- **Central Drive**: Neural output inhibited by propofol, remifentanil, and sevoflurane.
- **Muscle Factor**: Mechanical ability inhibited by NMBA (rocuronium).
- **HCVR (Hypercapnic Ventilatory Response)**:
  - Negative feedback loop: rising PaCO2 stimulates respiratory drive.
  - Baseline slope: ~2.2 L/min/mmHg above 40 mmHg setpoint (dynamic end-tidal forcing studies).
  - **Drug Depression**: remifentanil strongly depresses slope (C50 ~1.1-1.2 ng/mL), propofol uses a proxy aligned to CO2-response studies (~2 µg/mL), and sevoflurane depresses HCVR with C50 ~1.1 MAC.
  - Critical for realistic emergence timing (accelerates CO2 elimination as drugs fade).
- **Gas Exchange**: Deadspace mixing and alveolar ventilation calculations.
  - EtCO2 is derived from PaCO2 with a small gradient that increases with V/Q mismatch and obstruction.
  - Baseline VCO2 scales with patient size (resting VO2 ~3.6 mL/kg/min, RQ ~0.8).

### Pharmacokinetics (`patient/pk_models.py`)
- 3-compartment models (central, peripheral, effect-site)
- Implementations: Marsh, Schnider, Eleveld (Propofol), Minto (Remifentanil)
- Hemodynamic-adjusted clearances for hemorrhage scenarios
- Organ impairment scalars: renal/hepatic function adjust clearance/volume for select agents
- Context-sensitive half-time (CSHT): propofol CSHT increases with infusion duration; remifentanil stays constant (~3-5 min)

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
python3 -m pytest tests/ -v

# Run specific test file

## Benchmarking

```bash
# Run all benchmarks
python3 scripts/run_benchmarks.py

# Run specific benchmarks (e.g. hemo, resp, pk)
python3 scripts/run_benchmarks.py --bench hemo --steps 5000

# Profile execution
python3 scripts/run_benchmarks.py --bench hemo --profile
```
