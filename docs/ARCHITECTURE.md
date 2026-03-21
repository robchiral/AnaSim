# AnaSim Architecture

## Overview

AnaSim is a real-time anesthesia simulation engine that couples pharmacology, cardiorespiratory physiology, mechanical ventilation, and simulated monitors. Assisted ventilation (VCV/PCV/PSV/CPAP or bag-mask) is handled through the respiratory mechanics model, while purely spontaneous breathing bypasses mechanics and is handled by the respiratory drive model. Fresh gas flow includes O2/Air and optional N2O; N2O is delivered through the circuit rather than the vaporizer and contributes to total MAC with minimal BIS effect.

## State Contract

`SimulationState` has two explicit layers:

| Layer | Fields | Meaning |
|------|--------|---------|
| Raw physiology / raw monitor-model outputs | `map`, `hr`, `sbp`, `dbp`, `co`, `sv`, `svr`, `bis`, `etco2`, `spo2`, `sao2`, `pa_co2`, `alveolar_co2`, `pao2` | Canonical backend truth used for physiology, recorder export, analytics, and internal safety logic |
| Display / learner-facing numerics | `display_map`, `display_hr`, `display_sbp`, `display_dbp`, `display_bis`, `display_etco2`, `display_spo2` | Values shown to the learner after monitor-specific smoothing or display behavior |

Important implications:
- `state.map/hr/sbp/dbp` are raw arterial physiology every step.
- UI, CLI, and tutorial/scenario code that represents what the learner sees should read `display_*`.
- Recorder output includes both layers so logs remain useful for analysis and for reproducing monitor output.

## Module Structure

```text
AnaSim/
├── core/               # Simulation engine and control
│   ├── engine.py       # Main simulation orchestrator
│   ├── step_helpers.py # Step pipeline helpers
│   ├── tci.py          # Target-Controlled Infusion controllers
│   ├── state.py        # Simulation state dataclasses
│   ├── recorder.py     # CSV recorder
│   └── utils.py        # Shared utility functions
├── patient/            # Patient demographics and PK/PD models
│   ├── patient.py
│   ├── pk_models.py
│   ├── pd_models.py
│   └── volatile_pk.py
├── physiology/         # Hemodynamics, respiration, disturbances
│   ├── hemodynamics.py
│   ├── respiration.py
│   ├── resp_mech.py
│   └── disturbances.py
├── monitors/           # Monitor waveforms and alarms
│   ├── ecg.py
│   ├── capno.py
│   ├── nibp.py
│   ├── spo2.py
│   └── alarms.py
├── ui/                 # PySide UI
└── scripts/            # Benchmarks and utilities
```

## Step Pipeline

```text
SimulationEngine.step()
1. Disturbances and user-triggered events
2. PK hemodynamic scaling from current blood volume and CO
3. Active TCI controller resynchronization to live PK state
4. TCI controller updates -> infusion rates
5. Machine state update (ventilator, bag-mask, vaporizer, circuit)
6. PK model update -> Ce/Cp and volatile tissue states
7. Physiology update
   a. Respiratory mechanics
   b. Respiration / gas exchange
   c. Hemodynamics
8. Monitor synthesis -> waveforms, display_* values, alarms
9. Shivering update
10. Temperature update
11. Death detector
```

The key rule is:
- `_step_physiology()` writes raw physiologic fields only.
- `_step_monitors()` writes display fields and waveforms only.

## Model Notes

### Hemodynamics
- Based on the Su et al. 2023 mechanistic interaction model with additional volume, pulmonary, and vasoactive support.
- Septic shock and anaphylaxis are represented as explicit modifiers of vascular tone and related responses.
- Cache invalidation is now mutation-safe through property setters on key mutable inputs such as rhythm and shock severity.

### Respiration
- Central drive is depressed by propofol, remifentanil, and sevoflurane.
- Neuromuscular weakness is handled through rocuronium-dependent muscle factor.
- `alveolar_co2`, `pa_co2`, and `etco2` are modeled separately.
- Hemodynamics now consumes arterial `pa_co2`, not the alveolar proxy.
- Low cardiac output depresses EtCO2 by widening the PaCO2-EtCO2 gap instead of forcing arterial saturation to fall.
- Arterial `sao2` remains tied to PaO2 and the hemoglobin dissociation relationship.

### Monitors
- MAP/HR/BIS display numerics use dt-aware exponential smoothing rather than fixed per-step alphas.
- Arrest and near-arrest states bypass display smoothing so coarse `dt` does not leave falsely reassuring vital signs on screen.
- Poor perfusion affects pulse-ox display reliability and pleth amplitude, not raw arterial saturation.
- ABP waveform rendering in the UI still uses a synthetic display waveform rather than a dedicated arterial pressure waveform model.

### TCI
- Controllers can rebuild their discretized dynamics when the live PK model drifts materially because of hemorrhage or other hemodynamic changes.
- Controllers reseed their internal state estimate after external boluses so the controller state does not drift away from the live PK compartments.

## Data Ownership

| Subsystem | Writes | Does not write |
|----------|--------|----------------|
| PK sync | Drug concentrations (`*_ce`, `*_cp`) | Raw hemodynamics, display numerics |
| Physiology | `map`, `hr`, `sbp`, `dbp`, `co`, `sv`, `svr`, `etco2`, `pa_co2`, `alveolar_co2`, `pao2`, `sao2` | `display_*` |
| Monitor layer | `display_*`, waveforms, alarms, NIBP updates | Raw arterial pressure / heart rate |
| UI / CLI | Reads `display_*` | Mutates backend physiology |

## Testing

```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/test_state_semantics.py -v
python3 -m pytest tests/test_ui.py -v
```

`tests/conftest.py` forces `QT_QPA_PLATFORM=offscreen`, so UI tests run headlessly by default.

## Benchmarking

```bash
python3 scripts/run_benchmarks.py
python3 scripts/run_benchmarks.py --bench hemo --steps 5000
python3 scripts/run_benchmarks.py --bench hemo --profile
```
