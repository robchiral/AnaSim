# AnaSim architecture

## Overview

AnaSim couples pharmacology, cardiorespiratory physiology, ventilation, and
simulated monitors. Respiratory drive, mechanics, gas exchange, and hemodynamics
are modeled separately so that assisted and spontaneous ventilation each follow
their own path.

## State contract

`SimulationState` has two explicit layers:

| Layer | Fields | Meaning |
|------|--------|---------|
| Raw physiology / raw monitor-model outputs | `map`, `hr`, `sbp`, `dbp`, `co`, `sv`, `svr`, `bis`, `etco2`, `spo2`, `sao2`, `pa_co2`, `alveolar_co2`, `pao2` | Canonical backend truth used for physiology, recorder export, analytics, and internal safety logic |
| Display / learner-facing numerics | `display_map`, `display_hr`, `display_sbp`, `display_dbp`, `display_bis`, `display_etco2`, `display_spo2` | Values shown to the learner after monitor-specific smoothing or display behavior |

State access rules:

- `state.map/hr/sbp/dbp` are raw arterial physiology every step.
- UI, CLI, and tutorial/scenario code that represents what the learner sees should read `display_*`.
- Recorder output includes both layers so logs remain useful for analysis and for reproducing monitor output.
- Projection and monitor boundaries normalize public numeric fields to Python `float` values.

## Module structure

```text
AnaSim/
├── core/               # Simulation engine and control
│   ├── engine.py       # Facade/state container
│   ├── initialization.py # Startup target solving and subsystem seeding
│   ├── runtime.py      # Step orchestration
│   ├── projection.py   # Subsystem -> SimulationState projection
│   ├── monitors.py     # Display smoothing, NIBP, capno, alarms
│   ├── tci.py          # Target-Controlled Infusion controllers
│   ├── state.py        # Simulation state dataclasses
│   ├── action_log.py   # Timestamped control actions and scenario step activations
│   ├── recorder.py     # CSV recorder
│   └── utils.py        # Shared utility functions
├── patient/            # Patient demographics and PK/PD models
│   ├── patient.py
│   ├── pk_models.py
│   ├── pd/
│   │   ├── anesthesia.py
│   │   └── nmba.py
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
└── scripts/            # Utilities, including run_benchmarks.py
```

## Step pipeline

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
8. Projection -> raw subsystem state copied into `SimulationState`
9. Monitor synthesis -> waveforms, display_* values, alarms
10. Shivering update
11. Temperature update
12. Death detector
```

Pipeline ownership:

- `runtime.step_physiology()` computes the live physiology snapshot.
- `projection.project_runtime_physiology()` writes raw physiologic fields into `SimulationState`.
- `monitors.step_monitors()` writes display fields and waveforms only.

## Dependency graph

```mermaid
flowchart TD
    USER[User controls] --> DIST[Disturbances / events]
    USER --> TCI[TCI targets]
    USER --> MACHINE[Ventilator / circuit / vaporizer]

    DIST --> PKSCALE[PK hemodynamic scaling]
    PKSCALE --> TCISYNC[TCI resync]
    TCISYNC --> TCI

    TCI --> PKIV[IV PK models]
    MACHINE --> PKVOL[Volatile PK]
    MACHINE --> MECH[Respiratory mechanics]

    PKIV --> RESP[Respiration]
    PKIV --> HEMO[Hemodynamics]
    PKIV --> BIS[BIS / PD models]
    PKVOL --> RESP
    PKVOL --> HEMO
    PKVOL --> BIS

    MECH --> RESP
    MECH --> HEMO
    RESP -->|pa_co2, pao2| HEMO
    HEMO -->|co| PKVOL

    RESP --> MON[Monitor layer]
    HEMO --> MON
    BIS --> MON
    MON --> UI[UI / CLI / scenarios]
```

## Inputs by subsystem

### Hemodynamics

| Source | Data | Notes |
|--------|------|-------|
| PK Propofol | `propofol_cp` | Vasodilation, cardiac depression |
| PK Remifentanil | `remi_cp` | Bradycardia, vasodilation |
| Vasopressor PK | `nore_ce`, `epi_ce`, `phenyl_ce`, `vaso_ce`, `dobu_ce`, `mil_ce` | Vasoconstriction, inotropy, lusitropy |
| Volatile PK | `mac_sevo` | Cardiovascular depression |
| Resp mechanics | `pit`, `peep_cmH2O` | Preload and pulmonary coupling |
| Respiration | `pa_co2`, `pao2` | Chemoreflex and hypoxia effects |
| Disturbances | `d_hr`, `d_sv`, `d_svr` | Surgical stimulation |

### Respiration

| Source | Data | Notes |
|--------|------|-------|
| PK Propofol | `propofol_ce` | Depresses central drive |
| PK Remifentanil | `remi_ce` | Depresses drive and CO2 response |
| PK Rocuronium | `roc_ce` | Reduces muscle factor |
| Volatile PK | `mac_sevo` | Depresses ventilatory control |
| Resp mechanics | delivered VT, mean Paw, PEEP | Assisted ventilation and oxygenation |
| Hemodynamics | `co` (from prior step) | Influences perfusion-sensitive EtCO2 behavior |
| Thermal/shivering | metabolic factor, shiver level | Alters VCO2 and O2 demand |

### Monitor layer

| Source | Data | Notes |
|--------|------|-------|
| Hemodynamics | raw `map/hr/sbp/dbp` | Used to synthesize display numerics |
| Respiration | raw `etco2`, `sao2` | Used to synthesize capno and pulse oximetry behavior |
| BIS / TOF / LOC | raw model outputs | Displayed and alarmed values |
| Monitor settings | arterial line enabled, NIBP interval | Changes presentation mode |

## One-step lag cases

A few execution-order lags are part of the model contract:

| Value | Used by | Updated by | Reason |
|------|---------|------------|--------|
| `state.co` | Volatile PK scaling and respiration perfusion effects | Hemodynamics | CO is computed after PK and respiration in the same step |
| `state.va` | Volatile PK | Respiration | Alveolar ventilation is computed after machine and PK setup |
| `state.mv` | Circuit / machine context | Physiology | Minute ventilation is finalized after mechanics and respiration |

These lags are small at the intended simulation time steps and preserve the
physiology execution order.

## State synchronization examples

```text
pk_prop.state.ce      -> state.propofol_ce
pk_prop.state.c1      -> state.propofol_cp
pk_remi.state.ce      -> state.remi_ce
pk_remi.state.c1      -> state.remi_cp
resp_state.pa_co2     -> state.pa_co2
resp_state.p_alveolar_co2 -> state.alveolar_co2
hemo_state.map        -> state.map
hemo_state.map        -> state.display_map   (via monitor layer)
spo2_monitor output   -> state.spo2
spo2_monitor output   -> state.display_spo2
```

## Initialization

Startup runs separately from the visible loop:

- `awake` initializes directly from patient baselines with no hidden history.
- `steady_state` reaches its starting point by running a hidden bootstrap under
  controlled maintenance instead of solving for a mathematical equilibrium. The
  bootstrap skips recording, display history, death checks, and visible fluid or
  temperature bookkeeping.
- Visible time starts at zero from live subsystem state. Any norepinephrine
  needed to start at MAP 65 mmHg or higher remains visible and active.
- Early drift can occur as controllers, gases, and fluid balance continue from
  the bootstrap.

## Model notes

### Hemodynamics

- Based on Su et al. 2023 with added volume, pulmonary, vasoactive, septic shock, and anaphylaxis effects.
- Propofol and remifentanil cardiovascular effects consume plasma concentrations (`propofol_cp`, `remi_cp`); CNS depth, tolerance, BIS, and respiratory depression consume effect-site values (`propofol_ce`, `remi_ce`).

### Respiration

- Central drive is depressed by propofol, remifentanil, and sevoflurane.
- Neuromuscular weakness is handled through rocuronium-dependent muscle factor.
- `alveolar_co2`, `pa_co2`, and `etco2` are modeled separately.
- Low cardiac output widens the PaCO2-EtCO2 gap; arterial `sao2` remains tied to PaO2 and hemoglobin dissociation.

### Monitors

- MAP/HR/BIS display numerics use exponential smoothing with dt-aware time constants.
- Poor perfusion slows finger SpO2 and reduces pleth amplitude without changing raw arterial saturation.
- EtCO2 numerics update from completed capnogram breaths and become unavailable after 15 seconds without a valid exhaled sample.
- Arrest and near-arrest states bypass display smoothing.
- The UI draws the ABP waveform from a synthetic display trace; there is no dedicated arterial pressure waveform model.

### TCI

Controllers can rebuild after material PK changes and reseed after external boluses.

## Scenario objectives

`engine.actions` records controls and event transitions with simulation times.
The scenario overlay calls `begin_step()` when an objective becomes active.

Objectives fall into two kinds:

| Kind | Example | Check reads |
|------|---------|-------------|
| Action | "Give 500 mL", "start the vasopressor", "select the ETT" | `engine.actions` since the objective activated, plus current state when relevant |
| State / physiologic | "MAP > 65", "TOF below 25%", "circuit FiO2 below 30%" | Current engine state |

An action objective is satisfied only by an action taken after it activated, so
an intervention performed earlier in the case cannot complete it.

Step scoping uses log positions because paused actions can share a timestamp.
Step-scoped queries require an active objective.
