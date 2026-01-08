# Dependencies

## Execution Order (engine.step)

Each simulation step executes subsystems in this order:

```
1. Shivering       → Thermoregulatory heat/metabolic load
2. Disturbances    → Surgical stimulation, user events
3. TCI Controllers → Drug target → infusion rate calculation (synced to sim time)  
4. Machine         → Vaporizer, circuit → inspired agent concentration
5. PK Models       → Drug concentrations (Ce, Cp) updated
6. Physiology      → Resp Mechanics (assisted only) → Respiration → Hemodynamics
7. Monitors        → Waveforms, displayed values
8. Temperature     → Core temp, redistribution
9. Death Detector  → Viability check
```

## Data Flow Diagram

```mermaid
flowchart TD
    subgraph Inputs
        USER[User Controls]
        DIST[Disturbances]
    end

    subgraph Controllers
        TCI_P[TCI Propofol]
        TCI_R[TCI Remifentanil]
        TCI_V[TCI Vasopressors]
        TCI_NMBA[TCI Rocuronium]
    end

    subgraph Machine
        VAP[Vaporizer]
        VENT[Ventilator]
        BAG[Bag-mask]
        CIRC[Circuit]
    end

    subgraph PK["Pharmacokinetics"]
        PK_PROP[Propofol PK]
        PK_REMI[Remifentanil PK]
        PK_ROC[Rocuronium PK]
        PK_VASO[Vasopressor PK]
        PK_SEVO[Volatile PK]
    end

    subgraph Physiology
        RESP[Respiration Model]
        HEMO[Hemodynamics Model]
        MECH[Resp Mechanics]
    end

    subgraph Monitors
        BIS[BIS Monitor]
        ECG[ECG Monitor]
        CAPNO[Capnograph]
        SPO2[SpO2 Monitor]
        NIBP[NIBP Monitor]
    end

    %% User inputs
    USER --> TCI_P & TCI_R & TCI_V & TCI_NMBA
    USER --> VENT & VAP & BAG
    DIST --> HEMO

    %% TCI to infusion
    TCI_P --> PK_PROP
    TCI_R --> PK_REMI
    TCI_V --> PK_VASO
    TCI_NMBA --> PK_ROC

    %% Machine
    VAP --> PK_SEVO
    VENT --> MECH
    BAG --> MECH

    %% PK outputs
    PK_PROP -->|Ce| HEMO
    PK_PROP -->|Ce| BIS
    PK_REMI -->|Ce| HEMO
    PK_REMI -->|Ce| BIS
    PK_ROC -->|Ce| RESP
    PK_VASO -->|Ce| HEMO
    PK_SEVO -->|MAC| HEMO
    PK_SEVO -->|MAC| BIS

    %% Physiology interactions
    MECH -->|Paw, Pit| HEMO
    MECH -->|delivered VT| RESP
    RESP -->|PaCO2, PaO2| HEMO
    HEMO -->|CO| PK_SEVO
    HEMO -->|HR, MAP| ECG

    %% Monitor inputs
    RESP --> CAPNO
    RESP --> SPO2
    HEMO --> ECG
    HEMO --> NIBP
```

## Key Dependencies

### Hemodynamics receives from:
| Source | Data | Notes |
|--------|------|-------|
| PK Propofol | Ce | Vasodilation, cardiac depression |
| PK Remifentanil | Ce | Bradycardia, vasodilation |
| PK Vasopressors | Ce (Nore, Epi, Phenyl) | Vasoconstriction, inotropy |
| Volatile PK | MAC | Vasodilation, cardiac depression |
| Resp Mechanics | Pit (intrathoracic) | Preload reduction with assisted ventilation/PEEP |
| Respiration | PaCO2, PaO2 | Chemoreflex effects |
| Disturbances | d_hr, d_sv, d_tpr | Surgical stimulation |

### Respiration receives from:
| Source | Data | Notes |
|--------|------|-------|
| PK Propofol | Ce | Respiratory depression |
| PK Remifentanil | Ce | Respiratory depression |
| PK Rocuronium | Ce | Muscle paralysis |
| Volatile PK | MAC | Respiratory depression |
| Mechanics | Paw, delivered VT | Assisted ventilation only (ventilator or bag-mask) |
| Shivering | Intensity | Raises metabolic load (CO2/O2) |

### BIS receives from:
| Source | Data | Notes |
|--------|------|-------|
| PK Propofol | Ce | Primary hypnotic |
| PK Remifentanil | Ce | Synergistic interaction |
| Volatile PK | MAC | Volatile contribution |

## Monitoring Modes

- Arterial line enabled: ABP waveform and continuous SBP/DBP/MAP are displayed.
- Arterial line disabled: ABP panel is hidden and NIBP cuff values are displayed instead.

## One-Step Lag Cases

The following values have a one-step lag due to execution order:

| Value | Used By | Updated By | Lag Reason |
|-------|---------|------------|------------|
| `state.co` | Volatile PK | Hemodynamics | CO affects uptake, but hemo runs after PK |
| `state.mv` | Circuit mixing | Physiology | MV is computed after machine step |
| `state.va` | Volatile PK | Respiration | VA is computed after PK |
| `hemo.dist_svr` | Hemodynamics | Calculated at step start | Disturbance applied before hemo calculation |

These lags are acceptable at typical dt (0.5s) as physiological changes are slower than simulation steps.

## State Synchronization

State flows: **Subsystems → Engine State** (see `_sync_pk_state()`)

```
pk_prop.state.ce  ──→  engine.state.propofol_ce
pk_remi.state.ce  ──→  engine.state.remi_ce
hemo.state.map    ──→  engine.state.map
resp.state.rr     ──→  engine.state.rr
```

This happens at fixed points in `step()`, documented in `SimulationEngine` class docstring.
