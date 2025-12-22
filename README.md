# AnaSim

**Real-time anesthesia simulation engine for medical education.**

AnaSim models patient physiology, pharmacokinetics, pharmacodynamics, and monitoring equipment in an interactive graphical interface. It provides a highly responsive environment for observing the hemodynamic and respiratory effects of anesthetic drugs and interventions.

> [!WARNING]
> **For educational use only**
> *   **Not for clinical use:** This software must never be used to guide patient care or make clinical decisions.
> *   **Not for research:** While based on published models, the interactions and heuristic adaptations have not been validated for pharmacological research.
> *   **No warranty:** The models are simplifications of complex physiology and may not accurately predict human responses in all scenarios.

## Features

*   **Physiological modeling**
    Comprehensive cardiovascular and respiratory systems that respond dynamically to drugs, ventilation, and surgical stimulation. The engine runs a 60Hz physics loop simulating beat-to-beat hemodynamics.

*   **Pharmacology**
    Implements standard pharmacokinetic (PK) and pharmacodynamic (PD) models for common anesthetic drugs:
    *   **Hypnotics:** Propofol (Marsh, Schnider, Eleveld), Sevoflurane.
    *   **Opioids:** Remifentanil (Minto).
    *   **Neuromuscular blockers:** Rocuronium (Wierda) with Sugammadex reversal.
    *   **Vasoactive agents:** Norepinephrine, Epinephrine, Phenylephrine.

*   **Patient monitoring**
    Real-time simulated patient monitor featuring:
    *   **ECG:** Simulated Lead II with rate-dependent morphology.
    *   **SpO2:** Photoplethysmogram with saturation derivation.
    *   **Capnography:** EtCO2 waveforms reflecting ventilation and perfusion.
    *   **NIBP:** Oscillometric cycling logic.
    *   **Depth of anesthesia:** BIS-like index and suppression ratio.

*   **Clinical scenarios**
    Simulate critical events including rapid sequence induction, massive hemorrhage, anaphylaxis, and emergence from anesthesia.

## Model fidelity

AnaSim combines rigorous published models with heuristics for areas where real-time computational models are sparse.

### Literature-derived (high fidelity)
*   **Hemodynamics:** Su et al. (2023) mechanism-based interaction model describing the effects of propofol and remifentanil on MAP, CO, and SVR, extended with mechanism-based logic for volatile agents.
*   **Propofol & Remifentanil PK:** Standard 3-compartment models (Marsh, Schnider, Eleveld, Minto).
*   **Volatile Agent PK:** Physiological 4-compartment PBPK model (Yasuda 1991, Mapleson 1996) simulation gas exchange and tissue distribution for Sevoflurane.
*   **Norepinephrine PK:** Li et al. (2024) 2-compartment model explicitly modeling the reduction of norepinephrine clearance by propofol anesthesia.
*   **Neuromuscular Block:** Wierda et al. (1991) model for Rocuronium.

### Heuristic (approximation)
*   **Epinephrine & Phenylephrine PK:** 1-compartment models calibrated to match clinical onset/offset times (t1/2 ~2.5–7 min) and typical hemodynamic responses, rather than specific population PK studies.
*   **Respiratory Control:** Modified depression models based on Bouillon (2003), adapted to allow physiological overrides (e.g., hypercapnic drive exceeding baseline respiratory rate).
*   **Thermoregulation:** Simplified heat balance model accounting for redistribution and metabolic reduction.

## Project origin & attribution

AnaSim is a specialized fork and evolution of the **Python Anesthesia Simulator (PAS)**.
*   The original TIVA implementations and core PK/PD architecture are derived from [PAS](https://github.com/AnesthesiaSimulation/Python_Anesthesia_Simulator).
*   AnaSim extends this foundation with enhanced hemodynamic feedback loops, mechanism-based volatile gas kinetics, refined ventilator-patient coupling, and expanded vasoactive drug support.

## Installation

### Prerequisites
*   Python 3.10 or newer.

### Steps
1.  **Clone or download**
    Download the repository to your local machine.

2.  **Install dependencies**
    Open your terminal, navigate to the project directory, and run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the simulation, run the entry script from the project root:

```bash
python run.py
```

### Interface controls
*   **Drug infusions:** Click on drug labels to set infusion rates or give boluses.
*   **Ventilator:** Toggle between Spontaneous and Mechanical ventilation. Adjust RR, Vt, and PEEP.
*   **Events:** Trigger scenarios (e.g., "Hemorrhage") from the control panel.

## Troubleshooting

*   **Dependency errors:** Ensure you have installed requirements using `pip install -r requirements.txt`.
*   **Python version:** This application requires Python 3.10+. Check your version with `python --version`.
