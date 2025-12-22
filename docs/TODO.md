# AnaSim TODO

## Bug & Efficiency Fixes

No pending items.

---

## New Features

### Priority 1: Core Clinical Models (High Impact)

#### Cardiac Arrhythmia Modeling
**Effort:** 6-8 hours | **Files:** `monitors/ecg.py`, `physiology/hemodynamics.py`
**Current State:** Fixed HR waveforms only.
**Implementation:**
1. **Rhythms**: `SINUS, AFIB, SINUS_BRADY, SVT, VTACH, VFIB`.
2. **ECG Logic**:
   - AFib: irregular R-R, no P waves.
   - VFib: chaotic baseline details.
3. **Hemodynamic Impact**:
   - AFib: -15-25% CO (loss of atrial kick).
   - VTach: -40-60% CO.
   - VFib: Cardiac arrest (CO=0).
4. **Triggers**: Hyperkalemia, hypoxia, toxicity.

### Priority 2: Safety-Critical Models

#### Laryngospasm / Airway Obstruction
**Effort:** 3-4 hours | **Files:** `physiology/respiration.py`, `monitors/capno.py`
**Current State:** Binary airway (None/Mask/ETT).
**Implementation:**
1. **Obstruction Parameter**: `airway_obstruction` (0.0-1.0).
2. **Effects**: Increased resistance, reduced VT, shark-fin capnography.
3. **triggers**: Light anesthesia + stimulation.
4. **SpO2**: Desaturations follow severity.

#### Bronchospasm Model
**Effort:** 2-3 hours | **Files:** `physiology/respiration.py`, `core/engine.py`
**Current State:** Only anaphylaxis exists (basic resistance increase).
**Implementation:**
1. **Parameters**: `bronchospasm_severity` (0.0-1.0).
2. **Effects**: Increased expiratory time constant, reduced peak flow, wheezing cues.
3. **Capnography**: Prolonged upslope.

### Priority 3: Advanced Physiology & Enhancements

#### Shivering Model (Emergence)
**Effort:** 2-3 hours | **Files:** `core/engine.py`
**Triggers**: Temp < 36°C AND emerging (BIS > 60).
**Effects**: Increased O2 consumption (200-400%), increased CO2 production, HR increase.

#### Renal/Hepatic Impairment
**Status**: Missing
**Description**: Currently, drug clearance is constant.
**Recommendation**: Add optional clearance modifiers for patients with organ failure (modifying elimination rate constants for Rocuronium, Propofol, etc.).

#### Context-Sensitive Half-Time (CSHT)
**Status**: Missing UI feature
**Description**: The model simulates kinetics accurately, but the user has no visibility into wake-up times.
**Recommendation**: Add utility to calculate estimated time to specific decrement concentration (e.g., 1.0 µg/mL).

### Future Roadmap
- **Sepsis/Distributive Shock**: Warm shock phase, capillary leak, pressor resistance.
- **Right Heart Physiology**: PVR, RV-PA coupling, Tricuspid regurgitation.
- **Pharmacology Extensions**: Vasopressin, Dobutamine, Milrinone.
- **Fluid Kinetics**: Redistribution, third-spacing, coagulopathy from hemodilution.

## Testing

python3 -m pytest ./tests -v
