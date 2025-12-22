
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patient.patient import Patient
from physiology.respiration import RespiratoryModel
from core.step_helpers import StepHelpersMixin
from dataclasses import dataclass

def test_rr_ceiling():
    print("\n--- Test 1: Respiratory Rate Ceiling ---")
    patient = Patient(age=40, weight=70, height=170, sex="Male")
    # Baseline RR = 12
    resp = RespiratoryModel(patient)
    
    # Force Hypercapnia (60 mmHg) -> should boost drive
    resp.state.p_alveolar_co2 = 60.0
    
    # Step with no drugs
    state = resp.step(dt=1.0, ce_prop=0, ce_remi=0, mech_vent_mv=0, mech_rr=0, mech_vt_l=0)
    
    print(f"PaCO2: {state.p_alveolar_co2:.1f} mmHg")
    print(f"Drive Central: {state.drive_central:.2f}")
    print(f"Result RR: {state.rr:.2f} bpm (Baseline: {resp.rr_0})")
    
    if state.rr > resp.rr_0:
        print("PASS: RR exceeded baseline.")
    else:
        print("FAIL: RR did not exceed baseline.")

def test_vent_double_counting():
    print("\n--- Test 2: Ventilation Double Counting ---")
    patient = Patient(age=40, weight=70, height=170, sex="Male")
    resp = RespiratoryModel(patient)
    
    # Scenario: 
    # Vent: 12 bpm, 500 mL (6.0 L/min)
    # Patient: Triggering at 15 bpm (via CO2 boost), 500 mL
    # Expected VA should be based on 15 bpm * 0.5 L = 7.5 L/min (minus deadspace)
    # If double closing, it would be ~13.5 L/min
    
    # 1. Boost CO2 to get ~15 bpm spontaneous
    resp.state.p_alveolar_co2 = 60.0 # Enough for boost
    
    # 2. Step with Mechanical Ventilation Active
    state = resp.step(
        dt=1.0, 
        ce_prop=0, 
        ce_remi=0, 
        mech_vent_mv=6.0, 
        mech_rr=12.0, 
        mech_vt_l=0.5
    )
    
    deadspace = resp.vd_deadspace # ~0.154 L
    eff_rate = state.rr # Should be boosted ~15-20
    
    # Expected VA calculation logic being tested:
    # effective_rate = max(12, state.rr)
    # effective_vt = max(0.5, state.vt)
    # va = effective_rate * (effective_vt - deadspace)
    
    expected_va_approx = max(12, state.rr) * (max(0.5, state.vt/1000.0) - deadspace)
    
    print(f"Spontaneous RR: {state.rr:.2f}")
    print(f"Mechanical RR: 12.0")
    print(f"Calculated VA: {state.va:.2f} L/min")
    print(f"Expected VA (approx): {expected_va_approx:.2f} L/min")
    
    # A simple additive model would give:
    # VA_spont (~15 * 0.35) = 5.25
    # VA_mech (12 * 0.35) = 4.2
    # Total = 9.45
    #
    # Wait, my manual cals:
    # Spont RR ~ 15 -> 15 * (0.5 - 0.154) = 5.19
    # Mech RR 12 -> 12 * (0.5 - 0.154) = 4.15
    # Additive = 9.34
    # Synchronized (15 bpm) = 5.19
    
    if state.va < 7.0 and state.va > 4.0:
        print("PASS: VA reflects synchronized ventilation.")
    elif state.va > 9.0:
        print("FAIL: VA reflects additive (double) counting.")
    else:
        print(f"UNCERTAIN: VA {state.va:.2f} needs check.")

if __name__ == "__main__":
    test_rr_ceiling()
    test_vent_double_counting()
