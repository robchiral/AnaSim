import sys
import os
import pandas as pd
import numpy as np

# Add AnaSim root to path
ANASIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ANASIM_ROOT not in sys.path:
    sys.path.append(ANASIM_ROOT)

from core.engine import SimulationEngine
from core.state import SimulationConfig
from patient.patient import Patient
from utils import StandardPatient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_propofol_scenario():
    print("Running AnaSim Propofol Scenario...")
    std = StandardPatient()
    
    # Setup Patient
    patient = Patient(age=std.age, height=std.height, weight=std.weight, sex="male" if std.sex==0 else "female")
    
    # Setup Simulation
    # Force Schnider and Beloeil to match PAS defaults
    config = SimulationConfig(mode="dynamic", maint_type="tiva", pk_model_propofol="Schnider", pk_model_nore="Beloeil") 
    engine = SimulationEngine(patient, config)
    engine.start()
    
    records = []
    dt = 1.0
    total_steps = int(40 * 60 / dt)
    
    # Scenario
    bolus_dose_mg = 2.0 * std.weight
    infusion_rate_ug_kg_min = 100.0
    infusion_rate_mg_sec = (infusion_rate_ug_kg_min * std.weight) / 1000.0 / 60.0
    
    print(f"Propofol: Bolus {bolus_dose_mg} mg, Infusion {infusion_rate_mg_sec*60} mg/min")
    
    for step in range(total_steps):
        t = step * dt
        
        # Reset rates (control inputs persist until changed)
        
        # Bolus
        if step == int(10.0 / dt):
             engine.give_drug_bolus("Propofol", bolus_dose_mg)
             
        # Infusion
        if 60.0 <= t < (30 * 60 + 60.0):
            engine.propofol_rate_mg_sec = infusion_rate_mg_sec
        else:
            engine.propofol_rate_mg_sec = 0.0
            
        engine.step(dt)
        
        cp = engine.pk_prop.state.c1 if engine.pk_prop else 0.0
        ce = engine.pk_prop.state.ce if engine.pk_prop and hasattr(engine.pk_prop.state, 'ce') else 0.0
        
        records.append({
            "Time": engine.state.time,
            "Cp": cp,
            "Ce": ce,
            "BIS": engine.state.bis,
            "MAP": engine.state.map,
            "HR": engine.state.hr
        })
        
    df = pd.DataFrame(records)
    output_path = os.path.join(RESULTS_DIR, "anasim_propofol.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

def run_norepi_scenario():
    print("Running AnaSim Norepinephrine Scenario...")
    std = StandardPatient()
    patient = Patient(age=std.age, height=std.height, weight=std.weight, sex="male" if std.sex==0 else "female")
    config = SimulationConfig(mode="dynamic", pk_model_nore="Beloeil")
    engine = SimulationEngine(patient, config)
    engine.start()
    
    records = []
    dt = 1.0
    total_steps = int(25 * 60 / dt)
    
    infusion_rate_ug_kg_min = 0.1
    infusion_rate_ug_sec = (infusion_rate_ug_kg_min * std.weight) / 60.0
    
    for step in range(total_steps):
        t = step * dt
        
        if 10.0 <= t < (20 * 60 + 10.0):
            engine.nore_rate_ug_sec = infusion_rate_ug_sec
        else:
            engine.nore_rate_ug_sec = 0.0
            
        engine.step(dt)
        
        cp = engine.pk_nore.state.c1 if engine.pk_nore else 0.0
        
        records.append({
            "Time": engine.state.time,
            "Cp": cp,
            "MAP": engine.state.map,
            "HR": engine.state.hr
        })
        
    df = pd.DataFrame(records)
    output_path = os.path.join(RESULTS_DIR, "anasim_norepinephrine.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    
if __name__ == "__main__":
    run_propofol_scenario()
    run_norepi_scenario()
