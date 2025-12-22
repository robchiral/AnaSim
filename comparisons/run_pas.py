import sys
import os
import pandas as pd
import numpy as np

# Add PAS source to path
PAS_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PAS/src"))
if PAS_SRC not in sys.path:
    sys.path.append(PAS_SRC)



from python_anesthesia_simulator.simulator import Simulator
from python_anesthesia_simulator.patient import Patient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


from utils import StandardPatient

def run_propofol_scenario():
    print("Running PAS Propofol Scenario...")
    std = StandardPatient()
    
    # 0. Setup Patient
    # PAS Patient Constructor: [age, height, weight, sex]
    patient = Patient([std.age, std.height, std.weight, std.sex])
    
    # 1. Setup Simulator
    sim = Simulator(patient=patient, save_signals=False, noise=False)
    
    # Data storage
    records = []
    
    dt = 1.0 
    total_time_min = 40
    total_steps = int(total_time_min * 60 / dt)
    
    # Scenario:
    # 10s: Bolus 2 mg/kg
    # 60s: Infusion 100 mcg/kg/min for 30 min
    
    bolus_dose_mg = 2.0 * std.weight
    infusion_rate_ug_kg_min = 100.0
    infusion_rate_mg_s = (infusion_rate_ug_kg_min * std.weight) / 1000.0 / 60.0
    
    # Bolus delivered over 10s
    bolus_rate_mg_s = bolus_dose_mg / 10.0
    
    for step in range(total_steps):
        t = step * dt
        
        u_prop = 0.0
        
        # Bolus 10s - 20s
        if 10.0 <= t < 20.0:
            u_prop += bolus_rate_mg_s
            
        # Infusion 60s - 31m
        if 60.0 <= t < (30 * 60 + 60.0):
            u_prop += infusion_rate_mg_s
            
        sim.one_step(input_propo=u_prop)
        
        # Extract Data
        # Propofol PK state (Assuming standard 3-compartment)
        cp = sim.patient.propo_pk.x[0,0] if hasattr(sim.patient.propo_pk, 'x') else 0.0
        ce = sim.patient.propo_pk.x_e[0,0] if hasattr(sim.patient.propo_pk, 'x_e') else 0.0

        records.append({
            "Time": t,
            "Cp": cp, 
            "Ce": ce,
            "BIS": sim.bis,
            "MAP": sim.map,
            "HR": sim.hr
        })
        
    df = pd.DataFrame(records)
    output_path = os.path.join(RESULTS_DIR, "pas_propofol.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

def run_norepi_scenario():
    print("Running PAS Norepinephrine Scenario...")
    std = StandardPatient()
    patient = Patient([std.age, std.height, std.weight, std.sex])
    sim = Simulator(patient=patient, save_signals=False, noise=False)
    
    records = []
    dt = 1.0
    total_steps = int(25 * 60 / dt)
    
    # 10s: Start Infusion 0.1 mcg/kg/min for 20 min
    infusion_rate_ug_kg_min = 0.1
    # Input to PAS one_step input_nore is usually ug/s? 
    # doc string says: "input_nore : float, optional. Infusion rate (µg/s)"
    infusion_rate_ug_s = (infusion_rate_ug_kg_min * std.weight) / 60.0
    
    for step in range(total_steps):
        t = step * dt
        u_nore = 0.0
        
        if 10.0 <= t < (20 * 60 + 10):
            u_nore = infusion_rate_ug_s
            
        sim.one_step(input_nore=u_nore)
        
        # Norepinephrine PK state
        cp_nore = sim.patient.nore_pk.x[0,0] if hasattr(sim.patient, 'nore_pk') else 0.0
        
        records.append({
            "Time": t,
            "Cp": cp_nore,
            "MAP": sim.map,
            "HR": sim.hr
        })
        
    df = pd.DataFrame(records)
    output_path = os.path.join(RESULTS_DIR, "pas_norepinephrine.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    run_propofol_scenario()
    run_norepi_scenario()
