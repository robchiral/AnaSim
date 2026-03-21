import argparse
import sys
import time
import json
import os
from pathlib import Path

# Adjust path to find modules if running locally without install
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anasim.core.engine import SimulationEngine, SimulationConfig
from anasim.patient.patient import Patient




def run_headless(args):
    """Run simulation in headless mode."""
    print(f"Starting Headless Simulation (Duration: {args.duration}s)...")
    
    # Simple default config or load from file
    config_data = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    # Patient
    patient = Patient(
        age=config_data.get('age', 40),
        weight=config_data.get('weight', 70),
        height=config_data.get('height', 170),
        sex=config_data.get('sex', 'male'),
        asa=config_data.get('asa', 1),
        baseline_temp=config_data.get('baseline_temp', 37.0),
        baseline_hb=config_data.get('baseline_hb', 13.5),
        baseline_hct=config_data.get('baseline_hct', 0.42),
        renal_function=config_data.get('renal_function', 1.0),
        hepatic_function=config_data.get('hepatic_function', 1.0),
        baseline_hr=config_data.get('baseline_hr', 70.0),
        baseline_map=config_data.get('baseline_map', 90.0),
        baseline_rr=config_data.get('baseline_rr', 12.0),
        baseline_vt=config_data.get('baseline_vt', 500.0),
    )
    
    # Engine Config
    try:
        dt_val = float(config_data.get('dt', 0.01))
    except (TypeError, ValueError):
        dt_val = 0.01
    if dt_val <= 0:
        dt_val = 0.01
    sim_config = SimulationConfig(
        dt=dt_val,
        pk_model_propofol=config_data.get('pk_model_propofol', 'Eleveld'),
        pk_model_remi=config_data.get('pk_model_remi', 'Minto'),
        bis_model=config_data.get('bis_model', 'GrecoBouillon'),
        hemo_model=config_data.get('hemo_model', 'Su2023'),
        resp_model=config_data.get('resp_model', 'SingleCompartment'),
        pk_model_nore=config_data.get('pk_model_nore', 'Li'),
        pk_model_epi=config_data.get('pk_model_epi', 'Clutter'),
        loc_model=config_data.get('loc_model', 'Kern'),
        mode=config_data.get('mode', 'awake'),
        maint_type=config_data.get('maint_type', 'tiva'),
        disturbance_profile=config_data.get('disturbance_profile', None),
        baseline_hb=config_data.get('baseline_hb', 13.5),
        fidelity_mode=config_data.get('fidelity_mode', 'clinical'),
        volatile_agents=config_data.get('volatile_agents', ['sevoflurane']),
        maintenance_fluid_ml_hr=config_data.get('maintenance_fluid_ml_hr', None),
        simulation_speed=config_data.get('simulation_speed', 1.0),
        enable_death_detector=config_data.get('enable_death_detector', False),
        rng_seed=config_data.get('rng_seed', None),
    )
    
    engine = SimulationEngine(patient, sim_config)
    if args.record:
        engine.start_recording(output_dir=args.record_dir, sample_interval_sec=args.record_interval)
    engine.start()
    
    # Run loop
    start_real = time.time()
    steps = int(args.duration / sim_config.dt)
    
    for i in range(steps):
        engine.step(sim_config.dt)
        if i % 100 == 0:
            state = engine.get_latest_state()
            hr = state.display_value("hr")
            map_val = state.display_value("map")
            spo2 = state.display_value("spo2")
            print(f"Time: {state.time:.2f}s | HR: {hr:.1f} | MAP: {map_val:.1f} | SpO2: {spo2:.1f}")
            
    end_real = time.time()
    print(f"Simulation completed in {end_real - start_real:.2f}s real time.")

def run_ui():
    """Run simulation with UI."""
    from PySide6.QtWidgets import QApplication
    from anasim.ui.main_window import MainWindow

    # Check for existing QApplication (unlikely in main, but good practice)
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

def main():
    parser = argparse.ArgumentParser(description="AnaSim - Anesthesia Simulator")
    parser.add_argument("--mode", choices=["ui", "headless"], default="ui", help="Run mode (default: ui)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration for headless mode in seconds")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--record", action="store_true", help="Enable CSV recording (headless only)")
    parser.add_argument("--record-dir", type=str, default="recordings", help="Output directory for recordings")
    parser.add_argument("--record-interval", type=float, default=1.0, help="Sample interval in seconds for CSV")
    
    args = parser.parse_args()
    
    if args.mode == "headless":
        run_headless(args)
    else:
        run_ui()

if __name__ == "__main__":
    main()
