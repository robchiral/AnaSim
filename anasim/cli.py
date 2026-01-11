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

    # Patient default
    patient = Patient(
        age=config_data.get('age', 40),
        weight=config_data.get('weight', 70),
        height=config_data.get('height', 170),
        sex=config_data.get('sex', 'Male')
    )
    
    # Engine Config
    sim_config = SimulationConfig(
        mode=config_data.get('mode', 'awake'),
        maint_type=config_data.get('maint_type', 'tiva'),
        dt=0.01
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
            print(f"Time: {state.time:.2f}s | HR: {state.hr:.1f} | MAP: {state.map:.1f} | SpO2: {state.spo2:.1f}")
            
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
