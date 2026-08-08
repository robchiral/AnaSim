import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

from anasim.core.engine import SimulationConfig, SimulationEngine
from anasim.patient.patient import Patient

PATIENT_CONFIG_FIELDS = {
    field.name for field in fields(Patient)
    if field.name not in {"lbm", "bmi", "bsa"}
}
SIMULATION_CONFIG_FIELDS = {field.name for field in fields(SimulationConfig)}
CONFIG_FIELDS = PATIENT_CONFIG_FIELDS | SIMULATION_CONFIG_FIELDS


def build_models_from_config(config_data: dict) -> tuple[Patient, SimulationConfig]:
    """Build typed inputs and reject misspelled or obsolete keys."""
    unknown = set(config_data) - CONFIG_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown configuration key(s): {names}")

    patient_kwargs = {
        key: value for key, value in config_data.items()
        if key in PATIENT_CONFIG_FIELDS
    }
    simulation_kwargs = {
        key: value for key, value in config_data.items()
        if key in SIMULATION_CONFIG_FIELDS
    }
    return Patient(**patient_kwargs), SimulationConfig(**simulation_kwargs)


def run_headless(args):
    """Run simulation in headless mode."""
    print(f"Starting headless simulation for {args.duration:g} seconds")

    config_data = {}
    if args.config:
        try:
            config_data = json.loads(Path(args.config).read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading config: {e}")
            raise SystemExit(1) from e
    try:
        patient, sim_config = build_models_from_config(config_data)
    except ValueError as e:
        print(f"Error loading config: {e}")
        raise SystemExit(1) from e
    
    engine = SimulationEngine(patient, sim_config)
    if args.record:
        engine.start_recording(output_dir=args.record_dir, sample_interval_sec=args.record_interval)
    engine.start()
    
    start_real = time.perf_counter()
    steps = int(args.duration / sim_config.dt)
    
    try:
        for i in range(steps):
            engine.step(sim_config.dt)
            if i % 100 == 0:
                state = engine.get_latest_state()
                hr = state.display_value("hr")
                map_val = state.display_value("map")
                spo2 = state.display_value("spo2")
                print(f"Time: {state.time:.2f}s | HR: {hr:.1f} | MAP: {map_val:.1f} | SpO2: {spo2:.1f}")
        remainder = args.duration - steps * sim_config.dt
        if remainder > 1e-12:
            engine.step(remainder)
    finally:
        engine.stop_recording()
            
    end_real = time.perf_counter()
    print(f"Simulation completed in {end_real - start_real:.2f} seconds of real time")


def run_ui() -> int:
    """Run simulation with UI."""
    from anasim.ui.main_window import run

    return run()

def main():
    parser = argparse.ArgumentParser(description="AnaSim anesthesia simulator")
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
        raise SystemExit(run_ui())

if __name__ == "__main__":
    main()
