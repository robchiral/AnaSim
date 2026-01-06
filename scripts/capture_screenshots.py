
import sys
import os
import argparse
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtCore import QTimer

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anasim.ui.main_window import MainWindow
from anasim.ui.config_dialog import SimulationSetupDialog

class ScreenshotMainWindow(MainWindow):
    """Subclass to bypass the interactive setup dialog."""
    def __init__(self, sim_params=None):
        self.preset_params = sim_params
        super().__init__()

    def show_setup_dialog(self) -> bool:
        if self.preset_params:
            self.sim_params = self.preset_params
            return True
            
        # Default fallback
        self.sim_params = {
            'age': 40, 'weight': 70.0, 'height': 170.0, 'sex': 'male',
            'baseline_hb': 13.5, 'renal_function': 1.0, 'hepatic_function': 1.0,
            'renal_status': 'Normal', 'hepatic_status': 'Normal',
            'mode': 'awake', 'maint_type': 'tiva',
            'tutorial_mode': False, 'scenario_id': None,
            'pk_model_propofol': 'Eleveld', 'pk_model_nore': 'Li',
            'pk_model_epi': 'Clutter', 'bis_model': 'Bouillon', 
            'loc_model': 'Kern', 'fidelity_mode': 'clinical',
            'enable_death_detector': False, 'arterial_line_enabled': True
        }
        return True

from PySide6.QtGui import QPainter, QColor, QPen

def capture_widget(widget, name, output_dir):
    path = os.path.join(output_dir, f"{name}.png")
    print(f"Saving {path} with border...")
    
    # Grab the original pixmap
    pixmap = widget.grab()
    
    # Convert to QImage to allow painting
    image = pixmap.toImage()
    
    # Create a painter to draw on the image
    painter = QPainter(image)
    
    # Define border
    pen = QPen(QColor("#666666"))
    pen.setWidth(2) 
    painter.setPen(pen)
    
    # Draw rect on the edge (adjust for pen width)
    painter.drawRect(0, 0, image.width() - 1, image.height() - 1)
    
    painter.end()
    
    # Save modified image
    image.save(path)

def run_capture(target, output_dir, run_duration=0):
    app = QApplication(sys.argv)
    os.makedirs(output_dir, exist_ok=True)

    if target == 'config':
        dialog = SimulationSetupDialog()
        dialog.show()
        
        def capture_dialog():
            capture_widget(dialog, "config_dialog", output_dir)
            app.quit()
            
        QTimer.singleShot(500, capture_dialog)
        sys.exit(app.exec())

    elif target in ['main', 'induction', 'maintenance']:
        params = None
        if target == 'induction':
            params = {
                'age': 25, 'weight': 70.0, 'height': 175.0, 'sex': 'male',
                'baseline_hb': 14.0, 'renal_function': 1.0, 'hepatic_function': 1.0, 
                'renal_status': 'Normal', 'hepatic_status': 'Normal',
                'mode': 'awake', 'maint_type': 'tiva',
                'tutorial_mode': True, 
                'scenario_id': None, # Standard induction tutorial
                'pk_model_propofol': 'Eleveld', 'pk_model_nore': 'Li',
                'pk_model_epi': 'Clutter', 'bis_model': 'Bouillon', 
                'loc_model': 'Kern', 'fidelity_mode': 'clinical',
                'enable_death_detector': False, 'arterial_line_enabled': True
            }
        elif target == 'maintenance':
             params = {
                'age': 55, 'weight': 80.0, 'height': 170.0, 'sex': 'female',
                'baseline_hb': 12.0, 'renal_function': 1.0, 'hepatic_function': 1.0,
                'renal_status': 'Normal', 'hepatic_status': 'Normal',
                'mode': 'steady_state', # Already under anesthesia
                'maint_type': 'tiva',
                'tutorial_mode': False,
                'scenario_id': None,
                'pk_model_propofol': 'Eleveld', 'pk_model_nore': 'Li',
                'pk_model_epi': 'Clutter', 'bis_model': 'Bouillon', 
                'loc_model': 'Kern', 'fidelity_mode': 'clinical',
                'enable_death_detector': False, 'arterial_line_enabled': True
             }

        window = ScreenshotMainWindow(sim_params=params)
        # Wider window to ensure machine tab fits without scrolling
        window.resize(1800, 900) 
        window.show()

        def capture_main():
            prefix = f"{target}_" if target != 'main' else ""
            capture_widget(window, f"{prefix}full_window", output_dir)
            print("Done.")
            app.quit()

        if run_duration > 0:
            print(f"Running simulation ({target}) for {run_duration}s...")
            window.toggle_simulation()
            QTimer.singleShot(int(run_duration * 1000), capture_main)
        else:
            QTimer.singleShot(500, capture_main)

        sys.exit(app.exec())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture screenshots of AnaSim")
    parser.add_argument("--output", "-o", default="screenshots", help="Output directory")
    parser.add_argument("--target", "-t", choices=['config', 'main', 'induction', 'maintenance'], default='main', help="Target screen to capture")
    parser.add_argument("--run", "-r", type=float, default=0, help="Run simulation for N seconds before capture")
    
    args = parser.parse_args()
    run_capture(args.target, args.output, args.run)
