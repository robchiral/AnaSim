import sys
import time
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
    QFrame,
    QMessageBox,
)
from PySide6.QtCore import QTimer

from anasim.core.engine import SimulationEngine, SimulationConfig, Patient
from anasim.ui.monitor_widget import PatientMonitorWidget
from anasim.ui.controls_widget import ControlPanelWidget
from anasim.ui.config_dialog import SimulationSetupDialog
from anasim.ui.tutorial_overlay import TutorialOverlay, ScenarioOverlay
from anasim.ui.scenarios import SCENARIO_BUILDERS
from anasim.ui.styles import (
    COLORS,
    FONTS,
    STYLE_SPINBOX,
    get_base_widget_style,
    get_bar_style,
    get_button_style,
    get_toggle_button_style,
)

class MainWindow(QMainWindow):
    """Main application window container integrating Monitor and Controls."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnaSim - Anesthesia Simulator")
        self.resize(1600, 900)
        
        self.setStyleSheet(get_base_widget_style())
        
        # Show Setup Dialog
        if not self.show_setup_dialog():
            sys.exit(0)
            
        self.init_simulation()
        self.setup_ui()
        
        # Game Loop
        self.timer = QTimer()
        self.timer.setInterval(50)  # 20 FPS UI Update
        self.timer.timeout.connect(self.game_loop)
        
        self.last_real_time = 0.0

    def show_setup_dialog(self) -> bool:
        """Show config dialog and store params."""
        dlg = SimulationSetupDialog(self)
        if dlg.exec():
            self.sim_params = dlg.result_data
            return True
        return False

    def init_simulation(self):
        """Initialize the simulation engine with configured parameters."""
        if hasattr(self, "engine") and getattr(self.engine, "recorder", None):
            self.engine.stop_recording()
        p = self.sim_params
        self.arterial_line_enabled = p.get('arterial_line_enabled', True)
        self.patient = Patient(
            age=p['age'], 
            weight=p['weight'], 
            height=p['height'], 
            sex=p['sex'],
            baseline_hb=p.get('baseline_hb', 13.5),
            renal_function=p.get('renal_function', 1.0),
            hepatic_function=p.get('hepatic_function', 1.0),
            renal_status=p.get('renal_status', 'Normal'),
            hepatic_status=p.get('hepatic_status', 'Normal'),
        )
        self.config = SimulationConfig(
            pk_model_propofol=p.get('pk_model_propofol', 'Eleveld'),
            pk_model_nore=p.get('pk_model_nore', 'Beloeil'),
            pk_model_epi=p.get('pk_model_epi', 'Clutter'),
            bis_model=p.get('bis_model', 'Bouillon'),
            loc_model=p.get('loc_model', 'Kern'),
            mode=p.get('mode', 'awake'),
            maint_type=p.get('maint_type', 'tiva'),
            baseline_hb=p.get('baseline_hb', 13.5),
            fidelity_mode=p.get('fidelity_mode', 'clinical'),
            enable_death_detector=p.get('enable_death_detector', False)
        )
        self.engine = SimulationEngine(self.patient, self.config)
        self.tutorial_mode = p.get('tutorial_mode', False)
        self.death_dialog_shown = False # Reset for new session
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        # Base Layout
        base_layout = QVBoxLayout(central)
        base_layout.setContentsMargins(0, 0, 0, 0)
        base_layout.setSpacing(0)
        
        # Tutorial Overlay
        self.overlay = None
        if self.tutorial_mode:
            # Check if a specific scenario was selected
            scenario_id = self.sim_params.get('scenario_id')
            if scenario_id and scenario_id in SCENARIO_BUILDERS:
                scenario = SCENARIO_BUILDERS[scenario_id]()
                self.overlay = ScenarioOverlay(scenario)
            else:
                # Use legacy TutorialOverlay factory for induction/emergence
                self.overlay = TutorialOverlay(
                    mode=self.sim_params.get('mode', 'awake'),
                    maint_type=self.sim_params.get('maint_type', 'tiva')
                )
            base_layout.addWidget(self.overlay)
        
        # Main Layout: Monitor (Left) + Controls (Right)
        main_layout = QHBoxLayout()
        base_layout.addLayout(main_layout)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left Side: Monitor + Control Bar
        monitor_container = QWidget()
        mon_layout = QVBoxLayout(monitor_container)
        mon_layout.setContentsMargins(0, 0, 0, 0)
        mon_layout.setSpacing(0)
        
        self.monitor = PatientMonitorWidget(
            tutorial_mode=self.tutorial_mode,
            arterial_line_enabled=self.arterial_line_enabled
        )
        # Update patient label with simulation config
        self.monitor.update_patient_info(
            name="Simulated patient",  # or from p['name'] if added later
            age=self.patient.age,
            gender=self.patient.sex,
            weight=self.patient.weight,
            renal_status=getattr(self.patient, "renal_status", None),
            hepatic_status=getattr(self.patient, "hepatic_status", None),
        )
        mon_layout.addWidget(self.monitor, stretch=1)
        
        # Bottom Control Bar
        ctrl_bar = QFrame()
        ctrl_bar.setStyleSheet(get_bar_style("top"))
        ctrl_bar.setFixedHeight(56)
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(16, 8, 16, 8)
        ctrl_layout.setSpacing(16)
        
        # Start/Pause Button
        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet(
            get_button_style(variant="primary", padding="8px 20px", min_width=110)
        )
        self.btn_start.clicked.connect(self.toggle_simulation)
        ctrl_layout.addWidget(self.btn_start)

        # Record Toggle
        self.btn_record = QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.setStyleSheet(get_toggle_button_style(COLORS['danger']))
        self.btn_record.toggled.connect(self.toggle_recording)
        ctrl_layout.addWidget(self.btn_record)
        
        # Speed Control
        speed_container = QHBoxLayout()
        speed_container.setSpacing(8)
        
        lbl_speed = QLabel("Speed:")
        lbl_speed.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: {FONTS['size_small']};")
        speed_container.addWidget(lbl_speed)
        
        self.sb_speed = QDoubleSpinBox()
        self.sb_speed.setRange(0.1, 50.0)
        self.sb_speed.setValue(1.0)
        self.sb_speed.setSingleStep(0.5)
        self.sb_speed.setSuffix("x")
        self.sb_speed.setStyleSheet(STYLE_SPINBOX)
        self.sb_speed.setMinimumWidth(80)
        speed_container.addWidget(self.sb_speed)
        ctrl_layout.addLayout(speed_container)
        
        # Status Indicator
        self.lbl_status = QLabel("READY")
        self._set_status("READY", COLORS['text_dim'])
        ctrl_layout.addWidget(self.lbl_status)
        
        ctrl_layout.addStretch()
        
        # Time Display
        self.lbl_time = QLabel("00:00:00")
        self.lbl_time.setStyleSheet(f"""
            color: {COLORS['text']};
            font-size: {FONTS['size_display']};
            font-weight: 600;
        """)
        ctrl_layout.addWidget(self.lbl_time)
        
        mon_layout.addWidget(ctrl_bar)
        main_layout.addWidget(monitor_container, stretch=7)
        
        # Right Side: Controls
        self.controls = ControlPanelWidget(self.engine, tutorial_mode=self.tutorial_mode)
        main_layout.addWidget(self.controls, stretch=3)
        
        # Initial Sync
        self.controls.sync_with_engine()
        self._set_run_state("ready")

    def _set_status(self, text, color):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(
            f"color: {color}; font-size: {FONTS['size_small']}; font-weight: 600;"
        )

    def _set_run_state(self, state):
        if state == "running":
            self.btn_start.setText("Pause")
            self.btn_start.setStyleSheet(
                get_button_style(variant="warning", padding="8px 20px", min_width=110)
            )
            self._set_status("RUNNING", COLORS['success'])
            return
        if state == "paused":
            self.btn_start.setText("Resume")
            self.btn_start.setStyleSheet(
                get_button_style(
                    variant="primary",
                    outlined=True,
                    padding="8px 20px",
                    min_width=110,
                )
            )
            self._set_status("PAUSED", COLORS['warning'])
            return
        self.btn_start.setText("Start")
        self.btn_start.setStyleSheet(
            get_button_style(variant="primary", padding="8px 20px", min_width=110)
        )
        self._set_status("READY", COLORS['text_dim'])

    def toggle_simulation(self):
        if self.engine.running:
            self.engine.stop()
            self._set_run_state("paused")
            self.timer.stop()
        else:
            self.controls.sync_with_engine()
            self.engine.start()
            self._set_run_state("running")
            self.timer.start()
            self.last_real_time = time.time()
        self.time_accumulator = 0.0
        self.death_dialog_shown = False # Prevent multiple popups

    def toggle_recording(self, checked: bool):
        if checked:
            self.engine.start_recording(output_dir="recordings")
            self.btn_record.setText("Recording")
        else:
            self.engine.stop_recording()
            self.btn_record.setText("Record")
        
    def game_loop(self):
        # Time Management
        now = time.time()
        dt_real = now - self.last_real_time
        self.last_real_time = now
        
        if dt_real > 0.2: dt_real = 0.2
        
        # Apply Speed Factor
        speed = self.sb_speed.value()
        dt_sim_needed = dt_real * speed
        
        self.time_accumulator += dt_sim_needed
        
        # Run Engine Steps
        sim_step = getattr(self.engine.config, "dt", 0.01)
        max_steps = 100 
        steps_taken = 0
        
        while self.time_accumulator >= sim_step:
             self.engine.step(sim_step)
             self.time_accumulator -= sim_step
             steps_taken += 1
             if steps_taken >= max_steps:
                 self.time_accumulator = 0 
                 break
             
        # Update UI
        state = self.engine.get_latest_state()
        
        # Update Time Label
        total_seconds = int(state.time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        self.lbl_time.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        self.monitor.update_numerics(state)
        self.monitor.update_alarms(state)
        self.monitor.update_waveforms(self.engine)
        
        self.controls.sync_with_engine()
        
        if self.overlay:
            self.overlay.update_state(self.engine)

        # Check for death
        if state.is_dead and not self.death_dialog_shown:
            self.handle_patient_death(state.death_reason)

    def handle_patient_death(self, reason: str):
        """Handle patient death event."""
        self.death_dialog_shown = True # Mark as shown immediately
        
        # Stop engine properly without triggering toggle logic (which might restart it)
        was_running = self.engine.running
        self.engine.stop()

        if was_running:
             # Toggle_simulation updates UI text based on engine state.
             # Forcing a UI update:
             self.timer.stop()
             self.btn_start.setChecked(False)
             self._set_run_state("ready")
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Patient Deceased")
        msg.setText(f"The patient has died.\n\nReason: {reason}")
        msg.setStandardButtons(QMessageBox.Retry | QMessageBox.Close)
        msg.button(QMessageBox.Retry).setText("Restart Simulation")
        msg.button(QMessageBox.Close).setText("Keep Viewing State")
        
        ret = msg.exec()
        
        if ret == QMessageBox.Retry:
            # Restart triggers setup dialog again
            if self.show_setup_dialog():
                self.init_simulation()
                self.setup_ui()
                self.controls.sync_with_engine()
        # If Close, do nothing (simulation remains paused, user can inspect graphs)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
