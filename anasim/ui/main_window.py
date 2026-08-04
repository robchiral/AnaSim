import sys
import time
import math
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
from PySide6.QtCore import Qt, QTimer

from anasim.core.engine import SimulationEngine, SimulationConfig, Patient
from anasim.ui.monitor_widget import PatientMonitorWidget
from anasim.ui.controls_widget import ControlPanelWidget
from anasim.ui.config_dialog import SimulationSetupDialog
from anasim.ui.tutorial_overlay import ScenarioOverlay
from anasim.ui.scenarios import SCENARIO_BUILDERS
from anasim.ui.styles import (
    COLORS,
    FONTS,
    STYLE_SPINBOX,
    get_base_widget_style,
    get_bar_style,
    get_button_style,
    get_toggle_button_style,
    get_status_label_style,
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
        self.arterial_line_enabled = p['arterial_line_enabled']
        self.patient = Patient(
            age=p['age'], 
            weight=p['weight'], 
            height=p['height'], 
            sex=p['sex'],
            baseline_hb=p['baseline_hb'],
            renal_function=p['renal_function'],
            hepatic_function=p['hepatic_function'],
            renal_status=p['renal_status'],
            hepatic_status=p['hepatic_status'],
        )
        self.config = SimulationConfig(
            pk_model_propofol=p['pk_model_propofol'],
            pk_model_nore=p['pk_model_nore'],
            pk_model_epi=p['pk_model_epi'],
            bis_model=p['bis_model'],
            loc_model=p['loc_model'],
            mode=p['mode'],
            maint_type=p['maint_type'],
            baseline_hb=p['baseline_hb'],
            enable_death_detector=p['enable_death_detector'],
        )
        self.engine = SimulationEngine(self.patient, self.config)
        self.tutorial_mode = p['tutorial_mode']
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
            scenario_id = self.sim_params['scenario_id']
            scenario = SCENARIO_BUILDERS[scenario_id]()
            scenario.prepare(self.engine)
            self.overlay = ScenarioOverlay(scenario, self.engine)
            base_layout.addWidget(self.overlay)
        
        # Main Layout: Monitor (Left) + Controls (Right)
        main_layout = QHBoxLayout()
        base_layout.addLayout(main_layout)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(1)
        
        # Left Side: Monitor + Control Bar
        monitor_container = QWidget()
        mon_layout = QVBoxLayout(monitor_container)
        mon_layout.setContentsMargins(0, 0, 0, 0)
        mon_layout.setSpacing(0)
        
        self.monitor = PatientMonitorWidget(
            arterial_line_enabled=self.arterial_line_enabled
        )
        # Update patient label with simulation config
        self.monitor.update_patient_info(
            name="Simulated patient",  # or from p['name'] if added later
            age=self.patient.age,
            gender=self.patient.sex,
            weight=self.patient.weight,
            renal_status=self.patient.renal_status,
            hepatic_status=self.patient.hepatic_status,
        )
        mon_layout.addWidget(self.monitor, stretch=1)
        
        # Bottom Control Bar
        ctrl_bar = QFrame()
        ctrl_bar.setObjectName("controlBar")
        ctrl_bar.setStyleSheet(get_bar_style("top"))
        ctrl_bar.setFixedHeight(62)
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(14, 9, 14, 9)
        ctrl_layout.setSpacing(12)
        
        # Start/Pause Button
        self.btn_start = QPushButton("Start simulation")
        self.btn_start.setStyleSheet(
            get_button_style(variant="primary", padding="8px 18px", min_width=126)
        )
        self.btn_start.clicked.connect(self.toggle_simulation)
        ctrl_layout.addWidget(self.btn_start)

        # Record Toggle
        self.btn_record = QPushButton("Record data")
        self.btn_record.setCheckable(True)
        self.btn_record.setStyleSheet(
            get_toggle_button_style(COLORS['danger'], text_color=COLORS['text_secondary'])
        )
        self.btn_record.toggled.connect(self.toggle_recording)
        ctrl_layout.addWidget(self.btn_record)
        
        # Speed Control
        speed_container = QHBoxLayout()
        speed_container.setSpacing(8)
        
        lbl_speed = QLabel("Simulation speed")
        lbl_speed.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10px; font-weight: 600;"
        )
        speed_container.addWidget(lbl_speed)
        
        self.sb_speed = QDoubleSpinBox()
        self.sb_speed.setRange(0.1, 50.0)
        self.sb_speed.setValue(1.0)
        self.sb_speed.setSingleStep(0.5)
        self.sb_speed.setSuffix(" ×")
        self.sb_speed.setStyleSheet(STYLE_SPINBOX)
        self.sb_speed.setMinimumWidth(80)
        speed_container.addWidget(self.sb_speed)
        ctrl_layout.addLayout(speed_container)
        
        # Status Indicator
        self.lbl_status = QLabel("● Ready")
        self._set_status("Ready", COLORS['text_dim'])
        ctrl_layout.addWidget(self.lbl_status)
        
        ctrl_layout.addStretch()
        
        # Time Display
        time_layout = QVBoxLayout()
        time_layout.setSpacing(0)
        lbl_time_title = QLabel("Simulation time")
        lbl_time_title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_time_title.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10px; font-weight: 600;"
        )
        time_layout.addWidget(lbl_time_title)
        self.lbl_time = QLabel("00:00:00")
        self.lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_time.setStyleSheet(f"""
            color: {COLORS['text']};
            font-size: {FONTS['size_display']};
            font-weight: 700;
        """)
        time_layout.addWidget(self.lbl_time)
        ctrl_layout.addLayout(time_layout)
        
        mon_layout.addWidget(ctrl_bar)
        main_layout.addWidget(monitor_container, stretch=13)
        
        # Right Side: Controls
        self.controls = ControlPanelWidget(self.engine)
        if self.overlay is not None:
            self.overlay.navigate_requested.connect(self.controls.open_tab)
        main_layout.addWidget(self.controls, stretch=7)
        
        # Initial Sync
        self.controls.sync_with_engine()
        initial_state = self.engine.get_latest_state()
        self.monitor.update_numerics(initial_state)
        self.monitor.update_alarms(initial_state)
        self._set_run_state("ready")

    def _set_status(self, text, color):
        self.lbl_status.setText(f"● {text}")
        self.lbl_status.setStyleSheet(get_status_label_style(color))

    def _set_run_state(self, state):
        if state == "running":
            self.btn_start.setText("Pause simulation")
            self.btn_start.setStyleSheet(
                get_button_style(variant="warning", padding="8px 18px", min_width=126)
            )
            self._set_status("Running", COLORS['success'])
            return
        if state == "paused":
            self.btn_start.setText("Resume simulation")
            self.btn_start.setStyleSheet(
                get_button_style(
                    variant="primary",
                    outlined=True,
                    padding="8px 18px",
                    min_width=126,
                )
            )
            self._set_status("Paused", COLORS['warning'])
            return
        self.btn_start.setText("Start simulation")
        self.btn_start.setStyleSheet(
            get_button_style(variant="primary", padding="8px 18px", min_width=126)
        )
        self._set_status("Ready", COLORS['text_dim'])

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
            self.last_real_time = time.perf_counter()
        self.time_accumulator = 0.0
        self.death_dialog_shown = False # Prevent multiple popups

    def toggle_recording(self, checked: bool):
        if checked:
            self.engine.start_recording(output_dir="recordings")
            self.btn_record.setText("Stop recording")
        else:
            self.engine.stop_recording()
            self.btn_record.setText("Record data")
        
    def game_loop(self):
        # Time Management
        now = time.perf_counter()
        dt_real = now - self.last_real_time
        self.last_real_time = now
        
        dt_real = min(dt_real, 0.2)
        
        # Apply Speed Factor
        speed = self.sb_speed.value()
        dt_sim_needed = dt_real * speed
        
        self.time_accumulator += dt_sim_needed
        
        # Run Engine Steps
        sim_step = self.engine.config.dt
        # Budget enough work for the fastest selectable speed. A fixed
        # 100-step cap silently discarded simulated time above roughly 20x.
        max_steps = max(100, math.ceil(0.2 * self.sb_speed.maximum() / sim_step))
        steps_taken = 0
        
        while self.time_accumulator >= sim_step:
            self.engine.step(sim_step)
            self.time_accumulator -= sim_step
            steps_taken += 1
            if steps_taken >= max_steps:
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
            self.overlay.update_state()

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
