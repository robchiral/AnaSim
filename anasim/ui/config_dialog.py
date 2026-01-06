
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
                               QSpinBox, QDoubleSpinBox, QComboBox, QRadioButton, 
                               QGroupBox, QButtonGroup, QHBoxLayout, QLabel, QFrame,
                               QCheckBox, QGridLayout)
from PySide6.QtCore import Qt

from .styles import COLORS, get_dialog_style, get_button_style, get_frame_style

class SimulationSetupDialog(QDialog):
    """Dialog for configuring patient demographics and simulation mode."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AnaSim - New Simulation")
        self.setModal(True)
        self.result_data = None
        self.setMinimumWidth(860)
        
        self.setStyleSheet(get_dialog_style())
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 12, 16, 16)
        
        # Header
        header = QFrame()
        header.setStyleSheet(get_frame_style(bg_color=COLORS['card'], border_color=COLORS['border']))
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        lbl_title = QLabel("AnaSim")
        lbl_title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {COLORS['primary']};")
        header_layout.addWidget(lbl_title)
        
        lbl_subtitle = QLabel("Anesthesia Simulator")
        lbl_subtitle.setStyleSheet(f"font-size: 12px; color: {COLORS['text_secondary']};")
        header_layout.addWidget(lbl_subtitle)
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # Body (two-column layout)
        body = QHBoxLayout()
        body.setSpacing(12)
        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        body.addLayout(left_col, 1)
        body.addLayout(right_col, 1)
        layout.addLayout(body)

        # 1. Patient Demographics
        gb_patient = QGroupBox("Patient details")
        form = QFormLayout(gb_patient)
        form.setSpacing(8)
        form.setContentsMargins(12, 16, 12, 12)
        
        self.sb_age = QSpinBox()
        self.sb_age.setRange(1, 100)
        self.sb_age.setValue(40)
        self.sb_age.setSuffix(" years")
        form.addRow("Age:", self.sb_age)
        
        self.sb_weight = QDoubleSpinBox()
        self.sb_weight.setRange(10, 200)
        self.sb_weight.setValue(70.0)
        self.sb_weight.setSuffix(" kg")
        form.addRow("Weight:", self.sb_weight)
        
        self.sb_height = QDoubleSpinBox()
        self.sb_height.setRange(50, 250)
        self.sb_height.setValue(170.0)
        self.sb_height.setSuffix(" cm")
        form.addRow("Height:", self.sb_height)
        
        self.cb_sex = QComboBox()
        self.cb_sex.addItems(["Male", "Female"])
        form.addRow("Sex:", self.cb_sex)

        self.sb_hgb = QDoubleSpinBox()
        self.sb_hgb.setRange(6.0, 20.0)
        self.sb_hgb.setSingleStep(0.1)
        self.sb_hgb.setValue(13.5)
        self.sb_hgb.setSuffix(" g/dL")
        form.addRow("Baseline Hgb:", self.sb_hgb)
        
        left_col.addWidget(gb_patient)

        # 1b. Organ Function
        gb_organ = QGroupBox("Organ function")
        organ_form = QFormLayout(gb_organ)
        organ_form.setSpacing(8)
        organ_form.setContentsMargins(12, 16, 12, 12)

        self.cb_renal = QComboBox()
        self.cb_renal.addItem("Normal (eGFR >= 90)", 1.0)
        self.cb_renal.addItem("Mild (eGFR 60-89)", 0.8)
        self.cb_renal.addItem("Moderate (eGFR 30-59)", 0.6)
        self.cb_renal.addItem("Severe (eGFR < 30)", 0.4)
        organ_form.addRow("Renal:", self.cb_renal)

        self.cb_hepatic = QComboBox()
        self.cb_hepatic.addItem("Normal (no cirrhosis)", 1.0)
        self.cb_hepatic.addItem("Mild (Child-Pugh A)", 0.9)
        self.cb_hepatic.addItem("Moderate (Child-Pugh B)", 0.7)
        self.cb_hepatic.addItem("Severe (Child-Pugh C)", 0.5)
        organ_form.addRow("Hepatic:", self.cb_hepatic)

        left_col.addWidget(gb_organ)
        
        # 2. Start Scenario (Standard mode)
        self.gb_scenario = QGroupBox("Simulation start")
        l_scen = QVBoxLayout(self.gb_scenario)
        l_scen.setSpacing(10)
        l_scen.setContentsMargins(12, 16, 12, 12)

        lbl_start_hint = QLabel("Applies in standard mode; tutorial scenarios set the start state.")
        lbl_start_hint.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        lbl_start_hint.setWordWrap(True)
        l_scen.addWidget(lbl_start_hint)
        
        self.rb_awake = QRadioButton("Induction mode (awake patient)")
        self.rb_maint = QRadioButton("Maintenance mode (under anesthesia)")
        self.rb_maint.setChecked(True)
        
        grp = QButtonGroup(self)
        grp.addButton(self.rb_awake)
        grp.addButton(self.rb_maint)
        
        l_scen.addWidget(self.rb_awake)
        l_scen.addWidget(self.rb_maint)
        
        h_maint = QHBoxLayout()
        lbl_maint = QLabel("Anesthetic technique:")
        lbl_maint.setStyleSheet(f"color: {COLORS['text_secondary']};")
        h_maint.addWidget(lbl_maint)
        self.cb_maint_type = QComboBox()
        self.cb_maint_type.addItems(["TIVA (propofol)", "Inhalational (sevoflurane)"])
        h_maint.addWidget(self.cb_maint_type)
        h_maint.addStretch()
        l_scen.addLayout(h_maint)
        
        left_col.addStretch()
        
        # 3. UI Mode (Tutorial)
        gb_ui = QGroupBox("Session type")
        l_ui = QVBoxLayout(gb_ui)
        l_ui.setSpacing(8)
        l_ui.setContentsMargins(12, 16, 12, 12)
        
        self.rb_advanced = QRadioButton("Standard mode")
        self.rb_tutorial = QRadioButton("Tutorial mode (with guidance)")
        self.rb_advanced.setChecked(True)
        
        lbl_tutorial_info = QLabel("Standard mode is free play. Tutorial mode provides guided scenarios.")
        lbl_tutorial_info.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        lbl_tutorial_info.setWordWrap(True)
        
        grp_ui = QButtonGroup(self)
        grp_ui.addButton(self.rb_advanced)
        grp_ui.addButton(self.rb_tutorial)
        
        l_ui.addWidget(self.rb_advanced)
        l_ui.addWidget(self.rb_tutorial)
        l_ui.addWidget(lbl_tutorial_info)
        
        # Scenario selection (shown when tutorial mode is selected)
        self.scenario_container = QFrame()
        self.scenario_container.setStyleSheet("background: transparent;")
        h_scenario = QHBoxLayout(self.scenario_container)
        h_scenario.setContentsMargins(0, 4, 0, 0)
        lbl_scenario = QLabel("Tutorial scenario:")
        lbl_scenario.setStyleSheet(f"color: {COLORS['text_secondary']};")
        h_scenario.addWidget(lbl_scenario)
        self.cb_scenario = QComboBox()
        self.cb_scenario.addItem(
            "Induction tutorial (awake start)",
            {"scenario_id": None, "mode": "awake"},
        )
        self.cb_scenario.addItem(
            "Emergence tutorial (maintenance start)",
            {"scenario_id": None, "mode": "steady_state"},
        )
        self.cb_scenario.addItem(
            "Hemorrhage response",
            {"scenario_id": "hemorrhage_response", "mode": "steady_state"},
        )
        self.cb_scenario.addItem(
            "Anaphylaxis response",
            {"scenario_id": "anaphylaxis_response", "mode": "steady_state"},
        )
        self.cb_scenario.addItem(
            "Septic shock response",
            {"scenario_id": "sepsis_response", "mode": "steady_state"},
        )
        h_scenario.addWidget(self.cb_scenario, stretch=1)
        self.scenario_container.setVisible(False)
        l_ui.addWidget(self.scenario_container)
        
        # Connect tutorial mode toggle to show/hide scenario selection
        self.rb_tutorial.toggled.connect(self._on_tutorial_toggled)
        self.cb_scenario.currentTextChanged.connect(self._sync_start_mode_from_tutorial)
        self._on_tutorial_toggled(self.rb_tutorial.isChecked())
        
        right_col.addWidget(gb_ui)
        right_col.addWidget(self.gb_scenario)

        # 4. Model Configuration (Collapsible-style)
        gb_models = QGroupBox("Advanced settings")
        model_layout = QGridLayout(gb_models)
        model_layout.setSpacing(8)
        model_layout.setContentsMargins(12, 16, 12, 12)
        model_layout.setColumnStretch(1, 1)
        model_layout.setColumnStretch(3, 1)
        
        self.cb_prop_model = QComboBox()
        self.cb_prop_model.addItems(["Marsh", "Schnider", "Eleveld"])
        self.cb_prop_model.setCurrentText("Eleveld")
        model_layout.addWidget(QLabel("Propofol PK:"), 0, 0)
        model_layout.addWidget(self.cb_prop_model, 0, 1)
        
        self.cb_nore_model = QComboBox()
        self.cb_nore_model.addItems(["Beloeil", "Oualha", "Li"])
        self.cb_nore_model.setCurrentText("Li")
        model_layout.addWidget(QLabel("Norepinephrine PK:"), 1, 0)
        model_layout.addWidget(self.cb_nore_model, 1, 1)

        self.cb_epi_model = QComboBox()
        self.cb_epi_model.addItems(["Clutter", "Abboud", "Oualha"])
        self.cb_epi_model.setCurrentText("Clutter")
        model_layout.addWidget(QLabel("Epinephrine PK:"), 2, 0)
        model_layout.addWidget(self.cb_epi_model, 2, 1)

        self.cb_bis_model = QComboBox()
        self.cb_bis_model.addItems(["Bouillon", "Eleveld", "Fuentes", "Yumuk"])
        self.cb_bis_model.setCurrentText("Bouillon")
        model_layout.addWidget(QLabel("BIS model:"), 0, 2)
        model_layout.addWidget(self.cb_bis_model, 0, 3)

        self.cb_loc_model = QComboBox()
        self.cb_loc_model.addItems(["Kern", "Mertens", "Johnson"])
        self.cb_loc_model.setCurrentText("Kern")
        model_layout.addWidget(QLabel("LOC model:"), 1, 2)
        model_layout.addWidget(self.cb_loc_model, 1, 3)

        self.cb_fidelity = QComboBox()
        self.cb_fidelity.addItem("Clinical realism (tuned)", "clinical")
        self.cb_fidelity.addItem("Literature fidelity", "literature")
        self.cb_fidelity.setToolTip(
            "Toggle tuned parameters vs literature-derived values where noted in the models."
        )
        model_layout.addWidget(QLabel("Model fidelity:"), 2, 2)
        model_layout.addWidget(self.cb_fidelity, 2, 3)
        
        # 5. Simulation Rules
        gb_rules = QGroupBox("Simulation rules")
        rules_layout = QVBoxLayout(gb_rules)
        rules_layout.setContentsMargins(12, 16, 12, 12)
        
        self.cb_art_line = QCheckBox("Enable arterial line (continuous BP)")
        self.cb_art_line.setChecked(True)
        self.cb_art_line.setToolTip("If enabled, shows continuous arterial pressure waveform and values.")
        rules_layout.addWidget(self.cb_art_line)

        self.cb_death_detector = QCheckBox("Enable patient death detector")
        self.cb_death_detector.setToolTip("If enabled, extreme vitals (e.g. cardiac arrest) will end the simulation.")
        rules_layout.addWidget(self.cb_death_detector)
        
        right_col.addWidget(gb_rules)
        right_col.addStretch()

        layout.addWidget(gb_models)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_button = buttons.button(QDialogButtonBox.Ok)
        cancel_button = buttons.button(QDialogButtonBox.Cancel)
        ok_button.setText("Start simulation")
        ok_button.setStyleSheet(
            get_button_style(variant="primary", padding="8px 18px", min_width=140)
        )
        cancel_button.setStyleSheet(
            get_button_style(outlined=True, variant="neutral", padding="8px 18px", min_width=100)
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def _on_tutorial_toggled(self, checked):
        """Show/hide scenario selection when tutorial mode is toggled."""
        self.scenario_container.setVisible(checked)
        if hasattr(self, "gb_scenario"):
            self.gb_scenario.setEnabled(not checked)
        if checked:
            # Align induction/emergence tutorial with the current start mode
            if self.cb_scenario.currentIndex() in (0, 1):
                self.cb_scenario.setCurrentIndex(1 if self.rb_maint.isChecked() else 0)
            self._sync_start_mode_from_tutorial()

    def _sync_start_mode_from_tutorial(self):
        """Align disabled start mode with selected tutorial scenario."""
        if not hasattr(self, "cb_scenario"):
            return
        text = self.cb_scenario.currentText()
        data = self.cb_scenario.currentData() or {}
        if data.get("mode") == "steady_state":
            self.rb_maint.setChecked(True)
        else:
            self.rb_awake.setChecked(True)
    
    def accept(self):
        # Determine selected scenario
        scenario_id = None
        if self.rb_tutorial.isChecked():
            scenario_data = self.cb_scenario.currentData() or {}
            scenario_id = scenario_data.get("scenario_id")
            if scenario_data.get("mode") == "steady_state":
                self.rb_maint.setChecked(True)
            elif scenario_data.get("mode") == "awake":
                self.rb_awake.setChecked(True)
        
        renal_text = self.cb_renal.currentText()
        hepatic_text = self.cb_hepatic.currentText()
        self.result_data = {
            'age': self.sb_age.value(),
            'weight': self.sb_weight.value(),
            'height': self.sb_height.value(),
            'sex': self.cb_sex.currentText().lower(),
            'baseline_hb': self.sb_hgb.value(),
            'renal_function': self.cb_renal.currentData(),
            'hepatic_function': self.cb_hepatic.currentData(),
            'renal_status': renal_text.split(" (")[0],
            'hepatic_status': hepatic_text.split(" (")[0],
            'mode': 'steady_state' if self.rb_maint.isChecked() else 'awake',
            'maint_type': 'balanced' if 'Inhalational' in self.cb_maint_type.currentText() else 'tiva',
            'tutorial_mode': self.rb_tutorial.isChecked(),
            'scenario_id': scenario_id,
            'pk_model_propofol': self.cb_prop_model.currentText(),
            'pk_model_nore': self.cb_nore_model.currentText(),
            'pk_model_epi': self.cb_epi_model.currentText(),
            'bis_model': self.cb_bis_model.currentText(),
            'loc_model': self.cb_loc_model.currentText(),
            'fidelity_mode': self.cb_fidelity.currentData(),
            'enable_death_detector': self.cb_death_detector.isChecked(),
            'arterial_line_enabled': self.cb_art_line.isChecked()
        }
        super().accept()
