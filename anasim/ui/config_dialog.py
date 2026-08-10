from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from anasim.patient.domain import (
    AGE_RANGE_YEARS,
    HEIGHT_RANGE_CM,
    HEMOGLOBIN_RANGE_G_DL,
    WEIGHT_RANGE_KG,
)
from anasim.patient.patient import Patient

from .scenarios import SCENARIO_REGISTRY
from .styles import COLORS, get_button_style, get_dialog_style, get_frame_style


class SimulationSetupDialog(QDialog):
    """Collect the patient and session settings for a new simulation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start an AnaSim session")
        self.setModal(True)
        self.result_data = None
        self.setMinimumSize(900, 680)

        self.setStyleSheet(get_dialog_style())

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 12, 16, 16)

        header = QFrame()
        header.setObjectName("styledSurface")
        header.setStyleSheet(
            get_frame_style(bg_color=COLORS["card"], border_color=COLORS["border"])
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 12, 10)

        lbl_title = QLabel("AnaSim")
        lbl_title.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {COLORS['primary']};"
        )
        header_layout.addWidget(lbl_title)

        lbl_subtitle = QLabel("Adult anesthesia and physiology simulation")
        lbl_subtitle.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text_secondary']};"
        )
        header_layout.addWidget(lbl_subtitle)
        header_layout.addStretch()

        lbl_use = QLabel("Education and training only")
        lbl_use.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {COLORS['warning']};"
        )
        header_layout.addWidget(lbl_use)

        layout.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(12)
        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        body.addLayout(left_col, 1)
        body.addLayout(right_col, 1)
        layout.addLayout(body)

        gb_patient = QGroupBox("Patient")
        form = QFormLayout(gb_patient)
        form.setSpacing(8)
        form.setContentsMargins(12, 16, 12, 12)

        self.sb_age = QSpinBox()
        self.sb_age.setRange(int(AGE_RANGE_YEARS[0]), int(AGE_RANGE_YEARS[1]))
        self.sb_age.setValue(40)
        self.sb_age.setSuffix(" years")
        form.addRow("Age:", self.sb_age)

        self.sb_weight = QDoubleSpinBox()
        self.sb_weight.setRange(*WEIGHT_RANGE_KG)
        self.sb_weight.setValue(70.0)
        self.sb_weight.setSuffix(" kg")
        form.addRow("Weight:", self.sb_weight)

        self.sb_height = QDoubleSpinBox()
        self.sb_height.setRange(*HEIGHT_RANGE_CM)
        self.sb_height.setValue(170.0)
        self.sb_height.setSuffix(" cm")
        form.addRow("Height:", self.sb_height)

        self.cb_sex = QComboBox()
        self.cb_sex.addItems(["Male", "Female"])
        form.addRow("Sex:", self.cb_sex)

        self.sb_hgb = QDoubleSpinBox()
        self.sb_hgb.setRange(*HEMOGLOBIN_RANGE_G_DL)
        self.sb_hgb.setSingleStep(0.1)
        self.sb_hgb.setValue(13.5)
        self.sb_hgb.setSuffix(" g/dL")
        form.addRow("Hemoglobin:", self.sb_hgb)

        left_col.addWidget(gb_patient)

        gb_organ = QGroupBox("Organ function")
        organ_form = QFormLayout(gb_organ)
        organ_form.setSpacing(8)
        organ_form.setContentsMargins(12, 16, 12, 12)

        self.cb_renal = QComboBox()
        self.cb_renal.addItem("Normal (eGFR ≥ 90)", 1.0)
        self.cb_renal.addItem("Mild (eGFR 60 to 89)", 0.8)
        self.cb_renal.addItem("Moderate (eGFR 30 to 59)", 0.6)
        self.cb_renal.addItem("Severe (eGFR < 30)", 0.4)
        organ_form.addRow("Renal:", self.cb_renal)

        self.cb_hepatic = QComboBox()
        self.cb_hepatic.addItem("Normal (no cirrhosis)", 1.0)
        self.cb_hepatic.addItem("Mild (Child-Pugh A)", 0.9)
        self.cb_hepatic.addItem("Moderate (Child-Pugh B)", 0.7)
        self.cb_hepatic.addItem("Severe (Child-Pugh C)", 0.5)
        organ_form.addRow("Hepatic:", self.cb_hepatic)

        left_col.addWidget(gb_organ)

        self.gb_scenario = QGroupBox("Initial state")
        l_scen = QVBoxLayout(self.gb_scenario)
        l_scen.setSpacing(10)
        l_scen.setContentsMargins(12, 16, 12, 12)

        lbl_start_hint = QLabel(
            "Used for open simulation. Guided scenarios define their own initial state."
        )
        lbl_start_hint.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        lbl_start_hint.setWordWrap(True)
        l_scen.addWidget(lbl_start_hint)

        self.rb_awake = QRadioButton("Awake before induction")
        self.rb_maint = QRadioButton("Anesthetized maintenance")
        self.rb_maint.setChecked(True)

        grp = QButtonGroup(self)
        grp.addButton(self.rb_awake)
        grp.addButton(self.rb_maint)

        l_scen.addWidget(self.rb_awake)
        l_scen.addWidget(self.rb_maint)

        h_maint = QHBoxLayout()
        lbl_maint = QLabel("Maintenance technique:")
        lbl_maint.setStyleSheet(f"color: {COLORS['text_secondary']};")
        h_maint.addWidget(lbl_maint)
        self.cb_maint_type = QComboBox()
        self.cb_maint_type.addItems(["TIVA (propofol)", "Inhalational (sevoflurane)"])
        h_maint.addWidget(self.cb_maint_type)
        h_maint.addStretch()
        l_scen.addLayout(h_maint)

        left_col.addStretch()

        gb_ui = QGroupBox("Session type")
        l_ui = QVBoxLayout(gb_ui)
        l_ui.setSpacing(8)
        l_ui.setContentsMargins(12, 16, 12, 12)

        self.rb_advanced = QRadioButton("Open simulation")
        self.rb_tutorial = QRadioButton("Guided scenario")
        self.rb_advanced.setChecked(True)

        lbl_tutorial_info = QLabel(
            "Open simulation gives direct control. Guided scenarios provide ordered objectives."
        )
        lbl_tutorial_info.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        lbl_tutorial_info.setWordWrap(True)

        grp_ui = QButtonGroup(self)
        grp_ui.addButton(self.rb_advanced)
        grp_ui.addButton(self.rb_tutorial)

        l_ui.addWidget(self.rb_advanced)
        l_ui.addWidget(self.rb_tutorial)
        l_ui.addWidget(lbl_tutorial_info)

        self.scenario_container = QFrame()
        self.scenario_container.setStyleSheet("background: transparent;")
        h_scenario = QHBoxLayout(self.scenario_container)
        h_scenario.setContentsMargins(0, 4, 0, 0)
        lbl_scenario = QLabel("Scenario:")
        lbl_scenario.setStyleSheet(f"color: {COLORS['text_secondary']};")
        h_scenario.addWidget(lbl_scenario)
        self.cb_scenario = QComboBox()
        for spec in SCENARIO_REGISTRY:
            self.cb_scenario.addItem(
                spec.label,
                {
                    "scenario_id": spec.id,
                    "mode": spec.start_mode,
                    "maint_type": spec.maint_type,
                },
            )
        h_scenario.addWidget(self.cb_scenario, stretch=1)
        self.scenario_container.setVisible(False)
        l_ui.addWidget(self.scenario_container)

        self.rb_tutorial.toggled.connect(self._on_tutorial_toggled)
        self.cb_scenario.currentTextChanged.connect(self._sync_start_mode_from_tutorial)
        self._on_tutorial_toggled(self.rb_tutorial.isChecked())

        right_col.addWidget(gb_ui)
        right_col.addWidget(self.gb_scenario)

        gb_models = QGroupBox("Model selection")
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

        gb_rules = QGroupBox("Monitoring and endpoint")
        rules_layout = QVBoxLayout(gb_rules)
        rules_layout.setContentsMargins(12, 16, 12, 12)

        self.cb_art_line = QCheckBox("Display continuous arterial pressure")
        self.cb_art_line.setChecked(True)
        self.cb_art_line.setToolTip(
            "Show the arterial pressure waveform and continuous pressure values."
        )
        rules_layout.addWidget(self.cb_art_line)

        self.cb_death_detector = QCheckBox("End session when death criteria are met")
        self.cb_death_detector.setToolTip(
            "Stop simulation after configured extreme physiology persists."
        )
        rules_layout.addWidget(self.cb_death_detector)

        right_col.addWidget(gb_rules)
        right_col.addStretch()

        self.cb_show_models = QCheckBox("Show advanced model selection")
        self.cb_show_models.setToolTip(
            "Choose alternate pharmacokinetic and pharmacodynamic models."
        )
        self.gb_models = gb_models
        self.cb_show_models.toggled.connect(self._on_show_models_toggled)
        layout.addWidget(self.cb_show_models)
        layout.addWidget(gb_models)
        gb_models.setVisible(False)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_button = buttons.button(QDialogButtonBox.Ok)
        cancel_button = buttons.button(QDialogButtonBox.Cancel)
        ok_button.setText("Start simulation")
        ok_button.setStyleSheet(
            get_button_style(variant="primary", padding="8px 18px", min_width=140)
        )
        cancel_button.setStyleSheet(
            get_button_style(
                outlined=True, variant="neutral", padding="8px 18px", min_width=100
            )
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_tutorial_toggled(self, checked):
        """Show/hide scenario selection when tutorial mode is toggled."""
        self.scenario_container.setVisible(checked)
        self.gb_scenario.setEnabled(not checked)
        if checked:
            self._sync_start_mode_from_tutorial()

    def _on_show_models_toggled(self, checked):
        """Reveal optional model choices without crowding the default workflow."""
        self.gb_models.setVisible(checked)
        self.adjustSize()

    def _sync_start_mode_from_tutorial(self):
        """Align disabled start mode with selected tutorial scenario."""
        data = self.cb_scenario.currentData()
        if data["mode"] == "steady_state":
            self.rb_maint.setChecked(True)
        else:
            self.rb_awake.setChecked(True)
        if data["maint_type"] == "balanced":
            self.cb_maint_type.setCurrentText("Inhalational (sevoflurane)")
        else:
            self.cb_maint_type.setCurrentText("TIVA (propofol)")

    def accept(self):
        scenario_id = None
        if self.rb_tutorial.isChecked():
            self._sync_start_mode_from_tutorial()
            scenario_id = self.cb_scenario.currentData()["scenario_id"]

        patient_data = {
            "age": self.sb_age.value(),
            "weight": self.sb_weight.value(),
            "height": self.sb_height.value(),
            "sex": self.cb_sex.currentText().lower(),
            "baseline_hb": self.sb_hgb.value(),
            "renal_function": self.cb_renal.currentData(),
            "hepatic_function": self.cb_hepatic.currentData(),
        }
        try:
            Patient(**patient_data)
        except ValueError as exc:
            self.result_data = None
            QMessageBox.warning(self, "Unsupported patient", str(exc))
            return

        self.result_data = {
            **patient_data,
            "mode": "steady_state" if self.rb_maint.isChecked() else "awake",
            "maint_type": "balanced"
            if "Inhalational" in self.cb_maint_type.currentText()
            else "tiva",
            "tutorial_mode": self.rb_tutorial.isChecked(),
            "scenario_id": scenario_id,
            "pk_model_propofol": self.cb_prop_model.currentText(),
            "pk_model_nore": self.cb_nore_model.currentText(),
            "pk_model_epi": self.cb_epi_model.currentText(),
            "bis_model": self.cb_bis_model.currentText(),
            "loc_model": self.cb_loc_model.currentText(),
            "enable_death_detector": self.cb_death_detector.isChecked(),
            "arterial_line_enabled": self.cb_art_line.isChecked(),
        }
        super().accept()
