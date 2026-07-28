"""Guided scenario workflow shown above the simulator."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from .scenarios import Scenario
from .styles import (
    COLORS,
    STYLE_PROGRESSBAR,
    get_button_style,
    get_overlay_style,
)


class ScenarioOverlay(QFrame):
    """Present one scenario objective at a time and track completion."""

    navigate_requested = Signal(str)

    def __init__(self, scenario: Scenario, parent=None):
        super().__init__(parent)
        self.scenario = scenario
        self.current_step = 0
        self.requirements_met = False

        self.setObjectName("scenarioOverlay")
        self.setStyleSheet(get_overlay_style())
        self.setFixedHeight(178)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(12)

        self.lbl_scenario = QLabel(scenario.name)
        self.lbl_scenario.setStyleSheet(
            f"color: {COLORS['primary']}; font-size: 11px; font-weight: 650;"
        )
        header.addWidget(self.lbl_scenario)

        self.progress = QProgressBar()
        self.progress.setStyleSheet(STYLE_PROGRESSBAR)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        header.addWidget(self.progress, stretch=1)

        self.lbl_progress = QLabel("")
        self.lbl_progress.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10px;"
        )
        header.addWidget(self.lbl_progress)
        layout.addLayout(header)

        objective = QHBoxLayout()
        objective.setSpacing(12)

        self.lbl_step_number = QLabel("")
        self.lbl_step_number.setAlignment(Qt.AlignCenter)
        self.lbl_step_number.setFixedSize(38, 38)
        self.lbl_step_number.setStyleSheet(
            f"color: {COLORS['primary']}; background-color: {COLORS['control']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 5px; "
            "font-size: 15px; font-weight: 700;"
        )
        objective.addWidget(self.lbl_step_number, alignment=Qt.AlignTop)

        copy = QVBoxLayout()
        copy.setSpacing(3)
        self.lbl_title = QLabel("")
        self.lbl_title.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 15px; font-weight: 650;"
        )
        copy.addWidget(self.lbl_title)

        self.lbl_instruction = QLabel("")
        self.lbl_instruction.setTextFormat(Qt.RichText)
        self.lbl_instruction.setWordWrap(True)
        self.lbl_instruction.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_instruction.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        copy.addWidget(self.lbl_instruction)
        objective.addLayout(copy, stretch=1)
        layout.addLayout(objective, stretch=1)

        footer = QHBoxLayout()
        footer.setSpacing(8)

        self.lbl_status_icon = QLabel("●")
        self.lbl_status_icon.setFixedWidth(12)
        footer.addWidget(self.lbl_status_icon)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(
            f"color: {COLORS['warning']}; font-size: 11px; font-weight: 600;"
        )
        footer.addWidget(self.lbl_status, stretch=1)

        self.btn_target = QPushButton("")
        self.btn_target.setStyleSheet(
            get_button_style(
                variant="neutral",
                outlined=True,
                padding="6px 12px",
                min_width=94,
            )
        )
        self.btn_target.clicked.connect(self._navigate_to_target)
        footer.addWidget(self.btn_target)

        self.btn_next = QPushButton("Continue")
        self.btn_next.setStyleSheet(
            get_button_style(
                variant="success",
                padding="6px 16px",
                min_width=104,
            )
        )
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.on_next_clicked)
        footer.addWidget(self.btn_next)
        layout.addLayout(footer)

        self._show_current_step()

    def _show_current_step(self):
        """Refresh labels and actions for the current objective."""
        if self.current_step >= len(self.scenario):
            self._show_completion()
            return

        step = self.scenario[self.current_step]
        self.lbl_step_number.setText(str(self.current_step + 1))
        self.lbl_title.setText(step.title)
        self.lbl_instruction.setText(step.instruction)
        self.lbl_progress.setText(
            f"{self.current_step + 1} of {len(self.scenario)}"
        )
        self.btn_target.setVisible(step.target_tab is not None)
        if step.target_tab is not None:
            self.btn_target.setText(f"Open {step.target_tab.lower()}")
        self.requirements_met = False
        self._set_requirement_state(False, "Complete the objective to continue")

    def _show_completion(self):
        self.lbl_step_number.setText("✓")
        self.lbl_title.setText("Scenario complete")
        self.lbl_instruction.setText(self.scenario.description)
        self.lbl_progress.setText(f"{len(self.scenario)} of {len(self.scenario)}")
        self.progress.setValue(100)
        self.lbl_status_icon.setText("✓")
        self.lbl_status_icon.setStyleSheet(f"color: {COLORS['success']};")
        self.lbl_status.setText("All objectives complete")
        self.lbl_status.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 11px; font-weight: 600;"
        )
        self.btn_target.hide()
        self.btn_next.setText("Complete")
        self.btn_next.setEnabled(False)

    def _update_progress(self):
        total = len(self.scenario)
        if total == 0:
            self.progress.setValue(100)
            return
        completed = self.current_step + int(self.requirements_met)
        self.progress.setValue(round(100 * completed / total))

    def _set_requirement_state(self, met: bool, status: str):
        self.requirements_met = met
        if met:
            self.lbl_status_icon.setText("✓")
            self.lbl_status_icon.setStyleSheet(f"color: {COLORS['success']};")
            self.lbl_status.setText("Objective complete")
            self.lbl_status.setStyleSheet(
                f"color: {COLORS['success']}; font-size: 11px; font-weight: 600;"
            )
            is_last = self.current_step == len(self.scenario) - 1
            self.btn_next.setText("Finish scenario" if is_last else "Continue")
            self.btn_next.setEnabled(True)
        else:
            self.lbl_status_icon.setText("●")
            self.lbl_status_icon.setStyleSheet(f"color: {COLORS['warning']};")
            self.lbl_status.setText(status or "Complete the objective to continue")
            self.lbl_status.setStyleSheet(
                f"color: {COLORS['warning']}; font-size: 11px; font-weight: 600;"
            )
            self.btn_next.setText("Continue")
            self.btn_next.setEnabled(False)
        self._update_progress()

    def check_requirements(self, engine):
        """Return the current objective's completion state and feedback."""
        if self.current_step >= len(self.scenario):
            return True, ""
        return self.scenario[self.current_step].check_requirements(engine)

    def update_state(self, engine):
        """Update objective feedback from the latest simulation state."""
        if self.current_step >= len(self.scenario):
            return
        met, status = self.check_requirements(engine)
        self._set_requirement_state(bool(met), status)

    def _navigate_to_target(self):
        if self.current_step >= len(self.scenario):
            return
        target = self.scenario[self.current_step].target_tab
        if target is not None:
            self.navigate_requested.emit(target)

    def on_next_clicked(self):
        if self.requirements_met:
            self.advance_step()

    def advance_step(self):
        """Advance after the current objective has been completed."""
        self.current_step += 1
        self._show_current_step()
