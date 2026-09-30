"""Guided scenario workflow shown above the simulator."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .scenarios import Scenario
from .styles import (
    COLORS,
    get_button_style,
    get_overlay_style,
)


class ScenarioOverlay(QFrame):
    """Present one scenario objective at a time and track completion."""

    navigate_requested = Signal(str)

    def __init__(self, scenario: Scenario, engine, parent=None):
        super().__init__(parent)
        self.scenario = scenario
        self.engine = engine
        self.current_step = 0
        self.requirements_met = False
        self._last_requirement_display = None

        self.setObjectName("scenarioOverlay")
        self.setStyleSheet(get_overlay_style())
        self.setMinimumHeight(150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(12)

        self.lbl_scenario = QLabel(scenario.name)
        self.lbl_scenario.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; font-weight: 650;"
        )
        header.addWidget(self.lbl_scenario)

        header.addStretch()

        self.lbl_progress = QLabel("")
        self.lbl_progress.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10px;"
        )
        header.addWidget(self.lbl_progress)
        layout.addLayout(header)

        objective = QHBoxLayout()
        objective.setSpacing(12)

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

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; font-weight: 400;"
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
                variant="neutral",
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
        # Scope action requirements to this objective: only what the learner
        # does from now on can complete it.
        self.engine.actions.begin_step(step.id, self.engine.state.time)
        self.lbl_title.setText(step.title)
        self.lbl_instruction.setText(step.instruction)
        self.lbl_progress.setText(
            f"Step {self.current_step + 1} of {len(self.scenario)}"
        )
        self.btn_target.setVisible(step.target_tab is not None)
        if step.target_tab is not None:
            self.btn_target.setText(f"Open {step.target_tab.lower()}")
        self.requirements_met = False
        self._last_requirement_display = None
        self._set_requirement_state(False, "Complete the objective to continue")

    def _show_completion(self):
        self.lbl_title.setText("Scenario complete")
        self.lbl_instruction.setText(self.scenario.description)
        self.lbl_progress.clear()
        self.lbl_status.clear()
        self.btn_target.hide()
        self.btn_next.setText("Complete")
        self.btn_next.setEnabled(False)

    def _set_requirement_state(self, met: bool, status: str):
        self.requirements_met = met
        display_state = (self.current_step, met, status)
        if display_state == self._last_requirement_display:
            return
        self._last_requirement_display = display_state
        if met:
            self.lbl_status.clear()
            is_last = self.current_step == len(self.scenario) - 1
            self.btn_next.setText("Finish scenario" if is_last else "Continue")
            self.btn_next.setEnabled(True)
        else:
            self.lbl_status.setText(status or "Complete the objective to continue")
            self.btn_next.setText("Continue")
            self.btn_next.setEnabled(False)

    def check_requirements(self):
        """Return the current objective's completion state and feedback."""
        if self.current_step >= len(self.scenario):
            return True, ""
        return self.scenario[self.current_step].check_requirements(self.engine)

    def update_state(self):
        """Update objective feedback from the latest simulation state."""
        if self.current_step >= len(self.scenario):
            return
        met, status = self.check_requirements()
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
