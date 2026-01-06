"""
ScenarioOverlay - Data-driven tutorial/scenario overlay widget.

Refactored from the original TutorialOverlay to use the scenario system.
Scenarios are defined in ui/scenarios/ as data classes.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QProgressBar
from PySide6.QtCore import Qt

from .scenarios import (
    Scenario,
    SCENARIO_BUILDERS,
    create_induction_balanced,
    create_induction_tiva,
    create_emergence,
)
from .styles import COLORS, get_overlay_style, get_button_style, STYLE_PROGRESSBAR


class ScenarioOverlay(QFrame):
    """
    Overlay widget to guide user through scenario steps.
    
    This is a generic overlay that works with any Scenario object.
    The scenario defines the steps, instructions, and requirements.
    """
    
    def __init__(self, scenario: Scenario, parent=None):
        super().__init__(parent)
        self.scenario = scenario
        self.requirements_met = False
        self.current_step = 0
        
        self.setStyleSheet(get_overlay_style())
        # Give the instruction text enough vertical space to avoid truncation.
        self.setFixedHeight(260)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        
        # Header with step and progress
        header = QHBoxLayout()
        header.setSpacing(16)
        
        title = f"Tutorial: {self.scenario.name}"
        self.lbl_step = QLabel(title)
        self.lbl_step.setStyleSheet(f"font-weight: 700; font-size: 14px; color: {COLORS['primary']}; font-family: Arial;")
        header.addWidget(self.lbl_step)
        
        self.progress = QProgressBar()
        self.progress.setStyleSheet(STYLE_PROGRESSBAR)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        header.addWidget(self.progress, stretch=1)
        
        self.lbl_progress_text = QLabel("")
        self.lbl_progress_text.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px; font-family: Arial;")
        header.addWidget(self.lbl_progress_text)
        
        layout.addLayout(header)
        
        # Instruction text area
        self.lbl_instruction = QLabel("")
        self.lbl_instruction.setStyleSheet(f"font-size: 14px; line-height: 1.4; color: {COLORS['text']}; font-family: Arial;")
        self.lbl_instruction.setWordWrap(True)
        self.lbl_instruction.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.lbl_instruction, stretch=1)
        
        # Bottom bar with status and Next button
        bottom_bar = QHBoxLayout()
        
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(f"font-size: 12px; color: {COLORS['warning']}; font-weight: 600; font-family: Arial;")
        bottom_bar.addWidget(self.lbl_status)
        
        bottom_bar.addStretch()
        
        self.btn_next = QPushButton("Next step")
        self.btn_next.setStyleSheet(get_button_style(variant="success", padding="8px 24px"))
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.on_next_clicked)
        bottom_bar.addWidget(self.btn_next)
        
        layout.addLayout(bottom_bar)
        
        # Display initial step
        self._update_progress()
        self._update_instruction()
        
    @property
    def steps(self):
        """Compatibility property for tests that access steps list."""
        return [step.id for step in self.scenario.steps]
        
    def _update_progress(self):
        """Update progress indicator."""
        total = len(self.scenario)
        if total == 0:
            self.progress.setValue(0)
            return

        current = min(self.current_step + 1, total)
        
        # Update progress bar
        # If current_step == total, display 100% progress.
        if self.current_step >= total:
            percent = 100
        else:
            # Show progress to next step
            percent = int((self.current_step / total) * 100)
            
        self.progress.setValue(percent)
        self.lbl_progress_text.setText(f"Step {current} of {total}")
    
    def _update_instruction(self):
        """Update instruction text for current step."""
        if self.current_step >= len(self.scenario):
            self.lbl_instruction.setText(
                "<b>Tutorial complete!</b><br>"
                "You have successfully completed the scenario."
            )
        else:
            step = self.scenario[self.current_step]
            self.lbl_instruction.setText(step.instruction)
            
            # Call on_enter callback if present
            if step.on_enter:
                pass 

    def get_instruction_text(self, step_id):
        """Get instruction text for a step by ID. For backward compatibility."""
        if not step_id:
            return "<b>Tutorial complete!</b><br>You have successfully completed the anesthesia sequence."
        
        for step in self.scenario.steps:
            if step.id == step_id:
                return step.instruction
        return f"Unknown step: {step_id}"

    def check_requirements(self, engine):
        """Check if current step requirements are met. Returns True/False and status message."""
        if self.current_step >= len(self.scenario):
            return True, ""
            
        step = self.scenario[self.current_step]
        return step.check_requirements(engine)

    def update_state(self, engine):
        """Called each frame. Check requirements and update UI."""
        if self.current_step >= len(self.scenario):
            self.lbl_instruction.setText(
                "<b>Tutorial complete!</b><br>"
                "You have successfully completed the scenario."
            )
            self.btn_next.setEnabled(False)
            self.btn_next.setText("Complete")
            self.lbl_status.setText("")
            return

        step = self.scenario[self.current_step]
        
        self.lbl_instruction.setText(step.instruction)
        
        met, status_msg = step.check_requirements(engine)
        self.requirements_met = bool(met)
        
        self.btn_next.setEnabled(self.requirements_met)
        self.lbl_status.setText(status_msg)
        
    def on_next_clicked(self):
        """Handle Next button click."""
        if self.requirements_met:
            self.advance_step()
            
    def advance_step(self):
        """Move to next step."""
        self.current_step += 1
        self._update_progress()
        if self.current_step < len(self.scenario):
            self._update_instruction()
            self.requirements_met = False
            self.btn_next.setEnabled(False)
            self.lbl_status.setText("")
        else:
            self.lbl_instruction.setText(
                "<b>Tutorial complete!</b><br>"
                "You have successfully completed the scenario."
            )
            self.btn_next.setEnabled(False)
            self.btn_next.setText("Complete")
            self.lbl_status.setText("")
            
    def click_next(self):
        """Programmatic click for testing."""
        if self.requirements_met:
            self.on_next_clicked()


# Factory function for backward compatibility
def TutorialOverlay(mode="awake", maint_type="tiva", parent=None):
    """
    Factory function for backward compatibility.
    
    Creates the appropriate scenario based on mode and maint_type.
    
    Args:
        mode: "awake" for induction, "steady_state" for emergence
        maint_type: "balanced" or "tiva"
        parent: Qt parent widget
    """
    # Simple heuristic mapping; expand if config passes specific IDs.
    if mode in SCENARIO_BUILDERS:
        scenario = SCENARIO_BUILDERS[mode]()
    elif mode == "awake":
        if "balanced" in maint_type.lower():
            scenario = create_induction_balanced()
        else:
            scenario = create_induction_tiva()
    else:
        scenario = create_emergence(maint_type)
    
    return ScenarioOverlay(scenario, parent)
