"""
Base classes for the scenario/tutorial system.

Scenarios are data-driven definitions of guided learning sequences.
Each scenario contains a list of steps with instructions and requirements.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple


@dataclass
class ScenarioStep:
    """
    A single step in a scenario.
    
    Attributes:
        id: Unique step identifier (e.g., "APPLY_MASK")
        title: Display title for the step
        instruction: Rich HTML instruction text
        check_requirements: Function (engine) -> (bool, str) that checks
            if the step requirements are met. Returns (is_met, status_message).
        on_enter: Optional callback when step starts (for automation/setup)
    """
    id: str
    title: str
    instruction: str
    check_requirements: Callable[[object], Tuple[bool, str]]
    on_enter: Optional[Callable[[object], None]] = None


@dataclass
class Scenario:
    """
    A complete scenario/tutorial definition.
    
    Attributes:
        id: Unique scenario identifier (e.g., "induction_balanced")
        name: Display name for UI
        icon: Short label or icon text
        description: Brief description of what this scenario teaches
        steps: List of ScenarioStep objects
    """
    id: str
    name: str
    icon: str
    description: str
    steps: List[ScenarioStep] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def __getitem__(self, idx: int) -> ScenarioStep:
        return self.steps[idx]


# Requirement check helper functions
def require_airway(airway_type: str) -> Callable:
    """Create a requirement checker for airway type."""
    from core.state import AirwayType
    type_map = {"None": AirwayType.NONE, "Mask": AirwayType.MASK, "ETT": AirwayType.ETT}
    
    def check(engine) -> Tuple[bool, str]:
        target = type_map.get(airway_type, AirwayType.NONE)
        met = engine.state.airway_mode == target
        return met, "" if met else f"Select '{airway_type}' airway device"
    return check


def require_fgf_preox() -> Callable:
    """Check FGF set for preoxygenation (O2 >= 8, Air < 1)."""
    def check(engine) -> Tuple[bool, str]:
        o2_ok = engine.circuit.fgf_o2 >= 8.0
        air_ok = engine.circuit.fgf_air < 1.0
        if o2_ok and air_ok:
            return True, ""
        msgs = []
        if not o2_ok: msgs.append(f"O₂: {engine.circuit.fgf_o2:.1f}/8+ L/min")
        if not air_ok: msgs.append(f"Air: {engine.circuit.fgf_air:.1f}/0 L/min")
        return False, "" + ", ".join(msgs)
    return check


def require_propofol_cp(threshold: float = 2.0) -> Callable:
    """Check propofol plasma concentration."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.propofol_cp > threshold
        return met, "" if met else f"Propofol Cp: {engine.state.propofol_cp:.1f}/{threshold}+ µg/mL"
    return check


def require_bis_below(threshold: float) -> Callable:
    """Check BIS below threshold."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.bis < threshold
        return met, "" if met else f"BIS: {engine.state.bis:.0f}/<{threshold:.0f}"
    return check


def require_bis_above(threshold: float) -> Callable:
    """Check BIS above threshold."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.bis > threshold
        return met, "" if met else f"BIS: {engine.state.bis:.0f}/{threshold:.0f}+"
    return check


def require_ventilation_on() -> Callable:
    """Check that ventilation is active."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.vent.is_on
        return met, "" if met else "Turn ON Bag-Mask ventilation"
    return check


def require_rocuronium_cp(threshold: float = 0.5) -> Callable:
    """Check rocuronium plasma concentration."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.roc_cp > threshold
        return met, "" if met else f"Rocuronium Cp: {engine.state.roc_cp:.2f}/{threshold}+ µg/mL"
    return check


def require_tof_below(threshold: float = 25) -> Callable:
    """Check TOF below threshold."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.tof <= threshold
        return met, "" if met else f"TOF: {engine.state.tof:.0f}%/≤{threshold:.0f}%"
    return check


def require_etco2_above(threshold: float = 20) -> Callable:
    """Check EtCO2 above threshold."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.etco2 > threshold
        return met, "" if met else f"EtCO₂: {engine.state.etco2:.0f}/>{threshold:.0f} mmHg"
    return check


def require_mac_above(threshold: float = 0.5) -> Callable:
    """Check MAC above threshold."""
    def check(engine) -> Tuple[bool, str]:
        met = engine.state.mac > threshold
        return met, "" if met else f"MAC: {engine.state.mac:.2f}/{threshold}+"
    return check


def require_remi_running() -> Callable:
    """Check remifentanil infusion is running."""
    def check(engine) -> Tuple[bool, str]:
        running = (engine.remi_rate_ug_sec > 0) or (engine.tci_remi and engine.tci_remi.target > 0)
        return running, "" if running else "Start Remifentanil TCI or infusion"
    return check


def require_propofol_infusion_running() -> Callable:
    """Check propofol infusion is running."""
    def check(engine) -> Tuple[bool, str]:
        running = (engine.propofol_rate_mg_sec > 0) or (engine.tci_prop and engine.tci_prop.target > 0)
        return running, "" if running else "Propofol infusion not running"
    return check


def require_all(*checks) -> Callable:
    """Combine multiple requirement checks (all must pass)."""
    def combined(engine) -> Tuple[bool, str]:
        all_msgs = []
        all_met = True
        for check in checks:
            met, msg = check(engine)
            if not met:
                all_met = False
                if msg:
                    all_msgs.append(msg.replace("", ""))
        if all_met:
            return True, ""
        return False, "" + ", ".join(all_msgs)
    return combined


def always_pass() -> Callable:
    """Requirement that always passes (for manual advance steps)."""
    def check(engine) -> Tuple[bool, str]:
        return True, ""
    return check
