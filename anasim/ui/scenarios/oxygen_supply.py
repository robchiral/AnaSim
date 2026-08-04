"""Oxygen supply failure recognition and response scenario."""

from typing import Tuple

from anasim.core.action_log import ACTION_FGF

from .base import (
    Scenario,
    ScenarioStep,
    action_taken_this_step,
    join_messages,
    monitor_value,
    require_oxygen_supply_action,
)


def _prepare_oxygen_supply_failure(engine) -> None:
    """Start with a stable, oxygen-enriched air mixture on controlled ventilation."""
    engine.set_oxygen_supply_connected(True)
    engine.set_fgf(2.0, 8.0, 0.0)

    circuit = engine.circuit
    baseline_fio2 = (circuit.fgf_o2 + 0.21 * circuit.fgf_air) / (
        circuit.fgf_o2 + circuit.fgf_air
    )
    circuit.composition.fio2 = baseline_fio2
    circuit.composition.fin2 = 1.0 - baseline_fio2
    circuit.composition.fin2o = 0.0
    circuit.composition.fi_agent = 0.0
    engine.state.fio2 = baseline_fio2


def _require_safe_baseline(engine) -> Tuple[bool, str]:
    circuit = engine.circuit
    fio2 = circuit.composition.fio2
    spo2 = monitor_value(engine, "spo2")
    ready = (
        circuit.oxygen_supply_connected
        and 0.34 <= fio2 <= 0.40
        and spo2 >= 94
    )
    if ready:
        return True, ""

    messages = []
    if not circuit.oxygen_supply_connected:
        messages.append("Connect the O₂ supply")
    if not 0.34 <= fio2 <= 0.40:
        messages.append(f"Circuit FiO₂: {fio2 * 100:.0f}%/34–40%")
    if spo2 < 94:
        messages.append(f"SpO₂: {spo2:.0f}%/94%+")
    return False, join_messages(messages)


def _require_supply_failure(engine) -> Tuple[bool, str]:
    return require_oxygen_supply_action(
        False,
        "Disconnect the O₂ supply on the Machine tab",
    )(engine)


def _require_analyzer_warning(engine) -> Tuple[bool, str]:
    fio2 = engine.circuit.composition.fio2
    detected = not engine.circuit.oxygen_supply_connected and fio2 < 0.30
    if detected:
        return True, ""
    return False, f"Circuit FiO₂: {fio2 * 100:.0f}% — watch for a value below 30%"


def _require_backup_oxygen(engine) -> Tuple[bool, str]:
    circuit = engine.circuit
    source_ok = require_oxygen_supply_action(
        True,
        "Connect backup O₂",
    )(engine)[0]
    flow_set = action_taken_this_step(engine, ACTION_FGF)
    oxygen_ok = circuit.delivered_o2_flow() >= 8.0
    air_off = circuit.fgf_air <= 0.1
    n2o_off = circuit.fgf_n2o <= 0.1
    if source_ok and flow_set and oxygen_ok and air_off and n2o_off:
        return True, ""

    messages = []
    if not source_ok:
        messages.append("Connect backup O₂")
    if not flow_set:
        messages.append("Set fresh gas flow for this objective")
    if not oxygen_ok:
        messages.append(f"O₂ flow: {circuit.delivered_o2_flow():.1f}/8+ L/min")
    if not air_off:
        messages.append("Set air to 0 L/min")
    if not n2o_off:
        messages.append("Set N₂O to 0 L/min")
    return False, join_messages(messages)


def _require_oxygen_recovery(engine) -> Tuple[bool, str]:
    fio2 = engine.circuit.composition.fio2
    spo2 = monitor_value(engine, "spo2")
    recovered = (
        engine.circuit.oxygen_supply_connected
        and fio2 >= 0.85
        and spo2 >= 94
    )
    if recovered:
        return True, ""

    messages = []
    if fio2 < 0.85:
        messages.append(f"Circuit FiO₂: {fio2 * 100:.0f}%/85%+")
    if spo2 < 94:
        messages.append(f"SpO₂: {spo2:.0f}%/94%+")
    return False, join_messages(messages)


def create_oxygen_supply_failure() -> Scenario:
    """Create a guided oxygen pipeline disconnection scenario."""
    return Scenario(
        id="oxygen_supply_failure",
        name="O₂ supply failure",
        icon="",
        description=(
            "You recognized falling inspired oxygen before patient desaturation "
            "and moved ventilation to a backup oxygen source."
        ),
        setup_engine=_prepare_oxygen_supply_failure,
        steps=[
            ScenarioStep(
                id="CHECK_BASELINE",
                title="Check oxygen delivery",
                instruction=(
                    "Confirm a stable patient and a circuit FiO₂ near <b>37%</b>. "
                    "The circuit oxygen analyzer should warn of supply trouble "
                    "before pulse oximetry changes."
                ),
                check_requirements=_require_safe_baseline,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="DISCONNECT_OXYGEN",
                title="Simulate oxygen supply loss",
                instruction=(
                    "Select <b>Disconnect O₂ supply</b>. The oxygen-pressure "
                    "fail-safe also stops nitrous oxide delivery; the air source "
                    "continues."
                ),
                check_requirements=_require_supply_failure,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="RECOGNIZE_LOW_FIO2",
                title="Recognize the analyzer warning",
                instruction=(
                    "Watch the measured circuit FiO₂ fall below <b>30%</b>. "
                    "Do not wait for SpO₂ to decline before responding."
                ),
                check_requirements=_require_analyzer_warning,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="CONNECT_BACKUP_OXYGEN",
                title="Move to backup oxygen",
                instruction=(
                    "Select <b>Connect backup O₂</b>, set O₂ to at least "
                    "<b>8 L/min</b>, and turn air and N₂O to <b>0 L/min</b>."
                ),
                check_requirements=_require_backup_oxygen,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="CONFIRM_RECOVERY",
                title="Confirm oxygen recovery",
                instruction=(
                    "Continue ventilation and confirm circuit FiO₂ rises above "
                    "<b>85%</b> with SpO₂ at least <b>94%</b>."
                ),
                check_requirements=_require_oxygen_recovery,
                target_tab="Machine",
            ),
        ],
    )
