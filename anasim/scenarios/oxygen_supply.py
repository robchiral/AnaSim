"""Oxygen supply failure recognition and response scenario."""

from anasim.core.action_log import ACTION_FGF

from .base import (
    Scenario,
    ScenarioStep,
    action_taken_this_step,
    join_messages,
    require_all,
    require_oxygen_flow,
    require_oxygen_supply_action,
    require_oxygenation,
    require_tracheal_ventilation,
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


def _require_safe_baseline(engine) -> tuple[bool, str]:
    circuit = engine.circuit
    fio2 = circuit.composition.fio2
    ready = (
        circuit.oxygen_supply_connected
        and 0.34 <= fio2 <= 0.40
    )
    if ready:
        return True, ""

    messages = []
    if not circuit.oxygen_supply_connected:
        messages.append("Connect the O₂ supply")
    if not 0.34 <= fio2 <= 0.40:
        messages.append(f"Circuit FiO₂: {fio2 * 100:.0f}%/34-40%")
    return False, join_messages(messages)


def _require_analyzer_warning(engine) -> tuple[bool, str]:
    fio2 = engine.circuit.composition.fio2
    detected = not engine.circuit.oxygen_supply_connected and fio2 < 0.30
    if detected:
        return True, ""
    return False, f"Circuit FiO₂: {fio2 * 100:.0f}%; watch for a value below 30%"


def _require_oxygen_recovery(engine) -> tuple[bool, str]:
    fio2 = engine.circuit.composition.fio2
    recovered = (
        engine.circuit.oxygen_supply_connected
        and fio2 >= 0.85
    )
    if recovered:
        return True, ""

    messages = []
    if not engine.circuit.oxygen_supply_connected:
        messages.append("Connect backup O₂")
    if fio2 < 0.85:
        messages.append(f"Circuit FiO₂: {fio2 * 100:.0f}%/85%+")
    return False, join_messages(messages)


def create_oxygen_supply_failure() -> Scenario:
    """Create a guided oxygen pipeline disconnection scenario."""
    return Scenario(
        id="oxygen_supply_failure",
        name="O₂ supply failure",
        description=(
            "Recognize falling inspired oxygen before desaturation and restore "
            "ventilation with a verified backup oxygen source."
        ),
        setup_engine=_prepare_oxygen_supply_failure,
        steps=[
            ScenarioStep(
                id="CHECK_BASELINE",
                title="Check oxygen delivery",
                instruction=(
                    "Confirm ventilation through the tracheal tube, a stable patient, "
                    "and circuit FiO₂ near <b>37%</b>. "
                    "The circuit oxygen analyzer should warn of supply trouble "
                    "before pulse oximetry changes."
                ),
                check_requirements=require_all(
                    _require_safe_baseline,
                    require_oxygenation(),
                    require_tracheal_ventilation,
                ),
                target_tab="Machine",
            ),
            ScenarioStep(
                id="DISCONNECT_OXYGEN",
                title="Simulate oxygen supply loss",
                instruction=(
                    "Select <b>Disconnect O₂ supply</b>. In this simulator, the "
                    "oxygen-pressure fail-safe also stops N₂O; the air source continues. "
                    "A pressure fail-safe does not verify the delivered gas composition."
                ),
                check_requirements=require_oxygen_supply_action(
                    False, "Disconnect the O₂ supply on the Machine tab",
                ),
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
                    "Call for help. Select <b>Connect backup O₂</b>, set O₂ to "
                    "<b>10 L/min</b>, and set air and N₂O to <b>0 L/min</b> to flush the circuit. "
                    "This control represents isolating the failed supply and using verified backup oxygen. "
                    "If machine oxygen delivery is uncertain in practice, use a self-inflating bag with an independent oxygen source."
                ),
                check_requirements=require_all(
                    require_oxygen_supply_action(True, "Connect backup O₂"),
                    lambda engine: (
                        action_taken_this_step(engine, ACTION_FGF),
                        "Set fresh gas flow for this objective",
                    ),
                    require_oxygen_flow(),
                ),
                target_tab="Machine",
            ),
            ScenarioStep(
                id="CONFIRM_RECOVERY",
                title="Confirm oxygen recovery",
                instruction=(
                    "Continue ventilation and confirm circuit FiO₂ reaches "
                    "<b>85%</b> with SpO₂ at least <b>94%</b>. "
                    "Once oxygen delivery is restored, reduce flow to conserve the cylinder "
                    "and arrange a reliable ongoing supply. Cylinder depletion is not modeled."
                ),
                check_requirements=require_all(
                    _require_oxygen_recovery,
                    require_oxygenation(),
                    require_tracheal_ventilation,
                ),
                target_tab="Machine",
            ),
        ],
    )
