"""Oxygen supply failure recognition and response scenario."""

from anasim.core.action_log import ACTION_FGF, ACTION_OXYGEN_SUPPLY

from .base import (
    CRISIS_ONSET_S,
    CaseAction,
    CaseEvent,
    CaseOutcome,
    Scenario,
    ScenarioStep,
    action_taken_in_case,
    create_observe_step,
    equilibrate_maintenance_fio2,
    format_unmet_int,
    join_messages,
    require_all,
    require_oxygen_flow,
    require_oxygenation,
    require_tracheal_ventilation,
    stability_outcome,
    tracheal_ventilation_confirmed,
)


def _prepare_oxygen_supply_failure(engine) -> None:
    """Start on low-flow air/oxygen maintenance (~38% FiO2 across weights) so O2 loss causes hypoxic rebreathing."""
    engine.set_oxygen_supply_connected(True)
    equilibrate_maintenance_fio2(engine, target_fio2=0.385, min_air_l_min=0.75, air_vo2_factor=3.5)


def _backup_oxygen_reconnected(engine) -> bool:
    """Return whether backup oxygen was connected after a supply failure in this case."""
    if not engine.circuit.oxygen_supply_connected:
        return False
    saw_failure = False
    for record in engine.actions.since_case(ACTION_OXYGEN_SUPPLY):
        if record.label == "disconnected":
            saw_failure = True
        elif record.label == "connected" and saw_failure:
            return True
    return False


def _oxygen_outcome_recovered(engine) -> bool:
    """Accept either high-flow washout or safe cylinder-conserving O2 recovery once backup is connected."""
    return (
        _backup_oxygen_reconnected(engine)
        and engine.circuit.composition.fio2 >= 0.40
        and engine.state.sao2 >= 94.0
    )


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
        messages.append("Connect backup oxygen")
    if fio2 < 0.85:
        messages.append(f"Circuit FiO₂: {format_unmet_int(fio2 * 100, low=85)}%/85%+")
    return False, join_messages(messages)


def create_oxygen_supply_failure() -> Scenario:
    return Scenario(
        id="oxygen_supply_failure",
        name="O₂ supply failure",
        briefing=(
            "Surgery is in progress with controlled ventilation. "
            "Maintain oxygen delivery and manage equipment problems."
        ),
        setup_engine=_prepare_oxygen_supply_failure,
        events=(CaseEvent(
            "oxygen_supply", CRISIS_ONSET_S, "The workstation reports loss of oxygen supply pressure.",
            lambda engine: engine.set_oxygen_supply_connected(False),
        ),),
        actions=(CaseAction(
            "backup_oxygen", "Connect backup oxygen",
            lambda engine: engine.set_oxygen_supply_connected(True),
            lambda engine: not engine.circuit.oxygen_supply_connected,
            completed_label="Backup oxygen connected",
        ),),
        outcomes=(
            CaseOutcome("supply", "Oxygen supply restored", _backup_oxygen_reconnected,
                        "Restore oxygen delivery using a verified backup source."),
            CaseOutcome("oxygen", "Circuit oxygen concentration recovered",
                        _oxygen_outcome_recovered,
                        "Verify delivered oxygen at the circuit analyzer after changing the supply.",
                        milestone=True),
            CaseOutcome("ventilation", "Tracheal ventilation maintained", tracheal_ventilation_confirmed,
                        "Maintain ventilation while restoring the oxygen supply."),
            stability_outcome("Confirm patient recovery as well as restoration of the equipment supply."),
        ),
        steps=[
            create_observe_step(
                "disconnected", ACTION_OXYGEN_SUPPLY,
                "Follow circuit FiO₂ and the patient's oxygenation. The circuit oxygen analyzer "
                "should warn of supply trouble before pulse oximetry changes.",
            ),
            ScenarioStep(
                id="CONNECT_BACKUP_OXYGEN",
                title="Move to backup oxygen",
                instruction=(
                    "Act on the supply failure immediately; do not wait for FiO₂ or SpO₂ to fall. "
                    "Call for help and select <b>Connect backup oxygen</b>. Then set O₂ to "
                    "<b>10 L/min</b>, and set air and N₂O to <b>0 L/min</b> to flush the circuit. "
                    "In this simulator, the oxygen-pressure fail-safe also stops N₂O; the air source continues. "
                    "If machine oxygen delivery is uncertain in practice, disconnect the patient "
                    "and use a self-inflating bag with independent oxygen, or room air until it is available. "
                    "The machine's auxiliary O₂ outlet shares its supply and is not an independent source."
                ),
                check_requirements=require_all(
                    lambda engine: (
                        _backup_oxygen_reconnected(engine),
                        "" if _backup_oxygen_reconnected(engine) else "Connect backup oxygen",
                    ),
                    lambda engine: (
                        action_taken_in_case(engine, ACTION_FGF),
                        "Set fresh gas flow for this case",
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
