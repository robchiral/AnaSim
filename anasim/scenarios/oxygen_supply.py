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
    """Start on low-flow air/oxygen maintenance (~35% FiO2 across weights) so O2 loss causes hypoxic rebreathing."""
    engine.set_pipeline_connected("o2", True)
    engine.set_cylinder_valve("o2", False)
    engine.replace_cylinder("o2", 1.0)
    equilibrate_maintenance_fio2(engine, target_fio2=0.35, min_air_l_min=0.75, air_vo2_factor=3.5)


def _backup_oxygen_reconnected(engine) -> bool:
    """Return whether backup oxygen was connected after a supply failure in this case."""
    if not engine.circuit.oxygen_supply_connected:
        return False
    saw_failure = False
    for record in engine.actions.since_case(ACTION_OXYGEN_SUPPLY):
        if record.label in ("disconnected", "pipeline_disconnected"):
            saw_failure = True
        elif record.label in ("connected", "cylinder_open", "pipeline_connected") and saw_failure:
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
        and fio2 >= 0.50
    )
    if recovered:
        return True, ""

    messages = []
    if not engine.circuit.oxygen_supply_connected:
        messages.append("Connect backup oxygen")
    if fio2 < 0.50:
        messages.append(f"Circuit FiO₂: {format_unmet_int(fio2 * 100, low=50)}%/50%+")
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
            lambda engine: engine.set_cylinder_valve("o2", True),
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
            stability_outcome("Confirm both oxygen supply restoration and patient recovery."),
        ),
        steps=[
            create_observe_step(
                "disconnected", ACTION_OXYGEN_SUPPLY,
                "Watch circuit FiO₂ and SpO₂. Circuit FiO₂ drops before pulse oximetry changes.",
            ),
            ScenarioStep(
                id="CONNECT_BACKUP_OXYGEN",
                title="Move to backup oxygen",
                instruction=(
                    "Act immediately: call for help, disconnect the wall oxygen pipeline, select "
                    "<b>Connect backup oxygen</b>, and turn <b>air and N₂O to 0 L/min</b> with <b>O₂ ≥ 1 L/min</b> "
                    "(use brief high flow only if FiO₂ or SpO₂ has dropped, then reduce flow to conserve the E-cylinder).<br><br>"
                    "<i>The fail-safe cuts N₂O, not air. Switch to manual ventilation on O₂-driven bellows, or use "
                    "a self-inflating bag with an independent cylinder if machine delivery is uncertain.</i>"
                ),
                check_requirements=require_all(
                    lambda engine: (
                        _backup_oxygen_reconnected(engine),
                        "" if _backup_oxygen_reconnected(engine) else "Connect backup oxygen",
                    ),
                    lambda engine: (
                        action_taken_in_case(engine, ACTION_FGF),
                        "Set fresh gas flow",
                    ),
                    require_oxygen_flow(1.0),
                ),
                target_tab="Machine",
            ),
            ScenarioStep(
                id="CONFIRM_RECOVERY",
                title="Confirm oxygen recovery",
                instruction=(
                    "Continue ventilation until circuit <b>FiO₂ ≥ 50%</b> and <b>SpO₂ ≥ 94%</b>.<br><br>"
                    "<i>Keep O₂ flow low (0.5-2 L/min) once recovered to conserve cylinder oxygen while "
                    "arranging a replacement supply.</i>"
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
