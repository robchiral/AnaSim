"""Malignant hyperthermia during sevoflurane anesthesia."""

from anasim.core.action_log import ACTION_DRUG_BOLUS
from anasim.machine.ventilator import VentSettings

from .base import (
    CRISIS_BRIEFING,
    CRISIS_ONSET_S,
    CaseEvent,
    CaseOutcome,
    Scenario,
    ScenarioStep,
    create_observe_step,
    monitor_value,
    require_all,
    require_infusion_started,
    require_oxygen_flow,
    require_tracheal_ventilation,
    stability_outcome,
)


def _require_dantrolene(engine) -> tuple[bool, str]:
    doses = engine.actions.since_event("malignant_hyperthermia", ACTION_DRUG_BOLUS, labels=("dantrolene",))
    given, target = sum(dose.amount for dose in doses), 2.5 * engine.patient.weight
    if given >= target - 0.1:
        return True, ""
    if given > 0.0:
        return False, f"Dantrolene: {given:.0f}/{target:.0f} mg given"
    return False, "Give dantrolene 2.5 mg/kg IV"


def _require_vaporizer_off(engine) -> tuple[bool, str]:
    met = not engine.circuit.vaporizer_on
    return met, "" if met else "Turn off sevoflurane"


def _require_controlled(engine) -> tuple[bool, str]:
    met = engine.malignant_hyperthermia.severity < 0.05
    return met, "" if met else "Signs of MH persist; repeat dantrolene"


def _require_temperature_managed(engine) -> tuple[bool, str]:
    state = engine.state
    messages = []
    if state.temp_c >= 38.0:
        messages.append(f"Temperature: {state.temp_c:.1f}/<38.0 °C")
    elif state.surface_cooling:
        messages.append("Stop surface cooling")
    if state.bair_hugger_target > 0.0:
        messages.append("Turn off the forced-air warmer")
    return not messages, ", ".join(messages)


def _require_etco2_normal(engine) -> tuple[bool, str]:
    etco2 = monitor_value(engine, "etco2")
    met = engine.state.etco2_signal_valid and etco2 < 45.0
    return met, "" if met else f"EtCO₂: {etco2:.0f}/<45 mmHg"


def create_malignant_hyperthermia() -> Scenario:
    onset_vent: VentSettings | None = None
    onset_mv = 0.0

    def start(engine) -> None:
        nonlocal onset_vent, onset_mv
        onset_vent, onset_mv = engine.vent.settings, engine.state.mv
        engine.start_malignant_hyperthermia()

    def ventilation(engine) -> tuple[bool, str]:
        settings, before = engine.vent.settings, onset_vent
        increased = before is not None and (
            settings.rr > before.rr or settings.tv > before.tv or settings.p_insp > before.p_insp
        )
        if not (engine.vent.is_on and increased):
            return False, "Increase the ventilator rate or tidal volume"
        mv, target = engine.state.mv, 2.0 * onset_mv
        met = mv >= target
        return met, "" if met else f"Minute ventilation: {mv:.1f}/{target:.1f}+ L/min"

    propofol = require_infusion_started(
        "propofol", fail_message="Start propofol", event_name="malignant_hyperthermia",
    )
    oxygen = require_oxygen_flow(10.0)
    controlled = require_all(_require_dantrolene, _require_controlled)
    return Scenario(
        id="malignant_hyperthermia",
        name="Malignant hyperthermia",
        briefing=CRISIS_BRIEFING,
        events=(CaseEvent(
            "malignant_hyperthermia", CRISIS_ONSET_S,
            "EtCO₂ begins to rise although the ventilator settings are unchanged.", start,
        ),),
        steps=[
            create_observe_step("malignant_hyperthermia"),
            ScenarioStep(
                id="STOP_TRIGGERS",
                title="Stop triggers and hyperventilate",
                instruction=(
                    "Unexplained rising EtCO₂, tachycardia, and muscle rigidity suggest MH; "
                    "temperature often rises later. Call for help and the MH cart, "
                    "and ask the surgeon to stop.<br><br>"
                    "Turn off sevoflurane and avoid succinylcholine. Give <b>100% oxygen</b>: "
                    "set O₂ 10 L/min, air and N₂O 0 L/min. Increase minute ventilation to "
                    "<b>2-3 times baseline</b> and follow EtCO₂. Start <b>propofol</b> to "
                    "maintain anesthesia. In practice, add activated charcoal filters if available; "
                    "do not delay treatment to change the machine."
                ),
                check_requirements=require_all(
                    _require_vaporizer_off, oxygen,
                    require_tracheal_ventilation, ventilation, propofol,
                ),
                target_tab="Machine",
            ),
            ScenarioStep(
                id="DANTROLENE",
                title="Give dantrolene",
                instruction=(
                    "Give <b>dantrolene 2.5 mg/kg IV</b> rapidly (175 mg for 70 kg). "
                    "Repeat until EtCO₂, HR, and rigidity improve. Doses above 10 mg/kg are "
                    "occasionally needed; if signs persist, reconsider the diagnosis."
                ),
                check_requirements=_require_dantrolene,
                target_tab="Medications",
            ),
            ScenarioStep(
                id="STABILIZE",
                title="Stabilize and reassess",
                instruction=(
                    "Follow EtCO₂, HR, rigidity, blood gases, and core temperature. "
                    "Turn off the forced-air warmer. Start <b>surface cooling</b> if core temperature "
                    "exceeds 39 °C or rises rapidly, and stop it below 38 °C.<br><br>"
                    "In practice, check potassium, CK, myoglobin, glucose, coagulation, and urine output. "
                    "Treat hyperkalemia with calcium, sodium bicarbonate, and glucose with insulin; "
                    "do not give calcium channel blockers with dantrolene. Arrange at least 24 hours "
                    "of critical care monitoring, further dantrolene according to the local protocol, "
                    "and referral for MH testing."
                ),
                check_requirements=require_all(controlled, _require_etco2_normal, _require_temperature_managed),
                target_tab="Machine",
            ),
        ],
        outcomes=(
            CaseOutcome("triggers", "Sevoflurane stopped", lambda engine: _require_vaporizer_off(engine)[0],
                        "Stop volatile anesthetics and avoid succinylcholine."),
            CaseOutcome("oxygen", "High-flow oxygen given", lambda engine: oxygen(engine)[0],
                        "Give 100% oxygen at high flow to wash out the volatile agent."),
            CaseOutcome("ventilation", "Ventilation increased", lambda engine: ventilation(engine)[0],
                        "Increase minute ventilation to 2-3 times baseline and follow EtCO₂.", milestone=True),
            CaseOutcome("anesthesia", "Propofol anesthesia started", lambda engine: propofol(engine)[0],
                        "Maintain anesthesia with propofol after stopping the volatile agent."),
            CaseOutcome("dantrolene", "Dantrolene given", lambda engine: _require_dantrolene(engine)[0],
                        "Give dantrolene 2.5 mg/kg IV promptly and repeat until signs improve.", milestone=True),
            CaseOutcome("controlled", "Hypermetabolism controlled", lambda engine: controlled(engine)[0],
                        "Repeat dantrolene until EtCO₂, HR, and rigidity improve."),
            stability_outcome("Reassess oxygenation and blood pressure while treating the crisis."),
        ),
    )
