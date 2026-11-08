"""Ventricular fibrillation arrest during surgery."""

from anasim.core.action_log import (
    ACTION_COMPRESSIONS,
    ACTION_DRUG_BOLUS,
    ACTION_EVENT_START,
    ACTION_SHOCK,
)

from .base import (
    CRISIS_BRIEFING,
    CRISIS_ONSET_S,
    CaseEvent,
    CaseOutcome,
    Requirement,
    Scenario,
    ScenarioStep,
    create_observe_step,
    require_all,
    require_map_at_least,
    require_oxygen_flow,
    require_oxygenation,
    stability_outcome,
    tracheal_ventilation_confirmed,
)

EVENT = "ventricular_fibrillation"


def _start(engine) -> None:
    engine.actions.record(engine.state.time, ACTION_EVENT_START, label=EVENT)
    engine.set_rhythm("VFIB")


def _onset(engine) -> float | None:
    records = engine.actions.since_case(ACTION_EVENT_START, labels=(EVENT,))
    return records[-1].time if records else None


def _first_since_onset(engine, action: str, label: str) -> float | None:
    """Seconds from arrest onset to the first matching action."""
    onset = _onset(engine)
    records = engine.actions.since_event(EVENT, action, labels=(label,))
    return records[0].time - onset if onset is not None and records else None


def _circulation_restored(engine) -> bool:
    return _onset(engine) is not None and engine.spontaneous_circulation


def _within(action: str, label: str, limit_s: float):
    def check(engine) -> bool:
        seconds = _first_since_onset(engine, action, label)
        return seconds is not None and seconds <= limit_s
    return check


def _require_action(action: str, label: str, message: str) -> Requirement:
    def check(engine) -> tuple[bool, str]:
        met = _first_since_onset(engine, action, label) is not None
        return met, "" if met else message
    return check


def _require_epinephrine_or_rosc(engine) -> tuple[bool, str]:
    if _circulation_restored(engine):
        return True, ""
    doses = engine.actions.since_event(EVENT, ACTION_DRUG_BOLUS, labels=("epi",))
    met = any(record.amount >= 1000.0 for record in doses)
    return met, "" if met else "Give epinephrine 1000 mcg if VF persists after the second shock"


def _require_rosc(engine) -> tuple[bool, str]:
    if not _circulation_restored(engine):
        return False, "Continue the cycle of compressions, rhythm checks, and shocks"
    if engine.hemo.compressions:
        return False, "Stop compressions once circulation returns"
    return True, ""


def create_cardiac_arrest() -> Scenario:
    compressions = _require_action(ACTION_COMPRESSIONS, "on", "Start chest compressions")
    shock = _require_action(ACTION_SHOCK, "unsynchronized", "Deliver an unsynchronized shock")
    steps = [
        create_observe_step(EVENT, instruction="Follow the ECG, arterial pressure, oxygenation, and capnography."),
        ScenarioStep(
            id="START_COMPRESSIONS",
            title="Start chest compressions",
            instruction=(
                "Ventricular fibrillation with loss of the arterial pulse is cardiac arrest. "
                "Call for help and the defibrillator, and start <b>chest compressions</b> immediately. "
                "Give <b>100% oxygen</b>: set O₂ 10 L/min, air and N₂O 0 L/min, and keep ventilating "
                "through the tracheal tube at about 10 breaths/min.<br><br>"
                "Use the arterial line and capnography to judge compressions: diastolic pressure "
                "of at least 25 mmHg between compressions perfuses the heart, and EtCO₂ below "
                "10 mmHg suggests poor flow."
            ),
            check_requirements=require_all(compressions, require_oxygen_flow()),
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="DEFIBRILLATE",
            title="Defibrillate",
            instruction=(
                "Charge the defibrillator while continuing compressions. Once ready, clear the patient and "
                "deliver an <b>unsynchronized 200 J shock</b>, then resume compressions immediately "
                "for 2 minutes before checking the rhythm. Synchronized mode does not fire in VF, "
                "since there is no R wave to time the shock to."
            ),
            check_requirements=shock,
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="CONTINUE_CYCLES",
            title="Continue the resuscitation cycle",
            instruction=(
                "Every 2 minutes, pause compressions briefly to check the rhythm. Shock VF again "
                "and resume compressions at once. Give <b>epinephrine 1 mg</b> (1000 mcg) after the "
                "second shock and every 3-5 minutes while VF persists. After the third shock, give "
                "<b>amiodarone 300 mg</b> or <b>lidocaine 1-1.5 mg/kg</b>, then amiodarone 150 mg "
                "or lidocaine 0.5-0.75 mg/kg if VF persists. VF can recur after a successful "
                "shock.<br><br>"
                "An organized rhythm with arterial pulsations and a sudden rise in EtCO₂ indicates "
                "return of circulation."
            ),
            check_requirements=_require_epinephrine_or_rosc,
            target_tab="Medications",
        ),
        ScenarioStep(
            id="POST_ARREST",
            title="Stabilize after return of circulation",
            instruction=(
                "Stop compressions once an organized rhythm perfuses. Support <b>MAP ≥ 65 mmHg</b> "
                "with norepinephrine or epinephrine and ventilate to normal PaCO₂. In practice, "
                "titrate oxygen to SpO₂ 92-98%, look for the cause of arrest, obtain a 12-lead ECG, "
                "and plan temperature control."
            ),
            check_requirements=require_all(_require_rosc, require_map_at_least(), require_oxygenation()),
        ),
    ]
    return Scenario(
        id="cardiac_arrest",
        name="Cardiac arrest",
        briefing=CRISIS_BRIEFING,
        steps=steps,
        events=(CaseEvent(EVENT, CRISIS_ONSET_S, "The ECG changes abruptly and the arterial waveform flattens.", _start),),
        outcomes=(
            CaseOutcome("compressions", "Compressions within 1 minute of arrest",
                        _within(ACTION_COMPRESSIONS, "on", 60.0),
                        "Start compressions as soon as arrest is recognized.", milestone=True),
            CaseOutcome("defibrillation", "First shock within 2 minutes of arrest",
                        _within(ACTION_SHOCK, "unsynchronized", 120.0),
                        "Defibrillate VF as soon as a defibrillator is available.", milestone=True),
            CaseOutcome("rosc", "Circulation restored", _circulation_restored,
                        "Shockable rhythms respond best to early defibrillation with good compressions."),
            CaseOutcome("ventilation", "Tracheal ventilation maintained", tracheal_ventilation_confirmed,
                        "Keep ventilating with oxygen throughout the arrest."),
            stability_outcome("Support pressure and oxygenation after circulation returns."),
        ),
    )
