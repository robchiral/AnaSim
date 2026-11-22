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
    infusion_running,
    require_all,
    require_etco2_above,
    require_map_at_least,
    require_oxygen_flow,
    require_oxygenation,
    require_tracheal_ventilation,
    stability_outcome,
    tracheal_ventilation_confirmed,
)

EVENT = "ventricular_fibrillation"
_ANESTHETIC_INFUSIONS = ("propofol", "remi", "fentanyl", "midazolam", "ketamine", "etomidate")


class _RefractoryShockRNG:
    """Force initial shocks to fail so learners practice multi-cycle ACLS and epinephrine."""

    def __init__(self, base_rng, fail_shocks: int = 2) -> None:
        self._base = base_rng
        self._remaining_failures = fail_shocks

    def random(self) -> float:
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            return 0.999999
        return float(self._base.random())

    def __getattr__(self, name: str):
        return getattr(self._base, name)


def _start(engine) -> None:
    engine.actions.record(engine.state.time, ACTION_EVENT_START, label=EVENT)
    if not isinstance(engine._shock_rng, _RefractoryShockRNG):
        engine._shock_rng = _RefractoryShockRNG(engine._shock_rng, fail_shocks=2)
    engine.set_rhythm("VFIB")


def _anesthetics_stopped(engine) -> tuple[bool, str]:
    running_iv = infusion_running(engine, *_ANESTHETIC_INFUSIONS)
    volatile_on = engine.circuit.vaporizer_on and engine.circuit.vaporizer_setting > 0.0
    if not running_iv and not volatile_on:
        return True, ""
    return False, "Stop all anesthetic infusions and volatile agents"


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
    met = sum(record.amount for record in doses) >= 1000.0
    return met, "" if met else "Give epinephrine 1000 mcg if VF persists after the second shock"


def _require_rosc(engine) -> tuple[bool, str]:
    if not _circulation_restored(engine):
        return False, "Continue the cycle of compressions, rhythm checks, and shocks"
    if engine.hemo.compressions:
        return False, "Stop compressions once circulation returns"
    return True, ""


def _require_start_compressions_or_rosc(engine) -> tuple[bool, str]:
    if _circulation_restored(engine):
        return True, ""
    return require_all(
        _require_action(ACTION_COMPRESSIONS, "on", "Start chest compressions"),
        _anesthetics_stopped,
        require_oxygen_flow(),
    )(engine)


def _observe_instruction(engine) -> str:
    if engine.config.arterial_line_enabled:
        return "Follow the ECG, arterial pressure, oxygenation, and capnography."
    return "Follow the ECG, pulse oximetry, blood pressure, and capnography."


def _start_compressions_instruction(engine) -> str:
    if engine.config.arterial_line_enabled:
        recognition = "VF with loss of arterial pulse is cardiac arrest."
        feedback = (
            "<i>Judge CPR quality with the arterial line and capnography: "
            "target diastolic pressure ≥ 25 mmHg and EtCO₂ ≥ 10 mmHg.</i>"
        )
    else:
        recognition = "VF with loss of the pleth waveform and an abrupt EtCO₂ drop is cardiac arrest."
        feedback = (
            "<i>Judge CPR quality with capnography: EtCO₂ below 10 mmHg suggests poor flow.</i>"
        )
    return (
        f"{recognition} Call for help and the defibrillator, stop all anesthetic agents, "
        "and start <b>chest compressions</b> immediately. Set <b>100% oxygen</b> "
        "(O₂ 10 L/min, air and N₂O 0 L/min in Machine) and ventilate at about 10 breaths/min.<br><br>"
        f"{feedback}"
    )


def _continue_cycles_instruction(engine) -> str:
    rosc_cue = (
        "An organized rhythm with arterial pulsations and a sudden EtCO₂ rise indicates "
        "return of circulation."
        if engine.config.arterial_line_enabled
        else "An organized rhythm with a returning pulse and a sudden EtCO₂ rise indicates "
        "return of circulation."
    )
    return (
        "Check rhythm every 2 minutes, shock persistent VF, and resume compressions immediately. "
        "Give <b>epinephrine 1 mg</b> (1000 mcg) after the second shock and every 3-5 minutes. "
        "After the third shock, give <b>amiodarone 300 mg</b> or <b>lidocaine 1-1.5 mg/kg</b> "
        "(repeat amiodarone 150 mg or lidocaine 0.5-0.75 mg/kg if VF persists).<br><br>"
        f"<i>{rosc_cue} VF can recur after a successful shock.</i>"
    )


def _epinephrine_outcome(engine) -> bool:
    onset = _onset(engine)
    if onset is None:
        return False
    doses = [
        r for r in engine.actions.since_case(ACTION_DRUG_BOLUS, labels=("epi",))
        if r.time >= onset
    ]
    if sum(r.amount for r in doses) >= 1000.0:
        return True
    shocks = [
        r for r in engine.actions.since_case(ACTION_SHOCK, labels=("unsynchronized",))
        if r.time >= onset
    ]
    return _circulation_restored(engine) and len(shocks) < 2


def create_cardiac_arrest() -> Scenario:
    shock = _require_action(ACTION_SHOCK, "unsynchronized", "Deliver an unsynchronized shock")
    steps = [
        create_observe_step(EVENT, instruction=_observe_instruction),
        ScenarioStep(
            id="START_COMPRESSIONS",
            title="Start chest compressions",
            instruction=_start_compressions_instruction,
            check_requirements=_require_start_compressions_or_rosc,
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="DEFIBRILLATE",
            title="Defibrillate",
            instruction=(
                "Charge the defibrillator during compressions. When ready, clear the patient, "
                "deliver an <b>unsynchronized 200 J shock</b>, and resume compressions immediately "
                "for 2 minutes.<br><br>"
                "<i>Synchronized mode will not fire in VF because there is no R wave.</i>"
            ),
            check_requirements=shock,
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="CONTINUE_CYCLES",
            title="Continue the resuscitation cycle",
            instruction=_continue_cycles_instruction,
            check_requirements=_require_epinephrine_or_rosc,
            target_tab="Medications",
        ),
        ScenarioStep(
            id="POST_ARREST",
            title="Stabilize after return of circulation",
            instruction=(
                "Stop compressions once a perfusing rhythm returns. Maintain <b>MAP ≥ 65 mmHg</b> "
                "with norepinephrine or epinephrine and ventilate to normocapnia.<br><br>"
                "<i>Titrate oxygen to SpO₂ 92-98%, investigate the cause, get a 12-lead ECG, "
                "and plan temperature control.</i>"
            ),
            check_requirements=require_all(
                _require_rosc,
                require_map_at_least(),
                require_oxygenation(),
                require_tracheal_ventilation,
                require_etco2_above(20),
            ),
        ),
    ]
    return Scenario(
        id="cardiac_arrest",
        name="Cardiac arrest",
        briefing=CRISIS_BRIEFING,
        steps=steps,
        events=(CaseEvent(EVENT, CRISIS_ONSET_S, "The ECG changes abruptly and the pulse is lost.", _start),),
        outcomes=(
            CaseOutcome("compressions", "Compressions within 1 minute of arrest",
                        _within(ACTION_COMPRESSIONS, "on", 60.0),
                        "Start compressions as soon as arrest is recognized.", milestone=True),
            CaseOutcome("anesthetics", "Anesthetic agents stopped",
                        lambda engine: _onset(engine) is not None and _anesthetics_stopped(engine)[0],
                        "Stop all anesthetic infusions and volatile agents during cardiac arrest.", milestone=True),
            CaseOutcome("defibrillation", "First shock within 2 minutes of arrest",
                        _within(ACTION_SHOCK, "unsynchronized", 120.0),
                        "Defibrillate VF as soon as a defibrillator is available.", milestone=True),
            CaseOutcome("epinephrine", "Epinephrine 1 mg given for persistent VF",
                        _epinephrine_outcome,
                        "Give epinephrine 1 mg (1000 mcg) after the second shock for persistent VF.",
                        milestone=True),
            CaseOutcome("rosc", "Circulation restored", _circulation_restored,
                        "Shockable rhythms respond best to early defibrillation with good compressions."),
            CaseOutcome("ventilation", "Tracheal ventilation maintained", tracheal_ventilation_confirmed,
                        "Keep ventilating with oxygen throughout the arrest."),
            stability_outcome("Support pressure and oxygenation after circulation returns."),
        ),
    )
