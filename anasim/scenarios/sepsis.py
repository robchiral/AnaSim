"""Warm septic shock scenario."""

from anasim.core.action_log import ACTION_CASE_ACTION, ACTION_EVENT_START

from .base import (
    CaseAction,
    CaseEvent,
    CaseOutcome,
    CrisisStartMAP,
    Scenario,
    ScenarioStep,
    action_taken_in_case,
    create_observe_step,
    patient_stable,
    preroll_scenario,
    require_all,
    require_crisis_stopped,
    require_fluid_bolus,
    require_hypotension_or_treatment,
    require_infusion_started,
    require_map_at_least,
    require_oxygenation,
    require_tracheal_ventilation,
    require_vasopressor_if_hypotensive,
    stability_outcome,
    tracheal_ventilation_confirmed,
)

_DECOMPENSATION_AT_S = 15.0
_PRE_SEPSIS_SEVERITY = 0.40
_PRE_LEAK_FRACTION = 0.09
_PRE_LACTATE_ADD_MMOL_L = 1.4


def _prepare_septic_patient(engine) -> None:
    """Arrive in the OR already septic: febrile, mildly volume-depleted, and hyperlactatemic."""
    engine.state.temp_c = max(engine.state.temp_c, 38.5)
    leak_ml = _PRE_LEAK_FRACTION * engine.hemo.blood_volume_0
    engine.hemo.blood_volume -= leak_ml
    engine.hemo.total_third_space_ml += leak_ml
    engine.hemo._protein_third_space_ml += leak_ml
    engine.hemo.fluids.shift_water(-leak_ml, protein_fraction=1.0)
    engine.hemo.invalidate_state_cache()
    lactate_mmol = _PRE_LACTATE_ADD_MMOL_L * engine.lactate.volume_l
    engine.lactate.endogenous_mmol += lactate_mmol
    engine.resp.remove_buffer_base(lactate_mmol)
    engine.sepsis.severity = _PRE_SEPSIS_SEVERITY
    engine.sepsis.active = True
    engine.sepsis.onset_rate = 0.0
    preroll_scenario(engine, 240.0, dt=4.0, settle_s=15.0, sample_interval_s=5.0)


def _source_control_requested(engine) -> bool:
    return (
        action_taken_in_case(engine, ACTION_CASE_ACTION, "source_control")
        or require_crisis_stopped("sepsis", "")(engine)[0]
    )


def create_sepsis_response() -> Scenario:
    start_map = CrisisStartMAP()
    decompensation_started = False
    hypotension_noted = False
    observe_done = False
    fluids = require_fluid_bolus(500, labels=("crystalloid", "saline"), event_name="sepsis")
    vasopressor = require_infusion_started(
        "nore", fail_message="Start or adjust norepinephrine for persistent hypotension",
        event_name="sepsis",
    )

    def start(engine) -> None:
        nonlocal decompensation_started
        decompensation_started = True
        start_map.record(engine)
        engine.sepsis.onset_rate = (1.0 - engine.sepsis.severity) / 45.0
        if engine.sepsis.active:
            engine.actions.record(engine.state.time, ACTION_EVENT_START, label="sepsis")
        else:
            engine.start_sepsis()

    def hypotension_apparent(engine) -> bool:
        if not decompensation_started:
            return False
        if engine.sepsis.severity >= 0.90:
            return True
        threshold = 65.0 if start_map.value is None else min(65.0, start_map.value - 10.0)
        return engine.state.map < threshold

    def mark_hypotension(engine) -> None:
        nonlocal hypotension_noted, observe_done
        hypotension_noted = True
        observe_done = True
        if _source_control_requested(engine):
            engine.sepsis.onset_rate = 0.0

    def request_source_control(engine) -> None:
        nonlocal observe_done
        # Surgical drainage addresses the infectious focus, but endotoxin release
        # already underway still produces septic shock and ongoing vasoplegia.
        # Acting early on decompensation advances the observe step without aborting shock.
        observe_done = True
        if hypotension_noted:
            engine.sepsis.onset_rate = 0.0

    def require_source_control(engine) -> tuple[bool, str]:
        met = _source_control_requested(engine)
        return met, "" if met else "Request surgical source control"

    steps = [
        create_observe_step(
            "sepsis",
            instruction=(
                "The patient has intra-abdominal sepsis and is febrile with mild tachycardia. "
                "Follow blood pressure, heart rate, and perfusion closely as surgery proceeds."
            ),
            check=lambda _engine: (observe_done, ""),
        ),
        ScenarioStep(
            id="RECOGNIZE_WARM_SHOCK",
            title="Recognize septic shock",
            instruction=(
                "Falling blood pressure with active infection and vasodilation signals septic shock; "
                "treat promptly before severe hypotension develops.<br><br>"
                "<i>Draw an arterial blood gas to follow lactate and acid-base status, and also "
                "consider blood loss and anesthetic depth.</i>"
            ),
            check_requirements=require_hypotension_or_treatment(
                start_map, "sepsis", "begin fluid resuscitation and support pressure",
            ),
        ),
        ScenarioStep(
            id="GIVE_FLUIDS",
            title="Give fluids",
            instruction=(
                "Give a <b>500 mL crystalloid bolus</b> (lactated Ringer's is preferred over 0.9% saline) "
                "and reassess the response.<br><br>"
                "<i>Initial sepsis resuscitation targets around 30 mL/kg over 3 hours, with further fluid guided by perfusion.</i>"
            ),
            check_requirements=fluids,
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="START_VASOPRESSOR",
            title="Start vasopressor",
            instruction=(
                "If <b>MAP < 65 mmHg</b>, start or adjust <b>norepinephrine</b> in mcg/min "
                "(0.05-0.1 mcg/kg/min = 3.5-7 mcg/min for 70 kg) and titrate to pressure and perfusion. "
                "Start pressors alongside fluids in severe hypotension; skip this if pressure has already recovered."
            ),
            check_requirements=require_vasopressor_if_hypotensive(
                "nore", fail_message="Start or adjust norepinephrine for persistent hypotension",
                event_name="sepsis",
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="SOURCE_CONTROL",
            title="Request source control",
            instruction=(
                "Select <b>Request surgical source control</b> and maintain hemodynamic support; "
                "vasoplegia persists after drainage.<br><br>"
                "<i>Draw blood cultures without delaying broad-spectrum antibiotics, ideally within 1 hour.</i>"
            ),
            check_requirements=require_source_control,
        ),
        ScenarioStep(
            id="REASSESS",
            title="Reassess hemodynamics",
            instruction=(
                "Confirm source control is underway and sustain <b>MAP ≥ 65 mmHg</b> with ongoing "
                "fluid and vasopressor support.<br><br>"
                "<i>Reassess response before giving more fluid and plan ICU care for continued "
                "vasopressor titration. Follow lactate, urine output, exam, and bedside ultrasound.</i>"
            ),
            check_requirements=require_all(
                require_map_at_least(65),
                require_oxygenation(),
                require_tracheal_ventilation,
                require_source_control,
                lambda engine: (patient_stable(engine), "Reassess oxygenation and perfusion"),
            ),
        ),
    ]

    return Scenario(
        id="sepsis_response",
        name="Septic shock response",
        setup_engine=_prepare_septic_patient,
        briefing=(
            "Surgery is in progress for an intra-abdominal infection. The patient arrived febrile "
            "and mildly tachycardic. Monitor hemodynamics and manage any deterioration."
        ),
        steps=steps,
        events=(
            CaseEvent(
                "sepsis_decompensation", _DECOMPENSATION_AT_S,
                "The surgeon drains a purulent intra-abdominal collection.",
                start,
            ),
            CaseEvent(
                "sepsis_hypotension", _DECOMPENSATION_AT_S,
                "Blood pressure is falling with widening vasodilation.",
                mark_hypotension,
                when=hypotension_apparent,
            ),
        ),
        actions=(CaseAction(
            "source_control", "Request surgical source control",
            request_source_control, lambda engine: engine.sepsis.active and decompensation_started,
            completed_label="Source control requested",
        ),),
        outcomes=(
            CaseOutcome("volume", "Volume replacement started", lambda engine: fluids(engine)[0],
                        "Replace circulating volume while treating septic shock.", milestone=True),
            CaseOutcome("vasopressor", "Norepinephrine started",
                        lambda engine: vasopressor(engine)[0] or (
                            decompensation_started and hypotension_noted and require_map_at_least(65)(engine)[0]
                        ),
                        "Start norepinephrine promptly when fluid resuscitation does not restore MAP ≥ 65 mmHg.",
                        milestone=True),
            CaseOutcome("source", "Cause addressed", _source_control_requested,
                        "Request surgical source control alongside resuscitation."),
            CaseOutcome("ventilation", "Tracheal ventilation maintained", tracheal_ventilation_confirmed,
                        "Maintain ventilation and assess the airway during circulatory deterioration."),
            stability_outcome("Continue vasopressor and fluid support to maintain MAP ≥ 65 mmHg."),
        ),
    )
