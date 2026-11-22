"""Anaphylaxis response scenario."""

from .base import (
    CRISIS_BRIEFING,
    CaseAction,
    CaseEvent,
    CaseOutcome,
    Requirement,
    Scenario,
    ScenarioStep,
    bolus_given_in_case,
    create_observe_step,
    crisis_outcomes,
    equilibrate_maintenance_fio2,
    require_all,
    require_crisis_resolved_with_map,
    require_fluid_bolus,
    require_infusion_started,
    require_oxygen_flow,
    require_tracheal_ventilation,
)

_ANAPHYLAXIS_ONSET_S = 15.0


def _epinephrine_instruction(engine) -> str:
    if engine.config.maint_type == "balanced" or engine.circuit.vaporizer_on:
        anesthetic_line = (
            "If hypotension persists, turn off the volatile agent and vasodilating infusions, "
            "and consider midazolam for amnesia."
        )
    else:
        anesthetic_line = (
            "If hypotension persists, reduce or pause vasodilating infusions such as propofol, "
            "and consider midazolam for amnesia."
        )
    return (
        "Sudden hypotension with bronchospasm and rising airway pressure suggests anaphylaxis. "
        "Call for help, pause surgery, and select <b>Stop suspected trigger</b>.<br><br>"
        "Give <b>epinephrine 50-100 mcg IV</b> promptly and repeat to response; start an infusion "
        "if repeated boluses are needed (for example, 3.5 mcg/min = 0.05 mcg/kg/min for 70 kg). "
        "Set <b>100% oxygen</b> (O₂ 10 L/min, air and N₂O 0 L/min in Machine) and keep ventilating.<br><br>"
        f"<i>{anesthetic_line}</i>"
    )


def _require_bronchospasm_improved(engine) -> tuple[bool, str]:
    improved = engine.state.bronchospasm < 0.5
    return improved, "" if improved else "Continue treating bronchospasm and reassess airway pressure"


def _require_epinephrine_started() -> Requirement:
    """Check that epinephrine was bolused or started after anaphylaxis began."""
    infusion_check = require_infusion_started(
        "epi", fail_message="Give epinephrine (bolus or infusion)", event_name="anaphylaxis",
    )

    def check(engine) -> tuple[bool, str]:
        if bolus_given_in_case(engine, "epi", event_name="anaphylaxis"):
            return True, ""
        return infusion_check(engine)
    return check


def create_anaphylaxis_scenario() -> Scenario:
    baseline_map = 90.0
    baseline_ppeak = 18.0
    onset_started = False
    crisis_visible = False

    def prepare(engine) -> None:
        nonlocal baseline_map, baseline_ppeak
        # Brown 2004 gave 1000 mL saline over 1-3 min.
        engine.fluid_infusion_rate_ml_min = 500.0
        # Start on realistic air/O2 maintenance (~40% FiO2) so severe bronchospasm
        # produces both high airway pressures and visible desaturation unless treated.
        equilibrate_maintenance_fio2(engine, target_fio2=0.40, min_air_l_min=1.4, air_vo2_factor=5.0)
        baseline_map = float(engine.state.map)
        baseline_ppeak = float(engine.state.paw_peak)
        # Set high-pressure alarm ~15 cmH2O above baseline (as in routine OR setup)
        # so severe anaphylactic bronchospasm triggers a visible Ppeak alarm.
        engine.alarms.thresholds["Ppeak_max"] = min(40, max(34, round(baseline_ppeak + 14)))

    def start(engine) -> None:
        nonlocal onset_started
        onset_started = True
        engine.anaphylaxis.onset_rate = 1.0 / 45.0
        engine.start_anaphylaxis()

    def crisis_apparent(engine) -> bool:
        if not onset_started:
            return False
        if engine.anaphylaxis.severity >= 0.90:
            return True
        if engine.anaphylaxis.severity < 0.55:
            return False
        return (
            engine.state.map < min(68.0, baseline_map - 20.0)
            or engine.state.paw_peak >= baseline_ppeak + 6.0
        )

    def mark_visible(_engine) -> None:
        nonlocal crisis_visible
        crisis_visible = True

    epinephrine = _require_epinephrine_started()
    fluids = require_fluid_bolus(
        1000.0,
        labels=("crystalloid", "saline"),
        fail_prefix="Start crystalloid resuscitation",
        event_name="anaphylaxis",
    )
    steps = [
        create_observe_step(
            "anaphylaxis",
            check=lambda _engine: (crisis_visible, ""),
        ),
        ScenarioStep(
            id="EPINEPHRINE",
            title="Give epinephrine",
            instruction=_epinephrine_instruction,
            check_requirements=require_all(
                epinephrine,
                require_oxygen_flow(),
                require_tracheal_ventilation,
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="FLUIDS",
            title="Give fluids",
            instruction=(
                "Give a rapid <b>1000 mL crystalloid bolus</b> alongside epinephrine. "
                "Reassess and repeat as needed; capillary leak often requires large volumes."
            ),
            check_requirements=fluids,
            target_tab="Circulation",
        ),
        ScenarioStep(
            id="STABILIZE",
            title="Stabilize and reassess",
            instruction=(
                "Select <b>Stop suspected trigger</b> if not already done, and continue support until "
                "<b>MAP ≥ 65 mmHg</b> and airway pressure improves. Give more epinephrine and fluids "
                "if hypotension or bronchospasm persists.<br><br>"
                "<i>Arrange serial tryptase levels and allergy follow-up after stabilization.</i>"
            ),
            check_requirements=require_all(
                require_crisis_resolved_with_map("anaphylaxis", fail_crisis="Stop the suspected trigger"),
                _require_bronchospasm_improved,
            ),
        ),
    ]

    return Scenario(
        id="anaphylaxis_response",
        name="Anaphylaxis response",
        briefing=CRISIS_BRIEFING,
        setup_engine=prepare,
        steps=steps,
        events=(
            CaseEvent("anaphylaxis_onset", _ANAPHYLAXIS_ONSET_S, "", start),
            CaseEvent(
                "anaphylaxis", _ANAPHYLAXIS_ONSET_S,
                "Shortly after an antibiotic dose, blood pressure falls and airway pressure rises.",
                mark_visible,
                when=crisis_apparent,
            ),
        ),
        actions=(CaseAction(
            "stop_trigger", "Stop suspected trigger",
            lambda engine: engine.stop_anaphylaxis(),
            lambda engine: engine.anaphylaxis.active and (crisis_visible or engine.anaphylaxis.severity >= 0.55),
            completed_label="Suspected trigger stopped",
        ),),
        outcomes=(
            CaseOutcome("epinephrine", "Epinephrine started", lambda engine: epinephrine(engine)[0],
                        "Give epinephrine promptly for severe perioperative anaphylaxis.", milestone=True),
            CaseOutcome("bronchospasm", "Bronchospasm improved",
                        lambda engine: _require_bronchospasm_improved(engine)[0],
                        "Reassess airway pressure and ventilation after treating bronchospasm."),
            *crisis_outcomes("anaphylaxis", fluids),
        ),
    )
