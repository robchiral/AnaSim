"""Malignant hyperthermia during sevoflurane anesthesia."""

from dataclasses import replace

from anasim.core.action_log import ACTION_DRUG_BOLUS, ACTION_EVENT_START
from anasim.core.monitors import display_int
from anasim.core.utils import clamp, hill_function
from anasim.machine.ventilator import VentSettings

from .base import (
    CaseEvent,
    CaseOutcome,
    Scenario,
    ScenarioStep,
    action_taken_in_case,
    check_etco2_range,
    format_unmet_int,
    normocapnia_bounds,
    preroll_scenario,
    require_all,
    require_infusion_started,
    require_map_at_least,
    require_oxygen_flow,
    require_oxygenation,
    require_tracheal_ventilation,
    stability_outcome,
)

_MH_PREROLL_S = 210.0


def _require_dantrolene(engine) -> tuple[bool, str]:
    doses = engine.actions.since_event("malignant_hyperthermia", ACTION_DRUG_BOLUS, labels=("dantrolene",))
    given, target = sum(dose.amount for dose in doses), 2.5 * engine.patient.weight
    if given >= target - 0.1:
        return True, ""
    if given > 0.0:
        shown_target = int(display_int(target))
        return False, f"Dantrolene: {format_unmet_int(given, low=shown_target)}/{shown_target} mg given"
    return False, "Give dantrolene 2.5 mg/kg IV"


def _require_vaporizer_off(engine) -> tuple[bool, str]:
    met = not engine.circuit.vaporizer_on
    return met, "" if met else "Turn off sevoflurane"


def _triggers_avoided(engine) -> bool:
    sux_post = engine.actions.since_event("malignant_hyperthermia", ACTION_DRUG_BOLUS, labels=("succinylcholine",))
    return _require_vaporizer_off(engine)[0] and not sux_post


def _require_controlled(engine) -> tuple[bool, str]:
    met = engine.malignant_hyperthermia.severity < 0.05
    if met:
        return True, ""
    sux_cp = engine.pk["succinylcholine"].state.c1
    active_sevo = max(0.0, engine.pk_sevo.state.mac - 0.25)
    trigger_drive = clamp(active_sevo / 0.55, 0.0, 2.5) + clamp(sux_cp / 0.4, 0.0, 2.0)
    inhibition = hill_function(engine.state.dantrolene_ce, 1.5 * (1.0 + 4.5 * trigger_drive), 3.0)
    if inhibition >= 0.30 and not engine.circuit.vaporizer_on and sux_cp <= 0.01:
        return False, "MH resolving; continue monitoring EtCO₂, HR, and rigidity"
    return False, "Signs of MH persist; repeat dantrolene"


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


def _mh_normocapnia_bounds(engine) -> tuple[int, int]:
    return normocapnia_bounds(engine, baseline_low=30.0)


def _require_etco2_normal(engine) -> tuple[bool, str]:
    return check_etco2_range(engine, *_mh_normocapnia_bounds(engine))


def _stabilize_instruction(engine) -> str:
    low, high = _mh_normocapnia_bounds(engine)
    return (
        f"Wean ventilation toward <b>EtCO₂ {low}-{high} mmHg</b> as hypermetabolism resolves. "
        "Turn off the <b>forced-air warmer</b>. Start <b>surface cooling</b> if temperature exceeds 39 °C "
        "or rises rapidly, and stop below 38 °C.<br><br>"
        "<i>Monitor blood gases, potassium, CK, myoglobin, glucose, coagulation, and urine output. "
        "Treat hyperkalemia with calcium, sodium bicarbonate, and insulin with glucose; avoid calcium channel "
        "blockers with dantrolene. Arrange at least 24 hours of ICU monitoring, redosing per protocol, "
        "and MH testing referral.</i>"
    )


def create_malignant_hyperthermia() -> Scenario:
    onset_vent: VentSettings | None = None
    onset_mv = 0.0

    def capture_baseline(engine) -> None:
        nonlocal onset_vent, onset_mv
        if engine.state.bair_hugger_target <= 0.0:
            engine.set_bair_hugger(38.0)
        # Routine low-flow air/oxygen maintenance so high-flow 100% O2 is required for washout.
        engine.set_fgf(1.0, 1.0, 0.0)
        onset_vent, onset_mv = replace(engine.vent.settings), engine.state.mv
        engine.start_malignant_hyperthermia()
        preroll_scenario(engine, _MH_PREROLL_S, dt=2.0, settle_s=15.0, sample_interval_s=5.0)

    def start(engine) -> None:
        if onset_vent is None:
            capture_baseline(engine)
        if engine.malignant_hyperthermia.active:
            engine.actions.record(engine.state.time, ACTION_EVENT_START, label="malignant_hyperthermia")
        else:
            engine.start_malignant_hyperthermia()

    def ventilation(engine) -> tuple[bool, str]:
        settings, before = engine.vent.settings, onset_vent
        increased = before is not None and (
            settings.rr > before.rr or settings.tv > before.tv or settings.p_insp > before.p_insp
        )
        if not (engine.vent.is_on and increased):
            return False, "Increase the ventilator rate or tidal volume"
        cap = 13.0 if engine.patient.copd_severity >= 0.5 else 15.0
        target = min(2.0 * onset_mv, max(cap, onset_mv + 2.0))
        mv = engine.state.mv
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
        setup_engine=capture_baseline,
        briefing=(
            "Forty minutes into sevoflurane anesthesia for surgery. "
            "Monitor the patient and manage any deterioration."
        ),
        events=(CaseEvent(
            "malignant_hyperthermia", 0.0,
            "EtCO₂ has risen despite unchanged ventilator settings, with tachycardia and muscle rigidity.",
            start,
        ),),
        steps=[
            ScenarioStep(
                id="STOP_TRIGGERS",
                title="Stop triggers and hyperventilate",
                instruction=(
                    "Unexplained rising EtCO₂, tachycardia, and rigidity suggest MH; temperature often rises late. "
                    "Call for help and the MH cart, and ask the surgeon to stop.<br><br>"
                    "Turn off <b>sevoflurane</b> and avoid succinylcholine. Set <b>100% oxygen</b> "
                    "(O₂ 10 L/min, air and N₂O 0 L/min), increase minute ventilation to <b>2-3 times baseline</b>, "
                    "and start <b>propofol</b> in Drugs.<br><br>"
                    "<i>Add activated charcoal filters if available; do not delay treatment to change the machine.</i>"
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
                    "Give <b>dantrolene 2.5 mg/kg IV</b> rapidly (175 mg for 70 kg), and repeat until "
                    "EtCO₂, HR, and rigidity improve.<br><br>"
                    "<i>Doses above 10 mg/kg are occasionally needed; if signs persist, reconsider the diagnosis.</i>"
                ),
                check_requirements=_require_dantrolene,
                target_tab="Medications",
            ),
            ScenarioStep(
                id="STABILIZE",
                title="Stabilize and reassess",
                instruction=_stabilize_instruction,
                check_requirements=require_all(
                    _require_vaporizer_off,
                    propofol,
                    controlled,
                    _require_etco2_normal,
                    _require_temperature_managed,
                    require_map_at_least(),
                    require_oxygenation(),
                ),
                target_tab="Machine",
            ),
        ],
        outcomes=(
            CaseOutcome("triggers", "Triggering agents stopped and avoided", _triggers_avoided,
                        "Stop volatile anesthetics and avoid succinylcholine."),
            CaseOutcome(
                "oxygen", "High-flow oxygen given",
                lambda engine: (
                    action_taken_in_case(engine, ACTION_EVENT_START, "malignant_hyperthermia")
                    and oxygen(engine)[0]
                ),
                "Give 100% oxygen at high flow to wash out the volatile agent.", milestone=True,
            ),
            CaseOutcome(
                "ventilation", "Ventilation increased",
                lambda engine: (
                    action_taken_in_case(engine, ACTION_EVENT_START, "malignant_hyperthermia")
                    and ventilation(engine)[0]
                ),
                "Increase minute ventilation to 2-3 times baseline and follow EtCO₂.", milestone=True,
            ),
            CaseOutcome("anesthesia", "Propofol anesthesia started", lambda engine: propofol(engine)[0],
                        "Maintain anesthesia with propofol after stopping the volatile agent."),
            CaseOutcome("dantrolene", "Dantrolene given", lambda engine: _require_dantrolene(engine)[0],
                        "Give dantrolene 2.5 mg/kg IV promptly and repeat until signs improve.", milestone=True),
            CaseOutcome("controlled", "Hypermetabolism controlled", lambda engine: controlled(engine)[0],
                        "Repeat dantrolene until EtCO₂, HR, and rigidity improve."),
            stability_outcome("Reassess oxygenation and blood pressure while treating the crisis."),
        ),
    )
