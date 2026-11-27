"""High / total spinal anesthesia crisis scenario."""

from anasim.core.action_log import (
    ACTION_DRUG_BOLUS,
    ACTION_EVENT_START,
    ACTION_FLUID,
    ACTION_INFUSION_RATE,
    ACTION_TCI_TARGET,
)
from anasim.core.projection import loss_of_response
from anasim.core.state import AirwayType

from .base import (
    CaseEvent,
    CaseOutcome,
    Scenario,
    ScenarioStep,
    action_taken_in_case,
    create_observe_step,
    format_unmet_int,
    monitor_value,
    patient_stable,
    require_map_at_least,
    require_oxygen_flow,
    require_oxygenation,
)


def _setup_high_spinal(engine) -> None:
    """Awake patient with an epidural catheter in place just before a surgical top-up bolus."""
    engine.place_epidural(True)
    engine.set_fgf(10.0, 0.0, 0.0)
    engine.set_apl_valve(20.0)


def _trigger_high_spinal(engine) -> None:
    engine.trigger_high_spinal()


def _high_spinal_onset_reached(engine) -> bool:
    return (
        action_taken_in_case(engine, ACTION_EVENT_START, "high_spinal")
        and (
            engine.neuraxial.sensory_level >= 11.0
            or engine.state.map < 80.0
            or engine.state.hr < 60.0
        )
    )


def _vasoactive_given(engine) -> bool:
    boluses = engine.actions.since_event(
        "high_spinal",
        ACTION_DRUG_BOLUS,
        labels=("atropine", "glyco", "ephedrine", "epi", "phenyl", "nore", "vaso"),
    )
    infusions = engine.actions.since_event(
        "high_spinal",
        ACTION_INFUSION_RATE,
        labels=("nore", "epi", "phenyl", "vaso"),
    )
    tci_targets = engine.actions.since_event(
        "high_spinal",
        ACTION_TCI_TARGET,
        labels=("nore", "epi", "phenyl", "vaso"),
    )
    return any(r.amount > 0 for r in (*boluses, *infusions, *tci_targets))


def _fluid_given(engine) -> bool:
    return (
        sum(
            r.amount
            for r in engine.actions.since_event(
                "high_spinal", ACTION_FLUID, labels=("crystalloid", "saline", "albumin", "colloid", "blood"),
            )
        )
        >= 500.0
    )


def _airway_supported(engine) -> bool:
    s = engine.state
    return (
        s.airway_mode in (AirwayType.ETT, AirwayType.SGA, AirwayType.FONA)
        and not s.esophageal_intubation
        and (engine.vent.is_on or engine.bag_mask_active)
    )


def _preload_supported(engine) -> bool:
    return _fluid_given(engine) and (
        engine.patient.gestational_age_weeks <= 0.0 or engine.hemo.position == "left_lateral"
    )


def create_high_spinal() -> Scenario:
    def _resus_instruction(engine) -> str:
        pos_note = (
            "Set <b>Position to Left lateral</b> (left uterine displacement), "
            if engine.patient.gestational_age_weeks > 0.0
            else ""
        )
        return (
            "Rapidly ascending sensory block with hypotension and bradycardia after an epidural bolus "
            "indicates <b>accidental intrathecal injection (high / total spinal)</b>. "
            f"{pos_note}connect <b>Facemask</b> with <b>100% O₂</b> (O₂ ≥ 6 L/min, Air and N₂O 0 L/min), and start a "
            "<b>≥ 500 mL IV fluid bolus</b>.<br><br>"
            "<i>Restoring venous preload counters sympathectomy pooling and Bezold-Jarisch reflex bradycardia (Pollard 2001).</i>"
        )

    def check_o2_fluid(engine) -> tuple[bool, str]:
        missing = []
        if engine.patient.gestational_age_weeks > 0.0 and engine.hemo.position != "left_lateral":
            missing.append("Set Position to Left lateral")
        if not _fluid_given(engine):
            missing.append("Give >= 500 mL resuscitation fluid bolus")
        if engine.state.airway_mode == AirwayType.NONE:
            missing.append("Connect Facemask or airway")
        o2_ok, o2_msg = require_oxygen_flow(6.0)(engine)
        if not o2_ok:
            missing.append(o2_msg)
        return not missing, "; ".join(missing)

    def check_observe_high_spinal(engine) -> tuple[bool, str]:
        return _high_spinal_onset_reached(engine), ""

    def check_vasoactive(engine) -> tuple[bool, str]:
        met = _vasoactive_given(engine)
        return met, "" if met else "Give atropine / ephedrine / epinephrine / phenylephrine"

    def check_assist_vent(engine) -> tuple[bool, str]:
        if _airway_supported(engine):
            return True, ""
        missing = []
        if not (engine.bag_mask_active or engine.vent.is_on):
            missing.append("Start 100% O2 bag-mask ventilation")
        if engine.neuraxial.unconsciousness < 0.8 and loss_of_response(engine, include_non_drug=False) < 0.5:
            missing.append("Support ventilation until total spinal loss of consciousness (or induce gently)")
        return not missing, "; ".join(missing)

    def check_airway(engine) -> tuple[bool, str]:
        met = _airway_supported(engine)
        return met, "" if met else "Secure airway (Tracheal tube or SGA) and start ventilation"

    def check_stable(engine) -> tuple[bool, str]:
        if not _airway_supported(engine):
            return False, "Maintain controlled ventilation"
        missing = []
        map_ok, map_msg = require_map_at_least(65.0)(engine)
        if not map_ok:
            missing.append(map_msg)
        hr_val = monitor_value(engine, "hr")
        if hr_val < 55.0:
            missing.append(f"HR: {format_unmet_int(hr_val, low=55.0)}/55+ bpm")
        elif engine.state.hr < 55.0:
            missing.append("Reassess heart rate")
        spo2_ok, spo2_msg = require_oxygenation(94.0)(engine)
        if not spo2_ok:
            missing.append(spo2_msg)
        if not missing and not patient_stable(engine):
            missing.append("Reassess oxygenation and perfusion")
        return not missing, "; ".join(missing)

    return Scenario(
        id="high_spinal",
        name="High / total spinal anesthesia",
        briefing=(
            "Awake patient with an epidural catheter in place just receiving a surgical bolus "
            "(15 mL of 0.5% bupivacaine). Watch closely for ascending block height, hemodynamic "
            "collapse, and respiratory compromise."
        ),
        setup_engine=_setup_high_spinal,
        events=(
            CaseEvent(
                id="accidental_intrathecal",
                at=2.0,
                cue="",
                apply=_trigger_high_spinal,
            ),
            CaseEvent(
                id="high_spinal_onset",
                at=20.0,
                when=_high_spinal_onset_reached,
                cue=(
                    "Moments after the epidural top-up, the patient reports hand tingling, "
                    "difficulty taking a deep breath, and dizziness. Blood pressure and heart rate are falling."
                ),
                apply=lambda _e: None,
            ),
            CaseEvent(
                id="apnea_loc_cue",
                at=85.0,
                when=lambda e: e.neuraxial.sensory_level >= 27.0,
                cue=(
                    "The block has ascended past C4: speech is lost, diaphragmatic effort is failing, "
                    "and hypotension with bradycardia is deepening."
                ),
                apply=lambda _e: None,
            ),
        ),
        steps=[
            create_observe_step("high_spinal", check=check_observe_high_spinal),
            ScenarioStep(
                id="FLUID_O2",
                title="Support oxygenation and preload",
                instruction=_resus_instruction,
                check_requirements=check_o2_fluid,
                target_tab="Circulation",
            ),
            ScenarioStep(
                id="VASOACTIVE",
                title="Treat hypotension and bradycardia",
                instruction=(
                    "Give <b>atropine</b> (0.5–1 mg IV) and <b>ephedrine</b> (10–25 mg), "
                    "<b>phenylephrine</b>, or <b>epinephrine</b> (10–50 mcg, escalating promptly if "
                    "severe bradycardia/hypotension persists).<br><br>"
                    "<i>T1–T4 cardioaccelerator block plus low preload triggers Bezold-Jarisch vagal bradycardia (Pollard 2001); prompt epinephrine prevents progression to arrest (Caplan 1988).</i>"
                ),
                check_requirements=check_vasoactive,
                target_tab="Medications",
            ),
            ScenarioStep(
                id="ASSIST_VENTILATION",
                title="Assist ventilation as phrenic block ascends",
                instruction=(
                    "Start <b>Bag-mask</b> ventilation on 100% O₂ under Machine as breathing weakens "
                    "and continue until total spinal loss of consciousness develops (or give a gentle "
                    "cardiostable induction dose such as etomidate if intubating earlier)."
                ),
                check_requirements=check_assist_vent,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="SECURE_AIRWAY",
                title="Secure the airway and ventilate",
                instruction=(
                    "Place a <b>Tracheal tube</b> (or rescue <b>SGA</b>) and start mechanical ventilation.<br><br>"
                    "<i>At total spinal loss of consciousness, avoid extra propofol that worsens hypotension.</i>"
                ),
                check_requirements=check_airway,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="STABILIZE_SPINAL",
                title="Stabilize hemodynamics and oxygenation",
                instruction=(
                    "Titrate vasopressor support and ventilation until <b>MAP ≥ 65 mmHg</b>, "
                    "<b>HR ≥ 55 bpm</b>, and <b>SpO₂ ≥ 94%</b>."
                ),
                check_requirements=check_stable,
            ),
        ],
        outcomes=(
            CaseOutcome(
                id="fluid_resuscitation",
                label="Restored venous preload with ≥ 500 mL IV fluid",
                check=_preload_supported,
                review="Rapid IV fluid loading (plus left uterine displacement in pregnancy) restores preload and attenuates Bezold-Jarisch reflex bradycardia (Pollard 2001).",
                milestone=True,
            ),
            CaseOutcome(
                id="vasoactive_support",
                label="Treated bradycardia and hypotension with vasoactive drugs",
                check=_vasoactive_given,
                review="Use stepwise atropine, ephedrine/phenylephrine, and early epinephrine for high-spinal hypotension and bradycardia (Caplan 1988; Pollard 2001).",
                milestone=True,
            ),
            CaseOutcome(
                id="airway_secured",
                label="Secured airway and established controlled ventilation",
                check=_airway_supported,
                review="Phrenic paralysis (C3–C5) and total spinal require immediate oxygenation, airway control, and mechanical ventilation.",
                milestone=True,
            ),
            CaseOutcome(
                id="stabilized",
                label="Stabilized circulation and oxygenation (MAP ≥ 65, HR ≥ 55, SpO₂ ≥ 94%)",
                check=lambda e: _airway_supported(e) and patient_stable(e) and e.state.hr >= 55.0,
                review="Maintain perfusion (MAP ≥ 65 mmHg, HR ≥ 55 bpm) and oxygenation until the neuraxial block regresses.",
            ),
        ),
    )
