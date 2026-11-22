"""Adult general anesthesia inductions with volatile or TIVA maintenance."""

from anasim.core.projection import loss_of_response

from .base import (
    CaseOutcome,
    Scenario,
    ScenarioStep,
    format_unmet_int,
    infusion_running,
    monitor_value,
    require_airway_selected,
    require_all,
    require_bag_mask_started,
    require_bis_below,
    require_drug_bolus,
    require_etco2_above,
    require_fgf_reduced,
    require_fgf_set_for_preox,
    require_infusion_running,
    require_infusion_started,
    require_mac_above,
    require_maintenance_depth,
    require_map_at_least,
    require_oxygenation,
    require_preoxygenation,
    require_tof_count_at_most,
    require_tracheal_ventilation,
    require_vaporizer_started,
    stability_outcome,
    tracheal_ventilation_confirmed,
)


def _require_balanced_handover_depth(engine) -> tuple[bool, str]:
    """Allow BIS 30-60 while the induction propofol bolus overlaps early sevoflurane wash-in."""
    bis = monitor_value(engine, "bis")
    low = 30.0 if engine.state.propofol_ce > 0.3 else 40.0
    met = low <= bis <= 60.0
    shown_bis = format_unmet_int(bis, low=low, high=60.0)
    return met, "" if met else f"BIS: {shown_bis}/{int(low)}-60"


def _maintained(engine) -> bool:
    hypnotic_running = infusion_running(engine, "propofol", "etomidate", "ketamine", "midazolam")
    return (hypnotic_running or engine.circuit.vaporizer_on) and loss_of_response(engine) > 0.9


OUTCOMES = (
    CaseOutcome("preoxygenation", "Preoxygenation established",
                lambda engine: loss_of_response(engine) < 0.5 and require_preoxygenation(engine)[0],
                "Establish an oxygen reserve before induction and airway instrumentation.", milestone=True),
    CaseOutcome("airway", "Tracheal ventilation established", tracheal_ventilation_confirmed,
                "Confirm tube placement with sustained capnography and adequate ventilation."),
    CaseOutcome("anesthesia", "Anesthesia maintained", _maintained,
                "Maintain hypnosis after securing the airway."),
    stability_outcome("Reassess oxygenation and blood pressure after induction."),
)


def _briefing(technique: str) -> str:
    return f"Prepare for general anesthesia. Induce, secure the airway, and establish {technique} maintenance."


def _apply_mask() -> ScenarioStep:
    return ScenarioStep(
        id="APPLY_MASK",
        title="Apply facemask",
        instruction=(
            "Select <b>Facemask</b> to connect the patient to the breathing circuit.<br><br>"
            "<i>A tight seal keeps room air out during preoxygenation.</i>"
        ),
        check_requirements=require_airway_selected("Mask"),
        target_tab="Machine",
    )


def _set_fgf_preox() -> ScenarioStep:
    return ScenarioStep(
        id="SET_FGF_PREOX",
        title="Set fresh gas flow",
        instruction=(
            "Set <b>O₂ 10 L/min</b>, air 0 L/min, and N₂O 0 L/min.<br><br>"
            "<i>High flow washes nitrogen out of the circuit.</i>"
        ),
        check_requirements=require_fgf_set_for_preox,
        target_tab="Machine",
    )


def _preoxygenate() -> ScenarioStep:
    return ScenarioStep(
        id="PREOXYGENATE",
        title="Preoxygenate",
        instruction=(
            "Maintain a tight mask seal during tidal breathing until <b>end-tidal O₂ ≥ 90%</b>. "
            "If wash-in stalls, check oxygen flow, mask seal, and ventilation; assist breathing if needed.<br><br>"
            "<i>Preoxygenation builds oxygen reserve before apnea.</i>"
        ),
        check_requirements=require_preoxygenation,
        target_tab="Machine",
    )


def _confirm_loc() -> ScenarioStep:
    return ScenarioStep(
        id="CONFIRM_LOC",
        title="Confirm loss of consciousness",
        instruction=(
            "Wait for <b>BIS < 60</b> before giving rocuronium.<br><br>"
            "<i>At the bedside, confirm unresponsiveness clinically; apnea alone does not establish unconsciousness.</i>"
        ),
        check_requirements=require_bis_below(60),
    )


_PROPOFOL_INFUSION = require_infusion_started("propofol", fail_message="Start the propofol infusion")
_PROPOFOL_BOLUS = require_drug_bolus("propofol", "Give the titrated propofol induction bolus")


def _require_propofol_induction(engine) -> tuple[bool, str]:
    """Accept propofol TCI, which delivers its own loading dose, or a bolus with a manual infusion."""
    running, message = _PROPOFOL_INFUSION(engine)
    if not running:
        return False, message
    if engine.config.tci_enabled and engine.get_drug_state("propofol")["target"] > 0:
        return True, ""
    return _PROPOFOL_BOLUS(engine)


def _mask_ventilate() -> ScenarioStep:
    return ScenarioStep(
        id="MASK_VENTILATE",
        title="Mask ventilate",
        instruction=(
            "Set the <b>APL valve</b> to 20 cmH₂O, then select <b>Start bag ventilation</b>. "
            "Confirm an EtCO₂ waveform and stable SpO₂, and keep ventilating until intubation.<br><br>"
            "<i>Verify chest rise and mask seal at the bedside.</i>"
        ),
        check_requirements=require_all(
            require_bag_mask_started,
            require_etco2_above(20),
            require_oxygenation(),
        ),
        target_tab="Machine",
    )


def _give_rocuronium() -> ScenarioStep:
    return ScenarioStep(
        id="GIVE_NMB",
        title="Give rocuronium",
        instruction=(
            "Give <b>rocuronium 0.6 mg/kg</b> (42 mg for 70 kg) and continue mask ventilation "
            "while the block develops."
        ),
        check_requirements=require_drug_bolus("roc", "Give the rocuronium bolus"),
        target_tab="Medications",
    )


def _wait_for_paralysis() -> ScenarioStep:
    return ScenarioStep(
        id="WAIT_PARALYSIS",
        title="Assess neuromuscular block",
        instruction=(
            "Continue mask ventilation with <b>BIS < 60</b> until the <b>TOF count reaches 0-1</b>.<br><br>"
            "<i>A TOF ratio requires four twitches. Intubating conditions also depend on anesthetic depth and muscle group.</i>"
        ),
        check_requirements=require_all(require_bis_below(60), require_tof_count_at_most(1)),
    )


def _intubate() -> ScenarioStep:
    return ScenarioStep(
        id="INTUBATE",
        title="Intubate",
        instruction=(
            "Select <b>Tracheal tube</b> to intubate, and watch HR and blood pressure during laryngoscopy.<br><br>"
            "<i>Adequate hypnosis and analgesia blunt the sympathetic response.</i>"
        ),
        check_requirements=require_airway_selected("ETT"),
        target_tab="Machine",
    )


def _confirm_tube() -> ScenarioStep:
    return ScenarioStep(
        id="CONFIRM_ETT",
        title="Confirm tube placement",
        instruction=(
            "Select <b>Start ventilator</b> with <b>VCV, Vt 6-8 mL/kg predicted body weight, "
            "RR 12/min, and PEEP 5 cmH₂O</b>. Confirm sustained capnography and adjust ventilation to EtCO₂.<br><br>"
            "<i>Also check chest rise and breath sounds. Without sustained exhaled CO₂, "
            "immediately verify tube position, ventilation, and circulation.</i>"
        ),
        check_requirements=require_all(
            require_tracheal_ventilation,
            require_etco2_above(20),
        ),
        target_tab="Machine",
    )


def create_induction_balanced() -> Scenario:
    steps = [
        _apply_mask(),
        _set_fgf_preox(),
        ScenarioStep(
            id="GIVE_OPIOID",
            title="Give fentanyl",
            instruction=(
                "While the patient preoxygenates, give <b>fentanyl 1-2 mcg/kg</b> "
                "(about 100 mcg for 70 kg). Watch ventilation and blood pressure.<br><br>"
                "<i>Fentanyl peaks 3-5 minutes after injection, in time for laryngoscopy.</i>"
            ),
            check_requirements=require_drug_bolus("fentanyl", "Give the fentanyl bolus"),
            target_tab="Medications",
        ),
        _preoxygenate(),
        ScenarioStep(
            id="INDUCE",
            title="Induce",
            instruction=(
                "Give <b>propofol 1.5-2.5 mg/kg</b> titrated to effect "
                "(about 150 mg for 70 kg; reduce in older or hypotensive patients).<br><br>"
                "<i>Lidocaine 20-40 mg IV beforehand can reduce injection pain.</i>"
            ),
            check_requirements=require_drug_bolus("propofol", "Give the propofol induction bolus"),
            target_tab="Medications",
        ),
        _confirm_loc(),
        _mask_ventilate(),
        _give_rocuronium(),
        ScenarioStep(
            id="START_SEVO",
            title="Start sevoflurane",
            instruction=(
                "While rocuronium takes effect, set sevoflurane to <b>2-3%</b> and keep "
                "fresh gas flow high.<br><br>"
                "<i>The propofol bolus wears off within 5-10 minutes; sevoflurane "
                "keeps the patient anesthetized.</i>"
            ),
            check_requirements=require_vaporizer_started,
            target_tab="Machine",
        ),
        _wait_for_paralysis(),
        _intubate(),
        _confirm_tube(),
        ScenarioStep(
            id="MAINTENANCE",
            title="Begin maintenance",
            instruction=(
                "Reduce fresh gas to <b>2 L/min</b> (for example, O₂ 1 L/min and air 1 L/min). "
                "Titrate sevoflurane to <b>0.7-1 MAC</b> and <b>BIS 40-60</b> "
                "(30-60 while the induction bolus clears). Keep <b>MAP ≥ 65 mmHg</b>; assess depth, "
                "volume, and HR before giving a vasopressor such as phenylephrine 50-100 mcg IV.<br><br>"
                "<i>At low flow, follow the gas monitor because dial changes equilibrate slowly.</i>"
            ),
            check_requirements=require_all(
                require_fgf_reduced(2.0),
                require_mac_above(0.7),
                _require_balanced_handover_depth,
                require_map_at_least(65),
                require_tracheal_ventilation,
                require_oxygenation(),
            ),
            target_tab="Machine",
        ),
    ]

    return Scenario(
        id="induction_balanced",
        name="Balanced induction",
        briefing=_briefing("balanced"),
        steps=steps,
        outcomes=OUTCOMES,
    )


def create_induction_tiva() -> Scenario:
    steps = [
        _apply_mask(),
        _set_fgf_preox(),
        _preoxygenate(),
        ScenarioStep(
            id="START_ANALGESIA",
            title="Start remifentanil",
            instruction=(
                "Start remifentanil at <b>0.1-0.25 mcg/kg/min</b>. Support ventilation if needed.<br><br>"
                "<i>The opioid blunts the response to laryngoscopy.</i>"
            ),
            tci_instruction=(
                "Start remifentanil at <b>TCI 2-4 ng/mL</b> or 0.1-0.25 mcg/kg/min. "
                "Support ventilation if needed.<br><br>"
                "<i>The opioid blunts the response to laryngoscopy.</i>"
            ),
            check_requirements=require_infusion_started(
                "remi", fail_message="Start the remifentanil infusion"
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="INDUCE",
            title="Induce",
            instruction=(
                "Give <b>propofol 1.5-2.5 mg/kg</b> (about 150 mg for 70 kg; reduce in older or hypotensive patients) "
                "and start an infusion at <b>100-200 mcg/kg/min</b>, titrated to effect.<br><br>"
                "<i>Lidocaine 20-40 mg IV beforehand can reduce injection pain.</i>"
            ),
            tci_instruction=(
                "Start propofol <b>TCI 4 µg/mL</b> (lower in older or hypotensive patients) and titrate to response. "
                "TCI delivers the loading dose automatically, or you can use a manual bolus and infusion."
            ),
            check_requirements=_require_propofol_induction,
            target_tab="Medications",
        ),
        _confirm_loc(),
        _mask_ventilate(),
        _give_rocuronium(),
        _wait_for_paralysis(),
        _intubate(),
        _confirm_tube(),
        ScenarioStep(
            id="MAINTENANCE",
            title="Confirm maintenance",
            instruction=(
                "Keep propofol and remifentanil running, titrated to <b>BIS 40-60</b> and <b>MAP ≥ 65 mmHg</b>. "
                "Reduce fresh gas to <b>2 L/min</b> in Machine with at least 1 L/min O₂."
            ),
            tci_instruction=(
                "Keep propofol and remifentanil running, with <b>BIS 40-60</b> and <b>MAP ≥ 65 mmHg</b>. "
                "Reduce fresh gas to <b>2 L/min</b> in Machine with at least 1 L/min O₂.<br><br>"
                "<i>Typical targets: propofol 3-4 µg/mL, remifentanil 2-4 ng/mL.</i>"
            ),
            check_requirements=require_all(
                require_infusion_running("propofol", "Propofol infusion not running"),
                require_infusion_running("remi", "Remifentanil infusion not running"),
                require_fgf_reduced(2.0),
                require_maintenance_depth,
                require_map_at_least(65),
                require_tracheal_ventilation,
                require_oxygenation(),
            ),
            target_tab="Medications",
        ),
    ]

    return Scenario(
        id="induction_tiva",
        name="TIVA induction",
        briefing=_briefing("intravenous"),
        steps=steps,
        outcomes=OUTCOMES,
    )
