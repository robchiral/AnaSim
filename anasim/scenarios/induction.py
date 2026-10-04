"""Adult general anesthesia inductions with volatile or TIVA maintenance."""

from .base import (
    Scenario,
    ScenarioStep,
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
    require_map_above,
    require_preoxygenation,
    require_propofol_cp,
    require_rocuronium_cp,
    require_tof_below,
    require_vaporizer_started,
    require_ventilator_running,
)


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
            "Set <b>O₂ 10 L/min</b> and <b>air 0 L/min</b>.<br><br>"
            "<i>High flow washes nitrogen out of the circuit.</i>"
        ),
        check_requirements=require_fgf_set_for_preox(),
        target_tab="Machine",
    )


def _preoxygenate() -> ScenarioStep:
    return ScenarioStep(
        id="PREOXYGENATE",
        title="Preoxygenate",
        instruction=(
            "Keep the mask on for <b>about 3 minutes</b> of tidal breathing, or 8 deep "
            "breaths in 1 minute. Aim for end-tidal O₂ ≥ 90%.<br><br>"
            "<i>A healthy adult then tolerates about 8 minutes of apnea before SpO₂ falls below 90%.</i>"
        ),
        check_requirements=require_preoxygenation(),
        target_tab="Machine",
    )


def _confirm_loc() -> ScenarioStep:
    return ScenarioStep(
        id="CONFIRM_LOC",
        title="Confirm loss of consciousness",
        instruction=(
            "Check for no response to voice, loss of the lash reflex, apnea, and "
            "<b>BIS < 60</b>.<br><br>"
            "<i>Give the neuromuscular blocker only after the patient is unconscious.</i>"
        ),
        check_requirements=require_bis_below(60),
    )


def _mask_ventilate() -> ScenarioStep:
    return ScenarioStep(
        id="MASK_VENTILATE",
        title="Mask ventilate",
        instruction=(
            "Choose <b>Start bag ventilation</b>. Watch for chest rise, an EtCO₂ "
            "waveform, and stable SpO₂.<br><br>"
            "<i>Propofol causes apnea; ventilate until the tube is in.</i>"
        ),
        check_requirements=require_bag_mask_started(),
        target_tab="Machine",
    )


def _give_rocuronium() -> ScenarioStep:
    return ScenarioStep(
        id="GIVE_NMB",
        title="Give rocuronium",
        instruction=(
            "Give rocuronium <b>0.6 mg/kg</b> (about 50 mg for 70 kg). "
            "Use 1.2 mg/kg for rapid sequence induction.<br><br>"
            "<i>Intubating conditions develop in 60-90 seconds.</i>"
        ),
        check_requirements=require_all(
            require_drug_bolus("roc", "Give the rocuronium bolus"),
            require_rocuronium_cp(0.5),
        ),
        target_tab="Medications",
    )


def _wait_for_paralysis() -> ScenarioStep:
    return ScenarioStep(
        id="WAIT_PARALYSIS",
        title="Confirm paralysis",
        instruction=(
            "Wait for <b>TOF ≤ 5%</b> before laryngoscopy.<br><br>"
            "<i>Partial block makes laryngoscopy harder and can cause coughing or vocal cord injury.</i>"
        ),
        check_requirements=require_tof_below(5),
    )


def _intubate() -> ScenarioStep:
    return ScenarioStep(
        id="INTUBATE",
        title="Intubate",
        instruction=(
            "Perform laryngoscopy and place the tube: select <b>ETT</b>.<br><br>"
            "<i>Laryngoscopy raises HR and BP for about a minute. Opioids blunt "
            "the rise most; lidocaine blunts it slightly.</i>"
        ),
        check_requirements=require_airway_selected("ETT"),
        target_tab="Machine",
        on_met=lambda engine: engine.start_disturbance("stim_intubation_pulse"),
    )


def _confirm_tube() -> ScenarioStep:
    return ScenarioStep(
        id="CONFIRM_ETT",
        title="Confirm tube placement",
        instruction=(
            "Start the ventilator: <b>VCV, Vt 6-8 mL/kg predicted body weight "
            "(about 500 mL), RR 12, PEEP 5</b>. Confirm sustained EtCO₂, chest "
            "rise, and bilateral breath sounds.<br><br>"
            "<i>No EtCO₂ means the tube is not in the trachea until proven otherwise.</i>"
        ),
        check_requirements=require_all(
            require_ventilator_running(),
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
                "While the patient preoxygenates, give fentanyl <b>1-2 mcg/kg</b> "
                "(about 100 mcg for 70 kg). Midazolam 1-2 mg is often given before "
                "entering the room.<br><br>"
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
                "Give lidocaine <b>1-1.5 mg/kg</b> (about 100 mg), then propofol "
                "<b>1.5-2.5 mg/kg</b> (about 150 mg for 70 kg; less for older or frail "
                "patients).<br><br>"
                "<i>Lidocaine reduces propofol injection pain.</i>"
            ),
            check_requirements=require_all(
                require_drug_bolus("lidocaine", "Give the lidocaine bolus"),
                require_drug_bolus("propofol", "Give the propofol induction bolus"),
                require_propofol_cp(2.0),
            ),
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
            check_requirements=require_vaporizer_started(),
            target_tab="Machine",
        ),
        _wait_for_paralysis(),
        _intubate(),
        _confirm_tube(),
        ScenarioStep(
            id="MAINTENANCE",
            title="Begin maintenance",
            instruction=(
                "Reduce fresh gas to <b>2 L/min</b>, for example O₂ 1 L/min and air "
                "1 L/min. Bring end-tidal sevoflurane to <b>0.7-1 MAC</b> (1 MAC is "
                "about 2% at age 40). Keep <b>MAP ≥ 65</b>; treat post-induction "
                "hypotension with phenylephrine 50-100 mcg.<br><br>"
                "<i>At low flow, set the dial above the end-tidal target.</i>"
            ),
            check_requirements=require_all(
                require_fgf_reduced(2.0),
                require_mac_above(0.7),
                require_map_above(65),
            ),
            target_tab="Machine",
        ),
    ]

    return Scenario(
        id="induction_balanced",
        name="Induction (Balanced)",
        description="Fentanyl, lidocaine, propofol, and rocuronium induction with sevoflurane maintenance.",
        steps=steps,
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
                "Start remifentanil: <b>TCI 2-4 ng/mL</b> or <b>0.1-0.25 mcg/kg/min</b>.<br><br>"
                "<i>The opioid blunts the response to laryngoscopy.</i>"
            ),
            check_requirements=require_infusion_started(
                "remi", fail_message="Start Remifentanil TCI or infusion"
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="INDUCE",
            title="Induce",
            instruction=(
                "Give lidocaine <b>1-1.5 mg/kg</b> (about 100 mg), then propofol "
                "<b>1.5-2.5 mg/kg</b> (about 150 mg for 70 kg), and start propofol "
                "<b>TCI 4-6 µg/mL</b>.<br><br>"
                "<i>The infusion keeps the patient anesthetized after the bolus redistributes.</i>"
            ),
            check_requirements=require_all(
                require_drug_bolus("lidocaine", "Give the lidocaine bolus"),
                require_drug_bolus("propofol", "Give the propofol induction bolus"),
                require_propofol_cp(2.0),
                require_infusion_started(
                    "propofol", fail_message="Start the propofol infusion"
                ),
            ),
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
                "Keep propofol and remifentanil running. Reduce fresh gas to "
                "<b>2 L/min</b>. Keep <b>MAP ≥ 65</b> and BIS 40-60.<br><br>"
                "<i>Typical targets: propofol 3-4 µg/mL, remifentanil 2-4 ng/mL.</i>"
            ),
            check_requirements=require_all(
                require_infusion_running("propofol", "Propofol infusion not running"),
                require_infusion_running("remi", "Remifentanil infusion not running"),
                require_fgf_reduced(2.0),
                require_map_above(65),
            ),
            target_tab="Medications",
        ),
    ]

    return Scenario(
        id="induction_tiva",
        name="Induction (TIVA)",
        description="Remifentanil, lidocaine, and propofol induction with propofol-remifentanil maintenance.",
        steps=steps,
    )
