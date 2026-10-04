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
    require_maintenance_depth,
    require_map_at_least,
    require_oxygenation,
    require_preoxygenation,
    require_tof_at_most,
    require_tracheal_ventilation,
    require_vaporizer_started,
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
            "Keep a tight mask seal and continue tidal breathing until <b>end-tidal O₂ ≥ 90%</b>. "
            "Allow several minutes and follow the measured value. If wash-in stalls, "
            "check the oxygen supply, mask seal, and ventilation; assist breathing if needed.<br><br>"
            "<i>Preoxygenation increases oxygen reserve. Time to desaturation varies with the patient.</i>"
        ),
        check_requirements=require_preoxygenation,
        target_tab="Machine",
    )


def _confirm_loc() -> ScenarioStep:
    return ScenarioStep(
        id="CONFIRM_LOC",
        title="Confirm loss of consciousness",
        instruction=(
            "Wait for <b>BIS < 60</b> before giving rocuronium. "
            "BIS is the simulator's approximate marker of hypnosis.<br><br>"
            "<i>In practice, assess response to voice and other clinical signs. Apnea alone does not establish unconsciousness.</i>"
        ),
        check_requirements=require_bis_below(60),
    )


def _mask_ventilate() -> ScenarioStep:
    return ScenarioStep(
        id="MASK_VENTILATE",
        title="Mask ventilate",
        instruction=(
            "Choose <b>Start bag ventilation</b> and confirm an EtCO₂ waveform "
            "and stable SpO₂. Continue ventilation until the tracheal tube is placed. "
            "Assess chest movement and mask seal in practice."
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
            "Give rocuronium <b>0.6 mg/kg</b> for this routine induction "
            "(42 mg for 70 kg). Continue mask ventilation while the block develops."
        ),
        check_requirements=require_drug_bolus("roc", "Give the rocuronium bolus"),
        target_tab="Medications",
    )


def _wait_for_paralysis() -> ScenarioStep:
    return ScenarioStep(
        id="WAIT_PARALYSIS",
        title="Assess neuromuscular block",
        instruction=(
            "Continue mask ventilation with <b>BIS < 60</b> and wait for the modeled <b>TOF ≤ 5%</b>. "
            "This indicates substantial block in the simulator; clinical intubating "
            "conditions also depend on anesthetic depth and the muscles assessed."
        ),
        check_requirements=require_all(require_bis_below(60), require_tof_at_most(5)),
    )


def _intubate() -> ScenarioStep:
    return ScenarioStep(
        id="INTUBATE",
        title="Intubate",
        instruction=(
            "Select <b>Tracheal tube</b> to simulate laryngoscopy and intubation. "
            "This also triggers a brief airway stimulus. Observe HR and blood pressure; "
            "adequate anesthesia and analgesia reduce the response."
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
            "Start the ventilator with <b>VCV, Vt 6-8 mL/kg predicted body weight, "
            "RR 12/min, and PEEP 5 cmH₂O</b>. Confirm sustained capnography "
            "and adjust ventilation to EtCO₂.<br><br>"
            "<i>In practice, assess chest movement and breath sounds. Absent sustained "
            "exhaled CO₂ requires immediate assessment of tube position, ventilation, and circulation.</i>"
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
                "While the patient preoxygenates, give fentanyl <b>1-2 mcg/kg</b> "
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
                "Give propofol <b>1.5-2.5 mg/kg</b>, titrated to effect "
                "(about 150 mg for 70 kg). Use less in older patients or with hypotension. "
                "Optional lidocaine 20-40 mg IV before propofol can reduce injection pain."
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
                "Reduce fresh gas to <b>2 L/min</b>, for example O₂ 1 L/min and air "
                "1 L/min. Titrate sevoflurane to end-tidal <b>0.7-1 MAC</b> and BIS 40-60 (1 MAC is "
                "about 2% at age 40). Keep <b>MAP ≥ 65</b>; treat post-induction "
                "hypotension after assessing anesthetic depth, volume, and HR. "
                "Phenylephrine 50-100 mcg IV is one option for vasodilatory hypotension.<br><br>"
                "<i>At low flow, follow the gas monitor because dial changes take longer to reach the patient.</i>"
            ),
            check_requirements=require_all(
                require_fgf_reduced(2.0),
                require_mac_above(0.7),
                require_maintenance_depth,
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
        description="Routine fentanyl, propofol, and rocuronium induction with sevoflurane maintenance.",
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
                "Give propofol <b>1.5-2.5 mg/kg</b>, titrated to effect "
                "(about 150 mg for 70 kg; use less in older patients or with hypotension), and start propofol "
                "at <b>100-200 mcg/kg/min</b>. Titrate to effect.<br><br>"
                "Optional lidocaine 20-40 mg IV before propofol can reduce injection pain."
            ),
            tci_instruction=(
                "Give propofol <b>1.5-2.5 mg/kg</b>, titrated to effect "
                "(about 150 mg for 70 kg; use less in older patients or with hypotension), and start propofol "
                "<b>TCI 4-6 µg/mL</b>.<br><br>"
                "Optional lidocaine 20-40 mg IV before propofol can reduce injection pain."
            ),
            check_requirements=require_all(
                require_drug_bolus("propofol", "Give the propofol induction bolus"),
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
                "Keep propofol and remifentanil running. Titrate infusion rates to "
                "BIS 40-60 and <b>MAP ≥ 65</b>. Reduce fresh gas to <b>2 L/min</b>, "
                "including at least 1 L/min O₂."
            ),
            tci_instruction=(
                "Keep propofol and remifentanil running. Reduce fresh gas to "
                "<b>2 L/min</b>, including at least 1 L/min O₂. Keep <b>MAP ≥ 65</b> and BIS 40-60.<br><br>"
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
        description="Routine propofol and rocuronium induction with propofol-remifentanil maintenance.",
        steps=steps,
    )
