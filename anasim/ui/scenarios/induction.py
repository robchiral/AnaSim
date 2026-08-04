"""
Induction scenario definitions (Balanced and TIVA variants).
"""

from .base import (
    Scenario, ScenarioStep,
    require_airway_selected, require_fgf_set_for_preox,
    require_preoxygenation_flow, require_propofol_cp,
    require_bis_below, require_bag_mask_started, require_rocuronium_cp,
    require_tof_below, require_etco2_above, require_mac_above,
    require_drug_bolus, require_infusion_running, require_infusion_started,
    require_fgf_reduced, require_vaporizer_started,
    require_all
)

def create_induction_balanced() -> Scenario:
    """Create balanced anesthesia induction scenario."""
    
    steps = [
        ScenarioStep(
            id="APPLY_MASK",
            title="Apply facemask",
            instruction=(
                "Select <b>Facemask</b> to connect the patient to the breathing circuit.<br><br>"
                "<i>Ensure good mask seal on the patient's face for O₂ and anesthetic delivery.</i>"
            ),
            check_requirements=require_airway_selected("Mask"),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="SET_FGF_PREOX",
            title="Set fresh gas flow",
            instruction=(
                "Set <b>O₂ to 10 L/min</b> and <b>Air to 0 L/min</b> (100% O₂).<br><br>"
                "<i>High FGF rapidly washes nitrogen from the circuit.</i>"
            ),
            check_requirements=require_fgf_set_for_preox(),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="PREOXYGENATE",
            title="Preoxygenate",
            instruction=(
                "Patient breathes spontaneously via mask. Confirm <b>FiO₂ ~100%</b> on monitor.<br>"
                "In practice: wait 3-5 min or 8 vital capacity breaths.<br><br>"
                "<i>Replaces N₂ with O₂, extending safe apnea time to 8-10 minutes.</i>"
            ),
            check_requirements=require_preoxygenation_flow(),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="INDUCE",
            title="Induction",
            instruction=(
                "Administer propofol bolus: <b>1.5-2.5 mg/kg</b> (~105-175 mg for 70kg).<br><br>"
                "<i>LOC occurs at plasma concentration ~3-4 µg/mL. Inject over 20-30 seconds.</i>"
            ),
            check_requirements=require_all(
                require_drug_bolus("propofol", "Give the propofol induction bolus"),
                require_propofol_cp(2.0),
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="CONFIRM_LOC",
            title="Confirm loss of consciousness",
            instruction=(
                "Verify: <b>BIS < 60</b>, patient is apneic, no response to stimuli.<br><br>"
                "<i>Confirm adequate depth before giving a neuromuscular blocker.</i>"
            ),
            check_requirements=require_bis_below(60),
        ),
        ScenarioStep(
            id="MASK_VENTILATE",
            title="Bag-mask ventilation",
            instruction=(
                "Click <b>'Start bag-mask ventilation'</b>.<br>"
                "Confirm chest rise and <b>SpO₂ maintained</b>.<br><br>"
                "<i>Patient is apneic post-induction. Must ventilate to prevent hypoxia.</i>"
            ),
            check_requirements=require_bag_mask_started(),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="GIVE_NMB",
            title="Administer neuromuscular blocker",
            instruction=(
                "Give rocuronium: <b>0.6 mg/kg</b> (~42 mg for 70kg). Onset 60-90 sec.<br>"
                "For RSI: <b>1.2 mg/kg</b>.<br><br>"
                "<i>Muscle relaxation provides optimal intubating conditions.</i>"
            ),
            check_requirements=require_all(
                require_drug_bolus("roc", "Give the rocuronium bolus"),
                require_rocuronium_cp(0.5),
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="WAIT_PARALYSIS",
            title="Confirm adequate paralysis",
            instruction=(
                "Monitor <b>Train of Four (TOF)</b>. Wait for <b>TOF 0-1/4</b> before laryngoscopy.<br><br>"
                "<i>Incomplete paralysis risks vocal cord trauma and poor visualization.</i>"
            ),
            check_requirements=require_tof_below(25),
        ),
        ScenarioStep(
            id="INTUBATE",
            title="Secure airway",
            instruction=(
                "Perform laryngoscopy and insert ETT. Select <b>'ETT'</b> airway device.<br><br>"
                "<i>Advance ETT through cords, inflate cuff, connect to circuit.</i>"
            ),
            check_requirements=require_airway_selected("ETT"),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="CONFIRM_ETT",
            title="Confirm ETT placement",
            instruction=(
                "Verify: <b>EtCO₂ waveform present</b> (gold standard), bilateral breath sounds.<br><br>"
                "<i>No EtCO₂ = tube not in trachea until proven otherwise.</i>"
            ),
            check_requirements=require_etco2_above(20),
        ),
        ScenarioStep(
            id="MAINTENANCE",
            title="Begin maintenance",
            instruction=(
                "Turn on sevoflurane: <b>1.5-2%</b> (1 MAC ≈ 2.1%).<br>"
                "Reduce FGF to <b>2 L/min</b>. Target: <b>BIS 40-60</b>, <b>MAP > 65</b>.<br><br>"
                "<i>Lower flows reduce cost and environmental pollution.</i>"
            ),
            check_requirements=require_all(
                require_vaporizer_started(),
                require_fgf_reduced(2.0),
                require_mac_above(0.5),
            ),
            target_tab="Machine",
        ),
    ]
    
    return Scenario(
        id="induction_balanced",
        name="Induction (Balanced)",
        icon="",
        description="Learn the balanced anesthesia induction sequence with volatile maintenance.",
        steps=steps,
    )


def create_induction_tiva() -> Scenario:
    """Create TIVA induction scenario."""
    
    steps = [
        ScenarioStep(
            id="APPLY_MASK",
            title="Apply facemask",
            instruction=(
                "Select <b>Facemask</b> to connect the patient to the breathing circuit.<br><br>"
                "<i>Ensure good mask seal on the patient's face for O₂ and anesthetic delivery.</i>"
            ),
            check_requirements=require_airway_selected("Mask"),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="SET_FGF_PREOX",
            title="Set fresh gas flow",
            instruction=(
                "Set <b>O₂ to 10 L/min</b> and <b>Air to 0 L/min</b> (100% O₂).<br><br>"
                "<i>High FGF rapidly washes nitrogen from the circuit.</i>"
            ),
            check_requirements=require_fgf_set_for_preox(),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="PREOXYGENATE",
            title="Preoxygenate",
            instruction=(
                "Patient breathes spontaneously via mask. Confirm <b>FiO₂ ~100%</b> on monitor.<br>"
                "In practice: wait 3-5 min or 8 vital capacity breaths.<br><br>"
                "<i>Replaces N₂ with O₂, extending safe apnea time to 8-10 minutes.</i>"
            ),
            check_requirements=require_preoxygenation_flow(),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="START_ANALGESIA",
            title="Start analgesia (TIVA)",
            instruction=(
                "Start remifentanil: <b>TCI 2-4 ng/mL</b> or infusion <b>0.1-0.25 mcg/kg/min</b>.<br><br>"
                "<i>Opioid blunts sympathetic response to laryngoscopy.</i>"
            ),
            check_requirements=require_infusion_started(
                "remi", fail_message="Start Remifentanil TCI or infusion"
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="INDUCE",
            title="Induction (TIVA)",
            instruction=(
                "Administer propofol bolus: <b>1.5-2.5 mg/kg</b> (~105-175 mg for 70kg).<br>"
                "Also start propofol infusion: <b>TCI 4-6 µg/mL</b>.<br><br>"
                "<i>Continuous infusion maintains anesthesia after bolus redistributes.</i>"
            ),
            check_requirements=require_all(
                require_drug_bolus("propofol", "Give the propofol induction bolus"),
                require_propofol_cp(2.0),
                require_infusion_started(
                    "propofol", fail_message="Start the propofol infusion"
                ),
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="CONFIRM_LOC",
            title="Confirm loss of consciousness",
            instruction=(
                "Verify: <b>BIS < 60</b>, patient is apneic, no response to stimuli.<br><br>"
                "<i>Confirm adequate depth before giving a neuromuscular blocker.</i>"
            ),
            check_requirements=require_bis_below(60),
        ),
        ScenarioStep(
            id="MASK_VENTILATE",
            title="Bag-mask ventilation",
            instruction=(
                "Click <b>'Start bag-mask ventilation'</b>.<br>"
                "Confirm chest rise and <b>SpO₂ maintained</b>.<br><br>"
                "<i>Patient is apneic post-induction. Must ventilate to prevent hypoxia.</i>"
            ),
            check_requirements=require_bag_mask_started(),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="GIVE_NMB",
            title="Administer neuromuscular blocker",
            instruction=(
                "Give rocuronium: <b>0.6 mg/kg</b> (~42 mg for 70kg). Onset 60-90 sec.<br>"
                "For RSI: <b>1.2 mg/kg</b>.<br><br>"
                "<i>Muscle relaxation provides optimal intubating conditions.</i>"
            ),
            check_requirements=require_all(
                require_drug_bolus("roc", "Give the rocuronium bolus"),
                require_rocuronium_cp(0.5),
            ),
            target_tab="Medications",
        ),
        ScenarioStep(
            id="WAIT_PARALYSIS",
            title="Confirm adequate paralysis",
            instruction=(
                "Monitor <b>Train of Four (TOF)</b>. Wait for <b>TOF 0-1/4</b> before laryngoscopy.<br><br>"
                "<i>Incomplete paralysis risks vocal cord trauma and poor visualization.</i>"
            ),
            check_requirements=require_tof_below(25),
        ),
        ScenarioStep(
            id="INTUBATE",
            title="Secure airway",
            instruction=(
                "Perform laryngoscopy and insert ETT. Select <b>'ETT'</b> airway device.<br><br>"
                "<i>Advance ETT through cords, inflate cuff, connect to circuit.</i>"
            ),
            check_requirements=require_airway_selected("ETT"),
            target_tab="Machine",
        ),
        ScenarioStep(
            id="CONFIRM_ETT",
            title="Confirm ETT placement",
            instruction=(
                "Verify: <b>EtCO₂ waveform present</b> (gold standard), bilateral breath sounds.<br><br>"
                "<i>No EtCO₂ = tube not in trachea until proven otherwise.</i>"
            ),
            check_requirements=require_etco2_above(20),
        ),
        ScenarioStep(
            id="MAINTENANCE",
            title="Confirm maintenance (TIVA)",
            instruction=(
                "Verify propofol and remifentanil infusions running.<br>"
                "Reduce FGF to <b>2 L/min O₂</b>. Target: <b>BIS 40-60</b>, <b>MAP > 65</b>.<br><br>"
                "<i>Typical: Propofol TCI 3-4 µg/mL, Remi TCI 2-4 ng/mL.</i>"
            ),
            check_requirements=require_all(
                require_infusion_running("propofol", "Propofol infusion not running"),
                require_infusion_running("remi", "Remifentanil infusion not running"),
            ),
            target_tab="Medications",
        ),
    ]
    
    return Scenario(
        id="induction_tiva",
        name="Induction (TIVA)",
        icon="",
        description="Learn the TIVA induction sequence with propofol/remifentanil maintenance.",
        steps=steps,
    )
