"""
Induction scenario definitions (Balanced and TIVA variants).
"""

from .base import (
    Scenario, ScenarioStep,
    require_airway, require_fgf_preox, require_propofol_cp,
    require_bis_below, require_ventilation_on, require_rocuronium_cp,
    require_tof_below, require_etco2_above, require_mac_above,
    require_remi_running, require_propofol_infusion_running,
    require_all, always_pass
)


def _step_num(n: int, total: int, title: str) -> str:
    """Format step header."""
    return f"<b>Step {n}/{total}: {title}</b><br>"


def create_induction_balanced() -> Scenario:
    """Create balanced anesthesia induction scenario."""
    
    steps = [
        ScenarioStep(
            id="APPLY_MASK",
            title="Apply facemask",
            instruction=(
                "<b>Apply facemask</b><br>"
                "Select <b>'Mask'</b> airway device to connect the patient to the breathing circuit.<br><br>"
                "<i>Ensure good mask seal on the patient's face for O₂ and anesthetic delivery.</i>"
            ),
            check_requirements=require_airway("Mask"),
        ),
        ScenarioStep(
            id="SET_FGF_PREOX",
            title="Set fresh gas flow",
            instruction=(
                "<b>Set fresh gas flow</b><br>"
                "Set <b>O₂ to 10 L/min</b> and <b>Air to 0 L/min</b> (100% O₂).<br><br>"
                "<i>High FGF rapidly washes nitrogen from the circuit.</i>"
            ),
            check_requirements=require_fgf_preox(),
        ),
        ScenarioStep(
            id="PREOXYGENATE",
            title="Preoxygenate",
            instruction=(
                "<b>Preoxygenate</b><br>"
                "Patient breathes spontaneously via mask. Confirm <b>FiO₂ ~100%</b> on monitor.<br>"
                "In practice: wait 3-5 min or 8 vital capacity breaths. Click 'Next step' when ready.<br><br>"
                "<i>Replaces N₂ with O₂, extending safe apnea time to 8-10 minutes.</i>"
            ),
            check_requirements=require_fgf_preox(),  # Verify FGF remains adequate.
        ),
        ScenarioStep(
            id="INDUCE",
            title="Induction",
            instruction=(
                "<b>Induction</b><br>"
                "Administer propofol bolus: <b>1.5-2.5 mg/kg</b> (~105-175 mg for 70kg).<br><br>"
                "<i>LOC occurs at plasma concentration ~3-4 µg/mL. Inject over 20-30 seconds.</i>"
            ),
            check_requirements=require_propofol_cp(2.0),
        ),
        ScenarioStep(
            id="CONFIRM_LOC",
            title="Confirm loss of consciousness",
            instruction=(
                "<b>Confirm loss of consciousness</b><br>"
                "Verify: <b>BIS < 60</b>, patient is apneic, no response to stimuli.<br><br>"
                "<i>CRITICAL: Confirm adequate depth BEFORE giving neuromuscular blocker.</i>"
            ),
            check_requirements=require_bis_below(60),
        ),
        ScenarioStep(
            id="MASK_VENTILATE",
            title="Bag-mask ventilation",
            instruction=(
                "<b>Bag-mask ventilation</b><br>"
                "Click <b>'Bag-mask: OFF'</b> button to turn ON manual ventilation.<br>"
                "Confirm chest rise and <b>SpO₂ maintained</b>.<br><br>"
                "<i>Patient is apneic post-induction. Must ventilate to prevent hypoxia.</i>"
            ),
            check_requirements=require_ventilation_on(),
        ),
        ScenarioStep(
            id="GIVE_NMB",
            title="Administer neuromuscular blocker",
            instruction=(
                "<b>Administer neuromuscular blocker</b><br>"
                "Give rocuronium: <b>0.6 mg/kg</b> (~42 mg for 70kg). Onset 60-90 sec.<br>"
                "For RSI: <b>1.2 mg/kg</b>.<br><br>"
                "<i>Muscle relaxation provides optimal intubating conditions.</i>"
            ),
            check_requirements=require_rocuronium_cp(0.5),
        ),
        ScenarioStep(
            id="WAIT_PARALYSIS",
            title="Confirm adequate paralysis",
            instruction=(
                "<b>Confirm adequate paralysis</b><br>"
                "Monitor <b>Train of Four (TOF)</b>. Wait for <b>TOF 0-1/4</b> before laryngoscopy.<br><br>"
                "<i>Incomplete paralysis risks vocal cord trauma and poor visualization.</i>"
            ),
            check_requirements=require_tof_below(25),
        ),
        ScenarioStep(
            id="INTUBATE",
            title="Secure airway",
            instruction=(
                "<b>Secure airway</b><br>"
                "Perform laryngoscopy and insert ETT. Select <b>'ETT'</b> airway device.<br><br>"
                "<i>Advance ETT through cords, inflate cuff, connect to circuit.</i>"
            ),
            check_requirements=require_airway("ETT"),
        ),
        ScenarioStep(
            id="CONFIRM_ETT",
            title="Confirm ETT placement",
            instruction=(
                "<b>Confirm ETT placement</b><br>"
                "Verify: <b>EtCO₂ waveform present</b> (gold standard), bilateral breath sounds.<br><br>"
                "<i>No EtCO₂ = tube not in trachea until proven otherwise.</i>"
            ),
            check_requirements=require_etco2_above(20),
        ),
        ScenarioStep(
            id="MAINTENANCE",
            title="Begin maintenance",
            instruction=(
                "<b>Begin maintenance</b><br>"
                "Turn on sevoflurane: <b>1.5-2%</b> (1 MAC ≈ 2.1%).<br>"
                "Reduce FGF to <b>2 L/min</b>. Target: <b>BIS 40-60</b>, <b>MAP > 65</b>.<br><br>"
                "<i>Lower flows reduce cost and environmental pollution.</i>"
            ),
            check_requirements=require_mac_above(0.5),
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
                "<b>Apply facemask</b><br>"
                "Select <b>'Mask'</b> airway device to connect the patient to the breathing circuit.<br><br>"
                "<i>Ensure good mask seal on the patient's face for O₂ and anesthetic delivery.</i>"
            ),
            check_requirements=require_airway("Mask"),
        ),
        ScenarioStep(
            id="SET_FGF_PREOX",
            title="Set fresh gas flow",
            instruction=(
                "<b>Set fresh gas flow</b><br>"
                "Set <b>O₂ to 10 L/min</b> and <b>Air to 0 L/min</b> (100% O₂).<br><br>"
                "<i>High FGF rapidly washes nitrogen from the circuit.</i>"
            ),
            check_requirements=require_fgf_preox(),
        ),
        ScenarioStep(
            id="PREOXYGENATE",
            title="Preoxygenate",
            instruction=(
                "<b>Preoxygenate</b><br>"
                "Patient breathes spontaneously via mask. Confirm <b>FiO₂ ~100%</b> on monitor.<br>"
                "In practice: wait 3-5 min or 8 vital capacity breaths. Click 'Next step' when ready.<br><br>"
                "<i>Replaces N₂ with O₂, extending safe apnea time to 8-10 minutes.</i>"
            ),
            check_requirements=require_fgf_preox(),
        ),
        ScenarioStep(
            id="START_ANALGESIA",
            title="Start analgesia (TIVA)",
            instruction=(
                "<b>Start analgesia (TIVA)</b><br>"
                "Start remifentanil: <b>TCI 2-4 ng/mL</b> or infusion <b>0.1-0.25 mcg/kg/min</b>.<br><br>"
                "<i>Opioid blunts sympathetic response to laryngoscopy.</i>"
            ),
            check_requirements=require_remi_running(),
        ),
        ScenarioStep(
            id="INDUCE",
            title="Induction (TIVA)",
            instruction=(
                "<b>Induction (TIVA)</b><br>"
                "Administer propofol bolus: <b>1.5-2.5 mg/kg</b> (~105-175 mg for 70kg).<br>"
                "Also start propofol infusion: <b>TCI 4-6 µg/mL</b>.<br><br>"
                "<i>Continuous infusion maintains anesthesia after bolus redistributes.</i>"
            ),
            check_requirements=require_all(
                require_propofol_cp(2.0),
                require_propofol_infusion_running()
            ),
        ),
        ScenarioStep(
            id="CONFIRM_LOC",
            title="Confirm loss of consciousness",
            instruction=(
                "<b>Confirm loss of consciousness</b><br>"
                "Verify: <b>BIS < 60</b>, patient is apneic, no response to stimuli.<br><br>"
                "<i>CRITICAL: Confirm adequate depth BEFORE giving neuromuscular blocker.</i>"
            ),
            check_requirements=require_bis_below(60),
        ),
        ScenarioStep(
            id="MASK_VENTILATE",
            title="Bag-mask ventilation",
            instruction=(
                "<b>Bag-mask ventilation</b><br>"
                "Click <b>'Bag-mask: OFF'</b> button to turn ON manual ventilation.<br>"
                "Confirm chest rise and <b>SpO₂ maintained</b>.<br><br>"
                "<i>Patient is apneic post-induction. Must ventilate to prevent hypoxia.</i>"
            ),
            check_requirements=require_ventilation_on(),
        ),
        ScenarioStep(
            id="GIVE_NMB",
            title="Administer neuromuscular blocker",
            instruction=(
                "<b>Administer neuromuscular blocker</b><br>"
                "Give rocuronium: <b>0.6 mg/kg</b> (~42 mg for 70kg). Onset 60-90 sec.<br>"
                "For RSI: <b>1.2 mg/kg</b>.<br><br>"
                "<i>Muscle relaxation provides optimal intubating conditions.</i>"
            ),
            check_requirements=require_rocuronium_cp(0.5),
        ),
        ScenarioStep(
            id="WAIT_PARALYSIS",
            title="Confirm adequate paralysis",
            instruction=(
                "<b>Confirm adequate paralysis</b><br>"
                "Monitor <b>Train of Four (TOF)</b>. Wait for <b>TOF 0-1/4</b> before laryngoscopy.<br><br>"
                "<i>Incomplete paralysis risks vocal cord trauma and poor visualization.</i>"
            ),
            check_requirements=require_tof_below(25),
        ),
        ScenarioStep(
            id="INTUBATE",
            title="Secure airway",
            instruction=(
                "<b>Secure airway</b><br>"
                "Perform laryngoscopy and insert ETT. Select <b>'ETT'</b> airway device.<br><br>"
                "<i>Advance ETT through cords, inflate cuff, connect to circuit.</i>"
            ),
            check_requirements=require_airway("ETT"),
        ),
        ScenarioStep(
            id="CONFIRM_ETT",
            title="Confirm ETT placement",
            instruction=(
                "<b>Confirm ETT placement</b><br>"
                "Verify: <b>EtCO₂ waveform present</b> (gold standard), bilateral breath sounds.<br><br>"
                "<i>No EtCO₂ = tube not in trachea until proven otherwise.</i>"
            ),
            check_requirements=require_etco2_above(20),
        ),
        ScenarioStep(
            id="MAINTENANCE",
            title="Confirm maintenance (TIVA)",
            instruction=(
                "<b>Confirm maintenance (TIVA)</b><br>"
                "Verify propofol and remifentanil infusions running.<br>"
                "Reduce FGF to <b>2 L/min O₂</b>. Target: <b>BIS 40-60</b>, <b>MAP > 65</b>.<br><br>"
                "<i>Typical: Propofol TCI 3-4 µg/mL, Remi TCI 2-4 ng/mL.</i>"
            ),
            check_requirements=require_all(
                require_propofol_infusion_running(),
                require_remi_running()
            ),
        ),
    ]
    
    return Scenario(
        id="induction_tiva",
        name="Induction (TIVA)",
        icon="",
        description="Learn the TIVA induction sequence with propofol/remifentanil maintenance.",
        steps=steps,
    )
