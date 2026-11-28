"""Difficult laryngoscopy (ASA 2022 US OR pathway) and esophageal intubation scenarios."""

from anasim.core.action_log import ACTION_AIRWAY_TOOL, ACTION_DRUG_BOLUS
from anasim.core.state import AirwayType

from .base import (
    CaseEvent,
    CaseOutcome,
    Scenario,
    ScenarioStep,
    format_unmet_int,
    monitor_value,
    preroll_scenario,
    require_map_at_least,
    require_oxygen_flow,
    require_oxygenation,
)


def _setup_failed_laryngoscopy(engine) -> None:
    """Pre-roll induction so the patient starts apneic, paralyzed, preoxygenated, with a Grade III anterior airway."""
    engine.set_airway_mode("Mask")
    engine.set_fgf(10.0, 0.0, 0.0)
    engine.set_apl_valve(20.0)
    engine.set_mask_seal(0.60)
    wt = engine.patient.weight
    # Preoxygenate, then induce with fentanyl, propofol, and rocuronium + maintenance propofol.
    def _induce_step(eng, elapsed: float) -> None:
        if abs(elapsed - 30.0) < 1.1 and eng.actions.total(ACTION_DRUG_BOLUS, "roc") <= 0.0:
            eng.give_drug_bolus("fentanyl", round(1.5 * wt))
            if eng.config.tci_enabled:
                eng.set_drug_target("propofol", 3.1)
            else:
                eng.give_drug_bolus("propofol", round(2.0 * wt))
                eng.set_drug_rate("propofol", 75.0)
            eng.give_drug_bolus("roc", round(0.6 * wt))
            eng.set_bag_mask_ventilation(True, rr=12.0, vt=0.5, p_max=20.0)

    preroll_scenario(engine, 130.0, dt=1.0, settle_s=15.0, on_step=_induce_step)
    engine.set_difficult_airway("anterior_dl")


def _circulating(engine) -> bool:
    s = engine.state
    return (
        not s.cardiac_arrest
        and not s.chest_compressions
        and not engine.hemo.compressions
        and engine.spontaneous_circulation
    )


def _sugammadex_wakeup_met(engine) -> bool:
    """Valid ASA 2022 alternative: high-dose sugammadex reversal with spontaneous wake-up."""
    s = engine.state
    sug_given = sum(r.amount for r in engine.actions.since_case(ACTION_DRUG_BOLUS, labels=("sugammadex",)))
    return (
        sug_given >= 10.0 * engine.patient.weight
        and not engine.resp.state.apnea
        and s.tof >= 90.0
        and s.sao2 >= 92.0
    )


def _secured_with_capno(engine, min_etco2: float = 20.0) -> tuple[bool, str]:
    s = engine.state
    if s.airway_mode not in (AirwayType.ETT, AirwayType.SGA, AirwayType.FONA) or s.esophageal_intubation:
        return False, "Secure the trachea or SGA"
    if not (engine.vent.is_on or engine.bag_mask_active):
        return False, "Start ventilator or bag ventilation"
    if not s.etco2_signal_valid:
        return False, "Awaiting exhaled CO₂"
    etco2 = monitor_value(engine, "etco2")
    if etco2 < min_etco2:
        shown = format_unmet_int(etco2, low=min_etco2)
        return False, f"EtCO₂: {shown}/{min_etco2:.0f}+ mmHg"
    return True, ""


def _failed_dl_airway_secured(engine) -> bool:
    return _secured_with_capno(engine, 20.0)[0] or _sugammadex_wakeup_met(engine)


def _check_attempt_dl(engine) -> tuple[bool, str]:
    s = engine.state
    met = (
        engine.airway_crisis.laryngoscopy_attempts >= 1
        or (s.airway_mode in (AirwayType.ETT, AirwayType.SGA, AirwayType.FONA) and not s.esophageal_intubation)
    )
    return met, "" if met else "Attempt direct laryngoscopy (Tracheal tube)"


def _check_help_and_mask(engine) -> tuple[bool, str]:
    s = engine.state
    secured = s.airway_mode in (AirwayType.ETT, AirwayType.SGA, AirwayType.FONA) and not s.esophageal_intubation
    missing = []
    if not engine.airway_crisis.help_called:
        missing.append("Call for airway help")
    if not secured and not (engine.airway_crisis.oral_airway or engine.airway_crisis.two_hand_mask):
        missing.append("Enable OPA or two-hand jaw thrust")
    o2_ok, o2_msg = require_oxygen_flow(6.0)(engine)
    if not o2_ok:
        missing.append(o2_msg)
    return not missing, "; ".join(missing)


def _check_alternate_device(engine) -> tuple[bool, str]:
    s = engine.state
    met = (
        (s.airway_mode in (AirwayType.ETT, AirwayType.SGA, AirwayType.FONA) and not s.esophageal_intubation)
        or _sugammadex_wakeup_met(engine)
    )
    return met, "" if met else "Switch to VL + bougie / VL / fiberoptic and intubate, place an SGA, or reverse with sugammadex"


def _check_dl_vent_confirmed(engine) -> tuple[bool, str]:
    missing = []
    if not _sugammadex_wakeup_met(engine):
        airway_ok, airway_msg = _secured_with_capno(engine, 20.0)
        if not airway_ok:
            missing.append(airway_msg)
    spo2_ok, spo2_msg = require_oxygenation(92.0)(engine)
    if not spo2_ok:
        missing.append(spo2_msg)
    map_ok, map_msg = require_map_at_least(60.0)(engine)
    if not map_ok:
        missing.append(map_msg)
    return not missing, "; ".join(missing)


def create_failed_laryngoscopy() -> Scenario:
    return Scenario(
        id="failed_laryngoscopy",
        name="Failed laryngoscopy (difficult airway)",
        briefing=(
            "General anesthesia induction is complete (fentanyl, propofol, and rocuronium with "
            "propofol maintenance running). The patient is preoxygenated, apneic, paralyzed, and "
            "ready for tracheal intubation."
        ),
        setup_engine=_setup_failed_laryngoscopy,
        events=(
            CaseEvent(
                id="anterior_ready",
                at=1.0,
                cue="Induction and preoxygenation are complete. Attempt tracheal intubation.",
                apply=lambda _e: None,
            ),
        ),
        steps=[
            ScenarioStep(
                id="ATTEMPT_DL",
                title="Attempt direct laryngoscopy",
                instruction=(
                    "Attempt tracheal intubation using <b>Direct laryngoscopy (DL)</b> "
                    "(select <b>Tracheal tube</b> under Machine → Airway).<br><br>"
                    "<i>Assess the Cormack-Lehane view rather than forcing blind placement.</i>"
                ),
                check_requirements=_check_attempt_dl,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="CALL_HELP_OXYGENATE",
                title="Call for help and optimize mask oxygenation",
                instruction=(
                    "Direct laryngoscopy showed a <b>Grade III</b> anterior view. "
                    "Open <b>Airway tools</b>, select <b>Call for airway help</b>, enable "
                    "<b>Oral airway (OPA)</b> or <b>Two-hand jaw thrust</b>, and maintain 100% O₂ "
                    "(O₂ ≥ 6 L/min, air and N₂O 0 L/min).<br><br>"
                    "<i>Prioritize oxygenation and early help over repeated laryngoscopy attempts.</i>"
                ),
                check_requirements=_check_help_and_mask,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="ALTERNATE_AIRWAY",
                title="Escalate to an alternate airway device",
                instruction=(
                    "Change technique early (limit attempts to ≤ 3): choose <b>VL + bougie</b>, "
                    "<b>Video laryngoscopy (VL)</b>, or <b>Flexible bronchoscopy</b> in Airway tools "
                    "and select <b>Tracheal tube</b>, or place a rescue <b>SGA</b>.<br><br>"
                    "<i>If elective and mask ventilation is stable, stopping <b>propofol</b> and giving "
                    "<b>sugammadex 16 mg/kg</b> to wake the patient is also valid. Repeated attempts cause "
                    "airway edema and can precipitate CICO.</i>"
                ),
                check_requirements=_check_alternate_device,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="CONFIRM_VENTILATION",
                title="Confirm ventilation and oxygenation",
                instruction=(
                    "Start controlled ventilation (or maintain bag ventilation) and confirm "
                    "sustained capnography (<b>EtCO₂ ≥ 20 mmHg</b>), <b>SpO₂ ≥ 92%</b>, and <b>MAP ≥ 60 mmHg</b>."
                ),
                check_requirements=_check_dl_vent_confirmed,
                target_tab="Machine",
            ),
        ],
        outcomes=(
            CaseOutcome(
                id="help_called",
                label="Called for airway help and difficult airway cart",
                check=lambda e: e.airway_crisis.help_called,
                review="Call for help and the difficult airway cart immediately after an unexpected failed laryngoscopy.",
                milestone=True,
            ),
            CaseOutcome(
                id="mask_adjuncts",
                label="Optimized mask oxygenation with OPA or two-hand jaw thrust",
                check=lambda e: (
                    e.airway_crisis.oral_airway
                    or e.airway_crisis.two_hand_mask
                    or bool(e.actions.since_case(ACTION_AIRWAY_TOOL, labels=("opa_on", "two_hand_on")))
                ),
                review="Use an oral airway and two-handed jaw thrust to preserve oxygenation between attempts.",
                milestone=True,
            ),
            CaseOutcome(
                id="limit_attempts",
                label="Limited laryngoscopy attempts (≤ 3)",
                check=lambda e: 1 <= e.airway_crisis.laryngoscopy_attempts <= 3,
                review="Limit laryngoscopic attempts to ≤ 3 and change technique early to avoid hypoxemia and airway trauma.",
            ),
            CaseOutcome(
                id="airway_secured",
                label="Secured airway or reversed block with sustained ventilation",
                check=_failed_dl_airway_secured,
                review="Escalate to VL + bougie, flexible bronchoscopy, SGA rescue, eFONA, or high-dose sugammadex wake-up.",
            ),
            CaseOutcome(
                id="oxygenation",
                label="Maintained oxygenation and perfusion (SpO₂ ≥ 92%, MAP ≥ 60)",
                check=lambda e: _circulating(e) and e.state.sao2 >= 92.0 and e.state.map >= 60.0,
                review="Prioritize oxygenation and perfusion throughout difficult airway management.",
            ),
        ),
    )


def _setup_esophageal_intubation(engine) -> None:
    """Start moments after a blind esophageal intubation with ventilator running and decaying gastric CO2."""
    engine.set_airway_mode("Mask")
    engine.set_fgf(8.0, 0.0, 0.0)
    engine.set_apl_valve(20.0)
    wt = engine.patient.weight

    def _induce_step(eng, elapsed: float) -> None:
        if abs(elapsed - 30.0) < 1.1 and eng.actions.total(ACTION_DRUG_BOLUS, "roc") <= 0.0:
            eng.give_drug_bolus("fentanyl", round(1.5 * wt))
            if eng.config.tci_enabled:
                eng.set_drug_target("propofol", 3.7)
            else:
                eng.give_drug_bolus("propofol", round(2.4 * wt))
                eng.set_drug_rate("propofol", 100.0)
            eng.give_drug_bolus("roc", round(0.6 * wt))
            eng.set_bag_mask_ventilation(True, rr=12.0, vt=0.5, p_max=20.0)

    preroll_scenario(engine, 130.0, dt=1.0, settle_s=15.0, on_step=_induce_step)
    engine.set_bag_mask_ventilation(False)
    engine.set_difficult_airway("anterior_dl")
    engine.set_esophageal_intubation(True, gastric_co2_mmhg=18.0)
    engine.airway_crisis.last_view = "Grade III"
    engine.airway_crisis.laryngoscopy_attempts = 1
    engine.state.laryngoscopy_view = "Grade III"
    engine.state.laryngoscopy_attempts = 1
    engine.set_vent_settings(rr=12.0, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)


def _check_esoph_removed(engine) -> tuple[bool, str]:
    met = not engine.airway_crisis.esophageal_intubation
    return met, "" if met else "Remove the misplaced tube immediately (select Facemask, SGA, or re-intubate with VL)"


def _check_esoph_reox(engine) -> tuple[bool, str]:
    s = engine.state
    if engine.airway_crisis.esophageal_intubation:
        return False, "Remove the esophageal tube first"
    missing = []
    if s.airway_mode == AirwayType.NONE:
        missing.append("Connect Facemask or SGA")
    o2_ok, o2_msg = require_oxygen_flow(6.0)(engine)
    if not o2_ok:
        missing.append(o2_msg)
    if not (engine.bag_mask_active or engine.vent.is_on):
        missing.append("Start bag or mechanical ventilation")
    elif not s.etco2_signal_valid or monitor_value(engine, "etco2") < 15.0:
        missing.append("Deliver mask or SGA breaths to confirm exhaled CO₂")
    return not missing, "; ".join(missing)


def _check_esoph_reintubated(engine) -> tuple[bool, str]:
    s = engine.state
    met = (
        s.airway_mode in (AirwayType.ETT, AirwayType.SGA, AirwayType.FONA)
        and not s.esophageal_intubation
        and (engine.vent.is_on or engine.bag_mask_active)
    )
    return met, "" if met else "Place tracheal tube (VL / VL + bougie) or rescue SGA and start ventilation"


def _check_esoph_confirmed(engine) -> tuple[bool, str]:
    missing = []
    airway_ok, airway_msg = _secured_with_capno(engine, 22.0)
    if not airway_ok:
        missing.append(airway_msg)
    spo2_ok, spo2_msg = require_oxygenation(94.0)(engine)
    if not spo2_ok:
        missing.append(spo2_msg)
    map_ok, map_msg = require_map_at_least(60.0)(engine)
    if not map_ok:
        missing.append(map_msg)
    return not missing, "; ".join(missing)


def create_esophageal_intubation() -> Scenario:
    return Scenario(
        id="esophageal_intubation",
        name="Esophageal intubation",
        briefing=(
            "Induction and neuromuscular blockade are complete. A trainee has just placed an "
            "endotracheal tube after a Grade III view and connected the ventilator."
        ),
        setup_engine=_setup_esophageal_intubation,
        events=(
            CaseEvent(
                id="esoph_cue",
                at=15.0,
                when=lambda e: e.airway_crisis.esophageal_intubation,
                cue=(
                    "After several breaths, capnography has decayed toward flatline, returned tidal "
                    "volume (VTe) is very low, and breath sounds are absent over the chest."
                ),
                apply=lambda _e: None,
            ),
        ),
        steps=[
            ScenarioStep(
                id="REMOVE_TUBE",
                title="Remove the esophageal tube immediately",
                instruction=(
                    "Absent sustained exhaled CO₂ with low VTe indicates <b>esophageal intubation</b>. "
                    "Select <b>Facemask</b> (or <b>SGA</b>) under Machine → Airway to pull the tube immediately.<br><br>"
                    "<i>Always remove the tube when sustained exhaled CO₂ is absent. Do not wait for SpO₂ to fall; "
                    "after preoxygenation, pulse oximetry can stay normal for minutes.</i>"
                ),
                check_requirements=_check_esoph_removed,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="REOXYGENATE",
                title="Reoxygenate with 100% O₂",
                instruction=(
                    "Ventilate by facemask (or SGA) on <b>100% O₂</b> (O₂ ≥ 6 L/min, air and N₂O 0 L/min) "
                    "until exhaled CO₂ returns and oxygenation is restored. Call for airway help if needed."
                ),
                check_requirements=_check_esoph_reox,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="REINTUBATE",
                title="Re-intubate with an alternate technique",
                instruction=(
                    "Because initial direct laryngoscopy showed a Grade III view, switch to "
                    "<b>Video laryngoscopy (VL)</b> or <b>VL + bougie</b> in Airway tools and place the "
                    "<b>Tracheal tube</b> (or rescue <b>SGA</b>), then start mechanical ventilation."
                ),
                check_requirements=_check_esoph_reintubated,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="CONFIRM_CAPNO",
                title="Confirm sustained tracheal capnography",
                instruction=(
                    "Verify sustained tracheal capnography (<b>EtCO₂ ≥ 22 mmHg</b>), "
                    "<b>SpO₂ ≥ 94%</b>, and <b>MAP ≥ 60 mmHg</b>."
                ),
                check_requirements=_check_esoph_confirmed,
                target_tab="Machine",
            ),
        ],
        outcomes=(
            CaseOutcome(
                id="removed_esophageal_tube",
                label="Removed misplaced esophageal tube",
                check=lambda e: not e.airway_crisis.esophageal_intubation,
                review="Default to immediate tube removal whenever sustained exhaled CO₂ is absent.",
            ),
            CaseOutcome(
                id="tracheal_capnography",
                label="Confirmed sustained exhaled CO₂ through a valid airway",
                check=lambda e: _secured_with_capno(e, 22.0)[0],
                review="Confirm tracheal placement with sustained waveform capnography (≥ 7 breaths > 7.5 mmHg).",
            ),
            CaseOutcome(
                id="oxygenation_restored",
                label="Restored oxygenation and circulation (SpO₂ ≥ 94%, MAP ≥ 60)",
                check=lambda e: _circulating(e) and e.state.sao2 >= 94.0 and e.state.map >= 60.0,
                review="Reoxygenate with 100% O₂ by mask or SGA after tube removal before re-intubating.",
            ),
        ),
    )
