"""Tension pneumothorax, venous air embolism (VAE), and massive PE scenarios."""

from anasim.core.action_log import (
    ACTION_CHEST_DECOMPRESS,
    ACTION_DRUG_BOLUS,
    ACTION_EMBOLISM_TREAT,
    ACTION_EVENT_START,
    ACTION_INFUSION_RATE,
    ACTION_TCI_TARGET,
)
from anasim.core.state import AirwayType

from .base import (
    CaseAction,
    CaseEvent,
    CaseOutcome,
    Scenario,
    ScenarioStep,
    action_taken_in_case,
    create_observe_step,
    equilibrate_maintenance_fio2,
    format_unmet_int,
    monitor_value,
    patient_stable,
    require_map_at_least,
    require_oxygen_flow,
    require_oxygenation,
)


def _setup_n2o_balanced(engine) -> None:
    """Seed 50% N2O / O2 balanced maintenance so N2O cavity/bubble expansion is active immediately."""
    from anasim.core import projection as projection_core

    engine.set_fgf(1.5, 0.0, 1.5)
    engine.set_vaporizer(1.6)
    comp = engine.circuit.composition
    fi_agent = max(0.0, comp.fi_agent)
    comp.fin2o = 0.50
    comp.fio2 = max(0.21, 0.50 - fi_agent)
    comp.fin2 = max(0.0, 1.0 - comp.fio2 - comp.fin2o - fi_agent)
    p_n2o = 0.46
    n2o_st = engine.pk_n2o.state
    n2o_st.p_alv = n2o_st.p_art = n2o_st.p_ven = n2o_st.p_vrg = n2o_st.p_brain = n2o_st.p_mus = n2o_st.p_fat = p_n2o
    n2o_st.mac = p_n2o * 100.0 / engine.pk_n2o.mac_at_temp(engine.state.temp_c)
    engine.resp.equilibrate_oxygen(
        comp.fio2,
        1.0,
        engine.vent.settings.peep,
        fin2=comp.fin2,
    )
    engine._tracheal_gas = projection_core.inspired_gas_mixture(engine)
    projection_core.sync_inspired_gas(engine)
    projection_core.sync_inhaled_agents(engine)
    projection_core.sync_state_from_models(engine)
    if engine.etco2_readout.gas_valid:
        engine.etco2_readout.et_gases = (
            engine.state.et_o2,
            engine.state.et_sevo,
            engine.state.et_n2o,
            engine.state.et_mac,
        )


def _setup_pe_tiva(engine) -> None:
    """Intraoperative 45-50% O2/air maintenance so PE hypoxemia is visibly evident on SpO2."""
    equilibrate_maintenance_fio2(engine, target_fio2=0.45, min_air_l_min=1.2, air_vo2_factor=4.5)


def _ventilated_airway(engine) -> tuple[bool, str]:
    s = engine.state
    if s.airway_mode not in (AirwayType.ETT, AirwayType.SGA, AirwayType.FONA) or s.esophageal_intubation:
        return False, "Maintain a patent airway (ETT, SGA, or eFONA)"
    if not (engine.vent.is_on or engine.bag_mask_active):
        return False, "Maintain ventilation"
    return True, ""


def _circulating(engine, *, allow_va_ecmo: bool = False) -> bool:
    s = engine.state
    if allow_va_ecmo and engine.extracorporeal.is_va and engine.extracorporeal.state.blood_flow_l_min >= 2.0:
        return True
    return (
        not s.cardiac_arrest
        and not s.chest_compressions
        and not engine.hemo.compressions
        and engine.spontaneous_circulation
    )


def _require_monitored_map_ecmo(engine, threshold: float = 65.0) -> tuple[bool, str]:
    if not (engine.extracorporeal.is_va and engine.extracorporeal.state.blood_flow_l_min >= 2.0):
        return require_map_at_least(threshold)(engine)
    s = engine.state
    nonpulsatile_va = engine.hemo.state.sv <= 0.0 or (s.sbp - s.dbp) < 8.0
    if not engine.config.arterial_line_enabled:
        if nonpulsatile_va and s.nibp_measurement_failed:
            # Oscillometric NIBP cannot detect nonpulsatile VA ECMO flow; fall back to arterial perfusion MAP.
            if s.map < threshold:
                return False, f"MAP: {format_unmet_int(s.map, low=threshold)}/{threshold:.0f}+"
            return True, ""
        if s.nibp_measurement_failed or s.nibp_timestamp is None or s.time - s.nibp_timestamp > 60.0:
            return False, "Measure blood pressure again"
    map_val = monitor_value(engine, "map")
    if map_val < threshold:
        return False, f"MAP: {format_unmet_int(map_val, low=threshold)}/{threshold:.0f}+"
    if s.map < threshold:
        return False, "Reassess blood pressure and perfusion"
    return True, ""


def _ptx_decompressed(engine) -> bool:
    return (
        action_taken_in_case(engine, ACTION_EVENT_START, "pneumothorax")
        and engine.pneumothorax.decompressed
        and bool(engine.actions.since_event("pneumothorax", ACTION_CHEST_DECOMPRESS))
    )


def _pressor_used(engine, event_name: str) -> bool:
    boluses = engine.actions.since_event(
        event_name, ACTION_DRUG_BOLUS, labels=("nore", "epi", "phenyl", "ephedrine", "vaso"),
    )
    infusions = engine.actions.since_event(
        event_name, ACTION_INFUSION_RATE, labels=("nore", "epi", "phenyl", "vaso"),
    )
    tci_targets = engine.actions.since_event(
        event_name, ACTION_TCI_TARGET, labels=("nore", "epi", "phenyl", "vaso"),
    )
    return any(r.amount > 0 for r in (*boluses, *infusions, *tci_targets))


def create_tension_pneumothorax() -> Scenario:
    def start_ptx(engine) -> None:
        engine.start_pneumothorax(severity=0.95, initial_volume_l=0.45)

    def check_decompressed(engine) -> tuple[bool, str]:
        met = _ptx_decompressed(engine)
        return met, "" if met else "Perform Chest decompression (PTX) under Circulation → Crisis procedures"

    def check_ptx_stable(engine) -> tuple[bool, str]:
        if not _ptx_decompressed(engine):
            return False, "Decompress pneumothorax first"
        missing = []
        for check in (_ventilated_airway, require_map_at_least(65.0), require_oxygenation(94.0)):
            ok, msg = check(engine)
            if not ok:
                missing.append(msg)
        if not missing and not patient_stable(engine):
            missing.append("Reassess oxygenation and perfusion")
        return not missing, "; ".join(missing)

    return Scenario(
        id="tension_pneumothorax",
        name="Tension pneumothorax",
        briefing=(
            "General anesthesia is established on VCV (PEEP 5 cmH₂O) with sevoflurane and 50% N₂O/O₂ "
            "during upper abdominal laparoscopy. Monitor airway pressures, oxygenation, and hemodynamics."
        ),
        setup_engine=_setup_n2o_balanced,
        events=(
            CaseEvent(
                id="ptx_onset",
                at=12.0,
                cue=(
                    "Peak and plateau airway pressures are climbing sharply with a narrow Ppeak–Pplat gap. "
                    "Right-sided breath sounds are absent and blood pressure is falling."
                ),
                apply=start_ptx,
            ),
        ),
        steps=[
            create_observe_step("pneumothorax"),
            ScenarioStep(
                id="STOP_N2O_100_O2",
                title="Stop N₂O and switch to 100% O₂",
                instruction=(
                    "Rising peak and plateau pressures with hypotension during N₂O anesthesia indicate "
                    "<b>tension pneumothorax</b>. Turn <b>N₂O and air to 0 L/min</b> and set "
                    "<b>O₂ ≥ 6 L/min</b> immediately.<br><br>"
                    "<i>N₂O diffuses rapidly into the pleural space and doubles its volume in minutes (Eger and Saidman 1965).</i>"
                ),
                check_requirements=require_oxygen_flow(6.0),
                target_tab="Machine",
            ),
            ScenarioStep(
                id="DECOMPRESS_PTX",
                title="Decompress the chest immediately",
                instruction=(
                    "Open <b>Circulation → Crisis procedures</b> and perform "
                    "<b>Chest decompression (PTX)</b> (needle/tube thoracostomy).<br><br>"
                    "<i>Pleural decompression reverses mediastinal shift and obstructive shock rapidly (Nelson 2013).</i>"
                ),
                check_requirements=check_decompressed,
                target_tab="Circulation",
            ),
            ScenarioStep(
                id="STABILIZE_PTX",
                title="Confirm hemodynamic and airway recovery",
                instruction=(
                    "Confirm falling airway pressures and recovery of <b>MAP ≥ 65 mmHg</b> and "
                    "<b>SpO₂ ≥ 94%</b>. Support blood pressure with fluids or a vasopressor if needed."
                ),
                check_requirements=check_ptx_stable,
            ),
        ],
        outcomes=(
            CaseOutcome(
                id="stopped_n2o",
                label="Discontinued N₂O and delivered 100% O₂",
                check=lambda e: require_oxygen_flow(6.0)(e)[0],
                review="Stop N₂O immediately when pneumothorax is suspected because N₂O expands closed gas spaces rapidly (Eger and Saidman 1965).",
                milestone=True,
            ),
            CaseOutcome(
                id="decompressed",
                label="Decompressed tension pneumothorax",
                check=_ptx_decompressed,
                review="Perform immediate needle or tube thoracostomy decompression to relieve vena caval compression (Nelson 2013).",
                milestone=True,
            ),
            CaseOutcome(
                id="stabilized",
                label="Restored hemodynamics and oxygenation (MAP ≥ 65, SpO₂ ≥ 94%)",
                check=lambda e: _ptx_decompressed(e) and _ventilated_airway(e)[0] and patient_stable(e),
                review="Verify recovery of MAP, SpO₂, and airway pressures after decompression.",
            ),
        ),
    )


def _vae_aspirated(engine) -> bool:
    return bool(engine.actions.since_event("venous_air_embolism", ACTION_EMBOLISM_TREAT, labels=("vae_aspirate",)))


def _vae_stopped(engine) -> bool:
    return action_taken_in_case(engine, ACTION_EVENT_START, "venous_air_embolism") and not engine.embolism.vae_active


def create_venous_air_embolism() -> Scenario:
    def start_vae(engine) -> None:
        engine.start_venous_air_embolism(rate_ml_min=65.0, initial_bolus_ml=55.0)

    def check_source_stopped(engine) -> tuple[bool, str]:
        met = _vae_stopped(engine)
        return met, "" if met else "Use case action 'Alert surgeon: flood field & lower head'"

    def check_aspirate_position(engine) -> tuple[bool, str]:
        missing = []
        if not _vae_aspirated(engine):
            missing.append("Aspirate CVC air under Circulation → Crisis procedures")
        if engine.hemo.position != "left_lateral":
            missing.append("Position Left lateral (Durant) under Machine → Airway")
        return not missing, "; ".join(missing)

    def check_vae_stable(engine) -> tuple[bool, str]:
        if not _vae_stopped(engine):
            return False, "Stop surgical air entrainment first"
        missing = []
        for check in (_ventilated_airway, require_map_at_least(65.0), require_oxygenation(92.0)):
            ok, msg = check(engine)
            if not ok:
                missing.append(msg)
        if not missing and not (_circulating(engine) and engine.state.sao2 >= 92.0 and engine.state.map >= 65.0):
            missing.append("Reassess oxygenation and perfusion")
        return not missing, "; ".join(missing)

    return Scenario(
        id="venous_air_embolism",
        name="Venous air embolism (VAE)",
        briefing=(
            "Posterior fossa craniotomy in the sitting position under sevoflurane with 50% N₂O/O₂ "
            "and a right atrial multiorifice central venous catheter in place."
        ),
        setup_engine=_setup_n2o_balanced,
        events=(
            CaseEvent(
                id="vae_onset",
                at=12.0,
                cue=(
                    "Precordial Doppler develops a loud 'mill-wheel' roar. EtCO₂ plunges abruptly "
                    "while blood pressure falls."
                ),
                apply=start_vae,
            ),
        ),
        actions=(
            CaseAction(
                id="flood_field_lower_head",
                label="Alert surgeon: flood field & lower head",
                completed_label="Surgeon flooded field & lowered head",
                available=lambda e: e.embolism.vae_active,
                apply=lambda e: e.stop_venous_air_embolism(),
            ),
        ),
        steps=[
            create_observe_step("venous_air_embolism"),
            ScenarioStep(
                id="STOP_N2O_VAE",
                title="Stop N₂O and switch to 100% O₂",
                instruction=(
                    "Sudden EtCO₂ drop with hypotension in the sitting position indicates "
                    "<b>venous air embolism (VAE)</b>. Turn <b>N₂O and air off (0 L/min)</b> and set "
                    "<b>O₂ ≥ 8 L/min</b> immediately.<br><br>"
                    "<i>50% N₂O roughly doubles intravascular air volume (and higher fractions expand it further), worsening RV air-lock (Munson and Merrick 1966; Mirski 2007).</i>"
                ),
                check_requirements=require_oxygen_flow(8.0),
                target_tab="Machine",
            ),
            ScenarioStep(
                id="STOP_ENTRAINMENT",
                title="Stop surgical air entrainment",
                instruction=(
                    "Select <b>Alert surgeon: flood field & lower head</b> so the surgeon floods "
                    "the surgical field, waxes open veins, and lowers the surgical site below heart level."
                ),
                check_requirements=check_source_stopped,
            ),
            ScenarioStep(
                id="ASPIRATE_POSITION",
                title="Aspirate CVC air and position left lateral",
                instruction=(
                    "Open <b>Circulation → Crisis procedures</b> and click <b>CVC air aspiration (VAE)</b>, "
                    "and set <b>Patient position</b> to <b>Left lateral</b> (Durant position) under Machine → Airway."
                ),
                check_requirements=check_aspirate_position,
                target_tab="Circulation",
            ),
            ScenarioStep(
                id="STABILIZE_VAE",
                title="Support perfusion and confirm recovery",
                instruction=(
                    "Give a vasopressor (norepinephrine or phenylephrine) if needed while 100% O₂ clears "
                    "residual air, restoring <b>MAP ≥ 65 mmHg</b> and <b>SpO₂ ≥ 92%</b>."
                ),
                check_requirements=check_vae_stable,
                target_tab="Medications",
            ),
        ],
        outcomes=(
            CaseOutcome(
                id="stop_n2o",
                label="Discontinued N₂O and delivered 100% O₂",
                check=lambda e: require_oxygen_flow(8.0)(e)[0],
                review="Stop N₂O immediately to prevent bubble expansion and deliver 100% O₂ to accelerate nitrogen washout (Munson and Merrick 1966; Mirski 2007).",
                milestone=True,
            ),
            CaseOutcome(
                id="stop_air_entry",
                label="Stopped surgical air entrainment",
                check=_vae_stopped,
                review="Alert the surgeon immediately to flood the field, wax bone edges, and lower the surgical site (Mirski 2007).",
                milestone=True,
            ),
            CaseOutcome(
                id="aspirate_cvc",
                label="Aspirated right-atrial air via CVC",
                check=_vae_aspirated,
                review="Aspirate air from the right-atrial CVC to relieve RV outflow tract air-lock.",
                milestone=True,
            ),
            CaseOutcome(
                id="durant_or_pressor",
                label="Positioned left lateral (Durant) or supported MAP with vasopressors",
                check=lambda e: (
                    action_taken_in_case(e, ACTION_EVENT_START, "venous_air_embolism")
                    and (e.hemo.position == "left_lateral" or _pressor_used(e, "venous_air_embolism"))
                ),
                review="Support right-ventricular coronary perfusion with vasopressors and consider Durant positioning.",
                milestone=True,
            ),
            CaseOutcome(
                id="stabilized",
                label="Stabilized circulation and oxygenation (MAP ≥ 65, SpO₂ ≥ 92%)",
                check=lambda e: (
                    _vae_stopped(e)
                    and _ventilated_airway(e)[0]
                    and _circulating(e)
                    and e.state.map >= 65.0
                    and (e.state.upper_sao2 if e.extracorporeal.is_va else e.state.sao2) >= 92.0
                ),
                review="Maintain perfusion and 100% O₂ ventilation until EtCO₂ and MAP normalize.",
            ),
        ),
    )


def _pe_anticoagulated(engine) -> bool:
    return any(
        r.amount > 0
        for r in engine.actions.since_event("pulmonary_embolism", ACTION_DRUG_BOLUS, labels=("heparin",))
    )


def _pe_reperfused(engine) -> bool:
    if not action_taken_in_case(engine, ACTION_EVENT_START, "pulmonary_embolism"):
        return False
    return bool(
        engine.actions.since_event("pulmonary_embolism", ACTION_EMBOLISM_TREAT, labels=("thrombolysis",))
    ) or (engine.extracorporeal.is_va and engine.extracorporeal.state.blood_flow_l_min >= 2.0)


def create_pulmonary_embolism() -> Scenario:
    def start_pe(engine) -> None:
        engine.start_pulmonary_embolism(obstruction=0.62)

    def check_rv_support(engine) -> tuple[bool, str]:
        va_ok = (
            action_taken_in_case(engine, ACTION_EVENT_START, "pulmonary_embolism")
            and engine.extracorporeal.is_va
            and engine.extracorporeal.state.blood_flow_l_min >= 2.0
        )
        met = _pressor_used(engine, "pulmonary_embolism") or va_ok
        return met, "" if met else "Start norepinephrine (or epinephrine/vasopressin, or VA ECMO) for RV perfusion"

    def check_heparin_lysis(engine) -> tuple[bool, str]:
        missing = []
        if not _pe_anticoagulated(engine):
            missing.append("Give IV heparin")
        if not _pe_reperfused(engine):
            missing.append("Give Thrombolysis (PE) under Circulation → Crisis procedures (or VA ECMO)")
        return not missing, "; ".join(missing)

    def check_pe_stable(engine) -> tuple[bool, str]:
        if not _pe_reperfused(engine):
            return False, "Initiate thrombolysis or VA ECMO"
        missing = []
        airway_ok, airway_msg = _ventilated_airway(engine)
        if not airway_ok:
            missing.append(airway_msg)
        map_ok, map_msg = _require_monitored_map_ecmo(engine, 65.0)
        if not map_ok:
            missing.append(map_msg)
        if engine.extracorporeal.is_va and not engine.state.spo2_signal_valid:
            pe_starts = engine.actions.since_case(ACTION_EVENT_START, labels=("pulmonary_embolism",))
            pe_t0 = pe_starts[-1].time if pe_starts else 0.0
            ready_abgs = [
                g for g in engine.blood_gases
                if g.site in ("arterial", "right_radial")
                and g.drawn_at >= pe_t0
                and g.ready_at <= engine.state.time
            ]
            if not ready_abgs:
                missing.append("Confirm upper-body oxygenation with an arterial / right-radial ABG")
            elif ready_abgs[-1].sao2 < 92.0:
                missing.append(f"ABG SaO₂: {format_unmet_int(ready_abgs[-1].sao2, low=92.0)}/92%+")
        else:
            spo2_ok, spo2_msg = require_oxygenation(92.0)(engine)
            if not spo2_ok:
                missing.append(spo2_msg)
        if not engine.extracorporeal.is_va:
            if not engine.state.etco2_signal_valid:
                missing.append("Awaiting exhaled CO₂")
            else:
                etco2 = monitor_value(engine, "etco2")
                if etco2 < 24.0:
                    missing.append(f"EtCO₂: {format_unmet_int(etco2, low=24.0)}/24+ mmHg")
        eff_sao2 = engine.state.upper_sao2 if engine.extracorporeal.is_va else engine.state.sao2
        if not missing and not (_circulating(engine, allow_va_ecmo=True) and eff_sao2 >= 92.0 and engine.state.map >= 65.0):
            missing.append("Reassess oxygenation and perfusion")
        return not missing, "; ".join(missing)

    return Scenario(
        id="pulmonary_embolism",
        name="Massive pulmonary embolism (PE)",
        briefing=(
            "Total knee arthroplasty under general anesthesia (TIVA, FiO₂ ~45%). "
            "The thigh tourniquet is about to be deflated; monitor EtCO₂, SpO₂, and blood pressure."
        ),
        setup_engine=_setup_pe_tiva,
        events=(
            CaseEvent(
                id="pe_onset",
                at=12.0,
                cue=(
                    "Moments after tourniquet release, EtCO₂ plunges with a flat capnogram plateau, "
                    "peak airway pressure stays normal, SpO₂ falls, and obstructive shock develops."
                ),
                apply=start_pe,
            ),
        ),
        steps=[
            create_observe_step("pulmonary_embolism"),
            ScenarioStep(
                id="FIO2_100_PE",
                title="Increase FiO₂ to 100%",
                instruction=(
                    "Abrupt dead-space EtCO₂ fall with normal airway pressure and shock suggests "
                    "<b>massive pulmonary embolism</b>. Set <b>100% O₂</b> (O₂ ≥ 6 L/min, Air and N₂O 0 L/min) "
                    "to treat hypoxemia and limit hypoxic pulmonary vasoconstriction."
                ),
                check_requirements=require_oxygen_flow(6.0),
                target_tab="Machine",
            ),
            ScenarioStep(
                id="SUPPORT_RV",
                title="Support RV perfusion with a vasopressor",
                instruction=(
                    "Start <b>norepinephrine</b> (e.g. 5–12 mcg/min) or give another vasopressor "
                    "to protect right ventricular coronary perfusion pressure.<br><br>"
                    "<i>Avoid aggressive fluid boluses that worsen RV distension and septal shift.</i>"
                ),
                check_requirements=check_rv_support,
                target_tab="Medications",
            ),
            ScenarioStep(
                id="ANTICOAG_LYSIS",
                title="Give heparin and initiate reperfusion",
                instruction=(
                    "Give <b>heparin</b> (e.g. 5000 U IV under Drugs) and open "
                    "<b>Circulation → Crisis procedures</b> to administer <b>Thrombolysis (PE)</b> "
                    "(or start VA ECMO rescue)."
                ),
                check_requirements=check_heparin_lysis,
                target_tab="Circulation",
            ),
            ScenarioStep(
                id="STABILIZE_PE",
                title="Confirm reperfusion and stability",
                instruction=(
                    "Confirm rising EtCO₂ (<b>≥ 24 mmHg</b>) as pulmonary blood flow recovers, with "
                    "<b>MAP ≥ 65 mmHg</b> and <b>SpO₂ ≥ 92%</b>."
                ),
                check_requirements=check_pe_stable,
            ),
        ],
        outcomes=(
            CaseOutcome(
                id="fio2_100",
                label="Delivered 100% O₂",
                check=lambda e: require_oxygen_flow(6.0)(e)[0],
                review="Deliver 100% O₂ immediately to improve arterial oxygenation and blunt hypoxic pulmonary vasoconstriction.",
                milestone=True,
            ),
            CaseOutcome(
                id="vasopressor",
                label="Supported RV perfusion with a vasopressor",
                check=lambda e: check_rv_support(e)[0],
                review="Use norepinephrine (or epinephrine/vasopressin, or VA ECMO) to defend systemic MAP and RV coronary perfusion.",
                milestone=True,
            ),
            CaseOutcome(
                id="anticoagulated",
                label="Administered IV unfractionated heparin",
                check=_pe_anticoagulated,
                review="Give unfractionated heparin promptly to prevent thrombus propagation.",
                milestone=True,
            ),
            CaseOutcome(
                id="reperfusion",
                label="Initiated definitive reperfusion (thrombolysis or VA ECMO)",
                check=_pe_reperfused,
                review="Massive PE with obstructive shock requires systemic thrombolysis, embolectomy, or VA ECMO rescue.",
                milestone=True,
            ),
            CaseOutcome(
                id="stabilized",
                label="Restored circulation and gas exchange (MAP ≥ 65, SpO₂ ≥ 92%)",
                check=lambda e: (
                    _pe_reperfused(e)
                    and _ventilated_airway(e)[0]
                    and _circulating(e, allow_va_ecmo=True)
                    and e.state.map >= 65.0
                    and (e.state.upper_sao2 if e.extracorporeal.is_va else e.state.sao2) >= 92.0
                ),
                review="Monitor EtCO₂ recovery and MAP stabilization as pulmonary vascular resistance falls.",
            ),
        ),
    )
