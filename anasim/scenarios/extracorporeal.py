"""Extracorporeal support scenarios: VV ECMO, VA ECMO Harlequin, CPB weaning, and protamine reversal."""

from anasim.core.action_log import (
    ACTION_DRUG_BOLUS,
    ACTION_EXTRACORPOREAL,
    ACTION_INFUSION_RATE,
)
from anasim.core.monitors import display_int

from .base import (
    CaseOutcome,
    Scenario,
    ScenarioStep,
    equilibrate_maintenance_fio2,
    format_unmet_int,
    infusion_running,
    join_messages,
    monitor_value,
    preroll_scenario,
    require_all,
    require_etco2_above,
    require_map_at_least,
    require_oxygenation,
    require_tracheal_ventilation,
    stability_outcome,
    tracheal_ventilation_confirmed,
)


def _latest_blood_gas(engine, *, since: float = -1.0, sites: tuple[str, ...] | None = None):
    """Return the most recent drawn blood gas matching optional start time and site filter."""
    for sample in reversed(engine.blood_gases):
        if sample.drawn_at < since - 1e-6:
            continue
        if sites is not None and sample.site not in sites:
            continue
        return sample
    return None


def _ecmo_perfused_map(engine, threshold: float = 65.0) -> tuple[bool, str]:
    """Require MAP >= threshold while either native beats or arterial pump flow perfuses."""
    ec = engine.extracorporeal
    if ec.is_arterial_support and ec.state.blood_flow_l_min >= 1.0:
        map_val = monitor_value(engine, "map")
        if map_val < threshold:
            shown = format_unmet_int(map_val, low=threshold)
            return False, f"MAP: {shown}/{threshold:.0f}+"
        if engine.state.map < threshold:
            return False, "Reassess blood pressure and perfusion"
        return True, ""
    return require_map_at_least(threshold)(engine)


# ---------------------------------------------------------------------------
# 1. VV ECMO for severe refractory ARDS
# ---------------------------------------------------------------------------


def _prepare_vv_ecmo_ards(engine) -> None:
    """Severe ARDS hypoxemia and permissive hypercapnia despite FiO₂ 1.0 and PEEP 14 cmH₂O."""
    engine.config.arterial_line_enabled = True
    engine.set_fgf(6.0, 0.0, 0.0)
    engine.set_vent_power(True)
    engine.set_vent_settings(mode="VCV", rr=14.0, vt=0.35, peep=14.0, ie="1:2")
    engine.set_ards_shunt(0.62)
    preroll_scenario(engine, 120.0, dt=2.0, settle_s=15.0, sample_interval_s=5.0)


def _require_vv_started(engine) -> tuple[bool, str]:
    ec = engine.extracorporeal
    if not ec.is_vv:
        return False, "Start VV ECMO in the Circulation tab"
    msgs = []
    if ec.state.blood_flow_l_min < 3.0:
        msgs.append(f"Blood flow: {ec.state.blood_flow_l_min:.1f}/3.0+ L/min")
    if ec.state.sweep_gas_l_min < 2.0:
        msgs.append(f"Sweep gas: {ec.state.sweep_gas_l_min:.1f}/2.0+ L/min")
    return not msgs, join_messages(msgs)


def _require_vv_active(engine) -> tuple[bool, str]:
    if not engine.extracorporeal.is_vv:
        return False, "Maintain VV ECMO support"
    return True, ""


def _require_lung_rest(engine) -> tuple[bool, str]:
    """Require ultra-protective lung-rest ventilator settings once VV ECMO is established."""
    settings = engine.vent.settings
    msgs = []
    if settings.rr > 12.0:
        msgs.append(f"Ventilator RR: {settings.rr:g}/≤12/min (lung rest)")
    if settings.mode == "VCV" and settings.tv > 320.0:
        msgs.append(f"Ventilator VT: {settings.tv:.0f}/≤320 mL (lung rest)")
    elif settings.mode == "PCV" and settings.p_insp > 15.0:
        msgs.append(f"Inspiratory pressure: {settings.p_insp:g}/≤15 cmH₂O (lung rest)")
    return not msgs, join_messages(msgs)


def _require_vv_abg_confirmed(engine) -> tuple[bool, str]:
    starts = engine.actions.since_case(ACTION_EXTRACORPOREAL, labels=("start_vv_ecmo",))
    since = starts[-1].time if starts else 0.0
    latest = _latest_blood_gas(engine, since=since, sites=("arterial", "right_radial", "femoral"))
    if latest is None:
        return False, "Draw an ABG in the Labs tab to verify PaCO₂"
    if latest.ready_at > engine.state.time + 1e-6:
        return False, "Wait for the ABG result in Labs"
    if latest.pa_co2 > 50.0:
        return False, f"Latest ABG PaCO₂: {latest.pa_co2:.0f}/≤50 mmHg; increase sweep gas and recheck ABG"
    return True, ""


def _vv_stable(engine) -> bool:
    state = engine.state
    return (
        not state.cardiac_arrest
        and engine.spontaneous_circulation
        and state.sao2 >= 88.0
        and state.map >= 65.0
    )


def create_vv_ecmo_ards() -> Scenario:
    return Scenario(
        id="vv_ecmo_ards",
        name="VV ECMO for severe ARDS",
        setup_engine=_prepare_vv_ecmo_ards,
        briefing=(
            "Severe ARDS on lung-protective ventilation with 100% oxygen, low tidal volume, and PEEP 14 cmH₂O. "
            "Refractory hypoxemia and permissive hypercapnia persist despite maximal conventional ventilation."
        ),
        steps=[
            ScenarioStep(
                id="INITIATE_VV_ECMO",
                title="Initiate VV ECMO",
                instruction=(
                    "Open <b>Extracorporeal support</b> in the <b>Circulation</b> tab and start <b>VV ECMO</b>. "
                    "Target a circuit blood flow of at least <b>3.0 L/min</b> and sweep gas of at least <b>2.0 L/min</b> "
                    "at 100% sweep FiO₂.<br><br>"
                    "<i>Consider systemic heparinization at <b>50-100 U/kg</b> in Drugs (adjust the bolus card down "
                    "from its 300 U/kg CPB default) if there is no bleeding contraindication.</i>"
                ),
                check_requirements=_require_vv_started,
                target_tab="Circulation",
            ),
            ScenarioStep(
                id="STABILIZE_VV_ECMO",
                title="Transition to lung-rest ventilation and stabilize",
                instruction=(
                    "Titrate <b>blood flow</b> (watch inline <b>Pre-O₂ SvO₂</b> for recirculation) and <b>sweep gas</b> "
                    "to <b>SpO₂ ≥ 88%</b>, using low-dose <b>norepinephrine</b> if needed for <b>MAP ≥ 65 mmHg</b>.<br><br>"
                    "Switch the ventilator in <b>Machine</b> to <b>lung-rest settings</b>: <b>RR ≤ 12/min</b>, "
                    "<b>VT ≤ 320 mL</b> (or low driving pressure), PEEP ~10 cmH₂O, and FiO₂ ≤ 50% as tolerated."
                ),
                check_requirements=require_all(
                    _require_vv_active,
                    _require_lung_rest,
                    require_oxygenation(88.0),
                    require_map_at_least(65.0),
                    require_tracheal_ventilation,
                ),
                target_tab="Circulation",
            ),
            ScenarioStep(
                id="CONFIRM_VV_ABG",
                title="Confirm CO₂ clearance by ABG",
                instruction=(
                    "Draw an <b>arterial blood gas</b> in <b>Labs</b> and confirm <b>PaCO₂ ≤ 50 mmHg</b>; "
                    "adjust sweep gas gradually and recheck if elevated.<br><br>"
                    "<i>On lung-rest ventilation, EtCO₂ underestimates PaCO₂. Avoid dropping PaCO₂ by more than "
                    "50% rapidly, which increases intracranial hemorrhage risk.</i>"
                ),
                check_requirements=require_all(
                    _require_vv_active,
                    _require_vv_abg_confirmed,
                    require_oxygenation(88.0),
                    require_map_at_least(65.0),
                    require_tracheal_ventilation,
                ),
                target_tab="Labs",
            ),
        ],
        outcomes=(
            CaseOutcome(
                "vv_started",
                "VV ECMO initiated",
                lambda engine: _require_vv_started(engine)[0],
                "Initiate venovenous ECMO with adequate circuit flow and sweep gas for refractory ARDS hypoxemia.",
                milestone=True,
            ),
            CaseOutcome(
                "lung_rest",
                "Transitioned to ultra-protective lung-rest ventilation",
                lambda engine: engine.extracorporeal.is_vv and _require_lung_rest(engine)[0],
                "Reduce tidal volume/driving pressure and respiratory rate to ultra-protective lung-rest settings once on VV ECMO.",
                milestone=True,
            ),
            CaseOutcome(
                "oxygenation",
                "Systemic oxygenation restored (SpO₂ ≥ 88%)",
                lambda engine: engine.state.sao2 >= 88.0,
                "Titrate VV ECMO blood flow and sweep FiO₂ until arterial saturation reaches at least 88% (typically 85-88%).",
            ),
            CaseOutcome(
                "decarboxylation",
                "CO₂ clearance confirmed by ABG",
                lambda engine: _require_vv_abg_confirmed(engine)[0] and engine.state.pa_co2 <= 50.0,
                "Verify PaCO₂ ≤ 50 mmHg on an arterial blood gas and titrate sweep gas gradually to avoid rapid PaCO₂ shifts.",
            ),
            CaseOutcome(
                "stability",
                "Oxygenation and circulation stable",
                _vv_stable,
                "Maintain hemodynamic and respiratory stability on VV ECMO.",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 2. Peripheral VA ECMO differential hypoxia (Harlequin / North-South syndrome)
# ---------------------------------------------------------------------------


def _prepare_va_ecmo_harlequin(engine) -> None:
    """Peripheral femoral VA ECMO with recovering LV and hypoxic lungs on low-FiO₂ ventilation."""
    engine.config.arterial_line_enabled = True
    engine.set_vent_power(True)
    engine.set_vent_settings(mode="VCV", rr=10.0, vt=0.36, peep=5.0, ie="1:2")
    equilibrate_maintenance_fio2(engine, target_fio2=0.24, min_air_l_min=3.5, air_vo2_factor=12.0)
    engine.set_ards_shunt(0.66)
    engine.start_va_ecmo(
        blood_flow_l_min=3.0,
        sweep_gas_l_min=4.0,
        sweep_fio2=1.0,
        cannulation="peripheral_fem",
        prime_volume_ml=0.0,
    )
    preroll_scenario(engine, 120.0, dt=2.0, settle_s=15.0, sample_interval_s=5.0)


def _require_differential_abg(engine) -> tuple[bool, str]:
    upper = _latest_blood_gas(engine, since=0.0, sites=("right_radial", "arterial"))
    lower = _latest_blood_gas(engine, since=0.0, sites=("femoral",))
    msgs = []
    if upper is None:
        msgs.append("Draw a right-radial blood gas in the Labs tab")
    if lower is None:
        msgs.append("Draw a femoral blood gas in the Labs tab")
    if msgs:
        return False, join_messages(msgs)
    assert upper is not None and lower is not None
    if max(upper.ready_at, lower.ready_at) > engine.state.time + 1e-6:
        return False, "Wait for the right-radial and femoral blood gas results in Labs"
    return True, ""


def _require_upper_oxygenation(engine) -> tuple[bool, str]:
    msgs = []
    fio2_ok = engine.circuit.composition.fio2 >= 0.80
    if not fio2_ok:
        msgs.append(f"Circuit FiO₂: {format_unmet_int(engine.circuit.composition.fio2 * 100.0, low=80)}%/80%+")
    spo2_ok, spo2_msg = require_oxygenation(92.0)(engine)
    if not spo2_ok:
        ec = engine.extracorporeal
        nonpulsatile_va = (
            ec.is_arterial_support
            and ec.state.blood_flow_l_min >= 1.0
            and not engine.state.spo2_signal_valid
        )
        if nonpulsatile_va:
            # High retrograde VA flow can close the aortic valve and flatten the pleth;
            # confirm upper-body oxygenation via a resulted post-adjustment right-radial ABG.
            latest_rad = _latest_blood_gas(engine, since=0.0, sites=("right_radial", "arterial"))
            if latest_rad is not None and latest_rad.ready_at <= engine.state.time + 1e-6 and latest_rad.sao2 >= 92.0 and engine.state.upper_sao2 >= 92.0:
                spo2_ok = True
            elif latest_rad is not None and latest_rad.ready_at > engine.state.time + 1e-6:
                msgs.append("Wait for the confirmatory right-radial ABG in Labs")
            else:
                msgs.append("Confirm upper-body oxygenation with a right-radial ABG in Labs (nonpulsatile SpO₂)")
        else:
            msgs.append(spo2_msg)
    return not msgs, join_messages(msgs)


def _va_harlequin_stable(engine) -> bool:
    state = engine.state
    ec = engine.extracorporeal
    perfused = engine.spontaneous_circulation or (ec.is_arterial_support and ec.state.blood_flow_l_min >= 1.0)
    return not state.cardiac_arrest and perfused and state.upper_sao2 >= 92.0 and state.map >= 65.0


def create_va_ecmo_harlequin() -> Scenario:
    return Scenario(
        id="va_ecmo_harlequin",
        name="VA ECMO differential hypoxia",
        setup_engine=_prepare_va_ecmo_harlequin,
        briefing=(
            "A patient on peripheral femoral VA ECMO (3.0 L/min) has a recovering left ventricle with "
            "severe pulmonary edema/ARDS on low FiO₂ ventilation. Right-hand pulse oximetry is falling "
            "despite high ECMO oxygenator output."
        ),
        steps=[
            ScenarioStep(
                id="CONFIRM_HARLEQUIN",
                title="Confirm differential hypoxia",
                instruction=(
                    "On peripheral femoral VA ECMO, poorly oxygenated blood ejected by the native LV perfuses "
                    "the coronaries, brain, and right arm whenever the lungs fail. Open the <b>Labs</b> tab, "
                    "draw paired <b>Right radial</b> and <b>Femoral</b> blood gases, and wait for the results "
                    "to confirm the north-south oxygenation gradient."
                ),
                check_requirements=_require_differential_abg,
                target_tab="Labs",
            ),
            ScenarioStep(
                id="CORRECT_HARLEQUIN",
                title="Restore upper-body oxygenation",
                instruction=(
                    "In <b>Machine</b>, set <b>100% O₂</b>, raise <b>PEEP</b> to 10-14 cmH₂O, and increase tidal volume "
                    "to ~450 mL. In <b>Circulation</b>, raise <b>VA ECMO blood flow</b> cautiously (toward <b>5.0 L/min</b>) "
                    "until right-hand <b>SpO₂ ≥ 92%</b>.<br><br>"
                    "<i>Higher femoral VA flow raises LV afterload and can close the aortic valve (if the pleth flattens, "
                    "confirm with a <b>Right radial</b> ABG). Refractory cases require V-AV ECMO, axillary/central "
                    "cannulation, or conversion to VV ECMO.</i>"
                ),
                check_requirements=require_all(
                    _require_upper_oxygenation,
                    lambda engine: _ecmo_perfused_map(engine, 65.0),
                    require_tracheal_ventilation,
                ),
                target_tab="Machine",
            ),
        ],
        outcomes=(
            CaseOutcome(
                "differential_abg",
                "Differential blood gases compared",
                lambda engine: _require_differential_abg(engine)[0],
                "Compare right-radial (upper-body) and femoral (lower-body) blood gases to diagnose Harlequin syndrome.",
                milestone=True,
            ),
            CaseOutcome(
                "lung_optimized",
                "Native lung oxygenation optimized",
                lambda engine: engine.circuit.composition.fio2 >= 0.80 and engine.vent.is_on,
                "Increase ventilator FiO₂ and PEEP so native LV ejection carries oxygenated blood to the arch.",
            ),
            CaseOutcome(
                "upper_sao2",
                "Upper-body oxygenation restored",
                lambda engine: engine.state.upper_sao2 >= 92.0,
                "Restore right-radial (cerebral and coronary) saturation to at least 92%.",
            ),
            CaseOutcome(
                "stability",
                "Oxygenation and circulation stable",
                _va_harlequin_stable,
                "Maintain systemic perfusion and upper-body oxygenation on VA ECMO.",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 3. Cardiopulmonary bypass rewarming and weaning
# ---------------------------------------------------------------------------


def _prepare_cpb_weaning(engine) -> None:
    """Mid-case mildly hypothermic CPB with aortic cross-clamp, cardioplegic arrest, and full heparinization."""
    engine.config.arterial_line_enabled = True
    engine.set_fgf(1.0, 1.0, 0.0)
    engine.give_drug_bolus("heparin", 300.0 * engine.patient.weight)
    engine.start_cpb(
        blood_flow_l_min=4.5,
        sweep_gas_l_min=3.0,
        sweep_fio2=0.80,
        prime_volume_ml=1400.0,
        initial_reservoir_ml=800.0,
        heater_cooler_on=True,
        water_target_c=31.5,
        bypass_alarm_mode=True,
    )
    engine.set_cpb_cardioplegia(True)
    engine.set_vent_power(False)
    preroll_scenario(engine, 240.0, dt=2.0, settle_s=10.0, sample_interval_s=5.0)


def _require_rewarm_unclamp(engine) -> tuple[bool, str]:
    st = engine.extracorporeal.state
    msgs = []
    if engine.extracorporeal.is_cpb:
        if not st.heater_cooler_on or st.water_target_c < 36.5:
            msgs.append("Set CPB heater-cooler to 37 °C")
        if st.aortic_cross_clamp:
            msgs.append("Release the aortic cross-clamp")
        if st.cardioplegia_arrest:
            msgs.append("Stop cardioplegia arrest")
    if engine.state.temp_c < 36.0:
        msgs.append(f"Core temperature: {engine.state.temp_c:.1f}/36.0+ °C")
    return not msgs, join_messages(msgs)


def _require_ventilate_and_fill(engine) -> tuple[bool, str]:
    msgs = []
    if not engine.vent.is_on:
        msgs.append("Restart mechanical ventilation")
    if engine.circuit.composition.fio2 < 0.70:
        msgs.append(f"Circuit FiO₂: {format_unmet_int(engine.circuit.composition.fio2 * 100.0, low=70)}%/70%+")
    if engine.extracorporeal.is_cpb:
        st = engine.extracorporeal.state
        if st.venous_clamp_open > 0.25:
            msgs.append("Clamp the CPB venous line (≤25% open) to fill the heart")
        if st.blood_flow_l_min > 2.5:
            msgs.append(f"Reduce CPB flow: {st.blood_flow_l_min:.1f}/≤2.5 L/min")
    pulse_pressure = max(0.0, engine.state.art_sbp - engine.state.art_dbp)
    if pulse_pressure < 25.0:
        shown_pp = format_unmet_int(pulse_pressure, low=25.0)
        msgs.append(f"ART pulse pressure: {shown_pp}/25+ mmHg (fill the heart)")
    etco2_ok, etco2_msg = require_etco2_above(20.0)(engine)
    if not etco2_ok:
        msgs.append(etco2_msg)
    return not msgs, join_messages(msgs)


def _require_separated(engine) -> tuple[bool, str]:
    msgs = []
    if engine.extracorporeal.is_active:
        msgs.append("Stop cardiopulmonary bypass")
    if engine.extracorporeal.state.bypass_alarm_mode:
        msgs.append("Turn off bypass alarm mode")
    return not msgs, join_messages(msgs)


def create_cpb_weaning() -> Scenario:
    return Scenario(
        id="cpb_weaning",
        name="CPB rewarming and weaning",
        setup_engine=_prepare_cpb_weaning,
        briefing=(
            "Cardiac repair is finishing on mildly hypothermic cardiopulmonary bypass (fully heparinized at 300 U/kg) "
            "with aortic cross-clamp and cardioplegic arrest. Rewarm the patient, release the cross-clamp, resume "
            "ventilation, refill the heart, and separate cleanly from bypass."
        ),
        steps=[
            ScenarioStep(
                id="REWARM_AND_UNCLAMP",
                title="Rewarm and release cross-clamp",
                instruction=(
                    "In <b>Circulation → Extracorporeal support</b>, set the heater-cooler to <b>37.0 °C</b> "
                    "and <b>release the aortic cross-clamp</b> until organized rhythm returns and core temperature "
                    "reaches <b>≥ 36.0 °C</b>.<br><br>"
                    "<i>Avoid exceeding 37 °C arterial outlet temperature. Check ABG (K⁺, Ca²⁺, Hb, pH), confirm "
                    "pacing/defibrillation readiness, and assess de-airing and ventricles on TEE.</i>"
                ),
                check_requirements=_require_rewarm_unclamp,
                target_tab="Circulation",
            ),
            ScenarioStep(
                id="VENTILATE_AND_FILL",
                title="Resume ventilation and fill the heart",
                instruction=(
                    "In <b>Machine</b>, set <b>100% O₂</b>, recruit the lungs, and <b>start the ventilator</b>.<br><br>"
                    "In <b>Circulation → Extracorporeal support</b>, step <b>CPB flow</b> down toward <b>2.0 L/min</b> "
                    "and restrict the <b>venous line (0-10% open)</b> while watching reservoir volume until "
                    "<b>pulse pressure ≥ 25 mmHg</b> and <b>EtCO₂ > 20 mmHg</b>."
                ),
                check_requirements=_require_ventilate_and_fill,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="SEPARATE_FROM_CPB",
                title="Separate from bypass",
                instruction=(
                    "Turn off <b>bypass alarm mode</b> and click <b>Stop support</b> in <b>Circulation</b>. "
                    "Support <b>MAP ≥ 65 mmHg</b>, <b>SpO₂ ≥ 92%</b>, and <b>EtCO₂ > 25 mmHg</b> with norepinephrine "
                    "or inotropes (epinephrine, dobutamine, milrinone) guided by TEE.<br><br>"
                    "<i>Confirm cardiotomy pump suckers are off before giving protamine.</i>"
                ),
                check_requirements=require_all(
                    _require_separated,
                    require_map_at_least(65.0),
                    require_oxygenation(92.0),
                    require_etco2_above(25.0),
                    require_tracheal_ventilation,
                ),
                target_tab="Circulation",
            ),
        ],
        outcomes=(
            CaseOutcome(
                "rewarmed",
                "Patient rewarmed to normothermia",
                lambda engine: engine.state.temp_c >= 36.0,
                "Rewarm on CPB to at least 36.0 °C before separating from bypass.",
                milestone=True,
            ),
            CaseOutcome(
                "unclamped",
                "Aortic cross-clamp released",
                lambda engine: (
                    bool(engine.actions.since_case(ACTION_EXTRACORPOREAL, labels=("cross_clamp_off",)))
                    or not engine.extracorporeal.state.aortic_cross_clamp
                ),
                "Release the aortic cross-clamp and confirm organized cardiac rhythm before weaning.",
                milestone=True,
            ),
            CaseOutcome(
                "ventilated",
                "Mechanical ventilation resumed",
                tracheal_ventilation_confirmed,
                "Restart mechanical ventilation with oxygen before weaning off CPB.",
            ),
            CaseOutcome(
                "separated",
                "Separated cleanly from CPB",
                lambda engine: not engine.extracorporeal.is_active and engine.spontaneous_circulation,
                "Refill the vascular bed from the reservoir and stop CPB once native ejection is adequate.",
            ),
            stability_outcome("Maintain post-bypass arterial pressure, oxygenation, and ventilation."),
        ),
    )


# ---------------------------------------------------------------------------
# 4. Post-CPB protamine reversal and ACT normalization
# ---------------------------------------------------------------------------

# Estimated remaining heparin (~210 U/kg) after a 60-min post-300 U/kg bypass run.
_PROTAMINE_TARGET_MG_KG = 2.1


def _prepare_protamine_reversal(engine) -> None:
    """Recently separated from a 60-min CPB run with ~200 U/kg residual circulating heparin on ~50% air/O₂."""
    engine.config.arterial_line_enabled = True
    engine.set_fgf(1.0, 1.0, 0.0)
    engine.set_vent_power(True)
    engine.give_drug_bolus("heparin", 300.0 * engine.patient.weight)
    # 60 min preroll loads the peripheral heparin compartment and leaves ~195-200 U/kg in plasma.
    preroll_scenario(engine, 3600.0, dt=15.0, settle_s=10.0, sample_interval_s=30.0)


def _require_initial_act(engine) -> tuple[bool, str]:
    if not engine.acts:
        return False, "Measure baseline ACT in the Labs tab"
    if any(sample.ready_at <= engine.state.time + 1e-6 for sample in engine.acts):
        return True, ""
    return False, "Wait for the baseline ACT result in Labs"


def _reverse_heparin_instruction(engine) -> str:
    weight = engine.patient.weight
    est_mg = int(display_int(_PROTAMINE_TARGET_MG_KG * weight))
    return (
        "First confirm <b>cardiotomy pump suckers are OFF</b> to prevent circuit thrombosis.<br><br>"
        f"In <b>Drugs → Protamine</b>, give a slow <b>10-25 mg test dose</b>, then infuse roughly "
        f"<b>{est_mg} mg</b> total slowly (<b>15-25 mg/min</b> over 10-15 min) while keeping "
        "<b>MAP ≥ 65 mmHg</b>.<br><br>"
        "<i>Slow infusion avoids Type I vasodilation; watch for Type II anaphylactoid or Type III "
        "pulmonary hypertensive reactions during the test dose.</i>"
    )


def _require_protamine_delivered(engine) -> tuple[bool, str]:
    target_mg = _PROTAMINE_TARGET_MG_KG * engine.patient.weight
    given_mg = engine.coagulation.cumulative_protamine_mg
    if given_mg >= target_mg - 1.0 or (given_mg > 0.0 and _require_post_protamine_act(engine)[0]):
        return True, ""
    shown_target = int(display_int(target_mg))
    if given_mg > 0.0:
        shown_given = format_unmet_int(given_mg, low=shown_target)
        return False, f"Protamine delivered: {shown_given}/{shown_target} mg"
    return False, f"Start slow protamine infusion in Drugs (target ~{shown_target} mg)"


def _require_post_protamine_act(engine) -> tuple[bool, str]:
    if infusion_running(engine, "protamine"):
        return False, "Stop the protamine infusion in Drugs before confirming ACT"
    prot_actions = engine.actions.since_case(
        ACTION_DRUG_BOLUS, ACTION_INFUSION_RATE, labels=("protamine",),
    )
    if not prot_actions:
        return False, "Administer protamine in Drugs before checking confirmatory ACT"
    last_prot_time = prot_actions[-1].time
    post_acts = [sample for sample in engine.acts if sample.drawn_at >= last_prot_time - 1e-6]
    if not post_acts:
        return False, "Measure a confirmatory ACT in the Labs tab"
    latest = post_acts[-1]
    if latest.ready_at > engine.state.time + 1e-6:
        return False, "Wait for the confirmatory ACT result in Labs"
    if latest.act_s > 150.0:
        return (
            False,
            f"Latest measured ACT: {latest.act_s:.0f}/≤150 s; reassess protamine dosing and recheck ACT",
        )
    return True, ""


def create_protamine_reversal() -> Scenario:
    return Scenario(
        id="protamine_reversal",
        name="Post-CPB protamine reversal",
        setup_engine=_prepare_protamine_reversal,
        briefing=(
            "The patient has just separated cleanly from a 60-minute cardiopulmonary bypass run after "
            "300 U/kg heparin. Measure the baseline ACT in Labs, reverse residual heparin slowly with protamine "
            "while avoiding rapid-bolus hypotension, and verify normalization with a repeat ACT."
        ),
        steps=[
            ScenarioStep(
                id="MEASURE_PRE_ACT",
                title="Measure post-CPB ACT",
                instruction=(
                    "Open the <b>Labs</b> tab, click <b>Measure ACT</b>, and wait 60 seconds for the "
                    "point-of-care result before dosing protamine."
                ),
                check_requirements=_require_initial_act,
                target_tab="Labs",
            ),
            ScenarioStep(
                id="REVERSE_HEPARIN",
                title="Reverse heparin with slow protamine",
                instruction=_reverse_heparin_instruction,
                check_requirements=require_all(
                    _require_protamine_delivered,
                    require_map_at_least(65.0),
                    require_tracheal_ventilation,
                ),
                target_tab="Medications",
            ),
            ScenarioStep(
                id="CONFIRM_REVERSAL_ACT",
                title="Confirm ACT normalization",
                instruction=(
                    "Stop the protamine infusion in <b>Drugs</b>, then draw a confirmatory <b>ACT</b> in "
                    "<b>Labs</b> (3-5 min after completion in clinical practice) and confirm <b>ACT ≤ 150 s</b> "
                    "with <b>MAP ≥ 65 mmHg</b>.<br><br>"
                    "<i>If ACT stays high, distinguish residual heparin from transient protamine excess, "
                    "and watch for heparin rebound over the next 1-4 hours.</i>"
                ),
                check_requirements=require_all(
                    _require_post_protamine_act,
                    require_map_at_least(65.0),
                    require_oxygenation(92.0),
                    require_tracheal_ventilation,
                ),
                target_tab="Labs",
            ),
        ],
        outcomes=(
            CaseOutcome(
                "baseline_act",
                "Post-CPB ACT measured",
                lambda engine: _require_initial_act(engine)[0],
                "Measure baseline ACT in Labs before administering protamine.",
                milestone=True,
            ),
            CaseOutcome(
                "heparin_reversed",
                "Circulating heparin neutralized",
                lambda engine: engine.state.heparin_u_ml <= 0.25 and engine.state.act_s <= 150.0,
                "Titrate protamine (~1 mg per 100 U circulating heparin) without excessive overdose.",
            ),
            CaseOutcome(
                "confirmatory_act",
                "ACT normalization confirmed",
                lambda engine: _require_post_protamine_act(engine)[0],
                "Stop protamine and confirm a repeat Lab ACT ≤ 150 s.",
            ),
            stability_outcome("Maintain blood pressure and oxygenation during protamine reversal."),
        ),
    )
