"""Local anesthetic systemic toxicity (LAST) crisis scenario from intravascular bupivacaine."""

from anasim.core.action_log import (
    ACTION_DRUG_BOLUS,
    ACTION_EVENT_START,
    ACTION_FLUID,
    ACTION_INFUSION_RATE,
    ACTION_LIPID_INFUSION,
)
from anasim.core.monitors import display_int
from anasim.core.state import AirwayType

from .base import (
    CaseEvent,
    CaseOutcome,
    Scenario,
    ScenarioStep,
    action_taken_in_case,
    create_observe_step,
    format_unmet_int,
    patient_stable,
    require_map_at_least,
    require_oxygen_flow,
    require_oxygenation,
)


def _setup_last_crisis(engine) -> None:
    """Awake patient with an epidural catheter in place just before a surgical top-up bolus."""
    engine.place_epidural(True)
    engine.set_fgf(10.0, 0.0, 0.0)
    engine.set_apl_valve(20.0)


def _trigger_last_crisis(engine) -> None:
    """Deliver accidental intravascular 12 mL 0.5% bupivacaine (60 mg IV)."""
    engine.give_epidural_bolus(
        drug="bupivacaine",
        volume_ml=12.0,
        concentration_pct=0.5,
        accidental_intravascular=True,
    )


def _last_onset_reached(engine) -> bool:
    return (
        action_taken_in_case(engine, ACTION_EVENT_START, "last_crisis")
        and (
            engine.state.convulsions
            or engine.toxicity.conduction_block >= 0.25
            or engine.state.map < 72.0
        )
    )


def _benzodiazepine_given(engine) -> bool:
    boluses = engine.actions.since_event("last_crisis", ACTION_DRUG_BOLUS, labels=("midazolam",))
    infusions = engine.actions.since_event("last_crisis", ACTION_INFUSION_RATE, labels=("midazolam",))
    return any(r.amount > 0.0 for r in (*boluses, *infusions))


def _lipid_bolus_target_ml(engine) -> float:
    return 100.0 if engine.patient.weight >= 70.0 else round(1.5 * engine.patient.weight, 1)


def _lipid_infusion_target_ml_min(engine) -> float:
    """ASRA dosing: ~15 mL/min (200-250 mL over 15-20 min) for >= 70 kg; 0.25 mL/kg/min for < 70 kg."""
    return 15.0 if engine.patient.weight >= 70.0 else round(0.25 * engine.patient.weight, 1)


def _lipid_bolus_given(engine) -> bool:
    boluses = engine.actions.since_event("last_crisis", ACTION_FLUID, labels=("lipid",))
    given = sum(r.amount for r in boluses)
    return given >= 0.9 * _lipid_bolus_target_ml(engine)


def _lipid_infusion_started(engine) -> bool:
    rates = engine.actions.since_event("last_crisis", ACTION_LIPID_INFUSION)
    return any(r.amount >= 5.0 for r in rates) or engine.lipid_infusion_ml_min >= 5.0


def _airway_supported(engine) -> bool:
    s = engine.state
    if s.airway_mode in (AirwayType.ETT, AirwayType.SGA, AirwayType.FONA) and not s.esophageal_intubation:
        return engine.vent.is_on or engine.bag_mask_active
    return s.airway_mode == AirwayType.MASK and (engine.bag_mask_active or engine.vent.is_on)


def _contraindicated_drugs_avoided(engine) -> bool:
    boluses = engine.actions.since_event(
        "last_crisis", ACTION_DRUG_BOLUS, labels=("vaso", "esmolol", "labetalol", "lidocaine", "flumazenil"),
    )
    infusions = engine.actions.since_event(
        "last_crisis", ACTION_INFUSION_RATE, labels=("vaso", "esmolol", "lidocaine", "flumazenil"),
    )
    if any(r.amount > 0.0 for r in (*boluses, *infusions)):
        return False
    epi_boluses = engine.actions.since_event("last_crisis", ACTION_DRUG_BOLUS, labels=("epi",))
    epi_limit = max(100.0, 1.5 * engine.patient.weight)
    return not any(r.amount > epi_limit for r in epi_boluses)


def create_last_crisis() -> Scenario:
    def check_observe_last(engine) -> tuple[bool, str]:
        return _last_onset_reached(engine), ""

    def check_seizure_o2(engine) -> tuple[bool, str]:
        missing = []
        if not _benzodiazepine_given(engine):
            missing.append("Give midazolam IV to treat LAST seizures")
        if engine.state.airway_mode == AirwayType.NONE:
            missing.append("Connect Facemask or airway")
        o2_ok, o2_msg = require_oxygen_flow(6.0)(engine)
        if not o2_ok:
            missing.append(o2_msg)
        return not missing, "; ".join(missing)

    def check_lipid(engine) -> tuple[bool, str]:
        missing = []
        if not _lipid_bolus_given(engine):
            target = int(display_int(_lipid_bolus_target_ml(engine)))
            boluses = engine.actions.since_event("last_crisis", ACTION_FLUID, labels=("lipid",))
            given = sum(r.amount for r in boluses)
            if given > 0.0:
                missing.append(f"20% lipid bolus: {format_unmet_int(given, low=target)}/{target} mL")
            else:
                missing.append(f"Give 20% lipid emulsion bolus ({target} mL)")
        if not _lipid_infusion_started(engine):
            missing.append("Start 20% lipid emulsion infusion (e.g. 15-18 mL/min)")
        return not missing, "; ".join(missing)

    def check_airway(engine) -> tuple[bool, str]:
        o2_ok, o2_msg = require_oxygen_flow(6.0)(engine)
        missing = []
        if not _airway_supported(engine):
            missing.append("Assist ventilation (Bag-mask, SGA, or Tracheal tube on ventilator)")
        if not o2_ok:
            missing.append(o2_msg)
        return not missing, "; ".join(missing)

    def check_stable(engine) -> tuple[bool, str]:
        missing = []
        if engine.state.convulsions or engine.toxicity.seizure_activity >= 0.25:
            missing.append("Control seizure activity with midazolam")
        if engine.toxicity.conduction_block >= 0.35:
            missing.append("Continue lipid rescue and oxygenation until wide-complex block resolves")
        map_ok, map_msg = require_map_at_least(65.0)(engine)
        if not map_ok:
            missing.append(map_msg)
        spo2_ok, spo2_msg = require_oxygenation(94.0)(engine)
        if not spo2_ok:
            missing.append(spo2_msg)
        if not missing and not patient_stable(engine):
            missing.append("Reassess oxygenation and perfusion")
        return not missing, "; ".join(missing)

    def lipid_instruction(engine) -> str:
        bolus_ml = int(display_int(_lipid_bolus_target_ml(engine)))
        rate_ml_min = _lipid_infusion_target_ml_min(engine)
        return (
            f"Under Drugs (Rescue and antidotes), give a <b>20% lipid emulsion {bolus_ml} mL bolus</b> "
            f"and start an infusion near <b>{rate_ml_min:g} mL/min</b> "
            "(≥ 70 kg: 100 mL bolus then 200-250 mL over 15-20 min; < 70 kg: 1.5 mL/kg then 0.25 mL/kg/min).<br><br>"
            "<i>Avoid vasopressin, calcium channel blockers, beta-blockers, local anesthetics (lidocaine), "
            "flumazenil, and epinephrine doses above 1 mcg/kg.</i>"
        )

    return Scenario(
        id="last_crisis",
        name="Local anesthetic systemic toxicity (LAST)",
        briefing=(
            "Awake patient with an epidural catheter receiving a 12 mL bolus of 0.5% bupivacaine "
            "(60 mg). Watch closely for neurologic excitation and cardiovascular toxicity."
        ),
        setup_engine=_setup_last_crisis,
        events=(
            CaseEvent(
                id="intravascular_bupivacaine",
                at=2.0,
                cue="",
                apply=_trigger_last_crisis,
            ),
            CaseEvent(
                id="last_crisis_onset",
                at=12.0,
                when=_last_onset_reached,
                cue=(
                    "Seconds after the epidural top-up, the patient complains of metallic taste and ringing ears, "
                    "then develops generalized convulsions with widening QRS complexes and falling blood pressure."
                ),
                apply=lambda _e: None,
            ),
        ),
        steps=[
            create_observe_step("last_crisis", check=check_observe_last),
            ScenarioStep(
                id="SEIZURE_O2",
                title="Treat seizures and deliver 100% O₂",
                instruction=(
                    "Convulsions and wide-complex hypotension indicate <b>local anesthetic systemic toxicity (LAST)</b>. "
                    "Stop the injection and call for help and the LAST kit.<br><br>"
                    "Give <b>midazolam 2-5 mg IV</b> in Drugs, apply <b>Facemask</b>, and set <b>100% O₂</b> "
                    "(O₂ ≥ 6 L/min, air and N₂O 0 L/min).<br><br>"
                    "<i>Preventing hypoxemia and acidosis limits cardiotoxicity; avoid large propofol doses.</i>"
                ),
                check_requirements=check_seizure_o2,
                target_tab="Medications",
            ),
            ScenarioStep(
                id="LIPID_RESCUE",
                title="Administer 20% lipid emulsion",
                instruction=lipid_instruction,
                check_requirements=check_lipid,
                target_tab="Medications",
            ),
            ScenarioStep(
                id="AIRWAY_VENT",
                title="Maintain ventilation and prevent acidosis",
                instruction=(
                    "Assist ventilation on <b>100% O₂</b> (<b>Bag-mask</b>, <b>SGA</b>, or <b>Tracheal tube</b>) "
                    "to prevent hypercapnia and respiratory acidosis.<br><br>"
                    "<i>If severe hypotension persists, use low-dose epinephrine (≤ 1 mcg/kg); sodium bicarbonate "
                    "(50 mEq IV) can help narrow severe QRS widening.</i>"
                ),
                check_requirements=check_airway,
                target_tab="Machine",
            ),
            ScenarioStep(
                id="STABILIZE_LAST",
                title="Stabilize circulation and monitor recovery",
                instruction=(
                    "Continue lipid rescue, oxygenation, and perfusion support until seizures stop, "
                    "conduction block resolves, <b>MAP ≥ 65 mmHg</b>, and <b>SpO₂ ≥ 94%</b>.<br><br>"
                    "<i>Keep total 20% lipid below 12 mL/kg and monitor for recurrence (≥ 2 h after CNS signs; "
                    "≥ 4-6 h or ICU care after cardiovascular instability). Alert ECMO/CPB early if refractory.</i>"
                ),
                check_requirements=check_stable,
            ),
        ],
        outcomes=(
            CaseOutcome(
                id="seizure_control",
                label="Treated LAST seizures with a benzodiazepine",
                check=lambda e: _benzodiazepine_given(e) and not e.state.convulsions,
                review="Treat LAST seizures with benzodiazepines (midazolam); avoid large propofol doses during cardiovascular instability.",
                milestone=True,
            ),
            CaseOutcome(
                id="lipid_rescue",
                label="Administered 20% lipid emulsion bolus and infusion",
                check=lambda e: _lipid_bolus_given(e) and _lipid_infusion_started(e),
                review="Give 20% lipid emulsion promptly (≥ 70 kg: 100 mL bolus + 200-250 mL over 15-20 min; < 70 kg: 1.5 mL/kg + 0.25 mL/kg/min; max 12 mL/kg).",
                milestone=True,
            ),
            CaseOutcome(
                id="airway_oxygenation",
                label="Established 100% O₂ and assisted ventilation",
                check=lambda e: _airway_supported(e) and require_oxygen_flow(6.0)(e)[0],
                review="Ventilate with 100% O₂ to prevent hypoxemia and acidosis, which aggravate bupivacaine cardiotoxicity and impair lipid binding.",
                milestone=True,
            ),
            CaseOutcome(
                id="avoided_contraindicated",
                label="Avoided contraindicated agents during LAST",
                check=_contraindicated_drugs_avoided,
                review="Avoid vasopressin, calcium channel blockers, beta-blockers, local anesthetics (lidocaine), flumazenil, and high-dose epinephrine (> 1 mcg/kg) during LAST.",
            ),
            CaseOutcome(
                id="stabilized",
                label="Stabilized perfusion and oxygenation (MAP ≥ 65, SpO₂ ≥ 94%)",
                check=lambda e: (
                    patient_stable(e)
                    and not e.state.convulsions
                    and e.toxicity.conduction_block < 0.35
                ),
                review="Maintain lipid rescue, normocapnic oxygenation, and cautious perfusion support until conduction block and hypotension resolve.",
            ),
        ),
    )
