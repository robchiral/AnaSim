import sys
import time
from collections import deque
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from anasim.core.drug_registry import DRUG_REGISTRY
from anasim.core.state import SimulationConfig, SimulationState
from anasim.patient.domain import HEIGHT_RANGE_CM, WEIGHT_RANGE_KG
from anasim.ui.config_dialog import SimulationSetupDialog
from anasim.ui.controls_widget import ControlPanelWidget
from anasim.ui.main_window import MainWindow
from anasim.ui.monitor_widget import PatientMonitorWidget
from anasim.ui.scenarios import (
    SCENARIO_REGISTRY,
    create_hemorrhage_response,
    create_induction_balanced,
    create_sepsis_response,
)
from anasim.ui.tutorial_overlay import ScenarioOverlay

# Learner actions for each objective of every registered scenario.
SCENARIO_WALKTHROUGHS = {
    "hemorrhage_response": {
        "START_HEMORRHAGE": lambda e: e.start_hemorrhage(800.0),
        "GIVE_FLUIDS": lambda e: e.give_fluid(500),
        "START_VASOPRESSOR": lambda e: e.set_drug_rate("nore", 12.0),
        "STOP_BLEEDING": lambda e: e.stop_hemorrhage(),
        "REASSESS": lambda e: (
            e.give_blood(600),
            e.give_fluid(1000),
            e.set_drug_rate("nore", 25.0),
        ),
    },
    "sepsis_response": {
        "START_SEPSIS": lambda e: e.start_sepsis(),
        "GIVE_FLUIDS": lambda e: e.give_fluid(1000),
        "START_VASOPRESSOR": lambda e: e.set_drug_rate("nore", 15.0),
        "SOURCE_CONTROL": lambda e: e.stop_sepsis(),
    },
    "anaphylaxis_response": {
        "RECOGNIZE": lambda e: e.start_anaphylaxis(),
        "EPINEPHRINE": lambda e: e.give_drug_bolus("epi", 100),
        "FLUIDS": lambda e: e.give_fluid(500),
        "STABILIZE": lambda e: (e.stop_anaphylaxis(), e.set_drug_rate("epi", 6.0)),
    },
    "induction_tiva": {
        "APPLY_MASK": lambda e: e.set_airway_mode("Mask"),
        "SET_FGF_PREOX": lambda e: e.set_fgf(10.0, 0.0, 0.0),
        "START_ANALGESIA": lambda e: e.set_drug_target("remi", 4.0),
        "INDUCE": lambda e: (
            e.give_drug_bolus("propofol", 175),
            e.set_drug_target("propofol", 4.0),
        ),
        "MASK_VENTILATE": lambda e: e.set_bag_mask_ventilation(True),
        "GIVE_NMB": lambda e: e.give_drug_bolus("roc", 50),
        "INTUBATE": lambda e: (e.set_bag_mask_ventilation(False), e.set_airway_mode("ETT")),
        "CONFIRM_ETT": lambda e: e.set_vent_power(True),
        "MAINTENANCE": lambda e: e.set_fgf(2.0, 0.0, 0.0),
    },
    "induction_balanced": {
        "APPLY_MASK": lambda e: e.set_airway_mode("Mask"),
        "SET_FGF_PREOX": lambda e: e.set_fgf(10.0, 0.0, 0.0),
        "INDUCE": lambda e: e.give_drug_bolus("propofol", 175),
        "MASK_VENTILATE": lambda e: e.set_bag_mask_ventilation(True),
        "GIVE_NMB": lambda e: e.give_drug_bolus("roc", 50),
        "INTUBATE": lambda e: (e.set_bag_mask_ventilation(False), e.set_airway_mode("ETT")),
        "CONFIRM_ETT": lambda e: e.set_vent_power(True),
        "MAINTENANCE": lambda e: (
            e.set_vaporizer("Sevoflurane", 2.0),
            e.set_fgf(2.0, 0.0, 0.0),
        ),
    },
    "emergence_tiva": {
        "STOP_AGENTS": lambda e: (e.disable_tci("propofol"), e.disable_tci("remi")),
        "EXTUBATE": lambda e: e.set_airway_mode("Mask"),
    },
    "emergence_balanced": {
        "STOP_AGENTS": lambda e: (
            e.set_vaporizer("Sevoflurane", 0.0),
            e.set_fgf(10.0, 0.0, 0.0),
        ),
        "EXTUBATE": lambda e: e.set_airway_mode("Mask"),
    },
    "oxygen_supply_failure": {
        "DISCONNECT_OXYGEN": lambda e: e.set_oxygen_supply_connected(False),
        "CONNECT_BACKUP_OXYGEN": lambda e: (
            e.set_oxygen_supply_connected(True),
            e.set_fgf(10.0, 0.0, 0.0),
        ),
    },
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def test_setup_dialog_enforces_supported_patient_domain(qapp, monkeypatch):
    dialog = SimulationSetupDialog()

    assert dialog.sb_weight.minimum() == WEIGHT_RANGE_KG[0]
    assert dialog.sb_weight.maximum() == WEIGHT_RANGE_KG[1]
    assert dialog.sb_height.minimum() == HEIGHT_RANGE_CM[0]
    assert dialog.sb_height.maximum() == HEIGHT_RANGE_CM[1]
    dialog.sb_weight.setValue(WEIGHT_RANGE_KG[0])
    dialog.sb_height.setValue(HEIGHT_RANGE_CM[1])
    warning = {}
    monkeypatch.setattr(
        "anasim.ui.config_dialog.QMessageBox.warning",
        lambda _parent, title, message: warning.update(title=title, message=message),
    )

    dialog.accept()

    assert dialog.result_data is None
    assert warning["title"] == "Unsupported patient"
    assert "bmi" in warning["message"]
    dialog.close()


def test_patient_monitor_uses_art_fields_and_timestamp_history(qapp):
    widget = PatientMonitorWidget(arterial_line_enabled=True, sample_interval_s=0.5)
    states = deque(
        SimulationState(
            time=index / 100.0,
            art_pressure=float(index),
            art_sbp=126.0,
            art_dbp=74.0,
            art_map=91.0,
            pleth_voltage=0.25,
        )
        for index in range(1, 11)
    )

    widget.update_numerics(states[-1])
    widget.update_waveforms(SimpleNamespace(output_buffer=states))

    assert widget.num_art.lbl_val.text() == "126/74 (91)"
    assert widget.wave_write_index == 10
    assert widget.art_data[:10] == pytest.approx(range(1, 11))


def test_overlay_gates_each_objective_and_scopes_the_action_log(qapp, engine_factory):
    engine = engine_factory()
    overlay = ScenarioOverlay(create_induction_balanced(), engine)
    navigation = []
    overlay.navigate_requested.connect(navigation.append)
    assert engine.actions.current_step.label == "APPLY_MASK"

    engine.set_airway_mode("None")
    overlay.update_state()
    assert not overlay.btn_next.isEnabled()
    overlay.btn_target.click()
    assert navigation == ["Machine"]

    engine.set_airway_mode("Mask")
    overlay.update_state()
    assert overlay.btn_next.isEnabled()
    overlay.btn_next.click()

    assert overlay.current_step == 1
    assert engine.actions.current_step.label == "SET_FGF_PREOX"
    assert engine.actions.current_step.time == engine.state.time


def _step(scenario, step_id):
    """Return a scenario step by id."""
    return next(step for step in scenario.steps if step.id == step_id)


def _activate(engine, step):
    """Activate a step the way the overlay does, without building the widget."""
    engine.actions.begin_step(step.id, engine.state.time)


def _early_induction(engine):
    engine.give_drug_bolus("propofol", 150)
    for _ in range(20):
        engine.step(1.0)
    assert engine.state.propofol_cp > 2.0


def _early_extubation(engine):
    # Meet the awake criteria so only the airway action is missing.
    engine.state.display_bis = 90.0
    engine.state.rr = 12.0
    engine.state.apnea = False
    engine.set_airway_mode("Mask")


def _stop_tiva(engine):
    engine.disable_tci("propofol")
    engine.disable_tci("remi")


# (scenario, objective, action taken before the objective, the same action repeated during it)
EARLY_ACTIONS = [
    ("hemorrhage_response", "START_HEMORRHAGE",
     lambda e: e.start_hemorrhage(),
     lambda e: (e.stop_hemorrhage(), e.start_hemorrhage())),
    ("hemorrhage_response", "GIVE_FLUIDS",
     lambda e: e.give_fluid(1000),
     lambda e: e.give_fluid(500)),
    ("hemorrhage_response", "START_VASOPRESSOR",
     lambda e: e.set_drug_rate("nore", 5.0),
     lambda e: e.set_drug_rate("nore", 8.0)),
    ("hemorrhage_response", "STOP_BLEEDING",
     lambda e: (e.start_hemorrhage(), e.stop_hemorrhage()),
     lambda e: (e.start_hemorrhage(), e.stop_hemorrhage())),
    ("induction_balanced", "SET_FGF_PREOX",
     lambda e: e.set_fgf(10.0, 0.0, 0.0),
     lambda e: e.set_fgf(9.0, 0.0, 0.0)),
    ("induction_balanced", "INDUCE",
     _early_induction,
     lambda e: e.give_drug_bolus("propofol", 20)),
    ("induction_balanced", "MASK_VENTILATE",
     lambda e: e.set_bag_mask_ventilation(True),
     lambda e: (e.set_bag_mask_ventilation(False), e.set_bag_mask_ventilation(True))),
    ("induction_balanced", "INTUBATE",
     lambda e: e.set_airway_mode("ETT"),
     lambda e: (e.set_airway_mode("Mask"), e.set_airway_mode("ETT"))),
    ("emergence_balanced", "STOP_AGENTS",
     lambda e: (e.set_vaporizer("Sevoflurane", 0.0), e.set_fgf(8.0, 0.0, 0.0)),
     lambda e: (e.set_vaporizer("Sevoflurane", 0.0), e.set_fgf(9.0, 0.0, 0.0))),
    ("emergence_balanced", "EXTUBATE",
     _early_extubation,
     lambda e: (e.set_airway_mode("ETT"), e.set_airway_mode("Mask"))),
    ("emergence_tiva", "STOP_AGENTS",
     _stop_tiva,
     lambda e: (e.set_drug_target("propofol", 3.0), e.set_drug_target("remi", 2.0), _stop_tiva(e))),
    ("oxygen_supply_failure", "DISCONNECT_OXYGEN",
     lambda e: e.set_oxygen_supply_connected(False),
     lambda e: (e.set_oxygen_supply_connected(True), e.set_oxygen_supply_connected(False))),
]


@pytest.mark.parametrize(
    ("scenario_id", "step_id", "early", "repeat"),
    EARLY_ACTIONS,
    ids=[f"{scenario_id}-{step_id}" for scenario_id, step_id, *_ in EARLY_ACTIONS],
)
def test_objectives_ignore_actions_taken_before_activation(
    engine_factory, scenario_id, step_id, early, repeat
):
    spec = next(spec for spec in SCENARIO_REGISTRY if spec.id == scenario_id)
    engine = engine_factory(
        config=SimulationConfig(mode=spec.start_mode, maint_type=spec.maint_type)
    )
    scenario = spec.builder()
    scenario.prepare(engine)
    engine.start()
    step = _step(scenario, step_id)

    early(engine)
    _activate(engine, step)
    assert not step.check_requirements(engine)[0]

    repeat(engine)
    assert step.check_requirements(engine)[0]


def test_vasopressor_objective_needs_the_infusion_to_stay_running(engine_factory):
    engine = engine_factory(start=True)
    step = _step(create_hemorrhage_response(), "START_VASOPRESSOR")

    _activate(engine, step)
    engine.set_drug_rate("nore", 5.0)
    assert step.check_requirements(engine)[0]

    engine.set_drug_rate("nore", 0.0)
    assert not step.check_requirements(engine)[0]


def test_sepsis_fluid_objective_requires_crystalloid(engine_factory):
    engine = engine_factory(start=True)
    step = _step(create_sepsis_response(), "GIVE_FLUIDS")
    _activate(engine, step)

    engine.give_blood(600)
    engine.give_albumin(500)
    assert not step.check_requirements(engine)[0]

    engine.give_fluid(500)
    assert step.check_requirements(engine)[0]


@pytest.mark.parametrize("spec", SCENARIO_REGISTRY, ids=lambda spec: spec.id)
def test_scenario_objectives_stay_reachable(qapp, engine_factory, spec):
    """Every objective must still be completable by acting while it is active."""
    scenario_id = spec.id
    engine = engine_factory(
        config=SimulationConfig(mode=spec.start_mode, maint_type=spec.maint_type)
    )
    scenario = spec.builder()
    scenario.prepare(engine)
    overlay = ScenarioOverlay(scenario, engine)
    engine.start()

    actions = dict(SCENARIO_WALKTHROUGHS[scenario_id])
    sim_seconds = 0.0
    while overlay.current_step < len(scenario):
        step = scenario[overlay.current_step]
        action = actions.pop(step.id, None)
        if action is not None:
            action(engine)
        overlay.update_state()
        if overlay.requirements_met:
            overlay.btn_next.click()
            continue
        assert sim_seconds < 1200, (
            f"{scenario_id}/{step.id} unreachable: {overlay.check_requirements()[1]}"
        )
        engine.step(1.0)
        sim_seconds += 1.0

    assert overlay.btn_next.text() == "Complete"


def test_controls_are_generated_from_typed_registry(qapp, engine_factory):
    panel = ControlPanelWidget(engine_factory())

    assert tuple(panel.drug_widgets) == tuple(spec.key for spec in DRUG_REGISTRY)
    for spec in DRUG_REGISTRY:
        widgets = panel.drug_widgets[spec.key]
        assert widgets["rate"].suffix().strip() == spec.rate_unit
        assert widgets["target"].minimum() == spec.tci_range[0]
        assert widgets["target"].maximum() == spec.tci_range[1]
        assert widgets["bolus"].suffix().strip() == spec.bolus_unit
        assert widgets["bolus"].value() == spec.default_bolus


def test_controls_sync_external_engine_changes(qapp, engine_factory):
    engine = engine_factory()
    panel = ControlPanelWidget(engine)

    engine.set_airway_mode("ETT")
    engine.set_vent_settings(
        rr=10,
        vt=0.45,
        peep=8,
        ie="1:3",
        mode="PCV",
        p_insp=18,
    )
    engine.set_bronchospasm(0.4)
    engine.set_drug_rate("nore", 3.0)
    panel.sync_with_engine()

    assert panel.rb_ett.isChecked()
    assert panel.cb_vent_mode.currentData() == "PCV"
    assert panel.sb_rr.value() == 10
    assert panel.sb_peep.value() == 8
    assert panel.sb_pinsp.value() == 18
    assert panel.cb_ie.currentText() == "1:3"
    assert panel.sb_bronchospasm.value() == 40
    assert panel.drug_widgets["nore"]["rate"].value() == 3.0


def test_ventilator_starts_from_the_panel_and_keeps_its_settings(qapp, engine_factory):
    """Settings entered before starting are delivered and survive a stop and restart."""
    engine = engine_factory(start=True)
    panel = ControlPanelWidget(engine)
    panel.sync_with_engine()
    engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
    panel.rb_ett.click()
    panel.sb_peep.setValue(8)
    panel.btn_vent_power.click()
    for _ in range(60):
        engine.step(1.0)
    panel.sync_with_engine()
    assert engine.vent.is_on and panel.btn_vent_power.isChecked()
    assert engine.resp.state.apnea and engine.state.mv > 5.0

    assert panel.cb_vent_mode.isEnabled()
    panel.cb_vent_mode.setCurrentIndex(panel.cb_vent_mode.findData("PCV"))
    panel.btn_vent_power.click()
    panel.sync_with_engine()
    assert not engine.vent.is_on and not panel.btn_vent_power.isChecked()
    assert (engine.vent.settings.mode, engine.vent.settings.peep) == ("PCV", 8)

    panel.btn_vent_power.click()
    panel.sync_with_engine()
    assert engine.vent.is_on and panel.btn_vent_power.isChecked()
    assert (engine.vent.settings.mode, engine.vent.settings.peep) == ("PCV", 8)


def test_desktop_session_ends_at_cardiac_arrest(qapp, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    dialog = SimulationSetupDialog()
    dialog.accept()
    window = MainWindow(
        {**dialog.result_data, "mode": "awake", "tutorial_mode": False, "end_on_cardiac_arrest": True}
    )
    monkeypatch.setattr("anasim.ui.main_window.QMessageBox.exec", lambda self: QMessageBox.Close)
    engine = window.engine
    steps_after_arrest = []
    step = engine.step

    def recording_step(dt):
        if engine.state.cardiac_arrest:
            steps_after_arrest.append(dt)
        step(dt)

    monkeypatch.setattr(engine, "step", recording_step)
    engine.set_rhythm("Asystole")
    window.sb_speed.setValue(50.0)
    window.btn_start.click()
    while not engine.state.cardiac_arrest:
        window.last_real_time = time.perf_counter() - 0.2
        window.game_loop()

    # The final state is the one at the arrest, and the session cannot resume.
    assert not steps_after_arrest
    assert not window.btn_start.isEnabled()
    window.toggle_simulation()
    assert not engine.running
    window.close()


def test_controls_clear_finished_disturbance(qapp, engine_factory):
    engine = engine_factory()
    panel = ControlPanelWidget(engine)
    profile_index = next(
        index
        for index, (_, key) in enumerate(panel._disturbance_profiles)
        if key == "stim_intubation_pulse"
    )

    panel.cb_disturbance.setCurrentIndex(profile_index)
    panel.b_disturb.setChecked(True)
    assert engine.disturbance_active

    engine.state.time = engine.disturbance_start_time + 50.0
    engine.start()
    engine.step(0.1)
    panel.sync_with_engine()

    assert not engine.disturbance_active
    assert not panel.b_disturb.isChecked()
    assert panel.cb_disturbance.isEnabled()
    assert panel.cb_disturbance.currentIndex() == profile_index
