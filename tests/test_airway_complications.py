import pytest


def test_manual_airway_controls_update_state_and_resistance(awake_engine):
    engine = awake_engine
    engine.set_airway_mode("Mask")

    base_r = engine.resp_mech.resistance
    engine.set_airway_obstruction(0.8)
    engine.set_bronchospasm(0.5)

    engine.step(0.5)

    assert engine.state.airway_obstruction == pytest.approx(0.8, abs=0.05)
    assert engine.state.bronchospasm == pytest.approx(0.5, abs=0.05)
    assert engine.resp_mech.resistance > base_r


def test_auto_laryngospasm_triggers_with_intubation_stimulus(awake_engine):
    engine = awake_engine
    engine.set_airway_mode("Mask")
    engine.set_disturbance_profile("stim_intubation_pulse")
    engine.start_disturbance("stim_intubation_pulse")

    for _ in range(10):
        engine.step(0.2)

    assert engine.state.laryngospasm > 0.1
    assert engine.state.airway_obstruction >= engine.state.laryngospasm


def test_auto_laryngospasm_toggle_disables_response(awake_engine):
    engine = awake_engine
    engine.set_airway_mode("Mask")
    engine.set_auto_laryngospasm(False)
    engine.set_disturbance_profile("stim_intubation_pulse")
    engine.start_disturbance("stim_intubation_pulse")

    for _ in range(10):
        engine.step(0.2)

    assert engine.state.laryngospasm < 0.01
