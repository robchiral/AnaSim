import pytest

from anasim.core.state import SimulationConfig


@pytest.mark.parametrize("enabled", [True, False])
def test_intubation_stimulus_triggers_laryngospasm_unless_disabled(engine_factory, enabled):
    engine = engine_factory(start=True)
    engine.set_airway_mode("Mask")
    engine.set_auto_laryngospasm(enabled)
    engine.start_disturbance("stim_intubation_pulse")

    for _ in range(10):
        engine.step(0.2)

    if not enabled:
        assert engine.state.laryngospasm < 0.01
        return
    assert engine.state.laryngospasm > 0.1
    assert engine.state.airway_obstruction >= engine.state.laryngospasm

    # The 50 s stimulus ends and the spasm resolves.
    for _ in range(240):
        engine.step(0.2)
    assert not engine.disturbance_active
    severity_at_end = engine.state.laryngospasm
    for _ in range(100):
        engine.step(0.2)
    assert engine.state.laryngospasm < severity_at_end * 0.15


def test_mask_obstruction_raises_airway_pressure_and_leaks_delivered_breaths(awake_engine):
    """The ventilator still pushes set VT, so Ppeak rises; gas that cannot enter
    the lungs leaks at the mask, so exhaled VT and MV fall together."""
    engine = awake_engine
    engine.set_airway_mode("Mask")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)

    # Suppress spontaneous drive to isolate assisted ventilation.
    dose = 0.8 * engine.patient.weight
    engine.give_drug_bolus("Rocuronium", dose)
    for _ in range(120):
        engine.step(0.5)

    baseline = engine.get_latest_state()

    engine.set_airway_obstruction(0.8)
    for _ in range(120):
        engine.step(0.5)

    obstructed = engine.state
    assert obstructed.paw_peak > baseline.paw_peak + 5.0
    assert obstructed.vt < baseline.vt * 0.5
    assert obstructed.mv < baseline.mv * 0.5


def test_loss_of_consciousness_collapses_unsupported_airway(engine_factory):
    engine = engine_factory(config=SimulationConfig(tci_enabled=True), start=True)
    engine.enable_tci("propofol", 3.0)
    for _ in range(3000):
        engine.step(0.1)
    assert engine.state.airway_obstruction > 0.2

    engine.set_airway_mode("Mask")
    engine.set_vent_settings(rr=0.0, vt=0.0, peep=0.0, ie="1:2", mode="CPAP")
    engine.set_vent_power(True)
    engine.step(0.1)
    assert engine.state.airway_obstruction > 0.2

    engine.set_vent_settings(rr=0.0, vt=0.0, peep=5.0, ie="1:2", mode="CPAP")
    engine.step(0.1)
    assert engine.state.airway_obstruction == 0.0


def test_obstructed_airway_preserves_effort_without_restoring_ventilation(awake_engine):
    engine = awake_engine
    engine.resp.hcvr_slope_baseline = 0.0
    engine.set_airway_mode("ETT")
    for _ in range(200):
        engine.step(0.1)
    baseline_effort = engine.resp_mech.effort.amplitude
    baseline_co2 = engine.resp.state.pa_co2

    engine.set_airway_obstruction(1.0)
    for _ in range(600):
        engine.step(0.1)
    resp, effort = engine.resp.state, engine.resp_mech.effort
    assert resp.drive_central > 0.0 and resp.muscle_factor == 1.0
    assert effort.rr > 8.0 and effort.target_vt > 0.4
    assert effort.amplitude == pytest.approx(baseline_effort, rel=0.05)
    assert engine.state.mv < 0.01 and resp.va < 0.01
    assert resp.pa_co2 > baseline_co2 + 8.0

    engine.set_airway_obstruction(0.0)
    for _ in range(300):
        engine.step(0.1)
    assert engine.state.vt > 400.0 and engine.resp.state.va > 3.0

    engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
    for _ in range(1200):
        engine.step(0.1)
    assert engine.resp_mech.effort.amplitude == 0.0
