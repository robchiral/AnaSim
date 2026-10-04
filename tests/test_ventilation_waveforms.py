"""Mode behavior through the mechanics, gas analyzer, and display sensor."""

import pytest

from anasim.machine.ventilator import RISE_TIME


@pytest.mark.parametrize("mode,passive", [
    ("VCV", True), ("PCV", True), ("PCV-VG", True),
    ("VCV", False), ("PCV", False), ("PCV-VG", False),
    ("SIMV-VC", False), ("SIMV-PC", False), ("SIMV-VG", False),
    ("PSV", False), ("CPAP", False),
])
def test_displayed_waveforms_follow_mode_control_and_actual_gas_movement(awake_engine, mode, passive):
    engine = awake_engine
    engine.resp.hcvr_slope_baseline = 0.0
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(mode=mode, rr=12, vt=0.5, ie="1:2", peep=5, p_insp=12, p_support=8)
    engine.set_vent_power(True)
    if passive:
        engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
    for _ in range(600):
        engine.step(0.1)

    start_volume, previous_flow = engine.state.volume, engine.state.flow
    integrated, positive_time, rises = 0.0, 0.0, []
    above = engine.state.capno_co2 > 15.0
    low, high, checked = 60.0, 0.0, 0
    for _ in range(1500):
        engine.step(0.01)
        state, breath = engine.state, engine.vent._breath
        integrated += 0.01 * (previous_flow + state.flow) / 120.0
        previous_flow = state.flow
        low, high = min(low, state.capno_co2), max(high, state.capno_co2)
        if state.capno_co2 > 15.0 and not above:
            rises.append(state.time)
        above = state.capno_co2 > 15.0
        positive_time = positive_time + 0.01 if state.flow > 0.5 else 0.0
        # Analyzer delay can leave CO2 at the start of inspiration, but a
        # sustained inflow must wash it back to the inspired baseline.
        if positive_time > 0.7:
            assert state.capno_co2 < 1.0
        if mode in ("PSV", "CPAP") or not breath.inspiring:
            continue
        if breath.kind == "VC" and 0.2 < breath.t < breath.flow_end - 0.08:
            assert state.flow == pytest.approx(60.0 * breath.target / breath.flow_end, abs=0.05)
            checked += 1
        elif breath.kind in ("PC", "VG") and breath.t > RISE_TIME + 0.2:
            assert state.paw == pytest.approx(breath.peep + breath.target, abs=0.05)
            if passive:
                assert state.flow > -0.05  # Passive pressure breaths fill until cycling.
            checked += 1
    assert low < 1.0 and high > 20.0
    assert len(rises) >= 2
    assert 60.0 * (len(rises) - 1) / (rises[-1] - rises[0]) == pytest.approx(engine.state.rr, abs=0.2)
    # The displayed flow and volume must describe the same gas movement.
    assert integrated == pytest.approx(engine.state.volume - start_volume, abs=0.005)
    if mode not in ("PSV", "CPAP"):
        assert checked > 100
