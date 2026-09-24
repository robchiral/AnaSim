"""Four-hour TIVA case followed by routine perturbations."""

import numpy as np

from anasim.core.state import SimulationConfig

PERTURBATION_EVENTS = {
    120: lambda engine: engine.give_drug_bolus("Propofol", 200),
    300: lambda engine: engine.start_hemorrhage(500),
    420: lambda engine: (engine.stop_hemorrhage(), engine.give_fluid(1000)),
    540: lambda engine: engine.give_blood(300),
}
MONITORED = ("hr", "map", "spo2", "bis", "temp_c", "etco2", "co")


def test_four_hour_case_stays_stable(engine_factory):
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="tiva", dt=0.5),
        start=True,
        age=45,
    )
    initial_temp = engine.state.temp_c

    samples = {name: [] for name in MONITORED}
    for step in range(4 * 3600):
        engine.step(1.0)
        if step % 60 == 0:
            for name in MONITORED:
                value = getattr(engine.state, name)
                assert np.isfinite(value), f"{name} is not finite at {step} s"
                samples[name].append(value)

    assert all(30 <= hr <= 180 for hr in samples["hr"])
    assert all(20 <= map_ <= 200 for map_ in samples["map"])
    assert all(90 <= spo2 <= 100 for spo2 in samples["spo2"])
    assert all(5 <= bis <= 95 for bis in samples["bis"])
    assert all(32.0 <= temp <= 40.0 for temp in samples["temp_c"])
    assert abs(samples["temp_c"][120] - initial_temp) < 2.0
    # Perfusion-coupled EtCO2 can transiently fall below 20 with low CO.
    assert all(10 <= etco2 <= 60 for etco2 in samples["etco2"])
    assert all(co > 0.5 for co in samples["co"])
    assert abs(samples["map"][-1] - np.median(samples["map"])) < 20

    for step in range(1800):
        event = PERTURBATION_EVENTS.get(step)
        if event:
            event(engine)
        engine.step(1.0)

    for name in MONITORED:
        assert np.isfinite(getattr(engine.state, name)), f"{name} is not finite after perturbations"
