from anasim.monitors.alarms import AlarmSystem


class DummyTCI:
    def __init__(self, sampling_time: float):
        self.sampling_time = sampling_time
        self.target = 1.0
        self.calls = []

    def step(self, target: float, sim_time: float = None) -> float:
        self.calls.append(sim_time)
        return 0.0 if sim_time is None else sim_time


def test_tci_accumulator_uses_sampling_time(engine_factory):
    engine = engine_factory()
    engine.state.time = 0.0
    engine.propofol_rate_mg_sec = 0.0

    dummy = DummyTCI(0.25)
    engine.tci_prop = dummy

    engine._step_tci(1.0)
    assert len(dummy.calls) == 4
    assert abs(engine.propofol_rate_mg_sec - 1.0) < 1e-9
    assert abs(engine._tci_accumulators["tci_prop"]) < 1e-9

    dummy.calls.clear()
    engine._step_tci(0.3)
    assert len(dummy.calls) == 1
    assert abs(engine._tci_accumulators["tci_prop"] - 0.05) < 1e-9


def test_alarm_delay_respects_dt_override():
    alarms = AlarmSystem(delays={"HR": 2.0}, dt=1.0)

    for _ in range(3):
        res = alarms.update({"HR": 30}, dt=0.5)
        assert "HR" not in res

    res = alarms.update({"HR": 30}, dt=0.5)
    assert res["HR"]["low"]
