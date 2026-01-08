from anasim.core.state import SimulationConfig


class TestShivering:
    def test_shivering_activates_when_cold_and_awake(self, engine_factory, advance_time):
        config = SimulationConfig(mode="awake")
        engine = engine_factory(config=config, start=True)
        engine.state.temp_c = 35.0

        advance_time(engine, 90, dt=1.0)
        assert engine.state.shivering > 0.4, \
            f"Expected shivering to engage when cold and awake, got {engine.state.shivering:.2f}"

    def test_shivering_suppressed_by_nmba(self, engine_factory, advance_time):
        config = SimulationConfig(mode="awake")
        engine = engine_factory(config=config, start=True)
        engine.state.temp_c = 35.0

        engine.pk_roc.state.ce = 10.0
        engine.pk_roc.state.c1 = 10.0
        engine._sync_pk_state()

        advance_time(engine, 90, dt=1.0)
        assert engine.state.shivering < 0.1, \
            f"Expected NMBA to suppress shivering, got {engine.state.shivering:.2f}"
