"""Action log recording and scenario step scoping."""

import pytest

from anasim.core.action_log import (
    ACTION_DRUG_BOLUS,
    ACTION_EVENT_START,
    ACTION_EVENT_STOP,
    ACTION_FLUID,
    ACTION_INFUSION_RATE,
    ACTION_TCI_TARGET,
    ActionLog,
)


class TestActionLog:
    def test_records_keep_order_and_simulation_time(self):
        log = ActionLog()
        log.record(0.0, ACTION_FLUID, label="crystalloid", amount=250)
        log.record(12.5, ACTION_FLUID, label="blood", amount=300)

        assert [record.time for record in log.records] == [0.0, 12.5]
        assert [record.label for record in log.records] == ["crystalloid", "blood"]

    def test_step_scope_excludes_earlier_actions(self):
        log = ActionLog()
        log.record(0.0, ACTION_FLUID, label="crystalloid", amount=1000)
        log.begin_step("GIVE_FLUIDS", 30.0)
        log.record(35.0, ACTION_FLUID, label="crystalloid", amount=250)

        assert log.current_step.label == "GIVE_FLUIDS"
        assert log.current_step.time == 30.0
        assert log.total_since_step(ACTION_FLUID) == 250

    def test_step_scope_holds_while_the_clock_is_paused(self):
        """Positions, not timestamps, separate actions taken during a step."""
        log = ActionLog()
        log.record(42.0, ACTION_FLUID, label="crystalloid", amount=500)
        log.begin_step("GIVE_FLUIDS", 42.0)

        assert log.total_since_step(ACTION_FLUID) == 0.0

    def test_queries_filter_by_action_and_label(self):
        log = ActionLog()
        log.begin_step("START_VASOPRESSOR", 0.0)
        log.record(1.0, ACTION_INFUSION_RATE, label="nore", amount=5.0)
        log.record(2.0, ACTION_TCI_TARGET, label="propofol", amount=4.0)
        log.record(3.0, ACTION_DRUG_BOLUS, label="roc", amount=50.0)

        vasopressors = log.since_step(
            ACTION_INFUSION_RATE, ACTION_TCI_TARGET, labels=("nore", "phenyl", "epi")
        )
        assert [record.label for record in vasopressors] == ["nore"]
        assert log.since_step(ACTION_DRUG_BOLUS)[0].amount == 50.0
        assert log.since_step(ACTION_EVENT_START) == ()

    def test_step_query_requires_an_active_objective(self):
        log = ActionLog()
        log.record(0.0, ACTION_FLUID, label="crystalloid", amount=500)

        assert log.current_step is None
        with pytest.raises(RuntimeError, match="No scenario objective is active"):
            log.total_since_step(ACTION_FLUID)


class TestEngineRecording:
    def test_fluid_boluses_record_their_ordered_volume(self, anesthetized_engine):
        engine = anesthetized_engine
        engine.actions.begin_step("GIVE_FLUIDS", engine.state.time)

        engine.give_fluid(500)
        engine.give_albumin(250)
        engine.give_blood(300)

        assert engine.actions.total_since_step(ACTION_FLUID) == 1050
        assert [record.label for record in engine.actions.since_step(ACTION_FLUID)] == [
            "crystalloid",
            "colloid",
            "blood",
        ]

    def test_drug_controls_record_boluses_rates_and_targets(self, anesthetized_engine):
        engine = anesthetized_engine
        engine.actions.begin_step("INDUCE", engine.state.time)

        engine.give_drug_bolus("propofol", 150)
        engine.set_drug_rate("nore", 4.0)
        engine.set_drug_target("remi", 3.0)

        bolus = engine.actions.since_step(ACTION_DRUG_BOLUS, labels=("propofol",))
        assert [record.amount for record in bolus] == [150]
        rate = engine.actions.since_step(ACTION_INFUSION_RATE, labels=("nore",))
        assert [record.amount for record in rate] == [4.0]
        target = engine.actions.since_step(ACTION_TCI_TARGET, labels=("remi",))
        assert [record.amount for record in target] == [3.0]

    def test_events_record_transitions_only(self, anesthetized_engine):
        engine = anesthetized_engine
        engine.actions.begin_step("START_HEMORRHAGE", engine.state.time)

        engine.start_hemorrhage(400.0)
        engine.start_hemorrhage(400.0)
        engine.stop_hemorrhage()
        engine.stop_hemorrhage()

        started = engine.actions.since_step(ACTION_EVENT_START, labels=("hemorrhage",))
        stopped = engine.actions.since_step(ACTION_EVENT_STOP, labels=("hemorrhage",))
        assert [record.amount for record in started] == [400.0]
        assert len(stopped) == 1

    def test_actions_carry_the_simulation_time_they_were_taken(
        self, anesthetized_engine, advance_time
    ):
        engine = anesthetized_engine
        advance_time(engine, 30)
        engine.give_fluid(250)

        record = engine.actions.records[-1]
        assert record.action == ACTION_FLUID
        assert record.time == engine.state.time
