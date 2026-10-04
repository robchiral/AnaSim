"""Action log recording and scenario step scoping."""

from anasim.core.action_log import (
    ACTION_DRUG_BOLUS,
    ACTION_EVENT_START,
    ACTION_EVENT_STOP,
    ACTION_FLUID,
    ACTION_INFUSION_RATE,
    ACTION_TCI_TARGET,
)


def test_engine_logs_controls_and_event_transitions(anesthetized_engine):
    engine = anesthetized_engine
    engine.actions.begin_step("OBJECTIVE", engine.state.time)

    engine.give_fluid(500)
    engine.give_albumin(250)
    engine.give_blood(300)
    engine.give_drug_bolus("propofol", 150)
    engine.set_drug_rate("nore", 4.0)
    engine.set_drug_target("remi", 3.0)
    engine.start_hemorrhage(400.0)
    engine.start_hemorrhage(400.0)
    engine.stop_hemorrhage()
    engine.stop_hemorrhage()

    fluids = engine.actions.since_step(ACTION_FLUID)
    assert [(record.label, record.amount) for record in fluids] == [
        ("crystalloid", 500),
        ("colloid", 250),
        ("blood", 300),
    ]

    def amounts(action, label):
        return [record.amount for record in engine.actions.since_step(action, labels=(label,))]

    assert amounts(ACTION_DRUG_BOLUS, "propofol") == [150]
    assert amounts(ACTION_INFUSION_RATE, "nore") == [4.0]
    assert amounts(ACTION_TCI_TARGET, "remi") == [3.0]
    # Repeating a start or stop does not log another transition.
    assert amounts(ACTION_EVENT_START, "hemorrhage") == [400.0]
    assert len(amounts(ACTION_EVENT_STOP, "hemorrhage")) == 1
