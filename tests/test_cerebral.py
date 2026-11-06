import pytest

from anasim.core.clinical_actions import check_intubation
from anasim.physiology.cerebral import CerebralModel


@pytest.mark.parametrize(("map_mmhg", "saturation"), [(0, 98), (90, 10)])
def test_severe_supply_failure_suppresses_responsiveness_bis_and_breathing_until_it_returns(
        awake_engine, map_mmhg, saturation):
    brain = awake_engine.cerebral
    # Moderate hypotension and desaturation leave cerebral function intact.
    brain.step(600, 60, 13.5, 90)
    assert brain.dysfunction == 0
    brain.step(30, map_mmhg, 13.5, saturation)
    awake_engine.step(0.1)
    assert awake_engine.state.bis < 5 and awake_engine.state.loc > 0.95
    assert awake_engine.resp.state.effort_rr < 5
    assert check_intubation(awake_engine) is None
    brain.step(180, 90, 13.5, 98)
    awake_engine.step(0.1)
    assert awake_engine.state.loc < 0.01
    feedback = check_intubation(awake_engine)
    assert feedback is not None and feedback.blocked


def test_five_minutes_of_complete_failure_leave_coma_while_breathing_returns(awake_engine):
    brain = awake_engine.cerebral
    brain.step(CerebralModel.COMA_EXPOSURE_S, 0, 13.5, 98)
    brain.step(600, 90, 13.5, 98)
    awake_engine.step(0.1)
    assert awake_engine.state.bis < 5 and awake_engine.state.loc > 0.95
    assert awake_engine.resp.state.effort_rr > 5
