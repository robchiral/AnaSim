import pytest

from anasim.physiology.hemodynamics import HemodynamicModel


def test_awake_metabolic_acidosis_develops_respiratory_compensation(awake_engine):
    engine = awake_engine
    engine.resp.remove_buffer_base(250)
    for _ in range(1200):
        engine.step(0.5)
    state = engine.state
    assert state.hco3 < 12
    # Winter's rule; compensation does not normalize pH.
    assert state.pa_co2 == pytest.approx(1.5 * state.hco3 + 8, abs=3)
    assert 7.1 < state.ph < 7.3
    assert state.rr > 15


def test_severe_acidemia_reduces_output_and_adrenergic_rescue(patient):
    outputs: dict[float, tuple[float, ...]] = {}
    for ph in (7.4, 7.0):
        model = HemodynamicModel(patient)
        for _ in range(600):
            state = model.step(0.1, 0, 0, 0, pit=-2, paco2=40, pao2=95, ph=ph)
        outputs[ph] = state.co, state.map
        for _ in range(600):
            state = model.step(0.1, 0, 0, 25, pit=-2, paco2=40, pao2=95, ph=ph)
        outputs[ph] += (state.map - outputs[ph][1],)
    assert outputs[7.0][0] < 0.8 * outputs[7.4][0]
    assert outputs[7.0][2] < outputs[7.4][2]
